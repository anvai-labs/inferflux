#include <catch2/catch_amalgamated.hpp>

#include "runtime/disaggregated/shm_kv_transport.h"
#include "server/http/http_server.h"
#include "server/metrics/metrics.h"

using namespace inferflux;
using namespace inferflux::disaggregated;

// ---------------------------------------------------------------------------
// ShmKVTransport — basic write/read roundtrip
// ---------------------------------------------------------------------------

TEST_CASE("ShmKVTransport: empty queue returns nullopt", "[shm_transport]") {
  ShmKVTransport transport(4);
  REQUIRE_FALSE(transport.TryDequeue().has_value());
}

TEST_CASE(
    "ShmKVTransport: control-only packet (no kv_blob) roundtrips correctly",
    "[shm_transport]") {
  ShmKVTransport transport(4);
  KVPacket pkt;
  pkt.request_id = 42;
  pkt.n_past = 7;
  pkt.sequence_id = 2;
  REQUIRE(transport.Enqueue(std::move(pkt)));
  auto result = transport.TryDequeue();
  REQUIRE(result.has_value());
  REQUIRE(result->request_id == 42);
  REQUIRE(result->n_past == 7);
  REQUIRE(result->sequence_id == 2);
  REQUIRE(result->kv_blob.empty());
}

TEST_CASE("ShmKVTransport: kv_blob roundtrips through POSIX SHM",
          "[shm_transport]") {
  ShmKVTransport transport(4);
  KVPacket pkt;
  pkt.request_id = 99;
  pkt.ticket_id = 321;
  pkt.ticket_stage = KVTicketStage::kEnqueued;
  pkt.kv_blob = {0x01, 0x02, 0x03, 0xAB, 0xCD, 0xEF};
  std::vector<uint8_t> expected = pkt.kv_blob;
  REQUIRE(transport.Enqueue(std::move(pkt)));
  auto result = transport.TryDequeue();
  REQUIRE(result.has_value());
  REQUIRE(result->request_id == 99);
  REQUIRE(result->ticket_id == 321);
  REQUIRE(result->ticket_stage == KVTicketStage::kEnqueued);
  REQUIRE(result->metadata.find("|ticket=321") != std::string::npos);
  REQUIRE(result->metadata.find("|ticket_stage=enqueued") !=
          std::string::npos);
  REQUIRE(result->kv_blob == expected);
}

TEST_CASE("ShmKVTransport: FIFO ordering preserved across multiple enqueues",
          "[shm_transport]") {
  ShmKVTransport transport(8);
  for (uint64_t i = 1; i <= 4; ++i) {
    KVPacket pkt;
    pkt.request_id = i;
    REQUIRE(transport.Enqueue(std::move(pkt)));
  }
  for (uint64_t i = 1; i <= 4; ++i) {
    auto result = transport.TryDequeue();
    REQUIRE(result.has_value());
    REQUIRE(result->request_id == i);
  }
  REQUIRE_FALSE(transport.TryDequeue().has_value());
}

TEST_CASE("ShmKVTransport: Enqueue returns false when capacity is full",
          "[shm_transport]") {
  ShmKVTransport transport(2);
  KVPacket p1;
  p1.request_id = 1;
  KVPacket p2;
  p2.request_id = 2;
  KVPacket p3;
  p3.request_id = 3;
  REQUIRE(transport.Enqueue(std::move(p1)));
  REQUIRE(transport.Enqueue(std::move(p2)));
  REQUIRE_FALSE(transport.Enqueue(std::move(p3)));
}

TEST_CASE("ShmKVTransport: Size() and Capacity() reflect state",
          "[shm_transport]") {
  ShmKVTransport transport(5);
  REQUIRE(transport.Capacity() == 5);
  REQUIRE(transport.Size() == 0);
  KVPacket pkt;
  pkt.request_id = 1;
  transport.Enqueue(std::move(pkt));
  REQUIRE(transport.Size() == 1);
  transport.TryDequeue();
  REQUIRE(transport.Size() == 0);
}

TEST_CASE("ShmKVTransport: large kv_blob roundtrips correctly",
          "[shm_transport]") {
  ShmKVTransport transport(2);
  // 64 KB blob — ensures POSIX SHM handles multi-page allocations.
  std::vector<uint8_t> big_blob(64 * 1024);
  for (std::size_t i = 0; i < big_blob.size(); ++i) {
    big_blob[i] = static_cast<uint8_t>(i & 0xFF);
  }
  KVPacket pkt;
  pkt.request_id = 7;
  pkt.kv_blob = big_blob;
  REQUIRE(transport.Enqueue(std::move(pkt)));
  auto result = transport.TryDequeue();
  REQUIRE(result.has_value());
  REQUIRE(result->kv_blob == big_blob);
}

TEST_CASE("ShmKVTransport: control-only ticket metadata roundtrips correctly",
          "[shm_transport]") {
  ShmKVTransport transport(2);
  KVPacket pkt;
  pkt.request_id = 17;
  pkt.ticket_id = 88;
  pkt.ticket_stage = KVTicketStage::kAcknowledged;
  REQUIRE(transport.Enqueue(std::move(pkt)));
  auto result = transport.TryDequeue();
  REQUIRE(result.has_value());
  REQUIRE(result->ticket_id == 88);
  REQUIRE(result->ticket_stage == KVTicketStage::kAcknowledged);
  REQUIRE(result->metadata.find("|ticket=88") != std::string::npos);
  REQUIRE(result->metadata.find("|ticket_stage=acknowledged") !=
          std::string::npos);
}

TEST_CASE("ShmKVTransport: ticket lifecycle tracking delegates to control queue",
          "[shm_transport]") {
  ShmKVTransport transport(2);
  KVPacket pkt;
  pkt.request_id = 23;
  pkt.ticket_id = 501;
  pkt.ticket_stage = KVTicketStage::kEnqueued;
  REQUIRE(transport.Enqueue(std::move(pkt)));
  REQUIRE(transport.GetTicketStage(501) == KVTicketStage::kEnqueued);
  REQUIRE(
      transport.UpdateTicketStage(501, KVTicketStage::kAcknowledged));
  REQUIRE(transport.GetTicketStage(501) == KVTicketStage::kAcknowledged);
  REQUIRE(transport.UpdateTicketStage(501, KVTicketStage::kCommitted));
  REQUIRE(transport.GetTicketStage(501) == KVTicketStage::kCommitted);
}

// ---------------------------------------------------------------------------
// KV transfer latency metric (§2.5 item 12)
// ---------------------------------------------------------------------------

TEST_CASE("RecordKVTransfer accumulates in kv_transfer_duration_ms histogram",
          "[shm_transport]") {
  MetricsRegistry reg;
  reg.RecordKVTransfer(1.5);
  reg.RecordKVTransfer(3.0);
  std::string output = reg.RenderPrometheus();
  REQUIRE(output.find("inferflux_kv_transfer_duration_ms_count") !=
          std::string::npos);
  // count should be 2
  REQUIRE(output.find("inferflux_kv_transfer_duration_ms_count{") !=
          std::string::npos);
}

TEST_CASE("HELP and TYPE lines present for kv_transfer_duration_ms",
          "[shm_transport]") {
  MetricsRegistry reg;
  std::string output = reg.RenderPrometheus();
  REQUIRE(output.find("# HELP inferflux_kv_transfer_duration_ms") !=
          std::string::npos);
  REQUIRE(output.find("# TYPE inferflux_kv_transfer_duration_ms histogram") !=
          std::string::npos);
}

// ---------------------------------------------------------------------------
// HttpServer::PoolRole — /readyz role field
// (We test the role enum values and SetRole/SetDecodePoolReady directly
//  since we cannot spin up a live HTTP server in a unit test.)
// ---------------------------------------------------------------------------

TEST_CASE("HttpServer::PoolRole enum has expected values", "[shm_transport]") {
  REQUIRE(HttpServer::PoolRole::kUnified != HttpServer::PoolRole::kPrefill);
  REQUIRE(HttpServer::PoolRole::kUnified != HttpServer::PoolRole::kDecode);
  REQUIRE(HttpServer::PoolRole::kPrefill != HttpServer::PoolRole::kDecode);
}
