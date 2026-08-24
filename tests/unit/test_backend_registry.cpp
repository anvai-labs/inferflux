#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/backend_factory.h"
#include "runtime/backends/backend_registry.h"
#include "runtime/backends/gpu/cpu_device_strategy.h"
#include "runtime/backends/gpu/gpu_accelerated_backend.h"
#include "runtime/backends/gpu/gpu_device_strategy.h"
#include "runtime/backends/gpu/opencl_device_strategy.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "runtime/backends/mps/mps_backend.h"
#include "runtime/backends/opencl/opencl_backend.h"
#include "runtime/backends/vulkan/vulkan_backend.h"

using namespace inferflux;

TEST_CASE("BackendRegistry singleton returns same instance",
          "[backend_registry]") {
  auto &a = BackendRegistry::Instance();
  auto &b = BackendRegistry::Instance();
  REQUIRE(&a == &b);
}

TEST_CASE("BackendRegistry can register and create backends",
          "[backend_registry]") {
  auto &reg = BackendRegistry::Instance();

  // Register a test backend for OpenCL (which is normally not registered)
  reg.Register(LlamaBackendTarget::kOpenCL, BackendProvider::kLlamaCpp,
               [] { return std::make_shared<OpenClBackend>(); });

  REQUIRE(reg.Has(LlamaBackendTarget::kOpenCL, BackendProvider::kLlamaCpp));

  auto backend =
      reg.Create(LlamaBackendTarget::kOpenCL, BackendProvider::kLlamaCpp);
  REQUIRE(backend != nullptr);
  REQUIRE(dynamic_cast<OpenClBackend *>(backend.get()) != nullptr);
}

TEST_CASE("BackendRegistry returns nullptr for unregistered backend",
          "[backend_registry]") {
  auto &reg = BackendRegistry::Instance();
  auto backend =
      reg.Create(LlamaBackendTarget::kOpenCL, BackendProvider::kNative);
  REQUIRE(backend == nullptr);
}

TEST_CASE("BackendRegistry AvailableTargets includes registered targets",
          "[backend_registry]") {
  auto &reg = BackendRegistry::Instance();
  auto targets = reg.AvailableTargets();
  // CPU target should always be available (registered below or by other tests)
  // At minimum, OpenCL was registered in the test above
  bool has_opencl = false;
  for (auto t : targets) {
    if (t == LlamaBackendTarget::kOpenCL) {
      has_opencl = true;
    }
  }
  REQUIRE(has_opencl);
}

TEST_CASE("GpuDeviceStrategy CpuDeviceStrategy is always available",
          "[gpu_device_strategy]") {
  CpuDeviceStrategy strategy;
  REQUIRE(strategy.IsAvailable());
  REQUIRE(strategy.Initialize());
  REQUIRE(strategy.Target() == LlamaBackendTarget::kCpu);

  auto info = strategy.GetDeviceInfo();
  REQUIRE(info.device_name == "CPU");
  REQUIRE_FALSE(info.supports_flash_attention);
  REQUIRE(info.flash_attention_version == "none");
}

TEST_CASE("GpuDeviceStrategy OpenClDeviceStrategy is not available",
          "[gpu_device_strategy]") {
  OpenClDeviceStrategy strategy;
  REQUIRE_FALSE(strategy.IsAvailable());
  REQUIRE_FALSE(strategy.Initialize());
  REQUIRE(strategy.Target() == LlamaBackendTarget::kOpenCL);
}

TEST_CASE("GpuDeviceInfo default values are sensible",
          "[gpu_device_strategy]") {
  GpuDeviceInfo info;
  REQUIRE(info.device_name.empty());
  REQUIRE(info.arch.empty());
  REQUIRE(info.total_memory_mb == 0);
  REQUIRE(info.device_id == 0);
  REQUIRE_FALSE(info.supports_flash_attention);
  REQUIRE(info.flash_attention_version.empty());
}

TEST_CASE("LlamaBackendTarget kOpenCL parses and describes correctly",
          "[backend_registry]") {
  REQUIRE(ParseLlamaBackendTarget("opencl") == LlamaBackendTarget::kOpenCL);

  auto traits = DescribeLlamaBackendTarget(LlamaBackendTarget::kOpenCL);
  REQUIRE(traits.label == "opencl");
  REQUIRE(traits.gpu_accelerated);
  REQUIRE_FALSE(traits.supports_flash_attention);
}

TEST_CASE("VulkanBackend inherits GpuAcceleratedBackend",
          "[backend_registry]") {
  auto backend = std::make_shared<VulkanBackend>();
  REQUIRE(dynamic_cast<GpuAcceleratedBackend *>(backend.get()) != nullptr);
  REQUIRE(dynamic_cast<LlamaCppBackend *>(backend.get()) != nullptr);
}

TEST_CASE("MpsBackend inherits GpuAcceleratedBackend", "[backend_registry]") {
  auto backend = std::make_shared<MpsBackend>();
  REQUIRE(dynamic_cast<GpuAcceleratedBackend *>(backend.get()) != nullptr);
  REQUIRE(dynamic_cast<LlamaCppBackend *>(backend.get()) != nullptr);
}

TEST_CASE("CudaConfigExtension has correct defaults",
          "[backend_config_extensions]") {
  CudaConfigExtension ext;
  REQUIRE(ext.attention_kernel == "auto");
  REQUIRE_FALSE(ext.phase_overlap_scaffold);
  REQUIRE(ext.phase_overlap_min_prefill_tokens == 256);
  REQUIRE_FALSE(ext.phase_overlap_prefill_replica);
  REQUIRE(ext.kv_cache_dtype == "auto");
  REQUIRE(ext.dequantized_cache_policy == "none");
  REQUIRE_FALSE(ext.require_fused_quantized_matmul);
}

TEST_CASE("TuneLlamaBackendConfig populates cuda_ext for CUDA target",
          "[backend_config_extensions]") {
  LlamaBackendConfig cfg;
  cfg.cuda_attention_kernel = "fa3";
  cfg.cuda_phase_overlap_scaffold = true;
  cfg.cuda_phase_overlap_min_prefill_tokens = 512;
  cfg.cuda_phase_overlap_prefill_replica = true;
  cfg.inferflux_cuda_kv_cache_dtype = "fp16";
  cfg.inferflux_cuda_dequantized_cache_policy = "batch";
  cfg.inferflux_cuda_require_fused_quantized_matmul = true;

  auto tuned = TuneLlamaBackendConfig(LlamaBackendTarget::kCuda, cfg);

  REQUIRE(tuned.cuda_ext.attention_kernel == "fa3");
  REQUIRE(tuned.cuda_ext.phase_overlap_scaffold);
  REQUIRE(tuned.cuda_ext.phase_overlap_min_prefill_tokens == 512);
  REQUIRE(tuned.cuda_ext.phase_overlap_prefill_replica);
  REQUIRE(tuned.cuda_ext.kv_cache_dtype == "fp16");
  REQUIRE(tuned.cuda_ext.dequantized_cache_policy == "batch");
  REQUIRE(tuned.cuda_ext.require_fused_quantized_matmul);
}

TEST_CASE("CanonicalBackendId handles kOpenCL", "[backend_registry]") {
  REQUIRE(BackendFactory::CanonicalBackendId(BackendProvider::kLlamaCpp,
                                             LlamaBackendTarget::kOpenCL) ==
          "llama_cpp_opencl");
  REQUIRE(BackendFactory::CanonicalBackendId(BackendProvider::kNative,
                                             LlamaBackendTarget::kOpenCL) ==
          "inferflux_opencl");
}
