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
  // Register here rather than relying on the registration side effect of an
  // earlier case ("can register and create backends"): under randomized test
  // order that case may not have run yet. Re-registering the same factory is
  // idempotent, so declaration order stays green too.
  reg.Register(LlamaBackendTarget::kOpenCL, BackendProvider::kLlamaCpp,
               [] { return std::make_shared<OpenClBackend>(); });
  auto targets = reg.AvailableTargets();
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

TEST_CASE(
    "GPU initialization preserves omitted placement and explicit selectors",
    "[backend_registry][device_placement]") {
  class Strategy : public GpuDeviceStrategy {
  public:
    explicit Strategy(LlamaBackendTarget target) : target_(target) {}
    bool Initialize() override {
      ++default_calls;
      return true;
    }
    bool Initialize(int ordinal) override {
      ++ordinal_calls;
      ordinal_ = ordinal;
      return true;
    }
    bool IsAvailable() const override { return true; }
    GpuDeviceInfo GetDeviceInfo() const override {
      GpuDeviceInfo info;
      info.device_id = ordinal_;
      return info;
    }
    LlamaBackendTarget Target() const override { return target_; }
    void RecordMetrics(const LlamaBackendConfig &) override {}
    LlamaBackendTarget target_;
    int ordinal_{0};
    int default_calls{0};
    int ordinal_calls{0};
  };
  // Native CUDA specializes the no-argument initialization in a derived
  // strategy while inheriting the vendor strategy's ordinal overload.
  class SpecializedStrategy final : public Strategy {
  public:
    using Strategy::Strategy;
    bool Initialize() override {
      ++specialized_calls;
      return Strategy::Initialize();
    }
    int specialized_calls{0};
  };
  const auto target =
      GENERATE(LlamaBackendTarget::kCuda, LlamaBackendTarget::kRocm);
  const bool explicit_device = GENERATE(false, true);
  auto strategy = std::make_unique<SpecializedStrategy>(target);
  auto *observed = strategy.get();
  GpuAcceleratedBackend backend(std::move(strategy));
  LlamaBackendConfig config;
  config.gpu_layers = 8;
  if (explicit_device)
    config.device = target == LlamaBackendTarget::kCuda ? "cuda:1" : "rocm:1";
  LlamaBackendConfig tuned;
  REQUIRE(backend.InitializeDevice(config, &tuned));
  REQUIRE(tuned.device == config.device);
  REQUIRE(tuned.gpu_layers == 8);
  REQUIRE(backend.DeviceInfo().device_id == (explicit_device ? 1 : 0));
  REQUIRE(observed->default_calls == (explicit_device ? 0 : 1));
  REQUIRE(observed->specialized_calls == (explicit_device ? 0 : 1));
  REQUIRE(observed->ordinal_calls == (explicit_device ? 1 : 0));
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
