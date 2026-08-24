# InferFlux Configuration Guide

This directory contains configuration files for different deployment scenarios.

## Quick Start

Choose a config based on your hardware and model format:

| Config File | Use Case | Model Format | GPU |
|-------------|----------|--------------|-----|
| `server.yaml` | CPU-only inference | GGUF | No |
| `server.cuda.yaml` | CUDA GGUF models | GGUF | Yes |
| `server.cuda.safetensors.yaml` | CUDA safetensors models | Safetensors | Yes |
| `server.cuda.qwen14b.yaml` | Qwen 14B (optimized) | GGUF | Yes |
| `server.cuda.qwen32b.yaml` | Qwen 30B+ (optimized) | GGUF | Yes |
| `server.template.yaml` | Template for custom configs | Any | Any |

## Configuration Files

### server.yaml
**Default CPU-only configuration**

- Backend: CPU
- Flash Attention: Disabled
- Batch Size: 4
- KV Pages: 4096

```bash
./build/inferfluxd --config config/server.yaml
```

### server.cuda.yaml
**CUDA configuration for GGUF models (recommended)**

- Backend: `llama_cpp_cuda` (llama.cpp)
- Flash Attention: **Enabled** (Rule 2)
- Phase Overlap: **Enabled** (Rule 4)
- Batch Size: 32 (for 1-3B models)
- KV Pages: 4096

```bash
./build/inferfluxd --config config/server.cuda.yaml
```

### server.cuda.safetensors.yaml
**CUDA configuration for safetensors models**

- Backend: `inferflux_cuda`
- Flash Attention: **Enabled** (Rule 2)
- Phase Overlap: **Enabled** (Rule 4)
- Batch Size: 16 (for 3B models)
- KV Pages: 256

```bash
./build/inferfluxd --config config/server.cuda.safetensors.yaml
```

### server.cuda.qwen14b.yaml
**Optimized for Qwen 14B models**

- Batch Size: 16
- KV Pages: 2048

### server.cuda.qwen32b.yaml
**Optimized for Qwen 30B+ models**

- Batch Size: 16
- KV Pages: 2048

### server.template.yaml
**Template with all 8 advisor rules documented**

Copy and customize for your deployment.

## 8 Startup Advisor Rules

The startup advisor logs recommendations at startup when config is suboptimal.

| Rule | Trigger | Recommendation | Config Fix |
|------|---------|----------------|------------|
| 1. Backend mismatch | safetensors + CUDA + llama.cpp | Use InferFlux CUDA backend | `backend: inferflux_cuda` |
| 2. Attention kernel | GPU SM ≥ 8.0, FA disabled | Enable FA2 | `cuda.flash_attention.enabled: true` |
| 3. Batch size vs VRAM | Large VRAM, small batch | Increase batch | `runtime.scheduler.max_batch_size: <higher>` |
| 4. Phase overlap | CUDA, batch ≥ 4, disabled | Enable overlap | `cuda.phase_overlap.enabled: true` |
| 5. KV cache pages | Large VRAM, low pages | Increase pages | `runtime.paged_kv.cpu_pages: <higher>` |
| 6. Tensor parallelism | Multi-GPU, TP=1 | Use TP | `runtime.tensor_parallel: <gpu_count>` |
| 7. Unknown format | `format == "unknown"` | Set format | `models[*].format: gguf\|safetensors\|hf` |
| 8. GPU unused | GPU available, CPU backend | Enable CUDA | `models[*].backend: cuda` |

## Recommended Settings by Model Size

### 1-3B Models (e.g., TinyLlama, Qwen2.5-3B)

```yaml
runtime:
  scheduler:
    max_batch_size: 24-32
  paged_kv:
    cpu_pages: 256-512 (GPU) / 4096 (CPU)
```

### 7-8B Models (e.g., Llama3-8B, Qwen2.5-7B)

```yaml
runtime:
  scheduler:
    max_batch_size: 16-24
  paged_kv:
    cpu_pages: 512-1024 (GPU)
```

### 13-14B Models (e.g., Qwen2.5-14B)

```yaml
runtime:
  scheduler:
    max_batch_size: 12-16
  paged_kv:
    cpu_pages: 1536-2048 (GPU)
```

### 30B+ Models (e.g., Qwen3-30B)

```yaml
runtime:
  scheduler:
    max_batch_size: 8-16
  paged_kv:
    cpu_pages: 2048-3072 (GPU)
  tensor_parallel: 2  # For multi-GPU
```

### Optional Session Handle Layer (Stateless API stays default)

```yaml
runtime:
  scheduler:
    session_handles:
      enabled: true
      ttl_ms: 300000
      max_sessions: 1024
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `INFERFLUX_MODEL_PATH` | Override model path | `/models/llama3.gguf` |
| `INFERFLUX_MODELS` | Multi-model config | `id=model1,path=/path/to/model.gguf,...` |
| `INFERFLUX_CUDA_STRICT` | Strict InferFlux CUDA mode | `1` to fail if the InferFlux CUDA runtime reports fallback |
| `INFERFLUX_BACKEND_PRIORITY` | Backend selection | `cuda,cpu` or `cpu,cuda` |
| `INFERFLUX_SESSION_HANDLES_ENABLED` | Enable optional `session_id` mapping layer | `true` |
| `INFERFLUX_SESSION_TTL_MS` | Session handle TTL (ms) | `300000` |
| `INFERFLUX_SESSION_MAX` | Max tracked sessions | `1024` |
| `INFERFLUX_DISABLE_STARTUP_ADVISOR` | Disable advisor | `true` |

## Suppressing Advisor Output

If you want to disable the startup advisor:

```bash
INFERFLUX_DISABLE_STARTUP_ADVISOR=true ./build/inferfluxd --config config/server.yaml
```

## GPU Compute Capability Reference

| GPU | SM | FA2 Support | Recommended Config |
|-----|-------|------------|-------------------|
| RTX 4090, Ada | 8.9 | ✅ Yes | `server.cuda.yaml` |
| RTX 3090, Ampere | 8.6 | ✅ Yes | `server.cuda.yaml` |
| RTX 2080, Turing | 7.5 | ❌ No | Set `flash_attention.enabled: false` |
| GTX 1080, Pascal | 6.1 | ❌ No | Set `flash_attention.enabled: false` |

## Format-Specific Notes

### GGUF Format
- Uses llama.cpp backend (`llama_cpp_cuda` or `cpu`)
- Well-tested, production-ready
- Supports quantization (Q4_K_M, Q5_K_M, Q6_K, Q8_0)

### Safetensors Format
- Requires InferFlux CUDA backend (`inferflux_cuda`)
- Use `backend: inferflux_cuda`
- Supports BF16/FP16 weights
- Better for fine-tuned models from HuggingFace

### HuggingFace Format
- Use `format: hf` with `hf://org/repo` paths
- Automatically resolves to local cache
- Falls back to GGUF sidecar if needed

## Testing Your Configuration

Run the advisor test suite:

```bash
./scripts/smoke.sh backend-identity --help
```

Legacy advisor batch scripts were archived under `scripts/archive/advisor/`; use the canonical smoke and benchmark entry points instead.
