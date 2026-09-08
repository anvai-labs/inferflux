# Scripts

Supported entry points:

- `scripts/benchmark.sh`
  - `gguf-compare`
  - `multi-backend`
  - `throughput-gate`
  - `multi-backend` compares:
    - `inferflux_cuda`
    - `llama_cpp_cuda`
    - `ollama`
    - `lmstudio`
    - `vllm`
    - `sglang`
- `scripts/profile.sh`
  - `backend`
  - `backend-ncu`
  - `phase-timing`
  - `analyze-nsys`
- `scripts/smoke.sh`
  - `gguf-native`
  - `backend-identity`
- direct helpers that remain first-class because other tooling imports or calls them:
  - `build.sh`
  - `run_dev.sh`
  - `run_gguf_comparison_benchmark.sh`
  - `benchmark_multi_backend_comparison.sh`
  - `run_throughput_gate.py`
  - `profile_backend.sh`
  - `profile_backend_ncu.sh`
  - `parse_native_phase_timing.py`
  - `analyze_nsys_results.py`
  - `extract_native_dispatch_winners.py`
  - `classify_benchmark_response.py`
  - `check_backend_identity.py`
  - `check_docs_contract.py`
  - `generate_sbom.py`
  - `test_gguf_native_smoke.py`
  - `compare_decode_traces.py`

Archive policy:

- One-off probes, temporary experiments, and superseded wrappers live under `scripts/archive/`.
- New script additions should prefer extending the supported entry points above over creating new top-level files.

Multi-backend harness behavior:

- `benchmark_multi_backend_comparison.sh` runs each backend in an isolated child process.
- This is deliberate. It prevents local CUDA backends from sharing allocator / stream / process state during the same benchmark session.
- Use `INFERFLUX_BENCH_SINGLE_BACKEND=<backend_id>` to run one backend through the same harness path while keeping the same artifact layout.

External engine notes:

- `vllm` and `sglang` are treated as OpenAI-compatible HTTP backends in the multi-backend benchmark.
- They can either be pre-started externally or auto-launched locally by the harness.
- Set `AUTOSTART_VLLM=true` and/or `AUTOSTART_SGLANG=true` to have the benchmark launch and tear them down one at a time.
- Stock `llama-server` comparison: build it out-of-tree from the pinned
  submodule with a conda-free environment (anaconda's sysroot libm breaks
  the link): `env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME=$HOME
  cmake -S external/llama.cpp -B /tmp/llama-server-build -DGGML_CUDA=ON
  -DCMAKE_BUILD_TYPE=Release -DMATH_LIBRARY=/usr/lib/x86_64-linux-gnu/libm.so
  -DMATHVEC_LIBRARY=/usr/lib/x86_64-linux-gnu/libmvec.so.1 &&
  cmake --build /tmp/llama-server-build --target llama-server`. Suggested
  flags to mirror the tuned wrapper: `-ngl 99 -c 4096 -np 16 -fa on`.
- When the ROCm toolchain is installed system-wide (e.g. `/usr/bin/hipcc`), SGLang's kernel JIT misdetects HIP and fails to build. Export `TVM_FFI_GPU_BACKEND=cuda` for the benchmark (or globally) to force the CUDA toolchain.
- Use `VLLM_MODEL` / `SGLANG_MODEL` to override auto-discovery from `/v1/models`.
- Use `VLLM_MODEL_PATH` / `SGLANG_MODEL_PATH` to point local autostart at a safetensors model directory.
- Use `SGLANG_PYTHON` if you need to override the default `./.venv-sglang/bin/python -m sglang.launch_server` entrypoint.
- The harness auto-detects the supplied model format and only runs compatible backends.
- In practice:
  - GGUF runs include `inferflux_cuda`, `llama_cpp_cuda`, `ollama`, and other GGUF-capable engines.
  - Safetensors runs include `inferflux_cuda`, `vllm`, `sglang`, and other safetensors-capable engines.
