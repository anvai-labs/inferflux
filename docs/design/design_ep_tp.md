# Model Parallelism (TP/PP/EP) Design Brief

> Living design artifact for §2.3 / §2.6 roadmap items. Tracks the granular tasks needed to move **Model parallelism (TP/PP/EP)** from grade **D** toward **B/A**.

## Goals

1. **Expert Parallelism (EP)** — Route requests across expert shards (multi-GPU or multi-process) with load-aware dispatch, per-expert metrics, and graceful failure.
2. **Tensor Parallelism (TP)** — Slice heavyweight matrix multiplies across multiple GPUs using NCCL collectives; initial target: 2-way TP.
3. **Pipeline Parallelism (PP)** — (Optional stretch) Enable multi-stage execution for extremely large models once TP/EP foundations exist.
4. **Operational Resilience** — Detect and recover from shard/GPU/NCCL failures without dropping the service.
5. **Operator Ergonomics** — Provide clear documentation, Helm/CLI knobs, and metrics required to operate multi-GPU clusters.

## Current State (Q1 2026)

- MoE detection helpers (`IsMoE`, `ExpertCount`, `ActiveExperts`) and metrics (`inferflux_moe_requests_total`) are present.
- `EPDispatch` / `LocalEPDispatch` are stubs; scheduler has no notion of expert placement.
- CUDA build path exists but is CPU/MPS-only in production; no NCCL initialization or TP kernels.
- No design artifact covers target topologies or failure-handling expectations.

## High-Level Architecture

1. **Process Topology**
   - Single control plane (scheduler + HTTP server).
   - One or more *shards* per model. Each shard may own multiple GPUs.
   - Expert parallel: dispatch requests to shards hosting relevant experts; shards resolve to local experts and run decode.
   - Tensor parallel: within a shard, split compute across GPUs using NCCL collectives (all-reduce/all-gather) per layer.
   - Pipeline parallel (optional): stage-wise processing with activation buffers streamed between stages.

2. **Communication Paths**
   - Scheduler ↔ EP dispatcher: asynchronous RPC (gRPC/ZeroMQ/custom) carrying request metadata, expert routing instructions, and cancellation signals.
   - Shard ↔ Shard (TP/PP): NCCL for intra-shard collectives; potential RDMA path for cross-host activations.
   - Monitoring: Prometheus metrics per shard + aggregated view in control plane.

## Task Breakdown

### 1. Design & Documentation

- [ ] Create diagrams illustrating EP, TP, and (optional) PP data flows.
- [ ] Define configuration surface (env vars, YAML, Helm values) for:
  - Number of shards, GPUs per shard.
  - TP degree, PP stages.
  - Expert routing policies (round-robin, load-based).
- [ ] Specify acceptance criteria (target throughput, latency, failover behavior) and include them in this doc.

### 2. Expert-Parallel Execution

- [ ] Replace `EPDispatch` / `LocalEPDispatch` stubs with concrete interfaces (request proto, streaming channel, retry semantics).
- [ ] Extend `Scheduler::ResolveBackends` and `PendingRequest` to include:
  - Required experts / tokens per request.
  - Load-balancing metadata (weights, queue depths).
- [ ] Implement per-expert queues + Prometheus metrics (`inferflux_moe_expert_load_total`, `inferflux_moe_expert_latency_ms`).
- [ ] Add unit tests for dispatcher routing + fairness interactions.

### 3. Tensor / Pipeline Parallel

- [ ] Enable CUDA backend path to initialize NCCL communicators; parse TP/PP knobs in `server/main.cpp`.
- [ ] Implement 2-way TP inside `LlamaCUDABackend`:
  - Weight partitioning for key GEMM layers.
  - NCCL all-reduce/all-gather wrappers with error handling.
- [ ] (Stretch) Outline PP micro-batching: stage definitions, activation buffers, scheduler updates.
- [ ] Add CUDA-only compile+smoke test (CI compile-check + optional GPU run on self-hosted runner).

### 4. Runtime Resilience

- [ ] Implement heartbeat/health tracking for shards (NCCL timeouts, GPU watchdog).
- [ ] Define failover policy (e.g., remove shard from router, drain in-flight requests, retry).
- [ ] Add chaos/integration test that simulates GPU/NCCL failure.

### 5. Documentation & Operator Rollout

- [ ] Update `docs/Architecture.md` with the finalized model-parallel section (include diagrams).
- [ ] Write an operator guide (`docs/operators/tp_ep_setup.md`) covering deployment, monitoring, and failure recovery.
- [ ] Extend Helm charts (`deploy/helm/*`) with TP/EP knobs and examples.
- [ ] Document new metrics and alerts (Prometheus/Grafana).

## Open Questions

1. **Process model** — Single process managing all GPUs vs per-shard processes communicating over RPC?
2. **Dependency scope** — Use llama.cpp’s built-in TP support vs bespoke kernels?
3. **State management** — How do we checkpoint or reload experts independently?
4. **Backpressure** — How does EP dispatch coordinate with fairness controller to avoid starving non-MoE requests?
5. **Testing strategy** — Coverage for TP/EP paths on CI vs self-hosted GPU hardware?

These questions should be resolved before implementation phases begin; decisions and rationale should be captured in this document.

---

**Maintenance:** Update this file as tasks land (checklists, diagrams, links to PRs). Cross-link from the TechDebt scorecard so stakeholders can track progress.
