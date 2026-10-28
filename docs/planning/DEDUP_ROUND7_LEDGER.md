# Dedup Round 7 Ledger — CUDA + C++ surface (jscpd 150 → 100 → 50)

**Status: Active (opened 2026-10-28).** Rounds 1–6 covered host C++ with the
"byte-equivalence argued per site" bar (PRs #256–#263) and deliberately
skipped the native GPU lane. Round 7 targets that lane plus the residual
host tail, now that the PR-level CUDA compile check and the trusted
gpu-gates lane exist. Baseline config committed as `.jscpd.json`
(`minTokens` ratchets 150 → 100 → 50 at phase boundaries).

## Measured baseline (2026-10-28, jscpd 5.4.2, formats cpp,c,cpp-header,c-header + cu/cuh via formats-exts, dirs runtime/scheduler/server/io/net/policy/model/cli)

| Tokens | Clones | Dup lines | Dominant zones |
|---|---|---|---|
| 150 | 75 | 2,246 | CUDA kernels ≈96%: mmvq.cuh 968L×29, mmq.cuh 460L×10, fused_dequant_gemv.cuh 213+82+31, fused_quant_gemm.cu 128L×6, mmq_mma.cuh 80L×4, fused_rope_kv_append.cuh 70L; host stragglers ≤25L each (main.cpp, cublas_gemm.cpp, native_gpu_backend.cpp, cuda_kernels.cu, tokenizers, llama_cpp_backend.cpp) |
| 100 | 181 | 3,895 | adds transformer_forward.cu 136L×10, flash_attention.cu 99L×7, executor 51L×4, hybrid_kv_cache_gpu↔kv_cache_gpu 43L×3 |
| 50 | 667 | 8,226 | adds host tail: main.cpp 183L×15, inferflux_cuda_executor.cpp 163L×14, cuda_kernels.cu 143L×14, http_server.cpp 132L×14, native_gpu_backend.cpp 119L×7 |

## Evidence bars

- **Class A (host / straight-line)**: byte-equivalence argued per site; CPU
  suite + ctest green; PR CUDA compile job green; codecov patch ≥ 50%.
- **Class B (device codegen-affecting)**: Class A + SASS diff
  (`cuobjdump --dump-sass`, before/after, representative instantiations) +
  benchmark A/B ≤ 2% (median of 3) on the matching rig + adversarial review
  before push.
- Rigs run on the RTX 4000 Ada host. No PR may add/remove CUDA test
  executables (gpu-gates.yml build-list drift trap); new parity cases fold
  into existing test targets.

## Golden rig baselines (RTX 4000 Ada; median of 3) — recorded at PR-1

Full logs retained on the bench host (`~/dedup7-baselines/<rig>.run{1,2,3}.txt`,
worktree `~/code/inferflux-worktrees/dedup7` at `b37686bd3`, CUDA-only build,
nvcc 13.2). Representative median-run headline rows:

| Rig | Headline (median run) | PR |
|---|---|---|
| benchmark_q4k_mmq_rig | M=16 MMA splits=8: 57.66 us / 227 GB/s (maxrel-vs-policy 0.0; MMA-vs-dp4a maxrel 9.7e-04) | baseline |
| benchmark_q6k_kernels | M=1 v1\<1\>(N,M): 41.5 us vs 51 us floor (N=2048 K=11008) | baseline |
| benchmark_fa_decode_rig | per-kvlen cold/warm_us table + max_rel_err columns (see run logs) | baseline |
| benchmark_gate_up_kernels | N=11008 K=2048, floor 70.5 us (both mats), ~360 GB/s device | baseline |

A/B comparisons for Class-B PRs rebuild this exact worktree state (same
configure flags, same target set) at the PR commit and diff the same rows;
gate is ≤2% on the matching headline metrics, median of 3.

## Per-PR ledger

| PR | Phase | Files | Class | jscpd delta | Rig delta | SASS | Review |
|---|---|---|---|---|---|---|---|
| #284 | pre-round (bug fixes) | router/http/metrics | — | n/a | n/a | n/a | standard |
| (PR-1) | infrastructure | .jscpd.json, this ledger | — | config only | baselines recorded | — | — |

## Do-NOT-dedup registry (deliberate duplication, preserved)

1. `mmq.cuh` ↔ `mmq_grouped_ffn.cuh` structural mirroring — documented
   design intent in the header comment.
2. `mmq_mma.cuh` llama.cpp-port layout pins ("match llama exactly",
   "calibrated, keep as-is").
3. `flash_attention.cu` device-kernel variant instantiations (HEAD_DIM ×
   GQA ratio × dtype), perf-pinned by benchmark_fa_decode_rig — host GQA
   ladders only are deduped.
4. `mmvq.cuh` tuning generations (narrow / vec / wide-uint4 / fused
   gate-up-silu) — never merged across generations; each carries
   measured-perf rationale.
5. `http_server.cpp` route if-chain lines — frozen by
   scripts/check_docs_contract.py endpoint extraction; dedup stays inside
   handler braces.
6. `tests/unit/test_fused_kernels.{cpp,cu}` split — CUDA 13.2
   nvcc/Catch2 conflict workaround.
7. Sites rounds 1–6 deliberately left (see those PR bodies).
8. `model/gguf_tokenizer.h` ↔ `model/hf_tokenizer.h` sibling-wrapper
   interface declarations — same override surface over the same
   `MlxTokenizer` inner, but restructuring public wrapper classes for
   declaration-level duplication is not worth the header-API churn (added
   at round 7 triage).

## Benchmark recording template (per Class-B PR)

```
rig: <name> | runs: r1/r2/r3 | median-before: X | median-after: Y | delta: Z%
sass: <kernel instantiation> | before-sha256: … | after-sha256: … | identical: yes/no
parity: <test case + tolerance from tests/unit/…>
```
