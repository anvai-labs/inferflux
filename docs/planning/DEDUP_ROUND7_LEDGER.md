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
| #285 | infrastructure | .jscpd.json, this ledger | — | config only | baselines recorded | — | — |
| #286 | t150 host stragglers | cublas/llama/native backends | A | 75→72 clones, 2,246→2,182 L | n/a (host-only) | n/a | standard |
| #287 | gate hardening | ci.yml, hooks, instructions | — | n/a | n/a | n/a | standard |
| (PR-3) | t150 dispatch tables + tile skeleton | fused_quant_gemm.cu | A + B-lite | 72→71, 2,182→2,156 L | q4k rig dp4a-mmq medians −1.05%/−1.30% | differs ~50%/34% (nvcc rescheduled through inlined helpers; parity green) | adversarial (this table + PR body) |
| #289 | t150 mmvq epilogues + ladder | kernels/mmvq.cuh | A (device) | 71→63, 2,156→1,899 L | SASS byte-identical (3 instantiations); rig deltas within the session's contention floor (untouched mma rows drifted +3.5%) | identical (0/516, 0/644, 0/708 lines) | adversarial (PR body) |
| (batch) | t150/t100 CUDA batch: mmvq epilogue-policy bodies (4 families, 9→4 entry kernels + device-inline bodies), FA GQA host ladders (DispatchGqaRatio, fn-pointer form — nvcc parses fn-template names as expressions in template-template position), tf norm-checked helpers | mmvq.cuh, flash_attention.cu, transformer_forward.cu + llama_forward.h | A + B | 63→55 clones, 1,899→1,476 L | SASS byte-identical (3 repr); q6k min-of-3: 7/42 rows +2.0–4.2% incl. untouched llama.cpp reference row +3.8% (noise floor); fa_decode 0 rows over | identical | adversarial (PR body; batched per CI-economy instruction) |

Round-7 findings feeding later phases:
- SASS identity is NOT guaranteed by `__forceinline__` factoring alone — nvcc
  reschedules around the helper boundary (see PR-3). The Phase-4/5 mmvq gates
  must treat "byte-identical SASS" as the exception, with rig parity +
  correctness parity as the hard bars.
- The two down-proj tile decode cores (q4_k scale/min unpack vs q6_k ql/qh
  reconstruction) are structurally different — a traits merge rides the
  Phase-5 evidence gate rather than this phase, per the tuning-generation
  registry rule.

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
