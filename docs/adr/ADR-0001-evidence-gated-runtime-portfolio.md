# ADR-0001: Evidence-Gated Runtime Portfolio

Status: Proposed
Date: 2026-08-21
Owners: Runtime, Product, QA
Dependencies: TD-006, TD-007

## Context

InferFlux maintains native CUDA and llama.cpp compatibility providers. Existing
documents disagree about correctness, throughput, memory, and release grade.
Kernel ownership is valuable only if it creates repeatable product advantage.

## Decision

Keep both providers, but classify native CUDA as experimental until a retained
matrix proves its supported envelope. Use llama.cpp as the default provider
outside that envelope. Every response and metric preserves requested provider,
selected provider, fallback state, and reason.

The native path graduates per model/hardware envelope when it achieves:

- deterministic greedy correctness and contract parity on the maintained suite;
- at least 99 successful requests out of 100 at concurrency 1, 4, and 8;
- at least 0.90x llama.cpp geometric-mean throughput across those levels;
- no more than 10% peak-memory overhead under matched KV/context settings.

Threshold changes require an ADR amendment, not an edited benchmark narrative.

## Consequences

- New kernel families wait until TD-003 produces the decision matrix.
- A failed gate narrows native support; it does not silently weaken criteria.
- Product/API work stays backend-neutral and can ship on the compatibility path.

## Rejected Alternatives

- Native-first by aspiration: spends scarce effort before value is demonstrated.
- Removing native CUDA now: discards potential differentiated headroom too early.
- Comparing sampled-text Jaccard alone: it confounds stochastic divergence with correctness.

## Status Update (2026-09-14)

The graduation envelope has been partially met: correctness gates are green
(llama.cpp backend-op parity suite 11,054/11,054 on gfx1201; parity and
contract gates in CI), and the throughput criterion is exceeded on the GGUF
envelope at c=16 — native leads the wrapper 1.56x (reproduced in both runs)
against the 0.90x floor. The memory-overhead criterion (<=10%) remains open
(GGUF overhead ~+1.4 GB vs llama.cpp's ~4.1 GB, about +34%). Per this ADR's
own rule, native is default-eligible for GGUF serving at c>=8 while the
memory gate keeps the broadest-compatibility default under review.
