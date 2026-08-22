# TD-003: Eliminate Runtime Evidence and Product-Claim Drift

Status: Proposed
Priority: P0
Owners: Runtime Performance, Product, Documentation
Dependencies: TD-001, ADR-0001

## Liability

Canonical documents contain incompatible April throughput, memory, correctness,
and grade claims. Sampled-output Jaccard is sometimes described as a defect,
sometimes expected stochastic behavior, and sometimes fixed without one retained
post-fix matrix.

## Remediation

- Define a versioned benchmark manifest: commit, hardware, driver, model digest,
  config, prompt suite, decoding policy, warmup, repetitions, and thresholds.
- Use greedy token/logit comparisons for deterministic correctness and semantic
  scoring only as a separate product-quality measure.
- Compare matched context/KV and timeout settings across providers.
- Generate one canonical summary from retained raw artifacts.
- Remove or clearly archive release-facing claims that lack provenance.

## Exit Criteria

- ADR-0001 decision matrix is complete for at least two representative model sizes.
- Results reproduce within documented variance across three runs.
- `VISION`, `Roadmap`, benchmark, and competitive docs consume one evidence source.
- Native support envelope and default-provider decision are explicit.
