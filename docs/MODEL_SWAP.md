# Model swap guide (aiserver1 dual-GPU lineup)

How to swap the serving lineup on the dual-GPU host — with measured
evidence for the swap-in candidate this was written for (Qwen3.8-27B,
unlocked by the vendored llama.cpp v0.7.0) and a recommendation grounded
in those numbers.

All measurements: RTX-class bench host (Radeon AI PRO R9700, 32 GB,
ROCm) and Apple M1 Max (64 GB, MPS), 2026-10-27/29, InferFlux v0.6.x,
prompts of 433–516 tokens, 256-token decodes, temperature 0.

## Paths

### Path A — config variant + restart (supported, proven)

The reliable full-swap. One model of the 17 GB class fits per 32 GB GPU;
the swap exchanges the default.

1. Derive a swap config from the serving config, replacing the default
   model's entry (auth section unchanged):

```yaml
models:
- id: qwen3.8-27b
  path: /mnt/c/Users/vjsin/inferflux-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
  backend: llama_cpp_rocm
  device: rocm:0
  context_size: 65536      # total pool shared by the sequences
  gpu_layers: -1
  max_parallel_sequences: 4
  kv_cache_type: f16
  default: true
  format: gguf
# ...gpt-oss-20b (cuda:0) and bge-small-en-v1.5 unchanged
```

2. Point the service at it and restart:

```bash
systemctl --user edit-defaults  # or edit the unit's --config argument
systemctl --user daemon-reload && systemctl --user restart inferfluxd
```

3. Verify: `GET /livez`, then `GET /v1/models` — expect the swap model
   with `sequence_capacity: 4` and `context_tokens_per_sequence` equal to
   `context_size / max_parallel_sequences`. A 17.5 GB model loads in
   roughly 3–4 minutes.

Rollback is the mirror image: restore the previous `--config` argument
and restart. Keep a unit backup (`cp inferfluxd.service
inferfluxd.service.pre-swap`) before the first edit.

### Path B — admin API (best-effort; read the limits first)

`POST /v1/admin/models` (load), `DELETE /v1/admin/models` (unload),
`PUT /v1/admin/models/default` — see the Admin Guide §6.2. Two hard
limits for a **default-model swap**:

- Unload answers 409 `model_in_use` with the live reference count while
  anything holds the backend (serving leases, a speculative draft-model
  reference, startup references before init completes). An idle default
  can be unloaded; verify with the 409 body's count rather than assuming.
- An unloaded id stays **retired for the process lifetime** — reloading
  the same id needs a restart. Swapping A→B→A therefore ends at a
  restart anyway.

Practical rule: the API is for adding/removing **non-default** models
and draining before maintenance; a deliberate lineup change is Path A.

## Evidence: Qwen3.8-27B vs qwen3-coder-30b (R9700, Q4-class)

| Metric | qwen3-coder-30b (default) | qwen3.8-27b (swap-in) |
| --- | --- | --- |
| Decode throughput | 56.2 tok/s | 21.5 tok/s (2.6× slower) |
| TTFT (median) | 0.10 s | 0.20 s |
| KV prefix reuse (3 identical requests) | 432 / 432 / 432 cached | **0 / 0 / 0** |
| Repeat-traffic cost | ~0.10 s warm walls | every request re-prefills (~1.5 s walls) |
| M1 Max MPS decode (reference) | 71.8 tok/s | 11.4 tok/s |
| Output style | direct | reasoning model (`reasoning_content` streams first) |

The zero reuse is architectural, not a bug: Qwen3.8 is a hybrid
GDN-attention model and the backend logs
`CopySequencePrefix trim failed (hybrid memory); seq cleared for
full-prefill fallback` on every reuse attempt — the recurrent state
cannot be prefix-copied, so each request pays full prefill. Dense-27B
decode is also compute-heavier than the coder MoE's 3.3B active
parameters. Both reproduce identically across restarts and hosts.

## Recommendation

- **Keep `qwen3-coder-30b` as the default.** It is 2.6× faster on this
  GPU and its full-attention architecture gets ~100% accepted KV reuse
  on repeat-heavy agentic traffic (measured: 9 of 9 lookup hits
  accepted, 6,777 tokens skipped in a 12-request replay — see the
  2026-10-18 resolution doc's closing addendum).
- **Swap in `qwen3.8-27b` (Path A) for sessions that want dense-27B
  general reasoning quality on long, one-shot contexts** — where each
  request is fresh (no shared prefixes to lose) and its reasoning-style
  output is desirable. Swap back when the session ends.
- **Do not** use it for the repeat-heavy workload class (multi-turn
  agents, triage): zero prefix reuse means every turn re-prefills the
  full history.

VRAM check for the swap: 17.5 GB weights + KV pool
(`context_size` × per-token KV; this model ≈ 6 GB at 65536 f16) +
compute buffers ≈ 25 GB — fits the 32 GB card with headroom.
