# Q6_K Vectorized SM89 Validation

This record scopes the automatic Q6_K vectorized decode claim to the hardware
actually measured on 2026-08-30: an NVIDIA RTX 4000 Ada (SM 8.9, 48 SMs,
360 GB/s reported bandwidth) using CUDA 13.2.78. It does not claim equivalent
speedups for every SM 8.9 device.

## Correctness

```sh
./build-cuda/inferflux_tests '[q6k_vectorized]' --durations yes
```

The test passed 18,149 assertions across two cases. A pure selector test proves
automatic and rollback behavior without inferring dispatch from equal outputs.
Runtime parity forces scalar and vector dispatch for
M=3, 4, 5, and 8 across overwrite, FP16-accumulate, and FP32-accumulate APIs,
and includes N=2048/K=11008 down-projection geometry. Boundary cases M=9 and
M=16 verify that the vector policy does not
alter the tiled MMQ overwrite/FP16-accumulate path. FP32 accumulation is
decode-only and is expected to reject M>8.

## Performance

```sh
./build-cuda/benchmark_downproj_q81
```

The benchmark uses N=2048, K=11008, 30 warmups, and 200 timed iterations. Five
serial runs produced:

| M | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
|---:|------:|------:|------:|------:|------:|-------:|
| 3 | 1.448x | 1.449x | 0.939x | 1.160x | 1.639x | 1.448x |
| 4 | 1.679x | 1.154x | 1.848x | 1.877x | 1.434x | 1.679x |
| 5 | 1.796x | 1.671x | 1.638x | 1.797x | 1.643x | 1.671x |
| 8 | 1.944x | 2.060x | 1.883x | 1.611x | 1.814x | 1.883x |

Every run passed the benchmark's CUDA checks and scalar/vector parity gate
(`max_abs_diff <= 0.08`). Re-run on each additional SM89 profile before
generalizing the performance claim.

Five exact-geometry M=16 controls measured a 1.017x median with zero output
difference,
confirming that toggling the MMVQ vector policy does not change the M=9-64 MMQ
path. M=16 performance work belongs to the separately gated MMQ/MMA path.
