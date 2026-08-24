@echo off
setlocal

set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set INFERFLUX_ENABLE_FUSED_GATE_UP_SILU=0
set INFERFLUX_CUDA_SYNC_TRACE=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=C:\Windows\System32;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;C:\Users\vjsin\code\inferflux\build\bin\Release;C:\Users\vjsin\code\inferflux\build\Release;%PATH%
set MODEL=C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf
set PROBE=C:\Users\vjsin\code\inferflux\build\Release\inferflux_first_token_probe.exe

echo ============================================================
echo  A/B Throughput: Q8_1 Activations ON vs OFF
echo  Model: Qwen2.5-3B Q4_K_M, 100 tokens, RTX 4000 Ada
echo ============================================================

echo.
echo --- Run 1: Q8_1 OFF (baseline) ---
set INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=1
%PROBE% --backend inferflux_cuda --model %MODEL% --prompt "The capital of France is" --top-n 5 --max-tokens 100 2>&1 | findstr /R "token_count ok native_sync_trace.*batch_result_ready"

echo.
echo --- Run 2: Q8_1 ON ---
set INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=0
%PROBE% --backend inferflux_cuda --model %MODEL% --prompt "The capital of France is" --top-n 5 --max-tokens 100 2>&1 | findstr /R "token_count ok native_sync_trace.*batch_result_ready"

echo.
echo --- Run 3: Q8_1 OFF (confirm) ---
set INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=1
%PROBE% --backend inferflux_cuda --model %MODEL% --prompt "The capital of France is" --top-n 5 --max-tokens 100 2>&1 | findstr /R "token_count ok native_sync_trace.*batch_result_ready"

echo.
echo --- Run 4: Q8_1 ON (confirm) ---
set INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=0
%PROBE% --backend inferflux_cuda --model %MODEL% --prompt "The capital of France is" --top-n 5 --max-tokens 100 2>&1 | findstr /R "token_count ok native_sync_trace.*batch_result_ready"

echo.
echo ============================================================
echo  Done. Compare avg_us for batch_result_ready across runs.
echo ============================================================
