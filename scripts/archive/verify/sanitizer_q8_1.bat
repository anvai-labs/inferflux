@echo off
setlocal

set INFERFLUX_DISABLE_STARTUP_ADVISOR=true
set INFERFLUX_DISABLE_CUDA_GRAPH=1
set INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=0
set INFERFLUX_ENABLE_FUSED_GATE_UP_SILU=0
set INFERFLUX_CUDA_DEBUG_OPERATOR_SELECTION=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=C:\Windows\System32;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;C:\Users\vjsin\code\inferflux\build\bin\Release;C:\Users\vjsin\code\inferflux\build\Release;%PATH%

echo === compute-sanitizer memcheck on Q8_1 crash ===
"%CUDA_PATH%\bin\compute-sanitizer.bat" --tool memcheck --show-backtrace yes ^
  C:\Users\vjsin\code\inferflux\build\Release\inferflux_first_token_probe.exe ^
  --backend inferflux_cuda ^
  --model C:\Users\vjsin\code\inferflux\models\qwen2.5-3b-instruct\qwen2.5-3b-instruct-q4_k_m.gguf ^
  --prompt "The capital of France is" ^
  --top-n 5 ^
  --max-tokens 2

echo === Exit code: %ERRORLEVEL% ===
