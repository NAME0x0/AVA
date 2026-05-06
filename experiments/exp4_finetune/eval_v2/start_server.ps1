# Launch llama-server for AVA-v2 full evaluation.
# Q8_0 model, full GPU offload, Flash Attention, 4 parallel slots, Q8 KV cache.

$ErrorActionPreference = "Stop"

$LLAMA = "D:\AVA\experiments\exp5_gemma4\tools\llama.cpp-head\build\bin\llama-server.exe"
$MODEL = "D:\AVA\gguf_build\gguf\AVA-v2-Q8_0.gguf"
$LOG   = "D:\AVA\experiments\exp4_finetune\eval_v2\logs\server.log"

if (-not (Test-Path $LLAMA)) { throw "llama-server not found at $LLAMA" }
if (-not (Test-Path $MODEL)) { throw "model not found at $MODEL" }

$env:LLAMA_ARG_FLASH_ATTN = "on"
$env:GGML_CUDA_NO_PINNED  = "0"

$args = @(
    "-m", $MODEL,
    "-ngl", "99",
    "-fa", "on",
    "-c", "8192",
    "-np", "4",
    "-cb",
    "--jinja",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "-b", "2048",
    "-ub", "512",
    "--host", "127.0.0.1",
    "--port", "8765",
    "--threads", "8",
    "--threads-http", "4",
    "--metrics",
    "--no-webui"
)

Write-Host "Starting llama-server..."
Write-Host "  Model: $MODEL"
Write-Host "  Log:   $LOG"
Write-Host "  Args:  $($args -join ' ')"
& $LLAMA @args 2>&1 | Tee-Object -FilePath $LOG
