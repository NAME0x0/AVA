# Quantize AVA v2 BF16 GGUF into the release ladder, using the calibration imatrix.
# Run after llama-imatrix completes. Output: gguf_build/gguf/release/

$ErrorActionPreference = "Stop"
$bin = "D:\AVA\experiments\exp5_gemma4\tools\llama.cpp-head\build\bin"
$src = "D:\AVA\gguf_build\gguf\AVA-v2-BF16.gguf"
$imatrix = "D:\AVA\gguf_build\AVA-v2.imatrix"
$outDir = "D:\AVA\gguf_build\gguf\release"
New-Item -ItemType Directory -Force $outDir | Out-Null

# imatrix-guided small quants (the Gemma-4-QAT-style efficiency targets)
$quants = @("Q4_0", "Q4_K_M", "Q5_K_M", "IQ4_XS")
foreach ($q in $quants) {
    $out = Join-Path $outDir "AVA-v2-$q.gguf"
    if (Test-Path $out) { Write-Host "skip existing $q"; continue }
    Write-Host "=== quantizing $q ==="
    & "$bin\llama-quantize.exe" --imatrix $imatrix $src $out $q
    if ($LASTEXITCODE -ne 0) { throw "quantize $q failed" }
}

# Q8_0 reference (no imatrix needed at 8-bit)
$q8 = Join-Path $outDir "AVA-v2-Q8_0.gguf"
if (-not (Test-Path $q8)) {
    Write-Host "=== quantizing Q8_0 ==="
    & "$bin\llama-quantize.exe" $src $q8 Q8_0
    if ($LASTEXITCODE -ne 0) { throw "quantize Q8_0 failed" }
}

Get-ChildItem $outDir | Select-Object Name, @{n='GB';e={[math]::Round($_.Length/1GB,2)}} | Format-Table -AutoSize
