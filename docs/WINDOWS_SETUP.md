# Windows setup notes

> Standalone version of the Windows-specific build instructions. Also published as a public gist for easy sharing: **https://gist.github.com/NAME0x0/8fe9084e606d3e7ae17d4f1da6a96667** — keep both copies in sync when editing (`gh gist edit 8fe9084e606d3e7ae17d4f1da6a96667 docs/WINDOWS_SETUP.md`).

The hard part of running AVA v2 on Windows is **Triton + Flash-Linear-Attention + BitsAndBytes co-existence**. None of them love Windows. This doc captures every workaround that made the released training run work on Windows 11 + RTX A2000 Laptop + Python 3.13 + PyTorch 2.10.0+cu130.

## Prerequisites

1. **Visual Studio with C++ Build Tools** (2022 or 2026 both work). Triton needs MSVC `cl.exe`.
2. **NVIDIA driver** supporting CUDA 13.0+ (any modern driver from 2025+).
3. **Python 3.10–3.13**. We tested 3.13.
4. **Git LFS** if you plan to clone the released adapter.

## Core install

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install transformers==5.3.0 peft==0.18.1 bitsandbytes==0.49.2 datasets accelerate
```

## Triton (Windows)

The community port:

```powershell
pip install triton-windows==3.6.0.post26
```

Triton needs a C compiler at runtime. The bundled TinyCC fallback **does not work reliably**. Set `CC` to MSVC `cl.exe`:

```powershell
# Find your cl.exe
$cl = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter cl.exe |
      Where-Object { $_.FullName -like "*Hostx64\x64*" } |
      Select-Object -First 1 -ExpandProperty FullName
$env:CC = "`"$cl`""
```

Or hard-code:

```powershell
$env:CC = "`"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36014\bin\Hostx64\x64\cl.exe`""
```

Persist with `[Environment]::SetEnvironmentVariable("CC", $env:CC, "User")`.

## Flash-Linear-Attention + causal-conv1d

```powershell
pip install flash-linear-attention==0.4.2
```

`causal-conv1d` (FLA dep) does not build on stock Windows. Use the patched Windows fork:

```powershell
git clone https://github.com/sdbds/causal-conv1d-for-windows
cd causal-conv1d-for-windows
pip install . --no-build-isolation
```

The patch adds `/Zc:preprocessor` to the MSVC flags and targets your GPU compute capability.

## Critical config flag

When FLA is installed, **always** load the model with `attn_implementation="sdpa"`:

```python
AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-2B",
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # required — FLA crashes on BnB 4-bit weights otherwise
)
```

FLA tries to merge q/k/v into combined projections, which is incompatible with BnB 4-bit quantized tensors. SDPA mode bypasses that path.

## What does NOT work on Windows

| Component | Status | Workaround |
|---|---|---|
| Unsloth | OOM during model loading | use vanilla HF Trainer + manual freeze |
| `prepare_model_for_kbit_training()` | upcasts to fp32 → OOM on 4 GB | manually freeze base model params |
| stock `causal-conv1d` | MSVC preprocessor fails | use the [sdbds Windows fork](https://github.com/sdbds/causal-conv1d-for-windows) |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | not supported on Windows | ignore (warning only, no error) |
| Buffered training output | process dies, no logs | always run with `python -u` |

## Diagnostic checks

After install, sanity-check Triton compiles a kernel:

```powershell
python -c "import torch; from torch.nn.functional import scaled_dot_product_attention as sdpa; q=k=v=torch.randn(1,8,32,64,device='cuda',dtype=torch.bfloat16); print(sdpa(q,k,v).shape)"
```

If this hangs or errors, your `CC` is wrong or MSVC isn't on PATH.

Sanity-check BnB:

```powershell
python -c "import bitsandbytes as bnb; print(bnb.__version__); from bitsandbytes.nn import Linear4bit; print('ok')"
```

## Long-running training tips

- Use `python -u` so stdout flushes during the 100-minute run.
- Save checkpoints every 200 steps. Laptops thermally throttle and restart.
- HuggingFace Trainer's `--resume_from_checkpoint` works cleanly across restarts.
- Use `paged_adamw_8bit`. Standard AdamW will OOM at peak.

## Reference versions

| Component | Version |
|---|---|
| OS | Windows 11 26H2 |
| Python | 3.13 |
| PyTorch | 2.10.0+cu130 |
| CUDA | 13.0 |
| Transformers | 5.3.0 |
| PEFT | 0.18.1 |
| BitsAndBytes | 0.49.2 |
| Triton | 3.6.0.post26 (triton-windows) |
| Flash-Linear-Attention | 0.4.2 |
| causal-conv1d | 1.5.0.post8 (sdbds Windows fork) |

## Maintenance

Repo copy is the source of truth. After editing this file, mirror to the gist:

```bash
gh gist edit 8fe9084e606d3e7ae17d4f1da6a96667 docs/WINDOWS_SETUP.md
```

External links (HF model card, sharing on social) point to the gist; internal repo links point to this file.
