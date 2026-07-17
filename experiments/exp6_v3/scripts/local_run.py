"""Local run stack: merged model -> GGUF -> quantize -> tuned llama.cpp launch.

Target box = the design-target laptop: RTX A2000 4 GB (sm86 Ampere) + 32 GB RAM,
Windows. Automates the second leg of the export smoke (PLAN s7b decision 1) and
becomes the standing local launcher afterwards.

What "intelligent GPU/CPU offload" means here, concretely:
  - llama.cpp offload is per-layer (-ngl). We compute how many layers fit from
    ACTUAL free VRAM (nvidia-smi) against the quantized per-layer size, keep a
    safety margin for compute buffers, and offload that many; the rest run on
    CPU with the 32 GB RAM. Q4_K_M of the 4B is ~2.5 GB -> normally ALL 32
    layers + KV fit and -ngl 99 wins outright, but the calculation keeps us
    honest when context grows or another app holds VRAM.
  - The donor's hybrid architecture is a local superpower: only ~8 of 32
    layers are softmax-attention, so KV cache is ~4x smaller than a dense 4B
    at the same context. We still quantize KV (q8_0) for headroom.
  - Flash attention on (-fa). MTP/speculative note: our merged export does NOT
    carry the donor's MTP draft heads (AutoModelForCausalLM drops them), so no
    --spec-type here; revisit at C7 packing.

Usage (after export_v30_preview.py produced the merged dir):

    python scripts/local_run.py convert   # HF dir -> BF16 GGUF -> Q4_K_M GGUF
    python scripts/local_run.py chat      # tuned llama-cli session
    python scripts/local_run.py serve     # llama-server (harness endpoint)
    python scripts/local_run.py bench     # tok/s snapshot

llama.cpp binaries: set LLAMACPP_DIR to a recent Windows CUDA build (needs
qwen3.5 hybrid support -> use a fresh release; pin the working build tag in
the run log). convert step additionally needs the llama.cpp repo checkout for
convert_hf_to_gguf.py -> set LLAMACPP_SRC.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MERGED_DIR = Path(os.environ.get("AVA_EXPORT_DIR", "D:/AVA/exports/v30-preview"))
GGUF_BF16 = MERGED_DIR / "ava-v30-preview-bf16.gguf"
GGUF_Q4 = MERGED_DIR / "ava-v30-preview-q4_k_m.gguf"

# Q4_K_M ~= 0.58 bytes/param. 4.24B params over 32 blocks + embed/head.
_LAYER_GB = 4.24 * 0.58 / 32          # ~77 MB per block quantized
_NON_LAYER_GB = 0.55                   # embeddings + head + norms (248K vocab, Q6)
_SAFETY_GB = 0.7                       # compute buffers + KV(q8, hybrid) + slack


def _bins() -> Path:
    d = os.environ.get("LLAMACPP_DIR")
    if d and (Path(d) / "llama-cli.exe").exists():
        return Path(d)
    for cand in ("llama-cli", "llama-cli.exe"):
        w = shutil.which(cand)
        if w:
            return Path(w).parent
    sys.exit(
        "llama.cpp binaries not found. Download a recent Windows CUDA release\n"
        "(https://github.com/ggml-org/llama.cpp/releases, llama-*-bin-win-cuda-x64.zip\n"
        "— must be recent enough for qwen3.5 hybrid), unzip, and set LLAMACPP_DIR."
    )


def _free_vram_gb() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return float(out) / 1024.0
    except Exception:  # noqa: BLE001 - no NVML -> CPU-only fallback
        return 0.0


def plan_ngl(n_layers: int = 32) -> int:
    """Layers that fit in free VRAM with margin; 999 = everything (llama.cpp caps)."""
    free = _free_vram_gb()
    budget = free - _NON_LAYER_GB - _SAFETY_GB
    if budget <= 0:
        print(f"[local] free VRAM {free:.2f} GB -> CPU-only (-ngl 0)")
        return 0
    n = min(n_layers, int(budget / _LAYER_GB))
    ngl = 999 if n >= n_layers else n
    print(f"[local] free VRAM {free:.2f} GB -> offloading "
          f"{'ALL' if ngl == 999 else ngl} layers (est {n * _LAYER_GB + _NON_LAYER_GB:.2f} GB)")
    return ngl


def convert() -> None:
    src = os.environ.get("LLAMACPP_SRC")
    if not src or not (Path(src) / "convert_hf_to_gguf.py").exists():
        sys.exit("Set LLAMACPP_SRC to a llama.cpp checkout (needs convert_hf_to_gguf.py).")
    if not (MERGED_DIR / "config.json").exists():
        sys.exit(f"No merged model at {MERGED_DIR} — run export_v30_preview.py first.")
    if not GGUF_BF16.exists():
        subprocess.run(
            [sys.executable, str(Path(src) / "convert_hf_to_gguf.py"),
             str(MERGED_DIR), "--outfile", str(GGUF_BF16), "--outtype", "bf16"],
            check=True,
        )
    subprocess.run(
        [str(_bins() / "llama-quantize"), str(GGUF_BF16), str(GGUF_Q4), "q4_k_m"],
        check=True,
    )
    print(f"[local] GGUF ready: {GGUF_Q4} ({GGUF_Q4.stat().st_size / 1e9:.2f} GB)")


def _common_flags() -> list[str]:
    # Donor is thinking-by-default. For a coding tool, direct answers are the
    # sane default (thinking burns the token budget monologuing before code —
    # field 2026-07-18: a 220-token merge-two-lists gen never reached the code).
    # AVA_THINK=1 re-enables it for hard problems.
    think = os.environ.get("AVA_THINK") == "1"
    flags = [
        "-m", str(GGUF_Q4),
        "-ngl", str(plan_ngl()),
        "-fa", "on",                       # flash attention
        "--cache-type-k", "q8_0",          # KV headroom (hybrid = tiny KV anyway)
        "--cache-type-v", "q8_0",
        "-c", "8192",                      # sane default ctx on 4 GB; raise on demand
    ]
    if not think:
        flags += ["--chat-template-kwargs", '{"enable_thinking": false}']
    return flags


def chat() -> None:
    # -cnv = conversation mode (applies the chat template); interactive.
    subprocess.run([str(_bins() / "llama-cli"), *_common_flags(), "-cnv"], check=False)


def test() -> None:
    """Non-interactive single-turn proof: applies template, one turn, exits."""
    prompt = "Write a Python function that merges two sorted lists. Return only the code."
    subprocess.run(
        [str(_bins() / "llama-cli"), *_common_flags(), "-cnv", "-st",
         "--temp", "0", "-n", "256", "-p", prompt],
        check=False,
    )


def serve() -> None:
    subprocess.run(
        [str(_bins() / "llama-server"), *_common_flags(), "--jinja", "--port", "8080"],
        check=False,
    )


def bench() -> None:
    subprocess.run(
        [str(_bins() / "llama-bench"), "-m", str(GGUF_Q4), "-ngl", str(plan_ngl())],
        check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "chat", "test", "serve", "bench", "plan"])
    cmd = ap.parse_args().cmd
    if cmd == "plan":
        plan_ngl()
    else:
        {"convert": convert, "chat": chat, "test": test,
         "serve": serve, "bench": bench}[cmd]()


if __name__ == "__main__":
    main()
