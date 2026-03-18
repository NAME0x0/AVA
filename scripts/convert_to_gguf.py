"""Merge AVA v2 LoRA adapter into base model and convert to GGUF.

This script handles the full pipeline:
  1. Load Qwen3.5-2B base model (full precision, no quantization)
  2. Merge the LoRA adapter weights into the base
  3. Save the merged model in HuggingFace safetensors format
  4. Convert to GGUF via llama.cpp's convert_hf_to_gguf.py
  5. Quantize to Q4_K_M, Q8_0 (configurable)

Usage:
    # Merge only (for local conversion):
    python scripts/convert_to_gguf.py --merge-only

    # Full pipeline (requires llama.cpp clone):
    python scripts/convert_to_gguf.py --llama-cpp ./llama.cpp

    # Custom adapter source:
    python scripts/convert_to_gguf.py --adapter NAME0x0/AVA-v2 --llama-cpp ./llama.cpp

    # Specific quantizations:
    python scripts/convert_to_gguf.py --llama-cpp ./llama.cpp --quants Q4_K_M Q5_K_M Q8_0
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_ADAPTER = "NAME0x0/AVA-v2"
DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"]


def merge_adapter(adapter: str, output_dir: Path) -> Path:
    """Merge LoRA adapter into base model and save as standard HF model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {BASE_MODEL} (bfloat16, CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.bfloat16)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    param_count = sum(p.numel() for p in model.parameters())
    size_gb = sum(f.stat().st_size for f in output_dir.iterdir()) / 1e9
    print(f"Merged model: {param_count:,} parameters, {size_gb:.2f} GB on disk")
    return output_dir


def convert_to_gguf(merged_dir: Path, llama_cpp: Path, output_dir: Path) -> Path:
    """Convert merged HF model to F16 GGUF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert_script}. "
            f"Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
        )

    f16_path = output_dir / "AVA-v2-f16.gguf"
    cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ]
    print(f"Converting to GGUF F16: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"F16 GGUF: {f16_path} ({f16_path.stat().st_size / 1e9:.2f} GB)")
    return f16_path


def quantize_gguf(f16_path: Path, llama_cpp: Path, output_dir: Path, quants: list[str]) -> list[Path]:
    """Quantize F16 GGUF to specified formats."""
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp / "llama-quantize"
    if not quantize_bin.exists():
        # Try platform-specific locations
        for candidate in [
            llama_cpp / "build" / "llama-quantize",
            llama_cpp / "build" / "bin" / "llama-quantize",
            llama_cpp / "build" / "Release" / "llama-quantize.exe",
            llama_cpp / "build" / "bin" / "Release" / "llama-quantize.exe",
        ]:
            if candidate.exists():
                quantize_bin = candidate
                break
        else:
            raise FileNotFoundError(
                f"llama-quantize not found in {llama_cpp}. "
                f"Build llama.cpp: cd {llama_cpp} && cmake -B build && cmake --build build --config Release"
            )

    results = []
    for quant in quants:
        out_path = output_dir / f"AVA-v2-{quant}.gguf"
        cmd = [str(quantize_bin), str(f16_path), str(out_path), quant]
        print(f"\nQuantizing to {quant}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {quant}: {out_path} ({size_mb:.0f} MB)")
        results.append(out_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AVA v2 to GGUF for Ollama/llama.cpp")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help=f"Adapter source (default: {DEFAULT_ADAPTER})")
    parser.add_argument("--merged-dir", default="./gguf_build/merged", help="Directory for merged HF model")
    parser.add_argument("--output-dir", default="./gguf_build/gguf", help="Directory for GGUF output files")
    parser.add_argument("--llama-cpp", default=None, help="Path to llama.cpp clone (skip GGUF conversion if omitted)")
    parser.add_argument("--merge-only", action="store_true", help="Only merge adapter, skip GGUF conversion")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step (use existing merged model)")
    parser.add_argument("--quants", nargs="+", default=DEFAULT_QUANTS, help=f"Quantization formats (default: {DEFAULT_QUANTS})")
    parser.add_argument("--keep-f16", action="store_true", help="Keep the F16 GGUF (large, ~3.8 GB)")
    args = parser.parse_args()

    merged_dir = Path(args.merged_dir)
    output_dir = Path(args.output_dir)

    # Step 1: Merge
    if not args.skip_merge:
        merge_adapter(args.adapter, merged_dir)
    else:
        if not merged_dir.exists():
            parser.error(f"--skip-merge specified but {merged_dir} does not exist")
        print(f"Skipping merge, using existing model at {merged_dir}")

    if args.merge_only:
        print("\nMerge complete. To convert to GGUF, run again with --llama-cpp <path>")
        return

    # Step 2 & 3: Convert and quantize
    if args.llama_cpp is None:
        parser.error("--llama-cpp is required for GGUF conversion (or use --merge-only)")

    llama_cpp = Path(args.llama_cpp)
    f16_path = convert_to_gguf(merged_dir, llama_cpp, output_dir)
    quant_paths = quantize_gguf(f16_path, llama_cpp, output_dir, args.quants)

    # Clean up F16 if not keeping it
    if not args.keep_f16 and f16_path.exists():
        print(f"\nRemoving F16 intermediate ({f16_path.stat().st_size / 1e9:.2f} GB)")
        f16_path.unlink()

    print("\nDone! GGUF files:")
    for p in quant_paths:
        print(f"  {p.name}: {p.stat().st_size / 1e6:.0f} MB")

    print(f"\nTo use with Ollama:")
    print(f"  ollama create ava-v2 -f Modelfile")
    print(f"  ollama run ava-v2")


if __name__ == "__main__":
    main()
