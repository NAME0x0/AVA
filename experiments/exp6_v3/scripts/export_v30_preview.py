"""Export smoke test: step-2167 adapters -> merged BF16 -> GGUF-ready dir.

THE question this answers (PLAN_2026-07-15_v31 s7b decision 1): can AVA v3.0
actually ship on the 4 GB laptop lane? Chain under test:

    donor (Qwen3.5-4B, BF16 on CPU)                # 32 GB RAM box, no GPU needed
      + C5 adapters from the Hub (step ~2167)
      -> PeftModel.merge_and_unload()
      -> save_pretrained (BF16 safetensors + tokenizer)
      -> [manual next step] llama.cpp convert_hf_to_gguf.py + quantize + run

CRITICAL detail: the step-2167 checkpoint was trained BEFORE the DoRA switch
landed (use_dora shipped 5de6126, after that run) — those adapters are PLAIN
LoRA r=16. The reconstruction here must therefore use use_dora=False or the
saved state dict won't line up. Future checkpoints trained with DoRA must
flip MERGE_USE_DORA below (or read it from the run config of the checkpoint).

Run on the laptop (CPU-only is fine, ~20-40 min dominated by the 9 GB donor
download on first run):

    cd experiments/exp6_v3
    python scripts/export_v30_preview.py [--out D:/AVA/exports/v30-preview]

Then (manual, documents itself at the end): llama.cpp conversion + smoke chat.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# This is a CPU merge. If causal_conv1d / flash-linear-attention are installed,
# transformers' qwen3.5 path auto-selects their CUDA-only kernels and the CPU
# sanity generation dies ("Expected x.is_cuda()" / Triton "cpu tensor?" —
# field, 2026-07-17). Mask BOTH before any transformers import -> torch fallback.
for _m in ("causal_conv1d", "fla", "flash_linear_attention"):
    sys.modules[_m] = None  # type: ignore[assignment]

CKPT_REPO = "NAME0x0/AVA-v3-checkpoints"
DONOR = "Qwen/Qwen3.5-4B"
MERGE_USE_DORA = False  # step-2167 adapters predate the DoRA switch — see docstring


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path("D:/AVA/exports/v30-preview")))
    ap.add_argument("--repo", default=CKPT_REPO)
    ap.add_argument("--donor", default=DONOR)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.checkpoint_sync import CheckpointSync

    print("[export] loading donor BF16 on CPU (first run downloads ~9 GB)...")
    tokenizer = AutoTokenizer.from_pretrained(args.donor, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.donor, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("[export] mounting adapter scaffold (plain LoRA r=16 — pre-DoRA ckpt)...")
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
        use_dora=MERGE_USE_DORA,
    )
    model = get_peft_model(model, lcfg)

    print("[export] pulling latest C5 adapters from the Hub...")
    step = CheckpointSync(args.repo, phase="C5", trainable_only=True).resume(model) - 1
    print(f"[export] adapters restored from step {step}")

    print("[export] merging adapters into base weights...")
    model = model.merge_and_unload()

    print(f"[export] saving merged BF16 model -> {out}")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    # AutoModelForCausalLM drops the donor's MTP draft-head WEIGHTS, but the
    # saved config still declared mtp_num_hidden_layers=1 — llama.cpp then
    # expects a blk.32 that doesn't exist ("missing tensor blk.32.attn_norm",
    # field 2026-07-17). Zero it so the GGUF is self-consistent.
    import json

    cfg_path = out / "config.json"
    cfg_json = json.loads(cfg_path.read_text())
    if cfg_json.get("mtp_num_hidden_layers"):
        cfg_json["mtp_num_hidden_layers"] = 0
        cfg_path.write_text(json.dumps(cfg_json, indent=2))
        print("[export] config: mtp_num_hidden_layers -> 0 (weights not exported)")

    # quick sanity: merged model must still generate coherent code (CPU, slow)
    print("[export] CPU generation sanity check (may take ~1 min)...")
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a Python function that reverses a string."}],
        add_generation_prompt=True, return_tensors="pt", return_dict=False,
        enable_thinking=False,
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=80, do_sample=False)
    text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
    print("--- sample generation ---")
    print(text[:500])
    assert "def " in text or "return" in text, "merged model generated no code — merge suspect!"

    print(f"""
[export] MERGE OK — step {step} merged model at {out}

Next (llama.cpp leg of the smoke test — needs a recent llama.cpp checkout
with qwen3.5 hybrid support; pin the commit that works in the run log):

  git clone https://github.com/ggml-org/llama.cpp
  pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
  python llama.cpp/convert_hf_to_gguf.py {out} --outfile {out}/ava-v30-preview-bf16.gguf
  # quantize (download a llama.cpp release build for Windows, or build):
  llama-quantize {out}/ava-v30-preview-bf16.gguf {out}/ava-v30-preview-q4_k_m.gguf q4_k_m
  # first conversation with your own model:
  llama-cli -m {out}/ava-v30-preview-q4_k_m.gguf -p "Write a Python function to merge two sorted lists." -n 256 -ngl 99
""")


if __name__ == "__main__":
    main()
