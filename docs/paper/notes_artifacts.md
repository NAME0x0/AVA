# Public-artifact verification (Task 4) — verified 2026-07-20 via HF

All accessed 2026-07-20 via gstack browse. All Apache-2.0.

## Confirmed public
- **Adapter:** `NAME0x0/AVA-v2` — https://huggingface.co/NAME0x0/AVA-v2. PEFT/Safetensors, tags `qlora,4bit,low-resource,arc-challenge,gsm8k,mmlu,science,math,reasoning`. Model tree: `Qwen/Qwen3.5-2B-Base → Qwen/Qwen3.5-2B → this adapter`.
- **GGUF ladder:** `NAME0x0/AVA-v2-GGUF` — https://huggingface.co/NAME0x0/AVA-v2-GGUF. GGUF/imatrix; llama.cpp/ollama/lm-studio tags.
- **Merged bf16:** `NAME0x0/AVA-v2-merged` — referenced from the adapter card (pre-merged, plain transformers). Verify directly before citing as a standalone artifact.

## Correction to spec §9 / plan Task 18
The **live HF model card at `NAME0x0/AVA-v2` is already fully populated** (real description: "42 MB QLoRA adapter for Qwen/Qwen3.5-2B, trained and evaluated entirely on a single NVIDIA RTX A2000 Laptop GPU with 4 GB VRAM… 82% ARC-C… Method: QLoRA (4-bit NF4 base + LoRA rank 16, alpha 32 on all attention + MLP projections)"; already carries a citation title *"AVA v2: QLoRA Fine-tuning Under Extreme VRAM Constraints"*).

The **stub is only the local file** `experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v2/README.md`. So the reproducibility claim is safe (public card is good). Task 18 is therefore **"sync the local file to match the live card + apply the reconciled task-count phrasing + align with the paper,"** not "write a card from scratch."

Note: the live card's line "most peers above used cluster-scale training compute" is a compute-scale claim (not an accuracy-superiority claim); keep it only if defensible, and never let the paper imply AVA-v2 beats the base in general capability (see notes_base_model.md).

## Still to confirm before the appendix cites it
- GitHub repo `NAME0x0/AVA` (memory says pushed there; confirm URL + visibility when writing the reproducibility appendix).
