"""Upload AVA v2 adapter to Hugging Face Hub.

Usage:
    # First login:
    huggingface-cli login

    # Then upload:
    python experiments/exp4_finetune/scripts/upload_to_hf.py

    # Or with custom repo name:
    python experiments/exp4_finetune/scripts/upload_to_hf.py --repo-id your-username/AVA-v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def upload_model(repo_id: str, model_dir: str) -> str:
    api = HfApi()
    model_path = Path(model_dir)

    # Files to upload
    files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "training_report.json",
        "README.md",
    ]

    print(f"Creating repo: {repo_id}")
    api.create_repo(repo_id, exist_ok=True, repo_type="model")

    print(f"Uploading files from {model_path}...")
    for fname in files:
        fpath = model_path / fname
        if fpath.exists():
            print(f"  Uploading {fname} ({fpath.stat().st_size / 1e6:.1f} MB)")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
            )
        else:
            print(f"  Skipping {fname} (not found)")

    url = f"https://huggingface.co/{repo_id}"
    print(f"\nUpload complete: {url}")
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload AVA v2 to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default="NAME0x0/AVA-v2",
        help="HuggingFace repo ID (default: afsah/AVA-v2)",
    )
    parser.add_argument(
        "--model-dir",
        default="D:/AVA/experiments/exp4_finetune/models/AVA-v2",
        help="Local model directory",
    )
    args = parser.parse_args()
    upload_model(args.repo_id, args.model_dir)
