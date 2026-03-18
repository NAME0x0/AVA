"""Monitor training and auto-run benchmarks when complete.

Watches the training log file for completion, then runs
ARC-Challenge and GSM8K benchmarks on the fine-tuned model.
"""
import json
import re
import sys
import time
from pathlib import Path


LOG_FILE = Path("D:/AVA/experiments/exp4_finetune/training_fast_v2.log")
OUTPUT_DIR = Path("D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v1")
BASE_MODEL = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
RESULTS_DIR = Path("D:/AVA/experiments/exp4_finetune/results")


def check_training_status():
    """Parse log file for training status."""
    if not LOG_FILE.exists():
        return "not_started", {}

    text = LOG_FILE.read_text(encoding="utf-8", errors="replace")

    # Check for completion
    if "Training Complete!" in text:
        # Extract final metrics
        loss_match = re.search(r"Train loss: ([\d.]+)", text)
        eval_match = re.search(r"Eval loss: ([\d.]+)", text)
        time_match = re.search(r"Time: (\d+)s", text)
        return "complete", {
            "train_loss": float(loss_match.group(1)) if loss_match else None,
            "eval_loss": float(eval_match.group(1)) if eval_match else None,
            "elapsed_seconds": int(time_match.group(1)) if time_match else None,
        }

    # Check for error
    if "Error" in text.split("\n")[-1] or "Traceback" in text[-2000:]:
        return "error", {"last_lines": text[-500:]}

    # Check progress
    steps_match = list(re.finditer(r"(\d+)/(\d+).*?([\d.]+)s/it", text))
    loss_matches = list(re.finditer(r"'loss': '([\d.]+)'", text))

    if steps_match:
        last = steps_match[-1]
        current_step = int(last.group(1))
        total_steps = int(last.group(2))
        speed = float(last.group(3))
        remaining = (total_steps - current_step) * speed
        return "training", {
            "step": current_step,
            "total_steps": total_steps,
            "speed_s_per_step": speed,
            "remaining_seconds": remaining,
            "remaining_hours": round(remaining / 3600, 1),
            "latest_loss": float(loss_matches[-1].group(1)) if loss_matches else None,
        }

    return "loading", {}


def run_benchmarks():
    """Run benchmarks on the fine-tuned model."""
    # Check if adapter exists
    adapter_config = OUTPUT_DIR / "adapter_config.json"
    if not adapter_config.exists():
        print("ERROR: No adapter found. Training may not have saved properly.")
        return

    print("\n" + "=" * 60)
    print("Running benchmarks on fine-tuned model...")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from benchmark_full import run_benchmarks as bench, BenchmarkConfig

    results = bench(BenchmarkConfig(
        model_path=BASE_MODEL,
        adapter_path=str(OUTPUT_DIR),
        arc_limit=100,
        gsm8k_limit=50,
        output_dir=str(RESULTS_DIR),
    ))

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Benchmark':<25} {'Scratch 14M':>12} {'Qwen Base':>12} {'AVA Tuned':>12}")
    print("-" * 60)

    arc_acc = results["benchmarks"].get("arc_challenge", {}).get("accuracy", 0)
    gsm_acc = results["benchmarks"].get("gsm8k", {}).get("accuracy", 0)

    print(f"{'ARC-Challenge':<25} {'24%':>12} {'66%':>12} {f'{arc_acc*100:.1f}%':>12}")
    print(f"{'GSM8K':<25} {'0-2%':>12} {'4%':>12} {f'{gsm_acc*100:.1f}%':>12}")


def monitor():
    """Main monitoring loop."""
    print("AVA Training Monitor")
    print("=" * 40)
    print(f"Log: {LOG_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    while True:
        status, info = check_training_status()

        if status == "complete":
            print(f"\n[COMPLETE] Training finished!")
            print(f"  Train loss: {info.get('train_loss')}")
            print(f"  Eval loss: {info.get('eval_loss')}")
            print(f"  Time: {info.get('elapsed_seconds', 0)/60:.1f} min")
            run_benchmarks()
            break
        elif status == "error":
            print(f"\n[ERROR] Training failed!")
            print(info.get("last_lines", ""))
            break
        elif status == "training":
            step = info["step"]
            total = info["total_steps"]
            pct = step / total * 100
            loss = info.get("latest_loss", "?")
            remaining = info.get("remaining_hours", "?")
            print(f"  [{step}/{total}] {pct:.1f}% | loss={loss} | ETA: {remaining}h", end="\r")
        elif status == "loading":
            print("  Loading model...", end="\r")
        else:
            print("  Waiting for training to start...", end="\r")

        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    if "--check" in sys.argv:
        status, info = check_training_status()
        print(f"Status: {status}")
        print(json.dumps(info, indent=2))
    else:
        monitor()
