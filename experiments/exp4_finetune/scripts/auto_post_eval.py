"""Auto post-training evaluation.

Waits for training to complete, then runs quick eval and full benchmarks.
Designed to run in background while training proceeds.
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


def wait_for_completion(check_interval=120):
    """Poll log file until training completes or errors."""
    print("Waiting for training to complete...")
    while True:
        if not LOG_FILE.exists():
            time.sleep(check_interval)
            continue

        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")

        if "Training Complete!" in text:
            loss_match = re.search(r"Train loss: ([\d.]+)", text)
            time_match = re.search(r"Time: (\d+)s", text)
            print(f"\n[COMPLETE] Training finished!")
            if loss_match:
                print(f"  Final loss: {loss_match.group(1)}")
            if time_match:
                mins = int(time_match.group(1)) / 60
                print(f"  Time: {mins:.1f} min")
            return True

        if "Traceback" in text[-3000:] or "CUDA out of memory" in text[-3000:]:
            print(f"\n[ERROR] Training failed!")
            print(text[-500:])
            return False

        # Show progress
        steps_match = list(re.finditer(r"(\d+)/(\d+).*?([\d.]+)s/it", text))
        loss_matches = list(re.finditer(r"'loss': '([\d.]+)'", text))
        if steps_match:
            last = steps_match[-1]
            step, total = int(last.group(1)), int(last.group(2))
            pct = step / total * 100
            loss = loss_matches[-1].group(1) if loss_matches else "?"
            remaining_h = (total - step) * float(last.group(3)) / 3600
            print(f"  [{step}/{total}] {pct:.1f}% | loss={loss} | ETA: {remaining_h:.1f}h", flush=True)

        time.sleep(check_interval)


def run_quick_eval():
    """Run quick_eval.py with adapter."""
    print("\n" + "=" * 60)
    print("Phase 1: Quick Evaluation")
    print("=" * 60)

    # Add paths
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))

    from quick_eval import run_quick_eval as qe
    qe(adapter_path=str(OUTPUT_DIR))


def run_full_benchmarks():
    """Run benchmark_full.py with adapter."""
    print("\n" + "=" * 60)
    print("Phase 2: Full Benchmark Suite")
    print("=" * 60)

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(scripts_dir.parents[2] / "src"))

    from benchmark_full import run_benchmarks, BenchmarkConfig

    results = run_benchmarks(BenchmarkConfig(
        model_path=BASE_MODEL,
        adapter_path=str(OUTPUT_DIR),
        arc_limit=100,
        gsm8k_limit=50,
        output_dir=str(RESULTS_DIR),
    ))

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Benchmark':<25} {'Scratch 14M':>12} {'Qwen Base':>12} {'AVA Tuned':>12}")
    print("-" * 60)

    arc_acc = results["benchmarks"].get("arc_challenge", {}).get("accuracy", 0)
    gsm_acc = results["benchmarks"].get("gsm8k", {}).get("accuracy", 0)

    print(f"{'ARC-Challenge':<25} {'24%':>12} {'66%':>12} {f'{arc_acc*100:.1f}%':>12}")
    print(f"{'GSM8K':<25} {'0-2%':>12} {'4%':>12} {f'{gsm_acc*100:.1f}%':>12}")

    # Save summary
    summary = {
        "model": "Qwen3.5-2B-AVA-v1 (fast corpus, 5437 examples)",
        "training": "QLoRA r=16 alpha=32 lr=2e-4 seq=384 1epoch",
        "results": {
            "arc_challenge": f"{arc_acc*100:.1f}%",
            "gsm8k": f"{gsm_acc*100:.1f}%",
        },
        "comparison": {
            "scratch_14m": {"arc": "24%", "gsm8k": "0-2%"},
            "qwen_base": {"arc": "66%", "gsm8k": "4%"},
            "ava_tuned": {"arc": f"{arc_acc*100:.1f}%", "gsm8k": f"{gsm_acc*100:.1f}%"},
        },
    }
    summary_path = RESULTS_DIR / "ava_v1_fast_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    return results


if __name__ == "__main__":
    if "--no-wait" in sys.argv:
        # Skip waiting, run benchmarks immediately
        run_quick_eval()
        run_full_benchmarks()
    else:
        # Wait for training, then eval
        if wait_for_completion():
            # Small delay to ensure files are fully flushed
            time.sleep(10)
            run_quick_eval()
            run_full_benchmarks()
        else:
            print("Training failed. Skipping benchmarks.")
            sys.exit(1)
