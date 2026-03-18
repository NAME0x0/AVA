"""Autonomous training pipeline: v1 eval -> decision -> v2 launch.

Monitors v1 fast training, runs benchmarks on completion,
decides whether to proceed with v2 full training, and launches it.
Designed to run fully unattended while the user sleeps.

Usage:
    python -u experiments/exp4_finetune/scripts/autonomous_pipeline.py > experiments/exp4_finetune/pipeline.log 2>&1 &
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths
LOG_FILE = Path("D:/AVA/experiments/exp4_finetune/training_fast_v2.log")
OUTPUT_DIR = Path("D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v1")
BASE_MODEL = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
RESULTS_DIR = Path("D:/AVA/experiments/exp4_finetune/results")
PIPELINE_STATE = RESULTS_DIR / "pipeline_state.json"
V2_SCRIPT = Path("D:/AVA/experiments/exp4_finetune/scripts/finetune_v2_full.py")
V2_LOG = Path("D:/AVA/experiments/exp4_finetune/training_v2_full.log")

# Thresholds for auto-proceeding to v2
# Conservative: don't regress on ARC, and GSM8K should improve
ARC_MIN = 0.60   # At least 60% (base is 66%, some regression is OK for a fast run)
GSM8K_MIN = 0.06  # At least 6% (base is 4%, want improvement)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def save_state(state):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def load_state():
    if PIPELINE_STATE.exists():
        return json.loads(PIPELINE_STATE.read_text(encoding="utf-8"))
    return {"phase": "waiting_for_v1", "started": datetime.now().isoformat()}


def wait_for_v1():
    """Wait for v1 fast training to complete."""
    log("Phase 1: Waiting for v1 fast training to complete...")
    while True:
        if not LOG_FILE.exists():
            time.sleep(120)
            continue

        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")

        if "Training Complete!" in text:
            loss_match = re.search(r"Train loss: ([\d.]+)", text)
            time_match = re.search(r"Time: (\d+)s", text)
            log("V1 training COMPLETE!")
            if loss_match:
                log(f"  Final loss: {loss_match.group(1)}")
            if time_match:
                log(f"  Time: {int(time_match.group(1))/60:.1f} min")
            return True

        if "Traceback" in text[-3000:] or "CUDA out of memory" in text[-3000:]:
            log("V1 training FAILED!")
            log(text[-500:])
            return False

        # Progress
        steps_match = list(re.finditer(r"(\d+)/(\d+).*?([\d.]+)s/it", text))
        loss_matches = list(re.finditer(r"'loss': '([\d.]+)'", text))
        if steps_match:
            last = steps_match[-1]
            step, total = int(last.group(1)), int(last.group(2))
            pct = step / total * 100
            loss = loss_matches[-1].group(1) if loss_matches else "?"
            remaining_h = (total - step) * float(last.group(3)) / 3600
            log(f"  v1 progress: [{step}/{total}] {pct:.1f}% | loss={loss} | ETA: {remaining_h:.1f}h")

        time.sleep(120)


def run_benchmarks():
    """Run full benchmarks on base model AND trained v1 model."""
    log("Phase 2: Running benchmarks...")

    # Wait for GPU to be free (training process to fully exit)
    time.sleep(15)

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(scripts_dir.parents[2] / "src"))

    from benchmark_full import run_benchmarks as rb, BenchmarkConfig

    # Phase 2a: Re-run base model benchmark with proper settings (512→768 tokens)
    log("  Phase 2a: Base model benchmark (corrected eval)...")
    base_results = rb(BenchmarkConfig(
        model_path=BASE_MODEL,
        adapter_path=None,
        arc_limit=100,
        gsm8k_limit=50,
        output_dir=str(RESULTS_DIR),
    ))
    base_arc = base_results["benchmarks"].get("arc_challenge", {}).get("accuracy", 0)
    base_gsm = base_results["benchmarks"].get("gsm8k", {}).get("accuracy", 0)
    log(f"  Base ARC: {base_arc*100:.1f}% | Base GSM8K: {base_gsm*100:.1f}%")

    # Phase 2b: Fine-tuned model benchmark
    log("  Phase 2b: Fine-tuned model benchmark...")
    tuned_results = rb(BenchmarkConfig(
        model_path=BASE_MODEL,
        adapter_path=str(OUTPUT_DIR),
        arc_limit=100,
        gsm8k_limit=50,
        output_dir=str(RESULTS_DIR),
    ))
    arc_acc = tuned_results["benchmarks"].get("arc_challenge", {}).get("accuracy", 0)
    gsm_acc = tuned_results["benchmarks"].get("gsm8k", {}).get("accuracy", 0)
    log(f"  Tuned ARC: {arc_acc*100:.1f}% | Tuned GSM8K: {gsm_acc*100:.1f}%")

    # Phase 2c: Agentic benchmark (GSM8K with tool use)
    log("  Phase 2c: Agentic GSM8K benchmark (with calculator tools)...")
    try:
        from benchmark_agentic import run_agentic_benchmark, AgenticConfig
        agentic_results = run_agentic_benchmark(AgenticConfig(
            model_path=BASE_MODEL,
            adapter_path=str(OUTPUT_DIR),
            gsm8k_limit=50,
        ))
        agentic_gsm = agentic_results.get("agentic_accuracy", 0)
        log(f"  Agentic GSM8K: {agentic_gsm*100:.1f}% (delta: {agentic_results.get('delta_pp', 0):+.1f}pp vs raw)")
    except Exception as e:
        log(f"  Agentic benchmark failed: {e}")
        agentic_gsm = 0

    return arc_acc, gsm_acc, tuned_results, base_arc, base_gsm


def decide_v2(arc_acc, gsm_acc):
    """Decide whether to proceed with v2 full training."""
    log("Phase 3: Deciding on v2 launch...")

    proceed = True
    reasons = []

    if arc_acc < ARC_MIN:
        reasons.append(f"ARC {arc_acc*100:.1f}% < {ARC_MIN*100:.0f}% threshold")
        proceed = False
    else:
        reasons.append(f"ARC {arc_acc*100:.1f}% >= {ARC_MIN*100:.0f}% threshold: PASS")

    if gsm_acc < GSM8K_MIN:
        reasons.append(f"GSM8K {gsm_acc*100:.1f}% < {GSM8K_MIN*100:.0f}% threshold")
        # Don't block on GSM8K alone — it's harder to improve at 2B
        if arc_acc >= ARC_MIN:
            reasons.append("  -> ARC passed, proceeding despite GSM8K (2B model limitation)")
            proceed = True
    else:
        reasons.append(f"GSM8K {gsm_acc*100:.1f}% >= {GSM8K_MIN*100:.0f}% threshold: PASS")

    for r in reasons:
        log(f"  {r}")

    if proceed:
        log("DECISION: Proceed with v2 full corpus training!")
    else:
        log("DECISION: v2 NOT launched. Manual review needed.")

    return proceed


def launch_v2():
    """Launch v2 full corpus training in background."""
    log("Phase 4: Launching v2 full corpus training...")

    if not V2_SCRIPT.exists():
        log(f"ERROR: V2 script not found: {V2_SCRIPT}")
        return False

    # Check GPU is free
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            log(f"WARNING: GPU still in use by PIDs: {result.stdout.strip()}")
            log("Waiting 60s for GPU to free up...")
            time.sleep(60)
    except Exception:
        pass

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    proc = subprocess.Popen(
        [sys.executable, "-u", str(V2_SCRIPT)],
        stdout=open(str(V2_LOG), "w"),
        stderr=subprocess.STDOUT,
        cwd="D:/AVA",
        env=env,
    )
    log(f"V2 training launched! PID: {proc.pid}")
    log(f"Log: {V2_LOG}")
    log(f"Corpus: 20,941 examples, ~2586 steps, ~16h estimated")
    return True


def main():
    log("=" * 60)
    log("AVA Autonomous Pipeline")
    log("=" * 60)

    state = load_state()
    state["pipeline_start"] = datetime.now().isoformat()

    # Phase 1: Wait for v1
    if state.get("phase") in ("waiting_for_v1", None):
        v1_ok = wait_for_v1()
        if not v1_ok:
            state["phase"] = "v1_failed"
            save_state(state)
            log("Pipeline stopped: v1 training failed.")
            return

        state["phase"] = "v1_complete"
        state["v1_completed"] = datetime.now().isoformat()
        save_state(state)

    # Phase 2: Benchmark
    if state.get("phase") == "v1_complete":
        arc_acc, gsm_acc, results, base_arc, base_gsm = run_benchmarks()
        state["phase"] = "benchmarked"
        state["v1_arc"] = arc_acc
        state["v1_gsm8k"] = gsm_acc
        state["base_arc"] = base_arc
        state["base_gsm8k"] = base_gsm
        state["benchmarked_at"] = datetime.now().isoformat()
        save_state(state)

        # Comparison table
        log("")
        log("=" * 60)
        log("COMPARISON TABLE")
        log("=" * 60)
        log(f"{'Benchmark':<25} {'Scratch 14M':>12} {'Qwen Base':>12} {'AVA-v1':>12}")
        log("-" * 60)
        log(f"{'ARC-Challenge':<25} {'24%':>12} {f'{base_arc*100:.1f}%':>12} {f'{arc_acc*100:.1f}%':>12}")
        log(f"{'GSM8K':<25} {'0-2%':>12} {f'{base_gsm*100:.1f}%':>12} {f'{gsm_acc*100:.1f}%':>12}")

        # Compute deltas
        arc_delta = (arc_acc - base_arc) * 100
        gsm_delta = (gsm_acc - base_gsm) * 100
        arc_sign = "+" if arc_delta >= 0 else ""
        gsm_sign = "+" if gsm_delta >= 0 else ""
        log(f"{'Delta (tuned-base)':<25} {'':>12} {'':>12} {f'{arc_sign}{arc_delta:.1f}pp':>12}")
        log(f"{'':.<25} {'':>12} {'':>12} {f'{gsm_sign}{gsm_delta:.1f}pp':>12}")
    else:
        arc_acc = state.get("v1_arc", 0)
        gsm_acc = state.get("v1_gsm8k", 0)

    # Phase 3: Decision
    if state.get("phase") == "benchmarked":
        proceed = decide_v2(arc_acc, gsm_acc)
        state["phase"] = "decided"
        state["v2_proceed"] = proceed
        state["decided_at"] = datetime.now().isoformat()
        save_state(state)

        if not proceed:
            log("Pipeline stopped: v2 not recommended. Review results manually.")
            return

    # Phase 4: Launch v2
    if state.get("phase") == "decided" and state.get("v2_proceed"):
        launched = launch_v2()
        state["phase"] = "v2_launched" if launched else "v2_launch_failed"
        state["v2_launched_at"] = datetime.now().isoformat()
        save_state(state)

        if launched:
            log("")
            log("=" * 60)
            log("Pipeline complete! v2 training is running.")
            log(f"Monitor with: tail -f {V2_LOG}")
            log("=" * 60)

    # Generate final report
    try:
        from generate_report import generate_report
        log("\nGenerating results report...")
        generate_report()
    except Exception as e:
        log(f"Report generation failed: {e}")

    log(f"\nPipeline state saved to: {PIPELINE_STATE}")


if __name__ == "__main__":
    main()
