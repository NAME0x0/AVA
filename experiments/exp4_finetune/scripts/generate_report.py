"""Generate a comprehensive training and evaluation report.

Reads all saved results files and produces a formatted markdown report.
Run after benchmarks complete to get a full picture of the experiment.

Usage:
    python scripts/generate_report.py
"""
import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("D:/AVA/experiments/exp4_finetune/results")
MODELS_DIR = Path("D:/AVA/experiments/exp4_finetune/models")
REPORT_PATH = Path("D:/AVA/experiments/exp4_finetune/RESULTS_REPORT.md")


def load_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def format_pct(val):
    if isinstance(val, float) and val <= 1.0:
        return f"{val*100:.1f}%"
    if isinstance(val, str) and "%" in val:
        return val
    return str(val)


def generate_report():
    report = []
    report.append(f"# AVA Experiment 4: Training & Evaluation Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Training results
    report.append("## Training Summary\n")

    # Check for training reports
    for model_dir in sorted(MODELS_DIR.glob("Qwen3.5-2B-AVA-*")):
        train_report = model_dir / "training_report.json"
        if train_report.exists():
            data = load_json(train_report)
            report.append(f"### {model_dir.name}")
            report.append(f"- **Corpus**: {Path(data.get('corpus', '')).name}")
            report.append(f"- **Examples**: {data.get('train_examples', '?')}")
            report.append(f"- **Final loss**: {data.get('train_loss', '?'):.4f}" if isinstance(data.get('train_loss'), (int, float)) else f"- **Final loss**: {data.get('train_loss', '?')}")
            report.append(f"- **Time**: {data.get('elapsed_minutes', '?')} minutes")
            report.append(f"- **LoRA rank**: {data.get('lora_r', '?')}")
            report.append(f"- **Learning rate**: {data.get('learning_rate', '?')}")
            report.append(f"- **GPU memory**: {data.get('gpu_memory_gb', '?')} GB")
            report.append("")

    # Benchmark results
    report.append("## Benchmark Results\n")

    # Collect all benchmark files
    bench_files = sorted(RESULTS_DIR.glob("benchmark_*.json"))
    for bf in bench_files:
        data = load_json(bf)
        if not data:
            continue
        model = data.get("model", bf.stem)
        report.append(f"### {model}")
        benchmarks = data.get("benchmarks", {})
        for bench_name, bench_data in benchmarks.items():
            acc = bench_data.get("accuracy", 0)
            correct = bench_data.get("correct", 0)
            total = bench_data.get("total", 0)
            elapsed = bench_data.get("elapsed_seconds", 0)
            report.append(f"- **{bench_name}**: {correct}/{total} = {acc*100:.1f}% ({elapsed:.0f}s)")
        report.append("")

    # Agentic results
    agentic_files = sorted(RESULTS_DIR.glob("agentic_gsm8k_*.json"))
    if agentic_files:
        report.append("## Agentic Benchmark (GSM8K with Tools)\n")
        for af in agentic_files:
            data = load_json(af)
            if not data:
                continue
            report.append(f"### {data.get('model', af.stem)}")
            report.append(f"- **Raw accuracy**: {data.get('raw_accuracy', 0)*100:.1f}%")
            report.append(f"- **Agentic accuracy**: {data.get('agentic_accuracy', 0)*100:.1f}%")
            report.append(f"- **Delta**: {data.get('delta_pp', 0):+.1f}pp")
            report.append(f"- **Tool calls**: {data.get('tools_used', 0)} problems used tools")
            report.append(f"- **Tool helped**: {data.get('tool_helped', 0)} (wrong→right)")
            report.append(f"- **Tool hurt**: {data.get('tool_hurt', 0)} (right→wrong)")
            report.append("")

    # Self-consistency results
    sc_files = sorted(RESULTS_DIR.glob("sc*_gsm8k_*.json"))
    if sc_files:
        report.append("## Self-Consistency Benchmark\n")
        for sf in sc_files:
            data = load_json(sf)
            if not data:
                continue
            report.append(f"### {data.get('model', sf.stem)}")
            report.append(f"- **K samples**: {data.get('k_samples', '?')}")
            report.append(f"- **Greedy accuracy**: {data.get('greedy_accuracy', 0)*100:.1f}%")
            report.append(f"- **SC accuracy**: {data.get('sc_accuracy', 0)*100:.1f}%")
            report.append(f"- **Delta**: {data.get('delta_pp', 0):+.1f}pp")
            report.append("")

    # Checkpoint comparison
    ckpt_file = RESULTS_DIR / "checkpoint_comparison.json"
    if ckpt_file.exists():
        data = load_json(ckpt_file)
        if data:
            report.append("## Checkpoint Comparison\n")
            report.append(f"| {'Model':<25} | {'ARC':>12} | {'GSM8K':>12} |")
            report.append(f"|{'-'*27}|{'-'*14}|{'-'*14}|")
            for r in data:
                report.append(f"| {r['name']:<25} | {r['arc']:>12} | {r['gsm8k']:>12} |")
            report.append("")

    # Pipeline state
    pipeline_state = RESULTS_DIR / "pipeline_state.json"
    if pipeline_state.exists():
        data = load_json(pipeline_state)
        report.append("## Pipeline Status\n")
        report.append(f"- **Phase**: {data.get('phase', '?')}")
        if data.get('v1_arc'):
            report.append(f"- **V1 ARC**: {data['v1_arc']*100:.1f}%")
        if data.get('v1_gsm8k'):
            report.append(f"- **V1 GSM8K**: {data['v1_gsm8k']*100:.1f}%")
        if data.get('base_arc'):
            report.append(f"- **Base ARC**: {data['base_arc']*100:.1f}%")
        if data.get('base_gsm8k'):
            report.append(f"- **Base GSM8K**: {data['base_gsm8k']*100:.1f}%")
        if data.get('v2_proceed') is not None:
            report.append(f"- **V2 proceed**: {'Yes' if data['v2_proceed'] else 'No'}")
        report.append("")

    # Comparison table
    report.append("## Summary Comparison\n")
    report.append(f"| {'Benchmark':<20} | {'Scratch 14M':>12} | {'Qwen Base':>12} | {'AVA SFT':>12} | {'AVA+Tools':>12} |")
    report.append(f"|{'-'*22}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|")

    # Fill in what we know
    pipeline = load_json(pipeline_state) if pipeline_state.exists() else {}
    base_arc = pipeline.get('base_arc', 0.66)
    base_gsm = pipeline.get('base_gsm8k', 0.04)
    v1_arc = pipeline.get('v1_arc', '?')
    v1_gsm = pipeline.get('v1_gsm8k', '?')

    arc_sft = f"{v1_arc*100:.1f}%" if isinstance(v1_arc, float) else "pending"
    gsm_sft = f"{v1_gsm*100:.1f}%" if isinstance(v1_gsm, float) else "pending"

    # Check for agentic results
    agentic_gsm = "pending"
    for af in RESULTS_DIR.glob("agentic_gsm8k_*.json"):
        data = load_json(af)
        if data:
            agentic_gsm = f"{data.get('agentic_accuracy', 0)*100:.1f}%"

    report.append(f"| {'ARC-Challenge':<20} | {'24%':>12} | {f'{base_arc*100:.1f}%':>12} | {arc_sft:>12} | {'N/A':>12} |")
    report.append(f"| {'GSM8K':<20} | {'0-2%':>12} | {f'{base_gsm*100:.1f}%':>12} | {gsm_sft:>12} | {agentic_gsm:>12} |")
    report.append("")

    # Write report
    report_text = "\n".join(report)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Report saved to: {REPORT_PATH}")
    print("\n" + report_text)

    return report_text


if __name__ == "__main__":
    generate_report()
