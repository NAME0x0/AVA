"""Phase-1 measured baseline: throughput x speculative-mode x clock profile.

Produces the numbers the roadmap needs before any training night:
  - tokens/s (prompt + generation) for plain vs ngram-simple vs draft-mtp
  - mean/max GPU power + temperature per case (sampled 1 Hz during the run)
  - each case optionally under clock caps (unlocked / 0.70 / 0.50 of max SM)
    -- clock cases are skipped automatically when not elevated.

Usage (from experiments/exp6_v3):
  python scripts/phase1_baseline.py                # plain + ngram on the preview GGUF
  python scripts/phase1_baseline.py --mtp          # adds draft-mtp on the MTP donor GGUF
  python scripts/phase1_baseline.py --clocks       # adds clock-profile sweep (admin)

Output: table on stdout + JSONL at D:/AVA/ratchet/logs/phase1_baseline_<ts>.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ratchet.power_governor import lock_clocks, read_sensors, reset_clocks  # noqa: E402

ROOT = Path("D:/AVA")
BINS = ROOT / "tools" / "llamacpp-bin"
GGUF_PREVIEW = ROOT / "exports" / "v30-preview" / "ava-v30-preview-q4_k_m.gguf"
GGUF_MTP = ROOT / "models" / "gguf" / "Qwen3.5-4B-Q4_K_M.gguf"   # unsloth MTP build
LOG_DIR = ROOT / "ratchet" / "logs"

PROMPT = (
    "Write a Python function `parse_csv(text)` that parses CSV text with quoted "
    "fields and escaped quotes into a list of rows, plus 3 usage examples."
)


def _sample_power(stop: threading.Event, samples: list) -> None:
    while not stop.is_set():
        r = read_sensors()
        if r.gpu_power_w is not None:
            samples.append((r.gpu_power_w, r.gpu_temp_c))
        stop.wait(1.0)


def run_case(label: str, gguf: Path, spec_type: str | None, n_tokens: int = 256) -> dict:
    cmd = [
        str(BINS / "llama-cli"), "-m", str(gguf), "-ngl", "99", "-c", "4096",
        "--temp", "0", "-n", str(n_tokens), "-st", "-cnv",
        "--chat-template-kwargs", '{"enable_thinking": false}',
        "-p", PROMPT,
    ]
    if spec_type:
        cmd += ["--spec-type", spec_type]
    stop = threading.Event()
    samples: list = []
    thr = threading.Thread(target=_sample_power, args=(stop, samples), daemon=True)
    thr.start()
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall = time.monotonic() - t0
    stop.set()
    thr.join(timeout=3)

    # this build prints "[ Prompt: 132.3 t/s | Generation: 42.6 t/s ]" on STDOUT;
    # older builds print "... tokens per second" perf lines on stderr - try both
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = re.search(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]", text)
    if m:
        prompt_tps, gen_tps = float(m.group(1)), float(m.group(2))
    else:
        def tps(pattern: str) -> float | None:
            mm = re.search(pattern + r".*?([\d.]+)\s+tokens per second", text)
            return float(mm.group(1)) if mm else None
        gen_tps = tps(r"eval time")
        prompt_tps = tps(r"prompt eval time")
    err = proc.stderr or ""
    powers = [p for p, _ in samples]
    temps = [t for _, t in samples if t is not None]
    row = {
        "case": label, "gguf": gguf.name, "spec_type": spec_type or "none",
        "gen_tps": gen_tps, "prompt_tps": prompt_tps, "wall_s": round(wall, 1),
        "power_mean_w": round(sum(powers) / len(powers), 1) if powers else None,
        "power_max_w": round(max(powers), 1) if powers else None,
        "temp_max_c": max(temps) if temps else None,
        "tokens_per_joule": (round(gen_tps / (sum(powers) / len(powers)), 3)
                             if gen_tps and powers else None),
        "exit": proc.returncode,
    }
    if proc.returncode != 0:
        row["stderr_tail"] = err[-400:]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtp", action="store_true", help="include draft-mtp case (MTP GGUF)")
    ap.add_argument("--clocks", action="store_true", help="sweep clock profiles (admin)")
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    cases: list[tuple[str, Path, str | None]] = [
        ("plain", GGUF_PREVIEW, None),
        ("ngram", GGUF_PREVIEW, "ngram-simple"),
    ]
    if args.mtp:
        if GGUF_MTP.exists():
            cases += [("mtp-plain", GGUF_MTP, None), ("mtp-spec", GGUF_MTP, "draft-mtp")]
        else:
            print(f"[bench] MTP GGUF missing at {GGUF_MTP} - skipping mtp cases")

    profiles: list[tuple[str, float | None]] = [("unlocked", None)]
    if args.clocks:
        if lock_clocks(1.0):
            reset_clocks()
            profiles += [("cap70", 0.70), ("cap50", 0.50)]
        else:
            print("[bench] clock lock DENIED (not elevated) - unlocked profile only")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"phase1_baseline_{int(time.time())}.jsonl"
    rows: list[dict] = []
    try:
        for prof_name, frac in profiles:
            if frac is not None:
                lock_clocks(frac)
                time.sleep(2)
            for label, gguf, spec in cases:
                if not gguf.exists():
                    print(f"[bench] missing {gguf} - skip {label}")
                    continue
                print(f"[bench] {prof_name}/{label} ...", flush=True)
                row = run_case(f"{prof_name}/{label}", gguf, spec, args.n)
                rows.append(row)
                with open(out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"    gen {row['gen_tps']} t/s | prompt {row['prompt_tps']} t/s | "
                      f"power {row['power_mean_w']}W mean/{row['power_max_w']}W max | "
                      f"{row['tokens_per_joule']} tok/J | exit {row['exit']}")
    finally:
        reset_clocks()

    print(f"\n[bench] {len(rows)} rows -> {out_path}")
    ok = [r for r in rows if r["gen_tps"]]
    if ok:
        best = max(ok, key=lambda r: r["gen_tps"])
        eff = max((r for r in ok if r["tokens_per_joule"]),
                  key=lambda r: r["tokens_per_joule"], default=None)
        print(f"[bench] fastest: {best['case']} @ {best['gen_tps']} t/s")
        if eff:
            print(f"[bench] most efficient: {eff['case']} @ {eff['tokens_per_joule']} tok/J")


if __name__ == "__main__":
    main()
