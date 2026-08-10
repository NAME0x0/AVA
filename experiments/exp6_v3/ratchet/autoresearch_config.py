"""Unattended autoresearch over INFERENCE MECHANISM (not weights, not layers).

Karpathy's autoresearch edits train.py and scores val_bpb. This searches the
llama.cpp execution configuration and scores throughput: no training, seconds
per experiment, weights never change — so a win is a mechanism win, and it
transfers to a bigger sibling of the same architecture family.

Design borrowed from autoresearch: the loop may change the CONFIG but can never
touch the MEASUREMENT or the guards. That separation is what stops a search from
making the test easier instead of the engine faster.

SUBSTRATE (verified against build b10059, 2026-08-10)
  llama-bench       runtime knobs. Native -r repetitions with stddev, --delay
                    for thermal spacing, -o json (no regex scraping). Strictly
                    better than parsing llama-cli — except it has NO --spec-type.
  llama-cli         the spec-decode axis only, because llama-bench cannot do it
                    and spec-decode is the largest lever measured so far
                    (DFlash +46% over plain).
  llama-perplexity  quality guard for knobs that legitimately change output.

KNOBS EXCLUDED ON PURPOSE (verified absent or inert on this build — including
them would pad the space with knobs that cannot move the number, which makes
plateau detection worse, not better):
  --cache-reuse, -cram   llama-server only; absent from cli and bench
  --defrag-thold         present but DEPRECATED
  -sm, -ts, -mg          multi-GPU; this machine has one card
  -ncmoe                 MoE only; both our donors are dense
  --numa                 single socket

GUARDS (each earned by a failure we actually hit)
  LOCKFILE   refuses to start while another instance lives. Two concurrent
             instances competing for one GPU destroyed the 2026-08-09 run:
             baseline 420.8 -> "best" 330.6, i.e. the search ended below where
             it started, because every measurement was contaminated.
  EXCLUSIVE  aborts if another process already holds VRAM; re-checked before
             every experiment.
  VALIDATE   every knob VALUE is dry-run at startup; anything the binary
             rejects is dropped, never silently measured as "slow".
  NOISE      llama-bench stddev, with acceptance on the standard error of the
             DIFFERENCE OF MEANS (sd/sqrt(n)). Raw sd asks "would one run beat
             one run" — a ~12% bar that would reject every realistic win.
  QUALITY    tier A (pure runtime) must reproduce the reference output token
             for token; tier B/C (quality-affecting) is checked by perplexity.
  THERMAL    --delay between tests plus a temperature ceiling that pauses.
  DRIFT      the incumbent is re-measured periodically.

Perplexity is the fast in-loop FILTER, not the verdict: we established in
2026-08 that perplexity misses capability damage (INT6 improving perplexity
while half the SAE features died). The winning config must be re-validated on a
real task eval before adoption.

  python -m ratchet.autoresearch_config --model <gguf> --hours 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

BINDIR = Path("D:/AVA/tools/llamacpp-bin")
BENCH, CLI = BINDIR / "llama-bench", BINDIR / "llama-cli"
LOCK = Path("D:/AVA/ratchet/logs/autoresearch.lock")
PROMPT = ("Write a Python function `parse_csv(text)` that parses CSV text with "
          "quoted fields and escaped quotes into a list of rows, plus 3 examples.")

# --------------------------------------------------------------------- the box
# TIER A — pure runtime. Output must stay identical; if it moves, something is
# wrong. Measured by llama-bench.
TIER_A: dict[str, list] = {
    "b":    [256, 512, 1024, 2048],        # logical batch
    "ub":   [64, 128, 256, 512],           # physical micro-batch (prefill lever)
    "t":    [4, 6, 8, 12, 16],             # threads
    "poll": [0, 25, 50, 100],              # busy-wait level
    "nopo": [0, 1],                        # no-op-offload
}
# TIER B — legitimately changes output; guarded by perplexity, not token match.
TIER_B: dict[str, list] = {
    "ctk":  ["f16", "q8_0", "q5_1", "q4_0", "iq4_nl"],   # verified valid types
    "ctv":  ["f16", "q8_0", "q5_1", "q4_0", "iq4_nl"],
    "fa":   ["on", "off", "auto"],
    "nkvo": [0, 1],                        # keep KV off-GPU (frees VRAM)
}
# TIER C — hand-curated: -ot takes regexes, which cannot be enumerated. On a
# 4 GB card the useful move is pushing selected tensors to CPU to free VRAM.
TIER_C: dict[str, list] = {
    "ot": ["", r"\.ffn_(gate|up|down)\.weight=CPU", r"\.ffn_down\.weight=CPU",
           r"blk\.(0|1|2)\..*=CPU", r"\.attn_(k|v)\.weight=CPU"],
}
# SPEC — llama-cli only (llama-bench has no --spec-type).
TIER_SPEC: dict[str, list] = {
    "spec": ["none", "ngram-simple", "ngram-cache", "ngram-mod"],
}
SPACE = {**TIER_A, **TIER_B, **TIER_C, **TIER_SPEC}
REFERENCE = {"b": 2048, "ub": 512, "t": 8, "poll": 50, "nopo": 0,
             "ctk": "f16", "ctv": "f16", "fa": "auto", "nkvo": 0,
             "ot": "", "spec": "none"}
QUALITY_KNOBS = set(TIER_B) | set(TIER_C)          # need the perplexity guard
_CLI_TPS = re.compile(r"Generation:\s*([\d.]+)\s*t/s")

# Calibration corpus for the perplexity guard. MUST be diverse real text.
# A repeated block is memorised after one pass (ppl ~1.3) and the guard goes
# blind: measured 2026-08-10, q4_0 KV scored -2.25% "better" than f16 on a
# repetitive corpus, so every destructive quality knob would have passed.
_CALIB_FALLBACK = """Machine learning models are trained on large corpora in order to predict the
next token given the preceding context. The transformer architecture replaced
recurrent networks for most sequence tasks because attention parallelises across
positions, whereas recurrence forces sequential computation.

def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

The Treaty of Westphalia in 1648 ended the Thirty Years War and established the
principle of territorial sovereignty that still underpins international law.
Photosynthesis converts light energy into chemical energy stored as glucose,
occurring in the chloroplasts where chlorophyll absorbs primarily red and blue
wavelengths while reflecting green.

SELECT c.name, COUNT(o.id) AS order_count
FROM customers c LEFT JOIN orders o ON o.customer_id = c.id
WHERE o.created_at >= '2026-01-01' GROUP BY c.name HAVING COUNT(o.id) > 3;

In thermodynamics the second law states that the entropy of an isolated system
never decreases. Heat flows spontaneously from hotter to colder bodies, and no
process can convert heat entirely into work without other effect.
"""


def build_calibration(path: Path, target_chars: int = 60000) -> Path:
    """Diverse real text for the perplexity guard, cached on disk.

    Prefers WikiText (the standard perplexity corpus). Falls back to a varied
    inline sample offline. Never repeats a short block -- see the note above.
    """
    if path.exists() and path.stat().st_size > 20000:
        return path
    text = ""
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="test", streaming=True)
        parts = []
        for rec in ds:
            t = rec["text"].strip()
            if len(t) > 200:
                parts.append(t)
            if sum(map(len, parts)) > target_chars:
                break
        text = "\n\n".join(parts)
    except Exception:  # noqa: BLE001 - offline: fall back, but stay diverse
        text = ""
    if len(text) < 20000:
        # still diverse: a multi-topic block, lightly padded — never a short
        # repeated string, which is what blinded the guard in the first place
        text = (_CALIB_FALLBACK + "\n") * 3 + text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------- guards
class Lock:
    """Refuse to run concurrently — the exact bug that destroyed 2026-08-09."""

    def __enter__(self):
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        if LOCK.exists():
            raise SystemExit(
                f"another autoresearch instance holds {LOCK} "
                f"(pid {LOCK.read_text().strip()}).\n"
                f"If that process is dead, delete the file and retry.")
        LOCK.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        LOCK.unlink(missing_ok=True)


def gpu_state() -> tuple[int, int]:
    """(vram_used_mib, temp_c); (-1, -1) when unreadable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=15).stdout.strip().splitlines()[0]
        a, b = (x.strip() for x in out.split(","))
        return int(float(a)), int(float(b))
    except Exception:  # noqa: BLE001
        return -1, -1


def require_exclusive_gpu(budget_mib: int) -> None:
    used, _ = gpu_state()
    if used > budget_mib:
        raise SystemExit(
            f"GPU already holds {used} MiB (> {budget_mib}). Another job is "
            f"running and measurements would be contaminated. Aborting.")


def wait_thermal(ceiling_c: int, delay_s: float) -> None:
    for _ in range(60):
        _, temp = gpu_state()
        if temp < 0 or temp <= ceiling_c:
            break
        time.sleep(30)
    if delay_s:
        time.sleep(delay_s)


# ------------------------------------------------------------ measurement
def bench_args(cfg: dict) -> list[str]:
    a = ["-b", str(cfg["b"]), "-ub", str(cfg["ub"]), "-t", str(cfg["t"]),
         "--poll", str(cfg["poll"]), "-nopo", str(cfg["nopo"]),
         "-ctk", cfg["ctk"], "-ctv", cfg["ctv"], "-fa", cfg["fa"],
         "-nkvo", str(cfg["nkvo"]), "-ngl", "99"]
    if cfg.get("ot"):
        a += ["-ot", cfg["ot"]]
    return a


def run_bench(model: str, cfg: dict, repeats: int, n_gen: int, n_prompt: int,
              delay: float) -> dict | None:
    """Throughput via llama-bench: native repeats + stddev, JSON output."""
    cmd = [str(BENCH), "-m", model, *bench_args(cfg), "-p", str(n_prompt),
           "-n", str(n_gen), "-r", str(repeats), "--delay", str(delay),
           "-o", "json"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        rows = json.loads(p.stdout)
    except Exception:  # noqa: BLE001 - invalid combo, crash, or unparseable
        return None
    out: dict = {"build": None}
    for r in rows:
        out["build"] = r.get("build_number")
        if r.get("n_gen"):
            out["tg"], out["tg_sd"] = r["avg_ts"], r.get("stddev_ts", 0.0)
        elif r.get("n_prompt"):
            out["pp"], out["pp_sd"] = r["avg_ts"], r.get("stddev_ts", 0.0)
    return out if "tg" in out else None


def run_cli_spec(model: str, cfg: dict, n_gen: int) -> tuple[float | None, str]:
    """Spec-decode axis, and the text used for the tier-A agreement guard."""
    cmd = [str(CLI), "-m", model, "-ngl", "99", "-c", "2048",
           "-b", str(cfg["b"]), "-ub", str(cfg["ub"]), "-t", str(cfg["t"]),
           "-ctk", cfg["ctk"], "-ctv", cfg["ctv"], "-fa", cfg["fa"],
           "--temp", "0", "-n", str(n_gen), "-st", "-cnv", "-p", PROMPT]
    if cfg.get("spec", "none") != "none":
        cmd += ["--spec-type", cfg["spec"]]
    if cfg.get("ot"):
        cmd += ["-ot", cfg["ot"]]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None, ""
    m = _CLI_TPS.search((p.stdout or "") + (p.stderr or ""))
    return (float(m.group(1)) if m else None), (p.stdout or "")


def perplexity(model: str, cfg: dict, text_file: Path) -> float | None:
    """Quality guard for knobs that legitimately change output.

    The FAST filter, not the verdict: perplexity is known to miss capability
    damage, so the winning config gets re-validated on a real task eval.
    """
    cmd = [str(BINDIR / "llama-perplexity"), "-m", model, "-f", str(text_file),
           "-ngl", "99", "-ctk", cfg["ctk"], "-ctv", cfg["ctv"],
           "-fa", cfg["fa"], "-b", "512", "--chunks", "2"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    hits = re.findall(r"(?:Final estimate: PPL|\[\d+\])\s*=?\s*([\d.]+)",
                      (p.stdout or "") + (p.stderr or ""))
    return float(hits[-1]) if hits else None


def _toks(t: str) -> list[str]:
    return re.findall(r"\w+", t.lower())


def agreement(a: str, b: str, head: int = 120) -> float:
    ta, tb = _toks(a)[:head], _toks(b)[:head]
    if not ta:
        return 1.0
    return sum(1 for x, y in zip(ta, tb, strict=False) if x == y) / len(ta)


# ------------------------------------------------------------- validation
def validate_space(model: str) -> dict[str, list]:
    """Dry-run every knob VALUE once; drop whatever the binary rejects.

    Without this, an unsupported value is measured as "slow" rather than
    "unsupported", which quietly corrupts the entire search.
    """
    ok: dict[str, list] = {}
    for knob, values in SPACE.items():
        keep = []
        for v in values:
            cfg = dict(REFERENCE)
            cfg[knob] = v
            if knob == "spec":
                good = run_cli_spec(model, cfg, 8)[0] is not None
            else:
                good = run_bench(model, cfg, 1, 8, 32, 0) is not None
            print(f"  [validate] {knob}={v!r:<30} {'ok' if good else 'REJECTED'}",
                  flush=True)
            if good:
                keep.append(v)
        if keep:
            ok[knob] = keep
    return ok


# ----------------------------------------------------------------- search
def neighbour(cfg: dict, space: dict, rng: random.Random) -> dict:
    """One-knob mutation: every win stays attributable to a single change."""
    nxt = dict(cfg)
    k = rng.choice([k for k in space if len(space[k]) > 1])
    nxt[k] = rng.choice([x for x in space[k] if x != cfg[k]])
    return nxt


def evaluate(model: str, cfg: dict, args, ref: dict) -> dict:
    """Speed, plus whichever guard the changed knobs demand."""
    wait_thermal(args.temp_ceiling, args.delay)
    require_exclusive_gpu(args.vram_budget)

    if cfg.get("spec", "none") != "none":               # spec axis -> llama-cli
        tps, text = run_cli_spec(model, cfg, args.tokens)
        if tps is None:
            return {"ok": False, "why": "invalid/crash"}
        speed, sd = tps, ref.get("cli_sd", 0.0)
    else:
        b = run_bench(model, cfg, args.repeats, args.tokens,
                      args.prompt_tokens, args.delay)
        if b is None:
            return {"ok": False, "why": "invalid/crash"}
        speed, sd = b["tg"], b["tg_sd"]
        text = run_cli_spec(model, cfg, 64)[1] if args.check_text else ""

    changed = {k for k in cfg if cfg[k] != REFERENCE[k]}
    if changed & QUALITY_KNOBS:                         # tier B/C -> perplexity
        ppl = perplexity(model, cfg, Path(args.ppl_file))
        if ppl is None:
            return {"ok": False, "why": "ppl failed", "mean": speed, "sd": sd}
        drift = (ppl - ref["ppl"]) / ref["ppl"] if ref.get("ppl") else 0.0
        if drift > args.max_ppl_drift:
            return {"ok": False, "why": f"ppl +{drift:.1%}", "mean": speed,
                    "sd": sd}
        return {"ok": True, "mean": speed, "sd": sd, "ppl": round(ppl, 4)}

    if args.check_text and text:                        # tier A -> same output
        ag = agreement(ref.get("text", ""), text)
        if ag < args.min_agreement:
            return {"ok": False, "why": f"output drift {ag:.2f}", "mean": speed,
                    "sd": sd}
    return {"ok": True, "mean": speed, "sd": sd}


def search(model: str, args) -> dict:
    rng = random.Random(args.seed)
    sp = Path(args.state)
    deadline = time.monotonic() + args.hours * 3600
    state = json.loads(sp.read_text()) if sp.exists() else None

    if state is None:
        require_exclusive_gpu(args.vram_budget)
        ppl_file = build_calibration(Path(args.ppl_file))
        print("[auto] validating knob values against the binary...", flush=True)
        space = validate_space(model)
        print("[auto] measuring reference...", flush=True)
        b = run_bench(model, REFERENCE, args.repeats, args.tokens,
                      args.prompt_tokens, args.delay)
        if b is None:
            raise SystemExit("reference bench failed — cannot proceed")
        text = run_cli_spec(model, REFERENCE, 64)[1]
        ppl = perplexity(model, REFERENCE, ppl_file)
        used, temp = gpu_state()
        state = {"model": model, "space": space,
                 "ref": {"text": text, "ppl": ppl, "cli_sd": 0.0},
                 "best_cfg": REFERENCE, "best": b["tg"], "best_sd": b["tg_sd"],
                 "baseline": b["tg"], "build": b.get("build"),
                 "gpu_at_start": [used, temp], "history": [], "stale": 0, "n": 0}
        print(f"[auto] reference {b['tg']:.1f} ±{b['tg_sd']:.2f} t/s | "
              f"ppl {ppl} | build {b.get('build')}", flush=True)
        sp.write_text(json.dumps(state, indent=2))

    space, ref = state["space"], state["ref"]
    while time.monotonic() < deadline and state["stale"] < args.patience:
        cand = neighbour(state["best_cfg"], space, rng)
        if any(h["cfg"] == cand for h in state["history"]):
            continue
        res = evaluate(model, cand, args, ref)
        state["n"] += 1
        accepted = False
        if res["ok"]:
            # SEM of the DIFFERENCE OF MEANS, not raw sd (see module docstring)
            se = ((res["sd"] ** 2 + state["best_sd"] ** 2)
                  / max(args.repeats, 1)) ** 0.5
            if res["mean"] - state["best"] > args.sigma_k * max(0.2, se):
                state.update(best=res["mean"], best_sd=res["sd"],
                             best_cfg=cand, stale=0)
                accepted = True
        if not accepted:
            state["stale"] += 1
        state["history"].append(
            {"cfg": cand, "ok": res["ok"], "why": res.get("why", ""),
             "mean": round(res.get("mean") or 0, 2),
             "sd": round(res.get("sd") or 0, 2), "ppl": res.get("ppl"),
             "accepted": accepted})
        sp.write_text(json.dumps(state, indent=2))
        flag = "ACCEPT" if accepted else ("reject" if res["ok"] else "GUARD")
        deltas = {k: v for k, v in cand.items() if v != REFERENCE[k]}
        print(f"[auto] {state['n']:>3} {flag:>6} {res.get('mean') or 0:>7.1f} t/s "
              f"(best {state['best']:.1f}, stale {state['stale']}) {deltas} "
              f"{res.get('why', '')}", flush=True)

        if state["n"] % 15 == 0:                        # drift re-check
            b = run_bench(model, state["best_cfg"], args.repeats, args.tokens,
                          args.prompt_tokens, args.delay)
            if b:
                state["best"], state["best_sd"] = b["tg"], b["tg_sd"]
                print(f"[auto] incumbent re-measured: {b['tg']:.1f} t/s",
                      flush=True)
                sp.write_text(json.dumps(state, indent=2))

    why = "plateau" if state["stale"] >= args.patience else "time budget"
    gain = 100 * (state["best"] / state["baseline"] - 1)
    deltas = {k: v for k, v in state["best_cfg"].items() if v != REFERENCE[k]}
    print(f"\n[auto] STOPPED ({why}) after {state['n']} experiments")
    print(f"[auto] {state['baseline']:.1f} -> {state['best']:.1f} t/s ({gain:+.1f}%)")
    print(f"[auto] changes vs reference: {deltas or 'none'}")
    print("[auto] NEXT: re-validate this config on a real task eval — perplexity "
          "does not catch capability damage.")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(
        description="unattended inference-config autoresearch")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--sigma-k", type=float, default=2.0)
    ap.add_argument("--min-agreement", type=float, default=0.95)
    ap.add_argument("--max-ppl-drift", type=float, default=0.02)
    ap.add_argument("--vram-budget", type=int, default=900,
                    help="MiB another process may hold before we refuse to run")
    ap.add_argument("--temp-ceiling", type=int, default=80)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--no-check-text", dest="check_text", action="store_false")
    ap.add_argument("--ppl-file", default="D:/AVA/ratchet/logs/ppl_calib.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state", default="D:/AVA/ratchet/logs/autoresearch_state.json")
    ap.set_defaults(check_text=True)
    args = ap.parse_args()
    Path(args.state).parent.mkdir(parents=True, exist_ok=True)
    with Lock():
        search(args.model, args)


if __name__ == "__main__":
    main()
