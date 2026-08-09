"""Unattended autoresearch over INFERENCE MECHANISM (not weights, not layers).

Karpathy's autoresearch edits train.py and scores val_bpb. This searches the
llama.cpp execution configuration and scores throughput, which suits us better:
no training, seconds per experiment, and the weights never change — so a "win"
is a real mechanism win, transferable to the bigger sibling of the same family.

Design borrowed from autoresearch: the loop may change the CONFIG but can never
touch the MEASUREMENT or the quality guard. That separation is what stops a
search from making the test easier instead of the engine faster.

Guards against the three ways this could lie to itself:
  1. NOISE      - warm-up round discarded, arms measured with repeats, and a
                  candidate must beat the incumbent by > `sigma_k` pooled sd.
                  (Measured 2026-08-08: naive single runs vary +-26%.)
  2. QUALITY    - every candidate must still produce output matching the fp16-KV
                  reference by >= `min_agreement` token overlap. Aggressive KV
                  quantization buys speed with quality; that is not a win.
  3. DRIFT      - the incumbent is re-measured periodically, so thermal drift
                  cannot silently promote a bad config.

Stops on plateau (no accepted improvement in `patience` rounds) or the time
budget. Fully resumable: state is checkpointed after every experiment.

  python -m ratchet.autoresearch_config --model <gguf> --hours 8
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import subprocess
import time
from pathlib import Path

BIN = Path("D:/AVA/tools/llamacpp-bin/llama-cli")
PROMPT = ("Write a Python function `parse_csv(text)` that parses CSV text with "
          "quoted fields and escaped quotes into a list of rows, plus 3 examples.")

# Search space: knobs that change EXECUTION only, never the weights.
SPACE: dict[str, list] = {
    "ub":       [64, 128, 256, 512],          # prefill micro-batch
    "b":        [256, 512, 1024, 2048],       # logical batch
    "ctk":      ["f16", "q8_0", "q4_0"],      # KV cache dtypes
    "ctv":      ["f16", "q8_0", "q4_0"],
    "threads":  [4, 6, 8, 16],
    "spec":     ["none", "ngram-simple", "ngram-cache", "ngram-mod"],
    "fa":       ["on", "off"],                # flash attention
    "ngl":      [99],                         # small sibling: always full offload
}
REFERENCE = {"ub": 512, "b": 2048, "ctk": "f16", "ctv": "f16", "threads": 8,
             "spec": "none", "fa": "on", "ngl": 99}

_TPS = re.compile(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s")


def cmd(model: str, cfg: dict, n_tokens: int) -> list[str]:
    return [
        str(BIN), "-m", model, "-ngl", str(cfg["ngl"]), "-c", "2048",
        "-b", str(cfg["b"]), "-ub", str(cfg["ub"]),
        "-ctk", cfg["ctk"], "-ctv", cfg["ctv"], "-t", str(cfg["threads"]),
        "-fa", cfg["fa"], "--temp", "0", "-n", str(n_tokens), "-st", "-cnv",
        *(["--spec-type", cfg["spec"]] if cfg["spec"] != "none" else []),
        "-p", PROMPT,
    ]


def run_once(model: str, cfg: dict, n_tokens: int) -> tuple[float | None, str]:
    try:
        p = subprocess.run(cmd(model, cfg, n_tokens), capture_output=True,
                           text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, ""
    blob = (p.stdout or "") + (p.stderr or "")
    m = _TPS.search(blob)
    body = (p.stdout or "").split("]")[-1] if m else (p.stdout or "")
    return (float(m.group(2)) if m else None), body


def _toks(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def agreement(a: str, b: str, head: int = 120) -> float:
    """Fraction of the reference's leading tokens reproduced, in order."""
    ta, tb = _toks(a)[:head], _toks(b)[:head]
    if not ta:
        return 1.0
    same = sum(1 for x, y in zip(ta, tb, strict=False) if x == y)
    return same / len(ta)


def measure(model: str, cfg: dict, repeats: int, n_tokens: int,
            ref_text: str | None, min_agreement: float) -> dict:
    """Warm-up discarded, then `repeats` timed runs. Returns mean/sd/quality."""
    run_once(model, cfg, 32)                                   # warm-up
    speeds, last = [], ""
    for _ in range(repeats):
        tps, body = run_once(model, cfg, n_tokens)
        if tps is None:
            return {"ok": False, "why": "run failed"}
        speeds.append(tps)
        last = body
    agree = agreement(ref_text, last) if ref_text else 1.0
    return {
        "ok": agree >= min_agreement,
        "why": "" if agree >= min_agreement else f"quality {agree:.2f}",
        "mean": st.mean(speeds), "sd": st.stdev(speeds) if len(speeds) > 1 else 0.0,
        "agree": round(agree, 3), "text": last,
    }


def neighbour(cfg: dict, rng: random.Random) -> dict:
    """One-knob mutation: keeps attribution interpretable (we can say WHICH knob)."""
    nxt = dict(cfg)
    for _ in range(20):
        k = rng.choice([k for k in SPACE if len(SPACE[k]) > 1])
        v = rng.choice([x for x in SPACE[k] if x != cfg[k]])
        nxt[k] = v
        if nxt != cfg:
            return nxt
    return nxt


def search(model: str, hours: float, repeats: int, n_tokens: int, patience: int,
           sigma_k: float, min_agreement: float, state_path: Path,
           seed: int = 0) -> dict:
    rng = random.Random(seed)
    deadline = time.monotonic() + hours * 3600
    state = json.loads(state_path.read_text()) if state_path.exists() else None

    if state is None:
        print("[auto] measuring reference (fp16 KV, no spec)...", flush=True)
        ref = measure(model, REFERENCE, repeats, n_tokens, None, 0.0)
        if not ref["ok"]:
            raise RuntimeError(f"reference run failed: {ref['why']}")
        print(f"[auto] reference {ref['mean']:.1f} +-{ref['sd']:.2f} t/s", flush=True)
        state = {"model": model, "ref_text": ref["text"],
                 "best_cfg": REFERENCE, "best": ref["mean"], "best_sd": ref["sd"],
                 "baseline": ref["mean"], "history": [], "stale": 0, "n": 0}
        state_path.write_text(json.dumps(state, indent=2))

    ref_text = state["ref_text"]
    while time.monotonic() < deadline and state["stale"] < patience:
        cand = neighbour(state["best_cfg"], rng)
        if any(h["cfg"] == cand for h in state["history"]):
            continue                                    # already tried
        res = measure(model, cand, repeats, n_tokens, ref_text, min_agreement)
        state["n"] += 1
        rec = {"cfg": cand, "ok": res["ok"], "why": res.get("why", ""),
               "mean": round(res.get("mean", 0), 2), "sd": round(res.get("sd", 0), 2),
               "agree": res.get("agree")}
        # noise-aware acceptance: must clear the incumbent by sigma_k pooled sd
        accepted = False
        if res["ok"]:
            pooled = max(0.3, (res["sd"] ** 2 + state["best_sd"] ** 2) ** 0.5)
            if res["mean"] - state["best"] > sigma_k * pooled:
                state.update(best=res["mean"], best_sd=res["sd"], best_cfg=cand,
                             stale=0)
                accepted = True
            else:
                state["stale"] += 1
        else:
            state["stale"] += 1
        rec["accepted"] = accepted
        state["history"].append(rec)
        state_path.write_text(json.dumps(state, indent=2))
        flag = "ACCEPT" if accepted else ("reject" if res["ok"] else "GUARD")
        print(f"[auto] {state['n']:>3} {flag:>6} {rec['mean']:>6.1f} t/s "
              f"(best {state['best']:.1f}, stale {state['stale']}) "
              f"{ {k: v for k, v in cand.items() if v != REFERENCE[k]} } "
              f"{rec['why']}", flush=True)

        if state["n"] % 15 == 0:                        # drift re-check
            re_m = measure(model, state["best_cfg"], repeats, n_tokens, ref_text,
                           min_agreement)
            if re_m["ok"]:
                state["best"], state["best_sd"] = re_m["mean"], re_m["sd"]
                print(f"[auto] incumbent re-measured: {re_m['mean']:.1f} t/s",
                      flush=True)
                state_path.write_text(json.dumps(state, indent=2))

    why = "plateau" if state["stale"] >= patience else "time budget"
    gain = 100 * (state["best"] / state["baseline"] - 1)
    print(f"\n[auto] STOPPED ({why}) after {state['n']} experiments")
    print(f"[auto] baseline {state['baseline']:.1f} -> best {state['best']:.1f} t/s "
          f"({gain:+.1f}%)")
    print(f"[auto] best config: {state['best_cfg']}")
    deltas = {k: v for k, v in state["best_cfg"].items() if v != REFERENCE[k]}
    print(f"[auto] changes from reference: {deltas or 'none'}")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="unattended inference-config autoresearch")
    ap.add_argument("--model", required=True, help="GGUF path (small sibling)")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--patience", type=int, default=25, help="plateau: rejects in a row")
    ap.add_argument("--sigma-k", type=float, default=2.0, help="accept threshold in sd")
    ap.add_argument("--min-agreement", type=float, default=0.85)
    ap.add_argument("--state", default="D:/AVA/ratchet/logs/autoresearch_state.json")
    a = ap.parse_args()
    sp = Path(a.state)
    sp.parent.mkdir(parents=True, exist_ok=True)
    search(a.model, a.hours, a.repeats, a.tokens, a.patience, a.sigma_k,
           a.min_agreement, sp)


if __name__ == "__main__":
    main()
