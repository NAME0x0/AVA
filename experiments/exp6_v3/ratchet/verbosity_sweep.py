"""Series 1 / Exp A: can prompting suppress LFM2.5's preamble?

Problem (measured on LFM2.5-2.6B): it writes an analysis, re-quotes the ORIGINAL
program in a fence, then gives the edit. ~889-4096 tokens per edit vs the donor's
~900, i.e. ~4x. On a 4 GB box that's slower wall-clock and burns context that an
agentic harness needs for tool schemas and repo state.

Tested on LFM2.5-230M (same family: double-gated short-conv + GQA), where a
generation costs ~1s instead of ~4min. Prompt/template behaviour is an
architecture+training-family property, so the WINNING VARIANT should transfer to
2.6B even though absolute quality will not.

Greedy decoding -> deterministic, so one run per (variant, problem) suffices;
we are counting tokens and checking format, not timing.

Metrics per variant:
  tokens     - mean generated tokens (primary: verbosity)
  fences     - mean code blocks emitted (>1 means it re-quotes the original)
  parses     - fraction whose extracted code compiles (format compliance)
  pass       - fraction passing the real tests (capability guard: don't trade
               brevity for correctness)
"""
import json
import statistics as st
import sys
import time

import torch
import transformers

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/AVA/experiments/exp6_v3")

from train.canitedit_eval import (  # noqa: E402
    _classify,
    _extract_final_code,
    _load_rows,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM2.5-230M"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
MAXTOK = 2560
OUT = f"D:/AVA/ratchet/logs/verbosity_sweep_{MODEL.split('/')[-1]}.json"

BASE_INSTR = (
    "You are editing an existing Python program. Apply the requested change "
    "and reply with the COMPLETE edited program in a single ```python code "
    "block — not a diff, not only the changed part.\n\n"
)
TERSE = (
    "Output ONLY the complete edited program in one ```python block. "
    "Do not explain. Do not restate the original program. Do not comment on "
    "your changes. Begin your reply with ```python and nothing before it.\n\n"
)
SYSTEM = ("You are a code editor. You reply with code only — never prose, "
          "never explanation, never the original program.")


def build(before: str, instruction: str, style: str) -> tuple[list[dict], str]:
    """Returns (messages, assistant_prefill)."""
    body = (f"Current program:\n```python\n{before}\n```\n\n"
            f"Change to make:\n{instruction}")
    if style == "baseline":
        return [{"role": "user", "content": BASE_INSTR + body}], ""
    if style == "terse":
        return [{"role": "user", "content": TERSE + body}], ""
    if style == "system":
        return ([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": BASE_INSTR + body}], "")
    if style == "prefill":                      # force it straight into code
        return [{"role": "user", "content": BASE_INSTR + body}], "```python\n"
    if style == "terse_prefill":
        return [{"role": "user", "content": TERSE + body}], "```python\n"
    if style == "sys_terse_prefill":
        return ([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": TERSE + body}], "```python\n")
    raise ValueError(style)


VARIANTS = (["baseline", "system", "prefill"] if "2.6B" in MODEL else
            ["baseline", "terse", "system", "prefill", "terse_prefill",
             "sys_terse_prefill"])

bnb = transformers.BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
kw = {"dtype": torch.bfloat16, "trust_remote_code": True, "device_map": {"": 0}}
if "230M" not in MODEL:                      # small sibling runs fine in bf16
    kw["quantization_config"] = bnb
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, **kw)
tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model.eval()
print(f"{MODEL} loaded | vram {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

rows = _load_rows(N)
results: dict[str, dict] = {}

for style in VARIANTS:
    toks, fences, parses, passes = [], [], 0, 0
    t0 = time.time()
    for r in rows:
        msgs, prefill = build(r["before"], r["instruction_descriptive"], style)
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False)
        text += prefill
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAXTOK, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full = prefill + gen                  # prefill is part of the answer
        n = out.shape[1] - ids.shape[1]
        code = _extract_final_code(full if "```" in full else f"```python\n{full}\n```")
        ok, _ = _classify(code, r["tests"], 15.0, truncated=n >= MAXTOK - 32)
        toks.append(n)
        fences.append(full.count("```") // 2)
        try:
            compile(code, "<e>", "exec")
            parses += 1
        except SyntaxError:
            pass
        passes += ok
    results[style] = {
        "tokens_mean": round(st.mean(toks), 1),
        "tokens_median": round(st.median(toks), 1),
        "fences_mean": round(st.mean(fences), 2),
        "parses": f"{parses}/{len(rows)}",
        "pass": f"{passes}/{len(rows)}",
        "wall_s": round(time.time() - t0),
    }
    print(f"{style:>20}: {results[style]}", flush=True)

json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2)
base = results["baseline"]["tokens_mean"]
print(f"\n=== verbosity vs baseline ({base} tok) ===")
for s, v in sorted(results.items(), key=lambda kv: kv[1]["tokens_mean"]):
    print(f"  {s:>20}: {v['tokens_mean']:>7} tok  "
          f"({100*v['tokens_mean']/base:5.1f}% of baseline)  "
          f"fences {v['fences_mean']}  parses {v['parses']}  pass {v['pass']}")
print(f"\nwrote {OUT}")
