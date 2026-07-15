"""Print a HF dataset's schema + first row — verify BEFORE writing a mapper.

Hard-won rule (PLAN_2026-07-15_v31; the MBPP+/commitpackft mapper bugs):
never map a guessed schema. Run this on any new source in-session first:

    from scripts.inspect_dataset import inspect
    inspect("nvidia/Open-SWE-Traces")                 # default config/split
    inspect("nvidia/OpenMathReasoning", split="train")

It prints column names + types and a truncated first row so the mapper is
written against the REAL fields.
"""
from __future__ import annotations

import json
from typing import Any


def inspect(
    hf_id: str,
    config: str | None = None,
    split: str = "train",
    data_files: str | None = None,
) -> dict[str, Any]:
    from datasets import load_dataset

    if data_files:
        ds = load_dataset("json", data_files=data_files, split=split, streaming=True)
    elif config:
        ds = load_dataset(hf_id, config, split=split, streaming=True)
    else:
        ds = load_dataset(hf_id, split=split, streaming=True)

    row = next(iter(ds))
    print(f"=== {hf_id} (config={config}, split={split}) ===")
    print("COLUMNS:", sorted(row.keys()))
    for k, v in row.items():
        t = type(v).__name__
        if isinstance(v, list):
            inner = type(v[0]).__name__ if v else "empty"
            preview = json.dumps(v[0], default=str)[:200] if v else "[]"
            print(f"  {k}: list[{inner}] (len {len(v)}) first={preview}")
        else:
            s = str(v)
            print(f"  {k}: {t} = {s[:200]}{'...' if len(s) > 200 else ''}")
    return row
