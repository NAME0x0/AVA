"""CPU tests for the v3.0 training pipeline: data, gate, sandbox, dry-run loop."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.data import (  # noqa: E402
    DecontamFilter,
    MixtureStream,
    SourceSpec,
    extract_code,
    fim_transform,
    map_commitpackft,
    render_hashline_edit,
    strip_think,
)
from train.gate import check_gate  # noqa: E402
from train.sandbox_exec import check_solution, run_python  # noqa: E402

# --------------------------------------------------------------------------- text utils


def test_strip_think_and_extract_code() -> None:
    text = "<think>secret plan</think>Here:\n```python\nx = 1\n```"
    assert "secret" not in strip_think(text)
    assert extract_code(text) == "x = 1"


# --------------------------------------------------------------------------- hashline


def test_hashline_edit_anchors_and_content() -> None:
    old = "def f():\n    a = 1\n    return a\n"
    new = "def f():\n    a = 2\n    return a\n"
    edit = render_hashline_edit(old, new, "m.py")
    assert edit is not None
    assert edit.startswith("<<<EDIT m.py")
    assert "- a = 1" in edit.replace("    ", " ") or "-     a = 1" in edit
    assert "@@ " in edit  # hash anchor present
    assert edit.rstrip().endswith(">>>")


def test_hashline_none_on_identical_or_rewrite() -> None:
    code = "a = 1\n"
    assert render_hashline_edit(code, code) is None
    # near-total rewrite -> falls back to None (plain sample path)
    old = "\n".join(f"x{i} = {i}" for i in range(5))
    new = "\n".join(f"y{i} = {i * 3 + 1}" for i in range(7))
    assert render_hashline_edit(old, new) is None


def test_commitpackft_mapper_produces_edit_dialect() -> None:
    ex = {
        "old_contents": "def f():\n    a = 1\n    b = 2\n    return a + b\n",
        "new_contents": "def f():\n    a = 1\n    b = 3\n    return a + b\n",
        "message": "bump b",
        "new_file": "calc.py",
    }
    s = map_commitpackft(ex)
    assert s is not None and s.kind == "edit"
    assert "hash-anchored" in s.prompt
    assert "<<<EDIT calc.py" in s.response


# --------------------------------------------------------------------------- FIM


def test_fim_transform_round_trip_parts() -> None:
    code = "\n".join(f"line{i} = {i}" for i in range(20))
    rng = random.Random(0)
    fim = fim_transform(code, rng)
    assert fim is not None
    for tok in ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"):
        assert tok in fim
    # all content preserved across the three segments
    joined = fim.replace("<|fim_prefix|>", "").replace("<|fim_middle|>", "\n").replace(
        "<|fim_suffix|>", "\n"
    )
    for i in range(20):
        assert f"line{i} = {i}" in joined


def test_fim_transform_short_code_returns_none() -> None:
    assert fim_transform("a=1\nb=2", random.Random(0)) is None


# --------------------------------------------------------------------------- decontam


def test_decontam_blocks_eval_overlap() -> None:
    eval_text = " ".join(f"w{i}" for i in range(40))
    f = DecontamFilter([eval_text], n=10, max_collisions=2)
    assert not f.clean(eval_text)                       # verbatim leak blocked
    assert f.clean("totally unrelated content " * 10)   # clean passes
    assert f.rejected == 1


# --------------------------------------------------------------------------- mixture determinism + resume


def _rows(prefix: str, n: int) -> list[dict]:
    return [
        {"input": f"{prefix} question {i} " + "pad " * 20, "output": f"answer {i} " + "pad " * 20}
        for i in range(n)
    ]


def _stream(seed: int = 42) -> MixtureStream:
    rows_a, rows_b = _rows("a", 50), _rows("b", 50)
    return MixtureStream(
        [
            SourceSpec("opencodereasoning", 0.5, lambda r=rows_a: iter(r)),
            SourceSpec("opencodereasoning", 0.5, lambda r=rows_b: iter(r)),
        ],
        seed=seed,
    )


def test_mixture_deterministic() -> None:
    s1, s2 = _stream(), _stream()
    for _ in range(30):
        assert next(s1).prompt == next(s2).prompt


def test_mixture_resume_matches_uninterrupted() -> None:
    ref = _stream()
    seen = [next(ref).prompt for _ in range(30)]

    part = _stream()
    for _ in range(12):
        next(part)
    cursor = part.cursor()

    resumed = _stream()
    resumed.skip_to(cursor)
    for k in range(12, 30):
        assert next(resumed).prompt == seen[k]


def test_mixture_rejects_wrong_seed_cursor() -> None:
    s = _stream(seed=1)
    cur = s.cursor()
    other = _stream(seed=2)
    with pytest.raises(ValueError):
        other.skip_to(cur)


# --------------------------------------------------------------------------- sandbox


def test_sandbox_pass_fail_timeout() -> None:
    assert run_python("print('hi')").ok
    assert not run_python("raise ValueError('no')").ok
    res = run_python("while True: pass", timeout_s=1.5)
    assert res.timeout and not res.ok
    ok = check_solution("def add(a,b):\n    return a+b", "assert add(2,3)==5")
    assert ok.ok


# --------------------------------------------------------------------------- gate


def _report(he: float, mbpp: float, arc: float = 90.0, mmlu: float = 60.0) -> dict:
    return {
        "non_thinking": {
            "humaneval_plus": {"score": he, "n": 164},
            "mbpp_plus": {"score": mbpp, "n": 378},
            "arc_easy": {"score": arc, "n": 500},
            "mmlu": {"score": mmlu, "n": 1000},
        }
    }


def test_gate_passes_within_tolerance() -> None:
    r = check_gate(_report(80.0, 70.0), _report(78.5, 71.0))
    assert r.passed and r.deltas["humaneval_plus"] == -1.5


def test_gate_fails_on_regression_and_floor() -> None:
    r = check_gate(_report(80.0, 70.0), _report(75.0, 70.0))       # -5pp HE+
    assert not r.passed and any("humaneval_plus" in f for f in r.failures)
    r2 = check_gate(_report(80.0, 70.0), _report(80.0, 70.0, arc=60.0))  # floor break
    assert not r2.passed and any("arc_easy" in f for f in r2.failures)


# --------------------------------------------------------------------------- dry-run trainer (end-to-end)


def test_dry_run_shard_end_to_end() -> None:
    from train.sft import SFTConfig, train_shard

    cfg = SFTConfig(
        dry_run=True,
        seq_len=256,
        grad_accum=2,
        total_steps=3,
        shard_minutes=5.0,
        warmup_steps=2,
        sources=[
            {"name": "opencodereasoning", "weight": 1.0, "fim_fraction": 0.0},
        ],
        decontam_against=[],
    )
    summary = train_shard(cfg, test_rows={"opencodereasoning": _rows("t", 40)})
    assert summary["end_step"] == 3
    assert summary["last_loss"] == summary["last_loss"]  # not NaN
    assert summary["cursor"]["draws"] > 0


def test_hw_profile_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from train.hw_profile import detect_profile

    monkeypatch.delenv("COLAB_TPU_ADDR", raising=False)
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    profile = detect_profile()

    assert profile.name == "cpu"
    assert profile.compute_dtype == "float16"


def test_hw_profile_tpu_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from train.hw_profile import detect_profile

    monkeypatch.setenv("TPU_NAME", "x")

    with pytest.raises(RuntimeError):
        detect_profile()


def test_hw_profile_t4_and_p100(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from train.hw_profile import detect_profile

    class Props:
        total_memory = int(15e9)

    monkeypatch.delenv("COLAB_TPU_ADDR", raising=False)
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Tesla T4")
    # torch reports bf16=True on Turing via SLOW emulation — the table must
    # win for known GPUs (field bug 2026-07-11: T4 silently got bf16)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: Props())

    profile = detect_profile()

    assert profile.compute_dtype == "float16"
    assert profile.seq_len == 1024

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Tesla P100-PCIE-16GB")
    with pytest.raises(RuntimeError):
        detect_profile()


def test_apply_profile_mutates_cfg() -> None:
    from train.hw_profile import HWProfile, apply_profile
    from train.sft import SFTConfig

    cfg = SFTConfig(dry_run=True)
    profile = HWProfile(
        "unit",
        8,
        True,
        "bfloat16",
        512,
        7,
        9,
        3,
        "test profile",
    )

    dtype = apply_profile(cfg, profile)

    assert dtype == "bfloat16"
    assert cfg.seq_len == 512
    assert cfg.grad_accum == 7
    assert cfg.shard_minutes == 9
    assert cfg.sync_minutes == 3


# --------------------------------------------------------------------------- phase controller


def _decider(files: dict[str, dict]):
    """Fake Hub: {path: json_content}."""
    def exists(repo, path):
        return path in files
    def load(repo, path):
        return files[path]
    return exists, load


def test_phase_first_run_is_c1() -> None:
    from train.phase_controller import decide_phase

    ex, ld = _decider({})
    d = decide_phase("r", 100, ex, ld)
    assert d.phase == "C1" and d.modes_needed == (False, True)


_FULL_MODE = {
    "humaneval_plus": {"score": 1}, "mbpp_plus": {"score": 1},
    "arc_easy": {"score": 1}, "mmlu": {"score": 1},
}


def test_phase_partial_c1_resumes_missing_mode() -> None:
    from train.phase_controller import decide_phase

    # gate mode complete, thinking absent -> C5 proceeds (probe is optional)
    ex, ld = _decider(
        {"reports/c1_donor_baseline.json": {"non_thinking": dict(_FULL_MODE)}}
    )
    assert decide_phase("r", 100, ex, ld).phase == "C5"

    # gate mode killed mid-benchmark -> still C1 (per-bench resume)
    ex, ld = _decider({"reports/c1_donor_baseline.json": {
        "non_thinking": {"humaneval_plus": {"score": 1}}}})
    d = decide_phase("r", 100, ex, ld)
    assert d.phase == "C1" and d.modes_needed == (False,)


def test_phase_c5_start_and_resume() -> None:
    from train.phase_controller import decide_phase

    base = {"reports/c1_donor_baseline.json": {
        "non_thinking": dict(_FULL_MODE), "thinking": dict(_FULL_MODE)}}
    ex, ld = _decider(dict(base))
    assert decide_phase("r", 100, ex, ld).phase == "C5"

    ex, ld = _decider({**base, "checkpoints/C5/LATEST.json": {"step": 40}})
    d = decide_phase("r", 100, ex, ld)
    assert d.phase == "C5" and "41/100" in d.reason


def test_phase_c5_eval_then_done() -> None:
    from train.phase_controller import decide_phase

    base = {
        "reports/c1_donor_baseline.json": {
            "non_thinking": dict(_FULL_MODE), "thinking": dict(_FULL_MODE)},
        "checkpoints/C5/LATEST.json": {"step": 99},
    }
    ex, ld = _decider(dict(base))
    assert decide_phase("r", 100, ex, ld).phase == "C5_EVAL"

    ex, ld = _decider({**base, "reports/c5_candidate_eval.json": {}})
    assert decide_phase("r", 100, ex, ld).phase == "DONE"


def test_phase_unreachable_repo_falls_back_to_c1() -> None:
    from train.phase_controller import decide_phase

    def boom(repo, path):
        raise ConnectionError("offline")

    d = decide_phase("r", 100, boom, None)
    assert d.phase == "C1"


def test_gate_candidate_uses_hub_reports() -> None:
    from train.phase_controller import gate_candidate

    reports = {
        "reports/c1_donor_baseline.json": _report(80.0, 70.0),
        "reports/c5_candidate_eval.json": _report(81.0, 71.0),
    }
    res = gate_candidate("r", load_json_fn=lambda repo, path: reports[path])
    assert res.passed


def test_c1_seed_report_merges_existing(tmp_path) -> None:
    import json as _json

    from train.c1_eval import _seed_report

    out = tmp_path / "c1_donor_baseline.json"
    out.write_text(_json.dumps({"non_thinking": {"mmlu": {"score": 70}}}))
    seeded = _seed_report(out, hub_repo=None)
    assert "non_thinking" in seeded          # completed mode survives resume
    assert _seed_report(tmp_path / "missing.json", hub_repo=None) == {}
