"""Offline tests for the checkpoint-anywhere fabric (no network, fake Hub)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.checkpoint_sync as cs  # noqa: E402


class FakeApi:
    _fake = True

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def create_repo(self, repo_id: str, private: bool = True, exist_ok: bool = True) -> None:
        pass

    def upload_file(self, path_or_fileobj, path_in_repo: str, repo_id: str) -> None:
        self.files[path_in_repo] = path_or_fileobj.read()


def _fake_download_factory(api: FakeApi, tmp: Path):
    def fake_download(repo_id: str, path: str) -> str:
        if path not in api.files:
            raise FileNotFoundError(path)
        local = tmp / path.replace("/", "__")
        local.write_bytes(api.files[path])
        return str(local)

    return fake_download


def test_save_resume_roundtrip(tmp_path, monkeypatch) -> None:
    api = FakeApi()
    monkeypatch.setattr(cs, "hf_hub_download", _fake_download_factory(api, tmp_path))

    model = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 8)).sum().backward()
    opt.step()

    sync = cs.CheckpointSync("user/repo", phase="C2", api=api)
    path = sync.save(123, model, opt, extra={"shard": 7})
    assert path in api.files
    pointer = json.loads(api.files["checkpoints/C2/LATEST.json"])
    assert pointer["step"] == 123 and pointer["path"] == path

    fresh_model = torch.nn.Linear(8, 8)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    next_step = cs.CheckpointSync("user/repo", phase="C2", api=api).resume(
        fresh_model, fresh_opt
    )
    assert next_step == 124
    for a, b in zip(model.parameters(), fresh_model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_interval_gating(monkeypatch) -> None:
    api = FakeApi()
    sync = cs.CheckpointSync("user/repo", phase="C2", every_minutes=30, api=api)
    model = torch.nn.Linear(4, 4)
    assert sync.maybe_save(1, model) is True            # first push always fires
    assert sync.maybe_save(2, model) is False           # gated by interval
    assert sync.maybe_save(3, model, force=True) is True


def test_trainable_only_saves_subset_and_resumes(tmp_path, monkeypatch) -> None:
    api = FakeApi()
    monkeypatch.setattr(cs, "hf_hub_download", _fake_download_factory(api, tmp_path))

    # Frozen "donor" layer + a small trainable "adapter"; only the adapter trains.
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    for p in model[0].parameters():
        p.requires_grad_(False)

    sync = cs.CheckpointSync("user/repo", phase="C2", api=api, trainable_only=True)
    path = sync.save(5, model)
    payload = torch.load(
        _fake_download_factory(api, tmp_path)("user/repo", path),
        weights_only=False,
    )
    keys = set(payload["model"])
    assert all(k.startswith("1.") for k in keys)  # only the trainable second layer
    assert payload["trainable_only"] is True

    # Resume onto a full fresh model (frozen base already present) must not error.
    fresh = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    next_step = cs.CheckpointSync(
        "user/repo", phase="C2", api=api, trainable_only=True
    ).resume(fresh)
    assert next_step == 6
    for a, b in zip(model[1].parameters(), fresh[1].parameters(), strict=True):
        assert torch.equal(a, b)


def test_fresh_start_when_no_pointer(tmp_path, monkeypatch) -> None:
    api = FakeApi()
    monkeypatch.setattr(cs, "hf_hub_download", _fake_download_factory(api, tmp_path))
    sync = cs.CheckpointSync("user/repo", phase="C9", api=api)
    assert sync.resume(torch.nn.Linear(4, 4)) == 0
