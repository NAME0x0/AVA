"""Checkpoint-anywhere training fabric — HF Hub as the single source of truth.

Every training platform (laptop, Colab, Kaggle) is treated as a preemptible
worker. The Hub repo holds the canonical state; any worker can die at any
moment and any other worker can resume within one sync interval.

Layout in the Hub repo (default: NAME0x0/AVA-v3-checkpoints):

    checkpoints/<phase>/step00012345/state.pt     model + optimizer + RNG + meta
    checkpoints/<phase>/LATEST.json               atomic pointer, uploaded last

Resume protocol: read LATEST.json -> download state.pt -> restore model,
optimizer, RNG, and dataloader position (data order is deterministic from
seed + step, per the reproducibility invariants in docs/v3/RECIPE.md).

Auth: token comes from the HF_TOKEN environment variable or the cached
`huggingface-cli login` — NEVER from code or config files. The token needs
write permission on the checkpoint repo.

Usage in a trainer loop:

    sync = CheckpointSync("NAME0x0/AVA-v3-checkpoints", phase="C2")
    start_step = sync.resume(model, optimizer)          # 0 if fresh
    for step in range(start_step, total_steps):
        ...train...
        sync.maybe_save(step, model, optimizer)         # time-gated, safe to call every step
"""
from __future__ import annotations

import io
import json
import os
import random
import time
from typing import Any

import torch

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError as _err:  # pragma: no cover - import guard
    HfApi = None  # type: ignore[assignment]
    hf_hub_download = None  # type: ignore[assignment]
    _IMPORT_ERROR = _err
else:
    _IMPORT_ERROR = None

_LATEST = "LATEST.json"


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointSync:
    """Time-gated checkpoint push + latest-pointer resume against one Hub repo."""

    def __init__(
        self,
        repo_id: str,
        phase: str,
        every_minutes: float = 30.0,
        max_retries: int = 5,
        private: bool = True,
        api: Any | None = None,
    ) -> None:
        if api is None:
            if _IMPORT_ERROR is not None:
                raise ImportError(
                    "checkpoint_sync requires huggingface_hub: pip install huggingface_hub"
                ) from _IMPORT_ERROR
            api = HfApi()  # token resolved from HF_TOKEN env or login cache
        self.api = api
        self.repo_id = repo_id
        self.phase = phase
        self.every_seconds = every_minutes * 60.0
        self.max_retries = max_retries
        self._last_push = 0.0

        token_present = bool(os.environ.get("HF_TOKEN")) or bool(
            getattr(api, "token", None)
        )
        if not token_present and not hasattr(api, "_fake"):
            # Soft check — HfApi may still find a cached login at call time.
            print(
                "[checkpoint_sync] warning: HF_TOKEN not set; relying on cached "
                "`huggingface-cli login`. Uploads will fail without write access."
            )
        self.api.create_repo(repo_id, private=private, exist_ok=True)

    # -- save ---------------------------------------------------------------

    def maybe_save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        extra: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """Push a checkpoint if the sync interval elapsed. Returns True if pushed."""
        now = time.monotonic()
        if not force and (now - self._last_push) < self.every_seconds:
            return False
        self.save(step, model, optimizer, extra)
        self._last_push = time.monotonic()
        return True

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "step": step,
            "phase": self.phase,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "rng": _rng_state(),
            "extra": extra or {},
            "saved_unix": time.time(),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)

        folder = f"checkpoints/{self.phase}/step{step:08d}"
        state_path = f"{folder}/state.pt"
        self._upload(state_path, buffer.getvalue())
        # Pointer uploaded LAST -> a torn upload never corrupts the resume path.
        pointer = json.dumps(
            {"phase": self.phase, "step": step, "path": state_path,
             "saved_unix": payload["saved_unix"]}
        ).encode()
        self._upload(f"checkpoints/{self.phase}/{_LATEST}", pointer)
        return state_path

    def _upload(self, path_in_repo: str, data: bytes) -> None:
        delay = 5.0
        for attempt in range(self.max_retries):
            try:
                self.api.upload_file(
                    path_or_fileobj=io.BytesIO(data),
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                )
                return
            except Exception as err:  # noqa: BLE001 - network layer is noisy
                if attempt == self.max_retries - 1:
                    raise
                print(
                    f"[checkpoint_sync] upload failed ({err!r}); "
                    f"retry {attempt + 1}/{self.max_retries} in {delay:.0f}s"
                )
                time.sleep(delay)
                delay = min(delay * 2, 120.0)

    # -- resume -------------------------------------------------------------

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str = "cpu",
    ) -> int:
        """Restore the latest checkpoint for this phase. Returns next step (0 if fresh)."""
        pointer = self._read_latest()
        if pointer is None:
            return 0
        local = hf_hub_download(self.repo_id, pointer["path"])
        payload = torch.load(local, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload["optimizer"] is not None:
            optimizer.load_state_dict(payload["optimizer"])
        _restore_rng(payload["rng"])
        print(
            f"[checkpoint_sync] resumed {self.repo_id} {pointer['path']} "
            f"(phase {payload['phase']}, step {payload['step']})"
        )
        return int(payload["step"]) + 1

    def _read_latest(self) -> dict[str, Any] | None:
        try:
            local = hf_hub_download(
                self.repo_id, f"checkpoints/{self.phase}/{_LATEST}"
            )
        except Exception:  # noqa: BLE001 - missing pointer == fresh start
            return None
        with open(local, encoding="utf-8") as fh:
            return json.load(fh)
