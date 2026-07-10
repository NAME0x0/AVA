"""Fast-kernel installer for Qwen3.5's Gated DeltaNet path (FLA + causal-conv1d).

Problem: `flash-linear-attention` is pure Triton (pip-installable), but
`causal-conv1d` compiles a CUDA extension and pip's build isolation pairs it
with the wrong torch — the build fails on Colab/Kaggle and transformers falls
back to the slow torch implementation (observed: ~2-3x slower decode,
much slower prefill).

Strategy (ephemeral-VM friendly):
  1. already importable? done.
  2. try a wheel cached in the HF checkpoint repo (seconds).
  3. build with --no-build-isolation (~5-10 min), then upload the wheel to
     the Hub so every future session takes path 2.

Never fatal: the torch fallback is correct, just slow. Usage (notebook):

    from scripts.install_fast_kernels import ensure_fast_kernels
    ensure_fast_kernels(hub_repo=CKPT_REPO)
"""
from __future__ import annotations

import glob
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path


def _tag() -> str:
    """Wheel cache key: python + torch + cuda versions must all match."""
    import torch

    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    return f"{py}-torch{torch.__version__.split('+')[0]}-cu{torch.version.cuda}"


def _importable(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:  # noqa: BLE001 - any import failure means "not usable"
        return False


def _pip(*args: str, timeout: int = 1500) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        print(f"[kernels] pip {' '.join(args[:2])}... failed:\n{proc.stderr[-800:]}")
    return proc.returncode == 0


def ensure_fast_kernels(hub_repo: str | None = None) -> bool:
    """Returns True when the GDN fast path is available after the call."""
    if not _importable("torch"):
        print("[kernels] torch missing — skip")
        return False

    # -- flash-linear-attention: pure Triton, easy --------------------------
    if not _importable("fla"):
        _pip("install", "-q", "flash-linear-attention")

    # -- causal-conv1d: the compiled part ------------------------------------
    if not _importable("causal_conv1d"):
        tag = _tag()
        got = False
        # 2) cached wheel from the Hub
        if hub_repo:
            try:
                from huggingface_hub import HfApi, hf_hub_download

                api = HfApi()
                for f in api.list_repo_files(hub_repo):
                    if f.startswith(f"wheels/{tag}/") and f.endswith(".whl"):
                        local = hf_hub_download(hub_repo, f)
                        got = _pip("install", "-q", local)
                        print(f"[kernels] cached wheel {'installed' if got else 'failed'}: {f}")
                        break
            except Exception as err:  # noqa: BLE001
                print(f"[kernels] hub wheel lookup failed: {err!r}")
        # 3) build from source against the installed torch, then cache
        if not got:
            print("[kernels] building causal-conv1d (~5-10 min, one time)...")
            with tempfile.TemporaryDirectory() as tmp:
                built = _pip(
                    "wheel", "causal-conv1d", "--no-build-isolation",
                    "--no-deps", "-w", tmp,
                )
                wheels = glob.glob(f"{tmp}/causal_conv1d*.whl") if built else []
                if wheels:
                    got = _pip("install", "-q", wheels[0])
                    if got and hub_repo:
                        try:
                            from huggingface_hub import HfApi

                            api = HfApi()
                            api.create_repo(hub_repo, private=True, exist_ok=True)
                            api.upload_file(
                                path_or_fileobj=wheels[0],
                                path_in_repo=f"wheels/{_tag()}/{Path(wheels[0]).name}",
                                repo_id=hub_repo,
                            )
                            print("[kernels] wheel cached to Hub for future sessions")
                        except Exception as err:  # noqa: BLE001
                            print(f"[kernels] wheel upload failed (non-fatal): {err!r}")

    fla_ok = _importable("fla")
    conv_ok = _importable("causal_conv1d")
    print(f"[kernels] fast path: fla={fla_ok} causal_conv1d={conv_ok} "
          f"-> {'ENABLED' if fla_ok and conv_ok else 'torch fallback (slow, correct)'}")
    return fla_ok and conv_ok
