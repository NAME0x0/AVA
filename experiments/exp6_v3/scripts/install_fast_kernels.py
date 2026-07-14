"""Fast-kernel installer for Qwen3.5's Gated DeltaNet path (FLA + causal-conv1d).

Two libraries speed the GDN layers, very different to install:
  - flash-linear-attention (fla): pure Triton, pip-installable, no compile.
    This is the main GDN win and it always installs cleanly.
  - causal-conv1d: compiles a CUDA extension. The source build is the slow,
    fragile part — nvcc defaults to compiling EVERY GPU arch (~15-25 min of
    SILENT output, which looks like a hang), and a torch/CUDA mismatch under
    pip build-isolation makes it fail outright.

Strategy (ephemeral-VM friendly, never blocks training):
  1. already importable? done.
  2. fla: quick pip install if missing.
  3. causal-conv1d: cached wheel from the Hub (seconds) -> else an
     ARCH-LIMITED source build (only the detected GPU's arch, ~3-5 min,
     VERBOSE so it never looks stuck) with a hard timeout -> else give up
     and use the torch fallback (correct, slower).

Cache note: wheels are keyed py-torch-cuda, so a repo may hold one wheel per
platform (e.g. Colab torch 2.11 + Kaggle torch 2.10) — that is expected, not
a duplicate bug.

Skip entirely with AVA_SKIP_KERNELS=1. Never fatal.

    from scripts.install_fast_kernels import ensure_fast_kernels
    ensure_fast_kernels(hub_repo=CKPT_REPO)
"""
from __future__ import annotations

import glob
import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _tag() -> str:
    """Wheel cache key: python + torch + cuda versions must all match."""
    import torch

    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    return f"{py}-torch{torch.__version__.split('+')[0]}-cu{torch.version.cuda}"


def _arch() -> str | None:
    """Detected GPU compute capability as an nvcc arch string, e.g. '8.9' (L4)."""
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def _importable(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:  # noqa: BLE001 - any import failure means "not usable"
        return False


def _pip(*args: str, timeout: int = 900, env: dict | None = None, verbose: bool = False) -> bool:
    """Run pip. Returns True on success. Never raises — a timeout or crash
    returns False so the caller falls back instead of dying."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", *args],
            capture_output=not verbose,  # verbose -> stream to the notebook, no "hang"
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"[kernels] pip {' '.join(args[:2])}... timed out after {timeout}s — falling back")
        return False
    except Exception as err:  # noqa: BLE001
        print(f"[kernels] pip {' '.join(args[:2])}... errored ({err!r}) — falling back")
        return False
    if proc.returncode != 0 and not verbose:
        print(f"[kernels] pip {' '.join(args[:2])}... failed:\n{(proc.stderr or '')[-800:]}")
    return proc.returncode == 0


def ensure_fast_kernels(hub_repo: str | None = None, build_conv: bool = False) -> bool:
    """Ensure the GDN fast path. Returns True iff both libs end up importable.

    Default installs fla only (pure Triton, clean, the win we rely on).
    causal-conv1d is OPT-IN (build_conv=True) — it froze sessions via all-arch
    builds and stalled cached-wheel downloads for a marginal conv speedup.
    AVA_SKIP_KERNELS=1 skips everything (torch fallback).
    """
    if os.environ.get("AVA_SKIP_KERNELS") == "1":
        print("[kernels] AVA_SKIP_KERNELS=1 -> skipping, torch fallback (correct, slower)")
        return False
    if not _importable("torch"):
        print("[kernels] torch missing — skip")
        return False

    # -- flash-linear-attention: pure Triton, the main GDN win ---------------
    if not _importable("fla"):
        print("[kernels] installing flash-linear-attention (Triton, no compile)...")
        _pip("install", "-q", "flash-linear-attention", timeout=600)

    # -- causal-conv1d: the compiled, fragile part — OPT-IN ONLY -------------
    # Field history: this component froze sessions two ways — an all-arch
    # source build (~20 min silent) AND a stalled 171 MB cached-wheel download
    # (hf_hub_download has no wall-clock timeout). Its speedup is marginal (a
    # small 1D conv) vs fla's recurrence win. So it is now opt-in: default
    # off. fla alone is the fast path we actually rely on.
    if build_conv and not _importable("causal_conv1d"):
        got = False
        tag = _tag()
        # (a) cached wheel from the Hub — seconds (behind build_conv now, so a
        #     stalled download can't freeze the default path)
        if hub_repo:
            try:
                from huggingface_hub import HfApi, hf_hub_download

                for f in HfApi().list_repo_files(hub_repo):
                    if f.startswith(f"wheels/{tag}/") and f.endswith(".whl"):
                        local = hf_hub_download(hub_repo, f)
                        got = _pip("install", "-q", local, timeout=300)
                        print(f"[kernels] cached wheel {'installed' if got else 'failed'}: {f}")
                        break
            except Exception as err:  # noqa: BLE001
                print(f"[kernels] hub wheel lookup failed: {err!r}")
        # (b) arch-limited source build — VERBOSE, capped, non-blocking
        if not got:
            arch = _arch()
            env = dict(os.environ)
            if arch:
                env["TORCH_CUDA_ARCH_LIST"] = arch  # one arch, not all -> ~5x faster
            env["MAX_JOBS"] = env.get("MAX_JOBS", "4")
            print(f"[kernels] building causal-conv1d for arch {arch or 'default'} "
                  f"(~3-5 min, live output below; set AVA_SKIP_KERNELS=1 to skip)...")
            with tempfile.TemporaryDirectory() as tmp:
                built = _pip(
                    "wheel", "causal-conv1d", "--no-build-isolation", "--no-deps",
                    "-w", tmp, timeout=900, env=env, verbose=True,
                )
                wheels = glob.glob(f"{tmp}/causal_conv1d*.whl") if built else []
                if wheels:
                    got = _pip("install", "-q", wheels[0], timeout=300)
                    if got and hub_repo:
                        try:
                            from huggingface_hub import HfApi

                            api = HfApi()
                            api.create_repo(hub_repo, private=True, exist_ok=True)
                            api.upload_file(
                                path_or_fileobj=wheels[0],
                                path_in_repo=f"wheels/{tag}/{Path(wheels[0]).name}",
                                repo_id=hub_repo,
                            )
                            print("[kernels] wheel cached to Hub for future sessions")
                        except Exception as err:  # noqa: BLE001
                            print(f"[kernels] wheel upload failed (non-fatal): {err!r}")

    fla_ok = _importable("fla")
    conv_ok = _importable("causal_conv1d")
    if fla_ok and conv_ok:
        status = "ENABLED (full fast path)"
    elif fla_ok:
        status = "PARTIAL (fla only; conv on torch fallback — still faster than no fla)"
    else:
        status = "torch fallback (slow, correct)"
    print(f"[kernels] fast path: fla={fla_ok} causal_conv1d={conv_ok} -> {status}")
    return fla_ok and conv_ok
