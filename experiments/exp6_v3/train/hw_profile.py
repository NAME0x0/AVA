"""Hardware profiles for v3.0 SFT; see docs/v3/REVIEW_2026-07.md section 13."""

import os
from dataclasses import dataclass

import torch


@dataclass
class HWProfile:
    name: str
    vram_gb: float
    bf16: bool
    compute_dtype: str
    seq_len: int
    grad_accum: int
    micro_batch: int
    shard_minutes: float
    sync_minutes: float
    notes: str


PROFILES = {
    "t4": HWProfile(
        "t4",
        15,
        False,
        "float16",
        1024,
        8,
        2,
        150,
        30,
        "Turing: no bf16; fp16 compute",
    ),
    "l4": HWProfile(
        "l4",
        22,
        True,
        "bfloat16",
        2048,
        16,
        1,
        150,
        30,
        "Ada: bf16+Triton; mb1 — 248K-vocab CE chain ~6GB at seq2048 (~18.5GB usable)",
    ),
    "a100": HWProfile(
        "a100",
        40,
        True,
        "bfloat16",
        4096,
        4,
        4,
        110,
        15,
        "burst tier: short shards, tight sync — high CU burn",
    ),
    "v100": HWProfile(
        "v100",
        16,
        False,
        "float16",
        1024,
        8,
        2,
        150,
        30,
        "Volta: fp16",
    ),
    "h100": HWProfile(
        "h100",
        80,
        True,
        "bfloat16",
        4096,
        2,
        8,
        90,
        15,
        "extreme CU burn (~20-30 CU/h): burst only — QLoRA 4B underutilizes it",
    ),
    "h200": HWProfile(
        "h200",
        141,
        True,
        "bfloat16",
        4096,
        2,
        8,
        90,
        15,
        "extreme CU burn: burst only",
    ),
    "rtx pro 6000": HWProfile(
        "g4",
        96,
        True,
        "bfloat16",
        4096,
        2,
        8,
        90,
        15,
        "Colab G4 (Blackwell): burst tier",
    ),
    "a2000": HWProfile(
        "a2000",
        4,
        True,
        "bfloat16",
        640,
        32,
        1,
        600,
        30,
        "laptop lane",
    ),
}


# NOTE (2026-07-11 field fix): the table dtype is AUTHORITATIVE for known
# GPUs. torch.cuda.is_bf16_supported() returns True on Turing (T4) via slow
# software emulation, so "hardware truth wins" silently picked emulated bf16
# on T4 — exactly the perf trap the table exists to avoid. The runtime probe
# is only consulted for GPUs the table doesn't know.


def detect_profile() -> HWProfile:
    if "COLAB_TPU_ADDR" in os.environ or "TPU_NAME" in os.environ:
        raise RuntimeError(
            "TPU unsupported: pipeline is PyTorch+bitsandbytes (CUDA only). Select a GPU runtime."
        )

    if not torch.cuda.is_available():
        return HWProfile(
            "cpu",
            0,
            False,
            "float16",
            256,
            4,
            1,
            600,
            60,
            "dry-run/dev only",
        )

    device_name = torch.cuda.get_device_name(0).lower()
    if "p100" in device_name:
        raise RuntimeError(
            "P100 unsupported: bitsandbytes 4-bit needs compute capability >= 7.0, "
            "but P100 is 6.0. Switch Kaggle accelerator to T4 x2."
        )

    for matcher, profile in PROFILES.items():
        if matcher in device_name:
            return profile

    bf16 = bool(torch.cuda.is_bf16_supported())
    props = torch.cuda.get_device_properties(0)
    compute_dtype = "bfloat16" if bf16 else "float16"
    return HWProfile(
        "unknown",
        props.total_memory / 1_000_000_000,
        bf16,
        compute_dtype,
        1024,
        8,
        2,
        150,
        30,
        "unknown gpu: conservative defaults",
    )


def apply_profile(cfg, profile: HWProfile) -> str:
    cfg.seq_len = profile.seq_len
    cfg.grad_accum = profile.grad_accum
    cfg.micro_batch = profile.micro_batch
    cfg.shard_minutes = profile.shard_minutes
    cfg.sync_minutes = profile.sync_minutes
    print(
        "[hw] "
        f"{profile.name}: {profile.vram_gb:.1f} GB, dtype={profile.compute_dtype}, "
        f"seq_len={profile.seq_len}, mb={profile.micro_batch}x{profile.grad_accum}, "
        f"shard={profile.shard_minutes}m, sync={profile.sync_minutes}m; {profile.notes}"
    )
    return profile.compute_dtype
