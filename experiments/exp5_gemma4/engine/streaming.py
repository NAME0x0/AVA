"""Streaming quantization loader — loads any model without full bf16 materialization.

The problem: a 26B bf16 model is 50 GB, which doesn't fit in 32 GB RAM.
The solution: stream safetensors weight-by-weight, quantize each linear
layer to int4 in-place, never hold more than 1 layer of bf16 at a time.

Peak RAM for a 26B model: ~15 GB (quantized weights + 1 bf16 layer buffer).
This approach works for ANY model size — 70B, 405B, etc.

Usage:
    model, processor = streaming_load(
        "google/gemma-4-26B-A4B-it",
        gpu_memory_gb=3.5,
    )
"""
from __future__ import annotations

import gc
import json
import math
import os
import tempfile
import time
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_TRITON_WORKSPACE = Path(__file__).resolve().parents[3] / ".triton-runtime"
_TRITON_CACHE_DIR = _TRITON_WORKSPACE / "cache"
_TRITON_TEMP_DIR = _TRITON_WORKSPACE / "tmp"
for _path in (_TRITON_CACHE_DIR, _TRITON_TEMP_DIR):
    _path.mkdir(parents=True, exist_ok=True)

os.environ["TRITON_CACHE_DIR"] = str(_TRITON_CACHE_DIR)
os.environ["TMP"] = str(_TRITON_TEMP_DIR)
os.environ["TEMP"] = str(_TRITON_TEMP_DIR)
os.environ["TMPDIR"] = str(_TRITON_TEMP_DIR)
tempfile.tempdir = str(_TRITON_TEMP_DIR)


def _workspace_mkdtemp(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | None = None,
) -> str:
    """Create temp directories under the writable workspace on Windows."""
    root = Path(dir or tempfile.gettempdir())
    while True:
        candidate = root / f"{prefix or 'tmp'}{uuid.uuid4().hex}{suffix or ''}"
        try:
            candidate.mkdir(parents=True)
            return str(candidate)
        except FileExistsError:
            continue


tempfile.mkdtemp = _workspace_mkdtemp

if "CC" not in os.environ:
    for _candidate in (
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang.exe"),
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/bin/clang.exe"),
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36014/bin/Hostx64/x64/cl.exe"),
    ):
        if _candidate.exists():
            os.environ["CC"] = str(_candidate)
            break

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

_TRITON_INT4_DISABLED = False
_TRITON_INT4_INTEGRATION_ENABLED = os.environ.get("AVA_ENABLE_TRITON_INT4", "0") == "1"

if triton is not None and tl is not None:
    @triton.jit
    def _int4_linear_kernel(
        x_ptr,
        packed_ptr,
        scales_ptr,
        zeros_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_pn,
        stride_pk,
        stride_sn,
        stride_sg,
        stride_zn,
        stride_zg,
        stride_om,
        stride_on,
        group_size,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )

            packed_cols = offs_k // 2
            raw = tl.load(
                packed_ptr + offs_n[None, :] * stride_pn + packed_cols[:, None] * stride_pk,
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
                other=0,
            )
            nibble = tl.where(
                (offs_k % 2)[:, None] == 0,
                (raw >> 4) & 0x0F,
                raw & 0x0F,
            ).to(tl.float16)

            group_idx = offs_k // group_size
            scale = tl.load(
                scales_ptr + offs_n[None, :] * stride_sn + group_idx[:, None] * stride_sg,
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
                other=0.0,
            )
            zero = tl.load(
                zeros_ptr + offs_n[None, :] * stride_zn + group_idx[:, None] * stride_zg,
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
                other=0.0,
            )
            weight = nibble * scale + zero
            acc += tl.dot(x, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(tl.float16),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def _triton_int4_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run a packed-int4 linear directly on CUDA via Triton when possible."""
    global _TRITON_INT4_DISABLED

    if _TRITON_INT4_DISABLED or triton is None or tl is None:
        return None
    if (
        x.device.type != "cuda"
        or packed.device.type != "cuda"
        or scales.device.type != "cuda"
        or zeros.device.type != "cuda"
    ):
        return None
    if x.dtype != torch.float16 or scales.dtype != torch.float16 or zeros.dtype != torch.float16:
        return None
    if packed.dtype != torch.uint8 or x.shape[-1] > packed.shape[1] * 2:
        return None

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty((x_2d.shape[0], packed.shape[0]), device=x.device, dtype=torch.float16)
    block_m = 4 if x_2d.shape[0] >= 4 else 1
    block_n = 64 if packed.shape[0] >= 64 else 32
    block_k = 128 if x_2d.shape[1] % 128 == 0 else 64

    grid = (
        triton.cdiv(x_2d.shape[0], block_m),
        triton.cdiv(packed.shape[0], block_n),
    )

    try:
        _int4_linear_kernel[grid](
            x_2d,
            packed,
            scales,
            zeros,
            out,
            x_2d.shape[0],
            packed.shape[0],
            x_2d.shape[1],
            x_2d.stride(0),
            x_2d.stride(1),
            packed.stride(0),
            packed.stride(1),
            scales.stride(0),
            scales.stride(1),
            zeros.stride(0),
            zeros.stride(1),
            out.stride(0),
            out.stride(1),
            group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
    except Exception:
        _TRITON_INT4_DISABLED = True
        return None

    if bias is not None:
        out += bias.to(device=out.device, dtype=out.dtype)
    return out.view(*x.shape[:-1], packed.shape[0])


def _get_safetensors_index(model_path: Path) -> dict[str, str]:
    """Read the safetensors index mapping weight names to shard files.

    Returns:
        Dict mapping weight name -> shard filename.
    """
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
        return data["weight_map"]

    # Single file model — read header to get tensor names
    single = model_path / "model.safetensors"
    if single.exists():
        header = _read_safetensors_header(single)
        return {name: "model.safetensors" for name in header if name != "__metadata__"}

    raise FileNotFoundError(
        f"No safetensors files found in {model_path}. "
        "Download the model first with: huggingface-cli download <model_id>"
    )


def _read_safetensors_header(filepath: Path) -> dict:
    """Read safetensors header without memory-mapping the file.

    The safetensors format:
      - 8 bytes: uint64 header_size (little-endian)
      - header_size bytes: JSON header
      - remaining: raw tensor data
    """
    import struct
    with open(filepath, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
    return json.loads(header_json)


# Mapping safetensors dtype strings to torch dtypes
_ST_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _read_tensor_from_safetensors(
    filepath: Path,
    tensor_name: str,
    header: dict | None = None,
) -> torch.Tensor:
    """Read a single tensor from a safetensors file without memory-mapping.

    This avoids the segfault caused by mmap on files larger than available RAM.
    Reads only the bytes needed for the specific tensor.
    """
    import struct

    if header is None:
        header = _read_safetensors_header(filepath)

    meta = header[tensor_name]
    dtype_str = meta["dtype"]
    shape = meta["shape"]
    start, end = meta["data_offsets"]

    torch_dtype = _ST_DTYPE_MAP[dtype_str]

    # Read header size to compute absolute offset
    with open(filepath, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        data_start = 8 + header_size + start
        data_length = end - start

        f.seek(data_start)
        raw = f.read(data_length)

    # Create tensor from raw bytes
    tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype).reshape(shape)
    return tensor.clone()  # Clone to own the memory (detach from bytearray)


def _group_weights_by_layer(weight_map: dict[str, str]) -> dict[str, list[str]]:
    """Group weight names by their layer prefix for sequential loading.

    Returns dict mapping group_key -> [weight_names].
    Groups: 'layer_0', 'layer_1', ..., 'embeddings', 'other'.
    """
    import re
    groups: dict[str, list[str]] = defaultdict(list)

    for name in weight_map:
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            groups[f"layer_{int(m.group(1))}"] = groups.get(f"layer_{int(m.group(1))}", [])
            groups[f"layer_{int(m.group(1))}"].append(name)
        elif "embed" in name.lower():
            groups["embeddings"].append(name)
        else:
            groups["other"].append(name)

    return dict(groups)


class StreamingInt4Quantizer:
    """Fast per-group asymmetric int4 quantization for CPU tensors.

    Quantizes a bf16/fp32 weight tensor to packed int4 with group-wise
    scales and zero points.  No calibration data needed — uses min/max
    per group.

    Storage format per quantized tensor:
        - packed_weight: uint8, shape (out_features, in_features // 2)
          Two int4 values packed per byte (high nibble, low nibble)
        - scales: float16, shape (out_features, n_groups)
        - zeros: float16, shape (out_features, n_groups)
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def quantize(self, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize a 2D weight matrix to packed int4.

        Args:
            weight: shape (out_features, in_features), any dtype

        Returns:
            Dict with 'packed', 'scales', 'zeros' tensors.
        """
        assert weight.ndim == 2, f"Expected 2D weight, got {weight.ndim}D"
        out_f, in_f = weight.shape
        gs = self.group_size

        # Pad in_features to multiple of group_size
        n_groups = math.ceil(in_f / gs)
        padded_in = n_groups * gs
        if padded_in > in_f:
            weight = torch.nn.functional.pad(weight, (0, padded_in - in_f))

        w = weight.float().reshape(out_f, n_groups, gs)

        # Per-group min/max
        w_min = w.min(dim=2).values  # (out_f, n_groups)
        w_max = w.max(dim=2).values
        w_range = (w_max - w_min).clamp(min=1e-8)
        scale = w_range / 15.0  # 4-bit: 0-15

        # Quantize to [0, 15]
        quantized = ((w - w_min.unsqueeze(2)) / scale.unsqueeze(2)).round().clamp(0, 15)
        quantized = quantized.to(torch.uint8).reshape(out_f, padded_in)

        # Pack two int4 values per byte
        # Even indices go to high nibble, odd to low nibble
        packed = (quantized[:, 0::2] << 4) | quantized[:, 1::2]

        return {
            "packed": packed,
            "scales": scale.to(torch.float16),
            "zeros": w_min.to(torch.float16),
            "original_in_features": in_f,
        }

    def dequantize(self, qdata: dict[str, torch.Tensor]) -> torch.Tensor:
        """Dequantize packed int4 back to float16 for inference.

        Args:
            qdata: dict from quantize() with 'packed', 'scales', 'zeros'

        Returns:
            Reconstructed weight tensor in float16.
        """
        packed = qdata["packed"]
        scales = qdata["scales"].float()
        zeros = qdata["zeros"].float()
        in_f = qdata["original_in_features"]
        out_f = packed.shape[0]
        gs = self.group_size

        # Unpack: high nibble and low nibble
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F

        # Interleave back
        padded_in = packed.shape[1] * 2
        unpacked = torch.zeros(out_f, padded_in, dtype=torch.uint8, device=packed.device)
        unpacked[:, 0::2] = high
        unpacked[:, 1::2] = low

        # Reshape for group-wise dequantization
        n_groups = scales.shape[1]
        grouped = unpacked.float().reshape(out_f, n_groups, gs)

        # Dequantize: val * scale + zero_point
        dequantized = grouped * scales.unsqueeze(2) + zeros.unsqueeze(2)
        dequantized = dequantized.reshape(out_f, padded_in)

        # Trim padding
        return dequantized[:, :in_f].to(torch.float16)


def _dequantize_packed_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    """Reconstruct a float16 weight matrix from packed int4 storage."""
    out_f = packed.shape[0]
    padded_in = packed.shape[1] * 2

    high = (packed >> 4) & 0x0F
    low = packed & 0x0F

    unpacked = torch.zeros(out_f, padded_in, dtype=scales.dtype, device=packed.device)
    unpacked[:, 0::2] = high.to(scales.dtype)
    unpacked[:, 1::2] = low.to(scales.dtype)

    n_groups = scales.shape[1]
    grouped = unpacked.reshape(out_f, n_groups, group_size)
    dequantized = grouped * scales.unsqueeze(2) + zeros.unsqueeze(2)

    return dequantized.reshape(out_f, padded_in)[:, :in_features].to(torch.float16)


class DequantWeightCache:
    """Budgeted LRU cache for dequantized weights."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def get(
        self,
        key: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if key not in self._cache:
            self.misses += 1
            return None

        tensor = self._cache.pop(key)
        if tensor.device != device or tensor.dtype != dtype:
            self.current_bytes -= tensor.nbytes
            self.misses += 1
            return None

        self._cache[key] = tensor
        self.hits += 1
        return tensor

    def put(self, key: int, tensor: torch.Tensor) -> None:
        if self.max_bytes <= 0:
            return

        size = tensor.nbytes
        if size > self.max_bytes:
            return

        old = self._cache.pop(key, None)
        if old is not None:
            self.current_bytes -= old.nbytes

        while self.current_bytes + size > self.max_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self.current_bytes -= evicted.nbytes
            self.evictions += 1

        self._cache[key] = tensor
        self.current_bytes += size

    @property
    def size(self) -> int:
        return len(self._cache)

    def summary(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "evictions": self.evictions,
            "current_mb": round(self.current_bytes / 1e6, 1),
            "max_mb": round(self.max_bytes / 1e6, 1),
        }


class ExpertWeightCache:
    """Budgeted LRU cache for dequantized MoE expert weights."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()

    def get(
        self,
        key: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor] | None:
        entry = self._cache.pop(key, None)
        if entry is None:
            self.misses += 1
            return None

        if any(t.device != device or t.dtype != dtype for t in entry.values()):
            self.current_bytes -= sum(t.nbytes for t in entry.values())
            self.misses += 1
            return None

        self._cache[key] = entry
        self.hits += 1
        return entry

    def put(self, key: tuple[int, int], tensors: dict[str, torch.Tensor]) -> None:
        if self.max_bytes <= 0:
            return

        size = sum(t.nbytes for t in tensors.values())
        if size > self.max_bytes:
            return

        old = self._cache.pop(key, None)
        if old is not None:
            self.current_bytes -= sum(t.nbytes for t in old.values())

        while self.current_bytes + size > self.max_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self.current_bytes -= sum(t.nbytes for t in evicted.values())
            self.evictions += 1

        self._cache[key] = tensors
        self.current_bytes += size

    def summary(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "evictions": self.evictions,
            "current_mb": round(self.current_bytes / 1e6, 1),
            "max_mb": round(self.max_bytes / 1e6, 1),
        }


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear using int4 packed weights."""

    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        original_in_features: int,
        bias: torch.Tensor | None = None,
        group_size: int = 128,
    ):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        self.original_in_features = original_in_features
        self.group_size = group_size
        self.out_features = packed.shape[0]
        self.in_features = original_in_features
        self._dequant_cache: DequantWeightCache | None = None
        self._compute_dtype: torch.dtype | None = None
        self._cache_key = id(self)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def dequantized_weight_bytes(self) -> int:
        """Return the size of one dequantized weight copy."""
        return self.packed.shape[0] * self.original_in_features * 2

    def packed_storage_bytes(self) -> int:
        """Return packed/storage bytes needed to keep this module on GPU."""
        total = self.packed.nbytes + self.scales.nbytes + self.zeros.nbytes
        if self.bias is not None:
            total += self.bias.nbytes
        return total

    def configure_dequant_cache(
        self,
        cache: DequantWeightCache,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        """Attach a shared cache for dequantized hot weights."""
        self._dequant_cache = cache
        self._compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_device = x.device
        original_dtype = x.dtype
        target_device = self.packed.device if self.packed.device.type == "cuda" else x.device
        compute_dtype = self._compute_dtype if target_device.type == "cuda" else x.dtype
        if x.device != target_device or (compute_dtype is not None and x.dtype != compute_dtype):
            use_non_blocking = x.device.type == "cuda" and target_device.type == "cuda"
            compute_x = x.to(
                device=target_device,
                dtype=compute_dtype or x.dtype,
                non_blocking=use_non_blocking,
            )
        else:
            compute_x = x
        bias = self.bias.to(device=compute_x.device, dtype=compute_x.dtype) if self.bias is not None else None
        out = None
        if (
            _TRITON_INT4_INTEGRATION_ENABLED
            and compute_x.device.type == "cuda"
            and compute_x.dtype == torch.float16
        ):
            out = _triton_int4_linear(
                compute_x,
                self.packed,
                self.scales,
                self.zeros,
                self.group_size,
                bias=bias,
            )

        if out is None:
            weight = None
            if self._dequant_cache is not None:
                weight = self._dequant_cache.get(
                    self._cache_key,
                    compute_x.device,
                    compute_x.dtype,
                )

            if weight is None:
                weight = self._dequantize().to(device=compute_x.device, dtype=compute_x.dtype)
                if self._dequant_cache is not None:
                    self._dequant_cache.put(self._cache_key, weight)

            out = torch.nn.functional.linear(compute_x, weight, bias)
        if out.device != original_device or out.dtype != original_dtype:
            use_non_blocking = out.device.type == "cuda" and original_device.type == "cuda"
            return out.to(
                device=original_device,
                dtype=original_dtype,
                non_blocking=use_non_blocking,
            )
        return out

    def _dequantize(self) -> torch.Tensor:
        """Reconstruct fp16 weight from packed int4."""
        return _dequantize_packed_weight(
            self.packed,
            self.scales,
            self.zeros,
            self.original_in_features,
            self.group_size,
        )


class QuantizedMoEExperts(nn.Module):
    """Drop-in replacement for Gemma4TextExperts using int4 packed weights.

    Stores expert weights as packed int4 with per-group scales/zeros.
    During forward, only the SELECTED experts are dequantized to bf16
    on-the-fly, avoiding holding all 128 experts in full precision.

    Memory: 128 experts × int4 packed ≈ 1/4 of bf16 storage.
    """

    def __init__(
        self,
        num_experts: int,
        gate_up_packed: list[torch.Tensor],
        gate_up_scales: list[torch.Tensor],
        gate_up_zeros: list[torch.Tensor],
        gate_up_in_features: int,
        down_packed: list[torch.Tensor],
        down_scales: list[torch.Tensor],
        down_zeros: list[torch.Tensor],
        down_in_features: int,
        act_fn: nn.Module,
        group_size: int = 128,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.group_size = group_size
        self.act_fn = act_fn
        self.gate_up_in_features = gate_up_in_features
        self.down_in_features = down_in_features

        # Store packed weights for each expert as buffer lists
        for i in range(num_experts):
            self.register_buffer(f"gu_packed_{i}", gate_up_packed[i])
            self.register_buffer(f"gu_scales_{i}", gate_up_scales[i])
            self.register_buffer(f"gu_zeros_{i}", gate_up_zeros[i])
            self.register_buffer(f"dn_packed_{i}", down_packed[i])
            self.register_buffer(f"dn_scales_{i}", down_scales[i])
            self.register_buffer(f"dn_zeros_{i}", down_zeros[i])

        self._layer_idx: int | None = None
        self._expert_offloader: Any | None = None
        self._offload_device: torch.device | None = None
        self._compute_dtype = torch.bfloat16
        self._batched_dispatch_limit = 16
        self._gpu_cache_layout = "dequantized"
        self._expert_weight_cache: ExpertWeightCache | None = None
        self._decode_prefetch_token_limit = 4

    def expert_storage_bytes(self) -> int:
        """Return packed storage size for one expert."""
        if self.num_experts == 0:
            return 0
        total = 0
        for prefix in ("gu", "dn"):
            for suffix in ("packed", "scales", "zeros"):
                total += getattr(self, f"{prefix}_{suffix}_0").nbytes
        return total

    def expert_dequantized_bytes(self, dtype_bytes: int = 2) -> int:
        """Return the size of one cached dequantized expert copy."""
        if self.num_experts == 0:
            return 0
        gate_up = getattr(self, "gu_packed_0")
        down = getattr(self, "dn_packed_0")
        return (
            gate_up.shape[0] * self.gate_up_in_features
            + down.shape[0] * self.down_in_features
        ) * dtype_bytes

    def _get_expert_params(self, expert_idx: int) -> dict[str, torch.Tensor]:
        """Return the packed tensors for one expert without copying them."""
        return {
            "gu_packed": getattr(self, f"gu_packed_{expert_idx}"),
            "gu_scales": getattr(self, f"gu_scales_{expert_idx}"),
            "gu_zeros": getattr(self, f"gu_zeros_{expert_idx}"),
            "dn_packed": getattr(self, f"dn_packed_{expert_idx}"),
            "dn_scales": getattr(self, f"dn_scales_{expert_idx}"),
            "dn_zeros": getattr(self, f"dn_zeros_{expert_idx}"),
        }

    def configure_gpu_offload(
        self,
        layer_idx: int,
        offloader: Any,
        gpu_device: torch.device | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        batched_dispatch_limit: int = 16,
        cache_layout: str = "dequantized",
        expert_weight_cache: ExpertWeightCache | None = None,
    ) -> None:
        """Register CPU-resident experts with the shared GPU hot cache."""
        self._layer_idx = layer_idx
        self._expert_offloader = offloader
        self._offload_device = gpu_device
        self._compute_dtype = compute_dtype
        self._batched_dispatch_limit = batched_dispatch_limit
        self._gpu_cache_layout = cache_layout
        self._expert_weight_cache = expert_weight_cache

        for expert_idx in range(self.num_experts):
            offloader.register_expert(
                layer_idx,
                expert_idx,
                self._get_expert_params(expert_idx),
                transform=(
                    self._build_cached_expert_weights
                    if cache_layout == "dequantized"
                    else None
                ),
            )

    def _build_cached_expert_weights(
        self,
        params: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Build ready-to-run expert weights for the hot GPU cache."""
        gu_packed = params["gu_packed"].to(device, non_blocking=True)
        gu_scales = params["gu_scales"].to(device, non_blocking=True)
        gu_zeros = params["gu_zeros"].to(device, non_blocking=True)
        dn_packed = params["dn_packed"].to(device, non_blocking=True)
        dn_scales = params["dn_scales"].to(device, non_blocking=True)
        dn_zeros = params["dn_zeros"].to(device, non_blocking=True)

        return {
            "gu_weight": _dequantize_packed_weight(
                gu_packed,
                gu_scales,
                gu_zeros,
                self.gate_up_in_features,
                self.group_size,
            ).to(self._compute_dtype),
            "dn_weight": _dequantize_packed_weight(
                dn_packed,
                dn_scales,
                dn_zeros,
                self.down_in_features,
                self.group_size,
            ).to(self._compute_dtype),
        }

    def _dequantize_expert(
        self,
        prefix: str,
        expert_idx: int,
        in_features: int,
        tensors: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Dequantize one expert's weight matrix from packed int4 to float."""
        if tensors is None:
            packed = getattr(self, f"{prefix}_packed_{expert_idx}")
            scales = getattr(self, f"{prefix}_scales_{expert_idx}")
            zeros = getattr(self, f"{prefix}_zeros_{expert_idx}")
        else:
            packed = tensors[f"{prefix}_packed"]
            scales = tensors[f"{prefix}_scales"]
            zeros = tensors[f"{prefix}_zeros"]

        return _dequantize_packed_weight(
            packed,
            scales,
            zeros,
            in_features,
            self.group_size,
        )

    def _resolve_expert_weights(
        self,
        expert_idx: int,
        expert_tensors: dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve one expert into ready-to-run weights on the active device."""
        cache_key = None
        if (
            self._expert_weight_cache is not None
            and self._layer_idx is not None
            and "gu_weight" not in expert_tensors
            and "dn_weight" not in expert_tensors
        ):
            cache_key = (self._layer_idx, expert_idx)
            expert_device = next(iter(expert_tensors.values())).device
            cached = self._expert_weight_cache.get(cache_key, expert_device, dtype)
            if cached is not None:
                return cached["gu_weight"], cached["dn_weight"]

        gate_up_w = expert_tensors.get("gu_weight")
        if gate_up_w is None:
            gate_up_w = self._dequantize_expert(
                "gu", expert_idx, self.gate_up_in_features, tensors=expert_tensors,
            )

        down_w = expert_tensors.get("dn_weight")
        if down_w is None:
            down_w = self._dequantize_expert(
                "dn", expert_idx, self.down_in_features, tensors=expert_tensors,
            )

        gate_up_w = gate_up_w.to(dtype=dtype)
        down_w = down_w.to(dtype=dtype)

        if cache_key is not None and self._expert_weight_cache is not None:
            self._expert_weight_cache.put(
                cache_key,
                {"gu_weight": gate_up_w, "dn_weight": down_w},
            )

        return gate_up_w, down_w

    def _maybe_triton_expert_linear(
        self,
        x: torch.Tensor,
        expert_tensors: dict[str, torch.Tensor],
        prefix: str,
    ) -> torch.Tensor | None:
        """Run one packed expert projection directly on CUDA when possible."""
        packed = expert_tensors.get(f"{prefix}_packed")
        scales = expert_tensors.get(f"{prefix}_scales")
        zeros = expert_tensors.get(f"{prefix}_zeros")
        if packed is None or scales is None or zeros is None:
            return None
        if not _TRITON_INT4_INTEGRATION_ENABLED:
            return None
        return _triton_int4_linear(
            x,
            packed,
            scales,
            zeros,
            self.group_size,
        )

    def _run_expert_mlp(
        self,
        hidden_states: torch.Tensor,
        expert_idx: int,
        expert_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run one expert MLP using packed CUDA kernels when available."""
        gate_up_out = self._maybe_triton_expert_linear(hidden_states, expert_tensors, "gu")
        gate_up_w: torch.Tensor | None = None
        down_w: torch.Tensor | None = None
        if gate_up_out is None:
            gate_up_w, down_w = self._resolve_expert_weights(
                expert_idx,
                expert_tensors,
                hidden_states.dtype,
            )
            gate_up_out = nn.functional.linear(hidden_states, gate_up_w)

        gate, up = gate_up_out.chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up

        down_out = self._maybe_triton_expert_linear(current_hidden_states, expert_tensors, "dn")
        if down_out is None:
            if down_w is None:
                _, down_w = self._resolve_expert_weights(
                    expert_idx,
                    expert_tensors,
                    current_hidden_states.dtype,
                )
            down_out = nn.functional.linear(current_hidden_states, down_w)

        return down_out

    def _prefetch_next_token_experts(
        self,
        expert_ids: list[int],
        num_tokens: int,
    ) -> None:
        """Speculatively prefetch the same experts for the next decode step."""
        if (
            self._expert_offloader is None
            or self._layer_idx is None
            or not expert_ids
            or num_tokens > self._decode_prefetch_token_limit
        ):
            return
        prefetchable = self._expert_offloader.cacheable_decode_experts(
            self._layer_idx,
            expert_ids,
        )
        self._expert_offloader.record_decode_experts(self._layer_idx, expert_ids)
        if prefetchable:
            self._expert_offloader.prefetch_experts(
                self._layer_idx,
                sorted(prefetchable),
            )

    def _prepare_grouped_routes(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> dict[str, Any]:
        """Flatten routing decisions into expert-grouped slices."""
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)
        num_top_k = top_k_index.size(-1)

        token_idx = (
            torch.arange(num_tokens, device=top_k_index.device)
            .unsqueeze(1)
            .expand(-1, num_top_k)
            .reshape(-1)
        )
        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)

        valid_mask = expert_ids < self.num_experts
        if not bool(valid_mask.any()):
            return {
                "token_idx": token_idx[:0],
                "sample_weights": sample_weights[:0],
                "unique_expert_ids": [],
                "counts": [],
                "num_tokens": num_tokens,
                "hidden_dim": hidden_dim,
            }

        if not bool(valid_mask.all()):
            valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
            token_idx = token_idx.index_select(0, valid_positions)
            sample_weights = sample_weights.index_select(0, valid_positions)
            expert_ids = expert_ids.index_select(0, valid_positions)

        sort_order = torch.argsort(expert_ids)
        token_idx = token_idx.index_select(0, sort_order)
        sample_weights = sample_weights.index_select(0, sort_order)
        expert_ids = expert_ids.index_select(0, sort_order)

        unique_expert_ids, counts = torch.unique_consecutive(
            expert_ids,
            return_counts=True,
        )

        return {
            "token_idx": token_idx,
            "sample_weights": sample_weights,
            "unique_expert_ids": [int(e) for e in unique_expert_ids.tolist()],
            "counts": [int(c) for c in counts.tolist()],
            "num_tokens": num_tokens,
            "hidden_dim": hidden_dim,
        }

    def _forward_grouped_offload(
        self,
        hidden_states: torch.Tensor,
        routes: dict[str, Any],
    ) -> torch.Tensor:
        """Grouped MoE dispatch using the shared expert offloader."""
        assert self._expert_offloader is not None
        assert self._layer_idx is not None

        token_idx = routes["token_idx"]
        unique_expert_ids = routes["unique_expert_ids"]
        counts = routes["counts"]
        num_tokens = routes["num_tokens"]
        hidden_dim = routes["hidden_dim"]

        final_hidden_states = torch.zeros_like(hidden_states)
        if not unique_expert_ids:
            return final_hidden_states

        record_decode_access = num_tokens <= self._decode_prefetch_token_limit
        cacheable_decode_experts: set[int] | None = None
        if record_decode_access:
            cacheable_decode_experts = self._expert_offloader.cacheable_decode_experts(
                self._layer_idx,
                unique_expert_ids,
            )
        expert_cache = self._expert_offloader.load_experts_batch(
            self._layer_idx,
            unique_expert_ids,
            record_access=record_decode_access,
            cache_results=record_decode_access,
            cacheable_expert_indices=cacheable_decode_experts,
        )
        expert_device = next(iter(next(iter(expert_cache.values())).values())).device

        routed_hidden_states = hidden_states.index_select(0, token_idx).to(
            expert_device,
            dtype=self._compute_dtype,
            non_blocking=True,
        )
        routed_weights = routes["sample_weights"].to(
            expert_device,
            dtype=self._compute_dtype,
            non_blocking=True,
        )
        weighted_out = torch.empty(
            routed_hidden_states.size(0),
            hidden_dim,
            dtype=self._compute_dtype,
            device=expert_device,
        )

        max_group_count = max(counts)
        use_packed_triton = (
            routed_hidden_states.device.type == "cuda"
            and routed_hidden_states.dtype == torch.float16
            and triton is not None
            and tl is not None
            and not _TRITON_INT4_DISABLED
            and any(
                "gu_weight" not in expert_cache[expert_idx]
                or "dn_weight" not in expert_cache[expert_idx]
                for expert_idx in unique_expert_ids
            )
        )
        use_grouped_bmm = len(unique_expert_ids) <= 8 and max_group_count <= 4 and not use_packed_triton
        if use_grouped_bmm:
            grouped_states = torch.zeros(
                len(unique_expert_ids),
                max_group_count,
                routed_hidden_states.size(-1),
                dtype=self._compute_dtype,
                device=expert_device,
            )
            grouped_weights = torch.zeros(
                len(unique_expert_ids),
                max_group_count,
                1,
                dtype=self._compute_dtype,
                device=expert_device,
            )
            grouped_mask = torch.zeros(
                len(unique_expert_ids),
                max_group_count,
                1,
                dtype=torch.bool,
                device=expert_device,
            )
            gate_up_batch: list[torch.Tensor] = []
            down_batch: list[torch.Tensor] = []

        start = 0
        for group_idx, (expert_idx, count) in enumerate(zip(unique_expert_ids, counts)):
            state_slice = routed_hidden_states.narrow(0, start, count)
            weight_slice = routed_weights.narrow(0, start, count).unsqueeze(-1)
            expert_tensors = expert_cache[expert_idx]
            gate_up_w, down_w = self._resolve_expert_weights(
                expert_idx,
                expert_tensors,
                routed_hidden_states.dtype,
            )
            if use_grouped_bmm:
                grouped_states[group_idx, :count].copy_(state_slice)
                grouped_weights[group_idx, :count].copy_(weight_slice)
                grouped_mask[group_idx, :count] = True
                gate_up_batch.append(gate_up_w)
                down_batch.append(down_w)
            else:
                proj_out = self._run_expert_mlp(
                    state_slice,
                    expert_idx,
                    expert_tensors,
                )
                weighted_out.narrow(0, start, count).copy_(proj_out * weight_slice)
            start += count

        if use_grouped_bmm:
            gate_up_stack = torch.stack(gate_up_batch, dim=0).transpose(1, 2)
            proj_out = torch.bmm(grouped_states, gate_up_stack)
            gate, up = proj_out.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            down_stack = torch.stack(down_batch, dim=0).transpose(1, 2)
            proj_out = torch.bmm(current_hidden_states, down_stack)
            proj_out.mul_(grouped_weights)
            proj_out.masked_fill_(~grouped_mask, 0.0)

            start = 0
            for group_idx, count in enumerate(counts):
                weighted_out.narrow(0, start, count).copy_(proj_out[group_idx, :count])
                start += count

        final_hidden_states.index_add_(
            0,
            token_idx,
            weighted_out.to(device=final_hidden_states.device, dtype=final_hidden_states.dtype),
        )
        self._prefetch_next_token_experts(unique_expert_ids, num_tokens)
        return final_hidden_states

    def _forward_batched_offload(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fast path for small decode batches using per-sample batched GEMMs."""
        routes = self._prepare_grouped_routes(hidden_states, top_k_index, top_k_weights)
        return self._forward_grouped_offload(hidden_states, routes)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """MoE forward with per-expert on-the-fly dequantization.

        Mirrors the eager implementation from Gemma4TextExperts but
        dequantizes each expert's weight from int4 only when selected.
        """
        if (
            self._expert_offloader is not None
            and self._layer_idx is not None
            and hidden_states.size(0) * top_k_index.size(-1) <= self._batched_dispatch_limit
        ):
            return self._forward_batched_offload(hidden_states, top_k_index, top_k_weights)

        routes = self._prepare_grouped_routes(hidden_states, top_k_index, top_k_weights)
        if self._expert_offloader is not None and self._layer_idx is not None:
            return self._forward_grouped_offload(hidden_states, routes)

        final_hidden_states = torch.zeros_like(hidden_states)
        token_idx = routes["token_idx"]
        unique_expert_ids = routes["unique_expert_ids"]
        counts = routes["counts"]
        routed_hidden_states = hidden_states.index_select(0, token_idx)
        routed_weights = routes["sample_weights"]

        start = 0
        for expert_idx, count in zip(unique_expert_ids, counts):
            current_state = routed_hidden_states.narrow(0, start, count)
            routing_weights = routed_weights.narrow(0, start, count).unsqueeze(-1)
            gate_up_w = self._dequantize_expert(
                "gu", expert_idx, self.gate_up_in_features,
            ).to(current_state.dtype)
            gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up

            down_w = self._dequantize_expert(
                "dn", expert_idx, self.down_in_features,
            ).to(current_state.dtype)
            current_hidden_states = nn.functional.linear(current_hidden_states, down_w)
            final_hidden_states.index_add_(
                0,
                token_idx.narrow(0, start, count),
                (current_hidden_states * routing_weights).to(final_hidden_states.dtype),
            )
            start += count

        self._prefetch_next_token_experts(unique_expert_ids, routes["num_tokens"])
        return final_hidden_states


def _quantize_module_weights(
    module: nn.Module,
    quantizer: StreamingInt4Quantizer,
    prefix: str = "",
) -> int:
    """Replace nn.Linear modules with QuantizedLinear in-place.

    Returns number of linears quantized.
    """
    quantized_count = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            # Quantize this linear layer
            weight = child.weight.data
            if weight.ndim != 2:
                continue  # Skip non-standard shapes

            qdata = quantizer.quantize(weight)
            q_linear = QuantizedLinear(
                packed=qdata["packed"],
                scales=qdata["scales"],
                zeros=qdata["zeros"],
                original_in_features=qdata["original_in_features"],
                bias=child.bias.data if child.bias is not None else None,
                group_size=quantizer.group_size,
            )
            # Replace in parent
            setattr(module, name, q_linear)
            quantized_count += 1

            # Free original weight
            del weight, child
        else:
            # Recurse into non-Linear children
            quantized_count += _quantize_module_weights(child, quantizer, full_name)

    return quantized_count


def streaming_load(
    model_id: str,
    gpu_memory_gb: float = 3.5,
    cpu_memory_gb: float = 28.0,
    group_size: int = 128,
    dtype: str = "bfloat16",
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a model via streaming quantization.

    Instead of materializing the full bf16 model, this:
    1. Creates the model on meta device (0 RAM)
    2. Loads weights layer-by-layer from safetensors
    3. Quantizes linear weights to int4 before storing
    4. Peak RAM = 1 layer bf16 + all quantized layers so far

    This works for ANY model size — 26B, 70B, 405B — as long as the
    quantized model fits in available RAM.

    Args:
        model_id: HuggingFace model ID or local path
        gpu_memory_gb: GPU memory budget
        cpu_memory_gb: CPU memory budget
        group_size: quantization group size (128 = good balance)
        dtype: dtype for loading (bfloat16 recommended)

    Returns:
        model, processor, metadata
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    quantizer = StreamingInt4Quantizer(group_size=group_size)

    print(f"Streaming load: {model_id}")
    t0 = time.perf_counter()

    # Step 1: Load config and create model shell on meta device
    print("  Step 1/4: Creating model on meta device...")
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)

    # Step 2: Resolve safetensors path
    print("  Step 2/4: Resolving safetensors shards...")
    from huggingface_hub import snapshot_download
    model_path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))
    weight_map = _get_safetensors_index(model_path)
    groups = _group_weights_by_layer(weight_map)

    # Count total layers
    layer_groups = sorted(
        [k for k in groups if k.startswith("layer_")],
        key=lambda x: int(x.split("_")[1]),
    )
    n_layers = len(layer_groups)
    print(f"  Found {n_layers} layers across {len(set(weight_map.values()))} shards")

    # Step 3: Stream and quantize layer by layer
    print("  Step 3/4: Streaming weights + int4 quantization...")
    # Cache parsed shard headers for efficiency (avoids re-reading headers)
    shard_headers: dict[str, dict] = {}
    total_quantized = 0
    total_bf16_bytes = 0
    total_q4_bytes = 0

    def load_tensor(name: str) -> torch.Tensor:
        shard_name = weight_map[name]
        shard_path = model_path / shard_name
        if shard_name not in shard_headers:
            shard_headers[shard_name] = _read_safetensors_header(shard_path)
        return _read_tensor_from_safetensors(
            shard_path, name, shard_headers[shard_name],
        )

    def assign_tensor(name: str, tensor: torch.Tensor) -> None:
        """Assign a tensor to the model by navigating the module tree."""
        parts = name.split(".")
        module = model
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        param_name = parts[-1]
        old = getattr(module, param_name, None)

        if isinstance(old, nn.Parameter):
            # Replace parameter
            new_param = nn.Parameter(tensor, requires_grad=False)
            setattr(module, param_name, new_param)
        elif hasattr(module, param_name):
            # Buffer or regular attribute
            if isinstance(old, torch.Tensor) and hasattr(module, "_buffers") and param_name in module._buffers:
                module._buffers[param_name] = tensor
            else:
                setattr(module, param_name, tensor)
        else:
            setattr(module, param_name, tensor)

    # Process all weight groups
    process_order = (
        ["embeddings"]
        + layer_groups
        + ["other"]
    )

    for group_key in process_order:
        if group_key not in groups:
            continue

        weight_names = groups[group_key]
        group_bf16 = 0
        group_q4 = 0

        # Collect expert weights for batch quantization into QuantizedMoEExperts
        expert_tensors: dict[str, torch.Tensor] = {}

        for wname in weight_names:
            tensor = load_tensor(wname)
            bf16_bytes = tensor.nbytes
            group_bf16 += bf16_bytes
            total_bf16_bytes += bf16_bytes

            # Check if this is a 3D expert weight (gate_up_proj or down_proj)
            is_expert_weight = (
                tensor.ndim == 3
                and "expert" in wname
                and ("gate_up" in wname or "down_proj" in wname)
            )

            if is_expert_weight:
                # Defer — collect for batch processing after this layer
                expert_tensors[wname] = tensor
                continue

            # Quantize 2D linear weights (attention/MLP projections)
            is_linear_weight = (
                tensor.ndim == 2
                and tensor.shape[0] >= 64
                and tensor.shape[1] >= 64
                and any(k in wname for k in [".weight", "_proj", "gate_proj", "up_proj", "down_proj"])
                and "norm" not in wname
                and "embed" not in wname
                and "scalar" not in wname
                and "lm_head" not in wname
            )

            if is_linear_weight:
                qdata = quantizer.quantize(tensor)
                q4_bytes = qdata["packed"].nbytes + qdata["scales"].nbytes + qdata["zeros"].nbytes
                group_q4 += q4_bytes
                total_q4_bytes += q4_bytes
                total_quantized += 1

                # Build QuantizedLinear and replace the module
                _replace_linear_with_quantized(model, wname, qdata, quantizer.group_size)
                del tensor, qdata
            else:
                # Non-quantizable: assign directly (norms, embeddings, scalars)
                tensor = tensor.to(torch_dtype)
                assign_tensor(wname, tensor)
                group_q4 += tensor.nbytes
                total_q4_bytes += tensor.nbytes

        # Process collected expert weights -> QuantizedMoEExperts
        if expert_tensors:
            q4_bytes, n_quantized = _build_quantized_experts(
                model, expert_tensors, quantizer, config,
            )
            group_q4 += q4_bytes
            total_q4_bytes += q4_bytes
            total_quantized += n_quantized
            del expert_tensors

        if group_key.startswith("layer_"):
            layer_idx = int(group_key.split("_")[1])
            ratio = group_bf16 / max(group_q4, 1)
            if layer_idx % 5 == 0 or layer_idx == n_layers - 1:
                ram_gb = _get_process_rss_gb()
                print(f"    Layer {layer_idx:2d}/{n_layers}: "
                      f"{group_bf16 / 1e6:.0f} MB bf16 -> {group_q4 / 1e6:.0f} MB q4 "
                      f"({ratio:.1f}x) | RAM: {ram_gb:.1f} GB")

        # Force GC after each layer to keep RAM low
        gc.collect()

    # Clear cached headers
    shard_headers.clear()

    # Materialize any remaining meta tensors (non-persistent buffers
    # like embed_scale, inv_freq that are computed in __init__, not saved)
    meta_fixed = _materialize_meta_tensors(model, config, torch_dtype)
    if meta_fixed:
        print(f"  Materialized {meta_fixed} non-persistent buffers from meta device")

    # Re-establish weight tying (lm_head = embed_tokens for Gemma 4)
    model.tie_weights()

    load_time = time.perf_counter() - t0
    overall_ratio = total_bf16_bytes / max(total_q4_bytes, 1)

    print(f"\n  Step 4/4: Quantization complete")
    print(f"  Time: {load_time:.1f}s")
    print(f"  Linears quantized: {total_quantized}")
    print(f"  bf16: {total_bf16_bytes / 1e9:.1f} GB -> q4: {total_q4_bytes / 1e9:.1f} GB "
          f"({overall_ratio:.1f}x)")
    print(f"  Peak RAM: {_get_process_rss_gb():.1f} GB")

    metadata = {
        "model_id": model_id,
        "load_time_s": round(load_time, 1),
        "quantization": "streaming-int4",
        "group_size": group_size,
        "bf16_gb": round(total_bf16_bytes / 1e9, 1),
        "q4_gb": round(total_q4_bytes / 1e9, 1),
        "compression_ratio": round(overall_ratio, 1),
        "linears_quantized": total_quantized,
    }
    metadata["runtime"] = configure_quantized_runtime(
        model,
        gpu_memory_gb=gpu_memory_gb,
    )

    return model, processor, metadata


def configure_quantized_runtime(
    model: nn.Module,
    gpu_memory_gb: float = 3.5,
    resident_gpu_fraction: float = 0.45,
    expert_cache_fraction: float = 0.25,
    reserve_gpu_fraction: float = 0.15,
    cpu_linear_cache_gb: float = 4.0,
    pin_memory: bool = False,
    prefetch: bool = True,
) -> dict[str, Any]:
    """Configure GPU residency for a streaming-int4 model.

    Splits the available GPU memory into:
      - persistent layer residency for a contiguous decoder prefix
      - a shared hot-expert cache for CPU-resident MoE blocks
      - workspace / KV headroom
      - a fixed reserve to avoid fragmentation-driven OOMs
    """
    import re

    existing = getattr(model, "_streaming_runtime", None)
    if existing is not None:
        return existing

    total_fraction = resident_gpu_fraction + expert_cache_fraction + reserve_gpu_fraction
    if total_fraction > 1.0:
        scale = 1.0 / total_fraction
        resident_gpu_fraction *= scale
        expert_cache_fraction *= scale
        reserve_gpu_fraction *= scale

    total_budget_bytes = int(gpu_memory_gb * (1024**3))
    reserve_bytes = int(total_budget_bytes * reserve_gpu_fraction)
    resident_bytes = int(total_budget_bytes * resident_gpu_fraction)
    expert_cache_bytes = int(total_budget_bytes * expert_cache_fraction)
    workspace_bytes = max(
        total_budget_bytes - reserve_bytes - resident_bytes - expert_cache_bytes,
        0,
    )

    if not torch.cuda.is_available():
        runtime = {
            "enabled": False,
            "gpu_budget_mb": round(total_budget_bytes / 1e6, 1),
            "resident_budget_mb": round(resident_bytes / 1e6, 1),
            "expert_cache_budget_mb": round(expert_cache_bytes / 1e6, 1),
            "workspace_budget_mb": round(workspace_bytes / 1e6, 1),
            "reserve_budget_mb": round(reserve_bytes / 1e6, 1),
            "layers_on_gpu": 0,
            "linear_cache": {
                "enabled": False,
                "eligible_modules": 0,
            },
            "moe_offload": {
                "enabled": False,
                "cpu_modules": 0,
                "cache_size": 0,
            },
        }
        model._streaming_runtime = runtime
        return runtime

    from .loader import _move_layers_to_gpu
    from .moe_offload import ExpertOffloader

    all_expert_modules: list[tuple[int, QuantizedMoEExperts]] = []
    packed_expert_size_bytes = 0
    dequant_expert_size_bytes = 0
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedMoEExperts):
            continue

        match = re.search(r"layers\.(\d+)\.experts$", name)
        if match is None:
            continue

        first_tensor = getattr(module, "gu_packed_0", None)
        if first_tensor is None or first_tensor.device.type == "meta":
            continue

        layer_idx = int(match.group(1))
        all_expert_modules.append((layer_idx, module))
        packed_expert_size_bytes = max(packed_expert_size_bytes, module.expert_storage_bytes())
        dequant_expert_size_bytes = max(
            dequant_expert_size_bytes,
            module.expert_dequantized_bytes(),
        )

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    model_top_k_experts = int(
        getattr(text_config, "top_k_experts", getattr(getattr(model, "config", None), "top_k_experts", 0))
        or 0
    )
    try:
        packed_cache_override = int(os.environ.get("AVA_PACKED_EXPERT_TARGET", "0") or 0)
    except ValueError:
        packed_cache_override = 0
    packed_cache_target_entries = max(6, model_top_k_experts, packed_cache_override)
    if all_expert_modules and packed_expert_size_bytes > 0 and dequant_expert_size_bytes > 0:
        dequant_cache_size = (
            expert_cache_bytes // dequant_expert_size_bytes // len(all_expert_modules)
            if dequant_expert_size_bytes > 0
            else 0
        )
        packed_cache_size = (
            expert_cache_bytes // packed_expert_size_bytes // len(all_expert_modules)
            if packed_expert_size_bytes > 0
            else 0
        )
        if dequant_cache_size < 8 and packed_cache_size > packed_cache_target_entries:
            capped_expert_cache_bytes = (
                packed_cache_target_entries
                * len(all_expert_modules)
                * packed_expert_size_bytes
            )
            reclaimed_bytes = max(expert_cache_bytes - capped_expert_cache_bytes, 0)
            expert_cache_bytes = capped_expert_cache_bytes
            resident_bytes += reclaimed_bytes
            workspace_bytes = max(
                total_budget_bytes - reserve_bytes - resident_bytes - expert_cache_bytes,
                0,
            )

    runtime = {
        "enabled": True,
        "gpu_budget_mb": round(total_budget_bytes / 1e6, 1),
        "resident_budget_mb": round(resident_bytes / 1e6, 1),
        "expert_cache_budget_mb": round(expert_cache_bytes / 1e6, 1),
        "workspace_budget_mb": round(workspace_bytes / 1e6, 1),
        "reserve_budget_mb": round(reserve_bytes / 1e6, 1),
        "layers_on_gpu": 0,
        "linear_cache": {
            "enabled": False,
            "eligible_modules": 0,
        },
        "cpu_linear_cache": {
            "enabled": False,
            "eligible_modules": 0,
        },
        "moe_offload": {
            "enabled": False,
            "cpu_modules": 0,
            "cache_size": 0,
            "required_top_k_entries": model_top_k_experts,
            "packed_cache_target_entries": packed_cache_target_entries,
        },
    }

    _move_layers_to_gpu(model, gpu_budget_bytes=resident_bytes)
    runtime["layers_on_gpu"] = getattr(model, "_manual_gpu_layers", 0)
    resident_used_bytes = int(getattr(model, "_manual_gpu_used_bytes", 0))
    resident_remaining_bytes = max(resident_bytes - resident_used_bytes, 0)

    compute_dtype = torch.bfloat16
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type == "meta" or not tensor.is_floating_point():
            continue
        compute_dtype = tensor.dtype
        break
    gpu_compute_dtype = torch.float16 if torch.cuda.is_available() else compute_dtype
    runtime["gpu_compute_dtype"] = str(gpu_compute_dtype).replace("torch.", "")
    dynamic_workspace_reserve_bytes = 96 * 1024**2
    usable_linear_cache_bytes = max(workspace_bytes - dynamic_workspace_reserve_bytes, 0)
    runtime["workspace_dynamic_reserve_mb"] = round(dynamic_workspace_reserve_bytes / 1e6, 1)

    linear_modules = 0
    linear_working_set_bytes = 0
    eligible_linears: list[QuantizedLinear] = []
    cpu_linears: list[QuantizedLinear] = []
    cpu_linear_working_set_bytes = 0
    for module in model.modules():
        if not isinstance(module, QuantizedLinear):
            continue
        if module.packed.device.type != "cuda":
            cpu_linears.append(module)
            cpu_linear_working_set_bytes += module.dequantized_weight_bytes()
            continue
        eligible_linears.append(module)
        linear_modules += 1
        linear_working_set_bytes += module.dequantized_weight_bytes()

    base_linear_working_set_bytes = linear_working_set_bytes
    extra_gpu_linears = 0
    extra_linear_cache_bytes = 0
    if resident_remaining_bytes > 0 and usable_linear_cache_bytes > linear_working_set_bytes:
        linear_cache_headroom = usable_linear_cache_bytes - linear_working_set_bytes
        cpu_linears.sort(
            key=lambda module: module.dequantized_weight_bytes(),
            reverse=True,
        )

        for module in cpu_linears:
            packed_bytes = module.packed_storage_bytes()
            dequant_bytes = module.dequantized_weight_bytes()
            if packed_bytes > resident_remaining_bytes or dequant_bytes > linear_cache_headroom:
                continue
            module.to(torch.device("cuda:0"))
            eligible_linears.append(module)
            linear_modules += 1
            linear_working_set_bytes += dequant_bytes
            cpu_linear_working_set_bytes -= dequant_bytes
            resident_remaining_bytes -= packed_bytes
            linear_cache_headroom -= dequant_bytes
            extra_linear_cache_bytes += dequant_bytes
            extra_gpu_linears += 1

    linear_cache_bytes = min(
        usable_linear_cache_bytes,
        base_linear_working_set_bytes + extra_linear_cache_bytes,
    )
    linear_cache = DequantWeightCache(max_bytes=linear_cache_bytes)
    for module in eligible_linears:
        module.configure_dequant_cache(linear_cache, compute_dtype=gpu_compute_dtype)

    if linear_modules and linear_cache_bytes > 0:
        runtime["linear_cache"] = {
            "enabled": True,
            "eligible_modules": linear_modules,
            "budget_mb": round(linear_cache_bytes / 1e6, 1),
            "working_set_mb": round(linear_working_set_bytes / 1e6, 1),
            "extra_gpu_modules": extra_gpu_linears,
            **linear_cache.summary(),
        }
        model._linear_dequant_cache = linear_cache

    cpu_linears = [module for module in cpu_linears if module.packed.device.type == "cpu"]
    cpu_linear_cache_bytes = min(
        int(cpu_linear_cache_gb * (1024**3)),
        cpu_linear_working_set_bytes,
    )
    if cpu_linears and cpu_linear_cache_bytes > 0:
        cpu_linear_cache = DequantWeightCache(max_bytes=cpu_linear_cache_bytes)
        for module in cpu_linears:
            module.configure_dequant_cache(cpu_linear_cache)
        runtime["cpu_linear_cache"] = {
            "enabled": True,
            "eligible_modules": len(cpu_linears),
            "budget_mb": round(cpu_linear_cache_bytes / 1e6, 1),
            "working_set_mb": round(cpu_linear_working_set_bytes / 1e6, 1),
            **cpu_linear_cache.summary(),
        }
        model._cpu_linear_dequant_cache = cpu_linear_cache

    expert_modules = [
        (layer_idx, module)
        for layer_idx, module in all_expert_modules
        if getattr(module, "gu_packed_0", None) is not None
        and getattr(module, "gu_packed_0").device.type != "cuda"
    ]

    if expert_modules and dequant_expert_size_bytes > 0 and expert_cache_bytes > 0:
        cache_layout = "dequantized"
        cache_entry_bytes = dequant_expert_size_bytes
        dequant_cache_size = (
            expert_cache_bytes // dequant_expert_size_bytes // len(expert_modules)
            if dequant_expert_size_bytes > 0
            else 0
        )
        packed_cache_size = (
            expert_cache_bytes // packed_expert_size_bytes // len(expert_modules)
            if packed_expert_size_bytes > 0
            else 0
        )
        # If we cannot even hold one token's top-k experts per layer, prefer
        # packed residency and dequantize on use so the hot set is much larger.
        if packed_expert_size_bytes > 0 and dequant_cache_size < 8 and packed_cache_size > dequant_cache_size:
            cache_layout = "packed"
            cache_entry_bytes = packed_expert_size_bytes

        total_cache_entries = expert_cache_bytes // cache_entry_bytes
        cache_size = max(1, total_cache_entries // len(expert_modules)) if total_cache_entries else 0
        if cache_layout == "packed" and packed_cache_target_entries > 0:
            cache_size = min(cache_size, packed_cache_target_entries)

        if cache_size > 0:
            offloaders = []
            expert_weight_cache = None
            aggregate_hits = 0
            aggregate_misses = 0
            aggregate_evictions = 0
            aggregate_transfer_mb = 0.0
            packed_cache_total_bytes = cache_size * len(expert_modules) * cache_entry_bytes
            expert_weight_cache_bytes = max(expert_cache_bytes - packed_cache_total_bytes, 0)
            dequant_weight_entries = (
                expert_weight_cache_bytes // dequant_expert_size_bytes
                if dequant_expert_size_bytes > 0
                else 0
            )
            if cache_layout == "packed" and dequant_weight_entries >= 4:
                expert_weight_cache = ExpertWeightCache(expert_weight_cache_bytes)
            protected_cache_entries = 0
            if cache_size > 2:
                protected_cache_entries = min(
                    max(2, model_top_k_experts // 2),
                    max(cache_size - 1, 0),
                )

            for layer_idx, module in expert_modules:
                offloader = ExpertOffloader(
                    cache_size=cache_size,
                    cache_policy="lfu",
                    protected_cache_entries=protected_cache_entries,
                    pin_memory=pin_memory,
                    prefetch=prefetch,
                    gpu_device=torch.device("cuda:0"),
                )
                module.configure_gpu_offload(
                    layer_idx,
                    offloader,
                    gpu_device=torch.device("cuda:0"),
                    compute_dtype=gpu_compute_dtype,
                    cache_layout=cache_layout,
                    expert_weight_cache=expert_weight_cache,
                )
                offloaders.append(offloader)

                summary = offloader.summary()
                cache_stats = summary["cache_stats"]
                aggregate_hits += cache_stats["hits"]
                aggregate_misses += cache_stats["misses"]
                aggregate_evictions += cache_stats["evictions"]
                aggregate_transfer_mb += cache_stats["total_transfer_mb"]

            model._moe_offloaders = offloaders
            if expert_weight_cache is not None:
                model._expert_weight_cache = expert_weight_cache
            runtime["moe_offload"] = {
                "enabled": True,
                "cpu_modules": len(expert_modules),
                "cache_layout": cache_layout,
                "required_top_k_entries": model_top_k_experts,
                "packed_cache_target_entries": packed_cache_target_entries,
                "cache_size_per_module": cache_size,
                "protected_cache_entries": protected_cache_entries,
                "cache_entry_mb": round(cache_entry_bytes / 1e6, 3),
                "packed_entry_mb": round(packed_expert_size_bytes / 1e6, 3),
                "dequantized_entry_mb": round(dequant_expert_size_bytes / 1e6, 3),
                "estimated_total_cache_mb": round(
                    cache_size * len(expert_modules) * cache_entry_bytes / 1e6, 1,
                ),
                "dequant_weight_cache_mb": round(expert_weight_cache_bytes / 1e6, 1),
                "dequant_weight_cache_entries": dequant_weight_entries,
                "hits": aggregate_hits,
                "misses": aggregate_misses,
                "evictions": aggregate_evictions,
                "total_transfer_mb": round(aggregate_transfer_mb, 1),
            }
            if expert_weight_cache is not None:
                runtime["moe_offload"]["dequant_weight_cache"] = expert_weight_cache.summary()

    model._streaming_runtime = runtime
    return runtime


def _replace_linear_with_quantized(
    model: nn.Module,
    weight_name: str,
    qdata: dict[str, torch.Tensor],
    group_size: int,
) -> None:
    """Navigate the module tree and replace a Linear with QuantizedLinear."""
    parts = weight_name.split(".")
    # weight_name ends with .weight — parent is the Linear module
    if parts[-1] == "weight":
        parts = parts[:-1]

    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    child_name = parts[-1]
    old_module = getattr(module, child_name)

    # Get bias if present
    bias = None
    if isinstance(old_module, nn.Linear) and old_module.bias is not None:
        # Bias is on meta device, will be loaded separately
        bias = None  # biases loaded separately in the weight loop

    q_linear = QuantizedLinear(
        packed=qdata["packed"],
        scales=qdata["scales"],
        zeros=qdata["zeros"],
        original_in_features=qdata["original_in_features"],
        bias=bias,
        group_size=group_size,
    )
    setattr(module, child_name, q_linear)


def _materialize_meta_tensors(
    model: nn.Module,
    config: Any,
    dtype: torch.dtype,
) -> int:
    """Replace remaining meta-device tensors with real CPU tensors.

    After streaming weight loading, non-persistent buffers (computed in
    __init__, not saved in state_dict) remain on meta device.  This function
    re-instantiates the specific module types that create such buffers,
    which is more robust than trying to recompute values from config.
    """
    text_config = getattr(config, "text_config", config)
    fixed = 0

    # Strategy: find modules with meta buffers and re-instantiate them on CPU.
    # This runs their __init__ which correctly computes all non-persistent buffers.
    for name, module in list(model.named_modules()):
        has_meta = any(
            b is not None and b.device.type == "meta"
            for b in module._buffers.values()
        ) if hasattr(module, "_buffers") else False

        if not has_meta:
            continue

        cls_name = type(module).__name__

        # Re-instantiate RotaryEmbedding modules (inv_freq buffers)
        if "RotaryEmbedding" in cls_name:
            parent, attr = _get_parent_and_attr(model, name)
            if parent is not None:
                # Use the correct sub-config for the module type
                if "Vision" in cls_name:
                    sub_config = getattr(config, "vision_config", text_config)
                elif "Audio" in cls_name:
                    sub_config = getattr(config, "audio_config", text_config)
                else:
                    sub_config = text_config
                try:
                    new_module = type(module)(sub_config, device="cpu")
                    setattr(parent, attr, new_module)
                    fixed += sum(1 for _ in new_module._buffers)
                    continue
                except Exception:
                    pass  # Fall through to generic handler

        # Re-instantiate ScaledWordEmbedding (embed_scale buffer)
        if "ScaledWordEmbedding" in cls_name:
            # Preserve the loaded weight, just fix the buffer
            embed_scale = text_config.hidden_size ** 0.5
            module._buffers["embed_scale"] = torch.tensor(
                embed_scale, dtype=dtype, device="cpu",
            )
            fixed += 1
            continue

        # Generic fallback: replace meta buffers with zeros
        for buf_name, buf in list(module._buffers.items()):
            if buf is not None and buf.device.type == "meta":
                module._buffers[buf_name] = torch.zeros(
                    buf.shape, dtype=buf.dtype, device="cpu",
                )
                fixed += 1

    # Also fix any remaining meta parameters (shouldn't happen but safety net)
    for name, module in model.named_modules():
        for pname, param in list(module._parameters.items()):
            if param is not None and param.device.type == "meta":
                module._parameters[pname] = nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                    requires_grad=False,
                )
                fixed += 1

    return fixed


def _build_quantized_experts(
    model: nn.Module,
    expert_tensors: dict[str, torch.Tensor],
    quantizer: StreamingInt4Quantizer,
    config: Any,
) -> tuple[int, int]:
    """Quantize 3D expert weights and replace Experts modules with QuantizedMoEExperts.

    Args:
        model: the model (some params may be on meta)
        expert_tensors: dict mapping weight_name -> 3D tensor
        quantizer: the int4 quantizer
        config: model config (for act_fn)

    Returns:
        (total_q4_bytes, total_experts_quantized)
    """
    import re
    from transformers.activations import ACT2FN

    text_config = getattr(config, "text_config", config)
    act_fn = ACT2FN[text_config.hidden_activation]

    # Group by layer: {layer_idx: {weight_name: tensor}}
    by_layer: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    for wname, tensor in expert_tensors.items():
        m = re.search(r"layers\.(\d+)\.", wname)
        if m:
            by_layer[int(m.group(1))][wname] = tensor

    total_bytes = 0
    total_experts = 0

    for layer_idx, tensors in by_layer.items():
        gate_up = None
        down = None
        gate_up_name = None
        down_name = None

        for wname, tensor in tensors.items():
            if "gate_up" in wname:
                gate_up = tensor
                gate_up_name = wname
            elif "down_proj" in wname:
                down = tensor
                down_name = wname

        if gate_up is None or down is None:
            # Incomplete — assign as-is
            for wname, tensor in tensors.items():
                parts = wname.split(".")
                module = model
                for part in parts[:-1]:
                    module = module[int(part)] if part.isdigit() else getattr(module, part)
                param_name = parts[-1]
                setattr(module, param_name, nn.Parameter(tensor, requires_grad=False))
                total_bytes += tensor.nbytes
            continue

        num_experts = gate_up.shape[0]

        # Quantize each expert slice
        gu_packed, gu_scales, gu_zeros = [], [], []
        dn_packed, dn_scales, dn_zeros = [], [], []

        for e in range(num_experts):
            qd = quantizer.quantize(gate_up[e])
            gu_packed.append(qd["packed"])
            gu_scales.append(qd["scales"])
            gu_zeros.append(qd["zeros"])

            qd = quantizer.quantize(down[e])
            dn_packed.append(qd["packed"])
            dn_scales.append(qd["scales"])
            dn_zeros.append(qd["zeros"])

        # Calculate memory
        layer_bytes = sum(
            t.nbytes for lst in [gu_packed, gu_scales, gu_zeros, dn_packed, dn_scales, dn_zeros]
            for t in lst
        )
        total_bytes += layer_bytes
        total_experts += num_experts * 2  # gate_up + down per expert

        # Build QuantizedMoEExperts
        q_experts = QuantizedMoEExperts(
            num_experts=num_experts,
            gate_up_packed=gu_packed,
            gate_up_scales=gu_scales,
            gate_up_zeros=gu_zeros,
            gate_up_in_features=gate_up.shape[2],
            down_packed=dn_packed,
            down_scales=dn_scales,
            down_zeros=dn_zeros,
            down_in_features=down.shape[2],
            act_fn=act_fn,
            group_size=quantizer.group_size,
        )

        # Replace the experts module in the model
        # Navigate: model.language_model.layers.{layer_idx}.experts
        parent_path = gate_up_name.rsplit(".experts.", 1)[0] + ".experts"
        parent, attr = _get_parent_and_attr(model, parent_path)
        if parent is not None:
            # Delete original experts to free meta tensors
            setattr(parent, attr.split(".")[-1] if "." in attr else attr, q_experts)

        # Free original bf16 tensors
        del gate_up, down, gu_packed, gu_scales, gu_zeros, dn_packed, dn_scales, dn_zeros

    return total_bytes, total_experts


def _get_parent_and_attr(
    model: nn.Module,
    dotted_name: str,
) -> tuple[nn.Module | None, str]:
    """Get parent module and attribute name from dotted path."""
    parts = dotted_name.split(".")
    if len(parts) == 1:
        return model, parts[0]
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part, None)
            if parent is None:
                return None, ""
    return parent, parts[-1]


def _get_process_rss_gb() -> float:
    """Get current process RSS in GB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return 0.0


# ---------------------------------------------------------------------------
# Quantized weight cache — save/load quantized model to/from disk
# ---------------------------------------------------------------------------


def save_quantized(
    model: nn.Module,
    save_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    processor: Any | None = None,
) -> Path:
    """Save a streaming-quantized model to disk for fast reloading.

    Saves:
      - HuggingFace config (config.json etc.) for model reconstruction
      - processor/tokenizer files when available for offline reload
      - quantized_config.json — module map + metadata
      - Sharded safetensors files — all quantized buffers + bf16 params

    Args:
        model: quantized model from streaming_load()
        save_dir: directory to save into (created if needed)
        metadata: dict from streaming_load() (model_id, compression stats, etc.)
        processor: optional processor/tokenizer to save alongside the cache

    Returns:
        Path to save_dir.
    """
    from safetensors.torch import save_file

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving quantized model to {save_dir}")
    t0 = time.perf_counter()

    # 1. Save HF config for model reconstruction
    model.config.save_pretrained(str(save_dir))
    if processor is not None:
        processor.save_pretrained(str(save_dir))

    # 2. Build module map — records which modules are quantized and their config
    module_map: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            module_map[name] = {
                "type": "QuantizedLinear",
                "original_in_features": module.original_in_features,
                "group_size": module.group_size,
                "has_bias": module.bias is not None,
            }
        elif isinstance(module, QuantizedMoEExperts):
            module_map[name] = {
                "type": "QuantizedMoEExperts",
                "num_experts": module.num_experts,
                "gate_up_in_features": module.gate_up_in_features,
                "down_in_features": module.down_in_features,
                "group_size": module.group_size,
            }

    config_data = {
        "format": "ava-streaming-int4",
        "version": 1,
        "module_map": module_map,
        **(metadata or {}),
    }
    with open(save_dir / "quantized_config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    # 3. Save state dict as sharded safetensors
    state_dict = model.state_dict()

    # Split into shards (~4 GB each to stay safe with mmap on 32 GB systems)
    max_shard_bytes = 4 * 1024**3
    shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_bytes = 0
    weight_map: dict[str, str] = {}

    for key, tensor in state_dict.items():
        nbytes = tensor.nbytes
        if current_bytes + nbytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_bytes = 0
        current_shard[key] = tensor
        current_bytes += nbytes

    if current_shard:
        shards.append(current_shard)

    n_shards = len(shards)
    total_bytes = 0
    for i, shard in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        shard_path = save_dir / shard_name
        save_file(shard, str(shard_path))
        shard_bytes = sum(t.nbytes for t in shard.values())
        total_bytes += shard_bytes
        print(f"  Shard {i + 1}/{n_shards}: {len(shard)} tensors, "
              f"{shard_bytes / 1e9:.1f} GB -> {shard_path.name}")
        for key in shard:
            weight_map[key] = shard_name

    # Save safetensors index
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    with open(save_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"  Saved in {elapsed:.1f}s: {n_shards} shards, "
          f"{total_bytes / 1e9:.1f} GB, {len(state_dict)} tensors")

    return save_dir


def load_quantized(
    save_dir: str | Path,
    dtype: str = "bfloat16",
    gpu_memory_gb: float | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a previously saved quantized model from disk.

    Much faster than streaming_load() — reads pre-quantized int4 weights
    directly from safetensors instead of re-quantizing from bf16.

    Args:
        save_dir: directory from save_quantized()
        dtype: dtype for non-quantized tensors
        gpu_memory_gb: optional runtime GPU budget for layer residency and
            hot-expert caching after reload

    Returns:
        model, processor, metadata
    """
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    save_dir = Path(save_dir)
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    print(f"Loading quantized model from {save_dir}")
    t0 = time.perf_counter()

    # 1. Read configs
    with open(save_dir / "quantized_config.json") as f:
        config_data = json.load(f)
    module_map = config_data["module_map"]

    with open(save_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # 2. Create model shell on meta device
    print("  Creating model on meta device...")
    config = AutoConfig.from_pretrained(str(save_dir))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    model.eval()

    # 3. Replace modules with quantized versions (meta placeholders)
    print(f"  Replacing {len(module_map)} modules with quantized versions...")
    _replace_modules_from_map(model, module_map, config)

    # 4. Load tensors from safetensors shards one-by-one (memory-efficient)
    shard_files = sorted(set(weight_map.values()))
    print(f"  Loading weights from {len(shard_files)} shards...")

    loaded = 0
    for shard_name in shard_files:
        shard_path = save_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                _assign_state_dict_tensor(model, key, tensor)
                loaded += 1
        ram = _get_process_rss_gb()
        print(f"    {shard_name}: loaded, RAM: {ram:.1f} GB")

    # 5. Materialize non-persistent meta buffers
    meta_fixed = _materialize_meta_tensors(model, config, torch_dtype)
    if meta_fixed:
        print(f"  Materialized {meta_fixed} non-persistent buffers")

    # 6. Re-tie weights
    model.tie_weights()

    # 7. Load processor. Prefer files saved inside the quantized cache so the
    # reload path stays fully local. Fall back to the original model ID, but
    # still require local cached files instead of network access.
    model_id = config_data.get("model_id", str(save_dir))
    processor = _load_quantized_processor(save_dir, model_id)

    elapsed = time.perf_counter() - t0
    print(f"  Loaded in {elapsed:.1f}s ({loaded} tensors, RAM: {_get_process_rss_gb():.1f} GB)")

    if gpu_memory_gb is not None:
        config_data["runtime"] = configure_quantized_runtime(
            model,
            gpu_memory_gb=gpu_memory_gb,
        )

    return model, processor, config_data


def _load_quantized_processor(save_dir: Path, model_id: str) -> Any:
    """Load processor assets for a quantized cache without network access."""
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    processor_files = [
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.json",
        "preprocessor_config.json",
    ]
    has_local_processor = any((save_dir / name).exists() for name in processor_files)

    if has_local_processor:
        return AutoProcessor.from_pretrained(str(save_dir), local_files_only=True)

    try:
        local_snapshot = snapshot_download(model_id, local_files_only=True)
        return AutoProcessor.from_pretrained(local_snapshot, local_files_only=True)
    except OSError as exc:
        raise OSError(
            f"Processor assets are not available locally for {model_id}. "
            f"Save the processor into {save_dir} by rebuilding the quantized cache."
        ) from exc


def _replace_modules_from_map(
    model: nn.Module,
    module_map: dict[str, dict[str, Any]],
    config: Any,
) -> None:
    """Replace model modules with quantized versions using meta placeholders.

    Creates QuantizedLinear/QuantizedMoEExperts with empty meta tensors.
    The actual data is loaded later via _assign_state_dict_tensor().
    """
    from transformers.activations import ACT2FN

    text_config = getattr(config, "text_config", config)
    hidden_act = getattr(text_config, "hidden_activation", "gelu_pytorch_tanh")
    # ACT2FN is a ClassInstantier — must use [] not .get() to auto-instantiate
    act_fn = ACT2FN[hidden_act]

    for dotted_name, info in module_map.items():
        parent, attr = _get_parent_and_attr(model, dotted_name)
        if parent is None:
            continue

        if info["type"] == "QuantizedLinear":
            # Create with dummy meta tensors — shapes don't matter with assign=True
            q_linear = QuantizedLinear(
                packed=torch.empty(1, 1, device="meta", dtype=torch.uint8),
                scales=torch.empty(1, 1, device="meta", dtype=torch.float16),
                zeros=torch.empty(1, 1, device="meta", dtype=torch.float16),
                original_in_features=info["original_in_features"],
                bias=torch.empty(1, device="meta", dtype=torch.float16) if info.get("has_bias") else None,
                group_size=info["group_size"],
            )
            setattr(parent, attr, q_linear)

        elif info["type"] == "QuantizedMoEExperts":
            n = info["num_experts"]
            meta_t = torch.empty(1, 1, device="meta", dtype=torch.uint8)
            meta_s = torch.empty(1, 1, device="meta", dtype=torch.float16)

            q_experts = QuantizedMoEExperts(
                num_experts=n,
                gate_up_packed=[meta_t] * n,
                gate_up_scales=[meta_s] * n,
                gate_up_zeros=[meta_s] * n,
                gate_up_in_features=info["gate_up_in_features"],
                down_packed=[meta_t] * n,
                down_scales=[meta_s] * n,
                down_zeros=[meta_s] * n,
                down_in_features=info["down_in_features"],
                act_fn=act_fn,
                group_size=info["group_size"],
            )
            setattr(parent, attr, q_experts)


def _assign_state_dict_tensor(
    model: nn.Module,
    key: str,
    tensor: torch.Tensor,
) -> None:
    """Assign a single tensor from the state dict to the model.

    Handles both parameters and buffers, replacing meta placeholders.
    """
    parts = key.split(".")
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    attr = parts[-1]

    # Check if it's a parameter
    if attr in module._parameters:
        module._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
    elif attr in module._buffers:
        module._buffers[attr] = tensor
    else:
        setattr(module, attr, tensor)
