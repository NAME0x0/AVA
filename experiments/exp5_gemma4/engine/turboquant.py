"""TurboQuant V3 — KV cache compression for long-context inference.

Implements asymmetric key/value quantization using random rotation +
Lloyd-Max scalar quantization.  MSE-only design (no QJL residual) based
on community findings that MSE outperforms MSE+QJL after softmax.

Key design choices (following the TurboQuant V3 community consensus):
  - Keys get more bits than values (K4/V2 default)
  - First and last global-attention layers are protected at full precision
  - Per-group quantization with configurable group size (default 128)
  - Bit-packed storage for genuine compression

Reference:
  TurboQuant (ICLR 2026) — https://openreview.net/pdf/6593f484501e295cdbe7efcbc46d7f20fc7e741f
  tonbistudio/turboquant-pytorch — https://github.com/tonbistudio/turboquant-pytorch
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack quantized indices into bit-packed uint8 storage.

    Args:
        indices: uint8 tensor with values in [0, 2^bits - 1]
        bits: number of bits per index (1, 2, 4, or 8)

    Returns:
        Packed uint8 tensor — 8/bits indices per byte.
    """
    if bits == 8:
        return indices
    if bits not in (1, 2, 4):
        raise ValueError(f"Bit-packing supports 1, 2, 4, 8 bits, got {bits}")

    flat = indices.reshape(-1)
    n = flat.shape[0]
    per_byte = 8 // bits

    # Pad to multiple of per_byte
    remainder = n % per_byte
    if remainder:
        flat = torch.nn.functional.pad(flat, (0, per_byte - remainder))

    grouped = flat.reshape(-1, per_byte)  # (N/per_byte, per_byte)

    # Shift each element by its bit position and OR together
    packed = torch.zeros(grouped.shape[0], dtype=torch.uint8, device=indices.device)
    for i in range(per_byte):
        shift = bits * (per_byte - 1 - i)
        packed |= grouped[:, i].to(torch.uint8) << shift

    return packed


def unpack_indices(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """Unpack bit-packed uint8 storage back to individual indices.

    Args:
        packed: bit-packed uint8 tensor from pack_indices()
        bits: number of bits per index (1, 2, 4, or 8)
        count: number of original indices to recover

    Returns:
        uint8 tensor of unpacked indices with shape (count,).
    """
    if bits == 8:
        return packed[:count]
    if bits not in (1, 2, 4):
        raise ValueError(f"Bit-unpacking supports 1, 2, 4, 8 bits, got {bits}")

    per_byte = 8 // bits
    mask = (1 << bits) - 1

    # Extract each sub-byte field
    parts = []
    for i in range(per_byte):
        shift = bits * (per_byte - 1 - i)
        parts.append((packed >> shift) & mask)

    # Interleave: byte0_field0, byte0_field1, ..., byte1_field0, ...
    unpacked = torch.stack(parts, dim=1).reshape(-1)
    return unpacked[:count].to(torch.uint8)


class RotationMatrix:
    """Generates a deterministic random orthogonal matrix from a seed.

    Uses the QR decomposition of a random Gaussian matrix to produce
    a uniformly-distributed orthogonal rotation.  The matrix is cached
    per dimension so it's computed once per model load.
    """

    _cache: dict[tuple[int, int], torch.Tensor] = {}

    @classmethod
    def get(cls, dim: int, seed: int, device: torch.device) -> torch.Tensor:
        key = (dim, seed)
        if key not in cls._cache:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            random_matrix = torch.randn(dim, dim, generator=gen)
            q, r = torch.linalg.qr(random_matrix)
            # Ensure det(Q) = +1 (proper rotation)
            diag_sign = torch.sign(torch.diag(r))
            q = q * diag_sign.unsqueeze(0)
            cls._cache[key] = q
        return cls._cache[key].to(device)


class MSECompressor:
    """Scalar quantizer using random rotation + Lloyd-Max centroids.

    Pipeline:
      1. Record the L2 norm of each vector (for reconstruction)
      2. Normalize to unit sphere
      3. Multiply by random orthogonal matrix (decorrelates dimensions)
      4. Quantize each scalar independently using uniform grid
         (optimal for near-Gaussian rotated coordinates)
      5. Store quantized indices + norms

    Dequantization reverses the process: indices → centroids → R^T → scale by norm.
    """

    def __init__(self, bits: int, group_size: int = 128, seed: int = 42, pack: bool = True):
        self.bits = bits
        self.group_size = group_size
        self.seed = seed
        self.pack = pack  # Whether to bit-pack indices
        self.n_levels = 2**bits
        # Uniform grid centroids for standard-normal-like data after rotation
        # After orthogonal rotation of a normalized vector, coordinates are
        # approximately uniform in [-1/sqrt(d), 1/sqrt(d)], but we use a
        # data-adaptive range per group.
        self._centroid_cache: dict[int, torch.Tensor] = {}

    def _compute_centroids(self, n_levels: int, device: torch.device) -> torch.Tensor:
        """Uniform quantization grid in [-1, 1]."""
        if n_levels not in self._centroid_cache:
            self._centroid_cache[n_levels] = torch.linspace(
                -1.0, 1.0, n_levels, device=device
            )
        return self._centroid_cache[n_levels].to(device)

    def compress(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress a KV cache tensor.

        Args:
            x: shape (batch, heads, seq_len, head_dim) — the K or V cache

        Returns:
            indices: uint8 tensor of quantized indices
            norms: per-vector L2 norms
            mins: per-group minimums (for range reconstruction)
            scales: per-group scales
        """
        B, H, S, D = x.shape
        device = x.device
        dtype = x.dtype

        # Flatten to (B*H*S, D) for per-vector processing
        flat = x.reshape(-1, D).float()
        N = flat.shape[0]

        # Step 1: Record norms
        norms = flat.norm(dim=1, keepdim=True)  # (N, 1)
        # Avoid division by zero
        safe_norms = norms.clamp(min=1e-8)

        # Step 2: Normalize
        normalized = flat / safe_norms  # (N, D)

        # Step 3: Random rotation
        R = RotationMatrix.get(D, self.seed, device)
        rotated = normalized @ R  # (N, D)

        # Step 4: Group-wise quantization
        # Reshape into groups of `group_size` along the D dimension
        n_groups = math.ceil(D / self.group_size)
        padded_D = n_groups * self.group_size
        if padded_D > D:
            rotated = torch.nn.functional.pad(rotated, (0, padded_D - D))

        grouped = rotated.reshape(N, n_groups, self.group_size)  # (N, G, gs)

        # Per-group min/max for range
        g_min = grouped.min(dim=2, keepdim=True).values  # (N, G, 1)
        g_max = grouped.max(dim=2, keepdim=True).values
        g_range = (g_max - g_min).clamp(min=1e-8)
        g_scale = g_range / (self.n_levels - 1)

        # Quantize to [0, n_levels-1]
        normalized_grouped = (grouped - g_min) / g_scale
        indices = normalized_grouped.round().clamp(0, self.n_levels - 1).to(torch.uint8)

        # Pack norms, mins, scales for storage
        norms_out = norms.reshape(B, H, S, 1).to(dtype)
        mins_out = g_min.reshape(N, n_groups).to(dtype)
        scales_out = g_scale.reshape(N, n_groups).to(dtype)

        indices_shaped = indices.reshape(B, H, S, n_groups, self.group_size)

        if self.pack and self.bits < 8:
            # Bit-pack indices for real memory savings
            # Pack along the last dimension (group_size) within each group
            packed = pack_indices(
                indices_shaped.reshape(-1, self.group_size),
                self.bits,
            ).reshape(B, H, S, n_groups, -1)
            return packed, norms_out, mins_out, scales_out
        return indices_shaped, norms_out, mins_out, scales_out

    def decompress(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        mins: torch.Tensor,
        scales: torch.Tensor,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Decompress quantized KV cache back to full tensors.

        Args:
            indices: quantized indices from compress() (packed or unpacked).
                     Packed format is auto-detected from the last dimension
                     being smaller than group_size.
            norms: per-vector norms
            mins: per-group minimums
            scales: per-group scales
            original_shape: (B, H, S, D) of the original tensor

        Returns:
            Reconstructed tensor in the original shape and dtype
        """
        B, H, S, D = original_shape
        device = indices.device
        dtype = norms.dtype

        N = B * H * S
        n_groups = indices.shape[3]
        last_dim = indices.shape[4]

        # Detect bit-packed format: packed last dim = group_size * bits / 8
        is_packed = self.pack and self.bits < 8 and last_dim < self.group_size
        if is_packed:
            gs = self.group_size
            packed_flat = indices.reshape(-1)
            total_indices = N * n_groups * gs
            unpacked = unpack_indices(packed_flat, self.bits, total_indices)
            grouped = unpacked.float().reshape(N, n_groups, gs)
        else:
            gs = last_dim
            grouped = indices.float().reshape(N, n_groups, gs)
        mins_flat = mins.reshape(N, n_groups, 1)
        scales_flat = scales.reshape(N, n_groups, 1).float()
        mins_flat = mins_flat.float()

        dequantized = grouped * scales_flat + mins_flat  # (N, n_groups, gs)

        # Flatten back to (N, padded_D)
        rotated = dequantized.reshape(N, -1)
        # Trim padding
        padded_D = rotated.shape[1]
        if padded_D > D:
            rotated = rotated[:, :D]

        # Inverse rotation
        R = RotationMatrix.get(D, self.seed, device)
        normalized = rotated @ R.T  # (N, D)

        # Rescale by norms
        norms_flat = norms.reshape(N, 1).float()
        reconstructed = normalized * norms_flat

        return reconstructed.reshape(B, H, S, D).to(dtype)


class TurboQuantV3(nn.Module):
    """Orchestrates asymmetric K/V cache compression across layers.

    Usage:
        tq = TurboQuantV3(key_bits=4, value_bits=2)

        # During generation, compress KV cache:
        compressed_k = tq.compress_keys(key_states, layer_idx)
        compressed_v = tq.compress_values(value_states, layer_idx)

        # When needed for attention:
        key_states = tq.decompress_keys(compressed_k, layer_idx, original_shape)
        value_states = tq.decompress_values(compressed_v, layer_idx, original_shape)
    """

    def __init__(
        self,
        key_bits: int = 4,
        value_bits: int = 2,
        group_size: int = 128,
        protected_layers: int = 1,
        total_global_layers: int = 5,
        seed: int = 42,
        pack: bool = True,
    ):
        super().__init__()
        self.key_compressor = MSECompressor(key_bits, group_size, seed, pack=pack)
        self.value_compressor = MSECompressor(value_bits, group_size, seed + 1, pack=pack)
        self.protected_layers = protected_layers
        self.total_global_layers = total_global_layers
        # Global layer indices in Gemma 4 26B: [5, 11, 17, 23, 29]
        self.global_layer_indices = [5, 11, 17, 23, 29]

    def _is_protected(self, global_layer_idx: int) -> bool:
        """Check if this global layer should be kept at full precision."""
        # Map global_layer_idx to position among global layers
        if global_layer_idx not in self.global_layer_indices:
            return True  # Not a global layer — shouldn't be compressed anyway
        pos = self.global_layer_indices.index(global_layer_idx)
        n = self.protected_layers
        return pos < n or pos >= (self.total_global_layers - n)

    def compress_keys(
        self, keys: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Compress key cache for a given layer."""
        if self._is_protected(layer_idx):
            return keys  # Return uncompressed
        return self.key_compressor.compress(keys)

    def compress_values(
        self, values: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Compress value cache for a given layer."""
        if self._is_protected(layer_idx):
            return values
        return self.value_compressor.compress(values)

    def decompress_keys(
        self,
        compressed: tuple[torch.Tensor, ...] | torch.Tensor,
        layer_idx: int,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Decompress key cache."""
        if isinstance(compressed, torch.Tensor):
            return compressed  # Was not compressed (protected layer)
        indices, norms, mins, scales = compressed
        return self.key_compressor.decompress(indices, norms, mins, scales, original_shape)

    def decompress_values(
        self,
        compressed: tuple[torch.Tensor, ...] | torch.Tensor,
        layer_idx: int,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Decompress value cache."""
        if isinstance(compressed, torch.Tensor):
            return compressed
        indices, norms, mins, scales = compressed
        return self.value_compressor.decompress(indices, norms, mins, scales, original_shape)

    def estimate_compression_ratio(self, head_dim: int) -> dict[str, float]:
        """Estimate compression ratios for reporting."""
        original_bits = 16  # BF16
        n_groups = math.ceil(head_dim / self.key_compressor.group_size)

        # Per-element overhead: index bits + amortized norm/min/scale
        # Norm: 16 bits per vector (amortized over head_dim elements)
        # Min + Scale: 16 bits each per group (amortized over group_size elements)
        k_overhead = (32 / head_dim) + (32 / self.key_compressor.group_size)
        v_overhead = (32 / head_dim) + (32 / self.value_compressor.group_size)

        k_bits = self.key_compressor.bits + k_overhead
        v_bits = self.value_compressor.bits + v_overhead

        return {
            "key_compression": original_bits / k_bits,
            "value_compression": original_bits / v_bits,
            "average_compression": original_bits / ((k_bits + v_bits) / 2),
            "key_bits_effective": k_bits,
            "value_bits_effective": v_bits,
        }
