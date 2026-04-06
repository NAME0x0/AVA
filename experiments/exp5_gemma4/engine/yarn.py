"""YaRN (Yet Another RoPE Extension) for extending Gemma 4 context to 1M+.

Gemma 4 26B MoE uses proportional RoPE (p-RoPE) on its 5 global attention
layers with rope_theta=1e6 and partial_rotary_factor=0.25.  The sliding
attention layers use standard RoPE with theta=10000.

YaRN modifies only the global attention layers' RoPE to support positions
beyond the original 256K training window.  The key insight is that different
frequency bands of RoPE need different treatment:
  - Low frequencies (long wavelengths): scale linearly — these already
    generalize well to longer contexts
  - High frequencies (short wavelengths): don't scale — these capture
    local patterns that shouldn't change
  - Mid frequencies: interpolate smoothly between the two

This implementation follows the YaRN paper (arXiv:2309.00071) adapted for
Gemma 4's proportional RoPE.

Reference:
  YaRN: Efficient Context Window Extension of Large Language Models
  https://arxiv.org/abs/2309.00071
"""
from __future__ import annotations

import math

import torch


def compute_yarn_frequencies(
    head_dim: int,
    original_max_pos: int = 262_144,
    target_max_pos: int = 1_048_576,
    rope_theta: float = 1_000_000.0,
    partial_rotary_factor: float = 0.25,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, float]:
    """Compute YaRN-modified inverse frequencies for global attention layers.

    Args:
        head_dim: dimension of each attention head (512 for Gemma 4 global)
        original_max_pos: original max position (256K for Gemma 4)
        target_max_pos: target max position (1M)
        rope_theta: base frequency (1e6 for Gemma 4 global layers)
        partial_rotary_factor: fraction of head_dim that uses RoPE (0.25)
        beta_fast: upper bound for frequency ramp (default 32)
        beta_slow: lower bound for frequency ramp (default 1)
        device: torch device

    Returns:
        inv_freq: modified inverse frequencies, shape (rotary_dim // 2,)
        attention_scale: multiplicative factor for attention logits
    """
    scale_factor = target_max_pos / original_max_pos  # 4.0 for 256K→1M

    # Only the first `partial_rotary_factor * head_dim` dims use RoPE
    rotary_dim = int(head_dim * partial_rotary_factor)

    # Base inverse frequencies
    base_inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    )

    # Compute wavelengths for each frequency
    wavelengths = 2 * math.pi / base_inv_freq  # in token positions

    # Low/high frequency boundaries (in wavelength space)
    low_freq_wavelen = original_max_pos / beta_fast
    high_freq_wavelen = original_max_pos / beta_slow

    # Smooth ramp: 0 for high-freq (don't scale), 1 for low-freq (scale fully)
    ramp = (wavelengths - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    ramp = ramp.clamp(0.0, 1.0)

    # Interpolate: smoothly blend between NTK-scaled and original frequencies
    # For fully-scaled (ramp=1): divide freq by scale_factor (NTK-aware interpolation)
    # For not-scaled (ramp=0): keep original frequency
    scaled_inv_freq = base_inv_freq / scale_factor
    yarn_inv_freq = base_inv_freq * (1 - ramp) + scaled_inv_freq * ramp

    # Attention temperature scaling (from YaRN paper, eq. 20)
    # sqrt(1 / scale_factor) * (0.1 * ln(scale_factor) + 1)
    t = 0.1 * math.log(scale_factor) + 1.0
    attention_scale = t / math.sqrt(scale_factor)

    return yarn_inv_freq, attention_scale


def build_yarn_cos_sin_cache(
    head_dim: int,
    max_seq_len: int = 1_048_576,
    original_max_pos: int = 262_144,
    rope_theta: float = 1_000_000.0,
    partial_rotary_factor: float = 0.25,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build precomputed cos/sin tables for YaRN-extended RoPE.

    Returns:
        cos_cache: shape (max_seq_len, rotary_dim // 2)
        sin_cache: shape (max_seq_len, rotary_dim // 2)
        attention_scale: float multiplier for attention logits
    """
    inv_freq, attention_scale = compute_yarn_frequencies(
        head_dim=head_dim,
        original_max_pos=original_max_pos,
        target_max_pos=max_seq_len,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        device=device,
    )

    # Position indices
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

    # Outer product: (seq_len, rotary_dim // 2)
    freqs = torch.outer(positions, inv_freq)

    cos_cache = freqs.cos().to(dtype)
    sin_cache = freqs.sin().to(dtype)

    return cos_cache, sin_cache, attention_scale


def apply_yarn_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply YaRN-modified rotary position embeddings to Q and K.

    Standard RoPE application but using YaRN-modified cos/sin tables.

    Args:
        q: query tensor, shape (..., head_dim)
        k: key tensor, shape (..., head_dim)
        cos: cos cache from build_yarn_cos_sin_cache
        sin: sin cache from build_yarn_cos_sin_cache
        position_ids: position indices, shape (batch, seq_len)

    Returns:
        q_rotated, k_rotated with the same shapes as inputs
    """
    rotary_dim = cos.shape[-1] * 2

    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Gather cos/sin for the given positions
    # position_ids: (B, S) → (B, S, rotary_dim//2)
    cos_pos = cos[position_ids]  # (B, S, rotary_dim//2)
    sin_pos = sin[position_ids]

    # Expand for heads dimension
    # q_rot: (B, H, S, rotary_dim), cos_pos: (B, S, rotary_dim//2)
    cos_pos = cos_pos.unsqueeze(1)  # (B, 1, S, rotary_dim//2)
    sin_pos = sin_pos.unsqueeze(1)

    # Split rotary dims into pairs
    q1, q2 = q_rot[..., ::2], q_rot[..., 1::2]
    k1, k2 = k_rot[..., ::2], k_rot[..., 1::2]

    # Apply rotation
    q_rot_out = torch.cat([q1 * cos_pos - q2 * sin_pos,
                           q2 * cos_pos + q1 * sin_pos], dim=-1)
    k_rot_out = torch.cat([k1 * cos_pos - k2 * sin_pos,
                           k2 * cos_pos + k1 * sin_pos], dim=-1)

    # Concatenate with pass-through
    q_out = torch.cat([q_rot_out, q_pass], dim=-1)
    k_out = torch.cat([k_rot_out, k_pass], dim=-1)

    return q_out, k_out


class YarnContextExtender:
    """Manages YaRN context extension for Gemma 4 global attention layers.

    Usage:
        extender = YarnContextExtender(target_context=1_048_576)

        # Monkey-patch the model's global attention layers
        extender.patch_model(model)

        # Or get cos/sin caches for manual use
        cos, sin, scale = extender.get_caches(device)
    """

    # Gemma 4 26B MoE global layer indices
    GLOBAL_LAYERS = [5, 11, 17, 23, 29]

    def __init__(
        self,
        target_context: int = 1_048_576,
        original_context: int = 262_144,
        # Gemma 4 global attention config
        head_dim: int = 512,
        rope_theta: float = 1_000_000.0,
        partial_rotary_factor: float = 0.25,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        self.target_context = target_context
        self.original_context = original_context
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None
        self._attention_scale: float | None = None

    def get_caches(
        self, device: torch.device, dtype: torch.dtype = torch.bfloat16
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Get or build the cos/sin caches."""
        if self._cos_cache is None:
            self._cos_cache, self._sin_cache, self._attention_scale = (
                build_yarn_cos_sin_cache(
                    head_dim=self.head_dim,
                    max_seq_len=self.target_context,
                    original_max_pos=self.original_context,
                    rope_theta=self.rope_theta,
                    partial_rotary_factor=self.partial_rotary_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                    device=device,
                    dtype=dtype,
                )
            )
        return self._cos_cache.to(device), self._sin_cache.to(device), self._attention_scale

    @property
    def scale_factor(self) -> float:
        return self.target_context / self.original_context

    def summary(self) -> dict[str, object]:
        """Return a summary of the context extension configuration."""
        return {
            "original_context": self.original_context,
            "target_context": self.target_context,
            "scale_factor": self.scale_factor,
            "head_dim": self.head_dim,
            "rotary_dim": int(self.head_dim * self.partial_rotary_factor),
            "rope_theta": self.rope_theta,
            "global_layers_affected": self.GLOBAL_LAYERS,
            "sliding_layers_unaffected": 25,
        }
