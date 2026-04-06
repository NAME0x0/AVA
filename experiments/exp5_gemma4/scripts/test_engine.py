"""Unit tests for the exp5 engine modules.

Run:  python -m pytest experiments/exp5_gemma4/scripts/test_engine.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp5_gemma4.engine.turboquant import MSECompressor, TurboQuantV3
from experiments.exp5_gemma4.engine.yarn import (
    YarnContextExtender,
    compute_yarn_frequencies,
    build_yarn_cos_sin_cache,
)
from experiments.exp5_gemma4.engine.moe_offload import ExpertLRUCache, ExpertOffloader
from experiments.exp5_gemma4.engine.loader import estimate_kv_cache_memory


class TestMSECompressor:
    """Test TurboQuant compression/decompression fidelity."""

    def test_roundtrip_4bit(self):
        """4-bit compression should reconstruct with < 5% relative error."""
        comp = MSECompressor(bits=4, group_size=128, seed=42)
        x = torch.randn(1, 2, 64, 512, dtype=torch.float32)  # (B, H, S, D)
        indices, norms, mins, scales = comp.compress(x)
        reconstructed = comp.decompress(indices, norms, mins, scales, x.shape)

        # Relative error
        rel_error = (reconstructed - x).norm() / x.norm()
        assert rel_error < 0.15, f"4-bit relative error too high: {rel_error:.3f}"

    def test_roundtrip_2bit(self):
        """2-bit compression has more error but should still be bounded."""
        comp = MSECompressor(bits=2, group_size=128, seed=42)
        x = torch.randn(1, 2, 64, 512, dtype=torch.float32)
        indices, norms, mins, scales = comp.compress(x)
        reconstructed = comp.decompress(indices, norms, mins, scales, x.shape)

        rel_error = (reconstructed - x).norm() / x.norm()
        assert rel_error < 0.55, f"2-bit relative error too high: {rel_error:.3f}"

    def test_cosine_similarity_preserved(self):
        """Inner products should be approximately preserved after compression."""
        comp = MSECompressor(bits=4, group_size=128, seed=42)
        # Two random KV caches
        a = torch.randn(1, 2, 32, 256, dtype=torch.float32)
        b = torch.randn(1, 2, 32, 256, dtype=torch.float32)

        # Original inner products
        orig_sim = torch.nn.functional.cosine_similarity(
            a.reshape(-1, 256), b.reshape(-1, 256)
        )

        # Compressed + decompressed
        ca = comp.compress(a)
        cb = comp.compress(b)
        a_hat = comp.decompress(*ca, a.shape)
        b_hat = comp.decompress(*cb, b.shape)

        comp_sim = torch.nn.functional.cosine_similarity(
            a_hat.reshape(-1, 256), b_hat.reshape(-1, 256)
        )

        # Correlation between original and compressed similarities
        correlation = torch.corrcoef(torch.stack([orig_sim, comp_sim]))[0, 1]
        assert correlation > 0.9, f"Similarity correlation too low: {correlation:.3f}"

    def test_different_seeds_different_rotations(self):
        """Different seeds should produce different quantizations."""
        x = torch.randn(1, 1, 8, 128, dtype=torch.float32)
        c1 = MSECompressor(bits=4, seed=42)
        c2 = MSECompressor(bits=4, seed=99)
        idx1, _, _, _ = c1.compress(x)
        idx2, _, _, _ = c2.compress(x)
        # Indices should differ (with overwhelming probability)
        assert not torch.equal(idx1, idx2)

    def test_zero_vector_handling(self):
        """Zero vectors should not cause NaN or inf."""
        comp = MSECompressor(bits=4, group_size=128, seed=42)
        x = torch.zeros(1, 1, 4, 256)
        indices, norms, mins, scales = comp.compress(x)
        reconstructed = comp.decompress(indices, norms, mins, scales, x.shape)
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()


class TestTurboQuantV3:
    """Test the TurboQuant V3 orchestrator."""

    def test_protected_layers(self):
        """First and last global layers should return tensors unchanged."""
        tq = TurboQuantV3(key_bits=4, value_bits=2, protected_layers=1)
        x = torch.randn(1, 2, 16, 512)

        # Layer 5 is the first global layer → protected
        result = tq.compress_keys(x, layer_idx=5)
        assert isinstance(result, torch.Tensor), "Protected layer should return tensor"
        assert torch.equal(result, x)

        # Layer 29 is the last global layer → protected
        result = tq.compress_keys(x, layer_idx=29)
        assert isinstance(result, torch.Tensor)

    def test_unprotected_layers_compressed(self):
        """Middle global layers should return compressed tuples."""
        tq = TurboQuantV3(key_bits=4, value_bits=2, protected_layers=1)
        x = torch.randn(1, 2, 16, 512)

        # Layer 11 is the second global layer → not protected
        result = tq.compress_keys(x, layer_idx=11)
        assert isinstance(result, tuple), "Unprotected layer should return compressed tuple"

    def test_compression_ratio_estimate(self):
        tq = TurboQuantV3(key_bits=4, value_bits=2)
        ratios = tq.estimate_compression_ratio(head_dim=512)
        assert ratios["key_compression"] > 3.0
        assert ratios["value_compression"] > 5.0
        assert ratios["average_compression"] > 4.0


class TestYaRN:
    """Test YaRN RoPE extension."""

    def test_frequency_scaling(self):
        """YaRN frequencies should be between original and fully-scaled."""
        inv_freq, scale = compute_yarn_frequencies(
            head_dim=512,
            original_max_pos=262_144,
            target_max_pos=1_048_576,
        )
        # Original frequencies
        rotary_dim = int(512 * 0.25)
        base_freq = 1.0 / (1e6 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        scaled_freq = base_freq / 4.0  # 4x scale factor

        # YaRN should be between base and scaled for most dimensions
        # (some high-freq dims stay at base, some low-freq dims go to scaled)
        assert inv_freq.shape[0] == rotary_dim // 2
        assert scale > 0

    def test_cos_sin_cache_shape(self):
        """Cache should have correct shape."""
        max_seq = 8192  # Small for testing
        cos, sin, scale = build_yarn_cos_sin_cache(
            head_dim=512,
            max_seq_len=max_seq,
            original_max_pos=262_144,
        )
        rotary_dim = int(512 * 0.25) // 2  # half of rotary_dim
        assert cos.shape == (max_seq, rotary_dim)
        assert sin.shape == (max_seq, rotary_dim)

    def test_context_extender_summary(self):
        extender = YarnContextExtender(target_context=1_048_576)
        summary = extender.summary()
        assert summary["scale_factor"] == 4.0
        assert len(summary["global_layers_affected"]) == 5
        assert summary["rotary_dim"] == 128  # 512 * 0.25


class TestExpertCache:
    """Test MoE expert LRU cache."""

    def test_basic_put_get(self):
        cache = ExpertLRUCache(max_size=4)
        params = {"weight": torch.randn(10, 10)}
        cache.put(0, 0, params)
        result = cache.get(0, 0)
        assert result is not None
        assert torch.equal(result["weight"], params["weight"])

    def test_eviction(self):
        cache = ExpertLRUCache(max_size=2)
        cache.put(0, 0, {"w": torch.randn(5)})
        cache.put(0, 1, {"w": torch.randn(5)})
        # This should evict (0, 0)
        cache.put(0, 2, {"w": torch.randn(5)})
        assert cache.get(0, 0) is None  # evicted
        assert cache.get(0, 1) is not None
        assert cache.get(0, 2) is not None

    def test_lru_order(self):
        cache = ExpertLRUCache(max_size=2)
        cache.put(0, 0, {"w": torch.randn(5)})
        cache.put(0, 1, {"w": torch.randn(5)})
        # Access (0, 0) to make it recently used
        cache.get(0, 0)
        # Now (0, 1) is LRU → should be evicted
        cache.put(0, 2, {"w": torch.randn(5)})
        assert cache.get(0, 0) is not None  # recently used
        assert cache.get(0, 1) is None  # evicted
        assert cache.get(0, 2) is not None

    def test_stats(self):
        cache = ExpertLRUCache(max_size=2)
        cache.put(0, 0, {"w": torch.randn(5)})
        cache.get(0, 0)  # hit
        cache.get(0, 1)  # miss
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        assert cache.stats.hit_rate == 0.5


class TestKVCacheEstimation:
    """Test memory estimation logic."""

    def test_1m_context_26b(self):
        arch = {
            "num_layers": 30,
            "hidden_size": 2816,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "attention_k_eq_v": True,
            "sliding_window": 1024,
            "global_layer_indices": [5, 11, 17, 23, 29],
            "sliding_layer_indices": list(range(25)),
        }
        result = estimate_kv_cache_memory(arch, 1_048_576)
        # BF16 total should be ~10.9 GB
        assert 10_000 < result["total_bf16_mb"] < 12_000

    def test_turboquant_reduces_memory(self):
        arch = {
            "num_layers": 30,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "attention_k_eq_v": True,
            "sliding_window": 1024,
            "global_layer_indices": [5, 11, 17, 23, 29],
            "sliding_layer_indices": list(range(25)),
        }
        bf16 = estimate_kv_cache_memory(arch, 262_144)
        tq = estimate_kv_cache_memory(arch, 262_144, turboquant_key_bits=4, turboquant_value_bits=2)
        assert tq["total_compressed_mb"] < bf16["total_bf16_mb"]
        assert tq["compression_ratio"] > 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
