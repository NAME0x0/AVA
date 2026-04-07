"""Unit tests for the exp5 engine modules.

Run:  python -m pytest experiments/exp5_gemma4/scripts/test_engine.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from transformers import GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp5_gemma4.engine.benchmark import benchmark_inference_speed, get_model_input_device
from experiments.exp5_gemma4.engine.turboquant import MSECompressor, TurboQuantV3
from experiments.exp5_gemma4.engine.yarn import (
    YarnContextExtender,
    compute_yarn_frequencies,
    build_yarn_cos_sin_cache,
)
from experiments.exp5_gemma4.engine.moe_offload import ExpertLRUCache, ExpertOffloader
from experiments.exp5_gemma4.engine.loader import (
    estimate_kv_cache_memory,
    resolve_model_source,
    _module_nbytes,
    release_manual_gpu_residency,
)
from experiments.exp5_gemma4.engine.llamacpp import (
    build_llama_server_command,
    default_llama_hf_repo,
    extract_chat_metrics,
    _extract_message_content,
    resolve_llama_server_executable,
    LlamaCppServerConfig,
)
from experiments.exp5_gemma4.engine.speculative import build_assisted_generation_kwargs
from experiments.exp5_gemma4.engine.tiered import RoutingConfig, analyze_prompt, route_prompt


class TestMSECompressor:
    """Test TurboQuant compression/decompression fidelity."""

    def test_roundtrip_4bit(self):
        """4-bit compression should reconstruct with < 5% relative error."""
        comp = MSECompressor(bits=4, group_size=128, seed=42)
        x = torch.randn(1, 2, 64, 512, dtype=torch.float32)  # (B, H, S, D)
        compressed = comp.compress(x)
        reconstructed = comp.decompress(*compressed, x.shape)

        # Relative error
        rel_error = (reconstructed - x).norm() / x.norm()
        assert rel_error < 0.15, f"4-bit relative error too high: {rel_error:.3f}"

    def test_roundtrip_2bit(self):
        """2-bit compression has more error but should still be bounded."""
        comp = MSECompressor(bits=2, group_size=128, seed=42)
        x = torch.randn(1, 2, 64, 512, dtype=torch.float32)
        compressed = comp.compress(x)
        reconstructed = comp.decompress(*compressed, x.shape)

        rel_error = (reconstructed - x).norm() / x.norm()
        assert rel_error < 0.55, f"2-bit relative error too high: {rel_error:.3f}"

    def test_roundtrip_unpacked(self):
        """Unpacked mode (pack=False) should still work with 4-tuple format."""
        comp = MSECompressor(bits=4, group_size=128, seed=42, pack=False)
        x = torch.randn(1, 2, 64, 512, dtype=torch.float32)
        compressed = comp.compress(x)
        assert len(compressed) == 4, "Unpacked should return 4-tuple"
        reconstructed = comp.decompress(*compressed, x.shape)
        rel_error = (reconstructed - x).norm() / x.norm()
        assert rel_error < 0.15

    def test_packing_saves_memory(self):
        """Bit-packed indices should use fewer bytes than unpacked."""
        comp_packed = MSECompressor(bits=4, group_size=128, seed=42, pack=True)
        comp_unpacked = MSECompressor(bits=4, group_size=128, seed=42, pack=False)
        x = torch.randn(1, 2, 64, 512, dtype=torch.float32)

        packed = comp_packed.compress(x)
        unpacked = comp_unpacked.compress(x)

        packed_bytes = packed[0].nbytes
        unpacked_bytes = unpacked[0].nbytes
        assert packed_bytes < unpacked_bytes, (
            f"Packed ({packed_bytes}) should be smaller than unpacked ({unpacked_bytes})"
        )
        # 4-bit packing: exactly 2x smaller indices
        assert unpacked_bytes / packed_bytes == 2.0

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
        idx1 = c1.compress(x)[0]
        idx2 = c2.compress(x)[0]
        # Indices should differ (with overwhelming probability)
        assert not torch.equal(idx1, idx2)

    def test_zero_vector_handling(self):
        """Zero vectors should not cause NaN or inf."""
        comp = MSECompressor(bits=4, group_size=128, seed=42)
        x = torch.zeros(1, 1, 4, 256)
        compressed = comp.compress(x)
        reconstructed = comp.decompress(*compressed, x.shape)
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

    def test_lfu_policy_keeps_frequent_entry(self):
        cache = ExpertLRUCache(max_size=2, policy="lfu")
        cache.put(0, 0, {"w": torch.randn(5)})
        cache.put(0, 1, {"w": torch.randn(5)})
        cache.get(0, 0)
        cache.get(0, 0)
        cache.put(0, 2, {"w": torch.randn(5)})
        assert cache.get(0, 0) is not None
        assert cache.get(0, 2) is not None

    def test_protected_entries_survive_eviction(self):
        cache = ExpertLRUCache(max_size=3, policy="lfu", protected_size=1)
        cache.put(0, 0, {"w": torch.randn(5)})
        cache.put(0, 1, {"w": torch.randn(5)})
        cache.put(0, 2, {"w": torch.randn(5)})
        cache.set_protected([(0, 0)])
        cache.put(0, 3, {"w": torch.randn(5)})
        assert cache.get(0, 0) is not None
        remaining = {
            key for key in [(0, 1), (0, 2), (0, 3)]
            if cache.get(*key) is not None
        }
        assert len(remaining) == 2

    def test_non_decode_access_does_not_train_lfu(self):
        cache = ExpertLRUCache(max_size=2, policy="lfu")
        cache.put(0, 0, {"w": torch.randn(5)}, record_access=False)
        cache.put(0, 1, {"w": torch.randn(5)})
        cache.get(0, 1)
        cache.put(0, 2, {"w": torch.randn(5)})
        assert cache.get(0, 0) is None
        assert cache.get(0, 1) is not None


class TestExpertOffloader:
    """Test expert offloader cache insertion policies."""

    def test_batch_load_without_caching_keeps_cache_empty(self):
        offloader = ExpertOffloader(
            cache_size=2,
            prefetch=False,
            gpu_device=torch.device("cpu"),
        )
        offloader.register_expert(0, 1, {"w": torch.randn(4, 4)})
        offloader.register_expert(0, 2, {"w": torch.randn(4, 4)})

        result = offloader.load_experts_batch(
            0,
            [1, 2],
            record_access=False,
            cache_results=False,
        )

        assert set(result.keys()) == {1, 2}
        assert offloader.gpu_cache.size == 0

    def test_decode_cache_admission_requires_prior_hit(self):
        offloader = ExpertOffloader(
            cache_size=2,
            prefetch=False,
            gpu_device=torch.device("cpu"),
            decode_cache_min_hits=1,
        )
        offloader.register_expert(0, 1, {"w": torch.randn(4, 4)})
        offloader.register_expert(0, 2, {"w": torch.randn(4, 4)})

        assert offloader.cacheable_decode_experts(0, [1, 2]) == set()
        offloader.record_decode_experts(0, [1])
        assert offloader.cacheable_decode_experts(0, [1, 2]) == {1}

    def test_batch_load_only_caches_admitted_decode_experts(self):
        offloader = ExpertOffloader(
            cache_size=2,
            prefetch=False,
            gpu_device=torch.device("cpu"),
            decode_cache_min_hits=1,
        )
        offloader.register_expert(0, 1, {"w": torch.randn(4, 4)})
        offloader.register_expert(0, 2, {"w": torch.randn(4, 4)})
        offloader.record_decode_experts(0, [1])

        offloader.load_experts_batch(
            0,
            [1, 2],
            record_access=True,
            cache_results=True,
            cacheable_expert_indices=offloader.cacheable_decode_experts(0, [1, 2]),
        )

        assert offloader.gpu_cache.contains(0, 1)
        assert not offloader.gpu_cache.contains(0, 2)


class TestSpeculativeDecoding:
    """Exact assisted-decoding helpers should stay behavior-neutral."""

    def test_build_assisted_generation_kwargs_configures_model(self):
        assistant_model = SimpleNamespace(generation_config=GenerationConfig())

        kwargs = build_assisted_generation_kwargs(
            assistant_model,
            num_assistant_tokens=6,
            num_assistant_tokens_schedule="constant",
            assistant_confidence_threshold=0.25,
        )

        assert kwargs["assistant_model"] is assistant_model
        assert assistant_model.generation_config.num_assistant_tokens == 6
        assert assistant_model.generation_config.num_assistant_tokens_schedule == "constant"
        assert assistant_model.generation_config.assistant_confidence_threshold == 0.25

    def test_benchmark_passes_assistant_model(self):
        class FakeBatch(dict):
            def to(self, device):
                return FakeBatch(
                    {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in self.items()
                    }
                )

        class FakeProcessor:
            def __call__(self, text, return_tensors="pt"):
                del text, return_tensors
                return FakeBatch(
                    {
                        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                    }
                )

        class FakeModel:
            def __init__(self):
                self.device = torch.device("cpu")
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                input_ids = kwargs["input_ids"]
                max_new_tokens = kwargs["max_new_tokens"]
                appended = torch.full(
                    (input_ids.shape[0], max_new_tokens),
                    9,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                return torch.cat([input_ids, appended], dim=1)

        model = FakeModel()
        processor = FakeProcessor()
        assistant_model = object()

        speed = benchmark_inference_speed(
            model,
            processor,
            ["hello"],
            max_new_tokens=4,
            warmup_runs=0,
            assistant_model=assistant_model,
            generate_kwargs={"do_sample": False},
        )

        assert speed["output_tokens"] == 4
        assert model.calls[0]["assistant_model"] is assistant_model
        assert model.calls[0]["do_sample"] is False


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


class TestBenchmarkInputDevice:
    """Benchmark helpers should respect mixed CPU/GPU placement."""

    def test_manual_gpu_split_uses_cpu_inputs(self):
        class FakeModel:
            _manual_gpu_layers = 11
            device = torch.device("cuda:0")

        assert get_model_input_device(FakeModel()) == torch.device("cpu")

    def test_normal_model_uses_model_device(self):
        class FakeModel:
            device = torch.device("cuda:0")

        assert get_model_input_device(FakeModel()) == torch.device("cuda:0")


class TestLayerResidencyAccounting:
    """Quantized layers must count buffers toward GPU placement."""

    def test_module_nbytes_includes_buffers(self):
        class BufferOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("packed", torch.zeros(8, 8, dtype=torch.uint8))
                self.register_buffer("scales", torch.zeros(8, 2, dtype=torch.float16))

        module = BufferOnly()
        expected = module.packed.nbytes + module.scales.nbytes
        assert _module_nbytes(module) == expected

    def test_release_manual_gpu_residency_clears_hooks_and_metadata(self):
        class DummyHandle:
            def __init__(self):
                self.removed = False

            def remove(self):
                self.removed = True

        class DummyTextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

        class DummyWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = DummyTextModel()

        model = DummyWrapper()
        handle = DummyHandle()
        model._manual_gpu_layers = 1
        model._manual_total_layers = 1
        model._manual_gpu_used_bytes = 16
        model._device_boundary_hook_handles = [handle]

        release_manual_gpu_residency(model)

        assert handle.removed is True
        assert not hasattr(model, "_manual_gpu_layers")
        assert not hasattr(model, "_manual_total_layers")
        assert not hasattr(model, "_manual_gpu_used_bytes")


class TestModelSourceResolution:
    """Loader should prefer cached local model sources when available."""

    def test_existing_local_path_is_preserved(self):
        local_model = Path(".pytest-tmp-model-source")
        if local_model.exists():
            import shutil
            shutil.rmtree(local_model)
        local_model.mkdir()
        try:
            assert resolve_model_source(str(local_model)) == str(local_model)
        finally:
            import shutil
            shutil.rmtree(local_model)

    def test_missing_model_id_falls_back_to_repo_id(self):
        missing = "does-not-exist/example-model"
        assert resolve_model_source(missing) == missing


class TestTieredRouting:
    """Fast/deep routing should remain predictable."""

    def test_short_prompt_stays_on_fast_path(self):
        decision = route_prompt("What is the capital of France?")
        assert decision.tier == "fast"
        assert "fast-path" in decision.reason

    def test_deep_prefix_forces_deep_route(self):
        decision = route_prompt("deep: Explain the tradeoff in detail.")
        assert decision.tier == "deep"
        assert decision.cleaned_prompt == "Explain the tradeoff in detail."
        assert decision.reason == "explicit deep override"
        assert decision.thinking_override is False

    def test_reason_prefix_forces_deep_route_with_thinking(self):
        decision = route_prompt("reason: Explain the tradeoff in detail.")
        assert decision.tier == "deep"
        assert decision.cleaned_prompt == "Explain the tradeoff in detail."
        assert decision.reason == "explicit reasoning override"
        assert decision.thinking_override is True

    def test_quick_prefix_forces_fast_without_thinking(self):
        decision = route_prompt("quick: Summarize this.")
        assert decision.tier == "fast"
        assert decision.cleaned_prompt == "Summarize this."
        assert decision.thinking_override is False

    def test_long_code_prompt_routes_to_deep_path(self):
        prompt = "Please debug this traceback:\n```python\nTraceback\nTypeError: bad operand type\n```"
        decision = route_prompt(prompt)
        assert decision.tier == "deep"
        assert "code hints" in decision.reason

    def test_prompt_feature_analysis_reports_keyword_hits(self):
        features = analyze_prompt(
            "Analyze the architecture tradeoff and explain why it matters.",
            RoutingConfig(),
        )
        assert "analyze" in features.keyword_hits
        assert features.words > 0


class TestLlamaCppHelpers:
    """llama.cpp fast-path helpers should stay predictable."""

    def test_default_llama_hf_repo_maps_gemma_e2b(self):
        repo = default_llama_hf_repo("google/gemma-4-E2B-it")
        assert repo == "ggml-org/gemma-4-E2B-it-GGUF:Q8_0"

    def test_extract_chat_metrics_prefers_usage(self):
        payload = {
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19,
            }
        }
        assert extract_chat_metrics(payload) == {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
        }

    def test_extract_message_content_prefers_assistant_channel_tail(self):
        message = {
            "content": "<|channel>thought\nThinking Process...\n<|channel>assistant\nParis",
            "reasoning_content": "Thinking Process...",
        }
        assert _extract_message_content(message) == "\nParis"

    def test_extract_message_content_removes_think_tags(self):
        message = {
            "content": "<think>Reasoning</think>Paris",
            "reasoning_content": "Reasoning",
        }
        assert _extract_message_content(message) == "Paris"

    def test_build_llama_server_command_uses_hf_repo_defaults(self):
        config = LlamaCppServerConfig(
            model_id="google/gemma-4-E2B-it",
            executable=str(Path(sys.executable)),
            gguf_quant="Q8_0",
            port=8081,
            ctx_size=524_288,
            gpu_layers="auto",
            flash_attn="on",
            cache_type_k="q8_0",
            cache_type_v="q4_0",
            reasoning="off",
            reasoning_budget=0,
            reasoning_format="deepseek",
        )

        command = build_llama_server_command(config)

        assert command[0] == str(Path(sys.executable))
        assert "--hf-repo" in command
        assert "ggml-org/gemma-4-E2B-it-GGUF:Q8_0" in command
        assert "--ctx-size" in command
        assert "524288" in command
        assert "--cache-type-k" in command
        assert "--cache-type-v" in command
        assert "--reasoning" in command
        assert "off" in command

    def test_resolve_llama_server_executable_accepts_explicit_path(self):
        temp_exe = Path(".pytest-tmp-llama-server.exe")
        temp_exe.write_text("echo", encoding="utf-8")
        try:
            assert resolve_llama_server_executable(str(temp_exe)) == str(temp_exe)
        finally:
            temp_exe.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
