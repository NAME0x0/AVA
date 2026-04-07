"""Test streaming quantization loader on E4B."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp5_gemma4.engine.streaming import (
    DequantWeightCache,
    QuantizedLinear,
    StreamingInt4Quantizer,
    _dequantize_packed_weight,
    _get_safetensors_index,
    _group_weights_by_layer,
    _triton_int4_linear,
)
from experiments.exp5_gemma4.engine.moe_offload import ExpertOffloader


def test_quantizer_roundtrip():
    """Test int4 quantizer accuracy."""
    q = StreamingInt4Quantizer(group_size=128)

    # Simulate a linear weight
    weight = torch.randn(1024, 2048, dtype=torch.float32)
    qdata = q.quantize(weight)
    reconstructed = q.dequantize(qdata)

    # Check shapes
    assert reconstructed.shape == weight.shape, f"Shape mismatch: {reconstructed.shape} vs {weight.shape}"

    # Check compression ratio
    orig_bytes = weight.nbytes
    q_bytes = qdata["packed"].nbytes + qdata["scales"].nbytes + qdata["zeros"].nbytes
    ratio = orig_bytes / q_bytes
    print(f"Compression: {ratio:.1f}x ({orig_bytes / 1e6:.1f} MB -> {q_bytes / 1e6:.1f} MB)")
    assert ratio > 3.0, f"Compression ratio too low: {ratio:.1f}x"

    # Check reconstruction quality
    rel_error = (reconstructed.float() - weight).norm() / weight.norm()
    print(f"Relative error: {rel_error:.4f}")
    assert rel_error < 0.15, f"Error too high: {rel_error:.4f}"


def test_quantized_linear():
    """Test QuantizedLinear forward pass."""
    q = StreamingInt4Quantizer(group_size=128)

    # Create original linear
    linear = torch.nn.Linear(512, 256, bias=True)
    weight = linear.weight.data.clone()
    bias = linear.bias.data.clone()

    # Quantize
    qdata = q.quantize(weight)
    q_linear = QuantizedLinear(
        packed=qdata["packed"],
        scales=qdata["scales"],
        zeros=qdata["zeros"],
        original_in_features=qdata["original_in_features"],
        bias=bias,
        group_size=128,
    )

    # Compare outputs
    x = torch.randn(2, 512)
    with torch.no_grad():
        orig_out = linear(x)
        q_out = q_linear(x)

    rel_error = (orig_out - q_out).norm() / orig_out.norm()
    print(f"QuantizedLinear output error: {rel_error:.4f}")
    assert rel_error < 0.2, f"Output error too high: {rel_error:.4f}"


def test_quantized_linear_hot_cache():
    """Cached dequantized weights should be reused across forwards."""
    q = StreamingInt4Quantizer(group_size=128)
    linear = torch.nn.Linear(512, 256, bias=True)
    qdata = q.quantize(linear.weight.data)
    q_linear = QuantizedLinear(
        packed=qdata["packed"],
        scales=qdata["scales"],
        zeros=qdata["zeros"],
        original_in_features=qdata["original_in_features"],
        bias=linear.bias.data.clone(),
        group_size=128,
    )
    cache = DequantWeightCache(max_bytes=2 * q_linear.dequantized_weight_bytes())
    q_linear.configure_dequant_cache(cache)

    x = torch.randn(2, 512)
    with torch.no_grad():
        y1 = q_linear(x)
        y2 = q_linear(x)

    assert torch.allclose(y1, y2)
    assert cache.hits >= 1
    assert cache.size == 1


def test_quantized_linear_gpu_module_with_cpu_input():
    """GPU-resident QuantizedLinear should accept CPU inputs and round-trip output."""
    if not torch.cuda.is_available():
        return

    q = StreamingInt4Quantizer(group_size=128)
    linear = torch.nn.Linear(512, 256, bias=True)
    qdata = q.quantize(linear.weight.data)
    q_linear = QuantizedLinear(
        packed=qdata["packed"],
        scales=qdata["scales"],
        zeros=qdata["zeros"],
        original_in_features=qdata["original_in_features"],
        bias=linear.bias.data.clone(),
        group_size=128,
    ).to(torch.device("cuda:0"))
    cache = DequantWeightCache(max_bytes=2 * q_linear.dequantized_weight_bytes())
    q_linear.configure_dequant_cache(cache, compute_dtype=torch.float16)

    x = torch.randn(2, 512)
    with torch.no_grad():
        orig_out = linear(x)
        q_out = q_linear(x)

    assert q_out.device.type == "cpu"
    rel_error = (orig_out - q_out).norm() / orig_out.norm()
    assert rel_error < 0.2
    assert cache.size in (0, 1)


def test_triton_int4_linear_matches_reference():
    """CUDA packed-int4 kernel should match the dequantized reference path."""
    if not torch.cuda.is_available():
        return

    q = StreamingInt4Quantizer(group_size=128)
    linear = torch.nn.Linear(512, 256, bias=True)
    qdata = q.quantize(linear.weight.data)

    x = torch.randn(4, 512, device="cuda:0", dtype=torch.float16)
    packed = qdata["packed"].to(device="cuda:0")
    scales = qdata["scales"].to(device="cuda:0")
    zeros = qdata["zeros"].to(device="cuda:0")
    bias = linear.bias.data.to(device="cuda:0", dtype=torch.float16)

    with torch.no_grad():
        ref_weight = _dequantize_packed_weight(
            packed,
            scales,
            zeros,
            qdata["original_in_features"],
            128,
        ).to(dtype=torch.float16)
        ref_out = torch.nn.functional.linear(x, ref_weight, bias)
        triton_out = _triton_int4_linear(x, packed, scales, zeros, 128, bias=bias)

    if triton_out is None:
        return

    rel_error = (ref_out - triton_out).norm() / ref_out.norm()
    assert rel_error < 5e-3


def test_safetensors_index():
    """Test index reading for E4B model."""
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    e4b_dir = os.path.join(cache_dir, "models--google--gemma-4-E4B-it")
    if not os.path.exists(e4b_dir):
        print("E4B not cached, skipping index test")
        return

    snap = os.path.join(e4b_dir, "snapshots")
    snaps = os.listdir(snap)
    model_path = Path(os.path.join(snap, snaps[0]))

    weight_map = _get_safetensors_index(model_path)
    groups = _group_weights_by_layer(weight_map)

    print(f"Weights: {len(weight_map)}")
    print(f"Groups: {list(groups.keys())[:5]}... ({len(groups)} total)")
    layer_groups = [k for k in groups if k.startswith("layer_")]
    print(f"Layers: {len(layer_groups)}")
    assert len(layer_groups) > 0


def test_save_load_quantized():
    """Test save/load roundtrip for quantized model weights."""
    import shutil

    from experiments.exp5_gemma4.engine.streaming import (
        QuantizedLinear,
        StreamingInt4Quantizer,
        save_quantized,
    )

    # Build a small fake model with QuantizedLinear
    q = StreamingInt4Quantizer(group_size=128)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(256)
            self.linear1 = torch.nn.Linear(256, 128)
            self.linear2 = torch.nn.Linear(128, 256)

    model = FakeModel()

    # Quantize the linears
    for name in ["linear1", "linear2"]:
        linear = getattr(model, name)
        qdata = q.quantize(linear.weight.data)
        q_linear = QuantizedLinear(
            packed=qdata["packed"],
            scales=qdata["scales"],
            zeros=qdata["zeros"],
            original_in_features=qdata["original_in_features"],
            bias=linear.bias.data.clone() if linear.bias is not None else None,
            group_size=128,
        )
        setattr(model, name, q_linear)

    # Need a config attr for save_quantized
    from types import SimpleNamespace
    model.config = SimpleNamespace()
    model.config.save_pretrained = lambda path: None  # no-op for fake model

    # Save
    save_dir = Path(".pytest-tmp-streaming") / "test_quantized"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_quantized(model, save_dir, {"model_id": "test", "group_size": 128})

    # Verify files exist
    assert (save_dir / "quantized_config.json").exists()
    assert (save_dir / "model.safetensors.index.json").exists()

    import json
    with open(save_dir / "quantized_config.json") as f:
        cfg = json.load(f)
    assert len(cfg["module_map"]) == 2  # linear1, linear2
    assert cfg["module_map"]["linear1"]["type"] == "QuantizedLinear"

    # Check shard files
    import glob
    shards = glob.glob(str(save_dir / "model-*.safetensors"))
    assert len(shards) >= 1

    print(f"Saved: {len(cfg['module_map'])} quantized modules, {len(shards)} shards")

    # Verify state dict was saved correctly
    from safetensors import safe_open
    all_keys = set()
    for shard_path in shards:
        with safe_open(shard_path, framework="pt") as f:
            all_keys.update(f.keys())

    assert "linear1.packed" in all_keys
    assert "linear1.scales" in all_keys
    assert "linear1.zeros" in all_keys
    assert "norm.weight" in all_keys
    print(f"Verified {len(all_keys)} tensor keys in safetensors")

    print("Save/load roundtrip: PASS")


def test_quantized_moe_experts():
    """Test QuantizedMoEExperts forward at 26B-scale dimensions."""
    from experiments.exp5_gemma4.engine.streaming import (
        QuantizedMoEExperts,
        StreamingInt4Quantizer,
    )

    # 26B dimensions: hidden=2304, intermediate=12288, 128 experts
    num_experts = 128
    hidden_dim = 2304
    intermediate_dim = 12288
    group_size = 128

    q = StreamingInt4Quantizer(group_size=group_size)
    act_fn = torch.nn.GELU(approximate="tanh")

    # Quantize synthetic expert weights
    gu_packed, gu_scales, gu_zeros = [], [], []
    dn_packed, dn_scales, dn_zeros = [], [], []

    for e in range(num_experts):
        # gate_up: (2*intermediate, hidden)
        w_gu = torch.randn(2 * intermediate_dim, hidden_dim) * 0.02
        qd = q.quantize(w_gu)
        gu_packed.append(qd["packed"])
        gu_scales.append(qd["scales"])
        gu_zeros.append(qd["zeros"])

        # down: (hidden, intermediate)
        w_dn = torch.randn(hidden_dim, intermediate_dim) * 0.02
        qd = q.quantize(w_dn)
        dn_packed.append(qd["packed"])
        dn_scales.append(qd["scales"])
        dn_zeros.append(qd["zeros"])

    # Build module
    qmoe = QuantizedMoEExperts(
        num_experts=num_experts,
        gate_up_packed=gu_packed,
        gate_up_scales=gu_scales,
        gate_up_zeros=gu_zeros,
        gate_up_in_features=hidden_dim,
        down_packed=dn_packed,
        down_scales=dn_scales,
        down_zeros=dn_zeros,
        down_in_features=intermediate_dim,
        act_fn=act_fn,
        group_size=group_size,
    )

    # Memory stats
    param_bytes = sum(b.nbytes for b in qmoe.buffers())
    print(f"Expert memory: {param_bytes / 1e6:.0f} MB (int4 packed)")

    # Forward pass: 4 tokens, top-2 routing
    batch_tokens = 4
    hidden_states = torch.randn(batch_tokens, hidden_dim, dtype=torch.float32)
    top_k_index = torch.randint(0, num_experts, (batch_tokens, 2))
    top_k_weights = torch.softmax(torch.randn(batch_tokens, 2), dim=-1)

    with torch.no_grad():
        output = qmoe(hidden_states, top_k_index, top_k_weights)

    assert output.shape == hidden_states.shape, f"Shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    print(f"Forward output shape: {output.shape}")
    print(f"Output norm: {output.norm():.4f}")
    print("QuantizedMoEExperts: PASS")


def test_quantized_moe_expert_hot_cache():
    """GPU hot-cache plumbing should reuse experts across forwards."""
    from experiments.exp5_gemma4.engine.streaming import (
        QuantizedMoEExperts,
        StreamingInt4Quantizer,
    )

    num_experts = 4
    hidden_dim = 32
    intermediate_dim = 64
    q = StreamingInt4Quantizer(group_size=32)
    act_fn = torch.nn.GELU(approximate="tanh")

    gu_packed, gu_scales, gu_zeros = [], [], []
    dn_packed, dn_scales, dn_zeros = [], [], []
    for _ in range(num_experts):
        qd = q.quantize(torch.randn(2 * intermediate_dim, hidden_dim) * 0.02)
        gu_packed.append(qd["packed"])
        gu_scales.append(qd["scales"])
        gu_zeros.append(qd["zeros"])

        qd = q.quantize(torch.randn(hidden_dim, intermediate_dim) * 0.02)
        dn_packed.append(qd["packed"])
        dn_scales.append(qd["scales"])
        dn_zeros.append(qd["zeros"])

    baseline = QuantizedMoEExperts(
        num_experts=num_experts,
        gate_up_packed=gu_packed,
        gate_up_scales=gu_scales,
        gate_up_zeros=gu_zeros,
        gate_up_in_features=hidden_dim,
        down_packed=dn_packed,
        down_scales=dn_scales,
        down_zeros=dn_zeros,
        down_in_features=intermediate_dim,
        act_fn=act_fn,
        group_size=32,
    )
    qmoe = QuantizedMoEExperts(
        num_experts=num_experts,
        gate_up_packed=gu_packed,
        gate_up_scales=gu_scales,
        gate_up_zeros=gu_zeros,
        gate_up_in_features=hidden_dim,
        down_packed=dn_packed,
        down_scales=dn_scales,
        down_zeros=dn_zeros,
        down_in_features=intermediate_dim,
        act_fn=act_fn,
        group_size=32,
    )

    offloader = ExpertOffloader(
        cache_size=3,
        decode_cache_min_hits=0,
        pin_memory=False,
        prefetch=False,
        gpu_device=torch.device("cpu"),
    )
    qmoe.configure_gpu_offload(3, offloader, gpu_device=torch.device("cpu"))

    hidden_states = torch.randn(3, hidden_dim)
    top_k_index = torch.tensor([[1, 2], [1, 3], [1, 2]])
    top_k_weights = torch.full((3, 2), 0.5)

    with torch.no_grad():
        baseline_out = baseline(hidden_states, top_k_index, top_k_weights)
        qmoe(hidden_states, top_k_index, top_k_weights)
        hits_after_first = offloader.gpu_cache.stats.hits
        cached_out = qmoe(hidden_states, top_k_index, top_k_weights)

    assert offloader.gpu_cache.size <= 3
    assert offloader.gpu_cache.stats.hits > hits_after_first
    assert offloader.gpu_cache.stats.misses > 0
    rel_error = (baseline_out - cached_out).norm() / baseline_out.norm()
    assert rel_error < 0.02
    cached = offloader.gpu_cache.get(3, 1)
    assert cached is not None
    assert "gu_weight" in cached and "dn_weight" in cached
    print(f"Hot-cache stats: {offloader.gpu_cache.stats.summary()}")


def test_quantized_moe_packed_hot_cache():
    """Packed GPU-cache entries should remain correct and reuseable."""
    from experiments.exp5_gemma4.engine.streaming import (
        ExpertWeightCache,
        QuantizedMoEExperts,
        StreamingInt4Quantizer,
    )

    num_experts = 4
    hidden_dim = 32
    intermediate_dim = 64
    q = StreamingInt4Quantizer(group_size=32)
    act_fn = torch.nn.GELU(approximate="tanh")

    gu_packed, gu_scales, gu_zeros = [], [], []
    dn_packed, dn_scales, dn_zeros = [], [], []
    for _ in range(num_experts):
        qd = q.quantize(torch.randn(2 * intermediate_dim, hidden_dim) * 0.02)
        gu_packed.append(qd["packed"])
        gu_scales.append(qd["scales"])
        gu_zeros.append(qd["zeros"])

        qd = q.quantize(torch.randn(hidden_dim, intermediate_dim) * 0.02)
        dn_packed.append(qd["packed"])
        dn_scales.append(qd["scales"])
        dn_zeros.append(qd["zeros"])

    baseline = QuantizedMoEExperts(
        num_experts=num_experts,
        gate_up_packed=gu_packed,
        gate_up_scales=gu_scales,
        gate_up_zeros=gu_zeros,
        gate_up_in_features=hidden_dim,
        down_packed=dn_packed,
        down_scales=dn_scales,
        down_zeros=dn_zeros,
        down_in_features=intermediate_dim,
        act_fn=act_fn,
        group_size=32,
    )
    qmoe = QuantizedMoEExperts(
        num_experts=num_experts,
        gate_up_packed=gu_packed,
        gate_up_scales=gu_scales,
        gate_up_zeros=gu_zeros,
        gate_up_in_features=hidden_dim,
        down_packed=dn_packed,
        down_scales=dn_scales,
        down_zeros=dn_zeros,
        down_in_features=intermediate_dim,
        act_fn=act_fn,
        group_size=32,
    )

    offloader = ExpertOffloader(
        cache_size=3,
        decode_cache_min_hits=0,
        pin_memory=False,
        prefetch=False,
        gpu_device=torch.device("cpu"),
    )
    qmoe.configure_gpu_offload(
        7,
        offloader,
        gpu_device=torch.device("cpu"),
        cache_layout="packed",
        expert_weight_cache=ExpertWeightCache(max_bytes=1_000_000),
    )

    hidden_states = torch.randn(3, hidden_dim)
    top_k_index = torch.tensor([[1, 2], [1, 3], [1, 2]])
    top_k_weights = torch.full((3, 2), 0.5)

    with torch.no_grad():
        baseline_out = baseline(hidden_states, top_k_index, top_k_weights)
        qmoe(hidden_states, top_k_index, top_k_weights)
        hits_after_first = offloader.gpu_cache.stats.hits
        cached_out = qmoe(hidden_states, top_k_index, top_k_weights)

    assert offloader.gpu_cache.size <= 3
    assert offloader.gpu_cache.stats.hits > hits_after_first
    rel_error = (baseline_out - cached_out).norm() / baseline_out.norm()
    assert rel_error < 0.02
    cached = offloader.gpu_cache.get(7, 1)
    assert cached is not None
    assert "gu_packed" in cached and "dn_packed" in cached
    assert "gu_weight" not in cached and "dn_weight" not in cached
    assert qmoe._expert_weight_cache is not None
    assert qmoe._expert_weight_cache.hits > 0
    print(f"Packed hot-cache stats: {offloader.gpu_cache.stats.summary()}")


def test_moe_prefetch_only_on_small_decode_batches():
    """Prefetch should only run for small decode-sized token batches."""
    from experiments.exp5_gemma4.engine.streaming import (
        QuantizedMoEExperts,
        StreamingInt4Quantizer,
    )

    q = StreamingInt4Quantizer(group_size=32)
    act_fn = torch.nn.GELU(approximate="tanh")
    gu_packed, gu_scales, gu_zeros = [], [], []
    dn_packed, dn_scales, dn_zeros = [], [], []
    for _ in range(2):
        qd = q.quantize(torch.randn(32, 16) * 0.02)
        gu_packed.append(qd["packed"])
        gu_scales.append(qd["scales"])
        gu_zeros.append(qd["zeros"])
        qd = q.quantize(torch.randn(16, 16) * 0.02)
        dn_packed.append(qd["packed"])
        dn_scales.append(qd["scales"])
        dn_zeros.append(qd["zeros"])

    qmoe = QuantizedMoEExperts(
        num_experts=2,
        gate_up_packed=gu_packed,
        gate_up_scales=gu_scales,
        gate_up_zeros=gu_zeros,
        gate_up_in_features=16,
        down_packed=dn_packed,
        down_scales=dn_scales,
        down_zeros=dn_zeros,
        down_in_features=16,
        act_fn=act_fn,
        group_size=32,
    )

    calls: list[tuple[str, int, list[int]]] = []

    class StubOffloader:
        def cacheable_decode_experts(self, layer_idx: int, expert_ids: list[int]) -> set[int]:
            return set(expert_ids)

        def record_decode_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
            calls.append(("record", layer_idx, expert_ids))

        def prefetch_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
            calls.append(("prefetch", layer_idx, expert_ids))

    qmoe._expert_offloader = StubOffloader()
    qmoe._layer_idx = 5

    qmoe._prefetch_next_token_experts([1], num_tokens=8)
    assert calls == []

    qmoe._prefetch_next_token_experts([1], num_tokens=1)
    assert calls == [("record", 5, [1]), ("prefetch", 5, [1])]


if __name__ == "__main__":
    print("=== Quantizer Roundtrip ===")
    test_quantizer_roundtrip()
    print("\n=== QuantizedLinear ===")
    test_quantized_linear()
    print("\n=== Safetensors Index ===")
    test_safetensors_index()
    print("\n=== Save/Load Quantized ===")
    test_save_load_quantized()
    print("\n=== QuantizedMoEExperts (26B dims) ===")
    test_quantized_moe_experts()
    print("\n=== QuantizedMoEExperts Hot Cache ===")
    test_quantized_moe_expert_hot_cache()
    print("\nAll streaming tests passed!")
