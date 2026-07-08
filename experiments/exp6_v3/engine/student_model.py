"""AVA v3 student model — full assembly.

Architecture (docs/v3/ARCHITECTURE_V3.md, June 2026 revision):

    embeddings (tied)
      -> 6 x HRMRepeatUnit:
           refinement loop over [L, L, L]   (Mamba-3 + MoTE, shared weights,
                                             up to 6 iterations, ACT halting)
           -> H-block                       (gated GQA attention + RoPE + MoTE)
      -> RMSNorm -> tied LM head

    24 blocks total: 18 L + 6 H. ~3.26 B parameters at full size.

P2 scope: training-shape forward pass (no KV/state cache, no generation
loop). Inference caching and the llama.cpp export path land at P9.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hrm_core import HRMOutput, HRMRepeatUnit
from .mamba3_block import build_l_mixer
from .mote_ffn import MoTEFFN
from .pkm_memory import ProductKeyMemory


@dataclass
class V3Config:
    # Embedding / stack
    vocab_size: int = 248320
    hidden_size: int = 1792
    num_repeats: int = 6                       # x [L, L, L, H] = 24 blocks
    rms_norm_eps: float = 1e-6
    # H-block (gated softmax attention)
    num_q_heads: int = 14
    num_kv_heads: int = 4
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    # L-block (Mamba-3)
    l_num_heads: int = 14
    l_head_dim: int = 128
    l_state_size: int = 64
    l_kernel: str = "reference"                # "fla" on CUDA training runs
    # MoTE FFN
    num_routed_experts: int = 32
    moe_top_k: int = 4
    intermediate_size_routed: int = 768
    intermediate_size_shared: int = 4096
    ternary_group_size: int = 256
    # HRM recurrence (docs/v3/HRM_TEXT.md section 1b)
    max_l_steps: int = 6
    mean_l_steps_target: float = 2.5
    ponder_loss_weight: float = 0.05
    # Sample-level loss aggregation (SubQ-1.1 report section 3.4,
    # docs/v3/RESEARCH_ROUND_5.md): average CE per example, then over the batch,
    # so a few very long sequences do not dominate the gradient. Matters once
    # YaRN-extended variable-length batches enter training; token-level default
    # is unchanged for fixed-length training.
    sample_level_loss: bool = False
    supervised_steps: tuple[int, ...] = (2, 4, 6)
    detach_between_segments: bool = True
    restart_confidence_threshold: float = 0.35
    restart_perturbation_std: float = 0.02
    # OpenMythos-derived refinements (docs/v3/RESEARCH_ROUND_4.md), default off:
    # no-op at init, gated ablations A16 (loop-index) / A17 (LTI stability).
    use_loop_index_embedding: bool = False
    loop_index_dim: int = 64
    use_lti_injection: bool = False
    # Product-key memory tier (E1, docs/v3/EDGES.md) — RAM-resident capacity.
    # One shared pool, mounted as a residual branch on the listed units.
    use_memory: bool = False
    memory_num_keys: int = 1024            # slots = num_keys^2 (~1.05M)
    memory_key_dim: int = 256
    memory_topk: int = 32
    memory_heads: int = 2
    memory_units: tuple[int, ...] = (1, 4)

    @classmethod
    def tiny(cls) -> V3Config:
        """CPU-testable config: same topology, toy dimensions."""
        return cls(
            vocab_size=512,
            hidden_size=64,
            num_repeats=2,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=32,
            l_num_heads=2,
            l_head_dim=32,
            l_state_size=8,
            num_routed_experts=4,
            moe_top_k=2,
            intermediate_size_routed=32,
            intermediate_size_shared=64,
            ternary_group_size=64,
            max_l_steps=3,
            supervised_steps=(2, 3),
        )

    @classmethod
    def donor_qwen35_4b(cls) -> V3Config:
        """Qwen3.5-4B donor skeleton.

        dims are PROVISIONAL pending the donor config.json verification at C1
        (head_dim and GDN head dims are family-typical guesses; FFN/MoE fields
        are not donor-derived — v3.0 runs dense); engine's Mamba-3 reference
        mixer approximates the donor's Gated DeltaNet at matching dimensions
        for T3 mount experiments.
        """
        return cls(
            vocab_size=248320,
            hidden_size=2560,
            num_repeats=8,
            num_q_heads=16,
            num_kv_heads=4,
            head_dim=128,
            l_num_heads=32,
            l_head_dim=80,
            l_state_size=64,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> V3Config:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        hrm = raw.get("hrm", {})
        moe = raw.get("moe", {})
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            num_repeats=raw["attention_layout"]["repeats"],
            num_q_heads=raw["h_block"]["num_q_heads"],
            num_kv_heads=raw["h_block"]["num_kv_heads"],
            head_dim=raw["h_block"]["head_dim"],
            rope_theta=float(raw["rope_theta"]),
            l_num_heads=raw["l_block"]["num_heads"],
            l_head_dim=raw["l_block"]["head_dim"],
            l_state_size=raw["l_block"]["state_size"],
            l_kernel=raw["l_block"].get("kernel", "fla"),
            num_routed_experts=moe["num_routed_experts"],
            moe_top_k=moe["top_k"],
            intermediate_size_routed=moe["intermediate_size_routed"],
            intermediate_size_shared=moe["intermediate_size_shared"],
            ternary_group_size=raw["quantization"]["ternary_group_size"],
            max_l_steps=hrm["max_l_steps"],
            mean_l_steps_target=hrm["mean_l_steps_target"],
            ponder_loss_weight=hrm["halting"]["ponder_loss_weight"],
            supervised_steps=tuple(hrm["deep_supervision"]["supervised_steps"]),
            detach_between_segments=hrm["deep_supervision"]["detach_between_segments"],
            restart_confidence_threshold=hrm["latent_restart"]["confidence_threshold"],
            restart_perturbation_std=hrm["latent_restart"]["perturbation_std"],
            **(
                {
                    "use_memory": True,
                    "memory_num_keys": mem["num_keys"],
                    "memory_key_dim": mem["key_dim"],
                    "memory_topk": mem["topk"],
                    "memory_heads": mem["num_heads"],
                    "memory_units": tuple(mem["mount_units"]),
                }
                if (mem := raw.get("memory_layer", {})).get("enabled", False)
                else {}
            ),
        )


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    # q, k: [b, t, heads, head_dim]; cos/sin: [t, head_dim]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class GatedAttention(nn.Module):
    """GQA softmax attention with RoPE, per-head q/k norm, sigmoid output gate."""

    def __init__(self, cfg: V3Config) -> None:
        super().__init__()
        self.num_q_heads = cfg.num_q_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        inner = cfg.num_q_heads * cfg.head_dim
        kv_inner = cfg.num_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_inner, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_inner, bias=False)
        self.gate_proj = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.o_proj = nn.Linear(inner, cfg.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.num_q_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        q, k = _apply_rope(q, k, cos.to(x.dtype), sin.to(x.dtype))

        groups = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=2)
        v = v.repeat_interleave(groups, dim=2)

        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(attn * torch.sigmoid(self.gate_proj(x)))


class V3Block(nn.Module):
    """Pre-norm residual block: sequence mixer + MoTE FFN.

    ``kind`` is "L" (Mamba-3, ignores RoPE inputs) or "H" (gated attention,
    consumes RoPE inputs). Both kinds accept cos/sin so the HRM unit can pass
    one kwargs set to every block.
    """

    def __init__(self, cfg: V3Config, kind: str) -> None:
        super().__init__()
        assert kind in ("L", "H")
        self.kind = kind
        self.norm_mixer = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_ffn = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if kind == "H":
            self.mixer: nn.Module = GatedAttention(cfg)
        else:
            self.mixer = build_l_mixer(
                cfg.hidden_size,
                cfg.l_num_heads,
                cfg.l_head_dim,
                cfg.l_state_size,
                kernel=cfg.l_kernel,
            )
        self.ffn = MoTEFFN(
            cfg.hidden_size,
            cfg.intermediate_size_routed,
            cfg.intermediate_size_shared,
            cfg.num_routed_experts,
            cfg.moe_top_k,
            cfg.ternary_group_size,
        )

    def forward(self, x: Tensor, cos: Tensor | None = None, sin: Tensor | None = None) -> Tensor:
        h = self.norm_mixer(x)
        if self.kind == "H":
            assert cos is not None and sin is not None
            x = x + self.mixer(h, cos, sin)
        else:
            x = x + self.mixer(h)
        return x + self.ffn(self.norm_ffn(x))


@dataclass
class V3ModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    lm_loss: Tensor | None = None
    ponder_loss: Tensor | None = None
    unit_outputs: list[HRMOutput] = field(default_factory=list)

    @property
    def mean_l_steps(self) -> float:
        if not self.unit_outputs:
            return 0.0
        return sum(u.steps_used for u in self.unit_outputs) / len(self.unit_outputs)


class AVAv3ForCausalLM(nn.Module):
    """The AVA v3 student. ~3.26 B parameters at full config."""

    def __init__(self, cfg: V3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.units = nn.ModuleList(
            HRMRepeatUnit(
                l_blocks=nn.ModuleList(V3Block(cfg, "L") for _ in range(3)),
                h_block=V3Block(cfg, "H"),
                hidden_size=cfg.hidden_size,
                max_l_steps=cfg.max_l_steps,
                supervised_steps=cfg.supervised_steps,
                detach_between_segments=cfg.detach_between_segments,
                restart_confidence_threshold=cfg.restart_confidence_threshold,
                restart_perturbation_std=cfg.restart_perturbation_std,
                use_loop_index_embedding=cfg.use_loop_index_embedding,
                loop_index_dim=cfg.loop_index_dim,
                use_lti_injection=cfg.use_lti_injection,
            )
            for _ in range(cfg.num_repeats)
        )
        self.final_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # E1: one shared product-key memory pool, mounted on cfg.memory_units.
        self.memory: ProductKeyMemory | None = (
            ProductKeyMemory(
                cfg.hidden_size,
                num_keys=cfg.memory_num_keys,
                key_dim=cfg.memory_key_dim,
                topk=cfg.memory_topk,
                num_heads=cfg.memory_heads,
            )
            if cfg.use_memory
            else None
        )
        self.apply(self._init_weights)
        if self.memory is not None:
            self.memory.reset_parameters()  # restore zero-init out_proj after apply()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if module.weight.device.type != "meta":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        reasoning_budget: int | str = "auto",
    ) -> V3ModelOutput:
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary(input_ids.shape[1], input_ids.device)

        unit_outputs: list[HRMOutput] = []
        for i, unit in enumerate(self.units):
            # cond = unit input: re-injected at every L-iteration so the
            # refinement loop stays anchored (HRM input-injection convention).
            h, hrm_out = unit(
                h, cond=h, reasoning_budget=reasoning_budget, cos=cos, sin=sin
            )
            if self.memory is not None and i in self.cfg.memory_units:
                h = h + self.memory(h)
            unit_outputs.append(hrm_out)

        logits = F.linear(self.final_norm(h), self.embed_tokens.weight)  # tied head

        loss = lm_loss = ponder = None
        if labels is not None:
            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            if self.cfg.sample_level_loss:
                # Per-example mean CE, then mean over examples (SubQ §3.4): a few
                # long sequences cannot dominate the gradient. Examples with no
                # supervised tokens (all -100) are dropped from the batch mean.
                b, t, v = shift_logits.shape
                per_tok = F.cross_entropy(
                    shift_logits.reshape(-1, v),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(b, t)
                valid = (shift_labels != -100).float()
                counts = valid.sum(dim=1)
                per_ex = (per_tok * valid).sum(dim=1) / counts.clamp(min=1.0)
                has_tokens = counts > 0
                lm_loss = (
                    per_ex[has_tokens].mean()
                    if has_tokens.any()
                    else per_ex.sum() * 0.0
                )
            else:
                lm_loss = F.cross_entropy(
                    shift_logits.reshape(-1, self.cfg.vocab_size),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            expected = torch.stack([u.expected_steps.mean() for u in unit_outputs])
            ponder = (expected - self.cfg.mean_l_steps_target).abs().mean()
            loss = lm_loss + self.cfg.ponder_loss_weight * ponder

        return V3ModelOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm_loss,
            ponder_loss=ponder,
            unit_outputs=unit_outputs,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def count_full_size_params(cfg: V3Config | None = None) -> dict[str, Any]:
    """Instantiate the full-size model on the meta device and report sizes."""
    cfg = cfg or V3Config()
    with torch.device("meta"):
        model = AVAv3ForCausalLM(cfg)
    total = model.num_parameters()
    embed = model.embed_tokens.weight.numel()
    routed = shared = memory = 0
    for name, p in model.named_parameters():
        if ".experts." in name:
            routed += p.numel()
        elif ".shared_expert." in name:
            shared += p.numel()
        elif name.startswith("memory."):
            memory += p.numel()
    return {
        "total": total,
        "embedding_tied": embed,
        "routed_experts": routed,
        "shared_experts": shared,
        "memory_ram_tier": memory,
        "other": total - embed - routed - shared - memory,
    }
