from __future__ import annotations

import math
from typing import Any

from ava.config import ModelConfig

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: Any) -> Any:
            norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x * norm).type_as(x) * self.weight


    def _build_norm(config: ModelConfig) -> Any:
        if config.norm_type == "rmsnorm":
            return RMSNorm(config.n_embd)
        return nn.LayerNorm(config.n_embd)


    def _precompute_rope_freqs(
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: Any = None,
    ) -> tuple[Any, Any]:
        dim_pairs = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (theta ** (dim_pairs / head_dim))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, freqs)
        cos = angles.cos()
        sin = angles.sin()
        return cos, sin


    def _apply_rope(x: Any, freqs_cos: Any, freqs_sin: Any) -> Any:
        seq_len = x.shape[2]
        cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)
        x_r = x.float().reshape(*x.shape[:-1], -1, 2)
        x1 = x_r[..., 0]
        x2 = x_r[..., 1]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2).type_as(x)


    class SwiGLUMLP(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            hidden = ((int(2 / 3 * 4 * config.n_embd) + 63) // 64) * 64
            self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: Any) -> Any:
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


    class CausalSelfAttention(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            if config.n_embd % config.n_head != 0:
                raise ValueError("n_embd must be divisible by n_head")
            self.n_head = config.n_head
            self.head_dim = config.n_embd // config.n_head
            self.use_rope = config.position_encoding == "rope"
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: Any, freqs_cos: Any = None, freqs_sin: Any = None) -> Any:
            batch_size, sequence_length, channels = x.size()
            qkv = self.c_attn(x)
            query, key, value = qkv.split(channels, dim=2)
            query = query.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
            if self.use_rope and freqs_cos is not None and freqs_sin is not None:
                query = _apply_rope(query, freqs_cos, freqs_sin)
                key = _apply_rope(key, freqs_cos, freqs_sin)
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
            output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
            return self.c_proj(output)


    class MLP(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            hidden = 4 * config.n_embd
            self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: Any) -> Any:
            return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


    class Block(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.ln_1 = _build_norm(config)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = _build_norm(config)
            if config.activation == "swiglu":
                self.mlp = SwiGLUMLP(config)
            else:
                self.mlp = MLP(config)

        def forward(self, x: Any, freqs_cos: Any = None, freqs_sin: Any = None) -> Any:
            x = x + self.attn(self.ln_1(x), freqs_cos, freqs_sin)
            x = x + self.mlp(self.ln_2(x))
            return x


    class GPT(nn.Module):
        def __init__(self, config: ModelConfig, vocab_size: int) -> None:
            super().__init__()
            if config.architecture not in {"transformer", "looped", "recurrent_depth"}:
                raise ValueError(f"unsupported architecture: {config.architecture}")
            self.config = config
            self.wte = nn.Embedding(vocab_size, config.n_embd)
            if config.position_encoding == "rope":
                self.wpe = None
                self.register_buffer(
                    "rope_cos",
                    _precompute_rope_freqs(
                        config.n_embd // config.n_head,
                        config.block_size,
                        config.rope_theta,
                    )[0],
                    persistent=False,
                )
                self.register_buffer(
                    "rope_sin",
                    _precompute_rope_freqs(
                        config.n_embd // config.n_head,
                        config.block_size,
                        config.rope_theta,
                    )[1],
                    persistent=False,
                )
            else:
                self.wpe = nn.Embedding(config.block_size, config.n_embd)
                self.rope_cos = None
                self.rope_sin = None
            self.drop = nn.Dropout(config.dropout)
            self.prelude = nn.ModuleList([Block(config) for _ in range(config.recurrent_prelude_layers)])
            self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
            self.coda = nn.ModuleList([Block(config) for _ in range(config.recurrent_coda_layers)])
            self.loop_step_embeddings = (
                nn.Embedding(config.loop_repeats, config.n_embd)
                if config.architecture in {"looped", "recurrent_depth"}
                else None
            )
            if self.loop_step_embeddings is not None:
                nn.init.zeros_(self.loop_step_embeddings.weight)
            self.recurrent_step_scale = 1.0 / max(config.loop_repeats, 1) if config.architecture == "recurrent_depth" and not config.recurrent_gate else 1.0
            self.recurrent_gate_proj = (
                nn.Linear(config.n_embd, config.n_embd, bias=True)
                if config.recurrent_gate and config.architecture == "recurrent_depth"
                else None
            )
            if self.recurrent_gate_proj is not None:
                nn.init.zeros_(self.recurrent_gate_proj.weight)
                nn.init.constant_(self.recurrent_gate_proj.bias, -1.0)
            self.ln_f = _build_norm(config)
            self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight
            self._init_weights()

        def _init_weights(self) -> None:
            n_layers = self.config.effective_layers()
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            for block in list(self.prelude) + list(self.blocks) + list(self.coda):
                if hasattr(block.mlp, 'c_proj'):
                    torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
                elif hasattr(block.mlp, 'w2'):
                    torch.nn.init.normal_(block.mlp.w2.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
                torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
            if self.loop_step_embeddings is not None:
                nn.init.zeros_(self.loop_step_embeddings.weight)
            if self.recurrent_gate_proj is not None:
                nn.init.zeros_(self.recurrent_gate_proj.weight)
                nn.init.constant_(self.recurrent_gate_proj.bias, -1.0)

        def _repeat_count(self, override: int | None = None) -> int:
            if override is not None:
                return override
            # Check for test-time compute scaling attribute
            inference_override = getattr(self, "_inference_repeat_override", None)
            if inference_override is not None:
                return inference_override
            if self.config.architecture in {"looped", "recurrent_depth"}:
                return self.config.loop_repeats
            return 1

        def _apply_loop_embedding(self, x: Any, loop_index: int) -> Any:
            if self.loop_step_embeddings is None:
                return x
            # Clamp to last step for test-time scaling beyond trained loop count
            idx = min(loop_index, self.loop_step_embeddings.num_embeddings - 1)
            step = self.loop_step_embeddings.weight[idx].view(1, 1, -1)
            return x + step

        def _run_blocks(self, x: Any, blocks: Any, freqs_cos: Any = None, freqs_sin: Any = None) -> Any:
            for block in blocks:
                x = block(x, freqs_cos, freqs_sin)
            return x

        def _run_recurrent_depth(self, x: Any, freqs_cos: Any = None, freqs_sin: Any = None, *, repeat_override: int | None = None) -> Any:
            x = self._run_blocks(x, self.prelude, freqs_cos, freqs_sin)
            for loop_index in range(self._repeat_count(repeat_override)):
                loop_input = self._apply_loop_embedding(x, loop_index)
                candidate = self._run_blocks(loop_input, self.blocks, freqs_cos, freqs_sin)
                if self.recurrent_gate_proj is not None:
                    gate = torch.sigmoid(self.recurrent_gate_proj(x))
                    x = x + gate * (candidate - loop_input)
                else:
                    x = x + (self.recurrent_step_scale * (candidate - loop_input))
            x = self._run_blocks(x, self.coda, freqs_cos, freqs_sin)
            return x

        def forward(self, idx: Any, targets: Any | None = None, *, repeat_override: int | None = None) -> tuple[Any, Any | None]:
            _, sequence_length = idx.size()
            if sequence_length > self.config.block_size:
                raise ValueError("sequence length exceeds block size")
            x = self.wte(idx)
            if self.wpe is not None:
                positions = torch.arange(0, sequence_length, device=idx.device, dtype=torch.long)
                x = x + self.wpe(positions)
            x = self.drop(x)
            freqs_cos = self.rope_cos if self.rope_cos is not None else None
            freqs_sin = self.rope_sin if self.rope_sin is not None else None
            if self.config.architecture == "recurrent_depth":
                x = self._run_recurrent_depth(x, freqs_cos, freqs_sin, repeat_override=repeat_override)
            else:
                for block in self.blocks:
                    for loop_index in range(self._repeat_count(repeat_override)):
                        x = self._apply_loop_embedding(x, loop_index)
                        x = block(x, freqs_cos, freqs_sin)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss

        @torch.no_grad()
        def generate(self, idx: Any, max_new_tokens: int, temperature: float = 1.0) -> Any:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(next_token_logits, dim=1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
            return idx


    def build_model(config: ModelConfig, vocab_size: int) -> GPT:
        return GPT(config, vocab_size)


else:
    class GPT:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PyTorch is required to build AVA models. Install with `pip install -e .[train]`.")


    def build_model(_config: ModelConfig, _vocab_size: int) -> GPT:
        raise RuntimeError("PyTorch is required to build AVA models. Install with `pip install -e .[train]`.")
