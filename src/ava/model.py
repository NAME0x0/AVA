from __future__ import annotations

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
    class CausalSelfAttention(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            if config.n_embd % config.n_head != 0:
                raise ValueError("n_embd must be divisible by n_head")
            self.n_head = config.n_head
            self.head_dim = config.n_embd // config.n_head
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: Any) -> Any:
            batch_size, sequence_length, channels = x.size()
            qkv = self.c_attn(x)
            query, key, value = qkv.split(channels, dim=2)
            query = query.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
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
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = MLP(config)

        def forward(self, x: Any) -> Any:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


    class GPT(nn.Module):
        def __init__(self, config: ModelConfig, vocab_size: int) -> None:
            super().__init__()
            if config.architecture not in {"transformer", "looped"}:
                raise ValueError(f"unsupported architecture: {config.architecture}")
            self.config = config
            self.wte = nn.Embedding(vocab_size, config.n_embd)
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
            self.drop = nn.Dropout(config.dropout)
            self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
            self.loop_step_embeddings = (
                nn.Embedding(config.loop_repeats, config.n_embd)
                if config.architecture == "looped"
                else None
            )
            self.ln_f = nn.LayerNorm(config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight

        def _repeat_count(self) -> int:
            if self.config.architecture == "looped":
                return self.config.loop_repeats
            return 1

        def _apply_loop_embedding(self, x: Any, loop_index: int) -> Any:
            if self.loop_step_embeddings is None:
                return x
            step = self.loop_step_embeddings.weight[loop_index].view(1, 1, -1)
            return x + step

        def forward(self, idx: Any, targets: Any | None = None) -> tuple[Any, Any | None]:
            _, sequence_length = idx.size()
            if sequence_length > self.config.block_size:
                raise ValueError("sequence length exceeds block size")
            positions = torch.arange(0, sequence_length, device=idx.device, dtype=torch.long)
            x = self.wte(idx) + self.wpe(positions)
            x = self.drop(x)
            for block in self.blocks:
                for loop_index in range(self._repeat_count()):
                    x = self._apply_loop_embedding(x, loop_index)
                    x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        @torch.no_grad()
        def generate(self, idx: Any, max_new_tokens: int, temperature: float = 1.0) -> Any:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(next_token_logits, dim=-1)
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
