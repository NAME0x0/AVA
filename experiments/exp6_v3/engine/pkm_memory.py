"""Product-key memory layer — the RAM-tier factual capacity edge (E1).

A vast learned key-value table where each token retrieves only the top-k of
N = num_keys^2 value slots. The product-key factorization (Lample et al. 2019)
splits the query into two halves and searches two sqrt(N)-sized key sets, so
candidate scoring costs O(sqrt(N)) instead of O(N):

    q = [q1 ; q2]
    top-k over keys1 with q1  ->  k half-scores s1, indices i1
    top-k over keys2 with q2  ->  k half-scores s2, indices i2
    candidates = {(i, j) : score = s1_i + s2_j}      (k^2 of them)
    final top-k -> softmax weights -> weighted sum of value rows

Compute per token: 2 * sqrt(N) key dot-products + k value-row fetches —
bandwidth-trivial, which is why this parameter class does not need VRAM.
Deployment places the value table in system RAM / SSD (docs/v3/EDGES.md E1);
the GPU only sees the k fetched rows per token.

Design follows Meta's *Memory Layers at Scale* (arXiv:2412.09764): multi-head
retrieval into one shared value pool, silu output gating, value table shared
across the layers that mount the memory branch. Their result: memory-augmented
models beat dense models at 2x compute and beat parameter/compute-matched MoE,
with the largest gains on factual tasks.

References:
- Lample et al., Large Memory Layers with Product Keys: arxiv.org/abs/1907.05242
- Berges et al., Memory Layers at Scale: arxiv.org/abs/2412.09764
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ProductKeyMemory(nn.Module):
    """Sparse key-value memory with product-key retrieval.

    One instance is shared by every mount point in the model (Meta's shared-pool
    convention) — instantiate once, call from multiple layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_keys: int = 1024,          # slots = num_keys^2 (1024 -> ~1.05M slots)
        key_dim: int = 256,
        value_dim: int | None = None,
        topk: int = 32,
        num_heads: int = 2,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_keys = num_keys
        self.num_slots = num_keys * num_keys
        self.key_dim = key_dim
        self.topk = topk
        self.num_heads = num_heads
        value_dim = value_dim or hidden_size
        self.value_dim = value_dim

        # Two key sets per head; query supplies one half per set.
        self.query_proj = nn.Linear(hidden_size, num_heads * 2 * key_dim, bias=False, dtype=dtype)
        self.query_norm = nn.RMSNorm(key_dim, eps=1e-6)
        self.keys = nn.Parameter(torch.empty(2, num_heads, num_keys, key_dim, dtype=dtype))

        # The RAM tier: |slots| x value_dim. Only top-k rows touched per token.
        self.values = nn.EmbeddingBag(self.num_slots, value_dim, mode="sum")

        self.gate_proj = nn.Linear(hidden_size, value_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(value_dim, hidden_size, bias=False, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.keys.device.type != "meta":
            nn.init.normal_(self.keys, std=self.key_dim**-0.5)
            nn.init.normal_(self.values.weight, std=0.02)
            # Zero-init output: mounting the memory branch is a no-op at step 0,
            # so it can be added to a warm model without disturbing it.
            nn.init.zeros_(self.out_proj.weight)

    def _retrieve(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """q: [T, H, 2, key_dim] -> (slot indices [T, H*k], weights [T, H*k])."""
        t = q.shape[0]
        k = self.topk
        q = self.query_norm(q)

        # Half scores: [T, H, num_keys] each.
        s1 = torch.einsum("thd,hnd->thn", q[:, :, 0], self.keys[0])
        s2 = torch.einsum("thd,hnd->thn", q[:, :, 1], self.keys[1])
        top1, i1 = s1.topk(k, dim=-1)                    # [T, H, k]
        top2, i2 = s2.topk(k, dim=-1)

        # Product-key combine: k x k candidate grid, then final top-k.
        grid = top1.unsqueeze(-1) + top2.unsqueeze(-2)   # [T, H, k, k]
        scores, flat_idx = grid.view(t, self.num_heads, -1).topk(k, dim=-1)
        row = flat_idx // k                              # index into i1
        col = flat_idx % k                               # index into i2
        slots = i1.gather(-1, row) * self.num_keys + i2.gather(-1, col)

        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        return slots.reshape(t, -1), weights.reshape(t, -1)

    def forward(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])

        q = self.query_proj(flat).view(-1, self.num_heads, 2, self.key_dim)
        slots, weights = self._retrieve(q)
        looked_up = self.values(slots, per_sample_weights=weights.to(self.values.weight.dtype))
        looked_up = looked_up.to(flat.dtype)

        out = self.out_proj(looked_up * F.silu(self.gate_proj(flat)))
        return out.reshape(*batch_shape, -1)

    def extra_repr(self) -> str:
        return (
            f"slots={self.num_slots}, value_dim={self.value_dim}, "
            f"topk={self.topk}, heads={self.num_heads}"
        )
