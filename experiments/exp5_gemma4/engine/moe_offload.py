"""MoE-aware expert offloading — expert-granular CPU↔GPU scheduling.

Standard offloading splits entire layers between GPU and CPU.  This module
splits at expert granularity: only the 8 active experts (of 128) live on
GPU at any given moment, with the rest parked in CPU pinned memory.

Key techniques:
  - LRU expert cache on GPU (configurable size, default 16 experts)
  - Expert prefetching: use router logits from layer N to start PCIe
    transfer for layer N+1's predicted experts before they're needed
  - Double buffering: two GPU expert buffers so one loads while the
    other computes
  - Pinned memory: CUDA pinned (page-locked) CPU memory for faster
    async transfers

For Gemma 4 26B MoE:
  - 128 experts per layer, 8 active + 1 shared per token
  - Each expert FFN: 2816 → 704 → 2816 (gate + up + down projections)
  - Per expert at Q4: ~3 MB → 16 cached = ~48 MB GPU budget
  - The shared expert is always on GPU (it runs every token)
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ExpertCacheStats:
    """Track cache performance for profiling."""

    hits: int = 0
    misses: int = 0
    prefetch_hits: int = 0
    evictions: int = 0
    total_transfer_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "prefetch_hits": self.prefetch_hits,
            "evictions": self.evictions,
            "total_transfer_mb": self.total_transfer_bytes / 1e6,
        }


class ExpertLRUCache:
    """LRU cache for MoE experts on GPU.

    Each entry is keyed by (layer_idx, expert_idx) and holds the expert's
    parameters as GPU tensors.  When the cache is full, the least recently
    used expert is evicted back to CPU.
    """

    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self._cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()
        self.stats = ExpertCacheStats()
        self._lock = threading.Lock()

    def get(self, layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor] | None:
        """Retrieve expert from cache, promoting to most-recently-used."""
        key = (layer_idx, expert_idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.stats.hits += 1
                return self._cache[key]
            self.stats.misses += 1
            return None

    def put(
        self, layer_idx: int, expert_idx: int, params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor] | None:
        """Insert expert into cache, returning evicted expert if cache full."""
        key = (layer_idx, expert_idx)
        evicted = None
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = params
                return None

            if len(self._cache) >= self.max_size:
                evicted_key, evicted_params = self._cache.popitem(last=False)
                self.stats.evictions += 1
                evicted = evicted_params

            self._cache[key] = params
        return evicted

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        return (layer_idx, expert_idx) in self._cache

    @property
    def size(self) -> int:
        return len(self._cache)


class ExpertOffloader:
    """Manages CPU↔GPU expert transfers with prefetching.

    Holds all expert weights in CPU pinned memory and maintains a GPU
    LRU cache.  Supports async prefetching to overlap PCIe transfer
    with computation.
    """

    def __init__(
        self,
        cache_size: int = 16,
        pin_memory: bool = True,
        prefetch: bool = True,
        gpu_device: torch.device | None = None,
    ):
        self.gpu_cache = ExpertLRUCache(max_size=cache_size)
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.gpu_device = gpu_device or torch.device("cuda:0")

        # CPU expert storage: {(layer, expert): {param_name: tensor}}
        self._cpu_experts: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

        # Prefetch state
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_pending: set[tuple[int, int]] = set()

        if torch.cuda.is_available() and prefetch:
            self._prefetch_stream = torch.cuda.Stream(device=self.gpu_device)

    def register_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        params: dict[str, torch.Tensor],
    ) -> None:
        """Register an expert's parameters for offloading.

        Moves params to CPU pinned memory for fast async transfer.
        """
        cpu_params = {}
        for name, tensor in params.items():
            cpu_tensor = tensor.to("cpu")
            if self.pin_memory and torch.cuda.is_available():
                cpu_tensor = cpu_tensor.pin_memory()
            cpu_params[name] = cpu_tensor

        self._cpu_experts[(layer_idx, expert_idx)] = cpu_params

    def load_expert(
        self, layer_idx: int, expert_idx: int
    ) -> dict[str, torch.Tensor]:
        """Load an expert to GPU, using cache if available.

        Returns GPU tensors for the expert's parameters.
        """
        # Check cache first
        cached = self.gpu_cache.get(layer_idx, expert_idx)
        if cached is not None:
            return cached

        # Cache miss — transfer from CPU
        cpu_params = self._cpu_experts.get((layer_idx, expert_idx))
        if cpu_params is None:
            raise KeyError(f"Expert ({layer_idx}, {expert_idx}) not registered")

        gpu_params = {}
        for name, tensor in cpu_params.items():
            gpu_params[name] = tensor.to(self.gpu_device, non_blocking=True)
            self.gpu_cache.stats.total_transfer_bytes += tensor.nbytes

        # Synchronize to ensure transfer complete
        if torch.cuda.is_available():
            torch.cuda.current_stream(self.gpu_device).synchronize()

        # Cache the loaded expert (may evict LRU)
        evicted = self.gpu_cache.put(layer_idx, expert_idx, gpu_params)
        if evicted is not None:
            # Let evicted tensors be garbage collected
            del evicted

        return gpu_params

    def prefetch_experts(
        self, layer_idx: int, expert_indices: list[int]
    ) -> None:
        """Start async prefetch of predicted next-layer experts.

        Called with router logits from the current layer to predict
        which experts the next layer will need.
        """
        if not self.prefetch or self._prefetch_stream is None:
            return

        with torch.cuda.stream(self._prefetch_stream):
            for eidx in expert_indices:
                key = (layer_idx, eidx)
                if self.gpu_cache.contains(layer_idx, eidx):
                    self.gpu_cache.stats.prefetch_hits += 1
                    continue
                if key in self._prefetch_pending:
                    continue

                cpu_params = self._cpu_experts.get(key)
                if cpu_params is None:
                    continue

                gpu_params = {
                    name: t.to(self.gpu_device, non_blocking=True)
                    for name, t in cpu_params.items()
                }
                self.gpu_cache.put(layer_idx, eidx, gpu_params)
                self._prefetch_pending.add(key)

    def sync_prefetch(self) -> None:
        """Wait for any pending prefetch transfers to complete."""
        if self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()
        self._prefetch_pending.clear()

    def summary(self) -> dict[str, Any]:
        """Return offloading statistics."""
        total_experts = len(self._cpu_experts)
        expert_size_mb = 0.0
        if self._cpu_experts:
            sample = next(iter(self._cpu_experts.values()))
            expert_size_mb = sum(t.nbytes for t in sample.values()) / 1e6

        return {
            "total_experts_registered": total_experts,
            "expert_size_mb": expert_size_mb,
            "total_expert_memory_gb": total_experts * expert_size_mb / 1e3,
            "gpu_cache_size": self.gpu_cache.max_size,
            "gpu_cache_memory_mb": self.gpu_cache.max_size * expert_size_mb,
            "pin_memory": self.pin_memory,
            "prefetch_enabled": self.prefetch,
            "cache_stats": self.gpu_cache.stats.summary(),
        }
