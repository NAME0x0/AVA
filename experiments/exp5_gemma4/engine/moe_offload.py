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
  - Optional pinned memory for faster async transfers when the extra
    host-memory cost is acceptable

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

    def __init__(
        self,
        max_size: int = 16,
        policy: str = "lru",
        protected_size: int = 0,
    ):
        self.max_size = max_size
        self.policy = policy
        self.protected_size = max(0, min(protected_size, max_size))
        self._cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()
        self._access_count: dict[tuple[int, int], int] = {}
        self._protected_keys: set[tuple[int, int]] = set()
        self.stats = ExpertCacheStats()
        self._lock = threading.Lock()

    def get(
        self,
        layer_idx: int,
        expert_idx: int,
        record_access: bool = True,
    ) -> dict[str, torch.Tensor] | None:
        """Retrieve expert from cache, promoting to most-recently-used."""
        key = (layer_idx, expert_idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                if record_access:
                    self._access_count[key] = self._access_count.get(key, 0) + 1
                self.stats.hits += 1
                return self._cache[key]
            self.stats.misses += 1
            return None

    def put(
        self,
        layer_idx: int,
        expert_idx: int,
        params: dict[str, torch.Tensor],
        record_access: bool = True,
    ) -> dict[str, torch.Tensor] | None:
        """Insert expert into cache, returning evicted expert if cache full."""
        key = (layer_idx, expert_idx)
        evicted = None
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = params
                if record_access:
                    self._access_count[key] = self._access_count.get(key, 0) + 1
                return None

            if len(self._cache) >= self.max_size:
                eviction_candidates = [
                    candidate for candidate in self._cache.keys()
                    if candidate not in self._protected_keys
                ]
                if not eviction_candidates:
                    eviction_candidates = list(self._cache.keys())
                if self.policy == "lfu":
                    evicted_key = min(
                        eviction_candidates,
                        key=lambda candidate: self._access_count.get(candidate, 0),
                    )
                    evicted_params = self._cache.pop(evicted_key)
                else:
                    evicted_key = eviction_candidates[0]
                    evicted_params = self._cache.pop(evicted_key)
                self._access_count.pop(evicted_key, None)
                self._protected_keys.discard(evicted_key)
                self.stats.evictions += 1
                evicted = evicted_params

            self._cache[key] = params
            self._access_count[key] = self._access_count.get(key, 0) + (1 if record_access else 0)
        return evicted

    def set_protected(self, keys: list[tuple[int, int]]) -> None:
        """Protect a small hot set from eviction when alternatives exist."""
        if self.protected_size <= 0:
            self._protected_keys.clear()
            return

        selected = set(keys[: self.protected_size])
        with self._lock:
            self._protected_keys = {
                key for key in selected
                if key in self._cache or key in self._access_count
            }

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        return (layer_idx, expert_idx) in self._cache

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def protected_keys(self) -> set[tuple[int, int]]:
        return set(self._protected_keys)


class ExpertOffloader:
    """Manages CPU↔GPU expert transfers with prefetching.

    Holds all expert weights in CPU pinned memory and maintains a GPU
    LRU cache.  Supports async prefetching to overlap PCIe transfer
    with computation.
    """

    def __init__(
        self,
        cache_size: int = 16,
        cache_policy: str = "lru",
        protected_cache_entries: int = 0,
        decode_cache_min_hits: int = 1,
        pin_memory: bool = False,
        prefetch: bool = True,
        gpu_device: torch.device | None = None,
    ):
        self.gpu_cache = ExpertLRUCache(
            max_size=cache_size,
            policy=cache_policy,
            protected_size=protected_cache_entries,
        )
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.gpu_device = gpu_device or torch.device("cuda:0")
        self.decode_cache_min_hits = max(0, int(decode_cache_min_hits))

        # CPU expert storage: {(layer, expert): {param_name: tensor}}
        self._cpu_experts: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._expert_transforms: dict[tuple[int, int], Any] = {}
        self._decode_access_count: dict[tuple[int, int], int] = {}

        # Prefetch state
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_pending: set[tuple[int, int]] = set()

        if torch.cuda.is_available() and prefetch and self.gpu_device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.gpu_device)

    def _await_prefetched(self, keys: list[tuple[int, int]]) -> None:
        """Wait on prefetched experts before they are consumed."""
        if self._prefetch_stream is None or self.gpu_device.type != "cuda":
            return

        pending = [key for key in keys if key in self._prefetch_pending]
        if not pending:
            return

        torch.cuda.current_stream(self.gpu_device).wait_stream(self._prefetch_stream)
        for key in pending:
            self._prefetch_pending.discard(key)

    def register_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        params: dict[str, torch.Tensor],
        transform: Any | None = None,
    ) -> None:
        """Register an expert's parameters for offloading.

        Reuses existing CPU tensors when possible so the quantized model
        does not get duplicated in host RAM.
        """
        cpu_params = {}
        for name, tensor in params.items():
            cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.to("cpu")
            if (
                self.pin_memory
                and torch.cuda.is_available()
                and cpu_tensor.device.type == "cpu"
                and not cpu_tensor.is_pinned()
            ):
                cpu_tensor = cpu_tensor.pin_memory()
            cpu_params[name] = cpu_tensor

        self._cpu_experts[(layer_idx, expert_idx)] = cpu_params
        self._expert_transforms[(layer_idx, expert_idx)] = transform

    def _materialize_expert(
        self,
        layer_idx: int,
        expert_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Transfer one expert to the target device without synchronizing."""
        cpu_params = self._cpu_experts.get((layer_idx, expert_idx))
        if cpu_params is None:
            raise KeyError(f"Expert ({layer_idx}, {expert_idx}) not registered")

        for tensor in cpu_params.values():
            self.gpu_cache.stats.total_transfer_bytes += tensor.nbytes

        transform = self._expert_transforms.get((layer_idx, expert_idx))
        if transform is not None:
            return transform(cpu_params, self.gpu_device)

        return {
            name: tensor.to(self.gpu_device, non_blocking=True)
            for name, tensor in cpu_params.items()
        }

    def load_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        record_access: bool = True,
        cache_result: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Load an expert to GPU, using cache if available.

        Returns GPU tensors for the expert's parameters.
        """
        self._await_prefetched([(layer_idx, expert_idx)])

        # Check cache first
        cached = self.gpu_cache.get(layer_idx, expert_idx, record_access=record_access)
        if cached is not None:
            return cached

        gpu_params = self._materialize_expert(layer_idx, expert_idx)

        # Synchronize to ensure transfer complete
        if torch.cuda.is_available() and self.gpu_device.type == "cuda":
            torch.cuda.current_stream(self.gpu_device).synchronize()

        # Cache the loaded expert (may evict LRU)
        if cache_result:
            evicted = self.gpu_cache.put(
                layer_idx,
                expert_idx,
                gpu_params,
                record_access=record_access,
            )
            if evicted is not None:
                # Let evicted tensors be garbage collected
                del evicted

        return gpu_params

    def load_experts_batch(
        self,
        layer_idx: int,
        expert_indices: list[int],
        record_access: bool = True,
        cache_results: bool = True,
        cacheable_expert_indices: set[int] | None = None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Load multiple experts with a single synchronization point."""
        results: dict[int, dict[str, torch.Tensor]] = {}
        pending: list[tuple[int, dict[str, torch.Tensor]]] = []
        unique_indices = list(dict.fromkeys(expert_indices))

        self._await_prefetched([(layer_idx, expert_idx) for expert_idx in unique_indices])

        for expert_idx in unique_indices:
            cached = self.gpu_cache.get(
                layer_idx,
                expert_idx,
                record_access=record_access,
            )
            if cached is not None:
                results[expert_idx] = cached
                continue

            gpu_params = self._materialize_expert(layer_idx, expert_idx)
            pending.append((expert_idx, gpu_params))

        if pending and torch.cuda.is_available() and self.gpu_device.type == "cuda":
            torch.cuda.current_stream(self.gpu_device).synchronize()

        for expert_idx, gpu_params in pending:
            should_cache = (
                cache_results
                and (
                    cacheable_expert_indices is None
                    or expert_idx in cacheable_expert_indices
                )
            )
            if should_cache:
                evicted = self.gpu_cache.put(
                    layer_idx,
                    expert_idx,
                    gpu_params,
                    record_access=record_access,
                )
                if evicted is not None:
                    del evicted
            results[expert_idx] = gpu_params

        return results

    def cacheable_decode_experts(
        self,
        layer_idx: int,
        expert_indices: list[int],
    ) -> set[int]:
        """Return decode experts eligible for cache admission/prefetch."""
        cacheable: set[int] = set()
        for expert_idx in dict.fromkeys(expert_indices):
            key = (layer_idx, expert_idx)
            if (
                self.gpu_cache.contains(layer_idx, expert_idx)
                or key in self.gpu_cache.protected_keys
                or self._decode_access_count.get(key, 0) >= self.decode_cache_min_hits
            ):
                cacheable.add(expert_idx)
        return cacheable

    def prefetch_experts(
        self, layer_idx: int, expert_indices: list[int]
    ) -> None:
        """Start async prefetch of predicted next-layer experts.

        Called with router logits from the current layer to predict
        which experts the next layer will need.
        """
        if not self.prefetch or self._prefetch_stream is None or self.gpu_device.type != "cuda":
            return

        with torch.cuda.stream(self._prefetch_stream):
            for eidx in expert_indices:
                key = (layer_idx, eidx)
                if self.gpu_cache.contains(layer_idx, eidx):
                    self.gpu_cache.stats.prefetch_hits += 1
                    continue
                if key in self._prefetch_pending:
                    continue
                if self.gpu_cache.size >= self.gpu_cache.max_size:
                    continue

                if key not in self._cpu_experts:
                    continue

                gpu_params = self._materialize_expert(layer_idx, eidx)
                self.gpu_cache.put(layer_idx, eidx, gpu_params)
                self._prefetch_pending.add(key)

    def record_decode_experts(
        self,
        layer_idx: int,
        expert_indices: list[int],
    ) -> None:
        """Promote repeatedly used decode experts into a protected hot set."""
        for expert_idx in dict.fromkeys(expert_indices):
            key = (layer_idx, expert_idx)
            self._decode_access_count[key] = self._decode_access_count.get(key, 0) + 1

        if self.gpu_cache.protected_size <= 0:
            return

        hot_keys = [
            key
            for key, _ in sorted(
                self._decode_access_count.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        self.gpu_cache.set_protected(hot_keys)

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
            "gpu_cache_policy": self.gpu_cache.policy,
            "gpu_cache_protected": len(self.gpu_cache.protected_keys),
            "gpu_cache_memory_mb": self.gpu_cache.max_size * expert_size_mb,
            "pin_memory": self.pin_memory,
            "prefetch_enabled": self.prefetch,
            "cache_stats": self.gpu_cache.stats.summary(),
        }
