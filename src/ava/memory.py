from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryRecord:
    text: str
    surprise: float
    metadata: dict[str, Any] = field(default_factory=dict)
    accesses: int = 0


@dataclass
class TitansMemory:
    max_items: int = 256
    write_surprise_threshold: float = 0.45
    embedding_dim: int = 128
    records: list[MemoryRecord] = field(init=False, default_factory=list)
    vectors: list[list[float]] = field(init=False, default_factory=list)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.embedding_dim
        payload = text.encode("utf-8", errors="ignore")
        for index, value in enumerate(payload):
            vector[(value + index) % self.embedding_dim] += 1.0
            if index:
                vector[(payload[index - 1] * 31 + value) % self.embedding_dim] += 0.5
        norm = math.sqrt(sum(item * item for item in vector))
        if norm:
            vector = [item / norm for item in vector]
        return vector

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = set(left.lower().replace(".", " ").replace(",", " ").split())
        right_tokens = set(right.lower().replace(".", " ").replace(",", " ").split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def write(self, text: str, surprise: float, metadata: dict[str, Any] | None = None) -> bool:
        if surprise < self.write_surprise_threshold:
            return False
        record = MemoryRecord(text=text, surprise=surprise, metadata=metadata or {})
        vector = self._embed(text)
        if len(self.records) >= self.max_items:
            lowest_index = min(range(len(self.records)), key=lambda idx: self.records[idx].surprise)
            self.records[lowest_index] = record
            self.vectors[lowest_index] = vector
            return True
        self.records.append(record)
        self.vectors.append(vector)
        return True

    def retrieve(self, query: str, top_k: int = 4) -> list[MemoryRecord]:
        if not self.records:
            return []
        query_vector = self._embed(query)
        scored: list[tuple[float, MemoryRecord]] = []
        for vector, record in zip(self.vectors, self.records, strict=True):
            similarity = sum(a * b for a, b in zip(query_vector, vector, strict=True))
            overlap = self._token_overlap(query, record.text)
            score = similarity + (1.5 * overlap) + (0.1 * record.surprise)
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = [record for _, record in scored[:top_k]]
        for record in chosen:
            record.accesses += 1
        return chosen

    def summarize_context(self, query: str, top_k: int = 4) -> str:
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["Relevant memory:"]
        for record in hits:
            lines.append(f"- {record.text}")
        return "\n".join(lines)
