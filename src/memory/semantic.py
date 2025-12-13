"""
Semantic Memory Store for AVA

Manages storage and retrieval of semantic (factual knowledge) memories.
Semantic memories store learned facts and concepts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import SemanticMemory

logger = logging.getLogger(__name__)


class SemanticMemoryStore:
    """
    Storage and retrieval system for semantic memories.

    Semantic memories represent factual knowledge that AVA has learned
    or inferred from interactions. They are organized by domain and
    support confidence-based retrieval.
    """

    def __init__(
        self,
        data_path: str = "data/memory/semantic",
        max_memories: int = 50000,
    ):
        """
        Initialize the semantic memory store.

        Args:
            data_path: Path for storing memory files
            max_memories: Maximum number of facts to retain
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.max_memories = max_memories
        self.memories: Dict[str, SemanticMemory] = {}

        # Indexes for efficient retrieval
        self.index_by_domain: Dict[str, List[str]] = {}
        self.index_by_subject: Dict[str, List[str]] = {}

        self._load_memories()
        logger.info(f"SemanticMemoryStore loaded {len(self.memories)} facts")

    def _load_memories(self):
        """Load memories from disk."""
        memories_file = self.data_path / "semantic_memories.json"
        if memories_file.exists():
            try:
                with open(memories_file, "r") as f:
                    data = json.load(f)

                for mem_data in data.get("memories", []):
                    mem = SemanticMemory.from_dict(mem_data)
                    self.memories[mem.memory_id] = mem
                    self._index_memory(mem)

            except Exception as e:
                logger.error(f"Error loading semantic memories: {e}")

    def _save_memories(self):
        """Save all memories to disk."""
        memories_file = self.data_path / "semantic_memories.json"
        data = {
            "memories": [m.to_dict() for m in self.memories.values()],
            "total_count": len(self.memories),
            "last_updated": datetime.now().isoformat(),
        }
        with open(memories_file, "w") as f:
            json.dump(data, f, indent=2)

    def _index_memory(self, memory: SemanticMemory):
        """Add memory to indexes."""
        # Index by domain
        if memory.domain:
            if memory.domain not in self.index_by_domain:
                self.index_by_domain[memory.domain] = []
            if memory.memory_id not in self.index_by_domain[memory.domain]:
                self.index_by_domain[memory.domain].append(memory.memory_id)

        # Index by subject
        if memory.subject:
            subject_key = memory.subject.lower()
            if subject_key not in self.index_by_subject:
                self.index_by_subject[subject_key] = []
            if memory.memory_id not in self.index_by_subject[subject_key]:
                self.index_by_subject[subject_key].append(memory.memory_id)

    def store(self, memory: SemanticMemory) -> str:
        """
        Store a new semantic memory.

        If a similar fact exists, updates confidence instead.

        Args:
            memory: The memory to store

        Returns:
            The memory ID
        """
        # Check for existing similar fact
        existing = self.find_similar(memory.subject, memory.predicate, memory.object)
        if existing:
            # Update existing rather than duplicate
            existing.update_confidence(confirmed=True)
            existing.access()
            self._save_memories()
            return existing.memory_id

        # Check capacity
        if len(self.memories) >= self.max_memories:
            self._prune_low_confidence()

        # Store new memory
        self.memories[memory.memory_id] = memory
        self._index_memory(memory)
        self._save_memories()

        logger.debug(f"Stored semantic memory: {memory.summary}")
        return memory.memory_id

    def find_similar(
        self,
        subject: str,
        predicate: str,
        obj: str,
        threshold: float = 0.8,
    ) -> Optional[SemanticMemory]:
        """
        Find a memory with similar subject-predicate-object.

        Args:
            subject: Subject to match
            predicate: Predicate to match
            obj: Object to match
            threshold: Similarity threshold (simple matching for now)
        """
        subject_lower = subject.lower()
        if subject_lower in self.index_by_subject:
            for mid in self.index_by_subject[subject_lower]:
                mem = self.memories.get(mid)
                if mem and mem.predicate.lower() == predicate.lower():
                    if mem.object.lower() == obj.lower():
                        return mem
        return None

    def get(self, memory_id: str) -> Optional[SemanticMemory]:
        """Retrieve a memory by ID."""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access()
        return memory

    def search_by_subject(
        self,
        subject: str,
        min_confidence: float = 0.3,
        limit: int = 10,
    ) -> List[SemanticMemory]:
        """Search memories by subject."""
        subject_key = subject.lower()
        results = []

        if subject_key in self.index_by_subject:
            for mid in self.index_by_subject[subject_key]:
                mem = self.memories.get(mid)
                if mem and mem.confidence >= min_confidence:
                    results.append(mem)

        # Also do partial matching
        for key, mids in self.index_by_subject.items():
            if subject_key in key or key in subject_key:
                for mid in mids:
                    mem = self.memories.get(mid)
                    if mem and mem.confidence >= min_confidence and mem not in results:
                        results.append(mem)

        results.sort(key=lambda m: m.confidence, reverse=True)
        return results[:limit]

    def search_by_domain(
        self,
        domain: str,
        min_confidence: float = 0.3,
        limit: int = 20,
    ) -> List[SemanticMemory]:
        """Search memories by knowledge domain."""
        results = []

        if domain in self.index_by_domain:
            for mid in self.index_by_domain[domain]:
                mem = self.memories.get(mid)
                if mem and mem.confidence >= min_confidence:
                    results.append(mem)

        results.sort(key=lambda m: m.confidence, reverse=True)
        return results[:limit]

    def search_by_content(
        self,
        query: str,
        min_confidence: float = 0.3,
        limit: int = 10,
    ) -> List[SemanticMemory]:
        """Search memories by content matching."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []
        for mem in self.memories.values():
            if mem.confidence < min_confidence:
                continue

            # Match against subject, predicate, object, and content
            searchable = f"{mem.subject} {mem.predicate} {mem.object} {mem.content}".lower()
            searchable_words = set(searchable.split())

            overlap = len(query_words & searchable_words)
            if overlap > 0:
                score = overlap / len(query_words) * mem.confidence
                results.append((score, mem))

        results.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in results[:limit]]

    def get_facts_about(self, subject: str) -> List[Dict[str, str]]:
        """Get all known facts about a subject in structured format."""
        memories = self.search_by_subject(subject, min_confidence=0.4)
        return [
            {
                "subject": m.subject,
                "predicate": m.predicate,
                "object": m.object,
                "confidence": m.confidence,
            }
            for m in memories
        ]

    def confirm_fact(self, memory_id: str):
        """Confirm a fact is correct (strengthens confidence)."""
        if memory_id in self.memories:
            self.memories[memory_id].update_confidence(confirmed=True)
            self._save_memories()

    def contradict_fact(self, memory_id: str):
        """Mark a fact as contradicted (weakens confidence)."""
        if memory_id in self.memories:
            self.memories[memory_id].update_confidence(confirmed=False)
            self._save_memories()

    def _prune_low_confidence(self):
        """Remove lowest confidence memories to make room."""
        # Remove 10% of memories
        prune_count = max(1, len(self.memories) // 10)

        # Sort by confidence and strength
        sortable = [
            (mem.confidence * mem.strength, mid)
            for mid, mem in self.memories.items()
        ]
        sortable.sort()

        # Remove lowest scoring
        for _, mid in sortable[:prune_count]:
            mem = self.memories[mid]

            # Remove from indexes
            if mem.domain in self.index_by_domain:
                if mid in self.index_by_domain[mem.domain]:
                    self.index_by_domain[mem.domain].remove(mid)

            subject_key = mem.subject.lower()
            if subject_key in self.index_by_subject:
                if mid in self.index_by_subject[subject_key]:
                    self.index_by_subject[subject_key].remove(mid)

            del self.memories[mid]

        self._save_memories()
        logger.info(f"Pruned {prune_count} low-confidence facts")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        if not self.memories:
            return {"total": 0}

        confidences = [m.confidence for m in self.memories.values()]

        return {
            "total": len(self.memories),
            "domains": list(self.index_by_domain.keys()),
            "domain_counts": {d: len(mids) for d, mids in self.index_by_domain.items()},
            "avg_confidence": sum(confidences) / len(confidences),
            "high_confidence_count": sum(1 for c in confidences if c > 0.7),
            "low_confidence_count": sum(1 for c in confidences if c < 0.4),
        }
