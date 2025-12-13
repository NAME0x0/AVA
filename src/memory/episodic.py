"""
Episodic Memory Store for AVA

Manages storage and retrieval of episodic (event-based) memories.
Episodic memories record specific interactions and experiences.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import EpisodicMemory, MemoryItem

logger = logging.getLogger(__name__)


class EpisodicMemoryStore:
    """
    Storage and retrieval system for episodic memories.

    Memories are persisted to disk and indexed for efficient retrieval.
    Supports retrieval by recency, relevance, and emotional context.
    """

    def __init__(
        self,
        data_path: str = "data/memory/episodic",
        max_memories: int = 10000,
    ):
        """
        Initialize the episodic memory store.

        Args:
            data_path: Path for storing memory files
            max_memories: Maximum number of memories to retain
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.max_memories = max_memories
        self.memories: Dict[str, EpisodicMemory] = {}
        self.index_by_time: List[str] = []  # Memory IDs sorted by time

        self._load_memories()
        logger.info(f"EpisodicMemoryStore loaded {len(self.memories)} memories")

    def _load_memories(self):
        """Load memories from disk."""
        index_file = self.data_path / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    index_data = json.load(f)
                    self.index_by_time = index_data.get("time_index", [])

                # Load individual memory files
                for memory_id in self.index_by_time:
                    memory_file = self.data_path / f"{memory_id}.json"
                    if memory_file.exists():
                        with open(memory_file, "r") as f:
                            data = json.load(f)
                            self.memories[memory_id] = EpisodicMemory.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading episodic memories: {e}")

    def save_index(self):
        """Save the memory index to disk."""
        index_file = self.data_path / "index.json"
        index_data = {
            "time_index": self.index_by_time,
            "total_count": len(self.memories),
            "last_updated": datetime.now().isoformat(),
        }
        with open(index_file, "w") as f:
            json.dump(index_data, f, indent=2)

    def store(self, memory: EpisodicMemory) -> str:
        """
        Store a new episodic memory.

        Args:
            memory: The memory to store

        Returns:
            The memory ID
        """
        # Check capacity and prune if needed
        if len(self.memories) >= self.max_memories:
            self._prune_oldest()

        # Store in memory
        self.memories[memory.memory_id] = memory
        self.index_by_time.append(memory.memory_id)

        # Persist to disk
        memory_file = self.data_path / f"{memory.memory_id}.json"
        with open(memory_file, "w") as f:
            json.dump(memory.to_dict(), f, indent=2)

        self.save_index()
        logger.debug(f"Stored episodic memory: {memory.memory_id}")

        return memory.memory_id

    def get(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Retrieve a memory by ID."""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access()
            # Update on disk
            memory_file = self.data_path / f"{memory_id}.json"
            with open(memory_file, "w") as f:
                json.dump(memory.to_dict(), f, indent=2)
        return memory

    def get_recent(self, count: int = 10) -> List[EpisodicMemory]:
        """Get the most recent memories."""
        recent_ids = self.index_by_time[-count:]
        return [self.memories[mid] for mid in reversed(recent_ids) if mid in self.memories]

    def search_by_content(
        self,
        query: str,
        limit: int = 10,
        min_strength: float = 0.1,
    ) -> List[EpisodicMemory]:
        """
        Search memories by content similarity.

        Simple keyword matching for now - could be enhanced with embeddings.

        Args:
            query: Search query
            limit: Maximum results to return
            min_strength: Minimum memory strength threshold
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []
        for memory in self.memories.values():
            if memory.strength < min_strength:
                continue

            # Simple keyword matching
            content_lower = (memory.content + " " + memory.summary).lower()
            content_words = set(content_lower.split())

            # Calculate overlap
            overlap = len(query_words & content_words)
            if overlap > 0:
                # Score based on overlap and memory relevance
                match_score = overlap / len(query_words)
                relevance = memory.compute_relevance_score()
                combined_score = match_score * 0.6 + relevance * 0.4

                results.append((combined_score, memory))

        # Sort by score and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in results[:limit]]

    def search_by_emotion(
        self,
        valence_range: tuple = (-1.0, 1.0),
        min_intensity: float = 0.0,
        limit: int = 10,
    ) -> List[EpisodicMemory]:
        """
        Search memories by emotional characteristics.

        Args:
            valence_range: (min, max) valence to include
            min_intensity: Minimum emotional intensity
            limit: Maximum results to return
        """
        results = []
        for memory in self.memories.values():
            if (valence_range[0] <= memory.emotional_valence <= valence_range[1] and
                memory.emotional_intensity >= min_intensity):
                results.append(memory)

        # Sort by intensity and recency
        results.sort(
            key=lambda m: (m.emotional_intensity, -m.access_count),
            reverse=True
        )
        return results[:limit]

    def search_by_outcome(
        self,
        outcome: str,
        limit: int = 10,
    ) -> List[EpisodicMemory]:
        """Search memories by interaction outcome."""
        results = [m for m in self.memories.values() if m.outcome == outcome]
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results[:limit]

    def get_successful_patterns(self, limit: int = 20) -> List[EpisodicMemory]:
        """Get high-quality successful interactions for learning."""
        successes = [
            m for m in self.memories.values()
            if m.outcome == "success" and m.quality_score > 0.7
        ]
        successes.sort(key=lambda m: m.quality_score, reverse=True)
        return successes[:limit]

    def get_learning_samples(
        self,
        count: int = 50,
        include_failures: bool = True,
    ) -> List[EpisodicMemory]:
        """
        Get a balanced set of memories for learning/fine-tuning.

        Args:
            count: Total samples to return
            include_failures: Whether to include failure examples
        """
        successes = [m for m in self.memories.values() if m.outcome == "success"]
        successes.sort(key=lambda m: m.quality_score, reverse=True)

        if include_failures:
            failures = [m for m in self.memories.values() if m.outcome == "failure"]
            failures.sort(key=lambda m: m.importance_score, reverse=True)

            # Balance: 70% successes, 30% failures
            success_count = int(count * 0.7)
            failure_count = count - success_count

            return successes[:success_count] + failures[:failure_count]
        else:
            return successes[:count]

    def _prune_oldest(self):
        """Remove oldest, weakest memories to make room."""
        # Remove 10% of memories
        prune_count = max(1, len(self.memories) // 10)

        # Sort by combined age and weakness
        sortable = []
        for mid, mem in self.memories.items():
            age_days = (datetime.now() - mem.created_at).days
            score = mem.strength - (age_days * 0.01) + (mem.importance_score * 0.5)
            sortable.append((score, mid))

        sortable.sort()

        # Remove lowest scoring memories
        for _, mid in sortable[:prune_count]:
            del self.memories[mid]
            self.index_by_time.remove(mid)

            # Remove file
            memory_file = self.data_path / f"{mid}.json"
            if memory_file.exists():
                memory_file.unlink()

        self.save_index()
        logger.info(f"Pruned {prune_count} old memories")

    def update_memory(self, memory_id: str, updates: Dict[str, Any]):
        """Update a memory's attributes."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            for key, value in updates.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)

            # Save to disk
            memory_file = self.data_path / f"{memory_id}.json"
            with open(memory_file, "w") as f:
                json.dump(memory.to_dict(), f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        if not self.memories:
            return {"total": 0}

        outcomes = {"success": 0, "failure": 0, "partial": 0, "unknown": 0}
        total_strength = 0
        total_quality = 0

        for mem in self.memories.values():
            outcomes[mem.outcome] = outcomes.get(mem.outcome, 0) + 1
            total_strength += mem.strength
            total_quality += mem.quality_score

        return {
            "total": len(self.memories),
            "outcomes": outcomes,
            "avg_strength": total_strength / len(self.memories),
            "avg_quality": total_quality / len(self.memories),
            "oldest": min(m.created_at for m in self.memories.values()).isoformat(),
            "newest": max(m.created_at for m in self.memories.values()).isoformat(),
        }
