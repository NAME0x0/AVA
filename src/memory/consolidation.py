"""
Memory Consolidation for AVA

Implements memory decay and strengthening processes that mirror
human memory consolidation during sleep/rest.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .models import MemoryItem, EpisodicMemory, SemanticMemory

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Handles memory consolidation processes.

    Consolidation includes:
    - Decay of unused memories (forgetting)
    - Strengthening of frequently accessed memories
    - Extraction of semantic knowledge from repeated episodic patterns
    """

    def __init__(
        self,
        decay_rate_per_day: float = 0.05,
        min_strength_threshold: float = 0.1,
        access_strengthen_amount: float = 0.1,
    ):
        """
        Initialize the consolidator.

        Args:
            decay_rate_per_day: How much strength decreases per day
            min_strength_threshold: Minimum strength before memory is forgotten
            access_strengthen_amount: How much accessing a memory strengthens it
        """
        self.decay_rate_per_day = decay_rate_per_day
        self.min_strength_threshold = min_strength_threshold
        self.access_strengthen_amount = access_strengthen_amount

    def consolidate_episodic(
        self,
        memories: Dict[str, EpisodicMemory],
    ) -> Dict[str, EpisodicMemory]:
        """
        Run consolidation on episodic memories.

        Args:
            memories: Dictionary of memories to consolidate

        Returns:
            Consolidated memories (weak ones removed)
        """
        now = datetime.now()
        consolidated = {}
        removed_count = 0

        for mid, memory in memories.items():
            # Calculate decay based on time since last access
            days_since_access = (now - memory.last_accessed).days
            decay_amount = self.decay_rate_per_day * days_since_access

            # Apply decay
            memory.decay(decay_amount)

            # Keep if above threshold or important
            if memory.strength >= self.min_strength_threshold:
                consolidated[mid] = memory
            elif memory.importance_score > 0.7:
                # Important memories resist decay
                memory.strength = self.min_strength_threshold
                consolidated[mid] = memory
            else:
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Consolidation removed {removed_count} weak episodic memories")

        return consolidated

    def consolidate_semantic(
        self,
        memories: Dict[str, SemanticMemory],
    ) -> Dict[str, SemanticMemory]:
        """
        Run consolidation on semantic memories.

        Semantic memories decay more slowly and based on confidence.

        Args:
            memories: Dictionary of memories to consolidate

        Returns:
            Consolidated memories (weak ones removed)
        """
        now = datetime.now()
        consolidated = {}
        removed_count = 0

        for mid, memory in memories.items():
            # Semantic decay is slower and tied to confidence
            days_since_access = (now - memory.last_accessed).days
            decay_amount = (self.decay_rate_per_day * 0.5) * days_since_access

            # Low confidence facts decay faster
            if memory.confidence < 0.4:
                decay_amount *= 2

            memory.decay(decay_amount)

            # Keep if has reasonable strength and confidence
            if memory.strength >= self.min_strength_threshold:
                consolidated[mid] = memory
            elif memory.confidence > 0.7:
                # High confidence facts resist decay
                memory.strength = self.min_strength_threshold
                consolidated[mid] = memory
            else:
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Consolidation removed {removed_count} weak semantic memories")

        return consolidated

    def extract_patterns(
        self,
        episodic_memories: List[EpisodicMemory],
        min_occurrences: int = 3,
    ) -> List[Dict]:
        """
        Extract patterns from episodic memories that could become semantic.

        Looks for repeated patterns in successful interactions that
        could be generalized into factual knowledge.

        Args:
            episodic_memories: List of episodic memories to analyze
            min_occurrences: Minimum occurrences to consider a pattern

        Returns:
            List of extracted patterns
        """
        # Group by similar user inputs
        input_groups: Dict[str, List[EpisodicMemory]] = {}

        for mem in episodic_memories:
            if mem.outcome != "success":
                continue

            # Simple grouping by first few words
            key = " ".join(mem.user_input.lower().split()[:3])
            if key not in input_groups:
                input_groups[key] = []
            input_groups[key].append(mem)

        # Find patterns with enough occurrences
        patterns = []
        for key, group in input_groups.items():
            if len(group) >= min_occurrences:
                # Extract common elements
                avg_quality = sum(m.quality_score for m in group) / len(group)

                pattern = {
                    "pattern_key": key,
                    "occurrences": len(group),
                    "avg_quality": avg_quality,
                    "example_inputs": [m.user_input for m in group[:3]],
                    "example_responses": [m.ava_response for m in group[:3]],
                }
                patterns.append(pattern)

        logger.info(f"Extracted {len(patterns)} patterns from episodic memories")
        return patterns

    def strengthen_on_access(self, memory: MemoryItem):
        """Strengthen a memory when it's accessed."""
        memory.strength = min(1.0, memory.strength + self.access_strengthen_amount)
        memory.last_accessed = datetime.now()
        memory.access_count += 1

    def calculate_memory_importance(
        self,
        memory: MemoryItem,
        emotional_weight: float = 0.3,
        recency_weight: float = 0.2,
        access_weight: float = 0.2,
        quality_weight: float = 0.3,
    ) -> float:
        """
        Calculate overall importance score for a memory.

        Args:
            memory: Memory to evaluate
            emotional_weight: Weight for emotional intensity
            recency_weight: Weight for recency
            access_weight: Weight for access frequency
            quality_weight: Weight for quality (if episodic)

        Returns:
            Importance score (0.0 to 1.0)
        """
        # Emotional importance
        emotional_score = memory.emotional_intensity

        # Recency score
        days_ago = (datetime.now() - memory.created_at).days
        recency_score = max(0.0, 1.0 - (days_ago * 0.01))

        # Access frequency (log scale)
        import math
        access_score = min(1.0, math.log(memory.access_count + 1) / 5)

        # Quality score (for episodic memories)
        quality_score = 0.5
        if isinstance(memory, EpisodicMemory):
            quality_score = memory.quality_score

        importance = (
            emotional_weight * emotional_score +
            recency_weight * recency_score +
            access_weight * access_score +
            quality_weight * quality_score
        )

        return min(1.0, max(0.0, importance))

    def run_consolidation_cycle(
        self,
        episodic_store,
        semantic_store,
        extract_knowledge: bool = True,
    ):
        """
        Run a complete consolidation cycle.

        This should be called periodically (e.g., daily or after many interactions).

        Args:
            episodic_store: EpisodicMemoryStore instance
            semantic_store: SemanticMemoryStore instance
            extract_knowledge: Whether to extract patterns to semantic memory
        """
        logger.info("Starting memory consolidation cycle")

        # Consolidate episodic memories
        episodic_store.memories = self.consolidate_episodic(episodic_store.memories)
        episodic_store.save_index()

        # Consolidate semantic memories
        semantic_store.memories = self.consolidate_semantic(semantic_store.memories)
        semantic_store._save_memories()

        # Extract patterns if requested
        if extract_knowledge:
            patterns = self.extract_patterns(list(episodic_store.memories.values()))
            # Could convert patterns to semantic memories here
            logger.info(f"Found {len(patterns)} potential knowledge patterns")

        logger.info("Memory consolidation cycle complete")
