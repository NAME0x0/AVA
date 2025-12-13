"""
Memory Manager for AVA

Central orchestration of all memory systems including episodic and
semantic memory stores, consolidation, and context retrieval.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import (
    MemoryType,
    MemoryItem,
    EpisodicMemory,
    SemanticMemory,
    create_episodic_memory,
    create_semantic_memory,
)
from .episodic import EpisodicMemoryStore
from .semantic import SemanticMemoryStore
from .consolidation import MemoryConsolidator

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Central manager for AVA's memory systems.

    Orchestrates:
    - Episodic memory (interaction events)
    - Semantic memory (factual knowledge)
    - Memory consolidation (decay and strengthening)
    - Context retrieval for inference
    - Training sample export for fine-tuning
    """

    def __init__(
        self,
        data_path: str = "data/memory",
        consolidation_interval_hours: int = 24,
    ):
        """
        Initialize the memory manager.

        Args:
            data_path: Base path for all memory storage
            consolidation_interval_hours: Hours between consolidation cycles
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize stores
        self.episodic = EpisodicMemoryStore(
            data_path=str(self.data_path / "episodic")
        )
        self.semantic = SemanticMemoryStore(
            data_path=str(self.data_path / "semantic")
        )
        self.consolidator = MemoryConsolidator()

        # Consolidation tracking
        self.consolidation_interval = timedelta(hours=consolidation_interval_hours)
        self.last_consolidation: Optional[datetime] = None

        logger.info(
            f"MemoryManager initialized: "
            f"{len(self.episodic.memories)} episodic, "
            f"{len(self.semantic.memories)} semantic"
        )

    def store_episode(
        self,
        user_input: str,
        ava_response: str,
        outcome: str = "unknown",
        quality_score: float = 0.5,
        emotional_valence: float = 0.0,
        stage: int = 0,
        tool_calls: Optional[List[Dict]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store a new interaction as episodic memory.

        Args:
            user_input: The user's input
            ava_response: AVA's response
            outcome: "success", "partial", "failure", "unknown"
            quality_score: Quality rating (0.0 to 1.0)
            emotional_valence: Emotional context (-1.0 to 1.0)
            stage: Developmental stage when formed
            tool_calls: List of tool calls made
            tags: Tags for categorization

        Returns:
            Memory ID
        """
        memory = create_episodic_memory(
            user_input=user_input,
            ava_response=ava_response,
            outcome=outcome,
            quality_score=quality_score,
            emotional_valence=emotional_valence,
            stage=stage,
            tool_calls=tool_calls,
            tags=tags,
        )

        return self.episodic.store(memory)

    def store_fact(
        self,
        content: str,
        subject: str,
        predicate: str,
        obj: str,
        domain: str = "",
        source: str = "interaction",
        confidence: float = 0.5,
        stage: int = 0,
    ) -> str:
        """
        Store a new fact as semantic memory.

        Args:
            content: Full content/context
            subject: Subject of the fact
            predicate: Relationship/predicate
            obj: Object of the fact
            domain: Knowledge domain
            source: Where this came from
            confidence: Initial confidence
            stage: Developmental stage

        Returns:
            Memory ID
        """
        memory = create_semantic_memory(
            content=content,
            subject=subject,
            predicate=predicate,
            obj=obj,
            domain=domain,
            source=source,
            confidence=confidence,
            stage=stage,
        )

        return self.semantic.store(memory)

    def retrieve_relevant(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_strength: float = 0.1,
    ) -> List[Union[EpisodicMemory, SemanticMemory]]:
        """
        Retrieve memories relevant to a query.

        Searches both episodic and semantic stores and merges results.

        Args:
            query: Search query
            memory_types: Types to search (default: both)
            limit: Maximum results to return
            min_strength: Minimum memory strength

        Returns:
            List of relevant memories
        """
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC]

        results = []

        if MemoryType.EPISODIC in memory_types:
            episodic_results = self.episodic.search_by_content(
                query, limit=limit, min_strength=min_strength
            )
            results.extend(episodic_results)

        if MemoryType.SEMANTIC in memory_types:
            semantic_results = self.semantic.search_by_content(
                query, limit=limit, min_confidence=0.3
            )
            results.extend(semantic_results)

        # Sort by relevance and return top results
        results.sort(
            key=lambda m: m.compute_relevance_score(),
            reverse=True
        )

        return results[:limit]

    def get_context_window(
        self,
        current_input: str,
        max_tokens: int = 2000,
        include_recent: int = 3,
    ) -> str:
        """
        Build a context window from relevant memories.

        Creates a formatted context string suitable for including
        in the LLM prompt.

        Args:
            current_input: The current user input
            max_tokens: Approximate maximum tokens for context
            include_recent: Number of recent interactions to include

        Returns:
            Formatted context string
        """
        context_parts = []

        # Include recent conversation history
        recent = self.episodic.get_recent(include_recent)
        if recent:
            context_parts.append("## Recent Conversation")
            for mem in recent:
                context_parts.append(f"User: {mem.user_input}")
                context_parts.append(f"AVA: {mem.ava_response}")
                context_parts.append("")

        # Include relevant memories
        relevant = self.retrieve_relevant(current_input, limit=5)
        if relevant:
            context_parts.append("## Relevant Memories")
            for mem in relevant:
                if isinstance(mem, EpisodicMemory):
                    context_parts.append(f"[Past interaction] {mem.summary}")
                elif isinstance(mem, SemanticMemory):
                    context_parts.append(f"[Known fact] {mem.subject} {mem.predicate} {mem.object}")

        # Include relevant facts
        facts = self.semantic.search_by_content(current_input, limit=3)
        if facts:
            context_parts.append("## Relevant Knowledge")
            for fact in facts:
                context_parts.append(f"- {fact.subject} {fact.predicate} {fact.object}")

        context = "\n".join(context_parts)

        # Rough token limiting (4 chars per token approximation)
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n[...truncated]"

        return context

    def get_learning_context(
        self,
        user_input: str,
        tool_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get context specifically for learning/fine-tuning.

        Returns examples of similar past interactions that were successful.

        Args:
            user_input: Current input
            tool_name: Optional tool being used

        Returns:
            Learning context with examples
        """
        context = {
            "similar_successes": [],
            "similar_failures": [],
            "relevant_facts": [],
        }

        # Get similar successful interactions
        similar = self.episodic.search_by_content(user_input, limit=10)
        for mem in similar:
            if mem.outcome == "success":
                context["similar_successes"].append({
                    "input": mem.user_input,
                    "response": mem.ava_response,
                    "quality": mem.quality_score,
                })
            elif mem.outcome == "failure":
                context["similar_failures"].append({
                    "input": mem.user_input,
                    "response": mem.ava_response,
                    "what_went_wrong": mem.context.get("error", "unknown"),
                })

        # Get relevant facts
        facts = self.semantic.search_by_content(user_input, limit=5)
        context["relevant_facts"] = [
            f"{f.subject} {f.predicate} {f.object}"
            for f in facts
        ]

        return context

    def export_training_samples(
        self,
        sample_count: int = 100,
        min_quality: float = 0.6,
        include_failures: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Export memories as training samples for fine-tuning.

        Args:
            sample_count: Number of samples to export
            min_quality: Minimum quality threshold
            include_failures: Include failure examples

        Returns:
            List of training samples
        """
        samples = []

        # Get high-quality successes
        episodes = self.episodic.get_learning_samples(
            count=sample_count,
            include_failures=include_failures
        )

        for ep in episodes:
            if ep.quality_score >= min_quality or ep.outcome == "failure":
                sample = {
                    "input": ep.user_input,
                    "output": ep.ava_response,
                    "outcome": ep.outcome,
                    "quality": ep.quality_score,
                    "emotional_context": ep.emotional_valence,
                    "stage": ep.stage_when_formed,
                }

                # Include correction if available
                if ep.was_corrected and ep.correction_content:
                    sample["correction"] = ep.correction_content

                samples.append(sample)

        return samples

    def add_user_feedback(
        self,
        memory_id: str,
        feedback: str,  # "positive", "negative", "neutral"
        correction: Optional[str] = None,
    ):
        """
        Add user feedback to a memory.

        Args:
            memory_id: ID of the memory
            feedback: Type of feedback
            correction: Optional corrected response
        """
        updates = {"user_feedback": feedback}
        if correction:
            updates["was_corrected"] = True
            updates["correction_content"] = correction

        self.episodic.update_memory(memory_id, updates)

    def check_consolidation(self):
        """Check if consolidation should run and execute if needed."""
        now = datetime.now()

        if self.last_consolidation is None:
            should_consolidate = True
        else:
            should_consolidate = (now - self.last_consolidation) > self.consolidation_interval

        if should_consolidate:
            self.run_consolidation()

    def run_consolidation(self):
        """Run memory consolidation."""
        self.consolidator.run_consolidation_cycle(
            self.episodic,
            self.semantic,
            extract_knowledge=True
        )
        self.last_consolidation = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "episodic": self.episodic.get_stats(),
            "semantic": self.semantic.get_stats(),
            "last_consolidation": (
                self.last_consolidation.isoformat()
                if self.last_consolidation else None
            ),
        }

    def clear_all(self):
        """Clear all memories (use with caution!)."""
        import shutil

        # Clear episodic
        episodic_path = self.data_path / "episodic"
        if episodic_path.exists():
            shutil.rmtree(episodic_path)
        episodic_path.mkdir(parents=True)

        # Clear semantic
        semantic_path = self.data_path / "semantic"
        if semantic_path.exists():
            shutil.rmtree(semantic_path)
        semantic_path.mkdir(parents=True)

        # Reinitialize stores
        self.episodic = EpisodicMemoryStore(str(episodic_path))
        self.semantic = SemanticMemoryStore(str(semantic_path))

        logger.warning("All memories cleared!")
