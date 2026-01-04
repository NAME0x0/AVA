"""
Adapter Manager - Dynamic LoRA Expert Swapping
===============================================

Manages specialist LoRA adapters for different query domains.
Enables hot-swapping of adapters during inference to use
domain-specific experts without reloading the base model.

Specialists:
- coding: Software development, debugging, code explanation
- logic: Logical reasoning, proofs, deduction
- butler: Conversational assistance, scheduling, general help

Architecture:
- Adapters stored in System RAM (~100-500 MB each)
- Hot-swapped into VRAM buffer during layer iteration
- Base model layers remain in system RAM via AirLLM
- ~50-100ms swap time per adapter
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AdapterType(Enum):
    """Available specialist adapter types."""

    GENERAL = auto()  # Base model only, no adapter
    CODING = auto()  # Software development specialist
    LOGIC = auto()  # Logical reasoning specialist
    BUTLER = auto()  # Conversational assistant


@dataclass
class AdapterConfig:
    """Configuration for the Adapter Manager."""

    # Adapter paths (relative to models/adapters/)
    adapter_base_path: str = "models/adapters"

    # Specialist adapter names
    coding_adapter: str = "coding_expert"
    logic_adapter: str = "logic_expert"
    butler_adapter: str = "butler_expert"

    # Adapter selection thresholds
    confidence_threshold: float = 0.6  # Min confidence to use specialist

    # Performance tracking
    track_adapter_performance: bool = True

    # Fallback behavior
    fallback_to_general: bool = True  # Use base model if adapter fails


@dataclass
class AdapterStats:
    """Statistics for an individual adapter."""

    name: str
    load_count: int = 0
    unload_count: int = 0
    total_generations: int = 0
    total_tokens_generated: int = 0
    average_load_time_ms: float = 0.0
    last_used: datetime | None = None

    def record_load(self, load_time_ms: float) -> None:
        """Record an adapter load event."""
        self.load_count += 1
        # Running average
        self.average_load_time_ms = (
            self.average_load_time_ms * (self.load_count - 1) + load_time_ms
        ) / self.load_count

    def record_generation(self, tokens: int) -> None:
        """Record a generation using this adapter."""
        self.total_generations += 1
        self.total_tokens_generated += tokens
        self.last_used = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "load_count": self.load_count,
            "unload_count": self.unload_count,
            "total_generations": self.total_generations,
            "total_tokens_generated": self.total_tokens_generated,
            "average_load_time_ms": self.average_load_time_ms,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


class AdapterManager:
    """
    Manages LoRA adapter lifecycle for specialist domains.

    This class handles:
    - Query classification to select appropriate specialist
    - Adapter loading/unloading (hot-swap)
    - Performance tracking per adapter
    - Graceful fallback to base model

    Usage:
        manager = AdapterManager(config)

        # For each query:
        adapter_type = manager.classify_query(prompt)
        await manager.ensure_adapter_loaded(model, adapter_type)

        # Generate with model (adapter is now applied)
        response = model.generate(prompt)

        # Record stats
        manager.record_generation(adapter_type, len(response))
    """

    # Adapter type to path mapping
    ADAPTER_PATHS = {
        AdapterType.CODING: "coding_expert",
        AdapterType.LOGIC: "logic_expert",
        AdapterType.BUTLER: "butler_expert",
        AdapterType.GENERAL: None,  # No adapter
    }

    def __init__(self, config: AdapterConfig | None = None):
        """
        Initialize the Adapter Manager.

        Args:
            config: Adapter configuration
        """
        self.config = config or AdapterConfig()

        # Current state
        self.active_adapter: AdapterType = AdapterType.GENERAL
        self._adapter_loaded: bool = False

        # Loaded adapter reference (for unloading)
        self._current_adapter_weights: Any | None = None

        # Statistics per adapter
        self.stats: dict[AdapterType, AdapterStats] = {
            adapter_type: AdapterStats(name=adapter_type.name) for adapter_type in AdapterType
        }

        # Lock for thread-safe swapping
        self._swap_lock = asyncio.Lock()

        logger.info("AdapterManager initialized")

    def classify_query(self, prompt: str) -> tuple[AdapterType, float]:
        """
        Classify a query to determine the best specialist adapter.

        Uses keyword matching and pattern detection to determine
        which domain expert is most appropriate.

        Args:
            prompt: Input prompt text

        Returns:
            Tuple of (AdapterType, confidence score)
        """
        prompt_lower = prompt.lower()

        # Score each adapter type
        scores = {
            AdapterType.CODING: self._score_coding(prompt_lower),
            AdapterType.LOGIC: self._score_logic(prompt_lower),
            AdapterType.BUTLER: self._score_butler(prompt_lower),
            AdapterType.GENERAL: 0.3,  # Base confidence for general
        }

        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Apply threshold
        if best_score < self.config.confidence_threshold:
            logger.debug(f"No specialist confident enough ({best_score:.2f}), using GENERAL")
            return AdapterType.GENERAL, best_score

        logger.debug(f"Classified query as {best_type.name} (confidence: {best_score:.2f})")
        return best_type, best_score

    def _score_coding(self, text: str) -> float:
        """Score text for coding domain relevance."""
        score = 0.0

        # Strong indicators
        strong_keywords = [
            "code",
            "function",
            "class",
            "method",
            "variable",
            "debug",
            "error",
            "exception",
            "bug",
            "fix",
            "python",
            "javascript",
            "java",
            "rust",
            "golang",
            "api",
            "endpoint",
            "database",
            "sql",
            "query",
            "git",
            "commit",
            "branch",
            "merge",
            "repository",
            "import",
            "export",
            "module",
            "package",
            "library",
            "compile",
            "build",
            "deploy",
            "test",
            "unittest",
        ]

        # Medium indicators
        medium_keywords = [
            "program",
            "script",
            "algorithm",
            "data structure",
            "loop",
            "array",
            "list",
            "dictionary",
            "object",
            "string",
            "integer",
            "float",
            "boolean",
            "type",
            "frontend",
            "backend",
            "server",
            "client",
            "http",
        ]

        # Count matches
        for kw in strong_keywords:
            if kw in text:
                score += 0.15

        for kw in medium_keywords:
            if kw in text:
                score += 0.08

        # Code patterns
        import re

        if re.search(r"```[\w]*\n", text):  # Code blocks
            score += 0.3
        if re.search(r"def\s+\w+|class\s+\w+|function\s+\w+", text):  # Definitions
            score += 0.25
        if re.search(r"\w+\.\w+\(", text):  # Method calls
            score += 0.15

        return min(score, 1.0)

    def _score_logic(self, text: str) -> float:
        """Score text for logical reasoning domain relevance."""
        score = 0.0

        # Strong indicators
        strong_keywords = [
            "prove",
            "proof",
            "theorem",
            "lemma",
            "corollary",
            "deduce",
            "deduction",
            "infer",
            "inference",
            "therefore",
            "hence",
            "thus",
            "consequently",
            "if and only if",
            "implies",
            "equivalent",
            "valid",
            "invalid",
            "sound",
            "unsound",
            "premise",
            "conclusion",
            "argument",
            "logical",
            "fallacy",
            "contradiction",
        ]

        # Medium indicators
        medium_keywords = [
            "assume",
            "suppose",
            "given",
            "show that",
            "follows",
            "derive",
            "reason",
            "reasoning",
            "true",
            "false",
            "truth table",
            "necessary",
            "sufficient",
            "condition",
            "all",
            "some",
            "none",
            "every",
            "exists",
        ]

        # Count matches
        for kw in strong_keywords:
            if kw in text:
                score += 0.18

        for kw in medium_keywords:
            if kw in text:
                score += 0.1

        # Logic patterns
        import re

        if re.search(r"if\s+.+\s+then\s+", text):  # If-then statements
            score += 0.2
        if re.search(r"∀|∃|¬|∧|∨|→|↔", text):  # Logic symbols
            score += 0.3

        return min(score, 1.0)

    def _score_butler(self, text: str) -> float:
        """Score text for conversational assistant domain relevance."""
        score = 0.0

        # Strong indicators
        strong_keywords = [
            "schedule",
            "appointment",
            "meeting",
            "calendar",
            "remind",
            "reminder",
            "notification",
            "alert",
            "weather",
            "temperature",
            "forecast",
            "news",
            "headlines",
            "update",
            "recommend",
            "suggestion",
            "advice",
            "help me",
            "assist",
            "please",
        ]

        # Medium indicators
        medium_keywords = [
            "today",
            "tomorrow",
            "this week",
            "next",
            "find",
            "search",
            "look up",
            "check",
            "set",
            "turn on",
            "turn off",
            "enable",
            "disable",
            "what time",
            "how long",
            "when is",
            "thank",
            "thanks",
            "appreciate",
        ]

        # Count matches
        for kw in strong_keywords:
            if kw in text:
                score += 0.15

        for kw in medium_keywords:
            if kw in text:
                score += 0.08

        # Conversational patterns
        if text.startswith(("can you", "could you", "would you", "will you")):
            score += 0.2
        if "?" in text and len(text) < 100:  # Short questions
            score += 0.1

        return min(score, 1.0)

    async def ensure_adapter_loaded(
        self,
        model: Any,
        adapter_type: AdapterType,
    ) -> bool:
        """
        Ensure the specified adapter is loaded on the model.

        If a different adapter is currently loaded, it will be
        unloaded first. This is a no-op if the correct adapter
        is already loaded.

        Args:
            model: The base model to apply adapter to
            adapter_type: Type of adapter to load

        Returns:
            True if adapter is ready, False on failure
        """
        async with self._swap_lock:
            # Already loaded?
            if self.active_adapter == adapter_type and self._adapter_loaded:
                return True

            # Need to swap
            if self.active_adapter != adapter_type:
                # Unload current adapter
                if self._adapter_loaded and self.active_adapter != AdapterType.GENERAL:
                    await self._unload_adapter(model)

                # Load new adapter (unless GENERAL)
                if adapter_type != AdapterType.GENERAL:
                    success = await self._load_adapter(model, adapter_type)
                    if not success and self.config.fallback_to_general:
                        logger.warning("Falling back to GENERAL adapter")
                        self.active_adapter = AdapterType.GENERAL
                        self._adapter_loaded = False
                        return True
                    return success
                else:
                    self.active_adapter = AdapterType.GENERAL
                    self._adapter_loaded = False

            return True

    async def _load_adapter(
        self,
        model: Any,
        adapter_type: AdapterType,
    ) -> bool:
        """
        Load a LoRA adapter onto the model.

        Args:
            model: Base model
            adapter_type: Adapter to load

        Returns:
            True on success
        """
        import time

        start_time = time.time()

        adapter_name = self.ADAPTER_PATHS.get(adapter_type)
        if not adapter_name:
            return False

        adapter_path = Path(self.config.adapter_base_path) / adapter_name

        try:
            # Check if adapter exists
            if not adapter_path.exists():
                logger.warning(f"Adapter not found: {adapter_path}")
                return False

            # NOTE: PEFT integration not yet implemented
            # Real adapter loading requires Phase 7 completion (native model loading)
            # For now, track adapter state for when integration is ready
            logger.info(f"Loading adapter: {adapter_name} (simulated)")

            self._current_adapter_weights = adapter_name
            self.active_adapter = adapter_type
            self._adapter_loaded = True

            # Record stats
            load_time_ms = (time.time() - start_time) * 1000
            self.stats[adapter_type].record_load(load_time_ms)

            logger.info(f"Adapter {adapter_name} registered in {load_time_ms:.0f}ms")
            logger.debug("Real PEFT loading pending native model implementation")
            return True

        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_name}: {e}")
            return False

    async def _unload_adapter(self, model: Any) -> None:
        """
        Unload the current adapter from the model.

        Args:
            model: Model to remove adapter from
        """
        if not self._adapter_loaded or self.active_adapter == AdapterType.GENERAL:
            return

        try:
            logger.info(f"Unloading adapter: {self.active_adapter.name}")

            # Record stats
            self.stats[self.active_adapter].unload_count += 1

            # In real implementation, this would remove PEFT weights
            # model.unload()

            self._current_adapter_weights = None
            self._adapter_loaded = False

        except Exception as e:
            logger.error(f"Failed to unload adapter: {e}")

    def record_generation(
        self,
        adapter_type: AdapterType,
        tokens_generated: int,
    ) -> None:
        """
        Record a generation event for statistics.

        Args:
            adapter_type: Adapter used for generation
            tokens_generated: Number of tokens generated
        """
        self.stats[adapter_type].record_generation(tokens_generated)

    def get_stats(self) -> dict[str, Any]:
        """Get all adapter statistics."""
        return {
            "active_adapter": self.active_adapter.name,
            "adapter_loaded": self._adapter_loaded,
            "adapters": {
                adapter_type.name: stats.to_dict() for adapter_type, stats in self.stats.items()
            },
        }

    def get_active_adapter(self) -> AdapterType:
        """Get the currently active adapter type."""
        return self.active_adapter

    async def warmup(self, model: Any) -> None:
        """
        Pre-warm all adapters by loading each once.

        This ensures adapter files are cached and ready for
        fast swapping during inference.

        Args:
            model: Base model for warmup
        """
        logger.info("Warming up adapters...")

        for adapter_type in AdapterType:
            if adapter_type != AdapterType.GENERAL:
                await self._load_adapter(model, adapter_type)
                await self._unload_adapter(model)

        # Reset to general
        self.active_adapter = AdapterType.GENERAL
        self._adapter_loaded = False

        logger.info("Adapter warmup complete")
