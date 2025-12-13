"""
Thinking Engine for AVA

Implements test-time compute through extended reasoning before
generating responses. Thinking depth scales with developmental stage.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ThinkingStrategy(Enum):
    """Different strategies for extended thinking."""
    DIRECT = "direct"              # Minimal thinking, direct response
    CHAIN_OF_THOUGHT = "cot"       # Step-by-step reasoning
    DECOMPOSITION = "decompose"    # Break into sub-problems
    ANALOGY = "analogy"            # Reason by analogy
    DELIBERATION = "deliberate"    # Consider multiple options
    REFLECTION = "reflect"         # Think about thinking


@dataclass
class ThinkingResult:
    """Result of extended thinking process."""
    # The thinking process
    thinking_steps: List[str] = field(default_factory=list)
    strategy_used: ThinkingStrategy = ThinkingStrategy.DIRECT

    # Derived insights
    identified_intent: str = ""
    key_concepts: List[str] = field(default_factory=list)
    required_knowledge: List[str] = field(default_factory=list)

    # Tool considerations
    tools_considered: List[str] = field(default_factory=list)
    selected_tool: Optional[str] = None
    tool_reasoning: str = ""

    # Confidence and uncertainty
    confidence: float = 0.5
    uncertainty_areas: List[str] = field(default_factory=list)

    # Novelty detection
    encountered_novelty: bool = False
    novelty_description: str = ""

    # Metadata
    thinking_tokens_used: int = 0
    thinking_time_ms: float = 0.0

    def get_thinking_summary(self) -> str:
        """Get a summary of the thinking process."""
        if not self.thinking_steps:
            return "No extended thinking performed."

        summary_parts = [f"Strategy: {self.strategy_used.value}"]

        if self.identified_intent:
            summary_parts.append(f"Intent: {self.identified_intent}")

        if self.key_concepts:
            summary_parts.append(f"Key concepts: {', '.join(self.key_concepts)}")

        if self.selected_tool:
            summary_parts.append(f"Tool: {self.selected_tool}")

        summary_parts.append(f"Confidence: {self.confidence:.0%}")

        return " | ".join(summary_parts)


class ThinkingEngine:
    """
    Extended reasoning engine for test-time compute.

    Allocates thinking budget based on developmental stage and
    emotional state, then executes appropriate thinking strategies.
    """

    def __init__(
        self,
        base_thinking_budget: int = 512,
        max_thinking_budget: int = 2048,
    ):
        """
        Initialize the thinking engine.

        Args:
            base_thinking_budget: Base number of thinking tokens
            max_thinking_budget: Maximum thinking tokens allowed
        """
        self.base_thinking_budget = base_thinking_budget
        self.max_thinking_budget = max_thinking_budget

    def allocate_budget(
        self,
        query_complexity: float,
        stage_multiplier: float,
        emotional_multiplier: float = 1.0,
    ) -> int:
        """
        Allocate thinking token budget.

        Args:
            query_complexity: Estimated complexity (0.0 to 1.0)
            stage_multiplier: Developmental stage multiplier
            emotional_multiplier: Emotional state multiplier

        Returns:
            Number of thinking tokens allocated
        """
        # Base budget scaled by complexity
        budget = self.base_thinking_budget * (0.5 + query_complexity)

        # Apply stage multiplier
        budget *= stage_multiplier

        # Apply emotional multiplier
        budget *= emotional_multiplier

        # Clamp to valid range
        return int(min(self.max_thinking_budget, max(64, budget)))

    def estimate_complexity(self, query: str) -> float:
        """
        Estimate query complexity for budget allocation.

        Simple heuristics - could be enhanced with ML.
        """
        complexity = 0.3  # Base complexity

        # Length-based
        word_count = len(query.split())
        if word_count > 50:
            complexity += 0.2
        elif word_count > 20:
            complexity += 0.1

        # Question words suggest reasoning needed
        question_words = ["why", "how", "explain", "compare", "analyze", "evaluate"]
        query_lower = query.lower()
        for word in question_words:
            if word in query_lower:
                complexity += 0.1

        # Multiple questions
        if query.count("?") > 1:
            complexity += 0.15

        # Technical/complex indicators
        complex_indicators = ["algorithm", "implement", "design", "optimize", "debug"]
        for indicator in complex_indicators:
            if indicator in query_lower:
                complexity += 0.1

        return min(1.0, complexity)

    def select_strategy(
        self,
        query: str,
        stage: int,
        complexity: float,
    ) -> ThinkingStrategy:
        """
        Select appropriate thinking strategy based on context.

        Args:
            query: The input query
            stage: Developmental stage
            complexity: Estimated complexity

        Returns:
            Selected thinking strategy
        """
        query_lower = query.lower()

        # Early stages use simpler strategies
        if stage < 2:  # INFANT or TODDLER
            return ThinkingStrategy.DIRECT

        # Match strategy to query type
        if any(word in query_lower for word in ["step", "how to", "process"]):
            return ThinkingStrategy.CHAIN_OF_THOUGHT

        if any(word in query_lower for word in ["compare", "versus", "difference"]):
            return ThinkingStrategy.DELIBERATION

        if any(word in query_lower for word in ["like", "similar", "analogy"]):
            return ThinkingStrategy.ANALOGY

        if complexity > 0.7:
            return ThinkingStrategy.DECOMPOSITION

        if stage >= 4:  # YOUNG_ADULT or MATURE
            if any(word in query_lower for word in ["think", "consider", "reflect"]):
                return ThinkingStrategy.REFLECTION

        # Default based on complexity
        if complexity > 0.5:
            return ThinkingStrategy.CHAIN_OF_THOUGHT
        return ThinkingStrategy.DIRECT

    def think(
        self,
        query: str,
        context: str,
        stage: int,
        thinking_budget: int,
        available_tools: Optional[List[str]] = None,
    ) -> ThinkingResult:
        """
        Execute extended thinking on a query.

        Args:
            query: The user's query
            context: Relevant context (memories, conversation history)
            stage: Current developmental stage
            thinking_budget: Allocated thinking tokens
            available_tools: List of available tool names

        Returns:
            ThinkingResult with reasoning and insights
        """
        import time
        start_time = time.time()

        result = ThinkingResult()
        available_tools = available_tools or []

        # Estimate complexity and select strategy
        complexity = self.estimate_complexity(query)
        strategy = self.select_strategy(query, stage, complexity)
        result.strategy_used = strategy

        # Execute strategy-specific thinking
        if strategy == ThinkingStrategy.DIRECT:
            result = self._think_direct(query, result)
        elif strategy == ThinkingStrategy.CHAIN_OF_THOUGHT:
            result = self._think_chain_of_thought(query, context, result)
        elif strategy == ThinkingStrategy.DECOMPOSITION:
            result = self._think_decompose(query, context, result)
        elif strategy == ThinkingStrategy.DELIBERATION:
            result = self._think_deliberate(query, context, result)
        elif strategy == ThinkingStrategy.ANALOGY:
            result = self._think_analogy(query, context, result)
        elif strategy == ThinkingStrategy.REFLECTION:
            result = self._think_reflect(query, context, result)

        # Consider tools
        if available_tools:
            result = self._consider_tools(query, available_tools, result)

        # Detect novelty
        result = self._detect_novelty(query, context, result)

        # Calculate confidence based on analysis
        result.confidence = self._calculate_confidence(result, complexity)

        # Record timing
        result.thinking_time_ms = (time.time() - start_time) * 1000
        result.thinking_tokens_used = len(" ".join(result.thinking_steps).split())

        logger.debug(f"Thinking complete: {result.get_thinking_summary()}")

        return result

    def _think_direct(self, query: str, result: ThinkingResult) -> ThinkingResult:
        """Minimal direct thinking."""
        result.thinking_steps.append(f"Direct response to: {query[:50]}...")
        result.identified_intent = self._extract_intent(query)
        return result

    def _think_chain_of_thought(
        self,
        query: str,
        context: str,
        result: ThinkingResult
    ) -> ThinkingResult:
        """Step-by-step reasoning."""
        result.thinking_steps.append("Step 1: Understanding the question")
        result.identified_intent = self._extract_intent(query)

        result.thinking_steps.append("Step 2: Identifying key concepts")
        result.key_concepts = self._extract_concepts(query)

        result.thinking_steps.append("Step 3: Considering relevant knowledge")
        result.required_knowledge = self._identify_knowledge_needs(query, context)

        result.thinking_steps.append("Step 4: Formulating approach")
        if result.required_knowledge:
            result.thinking_steps.append(
                f"Need to address: {', '.join(result.required_knowledge[:3])}"
            )

        return result

    def _think_decompose(
        self,
        query: str,
        context: str,
        result: ThinkingResult
    ) -> ThinkingResult:
        """Break complex query into sub-problems."""
        result.thinking_steps.append("Decomposing complex query into sub-problems")

        # Simple decomposition by sentence/clause
        parts = query.replace("?", "?.").replace(".", "..").split(".")
        parts = [p.strip() for p in parts if p.strip()]

        for i, part in enumerate(parts[:4], 1):
            result.thinking_steps.append(f"Sub-problem {i}: {part}")

        result.identified_intent = self._extract_intent(query)
        result.key_concepts = self._extract_concepts(query)

        return result

    def _think_deliberate(
        self,
        query: str,
        context: str,
        result: ThinkingResult
    ) -> ThinkingResult:
        """Consider multiple perspectives/options."""
        result.thinking_steps.append("Deliberating on multiple approaches")

        result.thinking_steps.append("Option 1: Direct factual response")
        result.thinking_steps.append("Option 2: Explanatory response with context")
        result.thinking_steps.append("Option 3: Comparative analysis")

        result.thinking_steps.append("Evaluating options based on query needs")
        result.identified_intent = self._extract_intent(query)

        return result

    def _think_analogy(
        self,
        query: str,
        context: str,
        result: ThinkingResult
    ) -> ThinkingResult:
        """Reason by analogy."""
        result.thinking_steps.append("Seeking analogies and similar patterns")

        result.identified_intent = self._extract_intent(query)
        result.key_concepts = self._extract_concepts(query)

        result.thinking_steps.append(
            f"Looking for patterns similar to: {', '.join(result.key_concepts[:3])}"
        )

        return result

    def _think_reflect(
        self,
        query: str,
        context: str,
        result: ThinkingResult
    ) -> ThinkingResult:
        """Meta-cognitive reflection."""
        result.thinking_steps.append("Reflecting on how to approach this")
        result.thinking_steps.append("Considering my knowledge and limitations")

        result.identified_intent = self._extract_intent(query)

        result.thinking_steps.append("Assessing confidence in my ability to respond")
        result.uncertainty_areas = self._identify_uncertainties(query, context)

        if result.uncertainty_areas:
            result.thinking_steps.append(
                f"Areas of uncertainty: {', '.join(result.uncertainty_areas[:3])}"
            )

        return result

    def _consider_tools(
        self,
        query: str,
        available_tools: List[str],
        result: ThinkingResult
    ) -> ThinkingResult:
        """Consider whether tools are needed."""
        query_lower = query.lower()

        # Simple keyword matching for tool needs
        tool_keywords = {
            "calculate": ["calculator", "simple_math"],
            "time": ["current_time"],
            "math": ["calculator", "simple_math"],
            "count": ["word_count"],
            "random": ["random_number"],
            "temperature": ["temperature_convert"],
            "define": ["dictionary"],
            "meaning": ["dictionary"],
        }

        for keyword, tools in tool_keywords.items():
            if keyword in query_lower:
                for tool in tools:
                    if tool in available_tools:
                        result.tools_considered.append(tool)

        if result.tools_considered:
            result.selected_tool = result.tools_considered[0]
            result.tool_reasoning = f"Query contains '{keyword}' suggesting {result.selected_tool}"
            result.thinking_steps.append(f"Tool consideration: {result.tool_reasoning}")

        return result

    def _detect_novelty(
        self,
        query: str,
        context: str,
        result: ThinkingResult
    ) -> ThinkingResult:
        """Detect if query contains novel concepts."""
        # Simple heuristic: if many unique words not in context
        query_words = set(query.lower().split())
        context_words = set(context.lower().split()) if context else set()

        novel_words = query_words - context_words
        novelty_ratio = len(novel_words) / len(query_words) if query_words else 0

        if novelty_ratio > 0.5:
            result.encountered_novelty = True
            result.novelty_description = f"High novelty ({novelty_ratio:.0%} new concepts)"

        return result

    def _calculate_confidence(
        self,
        result: ThinkingResult,
        complexity: float
    ) -> float:
        """Calculate overall confidence in the thinking result."""
        confidence = 0.7  # Base confidence

        # Reduce for complexity
        confidence -= complexity * 0.2

        # Reduce for uncertainty areas
        confidence -= len(result.uncertainty_areas) * 0.1

        # Increase if tool selected
        if result.selected_tool:
            confidence += 0.1

        # Reduce for novelty
        if result.encountered_novelty:
            confidence -= 0.1

        return max(0.1, min(1.0, confidence))

    def _extract_intent(self, query: str) -> str:
        """Extract the primary intent from a query."""
        query_lower = query.lower()

        if "?" in query:
            if any(w in query_lower for w in ["what", "which"]):
                return "information_request"
            elif any(w in query_lower for w in ["how", "explain"]):
                return "explanation_request"
            elif any(w in query_lower for w in ["why"]):
                return "reasoning_request"
            elif any(w in query_lower for w in ["can you", "could you"]):
                return "action_request"
            return "question"

        if any(w in query_lower for w in ["please", "can you", "could you"]):
            return "action_request"

        return "statement"

    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from a query."""
        # Simple extraction: nouns and significant words
        words = query.split()
        # Filter out common words
        common = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in", "on", "at", "for", "with", "it", "this", "that", "i", "you", "he", "she", "they", "we", "my", "your"}
        concepts = [w.strip(".,?!") for w in words if w.lower() not in common and len(w) > 2]
        return concepts[:5]

    def _identify_knowledge_needs(self, query: str, context: str) -> List[str]:
        """Identify what knowledge is needed to answer."""
        needs = []
        query_lower = query.lower()

        if "why" in query_lower:
            needs.append("causal reasoning")
        if "how" in query_lower:
            needs.append("procedural knowledge")
        if any(w in query_lower for w in ["best", "better", "recommend"]):
            needs.append("evaluative judgment")
        if any(w in query_lower for w in ["example", "instance"]):
            needs.append("concrete examples")

        return needs

    def _identify_uncertainties(self, query: str, context: str) -> List[str]:
        """Identify areas of uncertainty."""
        uncertainties = []
        query_lower = query.lower()

        if any(w in query_lower for w in ["future", "predict", "will"]):
            uncertainties.append("prediction uncertainty")
        if any(w in query_lower for w in ["opinion", "think", "believe"]):
            uncertainties.append("subjective judgment")
        if any(w in query_lower for w in ["always", "never", "all"]):
            uncertainties.append("absolute claims")

        return uncertainties
