"""
Thinking Engine for AVA

Implements test-time compute through extended reasoning before
generating responses. Thinking depth scales with developmental stage.

Also implements Toolformer-style tool call detection and augmentation.
Reference: "Toolformer: Language Models Teach Themselves to Use Tools" (2023)
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

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
    thinking_steps: list[str] = field(default_factory=list)
    strategy_used: ThinkingStrategy = ThinkingStrategy.DIRECT

    # Derived insights
    identified_intent: str = ""
    key_concepts: list[str] = field(default_factory=list)
    required_knowledge: list[str] = field(default_factory=list)

    # Tool considerations
    tools_considered: list[str] = field(default_factory=list)
    selected_tool: str | None = None
    tool_reasoning: str = ""

    # Confidence and uncertainty
    confidence: float = 0.5
    uncertainty_areas: list[str] = field(default_factory=list)

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
        available_tools: list[str] | None = None,
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
        available_tools: list[str],
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

    def _extract_concepts(self, query: str) -> list[str]:
        """Extract key concepts from a query."""
        # Simple extraction: nouns and significant words
        words = query.split()
        # Filter out common words
        common = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in", "on", "at", "for", "with", "it", "this", "that", "i", "you", "he", "she", "they", "we", "my", "your"}
        concepts = [w.strip(".,?!") for w in words if w.lower() not in common and len(w) > 2]
        return concepts[:5]

    def _identify_knowledge_needs(self, query: str, context: str) -> list[str]:
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

    def _identify_uncertainties(self, query: str, context: str) -> list[str]:
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


# =============================================================================
# TOOLFORMER-STYLE TOOL CALL DETECTION AND AUGMENTATION
# =============================================================================
# Reference: "Toolformer: Language Models Teach Themselves to Use Tools" (2023)
#
# Key concepts:
# - Detect when tool calls would be beneficial
# - Parse tool calls from model output: [ToolName: args]
# - Execute tools and inject results
# - Score augmentation quality for self-distillation
# =============================================================================


@dataclass
class ToolCall:
    """A parsed tool call from model output."""
    tool_name: str
    arguments: str
    position: int  # Position in text where call was found
    raw_text: str  # Original [Tool: args] text
    result: str | None = None
    executed: bool = False
    success: bool = False
    error_message: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "position": self.position,
            "raw_text": self.raw_text,
            "result": self.result,
            "executed": self.executed,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class ToolAugmentationResult:
    """Result of augmenting text with tool calls."""
    original_text: str
    augmented_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    augmentation_quality: float = 0.0
    perplexity_reduction: float = 0.0  # Estimated benefit
    should_distill: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "augmented_text": self.augmented_text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "augmentation_quality": self.augmentation_quality,
            "perplexity_reduction": self.perplexity_reduction,
            "should_distill": self.should_distill,
        }


class ToolCallParser:
    """
    Parses tool calls from model output.

    Recognizes patterns like:
    - [Calculator: 2+2]
    - [Search: quantum computing]
    - [QA: What is the capital of France?]
    """

    # Pattern to match [ToolName: arguments] or [ToolName(arguments)]
    TOOL_PATTERN = re.compile(
        r'\[(\w+)(?::\s*|\()(.*?)(?:\)|\])',
        re.DOTALL
    )

    # Alternative pattern: [ToolName] = result
    RESULT_PATTERN = re.compile(
        r'\[(\w+):\s*(.*?)\]\s*(?:=|→|->)\s*(.+?)(?=\[|\n|$)',
        re.DOTALL
    )

    def __init__(self, available_tools: list[str] | None = None):
        """
        Initialize parser.

        Args:
            available_tools: List of valid tool names
        """
        self.available_tools = set(available_tools or [])

    def parse(self, text: str) -> list[ToolCall]:
        """
        Parse tool calls from text.

        Args:
            text: Text potentially containing tool calls

        Returns:
            List of parsed ToolCall objects
        """
        calls = []

        for match in self.TOOL_PATTERN.finditer(text):
            tool_name = match.group(1)
            arguments = match.group(2).strip()

            # Validate tool name if we have a list
            if self.available_tools and tool_name.lower() not in {t.lower() for t in self.available_tools}:
                continue

            call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                position=match.start(),
                raw_text=match.group(0),
            )
            calls.append(call)

        return calls

    def has_tool_calls(self, text: str) -> bool:
        """Check if text contains any tool calls."""
        return bool(self.TOOL_PATTERN.search(text))

    def strip_tool_calls(self, text: str) -> str:
        """Remove all tool calls from text."""
        return self.TOOL_PATTERN.sub('', text).strip()

    def inject_results(
        self,
        text: str,
        tool_calls: list[ToolCall],
    ) -> str:
        """
        Inject tool results into text.

        Replaces [Tool: args] with [Tool: args] = result

        Args:
            text: Original text with tool calls
            tool_calls: Tool calls with results

        Returns:
            Augmented text with results
        """
        # Sort by position descending to avoid offset issues
        sorted_calls = sorted(tool_calls, key=lambda x: x.position, reverse=True)

        result_text = text
        for call in sorted_calls:
            if call.executed and call.result is not None:
                replacement = f"{call.raw_text} = {call.result}"
                result_text = (
                    result_text[:call.position] +
                    replacement +
                    result_text[call.position + len(call.raw_text):]
                )

        return result_text


class ToolCallDetector:
    """
    Detects when tool calls would be beneficial.

    Uses heuristics to identify queries that would benefit
    from tool augmentation without requiring the model to
    explicitly request tools.
    """

    # Patterns that suggest tool use
    TOOL_INDICATORS = {
        "calculator": [
            r'\d+\s*[\+\-\*\/\%\^]\s*\d+',  # Math expressions
            r'calculate|compute|sum|multiply|divide|subtract|add',
            r'how much is|what is \d+',
            r'equals|=',
        ],
        "search": [
            r'who is|what is|where is|when was',
            r'tell me about|information about|facts about',
            r'define|meaning of|definition',
            r'current|latest|today|recent',
        ],
        "qa": [
            r'what does .+ mean',
            r'explain .+ to me',
            r'how does .+ work',
        ],
        "current_time": [
            r'what time|current time|time is it',
            r'what day|today|date',
        ],
        "temperature_convert": [
            r'celsius|fahrenheit|kelvin',
            r'convert .+ degrees',
            r'temperature',
        ],
        "word_count": [
            r'how many words|word count|count words',
            r'how many characters|character count',
        ],
    }

    def __init__(self, available_tools: list[str] | None = None):
        """
        Initialize detector.

        Args:
            available_tools: List of available tool names
        """
        self.available_tools = set(available_tools or [])

        # Compile patterns
        self.compiled_patterns: dict[str, list[re.Pattern]] = {}
        for tool, patterns in self.TOOL_INDICATORS.items():
            self.compiled_patterns[tool] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def detect_tool_opportunities(
        self,
        text: str,
    ) -> list[tuple[str, str, float]]:
        """
        Detect opportunities to use tools.

        Args:
            text: Text to analyze

        Returns:
            List of (tool_name, matched_text, confidence) tuples
        """
        opportunities = []

        for tool, patterns in self.compiled_patterns.items():
            # Skip if tool not available
            if self.available_tools and tool not in self.available_tools:
                continue

            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    # Calculate confidence based on pattern specificity
                    confidence = 0.7 + (len(match.group(0)) / len(text)) * 0.3
                    opportunities.append((tool, match.group(0), min(confidence, 1.0)))
                    break  # One match per tool is enough

        return opportunities

    def suggest_tool_call(
        self,
        text: str,
    ) -> ToolCall | None:
        """
        Suggest a tool call for the given text.

        Args:
            text: Text to analyze

        Returns:
            Suggested ToolCall or None
        """
        opportunities = self.detect_tool_opportunities(text)

        if not opportunities:
            return None

        # Pick highest confidence
        best = max(opportunities, key=lambda x: x[2])
        tool_name, matched_text, confidence = best

        # Extract arguments based on tool type
        args = self._extract_arguments(tool_name, text, matched_text)

        return ToolCall(
            tool_name=tool_name,
            arguments=args,
            position=0,
            raw_text=f"[{tool_name}: {args}]",
        )

    def _extract_arguments(
        self,
        tool_name: str,
        text: str,
        matched_text: str,
    ) -> str:
        """Extract appropriate arguments for a tool call."""
        text_lower = text.lower()

        if tool_name == "calculator":
            # Extract math expression
            math_match = re.search(r'[\d\.\s\+\-\*\/\(\)\^]+', text)
            if math_match:
                return math_match.group(0).strip()
            return matched_text

        elif tool_name == "search":
            # Extract search query
            for prefix in ["what is", "who is", "tell me about", "define"]:
                if prefix in text_lower:
                    idx = text_lower.find(prefix) + len(prefix)
                    return text[idx:].strip().rstrip("?.!")
            return matched_text

        elif tool_name == "current_time":
            return ""  # No arguments needed

        elif tool_name == "temperature_convert":
            # Extract temperature and units
            temp_match = re.search(r'(\d+(?:\.\d+)?)\s*(celsius|fahrenheit|kelvin|c|f|k)', text_lower)
            if temp_match:
                return f"{temp_match.group(1)} {temp_match.group(2)}"
            return matched_text

        return matched_text


class ToolformerAugmenter:
    """
    Augments model output with tool call results.

    Implements the Toolformer self-distillation pattern:
    1. Detect potential tool calls in output
    2. Execute tools and inject results
    3. Score augmentation quality
    4. Mark high-quality augmentations for distillation
    """

    def __init__(
        self,
        tool_executor: Callable[[str, str], tuple[str, bool]] | None = None,
        quality_threshold: float = 0.6,
        available_tools: list[str] | None = None,
    ):
        """
        Initialize augmenter.

        Args:
            tool_executor: Function (tool_name, args) -> (result, success)
            quality_threshold: Minimum quality to mark for distillation
            available_tools: List of available tools
        """
        self.tool_executor = tool_executor
        self.quality_threshold = quality_threshold

        self.parser = ToolCallParser(available_tools)
        self.detector = ToolCallDetector(available_tools)

        # Statistics
        self.total_augmentations = 0
        self.successful_augmentations = 0
        self.distillation_candidates = 0

    def augment(
        self,
        text: str,
        auto_detect: bool = True,
    ) -> ToolAugmentationResult:
        """
        Augment text with tool call results.

        Args:
            text: Text to augment (may contain explicit tool calls)
            auto_detect: Also detect implicit tool opportunities

        Returns:
            ToolAugmentationResult with augmented text and metadata
        """
        import time

        result = ToolAugmentationResult(
            original_text=text,
            augmented_text=text,
        )

        # Parse explicit tool calls
        tool_calls = self.parser.parse(text)

        # Auto-detect opportunities if enabled
        if auto_detect and not tool_calls:
            suggested = self.detector.suggest_tool_call(text)
            if suggested:
                # Insert the suggested tool call
                tool_calls = [suggested]
                # Modify text to include the tool call marker
                result.augmented_text = f"{text}\n{suggested.raw_text}"

        # Execute tool calls
        if self.tool_executor:
            for call in tool_calls:
                start_time = time.time()

                try:
                    call.result, call.success = self.tool_executor(
                        call.tool_name,
                        call.arguments,
                    )
                    call.executed = True
                except Exception as e:
                    call.executed = True
                    call.success = False
                    call.error_message = str(e)
                    call.result = f"Error: {e}"

                call.execution_time_ms = (time.time() - start_time) * 1000

        result.tool_calls = tool_calls

        # Inject results into text
        if tool_calls:
            result.augmented_text = self.parser.inject_results(
                result.augmented_text,
                tool_calls,
            )

        # Calculate augmentation quality
        result.augmentation_quality = self._calculate_quality(result)
        result.perplexity_reduction = self._estimate_perplexity_reduction(result)
        result.should_distill = result.augmentation_quality >= self.quality_threshold

        # Update statistics
        self.total_augmentations += 1
        if any(tc.success for tc in tool_calls):
            self.successful_augmentations += 1
        if result.should_distill:
            self.distillation_candidates += 1

        return result

    def _calculate_quality(self, result: ToolAugmentationResult) -> float:
        """Calculate augmentation quality score."""
        if not result.tool_calls:
            return 0.0

        # Base quality on execution success
        success_rate = sum(1 for tc in result.tool_calls if tc.success) / len(result.tool_calls)

        # Bonus for result length (longer = more informative)
        avg_result_length = sum(
            len(tc.result or "") for tc in result.tool_calls
        ) / len(result.tool_calls)
        length_bonus = min(avg_result_length / 100, 0.2)

        # Penalty for errors
        error_rate = sum(1 for tc in result.tool_calls if tc.error_message) / len(result.tool_calls)
        error_penalty = error_rate * 0.3

        quality = success_rate + length_bonus - error_penalty
        return max(0.0, min(1.0, quality))

    def _estimate_perplexity_reduction(self, result: ToolAugmentationResult) -> float:
        """
        Estimate perplexity reduction from augmentation.

        In a full implementation, this would be measured by
        comparing model perplexity before/after augmentation.
        Here we use a heuristic based on quality.
        """
        if result.augmentation_quality < 0.3:
            return 0.0

        # Estimate: good augmentations reduce perplexity by 5-15%
        reduction = result.augmentation_quality * 0.15
        return reduction

    def get_statistics(self) -> dict[str, Any]:
        """Get augmentation statistics."""
        return {
            "total_augmentations": self.total_augmentations,
            "successful_augmentations": self.successful_augmentations,
            "success_rate": self.successful_augmentations / max(self.total_augmentations, 1),
            "distillation_candidates": self.distillation_candidates,
            "distillation_rate": self.distillation_candidates / max(self.total_augmentations, 1),
        }


@dataclass
class DistillationSample:
    """A sample marked for self-distillation."""
    input_text: str
    original_output: str
    augmented_output: str
    tool_calls: list[dict[str, Any]]
    quality_score: float
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_text": self.input_text,
            "original_output": self.original_output,
            "augmented_output": self.augmented_output,
            "tool_calls": self.tool_calls,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat(),
        }


class SelfDistillationCollector:
    """
    Collects high-quality tool-augmented samples for self-distillation.

    These samples can be used to train the model to:
    1. Recognize when to use tools
    2. Generate proper tool call syntax
    3. Integrate tool results into responses
    """

    def __init__(
        self,
        min_quality: float = 0.6,
        max_samples: int = 1000,
    ):
        """
        Initialize collector.

        Args:
            min_quality: Minimum quality to collect
            max_samples: Maximum samples to store
        """
        self.min_quality = min_quality
        self.max_samples = max_samples

        self.samples: list[DistillationSample] = []
        self.total_collected = 0
        self.total_rejected = 0

    def collect(
        self,
        input_text: str,
        augmentation_result: ToolAugmentationResult,
    ) -> bool:
        """
        Collect a sample if it meets quality threshold.

        Args:
            input_text: Original input/query
            augmentation_result: Result from ToolformerAugmenter

        Returns:
            True if sample was collected
        """
        if augmentation_result.augmentation_quality < self.min_quality:
            self.total_rejected += 1
            return False

        sample = DistillationSample(
            input_text=input_text,
            original_output=augmentation_result.original_text,
            augmented_output=augmentation_result.augmented_text,
            tool_calls=[tc.to_dict() for tc in augmentation_result.tool_calls],
            quality_score=augmentation_result.augmentation_quality,
        )

        self.samples.append(sample)
        self.total_collected += 1

        # Trim if over capacity
        if len(self.samples) > self.max_samples:
            # Remove lowest quality samples
            self.samples.sort(key=lambda x: x.quality_score, reverse=True)
            self.samples = self.samples[:self.max_samples]

        return True

    def get_training_batch(
        self,
        batch_size: int = 32,
    ) -> list[dict[str, str]]:
        """
        Get a batch of samples for training.

        Returns samples formatted as input/output pairs.
        """
        import random

        if not self.samples:
            return []

        batch_size = min(batch_size, len(self.samples))
        batch = random.sample(self.samples, batch_size)

        return [
            {
                "input": s.input_text,
                "output": s.augmented_output,
                "quality": s.quality_score,
            }
            for s in batch
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get collector statistics."""
        if not self.samples:
            return {
                "current_samples": 0,
                "total_collected": self.total_collected,
                "total_rejected": self.total_rejected,
            }

        qualities = [s.quality_score for s in self.samples]

        return {
            "current_samples": len(self.samples),
            "max_samples": self.max_samples,
            "total_collected": self.total_collected,
            "total_rejected": self.total_rejected,
            "acceptance_rate": self.total_collected / max(self.total_collected + self.total_rejected, 1),
            "avg_quality": sum(qualities) / len(qualities),
            "min_quality": min(qualities),
            "max_quality": max(qualities),
        }

    def export_samples(self) -> list[dict[str, Any]]:
        """Export all samples as dictionaries."""
        return [s.to_dict() for s in self.samples]


def create_toolformer_system(
    tool_executor: Callable[[str, str], tuple[str, bool]] | None = None,
    available_tools: list[str] | None = None,
    quality_threshold: float = 0.6,
    max_distillation_samples: int = 1000,
) -> tuple[ToolformerAugmenter, SelfDistillationCollector]:
    """
    Factory function to create Toolformer components.

    Args:
        tool_executor: Function to execute tools
        available_tools: List of available tool names
        quality_threshold: Quality threshold for distillation
        max_distillation_samples: Max samples to store

    Returns:
        Tuple of (ToolformerAugmenter, SelfDistillationCollector)
    """
    augmenter = ToolformerAugmenter(
        tool_executor=tool_executor,
        quality_threshold=quality_threshold,
        available_tools=available_tools,
    )

    collector = SelfDistillationCollector(
        min_quality=quality_threshold,
        max_samples=max_distillation_samples,
    )

    return augmenter, collector
