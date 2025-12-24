"""
THE CONSCIOUS STREAM
====================

The real-time processing pipeline that integrates:
- Titans Neural Memory (test-time learning via surprise)
- Semantic Memory (vector database retrieval)
- Toolformer-style reasoning (tool call parsing and execution)
- Fast Weight accumulation (session-level learning)

This module is the "awake" mind - it processes every interaction,
learns in real-time via surprise-driven weight updates, and records
high-quality interactions for later consolidation.

Architecture:
    Input → Embed → Titans Memory → Context Construction → LLM → Tools → Response
                ↓                                                    ↓
         Surprise Update                               Quality Assessment
                                                              ↓
                                                     Replay Buffer

Reference: "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Current state of the conscious stream."""

    IDLE = "idle"
    PERCEIVING = "perceiving"  # Processing input
    REMEMBERING = "remembering"  # Retrieving from memory
    THINKING = "thinking"  # LLM inference
    ACTING = "acting"  # Tool execution
    REFLECTING = "reflecting"  # Post-interaction assessment


@dataclass
class StreamConfig:
    """Configuration for the Conscious Stream."""

    # Model dimensions (must match your Ollama model)
    model_dim: int = 4096  # Llama-3-8B: 4096, Llama-3.2-3B: 3072

    # Titans Memory settings
    memory_dim: int = 8192
    memory_learning_rate: float = 0.01
    surprise_threshold: float = 0.3  # Lower = more updates

    # Context construction
    max_neural_context_tokens: int = 512
    max_semantic_context_tokens: int = 1024
    max_episodic_context_items: int = 5

    # Quality thresholds for learning
    quality_threshold_for_replay: float = 0.6
    surprise_boost_factor: float = 1.5  # Multiply quality by this if surprised

    # Device settings
    device: str = "cuda"  # or "cpu"
    use_torch: bool = True

    # Timeouts
    inference_timeout: float = 30.0
    tool_execution_timeout: float = 10.0


@dataclass
class InteractionRecord:
    """Record of a single interaction for the replay buffer."""

    # Core interaction
    input_text: str
    output_text: str

    # Context used
    neural_context: str | None = None
    semantic_context: str | None = None
    episodic_context: list[str] | None = None

    # Metrics
    surprise_value: float = 0.0
    quality_score: float = 0.0
    inference_time: float = 0.0

    # Tool usage
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    # Chain-of-thought
    reasoning_steps: list[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    topic_context: str = ""
    emotional_state: dict[str, float] | None = None

    def combined_quality(self, surprise_boost: float = 1.5) -> float:
        """
        Calculate combined quality score with surprise boost.

        High surprise + reasonable quality = valuable learning sample.
        """
        base_quality = self.quality_score

        if self.surprise_value > 0.5:
            # Surprising interactions are more valuable for learning
            base_quality *= surprise_boost

        return min(1.0, base_quality)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "neural_context": self.neural_context,
            "semantic_context": self.semantic_context,
            "episodic_context": self.episodic_context,
            "surprise_value": self.surprise_value,
            "quality_score": self.quality_score,
            "inference_time": self.inference_time,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "reasoning_steps": self.reasoning_steps,
            "timestamp": self.timestamp.isoformat(),
            "topic_context": self.topic_context,
            "emotional_state": self.emotional_state,
        }


class ConsciousStream:
    """
    The Conscious Stream - Real-time cognitive processing.

    This is the "awake" mind that processes every interaction:
    1. Perceive: Embed input, query neural memory
    2. Remember: Retrieve from semantic/episodic memory
    3. Think: Construct prompt, run LLM inference
    4. Act: Execute tool calls if needed
    5. Reflect: Assess quality, record for learning

    The key innovation is integrating Titans Memory into the
    inference loop - every interaction updates the neural memory
    based on surprise, enabling test-time learning.
    """

    def __init__(
        self,
        config: StreamConfig | None = None,
        llm_interface: Any | None = None,
        tool_registry: Any | None = None,
        weight_manager: Any | None = None,
    ):
        """
        Initialize the Conscious Stream.

        Args:
            config: Stream configuration
            llm_interface: Interface to Ollama for inference
            tool_registry: Registry of available tools
            weight_manager: FastSlowWeightManager for recording interactions
        """
        self.config = config or StreamConfig()
        self.llm = llm_interface
        self.tool_registry = tool_registry
        self.weight_manager = weight_manager

        # Current state
        self.state = StreamState.IDLE
        self.current_interaction: InteractionRecord | None = None

        # Initialize Neural Memory (Titans)
        self.neural_memory = self._initialize_neural_memory()

        # Replay buffer for consolidation
        self.replay_buffer: list[InteractionRecord] = []
        self.max_replay_buffer_size = 1000

        # Session statistics
        self.session_stats = {
            "total_interactions": 0,
            "total_surprises": 0,
            "avg_surprise": 0.0,
            "avg_quality": 0.0,
            "tool_calls": 0,
            "memory_updates": 0,
        }

        logger.info(f"ConsciousStream initialized with config: {self.config}")

    def _initialize_neural_memory(self):
        """Initialize Titans Neural Memory."""
        try:
            from ..hippocampus.titans import create_titans_memory

            return create_titans_memory(
                input_dim=self.config.model_dim,
                memory_dim=self.config.memory_dim,
                learning_rate=self.config.memory_learning_rate,
                surprise_threshold=self.config.surprise_threshold,
                use_torch=self.config.use_torch,
                device=self.config.device,
            )
        except ImportError as e:
            logger.warning(f"Could not import Titans memory: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize neural memory: {e}")
            return None

    async def process(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Process a user input through the conscious stream.

        This is the main entry point for interaction processing.

        Args:
            user_input: The user's input text
            context: Optional additional context

        Returns:
            AVA's response
        """
        start_time = time.time()

        # Initialize interaction record
        self.current_interaction = InteractionRecord(
            input_text=user_input,
            output_text="",
            topic_context=context.get("topic", "") if context else "",
        )

        try:
            # ============================================
            # STEP A: PERCEPTION - Embed and compute surprise
            # ============================================
            self.state = StreamState.PERCEIVING

            input_embedding = await self._get_embedding(user_input)
            surprise_metrics = await self._update_neural_memory(input_embedding)

            self.current_interaction.surprise_value = surprise_metrics.get("surprise_value", 0.0)

            # ============================================
            # STEP B: REMEMBERING - Retrieve relevant context
            # ============================================
            self.state = StreamState.REMEMBERING

            neural_context = await self._retrieve_neural_context(input_embedding)
            semantic_context = await self._retrieve_semantic_context(user_input)
            episodic_context = await self._retrieve_episodic_context(user_input)

            self.current_interaction.neural_context = neural_context
            self.current_interaction.semantic_context = semantic_context
            self.current_interaction.episodic_context = episodic_context

            # ============================================
            # STEP C: THINKING - Construct prompt and infer
            # ============================================
            self.state = StreamState.THINKING

            prompt = self._construct_prompt(
                user_input=user_input,
                surprise_value=self.current_interaction.surprise_value,
                neural_context=neural_context,
                semantic_context=semantic_context,
                episodic_context=episodic_context,
                additional_context=context,
            )

            response, reasoning_steps = await self._thinking_loop(prompt)

            self.current_interaction.output_text = response
            self.current_interaction.reasoning_steps = reasoning_steps

            # ============================================
            # STEP D: REFLECTION - Assess and record
            # ============================================
            self.state = StreamState.REFLECTING

            quality_score = self._evaluate_quality(response)
            self.current_interaction.quality_score = quality_score
            self.current_interaction.inference_time = time.time() - start_time

            # Record to replay buffer if quality is sufficient
            await self._record_interaction()

            # Update session statistics
            self._update_statistics()

            return response

        except Exception as e:
            logger.error(f"Error in conscious stream processing: {e}")
            self.state = StreamState.IDLE
            raise
        finally:
            self.state = StreamState.IDLE
            self.session_stats["total_interactions"] += 1

    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text from LLM.

        This is critical - the embedding is how we interface
        with the Titans Neural Memory.
        """
        if not self.llm:
            # Return random embedding for testing
            return np.random.randn(self.config.model_dim).astype(np.float32)

        try:
            # Use Ollama's embedding endpoint
            embedding = await self.llm.get_embedding(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return np.random.randn(self.config.model_dim).astype(np.float32)

    async def _update_neural_memory(
        self,
        embedding: np.ndarray,
    ) -> dict[str, Any]:
        """
        Update neural memory with surprise-driven learning.

        This is the core Titans mechanism: compute surprise,
        update memory weights if surprising.
        """
        if not self.neural_memory:
            return {"surprise_value": 0.0, "memory_updated": False}

        try:
            # The neural memory's update() method handles:
            # 1. Computing surprise (reconstruction error)
            # 2. Updating weights if surprise > threshold
            # 3. Updating momentum buffer
            # 4. Computing forget gate
            metrics = self.neural_memory.update(embedding)

            if hasattr(metrics, "to_dict"):
                metrics_dict = metrics.to_dict()
            elif isinstance(metrics, dict):
                metrics_dict = metrics
            else:
                metrics_dict = {
                    "surprise_value": float(metrics.surprise_value),
                    "memory_updated": bool(metrics.memory_updated),
                }

            if metrics_dict.get("memory_updated", False):
                self.session_stats["memory_updates"] += 1

            return metrics_dict

        except Exception as e:
            logger.error(f"Neural memory update failed: {e}")
            return {"surprise_value": 0.0, "memory_updated": False}

    async def _retrieve_neural_context(
        self,
        embedding: np.ndarray,
    ) -> str | None:
        """
        Retrieve context from neural memory.

        The neural memory returns an embedding that encodes
        "intuition" about what's relevant - we need to decode
        this into something useful for the prompt.
        """
        if not self.neural_memory:
            return None

        try:
            # Retrieve from neural memory
            self.neural_memory.retrieve(embedding)

            # For now, we can't directly decode the neural output to text.
            # Instead, we use it as a relevance signal or pass statistics.
            # In a full implementation, you'd have a projection head.

            stats = self.neural_memory.get_statistics()

            return f"[Neural Memory Active: Surprise Level={stats.get('avg_recent_surprise', 0):.3f}, Updates={stats.get('total_updates', 0)}]"

        except Exception as e:
            logger.warning(f"Neural context retrieval failed: {e}")
            return None

    async def _retrieve_semantic_context(self, query: str) -> str | None:
        """
        Retrieve context from semantic memory (vector database).

        This is the "factual" memory - stored knowledge chunks
        that are relevant to the query.
        """
        # TODO: Integrate with semantic memory manager
        # For now, return placeholder
        return None

    async def _retrieve_episodic_context(
        self,
        query: str,
    ) -> list[str] | None:
        """
        Retrieve recent relevant interactions from episodic memory.

        This provides conversational context and learning from
        past interactions.
        """
        # TODO: Integrate with episodic memory manager
        return None

    def _construct_prompt(
        self,
        user_input: str,
        surprise_value: float,
        neural_context: str | None,
        semantic_context: str | None,
        episodic_context: list[str] | None,
        additional_context: dict[str, Any] | None,
    ) -> str:
        """
        Construct the prompt for LLM inference.

        This is where we inject all the context - neural memory,
        semantic facts, episodic history, and current state.
        """
        prompt_parts = []

        # System header
        prompt_parts.append(
            """[SYSTEM]
You are AVA, an advanced AI assistant with neural memory capabilities.
You can learn in real-time and adapt to new information.
Think step by step before responding. Use tools when helpful."""
        )

        # Neural memory context
        if neural_context:
            prompt_parts.append(f"\n[NEURAL MEMORY]\n{neural_context}")

        # Surprise indicator
        if surprise_value > 0.5:
            prompt_parts.append(
                f"\n[ATTENTION] High surprise detected ({surprise_value:.3f}). "
                "This appears to be novel information - consider carefully."
            )

        # Semantic context
        if semantic_context:
            prompt_parts.append(f"\n[KNOWLEDGE]\n{semantic_context}")

        # Episodic context
        if episodic_context:
            recent_history = "\n".join(episodic_context[-3:])
            prompt_parts.append(f"\n[RECENT CONTEXT]\n{recent_history}")

        # Additional context
        if additional_context:
            if additional_context.get("emotional_state"):
                emotions = additional_context["emotional_state"]
                prompt_parts.append(f"\n[EMOTIONAL STATE]\n{emotions}")

            if additional_context.get("developmental_stage"):
                stage = additional_context["developmental_stage"]
                prompt_parts.append(f"\n[DEVELOPMENTAL STAGE]\n{stage}")

        # User input
        prompt_parts.append(f"\n[USER]\n{user_input}")

        # Response instruction
        prompt_parts.append("\n[ASSISTANT]\nLet me think about this...")

        return "\n".join(prompt_parts)

    async def _thinking_loop(
        self,
        prompt: str,
    ) -> tuple[str, list[str]]:
        """
        The thinking loop - inference with optional tool use.

        Implements a simplified ReAct pattern:
        1. Generate thought/action
        2. If tool call, execute and observe
        3. Continue until final answer
        """
        reasoning_steps = []
        final_response = ""
        max_iterations = 5

        current_prompt = prompt

        for iteration in range(max_iterations):
            # Generate response
            response = await self._generate_response(current_prompt)

            if not response:
                break

            reasoning_steps.append(f"Step {iteration + 1}: {response[:200]}...")

            # Check for tool calls
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                self.state = StreamState.ACTING

                # Execute tools
                tool_results = await self._execute_tools(tool_calls)

                self.current_interaction.tool_calls.extend(tool_calls)
                self.current_interaction.tool_results.extend(tool_results)

                # Add observations to prompt
                observations = self._format_tool_observations(tool_results)
                current_prompt += f"\n\n[TOOL OBSERVATIONS]\n{observations}\n\n[CONTINUE]"

                self.state = StreamState.THINKING
            else:
                # No tool calls - this is the final response
                final_response = response
                break

        if not final_response:
            final_response = (
                response if response else "I apologize, I couldn't generate a response."
            )

        return final_response, reasoning_steps

    async def _generate_response(self, prompt: str) -> str | None:
        """Generate a response from the LLM."""
        if not self.llm:
            return "LLM interface not available."

        try:
            response = await asyncio.wait_for(
                self.llm.generate(prompt),
                timeout=self.config.inference_timeout,
            )
            return response
        except asyncio.TimeoutError:
            logger.warning("LLM inference timed out")
            return None
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """
        Parse tool calls from response.

        Uses Toolformer-style format: [ToolName:args]
        """
        import re

        tool_pattern = r"\[(\w+):([^\]]+)\]"
        matches = re.findall(tool_pattern, response)

        calls = []
        for tool_name, args in matches:
            calls.append(
                {
                    "name": tool_name,
                    "args": args.strip(),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return calls

    async def _execute_tools(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute tool calls and collect results."""
        results = []

        for call in tool_calls:
            try:
                if self.tool_registry:
                    result = await asyncio.wait_for(
                        self.tool_registry.execute(
                            call["name"],
                            call["args"],
                        ),
                        timeout=self.config.tool_execution_timeout,
                    )
                    results.append(
                        {
                            "name": call["name"],
                            "output": str(result),
                            "success": True,
                        }
                    )
                else:
                    results.append(
                        {
                            "name": call["name"],
                            "output": "[Tool registry not available]",
                            "success": False,
                        }
                    )

                self.session_stats["tool_calls"] += 1

            except asyncio.TimeoutError:
                results.append(
                    {
                        "name": call["name"],
                        "output": "[Tool execution timed out]",
                        "success": False,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "name": call["name"],
                        "output": f"[Error: {str(e)}]",
                        "success": False,
                    }
                )

        return results

    def _format_tool_observations(
        self,
        results: list[dict[str, Any]],
    ) -> str:
        """Format tool results as observations."""
        observations = []
        for result in results:
            status = "✓" if result.get("success") else "✗"
            observations.append(f"{status} {result['name']}: {result['output']}")
        return "\n".join(observations)

    def _evaluate_quality(self, response: str) -> float:
        """
        Evaluate the quality of a response.

        This is a heuristic assessment - in production, you might
        use a reward model or other more sophisticated approach.
        """
        score = 0.5  # Base score

        # Length bonus (prefer substantive responses)
        if len(response) > 100:
            score += 0.1
        if len(response) > 300:
            score += 0.1

        # Structure bonus (has reasoning indicators)
        reasoning_markers = ["because", "therefore", "since", "step", "first", "then"]
        for marker in reasoning_markers:
            if marker.lower() in response.lower():
                score += 0.05

        # Penalty for very short or generic responses
        if len(response) < 20:
            score -= 0.3
        if response.lower().startswith("i don't know"):
            score -= 0.2

        # Penalty for apologies without substance
        if "sorry" in response.lower() and len(response) < 100:
            score -= 0.1

        return max(0.0, min(1.0, score))

    async def _record_interaction(self):
        """
        Record the current interaction for later consolidation.

        Only records if quality meets threshold (boosted by surprise).
        """
        if not self.current_interaction:
            return

        combined_quality = self.current_interaction.combined_quality(
            surprise_boost=self.config.surprise_boost_factor
        )

        if combined_quality >= self.config.quality_threshold_for_replay:
            # Add to replay buffer
            self.replay_buffer.append(self.current_interaction)

            # Trim buffer if too large
            if len(self.replay_buffer) > self.max_replay_buffer_size:
                # Keep highest quality samples
                self.replay_buffer.sort(
                    key=lambda x: x.combined_quality(),
                    reverse=True,
                )
                self.replay_buffer = self.replay_buffer[: self.max_replay_buffer_size // 2]

            # Also record to weight manager if available
            if self.weight_manager:
                try:
                    self.weight_manager.record_interaction(
                        input_text=self.current_interaction.input_text,
                        output_text=self.current_interaction.output_text,
                        quality_score=combined_quality,
                        topic_context=self.current_interaction.topic_context,
                    )
                except Exception as e:
                    logger.warning(f"Failed to record to weight manager: {e}")

    def _update_statistics(self):
        """Update session statistics."""
        if not self.current_interaction:
            return

        n = self.session_stats["total_interactions"]

        # Running average of surprise
        old_avg_surprise = self.session_stats["avg_surprise"]
        new_surprise = self.current_interaction.surprise_value
        self.session_stats["avg_surprise"] = (old_avg_surprise * n + new_surprise) / (n + 1)

        # Running average of quality
        old_avg_quality = self.session_stats["avg_quality"]
        new_quality = self.current_interaction.quality_score
        self.session_stats["avg_quality"] = (old_avg_quality * n + new_quality) / (n + 1)

        if new_surprise > 0.5:
            self.session_stats["total_surprises"] += 1

    def get_replay_buffer(self) -> list[InteractionRecord]:
        """Get the current replay buffer for consolidation."""
        return self.replay_buffer

    def clear_replay_buffer(self):
        """Clear the replay buffer after consolidation."""
        self.replay_buffer = []

    def get_statistics(self) -> dict[str, Any]:
        """Get session statistics."""
        memory_stats = {}
        if self.neural_memory:
            memory_stats = self.neural_memory.get_statistics()

        return {
            **self.session_stats,
            "replay_buffer_size": len(self.replay_buffer),
            "current_state": self.state.value,
            "neural_memory": memory_stats,
        }

    def save_state(self, path: str):
        """Save stream state for persistence."""
        import json
        from pathlib import Path

        state = {
            "session_stats": self.session_stats,
            "replay_buffer": [r.to_dict() for r in self.replay_buffer],
        }

        # Save neural memory separately if available
        if self.neural_memory:
            memory_state = self.neural_memory.get_state()
            state["neural_memory_state"] = memory_state

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        """Load stream state from persistence."""
        import json
        from pathlib import Path

        if not Path(path).exists():
            return

        with open(path) as f:
            state = json.load(f)

        self.session_stats = state.get("session_stats", self.session_stats)

        # Restore neural memory
        if self.neural_memory and "neural_memory_state" in state:
            self.neural_memory.load_state(state["neural_memory_state"])
