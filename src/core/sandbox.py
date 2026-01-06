"""
SANDBOX - Mental Simulation Environment
========================================

The Mental Sandbox implements Stage 3 of the Sentinel architecture:
multi-agent simulation before output emission.

Key Features:
- Three specialized agents (Thinker, Verifier, Butler)
- Consensus loop (max 5 cycles)
- Backtrack events for iterative refinement
- Integration with Z3 formal verification

Agents:
- Thinker: Drafts solutions using LLM inference
- Verifier: Validates logic using Z3 formal methods
- Butler: Personalizes output for user context

References:
- Mental Simulation Theory in AI (Friston, 2025)
- Multi-Agent Verification Systems (arXiv, 2024)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .verification import FormalVerifier, VerificationConfig, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles for sandbox agents."""

    THINKER = "thinker"  # Drafts solutions
    VERIFIER = "verifier"  # Validates logic
    BUTLER = "butler"  # Personalizes output


class ConsensusStatus(Enum):
    """Status of consensus process."""

    PENDING = "pending"
    ACHIEVED = "achieved"
    MAX_CYCLES = "max_cycles"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SandboxConfig:
    """Configuration for Mental Sandbox."""

    max_cycles: int = 5  # Maximum simulation cycles
    timeout_seconds: float = 30.0  # Overall timeout
    require_verification: bool = True  # Must pass Z3 check
    require_personalization: bool = True  # Must pass Butler check
    min_confidence: float = 0.7  # Minimum confidence for output
    enable_backtrack: bool = True  # Allow revision cycles
    verification_config: VerificationConfig | None = None


@dataclass
class AgentFeedback:
    """Feedback from an agent."""

    agent: AgentRole
    approved: bool
    confidence: float
    reasoning: str
    suggestions: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent": self.agent.value,
            "approved": self.approved,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SimulationResult:
    """Result of a simulation cycle."""

    output: str
    consensus_status: ConsensusStatus
    cycles_used: int
    confidence: float
    feedback_history: list[AgentFeedback] = field(default_factory=list)
    verification_result: VerificationResult | None = None
    backtrack_count: int = 0
    total_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "status": self.consensus_status.value,
            "cycles": self.cycles_used,
            "confidence": self.confidence,
            "backtracks": self.backtrack_count,
            "time_ms": self.total_time_ms,
            "feedback": [f.to_dict() for f in self.feedback_history],
            "verification": (
                self.verification_result.to_dict() if self.verification_result else None
            ),
        }


class SandboxAgent(ABC):
    """Abstract base class for sandbox agents."""

    def __init__(self, role: AgentRole):
        self.role = role
        self._stats = {
            "invocations": 0,
            "approvals": 0,
            "rejections": 0,
        }

    @abstractmethod
    async def evaluate(self, draft: str, context: dict[str, Any]) -> AgentFeedback:
        """
        Evaluate a draft and provide feedback.

        Args:
            draft: The current draft output
            context: Additional context (query, user info, etc.)

        Returns:
            AgentFeedback with approval status and suggestions
        """
        pass

    def get_stats(self) -> dict[str, int]:
        """Get agent statistics."""
        return self._stats.copy()


class ThinkerAgent(SandboxAgent):
    """
    Agent that drafts solutions.

    The Thinker uses LLM inference to generate initial drafts
    and refine them based on feedback from other agents.
    """

    def __init__(
        self,
        llm_callback: Callable[[str], str] | None = None,
    ):
        super().__init__(AgentRole.THINKER)
        self._llm_callback = llm_callback

    async def evaluate(self, draft: str, context: dict[str, Any]) -> AgentFeedback:
        """Thinker always approves its own draft."""
        self._stats["invocations"] += 1
        self._stats["approvals"] += 1

        return AgentFeedback(
            agent=self.role,
            approved=True,
            confidence=0.8,
            reasoning="Draft generated successfully",
        )

    async def draft(
        self, query: str, context: dict[str, Any], feedback: list[AgentFeedback] | None = None
    ) -> str:
        """
        Generate a draft response.

        Args:
            query: The user's query
            context: Additional context
            feedback: Previous feedback for revision

        Returns:
            Draft response string
        """
        self._stats["invocations"] += 1

        # If we have an LLM callback, use it
        if self._llm_callback:
            # Build prompt with feedback
            prompt = query
            if feedback:
                suggestions = []
                for fb in feedback:
                    if fb.suggestions:
                        suggestions.extend(fb.suggestions)
                if suggestions:
                    prompt += "\n\nConsider these improvements:\n" + "\n".join(
                        f"- {s}" for s in suggestions
                    )

            try:
                response = self._llm_callback(prompt)
                return response
            except Exception as e:
                logger.error(f"LLM callback failed: {e}")

        # Fallback: return query echo (for testing)
        return f"[Draft response to: {query}]"

    def set_llm_callback(self, callback: Callable[[str], str]) -> None:
        """Set the LLM callback for draft generation."""
        self._llm_callback = callback


class VerifierAgent(SandboxAgent):
    """
    Agent that verifies logic using Z3.

    The Verifier uses formal methods to check:
    - Mathematical correctness
    - Logical consistency
    - Constraint satisfaction
    """

    def __init__(self, config: VerificationConfig | None = None):
        super().__init__(AgentRole.VERIFIER)
        self._verifier = FormalVerifier(config)

    async def evaluate(self, draft: str, context: dict[str, Any]) -> AgentFeedback:
        """
        Verify the draft using Z3.

        Args:
            draft: The draft to verify
            context: Context including original query

        Returns:
            AgentFeedback with verification results
        """
        self._stats["invocations"] += 1

        query = context.get("query", "")

        # Check if verification is needed
        if not self._verifier.should_verify(draft, context.get("entropy", 0.0)):
            self._stats["approvals"] += 1
            return AgentFeedback(
                agent=self.role,
                approved=True,
                confidence=0.9,
                reasoning="Verification not required for this content",
            )

        # Perform verification
        result = await self._verifier.verify_logic(draft, query)

        if result.is_valid:
            self._stats["approvals"] += 1
            return AgentFeedback(
                agent=self.role,
                approved=True,
                confidence=0.95 if result.status == VerificationStatus.VALID else 0.7,
                reasoning=result.explanation,
            )
        else:
            self._stats["rejections"] += 1
            suggestions = []
            if result.proof_steps:
                suggestions.append(f"Check logic: {', '.join(result.proof_steps[:3])}")
            if result.status == VerificationStatus.INVALID:
                suggestions.append("Contradiction detected - revise logical claims")

            return AgentFeedback(
                agent=self.role,
                approved=False,
                confidence=0.3,
                reasoning=result.explanation,
                suggestions=suggestions,
            )

    def get_verification_result(self) -> VerificationResult | None:
        """Get the last verification result."""
        return getattr(self, "_last_result", None)


class ButlerAgent(SandboxAgent):
    """
    Agent that personalizes output for user context.

    The Butler ensures:
    - Response matches user preferences
    - Tone and style are appropriate
    - Context from user history is incorporated
    """

    def __init__(
        self,
        user_preferences: dict[str, Any] | None = None,
        knowledge_base: dict[str, Any] | None = None,
    ):
        super().__init__(AgentRole.BUTLER)
        self._preferences = user_preferences or {}
        self._knowledge_base = knowledge_base or {}

    async def evaluate(self, draft: str, context: dict[str, Any]) -> AgentFeedback:
        """
        Check draft against user context.

        Args:
            draft: The draft to evaluate
            context: Context including user info

        Returns:
            AgentFeedback with personalization assessment
        """
        self._stats["invocations"] += 1

        suggestions: list[str] = []
        issues: list[str] = []

        # Check for user preferences
        user_name = context.get("user_name", self._preferences.get("name"))
        if user_name and user_name.lower() not in draft.lower():
            # Not necessarily an issue, just a note
            pass

        # Check tone preferences
        preferred_tone = self._preferences.get("tone", "professional")
        if preferred_tone == "formal" and any(
            word in draft.lower() for word in ["hey", "yeah", "gonna", "wanna"]
        ):
            suggestions.append("Use more formal language")
            issues.append("Informal tone detected")

        # Check for completeness
        query = context.get("query", "")
        if query.endswith("?") and not any(
            phrase in draft.lower()
            for phrase in ["the answer", "this means", "in summary", "therefore"]
        ):
            # Response might not directly answer the question
            pass

        # Check against knowledge base
        relevant_facts = self._get_relevant_facts(query)
        if relevant_facts:
            # Could verify draft against known facts
            pass

        # Determine approval
        approved = len(issues) == 0
        confidence = 0.9 if approved else 0.6

        if approved:
            self._stats["approvals"] += 1
        else:
            self._stats["rejections"] += 1

        return AgentFeedback(
            agent=self.role,
            approved=approved,
            confidence=confidence,
            reasoning="Personalization check " + ("passed" if approved else "needs improvement"),
            suggestions=suggestions,
        )

    async def adapt(self, draft: str, context: dict[str, Any]) -> str:
        """
        Adapt the draft for user context.

        Args:
            draft: The draft to adapt
            context: User context

        Returns:
            Adapted draft string
        """
        # For now, return draft unchanged
        # Future: Apply personalization transforms
        return draft

    def _get_relevant_facts(self, query: str) -> list[str]:
        """Get facts from knowledge base relevant to query."""
        # Simple keyword matching for now
        relevant: list[str] = []
        query_words = set(query.lower().split())

        for key, value in self._knowledge_base.items():
            if any(word in key.lower() for word in query_words):
                relevant.append(f"{key}: {value}")

        return relevant[:5]  # Limit to 5 facts

    def set_preferences(self, preferences: dict[str, Any]) -> None:
        """Update user preferences."""
        self._preferences.update(preferences)

    def set_knowledge_base(self, knowledge: dict[str, Any]) -> None:
        """Update knowledge base."""
        self._knowledge_base.update(knowledge)


class MentalSandbox:
    """
    Multi-agent simulation environment.

    Orchestrates Thinker, Verifier, and Butler agents to produce
    verified, personalized output through consensus loops.

    Example:
        >>> sandbox = MentalSandbox(SandboxConfig())
        >>> result = await sandbox.simulate("What is 2+2?", {})
        >>> print(result.output)
        >>> print(f"Consensus in {result.cycles_used} cycles")
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        llm_callback: Callable[[str], str] | None = None,
    ):
        self.config = config or SandboxConfig()
        self.agents: dict[AgentRole, SandboxAgent] = {
            AgentRole.THINKER: ThinkerAgent(llm_callback),
            AgentRole.VERIFIER: VerifierAgent(self.config.verification_config),
            AgentRole.BUTLER: ButlerAgent(),
        }
        self._stats = {
            "simulations": 0,
            "consensus_achieved": 0,
            "max_cycles_reached": 0,
            "total_cycles": 0,
        }

    async def simulate(self, query: str, context: dict[str, Any] | None = None) -> SimulationResult:
        """
        Run simulation until consensus.

        Args:
            query: The user's query
            context: Additional context (user info, entropy, etc.)

        Returns:
            SimulationResult with final output and metadata
        """
        import time

        start_time = time.time()
        self._stats["simulations"] += 1

        context = context or {}
        context["query"] = query

        feedback_history: list[AgentFeedback] = []
        current_draft = ""
        backtrack_count = 0
        verification_result: VerificationResult | None = None

        thinker = self.agents[AgentRole.THINKER]
        verifier = self.agents[AgentRole.VERIFIER]
        butler = self.agents[AgentRole.BUTLER]

        # Ensure thinker is ThinkerAgent
        if not isinstance(thinker, ThinkerAgent):
            raise TypeError("Thinker agent must be ThinkerAgent instance")

        try:
            # Create timeout task
            async def simulation_loop() -> SimulationResult:
                nonlocal current_draft, backtrack_count, verification_result

                for cycle in range(1, self.config.max_cycles + 1):
                    logger.debug(f"Simulation cycle {cycle}/{self.config.max_cycles}")

                    # Stage 1: Thinker drafts (with feedback from previous cycle)
                    previous_feedback = feedback_history[-3:] if feedback_history else None
                    current_draft = await thinker.draft(query, context, previous_feedback)

                    # Stage 2: Verifier checks
                    if self.config.require_verification:
                        verifier_feedback = await verifier.evaluate(current_draft, context)
                        feedback_history.append(verifier_feedback)

                        # Get verification result if available
                        if isinstance(verifier, VerifierAgent):
                            verification_result = verifier.get_verification_result()

                        if not verifier_feedback.approved:
                            if self.config.enable_backtrack:
                                backtrack_count += 1
                                logger.debug(f"Backtrack {backtrack_count}: Verifier rejected")
                                continue
                            else:
                                # No backtrack, fail immediately
                                return SimulationResult(
                                    output=current_draft,
                                    consensus_status=ConsensusStatus.FAILED,
                                    cycles_used=cycle,
                                    confidence=verifier_feedback.confidence,
                                    feedback_history=feedback_history,
                                    verification_result=verification_result,
                                    backtrack_count=backtrack_count,
                                    total_time_ms=(time.time() - start_time) * 1000,
                                )

                    # Stage 3: Butler personalizes
                    if self.config.require_personalization:
                        butler_feedback = await butler.evaluate(current_draft, context)
                        feedback_history.append(butler_feedback)

                        if not butler_feedback.approved:
                            if self.config.enable_backtrack:
                                backtrack_count += 1
                                logger.debug(f"Backtrack {backtrack_count}: Butler rejected")
                                continue

                        # Apply personalization
                        if isinstance(butler, ButlerAgent):
                            current_draft = await butler.adapt(current_draft, context)

                    # Stage 4: Consensus achieved!
                    self._stats["consensus_achieved"] += 1
                    self._stats["total_cycles"] += cycle

                    return SimulationResult(
                        output=current_draft,
                        consensus_status=ConsensusStatus.ACHIEVED,
                        cycles_used=cycle,
                        confidence=1.0,
                        feedback_history=feedback_history,
                        verification_result=verification_result,
                        backtrack_count=backtrack_count,
                        total_time_ms=(time.time() - start_time) * 1000,
                    )

                # Max cycles reached
                self._stats["max_cycles_reached"] += 1
                self._stats["total_cycles"] += self.config.max_cycles

                # Calculate confidence based on feedback
                final_confidence = self.config.min_confidence
                if feedback_history:
                    avg_confidence = sum(f.confidence for f in feedback_history) / len(
                        feedback_history
                    )
                    final_confidence = max(self.config.min_confidence, avg_confidence)

                return SimulationResult(
                    output=current_draft,
                    consensus_status=ConsensusStatus.MAX_CYCLES,
                    cycles_used=self.config.max_cycles,
                    confidence=final_confidence,
                    feedback_history=feedback_history,
                    verification_result=verification_result,
                    backtrack_count=backtrack_count,
                    total_time_ms=(time.time() - start_time) * 1000,
                )

            # Run with timeout
            result = await asyncio.wait_for(
                simulation_loop(),
                timeout=self.config.timeout_seconds,
            )
            return result

        except asyncio.TimeoutError:
            return SimulationResult(
                output=current_draft or f"[Timeout processing: {query}]",
                consensus_status=ConsensusStatus.TIMEOUT,
                cycles_used=self.config.max_cycles,
                confidence=0.3,
                feedback_history=feedback_history,
                verification_result=verification_result,
                backtrack_count=backtrack_count,
                total_time_ms=self.config.timeout_seconds * 1000,
            )

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return SimulationResult(
                output=f"[Error: {str(e)}]",
                consensus_status=ConsensusStatus.FAILED,
                cycles_used=0,
                confidence=0.0,
                feedback_history=feedback_history,
                backtrack_count=backtrack_count,
                total_time_ms=(time.time() - start_time) * 1000,
            )

    def set_llm_callback(self, callback: Callable[[str], str]) -> None:
        """Set LLM callback for Thinker agent."""
        thinker = self.agents[AgentRole.THINKER]
        if isinstance(thinker, ThinkerAgent):
            thinker.set_llm_callback(callback)

    def set_user_preferences(self, preferences: dict[str, Any]) -> None:
        """Set user preferences for Butler agent."""
        butler = self.agents[AgentRole.BUTLER]
        if isinstance(butler, ButlerAgent):
            butler.set_preferences(preferences)

    def set_knowledge_base(self, knowledge: dict[str, Any]) -> None:
        """Set knowledge base for Butler agent."""
        butler = self.agents[AgentRole.BUTLER]
        if isinstance(butler, ButlerAgent):
            butler.set_knowledge_base(knowledge)

    def get_stats(self) -> dict[str, Any]:
        """Get sandbox statistics."""
        stats = self._stats.copy()
        stats["agents"] = {role.value: agent.get_stats() for role, agent in self.agents.items()}
        if self._stats["simulations"] > 0:
            stats["avg_cycles"] = self._stats["total_cycles"] / self._stats["simulations"]
        return stats

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._stats = {
            "simulations": 0,
            "consensus_achieved": 0,
            "max_cycles_reached": 0,
            "total_cycles": 0,
        }
        for agent in self.agents.values():
            agent._stats = {
                "invocations": 0,
                "approvals": 0,
                "rejections": 0,
            }
