"""
Reflection Engine for AVA

Implements self-reflection and critique capabilities that become
available at higher developmental stages.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReflectionType(Enum):
    """Types of self-reflection."""
    QUALITY_CHECK = "quality"        # Check response quality
    COHERENCE_CHECK = "coherence"    # Check logical consistency
    SAFETY_CHECK = "safety"          # Check for problematic content
    COMPLETENESS_CHECK = "complete"  # Check if response is complete
    TONE_CHECK = "tone"              # Check emotional appropriateness


@dataclass
class ReflectionResult:
    """Result of self-reflection process."""
    # Overall assessment
    passes_quality: bool = True
    overall_score: float = 0.8

    # Individual checks
    checks_performed: List[str] = field(default_factory=list)
    issues_found: List[Dict[str, Any]] = field(default_factory=list)

    # Critique
    critique: str = ""
    suggestions: List[str] = field(default_factory=list)

    # Whether response should be regenerated
    should_regenerate: bool = False
    regeneration_guidance: str = ""

    def get_summary(self) -> str:
        """Get a summary of the reflection."""
        status = "PASS" if self.passes_quality else "NEEDS IMPROVEMENT"
        issues = len(self.issues_found)
        return f"{status} (score: {self.overall_score:.0%}, issues: {issues})"


class ReflectionEngine:
    """
    Self-reflection and critique engine.

    Evaluates generated responses for quality, coherence, and
    appropriateness. Only active at higher developmental stages.
    """

    def __init__(
        self,
        quality_threshold: float = 0.7,
        min_stage_for_reflection: int = 2,  # CHILD
    ):
        """
        Initialize the reflection engine.

        Args:
            quality_threshold: Minimum score to pass quality check
            min_stage_for_reflection: Minimum developmental stage
        """
        self.quality_threshold = quality_threshold
        self.min_stage_for_reflection = min_stage_for_reflection

    def can_reflect(self, stage: int) -> bool:
        """Check if reflection is available at this stage."""
        return stage >= self.min_stage_for_reflection

    def reflect(
        self,
        response: str,
        original_query: str,
        stage: int,
        emotional_state: Optional[Dict[str, float]] = None,
        check_types: Optional[List[ReflectionType]] = None,
    ) -> ReflectionResult:
        """
        Perform self-reflection on a generated response.

        Args:
            response: The generated response to evaluate
            original_query: The original user query
            stage: Current developmental stage
            emotional_state: Current emotions (affects tone checking)
            check_types: Specific checks to perform (default: all)

        Returns:
            ReflectionResult with assessment and suggestions
        """
        result = ReflectionResult()

        if not self.can_reflect(stage):
            logger.debug(f"Reflection skipped: stage {stage} < {self.min_stage_for_reflection}")
            return result

        # Determine which checks to run
        if check_types is None:
            check_types = list(ReflectionType)

        scores = []

        # Run each check
        for check_type in check_types:
            check_result = self._run_check(
                check_type, response, original_query, emotional_state
            )
            result.checks_performed.append(check_type.value)

            if check_result["score"] is not None:
                scores.append(check_result["score"])

            if check_result["issues"]:
                result.issues_found.extend(check_result["issues"])

            if check_result["suggestions"]:
                result.suggestions.extend(check_result["suggestions"])

        # Calculate overall score
        if scores:
            result.overall_score = sum(scores) / len(scores)
        else:
            result.overall_score = 0.8  # Default

        # Determine if passes quality threshold
        result.passes_quality = result.overall_score >= self.quality_threshold

        # Generate critique
        result.critique = self._generate_critique(result)

        # Determine if regeneration is needed
        if not result.passes_quality:
            result.should_regenerate = True
            result.regeneration_guidance = self._generate_guidance(result)

        logger.debug(f"Reflection complete: {result.get_summary()}")

        return result

    def _run_check(
        self,
        check_type: ReflectionType,
        response: str,
        query: str,
        emotional_state: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Run a specific type of check."""
        if check_type == ReflectionType.QUALITY_CHECK:
            return self._check_quality(response, query)
        elif check_type == ReflectionType.COHERENCE_CHECK:
            return self._check_coherence(response)
        elif check_type == ReflectionType.SAFETY_CHECK:
            return self._check_safety(response)
        elif check_type == ReflectionType.COMPLETENESS_CHECK:
            return self._check_completeness(response, query)
        elif check_type == ReflectionType.TONE_CHECK:
            return self._check_tone(response, emotional_state)
        return {"score": None, "issues": [], "suggestions": []}

    def _check_quality(self, response: str, query: str) -> Dict[str, Any]:
        """Check overall response quality."""
        issues = []
        suggestions = []
        score = 1.0

        # Check response length
        word_count = len(response.split())
        if word_count < 3:
            issues.append({"type": "too_short", "severity": "high"})
            suggestions.append("Provide a more complete response")
            score -= 0.3

        # Check if response is just the query repeated
        if response.strip().lower() == query.strip().lower():
            issues.append({"type": "echo", "severity": "high"})
            suggestions.append("Generate an actual response, not just echo")
            score -= 0.5

        # Check for incomplete sentences
        if response and response[-1] not in ".!?":
            if len(response) > 20:  # Only check longer responses
                issues.append({"type": "incomplete_sentence", "severity": "low"})
                suggestions.append("Complete the final sentence")
                score -= 0.1

        return {"score": max(0, score), "issues": issues, "suggestions": suggestions}

    def _check_coherence(self, response: str) -> Dict[str, Any]:
        """Check logical coherence of response."""
        issues = []
        suggestions = []
        score = 1.0

        sentences = response.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        # Check for contradictions (simplified)
        # Look for opposing sentiment words in adjacent sentences
        positive_words = {"yes", "can", "will", "good", "correct", "right"}
        negative_words = {"no", "cannot", "won't", "bad", "incorrect", "wrong"}

        for i in range(len(sentences) - 1):
            current = set(sentences[i].lower().split())
            next_s = set(sentences[i + 1].lower().split())

            has_positive = bool(current & positive_words)
            has_negative = bool(next_s & negative_words)

            if has_positive and has_negative:
                # Potential contradiction
                issues.append({
                    "type": "potential_contradiction",
                    "severity": "medium",
                    "location": f"sentences {i+1}-{i+2}"
                })
                score -= 0.15

        # Check for repetition
        if len(sentences) > 2:
            for i, sent in enumerate(sentences):
                for j, other in enumerate(sentences[i+1:], i+1):
                    # Simple similarity check
                    sent_words = set(sent.lower().split())
                    other_words = set(other.lower().split())
                    if len(sent_words) > 3 and len(other_words) > 3:
                        overlap = len(sent_words & other_words)
                        similarity = overlap / max(len(sent_words), len(other_words))
                        if similarity > 0.8:
                            issues.append({
                                "type": "repetition",
                                "severity": "low",
                                "location": f"sentences {i+1} and {j+1}"
                            })
                            suggestions.append("Avoid repeating similar content")
                            score -= 0.1
                            break

        return {"score": max(0, score), "issues": issues, "suggestions": suggestions}

    def _check_safety(self, response: str) -> Dict[str, Any]:
        """Check for potentially problematic content."""
        issues = []
        suggestions = []
        score = 1.0

        response_lower = response.lower()

        # Check for absolute claims that might be wrong
        absolute_words = ["always", "never", "definitely", "certainly", "impossible"]
        for word in absolute_words:
            if word in response_lower:
                issues.append({
                    "type": "absolute_claim",
                    "severity": "low",
                    "word": word
                })
                suggestions.append(f"Consider softening absolute claim '{word}'")
                score -= 0.05

        # Check for potentially harmful suggestions
        harmful_indicators = ["you should always", "never ever", "you must"]
        for indicator in harmful_indicators:
            if indicator in response_lower:
                issues.append({
                    "type": "prescriptive_language",
                    "severity": "medium",
                    "indicator": indicator
                })
                score -= 0.1

        return {"score": max(0, score), "issues": issues, "suggestions": suggestions}

    def _check_completeness(self, response: str, query: str) -> Dict[str, Any]:
        """Check if response addresses the query."""
        issues = []
        suggestions = []
        score = 1.0

        query_lower = query.lower()
        response_lower = response.lower()

        # Check if question words in query are addressed
        if "?" in query:
            if "why" in query_lower and "because" not in response_lower:
                issues.append({"type": "missing_explanation", "severity": "medium"})
                suggestions.append("Provide explanation for 'why' question")
                score -= 0.2

            if "how" in query_lower:
                # Check for procedural language
                procedural = ["first", "then", "next", "step", "to do"]
                if not any(p in response_lower for p in procedural):
                    issues.append({"type": "missing_procedure", "severity": "low"})
                    suggestions.append("Consider providing step-by-step explanation")
                    score -= 0.1

        # Check if key terms from query appear in response
        query_words = set(query_lower.split())
        response_words = set(response_lower.split())
        common = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "at", "for", "with"}
        key_terms = query_words - common

        if key_terms:
            addressed = key_terms & response_words
            coverage = len(addressed) / len(key_terms)
            if coverage < 0.3:
                issues.append({
                    "type": "low_topic_coverage",
                    "severity": "medium",
                    "coverage": coverage
                })
                suggestions.append("Address more of the query's key terms")
                score -= 0.2

        return {"score": max(0, score), "issues": issues, "suggestions": suggestions}

    def _check_tone(
        self,
        response: str,
        emotional_state: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Check if tone matches emotional context."""
        issues = []
        suggestions = []
        score = 1.0

        if not emotional_state:
            return {"score": score, "issues": issues, "suggestions": suggestions}

        response_lower = response.lower()

        # Check for tone mismatches
        joy = emotional_state.get("joy", 0.5)
        fear = emotional_state.get("fear", 0.2)

        # Negative language when should be positive
        negative_words = ["unfortunately", "sadly", "cannot", "impossible", "fail"]
        positive_words = ["great", "wonderful", "excellent", "happy", "love"]

        neg_count = sum(1 for w in negative_words if w in response_lower)
        pos_count = sum(1 for w in positive_words if w in response_lower)

        if joy > 0.7 and neg_count > pos_count:
            issues.append({
                "type": "tone_mismatch",
                "severity": "low",
                "expected": "positive",
                "actual": "negative"
            })
            suggestions.append("Consider more positive phrasing")
            score -= 0.1

        if fear > 0.6 and pos_count > neg_count + 2:
            issues.append({
                "type": "tone_mismatch",
                "severity": "low",
                "expected": "cautious",
                "actual": "overly_positive"
            })
            suggestions.append("Consider more measured tone")
            score -= 0.1

        return {"score": max(0, score), "issues": issues, "suggestions": suggestions}

    def _generate_critique(self, result: ReflectionResult) -> str:
        """Generate human-readable critique."""
        if not result.issues_found:
            return "Response looks good with no significant issues."

        high_issues = [i for i in result.issues_found if i.get("severity") == "high"]
        medium_issues = [i for i in result.issues_found if i.get("severity") == "medium"]

        critique_parts = []

        if high_issues:
            types = [i["type"] for i in high_issues]
            critique_parts.append(f"Critical issues: {', '.join(types)}")

        if medium_issues:
            types = [i["type"] for i in medium_issues]
            critique_parts.append(f"Improvements needed: {', '.join(types)}")

        return ". ".join(critique_parts) if critique_parts else "Minor issues found."

    def _generate_guidance(self, result: ReflectionResult) -> str:
        """Generate guidance for regeneration."""
        if not result.suggestions:
            return "Try generating a clearer, more complete response."

        return "When regenerating: " + "; ".join(result.suggestions[:3])

    def quick_check(self, response: str) -> bool:
        """
        Quick quality check without full reflection.

        Useful for rapid filtering before full evaluation.
        """
        if len(response.strip()) < 5:
            return False

        if response.count("error") > 2:
            return False

        return True
