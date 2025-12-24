"""
Tool Progression Manager for AVA

Manages the unlocking of tools based on developmental stage,
milestones, and competencies.
"""

import logging
from datetime import datetime
from typing import Any

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolProgressionManager:
    """
    Manages tool progression and unlocking based on development.

    Works with the DevelopmentTracker to determine which tools
    AVA can access at any given point in development.
    """

    def __init__(self, registry: ToolRegistry):
        """
        Initialize the progression manager.

        Args:
            registry: The tool registry to manage
        """
        self.registry = registry
        self.unlock_history: list[dict[str, Any]] = []

    def get_accessible_tools(
        self,
        stage: int,
        milestones: list[str] | None = None,
        competencies: dict[str, float] | None = None,
        emotional_bias: dict[str, float] | None = None,
    ) -> list[str]:
        """
        Get list of tool names accessible at current development level.

        Args:
            stage: Current developmental stage value
            milestones: Achieved milestones
            competencies: Current competency scores
            emotional_bias: Emotional biases affecting tool preference

        Returns:
            List of accessible tool names
        """
        accessible = []
        milestones = milestones or []
        competencies = competencies or {}

        for tool_name, tool in self.registry.tools.items():
            # Check safety level
            if tool.safety_level > stage:
                continue

            # Check milestones
            if tool.required_milestones:
                if not all(m in milestones for m in tool.required_milestones):
                    continue

            # Check competencies
            if tool.minimum_competencies:
                meets_competencies = all(
                    competencies.get(comp, 0) >= min_level
                    for comp, min_level in tool.minimum_competencies.items()
                )
                if not meets_competencies:
                    continue

            accessible.append(tool_name)

        # Sort by emotional preference if bias provided
        if emotional_bias:
            accessible = self._sort_by_emotional_preference(
                accessible, emotional_bias
            )

        return accessible

    def _sort_by_emotional_preference(
        self,
        tool_names: list[str],
        emotional_bias: dict[str, float],
    ) -> list[str]:
        """Sort tools by emotional preference."""
        safe_pref = emotional_bias.get("safe_preference", 0.5)
        complexity_pref = emotional_bias.get("complexity_preference", 0.5)

        def preference_score(name: str) -> float:
            tool = self.registry.get_tool(name)
            if not tool:
                return 0

            # Lower safety level = "safer"
            safety_score = (5 - tool.safety_level) / 5 * safe_pref

            # Higher safety level = "more complex"
            complexity_score = tool.safety_level / 5 * complexity_pref

            return safety_score + complexity_score

        return sorted(tool_names, key=preference_score, reverse=True)

    def get_next_unlockable_tools(
        self,
        current_stage: int,
        milestones: list[str],
        competencies: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Get tools that could be unlocked next with progress.

        Args:
            current_stage: Current developmental stage
            milestones: Achieved milestones
            competencies: Current competencies

        Returns:
            List of tools with unlock requirements
        """
        unlockable = []

        for tool_name, tool in self.registry.tools.items():
            # Skip already accessible tools
            access = self.registry.check_access(
                tool_name, current_stage, milestones, competencies
            )
            if access.allowed:
                continue

            # Check what's missing
            missing = {
                "tool_name": tool_name,
                "description": tool.description,
                "safety_level": tool.safety_level,
                "requirements": [],
                "progress": 0.0,
            }

            requirements_met = 0
            total_requirements = 0

            # Stage requirement
            if current_stage < tool.safety_level:
                stage_diff = tool.safety_level - current_stage
                missing["requirements"].append({
                    "type": "stage",
                    "needed": tool.safety_level,
                    "current": current_stage,
                    "gap": stage_diff,
                })
                total_requirements += 1
            else:
                requirements_met += 1
                total_requirements += 1

            # Milestone requirements
            for milestone in tool.required_milestones:
                total_requirements += 1
                if milestone in milestones:
                    requirements_met += 1
                else:
                    missing["requirements"].append({
                        "type": "milestone",
                        "needed": milestone,
                    })

            # Competency requirements
            for comp, min_level in tool.minimum_competencies.items():
                total_requirements += 1
                current = competencies.get(comp, 0)
                if current >= min_level:
                    requirements_met += 1
                else:
                    missing["requirements"].append({
                        "type": "competency",
                        "name": comp,
                        "needed": min_level,
                        "current": current,
                        "gap": min_level - current,
                    })

            # Calculate progress
            if total_requirements > 0:
                missing["progress"] = requirements_met / total_requirements

            # Only include tools that are close to being unlockable
            if missing["progress"] >= 0.5:
                unlockable.append(missing)

        # Sort by progress (closest to unlock first)
        unlockable.sort(key=lambda x: x["progress"], reverse=True)

        return unlockable

    def record_unlock(
        self,
        tool_name: str,
        stage: int,
        trigger: str = "stage_transition",
    ):
        """Record a tool unlock event."""
        self.unlock_history.append({
            "tool_name": tool_name,
            "stage": stage,
            "trigger": trigger,
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(f"Tool unlocked: {tool_name} at stage {stage}")

    def get_tool_mastery(
        self,
        safety_level: int,
        min_success_rate: float = 0.8,
    ) -> dict[str, Any]:
        """
        Check mastery of tools at a given safety level.

        Args:
            safety_level: Safety level to check
            min_success_rate: Minimum success rate for mastery

        Returns:
            Mastery status and statistics
        """
        tools_at_level = [
            t for t in self.registry.tools.values()
            if t.safety_level == safety_level
        ]

        if not tools_at_level:
            return {"mastered": True, "tools": [], "avg_success_rate": 1.0}

        mastery_info = {
            "total_tools": len(tools_at_level),
            "mastered_count": 0,
            "tools": [],
            "avg_success_rate": 0.0,
        }

        total_success_rate = 0
        tools_with_usage = 0

        for tool in tools_at_level:
            success_rate = tool.get_success_rate()

            tool_info = {
                "name": tool.name,
                "times_used": tool.times_used,
                "success_rate": success_rate,
                "mastered": False,
            }

            if tool.times_used > 0:
                tools_with_usage += 1
                total_success_rate += success_rate

                if success_rate >= min_success_rate:
                    tool_info["mastered"] = True
                    mastery_info["mastered_count"] += 1

            mastery_info["tools"].append(tool_info)

        if tools_with_usage > 0:
            mastery_info["avg_success_rate"] = total_success_rate / tools_with_usage

        mastery_info["mastered"] = (
            mastery_info["mastered_count"] == len(tools_at_level) and
            len(tools_at_level) > 0
        )

        return mastery_info

    def suggest_tool_for_task(
        self,
        task_description: str,
        accessible_tools: list[str],
        emotional_bias: dict[str, float] | None = None,
    ) -> str | None:
        """
        Suggest the best tool for a given task.

        Simple keyword matching - could be enhanced with embeddings.

        Args:
            task_description: Description of the task
            accessible_tools: List of accessible tool names
            emotional_bias: Emotional biases

        Returns:
            Suggested tool name or None
        """
        task_lower = task_description.lower()
        scores = []

        for tool_name in accessible_tools:
            tool = self.registry.get_tool(tool_name)
            if not tool:
                continue

            score = 0

            # Keyword matching
            description_lower = tool.description.lower()
            task_words = set(task_lower.split())
            desc_words = set(description_lower.split())

            overlap = len(task_words & desc_words)
            score += overlap * 10

            # Boost based on name match
            if tool_name.lower() in task_lower:
                score += 50

            # Apply success rate boost
            score += tool.get_success_rate() * 10

            # Apply emotional bias
            if emotional_bias:
                safe_pref = emotional_bias.get("safe_preference", 0.5)
                if safe_pref > 0.6:
                    score += (5 - tool.safety_level) * 5  # Prefer safer
                else:
                    score += tool.safety_level * 2  # Accept complex

            scores.append((score, tool_name))

        if scores:
            scores.sort(reverse=True)
            return scores[0][1]

        return None

    def get_progression_summary(
        self,
        current_stage: int,
        milestones: list[str],
        competencies: dict[str, float],
    ) -> dict[str, Any]:
        """Get a summary of tool progression status."""
        accessible = self.get_accessible_tools(current_stage, milestones, competencies)
        total_tools = len(self.registry.tools)

        # Check mastery at each level up to current
        mastery_by_level = {}
        for level in range(current_stage + 1):
            mastery_by_level[level] = self.get_tool_mastery(level)

        return {
            "total_tools": total_tools,
            "accessible_count": len(accessible),
            "accessible_tools": accessible,
            "accessibility_percentage": len(accessible) / total_tools * 100 if total_tools > 0 else 0,
            "mastery_by_level": mastery_by_level,
            "next_unlockable": self.get_next_unlockable_tools(
                current_stage, milestones, competencies
            )[:3],
            "unlock_history_count": len(self.unlock_history),
        }
