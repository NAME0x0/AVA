"""
Developmental Milestones for AVA

Defines competency milestones that AVA must achieve to progress
through developmental stages. Milestones are checked based on
behavioral patterns and demonstrated capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MilestoneCategory(Enum):
    """Categories of developmental milestones."""
    COMMUNICATION = "communication"
    REASONING = "reasoning"
    EMOTIONAL = "emotional"
    TOOL_USE = "tool_use"
    MEMORY = "memory"
    SOCIAL = "social"
    META_COGNITIVE = "meta_cognitive"


@dataclass
class Milestone:
    """
    A developmental milestone that AVA can achieve.

    Milestones are prerequisites for stage transitions and represent
    demonstrated competencies that show growth.
    """
    id: str
    name: str
    description: str
    category: MilestoneCategory

    # Stage this milestone is relevant for (unlocks next stage)
    target_stage: int

    # Criteria for achievement
    required_occurrences: int = 1  # How many times to demonstrate
    time_window_hours: Optional[float] = None  # Window for occurrences

    # Automatic check function name (if checkable programmatically)
    check_function: Optional[str] = None

    # Points/weight for maturation score
    weight: float = 1.0


@dataclass
class MilestoneProgress:
    """Tracks progress toward a milestone."""
    milestone_id: str
    occurrences: int = 0
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    achieved: bool = False
    achieved_at: Optional[datetime] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)


# Define all milestones
MILESTONES: Dict[str, Milestone] = {
    # INFANT -> TODDLER milestones
    "first_coherent_response": Milestone(
        id="first_coherent_response",
        name="First Coherent Response",
        description="Produce a grammatically correct, meaningful sentence",
        category=MilestoneCategory.COMMUNICATION,
        target_stage=0,
        required_occurrences=3,
        check_function="check_coherent_response",
    ),
    "first_tool_use": Milestone(
        id="first_tool_use",
        name="First Tool Use",
        description="Successfully execute any tool",
        category=MilestoneCategory.TOOL_USE,
        target_stage=0,
        required_occurrences=1,
        check_function="check_tool_success",
    ),
    "emotional_expression": Milestone(
        id="emotional_expression",
        name="Emotional Expression",
        description="Express emotions appropriately in responses",
        category=MilestoneCategory.EMOTIONAL,
        target_stage=0,
        required_occurrences=5,
        check_function="check_emotional_expression",
    ),

    # TODDLER -> CHILD milestones
    "multi_turn_memory": Milestone(
        id="multi_turn_memory",
        name="Multi-Turn Memory",
        description="Reference information from earlier in the conversation",
        category=MilestoneCategory.MEMORY,
        target_stage=1,
        required_occurrences=5,
        check_function="check_memory_reference",
    ),
    "basic_reasoning": Milestone(
        id="basic_reasoning",
        name="Basic Reasoning",
        description="Demonstrate simple chain-of-thought reasoning",
        category=MilestoneCategory.REASONING,
        target_stage=1,
        required_occurrences=3,
        check_function="check_basic_reasoning",
    ),
    "tool_mastery_level_1": Milestone(
        id="tool_mastery_level_1",
        name="Tool Mastery Level 1",
        description="Achieve 80%+ success rate on Level 1 tools",
        category=MilestoneCategory.TOOL_USE,
        target_stage=1,
        required_occurrences=10,  # 10 successful uses
        check_function="check_tool_mastery",
    ),
    "vocabulary_expansion": Milestone(
        id="vocabulary_expansion",
        name="Vocabulary Expansion",
        description="Use 500+ unique words across responses",
        category=MilestoneCategory.COMMUNICATION,
        target_stage=1,
        required_occurrences=1,
        check_function="check_vocabulary_size",
    ),

    # CHILD -> ADOLESCENT milestones
    "complex_reasoning": Milestone(
        id="complex_reasoning",
        name="Complex Reasoning",
        description="Solve multi-step problems with clear logic",
        category=MilestoneCategory.REASONING,
        target_stage=2,
        required_occurrences=5,
        check_function="check_complex_reasoning",
    ),
    "emotional_regulation": Milestone(
        id="emotional_regulation",
        name="Emotional Regulation",
        description="Maintain stable emotional responses under varied inputs",
        category=MilestoneCategory.EMOTIONAL,
        target_stage=2,
        required_occurrences=10,
        time_window_hours=24,
        check_function="check_emotional_stability",
    ),
    "knowledge_synthesis": Milestone(
        id="knowledge_synthesis",
        name="Knowledge Synthesis",
        description="Combine information from multiple memories to form insights",
        category=MilestoneCategory.MEMORY,
        target_stage=2,
        required_occurrences=3,
        check_function="check_knowledge_synthesis",
    ),
    "tool_mastery_level_2": Milestone(
        id="tool_mastery_level_2",
        name="Tool Mastery Level 2",
        description="Achieve 80%+ success rate on Level 2 tools",
        category=MilestoneCategory.TOOL_USE,
        target_stage=2,
        required_occurrences=15,
        check_function="check_tool_mastery",
    ),

    # ADOLESCENT -> YOUNG_ADULT milestones
    "self_reflection_demonstrated": Milestone(
        id="self_reflection_demonstrated",
        name="Self-Reflection Demonstrated",
        description="Show ability to critique and improve own responses",
        category=MilestoneCategory.META_COGNITIVE,
        target_stage=3,
        required_occurrences=5,
        check_function="check_self_reflection",
    ),
    "nuanced_understanding": Milestone(
        id="nuanced_understanding",
        name="Nuanced Understanding",
        description="Handle ambiguous or nuanced queries appropriately",
        category=MilestoneCategory.REASONING,
        target_stage=3,
        required_occurrences=5,
        check_function="check_nuanced_response",
    ),
    "independent_problem_solving": Milestone(
        id="independent_problem_solving",
        name="Independent Problem Solving",
        description="Solve novel problems without explicit guidance",
        category=MilestoneCategory.REASONING,
        target_stage=3,
        required_occurrences=3,
        check_function="check_independent_solving",
    ),
    "tool_mastery_level_3": Milestone(
        id="tool_mastery_level_3",
        name="Tool Mastery Level 3",
        description="Achieve 80%+ success rate on Level 3 tools",
        category=MilestoneCategory.TOOL_USE,
        target_stage=3,
        required_occurrences=20,
        check_function="check_tool_mastery",
    ),

    # YOUNG_ADULT -> MATURE milestones
    "wisdom_demonstration": Milestone(
        id="wisdom_demonstration",
        name="Wisdom Demonstration",
        description="Provide insightful, wise advice drawing on experience",
        category=MilestoneCategory.META_COGNITIVE,
        target_stage=4,
        required_occurrences=5,
        check_function="check_wisdom",
        weight=2.0,
    ),
    "emotional_intelligence": Milestone(
        id="emotional_intelligence",
        name="Emotional Intelligence",
        description="Demonstrate understanding of user's emotional state",
        category=MilestoneCategory.EMOTIONAL,
        target_stage=4,
        required_occurrences=10,
        check_function="check_emotional_intelligence",
    ),
    "meta_learning": Milestone(
        id="meta_learning",
        name="Meta-Learning",
        description="Show ability to learn how to learn better",
        category=MilestoneCategory.META_COGNITIVE,
        target_stage=4,
        required_occurrences=3,
        check_function="check_meta_learning",
        weight=2.0,
    ),
    "full_tool_proficiency": Milestone(
        id="full_tool_proficiency",
        name="Full Tool Proficiency",
        description="Master all available tools with 90%+ success rate",
        category=MilestoneCategory.TOOL_USE,
        target_stage=4,
        required_occurrences=1,
        check_function="check_full_tool_mastery",
    ),
}


class MilestoneChecker:
    """
    Checks and tracks milestone progress for AVA.

    Works with the DevelopmentTracker to evaluate whether milestones
    have been achieved based on interaction patterns and demonstrated
    competencies.
    """

    def __init__(self, data_path: str = "data/developmental"):
        self.data_path = data_path
        self.progress: Dict[str, MilestoneProgress] = {}
        self._load_progress()

    def _load_progress(self):
        """Load milestone progress from disk."""
        import json
        from pathlib import Path

        progress_file = Path(self.data_path) / "milestone_progress.json"
        if progress_file.exists():
            with open(progress_file, "r") as f:
                data = json.load(f)
                for mid, pdata in data.items():
                    self.progress[mid] = MilestoneProgress(
                        milestone_id=mid,
                        occurrences=pdata.get("occurrences", 0),
                        achieved=pdata.get("achieved", False),
                        achieved_at=datetime.fromisoformat(pdata["achieved_at"])
                        if pdata.get("achieved_at") else None,
                        evidence=pdata.get("evidence", []),
                    )
        else:
            # Initialize progress for all milestones
            for mid in MILESTONES:
                self.progress[mid] = MilestoneProgress(milestone_id=mid)

    def save_progress(self):
        """Save milestone progress to disk."""
        import json
        from pathlib import Path

        progress_file = Path(self.data_path) / "milestone_progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for mid, prog in self.progress.items():
            data[mid] = {
                "occurrences": prog.occurrences,
                "achieved": prog.achieved,
                "achieved_at": prog.achieved_at.isoformat() if prog.achieved_at else None,
                "evidence": prog.evidence[-10:],  # Keep last 10 evidence items
            }

        with open(progress_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_occurrence(
        self,
        milestone_id: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record an occurrence toward a milestone.

        Returns True if the milestone was just achieved.
        """
        if milestone_id not in MILESTONES:
            return False

        milestone = MILESTONES[milestone_id]
        progress = self.progress.get(milestone_id)

        if progress is None:
            progress = MilestoneProgress(milestone_id=milestone_id)
            self.progress[milestone_id] = progress

        if progress.achieved:
            return False  # Already achieved

        now = datetime.now()
        progress.occurrences += 1
        progress.last_occurrence = now

        if progress.first_occurrence is None:
            progress.first_occurrence = now

        if evidence:
            progress.evidence.append({
                "timestamp": now.isoformat(),
                **evidence
            })

        # Check if milestone is achieved
        if progress.occurrences >= milestone.required_occurrences:
            # Check time window if specified
            if milestone.time_window_hours:
                window_start = now.timestamp() - (milestone.time_window_hours * 3600)
                recent = sum(
                    1 for e in progress.evidence
                    if datetime.fromisoformat(e["timestamp"]).timestamp() > window_start
                )
                if recent >= milestone.required_occurrences:
                    progress.achieved = True
                    progress.achieved_at = now
            else:
                progress.achieved = True
                progress.achieved_at = now

        self.save_progress()
        return progress.achieved and progress.achieved_at == now

    def get_achieved_milestones(self) -> List[str]:
        """Get list of achieved milestone IDs."""
        return [mid for mid, prog in self.progress.items() if prog.achieved]

    def get_pending_for_stage(self, stage: int) -> List[str]:
        """Get milestones pending for a specific stage."""
        from .stages import STAGE_PROPERTIES, DevelopmentalStage

        stage_enum = DevelopmentalStage(stage)
        props = STAGE_PROPERTIES[stage_enum]

        pending = []
        for mid in props.required_milestones:
            if mid in self.progress and not self.progress[mid].achieved:
                pending.append(mid)

        return pending

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of milestone progress."""
        achieved = self.get_achieved_milestones()
        total = len(MILESTONES)

        by_category = {}
        for category in MilestoneCategory:
            cat_milestones = [m for m in MILESTONES.values() if m.category == category]
            cat_achieved = [m for m in cat_milestones if m.id in achieved]
            by_category[category.value] = {
                "total": len(cat_milestones),
                "achieved": len(cat_achieved),
            }

        return {
            "total_milestones": total,
            "achieved_count": len(achieved),
            "achieved_percentage": len(achieved) / total * 100 if total > 0 else 0,
            "achieved_milestones": achieved,
            "by_category": by_category,
        }
