"""
Developmental Stages for AVA

Defines the six developmental stages that AVA progresses through,
mirroring human cognitive development from infancy to maturity.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List


class DevelopmentalStage(IntEnum):
    """
    Developmental stages mirroring human growth.

    AVA progresses through these stages based on a hybrid of:
    - Time passage (simulated aging)
    - Interaction count and quality
    - Competency milestones achieved

    Development caps at MATURE (like human brain development ~25 years).
    """
    INFANT = 0        # 0-30 days: Garbled output, basic tools only
    TODDLER = 1       # 30-90 days: Improving clarity, simple tools
    CHILD = 2         # 90-365 days: Coherent, read-only tools
    ADOLESCENT = 3    # 365-730 days: Complex reasoning, write tools
    YOUNG_ADULT = 4   # 730-1825 days: Near-mature, powerful tools
    MATURE = 5        # 1825+ days: Full capability, all tools (CAPPED)


@dataclass
class StageProperties:
    """
    Properties associated with each developmental stage.

    These properties modulate AVA's behavior, capabilities, and learning
    based on its current developmental level.
    """
    stage: DevelopmentalStage

    # Age thresholds (in days, with time acceleration applied)
    min_age_days: float
    max_age_days: float

    # Interaction thresholds
    min_interactions: int
    max_interactions: int

    # Capability multipliers (0.0 to 1.0)
    articulation_clarity: float   # How clear/coherent responses are
    vocabulary_range: float       # Range of vocabulary used (0.1 = 10% of full)
    reasoning_depth: float        # Complexity of reasoning allowed
    emotional_stability: float    # Emotional volatility (lower = more volatile)
    attention_span: float         # Context window utilization (0.2 = 20% of max)

    # Tool access level (0-5, matching ToolSafetyLevel)
    tool_safety_level: int

    # Learning rate modifier (higher = learns faster)
    learning_rate_multiplier: float

    # Required milestones for progression to NEXT stage
    required_milestones: List[str] = field(default_factory=list)

    # Thinking budget multiplier for test-time compute
    thinking_budget_multiplier: float = 0.25

    # Self-reflection enabled
    self_reflection_enabled: bool = False

    def get_max_context_tokens(self, base_context: int = 4096) -> int:
        """Calculate max context tokens based on attention span."""
        return int(base_context * self.attention_span)

    def get_thinking_budget(self, base_budget: int = 1024) -> int:
        """Calculate thinking token budget for test-time compute."""
        return int(base_budget * self.thinking_budget_multiplier)


# Define properties for each stage
STAGE_PROPERTIES: Dict[DevelopmentalStage, StageProperties] = {
    DevelopmentalStage.INFANT: StageProperties(
        stage=DevelopmentalStage.INFANT,
        min_age_days=0,
        max_age_days=30,
        min_interactions=0,
        max_interactions=100,
        articulation_clarity=0.2,
        vocabulary_range=0.1,
        reasoning_depth=0.1,
        emotional_stability=0.3,
        attention_span=0.2,
        tool_safety_level=0,
        learning_rate_multiplier=1.5,
        thinking_budget_multiplier=0.25,
        self_reflection_enabled=False,
        required_milestones=[
            "first_coherent_response",
            "first_tool_use",
            "emotional_expression",
        ],
    ),

    DevelopmentalStage.TODDLER: StageProperties(
        stage=DevelopmentalStage.TODDLER,
        min_age_days=30,
        max_age_days=90,
        min_interactions=100,
        max_interactions=500,
        articulation_clarity=0.4,
        vocabulary_range=0.3,
        reasoning_depth=0.25,
        emotional_stability=0.4,
        attention_span=0.35,
        tool_safety_level=1,
        learning_rate_multiplier=1.3,
        thinking_budget_multiplier=0.4,
        self_reflection_enabled=False,
        required_milestones=[
            "multi_turn_memory",
            "basic_reasoning",
            "tool_mastery_level_1",
            "vocabulary_expansion",
        ],
    ),

    DevelopmentalStage.CHILD: StageProperties(
        stage=DevelopmentalStage.CHILD,
        min_age_days=90,
        max_age_days=365,
        min_interactions=500,
        max_interactions=2000,
        articulation_clarity=0.7,
        vocabulary_range=0.5,
        reasoning_depth=0.5,
        emotional_stability=0.55,
        attention_span=0.5,
        tool_safety_level=2,
        learning_rate_multiplier=1.2,
        thinking_budget_multiplier=0.6,
        self_reflection_enabled=True,
        required_milestones=[
            "complex_reasoning",
            "emotional_regulation",
            "knowledge_synthesis",
            "tool_mastery_level_2",
        ],
    ),

    DevelopmentalStage.ADOLESCENT: StageProperties(
        stage=DevelopmentalStage.ADOLESCENT,
        min_age_days=365,
        max_age_days=730,
        min_interactions=2000,
        max_interactions=5000,
        articulation_clarity=0.85,
        vocabulary_range=0.7,
        reasoning_depth=0.7,
        emotional_stability=0.5,  # Adolescent volatility!
        attention_span=0.65,
        tool_safety_level=3,
        learning_rate_multiplier=1.0,
        thinking_budget_multiplier=0.8,
        self_reflection_enabled=True,
        required_milestones=[
            "self_reflection_demonstrated",
            "nuanced_understanding",
            "independent_problem_solving",
            "tool_mastery_level_3",
        ],
    ),

    DevelopmentalStage.YOUNG_ADULT: StageProperties(
        stage=DevelopmentalStage.YOUNG_ADULT,
        min_age_days=730,
        max_age_days=1825,
        min_interactions=5000,
        max_interactions=10000,
        articulation_clarity=0.95,
        vocabulary_range=0.9,
        reasoning_depth=0.9,
        emotional_stability=0.75,
        attention_span=0.85,
        tool_safety_level=4,
        learning_rate_multiplier=0.9,
        thinking_budget_multiplier=0.95,
        self_reflection_enabled=True,
        required_milestones=[
            "wisdom_demonstration",
            "emotional_intelligence",
            "meta_learning",
            "full_tool_proficiency",
        ],
    ),

    DevelopmentalStage.MATURE: StageProperties(
        stage=DevelopmentalStage.MATURE,
        min_age_days=1825,
        max_age_days=float('inf'),  # No upper limit - capped here
        min_interactions=10000,
        max_interactions=float('inf'),
        articulation_clarity=1.0,
        vocabulary_range=1.0,
        reasoning_depth=1.0,
        emotional_stability=0.9,
        attention_span=1.0,
        tool_safety_level=5,
        learning_rate_multiplier=0.7,  # Slower learning when mature
        thinking_budget_multiplier=1.0,
        self_reflection_enabled=True,
        required_milestones=[],  # No further progression
    ),
}


def get_stage_properties(stage: DevelopmentalStage) -> StageProperties:
    """Get the properties for a given developmental stage."""
    return STAGE_PROPERTIES[stage]


def get_next_stage(current: DevelopmentalStage) -> DevelopmentalStage | None:
    """Get the next developmental stage, or None if at MATURE."""
    if current == DevelopmentalStage.MATURE:
        return None
    return DevelopmentalStage(current.value + 1)


def stage_from_string(name: str) -> DevelopmentalStage:
    """Convert a string name to DevelopmentalStage enum."""
    return DevelopmentalStage[name.upper()]
