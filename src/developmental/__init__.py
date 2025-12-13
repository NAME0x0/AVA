"""
AVA Developmental System

Core system for tracking and managing AVA's developmental progression
from infant to mature AI, including stage management, maturation tracking,
and competency milestones.
"""

from .stages import DevelopmentalStage, StageProperties, STAGE_PROPERTIES
from .tracker import DevelopmentTracker, DevelopmentalState
from .milestones import Milestone, MilestoneChecker, MILESTONES

__all__ = [
    "DevelopmentalStage",
    "StageProperties",
    "STAGE_PROPERTIES",
    "DevelopmentTracker",
    "DevelopmentalState",
    "Milestone",
    "MilestoneChecker",
    "MILESTONES",
]
