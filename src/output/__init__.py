"""
AVA Output System

Implements developmental filtering of responses, including:
- Articulation simulation (garbled speech at early stages)
- Vocabulary constraints
- Complexity limitations
"""

from .articulation import ArticulationModel
from .filter import DevelopmentalFilter, FilteredResponse

__all__ = [
    "ArticulationModel",
    "DevelopmentalFilter",
    "FilteredResponse",
]
