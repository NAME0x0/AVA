"""
AVA v3 Core - Cortex-Medulla Architecture
==========================================

This module implements the biomimetic Cortex-Medulla architecture designed
for autonomous, always-on operation on constrained hardware (RTX A2000 4GB).

Architecture Components:
========================

1. MEDULLA (The Reflexive Core):
   - Always-on sensory processing via 1-bit State Space Models (BitNet/Mamba)
   - Maintains fixed-size hidden state (O(1) memory) for infinite context
   - Handles wake-word detection, phatic responses, routing logic
   - VRAM footprint: ~1.5 GB (permanent resident)

2. CORTEX (The Reflective Core):
   - Deep reasoning via 70B+ parameter models (Llama-3 70B)
   - Layer-wise inference using AirLLM (paged from System RAM)
   - Activated only for high-complexity tasks
   - VRAM footprint: ~1.6 GB (revolving buffer during activation)

3. BRIDGE (State Projection):
   - Neural state projection from Medulla (Mamba) to Cortex (Transformer)
   - Enables instant context handoff without full-text pre-fill
   - Projection Adapter (MLP) maps hidden state to embedding space

4. AGENCY (Active Inference Controller):
   - Free Energy Principle (FEP) via pymdp library
   - Drives autonomous behavior through Variational Free Energy minimization
   - Resolves passive inference limitation through intrinsic motivation

5. MEMORY (Titans Neural Memory):
   - Test-time learning via surprise-driven weight updates
   - Compresses historical context into evolving synaptic weights
   - Fixed memory footprint regardless of operational duration

Reference Papers:
- BitNet b1.58: "The Era of 1-bit LLMs" (Microsoft Research, 2024)
- Mamba: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- AirLLM: "Run 70B LLMs on a Single 4GB GPU" (2024)
- Titans: "Learning to Memorize at Test Time" (2025)
- Active Inference: "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior" (2022)
"""

from .adapter_manager import AdapterConfig, AdapterManager, AdapterType
from .agency import (
    ActiveInferenceController,
    AgencyConfig,
    ASEAConfig,
    ASEAController,  # ASEA Algorithm
    ASEAState,
    PolicyType,
    VerificationResult,
)
from .bridge import Bridge, BridgeConfig
from .cortex import Cortex, CortexConfig
from .medulla import Medulla, MedullaConfig
from .system import AVACoreSystem, CoreConfig

# Note: V2 implementations have been archived to legacy/v2_core/

__all__ = [
    # Medulla (Reflexive Core)
    "Medulla",
    "MedullaConfig",
    # Cortex (Reflective Core)
    "Cortex",
    "CortexConfig",
    # Bridge (State Projection)
    "Bridge",
    "BridgeConfig",
    # Agency (Active Inference)
    "ActiveInferenceController",
    "AgencyConfig",
    "PolicyType",
    "VerificationResult",
    # ASEA (AVA Sentience & Efficiency Algorithm)
    "ASEAController",
    "ASEAConfig",
    "ASEAState",
    # Adapter Manager (Specialist LoRA Swapping)
    "AdapterManager",
    "AdapterConfig",
    "AdapterType",
    # Core System
    "AVACoreSystem",
    "CoreConfig",
]
