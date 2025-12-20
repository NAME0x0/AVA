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

from .medulla import Medulla, MedullaConfig
from .cortex_engine import Cortex, CortexConfig
from .bridge import Bridge, BridgeConfig
from .agency import ActiveInferenceController, AgencyConfig, PolicyType
from .core_loop import AVACoreSystem, CoreConfig

# V2 - Working implementations with Ollama
from .medulla_v2 import MedullaV2, MedullaConfig as MedullaConfigV2, CognitiveState, CognitiveLabel
from .cortex_v2 import CortexV2, CortexConfig as CortexConfigV2
from .core_v2 import AVACoreV2, CoreConfig as CoreConfigV2, ProcessResult, get_core, shutdown_core

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
    
    # Core System
    "AVACoreSystem",
    "CoreConfig",
]
