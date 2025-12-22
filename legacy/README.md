# Legacy Code Archive

This directory contains archived code from previous AVA versions that is **not active** in the v3 Cortex-Medulla architecture.

## Contents

### `developmental/`
Stage-based developmental system from v1/v2:
- `stages.py` - Developmental stage definitions (INFANT, TODDLER, etc.)
- `milestones.py` - Achievement tracking
- `tracker.py` - Stage progression logic

### `emotional/`
Emotion processing system from v1/v2:
- `engine.py` - Emotional state machine
- `models.py` - Emotion data models
- `modulation.py` - Response modulation based on emotions

### `output/`
Legacy output filtering system:
- `articulation.py` - Response articulation by developmental stage
- `filter.py` - Output filtering and safety checks

### `memory/`
Old memory management (replaced by Titans in v3):
- `episodic.py` - Episodic memory store
- `semantic.py` - Semantic memory store
- `consolidation.py` - Memory consolidation logic
- `manager.py` - Memory orchestration
- `models.py` - Memory data models

### `v2_core/`
V2 Cortex-Medulla implementations (superseded by v3):
- `core_v2.py` - V2 core system
- `medulla_v2.py` - V2 Medulla implementation
- `cortex_v2.py` - V2 Cortex implementation

### `old_servers/`
Legacy API servers:
- `api_server.py` - Original Frankensystem server
- `api_server_v3.py` - Intermediate v3 server (superseded by `server.py`)

## Why Archived?

The v3 architecture fundamentally changed:
1. **Memory**: Titans Neural Memory replaces episodic/semantic stores
2. **Emotions**: Removed in favor of cognitive states (entropy, surprise)
3. **Development Stages**: Removed in favor of continuous learning
4. **Output Filtering**: Simplified - no articulation by stage

## Restoration

If you need to restore any module, move it back to `src/` and update imports.
These files are kept for reference and potential future integration.
