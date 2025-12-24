# Specialist LoRA Adapters

This directory contains domain-specific LoRA adapters for hot-swapping during inference.

## Directory Structure

```
adapters/
├── coding_expert/    # Software development specialist
├── logic_expert/     # Logical reasoning specialist
└── butler_expert/    # Conversational assistant
```

## Adapter Format

Each adapter directory should contain:
- `adapter_config.json` - PEFT adapter configuration
- `adapter_model.safetensors` - Adapter weights

## Training

Adapters are trained using PEFT/LoRA on domain-specific datasets:
- **coding_expert**: Code completion, debugging, explanation tasks
- **logic_expert**: Logical proofs, deduction, inference tasks
- **butler_expert**: Scheduling, reminders, general assistance

## Usage

The `AdapterManager` (`src/core/adapter_manager.py`) automatically:
1. Classifies incoming queries by domain
2. Hot-swaps the appropriate adapter
3. Falls back to base model if confidence < 0.6

## Memory Footprint

Each adapter: ~100-500 MB in system RAM
Swap time: ~50-100ms per adapter
