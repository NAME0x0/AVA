# Bridge Adapter

Neural state projection adapter for Medulla-to-Cortex handoff.

## Purpose

Projects Medulla (Mamba SSM) hidden state to Cortex (Transformer) embedding space,
enabling instant context handoff without full-text pre-fill.

## Architecture

```
Medulla State (2560-dim)
        │
        ▼
   Linear(2560 → 4096)
        │
   LayerNorm + ReLU
        │
   Linear(4096 → 4096)
        │
   LayerNorm + ReLU
        │
   Linear(4096 → 8192)
        │
        ▼
Cortex Embeddings (8192-dim × 32 tokens)
```

## Configuration

From `config/cortex_medulla.yaml`:
- `medulla_state_dim`: 2560
- `cortex_embedding_dim`: 8192
- `hidden_dims`: [4096, 4096]
- `num_soft_tokens`: 32

## Training

Trained to minimize reconstruction loss between:
- Cortex output from full text pre-fill
- Cortex output from projected Medulla state

## Placeholder

This directory is ready for trained bridge weights. Place the following files here:
- `bridge_adapter.pt` or `bridge_adapter.safetensors`
