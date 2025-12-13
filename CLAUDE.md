# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AVA (Afsah's Virtual Assistant) is a local agentic AI designed to run on consumer hardware with limited VRAM (targeting NVIDIA RTX A2000 4GB). The project implements a complete agentic workflow with function calling, reasoning mechanisms, and multiple user interfaces.

## Commands

### Installation
```bash
# Create conda environment
conda create -n ava python=3.10 -y && conda activate ava

# Install PyTorch with CUDA (verify version on pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Running AVA CLI
```bash
# Interactive chat
python -m src.cli.main chat

# Single query
python -m src.cli.main query "Your question here"

# Check status
python -m src.cli.main status

# List available models
python -m src.cli.main models
```

### Model Quantization
```bash
# Test quantization with small model
python scripts/quantize_model.py --test

# Quantize a specific model
python scripts/quantize_model.py --model_id google/gemma-7b --output_path ./models/quantized/gemma-7b-4bit --strategy bnb_4bit
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

### Code Quality
```bash
# Linting
ruff check src/

# Formatting
black src/

# Type checking
mypy src/
```

## Architecture

### Core Components (`src/ava_core/`)

The agentic workflow is orchestrated by the `Agent` class which coordinates:

1. **DialogueManager** (`dialogue_manager.py`) - Manages conversation history and context
2. **FunctionCaller** (`function_calling.py`) - Parses and executes tool calls from LLM output
3. **ReasoningEngine** (`reasoning.py`) - Applies Chain-of-Thought and other reasoning mechanisms

The Agent processes input through this loop:
1. Perception (add to dialogue history)
2. Understanding (build context)
3. Planning (apply reasoning)
4. Action (execute function calls if needed)
5. Response (generate output)

### LLM Integration

- Primary: Ollama for local model serving
- Model format: 4-bit quantized using bitsandbytes
- Target: Gemma 3n 4B model (~2.3-2.6GB VRAM post-quantization)

### Tool System (`src/tools/`)

Tools are registered in `FunctionCaller` and called via structured output:
- String format: `[TOOL_CALL: tool_name(arg1="value")]`
- JSON format for native function calling

### Interfaces

- **CLI** (`src/cli/main.py`) - Typer-based with Rich formatting
- **Web** (`src/interfaces/web_interface.py`) - FastAPI with SSE streaming
- **GUI** - Open WebUI integration planned

### MCP Integration (`src/mcp_integration/`)

Model Context Protocol host for direct file/API access without RAG embeddings.

## Key Constraints

**4GB VRAM Limit**: All design decisions prioritize minimal VRAM usage:
- 4-bit quantization is mandatory
- Use QLoRA for fine-tuning
- Knowledge distillation from larger teacher models
- Monitor VRAM with `nvidia-smi` during development

## Configuration

- `config/base_config.yaml` - Default settings
- `config/user_config.yaml.example` - User config template
- `.env` - API keys (copy from `.env.example`)

## Development Notes

- Python 3.10 recommended
- Ollama must be running for full functionality (localhost:11434)
- When Ollama unavailable, Agent falls back to simulated responses for testing
