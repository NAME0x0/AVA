# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AVA is a **Developmental AI** - an artificial intelligence that matures like a human child. It starts as an "infant" with poor articulation but underlying LLM knowledge. Through interaction, it develops better communication, gains tool access, forms memories, and learns continuously.

**Core Paradigm:** Treat AI as an artificial human - slow, progressive learning with emotional states influencing behavior.

## Key Concepts

### Developmental Stages
- **INFANT** (0-30 days): 20% clarity, garbled speech, baby-safe tools
- **TODDLER** (30-90 days): 40% clarity, simple sentences
- **CHILD** (90-365 days): 70% clarity, conversations
- **ADOLESCENT** (365-730 days): 85% clarity, opinions
- **YOUNG_ADULT** (730-1825 days): 95% clarity, complex topics
- **MATURE** (1825+ days): 100% clarity, full capability, development caps

### Emotional System
Five emotions influence behavior:
- **Hope**: +learning rate, optimistic responses
- **Fear**: -learning rate, cautious responses
- **Joy**: +learning, enthusiastic responses
- **Surprise**: temporary boost, curious responses
- **Ambition**: +learning, goal-oriented responses

### Tool Safety Levels
Tools are gated by developmental stage (like not giving knives to babies):
- Level 0 (Infant): echo, time, simple_math
- Level 1 (Toddler): calculator, word_count
- Level 2 (Child): web_search, file_read
- Level 3 (Adolescent): file_write, note_save
- Level 4 (Young Adult): code_execute, api_call
- Level 5 (Mature): system_command

## Commands

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama pull llama3.2
```

### Running AVA
```bash
# Start CLI
python -m src.cli

# With options
python -m src.cli --model llama3.2 --data-dir ./data
```

### CLI Commands
- `/help` - Show commands
- `/status` - Show AVA's status
- `/stage` - Show developmental stage
- `/emotions` - Show emotional state
- `/good` / `/bad` - Feedback on last response
- `/correct <text>` - Provide correction
- `/quit` - Exit

### Testing
```bash
pytest tests/
```

## Architecture

### Directory Structure
```
src/
├── developmental/    # Stage tracking & milestones
│   ├── stages.py     # DevelopmentalStage enum, StageProperties
│   ├── tracker.py    # DevelopmentTracker, state persistence
│   └── milestones.py # Milestone definitions and checking
│
├── emotional/        # Emotion processing
│   ├── models.py     # EmotionVector, EmotionalState, triggers
│   ├── engine.py     # EmotionalEngine, decay, processing
│   └── modulation.py # How emotions affect learning/responses
│
├── memory/           # Memory systems
│   ├── episodic.py   # Event-based memories
│   ├── semantic.py   # Factual knowledge
│   ├── consolidation.py # Decay and strengthening
│   └── manager.py    # Memory orchestration
│
├── learning/         # Continual learning
│   ├── continual.py  # Sample collection
│   ├── fine_tuning.py # QLoRA scheduling
│   └── nested.py     # Meta-learning contexts
│
├── inference/        # Test-time compute
│   ├── thinking.py   # Extended reasoning
│   └── reflection.py # Self-critique
│
├── tools/            # Tool system with safety
│   ├── registry.py   # Tool registration with levels
│   ├── progression.py # Tool unlocking logic
│   └── base_tools.py # Tool implementations
│
├── output/           # Developmental filtering
│   ├── articulation.py # Speech development simulation
│   └── filter.py     # Vocabulary & complexity constraints
│
├── agent.py          # Main DevelopmentalAgent
└── cli.py            # CLI interface
```

### Main Interaction Loop (agent.py)
1. Apply emotion decay
2. Get developmental state & stage properties
3. Process emotional triggers from input
4. Retrieve relevant memories
5. Calculate thinking budget based on stage
6. Execute extended reasoning (test-time compute)
7. Check if tools needed → filter by safety level
8. Generate raw response from Ollama
9. Apply self-reflection (if mature enough)
10. Apply developmental filter (articulation, vocabulary)
11. Store episode in memory
12. Record learning sample
13. Update developmental metrics
14. Check milestones & stage transitions
15. Check fine-tuning triggers
16. Update emotions based on outcome

### LLM Integration
- Backend: Ollama (local)
- Default model: llama3.2
- Learning: QLoRA fine-tuning from collected samples

### Configuration Files
- `config/development_stages.yaml` - Stage thresholds
- `config/emotions.yaml` - Emotion parameters
- `config/tools.yaml` - Tool definitions
- `config/learning.yaml` - Fine-tuning settings

### Data Persistence
- `data/developmental/state.json` - Developmental state
- `data/emotional/state.json` - Emotional state
- `data/memory/` - Episodic and semantic memories
- `data/learning/samples/` - Training samples

## Key Implementation Details

### Articulation Model
At low clarity (INFANT/TODDLER), output is transformed:
- Letter substitutions: r→w, th→f ("rabbit"→"wabbit")
- Word dropping and simplification
- Sentence shortening
- Filler words added

### Fine-Tuning Triggers
- 100+ quality samples accumulated
- 7+ days since last fine-tune
- Stage transition occurred
- High ambition + joy (emotional boost)
- Performance degradation detected

### Memory Consolidation
Memories decay and strengthen over time:
- Recent memories are stronger
- Emotionally tagged memories persist longer
- Frequently accessed memories strengthen
- Sleep/consolidation cycles process memories

## Development Guidelines

### When adding new tools:
1. Define safety level (0-5)
2. Add to `base_tools.py`
3. Register in `registry.py`
4. Add unlock conditions in `tools.yaml`

### When modifying developmental behavior:
1. Check `STAGE_PROPERTIES` in `stages.py`
2. Update `development_stages.yaml` if needed
3. Ensure milestones align with stage transitions

### When working with emotions:
1. Triggers go through `EmotionalEngine.process_trigger()`
2. Modulation factors come from `modulation.py`
3. Baseline personality defined in `emotions.yaml`
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
