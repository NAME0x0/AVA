# AVA - Developmental AI 🌱🧒🧠

**An AI that learns and grows like a human child, from illegible babbling to coherent conversation.**

---

[![Project Status](https://img.shields.io/badge/status-active%20development-brightgreen)](https://github.com/NAME0x0/AVA)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-blue)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

## Overview

AVA is a **Developmental AI** - an artificial intelligence that matures like a human child. Instead of being a fully capable assistant from day one, AVA starts as an "infant" with poor articulation but underlying knowledge. Through interaction, it develops:

- **Better articulation** - Speech clarity improves from garbled to coherent
- **Emotional growth** - Hope, fear, joy, surprise, and ambition influence behavior  
- **Tool progression** - Access to tools unlocks as AVA matures
- **Memory formation** - Episodic and semantic memories shape future responses
- **Continual learning** - QLoRA fine-tuning from high-quality interactions

Just like humans stop major brain development around age 25, AVA's development caps at the "MATURE" stage.

## 🌟 Key Concepts

### Developmental Stages

| Stage | Age (days) | Clarity | Description |
|-------|------------|---------|-------------|
| 👶 INFANT | 0-30 | 20% | Garbled speech, basic emotions, baby-safe tools only |
| 🧒 TODDLER | 30-90 | 40% | Simple sentences, can ask questions, limited tools |
| 👦 CHILD | 90-365 | 70% | Conversations, enthusiasm, moderate tools |
| 🧑 ADOLESCENT | 365-730 | 85% | Opinions, personality, most tools |
| 👨 YOUNG_ADULT | 730-1825 | 95% | Complex topics, nuanced views, advanced tools |
| 🧓 MATURE | 1825+ | 100% | Full capability, all tools, development complete |

### Emotional System

AVA has five core emotions that influence its behavior:

- **Hope** 🌈 - Increases learning rate, optimistic responses
- **Fear** 😰 - Decreases learning rate, cautious responses  
- **Joy** 😊 - Boosts learning, enthusiastic responses
- **Surprise** 😲 - Temporary learning boost, curious responses
- **Ambition** 🚀 - Increased learning, goal-oriented responses

### Tool Progression

Tools are "gated" by developmental stage - like not giving knives to babies:

- **Level 0** (Infant): echo, time, simple_math
- **Level 1** (Toddler): calculator, word_count
- **Level 2** (Child): web_search, file_read
- **Level 3** (Adolescent): file_write, note_save
- **Level 4** (Young Adult): code_execute, api_call
- **Level 5** (Mature): system_command

### Knowledge Without Articulation

AVA has access to the underlying LLM's knowledge from birth - it knows things but can't express them well initially. Development is about learning to *articulate*, not learning *facts*.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with a model (e.g., `llama3.2`)

### Installation

```bash
# Clone the repository
git clone https://github.com/NAME0x0/AVA.git
cd AVA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with a model
ollama pull llama3.2
```

### Run AVA

```bash
# Start the CLI
python -m src.cli

# Or with options
python -m src.cli --model llama3.2 --data-dir ./data
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/status` | Show AVA's full status |
| `/stage` | Show developmental stage info |
| `/emotions` | Show emotional state |
| `/good` | Positive feedback on last response |
| `/bad` | Negative feedback on last response |
| `/correct <text>` | Provide correction for last response |
| `/quit` | Exit (saves state) |

## 📁 Project Structure

```
AVA/
├── config/
│   ├── development_stages.yaml    # Stage definitions
│   ├── emotions.yaml              # Emotion parameters
│   ├── tools.yaml                 # Tool safety levels
│   └── learning.yaml              # Fine-tuning settings
│
├── data/
│   ├── developmental/             # Stage persistence
│   ├── emotional/                 # Emotion history
│   ├── memory/                    # Episodic & semantic memories
│   └── learning/                  # Training samples
│
├── src/
│   ├── developmental/             # Stage tracking & milestones
│   ├── emotional/                 # Emotion engine & modulation
│   ├── memory/                    # Memory systems
│   ├── learning/                  # Continual learning & fine-tuning
│   ├── inference/                 # Test-time compute (thinking)
│   ├── tools/                     # Tool registry with safety levels
│   ├── output/                    # Developmental output filtering
│   ├── agent.py                   # Main DevelopmentalAgent
│   └── cli.py                     # Command-line interface
│
├── models/
│   └── fine_tuned_adapters/       # QLoRA adapters from learning
│
└── tests/
```

## 🔬 How It Works

### Interaction Loop

1. **Emotion decay** - Emotions decay toward baseline since last interaction
2. **Context retrieval** - Fetch relevant memories and learning context
3. **Extended thinking** - Test-time compute based on developmental stage
4. **Tool execution** - Execute any needed tools (if accessible)
5. **Response generation** - Generate raw response from LLM
6. **Self-reflection** - Critique and revise (if mature enough)
7. **Developmental filter** - Apply articulation and vocabulary constraints
8. **Memory storage** - Store interaction as episodic memory
9. **Learning sample** - Record for potential fine-tuning
10. **State update** - Update developmental metrics, check milestones

### Learning & Fine-Tuning

AVA collects high-quality interaction samples and periodically fine-tunes using QLoRA when:
- 100+ quality samples accumulated
- 7+ days since last fine-tune
- Stage transition occurred
- High ambition + joy (emotional boost)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

Key areas for contribution:
- New milestone definitions
- Additional tools at various safety levels
- Improved articulation simulation
- Better emotion triggers
- Memory consolidation algorithms

## 📜 License

MIT License - see [LICENSE](LICENSE)

---

*"The journey of a thousand miles begins with a single step." - AVA starts as a babbling infant and grows into a capable assistant through patient interaction.* 