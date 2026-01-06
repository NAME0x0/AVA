# AVA Environment Variables Reference

> **Version**: 4.2.3 (Sentinel Architecture)  
> **Last Updated**: 2025-01-15

This document provides a comprehensive reference for all environment variables used by AVA.

## Quick Start

For most users, only these variables are needed:

```bash
# Minimal configuration
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gemma3:4b
export AVA_PORT=8085
```

## Environment Variables by Category

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AVA_HOST` | `127.0.0.1` | Host address to bind the API server |
| `AVA_PORT` | `8085` | Port for the API server |
| `AVA_DEBUG` | `false` | Enable debug mode with verbose logging |
| `AVA_SIMULATION_MODE` | `false` | Enable simulation mode (no actual AI inference, returns mock responses) |

### Ollama Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint URL |
| `OLLAMA_MODEL` | `gemma3:4b` | Default model for Medulla (fast brain) |
| `OLLAMA_CORTEX_MODEL` | *(optional)* | Model for Cortex (deep thinking). If not set, uses same as OLLAMA_MODEL |
| `AVA_MODEL` | *(from config)* | Override model from Tauri UI. Takes precedence over OLLAMA_MODEL |

**Recommended Models by Hardware:**

| VRAM | Medulla Model | Cortex Model |
|------|---------------|--------------|
| 4GB | `gemma3:4b` | `gemma3:4b` |
| 8GB | `llama3.2:8b` | `llama3.1:8b` |
| 16GB | `llama3.1:8b` | `qwen2.5:32b-q4` |
| 24GB+ | `qwen2.5:14b` | `qwen2.5:72b-q4` |

### GPU & Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `AVA_MAX_GPU_POWER_PERCENT` | `15` | Max GPU power % (prevents thermal throttling on laptops) |
| `AVA_MAX_TOKENS` | `2048` | Maximum tokens per response |
| `AVA_GPU_MEMORY_FRACTION` | `0.8` | Fraction of GPU memory to use (0.0-1.0) |
| `AVA_GPU_MEMORY_LIMIT` | `3500` | GPU memory limit in MB |
| `AVA_BATCH_SIZE` | `1` | Batch size for inference |
| `AVA_FORCE_CPU` | `false` | Force CPU-only mode (ignores GPU) |

**Performance Tuning Tips:**

- For RTX A2000 (4GB): Set `AVA_GPU_MEMORY_LIMIT=3500` and `AVA_MAX_GPU_POWER_PERCENT=15`
- For laptop GPUs: Lower `AVA_GPU_MEMORY_FRACTION` to prevent OOM
- For shared environments: Set `AVA_BATCH_SIZE=1` to minimize VRAM usage

### Memory & Context (Sentinel Architecture)

| Variable | Default | Description |
|----------|---------|-------------|
| `AVA_MAX_MEMORY_TURNS` | `50` | Max conversation turns in memory |
| `AVA_TITANS_ENABLED` | `true` | Enable Titans neural memory for test-time learning |

**Titans Memory:**

When enabled, Titans provides:
- Test-time learning (updates during inference)
- Infinite context through memory augmentation
- Surprise-weighted memory updates

### Search & Web (Search-First Paradigm)

| Variable | Default | Description |
|----------|---------|-------------|
| `AVA_SEARCH_FIRST` | `true` | Enable search-first paradigm for factual queries |
| `AVA_MIN_SOURCES` | `3` | Minimum sources required for factual claims |

**Search API Keys (Optional):**

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google Custom Search API key |
| `GOOGLE_CSE_ID` | Google Custom Search Engine ID |
| `BING_API_KEY` | Bing Search API key |
| `SERPAPI_KEY` | SerpAPI key (supports multiple engines) |
| `BRAVE_API_KEY` | Brave Search API key |

**Note:** If no API keys are set, AVA falls back to DuckDuckGo search (no API key required).

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `AVA_REQUIRE_COMMAND_CONFIRMATION` | `true` | Require confirmation for system commands (Level 5 tools) |
| `AVA_ETHICAL_CONSTRAINTS` | `true` | Enable ethical content filtering |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |
| `AVA_LOG_LEVEL` | *(fallback)* | Alternative log level (uses LOG_LEVEL if not set) |
| `LOG_FILE` | *(stdout)* | Log file path (logs to stdout if not set) |

### Data & Configuration Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `AVA_DATA_DIR` | `./data` | Data directory for persistence (memory, checkpoints) |
| `AVA_CONFIG` | *(auto-detect)* | Configuration file path |
| `AVA_CONFIG_PATH` | *(fallback)* | Alternative config path |

### Development

| Variable | Default | Description |
|----------|---------|-------------|
| `AVA_HOT_RELOAD` | `false` | Enable hot reload for development |
| `AVA_PROFILING` | `false` | Enable profiling |

## Example Configurations

### Development Setup

```bash
# .env.development
AVA_DEBUG=true
AVA_LOG_LEVEL=DEBUG
AVA_HOT_RELOAD=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
AVA_PORT=8085
```

### Production Setup (Low VRAM)

```bash
# .env.production
AVA_DEBUG=false
AVA_LOG_LEVEL=INFO
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
AVA_MAX_GPU_POWER_PERCENT=20
AVA_GPU_MEMORY_LIMIT=3500
AVA_MAX_TOKENS=1024
AVA_SEARCH_FIRST=true
AVA_MIN_SOURCES=3
```

### Production Setup (High VRAM)

```bash
# .env.production.high_vram
AVA_DEBUG=false
AVA_LOG_LEVEL=INFO
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_CORTEX_MODEL=qwen2.5:32b-q4
AVA_GPU_MEMORY_FRACTION=0.9
AVA_MAX_TOKENS=4096
AVA_SEARCH_FIRST=true
AVA_TITANS_ENABLED=true
```

### Headless Server (No GUI)

```bash
# .env.headless
AVA_HOST=0.0.0.0
AVA_PORT=8085
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
AVA_FORCE_CPU=false
LOG_LEVEL=INFO
LOG_FILE=/var/log/ava/server.log
```

## Loading Environment Variables

### Using .env Files

AVA automatically loads `.env` files from the project root:

```bash
# Create .env file in project root
touch .env
echo "OLLAMA_MODEL=gemma3:4b" >> .env
```

### Python (src/core/)

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file
model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
```

### Rust (ui/src-tauri/)

```rust
use std::env;

let host = env::var("OLLAMA_HOST")
    .unwrap_or_else(|_| "http://localhost:11434".to_string());
```

## Troubleshooting

### "OLLAMA_HOST not found"

Ensure Ollama is running and the environment variable is set:

```bash
curl $OLLAMA_HOST/api/tags  # Should list available models
```

### "Out of GPU Memory"

Lower the GPU memory settings:

```bash
export AVA_GPU_MEMORY_LIMIT=2500
export AVA_GPU_MEMORY_FRACTION=0.6
export AVA_BATCH_SIZE=1
```

### "Model not found"

Pull the model in Ollama first:

```bash
ollama pull gemma3:4b
```

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Full configuration reference
- [INSTALLATION.md](INSTALLATION.md) - Installation guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
