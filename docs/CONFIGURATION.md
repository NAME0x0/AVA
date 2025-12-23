# Configuration Reference

This document describes all configuration options in `config/cortex_medulla.yaml`.

## Table of Contents

- [System](#system)
- [Medulla (Reflexive Core)](#medulla)
- [Cortex (Reflective Core)](#cortex)
- [Bridge (Neural Projection)](#bridge)
- [Agency (Active Inference)](#agency)
- [Search-First](#search-first)
- [Thermal Management](#thermal)
- [Episodic Memory](#episodic-memory)
- [Titans (Neural Memory)](#titans)
- [Hardware Profile](#hardware)
- [Sensory Inputs](#sensory)
- [Output](#output)
- [Development](#development)

---

## System

Global system settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `data_dir` | string | `"data"` | Directory for persistent data |
| `log_level` | string | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `main_loop_interval` | float | `0.1` | Main loop timing in seconds (100ms) |
| `idle_loop_interval` | float | `1.0` | Idle loop timing when waiting for input |
| `max_cortex_time` | float | `300.0` | Maximum seconds per Cortex invocation |
| `emergency_shutdown_phrase` | string | `"ava shutdown"` | Phrase to trigger emergency shutdown |
| `autosave_interval` | int | `100` | Save state every N interactions |

---

## Medulla

The always-on reflexive core using 1-bit State Space Models.

**VRAM Usage:** ~1.5 GB (resident)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `monitor_model` | string | `"slender-mamba-2.7b"` | SSM model for sensory processing |
| `talker_model` | string | `"bitnet-3b"` | 1.58-bit model for quick responses |
| `hidden_dim` | int | `2560` | Mamba hidden state dimension |
| `state_dim` | int | `16` | SSM state dimension |
| `low_surprise_threshold` | float | `0.3` | Below = routine (Medulla handles) |
| `high_surprise_threshold` | float | `2.0` | Above = invoke Cortex |
| `max_reflex_tokens` | int | `32` | Maximum tokens for quick response |
| `reflex_timeout_ms` | int | `200` | Target latency in milliseconds |
| `state_save_path` | string | `"data/memory/medulla_state.npz"` | State persistence path |
| `state_save_interval` | int | `100` | Save state every N updates |
| `device` | string | `"cuda"` | Compute device (cuda, cpu) |
| `use_fp16` | bool | `true` | Use half-precision floats |

### Surprise Thresholds

The surprise signal determines routing:

```
surprise < 0.3  → Medulla handles (reflex response)
surprise >= 2.0 → Cortex invoked (deep reasoning)
```

---

## Cortex

Deep reasoning via AirLLM layer-wise inference.

**VRAM Usage:** ~1.6 GB per layer (paged from system RAM)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_name` | string | `"meta-llama/Meta-Llama-3-70B-Instruct"` | HuggingFace model ID |
| `compression` | string | `"4bit"` | Quantization (4bit, 8bit, none) |
| `prefetch_layers` | int | `1` | Layers to prefetch (limited by VRAM) |
| `use_safetensors` | bool | `true` | Zero-copy memory mapping |
| `use_flash_attention` | bool | `true` | Use Flash Attention 2 |
| `max_new_tokens` | int | `512` | Maximum tokens to generate |
| `temperature` | float | `0.7` | Sampling temperature |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `top_k` | int | `40` | Top-K sampling |
| `repetition_penalty` | float | `1.1` | Repetition penalty |
| `max_context_length` | int | `4096` | Maximum context window |
| `max_input_tokens` | int | `2048` | Maximum input tokens |
| `offload_to_disk` | bool | `false` | Offload layers to disk |
| `disk_offload_path` | string | `"data/.cortex_cache"` | Disk cache path |
| `batch_size` | int | `1` | Batch size (always 1 for layer-wise) |
| `pin_memory` | bool | `true` | Pin memory for faster transfers |
| `device` | string | `"cuda"` | Compute device |
| `gpu_id` | int | `0` | GPU device ID |

### Alternative Models

For testing on smaller hardware:

```yaml
cortex:
  model_name: "meta-llama/Meta-Llama-3-8B-Instruct"  # 8B model
  # or
  model_name: "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Mixtral
```

---

## Bridge

Projects Medulla state to Cortex embeddings for instant context handoff.

**VRAM Usage:** ~50 MB

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `medulla_state_dim` | int | `2560` | Must match `medulla.hidden_dim` |
| `cortex_embedding_dim` | int | `8192` | Transformer hidden size |
| `hidden_dims` | list | `[4096, 4096]` | Projection MLP layers |
| `num_soft_tokens` | int | `32` | Virtual context tokens |
| `learning_rate` | float | `0.0001` | Training learning rate |
| `dropout` | float | `0.1` | Dropout rate |
| `use_layer_norm` | bool | `true` | Apply layer normalization |
| `use_residual` | bool | `true` | Use residual connections |
| `adapter_path` | string | `"models/fine_tuned_adapters/bridge"` | Adapter weights path |

---

## Agency

Active Inference controller using the Free Energy Principle.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `action_threshold` | float | `0.3` | Minimum G reduction to act |
| `urgency_threshold` | float | `0.7` | High urgency = immediate action |
| `idle_uncertainty_rate` | float | `0.01` | Uncertainty growth per second |
| `max_wait_time` | float | `300.0` | Max seconds before proactive action |
| `pragmatic_weight` | float | `0.4` | Weight for goal achievement |
| `epistemic_weight` | float | `0.6` | Weight for information gain |
| `cortex_effort_cost` | float | `0.5` | Penalty for Cortex (slow) |
| `tool_effort_cost` | float | `0.2` | Penalty for tool use |
| `search_effort_cost` | float | `0.05` | Penalty for web search (very low) |
| `search_first_enabled` | bool | `true` | Web search as default action |
| `belief_learning_rate` | float | `0.1` | Belief update rate |
| `preference_adaptation` | bool | `true` | Adapt to user preferences |
| `self_preservation_enabled` | bool | `true` | Enable self-preservation |
| `self_health_monitoring` | bool | `true` | Monitor system health |
| `health_check_interval` | float | `60.0` | Health check interval (seconds) |
| `require_confirmation_for_system` | bool | `true` | Confirm system commands |
| `blocked_system_commands` | list | see config | Blocked dangerous commands |
| `state_save_path` | string | `"data/memory/agency_state.json"` | State persistence path |

### Blocked Commands

The following system commands are always blocked:

```yaml
blocked_system_commands:
  - "rm -rf"
  - "del /f"
  - "format"
  - "shutdown"
  - "reboot"
  - "kill -9"
  - "taskkill /f"
  - "dd if="
  - "mkfs"
  - "fdisk"
```

---

## Search-First

Web search as the default action for informational queries.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable search-first paradigm |
| `min_sources` | int | `3` | Minimum sources to check |
| `max_sources` | int | `10` | Maximum sources to check |
| `fact_convergence_threshold` | float | `0.7` | Agreement threshold for facts |
| `primary_provider` | string | `"duckduckgo"` | Primary search provider |
| `fallback_providers` | list | `["brave", "searx"]` | Fallback providers |
| `max_content_length` | int | `5000` | Max chars per page |
| `extract_timeout_seconds` | int | `10` | Page extraction timeout |
| `cross_reference_minimum` | int | `2` | Minimum agreeing sources |
| `require_date_recency` | bool | `false` | Filter by date |

---

## Thermal

GPU thermal management for self-preservation.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable thermal management |
| `check_interval_seconds` | float | `5.0` | Temperature check interval |
| `warning_temp` | float | `75.0` | Warning threshold (°C) |
| `throttle_temp` | float | `80.0` | Throttle threshold (°C) |
| `pause_temp` | float | `85.0` | Pause threshold (°C) |
| `max_gpu_power_percent` | float | `15.0` | Max GPU power (% of TDP) |
| `throttle_on_warning` | bool | `true` | Throttle at warning temp |
| `pause_on_critical` | bool | `true` | Pause at critical temp |

### RTX A2000 Power Budget

```
TDP: 70W
15% limit: 10.5W
```

---

## Episodic Memory

Experience storage with timestamp-based retrieval.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable episodic memory |
| `storage_path` | string | `"data/memory/episodic"` | Storage directory |
| `max_entries` | int | `10000` | Maximum stored memories |
| `auto_save` | bool | `true` | Auto-save on interval |
| `save_interval` | int | `50` | Save every N new memories |
| `semantic_search_enabled` | bool | `true` | Enable semantic search |
| `date_range_search_enabled` | bool | `true` | Enable date range queries |

---

## Titans

Neural memory using test-time learning for infinite context.

**VRAM Usage:** ~200 MB (fixed, regardless of history length)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `input_dim` | int | `768` | Input embedding dimension |
| `hidden_dim` | int | `1024` | Memory MLP hidden size |
| `output_dim` | int | `768` | Output dimension |
| `num_layers` | int | `3` | Number of MLP layers |
| `use_layer_norm` | bool | `true` | Apply layer normalization |
| `dropout` | float | `0.1` | Dropout rate |
| `learning_rate` | float | `0.001` | Weight update rate |
| `momentum` | float | `0.9` | Gradient momentum |
| `forget_alpha` | float | `0.01` | Forgetting rate |
| `surprise_threshold` | float | `0.5` | Min surprise for update |
| `high_surprise_threshold` | float | `2.0` | Threshold for episodic storage |
| `max_stored_episodes` | int | `1000` | Max episodes stored |
| `state_save_path` | string | `"data/memory/titans_state.pkl"` | State persistence path |

---

## Hardware

Reference specifications for target hardware.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gpu_name` | string | `"NVIDIA RTX A2000"` | GPU model name |
| `vram_total_mb` | int | `4096` | Total VRAM in MB |
| `memory_bandwidth_gbps` | int | `192` | Memory bandwidth |
| `cuda_cores` | int | `3328` | CUDA core count |
| `tensor_cores` | bool | `true` | Has tensor cores |
| `pcie_gen` | int | `4` | PCIe generation |
| `pcie_lanes` | int | `16` | PCIe lanes |
| `effective_bandwidth_gbps` | int | `12` | Practical PCIe bandwidth |
| `tdp_watts` | int | `70` | Thermal design power |
| `layer_transfer_time_seconds` | float | `0.04` | Time per Cortex layer |
| `tokens_per_second_cortex` | float | `0.3` | Cortex token rate |
| `tokens_per_second_medulla` | int | `50` | Medulla token rate |

---

## Sensory

Input configuration for sensors.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_audio_input` | bool | `false` | Enable voice input |
| `audio_sample_rate` | int | `16000` | Audio sample rate |
| `audio_chunk_ms` | int | `100` | Audio chunk size |
| `whisper_model` | string | `"base.en"` | Whisper model for STT |
| `enable_log_monitoring` | bool | `true` | Monitor system logs |
| `log_sources` | list | see config | Log file paths |
| `monitor_cpu_temp` | bool | `true` | Monitor CPU temperature |
| `monitor_gpu_temp` | bool | `true` | Monitor GPU temperature |
| `monitor_memory` | bool | `true` | Monitor memory usage |

---

## Output

Output configuration.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_voice_output` | bool | `false` | Enable TTS output |
| `tts_model` | string | `"piper-tts"` | TTS model |
| `voice_speed` | float | `1.0` | Voice playback speed |
| `enable_rich_formatting` | bool | `true` | Enable rich text output |
| `max_response_length` | int | `2048` | Maximum response length |

---

## Development

Development and testing settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `simulation_mode` | bool | `false` | Run without real models |
| `use_small_models` | bool | `false` | Use smaller test models |
| `small_cortex_model` | string | `"meta-llama/Meta-Llama-3-8B-Instruct"` | Small Cortex model |
| `verbose_logging` | bool | `true` | Enable verbose logs |
| `log_agency_decisions` | bool | `true` | Log agency decisions |
| `log_surprise_signals` | bool | `true` | Log surprise signals |
| `enable_profiling` | bool | `false` | Enable performance profiling |
| `profile_output_path` | string | `"data/profiles"` | Profile output directory |

### Simulation Mode

For development without GPU:

```yaml
development:
  simulation_mode: true  # No real models loaded
```

---

## Environment Variables

Some settings can be overridden via environment variables:

| Variable | Description |
|----------|-------------|
| `AVA_SIMULATION_MODE` | Set to `"true"` to enable simulation mode |
| `NEXT_PUBLIC_BACKEND_URL` | Backend URL for the GUI (default: `http://localhost:8085`) |

---

## Example Configurations

### Minimal VRAM (Testing)

```yaml
medulla:
  monitor_model: "bi-mamba-1.3b"
  talker_model: "bitnet-1.3b"

cortex:
  model_name: "meta-llama/Meta-Llama-3-8B-Instruct"

development:
  simulation_mode: true
```

### Production (RTX A2000)

```yaml
medulla:
  monitor_model: "slender-mamba-2.7b"
  talker_model: "bitnet-3b"

cortex:
  model_name: "meta-llama/Meta-Llama-3-70B-Instruct"
  compression: "4bit"

development:
  simulation_mode: false
```

### High-Performance (RTX 3090/4090)

```yaml
cortex:
  model_name: "meta-llama/Meta-Llama-3-70B-Instruct"
  compression: "8bit"  # Better quality
  prefetch_layers: 4   # More prefetching

thermal:
  max_gpu_power_percent: 80.0  # Higher power limit
```
