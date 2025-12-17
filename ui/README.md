# AVA Neural UI

A lightweight, GPU-accelerated interface for the AVA cognitive system built in Rust with egui.

## Features

- **Minimalist Design**: Deep space aesthetic with neural accent colors
- **Real-time Metrics**: Live entropy, varentropy, and surprise visualization
- **Neural Activity Waveform**: Animated brain wave display
- **Cognitive State Badge**: Visual indicator of FLOW/HESITATION/CONFUSION/CREATIVE states
- **Sleep Cycle Monitor**: Track Nightmare Engine phases
- **Smooth Animations**: Eased transitions and pulsing effects
- **Lightweight**: ~5MB binary, <50MB RAM usage

## Prerequisites

- [Rust](https://rustup.rs/) (stable toolchain)
- AVA Python backend running

## Building

### Windows
```batch
.\build.bat
```

### Linux/macOS
```bash
chmod +x build.sh
./build.sh
```

### Manual
```bash
cargo build --release
```

## Running

1. Start the AVA API server:
```bash
python api_server.py --port 8080
```

2. Launch the UI:
```bash
# Windows
.\target\release\ava-ui.exe

# Linux/macOS
./target/release/ava-ui
```

Or use the launcher:
```bash
# From AVA root directory
.\launch.bat  # Windows
```

## Architecture

```
ui/
├── Cargo.toml          # Dependencies
├── src/
│   ├── main.rs         # Entry point
│   ├── app.rs          # Main application logic
│   ├── theme.rs        # Colors and visual styling
│   ├── components.rs   # Reusable UI widgets
│   ├── animations.rs   # Easing and animation utilities
│   └── backend.rs      # API communication
└── build.bat           # Windows build script
```

## Key Components

### Theme
- `NeuralColors`: Deep space blacks with cyan accent
- Cognitive state colors (FLOW=green, CONFUSION=red, etc.)
- Sleep phase gradients

### Widgets
- `NeuralActivityIndicator`: Live waveform visualization
- `MetricCard`: Animated value display
- `GlowingInput`: Input field with focus glow
- `surprise_gauge`: Circular surprise meter
- `cognitive_state_badge`: State indicator pill
- `chat_bubble`: Message display with metadata

### Animations
- `AnimatedValue`: Smooth value transitions
- `PulseAnimation`: Breathing/pulsing effects
- `SpringAnimation`: Physics-based motion
- `TypewriterAnimation`: Text reveal effect

## Demo Mode

If the backend isn't connected, the UI runs in demo mode with simulated responses and cognitive states.

## Performance

- 60 FPS with minimal CPU usage
- GPU-accelerated via wgpu
- ~5MB release binary (with LTO)
- ~30-50MB RAM usage

## License

MIT License - see LICENSE
