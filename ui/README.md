# AVA Neural Interface v3

A modern, performant UI for the AVA Cortex-Medulla architecture built with **Next.js 14**, **TypeScript**, **Tailwind CSS**, and **Tauri** (Rust).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AVA Neural Interface                      │
├─────────────────────────────────────────────────────────────┤
│  Next.js 14 (React 18) + TypeScript + Tailwind CSS          │
│  ├── Framer Motion (animations)                              │
│  ├── Zustand (state management)                              │
│  └── Lucide Icons                                            │
├─────────────────────────────────────────────────────────────┤
│  Tauri (Rust)                                                │
│  ├── Native window management                                │
│  ├── System tray integration                                 │
│  └── IPC bridge to Python backend                            │
├─────────────────────────────────────────────────────────────┤
│  Python Backend (server.py)                                  │
│  ├── HTTP REST API                                           │
│  ├── WebSocket streaming                                     │
│  └── Cortex-Medulla integration                              │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Visual Design
- **Deep space dark theme** with neural accent colors
- **Glassmorphism** effects with backdrop blur
- **Animated components** using Framer Motion
- **Responsive layout** with collapsible sidebar

### Real-time Monitoring
- **Cognitive state visualization** (Flow, Hesitation, Confusion, Creative)
- **Neural activity waveform** based on entropy/varentropy
- **Active component indicator** (Medulla vs Cortex)
- **Belief state visualization** from Active Inference

### Chat Interface
- **Message bubbles** with component attribution (Medulla/Cortex)
- **Streaming responses** via WebSocket
- **Response time tracking**
- **Cognitive state badges** per message

### System Controls
- **Force Cortex** - Trigger deep reasoning
- **Force Sleep** - Initiate consolidation cycle
- **Backend connection status**

## Project Structure

```
ui/
├── package.json           # Node.js dependencies
├── tsconfig.json          # TypeScript configuration
├── tailwind.config.ts     # Tailwind CSS configuration
├── next.config.js         # Next.js configuration
├── postcss.config.js      # PostCSS configuration
├── src/
│   ├── app/
│   │   ├── layout.tsx     # Root layout with fonts
│   │   ├── page.tsx       # Main page component
│   │   └── globals.css    # Global styles & Tailwind
│   ├── components/
│   │   ├── layout/
│   │   │   ├── TitleBar.tsx    # Custom window title bar
│   │   │   └── Sidebar.tsx     # Metrics sidebar
│   │   ├── chat/
│   │   │   ├── ChatArea.tsx    # Main chat container
│   │   │   ├── ChatInput.tsx   # Message input
│   │   │   └── MessageBubble.tsx
│   │   ├── metrics/
│   │   │   ├── CognitiveStateCard.tsx
│   │   │   ├── MetricCard.tsx
│   │   │   ├── NeuralActivity.tsx
│   │   │   └── BeliefStateCard.tsx
│   │   └── system/
│   │       └── SystemStatus.tsx
│   ├── stores/
│   │   └── core.ts        # Zustand state management
│   ├── hooks/
│   │   ├── useSystemPolling.ts
│   │   └── useStreamingChat.ts
│   └── lib/
│       └── utils.ts       # Utility functions
└── src-tauri/
    ├── Cargo.toml         # Rust dependencies
    ├── tauri.conf.json    # Tauri configuration
    └── src/
        ├── main.rs        # Tauri entry point
        ├── commands.rs    # IPC command handlers
        ├── state.rs       # Application state
        └── backend.rs     # Backend connection
```

## Getting Started

### Prerequisites

- **Node.js** 18+ (for Next.js)
- **Rust** (for Tauri) - Install from https://rustup.rs
- **Python** 3.10+ (for backend)

### Installation

```bash
# Navigate to UI directory
cd ui

# Install Node.js dependencies
npm install

# Install Rust dependencies (handled by Tauri)
```

### Development

#### Browser Mode (without Tauri)

```bash
# Start Next.js development server
npm run dev

# In another terminal, start Python backend
cd ..
python api_server_v3.py
```

Then open http://localhost:3000 in your browser.

#### Desktop Mode (with Tauri)

```bash
# Start Tauri development mode
npm run tauri:dev
```

This will:
1. Start Next.js dev server
2. Compile Rust code
3. Launch native window

### Building

#### Web Build

```bash
npm run build
```

Outputs static files to `out/` directory.

#### Desktop Build

```bash
npm run tauri:build
```

Creates platform-specific installers in `src-tauri/target/release/bundle/`.

## Configuration

### Backend URL

Default: `http://localhost:8085`

Can be changed in:
- **Browser**: Edit `backendUrl` in Zustand store
- **Tauri**: Use `set_backend_url` command

### Theme Colors

Edit `tailwind.config.ts` to customize:

```typescript
colors: {
  neural: {
    void: "#08080C",      // Deepest background
    surface: "#0E0E14",   // Card backgrounds
    elevated: "#16161E",  // Input backgrounds
    hover: "#1E1E2A",     // Hover states
  },
  accent: {
    primary: "#00D4C8",   // Main accent (cyan)
    dim: "#00968C",       // Dimmed accent
  },
  // ... more colors
}
```

## API Endpoints

The UI expects these endpoints from `api_server_v3.py`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/system/state` | Full system state |
| GET | `/cognitive` | Medulla cognitive state |
| GET | `/memory` | Titans memory stats |
| GET | `/belief` | Active Inference belief |
| POST | `/chat` | Send chat message |
| WS | `/chat/stream` | Streaming chat |
| POST | `/force_cortex` | Force Cortex mode |
| POST | `/sleep` | Trigger sleep cycle |

## Performance

### Optimizations

- **Static export** for Tauri (no server needed)
- **Tree-shaking** with Next.js 14
- **Selective component re-renders** with Zustand selectors
- **Canvas-based** neural activity visualization
- **Debounced polling** (2s intervals)

### Bundle Size

- Next.js: ~150KB (gzipped)
- Framer Motion: ~40KB
- Zustand: ~3KB
- Lucide Icons: Tree-shaken to used icons only

### Rust Binary

Release build with:
- LTO (Link Time Optimization)
- Single codegen unit
- Symbol stripping
- Abort on panic

## Troubleshooting

### "Connection failed" in UI

1. Ensure Python backend is running:
   ```bash
   python api_server_v3.py
   ```

2. Check CORS - backend should allow `http://localhost:3000`

3. Verify port 8085 is not in use

### Tauri build fails

1. Ensure Rust is installed:
   ```bash
   rustc --version
   ```

2. Update Rust:
   ```bash
   rustup update
   ```

3. Check Tauri prerequisites:
   ```bash
   npm run tauri info
   ```

### Styling issues

1. Clear Next.js cache:
   ```bash
   rm -rf .next
   npm run dev
   ```

2. Rebuild Tailwind:
   ```bash
   npx tailwindcss build
   ```

## License

Part of the AVA Project. See root LICENSE file.
