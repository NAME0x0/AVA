# Changelog

All notable changes to AVA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pre-commit configuration for code quality
- CHANGELOG.md for version history tracking

### Changed
- Replaced alert() with inline help modal in CommandPalette
- Added WebGL fallback for 3D visualizations
- Improved canvas null safety checks

### Fixed
- Fixed broken Tailwind class in SplitPane component
- Fixed potential canvas context null errors

---

## [3.2.0] - 2024-12-22

### Added
- Production polish and infrastructure improvements
- Type-check script for CI validation
- Canvas null safety checks
- WebGL fallback for unsupported browsers

### Changed
- Replaced print() statements with proper logging
- Updated UI README with correct server references
- Improved error handling in Three.js components

### Fixed
- Fixed `bg-surface-lighter` Tailwind class issue
- Fixed CommandPalette help using alert()

---

## [3.1.0] - 2024-12-15

### Added
- **Desktop GUI**: Native Tauri + Next.js application
  - Real-time neural activity visualization
  - Command palette (Ctrl+K)
  - Cognitive state monitoring
  - 3D brain visualization with Three.js
- **Terminal UI**: Full-featured TUI with Textual
  - Split-pane layout
  - Keyboard shortcuts
  - Real-time metrics display
- **HTTP API Server**: REST + WebSocket endpoints
  - Streaming responses
  - Tool execution API
  - Health monitoring

### Changed
- Unified configuration system (ava.yaml)
- Improved tool execution with MCP support
- Enhanced memory management

### Fixed
- Tool registration race conditions
- WebSocket reconnection logic
- Memory consolidation timing

---

## [3.0.0] - 2024-12-01

### Added
- **Cortex-Medulla Architecture**: Dual-brain system
  - Medulla: Fast reflexive responses for simple queries
  - Cortex: Deep reasoning for complex problems
  - Adaptive routing based on query complexity
- **Search-First Paradigm**: Web search as default for informational queries
- **Titans Neural Memory**: Infinite context through test-time learning
- **Active Inference**: Autonomous behavior using Free Energy Principle
- **Entropy-Based Decision Making**: Query complexity analysis

### Changed
- Complete architectural rewrite from v2
- New configuration format (YAML-based)
- Improved tool system with native MCP support

### Removed
- Legacy Frankenstein architecture
- Old emotional processing system
- Deprecated v1 memory system

---

## [2.0.0] - 2024-10-15

### Added
- Frankenstein multi-brain architecture
- Emotional processing engine
- Developmental learning system
- Memory consolidation

### Changed
- Modular brain components
- Enhanced tool integration

---

## [1.0.0] - 2024-08-01

### Added
- Initial release
- Basic chat functionality
- Ollama integration
- Simple tool support

[Unreleased]: https://github.com/NAME0x0/AVA/compare/v3.2.0...HEAD
[3.2.0]: https://github.com/NAME0x0/AVA/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/NAME0x0/AVA/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/NAME0x0/AVA/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/NAME0x0/AVA/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/NAME0x0/AVA/releases/tag/v1.0.0
