# Contributing to AVA

Thank you for your interest in contributing to AVA! This project thrives because of contributors like you. This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- **Be respectful**: Treat everyone with dignity and respect
- **Be constructive**: Offer helpful feedback and suggestions
- **Be inclusive**: Welcome newcomers and help them get started
- **Be patient**: Remember that everyone was a beginner once

Unacceptable behavior will not be tolerated. Report issues to the maintainers.

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.10+** installed
- **Node.js 18+** (for GUI development)
- **Git** for version control
- **Ollama** for running AI models
- An NVIDIA GPU with 4GB+ VRAM (recommended)

### First-Time Setup

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/AVA.git
cd AVA

# 3. Add upstream remote
git remote add upstream https://github.com/NAME0x0/AVA.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# 6. Install pre-commit hooks
pre-commit install

# 7. Verify setup
python -m pytest tests/ -v
```

---

## Development Setup

### Backend (Python)

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Type checking
mypy src/

# Linting
flake8 src/
black src/ --check
```

### Frontend (GUI)

```bash
cd ui

# Install dependencies
npm install

# Run development server
npm run dev

# Run with Tauri (desktop app)
npm run tauri dev

# Build for production
npm run build
npm run tauri build
```

### TUI (Terminal UI)

```bash
# Run TUI in development mode
textual run --dev tui.app:AVATUI

# With console for debugging
textual console  # In one terminal
textual run --dev tui.app:AVATUI  # In another
```

### Building the Installer

Prerequisites:
- **NSIS** - [Download](https://nsis.sourceforge.io/) (for Windows installer)
- **Rust & Cargo** - [Download](https://rustup.rs/)

```bash
# 1. Build the Tauri app first
cd ui
npm run tauri build

# 2. Build the installer
cd ../installer/scripts
python build_installer.py

# Or build portable version
python build_installer.py --portable

# Or build all variants
python build_installer.py --all
```

Output will be in `installer/dist/`:
- `AVA-{version}-Setup.exe` - Windows installer
- `AVA-{version}-portable.zip` - Portable version

---

## Project Structure

```
AVA/
├── src/                    # Python source code
│   ├── ava/               # Public API
│   │   ├── __init__.py    # AVA class export
│   │   ├── engine.py      # Main engine
│   │   ├── tools.py       # Tool system
│   │   ├── memory.py      # Memory management
│   │   └── config.py      # Configuration
│   ├── core/              # Cortex-Medulla architecture
│   │   ├── system.py      # Main orchestrator
│   │   ├── medulla.py     # Fast brain
│   │   ├── cortex.py      # Deep brain
│   │   ├── bridge.py      # Connection layer
│   │   └── agency.py      # Active Inference
│   ├── hippocampus/       # Memory systems
│   │   ├── titans.py      # Test-time learning
│   │   └── buffer.py      # Episodic buffer
│   └── ...
├── tui/                   # Terminal UI (Textual)
│   ├── app.py
│   ├── components/
│   └── styles/
├── ui/                    # Desktop GUI (Next.js + Tauri)
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   └── stores/
│   └── src-tauri/         # Rust backend
│       ├── src/
│       │   ├── main.rs
│       │   ├── tray.rs    # System tray
│       │   └── bug_report.rs  # Bug reporting
│       └── Cargo.toml
├── installer/             # Installer build system
│   ├── config/            # Installer configuration
│   ├── nsis/              # NSIS scripts
│   └── scripts/           # Build automation
├── tests/                 # Test suite
│   ├── unit/
│   └── integration/
├── docs/                  # Documentation
├── config/                # Configuration files
└── legacy/                # Archived v1/v2 code
```

### Key Files to Know

| File | Purpose |
|------|---------|
| `server.py` | HTTP API server entry point |
| `run_tui.py` | TUI entry point |
| `run_core.py` | Core system CLI |
| `src/core/system.py` | Main orchestrator |
| `src/core/medulla.py` | Fast response brain |
| `src/core/cortex.py` | Deep thinking brain |
| `config/cortex_medulla.yaml` | Main configuration |

---

## How to Contribute

### Types of Contributions

#### 1. Bug Fixes
- Check existing issues for duplicates
- Create an issue if one doesn't exist
- Reference the issue in your PR

#### 2. New Features
- **Discuss first**: Open an issue to propose the feature
- Wait for maintainer feedback before implementing
- Large features may need a design document

#### 3. Documentation
- Fix typos and clarify explanations
- Add examples and use cases
- Improve API documentation

#### 4. Tests
- Increase test coverage
- Add edge case tests
- Fix flaky tests

#### 5. Performance
- Profile before optimizing
- Document benchmark results
- Maintain readability

### Good First Issues

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - Community help needed
- `documentation` - Docs improvements
- `bug` - Bug fixes

---

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with these additions:

```python
# Use type hints
def process_input(text: str, config: Config) -> Response:
    """Process user input through the system.

    Args:
        text: The user's input text.
        config: Configuration settings.

    Returns:
        Response object with the result.
    """
    pass

# Use dataclasses for data containers
@dataclass
class CognitiveState:
    label: str
    entropy: float
    confidence: float

# Use async/await consistently
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Use descriptive variable names
user_message = input_text  # Good
um = it  # Bad

# Maximum line length: 100 characters
# Use Black formatter
```

### TypeScript/React Style

```typescript
// Use functional components with hooks
export function ChatPanel({ messages }: ChatPanelProps) {
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="chat-panel">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
    </div>
  );
}

// Use proper TypeScript types
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

// Use Tailwind CSS for styling
<div className="flex items-center gap-2 p-4 bg-surface rounded-lg">
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code change that neither fixes nor adds
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(medulla): add thermal throttling support

fix(tui): resolve input cursor flickering issue

docs(readme): update installation instructions

refactor(cortex): extract layer paging to separate module
```

---

## Testing Guidelines

### Writing Tests

```python
# tests/unit/test_medulla.py
import pytest
from src.core.medulla import Medulla, MedullaConfig

class TestMedulla:
    """Tests for Medulla component."""

    def test_initialization(self):
        """Test Medulla initializes correctly."""
        config = MedullaConfig(simulation_mode=True)
        medulla = Medulla(config)
        assert medulla.config == config

    @pytest.mark.asyncio
    async def test_process_simple_input(self):
        """Test processing simple input."""
        medulla = Medulla(MedullaConfig(simulation_mode=True))
        result = await medulla.process("Hello")
        assert "response" in result

    def test_surprise_calculation(self):
        """Test surprise score calculation."""
        # Test with known inputs
        assert calculate_surprise("Hello") < 0.3
        assert calculate_surprise("Quantum entanglement") > 0.7
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/unit/test_medulla.py

# With coverage
pytest --cov=src --cov-report=html tests/

# Only fast tests
pytest -m "not slow" tests/

# Parallel execution
pytest -n auto tests/
```

### Test Categories

Mark slow tests:
```python
@pytest.mark.slow
def test_full_cortex_inference():
    """This test takes > 30 seconds."""
    pass
```

Mark integration tests:
```python
@pytest.mark.integration
def test_medulla_cortex_handoff():
    """Requires full system setup."""
    pass
```

---

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Tests
   pytest tests/

   # Linting
   black src/ --check
   flake8 src/

   # Type checking
   mypy src/
   ```

3. **Update documentation** if needed

### Creating the PR

1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a PR on GitHub

3. Fill out the PR template:
   ```markdown
   ## Summary
   Brief description of changes.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement

   ## Testing
   - [ ] All tests pass
   - [ ] Added new tests for changes
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows project style
   - [ ] Documentation updated
   - [ ] No breaking changes

   Fixes #123
   ```

### Review Process

1. **Automated checks** run (tests, linting)
2. **Code review** by maintainer
3. **Address feedback** with additional commits
4. **Approval and merge**

---

## Issue Guidelines

### Bug Reports

Include:
- AVA version (`python -c "import ava; print(ava.__version__)"`)
- Python version (`python --version`)
- Operating system
- GPU model (if applicable)
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

Include:
- Problem you're trying to solve
- Proposed solution
- Alternatives considered
- Use cases

---

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Documentation**: Check `/docs` folder

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md`
- Release notes
- GitHub contributors page

---

## Architecture Decision Records (ADRs)

For significant changes, we use ADRs:

```markdown
# ADR-001: Use Textual for TUI

## Status
Accepted

## Context
We need a TUI framework that supports rich text and async.

## Decision
Use Textual because:
- Rich text support
- Async-first design
- Active development
- Good documentation

## Consequences
- Requires Python 3.8+
- CSS-like styling
- Learning curve for contributors
```

---

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release after merge
5. GitHub Actions builds and publishes

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Thank You!

Every contribution, no matter how small, helps make AVA better. We appreciate your time and effort!

Questions? Open an issue or start a discussion. We're here to help!
