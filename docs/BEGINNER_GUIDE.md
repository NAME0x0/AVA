# AVA Beginner's Guide

Welcome to AVA! This guide will help you understand what AVA is, how to set it up, and how to start using it - even if you've never worked with AI projects before.

---

## What is AVA?

AVA is a **personal AI assistant** that runs on your own computer. Think of it like having a helpful assistant that:

- Answers your questions
- Helps you think through complex problems
- Searches the web for information
- Remembers your conversations

**What makes AVA special?**

Unlike cloud-based AI assistants, AVA runs entirely on your computer. This means:
- Your conversations stay private
- It works offline (mostly)
- You have full control over how it behaves

---

## What You'll Need

Before we start, make sure you have:

| Requirement | What it means | How to check |
|-------------|---------------|--------------|
| **Python 3.9+** | A programming language that runs AVA | Open terminal/command prompt, type `python --version` |
| **4GB+ RAM** | Computer memory | Check your system settings |
| **~10GB disk space** | Storage for AI models | Check your available disk space |
| **Internet connection** | For initial setup and web search | Just make sure you're online |

**Optional but recommended:**
- NVIDIA GPU with 4GB+ VRAM (makes AVA faster)
- Git (for updates)

---

## Quick Setup (The Easy Way)

### Step 1: Download AVA

**Option A: Download ZIP**
1. Go to the AVA GitHub page
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file to a folder you'll remember

**Option B: Use Git (if you have it)**
```bash
git clone https://github.com/yourusername/AVA.git
cd AVA
```

### Step 2: Run the Setup Script

Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and navigate to the AVA folder:

```bash
cd path/to/AVA
python setup_ava.py
```

**What happens:**
- Creates a virtual environment (keeps AVA's files separate from your system)
- Installs required software packages
- Downloads AI models (this may take a few minutes)
- Verifies everything works

You'll see a progress bar like this:
```
[Step 3/7] Installing Dependencies
  [████████████░░░░░░░░] 60%  Installing httpx
```

### Step 3: Start AVA

After setup completes, start AVA:

```bash
python server.py
```

You'll see:
```
AVA server running at http://localhost:8085
```

### Step 4: Chat with AVA

Open your web browser and go to: `http://localhost:8085`

Or use the terminal interface:
```bash
python run_tui.py
```

---

## Understanding AVA's Brain

AVA has a unique "two-brain" system inspired by how humans think:

### The Medulla (Fast Brain)
- Handles simple questions instantly
- "What time is it?" - answered in milliseconds
- Always running, uses minimal resources

### The Cortex (Deep Brain)
- Handles complex questions that need careful thinking
- "Explain quantum computing" - takes longer but gives better answers
- Only activates when needed

**You don't need to do anything** - AVA automatically decides which brain to use!

---

## Common Commands

### In the Terminal UI (TUI)

| Key | What it does |
|-----|-------------|
| `Type + Enter` | Send a message |
| `Ctrl+K` | Open command palette |
| `Ctrl+L` | Clear chat history |
| `Ctrl+T` | Show/hide metrics |
| `Ctrl+4` | Show/hide settings |
| `Ctrl+5` | Show/hide tools |
| `F1` | Show help |
| `Ctrl+Q` | Quit |

### Special Prefixes

| Prefix | What it does | Example |
|--------|-------------|---------|
| `/search` | Force web search | `/search latest news` |
| `/think` | Force deep thinking | `/think explain relativity` |
| `/help` | Show help | `/help` |

---

## Troubleshooting

### "Python not found"
- Make sure Python is installed
- On Windows, you may need to use `python3` instead of `python`
- Check your system PATH includes Python

### "Ollama not found"
- Install Ollama from https://ollama.ai
- After installing, run `ollama serve` in a separate terminal

### "Connection refused"
- Make sure the AVA server is running (`python server.py`)
- Check if another program is using port 8085
- Try: `python server.py --port 8086`

### "Out of memory"
- Close other programs to free up RAM
- Use the `--minimal` flag: `python setup_ava.py --minimal`
- This uses smaller AI models

### Models won't download
- Check your internet connection
- Try running Ollama manually: `ollama pull gemma3:4b`
- Check if you have enough disk space

---

## Next Steps

Once you're comfortable with the basics:

1. **Explore Settings** - Press `Ctrl+4` in TUI to customize behavior
2. **Try Different Modes** - Use `/search` and `/think` prefixes
3. **Read the Architecture Docs** - Learn how AVA works internally
4. **Customize Config** - Edit `config/cortex_medulla.yaml`

---

## Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: Check the `docs/` folder for more guides
- **README**: Quick reference for common operations

---

## Glossary

| Term | Meaning |
|------|---------|
| **Virtual Environment (venv)** | A separate space for AVA's files that doesn't affect your system |
| **Ollama** | Software that runs AI models on your computer |
| **Terminal/Command Prompt** | Text-based interface for running commands |
| **TUI** | Terminal User Interface - AVA's text-based graphical interface |
| **GPU** | Graphics Processing Unit - makes AI faster |
| **VRAM** | Video RAM - memory on your graphics card |
| **Model** | The AI "brain" that processes your questions |

---

*Welcome to AVA! We hope you enjoy exploring AI on your own terms.*
