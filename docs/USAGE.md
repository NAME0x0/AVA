# AVA Usage Guide 💡
*Comprehensive guide to running and using AVA (Afsah's Virtual Assistant) on Windows*

This document provides detailed instructions for running AVA components, scripts, and interfaces on Windows. AVA is optimized for NVIDIA RTX A2000 (4GB VRAM) and designed as a local agentic AI daily driver.

**Prerequisites:** Ensure you've completed the setup in [`INSTALLATION.md`](./INSTALLATION.md) before proceeding.

---

## 🚀 Quick Start

### Option 1: Interactive CLI (Recommended for First-Time Users)
```cmd
# Navigate to AVA directory
cd C:\path\to\AVA

# Start interactive CLI
python -m src.interfaces.cli
# OR
python src\interfaces\cli.py
```

### Option 2: Web Interface
```cmd
# Start web interface with embedded server
python -m src.interfaces.web_interface

# Access at: http://localhost:8080
# Open WebUI integration available if Docker is running
```

### Option 3: Main Entry Point (Production)
```cmd
# Start full AVA system
python -m src.main
# OR
python main.py
```

---

## 📋 Core Components & When to Use Them

### 🖥️ User Interfaces

#### **CLI Interface** (`src\interfaces\cli.py`)
**Best for:** Developers, power users, automation, scripting

```cmd
# Interactive mode (default)
python src\interfaces\cli.py

# Single command execution
python src\interfaces\cli.py "What's the weather like today?"

# With specific options
python src\interfaces\cli.py --format json --verbose "help"

# Background execution (Windows)
python src\interfaces\cli.py --quiet "status" > system_status.txt
```

**Key Features:**
- Tab completion and command history
- Rich terminal output (if available)
- Streaming responses
- JSON/markdown/table output formats
- Command piping and automation support

#### **Web Interface** (`src\interfaces\web_interface.py`)
**Best for:** General users, visual interaction, remote access

```cmd
# Start embedded web server
python src\interfaces\web_interface.py

# With custom port
python src\interfaces\web_interface.py --port 3000

# Test mode
python src\interfaces\web_interface.py --test

# Background mode (Windows)
start /B python src\interfaces\web_interface.py > web.log 2>&1
```

**Access Methods:**
- **Local:** `http://localhost:8080` (embedded interface)
- **Open WebUI:** Automatic Docker container management
- **Remote:** Secure tunneling via LocalToNet/ngrok

---

### 🧠 Core System Components

#### **Main Entry Point** (`src\main.py`)
**Best for:** Production deployment, full system startup

```cmd
# Standard startup
python src\main.py

# With configuration file
python src\main.py --config C:\path\to\config.yaml

# Debug mode
python src\main.py --debug --verbose

# Specific interface mode
python src\main.py --interface cli
python src\main.py --interface web
python src\main.py --interface both
```

#### **Assistant Core** (`src\core\assistant.py`)
**Programmatic Usage:**
```python
import asyncio
from src.core.assistant import get_assistant, AssistantRequest

async def use_assistant():
    assistant = await get_assistant()
    
    request = AssistantRequest(
        request_id="test_001",
        session_id="user_session",
        user_input="Explain quantum computing"
    )
    
    response = await assistant.process_request(request)
    print(response.content)
```

---

### 🔧 Utility Scripts

#### **Model Quantization** (`scripts\quantize_model.py`)
**When to use:** First-time setup, model optimization, switching base models

```cmd
# Quantize Gemma 3n 4B model
python scripts\quantize_model.py ^
    --model "google/gemma-2-4b" ^
    --output ".\models\gemma-4b-q4" ^
    --quantization "nf4" ^
    --gpu-memory-limit 3500

# Batch quantization with different methods
python scripts\quantize_model.py ^
    --model "microsoft/DialoGPT-medium" ^
    --output ".\models\dialogpt-q4" ^
    --quantization "int4" ^
    --test-inference

# Verify quantized model
python scripts\quantize_model.py --test-only --model ".\models\gemma-4b-q4"
```

#### **Synthetic Data Generation** (`scripts\generate_synthetic_data.py`)
**When to use:** Fine-tuning preparation, augmenting training data

```cmd
# Generate general conversation data
python scripts\generate_synthetic_data.py ^
    --task "conversation" ^
    --num-samples 1000 ^
    --output ".\data\synthetic_conversations.jsonl"

# Generate function calling data
python scripts\generate_synthetic_data.py ^
    --task "function_calling" ^
    --num-samples 500 ^
    --output ".\data\function_calls.jsonl" ^
    --tools "calculator,web_search,file_operations"

# Generate reasoning data
python scripts\generate_synthetic_data.py ^
    --task "reasoning" ^
    --num-samples 300 ^
    --output ".\data\reasoning_chains.jsonl" ^
    --complexity "intermediate"
```

---

### 🛠️ Tools & Modules

#### **Calculator Tool** (`src\tools\calculator.py`)
```cmd
# Direct tool testing
python src\tools\calculator.py --test

# Via CLI
ava "calculate the compound interest for $1000 at 5% for 3 years"

# Programmatic usage
python -c "from src.tools.calculator import CalculatorTool; calc = CalculatorTool(); result = calc.calculate('2 * (3 + 4) / 2'); print(result)"
```

#### **Web Search Tool** (`src\tools\web_search.py`)
```cmd
# Test web search functionality
python src\tools\web_search.py --test

# Search via CLI
ava "search for latest AI research papers on quantization"

# With specific engines
ava "search --engine duckduckgo 'RTX A2000 performance benchmarks'"
```

#### **Text Generation** (`src\modules\text_generation.py`)
```cmd
# Test text generation capabilities
python src\modules\text_generation.py --test

# Generate with specific parameters
python src\modules\text_generation.py ^
    --prompt "Write a technical blog post about" ^
    --max-tokens 500 ^
    --temperature 0.7 ^
    --top-p 0.9
```

#### **Audio Processing** (`src\modules\audio_processing.py`)
```cmd
# Test audio capabilities
python src\modules\audio_processing.py --test

# Process audio file
python src\modules\audio_processing.py ^
    --input ".\audio\recording.wav" ^
    --output ".\audio\processed.wav" ^
    --noise-reduction ^
    --normalize
```

#### **Speech Recognition** (`src\modules\speech_recognition.py`)
```cmd
# Test speech recognition
python src\modules\speech_recognition.py --test

# Real-time speech recognition
python src\modules\speech_recognition.py --live

# Process audio file
python src\modules\speech_recognition.py ^
    --input ".\audio\speech.wav" ^
    --model "whisper-small" ^
    --language "en"
```

---

## 🔗 Integration Scenarios

### **Scenario 1: Development & Testing**
```cmd
# 1. Start with CLI for quick testing
python src\interfaces\cli.py

# 2. Test individual components
python src\tools\calculator.py --test
python src\modules\text_generation.py --test

# 3. Generate test data
python scripts\generate_synthetic_data.py --task conversation --num-samples 10

# 4. Run system tests (if pytest is installed)
python -m pytest tests\ -v
```

### **Scenario 2: Production Deployment**
```cmd
# 1. Ensure model is quantized
python scripts\quantize_model.py --model "google/gemma-2-4b" --output ".\models\production"

# 2. Start full system
python src\main.py --config production_config.yaml

# 3. Enable remote access (optional)
python src\interfaces\web_interface.py --tunnel localtonet

# 4. Monitor system
curl http://localhost:8080/status
# OR using PowerShell
powershell -Command "Invoke-RestMethod -Uri http://localhost:8080/status"
```

### **Scenario 3: Custom Automation**
```cmd
# 1. Create automation script
echo import asyncio > automation.py
echo from src.core.assistant import get_assistant >> automation.py
echo from src.core.command_handler import get_command_handler >> automation.py
echo. >> automation.py
echo async def daily_summary(): >> automation.py
echo     assistant = await get_assistant() >> automation.py
echo     # Your automation logic here >> automation.py
echo. >> automation.py
echo asyncio.run(daily_summary()) >> automation.py

# 2. Run automation
python automation.py

# 3. Schedule with Windows Task Scheduler
schtasks /create /tn "AVA Daily Summary" /tr "python C:\path\to\AVA\automation.py" /sc daily /st 08:00
```

---

## 🎛️ Command Reference

### **CLI Commands**
```cmd
# System commands
ava status                          # System status
ava help                           # Show help
ava config.get model.backend       # Get configuration
ava config.set interface.web_port 3000  # Set configuration

# Conversation commands
ava "Hello, how are you?"          # Basic chat
ava --stream "Tell me a story"     # Streaming response
ava --format json "What's 2+2?"   # JSON output

# Function/tool commands
ava calculate "15% of 200"         # Calculator
ava search "Python tutorials"      # Web search
ava summarize --file report.txt    # File summarization

# Session management
ava session.list                   # List active sessions
ava session.clear                  # Clear current session
```

### **Web API Endpoints**
```cmd
# Health check
curl http://localhost:8080/health

# System status
curl http://localhost:8080/status

# Chat (POST) - Using curl
curl -X POST http://localhost:8080/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"content\": \"Hello AVA\"}"

# Using PowerShell instead of curl
powershell -Command "Invoke-RestMethod -Uri http://localhost:8080/chat -Method Post -ContentType 'application/json' -Body '{\"content\": \"Hello AVA\"}'"

# Streaming chat
curl "http://localhost:8080/stream/session123?query=Tell me about AI"

# Execute command
curl -X POST http://localhost:8080/command ^
  -H "Content-Type: application/json" ^
  -d "{\"command\": \"status\"}"
```

---

## 🔍 Troubleshooting & Tips

### **Performance Optimization**
```cmd
# Monitor GPU memory usage
nvidia-smi -l 1

# Check system performance (PowerShell)
python -c "from src.core.assistant import get_assistant; import asyncio; async def check(): assistant = await get_assistant(); status = assistant.get_status(); print(f'Memory: {status.memory_usage_mb:.1f}MB'); print(f'GPU: {status.gpu_memory_usage_mb:.1f}MB'); asyncio.run(check())"

# Optimize for lower memory
set AVA_GPU_MEMORY_FRACTION=0.8
python src\main.py --config low_memory_config.yaml
```

### **Common Issues**

#### **Issue: Out of GPU Memory**
```cmd
# Solution 1: Reduce batch size
set AVA_BATCH_SIZE=1

# Solution 2: Use CPU fallback
set AVA_FORCE_CPU=true

# Solution 3: Requantize model
python scripts\quantize_model.py --model current_model --quantization int4
```

#### **Issue: Slow Response Times**
```cmd
# Check model loading
python scripts\quantize_model.py --test-only --model ".\models\current"

# Enable performance monitoring
python src\main.py --debug --profile

# Use lightweight models
python src\main.py --model "small" --fast-mode
```

#### **Issue: Module Import Errors**
```cmd
# Ensure Python path is set
set PYTHONPATH=%PYTHONPATH%;C:\path\to\AVA

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check virtual environment
where python
pip list | findstr /I "torch transformers fastapi"
```

---

## 📊 Monitoring & Maintenance

### **System Health Checks**
```cmd
# Create automated health check script (health_check.bat)
@echo off
echo === AVA Health Check ===
curl -s http://localhost:8080/health
curl -s http://localhost:8080/status
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Run the health check
health_check.bat
```

### **Log Analysis**
```cmd
# View recent logs (PowerShell)
Get-Content -Path logs\ava.log -Tail 50 -Wait

# Search for errors
findstr /I "error" logs\ava.log | more

# Performance metrics
findstr /I "performance timing" logs\ava.log
```

### **Regular Maintenance**
```cmd
# Create weekly maintenance script (maintenance.bat)
@echo off
echo Starting weekly maintenance...

# Clean old logs (older than 7 days)
forfiles /p logs\ /s /m *.log /d -7 /c "cmd /c del @path"

# Clear temporary files
del /Q tmp\*

# Verify models
python scripts\quantize_model.py --verify-models

# Generate usage report
python -c "from src.core.assistant import get_assistant; import asyncio; async def report(): assistant = await get_assistant(); sessions = assistant.list_active_sessions(); print(f'Active sessions: {len(sessions)}'); asyncio.run(report())"

echo Maintenance completed.

# Run the maintenance
maintenance.bat
```

---

## 🎯 Best Practices

### **For Developers**
- Use Command Prompt or PowerShell for running scripts
- Test individual modules before integration
- Monitor GPU memory usage during development using `nvidia-smi`
- Use verbose logging for debugging: `--debug --verbose`
- Set up Windows environment variables for configuration

### **For Production**
- Start with web interface for stability
- Enable remote access only when needed
- Monitor system resources using Windows Performance Monitor
- Set up Windows Task Scheduler for automated health checks
- Use Windows Services for long-running deployments

### **For Performance**
- Focus on agentic tasks rather than general knowledge queries
- Use streaming responses for better UX
- Leverage function calling for external data
- Optimize prompts for the 4GB VRAM constraint
- Use Windows-specific performance tools

---

## 🪟 Windows-Specific Features

### **Environment Variables**
```cmd
# Set AVA environment variables
set AVA_CONFIG_PATH=C:\path\to\config.yaml
set AVA_LOG_LEVEL=INFO
set AVA_GPU_MEMORY_LIMIT=3500

# Make permanent (requires admin)
setx AVA_CONFIG_PATH "C:\path\to\config.yaml" /M
```

### **Windows Service Setup** (Advanced)
```cmd
# Create a Windows service for AVA (requires admin)
sc create "AVA Assistant" binPath= "python C:\path\to\AVA\src\main.py --service" start= auto
sc description "AVA Assistant" "Local Agentic AI Assistant"

# Start the service
sc start "AVA Assistant"

# Check service status
sc query "AVA Assistant"
```

### **PowerShell Integration**
```powershell
# PowerShell function for AVA interaction
function Invoke-AVA {
    param([string]$Query)
    $response = Invoke-RestMethod -Uri "http://localhost:8080/chat" -Method Post -ContentType "application/json" -Body "{`"content`": `"$Query`"}"
    return $response.response
}

# Usage
Invoke-AVA "What's the weather like?"
```

---

**Next Steps:** 
- See [`DEVELOPMENT.md`](./DEVELOPMENT.md) for contributing guidelines
- Check [`ROADMAP.md`](./ROADMAP.md) for upcoming features
- Visit [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system design details

Based on the [Python Windows FAQ](https://docs.python.org/3/faq/windows.html), AVA is designed to work seamlessly on Windows with proper Python installation and configuration. Make sure your Python environment is correctly set up as described in the installation guide! 🪟🚀
