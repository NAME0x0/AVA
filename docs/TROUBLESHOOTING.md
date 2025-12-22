# AVA Troubleshooting Guide

Solutions to common problems when running AVA.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Connection Issues](#connection-issues)
- [Performance Issues](#performance-issues)
- [GPU Issues](#gpu-issues)
- [UI Issues](#ui-issues)
- [Error Messages](#error-messages)

---

## Installation Issues

### "ModuleNotFoundError: No module named 'xyz'"

**Cause**: Missing Python dependency.

**Solution**:
```bash
pip install -r requirements.txt
```

If a specific package is missing:
```bash
pip install package_name
```

### "pip: command not found"

**Solution**: Use `python -m pip` instead:
```bash
python -m pip install -r requirements.txt
```

### "Python version too old"

**Cause**: AVA requires Python 3.10+.

**Solution**:
1. Download Python 3.10+ from https://python.org
2. Recreate the virtual environment:
```bash
python3.10 -m venv venv
venv\Scripts\activate  # or source venv/bin/activate
pip install -r requirements.txt
```

### "Permission denied" on Windows

**Solution**: Run PowerShell as Administrator, or:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Connection Issues

### "Ollama connection refused" / "Cannot connect to localhost:11434"

**Cause**: Ollama service isn't running.

**Solution**:
```bash
# Start Ollama in a new terminal
ollama serve
```

**Verify**:
```bash
curl http://localhost:11434/api/tags
```

### "No models available"

**Cause**: No Ollama models downloaded.

**Solution**:
```bash
ollama pull gemma3:4b
```

List available models:
```bash
ollama list
```

### "AVA server won't start" / "Port 8085 in use"

**Cause**: Another process is using port 8085.

**Solution 1**: Use a different port:
```bash
python server.py --port 8086
```

**Solution 2**: Find and kill the process:
```bash
# Windows
netstat -ano | findstr :8085
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8085
kill -9 <PID>
```

### "CORS error in browser"

**Cause**: Browser blocking cross-origin requests.

**Solution**: AVA includes CORS headers by default. If you're running the UI separately:
```bash
# Start server with explicit host
python server.py --host 0.0.0.0
```

---

## Performance Issues

### "Responses are very slow"

**Possible Causes**:

1. **Model loading**: First response is always slower (model loads into GPU)
   - *Solution*: Wait for "Model loaded" in logs

2. **Large model**: Using a model too big for your GPU
   - *Solution*: Use smaller model:
   ```bash
   ollama pull phi3:mini
   ```

3. **CPU inference**: Running without GPU
   - *Solution*: Check GPU detection (see GPU Issues below)

4. **Thermal throttling**: GPU is overheating
   - *Solution*: Check temperature in status bar, improve cooling

### "Out of memory"

**Cause**: Model too large for available RAM/VRAM.

**Solutions**:
1. Use smaller model:
   ```bash
   ollama pull phi3:mini  # ~2GB
   ```

2. Close other applications using GPU memory

3. Reduce context length in config:
   ```yaml
   # config/cortex_medulla.yaml
   medulla:
     max_context_length: 2048  # Reduce from 4096
   ```

### "High CPU usage even when idle"

**Cause**: Polling or background processes.

**Solution**: Enable simulation mode for testing:
```bash
python server.py --simulation
```

---

## GPU Issues

### "CUDA not available" / "GPU not detected"

**Check GPU status**:
```bash
nvidia-smi
```

**If nvidia-smi fails**:
1. Install/update NVIDIA drivers from https://nvidia.com/drivers
2. Restart your computer

**If nvidia-smi works but CUDA unavailable**:
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "CUDA out of memory"

**Solutions**:
1. Use a smaller model
2. Close other GPU applications (games, other AI tools)
3. Enable layer-wise paging (reduces memory usage):
   ```yaml
   # config/cortex_medulla.yaml
   cortex:
     layer_paging: true
   ```

### "GPU temperature too high" / Thermal throttling

**Check temperature**:
```bash
nvidia-smi --query-gpu=temperature.gpu --format=csv
```

**Solutions**:
1. Improve case airflow
2. Clean GPU fans
3. Lower power limit in config:
   ```yaml
   thermal:
     max_gpu_power_percent: 10  # Reduce from 15
   ```

---

## UI Issues

### TUI displays garbled characters

**Cause**: Terminal doesn't support Unicode.

**Solutions**:
1. Use a modern terminal (Windows Terminal, iTerm2, Kitty)
2. Set terminal encoding:
   ```bash
   export LANG=en_US.UTF-8
   ```

### TUI colors look wrong

**Cause**: Terminal not configured for 256 colors.

**Solution**:
```bash
export TERM=xterm-256color
```

### GUI won't load / blank screen

**Cause**: JavaScript error or API connection issue.

**Solutions**:
1. Check browser console for errors (F12)
2. Verify API is running: http://localhost:8085/health
3. Clear browser cache and reload
4. Try a different browser

### "WebGL not supported"

**Cause**: 3D visualization requires WebGL.

**Solutions**:
1. Update your graphics drivers
2. Enable hardware acceleration in browser settings
3. Use a different browser (Chrome/Firefox recommended)

---

## Error Messages

### "I encountered an error while thinking"

**Cause**: Cortex (deep thinking) failed.

**Solutions**:
1. Try a simpler question
2. Check Ollama is still running
3. Check GPU memory usage

### "Request timed out"

**Cause**: Response took too long.

**Solutions**:
1. Ask a simpler question
2. Use Medulla (fast) instead of forcing Cortex
3. Increase timeout in config:
   ```yaml
   cortex:
     max_generation_time: 600  # seconds
   ```

### "Tool execution failed"

**Cause**: External tool (web search, calculator) failed.

**Solutions**:
1. Check internet connection
2. Try again (temporary failure)
3. Disable the problematic tool

### "Config not found"

**Cause**: Missing configuration file.

**Solution**:
```bash
# Copy example config
cp config/ava.yaml.example config/ava.yaml
```

---

## Getting More Help

### Enable Debug Logging

```bash
LOG_LEVEL=DEBUG python server.py
```

### Check System Status

```bash
python scripts/verify_install.py
```

### Report a Bug

1. Collect information:
   - Python version: `python --version`
   - OS version
   - GPU: `nvidia-smi`
   - Error message and stack trace

2. Create an issue: https://github.com/NAME0x0/AVA/issues

### Community Support

- GitHub Discussions: https://github.com/NAME0x0/AVA/discussions
- Check existing issues for similar problems

---

*Still stuck? Create a GitHub issue with your error details.*
