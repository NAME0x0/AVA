"""Helpers for running Gemma 4 practical branches through llama.cpp."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LLAMA_SERVER_CANDIDATES = [
    Path.home() / ".docker" / "bin" / "inference" / "llama-server.exe",
    Path.home() / ".docker" / "bin" / "inference" / "com.docker.llama-server.exe",
]

DEFAULT_GGUF_REPOS = {
    "google/gemma-4-E2B": "ggml-org/gemma-4-E2B-GGUF",
    "google/gemma-4-E2B-it": "ggml-org/gemma-4-E2B-it-GGUF",
    "google/gemma-4-E4B": "ggml-org/gemma-4-E4B-GGUF",
    "google/gemma-4-E4B-it": "ggml-org/gemma-4-E4B-it-GGUF",
}

DEFAULT_GGUF_QUANTS = {
    "google/gemma-4-E2B": "Q8_0",
    "google/gemma-4-E2B-it": "Q8_0",
    "google/gemma-4-E4B": "Q4_K_M",
    "google/gemma-4-E4B-it": "Q4_K_M",
}


@dataclass(slots=True)
class LlamaCppServerConfig:
    """Settings for a local llama.cpp server instance."""

    model_id: str
    executable: str | None = None
    model_path: str | None = None
    hf_repo: str | None = None
    hf_file: str | None = None
    hf_token: str | None = None
    gguf_quant: str | None = None
    host: str = "127.0.0.1"
    port: int = 0
    ctx_size: int = 262_144
    gpu_layers: str | int = -1
    flash_attn: str = "on"
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    reasoning: str = "auto"
    reasoning_format: str = "none"
    reasoning_budget: int = 0
    chat_template: str | None = None
    threads: int | None = None
    threads_batch: int | None = None
    alias: str | None = None
    offline: bool = False


def default_llama_hf_repo(model_id: str, quant: str | None = None) -> str | None:
    """Map a Gemma 4 Hugging Face repo ID to the official GGUF repo."""
    repo = DEFAULT_GGUF_REPOS.get(model_id)
    if repo is None:
        return None
    if quant is None:
        quant = DEFAULT_GGUF_QUANTS.get(model_id, "Q4_K_M")
    return f"{repo}:{quant}"


def resolve_llama_server_executable(explicit: str | None = None) -> str:
    """Locate a usable llama-server executable."""
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit))

    for env_name in ("LLAMA_CPP_SERVER", "LLAMA_SERVER"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))

    for binary in ("llama-server.exe", "llama-server"):
        resolved = shutil.which(binary)
        if resolved:
            candidates.append(Path(resolved))

    candidates.extend(DEFAULT_LLAMA_SERVER_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not locate llama-server. Pass --llama-server-exe or install a local llama.cpp server binary."
    )


def build_llama_server_command(config: LlamaCppServerConfig) -> list[str]:
    """Build the llama-server command line."""
    executable = resolve_llama_server_executable(config.executable)
    port = config.port or _find_free_port()
    command = [
        executable,
        "--host",
        config.host,
        "--port",
        str(port),
        "--ctx-size",
        str(config.ctx_size),
        "--flash-attn",
        config.flash_attn,
        "--cache-type-k",
        config.cache_type_k,
        "--cache-type-v",
        config.cache_type_v,
        "--reasoning",
        config.reasoning,
        "--reasoning-format",
        config.reasoning_format,
        "--reasoning-budget",
        str(config.reasoning_budget),
        "--no-webui",
        "--jinja",
        "--parallel",
        "1",
        "--slot-prompt-similarity",
        "0.10",
    ]

    if config.threads is not None:
        command.extend(["--threads", str(config.threads)])
    if config.threads_batch is not None:
        command.extend(["--threads-batch", str(config.threads_batch)])

    if config.model_path:
        command.extend(["--model", config.model_path])
    else:
        hf_repo = config.hf_repo or default_llama_hf_repo(config.model_id, config.gguf_quant)
        if hf_repo is None:
            raise ValueError(
                "No GGUF source specified. Pass --llama-model or --llama-hf-repo for this backend."
            )
        command.extend(["--hf-repo", hf_repo])
        if config.hf_file:
            command.extend(["--hf-file", config.hf_file])
        if config.hf_token:
            command.extend(["--hf-token", config.hf_token])
        if config.offline:
            command.append("--offline")

    if config.alias:
        command.extend(["--alias", config.alias])
    if config.chat_template:
        command.extend(["--chat-template", config.chat_template])

    if config.gpu_layers is not None:
        command.extend(["--gpu-layers", str(config.gpu_layers)])

    return command


def extract_chat_metrics(payload: dict[str, Any]) -> dict[str, int]:
    """Extract prompt/completion token counts from llama.cpp responses."""
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    if prompt_tokens is None:
        prompt_tokens = payload.get("tokens_evaluated")
    if completion_tokens is None:
        completion_tokens = payload.get("tokens_predicted")

    timings = payload.get("timings") or {}
    if prompt_tokens is None:
        prompt_tokens = timings.get("prompt_n")
    if completion_tokens is None:
        completion_tokens = timings.get("predicted_n")

    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def benchmark_llamacpp_chat_speed(
    server: "LlamaCppServer",
    prompts: list[str],
    *,
    max_new_tokens: int,
    enable_thinking: bool,
) -> dict[str, float]:
    """Measure chat throughput through a live llama.cpp server."""
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        result = server.generate_chat_response(
            prompt,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_time += result["elapsed_s"]

    total_tokens = total_input_tokens + total_output_tokens
    return {
        "total_tokens": total_tokens,
        "output_tokens": total_output_tokens,
        "total_time_s": total_time,
        "tok_per_s": total_tokens / total_time if total_time > 0 else 0.0,
        "decode_tok_per_s": total_output_tokens / total_time if total_time > 0 else 0.0,
        "avg_latency_s": total_time / len(prompts) if prompts else 0.0,
    }


class LlamaCppServer:
    """Thin lifecycle wrapper for a local llama.cpp server."""

    def __init__(
        self,
        config: LlamaCppServerConfig,
        *,
        log_path: Path,
        startup_timeout_s: float = 180.0,
    ) -> None:
        self.config = config
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: Any | None = None
        self.command: list[str] = []
        self.port = config.port or _find_free_port()
        self.startup_elapsed_s = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.port}"

    def start(self) -> dict[str, Any]:
        """Start the server and wait until it responds."""
        t0 = time.perf_counter()
        self.config.port = self.port
        self.command = build_llama_server_command(self.config)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path.cwd()),
            creationflags=creationflags,
        )

        deadline = time.perf_counter() + self.startup_timeout_s
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                break
            if _server_ready(self.base_url):
                self.startup_elapsed_s = time.perf_counter() - t0
                return {
                    "command": self.command,
                    "port": self.port,
                    "log_path": str(self.log_path),
                    "load_time_s": round(self.startup_elapsed_s, 1),
                }
            time.sleep(0.5)

        self.stop()
        raise RuntimeError(
            "llama.cpp server did not become ready. "
            f"Last log lines:\n{_tail_text(self.log_path, 40)}"
        )

    def stop(self) -> None:
        """Terminate the server process if it is still running."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def generate_chat_response(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        """Send a single chat completion request."""
        del enable_thinking  # thinking is configured server-side for now
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "stream": False,
        }

        t0 = time.perf_counter()
        response = _http_json(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            payload=payload,
            timeout=600,
        )
        elapsed = time.perf_counter() - t0

        message = ((response.get("choices") or [{}])[0]).get("message") or {}
        content = _extract_message_content(message)
        metrics = extract_chat_metrics(response)
        return {
            "response": content.strip(),
            "input_tokens": metrics["prompt_tokens"],
            "output_tokens": metrics["completion_tokens"],
            "elapsed_s": elapsed,
            "raw_response": response,
        }

    def __enter__(self) -> "LlamaCppServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.stop()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _server_ready(base_url: str) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            _http_json("GET", f"{base_url}{path}", timeout=5)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            continue
    return False


def _tail_text(path: Path, lines: int) -> str:
    if not path.exists():
        return "<log file missing>"
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _extract_message_content(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        text_parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text_parts.append(str(chunk.get("text", "")))
            elif isinstance(chunk, str):
                text_parts.append(chunk)
        content = "".join(text_parts)

    reasoning_content = message.get("reasoning_content") or ""
    text = str(content)

    for marker in ("<|channel|>assistant", "<|channel>assistant"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]

    for marker in ("<|channel|>thought", "<|channel>thought"):
        if marker in text:
            text = text.split(marker, 1)[0]

    for marker in ("<think>", "</think>", "<|thought|>", "</|thought|>"):
        text = text.replace(marker, "")

    if reasoning_content and text.startswith(str(reasoning_content)):
        text = text[len(str(reasoning_content)) :]

    return text
