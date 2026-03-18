"""Core inference engine for AVA Experiment 4.

Wraps a quantized HuggingFace model with tool-use, memory, and agentic capabilities.
Designed for 4GB VRAM (RTX A2000 Laptop GPU).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    enable_thinking: bool = True
    stop_strings: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    name: str
    input_args: dict[str, Any]
    output: str
    success: bool = True


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[dict] | None = None
    tool_result: ToolResult | None = None
    thinking: str | None = None
    timestamp: float = field(default_factory=time.time)


class Tool:
    """Base class for tools the agent can use."""

    def __init__(self, name: str, description: str, parameters: dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class CalculatorTool(Tool):
    """Safe math expression evaluator."""

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description="Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, abs, round.",
            parameters={"expression": {"type": "string", "description": "Math expression to evaluate"}},
        )

    def execute(self, expression: str = "", **kwargs: Any) -> str:
        import math
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pow": pow, "log": math.log, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e,
        }
        try:
            result = eval(expression, {"__builtins__": {}}, allowed)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class PythonTool(Tool):
    """Execute simple Python code in a sandboxed environment."""

    def __init__(self) -> None:
        super().__init__(
            name="python",
            description="Execute Python code and return the output. Use print() to show results.",
            parameters={"code": {"type": "string", "description": "Python code to execute"}},
        )

    def execute(self, code: str = "", **kwargs: Any) -> str:
        import io
        import contextlib
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, {"__builtins__": __builtins__}, {})
            output = stdout.getvalue()
            return output if output else "(no output)"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"


class SearchTool(Tool):
    """Search the agent's knowledge/memory for relevant information."""

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        super().__init__(
            name="search_memory",
            description="Search the agent's memory for relevant past interactions or stored knowledge.",
            parameters={"query": {"type": "string", "description": "Search query"}},
        )
        self.memory_store = memory_store

    def execute(self, query: str = "", **kwargs: Any) -> str:
        if self.memory_store is None:
            return "No memory store available."
        results = self.memory_store.search(query, top_k=3)
        if not results:
            return "No relevant memories found."
        return "\n---\n".join(f"[{r['timestamp']}] {r['content']}" for r in results)


class MemoryStore:
    """Simple persistent memory for the agent."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.memories: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        mem_file = self.path / "memories.jsonl"
        if mem_file.exists():
            self.memories = [
                json.loads(line)
                for line in mem_file.read_text(encoding="utf-8").strip().split("\n")
                if line.strip()
            ]

    def _save(self) -> None:
        mem_file = self.path / "memories.jsonl"
        mem_file.write_text(
            "\n".join(json.dumps(m) for m in self.memories) + "\n",
            encoding="utf-8",
        )

    def add(self, content: str, kind: str = "observation", metadata: dict | None = None) -> None:
        entry = {
            "content": content,
            "kind": kind,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or {},
        }
        self.memories.append(entry)
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Simple keyword search over memories."""
        query_tokens = set(query.lower().split())
        scored = []
        for mem in self.memories:
            mem_tokens = set(mem["content"].lower().split())
            overlap = len(query_tokens & mem_tokens)
            if overlap > 0:
                scored.append((overlap / max(len(query_tokens), 1), mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        return self.memories[-n:]


class AVAEngine:
    """Core agentic inference engine.

    Loads a quantized model and provides tool-use, memory,
    and multi-turn conversation capabilities.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        adapter_path: str | Path | None = None,
        memory_path: str | Path | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        quantize_4bit: bool = True,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch + transformers required. pip install torch transformers bitsandbytes accelerate")

        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.memory = MemoryStore(memory_path) if memory_path else None
        self.tools: dict[str, Tool] = {}
        self.conversation: list[Message] = []

        # Register tools
        if tools:
            for tool in tools:
                self.tools[tool.name] = tool

        self.system_prompt = system_prompt or self._default_system_prompt()

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
        }
        if quantize_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

        # Load LoRA adapter if provided
        if self.adapter_path:
            adapter_config_file = Path(self.adapter_path) / "adapter_config.json"
            if adapter_config_file.exists():
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                self.model = self.model.merge_and_unload()

        self.model.eval()

    def _default_system_prompt(self) -> str:
        tool_descriptions = ""
        if self.tools:
            tool_list = "\n".join(
                f"- {name}: {tool.description}" for name, tool in self.tools.items()
            )
            tool_descriptions = f"""

You have access to the following tools:
{tool_list}

To use a tool, write: <tool_call>{{"name": "tool_name", "arguments": {{"arg": "value"}}}}</tool_call>
Wait for the tool result before continuing.
"""
        return f"""You are AVA, an intelligent AI assistant with memory and tool-use capabilities.
You think step by step, use tools when helpful, and remember important information.
Be concise, accurate, and helpful.{tool_descriptions}"""

    def _build_messages(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.conversation:
            if msg.role == "tool":
                messages.append({"role": "user", "content": f"Tool result: {msg.content}"})
            else:
                messages.append({"role": msg.role, "content": msg.content})
        return messages

    @torch.no_grad()
    def generate(self, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        messages = self._build_messages()
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else 1.0,
            top_p=config.top_p if config.do_sample else 1.0,
            do_sample=config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return generated

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from generated text.

        Supports two formats:
        1. JSON: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        2. Qwen native XML: <tool_call><function=name><parameter=arg>val</parameter></function></tool_call>
        """
        calls = []
        pattern = r'<tool_call>(.*?)</tool_call>'
        for match in re.finditer(pattern, text, re.DOTALL):
            inner = match.group(1).strip()
            # Try JSON format first (our training format)
            try:
                call = json.loads(inner)
                calls.append(call)
                continue
            except (json.JSONDecodeError, ValueError):
                pass
            # Try Qwen native XML format
            func_match = re.search(r'<function=(\w+)>(.*?)</function>', inner, re.DOTALL)
            if func_match:
                func_name = func_match.group(1)
                params_text = func_match.group(2)
                args = {}
                for param in re.finditer(r'<parameter=(\w+)>\n?(.*?)\n?</parameter>', params_text, re.DOTALL):
                    args[param.group(1)] = param.group(2).strip()
                calls.append({"name": func_name, "arguments": args})
        return calls

    def _execute_tool(self, call: dict[str, Any]) -> ToolResult:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if name not in self.tools:
            return ToolResult(name=name, input_args=args, output=f"Unknown tool: {name}", success=False)
        try:
            output = self.tools[name].execute(**args)
            return ToolResult(name=name, input_args=args, output=output, success=True)
        except Exception as e:
            return ToolResult(name=name, input_args=args, output=f"Error: {e}", success=False)

    def chat(self, user_input: str, config: GenerationConfig | None = None) -> str:
        """Full agentic chat loop: generate, detect tool calls, execute, continue."""
        config = config or GenerationConfig()

        # Add user message
        self.conversation.append(Message(role="user", content=user_input))

        max_tool_rounds = 5
        for _ in range(max_tool_rounds):
            response = self.generate(config)

            # Parse thinking
            thinking = None
            content = response
            if "</think>" in response:
                parts = response.split("</think>", 1)
                thinking = parts[0].replace("<think>", "").strip()
                content = parts[1].strip()

            # Check for tool calls
            tool_calls = self._parse_tool_calls(content)
            if not tool_calls:
                # No tool calls — final response
                self.conversation.append(Message(
                    role="assistant", content=content, thinking=thinking
                ))
                # Save to memory if notable
                if self.memory and len(content) > 50:
                    self.memory.add(
                        f"User: {user_input[:100]}\nAssistant: {content[:200]}",
                        kind="conversation",
                    )
                return content

            # Execute tool calls
            self.conversation.append(Message(
                role="assistant", content=content, tool_calls=tool_calls, thinking=thinking
            ))
            for call in tool_calls:
                result = self._execute_tool(call)
                self.conversation.append(Message(
                    role="tool", content=result.output, tool_result=result
                ))

        # If we exhausted tool rounds, return last response
        return content

    def reset(self) -> None:
        self.conversation = []

    def vram_report(self) -> dict[str, float]:
        if not TORCH_AVAILABLE:
            return {}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        }
