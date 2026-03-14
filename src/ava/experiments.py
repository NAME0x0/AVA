from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from ava.config import ExperimentConfig, load_experiment_config
from ava.eval import BenchmarkTask, default_benchmark
from ava.memory import TitansMemory
from ava.tokenizer import ByteTokenizer
from ava.tools import list_protocol_names, render_tool_trace


@dataclass(frozen=True, slots=True)
class BudgetEstimate:
    name: str
    parameters: int
    train_vram_gb: float
    infer_vram_gb: float
    tokens_per_optimizer_step: int
    fits_4gb: bool


@dataclass(frozen=True, slots=True)
class ProtocolEstimate:
    protocol: str
    average_tokens: float
    total_tokens: int


@dataclass(frozen=True, slots=True)
class MemorySweepEstimate:
    threshold: float
    recall_at_1: float
    average_context_tokens: float


@dataclass(frozen=True, slots=True)
class TestTimeStrategyEstimate:
    strategy: str
    average_extra_tokens: float
    total_extra_tokens: int
    hard_task_coverage: float


MEMORY_CASES = (
    {
        "query": "Which tool should I use for arithmetic?",
        "expected": "calculator",
        "records": (
            ("Use the calculator tool for arithmetic and quick numeric checks.", 0.55),
            ("General English style preference: concise and direct.", 0.22),
            ("Store rarely used biography facts only when relevant.", 0.18),
        ),
    },
    {
        "query": "How do I handle square roots?",
        "expected": "square roots",
        "records": (
            ("The calculator can evaluate square roots and powers exactly for small problems.", 0.52),
            ("Use external memory to retain long-horizon preferences.", 0.33),
            ("Avoid long answers unless the user asks for detail.", 0.27),
        ),
    },
    {
        "query": "What helps with multi-step algebra?",
        "expected": "algebra",
        "records": (
            ("Do algebraic rearrangement before tool calls when the expression is symbolic.", 0.61),
            ("Summaries should stay compact.", 0.24),
            ("Favor exact arithmetic when possible.", 0.44),
        ),
    },
    {
        "query": "How should AVA remember user preferences?",
        "expected": "preferences",
        "records": (
            ("Write user preferences into external memory only when they are stable and surprising enough.", 0.58),
            ("Do not overuse memory for transient requests.", 0.41),
            ("The calculator is for math only.", 0.2),
        ),
    },
)


def _dtype_bytes(dtype: str) -> int:
    if dtype in {"float16", "bfloat16"}:
        return 2
    return 4


def estimate_budget(config: ExperimentConfig, vram_limit_gb: float = 3.4) -> BudgetEstimate:
    parameters = config.model.estimated_parameters(config.tokenizer.vocab_size)
    dtype_bytes = _dtype_bytes(config.training.dtype)
    weights = parameters * dtype_bytes
    grads = parameters * dtype_bytes
    optimizer = parameters * 8
    activations = (
        config.training.micro_batch_size
        * config.model.block_size
        * config.model.n_embd
        * config.model.n_layer
        * dtype_bytes
        * 28
    )
    attention_overhead = (
        config.training.micro_batch_size
        * config.model.block_size
        * config.model.block_size
        * config.model.n_head
        * dtype_bytes
    )
    runtime_margin = 768 * 1024 * 1024
    train_total = weights + grads + optimizer + activations + attention_overhead + runtime_margin
    infer_total = weights + (activations // 8) + runtime_margin
    train_vram_gb = train_total / (1024**3)
    infer_vram_gb = infer_total / (1024**3)
    tokens_per_optimizer_step = (
        config.training.micro_batch_size
        * config.training.gradient_accumulation_steps
        * config.model.block_size
    )
    return BudgetEstimate(
        name=config.name,
        parameters=parameters,
        train_vram_gb=round(train_vram_gb, 3),
        infer_vram_gb=round(infer_vram_gb, 3),
        tokens_per_optimizer_step=tokens_per_optimizer_step,
        fits_4gb=train_vram_gb <= vram_limit_gb,
    )


def run_budget_sweep(config_paths: list[str | Path]) -> list[BudgetEstimate]:
    configs = [load_experiment_config(path) for path in config_paths]
    with ThreadPoolExecutor(max_workers=min(4, len(configs) or 1)) as pool:
        results = list(pool.map(estimate_budget, configs))
    return sorted(results, key=lambda item: item.parameters)


def estimate_prompt_protocol(protocol: str, tokenizer: ByteTokenizer) -> ProtocolEstimate:
    total_tokens = 0
    tool_tasks = [task for task in default_benchmark() if task.category in {"math", "tool"}]
    for task in tool_tasks:
        expression = task.prompt.split("for ", 1)[-1] if task.category == "tool" else task.expected
        rendered = render_tool_trace(protocol, expression, task.expected)
        total_tokens += tokenizer.count_tokens(rendered, add_bos=True, add_eos=True)
    average_tokens = total_tokens / max(len(tool_tasks), 1)
    return ProtocolEstimate(
        protocol=protocol,
        average_tokens=round(average_tokens, 2),
        total_tokens=total_tokens,
    )


def run_prompt_protocol_sweep(tokenizer: ByteTokenizer | None = None) -> list[ProtocolEstimate]:
    tokenizer = tokenizer or ByteTokenizer()
    protocols = list_protocol_names()
    with ThreadPoolExecutor(max_workers=min(4, len(protocols) or 1)) as pool:
        results = list(pool.map(lambda name: estimate_prompt_protocol(name, tokenizer), protocols))
    return sorted(results, key=lambda item: item.average_tokens)


def estimate_memory_threshold(threshold: float, tokenizer: ByteTokenizer) -> MemorySweepEstimate:
    correct = 0
    total_context_tokens = 0
    for case in MEMORY_CASES:
        memory = TitansMemory(max_items=16, write_surprise_threshold=threshold)
        for text, surprise in case["records"]:
            memory.write(text, surprise=surprise)
        hits = memory.retrieve(case["query"], top_k=1)
        if hits and case["expected"] in hits[0].text.lower():
            correct += 1
        context = memory.summarize_context(case["query"], top_k=2)
        total_context_tokens += tokenizer.count_tokens(context, add_bos=True, add_eos=True)
    recall = correct / len(MEMORY_CASES)
    average_context_tokens = total_context_tokens / len(MEMORY_CASES)
    return MemorySweepEstimate(
        threshold=threshold,
        recall_at_1=round(recall, 2),
        average_context_tokens=round(average_context_tokens, 2),
    )


def run_memory_sweep(tokenizer: ByteTokenizer | None = None) -> list[MemorySweepEstimate]:
    tokenizer = tokenizer or ByteTokenizer()
    thresholds = (0.25, 0.45, 0.65)
    with ThreadPoolExecutor(max_workers=len(thresholds)) as pool:
        results = list(pool.map(lambda threshold: estimate_memory_threshold(threshold, tokenizer), thresholds))
    return sorted(results, key=lambda item: (-(item.recall_at_1), item.average_context_tokens))


def _is_hard_math_task(task: BenchmarkTask) -> bool:
    return task.category == "math" and ("solve" in task.prompt.lower() or len(task.prompt) > 18)


def estimate_test_time_strategy(strategy: str) -> TestTimeStrategyEstimate:
    tasks = default_benchmark()
    total_extra_tokens = 0
    hard_tasks = [task for task in tasks if _is_hard_math_task(task)]
    covered = 0

    for task in tasks:
        extra = 0
        if strategy == "none":
            extra = 0
        elif strategy == "hard_math_only":
            extra = 32 if _is_hard_math_task(task) else 0
        elif strategy == "math_and_tool":
            extra = 32 if task.category in {"math", "tool"} else 0
        elif strategy == "all_tasks":
            extra = 32
        total_extra_tokens += extra
        if _is_hard_math_task(task) and extra > 0:
            covered += 1

    coverage = covered / max(len(hard_tasks), 1)
    average_extra_tokens = total_extra_tokens / len(tasks)
    return TestTimeStrategyEstimate(
        strategy=strategy,
        average_extra_tokens=round(average_extra_tokens, 2),
        total_extra_tokens=total_extra_tokens,
        hard_task_coverage=round(coverage, 2),
    )


def run_test_time_strategy_sweep() -> list[TestTimeStrategyEstimate]:
    strategies = ("none", "hard_math_only", "math_and_tool", "all_tasks")
    with ThreadPoolExecutor(max_workers=len(strategies)) as pool:
        results = list(pool.map(estimate_test_time_strategy, strategies))
    return sorted(results, key=lambda item: (-item.hard_task_coverage, item.average_extra_tokens))


def choose_next_step(
    budgets: list[BudgetEstimate],
    protocols: list[ProtocolEstimate],
) -> dict[str, object]:
    fitting = [budget for budget in budgets if budget.fits_4gb]
    if not fitting:
        chosen_budget = budgets[0]
    else:
        preferred = [budget for budget in fitting if 40_000_000 <= budget.parameters <= 70_000_000]
        chosen_budget = preferred[0] if preferred else fitting[0]
    chosen_protocol = min(protocols, key=lambda item: item.average_tokens)
    return {
        "recommended_model": chosen_budget.name,
        "recommended_protocol": chosen_protocol.protocol,
        "reason": (
            "Start in the middle of the budget range: large enough to learn non-trivial language, math, science, and coding "
            "structure, small enough to keep pretraining on a 4 GB card realistic."
        ),
        "next_actions": [
            "Pretrain the recommended byte-level baseline until the loss curve stabilizes.",
            "Replace the byte tokenizer with a compact SentencePiece/BPE tokenizer and rerun the protocol sweep.",
            "Distill calculator traces with the recommended compact protocol.",
            "Run an ablation with Titans-inspired memory on and off after the base model can answer short tasks.",
        ],
    }


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def serialize_budget_sweep(results: list[BudgetEstimate]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]


def serialize_protocol_sweep(results: list[ProtocolEstimate]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]


def serialize_memory_sweep(results: list[MemorySweepEstimate]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]


def serialize_test_time_sweep(results: list[TestTimeStrategyEstimate]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]


