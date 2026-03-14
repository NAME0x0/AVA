from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    key: str
    name: str
    modality: str
    capability: str
    metric: str
    stage: str
    priority: int
    source_url: str
    runner: str
    note: str


BENCHMARKS = (
    BenchmarkSpec(
        key="ifeval",
        name="IFEval",
        modality="text",
        capability="instruction_following",
        metric="strict_prompt_accuracy",
        stage="foundation",
        priority=1,
        source_url="https://arxiv.org/abs/2311.07911",
        runner="planned_external",
        note="Core text compliance benchmark for constrained formatting and instruction obedience.",
    ),
    BenchmarkSpec(
        key="mmlu-pro",
        name="MMLU-Pro",
        modality="text",
        capability="knowledge_reasoning",
        metric="accuracy",
        stage="foundation",
        priority=2,
        source_url="https://github.com/TIGER-AI-Lab/MMLU-Pro",
        runner="planned_external",
        note="Strong default text benchmark for broad academic reasoning.",
    ),
    BenchmarkSpec(
        key="gpqa-diamond",
        name="GPQA Diamond",
        modality="text",
        capability="hard_science_reasoning",
        metric="accuracy",
        stage="foundation",
        priority=3,
        source_url="https://github.com/idavidrein/gpqa",
        runner="planned_external",
        note="Hard science reasoning gate that remains useful at frontier scale.",
    ),
    BenchmarkSpec(
        key="math",
        name="MATH",
        modality="text",
        capability="competition_math",
        metric="exact_match",
        stage="foundation",
        priority=4,
        source_url="https://github.com/hendrycks/math",
        runner="planned_external",
        note="Broad symbolic and competition-math benchmark with established evaluation code.",
    ),
    BenchmarkSpec(
        key="bfcl",
        name="BFCL",
        modality="text",
        capability="tool_use",
        metric="function_call_accuracy",
        stage="foundation",
        priority=5,
        source_url="https://gorilla.cs.berkeley.edu/leaderboard",
        runner="planned_external",
        note="Function-calling benchmark for tool selection, argument correctness, and abstention.",
    ),
    BenchmarkSpec(
        key="humaneval-plus",
        name="HumanEval+",
        modality="code",
        capability="code_generation",
        metric="pass_at_1",
        stage="coding",
        priority=6,
        source_url="https://github.com/evalplus/evalplus",
        runner="planned_external",
        note="Widely used coding benchmark for function synthesis with stronger tests than the original HumanEval.",
    ),
    BenchmarkSpec(
        key="mbpp-plus",
        name="MBPP+",
        modality="code",
        capability="code_generation",
        metric="pass_at_1",
        stage="coding",
        priority=7,
        source_url="https://github.com/evalplus/evalplus",
        runner="planned_external",
        note="Small but practical Python programming benchmark for early code-specialized model checks.",
    ),
    BenchmarkSpec(
        key="m-ifeval",
        name="M-IFEval",
        modality="multilingual",
        capability="multilingual_instruction_following",
        metric="strict_prompt_accuracy",
        stage="multilingual",
        priority=8,
        source_url="https://github.com/lightblue-tech/M-IFEval",
        runner="planned_external",
        note="Measures whether instruction-following transfers across languages instead of only English.",
    ),
    BenchmarkSpec(
        key="mgsm",
        name="MGSM",
        modality="multilingual",
        capability="multilingual_math_reasoning",
        metric="exact_match",
        stage="multilingual",
        priority=9,
        source_url="https://arxiv.org/abs/2210.03057",
        runner="planned_external",
        note="Tracks whether math reasoning survives language transfer.",
    ),
    BenchmarkSpec(
        key="mmlu-prox",
        name="MMLU-ProX",
        modality="multilingual",
        capability="multilingual_reasoning",
        metric="accuracy",
        stage="multilingual",
        priority=10,
        source_url="https://github.com/weihao1115/MMLU-ProX",
        runner="planned_external",
        note="Higher-difficulty multilingual reasoning benchmark built on MMLU-Pro.",
    ),
    BenchmarkSpec(
        key="belebele",
        name="Belebele",
        modality="multilingual",
        capability="multilingual_reading_comprehension",
        metric="accuracy",
        stage="multilingual",
        priority=11,
        source_url="https://github.com/facebookresearch/belebele",
        runner="planned_external",
        note="Massively multilingual reading comprehension check across 122 language variants.",
    ),
    BenchmarkSpec(
        key="flores-200",
        name="FLORES-200",
        modality="multilingual",
        capability="translation_transfer",
        metric="chrF++",
        stage="multilingual",
        priority=12,
        source_url="https://github.com/facebookresearch/flores",
        runner="planned_external",
        note="Translation and language transfer sanity check for multilingual expansion.",
    ),
    BenchmarkSpec(
        key="mmmu",
        name="MMMU",
        modality="vision",
        capability="expert_multimodal_reasoning",
        metric="accuracy",
        stage="multimodal",
        priority=13,
        source_url="https://mmmu-benchmark.github.io/",
        runner="registry_only",
        note="Primary multimodal knowledge-and-reasoning benchmark for later vision-language scaling.",
    ),
    BenchmarkSpec(
        key="mathvista",
        name="MathVista",
        modality="vision",
        capability="visual_math_reasoning",
        metric="accuracy",
        stage="multimodal",
        priority=14,
        source_url="https://mathvista.github.io/",
        runner="registry_only",
        note="Best fit vision benchmark for AVA's eventual math-heavy multimodal branch.",
    ),
    BenchmarkSpec(
        key="scienceqa",
        name="ScienceQA",
        modality="vision",
        capability="multimodal_science_reasoning",
        metric="accuracy",
        stage="multimodal",
        priority=15,
        source_url="https://scienceqa.github.io/",
        runner="registry_only",
        note="Science-focused multimodal benchmark that ties AVA's science scope to later vision-language work.",
    ),
    BenchmarkSpec(
        key="ai2d",
        name="AI2D",
        modality="vision",
        capability="diagram_understanding",
        metric="accuracy",
        stage="multimodal",
        priority=16,
        source_url="https://prior.allenai.org/projects/diagram-understanding",
        runner="registry_only",
        note="Diagram reasoning benchmark useful for science and educational visual understanding.",
    ),
    BenchmarkSpec(
        key="chartqa",
        name="ChartQA",
        modality="vision",
        capability="chart_reasoning",
        metric="accuracy",
        stage="multimodal",
        priority=17,
        source_url="https://github.com/vis-nlp/ChartQA",
        runner="registry_only",
        note="Evaluates reasoning over plots and charts rather than generic image captioning.",
    ),
    BenchmarkSpec(
        key="docvqa",
        name="DocVQA",
        modality="vision",
        capability="document_reasoning",
        metric="anls",
        stage="multimodal",
        priority=18,
        source_url="https://www.docvqa.org/",
        runner="registry_only",
        note="Document-image QA benchmark for OCR-heavy and layout-sensitive use cases.",
    ),
    BenchmarkSpec(
        key="deepplanning",
        name="DeepPlanning",
        modality="agentic",
        capability="long_horizon_planning",
        metric="constraint_satisfaction",
        stage="agentic",
        priority=19,
        source_url="https://arxiv.org/abs/2601.18137",
        runner="registry_only",
        note="Long-horizon planning benchmark with verifiable budget and ordering constraints.",
    ),
    BenchmarkSpec(
        key="livecodebench",
        name="LiveCodeBench",
        modality="code",
        capability="coding_generalization",
        metric="pass_at_1",
        stage="scale_only",
        priority=20,
        source_url="https://livecodebench.github.io/",
        runner="registry_only",
        note="Contamination-aware coding benchmark for later scale stages.",
    ),
    BenchmarkSpec(
        key="swe-bench-verified",
        name="SWE-bench Verified",
        modality="agentic",
        capability="software_issue_resolution",
        metric="percent_resolved",
        stage="scale_only",
        priority=21,
        source_url="https://www.swebench.com/",
        runner="registry_only",
        note="High-value frontier-style software benchmark, but not appropriate for the first 4 GB training line.",
    ),
    BenchmarkSpec(
        key="tau2-bench",
        name="tau2-bench",
        modality="agentic",
        capability="policy_constrained_tool_use",
        metric="task_success",
        stage="scale_only",
        priority=22,
        source_url="https://github.com/sierra-research/tau2-bench",
        runner="registry_only",
        note="Agentic customer-service benchmark for multi-turn policy and tool adherence.",
    ),
)


def benchmark_registry() -> list[BenchmarkSpec]:
    return sorted(BENCHMARKS, key=lambda item: item.priority)


def filter_benchmark_registry(
    *,
    modality: str | None = None,
    stage: str | None = None,
) -> list[BenchmarkSpec]:
    items = benchmark_registry()
    if modality and modality != "all":
        items = [item for item in items if item.modality == modality]
    if stage and stage != "all":
        items = [item for item in items if item.stage == stage]
    return items


def serialize_benchmark_registry(
    *,
    modality: str | None = None,
    stage: str | None = None,
) -> list[dict[str, object]]:
    return [asdict(item) for item in filter_benchmark_registry(modality=modality, stage=stage)]


def benchmark_registry_summary() -> dict[str, object]:
    items = benchmark_registry()
    by_modality: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for item in items:
        by_modality[item.modality] = by_modality.get(item.modality, 0) + 1
        by_stage[item.stage] = by_stage.get(item.stage, 0) + 1
    return {
        "total": len(items),
        "by_modality": by_modality,
        "by_stage": by_stage,
    }
