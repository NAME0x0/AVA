from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ResearchPaper:
    key: str
    title: str
    arxiv_url: str
    submitted: str
    updated: str | None
    theme: str
    ava_takeaway: str


@dataclass(frozen=True, slots=True)
class ResearchHypothesis:
    key: str
    name: str
    priority: int
    paper_keys: tuple[str, ...]
    experiment: str
    success_signal: str


PAPERS = (
    ResearchPaper(
        key="switch-transformer",
        title="Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
        arxiv_url="https://arxiv.org/abs/2101.03961",
        submitted="2021-01-11",
        updated="2022-06-16",
        theme="sparse moe scaling",
        ava_takeaway="Sparse MoE raises parameter count at near-constant compute, but routing complexity and instability are first-class costs.",
    ),
    ResearchPaper(
        key="st-moe",
        title="ST-MoE: Designing Stable and Transferable Sparse Expert Models",
        arxiv_url="https://arxiv.org/abs/2202.08906",
        submitted="2022-02-17",
        updated="2022-04-29",
        theme="sparse moe stability",
        ava_takeaway="MoE can be made strong and stable, but the successful regimes are still far above AVA's single-GPU budget.",
    ),
    ResearchPaper(
        key="limo",
        title="LIMO: Less is More for Reasoning",
        arxiv_url="https://arxiv.org/abs/2502.03387",
        submitted="2025-02-05",
        updated="2025-07-29",
        theme="reasoning data efficiency",
        ava_takeaway="Use a small, high-quality reasoning set with short rationales before scaling volume.",
    ),
    ResearchPaper(
        key="deepseek-r1",
        title="DeepSeek-R1",
        arxiv_url="https://arxiv.org/abs/2501.12948",
        submitted="2025-01-22",
        updated="2026-01-04",
        theme="verifiable RL reasoning",
        ava_takeaway="Use verifiable post-training on math and tool tasks after the base model is competent.",
    ),
    ResearchPaper(
        key="s1",
        title="s1: Simple test-time scaling",
        arxiv_url="https://arxiv.org/abs/2501.19393",
        submitted="2025-01-31",
        updated="2025-03-01",
        theme="test-time scaling",
        ava_takeaway="Spend extra thinking tokens only on hard math prompts.",
    ),
    ResearchPaper(
        key="toolace-r",
        title="ToolACE-R: Revamping Tool Learning for LLMs via Multi-Query, Model-aware Iterative Refinement",
        arxiv_url="https://arxiv.org/abs/2504.01400",
        submitted="2025-04-02",
        updated="2026-01-10",
        theme="tool learning",
        ava_takeaway="Refine tool traces against the current model rather than freezing a single synthetic dataset.",
    ),
    ResearchPaper(
        key="toolformer",
        title="Toolformer: Language Models Can Teach Themselves to Use Tools",
        arxiv_url="https://arxiv.org/abs/2302.04761",
        submitted="2023-02-09",
        updated=None,
        theme="tool augmentation",
        ava_takeaway="A few strong tool demonstrations can unlock self-supervised tool use patterns.",
    ),
    ResearchPaper(
        key="deepseekmath",
        title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        arxiv_url="https://arxiv.org/abs/2402.03300",
        submitted="2024-02-05",
        updated="2024-04-27",
        theme="math domain adaptation",
        ava_takeaway="Math-heavy continued pretraining is worth doing before math-specific post-training.",
    ),
    ResearchPaper(
        key="mixtral",
        title="Mixtral of Experts",
        arxiv_url="https://arxiv.org/abs/2401.04088",
        submitted="2024-01-08",
        updated=None,
        theme="sparse moe quality",
        ava_takeaway="Sparse MoE can beat larger dense models, but the stored parameter count is still much too large for a 4 GB local-first product.",
    ),
    ResearchPaper(
        key="deepseek-moe",
        title="DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models",
        arxiv_url="https://arxiv.org/abs/2401.06066",
        submitted="2024-01-11",
        updated=None,
        theme="sparse moe specialization",
        ava_takeaway="Better expert specialization improves MoE efficiency, but it does not remove the storage and systems costs that dominate at 4 GB.",
    ),
    ResearchPaper(
        key="jetmoe",
        title="JetMoE: Reaching Llama2 Performance with 0.1M Dollars",
        arxiv_url="https://arxiv.org/abs/2404.07413",
        submitted="2024-04-11",
        updated=None,
        theme="accessible sparse moe",
        ava_takeaway="JetMoE is the strongest evidence that accessible MoE can work, but even JetMoE-class models are still too large for a comfortable 4 GB standalone target.",
    ),
    ResearchPaper(
        key="titans",
        title="Titans: Learning to Memorize at Test Time",
        arxiv_url="https://arxiv.org/abs/2501.00663",
        submitted="2024-12-31",
        updated=None,
        theme="external memory",
        ava_takeaway="Keep long-term memory outside the tiny base model and gate writes by surprise.",
    ),
    ResearchPaper(
        key="phi-4",
        title="Phi-4 Technical Report",
        arxiv_url="https://arxiv.org/abs/2412.08905",
        submitted="2024-12-12",
        updated=None,
        theme="small-model quality",
        ava_takeaway="Data quality and curriculum design can dominate parameter count for narrow tasks.",
    ),
    ResearchPaper(
        key="bitnet",
        title="BitNet b1.58 2B4T Technical Report",
        arxiv_url="https://arxiv.org/abs/2504.12285",
        submitted="2025-04-16",
        updated="2025-04-25",
        theme="deployment efficiency",
        ava_takeaway="Aggressive low-bit deployment is a later branch, not a replacement for a strong dense teacher.",
    ),
    ResearchPaper(
        key="deepseek-v3",
        title="DeepSeek-V3 Technical Report",
        arxiv_url="https://arxiv.org/abs/2412.19437",
        submitted="2024-12-27",
        updated="2025-02-18",
        theme="frontier sparse moe",
        ava_takeaway="Modern frontier-competitive MoE is real, but it arrives at scales utterly incompatible with a 4 GB local-first deployment target.",
    ),
    ResearchPaper(
        key="dapo",
        title="DAPO: An Open-Source LLM Reinforcement Learning System at Scale",
        arxiv_url="https://arxiv.org/abs/2503.14476",
        submitted="2025-03-18",
        updated="2025-05-20",
        theme="RL optimization",
        ava_takeaway="If AVA uses verifiable RL, the trainer design will matter as much as the reward.",
    ),
    ResearchPaper(
        key="qmoe",
        title="QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models",
        arxiv_url="https://arxiv.org/abs/2310.16795",
        submitted="2023-10-25",
        updated=None,
        theme="moe compression",
        ava_takeaway="Compression helps, but even aggressive MoE compression is still a server-scale story rather than a 4 GB laptop story.",
    ),
    ResearchPaper(
        key="squeezellm",
        title="SqueezeLLM: Dense-and-Sparse Quantization",
        arxiv_url="https://arxiv.org/abs/2306.07629",
        submitted="2023-06-13",
        updated="2024-06-05",
        theme="memory bandwidth",
        ava_takeaway="Single-batch local inference is memory-bandwidth bound, so offloading MoE experts can easily erase the theoretical compute win.",
    ),
    ResearchPaper(
        key="icrl-tool-use",
        title="In-Context Reinforcement Learning for Tool Use in Large Language Models",
        arxiv_url="https://arxiv.org/abs/2603.08068",
        submitted="2026-03-09",
        updated=None,
        theme="tool RL without SFT",
        ava_takeaway="Promising later-stage tool RL recipe: warm-start tool behavior with in-context examples during rollouts, then anneal toward zero-shot tool use.",
    ),
    ResearchPaper(
        key="torl",
        title="ToRL: Scaling Tool-Integrated RL",
        arxiv_url="https://arxiv.org/abs/2503.23383",
        submitted="2025-03-30",
        updated=None,
        theme="tool-integrated RL",
        ava_takeaway="Pure reward-driven tool use can produce strategic invocation behavior once the RL stack is stable enough.",
    ),
    ResearchPaper(
        key="retool",
        title="ReTool: Reinforcement Learning for Strategic Tool Use in LLMs",
        arxiv_url="https://arxiv.org/abs/2504.11536",
        submitted="2025-04-15",
        updated="2025-04-17",
        theme="strategic tool RL",
        ava_takeaway="Tool RL is strongest when the model learns when not to call tools as well as how to call them.",
    ),
    ResearchPaper(
        key="deepplanning",
        title="DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints",
        arxiv_url="https://arxiv.org/abs/2601.18137",
        submitted="2026-01-26",
        updated=None,
        theme="agentic planning benchmark",
        ava_takeaway="AVA should add explicit planning benchmarks before making strong agent claims.",
    ),
    ResearchPaper(
        key="skillnet",
        title="SkillNet: Create, Evaluate, and Connect AI Skills",
        arxiv_url="https://arxiv.org/abs/2603.04448",
        submitted="2026-02-26",
        updated=None,
        theme="skill library runtime",
        ava_takeaway="Reusable skills belong in the product runtime layer and evaluation stack more than in the base model weights.",
    ),
    ResearchPaper(
        key="penguin-vl",
        title="Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based Vision Encoders",
        arxiv_url="https://arxiv.org/abs/2603.06569",
        submitted="2026-03-06",
        updated=None,
        theme="efficient multimodal",
        ava_takeaway="For AVA's future vision branch, compact multimodal design matters more than copying large VLM recipes.",
    ),
    ResearchPaper(
        key="internvl-u",
        title="InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing",
        arxiv_url="https://arxiv.org/abs/2603.09877",
        submitted="2026-03-10",
        updated=None,
        theme="unified multimodal",
        ava_takeaway="Useful scale reference for a later multimodal family, but too broad and heavy for AVA's current 4 GB mainline.",
    ),
    ResearchPaper(
        key="areal",
        title="AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning",
        arxiv_url="https://arxiv.org/abs/2505.24298",
        submitted="2025-05-30",
        updated="2026-03-02",
        theme="asynchronous RL systems",
        ava_takeaway="Only invest in large RL systems work if rollout throughput becomes the bottleneck after simpler tool and verifier loops already work.",
    ),
)


HYPOTHESES = (
    ResearchHypothesis(
        key="exp-001",
        name="LIMO-style micro-rationale curriculum",
        priority=1,
        paper_keys=("limo", "phi-4"),
        experiment="Create a compact language, math, science, and coding demonstration set with short rationales instead of verbose chain-of-thought.",
        success_signal="Lower training-token cost with no regression on arithmetic, short science QA, simple coding tasks, and concise instruction following.",
    ),
    ResearchHypothesis(
        key="exp-002",
        name="Math, science, and code DAPT before tool tuning",
        priority=2,
        paper_keys=("deepseekmath", "phi-4"),
        experiment="Bias early continued pretraining toward high-quality math, science, code, and textbook-style instruction corpora.",
        success_signal="Better math, science, and coding benchmark performance before any tool traces are added.",
    ),
    ResearchHypothesis(
        key="exp-003",
        name="Iterative compact tool curriculum",
        priority=3,
        paper_keys=("toolformer", "toolace-r"),
        experiment="Train compact calculator call/observe/integrate traces, then regenerate weak cases with the current checkpoint in the loop.",
        success_signal="Higher calculator-tool success rate without large prompt overhead.",
    ),
    ResearchHypothesis(
        key="exp-004",
        name="Verifiable RL only after competence",
        priority=4,
        paper_keys=("deepseek-r1", "dapo"),
        experiment="Apply verifiable RL to arithmetic, short science QA, simple code execution tasks, and tool correctness only after the base model is already stable.",
        success_signal="Improved exact-match math, science, coding, and tool metrics without harming concise language responses.",
    ),
    ResearchHypothesis(
        key="exp-005",
        name="Selective budget forcing",
        priority=5,
        paper_keys=("s1",),
        experiment="Use extra test-time budget only on hard reasoning prompts instead of every request.",
        success_signal="Reasoning gains with small average token overhead.",
    ),
    ResearchHypothesis(
        key="exp-006",
        name="Titans-inspired external memory",
        priority=6,
        paper_keys=("titans",),
        experiment="Keep long-term memory external, sparse, and surprise-gated.",
        success_signal="Better long-horizon retrieval with bounded context inflation.",
    ),
    ResearchHypothesis(
        key="exp-007",
        name="BitNet deployment branch",
        priority=7,
        paper_keys=("bitnet",),
        experiment="Quantize or retrain a separate deployment branch only after the dense model becomes a strong teacher.",
        success_signal="Deployment footprint drops without becoming the main blocker for quality.",
    ),
    ResearchHypothesis(
        key="exp-008",
        name="Tiny MoE branch after dense teacher",
        priority=8,
        paper_keys=("switch-transformer", "st-moe", "mixtral", "jetmoe", "qmoe", "squeezellm"),
        experiment="Only explore a tiny sparse MoE branch after the dense AVA baseline is strong, and judge it against 4 GB residency plus narrow language, math, science, and coding wins rather than frontier-level claims.",
        success_signal="A quantized or partially offloaded MoE branch beats the dense AVA checkpoint on selected language, math, science, and coding tasks without breaking the 4 GB product budget.",
    ),
    ResearchHypothesis(
        key="exp-009",
        name="In-context tool RL after compact tool SFT",
        priority=9,
        paper_keys=("icrl-tool-use", "torl", "retool"),
        experiment="After compact calculator and search traces work under supervised training, run a small RL branch that starts with in-context tool demonstrations during rollouts and anneals them away.",
        success_signal="Higher tool accuracy and better abstention behavior than SFT-only traces without a large prompt tax.",
    ),
    ResearchHypothesis(
        key="exp-010",
        name="Planning benchmarks before agent claims",
        priority=10,
        paper_keys=("deepplanning", "skillnet"),
        experiment="Add long-horizon planning evaluation and reusable skill instrumentation before claiming agentic competence.",
        success_signal="Session packets report planning success, constraint satisfaction, and tool-step efficiency instead of vague demos.",
    ),
    ResearchHypothesis(
        key="exp-011",
        name="Compact multimodal branch after text stability",
        priority=11,
        paper_keys=("penguin-vl", "internvl-u"),
        experiment="After the text backbone is stable, prototype a compact vision branch that reuses the text model and targets MathVista, ScienceQA, and DocVQA first.",
        success_signal="Useful multimodal gains under a tight VRAM budget without regressing the text product.",
    ),
    ResearchHypothesis(
        key="exp-012",
        name="Async RL infra only when rollouts dominate cost",
        priority=12,
        paper_keys=("areal",),
        experiment="Only invest in asynchronous RL infrastructure if rollout throughput, not model quality or data quality, becomes the main bottleneck.",
        success_signal="Better verifier throughput and wall-clock efficiency on RL runs without turning the stack into a black box.",
    ),
)


RECENT_HF_PAPER_KEYS = (
    "icrl-tool-use",
    "deepplanning",
    "skillnet",
    "penguin-vl",
    "internvl-u",
    "torl",
    "retool",
    "areal",
)

RECENT_HF_HYPOTHESIS_KEYS = (
    "exp-009",
    "exp-010",
    "exp-011",
    "exp-012",
)


def research_papers() -> list[ResearchPaper]:
    return list(PAPERS)


def research_hypotheses() -> list[ResearchHypothesis]:
    return sorted(HYPOTHESES, key=lambda item: item.priority)


def _paper_index() -> dict[str, ResearchPaper]:
    return {paper.key: paper for paper in research_papers()}


def _hypothesis_index() -> dict[str, ResearchHypothesis]:
    return {hypothesis.key: hypothesis for hypothesis in research_hypotheses()}


def research_papers_by_keys(keys: tuple[str, ...]) -> list[ResearchPaper]:
    paper_map = _paper_index()
    return [paper_map[key] for key in keys]


def research_hypotheses_by_keys(keys: tuple[str, ...]) -> list[ResearchHypothesis]:
    hypothesis_map = _hypothesis_index()
    return [hypothesis_map[key] for key in keys]


def recent_hf_papers() -> list[ResearchPaper]:
    return research_papers_by_keys(RECENT_HF_PAPER_KEYS)


def recent_hf_hypotheses() -> list[ResearchHypothesis]:
    return research_hypotheses_by_keys(RECENT_HF_HYPOTHESIS_KEYS)


def serialize_papers() -> list[dict[str, object]]:
    return [asdict(paper) for paper in research_papers()]


def serialize_hypotheses() -> list[dict[str, object]]:
    return [asdict(hypothesis) for hypothesis in research_hypotheses()]


def serialize_recent_hf_papers() -> list[dict[str, object]]:
    return [asdict(paper) for paper in recent_hf_papers()]


def serialize_recent_hf_hypotheses() -> list[dict[str, object]]:
    return [asdict(hypothesis) for hypothesis in recent_hf_hypotheses()]
