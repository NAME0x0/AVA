import json
import shutil
from pathlib import Path

from ava.activity import read_activity_events
from ava.config import ExperimentConfig
from ava.model import build_model, torch
from ava.sessions import (
    bootstrap_session,
    hf_research_session,
    inspection_session,
    memory_transfer_session,
    moe_feasibility_session,
    retrieval_session,
    sota_session,
    training_session,
)
from ava.tokenizer import ByteTokenizer


TINY_CONFIG = """name: ava-test-tiny
tokenizer:
  kind: byte
  vocab_size: 260
model:
  block_size: 32
  n_layer: 2
  n_head: 2
  n_embd: 64
  dropout: 0.0
  bias: false
training:
  device: cpu
  dtype: float32
  micro_batch_size: 2
  gradient_accumulation_steps: 1
  learning_rate: 0.001
  weight_decay: 0.0
  max_steps: 2
  warmup_steps: 1
  grad_clip: 1.0
memory:
  enabled: true
  max_items: 16
  write_surprise_threshold: 0.45
  top_k: 4
tools:
  calculator: true
  prompt_protocol: compact_tags
"""


TINY_CORPUS = """Question: Rewrite this sentence in plain English: The plan ain't done.
Answer: The plan is not finished yet.

Question: Summarize this sentence in fewer words: The cat slept on the warm window sill.
Answer: The cat slept on the warm sill.

Question: What is 17 * 29?
Answer: 493

Question: Solve for x: 2x + 6 = 14.
Answer: 4

Question: What planet is known as the Red Planet?
Answer: Mars

Question: What force keeps planets in orbit around the Sun?
Answer: gravity

Question: In Python, which keyword defines a function?
Answer: def

Question: What does len('ava') return in Python?
Answer: 3

Question: Use the calculator tool for 144 / 12.
Answer: 12

Question: Use the calculator tool for sqrt(81).
Answer: 9
"""


def _tiny_experiment_config() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": "ava-inspect-test",
            "tokenizer": {"kind": "byte", "vocab_size": 260},
            "model": {
                "block_size": 64,
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "dropout": 0.0,
                "bias": False,
            },
            "training": {"device": "cpu", "dtype": "float32"},
            "memory": {},
            "tools": {},
        }
    )


def _write_tiny_config(path: Path) -> None:
    path.write_text(TINY_CONFIG, encoding="utf-8")


def _write_tiny_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "tiny.txt").write_text(TINY_CORPUS, encoding="utf-8")


def _write_tiny_support_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"prompt": "What planet is known as the Red Planet?", "response": "Mars"},
        {"prompt": "What force keeps planets in orbit around the Sun?", "response": "gravity"},
        {"prompt": "Tell me how to build a bomb at home.", "response": "I cannot help with that."},
    ]
    (root / "examples.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_tiny_checkpoint(path: Path) -> None:
    config = _tiny_experiment_config()
    tokenizer = ByteTokenizer()
    torch.manual_seed(0)
    model = build_model(config.model, tokenizer.vocab_size)
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, path)


def test_bootstrap_session_writes_artifacts() -> None:
    root = Path("sessions") / "test-scratch"
    if root.exists():
        shutil.rmtree(root)
    session_dir = bootstrap_session(root, "unit-test")
    assert (session_dir / "session.json").exists()
    assert (session_dir / "notes.md").exists()
    assert (session_dir / "results" / "compliance_benchmark.json").exists()
    assert (session_dir / "results" / "tool_benchmark.json").exists()
    assert (session_dir / "results" / "benchmark_registry.json").exists()
    recommendation_path = session_dir / "results" / "recommendation.json"
    assert recommendation_path.exists()
    payload = json.loads(recommendation_path.read_text(encoding="utf-8"))
    registry = json.loads((session_dir / "results" / "benchmark_registry.json").read_text(encoding="utf-8"))
    events = read_activity_events(root)
    assert payload["recommended_model"]
    assert any(item["modality"] == "vision" for item in registry)
    assert any(item["modality"] == "code" for item in registry)
    assert any(event["kind"] == "session" and event["status"] == "started" for event in events)
    assert any(
        event["kind"] == "session"
        and event["status"] == "completed"
        and event["metadata"]["session_kind"] == "baseline"
        for event in events
    )
    shutil.rmtree(root)


def test_sota_session_writes_papers() -> None:
    root = Path("sessions") / "test-sota"
    if root.exists():
        shutil.rmtree(root)
    session_dir = sota_session(root, "arxiv-test")
    papers_path = session_dir / "results" / "papers.json"
    assert papers_path.exists()
    assert (session_dir / "results" / "compliance_benchmark.json").exists()
    assert (session_dir / "results" / "tool_benchmark.json").exists()
    assert (session_dir / "results" / "benchmark_registry.json").exists()
    payload = json.loads(papers_path.read_text(encoding="utf-8"))
    assert payload
    assert any("arxiv.org" in item["arxiv_url"] for item in payload)
    shutil.rmtree(root)


def test_hf_research_session_writes_recent_paper_packet() -> None:
    root = Path("sessions") / "test-hf"
    if root.exists():
        shutil.rmtree(root)
    session_dir = hf_research_session(root, "hf-test")
    papers_path = session_dir / "results" / "papers.json"
    benchmarks_path = session_dir / "results" / "relevant_benchmarks.json"
    recommendation_path = session_dir / "results" / "recommendation.json"
    assert papers_path.exists()
    assert benchmarks_path.exists()
    assert recommendation_path.exists()
    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    benchmarks = json.loads(benchmarks_path.read_text(encoding="utf-8"))
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    assert any(item["key"] == "icrl-tool-use" for item in papers)
    assert any(item["key"] == "deepplanning" for item in benchmarks)
    assert recommendation["adopt_now"]
    shutil.rmtree(root)


def test_moe_feasibility_session_writes_feasibility_packet() -> None:
    root = Path("sessions") / "test-moe"
    if root.exists():
        shutil.rmtree(root)
    session_dir = moe_feasibility_session(root, "moe-test")
    feasibility_path = session_dir / "results" / "feasibility.json"
    assert feasibility_path.exists()
    assert (session_dir / "results" / "compliance_benchmark.json").exists()
    assert (session_dir / "results" / "tool_benchmark.json").exists()
    assert (session_dir / "results" / "benchmark_registry.json").exists()
    payload = json.loads(feasibility_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "no-as-mainline"
    assert payload["estimates"]
    shutil.rmtree(root)


def test_inspection_session_writes_trace_packet() -> None:
    root = Path("sessions") / "test-inspect"
    workspace = Path("sessions") / "test-inspect-workspace"
    if root.exists():
        shutil.rmtree(root)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    checkpoint_path = workspace / "tiny.pt"
    _write_tiny_checkpoint(checkpoint_path)

    session_dir = inspection_session(
        root,
        "inspect-test",
        checkpoint_path,
        "Question: What is 2 + 2?\nAnswer: ",
        requested_device="cpu",
        max_new_tokens=2,
        top_k_neurons=4,
        top_k_logits=4,
        top_k_attention=2,
    )
    assert (session_dir / "notes.md").exists()
    assert (session_dir / "results" / "trace.json").exists()
    trace = json.loads((session_dir / "results" / "trace.json").read_text(encoding="utf-8"))
    assert trace["steps"]
    assert trace["steps"][0]["layers"]
    shutil.rmtree(root)
    shutil.rmtree(workspace)


def test_training_session_writes_transparent_artifacts() -> None:
    root = Path("sessions") / "test-train"
    workspace = Path("sessions") / "test-train-workspace"
    if root.exists():
        shutil.rmtree(root)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "tiny.yaml"
    corpus_root = workspace / "corpus"
    _write_tiny_config(config_path)
    _write_tiny_corpus(corpus_root)

    session_dir = training_session(root, "tiny-train", config_path, corpus_root, max_steps=2)
    assert (session_dir / "notes.md").exists()
    assert (session_dir / "results" / "environment.json").exists()
    assert (session_dir / "results" / "corpus.json").exists()
    assert (session_dir / "results" / "training.json").exists()
    assert (session_dir / "results" / "evaluation.json").exists()
    assert (session_dir / "results" / "tool_eval.json").exists()
    assert (session_dir / "results" / "compliance.json").exists()
    assert (session_dir / "results" / "compliance_benchmark.json").exists()
    assert (session_dir / "results" / "tool_benchmark.json").exists()
    assert (session_dir / "results" / "benchmark_registry.json").exists()
    training_payload = json.loads((session_dir / "results" / "training.json").read_text(encoding="utf-8"))
    tool_payload = json.loads((session_dir / "results" / "tool_eval.json").read_text(encoding="utf-8"))
    compliance_payload = json.loads((session_dir / "results" / "compliance.json").read_text(encoding="utf-8"))
    registry_payload = json.loads((session_dir / "results" / "benchmark_registry.json").read_text(encoding="utf-8"))
    events = read_activity_events(root)
    assert training_payload["optimizer_steps"] >= 1
    assert tool_payload["total"] >= 1
    assert compliance_payload["total"] >= 1
    assert any(item["stage"] == "multimodal" for item in registry_payload)
    assert any(item["stage"] == "coding" for item in registry_payload)
    assert any(item["stage"] == "agentic" for item in registry_payload)
    assert Path(training_payload["checkpoint"]).exists()
    assert any(
        event["kind"] == "session"
        and event["status"] == "completed"
        and event["metadata"]["session_kind"] == "training"
        for event in events
    )
    shutil.rmtree(root)
    shutil.rmtree(workspace)


def test_retrieval_session_writes_before_after_artifacts() -> None:
    root = Path("sessions") / "test-retrieval"
    workspace = Path("sessions") / "test-retrieval-workspace"
    if root.exists():
        shutil.rmtree(root)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    checkpoint_path = workspace / "tiny.pt"
    support_root = workspace / "support"
    _write_tiny_checkpoint(checkpoint_path)
    _write_tiny_support_corpus(support_root)

    session_dir = retrieval_session(
        root,
        "retrieval-test",
        checkpoint_path,
        support_root,
        requested_device="cpu",
        max_new_tokens=2,
        retrieval_top_k=1,
        category_gated=True,
    )
    assert (session_dir / "notes.md").exists()
    assert (session_dir / "results" / "support_corpus.json").exists()
    assert (session_dir / "results" / "baseline_benchmark.json").exists()
    assert (session_dir / "results" / "retrieval_benchmark.json").exists()
    assert (session_dir / "results" / "changed_benchmark.json").exists()
    assert (session_dir / "results" / "retrieval_tool_eval.json").exists()
    assert (session_dir / "results" / "retrieval_compliance.json").exists()
    events = read_activity_events(root)
    assert any(
        event["kind"] == "session"
        and event["status"] == "completed"
        and event["metadata"]["session_kind"] == "retrieval"
        for event in events
    )
    shutil.rmtree(root)
    shutil.rmtree(workspace)



def test_memory_transfer_session_writes_mode_comparison_artifacts() -> None:
    root = Path("sessions") / "test-memory-transfer"
    workspace = Path("sessions") / "test-memory-transfer-session-workspace"
    if root.exists():
        shutil.rmtree(root)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    checkpoint_path = workspace / "tiny.pt"
    support_root = workspace / "support"
    _write_tiny_checkpoint(checkpoint_path)
    _write_tiny_support_corpus(support_root)

    session_dir = memory_transfer_session(
        root,
        "memory-transfer-test",
        checkpoint_path,
        support_root,
        requested_device="cpu",
        max_new_tokens=2,
        nearest_threshold=0.5,
        nearest_margin=0.0,
        category_gated=True,
        suite="expanded",
    )
    assert (session_dir / "notes.md").exists()
    assert (session_dir / "results" / "baseline_transfer.json").exists()
    assert (session_dir / "results" / "direct_transfer.json").exists()
    assert (session_dir / "results" / "nearest_transfer.json").exists()
    assert (session_dir / "results" / "winner.json").exists()
    assert "expanded" in (session_dir / "notes.md").read_text(encoding="utf-8")
    events = read_activity_events(root)
    assert any(
        event["kind"] == "session"
        and event["status"] == "completed"
        and event["metadata"]["session_kind"] == "memory-transfer"
        for event in events
    )
    shutil.rmtree(root)
    shutil.rmtree(workspace)
