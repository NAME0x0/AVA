from ava.benchmarks import benchmark_registry_summary, filter_benchmark_registry, serialize_benchmark_registry


def test_benchmark_registry_includes_multimodal_multilingual_and_code() -> None:
    payload = serialize_benchmark_registry()
    modalities = {item["modality"] for item in payload}
    assert "text" in modalities
    assert "multilingual" in modalities
    assert "vision" in modalities
    assert "code" in modalities
    assert "agentic" in modalities


def test_benchmark_registry_filters_by_modality() -> None:
    vision = filter_benchmark_registry(modality="vision")
    assert vision
    assert all(item.modality == "vision" for item in vision)

    code = filter_benchmark_registry(modality="code")
    assert code
    assert all(item.modality == "code" for item in code)

    agentic = filter_benchmark_registry(modality="agentic")
    assert agentic
    assert all(item.modality == "agentic" for item in agentic)


def test_benchmark_registry_summary_counts() -> None:
    summary = benchmark_registry_summary()
    assert summary["total"] >= 12
    assert summary["by_stage"]["foundation"] >= 5
    assert summary["by_stage"]["coding"] >= 2
    assert summary["by_stage"]["agentic"] >= 1
    assert summary["by_modality"]["vision"] >= 5
    assert summary["by_modality"]["agentic"] >= 2
