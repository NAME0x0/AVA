from ava.research import (
    recent_hf_hypotheses,
    recent_hf_papers,
    research_hypotheses,
    research_papers,
)


def test_research_papers_include_recent_arxiv_items() -> None:
    papers = research_papers()
    keys = {paper.key for paper in papers}
    assert "limo" in keys
    assert "deepseek-r1" in keys
    assert "toolace-r" in keys


def test_research_papers_include_moe_reality_check_items() -> None:
    papers = research_papers()
    keys = {paper.key for paper in papers}
    assert "mixtral" in keys
    assert "jetmoe" in keys
    assert "deepseek-v3" in keys


def test_recent_hf_papers_cover_architecture_retrieval_and_multimodal() -> None:
    papers = recent_hf_papers()
    keys = {paper.key for paper in papers}
    assert "latent-recurrent-depth" in keys
    assert "hipporag2" in keys
    assert "qwen3" in keys
    assert "deepplanning" in keys
    assert "penguin-vl" in keys
    assert "areal" in keys


def test_research_hypotheses_sorted_by_priority() -> None:
    hypotheses = research_hypotheses()
    priorities = [item.priority for item in hypotheses]
    assert priorities == sorted(priorities)
    assert any(item.key == "exp-008" for item in hypotheses)


def test_recent_hf_hypotheses_exist() -> None:
    hypotheses = recent_hf_hypotheses()
    keys = {item.key for item in hypotheses}
    assert keys == {
        "exp-009",
        "exp-010",
        "exp-011",
        "exp-012",
        "exp-013",
        "exp-014",
        "exp-015",
        "exp-016",
    }
