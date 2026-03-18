import os
from pathlib import Path

from ava.env import huggingface_token, load_project_env


def test_huggingface_token_supports_lowercase_alias(monkeypatch) -> None:
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("hugging_face_token", "secret-token")
    assert huggingface_token() == "secret-token"


def test_load_project_env_discovers_repo_root_from_nested_dir(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    nested = root / "a" / "b"
    nested.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('[project]\nname="ava-test"\n', encoding="utf-8")
    (root / ".env").write_text("HF_TOKEN=nested-secret\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    payload = load_project_env()
    assert payload["loaded"] is True
    assert Path(payload["path"]).resolve() == (root / ".env").resolve()
    assert os.environ["HF_TOKEN"] == "nested-secret"
