from __future__ import annotations

import os
from pathlib import Path


def _discover_project_root(start: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parents[2])

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        search_roots = [resolved] if resolved.is_dir() else [resolved.parent]
        for root in search_roots:
            for parent in (root, *root.parents):
                if (parent / '.env').exists() or (parent / 'pyproject.toml').exists():
                    return parent
    return Path(start).resolve() if start is not None else Path.cwd().resolve()


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value


def load_project_env(root: str | Path | None = None, *, override: bool = False) -> dict[str, object]:
    root_path = _discover_project_root(root)
    env_path = root_path / ".env"
    loaded_keys: list[str] = []
    if not env_path.exists():
        return {
            "path": str(env_path),
            "loaded": False,
            "keys": loaded_keys,
        }
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key in os.environ and not override:
            continue
        os.environ[key] = value
        loaded_keys.append(key)
    return {
        "path": str(env_path),
        "loaded": True,
        "keys": loaded_keys,
    }


def huggingface_token() -> str | None:
    for key in (
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGING_FACE_TOKEN",
        "HUGGINGFACE_TOKEN",
        "hugging_face_token",
        "huggingface_token",
    ):
        value = os.environ.get(key)
        if value:
            return value
    return None
