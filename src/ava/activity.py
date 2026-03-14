from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from re import sub
from time import monotonic


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "activity"


def render_command(arguments: list[str]) -> str:
    return subprocess.list2cmdline([str(argument) for argument in arguments])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_command(arguments: list[str], *, cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def repo_manifest(cwd: str | Path | None = None) -> dict[str, object]:
    head = _run_command(["git", "rev-parse", "HEAD"], cwd=cwd)
    status_output = _run_command(["git", "status", "--short", "--untracked-files=all"], cwd=cwd)
    dirty_paths = status_output.splitlines() if status_output else []
    return {
        "head": head,
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def _activity_dir(root: str | Path) -> Path:
    activity_dir = Path(root) / "activity"
    activity_dir.mkdir(parents=True, exist_ok=True)
    return activity_dir


def _activity_log_path(root: str | Path, *, timestamp: datetime | None = None) -> Path:
    event_time = timestamp or _utc_now()
    return _activity_dir(root) / f"{event_time.strftime('%Y-%m-%d')}.jsonl"


def record_activity(
    root: str | Path,
    kind: str,
    status: str,
    *,
    event_id: str | None = None,
    command: str | None = None,
    argv: list[str] | None = None,
    metadata: dict[str, object] | None = None,
    cwd: str | Path | None = None,
    duration_ms: int | None = None,
) -> dict[str, object]:
    timestamp = _utc_now()
    resolved_cwd = Path(cwd) if cwd is not None else Path.cwd()
    payload: dict[str, object] = {
        "event_id": event_id or uuid.uuid4().hex,
        "timestamp_utc": timestamp.isoformat(),
        "kind": kind,
        "status": status,
        "cwd": str(resolved_cwd),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "repo": repo_manifest(resolved_cwd),
    }
    if command is not None:
        payload["command"] = command
    if argv is not None:
        payload["argv"] = [str(argument) for argument in argv]
    if metadata is not None:
        payload["metadata"] = metadata
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms

    log_path = _activity_log_path(root, timestamp=timestamp)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return payload


def snapshot_repo_state(
    root: str | Path,
    *,
    label: str = "manual",
    cwd: str | Path | None = None,
) -> Path:
    resolved_cwd = Path(cwd) if cwd is not None else Path.cwd()
    timestamp = _utc_now()
    snapshot_dir = _activity_dir(root) / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{timestamp.strftime('%Y-%m-%d-%H%M%S')}-{slugify(label)}.json"
    payload = {
        "label": label,
        "created_at_utc": timestamp.isoformat(),
        "cwd": str(resolved_cwd),
        "repo": repo_manifest(resolved_cwd),
    }
    snapshot_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    record_activity(
        root,
        "repo_snapshot",
        "completed",
        event_id=snapshot_path.stem,
        metadata={
            "label": label,
            "snapshot_path": str(snapshot_path),
            "dirty_paths_count": len(payload["repo"]["dirty_paths"]),
        },
        cwd=resolved_cwd,
    )
    return snapshot_path


def run_logged_command(
    root: str | Path,
    command: list[str],
    *,
    label: str | None = None,
    cwd: str | Path | None = None,
) -> dict[str, object]:
    resolved_cwd = Path(cwd) if cwd is not None else Path.cwd()
    command_list = [str(part) for part in command]
    resolved_label = label or Path(command_list[0]).name
    timestamp = _utc_now()
    artifact_dir = (
        _activity_dir(root)
        / "commands"
        / f"{timestamp.strftime('%Y-%m-%d-%H%M%S')}-{slugify(resolved_label)}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    command_text = render_command(command_list)
    event_id = artifact_dir.name
    record_activity(
        root,
        "external_command",
        "started",
        event_id=event_id,
        command=command_text,
        argv=command_list,
        metadata={"label": resolved_label, "artifact_dir": str(artifact_dir)},
        cwd=resolved_cwd,
    )

    started_at = monotonic()
    result = subprocess.run(
        command_list,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(resolved_cwd),
        encoding="utf-8",
        errors="replace",
    )
    duration_ms = int((monotonic() - started_at) * 1000)

    command_path = artifact_dir / "command.txt"
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    result_path = artifact_dir / "result.json"
    command_path.write_text(command_text + "\n", encoding="utf-8")
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    result_path.write_text(
        json.dumps(
            {
                "label": resolved_label,
                "command": command_text,
                "argv": command_list,
                "returncode": result.returncode,
                "cwd": str(resolved_cwd),
                "duration_ms": duration_ms,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    record_activity(
        root,
        "external_command",
        "completed" if result.returncode == 0 else "failed",
        event_id=event_id,
        command=command_text,
        argv=command_list,
        metadata={
            "label": resolved_label,
            "artifact_dir": str(artifact_dir),
            "command_path": str(command_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_path": str(result_path),
            "returncode": result.returncode,
        },
        cwd=resolved_cwd,
        duration_ms=duration_ms,
    )
    return {
        "label": resolved_label,
        "artifact_dir": str(artifact_dir),
        "command_path": str(command_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_path": str(result_path),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": duration_ms,
    }


def read_activity_events(root: str | Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for path in sorted(_activity_dir(root).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events
