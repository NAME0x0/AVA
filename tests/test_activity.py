import shutil
import sys
from pathlib import Path

from ava.activity import read_activity_events, record_activity, run_logged_command, snapshot_repo_state
from ava.cli import main


def test_record_activity_appends_jsonl() -> None:
    root = Path("sessions") / "test-activity-events"
    if root.exists():
        shutil.rmtree(root)

    record_activity(root, "unit_event", "completed", metadata={"label": "unit"})

    events = read_activity_events(root)
    assert len(events) == 1
    assert events[0]["kind"] == "unit_event"
    assert events[0]["status"] == "completed"
    assert events[0]["metadata"]["label"] == "unit"
    shutil.rmtree(root)


def test_snapshot_repo_state_writes_snapshot_and_event() -> None:
    root = Path("sessions") / "test-activity-snapshot"
    if root.exists():
        shutil.rmtree(root)

    snapshot_path = snapshot_repo_state(root, label="unit-snapshot")

    assert snapshot_path.exists()
    events = read_activity_events(root)
    assert any(event["kind"] == "repo_snapshot" for event in events)
    shutil.rmtree(root)


def test_run_logged_command_captures_output() -> None:
    root = Path("sessions") / "test-activity-run"
    if root.exists():
        shutil.rmtree(root)

    result = run_logged_command(
        root,
        [sys.executable, "-c", "print('activity-run-ok')"],
        label="python-smoke",
    )

    assert result["returncode"] == 0
    assert "activity-run-ok" in result["stdout"]
    assert Path(result["stdout_path"]).exists()
    events = read_activity_events(root)
    assert any(event["kind"] == "external_command" and event["status"] == "completed" for event in events)
    shutil.rmtree(root)


def test_cli_activity_run_logs_cli_and_external_command(monkeypatch, capsys) -> None:
    root = Path("sessions") / "test-activity-cli"
    if root.exists():
        shutil.rmtree(root)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ava",
            "activity",
            "run",
            "--root",
            str(root),
            "--label",
            "cli-unit",
            "--",
            sys.executable,
            "-c",
            "print('cli-activity-ok')",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "cli-activity-ok" in captured.out
    events = read_activity_events(root)
    assert any(event["kind"] == "cli_command" for event in events)
    assert any(event["kind"] == "external_command" for event in events)
    shutil.rmtree(root)
