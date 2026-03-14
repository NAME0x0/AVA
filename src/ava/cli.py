from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from time import monotonic

from ava.activity import record_activity, render_command, run_logged_command, snapshot_repo_state
from ava.benchmarks import serialize_benchmark_registry
from ava.config import load_experiment_config
from ava.inspect import trace_checkpoint
from ava.sessions import (
    bootstrap_session,
    create_session,
    hf_research_session,
    inspection_session,
    moe_feasibility_session,
    sota_session,
    training_session,
)
from ava.train import dry_run_summary, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ava")
    subparsers = parser.add_subparsers(dest="command", required=True)

    session_parser = subparsers.add_parser("session")
    session_subparsers = session_parser.add_subparsers(dest="session_command", required=True)

    session_init = session_subparsers.add_parser("init")
    session_init.add_argument("name")
    session_init.add_argument("--root", default="sessions")

    session_bootstrap = session_subparsers.add_parser("bootstrap")
    session_bootstrap.add_argument("name")
    session_bootstrap.add_argument("--root", default="sessions")

    session_sota = session_subparsers.add_parser("sota")
    session_sota.add_argument("name")
    session_sota.add_argument("--root", default="sessions")

    session_hf = session_subparsers.add_parser("hf-research")
    session_hf.add_argument("name")
    session_hf.add_argument("--root", default="sessions")

    session_moe = session_subparsers.add_parser("moe-feasibility")
    session_moe.add_argument("name")
    session_moe.add_argument("--root", default="sessions")

    session_train = session_subparsers.add_parser("train")
    session_train.add_argument("name")
    session_train.add_argument("config")
    session_train.add_argument("corpus_root")
    session_train.add_argument("--root", default="sessions")
    session_train.add_argument("--max-steps", type=int)

    session_inspect = session_subparsers.add_parser("inspect")
    session_inspect.add_argument("name")
    session_inspect.add_argument("checkpoint")
    session_inspect.add_argument("--prompt", required=True)
    session_inspect.add_argument("--root", default="sessions")
    session_inspect.add_argument("--device", default="cuda")
    session_inspect.add_argument("--max-new-tokens", type=int, default=16)
    session_inspect.add_argument("--top-k-neurons", type=int, default=8)
    session_inspect.add_argument("--top-k-logits", type=int, default=8)
    session_inspect.add_argument("--top-k-attention", type=int, default=4)

    train_parser = subparsers.add_parser("train")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)

    dry_run = train_subparsers.add_parser("dry-run")
    dry_run.add_argument("config")

    run = train_subparsers.add_parser("run")
    run.add_argument("config")
    run.add_argument("corpus_root")
    run.add_argument("--max-steps", type=int, default=1000)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_subparsers = inspect_parser.add_subparsers(dest="inspect_command", required=True)

    inspect_checkpoint = inspect_subparsers.add_parser("checkpoint")
    inspect_checkpoint.add_argument("checkpoint")
    inspect_checkpoint.add_argument("--prompt", required=True)
    inspect_checkpoint.add_argument("--device", default="cuda")
    inspect_checkpoint.add_argument("--max-new-tokens", type=int, default=16)
    inspect_checkpoint.add_argument("--top-k-neurons", type=int, default=8)
    inspect_checkpoint.add_argument("--top-k-logits", type=int, default=8)
    inspect_checkpoint.add_argument("--top-k-attention", type=int, default=4)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)

    benchmark_registry = benchmark_subparsers.add_parser("registry")
    benchmark_registry.add_argument("--modality", default="all")
    benchmark_registry.add_argument("--stage", default="all")

    activity_parser = subparsers.add_parser("activity")
    activity_subparsers = activity_parser.add_subparsers(dest="activity_command", required=True)

    activity_snapshot = activity_subparsers.add_parser("snapshot")
    activity_snapshot.add_argument("--root", default="sessions")
    activity_snapshot.add_argument("--label", default="manual")

    activity_run = activity_subparsers.add_parser("run")
    activity_run.add_argument("--root", default="sessions")
    activity_run.add_argument("--label")
    activity_run.add_argument("command_args", nargs=argparse.REMAINDER)

    return parser


def _activity_root(args: argparse.Namespace) -> Path:
    if hasattr(args, "root"):
        return Path(args.root)
    return Path("sessions")


def _normalize_command_args(command_args: list[str]) -> list[str]:
    if command_args and command_args[0] == "--":
        return command_args[1:]
    return command_args


def _dispatch(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    if args.command == "session" and args.session_command == "init":
        session_dir = create_session(Path(args.root), args.name)
        print(f"Created session: {session_dir}")
        return 0, {
            "session_dir": str(session_dir),
            "session_kind": "init",
            "session_name": args.name,
        }

    if args.command == "session" and args.session_command == "bootstrap":
        session_dir = bootstrap_session(Path(args.root), args.name)
        print(f"Bootstrapped session: {session_dir}")
        print(f"Notes: {session_dir / 'notes.md'}")
        return 0, {
            "session_dir": str(session_dir),
            "session_kind": "bootstrap",
            "session_name": args.name,
        }

    if args.command == "session" and args.session_command == "sota":
        session_dir = sota_session(Path(args.root), args.name)
        print(f"SOTA session: {session_dir}")
        print(f"Notes: {session_dir / 'notes.md'}")
        return 0, {
            "session_dir": str(session_dir),
            "session_kind": "sota",
            "session_name": args.name,
        }

    if args.command == "session" and args.session_command == "hf-research":
        session_dir = hf_research_session(Path(args.root), args.name)
        print(f"HF research session: {session_dir}")
        print(f"Notes: {session_dir / 'notes.md'}")
        return 0, {
            "session_dir": str(session_dir),
            "session_kind": "hf-research",
            "session_name": args.name,
        }

    if args.command == "session" and args.session_command == "moe-feasibility":
        session_dir = moe_feasibility_session(Path(args.root), args.name)
        print(f"MoE feasibility session: {session_dir}")
        print(f"Notes: {session_dir / 'notes.md'}")
        return 0, {
            "session_dir": str(session_dir),
            "session_kind": "moe-feasibility",
            "session_name": args.name,
        }

    if args.command == "session" and args.session_command == "train":
        session_dir = training_session(
            Path(args.root),
            args.name,
            args.config,
            args.corpus_root,
            max_steps=args.max_steps,
        )
        print(f"Training session: {session_dir}")
        print(f"Notes: {session_dir / 'notes.md'}")
        return 0, {
            "session_dir": str(session_dir),
            "session_kind": "training",
            "session_name": args.name,
            "config": args.config,
            "corpus_root": args.corpus_root,
        }

    if args.command == "session" and args.session_command == "inspect":
        session_dir = inspection_session(
            Path(args.root),
            args.name,
            args.checkpoint,
            args.prompt,
            requested_device=args.device,
            max_new_tokens=args.max_new_tokens,
            top_k_neurons=args.top_k_neurons,
            top_k_logits=args.top_k_logits,
            top_k_attention=args.top_k_attention,
        )
        print(f"Inspection session: {session_dir}")
        print(f"Notes: {session_dir / 'notes.md'}")
        return 0, {
            "session_dir": str(session_dir),
            "session_kind": "inspection",
            "session_name": args.name,
            "checkpoint": args.checkpoint,
        }

    if args.command == "train" and args.train_command == "dry-run":
        payload = dry_run_summary(load_experiment_config(args.config))
        print(json.dumps(payload, indent=2))
        return 0, {"train_command": "dry-run", "config": args.config}

    if args.command == "train" and args.train_command == "run":
        result = run_training(args.config, args.corpus_root, max_steps=args.max_steps)
        print(json.dumps(result, indent=2))
        return 0, {
            "train_command": "run",
            "config": args.config,
            "corpus_root": args.corpus_root,
            "checkpoint": result.get("checkpoint"),
        }

    if args.command == "inspect" and args.inspect_command == "checkpoint":
        payload = trace_checkpoint(
            args.checkpoint,
            args.prompt,
            requested_device=args.device,
            max_new_tokens=args.max_new_tokens,
            top_k_neurons=args.top_k_neurons,
            top_k_logits=args.top_k_logits,
            top_k_attention=args.top_k_attention,
        )
        print(json.dumps(payload, indent=2))
        return 0, {
            "inspect_command": "checkpoint",
            "checkpoint": args.checkpoint,
            "device": args.device,
        }

    if args.command == "benchmark" and args.benchmark_command == "registry":
        payload = serialize_benchmark_registry(modality=args.modality, stage=args.stage)
        print(json.dumps(payload, indent=2))
        return 0, {
            "benchmark_command": "registry",
            "modality": args.modality,
            "stage": args.stage,
            "rows": len(payload),
        }

    if args.command == "activity" and args.activity_command == "snapshot":
        snapshot_path = snapshot_repo_state(Path(args.root), label=args.label)
        print(f"Snapshot: {snapshot_path}")
        return 0, {
            "activity_command": "snapshot",
            "snapshot_path": str(snapshot_path),
            "label": args.label,
        }

    if args.command == "activity" and args.activity_command == "run":
        result = run_logged_command(Path(args.root), args.command_args, label=args.label)
        if result["stdout"]:
            sys.stdout.write(str(result["stdout"]))
        if result["stderr"]:
            sys.stderr.write(str(result["stderr"]))
        return int(result["returncode"]), {
            "activity_command": "run",
            "label": result["label"],
            "artifact_dir": result["artifact_dir"],
            "returncode": result["returncode"],
        }

    raise RuntimeError("unknown command")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "activity" and args.activity_command == "run":
        args.command_args = _normalize_command_args(args.command_args)
        if not args.command_args:
            parser.error("activity run requires a command after --")

    activity_root = _activity_root(args)
    command_argv = ["ava", *sys.argv[1:]]
    event_id = uuid.uuid4().hex
    record_activity(
        activity_root,
        "cli_command",
        "started",
        event_id=event_id,
        command=render_command(command_argv),
        argv=command_argv,
        metadata={"command": args.command},
    )

    started_at = monotonic()
    try:
        exit_code, metadata = _dispatch(args)
    except Exception as exc:
        record_activity(
            activity_root,
            "cli_command",
            "failed",
            event_id=event_id,
            command=render_command(command_argv),
            argv=command_argv,
            metadata={"command": args.command, "error": repr(exc)},
            duration_ms=int((monotonic() - started_at) * 1000),
        )
        raise

    record_activity(
        activity_root,
        "cli_command",
        "completed" if exit_code == 0 else "failed",
        event_id=event_id,
        command=render_command(command_argv),
        argv=command_argv,
        metadata={"command": args.command, **metadata},
        duration_ms=int((monotonic() - started_at) * 1000),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
