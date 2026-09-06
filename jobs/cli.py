"""Thin 5-command CLI: up, train, ui, promote, down."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "jobs"))

COMPOSE = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "--project-directory", str(ROOT)]


def cmd_up(_: argparse.Namespace) -> int:
    """docker compose up --build -d for Floci, MLflow, and the console."""
    return subprocess.call([*COMPOSE, "up", "-d", "--build"], cwd=ROOT)


def cmd_down(_: argparse.Namespace) -> int:
    """Stop the local compose stack (volumes are kept)."""
    return subprocess.call([*COMPOSE, "down"], cwd=ROOT)


def cmd_train(args: argparse.Namespace) -> int:
    """Delegate to jobs/estimator.py with --env and --instance."""
    estimator = ROOT / "jobs" / "estimator.py"
    cmd = [sys.executable, str(estimator), "--env", args.env, "--instance", args.instance]
    if args.experiment:
        cmd.extend(["--experiment", args.experiment])
    return subprocess.call(cmd, cwd=ROOT)


def cmd_ui(args: argparse.Namespace) -> int:
    """Print console / MLflow URLs; --open launches the console in a browser."""
    console = os.environ.get("FLO_ML_CONSOLE_URL", "http://localhost:8088")
    mlflow = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"FLO-ML console: {console}")
    print(f"MLflow UI:      {mlflow}")
    print("Floci S3:       http://localhost:4566")
    if args.open:
        webbrowser.open(console)
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """Alias the latest registered model as staging|production; Production is blocked locally."""
    sys.path.insert(0, str(SRC))
    from config import apply_runtime_env, load_config

    cfg = load_config(args.env)
    apply_runtime_env(cfg, in_container=False)
    if args.stage.lower() == "production" and not cfg.get("allow_production_register"):
        raise SystemExit("local runs never silently become production — set ENV_NAME != local")

    import mlflow
    from mlflow.tracking import MlflowClient

    tracking = str(cfg.get("mlflow_tracking_uri"))
    mlflow.set_tracking_uri(tracking)
    client = MlflowClient(tracking_uri=tracking)
    name = args.name or cfg.get("model_name") or "churn"
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(f"no registered versions for {name}")
    latest = max(versions, key=lambda v: int(v.version))
    alias = args.stage.lower()
    try:
        client.set_registered_model_alias(name, alias, latest.version)
    except Exception:
        client.transition_model_version_stage(
            name=name,
            version=latest.version,
            stage=args.stage,
            archive_existing_versions=args.stage.lower() == "production",
        )
    print(json.dumps({"name": name, "version": latest.version, "stage": args.stage, "alias": alias, "run_id": latest.run_id}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """flo-ml up|down|train|ui|promote subcommands."""
    parser = argparse.ArgumentParser(prog="flo-ml", description="FLO-ML local-first MLOps CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("up", help="Start Floci + MLflow + bootstrap").set_defaults(func=cmd_up)
    sub.add_parser("down", help="Stop the local stack").set_defaults(func=cmd_down)

    train = sub.add_parser("train", help="Run the sample sklearn job")
    train.add_argument("--env", default=os.environ.get("ENV_NAME", "local"))
    train.add_argument("--instance", default=os.environ.get("SAGEMAKER_INSTANCE", "local"))
    train.add_argument("--experiment", default=None)
    train.set_defaults(func=cmd_train)

    ui = sub.add_parser("ui", help="Print (or open) the MLflow UI")
    ui.add_argument("--open", action="store_true")
    ui.set_defaults(func=cmd_ui)

    promote = sub.add_parser("promote", help="Move the latest registered model to a stage")
    promote.add_argument("--name", default=None)
    promote.add_argument("--stage", default="Staging", choices=["Staging", "Production", "Archived"])
    promote.add_argument("--env", default=os.environ.get("ENV_NAME", "local"))
    promote.set_defaults(func=cmd_promote)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse argv and run the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
