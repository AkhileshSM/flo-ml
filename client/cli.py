"""CLI for the churn scoring client: describe, score one row / JSON / CSV."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from client.churn import FEATURE_NAMES, ChurnClient, iter_scores


def _build_client(args: argparse.Namespace) -> ChurnClient:
    """Load from a joblib dir if given, else from the MLflow registry."""
    if args.model_dir:
        return ChurnClient.from_joblib(args.model_dir)
    return ChurnClient.from_registry(
        tracking_uri=args.tracking_uri,
        name=args.name,
        stage=args.stage,
    )


def cmd_describe(args: argparse.Namespace) -> int:
    """Print which model version the client would load."""
    client = _build_client(args)
    print(json.dumps(client.describe(), indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score --json, --csv, or flags matching the training features."""
    client = _build_client(args)
    if args.csv:
        scores = client.score_csv(args.csv)
    elif args.json:
        payload = json.loads(Path(args.json).read_text())
        records = payload if isinstance(payload, list) else payload.get("records") or [payload]
        scores = client.score_many(records)
    else:
        features = {name: getattr(args, name) for name in FEATURE_NAMES}
        if any(v is None for v in features.values()):
            raise SystemExit(
                "pass --json, --csv, or all feature flags: "
                + ", ".join(f"--{n.replace('_', '-')}" for n in FEATURE_NAMES)
            )
        scores = [client.score_one(features)]
    print(json.dumps(iter_scores(scores), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """client describe | score."""
    parser = argparse.ArgumentParser(
        prog="python -m client",
        description="Score accounts with the FLO-ML registered churn model.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    parser.add_argument("--name", default="churn")
    parser.add_argument("--stage", default="staging", help="MLflow alias, default staging")
    parser.add_argument("--model-dir", default=None, help="Optional joblib dir instead of the registry")
    sub = parser.add_subparsers(dest="command", required=True)

    desc = sub.add_parser("describe", help="Show loaded model identity")
    desc.set_defaults(func=cmd_describe)

    score = sub.add_parser("score", help="Score one JSON object, a JSON list, or a CSV")
    score.add_argument("--json", default=None, help="Path to a feature object or list")
    score.add_argument("--csv", default=None, help="CSV with the training columns")
    for name in FEATURE_NAMES:
        score.add_argument(f"--{name.replace('_', '-')}", type=float, default=None)
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse argv and run describe or score."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
