"""Processing-style evaluation against a saved joblib pipeline.

Used as a SageMaker Processing-shaped step: load the model, score the
validation channel, write metrics JSON + a tiny HTML report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from train import TARGET, _first_csv


def parse_args() -> argparse.Namespace:
    """CLI flags matching SageMaker Processing channel / output env vars."""
    parser = argparse.ArgumentParser(description="FLO-ML evaluate")
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument(
        "--validation",
        default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    return parser.parse_args()


def score_holdout(pipe, frame: pd.DataFrame) -> dict:
    """Compute ROC-AUC, average precision, and accuracy on a labeled holdout frame."""
    y = frame[TARGET].astype(int)
    x = frame.drop(columns=[TARGET])
    proba = pipe.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "holdout_roc_auc": float(roc_auc_score(y, proba)),
        "holdout_average_precision": float(average_precision_score(y, proba)),
        "holdout_accuracy": float(accuracy_score(y, pred)),
        "n_rows": int(len(frame)),
    }


def write_report(metrics: dict, output_dir: Path) -> None:
    """Persist eval_metrics.json and a one-page HTML table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    html = [
        "<html><body><h1>FLO-ML evaluation</h1><table>",
        *(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in metrics.items()),
        "</table></body></html>",
    ]
    (output_dir / "eval_report.html").write_text("\n".join(html))


def main() -> int:
    """Load model.joblib, score the validation CSV, write artifacts, print JSON."""
    args = parse_args()
    pipe = joblib.load(Path(args.model_dir) / "model.joblib")
    frame = pd.read_csv(_first_csv(args.validation))
    metrics = score_holdout(pipe, frame)
    write_report(metrics, Path(args.output_dir))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
