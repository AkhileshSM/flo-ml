"""Optional scoring handler: load the joblib pipeline and predict churn probability."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd


def load_model(model_dir: str):
    """Load the fitted sklearn pipeline from SM_MODEL_DIR/model.joblib."""
    return joblib.load(Path(model_dir) / "model.joblib")


def predict(pipe, records: list[dict]) -> list[dict]:
    """Score feature dicts; returns P(churn) and a 0.5-threshold label per row."""
    frame = pd.DataFrame.from_records(records)
    proba = pipe.predict_proba(frame)[:, 1]
    return [{"churn_probability": float(p), "churn": int(p >= 0.5)} for p in proba]


def main() -> int:
    """CLI: --model-dir plus --input JSON list of feature objects."""
    parser = argparse.ArgumentParser(description="FLO-ML infer")
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--input", required=True, help="JSON list of feature objects")
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text())
    pipe = load_model(args.model_dir)
    print(json.dumps(predict(pipe, records), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
