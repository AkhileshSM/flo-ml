"""SageMaker-shaped training entry point for the sample churn classifier.

Reads SM_CHANNEL_* / SM_MODEL_DIR (or CLI flags), trains a
HistGradientBoostingClassifier, runs data + metric quality checks, logs the
run to MLflow, and writes the pipeline to SM_MODEL_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "churn"
DEFAULT_MIN_ROC_AUC = 0.70


def _channel(name: str, default: str) -> str:
    """Resolve a SageMaker input channel path from SM_CHANNEL_* or a default."""
    env_key = f"SM_CHANNEL_{name.upper()}"
    return os.environ.get(env_key, default)


def _first_csv(directory: str) -> Path:
    """Return the first CSV under a file or directory (SageMaker channel layout)."""
    root = Path(directory)
    if root.is_file():
        return root
    csvs = sorted(root.glob("**/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no csv under {directory}")
    return csvs[0]


def resolve_channel(directory: str, s3_uri: str | None) -> str:
    """Use a local SageMaker channel if present, otherwise pull the CSV from Floci/S3."""
    try:
        return str(_first_csv(directory))
    except FileNotFoundError:
        pass
    if not s3_uri:
        raise FileNotFoundError(
            f"no csv under {directory} and no S3 uri set (FLO_ML_DATA_URI / FLO_ML_VALID_URI)"
        )
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    from aws_session import download_prefix

    written = download_prefix(s3_uri, str(dest))
    if not written:
        raise FileNotFoundError(f"no objects at {s3_uri}")
    return str(_first_csv(str(dest)))


def load_xy(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load features X and binary churn label y from a CSV path or channel dir."""
    frame = pd.read_csv(_first_csv(path))
    if TARGET not in frame.columns:
        raise ValueError(f"expected `{TARGET}` column in {path}")
    y = frame[TARGET].astype(int)
    x = frame.drop(columns=[TARGET])
    return x, y


def profile_split(x: pd.DataFrame, y: pd.Series, split: str) -> dict:
    """Row counts, class balance, and missingness for one train/valid split."""
    return {
        f"{split}_n_rows": int(len(x)),
        f"{split}_churn_rate": float(y.mean()) if len(y) else 0.0,
        f"{split}_n_features": int(x.shape[1]),
        f"{split}_missing_cells": int(x.isna().sum().sum()),
    }


def assert_data_quality(
    x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series
) -> None:
    """Fail the job if the tables are not trainable (schema, target, empty features)."""
    if list(x_train.columns) != list(x_valid.columns):
        raise ValueError("train/valid feature columns do not match")
    illegal = set(pd.unique(pd.concat([y_train, y_valid], ignore_index=True))) - {0, 1}
    if illegal:
        raise ValueError(f"churn target must be 0/1, saw {sorted(illegal)}")
    if y_train.nunique() < 2 or y_valid.nunique() < 2:
        raise ValueError("both classes must appear in train and validation")
    empty = [c for c in x_train.columns if x_train[c].isna().all()]
    if empty:
        raise ValueError(f"feature(s) entirely missing: {empty}")


def min_roc_auc() -> float:
    """Holdout ROC-AUC floor; override with FLO_ML_MIN_ROC_AUC."""
    return float(os.environ.get("FLO_ML_MIN_ROC_AUC", str(DEFAULT_MIN_ROC_AUC)))


def assert_metric_gate(metrics: dict) -> None:
    """Reject the run if holdout ROC-AUC is below the quality floor."""
    auc = float(metrics["valid_roc_auc"])
    floor = min_roc_auc()
    if auc < floor:
        raise ValueError(
            f"quality gate failed: valid_roc_auc {auc:.3f} < {floor:.3f} "
            "(set FLO_ML_MIN_ROC_AUC to change the floor)"
        )


def build_pipeline(max_depth: int, learning_rate: float, max_iter: int, l2: float) -> Pipeline:
    """Impute + scale all columns, then histogram gradient boosting on CPU."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    model = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_regularization=l2,
        random_state=42,
    )
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[("num", numeric, slice(None))],
                    remainder="drop",
                ),
            ),
            ("clf", model),
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI / SageMaker hyperparameter flags for the churn trainer."""
    parser = argparse.ArgumentParser(description="FLO-ML churn trainer")
    parser.add_argument("--train", default=_channel("train", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", default=_channel("validation", "/opt/ml/input/data/validation"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--max-depth", type=int, default=int(os.environ.get("SM_HP_MAX_DEPTH", "3")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("SM_HP_LEARNING_RATE", "0.08")))
    parser.add_argument("--max-iter", type=int, default=int(os.environ.get("SM_HP_MAX_ITER", "80")))
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=float(os.environ.get("SM_HP_L2_REGULARIZATION", "0.1")),
    )
    parser.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "churn"))
    return parser.parse_args(argv)


def _maybe_mlflow():
    """Connect to the tracking server when MLFLOW_TRACKING_URI is set; else skip logging."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    import mlflow
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "churn"))
    return mlflow, infer_signature


def _log_lineage(mlflow) -> None:
    """Record data URI, image digest, git SHA, and env so a run can be reproduced."""
    params = {
        "data_uri": os.environ.get("FLO_ML_DATA_URI", ""),
        "data_etag": os.environ.get("FLO_ML_DATA_ETAG", ""),
        "image_digest": os.environ.get("FLO_ML_IMAGE_DIGEST", ""),
        "git_sha": os.environ.get("FLO_ML_GIT_SHA", ""),
        "instance_type": os.environ.get("FLO_ML_INSTANCE_TYPE", "local"),
        "env_name": os.environ.get("FLO_ML_ENV_NAME", os.environ.get("ENV_NAME", "local")),
        "model_class": "HistGradientBoostingClassifier",
        "min_roc_auc": min_roc_auc(),
    }
    mlflow.log_params({k: v for k, v in params.items() if v != ""})
    mlflow.set_tags(
        {
            "project": os.environ.get("FLO_ML_PROJECT", "churn"),
            "team": os.environ.get("FLO_ML_TEAM", "mlops"),
            "dataset_version": os.environ.get("FLO_ML_DATA_ETAG", "fixture"),
            "approved_for_cloud": os.environ.get("FLO_ML_APPROVED_FOR_CLOUD", "false"),
        }
    )


def _confusion_png(y_true, y_pred, path: Path) -> None:
    """Write a holdout confusion-matrix image used as an MLflow artifact."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, int(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(args: argparse.Namespace) -> dict:
    """Fit the pipeline, enforce quality gates, persist the model, and log the MLflow run."""
    train_path = resolve_channel(args.train, os.environ.get("FLO_ML_DATA_URI"))
    valid_path = resolve_channel(args.validation, os.environ.get("FLO_ML_VALID_URI"))
    x_train, y_train = load_xy(train_path)
    x_valid, y_valid = load_xy(valid_path)
    assert_data_quality(x_train, y_train, x_valid, y_valid)
    data_profile = {
        **profile_split(x_train, y_train, "train"),
        **profile_split(x_valid, y_valid, "valid"),
    }

    pipe = build_pipeline(args.max_depth, args.learning_rate, args.max_iter, args.l2_regularization)

    mlflow_bundle = _maybe_mlflow()
    run_ctx = None
    mlflow = None
    infer_signature = None
    if mlflow_bundle:
        mlflow, infer_signature = mlflow_bundle
        mlflow.sklearn.autolog(log_models=False, silent=True)
        run_ctx = mlflow.start_run(run_name=os.environ.get("FLO_ML_RUN_NAME", "local-train"))

    if run_ctx:
        run_ctx.__enter__()
        _log_lineage(mlflow)
        mlflow.log_params(
            {
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "max_iter": args.max_iter,
                "l2_regularization": args.l2_regularization,
            }
        )
        mlflow.log_metrics({k: v for k, v in data_profile.items() if isinstance(v, (int, float))})

    pipe.fit(x_train, y_train)

    proba = pipe.predict_proba(x_valid)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "valid_roc_auc": float(roc_auc_score(y_valid, proba)),
        "valid_average_precision": float(average_precision_score(y_valid, proba)),
        "valid_accuracy": float(accuracy_score(y_valid, pred)),
        **data_profile,
    }
    assert_metric_gate(metrics)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, model_dir / "model.joblib")
    (model_dir / "feature_names.json").write_text(json.dumps(list(x_train.columns)))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _confusion_png(y_valid, pred, output_dir / "confusion_matrix.png")

    if mlflow:
        mlflow.log_metrics(
            {
                "valid_roc_auc": metrics["valid_roc_auc"],
                "valid_average_precision": metrics["valid_average_precision"],
                "valid_accuracy": metrics["valid_accuracy"],
            }
        )
        mlflow.log_artifact(str(output_dir / "metrics.json"))
        mlflow.log_artifact(str(output_dir / "confusion_matrix.png"))
        signature = infer_signature(x_train.head(5), pipe.predict(x_train.head(5)))
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            input_example=x_train.head(3),
            registered_model_name=os.environ.get("FLO_ML_REGISTER_MODEL") or None,
        )
        run_id = mlflow.active_run().info.run_id
        metrics["mlflow_run_id"] = run_id
        (output_dir / "run.json").write_text(json.dumps({"run_id": run_id}, indent=2))
        run_ctx.__exit__(None, None, None)

    prefix = os.environ.get("FLO_ML_MODEL_PREFIX", "").rstrip("/")
    run_id = metrics.get("mlflow_run_id") or "local"
    if prefix.startswith("s3://") and (model_dir / "model.joblib").exists():
        from aws_session import upload_file

        uploaded = upload_file(str(model_dir / "model.joblib"), f"{prefix}/{run_id}/model.joblib")
        metrics["model_uri"] = uploaded
        print(f"uploaded {uploaded}", flush=True)

    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Parse argv and run one training job; returns a process exit code."""
    args = parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
