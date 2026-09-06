"""Load a registered churn model and score customers.

This is the application-side client. Training lives in src/train.py; this module
only *consumes* a model from the MLflow registry (or a local joblib file).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.model_card import ALGORITHM, FEATURES

FEATURE_NAMES: tuple[str, ...] = tuple(f["name"] for f in FEATURES)
THRESHOLD = float(ALGORITHM.get("decision_threshold") or 0.5)
HIGH_RISK = 0.70


@dataclass(frozen=True)
class Score:
    """One scored account: probability, 0/1 label, and a suggested next action."""

    churn_probability: float
    churn: int
    label: str
    action: str
    action_reason: str
    model_uri: str
    model_version: str
    features: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready payload for HTTP/CLI."""
        return asdict(self)


def recommend_action(probability: float) -> tuple[str, str]:
    """Map P(churn) to a retention-desk action. Thresholds are product policy, not the model."""
    if probability >= HIGH_RISK:
        return (
            "offer_retention",
            f"P(churn) {probability:.0%} ≥ {HIGH_RISK:.0%}: high risk — outreach / save offer.",
        )
    if probability >= THRESHOLD:
        return (
            "monitor",
            f"P(churn) {probability:.0%} ≥ {THRESHOLD:.0%}: elevated — watch the next bill cycle.",
        )
    return (
        "no_action",
        f"P(churn) {probability:.0%} < {THRESHOLD:.0%}: likely to stay.",
    )


def features_to_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame in training column order; missing keys become errors."""
    if not records:
        raise ValueError("no records to score")
    rows = []
    for i, rec in enumerate(records):
        missing = [name for name in FEATURE_NAMES if name not in rec]
        if missing:
            raise ValueError(f"record {i} missing features: {missing}")
        rows.append({name: rec[name] for name in FEATURE_NAMES})
    return pd.DataFrame(rows, columns=list(FEATURE_NAMES))


class ChurnClient:
    """Consumer of models:/churn/<version|alias> (or a joblib pipeline on disk)."""

    def __init__(
        self,
        pipe: Any,
        *,
        model_uri: str,
        model_version: str,
        threshold: float = THRESHOLD,
    ) -> None:
        self._pipe = pipe
        self.model_uri = model_uri
        self.model_version = str(model_version)
        self.threshold = threshold

    @classmethod
    def from_pipeline(cls, pipe: Any, *, model_uri: str = "memory", model_version: str = "local") -> "ChurnClient":
        """Wrap an already-fitted sklearn pipeline (used in tests and offline scoring)."""
        return cls(pipe, model_uri=model_uri, model_version=model_version)

    @classmethod
    def from_joblib(cls, model_dir: str | Path) -> "ChurnClient":
        """Load SM_MODEL_DIR/model.joblib — same file the trainer writes."""
        import joblib

        path = Path(model_dir)
        file = path if path.is_file() else path / "model.joblib"
        pipe = joblib.load(file)
        return cls(pipe, model_uri=str(file), model_version="joblib")

    @classmethod
    def from_registry(
        cls,
        *,
        tracking_uri: str | None = None,
        name: str = "churn",
        stage: str = "staging",
    ) -> "ChurnClient":
        """Load the MLflow registered model. Prefers alias `stage`, else latest version."""
        import mlflow
        from mlflow.tracking import MlflowClient

        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        mlflow.set_tracking_uri(uri)
        store = MlflowClient(tracking_uri=uri)
        version, model_uri = _resolve_registered(store, name, stage)
        pipe = mlflow.sklearn.load_model(model_uri)
        return cls(pipe, model_uri=model_uri, model_version=str(version))

    def score_one(self, features: dict[str, Any]) -> Score:
        """Score a single customer feature dict."""
        return self.score_many([features])[0]

    def score_many(self, records: Sequence[dict[str, Any]]) -> list[Score]:
        """Score a batch; returns one Score per input row in the same order."""
        frame = features_to_frame(records)
        proba = self._pipe.predict_proba(frame)[:, 1]
        out: list[Score] = []
        for rec, p in zip(records, proba):
            p = float(p)
            label_bit = int(p >= self.threshold)
            action, reason = recommend_action(p)
            out.append(
                Score(
                    churn_probability=round(p, 4),
                    churn=label_bit,
                    label="likely to churn" if label_bit else "likely to stay",
                    action=action,
                    action_reason=reason,
                    model_uri=self.model_uri,
                    model_version=self.model_version,
                    features={name: float(rec[name]) for name in FEATURE_NAMES},
                )
            )
        return out

    def score_csv(self, path: str | Path) -> list[Score]:
        """Score every row of a CSV that has the training feature columns (churn col optional)."""
        frame = pd.read_csv(path)
        records = frame.drop(columns=["churn"], errors="ignore").to_dict(orient="records")
        return self.score_many(records)

    def describe(self) -> dict[str, Any]:
        """Identity of the loaded model for /v1/model and CLI `describe`."""
        return {
            "model_uri": self.model_uri,
            "model_version": self.model_version,
            "features": list(FEATURE_NAMES),
            "threshold": self.threshold,
            "high_risk": HIGH_RISK,
            "algorithm": ALGORITHM["name"],
        }


def _resolve_registered(store, name: str, stage: str) -> tuple[str, str]:
    """Pick alias `stage` if it exists, otherwise the highest numeric version."""
    alias = stage.lower()
    try:
        mv = store.get_model_version_by_alias(name, alias)
        return str(mv.version), f"models:/{name}/{mv.version}"
    except Exception:
        pass
    versions = list(store.search_model_versions(f"name='{name}'"))
    if not versions:
        raise FileNotFoundError(
            f"no registered model '{name}'. Train and register first "
            "(console Train, or python jobs/estimator.py --instance process)."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return str(latest.version), f"models:/{name}/{latest.version}"


def iter_scores(scores: Iterable[Score]) -> list[dict[str, Any]]:
    """Convert Score objects to dicts for JSON dumps."""
    return [s.as_dict() for s in scores]
