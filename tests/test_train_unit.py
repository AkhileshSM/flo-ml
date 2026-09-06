import json
from pathlib import Path

import pandas as pd
import pytest

from src.train import (
    assert_data_quality,
    assert_metric_gate,
    parse_args,
    resolve_channel,
    train,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures"


def test_resolve_channel_local_csv() -> None:
    path = resolve_channel(str(FIXTURES / "churn_train.csv"), None)
    assert path.endswith("churn_train.csv")


def test_train_on_fixtures(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    args = parse_args(
        [
            "--train",
            str(FIXTURES / "churn_train.csv"),
            "--validation",
            str(FIXTURES / "churn_valid.csv"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-iter",
            "40",
        ]
    )
    metrics = train(args)
    assert metrics["valid_roc_auc"] > 0.7
    assert (tmp_path / "model" / "model.joblib").exists()
    written = json.loads((tmp_path / "output" / "metrics.json").read_text())
    assert "valid_roc_auc" in written
    names = json.loads((tmp_path / "model" / "feature_names.json").read_text())
    assert "monthly_charges" in names
    assert pd.read_csv(FIXTURES / "churn_train.csv")["churn"].nunique() == 2
    assert metrics["train_n_rows"] == 500
    assert metrics["valid_n_rows"] == 150


def test_metric_gate_rejects_weak_auc(monkeypatch) -> None:
    monkeypatch.setenv("FLO_ML_MIN_ROC_AUC", "0.99")
    with pytest.raises(ValueError, match="quality gate failed"):
        assert_metric_gate({"valid_roc_auc": 0.80})


def test_data_quality_rejects_column_mismatch() -> None:
    x_train = pd.DataFrame({"a": [1, 0], "b": [1, 1]})
    x_valid = pd.DataFrame({"a": [1]})
    y_train = pd.Series([0, 1])
    y_valid = pd.Series([0])
    with pytest.raises(ValueError, match="columns"):
        assert_data_quality(x_train, y_train, x_valid, y_valid)
