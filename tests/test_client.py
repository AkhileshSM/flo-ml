"""Unit tests for the downstream churn client (no MLflow required)."""

from pathlib import Path

import pytest

from client.churn import FEATURE_NAMES, ChurnClient, recommend_action
from src.train import parse_args, train

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures"


@pytest.fixture(scope="module")
def trained_client(tmp_path_factory) -> ChurnClient:
    """Fit the fixture model once per test module."""
    import os

    os.environ.pop("MLFLOW_TRACKING_URI", None)
    work = tmp_path_factory.mktemp("model")
    args = parse_args(
        [
            "--train",
            str(FIXTURES / "churn_train.csv"),
            "--validation",
            str(FIXTURES / "churn_valid.csv"),
            "--model-dir",
            str(work / "model"),
            "--output-dir",
            str(work / "output"),
            "--max-iter",
            "40",
        ]
    )
    train(args)
    return ChurnClient.from_joblib(work / "model")


def test_recommend_action_bands() -> None:
    assert recommend_action(0.81)[0] == "offer_retention"
    assert recommend_action(0.55)[0] == "monitor"
    assert recommend_action(0.12)[0] == "no_action"


def test_score_one_and_batch(trained_client: ChurnClient) -> None:
    high = {
        "tenure_months": 14,
        "monthly_charges": 70.9,
        "total_charges": 843.71,
        "contract_month_to_month": 1,
        "has_fiber": 1,
        "support_tickets": 4,
        "late_payments": 3,
        "addon_count": 0,
    }
    one = trained_client.score_one(high)
    assert 0.0 <= one.churn_probability <= 1.0
    assert one.churn in (0, 1)
    assert one.action in {"offer_retention", "monitor", "no_action"}
    assert set(one.features) == set(FEATURE_NAMES)

    stay = {
        "tenure_months": 13,
        "monthly_charges": 60.99,
        "total_charges": 673.94,
        "contract_month_to_month": 0,
        "has_fiber": 0,
        "support_tickets": 1,
        "late_payments": 1,
        "addon_count": 5,
    }
    batch = trained_client.score_many([high, stay])
    assert len(batch) == 2
    assert batch[0].churn_probability >= batch[1].churn_probability


def test_missing_feature_raises(trained_client: ChurnClient) -> None:
    with pytest.raises(ValueError, match="missing features"):
        trained_client.score_one({"tenure_months": 1})


def test_score_csv(trained_client: ChurnClient) -> None:
    scores = trained_client.score_csv(FIXTURES / "churn_valid.csv")
    assert len(scores) == 150
    assert trained_client.describe()["algorithm"] == "HistGradientBoostingClassifier"
