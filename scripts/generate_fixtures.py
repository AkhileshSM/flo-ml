#!/usr/bin/env python3
"""Generate the tiny anonymized churn fixture used by the local inner loop."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fixtures"
COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_month_to_month",
    "has_fiber",
    "support_tickets",
    "late_payments",
    "addon_count",
]


def main() -> None:
    """Build 500/150 stratified train/valid CSVs from make_classification (seed 42)."""
    x, y = make_classification(
        n_samples=650,
        n_features=len(COLUMNS),
        n_informative=5,
        n_redundant=1,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.72, 0.28],
        class_sep=1.2,
        random_state=42,
    )
    frame = pd.DataFrame(x, columns=COLUMNS)
    frame["tenure_months"] = (frame["tenure_months"].rank(pct=True) * 72).clip(1, 72).round(0)
    frame["monthly_charges"] = (45 + 35 * frame["monthly_charges"].rank(pct=True)).round(2)
    frame["total_charges"] = (frame["tenure_months"] * frame["monthly_charges"] * 0.85).round(2)
    frame["contract_month_to_month"] = (frame["contract_month_to_month"] > 0).astype(int)
    frame["has_fiber"] = (frame["has_fiber"] > 0).astype(int)
    frame["support_tickets"] = (frame["support_tickets"].rank(pct=True) * 8).round(0).astype(int)
    frame["late_payments"] = (frame["late_payments"].rank(pct=True) * 6).round(0).astype(int)
    frame["addon_count"] = (frame["addon_count"].rank(pct=True) * 5).round(0).astype(int)
    frame["churn"] = y.astype(int)

    train, valid = train_test_split(frame, test_size=0.23, random_state=42, stratify=frame["churn"])
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "churn_train.csv", index=False)
    valid.to_csv(OUT / "churn_valid.csv", index=False)
    print(f"wrote {len(train)} train / {len(valid)} valid rows to {OUT}")


if __name__ == "__main__":
    main()
