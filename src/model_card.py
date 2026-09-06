"""Human-readable description of the sample churn task.

This is not a production credit-risk or telco dataset. It is a small, synthetic
fixture so the FLO-ML pipeline can be exercised without copying customer data
onto a laptop.
"""

from __future__ import annotations

FEATURES = [
    {
        "name": "tenure_months",
        "meaning": "How long the account has been open (1–72 months).",
        "signal": "Shorter tenure tends to churn more.",
    },
    {
        "name": "monthly_charges",
        "meaning": "Recurring monthly bill.",
        "signal": "Higher bills with weak tenure often churn.",
    },
    {
        "name": "total_charges",
        "meaning": "Approximate lifetime spend (tenure × monthly × 0.85).",
        "signal": "Derived from tenure and monthly charges.",
    },
    {
        "name": "contract_month_to_month",
        "meaning": "1 if the customer is on a month-to-month contract.",
        "signal": "No term commitment is a classic churn flag.",
    },
    {
        "name": "has_fiber",
        "meaning": "1 if the account has a fiber-like high-speed product.",
        "signal": "Product mix; not a guarantee of stay-or-leave.",
    },
    {
        "name": "support_tickets",
        "meaning": "Count of recent support contacts (0–8).",
        "signal": "More tickets → more friction → more churn.",
    },
    {
        "name": "late_payments",
        "meaning": "Count of late payments (0–6).",
        "signal": "Payment stress correlates with leaving.",
    },
    {
        "name": "addon_count",
        "meaning": "Number of add-on products (0–5).",
        "signal": "More attach often means stickier accounts.",
    },
]

TARGET = {
    "name": "churn",
    "meaning": "1 = the customer left in the observation window, 0 = they stayed.",
    "prevalence": "About 28% positives (class weight 0.28 in the generator).",
}

DATA = {
    "kind": "synthetic telco-style churn table",
    "generator": "sklearn.datasets.make_classification, then rescaled into business-looking columns",
    "rows": "500 train / 150 valid, stratified on churn",
    "seed": 42,
    "location": "data/fixtures/churn_{train,valid}.csv → s3://datasets/churn/{train,valid}/",
    "not": "Not production data. Do not treat metrics as a real business lift.",
}

ALGORITHM = {
    "name": "HistGradientBoostingClassifier",
    "library": "scikit-learn",
    "why": (
        "Histogram gradient boosting is a strong default for small/medium tabular "
        "data: it handles mixed numeric features, trains fast on CPU, and does not "
        "need a GPU. It is the sklearn successor to LightGBM-style boosting."
    ),
    "pipeline": [
        "median impute missing numerics",
        "standard-scale all columns",
        "boosted trees (max_depth, learning_rate, max_iter, l2_regularization)",
    ],
    "defaults": {
        "max_depth": 3,
        "learning_rate": 0.08,
        "max_iter": 80,
        "l2_regularization": 0.1,
        "random_state": 42,
    },
    "decision_threshold": 0.5,
}

QUALITY = {
    "data_checks": [
        "train and valid share the same feature columns",
        "target is binary 0/1",
        "both classes appear in each split",
        "no feature is entirely missing",
    ],
    "holdout_metrics": [
        "valid_roc_auc — ranking quality (primary gate)",
        "valid_average_precision — precision-recall under class imbalance",
        "valid_accuracy — 0.5-threshold accuracy (secondary; imbalance-sensitive)",
        "confusion matrix PNG logged to MLflow",
    ],
    "gate": {
        "metric": "valid_roc_auc",
        "minimum": 0.70,
        "env": "FLO_ML_MIN_ROC_AUC",
        "on_fail": "training exits non-zero and the model is not treated as shippable",
    },
    "lineage": [
        "data S3 URI + etag",
        "git SHA",
        "training image digest",
        "MLflow model signature + input example",
    ],
}


def as_dict() -> dict:
    """JSON payload for the console model-card panel."""
    return {
        "data": DATA,
        "target": TARGET,
        "features": FEATURES,
        "algorithm": ALGORITHM,
        "quality": QUALITY,
    }
