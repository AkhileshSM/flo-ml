"""Downstream consumer of the FLO-ML churn model (MLflow registry or a local joblib)."""

from client.churn import ChurnClient, Score

__all__ = ["ChurnClient", "Score"]
