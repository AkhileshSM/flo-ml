"""HTTP scoring service — the shape a billing/CRM app would call.

Loads the MLflow model lazily so the container can start before the first train.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from client.churn import ChurnClient, iter_scores

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="FLO-ML churn client", version="1.0")
_client: ChurnClient | None = None
_client_error: str | None = None


def get_client() -> ChurnClient:
    """Return a cached ChurnClient; reload after a failed first attempt."""
    global _client, _client_error
    if _client is not None:
        return _client
    try:
        _client = ChurnClient.from_registry(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            name=os.environ.get("MODEL_NAME", "churn"),
            stage=os.environ.get("MODEL_STAGE", "staging"),
        )
        _client_error = None
        return _client
    except Exception as exc:  # noqa: BLE001
        _client_error = str(exc)
        raise HTTPException(
            503,
            "Model is not loaded yet. Train (and optionally Promote to Staging) in the console, then retry. "
            f"Detail: {_client_error}",
        ) from exc


@app.get("/")
def index() -> FileResponse:
    """Small scoring UI that posts to /v1/score."""
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus whether a model has been loaded."""
    loaded = _client is not None
    if not loaded:
        try:
            get_client()
            loaded = True
        except HTTPException:
            loaded = False
    return {
        "status": "ok" if loaded else "waiting_for_model",
        "loaded": loaded,
        "error": _client_error,
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    }


@app.get("/v1/model")
def model_info() -> dict[str, Any]:
    """Identity of the model this client is consuming."""
    return get_client().describe()


@app.post("/v1/score")
def score_one(body: dict[str, Any]) -> dict[str, Any]:
    """Score one customer. Body is a feature object, or {\"features\": {...}}."""
    features = body["features"] if isinstance(body.get("features"), dict) and "tenure_months" not in body else body
    try:
        return get_client().score_one(features).as_dict()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/score/batch")
async def score_batch(request: Request) -> dict[str, Any]:
    """Score many customers; body is a JSON list or {\"records\": [...]}."""
    body = await request.json()
    records = body if isinstance(body, list) else (body or {}).get("records") or []
    try:
        scores = get_client().score_many(records)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"n": len(scores), "results": iter_scores(scores)}


@app.post("/v1/reload")
def reload_model() -> dict[str, Any]:
    """Drop the cached model so the next request picks up a newly registered version."""
    global _client, _client_error
    _client = None
    _client_error = None
    return get_client().describe()


def main() -> None:
    """Run uvicorn on PORT (default 8090)."""
    import uvicorn

    uvicorn.run("client.service:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8090")))


if __name__ == "__main__":
    main()
