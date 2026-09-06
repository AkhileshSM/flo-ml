#!/usr/bin/env bash
# CI / laptop smoke: stack is up, buckets exist, a local train produces an MLflow run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:4566}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_EC2_METADATA_DISABLED=true
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export ENV_NAME=local

echo "==> aws s3 ls"
aws s3 ls --endpoint-url "$AWS_ENDPOINT_URL"
aws s3 ls s3://datasets --endpoint-url "$AWS_ENDPOINT_URL"
aws s3 ls s3://mlflow-artifacts --endpoint-url "$AWS_ENDPOINT_URL"
aws s3 ls s3://models --endpoint-url "$AWS_ENDPOINT_URL"

echo "==> mlflow health"
curl -fsS "$MLFLOW_TRACKING_URI/health" >/dev/null

echo "==> train (process mode)"
python3 "$ROOT/jobs/estimator.py" --env local --instance process --experiment churn

echo "==> assert MLflow run"
python3 - <<'PY'
import json, os, urllib.request
base = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
req = urllib.request.Request(
    base + "/api/2.0/mlflow/experiments/get-by-name?experiment_name=churn"
)
with urllib.request.urlopen(req) as resp:
    body = json.load(resp)
exp_id = body["experiment"]["experiment_id"]
payload = json.dumps({"experiment_ids": [exp_id], "max_results": 5}).encode()
req = urllib.request.Request(
    base + "/api/2.0/mlflow/runs/search",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    runs = json.load(resp)
assert runs.get("runs"), "expected at least one MLflow run"
print("ok", len(runs["runs"]), "run(s); latest", runs["runs"][0]["info"]["run_id"])
PY

echo "smoke passed"
