"""FLO-ML console: status, train, inspect runs, promote, score."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STATIC = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "jobs"))

from aws_session import client as aws_client  # noqa: E402
from client.churn import ChurnClient  # noqa: E402
from config import apply_runtime_env, load_config  # noqa: E402
from model_card import as_dict as model_card_dict  # noqa: E402

PUBLIC_SCORER = os.environ.get("FLO_ML_PUBLIC_SCORER", "http://localhost:8090")
SCORER_HEALTH_URL = os.environ.get("SCORER_HEALTH_URL", "http://localhost:8090/health")

PUBLIC_MLFLOW = os.environ.get("FLO_ML_PUBLIC_MLFLOW", "http://localhost:5000")
PUBLIC_FLOCI = os.environ.get("FLO_ML_PUBLIC_FLOCI", "http://localhost:4566")
TRAIN_IMAGE = os.environ.get("TRAIN_IMAGE", "flo-ml/sklearn-train:local")
COMPOSE_NETWORK = os.environ.get("COMPOSE_NETWORK", "flo-ml")
SAMPLE_FEATURES = {
    "tenure_months": 14,
    "monthly_charges": 70.9,
    "total_charges": 843.71,
    "contract_month_to_month": 1,
    "has_fiber": 1,
    "support_tickets": 4,
    "late_payments": 3,
    "addon_count": 0,
}

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="FLO-ML console", version="1.0")
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


class TrainRequest(BaseModel):
    instance: str = Field("process", description="process | local")
    experiment: str = "churn"
    max_depth: int = 3
    learning_rate: float = 0.08
    max_iter: int = 80
    l2_regularization: float = 0.1


class PromoteRequest(BaseModel):
    name: str = "churn"
    stage: str = "Staging"
    env: str = "local"


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(default_factory=lambda: dict(SAMPLE_FEATURES))
    model_name: str = "churn"


def _cfg(env: str = "local") -> dict[str, Any]:
    """Load YAML config and apply host vs in-container endpoint URLs."""
    cfg = load_config(env)
    in_container = Path("/.dockerenv").exists()
    apply_runtime_env(cfg, in_container=in_container)
    return cfg


def _mlflow_client(cfg: dict[str, Any]):
    """MlflowClient pointed at the tracking URI from env or config."""
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg.get("mlflow_tracking_uri")
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri), uri


def _probe(url: str, timeout: float = 2.5) -> bool:
    """True if url returns HTTP 2xx (used for MLflow /health)."""
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _image_present() -> bool:
    """Whether the SageMaker-shaped training image exists on the Docker daemon."""
    try:
        import docker

        docker.from_env().images.get(TRAIN_IMAGE)
        return True
    except Exception:
        return False


@app.get("/")
def index() -> FileResponse:
    """Serve the console single-page app."""
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    """Health of Floci, MLflow, buckets, and the training image."""
    cfg = _cfg()
    tracking = os.environ.get("MLFLOW_TRACKING_URI") or cfg.get("mlflow_tracking_uri")
    floci_ok = False
    buckets: list[str] = []
    objects: dict[str, int] = {}
    try:
        s3 = aws_client("s3")
        buckets = sorted(b["Name"] for b in s3.list_buckets().get("Buckets", []))
        floci_ok = True
        for name in ("datasets", "mlflow-artifacts", "models"):
            if name in buckets:
                resp = s3.list_objects_v2(Bucket=name, MaxKeys=50)
                objects[name] = resp.get("KeyCount") or len(resp.get("Contents") or [])
    except Exception as exc:  # noqa: BLE001
        floci_error = str(exc)
    else:
        floci_error = None
    mlflow_ok = _probe(str(tracking).rstrip("/") + "/health")
    return {
        "floci": {
            "ok": floci_ok,
            "endpoint": os.environ.get("AWS_ENDPOINT_URL") or PUBLIC_FLOCI,
            "public_url": PUBLIC_FLOCI,
            "buckets": buckets,
            "objects": objects,
            "error": floci_error,
        },
        "mlflow": {
            "ok": mlflow_ok,
            "tracking_uri": tracking,
            "public_url": PUBLIC_MLFLOW,
        },
        "trainer_image": {"name": TRAIN_IMAGE, "present": _image_present()},
        "experiment": cfg.get("experiment") or "churn",
        "model_name": cfg.get("model_name") or "churn",
        "env_name": cfg.get("env_name") or "local",
        "allow_production_register": bool(cfg.get("allow_production_register")),
        "sample_features": SAMPLE_FEATURES,
        "min_roc_auc": float(os.environ.get("FLO_ML_MIN_ROC_AUC", "0.70")),
        "scorer": {
            "ok": _probe(SCORER_HEALTH_URL),
            "public_url": PUBLIC_SCORER,
        },
    }


@app.get("/api/model-card")
def model_card() -> dict[str, Any]:
    """Dataset, algorithm, and quality-gate copy for the console."""
    return model_card_dict()


def _append(job_id: str, line: str) -> None:
    """Append one log line to an in-memory training job."""
    with JOBS_LOCK:
        JOBS[job_id]["logs"].append(line.rstrip("\n"))


def _set_job(job_id: str, **fields: Any) -> None:
    """Merge status/result fields onto a job record."""
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


def _run_process(job_id: str, cfg: dict[str, Any], req: TrainRequest) -> None:
    """Run src/train.py in this process (Fast mode); stream stdout into the job log."""
    work = Path(tempfile.mkdtemp(prefix="flo-ml-ui-"))
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MLFLOW_TRACKING_URI": str(
                os.environ.get("MLFLOW_TRACKING_URI") or cfg.get("mlflow_tracking_uri")
            ),
            "MLFLOW_EXPERIMENT_NAME": req.experiment,
            "FLO_ML_DATA_URI": str(cfg["s3_data_uri"]),
            "FLO_ML_VALID_URI": str(cfg["s3_valid_uri"]),
            "FLO_ML_MODEL_PREFIX": str(cfg["s3_model_uri"]).rstrip("/"),
            "FLO_ML_ENV_NAME": str(cfg.get("env_name") or "local"),
            "FLO_ML_INSTANCE_TYPE": "process",
            "FLO_ML_PROJECT": str(cfg.get("project") or "churn"),
            "FLO_ML_REGISTER_MODEL": str(cfg.get("model_name") or "churn"),
            "FLO_ML_RUN_NAME": f"console-process-{job_id[:8]}",
            "SM_MODEL_DIR": str(work / "model"),
            "SM_OUTPUT_DATA_DIR": str(work / "output"),
            "SM_CHANNEL_TRAIN": str(work / "train"),
            "SM_CHANNEL_VALIDATION": str(work / "valid"),
            "GIT_PYTHON_REFRESH": "quiet",
        }
    )
    cmd = [
        sys.executable,
        str(SRC / "train.py"),
        "--train",
        str(work / "train"),
        "--validation",
        str(work / "valid"),
        "--model-dir",
        str(work / "model"),
        "--output-dir",
        str(work / "output"),
        "--max-depth",
        str(req.max_depth),
        "--learning-rate",
        str(req.learning_rate),
        "--max-iter",
        str(req.max_iter),
        "--l2-regularization",
        str(req.l2_regularization),
        "--experiment",
        req.experiment,
    ]
    _append(job_id, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _append(job_id, line)
    code = proc.wait()
    result = None
    metrics_file = work / "output" / "metrics.json"
    if metrics_file.exists():
        result = json.loads(metrics_file.read_text())
        run_file = work / "output" / "run.json"
        if run_file.exists():
            result.update(json.loads(run_file.read_text()))
    if code != 0:
        raise RuntimeError(f"train.py exited {code}")
    _set_job(job_id, result=result)


def _run_docker(job_id: str, cfg: dict[str, Any], req: TrainRequest) -> None:
    """docker run the training image on the compose network (SageMaker-shaped mode)."""
    import docker

    client = docker.from_env()
    try:
        client.images.get(TRAIN_IMAGE)
    except Exception as exc:
        raise RuntimeError(
            f"Training image {TRAIN_IMAGE} is not built yet. "
            "Use Fast (laptop Python) or run: docker compose --profile train build trainer"
        ) from exc
    environment = {
        "PYTHONUNBUFFERED": "1",
        "AWS_ENDPOINT_URL": str(cfg.get("aws_container_endpoint_url") or "http://floci:4566"),
        "AWS_ACCESS_KEY_ID": str(cfg.get("aws_access_key_id") or "test"),
        "AWS_SECRET_ACCESS_KEY": str(cfg.get("aws_secret_access_key") or "test"),
        "AWS_DEFAULT_REGION": str(cfg.get("aws_region") or "us-east-1"),
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_S3_ADDRESSING_STYLE": "path",
        "MLFLOW_TRACKING_URI": str(cfg.get("mlflow_container_tracking_uri") or "http://mlflow:5000"),
        "MLFLOW_EXPERIMENT_NAME": req.experiment,
        "FLO_ML_DATA_URI": str(cfg["s3_data_uri"]),
        "FLO_ML_VALID_URI": str(cfg["s3_valid_uri"]),
        "FLO_ML_MODEL_PREFIX": str(cfg["s3_model_uri"]).rstrip("/"),
        "FLO_ML_ENV_NAME": str(cfg.get("env_name") or "local"),
        "FLO_ML_INSTANCE_TYPE": "local",
        "FLO_ML_PROJECT": str(cfg.get("project") or "churn"),
        "FLO_ML_REGISTER_MODEL": str(cfg.get("model_name") or "churn"),
        "FLO_ML_RUN_NAME": f"console-docker-{job_id[:8]}",
        "SM_CHANNEL_TRAIN": "/opt/ml/input/data/train",
        "SM_CHANNEL_VALIDATION": "/opt/ml/input/data/validation",
        "SM_MODEL_DIR": "/opt/ml/model",
        "SM_OUTPUT_DATA_DIR": "/opt/ml/output/data",
        "GIT_PYTHON_REFRESH": "quiet",
    }
    command = [
        "--max-depth",
        str(req.max_depth),
        "--learning-rate",
        str(req.learning_rate),
        "--max-iter",
        str(req.max_iter),
        "--l2-regularization",
        str(req.l2_regularization),
        "--experiment",
        req.experiment,
    ]
    _append(job_id, f"docker run --network {COMPOSE_NETWORK} {TRAIN_IMAGE} {' '.join(command)}")
    container = client.containers.run(
        TRAIN_IMAGE,
        command=command,
        environment=environment,
        network=COMPOSE_NETWORK,
        detach=True,
        remove=False,
    )
    try:
        for chunk in container.logs(stream=True, follow=True):
            text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
            for line in text.splitlines():
                if line:
                    _append(job_id, line)
        wait = container.wait()
        code = wait.get("StatusCode", 1) if isinstance(wait, dict) else 1
        if code != 0:
            raise RuntimeError(f"training container exited {code}")
        logs = "\n".join(JOBS[job_id]["logs"])
        result = None
        for line in reversed(logs.splitlines()):
            line = line.strip()
            if line.startswith("{") and "valid_roc_auc" in line:
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        _set_job(job_id, result=result)
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass


def _worker(job_id: str, req: TrainRequest) -> None:
    """Background thread: pick process vs docker, capture success or traceback."""
    _set_job(job_id, status="running", started_at=time.time())
    try:
        cfg = _cfg()
        if req.instance in {"local", "docker", "local_gpu"}:
            _run_docker(job_id, cfg, req)
        else:
            _run_process(job_id, cfg, req)
        _set_job(job_id, status="succeeded", finished_at=time.time())
        _append(job_id, "done")
    except Exception as exc:  # noqa: BLE001
        _append(job_id, traceback.format_exc())
        _set_job(job_id, status="failed", error=str(exc), finished_at=time.time())


@app.post("/api/train")
def start_train(req: TrainRequest) -> dict[str, str]:
    """Queue a training job and return job_id for polling."""
    if req.instance not in {"process", "local", "docker", "local_gpu"}:
        raise HTTPException(400, "instance must be process (laptop Python) or local (training container)")
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "instance": req.instance,
            "logs": [],
            "result": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
        }
    threading.Thread(target=_worker, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Job status, logs, and metrics payload for the console log pane."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        return dict(job)


@app.get("/api/runs")
def list_runs(experiment: str = "churn") -> dict[str, Any]:
    """Recent MLflow runs for the experiment, newest first."""
    cfg = _cfg()
    mlflow_client, _ = _mlflow_client(cfg)
    try:
        exp = mlflow_client.get_experiment_by_name(experiment)
    except Exception:
        exp = None
    if exp is None:
        return {"experiment": experiment, "runs": []}
    runs = mlflow_client.search_runs([exp.experiment_id], max_results=25, order_by=["attributes.start_time DESC"])
    out = []
    for run in runs:
        out.append(
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": dict(run.data.metrics),
                "params": {
                    k: v
                    for k, v in dict(run.data.params).items()
                    if k in {
                        "max_depth",
                        "learning_rate",
                        "max_iter",
                        "instance_type",
                        "env_name",
                        "model_class",
                    }
                },
                "tags": {
                    k: v
                    for k, v in dict(run.data.tags).items()
                    if k in {"project", "dataset_version"} or k.startswith("mlflow.")
                },
                "ui_url": f"{PUBLIC_MLFLOW.rstrip('/')}/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}",
            }
        )
    return {"experiment": experiment, "experiment_id": exp.experiment_id, "runs": out}


@app.get("/api/registry")
def registry(name: str = "churn") -> dict[str, Any]:
    """Registered model versions and staging/production aliases."""
    cfg = _cfg()
    mlflow_client, _ = _mlflow_client(cfg)
    try:
        versions = list(mlflow_client.search_model_versions(f"name='{name}'"))
    except Exception:
        versions = []
    aliases: dict[str, str] = {}
    try:
        rm = mlflow_client.get_registered_model(name)
        raw = getattr(rm, "aliases", None) or []
        if isinstance(raw, dict):
            aliases = {str(k): str(v) for k, v in raw.items()}
        else:
            aliases = {str(a.alias): str(a.version) for a in raw}
    except Exception:
        pass
    for alias in ("staging", "production", "archived"):
        if alias in aliases:
            continue
        try:
            mv = mlflow_client.get_model_version_by_alias(name, alias)
            aliases[alias] = str(mv.version)
        except Exception:
            continue
    return {
        "name": name,
        "aliases": aliases,
        "versions": [
            {
                "version": v.version,
                "run_id": v.run_id,
                "status": v.status,
                "aliases": [alias for alias, ver in aliases.items() if str(ver) == str(v.version)],
            }
            for v in sorted(versions, key=lambda x: int(x.version), reverse=True)
        ],
    }


@app.post("/api/promote")
def promote(req: PromoteRequest) -> dict[str, Any]:
    """Alias the latest version as staging. Production is rejected while ENV_NAME=local."""
    cfg = _cfg(req.env)
    if req.stage.lower() == "production" and not cfg.get("allow_production_register"):
        raise HTTPException(
            400,
            "Local runs cannot be marked Production. That gate exists so a laptop experiment never becomes a live endpoint.",
        )
    mlflow_client, _ = _mlflow_client(cfg)
    versions = list(mlflow_client.search_model_versions(f"name='{req.name}'"))
    if not versions:
        raise HTTPException(404, f"no registered versions for {req.name} — train first")
    latest = max(versions, key=lambda v: int(v.version))
    alias = req.stage.lower()
    mlflow_client.set_registered_model_alias(req.name, alias, latest.version)
    return {"name": req.name, "version": latest.version, "stage": req.stage, "alias": alias, "run_id": latest.run_id}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    """Score one customer via the shared ChurnClient (same path as the scoring service)."""
    cfg = _cfg()
    tracking = os.environ.get("MLFLOW_TRACKING_URI") or cfg.get("mlflow_tracking_uri")
    try:
        client = ChurnClient.from_registry(
            tracking_uri=str(tracking),
            name=req.model_name,
            stage="staging",
        )
        return client.score_one(req.features).as_dict()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/sample-customer")
def sample_customer() -> dict[str, Any]:
    """Default feature row used to prefill the Score form."""
    return {"features": SAMPLE_FEATURES, "hint": "A short-tenure fiber customer with several support tickets."}


def main() -> None:
    """Run uvicorn on PORT (default 8088) when invoked as python -m ui.app."""
    import uvicorn

    uvicorn.run("ui.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8088")), reload=False)


if __name__ == "__main__":
    main()
