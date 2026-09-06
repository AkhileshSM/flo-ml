"""Construct and run the churn training job.

Local default uses the same training image as SageMaker, launched on the
compose network (Floci S3 + MLflow). `process` skips Docker for a faster inner loop.
`ml.*` instance types are reserved for real SageMaker (P1).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from aws_session import client, download_prefix, head_etag, upload_file  # noqa: E402
from config import apply_runtime_env, load_config  # noqa: E402


def git_sha() -> str:
    """Current HEAD SHA for lineage, or 'unknown' outside a git checkout."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def image_digest(image: str) -> str:
    """Docker image id used as the training-image digest logged on the run."""
    try:
        out = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.Id}}", image],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return image


def ensure_train_image(image: str) -> None:
    """Build flo-ml/sklearn-train:local via compose if it is not already present."""
    probe = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    if probe.returncode == 0:
        return
    print(f"building {image}", flush=True)
    subprocess.check_call(
        [
            "docker",
            "compose",
            "-f",
            str(ROOT / "docker-compose.yml"),
            "--profile",
            "train",
            "build",
            "trainer",
        ],
        cwd=ROOT,
    )


def sm_layout(work: Path) -> dict[str, Path]:
    """Create SageMaker-shaped train/validation/model/output directories."""
    paths = {
        "train": work / "input" / "data" / "train",
        "validation": work / "input" / "data" / "validation",
        "model": work / "model",
        "output": work / "output" / "data",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def stage_channels(cfg: dict, paths: dict[str, Path]) -> dict[str, str]:
    """Download S3 train/valid prefixes into local channels; return URI + etag lineage."""
    train_uri = cfg["s3_data_uri"]
    valid_uri = cfg["s3_valid_uri"]
    downloaded_train = download_prefix(train_uri, str(paths["train"]))
    downloaded_valid = download_prefix(valid_uri, str(paths["validation"]))
    if not downloaded_train or not downloaded_valid:
        raise SystemExit(
            "S3 channels were empty. Is the local stack up? Try: make up\n"
            f" train={train_uri} valid={valid_uri}"
        )
    etag = head_etag(train_uri)
    return {"data_uri": train_uri, "data_etag": etag or "", "n_train": str(len(downloaded_train))}


def hyper_args(cfg: dict) -> list[str]:
    """Turn configs/*.yaml hyperparameters into train.py CLI flags."""
    hp = cfg.get("hyperparameters") or {}
    args: list[str] = []
    mapping = {
        "max_depth": "--max-depth",
        "learning_rate": "--learning-rate",
        "max_iter": "--max-iter",
        "l2_regularization": "--l2-regularization",
    }
    for key, flag in mapping.items():
        if key in hp:
            args.extend([flag, str(hp[key])])
    return args


def common_env(cfg: dict, lineage: dict[str, str], instance: str) -> dict[str, str]:
    """Environment injected into process-mode or the training container."""
    env = {
        "FLO_ML_DATA_URI": lineage.get("data_uri", cfg["s3_data_uri"]),
        "FLO_ML_VALID_URI": str(cfg.get("s3_valid_uri") or ""),
        "FLO_ML_MODEL_PREFIX": str(cfg.get("s3_model_uri") or "").rstrip("/"),
        "FLO_ML_DATA_ETAG": lineage.get("data_etag", ""),
        "FLO_ML_IMAGE_DIGEST": lineage.get("image_digest", ""),
        "FLO_ML_GIT_SHA": git_sha(),
        "FLO_ML_INSTANCE_TYPE": instance,
        "FLO_ML_ENV_NAME": str(cfg.get("env_name") or "local"),
        "FLO_ML_PROJECT": str(cfg.get("project") or "churn"),
        "FLO_ML_REGISTER_MODEL": str(cfg.get("model_name") or "churn"),
        "FLO_ML_APPROVED_FOR_CLOUD": "false" if cfg.get("env_name") == "local" else "true",
        "MLFLOW_EXPERIMENT_NAME": str(cfg.get("experiment") or "churn"),
        "ENV_NAME": str(cfg.get("env_name") or "local"),
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_REQUEST_CHECKSUM_CALCULATION": "when_required",
        "AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required",
        "AWS_DEFAULT_REGION": str(cfg.get("aws_region") or "us-east-1"),
        "AWS_S3_ADDRESSING_STYLE": str(cfg.get("aws_s3_addressing_style") or "path"),
        "GIT_PYTHON_REFRESH": "quiet",
    }
    return env


def run_process(cfg: dict, instance: str) -> dict:
    """Fast inner loop: stage S3 channels and run src/train.py with host Python."""
    apply_runtime_env(cfg, in_container=False)
    work = Path(tempfile.mkdtemp(prefix="flo-ml-"))
    paths = sm_layout(work)
    lineage = stage_channels(cfg, paths)
    lineage["image_digest"] = "process"
    env = os.environ.copy()
    env.update(common_env(cfg, lineage, instance))
    env["SM_CHANNEL_TRAIN"] = str(paths["train"])
    env["SM_CHANNEL_VALIDATION"] = str(paths["validation"])
    env["SM_MODEL_DIR"] = str(paths["model"])
    env["SM_OUTPUT_DATA_DIR"] = str(paths["output"])
    env["MLFLOW_TRACKING_URI"] = str(cfg.get("mlflow_tracking_uri") or env.get("MLFLOW_TRACKING_URI", ""))
    cmd = [sys.executable, str(SRC / "train.py"), "--experiment", str(cfg.get("experiment") or "churn")]
    cmd.extend(hyper_args(cfg))
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)
    return finalize(cfg, paths, work)


def run_docker(cfg: dict, instance: str) -> dict:
    """SageMaker-shaped path: docker run the training image on the compose network."""
    apply_runtime_env(cfg, in_container=False)
    image = str(cfg.get("image") or "flo-ml/sklearn-train:local")
    ensure_train_image(image)
    work = Path(tempfile.mkdtemp(prefix="flo-ml-"))
    paths = sm_layout(work)
    lineage = stage_channels(cfg, paths)
    lineage["image_digest"] = image_digest(image)
    env_pairs = common_env(cfg, lineage, instance)
    env_pairs["MLFLOW_TRACKING_URI"] = str(cfg.get("mlflow_container_tracking_uri") or "http://mlflow:5000")
    env_pairs["AWS_ENDPOINT_URL"] = str(cfg.get("aws_container_endpoint_url") or "http://floci:4566")
    env_pairs["AWS_ACCESS_KEY_ID"] = str(cfg.get("aws_access_key_id") or "test")
    env_pairs["AWS_SECRET_ACCESS_KEY"] = str(cfg.get("aws_secret_access_key") or "test")
    env_pairs["SM_CHANNEL_TRAIN"] = "/opt/ml/input/data/train"
    env_pairs["SM_CHANNEL_VALIDATION"] = "/opt/ml/input/data/validation"
    env_pairs["SM_MODEL_DIR"] = "/opt/ml/model"
    env_pairs["SM_OUTPUT_DATA_DIR"] = "/opt/ml/output/data"

    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        str(cfg.get("compose_network") or "flo-ml"),
        "-v",
        f"{paths['train']}:/opt/ml/input/data/train",
        "-v",
        f"{paths['validation']}:/opt/ml/input/data/validation",
        "-v",
        f"{paths['model']}:/opt/ml/model",
        "-v",
        f"{paths['output']}:/opt/ml/output/data",
    ]
    for key, value in env_pairs.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.append(image)
    cmd.extend(hyper_args(cfg))
    print("docker run", image, flush=True)
    subprocess.check_call(cmd, cwd=ROOT)
    return finalize(cfg, paths, work)


def run_sagemaker(cfg: dict, instance: str) -> dict:
    """P1 path: SKLearn Estimator on a real ml.* instance. Blocked while ENV_NAME=local."""
    if cfg.get("env_name") == "local":
        raise SystemExit("refusing cloud instance_type while ENV_NAME=local")
    try:
        from sagemaker.sklearn.estimator import SKLearn
    except ImportError as exc:
        raise SystemExit("pip install sagemaker to launch managed training") from exc

    apply_runtime_env(cfg, in_container=False)
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(SRC),
        role=cfg["role_arn"],
        instance_type=instance,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        hyperparameters=cfg.get("hyperparameters") or {},
        environment={
            "MLFLOW_TRACKING_URI": str(cfg.get("mlflow_tracking_uri") or ""),
            "MLFLOW_EXPERIMENT_NAME": str(cfg.get("experiment") or "churn"),
            "FLO_ML_ENV_NAME": str(cfg.get("env_name")),
            "FLO_ML_DATA_URI": str(cfg.get("s3_data_uri")),
            "FLO_ML_INSTANCE_TYPE": instance,
            "FLO_ML_GIT_SHA": git_sha(),
        },
        output_path=str(cfg.get("s3_model_uri")),
    )
    estimator.fit(
        {
            "train": cfg["s3_data_uri"],
            "validation": cfg["s3_valid_uri"],
        }
    )
    return {"job_name": estimator.latest_training_job.name, "instance": instance}


def finalize(cfg: dict, paths: dict[str, Path], work: Path) -> dict:
    """Upload model.joblib to s3://models/{project}/{run_id}/ and print the run summary."""
    run_id = "local"
    run_file = paths["output"] / "run.json"
    if run_file.exists():
        run_id = json.loads(run_file.read_text()).get("run_id") or run_id
    model_uri = f"{cfg['s3_model_uri'].rstrip('/')}/{run_id}/model.joblib"
    local_model = paths["model"] / "model.joblib"
    if local_model.exists():
        uploaded = upload_file(str(local_model), model_uri)
        print(f"uploaded {uploaded}", flush=True)
    metrics = {}
    metrics_file = paths["output"] / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
    result = {"run_id": run_id, "model_uri": model_uri, "metrics": metrics, "work": str(work)}
    print(json.dumps({k: v for k, v in result.items() if k != "work"}, indent=2), flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return result


def wait_mlflow(uri: str, timeout: int = 90) -> None:
    """Block until the tracking server /health returns 2xx, or abort."""
    import urllib.request

    health = uri.rstrip("/") + "/health"
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1)
    raise SystemExit(f"MLflow not reachable at {health}: {last}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """--env local|aws and --instance process|local|ml.*."""
    parser = argparse.ArgumentParser(description="FLO-ML estimator")
    parser.add_argument("--env", default=os.environ.get("ENV_NAME", "local"))
    parser.add_argument("--experiment", default=None)
    parser.add_argument(
        "--instance",
        default=None,
        help="local (docker image) | process (host python) | local_gpu | ml.*",
    )
    parser.add_argument("--config", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch to process, docker, or SageMaker based on instance type."""
    args = parse_args(argv)
    cfg = load_config(args.env, args.config)
    if args.experiment:
        cfg["experiment"] = args.experiment
    instance = args.instance or os.environ.get("SAGEMAKER_INSTANCE") or cfg.get("sagemaker_instance") or "local"
    print(f"env={cfg.get('env_name')} instance={instance}", flush=True)
    if instance.startswith("ml."):
        run_sagemaker(cfg, instance)
        return 0
    if cfg.get("env_name") == "local" and cfg.get("mlflow_tracking_uri"):
        wait_mlflow(str(cfg["mlflow_tracking_uri"]))
        client("s3").list_buckets()
    if instance in {"process", "local_process"}:
        run_process(cfg, "process")
    elif instance in {"local", "local_gpu", "docker"}:
        run_docker(cfg, instance)
    else:
        raise SystemExit(f"unknown instance type: {instance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
