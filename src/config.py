"""Load configs/local.yaml or configs/aws.yaml with env interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate(value: Any) -> Any:
    """Replace ${ENV_VAR} placeholders in strings nested inside dicts/lists."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_config(env: str | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/<env>.yaml (local|aws) and interpolate environment variables."""
    env_name = (env or os.environ.get("ENV_NAME") or "local").strip().lower()
    cfg_path = Path(path) if path else ROOT / "configs" / f"{env_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open() as fh:
        data = yaml.safe_load(fh) or {}
    data = _interpolate(data)
    data["env_name"] = data.get("env_name") or env_name
    return data


def apply_runtime_env(cfg: dict[str, Any], *, in_container: bool = False) -> None:
    """Push the contract into process env so boto/MLflow pick it up."""
    endpoint = (
        cfg.get("aws_container_endpoint_url") if in_container else cfg.get("aws_endpoint_url")
    )
    tracking = (
        cfg.get("mlflow_container_tracking_uri") if in_container else cfg.get("mlflow_tracking_uri")
    )
    if endpoint:
        os.environ["AWS_ENDPOINT_URL"] = str(endpoint)
    elif cfg.get("env_name") != "local":
        os.environ.pop("AWS_ENDPOINT_URL", None)
    if cfg.get("aws_access_key_id"):
        os.environ.setdefault("AWS_ACCESS_KEY_ID", str(cfg["aws_access_key_id"]))
    if cfg.get("aws_secret_access_key"):
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", str(cfg["aws_secret_access_key"]))
    os.environ.setdefault("AWS_DEFAULT_REGION", str(cfg.get("aws_region") or "us-east-1"))
    if tracking:
        os.environ["MLFLOW_TRACKING_URI"] = str(tracking)
    if cfg.get("aws_s3_addressing_style"):
        os.environ["AWS_S3_ADDRESSING_STYLE"] = str(cfg["aws_s3_addressing_style"])
    os.environ["ENV_NAME"] = str(cfg.get("env_name") or "local")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
