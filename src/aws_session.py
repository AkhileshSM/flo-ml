"""Boto session factory for Floci (path-style S3) and real AWS.

SageMaker Local Mode containers cannot use localhost:4566 — that is the
container loopback. Host code uses AWS_ENDPOINT_URL; container code uses
AWS_CONTAINER_ENDPOINT_URL (http://floci:4566) when running on the compose network.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def _truthy(name: str, default: str = "true") -> bool:
    """Parse a boolean environment flag."""
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def resolve_endpoint_url(endpoint_url: str | None = None) -> str | None:
    """Return an explicit endpoint, else AWS_ENDPOINT_URL, else None (real AWS)."""
    if endpoint_url is not None:
        return endpoint_url or None
    return os.environ.get("AWS_ENDPOINT_URL") or None


def addressing_style() -> str:
    """Force path-style S3 against Floci; use virtual-hosted style on real AWS."""
    explicit = os.environ.get("AWS_S3_ADDRESSING_STYLE")
    if explicit:
        return explicit
    return "path" if resolve_endpoint_url() else "auto"


def botocore_config(**overrides: Any) -> Config:
    """Retries, timeouts, and S3 addressing for Floci or Amazon S3."""
    style = addressing_style()
    s3 = {"addressing_style": style} if style != "auto" else {}
    kwargs: dict[str, Any] = {
        "retries": {"max_attempts": 8, "mode": "standard"},
        "connect_timeout": 5,
        "read_timeout": 60,
        "s3": s3,
    }
    kwargs.update(overrides)
    return Config(**kwargs)


def client(service: str, endpoint_url: str | None = None, **kwargs: Any):
    """boto3 client pointed at Floci when AWS_ENDPOINT_URL is set."""
    endpoint = resolve_endpoint_url(endpoint_url)
    region = kwargs.pop("region_name", None) or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client(
        service,
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=kwargs.pop("aws_access_key_id", os.environ.get("AWS_ACCESS_KEY_ID") or None),
        aws_secret_access_key=kwargs.pop(
            "aws_secret_access_key", os.environ.get("AWS_SECRET_ACCESS_KEY") or None
        ),
        config=kwargs.pop("config", botocore_config()),
        **kwargs,
    )


def resource(service: str, endpoint_url: str | None = None, **kwargs: Any):
    """boto3 resource with the same endpoint/path-style rules as client()."""
    endpoint = resolve_endpoint_url(endpoint_url)
    region = kwargs.pop("region_name", None) or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.resource(
        service,
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=kwargs.pop("aws_access_key_id", os.environ.get("AWS_ACCESS_KEY_ID") or None),
        aws_secret_access_key=kwargs.pop(
            "aws_secret_access_key", os.environ.get("AWS_SECRET_ACCESS_KEY") or None
        ),
        config=kwargs.pop("config", botocore_config()),
        **kwargs,
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def ensure_bucket(name: str, s3=None) -> None:
    """Create the bucket if head_bucket fails (idempotent for local bootstrap)."""
    s3 = s3 or client("s3")
    try:
        s3.head_bucket(Bucket=name)
    except ClientError:
        s3.create_bucket(Bucket=name)


def upload_file(local_path: str, s3_uri: str, s3=None) -> str:
    """Put a local file at s3_uri; if the URI is a prefix, append the filename."""
    s3 = s3 or client("s3")
    bucket, key = parse_s3_uri(s3_uri)
    if key.endswith("/") or not key:
        key = f"{key}{os.path.basename(local_path)}"
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def download_prefix(s3_uri: str, dest_dir: str, s3=None) -> list[str]:
    """Download every object under an S3 prefix into dest_dir; return local paths."""
    s3 = s3 or client("s3")
    bucket, prefix = parse_s3_uri(s3_uri)
    os.makedirs(dest_dir, exist_ok=True)
    written: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/") if key.startswith(prefix) else os.path.basename(key)
            if not rel:
                rel = os.path.basename(key)
            path = os.path.join(dest_dir, rel)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            s3.download_file(bucket, key, path)
            written.append(path)
    return written


def head_etag(s3_uri: str, s3=None) -> str | None:
    """ETag of an object, or of the first object under a prefix — used as a data snapshot id."""
    s3 = s3 or client("s3")
    bucket, key = parse_s3_uri(s3_uri.rstrip("/"))
    if key.endswith("/") or not key:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=key if key else "", MaxKeys=1)
        contents = resp.get("Contents") or []
        return contents[0]["ETag"].strip('"') if contents else None
    try:
        return s3.head_object(Bucket=bucket, Key=key)["ETag"].strip('"')
    except ClientError:
        return None


@lru_cache(maxsize=1)
def disable_imds() -> None:
    """Skip EC2 metadata lookups (they hang locally) and relax checksums for Floci."""
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")


disable_imds()
