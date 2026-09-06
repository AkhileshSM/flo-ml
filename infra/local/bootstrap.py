"""Idempotent local AWS bootstrap: buckets, IAM role, sample dataset."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://floci:4566")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
WAIT_SECONDS = int(os.environ.get("FLOCI_WAIT_SECONDS", "90"))
FIXTURES = Path(os.environ.get("FLOCI_FIXTURES", "/fixtures"))

BUCKETS = ("datasets", "mlflow-artifacts", "models")
ROLE_NAME = "SageMakerExecutionRole"

ASSUME_ROLE = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}

ROLE_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3TrainArtifacts",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
            "Resource": [
                "arn:aws:s3:::datasets",
                "arn:aws:s3:::datasets/*",
                "arn:aws:s3:::models",
                "arn:aws:s3:::models/*",
                "arn:aws:s3:::mlflow-artifacts",
                "arn:aws:s3:::mlflow-artifacts/*",
            ],
        },
        {
            "Sid": "Logs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": "*",
        },
        {
            "Sid": "EcrPull",
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",
            ],
            "Resource": "*",
        },
    ],
}


def session_client(service: str):
    """Path-style boto3 client aimed at the Floci container."""
    return boto3.client(
        service,
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        config=Config(
            s3={"addressing_style": "path"},
            retries={"max_attempts": 8, "mode": "standard"},
            connect_timeout=3,
            read_timeout=10,
        ),
    )


def wait_for_floci() -> None:
    """Poll ListBuckets until Floci answers or FLOCI_WAIT_SECONDS elapses."""
    s3 = session_client("s3")
    deadline = time.time() + WAIT_SECONDS
    last_error = None
    while time.time() < deadline:
        try:
            s3.list_buckets()
            print(f"floci ready at {ENDPOINT}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 — retry any connect/protocol flake
            last_error = exc
            time.sleep(1)
    raise SystemExit(f"floci did not become ready within {WAIT_SECONDS}s: {last_error}")


def ensure_buckets(s3) -> None:
    """Create datasets, mlflow-artifacts, and models buckets if missing."""
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    for name in BUCKETS:
        if name in existing:
            print(f"bucket exists: s3://{name}", flush=True)
            continue
        s3.create_bucket(Bucket=name)
        print(f"created bucket s3://{name}", flush=True)


def ensure_role(iam) -> None:
    """Create the local SageMakerExecutionRole and attach the S3/logs/ECR policy."""
    try:
        iam.get_role(RoleName=ROLE_NAME)
        print(f"iam role exists: {ROLE_NAME}", flush=True)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in {"NoSuchEntity", "NoSuchEntityException"}:
            raise
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(ASSUME_ROLE),
            Description="FLO-ML local SageMaker-shaped execution role",
        )
        print(f"created iam role {ROLE_NAME}", flush=True)
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="FLOMLLocalTraining",
        PolicyDocument=json.dumps(ROLE_POLICY),
    )


def upload_fixtures(s3) -> None:
    """Copy the synthetic churn CSVs to s3://datasets/churn/{train,valid}/."""
    mapping = {
        "churn_train.csv": "churn/train/data.csv",
        "churn_valid.csv": "churn/valid/data.csv",
    }
    for filename, key in mapping.items():
        path = FIXTURES / filename
        if not path.exists():
            print(f"skip missing fixture {path}", flush=True)
            continue
        s3.upload_file(str(path), "datasets", key)
        print(f"uploaded s3://datasets/{key}", flush=True)


def main() -> int:
    """Wait for Floci, seed buckets/role/data, then exit 0 so MLflow can start."""
    wait_for_floci()
    s3 = session_client("s3")
    iam = session_client("iam")
    ensure_buckets(s3)
    try:
        ensure_role(iam)
    except ClientError as exc:
        print(f"iam bootstrap skipped ({exc})", flush=True)
    upload_fixtures(s3)
    print("bootstrap complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
