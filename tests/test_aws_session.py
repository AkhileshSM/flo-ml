from src.aws_session import botocore_config, parse_s3_uri


def test_parse_s3_uri() -> None:
    bucket, key = parse_s3_uri("s3://datasets/churn/train/data.csv")
    assert bucket == "datasets"
    assert key == "churn/train/data.csv"


def test_path_style_when_endpoint_set(monkeypatch) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:4566")
    monkeypatch.delenv("AWS_S3_ADDRESSING_STYLE", raising=False)
    cfg = botocore_config()
    assert cfg.s3["addressing_style"] == "path"
