FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_EC2_METADATA_DISABLED=true \
    AWS_DEFAULT_REGION=us-east-1 \
    AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
    AWS_RESPONSE_CHECKSUM_VALIDATION=when_required

COPY docker/mlflow/aws.config /root/.aws/config
COPY infra/local/bootstrap.py /bootstrap.py
COPY data/fixtures /fixtures

RUN pip install --no-cache-dir "boto3==1.38.0" "botocore==1.38.0"

ENTRYPOINT ["python", "/bootstrap.py"]
