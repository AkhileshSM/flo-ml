# Slim custom training image (product plan §16). Same entry point locally and on SageMaker.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_EC2_METADATA_DISABLED=true \
    AWS_DEFAULT_REGION=us-east-1 \
    AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
    AWS_RESPONSE_CHECKSUM_VALIDATION=when_required \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /opt/ml/code

COPY requirements-train.txt /tmp/requirements-train.txt
RUN pip install --no-cache-dir -r /tmp/requirements-train.txt

COPY src/ /opt/ml/code/
COPY docker/mlflow/aws.config /root/.aws/config

ENTRYPOINT ["python", "/opt/ml/code/train.py"]
