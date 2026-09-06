FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_EC2_METADATA_DISABLED=true \
    AWS_DEFAULT_REGION=us-east-1 \
    AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
    AWS_RESPONSE_CHECKSUM_VALIDATION=when_required \
    GIT_PYTHON_REFRESH=quiet \
    PORT=8088

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-train.txt requirements-ui.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-ui.txt

COPY src /app/src
COPY jobs /app/jobs
COPY configs /app/configs
COPY ui /app/ui
COPY client /app/client
COPY docker/mlflow/aws.config /root/.aws/config

EXPOSE 8088
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD curl -fsS http://127.0.0.1:8088/api/status || exit 1

CMD ["python", "-m", "uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8088"]
