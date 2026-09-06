FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_EC2_METADATA_DISABLED=true \
    GIT_PYTHON_REFRESH=quiet \
    PORT=8090

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-train.txt requirements-ui.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-ui.txt

COPY src /app/src
COPY client /app/client
COPY configs /app/configs
COPY docker/mlflow/aws.config /root/.aws/config

EXPOSE 8090
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=12 \
    CMD curl -fsS http://127.0.0.1:8090/health || exit 1

CMD ["python", "-m", "uvicorn", "client.service:app", "--host", "0.0.0.0", "--port", "8090"]
