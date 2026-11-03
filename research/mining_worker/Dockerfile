FROM python:3.11-slim-bookworm@sha256:8fb0b110b022c70cbe09d3ff1fb313654f31f9c814abb0d48180af56d906a199

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY research/mining_worker/requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /app/requirements.lock

COPY research/mining_worker /app/mining_worker
COPY core/common/__init__.py /app/common/__init__.py
COPY core/common/logging_conf.py /app/common/logging_conf.py
COPY configs/mining/worker.json /app/configs/mining/worker.json

EXPOSE 8790

CMD ["python", "-m", "mining_worker"]
