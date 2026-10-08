ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}
LABEL org.proteus.runtime="standalone-v1" org.proteus.version="0.2.0"
ARG TORCH_VERSION=2.11.0+cu126
ARG TORCHVISION_VERSION=0.26.0+cu126
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
COPY deploy/docker/requirements-gpu.txt /app/requirements-gpu.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-gpu.txt
COPY proteus_realtime /app/proteus_realtime
COPY deploy/docker/models.json /config/models.json
ENV PYTHONUNBUFFERED=1 HF_HOME=/models TORCH_HOME=/models \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "-m"]
CMD ["proteus_realtime.server"]
