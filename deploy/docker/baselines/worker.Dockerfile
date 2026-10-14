ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}
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
COPY proteus_baselines /app/proteus_baselines
COPY core/common.py core/exceptions.py core/executor.py core/predictor.py /app/core/
COPY algorithms/clipper.py /app/algorithms/clipper.py
COPY algorithms/ilp.py /app/algorithms/ilp.py
COPY run.py /app/run.py
COPY deploy/docker/baselines/models.json /config/models.json
LABEL org.proteus.baseline.runtime="original-single-gpu-v1"
LABEL org.proteus.baseline.sommelier="fixed-placement-single-gpu-v1"
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 HF_HOME=/models TORCH_HOME=/models \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /run
ENTRYPOINT ["python", "-m"]
CMD ["proteus_baselines.server"]
