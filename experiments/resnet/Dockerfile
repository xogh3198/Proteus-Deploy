FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ARG TORCH_VARIANT=cu126
RUN python -m pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
    "torch==2.11.0+${TORCH_VARIANT}" "torchvision==0.26.0+${TORCH_VARIANT}"
COPY experiments/resnet/requirements.txt /opt/requirements.txt
RUN if [ "$TORCH_VARIANT" = cpu ]; then sed -i '/^onnxruntime-gpu==/d' /opt/requirements.txt; fi \
    && python -m pip install --no-cache-dir -r /opt/requirements.txt
WORKDIR /app
COPY experiments/resnet /app/experiments/resnet
COPY experiments/reproduce /app/experiments/reproduce
COPY proteus_realtime /app/proteus_realtime
COPY proteus_baselines /app/proteus_baselines
COPY core /app/core
COPY algorithms /app/algorithms
COPY traces/model_variants/resnet /app/traces/model_variants/resnet
COPY run.py /app/run.py
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /tmp
CMD ["python", "-m", "experiments.resnet.server"]
