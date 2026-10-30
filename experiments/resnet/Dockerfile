FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.11.0+cu126 torchvision==0.26.0+cu126
COPY experiments/resnet/requirements.txt /opt/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/requirements.txt
WORKDIR /app
COPY experiments/resnet /app/experiments/resnet
COPY proteus_realtime /app/proteus_realtime
COPY proteus_baselines /app/proteus_baselines
COPY core/common.py core/exceptions.py core/executor.py core/predictor.py /app/core/
COPY algorithms/clipper.py algorithms/ilp.py /app/algorithms/
COPY run.py /app/run.py
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /tmp
CMD ["python", "-m", "experiments.resnet.server"]
