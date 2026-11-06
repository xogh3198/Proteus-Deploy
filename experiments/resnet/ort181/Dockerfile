ARG RESNET_BASE
FROM python:3.11-slim-bookworm AS legacy
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl g++ \
    && rm -rf /var/lib/apt/lists/*
COPY install_cuda.py bridge.cc /tmp/
RUN curl -fsSL --retry 3 \
      https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz \
      -o /tmp/ort.tgz \
    && echo 'c2120ff3b383ff722ad7adc4b8102074f8f1d5eefc3e4e8a5dc51b90bafad34e  /tmp/ort.tgz' | sha256sum -c - \
    && mkdir -p /opt/ort181 \
    && tar -xzf /tmp/ort.tgz -C /opt/ort181 --strip-components=1 \
    && rm /tmp/ort.tgz \
    && python /tmp/install_cuda.py build /opt/ort181 \
    && g++ -std=c++14 -O2 -fPIC -shared -Wl,-z,defs \
      -I/opt/ort181/include -I/opt/ort181/cuda-include /tmp/bridge.cc \
      -L/opt/ort181/lib -L/opt/ort181/cuda11 -lonnxruntime -l:libcudart.so.11.0 \
      -o /opt/ort181/lib/libproteus_ort181.so

FROM ${RESNET_BASE}
COPY --from=legacy /opt/ort181/lib /opt/ort181/lib
COPY --from=legacy /opt/ort181/cuda11 /opt/ort181/cuda11
COPY --from=legacy /opt/ort181/LICENSE /opt/ort181/ThirdPartyNotices.txt /opt/ort181/
COPY install_cuda.py /tmp/
ENV PROTEUS_ORT181_BRIDGE=/opt/ort181/lib/libproteus_ort181.so \
    LD_LIBRARY_PATH=/opt/ort181/lib:/opt/ort181/cuda11:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN set -eu; \
    python /tmp/install_cuda.py runtime /opt/ort181; \
    rm /tmp/install_cuda.py; \
    deps="$(ldd /opt/ort181/lib/libonnxruntime_providers_cuda.so)"; \
    printf '%s\n' "$deps"; \
    missing="$(printf '%s\n' "$deps" | awk '/not found/ && $1 != "libcuda.so.1"')"; \
    test -z "$missing"; \
    python -c 'from experiments.resnet.legacy_ort import LegacyOrt; print(LegacyOrt().version)'
