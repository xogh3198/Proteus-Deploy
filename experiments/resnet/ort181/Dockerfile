ARG RESNET_BASE
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04@sha256:c3577698256c1ea9cbfa4a8cb853bd8e11bb6cc1e397c463aa99000f822a9938 AS legacy
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl g++ \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL --retry 3 \
      https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz \
      -o /tmp/ort.tgz \
    && echo 'c2120ff3b383ff722ad7adc4b8102074f8f1d5eefc3e4e8a5dc51b90bafad34e  /tmp/ort.tgz' | sha256sum -c - \
    && mkdir -p /opt/ort181/cuda11 \
    && tar -xzf /tmp/ort.tgz -C /opt/ort181 --strip-components=1 \
    && rm /tmp/ort.tgz
COPY bridge.cc /tmp/bridge.cc
RUN g++ -std=c++14 -O2 -fPIC -shared -Wl,-z,defs \
      -I/opt/ort181/include -I/usr/local/cuda/include /tmp/bridge.cc \
      -L/opt/ort181/lib -L/usr/local/cuda/lib64 -lonnxruntime -lcudart \
      -o /opt/ort181/lib/libproteus_ort181.so \
    && for lib in libcudart.so.11.0 libcublas.so.11 libcublasLt.so.11 libcufft.so.10 libcurand.so.10 \
                  libcudnn.so.8 libcudnn_adv_infer.so.8 libcudnn_cnn_infer.so.8 libcudnn_ops_infer.so.8; do \
         path="$(find -L /usr/local/cuda /usr/lib/x86_64-linux-gnu -name "$lib" -print -quit)" \
           && test -n "$path" && cp -L "$path" "/opt/ort181/cuda11/$lib" || exit 1; \
       done

FROM ${RESNET_BASE}
COPY --from=legacy /opt/ort181/lib /opt/ort181/lib
COPY --from=legacy /opt/ort181/cuda11 /opt/ort181/cuda11
COPY --from=legacy /opt/ort181/LICENSE /opt/ort181/ThirdPartyNotices.txt /opt/ort181/
ENV PROTEUS_ORT181_BRIDGE=/opt/ort181/lib/libproteus_ort181.so \
    LD_LIBRARY_PATH=/opt/ort181/lib:/opt/ort181/cuda11:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN set -eu; \
    deps="$(ldd /opt/ort181/lib/libonnxruntime_providers_cuda.so)"; \
    printf '%s\n' "$deps"; \
    missing="$(printf '%s\n' "$deps" | awk '/not found/ && $1 != "libcuda.so.1"')"; \
    test -z "$missing"; \
    python -c 'from experiments.resnet.legacy_ort import LegacyOrt; print(LegacyOrt().version)'
