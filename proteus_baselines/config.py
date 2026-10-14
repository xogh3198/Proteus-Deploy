"""Host-only literal settings and isolated Compose; never select a Docker daemon."""
import hashlib
import json
import math
import os
import re
from pathlib import Path

from . import BASELINES
from proteus_realtime.deployment import read_env

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCES = {
    "core/predictor.py": "09611cca762783c52f1db2929db4c0096207d91a7e812fe8f797744202c57c48",
    "core/executor.py": "f1622f1960c14b7e97ea685a55513107314e3d86793ff4b94d3b63853d6f9131",
    "core/common.py": "095fa65ee2863c11b1bc67728af05529bf9843f7159c55484cae37818f7c6192",
    "core/exceptions.py": "c17125b08ea6890a000b7315b378de2a46b449b4439738a7d096f27cf18a8790",
    "algorithms/clipper.py": "d1397a04e699648be5bc47e5b9061e30783a3498b8a8158e10c45fb5b7e9c013",
    "proteus_realtime/runtime.py": "d5c2f4f07ceede332a7b505ba777a77708a0061e0b42b2488e7a1913142f18f6",
}


def verify_sources(root=ROOT):
    actual = {name: hashlib.sha256((Path(root) / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for name in EXPECTED_SOURCES}
    changed = [name for name, value in actual.items() if value != EXPECTED_SOURCES[name]]
    if changed:
        raise ValueError("Original policy/shared GPU backend differs from verified snapshot: " + ", ".join(changed))
    return actual


def settings(baseline, root=ROOT):
    if baseline not in BASELINES:
        raise ValueError("Unknown baseline")
    root = Path(root).resolve()
    default = root / "deploy/docker/baselines/baseline.env"
    values = read_env(default)
    local = Path(os.environ.get("BASELINE_ENV", root / "deploy/docker/baselines/baseline.local.env"))
    if local.exists():
        overrides = read_env(local)
        if set(overrides) - set(values):
            raise ValueError("Unknown baseline local settings")
        values.update(overrides)
    for key in values:
        if "BASELINE_" + key in os.environ:
            values[key] = os.environ["BASELINE_" + key]
    for key, low, high in (("GPU", 0, 1024), ("CPU_THREADS", 1, 128), ("RPC_THREADS", 2, 1024),
                           ("MAX_QUEUE_SIZE", 1, 100000), ("PROFILE_TRIALS", 1, 1000), ("WARMUP_QUERIES", 0, 1000)):
        if not low <= int(values[key]) <= high:
            raise ValueError("Invalid numeric setting: " + key)
    for key in ("SLO_MS", "ALLOCATION_INTERVAL"):
        if not math.isfinite(float(values[key])) or float(values[key]) <= 0:
            raise ValueError("Positive finite setting required: " + key)
    if values["ALLOW_GPU_SHARING"] not in {"0", "1"}:
        raise ValueError("ALLOW_GPU_SHARING must be 0 or 1")
    if values["GPU_TYPE"] != "v100":
        raise ValueError("This matched experiment requires GPU_TYPE=v100")
    if not re.fullmatch(r"proteus-baseline-worker:[a-zA-Z0-9_.-]+", values["WORKER_IMAGE"]):
        raise ValueError("Use a separate proteus-baseline-worker tag; do not overwrite original deployment images")
    if values["MODEL_NAME"] != "resnet-50" or values["APPLICATION"] != "imagenet":
        raise ValueError("This experiment is explicitly the singleton ResNet-50/ImageNet scope")
    sizes = [int(value) for value in values["BATCH_SIZES"].split(",")]
    if not sizes or sizes != sorted(set(sizes)) or sizes[0] != 1 or any(b > 64 or b & (b - 1) for b in sizes):
        raise ValueError("BATCH_SIZES must be sorted powers of two <=64, including 1")
    for key in ("DATA_ROOT", "INPUT_IMAGE", "MODEL_REGISTRY"):
        path = Path(values[key]).expanduser()
        values[key] = str((root / path).resolve() if not path.is_absolute() else path.resolve())
    registry = json.loads(Path(values["MODEL_REGISTRY"]).read_text(encoding="utf-8-sig"))["models"]
    if set(registry) != {"resnet-50"} or registry["resnet-50"].get("enabled") is not True or \
       registry["resnet-50"].get("model_id") != "microsoft/resnet-50" or \
       registry["resnet-50"].get("revision") != "34c2154c194f829b11125337b98c8f5f9965ff19":
        raise ValueError("Use the pinned singleton baseline registry, not a multi-model/unpinned original registry")
    values.update({"BASELINE": baseline, "ROOT": str(root), "PORT": str(19081 + BASELINES.index(baseline)),
        "PROJECT": "proteus-baseline-" + baseline, "NETWORK_NAME": "proteus-baseline-" + baseline,
        "CONTAINER_NAME": "proteus-baseline-" + baseline,
        "RUN_DIR": str(Path(values["DATA_ROOT"]) / baseline),
        "MODEL_DIR": str(Path(values["DATA_ROOT"]) / baseline / "models"),
        "RESULT_DIR": str(Path(values["DATA_ROOT"]) / baseline / "results")})
    return values


def compose_document(c, user):
    env = {key: c[key] for key in ("BASELINE", "GPU_TYPE", "MODEL_NAME", "APPLICATION", "SLO_MS",
           "CPU_THREADS", "WARMUP_QUERIES", "RPC_THREADS", "MAX_QUEUE_SIZE", "BATCH_SIZES", "ALLOCATION_INTERVAL")}
    env.update({"MODEL_REGISTRY": "/config/models.json", "HF_HOME": "/models", "TORCH_HOME": "/models",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return {"services": {"worker": {"image": c["WORKER_IMAGE"], "container_name": c["CONTAINER_NAME"],
        "user": user, "command": ["proteus_baselines.server"], "environment": env,
        "labels": {"org.proteus.baseline.name": c["BASELINE"]},
        "ports": ["127.0.0.1:" + c["PORT"] + ":9000"],
        "volumes": [{"type": "bind", "source": c["MODEL_DIR"], "target": "/models", "read_only": True},
                    {"type": "bind", "source": c["MODEL_REGISTRY"], "target": "/config/models.json", "read_only": True},
                    {"type": "bind", "source": c["RUN_DIR"], "target": "/run"}],
        "networks": {"default": {"aliases": ["baseline-worker"]}}, "restart": "no", "shm_size": "1gb",
        "stop_grace_period": "30s", "deploy": {"resources": {"reservations": {"devices": [{
            "driver": "nvidia", "device_ids": [c["GPU"]], "capabilities": ["gpu"]}]}}},
        "healthcheck": {"test": ["CMD", "python", "-m", "proteus_baselines.client", "state", "--quiet",
            "--baseline", c["BASELINE"], "--controller", "127.0.0.1:9000", "--timeout", "5"],
            "interval": "5s", "timeout": "8s", "start_period": "45s", "retries": 12}}},
        "networks": {"default": {"name": c["NETWORK_NAME"], "driver": "bridge"}}}
