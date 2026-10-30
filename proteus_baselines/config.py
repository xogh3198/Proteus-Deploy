"""Host-only literal settings and isolated Compose; never select a Docker daemon."""
import hashlib
import json
import math
import os
import re
from pathlib import Path

from . import BASELINES, SUPPORTED_BASELINES
from proteus_realtime.deployment import read_env
from proteus_realtime.registry import load_registry

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCES = {
    "core/predictor.py": "09611cca762783c52f1db2929db4c0096207d91a7e812fe8f797744202c57c48",
    "core/executor.py": "f1622f1960c14b7e97ea685a55513107314e3d86793ff4b94d3b63853d6f9131",
    "core/common.py": "095fa65ee2863c11b1bc67728af05529bf9843f7159c55484cae37818f7c6192",
    "core/exceptions.py": "c17125b08ea6890a000b7315b378de2a46b449b4439738a7d096f27cf18a8790",
    "algorithms/clipper.py": "d1397a04e699648be5bc47e5b9061e30783a3498b8a8158e10c45fb5b7e9c013",
    "proteus_realtime/runtime.py": "9170282284d1b798ff39986ee4a401782374a9aca20becbd114ff3b2bcbf2e0f",
    "proteus_realtime/pipeline.py": "fd69c2ec4f2a9876a920a17253e5aa4d86a40d67459010c29ecc277be537564e",
    "proteus_realtime/service.py": "e9e3304b4db4da59970856792780fa0d5ddaba084e9d494cb5ea1bfc5cd52d44",
}
SOMMELIER_SOURCES = {
    "run.py": "d7daee55c9eab63456b4cbce4d2641088cee90464581417ea6b11dc0dbcaeaa7",
    "algorithms/ilp.py": "0f448fe0a9b703b8d9a1e0e963669a5d016428d5fcca09c4eb49c505fb3cef33",
    # Shared port: opt-in idle-aware profiles/RPC reserve. Sommelier's reserve
    # stays zero unless explicitly passed; original core/ILP remain unchanged.
    "proteus_realtime/policy.py": "8b611439391845bb8059e70e636f7b9cad08d452c352ff8f173e38bccd599fcb",
    "proteus_realtime/serving.py": "9b5c2513eb257b3e0de5baa9b7ad6c444fdcd8babf1667076bf1fdaa30f37c10",
    "proteus_realtime/profiling.py": "09d04c528606b31832f16ae8dfc7352a399ba78cfd83d8740724d1d631ca4bc7",
}


def verify_sources(root=ROOT, baseline=None):
    expected = {**EXPECTED_SOURCES, **(SOMMELIER_SOURCES if baseline == "sommelier" else {})}
    actual = {name: hashlib.sha256((Path(root) / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for name in expected}
    changed = [name for name, value in actual.items() if value != expected[name]]
    if changed:
        raise ValueError("Original policy/shared GPU backend differs from verified snapshot: " + ", ".join(changed))
    return actual


def settings(baseline, root=ROOT):
    if baseline not in SUPPORTED_BASELINES:
        raise ValueError("Unknown baseline")
    root = Path(root).resolve()
    sommelier = baseline == "sommelier"
    stem = "sommelier" if sommelier else "baseline"
    default = root / ("deploy/docker/baselines/" + stem + ".env")
    values = read_env(default)
    local = Path(os.environ.get("SOMMELIER_ENV" if sommelier else "BASELINE_ENV",
                                root / ("deploy/docker/baselines/" + stem + ".local.env")))
    if local.exists():
        overrides = read_env(local)
        if set(overrides) - set(values):
            raise ValueError("Unknown baseline local settings")
        values.update(overrides)
    for key in values:
        if "BASELINE_" + key in os.environ:
            values[key] = os.environ["BASELINE_" + key]
        if sommelier and "SOMMELIER_" + key in os.environ:
            values[key] = os.environ["SOMMELIER_" + key]
    for key, low, high in (("GPU", 0, 1024), ("CPU_THREADS", 1, 128), ("RPC_THREADS", 2, 1024),
                           ("MAX_QUEUE_SIZE", 1, 100000), ("PROFILE_TRIALS", 1, 1000), ("WARMUP_QUERIES", 0, 1000),
                           ("CPU_INTEROP_THREADS", 1, 128), ("PREPROCESS_WORKERS", 1, 32)):
        if not low <= int(values[key]) <= high:
            raise ValueError("Invalid numeric setting: " + key)
    for key in ("SLO_MS", "ALLOCATION_INTERVAL"):
        if not math.isfinite(float(values[key])) or float(values[key]) <= 0:
            raise ValueError("Positive finite setting required: " + key)
    if values["ALLOW_GPU_SHARING"] not in {"0", "1"}:
        raise ValueError("ALLOW_GPU_SHARING must be 0 or 1")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", values["GPU_TYPE"]):
        raise ValueError("GPU_TYPE must be a simple device type name")
    if not re.fullmatch(r"proteus-baseline-worker:[a-zA-Z0-9_.-]+", values["WORKER_IMAGE"]):
        raise ValueError("Use a separate proteus-baseline-worker tag; do not overwrite original deployment images")
    if values["APPLICATION"] != "imagenet":
        raise ValueError("This adapter supports one ImageNet classification application")
    sizes = [int(value) for value in values["BATCH_SIZES"].split(",")]
    if not sizes or sizes != sorted(set(sizes)) or sizes[0] != 1 or any(b > 64 or b & (b - 1) for b in sizes):
        raise ValueError("BATCH_SIZES must be sorted powers of two <=64, including 1")
    for key in ("DATA_ROOT", "INPUT_IMAGE", "MODEL_REGISTRY"):
        path = Path(values[key]).expanduser()
        values[key] = str((root / path).resolve() if not path.is_absolute() else path.resolve())
    registry = load_registry(values["MODEL_REGISTRY"], values["MODEL_NAME"],
                             values["MODEL_ID"], values["MODEL_REVISION"])
    enabled = {name: item for name, item in registry.items() if item.get("enabled", True)}
    if sommelier or baseline == "infaas-accuracy":
        if values["MODEL_NAME"] not in enabled or any(not re.fullmatch(r"[0-9a-f]{40}", item.get("revision", ""))
                                                       for item in enabled.values()):
            raise ValueError("Model selection needs enabled pinned models and an enabled MODEL_NAME")
        if baseline == "infaas-accuracy" and len(enabled) > 1:
            from proteus_realtime.registry import normalized_accuracies
            normalized_accuracies(enabled)
    if sommelier:
        for key in ("INITIAL_RATE", "ALLOCATION_INTERVAL"):
            if not math.isfinite(float(values[key])) or float(values[key]) <= 0:
                raise ValueError("Invalid Sommelier setting: " + key)
        if not 0 < float(values["EWMA_ALPHA"]) <= 1:
            raise ValueError("EWMA_ALPHA must be in (0,1]")
        if not 0 <= float(values["DEADLINE_GUARD_MS"]) < float(values["SLO_MS"]):
            raise ValueError("DEADLINE_GUARD_MS must be in [0,SLO)")
        values["EVAL_DIR"] = str((root / Path(values["EVAL_DIR"]).expanduser()).resolve())
        if values["GUROBI_LICENSE_FILE"]:
            values["GUROBI_LICENSE_FILE"] = str((root / Path(values["GUROBI_LICENSE_FILE"]).expanduser()).resolve())
            if not Path(values["GUROBI_LICENSE_FILE"]).is_file():
                raise ValueError("GUROBI_LICENSE_FILE does not exist")
    elif baseline != "infaas-accuracy" and (set(enabled) != {values["MODEL_NAME"]} or not re.fullmatch(
            r"[0-9a-f]{40}", enabled[values["MODEL_NAME"]].get("revision", ""))):
        raise ValueError("Use exactly one enabled pinned singleton model matching MODEL_NAME")
    values.update({"BASELINE": baseline, "ROOT": str(root), "PORT": "19084" if sommelier else str(19081 + BASELINES.index(baseline)),
        "PROJECT": "proteus-baseline-" + baseline, "NETWORK_NAME": "proteus-baseline-" + baseline,
        "CONTAINER_NAME": "proteus-baseline-" + baseline,
        "RUN_DIR": str(Path(values["DATA_ROOT"]) / baseline),
        "MODEL_DIR": str(Path(values["DATA_ROOT"]) / baseline / "models"),
        "RESULT_DIR": str(Path(values["DATA_ROOT"]) / baseline / "results")})
    return values


def compose_document(c, user):
    env = {key: c[key] for key in ("BASELINE", "GPU_TYPE", "MODEL_NAME", "MODEL_ID", "MODEL_REVISION", "APPLICATION", "SLO_MS",
           "CPU_THREADS", "WARMUP_QUERIES", "RPC_THREADS", "MAX_QUEUE_SIZE", "BATCH_SIZES", "ALLOCATION_INTERVAL",
           "CPU_INTEROP_THREADS", "PREPROCESS_WORKERS")}
    env.update({"MODEL_REGISTRY": "/config/models.json", "HF_HOME": "/models", "TORCH_HOME": "/models",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    document = {"services": {"worker": {"image": c["WORKER_IMAGE"], "container_name": c["CONTAINER_NAME"],
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
    if c["BASELINE"] == "sommelier":
        env.update({key: c[key] for key in ("EWMA_ALPHA", "INITIAL_RATE", "DEADLINE_GUARD_MS")})
        env["EVAL_DIR"] = "/eval"
        worker = document["services"]["worker"]
        worker["volumes"].append({"type": "bind", "source": c["EVAL_DIR"], "target": "/eval", "read_only": True})
        if c["GUROBI_LICENSE_FILE"]:
            worker["volumes"].append({"type": "bind", "source": c["GUROBI_LICENSE_FILE"],
                                      "target": "/license/gurobi.lic", "read_only": True})
            env["GRB_LICENSE_FILE"] = "/license/gurobi.lic"
    return document
