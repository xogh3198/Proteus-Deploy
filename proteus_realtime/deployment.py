"""Host-only configuration/Compose generation; no serving or INFaaS dependency."""
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from .registry import load_registry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / "deploy/docker/proteus-docker.env"


def read_env(path):
    values = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=value")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not value.startswith(("\"", "'")):
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"Invalid settings key on line {number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if any(c in value for c in ("\n", "\r", "\0", "$", chr(96))):
            raise ValueError(f"Use a literal value for {key}; shell evaluation is forbidden")
        values[key] = value
    return values


def _path(value, base):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def settings(env_file=None):
    path = Path(env_file or os.environ.get("PROTEUS_ENV", DEFAULT_ENV)).resolve()
    values = read_env(DEFAULT_ENV)
    if path != DEFAULT_ENV.resolve():
        overrides = read_env(path)
        unknown = set(overrides) - set(values)
        if unknown:
            raise ValueError("Old/unknown settings: " + ", ".join(sorted(unknown)) +
                ". Back up old local.env and copy the new default; do not reuse bootstrap settings.")
        values.update(overrides)
    for key in values:
        if key in os.environ:
            values[key] = os.environ[key]
    for key, low, high in [("GPU", 0, 1024), ("PORT", 1024, 65535),
        ("WARMUP_QUERIES", 0, 1000), ("CPU_THREADS", 1, 128), ("RPC_THREADS", 2, 1024),
        ("MAX_QUEUE_SIZE", 1, 100000), ("PROFILE_TRIALS", 1, 1000)]:
        if not low <= int(values[key]) <= high:
            raise ValueError(f"{key} must be in {low}..{high}")
    for key in ("SLO_MS", "ALLOCATION_INTERVAL", "INITIAL_RATE"):
        if not math.isfinite(float(values[key])) or float(values[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < float(values["EWMA_ALPHA"]) <= 1:
        raise ValueError("EWMA_ALPHA must be in (0,1]")
    if not 0 <= float(values["DEADLINE_GUARD_MS"]) < float(values["SLO_MS"]):
        raise ValueError("DEADLINE_GUARD_MS must be in [0,SLO)")
    batches = [int(b) for b in values["BATCH_SIZES"].split(",")]
    if batches != sorted(set(batches)) or not batches or batches[0] != 1 or any(b > 64 or b & (b - 1) for b in batches):
        raise ValueError("BATCH_SIZES must be sorted powers of two, include 1, and stay <=64")
    if values["BIND_ADDRESS"] != "127.0.0.1":
        raise ValueError("This unauthenticated API must bind to 127.0.0.1")
    if values["ALLOW_GPU_SHARING"] not in {"0", "1"}:
        raise ValueError("ALLOW_GPU_SHARING must be 0 or 1")
    if values["MODEL_DIR"] == "auto" or "infaas" in values["WORKER_IMAGE"].lower():
        raise ValueError("Use a standalone Proteus cache and image, not bootstrap settings")
    for key in ("GPU_TYPE", "NETWORK_NAME", "CONTAINER_NAME", "APPLICATION"):
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", values[key]):
            raise ValueError(f"Invalid {key}")
    for key in ("MODEL_REGISTRY", "MODEL_DIR", "DATA_DIR", "INPUT_IMAGE", "EVAL_DIR"):
        values[key] = _path(values[key], ROOT)
    registry = load_registry(values["MODEL_REGISTRY"], values["MODEL_NAME"],
                             values["MODEL_ID"], values["MODEL_REVISION"])
    if values["MODEL_NAME"] not in registry or not registry[values["MODEL_NAME"]].get("enabled", True):
        raise ValueError("MODEL_NAME must be enabled in MODEL_REGISTRY")
    if values["GUROBI_LICENSE_FILE"]:
        values["GUROBI_LICENSE_FILE"] = _path(values["GUROBI_LICENSE_FILE"], ROOT)
        if not Path(values["GUROBI_LICENSE_FILE"]).is_file():
            raise ValueError("GUROBI_LICENSE_FILE does not exist (do not share its contents)")
    values["ROOT"], values["ENV_FILE"] = str(ROOT), str(path)
    return values


def container_user():
    """Honor the selected daemon without switching its context or config.

    Rootless container UID 0 maps to the login user; using the host UID there
    would map to a subordinate UID and prevent writes to host-owned bind mounts.
    """
    security = json.loads(subprocess.check_output(
        ["docker", "info", "--format", "{{json .SecurityOptions}}"], text=True))
    if not isinstance(security, list) or not all(isinstance(v, str) for v in security):
        raise ValueError("Cannot determine Docker security mode; refusing to guess container UID")
    if "name=rootless" in security:
        return "0:0"
    return f"{getattr(os, 'getuid', lambda: 0)()}:{getattr(os, 'getgid', lambda: 0)()}"


def compose_document(values, user=None):
    environment = {key: values[key] for key in [
        "GPU_TYPE", "MODEL_NAME", "MODEL_ID", "MODEL_REVISION", "APPLICATION", "WARMUP_QUERIES", "CPU_THREADS",
        "RPC_THREADS", "MAX_QUEUE_SIZE", "SLO_MS", "BATCH_SIZES",
        "ALLOCATION_INTERVAL", "EWMA_ALPHA", "INITIAL_RATE", "DEADLINE_GUARD_MS"]}
    environment.update({"HF_HOME": "/models", "TORCH_HOME": "/models",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "WORKER_PORT": "9000", "MODEL_REGISTRY": "/config/models.json",
        "PROFILE_DIR": "/profiles", "EVAL_DIR": "/eval", "PYTHONUNBUFFERED": "1"})
    volumes = [
        {"type": "bind", "source": values["MODEL_DIR"], "target": "/models", "read_only": True},
        {"type": "bind", "source": values["MODEL_REGISTRY"], "target": "/config/models.json", "read_only": True},
        {"type": "bind", "source": str(Path(values["DATA_DIR"]) / "profiles"), "target": "/profiles"},
        {"type": "bind", "source": values["EVAL_DIR"], "target": "/eval", "read_only": True},
    ]
    if values["GUROBI_LICENSE_FILE"]:
        volumes.append({"type": "bind", "source": values["GUROBI_LICENSE_FILE"],
                        "target": "/license/gurobi.lic", "read_only": True})
        environment["GRB_LICENSE_FILE"] = "/license/gurobi.lic"
    return {"services": {"worker": {
        "image": values["WORKER_IMAGE"], "container_name": values["CONTAINER_NAME"],
        "user": user if user is not None else f"{getattr(os, 'getuid', lambda: 0)()}:{getattr(os, 'getgid', lambda: 0)()}",
        "command": ["proteus_realtime.server"], "environment": environment,
        "ports": [f"{values['BIND_ADDRESS']}:{values['PORT']}:9000"], "volumes": volumes,
        "networks": {"default": {"aliases": ["proteus-worker"]}},
        "deploy": {"resources": {"reservations": {"devices": [{
            "driver": "nvidia", "device_ids": [values["GPU"]], "capabilities": ["gpu"]}]}}},
        "restart": "no", "stop_grace_period": "30s", "shm_size": "1gb",
        "healthcheck": {"test": ["CMD", "python", "-m", "proteus_realtime.client", "state",
            "--controller", "127.0.0.1:9000", "--quiet", "--timeout", "5"],
            "interval": "5s", "timeout": "8s", "start_period": "45s", "retries": 12},
    }}, "networks": {"default": {"name": values["NETWORK_NAME"], "driver": "bridge"}}}


def check_gpu_idle(values):
    if values["ALLOW_GPU_SHARING"] == "1":
        print("WARNING: GPU sharing enabled; do not interpret results as isolated.")
        return
    selected = subprocess.check_output(["nvidia-smi", "--id=" + values["GPU"],
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    own = subprocess.run(["docker", "inspect", "--format", "{{.State.Pid}}", values["CONTAINER_NAME"]],
        capture_output=True, text=True)
    own_pid = own.stdout.strip() if own.returncode == 0 else ""
    conflicts = [row for row in csv.reader(processes.splitlines())
                 if len(row) >= 2 and row[0].strip() == selected and row[1].strip() != own_pid]
    if conflicts:
        raise ValueError("Selected GPU is occupied by other processes. Stop your other experiment first; "
                         "this script never stops another project's containers.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["settings", "gen", "get", "gpus", "idle", "user"])
    parser.add_argument("key", nargs="?")
    parser.add_argument("--env")
    args = parser.parse_args()
    values = settings(args.env)
    if args.command == "get":
        if args.key not in values:
            raise ValueError("Unknown setting: " + str(args.key))
        print(values[args.key])
    elif args.command == "settings":
        print(json.dumps(values, indent=2))
    elif args.command == "gen":
        directory = ROOT / "deploy/docker/generated"
        directory.mkdir(parents=True, exist_ok=True)
        for path in (values["MODEL_DIR"], values["EVAL_DIR"],
                     str(Path(values["DATA_DIR"]) / "results"), str(Path(values["DATA_DIR"]) / "profiles")):
            Path(path).mkdir(parents=True, exist_ok=True)
        (directory / "docker-compose.yml").write_text(json.dumps(compose_document(values, container_user()), indent=2), encoding="utf-8")
        (directory / "resolved-settings.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
        print("wrote", directory / "docker-compose.yml")
    elif args.command == "idle":
        check_gpu_idle(values)
    elif args.command == "user":
        print(container_user())
    else:
        subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv"], check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
