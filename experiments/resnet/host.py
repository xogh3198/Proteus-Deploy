"""Host-only isolated Docker lifecycle. Never select or edit a Docker daemon."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.request


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_env(path):
    result = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"Invalid literal setting on line {number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if any(c in value for c in ("$", "`", "\0", "\r", "\n")):
            raise ValueError("Shell expressions are not allowed in settings")
        result[key] = value
    return result


def settings(engine, root=ROOT):
    root = Path(root).resolve()
    here = root / "experiments/resnet"
    if engine not in json.loads((here/"engines.json").read_text()):
        raise ValueError("Select an engine implemented by this repository")
    values = read_env(here/"settings.env")
    local = Path(os.environ.get("RESNET_ENV", here/"local.env"))
    if local.exists():
        updates = read_env(local)
        if set(updates)-set(values):
            raise ValueError("Unknown settings: "+", ".join(sorted(set(updates)-set(values))))
        values.update(updates)
    elif "RESNET_ENV" in os.environ:
        raise FileNotFoundError(local)
    phase = json.loads((here/"phase.json").read_text())
    if (phase["phase"] not in {"v1", "v2"}
            or phase["request_preprocessing"] is not (phase["phase"] == "v2")):
        raise ValueError("Phase manifest is inconsistent")
    for key in ("GPU", "CPU_THREADS", "CPU_INTEROP_THREADS", "PREPROCESS_WORKERS", "RPC_THREADS",
                "WORKER_THREADS", "MAX_QUEUE_SIZE", "PROFILE_TRIALS", "QUERY_THREADS", "TRACE_CONCURRENCY"):
        low = 0 if key == "GPU" else 1
        if not values[key].isdigit() or not low <= int(values[key]) <= 4096:
            raise ValueError("Invalid integer setting: "+key)
    if int(values["PREPROCESS_WORKERS"]) > 32 or int(values["TRACE_CONCURRENCY"]) > 1024:
        raise ValueError("Too many preparation workers or client slots")
    for key in ("SLO_MS", "TRACE_SPEED", "RPC_TIMEOUT_SECONDS", "PROTEUS_ALLOCATION_INTERVAL",
                "ACCURACY_ALLOCATION_INTERVAL", "INFAAS_ALLOCATION_INTERVAL", "EWMA_ALPHA", "INITIAL_RATE"):
        if not math.isfinite(float(values[key])) or float(values[key]) <= 0:
            raise ValueError("Expected finite positive setting: "+key)
    batches = list(map(int, values["BATCH_SIZES"].split(",")))
    if not batches or batches != sorted(set(batches)) or batches[0] != 1 or any(b not in (1,2,4,8,16,32) for b in batches):
        raise ValueError("BATCH_SIZES must contain sorted distinct powers of two, starting at 1")
    if not 1024 <= int(values["HOST_PORT"]) <= 65535 or values["CLIPPER_MODE"] not in {"ha", "ht"}:
        raise ValueError("Invalid port or Clipper fixed-model mode")
    if values["START_STATE"] not in {"cold", "active"} or values["WORKLOAD"] not in {"twitter", "uniform", "gamma", "bursty"}:
        raise ValueError("Invalid starting state or workload")
    if values["GPU_TYPE"] != "v100":
        raise ValueError("The first paired experiment requires GPU_TYPE=v100")
    for key in ("MODEL_DIR", "DATA_DIR", "INPUT_IMAGE"):
        p = Path(values[key]).expanduser()
        values[key] = str((p if p.is_absolute() else root/p).resolve())
    phase = phase["phase"]
    values.update(ROOT=str(root), ENGINE=engine, EXPERIMENT_PHASE=phase,
                  WORKER_IMAGE=f"paired-{engine}-resnet:{phase}",
                  PROJECT=f"paired-resnet-{engine}-{phase}")
    values["RUN_DIR"] = str(Path(values["DATA_DIR"])/engine/phase)
    values["MODEL_NAME"] = "resnet18_v1" if engine == "clipper" and values["CLIPPER_MODE"] == "ht" else (
        "resnet152_v1" if engine == "clipper" else "resnet50_v1")
    return values


def run(args, capture=False, cwd=ROOT):
    result = subprocess.run([str(a) for a in args], cwd=cwd, text=True, check=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def docker_info(c):
    raw = run(["docker", "info", "--format", "{{json .}}"], True)
    if not raw:
        raise RuntimeError("Selected Docker daemon returned no info; check DOCKER_HOST/context")
    info = json.loads(raw)
    actual = info["DockerRootDir"]
    if c["EXPECTED_DOCKER_ROOT"] and actual != c["EXPECTED_DOCKER_ROOT"]:
        raise RuntimeError(f"Selected Docker root is {actual}; expected {c['EXPECTED_DOCKER_ROOT']}. Select your user daemon in this terminal.")
    print("Docker data-root:", actual, flush=True)
    return info


def docker_user(info):
    if any("rootless" in x for x in info.get("SecurityOptions", [])):
        return "0:0"
    return f"{os.getuid()}:{os.getgid()}"


def source_fingerprint(c):
    paths = [HERE]
    if c["ENGINE"] in {"proteus", "infaas-accuracy"}:
        paths += [ROOT/"proteus_realtime", ROOT/"proteus_baselines", ROOT/"core", ROOT/"algorithms"]
    elif c["ENGINE"] == "infaas":
        paths += [ROOT/"infaas"]
    else:
        paths += [ROOT/"clipper_gpu", ROOT/"containers/python/rpc.py", ROOT/"src"]
    h = hashlib.sha256()
    for base in paths:
        for path in sorted([base] if base.is_file() else base.rglob("*")):
            if (not path.is_file() or path.name == "local.env" or "__pycache__" in path.parts
                    or "generated" in path.parts or path.suffix in {".pyc", ".md"}):
                continue
            h.update(path.relative_to(ROOT).as_posix().encode())
            h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def manifest_path(c):
    return Path(c["RUN_DIR"])/"build.json"


def verify_build(c):
    data = json.loads(manifest_path(c).read_text())
    if data["source_sha256"] != source_fingerprint(c):
        raise RuntimeError("Source/phase changed since build; rebuild this phase")
    for image, identity in data["images"].items():
        if run(["docker", "image", "inspect", "--format", "{{.Id}}", image], True) != identity:
            raise RuntimeError("Experiment image changed after build: "+image)
    return data


def core_images():
    return ["paired-clipper-query:resnet-fp32", "paired-clipper-management:resnet-fp32", "redis:6.2-alpine"]


def build(c):
    docker_info(c)
    images = [c["WORKER_IMAGE"]]
    if c["ENGINE"] == "clipper":
        from clipper_gpu.config import read_env as original_env
        from clipper_gpu.deploy import check_original, prepare_base
        original = original_env(ROOT/"deploy/docker/clipper-docker.env")
        original.update(ROOT=str(ROOT), LIB_BASE_IMAGE="paired-clipper/lib_base:resnet-fp32")
        check_original(original)
        run(["git", "submodule", "update", "--init", "--recursive"])
        status = run(["git", "submodule", "status", "--recursive"], True)
        if any(line.startswith(("+", "-", "U")) for line in status.splitlines()):
            raise RuntimeError("Clipper dependency revisions differ")
        for dependency in ("src/libs/spdlog", "src/libs/redox"):
            if run(["git", "status", "--porcelain"], True, ROOT/dependency):
                raise RuntimeError("Modified Clipper dependency: "+dependency)
        registry, version = prepare_base(original)
        for dockerfile, image in zip(["QueryFrontendDockerfile", "ManagementFrontendDockerfile"], core_images()[:2]):
            run(["docker", "build", "--build-arg", "REGISTRY="+registry, "--build-arg", "CODE_VERSION="+version,
                 "-t", image, "-f", "dockerfiles/"+dockerfile, "."])
        run(["docker", "pull", "redis:6.2-alpine"])
        images += core_images()
    run(["docker", "build", "-t", c["WORKER_IMAGE"], "-f", "experiments/resnet/Dockerfile", "."])
    output = manifest_path(c)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source_sha256": source_fingerprint(c),
        "source_commit": run(["git", "rev-parse", "HEAD"], True),
        "phase": c["EXPERIMENT_PHASE"], "images": {image: run(["docker", "image", "inspect",
            "--format", "{{.Id}}", image], True) for image in images}}, indent=2))


def environment(c):
    env = {k: c[k] for k in ["ENGINE", "EXPERIMENT_PHASE", "MODEL_NAME", "SLO_MS", "BATCH_SIZES",
        "CPU_THREADS", "CPU_INTEROP_THREADS", "PREPROCESS_WORKERS", "RPC_THREADS", "WORKER_THREADS",
        "MAX_QUEUE_SIZE", "PROFILE_TRIALS", "WARMUP_QUERIES", "EWMA_ALPHA", "INITIAL_RATE",
        "DEADLINE_GUARD_MS", "RPC_BUDGET_MS", "HW_COST", "LAMBDA", "SLACK_THRESHOLD", "MIN_ACCURACY",
        "WORKER_IMAGE", "TRACE_SPEED", "TRACE_CONCURRENCY", "RPC_TIMEOUT_SECONDS", "START_STATE", "CLIPPER_MODE"]}
    interval = {"proteus": "PROTEUS_ALLOCATION_INTERVAL", "infaas-accuracy": "ACCURACY_ALLOCATION_INTERVAL",
                "infaas": "INFAAS_ALLOCATION_INTERVAL", "clipper": "PROTEUS_ALLOCATION_INTERVAL"}[c["ENGINE"]]
    env.update(ALLOCATION_INTERVAL=c[interval], GPU_TYPE="v100", APPLICATION="imagenet", MODEL_DIR="/models",
               HOST_GPU_INDEX=c["GPU"], QUERY_THREADS=c["QUERY_THREADS"],
               PROFILE_DIR="/profiles", MODEL_REGISTRY="/app/experiments/resnet/models.json",
               CLIPPER_MODEL_NAME=c["MODEL_NAME"], CLIPPER_MODEL_VERSION="1", CLIPPER_INPUT_TYPE="strings",
               CLIPPER_MODEL_PATH="/models", CLIPPER_IP="query", CLIPPER_PORT="7000",
               PROFILE_STRATEGY="continuous", PROFILE_RUNS=c["PROFILE_TRIALS"])
    return env


def mounts(c):
    return [{"type": "bind", "source": c["MODEL_DIR"], "target": "/models", "read_only": True},
            {"type": "bind", "source": c["INPUT_IMAGE"], "target": "/inputs/image.jpg", "read_only": True},
            {"type": "bind", "source": str(Path(c["RUN_DIR"])/"profiles"), "target": "/profiles"},
            {"type": "bind", "source": str(Path(c["RUN_DIR"])/"results"), "target": "/results"}]


def compose_spec(c, user="0:0"):
    base = {"image": c["WORKER_IMAGE"], "user": user, "restart": "no", "environment": environment(c),
            "volumes": mounts(c)}
    port = 50052 if c["ENGINE"] == "infaas" else 9000
    worker = {**base, "deploy": {"resources": {"reservations": {"devices": [
        {"driver": "nvidia", "device_ids": [c["GPU"]], "capabilities": ["gpu"]}]}}}}
    if c["ENGINE"] != "clipper":
        worker["ports"] = [f"127.0.0.1:{c['HOST_PORT']}:{port}"]
        return {"services": {"worker": worker}}
    worker["depends_on"] = ["query"]
    gateway = {**base, "command": ["python", "-m", "experiments.resnet.clipper"],
               "ports": [f"127.0.0.1:{c['HOST_PORT']}:8080"], "depends_on": ["query"]}
    # v1/v2 keep native Clipper's adaptive batching, prediction cache=0 and
    # original unbounded ready queue. The common MAX_QUEUE_SIZE is not imposed.
    return {"services": {
        "redis": {"image": core_images()[2], "command": ["redis-server", "--save", "", "--appendonly", "no",
            "--protected-mode", "no", "--notify-keyspace-events", "AKE"],
            "healthcheck": {"test": ["CMD", "redis-cli", "ping"], "interval": "2s", "timeout": "2s", "retries": 30}},
        "query": {"image": core_images()[0], "command": ["--redis_ip=redis", "--redis_port=6379",
            "--rpc_service_port=7000", "--prediction_cache_size=0", "--thread_pool_size="+c["QUERY_THREADS"]],
            "environment": {"CLIPPER_MAX_READY_QUEUE": "0"}, "depends_on": {"redis": {"condition": "service_healthy"}}},
        "management": {"image": core_images()[1], "command": ["--redis_ip=redis", "--redis_port=6379"],
            "depends_on": {"redis": {"condition": "service_healthy"}}},
        "gateway": gateway, "worker": worker}}


def compose(c, *args):
    return run(["docker", "compose", "-p", c["PROJECT"], "-f", str(Path(c["RUN_DIR"])/"compose.json"), *args])


def client(c, action, extra=(), capture=False, network=True):
    info = docker_info(c)
    build_data = verify_build(c)
    args = ["docker", "run", "--rm", "--user", docker_user(info)]
    if network:
        args += ["--network", c["PROJECT"]+"_default"]
    else:
        args += ["--network", "none"]
    for key, value in environment(c).items():
        args += ["-e", key+"="+value]
    args += ["-e", "EXPERIMENT_IMAGE_ID="+build_data["images"][c["WORKER_IMAGE"]]]
    args += ["-e", "EXPERIMENT_SOURCE_COMMIT="+build_data["source_commit"]]
    args += ["-e", "EXPERIMENT_SOURCE_SHA256="+build_data["source_sha256"]]
    for mount in mounts(c):
        args += ["--mount", "type=bind,src="+mount["source"]+",dst="+mount["target"]+( ",readonly" if mount.get("read_only") else "")]
    trace = Path(c["MODEL_DIR"])/("trace-"+c["WORKLOAD"]+".txt")
    args += ["--mount", f"type=bind,src={trace},dst=/data/trace.txt,readonly",
             c["WORKER_IMAGE"], "python", "-m", "experiments.resnet.client", action, *extra]
    return run(args, capture)


def setup(c):
    info = docker_info(c)
    verify_build(c)
    root = Path(c["MODEL_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    run(["docker", "run", "--rm", "--user", docker_user(info), "--mount", f"type=bind,src={root},dst=/models",
         c["WORKER_IMAGE"], "python", "-m", "experiments.resnet.assets", "download", "--root", "/models"])
    traces = json.loads((HERE/"traces.json").read_text())
    for name, entry in traces.items():
        path = root/("trace-"+name+".txt")
        if not path.exists():
            with urllib.request.urlopen(entry["url"], timeout=60) as response:
                data = response.read(32 << 20)
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("Public trace checksum differs")
            with path.open("xb") as output:
                output.write(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Existing trace content changed: "+str(path))
    image = Path(c["INPUT_IMAGE"])
    if not image.exists():
        image.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", timeout=60) as response:
            data = response.read(16 << 20)
        if hashlib.sha256(data).hexdigest() != "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a":
            raise ValueError("Sample JPEG changed; supply INPUT_IMAGE explicitly")
        with image.open("xb") as output:
            output.write(data)
    for name in ("profiles", "results"):
        (Path(c["RUN_DIR"])/name).mkdir(parents=True, exist_ok=True)
    print("Assets prepared. No GPU inference or workload was run.")


def up(c):
    info = docker_info(c)
    verify_build(c)
    if run(["docker", "ps", "-q", "--filter", "label=com.docker.compose.project="+c["PROJECT"]], True):
        raise RuntimeError("This experiment is already running; use state or down first")
    name = run(["nvidia-smi", "-i", c["GPU"], "--query-gpu=name", "--format=csv,noheader"], True)
    if "V100" not in name:
        raise RuntimeError("Selected GPU is not a V100")
    processes = run(["nvidia-smi", "-i", c["GPU"], "--query-compute-apps=pid,process_name", "--format=csv,noheader"], True)
    if processes:
        raise RuntimeError("Selected GPU has compute processes. No existing service was stopped:\n"+processes)
    if not Path(c["INPUT_IMAGE"]).is_file():
        raise FileNotFoundError("Run setup before up")
    for directory in ("profiles", "results"):
        (Path(c["RUN_DIR"])/directory).mkdir(parents=True, exist_ok=True)
    path = Path(c["RUN_DIR"])/"compose.json"
    path.write_text(json.dumps(compose_spec(c, docker_user(info)), indent=2))
    if c["ENGINE"] == "clipper":
        compose(c, "up", "-d", "redis", "query", "management", "gateway")
        for attempt in range(60):
            try:
                client(c, "native-register", capture=True); break
            except subprocess.CalledProcessError:
                if attempt == 59:
                    raise
                time.sleep(1)
        compose(c, "up", "-d", "worker")
    else:
        compose(c, "up", "-d", "worker")
    for attempt in range(90):
        try:
            client(c, "health", capture=True); break
        except subprocess.CalledProcessError:
            if attempt == 89:
                raise
            time.sleep(1)
    print("GPU service started. Run register, then trace; up does not replay a workload.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["proteus", "infaas-accuracy", "infaas", "clipper"])
    parser.add_argument("action", choices=["settings", "compose", "build", "setup", "up", "down", "state", "register", "query", "trace", "reset"])
    args, extra = parser.parse_known_args(argv)
    c = settings(args.engine)
    if args.action == "settings":
        print(json.dumps(c, ensure_ascii=False, indent=2)); return
    if args.action == "compose":
        print(json.dumps(compose_spec(c), indent=2)); return
    if args.action in {"build", "setup", "up"}:
        return {"build": build, "setup": setup, "up": up}[args.action](c)
    if args.action == "down":
        docker_info(c)
        path = Path(c["RUN_DIR"])/"compose.json"
        if path.exists():
            compose(c, "down")
        else:
            print("No compose file for this experiment; no service was stopped.")
        return
    return client(c, args.action, extra, network=not (args.action == "trace" and "--dry-run" in extra))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print("ERROR:", error, file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(1)
