"""Explicit Docker project per baseline; preserve all existing deployments."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from . import BASELINES
from .config import ROOT, compose_document, settings, verify_sources
from proteus_realtime.deployment import check_gpu_idle, container_user


def run(args, allow_failure=False, capture=False):
    result = subprocess.run(args, check=False, text=True, capture_output=capture)
    if result.returncode and not allow_failure:
        raise subprocess.CalledProcessError(result.returncode, args, result.stdout, result.stderr)
    return result


def image_required(c):
    kind = run(["docker", "image", "inspect", "--format", '{{index .Config.Labels "org.proteus.baseline.runtime"}}',
                c["WORKER_IMAGE"]], capture=True).stdout.strip()
    if kind != "original-single-gpu-v1":
        raise ValueError("Independent baseline image missing/wrong; run this script's build")


def compose(c, *args):
    file = ROOT / "deploy/docker/baselines/generated" / c["BASELINE"] / "docker-compose.yml"
    if not file.exists():
        raise ValueError("No generated project. Run this baseline's up first")
    return run(["docker", "compose", "--project-name", c["PROJECT"], "-f", str(file), *args])


def container_status(c):
    result = run(["docker", "inspect", c["CONTAINER_NAME"]], allow_failure=True, capture=True)
    if result.returncode:
        return None
    row = json.loads(result.stdout)[0]
    if row["Config"].get("Labels", {}).get("org.proteus.baseline.name") != c["BASELINE"]:
        raise ValueError("Container name belongs to another project; refusing to touch it")
    return row["State"]


def oneoff(c, module, *args, gpu=False, online=False, extra_mounts=()):
    image_required(c)
    Path(c["MODEL_DIR"]).mkdir(parents=True, exist_ok=True)
    command = ["docker", "run", "--rm", "--pull=never", "--user", container_user(),
        "-v", c["MODEL_REGISTRY"] + ":/config/models.json:ro",
        "-v", c["MODEL_DIR"] + ":/models" + ("" if online else ":ro"),
        "-e", "GPU_TYPE=" + c["GPU_TYPE"], "-e", "CPU_THREADS=" + c["CPU_THREADS"],
        "-e", "WARMUP_QUERIES=" + c["WARMUP_QUERIES"],
        "-e", "MODEL_NAME=" + c["MODEL_NAME"], "-e", "MODEL_ID=" + c["MODEL_ID"],
        "-e", "MODEL_REVISION=" + c["MODEL_REVISION"]]
    if online:
        command += ["-e", "HF_HUB_OFFLINE=0", "-e", "TRANSFORMERS_OFFLINE=0"]
    if gpu:
        check_gpu_idle(c)
        command += ["--gpus", "device=" + c["GPU"]]
    for mount in extra_mounts:
        command += ["-v", mount]
    return run([*command, c["WORKER_IMAGE"], module, *args])


def client(c, command, image=None, options=(), capture=False, allow_failure=False):
    image_required(c)
    Path(c["RESULT_DIR"]).mkdir(parents=True, exist_ok=True)
    args = ["docker", "run", "--rm", "--pull=never", "--user", container_user(),
        "--network", c["NETWORK_NAME"], "-v", c["RESULT_DIR"] + ":/results"]
    image_args = []
    if image:
        path = Path(image).resolve()
        if not path.is_file():
            raise ValueError("Input JPEG missing: " + str(path))
        args += ["-v", str(path) + ":/in/image.jpg:ro"]
        image_args = ["--image", "/in/image.jpg"]
    args += [c["WORKER_IMAGE"], "proteus_baselines.client", command, "--baseline", c["BASELINE"],
        "--controller", "baseline-worker:9000", "--output", "/results", "--model", c["APPLICATION"],
        "--slo", c["SLO_MS"], *image_args, *options]
    return run(args, capture=capture, allow_failure=allow_failure)


def up(c):
    image_required(c)
    state = container_status(c)
    if state and state.get("Running"):
        print("Already running; existing container/settings are not replaced")
        client(c, "state")
        return
    check_gpu_idle(c)
    for path in (c["MODEL_DIR"], c["RESULT_DIR"], str(Path(c["RUN_DIR"]) / "profiles"),
                 str(Path(c["RUN_DIR"]) / "logs/per_predictor")):
        Path(path).mkdir(parents=True, exist_ok=True)
    directory = ROOT / "deploy/docker/baselines/generated" / c["BASELINE"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "docker-compose.yml").write_text(json.dumps(compose_document(c, container_user()), indent=2))
    (directory / "settings.json").write_text(json.dumps(c, indent=2))
    compose(c, "up", "-d")
    until = time.monotonic() + 180
    while time.monotonic() < until:
        state = container_status(c)
        if state and state.get("Running") and state.get("Health", {}).get("Status") == "healthy":
            client(c, "state")
            return
        if state and state.get("Status") in {"exited", "dead"}:
            run(["docker", "logs", "--tail", "100", c["CONTAINER_NAME"]])
            raise RuntimeError("Baseline worker exited")
        time.sleep(2)
    raise RuntimeError("Baseline readiness timeout; inspect its logs")


def down(c):
    if container_status(c) is None:
        print("This baseline's container does not exist; other projects left untouched")
        return
    compose(c, "down")
    print("Only this baseline's container/network stopped. Images/checkpoints/profiles/results retained.")


def register(c, image=None, trials=None):
    return client(c, "profile", image or c["INPUT_IMAGE"], options=["--trials", str(trials or c["PROFILE_TRIALS"])])


def compare(args):
    configs = [settings(name) for name in BASELINES]
    if args.image:
        for c in configs:
            c["INPUT_IMAGE"] = str(Path(args.image).resolve())
    for c in configs:
        if container_status(c) is not None:
            raise ValueError("Stop your existing baseline containers explicitly before compare; none are stopped automatically")
        image_required(c)
        if not Path(c["INPUT_IMAGE"]).is_file():
            raise ValueError("Sample JPEG must already exist")
    rates = [float(v) for v in args.rates.split(",")]
    if not rates or any(not math.isfinite(v) or v <= 0 for v in rates):
        raise ValueError("Rates must be positive finite QPS")
    if not math.isfinite(args.duration) or args.duration <= 0 or args.concurrency < 1:
        raise ValueError("Invalid workload duration/concurrency")
    if not math.isfinite(args.shape) or args.shape <= 0:
        raise ValueError("Gamma shape must be positive and finite")
    reports = []
    for c in configs:
        created = False
        try:
            oneoff(c, "proteus_realtime.download", "weights", "--models", c["MODEL_NAME"], online=True)
            # Set ownership before startup so failed readiness still cleans up only
            # the name verified by container_status, never somebody else's service.
            created = True
            up(c)
            register(c)
            client(c, "test", c["INPUT_IMAGE"], options=["--warm-requests", "10"])
            for rate in rates:
                directory = Path(c["RESULT_DIR"])
                before = set(directory.glob("bench-*.json"))
                client(c, "bench", c["INPUT_IMAGE"], options=["--rate", str(rate), "--duration", str(args.duration),
                    "--distribution", args.distribution, "--shape", str(args.shape), "--seed", str(args.seed),
                    "--concurrency", str(args.concurrency)], allow_failure=True)
                new = set(directory.glob("bench-*.json")) - before
                if len(new) != 1:
                    raise RuntimeError("Workload failed before writing a measurement report")
                path = next(iter(new))
                report = json.loads(path.read_text())
                reports.append({"baseline": c["BASELINE"], "rate": rate, "report_file": str(path),
                    **{key: report.get(key) for key in ("scheduled_trace_sha256", "image_sha256", "requests",
                        "successful_replies", "failed_or_rejected", "slo_met_ratio_of_all_requests",
                        "successful_reply_qps", "slo_goodput_qps", "p50_e2e_ms", "p95_e2e_ms", "p99_e2e_ms",
                        "client_skipped_scheduled_arrivals", "max_client_send_lag_ms")}})
        finally:
            if created:
                down(c)
    for rate in rates:
        group = [row for row in reports if row["rate"] == rate]
        if len({row["scheduled_trace_sha256"] for row in group}) != 1 or len({row["image_sha256"] for row in group}) != 1:
            raise RuntimeError("Requested traces/images differ; do not call this a matched comparison")
    root = Path(configs[0]["DATA_ROOT"])
    output = root / ("comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.with_suffix(".json").write_text(json.dumps({"scope": "three singleton original-policy GPU adapters",
        "reports": reports, "accuracy_evaluation": False,
        "note": "Trace hash verifies scheduled arrivals, not realized send timing; inspect lag/skips. "
                "Cold/warm diagnostics use fixed1, benchmarks use native policies."}, indent=2))
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    print("COMPARISON SAVED:", output.with_suffix(".json"), output.with_suffix(".csv"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", choices=[*BASELINES, "all"])
    parser.add_argument("command", choices=["build", "settings", "download", "seed", "sample", "doctor", "up",
        "register", "state", "query", "test", "bench", "logs", "down", "compare"])
    parser.add_argument("--rates", default="1,5,10")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--distribution", choices=["uniform", "poisson", "gamma"], default="uniform")
    parser.add_argument("--shape", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--image")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--models", help="Optional name; must match MODEL_NAME in settings")
    parser.add_argument("--from", dest="source")
    args, extra = parser.parse_known_args(argv)
    if extra and args.command not in {"query", "test", "bench", "state"}:
        parser.error("Unknown options: " + " ".join(extra))
    if args.command == "compare":
        if args.baseline != "all" or extra:
            parser.error("Use all compare")
        compare(args)
        return 0
    if args.baseline == "all":
        parser.error("all is only supported by compare; use an explicit baseline for all other operations")
    c = settings(args.baseline)
    if args.models and args.models != c["MODEL_NAME"]:
        parser.error("--models must match MODEL_NAME; set MODEL_NAME/MODEL_ID/MODEL_REVISION in local.env")
    if args.command == "settings":
        print(json.dumps(c, indent=2))
    elif args.command == "build":
        sources = verify_sources()
        run(["docker", "build", "--build-arg", "PYTHON_IMAGE=" + c["PYTHON_IMAGE"],
            "--build-arg", "TORCH_VERSION=" + c["TORCH_VERSION"],
            "--build-arg", "TORCHVISION_VERSION=" + c["TORCHVISION_VERSION"],
            "-t", c["WORKER_IMAGE"], "-f", str(ROOT / "deploy/docker/baselines/worker.Dockerfile"), str(ROOT)])
        print("Verified original sources:", json.dumps(sources))
    elif args.command == "download":
        oneoff(c, "proteus_realtime.download", "weights", "--models", c["MODEL_NAME"], online=True)
    elif args.command == "seed":
        if not args.source or not Path(args.source).is_dir() or Path(args.source).resolve() == Path(c["MODEL_DIR"]):
            raise ValueError("--from must be a different existing HF_HOME directory")
        oneoff(c, "proteus_realtime.download", "seed", "--source", "/source", "--models", c["MODEL_NAME"],
               online=True, extra_mounts=[str(Path(args.source).resolve()) + ":/source:ro"])
    elif args.command == "sample":
        path = Path(c["INPUT_IMAGE"])
        if not path.exists():
            raise ValueError("Copy the EXISTING test-dog.jpg to INPUT_IMAGE; this script never downloads/replaces inputs")
        print("Existing JPEG:", path, "sha256=" + hashlib.sha256(path.read_bytes()).hexdigest())
    elif args.command == "doctor":
        oneoff(c, "proteus_baselines.doctor", gpu=True)
    elif args.command == "up":
        up(c)
    elif args.command == "down":
        down(c)
    elif args.command == "register":
        register(c, args.image, args.trials)
    elif args.command == "logs":
        container_status(c)
        run(["docker", "logs", "--tail", "100", c["CONTAINER_NAME"]])
    else:
        options = list(extra)
        if args.command == "bench":
            options += ["--duration", str(args.duration), "--distribution", args.distribution,
                "--shape", str(args.shape), "--seed", str(args.seed), "--concurrency", str(args.concurrency)]
        return client(c, args.command, (args.image or c["INPUT_IMAGE"]) if args.command in {"query", "test", "bench"} else None,
                      options=options, allow_failure=True).returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
