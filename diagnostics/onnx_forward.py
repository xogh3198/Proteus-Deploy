"""Isolated ONNX diagnostics using the running worker's immutable image.

Original weights/profiles are read-only. No register, policy edit, CPU power
change, CUDA-event assumption about ORT's private streams, or replacement model
selection is used. Host API timing is not pure GPU kernel timing.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, q):
    values = sorted(values)
    p = (len(values)-1)*q/100
    lo = int(p)
    return values[lo]+(values[min(lo+1, len(values)-1)]-values[lo])*(p-lo)


def resources():
    result = {}
    for name in ("cpu.max", "cpu.stat", "cpu.pressure", "cpuset.cpus.effective",
                 "cpu/cpu.cfs_quota_us", "cpu/cpu.cfs_period_us", "cpu/cpu.stat"):
        p = Path("/sys/fs/cgroup")/name
        try:
            result[name] = p.read_text().strip()
        except OSError:
            pass
    if hasattr(os, "sched_getaffinity"):
        result["affinity"] = sorted(os.sched_getaffinity(0))
    result["governors"] = {}
    for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"):
        try:
            result["governors"][p.parent.name] = p.read_text().strip()
        except OSError:
            pass
    return result


def ancestor_cpu_limits(pid):
    """Read host ancestors too: a container can inherit a parent's quota."""
    result = {}
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        relative = next(line[3:] for line in lines if line.startswith("0::"))
        root = Path("/sys/fs/cgroup").resolve()
        current = (root/relative.lstrip("/")).resolve()
        while current.is_relative_to(root):
            values = {}
            for name in ("cpu.max", "cpu.stat", "cpuset.cpus.effective"):
                try:
                    values[name] = (current/name).read_text().strip()
                except OSError:
                    pass
            if values:
                result[str(current)] = values
            if current == root:
                break
            current = current.parent
    except (OSError, StopIteration):
        pass
    return result


class Recorder:
    def __init__(self):
        self.rows, self.local = [], threading.local()
        self.case, self.details, self.active = "warmup", False, False

    def call(self, name, function, *args, **kwargs):
        row = getattr(self.local, "row", None)
        if row is None or not self.details:
            return function(*args, **kwargs)
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            return function(*args, **kwargs)
        finally:
            stage = row["stages"].setdefault(name, {"wall_ms": 0, "thread_cpu_ms": 0})
            stage["wall_ms"] += (time.perf_counter_ns()-wall)/1e6
            stage["thread_cpu_ms"] += (time.thread_time_ns()-cpu)/1e6

    def attach(self, runtime):
        real_infer, real_load = runtime.infer_prepared, runtime.load
        real_sync = runtime.torch.cuda.synchronize

        def infer(prepared):
            if not self.active:
                return real_infer(prepared)
            row = {"case": self.case, "observer": self.details, "model": runtime.current,
                   "batch": len(prepared), "thread_id": threading.get_native_id(), "stages": {}}
            self.local.row = row
            wall, cpu, process = time.perf_counter_ns(), time.thread_time_ns(), time.process_time_ns()
            try:
                return real_infer(prepared)
            finally:
                row["wall_ms"] = (time.perf_counter_ns()-wall)/1e6
                row["thread_cpu_ms"] = (time.thread_time_ns()-cpu)/1e6
                row["process_cpu_ms"] = (time.process_time_ns()-process)/1e6
                if self.details:
                    row["outside_api_wall_ms"] = max(0, row["wall_ms"]-sum(s["wall_ms"] for s in row["stages"].values()))
                self.rows.append(row)
                self.local.row = None

        def load(name=None):
            value = real_load(name)
            session = runtime.session
            if isinstance(session, SessionProbe):
                session = session.real
            runtime.session = SessionProbe(session, self) if self.details and session is not None else session
            return value

        runtime.infer_prepared, runtime.load = infer, load
        runtime.torch.cuda.synchronize = lambda *a, **k: self.call("cuda_synchronize", real_sync, *a, **k)

        def restore():
            runtime.infer_prepared, runtime.load = real_infer, real_load
            runtime.torch.cuda.synchronize = real_sync
            if isinstance(runtime.session, SessionProbe):
                runtime.session = runtime.session.real
        return restore


class BindingProbe:
    def __init__(self, real, recorder):
        self.real, self.recorder = real, recorder

    def __getattr__(self, name):
        return getattr(self.real, name)

    def bind_cpu_input(self, *args, **kwargs):
        return self.recorder.call("bind_cpu_input", self.real.bind_cpu_input, *args, **kwargs)

    def bind_output(self, *args, **kwargs):
        return self.recorder.call("bind_output", self.real.bind_output, *args, **kwargs)

    def copy_outputs_to_cpu(self):
        return self.recorder.call("copy_outputs_to_cpu", self.real.copy_outputs_to_cpu)


class SessionProbe:
    def __init__(self, real, recorder):
        self.real, self.recorder = real, recorder

    def __getattr__(self, name):
        return getattr(self.real, name)

    def io_binding(self):
        return BindingProbe(self.recorder.call("io_binding", self.real.io_binding), self.recorder)

    def run_with_iobinding(self, binding):
        return self.recorder.call("ort_run", self.real.run_with_iobinding, binding.real)


def summarize(rows, profiles):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["observer"], row["model"], row["batch"])].append(row)
    result = []
    for (case, observer, model, batch), samples in sorted(groups.items()):
        old = profiles["profiles"][model]["batches"][str(batch)]["samples_ms"]
        item = {"case": case, "observer": observer, "model": model, "batch": batch,
                "batch_calls": len(samples), "saved_profile_p50_ms": percentile(old, 50)}
        for key in ("wall_ms", "thread_cpu_ms", "process_cpu_ms"):
            item[key] = {"p50": percentile([r[key] for r in samples], 50),
                         "p95": percentile([r[key] for r in samples], 95)}
        item["stages"] = {}
        if observer:
            item["outside_api_wall_p50_ms"] = percentile([r["outside_api_wall_ms"] for r in samples], 50)
        for stage in sorted({s for r in samples for s in r["stages"]}):
            item["stages"][stage] = {key: percentile([r["stages"][stage][key] for r in samples], 50)
                                    for key in ("wall_ms", "thread_cpu_ms")}
        result.append(item)
    return result


def rpc_case(runtime, recorder, profile_path, raw, arrivals, output, detailed):
    import grpc
    from experiments.reproduce.proteus_executor import Executor
    from experiments.resnet.client import Client, replay, summary
    from proteus_realtime.protocol import InferRequest, InferReply, INFER_METHOD
    from proteus_realtime.server import create_server
    profiles = json.loads(profile_path.read_text())
    recorder.active = False
    runtime.reset()
    executor = Executor(runtime, "proteus", profile_path=profile_path, slo=profiles["slo_ms"],
                        batches=profiles["batches"], interval=float(os.environ["ALLOCATION_INTERVAL"]))
    server, channel = None, None
    try:
        if not executor.state()["policy_ready"]:
            raise RuntimeError("Saved profile is incompatible with this image/runtime; register is deliberately not called")
        # Recreate the session on the same GPU executor thread used for serving.
        executor.warmup()
        recorder.case = "native_rpc_twitter"
        server, port = create_server(executor, "127.0.0.1:0", int(os.environ["RPC_THREADS"]))
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(channel).result(timeout=10)
        client = Client.__new__(Client)
        client.engine, client.timeout, client.channel = "proteus", 10, channel
        client.Request = InferRequest
        client.rpc = channel.unary_unary(INFER_METHOD, request_serializer=InferRequest.SerializeToString,
                                        response_deserializer=InferReply.FromString)
        recorder.active = True
        rows, elapsed = replay(client, raw, arrivals, profiles["slo_ms"], 1, 256, use_trace_slo=True)
        recorder.active = False
        result = summary(rows, arrivals[-1][0]/1000, profiles["slo_ms"])
        result["elapsed_seconds"] = elapsed
        (output/f"rpc-observer-{int(detailed)}.json").write_text(json.dumps({"summary": result, "rows": rows}, indent=2))
        if not result["successful_replies"]:
            raise RuntimeError("No native RPC inference completed; inspect saved rpc-observer results before interpreting timings")
    finally:
        recorder.active = False
        if channel is not None:
            channel.close()
        if server is not None:
            server.stop(5).wait()
        executor.close()


def measure(runtime, args, profile_path, raw, arrivals, output):
    profiles = json.loads(profile_path.read_text())
    if profiles["fingerprint"] != runtime.fingerprint():
        raise RuntimeError("Saved profile/runtime fingerprint differs; diagnostic will not reprofile or overwrite it")
    if any(batch not in profiles["batches"] for batch in args.batches):
        raise ValueError("Diagnostic batches must already exist in the saved profile")
    recorder = Recorder()
    restore = recorder.attach(runtime)
    report = {"status": "running", "note": "Host API wall/CPU times, not GPU kernel durations. Calling-thread CPU excludes ORT worker threads.",
              "original_profile_sha256": digest(profile_path), "resources": [], "rows": recorder.rows}
    try:
        prepared = runtime.prepare(raw)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="diag-direct-gpu") as pool:
            for gap in args.gaps_ms:
                for detailed in (False, True):
                    recorder.details, recorder.active = detailed, False
                    pool.submit(runtime.load, args.model).result()
                    for batch in args.batches:
                        recorder.case = f"direct_gap_{gap:g}ms"
                        print(f"START {recorder.case} batch={batch} observer={int(detailed)}", flush=True)
                        for _ in range(3):
                            time.sleep(gap/1000)
                            pool.submit(runtime.infer_prepared, [prepared]*batch).result()
                        before = resources()
                        recorder.active = True
                        for _ in range(args.trials):
                            time.sleep(gap/1000)
                            pool.submit(runtime.infer_prepared, [prepared]*batch).result()
                        recorder.active = False
                        report["resources"].append({"case": recorder.case, "observer": detailed,
                            "batch": batch, "before": before, "after": resources()})
            pool.submit(runtime.reset).result()
        if arrivals:
            for detailed in (False, True):
                recorder.details = detailed
                print(f"START native_rpc_twitter requests={len(arrivals)} observer={int(detailed)}", flush=True)
                before = resources()
                rpc_case(runtime, recorder, profile_path, raw, arrivals, output, detailed)
                report["resources"].append({"case": "native_rpc_twitter", "observer": detailed,
                                             "before": before, "after": resources()})
        report["status"] = "complete"
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        recorder.active = False
        restore()
        report["summary"] = summarize(recorder.rows, profiles)
        (output/"forward-diagnostic.json").write_text(json.dumps(report, indent=2))
        print("case / observer / model / batch / calls / profile-P50 / actual-P50-P95 / thread-CPU-P50 / process-CPU-P50 (ms)", flush=True)
        for item in report["summary"]:
            print(f"{item['case']} / {int(item['observer'])} / {item['model'].removesuffix('_v1')} / {item['batch']} / {item['batch_calls']} / "
                  f"{item['saved_profile_p50_ms']:.2f} / {item['wall_ms']['p50']:.2f}-{item['wall_ms']['p95']:.2f} / "
                  f"{item['thread_cpu_ms']['p50']:.2f} / {item['process_cpu_ms']['p50']:.2f}", flush=True)
            for name, stage in item["stages"].items():
                print(f"  {name}: wall-P50={stage['wall_ms']:.3f}, CPU-P50={stage['thread_cpu_ms']:.3f}", flush=True)
            if item["observer"]:
                print(f"  outside_observed_APIs: wall-P50={item['outside_api_wall_p50_ms']:.3f}", flush=True)
    return report


def docker_status(container):
    program = ("import grpc,json; c=grpc.insecure_channel('127.0.0.1:9000'); "
               "f=c.unary_unary('/paired.resnet.Admin/Status',request_serializer=lambda x:json.dumps(x).encode(),"
               "response_deserializer=json.loads); print(json.dumps(f({},timeout=10)))")
    return json.loads(subprocess.check_output(["docker", "exec", container, "python", "-c", program], text=True))


def launch_plan(inspect, script, output, args):
    mounts = {m["Destination"]: m for m in inspect["Mounts"]}
    for target in ("/models", "/inputs/image.jpg", "/profiles"):
        if target not in mounts or mounts[target]["Type"] != "bind":
            raise ValueError("Required original bind mount missing: "+target)
    devices = inspect["HostConfig"].get("DeviceRequests") or []
    if len(devices) != 1 or len(devices[0].get("DeviceIDs") or []) != 1:
        raise ValueError("Diagnostic requires one explicitly selected original GPU")
    gpu = devices[0]["DeviceIDs"][0]
    name = "proteus-forward-diag-"+uuid.uuid4().hex[:10]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--gpus", "device="+gpu]
    if inspect["Config"].get("User"):
        command += ["--user", inspect["Config"]["User"]]
    for value in inspect["Config"]["Env"]:
        command += ["-e", value]
    for target in ("/models", "/inputs/image.jpg", "/profiles"):
        command += ["--mount", "type=bind,src="+mounts[target]["Source"]+",dst="+target+",readonly"]
    command += ["--mount", f"type=bind,src={script},dst=/diagnostics/onnx_forward.py,readonly",
                "--mount", f"type=bind,src={output},dst=/results",
                "--mount", f"type=bind,src={mounts['/models']['Source']}/trace-twitter.txt,dst=/diagnostics/trace.txt,readonly"]
    # Preserve any original CPU quota/affinity rather than silently removing it.
    for field, flag in (("NanoCpus", "--cpus"), ("CpuQuota", "--cpu-quota"), ("CpuPeriod", "--cpu-period"),
                        ("CpuShares", "--cpu-shares"), ("CpusetCpus", "--cpuset-cpus"), ("CpusetMems", "--cpuset-mems")):
        value = inspect["HostConfig"].get(field)
        if value:
            command += [flag, str(value/1e9 if field == "NanoCpus" else value)]
    command += [inspect["Image"], "python", "/diagnostics/onnx_forward.py", "--worker", "--model", args.model,
                "--trials", str(args.trials), "--trace-limit", str(args.trace_limit),
                "--batches", ",".join(map(str, args.batches)), "--gaps-ms", ",".join(map(str, args.gaps_ms))]
    return command, name, gpu, Path(mounts["/profiles"]["Source"])/"published-source-profiles.json"


def host_main(args):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from experiments.resnet.host import settings, docker_info
    c = settings("proteus", published_workload="twitter")
    docker_info(c)
    found = subprocess.check_output(["docker", "ps", "-q", "--filter", "label=com.docker.compose.project="+c["PROJECT"],
                                     "--filter", "label=com.docker.compose.service=worker"], text=True).split()
    if len(found) != 1:
        raise RuntimeError("Start the registered Twitter worker first; exactly one running worker is required")
    container = found[0]
    inspect = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[0]
    state = docker_status(container)
    if not state.get("policy_ready") or state.get("busy") or state.get("outstanding_requests") or state.get("maintenance"):
        raise RuntimeError("Worker must be registered and idle; no existing requests will be interrupted")
    if state.get("experiment_phase") != "v1":
        raise RuntimeError("This diagnostic targets the preprocessing-excluded v1 experiment")
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    output = Path(c["RUN_DIR"])/"results"/"forward-diagnostics"/token
    command, name, gpu, profile_path = launch_plan(inspect, Path(__file__).resolve(), output, args)
    before = digest(profile_path)
    metadata = {"source_container": inspect["Name"].lstrip("/"), "image_id": inspect["Image"],
                "profile_sha256": before, "gpu": gpu, "output": str(output), "settings": vars(args),
                "ancestor_cpu_limits": ancestor_cpu_limits(inspect["State"]["Pid"]),
                "warning": "Original worker will restart; in-memory counters/history and residency reset. Profiles/results are preserved."}
    shown = {**metadata, "settings": {**vars(args), "model": args.model.removesuffix("_v1")}}
    print(json.dumps(shown, indent=2), flush=True)
    if args.dry_run:
        print("DRY RUN: no containers stopped/created and no files written")
        return
    output.mkdir(parents=True, exist_ok=False)
    (output/"host.json").write_text(json.dumps(metadata, indent=2))
    stopped = False
    try:
        stopped = True
        subprocess.run(["docker", "stop", container], check=True)
        other = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
        if other:
            raise RuntimeError("Selected GPU still has compute processes; no other service will be stopped: "+other)
        with (output/"container.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError("Diagnostic failed; inspect "+str(output/"container.log"))
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ancestors_after = {}
        for path in metadata["ancestor_cpu_limits"]:
            values = {}
            for field in ("cpu.max", "cpu.stat", "cpuset.cpus.effective"):
                try:
                    values[field] = (Path(path)/field).read_text().strip()
                except OSError:
                    pass
            if values:
                ancestors_after[path] = values
        if stopped:
            subprocess.run(["docker", "start", container], check=True)
            for _ in range(90):
                try:
                    if docker_status(container).get("policy_ready"):
                        break
                except (subprocess.SubprocessError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Original container restarted but is not ready; inspect its Docker logs")
            print("RESTORED original worker; counters/history reset", flush=True)
        (output/"host-after.json").write_text(json.dumps(ancestors_after, indent=2))
        if digest(profile_path) != before:
            raise RuntimeError("Original profile hash changed outside the read-only diagnostic")
        print("Original profile SHA256 unchanged. Results:", output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", default="resnet152")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--trace-limit", type=int, default=100)
    p.add_argument("--batches", default="1,8")
    p.add_argument("--gaps-ms", default="0,100,300")
    args = p.parse_args()
    if not args.model.endswith("_v1"):
        args.model += "_v1"
    if args.model not in {f"resnet{n}_v1" for n in (18,34,50,101,152)}:
        p.error("model must be resnet18, resnet34, resnet50, resnet101, or resnet152")
    args.batches = [int(x) for x in args.batches.split(",")]
    args.gaps_ms = [float(x) for x in args.gaps_ms.split(",")]
    if not 2 <= args.trials <= 100 or not 0 <= args.trace_limit <= 3776 or not args.batches or not args.gaps_ms:
        p.error("trials=2..100, trace-limit=0..3776, nonempty batches/gaps required")
    if any(b not in (1,2,4,8) for b in args.batches) or any(not 0 <= g <= 1000 for g in args.gaps_ms):
        p.error("batches=1,2,4,8 and finite gaps=0..1000ms required")
    if args.worker:
        from experiments.resnet.runtime import Runtime
        from experiments.resnet.client import read_trace, validate_published_trace
        raw = Path("/inputs/image.jpg").read_bytes()
        trace = Path("/diagnostics/trace.txt")
        arrivals = read_trace(trace)
        validate_published_trace(trace, arrivals, "twitter", float(os.environ["SLO_MS"]), 1)
        arrivals = arrivals[:args.trace_limit] if args.trace_limit else []
        runtime = Runtime()
        try:
            measure(runtime, args, Path("/profiles/published-source-profiles.json"), raw, arrivals, Path("/results"))
        finally:
            runtime.reset()
    else:
        host_main(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("ERROR:", error, file=sys.stderr, flush=True)
        raise SystemExit(1)
