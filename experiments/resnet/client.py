"""One open-loop schedule and metric definition for the four native APIs."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import threading
import time
import urllib.error
import urllib.request
import uuid

from .assets import manifest


def http(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class Client:
    def __init__(self, engine, timeout=10):
        self.engine, self.timeout = engine, timeout
        if engine == "clipper":
            return
        import grpc
        port = 50052 if engine == "infaas" else 9000
        self.channel = grpc.insecure_channel(f"worker:{port}", options=(
            ("grpc.max_receive_message_length", 32 << 20), ("grpc.max_send_message_length", 32 << 20)))
        grpc.channel_ready_future(self.channel).result(timeout=timeout)
        if engine == "infaas":
            from infaas.proto import queryfe_pb2, queryfe_pb2_grpc, request_reply_pb2
            self.pb, self.success = queryfe_pb2, request_reply_pb2.SUCCESS
            self.stub = queryfe_pb2_grpc.QueryStub(self.channel)
        else:
            from proteus_realtime.protocol import InferRequest, InferReply, INFER_METHOD
            self.Request = InferRequest
            self.rpc = self.channel.unary_unary(INFER_METHOD,
                request_serializer=InferRequest.SerializeToString, response_deserializer=InferReply.FromString)

    def admin(self, action, payload=None, timeout=900):
        if self.engine == "clipper":
            if action == "Status":
                value = http("http://worker:9090/state")
                value["cpu_stage"] = http("http://gateway:8080/preparation")
                replicas = http("http://management:1338/admin/get_all_containers", {"verbose": True})
                value["native_registered"] = any(r.get("model_name") == os.environ["MODEL_NAME"] for r in replicas)
                value["policy_ready"] = value["native_registered"]
                return value
            if action == "Reset":
                return http("http://worker:9090/reset", {})
            if action == "Warmup":
                return http("http://worker:9090/warmup", {}, timeout=timeout)
            return {"policy_ready": True, "note": "Native Clipper learns batching online; no offline profile is substituted."}
        call = self.channel.unary_unary("/paired.resnet.Admin/"+action,
            request_serializer=lambda data: json.dumps(data).encode(), response_deserializer=json.loads)
        return call(payload or {}, timeout=timeout)

    def query(self, raw, slo, request_id):
        if self.engine == "clipper":
            payload = {"input": json.dumps({"request_id": request_id,
                                           "image_b64": base64.b64encode(raw).decode()})}
            request = urllib.request.Request("http://gateway:8080/imagenet/predict",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        elif self.engine == "infaas":
            request = self.pb.QueryOnlineRequest(parent_model="imagenet", raw_input=[raw])
            request.slo.LatencyInUSec = int(slo*1000)
            request.slo.MinAccuracy = 0
        else:
            request = self.Request(model="imagenet", image=raw, slo=slo)
        started = time.perf_counter()
        row = {"sent": True, "ok": False, "model": "", "batch_size": 0, "server_ms": None,
               "timings": {}, "reject_kind": "", "error": "", "rpc_started": started}
        try:
            if self.engine == "clipper":
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.load(response)
                row["ok"] = not value.get("default", False) and "output" in value
                if row["ok"]:
                    output = value["output"]
                    output = json.loads(output) if isinstance(output, str) else output
                    row.update(model=output["model"], batch_size=output["batch_size"],
                               server_ms=output["server_e2e_ms"], timings=output["timings"])
                else:
                    row.update(reject_kind="slo_fallback", error=value.get("default_explanation", str(value)))
            elif self.engine == "infaas":
                response = self.stub.QueryOnline(request, timeout=self.timeout)
                row.update(ok=response.status.status == self.success,
                    model=response.variant.split("__", 1)[0], server_ms=response.latency_ms,
                    timings=dict(response.timings), reject_kind=response.reject_kind,
                    error=response.status.msg)
                row["batch_size"] = int(row["timings"].get("worker.batch_size", 0))
            else:
                response = self.rpc(request, timeout=self.timeout)
                row.update(ok=not response.rejected, model=response.variant, server_ms=response.latency,
                    timings=dict(response.timings), reject_kind=response.reject_kind, error=response.reason)
                row["batch_size"] = int(row["timings"].get("worker.batch_size", 0))
        except Exception as error:
            row.update(reject_kind=type(error).__name__, error=str(error))
        row["e2e_ms"] = (time.perf_counter()-started)*1000
        row["slo_met"] = bool(row["ok"] and row["e2e_ms"] <= slo)
        return row


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    position = (len(values)-1)*percent/100
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right]-values[left])*(position-left)


def read_trace(path, limit=0):
    rows, previous = [], -1
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 3:
            raise ValueError(f"Invalid trace line {number}")
        arrival, qos, source_slo = map(int, parts)
        if arrival < previous or arrival < 0 or source_slo <= 0:
            raise ValueError("Invalid or unordered trace")
        rows.append((arrival, qos, source_slo))
        previous = arrival
    if not rows or limit < 0:
        raise ValueError("Nonempty trace and nonnegative limit required")
    return rows[:limit] if limit else rows


def replay(client, raw, arrivals, slo, speed, concurrency):
    """No closed-loop feedback or hidden client backlog changes the arrivals."""
    if not math.isfinite(speed) or speed <= 0 or not 1 <= concurrency <= 1024:
        raise ValueError("Invalid replay speed or concurrency")
    slots = threading.BoundedSemaphore(concurrency)
    rows = [None]*len(arrivals)
    futures = []
    origin = time.perf_counter()
    next_report = origin+10

    def send(index, target, qos, source_slo):
        try:
            row = client.query(raw, slo, uuid.uuid4().hex)
            row.update(index=index, arrival_ms=arrivals[index][0], qos=qos, source_slo_ms=source_slo,
                       send_lag_ms=max(0, (row.pop("rpc_started")-target)*1000))
            row["scheduled_slo_met"] = row["slo_met"] and row["e2e_ms"]+row["send_lag_ms"] <= slo
            rows[index] = row
        finally:
            slots.release()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for index, (arrival, qos, source_slo) in enumerate(arrivals):
            target = origin+arrival/1000/speed
            remaining = target-time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            if slots.acquire(blocking=False):
                futures.append(pool.submit(send, index, target, qos, source_slo))
            else:
                rows[index] = {"index": index, "arrival_ms": arrival, "qos": qos, "source_slo_ms": source_slo,
                               "sent": False, "ok": False, "slo_met": False, "scheduled_slo_met": False,
                               "reject_kind": "client_not_sent", "error": "Client in-flight limit", "timings": {}}
            if time.perf_counter() >= next_report:
                print(f"scheduled={index+1}/{len(arrivals)} elapsed={time.perf_counter()-origin:.1f}s", flush=True)
                next_report = time.perf_counter()+10
        for future in futures:
            future.result()
    return rows, time.perf_counter()-origin


def summary(rows, duration, slo=200):
    sent = [r for r in rows if r["sent"]]
    successful = [r for r in sent if r["ok"]]
    good = [r for r in successful if r["slo_met"]]
    values = [r["e2e_ms"] for r in successful]
    counts, batches = {}, {}
    scores = manifest()["models"]
    for row in successful:
        counts[row["model"]] = counts.get(row["model"], 0)+1
        key = str(row["batch_size"])
        batches[key] = batches.get(key, 0)+1
    lags = [r["send_lag_ms"] for r in sent]
    unknown = sorted(set(counts)-set(scores))
    server_values = [r["server_ms"] for r in successful if r.get("server_ms") is not None]
    server_good = sum(v <= slo for v in server_values)
    return {"scheduled_requests": len(rows), "sent_requests": len(sent), "client_not_sent": len(rows)-len(sent),
            "successful_replies": len(successful), "slo_met_sent_requests": len(good),
            "successful_late_replies": len(successful)-len(good),
            "failed_or_rejected_sent_requests": len(sent)-len(successful),
            "slo_unmet_ratio_of_sent_requests": 1-len(good)/len(sent) if sent else None,
            "slo_unmet_ratio_of_all_scheduled_requests": 1-len(good)/len(rows),
            "scheduled_deadline_unmet_ratio": 1-sum(r["scheduled_slo_met"] for r in rows)/len(rows),
            "successful_requests_by_model": counts, "response_counts_by_batch": batches,
            "mean_reference_accuracy_of_successful_replies":
                fmean(scores[r["model"]]["accuracy"] for r in successful) if successful and not unknown else None,
            "mean_reference_accuracy_of_slo_met_replies":
                fmean(scores[r["model"]]["accuracy"] for r in good) if good and not unknown else None,
            "mean_e2e_ms": fmean(values) if values else None,
            "p50_e2e_ms": percentile(values, 50), "p95_e2e_ms": percentile(values, 95),
            "p99_e2e_ms": percentile(values, 99), "latency_population": "successful replies, including late replies",
            "planned_mean_qps": len(rows)/duration, "slo_met_qps": len(good)/duration,
            "max_client_send_lag_ms": max(lags) if lags else None,
            "requests_above_20ms_send_lag": sum(v > 20 for v in lags),
            "trace_timing_valid": len(sent) == len(rows) and all(v <= 20 for v in lags),
            "server_internal": {"successful_latency_samples": len(server_values),
                "slo_met_successful_replies": server_good,
                "slo_unmet_ratio_of_sent_including_failures": 1-server_good/len(sent) if sent else None,
                "mean_ms": fmean(server_values) if server_values else None,
                "p50_ms": percentile(server_values, 50), "p95_ms": percentile(server_values, 95),
                "boundary_note": "Native server boundaries differ: Clipper includes ingress/native frontend; use client E2E for cross-system comparison."},
            "accuracy_note": "Supplied Proteus model reference scores; not measured checkpoint/image correctness",
            "unknown_variants": unknown}


def native_register():
    app = {"name": "imagenet", "input_type": "strings", "default_output": "CLIPPER_SLO_FALLBACK",
           "latency_slo_micros": int(float(os.environ["SLO_MS"])*1000)}
    model = {"model_name": os.environ["MODEL_NAME"], "model_version": "1", "labels": [],
             "input_type": "strings", "container_name": os.environ["WORKER_IMAGE"],
             "model_data_path": "DEPRECATED", "batch_size": -1}
    base = "http://management:1338/admin/"
    existing = http(base+"get_all_applications", {"verbose": True})
    selected = next((x for x in existing if x["name"] == "imagenet"), None)
    if selected is None:
        http(base+"add_app", app)
    elif any(str(selected.get(k)) != str(v) for k, v in app.items()):
        raise ValueError("Existing Clipper application differs; down/up only this experiment")
    models = http(base+"get_all_models", {"verbose": True})
    if not any(x["model_name"] == model["model_name"] and str(x["model_version"]) == "1" for x in models):
        http(base+"add_model", model)
    http(base+"set_model_version", {"model_name": model["model_name"], "model_version": "1"})
    existing = http(base+"get_all_applications", {"verbose": True})
    linked = next(x for x in existing if x["name"] == "imagenet").get("linked_models", [])
    if not linked:
        http(base+"add_model_links", {"app_name": "imagenet", "model_names": [model["model_name"]]})
    elif linked != [model["model_name"]]:
        raise ValueError("Unexpected fixed Clipper model")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["state", "health", "register", "reset", "query", "trace", "native-register"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--speed", type=float, default=float(os.environ.get("TRACE_SPEED", "2")))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("TRACE_CONCURRENCY", "256")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("RPC_TIMEOUT_SECONDS", "10")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.speed) or args.speed <= 0 or not math.isfinite(args.timeout)
            or args.timeout <= 0 or not 1 <= args.concurrency <= 1024):
        parser.error("speed/timeout must be positive and finite; concurrency must be 1..1024")
    if args.action == "native-register":
        return native_register()
    raw = Path("/inputs/image.jpg").read_bytes()
    slo = float(os.environ["SLO_MS"])
    if args.action == "trace":
        path = Path("/data/trace.txt")
        arrivals = read_trace(path, args.limit)
        duration = arrivals[-1][0]/1000/args.speed
        config = {"engine": os.environ["ENGINE"], "phase": os.environ["EXPERIMENT_PHASE"],
            "dtype": "fp32", "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_requests": len(arrivals), "speed": args.speed, "planned_duration_seconds": duration,
            "slo_override_ms": slo, "source_slo_values": sorted({r[2] for r in arrivals}),
            "concurrency": args.concurrency, "timeout_seconds": args.timeout,
            "input_sha256": hashlib.sha256(raw).hexdigest(), "source_image_repeated": True,
            "start_state": os.environ["START_STATE"], "image_id": os.environ.get("EXPERIMENT_IMAGE_ID"),
            "source_commit": os.environ.get("EXPERIMENT_SOURCE_COMMIT"),
            "source_sha256": os.environ.get("EXPERIMENT_SOURCE_SHA256"),
            "configured_model": os.environ.get("MODEL_NAME"), "clipper_mode": os.environ.get("CLIPPER_MODE"),
            "settings": {k: os.environ.get(k) for k in ["HOST_GPU_INDEX", "CPU_THREADS", "CPU_INTEROP_THREADS",
                "PREPROCESS_WORKERS", "RPC_THREADS", "WORKER_THREADS", "BATCH_SIZES", "MAX_QUEUE_SIZE",
                "WARMUP_QUERIES", "PROFILE_TRIALS", "ALLOCATION_INTERVAL", "EWMA_ALPHA", "INITIAL_RATE",
                "DEADLINE_GUARD_MS", "RPC_BUDGET_MS", "HW_COST", "LAMBDA", "SLACK_THRESHOLD", "MIN_ACCURACY", "QUERY_THREADS"]},
            "slo_definition": "client RPC call start to response; scheduled send lag reported separately"}
        if args.dry_run:
            print(json.dumps(config, indent=2)); return
    client = Client(os.environ["ENGINE"], args.timeout)
    if args.action in {"state", "health"}:
        state = client.admin("Status", timeout=10)
        if args.action == "health":
            if state.get("native_registered", True) is not True:
                raise SystemExit("Native Clipper worker not yet registered")
            print("healthy"); return
        native = state.get("native_state", state)
        keys = ["policy_ready", "policy_error", "plan", "plan_details", "selected_model", "resident_model", "batch_size_counts",
                "successful_requests_by_model", "processed", "failed", "deadline_drops"]
        print(json.dumps({**{k: native.get(k) for k in keys}, **{k: state.get(k) for k in
            ["engine", "experiment_phase", "dtype", "model_name", "fixed_model", "enabled_models", "resident", "actual_load_count",
             "model_switches_after_initial_load", "cpu_stage"]}}, indent=2)); return
    if args.action == "register":
        print(json.dumps(client.admin("Profile"), indent=2)); return
    if args.action == "reset":
        print(json.dumps(client.admin("Reset"))); return
    if args.action == "query":
        value = client.query(raw, slo, uuid.uuid4().hex)
        value.pop("rpc_started", None)
        print(json.dumps(value, indent=2)); return
    initial = client.admin("Status")
    if not initial.get("native_state", initial).get("policy_ready", os.environ["ENGINE"] == "clipper"):
        raise ValueError("Run register before trace")
    if os.environ["START_STATE"] == "cold":
        client.admin("Reset")
    elif os.environ["START_STATE"] == "active":
        client.admin("Warmup")
    else:
        raise ValueError("START_STATE must be cold or active")
    before = client.admin("Status")
    native = before.get("native_state", before)
    if not native.get("policy_ready", os.environ["ENGINE"] == "clipper"):
        raise ValueError("Run register before trace")
    rows, elapsed = replay(client, raw, arrivals, slo, args.speed, args.concurrency)
    after = client.admin("Status")
    result = summary(rows, duration, slo)
    result["elapsed_seconds_including_drain"] = elapsed
    result["model_load_events_during_run"] = [e for e in after["actual_load_events"]
        if e["id"] > before["actual_load_count"]]
    events = result["model_load_events_during_run"]
    result["model_loads_during_run"] = len(events)
    result["model_switches_during_run"] = sum(e["previous_model"] is not None for e in events)
    result["total_model_load_ms_during_run"] = sum(e["load_ms"] for e in events)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    output = Path("/results") / ("trace-"+token)
    document = {"config": config, "summary": result, "worker_before": before, "worker_after": after}
    output.with_suffix(".json").write_text(json.dumps(document, indent=2, allow_nan=False))
    fields = ["index", "arrival_ms", "qos", "source_slo_ms", "sent", "ok", "slo_met", "scheduled_slo_met",
              "e2e_ms", "server_ms", "send_lag_ms", "model", "batch_size", "reject_kind", "error", "timings"]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "timings": json.dumps(row["timings"])})
    print(json.dumps(result, indent=2))
    print("Results:", output.with_suffix(".json"), output.with_suffix(".csv"))


if __name__ == "__main__":
    main()
