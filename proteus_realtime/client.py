"""Inference, cold/warm, profiling and bounded rate-driven workload clients."""
import argparse
import base64
import concurrent.futures
import csv
import json
import math
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import grpc

from .metrics import summarize, compact_bench_report
from .protocol import ADMIN_SERVICE, INFER_METHOD, InferReply, InferRequest

OPTIONS = [("grpc.max_receive_message_length", 32 * 1024 * 1024),
           ("grpc.max_send_message_length", 32 * 1024 * 1024)]


def admin_call(channel, name, payload=None, timeout=180):
    rpc = channel.unary_unary(
        f"/{ADMIN_SERVICE}/{name}",
        request_serializer=lambda value: json.dumps(value).encode(),
        response_deserializer=lambda data: json.loads(data.decode()))
    return rpc(payload or {}, timeout=timeout)


def query_once(channel, model, image, slo, timeout=180, index=0):
    rpc = channel.unary_unary(INFER_METHOD, request_serializer=InferRequest.SerializeToString,
                             response_deserializer=InferReply.FromString)
    started = time.perf_counter()
    try:
        reply = rpc(InferRequest(model=model, image=image, slo=slo), timeout=timeout)
        elapsed = (time.perf_counter() - started) * 1000
        timings = dict(reply.timings)
        return {"index": index, "ok": not reply.rejected, "e2e_ms": elapsed,
                "server_ms": reply.latency, "slo_ms": slo,
                "slo_met": not reply.rejected and elapsed <= slo,
                "client_e2e_slo_met": not reply.rejected and elapsed <= slo,
                "server_slo_met": not reply.rejected and reply.latency <= slo,
                "server_budget_met": not reply.rejected and reply.latency <= timings.get("worker.server_budget_ms", slo),
                "load_ms": timings.get("worker.load", 0),
                "queue_ms": timings.get("worker.queue", 0),
                "service_ms": timings.get("worker.service", 0),
                "path": "rejected" if reply.rejected else (
                    "inactive" if timings.get("worker.model_loaded", 0) else "active"),
                "variant": reply.variant,
                "batch_size": int(timings.get("worker.batch_size", 0 if reply.rejected else 1)),
                "label": reply.inference, "gpu": reply.gpu,
                "request_name": reply.request_name, "reject_kind": reply.reject_kind,
                "error": reply.reason, "timings": timings}
    except grpc.RpcError as error:
        return {"index": index, "ok": False,
                "e2e_ms": (time.perf_counter() - started) * 1000,
                "server_ms": None, "slo_ms": slo, "slo_met": False,
                "client_e2e_slo_met": False, "server_slo_met": False, "server_budget_met": False,
                "load_ms": None, "queue_ms": None, "service_ms": None,
                "path": "rpc_error", "label": "", "gpu": "", "request_name": "",
                "variant": "", "batch_size": 0,
                "reject_kind": error.code().name, "error": error.details(), "timings": {}}


def print_row(row):
    def number(key):
        return "-" if row[key] is None else f"{row[key]:.1f}"
    print(f"{'OK' if row['ok'] else 'ERR'} e2e={row['e2e_ms']:7.1f}ms "
          f"server={number('server_ms')}ms gpu={row['gpu'] or '-'} "
          f"path={row['path']} label={row['label']!r} "
          f"variant={row['variant'] or '-'} batch={row['batch_size']} "
          f"load={number('load_ms')}ms queue={number('queue_ms')}ms "
          f"SLO={'MET' if row['slo_met'] else 'MISSED/FAILED'}"
          + (f" [{row['reject_kind']}] {row['error']}" if not row['ok'] else ""), flush=True)


def _write_report(output, stem, rows, report):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    base = directory / f"{stem}-{stamp}"
    fields = list(rows[0]) if rows else ["index", "ok", "e2e_ms"]
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "timings": json.dumps(row.get("timings", {}))})
    base.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"saved: {base.name}.csv / {base.name}.json", flush=True)


def _positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _count(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["query", "test", "bench", "state", "reset", "profile", "mode", "evaluate", "policy-test"])
    ap.add_argument("--controller", default="127.0.0.1:18081")
    ap.add_argument("--model", default="resnet-50")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--mode", choices=["fixed", "proteus"], default="proteus")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--image")
    ap.add_argument("--slo", type=_positive, default=200)
    ap.add_argument("-n", type=_count, default=1)
    ap.add_argument("--warm-requests", type=_count, default=10)
    ap.add_argument("--warmup", type=_count, default=5)
    ap.add_argument("--trials", type=_count, default=30)
    ap.add_argument("--rate", type=_positive, default=10)
    ap.add_argument("--duration", type=_positive, default=30)
    ap.add_argument("--distribution", choices=["uniform", "poisson", "gamma"], default="uniform")
    ap.add_argument("--shape", type=_positive, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=_count, default=64)
    ap.add_argument("--timeout", type=_positive, default=180)
    ap.add_argument("--output", default="data/proteus/results")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--watch", type=_positive)
    return ap


def cold_warm(channel, args, image):
    before_mode = admin_call(channel, "Status")
    standalone = before_mode.get("implementation") == "proteus-single-gpu-port"
    if standalone:
        admin_call(channel, "Mode", {"mode": "fixed"})
    try:
        return _cold_warm(channel, args, image)
    finally:
        if standalone and before_mode.get("mode") == "proteus":
            admin_call(channel, "Mode", {"mode": "proteus"})


def _cold_warm(channel, args, image):
    admin_call(channel, "Reset")
    before = admin_call(channel, "Status")
    if before["resident"]:
        raise RuntimeError("Reset did not make the model non-resident")
    first_started = time.perf_counter()
    first = query_once(channel, args.model, image, args.slo, args.timeout, 0)
    first_elapsed = time.perf_counter() - first_started
    print("FIRST (model non-resident; not OS/CUDA-cache cold)")
    print_row(first)
    if not first["ok"]:
        raise RuntimeError("First inference failed; inspect worker logs before interpreting warm results")
    rows = [first]
    warm_started = time.perf_counter()
    for index in range(1, args.warm_requests + 1):
        row = query_once(channel, args.model, image, args.slo, args.timeout, index)
        rows.append(row)
        print_row(row)
    warm_elapsed = time.perf_counter() - warm_started
    warm = summarize(rows[1:], warm_elapsed)
    report = {
        "experiment": "independent fixed1 cold/warm diagnostic, NOT Proteus policy comparison",
        "cold_definition": "model non-resident; CUDA context, OS page cache and libraries may be warm",
        "first": first, "first_elapsed_seconds": first_elapsed, "warm": warm,
        "warm_all_load_zero": all(row["load_ms"] == 0 for row in rows[1:]),
        "labels_consistent": len({row["label"] for row in rows if row["ok"]}) == 1,
        "runtime": admin_call(channel, "Status"),
    }
    print(f"WARM n={warm['successful_replies']}/{args.warm_requests} "
          f"mean={warm['mean_e2e_ms']}ms range={warm['min_e2e_ms']}..{warm['max_e2e_ms']}ms "
          f"SLO_met={warm['slo_met_replies']} load_zero={report['warm_all_load_zero']}")
    _write_report(args.output, "cold-warm", rows, report)
    return int(any(not row["ok"] for row in rows))


def arrival_offsets(rate, duration, distribution, shape, seed):
    rng = random.Random(seed)
    offset = 0.0
    # Uniform means fixed, evenly spaced arrivals, not Uniform(0,a) samples.
    # Poisson/Gamma use random interarrival times with the same MEAN rate.
    while offset < duration:
        yield offset
        if distribution == "uniform":
            gap = 1.0 / rate
        elif distribution == "poisson":
            gap = rng.expovariate(rate)
        else:
            gap = rng.gammavariate(shape, 1.0 / (shape * rate))
        offset += max(gap, 1e-9)


def bench(channel, args, image):
    for index in range(args.warmup):
        row = query_once(channel, args.model, image, args.slo, args.timeout, index)
        if not row["ok"]:
            raise RuntimeError("Benchmark warmup failed: " + row["error"])
    runtime = admin_call(channel, "Status")
    rows = []
    skipped = 0
    outstanding_limit = args.concurrency * 2
    slots = threading.BoundedSemaphore(outstanding_limit)
    started = time.perf_counter()
    def execute(index, scheduled):
        send_time = time.perf_counter()
        row = query_once(channel, args.model, image, args.slo, args.timeout, index)
        row["scheduled_offset_ms"] = scheduled * 1000
        row["client_send_lag_ms"] = (send_time - started - scheduled) * 1000
        return row
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for index, offset in enumerate(arrival_offsets(
                args.rate, args.duration, args.distribution, args.shape, args.seed)):
            delay = started + offset - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            if not slots.acquire(blocking=False):
                skipped += 1
                continue
            future = pool.submit(execute, index, offset)
            future.add_done_callback(lambda _: slots.release())
            futures.append(future)
        for future in futures:
            rows.append(future.result())
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: row["index"])
    report = summarize(rows, elapsed)
    report.update({"experiment": "rate-driven test of the active serving mode",
                   "requested_mean_rate": args.rate, "offering_window_seconds": args.duration,
                   "distribution": args.distribution, "shape": args.shape, "seed": args.seed,
                   "client_concurrency": args.concurrency, "client_outstanding_limit": outstanding_limit,
                   "client_skipped_scheduled_arrivals": skipped,
                   "sent_requests_per_offering_second": len(rows) / args.duration,
                   "offered_load_warning": "Bounded client; inspect skipped arrivals and send lag. "
                                           "Drain time is included in measured reply QPS.",
                   "max_client_send_lag_ms": max((r["client_send_lag_ms"] for r in rows), default=0),
                   "runtime": runtime})
    report["runtime_after"] = admin_call(channel, "Status")
    print(json.dumps(compact_bench_report(report), indent=2, ensure_ascii=False))
    _write_report(args.output, "bench", rows, report)
    return int(any(not row["ok"] for row in rows) or skipped > 0)


def policy_test(channel, args, image):
    before = admin_call(channel, "Status")
    if not before.get("milp_enabled") or not before.get("policy_ready") or before.get("mode") != "proteus":
        raise RuntimeError("Run register first; policy mode + an optimal MILP plan are required")
    maximum = before["plan"]["max_batch"]
    if maximum < 2:
        raise RuntimeError("No batch>1 fits SLO/2 in measured profiles; inspect register output")
    # Load/reset cost is not part of this batching verification.
    warm = query_once(channel, args.model, image, args.slo, args.timeout)
    if not warm["ok"]:
        raise RuntimeError("Policy warmup failed: " + warm["error"])
    count = maximum + 1
    barrier = threading.Barrier(count)
    def execute(index):
        barrier.wait(timeout=10)
        return query_once(channel, args.model, image, args.slo, args.timeout, index)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        rows = list(pool.map(execute, range(count)))
    after = admin_call(channel, "Status")
    passed = all(r["ok"] for r in rows) and any(r["batch_size"] > 1 for r in rows)
    for row in rows:
        print_row(row)
    report = {"experiment": "real-GPU Proteus MILP and actual adaptive batch verification",
              "passed": passed, "runtime_before": before, "runtime_after": after,
              **summarize(rows, time.perf_counter() - start)}
    _write_report(args.output, "policy-test", rows, report)
    print("POLICY_TEST=" + ("PASS" if passed else "FAIL"))
    return int(not passed)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in {"query", "test", "bench", "profile", "policy-test"} and not args.image:
        raise SystemExit("--image is required")
    image = Path(args.image).read_bytes() if args.image else b""
    with grpc.insecure_channel(args.controller, options=OPTIONS) as channel:
        # The experiment clock starts after the connection is established.
        grpc.channel_ready_future(channel).result(timeout=min(args.timeout, 15))
        if args.command == "state":
            while True:
                state = admin_call(channel, "Status", timeout=10)
                if not args.quiet:
                    print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)
                if not state.get("ready"):
                    return 1
                if not args.watch:
                    return 0
                time.sleep(args.watch)
        if args.command == "reset":
            print(json.dumps(admin_call(channel, "Reset"), indent=2))
            return 0
        if args.command == "mode":
            print(json.dumps(admin_call(channel, "Mode", {"mode": args.mode}), indent=2))
            return 0
        if args.command == "evaluate":
            report = admin_call(channel, "Evaluate", {"manifest": args.manifest}, args.timeout)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "profile":
            report = admin_call(channel, "Profile", {
                "image_base64": base64.b64encode(image).decode(), "trials": args.trials,
                "models": args.models}, args.timeout)
            if report.get("profile_configuration", {}).get("strategy") == "idle-aware-v1":
                from .profiling import print_summary
                print_summary(report)
            else:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            directory = Path(args.output)
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            destination = directory / f"profile-{stamp}.json"
            destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print("saved:", destination.name)
            return 0
        if args.command == "test":
            return cold_warm(channel, args, image)
        if args.command == "bench":
            return bench(channel, args, image)
        if args.command == "policy-test":
            return policy_test(channel, args, image)
        rows = []
        started = time.perf_counter()
        for index in range(args.n):
            row = query_once(channel, args.model, image, args.slo, args.timeout, index)
            rows.append(row)
            print_row(row)
        _write_report(args.output, "query", rows, {
            **summarize(rows, time.perf_counter() - started),
            "runtime": admin_call(channel, "Status")})
        return int(any(not row["ok"] for row in rows))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
