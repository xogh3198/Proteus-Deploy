#!/usr/bin/env python3
"""CPU-only Twitter replay for the original-policy baseline GPU adapters.

Arrival scheduling, concurrency handling, and summary definitions are copied
unchanged from the previous Proteus trace client. No resets or warmups occur.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Arrival:
    line: int
    arrival_ms: float
    qos: str
    original_slo_ms: float


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def count(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def read_trace(path, limit=100):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    raw = Path(path).read_bytes()
    arrivals = []
    previous = -1.0
    for line, fields in enumerate(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), 1):
        if not fields or not any(part.strip() for part in fields):
            continue
        if fields[0].lstrip().startswith("#"):
            continue
        if len(fields) != 3:
            raise ValueError(f"Trace line {line}: expected arrival_ms,qos,slo_ms (no header)")
        arrival, original_slo = float(fields[0]), float(fields[2])
        if not math.isfinite(arrival) or arrival < 0 or arrival < previous:
            raise ValueError(f"Trace line {line}: nonnegative, nondecreasing finite times required")
        if not math.isfinite(original_slo) or original_slo <= 0:
            raise ValueError(f"Trace line {line}: invalid original SLO")
        previous = arrival
        arrivals.append(Arrival(line, arrival, fields[1].strip(), original_slo))
    if not arrivals:
        raise ValueError("Trace contains no requests")
    selected = arrivals[:limit] if limit else arrivals
    return selected, len(arrivals), hashlib.sha256(raw).hexdigest()


def read_accuracy(path):
    raw = Path(path).read_bytes()
    registry = json.loads(raw.decode("utf-8-sig"))
    entries = {name: entry for name, entry in registry["models"].items()
               if entry.get("enabled", True)}
    if not entries:
        raise ValueError("Enable at least one accuracy-registry model")
    datasets = set()
    for name, entry in entries.items():
        accuracy = entry.get("accuracy")
        if (isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
                or not math.isfinite(accuracy) or not 0 < accuracy <= 100):
            raise ValueError(f"Offline/reference accuracy is required for {name}")
        if not entry.get("accuracy_source") or not entry.get("accuracy_dataset"):
            raise ValueError(f"Accuracy source and dataset are required for {name}")
        datasets.add(entry["accuracy_dataset"])
    if len(datasets) != 1:
        raise ValueError("All model accuracies must have the same dataset and metric")
    return entries, hashlib.sha256(raw).hexdigest()


def check_worker(state, entries, baseline):
    if state.get("baseline") != baseline or state.get("implementation") != "proteus-original-baseline-single-gpu-adapter":
        raise ValueError("Endpoint is not the requested independent baseline")
    if not state.get("policy_ready") or state.get("mode") != "native":
        raise ValueError("Register all candidate profiles and use native policy mode before replay")
    if set(state.get("enabled_models", [])) != set(entries):
        raise ValueError("Worker candidates differ from supplied reference registry")
    checkpoints = state.get("checkpoints", {})
    for name, entry in entries.items():
        identity = checkpoints.get(name, {})
        if identity.get("model_id") != entry.get("model_id") or identity.get("revision") != entry.get("revision"):
            raise ValueError("Worker checkpoint identity differs for " + name)

def empty_reply(error, kind):
    return {"ok": False, "e2e_ms": None, "server_ms": None, "slo_met": False,
            "variant": "", "batch_size": 0, "load_ms": None, "queue_ms": None,
            "service_ms": None, "path": "", "label": "", "gpu": "",
            "request_name": "", "reject_kind": kind, "error": error, "timings": {}}


def replay(arrivals, send, *, speed=1.0, slo_ms=200.0, concurrency=64,
           progress=True, clock=time.perf_counter, sleep=time.sleep):
    """Open-loop schedule, bounded active calls; never await one reply to send the next.

    Client saturation produces an explicit unsent row, not a hidden skipped request.
    Server SLO starts on actual sending; scheduled-deadline metrics also include lag.
    """
    if (not math.isfinite(speed) or speed <= 0 or not math.isfinite(slo_ms)
            or slo_ms <= 0 or concurrency < 1):
        raise ValueError("Positive speed, SLO and concurrency required")
    rows, futures = [], []
    slots = threading.BoundedSemaphore(concurrency)
    start = clock()
    last_progress = start

    def execute(index, arrival, due, offset):
        sent = clock()
        try:
            reply = dict(send(index))
        except Exception as error:
            reply = empty_reply(str(error), "client_exception")
        finished = clock()
        elapsed = (finished - sent) * 1000
        return {**reply, "index": index, "trace_line": arrival.line,
                "trace_arrival_ms": arrival.arrival_ms, "trace_qos": arrival.qos,
                "trace_original_slo_ms": arrival.original_slo_ms,
                "scheduled_offset_ms": offset * 1000, "slo_ms": slo_ms,
                "sent": True, "actual_send_offset_ms": (sent - start) * 1000,
                "client_send_lag_ms": max(0, (sent - due) * 1000),
                "completion_offset_ms": (finished - start) * 1000,
                "e2e_ms": elapsed,
                "scheduled_e2e_ms": (finished - due) * 1000,
                "slo_met": bool(reply["ok"] and elapsed <= slo_ms),
                "scheduled_slo_met": bool(reply["ok"] and (finished - due) * 1000 <= slo_ms)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for index, arrival in enumerate(arrivals):
            offset = arrival.arrival_ms / (1000 * speed)
            due = start + offset
            delay = due - clock()
            if delay > 0:
                sleep(delay)
            if not slots.acquire(blocking=False):
                rows.append({**empty_reply("Client concurrency limit reached; request NOT sent",
                                          "client_not_sent"),
                    "index": index, "trace_line": arrival.line,
                    "trace_arrival_ms": arrival.arrival_ms, "trace_qos": arrival.qos,
                    "trace_original_slo_ms": arrival.original_slo_ms,
                    "scheduled_offset_ms": offset * 1000, "slo_ms": slo_ms,
                    "sent": False, "actual_send_offset_ms": None,
                    "client_send_lag_ms": None, "completion_offset_ms": None,
                    "scheduled_e2e_ms": None, "scheduled_slo_met": False})
            else:
                future = pool.submit(execute, index, arrival, due, offset)
                future.add_done_callback(lambda completed: slots.release())
                futures.append(future)
            now = clock()
            if progress and now - last_progress >= 10:
                print(f"scheduled={index + 1}/{len(arrivals)} submitted={len(futures)} "
                      f"client_not_sent={len(rows)} elapsed={now - start:.1f}s", flush=True)
                last_progress = now
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["index"])
    return rows, clock() - start


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(rows, entries, elapsed, max_lag_ms=20.0):
    if not rows:
        raise ValueError("No scheduled requests")
    sent = [row for row in rows if row["sent"]]
    success = [row for row in sent if row["ok"]]
    good = [row for row in sent if row["slo_met"]]
    scheduled_good = [row for row in rows if row["scheduled_slo_met"]]
    latencies = [row["e2e_ms"] for row in success]
    counts = Counter(row.get("variant", "") for row in success)
    unknown = sorted(name for name in counts if name not in entries)

    def reference_mean(selected):
        if not selected or any(row.get("variant") not in entries for row in selected):
            return None
        return sum(entries[row["variant"]]["accuracy"] for row in selected) / len(selected)

    best = max(entry["accuracy"] for entry in entries.values())
    mean = reference_mean(success)
    lag = [row["client_send_lag_ms"] for row in sent]
    late = sum(value > max_lag_ms for value in lag)
    not_sent = len(rows) - len(sent)
    offering_seconds = rows[-1]["scheduled_offset_ms"] / 1000
    return {"scheduled_requests": len(rows), "sent_requests": len(sent),
            "client_not_sent": not_sent, "successful_replies": len(success),
            "failed_or_rejected_sent_requests": len(sent) - len(success),
            "slo_met_sent_requests": len(good),
            "slo_unmet_ratio_of_sent_requests": (1 - len(good) / len(sent)) if sent else None,
            "scheduled_deadline_unmet_ratio": 1 - len(scheduled_good) / len(rows),
            "scheduled_deadline_note": "Includes unsent requests and client send lag; "
                                       "not solely a server SLO metric",
            "successful_requests_by_model": dict(counts),
            "slo_met_requests_by_model": dict(Counter(row["variant"] for row in good)),
            "mean_reference_accuracy_of_successful_replies": mean,
            "mean_reference_accuracy_of_slo_met_replies": reference_mean(good),
            "normalized_reference_accuracy_of_successful_replies":
                100 * mean / best if mean is not None else None,
            "unknown_variants_for_accuracy": unknown,
            "accuracy_note": "Request-weighted supplied model scores, NOT measured image "
                             "correctness or verification of current checkpoint accuracy",
            "mean_e2e_ms": sum(latencies) / len(latencies) if latencies else None,
            "p50_e2e_ms": percentile(latencies, 50),
            "p95_e2e_ms": percentile(latencies, 95),
            "p99_e2e_ms": percentile(latencies, 99),
            "latency_population": "Successful replies only, including successful late replies",
            "max_client_send_lag_ms": max(lag, default=None),
            "p95_client_send_lag_ms": percentile(lag, 95),
            "send_lag_threshold_ms": max_lag_ms, "requests_above_send_lag_threshold": late,
            "trace_timing_valid": not_sent == 0 and late == 0,
            "offering_window_seconds": offering_seconds,
            "actual_sent_per_offering_second": len(sent) / offering_seconds if offering_seconds else None,
            "elapsed_seconds_including_drain": elapsed,
            "successful_reply_qps_including_drain": len(success) / max(elapsed, 1e-12)}


def write_results(output, rows, report):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "trace-replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    csv_path, json_path = directory / (stem + ".csv"), directory / (stem + ".json")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "timings": json.dumps(row.get("timings", {}))})
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return csv_path, json_path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trace", required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--accuracy-registry", required=True)
    result.add_argument("--controller", default="baseline-worker:9000")
    result.add_argument("--baseline", choices=("infaas-accuracy", "clipper-ha", "clipper-ht"), required=True)
    result.add_argument("--application", default="imagenet")
    result.add_argument("--slo", type=positive, default=200.0,
                        help="Override every request's trace SLO, in milliseconds")
    result.add_argument("--speed", type=positive, default=1.0,
                        help="1=original timing; 2=double arrival rate and half duration")
    result.add_argument("--limit", type=count, default=100, help="Default 100; 0=all trace rows")
    result.add_argument("--concurrency", type=int, default=64)
    result.add_argument("--timeout", type=positive, default=10.0, help="RPC timeout, not the SLO")
    result.add_argument("--max-send-lag-ms", type=positive, default=20.0)
    result.add_argument("--output", default="/results")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.concurrency <= 512:
        raise ValueError("concurrency must be in 1..512")
    arrivals, total, trace_hash = read_trace(args.trace, args.limit)
    entries, accuracy_hash = read_accuracy(args.accuracy_registry)
    image = Path(args.image).read_bytes()
    if not image or len(image) > 16 * 1024 * 1024:
        raise ValueError("Image must be nonempty and at most 16 MiB")
    config = {"trace_path": str(args.trace), "trace_sha256": trace_hash,
              "trace_total_requests": total, "selected_requests": len(arrivals),
              "limit": args.limit, "speed": args.speed,
              "planned_duration_seconds": arrivals[-1].arrival_ms / (1000 * args.speed),
              "arrival_time_origin": "Original trace zero; first-arrival offset is preserved",
              "slo_override_ms": args.slo, "qos_note": "Trace QoS recorded, ignored in this single-SLO run",
              "baseline": args.baseline, "application": args.application, "controller": args.controller,
              "image_path": str(args.image), "image_sha256": hashlib.sha256(image).hexdigest(),
              "accuracy_registry_sha256": accuracy_hash, "accuracy_metadata": entries,
              "concurrency": args.concurrency, "rpc_timeout_seconds": args.timeout,
              "scope": "Repeated-JPEG open-loop trace replay; no model reset or warmup requests"}
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("DRY RUN: no connection or inference requests were made.")
        return 0

    import grpc
    from proteus_realtime.client import OPTIONS, admin_call, query_once

    with grpc.insecure_channel(args.controller, options=OPTIONS) as channel:
        grpc.channel_ready_future(channel).result(timeout=args.timeout)
        before = admin_call(channel, "Status", timeout=args.timeout)
        check_worker(before, entries, args.baseline)
        if before.get("application") != args.application:
            raise ValueError("Application must match the worker; do not pin an individual model")

        def send(index):
            return query_once(channel, args.application, image, args.slo, args.timeout, index)

        rows, elapsed = replay(arrivals, send, speed=args.speed, slo_ms=args.slo,
                               concurrency=args.concurrency)
        try:
            after = admin_call(channel, "Status", timeout=args.timeout)
        except Exception as error:
            after = {"status_read_error": str(error)}
    summary = summarize(rows, entries, elapsed, args.max_send_lag_ms)
    from proteus_realtime.metrics import split_latency_metrics
    summary["slo_metrics"] = split_latency_metrics([row for row in rows if row["sent"]])
    report = {"schema_version": 1, "config": config, "summary": summary,
              "worker_before": before, "worker_after": after}
    csv_path, json_path = write_results(args.output, rows, report)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"CSV: {csv_path}\nJSON: {json_path}", flush=True)
    # Server failures/SLO misses are experiment outcomes, not a client setup failure.
    return 0 if summary["trace_timing_valid"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"ERROR: {error}") from error
