"""Reuse identical gRPC/JPEG metrics; label native tests and fixed diagnostics."""
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import grpc

from . import BASELINES
from proteus_realtime import client as common
from proteus_realtime.metrics import summarize


def trace_identity(args):
    digest, count = hashlib.sha256(), 0
    for offset in common.arrival_offsets(args.rate, args.duration, args.distribution, args.shape, args.seed):
        digest.update(struct.pack("!d", offset))
        count += 1
    return {"scheduled_trace_sha256": digest.hexdigest(), "scheduled_requests": count}


def cold_warm(channel, args, image):
    before = common.admin_call(channel, "Status")
    if not before["policy_ready"]:
        raise ValueError("Run register before a diagnostic that restores native policy")
    common.admin_call(channel, "Mode", {"mode": "fixed"})
    try:
        common.admin_call(channel, "Reset")
        first = common.query_once(channel, args.model, image, args.slo, args.timeout)
        common.print_row(first)
        if not first["ok"]:
            raise RuntimeError("Cold actual GPU request failed; inspect this baseline's logs")
        started = time.perf_counter()
        rows = [common.query_once(channel, args.model, image, args.slo, args.timeout, i + 1)
                for i in range(args.warm_requests)]
        elapsed = time.perf_counter() - started
        for row in rows:
            common.print_row(row)
        report = {"experiment": "fixed-batch-1 cold/warm diagnostic, NOT native baseline policy comparison",
            "baseline": args.baseline, "image_sha256": hashlib.sha256(image).hexdigest(), "first": first,
            "warm": summarize(rows, elapsed), "warm_all_load_zero": all(row["load_ms"] == 0 for row in rows),
            "cold_definition": "GPU model non-resident; OS/CUDA/library caches not cleared", "runtime": before}
        common._write_report(args.output, "baseline-cold-warm", [first] + rows, report)
        print(json.dumps(report, indent=2))
        return int(any(not row["ok"] for row in rows))
    finally:
        common.admin_call(channel, "Mode", {"mode": "native"})


def main(argv=None):
    parser = common.parser()
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    args = parser.parse_args(argv)
    if args.command not in {"state", "profile", "query", "test", "bench", "reset"}:
        parser.error("Use state/profile/query/test/bench/reset; no Proteus mode or MILP policy-test in this adapter")
    image = Path(args.image).read_bytes() if args.image else b""
    with grpc.insecure_channel(args.controller, options=common.OPTIONS) as channel:
        grpc.channel_ready_future(channel).result(timeout=min(args.timeout, 15))
        state = common.admin_call(channel, "Status", timeout=10)
        if state.get("baseline") != args.baseline:
            raise ValueError("Endpoint baseline differs from requested baseline; stop before sending data")
        if args.command == "test":
            if not image:
                parser.error("Image required")
            return cold_warm(channel, args, image)
        if args.command == "bench":
            if not image or state.get("mode") != "native" or not state["policy_ready"]:
                raise ValueError("Actual profile + native policy + image required")
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            before = set(output.glob("bench-*.json"))
            warmup_count = args.warmup
            # Sparse sequential warmup must not silently exercise/fill the AIMD
            # queue. Warm the same GPU model through an explicit fixed1 diagnostic,
            # then reset the original native policy before the offered workload.
            common.admin_call(channel, "Mode", {"mode": "fixed"})
            try:
                for index in range(warmup_count):
                    row = common.query_once(channel, args.model, image, args.slo, args.timeout, index)
                    if not row["ok"]:
                        raise RuntimeError("Fixed1 benchmark warmup failed")
            finally:
                common.admin_call(channel, "Mode", {"mode": "native"})
            args.warmup = 0
            status = common.bench(channel, args, image)
            generated = set(output.glob("bench-*.json")) - before
            if len(generated) != 1:
                raise RuntimeError("Expected exactly one new native workload report")
            path = next(iter(generated))
            report = json.loads(path.read_text())
            report.update({"experiment": "real-GPU workload of Proteus original singleton comparison baseline",
                "baseline": args.baseline, "image_sha256": hashlib.sha256(image).hexdigest(), **trace_identity(args)})
            report["fixed1_prebenchmark_warmup_requests"] = warmup_count
            path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print("Native comparison metadata:", json.dumps({key: report[key] for key in (
                "baseline", "image_sha256", "scheduled_trace_sha256", "scheduled_requests")}))
            return status
    # Existing non-experiment client commands are reused without source edits.
    remainder = list(argv) if argv is not None else __import__("sys").argv[1:]
    index = remainder.index("--baseline")
    del remainder[index:index + 2]
    return common.main(remainder)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error))
