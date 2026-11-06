"""Measure GPU batch switches and repeats before registering a serving profile."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from proteus_realtime.metrics import percentile

from .runtime import Runtime


def measure(runtime, model, batches, trials, prepared):
    groups = defaultdict(list)
    previous = None
    label = None

    def run(batch, case=None):
        nonlocal previous, label
        inputs = [prepared] * batch
        started = time.perf_counter()
        output = runtime.infer_prepared(inputs)
        elapsed = (time.perf_counter() - started) * 1000
        if len(output) != batch:
            raise RuntimeError("Unexpected batch output size")
        if label is None:
            label = output[0]["label_id"]
        if any(row["label_id"] != label for row in output):
            raise RuntimeError("ImageNet prediction changed for the repeated fixture")
        if case is not None:
            groups[case, previous, batch].append(elapsed)
        previous = batch

    load = runtime.load(model)
    for batch in batches:
        for _ in range(3):
            run(batch)
        for _ in range(trials):
            run(batch, "fixed")
    for _ in range(trials):
        for batch in batches:
            run(batch, "switch")
            run(batch, "repeat")
    return {
        "model": model, "batches": list(batches), "trials": trials, "label_id": label,
        "actual_load": load, "fingerprint": runtime.fingerprint(),
        "measurement": "infer_prepared wall time including binding, GPU execution and output copy",
        "summary": [{"case": case, "previous_batch": before, "batch": batch,
                     "p50_ms": percentile(samples, 50), "p95_ms": percentile(samples, 95),
                     "samples_ms": samples} for (case, before, batch), samples in groups.items()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet152")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.trials <= 1000:
        parser.error("--trials must be between 2 and 1000")
    batches = tuple(map(int, os.environ["BATCH_SIZES"].split(",")))
    if len(batches) < 2:
        parser.error("At least two distinct batch sizes are needed")
    model = args.model.removesuffix("_v1") + "_v1"
    runtime = Runtime()
    prepared = runtime.prepare(Path("/inputs/image.jpg").read_bytes())
    try:
        result = measure(runtime, model, batches, args.trials, prepared)
    finally:
        runtime.reset()
    name = datetime.now(timezone.utc).strftime("batch-probe-%Y%m%dT%H%M%S.%fZ.json")
    path = Path("/results")/name
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    print("Saved:", path)


if __name__ == "__main__":
    main()
