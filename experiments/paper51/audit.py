"""Read-only paper inventory and checksum-verified public trace acquisition.

No model weights, latency samples, missing accuracy scores, or inference results
are manufactured here. This module does not modify the serving implementation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request


CATALOG = Path(__file__).with_name("catalog.json")
COMMIT = "6967c9327e66f6d6fe94993dcaa7521efc2f72f5"
GROUP_COUNTS = {"bert": 12, "densenet": 4, "efficientnet": 8, "gpt2": 4,
                "mobilenet": 4, "resnest": 4, "resnet": 5, "t5": 5, "yolo": 5}
MISSING = {"efficientnet-b2", "efficientnet-b4", "efficientnet-b6", "t5-3b", "t5-11b"}
WORKLOADS = {"twitter": "zipf_exponential", "bursty": "zipf_exponential_bursty",
             "uniform": "zipf_uniform", "gamma": "zipf_gamma"}


def read_catalog(path=CATALOG):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["upstream_commit"] != COMMIT:
        raise ValueError("Unexpected upstream source revision")
    if (data["experiment_precision"] != "fp16" or data["physical_gpus"] != 1
            or data["total_groups"] != 9 or data["total_variants"] != 51):
        raise ValueError("Expected the selected single-GPU FP16 9-group/51-model experiment")
    models = data["models"]
    if len(models) != 51 or len({m["name"] for m in models}) != 51:
        raise ValueError("Expected 51 distinct model variants")
    for group, count in GROUP_COUNTS.items():
        if sum(m["group"] == group for m in models) != count:
            raise ValueError(f"Incorrect model count for {group}")
    if {m["name"] for m in models if not m["public_artifact"]} != MISSING:
        raise ValueError("Paper/public-artifact difference changed")
    for model in models:
        score = model["reference_accuracy"]
        if model["name"] in MISSING:
            if score is not None:
                raise ValueError("Do not present added model scores as public artifact values")
        elif (isinstance(score, bool) or not isinstance(score, (int, float))
              or not math.isfinite(score) or not 0 < score <= 100):
            raise ValueError("Invalid public reference accuracy")
    groups_by_model = {m["name"]: m["group"] for m in models}
    for entry in data["clipper_fixed"].values():
        fixed = entry["models"]
        if len(fixed) != 9 or {groups_by_model[m] for m in fixed} != set(GROUP_COUNTS):
            raise ValueError("Clipper requires one published fixed model for each group")
    return data


def policy_scores(catalog, supplement=None):
    """Merge explicitly sourced additions, preserving every public score."""
    if supplement is None:
        supplement = json.loads(CATALOG.with_name("supplemental_scores.json").read_text(encoding="utf-8"))
    additions = supplement["entries"]
    if set(additions) != MISSING:
        raise ValueError("Supplement must cover exactly the five unpublished model scores")
    scores = {m["name"]: m["reference_accuracy"] for m in catalog["models"]}
    by_name = {m["name"]: m for m in catalog["models"]}
    for name, row in additions.items():
        maximum = 84.3 if row["group"] == "efficientnet" else 32.1
        raw = row["raw_score"]
        if (row["group"] != by_name[name]["group"] or row["reference_maximum"] != maximum
                or isinstance(raw, bool) or not isinstance(raw, (int, float))
                or not math.isfinite(raw) or not 0 < raw <= maximum
                or not row.get("source", "").startswith("https://") or not row.get("metric")):
            raise ValueError(f"Invalid supplemental score provenance: {name}")
        scores[name] = 100 * raw / maximum
    return scores


def inspect_trace(raw):
    """Source zero is preserved; timestamps are never renumbered or jittered."""
    count, previous, slos, qos = 0, -1, set(), set()
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 3:
            raise ValueError(f"Trace line {number}: expected arrival_ms,qos,slo_ms")
        arrival, quality, slo = (int(p.strip()) for p in parts)
        if arrival < previous or arrival < 0 or quality < 0 or slo <= 0:
            raise ValueError(f"Trace line {number}: invalid or unordered values")
        previous = arrival
        count += 1
        slos.add(slo)
        qos.add(quality)
    if not count:
        raise ValueError("Empty trace")
    return {"requests": count, "last_arrival_ms": previous,
            "slo_values": sorted(slos), "qos_values": sorted(qos),
            "sha256": hashlib.sha256(raw).hexdigest()}


def verify_bytes(raw, expected):
    actual = inspect_trace(raw)
    for key in ("requests", "last_arrival_ms", "slo_values", "sha256"):
        if actual[key] != expected[key]:
            raise ValueError(f"Trace {key} differs from pinned public source")
    return actual


def trace_url(workload, group):
    return (f"https://raw.githubusercontent.com/UMass-LIDS/Proteus/{COMMIT}/"
            f"traces/twitter/asplos/{WORKLOADS[workload]}/300/{group}.txt")


def acquire(catalog, workload, destination, fetch=False, opener=urllib.request.urlopen):
    destination = Path(destination)
    rows = {}
    for group, expected in catalog["trace_sets"][WORKLOADS[workload]].items():
        path = destination / WORKLOADS[workload] / f"{group}.txt"
        if path.exists():
            raw = path.read_bytes()
            rows[group] = verify_bytes(raw, expected)
            continue
        if not fetch:
            raise FileNotFoundError(f"Missing pinned trace: {path}; run fetch-traces")
        request = urllib.request.Request(trace_url(workload, group),
                                         headers={"User-Agent": "proteus-paper51-audit/1"})
        with opener(request, timeout=60) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("Unexpectedly large trace")
        rows[group] = verify_bytes(raw, expected)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: never overwrite an existing file, even in a race.
        with path.open("xb") as output:
            output.write(raw)
    return rows


def workload_summary(rows, speed):
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    result = {}
    for group, row in rows.items():
        seconds = row["last_arrival_ms"] / 1000 / speed
        result[group] = {"requests": row["requests"], "duration_seconds": seconds,
                         "planned_mean_qps": row["requests"] / seconds}
    seconds = max(row["last_arrival_ms"] for row in rows.values()) / 1000 / speed
    total = sum(row["requests"] for row in rows.values())
    return {"speed": speed, "per_group": result,
            "mixed_trace": {"requests": total, "duration_seconds": seconds,
                            "planned_mean_qps": total / seconds},
            "note": "Mixed total describes source arrivals; it does not select a single-GPU scheduling policy."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("catalog")
    for action in ("fetch-traces", "verify-traces", "workload"):
        p = sub.add_parser(action)
        p.add_argument("--workload", choices=WORKLOADS, default="twitter")
        p.add_argument("--dest", type=Path, default=Path("data/paper51/traces"))
        p.add_argument("--speed", type=float, default=1)
    args = parser.parse_args(argv)
    catalog = read_catalog()
    if args.action == "catalog":
        output = {"groups": GROUP_COUNTS, "total_variants": 51,
                  "precision": "fp16", "missing_public_scores": sorted(MISSING),
                  "supplemental_policy_scores": {name: policy_scores(catalog)[name] for name in sorted(MISSING)},
                  "clipper_fixed": catalog["clipper_fixed"],
                  "weights": "not published in the original simulator artifact"}
    else:
        if args.action == "workload":
            rows = catalog["trace_sets"][WORKLOADS[args.workload]]
        else:
            rows = acquire(catalog, args.workload, args.dest, fetch=args.action == "fetch-traces")
        output = workload_summary(rows, args.speed)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
