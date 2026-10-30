"""Profiling conditions for the real-GPU port, not the original simulator.

Capacity uses uninterrupted batch execution. Deadline prediction may use an
envelope measured after CPU/GPU idle gaps. Sleeps are outside service timers.
"""
import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

from .metrics import percentile


@dataclass(frozen=True)
class ProfileOptions:
    strategy: str = "continuous"
    idle_gaps_ms: tuple = (0.0, 100.0, 1000.0)
    prediction_percentile: int = 50

    def __post_init__(self):
        if self.strategy not in {"continuous", "idle-aware-v1"}:
            raise ValueError("PROFILE_STRATEGY must be continuous or idle-aware-v1")
        gaps = self.idle_gaps_ms
        if (not gaps or len(gaps) > 8 or tuple(sorted(set(gaps))) != gaps or gaps[0] != 0
                or any(not math.isfinite(gap) or not 0 <= gap <= 5000 for gap in gaps)):
            raise ValueError("PROFILE_IDLE_GAPS_MS must be sorted distinct values in 0..5000, starting with 0")
        if self.strategy == "idle-aware-v1" and len(gaps) < 2:
            raise ValueError("idle-aware-v1 requires at least one nonzero idle gap")
        if self.prediction_percentile not in {50, 95}:
            raise ValueError("PROFILE_PREDICTION_PERCENTILE must be 50 or 95")

    @classmethod
    def from_env(cls, values=None):
        values = os.environ if values is None else values
        return cls(values.get("PROFILE_STRATEGY", "continuous"),
                   tuple(float(value) for value in values.get("PROFILE_IDLE_GAPS_MS", "0,100,1000").split(",")),
                   int(values.get("PROFILE_PREDICTION_PERCENTILE", "50")))

    @property
    def gaps(self):
        return (0.0,) if self.strategy == "continuous" else self.idle_gaps_ms

    @property
    def directory(self):
        return "cpu-ready-v1" if self.strategy == "continuous" else "cpu-ready-idle-v1"

    def identity(self):
        return {"strategy": self.strategy, "idle_gaps_ms": list(self.gaps),
                "prediction_percentile": 50 if self.strategy == "continuous" else self.prediction_percentile,
                "warmup_per_condition": 3, "idle_definition": "completion-to-next-start; excluded from service timer",
                "scope": "prepared-batch-service"}

    def minimum_idle_seconds(self, trials, model_count, batch_count):
        return trials * sum(self.gaps) / 1000 * model_count * batch_count


def execution_latency_ms(row):
    """Legacy rows keep their P50; aware rows carry an explicit prediction."""
    value = float(row["prediction_ms"] if "prediction_ms" in row else row["p50_ms"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Execution prediction must be finite and positive")
    return value


def validate_idle_batch(row, options):
    conditions = row.get("conditions", {})
    if {float(key) for key in conditions} != set(options.gaps):
        raise ValueError("Idle conditions mismatch")
    for condition in conditions.values():
        samples = condition.get("samples_ms", [])
        if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("Invalid idle-aware samples")
        for p in (50, 95):
            if not math.isclose(float(condition["p{}_ms".format(p)]), percentile(samples, p), rel_tol=1e-8):
                raise ValueError("Condition percentile differs from saved samples")
    for key in ("p50_ms", "p95_ms"):
        if not math.isclose(float(row[key]), float(conditions["0"][key]), rel_tol=1e-8):
            raise ValueError("Continuous profile differs from condition 0")
    expected = max(condition["p{}_ms".format(options.prediction_percentile)] for condition in conditions.values())
    if "prediction_ms" not in row or not math.isclose(execution_latency_ms(row), expected, rel_tol=1e-8):
        raise ValueError("Missing/inconsistent idle-aware execution prediction")


def measure_batch(infer, prepared, size, trials, options, *, clock=time.perf_counter,
                  sleep=time.sleep, progress=None):
    conditions = {}
    for gap in options.gaps:
        if progress:
            progress(gap)
        for _ in range(3):
            infer([prepared] * size)
        samples, idle_samples = [], []
        for _ in range(trials):
            if gap:
                idle_started = clock()
                sleep(gap / 1000)
                idle_samples.append((clock() - idle_started) * 1000)
            started = clock()
            infer([prepared] * size)
            duration = (clock() - started) * 1000
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Profiling produced a nonpositive/nonfinite execution time")
            samples.append(duration)
        conditions[str(int(gap)) if float(gap).is_integer() else str(gap)] = {
            "idle_gap_ms": gap, "p50_ms": percentile(samples, 50),
            "p95_ms": percentile(samples, 95), "samples_ms": samples,
            "actual_idle_ms": idle_samples}
    continuous = conditions["0"]
    result = {key: continuous[key] for key in ("p50_ms", "p95_ms", "samples_ms")}
    if options.strategy == "idle-aware-v1":
        statistic = "p{}_ms".format(options.prediction_percentile)
        selected = max(conditions.values(), key=lambda condition: condition[statistic])
        result.update({"conditions": conditions, "prediction_ms": selected[statistic],
                       "prediction_statistic": "max_condition_p{}".format(options.prediction_percentile),
                       "prediction_source_gap_ms": selected["idle_gap_ms"]})
    return result


def profile_summary(profiles):
    return {name: {size: {"continuous_p50_ms": row["p50_ms"],
                          "continuous_p95_ms": row.get("p95_ms"),
                          "prediction_ms": execution_latency_ms(row),
                          "prediction_source_gap_ms": row.get("prediction_source_gap_ms", 0)}
                   for size, row in profile["batches"].items()}
            for name, profile in profiles.items()}


def print_summary(report):
    configuration = report.get("profile_configuration", {"strategy": "legacy-continuous"})
    print("Profiling:", json.dumps(configuration, ensure_ascii=False))
    print("model              batch  continuous P50/P95 ms  deadline prediction ms  source idle ms")
    for name, batches in profile_summary(report["profiles"]).items():
        for size, row in sorted(batches.items(), key=lambda item: int(item[0])):
            p95 = row["continuous_p95_ms"]
            print("{:<18s} {:>5s} {:9.3f}/{:<9s} {:23.3f} {:15g}".format(
                name, size, row["continuous_p50_ms"], "n/a" if p95 is None else "{:.3f}".format(p95),
                row["prediction_ms"], row["prediction_source_gap_ms"]))
    print("Capacity: continuous P50. Deadline/SLO/2 eligibility: prediction. CPU prep/RPC/load excluded.")
    if "policy_ready" in report:
        print("policy_ready:", report["policy_ready"], "policy_error:", report.get("policy_error"))
    if "plan" in report:
        print("plan:", json.dumps(report["plan"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Read a saved profile; does not access CUDA or modify files")
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if not args.summary.is_file():
        raise SystemExit("Profile not found: {}. Run register for the selected strategy first.".format(args.summary))
    print_summary(json.loads(args.summary.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
