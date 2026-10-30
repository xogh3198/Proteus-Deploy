"""Separate client-E2E and worker-internal metrics; never relabel old results."""
import math


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _stats(values):
    values = [n for value in values if (n := _number(value)) is not None]
    return {"samples": len(values),
            "mean_ms": sum(values) / len(values) if values else None,
            "p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95),
            "p99_ms": percentile(values, 99)}


def split_latency_metrics(rows):
    """Failures count as unmet; missing server timestamps are NOT successes.

    This is a per-request comparison, not a subtraction of aggregate P95s.
    Counts cover the provided requests only; unsent trace arrivals stay separate.
    """
    rows = [row for row in rows if row.get("sent") is not False]
    success = [row for row in rows if row["ok"]]
    total = len(rows)

    def scope(key, definition, legacy_client=False):
        met = unknown = 0
        times = []
        for row in success:
            value, slo = _number(row.get(key)), _number(row.get("slo_ms"))
            if value is not None and value >= 0:
                times.append(value)
            if value is not None and value >= 0 and slo is not None and slo > 0:
                met += int(value <= slo)
            elif legacy_client and isinstance(row.get("slo_met"), bool):
                # Old client rows already define this flag as E2E, never server.
                met += int(row["slo_met"])
            else:
                unknown += 1
        unmet = total - met if not unknown else None
        return {"definition": definition, "slo_met_requests": met,
                "slo_unmet_requests": unmet, "unknown_successful_slo_requests": unknown,
                "slo_unmet_ratio_of_all_requests": unmet / total if total and unmet is not None else None,
                "latency_population": "successful replies with valid recorded latency",
                "latency": _stats(times)}

    extra = []
    prediction_error = []
    for row in success:
        e2e, server = _number(row.get("e2e_ms")), _number(row.get("server_ms"))
        if e2e is not None and server is not None and e2e >= 0 and server >= 0:
            extra.append(e2e - server)
        timings = row.get("timings", {})
        if isinstance(timings, dict):
            prediction_error.append(timings.get("worker.gpu_prediction_error"))
    return {
        "client_e2e": scope("e2e_ms", "client RPC call start -> response received; excludes scheduled send lag", True),
        "server_internal": scope("server_ms", "worker request acceptance -> GPU inference completion; NOT full frontend latency"),
        "rpc_and_unmeasured_extra": {"definition": "paired E2E minus worker latency; NOT network-only",
                                     **_stats(extra)},
        "gpu_prediction_error": {"definition": "actual prepared-batch GPU service minus configured prediction",
                                 **_stats(prediction_error)},
    }


def compact_bench_report(report):
    """Full runtime/controller history stays in JSON, not the terminal summary."""
    keys = ("requests", "successful_replies", "failed_or_rejected", "requested_mean_rate",
            "offering_window_seconds", "sent_requests_per_offering_second",
            "client_skipped_scheduled_arrivals", "max_client_send_lag_ms",
            "legacy_slo_fields_scope", "slo_metrics")
    result = {key: report[key] for key in keys if key in report}
    runtime = report.get("runtime_after", report.get("runtime", {}))
    result["deadline_budget"] = runtime.get("deadline_budget")
    return result


def summarize(rows, elapsed_seconds):
    success = [row for row in rows if row["ok"]]
    latencies = [row["e2e_ms"] for row in success]
    good = [row for row in success if row["slo_met"]]
    misses = [row for row in success if not row["slo_met"]]
    seconds = max(elapsed_seconds, 1e-12)
    return {
        "requests": len(rows), "successful_replies": len(success),
        "failed_or_rejected": len(rows) - len(success), "slo_missed_replies": len(misses),
        "slo_met_replies": len(good),
        "slo_miss_ratio_among_replies": len(misses) / len(success) if success else None,
        "slo_met_ratio_of_all_requests": len(good) / len(rows) if rows else None,
        "successful_reply_qps": len(success) / seconds,
        "slo_goodput_qps": len(good) / seconds,
        "elapsed_seconds_including_drain": elapsed_seconds,
        "mean_e2e_ms": sum(latencies) / len(latencies) if latencies else None,
        "min_e2e_ms": min(latencies) if latencies else None,
        "max_e2e_ms": max(latencies) if latencies else None,
        "p50_e2e_ms": percentile(latencies, 50), "p95_e2e_ms": percentile(latencies, 95),
        "p99_e2e_ms": percentile(latencies, 99),
        "accuracy": None, "accuracy_note": "Repeated-image tests do not measure prediction accuracy",
        "legacy_slo_fields_scope": "client_e2e (unchanged)",
        "slo_metrics": split_latency_metrics(rows),
    }


def read_csv_rows(path):
    import csv
    import json

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"ok", "e2e_ms"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires ok and e2e_ms columns")
        rows = list(reader)
    for line, row in enumerate(rows, 2):
        ok = row["ok"].strip().lower()
        if ok not in {"true", "false"}:
            raise ValueError(f"Invalid ok flag at CSV line {line}")
        row["ok"] = ok == "true"
        if "sent" in row:
            sent = row["sent"].strip().lower()
            if sent not in {"true", "false"}:
                raise ValueError(f"Invalid sent flag at CSV line {line}")
            row["sent"] = sent == "true"
        if "slo_met" in row:
            flag = row["slo_met"].strip().lower()
            row["slo_met"] = {"true": True, "false": False}.get(flag)
        row["timings"] = json.loads(row.get("timings") or "{}")
    return rows


def main(argv=None):
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Read-only split SLO summary of a saved CSV; no Docker/GPU")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", "--file", type=Path, dest="csv")
    source.add_argument("--results-dir", type=Path)
    args = parser.parse_args(argv)
    path = args.csv
    if path is None:
        paths = sorted(args.results_dir.glob("bench-*.csv"))
        if not paths:
            raise SystemExit("No bench-*.csv in " + str(args.results_dir))
        path = paths[-1]
    all_rows = read_csv_rows(path)
    rows = [row for row in all_rows if row.get("sent") is not False]
    report = {"source_csv": str(path.resolve()), "requests": len(rows),
              "csv_rows": len(all_rows), "client_not_sent": len(all_rows) - len(rows),
              "successful_replies": sum(row["ok"] for row in rows),
              "failed_or_rejected": sum(not row["ok"] for row in rows),
              "slo_metrics": split_latency_metrics(rows),
              "note": "Read-only; original CSV/JSON unchanged. Trace unsent arrivals are NOT counted here."}
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
