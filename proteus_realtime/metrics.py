"""Client-E2E statistics; a successful reply does not imply SLO compliance."""
import math


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


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
    }
