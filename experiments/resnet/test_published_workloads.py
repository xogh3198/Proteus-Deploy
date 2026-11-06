"""Public trace/deadline reproduction with no GPU or inference traffic."""
import json
from pathlib import Path
import time

import pytest

from experiments.resnet import client, host


@pytest.mark.parametrize("workload,directory,count", [
    ("twitter", "zipf_exponential", 3776),
    ("bursty", "zipf_exponential_bursty", 8309),
])
def test_public_dry_run_checks_full_file_without_opening_an_inference_client(
        workload, directory, count, tmp_path, monkeypatch, capsys):
    trace = tmp_path/"trace.txt"
    trace.write_bytes((host.ROOT/f"traces/twitter/asplos/{directory}/300/resnet.txt").read_bytes())
    image = tmp_path/"image.jpg"
    image.write_bytes(b"dry-run does not decode or send this input")
    real_path = Path
    def paths(value):
        return {"/inputs/image.jpg": image, "/data/trace.txt": trace}.get(str(value), real_path(value))
    monkeypatch.setattr(client, "Path", paths)
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not connect or send inference requests")
    monkeypatch.setattr(client, "Client", forbidden)
    monkeypatch.setenv("ENGINE", "proteus")
    monkeypatch.setenv("EXPERIMENT_PHASE", "v1")
    monkeypatch.setenv("PUBLISHED_WORKLOAD", workload)
    monkeypatch.setenv("WORKLOAD", workload)
    monkeypatch.setenv("SLO_MS", "300")
    monkeypatch.setenv("TRACE_SPEED", "1")
    monkeypatch.setenv("START_STATE", "cold")
    client.main(["trace", "--dry-run"])
    config = json.loads(capsys.readouterr().out)
    assert config["trace_total_requests"] == config["selected_requests"] == count
    assert config["speed"] == 1 and config["slo_override_ms"] is None
    assert config["slo_source"] == "trace column 3"
    assert config["source_slo_values"] == [300]
    assert config["configured_worker_slo_ms"] == 300
    assert config["source_trace_url"].endswith(f"{directory}/300/resnet.txt")
    original = trace.read_bytes()
    with pytest.raises(ValueError, match="speed=1"):
        client.main(["trace", "--dry-run", "--speed", "2"])
    trace.write_bytes(original+b"\n")
    with pytest.raises(ValueError, match="checksum"):
        client.main(["trace", "--dry-run"])


def test_presets_override_old_slo_and_speed_but_preserve_device_and_storage(tmp_path, monkeypatch):
    local = tmp_path/"settings.env"
    local.write_text("GPU=3\nSLO_MS=200\nTRACE_SPEED=2\nPREPROCESS_WORKERS=4\n"
                     "MODEL_DIR=custom-model-cache\nDATA_DIR=custom-results\n")
    monkeypatch.setenv("RESNET_ENV", str(local))
    original = host.settings("proteus")
    presets = [host.settings("proteus", published_workload=name) for name in ("twitter", "bursty")]
    for config in presets:
        assert config["SLO_MS"] == "300" and config["TRACE_SPEED"] == "1"
        for key in ("GPU", "MODEL_DIR", "DATA_DIR", "PREPROCESS_WORKERS", "BATCH_SIZES",
                    "CPU_THREADS", "CPU_INTEROP_THREADS", "PROTEUS_ALLOCATION_INTERVAL"):
            assert config[key] == original[key]
        assert config["PROJECT"] != original["PROJECT"]
        assert Path(config["RUN_DIR"]).is_relative_to(original["RUN_DIR"])
        assert host.environment(config)["PUBLISHED_WORKLOAD"] == config["WORKLOAD"]
    assert presets[0]["RUN_DIR"] != presets[1]["RUN_DIR"]
    assert presets[0]["PROJECT"] != presets[1]["PROJECT"]
    assert host.settings("proteus") == original


def test_legacy_profile_and_results_are_isolated_from_the_public_baseline():
    baseline = host.settings("proteus", published_workload="twitter")
    legacy = host.settings("proteus", published_workload="twitter", runtime="ort181")
    for key in ("PROJECT", "WORKER_IMAGE", "RUN_DIR"):
        assert legacy[key] == baseline[key] + "-ort181"
    for key in baseline.keys() - {"PROJECT", "WORKER_IMAGE", "RUN_DIR"}:
        assert legacy[key] == baseline[key]
    baseline_paths = {row["source"] for row in host.mounts(baseline) if not row.get("read_only")}
    legacy_paths = {row["source"] for row in host.mounts(legacy) if not row.get("read_only")}
    assert baseline_paths.isdisjoint(legacy_paths)
    assert host.environment(legacy)["ALLOCATION_INTERVAL"] == host.environment(baseline)["ALLOCATION_INTERVAL"]


def test_replay_sends_trace_deadline_instead_of_the_old_experiment_deadline():
    received = []
    class Recorder:
        def query(self, raw, slo, request_id):
            received.append(slo)
            return {"sent": True, "ok": True, "slo_met": True, "e2e_ms": 0,
                    "rpc_started": time.perf_counter(), "timings": {}}
    arrivals = [(0, 1, 300), (0, 1, 600)]
    rows, _ = client.replay(Recorder(), b"prepared", arrivals, 200, 1, 2, use_trace_slo=True)
    assert sorted(received) == [300, 600]
    assert [row["request_slo_ms"] for row in rows] == [300, 600]
    assert arrivals == [(0, 1, 300), (0, 1, 600)]
