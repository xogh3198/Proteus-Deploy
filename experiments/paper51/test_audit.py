"""Tests of source identity, honest missing metadata, and immutable trace inputs."""
import importlib.util
import io
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("paper_audit", Path(__file__).with_name("audit.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_full_inventory_and_nonmaximal_actual_clipper_ha():
    data = audit.read_catalog()
    assert len(data["models"]) == 51
    fixed = set(data["clipper_fixed"]["ha"]["models"])
    assert {"densenet201", "gpt2-medium", "t5-base"} <= fixed
    assert not {"densenet161", "gpt2-xl", "t5-11b"} & fixed
    assert sum(m["reference_accuracy"] is None for m in data["models"]) == 5


def test_reject_unsupported_metadata_claims(tmp_path):
    data = audit.read_catalog()
    added = next(m for m in data["models"] if not m["public_artifact"])
    added["reference_accuracy"] = 100
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="public artifact"):
        audit.read_catalog(path)


def test_additional_scores_use_primary_sources_and_keep_nonmonotone_t5_ranking():
    catalog = audit.read_catalog()
    scores = audit.policy_scores(catalog)
    assert len(scores) == 51
    assert scores["t5-3b"] == pytest.approx(100 * 31.8 / 32.1)
    assert scores["t5-3b"] < scores["t5-large"] < scores["t5-11b"]
    assert scores["efficientnet-b2"] == pytest.approx(100 * 80.1 / 84.3)
    for model in catalog["models"]:
        if model["public_artifact"]:
            assert scores[model["name"]] == model["reference_accuracy"]
    bad = json.loads(audit.CATALOG.with_name("supplemental_scores.json").read_text())
    bad["entries"]["t5-small"] = dict(bad["entries"]["t5-3b"])
    with pytest.raises(ValueError, match="exactly"):
        audit.policy_scores(catalog, bad)


def test_original_zero_and_speed_without_arrival_rewrite():
    rows = {"resnet": {"requests": 3776, "last_arrival_ms": 290308}}
    summary = audit.workload_summary(rows, 2)
    assert summary["per_group"]["resnet"]["duration_seconds"] == 145.154
    assert summary["mixed_trace"]["requests"] == 3776
    assert rows["resnet"]["last_arrival_ms"] == 290308
    for speed in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            audit.workload_summary(rows, speed)


def test_all_nine_groups_are_not_the_historical_resnet_subset():
    rows = audit.read_catalog()["trace_sets"]["zipf_exponential"]
    summary = audit.workload_summary(rows, 2)
    assert rows["resnet"]["requests"] == 3776
    assert summary["mixed_trace"]["requests"] == 75253
    assert summary["mixed_trace"]["duration_seconds"] == 145.3105


def test_acquisition_verifies_before_creation_and_never_replaces(tmp_path):
    raw = b"100,1,300\n110,1,300\n"
    expected = audit.inspect_trace(raw)
    catalog = {"trace_sets": {"zipf_exponential": {"resnet": expected}}}
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        assert timeout == 60
        return io.BytesIO(raw)

    audit.acquire(catalog, "twitter", tmp_path, True, opener)
    path = tmp_path / "zipf_exponential/resnet.txt"
    assert path.read_bytes() == raw
    assert audit.COMMIT in calls[0]
    path.write_bytes(b"100,1,200\n110,1,200\n")
    with pytest.raises(ValueError):
        audit.acquire(catalog, "twitter", tmp_path, True, opener)
    assert len(calls) == 1


def test_unverified_download_cannot_create_a_trace(tmp_path):
    expected = audit.inspect_trace(b"100,1,300\n")
    catalog = {"trace_sets": {"zipf_exponential": {"resnet": expected}}}
    with pytest.raises(ValueError):
        audit.acquire(catalog, "twitter", tmp_path, True,
                      lambda *a, **kw: io.BytesIO(b"100,1,200\n"))
    assert not list(tmp_path.rglob("*.txt"))


@pytest.mark.parametrize("raw", [b"", b"1,1\n", b"2,1,300\n1,1,300\n", b"1,1,0\n"])
def test_reject_malformed_source(raw):
    with pytest.raises(ValueError):
        audit.inspect_trace(raw)
