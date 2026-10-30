"""CPU-side invariants; GPU/Container execution is explicitly not simulated as a pass."""
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.resnet import assets, client, host
from experiments.resnet.preparation import Preparation
from experiments.resnet.runtime import Runtime


@pytest.fixture
def jpeg():
    data = io.BytesIO()
    Image.new("RGB", (310, 260), (100, 130, 160)).save(data, format="JPEG")
    return data.getvalue()


def test_only_request_preprocessing_changes_and_input_is_identical(jpeg):
    a, b = Preparation(jpeg, "v1"), Preparation(jpeg, "v2")
    before_a, before_b = a.decode_calls, b.decode_calls
    for _ in range(3):
        x, y = a.prepare(jpeg), b.prepare(jpeg)
        np.testing.assert_array_equal(x["inputs"]["pixel_values"], y["inputs"]["pixel_values"])
        assert x["contract"] == y["contract"]
        assert x["timings"] == {"decode": 0, "preprocess": 0}
    assert a.decode_calls == before_a
    assert b.decode_calls == before_b+3
    assert x["inputs"]["pixel_values"].dtype == np.float32
    assert not x["inputs"]["pixel_values"].flags.writeable
    with pytest.raises(ValueError, match="pinned"):
        a.prepare(jpeg+b"changed")


def test_v1_reuses_inputs_but_executes_every_gpu_batch(jpeg):
    backend = Runtime.__new__(Runtime)
    preparation = Preparation(jpeg, "v1")
    backend.gpu_lock = threading.RLock()
    backend.preprocess_contract = preparation.contract
    calls = []
    def forward(array):
        calls.append(array)
        result = np.zeros((len(array),1000), dtype=np.float32)
        result[:,len(calls)] = 1
        return result
    backend._forward = forward
    prepared = [preparation.prepare(jpeg)]*2
    assert backend.infer_prepared(prepared)[0]["label_id"] == 1
    assert backend.infer_prepared(prepared)[0]["label_id"] == 2
    assert len(calls) == 2 and not np.shares_memory(calls[0], prepared[0]["inputs"]["pixel_values"])


def test_replay_cap_does_not_delay_and_resend_skipped_arrivals(jpeg):
    class SlowClient:
        def query(self, raw, slo, request_id):
            started = time.perf_counter()
            time.sleep(.05)
            return {"sent": True, "ok": True, "slo_met": True, "model": "resnet18_v1", "batch_size": 1,
                    "e2e_ms": 50, "rpc_started": started, "timings": {}}
    rows, _ = client.replay(SlowClient(), jpeg, [(1,1,300)]*4, 200, 1, 1)
    assert sum(r["sent"] for r in rows) == 1
    assert sum(r.get("reject_kind") == "client_not_sent" for r in rows) == 3
    report = client.summary(rows, 1)
    assert report["slo_unmet_ratio_of_all_scheduled_requests"] == .75
    assert report["trace_timing_valid"] is False


def test_late_success_rejection_and_unsent_are_distinct_slo_failures():
    common = {"timings": {}, "model": "resnet18_v1", "batch_size": 1, "send_lag_ms": 0}
    rows = [dict(common, sent=True, ok=True, slo_met=True, scheduled_slo_met=True, e2e_ms=100, server_ms=95),
            dict(common, sent=True, ok=True, slo_met=False, scheduled_slo_met=False, e2e_ms=250, server_ms=190),
            dict(common, sent=True, ok=False, slo_met=False, scheduled_slo_met=False, e2e_ms=100),
            dict(common, sent=False, ok=False, slo_met=False, scheduled_slo_met=False)]
    report = client.summary(rows, 2)
    assert report["successful_late_replies"] == 1
    assert report["failed_or_rejected_sent_requests"] == 1
    assert report["client_not_sent"] == 1
    assert report["slo_unmet_ratio_of_sent_requests"] == pytest.approx(2/3)
    assert report["slo_unmet_ratio_of_all_scheduled_requests"] == .75
    assert report["mean_e2e_ms"] == 175
    assert report["slo_met_qps"] == .5
    assert report["server_internal"]["slo_met_successful_replies"] == 2
    assert report["server_internal"]["slo_unmet_ratio_of_sent_including_failures"] == pytest.approx(1/3)


def test_source_models_are_exactly_the_five_resnet_v1_variants():
    models = assets.manifest()["models"]
    assert set(models) == {f"resnet{n}_v1" for n in [18,34,50,101,152]}
    assert all(len(v["sha256"]) == 64 for v in models.values())
    assert models["resnet50_v1"]["accuracy"] != models["resnet50_v1"]["checkpoint_published_top1"]


def test_fp32_input_does_not_hide_half_precision_graph_conversion(tmp_path):
    import onnx
    from onnx import helper,TensorProto
    source = helper.make_tensor_value_info("data",TensorProto.FLOAT,["N",3,224,224])
    output = helper.make_tensor_value_info("result",TensorProto.FLOAT,["N",3,224,224])
    weight = helper.make_tensor("scale",TensorProto.FLOAT,[1],[1])
    nodes = [helper.make_node("Mul",["data","scale"],["a"]),
             helper.make_node("Cast",["a"],["b"],to=TensorProto.FLOAT16),
             helper.make_node("Cast",["b"],["result"],to=TensorProto.FLOAT)]
    model = helper.make_model(helper.make_graph(nodes,"mixed",[source],[output],[weight]))
    path = tmp_path/"mixed.onnx"
    onnx.save(model,str(path))
    entry = {"file":path.name,"bytes":path.stat().st_size,"sha256":assets.sha256(path)}
    with pytest.raises(ValueError,match="Mixed/half"):
        assets.check_model(path,entry)


def test_phase_selection_does_not_change_other_engine_configuration(tmp_path):
    source = host.HERE
    destination = tmp_path/"experiments/resnet"
    destination.mkdir(parents=True)
    for name in ["settings.env", "engines.json", "phase.json"]:
        (destination/name).write_bytes((source/name).read_bytes())
    engine = json.loads((destination/"engines.json").read_text())[0]
    (destination/"phase.json").write_text('{"phase":"v1","request_preprocessing":false}')
    a = host.settings(engine,tmp_path)
    (destination/"phase.json").write_text('{"phase":"v2","request_preprocessing":true}')
    b = host.settings(engine,tmp_path)
    changed = {k for k in a if a[k] != b[k]}
    assert changed == {"EXPERIMENT_PHASE", "WORKER_IMAGE", "PROJECT", "RUN_DIR"}


def test_compose_is_scoped_to_selected_gpu_and_native_clipper_policy():
    engine = json.loads((host.HERE/"engines.json").read_text())[0]
    c = host.settings(engine)
    c.update(ENGINE="clipper", MODEL_NAME="resnet152_v1")
    spec = host.compose_spec(c)
    worker = spec["services"]["worker"]
    assert worker["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["2"]
    assert "deploy" not in spec["services"]["gateway"]
    assert "--prediction_cache_size=0" in spec["services"]["query"]["command"]
    assert spec["services"]["query"]["environment"]["CLIPPER_MAX_READY_QUEUE"] == "0"
    assert all(m.get("read_only") for m in worker["volumes"][:2])


def test_wrong_daemon_fails_before_any_mutation(monkeypatch):
    calls = []
    monkeypatch.setattr(host, "run", lambda args, *a, **kw:
        calls.append(args) or '{"DockerRootDir":"/mnt/sdb/docker","SecurityOptions":[]}')
    with pytest.raises(RuntimeError, match="expected"):
        host.docker_info({"EXPECTED_DOCKER_ROOT":"/data4tb/taeho-proteus/docker-data"})
    assert calls == [["docker","info","--format","{{json .}}"]]


def test_empty_docker_output_has_a_useful_error(monkeypatch):
    monkeypatch.setattr(host, "run", lambda *a, **kw: "")
    with pytest.raises(RuntimeError, match="no info"):
        host.docker_info({"EXPECTED_DOCKER_ROOT":""})


def test_original_arrival_zero_and_qos_are_preserved(tmp_path):
    path = tmp_path/"trace.txt"
    path.write_text("100,1,300\n1000,1,300\n")
    assert client.read_trace(path) == [(100,1,300),(1000,1,300)]
    assert client.read_trace(path, 1) == [(100,1,300)]
    path.write_text("1000,1,300\n100,1,300\n")
    with pytest.raises(ValueError):
        client.read_trace(path)
