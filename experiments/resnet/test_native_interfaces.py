"""Real native RPC/policy paths with an explicitly fake GPU backend."""
import base64
import io
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.resnet.assets import manifest
from experiments.resnet.preparation import Preparation
from experiments.resnet.runtime import Runtime, InfaasBackend


def fake_backend():
    stream = io.BytesIO()
    Image.new("RGB", (250,250), (130,140,150)).save(stream,format="JPEG")
    jpeg = stream.getvalue()
    runtime = Runtime.__new__(Runtime)
    runtime.preparation = Preparation(jpeg,"v1")
    runtime.preprocess_contract = runtime.preparation.contract
    runtime.registry = manifest()["models"]
    runtime.enabled = list(runtime.registry)
    runtime.application, runtime.gpu_type = "imagenet", "v100"
    runtime.model_name = "resnet50_v1"
    runtime.model_id = runtime.registry[runtime.model_name]["model_id"]
    runtime.current = None
    runtime.gpu_lock = threading.RLock()
    runtime.memory_baseline_free = 1_000_000_000
    def forward(array):
        result = np.zeros((len(array),1000),dtype=np.float32)
        result[:,258] = 1
        return result
    runtime._forward = forward
    def load(name=None):
        name = name or runtime.model_name
        loaded = name != runtime.current
        runtime.current = name
        return {"loaded":loaded,"load_ms":.1 if loaded else 0,"ready_ms":.1,"stages":{}}
    runtime.load = load
    runtime.reset = lambda: setattr(runtime,"current",None)
    runtime.fingerprint = lambda: {"test_double_not_gpu":True}
    runtime.describe = lambda: {"application":"imagenet", "enabled_models":runtime.enabled,
        "model_name":runtime.current or runtime.model_name,"resident":runtime.current is not None,
        "preprocessing_loading_overlap":False,"runtime_backend":"test-double-not-gpu"}
    runtime.torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda:
        (runtime.memory_baseline_free-runtime.registry[runtime.current]["bytes"],2_000_000_000)))
    return runtime,jpeg


@pytest.mark.parametrize("engine",["proteus","infaas-accuracy"])
def test_native_proteus_and_accuracy_rpc_accept_the_new_backend(engine,monkeypatch):
    module = pytest.importorskip("proteus_realtime.server")
    import grpc
    from experiments.resnet.client import Client
    from experiments.resnet.server import add_admin
    runtime,jpeg = fake_backend()
    monkeypatch.setenv("PREPROCESS_WORKERS","1")
    monkeypatch.setenv("PROFILE_STRATEGY","continuous")
    if engine == "proteus":
        from proteus_realtime.serving import ProteusExecutor
        # This test targets RPC/backend binding, not OS timer precision. Native
        # batching has its own existing tests; sub-millisecond fake profiles on
        # Windows must not determine whether the interface test succeeds.
        executor = ProteusExecutor(runtime,slo=200,batches=(1,),interval=60,rpc_budget_ms=6,guard_ms=50)
    else:
        from proteus_baselines.infaas_accuracy import AccuracyExecutor
        executor = AccuracyExecutor(runtime,slo=200,batches=(1,),interval=60)
    executor.profile(jpeg,2)
    server,port = module.create_server(executor,"127.0.0.1:0",4)
    add_admin(server,executor,runtime,engine)
    server.start()
    original = grpc.insecure_channel
    monkeypatch.setattr(grpc,"insecure_channel",lambda address,**kw:original(f"127.0.0.1:{port}",**kw))
    client = Client(engine,5)
    try:
        assert client.admin("Status")["native_state"]["policy_ready"] is True
        row = client.query(jpeg,200,"test")
        assert row["ok"] is True, json.dumps(row,indent=2)
        assert row["model"] == ("resnet152_v1" if engine == "proteus" else "resnet18_v1")
        assert row["batch_size"] >= 1
    finally:
        client.channel.close()
        server.stop(0).wait()
        executor.close()


def test_native_infaas_proto_rpc_retains_cost_selection(monkeypatch,tmp_path):
    pytest.importorskip("infaas.single_gpu.service")
    import grpc
    from concurrent.futures import ThreadPoolExecutor
    from infaas.single_gpu.settings import Settings
    from infaas.single_gpu.server import add_services
    from experiments.resnet.assets import HERE
    from experiments.resnet.client import Client
    from experiments.resnet.server import add_admin,make_infaas_executor
    runtime,jpeg = fake_backend()
    backend = InfaasBackend.__new__(InfaasBackend)
    backend.runtime,backend.fingerprint = runtime,runtime.fingerprint()
    settings = Settings({"MODEL_REGISTRY":str(HERE/"models.json"),"PROFILE_DIR":str(tmp_path),"PROFILE_RUNS":"2"})
    monkeypatch.setenv("HW_COST","v100:1")
    monkeypatch.setenv("LAMBDA","1")
    # Native costs snapshot environment on import; collection may import them
    # before this test. Production receives HW_COST before Python starts.
    from infaas.common import config as cost_config
    monkeypatch.setattr(cost_config, "HW_COST", {"v100": 1.0})
    executor = make_infaas_executor(settings,backend)
    for name in runtime.enabled:
        executor.profile(name,jpeg,2)
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    add_services(server,executor)
    add_admin(server,executor,runtime,"infaas")
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    original = grpc.insecure_channel
    monkeypatch.setattr(grpc,"insecure_channel",lambda address,**kw:original(f"127.0.0.1:{port}",**kw))
    client = Client("infaas",5)
    try:
        assert client.admin("Status")["native_state"]["policy_ready"] is True
        row = client.query(jpeg,200,"test")
        assert row["ok"] is True and row["model"] == "resnet18_v1", json.dumps(row,indent=2)
        assert row["batch_size"] == 1
    finally:
        client.channel.close()
        server.stop(0).wait()
        executor.close()


def test_clipper_uses_its_existing_prepared_wire_contract(monkeypatch):
    pytest.importorskip("clipper_gpu.prepared")
    import torch
    from experiments.resnet.clipper import GatewayProcessor,Model
    runtime,jpeg = fake_backend()
    runtime.torch = torch
    processor = GatewayProcessor.__new__(GatewayProcessor)
    processor.preparation = runtime.preparation
    received = time.time_ns()//1000
    wire = processor.prepare({"request_id":"id","image_b64":base64.b64encode(jpeg).decode()},received,received+200_000)
    monkeypatch.setenv("MODEL_NAME","resnet152_v1")
    model = Model(runtime)
    result = json.loads(model.predict_strings([json.dumps(wire)])[0])
    assert result["request_id"] == "id"
    assert result["model"] == "resnet152_v1"
    assert result["label"] == "ImageNet class 258"
    assert result["batch_size"] == 1
    assert result["timings"]["worker.cpu_service"] == 0
