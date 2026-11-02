"""Published policy differential checks; fake GPU is explicitly not GPU validation."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import subprocess

import pytest
from core.common import Event,EventType,TaskAssignment
from core.predictor import Predictor
from algorithms.ilp import Ilp
from experiments.resnet.test_native_interfaces import fake_backend
from experiments.reproduce.proteus_executor import Executor
from experiments.reproduce.integrity import check
from proteus_realtime.protocol import InferRequest


@pytest.mark.parametrize("engine",["proteus","infaas-accuracy"])
def test_actual_rpc_executor_calls_original_routing_and_batch_methods(engine,monkeypatch,tmp_path):
    import core.executor
    calls={"routing":0,"batching":0}
    route=core.executor.Executor.enqueue_request
    batch=Predictor.enqueue_request
    def routed(self,*args,**kwargs):
        calls["routing"]+=1
        return route(self,*args,**kwargs)
    def batched(self,*args,**kwargs):
        calls["batching"]+=1
        return batch(self,*args,**kwargs)
    monkeypatch.setattr(core.executor.Executor,"enqueue_request",routed)
    monkeypatch.setattr(Predictor,"enqueue_request",batched)
    # A call to the earlier replacement policy would fail this test.
    import proteus_realtime.policy as replacement
    def forbidden(*args,**kwargs):raise AssertionError("Replacement policy must not run")
    monkeypatch.setattr(replacement,"solve_milp",forbidden)
    monkeypatch.setattr(replacement,"adaptive_decision",forbidden)
    backend,jpeg=fake_backend()
    executor=Executor(backend,engine,profile_path=tmp_path/"profiles.json",batches=(1,),interval=60)
    try:
        executor.profile(jpeg,2)
        reply=executor.submit(InferRequest(model="imagenet",image=jpeg,slo=200)).result(timeout=3)
        assert not reply.rejected
        assert calls=={"routing":1,"batching":1}
        assert reply.timings["worker.batch_size"]==1
        assert reply.timings["measurement.post_preprocess"]>0
        state=executor.state()
        assert state["additional_rpc_reserve_ms"]==state["additional_deadline_guard_ms"]==0
        assert state["additional_early_drop"] is False
        if engine=="proteus":
            executor.native.total_requests_arr=[10]
            executor.native.sample_demand()
            executor.native.compute_control()
            assert executor.native.solver.__class__ is Ilp
            assert executor.native.configured_accelerators=={"GPU_AMPERE-0"}
            assert executor.native.solver.cached_solution is not None
    finally:executor.close()


def test_source_manifest_permits_only_two_platform_adaptations():
    report=check()
    assert set(report["adaptations"])=={"algorithms/ilp.py","core/predictor.py"}
    assert report["custom_MILP_or_batch_policy_used"] is False


def test_container_contains_original_profile_loader_and_model_family_metadata(tmp_path,monkeypatch):
    import shutil
    from experiments.reproduce.proteus_policy import ROOT
    app=tmp_path/"app"
    # Match the actual COPY layout, including data used by original Simulator.
    for name in ("core","algorithms","experiments/resnet","experiments/reproduce"):
        shutil.copytree(ROOT/name,app/name,ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT/"run.py",app/"run.py")
    destination=app/"traces/model_variants";destination.mkdir(parents=True)
    shutil.copy2(ROOT/"traces/model_variants/resnet",destination/"resnet")
    assert check(app)["verified_files"]==11
    assert len((destination/"resnet").read_text().splitlines())==5
    docker=(ROOT/"experiments/resnet/Dockerfile").read_text()
    assert "COPY traces/model_variants/resnet /app/traces/model_variants/resnet" in docker


def original_expiry_callback():
    text=subprocess.check_output(["git","show",
        "6967c9327e66f6d6fe94993dcaa7521efc2f72f5:core/predictor.py"],text=True)
    tree=ast.parse(text)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="Predictor")
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="slo_expiring_callback")
    import core.predictor
    namespace=dict(vars(core.predictor))
    exec(compile(ast.Module(body=[method],type_ignores=[]),"<published Predictor>","exec"),namespace)
    return namespace["slo_expiring_callback"]


def timer_case(callback,event_time,clock,stored_time):
    calls=[]
    request=Event(0,EventType.START_REQUEST,"imagenet",deadline=200,id="request")
    logger=SimpleNamespace(debug=lambda *a,**k:None,warn=lambda *a,**k:None)
    obj=SimpleNamespace(busy=False,batching_algo="accscale",task_assignment=TaskAssignment.CANARY,log=logger,
        slo_expiring_dict={"request":stored_time},request_queue=[request],
        find_batch_size=lambda requests:1,max_batch_size=1,
        process_batch=lambda when,size:calls.append((when,size)))
    timer=Event(event_time,EventType.SLO_EXPIRING,"imagenet",event_counter="request")
    callback(obj,timer,clock)
    return calls


@pytest.mark.parametrize("event,stored",[(1000,1000),(900,1000)])
def test_timestamp_adaptation_is_equivalent_under_original_virtual_clock(event,stored):
    assert timer_case(original_expiry_callback(),event,event,stored)==timer_case(
        Predictor.slo_expiring_callback,event,event,stored)


def test_os_wakeup_lag_does_not_turn_a_current_event_into_a_stale_event():
    assert timer_case(Predictor.slo_expiring_callback,1000,1005,1000)==[(1005,1)]


def test_new_executor_is_reachable_through_the_real_protobuf_rpc(monkeypatch,tmp_path):
    import grpc
    from proteus_realtime.server import create_server
    from experiments.resnet.server import add_admin
    from experiments.resnet.client import Client
    backend,jpeg=fake_backend()
    executor=Executor(backend,"proteus",profile_path=tmp_path/"p.json",batches=(1,),interval=60)
    server=None
    client=None
    try:
        executor.profile(jpeg,2)
        server,port=create_server(executor,"127.0.0.1:0",4)
        add_admin(server,executor,backend,"proteus")
        server.start()
        original=grpc.insecure_channel
        monkeypatch.setattr(grpc,"insecure_channel",lambda address,**kw:original(f"127.0.0.1:{port}",**kw))
        client=Client("proteus",5)
        state=client.admin("Status")
        assert state["source_integrity"]["custom_MILP_or_batch_policy_used"] is False
        assert state["plan"]["model"]==state["model_name"]=="resnet152_v1"
        reply=client.query(jpeg,200,"id")
        assert reply["ok"] and reply["model"]=="resnet152_v1"
        assert "measurement.post_preprocess" in reply["timings"]
    finally:
        if client:client.channel.close()
        if server:server.stop(0).wait()
        executor.close()
