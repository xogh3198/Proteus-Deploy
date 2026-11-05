"""CPU-only checks for diagnostic isolation, observation and native RPC wiring."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from diagnostics import onnx_forward as diag


def args(**updates):
    values = dict(model="resnet152_v1", trials=2, trace_limit=2,
                  batches=[1,8], gaps_ms=[0], dry_run=False, worker=False)
    values.update(updates)
    return SimpleNamespace(**values)


def inspect_worker(tmp_path):
    profile = tmp_path/"profiles"
    profile.mkdir()
    (profile/"published-source-profiles.json").write_text('{"test":"original"}')
    mounts = [{"Destination": dst, "Source": str(src), "Type": "bind"} for dst,src in (
        ("/models",tmp_path/"models"), ("/inputs/image.jpg",tmp_path/"image.jpg"),
        ("/profiles",profile), ("/results",tmp_path/"original-results"))]
    return {"Name":"/original-worker", "Image":"sha256:immutable",
            "State":{"Pid":123}, "Mounts":mounts,
            "Config":{"User":"0:0", "Env":["CPU_THREADS=2", "EXPERIMENT_PHASE=v1"]},
            "HostConfig":{"DeviceRequests":[{"DeviceIDs":["2"]}], "CpuQuota":200000, "CpuPeriod":100000}}


def test_probe_forwards_the_same_binding_and_results():
    calls=[]
    class Binding:
        def bind_cpu_input(self,*a): calls.append(("input",a))
        def bind_output(self,*a): calls.append(("output",a))
        def copy_outputs_to_cpu(self): return ["same result"]
    real_binding=Binding()
    class Session:
        def io_binding(self): return real_binding
        def run_with_iobinding(self,binding):
            assert binding is real_binding
            calls.append(("run",binding))
    recorder=diag.Recorder()
    recorder.details=True
    row={"stages":{}}
    recorder.local.row=row
    session=diag.SessionProbe(Session(),recorder)
    binding=session.io_binding()
    binding.bind_cpu_input("data","same input")
    binding.bind_output("output","cuda",0)
    session.run_with_iobinding(binding)
    assert binding.copy_outputs_to_cpu()==["same result"]
    assert len(calls)==3
    assert set(row["stages"])=={"io_binding","bind_cpu_input","bind_output","ort_run","copy_outputs_to_cpu"}
    assert all(s["wall_ms"]>=0 and s["thread_cpu_ms"]>=0 for s in row["stages"].values())


def test_plan_uses_original_image_and_read_only_assets(tmp_path):
    info=inspect_worker(tmp_path)
    original=copy.deepcopy(info)
    command,name,gpu,profile=diag.launch_plan(info, tmp_path/"probe.py", tmp_path/"new-results", args())
    assert info==original and gpu=="2"
    assert "sha256:immutable" in command
    bindings=[command[i+1] for i,arg in enumerate(command) if arg=="--mount"]
    for target in ("/models","/inputs/image.jpg","/profiles"):
        assert any(("dst="+target+",") in mount and mount.endswith(",readonly") for mount in bindings)
    assert not any("original-results" in mount for mount in bindings)
    assert "--cpu-quota" in command and "--cpu-period" in command
    assert diag.digest(profile)==diag.digest(tmp_path/"profiles/published-source-profiles.json")


@pytest.mark.parametrize("dry",[True,False])
def test_dry_run_is_read_only_and_worker_failure_restores_original(tmp_path,monkeypatch,dry):
    from experiments.resnet import host
    info=inspect_worker(tmp_path)
    config={"PROJECT":"source-proteus-v1-twitter", "RUN_DIR":str(tmp_path/"run")}
    monkeypatch.setattr(host,"settings",lambda *a,**k:config)
    monkeypatch.setattr(host,"docker_info",lambda c:None)
    monkeypatch.setattr(diag,"ancestor_cpu_limits",lambda p:{})
    monkeypatch.setattr(diag,"docker_status",lambda c:{"policy_ready":True,"experiment_phase":"v1"})
    def read(command,**kwargs):
        if command[1]=="ps": return "original-id\n"
        if command[1]=="inspect": return json.dumps([info])
        if command[0]=="nvidia-smi": return ""
        raise AssertionError(command)
    monkeypatch.setattr(diag.subprocess,"check_output",read)
    mutations=[]
    monkeypatch.setattr(diag.subprocess,"run",lambda command,**kw:mutations.append(command))
    class Failure:
        stdout=["diagnostic failed on purpose\n"]
        def wait(self): return 1
    monkeypatch.setattr(diag.subprocess,"Popen",lambda *a,**k:Failure())
    options=args()
    options.dry_run=dry
    if dry:
        diag.host_main(options)
        assert mutations==[] and not (tmp_path/"run").exists()
    else:
        with pytest.raises(RuntimeError,match="Diagnostic failed"):
            diag.host_main(options)
        assert ["docker","stop","original-id"] in mutations
        assert ["docker","start","original-id"] in mutations
        assert json.loads((tmp_path/"profiles/published-source-profiles.json").read_text())=={"test":"original"}


@pytest.mark.parametrize("context",[False,True])
def test_real_native_rpc_path_with_explicit_fake_gpu(tmp_path,monkeypatch,context):
    from experiments.resnet.test_native_interfaces import fake_backend
    from experiments.reproduce.proteus_executor import Executor
    runtime,jpeg=fake_backend()
    runtime.session=None
    runtime.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
    profile=tmp_path/"profile.json"
    executor=Executor(runtime,"proteus",profile_path=profile,slo=300,batches=(1,8),interval=60)
    try:
        executor.profile(jpeg,2)
    finally:
        executor.close()
    monkeypatch.setenv("ALLOCATION_INTERVAL","60")
    monkeypatch.setenv("RPC_THREADS","4")
    operation=diag.measure_context if context else diag.measure
    report=operation(runtime,args(),profile,jpeg,[(1,1,300),(400,1,300)],tmp_path)
    assert report["status"]=="complete"
    assert {row["observer"] for row in report["rows"]}==({True} if context else {False,True})
    assert any(row["case"]==("context_grpc" if context else "native_rpc_twitter") for row in report["rows"])
    names = ("context_grpc.json","context_submit_no_grpc.json") if context else ("rpc-observer-0.json","rpc-observer-1.json")
    for name in names:
        data=json.loads((tmp_path/name).read_text())
        assert data["summary"]["sent_requests"]==2
        assert data["summary"]["successful_replies"]==2
    if context:
        locations={r["case"]:r["thread_id"] for r in report["contexts"]}
        assert locations["context_before_imports"]==locations["context_after_imports"]
        assert locations["context_native_gpu_thread"]==locations["context_server_started"]==locations["context_after_rpc"]
    assert diag.digest(profile)==report["original_profile_sha256"]
