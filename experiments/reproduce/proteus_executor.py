"""Wall-clock I/O adapter for published policies; no new scheduling decisions."""
from collections import deque
from concurrent.futures import Future,ThreadPoolExecutor
from dataclasses import dataclass
import heapq
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid

from core.common import EventType
from proteus_realtime.protocol import InferReply
from .proteus_policy import PublishedSimulator
from .integrity import check


@dataclass
class Job:
    request:object
    prepared:object
    arrival:float
    ready:float
    name:str
    future:Future


class Executor:
    def __init__(self,backend,engine,profile_path="/profiles/published-source-profiles.json",
                 slo=200,batches=(1,2,4,8),interval=10):
        self.backend,self.engine=backend,engine
        self.integrity=check()
        self.slo,self.batches,self.interval=float(slo),tuple(batches),float(interval)
        if engine not in ("proteus","infaas-accuracy"):raise ValueError("Unsupported published policy")
        self.origin=time.perf_counter()
        self.condition=threading.Condition(threading.RLock())
        self.jobs,self.incoming,self.completed={},deque(),deque()
        self.timers,self.sequence=[],0
        self.control_solutions=deque()
        self.gpu=ThreadPoolExecutor(max_workers=1,thread_name_prefix="source-real-gpu")
        self.control=ThreadPoolExecutor(max_workers=1,thread_name_prefix="source-Ilp")
        self.control_future=None
        self.native=None
        self.profile_path=Path(profile_path)
        self.profiles={}
        self.maintenance=self.stopping=False
        self.processed=self.failed=self.active_batches=0
        self.batch_counts,self.model_counts={},{}
        self.error=None
        for p in ("logs/ilp","logs/latency","logs/per_predictor"):
            Path(p).mkdir(parents=True,exist_ok=True)
        if self.profile_path.exists():
            d=json.loads(self.profile_path.read_text())
            if (d.get("schema")=="published-policies-real-gpu-v1" and d["fingerprint"]==backend.fingerprint()
                    and d["slo_ms"]==self.slo and d["batches"]==list(self.batches)):
                self.profiles=d["profiles"];self.install()
            else:self.error="Source profile mismatch; run register"
        self.next_sample=time.perf_counter()+1
        self.next_control=time.perf_counter()+self.interval
        self.thread=threading.Thread(target=self._loop,daemon=True,name="source-events")
        self.thread.start()

    def clock_ms(self):
        return (time.perf_counter()-self.origin)*1000

    def install(self):
        if self.native:self.native.close()
        self.timers.clear()
        self.control_solutions.clear()
        self.native=PublishedSimulator(self,self.profiles)
        self.error=None

    def submit(self,request):
        accepted=time.perf_counter()
        if request.model!=self.backend.application or abs(request.slo-self.slo)>1e-6:
            raise ValueError("Use the configured application and SLO")
        prepared=self.backend.prepare(request.image)
        job=Job(request,prepared,(accepted-self.origin),(time.perf_counter()-self.origin),
                uuid.uuid4().hex,Future())
        with self.condition:
            if self.maintenance or self.stopping or self.native is None:
                raise ValueError("Wait until registered and idle")
            self.jobs[job.name]=job
            self.incoming.append(job)
            self.condition.notify_all()
        return job.future

    def schedule(self,when,event):
        self.sequence+=1
        heapq.heappush(self.timers,(float(when),self.sequence,event))

    def reject(self,identifier,reason):
        job=self.jobs.pop(identifier,None)
        if job is not None:
            self.failed+=1
            if not job.future.done():
                job.future.set_result(InferReply(rejected=True,reject_kind=reason,reason=reason,
                    request_name=identifier,latency=self.clock_ms()-job.arrival*1000))

    def launch(self,events,predictor):
        jobs=[self.jobs[e.id] for e in events if e.id in self.jobs]
        if len(jobs)!=len(events):raise RuntimeError("Original batch membership missing a real request")
        self.active_batches+=1
        def execute():
            start=time.perf_counter()
            loaded=self.backend.load(predictor.variant_name)
            gpu_start=time.perf_counter()
            outputs=self.backend.infer_prepared([j.prepared for j in jobs])
            finish=time.perf_counter()
            if len(outputs)!=len(jobs):raise RuntimeError("Real batch output cardinality differs")
            replies=[]
            for job,out in zip(jobs,outputs):
                server=(finish-self.origin-job.arrival)*1000
                common=(finish-self.origin-job.ready)*1000
                gpu_ms=(finish-gpu_start)*1000
                timings={"worker."+k:float(v) for k,v in out["timings"].items()}
                timings.update({"worker.batch_size":len(jobs),"worker.load":loaded["load_ms"],
                    "worker.gpu_service":gpu_ms,"worker.ready_queue":(start-self.origin-job.ready)*1000,
                    "measurement.post_preprocess":common,"server.slo_missed":float(server>self.slo)})
                replies.append(InferReply(request_name=job.name,inference=out["label"],variant=predictor.variant_name,
                    gpu=self.backend.gpu_type,latency=server,worker_ms=(finish-start)*1000,timings=timings))
            return replies
        future=self.gpu.submit(execute)
        def done(f):
            with self.condition:
                self.completed.append((events,predictor,jobs,f))
                self.condition.notify_all()
        future.add_done_callback(done)

    def _complete(self):
        while self.completed:
            events,predictor,jobs,future=self.completed.popleft()
            self.active_batches-=1
            try:
                replies=future.result()
                for job,reply in zip(jobs,replies):
                    self.jobs.pop(job.name,None)
                    self.processed+=1
                    self.model_counts[reply.variant]=self.model_counts.get(reply.variant,0)+1
                    if not job.future.done():job.future.set_result(reply)
                n=str(len(jobs));self.batch_counts[n]=self.batch_counts.get(n,0)+1
            except Exception as error:
                for job in jobs:self.reject(job.name,"gpu_error: "+str(error))
            # Only physical completion replaces the simulator's predicted END/FINISH.
            self.native.complete(events,predictor)

    def _loop(self):
        while True:
            with self.condition:
                if self.stopping:return
                try:
                    self._complete()
                    while self.control_solutions:self.native.apply_completed_control(self.control_solutions.popleft())
                    if self.maintenance:
                        self.condition.wait(.01);continue
                    while self.incoming:self.native.enqueue(self.incoming.popleft())
                    now=time.perf_counter()
                    while self.timers and self.timers[0][0]<=self.clock_ms():
                        self.native.timer(heapq.heappop(self.timers)[2])
                    if self.native is not None and now>=self.next_sample:
                        # Published 1-second accounting window and published EWMA.
                        self.native.sample_demand()
                        self.next_sample=now+1
                    if self.native is not None and now>=self.next_control:
                        if self.engine=="proteus":
                            if self.control_future is None or self.control_future.done():
                                if self.control_future is not None:
                                    try:self.control_future.result()
                                    except Exception as e:self.error="Original Ilp: "+str(e)
                                self.control_future=self.control.submit(self.native.compute_control)
                        else:self.native.compute_control()
                        self.next_control=now+self.interval
                    delay=.01
                    if self.timers:delay=min(delay,max(0,(self.timers[0][0]-self.clock_ms())/1000))
                    self.condition.wait(delay)
                except Exception as error:
                    logging.exception("Published policy reported an error")
                    self.error=str(error)
                    for identifier in list(self.jobs):self.reject(identifier,"original_policy_error")
                    self.incoming.clear();self.timers.clear()
                    self.condition.wait(.05)

    def _exclusive(self,function):
        with self.condition:
            if self.jobs or self.active_batches or self.maintenance:
                raise ValueError("Administrative operation requires no outstanding original-policy requests")
            if self.control_future is not None and not self.control_future.done():
                raise ValueError("Wait for original Ilp invocation to finish")
            self.maintenance=True
        try:return self.gpu.submit(function).result()
        finally:
            with self.condition:self.maintenance=False;self.condition.notify_all()

    def profile(self,image,trials=30):
        def measure():
            profiles={}
            for name in self.backend.enabled:
                loaded=self.backend.load(name)
                prepared=self.backend.prepare(image)
                batches={}
                for batch in self.batches:
                    for _ in range(3):self.backend.infer_prepared([prepared]*batch)
                    samples=[]
                    for _ in range(int(trials)):
                        start=time.perf_counter()
                        self.backend.infer_prepared([prepared]*batch)
                        samples.append((time.perf_counter()-start)*1000)
                    batches[str(batch)]={"samples_ms":samples,"mean_ms":sum(samples)/len(samples)}
                profiles[name]={"batches":batches,"actual_load":loaded}
                self.backend.reset()
            self.profiles=profiles
            with self.condition:self.install()
            data={"schema":"published-policies-real-gpu-v1","fingerprint":self.backend.fingerprint(),
                "slo_ms":self.slo,"batches":list(self.batches),"profiles":profiles}
            self.profile_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.profile_path.with_suffix(".tmp");temporary.write_text(json.dumps(data,indent=2))
            os.replace(temporary,self.profile_path)
            return {"policy_ready":True,"profiles":profiles}
        if int(trials)<2:raise ValueError("At least two profiling trials required")
        return self._exclusive(measure)

    def selected_model(self):
        if not self.native:raise ValueError("Register first")
        names=[p.variant_name for e in self.native.executors.values() for p in e.predictors.values()]
        if len(names)!=1:raise ValueError("Original policy has no single resident allocation")
        return names[0]

    def warmup(self):
        return self._exclusive(lambda:self.backend.load(self.selected_model()))

    def reset(self):
        return self._exclusive(lambda:(self.backend.reset() or {"resident":False}))

    def state(self):
        with self.condition:
            predictors=[p for e in self.native.executors.values() for p in e.predictors.values()] if self.native else []
            selected=predictors[0].variant_name if len(predictors)==1 else None
            plan={"model":selected,"max_batch":predictors[0].max_batch_size,
                "capacity_qps":predictors[0].peak_throughput,"source":"published predictor/allocation"} if selected else None
            return {**self.backend.describe(),"implementation":"published-proteus-policy-real-gpu-adapter",
                "plan":plan,"selected_model":selected,"resident_model":getattr(self.backend,"current",None),
                "model_name":getattr(self.backend,"current",None) or selected,
                "model_id":self.backend.registry[selected]["model_id"] if selected else None,
                "engine":self.engine,"policy_ready":self.native is not None,"policy_error":self.error,
                "source_integrity":self.integrity,
                "policy_entrypoints":["algorithms.ilp.Ilp.run" if self.engine=="proteus" else
                    "core.simulator.Simulator.trigger_infaas_v2_upscaling/downscaling",
                    "core.executor.Executor.enqueue_request","core.predictor.Predictor.enqueue_request"],
                "source_model_assignment":"ilp" if self.engine=="proteus" else "infaas_v2",
                "batching":"accscale" if self.engine=="proteus" else "infaas",
                "processed":self.processed,"failed":self.failed,"outstanding_requests":len(self.jobs),
                "busy":self.active_batches>0,"batch_size_counts":self.batch_counts,
                "successful_requests_by_model":self.model_counts,
                "model_selection_history":list(self.native.selections) if self.native else [],
                "source_ilp_stats":self.native.ilp_stats if self.native else {},
                "native_controller_calls":self.native.controller_calls if self.native else 0,
                "request_slo_ms":self.slo,"additional_rpc_reserve_ms":0,"additional_deadline_guard_ms":0,
                "additional_early_drop":False,"additional_queue_limit":False,
                "profile_statistic":"original CSV 50th_pct mapping retained; mean also recorded",
                "scope":"Real GPU completion adapter, not the unpublished production cluster code"}

    def close(self):
        with self.condition:self.stopping=True;self.condition.notify_all()
        self.thread.join(timeout=10)
        self.control.shutdown(wait=True)
        self.gpu.shutdown(wait=True)
        if self.native:self.native.close()
        self.backend.reset()
