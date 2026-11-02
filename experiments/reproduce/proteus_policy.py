"""Use published Simulator/Executor/Predictor/Ilp; bridge completions to real GPU work.

No local MILP objective, adaptive-batch decision, early-drop predicate or RPC
reserve is defined here. The two declared source adaptations filter the physical
device inventory and retain timer identity when the OS wakes after its due time.
"""
from collections import defaultdict,deque
import csv
import hashlib
import heapq
import json
import logging
from pathlib import Path
import tempfile
from statistics import median

import numpy as np
from core.simulator import Simulator
from core.common import Event,EventType
from core.predictor import AccType
from algorithms.ilp import Ilp

ROOT=Path(__file__).resolve().parents[2]
BASE="6967c9327e66f6d6fe94993dcaa7521efc2f72f5"


class PublishedSimulator(Simulator):
    def __init__(self,owner,profiles):
        self.owner=owner
        self.inputs=tempfile.TemporaryDirectory(prefix="proteus-original-inputs-",ignore_cleanup_errors=True)
        root=Path(self.inputs.name)
        trace=root/"trace";trace.mkdir()
        variants=root/"variants";variants.mkdir()
        app=owner.backend.application
        (trace/(app+".txt")).write_text(f"0,1,{owner.slo}\n")
        original=dict(line.split(",") for line in (ROOT/"traces/model_variants/resnet").read_text().splitlines())
        (variants/app).write_text("".join(f"{name},{original[name]}\n" for name in owner.backend.enabled))
        p=root/"profiles.csv"
        with p.open("w",newline="") as f:
            writer=csv.DictWriter(f,["Model","Accel","Trials","batchsize","50th_pct","avg_latency(ms)","Accuracy","Normalized_Accuracy"])
            writer.writeheader()
            for name,profile in profiles.items():
                for batch,row in profile["batches"].items():
                    writer.writerow({"Model":name,"Accel":"onnxruntime_gpu_ampere","Trials":len(row["samples_ms"]),
                        "batchsize":batch,"50th_pct":median(row["samples_ms"]),
                        "avg_latency(ms)":sum(row["samples_ms"])/len(row["samples_ms"]),
                        "Accuracy":owner.backend.registry[name]["accuracy"],"Normalized_Accuracy":original[name]})
        mode=owner.engine
        super().__init__("canary_routing" if mode=="proteus" else "infaas",logging.WARNING,
            trace_path=trace.as_posix(),mode="debugging",max_acc_per_type=1,predictors_max=[0,1,0,0],
            fixed_seed=10,batching=True,model_assignment="ilp" if mode=="proteus" else "infaas_v2",
            batching_algo="accscale" if mode=="proteus" else "infaas",
            profiling_data=p.as_posix(),allowed_variants_path=variants.as_posix(),
            max_batch_size=max(owner.batches),ewma_decay=1.6,infaas_slack=.05,infaas_downscale_slack=.6)
        self.event_queue.clear()
        self.configured_accelerators={"GPU_AMPERE-0"}
        self.pending=[]
        self.controller_calls=0
        self.selections=deque(maxlen=512)
        self.solver=None
        if mode=="proteus":
            # Model/device starting allocation is experiment configuration.
            feasible=[n for n in owner.backend.enabled if self.largest_batch_sizes[("GPU_AMPERE",n)]>0]
            selected=max(feasible,key=lambda n:float(original[n]))
            Simulator.apply_ilp_solution(self,{(selected,"GPU_AMPERE"):1},
                {("GPU_AMPERE-0",0):1.0},{(selected,"GPU_AMPERE-0"):1})
            self.solver=Ilp(allocation_window=1000,beta=1.05,logging_level=logging.WARNING)
            self.solver.set_simulator(self)
        else:
            # This is the original startup rule, not a new accuracy ranking.
            self.executors[app].add_predictor(AccType.GPU)
            self.available_predictors[1]-=1
        self.record("source_initial_allocation")

    def read_variants_from_file(self,filename):
        # Path formatting only; original parser splits on '/' (Linux deployment).
        return Simulator.read_variants_from_file(self,Path(filename).as_posix())

    def record(self,reason):
        self.selections.append({"time_ms":self.owner.clock_ms(),"reason":reason,
            "predictors":[{"model":p.variant_name,"max_batch":p.max_batch_size,
                "infaas_batch":p.infaas_batch_size} for e in self.executors.values() for p in e.predictors.values()]})

    def enqueue(self,job):
        self.clock=self.owner.clock_ms()
        event=Event(job.arrival*1000,EventType.START_REQUEST,self.owner.backend.application,
                    deadline=job.request.slo,id=job.name,qos_level=0,accuracy=100.0)
        Simulator.process_start_event(self,event,self.clock)

    def generate_end_request_event(self,event,end_time,accuracy_seen,qos_met):
        # Preserve membership selected by original process_batch; a predicted
        # end_time is never used as an actual successful inference completion.
        self.pending.append(event)

    def generate_finish_batch_event(self,finish_time,predictor,executor):
        events,self.pending=self.pending,[]
        self.owner.launch(events,predictor)

    def insert_event(self,start_time,type,desc,**kwargs):
        # Constructor reads one configuration row; runtime timers go to owner.
        event=Event(start_time,type,desc,**kwargs)
        if hasattr(self,"pending"):
            self.owner.schedule(start_time,event)
        else:
            self.event_queue.append(event)

    def timer(self,event):
        clock=self.owner.clock_ms()
        self.clock=clock
        if event.type==EventType.SLO_EXPIRING:
            Simulator.process_slo_expiring_event(self,event,clock)
        elif event.type==EventType.BATCH_EXPIRING:
            Simulator.process_batch_expiring_event(self,event,clock)
        else:raise RuntimeError("Unexpected original event type")

    def bump_failed_request_stats(self,event):
        # Observation only: the original code has already decided to reject.
        self.owner.reject(event.id,"original_policy_drop")

    def complete(self,events,predictor):
        self.clock=self.owner.clock_ms()
        for event in events:
            self.executors[event.desc].finish_request(event,self.owner.clock_ms())
        predictor.finish_batch_callback(self.owner.clock_ms())

    def sample_demand(self):
        Simulator.add_moving_demand(self,np.asarray(self.total_requests_arr).reshape(-1,1))
        self.total_requests_arr=[0 for _ in self.total_requests_arr]

    def compute_control(self):
        self.controller_calls+=1
        if self.solver is not None:
            observation=np.zeros((len(self.executors)+1,12))
            for i,name in self.idx_to_executor.items():
                observation[i,:4]=self.executors[name].num_predictor_types[:4]
            return self.solver.run(observation,4,1)
        Simulator.trigger_infaas_v2_upscaling(self)
        Simulator.trigger_infaas_v2_downscaling(self)
        self.record("source_infaas_control")

    def apply_ilp_solution(self,required_predictors,canary_dict,ilp_x):
        # Applying a completed control result on the event thread is synchronization,
        # not a new placement decision. Original postprocessing and routing run below.
        self.owner.control_solutions.append((required_predictors,canary_dict,ilp_x))
        with self.owner.condition:self.owner.condition.notify_all()

    def apply_completed_control(self,solution):
        Simulator.apply_ilp_solution(self,*solution)
        self.record("source_Ilp_run")

    def close(self):
        for e in self.executors.values():
            for p in e.predictors.values():
                for h in list(p.predictor_log.handlers):p.predictor_log.removeHandler(h);h.close()
        for f in self.trace_files.values():f.close()
        self.accuracy_logfile.close();self.latency_logfile.close()
        if self.solver:
            for h in list(self.solver.log.handlers):self.solver.log.removeHandler(h);h.close()
        self.inputs.cleanup()
