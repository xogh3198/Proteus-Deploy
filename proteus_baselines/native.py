"""Call unchanged original policy methods; replace simulated completions only.

Scope: one ImageNet application, one registered checkpoint, one physical GPU.
The adapter supplies only SLO/2-admitted batch sizes. It does not fabricate CPU
capacity, extra GPU replicas, offline accuracy measurements, or model variants.
"""
import hashlib
import logging
import math
from pathlib import Path

from core.common import Event, EventType
from core.executor import Executor
from core.predictor import AccType


class NativePolicy:
    def __init__(self, owner, latencies):
        self.owner = owner
        self.model_assignment = "infaas_v2" if owner.baseline == "infaas-accuracy" else "clipper"
        self.batching_algo = "infaas" if owner.baseline == "infaas-accuracy" else "aimd"
        self.allowed_batch_sizes = sorted(size for size, latency in latencies.items()
                                         if latency < owner.slo / 2)
        if not self.allowed_batch_sizes or self.allowed_batch_sizes[0] != 1:
            raise ValueError("No measured batch-1 service latency below SLO/2; profile again or change SLO")
        self.max_batch = max(self.allowed_batch_sizes)
        self.aimd_stats = {"increased": 0, "decreased": 0}
        self.batch_size_counters = {size: 0 for size in self.allowed_batch_sizes}
        self.pending = []
        self.available_predictors = [0, 0, 0, 0]  # The only GPU slot is already allocated.
        self.isi_to_idx = {owner.backend.application: 0}
        self.total_requests_arr = [0.0]
        self.infaas_slack, self.infaas_downscale_slack = 0.95, 0.4
        self.largest = {(kind, owner.backend.model_name): self.max_batch if kind == "GPU_AMPERE" else 0
                        for kind in ("CPU", "GPU_AMPERE", "VPU", "GPU_PASCAL")}
        app, name = owner.backend.application, owner.backend.model_name
        runtimes = {(app, name, size): value for size, value in latencies.items()}
        unavailable = {key: math.inf for key in runtimes}
        variant_runtimes = {1: dict(unavailable), 2: runtimes, 3: dict(unavailable), 4: dict(unavailable)}
        self.executor = Executor(app, "infaas" if self.batching_algo == "infaas" else "canary_routing",
                                 logging.WARNING, simulator=self, variant_runtimes=variant_runtimes,
                                 max_acc_per_type=1, configured_max_batch_size=self.max_batch)
        self.executor.set_model_variants({app: [name]})
        # A singleton is 100 relative to itself; this is NOT measured Top-1.
        self.executor.set_variant_accuracies({(app, name): 100.0})
        Path("logs/per_predictor").mkdir(parents=True, exist_ok=True)
        identifier = self.executor.add_predictor(AccType.GPU, variant_name=name)
        self.predictor = self.executor.predictors[identifier]
        self.controller_calls = 0

    def get_largest_batch_sizes(self):
        return self.largest

    def get_isi_from_variant_name(self, name):
        if name != self.owner.backend.model_name:
            raise ValueError("Native controller requested a variant outside singleton scope")
        return self.owner.backend.application

    def enqueue(self, job):
        event = Event(job.arrival * 1000, EventType.START_REQUEST, self.owner.backend.application,
                      deadline=job.request.slo, id=job.name)
        self.predictor.enqueue_request(event, self.owner.clock_ms())

    def generate_end_request_event(self, event, end_time, accuracy_seen, qos_met):
        # Original process_batch emits predicted END_REQUEST events. Record the
        # batch membership, but NEVER use predicted completion as an RPC result.
        self.pending.append(event)

    def generate_finish_batch_event(self, finish_time, predictor, executor):
        events, self.pending = self.pending, []
        self.owner.launch(events, nominal_finish_ms=finish_time)

    def _timer(self, when, request, predictor, kind, event_counter):
        event = Event(when, kind, request.desc, id=request.id, deadline=request.deadline,
                      predictor=predictor, executor=self.executor, event_counter=event_counter)
        self.owner.schedule(when, event)

    def generate_slo_expiring_event(self, time, request, predictor, executor, event_counter):
        self._timer(time, request, predictor, EventType.SLO_EXPIRING, event_counter)

    def generate_batch_expiring_event(self, time, request, predictor, executor, event_counter):
        self._timer(time, request, predictor, EventType.BATCH_EXPIRING, event_counter)

    def bump_failed_request_stats(self, event):
        self.owner.reject(event.id, "Original Predictor lazy-drop: queued request older than 2*SLO")
        self.predictor.finish_request(event)

    def timer(self, event):
        if event.type == EventType.SLO_EXPIRING:
            self.predictor.slo_expiring_callback(event, self.owner.clock_ms())
        else:
            self.predictor.batch_expiring_callback(event, self.owner.clock_ms())

    def finish(self, events):
        for event in events:
            self.predictor.finish_request(event)
        self.predictor.finish_batch_callback(self.owner.clock_ms())

    def control(self, rate):
        if self.batching_algo != "infaas":
            return
        self.total_requests_arr[0] = rate
        self.executor.trigger_infaas_v2_upscaling()
        self.executor.trigger_infaas_v2_downscaling()
        self.controller_calls += 1
        if len(self.executor.predictors) != 1 or self.predictor.id not in self.executor.predictors:
            raise RuntimeError("Original scaling left the declared one-GPU singleton scope")

    def close(self):
        logger = self.predictor.predictor_log
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = ("core/predictor.py", "core/executor.py", "core/common.py", "core/exceptions.py",
             "algorithms/clipper.py", "proteus_realtime/runtime.py")
    return {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in paths}
