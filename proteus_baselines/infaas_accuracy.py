"""Multi-candidate, single-physical-GPU bridge to unchanged INFaaS-v2 code.

No SLO-violation threshold, MILP, accuracy-maximizing substitute policy, or
additional GPU replica is introduced. Controls run at the configured interval;
a decision due during inference is applied after that batch completes.
"""
from collections import Counter, deque
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time

from core.executor import Executor
from core.predictor import AccType
from .native import NativePolicy
from .serving import BaselineExecutor
from proteus_realtime.metrics import percentile


def close_predictor(predictor):
    for handler in list(predictor.predictor_log.handlers):
        predictor.predictor_log.removeHandler(handler)
        handler.close()


class MultiAccuracyPolicy(NativePolicy):
    def __init__(self, owner, latencies):
        self.owner = owner
        self.model_assignment, self.batching_algo = "infaas_v2", "infaas"
        names = list(owner.backend.enabled)
        app = owner.backend.application
        admitted = {name: [b for b, value in rows.items() if value < owner.slo / 2]
                    for name, rows in latencies.items()}
        if any(not sizes or 1 not in sizes for sizes in admitted.values()):
            raise ValueError("Every enabled candidate needs measured batch-1 latency below SLO/2")
        self.allowed_batch_sizes = sorted({b for sizes in admitted.values() for b in sizes})
        self.max_batch = max(self.allowed_batch_sizes)
        self.largest = {(kind, name): max(admitted[name]) if kind == "GPU_AMPERE" else 0
                        for name in names for kind in ("CPU", "GPU_AMPERE", "VPU", "GPU_PASCAL")}
        self.aimd_stats = {"increased": 0, "decreased": 0}
        self.batch_size_counters = {b: 0 for b in self.allowed_batch_sizes}
        self.pending, self.pending_rate = [], None
        self.available_predictors = [0, 0, 0, 0]
        self.isi_to_idx, self.total_requests_arr = {app: 0}, [0.0]
        self.infaas_slack, self.infaas_downscale_slack = .95, .4
        runtimes = {(app, name, b): value for name, rows in latencies.items() for b, value in rows.items()}
        unavailable = {key: math.inf for key in runtimes}
        self.executor = Executor(app, "infaas", logging.WARNING, simulator=self,
            variant_runtimes={1: dict(unavailable), 2: runtimes, 3: dict(unavailable), 4: dict(unavailable)},
            max_acc_per_type=1, configured_max_batch_size=self.max_batch)
        self.executor.set_model_variants({app: names})
        self.executor.set_variant_accuracies({(app, name): float(owner.backend.registry[name]["accuracy"])
                                              for name in names})
        Path("logs/per_predictor").mkdir(parents=True, exist_ok=True)
        # Keep the original Executor's minimum-accuracy startup rule.
        identifier = self.executor.add_predictor(AccType.GPU)
        self.predictor = self.executor.predictors[identifier]
        self.controller_calls = self.deferred_controls = 0
        self.selections = deque(maxlen=512)
        self._record("original_add_predictor")

    def _record(self, reason):
        self.selections.append({"epoch_seconds": time.time(), "reason": reason,
            "model": self.predictor.variant_name, "batch": self.predictor.infaas_batch_size,
            "accuracy_loss_cost": self.predictor.infaas_cost,
            "demand_qps": self.total_requests_arr[0]})

    def get_isi_from_variant_name(self, name):
        if name not in self.owner.backend.enabled:
            raise ValueError("Native controller requested an unregistered model")
        return self.owner.backend.application

    def generate_finish_batch_event(self, finish_time, predictor, executor):
        events, self.pending = self.pending, []
        self.owner.launch(events, nominal_finish_ms=finish_time, model_name=predictor.variant_name)

    def timer(self, event):
        if event.predictor is self.predictor:
            super().timer(event)

    def control(self, rate):
        if self.owner.busy:
            self.pending_rate = rate
            self.deferred_controls += 1
            return
        self._control_now(rate)

    def _control_now(self, rate):
        old = self.predictor
        waiting = list(old.request_queue)
        self.total_requests_arr[0] = rate
        self.executor.trigger_infaas_v2_upscaling()
        self.executor.trigger_infaas_v2_downscaling()
        self.controller_calls += 1
        if len(self.executor.predictors) != 1:
            raise RuntimeError("Original controller requested more than the one physical GPU slot")
        replacement = next(iter(self.executor.predictors.values()))
        if replacement.acc_type != AccType.GPU.value:
            raise RuntimeError("CPU inference/additional hardware is unavailable in this experiment")
        self.predictor = replacement
        self._record("original_infaas_v2_upscaling_then_downscaling")
        if replacement is not old:
            # Discard timers referencing the removed logical predictor. Preserve
            # request identity and original arrival/deadline when re-enqueuing.
            self.owner.timers.clear()
            close_predictor(old)
            for event in waiting:
                if event.id in self.owner.jobs:
                    replacement.enqueue_request(event, self.owner.clock_ms())
            return True
        return False

    def finish(self, events):
        old = self.predictor
        for event in events:
            old.finish_request(event)
        if self.pending_rate is not None:
            rate, self.pending_rate = self.pending_rate, None
            if self._control_now(rate):
                return
        old.finish_batch_callback(self.owner.clock_ms())


class AccuracyExecutor(BaselineExecutor):
    supports_multiple_models = True

    def __init__(self, backend, baseline="infaas-accuracy", **kwargs):
        if baseline != "infaas-accuracy":
            raise ValueError("Multi-model bridge is only for INFaaS-Accuracy")
        self.model_counts = Counter()
        self.switch_events = deque(maxlen=512)
        self.serving_model = None
        self.actual_switches = 0
        super().__init__(backend, baseline, **kwargs)

    def _read_profile(self):
        if not self.profile_path or not self.profile_path.exists():
            return
        row = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if (row.get("fingerprint") != self.backend.fingerprint() or row.get("slo_ms") != self.slo
                or set(row.get("models", {})) != set(self.backend.enabled)
                or any(set(p["batches"]) != {str(b) for b in self.batches} for p in row.get("models", {}).values())):
            self.error = "Multi-model profile fingerprint/SLO/candidates changed; run register again"
            return
        self.profile_data = row
        self._install_native()

    def _install_native(self):
        latencies = {name: {int(b): float(row["p50_ms"]) for b, row in profile["batches"].items()}
                     for name, profile in self.profile_data["models"].items()}
        if any(not math.isfinite(value) or value <= 0 for rows in latencies.values() for value in rows.values()):
            raise ValueError("Actual finite positive model/batch profiles required")
        replacement = MultiAccuracyPolicy(self, latencies)
        if self.native:
            self.native.close()
        self.native = replacement
        self.timers.clear()
        self.control_count, self.control_clock = self.native_arrivals, self.clock_ms()
        self.error = None

    def profile(self, image, trials=30):
        if not image or not isinstance(trials, int) or not 1 <= trials <= 1000:
            raise ValueError("Raw JPEG and trials in 1..1000 required")
        def measure():
            profiles = {}
            for name in self.backend.enabled:
                load = self.backend.load(name)
                before = time.perf_counter()
                prepared = self.cpu.prepare_once(image)
                cpu_ms = (time.perf_counter() - before) * 1000
                batches = {}
                for size in self.batches:
                    for _ in range(3):
                        self.backend.infer_prepared([prepared] * size)
                    samples = []
                    for _ in range(trials):
                        start = time.perf_counter()
                        self.backend.infer_prepared([prepared] * size)
                        samples.append((time.perf_counter() - start) * 1000)
                    batches[str(size)] = {"p50_ms": percentile(samples, 50),
                        "p95_ms": percentile(samples, 95), "samples_ms": samples}
                profiles[name] = {"batches": batches, "load": load, "cpu_preparation_ms": cpu_ms}
            previous = self.profile_data
            self.profile_data = {"schema": "infaas-accuracy-multi-gpu-v1", "models": profiles,
                "fingerprint": self.backend.fingerprint(), "slo_ms": self.slo,
                "profile_scope": "prepared-batch-service", "input_sha256": hashlib.sha256(image).hexdigest()}
            try:
                self._install_native()
                if self.profile_path:
                    self.profile_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.profile_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(self.profile_data, indent=2, allow_nan=False), encoding="utf-8")
                    os.replace(temporary, self.profile_path)
                self.backend.load(self.native.predictor.variant_name)
            except Exception:
                self.profile_data = previous
                raise
            return {"profile": self.profile_data, "policy_ready": True,
                    "selected_model": self.native.predictor.variant_name, "max_batch": self.native.max_batch}
        return self._exclusive(measure)

    def _gpu(self, jobs, mode, model_name=None):
        selected = model_name or self.backend.model_name
        replies = super()._gpu(jobs, mode, selected)
        measured = self.profile_data["models"][selected]["batches"].get(str(len(jobs)))
        if measured:
            for reply in replies:
                reply.timings["worker.predicted_gpu_service"] = measured["p50_ms"]
                reply.timings["worker.gpu_prediction_error"] = reply.timings["worker.gpu_service"] - measured["p50_ms"]
        with self.condition:
            self.serving_model = selected
            self.model_counts[selected] += len(jobs)
        return replies

    def _model_loaded(self, selected, previous, loaded, count):
        with self.condition:
            if previous is not None and previous != selected:
                self.actual_switches += 1
            if loaded["loaded"]:
                self.switch_events.append({"epoch_seconds": time.time(), "from": previous,
                    "to": selected, "load_ms": loaded["load_ms"], "requests": count})

    def state(self):
        state = super().state()
        native = self.native
        name = native.predictor.variant_name if native else None
        return {**state, "model_scaling_enabled": True,
            "selected_model": name, "resident_model": getattr(self.backend, "current", None),
            "model_switches_after_initial_load": self.actual_switches,
            "model_switch_events": list(self.switch_events), "successful_requests_by_model": dict(self.model_counts),
            "model_selection_history": list(native.selections) if native else [],
            "deferred_controls": native.deferred_controls if native else 0,
            "infaas_accuracy_cost": native.predictor.infaas_cost if native else None,
            "normalized_accuracy": (100 * self.backend.registry[name]["accuracy"] /
                max(self.backend.registry[n]["accuracy"] for n in self.backend.enabled)) if name else None,
            "selection_rule": "unchanged core.executor INFaaS-v2; no SLO-miss feedback added",
            "scope_note": "One physical GPU, multiple profiled checkpoints; original startup and downscale "
                "minimum-accuracy behavior retained. CPU preparation stays ahead of the native queue. "
                "Controls due during a GPU batch are applied at its completion; no extra replica is fabricated."}
