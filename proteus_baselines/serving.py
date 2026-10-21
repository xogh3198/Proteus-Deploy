"""Wall-clock event adapter plus one real CUDA execution stream per container."""
import concurrent.futures
import hashlib
import heapq
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from . import BASELINES, DISPLAY_NAMES
from .native import NativePolicy, source_hashes
from proteus_realtime.metrics import percentile
from proteus_realtime.protocol import InferReply
from proteus_realtime.service import BusyError, Job
from proteus_realtime.pipeline import CpuStage, stage_timings

log = logging.getLogger("proteus.baselines")


class BaselineExecutor:
    def __init__(self, backend, baseline, profile_path=None, slo=200, batches=(1, 2, 4, 8),
                 max_queue=256, interval=10.0):
        if baseline not in BASELINES or len(backend.enabled) != 1:
            raise ValueError("Select a supported baseline and exactly one enabled checkpoint")
        self.backend, self.baseline = backend, baseline
        self.slo, self.batches, self.max_queue = float(slo), tuple(batches), int(max_queue)
        if not math.isfinite(self.slo) or self.slo <= 0 or not interval > 0 or max_queue < 1:
            raise ValueError("Invalid SLO, queue bound or allocation interval")
        if tuple(sorted(set(batches))) != tuple(batches) or not batches or batches[0] != 1:
            raise ValueError("Profile sorted distinct batch sizes including 1")
        self.interval, self.origin = interval, time.perf_counter()
        self.profile_path = Path(profile_path) if profile_path else None
        self.condition = threading.Condition(threading.RLock())
        self.incoming, self.completed, self.jobs = deque(), deque(), {}
        self.cpu = CpuStage(backend, self.condition, self._prepared, self._prepare_failed)
        self.timers, self.timer_seq = [], 0
        self.native, self.profile_data = None, None
        self.mode = "native"
        self.busy = self.maintenance = self.stopping = False
        self.processed = self.failed = self.dropped = self.arrivals = self.server_slo_misses = 0
        self.policy_errors = 0
        self.batch_counts = {}
        self.native_arrivals = self.control_count = 0
        self.control_clock = 0.0
        self.error = "Run register to measure actual GPU batch profiles"
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="baseline-cuda")
        self._read_profile()
        self.thread = threading.Thread(target=self._loop, name="baseline-policy", daemon=True)
        self.thread.start()

    def clock_ms(self):
        return (time.perf_counter() - self.origin) * 1000

    def _read_profile(self):
        if not self.profile_path or not self.profile_path.exists():
            return
        row = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if row.get("fingerprint") != self.backend.fingerprint() or row.get("slo_ms") != self.slo:
            self.error = "Profile fingerprint/SLO mismatch; run register again"
            return
        if set(row.get("batches", {})) != {str(size) for size in self.batches}:
            self.error = "Profile batch candidates changed; run register again"
            return
        self.profile_data = row
        self._install_native()

    def _install_native(self):
        latencies = {int(size): float(row["p50_ms"]) for size, row in self.profile_data["batches"].items()}
        if set(latencies) != set(self.batches) or any(not math.isfinite(v) or v <= 0 for v in latencies.values()):
            raise ValueError("Every configured size needs a finite positive actual GPU profile")
        replacement = NativePolicy(self, latencies)
        if self.native:
            self.native.close()
        self.native = replacement
        self.timers.clear()
        self.control_count, self.control_clock = self.native_arrivals, self.clock_ms()
        self.error = None

    def submit(self, request):
        if request.model not in {self.backend.application, self.backend.model_name, self.backend.model_id}:
            raise ValueError("Use the singleton application/model name")
        if request.dispatch not in {"", self.baseline}:
            raise ValueError("Dispatch must be empty or this independent baseline's name")
        if not request.image or len(request.image) > 16 * 1024 * 1024:
            raise ValueError("Raw image bytes required, at most 16 MiB")
        if not math.isfinite(request.slo) or abs(request.slo - self.slo) > 1e-6:
            raise ValueError("Request SLO must equal the configured/profiled SLO")
        now = time.perf_counter()
        job = Job(request, concurrent.futures.Future(), now - self.origin, time.time() * 1000, uuid.uuid4().hex)
        with self.condition:
            if self.maintenance or self.stopping:
                raise BusyError("Baseline executor in maintenance/shutdown")
            if self.mode == "native" and self.native is None:
                raise ValueError("Policy not ready: " + str(self.error))
            if len(self.jobs) >= self.max_queue:
                raise BusyError("Independent baseline's bounded queue is full")
            self.jobs[job.name] = job
            self.cpu.submit(job)
            self.arrivals += 1
            self.native_arrivals += int(self.mode == "native")
            self.condition.notify_all()
        return job.future

    def _prepared(self, job):
        if self.stopping or job.future.cancelled():
            self.jobs.pop(job.name, None)
            job.future.cancel()
        else:
            self.incoming.append(job)

    def _prepare_failed(self, job, error):
        self.jobs.pop(job.name, None)
        if not job.future.done():
            self.failed += 1
            job.future.set_exception(error)

    def schedule(self, when, event):
        self.timer_seq += 1
        heapq.heappush(self.timers, (float(when), self.timer_seq, event))

    def reject(self, identifier, reason, kind="legacy_lazy_drop"):
        job = self.jobs.pop(identifier, None)
        if job:
            self.dropped += int(kind == "legacy_lazy_drop")
            if not job.future.done():
                job.future.set_result(InferReply(request_name=job.name, rejected=True,
                    reject_kind=kind, reason=reason, gpu=self.backend.gpu_type,
                    latency=self.clock_ms() - job.arrival * 1000))

    def launch(self, events, nominal_finish_ms=None):
        if self.busy:
            raise RuntimeError("Original policy attempted overlapping GPU batches")
        jobs = [self.jobs[e.id] for e in events if e.id in self.jobs]
        if not jobs:
            raise RuntimeError("Original batch contains no outstanding real requests")
        self.busy = True
        for job in jobs:
            job.future.set_running_or_notify_cancel()
        mode = self.mode
        future = self.pool.submit(self._gpu, jobs, mode)
        def done(result):
            with self.condition:
                self.completed.append((events, jobs, result))
                self.condition.notify_all()
        future.add_done_callback(done)

    def _gpu(self, jobs, mode):
        started = time.perf_counter()
        loaded = self.backend.load(self.backend.model_name)
        service_start = time.perf_counter()
        results = self.backend.infer_prepared([job.prepared for job in jobs])
        finished = time.perf_counter()
        if len(results) != len(jobs):
            raise RuntimeError("Real GPU batch reply count mismatch")
        replies = []
        for job, result in zip(jobs, results):
            latency = (finished - self.origin - job.arrival) * 1000
            timings = {"worker." + key: float(value) for key, value in result["timings"].items()}
            timings.update({"worker.queue": (started - self.origin - job.arrival) * 1000,
                "worker.load": loaded["load_ms"], "worker.ready": loaded["ready_ms"],
                "worker.service": (finished - service_start) * 1000,
                "worker.model_loaded": float(loaded["loaded"]), "worker.batch_size": float(len(jobs)),
                "server.slo_missed": float(latency > self.slo)})
            timings.update(stage_timings(job, started, (finished - service_start) * 1000, self.origin))
            timings.update({"worker." + key: value for key, value in loaded.get("stages", {}).items()})
            width, height = result["image_size"]
            replies.append(InferReply(request_name=job.name, inference=result["label"],
                latency=latency, gpu=self.backend.gpu_type, variant=self.backend.model_name,
                worker_ms=(finished - started) * 1000, img_width=width, img_height=height,
                timings=timings, arrive_epoch_ms=job.arrival_epoch_ms, reply_epoch_ms=time.time() * 1000))
        log.info("GPU baseline=%s mode=%s actual_batch=%d load=%.2f service=%.2f",
                 self.baseline, mode, len(jobs), loaded["load_ms"], (finished - service_start) * 1000)
        return replies

    def _complete(self):
        while self.completed:
            events, jobs, result = self.completed.popleft()
            self.busy = False
            try:
                replies = result.result()
                self.processed += len(jobs)
                self.batch_counts[str(len(jobs))] = self.batch_counts.get(str(len(jobs)), 0) + 1
                self.server_slo_misses += sum(reply.latency > self.slo for reply in replies)
                for job, reply in zip(jobs, replies):
                    self.jobs.pop(job.name, None)
                    if not job.future.done():
                        job.future.set_result(reply)
            except Exception as error:
                self.failed += len(jobs)
                for job in jobs:
                    self.jobs.pop(job.name, None)
                    if not job.future.done():
                        job.future.set_exception(error)
            if self.mode == "native" and self.native is not None and not self.stopping:
                self.native.finish(events)
            self.condition.notify_all()

    def _loop(self):
        while True:
            with self.condition:
                try:
                    self._complete()
                    if self.stopping:
                        for identifier in list(self.jobs):
                            if identifier not in {j.name for _, js, _ in self.completed for j in js}:
                                # close() is normally entered after RPCs drain.
                                self.reject(identifier, "Baseline server stopping", "shutdown")
                        if not self.busy:
                            return
                        self.condition.wait(0.05)
                        continue
                    if self.maintenance:
                        self.condition.wait(0.05)
                        continue
                    if self.mode == "fixed":
                        if self.incoming and not self.busy:
                            job = self.incoming.popleft()
                            self.launch([type("FixedEvent", (), {"id": job.name})()])
                    else:
                        while self.incoming:
                            self.native.enqueue(self.incoming.popleft())
                        now = self.clock_ms()
                        while self.timers and self.timers[0][0] <= now:
                            _, _, event = heapq.heappop(self.timers)
                            self.native.timer(event)
                            now = self.clock_ms()
                        if self.native:
                            # Liveness watchdog only invokes the original >2*SLO
                            # lazy-drop predicate; it does not invent partial batches.
                            self.native.predictor.drop_expired_requests(now)
                            if now - self.control_clock >= self.interval * 1000:
                                rate = (self.native_arrivals - self.control_count) * 1000 / (now - self.control_clock)
                                self.native.control(rate)
                                self.control_count, self.control_clock = self.native_arrivals, now
                    delay = 0.01
                    if self.timers:
                        delay = min(delay, max(0.001, (self.timers[0][0] - self.clock_ms()) / 1000))
                    self.condition.wait(delay)
                except Exception as error:
                    log.exception("Native policy adapter failed")
                    self.error = str(error)
                    self.policy_errors += 1
                    for identifier in list(self.jobs):
                        self.reject(identifier, "Policy adapter failed: " + str(error), "policy_adapter_error")
                    self.incoming.clear()
                    self.timers.clear()
                    if self.native:
                        self.native.close()
                    self.native = None
                    self.condition.wait(0.05)

    def _exclusive(self, function):
        with self.condition:
            if self.jobs or self.busy or self.maintenance or self.stopping:
                raise BusyError("Wait until this baseline's real GPU queue is idle")
            self.maintenance = True
        try:
            return self.pool.submit(function).result()
        finally:
            with self.condition:
                self.maintenance = False
                self.condition.notify_all()

    def profile(self, image, trials=30):
        if not image or not isinstance(trials, int) or not 1 <= trials <= 1000:
            raise ValueError("Raw JPEG and trials in 1..1000 required")
        def measure():
            self.backend.load()
            cpu_start = time.perf_counter()
            prepared = self.cpu.prepare_once(image)
            cpu_ms = (time.perf_counter() - cpu_start) * 1000
            batches = {}
            for size in self.batches:
                for _ in range(3):
                    self.backend.infer_prepared([prepared] * size)
                samples = []
                for _ in range(trials):
                    start = time.perf_counter()
                    self.backend.infer_prepared([prepared] * size)
                    samples.append((time.perf_counter() - start) * 1000)
                batches[str(size)] = {"p50_ms": percentile(samples, 50), "p95_ms": percentile(samples, 95),
                                      "samples_ms": samples}
            previous_profile = self.profile_data
            self.profile_data = {"schema": "baseline-cpu-ready-v1", "fingerprint": self.backend.fingerprint(),
                "profile_scope": "prepared-batch-service", "cpu_preparation_ms": cpu_ms,
                "slo_ms": self.slo, "model": self.backend.model_name, "batches": batches,
                "input_sha256": hashlib.sha256(image).hexdigest()}
            try:
                self._install_native()
            except Exception:
                self.profile_data = previous_profile
                raise
            if self.profile_path:
                self.profile_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.profile_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.profile_data, indent=2, allow_nan=False), encoding="utf-8")
                os.replace(temporary, self.profile_path)
            return {"profile": self.profile_data, "policy_ready": True, "max_batch": self.native.max_batch}
        return self._exclusive(measure)

    def set_mode(self, mode):
        if mode not in {"fixed", "native"}:
            raise ValueError("Mode must be fixed diagnostic or this container's native baseline")
        def change():
            if mode == "native":
                if self.profile_data is None:
                    raise ValueError("Run register first")
                self._install_native()
            self.mode = mode
            return {"mode": mode, "baseline": self.baseline}
        return self._exclusive(change)

    def reset(self):
        return self._exclusive(lambda: (self.backend.reset() or {
            "ok": True, "cold_definition": "model non-resident; OS/CUDA caches NOT cleared"}))

    def evaluate(self, *_):
        raise ValueError("Repeated-image singleton baseline test does not measure classification accuracy")

    def state(self):
        with self.condition:
            native = self.native
            return {**self.backend.describe(), "implementation": "proteus-original-baseline-single-gpu-adapter",
                "baseline": self.baseline, "display_name": DISPLAY_NAMES[self.baseline], "mode": self.mode,
                "policy_ready": native is not None, "policy_error": self.error,
                "batching": (native.batching_algo if native else "unprofiled") if self.mode == "native" else "fixed1-diagnostic",
                "max_batch": native.max_batch if native else None,
                "aimd_batch_size": native.predictor.aimd_batch_size if native else None,
                "infaas_batch_size": native.predictor.infaas_batch_size if native else None,
                "infaas_controller_calls": native.controller_calls if native else 0,
                "source_hashes": source_hashes(), "profile": self.profile_data,
                "actual_batch_counts": dict(self.batch_counts), "queue_depth": len(self.jobs),
                "cpu_queue_depth": len(self.cpu.pending),
                "ready_queue_depth": len(self.incoming) + (len(native.predictor.request_queue) if native else 0),
                "preprocess_workers": self.cpu.workers, "pipeline": "cpu-ready-v1",
                "profile_scope": "prepared-batch-service", "cpu_gpu_pipeline": True,
                "busy": self.busy, "processed": self.processed, "failed": self.failed,
                "policy_errors": self.policy_errors,
                "deadline_drops": self.dropped, "server_slo_misses": self.server_slo_misses,
                "slo_ms": self.slo, "classification_accuracy": None, "normalized_accuracy": 100.0,
                "infaas_accuracy_cost": 0.0, "model_scaling_enabled": False, "replication_enabled": False,
                "scope_note": "One checkpoint: HA=HT model choice; 100 is relative normalization, not measured accuracy. "
                    "Only SLO/2-admitted sizes supplied to original policy; original predicted feedback retained. "
                    "Original >2*SLO drop watchdog guarantees bounded waiting; original AIMD expiry callback remains a no-op."}

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.cpu.close()
        self.thread.join(timeout=180)
        self.pool.shutdown(wait=True)
        if self.thread.is_alive():
            raise RuntimeError("Baseline policy thread did not stop")
        if self.native:
            self.native.close()
        self.backend.reset()
