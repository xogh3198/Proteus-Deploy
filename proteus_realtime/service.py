"""Bounded FIFO batch-1 executor with explicit scope and measured timestamps."""
import concurrent.futures
import logging
import math
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

from .protocol import InferReply

log = logging.getLogger("proteus.bootstrap")


class BusyError(RuntimeError):
    pass


@dataclass
class Job:
    request: object
    future: concurrent.futures.Future
    arrival: float
    arrival_epoch_ms: float
    name: str
    prepared: object = None
    cpu_started: float = 0.0
    cpu_finished: float = 0.0


class Executor:
    """A single GPU execution stream; administration cannot race model resets."""
    def __init__(self, backend, max_queue=64):
        self.backend = backend
        self.queue = queue.Queue(maxsize=max_queue)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.outstanding = {}
        self.operations = deque()
        from .pipeline import CpuStage
        self.cpu = CpuStage(backend, self.condition, self._prepared, self._prepare_failed)
        self.busy = False
        self.maintenance = False
        self.stopping = False
        self.processed = 0
        self.failed = 0
        self.server_slo_misses = 0
        self.thread = threading.Thread(target=self._work, name="proteus-fifo", daemon=True)
        self.thread.start()

    def submit(self, request):
        if not math.isfinite(request.slo) or request.slo <= 0:
            raise ValueError("slo must be a finite positive number of milliseconds")
        if request.model not in {self.backend.model_name, self.backend.model_id}:
            raise ValueError("Only the configured fixed model is supported in this bootstrap")
        if not request.image:
            raise ValueError("image must contain raw JPEG/image bytes")
        if len(request.image) > 16 * 1024 * 1024:
            raise ValueError("image exceeds the 16 MiB limit")
        if request.dispatch not in {"", "fixed"}:
            raise ValueError("Only fixed dispatch is supported; no MILP/predictor routing is active")
        job = Job(request, concurrent.futures.Future(), time.perf_counter(),
                  time.time() * 1000.0, uuid.uuid4().hex)
        with self.lock:
            if self.maintenance or self.stopping:
                raise BusyError("Executor is in maintenance/shutdown")
            if len(self.outstanding) >= self.queue.maxsize:
                raise BusyError("Request queue is full")
            self.outstanding[job.name] = job
            self.cpu.submit(job)
        return job.future

    def _prepared(self, job):
        if self.stopping or job.future.cancelled():
            self.outstanding.pop(job.name, None)
            job.future.cancel()
        else:
            self.queue.put_nowait(job)

    def _prepare_failed(self, job, error):
        self.outstanding.pop(job.name, None)
        if not job.future.done():
            self.failed += 1
            job.future.set_exception(error)

    def _work(self):
        while True:
            # Take a job under the same lock as administrative admission so an
            # empty queue cannot hide a dequeued-but-not-marked-busy job.
            with self.lock:
                operation = self.operations.popleft() if self.operations else None
                try:
                    job = self.queue.get_nowait() if operation is None else None
                except queue.Empty:
                    job = None
                if job is not None:
                    self.busy = True
                if self.stopping and job is None and operation is None:
                    return
            if operation is not None:
                function, future = operation
                try:
                    future.set_result(function())
                except BaseException as error:
                    future.set_exception(error)
                continue
            if job is None:
                time.sleep(0.001)
                continue
            reply = None
            failure = None
            try:
                if not job.future.set_running_or_notify_cancel():
                    continue
                started = time.perf_counter()
                loaded = self.backend.load()
                infer_started = time.perf_counter()
                result = self.backend.infer_prepared([job.prepared])[0]
                finished = time.perf_counter()
                timings = {f"worker.{key}": float(value) for key, value in result["timings"].items()}
                timings.update({
                    "worker.queue": (started - job.arrival) * 1000,
                    "worker.load": loaded["load_ms"],
                    "worker.ready": loaded["ready_ms"],
                    "worker.service": (finished - infer_started) * 1000,
                    "worker.batch_size": 1.0,
                    "worker.model_loaded": float(loaded["loaded"]),
                })
                from .pipeline import stage_timings
                timings.update(stage_timings(job, started, (finished - infer_started) * 1000))
                server_ms = (finished - job.arrival) * 1000
                timings["server.slo_missed"] = float(server_ms > job.request.slo)
                width, height = result["image_size"]
                reply = InferReply(
                    request_name=job.name, inference=result["label"], latency=server_ms,
                    gpu=self.backend.gpu_type, worker_ms=(finished - started) * 1000,
                    img_width=width, img_height=height, timings=timings,
                    arrive_epoch_ms=job.arrival_epoch_ms, reply_epoch_ms=time.time() * 1000)
                for x1, y1, x2, y2, label, score in result["boxes"]:
                    reply.detections.add(x1=x1, y1=y1, x2=x2, y2=y2, label=label, score=score)
                with self.lock:
                    self.processed += 1
                    self.server_slo_misses += int(server_ms > job.request.slo)
                log.info("request=%s server=%.1fms load=%.1fms resident=%s label=%r slo_miss=%s",
                         job.name, server_ms, loaded["load_ms"], not loaded["loaded"],
                         result["label"], server_ms > job.request.slo)
            except Exception as error:
                with self.lock:
                    self.failed += 1
                log.exception("Request failed: %s", job.name)
                failure = error
            finally:
                with self.lock:
                    self.outstanding.pop(job.name, None)
                    self.busy = False
                self.queue.task_done()
            # Complete the client future only after queue/maintenance bookkeeping.
            if failure is not None:
                job.future.set_exception(failure)
            elif reply is not None:
                job.future.set_result(reply)

    def state(self):
        with self.lock:
            counters = {"queue_depth": len(self.outstanding), "ready_queue_depth": self.queue.qsize(),
                        "cpu_queue_depth": len(self.cpu.pending), "pipeline": "cpu-ready-v1", "busy": self.busy,
                        "maintenance": self.maintenance, "processed": self.processed,
                        "failed": self.failed, "server_slo_misses": self.server_slo_misses}
        return {**self.backend.describe(), **counters,
                "implementation": "real-gpu-bootstrap", "batching": "fixed1",
                "batch_size": 1, "milp_enabled": False, "accuracy_scaling_enabled": False,
                "accuracy_measured": False, "preprocessing_loading_overlap": True}

    def _exclusive(self, operation):
        with self.lock:
            if self.busy or self.outstanding or self.maintenance or self.stopping:
                raise BusyError("Wait for all requests to finish before reset/profile")
            self.maintenance = True
            future = concurrent.futures.Future()
            self.operations.append((operation, future))
        try:
            return future.result()
        finally:
            with self.lock:
                self.maintenance = False

    def reset(self):
        def operation():
            self.backend.reset()
            return {"ok": True, "cold_definition": "model non-resident; OS/CUDA caches NOT cleared"}
        return self._exclusive(operation)

    def profile(self, image, trials=30):
        if not isinstance(trials, int) or not 1 <= trials <= 1000:
            raise ValueError("trials must be an integer in 1..1000")
        if not image or len(image) > 16 * 1024 * 1024:
            raise ValueError("profile requires a nonempty image up to 16 MiB")
        def operation():
            load = self.backend.load()
            prepared = self.cpu.prepare_once(image)
            samples = []
            stages = {}
            for _ in range(trials):
                started = time.perf_counter()
                result = self.backend.infer_prepared([prepared])[0]
                samples.append((time.perf_counter() - started) * 1000)
                for key, value in result["timings"].items():
                    stages.setdefault(key, []).append(float(value))
            from .metrics import percentile
            return {"scope": "prepared batch-1 GPU service profile; NOT MILP registration",
                    "batch_size": 1, "trials": trials, "load_ms": load["load_ms"],
                    "mean_service_ms": sum(samples) / len(samples),
                    "p50_service_ms": percentile(samples, 50),
                    "p95_service_ms": percentile(samples, 95),
                    "reciprocal_mean_service_qps": 1000 / (sum(samples) / len(samples)),
                    "qps_warning": "Reciprocal service time is NOT measured end-to-end throughput",
                    "mean_stage_ms": {key: sum(values) / len(values) for key, values in stages.items()},
                    "accuracy": None, "runtime": self.backend.describe()}
        return self._exclusive(operation)

    def close(self):
        with self.lock:
            self.stopping = True
        self.cpu.close()
        self.thread.join(timeout=180)
        if not self.thread.is_alive():
            self.backend.reset()
