"""CPU preparation ahead of the policy queue, preserving admission order.

All state transitions use the owner's condition. No CUDA work is performed by
this stage. The owner's outstanding-request bound covers preparation + ready
queue + running GPU work, so the pool's internal queue cannot grow unbounded.
"""
import concurrent.futures
import os
import time
from collections import deque


class CpuStage:
    def __init__(self, backend, condition, ready, failed):
        self.backend, self.condition = backend, condition
        self.ready, self.failed = ready, failed
        self.pending, self.completed = deque(), {}
        workers = int(os.environ.get("PREPROCESS_WORKERS", "1"))
        if not 1 <= workers <= 32:
            raise ValueError("PREPROCESS_WORKERS must be in 1..32")
        self.workers = workers
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                         thread_name_prefix="cpu-preprocess")

    def submit(self, job):
        # Called with the owner's condition held.
        self.pending.append(job)
        self.pool.submit(self._prepare, job)

    def _prepare(self, job):
        error = None
        try:
            if job.future.cancelled():
                raise concurrent.futures.CancelledError()
            job.cpu_started = time.perf_counter()
            job.prepared = self.backend.prepare(job.request.image)
        except Exception as caught:
            error = caught
        finally:
            job.cpu_finished = time.perf_counter()
        with self.condition:
            self.completed[job.name] = error
            # Multiple CPU workers must not reorder original FIFO admissions.
            while self.pending and self.pending[0].name in self.completed:
                item = self.pending.popleft()
                failure = self.completed.pop(item.name)
                if failure is not None:
                    self.failed(item, failure)
                else:
                    self.ready(item)
            self.condition.notify_all()

    def close(self):
        self.pool.shutdown(wait=True)

    def prepare_once(self, image):
        """Administrative preparation, only while owner is in maintenance."""
        return self.pool.submit(self.backend.prepare, image).result()


def stage_timings(job, gpu_started, gpu_service_ms, origin=0.0):
    cpu_ms = (job.cpu_finished - job.cpu_started) * 1000
    cpu_queue_ms = (job.cpu_started - origin - job.arrival) * 1000
    ready_queue_ms = (gpu_started - job.cpu_finished) * 1000
    return {"worker.cpu_queue": cpu_queue_ms, "worker.cpu_service": cpu_ms,
            "worker.ready_queue": ready_queue_ms, "worker.gpu_service": gpu_service_ms,
            "worker.queue": cpu_queue_ms + ready_queue_ms,
            "worker.service": cpu_ms + gpu_service_ms}
