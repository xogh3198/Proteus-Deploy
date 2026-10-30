"""CPU preparation ahead of the policy queue, preserving admission order.

All state transitions use the owner's condition. No CUDA work is performed by
this stage. Admission is bounded by the owner's existing queue semantics:
Proteus counts CPU/ready waiting work, baselines count all outstanding work.
The pool therefore cannot bypass either owner's admission limit.
"""
import concurrent.futures
import os
import time
from collections import deque


class CpuStage:
    def __init__(self, backend, condition, ready, failed, before_prepare=None):
        self.backend, self.condition = backend, condition
        self.ready, self.failed = ready, failed
        self.pending, self.completed = deque(), {}
        self.pending_names, self.active, self.tasks = set(), set(), {}
        self.before_prepare = before_prepare
        workers = int(os.environ.get("PREPROCESS_WORKERS", "1"))
        if not 1 <= workers <= 32:
            raise ValueError("PREPROCESS_WORKERS must be in 1..32")
        self.workers = workers
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                         thread_name_prefix="cpu-preprocess")

    def submit(self, job):
        # Called with the owner's condition held.
        self.pending.append(job)
        self.pending_names.add(job.name)
        # The owner holds condition, so the task cannot race this assignment.
        self.tasks[job.name] = self.pool.submit(self._prepare, job)

    def discard(self, job):
        """Owner-condition held: detach expired work, cancel queued CPU work.

        An already-running CPU call is not interrupted. Its eventual output is
        discarded; active continues to protect maintenance until it returns.
        """
        self.pending_names.discard(job.name)
        self.pending = deque(item for item in self.pending if item.name != job.name)
        self.completed.pop(job.name, None)
        task = self.tasks.pop(job.name, None)
        if task is not None:
            task.cancel()

    def promote(self):
        """Promote surviving completions in original FIFO admission order."""
        while self.pending and self.pending[0].name in self.completed:
            item = self.pending.popleft()
            self.pending_names.discard(item.name)
            failure = self.completed.pop(item.name)
            if failure is not None:
                self.failed(item, failure)
            else:
                self.ready(item)

    def _prepare(self, job):
        error = None
        try:
            with self.condition:
                if job.name not in self.pending_names or job.future.done():
                    raise concurrent.futures.CancelledError()
                if self.before_prepare is not None and not self.before_prepare(job):
                    raise concurrent.futures.CancelledError()
                job.cpu_started = time.perf_counter()
                self.active.add(job.name)
            job.prepared = self.backend.prepare(job.request.image)
        except Exception as caught:
            error = caught
        finally:
            finished = time.perf_counter() if job.cpu_started else 0.0
        with self.condition:
            job.cpu_finished = finished
            self.active.discard(job.name)
            self.tasks.pop(job.name, None)
            if job.name in self.pending_names:
                self.completed[job.name] = error
                self.promote()
            else:
                # Deadline/cancellation already resolved the client future.
                job.prepared = None
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
