"""Independent real-GPU Proteus port: MILP controller + deadline batch worker."""
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from .metrics import percentile
from .policy import adaptive_decision, solve_milp
from .protocol import InferReply
from .registry import normalized_accuracies
from .service import BusyError, Job

log = logging.getLogger("proteus.serving")


class ProteusExecutor:
    def __init__(self, backend, max_queue=64, profile_path=None, slo=200,
                 batches=(1, 2, 4, 8), interval=1.0, guard_ms=2.0, solver=solve_milp):
        self.backend = backend
        self.max_queue, self.slo = max_queue, float(slo)
        self.batches = tuple(sorted(set(batches)))
        if not self.batches or self.batches[0] != 1 or any(b < 1 for b in self.batches):
            raise ValueError("BATCH_SIZES must include 1 and contain positive sizes")
        self.interval, self.guard_ms, self.solver = interval, guard_ms, solver
        self.condition = threading.Condition(threading.RLock())
        self.queue = deque()
        self.busy = self.maintenance = self.stopping = False
        self.mode = "proteus"
        self.processed = self.failed = self.dropped = self.arrivals = self.server_slo_misses = 0
        self.batch_counts = {}
        self.plan = None
        self.profile_path = Path(profile_path) if profile_path else None
        self.profiles, self.measured_accuracy = {}, {}
        self.policy_error = "Run register to measure actual GPU batch profiles"
        self.last_solver_ms = 0.0
        self.solves = 0
        self.ewma_rate = None
        self.demand_alpha = float(os.environ.get("EWMA_ALPHA", "0.7"))
        if not 0 < self.demand_alpha <= 1 or not 0 < interval or not guard_ms >= 0:
            raise ValueError("Invalid controller interval, EWMA_ALPHA or deadline guard")
        self.history = deque(maxlen=100)
        self.generation = 0
        self._read_profiles()
        if self.profiles:
            self._optimize(float(os.environ.get("INITIAL_RATE", "1")))
            if self.plan:
                self.backend.load(self.plan.model)
        self.thread = threading.Thread(target=self._work, name="proteus-gpu", daemon=True)
        self.controller = threading.Thread(target=self._control, name="proteus-milp", daemon=True)
        self.thread.start()
        self.controller.start()

    def _read_profiles(self):
        if not self.profile_path or not self.profile_path.exists():
            return
        data = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != self.backend.fingerprint():
            self.policy_error = "GPU/runtime/checkpoint fingerprint changed; run register again"
            return
        self.profiles = {n: row for n, row in data.get("profiles", {}).items() if n in self.backend.enabled}
        self.measured_accuracy = data.get("accuracy", {})

    def _save_profiles(self):
        if self.profile_path:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.profile_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"schema_version": 2,
                "fingerprint": self.backend.fingerprint(), "profiles": self.profiles,
                "accuracy": self.measured_accuracy}, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, self.profile_path)

    def _optimize(self, demand):
        try:
            with self.condition:
                if self.maintenance or self.stopping:
                    return
                profiles = dict(self.profiles)
                enabled = {n: self.backend.registry[n] for n in self.backend.enabled}
                if set(profiles) != set(enabled):
                    raise ValueError("Register profiles for every enabled model first")
                scores = normalized_accuracies(enabled, self.measured_accuracy)
                generation = self.generation
            start = time.perf_counter()
            plan = self.solver(profiles, scores, demand, self.slo)
            with self.condition:
                if self.maintenance or self.stopping or generation != self.generation:
                    return
                self.plan = plan
                self.solves += 1
                self.last_solver_ms = (time.perf_counter() - start) * 1000
                self.policy_error = None
                self.history.append(plan.to_dict())
                self.condition.notify_all()
            log.info("MILP plan=%s", json.dumps(plan.to_dict()))
        except Exception as error:
            with self.condition:
                self.policy_error = str(error)
            log.warning("MILP not applied: %s", error)

    def _control(self):
        previous_count, previous_time = 0, time.perf_counter()
        while True:
            with self.condition:
                self.condition.wait(timeout=self.interval)
                if self.stopping:
                    return
                now = time.perf_counter()
                elapsed = now - previous_time
                # Arrival notifications must not shorten the allocation window.
                if elapsed < self.interval * 0.95:
                    continue
                rate = (self.arrivals - previous_count) / elapsed
                previous_count, previous_time = self.arrivals, now
                if self.maintenance or self.mode != "proteus" or rate <= 0:
                    continue
                self.ewma_rate = rate if self.ewma_rate is None else (
                    self.demand_alpha * rate + (1 - self.demand_alpha) * self.ewma_rate)
                demand = self.ewma_rate
            self._optimize(demand)

    def submit(self, request):
        if not math.isfinite(request.slo) or request.slo <= 0 or not request.image:
            raise ValueError("Request requires image bytes and a finite positive SLO")
        if len(request.image) > 16 * 1024 * 1024:
            raise ValueError("Image exceeds 16 MiB")
        aliases = {self.backend.application}
        if len(self.backend.enabled) == 1:
            aliases.update({self.backend.model_name, self.backend.model_id})
        if request.model not in aliases or request.dispatch not in {"", "proteus"}:
            raise ValueError("Use the application name; multiple-model requests do not pin a variant")
        if self.mode == "proteus" and abs(request.slo - self.slo) > 1e-6:
            raise ValueError("Request SLO must match the configured/profiler SLO in policy mode")
        now = time.perf_counter()
        job = Job(request, concurrent.futures.Future(), now, time.time() * 1000, uuid.uuid4().hex)
        with self.condition:
            if self.maintenance or self.stopping:
                raise BusyError("Executor in maintenance/shutdown")
            if self.mode == "proteus" and self.plan is None:
                raise ValueError("Proteus policy not ready: " + str(self.policy_error))
            if len(self.queue) >= self.max_queue:
                raise BusyError("Request queue is full")
            self.queue.append(job)
            self.arrivals += 1
            self.condition.notify_all()
        return job.future

    def _rejection(self, job, reason):
        return InferReply(request_name=job.name, rejected=True, reject_kind="deadline_drop",
                          reason=reason, gpu=self.backend.gpu_type,
                          latency=(time.perf_counter() - job.arrival) * 1000)

    def _work(self):
        while True:
            with self.condition:
                while not self.queue or self.maintenance:
                    if self.stopping:
                        return
                    self.condition.wait()
                while self.queue and self.queue[0].future.cancelled():
                    self.queue.popleft()
                if not self.queue:
                    continue
                plan, mode = self.plan, self.mode
                if mode == "proteus":
                    profile = self.profiles[plan.model]
                    latencies = {b: row["p50_ms"] for b, row in profile["batches"].items()}
                    decision = adaptive_decision(
                        [j.arrival + j.request.slo / 1000 for j in self.queue],
                        latencies, plan.max_batch, time.perf_counter(), self.guard_ms)
                    if decision.action == "wait":
                        self.condition.wait(timeout=max(0, decision.wake_at - time.perf_counter()))
                        continue
                    if decision.action == "drop":
                        job = self.queue.popleft()
                        self.dropped += 1
                        if job.future.set_running_or_notify_cancel():
                            job.future.set_result(self._rejection(job, "Predicted batch cannot meet head SLO"))
                        continue
                    count, selected = decision.size, plan.model
                else:
                    count, selected = 1, self.backend.model_name
                jobs = []
                for _ in range(min(count, len(self.queue))):
                    job = self.queue.popleft()
                    if job.future.set_running_or_notify_cancel():
                        jobs.append(job)
                if not jobs:
                    continue
                self.busy = True
            replies, error = [], None
            try:
                start = time.perf_counter()
                load = self.backend.load(selected)
                service_start = time.perf_counter()
                results = self.backend.infer_batch([j.request.image for j in jobs])
                if len(results) != len(jobs):
                    raise RuntimeError("GPU batch reply count mismatch")
                finish = time.perf_counter()
                for job, result in zip(jobs, results):
                    latency = (finish - job.arrival) * 1000
                    timings = {"worker." + key: float(v) for key, v in result["timings"].items()}
                    timings.update({"worker.queue": (start - job.arrival) * 1000,
                        "worker.load": load["load_ms"], "worker.ready": load["ready_ms"],
                        "worker.service": (finish - service_start) * 1000,
                        "worker.model_loaded": float(load["loaded"]), "worker.batch_size": float(len(jobs)),
                        "server.slo_missed": float(latency > job.request.slo)})
                    timings.update({"worker." + k: v for k, v in load.get("stages", {}).items()})
                    width, height = result["image_size"]
                    replies.append(InferReply(request_name=job.name, inference=result["label"],
                        latency=latency, gpu=self.backend.gpu_type, variant=selected,
                        worker_ms=(finish - start) * 1000, img_width=width, img_height=height,
                        timings=timings, arrive_epoch_ms=job.arrival_epoch_ms,
                        reply_epoch_ms=time.time() * 1000))
                with self.condition:
                    self.processed += len(jobs)
                    self.batch_counts[str(len(jobs))] = self.batch_counts.get(str(len(jobs)), 0) + 1
                    self.server_slo_misses += sum(int(r.latency > j.request.slo) for r, j in zip(replies, jobs))
                log.info("GPU mode=%s model=%s batch=%d load=%.1fms service=%.1fms", mode,
                         selected, len(jobs), load["load_ms"], (finish - service_start) * 1000)
            except Exception as exc:
                error = exc
                with self.condition:
                    self.failed += len(jobs)
                log.exception("GPU batch failed")
            finally:
                with self.condition:
                    self.busy = False
                    self.condition.notify_all()
            for index, job in enumerate(jobs):
                if error:
                    job.future.set_exception(error)
                else:
                    job.future.set_result(replies[index])

    def state(self):
        with self.condition:
            return {**self.backend.describe(), "implementation": "proteus-single-gpu-port",
                "mode": self.mode, "batching": "accscale" if self.mode == "proteus" else "fixed1-diagnostic",
                "milp_enabled": self.mode == "proteus" and self.plan is not None,
                "policy_ready": self.plan is not None, "policy_error": self.policy_error,
                "plan": self.plan.to_dict() if self.plan else None,
                "milp_solve_count": self.solves, "milp_history": list(self.history),
                "accuracy_scaling_enabled": self.mode == "proteus" and self.plan is not None and len(self.backend.enabled) > 1,
                "accuracy_measured": bool(self.measured_accuracy), "preprocessing_loading_overlap": False,
                "batch_size_counts": dict(self.batch_counts), "queue_depth": len(self.queue),
                "busy": self.busy, "maintenance": self.maintenance, "processed": self.processed,
                "failed": self.failed, "deadline_drops": self.dropped, "server_slo_misses": self.server_slo_misses}

    def _exclusive(self, operation):
        with self.condition:
            if self.queue or self.busy or self.maintenance or self.stopping:
                raise BusyError("Wait until the request queue is idle")
            self.maintenance = True
            self.generation += 1
        try:
            return operation()
        finally:
            with self.condition:
                self.maintenance = False
                self.condition.notify_all()

    def reset(self):
        return self._exclusive(lambda: (self.backend.reset() or {
            "ok": True, "cold_definition": "model non-resident; OS/CUDA caches NOT cleared"}))

    def set_mode(self, mode):
        if mode not in {"fixed", "proteus"}:
            raise ValueError("mode must be fixed or proteus")
        def change():
            if mode == "proteus" and self.plan is None:
                raise ValueError("Run register/evaluate before enabling policy mode")
            if mode == "proteus":
                self.backend.load(self.plan.model)
            self.mode = mode
            return {"mode": mode}
        return self._exclusive(change)

    def profile(self, image, trials=30, models=None):
        if not image or not isinstance(trials, int) or not 1 <= trials <= 1000:
            raise ValueError("Image and trials in 1..1000 required")
        names = models or self.backend.enabled
        if not names or any(n not in self.backend.enabled for n in names):
            raise ValueError("Profile only enabled registry models")
        def measure():
            measured_profiles = dict(self.profiles)
            for name in names:
                load = self.backend.load(name)
                batches = {}
                for size in self.batches:
                    for _ in range(3):
                        self.backend.infer_batch([image] * size)
                    samples = []
                    for _ in range(trials):
                        start = time.perf_counter()
                        self.backend.infer_batch([image] * size)
                        samples.append((time.perf_counter() - start) * 1000)
                    batches[str(size)] = {"p50_ms": percentile(samples, 50),
                        "p95_ms": percentile(samples, 95), "samples_ms": samples}
                measured_profiles[name] = {"batches": batches, "model_id": self.backend.registry[name]["model_id"],
                    "checkpoint_commit": self.backend.describe().get("checkpoint_commit"),
                    "profile_input_sha256": hashlib.sha256(image).hexdigest(), "load": load}
            # Failed profiling must not leave a mixture of old/new measurements.
            self.profiles = measured_profiles
            self._save_profiles()
            return {"scope": "actual-GPU batch service profiles for Proteus MILP and batching",
                    "profiles": self.profiles, "fingerprint": self.backend.fingerprint()}
        result = self._exclusive(measure)
        self._optimize(float(os.environ.get("INITIAL_RATE", "1")))
        if self.plan:
            self._exclusive(lambda: self.backend.load(self.plan.model))
        return {**result, "policy_ready": self.plan is not None, "policy_error": self.policy_error,
                "plan": self.plan.to_dict() if self.plan else None}

    def evaluate(self, manifest):
        root = Path(os.environ.get("EVAL_DIR", "/eval")).resolve()
        manifest_path = (root / manifest).resolve()
        if not manifest_path.is_relative_to(root):
            raise ValueError("Manifest must stay inside EVAL_DIR")
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("Manifest must be a nonempty JSON list of path,label pairs")
        prepared, digest = [], hashlib.sha256()
        for row in rows:
            path = (root / row["path"]).resolve()
            label = row["label"]
            if not path.is_relative_to(root) or isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 1000:
                raise ValueError("Paths must stay inside EVAL_DIR; labels must be ImageNet IDs 0..999")
            data = path.read_bytes()
            # Length-prefix inputs to make dataset identity unambiguous.
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
            digest.update(str(label).encode())
            prepared.append((path, label, hashlib.sha256(data).digest()))
        def measure():
            results = {}
            for name in self.backend.enabled:
                self.backend.load(name)
                correct = 0
                for path, label, expected_hash in prepared:
                    data = path.read_bytes()
                    if hashlib.sha256(data).digest() != expected_hash:
                        raise ValueError("Evaluation dataset changed during measurement; retry with immutable inputs")
                    correct += self.backend.infer(data)["label_id"] == label
                results[name] = {"accuracy": 100 * correct / len(prepared), "correct": correct,
                    "total": len(prepared), "accuracy_source": "offline evaluation by this runtime",
                    "accuracy_dataset": "sha256:" + digest.hexdigest() + ":imagenet-top1",
                    "checkpoint_commit": self.backend.describe().get("checkpoint_commit")}
            self.measured_accuracy = results
            self._save_profiles()
            return results
        result = self._exclusive(measure)
        self._optimize(float(os.environ.get("INITIAL_RATE", "1")))
        if self.plan:
            self._exclusive(lambda: self.backend.load(self.plan.model))
        return {"accuracy": result, "policy_ready": self.plan is not None, "policy_error": self.policy_error}

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.controller.join(timeout=5)
        self.thread.join(timeout=180)
        if not self.thread.is_alive():
            self.backend.reset()
