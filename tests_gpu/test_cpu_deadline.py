"""Concurrency/lifecycle regressions. Fake timings are NOT GPU measurements."""
import os
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import grpc

from proteus_realtime.client import query_once
from proteus_realtime.deployment import compose_document, settings
from proteus_realtime.metrics import split_latency_metrics
from proteus_realtime.policy import BatchDecision, Plan
from proteus_realtime.protocol import InferRequest
from proteus_realtime.runtime import GpuRuntime
from proteus_realtime.server import create_server
from proteus_realtime.service import BusyError, Job
from proteus_realtime.serving import ProteusExecutor
from test_standalone import FakeGpu


def solver(profiles, scores, demand, slo):
    return Plan("resnet-50", 1, 100, demand, 1, 100, 0, "TEST FIXTURE")


def wait_until(predicate, timeout=3):
    end = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() >= end:
            raise AssertionError("Fixture did not finish")
        time.sleep(0.005)


class DummyImage:
    size = (224, 224)

    @staticmethod
    def open(data):
        return DummyImage()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def convert(self, mode):
        assert mode == "RGB"
        return self


class DummyProcessor:
    barrier = None

    def __init__(self, config=None):
        self.config = config or {"mean": [0.1, 0.2, 0.3], "size": 224}
        self.calls = 0

    def to_dict(self):
        return {"mean": list(self.config["mean"]), "size": self.config["size"]}

    @classmethod
    def from_dict(cls, config):
        return cls(config)

    def __call__(self, images, return_tensors):
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        self.calls += 1
        return {"pixel_values": SimpleNamespace(device=SimpleNamespace(type="cpu"),
                value=(tuple(self.config["mean"]), self.config["size"]))}


class ThreadLocalProcessorTests(unittest.TestCase):
    def runtime(self):
        runtime = GpuRuntime.__new__(GpuRuntime)
        runtime.Image = DummyImage
        runtime.cpu_processor = DummyProcessor()
        runtime._cpu_processors = threading.local()
        runtime.preprocess_contract = "identical-config-fixture"
        return runtime

    def test_each_worker_owns_processor_and_calls_can_overlap(self):
        runtime = self.runtime()
        def run():
            a = runtime.prepare(b"jpeg")
            first = runtime._thread_processor()
            b = runtime.prepare(b"jpeg")
            self.assertIs(first, runtime._thread_processor())
            self.assertEqual(a["inputs"]["pixel_values"].value,
                             b["inputs"]["pixel_values"].value)
            return first
        with patch.object(DummyProcessor, "barrier", threading.Barrier(4)):
            with ThreadPoolExecutor(max_workers=4) as pool:
                processors = list(pool.map(lambda _: run(), range(4)))
        self.assertEqual(len({id(p) for p in processors}), 4)
        self.assertTrue(all(p.calls == 2 for p in processors))
        self.assertEqual(runtime.cpu_processor.calls, 0)

    def test_configuration_and_output_equal_serial_reference(self):
        runtime = self.runtime()
        reference = runtime.cpu_processor(images=[DummyImage()], return_tensors="pt")
        actual = runtime.prepare(b"jpeg")
        self.assertEqual(actual["inputs"]["pixel_values"].value,
                         reference["pixel_values"].value)
        self.assertEqual(actual["contract"], runtime.preprocess_contract)
        runtime._thread_processor().config["mean"][0] = 9
        self.assertEqual(runtime.cpu_processor.config["mean"][0], 0.1)


class EarlyDeadlineTests(unittest.TestCase):
    def make(self, backend=None, slo=100, early=True, rpc=0, max_queue=64):
        owner = ProteusExecutor(backend or FakeGpu(), slo=slo, batches=(1,),
            interval=100, solver=solver, early_deadline_drop=early, rpc_budget_ms=rpc,
            max_queue=max_queue)
        owner.profiles = {"resnet-50": {"batches": {"1": {"p50_ms": 5}}}}
        owner._optimize(1)
        self.addCleanup(owner.close)
        return owner

    def request(self, image=b"jpeg", slo=100):
        return InferRequest(model="imagenet", image=image, slo=slo)

    def test_queued_and_running_cpu_requests_expire_without_waiting_for_cpu(self):
        backend = FakeGpu()
        started, release = threading.Event(), threading.Event()
        calls = []
        def prepare(image):
            calls.append(image)
            started.set()
            self.assertTrue(release.wait(3))
            return {"raw": image}
        backend.prepare = prepare
        with patch.dict(os.environ, {"PREPROCESS_WORKERS": "1"}):
            owner = self.make(backend)
        try:
            with patch.object(backend, "infer_prepared", wraps=backend.infer_prepared) as gpu:
                first = owner.submit(self.request(b"first"))
                self.assertTrue(started.wait(1))
                second = owner.submit(self.request(b"queued"))
                a, b = first.result(1), second.result(1)
                self.assertTrue(a.rejected and b.rejected)
                self.assertEqual(a.reject_kind, "deadline_drop")
                self.assertEqual(a.timings["worker.rejection_stage.cpu_service"], 1)
                self.assertEqual(b.timings["worker.rejection_stage.cpu_queue"], 1)
                self.assertEqual(a.timings["worker.cpu_service_complete"], 0)
                self.assertEqual(b.timings["worker.cpu_started"], 0)
                self.assertEqual(owner.state()["queue_depth"], 0)
                self.assertEqual(owner.state()["cpu_running_requests"], 1)
                with self.assertRaises(BusyError):
                    owner.reset()
                release.set()
                wait_until(lambda: not owner.cpu.active)
                self.assertEqual(calls, [b"first"])
                self.assertEqual(len(owner.cpu.pending), 0)
                self.assertEqual(len(owner.cpu.completed), 0)
                gpu.assert_not_called()
                self.assertTrue(owner.expirer.is_alive())
        finally:
            release.set()

    def test_before_cpu_check_does_not_depend_on_watcher_scheduling(self):
        backend = FakeGpu()
        owner = self.make(backend, early=False)
        owner.early_deadline_drop = True  # Deliberately no expiry thread.
        job = Job(self.request(), Future(), time.perf_counter() - 1, 0, "stale")
        with patch.object(backend, "prepare", wraps=backend.prepare) as prepare:
            with owner.condition:
                owner.outstanding[job.name] = job
                owner.cpu.submit(job)
            reply = job.future.result(1)
            prepare.assert_not_called()
        self.assertEqual(reply.timings["worker.rejection_stage.before_cpu"], 1)
        self.assertEqual(reply.timings["worker.cpu_started"], 0)

    def test_after_cpu_check_does_not_restart_deadline(self):
        owner = self.make(early=False)
        owner.early_deadline_drop = True
        now = time.perf_counter()
        job = Job(self.request(), Future(), now - 1, 0, "finished",
                  prepared={"raw": b"jpeg"}, cpu_started=now - 0.9, cpu_finished=now - 0.8)
        with owner.condition:
            owner.outstanding[job.name] = job
            owner._prepared(job)
        reply = job.future.result(1)
        self.assertEqual(reply.timings["worker.rejection_stage.after_cpu"], 1)
        self.assertEqual(reply.timings["worker.cpu_service_complete"], 1)
        self.assertGreater(reply.timings["worker.ready_queue"], 700)
        self.assertEqual(len(owner.queue), 0)

    def test_ready_queue_expires_while_gpu_runs_nonpreemptively(self):
        backend = FakeGpu()
        started, release = threading.Event(), threading.Event()
        original = backend.infer_prepared
        def infer(items):
            started.set()
            self.assertTrue(release.wait(3))
            return original(items)
        backend.infer_prepared = infer
        owner = self.make(backend)
        try:
            with patch("proteus_realtime.serving.adaptive_decision", return_value=BatchDecision("run", 1)):
                first = owner.submit(self.request(b"running"))
                self.assertTrue(started.wait(1))
                second = owner.submit(self.request(b"waiting"))
                reply = second.result(1)
                self.assertTrue(reply.rejected)
                self.assertEqual(reply.timings["worker.rejection_stage.ready_queue"], 1)
                self.assertFalse(first.done(), "Running GPU inference must not be preempted")
                release.set()
                self.assertFalse(first.result(1).rejected)
        finally:
            release.set()

    def test_cancelled_cpu_work_does_not_kill_watcher_or_double_count(self):
        backend = FakeGpu()
        started, release = threading.Event(), threading.Event()
        original = backend.prepare
        def prepare(image):
            if image == b"blocked":
                started.set()
                self.assertTrue(release.wait(3))
            return original(image)
        backend.prepare = prepare
        with patch.dict(os.environ, {"PREPROCESS_WORKERS": "1"}):
            owner = self.make(backend)
        try:
            first = owner.submit(self.request(b"blocked"))
            self.assertTrue(started.wait(1))
            cancelled = owner.submit(self.request(b"cancel"))
            self.assertTrue(cancelled.cancel())
            self.assertTrue(first.result(1).rejected)
            release.set()
            wait_until(lambda: not owner.cpu.active and not owner.outstanding)
            self.assertEqual(owner.state()["deadline_drops"], 1)
            self.assertTrue(owner.expirer.is_alive())
            with patch("proteus_realtime.serving.adaptive_decision", return_value=BatchDecision("run", 1)):
                self.assertFalse(owner.submit(self.request()).result(1).rejected)
        finally:
            release.set()

    def test_younger_completion_can_proceed_after_expired_fifo_head_is_removed(self):
        backend = FakeGpu()
        started, release = threading.Event(), threading.Event()
        original = backend.prepare
        def prepare(image):
            if image == b"old":
                started.set()
                self.assertTrue(release.wait(3))
            return original(image)
        backend.prepare = prepare
        with patch.dict(os.environ, {"PREPROCESS_WORKERS": "2"}):
            owner = self.make(backend, slo=200)
        try:
            with patch("proteus_realtime.serving.adaptive_decision", return_value=BatchDecision("run", 1)):
                old = owner.submit(self.request(b"old", slo=200))
                self.assertTrue(started.wait(1))
                # Separate deadlines; the younger prepared job remains valid
                # after the blocked older CPU job has expired.
                time.sleep(0.08)
                young = owner.submit(self.request(b"young", slo=200))
                self.assertTrue(old.result(1).rejected)
                self.assertFalse(young.result(1).rejected)
                self.assertFalse(release.is_set())
        finally:
            release.set()

    def test_fixed_diagnostic_preserves_late_execution(self):
        backend = FakeGpu()
        original = backend.prepare
        def prepare(image):
            time.sleep(0.04)
            return original(image)
        backend.prepare = prepare
        owner = self.make(backend, slo=10)
        owner.set_mode("fixed")
        reply = owner.submit(self.request(slo=10)).result(1)
        self.assertFalse(reply.rejected)
        self.assertGreater(reply.latency, 40)
        self.assertFalse(owner.state()["early_deadline_drop_active"])

    def test_burst_completes_once_without_late_cpu_outputs_reentering_gpu(self):
        backend = FakeGpu()
        original = backend.prepare
        def prepare(image):
            time.sleep(0.025)
            return original(image)
        backend.prepare = prepare
        with patch.dict(os.environ, {"PREPROCESS_WORKERS": "4"}):
            owner = self.make(backend, slo=120, max_queue=256)
        futures = [owner.submit(self.request(str(i).encode(), slo=120)) for i in range(100)]
        replies = [future.result(2) for future in futures]
        wait_until(lambda: not owner.cpu.active and not owner.outstanding)
        state = owner.state()
        self.assertEqual(len(replies), 100)
        self.assertEqual(state["processed"] + state["deadline_drops"], 100)
        self.assertEqual(state["failed"], 0)
        self.assertEqual(sum(state["deadline_drop_stages"].values()), state["deadline_drops"])
        self.assertEqual(len(owner.cpu.pending), 0)
        self.assertEqual(len(owner.cpu.completed), 0)
        self.assertEqual(len(owner.cpu.tasks), 0)
        self.assertTrue(owner.thread.is_alive() and owner.expirer.is_alive())

    def test_disabled_feature_has_no_timer_and_legacy_cpu_path(self):
        backend = FakeGpu()
        owner = self.make(backend, early=False)
        self.assertIsNone(owner.expirer)
        self.assertFalse(owner.state()["early_deadline_drop"])
        with patch.object(backend, "prepare", wraps=backend.prepare) as prepare:
            with patch("proteus_realtime.serving.adaptive_decision", return_value=BatchDecision("run", 1)):
                self.assertFalse(owner.submit(self.request()).result(1).rejected)
            prepare.assert_called_once()

    def test_expiry_uses_same_rpc_reduced_budget_once(self):
        owner = self.make(early=False, rpc=6)
        owner.early_deadline_drop = True
        now = time.perf_counter()
        job = Job(self.request(), Future(), now - 0.095, 0, "budget")
        with owner.condition:
            owner.outstanding[job.name] = job
            self.assertAlmostEqual(owner._deadline(job), job.arrival + 0.094)
            self.assertTrue(owner._expire_if_due(job, "cpu_queue", now))
        self.assertTrue(job.future.result().rejected)

    def test_rejections_count_as_slo_unmet_not_fast_success(self):
        owner = self.make(early=False)
        job = Job(self.request(), Future(), time.perf_counter(), 0, "reply")
        reply = owner._rejection(job, "fixture", "cpu_queue")
        report = split_latency_metrics([{"ok": not reply.rejected, "slo_ms": 100,
            "server_ms": reply.latency, "e2e_ms": reply.latency, "timings": dict(reply.timings)}])
        self.assertEqual(report["client_e2e"]["slo_unmet_ratio_of_all_requests"], 1)
        self.assertEqual(report["server_internal"]["slo_unmet_ratio_of_all_requests"], 1)
        self.assertEqual(report["client_e2e"]["latency"]["samples"], 0)

    def test_single_rpc_handler_is_released_while_cpu_is_still_busy(self):
        backend = FakeGpu()
        started, release = threading.Event(), threading.Event()
        def prepare(image):
            started.set()
            self.assertTrue(release.wait(3))
            return {"raw": image}
        backend.prepare = prepare
        owner = self.make(backend)
        server, port = create_server(owner, "127.0.0.1:0", threads=1)
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                grpc.channel_ready_future(channel).result(2)
                with ThreadPoolExecutor(max_workers=2) as clients:
                    a = clients.submit(query_once, channel, "imagenet", b"first", 100, 2)
                    self.assertTrue(started.wait(1))
                    b = clients.submit(query_once, channel, "imagenet", b"second", 100, 2)
                    for future in (a, b):
                        row = future.result(2)
                        self.assertFalse(row["ok"])
                        self.assertEqual(row["reject_kind"], "deadline_drop")
                    self.assertFalse(release.is_set())
        finally:
            release.set()
            server.stop(0).wait(2)


class EarlyDeadlineSettingsTests(unittest.TestCase):
    def test_constructor_rejects_non_boolean_switch(self):
        for value in ("0", "1", 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ProteusExecutor(FakeGpu(), early_deadline_drop=value)

    def test_server_rejects_bad_switch_before_cuda_initialization(self):
        from proteus_realtime.server import main
        with patch.dict(os.environ, {"EARLY_DEADLINE_DROP": "bad"}), \
             patch("proteus_realtime.server.logging.basicConfig"), \
             patch("proteus_realtime.server.GpuRuntime") as runtime:
            with self.assertRaises(ValueError):
                main()
            runtime.assert_not_called()

    def test_switch_is_forwarded_but_does_not_change_cpu_or_gpu_profile(self):
        with patch.dict(os.environ, {}, clear=True):
            config = settings()
        self.assertEqual(config["EARLY_DEADLINE_DROP"], "1")
        env = compose_document(config)["services"]["worker"]["environment"]
        self.assertEqual(env["EARLY_DEADLINE_DROP"], "1")
        self.assertEqual(env["PREPROCESS_WORKERS"], "1")
        self.assertEqual(env["CPU_THREADS"], "2")
        self.assertEqual(env["CPU_INTEROP_THREADS"], "1")

    def test_invalid_switch_is_rejected(self):
        for value in ("true", "-1", "2", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"EARLY_DEADLINE_DROP": value}):
                with self.assertRaises(ValueError):
                    settings()

    def test_other_baselines_do_not_inherit_main_worker_flag(self):
        from proteus_baselines.config import compose_document as compose, settings as baseline_settings
        with patch.dict(os.environ, {"EARLY_DEADLINE_DROP": "1"}):
            for baseline in ("infaas-accuracy", "clipper-ha", "clipper-ht", "sommelier"):
                env = compose(baseline_settings(baseline), "0:0")["services"]["worker"]["environment"]
                self.assertNotIn("EARLY_DEADLINE_DROP", env)


if __name__ == "__main__":
    unittest.main()
