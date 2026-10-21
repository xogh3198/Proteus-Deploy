"""Pipeline ordering/overlap tests; no CUDA timings or model results are claimed."""
import threading
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from proteus_realtime.pipeline import CpuStage
from proteus_realtime.protocol import InferRequest
from proteus_realtime.service import BusyError, Job
from proteus_realtime.serving import ProteusExecutor
from tests_gpu.test_standalone import FakeGpu


class CpuReadyTests(unittest.TestCase):
    def test_parallel_preparation_promotes_in_admission_order(self):
        release = threading.Event()
        second_prepared = threading.Event()
        promoted = []

        def prepare(image):
            if image == b"first":
                self.assertTrue(release.wait(3))
            else:
                second_prepared.set()
            return image

        condition = threading.Condition(threading.RLock())
        with patch.dict("os.environ", {"PREPROCESS_WORKERS": "2"}):
            stage = CpuStage(SimpleNamespace(prepare=prepare), condition,
                             lambda job: promoted.append(job.name), lambda job, error: None)
        try:
            with condition:
                for name in ("first", "second"):
                    stage.submit(Job(SimpleNamespace(image=name.encode()), Future(),
                                     time.perf_counter(), 0, name))
            self.assertTrue(second_prepared.wait(3))
            self.assertEqual(promoted, [])
            release.set()
        finally:
            release.set()
            stage.close()
        self.assertEqual(promoted, ["first", "second"])

    def test_cpu_next_request_overlaps_gpu_and_admission_is_bounded(self):
        backend = FakeGpu()
        gpu_started, release_gpu, next_prepared = (threading.Event() for _ in range(3))
        original_prepare, original_infer = backend.prepare, backend.infer_prepared

        def prepare(image):
            if image == b"second":
                next_prepared.set()
            return original_prepare(image)

        def infer(items):
            if items[0]["raw"] == b"first":
                gpu_started.set()
                self.assertTrue(release_gpu.wait(3))
            return original_infer(items)

        backend.prepare, backend.infer_prepared = prepare, infer
        executor = ProteusExecutor(backend, max_queue=2)
        executor.mode = "fixed"
        try:
            first = executor.submit(InferRequest(model="imagenet", image=b"first", slo=200))
            self.assertTrue(gpu_started.wait(3))
            second = executor.submit(InferRequest(model="imagenet", image=b"second", slo=200))
            self.assertTrue(next_prepared.wait(3), "CPU blocked behind GPU execution")
            with self.assertRaises(BusyError):
                executor.submit(InferRequest(model="imagenet", image=b"third", slo=200))
            with self.assertRaises(BusyError):
                executor.reset()
            release_gpu.set()
            for future in (first, second):
                reply = future.result(3)
                self.assertIn("worker.cpu_service", reply.timings)
                self.assertIn("worker.ready_queue", reply.timings)
                self.assertIn("worker.gpu_service", reply.timings)
        finally:
            release_gpu.set()
            executor.close()

    def test_cpu_time_does_not_restart_original_slo(self):
        backend = FakeGpu()
        original = backend.prepare

        def prepare(image):
            time.sleep(0.05)
            return original(image)

        backend.prepare = prepare
        executor = ProteusExecutor(backend)
        executor.mode = "fixed"
        try:
            reply = executor.submit(InferRequest(model="imagenet", image=b"jpeg", slo=10)).result(3)
            self.assertGreater(reply.latency, 50)
            self.assertEqual(reply.timings["server.slo_missed"], 1)
        finally:
            executor.close()

    def test_profile_prepares_once_outside_samples_on_serving_gpu_thread(self):
        backend = FakeGpu()
        calls, gpu_threads = [], set()
        original_prepare, original_infer = backend.prepare, backend.infer_prepared

        def prepare(image):
            calls.append(threading.current_thread().name)
            time.sleep(0.05)
            return original_prepare(image)

        def infer(items):
            gpu_threads.add(threading.current_thread().name)
            return original_infer(items)

        backend.prepare, backend.infer_prepared = prepare, infer
        executor = ProteusExecutor(backend, batches=(1,))
        try:
            result = executor.profile(b"jpeg", trials=3)
            row = result["profiles"]["resnet-50"]
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0].startswith("cpu-preprocess"))
            self.assertEqual(gpu_threads, {"proteus-gpu"})
            self.assertGreater(row["cpu_preparation_ms"], 40)
            self.assertEqual(row["profile_scope"], "prepared-batch-service")
        finally:
            executor.close()

    def test_preprocessing_failure_and_cancellation_release_admission(self):
        backend = FakeGpu()
        started, release = threading.Event(), threading.Event()

        def prepare(image):
            started.set()
            self.assertTrue(release.wait(3))
            raise ValueError("bad JPEG")

        backend.prepare = prepare
        executor = ProteusExecutor(backend, max_queue=1)
        executor.mode = "fixed"
        try:
            future = executor.submit(InferRequest(model="imagenet", image=b"jpeg", slo=200))
            self.assertTrue(started.wait(3))
            self.assertTrue(future.cancel())
            release.set()
            executor.cpu.close()
            self.assertEqual(executor.state()["queue_depth"], 0)
        finally:
            release.set()
            executor.close()
