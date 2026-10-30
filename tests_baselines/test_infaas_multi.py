"""Real original controllers and gRPC wiring with a fake GPU, not V100 results."""
import concurrent.futures
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from proteus_baselines.config import ROOT, settings, verify_sources
from proteus_baselines.infaas_accuracy import AccuracyExecutor
from proteus_baselines.trace_replay import check_worker
from proteus_baselines import trace_replay, deploy
from proteus_realtime.protocol import InferRequest
from proteus_realtime.server import create_server
from proteus_realtime.service import Job
from tests_baselines.test_gpu_baselines import FakeGpu


class ModelsGpu(FakeGpu):
    enabled = ["resnet-18", "resnet-50", "resnet-101"]

    def __init__(self):
        super().__init__(.001)
        self.registry = json.loads((ROOT / "deploy/docker/baselines/models.three-reference.json").read_text())["models"]
        self.loads = []

    def load(self, name=None):
        selected = name or self.model_name
        changed = self.current != selected
        self.current = selected
        self.loads.append(selected)
        return {"loaded": changed, "load_ms": 3.0 if changed else 0.0, "ready_ms": 3.0, "stages": {}}

    def fingerprint(self):
        return {"gpu": "TEST FIXTURE", "checkpoints": self.enabled}

    def describe(self):
        return {**super().describe(), "enabled_models": self.enabled, "application": self.application,
                "checkpoints": {n: {k: v[k] for k in ("model_id", "revision")}
                                for n, v in self.registry.items()}}


class MultiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.handlers = patch("logging.FileHandler", side_effect=lambda *a, **k: logging.NullHandler())
        self.handlers.start()
        self.backend = ModelsGpu()
        self.executor = AccuracyExecutor(self.backend, profile_path=Path(self.temp.name) / "profiles.json",
                                         batches=(1, 2, 4), interval=100)

    def tearDown(self):
        self.executor.close()
        self.handlers.stop()
        self.temp.cleanup()

    def profile(self):
        return self.executor.profile(b"jpeg", trials=2)

    def select_initial(self, name):
        native = self.executor.native
        native.close()
        native.executor.remove_predictor_by_id(native.predictor.id)
        identifier = native.executor.add_predictor(variant_name=name)
        native.predictor = native.executor.predictors[identifier]

    def test_profile_all_candidates_and_original_startup(self):
        result = self.profile()
        self.assertEqual(set(result["profile"]["models"]), set(self.backend.enabled))
        self.assertEqual(result["selected_model"], "resnet-18")
        self.assertAlmostEqual(self.executor.native.predictor.infaas_cost, 78.34 - 70.93)
        self.assertEqual(verify_sources(), __import__("proteus_baselines.config", fromlist=["EXPECTED_SOURCES"]).EXPECTED_SOURCES)

    def test_original_downscale_rule_not_replaced_by_accuracy_maximizer(self):
        self.profile()
        with self.executor.condition:
            self.select_initial("resnet-101")
            self.executor.native.control(1)
            self.assertEqual(self.executor.native.predictor.variant_name, "resnet-18")
            self.assertEqual(len(self.executor.native.executor.predictors), 1)
            self.assertEqual(self.executor.native.available_predictors, [0, 0, 0, 0])

    def test_real_gpu_completion_boundary_preserves_inflight_model(self):
        self.executor.batches = (1,)
        self.profile()
        self.backend.delay = .06
        with self.executor.condition:
            self.select_initial("resnet-101")
        first = self.executor.submit(InferRequest(model="imagenet", image=b"jpeg", slo=200))
        until = time.monotonic() + 2
        with self.executor.condition:
            while not self.executor.busy and time.monotonic() < until:
                self.executor.condition.wait(.01)
            self.assertTrue(self.executor.busy)
            self.executor.native.control(1)
            self.assertEqual(self.executor.native.predictor.variant_name, "resnet-101")
        reply = first.result(2)
        self.assertEqual(reply.variant, "resnet-101")
        second = self.executor.submit(InferRequest(model="imagenet", image=b"jpeg", slo=200))
        self.assertEqual(second.result(2).variant, "resnet-18")
        state = self.executor.state()
        self.assertEqual(state["model_switches_after_initial_load"], 2)
        self.assertEqual(state["successful_requests_by_model"], {"resnet-101": 1, "resnet-18": 1})
        self.assertEqual(state["model_switch_events"][-1]["load_ms"], 3)

    def test_pending_requests_keep_original_arrival_when_predictor_changes(self):
        self.profile()
        with self.executor.condition:
            self.select_initial("resnet-101")
            future = concurrent.futures.Future()
            job = Job(InferRequest(model="imagenet", image=b"jpeg", slo=200), future,
                      time.perf_counter() - self.executor.origin, time.time() * 1000, "migration-job")
            job.prepared = self.backend.prepare(b"jpeg")
            job.cpu_started = job.cpu_finished = time.perf_counter()
            self.executor.jobs[job.name] = job
            self.executor.native.enqueue(job)
            old_event = self.executor.native.predictor.request_queue[0]
            self.executor.native.control(1)
            event = self.executor.native.predictor.request_queue[0]
            self.assertEqual((event.id, event.start_time, event.deadline),
                             (old_event.id, old_event.start_time, old_event.deadline))
            self.assertIn(job.name, self.executor.jobs)
            self.executor.native.predictor.set_infaas_batch_size(1)
            self.executor.native.predictor.finish_batch_callback(self.executor.clock_ms())
        self.assertEqual(future.result(2).variant, "resnet-18")

    def test_actual_grpc_replay_keeps_all_scheduled_rows_and_reference_accuracy(self):
        self.executor.batches = (1,)
        self.profile()
        root = Path(self.temp.name)
        trace, image, registry = root / "arrivals.txt", root / "image.jpg", root / "models.json"
        trace.write_text("0,1,300\n10,1,300\n20,1,300\n", encoding="utf-8")
        image.write_bytes(b"jpeg")
        registry.write_text(json.dumps({"models": self.backend.registry}), encoding="utf-8")
        server, port = create_server(self.executor, "127.0.0.1:0", 16)
        server.start()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = trace_replay.main(["--baseline", "infaas-accuracy", "--controller", f"127.0.0.1:{port}",
                    "--trace", str(trace), "--image", str(image), "--accuracy-registry", str(registry),
                    "--limit", "0", "--speed", "2", "--concurrency", "16", "--timeout", "2",
                    "--output", str(root / "results"), "--max-send-lag-ms", "2000"])
            self.assertEqual(code, 0)
            report = json.loads(next((root / "results").glob("*.json")).read_text())
            self.assertEqual(report["summary"]["scheduled_requests"], 3)
            self.assertEqual(report["summary"]["sent_requests"], 3)
            self.assertEqual(report["summary"]["successful_requests_by_model"], {"resnet-18": 3})
            self.assertAlmostEqual(report["summary"]["mean_reference_accuracy_of_successful_replies"], 70.93)
        finally:
            server.stop(0).wait(2)

    def test_restore_complete_multi_profile_and_trace_identity_checks(self):
        self.profile()
        state = self.executor.state()
        check_worker(state, self.backend.registry, "infaas-accuracy")
        with self.assertRaisesRegex(ValueError, "requested"):
            check_worker(state, self.backend.registry, "clipper-ha")
        bad = dict(self.backend.registry)
        bad["resnet-18"] = {**bad["resnet-18"], "revision": "0" * 40}
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            check_worker(state, bad, "infaas-accuracy")
        self.executor._exclusive(self.executor._read_profile)
        self.assertTrue(self.executor.state()["policy_ready"])

    def test_unprofiled_partial_batch_does_not_fabricate_a_prediction_or_fail(self):
        self.profile()
        jobs = []
        for index in range(3):
            now = time.perf_counter()
            job = Job(InferRequest(model="imagenet", image=b"jpeg", slo=200), concurrent.futures.Future(),
                      now - self.executor.origin, time.time() * 1000, str(index))
            job.cpu_started = job.cpu_finished = now
            job.prepared = self.backend.prepare(b"jpeg")
            jobs.append(job)
        replies = self.executor._gpu(jobs, "native", "resnet-18")
        self.assertEqual(len(replies), 3)
        self.assertTrue(all(r.timings["worker.batch_size"] == 3 for r in replies))
        self.assertTrue(all("worker.gpu_prediction_error" not in r.timings for r in replies))

    def test_model_load_is_recorded_even_if_forward_fails(self):
        self.profile()
        with self.assertRaisesRegex(ValueError, "Fake invalid"):
            self.executor._gpu([SimpleNamespace(prepared={"raw": b"broken"})], "native", "resnet-101")
        state = self.executor.state()
        self.assertEqual(state["model_switches_after_initial_load"], 1)
        self.assertEqual(state["successful_requests_by_model"], {})
        self.assertEqual(state["model_switch_events"][-1]["to"], "resnet-101")


class MultiSettingsTests(unittest.TestCase):
    def test_docker_trace_route_passes_speed_limit_and_shared_inputs(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, root)
        trace, image = root / "arrivals.txt", root / "image.jpg"
        trace.write_text("0,1,200\n", encoding="utf-8")
        image.write_bytes(b"jpeg")
        config = str(ROOT / "deploy/docker/baselines/baseline.three-model.env")
        with patch.dict(os.environ, {"BASELINE_ENV": config, "BASELINE_DATA_ROOT": str(root)}), \
                patch.object(deploy, "image_required"), patch.object(deploy, "container_user", return_value="0:0"), \
                patch.object(deploy, "run") as run:
            deploy.main(["infaas-accuracy", "trace", "--trace", str(trace), "--image", str(image),
                         "--limit", "0", "--speed", "2", "--concurrency", "256", "--timeout", "10"])
        args = run.call_args.args[0]
        self.assertIn("proteus_baselines.trace_replay", args)
        for key, value in (("--speed", "2"), ("--limit", "0"), ("--concurrency", "256")):
            self.assertEqual(args[args.index(key) + 1], value)

    def test_only_infaas_accuracy_accepts_three_candidates(self):
        config = str(ROOT / "deploy/docker/baselines/baseline.three-model.env")
        with patch.dict(os.environ, {"BASELINE_ENV": config}):
            values = settings("infaas-accuracy")
            self.assertEqual(values["PREPROCESS_WORKERS"], "8")
            self.assertEqual(values["ALLOCATION_INTERVAL"], "10")
            with self.assertRaisesRegex(ValueError, "singleton"):
                settings("clipper-ha")


if __name__ == "__main__":
    unittest.main()
