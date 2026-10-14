"""Fake backend + REAL gRPC/original policy tests, not V100 measurements."""
import argparse
import concurrent.futures
import contextlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import grpc

from proteus_baselines import BASELINES
from proteus_baselines.client import cold_warm, main as client_main, trace_identity
from proteus_baselines.config import EXPECTED_SOURCES, ROOT, compose_document, settings, verify_sources
from proteus_baselines.native import NativePolicy
from proteus_baselines.serving import BaselineExecutor
from proteus_baselines import deploy
from proteus_realtime.client import admin_call, query_once
from proteus_realtime.protocol import InferRequest
from proteus_realtime.server import create_server


class FakeGpu:
    enabled = ["resnet-50"]
    model_name, model_id, application, gpu_type = "resnet-50", "microsoft/resnet-50", "imagenet", "TEST-NOT-V100"

    def __init__(self, delay=0.004):
        self.current, self.delay, self.forward_sizes = None, delay, []

    def load(self, name=None):
        loaded = self.current is None
        self.current = name or self.model_name
        return {"loaded": loaded, "load_ms": 1.0 if loaded else 0.0, "ready_ms": 1.0, "stages": {}}

    def infer_batch(self, images):
        self.forward_sizes.append(len(images))
        time.sleep(self.delay)
        if b"broken" in images:
            raise ValueError("Fake invalid image, NOT GPU failure")
        return [{"label": "Samoyed", "label_id": 258, "image_size": (20, 20), "timings": {"forward": 4.0}}
                for image in images]

    def reset(self):
        self.current = None

    def fingerprint(self):
        return {"gpu": "TEST FIXTURE", "revision": "unit-only"}

    def describe(self):
        return {"resident": self.current is not None, "model_name": self.current, "gpu_type": self.gpu_type,
                "runtime_backend": "UNIT TEST", "classification_accuracy": None}


class SettingsTests(unittest.TestCase):
    def test_three_isolated_paths_ports_and_names(self):
        configs = [settings(name) for name in BASELINES]
        for key in ("PORT", "PROJECT", "NETWORK_NAME", "CONTAINER_NAME", "RUN_DIR", "MODEL_DIR", "RESULT_DIR"):
            self.assertEqual(len({c[key] for c in configs}), 3, key)
        for c in configs:
            doc = compose_document(c, "0:0")
            worker = doc["services"]["worker"]
            self.assertEqual(worker["command"], ["proteus_baselines.server"])
            self.assertTrue(worker["ports"][0].startswith("127.0.0.1:1908"))
            self.assertEqual(worker["environment"]["CPU_THREADS"], "2")
            self.assertEqual(worker["environment"]["SLO_MS"], "200")
            self.assertEqual(worker["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"], ["2"])
            self.assertFalse(any("docker.sock" in v["source"] for v in worker["volumes"]))
            self.assertEqual(worker["labels"]["org.proteus.baseline.name"], c["BASELINE"])

    def test_old_generic_environment_does_not_share_cache(self):
        with patch.dict(os.environ, {"MODEL_DIR": "/other/clipper/models", "DATA_DIR": "/other/proteus",
                                     "PORT": "18081", "GPU": "0"}):
            c = settings("clipper-ha")
            self.assertNotIn("other", c["MODEL_DIR"])
            self.assertEqual(c["PORT"], "19082")
            self.assertEqual(c["GPU"], "2")

    def test_private_root_override_without_daemon_mutation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"BASELINE_DATA_ROOT": directory}):
            c = settings("clipper-ht")
            self.assertEqual(c["RUN_DIR"], str(Path(directory).resolve() / "clipper-ht"))
            self.assertNotIn("DOCKER_CONTEXT", c)
            self.assertNotIn("DOCKER_HOST", c)

    def test_sources_are_original_and_unchanged(self):
        self.assertEqual(verify_sources(), EXPECTED_SOURCES)

    def test_modified_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in EXPECTED_SOURCES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            with (root / "core/predictor.py").open("ab") as stream:
                stream.write(b"# changed\n")
            with self.assertRaisesRegex(ValueError, "core/predictor"):
                verify_sources(root)

    def test_registry_and_gpu_scope_fail_closed(self):
        with patch.dict(os.environ, {"BASELINE_GPU_TYPE": "../other"}):
            with self.assertRaisesRegex(ValueError, "GPU_TYPE"):
                settings("infaas-accuracy")
        with patch.dict(os.environ, {"BASELINE_MODEL_NAME": "not-enabled"}):
            with self.assertRaisesRegex(ValueError, "singleton"):
                settings("clipper-ha")
        with patch.dict(os.environ, {"BASELINE_WORKER_IMAGE": "proteus-worker:0.2.0"}):
            with self.assertRaisesRegex(ValueError, "separate"):
                settings("clipper-ha")

    def test_build_context_excludes_weights_credentials_and_original_changes(self):
        text = (ROOT / "deploy/docker/baselines/worker.Dockerfile.dockerignore").read_text()
        self.assertTrue(text.startswith("**\n"))
        self.assertNotIn("!data", text)
        self.assertNotIn("!traces", text)
        self.assertNotIn("!.git", text)
        recipe = (ROOT / "deploy/docker/baselines/worker.Dockerfile").read_text()
        self.assertIn("COPY core/common.py core/exceptions.py", recipe)
        self.assertNotIn("core/__init__.py", recipe)
        self.assertNotIn("infaas-worker:", recipe)
        self.assertNotIn("clipper/lib_base", recipe)

    def test_every_docker_copy_source_exists(self):
        recipe = (ROOT / "deploy/docker/baselines/worker.Dockerfile").read_text()
        copies = [shlex.split(line)[1:] for line in recipe.splitlines() if line.startswith("COPY ")]
        self.assertTrue(copies)
        for sources in copies:
            for source in sources[:-1]:
                with self.subTest(source=source):
                    self.assertTrue((ROOT / source).exists(), "Docker COPY source missing: " + source)

    def test_container_copy_layout_imports_namespace_core_without_initializer(self):
        # Reconstruct COPY destinations outside the checkout. Importing from the
        # full checkout alone cannot detect missing container source files.
        recipe = (ROOT / "deploy/docker/baselines/worker.Dockerfile").read_text()
        with tempfile.TemporaryDirectory() as directory:
            container_root = Path(directory)
            for line in recipe.splitlines():
                if not line.startswith("COPY "):
                    continue
                tokens = shlex.split(line)[1:]
                sources, target_text = tokens[:-1], tokens[-1]
                target = container_root / target_text.lstrip("/")
                for source_text in sources:
                    source = ROOT / source_text
                    if source.is_dir():
                        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    else:
                        destination = target / source.name if target_text.endswith("/") else target
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
            app = container_root / "app"
            self.assertFalse((app / "core/__init__.py").exists())
            env = dict(os.environ, PYTHONPATH=str(app), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
            check = (
                "import core; "
                "assert core.__spec__.origin is None, core.__spec__; "
                "import core.executor, core.predictor; "
                "import proteus_baselines.server, proteus_baselines.client; "
                "proteus_baselines.server.verify_sources(); "
                "print('Copied container layout imports OK')"
            )
            result = subprocess.run([sys.executable, "-c", check], cwd=app, env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Copied container layout imports OK", result.stdout)

    def test_collision_with_unrelated_container_never_deleted(self):
        foreign = {"Config": {"Labels": {"org.proteus.baseline.name": "another"}}, "State": {"Running": True}}
        result = type("Result", (), {"returncode": 0, "stdout": json.dumps([foreign])})()
        with patch.object(deploy, "run", return_value=result) as mocked:
            with self.assertRaisesRegex(ValueError, "another project"):
                deploy.down(settings("clipper-ha"))
            self.assertEqual(mocked.call_count, 1)


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.handlers = patch("logging.FileHandler", side_effect=lambda *a, **k: logging.NullHandler())
        self.handlers.start()
        self.resources = []

    def tearDown(self):
        for executor, server, channel in reversed(self.resources):
            if channel:
                channel.close()
            if server:
                server.stop(0).wait(2)
            executor.close()
        self.handlers.stop()
        self.temp.cleanup()

    def make(self, baseline="infaas-accuracy", slo=200, delay=0.004, latencies=None):
        backend = FakeGpu(delay)
        executor = BaselineExecutor(backend, baseline, Path(self.temp.name) / (baseline + ".json"),
                                    slo=slo, batches=(1, 2, 4), interval=0.03)
        if latencies:
            executor.profile_data = {"fingerprint": backend.fingerprint(), "slo_ms": slo,
                                     "batches": {str(b): {"p50_ms": v} for b, v in latencies.items()}}
            executor._exclusive(executor._install_native)
        server, port = create_server(executor, "127.0.0.1:0", 64)
        server.start()
        channel = grpc.insecure_channel("127.0.0.1:" + str(port))
        grpc.channel_ready_future(channel).result(2)
        self.resources.append((executor, server, channel))
        return executor, channel, port

    def test_unprofiled_native_request_is_not_fake_success(self):
        _, channel, _ = self.make()
        row = query_once(channel, "imagenet", b"jpeg", 200, timeout=2)
        self.assertFalse(row["ok"])
        self.assertIn("register", row["error"])

    def test_actual_profile_installs_original_predictor(self):
        executor, channel, _ = self.make()
        report = executor.profile(b"jpeg", trials=2)
        self.assertTrue(report["policy_ready"])
        self.assertIsInstance(executor.native, NativePolicy)
        self.assertEqual(type(executor.native.predictor).__module__, "core.predictor")
        self.assertEqual(type(executor.native.executor).__module__, "core.executor")
        self.assertEqual(set(report["profile"]["batches"]), {"1", "2", "4"})
        self.assertTrue(admin_call(channel, "Status")["resident"])

    def test_failed_profile_preserves_previous_policy_and_profile(self):
        executor, channel, _ = self.make(latencies={1: 90, 2: 142, 4: 261})
        before_profile, before_native = executor.profile_data, executor.native
        executor.backend.delay = 0.0
        with patch("proteus_baselines.serving.percentile", return_value=1000):
            with self.assertRaisesRegex(ValueError, "SLO/2"):
                executor.profile(b"jpeg", trials=1)
        self.assertIs(executor.profile_data, before_profile)
        self.assertIs(executor.native, before_native)
        self.assertTrue(query_once(channel, "imagenet", b"jpeg", 200, 2)["ok"])

    def test_all_three_real_rpc_paths_batch1_with_half_slo_bound(self):
        for baseline in BASELINES:
            with self.subTest(baseline=baseline):
                executor, channel, _ = self.make(baseline, latencies={1: 90, 2: 142, 4: 261})
                row = query_once(channel, "imagenet", b"jpeg", 200, timeout=2)
                self.assertTrue(row["ok"])
                self.assertEqual(row["label"], "Samoyed")
                self.assertEqual(row["batch_size"], 1)
                self.assertEqual(executor.native.max_batch, 1)
                self.assertEqual(executor.native.allowed_batch_sizes, [1])
                executor._exclusive(lambda: executor.native.control(100000))
                self.assertIsNone(executor.state()["policy_error"])

    def test_predicted_completion_never_replaces_actual_execution(self):
        executor, channel, _ = self.make(delay=0.06, latencies={1: 1, 2: 142, 4: 261})
        row = query_once(channel, "imagenet", b"jpeg", 200, timeout=2)
        self.assertTrue(row["ok"])
        self.assertGreaterEqual(row["server_ms"], 50)
        self.assertGreaterEqual(row["service_ms"], 50)
        self.assertEqual(executor.processed, 1)

    def test_clip_ha_ht_are_same_single_model_not_accuracy_measurements(self):
        for baseline in ("clipper-ha", "clipper-ht"):
            executor, _, _ = self.make(baseline, latencies={1: 5, 2: 8, 4: 12})
            state = executor.state()
            self.assertFalse(state["model_scaling_enabled"])
            self.assertIsNone(state["classification_accuracy"])
            self.assertEqual(state["normalized_accuracy"], 100)
            self.assertEqual(state["batching"], "aimd")

    def test_burst_calls_original_batching_and_actual_batched_backend(self):
        for baseline in BASELINES:
            with self.subTest(baseline=baseline):
                executor, channel, _ = self.make(baseline, latencies={1: 5, 2: 8, 4: 12})
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                    futures = [pool.submit(query_once, channel, "imagenet", b"jpeg", 200, 3, i) for i in range(16)]
                    rows = [future.result() for future in futures]
                self.assertTrue(any(row["ok"] for row in rows))
                self.assertGreater(max(executor.backend.forward_sizes), 1)
                self.assertEqual(sum(int(b) * n for b, n in executor.batch_counts.items()), executor.processed)
                self.assertIsNone(executor.state()["policy_error"])

    def test_original_aimd_expiry_noop_is_reported_not_synthetic_partial_batch(self):
        executor, channel, _ = self.make("clipper-ha", slo=60, latencies={1: 5, 2: 8, 4: 12})
        self.assertTrue(query_once(channel, "imagenet", b"jpeg", 60, 2)["ok"])
        self.assertEqual(executor.native.predictor.aimd_batch_size, 2)
        row = query_once(channel, "imagenet", b"jpeg", 60, 2)
        self.assertFalse(row["ok"])
        self.assertEqual(row["reject_kind"], "legacy_lazy_drop")
        self.assertEqual(executor.backend.forward_sizes, [1])

    def test_infaas_calls_original_up_down_controller(self):
        executor, channel, _ = self.make(latencies={1: 5, 2: 8, 4: 12})
        until = time.monotonic() + 1
        while executor.native.controller_calls == 0 and time.monotonic() < until:
            time.sleep(0.01)
        self.assertGreater(executor.native.controller_calls, 0)
        self.assertEqual(executor.native.predictor.get_infaas_cost(), 0)
        self.assertEqual(len(executor.native.executor.predictors), 1)
        self.assertEqual(executor.native.available_predictors, [0, 0, 0, 0])

    def test_fixed_cold_warm_is_separate_and_restores_native(self):
        executor, channel, _ = self.make("clipper-ha", latencies={1: 5, 2: 8, 4: 12})
        args = argparse.Namespace(model="imagenet", slo=200, timeout=3, output=self.temp.name,
                                  baseline="clipper-ha", warm_requests=10)
        with patch("builtins.print"):
            self.assertEqual(cold_warm(channel, args, b"jpeg"), 0)
        report = json.loads(next(Path(self.temp.name).glob("baseline-cold-warm*.json")).read_text())
        self.assertIn("NOT native", report["experiment"])
        self.assertTrue(report["warm_all_load_zero"])
        self.assertEqual(report["warm"]["successful_replies"], 10)
        self.assertEqual(executor.mode, "native")
        self.assertEqual(executor.native.predictor.aimd_batch_size, 1)
        self.assertEqual(executor.native_arrivals, 0)

    def test_wrong_endpoint_fails_before_request_or_reset(self):
        executor, _, port = self.make("clipper-ha", latencies={1: 90, 2: 142, 4: 261})
        with self.assertRaisesRegex(ValueError, "Endpoint baseline differs"):
            client_main(["state", "--baseline", "clipper-ht", "--controller", "127.0.0.1:" + str(port)])
        self.assertEqual(executor.arrivals, 0)

    def test_native_bench_outputs_matched_trace_metadata_for_all_three(self):
        reports = []
        image = Path(self.temp.name) / "input.jpg"
        image.write_bytes(b"jpeg")
        for baseline in BASELINES:
            executor, channel, port = self.make(baseline, latencies={1: 90, 2: 142, 4: 261})
            output = Path(self.temp.name) / baseline
            with patch("builtins.print"):
                self.assertEqual(client_main(["bench", "--baseline", baseline, "--controller",
                    "127.0.0.1:" + str(port), "--image", str(image), "--model", "imagenet", "--rate", "30",
                    "--duration", "0.1", "--seed", "10", "--timeout", "2", "--output", str(output)]), 0)
            report = json.loads(next(output.glob("bench-*.json")).read_text())
            self.assertEqual(report["baseline"], baseline)
            self.assertEqual(report["runtime"]["mode"], "native")
            self.assertEqual(report["fixed1_prebenchmark_warmup_requests"], 5)
            self.assertEqual(report["scheduled_requests"], 3)
            self.assertIsNone(report["runtime_after"]["policy_error"])
            reports.append(report)
        self.assertEqual(len({row["scheduled_trace_sha256"] for row in reports}), 1)
        self.assertEqual(len({row["image_sha256"] for row in reports}), 1)

    def test_original_profile_feedback_and_actual_slo_misses_both_visible(self):
        executor, channel, _ = self.make("clipper-ha", slo=40, delay=0.05, latencies={1: 5, 2: 30, 4: 60})
        row = query_once(channel, "imagenet", b"jpeg", 40, 2)
        self.assertTrue(row["ok"])
        self.assertFalse(row["slo_met"])
        self.assertEqual(executor.state()["server_slo_misses"], 1)

    def test_slo_mismatch_and_bad_image_are_not_success(self):
        executor, channel, _ = self.make(latencies={1: 90, 2: 142, 4: 261})
        self.assertFalse(query_once(channel, "imagenet", b"jpeg", 100, 2)["ok"])
        self.assertFalse(query_once(channel, "imagenet", b"broken", 200, 2)["ok"])
        self.assertEqual(executor.failed, 1)


class TraceTests(unittest.TestCase):
    def test_all_three_modes_use_identical_seeded_schedule(self):
        for distribution in ("uniform", "poisson", "gamma"):
            args = argparse.Namespace(rate=5, duration=2, distribution=distribution, shape=0.05, seed=10)
            self.assertEqual(trace_identity(args), trace_identity(args))
            if distribution != "uniform":
                changed = argparse.Namespace(**{**vars(args), "seed": 11})
                self.assertNotEqual(trace_identity(args)["scheduled_trace_sha256"],
                                    trace_identity(changed)["scheduled_trace_sha256"])


if __name__ == "__main__":
    unittest.main()
