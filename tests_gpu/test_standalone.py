"""Policy/API integration uses a fake GPU; solver tests invoke real Gurobi.

No timing printed by these tests is a GPU experiment result.
"""
import argparse
import ast
import concurrent.futures
import json
import logging
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import grpc

from core.predictor import Predictor
from proteus_realtime.client import admin_call, cold_warm, policy_test, query_once
from proteus_realtime.deployment import compose_document, settings
from proteus_realtime.policy import BatchDecision, Plan, adaptive_decision, solve_milp
from proteus_realtime.registry import normalized_accuracies
from proteus_realtime.server import create_server
from proteus_realtime.serving import ProteusExecutor


class FakeGpu:
    model_name, model_id, application, gpu_type = "resnet-50", "microsoft/resnet-50", "imagenet", "TEST-NOT-A-GPU"

    def __init__(self, multiple=False):
        self.registry = {"resnet-50": {"model_id": self.model_id, "accuracy": 90,
            "accuracy_source": "unit fixture", "accuracy_dataset": "unit-top1"}}
        if multiple:
            self.registry["resnet-18"] = {"model_id": "microsoft/resnet-18", "accuracy": 75,
                "accuracy_source": "unit fixture", "accuracy_dataset": "unit-top1"}
        self.enabled = list(self.registry)
        self.current = None
        self.loads = 0

    def load(self, name=None):
        name = name or self.model_name
        loaded = self.current != name
        if loaded:
            self.loads += 1
            self.current = name
            time.sleep(0.002)
        return {"loaded": loaded, "load_ms": 2.0 if loaded else 0.0,
                "ready_ms": 1.0 if loaded else 0.0, "stages": {}}

    def infer_batch(self, images):
        if b"broken" in images:
            raise ValueError("fake invalid image")
        time.sleep(0.004 + 0.001 * len(images))
        return [{"label": "Samoyed", "label_id": 258, "image_size": (20, 20),
                 "boxes": [], "timings": {"forward": 4.0}} for _ in images]

    def prepare(self, image):
        return {"raw": image}

    def infer_prepared(self, items):
        return self.infer_batch([item["raw"] for item in items])

    def infer(self, image):
        return self.infer_batch([image])[0]

    def reset(self):
        self.current = None

    def fingerprint(self):
        return {"gpu": "TEST FIXTURE", "revision": "test-only"}

    def describe(self):
        return {"resident": self.current is not None, "model_name": self.current,
                "checkpoint_commit": "test-only", "independent_runtime": True}


class MilpTests(unittest.TestCase):
    def profiles(self):
        return {"large": {"batches": {"1": {"p50_ms": 20}, "2": {"p50_ms": 30}}},
                "small": {"batches": {"1": {"p50_ms": 5}, "2": {"p50_ms": 6}}}}

    def test_real_gurobi_selects_accuracy_then_capacity(self):
        low = solve_milp(self.profiles(), {"large": 100, "small": 80}, 20, 200)
        high = solve_milp(self.profiles(), {"large": 100, "small": 80}, 200, 200)
        self.assertEqual(low.model, "large")
        self.assertEqual(high.model, "small")
        self.assertEqual(high.routing_fraction, 1)
        self.assertEqual(high.solver_status, "optimal")

    def test_beta_relaxes_only_when_capacity_insufficient(self):
        plan = solve_milp(self.profiles(), {"large": 100, "small": 80}, 400, 200)
        self.assertEqual(plan.model, "small")
        self.assertLess(plan.beta, 1)
        self.assertGreaterEqual(plan.capacity_qps, 400 * plan.beta)

    def test_slo_half_disallows_large_batches(self):
        plan = solve_milp(self.profiles(), {"large": 100, "small": 80}, 1, 40)
        self.assertEqual(plan.model, "small")

    def test_infeasible_and_zero_demand_not_faked(self):
        with self.assertRaises(ValueError):
            solve_milp(self.profiles(), {"large": 100, "small": 80}, 0, 200)
        with self.assertRaises(ValueError):
            solve_milp(self.profiles(), {"large": 100, "small": 80}, 100000, 200)

    def test_missing_multi_model_accuracy_rejected(self):
        with self.assertRaisesRegex(ValueError, "offline accuracy"):
            normalized_accuracies({"large": {"accuracy": None}, "small": {"accuracy": None}})
        self.assertEqual(normalized_accuracies({"large": {"accuracy": None}}), {"large": 100})

    def test_accuracy_normalization(self):
        models = {n: {"accuracy": v, "accuracy_source": "validation", "accuracy_dataset": "same-top1"}
                  for n, v in [("large", 80), ("small", 60)]}
        self.assertEqual(normalized_accuracies(models), {"large": 100, "small": 75})


class BatchingTests(unittest.TestCase):
    latencies = {"1": 10, "2": 15, "4": 20, "8": 30}

    def test_wait_until_deadline_not_immediate_fifo(self):
        d = adaptive_decision([1.2], self.latencies, 8, 1, 2)
        self.assertEqual(d.action, "wait")
        self.assertAlmostEqual(d.wake_at, 1.188)

    def test_timer_guard_tolerates_normal_overshoot(self):
        self.assertEqual(adaptive_decision([1.2], self.latencies, 8, 1.189, 2).action, "run")

    def test_expired_head_drops(self):
        self.assertEqual(adaptive_decision([1.005], self.latencies, 8, 1).action, "drop")

    def test_over_full_maximum_dispatches(self):
        d = adaptive_decision([1.2] * 9, self.latencies, 8, 1)
        self.assertEqual((d.action, d.size), ("run", 8))

    def test_original_predictor_batch_size_rule_is_preserved(self):
        original = Predictor.__new__(Predictor)
        original.batch_sizes_allowed, original.max_batch_size = [1, 2, 4, 8], 8
        for count in range(1, 15):
            ours = adaptive_decision([1.2] * count, self.latencies, 8, 1)
            self.assertEqual(ours.size, original.find_batch_size(count))

    def test_idle(self):
        self.assertEqual(adaptive_decision([], self.latencies, 8, 0).action, "idle")


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.backend = FakeGpu()
        self.executor = ProteusExecutor(self.backend, profile_path=Path(self.tmp.name) / "profiles.json",
                                        slo=200, interval=0.1, guard_ms=20)
        self.server, port = create_server(self.executor, "127.0.0.1:0", 16)
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(self.channel).result(2)

    def tearDown(self):
        self.channel.close()
        self.server.stop(0).wait(2)
        self.executor.close()
        self.tmp.cleanup()

    def register(self):
        result = self.executor.profile(b"jpeg", trials=2)
        self.assertTrue(result["policy_ready"])
        return result

    def test_fresh_worker_requires_real_profile(self):
        row = query_once(self.channel, "imagenet", b"jpeg", 200)
        self.assertFalse(row["ok"])
        self.assertIn("register", row["error"])

    def test_register_connects_actual_solver_and_profiles(self):
        self.register()
        state = self.executor.state()
        self.assertTrue(state["milp_enabled"])
        self.assertEqual(state["batching"], "accscale")
        self.assertTrue(self.backend.current)
        data = json.loads((Path(self.tmp.name) / "profiles.json").read_text())
        self.assertEqual(set(data["profiles"]["resnet-50"]["batches"]), {"1", "2", "4", "8"})

    def test_real_grpc_burst_forms_batch(self):
        self.register()
        args = argparse.Namespace(model="imagenet", slo=200, timeout=3, output=self.tmp.name)
        with patch("builtins.print"):
            self.assertEqual(policy_test(self.channel, args, b"jpeg"), 0)
        self.assertTrue(any(int(n) > 1 for n in self.executor.state()["batch_size_counts"]))

    def test_fixed_cold_warm_restores_policy(self):
        self.register()
        args = argparse.Namespace(model="imagenet", slo=200, timeout=3,
                                  warm_requests=2, output=self.tmp.name)
        with patch("builtins.print"):
            self.assertEqual(cold_warm(self.channel, args, b"jpeg"), 0)
        self.assertEqual(self.executor.mode, "proteus")
        self.assertTrue(self.executor.state()["milp_enabled"])

    def test_controller_measures_arrivals(self):
        self.register()
        query_once(self.channel, "imagenet", b"jpeg", 200)
        time.sleep(0.25)
        self.assertGreaterEqual(self.executor.solves, 2)
        self.assertIsNotNone(self.executor.ewma_rate)

    def test_slo_configuration_mismatch_rejected(self):
        self.register()
        row = query_once(self.channel, "imagenet", b"jpeg", 100)
        self.assertFalse(row["ok"])
        self.assertIn("SLO must match", row["error"])

    def test_changed_hardware_invalidates_cached_profile(self):
        self.register()
        with patch.object(self.backend, "fingerprint", return_value={"different": "GPU"}):
            other = ProteusExecutor(self.backend, profile_path=Path(self.tmp.name) / "profiles.json")
            try:
                self.assertIsNone(other.plan)
                self.assertIn("fingerprint", other.policy_error)
            finally:
                other.close()

    def test_evaluation_records_correct_count_and_dataset_identity(self):
        root = Path(self.tmp.name)
        (root / "sample.jpg").write_bytes(b"jpeg")
        (root / "manifest.json").write_text(json.dumps([{"path": "sample.jpg", "label": 258}]))
        with patch.dict(os.environ, {"EVAL_DIR": str(root)}):
            result = self.executor.evaluate("manifest.json")
        score = result["accuracy"]["resnet-50"]
        self.assertEqual((score["correct"], score["total"], score["accuracy"]), (1, 1, 100))
        self.assertIn("sha256", score["accuracy_dataset"])

    def test_evaluation_forbids_path_escape(self):
        with patch.dict(os.environ, {"EVAL_DIR": self.tmp.name}), self.assertRaises(ValueError):
            self.executor.evaluate("../outside.json")

    def test_failed_profile_keeps_previous_measurements(self):
        self.register()
        previous = json.loads(json.dumps(self.executor.profiles))
        with patch.object(self.backend, "infer_batch", side_effect=ValueError("broken image")):
            with self.assertRaises(ValueError):
                self.executor.profile(b"broken", trials=2)
        self.assertEqual(self.executor.profiles, previous)
        self.assertFalse(self.executor.maintenance)

    def test_evaluation_rejects_changed_dataset(self):
        root = Path(self.tmp.name)
        image = root / "sample.jpg"
        image.write_bytes(b"jpeg")
        (root / "manifest.json").write_text(json.dumps([{"path": "sample.jpg", "label": 258}]))
        original_load = self.backend.load
        def change_dataset(name):
            image.write_bytes(b"changed")
            return original_load(name)
        with patch.dict(os.environ, {"EVAL_DIR": str(root)}), patch.object(self.backend, "load", side_effect=change_dataset):
            with self.assertRaisesRegex(ValueError, "dataset changed"):
                self.executor.evaluate("manifest.json")
        self.assertEqual(self.executor.measured_accuracy, {})

    def test_multi_model_milp_changes_actual_executed_variant(self):
        backend = FakeGpu(multiple=True)
        executor = ProteusExecutor(backend, slo=200, interval=10, guard_ms=20)
        try:
            # Synthetic fixture profiles, never reported as GPU measurements.
            executor.profiles = {
                "resnet-50": {"batches": {"1": {"p50_ms": 20}, "2": {"p50_ms": 30}}},
                "resnet-18": {"batches": {"1": {"p50_ms": 5}, "2": {"p50_ms": 6}}}}
            executor._optimize(20)
            self.assertEqual(executor.plan.model, "resnet-50")
            executor._optimize(200)
            self.assertEqual(executor.plan.model, "resnet-18")
            from proteus_realtime.protocol import InferRequest
            # Deadline batching is covered separately. Isolate application of
            # the real MILP result from Windows OS timer scheduling jitter.
            with patch("proteus_realtime.serving.adaptive_decision", return_value=BatchDecision("run", 1)):
                reply = executor.submit(InferRequest(model="imagenet", image=b"jpeg", slo=200)).result(3)
            self.assertFalse(reply.rejected, reply.reason)
            self.assertEqual(reply.variant, "resnet-18")
            self.assertEqual(backend.current, "resnet-18")
            self.assertTrue(executor.state()["accuracy_scaling_enabled"])
        finally:
            executor.close()


class IndependenceTests(unittest.TestCase):
    def test_no_infaas_or_lumina_imports(self):
        root = Path(__file__).resolve().parents[1]
        for file in (root / "proteus_realtime").glob("*.py"):
            for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith(("infaas", "lumina")), file.name)
                elif isinstance(node, ast.Import):
                    self.assertTrue(all(not n.name.startswith(("infaas", "lumina")) for n in node.names), file.name)

    def test_configuration_is_standalone(self):
        with patch.dict(os.environ, {}, clear=True):
            values = settings()
        self.assertNotIn("BASE_IMAGE", values)
        self.assertNotIn("INFAAS_ENV_FILE", values)
        self.assertIn("data", values["MODEL_DIR"])
        worker = compose_document(values)["services"]["worker"]
        self.assertEqual(worker["environment"]["PROFILE_DIR"], "/profiles/cpu-ready-v1")
        self.assertTrue(worker["volumes"][0]["read_only"])


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
