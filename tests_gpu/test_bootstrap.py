import argparse
import concurrent.futures
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import grpc

from proteus_realtime.client import admin_call, arrival_offsets, bench, cold_warm, query_once
from proteus_realtime.deployment import compose_document, read_env, settings
from proteus_realtime.metrics import percentile, summarize
from proteus_realtime.protocol import InferReply, InferRequest
from proteus_realtime.server import create_server
from proteus_realtime.service import BusyError, Executor


class FakeBackend:
    model_name = "resnet-50"
    model_id = "microsoft/resnet-50"
    gpu_type = "test-fake"

    def __init__(self, delay=0.001):
        self.resident = False
        self.delay = delay
        self.loads = 0

    def load(self):
        loaded = not self.resident
        if loaded:
            self.loads += 1
            time.sleep(0.002)
        self.resident = True
        return {"loaded": loaded, "load_ms": 2.0 if loaded else 0.0, "ready_ms": 1.0 if loaded else 0.0}

    def infer(self, image):
        time.sleep(self.delay)
        if image == b"broken":
            raise RuntimeError("test decode failure")
        return {"label": "Samoyed", "boxes": [], "image_size": (1546, 1213),
                "timings": {"forward": 1.0, "decode": 0.2}}

    def reset(self):
        self.resident = False

    def prepare(self, image):
        return {"raw": image}

    def infer_prepared(self, items):
        return [self.infer(item["raw"]) for item in items]

    def describe(self):
        return {"resident": self.resident, "gpu_name": "FAKE: unit test only"}


class ProtocolTests(unittest.TestCase):
    def test_request_and_reply_roundtrip(self):
        request = InferRequest(model="resnet-50", image=b"jpeg", slo=200)
        self.assertEqual(InferRequest.FromString(request.SerializeToString()), request)
        reply = InferReply(inference="Samoyed", gpu="v100", latency=70, timings={"worker.load": 0})
        self.assertEqual(InferReply.FromString(reply.SerializeToString()), reply)

    def test_wire_field_numbers_match_guide(self):
        self.assertEqual(InferRequest.DESCRIPTOR.fields_by_name["slo"].number, 3)
        self.assertEqual(InferReply.DESCRIPTOR.fields_by_name["inference"].number, 2)
        self.assertEqual(InferReply.DESCRIPTOR.fields_by_name["timings"].number, 14)


class MetricsTests(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(percentile([70, 60, 80], 50), 70)
        self.assertAlmostEqual(percentile([60, 80], 95), 79)
        self.assertIsNone(percentile([], 50))

    def test_success_is_not_slo_met(self):
        result = summarize([
            {"ok": True, "e2e_ms": 800, "slo_met": False},
            {"ok": True, "e2e_ms": 70, "slo_met": True},
            {"ok": False, "e2e_ms": 10, "slo_met": False}], 1)
        self.assertEqual(result["successful_replies"], 2)
        self.assertEqual(result["slo_met_replies"], 1)
        self.assertEqual(result["failed_or_rejected"], 1)
        self.assertEqual(result["slo_miss_ratio_among_replies"], 0.5)
        self.assertEqual(result["slo_met_ratio_of_all_requests"], 1 / 3)
        self.assertIsNone(result["accuracy"])

    def test_empty_summary(self):
        result = summarize([], 0)
        self.assertIsNone(result["mean_e2e_ms"])
        self.assertEqual(result["slo_goodput_qps"], 0)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.executor = Executor(self.backend, max_queue=2)

    def tearDown(self):
        self.executor.close()

    def request(self, slo=200, image=b"jpeg"):
        return InferRequest(model="resnet-50", image=image, slo=slo)

    def test_cold_then_warm(self):
        first = self.executor.submit(self.request()).result(2)
        second = self.executor.submit(self.request()).result(2)
        self.assertEqual(first.timings["worker.model_loaded"], 1)
        self.assertEqual(second.timings["worker.load"], 0)
        self.assertEqual(self.backend.loads, 1)
        self.assertEqual(self.executor.state()["processed"], 2)

    def test_late_success_not_rejection(self):
        reply = self.executor.submit(self.request(slo=0.001)).result(2)
        self.assertFalse(reply.rejected)
        self.assertEqual(reply.timings["server.slo_missed"], 1)

    def test_invalid_slo_and_model(self):
        for slo in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.executor.submit(self.request(slo=slo))
        with self.assertRaises(ValueError):
            self.executor.submit(InferRequest(model="resnet-18", image=b"jpeg", slo=200))
        with self.assertRaises(ValueError):
            self.executor.submit(self.request(image=b""))

    def test_reset_and_scope(self):
        self.executor.submit(self.request()).result(2)
        # Wait for the completion bookkeeping to release the busy marker.
        for _ in range(100):
            if not self.executor.state()["busy"]:
                break
            time.sleep(0.001)
        self.executor.reset()
        self.assertFalse(self.backend.resident)
        self.assertFalse(self.executor.state()["milp_enabled"])
        self.assertFalse(self.executor.state()["accuracy_measured"])

    def test_admin_cannot_reset_active_request(self):
        self.backend.delay = 0.1
        future = self.executor.submit(self.request())
        with self.assertRaises(BusyError):
            self.executor.reset()
        future.result(2)

    def test_profile_uses_service_measurements(self):
        result = self.executor.profile(b"jpeg", 3)
        self.assertEqual(result["trials"], 3)
        self.assertEqual(result["batch_size"], 1)
        self.assertGreater(result["mean_service_ms"], 0)
        self.assertIn("NOT measured", result["qps_warning"])
        self.assertIsNone(result["accuracy"])

    def test_failure_is_recorded_and_worker_survives(self):
        with self.assertRaises(RuntimeError):
            self.executor.submit(self.request(image=b"broken")).result(2)
        self.assertEqual(self.executor.submit(self.request()).result(2).inference, "Samoyed")
        self.assertEqual(self.executor.state()["failed"], 1)

    def test_queue_is_bounded(self):
        self.backend.delay = 0.1
        futures = [self.executor.submit(self.request())]
        for _ in range(100):
            if self.executor.state()["busy"]:
                break
            time.sleep(0.001)
        # Limit now covers CPU preparation + ready queue + running GPU work.
        futures.append(self.executor.submit(self.request()))
        with self.assertRaises(BusyError):
            self.executor.submit(self.request())
        for future in futures:
            future.result(2)

    def test_cancelled_queued_request_is_not_executed(self):
        self.backend.delay = 0.03
        first = self.executor.submit(self.request())
        for _ in range(100):
            if self.executor.state()["busy"]:
                break
            time.sleep(0.001)
        cancelled = self.executor.submit(self.request())
        self.assertTrue(cancelled.cancel())
        first.result(2)
        for _ in range(100):
            if self.executor.queue.empty() and not self.executor.state()["busy"]:
                break
            time.sleep(0.001)
        self.assertEqual(self.executor.state()["processed"], 1)


class GrpcTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.executor = Executor(self.backend)
        self.server, port = create_server(self.executor, "127.0.0.1:0", threads=8)
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(self.channel).result(2)

    def tearDown(self):
        self.channel.close()
        self.server.stop(0).wait(2)
        self.executor.close()

    def test_grpc_cold_warm_and_status(self):
        first = query_once(self.channel, "resnet-50", b"jpeg", 200)
        second = query_once(self.channel, "resnet-50", b"jpeg", 200)
        self.assertTrue(first["ok"])
        self.assertEqual(first["path"], "inactive")
        self.assertEqual(second["path"], "active")
        self.assertEqual(second["load_ms"], 0)
        self.assertTrue(admin_call(self.channel, "Status")["ready"])

    def test_bad_request_and_failure(self):
        bad = query_once(self.channel, "no-model", b"jpeg", 200)
        self.assertEqual(bad["reject_kind"], "bad_request")
        broken = query_once(self.channel, "resnet-50", b"broken", 200)
        self.assertEqual(broken["reject_kind"], "INTERNAL")

    def test_complete_cold_warm_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(model="resnet-50", slo=200, timeout=2,
                                      warm_requests=10, output=directory)
            self.assertEqual(cold_warm(self.channel, args, b"jpeg"), 0)
            reports = list(Path(directory).glob("*.json"))
            rows = list(Path(directory).glob("*.csv"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(len(rows), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report["warm"]["requests"], 10)
            self.assertTrue(report["warm_all_load_zero"])
            self.assertTrue(report["labels_consistent"])
            self.assertFalse(report["runtime"]["milp_enabled"])

    def test_overload_is_bounded_and_client_skips_are_reported(self):
        self.backend.delay = 0.03
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(model="resnet-50", slo=200, timeout=2, warmup=1,
                                      output=directory, rate=100000, duration=0.005,
                                      distribution="uniform", shape=0.05, seed=0, concurrency=1)
            with patch("builtins.print"):
                self.assertEqual(bench(self.channel, args, b"jpeg"), 1)
            report = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertGreater(report["client_skipped_scheduled_arrivals"], 0)
            self.assertLessEqual(report["requests"], 2)
            self.assertIsNone(report["accuracy"])


class DeploymentTests(unittest.TestCase):
    def test_env_literal_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.env"
            path.write_text('# comment\nMODEL_DIR="/a path/models"\nPORT=18081\n')
            self.assertEqual(read_env(path)["MODEL_DIR"], "/a path/models")

    def test_reject_shell_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.env"
            path.write_text('MODEL_DIR=$(echo secret)\n')
            with self.assertRaises(ValueError):
                read_env(path)

    def test_unquoted_inline_comment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.env"
            path.write_text("MODEL_DIR=/data/models  # existing cache\n")
            self.assertEqual(read_env(path)["MODEL_DIR"], "/data/models")

    def test_compose_gpu_and_local_port(self):
        with patch.dict(os.environ, {}, clear=True):
            values = settings()
        worker = compose_document(values)["services"]["worker"]
        self.assertEqual(worker["ports"], ["127.0.0.1:18081:9000"])
        self.assertEqual(worker["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"], ["2"])
        self.assertTrue(worker["volumes"][0]["read_only"])

    def test_old_infaas_cache_auto_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / "infaas/deploy/docker/infaas-docker.env"
            env.parent.mkdir(parents=True)
            env.write_text("MODEL_DIR=data/models\n")
            local = root / "proteus.env"
            local.write_text(f"INFAAS_ENV_FILE={env.as_posix()}\nMODEL_DIR=auto\n")
            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "Old/unknown"):
                settings(str(local))

    def test_external_bind_is_rejected(self):
        with patch.dict(os.environ, {"BIND_ADDRESS": "0.0.0.0"}):
            with self.assertRaises(ValueError):
                settings()


class WorkloadTests(unittest.TestCase):
    def test_uniform_arrivals(self):
        self.assertEqual(list(arrival_offsets(2, 1, "uniform", 0.05, 0)), [0, 0.5])

    def test_distributions_are_deterministic_with_seed(self):
        for distribution in ("poisson", "gamma"):
            first = list(arrival_offsets(20, 2, distribution, 0.05, 3))
            self.assertEqual(first, list(arrival_offsets(20, 2, distribution, 0.05, 3)))
            self.assertTrue(all(0 <= offset < 2 for offset in first))
            self.assertEqual(first, sorted(first))


if __name__ == "__main__":
    unittest.main()
