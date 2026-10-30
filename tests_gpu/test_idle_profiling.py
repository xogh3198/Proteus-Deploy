"""Deterministic profiling/consumer/rollback tests; NOT physical GPU results."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import grpc

from proteus_realtime.client import admin_call, query_once
from proteus_realtime.deployment import compose_document, settings
from proteus_realtime.policy import BatchDecision, solve_milp
from proteus_realtime.profiling import (ProfileOptions, execution_latency_ms,
                                        measure_batch, print_summary, validate_idle_batch)
from proteus_realtime.server import create_server
from proteus_realtime.serving import ProteusExecutor
from tests_gpu.test_standalone import FakeGpu


class FakeClock:
    def __init__(self):
        self.now, self.gap = 0.0, 0.0
        self.calls, self.sleeps = [], []
    def clock(self):
        return self.now
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.gap = seconds
        self.now += seconds
    def infer(self, batch):
        self.calls.append((len(batch), self.gap))
        self.now += .007 if not self.gap else .014
        self.gap = 0


class SamplingTests(unittest.TestCase):
    def test_idle_time_not_counted_in_execution_and_continuous_is_preserved(self):
        clock = FakeClock()
        options = ProfileOptions("idle-aware-v1", (0, 1000), 95)
        row = measure_batch(clock.infer, object(), 2, 4, options,
                            clock=clock.clock, sleep=clock.sleep)
        self.assertAlmostEqual(row["p50_ms"], 7)
        self.assertAlmostEqual(row["p95_ms"], 7)
        self.assertAlmostEqual(row["conditions"]["1000"]["p50_ms"], 14)
        self.assertAlmostEqual(row["prediction_ms"], 14)
        self.assertEqual(row["prediction_source_gap_ms"], 1000)
        self.assertEqual(len(row["conditions"]["1000"]["samples_ms"]), 4)
        self.assertEqual(len(clock.calls), 2 * (3 + 4))
        self.assertEqual(clock.sleeps, [1] * 4)
        self.assertTrue(all(len_ == 2 for len_, _ in clock.calls))
        self.assertTrue(all(value < 20 for value in row["conditions"]["1000"]["samples_ms"]))

    def test_legacy_continuous_has_original_p50_shape_and_no_idle(self):
        clock = FakeClock()
        row = measure_batch(clock.infer, object(), 1, 3, ProfileOptions(),
                            clock=clock.clock, sleep=clock.sleep)
        self.assertEqual(set(row), {"p50_ms", "p95_ms", "samples_ms"})
        self.assertAlmostEqual(execution_latency_ms(row), 7)
        self.assertEqual(clock.sleeps, [])

    def test_selected_percentile_is_explicit_not_a_hardcoded_multiplier(self):
        calls = [0]
        clock = FakeClock()
        def infer(batch):
            clock.now += .01 if calls[0] < 7 else .02 + .001 * calls[0]
            calls[0] += 1
        options = ProfileOptions("idle-aware-v1", (0., 10.), 50)
        row = measure_batch(infer, object(), 1, 4, options, clock=clock.clock, sleep=clock.sleep)
        self.assertEqual(row["prediction_statistic"], "max_condition_p50")
        self.assertAlmostEqual(row["prediction_ms"], max(r["p50_ms"] for r in row["conditions"].values()))

    def test_nonpositive_measurement_refused(self):
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            measure_batch(lambda batch: None, object(), 1, 3, ProfileOptions(), clock=lambda: 0)

    def test_options_reject_bad_strategy_gaps_and_percentile(self):
        for args in [("other", (0, 100), 95), ("idle-aware-v1", (0,), 95),
                     ("idle-aware-v1", (100, 1000), 95), ("idle-aware-v1", (0, 100, 50), 95),
                     ("idle-aware-v1", (0, 6000), 95), ("idle-aware-v1", (0, float("nan")), 95),
                     ("idle-aware-v1", (0, 100), 90)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                ProfileOptions(*args)

    def test_minimum_idle_budget_matches_two_models_four_batches(self):
        self.assertEqual(ProfileOptions("idle-aware-v1").minimum_idle_seconds(30, 2, 4), 264)

    def test_saved_prediction_must_match_original_samples(self):
        clock = FakeClock()
        options = ProfileOptions("idle-aware-v1", (0, 1000), 95)
        row = measure_batch(clock.infer, object(), 1, 4, options, clock=clock.clock, sleep=clock.sleep)
        validate_idle_batch(row, options)
        row["prediction_ms"] = 7
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_idle_batch(row, options)


class ConsumerTests(unittest.TestCase):
    def test_capacity_keeps_continuous_p50_but_deadline_prediction_is_separate(self):
        profiles = {"model": {"batches": {"1": {"p50_ms": 7, "prediction_ms": 14},
                                           "2": {"p50_ms": 10, "prediction_ms": 20}}}}
        plan = solve_milp(profiles, {"model": 100}, 1, 200)
        self.assertEqual(plan.max_batch, 2)
        self.assertEqual(plan.capacity_qps, 200)  # NOT 2*1000/20

    def test_slo_half_eligibility_uses_execution_prediction(self):
        profiles = {"model": {"batches": {"1": {"p50_ms": 7, "prediction_ms": 14},
                                           "2": {"p50_ms": 10, "prediction_ms": 120}}}}
        self.assertEqual(solve_milp(profiles, {"model": 100}, 1, 200).max_batch, 1)

    def test_nonfinite_prediction_never_reaches_scheduler(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                execution_latency_ms({"p50_ms": 7, "prediction_ms": value})


class ExecutorTests(unittest.TestCase):
    def make(self, root, options=None):
        return ProteusExecutor(FakeGpu(), profile_path=root / "profiles.json", batches=(1,),
                               interval=100, profile_options=options or ProfileOptions("idle-aware-v1", (0, .1), 95))

    def test_profile_measurements_use_real_worker_thread_and_prepare_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = self.make(Path(directory))
            threads, preparations = set(), []
            original, prepare = owner.backend.infer_prepared, owner.backend.prepare
            def infer(items):
                threads.add(threading.current_thread().name)
                return original(items)
            def preparation(image):
                preparations.append(threading.current_thread().name)
                return prepare(image)
            owner.backend.infer_prepared, owner.backend.prepare = infer, preparation
            try:
                report = owner.profile(b"jpeg", trials=3)
                row = report["profiles"]["resnet-50"]["batches"]["1"]
                self.assertEqual(threads, {"proteus-gpu"})
                self.assertEqual(len(preparations), 1)
                self.assertTrue(preparations[0].startswith("cpu-preprocess"))
                self.assertEqual(set(row["conditions"]), {"0", "0.1"})
                self.assertGreaterEqual(row["prediction_ms"], row["p95_ms"])
                self.assertTrue(report["policy_ready"])
            finally:
                owner.close()

    def test_existing_profile_backed_up_and_sampling_changes_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = self.make(root)
            try:
                owner.profile(b"jpeg", trials=2)
                first = (root / "profiles.json").read_bytes()
                owner.profile(b"jpeg", trials=2)
                self.assertEqual(Path(owner.last_profile_backup).read_bytes(), first)
                self.assertEqual(json.loads((root / "profiles.json").read_text())["schema_version"], 3)
            finally:
                owner.close()
            other = self.make(root, ProfileOptions("idle-aware-v1", (0, .2), 95))
            try:
                self.assertIsNone(other.plan)
                self.assertIn("sampling configuration", other.policy_error)
            finally:
                other.close()

    def test_old_continuous_profile_is_not_used_as_idle_aware_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = self.make(root, ProfileOptions())
            try:
                legacy.profile(b"jpeg", trials=2)
            finally:
                legacy.close()
            owner = self.make(root)
            try:
                self.assertIsNone(owner.plan)
                self.assertIn("sampling configuration", owner.policy_error)
            finally:
                owner.close()

    def test_measurement_failure_keeps_memory_and_saved_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = self.make(root)
            try:
                owner.profile(b"jpeg", trials=2)
                previous = json.loads(json.dumps(owner.profiles))
                saved = (root / "profiles.json").read_bytes()
                with patch.object(owner.backend, "infer_prepared", side_effect=ValueError("inference failure")):
                    with self.assertRaisesRegex(ValueError, "inference failure"):
                        owner.profile(b"jpeg", trials=2)
                self.assertEqual(owner.profiles, previous)
                self.assertEqual((root / "profiles.json").read_bytes(), saved)
                self.assertFalse(owner.maintenance)
            finally:
                owner.close()

    def test_atomic_save_failure_keeps_memory_and_saved_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = self.make(root)
            try:
                owner.profile(b"jpeg", trials=2)
                previous = json.loads(json.dumps(owner.profiles))
                saved = (root / "profiles.json").read_bytes()
                with patch("proteus_realtime.serving.os.replace", side_effect=OSError("write failure")):
                    with self.assertRaisesRegex(OSError, "write failure"):
                        owner.profile(b"jpeg", trials=2)
                self.assertEqual(owner.profiles, previous)
                self.assertEqual((root / "profiles.json").read_bytes(), saved)
            finally:
                owner.close()

    def test_insufficient_rpc_timeout_rejected_before_loading_or_sleeping(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = self.make(Path(directory), ProfileOptions("idle-aware-v1"))
            try:
                with self.assertRaisesRegex(ValueError, "increase register --timeout"):
                    owner.profile(b"jpeg", trials=30, timeout_s=1)
                self.assertEqual(owner.backend.loads, 0)
                self.assertFalse(owner.maintenance)
            finally:
                owner.close()

    def test_adaptive_scheduler_receives_prediction_and_reply_logs_error(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = self.make(Path(directory))
            owner.profiles = {"resnet-50": {"batches": {"1": {"p50_ms": 7, "prediction_ms": 14}}}}
            owner._optimize(1)
            server, port = create_server(owner, "127.0.0.1:0", 8)
            server.start()
            try:
                with grpc.insecure_channel("127.0.0.1:" + str(port)) as channel, \
                     patch("proteus_realtime.serving.adaptive_decision", return_value=BatchDecision("run", 1)) as decision:
                    grpc.channel_ready_future(channel).result(2)
                    row = query_once(channel, "imagenet", b"jpeg", 200, 3)
                self.assertTrue(row["ok"], row["error"])
                self.assertEqual(decision.call_args.args[1], {"1": 14})
                self.assertEqual(row["timings"]["worker.predicted_gpu_service"], 14)
                self.assertEqual(row["timings"]["worker.profile_continuous_p50"], 7)
                self.assertIn("worker.gpu_prediction_error", row["timings"])
            finally:
                server.stop(0).wait(2)
                owner.close()

    def test_profile_rpc_round_trip_uses_server_options_and_returns_saved_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = self.make(Path(directory))
            server, port = create_server(owner, "127.0.0.1:0", 8)
            server.start()
            try:
                with grpc.insecure_channel("127.0.0.1:" + str(port)) as channel:
                    grpc.channel_ready_future(channel).result(2)
                    result = admin_call(channel, "Profile", {
                        "image_base64": base64.b64encode(b"jpeg").decode(), "trials": 2}, 60)
                self.assertEqual(result["profile_configuration"]["strategy"], "idle-aware-v1")
                self.assertTrue(result["policy_ready"])
                self.assertIn("prediction_ms", result["profiles"]["resnet-50"]["batches"]["1"])
            finally:
                server.stop(0).wait(2)
                owner.close()

    def test_shared_sommelier_profile_rpc_keeps_timeout_compatibility(self):
        from proteus_baselines.sommelier import SommelierExecutor
        owner = SommelierExecutor(FakeGpu(), batches=(1,), interval=100, profile_options=ProfileOptions())
        server, port = create_server(owner, "127.0.0.1:0", 8)
        server.start()
        try:
            with grpc.insecure_channel("127.0.0.1:" + str(port)) as channel:
                grpc.channel_ready_future(channel).result(2)
                result = admin_call(channel, "Profile", {
                    "image_base64": base64.b64encode(b"jpeg").decode(), "trials": 2}, 60)
            self.assertTrue(result["policy_ready"])
            self.assertEqual(result["profile_configuration"]["strategy"], "continuous")
        finally:
            server.stop(0).wait(2)
            owner.close()


class SettingsAndDocsTests(unittest.TestCase):
    def test_new_config_propagates_and_selects_separate_directory(self):
        with patch.dict(os.environ, {"PROFILE_STRATEGY": "idle-aware-v1", "PROFILE_IDLE_GAPS_MS": "0,100,1000",
                                    "PROFILE_PREDICTION_PERCENTILE": "95"}):
            values = settings()
            env = compose_document(values)["services"]["worker"]["environment"]
        self.assertEqual(env["PROFILE_STRATEGY"], "idle-aware-v1")
        self.assertEqual(env["PROFILE_DIR"], "/profiles/cpu-ready-idle-v1")
        self.assertEqual(env["DEADLINE_GUARD_MS"], "2")
        self.assertEqual(env["CPU_THREADS"], "2")
        self.assertEqual(env["PREPROCESS_WORKERS"], "1")
        self.assertTrue(values["PROFILE_PATH"].endswith(str(Path("cpu-ready-idle-v1/profiles.json"))))

    def test_compact_summary_does_not_dump_raw_samples(self):
        report = {"profiles": {"model": {"batches": {"1": {
            "p50_ms": 7, "p95_ms": 8, "prediction_ms": 15, "samples_ms": [999999],
            "prediction_source_gap_ms": 1000}}}}}
        stream = io.StringIO()
        with patch("sys.stdout", stream):
            print_summary(report)
        self.assertIn("15.000", stream.getvalue())
        self.assertNotIn("999999", stream.getvalue())

    def test_new_document_and_script_agree(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "IDLE_AWARE_PROFILING.md").read_text(encoding="utf-8")
        self.assertEqual(text.count("```") % 2, 0)
        script = (root / "deploy/docker/proteus-docker.sh").read_text(encoding="utf-8")
        self.assertIn("profile-summary)", script)
        self.assertIn('--timeout "$profile_timeout"', script)
        for key in ("PROFILE_STRATEGY", "PROFILE_IDLE_GAPS_MS", "PROFILE_PREDICTION_PERCENTILE"):
            self.assertIn(key, text)
        self.assertIn("IDLE_AWARE_PROFILING.md", (root / "setting_readme.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
