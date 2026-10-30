"""RPC reserve arithmetic/reporting tests. Fake GPU times are NOT measurements."""
import csv
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import grpc

from proteus_realtime.client import query_once
from proteus_realtime.deployment import compose_document, settings
from proteus_realtime.metrics import (compact_bench_report, main as metrics_main,
                                     read_csv_rows, split_latency_metrics, summarize)
from proteus_realtime.policy import BatchDecision, Plan, adaptive_decision, solve_milp
from proteus_realtime.server import create_server
from proteus_realtime.serving import ProteusExecutor
from test_standalone import FakeGpu


def fake_solver(profiles, scores, demand, slo):
    return Plan("resnet-50", 1, 100, demand, 1, 100, 0, "TEST FIXTURE")


class BudgetSettingsTests(unittest.TestCase):
    def test_zero_default_preserves_existing_slo_guard_and_profile_path(self):
        with patch.dict(os.environ, {}, clear=True):
            values = settings()
        self.assertEqual(values["RPC_BUDGET_MS"], "0")
        self.assertEqual(values["SLO_MS"], "200")
        self.assertEqual(values["DEADLINE_GUARD_MS"], "2")
        self.assertTrue(values["PROFILE_PATH"].endswith(str(Path("cpu-ready-v1/profiles.json"))))

    def test_env_reaches_worker_without_changing_gpu_measurement_scope(self):
        with patch.dict(os.environ, {"RPC_BUDGET_MS": "6", "PROFILE_STRATEGY": "idle-aware-v1"}):
            values = settings()
            worker = compose_document(values)["services"]["worker"]
        for key, value in (("RPC_BUDGET_MS", "6"), ("SLO_MS", "200"), ("DEADLINE_GUARD_MS", "2"),
                           ("CPU_THREADS", "2"), ("PREPROCESS_WORKERS", "1")):
            self.assertEqual(worker["environment"][key], value)
        self.assertEqual(worker["environment"]["PROFILE_DIR"], "/profiles/cpu-ready-idle-v1")

    def test_invalid_budgets_fail_on_host(self):
        for value in ("nan", "inf", "-1", "200", "198", "199", "bad"):
            with self.subTest(value=value), patch.dict(os.environ, {"RPC_BUDGET_MS": value}):
                with self.assertRaises(ValueError):
                    settings()

    def test_invalid_budgets_fail_before_runtime_threads_start(self):
        for value in (math.nan, math.inf, -1, 200, 198):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ProteusExecutor(FakeGpu(), rpc_budget_ms=value)

    def test_baseline_config_does_not_inherit_main_worker_reserve(self):
        from proteus_baselines.config import compose_document as baseline_compose, settings as baseline_settings
        with patch.dict(os.environ, {"RPC_BUDGET_MS": "6"}):
            for baseline in ("infaas-accuracy", "clipper-ha", "clipper-ht", "sommelier"):
                env = baseline_compose(baseline_settings(baseline), "0:0")["services"]["worker"]["environment"]
                self.assertNotIn("RPC_BUDGET_MS", env)


class BudgetArithmeticTests(unittest.TestCase):
    def test_six_ms_reserve_advances_dispatch_by_exactly_six_ms(self):
        latencies = {"1": 16.205, "2": 15.592, "4": 15.846, "8": 16.561}
        old = adaptive_decision([1.2], latencies, 8, 1.0, 2)
        new = adaptive_decision([1.194], latencies, 8, 1.0, 2)
        self.assertEqual((old.action, new.action), ("wait", "wait"))
        self.assertAlmostEqual(old.wake_at - new.wake_at, 0.006)
        self.assertAlmostEqual(new.wake_at, 1.175795)

    def test_reduced_budget_changes_slo_half_eligibility_not_capacity_formula(self):
        profiles = {"high": {"batches": {"1": {"p50_ms": 98}}},
                    "low": {"batches": {"1": {"p50_ms": 50}}}}
        self.assertEqual(solve_milp(profiles, {"high": 100, "low": 80}, 1, 200).model, "high")
        new = solve_milp(profiles, {"high": 100, "low": 80}, 1, 194)
        self.assertEqual(new.model, "low")
        self.assertEqual(new.capacity_qps, 20)

    def test_no_server_timestamp_does_not_become_zero_ms_success(self):
        report = split_latency_metrics([{"ok": True, "e2e_ms": 20, "slo_ms": 200, "slo_met": True}])
        self.assertEqual(report["server_internal"]["unknown_successful_slo_requests"], 1)
        self.assertIsNone(report["server_internal"]["slo_unmet_ratio_of_all_requests"])


class BudgetServingTests(unittest.TestCase):
    def make(self, rpc=6, profile_path=None):
        return ProteusExecutor(FakeGpu(), batches=(1,), interval=100,
                               solver=fake_solver, rpc_budget_ms=rpc, profile_path=profile_path)

    def test_solver_and_scheduler_share_reduced_budget_and_keep_request_slo(self):
        owner = self.make()
        server = None
        try:
            owner.profiles = {"resnet-50": {"batches": {"1": {"p50_ms": 8, "prediction_ms": 16}}}}
            with patch.object(owner, "solver", side_effect=fake_solver) as solver:
                owner._optimize(1)
            self.assertEqual(solver.call_args.args[3], 194)
            server, port = create_server(owner, "127.0.0.1:0", 8)
            server.start()
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                grpc.channel_ready_future(channel).result(2)
                with patch("proteus_realtime.serving.adaptive_decision",
                           return_value=BatchDecision("run", 1)) as decision:
                    row = query_once(channel, "imagenet", b"jpeg", 200, 3)
            self.assertTrue(row["ok"], row["error"])
            self.assertEqual(row["slo_ms"], 200)
            self.assertEqual(row["timings"]["worker.rpc_budget_ms"], 6)
            self.assertEqual(row["timings"]["worker.server_budget_ms"], 194)
            self.assertEqual(row["timings"]["worker.predicted_gpu_service"], 16)
            self.assertEqual(decision.call_args.args[1], {"1": 16})
            self.assertEqual(decision.call_args.args[4], 2)
            outstanding_deadline = decision.call_args.args[0][0]
            self.assertGreater(outstanding_deadline, 0)
            state = owner.state()["deadline_budget"]
            self.assertEqual(state["server_budget_ms"], 194)
            self.assertFalse(state["client_deadline_propagated"])
            self.assertTrue(row["client_e2e_slo_met"])
            self.assertTrue(row["server_slo_met"])
        finally:
            if server:
                server.stop(0).wait(2)
            owner.close()

    def test_deadline_reduction_is_exactly_once_and_cpu_time_is_not_reset(self):
        owner = self.make()
        captured = []
        original_submit = owner.cpu.submit
        def capture(job):
            captured.append(job)
            return original_submit(job)
        try:
            owner.profiles = {"resnet-50": {"batches": {"1": {"p50_ms": 8}}}}
            owner._optimize(1)
            from proteus_realtime.protocol import InferRequest
            with patch.object(owner.cpu, "submit", side_effect=capture), \
                 patch("proteus_realtime.serving.adaptive_decision", return_value=BatchDecision("run", 1)) as decision:
                owner.submit(InferRequest(model="imagenet", image=b"jpeg", slo=200)).result(3)
            self.assertAlmostEqual(decision.call_args.args[0][0], captured[0].arrival + 0.194)
        finally:
            owner.close()

    def test_fixed_diagnostic_ignores_dispatch_reserve(self):
        owner = self.make()
        try:
            owner.set_mode("fixed")
            from proteus_realtime.protocol import InferRequest
            with patch("proteus_realtime.serving.adaptive_decision") as decision:
                reply = owner.submit(InferRequest(model="imagenet", image=b"jpeg", slo=200)).result(3)
            decision.assert_not_called()
            self.assertEqual(reply.timings["worker.rpc_budget_ms"], 0)
            self.assertEqual(reply.timings["worker.server_budget_ms"], 200)
            self.assertFalse(owner.state()["deadline_budget"]["applied_to_dispatch"])
        finally:
            owner.close()

    def test_cached_profiles_reused_without_remeasurement_or_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            first = self.make(0, path)
            try:
                first.profile(b"jpeg", trials=2)
            finally:
                first.close()
            before = path.read_bytes()
            with patch.object(FakeGpu, "infer_prepared", side_effect=AssertionError("must not reprofile")):
                second = self.make(6, path)
                try:
                    self.assertIsNotNone(second.plan)
                    self.assertIsNone(second.policy_error)
                    self.assertEqual(second.state()["deadline_budget"]["rpc_budget_ms"], 6)
                finally:
                    second.close()
            self.assertEqual(path.read_bytes(), before)


class SplitMetricsTests(unittest.TestCase):
    def rows(self):
        return [
            {"ok": True, "slo_ms": 200, "e2e_ms": 201, "server_ms": 196, "slo_met": False,
             "timings": {"worker.gpu_prediction_error": -1.5}},
            {"ok": True, "slo_ms": 200, "e2e_ms": 199, "server_ms": 195, "slo_met": True,
             "timings": {"worker.gpu_prediction_error": 0.8}},
            {"ok": False, "slo_ms": 200, "e2e_ms": 10, "server_ms": 0, "slo_met": False},
        ]

    def test_scopes_are_distinct_and_failures_count_in_both(self):
        summary = summarize(self.rows(), 1)
        client = summary["slo_metrics"]["client_e2e"]
        server = summary["slo_metrics"]["server_internal"]
        self.assertEqual(client["slo_met_requests"], 1)
        self.assertEqual(server["slo_met_requests"], 2)
        self.assertEqual(client["slo_unmet_ratio_of_all_requests"], 2 / 3)
        self.assertEqual(server["slo_unmet_ratio_of_all_requests"], 1 / 3)
        self.assertEqual(server["latency"]["samples"], 2)
        self.assertEqual(summary["slo_missed_replies"], 1)
        self.assertEqual(summary["slo_met_replies"], 1)
        self.assertEqual(summary["slo_metrics"]["rpc_and_unmeasured_extra"]["mean_ms"], 4.5)

    def test_nonfinite_and_negative_server_times_do_not_make_successes(self):
        for value in (math.nan, math.inf, -1, "bad"):
            row = {**self.rows()[0], "server_ms": value}
            result = split_latency_metrics([row])["server_internal"]
            self.assertEqual(result["unknown_successful_slo_requests"], 1)
            self.assertIsNone(result["slo_unmet_ratio_of_all_requests"])

    def test_empty_split_does_not_report_zero_percent_as_measured(self):
        result = split_latency_metrics([])
        self.assertIsNone(result["client_e2e"]["slo_unmet_ratio_of_all_requests"])
        self.assertIsNone(result["server_internal"]["latency"]["mean_ms"])

    def test_exactly_at_slo_is_met(self):
        row = {**self.rows()[0], "server_ms": 200, "e2e_ms": 200}
        result = split_latency_metrics([row])
        self.assertEqual(result["client_e2e"]["slo_met_requests"], 1)
        self.assertEqual(result["server_internal"]["slo_met_requests"], 1)

    def test_unmeasured_pred_error_is_not_invented(self):
        result = split_latency_metrics([{**self.rows()[0], "timings": {}}])
        self.assertEqual(result["gpu_prediction_error"]["samples"], 0)
        self.assertIsNone(result["gpu_prediction_error"]["mean_ms"])

    def test_terminal_summary_omits_history_but_full_report_keeps_it(self):
        report = {**summarize(self.rows(), 1), "runtime_after": {
            "deadline_budget": {"rpc_budget_ms": 6}, "milp_history": ["not for terminal"]}}
        small = compact_bench_report(report)
        self.assertEqual(small["deadline_budget"]["rpc_budget_ms"], 6)
        self.assertNotIn("milp_history", json.dumps(small))
        self.assertIn("milp_history", report["runtime_after"])

    def test_saved_csv_readonly_summary_includes_original_server_times(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench-fixture.csv"
            rows = self.rows()
            fields = sorted(set().union(*(row.keys() for row in rows)))
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({**row, "timings": json.dumps(row.get("timings", {}))})
            before = path.read_bytes()
            with patch("sys.stdout", io.StringIO()):
                report = metrics_main(["--file", str(path)])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(report["slo_metrics"]["server_internal"]["slo_met_requests"], 2)
            self.assertEqual(report["requests"], 3)

    def test_unsent_trace_rows_are_not_reported_as_server_failures(self):
        rows = self.rows()[:1] + [{"ok": False, "sent": False}]
        result = split_latency_metrics(rows)
        self.assertEqual(result["server_internal"]["slo_unmet_requests"], 0)
        self.assertEqual(result["client_e2e"]["slo_unmet_requests"], 1)

    def test_bad_csv_flag_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("ok,e2e_ms\nprobably,3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ok flag"):
                read_csv_rows(path)

    def test_latest_file_selection_reports_the_selected_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stamp in ("20200101", "20200102"):
                (root / f"bench-{stamp}.csv").write_text(
                    "ok,e2e_ms,server_ms,slo_ms,slo_met\nTrue,201,196,200,False\n", encoding="utf-8")
            with patch("sys.stdout", io.StringIO()):
                report = metrics_main(["--results-dir", str(root)])
            self.assertTrue(report["source_csv"].endswith("bench-20200102.csv"))


if __name__ == "__main__":
    unittest.main()
