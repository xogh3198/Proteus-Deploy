"""Real Gurobi + real gRPC, with a clearly synthetic GPU backend (NO V100)."""
import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import grpc

from proteus_baselines import BASELINES, SUPPORTED_BASELINES, deploy
from proteus_baselines.client import cold_warm, main as client_main, offline_accuracy_summary
from proteus_baselines.config import ROOT, SOMMELIER_SOURCES, compose_document, settings, verify_sources
from proteus_baselines.sommelier import SommelierExecutor, solve_fixed_placement, verify_class_mapping
from proteus_realtime.client import admin_call, query_once
from proteus_realtime.protocol import InferRequest
from proteus_realtime.server import create_server


class FakeGpu:
    model_name, model_id, application, gpu_type = "resnet-50", "microsoft/resnet-50", "imagenet", "FAKE-NOT-V100"

    def __init__(self, multiple=True, scores=True):
        self.registry = {"resnet-50": {"model_id": self.model_id, "revision": "a" * 40,
                         "accuracy": 90 if scores else None,
                         "accuracy_source": "unit-fixture-only", "accuracy_dataset": "unit-imagenet-top1"}}
        if multiple:
            self.registry["resnet-18"] = {"model_id": "microsoft/resnet-18", "revision": "b" * 40,
                         "accuracy": 75 if scores else None,
                         "accuracy_source": "unit-fixture-only", "accuracy_dataset": "unit-imagenet-top1"}
        self.enabled = list(self.registry)
        self.current, self.loads, self.forwards = None, [], []

    def load(self, name=None):
        name = name or self.model_name
        loaded = self.current != name
        if loaded:
            self.loads.append(name)
        self.current = name
        return {"loaded": loaded, "load_ms": 2.0 if loaded else 0.0, "ready_ms": 1, "stages": {}}

    def infer_batch(self, images):
        if b"broken" in images:
            raise ValueError("synthetic image failure")
        self.forwards.append((self.current, len(images)))
        time.sleep(0.003)
        label = 0 if self.current == "resnet-50" else 1
        return [{"label": "UNIT-LABEL", "label_id": label, "image_size": (20, 20),
                 "timings": {"forward": 3.0}} for _ in images]

    def infer(self, image):
        return self.infer_batch([image])[0]

    def prepare(self, image):
        return {"raw": image}

    def infer_prepared(self, items):
        return self.infer_batch([item["raw"] for item in items])

    def reset(self):
        self.current = None

    def fingerprint(self):
        return {"gpu": "UNIT-FAKE", "checkpoints": {n: r["revision"] for n, r in self.registry.items()}}

    def describe(self):
        return {"resident": self.current is not None, "enabled_models": self.enabled,
                "checkpoint_commit": self.registry[self.current or self.model_name]["revision"]}


class ConfigTests(unittest.TestCase):
    def test_new_baseline_never_changes_existing_all_compare(self):
        self.assertEqual(BASELINES, ("infaas-accuracy", "clipper-ha", "clipper-ht"))
        self.assertIn("sommelier", SUPPORTED_BASELINES)

    def test_isolation_and_default_single_model_smoke(self):
        c = settings("sommelier")
        self.assertEqual(c["PORT"], "19084")
        self.assertEqual(c["WORKER_IMAGE"], "proteus-baseline-worker:sommelier-cpu-ready-v1")
        self.assertTrue(c["MODEL_DIR"].endswith(str(Path("sommelier/models"))))
        document = compose_document(c, "0:0")["services"]["worker"]
        self.assertEqual(document["environment"]["BASELINE"], "sommelier")
        self.assertEqual(document["environment"]["EVAL_DIR"], "/eval")
        self.assertTrue(next(v for v in document["volumes"] if v["target"] == "/eval")["read_only"])
        self.assertEqual(document["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"], ["2"])

    def test_multimodel_registry_does_not_require_fabricated_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = json.loads((ROOT / "deploy/docker/baselines/sommelier-models.json").read_text())
            registry["models"]["resnet-18"]["enabled"] = True
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(registry))
            with patch.dict(os.environ, {"SOMMELIER_MODEL_REGISTRY": str(path)}):
                self.assertEqual(settings("sommelier")["MODEL_REGISTRY"], str(path.resolve()))

    def test_sommelier_local_settings_do_not_read_singleton_baseline_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "wrong.env"
            local.write_text("MODEL_NAME=unknown\n")
            with patch.dict(os.environ, {"BASELINE_ENV": str(local)}):
                self.assertEqual(settings("sommelier")["MODEL_NAME"], "resnet-50")

    def test_sommelier_prefix_overrides_common_baseline_prefix(self):
        with patch.dict(os.environ, {"BASELINE_GPU": "0", "SOMMELIER_GPU": "3"}):
            self.assertEqual(settings("sommelier")["GPU"], "3")

    def test_source_hashes_cover_original_fixed_application_constraints(self):
        hashes = verify_sources(baseline="sommelier")
        for key, value in SOMMELIER_SOURCES.items():
            self.assertEqual(hashes[key], value)

    def test_download_fetches_all_enabled_not_only_default_name(self):
        with patch.object(deploy, "oneoff") as run:
            self.assertEqual(deploy.main(["sommelier", "download"]), 0)
            self.assertEqual(run.call_args.args[2:], ("weights",))

    def test_evaluate_is_scoped_and_manifest_passes_to_client(self):
        with patch.object(deploy, "client") as run:
            run.return_value.returncode = 0
            self.assertEqual(deploy.main(["sommelier", "evaluate", "--manifest", "validation.json", "--timeout", "7200"]), 0)
            self.assertEqual(run.call_args.kwargs["options"], ["--manifest", "validation.json", "--timeout", "7200"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            deploy.main(["clipper-ha", "evaluate"])

    def test_license_mount_is_readonly(self):
        with tempfile.TemporaryDirectory() as directory:
            licence = Path(directory) / "private.lic"
            licence.write_text("NOT-A-REAL-LICENCE")
            with patch.dict(os.environ, {"SOMMELIER_GUROBI_LICENSE_FILE": str(licence)}):
                worker = compose_document(settings("sommelier"), "0:0")["services"]["worker"]
            self.assertTrue(next(v for v in worker["volumes"] if v["target"] == "/license/gurobi.lic")["read_only"])


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.resources = []

    def tearDown(self):
        for executor, server, channel in reversed(self.resources):
            channel.close()
            server.stop(0).wait(2)
            executor.close()
        self.temp.cleanup()

    def make(self, multiple=True, scores=True):
        backend = FakeGpu(multiple, scores)
        executor = SommelierExecutor(backend, batches=(1, 2), slo=1000, interval=30,
             guard_ms=50, profile_path=Path(self.temp.name) / "profile.json")
        server, port = create_server(executor, "127.0.0.1:0", 64)
        server.start()
        channel = grpc.insecure_channel("127.0.0.1:" + str(port))
        grpc.channel_ready_future(channel).result(2)
        self.resources.append((executor, server, channel))
        return executor, channel, port

    def fixture_profiles(self, executor):
        executor.profiles = {"resnet-50": {"batches": {"1": {"p50_ms": 20}, "2": {"p50_ms": 30}}},
                             "resnet-18": {"batches": {"1": {"p50_ms": 5}, "2": {"p50_ms": 6}}}}

    def test_missing_scores_blocks_native_requests_but_retains_actual_profiles(self):
        executor, channel, _ = self.make(scores=False)
        report = admin_call(channel, "Profile", {"image_base64": "anBlZw==", "trials": 1})
        self.assertFalse(report["policy_ready"])
        self.assertEqual(set(report["profiles"]), {"resnet-50", "resnet-18"})
        self.assertFalse(query_once(channel, "imagenet", b"jpeg", 1000, 2)["ok"])

    def test_real_solver_low_high_demand_selects_and_executes_different_models(self):
        executor, channel, _ = self.make()
        self.fixture_profiles(executor)
        for rate, selected in ((20, "resnet-50"), (200, "resnet-18")):
            executor._optimize(rate)
            # Avoid timer scheduling in this selection/real completion test.
            with patch("proteus_realtime.serving.adaptive_decision") as decision:
                from proteus_realtime.policy import BatchDecision
                decision.return_value = BatchDecision("run", 1)
                row = query_once(channel, "imagenet", b"jpeg", 1000, 2)
            self.assertTrue(row["ok"])
            self.assertEqual(row["variant"], selected)
        state = executor.state()
        self.assertEqual(state["placement"], {"device": "cuda:0", "application": "imagenet", "fixed": True})
        self.assertEqual(state["actual_variant_switches"], 1)
        self.assertGreater(state["switch_load_ms_total"], 0)
        self.assertEqual(state["successful_native_requests_by_model"], {"resnet-50": 1, "resnet-18": 1})

    def test_profile_actual_backend_and_cold_warm_restore_native(self):
        executor, channel, _ = self.make(multiple=False)
        self.assertTrue(executor.profile(b"jpeg", 1)["policy_ready"])
        args = argparse.Namespace(model="imagenet", slo=1000, timeout=3, warm_requests=2,
                                  baseline="sommelier", output=self.temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cold_warm(channel, args, b"jpeg"), 0)
        self.assertEqual(executor.state()["mode"], "native")
        self.assertFalse(executor.state()["model_scaling_enabled"])
        self.assertEqual(executor.state()["successful_native_requests_by_model"], {})

    def test_offline_evaluation_measures_both_models_on_same_dataset(self):
        executor, channel, _ = self.make(scores=False)
        root = Path(self.temp.name)
        rows = []
        for index, label in enumerate((0, 1, 0)):
            name = str(index) + ".jpg"
            (root / name).write_bytes(b"unit-not-real-jpeg")
            rows.append({"path": name, "label": label})
        (root / "manifest.json").write_text(json.dumps(rows))
        labels = {str(i): "unit-label-" + str(i) for i in range(1000)}
        for item in executor.backend.registry.values():
            cache_path = root / ("hub/models--" + item["model_id"].replace("/", "--")) / "snapshots" / item["revision"]
            cache_path.mkdir(parents=True)
            (cache_path / "config.json").write_text(json.dumps({"id2label": labels}))
        with patch.dict(os.environ, {"EVAL_DIR": str(root), "HF_HOME": str(root)}):
            result = admin_call(channel, "Evaluate", {"manifest": "manifest.json"})
        scores = result["accuracy"]
        self.assertAlmostEqual(scores["resnet-50"]["accuracy"], 200 / 3)
        self.assertAlmostEqual(scores["resnet-18"]["accuracy"], 100 / 3)
        self.assertEqual(scores["resnet-50"]["accuracy_dataset"], scores["resnet-18"]["accuracy_dataset"])
        self.assertIsNone(executor.state()["classification_accuracy"])
        self.assertTrue(executor.profile(b"jpeg", 1)["policy_ready"])

    def test_native_bench_records_actual_model_counts_and_offline_weighted_score(self):
        executor, _, port = self.make(multiple=False)
        executor.profile(b"jpeg", 1)
        image = Path(self.temp.name) / "fixture.jpg"
        image.write_bytes(b"unit-not-real-jpeg")
        with contextlib.redirect_stdout(io.StringIO()):
            status = client_main(["bench", "--baseline", "sommelier", "--controller", "127.0.0.1:" + str(port),
                "--image", str(image), "--slo", "1000", "--rate", "6", "--duration", "0.2",
                "--warmup", "1", "--timeout", "3", "--output", self.temp.name])
        self.assertEqual(status, 0)
        report = json.loads(next(Path(self.temp.name).glob("bench-*.json")).read_text())
        self.assertEqual(report["effective_offline_top1_pct"], 90)
        self.assertEqual(report["successful_requests_by_model"], {"resnet-50": 2})
        self.assertEqual(report["runtime_after"]["baseline"], "sommelier")
        self.assertEqual(report["scheduled_requests"], 2)

    def test_explicit_other_dispatch_is_rejected_without_mutation(self):
        executor, _, _ = self.make()
        request = InferRequest(model="imagenet", image=b"jpeg", slo=1000, dispatch="clipper-ha")
        with self.assertRaises(ValueError):
            executor.submit(request)
        self.assertEqual(request.dispatch, "clipper-ha")

    def test_client_rejects_wrong_endpoint_before_maintenance(self):
        _, _, port = self.make()
        with self.assertRaisesRegex(ValueError, "differs"):
            client_main(["state", "--baseline", "clipper-ha", "--controller", "127.0.0.1:" + str(port)])

    def test_same_single_gpu_inputs_are_equivalent_to_proteus_not_fake_difference(self):
        from proteus_realtime.policy import solve_milp
        fixture = {"high": {"batches": {"1": {"p50_ms": 20}}}, "low": {"batches": {"1": {"p50_ms": 5}}}}
        scores = {"high": 100, "low": 80}
        for rate in (20, 100):
            a, b = solve_fixed_placement(fixture, scores, rate, 200), solve_milp(fixture, scores, rate, 200)
            self.assertEqual((a.model, a.max_batch, a.capacity_qps, a.beta), (b.model, b.max_batch, b.capacity_qps, b.beta))

    def test_zero_or_invalid_accuracy_never_becomes_measured_100(self):
        executor, _, _ = self.make(scores=False)
        self.fixture_profiles(executor)
        executor._optimize(1)
        self.assertIsNone(executor.plan)
        self.assertIsNone(executor.state()["normalized_model_accuracy"])


class ReportTests(unittest.TestCase):
    def test_class_mapping_mismatch_fails_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = {}
            for name, prefix in (("first", "label"), ("second", "different-label")):
                registry[name] = {"model_id": "unit/" + name, "revision": "a" * 40}
                path = Path(directory) / ("hub/models--unit--" + name) / "snapshots" / ("a" * 40)
                path.mkdir(parents=True)
                (path / "config.json").write_text(json.dumps({"id2label": {str(i): prefix + str(i) for i in range(1000)}}))
            with self.assertRaisesRegex(ValueError, "different class-ID"):
                verify_class_mapping(registry, directory)

    def test_new_document_links_commands_and_configuration_agree(self):
        import re
        text = (ROOT / "SOMMELIER_GPU.md").read_text(encoding="utf-8")
        self.assertEqual(text.count("```") % 2, 0)
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" not in target:
                self.assertTrue((ROOT / target.split("#")[0]).is_file(), target)
        self.assertIn("SOMMELIER_GPU.md", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("19084", text)
        self.assertIn("같은 프로파일", text)

    def test_effective_accuracy_weights_only_successful_observed_model_replies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["ok", "variant"])
                writer.writeheader()
                writer.writerows([{"ok": True, "variant": "resnet-50"}] * 3 +
                                 [{"ok": True, "variant": "resnet-18"}, {"ok": False, "variant": ""}])
            report = offline_accuracy_summary(path, {"offline_accuracy": {
                "resnet-50": {"accuracy": 90}, "resnet-18": {"accuracy": 75}}})
        self.assertEqual(report["effective_offline_top1_pct"], 86.25)
        self.assertEqual(sum(report["successful_requests_by_model"].values()), 4)

    def test_missing_offline_accuracy_stays_unmeasured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            path.write_text("ok,variant\nTrue,resnet-50\n")
            report = offline_accuracy_summary(path, {"offline_accuracy": {"resnet-50": {"accuracy": None}}})
        self.assertIsNone(report["effective_offline_top1_pct"])


if __name__ == "__main__":
    unittest.main()
