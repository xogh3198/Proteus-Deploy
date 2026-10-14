"""Settings propagation and cached-profile invalidation; no physical GPU used."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proteus_realtime.deployment import settings, compose_document
from proteus_realtime.registry import load_registry
from proteus_realtime.serving import ProteusExecutor
from proteus_baselines import deploy
from proteus_baselines.config import settings as baseline_settings, compose_document as baseline_compose
from proteus_baselines.serving import BaselineExecutor

ROOT = Path(__file__).resolve().parents[1]
REVISION = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
MODEL = {"MODEL_NAME": "resnet-18", "MODEL_ID": "microsoft/resnet-18", "MODEL_REVISION": REVISION}


class SettingsWorkflowTests(unittest.TestCase):
    def test_proteus_model_gpu_batches_reach_container(self):
        with patch.dict(os.environ, {**MODEL, "GPU": "3", "BATCH_SIZES": "1,2"}):
            c = settings()
            registry = load_registry(c["MODEL_REGISTRY"])
            worker = compose_document(c)["services"]["worker"]
        self.assertEqual(set(registry), {"resnet-18"})
        self.assertEqual(registry["resnet-18"]["revision"], REVISION)
        self.assertIsNone(registry["resnet-18"]["accuracy"])
        for key, value in MODEL.items():
            self.assertEqual(worker["environment"][key], value)
        self.assertEqual(worker["environment"]["BATCH_SIZES"], "1,2")
        self.assertEqual(worker["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"], ["3"])

    def test_all_baselines_select_same_env_checkpoint(self):
        overrides = {"BASELINE_" + k: v for k, v in MODEL.items()}
        with patch.dict(os.environ, {**overrides, "BASELINE_GPU_TYPE": "3090", "BASELINE_BATCH_SIZES": "1,2"}):
            for name in ("infaas-accuracy", "clipper-ha", "clipper-ht"):
                c = baseline_settings(name)
                env = baseline_compose(c, "0:0")["services"]["worker"]["environment"]
                self.assertEqual(env["MODEL_NAME"], "resnet-18")
                self.assertEqual(env["MODEL_ID"], "microsoft/resnet-18")
                self.assertEqual(env["MODEL_REVISION"], REVISION)
                self.assertEqual(env["GPU_TYPE"], "3090")
                self.assertEqual(env["BATCH_SIZES"], "1,2")

    def test_generic_model_env_cannot_override_baseline_host_selection(self):
        with patch.dict(os.environ, MODEL):
            self.assertEqual(baseline_settings("clipper-ha")["MODEL_NAME"], "resnet-50")

    def test_unpinned_or_incomplete_override_rejected(self):
        for change in ({"MODEL_REVISION": "main"}, {"MODEL_REVISION": ""},
                       {"MODEL_ID": "../weights"}, {"MODEL_ID": ""}):
            with self.subTest(change=change), patch.dict(os.environ, {**MODEL, **change}):
                with self.assertRaises(ValueError):
                    settings()

    def test_baseline_download_uses_selected_name(self):
        with patch.dict(os.environ, {"BASELINE_" + k: v for k, v in MODEL.items()}), \
                patch.object(deploy, "oneoff") as oneoff:
            self.assertEqual(deploy.main(["clipper-ht", "download"]), 0)
            self.assertEqual(oneoff.call_args.args[2:], ("weights", "--models", "resnet-18"))
            self.assertTrue(oneoff.call_args.kwargs["online"])

    def test_explicit_model_mismatch_is_not_silently_ignored(self):
        with patch.object(deploy, "oneoff") as oneoff:
            with self.assertRaises(SystemExit):
                deploy.main(["clipper-ha", "download", "--models", "resnet-18"])
            oneoff.assert_not_called()

    def test_batch_change_invalidates_both_cached_profiles_without_startup_failure(self):
        class Fake:
            enabled = ["resnet-50"]
            def fingerprint(self):
                return {"fixture": "not-a-real-gpu"}
        backend = Fake()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            batches = {str(n): {"p50_ms": 10.0} for n in (1, 2, 4, 8)}
            path.write_text(json.dumps({"fingerprint": backend.fingerprint(), "slo_ms": 200,
                          "batches": batches, "profiles": {"resnet-50": {"batches": batches}}}))
            for cls in (ProteusExecutor, BaselineExecutor):
                owner = object.__new__(cls)
                owner.backend, owner.profile_path, owner.batches, owner.slo = backend, path, (1, 2), 200
                owner.profiles = {}
                owner.profile_data = None
                if cls is ProteusExecutor:
                    owner._read_profiles()
                    self.assertEqual(owner.profiles, {})
                    self.assertIn("run register", owner.policy_error)
                else:
                    owner._read_profile()
                    self.assertIsNone(owner.profile_data)
                    self.assertIn("run register", owner.error)

    def test_new_setup_document_links_and_fences(self):
        import re
        path = ROOT / "setting_readme.md"
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count("```") % 2, 0)
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" not in target:
                self.assertTrue((ROOT / target.split("#")[0]).is_file(), target)
        self.assertIn("setting_readme.md", (ROOT / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
