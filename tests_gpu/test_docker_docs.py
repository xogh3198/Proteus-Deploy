"""Keep the Korean quickstart aligned with the shipped script and defaults."""
import re
import unittest
from pathlib import Path

from proteus_realtime.deployment import settings


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "docker.md", "deploy/docker/README.md", "GPU_SERVING.md")


class DockerDocsTests(unittest.TestCase):
    def test_code_fences_and_local_links(self):
        for name in DOCUMENTS:
            path = ROOT / name
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("```") % 2, 0, name)
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                self.assertTrue((path.parent / target.split("#")[0]).exists(), f"{name}: {target}")

    def test_all_documented_script_commands_exist(self):
        script = (ROOT / "deploy/docker/proteus-docker.sh").read_text(encoding="utf-8")
        commands = set()
        for name in DOCUMENTS:
            commands.update(re.findall(r"^bash deploy/docker/proteus-docker\.sh ([\w-]+)",
                                       (ROOT / name).read_text(encoding="utf-8"), re.MULTILINE))
        for command in commands:
            self.assertRegex(script, r"(?:^|[|\s])" + re.escape(command) + r"(?:\||\))")

    def test_quickstart_matches_default_experiment(self):
        values = settings()
        for key, expected in {"GPU": "2", "GPU_TYPE": "v100", "MODEL_NAME": "resnet-50",
                              "APPLICATION": "imagenet", "SLO_MS": "200", "PORT": "18081"}.items():
            self.assertEqual(values[key], expected)
        text = (ROOT / "docker.md").read_text(encoding="utf-8")
        self.assertIn("query imagenet inputs/test-dog.jpg 200 -n 11", text)
        self.assertIn("register --models resnet-50 --trials 30", text)

    def test_script_does_not_select_a_daemon(self):
        script = (ROOT / "deploy/docker/proteus-docker.sh").read_text(encoding="utf-8")
        self.assertNotIn("export DOCKER_CONTEXT", script)
        self.assertNotIn("export DOCKER_HOST", script)
        self.assertNotIn("docker context use", script)
        self.assertNotIn("--context", script)
        self.assertNotIn("nvidia-ctk", script)
        self.assertNotIn("rootless-setup", script)


if __name__ == "__main__":
    unittest.main()
