"""Daemon-aware UID selection, without changing context or NVIDIA settings."""
import unittest
from unittest.mock import patch

from proteus_realtime.deployment import container_user, compose_document, settings


class DockerUserTests(unittest.TestCase):
    def test_rootless_maps_host_owned_mounts_to_container_root(self):
        with patch("proteus_realtime.deployment.subprocess.check_output",
                   return_value='["name=seccomp,profile=builtin", "name=rootless"]') as info:
            self.assertEqual(container_user(), "0:0")
        info.assert_called_once_with(
            ["docker", "info", "--format", "{{json .SecurityOptions}}"], text=True)

    def test_rootful_keeps_login_user(self):
        with patch("proteus_realtime.deployment.subprocess.check_output", return_value="[]"), \
             patch("proteus_realtime.deployment.os.getuid", return_value=1234, create=True), \
             patch("proteus_realtime.deployment.os.getgid", return_value=5678, create=True):
            self.assertEqual(container_user(), "1234:5678")

    def test_unknown_security_mode_is_not_guessed(self):
        for value in ('null', '{}', '[123]', 'not-json'):
            with self.subTest(value=value), \
                 patch("proteus_realtime.deployment.subprocess.check_output", return_value=value):
                with self.assertRaises(ValueError):
                    container_user()

    def test_compose_user_does_not_change_gpu_or_readonly_weights(self):
        values = settings()
        service = compose_document(values, "0:0")["services"]["worker"]
        self.assertEqual(service["user"], "0:0")
        device = service["deploy"]["resources"]["reservations"]["devices"][0]
        self.assertEqual(device["device_ids"], [values["GPU"]])
        model_mount = next(v for v in service["volumes"] if v["target"] == "/models")
        self.assertTrue(model_mount["read_only"])


if __name__ == "__main__":
    unittest.main()
