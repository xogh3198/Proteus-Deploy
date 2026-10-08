"""Exercise Bash command routing with a fake Docker executable, never a GPU.

Run from Linux or Windows with Git Bash:
    python tests_gpu/smoke_scripts.py --bash /path/to/bash
Temporary helper scripts are test fixtures and are removed with their directory.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shell_path(path):
    value = Path(path).as_posix()
    if len(value) > 2 and value[1] == ":":
        return "/" + value[0].lower() + value[2:]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bash", default="bash")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        fixtures = Path(directory)
        # Docker is mocked: these tests validate argv/routing, not Docker startup.
        (fixtures / "python3").write_text(
            '#!/bin/sh\nexec "$SMOKE_PYTHON" "$@"\n', encoding="utf-8")
        (fixtures / "realpath").write_text(
            '#!/bin/sh\nexec "$SMOKE_PYTHON" -c \'import os,sys; '
            'print(os.path.realpath(sys.argv[1]))\' "$@"\n', encoding="utf-8")
        (fixtures / "docker").write_text(
            '#!/bin/sh\n'
            '"$SMOKE_PYTHON" -c \'import json,os,sys; '
            'f=open(os.environ["SMOKE_LOG"],"a"); '
            'f.write(json.dumps(sys.argv[1:])+"\\n")\' "$@"\n'
            'if [ "$1" = inspect ]; then printf "running|healthy\\n"; fi\n'
            'if [ "$1" = image ] && [ "$2" = inspect ]; then printf "standalone-v1\\n"; fi\n',
            encoding="utf-8")
        for helper in (fixtures / "python3", fixtures / "docker", fixtures / "realpath"):
            helper.chmod(0o755)
        env_file = fixtures / "smoke.env"
        env_file.write_text(
            f"MODEL_DIR={fixtures.as_posix()}/models\nDATA_DIR={fixtures.as_posix()}/data\n"
            "ALLOW_GPU_SHARING=1\n",  # mock routing only; never do this for measurements
            encoding="utf-8")
        image = fixtures / "input.jpg"
        image.write_bytes(b"FAKE image: only shell routing is tested")
        source = fixtures / "source"
        source.mkdir()
        env = dict(os.environ, PROTEUS_ENV=str(env_file), SMOKE_PYTHON=shell_path(sys.executable),
                   SMOKE_LOG=str(fixtures / "docker.jsonl"))
        env["PATH"] = str(fixtures) + os.pathsep + str(Path(args.bash).resolve().parent) + os.pathsep + env["PATH"]
        script = "deploy/docker/proteus-docker.sh"
        commands = [
            ["help"], ["settings"], ["build"], ["doctor"],
            ["download", "--models", "resnet-50"],
            ["seed", "--from", str(source), "--models", "resnet-50"],
            ["query", "resnet-50", str(image), "200", "-n", "11"],
            ["test", str(image), "--warm-requests", "10"],
            ["register", "--models", "resnet-50", "--image", str(image), "--trials", "3"],
            ["bench", str(image), "--rate", "10", "--duration", "0.1"],
            ["policy-test", str(image)], ["mode", "--mode", "proteus"],
            ["evaluate", "--manifest", "manifest.json"],
            ["up"], ["ps"], ["state", "--watch", "2"], ["reset"],
            ["logs", "--follow"], ["down"],
        ]
        for command in commands:
            result = subprocess.run([args.bash, script, *command], cwd=ROOT, env=env,
                                    text=True, encoding="utf-8", errors="replace",
                                    capture_output=True, timeout=30)
            if result.returncode:
                raise RuntimeError(f"{command}: {result.stdout}\n{result.stderr}")
        invalid = subprocess.run([args.bash, script, "register", "--bogus"],
                                 cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace",
                                 capture_output=True, timeout=30)
        if invalid.returncode == 0:
            raise RuntimeError("Unsupported profile option was not rejected")
        logged = [json.loads(line) for line in (fixtures / "docker.jsonl").read_text().splitlines()]
        calls = [call for call in logged if call and call[0] == "run"]
        if not all("--pull=never" in call for call in calls):
            raise RuntimeError("Unexpected automatic image pull")
        if not any("proteus_realtime.client" in call and "test" in call for call in calls):
            raise RuntimeError("Cold/warm command not routed to the client")
        if not any("proteus_realtime.client" in call and "policy-test" in call for call in calls):
            raise RuntimeError("Policy verification not routed to the client")
        if any("infaas-worker" in str(call) for call in logged):
            raise RuntimeError("Unexpected INFaaS runtime/image dependency")
        print(f"PASS: {len(commands)} Bash routes and bad-option rejection; Docker mocked.")


if __name__ == "__main__":
    main()
