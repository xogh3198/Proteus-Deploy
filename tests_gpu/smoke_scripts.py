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
            '#!/bin/sh\nexec "$SMOKE_PYTHON" "$SMOKE_HOST_WRAPPER" "$@"\n', encoding="utf-8")
        # A portable host adapter mocks only Docker info. Windows cannot launch
        # the shell Docker fixture via Python's native CreateProcess API.
        host_wrapper = fixtures / "host_python.py"
        host_wrapper.write_text(
            'import json,os,subprocess,sys\n'
            'sys.path.insert(0,os.getcwd())\n'
            'if sys.argv[1:3] == ["-m","proteus_realtime.metrics"]:\n'
            '    from proteus_realtime.metrics import main\n'
            '    main(sys.argv[3:])\n'
            '    raise SystemExit(0)\n'
            'if sys.argv[1:3] != ["-m","proteus_realtime.deployment"]:\n'
            '    raise RuntimeError("Unexpected host Python invocation")\n'
            'from proteus_realtime import deployment\n'
            'original=deployment.subprocess.check_output\n'
            'def mock_info(args,**kwargs):\n'
            '    if args == ["docker","info","--format","{{json .SecurityOptions}}"]:\n'
            '        with open(os.environ["SMOKE_LOG"],"a") as stream:\n'
            '            stream.write(json.dumps(args[1:])+"\\n")\n'
            '        if os.environ.get("SMOKE_DOCKER_INFO_FAIL") == "1":\n'
            '            raise subprocess.CalledProcessError(1,args)\n'
            '        return json.dumps(["name=rootless"] if os.environ.get("SMOKE_ROOTLESS") == "1" else [])\n'
            '    return original(args,**kwargs)\n'
            'deployment.subprocess.check_output=mock_info\n'
            'sys.argv=["proteus_realtime.deployment",*sys.argv[3:]]\n'
            'deployment.main()\n', encoding="utf-8")
        (fixtures / "realpath").write_text(
            '#!/bin/sh\nexec "$SMOKE_PYTHON" -c \'import os,sys; '
            'print(os.path.realpath(sys.argv[1]))\' "$@"\n', encoding="utf-8")
        (fixtures / "docker").write_text(
            '#!/bin/sh\n'
            '"$SMOKE_PYTHON" -c \'import json,os,sys; '
            'f=open(os.environ["SMOKE_LOG"],"a"); '
            'f.write(json.dumps(sys.argv[1:])+"\\n")\' "$@"\n'
            'if [ "$1" = info ]; then\n'
            '  if [ "${SMOKE_ROOTLESS:-0}" = 1 ]; then printf \'["name=rootless"]\\n\'; '
            'else printf \'[]\\n\'; fi\n'
            'fi\n'
            'if [ "$1" = inspect ]; then printf "running|healthy\\n"; fi\n'
            'if [ "$1" = image ] && [ "$2" = inspect ]; then printf "standalone-v1\\n"; fi\n',
            encoding="utf-8")
        for helper in (fixtures / "python3", fixtures / "docker", fixtures / "realpath"):
            helper.chmod(0o755)
        env_file = fixtures / "smoke.env"
        env_file.write_text(
            f"MODEL_DIR={fixtures.as_posix()}/models\nDATA_DIR={fixtures.as_posix()}/data\n"
            f"INPUT_IMAGE={(fixtures / 'input.jpg').as_posix()}\nPREPROCESS_WORKERS=4\n"
            "ALLOW_GPU_SHARING=1\n",  # mock routing only; never do this for measurements
            encoding="utf-8")
        image = fixtures / "input.jpg"
        image.write_bytes(b"FAKE image: only shell routing is tested")
        source = fixtures / "source"
        source.mkdir()
        env = dict(os.environ, PROTEUS_ENV=str(env_file), SMOKE_PYTHON=shell_path(sys.executable),
                   SMOKE_LOG=str(fixtures / "docker.jsonl"), SMOKE_HOST_WRAPPER=shell_path(host_wrapper))
        env["PATH"] = str(fixtures) + os.pathsep + str(Path(args.bash).resolve().parent) + os.pathsep + env["PATH"]
        script = "deploy/docker/proteus-docker.sh"
        result_dir = fixtures / "data/results"
        result_dir.mkdir(parents=True)
        saved_csv = result_dir / "bench-20000101.csv"
        saved_csv.write_text("ok,e2e_ms,server_ms,slo_ms,slo_met\nTrue,201,196,200,False\n", encoding="utf-8")
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
            ["bench-summary"], ["bench-summary", "--file", str(saved_csv)],
            ["preprocess-check"],
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
        checks = [call for call in calls if "proteus_realtime.check_preprocessing" in call]
        if not checks or any("--gpus" in call or call[call.index("--network") + 1] != "none"
                             or call[call.index("--workers") + 1] != "4" for call in checks):
            raise RuntimeError("CPU tensor check must be offline, GPU-free and honor worker count")
        if any("infaas-worker" in str(call) for call in logged):
            raise RuntimeError("Unexpected INFaaS runtime/image dependency")
        rootful_uid = f"{getattr(os, 'getuid', lambda: 0)()}:{getattr(os, 'getgid', lambda: 0)()}"
        if any(call[call.index("--user") + 1] != rootful_uid for call in calls):
            raise RuntimeError("Rootful routes did not preserve the login UID")
        offset = len(logged)
        rootless_commands = [commands[i] for i in (3, 4, 5, 6, 7, 8, 13)] + [commands[-1]]
        rootless_env = dict(env, SMOKE_ROOTLESS="1")
        for command in rootless_commands:
            result = subprocess.run([args.bash, script, *command], cwd=ROOT, env=rootless_env,
                                    text=True, encoding="utf-8", errors="replace",
                                    capture_output=True, timeout=30)
            if result.returncode:
                raise RuntimeError(f"rootless {command}: {result.stdout}\n{result.stderr}")
        rootless_log = [json.loads(line) for line in (fixtures / "docker.jsonl").read_text().splitlines()][offset:]
        rootless_calls = [call for call in rootless_log if call and call[0] == "run"]
        if not rootless_calls or any(call[call.index("--user") + 1] != "0:0" for call in rootless_calls):
            raise RuntimeError("Rootless bind-mount UID mapping was not applied")
        compose = json.loads((ROOT / "deploy/docker/generated/docker-compose.yml").read_text())
        if compose["services"]["worker"]["user"] != "0:0":
            raise RuntimeError("Rootless worker UID mapping was not applied")
        if any(call[:2] == ["context", "use"] or "--context" in call or "--host" in call
               for call in logged + rootless_log):
            raise RuntimeError("Script changed or forced Docker connection settings")
        offset = len(logged) + len(rootless_log)
        failure = subprocess.run([args.bash, script, "download", "--models", "resnet-50"],
                                 cwd=ROOT, env=dict(env, SMOKE_DOCKER_INFO_FAIL="1"),
                                 text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=30)
        failure_log = [json.loads(line) for line in (fixtures / "docker.jsonl").read_text().splitlines()][offset:]
        if failure.returncode == 0 or any(call and call[0] == "run" for call in failure_log):
            raise RuntimeError("Container executed after daemon mode detection failed")
        print(f"PASS: {len(commands) + len(rootless_commands)} Bash routes, bad-option rejection and fail-closed UID; Docker mocked.")


if __name__ == "__main__":
    main()
