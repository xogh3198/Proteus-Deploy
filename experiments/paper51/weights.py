"""Acquire identified public source weights; never substitute or random-init.

ONNX conversion and GPU validation are distinct subsequent steps. A successful
download is not a claim that a model fits the GPU or reproduces paper accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import urllib.request
import zipfile

from .audit import read_catalog


SOURCES = Path(__file__).with_name("sources.json")


def read_sources(path=SOURCES):
    sources = json.loads(Path(path).read_text(encoding="utf-8"))
    models = sources["models"]
    catalog = {m["name"]: m for m in read_catalog()["models"]}
    if set(models) != set(catalog) or sources["precision"] != "fp16":
        raise ValueError("Source inventory must cover every selected FP16 model")
    for name, entry in models.items():
        if entry["group"] != catalog[name]["group"]:
            raise ValueError("Model family differs from the paper inventory")
        kind = entry["kind"]
        if kind == "huggingface":
            if (not re.fullmatch(r"[0-9a-f]{40}", entry["revision"])
                    or entry["weight_format"] not in {"safetensors", "pytorch-bin"}):
                raise ValueError("Pinned Hugging Face revision and explicit weight format required")
        elif kind == "gluoncv":
            if not re.fullmatch(r"[0-9a-f]{40}", entry["parameter_sha1"]):
                raise ValueError("GluonCV upstream parameter checksum required")
            expected = ("https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/"
                        f"{name}-{entry['parameter_sha1'][:8]}.zip")
            if entry["download_url"] != expected:
                raise ValueError("Unexpected GluonCV source URL")
        elif kind == "yolov5":
            if (entry["download_url"] !=
                    f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{name}.pt"
                    or entry["asset_size"] <= 0 or entry["asset_id"] <= 0):
                raise ValueError("Unexpected YOLOv5 release identity")
        else:
            raise ValueError(f"Unsupported model source: {kind}")
    return sources


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_id(name):
    # BERT names can contain '/'; every variant still gets one unique directory.
    return name.replace("/", "--")


def download(url, path, expected_size=None, opener=urllib.request.urlopen):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Will not replace existing source weights: " + str(path))
    request = urllib.request.Request(url, headers={"User-Agent": "proteus-paper51-weights/1"})
    with opener(request, timeout=60) as response, path.open("xb") as output:
        total = 0
        for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
            total += len(block)
            if total > 2 * 1024 ** 3:
                raise ValueError("Release archive exceeds the allowed 2 GiB size")
            output.write(block)
    if not total or (expected_size is not None and total != expected_size):
        raise ValueError("Source weight length differs from release metadata")


def get_gluon(entry, destination):
    name = entry["model_id"]
    filename = name + "-" + entry["parameter_sha1"][:8] + ".params"
    with tempfile.TemporaryDirectory(prefix="download-", dir=destination) as stage:
        archive = Path(stage) / "source.zip"
        download(entry["download_url"], archive)
        # Read exactly the expected member. Never extract arbitrary archive paths.
        with zipfile.ZipFile(archive) as source:
            matches = [m for m in source.infolist() if m.filename == filename]
            if len(matches) != 1 or not 0 < matches[0].file_size < 2 * 1024 ** 3:
                raise ValueError("Expected GluonCV parameter file missing from archive")
            output = Path(stage) / filename
            with source.open(matches[0]) as data, output.open("xb") as sink:
                for block in iter(lambda: data.read(8 * 1024 * 1024), b""):
                    sink.write(block)
        if digest(output, "sha1") != entry["parameter_sha1"]:
            raise ValueError("GluonCV pretrained parameter checksum mismatch")
        target = destination / filename
        if target.exists():
            raise FileExistsError(target)
        output.rename(target)
    return {filename: {"sha256": digest(target), "bytes": target.stat().st_size}}


def get_yolo(entry, destination):
    # Pin the original numeric asset identity as well as its release URL/size.
    url = f"https://api.github.com/repos/ultralytics/yolov5/releases/assets/{entry['asset_id']}"
    with urllib.request.urlopen(url, timeout=60) as response:
        asset = json.load(response)
    if (asset["browser_download_url"] != entry["download_url"]
            or asset["size"] != entry["asset_size"]):
        raise ValueError("YOLOv5 release asset changed")
    filename = entry["model_id"] + ".pt"
    with tempfile.TemporaryDirectory(prefix="download-", dir=destination) as stage:
        path = Path(stage) / filename
        download(entry["download_url"], path, entry["asset_size"])
        target = destination / filename
        if target.exists():
            raise FileExistsError(target)
        path.rename(target)
    return {filename: {"sha256": digest(target), "bytes": target.stat().st_size}}


def get_hf(entry, destination):
    from huggingface_hub import snapshot_download
    suffix = "*.safetensors" if entry["weight_format"] == "safetensors" else "pytorch_model*.bin"
    snapshot_download(repo_id=entry["model_id"], revision=entry["revision"],
                      local_dir=destination,
                      allow_patterns=["*.json", "*.model", "*.txt", "*.tiktoken", suffix])
    files = [p for p in destination.rglob("*") if p.is_file()
             and ".cache" not in p.relative_to(destination).parts]
    if not any(p.suffix in {".bin", ".safetensors"} for p in files):
        raise ValueError("No real pretrained weights downloaded")
    return {p.relative_to(destination).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in files}


def verify_model(path, expected, source):
    if expected.get("source") != source:
        raise ValueError("Stored checkpoint source differs from selected revision")
    files = expected.get("files", {})
    if not files:
        raise ValueError("Source checkpoint manifest has no weight files")
    for name, metadata in files.items():
        relative = Path(name)
        if (not name or relative.anchor or name.startswith("/") or ".." in relative.parts
                or "\\" in name or ":" in name):
            raise ValueError("Unsafe artifact name")
        artifact = path / relative
        if (not artifact.is_file() or artifact.stat().st_size != metadata["bytes"]
                or digest(artifact) != metadata["sha256"]):
            raise ValueError(f"Source checkpoint content changed: {artifact}")


def acquire_model(name, entry, root):
    path = root / source_id(name)
    manifest = path / "checkpoint.lock.json"
    if manifest.exists():
        expected = json.loads(manifest.read_text(encoding="utf-8"))
        verify_model(path, expected, entry)
        return expected
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Unverified partial model directory: {path}; keep it for inspection and choose a new root")
    path.mkdir(parents=True, exist_ok=True)
    files = {"gluoncv": get_gluon, "yolov5": get_yolo, "huggingface": get_hf}[entry["kind"]](entry, path)
    result = {"schema": "paper51-checkpoint-lock-v1", "name": name,
              "source": entry, "files": files}
    # Commit the source lock only after the whole model is present and verified.
    with manifest.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "download", "verify"])
    parser.add_argument("--group", default="all")
    parser.add_argument("--root", type=Path, default=Path("data/paper51/source-weights"))
    args = parser.parse_args(argv)
    models = read_sources()["models"]
    if args.group != "all" and args.group not in {m["group"] for m in models.values()}:
        parser.error("Unknown paper model group")
    selected = {n: m for n, m in models.items() if args.group in {"all", m["group"]}}
    if args.action == "list":
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return
    args.root.mkdir(parents=True, exist_ok=True)
    for name, entry in selected.items():
        if args.action == "verify":
            path = args.root / source_id(name)
            expected = json.loads((path / "checkpoint.lock.json").read_text(encoding="utf-8"))
            verify_model(path, expected, entry)
        else:
            acquire_model(name, entry, args.root)
        print(f"{name}: source checkpoint verified", flush=True)


if __name__ == "__main__":
    main()
