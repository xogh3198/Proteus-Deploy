"""Explicit independent checkpoint download, or copy only specified local caches."""
import argparse
import json
import os
import shutil
from pathlib import Path

from .registry import load_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["weights", "seed", "sample"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--source", default="/source")
    args = parser.parse_args()
    if args.command == "sample":
        import urllib.request
        target = Path("/inputs/test-dog.jpg")
        if not target.exists():
            urllib.request.urlretrieve("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", target)
        print(target)
        return
    registry = load_registry(os.environ.get("MODEL_REGISTRY", "/config/models.json"))
    names = args.models or [n for n, item in registry.items() if item.get("enabled", True)]
    if any(n not in registry for n in names):
        raise ValueError("Unknown model name")
    cache = Path(os.environ.get("HF_HOME", "/models")) / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    for name in names:
        item = registry[name]
        directory_name = "models--" + item["model_id"].replace("/", "--")
        if args.command == "seed":
            source = Path(args.source)
            source = source / "hub" / directory_name if (source / "hub").exists() else source / directory_name
            destination = cache / directory_name
            if not source.is_dir():
                raise ValueError(f"Cache missing: {source}")
            if destination.exists():
                raise ValueError(f"Refusing to overwrite existing cache: {destination}; use download instead")
            shutil.copytree(source, destination, symlinks=False)
            print(f"Copied standalone cache for {name} to {destination}")
        else:
            from huggingface_hub import snapshot_download
            path = snapshot_download(item["model_id"], revision=item.get("revision", "main"),
                cache_dir=str(cache), allow_patterns=["*.json", "*.safetensors", "*.bin"])
            print(json.dumps({"model": name, "snapshot": Path(path).name, "path": path}))


if __name__ == "__main__":
    main()
