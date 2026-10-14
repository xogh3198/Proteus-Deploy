"""CUDA-only startup/checkpoint check; not inference or policy verification."""
import json
import os
from pathlib import Path

from .config import verify_sources
from proteus_realtime.runtime import GpuRuntime


def main():
    sources = verify_sources()
    backend = GpuRuntime()
    try:
        item = backend.registry[backend.model_name]
        checkpoint = Path(os.environ.get("HF_HOME", "/models")) / (
            "hub/models--" + item["model_id"].replace("/", "--") + "/snapshots/" + item["revision"])
        if not checkpoint.is_dir():
            raise ValueError("Pinned checkpoint cache is missing; run download first")
        print(json.dumps({"cuda_math_check": "passed", "checkpoint_directory": str(checkpoint),
                          "source_hashes": sources, "runtime": backend.describe(),
                          "scope": "No training, model inference, MILP, or native policy proof in doctor"}, indent=2))
    finally:
        backend.reset()


if __name__ == "__main__":
    main()
