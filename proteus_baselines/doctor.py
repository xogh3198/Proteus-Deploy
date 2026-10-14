"""CUDA-only startup/checkpoint check; not inference or policy verification."""
import json
import os
from pathlib import Path

from .config import verify_sources
from proteus_realtime.runtime import GpuRuntime


def main():
    baseline = os.environ.get("BASELINE")
    sources = verify_sources(baseline=baseline)
    backend = GpuRuntime()
    try:
        checkpoints = {}
        for name in backend.enabled:
            item = backend.registry[name]
            checkpoint = Path(os.environ.get("HF_HOME", "/models")) / (
                "hub/models--" + item["model_id"].replace("/", "--") + "/snapshots/" + item["revision"])
            if not checkpoint.is_dir():
                raise ValueError("Pinned checkpoint cache is missing for " + name + "; run download first")
            checkpoints[name] = str(checkpoint)
        solver = None
        if baseline == "sommelier":
            from .sommelier import solve_fixed_placement
            solver = solve_fixed_placement({"installation-check": {"batches": {"1": {"p50_ms": 1}}}},
                                          {"installation-check": 100}, 1, 200).to_dict()
        print(json.dumps({"cuda_math_check": "passed", "checkpoint_directory": checkpoints[backend.model_name],
                          "checkpoint_directories": checkpoints,
                          "solver_installation_check_only": solver,
                          "source_hashes": sources, "runtime": backend.describe(),
                          "scope": "No model inference or policy performance proof; solver uses a synthetic installation check"}, indent=2))
    finally:
        backend.reset()


if __name__ == "__main__":
    main()
