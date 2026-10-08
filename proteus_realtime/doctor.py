"""Validate independent GPU runtime, checkpoint cache and a real Gurobi solve."""
import importlib.metadata
import json
import os
from pathlib import Path

from .policy import solve_milp
from .registry import load_registry


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check NVIDIA Container Toolkit")
    tensor = torch.ones((2, 2), device="cuda")
    value = (tensor @ tensor)[0, 0].item()
    torch.cuda.synchronize()
    if value != 2:
        raise RuntimeError("CUDA math failed")
    if "V100" in torch.cuda.get_device_name(0) and "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("Torch wheel does not contain V100 sm_70 kernels")
    registry = load_registry(os.environ.get("MODEL_REGISTRY", "/config/models.json"))
    cache = Path(os.environ.get("HF_HOME", "/models")) / "hub"
    models = {}
    for name, item in registry.items():
        if not item.get("enabled", True):
            continue
        directory = cache / ("models--" + item["model_id"].replace("/", "--"))
        models[name] = {"model_id": item["model_id"], "requested_revision": item.get("revision"),
                        "cache_directory_exists": directory.exists(),
                        "snapshots": sorted(p.name for p in (directory / "snapshots").glob("*"))}
    # Synthetic numbers here check solver installation ONLY, never a profile.
    solver = solve_milp({"installation-check": {"batches": {"1": {"p50_ms": 1}}}},
                        {"installation-check": 100}, 1, 200).to_dict()
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "torchvision", "transformers", "grpcio", "protobuf", "pillow", "gurobipy")}
    print(json.dumps({"independent_runtime": True, "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda, "arch": torch.cuda.get_arch_list(),
        "cuda_test": value, "versions": versions, "models": models,
        "gurobi_installation_check": solver,
        "warning": "Installation check only; register measures actual GPU profiles"}, indent=2))


if __name__ == "__main__":
    main()
