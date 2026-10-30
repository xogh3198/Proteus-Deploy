"""Offline CPU-only tensor equivalence check, not a latency benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
import threading
from pathlib import Path

from .registry import load_registry
from .runtime import GpuRuntime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--registry", default="/config/models.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 32 or not 1 <= args.repeats <= 100:
        parser.error("workers must be in 1..32; repeats in 1..100")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor
    torch.set_num_threads(int(os.environ.get("CPU_THREADS", "2")))
    torch.set_num_interop_threads(int(os.environ.get("CPU_INTEROP_THREADS", "1")))
    # Do not construct a CUDA runtime, load model weights or access the network.
    runtime = GpuRuntime.__new__(GpuRuntime)
    runtime.Image, runtime.AutoProcessor = Image, AutoImageProcessor
    runtime.registry = load_registry(args.registry)
    runtime.enabled = [name for name, row in runtime.registry.items() if row.get("enabled", True)]
    if not runtime.enabled:
        raise ValueError("Enable at least one model")
    runtime.cpu_processor, runtime.preprocess_contract = runtime._shared_processor()
    runtime._cpu_processors = threading.local()
    raw = Path(args.image).read_bytes()
    with Image.open(io.BytesIO(raw)) as source:
        image = source.convert("RGB")
    reference = dict(runtime.cpu_processor(images=[image], return_tensors="pt"))

    def check(_):
        actual = runtime.prepare(raw)
        if actual["image_size"] != image.size or set(actual["inputs"]) != set(reference):
            raise AssertionError("Prepared input keys/image dimensions changed")
        for key, expected in reference.items():
            value = actual["inputs"][key]
            if value.device.type != "cpu" or value.dtype != expected.dtype or not torch.equal(value, expected):
                raise AssertionError("CPU tensor differs from serial reference: " + key)
        return threading.get_ident()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        used_threads = set(pool.map(check, range(args.workers * args.repeats)))
    print(json.dumps({"equivalent": True, "scope": "CPU tensors only; no GPU or latency claim",
        "processor_ownership": "thread-local-v1", "models": runtime.enabled,
        "workers": args.workers, "threads_used": len(used_threads),
        "comparisons": args.workers * args.repeats,
        "image_sha256": hashlib.sha256(raw).hexdigest(),
        "preprocess_contract": runtime.preprocess_contract}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
