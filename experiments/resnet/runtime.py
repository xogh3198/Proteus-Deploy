"""One real FP32 ONNX/CUDA model resident at a time; no prediction cache."""
from collections import deque
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from .assets import acquire, manifest
from .preparation import Preparation


class Runtime:
    def __init__(self):
        import torch
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
        if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("A working CUDA device and ONNX CUDAExecutionProvider are required")
        self.torch, self.ort = torch, ort
        torch.set_num_threads(int(os.environ.get("CPU_THREADS", "2")))
        torch.set_num_interop_threads(int(os.environ.get("CPU_INTEROP_THREADS", "1")))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device_name = torch.cuda.get_device_name(0)
        if "V100" not in self.device_name:
            raise ValueError("The paired experiment requires the selected V100")
        # CUDA initialization is distinct from checkpoint loading and done before requests.
        probe = torch.ones((2, 2), device="cuda", dtype=torch.float32)
        if (probe @ probe)[0, 0].item() != 2:
            raise RuntimeError("CUDA math check failed")
        torch.cuda.synchronize()
        del probe
        torch.cuda.empty_cache()
        self.root = Path(os.environ.get("MODEL_DIR", "/models"))
        self.verified = acquire(self.root)
        self.registry = manifest()["models"]
        self.enabled = list(self.registry)
        self.application, self.gpu_type = "imagenet", "v100"
        self.model_name = os.environ.get("MODEL_NAME", "resnet50_v1")
        if self.model_name not in self.registry:
            raise ValueError("MODEL_NAME must name one of the five ResNet-v1 variants")
        self.model_id = self.registry[self.model_name]["model_id"]
        phase = os.environ["EXPERIMENT_PHASE"]
        if phase != json.loads(Path(__file__).with_name("phase.json").read_text())["phase"]:
            raise ValueError("Container image phase differs from requested phase; rebuild/check out the correct branch")
        self.preparation = Preparation(Path("/inputs/image.jpg").read_bytes(), phase,
                                       int(os.environ.get("PREPROCESS_WORKERS", "8")))
        self.preprocess_contract = self.preparation.contract
        self.gpu_lock = threading.RLock()
        self.session = self.current = None
        self.warmups = int(os.environ.get("WARMUP_QUERIES", "10"))
        self.loads = deque(maxlen=4096)
        self.load_count = self.switch_count = self.forward_count = 0
        self.memory_baseline_free = int(torch.cuda.mem_get_info()[0])
        self.load_stages = {}

    def prepare(self, raw):
        return self.preparation.prepare(raw)

    def _forward(self, array):
        if self.session is None:
            raise RuntimeError("No GPU model is resident")
        binding = self.session.io_binding()
        array = np.ascontiguousarray(array, dtype=np.float32)
        binding.bind_cpu_input(self.input_name, array)
        binding.bind_output(self.output_name, "cuda", 0)
        self.session.run_with_iobinding(binding)
        result = binding.copy_outputs_to_cpu()[0]
        self.torch.cuda.synchronize()
        if result.shape != (len(array), 1000) or not np.isfinite(result).all():
            raise RuntimeError("Expected finite ImageNet-1k inference outputs")
        self.forward_count += 1
        return result

    def load(self, name=None):
        name = name or self.model_name
        if name not in self.enabled:
            raise ValueError("Unknown ResNet candidate")
        with self.gpu_lock:
            if self.current == name:
                return {"loaded": False, "load_ms": 0.0, "ready_ms": 0.0, "stages": {}}
            before, started = self.current, time.perf_counter()
            self.reset()
            options = self.ort.SessionOptions()
            options.intra_op_num_threads = self.torch.get_num_threads()
            options.inter_op_num_threads = self.torch.get_num_interop_threads()
            session = self.ort.InferenceSession(str(self.root / self.registry[name]["file"]),
                sess_options=options, providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})])
            session.disable_fallback()
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("Refusing CPU-only inference fallback")
            self.session, self.current = session, name
            self.input_name = session.get_inputs()[0].name
            self.output_name = session.get_outputs()[0].name
            ready = time.perf_counter()
            for _ in range(self.warmups):
                self._forward(self.preparation.fixture["inputs"]["pixel_values"])
            end = time.perf_counter()
            self.load_count += 1
            self.switch_count += int(before is not None and before != name)
            self.load_stages = {"onnx_session_and_device_setup_ms": (ready-started)*1000,
                                "warmup_ms": (end-ready)*1000}
            result = {"loaded": True, "load_ms": (end-started)*1000,
                      "ready_ms": (ready-started)*1000, "stages": dict(self.load_stages)}
            self.loads.append({"id": self.load_count, "model": name, "previous_model": before,
                               "unix_seconds": time.time(), **result})
            return result

    def infer_prepared(self, prepared):
        with self.gpu_lock:
            if not prepared or any(p["contract"] != self.preprocess_contract for p in prepared):
                raise ValueError("Prepared input contract mismatch")
            start = time.perf_counter()
            array = np.concatenate([p["inputs"]["pixel_values"] for p in prepared], axis=0)
            packed = time.perf_counter()
            outputs = self._forward(array)
            end = time.perf_counter()
            labels = outputs.argmax(axis=-1).tolist()
            return [{"label": f"ImageNet class {index}", "label_id": int(index), "boxes": [],
                     "image_size": p["image_size"], "timings": {**p["timings"],
                         "batch_pack": (packed-start)*1000, "forward": (end-packed)*1000}}
                    for p, index in zip(prepared, labels)]

    def infer_batch(self, images):
        return self.infer_prepared([self.prepare(image) for image in images])

    def infer(self, raw):
        return self.infer_batch([raw])[0]

    def reset(self):
        with self.gpu_lock:
            self.torch.cuda.synchronize()
            self.session = self.current = None
            gc.collect()
            self.torch.cuda.empty_cache()

    def fingerprint(self):
        return {"runtime": "onnxruntime-cuda-resnet", "dtype": "fp32", "gpu": self.device_name,
                "cuda": self.torch.version.cuda,
                "versions": {p: importlib.metadata.version(p) for p in
                             ("onnxruntime-gpu", "onnx", "torch", "transformers", "pillow", "numpy")},
                "checkpoints": self.verified, "phase": self.preparation.phase,
                "preprocessing_contract": self.preprocess_contract,
                "fixture_sha256": self.preparation.fixture_sha256,
                "cpu_threads": self.torch.get_num_threads(),
                "cpu_interop_threads": self.torch.get_num_interop_threads(), "warmups": self.warmups,
                "preprocess_workers": int(os.environ.get("PREPROCESS_WORKERS", "8"))}

    def describe(self):
        return {"runtime_backend": "paired-resnet-onnx-cuda", "independent_runtime": True,
                "application": self.application, "model_name": self.current or self.model_name,
                "model_id": self.registry[self.current or self.model_name]["model_id"],
                "enabled_models": self.enabled, "gpu_type": "v100", "gpu_name": self.device_name,
                "dtype": "fp32", "resident": self.current is not None,
                "experiment_phase": self.preparation.phase,
                "request_preprocessing": self.preparation.phase == "v2",
                "preprocessing": "PIL RGB + HF slow BitImageProcessor, 256 resize / 224 crop",
                "decode_calls_including_startup_fixture": self.preparation.decode_calls,
                "preprocessing_loading_overlap": False,
                "same_request_preprocessing_loading_overlap": False,
                "load_stages": self.load_stages, "actual_load_count": self.load_count,
                "model_switches_after_initial_load": self.switch_count,
                "actual_forward_calls": self.forward_count, "actual_load_events": list(self.loads),
                "profile_scope": "prepared-batch-service", "fingerprint": self.fingerprint()}


class InfaasBackend:
    """Only replace the executor backend; retain this repository's cost policy."""
    def __init__(self, settings):
        self.runtime = Runtime()
        self.fingerprint = self.runtime.fingerprint()

    @property
    def resident(self):
        return self.runtime.current

    def prepare(self, raw):
        start = time.monotonic()
        value = self.runtime.prepare(raw)
        end = time.monotonic()
        return {**value, "size": value["image_size"], "cpu_start": start, "cpu_end": end,
                "decode": value["timings"]["decode"], "preprocess": value["timings"]["preprocess"]}

    def load(self, name):
        value = self.runtime.load(name)
        return value["load_ms"], value["loaded"]

    def execute(self, model, prepared):
        if self.runtime.current != model:
            raise RuntimeError("Selected model is not resident")
        started = time.perf_counter()
        rows = self.runtime.infer_prepared(prepared)
        return [r["label"] for r in rows], (time.perf_counter()-started)*1000

    def profile_batch(self, model, prepared, batch, runs):
        from statistics import fmean
        from infaas.common.trace_replay import percentile
        samples = []
        payload = [prepared] * batch
        for _ in range(3):
            self.execute(model, payload)
        for _ in range(runs):
            samples.append(self.execute(model, payload)[1])
        free = int(self.runtime.torch.cuda.mem_get_info()[0])
        # ORT owns CUDA allocations: torch.max_memory_allocated would incorrectly
        # report only PyTorch's initialization allocations. Record this scope.
        memory = self.runtime.memory_baseline_free - free
        if memory <= 0:
            raise RuntimeError("Cannot measure a positive ORT device-memory delta; check exclusive GPU use")
        return {"p50_ms": percentile(samples, 50), "p95_ms": percentile(samples, 95),
                "mean_ms": fmean(samples), "samples_ms": samples,
                "capacity_qps": 1000*batch/fmean(samples), "memory_bytes": memory,
                "memory_scope": "device-used delta after repeated batches; not transient allocation peak",
                "runs": runs}

    def unload(self):
        self.runtime.reset()
