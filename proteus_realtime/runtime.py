"""Independent PyTorch/Hugging Face GPU executor; no INFaaS imports or RPCs."""
import gc
import io
import importlib.metadata
import os
import time

from .registry import load_registry
from pathlib import Path


class GpuRuntime:
    def __init__(self):
        startup = time.perf_counter()
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU inference fallback is forbidden")
        self.torch, self.Image = torch, Image
        self.AutoProcessor, self.AutoModel = AutoImageProcessor, AutoModelForImageClassification
        self.registry = load_registry(os.environ.get("MODEL_REGISTRY", "/config/models.json"))
        self.enabled = [name for name, item in self.registry.items() if item.get("enabled", True)]
        if not self.enabled:
            raise ValueError("Enable at least one model in the registry")
        self.model_name = os.environ.get("MODEL_NAME", "resnet-50")
        if self.model_name not in self.enabled:
            raise ValueError("MODEL_NAME must be enabled in the registry")
        self.model_id = self.registry[self.model_name]["model_id"]
        self.application = os.environ.get("APPLICATION", "imagenet")
        self.gpu_type = os.environ.get("GPU_TYPE", "v100")
        if self.gpu_type == "v100" and "V100" not in torch.cuda.get_device_name(0):
            raise ValueError("GPU_TYPE=v100 but the selected device is not a V100")
        torch.set_num_threads(int(os.environ.get("CPU_THREADS", "2")))
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = self.processor = self.current = None
        self.load_stages = {}
        self.warmups = int(os.environ.get("WARMUP_QUERIES", "10"))
        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            tensor = torch.ones((2, 2), device="cuda")
            if (tensor @ tensor)[0, 0].item() != 2:
                raise RuntimeError("CUDA math check failed")
        self.stream.synchronize()
        self.startup_ms = (time.perf_counter() - startup) * 1000

    def load(self, name=None):
        name = name or self.model_name
        if name not in self.enabled:
            raise ValueError(f"Model not enabled: {name}")
        if self.current == name:
            return {"loaded": False, "load_ms": 0.0, "ready_ms": 0.0, "stages": {}}
        started = time.perf_counter()
        self.reset()
        item = self.registry[name]
        options = {"revision": item.get("revision", "main"), "local_files_only": True}
        read_start = time.perf_counter()
        processor = self.AutoProcessor.from_pretrained(item["model_id"], use_fast=False, **options)
        model = self.AutoModel.from_pretrained(item["model_id"], **options).eval().float()
        read_end = time.perf_counter()
        if int(getattr(model.config, "num_labels", 0)) != 1000:
            raise ValueError("This port supports ImageNet-1k classification models only")
        transfer_start = time.perf_counter()
        with self.torch.cuda.stream(self.stream):
            model = model.to("cuda")
        self.stream.synchronize()
        transfer_end = time.perf_counter()
        self.model, self.processor, self.current = model, processor, name
        warm_start = time.perf_counter()
        probe = self.Image.new("RGB", (224, 224), (128, 128, 128))
        inputs = processor(images=[probe], return_tensors="pt")
        for _ in range(self.warmups):
            self._forward(inputs)
        warm_end = time.perf_counter()
        self.load_stages = {
            "checkpoint_read_ms": (read_end - read_start) * 1000,
            "model_h2d_ms": (transfer_end - transfer_start) * 1000,
            "warmup_ms": (warm_end - warm_start) * 1000,
        }
        return {"loaded": True, "load_ms": (warm_end - started) * 1000,
                "ready_ms": (transfer_end - started) * 1000, "stages": dict(self.load_stages)}

    def _forward(self, inputs):
        with self.torch.cuda.stream(self.stream), self.torch.inference_mode():
            gpu_inputs = {key: value.to("cuda", non_blocking=False) for key, value in inputs.items()}
            labels = self.model(**gpu_inputs).logits.argmax(-1).cpu().tolist()
        self.stream.synchronize()
        return labels

    def infer_batch(self, image_bytes):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        started = time.perf_counter()
        images = []
        for data in image_bytes:
            with self.Image.open(io.BytesIO(data)) as image:
                images.append(image.convert("RGB"))
        decoded = time.perf_counter()
        inputs = self.processor(images=images, return_tensors="pt")
        preprocessed = time.perf_counter()
        labels = self._forward(inputs)
        forwarded = time.perf_counter()
        timings = {"decode": (decoded - started) * 1000,
                   "preprocess": (preprocessed - decoded) * 1000,
                   "forward": (forwarded - preprocessed) * 1000}
        return [{"label": self.model.config.id2label[int(label)], "label_id": int(label),
                 "boxes": [], "image_size": image.size, "timings": dict(timings)}
                for image, label in zip(images, labels)]

    def infer(self, image):
        return self.infer_batch([image])[0]

    def reset(self):
        if hasattr(self, "stream"):
            self.stream.synchronize()
        self.model = self.processor = self.current = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def fingerprint(self):
        revisions = {}
        cache = Path(os.environ.get("HF_HOME", "/models")) / "hub"
        for name in self.enabled:
            item = self.registry[name]
            revision = item.get("revision", "main")
            ref = cache / ("models--" + item["model_id"].replace("/", "--")) / "refs" / revision
            revisions[name] = {"model_id": item["model_id"], "revision":
                ref.read_text().strip() if ref.is_file() else revision}
        return {"gpu": self.torch.cuda.get_device_name(0), "torch": self.torch.__version__,
                "cuda": self.torch.version.cuda, "dtype": "fp32",
                "preprocessing": "PIL RGB + HF slow AutoImageProcessor",
                "cpu_threads": self.torch.get_num_threads(), "warmup_queries": self.warmups,
                "versions": {p: importlib.metadata.version(p) for p in ("transformers", "pillow", "torchvision")},
                "checkpoints": revisions}

    def describe(self):
        return {"runtime_backend": "proteus_realtime.runtime.GpuRuntime",
                "independent_runtime": True, "application": self.application,
                "model_name": self.current or self.model_name,
                "model_id": self.registry[self.current or self.model_name]["model_id"],
                "enabled_models": list(self.enabled), "gpu_type": self.gpu_type,
                "gpu_name": self.torch.cuda.get_device_name(0), "resident": self.model is not None,
                "checkpoint_commit": getattr(getattr(self.model, "config", None), "_commit_hash", None),
                "allocated_mib": self.torch.cuda.memory_allocated() / (1024 * 1024),
                "reserved_mib": self.torch.cuda.memory_reserved() / (1024 * 1024),
                "startup_ms": self.startup_ms, "load_stages": dict(self.load_stages), **self.fingerprint()}
