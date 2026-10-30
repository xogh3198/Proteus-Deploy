"""Identical wire input; v1 reuses CPU input prepared before measurement.

v2 decodes/transforms every request. Neither phase caches model predictions.
"""
import hashlib
import copy
import io
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent


class Preparation:
    def __init__(self, image, phase, workers=8):
        from transformers import AutoImageProcessor
        if phase not in {"v1", "v2"} or not 1 <= int(workers) <= 32:
            raise ValueError("Expected v1/v2 and 1..32 CPU preparation workers")
        self.phase = phase
        self.config = json.loads((HERE / "preprocessor_config.json").read_text())
        self.processor_type = type(AutoImageProcessor.from_pretrained(HERE, use_fast=False,
                                                                    local_files_only=True))
        self.contract = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.local = threading.local()
        self.slots = threading.BoundedSemaphore(int(workers))
        self.fixture_sha256 = hashlib.sha256(image).hexdigest()
        self.decode_calls = 0
        self.counter_lock = threading.Lock()
        # This preparation is explicitly outside the request/trace measurement window.
        self.fixture = self._prepare(image)
        self.fixture["inputs"]["pixel_values"].setflags(write=False)

    def _prepare(self, raw):
        if not raw or len(raw) > 16 << 20:
            raise ValueError("JPEG must be in 1..16 MiB")
        start = time.perf_counter()
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
        decoded = time.perf_counter()
        processor = getattr(self.local, "processor", None)
        if processor is None:
            processor = self.processor_type.from_dict(copy.deepcopy(self.config))
            self.local.processor = processor
        array = np.asarray(processor(images=[image], return_tensors="np")["pixel_values"], dtype=np.float32)
        if array.shape != (1, 3, 224, 224) or not np.isfinite(array).all():
            raise ValueError("Invalid prepared image tensor")
        finished = time.perf_counter()
        with self.counter_lock:
            self.decode_calls += 1
        return {"inputs": {"pixel_values": array}, "image_size": image.size,
                "contract": self.contract, "timings": {"decode": (decoded-start)*1000,
                                                        "preprocess": (finished-decoded)*1000}}

    def prepare(self, raw):
        if self.phase == "v1":
            self.local.slot_wait_ms = 0.0
            if hashlib.sha256(raw).hexdigest() != self.fixture_sha256:
                raise ValueError("v1 accepts only the pinned benchmark JPEG; prepare a new fixture for a new input")
            return {**self.fixture, "timings": {"decode": 0.0, "preprocess": 0.0}}
        queued = time.perf_counter()
        with self.slots:
            self.local.slot_wait_ms = (time.perf_counter()-queued)*1000
            return self._prepare(raw)
