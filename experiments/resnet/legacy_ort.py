"""FP32 ResNet inference through the official ONNX Runtime 1.8.1 C API."""
import ctypes as ct
import os
from weakref import finalize

import numpy as np


class LegacyOrt:
    def __init__(self):
        library = ct.CDLL(os.environ["PROTEUS_ORT181_BRIDGE"])
        size_out = ct.POINTER(ct.c_size_t)
        signatures = {
            "error": (ct.c_char_p, ()),
            "version": (ct.c_char_p, ()),
            "cuda_version": (ct.c_int, (ct.POINTER(ct.c_int),)),
            "device": (ct.c_int, (ct.POINTER(ct.c_char), ct.c_size_t)),
            "synchronize": (ct.c_int, ()),
            "memory": (ct.c_int, (size_out, size_out)),
            "open": (ct.c_void_p, (ct.c_char_p, ct.c_int, ct.c_int)),
            "close": (None, (ct.c_void_p,)),
            "run": (ct.c_int, (ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_void_p)),
        }
        self.api = {}
        for name, (result, arguments) in signatures.items():
            function = getattr(library, "proteus_ort181_" + name)
            function.restype, function.argtypes = result, arguments
            self.api[name] = function
        self.version = self.api["version"]().decode()
        if self.version != "1.8.1":
            raise RuntimeError(f"Expected ONNX Runtime 1.8.1, loaded {self.version}")

    def check(self, status):
        if status:
            raise RuntimeError(self.api["error"]().decode())

    def versions(self):
        cuda = ct.c_int()
        self.check(self.api["cuda_version"](ct.byref(cuda)))
        cudnn_version = ct.CDLL("libcudnn.so.8").cudnnGetVersion
        cudnn_version.argtypes, cudnn_version.restype = (), ct.c_size_t
        return {"cuda": cuda.value, "cudnn": cudnn_version()}

    def get_device_name(self, device):
        if device != 0:
            raise ValueError("Expose exactly one GPU to the legacy runtime")
        name = ct.create_string_buffer(256)
        self.check(self.api["device"](name, len(name)))
        return name.value.decode()

    def synchronize(self):
        self.check(self.api["synchronize"]())

    def mem_get_info(self):
        free, total = ct.c_size_t(), ct.c_size_t()
        self.check(self.api["memory"](ct.byref(free), ct.byref(total)))
        return free.value, total.value

    def open(self, path, intra_threads, inter_threads):
        handle = self.api["open"](os.fsencode(path), intra_threads, inter_threads)
        if not handle:
            raise RuntimeError(self.api["error"]().decode())
        return Session(self, handle)


class Session:
    def __init__(self, runtime, handle):
        self.runtime = runtime
        self.handle = handle
        finalize(self, runtime.api["close"], handle)

    def run(self, array):
        if array.ndim != 4 or array.shape[1:] != (3, 224, 224) or not len(array):
            raise ValueError("Expected nonempty ResNet input [batch, 3, 224, 224]")
        if array.dtype != np.float32 or not array.flags.c_contiguous:
            raise ValueError("Expected contiguous FP32 ResNet input")
        output = np.empty((len(array), 1000), dtype=np.float32)
        self.runtime.check(self.runtime.api["run"](
            self.handle, array.ctypes.data, len(array), output.ctypes.data))
        return output
