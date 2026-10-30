"""Content-pinned FP32 model acquisition. Serving never downloads weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


HERE = Path(__file__).resolve().parent


def manifest():
    data = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
    expected = {f"resnet{n}_v1" for n in (18, 34, 50, 101, 152)}
    if set(data["models"]) != expected or data["dtype"] != "fp32":
        raise ValueError("Use all five ResNet-v1 variants in FP32")
    return data


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_model(path, entry):
    import onnx
    from onnx import TensorProto
    path = Path(path)
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint hash/size mismatch: {path}")
    graph = onnx.load(str(path), load_external_data=False)
    floats = {t.data_type for t in graph.graph.initializer
              if t.data_type in {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE,
                                 TensorProto.BFLOAT16}}
    if floats != {TensorProto.FLOAT}:
        raise ValueError(f"Expected real FP32 weights, found tensor types {floats}")
    forbidden = {TensorProto.FLOAT16, TensorProto.DOUBLE, TensorProto.BFLOAT16}
    for node in graph.graph.node:
        for attribute in node.attribute:
            if ((node.op_type == "Cast" and attribute.name == "to" and attribute.i in forbidden)
                    or (attribute.HasField("t") and attribute.t.data_type in forbidden)):
                raise ValueError("Mixed/half floating-point graph conversion is not allowed")
    initializer_names = {t.name for t in graph.graph.initializer}
    inputs = [t for t in graph.graph.input if t.name not in initializer_names]
    if len(inputs) != 1 or inputs[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Expected one FP32 image input")
    shape = inputs[0].type.tensor_type.shape.dim
    if len(shape) != 4 or [d.dim_value for d in shape[1:]] != [3, 224, 224]:
        raise ValueError("Expected NCHW [N,3,224,224]")
    if shape[0].dim_value > 0:
        raise ValueError("Dynamic batch dimension required for batch profiling")
    onnx.checker.check_model(graph)
    return {"file": entry["file"], "sha256": entry["sha256"],
            "weight_dtype": "float32", "input_dtype": "float32",
            "input_shape": ["N", 3, 224, 224],
            "opsets": [{"domain": o.domain, "version": o.version} for o in graph.opset_import]}


def acquire(root, fetch=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name, entry in manifest()["models"].items():
        target = root / entry["file"]
        if not target.exists():
            if not fetch:
                raise FileNotFoundError(f"Run setup first: {target}")
            # Download to a fresh staging directory; existing assets are never replaced.
            with tempfile.TemporaryDirectory(prefix="resnet-download-", dir=root) as temporary:
                candidate = Path(temporary) / entry["file"]
                request = urllib.request.Request(entry["url"], headers={"User-Agent": "paired-resnet/1"})
                with urllib.request.urlopen(request, timeout=60) as source, candidate.open("xb") as sink:
                    received = 0
                    for chunk in iter(lambda: source.read(8 << 20), b""):
                        received += len(chunk)
                        if received > entry["bytes"]:
                            raise ValueError("Downloaded model exceeds pinned size")
                        sink.write(chunk)
                check_model(candidate, entry)
                # A hard link creates the final name exclusively on Linux and Windows.
                try:
                    target.hardlink_to(candidate)
                except FileExistsError:
                    check_model(target, entry)
        verified[name] = check_model(target, entry)
        print(f"{name}: verified FP32 checkpoint {entry['sha256'][:12]}", flush=True)
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "verify"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(acquire(args.root, args.action == "download"), indent=2))


if __name__ == "__main__":
    main()
