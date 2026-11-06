"""Install only the headers and shared libraries used by the ORT 1.8.1 image."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen


REPOSITORY = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/"
CUDA_HEADERS = PurePosixPath("usr/local/cuda-11.0/targets/x86_64-linux/include")

# Versions and checksums from the NVIDIA repository used by CUDA 11.0.3.
# An empty SONAME list selects only compiler headers from the package.
BUILD_PACKAGES = (
    ("cuda-cudart-dev-11-0_11.0.221-1_amd64.deb",
     "55c109c34603ed23f3e317487c569bbd24505831b9bce920f500315e34f84b16", ()),
    ("cuda-nvcc-11-0_11.0.221-1_amd64.deb",
     "dfdfe5088c1d86b9b47580de16ef288ef619d62232c4b483d2fe3fee0d57607d", ()),
    ("cuda-cudart-11-0_11.0.221-1_amd64.deb",
     "fb4faccde1994d6d1b14b6235239f949a173a6cae6900773845dbe3aae607f4b", ("libcudart.so.11.0",)),
)
RUNTIME_PACKAGES = (
    ("libcublas-11-0_11.2.0.252-1_amd64.deb",
     "0abe7deaf1ef63c0cc60c28ef9d11ae5ff099314798a6689a479cbee75995043",
     ("libcublas.so.11", "libcublasLt.so.11")),
    ("libcufft-11-0_10.2.1.245-1_amd64.deb",
     "9e14d83854197f2a1e9547904063e49c8735e46feefa775c15908a92c4e035c2", ("libcufft.so.10",)),
    ("libcurand-11-0_10.2.1.245-1_amd64.deb",
     "e1317418dd79bafdf244c07d509f102789b338723fac52adce4ec8473993dbad", ("libcurand.so.10",)),
    ("libcudnn8_8.0.5.39-1+cuda11.0_amd64.deb",
     "184841dcc705c2f653b351c69ad51b8fa03ee6bf5e662e6315eda994253be38c",
     ("libcudnn.so.8", "libcudnn_adv_infer.so.8", "libcudnn_cnn_infer.so.8", "libcudnn_ops_infer.so.8")),
)


def extract_selected(archive, destination, sonames):
    copied = set()
    for entry in archive:
        if not entry.isfile():
            continue
        source = PurePosixPath(entry.name)
        if sonames:
            filename = next((name for name in sonames
                             if source.name == name or source.name.startswith(name + ".")), None)
            if filename is None:
                continue
            target = destination / filename
            copied.add(filename)
        else:
            if not source.is_relative_to(CUDA_HEADERS):
                continue
            relative = source.relative_to(CUDA_HEADERS)
            if ".." in relative.parts:
                raise ValueError(f"Invalid header path: {entry.name}")
            target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(entry) as stream, target.open("xb") as output:
            shutil.copyfileobj(stream, output, 1 << 20)
    missing = set(sonames) - copied
    if missing:
        raise RuntimeError(f"Missing shared libraries: {', '.join(sorted(missing))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "runtime"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    packages = BUILD_PACKAGES if args.stage == "build" else RUNTIME_PACKAGES
    with tempfile.TemporaryDirectory(prefix="ort181-cuda-") as directory:
        deb = Path(directory) / "package.deb"
        for name, checksum, sonames in packages:
            print("Installing selected files from", name, flush=True)
            with urlopen(REPOSITORY + name, timeout=90) as response, deb.open("wb") as output:
                shutil.copyfileobj(response, output, 1 << 20)
            with deb.open("rb") as downloaded:
                if hashlib.file_digest(downloaded, "sha256").hexdigest() != checksum:
                    raise ValueError(f"NVIDIA package checksum differs: {name}")
            destination = args.root / ("cuda11" if sonames else "cuda-include")
            with subprocess.Popen(["dpkg-deb", "--fsys-tarfile", str(deb)], stdout=subprocess.PIPE) as process:
                with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                    extract_selected(archive, destination, sonames)
                if process.wait() != 0:
                    raise RuntimeError(f"Could not unpack {name}")
            deb.unlink()


if __name__ == "__main__":
    main()
