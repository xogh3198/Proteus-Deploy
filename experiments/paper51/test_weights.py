import hashlib
import io
import zipfile

import pytest

from experiments.paper51 import weights


def test_sources_cover_all_model_families_without_random_classifier_heads():
    sources = weights.read_sources()["models"]
    assert len(sources) == 51
    assert sources["mobilenet0.25"]["kind"] == "gluoncv"
    assert sources["mobilenet0.75"]["model_id"] == "mobilenet0.75"
    assert sources["t5-11b"]["model_id"] == "google-t5/t5-11b"
    assert all(m["task"] == "feature-extraction" for m in sources.values() if m["group"] == "bert")
    assert sum(m["kind"] == "huggingface" for m in sources.values()) == 29


def test_source_lock_detects_corruption_and_source_substitution(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "weights.bin").write_bytes(b"verified pretrained weights")
    source = {"revision": "fixed"}
    metadata = {"source": source, "files": {"weights.bin": {
        "bytes": (path / "weights.bin").stat().st_size,
        "sha256": weights.digest(path / "weights.bin")}}}
    weights.verify_model(path, metadata, source)
    with pytest.raises(ValueError, match="revision"):
        weights.verify_model(path, metadata, {"revision": "changed"})
    (path / "weights.bin").write_bytes(b"incorrect weights")
    with pytest.raises(ValueError, match="content"):
        weights.verify_model(path, metadata, source)


@pytest.mark.parametrize("name", ["../escape.bin", "/escape.bin", "C:/escape.bin", "..\\escape.bin"])
def test_source_lock_cannot_read_arbitrary_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        weights.verify_model(tmp_path, {"source": {}, "files": {name: {}}}, {})


def test_gluon_archive_uses_only_named_verified_parameters(tmp_path, monkeypatch):
    payload = b"the pretrained parameter bytes"
    checksum = hashlib.sha1(payload).hexdigest()
    filename = "mobilenet0.25-" + checksum[:8] + ".params"

    def download(url, path):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(filename, payload)
            archive.writestr("../unrelated.txt", "must never be extracted")

    monkeypatch.setattr(weights, "download", download)
    entry = {"model_id": "mobilenet0.25", "parameter_sha1": checksum, "download_url": "unused"}
    result = weights.get_gluon(entry, tmp_path)
    assert set(result) == {filename}
    assert (tmp_path / filename).read_bytes() == payload
    assert not (tmp_path.parent / "unrelated.txt").exists()


def test_failed_parameter_verification_leaves_no_accepted_model(tmp_path, monkeypatch):
    entry = {"model_id": "mobilenet0.25", "parameter_sha1": "0" * 40, "download_url": "unused"}

    def download(url, path):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mobilenet0.25-00000000.params", b"incorrect")

    monkeypatch.setattr(weights, "download", download)
    with pytest.raises(ValueError, match="checksum"):
        weights.get_gluon(entry, tmp_path)
    assert not list(tmp_path.glob("*.params"))


def test_partial_unverified_weight_directories_are_preserved(tmp_path):
    path = tmp_path / "t5-11b"
    path.mkdir()
    checkpoint = path / "partial.bin"
    checkpoint.write_bytes(b"unfinished download")
    with pytest.raises(ValueError, match="partial"):
        weights.acquire_model("t5-11b", {"kind": "huggingface"}, tmp_path)
    assert checkpoint.read_bytes() == b"unfinished download"


def test_download_does_not_replace_an_existing_file(tmp_path):
    path = tmp_path / "existing.pt"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        weights.download("unused", path, opener=lambda *a, **kw: io.BytesIO(b"new"))
    assert path.read_bytes() == b"keep"
