"""Verify exact pinned policy files, including the two declared platform adaptations."""
import hashlib
import json
from pathlib import Path


def check(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[2]
    document=json.loads(Path(__file__).with_name("proteus_source.json").read_text())
    changed=[]
    for name,entry in document["files"].items():
        path=root/name
        digest=hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
        if digest!=entry["runtime_sha256"]:raise RuntimeError("Policy source differs from audited v1: "+name)
        if digest!=entry["upstream_sha256"]:changed.append(name)
    if set(changed)!=set(document["allowed_adaptations"]):
        raise RuntimeError("Declared source adaptation set differs")
    return {"upstream":document["repository"],"commit":document["commit"],
        "verified_files":len(document["files"]),"adaptations":document["allowed_adaptations"],
        "custom_MILP_or_batch_policy_used":False}
