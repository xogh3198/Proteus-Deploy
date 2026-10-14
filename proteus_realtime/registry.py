"""Explicit checkpoint identity and independently supplied offline accuracy."""
import json
import math
import os
import re
from pathlib import Path


def load_registry(path, model_name=None, model_id=None, revision=None):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    models = data.get("models", {})
    # A pinned env override selects ONE checkpoint without editing the registry.
    # Empty explicit arguments disable ambient env lookup in host-side settings.
    model_id = os.environ.get("MODEL_ID", "") if model_id is None else model_id
    revision = os.environ.get("MODEL_REVISION", "") if revision is None else revision
    if model_id:
        model_name = model_name or os.environ.get("MODEL_NAME", "")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", model_name):
            raise ValueError("MODEL_NAME must be a simple name for the env checkpoint")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*/[a-zA-Z0-9][a-zA-Z0-9_.-]*", model_id):
            raise ValueError("MODEL_ID must be a Hugging Face owner/model ID")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("MODEL_REVISION must be a pinned 40-character commit for MODEL_ID")
        models = {model_name: {"enabled": True, "model_id": model_id, "revision": revision,
                    "accuracy": None, "accuracy_source": None, "accuracy_dataset": None}}
    elif revision:
        raise ValueError("MODEL_REVISION requires MODEL_ID; clear both to use MODEL_REGISTRY")
    if not isinstance(models, dict) or not models:
        raise ValueError("Registry must contain a nonempty models object")
    for name, item in models.items():
        if not isinstance(item, dict) or not isinstance(item.get("model_id"), str):
            raise ValueError(f"Invalid model registry entry: {name}")
        if item.get("accuracy") is not None:
            value = item["accuracy"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 100:
                raise ValueError(f"accuracy must be an offline percentage in (0,100]: {name}")
            if not item.get("accuracy_source"):
                raise ValueError(f"accuracy_source is required: {name}")
    return models


def normalized_accuracies(models, measured=None):
    measured = measured or {}
    if len(models) == 1:
        # Relative-to-best is identically 100 for one model, NOT measured Top-1.
        return {next(iter(models)): 100.0}
    scores = {}
    sources = set()
    for name, item in models.items():
        entry = measured.get(name, item)
        score = entry.get("accuracy")
        if score is None or not entry.get("accuracy_source"):
            raise ValueError(f"Multiple-model MILP requires offline accuracy + source for {name}; use evaluate")
        if not 0 < float(score) <= 100:
            raise ValueError("Offline accuracy must be in (0,100]")
        scores[name] = float(score)
        sources.add(entry.get("accuracy_dataset", item.get("accuracy_dataset")))
    if None in sources or len(sources) != 1:
        raise ValueError("All variants must use the same accuracy_dataset and metric")
    best = max(scores.values())
    return {name: 100 * score / best for name, score in scores.items()}
