"""Single-GPU reduction of Proteus's Sommelier comparison, NOT Yale's index.

Original run.py selects Ilp(static='spec_acc'); add_spec_acc_constraints fixes
each device's APPLICATION, while allowing that application's model variants.
With one already-assigned GPU/application this constraint is vacuous. The
existing single-GPU MILP and accscale wall-clock port therefore give exactly
the same selection/batching as Proteus when inputs/controller settings match.
Do not add an artificial heuristic to manufacture a performance difference.
"""
import hashlib
import json
import os
from pathlib import Path
from proteus_realtime.policy import solve_milp
from proteus_realtime.protocol import InferRequest
from proteus_realtime.registry import normalized_accuracies
from proteus_realtime.serving import ProteusExecutor


def solve_fixed_placement(profiles, accuracies, demand, slo_ms, **kwargs):
    """All candidates belong to the already-fixed application on cuda:0."""
    return solve_milp(profiles, accuracies, demand, slo_ms, **kwargs)


def verify_class_mapping(registry, cache):
    reference = None
    for name, item in registry.items():
        path = Path(cache) / ("hub/models--" + item["model_id"].replace("/", "--")) / "snapshots" / item["revision"] / "config.json"
        labels = json.loads(path.read_text(encoding="utf-8")).get("id2label", {})
        if set(labels) != {str(i) for i in range(1000)} or any(not isinstance(v, str) or not v for v in labels.values()):
            raise ValueError("ImageNet class-ID mapping is missing/invalid for " + name)
        if reference is not None and labels != reference:
            raise ValueError("Models use different class-ID mappings; cannot evaluate one shared label manifest")
        reference = labels
    if reference is None:
        raise ValueError("No models for offline evaluation")
    return hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()


class SommelierExecutor(ProteusExecutor):
    def __init__(self, backend, **kwargs):
        if not backend.enabled:
            raise ValueError("Enable at least one ImageNet model for Sommelier")
        self.served_variants = {}
        self.previous_native_variant = None
        self.actual_variant_switches = 0
        self.switch_load_ms_total = 0.0
        super().__init__(backend, solver=solve_fixed_placement, **kwargs)

    def submit(self, request):
        if request.dispatch not in {"", "sommelier"}:
            raise ValueError("Dispatch must be empty or sommelier")
        # Never mutate the caller's protobuf; parent accepts its own dispatch tag.
        copied = InferRequest.FromString(request.SerializeToString())
        copied.dispatch = ""
        with self.condition:
            native = self.mode == "proteus"
            future = super().submit(copied)

        def completed(result):
            if not native or result.cancelled() or result.exception() is not None:
                return
            reply = result.result()
            if reply.rejected:
                return
            with self.condition:
                name = reply.variant
                self.served_variants[name] = self.served_variants.get(name, 0) + 1
                if self.previous_native_variant is not None and self.previous_native_variant != name:
                    self.actual_variant_switches += 1
                    self.switch_load_ms_total += reply.timings.get("worker.load", 0.0)
                self.previous_native_variant = name
        future.add_done_callback(completed)
        return future

    def set_mode(self, mode):
        if mode not in {"fixed", "native"}:
            raise ValueError("Sommelier mode must be native or fixed diagnostic")
        super().set_mode("proteus" if mode == "native" else "fixed")
        return {"mode": mode, "baseline": "sommelier"}

    def profile(self, image, trials=30, models=None):
        result = super().profile(image, trials, models)
        result["scope"] = "Actual GPU profiles for the fixed-placement Sommelier comparison port"
        return result

    def evaluate(self, manifest):
        registry = {name: self.backend.registry[name] for name in self.backend.enabled}
        mapping_hash = verify_class_mapping(registry, os.environ.get("HF_HOME", "/models"))
        result = super().evaluate(manifest)
        result["class_mapping_sha256"] = mapping_hash
        return result

    def state(self):
        with self.condition:
            result = super().state()
            registry = {name: self.backend.registry[name] for name in self.backend.enabled}
            offline = {name: self.measured_accuracy.get(name, item) for name, item in registry.items()}
            try:
                normalized = normalized_accuracies(registry, self.measured_accuracy)
            except ValueError:
                normalized = None
            result.update({"implementation": "proteus-sommelier-fixed-placement-single-gpu-port",
                "baseline": "sommelier", "mode": "native" if self.mode == "proteus" else "fixed",
                "placement": {"device": "cuda:0", "application": self.backend.application, "fixed": True},
                "model_placement_dynamic": False, "replication_enabled": False,
                "model_scaling_enabled": self.mode == "proteus" and self.plan is not None and len(registry) > 1,
                "classification_accuracy": None, "offline_accuracy": offline,
                "normalized_model_accuracy": normalized, "successful_native_requests_by_model": dict(self.served_variants),
                "actual_variant_switches": self.actual_variant_switches,
                "switch_load_ms_total": self.switch_load_ms_total,
                "scope_note": "Proteus's Sommelier comparison policy reduction, not the original semantic index. "
                    "One GPU/application has no placement choice: matched inputs give the same policy as Proteus. "
                    "Repeated JPEG requests do not measure classification accuracy. Model loading remains on the request path."})
            return result
