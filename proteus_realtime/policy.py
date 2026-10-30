"""Wall-clock port of Proteus rules, scoped to one GPU / one application.

Source: core/predictor.py accscale and algorithms/ilp.py at 6967c932.
Single GPU makes routing z=1; the bilinear objective reduces to a linear MILP.
"""
import math
import time
from dataclasses import dataclass, asdict
from .profiling import execution_latency_ms


@dataclass
class Plan:
    model: str
    max_batch: int
    capacity_qps: float
    demand_qps: float
    beta: float
    normalized_accuracy: float
    solver_ms: float
    solver_status: str
    routing_fraction: float = 1.0

    def to_dict(self):
        return asdict(self)


def solve_milp(profiles, accuracies, demand, slo_ms, beta=1.0, minimum_beta=0.2,
               time_limit=2.0):
    import gurobipy as gp
    from gurobipy import GRB
    if not math.isfinite(demand) or demand <= 0:
        raise ValueError("MILP demand must be positive")
    if not 0 < minimum_beta <= beta <= 1:
        raise ValueError("Require 0 < minimum_beta <= beta <= 1")
    candidates = {}
    for name, profile in profiles.items():
        valid = [int(size) for size, row in profile["batches"].items()
                 if execution_latency_ms(row) < slo_ms / 2]
        if valid:
            batch = max(valid)
            latency = float(profile["batches"][str(batch)]["p50_ms"])
            candidates[name] = (batch, math.floor(1000 * batch / latency))
    if not candidates:
        raise ValueError("No measured model/batch can satisfy execution-time < SLO/2")
    started = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        target = beta
        while target >= minimum_beta - 1e-9:
            with gp.Model("Proteus-single-GPU", env=env) as model:
                model.Params.TimeLimit = time_limit
                model.Params.Threads = 1
                model.Params.MIPGap = 0.0
                x = model.addVars(list(candidates), vtype=GRB.BINARY, name="placement")
                y = model.addVar(lb=0, name="served_qps")
                model.addConstr(gp.quicksum(x.values()) == 1, "one_model_per_device")
                model.addConstr(y <= gp.quicksum(candidates[n][1] * x[n] for n in candidates), "capacity")
                model.addConstr(y <= demand, "demand")
                model.addConstr(y >= target * demand, "minimum_served_fraction")
                model.setObjective(gp.quicksum(accuracies[n] * x[n] for n in candidates), GRB.MAXIMIZE)
                model.optimize()
                if model.Status == GRB.OPTIMAL:
                    selected = next(n for n in candidates if x[n].X > 0.5)
                    batch, capacity = candidates[selected]
                    return Plan(selected, batch, capacity, demand, target, accuracies[selected],
                                (time.perf_counter() - started) * 1000, "optimal")
                if model.Status != GRB.INFEASIBLE:
                    raise RuntimeError(f"MILP did not prove an optimum: status={model.Status}")
            if target <= minimum_beta + 1e-9:
                break
            target = max(minimum_beta, round(target - 0.05, 8))
    finally:
        env.dispose()
    raise ValueError("Demand infeasible even after beta relaxation; retain last plan")


@dataclass
class BatchDecision:
    action: str
    size: int = 0
    wake_at: float = 0.0


def adaptive_decision(deadlines, latencies_ms, max_batch, now, guard_ms=2.0):
    """Non-work-conserving accscale: wait until head deadline minus batch time.

    deadlines are monotonic seconds. Only fully filled profiled sizes are used.
    latencies_ms carries the configured execution prediction, not necessarily P50.
    """
    if not deadlines:
        return BatchDecision("idle")
    sizes = sorted(int(size) for size in latencies_ms if int(size) <= max_batch)
    if not sizes or sizes[0] != 1:
        raise ValueError("Measured batch-1 latency is required")
    count = min(len(deadlines), max_batch)
    size = max(b for b in sizes if b <= count)
    duration = (float(latencies_ms[str(size)]) + guard_ms) / 1000
    head_deadline = min(deadlines[:size])
    # The guard is early-dispatch slack, not an extra feasibility requirement.
    # Otherwise normal timer overshoot would drop every just-woken warm request.
    if now + float(latencies_ms[str(size)]) / 1000 > head_deadline + 1e-9:
        return BatchDecision("drop", 1)
    if len(deadlines) > max_batch:
        return BatchDecision("run", max_batch)
    wake_at = head_deadline - duration
    if now >= wake_at - 1e-6:
        return BatchDecision("run", size)
    return BatchDecision("wait", size, wake_at)
