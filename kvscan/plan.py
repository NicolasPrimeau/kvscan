"""Turn per-tensor sensitivities into engine settings."""
from __future__ import annotations

import numpy as np

from .formats import LLAMACPP_BPE, LLAMACPP_LADDER


def vllm_skip_curve(sens: dict, n_layers: int, max_skip: int = 8):
    """vLLM skips whole layers (K and V). Greedy by layer sensitivity: [(skip_list, predicted_kl), ...]."""
    layer = {i: sens[("k", i)] + sens[("v", i)] for i in range(n_layers)}
    order = sorted(layer, key=lambda i: -layer[i])
    total = sum(layer.values())
    curve, skipped, rem = [([], total)], [], total
    for i in order[:max_skip]:
        skipped.append(i)
        rem -= layer[i]
        curve.append((sorted(skipped), max(rem, 0.0)))
    return curve


def pick_skip(curve, target_kl: float):
    """Smallest skip list whose predicted KL meets the target (or the best available)."""
    for skip, pred in curve:
        if pred <= target_kl:
            return skip, pred
    return curve[-1]


def boundary_heuristic(n_layers: int, n: int = 2):
    """vLLM TurboQuant's fixed rule: first n and last n layers."""
    return sorted(set(range(n)) | set(range(n_layers - n, n_layers)))


def llamacpp_plan(table: dict, n_layers: int, budget_bpe: float):
    """Per-tensor llama.cpp types minimizing predicted KL under a mean-bpe budget (Lagrangian sweep).
    table[(kv, layer, type)] = KL with only that tensor at that type (f16 = 0)."""
    keys = [(kv, i) for i in range(n_layers) for kv in "kv"]
    best = None
    for lam in np.logspace(-9, 2, 600):
        p = {k: min(LLAMACPP_LADDER, key=lambda t: table[(k[0], k[1], t)] + lam * LLAMACPP_BPE[t]) for k in keys}
        bpe = float(np.mean([LLAMACPP_BPE[t] for t in p.values()]))
        if bpe <= budget_bpe + 1e-9:
            pred = sum(table[(k[0], k[1], t)] for k, t in p.items())
            if best is None or pred < best[1]:
                best = (p, pred, bpe)
    return best


def llamacpp_env(plan: dict, default: str = "q4_0") -> str:
    return ",".join(f"{kv}{i}={t}" for (kv, i), t in sorted(plan.items(), key=lambda x: (x[0][1], x[0][0])) if t != default)
