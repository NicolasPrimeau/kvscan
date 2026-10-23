import json

import torch
from safetensors.torch import save_file

from kvscan import plan as P
from kvscan.formats import LLAMACPP_LADDER
from kvscan.weights import scan


def test_skip_curve_greedy_and_monotone():
    sens = {("k", i): 0.001 for i in range(6)} | {("v", i): 0.0005 for i in range(6)}
    sens[("k", 0)] = 5.0
    sens[("k", 4)] = 0.05
    curve = P.vllm_skip_curve(sens, 6, max_skip=3)
    assert curve[1][0] == [0] and curve[2][0] == [0, 4]
    preds = [p for _, p in curve]
    assert preds == sorted(preds, reverse=True)
    skip, pred = P.pick_skip(curve, 0.02)
    assert skip == [0, 4] and pred <= 0.02


def test_boundary_heuristic():
    assert P.boundary_heuristic(28) == [0, 1, 26, 27]


def test_llamacpp_plan_upgrades_only_sensitive_tensor():
    L = 4
    table = {}
    for i in range(L):
        for kv in "kv":
            base = 3.0 if (kv, i) == ("k", 0) else 1e-3
            for j, t in enumerate(LLAMACPP_LADDER):
                table[(kv, i, t)] = 0.0 if t == "f16" else base / 4**j
    p, pred, bpe = P.llamacpp_plan(table, L, budget_bpe=5.0)
    assert bpe <= 5.0
    assert p[("k", 0)] in ("q8_0", "f16")
    assert sum(t != "q4_0" for t in p.values()) <= 2
    env = P.llamacpp_env(p)
    assert env.startswith("k0=")


def test_weights_scan_local(tmp_path):
    L = 8
    tensors = {}
    for i in range(L):
        b = torch.full((256,), 2.0 + i % 3)
        if i in (0, 5):
            b[7] = 300.0
        tensors[f"model.layers.{i}.self_attn.k_proj.bias"] = b
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": L}))
    r = scan(str(tmp_path))
    assert r["has_key_bias"] and r["flagged"] == [0, 5]


def test_weights_scan_no_bias(tmp_path):
    save_file({"model.layers.0.self_attn.k_proj.weight": torch.zeros(4, 4)}, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    assert scan(str(tmp_path))["has_key_bias"] is False
