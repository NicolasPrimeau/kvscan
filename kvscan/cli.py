"""kvscan: find the layers of a model that break under a quantized KV cache, and print the engine flags that fix it."""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import resources

from . import plan as P
from .formats import FORMATS, LLAMACPP_LADDER


def _say(*a):
    print(*a, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kvscan", description=__doc__)
    ap.add_argument("model", help="Hugging Face model id or local directory")
    ap.add_argument("--dtype", default="int4_per_token_head", choices=list(FORMATS),
                    help="KV cache format to test (vLLM names, or llama.cpp q4_0/q4_1/q5_0/q5_1/q8_0)")
    ap.add_argument("--scan-only", action="store_true", help="weights-only scan (seconds, no model load)")
    ap.add_argument("--target-kl", type=float, default=0.01, help="vLLM: KL target for choosing skip layers")
    ap.add_argument("--budget", type=float, default=5.0, help="llama.cpp: mean bits per element for the plan")
    ap.add_argument("--text", help="calibration/eval text file (default: bundled public-domain sample)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--ncal", type=int, default=2)
    ap.add_argument("--neval", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", help="write full results to this file")
    a = ap.parse_args(argv)
    report = dict(model=a.model, dtype=a.dtype)

    from .weights import scan

    t0 = time.time()
    w = scan(a.model)
    report["weights"] = w
    if w["has_key_bias"]:
        flag = ", ".join(f"{i} ({next(r['max_abs_key_bias'] for r in w['layers'] if r['layer'] == i):.0f})" for i in w["flagged"])
        _say(f"weights scan ({time.time() - t0:.0f}s): key-projection biases present, median max|bias| {w['median']:.1f}")
        _say(f"  outlier layers (max|bias|): {flag or 'none'}")
    else:
        _say(f"weights scan ({time.time() - t0:.0f}s): no key-projection biases (bias check does not apply; use measured mode)")
    if a.scan_only:
        return _finish(a, report)

    from .measure import Session

    text = open(a.text, encoding="utf-8", errors="ignore").read() if a.text else \
        resources.files("kvscan").joinpath("data/sample.txt").read_text(encoding="utf-8")
    t0 = time.time()
    S = Session(a.model, device=a.device, seq=a.seq, ncal=a.ncal, neval=a.neval, text=text)
    L = S.n_layers
    _say(f"loaded {a.model} on {S.device} ({L} layers) in {time.time() - t0:.0f}s; measuring {a.dtype} per layer ...")
    fmt = FORMATS[a.dtype]

    def prog(i, out):
        _say(f"  layer {i:3d}: K {out[('k', i)]:.2e}  V {out[('v', i)]:.2e}")

    if fmt.engine == "vllm":
        sens = S.sensitivity(a.dtype, prog)
        report["sensitivity"] = {f"{kv}{i}": v for (kv, i), v in sens.items()}
        curve = P.vllm_skip_curve(sens, L)
        skip, pred = P.pick_skip(curve, a.target_kl)
        heur = P.boundary_heuristic(L)
        top_same = curve[min(len(heur), len(curve) - 1)][0]
        everything = {(kv, i): a.dtype for i in range(L) for kv in "kv"}

        def without(layers):
            return {k: t for k, t in everything.items() if k[1] not in layers}

        rows = [("uniform " + a.dtype, [], S.kl(everything)),
                ("first/last 2 (TurboQuant rule)", heur, S.kl(without(heur))),
                (f"measured top-{len(top_same)}", top_same, S.kl(without(top_same)))]
        if skip not in (heur, top_same, []):
            rows.append((f"recommended (target {a.target_kl:g})", skip, S.kl(without(skip))))
        _say("\nheld-out KL (lower is better):")
        for name, sk, kl in rows:
            _say(f"  {name:34s} skip={sk!s:22s} KL {kl:.4f}")
        report["eval"] = [dict(name=n, skip=s, kl=k) for n, s, k in rows]
        best = min(rows, key=lambda r: r[2] if r[1] or r[0].startswith("uniform") else 9e9)
        flags = f"--kv-cache-dtype {a.dtype}" + (f" --kv-cache-dtype-skip-layers {' '.join(map(str, skip))}" if skip else "")
        _say(f"\nrecommend (vLLM):\n  {flags}")
        report["recommend"] = flags
    else:
        table = {}
        for t in LLAMACPP_LADDER[:-1]:
            _say(f" measuring {t} ...")
            s = S.sensitivity(t)
            table.update({(kv, i, t): v for (kv, i), v in s.items()})
        table.update({(kv, i, "f16"): 0.0 for i in range(L) for kv in "kv"})
        report["table"] = {f"{kv}{i}:{t}": v for (kv, i, t), v in table.items()}
        p, pred, bpe = P.llamacpp_plan(table, L, a.budget)
        uni = {(kv, i): a.dtype for i in range(L) for kv in "kv"}
        rows = [(f"uniform {a.dtype}", FORMATS[a.dtype].bpe, S.kl(uni)),
                ("uniform q8_0", 8.5, S.kl({k: "q8_0" for k in uni})),
                (f"planned (budget {a.budget:g})", bpe, S.kl(p))]
        _say("\nheld-out KL (lower is better):")
        for name, b, kl in rows:
            _say(f"  {name:28s} {b:5.2f} bpe   KL {kl:.4f}")
        env = P.llamacpp_env(p)
        _say(f"\nrecommend (llama.cpp, needs per-layer KV types):\n  LLAMA_KV_TYPE_PLAN=\"{env}\" -ctk q4_0 -ctv q4_0")
        report["eval"] = [dict(name=n, bpe=b, kl=k) for n, b, k in rows]
        report["recommend"] = env
    return _finish(a, report)


def _finish(a, report):
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
