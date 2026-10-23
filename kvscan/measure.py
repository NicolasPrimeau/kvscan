"""Measured mode: per-layer sensitivity of a model to a quantized KV cache, gradient-free.

A custom transformers attention function quantizes the post-RoPE keys and values of selected layers (exactly what
an engine caches), then calls SDPA. Sensitivity of a tensor = KL(original || quantized) of the next-token
distribution with only that tensor quantized. Sensitivities add up roughly, which makes planning cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from .formats import FORMATS

_plan: dict = {}  # ("k" | "v", layer) -> format name


def _kvscan_attention(module, query, key, value, attention_mask, **kwargs):
    i = getattr(module, "layer_idx", None)
    tk, tv = _plan.get(("k", i)), _plan.get(("v", i))
    if tk:
        key = FORMATS[tk].fn(key, "k").to(key.dtype)
    if tv:
        value = FORMATS[tv].fn(value, "v").to(value.dtype)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("kvscan", _kvscan_attention)


@dataclass
class Session:
    model_id: str
    device: str = "auto"
    seq: int = 512
    ncal: int = 2
    neval: int = 4
    text: str | None = None
    model: object = field(init=False)
    n_layers: int = field(init=False)

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        if self.device == "cuda":  # spill to CPU RAM if the model does not fit in VRAM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype, attn_implementation="kvscan", device_map="auto").eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype, attn_implementation="kvscan").to(self.device).eval()
        self.n_layers = self.model.config.get_text_config().num_hidden_layers
        ids = self.tok(self.text, return_tensors="pt").input_ids[0]
        need = (self.ncal + self.neval) * self.seq
        if len(ids) < need:
            raise SystemExit(f"text too short: {len(ids)} tokens, need {need} (--seq x (--ncal + --neval))")
        # calibration from the first part, evaluation from the end (disjoint)
        cal = [ids[k * self.seq : (k + 1) * self.seq] for k in range(self.ncal)]
        ev = [ids[len(ids) - (k + 1) * self.seq : len(ids) - k * self.seq] for k in range(self.neval)]
        self.XC = torch.stack(cal).to(self.device)
        self.XE = torch.stack(ev).to(self.device)
        self.base_c = self._logprobs(self.XC)
        self.base_e = self._logprobs(self.XE)

    def _logprobs(self, X):
        _plan.clear()
        return self._run(X)

    @torch.no_grad()
    def _run(self, X):
        out = []
        for b in range(X.shape[0]):  # one segment at a time keeps full-vocab logits small
            out.append(torch.log_softmax(self.model(X[b : b + 1]).logits[0].float(), -1))
        return out

    def kl(self, plan: dict, split: str = "eval") -> float:
        X, base = (self.XE, self.base_e) if split == "eval" else (self.XC, self.base_c)
        _plan.clear()
        _plan.update(plan)
        try:
            lps = self._run(X)
        finally:
            _plan.clear()
        return sum((b.exp() * (b - lp)).sum(-1).mean().item() for b, lp in zip(base, lps)) / len(base)

    def sensitivity(self, fmt: str, progress=None) -> dict:
        """(kv, layer) -> calibration KL with only that tensor quantized to fmt."""
        out = {}
        for i in range(self.n_layers):
            for kv in "kv":
                out[(kv, i)] = self.kl({(kv, i): fmt}, split="cal")
            if progress:
                progress(i, out)
        return out
