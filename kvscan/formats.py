"""Round-trip simulations of the KV cache formats used by vLLM and llama.cpp.

Every quantizer takes post-RoPE keys or values shaped [batch, kv_heads, tokens, head_dim] and returns a tensor of
the same shape after quantize -> dequantize, so the model sees exactly what the engine would read back.
Each format records its storage cost in bits per element (bpe), including scales.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

FP8_MAX = 448.0


@dataclass(frozen=True)
class Format:
    name: str
    engine: str  # "vllm" or "llama.cpp"
    bpe: float
    fn: Callable[[torch.Tensor, str], torch.Tensor]  # (x, "k" | "v") -> x_hat


def _round_half_away(x):
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _fp16(x):
    return x.half().float()


def _hadamard(n: int, device=None) -> torch.Tensor:
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def _largest_pow2_divisor(n: int) -> int:
    return n & -n


_rht_cache: dict = {}


def _rht(d: int, device) -> torch.Tensor:
    """A fixed randomized Hadamard transform (random signs, then Hadamard). It is orthogonal, so it is undone exactly."""
    key = (d, str(device))
    if key not in _rht_cache:
        n = _largest_pow2_divisor(d)
        g = torch.Generator().manual_seed(0)
        signs = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device)
        _rht_cache[key] = (torch.diag(signs) @ _hadamard(n, device), n)
    return _rht_cache[key]


def _rotate(x, R, n):
    sh = x.shape
    return (x.reshape(-1, n) @ R).reshape(sh)


# ---------------------------------------------------------------- vLLM


def vllm_int8_per_token_head(x, kv):
    s = (x.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-6)
    return torch.clamp(_round_half_away(x / s), -128, 127) * s


def vllm_fp8_per_token_head(x, kv):
    s = (x.abs().amax(-1, keepdim=True) / FP8_MAX).clamp_min(1e-6)
    return (x / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s


def vllm_fp8(x, kv):
    """--kv-cache-dtype fp8 with the default (uncalibrated) per-tensor scale of 1.0."""
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


def vllm_int4_per_token_head(x, kv):
    """One randomized Hadamard rotation, then asymmetric 4-bit per (token, head) with an integer zero-point."""
    R, n = _rht(x.shape[-1], x.device)
    y = _rotate(x.float(), R, n)
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    s = ((mx - mn) / 15.0).clamp_min(1e-6)
    zp = torch.clamp(_round_half_away(-mn / s), 0, 15)
    q = torch.clamp(_round_half_away(y / s + zp), 0, 15)
    return _rotate((q - zp) * s, R.T, n)


# ---------------------------------------------------------------- llama.cpp (ggml block formats + default attn_rot)


def _ggml_blocks(x, t):
    b = x.reshape(-1, 32).float()
    if t in ("q4_0", "q5_0"):
        half = 8 if t == "q4_0" else 16
        mx = b.gather(1, b.abs().argmax(1, keepdim=True))
        d = _fp16(mx / -half)
        inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))
        y = (torch.clamp(torch.floor(b * inv + half + 0.5), 0, 2 * half - 1) - half) * d
    elif t in ("q4_1", "q5_1"):
        lv = 15 if t == "q4_1" else 31
        mn, mx = b.min(1, keepdim=True).values, b.max(1, keepdim=True).values
        d, m = _fp16((mx - mn) / lv), _fp16(mn)
        inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))
        y = torch.clamp(torch.floor((b - m) * inv + 0.5), 0, lv) * d + m
    elif t == "q8_0":
        d = _fp16(b.abs().max(1, keepdim=True).values / 127)
        inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))
        y = torch.round(b * inv) * d
    else:
        raise ValueError(t)
    return y.reshape(x.shape)


def _llamacpp(t):
    def fn(x, kv):
        d = x.shape[-1]
        # attn_rot: Hadamard of size 64..head_dim for K, 64 for V (llama-kv-cache.cpp)
        n = 64
        if kv == "k":
            while d % (2 * n) == 0:
                n *= 2
        if d % 64 != 0:
            return _ggml_blocks(x, t)
        H = _hadamard(n, x.device)
        return _rotate(_ggml_blocks(_rotate(x.float(), H, n), t), H.T, n)

    return fn


FORMATS: dict[str, Format] = {
    f.name: f
    for f in [
        Format("int4_per_token_head", "vllm", 4 + 32 / 128, vllm_int4_per_token_head),
        Format("int8_per_token_head", "vllm", 8 + 32 / 128, vllm_int8_per_token_head),
        Format("fp8_per_token_head", "vllm", 8 + 32 / 128, vllm_fp8_per_token_head),
        Format("fp8", "vllm", 8.0, vllm_fp8),
        Format("q4_0", "llama.cpp", 4.5, _llamacpp("q4_0")),
        Format("q4_1", "llama.cpp", 5.0, _llamacpp("q4_1")),
        Format("q5_0", "llama.cpp", 5.5, _llamacpp("q5_0")),
        Format("q5_1", "llama.cpp", 6.0, _llamacpp("q5_1")),
        Format("q8_0", "llama.cpp", 8.5, _llamacpp("q8_0")),
    ]
}
LLAMACPP_LADDER = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16"]
LLAMACPP_BPE = {**{n: FORMATS[n].bpe for n in LLAMACPP_LADDER[:-1]}, "f16": 16.0}
