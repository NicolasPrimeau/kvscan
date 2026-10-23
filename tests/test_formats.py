import pytest
import torch

from kvscan.formats import FORMATS, LLAMACPP_BPE, LLAMACPP_LADDER


def kv(seed=0, d=128):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 4, 64, d, generator=g) * 3  # [batch, kv_heads, tokens, head_dim]


def rel_mse(x, y):
    return ((x - y) ** 2).mean().item() / (x**2).mean().item()


@pytest.mark.parametrize("name", list(FORMATS))
def test_shape_and_finite(name):
    x = kv()
    for side in "kv":
        y = FORMATS[name].fn(x, side)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


@pytest.mark.parametrize("name,max_err", [
    ("int8_per_token_head", 1e-4), ("fp8_per_token_head", 2e-3), ("q8_0", 1e-4),
    ("q5_0", 5e-3), ("q5_1", 5e-3), ("q4_0", 2e-2), ("q4_1", 2e-2), ("int4_per_token_head", 3e-2),
])
def test_error_levels(name, max_err):
    x = kv()
    assert rel_mse(x, FORMATS[name].fn(x, "k")) < max_err


def test_more_bits_less_error():
    x = kv()
    errs = [rel_mse(x, FORMATS[t].fn(x, "k")) for t in ["q4_0", "q5_0", "q8_0"]]
    assert errs[0] > errs[1] > errs[2]


def test_constant_offset_hurts_absmax_formats():
    """A large constant offset (like a key bias) inflates the range and destroys low-bit resolution."""
    x = kv()
    xb = x + 100.0
    err_plain = ((FORMATS["q4_0"].fn(x, "k") - x) ** 2).mean()
    err_bias = ((FORMATS["q4_0"].fn(xb, "k") - xb) ** 2).mean()
    assert err_bias > 10 * err_plain


def test_fp8_default_scale_clips():
    x = torch.full((1, 1, 1, 128), 1000.0)
    assert FORMATS["fp8"].fn(x, "k").max().item() == 448.0


def test_ladder_bpe_increasing():
    b = [LLAMACPP_BPE[t] for t in LLAMACPP_LADDER]
    assert b == sorted(b)
