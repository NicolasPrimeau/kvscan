"""Weights-only scan: large key-projection biases add a constant offset to every key, which stretches the
quantizer's range and starves the actual signal. On Qwen2.5-1.5B the four layers with the largest key bias are
exactly the four most quantization-sensitive layers. Reads safetensors directly; no model load, no GPU."""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_safetensors_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session
from safetensors import safe_open

KEY_BIAS = re.compile(r"layers\.(\d+)\.self_attn\.k_proj\.bias$")


_DT = {"F32": np.float32, "F16": np.float16, "BF16": "bf16"}


def _decode(raw: bytes, dtype: str) -> torch.Tensor:
    if dtype == "BF16":
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
    return torch.from_numpy(np.frombuffer(raw, dtype=_DT[dtype]).astype(np.float32))


def _remote_key_biases(repo: str) -> dict:
    """Fetch only the key-bias tensors from the Hub with HTTP range requests (no full download)."""
    meta = get_safetensors_metadata(repo)
    session, headers = get_session(), build_hf_headers()
    rows = {}
    for fname, fm in meta.files_metadata.items():
        names = [n for n in fm.tensors if KEY_BIAS.search(n)]
        if not names:
            continue
        url = hf_hub_url(repo, fname)
        r = session.get(url, headers={**headers, "Range": "bytes=0-7"}, follow_redirects=True)
        r.raise_for_status()
        start = 8 + struct.unpack("<Q", r.content[:8])[0]
        for n in names:
            info = fm.tensors[n]
            a, b = info.data_offsets
            r = session.get(url, headers={**headers, "Range": f"bytes={start + a}-{start + b - 1}"}, follow_redirects=True)
            r.raise_for_status()
            rows[int(KEY_BIAS.search(n).group(1))] = _decode(r.content, info.dtype).abs().max().item()
    return rows


def _local_key_biases(root: Path) -> dict:
    rows = {}
    for f in sorted(root.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for name in s.keys():
                m = KEY_BIAS.search(name)
                if m:
                    rows[int(m.group(1))] = s.get_tensor(name).float().abs().max().item()
    return rows


def scan(model: str) -> dict:
    if Path(model).is_dir():
        root = Path(model)
        rows = _local_key_biases(root)
        cfg = json.loads((root / "config.json").read_text())
    else:
        root = model
        rows = _remote_key_biases(model)
        cfg = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    n_layers = cfg.get("num_hidden_layers") or cfg.get("text_config", {}).get("num_hidden_layers")
    if not rows:
        return dict(root=str(root), n_layers=n_layers, has_key_bias=False, layers=[], flagged=[])
    vals = sorted(rows.values())
    median = vals[len(vals) // 2]
    # flag layers whose largest key bias is far above the model's typical layer
    flagged = sorted((i for i, v in rows.items() if v > 4 * median and v > 16), key=lambda i: -rows[i])
    return dict(root=str(root), n_layers=n_layers, has_key_bias=True, median=median,
                layers=[dict(layer=i, max_abs_key_bias=v) for i, v in sorted(rows.items())], flagged=flagged)
