# kvscan

[![ci](https://github.com/NicolasPrimeau/kvscan/actions/workflows/ci.yml/badge.svg)](https://github.com/NicolasPrimeau/kvscan/actions/workflows/ci.yml)

**Find the layers of an LLM that break under a quantized KV cache, and get the vLLM / llama.cpp flags that fix it.**

Quantizing the KV cache to 4 or 8 bits saves a lot of memory, and on most layers it is harmless. But on some models
a handful of layers are hypersensitive. One bad layer is enough to make the model answer at chance, while the
engine reports success. Today people find this out from bad outputs and folklore ("keep the first layers in
fp16"). kvscan measures it.

```text
$ kvscan Qwen/Qwen2.5-1.5B --scan-only
weights scan (5s): key-projection biases present, median max|bias| 5.2
  outlier layers (max|bias|): 0 (316), 1 (102), 15 (51), 7 (34)
```

## Install

```bash
pip install git+https://github.com/NicolasPrimeau/kvscan
```

Python ≥ 3.10, PyTorch, transformers ≥ 5. The weights scan needs neither a GPU nor a model download.

## Usage

```bash
# 1. Weights-only scan: seconds, reads only the needed bytes from the Hub (HTTP range requests)
kvscan Qwen/Qwen2.5-72B-Instruct --scan-only

# 2. Measured mode for vLLM: per-layer KL under a vLLM KV format, then a skip list
kvscan Qwen/Qwen2.5-1.5B --dtype int4_per_token_head        # also: fp8, fp8_per_token_head, int8_per_token_head

# 3. Measured mode for llama.cpp: per-tensor types under a bits-per-element budget
kvscan Qwen/Qwen2.5-1.5B --dtype q4_0 --budget 5.0
```

Options: `--text FILE` sets your own calibration and evaluation text (by default, a bundled public-domain
sample). `--json out.json` writes the full report. `--device cuda` runs on a GPU; models larger than VRAM spill to
CPU RAM automatically. `--target-kl` sets the quality target used to pick vLLM skip layers.

The output ends with flags you can paste:

```text
recommend (vLLM):
  --kv-cache-dtype int4_per_token_head --kv-cache-dtype-skip-layers 0 1 7 15
```

## How it works

1. **Weights scan.** A large bias on `k_proj` adds the same offset to every key. Absmax and min/max quantizers
   spend their range on that offset and starve the signal. kvscan reads only the key-bias tensors from the
   safetensors files and flags layers far above the model's median. On Qwen2.5-1.5B, the four flagged layers are
   exactly the four most sensitive layers measured.
2. **Measured mode.** A custom transformers attention function quantizes the **post-RoPE** keys and values of
   chosen layers, exactly what an engine caches, then calls SDPA. This works for any model on transformers'
   attention interface. Sensitivity of a tensor = KL of the next-token distribution with only that tensor
   quantized. Sensitivities add up roughly, so plans are cheap: greedy for vLLM skip lists, and a Lagrangian
   knapsack for llama.cpp per-tensor types. Every plan is then checked with everything quantized at once, on
   held-out text.
3. **Faithful formats.** The quantizers mirror the engines' kernels:

   | format | engine | simulated as |
   |---|---|---|
   | `int4_per_token_head` | vLLM | randomized Hadamard, asymmetric 4-bit per (token, head), integer zero-point |
   | `int8_per_token_head` / `fp8_per_token_head` | vLLM | symmetric absmax per (token, head) |
   | `fp8` | vLLM | e4m3 with the default (uncalibrated) scale of 1.0 |
   | `q4_0` `q4_1` `q5_0` `q5_1` `q8_0` | llama.cpp | ggml 32-element blocks, fp16 scales, default Hadamard rotation |

   Against llama.cpp's own `--kl-divergence`, the simulation predicted q8_0 at 0.0027 vs 0.0024 measured, and
   q4_0 at 5.27 vs 5.17.

## Findings

Details are in [FINDINGS.md](FINDINGS.md). The highlights:

- On Qwen2.5-1.5B, **one tensor** (layer 0 keys) turns a 4-bit cache from fine into broken (KL 4.7 alone).
  Hadamard rotation does not fix it.
- **Per-layer llama.cpp types give q8_0 quality with 24% less KV memory**, or near-q4 size (4.75 bpe) at KL
  0.018. These were measured with llama.cpp itself.
- On **Qwen2.5-7B and its derivatives, the largest key bias is in the last layer**, not the first. Fixed rules
  ("protect layer 0", "first and last 2") don't match what the weights show.
- Background: an audio-codec ("psychoacoustic") view of the KV cache, including masking measurements and an
  honest negative result.

## Limitations

- kvscan simulates the engines. It doesn't run them. The llama.cpp path is validated against llama.cpp. vLLM skip
  lists still need validating inside vLLM (in progress).
- KL on a short text sample is a quality proxy, not task accuracy. Use `--text` with text from your domain.
- TurboQuant and NVFP4 formats aren't simulated yet. MLA models (DeepSeek-V3-style) aren't supported.
- **Per-layer llama.cpp types need a patch.** Upstream llama.cpp sets one KV type for every layer. The prototype
  patch is in [`contrib/llama.cpp-kv-type-plan.patch`](contrib/llama.cpp-kv-type-plan.patch); it adds the
  `LLAMA_KV_TYPE_PLAN` environment variable. vLLM supports skip lists natively
  (`--kv-cache-dtype-skip-layers`).

## Development

```bash
uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install -e . pytest
pytest -q tests                 # CPU-only, about 2 s
./run_gpu_suite.sh              # measured mode on a model suite with an NVIDIA GPU
```

## License

Apache-2.0. The bundled calibration sample is an excerpt of *Pride and Prejudice* (public domain, Project
Gutenberg).
