# Findings

Notes from the experiments behind kvscan (October 2026). Most numbers come from **Qwen2.5-1.5B** on CPU.
KL is `KL(original ‖ quantized)` of the next-token distribution, in nats per token, averaged over held-out text.
For scale: 0.001 is invisible, 0.01 is a small but measurable drift, and above 1 the model is broken.

## 1. A few tensors decide whether a quantized KV cache works

Quantizing **one tensor at a time** to llama.cpp `q4_0` (with llama.cpp's default Hadamard rotation):

| tensor | KL with only this tensor at q4_0 |
|---|---|
| layer 0 keys | **4.7** (the model is broken) |
| layer 1 keys | 0.061 |
| layer 7 keys | 0.040 |
| layer 15 keys | 0.0039 |
| any other tensor (52 of 56) | ≤ 0.0015 |

Values are almost never the problem. Uniform `q4_0` on this model gives KL 5.2: one tensor ruins the whole cache.

## 2. The sensitive layers are the ones with large key-projection biases

A large constant bias on `k_proj` shifts every key by the same vector. Absmax and min/max quantizers then spend
their range on that offset, and the actual key signal gets too few levels.

| layer | max \|key bias\| | sensitivity rank |
|---|---|---|
| 0 | 316 | 1 |
| 1 | 102 | 2 |
| 15 | 51 | 4 |
| 7 | 34 | 3 |
| median layer | 5.2 | — |

The four layers with the largest biases are exactly the four most sensitive layers. Below the top four, bias size
predicts little (Spearman 0.38 over all layers), so the measured mode exists for everything else.

The bias pattern differs across the Qwen2.5 family. The weights scan reads it in seconds without downloading:

| model | outlier layers (max \|key bias\|) |
|---|---|
| Qwen2.5-1.5B | 0 (316), 1 (102), 15 (51), 7 (34) |
| Qwen2.5-7B | **27 (414)**, 0 (171), 3 (68), 1 (62), 19 (45), 13 (40) |
| DeepSeek-R1-Distill-Qwen-7B | **27 (428)**, 0 (178), 1 (73), 3 (68), 19 (45), 13 (40) |
| Qwen2.5-Coder-7B-Instruct | **27 (442)**, 0 (188), 1 (86), 3 (67) |
| Qwen2.5-72B-Instruct | 0 (18), with a median of 3.9, so mild |
| Qwen3-4B | no key biases |

On the 7B models the largest bias is in the **last** layer. A fixed "protect layer 0" rule does not cover it.
Whether layer 27 is actually *sensitive* still needs a measured 7B run (pending, GPU).

## 3. Per-layer types: q8_0 quality at 24% less memory (real llama.cpp)

Calibrated per-tensor types chosen under a bits-per-element budget, evaluated with llama.cpp's own
`llama-perplexity --kl-divergence` (16 × 512 held-out tokens, `-fa on`, default rotation, prototype
[patch](contrib/llama.cpp-kv-type-plan.patch)):

| KV cache | bits/element | KLD |
|---|---|---|
| uniform q4_0 | 4.5 | 5.17 |
| K q8_0 / V q4_0 (common hand-tune) | 6.5 | 0.0054 |
| uniform q8_0 | 8.5 | 0.0024 |
| **planned** | **4.75** | 0.018 |
| **planned** | **5.44** | 0.0056: same quality as K8/V4, 16% smaller |
| **planned** | **6.5** | **0.0018: better than q8_0, 24% smaller** |

kvscan's Python simulation of the ggml formats predicted llama.cpp's numbers closely (q8_0: 0.0027 simulated vs
0.0024 measured; q4_0: 5.27 vs 5.17).

## 4. Background: a "psychoacoustic model" for the KV cache (and a negative result)

The project started from an audio-codec analogy: MP3 wins by knowing what the ear can't hear. Could a model's own
sensitivity tell a codec where error is inaudible?

- **Masking is real for keys.** The end-to-end Fisher information `F` of the output w.r.t. cached keys predicts
  KL ≈ ½ eᵀFe. Equal-energy noise in the top-Fisher directions costs **10–100× more KL** than noise in the bottom
  directions, in every layer except layer 0. Values mask only 1.5–3×.
- **The Fisher-weighted codec helps.** Whitening by F^½, decorrelating, then using one uniform step everywhere
  (reverse water-filling under KL) gives about **2.3× lower KL than plain transform coding** at 2–4 bits.
- **But the full Fisher adds nothing over the diagonal one.** A plain diagonal Fisher matched the full
  directional version, and diagonal Fisher weighting is already in the literature (KVQuant, RateQuant). This
  result confirms existing work rather than adding to it.
- The cheap, practical part survived: **allocate precision per tensor by measured sensitivity**, using the formats
  engines already ship. That is what kvscan does.

## 5. vLLM `int4_per_token_head`

Simulated vLLM `int4_per_token_head` (current kernel, without the key-bias correction proposed in #59674).
Qwen2.5-1.5B, 4 × 512 held-out tokens, everything quantized except the skipped layers:

| skip layers | rule | KL |
|---|---|---|
| none | uniform int4 | 5.65 (broken) |
| 0, 1, 26, 27 | first/last 2 (TurboQuant's fixed rule) | 0.058 |
| **0, 1, 7, 15** | **measured top-4 (kvscan)** | **0.019** (3.1× lower at the same cost) |
| 0, 1, 2, 6, 7, 9, 15, 18 | kvscan, target KL 0.01 | 0.012 |

The last layers (26, 27) are among the *least* sensitive on this model (layer 27 keys: 1e-4).
Not yet validated inside vLLM itself.

## Related work and upstream status

- vLLM [#59671](https://github.com/vllm-project/vllm/issues/59671) reports the Qwen2 key-bias collapse.
  [#59674](https://github.com/vllm-project/vllm/pull/59674) corrects the bias in the int4 kernel, and
  [#63081](https://github.com/vllm-project/vllm/pull/63081) warns at startup (it checks the first layer).
- vLLM TurboQuant skips the first 2 and last 2 layers by fixed rule (`get_boundary_skip_layers`).
- llama.cpp [#24139](https://github.com/ggml-org/llama.cpp/issues/24139) (KVarN) includes reports that Qwen2.5-7B
  needs its first and last layers in fp16 at 3-bit keys.
- KVQuant, RateQuant, AATC and KVTC cover sensitivity-weighted and rate-distortion KV quantization in research
  code.
