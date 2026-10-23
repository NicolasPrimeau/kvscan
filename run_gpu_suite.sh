#!/usr/bin/env bash
# Run kvscan's measured mode on a set of models with an NVIDIA GPU (Linux or WSL2).
# Usage: ./run_gpu_suite.sh            -> results/*.json and *.log, then copy results/ back.
# Needs: uv (https://docs.astral.sh/uv/), an NVIDIA driver. Optional: HF_TOKEN for gated models (Llama, Gemma).
set -euo pipefail
cd "$(dirname "$0")"
uv venv -q --allow-existing .venv
uv pip install -q --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python .venv -e .
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print(torch.cuda.get_device_name(0))"
mkdir -p results
MODELS=(${MODELS:-Qwen/Qwen2.5-7B Qwen/Qwen2.5-3B Qwen/Qwen3-4B meta-llama/Llama-3.2-3B mistralai/Mistral-7B-v0.3 google/gemma-3-4b-pt})
DTYPES=(${DTYPES:-int4_per_token_head fp8})
for m in "${MODELS[@]}"; do
  for d in "${DTYPES[@]}"; do
    out="results/$(basename "$m")_$d"
    [ -f "$out.json" ] && { echo "skip $out (done)"; continue; }
    echo "== $m $d"
    .venv/bin/kvscan "$m" --dtype "$d" --json "$out.json" 2>&1 | tee "$out.log" || echo "FAILED: $m $d"
  done
done
echo "done. Copy the results/ folder back to poseidon: scp -r results poseidon.casita:~/projects/kvscan/results/gpu"
