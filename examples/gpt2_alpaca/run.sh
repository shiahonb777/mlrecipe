#!/usr/bin/env bash
# End-to-end demo: package a real GPT-2 alpaca LoRA as a recipe,
# then materialize it back to a merged checkpoint.
#
#   base:    gpt2              (~500 MB)
#   adapter: monsterapi/gpt2_alpaca-lora
#   output:  merged GPT-2 with LoRA applied to c_attn
#
# This script must be runnable on a clean machine. It assumes:
#   - mlrecipe is installed (`pip install -e .` from repo root)
#   - huggingface_hub can reach huggingface.co
#   - ~600 MB of free disk space

set -euo pipefail

cd "$(dirname "$0")"

EXAMPLE_DIR="$(pwd)"
WORK="$EXAMPLE_DIR/work"
mkdir -p "$WORK"

echo "=== 1. Download the real LoRA adapter ==="
python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='monsterapi/gpt2_alpaca-lora',
    local_dir='$WORK/lora_src',
)
print('downloaded to', p)
"

echo
echo "=== 2. Convert .bin to safetensors (mlrecipe wants safetensors) ==="
python3 - <<'PY'
import os, torch
from safetensors.torch import save_file
src = "work/lora_src/adapter_model.bin"
dst = "work/lora.safetensors"
state = torch.load(src, map_location="cpu", weights_only=True)
# The state dict from this LoRA is keyed like
#   "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight"
# which is exactly what mlrecipe expects (PEFT convention).
save_file(state, dst)
print(f"converted {len(state)} tensors -> {dst} ({os.path.getsize(dst):,} B)")
PY

echo
echo "=== 3. mlrecipe init / commit ==="
mkdir -p work/repo && cd work/repo
mlrecipe init
mlrecipe commit \
    --name gpt2-alpaca \
    --base gpt2 \
    --adapter ../lora.safetensors \
    --target-modules c_attn \
    --rank 8 \
    --alpha 16 \
    --fan-in-fan-out \
    --seed 42 \
    --steps 1 \
    --lr 0.0003

echo
echo "=== 4. Show the recipe ==="
mlrecipe show

echo
echo "=== 5. Size comparison ==="
RECIPE_BYTES=$(du -sb .recipe 2>/dev/null | cut -f1 || du -sk .recipe | awk '{print $1*1024}')
ADAPTER_BYTES=$(stat -f%z ../lora.safetensors 2>/dev/null || stat -c%s ../lora.safetensors)
echo "recipe bundle (.recipe/): $RECIPE_BYTES bytes"
echo "(of which the LoRA adapter is $ADAPTER_BYTES bytes)"
echo "vs a hypothetical merged checkpoint of gpt2: ~500,000,000 bytes"

echo
echo "=== 6. mlrecipe materialize ==="
cd "$EXAMPLE_DIR"
mlrecipe materialize work/merged --repo work/repo

echo
echo "=== 7. Verify the merged model loads ==="
python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

merged = AutoModelForCausalLM.from_pretrained("work/merged", torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained("work/merged")
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
ids = tok(prompt, return_tensors="pt").input_ids
out = merged.generate(ids, max_new_tokens=20, do_sample=False)
print("generated:", tok.decode(out[0], skip_special_tokens=True))
PY

echo
echo "=== Done ==="
echo "  recipe bundle:   work/repo/.recipe/"
echo "  merged model:    work/merged/"
echo
echo "To distribute the recipe:"
echo "  cd work/repo && mlrecipe push <your-github-user>/gpt2-alpaca@v1"
