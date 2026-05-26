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
echo "=== 2. mlrecipe from-peft (auto-reads PEFT config + .bin or .safetensors) ==="
mkdir -p work/repo && cd work/repo
mlrecipe from-peft ../lora_src --name gpt2-alpaca

echo
echo "=== 3. Show the recipe ==="
mlrecipe show

echo
echo "=== 4. Size comparison ==="
RECIPE_BYTES=$(du -sb .recipe 2>/dev/null | cut -f1 || du -sk .recipe | awk '{print $1*1024}')
echo "recipe bundle (.recipe/): $RECIPE_BYTES bytes"
echo "vs a hypothetical merged checkpoint of gpt2: ~500,000,000 bytes"

echo
echo "=== 5. mlrecipe materialize ==="
cd "$EXAMPLE_DIR"
mlrecipe materialize work/merged --repo work/repo

echo
echo "=== 6. Verify the merged model loads ==="
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
