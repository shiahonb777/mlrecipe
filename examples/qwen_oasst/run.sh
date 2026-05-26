#!/usr/bin/env bash
# Larger end-to-end demo: a real Qwen2.5-1.5B LoRA fine-tune.
#
#   base:    Qwen/Qwen2.5-1.5B                       (~3 GB)
#   adapter: kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA (~74 MB, rank 16, 7 modules)
#
# This stresses the same code path as gpt2_alpaca but with:
#   - a bigger base (real LLM scale, fp16 weights)
#   - a real safetensors LoRA (no .bin conversion)
#   - 7 target modules (q_proj, k_proj, v_proj, o_proj,
#                       gate_proj, up_proj, down_proj)
#   - fan_in_fan_out=False (standard nn.Linear, not GPT-2 Conv1D)
#
# Disk needed: ~4 GB. Time: ~3-5 min on a typical home connection.

set -euo pipefail
cd "$(dirname "$0")"

EXAMPLE_DIR="$(pwd)"
WORK="$EXAMPLE_DIR/work"
mkdir -p "$WORK"

echo "=== 1. Download the LoRA adapter ==="
python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA-adapter',
    local_dir='$WORK/lora_src',
    allow_patterns=['adapter_config.json', 'adapter_model.safetensors'],
)
print('downloaded to', p)
"

echo
echo "=== 2. mlrecipe from-peft (one command) ==="
mkdir -p work/repo && cd work/repo
mlrecipe from-peft ../lora_src --name qwen2.5-1.5b-oasst-guanaco

echo
echo "=== 3. Show ==="
mlrecipe show

echo
echo "=== 4. Size comparison ==="
RECIPE_BYTES=$(du -sb .recipe 2>/dev/null | cut -f1 || du -sk .recipe | awk '{print $1*1024}')
echo "recipe bundle (.recipe/): $RECIPE_BYTES bytes"
echo "vs merged Qwen2.5-1.5B fp16 checkpoint: ~3,000,000,000 bytes (3 GB)"
echo "compression ratio: $((3000000000 / RECIPE_BYTES))x"

echo
echo "=== 5. mlrecipe materialize (downloads ~3 GB Qwen base on first run) ==="
cd "$EXAMPLE_DIR"
mlrecipe materialize work/merged --repo work/repo

echo
echo "=== 6. Quick generation check ==="
python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
m = AutoModelForCausalLM.from_pretrained("work/merged", torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained("work/merged")
prompt = "Q: What is the capital of France?\nA:"
ids = tok(prompt, return_tensors="pt").input_ids
out = m.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
print("generated:", tok.decode(out[0], skip_special_tokens=True))
PY

echo
echo "=== Done ==="
echo "  recipe bundle:   work/repo/.recipe/  (~74 MB)"
echo "  merged model:    work/merged/        (~3 GB)"
