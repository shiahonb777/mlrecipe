"""Verify mlrecipe's materialized weights match PEFT's official merge_and_unload.

If both produce the same merged weights (within float32 rounding), our
recipe pipeline is correct.
"""
import os
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from safetensors import safe_open

WORK = os.path.join(os.path.dirname(__file__), "work")

print("Loading mlrecipe-materialized model...")
mlrecipe_state = {}
with safe_open(os.path.join(WORK, "merged", "model.safetensors"), framework="np") as f:
    for k in f.keys():
        mlrecipe_state[k] = f.get_tensor(k)
print(f"  {len(mlrecipe_state)} tensors")

print("Loading PEFT-merged reference model...")
base = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(base, os.path.join(WORK, "lora_src"))
peft_merged = peft_model.merge_and_unload()

print("Comparing...")
n_compared, n_matched = 0, 0
max_diff = 0.0
peft_state = {k: v for k, v in peft_merged.state_dict().items()}
# PEFT names tensors like "transformer.h.0..."; mlrecipe stores them as
# "h.0..." (the GPT2Model namespace). Strip the "transformer." prefix.
peft_norm = {}
for k, v in peft_state.items():
    norm = k
    if norm.startswith("transformer."):
        norm = norm[len("transformer."):]
    peft_norm[norm] = v

for name, a in mlrecipe_state.items():
    if name not in peft_norm:
        continue
    n_compared += 1
    b = peft_norm[name].detach().cpu().numpy()
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH on {name}: {a.shape} vs {b.shape}")
        continue
    diff = float(np.max(np.abs(a - b)))
    max_diff = max(max_diff, diff)
    if diff < 1e-4:
        n_matched += 1
    else:
        if n_compared - n_matched < 5:
            print(f"  DIFFER on {name}: max abs diff = {diff:.6e}")

print(f"\n{n_matched}/{n_compared} tensors match within 1e-4 tolerance")
print(f"max absolute difference across all tensors: {max_diff:.6e}")
sys.exit(0 if n_matched == n_compared else 1)
