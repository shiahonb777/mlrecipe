"""Verify mlrecipe-merged Qwen2.5 matches PEFT's official merge_and_unload."""
import os
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from safetensors import safe_open

WORK = os.path.join(os.path.dirname(__file__), "work")

print("Loading mlrecipe-materialized model...")
mlrecipe_state: dict[str, np.ndarray] = {}
# Qwen safetensors may be sharded.
import glob
shard_files = sorted(glob.glob(os.path.join(WORK, "merged", "*.safetensors")))
print(f"  scanning {len(shard_files)} shard(s)")
for shard in shard_files:
    with safe_open(shard, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            # Cast bf16/fp16 to fp32 for comparison.
            mlrecipe_state[k] = t.to(torch.float32).numpy()
print(f"  {len(mlrecipe_state)} tensors")

print("Loading PEFT-merged reference model...")
# Use fp16 for PEFT reference loading to avoid OOM on machines with
# modest RAM (this is a 1.5B model; fp32 base = ~6 GB).
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float16)
peft_model = PeftModel.from_pretrained(base, os.path.join(WORK, "lora_src"))
peft_merged = peft_model.merge_and_unload()

print("Comparing...")
n_compared, n_matched = 0, 0
max_diff = 0.0
peft_state = {k: v for k, v in peft_merged.state_dict().items()}

for name, a in mlrecipe_state.items():
    if name not in peft_state:
        continue
    n_compared += 1
    b = peft_state[name].detach().to(torch.float32).cpu().numpy()
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH on {name}: {a.shape} vs {b.shape}")
        continue
    diff = float(np.max(np.abs(a - b)))
    max_diff = max(max_diff, diff)
    # Tolerance is loose because mlrecipe stores in bf16 (lossy round-trip)
    # while PEFT computes in fp32. The relevant question is "indistinguishable
    # from a normal bf16 cast", not "bit-equal to a fp32 reference".
    if diff < 1e-2:
        n_matched += 1
    else:
        if n_compared - n_matched < 5:
            print(f"  DIFFER on {name}: max abs diff = {diff:.6e}")

print(f"\n{n_matched}/{n_compared} tensors match within 1e-2 tolerance "
      f"(bf16 round-trip noise floor)")
print(f"max absolute difference across all tensors: {max_diff:.6e}")
sys.exit(0 if n_matched == n_compared else 1)
