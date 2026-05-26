# Example: real GPT-2 Alpaca LoRA

End-to-end demo that takes a real PEFT-trained LoRA from HuggingFace
Hub, packages it as an `mlrecipe` recipe, materializes it back into a
merged checkpoint, and verifies the result is bit-identical to PEFT's
official `merge_and_unload`.

**Models used:**

| | Source | Size |
|---|---|---|
| Base | [`gpt2`](https://huggingface.co/gpt2) | ~500 MB |
| LoRA | [`monsterapi/gpt2_alpaca-lora`](https://huggingface.co/monsterapi/gpt2_alpaca-lora) | ~1.2 MB |

The LoRA was trained on Alpaca instruction data and applied to GPT-2's
`c_attn` (the merged Q/K/V projection). It uses `Conv1D`-style
`(in, out)` weight layout which requires `fan_in_fan_out=True`.

## Run

```bash
# from repo root
pip install -e .
bash examples/gpt2_alpaca/run.sh
```

The script downloads the LoRA, converts `.bin` → `.safetensors`, runs
the full `init / commit / show / materialize` pipeline, then asks the
merged model to answer a real prompt.

## Verify against PEFT

```bash
pip install peft
python examples/gpt2_alpaca/verify.py
```

Expected output:

```
148/148 tensors match within 1e-4 tolerance
max absolute difference across all tensors: 0.000000e+00
```

mlrecipe's merge is **bit-identical** to PEFT's `merge_and_unload`.

## Storage win

```
recipe bundle (.recipe/):       1,191,936 bytes  (1.13 MB)
merged checkpoint (gpt2 fp32):  ~500 MB
ratio:                          ~420x smaller
```

For LLaMA-3-8B with a similar-rank LoRA, the comparable ratio is
roughly **~10,000x** (50 MB recipe vs 14 GB merged) because the base
model is much larger relative to the adapter.

## What this proves

1. mlrecipe handles real-world PEFT LoRAs without modification.
2. It correctly handles GPT-2's `Conv1D` (`fan_in_fan_out`) layout.
3. It correctly handles model-class prefix mismatches (the LoRA was
   saved with `transformer.` in keys, but the base safetensors omits
   it; mlrecipe matches by suffix).
4. The materialized weights are numerically identical to PEFT's own
   merge implementation.

## File layout produced by the demo

```
examples/gpt2_alpaca/
├── README.md
├── run.sh
├── verify.py
└── work/
    ├── lora_src/             # downloaded HF snapshot
    ├── lora.safetensors      # converted to mlrecipe's expected format
    ├── repo/.recipe/
    │   ├── recipe.toml       # the recipe (a few hundred bytes)
    │   └── artifacts/ad/...  # the LoRA, content-addressed
    └── merged/               # full GPT-2 + LoRA checkpoint, ready to load
```
