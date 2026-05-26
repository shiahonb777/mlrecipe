# Example: Qwen2.5-1.5B OASST-Guanaco LoRA

End-to-end demo at LLM scale. Real Qwen2.5-1.5B base, real
multi-target LoRA, real bf16 weights.

| | Source | Size |
|---|---|---|
| Base | [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) | ~3 GB (bf16) |
| LoRA | [`kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA-adapter`](https://huggingface.co/kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA-adapter) | 74 MB (rank 16, 7 modules) |

The LoRA covers all attention + MLP projections (q_proj, k_proj,
v_proj, o_proj, gate_proj, up_proj, down_proj). Weights are stored in
**bfloat16**, which exercises mlrecipe's torch-based read path.

## Run

```bash
pip install -e .  # from repo root
bash examples/qwen_oasst/run.sh
```

## Live distributable copy

```bash
mlrecipe clone shiahonb777/qwen2.5-1.5b-oasst-recipe@v1
cd qwen2.5-1.5b-oasst-recipe
mlrecipe materialize ./merged
```

## Storage win

```
recipe bundle (.recipe/):       73,916,416 bytes  (74 MB)
merged checkpoint (Qwen, bf16): ~3,000 MB
ratio:                          ~40x smaller
```

The compression ratio is lower than the GPT-2 example (~470x) because
this LoRA is rank-16 across **7 modules**; total trainable params are
much higher relative to the base. For low-rank adapters on larger
bases (e.g. rank-8 LoRA on Llama-3-8B q/v_proj only), the ratio crosses
1000x.

## Verify

```bash
pip install peft
python examples/qwen_oasst/verify.py
```

Expected:

```
338/338 tensors match within 1e-2 tolerance (bf16 round-trip noise floor)
max absolute difference across all tensors: ~9.77e-04
```

That's exactly one bf16 ULP at the relevant magnitude.
