"""PEFT integration: build a Recipe from a PEFT model or saved adapter dir.

Two entry points:

  from_peft_model(peft_model, base_ref) -> (Recipe, bytes)
      Snapshot a live PEFT model in memory.

  from_peft_dir(adapter_dir, base_ref) -> (Recipe, bytes)
      Read a directory containing adapter_config.json + adapter_model
      (.safetensors or .bin) and produce a Recipe. This is what users
      with already-trained checkpoints will reach for.

Both return the recipe and the raw adapter bytes (safetensors). The
caller is expected to write them via `save_recipe` / `store_artifact`.

We deliberately don't import `peft` or `torch` at module load; we only
require them inside the functions that need them. That way `import
mlrecipe` is cheap.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from mlrecipe.recipe import (
    Adapter,
    BaseRef,
    Recipe,
    TrainingMetadata,
    hash_bytes,
)


def _adapter_state_to_safetensors_bytes(state: dict) -> bytes:
    """Serialize a state dict to in-memory safetensors bytes."""
    from safetensors.torch import save as safe_save  # type: ignore[import-not-found]
    return safe_save(state)


def _read_adapter_config(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_dir}; "
            "is this a PEFT adapter directory?"
        )
    with open(cfg_path) as f:
        return json.load(f)


def _load_adapter_state(adapter_dir: Path) -> dict:
    """Return a torch state dict from either adapter_model.safetensors or .bin."""
    safe = adapter_dir / "adapter_model.safetensors"
    if safe.is_file():
        from safetensors.torch import load_file  # type: ignore[import-not-found]
        return load_file(str(safe))
    bin_ = adapter_dir / "adapter_model.bin"
    if bin_.is_file():
        import torch
        return torch.load(str(bin_), map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"no adapter_model.safetensors or .bin in {adapter_dir}"
    )


def _build_recipe_from_config(
    name: str,
    cfg: dict,
    adapter_bytes: bytes,
    base_ref: Optional[str],
    revision: Optional[str],
    training: Optional[TrainingMetadata],
) -> Tuple[Recipe, bytes]:
    """Common path: turn a peft_config dict + adapter bytes into a Recipe."""
    if cfg.get("peft_type") not in ("LORA", None):
        raise NotImplementedError(
            f"only LoRA adapters are supported in this version "
            f"(got peft_type={cfg.get('peft_type')!r})"
        )

    # Resolve base_ref: explicit user value wins, else fall back to
    # what the adapter_config.json knows.
    cfg_base = cfg.get("base_model_name_or_path")
    if base_ref is None:
        base_ref = cfg_base
    if not base_ref:
        raise ValueError(
            "base_ref is required and not present in adapter_config.json; "
            "pass base_ref='org/model' explicitly"
        )

    target_modules = cfg.get("target_modules") or []
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    rank = int(cfg.get("r", 0)) or None
    alpha = cfg.get("lora_alpha")
    if alpha is not None:
        alpha = float(alpha)
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))

    adapter_hash = hash_bytes(adapter_bytes)
    extra: dict = {}
    if fan_in_fan_out:
        extra["fan_in_fan_out"] = True
    # Preserve a few PEFT-config keys for round-trip fidelity.
    for k in ("modules_to_save", "bias", "lora_dropout"):
        v = cfg.get(k)
        if v is not None and v not in (False, "", []):
            extra[k] = v

    adapter = Adapter(
        type="lora",
        artifact=adapter_hash,
        target_modules=list(target_modules),
        rank=rank,
        alpha=alpha,
        extra=extra,
    )

    base = BaseRef(ref=base_ref, revision=revision)
    if training is None:
        training = TrainingMetadata(method="lora")
    recipe = Recipe(name=name, base=base, adapters=[adapter], training=training)
    return recipe, adapter_bytes


def from_peft_dir(
    adapter_dir: os.PathLike | str,
    base_ref: Optional[str] = None,
    revision: Optional[str] = None,
    name: Optional[str] = None,
    training: Optional[TrainingMetadata] = None,
) -> Tuple[Recipe, bytes]:
    """Build a Recipe from a saved PEFT adapter directory.

    Returns (recipe, adapter_bytes). `adapter_bytes` is the LoRA payload
    in safetensors format; the caller is expected to store it as the
    artifact referenced by `recipe.adapters[0].artifact`.

    Args:
        adapter_dir: directory containing `adapter_config.json` and
            either `adapter_model.safetensors` or `adapter_model.bin`.
        base_ref: HF Hub repo ID of the base model (e.g. `"gpt2"`).
            If `None`, falls back to `base_model_name_or_path` from
            adapter_config.json.
        revision: optional HF commit SHA to pin the base.
        name: optional recipe name. Defaults to the directory's basename.
        training: optional TrainingMetadata. If `None`, a minimal
            method="lora" record is used.
    """
    adapter_dir = Path(adapter_dir)
    cfg = _read_adapter_config(adapter_dir)
    state = _load_adapter_state(adapter_dir)
    adapter_bytes = _adapter_state_to_safetensors_bytes(state)

    recipe_name = name or adapter_dir.name
    return _build_recipe_from_config(
        name=recipe_name,
        cfg=cfg,
        adapter_bytes=adapter_bytes,
        base_ref=base_ref,
        revision=revision,
        training=training,
    )


def from_peft_model(
    peft_model: Any,
    base_ref: Optional[str] = None,
    revision: Optional[str] = None,
    name: str = "draft",
    training: Optional[TrainingMetadata] = None,
) -> Tuple[Recipe, bytes]:
    """Build a Recipe from a live `PeftModel` in memory.

    Args:
        peft_model: an instance produced by `peft.get_peft_model(...)` or
            `PeftModel.from_pretrained(...)`. We read its `peft_config`
            and call `get_peft_model_state_dict` to extract the adapter.
        base_ref: HF Hub repo ID of the base model. If `None`, we try
            to read `peft_model.peft_config.base_model_name_or_path`.
        revision: optional HF commit SHA to pin the base.
        name: recipe name.
        training: optional TrainingMetadata.
    """
    try:
        from peft import get_peft_model_state_dict  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "from_peft_model requires `peft` installed. "
            "pip install peft"
        ) from e

    # peft_config is a dict-like keyed by adapter name. Pick the active one
    # if the user hasn't specified, falling back to the first.
    peft_config = getattr(peft_model, "peft_config", None)
    if peft_config is None:
        raise ValueError(
            "peft_model.peft_config is missing; is this really a PeftModel?"
        )
    if isinstance(peft_config, dict):
        active = getattr(peft_model, "active_adapter", None) or next(iter(peft_config))
        cfg_obj = peft_config[active]
    else:
        cfg_obj = peft_config

    cfg = _peft_config_to_dict(cfg_obj)

    state = get_peft_model_state_dict(peft_model)
    adapter_bytes = _adapter_state_to_safetensors_bytes(state)

    return _build_recipe_from_config(
        name=name,
        cfg=cfg,
        adapter_bytes=adapter_bytes,
        base_ref=base_ref,
        revision=revision,
        training=training,
    )


def _peft_config_to_dict(cfg_obj: Any) -> dict:
    """Coerce a PEFT config object (LoraConfig dataclass) into the same
    shape we read from adapter_config.json."""
    if isinstance(cfg_obj, dict):
        return cfg_obj
    out: dict = {}
    for attr in (
        "peft_type",
        "base_model_name_or_path",
        "target_modules",
        "r",
        "lora_alpha",
        "lora_dropout",
        "bias",
        "modules_to_save",
        "fan_in_fan_out",
    ):
        v = getattr(cfg_obj, attr, None)
        # peft_type may be an enum; render it as a string
        if v is not None and hasattr(v, "value"):
            v = v.value
        elif v is not None and hasattr(v, "name"):
            v = v.name
        out[attr] = v
    return out


def commit_from_peft_dir(
    adapter_dir: os.PathLike | str,
    repo_dir: os.PathLike | str,
    base_ref: Optional[str] = None,
    revision: Optional[str] = None,
    name: Optional[str] = None,
    training: Optional[TrainingMetadata] = None,
) -> Recipe:
    """High-level convenience: build a recipe from `adapter_dir` and write
    it (with the artifact) into `repo_dir/.recipe/`. Returns the recipe.

    Equivalent to `from_peft_dir` + manual `save_recipe` + manual
    artifact-store dance, in one call.
    """
    from mlrecipe.recipe import save_recipe, artifact_path

    adapter_dir = Path(adapter_dir)
    repo_dir = Path(repo_dir)
    if not (repo_dir / "recipe.toml").is_file() and not (repo_dir / ".recipe").is_dir():
        # Bootstrap: treat repo_dir as the parent and create .recipe/ inside it.
        target = repo_dir / ".recipe"
    elif (repo_dir / ".recipe").is_dir():
        target = repo_dir / ".recipe"
    else:
        target = repo_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / "artifacts").mkdir(exist_ok=True)

    recipe, adapter_bytes = from_peft_dir(
        adapter_dir,
        base_ref=base_ref,
        revision=revision,
        name=name,
        training=training,
    )

    # Persist the artifact under .recipe/artifacts/<sha>.
    h = recipe.adapters[0].artifact
    sha = h.split(":", 1)[1]
    out_blob = target / "artifacts" / sha[:2] / sha
    out_blob.parent.mkdir(parents=True, exist_ok=True)
    if not out_blob.exists():
        out_blob.write_bytes(adapter_bytes)

    save_recipe(recipe, target)
    return recipe


def commit_from_peft_model(
    peft_model: Any,
    repo_dir: os.PathLike | str,
    base_ref: Optional[str] = None,
    revision: Optional[str] = None,
    name: str = "draft",
    training: Optional[TrainingMetadata] = None,
) -> Recipe:
    """Same as `commit_from_peft_dir` but for a live PeftModel."""
    from mlrecipe.recipe import save_recipe

    repo_dir = Path(repo_dir)
    if (repo_dir / ".recipe").is_dir():
        target = repo_dir / ".recipe"
    else:
        target = repo_dir / ".recipe"
    target.mkdir(parents=True, exist_ok=True)
    (target / "artifacts").mkdir(exist_ok=True)

    recipe, adapter_bytes = from_peft_model(
        peft_model,
        base_ref=base_ref,
        revision=revision,
        name=name,
        training=training,
    )

    h = recipe.adapters[0].artifact
    sha = h.split(":", 1)[1]
    out_blob = target / "artifacts" / sha[:2] / sha
    out_blob.parent.mkdir(parents=True, exist_ok=True)
    if not out_blob.exists():
        out_blob.write_bytes(adapter_bytes)

    save_recipe(recipe, target)
    return recipe
