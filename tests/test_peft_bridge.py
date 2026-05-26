"""Tests for the PEFT bridge using fake adapter directories.

We don't import peft for these tests; instead we construct a minimal
`adapter_config.json` + `adapter_model.safetensors` that resemble what
PEFT would produce, and verify the bridge wires it into a Recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlrecipe import commit_from_peft, from_peft, load_recipe
from mlrecipe.peft_bridge import (
    commit_from_peft_dir,
    from_peft_dir,
)


def _write_fake_peft_adapter(d: Path, fan_in_fan_out: bool = False) -> None:
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": "fake-org/fake-base",
        "target_modules": ["q_proj", "v_proj"],
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "fan_in_fan_out": fan_in_fan_out,
    }
    (d / "adapter_config.json").write_text(json.dumps(cfg))

    from safetensors.numpy import save_file
    rng = np.random.default_rng(0)
    state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            (rng.standard_normal((8, 16)) * 0.01).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
            (rng.standard_normal((32, 8)) * 0.01).astype(np.float32),
    }
    save_file(state, str(d / "adapter_model.safetensors"))


def test_from_peft_dir_reads_config(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    recipe, blob = from_peft_dir(adapter_dir)
    assert recipe.name == "adapter"
    assert recipe.base.ref == "fake-org/fake-base"
    assert len(recipe.adapters) == 1
    a = recipe.adapters[0]
    assert a.type == "lora"
    assert a.rank == 8
    assert a.alpha == 16.0
    assert a.target_modules == ["q_proj", "v_proj"]
    assert a.extra.get("bias") == "none"
    assert a.extra.get("lora_dropout") == 0.05
    # Hash of returned blob should match the recipe's artifact ref.
    import hashlib
    assert a.artifact == "sha256:" + hashlib.sha256(blob).hexdigest()


def test_from_peft_dir_propagates_fan_in_fan_out(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir, fan_in_fan_out=True)
    recipe, _ = from_peft_dir(adapter_dir)
    assert recipe.adapters[0].extra.get("fan_in_fan_out") is True


def test_from_peft_dir_overrides_base(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    recipe, _ = from_peft_dir(adapter_dir, base_ref="other/base", revision="abc")
    assert recipe.base.ref == "other/base"
    assert recipe.base.revision == "abc"


def test_from_peft_dir_handles_bin(tmp_path: Path):
    """Old-style .bin adapters (PyTorch pickle) should also load."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": "fake/base",
        "target_modules": ["q_proj"],
        "r": 4,
        "lora_alpha": 8,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg))

    import torch
    rng = np.random.default_rng(1)
    state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            torch.from_numpy((rng.standard_normal((4, 16)) * 0.01).astype(np.float32)),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
            torch.from_numpy((rng.standard_normal((32, 4)) * 0.01).astype(np.float32)),
    }
    torch.save(state, str(adapter_dir / "adapter_model.bin"))

    recipe, _ = from_peft_dir(adapter_dir)
    assert recipe.adapters[0].rank == 4
    assert recipe.adapters[0].alpha == 8.0


def test_commit_from_peft_dir_writes_repo(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    repo_dir = tmp_path / "myproject"

    recipe = commit_from_peft_dir(adapter_dir, repo_dir, name="t1")

    # Should have created .recipe/ inside repo_dir.
    assert (repo_dir / ".recipe" / "recipe.toml").is_file()
    # Artifact should be present.
    sha = recipe.adapters[0].artifact.split(":", 1)[1]
    assert (repo_dir / ".recipe" / "artifacts" / sha[:2] / sha).is_file()

    # And it loads back identically.
    loaded = load_recipe(repo_dir / ".recipe")
    assert loaded.name == "t1"
    assert loaded.adapters[0].rank == 8


def test_top_level_from_peft_dispatches_to_dir(tmp_path: Path):
    """`mlrecipe.from_peft(some_path)` should call from_peft_dir."""
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    recipe, _ = from_peft(adapter_dir)
    assert recipe.adapters[0].rank == 8


def test_top_level_commit_from_peft_dispatches_to_dir(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    _write_fake_peft_adapter(adapter_dir)
    repo_dir = tmp_path / "p"
    commit_from_peft(adapter_dir, repo_dir, name="t2")
    assert (repo_dir / ".recipe" / "recipe.toml").is_file()


def test_missing_adapter_config_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        from_peft_dir(empty)
