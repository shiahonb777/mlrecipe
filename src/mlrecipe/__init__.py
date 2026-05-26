"""recipe — ship model recipes, not weights.

A recipe is a small text+adapter bundle that fully determines a fine-tuned
model. Instead of uploading a 14 GB merged checkpoint to HF Hub, you push
a 50 KB recipe + (optionally) the LoRA adapter. Anyone with the base model
referenced in the recipe can rebuild the merged weights bit-exactly on
their own machine.

Public API:
    Recipe         — the recipe data structure
    load_recipe    — read a recipe from disk
    save_recipe    — write a recipe to disk
    materialize    — apply the recipe (download base + adapter + merge)
"""

from __future__ import annotations

from mlrecipe.recipe import Recipe, load_recipe, save_recipe
from mlrecipe.materialize import materialize

__version__ = "0.1.0"
__all__ = ["Recipe", "load_recipe", "save_recipe", "materialize", "__version__"]


def from_peft(*args, **kwargs):
    """Convenience entry point. Dispatches to `peft_bridge.from_peft_dir`
    if the first arg is path-like, else `from_peft_model`. The heavy
    `peft_bridge` module is imported lazily so `import mlrecipe` stays
    cheap when peft isn't needed."""
    from mlrecipe import peft_bridge
    arg0 = args[0] if args else None
    import os
    if isinstance(arg0, (str, os.PathLike)):
        return peft_bridge.from_peft_dir(*args, **kwargs)
    return peft_bridge.from_peft_model(*args, **kwargs)


def commit_from_peft(*args, **kwargs):
    """Same dispatch as `from_peft` but writes the recipe + artifact into a
    repo dir in one call."""
    from mlrecipe import peft_bridge
    arg0 = args[0] if args else None
    import os
    if isinstance(arg0, (str, os.PathLike)):
        return peft_bridge.commit_from_peft_dir(*args, **kwargs)
    return peft_bridge.commit_from_peft_model(*args, **kwargs)


__all__ += ["from_peft", "commit_from_peft"]
