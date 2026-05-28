"""`mlrecipe` — command-line entry point.

Subcommands:
  mlrecipe init                Create a .recipe/ directory in cwd.
  mlrecipe commit              Create a recipe from --base / --adapter inputs.
  mlrecipe show                Pretty-print a recipe.
  mlrecipe materialize         Apply a recipe to produce a merged checkpoint.
  mlrecipe push                Push a recipe to a GitHub Release.
  mlrecipe clone               Pull a recipe from a GitHub Release.

Design constraints:
  - No required dependencies beyond what `mlrecipe` itself needs.
  - Friendly errors. The first time someone runs the wrong command we want
    them to know exactly what to fix.
  - Subcommands are flat; we don't nest beyond one level.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mlrecipe import (
    Recipe,
    load_recipe,
    save_recipe,
)
from mlrecipe.recipe import (
    Adapter,
    BaseRef,
    TrainingMetadata,
    artifact_path,
    hash_file,
    store_artifact,
)


# ---------- helpers ----------


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _find_repo(start: Path) -> Path:
    """Walk up from `start` to find a `.recipe` directory."""
    p = start.resolve()
    while True:
        candidate = p / ".recipe"
        if candidate.is_dir():
            return candidate
        if p.parent == p:
            raise FileNotFoundError(
                "not a recipe repo (or any parent up to filesystem root). "
                "run `mlrecipe init` first."
            )
        p = p.parent


# ---------- subcommands ----------


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.path).resolve()
    repo_dir = target / ".recipe"
    if repo_dir.exists():
        _err(f"{repo_dir} already exists")
        return 1
    repo_dir.mkdir(parents=True)
    (repo_dir / "artifacts").mkdir()
    (repo_dir / "HEAD").write_text("draft\n")
    print(f"initialized empty recipe repo in {repo_dir}")
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    try:
        repo_dir = _find_repo(Path.cwd())
    except FileNotFoundError as e:
        _err(str(e))
        return 1
    if not args.base:
        _err("--base is required")
        return 2
    if not args.adapter and not args.allow_empty:
        _err("--adapter is required (or pass --allow-empty for a base-only recipe)")
        return 2

    base = BaseRef(ref=args.base, revision=args.revision)
    adapters: list[Adapter] = []

    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            _err(f"adapter file not found: {adapter_path}")
            return 1
        h = store_artifact(repo_dir, adapter_path)
        extra = {}
        if args.fan_in_fan_out:
            extra["fan_in_fan_out"] = True
        adapter = Adapter(
            type="lora",
            artifact=h,
            target_modules=args.target_modules or [],
            rank=args.rank,
            alpha=args.alpha,
            extra=extra,
        )
        adapters.append(adapter)

    training = TrainingMetadata(
        method="lora" if adapters else "none",
        seed=args.seed,
        steps=args.steps,
        learning_rate=args.lr,
        dataset_hash=args.dataset_hash,
    )

    name = args.name or "draft"
    recipe = Recipe(name=name, base=base, adapters=adapters, training=training)
    out = save_recipe(recipe, repo_dir)
    print(f"recipe saved to {out}")
    if adapters:
        print(f"  base       : {base.ref}" + (f"@{base.revision}" if base.revision else ""))
        for a in adapters:
            size = artifact_path(repo_dir, a.artifact).stat().st_size
            print(f"  adapter    : {a.artifact[:24]}... ({size:,} bytes, "
                  f"rank={a.rank}, alpha={a.alpha})")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if args.path:
        repo_dir = Path(args.path)
        if (repo_dir / ".recipe").is_dir() and not (repo_dir / "recipe.toml").is_file():
            repo_dir = repo_dir / ".recipe"
    else:
        try:
            repo_dir = _find_repo(Path.cwd())
        except FileNotFoundError as e:
            _err(str(e))
            return 1
    recipe = load_recipe(repo_dir)
    print(f"# {recipe.name}")
    print(f"format    {recipe.format_version}")
    print(f"base      {recipe.base.ref}"
          + (f"@{recipe.base.revision}" if recipe.base.revision else ""))
    if recipe.parent:
        print(f"parent    {recipe.parent}")
    if recipe.training.method != "unknown":
        t = recipe.training
        print("training:")
        print(f"  method  {t.method}")
        for k in ("seed", "steps", "learning_rate", "dataset_hash"):
            v = getattr(t, k)
            if v is not None:
                print(f"  {k:8s}{v}")
    if recipe.adapters:
        print("adapters:")
        for i, a in enumerate(recipe.adapters):
            ap = artifact_path(repo_dir, a.artifact)
            size = ap.stat().st_size if ap.exists() else None
            size_s = f"{size:,} B" if size is not None else "(missing)"
            print(f"  [{i}] {a.type} {a.artifact[:32]}...  {size_s}")
            if a.rank is not None:
                print(f"      rank={a.rank} alpha={a.alpha} targets={a.target_modules}")
    return 0


def cmd_materialize(args: argparse.Namespace) -> int:
    from mlrecipe.materialize import materialize
    if args.repo:
        repo_dir = Path(args.repo)
        # Allow `--repo work/myproject` (parent of .recipe) or `--repo work/myproject/.recipe`.
        if (repo_dir / ".recipe").is_dir() and not (repo_dir / "recipe.toml").is_file():
            repo_dir = repo_dir / ".recipe"
    else:
        try:
            repo_dir = _find_repo(Path.cwd())
        except FileNotFoundError as e:
            _err(str(e))
            return 1
    recipe = load_recipe(repo_dir)
    out = Path(args.out)
    print(f"materializing {recipe.name} -> {out}")
    materialize(recipe, out, repo_dir=repo_dir)
    print(f"done. checkpoint at {out}")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    """Push the current recipe to a GitHub Release.

    Two parallel transports for the same content:

    1. The .recipe/ tree (recipe.toml + artifacts/) is committed to the
       repo's default branch. This makes the same files reachable through
       raw.githubusercontent.com, which sends open CORS headers — needed
       for browser-side materialize.

    2. The whole .recipe/ tree is bundled into a .tar.gz and attached as
       a release asset under the requested tag. This is what
       `mlrecipe clone` downloads.

    Both transports point at the same bytes; the SHA-256 hashes inside
    `recipe.toml` keep the two views consistent.
    """
    import subprocess
    try:
        repo_dir = _find_repo(Path.cwd())
    except FileNotFoundError as e:
        _err(str(e))
        return 1
    recipe = load_recipe(repo_dir)
    target = args.target  # "user/repo" or "user/repo@tag"
    if "@" in target:
        repo, tag = target.split("@", 1)
    else:
        repo = target
        tag = recipe.name

    # Bundle.
    bundle = repo_dir.parent / f".recipe-bundle-{tag}.tar.gz"
    print(f"bundling -> {bundle}")
    import tarfile
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(repo_dir, arcname=".recipe")
    size = bundle.stat().st_size
    print(f"bundle size: {size:,} bytes")

    # Also expose just the recipe.toml as a separate, tiny asset so web
    # tools (and humans) can read the structure without downloading the
    # full bundle.
    recipe_toml = repo_dir / "recipe.toml"
    extra_asset = repo_dir.parent / f".recipe-{tag}.toml"
    if recipe_toml.is_file():
        import shutil
        shutil.copy2(recipe_toml, extra_asset)

    # Use `gh` to create / attach.
    print(f"creating release {tag} on {repo}")
    assets = [str(bundle)]
    if extra_asset.exists():
        assets.append(str(extra_asset))
    create = subprocess.run(
        ["gh", "release", "create", tag, *assets,
         "--repo", repo, "--title", tag,
         "--notes", f"recipe `{recipe.name}` (format {recipe.format_version})"],
        capture_output=True, text=True,
    )
    if create.returncode != 0:
        # Maybe the release already exists; try uploading the assets.
        if "already exists" in (create.stderr or ""):
            up = subprocess.run(
                ["gh", "release", "upload", tag, *assets,
                 "--repo", repo, "--clobber"],
                capture_output=True, text=True,
            )
            if up.returncode != 0:
                _err(f"gh release upload failed: {up.stderr.strip()}")
                return 1
        else:
            _err(f"gh release create failed: {create.stderr.strip()}")
            return 1
    print(f"pushed: https://github.com/{repo}/releases/tag/{tag}")

    # Commit the .recipe/ tree to the repo's default branch as well.
    # This makes the same content reachable from raw.githubusercontent.com,
    # which is needed for browser-side materialize (release-download URLs
    # do not send CORS headers; raw.* URLs do).
    rc = _commit_recipe_to_repo_tree(repo_dir, repo)
    if rc != 0:
        _err(
            "warning: release was created, but committing .recipe/ to the repo "
            "tree failed. The CLI ('mlrecipe clone' / 'materialize') will still work; "
            "browser-side materialize at https://shiahonb777.github.io/mlrecipe/run.html "
            "will not."
        )
        # Don't fail the whole push; the release went out fine.
    return 0


def _commit_recipe_to_repo_tree(repo_dir: Path, repo: str) -> int:
    """Mirror the local .recipe/ contents into the named GitHub repo's
    default branch via a transient clone, commit, push.

    `repo_dir` is the local .recipe/ directory.
    `repo` is "user/repo" — the GitHub repo we just pushed a release to.

    Returns 0 on success, nonzero if anything went wrong; the caller
    decides whether to treat that as fatal.
    """
    import subprocess
    import tempfile
    import shutil

    print(f"mirroring .recipe/ into {repo} default branch")
    # Look at recipe.toml + artifacts/ (the things we want to land in the
    # repo tree). Other files in repo_dir (HEAD, etc.) are local-only state.
    files_to_copy = []
    if (repo_dir / "recipe.toml").is_file():
        files_to_copy.append(("recipe.toml", repo_dir / "recipe.toml"))
    artifacts_dir = repo_dir / "artifacts"
    if artifacts_dir.is_dir():
        for f in artifacts_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(repo_dir)
                files_to_copy.append((str(rel), f))
    if not files_to_copy:
        _err("nothing to mirror (no recipe.toml or artifacts/)")
        return 1

    with tempfile.TemporaryDirectory(prefix=".mlrecipe-mirror-") as td:
        td = Path(td) / "clone"
        # Clone via SSH; the user's existing SSH auth is what `mlrecipe push`
        # already relies on through `gh`. (gh's https flow works too, but the
        # release path uses gh, and it's reasonable to assume SSH for git push.)
        clone_url = f"git@github.com:{repo}.git"
        clone = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(td)],
            capture_output=True, text=True,
        )
        if clone.returncode != 0:
            _err(f"git clone failed: {clone.stderr.strip()}")
            return 1
        # Prepare .recipe/ inside the clone.
        target = td / ".recipe"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for rel, src in files_to_copy:
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        # git add + diff check.
        subprocess.run(["git", "add", ".recipe"], cwd=td, check=True)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=td,
        )
        if diff.returncode == 0:
            print("  .recipe/ in repo tree already up to date")
            return 0
        # Commit + push. Use the project's no-reply identity if user.email
        # is unset locally, so we don't accidentally bake personal email
        # into commits.
        env_args = []
        # Check whether user.email and user.name are configured.
        u_email = subprocess.run(
            ["git", "config", "user.email"], cwd=td, capture_output=True, text=True,
        ).stdout.strip()
        u_name = subprocess.run(
            ["git", "config", "user.name"], cwd=td, capture_output=True, text=True,
        ).stdout.strip()
        if not u_email:
            env_args += ["-c", "user.email=mlrecipe@users.noreply.github.com"]
        if not u_name:
            env_args += ["-c", "user.name=mlrecipe"]
        commit = subprocess.run(
            ["git", *env_args, "commit", "-m",
             "Update .recipe/ tree (mlrecipe push)"],
            cwd=td, capture_output=True, text=True,
        )
        if commit.returncode != 0:
            _err(f"git commit failed: {commit.stderr.strip()}")
            return 1
        push = subprocess.run(
            ["git", "push"],
            cwd=td, capture_output=True, text=True,
        )
        if push.returncode != 0:
            _err(f"git push failed: {push.stderr.strip()}")
            return 1
    print("  .recipe/ mirrored to repo tree")
    return 0


def cmd_from_peft(args: argparse.Namespace) -> int:
    """Build a recipe from a PEFT adapter directory in one call.

    Reads adapter_config.json, picks rank/alpha/targets/fan_in_fan_out
    automatically, and writes the recipe + content-addressed adapter
    into the current repo (auto-creating one if needed).
    """
    from mlrecipe.peft_bridge import commit_from_peft_dir

    try:
        repo_dir = _find_repo(Path.cwd())
        project_dir = repo_dir.parent
    except FileNotFoundError:
        # No existing repo: auto-init in cwd.
        project_dir = Path.cwd()
        (project_dir / ".recipe").mkdir(exist_ok=True)
        (project_dir / ".recipe" / "artifacts").mkdir(exist_ok=True)

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_dir():
        _err(f"adapter directory not found: {adapter_dir}")
        return 1

    try:
        recipe = commit_from_peft_dir(
            adapter_dir,
            project_dir,
            base_ref=args.base,
            revision=args.revision,
            name=args.name,
        )
    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        _err(str(e))
        return 1

    print(f"recipe `{recipe.name}` committed to {project_dir / '.recipe'}/")
    print(f"  base       : {recipe.base.ref}"
          + (f"@{recipe.base.revision}" if recipe.base.revision else ""))
    a = recipe.adapters[0]
    print(f"  adapter    : {a.artifact[:24]}...  rank={a.rank} alpha={a.alpha}")
    if a.target_modules:
        print(f"  targets    : {a.target_modules}")
    if a.extra.get("fan_in_fan_out"):
        print("  layout     : fan_in_fan_out (Conv1D)")
    return 0


def cmd_clone(args: argparse.Namespace) -> int:
    """Pull a recipe from a GitHub Release into a fresh directory."""
    import subprocess
    target = args.target
    if "@" in target:
        repo, tag = target.split("@", 1)
    else:
        repo = target
        tag = "latest"
    out_dir = Path(args.out or target.split("/")[-1].split("@")[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = out_dir / "bundle.tar.gz"
    print(f"fetching {repo}@{tag} -> {bundle}")
    asset_pattern = "*.tar.gz"
    download = subprocess.run(
        ["gh", "release", "download", tag,
         "--repo", repo,
         "--pattern", asset_pattern,
         "--output", str(bundle),
         "--clobber"],
        capture_output=True, text=True,
    )
    if download.returncode != 0:
        _err(f"gh release download failed: {download.stderr.strip()}")
        return 1

    print("unpacking...")
    import tarfile
    with tarfile.open(bundle) as tf:
        tf.extractall(out_dir)
    bundle.unlink()
    recipe = load_recipe(out_dir / ".recipe")
    print(f"cloned recipe `{recipe.name}` into {out_dir}/")
    print(f"to materialize: cd {out_dir} && mlrecipe materialize ./merged")
    return 0


# ---------- main ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlrecipe",
        description="Ship model recipes, not weights.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize a recipe repo here")
    p_init.add_argument("path", nargs="?", default=".")
    p_init.set_defaults(func=cmd_init)

    p_commit = sub.add_parser("commit", help="record a recipe in this repo")
    p_commit.add_argument("--name")
    p_commit.add_argument("--base", required=True,
                          help="HF Hub ref (e.g. meta-llama/Llama-3-8B)")
    p_commit.add_argument("--revision",
                          help="optional HF commit SHA to pin the base")
    p_commit.add_argument("--adapter",
                          help="path to a LoRA adapter (.safetensors or directory)")
    p_commit.add_argument("--target-modules", nargs="*",
                          help="modules the LoRA was applied to")
    p_commit.add_argument("--rank", type=int)
    p_commit.add_argument("--alpha", type=float)
    p_commit.add_argument("--seed", type=int)
    p_commit.add_argument("--steps", type=int)
    p_commit.add_argument("--lr", type=float)
    p_commit.add_argument("--dataset-hash")
    p_commit.add_argument("--allow-empty", action="store_true",
                          help="allow a recipe with no adapter")
    p_commit.add_argument("--fan-in-fan-out", action="store_true",
                          help="set if base uses Conv1D-style (in,out) weights "
                               "(GPT-2's c_attn etc.)")
    p_commit.set_defaults(func=cmd_commit)

    p_show = sub.add_parser("show", help="display the current recipe")
    p_show.add_argument("path", nargs="?")
    p_show.set_defaults(func=cmd_show)

    p_mat = sub.add_parser("materialize",
                           help="rebuild merged weights from the recipe")
    p_mat.add_argument("out", help="output directory")
    p_mat.add_argument("--repo",
                       help="recipe directory (default: search upward for .recipe)")
    p_mat.set_defaults(func=cmd_materialize)

    p_push = sub.add_parser("push",
                            help="push the recipe to a GitHub Release")
    p_push.add_argument("target", help="user/repo or user/repo@tag")
    p_push.set_defaults(func=cmd_push)

    p_fp = sub.add_parser("from-peft",
                          help="commit a recipe from an existing PEFT adapter directory")
    p_fp.add_argument("adapter_dir",
                      help="directory with adapter_config.json + adapter_model.{safetensors,bin}")
    p_fp.add_argument("--name",
                      help="recipe name (default: adapter directory's basename)")
    p_fp.add_argument("--base",
                      help="override base_model_name_or_path from adapter_config.json")
    p_fp.add_argument("--revision",
                      help="optional HF commit SHA to pin the base")
    p_fp.set_defaults(func=cmd_from_peft)

    p_clone = sub.add_parser("clone", help="pull a recipe from a GitHub Release")
    p_clone.add_argument("target", help="user/repo or user/repo@tag")
    p_clone.add_argument("out", nargs="?", help="output directory")
    p_clone.set_defaults(func=cmd_clone)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
