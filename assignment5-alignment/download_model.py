#!/usr/bin/env python3
"""
Download a Hugging Face model once and save to a fixed directory,
so you can load it locally later.

Examples:
  python download_model.py \
    --repo-id Qwen/Qwen2.5-Math-1.5B \
    --save-dir /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --method snapshot --no-symlinks --verify

  python download_model.py \
    --repo-id Qwen/Qwen2.5-Math-1.5B \
    --save-dir /data/a5-alignment/models/Qwen2.5-Math-1.5B \
    --method transformers --trust-remote-code
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download and save a HF model to a local directory.")
    p.add_argument(
        "--repo-id",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Hugging Face repo id, e.g. 'Qwen/Qwen2.5-Math-1.5B'.",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="/data/a5-alignment/models/Qwen2.5-Math-1.5B",
        help="Target local directory to store the model files.",
    )
    p.add_argument(
        "--method",
        choices=["snapshot", "transformers"],
        default="snapshot",
        help="Download method: 'snapshot' (full repo) or 'transformers' (model+tokenizer via save_pretrained).",
    )
    p.add_argument("--revision", type=str, default=None, help="Optional git revision / tag / commit.")
    p.add_argument(
        "--local-files-only", action="store_true", help="Do not attempt to download; use local cache only."
    )
    p.add_argument(
        "--force", action="store_true", help="If set, delete existing save-dir before downloading."
    )
    p.add_argument("--symlinks", dest="symlinks", action="store_true", help="Use symlinks (snapshot method).")
    p.add_argument(
        "--no-symlinks", dest="symlinks", action="store_false", help="Copy actual files (snapshot method)."
    )
    p.set_defaults(symlinks=False)
    p.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF token. If omitted, uses env HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    p.add_argument(
        "--trust-remote-code", action="store_true", help="(transformers) Allow custom code from repo."
    )
    p.add_argument(
        "--verify", action="store_true", help="After download, try loading back from save-dir to verify."
    )
    return p.parse_args()


def get_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def ensure_clean_dir(path: Path, force: bool):
    if path.exists():
        if force:
            print(f"[info] Removing existing directory: {path}")
            shutil.rmtree(path)
        else:
            print(f"[skip] {path} already exists. Use --force to overwrite.")
            sys.exit(0)
    path.mkdir(parents=True, exist_ok=True)


def download_snapshot(
    repo_id: str,
    save_dir: Path,
    *,
    token: str | None,
    revision: str | None,
    local_files_only: bool,
    symlinks: bool,
):
    from huggingface_hub import snapshot_download

    print(f"[download] snapshot_download repo_id={repo_id} -> {save_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(save_dir),
        local_dir_use_symlinks=symlinks,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
    )
    print("[done] snapshot_download complete.")


def download_transformers(
    repo_id: str,
    save_dir: Path,
    *,
    token: str | None,
    revision: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[download] transformers.from_pretrained repo_id={repo_id}")
    tok = AutoTokenizer.from_pretrained(
        repo_id,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    print(f"[save] Saving to {save_dir}")
    tok.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    print("[done] transformers save_pretrained complete.")


def verify_local_load(save_dir: Path):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[verify] Loading back from {save_dir}")
        _tok = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
        _model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
        print("[verify] OK: model and tokenizer load locally.")
    except Exception as e:
        print("[verify] FAILED:", e)
        sys.exit(2)


def main():
    args = parse_args()
    token = get_token(args.hf_token)
    save_dir = Path(args.save_dir)

    # Prepare directory (skip if exists and not --force)
    ensure_clean_dir(save_dir, force=args.force)

    try:
        if args.method == "snapshot":
            download_snapshot(
                repo_id=args.repo_id,
                save_dir=save_dir,
                token=token,
                revision=args.revision,
                local_files_only=args.local_files_only,
                symlinks=args.symlinks,
            )
        else:
            download_transformers(
                repo_id=args.repo_id,
                save_dir=save_dir,
                token=token,
                revision=args.revision,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
            )
    except Exception as e:
        print("[error] Download failed:", e)
        # If we created the dir and failed, avoid leaving a partial tree (unless user wants to inspect)
        if save_dir.exists() and not any(save_dir.iterdir()):
            try:
                save_dir.rmdir()
            except Exception:
                pass
        sys.exit(1)

    if args.verify:
        verify_local_load(save_dir)

    print(
        f"[success] Ready to load locally with:\n"
        f"  AutoTokenizer.from_pretrained('{save_dir}')\n"
        f"  AutoModelForCausalLM.from_pretrained('{save_dir}')"
    )


if __name__ == "__main__":
    main()
