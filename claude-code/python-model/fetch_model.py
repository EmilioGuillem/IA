#!/usr/bin/env python3
"""Download or snapshot a Hugging Face repo to a local directory.

Usage:
  python fetch_model.py --repo_id DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF \
        --local_dir ./models/qwen-clone

If you already have the model files locally (e.g., a GGUF file), you can skip this and point other scripts to that folder.
"""
from argparse import ArgumentParser
from huggingface_hub import snapshot_download
from pathlib import Path


def main():
    p = ArgumentParser()
    p.add_argument('--repo_id', required=True, help='Hugging Face repo id or url')
    p.add_argument('--local_dir', required=True, help='Target directory to store model files')
    p.add_argument('--revision', default=None, help='Repo revision/commit/branch')
    args = p.parse_args()

    dest = Path(args.local_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f'Downloading repo {args.repo_id} to {dest} (may be large)...')
    snapshot_download(repo_id=args.repo_id, revision=args.revision, cache_dir=str(dest), local_dir=str(dest), allow_regex=True)
    print('Download/snapshot complete. Check', dest)


if __name__ == '__main__':
    main()
