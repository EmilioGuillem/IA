# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Download a Hugging Face model snapshot to a local directory."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository id")
    parser.add_argument("--local-dir", required=True, help="Target directory for model files")
    parser.add_argument("--revision", default=None, help="Optional repo revision, branch, or commit")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit("Install optional dependencies first: python -m pip install -e .[hf]") from exc

    destination = Path(args.local_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )
    print(f"Model snapshot written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())