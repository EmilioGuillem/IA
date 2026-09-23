# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Download a Hugging Face model snapshot to a local directory."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from code_agent.model_backend import DEFAULT_MODEL_FILENAME, DEFAULT_MODEL_ID, DEFAULT_MODEL_PATH


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_MODEL_ID, help="Hugging Face repository id")
    parser.add_argument("--local-dir", default=str(DEFAULT_MODEL_PATH.parent), help="Target directory for model files")
    parser.add_argument("--filename", default=DEFAULT_MODEL_FILENAME, help="GGUF filename to download")
    parser.add_argument("--revision", default=None, help="Optional repo revision, branch, or commit")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit("Install model dependencies first: python -m pip install -e .[llama]") from exc

    destination = Path(args.local_dir)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        local_dir=str(destination),
    )
    print(f"Model written to {downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())