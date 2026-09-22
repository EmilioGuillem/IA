<!-- AI_DISCLAIMER v1.0 -->
# Code Agent Local

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

This folder is a clean-room Python scaffold derived from the useful structure of `python-model`, with neutral naming and optional machine-learning dependencies. It is designed to import and test on Python 3.14 even when `transformers` and `torch` are not installed.

## Quick Start

```powershell
cd C:\Users\eguillemsimon\Documents\IA\IA\claude-code\python-model-local
python -m pip install -r requirements.txt
python -m pytest
python -m code_agent.cli run --message "Hola"
```

The default backend is `echo`, which proves the CLI and package work without ML dependencies.

## Hugging Face Backend

Use Python 3.11 or 3.12 for best compatibility with `torch` wheels. Python 3.14 support depends on upstream packages.

```powershell
python -m pip install -e ".[hf,test]"
python -m code_agent.cli run --backend hf --model-path C:\path\to\model --message "Resume este proyecto"
```

## Design Notes

- `code_agent` is a neutral package name.
- Model libraries are imported lazily only when the `hf` backend is selected.
- Tests do not require a downloaded model.
- Shell execution is intentionally not implemented in this first version.