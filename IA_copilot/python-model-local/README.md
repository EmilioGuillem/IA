<!-- AI_DISCLAIMER v1.0 -->
# Code Agent Local

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

This folder contains a clean-room Python coding agent with neutral naming and optional machine-learning dependencies. It imports and tests on Python 3.14 even when `transformers` and `torch` are not installed.

## Quick Start

```powershell
cd C:\Users\eguillemsimon\Documents\IA\IA\IA_copilot\python-model-local
python -m pip install -e ".[llama,test]"
python -m pytest
python -m code_agent.cli download-model
python -m code_agent.cli run --message "Hola"
python -m code_agent.cli chat --workspace C:\ruta\a\tu\proyecto
```

After `download-model`, the default backend is the local GGUF model through `llama.cpp`. Use `--backend echo` only for a dependency-free smoke test.

To validate the package without downloading a model:

```powershell
python -m pip install -r requirements.txt
python -m pytest
python -m code_agent.cli run --backend echo --message "Hola"
```

## Default Model And Hardware

The default model is `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`, file `qwen2.5-coder-7b-instruct-q4_k_m.gguf`. It is an open model suitable for coding and general conversation. Q4 quantization is a practical compromise for a machine with 16 GB RAM, but CPU generation can be slow. The runtime uses 8K context and 6 CPU threads.

The download is several gigabytes and requires network access. It is stored in `models/qwen2.5-coder-7b-instruct-q4_k_m.gguf`. The runtime uses CPU by default.

To override the default model:

```powershell
python -m code_agent.cli download-model --repo-id Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --filename qwen2.5-coder-7b-instruct-q5_k_m.gguf --local-dir .\models
python -m code_agent.cli chat --model-path .\models\qwen2.5-coder-7b-instruct-q5_k_m.gguf --workspace C:\ruta\a\tu\proyecto
```

## Agent Commands

Interactive chat supports bounded workspace tools:

```text
/help
/tree [path]
/read path [start:end]
/find text [path]
/context pregunta sobre el proyecto
/replace {"path":"src/file.py","old":"old text","new":"new text"}
/write {"path":"notes.txt","content":"contenido nuevo","overwrite":false}
/run python -m pytest
/quit
```

`/replace` and `/write` ask for confirmation before changing files. `/run` asks for confirmation and only accepts allowlisted executables (`python`, `pytest`, `ruff`, and `git`), runs without a shell, uses the workspace as its current directory, and enforces a timeout. Access is restricted to the workspace root and blocks environment files, private keys, virtual environments, caches, and other sensitive paths.

Every successful edit or command is recorded in `.code-agent-audit.jsonl` inside the workspace.

## Visual Two-Model Conversation Lab

The lab starts a local browser interface where two local model backends alternate turns. It shows the full conversation, allows human messages during the run, stops future turns, and saves each message immediately as JSONL and Markdown.

First test the interface without model weights:

```powershell
python -m code_agent.cli lab `
	--model-a .\models\placeholder.gguf `
	--model-b echo `
	--backend-a echo `
	--backend-b echo
```

For the real setup, use the downloaded GGUF as model A and a local Transformers model or Hugging Face model id as model B:

```powershell
python -m code_agent.cli lab `
	--model-a .\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf `
	--backend-a llama-cpp `
	--model-b Qwen/Qwen2.5-Coder-1.5B-Instruct `
	--backend-b hf `
	--output-dir .\conversations
```

Open `http://127.0.0.1:8765` if the browser does not open automatically. Set the number of turns and initial prompt, then choose **Iniciar**. **Parar** prevents new turns and **Enviar al diálogo** records a human intervention. Each run gets its own timestamped folder under `conversations/`.

The stop action is cooperative: it always prevents the next turn, but a backend may finish the generation already in progress before returning control. The server binds only to localhost.

## Training With Conversations And Text

The GGUF model is for inference and is not trained directly. Training uses a Transformers checkpoint and saves either a complete model or a small LoRA adapter. On this CPU, use a 1.5B base model and LoRA; training the 7B model locally will be very slow and memory-intensive.

Install training dependencies with Python 3.11 or 3.12:

```powershell
python -m pip install -e ".[hf,training,test]"
```

Prepare conversations and text files without training:

```powershell
python data_pipeline.py `
	data\sample_training_data.jsonl `
	data\conversations-2026-09-23.jsonl `
	C:\ruta\a\documentos `
	--output data\prepared_training.jsonl
```

Train a LoRA adapter from conversations and text:

```powershell
python train.py `
	--model-name-or-path Qwen/Qwen2.5-Coder-1.5B-Instruct `
	--data data\sample_training_data.jsonl C:\ruta\a\documentos `
	--output-dir models\adapter-coding `
	--use-lora `
	--num-train-epochs 1
```

The command saves `prepared_dataset.jsonl`, tokenizer metadata, checkpoints, and the LoRA adapter in `models\adapter-coding`. The source data must be user-owned or permissively licensed and must not contain secrets or personal data.

Use the saved adapter in later executions:

```powershell
python -m code_agent.cli chat `
	--backend hf `
	--model-path Qwen/Qwen2.5-Coder-1.5B-Instruct `
	--adapter-path .\models\adapter-coding `
	--workspace C:\ruta\a\tu\proyecto
```

For the default 7B GGUF runtime, use `--backend llama-cpp`; the LoRA adapter is not automatically merged into the GGUF. Merging and quantizing an adapter into a new GGUF is a separate GPU/storage-heavy step.

## Design Notes

- `code_agent` is a neutral package name.
- Model libraries are imported lazily only when their backend is selected.
- Tests do not require a downloaded model.
- Arbitrary shell syntax is not supported; commands are executed with `shell=False`.