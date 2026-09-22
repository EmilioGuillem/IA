<!-- AI_DISCLAIMER v1.0 -->
# Clean-Room Assessment - claude-code Python Agent

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Scope

This assessment reviews `claude-code/` as local reference material for a clean-room Python implementation. The TypeScript source is treated as proprietary reference material and must not be copied, translated, or mechanically ported. The viable target is a new Python agent that recreates general capabilities using original code, permissive dependencies, and user-owned training data.

## Executive Summary

The existing `claude-code/python-model/` folder is a scaffold, not an equivalent coding agent. It can load a Hugging Face causal LM, build a simple prompt, generate a response, and append conversation logs. It does not yet implement the core agent loop required for Claude Code-like capability: tool planning, structured tool invocation, permission gates, workspace indexing, retrieval, memory management, LSP feedback, MCP integration, sandboxed command execution, or evaluation.

The strongest clean-room path is a hybrid architecture: an open model runtime plus retrieval over the workspace plus a controlled Python tool layer. Fine-tuning alone will not make the system equally capable; the missing power is mostly orchestration, context selection, tool execution, and feedback loops.

## Current Python Scaffold Findings

- `python-model/chat.py` and `python-model/offline_chat.py` implement direct text generation only.
- `python-model/claude_port/query_engine.py` wraps tokenizer/model loading and prompt generation, but has no persistent multi-turn state beyond what the caller provides.
- `python-model/conversation_logger.py` records daily JSONL conversations and can combine or normalize them for later training.
- `python-model/finetune.py` currently contains an invalid import statement and is expected to fail before training starts.
- Tests are smoke-level only and require `TEST_LOCAL_MODEL`; they do not validate tool behavior, RAG, memory, or safety constraints.

## Capability Gap

- Agent loop: missing planner/executor cycle, tool-call parsing, and bounded turn management.
- Tools: missing file read/search/edit abstractions, shell command policy, notebook/PDF handling, and web/MCP adapters.
- Context: missing repository indexing, chunking, embeddings, reranking, and token budgeting.
- Memory: missing session memory, durable project memory, summarization, and compaction.
- Safety: missing allowlists, path constraints, destructive-action protection, audit logs, and prompt-injection handling.
- Model operations: missing quantization strategy, device mapping, streaming output, LoRA/QLoRA robustness, and benchmark harness.
- Product surface: missing CLI command structure, configuration, diagnostics, and reproducible packaging.

## Recommended Target

Build a Python package with these layers:

- `core`: conversation state, model adapter interface, streaming generation, structured events.
- `tools`: safe file, search, edit, shell, test, and optional web/MCP tools behind explicit policies.
- `retrieval`: workspace scanner, embeddings index, chunk store, and query-time context builder.
- `memory`: session summaries, project notes, and conversation log normalization.
- `training`: LoRA/QLoRA fine-tuning over synthetic and user-owned data only.
- `cli`: interactive chat, one-shot run, index, train, evaluate, and config commands.
- `evals`: regression tasks for coding, file operations, RAG relevance, and safety behavior.

## Legal And Governance Constraint

Do not copy source, prompts, names, assets, or proprietary implementation details from the local `claude-code/src/` tree. Use it only to identify broad capability categories. New files must be original clean-room work.

## Baseline Decision

The baseline snapshot required by the refactor workflow was not copied because duplicating leaked/proprietary source into `refactor/as-is/codebase/` would increase redistribution risk. The existing local folder remains the read-only reference anchor.