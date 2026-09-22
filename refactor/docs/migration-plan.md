<!-- AI_DISCLAIMER v1.0 -->
# Migration Plan - Clean-Room Python Coding Agent

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Goal

Transform `claude-code/python-model/` from a simple Hugging Face chat scaffold into an original Python hybrid coding agent with local/open model support, workspace retrieval, policy-gated tools, memory, training, and evaluation.

## Ground Rules

- Follow ADR-0001: no copying or mechanical porting from proprietary reference material.
- Keep implementation under `claude-code/python-model/` unless explicitly approved otherwise.
- Fix existing broken Python only when it supports the approved architecture.
- Add tests with every implementation phase.
- Plan approved by the user on 2026-09-22.
- The first implementation version starts without shell execution; shell support is deferred until safety tests exist.
- The Python package should move to a neutral clean-room name instead of retaining proprietary product identity.

## Phase 0 - Baseline Validation

Tasks:

- `P0-T01`: Run a Python syntax/import baseline for `python-model`.
- `P0-T02`: Confirm current tests and document expected failures.
- `P0-T03`: Identify minimum supported Python/runtime path for local development.

Gate:

- Baseline failures are known and scoped before refactoring starts.

## Phase 1 - Package Foundation

Tasks:

- `P1-T01`: Create a clean package layout for `agent_core`, `model_adapters`, `tools`, `retrieval`, `memory`, `training`, `cli`, and `evals`.
- `P1-T02`: Replace duplicated prompt wrappers with an original prompt/config module.
- `P1-T03`: Fix the invalid `finetune.py` import and move training logic behind reusable functions.
- `P1-T04`: Add configuration loading for model path, workspace root, data dir, and policy file.

Gate:

- CLI can load config and run a one-shot generation with the existing model adapter.

## Phase 2 - Model Adapter And Streaming

Tasks:

- `P2-T01`: Define a model adapter protocol with `generate`, `stream`, and token-budget metadata.
- `P2-T02`: Implement a Transformers local adapter.
- `P2-T03`: Add optional quantization/device-map parameters without hardcoding a model vendor.
- `P2-T04`: Add deterministic tests with fake model/tokenizer doubles.

Gate:

- The agent core can run against fake adapters in tests and a local HF model in manual mode.

## Phase 3 - Policy-Gated Tools

Tasks:

- `P3-T01`: Define structured tool schemas and result objects.
- `P3-T02`: Implement safe read-only tools: file read, glob, grep, and repository tree summary.
- `P3-T03`: Implement guarded write/edit tools with explicit approval hooks.
- `P3-T04`: Implement optional shell/test command tool behind allowlist policy.
- `P3-T05`: Add audit events for every tool call.

Gate:

- Tests cover allowed, denied, malformed, and out-of-root tool calls.

## Phase 4 - Retrieval And Context Builder

Tasks:

- `P4-T01`: Build a workspace scanner that excludes secrets, virtual envs, generated folders, and binary files.
- `P4-T02`: Add chunking and lexical retrieval first.
- `P4-T03`: Add optional embeddings index with a permissive dependency path.
- `P4-T04`: Implement a context builder that respects token budgets.

Gate:

- Given a repository question, the system returns relevant file snippets without reading excluded paths.

## Phase 5 - Agent Loop And Memory

Tasks:

- `P5-T01`: Implement a bounded plan-act-observe loop with max turns and stop reasons.
- `P5-T02`: Add session memory summaries and project memory notes.
- `P5-T03`: Normalize conversation logs with redaction and training suitability flags.
- `P5-T04`: Add compaction for long sessions.

Gate:

- The agent can answer, inspect files, use tools, and preserve useful context across turns.

## Phase 6 - Training And Evaluation

Tasks:

- `P6-T01`: Repair and modularize LoRA/QLoRA training.
- `P6-T02`: Add dataset validation for schema, secrets, and PII placeholders.
- `P6-T03`: Create eval fixtures for coding tasks, retrieval relevance, tool safety, and regression behavior.
- `P6-T04`: Add a documented command to run the eval suite.

Gate:

- Fine-tuning is optional, reproducible, and gated by eval results.

## Phase 7 - Documentation And Packaging

Tasks:

- `P7-T01`: Update `python-model/README.md` for clean-room architecture and commands.
- `P7-T02`: Add example policy/config files.
- `P7-T03`: Add packaging metadata if the project should become installable.
- `P7-T04`: Produce a final implementation summary and update ADR statuses if superseded.

Gate:

- A new user can install dependencies, configure a model, index a repo, run chat, and run tests from documentation.

## Initial Implementation Order

1. Baseline syntax/tests.
2. Fix package/import health.
3. Introduce model adapter protocol and fake tests.
4. Add read-only tools and policy gate.
5. Add retrieval.
6. Add write/shell tools only after safety tests pass.
7. Add memory and training improvements.

## Open Questions Before Implementation

- Which local model should be the first supported target on this machine?
- Is GPU available, or should the first milestone assume CPU/quantized mode?
- Shell execution decision: start without shell execution in v1.
- Package naming decision: move to a neutral clean-room name.