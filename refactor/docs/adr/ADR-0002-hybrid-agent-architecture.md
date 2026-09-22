<!-- AI_DISCLAIMER v1.0 -->
# ADR-0002 - Build a hybrid model, RAG, tools, and memory architecture

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Status

Accepted

## Context

The current Python scaffold performs direct text generation. A powerful coding assistant requires more than fine-tuning: it needs context retrieval, tool execution, persistent state, safety controls, and evaluation.

## Options Considered

- Fine-tune only.
- RAG-only assistant over repository files.
- Hybrid agent with model adapter, RAG, tools, memory, and evals.

## Decision Outcome

Build a hybrid Python agent. The model produces reasoning and structured intents; retrieval supplies repository context; tools perform controlled file/search/test actions; memory preserves session/project facts; evaluation guards regressions.

## Consequences

- Package boundaries should separate `core`, `tools`, `retrieval`, `memory`, `training`, `cli`, and `evals`.
- The initial milestone should make the agent useful before any fine-tuning.
- Fine-tuning becomes an optimization step, not the foundation.

## Risks

- More moving parts than a simple chatbot.
- Tool safety and context ranking become critical quality factors.
- Weak local hardware may require quantized or remote-compatible adapters.

## WAF Pillar Alignment

- Reliability: explicit tool loop and eval harness.
- Performance Efficiency: retrieval reduces prompt bloat.
- Security: tools can be constrained centrally.
- Cost Optimization: supports local/open models.

## Constraints

- Tool execution must be policy-gated.
- Retrieval must respect sensitive file exclusions.
- The agent must work with local models first, with optional adapters later.

## Related Decisions

- ADR-0003
- ADR-0004