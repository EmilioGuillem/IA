<!-- AI_DISCLAIMER v1.0 -->
# ADR-0004 - Use open model adapters and user-owned training data

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Status

Accepted

## Context

The desired system should be powerful, but the repository does not include proprietary model weights and should not depend on proprietary leaked assets. The current scaffold uses Hugging Face Transformers and optional LoRA dependencies.

## Options Considered

- Attempt to recreate proprietary model behavior from leaked source.
- Use a local open model with no training.
- Use model adapters with RAG first, then LoRA/QLoRA over safe datasets.

## Decision Outcome

Use open model adapters as the runtime abstraction. Start with a local instruction/code model compatible with Transformers or llama.cpp-style runtimes, add RAG/tooling first, and fine-tune only on synthetic, user-owned, or permissively licensed data.

## Consequences

- The project can support multiple model backends without changing agent logic.
- Training scripts should be corrected, reproducible, and evaluation-driven.
- Conversation logs need consent, redaction, and dataset quality filters before training.

## Risks

- Local model quality may trail hosted frontier models.
- Fine-tuning on low-quality logs can degrade behavior.
- GPU/RAM constraints may force smaller models or quantization.

## WAF Pillar Alignment

- Cost Optimization: local/open model path.
- Performance Efficiency: adapter abstraction supports quantized runtimes.
- Security: user-owned data and redaction before training.

## Constraints

- No proprietary weights.
- No training on copied proprietary source.
- No training on logs containing secrets or PII.

## Related Decisions

- ADR-0001
- ADR-0002