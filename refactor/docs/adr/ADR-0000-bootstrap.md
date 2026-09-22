<!-- AI_DISCLAIMER v1.0 -->
# ADR-0000 - Bootstrap clean-room refactor track

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Status

Accepted

## Context

The user wants to create a powerful Python coding model/agent based on the information present in `claude-code/`. The folder includes a TypeScript codebase described by its README as leaked/proprietary npm sourcemap material, plus a small Python scaffold under `python-model/`.

## Options Considered

- Copy or port the TypeScript implementation directly.
- Use the TypeScript tree only as a high-level capability map and write original Python code.
- Continue with the current Python scaffold without architectural planning.

## Decision Outcome

Proceed with a clean-room Python refactor track. Use `refactor/as-is/README.md` as the assessment entry point and this ADR registry as the decision log. Do not copy or mechanically translate proprietary source.

## Consequences

- Implementation will be slower than direct porting but legally safer and technically cleaner.
- Existing Python files can be fixed or replaced only after the plan is approved.
- Planning must focus on agent capability, not only model fine-tuning.

## Risks

- A clean-room agent may not match proprietary product behavior exactly.
- Hardware limits may constrain local model size and latency.
- Training quality depends on user-owned datasets and evaluation loops.

## WAF Pillar Alignment

- Security: reduces license and data-leak risk.
- Reliability: creates explicit gates before changes.
- Operational Excellence: records decisions before implementation.

## Constraints

- No direct source copying from proprietary material.
- No use of proprietary model weights.
- No Phase 2 plan until ADRs are confirmed.

## Related Decisions

- ADR-0001
- ADR-0002