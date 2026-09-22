<!-- AI_DISCLAIMER v1.0 -->
# ADR-0001 - Use clean-room implementation boundary

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Status

Accepted

## Context

The source material under `claude-code/src/` is not suitable for direct reuse. The goal is to build a Python agent with comparable categories of capability while avoiding copied source, prompts, names, assets, or proprietary product identity.

## Options Considered

- Direct port from TypeScript to Python.
- Prompt-only mimicry using copied system prompts.
- Clean-room rewrite from functional requirements and permissive dependencies.

## Decision Outcome

Adopt a clean-room boundary. Permitted inputs are broad capability categories, local user requirements, public documentation for open libraries, and original design. Prohibited inputs are copied implementation, copied prompts, copied assets, proprietary names in product identity, and mechanical translation.

## Consequences

- The new project should use its own naming, prompts, schemas, and UX.
- The TypeScript source can inform a feature inventory but not implementation details.
- All training data must be synthetic, user-owned, or permissively licensed.

## Risks

- Some behavior may diverge from the reference product.
- Extra review is needed when prompts or tool descriptions resemble reference text.

## WAF Pillar Alignment

- Security: reduces IP and redistribution risk.
- Cost Optimization: avoids dependence on proprietary APIs where possible.
- Operational Excellence: creates an auditable design boundary.

## Constraints

- No proprietary code reproduction.
- No proprietary prompt reproduction.
- No proprietary assets in generated deliverables.

## Related Decisions

- ADR-0000
- ADR-0004