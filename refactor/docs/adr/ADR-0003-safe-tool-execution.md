<!-- AI_DISCLAIMER v1.0 -->
# ADR-0003 - Enforce policy-gated tool execution

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Status

Accepted

## Context

A coding agent becomes risky once it can read, write, run commands, or call external systems. The Python implementation needs safe defaults from the first useful version.

## Options Considered

- Let the model execute arbitrary tool calls.
- Add safety checks after implementation.
- Design every tool behind permissions, schemas, audit events, and path constraints.

## Decision Outcome

Every tool must expose a structured schema and pass through a policy gate before execution. File and command tools must enforce workspace roots, sensitive file exclusions, command allowlists, dry-run support where practical, and audit logging.

## Consequences

- Tool APIs will be more verbose but safer to test.
- The CLI must show clear approval prompts for risky actions.
- Tests must cover denied, allowed, and malformed tool calls.

## Risks

- Overly strict policies may reduce usefulness.
- Under-specified policies may create destructive behavior.

## WAF Pillar Alignment

- Security: least privilege and explicit approval gates.
- Reliability: predictable tool behavior and error handling.
- Operational Excellence: audit events for debugging and governance.

## Constraints

- No secret file reads.
- No destructive command execution without explicit user approval.
- No external network use unless configured and approved.

## Related Decisions

- ADR-0001
- ADR-0002