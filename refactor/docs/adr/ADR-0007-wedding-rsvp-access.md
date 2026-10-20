<!-- AI_DISCLAIMER v1.0 -->
# ADR-0007: Wedding RSVP Access Model

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Context

The landing experience must offer separate guest and administrator entry points. Guests should not need an account. Administrators need to read and change the complete RSVP list.

## Options Considered

- Open admin routes with an obscure URL.
- A shared admin password stored only in the server environment.
- Full user accounts with email verification and password reset.

## Decision Outcome

Use a shared admin password configured through `ADMIN_PASSWORD`. The server issues a short-lived authenticated session after successful login. Guest endpoints remain public but only accept the RSVP fields required by the form. No admin secret is sent to or stored in the browser.

## Consequences

- The setup is simple for a small private event.
- Admin routes can be protected consistently at the API boundary.
- Multiple administrators share one credential and do not have individual audit identity.

## Risks

- The password must be distributed privately and rotated if exposed.
- A public deployment should use HTTPS and secure cookie settings.

## WAF Pillar Alignment

- **Well-Architected:** authorization is enforced server-side.
- **Functional:** guests and administrators have distinct workflows.
- **Operational:** credentials are configurable per environment.

## Constraints

- No secrets in committed files.
- No admin data embedded in the public HTML.
- Validate and normalize names before duplicate checks.

## Related Decisions

- ADR-0006 defines the server stack.
- ADR-0008 defines duplicate detection and admin operations.