<!-- AI_DISCLAIMER v1.0 -->
# ADR-0006: Wedding RSVP Technology Stack

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Context

The project has no existing implementation. It needs a pleasant responsive frontend, a small backend, durable RSVP storage, and a path from local development to online publication.

## Options Considered

- React with Vite and an in-memory or browser-only store.
- React with Vite, a Node.js API, and SQLite.
- A hosted backend such as Firebase or Supabase.

## Decision Outcome

Use React, Vite, and TypeScript for the frontend. Use a Node.js API with Express and SQLite for persistence. Keep the API and frontend in the same repository, with environment variables for the admin password and database location.

## Consequences

- TypeScript provides a shared, explicit data contract between UI and API.
- SQLite keeps local setup simple and is adequate for a wedding-sized guest list.
- A future deployment may move the database to a managed service or persistent volume.

## Risks

- SQLite file storage must be mounted on a persistent volume in production.
- Express and SQLite add a server process compared with a hosted form.

## WAF Pillar Alignment

- **Well-Architected:** small, understandable components with a clear persistence boundary.
- **Functional:** supports validation, duplicate checks, edits, and statistics.
- **Operational:** local setup remains lightweight and reproducible.

## Constraints

- Do not add unnecessary external services or dependencies.
- Do not expose the admin password in frontend code.
- Keep the public guest flow usable on mobile devices.

## Related Decisions

- ADR-0005 defines the product scope.
- ADR-0007 defines authentication.
- ADR-0008 defines data and UI behavior.