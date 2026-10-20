<!-- AI_DISCLAIMER v1.0 -->
# ADR-0005: Wedding RSVP Application Scope

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Context

`MyBoda` is empty and the requested product is a small wedding RSVP application for 29 May 2027. Guests need to submit their names, choose an attendance status, and optionally add other guests. An administrator needs to inspect, edit, and summarise responses.

## Options Considered

- A static page with no persistence.
- A hosted form backed by a third-party service.
- A small full-stack application with a local database and a deployable production shape.

## Decision Outcome

Build a focused full-stack RSVP application in `MyBoda`. The guest flow will support `No voy a ir`, `Voy a ir`, and `Tengo dudas`; the admin flow will provide protected listing, editing, filtering, deletion, and attendance statistics.

## Consequences

- The application can be run locally without depending on a third-party form service.
- A deployment configuration can be added later without redesigning the guest experience.
- Admin authentication and database persistence are required from the first implementation.

## Risks

- A local database needs a durable hosting strategy for public deployment.
- The simple admin authentication model is suitable for a private wedding tool, not a general-purpose identity system.

## WAF Pillar Alignment

- **Well-Architected:** clear separation between guest and admin workflows.
- **Functional:** covers RSVP collection and administration requirements.
- **Operational:** supports local execution and future deployment.

## Constraints

- The date is 29 May 2027.
- The initial UI language is Spanish.
- The design should use restrained nature-inspired visual elements.

## Related Decisions

- ADR-0006 defines the technology and persistence choices.
- ADR-0007 defines authentication and role boundaries.
- ADR-0008 defines the RSVP data model and admin experience.