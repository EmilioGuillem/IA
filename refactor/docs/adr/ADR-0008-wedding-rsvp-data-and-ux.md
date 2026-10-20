<!-- AI_DISCLAIMER v1.0 -->
# ADR-0008: Wedding RSVP Data and User Experience

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Context

Guests need to register a name and attendance status, with an option to add more guests in the same flow. Before submission, the application must detect whether the person already exists. The administrator needs an understandable overview rather than raw records only.

## Options Considered

- One RSVP record per submission with a free-text guest list.
- One primary guest record with structured additional guest records.
- A household or invitation entity with separate guest memberships.

## Decision Outcome

Store one normalized guest record per person, grouped by a submission identifier when several people are added together. Each guest has first name, surname, attendance status, timestamps, and optional group membership. Duplicate detection compares normalized first name plus surname before submission and clearly asks the guest to review an existing record.

The admin view will include summary cards, status distribution, searchable and filterable guest rows, inline or modal editing, and deletion with confirmation. The visual direction will use botanical photography or subtle nature textures, warm neutral surfaces, and restrained accent colors without obscuring form readability.

## Consequences

- Statistics remain accurate because each person is a first-class record.
- A guest can add companions without repeating the entire page flow.
- Name-only duplicate detection can produce false positives for people sharing a name.

## Risks

- Duplicate review must avoid silently overwriting an existing RSVP.
- Publicly showing guest information is disallowed; only authenticated admins can view the list.

## WAF Pillar Alignment

- **Well-Architected:** normalized data and explicit validation reduce ambiguity.
- **Functional:** supports all three statuses, additional guests, duplicate checks, edits, and statistics.
- **Operational:** filtering and summary metrics make administration practical.

## Constraints

- UI labels and validation messages are in Spanish.
- The responsive layout must work on mobile and desktop.
- Avoid collecting unnecessary personal data.

## Related Decisions

- ADR-0005 defines the product scope.
- ADR-0007 defines access control.