<!-- AI_DISCLAIMER v1.0 -->
# Migration Plan - Wedding RSVP Application

> This file has been created (totally or partially) with the assistance of artificial intelligence tools.
> All content has been generated under the direct supervision of a named individual,
> and under the AI.Backbone Orchestrator Compliance framework

## Goal

Create a Spanish-language wedding RSVP application in `MyBoda` for 29 May 2027. Guests can submit or update their attendance status and add companions; administrators can securely manage the full guest list and view useful statistics.

## Approved Architecture

- Frontend: React, Vite, and TypeScript.
- Backend: Node.js and Express.
- Persistence: SQLite.
- Authentication: server-side shared admin password from `ADMIN_PASSWORD`.
- Scope: local-first, with a deployment-ready structure.

## Implementation Tasks

### Phase 0: Baseline and scaffold

- `P0-T01`: Confirm `MyBoda` is empty and establish the application structure.
- `P0-T02`: Add package scripts, environment configuration, and a reproducible local start command.
- `P0-T03`: Define shared RSVP types and API error conventions.

Gate: the frontend and API can start independently and the test harness can load the shared contract.

### Phase 1: Persistence and API

- `P1-T01`: Create the SQLite schema for guests and submission groups.
- `P1-T02`: Implement create, duplicate-check, list, update, and delete API operations.
- `P1-T03`: Add admin login/session protection and validation of all incoming fields.
- `P1-T04`: Add aggregate statistics by attendance status.

Gate: API tests cover valid submissions, duplicate names, invalid input, unauthorized access, edits, deletion, and statistics.

### Phase 2: Guest experience

- `P2-T01`: Build the entry choice between guest and admin views.
- `P2-T02`: Build the guest form for name, surname, status, and additional guests.
- `P2-T03`: Add pre-submit duplicate review and clear success/error states.
- `P2-T04`: Make the guest flow responsive and keyboard accessible.

Gate: a guest can complete each of the three RSVP choices and add companions without exposing admin data.

### Phase 3: Admin experience and visual design

- `P3-T01`: Build admin login and session handling in the UI.
- `P3-T02`: Build dashboard summary cards, status charts/tables, search, and filters.
- `P3-T03`: Add human-friendly editing and deletion confirmation.
- `P3-T04`: Apply restrained botanical visual styling, responsive layout, focus states, and reduced-motion support.

Gate: an authenticated admin can understand, filter, edit, and maintain the complete guest list on desktop and mobile.

### Phase 4: Verification and delivery

- `P4-T01`: Run unit and API tests.
- `P4-T02`: Run a production build and validate environment configuration.
- `P4-T03`: Perform a focused accessibility and security review.
- `P4-T04`: Document local setup and deployment requirements.

Gate: tests pass, the production build succeeds, and no secret is committed.

## Non-Goals

- Individual guest accounts or email verification.
- Public display of the guest list.
- Payments, menu selection, seating plans, or invitation management.

## Rollback

Each phase is isolated by API/UI boundary. If a phase fails, retain the previous working phase and revert only the phase branch changes; do not remove the database without an explicit migration decision.

## Approval Required

Implementation starts only after the user approves this plan.