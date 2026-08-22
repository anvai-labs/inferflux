# FTR-001: Managed Local and Explicit-Endpoint CLI

Status: Proposed
Priority: P0
Owners: CLI, Developer Experience
Dependencies: TD-001, TD-004, ADR-0002

## User Outcome

A new user gets a first response in under five minutes, while an operator can
target local or remote InferFlux explicitly and safely.

## MVP Scope

- Add `--endpoint` and `INFERFLUX_ENDPOINT` with URL/TLS-aware parsing.
- Add named contexts with endpoint and credential references; never store raw
  credentials in world-readable configuration.
- Add `inferctl run MODEL [chat options]` to configure/start a managed daemon,
  wait on readiness, send the request, and print recovery guidance on failure.
- Prefer a user-scoped Unix socket on Linux/macOS and named pipe on Windows;
  retain loopback TCP for SDK compatibility.
- Replace `failed to connect` with destination, diagnosis, and exact next commands.

## Acceptance Evidence

- Fresh-home end-to-end test reaches a response with one user command.
- Endpoint precedence and conflicting flag cases have table-driven tests.
- Non-interactive `chat` never autostarts unless explicitly requested.
- Stale PID/socket recovery, port collision, readiness timeout, and daemon crash are tested.
- Local and remote paths produce the same request/auth/policy semantics.

## Non-Goals

- Linking the inference runtime into `inferctl`.
- Replacing the public HTTP/OpenAI-compatible endpoint.
- Building a desktop UI.
