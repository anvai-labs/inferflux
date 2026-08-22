# ADR-0002: Server-First CLI with Managed Local Transport

Status: Accepted
Date: 2026-08-21
Owners: CLI, Runtime, Developer Experience
Dependencies: TD-001

## Context

`inferctl chat` currently assumes `http://127.0.0.1:8080` and returns a generic
connection failure when no server runs. Embedding inference in the CLI would
duplicate model, KV, scheduler, policy, and observability behavior.

## Decision

Keep one `inferfluxd` runtime. Make local use feel embedded by letting
`inferctl run MODEL` manage a user-scoped daemon, wait for readiness, issue the
request, and report lifecycle state.

Resolve destinations in this order:

1. `--endpoint URL`;
2. `INFERFLUX_ENDPOINT`;
3. selected named context;
4. managed local Unix-domain socket or Windows named pipe;
5. loopback TCP compatibility endpoint.

Autostart occurs only through `run`, `--local`, `--start-if-needed`, or an
interactive confirmation. Non-interactive `chat` fails with exact recovery
commands. TCP/HTTPS remains available for OpenAI-compatible SDKs.

## Consequences

- Local transport can avoid an exposed port without creating a second runtime.
- Endpoint configuration must include scheme, authority, optional base path,
  TLS settings, and credential reference rather than only host/port.
- Daemon PID/socket/config ownership and stale-state recovery become tested contracts.

## Rejected Alternatives

- In-process inference in `inferctl`: duplicate behavior and memory ownership.
- Implicit autostart for all commands: surprising for automation and remote contexts.
- Unix socket only: incompatible with Windows and general OpenAI clients.
