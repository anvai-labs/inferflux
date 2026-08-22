# TD-004: Extract CLI Transport, Context, and Daemon Seams

Status: Proposed
Priority: P1
Owners: CLI
Dependencies: TD-001, ADR-0002

## Liability

`cli/main.cpp` combines argument parsing, URL construction, HTTP/SSE, rendering,
admin commands, process management, and daemon state. URLs are built as HTTP
host/port strings, blocking testable endpoint/TLS/socket evolution.

## Remediation

- Extract typed command options and destination resolution.
- Introduce a transport interface for HTTP(S), Unix socket, and named pipe adapters.
- Extract context/credential references and managed-daemon lifecycle services.
- Keep output rendering independent from transport responses.
- Add unit contracts before wiring FTR-001 behavior.

## Exit Criteria

- `chat`, `completion`, `models`, and admin commands share one destination resolver.
- Endpoint precedence is testable without network or process creation.
- HTTP scheme is not hard-coded in command handlers.
- Existing CLI output and exit-code contracts remain green.
