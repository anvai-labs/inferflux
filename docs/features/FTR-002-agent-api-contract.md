# FTR-002: Agent API and Native Structured-Output Contract

Status: Proposed
Priority: P0
Owners: API, Runtime, QA
Dependencies: TD-002, TD-003, ADR-0001

## User Outcome

Agent developers receive valid structured data and tool calls without depending
on hidden provider fallback.

## MVP Scope

- Implement a documented `/v1/responses` text-generation subset by mapping one
  canonical internal request/stream model to existing chat/completion execution.
- Own JSON-schema/grammar token filtering on the native path.
- Validate tool-call names and arguments before emitting terminal events.
- Expose capability/provider/fallback metadata consistently for Responses,
  chat, and model inventory.
- Define stable streaming events, cancellation, usage, and error contracts.

## Acceptance Evidence

- Maintained schema corpus achieves >=99% valid outputs and 100% deterministic
  rejection of unsupported schema features.
- Tool contract suite covers zero, one, parallel, malformed, and cancelled calls.
- Native-supported cases run with the compatibility delegate disabled.
- Cross-provider golden contract tests verify API equivalence, not token identity.
- Documentation states the supported Responses subset and unsupported fields.

## Non-Goals

- Stateful response storage in the first release.
- Audio, image generation, or computer-use tools.
- Pretending unsupported model tool behavior can be repaired by the server.
