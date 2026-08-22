# Worktree Reconciliation Record

**Status:** Reviewed; stale registrations may be pruned

**Review point:** `origin/main` at `90cda76`

| Branch / registration | Unique commits vs `origin/main` | Decision |
|---|---:|---|
| `product-critical-path-roadmap` | Current uncommitted review | Keep; this is the validated integration worktree |
| `distributed-runtime-bminus-foundation` | 0 | Keep the active root worktree; update only after the review branch is preserved |
| `native-kv-device-addressing` | 0 | Prune stale worktree metadata; no commit recovery is required |
| `scheduler-executor-metadata-refactor` | 6 | Preserve the branch; reimplement selected intent, do not cherry-pick |
| `merge-scheduler-metadata` | Same 6 plus a merge commit | Prune stale worktree metadata; it adds no implementation beyond the preserved source branch |

## Metadata Refactor Assessment

The six commits centralize unified-batch metadata, request diagnostics, HTTP
metadata ingestion, generation request assembly/execution context, and tool
helpers. Every patch conflicts when replayed onto current `origin/main`.
Mainline has since extracted `server/http/completion_payload.h` and materially
changed scheduler, executor, HTTP, and their tests; direct cherry-picking would
reintroduce duplicate `ToolCallResult` definitions and overwrite newer logic.

Preserve `scheduler-executor-metadata-refactor` as design input. Reimplement
only two still-useful seams after TD-007: move `MakeUnifiedBatchInput` into a
shared backend helper, and extract request assembly/context from the large HTTP
translation unit using current `completion_payload.h` types. Retain the current
tests and add helpers incrementally. The merge-only branch can be deleted after
the source branch is backed up or superseded.

Pruning a stale registration removes only Git administrative records for an
already-missing directory; it does not delete its branch or commits.
