# UI Launcher and Embedded WebUI

**Snapshot date:** March 9, 2026  
**Status:** partial implementation, not yet a full desktop product

## 1) Current Shape

```mermaid
flowchart LR
    A[ENABLE_WEBUI=ON] --> B[/ui served by inferfluxd]
    C[inferctl serve --ui] --> D[passes compatibility flag]
    B --> E[litehtml-based renderer + embedded assets]
```

## 2) Current Code Reality

| Area | Current state |
|---|---|
| Build flag | `ENABLE_WEBUI` exists in CMake |
| Server route | `/ui` is served when the feature is compiled in |
| Renderer | embedded HTML/CSS/JS are rendered through the litehtml-based UI renderer |
| `--ui` flag | accepted for compatibility, but currently a runtime no-op rather than a native desktop window |
| Quickstart | `inferctl quickstart` scaffolds config and backend choice; `inferctl serve --ui` forwards the UI flag |

## 3) What Is Not Shipped Yet

| Not shipped | Why it matters |
|---|---|
| Native desktop launcher window | Current implementation is browser-served UI, not a packaged desktop shell |
| Full settings/metrics/operator console | `/ui` exists, but this is not yet a complete local-product experience |
| Installer-grade packaging story | Still separate from the core runtime contract work |

## 4) Design Rules

1. `/ui` should remain a thin layer over the existing authenticated API surface.
2. UI work must not fork the control plane or introduce hidden server-only semantics.
3. Desktop packaging is secondary to keeping the API/runtime contracts stable.

## 5) Next Gates

| Priority | Gate |
|---|---|
| P1 | Expand `/ui` only through the existing API/admin contracts |
| P2 | Decide whether a native launcher window is worth the maintenance cost |
| P2 | Revisit installer and local-product polish after runtime foundations are stronger |

## 6) Related Docs

- [../API_SURFACE](../API_SURFACE.md)
- [../Quickstart](../Quickstart.md)
- [../Installer](../Installer.md)
