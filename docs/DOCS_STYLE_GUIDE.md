# InferFlux Docs Style Guide (Infographic-First)

## Purpose

Standardize OSS documentation so release docs are:
- code-aligned
- infographic-first
- contract-oriented
- short enough to scan in one pass

## Required Structure for Release-Facing Docs

1. **One visual in first screen**
- Mermaid diagram (flow/map/matrix) before long prose.

2. **Contract table early**
- Endpoint/flag/config matrix near top.

3. **Executable examples**
- Commands must map to existing binaries/flags (`inferfluxd`, `inferctl`).

4. **Status clarity**
- Mark sections explicitly: `Done`, `In Progress`, `Experimental`, `Archived`.

5. **Source alignment rule**
- If doc claims API/CLI behavior, verify against source:
  - `server/http/http_server.cpp`
  - `cli/main.cpp`
  - `CMakeLists.txt`

6. **Concise-first rule**
- Prefer tables/checklists over paragraphs.
- Avoid prose blocks longer than 5 lines.
- Keep section intros to 1-2 sentences max.

## Writing Rules

| Rule | Good | Avoid |
|---|---|---|
| Lead with structure | diagram + table + steps | long intro prose |
| Prefer concrete nouns | endpoint names, flags, files | generic marketing terms |
| Keep sections short | 5-20 lines chunks | dense 200-line narrative blocks |
| Use stable terms | `inferfluxd`, `inferctl`, `/v1/models` | renamed aliases not in code |
| Default to visual contracts | status matrix, flowchart, DoD table | essay-style milestone prose |
| Optimize for skim readers | bold labels + compact bullets | repeated explanatory paragraphs |

## Canonical Docs Set (OSS)

- `README.md`
- `docs/INDEX.md`
- `docs/Quickstart.md`
- `docs/API_SURFACE.md`
- `docs/AdminGuide.md`
- `docs/CONFIG_REFERENCE.md`
- `docs/DeveloperGuide.md`
- `docs/Architecture.md`
- `docs/ARCHIVE_INDEX.md` (catalog only; reference classification)

## Legacy / Deep-Dive Docs

Profiling and exploratory analysis docs may remain in-repo, but they should not be treated as first-entry release docs.

## Review Checklist for Doc PRs

- [ ] Contains at least one top-level diagram.
- [ ] Contains at least one contract table.
- [ ] Uses concise sections (no long narrative blocks).
- [ ] Commands were validated against current CLI usage text.
- [ ] Links point to existing files.
- [ ] Claims about API endpoints match current source.

## Planning Artifacts

Architecture decisions, feature specifications, and technical-debt records live
under `docs/adr/`, `docs/features/`, and `docs/technical-debt/`. Keep IDs unique,
declare dependencies explicitly, update the directory index and `docs/Roadmap.md`,
then run `python3 scripts/check_planning_artifacts.py`.
