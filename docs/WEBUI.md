# Embedded WebUI

Build with `-DENABLE_WEBUI=ON` and open `/ui` on the server origin. The default
build disables this optional UI. Its public shell contains no protected data;
model, generation, metrics and administrative requests use the server's existing
[authorization contract](API_SURFACE.md). Serve remote access through trusted HTTPS.

The UI accepts a bearer access token or an explicitly provisioned API key. It does
not implement browser OIDC sign-in, role assignment or token refresh. A visible
administrative button does not grant permission: the server checks every request.
Use credentials with the minimum server-granted scopes needed. A future browser
SSO flow needs its own authentication and session acceptance before deployment.

Credentials, prompts and history remain in page memory. Reloading or leaving the
page drops them. **Clear credential** also removes displayed protected data and
invalidates in-flight responses. HTTP 401 clears the rejected credential; HTTP 403
reports insufficient access without substituting another credential. There is no
development-key fallback. This UI removes its four legacy localStorage entries on
load, without reading or rendering their contents. If browser storage is blocked,
clear site data separately to remove any old saved credentials.

## History contract

Export/import is versioned JSON, rendered exclusively as text:

```json
{"version":1,"entries":[{"role":"user","text":"Hello"},{"role":"assistant","text":"Hello"}]}
```

Only `version` and `entries` are accepted at the root; each entry has exactly
`role` and `text`. Roles are `user`, `assistant` or `completion`. Limits are 200
entries, 65,536 JavaScript string code units per text and 1 MiB for the UTF-8 JSON
payload and imported file. Invalid imports leave current history intact. HTML
history exports from older versions are intentionally rejected, never executed.
Exports contain conversation text; treat downloaded files accordingly.

## Status and model controls

Health/readiness are read from `/healthz` and `/readyz`. Metrics require a
credential and parse the server's Prometheus counters, including backend labels.
Missing, malformed or failed metrics display an explicit error and unknown values,
not zero or the previous successful counters. Refresh is manual.

The single **Send prompt** action uses `/v1/chat/completions`. Model load, unload
and default controls use the existing admin API. Advanced GPU placement remains
owned by server configuration and the CLI. Browser checks do not establish GPU
placement, actual inference, token accounting or mixed-team acceptance.

## Validation and rollout

`tests/webui/test_dashboard.py` compiles the actual renderer and embedded bundle,
then checks them in Chromium against a disposable loopback API fixture. Run:

```bash
.venv-codesign/bin/python -m pip install 'pytest>=8,<10' 'playwright==1.58.0'
.venv-codesign/bin/python -m playwright install chromium
.venv-codesign/bin/python -m pytest tests/webui -q
```

The blocking WebUI CI job also builds `inferfluxd` with the UI enabled. Existing
HTTP/auth tests retain ownership of server permissions. Headed AgentBrowser
inspection supplements automated security and responsive-layout assertions.

Verify a UI-enabled released binary separately before enabling this surface on a
GPU deployment. A GPU gate built with WebUI disabled does not validate `/ui`.
Rollback by disabling this optional UI or using a reviewed safe release; do not
restore legacy HTML imports or persistent credentials. Preserve server state and
private configuration during deployment.
