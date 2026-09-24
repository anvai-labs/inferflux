#include "webui/ui_bundle.h"

namespace inferflux {
namespace webui {

const std::string &UiHtml() {
  // Delimiter HTML: the document body contains ')' followed by '"' in inline
  // onclick handlers (e.g. saveApiKey()), which would terminate a default
  // raw-string literal early. Same reason kCss/kJs below carry delimiters.
  static const std::string kHtml = R"HTML(<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>InferFlux WebUI</title>
    <style>{{css}}</style>
  </head>
  <body>
    <div class="container">
      <header>
        <div>
          <h1>InferFlux WebUI</h1>
          <p class="subtitle">Model serving and runtime status</p>
        </div>
        <div class="api-key">
          <label for="apiKey">Access token or API key</label>
          <input id="apiKey" type="password" autocomplete="off" placeholder="Enter bearer credential" aria-describedby="authStatus" />
          <div class="btn-row"><button onclick="saveApiKey()">Use credential</button><button class="secondary" onclick="clearCredential()">Clear credential</button></div>
          <p id="authStatus" role="status">Credentials stay in this page only. Browser SSO sign-in is not available in this UI.</p>
        </div>
      </header>
      <section class="split">
        <div class="card left">
          <label for="modelSelect">Models</label>
          <select id="modelSelect"></select>
          <div class="btn-row">
            <button class="secondary" onclick="refreshModels()">Refresh models</button>
            <button class="secondary" onclick="setDefaultModel()">Set Default</button>
            <button class="secondary" onclick="unloadModel()">Unload</button>
          </div>
          <p class="subtitle">Model actions require server-granted admin scope. Configure advanced GPU placement through the server configuration or CLI.</p>
          <label for="loadModelPath">Load Model (path)</label>
          <input id="loadModelPath" placeholder="/models/llama3.gguf" />
          <label for="loadModelBackend">Backend</label>
          <input id="loadModelBackend" placeholder="cpu / cuda / rocm" />
          <button onclick="loadModel()">Load Model</button>
          <label for="prompt">Prompt</label>
          <textarea id="prompt" rows="6">Hello, InferFlux!</textarea>
          <div class="btn-row">
            <button onclick="sendChat()">Send prompt</button>
          </div>
        </div>
        <div class="card right">
          <h2>Output</h2>
          <pre id="output" aria-live="polite" aria-label="Request output"></pre>
        </div>
      </section>
      <section class="card status-card">
        <div class="status-header">
          <div>
            <h2>Status</h2>
            <p id="statusText" role="status">Loading...</p>
          </div>
          <div class="btn-row">
            <button class="secondary" onclick="refreshStatus()">Refresh status</button>
            <button class="secondary" onclick="exportHistory()">Export History</button>
            <label class="file-btn">
              Import<input type="file" id="importFile" accept=".json,application/json" onchange="importHistory(event)" />
            </label>
          </div>
        </div>
        <p id="metricsStatus" role="status">Enter a credential to load metrics.</p>
        <div class="metrics-grid">
          <div class="metric-card">
            <h3>Queue Depth</h3>
            <p id="metricQueue">--</p>
          </div>
          <div class="metric-card">
            <h3>Requests Total</h3>
            <p id="metricRequests">--</p>
          </div>
          <div class="metric-card">
            <h3>Errors Total</h3>
            <p id="metricErrors">--</p>
          </div>
        </div>
      </section>
      <section class="card">
        <div class="history-header">
          <h2>Chat History</h2>
          <button class="secondary" onclick="clearHistory()">Clear History</button>
        </div>
        <p class="subtitle">History stays in this page. Export/import uses versioned JSON; HTML history is not accepted.</p>
        <ul id="history"></ul>
      </section>
    </div>
    <script>{{js}}</script>
  </body>
</html>
)HTML";
  return kHtml;
}

const std::string &UiCss() {
  static const std::string kCss = R"CSS(
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
  padding: 24px;
  background: #0f172a;
  color: #f8fafc;
}
.container {
  max-width: 1100px;
  margin: 0 auto;
}
header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 24px;
}
.subtitle {
  color: #cbd5f5;
}
.api-key {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: min(100%, 420px);
  min-width: 0;
}
.split {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
}
.card {
  background: #0b1224;
  border-radius: 12px;
  padding: 18px;
  border: 1px solid #1e293b;
  flex: 1;
}
.left,
.right {
  min-width: min(100%, 320px);
}
.status-card .btn-row {
  justify-content: flex-end;
  flex-wrap: wrap;
}
textarea,
input,
select {
  width: 100%;
  border-radius: 8px;
  border: 1px solid #334155;
  padding: 12px;
  background: #0f172a;
  color: #f8fafc;
  font-size: 16px;
  margin-bottom: 12px;
}
.btn-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
button {
  border: none;
  border-radius: 6px;
  padding: 10px 16px;
  cursor: pointer;
  background: #38bdf8;
  color: #0f172a;
  font-weight: 600;
}
button.secondary {
  background: #1e293b;
  color: #f8fafc;
  border: 1px solid #334155;
}
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #020617;
  border-radius: 8px;
  padding: 16px;
  min-height: 220px;
  border: 1px solid #1e293b;
}
.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
ul#history {
  list-style: none;
  padding: 0;
  margin: 12px 0 0;
  max-height: 260px;
  overflow-y: auto;
}
ul#history li {
  padding: 10px 12px;
  border-bottom: 1px solid #1e293b;
}
ul#history li:last-child {
  border-bottom: none;
}
.file-btn {
  position: relative;
  overflow: hidden;
  display: inline-flex;
  align-items: center;
  padding: 10px 16px;
  border-radius: 6px;
  background: #1e293b;
  color: #f8fafc;
  cursor: pointer;
}
.file-btn input {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}
.metrics-grid {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 12px;
}
.metric-card {
  flex: 1;
  min-width: min(100%, 160px);
  background: #020617;
  border-radius: 10px;
  padding: 12px;
  border: 1px solid #1e293b;
}
.metric-card h3 {
  margin: 0 0 8px;
  font-size: 14px;
  color: #94a3b8;
}
.metric-card p {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
}
label { display: block; margin-block: 12px 6px; }
.card { min-width: 0; }
.container > .card { margin-top: 24px; }
#authStatus, #metricsStatus, .subtitle { color: #cbd5e1; line-height: 1.5; }
#authStatus { max-width: 42ch; }
#history li { white-space: pre-wrap; overflow-wrap: anywhere; }
:focus-visible { outline: 3px solid #38bdf8; outline-offset: 3px; }
@media (max-width: 600px) {
  body { padding: 16px; }
  .split { display: block; }
  .split .card + .card { margin-top: 16px; }
  .history-header, .status-header { flex-wrap: wrap; gap: 12px; }
  h1 { font-size: 1.6rem; }
}
)CSS";
  return kCss;
}

const std::string &UiJs() {
  static const std::string kJs = R"JS(
const apiKeyInput = document.getElementById("apiKey");
const historyList = document.getElementById("history");
const promptBox = document.getElementById("prompt");
const modelSelect = document.getElementById("modelSelect");
const output = document.getElementById("output");
let credential = "";
let revision = 0;
let controller = new AbortController();
let history = [];
const maxHistoryBytes = 1024 * 1024;

// Retire this UI's old persistent secrets/HTML without reading or rendering them.
try {
  for (const key of ["inferflux_api_key", "inferflux_history", "inferflux_prompt", "inferflux_model"])
    localStorage.removeItem(key);
} catch { document.getElementById("authStatus").textContent = "Browser storage could not be cleared; clear site data to remove legacy saved credentials."; }

function resetCredential() {
  revision++;
  controller.abort();
  controller = new AbortController();
  credential = "";
  apiKeyInput.value = "";
  modelSelect.replaceChildren();
  output.textContent = "";
  promptBox.value = "";
  clearHistory();
  clearMetrics();
}
function clearCredential() {
  resetCredential();
  document.getElementById("authStatus").textContent = "Credential cleared. Protected data was removed.";
  document.getElementById("metricsStatus").textContent = "Enter a credential to load metrics.";
}
function saveApiKey() {
  const token = apiKeyInput.value.trim().replace(/^Bearer\s+/i, "");
  resetCredential();
  if (!token || !/^[\x21-\x7e]+$/.test(token)) {
    document.getElementById("authStatus").textContent = "Enter a valid access token or API key.";
    return;
  }
  credential = token;
  document.getElementById("authStatus").textContent = "Credential held in memory. The server checks each request's permissions.";
  refreshModels();
  refreshStatus();
}
async function request(path, options = {}, format = "json", authenticated = true) {
  if (authenticated && !credential) throw new Error("Enter a credential first.");
  const current = revision;
  const headers = { "Content-Type": "application/json" };
  if (authenticated) headers.Authorization = "Bearer " + credential;
  const response = await fetch(path, { ...options, headers, signal: controller.signal });
  if (current !== revision) throw new DOMException("Credential changed", "AbortError");
  if (!response.ok) {
    if (response.status === 401 && authenticated) {
      clearCredential();
      document.getElementById("authStatus").textContent = "Credential rejected or expired. Enter a new credential.";
    }
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  const value = format === "text" ? await response.text() : await response.json();
  if (current !== revision) throw new DOMException("Credential changed", "AbortError");
  return value;
}
function showError(prefix, error) {
  if (error.name !== "AbortError") output.textContent = prefix + error.message;
}
async function refreshModels() {
  const selected = modelSelect.value;
  modelSelect.replaceChildren();
  try {
    const data = await request("/v1/models");
    if (!Array.isArray(data.data) || data.data.some(model => typeof model.id !== "string"))
      throw new Error("Invalid models response.");
    for (const model of data.data) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.id + (model.ready === true ? " — ready" : " — readiness not confirmed");
      modelSelect.appendChild(option);
    }
    if (data.data.some(model => model.id === selected)) modelSelect.value = selected;
  } catch (error) { showError("Models unavailable: ", error); }
}
function clearMetrics() {
  for (const id of ["metricQueue", "metricRequests", "metricErrors"])
    document.getElementById(id).textContent = "--";
}
function metricValue(text, name) {
  // These are the server's documented Prometheus series, including backend labels.
  const rows = text.split("\n").filter(line => line.startsWith(name + " ") || line.startsWith(name + "{"));
  if (!rows.length) throw new Error(`Missing metric ${name}.`);
  let total = 0;
  for (const row of rows) {
    const match = row.match(new RegExp("^" + name + '(?:\\{(?:[^"}\\\\]|\\\\.|"(?:[^"\\\\]|\\\\.)*")*\\})?\\s+([0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\\s*$'));
    if (!match || !Number.isFinite(Number(match[1])) || Number(match[1]) < 0)
      throw new Error(`Invalid metric ${name}.`);
    total += Number(match[1]);
  }
  if (!Number.isFinite(total)) throw new Error(`Invalid metric ${name}.`);
  return String(total);
}
async function refreshStatus() {
  const status = document.getElementById("statusText");
  try {
    const health = await request("/healthz", {}, "json", false);
    const ready = await request("/readyz", {}, "json", false);
    if (typeof health.status !== "string" || typeof ready.status !== "string")
      throw new Error("Invalid health response.");
    status.textContent = `Health: ${health.status} | Ready: ${ready.status}`;
  } catch (error) {
    if (error.name !== "AbortError") status.textContent = "Status unavailable: " + error.message;
  }
  const metricsStatus = document.getElementById("metricsStatus");
  if (!credential) {
    clearMetrics();
    metricsStatus.textContent = "Enter a credential to load metrics.";
    return;
  }
  try {
    const metrics = await request("/metrics", {}, "text");
    const values = ["inferflux_scheduler_queue_depth", "inferflux_requests_total", "inferflux_errors_total"]
      .map(name => metricValue(metrics, name));
    ["metricQueue", "metricRequests", "metricErrors"].forEach((id, i) => document.getElementById(id).textContent = values[i]);
    metricsStatus.textContent = "Metrics loaded.";
  } catch (error) {
    if (error.name !== "AbortError") {
      clearMetrics();
      metricsStatus.textContent = "Metrics unavailable: " + error.message;
    }
  }
}
function validateHistory(packet) {
  if (!packet || packet.version !== 1 || Object.keys(packet).sort().join() !== "entries,version"
      || !Array.isArray(packet.entries) || packet.entries.length > 200
      || packet.entries.some(entry => !entry || Object.keys(entry).sort().join() !== "role,text"
        || !["user", "assistant", "completion"].includes(entry.role)
        || typeof entry.text !== "string" || entry.text.length > 65536)
      || new TextEncoder().encode(JSON.stringify(packet)).length > maxHistoryBytes)
    throw new Error("Expected version 1 history JSON with at most 200 role/text entries (1 MiB total, 65536 characters per entry).");
  return packet.entries.map(entry => ({role: entry.role, text: entry.text}));
}
function renderHistory() {
  historyList.replaceChildren();
  for (const entry of history) {
    const item = document.createElement("li");
    item.textContent = entry.role + ": " + entry.text;
    historyList.appendChild(item);
  }
}
function clearHistory() { history = []; renderHistory(); }
async function sendChat() {
  const prompt = promptBox.value;
  const model = modelSelect.value;
  if (!model) { output.textContent = "Select an available model first."; return; }
  try {
    const data = await request("/v1/chat/completions", {method: "POST",
      body: JSON.stringify({model, messages: [{role: "user", content: prompt}], max_tokens: 128})});
    const answer = data.choices?.[0]?.message?.content;
    if (typeof answer !== "string") throw new Error("Invalid chat response: missing message content.");
    output.textContent = answer;
    // Never silently truncate or partially replace history when a bound is exceeded.
    history = validateHistory({version: 1, entries: [{role: "assistant", text: answer}, {role: "user", text: prompt}, ...history]});
    renderHistory();
  } catch (error) { showError("Request failed: ", error); }
}
function exportHistory() {
  const blob = new Blob([JSON.stringify({version: 1, entries: history}, null, 2)], {type: "application/json"});
  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.download = "inferflux-history.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function importHistory(event) {
  const file = event.target.files[0];
  if (!file) return;
  const current = revision;
  try {
    if (file.size > maxHistoryBytes) throw new Error("History file exceeds 1 MiB.");
    const text = await file.text();
    if (current !== revision) return;
    const next = validateHistory(JSON.parse(text));
    history = next;
    renderHistory();
    output.textContent = "History imported as plain text. Nothing was sent to a model.";
  } catch (error) { if (current === revision) showError("History import failed: ", error); }
  finally { event.target.value = ""; }
}
async function modelAction(path, method, payload) {
  try {
    output.textContent = await request(path, {method, body: JSON.stringify(payload)}, "text");
    await refreshModels();
  } catch (error) { showError("Model action failed: ", error); }
}
function loadModel() {
  const path = document.getElementById("loadModelPath").value.trim();
  const backend = document.getElementById("loadModelBackend").value.trim();
  if (!path) { output.textContent = "Enter a model path."; return; }
  modelAction("/v1/admin/models", "POST", {path, ...(backend ? {backend} : {})});
}
function unloadModel() {
  if (modelSelect.value) modelAction("/v1/admin/models", "DELETE", {id: modelSelect.value});
}
function setDefaultModel() {
  if (modelSelect.value) modelAction("/v1/admin/models/default", "PUT", {id: modelSelect.value});
}
window.addEventListener("DOMContentLoaded", () => refreshStatus());
)JS";
  return kJs;
}

} // namespace webui
} // namespace inferflux
