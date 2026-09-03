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
    <title>InferFlux WebUI</title>
    <style>{{css}}</style>
  </head>
  <body>
    <div class="container">
      <header>
        <div>
          <h1>InferFlux WebUI</h1>
          <p class="subtitle">Serving model: <strong>{{backend}}</strong></p>
        </div>
        <div class="api-key">
          <label for="apiKey">API Key</label>
          <input id="apiKey" placeholder="Bearer dev-key-123" />
          <button onclick="saveApiKey()">Save</button>
        </div>
      </header>
      <section class="split">
        <div class="card left">
          <label for="modelSelect">Models</label>
          <select id="modelSelect"></select>
          <div class="btn-row">
            <button class="secondary" onclick="refreshModels()">Refresh</button>
            <button class="secondary" onclick="setDefaultModel()">Set Default</button>
            <button class="secondary" onclick="unloadModel()">Unload</button>
          </div>
          <label for="loadModelPath">Load Model (path)</label>
          <input id="loadModelPath" placeholder="/models/llama3.gguf" />
          <label for="loadModelBackend">Backend</label>
          <input id="loadModelBackend" placeholder="cpu/cuda" />
          <button onclick="loadModel()">Load Model</button>
          <label for="prompt">Prompt</label>
          <textarea id="prompt" rows="6">Hello, InferFlux!</textarea>
          <div class="btn-row">
            <button onclick="sendCompletion()">Completion</button>
            <button onclick="sendChat()">Chat</button>
          </div>
        </div>
        <div class="card right">
          <label>Output</label>
          <pre id="output"></pre>
        </div>
      </section>
      <section class="card status-card">
        <div class="status-header">
          <div>
            <h2>Status</h2>
            <p id="statusText">Loading...</p>
          </div>
          <div class="btn-row">
            <button class="secondary" onclick="refreshStatus()">Refresh</button>
            <button class="secondary" onclick="exportHistory()">Export History</button>
            <label class="file-btn">
              Import<input type="file" id="importFile" onchange="importHistory(event)" />
            </label>
          </div>
        </div>
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
  min-width: 260px;
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
  min-width: 320px;
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
  min-width: 160px;
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
)CSS";
  return kCss;
}

const std::string &UiJs() {
  static const std::string kJs = R"JS(
const apiKeyInput = document.getElementById("apiKey");
const historyList = document.getElementById("history");
const promptBox = document.getElementById("prompt");
const modelSelect = document.getElementById("modelSelect");

function headers() {
  const token =
    localStorage.getItem("inferflux_api_key") ||
    apiKeyInput.value ||
    "Bearer dev-key-123";
  return { "Content-Type": "application/json", Authorization: token };
}
function saveApiKey() {
  localStorage.setItem("inferflux_api_key", apiKeyInput.value);
}
function saveHistory() {
  localStorage.setItem("inferflux_history", historyList.innerHTML);
}
function restoreHistory() {
  const saved = localStorage.getItem("inferflux_history");
  if (saved) historyList.innerHTML = saved;
}
function clearHistory() {
  historyList.innerHTML = "";
  saveHistory();
}
function persistPrompt() {
  localStorage.setItem("inferflux_prompt", promptBox.value);
}
async function refreshModels() {
  modelSelect.innerHTML = "";
  try {
    const res = await fetch("/v1/models", { headers: headers() });
    const data = await res.json();
    data.data.forEach((model) => {
      const opt = document.createElement("option");
      opt.value = model.id;
      opt.textContent = `${model.id} ${model.ready ? '✅' : '⏳'}`;
      modelSelect.appendChild(opt);
    });
    const saved = localStorage.getItem("inferflux_model");
    if (saved) modelSelect.value = saved;
  } catch (err) {
    document.getElementById("output").textContent =
      "Failed to load models: " + err;
  }
}
async function refreshStatus() {
  try {
    const health = await fetch("/healthz").then((r) => r.json());
    const ready = await fetch("/readyz").then((r) => r.json());
    document.getElementById("statusText").textContent =
      `Health: ${health.status} | Ready: ${ready.status}`;
  } catch (err) {
    document.getElementById("statusText").textContent =
      "Status error: " + err;
  }
  try {
    const metrics = await fetch("/metrics").then((r) => r.text());
    const queueMatch = metrics.match(/inferflux_scheduler_queue_depth\s+(\d+)/);
    const reqMatch = metrics.match(/inferflux_requests_total\s+(\d+)/);
    const errMatch = metrics.match(/inferflux_errors_total\s+(\d+)/);
    document.getElementById("metricQueue").textContent =
      queueMatch ? queueMatch[1] : "--";
    document.getElementById("metricRequests").textContent =
      reqMatch ? reqMatch[1] : "--";
    document.getElementById("metricErrors").textContent =
      errMatch ? errMatch[1] : "--";
  } catch (err) {
    document.getElementById("metricQueue").textContent = "--";
    document.getElementById("metricRequests").textContent = "--";
    document.getElementById("metricErrors").textContent = "--";
  }
}

async function sendCompletion() {
  const prompt = promptBox.value;
  const model = modelSelect.value;
  localStorage.setItem("inferflux_model", model);
  persistPrompt();
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ messages: [{role: "user", content: prompt}], model, max_tokens: 128 }),
  });
  const data = await res.json();
  const text = data.choices?.[0]?.message?.content ?? JSON.stringify(data);
  document.getElementById("output").textContent = text;
  appendHistory("completion", prompt, text);
}
async function sendChat() {
  const prompt = promptBox.value;
  const model = modelSelect.value;
  localStorage.setItem("inferflux_model", model);
  persistPrompt();
  appendHistory("user", prompt);
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 128,
    }),
  });
  const data = await res.json();
  const choice = data.choices?.[0]?.message?.content || JSON.stringify(data);
  appendHistory("assistant", choice);
  document.getElementById("output").textContent = JSON.stringify(
    data,
    null,
    2
  );
}
function appendHistory(role, text, raw = "") {
  const li = document.createElement("li");
  li.textContent =
    role === "assistant"
      ? "Assistant: " + text
      : role === "completion"
      ? "Completion:\n" + raw
      : "User: " + text;
  historyList.prepend(li);
  saveHistory();
}
window.addEventListener("DOMContentLoaded", () => {
  const savedKey = localStorage.getItem("inferflux_api_key");
  if (savedKey) apiKeyInput.value = savedKey;
  const savedPrompt = localStorage.getItem("inferflux_prompt");
  if (savedPrompt) promptBox.value = savedPrompt;
  restoreHistory();
  refreshModels();
  refreshStatus();
});
promptBox.addEventListener("change", persistPrompt);

function exportHistory() {
  const blob = new Blob([historyList.innerHTML], { type: "text/html" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "inferflux-history.html";
  link.click();
}

function importHistory(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    historyList.innerHTML = e.target.result;
    saveHistory();
  };
  reader.readAsText(file);
}

async function loadModel() {
  const path = document.getElementById("loadModelPath").value.trim();
  if (!path) return;
  const backend = document.getElementById("loadModelBackend").value.trim();
  const payload = { path };
  if (backend) payload.backend = backend;
  try {
    const res = await fetch("/v1/admin/models", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(payload),
    });
    document.getElementById("output").textContent = await res.text();
    refreshModels();
  } catch (err) {
    document.getElementById("output").textContent = err;
  }
}

async function unloadModel() {
  const model = modelSelect.value;
  if (!model) return;
  try {
    const res = await fetch("/v1/admin/models", {
      method: "DELETE",
      headers: headers(),
      body: JSON.stringify({ id: model }),
    });
    document.getElementById("output").textContent = await res.text();
    refreshModels();
  } catch (err) {
    document.getElementById("output").textContent = err;
  }
}

async function setDefaultModel() {
  const model = modelSelect.value;
  if (!model) return;
  try {
    const res = await fetch("/v1/admin/models/default", {
      method: "PUT",
      headers: headers(),
      body: JSON.stringify({ id: model }),
    });
    document.getElementById("output").textContent = await res.text();
  } catch (err) {
    document.getElementById("output").textContent = err;
  }
}
)JS";
  return kJs;
}

} // namespace webui
} // namespace inferflux
