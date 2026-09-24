"""Browser contracts for the actual compiled WebUiRenderer; no GPU or real credentials.

The loopback fixture implements only the documented responses needed by the UI.
Server authorization remains owned by the existing HTTP/auth tests.
"""
import json
import os
from pathlib import Path
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "disposable-browser-fixture"
PAYLOAD = '<img src="/missing" onerror="window.historyExecuted=true">'


@pytest.fixture(scope="session")
def rendered(tmp_path_factory):
    folder = tmp_path_factory.mktemp("webui-renderer")
    source = folder / "render.cpp"
    source.write_text('#include <iostream>\n#include "webui/ui_renderer.h"\n'
                      'int main(){std::cout << inferflux::WebUiRenderer().RenderIndex("");}\n')
    binary = folder / "render"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-I", str(ROOT),
                    "-I", str(ROOT / "third_party/litehtml_stub"), str(source),
                    str(ROOT / "webui/ui_renderer.cpp"), str(ROOT / "webui/ui_bundle.cpp"),
                    "-o", str(binary)], check=True)
    return subprocess.check_output([str(binary)])


@pytest.fixture
def gateway(rendered):
    state = SimpleNamespace(requests=[], metrics_status=200)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            status, kind = 200, "application/json"
            if self.path == "/ui":
                body, kind = rendered, "text/html"
            elif self.path in ("/healthz", "/readyz"):
                body = b'{"status":"ok"}'
            else:
                state.requests.append((self.path, self.headers.get("Authorization")))
                if self.headers.get("Authorization") != "Bearer " + TOKEN:
                    status, body = 401, b'{"error":{"message":"unauthorized"}}'
                elif self.path == "/v1/models":
                    body = b'{"data":[{"id":"fixture-model","ready":true}]}'
                elif self.path == "/metrics":
                    status, kind = state.metrics_status, "text/plain"
                    body = (b'# TYPE inferflux_requests_total counter\n'
                            b'inferflux_requests_total{backend="cuda"} 18\n'
                            b'inferflux_errors_total{backend="cuda"} 2\n'
                            b'inferflux_scheduler_queue_depth 3\n')
                else:
                    status, body = 404, b'{"error":{"message":"not found"}}'
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/ui"
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    yield page
    context.close()
    assert not errors, errors


def connect(page, gateway):
    page.goto(gateway.url)
    page.locator("#apiKey").fill(TOKEN)
    page.get_by_role("button", name="Use credential", exact=True).click()
    expect(page.locator("#modelSelect")).to_have_value("fixture-model")


def test_legacy_html_and_credentials_never_restore(page, gateway):
    page.add_init_script("localStorage.setItem('inferflux_history', " + json.dumps(PAYLOAD) + ");"
                         "localStorage.setItem('inferflux_api_key', 'legacy-secret');"
                         "localStorage.setItem('inferflux_prompt', 'private prompt');")
    page.goto(gateway.url)
    expect(page.locator("#history img")).to_have_count(0)
    assert page.evaluate("window.historyExecuted !== true")
    assert page.evaluate("localStorage.getItem('inferflux_api_key')") is None
    assert page.evaluate("localStorage.getItem('inferflux_history')") is None
    expect(page.locator("#apiKey")).to_have_value("")
    assert gateway.requests == [], "no default/development credential may be sent"


def test_history_json_is_inert_and_roundtrips(page, gateway, tmp_path):
    page.goto(gateway.url)
    packet = {"version": 1, "entries": [{"role": "user", "text": PAYLOAD}]}
    page.locator("#importFile").set_input_files({"name": "history.json", "mimeType": "application/json",
                                               "buffer": json.dumps(packet).encode()})
    expect(page.locator("#history")).to_contain_text(PAYLOAD)
    expect(page.locator("#history img")).to_have_count(0)
    assert page.evaluate("window.historyExecuted !== true")
    with page.expect_download() as downloaded:
        page.get_by_role("button", name="Export History").click()
    output = tmp_path / "history.json"
    downloaded.value.save_as(output)
    assert json.loads(output.read_text()) == packet
    page.reload()
    expect(page.locator("#history li")).to_have_count(0)


@pytest.mark.parametrize("content", [PAYLOAD, '{"version":2,"entries":[]}',
                                      '{"version":1,"entries":[{"role":"admin","text":"x"}]}'])
def test_invalid_history_is_rejected_without_replacing_current(page, gateway, content):
    page.goto(gateway.url)
    packet = {"version": 1, "entries": [{"role": "user", "text": "keep me"}]}
    field = page.locator("#importFile")
    field.set_input_files({"name": "valid.json", "mimeType": "application/json", "buffer": json.dumps(packet).encode()})
    expect(page.locator("#history")).to_contain_text("keep me")
    field.set_input_files({"name": "invalid.json", "mimeType": "application/json", "buffer": content.encode()})
    expect(page.locator("#output")).to_contain_text("History import failed")
    expect(page.locator("#history")).to_contain_text("keep me")
    expect(page.locator("#history img")).to_have_count(0)


def test_authenticated_metrics_and_clear_prevent_stale_data(page, gateway):
    connect(page, gateway)
    expect(page.locator("#metricRequests")).to_have_text("18")
    expect(page.locator("#metricErrors")).to_have_text("2")
    expect(page.locator("#metricQueue")).to_have_text("3")
    assert all(auth == "Bearer " + TOKEN for _, auth in gateway.requests)
    assert page.evaluate("localStorage.length + sessionStorage.length") == 0
    gateway.metrics_status = 503
    page.get_by_role("button", name="Refresh status", exact=True).click()
    expect(page.locator("#metricRequests")).to_have_text("--")
    expect(page.locator("#metricsStatus")).to_contain_text("503")
    page.get_by_role("button", name="Clear credential", exact=True).click()
    expect(page.locator("#modelSelect option")).to_have_count(0)
    expect(page.locator("#apiKey")).to_have_value("")
    before = len(gateway.requests)
    page.get_by_role("button", name="Refresh status", exact=True).click()
    expect(page.locator("#metricsStatus")).to_contain_text("credential")
    assert len(gateway.requests) == before


@pytest.mark.parametrize("width", [390, 1440])
def test_accessible_responsive_shell(page, gateway, width):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(gateway.url)
    expect(page.locator("#apiKey")).to_have_attribute("type", "password")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    expect(page.locator("#statusText")).to_have_attribute("role", "status")
    expect(page.locator("#output")).to_have_attribute("aria-live", "polite")


def test_clear_discards_an_inflight_model_response(page, gateway):
    connect(page, gateway)
    page.evaluate("""() => {
      const originalFetch = window.fetch;
      window.fetch = async (...args) => {
        const response = await originalFetch(...args);
        if (args[0] === '/v1/models') {
          await new Promise(resolve => { window.releaseModels = resolve; });
        }
        return response;
      };
    }""")
    page.get_by_role("button", name="Refresh models", exact=True).click()
    page.wait_for_function("typeof window.releaseModels === 'function'")
    page.get_by_role("button", name="Clear credential", exact=True).click()
    page.evaluate("async () => { window.releaseModels(); await new Promise(r => setTimeout(r, 0)); }")
    expect(page.locator("#modelSelect option")).to_have_count(0)
    expect(page.locator("#output")).to_be_empty()
    expect(page.locator("#authStatus")).to_contain_text("Credential cleared")
