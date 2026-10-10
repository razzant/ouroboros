"""Browser operations use service facts; JavaScript words do not grant or veto."""
from __future__ import annotations

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from ouroboros import browser_policy, config
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.server_process import clear_service_binding, record_service_binding
from ouroboros.tools import browser
from ouroboros.tools.registry import ToolContext


@pytest.fixture
def control_page(tmp_path, monkeypatch):
    """A dedicated test service publishes its real process/socket identity."""
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b"<html><body><h1>Agency browser ready</h1><p id='result'>Waiting</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            hits.append((self.path, json.loads(body)))
            response = json.dumps({"received": len(hits)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    binding = record_service_binding(tmp_path, "main", "127.0.0.1", port, pid=os.getpid())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        clear_service_binding(tmp_path, "main", binding)
        assert not thread.is_alive()


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
def test_http_policy_uses_actual_operation_and_identity(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    record_service_binding(tmp_path, "main", "127.0.0.1", 49159, pid=os.getpid())
    request = SimpleNamespace(url="http://127.0.0.1:49159/api/settings", method="POST",
                              post_data='{"OUROBOROS_REVIEW_ENFORCEMENT":"advisory"}')
    decision = browser_policy.browser_request_block_reason
    assert bool(decision(request, None, restricted=False, runtime_mode=mode)) == (mode != "cyber_pro")
    request.method = "GET"
    assert decision(request, None, restricted=False, runtime_mode=mode) == ""
    request.method, request.url = "POST", "http://127.0.0.1:49160/api/settings"
    assert decision(request, None, restricted=False, runtime_mode=mode) == ""


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data", "http://[fe80::1]/", "http://2852039166/",
    "http://10.0.0.1/",
])
def test_cyber_targets_do_not_require_internal_permission(url):
    assert browser_policy.browser_url_block_reason(url, restricted=False, runtime_mode="cyber_pro") == ""
    assert browser_policy.browser_url_block_reason(url, restricted=True, runtime_mode="cyber_pro")


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
def test_local_file_reads_follow_parent_reach_without_a_workspace_fence(mode):
    for restricted in (False, True):
        assert browser_policy.browser_url_block_reason(
            "file:///outside/workspace/report.html", restricted=restricted, runtime_mode=mode) == ""


def _evaluated(ctx, value):
    """The evaluated value. A successful evaluate carries its generation's browser
    conditions as a host note AFTER the clean data (#1606), never instead of it."""
    text = browser._browser_action(ctx, "evaluate", value=value)
    data, note, conditions = text.rpartition("\n\nBrowser conditions: ")
    assert note and "Browser conditions: " + conditions == ctx.browser_state.browser_conditions, text
    return data


@pytest.mark.browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_live_evaluate_reads_policy_words_and_cyber_posts_once(control_page, tmp_path, monkeypatch, engine):
    url, hits = control_page
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "pro")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    try:
        assert browser_policy.runtime_service_kind(url, ctx) == "main"
        assert "Agency browser ready" in browser._browse_page(ctx, url, engine=engine)
        # These are literal strings, not HTTP operations or changes to settings.
        literal = "/api/owner/context-mode low /api/owner/safety-mode /api/owner/skills/x/attest-review "
        literal += "settings.json OUROBOROS_REVIEW_ENFORCEMENT OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS "
        literal += "OUROBOROS_POST_TASK_EVOLUTION OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE"
        assert _evaluated(ctx, json.dumps(literal)) == literal
        expression = "() => fetch('/api/settings', {method:'POST', body:JSON.stringify({"
        expression += "OUROBOROS_REVIEW_ENFORCEMENT:'advisory'})}).then(r => r.json()).then(r => {"
        expression += "document.querySelector('#result').textContent='Completed '+r.received; return r.received;})"
        assert "BROWSER_OWNER_CONTROL_BLOCKED" in browser._browser_action(ctx, "evaluate", value=expression)
        assert hits == []
        # The next operation reuses the browser and reads the effective mode.
        monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
        assert _evaluated(ctx, expression) == "1"
        assert hits == [("/api/settings", {"OUROBOROS_REVIEW_ENFORCEMENT": "advisory"})]
        assert _evaluated(ctx, "document.querySelector('#result').textContent") == "Completed 1"
        assert "Screenshot captured" in browser._browser_action(ctx, "screenshot")
        screenshot = base64.b64decode(ctx.browser_state.last_screenshot_b64)
        assert screenshot.startswith(b"\x89PNG")
        screenshot_path = tmp_path / f"agency-{engine}.png"
        screenshot_path.write_bytes(screenshot)
        print(f"Browser evidence: {screenshot_path}")
        with pytest.raises(Exception, match="SyntaxError"):
            browser._browser_action(ctx, "evaluate", value="const = ;")
        assert len(hits) == 1
        assert _evaluated(ctx, "return 2 + 3;") == "5"
        with pytest.raises(Exception, match="SyntaxError"):
            browser._browser_action(ctx, "evaluate", value="window.effectCount = (window.effectCount || 0) + 1; JSON.parse('invalid')")
        assert _evaluated(ctx, "window.effectCount") == "1"
    finally:
        browser.cleanup_browser(ctx)


@pytest.mark.browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_live_cyber_acting_access_preserves_explicit_readonly(control_page, tmp_path, monkeypatch, engine):
    url, hits = control_page
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    readonly = ToolContext(repo_dir=tmp_path, drive_root=tmp_path,
        task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False))
    assert "BROWSER_LOCAL_READONLY_BLOCKED" in browser._browse_page(readonly, url, engine=engine)
    assert "BROWSER_LOCAL_READONLY_BLOCKED" in browser._browser_action(readonly, "evaluate", value="1+1")
    acting = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, workspace_root=str(tmp_path),
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree"))
    try:
        assert "Agency browser ready" in browser._browse_page(acting, url, engine=engine)
        code = "() => fetch('/api/owner/context-mode', {method:'POST', body:'{\"mode\":\"low\"}'})"
        code += ".then(r => r.json()).then(r => r.received)"
        assert _evaluated(acting, code) == "1"
        assert hits == [("/api/owner/context-mode", {"mode": "low"})]
    finally:
        browser.cleanup_browser(acting)
