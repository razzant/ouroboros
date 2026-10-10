"""Real, isolated @playwright/mcp pointer evidence; never starts Extension/GUI.

Opt in with OUROBOROS_PLAYWRIGHT_MCP_CLI and
OUROBOROS_PLAYWRIGHT_HEADLESS_SHELL. The package must be pinned to 0.0.82.
These checks prove only the non-extension headless path, not owner focus.
"""

import http.server
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time

import pytest

from scripts import browser_focus_probe as probe
from ouroboros import config, mcp_client, mcp_task_sessions, owner_pause
from ouroboros.tools.registry_core import ToolRegistry
from ouroboros.task_results import write_task_result

pytestmark = pytest.mark.serial
PAGE_DIR = Path(__file__).parent / "fixtures" / "browser_focus"
MARKER = "POINTER_PROBE_JSON:"


def test_extension_recipe_rejects_temporary_or_mismatched_live_root(tmp_path, monkeypatch):
    common = ["--source-root", str(Path(__file__).resolve().parents[1]),
              "--cli", "/pinned/cli.js", "--owner-ready", "--extension",
              "--task-id", "live-task", "--task-attempt", "2"]
    with pytest.raises(SystemExit):
        probe._validated_args([*common, "--data-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        probe._validated_args([*common, "--data-root", str(tmp_path),
                               "--live-data-root", "/different/live/data"])
    with pytest.raises(SystemExit):
        probe._validated_args([*common, "--data-root", str(tmp_path),
                               "--live-data-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        probe._validated_args(["--source-root", str(Path(__file__).resolve().parents[1]),
                               "--data-root", str(tmp_path), "--owner-ready"])
    valid_root = Path("/non-temporary/canonical-data")
    monkeypatch.setattr(Path, "is_dir", lambda self: self == valid_root)
    args = probe._validated_args([*common, "--data-root", str(valid_root),
                                  "--live-data-root", str(valid_root)])
    assert args.data_root == str(valid_root) and args.task_attempt == 2


def test_extension_recipe_requires_existing_running_task_without_overwrite(tmp_path):
    task_id = "live-task"
    from ouroboros.task_results import stamp_task_result_schema, task_result_path

    path = task_result_path(tmp_path, task_id, create=False)
    with pytest.raises(RuntimeError, match="matching current running task"):
        probe._require_running_task(tmp_path, task_id, 2)
    path.parent.mkdir()
    row = stamp_task_result_schema({"task_id": task_id, "status": "running", "task_attempt": 2})
    path.write_text(json.dumps(row))
    original = path.read_bytes()
    probe._require_running_task(tmp_path, task_id, 2)
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError, match="matching current running task"):
        probe._require_running_task(tmp_path, task_id, 3)
    path.write_text(json.dumps({**row, "status": "completed"}))
    with pytest.raises(RuntimeError, match="matching current running task"):
        probe._require_running_task(tmp_path, task_id, 2)


def test_extension_recipe_stops_after_focus_observer_loss(monkeypatch):
    assert probe._require_b_focus({"b_active": True}, False) is True
    monkeypatch.setattr(probe.sys, "platform", "darwin")
    monkeypatch.setattr(probe.shutil, "which", lambda _name: "/usr/bin/osascript")

    def unavailable(*_args, **_kwargs):
        raise probe.subprocess.TimeoutExpired("osascript", 8)

    monkeypatch.setattr(probe.subprocess, "run", unavailable)
    focus = probe._sample_focus("http://127.0.0.1:1/b.html")
    assert focus["observer"] == "unavailable" and focus["b_active"] is None
    with pytest.raises(RuntimeError, match="further pointer actions stopped"):
        probe._require_b_focus(focus, False)
    assert probe._require_b_focus(focus, True) is False
    with pytest.raises(RuntimeError, match="further pointer actions stopped"):
        probe._require_b_focus({"b_active": False}, True)


def test_focus_observer_retains_loss_and_joins_without_chrome(monkeypatch):
    sampled = threading.Event()
    observer = probe._FocusObserver("http://127.0.0.1:1/b.html")

    def sample(_url):
        row = {"observed_at": "synthetic", "monotonic": 1.0,
               "observer": "fixture", "b_active": False}
        sampled.set()
        return row

    monkeypatch.setattr(probe, "_sample_focus", sample)
    observer.start()
    assert sampled.wait(2)
    report = observer.finish()
    assert report["observer_closed"] and not report["bounded_out"]
    assert report["samples"] and report["samples"][0]["b_active"] is False
    assert report["coverage"] == "samples; transient switches may be missed"
    with pytest.raises(RuntimeError, match="further pointer actions stopped"):
        probe._require_b_focus(report["samples"][0], False)


def test_focus_observer_finishes_even_when_session_close_raises(tmp_path, monkeypatch):
    """Exercise the prepared consumer's failure cleanup without Chrome or a transport."""
    from types import SimpleNamespace

    cli = tmp_path / "cli.js"
    (tmp_path / "package.json").write_text(json.dumps({"version": "0.0.82"}))
    args = SimpleNamespace(cli=str(cli), source_root=str(tmp_path), data_root=str(tmp_path),
                           task_id="fixture", task_attempt=1, allow_unobserved_focus=False)
    registry = SimpleNamespace(_ctx=SimpleNamespace())
    manager = SimpleNamespace(reconfigure=lambda *_a: None,
                              refresh_server=lambda *_a, **_k: {"ok": True},
                              list_tools_for_registry=lambda: [{"raw_name": "browser_run_code_unsafe",
                                  "name": "mcp_browser__browser_run_code_unsafe",
                                  "schema": {"properties": {"code": {"type": "string"}}}}])
    finished = []
    observer = SimpleNamespace(samples=[], start=lambda: None,
                               finish=lambda: finished.append(True) or {
                                   "samples": [], "observer_closed": True, "bounded_out": False})
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(probe.shutil, "which", lambda *_a: "node")
    monkeypatch.setattr(probe, "_require_running_task", lambda *_a: None)
    monkeypatch.setattr("ouroboros.tools.registry_core.ToolRegistry", lambda **_k: registry)
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    monkeypatch.setattr("builtins.input", lambda *_a: "")
    monkeypatch.setattr(probe, "_parse_result", lambda *_a: {"url": "A", "state": {"counter": 0}})
    monkeypatch.setattr(probe, "_run_code", lambda *_a: SimpleNamespace(
        status="error", code="MCP_ERROR", text="fixture action lost its answer"))
    monkeypatch.setattr(probe, "_sample_focus", lambda *_a: {"b_active": True})
    monkeypatch.setattr(probe, "_FocusObserver", lambda *_a: observer)

    def failed_close(*_a):
        raise RuntimeError("fixture session close failed")

    monkeypatch.setattr(mcp_task_sessions, "stop_task", failed_close)
    ready = SimpleNamespace(clear=lambda: None, wait=lambda **_k: True)
    with pytest.raises(RuntimeError, match="fixture session close failed"):
        probe._extension_consumer(args, "A", "B", ready)
    assert finished == [True]


def _payload(result):
    assert result.status == "ok", (result.status, result.code, result.text)
    match = re.search(re.escape(MARKER) + r"([^\r\n]+)", result.text)
    assert match, result.text
    # Upstream JSON-encodes a returned JS string in its ``### Result`` line.
    return json.loads(json.loads('"' + MARKER + match.group(1)).removeprefix(MARKER))


def _code(registry, tool, body):
    return registry.execute_result(tool, {"code": f"async (page) => {{ {body} }}"})


@pytest.fixture
def consumer(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("task bridge is POSIX-only")
    pytest.importorskip("mcp")
    cli = os.environ.get("OUROBOROS_PLAYWRIGHT_MCP_CLI", "")
    shell = os.environ.get("OUROBOROS_PLAYWRIGHT_HEADLESS_SHELL", "")
    node = shutil.which("node")
    if not (cli and shell and node and Path(cli).is_file() and Path(shell).is_file()
            and "headless-shell" in Path(shell).name):
        pytest.skip("pinned CLI and headless shell are required")
    package = Path(cli).parent / "package.json"
    assert json.loads(package.read_text())["version"] == "0.0.82"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    monkeypatch.setattr(owner_pause, "submit_async_preparation", lambda factory, **_kw: factory())
    # Keep production check_safety and bridge request policy; only replace its
    # paid external assessment so this isolated test needs no owner credential.
    monkeypatch.setattr("ouroboros.safety._run_llm_check", lambda *_a, **_k: (True, ""))
    manager = mcp_client.MCPManager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 30, "MCP_SERVERS": [{
        "id": "browser", "enabled": True, "transport": "stdio", "command": node,
        "browser_bridge": True,
        "args": [cli, "--headless", "--isolated", "--browser", "chromium", "--executable-path", shell],
    }]})
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kw: None)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = f"pointer-{tmp_path.name}"
    registry._ctx.task_attempt = 1
    registry._ctx.task_lifecycle_bound = True
    registry._ctx.messages = []
    write_task_result(tmp_path, registry._ctx.task_id, "running")
    assert manager.refresh_server("browser", authority=registry._ctx)["ok"]
    tools = {row["raw_name"]: row for row in manager.list_tools_for_registry()}
    assert "browser_run_code_unsafe" in tools, sorted(tools)
    assert "browser_click" in tools and tools["browser_click"]["name"] != tools["browser_run_code_unsafe"]["name"]
    schema = tools["browser_run_code_unsafe"]["schema"]
    assert schema["properties"]["code"]["type"] == "string", schema
    tool = tools["browser_run_code_unsafe"]["name"]
    assert tool == "mcp_browser__browser_run_code_unsafe"

    observed = {"events": [], "states": []}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PAGE_DIR), **kwargs)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path not in ("/event", "/state"):
                self.send_error(404)
                return
            observed["events" if self.path == "/event" else "states"].append(json.loads(body))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_args):
            pass

    site = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=site.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{site.server_address[1]}"
    try:
        yield registry, manager, tool, base, observed
    finally:
        assert mcp_task_sessions.stop_task(registry._ctx) == [{"server": "browser", "closure": "confirmed"}]
        site.shutdown()
        site.server_close()
        thread.join(timeout=5)


def _open_a(registry, tool, base):
    state = _payload(_code(registry, tool,
        f"await page.goto({json.dumps(base + '/a.html')}); return '{MARKER}' + JSON.stringify(await page.evaluate(() => window.probe.state()));"))
    assert state["counter"] == 0 and state["events"] == []


def test_official_force_pointer_three_exact_trusted_events(consumer):
    registry, _manager, tool, base, _observed = consumer
    _open_a(registry, tool, base)
    exposure = _payload(_code(registry, tool,
        "const exposure = await page.evaluate(() => new Promise(resolve => { "
        "let settled = false; requestAnimationFrame(() => { if (!settled) { settled = true; "
        "resolve({ visibilityState: document.visibilityState, rafFired: true }); } }); "
        "setTimeout(() => { if (!settled) resolve({ visibilityState: document.visibilityState, rafFired: false }); }, 500); "
        "})); "
        f"return '{MARKER}' + JSON.stringify(exposure);"))
    assert exposure["visibilityState"] in ("visible", "hidden")
    assert isinstance(exposure["rafFired"], bool)
    print("HEADLESS_PAGE_EXPOSURE", json.dumps(exposure, sort_keys=True))
    for count in (1, 2, 3):
        state = _payload(_code(registry, tool,
            f"await page.getByRole('button', {{ name: 'Count' }}).click({{ force: true }}); "
            f"return '{MARKER}' + JSON.stringify(await page.evaluate(() => window.probe.state()));"))
        assert state["counter"] == count, state
        assert state["events"] == [
            {"type": event, "target": "count", "isTrusted": True}
            for _ in range(count) for event in ("pointerdown", "mousedown", "mouseup", "click")
        ], state


def test_overlay_and_detached_target_do_not_count(consumer):
    registry, _manager, tool, base, _observed = consumer
    _open_a(registry, tool, base)
    over = _payload(_code(registry, tool,
        "await page.evaluate(() => window.probe.cover()); "
        "let error = ''; try { await page.getByRole('button', { name: 'Count' }).click({ force: true, timeout: 1500 }); } "
        "catch (e) { error = String(e); } "
        f"return '{MARKER}' + JSON.stringify({{error, state: await page.evaluate(() => window.probe.state())}});"))
    assert over["state"]["counter"] == 0, over
    assert all(event["target"] != "count" for event in over["state"]["events"]), over
    assert over["state"]["events"] == [
        {"type": event, "target": "cover", "isTrusted": True}
        for event in ("pointerdown", "mousedown", "mouseup", "click")
    ], over
    _payload(_code(registry, tool,
        f"await page.goto({json.dumps(base + '/a.html')}); return '{MARKER}' + JSON.stringify(await page.evaluate(() => window.probe.state()));"))
    detached = _payload(_code(registry, tool,
        "const handle = await page.getByRole('button', { name: 'Count' }).elementHandle(); "
        "await page.evaluate(() => window.probe.detach()); "
        "let error = ''; try { await handle.click({ force: true, timeout: 1500 }); } catch (e) { error = String(e); } "
        f"return '{MARKER}' + JSON.stringify({{error, state: await page.evaluate(() => window.probe.state())}});"))
    assert "not attached to the dom" in detached["error"].lower(), detached
    assert detached["state"]["counter"] == 0 and detached["state"]["events"] == [], detached


def test_dom_click_counter_is_not_pointer_evidence(consumer):
    registry, _manager, tool, base, _observed = consumer
    _open_a(registry, tool, base)
    state = _payload(_code(registry, tool,
        "await page.evaluate(() => document.querySelector('#count').click()); "
        f"return '{MARKER}' + JSON.stringify(await page.evaluate(() => window.probe.state()));"))
    assert state["counter"] == 1, state
    assert state["events"] == [{"type": "click", "target": "count", "isTrusted": False}], state


def test_one_send_timeout_reports_unknown_without_retry(consumer, monkeypatch):
    registry, manager, tool, base, observed = consumer
    _open_a(registry, tool, base)
    manager._tool_timeout_sec = 1
    sent = []
    original = mcp_task_sessions.call

    def counting_call(cfg, prefixed, name, args, ctx, timeout):
        if name == "browser_run_code_unsafe":
            sent.append(args["code"])
        return original(cfg, prefixed, name, args, ctx, timeout)

    monkeypatch.setattr(mcp_task_sessions, "call", counting_call)
    result = _code(registry, tool,
        "await page.getByRole('button', { name: 'Count' }).click({ force: true }); "
        "await page.waitForTimeout(4000); return 'late';")
    assert len(sent) == 1, sent
    assert result.status == "timeout" and result.code == "MCP_TIMEOUT", result.text
    assert "effect is unknown" in result.text, result.text
    assert "session was closed: confirmed" in result.text, result.text
    deadline = time.monotonic() + 2
    while (not observed["states"] or len(observed["events"]) < 4) and time.monotonic() < deadline:
        time.sleep(.02)
    # This independent target-side receipt reconciles this instance. The
    # timeout text alone never proves whether the click ran.
    assert observed["states"] == [{"counter": 1}], observed
    assert [{key: item[key] for key in ("type", "target", "isTrusted")}
            for item in sorted(observed["events"], key=lambda item: item["seq"])] == [
        {"type": event, "target": "count", "isTrusted": True}
        for event in ("pointerdown", "mousedown", "mouseup", "click")
    ], observed
