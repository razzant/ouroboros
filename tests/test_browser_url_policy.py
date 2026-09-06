"""Unit tests for the subagent browser URL policy (no Playwright needed).

Verifies the shared target decision: readonly/acting subagents
may browse external HTTP(S), localhost on non-Ouroboros ports, and file:// under
their workspace, while the Ouroboros control-plane ports, private/link-local IPs,
DNS-rebind, and other schemes stay blocked.
"""
from __future__ import annotations

import os
from functools import partial
from types import SimpleNamespace

import pytest

from ouroboros.browser_policy import browser_url_block_reason
from ouroboros.server_process import record_service_binding

_blocked = partial(browser_url_block_reason, restricted=True)


@pytest.fixture(autouse=True)
def live_bindings(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    for kind, port in (("main", 8765), ("local_model", 8766), ("host_service", 8767)):
        record_service_binding(tmp_path, kind, "::", port, pid=os.getpid())
    return tmp_path


def _ctx(workspace_root: str = ""):
    return SimpleNamespace(workspace_root=workspace_root)


def test_ouroboros_control_ports_blocked_on_loopback():
    for url in (
        "http://127.0.0.1:8765",
        "http://localhost:8765/api/settings",
        "http://127.0.0.1:8766",   # local model
        "http://127.0.0.1:8767",   # host service
        "http://[::1]:8765",
    ):
        assert _blocked(url, _ctx()), url


def test_localhost_non_control_ports_allowed():
    for url in (
        "http://localhost:3000/",
        "http://127.0.0.1:5173/app",
        "http://localhost/",
        "http://127.0.0.1:8080",
    ):
        assert _blocked(url, _ctx()) == "", url


def test_private_and_linklocal_blocked():
    for url in (
        "http://192.168.1.1",
        "http://10.0.0.1",
        "http://169.254.169.254",
        "http://[::]/",
        "http://172.16.0.1",
    ):
        assert _blocked(url, _ctx()), url


def test_non_http_schemes_blocked():
    for url in ("data:text/html,<h1>x</h1>", "about:blank", "ws://localhost:3000"):
        assert _blocked(url, _ctx()), url


def test_file_url_blocked_without_workspace():
    assert _blocked("file:///etc/passwd", _ctx())
    assert _blocked("file:///tmp/app/index.html", _ctx(""))


def test_file_url_scoped_to_workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "build").mkdir(parents=True)
    app = ws / "build" / "index.html"
    app.write_text("<h1>app</h1>", encoding="utf-8")
    outside = tmp_path / "secret.json"
    outside.write_text("{}", encoding="utf-8")
    ctx = _ctx(str(ws))
    # Path.as_uri() yields a platform-correct file URL (file:///C:/... on Windows).
    assert _blocked(app.as_uri(), ctx) == ""
    assert _blocked(outside.as_uri(), ctx)


def test_actual_service_bindings_win_over_env_and_default_ports(live_bindings, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SERVER_PORT", "7777")
    monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", "7778")
    record_service_binding(live_bindings, "main", "127.0.0.1", 8900, pid=os.getpid())
    record_service_binding(live_bindings, "host_service", "127.0.0.1", 8902, pid=os.getpid())
    assert _blocked("http://127.0.0.1:8900", _ctx())
    assert _blocked("http://127.0.0.1:8902", _ctx())
    for port in (8901, 7777, 7778, 8765, 8767, 3000):
        assert not _blocked(f"http://127.0.0.1:{port}", _ctx())


def test_integer_port_file_is_not_an_invented_live_service(live_bindings, monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL_PORT", "9002")
    record_service_binding(live_bindings, "local_model", "127.0.0.1", 9001, pid=os.getpid())
    assert _blocked("http://127.0.0.1:9001", _ctx())
    assert not _blocked("http://127.0.0.1:9002", _ctx())
    (live_bindings / "state" / "server_port").write_text("9100", encoding="utf-8")
    assert not _blocked("http://127.0.0.1:9100", _ctx())
    record_service_binding(live_bindings, "main", "127.0.0.1", 9100, pid=os.getpid())
    assert _blocked("http://127.0.0.1:9100", _ctx())


def test_owner_settings_post_blocked_in_browser():
    """A browser POST /api/settings carrying an owner-only self-modification toggle
    (the click+Save bypass) must be blocked for every browser session."""
    from ouroboros.browser_policy import _is_owner_settings_self_elevation_post

    def req(method, url, body):
        return SimpleNamespace(method=method, url=url, post_data=body)

    base = "http://127.0.0.1:8765/api/settings"
    assert _is_owner_settings_self_elevation_post(req("POST", base, '{"OUROBOROS_POST_TASK_EVOLUTION":"true"}'))
    assert _is_owner_settings_self_elevation_post(req("POST", base, '{"OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS":"on"}'))
    assert _is_owner_settings_self_elevation_post(req("POST", base, '{"OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE":"x"}'))
    # P34P1.10: the delegated-executor POLICY. D1 makes the executor axis the OWNER's,
    # and this key IS that axis — which route answers, on whose subscription. It rides
    # the generic settings path so the Settings UI can set a route string without
    # ceremony, so the agent-reachable browser POST is where it has to be refused.
    assert _is_owner_settings_self_elevation_post(req("POST", base, '{"OUROBOROS_SUBAGENT_HARNESS":"codex=gpt-5.6:high"}'))
    assert _is_owner_settings_self_elevation_post(req("POST", base, '{"OUROBOROS_SUBAGENT_HARNESS":""}'))
    assert _is_owner_settings_self_elevation_post(req("GET", base, None)) is False
    assert _is_owner_settings_self_elevation_post(req("POST", base, '{"OPENAI_API_KEY":"x"}')) is False
    assert _is_owner_settings_self_elevation_post(
        req("POST", "http://127.0.0.1:8765/api/tasks", '{"OUROBOROS_POST_TASK_EVOLUTION":"true"}')) is False


def test_vlm_and_screenshot_available_to_subagents():
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )
    # Child profiles remain the explicit narrowing layer regardless of whether
    # their assigned worktree is represented as a workspace.
    for name in ("analyze_screenshot", "vlm_query", "browse_page", "browser_action"):
        assert name in LOCAL_READONLY_SUBAGENT_TOOL_NAMES, name
        assert name in ACTING_SUBAGENT_TOOL_NAMES, name


def test_expected_endpoints_without_a_snapshot_refuse_unknown_identity(tmp_path, monkeypatch):
    """No bindings: the recorded main port and the configured Host Service port are
    expected Ouroboros endpoints whose owner cannot be verified — refused, not foreign —
    while every other loopback port keeps working."""
    from ouroboros import config
    from ouroboros.browser_policy import browser_request_block_reason, runtime_service_kind
    from ouroboros.server_process import SERVICE_IDENTITY_UNKNOWN

    root = tmp_path / "legacy"
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", "9105")
    assert not _blocked("http://127.0.0.1:9100", _ctx())
    assert not _blocked("http://127.0.0.1:9105", _ctx())  # no main expected, no Host Service expected
    (root / "state").mkdir(parents=True)
    (root / "state" / "server_port").write_text("9100", encoding="utf-8")
    for url in ("http://127.0.0.1:9100/api/settings", "http://localhost:9100", "http://[::1]:9100"):
        assert runtime_service_kind(url, _ctx()) == SERVICE_IDENTITY_UNKNOWN, url
        assert "identity could not be verified" in _blocked(url, _ctx()), url
    assert runtime_service_kind("http://127.0.0.1:9105/identity", _ctx()) == SERVICE_IDENTITY_UNKNOWN
    assert "identity could not be verified" in _blocked("http://127.0.0.1:9105/identity", _ctx())
    # The Host Service binds the IPv4 loopback only; other loopback ports stay open.
    for url in ("http://[::1]:9105", "http://127.0.0.1:9101", "http://localhost:8765", "http://127.0.0.1:8767"):
        assert runtime_service_kind(url, _ctx()) == "" and not _blocked(url, _ctx()), url
    owner_post = SimpleNamespace(method="POST", url="http://127.0.0.1:9100/api/owner/context-mode", post_data=None)
    assert "BROWSER_OWNER_CONTROL_BLOCKED" in browser_request_block_reason(owner_post, _ctx(), restricted=False)
    owner_post.url = "http://127.0.0.1:9101/api/owner/context-mode"
    assert browser_request_block_reason(owner_post, _ctx(), restricted=False) == ""


def test_launcher_process_record_proves_the_recorded_main(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.browser_policy import runtime_service_kind
    from ouroboros.platform_layer import process_command
    from ouroboros.server_process import SERVICE_IDENTITY_UNKNOWN
    from ouroboros.utils import atomic_write_json

    root = tmp_path / "packaged"
    monkeypatch.setattr(config, "DATA_DIR", root)
    (root / "state").mkdir(parents=True)
    (root / "state" / "server_port").write_text("9100", encoding="utf-8")
    command = process_command(os.getpid())
    if not command:
        pytest.skip("no readable command line on this platform")
    record = {"pid": os.getpid(), "server_path": command.split()[0], "port": 9100}
    atomic_write_json(root / "state" / "server_process.json", record)
    assert runtime_service_kind("http://127.0.0.1:9100", _ctx()) == "main"
    assert "actual Ouroboros" in _blocked("http://127.0.0.1:9100", _ctx())
    atomic_write_json(root / "state" / "server_process.json", {**record, "port": 9200})
    assert runtime_service_kind("http://127.0.0.1:9100", _ctx()) == SERVICE_IDENTITY_UNKNOWN
    atomic_write_json(root / "state" / "server_process.json", {**record, "server_path": "/nowhere/server.py"})
    assert runtime_service_kind("http://127.0.0.1:9100", _ctx()) == SERVICE_IDENTITY_UNKNOWN


def test_stale_snapshot_is_evidence_of_a_gone_process_not_a_free_port(tmp_path, monkeypatch):
    from ouroboros import config, server_process
    from ouroboros.browser_policy import runtime_service_kind
    from ouroboros.server_process import SERVICE_IDENTITY_UNKNOWN
    from ouroboros.utils import update_json_locked

    root = tmp_path / "stale"
    monkeypatch.setattr(config, "DATA_DIR", root)
    record_service_binding(root, "main", "127.0.0.1", 9100, pid=os.getpid())
    record_service_binding(root, "host_service", "127.0.0.1", 9105, pid=os.getpid())
    monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", "9105")
    (root / "state" / "server_port").write_text("9100", encoding="utf-8")

    def stale(current):
        for binding in current.values():
            binding["fingerprint"] = {"start_time": "not-the-current-process"}
        return current

    update_json_locked(root / "state" / "server_port.bindings.json", stale)
    assert runtime_service_kind("http://127.0.0.1:9100", _ctx()) == SERVICE_IDENTITY_UNKNOWN
    assert runtime_service_kind("http://127.0.0.1:9105", _ctx()) == SERVICE_IDENTITY_UNKNOWN
    # The binding still names its endpoint even if the integer port file moved.
    (root / "state" / "server_port").write_text("9300", encoding="utf-8")
    assert runtime_service_kind("http://127.0.0.1:9100", _ctx()) == SERVICE_IDENTITY_UNKNOWN
    assert runtime_service_kind("http://127.0.0.1:9300", _ctx()) == SERVICE_IDENTITY_UNKNOWN
    # A live main replaces its own old binding, but cannot retire another
    # service's record. Only that service owner's exact retirement frees it.
    record_service_binding(root, "main", "127.0.0.1", 9400, pid=os.getpid())
    assert runtime_service_kind("http://127.0.0.1:9400", _ctx()) == "main"
    assert runtime_service_kind("http://127.0.0.1:9105", _ctx()) == SERVICE_IDENTITY_UNKNOWN
    old_host = server_process.read_service_bindings(root)["host_service"]
    server_process.clear_service_binding(root, "host_service", old_host)
    for url in ("http://127.0.0.1:9300", "http://127.0.0.1:9105", "http://127.0.0.1:9100"):
        assert runtime_service_kind(url, _ctx()) == "", url
    # An identity probe that cannot answer is unknown, never foreign.
    monkeypatch.setattr(server_process, "service_binding_is_live",
                        lambda _binding: (_ for _ in ()).throw(OSError(5, "access denied")))
    assert runtime_service_kind("http://127.0.0.1:9400", _ctx()) == SERVICE_IDENTITY_UNKNOWN
    assert "identity could not be verified" in _blocked("http://127.0.0.1:9400", _ctx())


def test_unreadable_snapshot_keeps_the_recorded_expectations(tmp_path, monkeypatch, caplog):
    from ouroboros import config
    from ouroboros.browser_policy import runtime_service_kind
    from ouroboros.server_process import SERVICE_IDENTITY_UNKNOWN

    root = tmp_path / "corrupt"
    monkeypatch.setattr(config, "DATA_DIR", root)
    (root / "state").mkdir(parents=True)
    (root / "state" / "server_port.bindings.json").write_text('{"main": 1, "stranger": {}}', encoding="utf-8")
    (root / "state" / "server_port").write_text("9100", encoding="utf-8")
    with caplog.at_level("WARNING", logger="server"):
        assert runtime_service_kind("http://127.0.0.1:9100", _ctx()) == SERVICE_IDENTITY_UNKNOWN
        assert runtime_service_kind("http://127.0.0.1:9101", _ctx()) == ""
    assert any("unreadable" in record.getMessage() for record in caplog.records)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process observations; Windows has its creation-time tests")
@pytest.mark.parametrize("kind", ["main", "host_service", "local_model"])
@pytest.mark.parametrize("uncertainty", ["mismatched", "unmeasured", "missing_fingerprint"])
def test_matching_binding_uncertainty_does_not_permit_a_control_endpoint(tmp_path, monkeypatch, kind, uncertainty):
    from ouroboros import config, process_custody, server_process
    from ouroboros.browser_policy import runtime_service_kind
    from ouroboros.utils import update_json_locked

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    record_service_binding(tmp_path, kind, "127.0.0.1", 9410, pid=os.getpid())
    if uncertainty == "unmeasured":
        monkeypatch.setattr(process_custody, "process_start_time", lambda _pid: "")
        monkeypatch.setattr(process_custody, "_live_cmd_sha256", lambda _pid: "")
    else:
        def change(current):
            current[kind]["fingerprint"] = (
                {"start_time": "not-this-process", "cmd_sha256": "wrong"}
                if uncertainty == "mismatched" else {}
            )
            return current
        update_json_locked(tmp_path / "state/server_port.bindings.json", change)

    assert runtime_service_kind("http://127.0.0.1:9410", _ctx()) == server_process.SERVICE_IDENTITY_UNKNOWN
    assert "identity could not be verified" in _blocked("http://127.0.0.1:9410", _ctx())
    assert not _blocked("http://127.0.0.1:9411/api/owner/context-mode", _ctx())
