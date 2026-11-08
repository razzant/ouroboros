"""Successful extension children disclose best-effort relay loss without failing work."""
import concurrent.futures
import contextlib
import http.server
import json
import os
from pathlib import Path
import socket
import threading
import time

import pytest

from tests._extension_loader_shared import _prepare_extension
from tests._shared import clean_extension_runtime_state


@pytest.fixture(autouse=True)
def relay_state(monkeypatch):
    from ouroboros.extension_plugin_api import take_child_ws_relay_failures
    from ouroboros.tools.process_facts import consume_last_process_facts

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    take_child_ws_relay_failures()
    consume_last_process_facts()
    yield
    take_child_ws_relay_failures()
    consume_last_process_facts()
    clean_extension_runtime_state()


@contextlib.contextmanager
def relay_endpoint(status):
    """Own every endpoint; never use the installation's real Host Service."""
    calls = []
    if status == "missing_env":
        yield {}, calls
        return
    if status == "connection_refused":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            yield {"HOST_SERVICE_URL": f"http://127.0.0.1:{sock.getsockname()[1]}",
                   "HOST_SERVICE_TOKEN": "ws-test-fixture"}, calls
        return

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", "0")))
            calls.append(self.path)
            body = json.dumps({"ok": status == 202}).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield {"HOST_SERVICE_URL": f"http://127.0.0.1:{server.server_address[1]}",
               "HOST_SERVICE_TOKEN": "ws-test-fixture"}, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


def child_env(tmp_path, drive, repo):
    env = dict(os.environ)
    env.pop("HOST_SERVICE_TOKEN", None)
    env.pop("HOST_SERVICE_URL", None)
    env.update({"OUROBOROS_EXTENSION_PROCESS_CHILD": "1",
                "OUROBOROS_APP_ROOT": str(tmp_path), "OUROBOROS_REPO_DIR": str(repo),
                "OUROBOROS_DATA_DIR": str(drive),
                "OUROBOROS_SETTINGS_PATH": str(drive / "settings.json")})
    return env


def run_relay_child(tmp_path, status, *, mode="tool", concurrent=False):
    from ouroboros import extension_process_runner as runner
    from ouroboros.extension_surface_names import extension_surface_name

    plugin = '''def register(api):
    def send(index):
        assert api.send_ws_message('progress', {'index': index}) is None
    def run(ctx):
        SEND
        return 'successful producer result'
    api.register_tool('run', run, description='relay fixture', schema={})
    api.register_ws_handler('run', run)
    api.register_route('run', lambda request: {'value': run(request)})
    CATALOG
'''.replace('SEND', "import concurrent.futures\n        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:\n            list(pool.map(send, range(1000)))" if concurrent else "for index in range(5):\n            send(index)")
    plugin = plugin.replace('CATALOG', 'run(None)' if mode == 'catalog' else 'pass')
    skill, skills_repo, drive = _prepare_extension(
        tmp_path, "ws_diagnostic", plugin, permissions=["tool", "ws_handler", "route"])
    repo = Path(__file__).resolve().parents[1]
    env = child_env(tmp_path, drive, repo)
    with relay_endpoint(status) as (bridge, requests):
        env.update(bridge)
        surface = (f"/api/extensions/{skill.name}/run" if mode == 'route'
                   else extension_surface_name(skill.name, "run"))
        result = runner._run_child(
            {"mode": mode, "skill_name": skill.name, "surface": surface,
             "args": {}, "message": {}, "request": {"method": "GET"},
             "drive_root": str(drive), "repo_dir": str(repo),
             "skills_repo_path": str(skills_repo)},
            skill_dir=skill.skill_dir, drive_root=drive, repo_dir=repo,
            env=env, timeout_sec=30)
    return result, requests


@pytest.mark.parametrize(("status", "reason"), [
    (202, None), (302, "http_error"),
    ("connection_refused", "transport_error"),
])
def test_successful_child_publishes_relay_aggregate(tmp_path, status, reason, caplog):
    from ouroboros.tools.process_facts import consume_last_process_facts

    result, requests = run_relay_child(tmp_path, status)
    assert result["ok"] is True
    assert result["result"] == "successful producer result"
    facts = consume_last_process_facts()
    assert facts["exit_code"] == 0
    reports = [row for row in caplog.records if "WS relay failures" in row.getMessage()]
    if reason:
        assert result["ws_relay_failures"] == facts["ws_relay_failures"] == {reason: 5}
        assert len(reports) == 1
        assert "ws_diagnostic" in reports[0].getMessage()
        assert "ws-test-fixture" not in reports[0].getMessage()
        assert "http://" not in reports[0].getMessage()
    else:
        assert "ws_relay_failures" not in result and "ws_relay_failures" not in facts
        assert reports == []  # Accepted does not become a browser-delivery claim.
    assert len(requests) == (5 if isinstance(status, int) else 0)


@pytest.mark.parametrize("mode", ["catalog", "ws"])
def test_one_shot_modes_share_diagnostic_delivery(tmp_path, mode, caplog):
    from ouroboros.tools.process_facts import consume_last_process_facts

    result, _ = run_relay_child(tmp_path, "missing_env", mode=mode)
    assert result["ok"] is True
    assert result["ws_relay_failures"] == {"missing_transport": 5}
    assert consume_last_process_facts()["ws_relay_failures"] == {"missing_transport": 5}
    assert len([row for row in caplog.records if "WS relay failures" in row.getMessage()]) == 1


def test_concurrent_child_sends_have_one_bounded_aggregate(tmp_path, caplog):
    result, _ = run_relay_child(tmp_path, "missing_env", concurrent=True)
    assert result["result"] == "successful producer result"
    assert result["ws_relay_failures"] == {"missing_transport": 1000}
    assert len(json.dumps(result["ws_relay_failures"])) < 100
    assert len([row for row in caplog.records if "WS relay failures" in row.getMessage()]) == 1


def test_reported_diagnostics_cannot_replace_measured_process_facts():
    from ouroboros.tools.process_facts import consume_last_process_facts, publish_process_facts

    publish_process_facts(returncode=0, started_ts=time.monotonic(), ws_relay_failures={
        "exit_code": -9, "url": "private", "rate_limited": 3,
        "http_error": "private", "missing_transport": True,
        "http_client_error": -1, "http_server_error": 0,
    })
    facts = consume_last_process_facts()
    assert facts["exit_code"] == 0 and "signal" not in facts
    assert facts["ws_relay_failures"] == {"rate_limited": 3}
    publish_process_facts(returncode=0, started_ts=time.monotonic())
    assert "ws_relay_failures" not in consume_last_process_facts()


def test_collector_drains_without_cross_call_leakage():
    from ouroboros.extension_plugin_api import _record_ws_relay_failure, take_child_ws_relay_failures

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: _record_ws_relay_failure("transport_error"), range(2000)))
    assert take_child_ws_relay_failures() == {"transport_error": 2000}
    assert take_child_ws_relay_failures() == {}


def test_real_child_diagnostic_reaches_successful_tool_trace(tmp_path):
    from tests.test_process_signal_observability import _run_single

    def execute(_name, _args):
        result, _ = run_relay_child(tmp_path, "missing_env")
        return result["result"]

    result = _run_single(tmp_path, execute, tool="ext_ws_diagnostic_run")
    assert result["result"] == "successful producer result"
    assert result["result_meta"]["exit_code"] == 0
    assert result["result_meta"]["ws_relay_failures"] == {"missing_transport": 5}
    rows = [json.loads(line) for line in (tmp_path / "logs" / "tools.jsonl").read_text().splitlines()]
    assert rows[-1]["ws_relay_failures"] == {"missing_transport": 5}
