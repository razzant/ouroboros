"""Informational endpoint identity is separate from process/reaper authority."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.mark.serial
def test_binding_metadata_failure_keeps_the_listener_and_legacy_browser_boundary(tmp_path, monkeypatch, caplog):
    import socket
    from ouroboros import server_process
    from ouroboros.browser_policy import browser_url_block_reason, runtime_service_kind
    from ouroboros.server_entrypoint import bound_service_socket, write_port_file

    root, child = tmp_path / 'host', tmp_path / 'child'
    monkeypatch.setattr(config, 'DATA_DIR', root)
    def unavailable(*args, **kwargs):
        raise RuntimeError('process start time unavailable')
    monkeypatch.setattr(server_process, 'record_service_binding', unavailable)
    with caplog.at_level('WARNING'), bound_service_socket(root, 'main', '127.0.0.1', 0) as listener:
        port = listener.getsockname()[1]
        write_port_file(root / 'state/server_port', port)
        listener.listen(1)
        with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
            conn, _ = listener.accept()
            with conn:
                conn.sendall(b'listener survived')
            assert client.recv(64) == b'listener survived'
        ctx = type('Context', (), {'drive_root': child})()
        url = f'http://127.0.0.1:{port}'
        assert runtime_service_kind(url, ctx) == server_process.SERVICE_IDENTITY_UNKNOWN
        assert 'identity could not be verified' in browser_url_block_reason(url, ctx, restricted=True)
        assert not (root / 'state/server_port.bindings.json').exists()
    assert 'continuing with the bound socket' in caplog.text


def test_isolated_child_also_checks_the_configured_host_bindings(tmp_path, monkeypatch):
    from ouroboros.browser_policy import runtime_service_kind

    root, child = tmp_path / 'host', tmp_path / 'child'
    monkeypatch.setattr(config, 'DATA_DIR', root)
    server_process.record_service_binding(root, 'main', '127.0.0.1', 9460, pid=os.getpid())
    ctx = type('Context', (), {'drive_root': child})()
    assert runtime_service_kind('http://127.0.0.1:9460', ctx) == 'main'
    assert runtime_service_kind('http://127.0.0.1:9461', ctx) == ''

from ouroboros import config, server_process
from ouroboros.browser_policy import runtime_service_kind
from ouroboros.platform_layer import subprocess_new_group_kwargs
from ouroboros.server_entrypoint import bound_service_socket


def test_bound_socket_publishes_actual_port_and_cleans_only_its_binding(tmp_path):
    ctx = type("Context", (), {"drive_root": tmp_path})()
    with bound_service_socket(tmp_path, "main", "127.0.0.1", 0) as listener:
        port = listener.getsockname()[1]
        assert port > 0
        assert runtime_service_kind(f"http://127.0.0.1:{port}/anything", ctx) == "main"
        assert runtime_service_kind(f"http://127.0.0.2:{port}/api/settings", ctx) == ""
        old = server_process.read_service_bindings(tmp_path)["main"]
        replacement = server_process.record_service_binding(tmp_path, "main", "127.0.0.1", port + 1, pid=os.getpid())
        server_process.clear_service_binding(tmp_path, "main", old)
        assert server_process.read_service_bindings(tmp_path)["main"] == replacement
    # The old context's finalizer also cannot erase the replacement.
    assert server_process.read_service_bindings(tmp_path)["main"] == replacement
    server_process.clear_service_binding(tmp_path, "main", replacement)
    assert server_process.read_service_bindings(tmp_path) == {}
    with socket.socket() as reused:
        reused.bind(("127.0.0.1", port))
        assert runtime_service_kind(f"http://127.0.0.1:{port}/api/settings", ctx) == ""
    assert not (tmp_path / "state" / "process_ledger.jsonl").exists()


def test_failed_startup_closes_prebound_socket(tmp_path):
    port = 0
    with pytest.raises(RuntimeError, match="startup failed"):
        with bound_service_socket(tmp_path, "host_service", "127.0.0.1", 0) as listener:
            port = listener.getsockname()[1]
            raise RuntimeError("startup failed")
    assert server_process.read_service_bindings(tmp_path) == {}
    with socket.socket() as reused:
        reused.bind(("127.0.0.1", port))


def test_binding_write_failure_keeps_serving_without_enrollment_or_a_leaked_socket(tmp_path, monkeypatch, caplog):
    with socket.socket() as chosen:
        chosen.bind(("127.0.0.1", 0))
        port = chosen.getsockname()[1]
    monkeypatch.setattr(server_process, "update_json_locked", lambda *_a, **_k: (_ for _ in ()).throw(OSError("full")))
    with caplog.at_level('WARNING'), bound_service_socket(tmp_path, "main", "127.0.0.1", port) as listener:
        assert listener.getsockname()[1] == port
        listener.listen(1)
    with socket.socket() as reused:
        reused.bind(("127.0.0.1", port))
    assert not (tmp_path / "state" / "process_ledger.jsonl").exists()
    assert 'continuing with the bound socket' in caplog.text


def test_concurrent_bindings_merge_and_recycled_identity_is_not_live(tmp_path):
    def publish(item):
        return server_process.record_service_binding(tmp_path, item[0], "127.0.0.1", item[1], pid=os.getpid())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(publish, [("main", 3000), ("host_service", 3001)]))
    bindings = server_process.read_service_bindings(tmp_path)
    assert set(bindings) == {"main", "host_service"}
    assert server_process.service_binding_is_live(bindings["main"])
    stale = {**bindings["main"], "fingerprint": {"start_time": "not-the-current-process"}}
    assert not server_process.service_binding_is_live(stale)


def test_child_observes_late_local_model_health_binding_and_stop(tmp_path, monkeypatch):
    from ouroboros.local_model import LocalModelManager

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    model_script = tmp_path / "model_fixture.py"
    model_script.write_text('''from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  body=b'{"data":[{"id":"fixture","context_window":4096}]}'
  self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
 def log_message(self,*args): pass
server=HTTPServer(('127.0.0.1',0),Handler)
print(server.server_address[1],flush=True)
server.serve_forever()
''', encoding="utf-8")
    model = subprocess.Popen([sys.executable, str(model_script)], stdout=subprocess.PIPE,
                             text=True, **subprocess_new_group_kwargs())
    reader_code = """import json,sys
from pathlib import Path
from ouroboros.server_process import read_service_bindings
for line in sys.stdin:
 print(json.dumps(read_service_bindings(Path(sys.argv[1]))),flush=True)
"""
    reader = subprocess.Popen([sys.executable, "-c", reader_code, str(tmp_path)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                              **subprocess_new_group_kwargs())
    manager = LocalModelManager()
    try:
        port = int(model.stdout.readline())
        def read():
            reader.stdin.write("read\n"); reader.stdin.flush()
            return json.loads(reader.stdout.readline())
        assert read() == {}
        manager._proc, manager._port = model, port
        manager._wait_for_healthy(timeout=2)
        assert manager.is_running
        bound = read()["local_model"]
        assert bound["pid"] == model.pid and bound["port"] == port
        assert server_process.service_binding_is_live(bound)
        if os.name != "nt":
            groups = {"model": os.getpgid(model.pid), "reader": os.getpgid(reader.pid),
                      "pytest": os.getpgrp(), "checker": os.getpgid(os.getppid())}
            assert groups["model"] == model.pid and groups["reader"] == reader.pid
            assert groups["model"] not in {groups["pytest"], groups["checker"], groups["reader"]}
            print("isolated fixture process groups:", groups)
        manager.stop_server()
        assert model.poll() is not None
        assert read() == {}
        assert not server_process.service_binding_is_live(bound)
    finally:
        for proc in (model, reader):
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=5)
            if proc.stdin: proc.stdin.close()
            if proc.stdout: proc.stdout.close()


@pytest.mark.serial
def test_local_model_custody_row_identifies_a_legacy_server(tmp_path, monkeypatch):
    """Without a binding, the local model's own custody row plus its live argv prove
    the endpoint; a dead row proves nothing and the port becomes ordinary."""
    from ouroboros.browser_policy import runtime_service_kind
    from ouroboros.platform_layer import process_command
    from ouroboros.process_custody import record_process

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    ctx = type("Context", (), {"drive_root": tmp_path})()
    argv = [sys.executable, "-c", "import time; time.sleep(120)", "--port", "9333"]
    model = subprocess.Popen(argv, **subprocess_new_group_kwargs())
    try:
        if "--port 9333" not in process_command(model.pid):
            pytest.skip("no readable command line on this platform")
        record_process(tmp_path, pid=model.pid, cmd=argv, purpose="local_model_server", scope="session")
        assert runtime_service_kind("http://127.0.0.1:9333/v1/models", ctx) == "local_model"
        assert runtime_service_kind("http://localhost:9333", ctx) == "local_model"
        assert runtime_service_kind("http://127.0.0.1:9334", ctx) == ""
        assert not (tmp_path / "state" / "server_port.bindings.json").exists()
    finally:
        model.terminate()
        model.wait(timeout=5)
    assert runtime_service_kind("http://127.0.0.1:9333/v1/models", ctx) == ""
