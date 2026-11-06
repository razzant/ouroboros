"""Real loopback HTTP confirms the meter's wall-clock bound without a provider."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import subprocess
import threading
import time

import pytest

from devtools.benchmarks.cowork_bench import campaign



@contextmanager
def meter_endpoint(header_delay, body_delay, usage):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(dict(self.headers))
            body = json.dumps({"data": {"usage": usage}}).encode("utf-8")
            try:
                time.sleep(header_delay)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                time.sleep(body_delay)
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The expired client's worker was killed and reaped.

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/key", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


@pytest.fixture
def meter_worker(monkeypatch, tmp_path):
    roots = {
        "OUROBOROS_APP_ROOT": tmp_path,
        "OUROBOROS_REPO_DIR": tmp_path / "repo",
        "OUROBOROS_DATA_DIR": tmp_path / "data",
        "OUROBOROS_SETTINGS_PATH": tmp_path / "data" / "settings.json",
    }
    # Exercise CI's absent outer environment after safe module imports. The
    # fixture supplies the isolated roots whose worker inheritance we assert.
    for name in roots:
        monkeypatch.delenv(name, raising=False)
    for name, path in roots.items():
        monkeypatch.setenv(name, str(path))
    key = "synthetic-meter-key-never-in-argv"
    monkeypatch.setenv("TEST_METER_SECRET", key)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    real_popen = subprocess.Popen
    workers = []
    def capture(command, **kwargs):
        assert key not in " ".join(command)
        assert key not in kwargs["env"].values()
        assert command[1:3] == ["-I", "-S"]  # No editable finder or live imports.
        for name in roots:
            assert kwargs["env"][name] == os.environ[name]
        worker = real_popen(command, **kwargs)
        workers.append(worker)
        return worker
    monkeypatch.setattr(campaign.subprocess, "Popen", capture)
    yield key, workers
    assert len(workers) == 1
    assert workers[0].poll() is not None
    if os.name != "nt":
        with pytest.raises(ChildProcessError):
            os.waitpid(workers[0].pid, os.WNOHANG)


def test_delayed_headers_and_body_complete_inside_one_deadline(monkeypatch, meter_worker):
    key, workers = meter_worker
    with meter_endpoint(0.1, 0.15, 123.5) as (url, seen):
        monkeypatch.setattr(campaign, "_KEY_USAGE_URL", url)
        assert campaign.key_usage(key, timeout=1.5) == 123.5
        assert seen[0]["Authorization"] == "Bearer " + key
        assert seen[0]["Cache-Control"] == "no-cache"
        assert workers[0].returncode == 0


def test_headers_and_body_cannot_each_consume_the_full_timeout(monkeypatch, meter_worker):
    key, workers = meter_worker
    # Each delay is below 1.5 s; their sum is above it, reproducing the socket-timeout gap.
    with meter_endpoint(0.9, 0.9, 123.5) as (url, _seen):
        monkeypatch.setattr(campaign, "_KEY_USAGE_URL", url)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="wall-clock deadline") as caught:
            campaign.key_usage(key, timeout=1.5)
        elapsed = time.monotonic() - started
        assert elapsed < 1.75
        assert workers[0].poll() is not None
        assert key not in str(caught.value)


def test_meter_worker_error_never_exposes_the_stdin_credential(monkeypatch, meter_worker):
    key, _workers = meter_worker
    with meter_endpoint(0, 0, key) as (url, _seen):
        monkeypatch.setattr(campaign, "_KEY_USAGE_URL", url)
        with pytest.raises(RuntimeError) as caught:
            campaign.key_usage(key, timeout=1.5)
        assert key not in str(caught.value)
        assert "[redacted]" in str(caught.value)
