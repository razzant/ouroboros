"""The live-server fixtures the browser smoke suites drive the real UI against.

Split out of ``tests/test_ui_smoke_playwright.py`` when that module was divided by theme;
the definitions are verbatim, so every sibling suite boots the same direct-mode server,
the same seeded drive and the same readiness waits it was written against.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from tests.fixtures_mock_llm import MockLLMServer


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def _wait_health(url: str, timeout_sec: int = 30) -> None:
    deadline = time.time() + timeout_sec
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=2) as resp:  # noqa: S310 - local test server
                if resp.status == 200:
                    return
        except Exception as exc:
            last = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {last}")

def _wait_supervisor_ready(url: str, timeout_sec: int = 45) -> None:
    """Wait past port readiness until the direct test runtime can serve history."""
    deadline = time.time() + timeout_sec
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/state", timeout=2) as resp:  # noqa: S310 - local test server
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("supervisor_ready") is True:
                    return
        except Exception as exc:
            last = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"server supervisor did not become ready: {last}")

@pytest.fixture()
def direct_server_with_data(tmp_path):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 to run browser UI smoke")
    with MockLLMServer() as llm:
        port = _free_port()
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        model = "openai-compatible::mock-model"
        (data_dir / "settings.json").write_text(
            json.dumps(
                {
                    "OPENAI_COMPATIBLE_API_KEY": "ui-smoke-key",
                    "OPENAI_COMPATIBLE_BASE_URL": llm.base_url,
                    "OUROBOROS_MODEL": model,
                    "OUROBOROS_MODEL_HEAVY": model,
                    "OUROBOROS_MODEL_LIGHT": model,
                    "OUROBOROS_MODEL_FALLBACKS": model,
                    # Every smoke case is single-task or deterministic log replay;
                    # a ten-process default pool adds only process churn and makes
                    # sequential browser history fetches flaky on shared hosts.
                    "OUROBOROS_MAX_WORKERS": 1,
                    "OUROBOROS_RUNTIME_MODE": "light",
                }
            ),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "OUROBOROS_APP_ROOT": str(tmp_path),
            "OUROBOROS_DATA_DIR": str(data_dir),
            "OUROBOROS_SETTINGS_PATH": str(data_dir / "settings.json"),
            "OUROBOROS_REPO_DIR": REPO_ROOT,
            "OUROBOROS_SERVER_HOST": "127.0.0.1",
            "OUROBOROS_SERVER_PORT": str(port),
            "OUROBOROS_HOST_SERVICE_PORT": str(port + 1),
            "OUROBOROS_NETWORK_PASSWORD": "ui-smoke-password",
        }
        url = f"http://127.0.0.1:{port}"
        active_proc = None

        def stop_server() -> None:
            nonlocal active_proc
            if active_proc is None or active_proc.poll() is not None:
                return
            from ouroboros.platform_layer import IS_WINDOWS, kill_process_tree

            # Windows terminate() is an immediate TerminateProcess, so the parent
            # can disappear before its worker tree and bypass the timeout cleanup.
            # taskkill /T must own that path from the start.
            if IS_WINDOWS:
                kill_process_tree(active_proc)
                active_proc.wait(timeout=5)
                active_proc = None
                return
            active_proc.terminate()
            try:
                active_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # A timed-out UI-smoke server still owns its worker pool. Killing
                # only the parent leaks ten orphan workers into later smoke tests,
                # producing suite-order history/card timeouts. The server starts in
                # its own process group below, so the shared cross-platform helper
                # can close the complete tree without touching pytest.
                kill_process_tree(active_proc)
                active_proc.wait(timeout=5)
            finally:
                active_proc = None

        def start_server() -> None:
            nonlocal active_proc
            from ouroboros.platform_layer import subprocess_new_group_kwargs

            active_proc = subprocess.Popen(
                [sys.executable, "server.py"],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **subprocess_new_group_kwargs(),
            )
            _wait_health(url)
            _wait_supervisor_ready(url)

        def restart_server() -> None:
            stop_server()
            start_server()

        try:
            start_server()
            yield {"url": url, "data_dir": data_dir, "restart_server": restart_server}
        finally:
            stop_server()

@pytest.fixture()
def direct_server(direct_server_with_data):
    return direct_server_with_data["url"]
