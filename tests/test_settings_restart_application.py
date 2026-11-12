"""Saved settings are compared with actual component startup/health facts."""

from ouroboros.local_model import RUNTIME_PROBE
import asyncio
import json
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from starlette.requests import Request

from ouroboros import config, local_model, server_process
from ouroboros.gateway import models, settings
from ouroboros.server_entrypoint import bound_service_socket


@pytest.fixture
def applied(monkeypatch):
    monkeypatch.setattr(server_process, "_applied_restart_settings", {})
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    manager = local_model.LocalModelManager()
    monkeypatch.setattr(local_model, "_manager", manager)
    saved = dict(config.SETTINGS_DEFAULTS)
    server_process.record_applied_restart_settings({key: saved[key] for key in (
        "OUROBOROS_SERVER_HOST", "OUROBOROS_HOST_SERVICE_PORT",
        "OUROBOROS_SKILLS_REPO_PATH", "OUROBOROS_UI_DIR",  # the last: server_web.ServedUI.apply
    )}, server_host_source="settings")
    return saved, manager


def test_server_pending_truth_survives_reload_revert_and_environment_save(applied, monkeypatch):
    saved, _manager = applied
    for key, value in {
        "OUROBOROS_RUNTIME_MODE": "pro",
        "OUROBOROS_SERVER_HOST": "0.0.0.0", "OUROBOROS_HOST_SERVICE_PORT": 9123,
        "OUROBOROS_SKILLS_REPO_PATH": "/example/skills", "OUROBOROS_UI_DIR": "none",
    }.items():
        changed = {**saved, key: value}
        monkeypatch.setenv(key, str(value))  # a Save is not application
        state = settings._build_restart_state(changed)
        assert state["restart_keys"] == [key]
        assert settings._build_restart_state({**changed, "TOTAL_BUDGET": 300}) == state
        assert settings._build_restart_state(saved)["restart_required"] is False


def test_missing_component_facts_stay_unknown(applied, monkeypatch):
    saved, _manager = applied
    monkeypatch.setattr(server_process, "_applied_restart_settings", {})
    state = settings._build_restart_state(saved)
    assert len(state["unknown_keys"]) == 4  # the owner's worker cap is read live on every pass: never a restart key
    assert "not reported" in state["summary"]
    changed = {**saved, "OUROBOROS_MAX_WORKERS": 19}
    assert "OUROBOROS_MAX_WORKERS" not in settings._build_restart_state(changed)["restart_keys"]


def test_socket_owners_publish_only_successfully_bound_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(server_process, "_applied_restart_settings", {})
    with bound_service_socket(tmp_path, "host_service", "127.0.0.1", 0) as sock:
        port = sock.getsockname()[1]
        assert server_process.applied_restart_settings()["OUROBOROS_HOST_SERVICE_PORT"] == port
        # Exercise failed-bind publication without OS-specific port reuse rules.
        with patch("socket.socket.bind", side_effect=OSError("fixture bind refused")):
            with pytest.raises(OSError, match="fixture bind refused"):
                with bound_service_socket(tmp_path, "main", "127.0.0.1", port):
                    pass
        assert "OUROBOROS_SERVER_HOST" not in server_process.applied_restart_settings()


def test_every_local_setting_is_compared_only_after_confirmed_health(applied):
    saved, manager = applied
    saved.update(LOCAL_MODEL_SOURCE="fixture/model", LOCAL_MODEL_FILENAME="fixture.gguf")
    manager._launch_settings = local_model.local_model_settings(saved)
    manager._status = "loading"
    assert manager.settings_application(saved)["pending_keys"] == []
    assert "not applied" in manager.settings_application(saved)["summary"]
    manager._status = "ready"
    manager._applied_settings = dict(manager._launch_settings)
    for key, value in {
        "LOCAL_MODEL_SOURCE": "another/model", "LOCAL_MODEL_FILENAME": "other.gguf",
        "LOCAL_MODEL_PORT": 9345, "LOCAL_MODEL_N_GPU_LAYERS": 17,
        "LOCAL_MODEL_CONTEXT_LENGTH": 32768, "LOCAL_MODEL_CHAT_FORMAT": "other-format",
    }.items():
        state = settings._build_restart_state({**saved, key: value})
        assert state["restart_required"] is False
        assert state["local_model"]["pending_keys"] == [key]
    assert manager.settings_application({**saved, "LOCAL_MODEL_CONTEXT_LENGTH": 0})["pending_keys"] == []
    manager._status = "error"
    assert "not applied" in manager.settings_application(saved)["summary"]
    assert manager.settings_application(saved)["status"] == "error"


def _request(body):
    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
    return Request({"type": "http", "method": "POST", "path": "/api/local-model/start",
                    "headers": [], "app": SimpleNamespace(state=SimpleNamespace())}, receive)


@pytest.fixture
def healthy_model_process(tmp_path, monkeypatch):
    """The real manager starts a lightweight /v1/models child, never downloads a model."""
    manager = local_model.LocalModelManager()
    monkeypatch.setattr(local_model, "_manager", manager)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    script = tmp_path / "model_server.py"
    script.write_text(
        'import http.server,json,socket,socketserver,sys\n'
        'def no_reverse_dns(*args):\n'
        ' raise AssertionError("loopback fixture must not resolve hostnames")\n'
        'socket.getfqdn=no_reverse_dns\n'
        'class LoopbackHTTPServer(http.server.HTTPServer):\n'
        ' def server_bind(self):\n'
        '  socketserver.TCPServer.server_bind(self)\n'
        '  self.server_name,self.server_port=self.server_address[:2]\n'
        'class Handler(http.server.BaseHTTPRequestHandler):\n'
        ' def do_GET(self):\n'
        '  body=json.dumps({"data":[{"id":"fixture","context_window":9999}]}).encode()\n'
        '  self.send_response(200);self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)\n'
        ' def log_message(self,*args):pass\n'
        'LoopbackHTTPServer(("127.0.0.1",int(sys.argv[1])),Handler).serve_forever()\n', encoding="utf-8")
    original_popen, original_run = subprocess.Popen, subprocess.run
    commands = []

    def popen(command, **kwargs):
        if "ouroboros.local_model_server" in command:
            commands.append(list(command))
            command = [sys.executable, str(script), str(command[command.index("--port") + 1])]
        return original_popen(command, **kwargs)

    def run(command, **kwargs):
        if command[-1] == RUNTIME_PROBE:
            return subprocess.CompletedProcess(command, 0, "", "")
        return original_run(command, **kwargs)

    settled = threading.Event()
    original_health = manager._wait_for_healthy

    def health():
        try:
            original_health(timeout=8)
        finally:
            settled.set()

    monkeypatch.setattr(local_model.subprocess, "Popen", popen)
    monkeypatch.setattr(local_model.subprocess, "run", run)
    monkeypatch.setattr(manager, "_wait_for_healthy", health)
    monkeypatch.setattr(manager, "download_model", lambda source, filename: str(tmp_path / filename))
    try:
        yield manager, settled, commands
    finally:
        manager.stop_server()


def test_actual_local_start_health_and_restart_clear_saved_mismatch(healthy_model_process, monkeypatch, free_tcp_port):
    manager, settled, commands = healthy_model_process
    saved = {**config.SETTINGS_DEFAULTS, "LOCAL_MODEL_SOURCE": "fixture/model",
             "LOCAL_MODEL_FILENAME": "fixture.gguf", "LOCAL_MODEL_PORT": free_tcp_port,
             "LOCAL_MODEL_CONTEXT_LENGTH": 0, "LOCAL_MODEL_N_GPU_LAYERS": 3}
    monkeypatch.setattr(models, "load_settings", lambda: saved)
    body = {"source": "fixture/model", "filename": "fixture.gguf", "port": free_tcp_port,
            "n_ctx": 0, "n_gpu_layers": 3, "chat_format": ""}
    response = asyncio.run(models.api_local_model_start(_request(body)))
    assert response.status_code == 200
    assert settled.wait(10)
    assert manager.is_running, manager.status_dict()
    # 0 starts the runtime with the factory window.
    assert commands[-1][commands[-1].index("--n_ctx") + 1] == str(config.SETTINGS_DEFAULTS["LOCAL_MODEL_CONTEXT_LENGTH"])
    assert manager.settings_application(saved)["pending_keys"] == []
    saved["LOCAL_MODEL_CONTEXT_LENGTH"] = 8192
    assert manager.settings_application(saved)["pending_keys"] == ["LOCAL_MODEL_CONTEXT_LENGTH"]
    asyncio.run(models.api_local_model_stop(_request({})))
    assert manager.settings_application(saved)["status"] == "offline"
    settled.clear()
    response = asyncio.run(models.api_local_model_start(_request({**body, "n_ctx": 8192})))
    assert response.status_code == 200
    assert settled.wait(10)
    status = json.loads(asyncio.run(models.api_local_model_status(_request({}))).body)
    assert status["status"] == "ready"
    assert status["settings_application"]["pending_keys"] == []
    # Provider training metadata does not replace the effective launch window.
    assert status["context_length"] == 9999
    assert manager._applied_settings["LOCAL_MODEL_CONTEXT_LENGTH"] == 8192
