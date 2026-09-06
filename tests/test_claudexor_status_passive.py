"""Account status reads metadata; only launch preparation probes executables."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient
from starlette.routing import Route

from ouroboros import claudexor_daemon as owned
from ouroboros import claudexor_runtime as runtime
from ouroboros.gateway.claudexor_accounts import api_claudexor_status


@pytest.mark.parametrize("state", ["not_provisioned", "stale"])
def test_status_get_never_prepares_or_probes_a_launch(monkeypatch, tmp_path, state):
    """Installed metadata remains visible without even the first Node probe."""
    manager = runtime.ClaudexorRuntimeManager()
    pin = manager.pin
    assert pin is not None
    metadata = {
        "version": pin.version, "build_sha": pin.build_sha,
        "node_version": pin.node_version, "archive_source": "cache",
    }
    monkeypatch.setattr(manager, "_managed_metadata", lambda: dict(metadata))
    monkeypatch.setattr(manager, "_install_in_progress", lambda: False)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    monkeypatch.setattr(runtime, "managed_runtime_root", lambda: tmp_path / "runtime")
    daemon = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(daemon, "_classify_liveness", lambda: (None, state, ""))
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: daemon)
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "claudexor")
    monkeypatch.setattr(owned, "verify_owned_home", lambda: "")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("passive status must not prepare a command or start a process")

    monkeypatch.setattr(manager, "resolve_command", forbidden)
    monkeypatch.setattr(manager, "ensure", forbidden)
    monkeypatch.setattr(daemon, "ensure_running", forbidden)
    monkeypatch.setattr(owned.subprocess, "Popen", forbidden)
    app = Starlette(routes=[Route("/api/claudexor/status", api_claudexor_status)])
    with TestClient(app) as client:
        for _ in range(2):
            response = client.get("/api/claudexor/status?include=models")
            assert response.status_code == 200
            payload = response.json()
            assert payload["daemon"]["state"] == state
            assert payload["daemon"]["runtime"]["state"] == "ready"
            assert payload["daemon"]["runtime"]["node_version"] == pin.node_version
            assert "binary" not in payload["daemon"]
            assert payload["reads"] == dict.fromkeys(("catalog", "accounts", "quota"), "not_read")


def test_launch_preparation_still_probes_the_exact_node(monkeypatch, tmp_path):
    manager = runtime.ClaudexorRuntimeManager()
    pin = manager.pin
    assert pin is not None
    monkeypatch.delenv("OUROBOROS_CLAUDEXOR_BIN", raising=False)
    monkeypatch.setattr(manager, "_managed_metadata", lambda: {"version": pin.version})
    monkeypatch.setattr(runtime, "managed_runtime_root", lambda: tmp_path / "runtime")
    calls = []

    def resolve_node(requested_pin):
        assert requested_pin is pin
        calls.append("node")
        return str(tmp_path / "node")

    monkeypatch.setattr(manager, "_resolve_node", resolve_node)
    monkeypatch.setattr(manager, "_probe", lambda command, requested_pin: calls.append("engine"))
    assert manager.ensure()[0] == str(tmp_path / "node")
    assert calls == ["node", "engine"]
