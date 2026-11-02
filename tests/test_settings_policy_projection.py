"""Owner Settings projection keeps configured and effective policy truthful."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from starlette.requests import Request


def test_setup_contract_exposes_cyber_pro_as_fourth_access_level():
    from ouroboros.settings_setup_contract import build_setup_contract

    modes = build_setup_contract("web")["runtimeModes"]
    assert [mode["value"] for mode in modes] == [
        "light", "advanced", "pro", "cyber_pro"
    ]
    cyber = modes[-1]
    assert cyber["label"] == "Cyber Pro"
    assert "Blocking" in cyber["copy"] and "Advisory" in cyber["copy"]
    # The choice is kept, but review findings do not block actions in Cyber Pro (#1736);
    # Panic and the owner's other controls still do, so the copy never says nothing blocks.
    assert "review findings do not block actions in Cyber Pro" in cyber["copy"]
    assert "nothing blocks" not in cyber["copy"] and "owner-controlled" not in cyber["copy"]


def test_policy_projection_distinguishes_restart_and_next_task(monkeypatch):
    import ouroboros.config as config
    import ouroboros.gateway.settings as gateway

    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(config, "get_safety_mode", lambda kind="owner": "full")
    monkeypatch.setattr(config, "normalize_runtime_mode", lambda value: str(value or "advanced"))
    monkeypatch.setattr(config, "normalize_safety_mode", lambda value: str(value or "full"))
    monkeypatch.setattr(gateway, "_has_started_agent_tasks", lambda: True)
    monkeypatch.setattr(
        "ouroboros.review_model_routes.get_review_enforcement",
        lambda: "advisory",
    )

    state = gateway._build_policy_state({
        "OUROBOROS_RUNTIME_MODE": "cyber_pro",
        "OUROBOROS_SAFETY_MODE": "light",
        "OUROBOROS_REVIEW_ENFORCEMENT": "blocking",
    })

    assert state["access"] == {
        "configured": "cyber_pro",
        "effective": "advanced",
        "current_process": "advanced",
        "next_task": "cyber_pro",
        "restart_required": True,
        "applies": "restart",
    }
    assert state["supervisor"]["current_process"] == "full"
    assert state["supervisor"]["next_task"] == "light"
    assert state["supervisor"]["pending"] is True
    assert state["supervisor"]["active_task_snapshot"] is True
    assert state["supervisor"]["applies"] == "next_task"
    assert state["review"]["current_process"] == "advisory"
    assert state["review"]["next_task"] == "blocking"
    assert state["review"]["pending"] is True
    assert state["review"]["active_task_snapshot"] is True
    assert state["review"]["applies"] == "next_task"
    assert state["running_task_snapshot"] is True


def test_settings_get_exposes_policy_state_in_existing_meta(monkeypatch):
    import ouroboros.gateway.settings as gateway

    settings = {
        "OUROBOROS_RUNTIME_MODE": "cyber_pro",
        "OUROBOROS_SAFETY_MODE": "full",
        "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
    }
    monkeypatch.setattr(gateway, "load_settings", lambda: dict(settings))
    monkeypatch.setattr(gateway, "apply_runtime_provider_defaults", lambda value: (value, False, []))
    monkeypatch.setattr(gateway, "_build_network_meta", lambda *_args: {})
    monkeypatch.setattr(gateway, "_port_file", lambda _request: SimpleNamespace(exists=lambda: False))
    monkeypatch.setattr(gateway, "_default_port", lambda _request: 8765)
    monkeypatch.setattr(gateway, "_has_started_agent_tasks", lambda: False)
    monkeypatch.setattr(gateway, "_build_policy_state", lambda _value: {
        "access": {"configured": "cyber_pro", "effective": "advanced", "current_process": "advanced", "next_task": "cyber_pro", "restart_required": True, "applies": "restart"},
        "supervisor": {"configured": "full", "effective": "full", "current_process": "full", "next_task": "full", "pending": False, "applies": "next_task", "active_task_snapshot": False},
        "review": {"configured": "advisory", "effective": "advisory", "current_process": "advisory", "next_task": "advisory", "pending": False, "applies": "next_task", "active_task_snapshot": False},
        "running_task_snapshot": False,
    })
    scope = {
        "type": "http", "method": "GET", "path": "/api/settings",
        "headers": [], "query_string": b"", "client": ("test", 1),
        "app": SimpleNamespace(state=SimpleNamespace()),
    }
    response = asyncio.run(gateway.api_settings_get(Request(scope)))
    payload = json.loads(response.body)
    assert payload["_meta"]["policy_state"]["access"]["restart_required"] is True
