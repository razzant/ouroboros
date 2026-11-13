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
    assert cyber["copy"] == "Full say over itself and this computer. Asks for reviews when it judges."
    assert modes[2]["copy"] == "May change its protected core. Reads a review before each change."
    # Panic and the owner's other controls still stop it, so the copy never says nothing blocks.
    assert "Blocking" not in cyber["copy"] and "Advisory" not in cyber["copy"]
    assert "nothing blocks" not in cyber["copy"] and "owner-controlled" not in cyber["copy"]


def test_policy_projection_distinguishes_restart_and_next_task(monkeypatch):
    import ouroboros.config as config
    import ouroboros.gateway.settings as gateway

    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(config, "get_safety_mode", lambda kind="owner": "full")
    monkeypatch.setattr(config, "normalize_runtime_mode", lambda value: str(value or "advanced"))
    monkeypatch.setattr(config, "normalize_safety_mode", lambda value: str(value or "full"))
    monkeypatch.setattr(gateway, "_has_started_agent_tasks", lambda: True)

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
    assert "review" not in state, "the retired review-enforcement axis is never projected, whatever the document says"
    assert state["running_task_snapshot"] is True


def test_settings_get_exposes_policy_state_in_existing_meta(monkeypatch):
    import ouroboros.gateway.settings as gateway

    settings = {
        "OUROBOROS_RUNTIME_MODE": "cyber_pro",
        "OUROBOROS_SAFETY_MODE": "full",
    }
    monkeypatch.setattr(gateway, "settings_read_view", lambda: (dict(settings), {}))
    monkeypatch.setattr(gateway, "apply_runtime_provider_defaults", lambda value: (value, False, []))
    monkeypatch.setattr(gateway, "_build_network_meta", lambda *_args: {})
    monkeypatch.setattr(gateway, "_port_file", lambda _request: SimpleNamespace(exists=lambda: False))
    monkeypatch.setattr(gateway, "_default_port", lambda _request: 8765)
    monkeypatch.setattr(gateway, "_has_started_agent_tasks", lambda: False)
    monkeypatch.setattr(gateway, "_build_policy_state", lambda _value: {
        "access": {"configured": "cyber_pro", "effective": "advanced", "current_process": "advanced", "next_task": "cyber_pro", "restart_required": True, "applies": "restart"},
        "supervisor": {"configured": "full", "effective": "full", "current_process": "full", "next_task": "full", "pending": False, "applies": "next_task", "active_task_snapshot": False},
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


def test_a_managed_deployment_projects_what_its_policy_fixes_and_no_secret(tmp_path, monkeypatch):
    import hashlib
    import ouroboros.gateway.settings as gateway
    from ouroboros import config, deployment_policy

    policy = tmp_path / "admin" / "deployment-policy.json"
    policy.parent.mkdir()
    policy.write_text(json.dumps({"version": 1, "access": {"max": "advanced"}, "safety": {
        "min_coverage": "full", "checker": {"model": "anthropic@safety::claude-opus-5-5"}},
        "rules": "Keep customer records internal."}), encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-never-projected")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setattr(gateway, "_has_started_agent_tasks", lambda: False)
    config.reset_runtime_mode_baseline_for_tests()
    deployment_policy.reset_for_tests()
    try:
        unmanaged = gateway._build_policy_state({"OUROBOROS_RUNTIME_MODE": "cyber_pro"})
        monkeypatch.setenv(deployment_policy.POLICY_ENV, str(policy))
        deployment_policy.reset_for_tests()
        managed = gateway._build_policy_state({"OUROBOROS_RUNTIME_MODE": "cyber_pro"})
    finally:
        deployment_policy.reset_for_tests()
    assert "deployment" not in unmanaged and unmanaged["access"]["effective"] == "cyber_pro"
    assert managed["deployment"] == {
        "source": "deployment_policy", "version": 1,
        "digest": hashlib.sha256(policy.read_bytes()).hexdigest()[:16],
        "access_max": "advanced", "min_coverage": "full",
        "checker": {"model": "anthropic@safety::claude-opus-5-5", "account": "", "fallbacks": [],
                    "connections": ["anthropic@safety"], "fields": [], "local": False},
        "allowed_exits": ["repeat_after_dangerous", "release_after_dangerous"],  # checker: no unjudged release
        "organization_rules": True, "extensions": None,
    }
    assert (managed["access"]["effective"], managed["supervisor"]["effective"]) == ("advanced", "full")
    assert "sk-ant" not in json.dumps(managed) and "Keep customer records" not in json.dumps(managed)
