"""Supervisor selection and full send identity, without engines or model calls."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.model_slots import task_model_binding, task_processing_preference
from ouroboros.subagent_runtime import SubagentSelectionError
from ouroboros.supervisor_route import capture_cognitive_route, validate_cognitive_route
from tests.test_available_subagents_runtime import _api_row, _session_row, _settings, _snapshot


def cognition_row():
    row = _api_row("cognition", "claudexor::codex=gpt-5.6-sol", "high")
    row["route"]["credential_profile_id"] = "account-a"
    row["processing_preference"] = "economy"
    return row


def parent(**kwargs):
    return SimpleNamespace(**{
        "task_metadata": {}, "active_model": "claudexor::codex=gpt-5.6-sol",
        "active_effort": "xhigh", "active_use_local": False, **kwargs,
    })


def selected_task(ctx=None):
    settings = _settings(_session_row(), cognition_row())
    leaf = _snapshot(settings, "session-builder")
    route = capture_cognitive_route(ctx or parent(), leaf, settings, supervisor="cognition")
    return {"configured_subagent": leaf, "parent_cognitive_route": route}


def test_explicit_supervisor_freezes_account_processing_effort_and_own_role(monkeypatch):
    task = selected_task()
    route = task["parent_cognitive_route"]
    assert route["schema"] == 2
    assert route["source_role"] == "subagent:cognition"
    assert route["effort"] == "high" and route["chosen_by"] == "delegator"
    assert route["selected_supervisor_snapshot"]["selected_subagent_id"] == "cognition"
    # Neither the session's different account nor Main's settings can retarget it.
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"main": "changed-main"}))
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", _settings(cognition_row())["OUROBOROS_SUBAGENTS"])
    assert task_model_binding(task) == ("subagent:cognition", "account-a")
    assert task_processing_preference(task) == "economy"
    assert task_processing_preference(task, model_role="fallback:2") == "economy"
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", '{"fallback":["","","standard"]}')
    assert task_processing_preference(task, model_role="fallback:2") == "standard"


@pytest.mark.parametrize("role", ["light", "fallback:1", "consciousness"])
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("pin", ["", "chosen-account"])
def test_inheritance_captures_actual_role_and_live_choice_never_observed_auto(monkeypatch, role, live, pin):
    accounts = {"main": "wrong-main", "light": pin, "consciousness": pin, "fallback": ["wrong-first", pin]}
    settings = {**_settings(_session_row()), "OUROBOROS_MODEL_ACCOUNTS": json.dumps(accounts)}
    monkeypatch.setenv("OUROBOROS_MODEL_PROCESSING_PREFERENCES", '{"light":"economy","consciousness":"economy","fallback":["","economy"]}')
    overrides = {role: {"model": "claudexor::codex=owner-model", "use_local": False,
                        "model_account_override": pin}} if live else {}
    ctx = parent(context_fit_plan=SimpleNamespace(model_role=role, model_route={"credentialProfileId": "observed-auto"}),
                 model_wait_context=SimpleNamespace(overrides=overrides))
    leaf = _snapshot(settings, "session-builder")
    route = capture_cognitive_route(ctx, leaf, settings)
    assert route["model"] == ("claudexor::codex=owner-model" if live else ctx.active_model)
    assert route["source_role"] == role and route["credential_profile_id"] == pin
    assert route["processing_preference"] == "economy" and route["effort"] == "xhigh"
    assert route["chosen_by"] == ("parent_fallback" if role.startswith("fallback:") else "inherited")
    assert "selected_supervisor_snapshot" not in route
    task = {"configured_subagent": leaf, "parent_cognitive_route": route}
    assert task_model_binding(task) == ("subagent:session-builder", pin)
    assert task_processing_preference(task) == "economy"


def test_inheriting_an_existing_supervisor_keeps_frozen_intent(monkeypatch):
    task = selected_task()
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    settings = _settings(_session_row("next-leaf"))
    leaf = _snapshot(settings, "next-leaf")
    inherited = capture_cognitive_route(parent(task_metadata=task), leaf, settings)
    assert inherited["credential_profile_id"] == "account-a"
    assert inherited["processing_preference"] == "economy"
    assert inherited["source_role"] == "subagent:cognition"
    assert inherited["chosen_by"] == "inherited"


@pytest.mark.parametrize("model,local", [("openai::gpt-5.6-sol", False), ("local-model", True)])
def test_nonmanaged_parent_does_not_inherit_a_managed_account(model, local):
    settings = {**_settings(_session_row()), "OUROBOROS_MODEL_ACCOUNTS": '{"main":"managed-account"}'}
    route = capture_cognitive_route(parent(active_model=model, active_use_local=local),
                                    _snapshot(settings, "session-builder"), settings)
    assert route["model"] == model and route["use_local_model"] is local
    assert route["credential_profile_id"] == ""


@pytest.mark.parametrize("fault,code", [
    ("session", "supervisor_requires_api_model"), ("disabled", "supervisor_disabled"),
    ("credentials", "supervisor_credentials_unavailable"), ("api_child", "supervisor_inapplicable"),
    ("empty", "supervisor_selection_required"), ("null", "supervisor_selection_required"),
    ("number", "supervisor_selection_required"),
])
def test_selection_refusals_and_the_same_valid_row(fault, code):
    row = cognition_row()
    selector = "cognition"
    actor_id = "session-builder"
    if fault == "session":
        row = _session_row("cognition")
    elif fault == "disabled":
        row["enabled"] = False
    elif fault == "credentials":
        row = _api_row("cognition", "openai::gpt-5.6-sol")
    elif fault == "api_child":
        actor_id = "native"
    else:
        selector = {"empty": "", "null": None, "number": 17}.get(fault, selector)
    settings = _settings(_session_row(), _api_row("native"), row)
    with pytest.raises(SubagentSelectionError) as refused:
        capture_cognitive_route(parent(), _snapshot(settings, actor_id), settings, supervisor=selector)
    assert refused.value.code == code
    valid = selected_task()
    assert valid["parent_cognitive_route"]["chosen_by"] == "delegator"
    # No API-model supervisor row is needed when the selector is omitted.
    inherited_settings = _settings(_session_row())
    assert capture_cognitive_route(parent(), _snapshot(inherited_settings, "session-builder"), inherited_settings)["chosen_by"] == "inherited"


def test_credentialed_api_and_local_supervisors_are_eligible():
    for model, credentials, local in [("openai::gpt-5.6-sol", {"OPENAI_API_KEY": "test-key"}, False),
                                      ("fixture-model (local)", {"LOCAL_MODEL_SOURCE": "fixture-local"}, True)]:
        settings = {**_settings(_session_row(), _api_row("cognition", model)), **credentials}
        route = capture_cognitive_route(parent(), _snapshot(settings, "session-builder"), settings, supervisor="cognition")
        assert route["use_local_model"] is local
        assert route["model"] == ("fixture-model" if local else model)


@pytest.mark.parametrize("patch", [
    {"schema": 3}, {"use_local_model": "false"}, {"credential_profile_id": None},
    {"source_role": ""}, {"processing_preference": "invalid"}, {"chosen_by": "guessed"},
    {"selected_supervisor_snapshot": {}}, {"chosen_by": "inherited"},
])
def test_v2_integrity_guard_refuses_incomplete_carrier_and_accepts_complete(patch):
    route = selected_task()["parent_cognitive_route"]
    assert validate_cognitive_route(route) == route
    with pytest.raises(SubagentSelectionError, match="invalid_parent_cognitive_route"):
        validate_cognitive_route({**route, **patch})
    copy_of_route = validate_cognitive_route(route)
    copy_of_route["selected_supervisor_snapshot"]["route"]["target_id"] = "changed"
    assert route["selected_supervisor_snapshot"]["route"]["target_id"] != "changed"


def test_legacy_keeps_main_binding_and_native_actor_keeps_its_row(monkeypatch):
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    task = selected_task()
    for old in ({}, {"model": "legacy", "effort": "low", "use_local_model": False}):
        task["parent_cognitive_route"] = old
        assert validate_cognitive_route(old) == old
        assert task_model_binding(task) == ("main", None)
        assert task_processing_preference(task) == "fast"
    actor = _snapshot(_settings(cognition_row()), "cognition")
    assert task_model_binding({"configured_subagent": actor}) == ("subagent:cognition", "account-a")


def test_wait_card_persistence_cannot_convert_inherited_leaf_into_cognition():
    from ouroboros.model_slots import apply_model_role_override

    settings = _settings(_session_row(), cognition_row())
    before = copy.deepcopy(settings)
    options = {"model": "claudexor::codex=owner-choice", "credential_profile_id": "owner-account", "use_local": False}
    with pytest.raises(ValueError, match="inherited cognition"):
        apply_model_role_override(settings, role="subagent:session-builder", **options)
    changed = apply_model_role_override(settings, role="subagent:cognition", **options)
    assert settings == before
    rows = json.loads(changed["OUROBOROS_SUBAGENTS"])["items"]
    assert rows[0]["route"]["kind"] == "agent_session"
    assert rows[1]["route"]["target_id"] == options["model"]


def test_later_owner_switch_overrides_pin_and_role_without_rewriting_snapshot():
    task = selected_task()
    before = copy.deepcopy(task)
    override = {"subagent:cognition": {"model_account_override": "owner-account"}}
    assert task_model_binding(task, overrides=override) == ("subagent:cognition", "owner-account")
    assert task_model_binding(task, context_fit_plan=SimpleNamespace(model_role="light")) == ("light", None)
    assert task == before


def test_captured_empty_processing_does_not_inherit_changed_global(monkeypatch):
    task = selected_task()
    task["parent_cognitive_route"]["processing_preference"] = ""
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    assert task_processing_preference(task) == ""
    assert task_processing_preference(task, model_role="fallback:1") == ""
    assert task_processing_preference({}, model_role="main") == "fast"


def test_v2_cannot_smuggle_a_session_snapshot_into_cognition():
    task = selected_task()
    bad = {**task["parent_cognitive_route"], "selected_supervisor_snapshot": task["configured_subagent"]}
    with pytest.raises(SubagentSelectionError, match="invalid_parent_cognitive_route"):
        validate_cognitive_route(bad)
    assert validate_cognitive_route(task["parent_cognitive_route"])["chosen_by"] == "delegator"
