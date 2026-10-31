"""Exact actor inspection: task snapshot intent and cached engine facts only."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ouroboros.claudexor_observations import quota_availability, read_quota
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot
from ouroboros.subagent_inspection import inspect_subagent


def _snapshot(target="claude=claude-fable-5-1", *, kind="agent_session", enabled=True):
    return task_settings_snapshot({"OUROBOROS_SUBAGENTS": json.dumps({
        "enabled": True, "items": [{"subagent_id": "builder", "enabled": enabled,
            "recommended_use": "fixture", **({"access": "readonly"} if kind == "agent_session" else {}),
            "route": {"kind": kind, "target_id": target,
                      "credential_profile_id": "chosen" if kind == "agent_session" else ""}}],
    })}, {})


# A measured window for every model and kind: an engine "available" is a claim
# about this request only with one (OB1851); its ratio sets no headroom policy.
MEASURED = {"used_ratio": 0.2, "resets_at": "2099-01-01T00:00:00Z"}


def _quota(*, scope=None, state="exhausted", fresh="fresh"):
    return {"snapshots": [{"subject": {"harness": "claude", "subject_id": "chosen"},
        "freshness": fresh, "constraints": [dict(MEASURED)],
        "availability": {"state": state, "resets_at": "2099-01-01T00:00:00Z"}}],
        "absences": [], "resources": [],
        "availability_scope": scope or {"model": "claude-fable-5-1", "requestKind": "text", "basis": "quota_block"}}


def _resource(profile="chosen", harness="cursor"):
    """AccountResourceSnapshot / AccountTarget, engine schema account-resources.ts:91.

    Facets deliberately keep source, units, scope, period and null knowledge.
    Resource identity is NOT ControlQuotaSnapshot.subject.
    """
    golden = json.loads((Path(__file__).parent / "fixtures" /
                         "claudexor_resources_freshness_cursor.json").read_text(encoding="utf-8"))
    row = golden["resources"][0]
    row["target"] = {"harness": harness, "profile_id": profile}
    return row


@pytest.mark.parametrize("scope", [None, {}, {"model": "fable", "requestKind": "text", "basis": "quota_block"},
                                   {"model": "claude-fable-5-1", "requestKind": "image"}])
def test_sent_selector_is_not_honored_selector(scope):
    value = _quota()
    value["availability_scope"] = scope
    assert quota_availability(value, "claude", model="claude-fable-5-1")["state"] == "unknown"


def test_engine_verdict_not_raw_ratio_or_money_controls_availability():
    value = _quota(state="available")
    value["snapshots"][0]["constraints"] = [{"used_ratio": 1, "applies_to_models": ["fable"]}]
    value["resources"] = [_resource(harness="claude")]
    # A model-scoped window alone witnesses no applicable measurement.
    assert quota_availability(value, "claude", model="claude-fable-5-1")["state"] == "unknown"
    value["snapshots"][0]["constraints"].append(dict(MEASURED))
    assert quota_availability(value, "claude", model="claude-fable-5-1")["state"] == "available"
    value["snapshots"][0]["freshness"] = "stale"
    assert quota_availability(value, "claude", model="claude-fable-5-1")["state"] == "unknown"


def test_pin_and_absences_are_subject_exact():
    value = _quota()
    sibling = deepcopy(value["snapshots"][0])
    sibling["subject"]["subject_id"] = "healthy"
    sibling["availability"]["state"] = "available"
    value["snapshots"].append(sibling)
    assert quota_availability(value, "claude", model="claude-fable-5-1")["state"] == "available"
    assert quota_availability(value, "claude", model="claude-fable-5-1", profile="chosen")["state"] == "exhausted"
    value["absences"] = [{"subject": {"harness": "claude", "subject_id": "chosen"}}]
    # The pin's own dated block outlives its refresh gap; its gap still blocks a positive claim.
    assert quota_availability(value, "claude", model="claude-fable-5-1", profile="chosen")["state"] == "exhausted"
    value["absences"] = [{"subject": {"harness": "claude", "subject_id": "healthy"}}]
    assert quota_availability(value, "claude", model="claude-fable-5-1", profile="healthy")["state"] == "unknown"
    assert quota_availability(value, "claude", model="claude-fable-5-1")["state"] == "unknown"


@pytest.mark.parametrize("failure", ["timeout", "malformed", 403, 500])
def test_no_legacy_repeat_for_unknown_or_non_selector_refusal(failure):
    class Gateway:
        calls = []
        def quota_state(self, **kwargs):
            self.calls.append(kwargs)
            if failure == "malformed":
                return {"snapshots": "broken"}
            raise ClaudexorUnavailable("daemon_unreachable", "fixture", status_code=failure if isinstance(failure, int) else 0)
    gateway = Gateway()
    with pytest.raises(ClaudexorUnavailable):
        read_quota(gateway, model="exact")
    assert gateway.calls == [{"view": "resources_freshness", "model": "exact", "request_kind": "text"}]


def test_definite_400_allows_legacy_views_but_not_exact_claim():
    class Gateway:
        calls = []
        def quota_state(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs:
                raise ClaudexorUnavailable("http_error", "fixture", status_code=400)
            return {"snapshots": [], "absences": []}
    gateway = Gateway()
    value = read_quota(gateway, model="exact")
    assert gateway.calls == [
        {"view": "resources_freshness", "model": "exact", "request_kind": "text"},
        {"view": "constraint_freshness"}, {},
    ]
    assert quota_availability(value, "claude", model="exact")["state"] == "unknown"


def test_disabled_configured_row_is_not_a_missing_live_observation(monkeypatch):
    from ouroboros import config
    from ouroboros.gateway import claudexor_passive
    calls = []
    def passive(**scope):
        calls.append(scope)
        return {"reads": {"quota": "ok", "accounts": "ok"}, "read_errors": {}, "timings_ms": {},
                "quota": _quota()["snapshots"], "quota_absences": [], "resources": [],
                "availability_scope": _quota()["availability_scope"], "profiles": {"profiles": []}}
    monkeypatch.setattr(claudexor_passive, "_quota_payload", passive)
    monkeypatch.setattr(config, "load_settings", lambda: {"OUROBOROS_SUBAGENTS": "corrupt-live-change"})
    with task_settings_scope(_snapshot(enabled=False)):
        result = inspect_subagent("builder")
    assert result["configured"]["enabled"] is False
    assert result["configured"]["access"] == "readonly"
    assert result["availability"]["state"] == "exhausted"
    assert result["execution_guaranteed"] is False
    assert calls == [{"model": "claude-fable-5-1", "request_kind": "text"}]


def test_api_actor_without_managed_route_never_reads_engine(monkeypatch):
    from ouroboros.gateway import claudexor_passive
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda **kw: pytest.fail("direct API has no engine quota"))
    with task_settings_scope(_snapshot("openai/gpt-example", kind="api_model")):
        result = inspect_subagent("builder")
    assert result["availability"]["state"] == "unknown"
    assert "engine" not in result


def test_public_registry_consumer_works_without_any_quota_skill(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.gateway import claudexor_passive
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda **kw: pytest.fail("direct API"))
    registry = ToolRegistry(tmp_path / "repo", tmp_path / "data")
    with task_settings_scope(_snapshot("openai/gpt-example", kind="api_model")):
        result = registry.execute_result("inspect_subagent", {"subagent_id": "builder", "request_kind": "image"})
    assert result.status == "ok", result.text
    value = json.loads(result.text)
    assert value["configured"]["route"]["target_id"] == "openai/gpt-example"
    assert value["request_kind"] == "image" and not value["execution_guaranteed"]


def test_registered_cursor_inspector_keeps_real_resource_schema_and_pin(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.gateway import claudexor_passive
    chosen = _resource()
    payload = {"reads": {"accounts": "ok", "quota": "ok"}, "read_errors": {}, "timings_ms": {},
        "profiles": {"profiles": [], "harnessAccounts": []}, "quota": [], "quota_absences": [],
        "availability_scope": {"model": "auto", "requestKind": "text", "basis": "quota_block"},
        "resources": [_resource("healthy-sibling"), chosen, _resource(harness="claude")]}
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda **kw: deepcopy(payload))
    registry = ToolRegistry(tmp_path / "repo", tmp_path / "data")
    with task_settings_scope(_snapshot("cursor=auto")):
        answer = registry.execute_result("inspect_subagent", {"subagent_id": "builder"})
    assert answer.status == "ok", answer.text
    body = json.loads(answer.text)
    assert body["engine"]["resources"] == [chosen]  # every nested base fact is unchanged
    assert body["availability"]["state"] == "unknown"  # money is not quota/admission
    assert body["configured"]["access"] == "readonly"


def test_resource_target_is_not_a_nullable_quota_subject_alias():
    from ouroboros.claudexor_observations import resource_rows
    explicit_default = _resource("cursor-default")
    null = _resource()
    null["target"]["profile_id"] = None
    rows = [_resource(), explicit_default, null, {"subject": {"harness": "cursor", "subject_id": None}}]
    assert resource_rows(rows, "cursor", "cursor-default") == [explicit_default]
    assert resource_rows(rows, "cursor", "chosen") == [rows[0]]
    assert resource_rows(rows, "cursor", "missing") == []
    assert resource_rows(rows, "cursor") == rows[:2]


def test_complete_engine_combined_golden_passes_through_unchanged():
    """Generated by current engine parseCursorResources/mergeResources/quotaControlServices,
    validated by its ControlAccountResourcesFreshnessResponse, not copied host assumptions.
    """
    golden = json.loads((Path(__file__).parent / "fixtures" /
                         "claudexor_resources_freshness_cursor.json").read_text(encoding="utf-8"))
    class Gateway:
        def quota_state(self, **scope):
            assert scope == {"view": "resources_freshness", "model": "auto", "request_kind": "text"}
            return deepcopy(golden)
    assert read_quota(Gateway(), model="auto") == golden


# -- #244: a quota subject is matched exactly; `<harness>-default` borrows no null window ----------

def _pinned(pin):
    return task_settings_snapshot({"OUROBOROS_SUBAGENTS": json.dumps({
        "enabled": True, "items": [{"subagent_id": "builder", "enabled": True, "recommended_use": "fixture",
            "access": "readonly", "route": {"kind": "agent_session", "target_id": "claude=claude-fable-5-1",
                                            "credential_profile_id": pin}}],
    })}, {})


def _window(subject, state):
    return {"subject": {"harness": "claude", "subject_id": subject}, "freshness": "fresh", "constraints": [dict(MEASURED)],
            "availability": {"state": state, "resets_at": "2099-01-01T00:00:00Z"}}


def _unified_payload(*windows):
    return {"reads": {"quota": "ok", "accounts": "ok"}, "read_errors": {}, "timings_ms": {},
            "quota": list(windows), "quota_absences": [], "resources": [], "unified_accounts": True,
            "availability_scope": {"model": "claude-fable-5-1", "requestKind": "text", "basis": "quota_block"},
            "profiles": {"profiles": [{"profile": {"harness_id": "claude", "profile_id": "claude-default",
                                                   "enabled": True}}],
                         "harnessAccounts": [], "accountPools": []}}


def test_an_unprofiled_window_never_answers_for_a_pinned_default(monkeypatch):
    """A current explicit API-key route writes a fresh NULL-subject window; the named
    `claude-default` profile has none of its own. The pin stays unknown."""
    from ouroboros.gateway import claudexor_passive
    payload = _unified_payload(_window(None, "exhausted"))
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda **kw: deepcopy(payload))
    with task_settings_scope(_pinned("claude-default")):
        result = inspect_subagent("builder")
    assert result["availability"]["state"] == "unknown"
    assert result["availability"]["reason"] == "quota_evidence_missing"
    assert result["engine"]["snapshots"] == []
    assert "subject_aliases" not in result["engine"] and "native_accounts" not in result["engine"]


@pytest.mark.parametrize("pin,state,subjects", [
    ("claude-default", "exhausted", ["claude-default"]),
    ("claude-default-2", "available", ["claude-default-2"]),
    ("", "available", [None, "claude-default", "claude-default-2"]),
], ids=["exact-default", "reused-suffix-id", "auto-keeps-unprofiled"])
def test_exact_subjects_still_answer_and_auto_keeps_unprofiled_windows(monkeypatch, pin, state, subjects):
    from ouroboros.gateway import claudexor_passive
    payload = _unified_payload(_window(None, "exhausted"), _window("claude-default", "exhausted"),
                               _window("claude-default-2", "available"))
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda **kw: deepcopy(payload))
    with task_settings_scope(_pinned(pin)):
        result = inspect_subagent("builder")
    assert result["availability"]["state"] == state
    assert [row["subject"]["subject_id"] for row in result["engine"]["snapshots"]] == subjects
