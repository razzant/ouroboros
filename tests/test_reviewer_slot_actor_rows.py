"""Configured-subagent references on reviewer rows (generic-actor bridge).

A reviewer row may reference an ``OUROBOROS_SUBAGENTS`` roster row instead of
carrying an inline route. Resolution happens once at load/admission from the
APPLIED env; the resolved slot carries the actor id as identity/provenance and
the roster row's execution facts. An api_model actor is the RETRIEVES class
(bounded native tool rounds) and must never enter the assembled-packet plane
(the pack-assembly predicate); its roster model id still projects into the
legacy comma key, which no review surface reads once the structured key exists.
"""

import json

import pytest

from ouroboros.reviewer_slot_config import (
    REVIEWER_SLOTS_ENV,
    commit_triad_delivery,
    load_reviewer_slot_config,
    parse_reviewer_slots,
    project_reviewer_slots_into_env,
    structured_scope_review_slots,
)

_ROSTER = {
    "enabled": True,
    "items": [
        {
            "subagent_id": "api-critic",
            "name": "API critic",
            "recommended_use": "Exact recursive API reviewer.",
            "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-terra"},
            "effort": "medium",
        },
        {
            "subagent_id": "session-critic",
            "name": "Session critic",
            "recommended_use": "Subscription reviewer.",
            "route": {
                "kind": "agent_session",
                "target_id": "codex=gpt-5.6-sol",
                "credential_profile_id": "profile-1",
            },
            "effort": "high",
        },
    ],
}


def _payload(triad_rows, scope_rows=None):
    return json.dumps({
        "triad": triad_rows,
        "scope": scope_rows or [
            {"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-terra"}},
        ],
    })


@pytest.fixture()
def roster_env(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(_ROSTER))
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODELS",
                "OUROBOROS_SCOPE_REVIEW_MODEL"):
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch


def test_session_actor_row_resolves_from_roster(roster_env):
    roster_env.setenv(REVIEWER_SLOTS_ENV, _payload(
        [{"slot_id": "t1", "subagent_id": "session-critic"}]))
    row = load_reviewer_slot_config().triad[0]
    assert row.subagent_id == "session-critic"
    assert row.is_session and row.retrieves and not row.native_retrieval
    assert row.target_id == "codex=gpt-5.6-sol"
    assert row.session_target == "codex=gpt-5.6-sol"
    assert row.profile_id == "profile-1"
    # Roster effort applies when the row has no explicit one.
    assert row.effort == "high"


def test_api_actor_row_is_native_retrieval(roster_env):
    roster_env.setenv(REVIEWER_SLOTS_ENV, _payload(
        [{"slot_id": "t1", "subagent_id": "api-critic"}]))
    row = load_reviewer_slot_config().triad[0]
    assert row.subagent_id == "api-critic"
    assert row.kind == "api_chat"  # wire vocabulary stays closed
    assert row.native_retrieval and row.retrieves and not row.is_session
    assert row.target_id == "openai/gpt-5.6-terra"
    assert row.effort == "medium"


def test_explicit_row_effort_outranks_roster_effort(roster_env):
    roster_env.setenv(REVIEWER_SLOTS_ENV, _payload(
        [{"slot_id": "t1", "subagent_id": "api-critic", "effort": "xhigh"}]))
    assert load_reviewer_slot_config().triad[0].effort == "xhigh"


def test_unknown_subagent_id_refuses_typed(roster_env):
    with pytest.raises(ValueError, match="unknown_subagent_id"):
        parse_reviewer_slots(_payload([{"slot_id": "t1", "subagent_id": "ghost"}]))


def test_route_and_subagent_id_are_mutually_exclusive(roster_env):
    with pytest.raises(ValueError, match="either route or subagent_id"):
        parse_reviewer_slots(_payload([{
            "slot_id": "t1", "subagent_id": "api-critic",
            "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-terra"},
        }]))


def test_empty_subagent_id_refuses(roster_env):
    with pytest.raises(ValueError, match="subagent_id"):
        parse_reviewer_slots(_payload([{"slot_id": "t1", "subagent_id": "  "}]))


def test_api_rows_of_both_forms_project_their_model_ids_into_the_legacy_key(roster_env):
    """The legacy comma key is a projection of api MODEL IDS for legacy readers
    (external review tooling, benchmark manifests) — an actor row's roster model
    id is one, a session row's `harness[=model]` target is not. No review surface
    reads the key while the structured key exists (owner R2 retired the acceptance
    pin that used to filter actor rows out of it)."""
    roster_env.setenv(REVIEWER_SLOTS_ENV, _payload([
        {"slot_id": "t1", "subagent_id": "api-critic"},
        {"slot_id": "t2", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.5"}},
        {"slot_id": "t3", "subagent_id": "session-critic"},
    ]))
    project_reviewer_slots_into_env()
    import os

    assert os.environ["OUROBOROS_REVIEW_MODELS"] == "openai/gpt-5.6-terra,openai/gpt-5.5"


def test_actor_and_session_triad_reaches_acceptance_as_configured(roster_env):
    """A triad of a native-retrieving actor and a session row IS the acceptance
    panel (R0/R2): no API-default substitution, no disclosure of one, both
    rows carried with their actor binding and pin."""
    from ouroboros.reviewer_slot_config import triad_delivery_slots

    roster_env.setenv(REVIEWER_SLOTS_ENV, _payload([
        {"slot_id": "t1", "subagent_id": "api-critic"},
        {"slot_id": "t2", "subagent_id": "session-critic"},
    ]))
    slots = triad_delivery_slots(role_hint="task acceptance")
    assert [slot.slot_id for slot in slots] == ["t1", "t2"]
    assert slots[0].native_retrieval and slots[0].subagent_id == "api-critic"
    assert slots[1].route.value == "agent_session" and slots[1].session_profile == "profile-1"
    assert all(slot.retrieves for slot in slots)


def test_commit_triad_delivery_carries_actor_vector(roster_env):
    roster_env.setenv(REVIEWER_SLOTS_ENV, _payload([
        {"slot_id": "t1", "subagent_id": "api-critic"},
        {"slot_id": "t2", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.5"}},
    ]))
    plan = commit_triad_delivery()
    assert plan["subagent_ids"] == ["api-critic", ""]


def test_scope_actor_slot_reaches_review_slot(roster_env):
    roster_env.setenv(REVIEWER_SLOTS_ENV, _payload(
        [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.5"}}],
        scope_rows=[{"slot_id": "s1", "subagent_id": "api-critic"}],
    ))
    slots = structured_scope_review_slots()
    assert slots is not None and len(slots) == 1
    assert slots[0].subagent_id == "api-critic"
    assert slots[0].native_retrieval and slots[0].retrieves


def test_actor_binding_is_attempt_identity(roster_env):
    """A changed actor reference mints a new custody attempt key (#285 class)."""
    from types import SimpleNamespace

    from ouroboros.review_custody import _attempt_key
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    request = SimpleNamespace(retry_key="", slot_messages={}, surface="multi_model_review",
                              task_id="t", call_type="multi_model_review")
    base = dict(slot_id="t1", model="openai/gpt-5.6-terra", effort="medium",
                route=ReviewRouteKind.API_CHAT)
    a = ReviewSlot(subagent_id="api-critic", **base)
    b = ReviewSlot(subagent_id="", **base)
    assert _attempt_key(request, a) != _attempt_key(request, b)


def test_roster_edit_changes_next_load_only(roster_env):
    roster_env.setenv(REVIEWER_SLOTS_ENV, _payload(
        [{"slot_id": "t1", "subagent_id": "api-critic"}]))
    before = load_reviewer_slot_config().triad[0]
    mutated = json.loads(json.dumps(_ROSTER))
    mutated["items"][0]["route"]["target_id"] = "openai/gpt-5.5"
    roster_env.setenv("OUROBOROS_SUBAGENTS", json.dumps(mutated))
    after = load_reviewer_slot_config().triad[0]
    assert before.target_id == "openai/gpt-5.6-terra"  # frozen materialization
    assert after.target_id == "openai/gpt-5.5"  # next load sees the edit


def test_endpoint_round_trips_the_actor_reference(roster_env):
    """GET /api/reviewer-slots returns an actor row as its subagent_id
    REFERENCE (resolved route only as read-only disclosure) — else the next
    UI save rewrites the reference into an inline route and the roster row
    stops being the SSOT for that reviewer."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.settings import api_reviewer_slots

    roster_env.setenv(REVIEWER_SLOTS_ENV, _payload(
        [{"slot_id": "t1", "subagent_id": "api-critic"}]))
    request = Request({"type": "http", "method": "GET", "path": "/api/reviewer-slots",
                       "headers": [], "query_string": b""})
    body = json.loads(asyncio.run(api_reviewer_slots(request)).body)
    row = body["triad"][0]
    assert row["subagent_id"] == "api-critic"
    assert "route" not in row  # the reference IS the stored form
    assert row["resolved_route"]["kind"] == "api_chat"
    assert row["resolved_route"]["target_id"] == "openai/gpt-5.6-terra"


def test_legacy_sdk_advisory_target_migration_branches(roster_env):
    """The three non-trivial branches of the retired Claude-SDK target
    migration (owner decision 2026-08-29): same-model translation, [1m]
    strip, and the fail-closed unmapped target that force-disables the row
    with a typed reason — never a silently swapped reviewer model."""
    import json as _json

    from ouroboros.reviewer_slot_config import parse_reviewer_slots

    def _advisory(kind, target, enabled=True):
        return parse_reviewer_slots(_json.dumps({
            "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/m"}}],
            "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/m"}}],
            "advisory": {"enabled": enabled, "route": {"kind": kind, "target_id": target}},
        })).advisory

    # claude-* bare name → routed catalog id of the SAME model.
    migrated = _advisory("api", "claude-opus-4.6")
    assert migrated.kind == "api_chat"
    assert migrated.target_id == "anthropic/claude-opus-4.6"
    assert migrated.enabled is True and not migrated.disabled_reason

    # The [1m] Claude-SDK selector is stripped before translation.
    stripped = _advisory("api", "claude-opus-4.6[1m]")
    assert stripped.target_id == "anthropic/claude-opus-4.6"

    # An unmapped legacy spelling ('opus') force-disables with the typed
    # reason — the row is never silently pointed at a different model.
    unmapped = _advisory("api", "opus")
    assert unmapped.enabled is False
    assert unmapped.disabled_reason == "legacy_claude_sdk_target_unmapped"


def test_migration_disable_reason_reaches_the_gate_diagnostic(roster_env):
    """advisory_gate_unavailability_reason distinguishes the migration
    force-disable from a standing owner disable (F2: the typed reason must
    not dead-end in a parse-time log line)."""
    import json as _json

    roster_env.setenv(REVIEWER_SLOTS_ENV, _json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/m"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/m"}}],
        "advisory": {"enabled": True, "route": {"kind": "api", "target_id": "opus"}},
    }))
    from ouroboros.tools.claude_advisory_review import advisory_gate_unavailability_reason
    assert advisory_gate_unavailability_reason() == (
        "advisory_slot_disabled:legacy_claude_sdk_target_unmapped"
    )


def test_settings_save_validates_actor_refs_against_the_incoming_roster(roster_env, tmp_path, monkeypatch):
    """S4 atomicity: one save may add a roster row AND reference it (validated
    against the incoming roster, not the stale env); a roster-only save that
    removes a still-referenced actor is refused instead of stranding every
    strict review surface post-save."""
    import asyncio
    import json as _json

    from starlette.requests import Request

    import ouroboros.gateway.settings as gws

    saved = {}

    def _fake_load():
        from ouroboros.config import SETTINGS_DEFAULTS
        out = dict(SETTINGS_DEFAULTS)
        out.update(saved)
        return out

    def _fake_write(payload, *, allow_elevation=False, allow_context_lowering=False,
                    authored_keys=(), boundary=None):
        saved.clear(); saved.update(payload)
        if boundary is not None:
            boundary.commit()
        return payload

    monkeypatch.setattr(gws, "load_settings", _fake_load)
    monkeypatch.setattr(gws, "_owner_write_settings", _fake_write)
    # A successful save exports settings into os.environ; keep this test
    # hermetic (the exported reviewer slots would leak into later tests).
    monkeypatch.setattr(gws, "_apply_settings_to_env", lambda *a, **k: None)

    def _post(body):
        async def _receive():
            return {"type": "http.request", "body": _json.dumps(body).encode()}
        request = Request({"type": "http", "method": "POST", "path": "/api/settings",
                           "headers": [("content-type", "application/json")],
                           "query_string": b"", "app": None}, receive=_receive)
        return asyncio.run(gws.api_settings_post(request))

    roster_env.delenv("OUROBOROS_SUBAGENTS", raising=False)
    new_roster = _json.dumps({"enabled": True, "items": [{
        "subagent_id": "fresh-critic", "name": "Fresh", "recommended_use": "x",
        "route": {"kind": "api_model", "target_id": "openai/gpt-5.5"}, "effort": "low"}]})
    slots = _json.dumps({
        "triad": [{"slot_id": "t1", "subagent_id": "fresh-critic"}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/m"}}],
        "advisory": {"enabled": False},
    })
    resp = _post({"OUROBOROS_SUBAGENTS": new_roster, "OUROBOROS_REVIEWER_SLOTS": slots})
    assert resp.status_code == 200, resp.body[:300]

    # Roster-only save dropping the still-referenced actor: refused.
    saved["OUROBOROS_REVIEWER_SLOTS"] = slots
    empty_roster = _json.dumps({"enabled": True, "items": [{
        "subagent_id": "other", "name": "Other", "recommended_use": "x",
        "route": {"kind": "api_model", "target_id": "openai/gpt-5.5"}, "effort": "low"}]})
    resp2 = _post({"OUROBOROS_SUBAGENTS": empty_roster})
    assert resp2.status_code == 400, resp2.body[:300]
