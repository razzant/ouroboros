"""Roster handles: a row is NAMED by a projection of its route, never by a stored label.

The stored ``subagent_id`` stays the hidden join key (reviewer references,
snapshots, custody, history). These tests pin the projection, the one argument
resolver, the save-time engine uniqueness and the facts-only model catalog —
each guard in both directions.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_onboarding_complete_endpoint import (
    WIZARD_PAYLOAD,
    onboarding as onboarding,  # explicit re-export of the real atomic settings fixture
)
from ouroboros.configured_subagents import (
    engine_identity,
    parse_configured_subagents,
    roster_handles,
    subagent_handle,
    validate_unique_engines,
)

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "web" / "tests" / "fixtures"
     / "subagent_handle_parity.json").read_text(encoding="utf-8")
)
PARITY, SNAPSHOTS = FIXTURE["rosters"], FIXTURE["snapshots"]
NO_GLOBAL: dict = {}


def _config(*rows):
    return parse_configured_subagents({"enabled": True, "items": list(rows)})


def _settings(*rows):
    return {"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": list(rows)})}


def _api(row_id, target="x-ai/grok-4.6", **extra):
    return {"subagent_id": row_id, "recommended_use": f"use {row_id}",
            "route": {"kind": "api_model", "target_id": target}, **extra}


def _session(row_id, target="codex=gpt-6-astra", pin="", **extra):
    route = {"kind": "agent_session", "target_id": target}
    if pin:
        route["credential_profile_id"] = pin
    return {"subagent_id": row_id, "recommended_use": f"use {row_id}", "route": route, **extra}


def _duplicate_of(config, settings):
    """Index pair a save refuses, read off the validator's own message."""
    try:
        validate_unique_engines(config, settings)
    except ValueError as exc:
        later, earlier = (int(part.split("]")[0]) for part in str(exc).split("items[")[1:3])
        return later, earlier
    return None


@pytest.mark.parametrize("roster", PARITY, ids=[item["case"] for item in PARITY])
def test_the_shared_table_pins_handles_roster_labels_and_refused_twins(roster):
    config = _config(*roster["items"])
    settings = {"OUROBOROS_PROCESSING_PREFERENCE": roster["global_processing"]}
    labels = roster_handles(config, settings)
    first_twin = next(
        ((index, want["same_engine_as"]) for index, want in enumerate(roster["expected"])
         if want["same_engine_as"] is not None), None)
    assert _duplicate_of(config, settings) == first_twin
    for row, want in zip(config.items, roster["expected"]):
        assert subagent_handle(row, settings) == want["handle"]
        assert labels[row.subagent_id] == want["roster"]


@pytest.mark.parametrize("record", SNAPSHOTS, ids=[item["case"] for item in SNAPSHOTS])
def test_a_frozen_record_is_named_from_its_own_facts(record):
    from ouroboros.subagent_history import execution_identity, recorded_handle, snapshot_handle

    assert snapshot_handle(record["snapshot"]) == record["handle"]
    assert recorded_handle({"identity": execution_identity(record["snapshot"])}) == record["handle"]


@pytest.mark.parametrize("global_processing", ["", "standard", "fast"])
@pytest.mark.parametrize("row", [
    {"effort": "xhigh"}, {"effort": "xhigh", "processing_preference": "standard"},
    {"processing_preference": "economy"}, {"access": "workspace_write", "pin": "koshak"},
], ids=["unset", "explicit-standard", "explicit-economy", "lowered-and-pinned"])
def test_one_engine_has_one_name_on_every_surface(row, global_processing, tmp_path):
    """The live catalog, the frozen snapshot, the startup receipt, the dated
    history fact and the reflection evidence must all carry the SAME handle for
    one row - including a row that inherits the global processing preference,
    which the snapshot freezes RESOLVED while the saved row leaves it empty."""
    from ouroboros import post_task_synthesis
    from ouroboros.subagent_history import execution_identity, recorded_handle, snapshot_handle
    from ouroboros.subagent_runtime import model_visible_subagent_catalog, select_subagent_snapshot
    from ouroboros.task_results import write_task_result

    settings = {**_settings(_session("primary-builder", target="claude=claude-opus-5", **row)),
                "OUROBOROS_PROCESSING_PREFERENCE": global_processing}
    catalog_name = model_visible_subagent_catalog(settings)["rows"][0]["subagent_id"]
    snapshot, _ = select_subagent_snapshot(settings, subagent_id=catalog_name)  # the name resolves
    write_task_result(tmp_path, "kid", "completed", result="ok", configured_subagent=snapshot,
                      parent_task_id="root", root_task_id="root", delegation_role="subagent")
    _text, evidence = post_task_synthesis._child_task_evidence(
        SimpleNamespace(drive_root=tmp_path), {"id": "root"})

    assert {
        catalog_name,
        snapshot_handle(snapshot),
        recorded_handle({"identity": execution_identity(snapshot)}),
        evidence[0]["engine"]["subagent_id"],
    } == {catalog_name}
    assert "standard" not in catalog_name and "full" not in catalog_name
    assert catalog_name.endswith("/fast") == (global_processing == "fast" and "processing_preference" not in row)


def test_a_handle_is_a_function_of_one_row_so_a_new_sibling_never_renames_it():
    alone = _config(_api("one"))
    crowded = _config(_api("one"), _api("two", effort="low"), _session("three"))
    assert subagent_handle(alone.items[0], NO_GLOBAL) == subagent_handle(crowded.items[0], NO_GLOBAL) == "x-ai/grok-4.6"
    assert roster_handles(crowded, NO_GLOBAL)["one"] == "x-ai/grok-4.6"


def test_row_identity_and_snapshot_identity_are_one_shape():
    """The save-time identity and the frozen-snapshot identity must not drift:
    the same row read through either reader yields the same facts."""
    from ouroboros.subagent_history import execution_identity
    from ouroboros.subagent_runtime import select_subagent_snapshot

    rows = (_session("s", pin="koshak", effort="xhigh", access="workspace_write"),
            _api("a", effort="low", processing_preference="fast"), _api("inherits", target="openai/gpt-5.6-sol"))
    settings = {**_settings(*rows), "OUROBOROS_PROCESSING_PREFERENCE": "economy"}
    for row in _config(*rows).items:
        snapshot, _legacy = select_subagent_snapshot(settings, subagent_id=row.subagent_id)
        assert execution_identity(snapshot) == engine_identity(row, settings)


@pytest.mark.parametrize("facet", [
    {"effort": "low"}, {"access": "workspace_write"}, {"processing_preference": "economy"},
])
def test_identical_engines_are_refused_and_one_differing_facet_is_accepted(facet):
    base = _session("first", effort="high")
    twin = _session("second", effort="high", recommended_use="different words, same engine")
    with pytest.raises(ValueError, match=r"items\[1\] runs the same engine as items\[0\]"):
        validate_unique_engines(_config(base, twin), NO_GLOBAL)
    validate_unique_engines(_config(base, {**twin, **facet}), NO_GLOBAL)
    validate_unique_engines(_config(base, _session("second", pin="other-account", effort="high")), NO_GLOBAL)
    validate_unique_engines(_config(base, _session("second", target="codex=gpt-5.6-sol", effort="high")), NO_GLOBAL)


def test_uniqueness_uses_effective_defaults_and_reads_stay_tolerant():
    """A session row without ``access`` IS a ``full`` row; and a roster saved
    before the rule still parses, resolves and projects — only a SAVE is refused."""
    from ouroboros.configured_subagents import resolve_configured_subagents
    from ouroboros.subagent_runtime import model_visible_subagent_catalog, select_subagent_snapshot

    rows = (_session("implicit"), _session("explicit", access="full"))
    with pytest.raises(ValueError, match="same engine"):
        validate_unique_engines(_config(*rows), NO_GLOBAL)
    settings = _settings(*rows)
    assert resolve_configured_subagents(settings).config is not None
    labels = [row["subagent_id"] for row in model_visible_subagent_catalog(settings)["rows"]]
    assert labels == ["codex=gpt-6-astra~implicit", "codex=gpt-6-astra~explicit"]
    for label, stored in zip(labels, ("implicit", "explicit")):
        assert select_subagent_snapshot(settings, subagent_id=label)[0]["selected_subagent_id"] == stored


def test_the_argument_resolves_by_handle_then_by_stored_id_and_refuses_a_cross_row_collision():
    from ouroboros.subagent_runtime import SubagentSelectionError, select_subagent_snapshot

    settings = _settings(_session("primary-builder", effort="xhigh"), _api("fast-scout"))
    by_handle, _ = select_subagent_snapshot(settings, subagent_id="codex=gpt-6-astra/xhigh")
    by_stored, _ = select_subagent_snapshot(settings, subagent_id="primary-builder")
    assert by_handle["selected_subagent_id"] == by_stored["selected_subagent_id"] == "primary-builder"
    assert by_handle["route"] == by_stored["route"]

    with pytest.raises(SubagentSelectionError) as unknown:
        select_subagent_snapshot(settings, subagent_id="codex=gpt-6-astra")  # facets are part of the name
    assert unknown.value.code == "unknown_subagent_id"
    assert "'codex=gpt-6-astra/xhigh'" in unknown.value.detail and "'x-ai/grok-4.6'" in unknown.value.detail
    assert "primary-builder" not in unknown.value.detail, "the refusal lists handles, not stored keys"

    # A bare session target is a legal stored id too: one string, two rows.
    collision = _settings(_session("first", target="codex"), _api("codex", target="openai/gpt-5.6-sol"))
    with pytest.raises(SubagentSelectionError) as conflict:
        select_subagent_snapshot(collision, subagent_id="codex")
    assert conflict.value.code == "subagent_selector_conflict"
    assert "'first'" in conflict.value.detail and "'openai/gpt-5.6-sol'" in conflict.value.detail
    # Both named values still reach their rows; the same string on ONE row is no conflict.
    assert select_subagent_snapshot(collision, subagent_id="first")[0]["route"]["target_id"] == "codex"
    assert select_subagent_snapshot(
        collision, subagent_id="openai/gpt-5.6-sol")[0]["selected_subagent_id"] == "codex"
    same_row = _settings(_session("codex", target="codex"))
    assert select_subagent_snapshot(same_row, subagent_id="codex")[0]["selected_subagent_id"] == "codex"


def test_a_switched_off_row_resolves_first_and_is_refused_as_itself():
    """The row switch composes with the resolver in ONE order: handle-or-stored-id
    to the row, then the switch. A switched-off row is refused `subagent_disabled`
    by either name; an unknown selector is offered the ENABLED choice set only;
    the cross-row conflict rule does not look at the switch."""
    from ouroboros.subagent_runtime import SubagentSelectionError, select_subagent_snapshot

    settings = _settings(
        _session("primary-builder", effort="xhigh"),
        {**_api("fast-scout"), "enabled": False},
    )
    for selector in ("fast-scout", "x-ai/grok-4.6"):
        with pytest.raises(SubagentSelectionError) as off:
            select_subagent_snapshot(settings, subagent_id=selector)
        assert off.value.code == "subagent_disabled", selector
    # ...and the same row, switched on, is selected by either name.
    on = _settings(_session("primary-builder", effort="xhigh"), _api("fast-scout"))
    for selector in ("fast-scout", "x-ai/grok-4.6"):
        assert select_subagent_snapshot(on, subagent_id=selector)[0]["selected_subagent_id"] == "fast-scout"

    with pytest.raises(SubagentSelectionError) as unknown:
        select_subagent_snapshot(settings, subagent_id="no-such-row")
    assert unknown.value.code == "unknown_subagent_id"
    assert "'codex=gpt-6-astra/xhigh'" in unknown.value.detail
    assert "grok" not in unknown.value.detail, "a switched-off row is not part of the choice set"

    collision = _settings({**_session("first", target="codex"), "enabled": False},
                          _api("codex", target="openai/gpt-5.6-sol"))
    with pytest.raises(SubagentSelectionError) as conflict:
        select_subagent_snapshot(collision, subagent_id="codex")
    assert conflict.value.code == "subagent_selector_conflict"


def test_a_bound_start_is_not_unbound_by_switching_its_row_off(tmp_path, monkeypatch):
    """A running episode is bound to its frozen snapshot: the owner switching the
    row off afterwards refuses NEW selections, never the bound actor's own start."""
    import ouroboros.subagent_runtime as runtime

    row = _session("primary-builder", effort="xhigh")
    snapshot, _ = runtime.select_subagent_snapshot(_settings(row), subagent_id="primary-builder")
    # The live row has since been re-pointed (another handle) AND switched off.
    live = _settings({**row, "effort": "high", "enabled": False})
    monkeypatch.setattr("ouroboros.config.runtime_settings", lambda: live)
    monkeypatch.setattr(runtime, "effective_runtime_subagent_settings", dict)

    def _start(selector):
        ctx = SimpleNamespace(
            task_id="bound-child", drive_root=tmp_path, budget_drive_root=str(tmp_path), task_metadata={},
            _configured_actor_bootstrap={"snapshot": snapshot, "selected_subagent_id": "primary-builder"})
        return json.loads(runtime.delegate_start_entry(ctx, "", subagent_id=selector).text)["reason"]

    for own in ("primary-builder", "codex=gpt-6-astra/xhigh", "codex=gpt-6-astra/high"):
        assert _start(own) == "configured_work_order_unavailable", own
    assert _start("cursor=kimi-k3-high") == "configured_actor_route_mismatch"


def test_the_model_catalog_is_facts_only_and_keyed_by_handle():
    from ouroboros.subagent_runtime import model_visible_subagent_catalog

    verbatim = "Любой язык.\nKeep punctuation: a/b, quotes, and cost $0."
    catalog = model_visible_subagent_catalog(_settings(
        {**_api("fast-scout", target="google/gemini-3.8-flash"), "recommended_use": verbatim},
        {**_session("primary-builder", pin="koshak", effort="xhigh"), "recommended_use": "Builds."},
    ))
    assert set(catalog) == {"rows"}, "no host-authored guidance, source or fingerprint reaches the model"
    api, session = catalog["rows"]
    assert api == {
        "subagent_id": "google/gemini-3.8-flash", "route_class": "API model",
        "requested_effort": "(not explicitly set)", "review_eligible": False, "recommended_use": verbatim,
    }
    assert list(session) == [
        "subagent_id", "route_class", "requested_effort", "requested_target",
        "mutating_access", "credential_profile_id", "review_eligible", "recommended_use",
    ], "facts lead, the owner's words ride last"
    assert session["subagent_id"] == "codex=gpt-6-astra/xhigh/@koshak"
    assert session["requested_target"] == "codex=gpt-6-astra"
    text = json.dumps(catalog)
    for stored_or_dropped in ("fast-scout", "primary-builder", "account_policy", "config_fingerprint"):
        assert stored_or_dropped not in text


def _post_settings(monkeypatch, body, stored=None):
    import asyncio

    from starlette.requests import Request

    import ouroboros.gateway.settings as gws

    saved = dict(stored or {})

    def _fake_load():
        from ouroboros.config import SETTINGS_DEFAULTS
        return {**SETTINGS_DEFAULTS, **saved}

    def _fake_write(payload, *, allow_elevation=False, allow_context_lowering=False,
                    authored_keys=(), boundary=None):
        saved.clear()
        saved.update(payload)
        if boundary is not None:
            boundary.commit()
        return payload

    monkeypatch.setattr(gws, "load_settings", _fake_load)
    monkeypatch.setattr(gws, "_owner_write_settings", _fake_write)
    monkeypatch.setattr(gws, "_unrecognised_review_models", lambda models: [])
    monkeypatch.setattr(gws, "_apply_settings_to_env", lambda *a, **k: None)

    async def _receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    request = Request({"type": "http", "method": "POST", "path": "/api/settings",
                       "headers": [("content-type", "application/json")],
                       "query_string": b"", "app": None}, receive=_receive)
    return asyncio.run(gws.api_settings_post(request)), saved


NO_REVIEWERS_CONFIRMED = {"allow_empty_review_pool": True}


def test_every_save_path_refuses_identical_engines_and_accepts_a_near_duplicate(monkeypatch):
    twins = {"enabled": True, "items": [_api("one", effort="low"), _api("two", effort="low")]}
    near = {"enabled": True, "items": [_api("one", effort="low"), _api("two", effort="high")]}

    refused, saved = _post_settings(monkeypatch, {"OUROBOROS_SUBAGENTS": twins})
    assert refused.status_code == 400 and b"same engine" in refused.body
    assert "OUROBOROS_SUBAGENTS" not in saved
    # None of these rows is a reviewer: the owner confirms the empty review pool.
    accepted, saved = _post_settings(monkeypatch, {"OUROBOROS_SUBAGENTS": near, **NO_REVIEWERS_CONFIRMED})
    assert accepted.status_code == 200, accepted.body[:300]
    assert json.loads(saved["OUROBOROS_SUBAGENTS"])["items"][1]["effort"] == "high"

    # The engine is judged under THIS save's effective facts: the same body that
    # turns the global preference to fast makes an unset row and an explicit fast row one engine.
    inherits = {"enabled": True, "items": [_api("one"), _api("two", processing_preference="fast")]}
    accepted, _ = _post_settings(monkeypatch, {"OUROBOROS_SUBAGENTS": inherits, **NO_REVIEWERS_CONFIRMED})
    assert accepted.status_code == 200, accepted.body[:300]
    refused, _ = _post_settings(
        monkeypatch, {"OUROBOROS_SUBAGENTS": inherits, "OUROBOROS_PROCESSING_PREFERENCE": "fast"})
    assert refused.status_code == 400 and b"same engine" in refused.body


TWINS = {"enabled": True, "items": [_api("one", effort="low"), _api("two", effort="low")]}


def test_stored_twins_never_block_an_unrelated_save_but_any_roster_edit_is_judged(monkeypatch):
    """Every Settings save re-posts the roster, so an install that saved twins
    before the rule must still save its other settings; the refusal applies only
    when the save CHANGES the roster - keeping the twin, or making a new one."""
    from ouroboros.configured_subagents import serialize_configured_subagents

    stored = {"OUROBOROS_SUBAGENTS": serialize_configured_subagents(_config(*TWINS["items"]))}
    # (1) untouched roster + an unrelated key: accepted, in the dict AND the canonical string form.
    for same in (TWINS, stored["OUROBOROS_SUBAGENTS"]):
        accepted, saved = _post_settings(
            monkeypatch, {"OUROBOROS_SUBAGENTS": same, "OUROBOROS_REVIEW_MAX_CYCLES": "3"}, stored)
        assert accepted.status_code == 200, accepted.body[:300]
        assert saved["OUROBOROS_REVIEW_MAX_CYCLES"] == "3"
    # (2) a roster edit that KEEPS the twin (another row's words) or ADDS a row beside it: refused.
    reworded = {"enabled": True, "items": [{**TWINS["items"][0], "recommended_use": "new words"}, TWINS["items"][1]]}
    grown = {"enabled": True, "items": [*TWINS["items"], _api("three", target="moonshotai/kimi-k3")]}
    for edited in (reworded, grown):
        refused, saved = _post_settings(monkeypatch, {"OUROBOROS_SUBAGENTS": edited}, stored)
        assert refused.status_code == 400 and b"same engine" in refused.body
        assert saved == stored
    # ...and an edit that tells the twins apart is an ordinary save.
    fixed = {"enabled": True, "items": [TWINS["items"][0], {**TWINS["items"][1], "effort": "high"}]}
    accepted, _ = _post_settings(monkeypatch, {"OUROBOROS_SUBAGENTS": fixed, **NO_REVIEWERS_CONFIRMED}, stored)
    assert accepted.status_code == 200, accepted.body[:300]
    # (3) the same twins on an install that stores none are a FRESH twin: refused.
    refused, _ = _post_settings(monkeypatch, {"OUROBOROS_SUBAGENTS": TWINS})
    assert refused.status_code == 400 and b"same engine" in refused.body


def test_the_row_switch_is_not_an_engine_facet_one_engine_one_seat(monkeypatch):
    """`enabled` never enters the handle or the identity. Uniqueness is judged across
    ALL rows: switching a twin off is a roster edit that keeps the twin, so it is
    refused like any other (change or remove one instead); an untouched roster with a
    switched-off twin still saves. Off rows stay named for cards and pickers, while
    the model's catalog lists enabled rows only - under the name the roster gives them."""
    from ouroboros.configured_subagents import serialize_configured_subagents
    from ouroboros.subagent_runtime import model_visible_subagent_catalog

    on, off = _api("one", effort="low"), {**_api("two", effort="low"), "enabled": False}
    config = _config(on, off)
    assert engine_identity(config.items[0], NO_GLOBAL) == engine_identity(config.items[1], NO_GLOBAL)
    assert roster_handles(config, NO_GLOBAL) == {"one": "x-ai/grok-4.6/low~one", "two": "x-ai/grok-4.6/low~two"}
    with pytest.raises(ValueError, match="same engine"):
        validate_unique_engines(config, NO_GLOBAL)
    # Near-duplicate direction: an off row with its own engine is an ordinary seat.
    validate_unique_engines(_config(on, {**off, "effort": "high"}), NO_GLOBAL)

    stored = {"OUROBOROS_SUBAGENTS": serialize_configured_subagents(_config(*TWINS["items"]))}
    switched = {"enabled": True, "items": [TWINS["items"][0], {**TWINS["items"][1], "enabled": False}]}
    refused, saved = _post_settings(monkeypatch, {"OUROBOROS_SUBAGENTS": switched}, stored)
    assert refused.status_code == 400 and b"same engine" in refused.body and saved == stored
    parked = {"OUROBOROS_SUBAGENTS": serialize_configured_subagents(_config(on, off))}
    accepted, _ = _post_settings(
        monkeypatch, {"OUROBOROS_SUBAGENTS": {"enabled": True, "items": [on, off]},
                      "OUROBOROS_REVIEW_MAX_CYCLES": "3"}, parked)
    assert accepted.status_code == 200, accepted.body[:300]

    names = [row["subagent_id"] for row in model_visible_subagent_catalog(parked)["rows"]]
    assert names == ["x-ai/grok-4.6/low~one"], "the off twin is absent; the live one keeps its roster name"


def test_onboarding_tolerates_stored_twins_it_leaves_untouched_and_judges_an_edit(onboarding):
    from ouroboros.configured_subagents import serialize_configured_subagents

    onboarding.settings_path.write_text(json.dumps({
        "OUROBOROS_SUBAGENTS": serialize_configured_subagents(_config(*TWINS["items"])),
        "OUROBOROS_MODEL": "openai/gpt-5.6-luna", "OPENROUTER_API_KEY": "sk-or-v1-abcdefghijklmnop",
    }), encoding="utf-8")
    grown = {"enabled": True, "items": [*TWINS["items"], _api("three", target="moonshotai/kimi-k3")]}
    for path in ("/api/onboarding/subagents/preview", "/api/onboarding/complete"):
        untouched = onboarding.client.post(path, json={**WIZARD_PAYLOAD, "OUROBOROS_SUBAGENTS": TWINS})
        assert untouched.status_code == 200, untouched.text
        edited = onboarding.client.post(path, json={**WIZARD_PAYLOAD, "OUROBOROS_SUBAGENTS": grown})
        assert edited.status_code == 400 and "same engine" in edited.json()["error"], edited.text
    assert len(json.loads(onboarding.saved()["OUROBOROS_SUBAGENTS"])["items"]) == 2


def test_onboarding_preview_and_completion_refuse_identical_engines(onboarding):
    """Both wizard endpoints write or preview the owner's roster through one
    draft seam; neither may admit twins, and a near-duplicate completes."""
    twins = {"enabled": True, "items": [_api("one", effort="low"), _api("two", effort="low")]}
    near = {"enabled": True, "items": [_api("one", effort="low"), _api("two", effort="high")]}
    for path in ("/api/onboarding/subagents/preview", "/api/onboarding/complete"):
        response = onboarding.client.post(
            path, json={**WIZARD_PAYLOAD, "subscriptionsConnected": True, "OUROBOROS_SUBAGENTS": twins})
        assert response.status_code == 400, response.text
        assert "same engine" in response.json()["error"]
    assert onboarding.calls["snapshot"] == 0 and not onboarding.settings_path.exists()

    response = onboarding.client.post(
        "/api/onboarding/complete",
        json={**WIZARD_PAYLOAD, "subscriptionsConnected": True, "OUROBOROS_SUBAGENTS": near})
    assert response.status_code == 200, response.text
    saved = json.loads(onboarding.saved()["OUROBOROS_SUBAGENTS"])["items"]
    # The owner's draft rides first; the wizard's review seats are MARKED catalog
    # rows on the same list (the pool has no inline lane to live in).
    assert [row["effort"] for row in saved[:2]] == ["low", "high"]
    assert saved[2:] and all(row["review_eligible"] and row["route"]["kind"] == "agent_session"
                             for row in saved[2:])


def test_an_actor_first_start_accepts_its_own_handle_and_stored_id_and_refuses_another_row(
    tmp_path, monkeypatch,
):
    """The bound start used to compare the raw argument with the stored id, so a
    handle accepted by ``schedule_subagent`` was rejected at the physical start."""
    import ouroboros.subagent_runtime as runtime

    rows = (_session("primary-builder", effort="xhigh"), _session("other", target="cursor=kimi-k3-high"))
    settings = _settings(*rows)
    monkeypatch.setattr("ouroboros.config.runtime_settings", lambda: settings)
    monkeypatch.setattr(runtime, "effective_runtime_subagent_settings", dict)
    snapshot, _ = runtime.select_subagent_snapshot(settings, subagent_id="primary-builder")

    def _start(selector, *, retry=False):
        ctx = SimpleNamespace(
            task_id="bound-child", drive_root=tmp_path, budget_drive_root=str(tmp_path), task_metadata={},
            # No canonical work order: a selector that NAMES the bound actor
            # passes the binding check and stops at the next typed refusal.
            _configured_actor_bootstrap={"snapshot": snapshot, "selected_subagent_id": "primary-builder"},
        )
        extra = {"retry_of": "inv-1"} if retry else {}
        return json.loads(runtime.delegate_start_entry(ctx, "", subagent_id=selector, **extra).text)["reason"]

    for retry in (False, True):
        for own in ("primary-builder", "codex=gpt-6-astra/xhigh"):
            assert _start(own, retry=retry) == "configured_work_order_unavailable", (own, retry)
        for foreign in ("other", "cursor=kimi-k3-high", "no-such-row"):
            assert _start(foreign, retry=retry) == "configured_actor_route_mismatch", (foreign, retry)

    # The roster moved on; the episode is still bound to its frozen snapshot.
    monkeypatch.setattr("ouroboros.config.runtime_settings", lambda: _settings(rows[1]))
    assert _start("codex=gpt-6-astra/xhigh") == "configured_work_order_unavailable"
    assert _start("cursor=kimi-k3-high") == "configured_actor_route_mismatch"


def test_history_is_named_from_each_records_own_facts_never_from_the_live_roster(tmp_path, monkeypatch):
    """The live row stored as ``fast-scout`` runs another engine today; the dated
    fact keeps the engine it ran, and an old record without a typed identity
    shows its recorded route target."""
    from ouroboros.context_runtime_facts import _delegation_capability_fact
    from ouroboros.subagent_history import recorded_handle

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("ouroboros.config.load_settings",
                        lambda: _settings(_api("fast-scout", target="moonshotai/kimi-k3")))
    typed = {
        "ts": "2026-09-20T10:00:00+00:00", "route": "cursor", "requested_model": "grok-4.6",
        "applied_model": "grok-4.6", "selected_subagent_id": "fast-scout", "run_id": "run-2",
        "identity": {"kind": "agent_session", "target_id": "cursor=grok-4.6", "effort": "high",
                     "credential_profile_id": "", "processing_preference": "", "access": "full"},
    }
    untyped = {"ts": "2026-08-18T02:00:00+00:00", "route": "api_model",
               "requested_model": "google/gemini-3.7-flash", "applied_model": "",
               "selected_subagent_id": "fast-scout-2", "run_id": "run-1"}
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "subagent_last_delegation.json").write_text(json.dumps({
        **typed, "latest_by_subagent": {"fast-scout": typed, "fast-scout-2": untyped},
    }), encoding="utf-8")

    delegation = _delegation_capability_fact()
    assert delegation["subagent_last_delegation"]["selected_subagent_id"] == "cursor=grok-4.6/high"
    assert [row["selected_subagent_id"] for row in delegation["subagents_last_executions"]] == [
        "cursor=grok-4.6/high", "google/gemini-3.7-flash",
    ]
    assert "kimi" not in json.dumps(delegation), "the past is never relabelled from the live roster"
    assert recorded_handle({"route": "codex", "requested_model": ""}) == "codex"
    assert recorded_handle({}) == ""


def test_the_startup_receipt_and_the_start_result_name_the_snapshots_own_handle(tmp_path, monkeypatch):
    import ouroboros.subagent_bootstrap as bootstrap
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros.delegate_shared import delegate_result

    snapshot, _ = runtime.select_subagent_snapshot(
        _settings(_session("primary-builder", pin="koshak", effort="xhigh")), subagent_id="primary-builder")
    monkeypatch.setattr(bootstrap, "_durable_zero_run_receipt", lambda *_a, **_k: {})
    ctx = SimpleNamespace(task_id="child", drive_root=tmp_path, budget_drive_root=str(tmp_path))
    receipt = json.loads(bootstrap._prepare_actor_first_bootstrap(
        ctx, {"id": "child", "objective": "Build", "configured_subagent": snapshot},
        SimpleNamespace(blocked=False),
    ))
    assert receipt["startup"]["selected_subagent_id"] == "codex=gpt-6-astra/xhigh/@koshak"
    assert ctx._configured_actor_bootstrap["selected_subagent_id"] == "primary-builder", "custody keeps the stored key"

    monkeypatch.setattr(delegate, "_delegate_start",
                        lambda *_a, **_k: delegate_result({"status": "started", "run_id": "run-1"}))
    started = json.loads(runtime.exact_start(
        SimpleNamespace(task_id="child", drive_root=tmp_path, budget_drive_root=str(tmp_path), task_metadata={}),
        "brief", {"snapshot": snapshot}).text)
    assert started["selected_subagent_id"] == "codex=gpt-6-astra/xhigh/@koshak"


def test_every_childs_engine_survives_the_evidence_cap(tmp_path):
    """The evidence text is capped at 6000 chars while one verbose row carries up
    to 1600+800 chars of result and trace, so after about three children the rest
    - and with them WHO ran each one - used to be truncated away. A compact
    overview of ALL children leads; the verbose rows follow and may be cut."""
    from ouroboros import post_task_synthesis
    from ouroboros.subagent_runtime import select_subagent_snapshot
    from ouroboros.task_results import write_task_result

    for index in range(12):  # twelve children, each frozen from its own row (a roster holds ten)
        row = _api(f"stored-key-{index}", target=f"vendor/model-{index}", effort="low")
        snapshot, _ = select_subagent_snapshot(_settings(row), subagent_id=row["subagent_id"])
        write_task_result(
            tmp_path, f"kid-{index:02d}", "completed", result="R" * 3000, trace_summary="T" * 2000,
            configured_subagent=snapshot, parent_task_id="root", root_task_id="root", delegation_role="subagent",
            started_at="2026-09-20T10:00:00+00:00", ts="2026-09-20T10:01:00+00:00")

    text, children = post_task_synthesis._child_task_evidence(
        SimpleNamespace(drive_root=tmp_path), {"id": "root"})

    assert len(children) == 12 and all("engine" in row for row in children), "the typed rows stay whole"
    assert "OMISSION NOTE" in text, "the fixture really overflows the cap"
    for index in range(12):
        assert f'"engine": "vendor/model-{index}/low"' in text, f"child {index} lost its engine to the cap"
    assert text.index('"children_overview"') < text.index('"children"')
    assert '"duration_sec": 60.0' in text and "stored-key-" not in text


def test_a_processing_only_save_may_create_effective_twins_by_decision(monkeypatch):
    """A deliberate residual, pinned: uniqueness is judged only when a save changes
    the ROSTER. Turning the global processing preference to `fast` makes an
    inheriting row and an explicit-`fast` row one engine, and that save is
    ACCEPTED (the roster is untouched); the live roster then tells them apart as
    `<handle>~<stored id>`, and the next roster edit is judged."""
    from ouroboros.configured_subagents import serialize_configured_subagents
    from ouroboros.subagent_runtime import model_visible_subagent_catalog

    rows = (_api("inherits"), _api("explicit", processing_preference="fast"))
    stored = {"OUROBOROS_SUBAGENTS": serialize_configured_subagents(_config(*rows))}
    roster = {"enabled": True, "items": list(rows)}

    accepted, saved = _post_settings(
        monkeypatch, {"OUROBOROS_SUBAGENTS": roster, "OUROBOROS_PROCESSING_PREFERENCE": "fast"}, stored)
    assert accepted.status_code == 200, accepted.body[:300]
    assert saved["OUROBOROS_PROCESSING_PREFERENCE"] == "fast"
    names = [row["subagent_id"] for row in model_visible_subagent_catalog(saved)["rows"]]
    assert names == ["x-ai/grok-4.6/fast~inherits", "x-ai/grok-4.6/fast~explicit"]

    reworded = {"enabled": True, "items": [{**rows[0], "recommended_use": "new words"}, rows[1]]}
    refused, _ = _post_settings(monkeypatch, {"OUROBOROS_SUBAGENTS": reworded}, saved)
    assert refused.status_code == 400 and b"same engine" in refused.body

