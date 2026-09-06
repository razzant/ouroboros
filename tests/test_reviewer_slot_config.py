"""Reviewer-slot SSOT (phase 6.1 / ABI 7.0): structured parse, default panel, projection.

The one structured setting is the ONE configuration surface (ABI-10, owner
5.4=A: the legacy comma-list migration read is REMOVED); without it the loader
serves the shipped default panel over the derived env plane. The comma keys
survive only as a runtime projection for legacy consumers (external review
tooling, benchmark manifests) — no review surface reads them (owner R2 retired
the task-acceptance API pin). Malformed configuration REFUSES typed on every
surface — an unknown token must never silently pick a transport, in either
direction.
"""
import json

import pytest

from ouroboros.reviewer_slot_config import (
    REVIEWER_SLOTS_ENV,
    SCOPE_SLOT_LIMIT,
    TRIAD_SLOT_LIMIT,
    advisory_slot_config,
    commit_triad_delivery,
    commit_triad_rows,
    load_reviewer_slot_config,
    parse_reviewer_slots,
    project_reviewer_slots_into_env,
    reviewer_slot_config_error,
    reviewer_slot_save_check,
)

_STRUCTURED = {
    "triad": [
        {"slot_id": "t_api", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-luna"},
         "effort": "high"},
        {"slot_id": "t_sess", "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
         "effort": "xhigh"},
    ],
    "scope": [
        {"slot_id": "s_api", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-terra"}},
    ],
    "advisory": {"enabled": False,
                 "route": {"kind": "agent_session", "target_id": "codex"},
                 "effort": "low"},
}


def _set_structured(monkeypatch, payload=None):
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload or _STRUCTURED))


def _clear_legacy(monkeypatch):
    for key in ("OUROBOROS_REVIEW_ROUTES", "OUROBOROS_SCOPE_REVIEW_ROUTES",
                "OUROBOROS_ADVISORY_REVIEW_ROUTE", "OUROBOROS_SCOPE_REVIEW_MODEL"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Structured parse.
# ---------------------------------------------------------------------------


def test_structured_config_round_trips(monkeypatch):
    _set_structured(monkeypatch)
    config = load_reviewer_slot_config()
    assert config.source == "structured"
    assert [r.slot_id for r in config.triad] == ["t_api", "t_sess"]
    api_row, sess_row = config.triad
    assert (api_row.kind, api_row.session_target) == ("api_chat", "")
    assert sess_row.session_target == "codex=gpt-5.6-sol"
    assert sess_row.effort == "xhigh"
    assert config.scope[0].effort == ""  # "" = the surface's global default
    assert config.advisory.enabled is False
    assert config.advisory.kind == "agent_session"
    assert config.advisory.target_id == "codex"
    delivery = commit_triad_delivery()
    assert delivery["slot_ids"] == ["t_api", "t_sess"]
    assert delivery["legacy_skill_fingerprint"] is False


@pytest.mark.parametrize("mutate,fragment", [
    (lambda p: p.__setitem__("triad", []), "at least one slot"),
    (lambda p: p.__setitem__("scope", []), "at least one slot"),
    (lambda p: p["triad"][0]["route"].__setitem__("kind", "codex"), "unknown route kind"),
    (lambda p: p["triad"][0]["route"].__setitem__("target_id", ""), "target_id is empty"),
    (lambda p: p["triad"][1]["route"].__setitem__("target_id", "codex::gpt-5.6"), "'::'"),
    (lambda p: p["triad"][1].__setitem__("slot_id", "t_api"), "appears twice"),
    (lambda p: p["triad"][0].__setitem__("effort", "enormous"), "unknown effort"),
    (lambda p: p["triad"][1].update({
        "route": {"kind": "agent_session", "target_id": "cursor=gpt-5.6-sol-high-fast"},
        "effort": "medium",
    }), "conflicts with compound route effort"),
    (lambda p: p.__setitem__("bogus", 1), "unknown top-level keys"),
    (lambda p: p.__setitem__("triad", p["triad"] * 6), f"limit is {TRIAD_SLOT_LIMIT}"),
], ids=["empty-triad", "empty-scope", "vendor-kind", "empty-target", "double-colon",
        "dup-slot-id", "bad-effort", "compound-effort-conflict", "unknown-key", "triad-cap"])
def test_malformed_structured_config_refuses_typed(mutate, fragment):
    payload = json.loads(json.dumps(_STRUCTURED))
    mutate(payload)
    with pytest.raises(ValueError) as err:
        parse_reviewer_slots(json.dumps(payload))
    assert fragment in str(err.value)


def test_scope_cap_is_enforced():
    payload = json.loads(json.dumps(_STRUCTURED))
    payload["scope"] = [
        {"slot_id": f"s_{i}", "route": {"kind": "api_chat", "target_id": "m"}}
        for i in range(SCOPE_SLOT_LIMIT + 1)
    ]
    with pytest.raises(ValueError, match=f"limit is {SCOPE_SLOT_LIMIT}"):
        parse_reviewer_slots(json.dumps(payload))


def test_non_json_refuses_typed():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_reviewer_slots("{nope")


@pytest.mark.parametrize("mutate,fragment", [
    (lambda row: row.__setitem__("effrot", "low"), "unknown keys"),
    (lambda row: row.__setitem__("slot_id", ["t_api"]), "slot_id must be a string"),
    (lambda row: row.__setitem__("effort", 1), "effort must be a string"),
    (lambda row: row["route"].__setitem__("kind", ["api_chat"]), "route.kind must be a string"),
    (lambda row: row["route"].__setitem__("target_id", ["model"]), "route.target_id must be a string"),
    (lambda row: row["route"].__setitem__("profile_id", ["profile"]), "route.profile_id must be a string"),
    (lambda row: row["route"].__setitem__("typo", True), "route has unknown keys"),
    (lambda row: row["route"].__setitem__("profile_id", "api-profile"), "meaningful only for agent_session"),
], ids=[
    "unknown-row-key", "slot-id-type", "effort-type", "route-kind-type",
    "route-target-type", "route-profile-type", "unknown-route-key", "api-profile-pin",
])
def test_structured_rows_never_coerce_or_ignore_malformed_fields(mutate, fragment):
    payload = json.loads(json.dumps(_STRUCTURED))
    row = payload["triad"][0]
    mutate(row)
    with pytest.raises(ValueError) as err:
        parse_reviewer_slots(json.dumps(payload))
    assert fragment in str(err.value)


@pytest.mark.parametrize("advisory,fragment", [
    ({"enabled": "false"}, "enabled must be a boolean"),
    ({"enabled": True, "effrot": "low"}, "unknown keys"),
    ({"enabled": True, "effort": 1}, "effort must be a string"),
    ({"enabled": True, "kind": ["api"]}, "kind must be a string"),
    ({"enabled": True, "target_id": ["model"]}, "target_id must be a string"),
    ({"enabled": True, "route": {"kind": "api", "target_id": "", "typo": True}},
     "route has unknown keys"),
    ({"enabled": True, "route": {"kind": ["api"], "target_id": ""}},
     "route.kind must be a string"),
    ({"enabled": True, "route": {"kind": "api", "target_id": ["model"]}},
     "route.target_id must be a string"),
    ({"enabled": True, "route": {"kind": "agent_session", "target_id": "codex",
                                   "profile_id": ["profile"]}},
     "route.profile_id must be a string"),
    ({"enabled": True, "route": {"kind": "api", "target_id": "",
                                   "profile_id": "api-profile"}},
     "meaningful only for agent_session"),
    ({"enabled": True, "route": {"kind": "agent_session", "target_id": ""}},
     "needs a non-empty target_id"),
    ({"enabled": True, "kind": "agent_session", "target_id": ""},
     "needs a non-empty target_id"),
    ({"enabled": True, "kind": "agent_session", "target_id": "codex",
      "route": {"kind": "api", "target_id": "sonnet"}},
     "either route or legacy kind/target_id, not both"),
    ({"enabled": True, "kind": "api", "target_id": "sonnet",
      "route": {"kind": "api", "target_id": "sonnet"}},
     "either route or legacy kind/target_id, not both"),
], ids=[
    "enabled-type", "unknown-advisory-key", "effort-type", "legacy-kind-type",
    "legacy-target-type", "unknown-route-key", "route-kind-type", "route-target-type",
    "route-profile-type", "api-profile-pin", "empty-session-route",
    "empty-legacy-session-route", "conflicting-route-authorities", "duplicate-route-authorities",
])
def test_advisory_never_coerces_or_ignores_malformed_fields(advisory, fragment):
    payload = json.loads(json.dumps(_STRUCTURED))
    payload["advisory"] = advisory
    with pytest.raises(ValueError) as err:
        parse_reviewer_slots(json.dumps(payload))
    assert fragment in str(err.value)


def test_advisory_keeps_recognized_legacy_shape_and_empty_api_target():
    payload = json.loads(json.dumps(_STRUCTURED))
    payload["advisory"] = {
        "enabled": True,
        "kind": "agent_session",
        "target_id": "codex=claude-fable-5",
        "effort": "high",
    }
    advisory = parse_reviewer_slots(json.dumps(payload)).advisory
    assert (advisory.kind, advisory.target_id, advisory.effort) == (
        "agent_session", "codex=claude-fable-5", "high",
    )

    payload["advisory"] = {"enabled": True, "route": {"kind": "api"}}
    advisory = parse_reviewer_slots(json.dumps(payload)).advisory
    # The retired legacy "api" kind migrates to the shared api_chat vocabulary;
    # an empty target keeps meaning the shipped routed default.
    assert (advisory.kind, advisory.target_id, advisory.effort) == ("api_chat", "", "low")

    payload["advisory"] = {
        "enabled": False,
        "route": {"kind": "agent_session", "target_id": ""},
    }
    advisory = parse_reviewer_slots(json.dumps(payload)).advisory
    assert (advisory.enabled, advisory.kind, advisory.target_id) == (
        False, "agent_session", "",
    )


def test_settings_save_refuses_a_malformed_row_before_persistence():
    from starlette.requests import Request

    from ouroboros.gateway.settings import _api_settings_post_locked

    payload = json.loads(json.dumps(_STRUCTURED))
    payload["triad"][0]["effrot"] = "low"
    request = Request({
        "type": "http", "method": "POST", "path": "/api/settings",
        "headers": [], "query_string": b"",
    })
    response = _api_settings_post_locked(
        request,
        {REVIEWER_SLOTS_ENV: json.dumps(payload)},
    )
    body = json.loads(response.body)
    assert response.status_code == 400
    assert body["saved"] is False
    assert "triad[0] has unknown keys" in body["error"]


def test_settings_save_refuses_an_enabled_empty_advisory_session_route():
    from starlette.requests import Request

    from ouroboros.gateway.settings import _api_settings_post_locked

    payload = json.loads(json.dumps(_STRUCTURED))
    payload["advisory"] = {
        "enabled": True,
        "route": {"kind": "agent_session", "target_id": ""},
    }
    request = Request({
        "type": "http", "method": "POST", "path": "/api/settings",
        "headers": [], "query_string": b"",
    })
    response = _api_settings_post_locked(
        request,
        {REVIEWER_SLOTS_ENV: json.dumps(payload)},
    )
    body = json.loads(response.body)
    assert response.status_code == 400
    assert body["saved"] is False
    assert "needs a non-empty target_id" in body["error"]


# ---------------------------------------------------------------------------
# reviewer_slot_config_error (#116): the loud-check facade for plan/skill review.
# ---------------------------------------------------------------------------


def test_config_error_is_empty_on_absent_valid_and_legacy_only(monkeypatch):
    from ouroboros.reviewer_slot_config import reviewer_slot_config_error

    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    assert reviewer_slot_config_error() == ""
    _set_structured(monkeypatch)
    assert reviewer_slot_config_error() == ""
    # Legacy-only configs (bench constraint: benches configure ONLY the comma
    # keys — including deliberate single-reviewer duplicates) must never trip
    # this check: the facade reads the STRUCTURED raw value alone. Even a
    # broken legacy route env stays out of scope here (it refuses typed on its
    # own consumers instead).
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m/one,m/one")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "m/scope")
    assert reviewer_slot_config_error() == ""
    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "carrier-pigeon")
    assert reviewer_slot_config_error() == ""


def test_config_error_reports_row_precise_text(monkeypatch):
    from ouroboros.reviewer_slot_config import reviewer_slot_config_error

    monkeypatch.setenv(REVIEWER_SLOTS_ENV, "{broken")
    assert "not valid JSON" in reviewer_slot_config_error()
    monkeypatch.setenv(
        REVIEWER_SLOTS_ENV,
        json.dumps({"triad": [], "scope": [], "advisory": None}),
    )
    assert "triad needs at least one slot" in reviewer_slot_config_error()


def test_authored_state_is_three_valued_and_invalid_means_the_loader_refuses(monkeypatch):
    """``authored_reviewer_slots_state`` is what the retired-keys notice reads: absent and
    invalid are DIFFERENT states, because on malformed text the loader raises instead of
    serving the shipped default (the fact the notice sentence has to state). The facade
    ``reviewer_slot_config_error`` is the env-read projection of the same state."""
    from ouroboros.reviewer_slot_config import authored_reviewer_slots_state

    assert authored_reviewer_slots_state("") == ("absent", "")
    assert authored_reviewer_slots_state("   ") == ("absent", "")
    assert authored_reviewer_slots_state(json.dumps(_STRUCTURED)) == ("authored", "")
    state, err = authored_reviewer_slots_state("{broken")
    assert state == "invalid" and "not valid JSON" in err
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, "{broken")
    assert reviewer_slot_config_error() == err
    with pytest.raises(ValueError, match="not valid JSON"):
        load_reviewer_slot_config()  # no default panel serves on malformed text


def test_api_rows_keep_the_existing_provider_tagged_spelling():
    """`provider::model` is the EXISTING direct-routing spelling for API model
    ids; the owner's no-'::' directive is about harness routes only."""
    payload = json.loads(json.dumps(_STRUCTURED))
    payload["scope"][0]["route"]["target_id"] = "openai-compatible::scope-reviewer-x"
    config = parse_reviewer_slots(json.dumps(payload))
    assert config.scope[0].target_id == "openai-compatible::scope-reviewer-x"


def test_slot_id_is_never_an_array_index(monkeypatch):
    """Reordering rows must not move a row's identity (6.1)."""
    reordered = json.loads(json.dumps(_STRUCTURED))
    reordered["triad"] = list(reversed(reordered["triad"]))
    _set_structured(monkeypatch, reordered)
    rows = {r.slot_id: r for r in commit_triad_rows()}
    assert rows["t_sess"].session_target == "codex=gpt-5.6-sol"
    assert rows["t_api"].target_id == "openai/gpt-5.6-luna"


# ---------------------------------------------------------------------------
# Caps are pinned to their real owners, not free-floating copies.
# ---------------------------------------------------------------------------


def test_triad_cap_is_the_commit_review_ceiling():
    from ouroboros.tools.review import MAX_MODELS

    assert TRIAD_SLOT_LIMIT == MAX_MODELS


def test_scope_cap_is_the_parallel_review_pool_width():
    import inspect

    from ouroboros.tools import parallel_review

    source = inspect.getsource(parallel_review)
    assert f"min(len(scope_slots), {SCOPE_SLOT_LIMIT})" in source, (
        "SCOPE_SLOT_LIMIT no longer matches the scope thread-pool width — "
        "move both or neither"
    )


# ---------------------------------------------------------------------------
# Legacy migration read (comma-lists as API slots + copied global efforts).
# ---------------------------------------------------------------------------


def test_absent_structured_key_serves_the_default_panel(monkeypatch):
    """ABI 7.0 (ABI-10): no structured key -> the shipped default panel over
    the derived env plane (which a bench launcher's env override still feeds);
    historical positional ids keep receipts lining up."""
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m/one,m/two")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "m/scope")
    config = load_reviewer_slot_config()
    assert config.source == "default"
    # Row effort stays '' and resolves to the surface default at use time.
    assert [(r.slot_id, r.kind, r.target_id, r.effort) for r in config.triad] == [
        ("slot_1", "api_chat", "m/one", ""),
        ("slot_2", "api_chat", "m/two", ""),
    ]
    assert [(r.slot_id, r.target_id) for r in config.scope] == [
        ("scope_slot_1", "m/scope")]
    assert config.advisory.enabled is True and config.advisory.kind == "api_chat"
    assert commit_triad_delivery()["legacy_skill_fingerprint"] is True


def test_default_panel_efforts_resolve_to_the_surface_defaults(monkeypatch):
    from ouroboros.reviewer_slot_config import row_effort

    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m/one")
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "medium")
    row = load_reviewer_slot_config().triad[0]
    assert row_effort(row, "review") == "medium"


def test_retired_phase5_route_envs_are_ignored(monkeypatch):
    """ABI 7.0 (ABI-10): the phase-5 per-row/advisory route envs are RETIRED —
    they no longer route any default-panel row into a session. Delegated
    reviewer rows exist only through the structured setting."""
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m/one,m/two")
    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "api_chat,agent_session")
    monkeypatch.setenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", "agent_session")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "claude=claude-fable-5:high")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROFILE", "legacy-profile")
    config = load_reviewer_slot_config()
    assert all(row.kind == "api_chat" for row in config.triad)
    assert all(row.session_target == "" and row.profile_id == "" for row in config.triad)
    assert config.advisory.kind == "api_chat"
    assert config.advisory.target_id == ""
    delivery = commit_triad_delivery()
    assert delivery["legacy_skill_fingerprint"] is True
    assert delivery["session_targets"] == ["", ""]


def test_default_panel_round_trips_through_the_settings_endpoint(monkeypatch):
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.settings import api_reviewer_slots
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m/one")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "m/scope")

    request = Request({
        "type": "http", "method": "GET", "path": "/api/reviewer-slots",
        "headers": [], "query_string": b"",
    })
    body = json.loads(asyncio.run(api_reviewer_slots(request)).body)
    assert body["source"] == "default"
    migrated = json.dumps({key: body[key] for key in ("triad", "scope", "advisory")})
    assert reviewer_slot_save_check(migrated) == ""


# ---------------------------------------------------------------------------
# Runtime projection into the legacy comma keys (legacy consumers only).
# ---------------------------------------------------------------------------


def test_projection_exposes_only_api_rows(monkeypatch):
    _set_structured(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "stale/comma-value")
    project_reviewer_slots_into_env()
    import os

    assert os.environ["OUROBOROS_REVIEW_MODELS"] == "openai/gpt-5.6-luna"
    assert os.environ["OUROBOROS_SCOPE_REVIEW_MODELS"] == "openai/gpt-5.6-terra"


def test_all_delegated_triad_projects_no_api_model_and_acceptance_follows_the_rows(monkeypatch):
    """An all-session triad has no api model id to project: the comma key keeps
    the shipped default for its LEGACY readers only (never a stale comma value),
    while task acceptance — like every review surface — reads the session row
    itself (owner R2; the former API-default substitution is gone)."""
    from ouroboros.reviewer_slot_config import triad_delivery_slots

    payload = json.loads(json.dumps(_STRUCTURED))
    payload["triad"] = [payload["triad"][1]]
    _set_structured(monkeypatch, payload)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "stale/comma-value")
    project_reviewer_slots_into_env()
    import os

    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

    # The comma key keeps the shipped default for its legacy readers, never a
    # stale comma value. ABI-10 retired ``SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"]``,
    # so the default list is read from its v7 SSOT.
    assert os.environ["OUROBOROS_REVIEW_MODELS"] == ",".join(OPENROUTER_REVIEW_DEFAULTS["triad"])
    slots = triad_delivery_slots(role_hint="task acceptance")
    assert [(s.slot_id, s.route.value, s.session_target, s.effort) for s in slots] == [
        ("t_sess", "agent_session", "codex=gpt-5.6-sol", "xhigh"),
    ]


def test_projection_malformed_leaves_legacy_keys_and_floors(monkeypatch):
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, "{broken")
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "owner/comma-value")
    project_reviewer_slots_into_env()
    import os

    # Env-apply must not take the server down; the review surfaces re-parse
    # strictly and block with the precise error instead.
    assert os.environ["OUROBOROS_REVIEW_MODELS"] == "owner/comma-value"
    with pytest.raises(ValueError):
        commit_triad_rows()
    # Task acceptance refuses on the same parse (R3) instead of reading the
    # comma key its projection left in place.
    from ouroboros.reviewer_slot_config import triad_delivery_slots

    with pytest.raises(ValueError):
        triad_delivery_slots(role_hint="task acceptance")


# ---------------------------------------------------------------------------
# The rows feed the review machinery with per-row properties.
# ---------------------------------------------------------------------------


def test_scope_slots_from_structured_config(monkeypatch):
    payload = json.loads(json.dumps(_STRUCTURED))
    payload["scope"] = [
        {"slot_id": "s_owner", "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
         "effort": "max"},
    ]
    _set_structured(monkeypatch, payload)
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import scope_reviewer_slots

    slots = scope_reviewer_slots()
    assert len(slots) == 1
    slot = slots[0]
    assert slot.slot_id == "s_owner"
    assert slot.route is ReviewRouteKind.AGENT_SESSION
    assert slot.session_target == "codex=gpt-5.6-sol"
    assert slot.effort == "max"


def test_session_executor_prefers_the_slots_own_target(monkeypatch):
    from ouroboros.review_execution import (
        AgentSessionReviewExecutor,
        ReviewAssignment,
        ReviewRouteKind,
        ReviewRouteUnavailable,
    )
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    monkeypatch.delenv("OUROBOROS_REVIEW_SESSION_ROUTE", raising=False)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)
    request = ReviewRequest(surface="scope_review", goal="g")
    slot = ReviewSlot(slot_id="s_owner", model="codex=gpt-5.6-sol", effort="xhigh",
                      route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex=gpt-5.6-sol")
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=request, slot=slot))
    route = executor._session_route()
    assert (route.route_id, route.model, route.effort) == ("codex", "gpt-5.6-sol", "xhigh")

    # Without a per-row target the shared-route absence stays a typed refusal.
    bare = AgentSessionReviewExecutor(ReviewAssignment(
        request=request,
        slot=ReviewSlot(slot_id="s2", model="m", route=ReviewRouteKind.AGENT_SESSION)))
    with pytest.raises(ReviewRouteUnavailable):
        bare._session_route()


# ---------------------------------------------------------------------------
# «Выполняется как» (D22): the last effective execution beside each saved row.
# ---------------------------------------------------------------------------


def test_the_two_surfaces_that_run_concurrently_do_not_erase_each_others_rows(monkeypatch):
    """`run_parallel_review` runs the triad and the scope surfaces CONCURRENTLY (its
    own first line says so), in two threads of one process, and each finishes by
    folding its rows into ONE projection file. `write_text_atomic` makes the write
    untearable but says nothing about the read-modify-write around it: both threads
    read the same "before", and whichever wrote last erased the other surface's rows
    outright — the panel lost a whole row's «Выполняется как» line, silently.

    The interleave is forced rather than raced: the first writer is held between its
    read and its write for long enough that an unlocked second writer would read the
    stale empty file underneath it."""
    import threading
    import time
    from types import SimpleNamespace

    from ouroboros import utils as ouro_utils
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.reviewer_slot_config import (
        record_reviewer_slot_executions,
        reviewer_slot_last_executions,
    )

    real_now = ouro_utils.utc_now_iso
    held = threading.Event()

    def _slow_first_writer():
        # Called once per recorded row, AFTER the read and BEFORE the write.
        if not held.is_set():
            held.set()
            time.sleep(0.5)
        return real_now()

    monkeypatch.setattr(ouro_utils, "utc_now_iso", _slow_first_writer)

    def _record(slot_id, surface):
        slot = ReviewSlot(slot_id=slot_id, model="openai/gpt-5.6-sol", effort="high",
                          route=ReviewRouteKind.API_CHAT)
        actor = SimpleNamespace(slot_id=slot_id, status="ok", usage={})
        record_reviewer_slot_executions(surface, [actor], {slot_id: slot})

    triad = threading.Thread(target=_record, args=("t_triad", "multi_model_review"))
    triad.start()
    held.wait(2.0)          # the triad thread is now parked between read and write
    time.sleep(0.1)
    scope = threading.Thread(target=_record, args=("s_scope", "scope_review"))
    scope.start()
    triad.join(5.0)
    scope.join(5.0)

    rows = reviewer_slot_last_executions()
    assert "t_triad" in rows and "s_scope" in rows, sorted(rows)
    assert rows["t_triad"]["surface"] == "multi_model_review"
    assert rows["s_scope"]["surface"] == "scope_review"


def test_last_execution_projection_round_trips():
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.reviewer_slot_config import (
        record_reviewer_slot_executions,
        reviewer_slot_last_executions,
    )

    slot = ReviewSlot(slot_id="t_sess", model="codex=gpt-5.6-sol", effort="xhigh",
                      route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex=gpt-5.6-sol")
    actor = SimpleNamespace(slot_id="t_sess", status="ok", usage={
        "delegated_route": "codex", "resolved_model": "gpt-5.6-sol",
        "verdict_method": "light_model_extraction",
        "capability_delta": [{"kind": "capability_delta",
                              "reason": "extraction_instead_of_schema"}],
    })
    record_reviewer_slot_executions("multi_model_review", [actor], {"t_sess": slot})
    projection = reviewer_slot_last_executions()["t_sess"]
    # The saved row vs what it REALLY ran as — the whole point of the block.
    assert projection["requested"]["session_target"] == "codex=gpt-5.6-sol"
    assert projection["effective"]["route"] == "agent_session:codex"
    assert projection["effective"]["model"] == "gpt-5.6-sol"
    assert projection["effective"]["verdict_method"] == "light_model_extraction"
    assert projection["capability_delta"][0]["reason"] == "extraction_instead_of_schema"


def test_reviewer_slots_endpoint_reports_rows_and_config_errors(monkeypatch):
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.settings import api_reviewer_slots

    def _get():
        request = Request({"type": "http", "method": "GET", "path": "/api/reviewer-slots",
                           "headers": [], "query_string": b""})
        return asyncio.run(api_reviewer_slots(request))

    _set_structured(monkeypatch)
    body = json.loads(_get().body)
    assert body["source"] == "structured"
    assert body["limits"] == {"triad": TRIAD_SLOT_LIMIT, "scope": SCOPE_SLOT_LIMIT, "advisory": 1, "deep_review": 1}
    assert body["triad"][1]["route"]["kind"] == "agent_session"
    assert body["advisory"]["enabled"] is False

    monkeypatch.setenv(REVIEWER_SLOTS_ENV, "{broken")
    broken = json.loads(_get().body)
    # A typed error beside the editor that can fix it — never a 500.
    assert "config_error" in broken and "not valid JSON" in broken["config_error"]


def test_reviewer_slots_endpoint_round_trips_the_manual_credential_pin(monkeypatch):
    """Audit #3.4: GET must return the Q2 manual pin (route.profile_id), or a
    subsequent save silently wipes it. Reversible by design, honest round-trip."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.settings import api_reviewer_slots

    payload = {
        "triad": [{"slot_id": "t1",
                   "route": {"kind": "agent_session", "target_id": "codex", "profile_id": "koshak"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-terra"}}],
        "advisory": {"enabled": True,
                     "route": {"kind": "agent_session", "target_id": "codex", "profile_id": "backup"}},
    }
    _set_structured(monkeypatch, payload)
    request = Request({"type": "http", "method": "GET", "path": "/api/reviewer-slots",
                       "headers": [], "query_string": b""})
    body = json.loads(asyncio.run(api_reviewer_slots(request)).body)
    assert body["triad"][0]["route"]["profile_id"] == "koshak"
    assert body["advisory"]["route"]["profile_id"] == "backup"
    # An api row carries no pin, so the key stays absent (not a null).
    assert "profile_id" not in body["scope"][0]["route"]


def test_login_request_honors_the_engine_client_pty_wire_contract():
    """Audit #3.3: codex client_pty REQUIRES loginFlow=browser_redirect (else a
    hard 400); loginFlow is codex-only, so it is never sent for another harness."""
    from ouroboros.gateway.claudexor_accounts import _build_login_request

    codex_pty = _build_login_request("codex", "", "client_pty", "")
    assert codex_pty["loginFlow"] == "browser_redirect"
    assert codex_pty["transport"] == "client_pty"
    # The in-app device flow keeps its own loginFlow.
    assert _build_login_request("codex", "", "", "device_auth")["loginFlow"] == "device_auth"
    # A non-codex client_pty carries NO loginFlow (the schema rejects it).
    claude_pty = _build_login_request("claude", "main", "client_pty", "")
    assert "loginFlow" not in claude_pty and claude_pty["transport"] == "client_pty"
    assert "loginFlow" not in _build_login_request("claude", "", "", "device_auth")


# ---------------------------------------------------------------------------
# Audit claim G (verification-confirmed): three narrow disclose-don't-forbid fixes.
# ---------------------------------------------------------------------------


def test_effort_field_is_the_single_source_over_an_embedded_target_effort(monkeypatch):
    """Claim 2: target_id carries route identity ONLY; the per-slot effort field
    is the one SSOT (D1/6.3). An effort embedded in the spec must never win."""
    _clear_legacy(monkeypatch)
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    from ouroboros.review_execution import (
        AgentSessionReviewExecutor,
        ReviewAssignment,
        ReviewRouteKind,
    )
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    request = ReviewRequest(surface="scope_review", goal="g")
    # Field says max; the target embeds :low. The field must win.
    slot = ReviewSlot(slot_id="s", model="codex=gpt-5.6-sol:low", effort="max",
                      route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex=gpt-5.6-sol:low")
    route = AgentSessionReviewExecutor(
        ReviewAssignment(request=request, slot=slot))._session_route()
    assert (route.route_id, route.model, route.effort) == ("codex", "gpt-5.6-sol", "max")
    # Empty field → empty effort (embedded value is dropped, not resurrected).
    bare = ReviewSlot(slot_id="s2", model="x", effort="",
                      route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex=gpt-5.6-sol:low")
    assert AgentSessionReviewExecutor(
        ReviewAssignment(request=request, slot=bare))._session_route().effort == ""


def test_all_delegated_triad_writes_no_fallback_record_and_reaches_acceptance(monkeypatch):
    """Owner R2: when every triad row is delegated, task acceptance RUNS those
    rows — there is no API-default substitution to disclose, no durable
    fallback record, and the retired disclosure apparatus is gone from the
    module. The save check still validates (400 on malformed) and stays quiet."""
    import pathlib

    from ouroboros.config import DATA_DIR
    from ouroboros.reviewer_slot_config import (
        project_reviewer_slots_into_env,
        reviewer_slot_save_check,
        triad_delivery_slots,
    )

    payload = {
        "triad": [{"slot_id": "t1", "route": {"kind": "agent_session", "target_id": "codex"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "agent_session", "target_id": "codex"}}],
        "advisory": {"enabled": True, "route": {"kind": "agent_session", "target_id": "codex"}},
    }
    # R12: the FIRST save that makes the triad retrieve discloses once, with the
    # measured numbers and the rows; a save that keeps it retrieving is silent.
    disclosure = reviewer_slot_save_check(json.dumps(payload))
    assert "t1 (agent session codex" in disclosure and "≈12 s" in disclosure and "$0.07" in disclosure
    assert reviewer_slot_save_check(json.dumps(payload), previous_raw="") == disclosure
    assert reviewer_slot_save_check(json.dumps(payload), previous_raw=json.dumps(payload)) == ""
    _set_structured(monkeypatch, payload)
    project_reviewer_slots_into_env()
    assert not (pathlib.Path(DATA_DIR) / "state" / "reviewer_slot_api_fallback.json").exists()
    slots = triad_delivery_slots(role_hint="task acceptance")
    assert [(s.slot_id, s.route.value, s.session_target) for s in slots] == [("t1", "agent_session", "codex")]
    with pytest.raises(ValueError, match="triad needs at least one slot"):
        reviewer_slot_save_check(json.dumps({**payload, "triad": []}))


# The acceptance API-pin apparatus retired with owner R2/R12 (2026-09-01): its
# helpers lived in `reviewer_slot_config` and their one importer was
# `claudexor_daemon` (both cleared at a3599ecd; the fallback record
# `reviewer_slot_api_fallback.json` had no writer but
# `_record_api_fallback_substitution`). A surviving module attribute is the
# hook a fallback would grow back on.
_RETIRED_API_PIN_NAMES = (
    "_fallback_warning_text",
    "_record_api_fallback_substitution",
    "api_fallback_disclosure",
    "reviewer_slot_api_fallback_warning",
)


def test_the_retired_acceptance_api_pin_apparatus_is_gone():
    from ouroboros import claudexor_daemon, reviewer_slot_config

    assert [(module.__name__, name) for module in (reviewer_slot_config, claudexor_daemon)
            for name in _RETIRED_API_PIN_NAMES if hasattr(module, name)] == []


def test_mixed_triad_reaches_acceptance_in_row_order(monkeypatch):
    """A session row beside an api row: acceptance carries BOTH, in the owner's
    order, each with its own delivery — the mutation that proves no row is
    filtered out of the panel any more."""
    from ouroboros.reviewer_slot_config import triad_delivery_slots

    payload = {
        "triad": [
            {"slot_id": "t1", "route": {"kind": "agent_session", "target_id": "codex"}},
            {"slot_id": "t2", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-luna"}},
        ],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-terra"}}],
        "advisory": {"enabled": True, "route": {"kind": "api"}},
    }
    _set_structured(monkeypatch, payload)
    slots = triad_delivery_slots(role_hint="task acceptance")
    assert [(s.slot_id, s.route.value, s.model) for s in slots] == [
        ("t1", "agent_session", "codex"), ("t2", "api_chat", "openai/gpt-5.6-luna"),
    ]
    assert [s.retrieves for s in slots] == [True, False]


def test_advisory_disabled_is_a_standing_owner_decision(monkeypatch):
    _set_structured(monkeypatch)
    from ouroboros.tools.claude_advisory_review import (
        advisory_review_route,
        advisory_slot_enabled,
    )

    assert advisory_slot_enabled() is False
    assert advisory_review_route() == "agent_session"
    assert advisory_slot_config().effort == "low"


def test_runs_as_records_applied_facts_never_requested_as_applied():
    """(c) The «выполняется как» block renders APPLIED facts from the run's own
    telemetry receipt: authRoute.profileId + effectiveAccess + the resolved
    model. When telemetry predates the receipt, the record shows ABSENCE —
    never the requested config dressed up as applied."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.reviewer_slot_config import (
        record_reviewer_slot_executions,
        reviewer_slot_last_executions,
    )

    slot = ReviewSlot(slot_id="t_applied", model="codex=gpt-5.6-sol", effort="xhigh",
                      route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex=gpt-5.6-sol", session_profile="pinned-acct")
    # Receipt PRESENT: applied account/access/model all land in `effective`.
    with_receipt = SimpleNamespace(slot_id="t_applied", status="ok", usage={
        "delegated_route": "codex", "resolved_model": "gpt-5.6-sol",
        "applied_profile": "koshak", "applied_access": "readonly",
        "verdict_method": "structured",
    })
    record_reviewer_slot_executions("multi_model_review", [with_receipt], {"t_applied": slot})
    row = reviewer_slot_last_executions()["t_applied"]
    assert row["effective"]["profile_id"] == "koshak"
    assert row["effective"]["access"] == "readonly"
    assert row["effective"]["model"] == "gpt-5.6-sol"
    # requested stays REQUESTED, distinct from applied: the pin the owner asked
    # for is visible beside the account that actually ran.
    assert row["requested"]["profile_id"] == "pinned-acct"
    assert row["requested"]["session_target"] == "codex=gpt-5.6-sol"

    # Receipt ABSENT (old telemetry): applied keys are ABSENT, and the session
    # model is EMPTY — the requested model must not masquerade as applied.
    without_receipt = SimpleNamespace(slot_id="t_applied", status="ok", usage={
        "delegated_route": "codex",
    })
    record_reviewer_slot_executions("multi_model_review", [without_receipt], {"t_applied": slot})
    bare = reviewer_slot_last_executions()["t_applied"]
    assert "profile_id" not in bare["effective"]
    assert "access" not in bare["effective"]
    assert bare["effective"]["model"] == ""  # absence, not slot.model
    assert bare["requested"]["model"] == "codex=gpt-5.6-sol"  # still shown as requested


def test_runner_facts_carry_the_applied_receipt_fields():
    """The session runner surfaces authRoute.profileId/effectiveAccess from the
    summary — the one source settle_run (D29) reads too."""
    import inspect

    from ouroboros import review_execution

    source = inspect.getsource(review_execution.run_delegated_review_session)
    assert '"applied_profile"' in source and "authRoute" in source
    assert '"applied_access"' in source and "effectiveAccess" in source


def test_malformed_advisory_route_raises_typed_not_attributeerror(monkeypatch):
    """A non-dict advisory route must be a ValueError, not an AttributeError.

    The commit gate's fail-closed branch and reviewer_slot_config_error's
    callers all treat this parser as the TYPED authority and catch ValueError
    only; `(raw.get("route") or {}).get(...)` let a string/list route raise
    AttributeError straight through those handlers.
    """
    from ouroboros import reviewer_slot_config as rsc

    base_rows = {
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "m"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "m"}}],
    }
    for bad_route in ("notadict", [1, 2], 7):
        raw = json.dumps({**base_rows, "advisory": {"route": bad_route}})
        with pytest.raises(ValueError, match="advisory route must be an object"):
            rsc.parse_reviewer_slots(raw)
        monkeypatch.setenv(rsc.REVIEWER_SLOTS_ENV, raw)
        assert "advisory route must be an object" in rsc.reviewer_slot_config_error()

    good = json.dumps({**base_rows, "advisory": {"route": {"kind": "agent_session", "target_id": "codex"}}})
    monkeypatch.setenv(rsc.REVIEWER_SLOTS_ENV, good)
    assert rsc.reviewer_slot_config_error() == ""
    assert rsc.parse_reviewer_slots(good).advisory.target_id == "codex"


def test_last_execution_carries_typed_failure_facts():
    """B1: the last-execution projection keeps the typed failure facts a failed slot
    carried (failure_code / reset_at / transport_status / http_status) so a later
    health surface (B4-lite) can read them; a healthy row grows no placeholder keys."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.reviewer_slot_config import (
        record_reviewer_slot_executions,
        reviewer_slot_last_executions,
    )

    dead_slot = ReviewSlot(slot_id="t_dead", model="cursor=grok", effort="high",
                           route=ReviewRouteKind.AGENT_SESSION, session_target="cursor=grok")
    dead = SimpleNamespace(slot_id="t_dead", status="error", usage={},
                           failure_code="subscription_window_exhausted",
                           reset_at="2030-01-01T00:00:00Z", http_status=429,
                           transport_status="provider_transport_error")
    ok_slot = ReviewSlot(slot_id="t_alive", model="m/a", effort="high",
                         route=ReviewRouteKind.API_CHAT)
    alive = SimpleNamespace(slot_id="t_alive", status="ok", usage={})
    record_reviewer_slot_executions(
        "multi_model_review", [dead, alive], {"t_dead": dead_slot, "t_alive": ok_slot})
    rows = reviewer_slot_last_executions()
    assert rows["t_dead"]["failure_code"] == "subscription_window_exhausted"
    assert rows["t_dead"]["reset_at"] == "2030-01-01T00:00:00Z"
    assert rows["t_dead"]["transport_status"] == "provider_transport_error"
    assert rows["t_dead"]["http_status"] == 429
    for key in ("failure_code", "reset_at", "transport_status", "http_status"):
        assert key not in rows["t_alive"]
