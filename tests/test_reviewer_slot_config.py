"""Reviewer-slot SSOT (phase 6.1): structured parse, legacy migration, projection.

The one structured setting supersedes the comma-lists; the comma keys survive
only as a runtime projection for the API-pinned surfaces (D15). Malformed
configuration REFUSES typed — an unknown token must never silently pick a
transport, in either direction.
"""
import json

import pytest

from ouroboros.reviewer_slot_config import (
    REVIEWER_SLOTS_ENV,
    SCOPE_SLOT_LIMIT,
    TRIAD_SLOT_LIMIT,
    advisory_slot_config,
    commit_triad_rows,
    load_reviewer_slot_config,
    parse_reviewer_slots,
    project_reviewer_slots_into_env,
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


@pytest.mark.parametrize("mutate,fragment", [
    (lambda p: p.__setitem__("triad", []), "at least one slot"),
    (lambda p: p.__setitem__("scope", []), "at least one slot"),
    (lambda p: p["triad"][0]["route"].__setitem__("kind", "codex"), "unknown route kind"),
    (lambda p: p["triad"][0]["route"].__setitem__("target_id", ""), "target_id is empty"),
    (lambda p: p["triad"][1]["route"].__setitem__("target_id", "codex::gpt-5.6"), "'::'"),
    (lambda p: p["triad"][1].__setitem__("slot_id", "t_api"), "appears twice"),
    (lambda p: p["triad"][0].__setitem__("effort", "enormous"), "unknown effort"),
    (lambda p: p.__setitem__("bogus", 1), "unknown top-level keys"),
    (lambda p: p.__setitem__("triad", p["triad"] * 6), f"limit is {TRIAD_SLOT_LIMIT}"),
], ids=["empty-triad", "empty-scope", "vendor-kind", "empty-target", "double-colon",
        "dup-slot-id", "bad-effort", "unknown-key", "triad-cap"])
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


def test_legacy_comma_lists_read_as_api_slots(monkeypatch):
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m/one,m/two")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "m/scope")
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "medium")
    monkeypatch.setenv("OUROBOROS_EFFORT_SCOPE_REVIEW", "xhigh")
    config = load_reviewer_slot_config()
    assert config.source == "legacy"
    # Historical positional ids, so pre-migration receipts keep lining up.
    assert [(r.slot_id, r.kind, r.target_id, r.effort) for r in config.triad] == [
        ("slot_1", "api_chat", "m/one", "medium"),
        ("slot_2", "api_chat", "m/two", "medium"),
    ]
    assert [(r.slot_id, r.effort) for r in config.scope] == [("scope_slot_1", "xhigh")]
    assert config.advisory.enabled is True and config.advisory.kind == "api"


def test_legacy_phase5_route_envs_are_honored(monkeypatch):
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m/one,m/two")
    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "api_chat,agent_session")
    monkeypatch.setenv("OUROBOROS_ADVISORY_REVIEW_ROUTE", "agent_session")
    config = load_reviewer_slot_config()
    session_row = config.triad[1]
    assert session_row.kind == "agent_session"
    # A legacy session row has no per-row target: delivery stays on the shared
    # session-route key, which is exactly the phase-5 behavior.
    assert session_row.session_target == ""
    assert config.advisory.kind == "agent_session"


def test_legacy_bad_route_token_still_refuses(monkeypatch):
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m/one")
    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "carrier-pigeon")
    with pytest.raises(ValueError, match="unknown review route"):
        commit_triad_rows()


# ---------------------------------------------------------------------------
# Runtime projection for the API-pinned surfaces (D15).
# ---------------------------------------------------------------------------


def test_projection_exposes_only_api_rows(monkeypatch):
    _set_structured(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "stale/comma-value")
    project_reviewer_slots_into_env()
    import os

    assert os.environ["OUROBOROS_REVIEW_MODELS"] == "openai/gpt-5.6-luna"
    assert os.environ["OUROBOROS_SCOPE_REVIEW_MODELS"] == "openai/gpt-5.6-terra"


def test_projection_all_delegated_triad_floors_to_defaults(monkeypatch):
    payload = json.loads(json.dumps(_STRUCTURED))
    payload["triad"] = [payload["triad"][1]]
    _set_structured(monkeypatch, payload)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "stale/comma-value")
    project_reviewer_slots_into_env()
    import os

    from ouroboros.config import SETTINGS_DEFAULTS

    # The API surfaces (plan review, task acceptance, skill review — D15) fall
    # back to the shipped defaults, never to zero reviewers or a stale comma key.
    assert os.environ["OUROBOROS_REVIEW_MODELS"] == str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"])


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


@pytest.mark.serial
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
    assert body["limits"] == {"triad": TRIAD_SLOT_LIMIT, "scope": SCOPE_SLOT_LIMIT, "advisory": 1}
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


def test_legacy_session_row_migrates_to_the_shared_harness_not_the_model(monkeypatch):
    """Claim 3: a legacy phase-5 session row delivered via the SHARED session
    route; its comma-list model was never a harness. The migration must map its
    identity to that harness, not leave a model in the harness-shaped field."""
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "openai/gpt-5.6-luna,anthropic/claude-sonnet-5")
    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "api_chat,agent_session")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "codex=gpt-5.6-sol:xhigh")
    config = load_reviewer_slot_config()
    api_row, session_row = config.triad
    # The api row keeps its model as identity.
    assert (api_row.kind, api_row.target_id) == ("api_chat", "openai/gpt-5.6-luna")
    # The session row's identity is the shared HARNESS, effort in the field.
    assert session_row.kind == "agent_session"
    assert session_row.target_id == "codex=gpt-5.6-sol"
    assert session_row.session_target == "codex=gpt-5.6-sol"
    assert session_row.effort == "xhigh"
    # The model name never leaks into the harness-shaped field.
    assert "gpt-5.6-luna" not in session_row.target_id
    assert "claude-sonnet-5" not in session_row.target_id


def test_legacy_session_row_with_no_shared_route_is_empty_not_a_model(monkeypatch):
    """A legacy session row with no shared route configured was undeliverable;
    the migration keeps that honest (empty target) rather than a fake harness."""
    monkeypatch.delenv(REVIEWER_SLOTS_ENV, raising=False)
    _clear_legacy(monkeypatch)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "openai/gpt-5.6-luna")
    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "agent_session")
    session_row = load_reviewer_slot_config().triad[0]
    assert session_row.kind == "agent_session"
    assert session_row.target_id == ""
    assert session_row.session_target == ""


def test_all_delegated_commit_surface_discloses_the_api_fallback(monkeypatch):
    """Claim 1: when every commit/scope row is delegated the API-pinned surfaces
    fall back to default models and spend budget — disclosed, never silent, and
    the save is NOT blocked (recommendation A, reversible default).

    The disclosure is NEUTRAL routing information, not advice: an
    all-subscription triad IS the ratified default (D-3), so the sentence must
    not tell the owner to keep an API row and undo it."""
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.reviewer_slot_config import (
        api_fallback_disclosure,
        parse_reviewer_slots,
        project_reviewer_slots_into_env,
        reviewer_slot_api_fallback_warning,
    )

    payload = {
        "triad": [{"slot_id": "t1", "route": {"kind": "agent_session", "target_id": "codex"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "agent_session", "target_id": "codex"}}],
        "advisory": {"enabled": True, "route": {"kind": "agent_session", "target_id": "codex"}},
    }
    disclosure = api_fallback_disclosure(parse_reviewer_slots(json.dumps(payload)))
    assert disclosure["triad"] == str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"]).split(",")
    assert disclosure["scope"] == str(SETTINGS_DEFAULTS["OUROBOROS_SCOPE_REVIEW_MODELS"]).split(",")

    _set_structured(monkeypatch, payload)
    warning = reviewer_slot_api_fallback_warning()
    assert warning and "commit review" in warning and "scope review" in warning
    # It names both halves of the routing fact: what moved to subscriptions,
    # and which surfaces the API still serves with which models.
    assert "agent subscription" in warning
    assert "Plan review, task acceptance and skill review" in warning
    assert str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"]).split(",")[0] in warning
    # The retired advice: telling the owner to keep an API reviewer row
    # contradicts the ratified all-subscription default.
    assert "Keep at least one API" not in warning
    assert "coding agent" not in warning
    # The projection still keeps the reviewers WORKING (defaults present), and
    # writes the durable record — disclose-and-continue, never a block.
    import os as _os
    project_reviewer_slots_into_env()
    assert _os.environ["OUROBOROS_REVIEW_MODELS"] == str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"])
    import pathlib

    from ouroboros.config import DATA_DIR
    record = pathlib.Path(DATA_DIR) / "state" / "reviewer_slot_api_fallback.json"
    assert record.is_file()


def test_one_api_row_avoids_the_fallback_and_the_warning(monkeypatch):
    """The mutation that proves the disclosure is scoped: a single surviving API
    row in each group means no fallback and no warning."""
    from ouroboros.reviewer_slot_config import (
        api_fallback_disclosure,
        parse_reviewer_slots,
        reviewer_slot_api_fallback_warning,
    )

    payload = {
        "triad": [
            {"slot_id": "t1", "route": {"kind": "agent_session", "target_id": "codex"}},
            {"slot_id": "t2", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-luna"}},
        ],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-terra"}}],
        "advisory": {"enabled": True, "route": {"kind": "api"}},
    }
    assert api_fallback_disclosure(parse_reviewer_slots(json.dumps(payload))) == {}
    _set_structured(monkeypatch, payload)
    assert reviewer_slot_api_fallback_warning() == ""


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
