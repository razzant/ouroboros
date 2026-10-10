"""The review pool's reviewer rows, as every surface RUNS them (PR-3, one transition).

The lane parser that lived here (``parse_reviewer_slots`` and its advisory /
scope / deep-review rows, the default panel, the comma-key projection) left with
the lanes; its frozen copies serve ``review_pool_migration`` alone. What stays:
the pool's configuration-error facade, the session row's own target as the
executor's authority, and the «Выполняется как» projection beside each saved row.
The pool itself (marks, order, delivery, quorum) is ``tests/test_review_pool.py``.
"""
import pytest

from tests.review_pool_rosters import pool_roster, pool_seat


def test_config_error_is_empty_on_absent_or_valid_and_names_a_malformed_catalog(monkeypatch):
    from ouroboros.reviewer_slot_config import reviewer_slot_config_error

    monkeypatch.delenv("OUROBOROS_SUBAGENTS", raising=False)
    assert reviewer_slot_config_error() == ""
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", pool_roster(pool_seat("t_api", "openai/gpt-5.6-luna")))
    assert reviewer_slot_config_error() == ""
    # An empty pool is a configured fact (``pool_empty`` on every surface), never
    # this error: a catalog whose rows carry no review mark reads clean here.
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", pool_roster(
        pool_seat("scout", "openai/gpt-5.6-luna", marked=False)))
    assert reviewer_slot_config_error() == ""
    # A malformed catalog is the parser's row-precise text.
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", "{not json")
    assert reviewer_slot_config_error() != ""


def test_session_executor_prefers_the_slots_own_target(monkeypatch):
    from ouroboros.review_execution import (
        ReviewRouteKind,
        ReviewRouteUnavailable,
        session_route_for_review_slot,
    )
    from ouroboros.review_substrate import ReviewSlot

    monkeypatch.delenv("OUROBOROS_REVIEW_SESSION_ROUTE", raising=False)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)
    slot = ReviewSlot(slot_id="s_owner", model="codex=gpt-5.6-sol", effort="xhigh",
                      route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex=gpt-5.6-sol")
    route = session_route_for_review_slot(slot)
    assert (route.route_id, route.model, route.effort) == ("codex", "gpt-5.6-sol", "xhigh")

    # Without a per-row target the shared-route absence stays a typed refusal.
    bare = ReviewSlot(slot_id="s2", model="m", route=ReviewRouteKind.AGENT_SESSION)
    with pytest.raises(ReviewRouteUnavailable):
        session_route_for_review_slot(bare)


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


def test_effort_field_is_the_single_source_over_an_embedded_target_effort():
    """Claim 2: target_id carries route identity ONLY; the per-slot effort field
    is the one SSOT (D1/6.3). An effort embedded in the spec must never win."""
    from ouroboros.review_execution import (
        ReviewRouteKind,
        session_route_for_review_slot,
    )
    from ouroboros.review_substrate import ReviewSlot
    # Field says max; the target embeds :low. The field must win.
    slot = ReviewSlot(slot_id="s", model="codex=gpt-5.6-sol:low", effort="max",
                      route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex=gpt-5.6-sol:low")
    route = session_route_for_review_slot(slot)
    assert (route.route_id, route.model, route.effort) == ("codex", "gpt-5.6-sol", "max")
    # Empty field → empty effort (embedded value is dropped, not resurrected).
    bare = ReviewSlot(slot_id="s2", model="x", effort="",
                      route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex=gpt-5.6-sol:low")
    assert session_route_for_review_slot(bare).effort == ""


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
    """The session runner shares final-attempt identity with settlement;
    effectiveAccess remains an independent engine fact."""
    import inspect

    from ouroboros import review_execution

    source = inspect.getsource(review_execution.run_delegated_review_session)
    assert '"applied_profile"' in source and "final_attempt_facts" in source
    assert '"applied_access"' in source and "effectiveAccess" in source



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
