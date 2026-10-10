"""Delivery mechanics of the delegated review session.

Split by theme out of ``tests/test_review_agent_session_route.py``. This module
owns what a delivered session may claim: error actors over verdicts, typed
terminal refusals, timeouts, truncated-output resolution, durable invocation
custody, started/pending recovery, retry replay and spend reconciliation.
"""

import json

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.review_execution import (
    REVIEW_SESSION_ROUTE_ENV,
)
from ouroboros.review_substrate import (
    run_review_request,
)
from ouroboros.triad_review import empty_array_is_verified_clean

from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport
from tests._review_session_route_shared import fake_route as __fake_route

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport
fake_route = __fake_route

from tests._review_session_route_shared import (
    FakeGateway,
    FakeLLM,
    _agent_request,
    _agent_slot,
    _run_session_directly,
    _terminal_detail,
)
from tests._usage_store_testing import ledger_rows

# ---------------------------------------------------------------------------
# Delivery mechanics
# ---------------------------------------------------------------------------


def test_failed_session_state_is_an_error_actor_not_a_verdict(tmp_path, fake_route):
    fake_route.detail = _terminal_detail("partial…", state="failed")
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "ended failed" in actor["error"]


def test_an_untyped_engine_failure_keeps_its_reported_words_as_a_record_field(tmp_path, fake_route):
    """An engine failure with `code: null` still says WHY in `safeMessage`. The host's
    `run_failed` is only the run's terminal STATE; the words ride beside it as the
    `reported_cause` field instead of surviving only inside the `error` prose."""
    words = "Selected model is at capacity. Please try a different model."
    detail = _terminal_detail("", state="failed")
    detail["summary"]["failure"] = {"phase": "harness", "category": "harness_error",
                                    "code": None, "safeMessage": words}
    fake_route.detail = detail
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    actor = result.actors[0]
    assert (actor["status"], actor["failure_code"], actor["reported_cause"]) == (
        "error", "run_failed", words)
    # Other direction: a failure the engine said nothing about, and a delivered review,
    # carry no words.
    fake_route.detail = _terminal_detail("partial…", state="failed")
    custody._CUSTODY.clear()
    silent = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path / "silent", llm=FakeLLM()).actors[0]
    assert (silent["failure_code"], silent["reported_cause"]) == ("run_failed", "")
    fake_route.detail = _terminal_detail("[]", conformance="passed")
    custody._CUSTODY.clear()
    delivered = run_review_request(_agent_request(), slots=[_agent_slot()],
                                   drive_root=tmp_path / "ok", llm=FakeLLM()).actors[0]
    assert (delivered["status"], delivered["reported_cause"]) == ("ok", "")


def test_applied_access_is_the_receipt_alone_never_the_request_echoed_back(tmp_path, fake_route):
    """`applied_access` promises APPLIED facts, verbatim from the run's own telemetry
    receipt. The daemon computes `access` as `effectiveAccess ?? the client's own parsed
    request`, so falling back to it published our ASK as if the engine had confirmed it —
    the same non-witness `_widened_access` already refuses to read."""
    detail = _terminal_detail("[]", conformance="passed")
    detail["summary"]["access"] = "workspace_write"   # the request, echoed
    fake_route.detail = detail
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    assert result.actors[0]["usage"]["applied_access"] == ""

    detail = _terminal_detail("[]", conformance="passed")
    detail["summary"]["effectiveAccess"] = "readonly"  # the derived witness
    detail["summary"]["access"] = "workspace_write"
    fake_route.detail = detail
    custody._CUSTODY.clear()
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path / "b", llm=FakeLLM())
    assert result.actors[0]["usage"]["applied_access"] == "readonly"


def _exhausted_window_detail():
    """A terminal whose RunFailure states a spent subscription window, verbatim in the
    engine's own shape (`RunFailureCode` + the STRUCTURAL `resetsAt`)."""
    detail = _terminal_detail("", state="failed")
    detail["summary"]["failure"] = {
        "phase": "routing", "category": "harness_unavailable",
        "code": "subscription_window_exhausted",
        "safeMessage": "every credential profile for this route is spent",
        "resetsAt": "2030-01-01T00:00:00Z",
        "nextActions": ["wait for the window to reopen"],
    }
    return detail


def test_a_typed_terminal_refusal_keeps_its_code_and_its_reset_time(tmp_path, fake_route):
    """The engine says WHY in a typed RunFailure. Flattening it into prose — and
    truncating that prose at 500 chars — threw away both the `code` a caller
    classifies on and the `resetsAt` it is meant to schedule against."""
    from ouroboros.gateways.claudexor import ClaudexorSubscriptionWindowExhausted
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    fake_route.detail = _exhausted_window_detail()
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(),
                         call_id="c-window", call_type="scope_review",
                         custody_root=tmp_path),
        llm=FakeLLM(),
    )
    with pytest.raises(ClaudexorSubscriptionWindowExhausted) as excinfo:
        executor.execute()
    assert excinfo.value.code == "subscription_window_exhausted"
    assert excinfo.value.reset_at == "2030-01-01T00:00:00Z"


def test_a_typed_refusal_is_not_relaunched_into_a_second_billed_session(tmp_path, fake_route):
    """The P3 slot rail is allowed two physical sends, for a transport transient or a
    format repair. A typed Claudexor refusal is neither: it says "this transport is not
    usable", so the second send is a deterministic re-refusal that spends vendor money
    for zero extra verdicts."""
    fake_route.detail = _exhausted_window_detail()
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    assert sum(len(inst.start_requests) for inst in fake_route.instances) == 1
    actor = result.actors[0]
    assert actor["status"] == "error"
    # B1: the typed facts ride the record as FIELDS, never as substrings of the
    # prose — the code, the healing instant and the transport class all survive.
    assert actor["failure_code"] == "subscription_window_exhausted"
    assert actor["reset_at"] == "2030-01-01T00:00:00Z"
    assert actor["transport_status"] == "provider_transport_error"


def test_timeout_cancels_the_run_and_fails_typed(tmp_path, fake_route):
    """The nanny owns the time cap: a run that never terminates is cancelled
    through the verified-cancel path and the slot fails as an ordinary timeout.
    Driven at the executor (the coordinator's own queue wait shares the same
    clock, so an end-to-end race would test the scheduler, not the cap)."""
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    fake_route.nonterminal = True
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(timeout_sec=1),
                         call_id="c-timeout", call_type="scope_review",
                         custody_root=tmp_path),
        llm=FakeLLM(),
    )
    with pytest.raises(TimeoutError):
        executor.execute()
    assert any(reason == "review_slot_timeout"
               for _rid, reason in fake_route.instances[0].cancels)


def test_truncated_primary_output_is_resolved_from_the_full_artifact(tmp_path, fake_route):
    """D7: the verdict is read from the FULL artifact, never a bounded preview."""
    full = "narrative " * 10 + "\n[]\nNO_FINDINGS"
    fake_route.manifest_capabilities = {}
    fake_route.detail = _terminal_detail(full[:20], truncated=True, path="primary.md",
                                         reported_bytes=len(full.encode()))
    fake_route.artifact_bytes = full.encode()
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    actor = result.actors[0]
    assert fake_route.instances[0].artifact_gets == [("run-1", "primary.md")]
    assert actor["status"] == "ok"
    assert empty_array_is_verified_clean(actor["raw_text"])


def test_unresolvable_truncated_output_refuses_instead_of_judging_a_preview(tmp_path, fake_route):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    fake_route.detail = _terminal_detail("head…", truncated=True, path="primary.md",
                                         reported_bytes=999_999)
    fake_route.artifact_error = ClaudexorUnavailable("http_410", "reclaimed", status_code=410)
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "never read from a preview" in actor["error"]
    # And it is refused ONCE. The session SUCCEEDED and was fully billed; only
    # reading its transcript back failed, deterministically (the artifact is
    # reclaimed — a second identical fetch cannot find it). Relaunching bought a
    # second billed session and no second verdict.
    assert sum(len(inst.start_requests) for inst in fake_route.instances) == 1


def test_transport_retry_reuses_the_pending_invocation_id(tmp_path, fake_route):
    """Job-4 scheme: an indefinite start failure leaves the invocation PENDING,
    and the slot's permitted retry presents the SAME wire key with the same
    body, so the engine can return the run it already accepted."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    fake_route.start_error = ClaudexorUnavailable("daemon_unreachable", "boom", status_code=0)
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    assert result.actors[0]["status"] == "ok"  # the retry launched and finished
    keys = [k for inst in fake_route.instances for k in inst.start_keys]
    assert len(keys) == 2 and keys[0] == keys[1]
    bodies = [b for inst in fake_route.instances for b in inst.start_requests]
    assert bodies[0] == bodies[1]  # byte-identical replay, maxSeconds included



def _lineage_scope(*, skill_review=False):
    from ouroboros.usage_accounting import UsageScope

    review = ({
        "category": "skill_review_review", "source": "review_substrate",
        "review_skill": "happy_farm", "review_wave_id": "wave-restart",
        "review_slot_id": "skill-triad-2",
    } if skill_review else {})
    return UsageScope(
        task_id="t-agent", root_task_id="t-root", parent_task_id="t-parent", **review,
    )


def _custody_rows(drive_root):
    return [json.loads(line) for line in
            custody.event_log_path(drive_root).read_text().splitlines() if line.strip()]


def _seed_started_review_invocation(
    drive_root, *, invocation_id="inv-started", request_task_id="t-b",
    custody_task_id="t-b", run_id="run-started",
):
    """Write the exact request + STARTED facts an interrupted retry recovers."""
    route_id = "stored-review"
    request = {
        "prompt": "review this",
        "instructions": "stored review instructions",
        "authPreference": "subscription",
        "mode": "ask",
        "access": "readonly",
        "scope": {"kind": "project", "root": "/tmp/fake-repo"},
        "harnesses": [route_id],
        "primaryHarness": route_id,
        "maxSeconds": 30,
        "model": "stored-model",
        "effort": "xhigh",
        "outputSchema": {"type": "object"},
    }
    assert custody.record_start_requested(
        drive_root, run_id="", task_id=request_task_id,
        idempotency_key="stored-logical-key", invocation_id=invocation_id,
        max_seconds=30, request=request, project_id="proj-owned",
        project_owned=True, route=route_id, surface="scope_review",
        slot_id="scope_slot_1", root_task_id="stored-root",
        parent_task_id="stored-parent",
    )
    entry = custody.RunCustody(
        run_id=run_id, task_id=custody_task_id, route_id=route_id,
        model="stored-model", project_id="proj-owned", project_owned=True, source="review_substrate",
        root_task_id="stored-root", parent_task_id="stored-parent",
        ledger_root=str(drive_root), idempotency_key="stored-logical-key",
        invocation_id=invocation_id,
    )
    assert custody.record_started(drive_root, entry, shape={
        "effort": "xhigh", "access": "readonly", "mode": "ask",
        "isolation": "", "delegated": False, "root": "/tmp/fake-repo",
        "surface": "scope_review", "slot_id": "scope_slot_1",
    })


def test_started_invocation_recovery_reuses_exact_durable_custody(
    tmp_path, fake_route, monkeypatch,
):
    """#167: an already-STARTED retry is wait-only.

    It reuses the original custody/request identity, ignores current route and
    quota drift, and never writes a second STARTED row.
    """
    from ouroboros import subagents

    invocation_id = "inv-started-happy"
    _seed_started_review_invocation(tmp_path, invocation_id=invocation_id)
    custody._CUSTODY.clear()  # prove recovery from the durable rows, not the memo
    state = {"pending_invocation_id": invocation_id}

    monkeypatch.setenv(REVIEW_SESSION_ROUTE_ENV, "drifted-route=drifted-model:low")
    health_calls = []

    def _health_must_not_run(*args, **kwargs):
        health_calls.append((args, kwargs))
        raise AssertionError("route health is admission, not recovery")

    monkeypatch.setattr(subagents, "route_health", _health_must_not_run)
    fake_route.detail = _terminal_detail(
        '{"findings": []}', conformance="passed", model="stored-model",
    )

    facts = _run_session_directly(tmp_path, retry_state=state)

    gateway = fake_route.instances[-1]
    assert gateway.start_requests == [] and gateway.start_keys == []
    assert health_calls == []
    assert gateway.run_gets == ["run-started"]
    assert gateway.project_lookups == [] and gateway.registrations == []
    assert gateway.removals == ["proj-owned"]
    assert facts["run_id"] == "run-started"
    assert facts["route_id"] == "stored-review"
    assert facts["model"] == "stored-model"
    assert facts["schema_asked"] is True
    assert facts["custody_durable"] is True
    assert facts["idempotent_recovery"] is True
    assert facts["settlement"]["settled"] is True
    assert state == {}

    rows = _custody_rows(tmp_path)
    started = [row for row in rows if row["type"] == custody.STARTED]
    assert len(started) == 1, started
    assert started[0]["route"] == "stored-review"
    assert started[0]["model"] == "stored-model"
    assert started[0]["effort"] == "xhigh"
    assert started[0]["project_id"] == "proj-owned"
    assert started[0]["project_owned"] is True
    assert started[0]["idempotency_key"] == "stored-logical-key"
    assert custody.open_runs(tmp_path) == []


@pytest.mark.parametrize(
    "case,request_owner,custody_owner,expected_lookup",
    [
        ("foreign", "durable-owner", "durable-owner", custody.FOREIGN),
        ("unknown", "claimant", "claimant", custody.UNKNOWN),
        ("durable_owner_mismatch", "durable-owner", "claimant", custody.OWNED),
    ],
)
def test_started_invocation_recovery_refuses_unproven_ownership_without_effects(
    tmp_path, fake_route, monkeypatch, case, request_owner, custody_owner,
    expected_lookup,
):
    """#167: the CURRENT task is claimant, and refusal consumes nothing."""
    from ouroboros import subagents
    from ouroboros.review_execution import ReviewRouteUnavailable

    invocation_id = f"inv-started-{case}"
    _seed_started_review_invocation(
        tmp_path, invocation_id=invocation_id, request_task_id=request_owner,
        custody_task_id=custody_owner,
    )
    custody._CUSTODY.clear()
    before = _custody_rows(tmp_path)
    state = {"pending_invocation_id": invocation_id}
    lookup_calls = []
    real_lookup = custody.lookup

    def _tracked_lookup(drive_root, claimant, run_id):
        lookup_calls.append((drive_root, claimant, run_id))
        if case == "unknown":
            return custody.UNKNOWN, None
        return real_lookup(drive_root, claimant, run_id)

    def _health_must_not_run(*_args, **_kwargs):
        raise AssertionError("unowned recovery reached route health")

    monkeypatch.setattr(custody, "lookup", _tracked_lookup)
    monkeypatch.setattr(subagents, "route_health", _health_must_not_run)

    with pytest.raises(ReviewRouteUnavailable, match="corroborate ownership"):
        _run_session_directly(tmp_path, task_id="claimant", retry_state=state)

    assert [(claimant, run_id) for _drive, claimant, run_id in lookup_calls] == [
        ("claimant", "run-started")
    ]
    if case != "unknown":
        assert real_lookup(tmp_path, "claimant", "run-started")[0] == expected_lookup
    assert state == {"pending_invocation_id": invocation_id}
    assert fake_route.instances == []  # no gateway means no poll, POST, or retirement
    assert _custody_rows(tmp_path) == before
    assert not any(row["type"] in (
        custody.PROJECT_RETIRED, custody.SETTLED, custody.LEDGER_RECORDED,
    ) for row in before)


def test_custody_rows_carry_lineage_from_the_bound_usage_scope(tmp_path, fake_route):
    """#112: BOTH custody writers — the pre-POST request row and the STARTED
    row — carry root/parent from the ambient UsageScope (the coordinator binds
    review_usage_scope per slot thread). Unbound stays EMPTY: the settlement
    layer owns the task_id fallback convention, never these writers."""
    from ouroboros.usage_accounting import usage_scope

    with usage_scope(_lineage_scope()):
        _run_session_directly(tmp_path, task_id="t-agent")

    rows = _custody_rows(tmp_path)
    requested = [r for r in rows if r["type"] == custody.START_REQUESTED]
    started = [r for r in rows if r["type"] == custody.STARTED]
    assert requested and started
    for row in (requested[-1], started[-1]):
        assert row["root_task_id"] == "t-root", row
        assert row["parent_task_id"] == "t-parent", row

    # No ambient scope → empty lineage, never an `or task_id` fallback here.
    custody._CUSTODY.clear()
    _run_session_directly(tmp_path / "unbound", task_id="t-agent")
    unbound = [r for r in _custody_rows(tmp_path / "unbound")
               if r["type"] == custody.STARTED]
    assert unbound and unbound[-1]["root_task_id"] == ""
    assert unbound[-1]["parent_task_id"] == ""


def test_restart_reconciliation_settles_review_spend_to_the_recorded_root(
    tmp_path, fake_route, monkeypatch
):
    """#112 Path A: a run whose worker died before settling is reconciled by
    the SUPERVISOR (no ambient scope). The replayed custody must carry the
    recorded lineage, so the subscription-session ledger row lands on the real
    root "t-root" — not on the review's own task id as a fake root."""
    import ouroboros.usage_accounting as ua
    from ouroboros.usage_accounting import usage_scope

    # The live run's ledger write fails, leaving an unsettled STARTED row.
    with monkeypatch.context() as m:
        m.setattr(ua, "record_subscription_session",
                  lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ledger down")))
        with usage_scope(_lineage_scope(skill_review=True)):
            facts = _run_session_directly(tmp_path, task_id="t-agent")
    assert facts["settlement"]["settled"] is False

    # Restart: the in-process memo is gone and no scope is bound.
    custody._CUSTODY.clear()
    outcomes = custody.reconcile_orphaned_runs(
        tmp_path, running_task_ids=set(), gateway_factory=lambda: FakeGateway(),
    )
    assert [o["action"] for o in outcomes] == ["settle_attempted"]

    ledger = ledger_rows(tmp_path)
    sessions = [r for r in ledger if r.get("kind") == "subscription_session"]
    assert sessions, "reconciliation must write the subscription-session row"
    assert sessions[-1]["task_id"] == "t-agent"
    assert sessions[-1]["root_task_id"] == "t-root", sessions[-1]
    assert sessions[-1]["parent_task_id"] == "t-parent"
    assert (sessions[-1]["category"], sessions[-1]["source"]) == (
        "skill_review_review", "review_substrate",
    )
    assert (
        sessions[-1]["review_skill"], sessions[-1]["review_wave_id"],
        sessions[-1]["review_slot_id"],
    ) == ("happy_farm", "wave-restart", "skill-triad-2")


def test_pending_invocation_recovery_replays_the_recorded_lineage(tmp_path, fake_route):
    """#112 Path B: a start whose POST outcome stayed unknown leaves ONLY the
    START_REQUESTED row, and that record must carry the lineage rather than any
    ambient scope — the stored record is the single source of a replay's facts.

    Who replays it changed with issue #1006: this invocation belongs
    to a REVIEW panel, and the review substrate rejoins it by its own exact key
    (``review_session_custody``). The delegation sweep therefore RETAINS it
    instead of re-POSTing it, which would bind a second reviewer nobody asked
    for. The general recovery-and-replay mechanism for the task's own delegation
    stays pinned in ``tests/test_delegated_run_custody.py``.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.usage_accounting import usage_scope

    fake_route.start_error = ClaudexorUnavailable("daemon_unreachable", "boom", status_code=0)
    state: dict = {}
    with usage_scope(_lineage_scope(skill_review=True)):
        with pytest.raises(ClaudexorUnavailable):
            _run_session_directly(tmp_path, task_id="t-agent", retry_state=state)
    assert state["pending_invocation_id"]

    pending = custody.pending_invocations(tmp_path)
    assert len(pending) == 1
    record = pending[0]
    assert record["root_task_id"] == "t-root"
    assert record["parent_task_id"] == "t-parent"
    assert (record["category"], record["source"]) == (
        "skill_review_review", "review_substrate",
    )
    assert (record["review_skill"], record["review_wave_id"], record["review_slot_id"]) == (
        "happy_farm", "wave-restart", "skill-triad-2",
    )

    # The delegation sweep names the owner and stands down: no POST, no bound
    # run, and the invocation is still there for its panel to rejoin.
    result = custody._recover_pending_invocation(tmp_path, FakeGateway(), record)
    assert result == {
        "invocation_id": record["invocation_id"], "task_id": "t-agent",
        "action": "invocation_retained", "reason": "review_panel_owns_invocation",
    }
    assert [r for r in _custody_rows(tmp_path)
            if r["type"] == custody.STARTED
            and r.get("recovered_from_pending_invocation")] == []
    assert custody.pending_invocations(tmp_path) == pending


def test_retry_replays_the_stored_route_and_registers_nothing_new(tmp_path, fake_route,
                                                                 monkeypatch):
    """A retry replays the STORED invocation, so every fact about it comes from the
    record — not from the environment as it stands at retry time.

    The old order computed the current route's project, key and schema ask BEFORE
    reading the pending invocation, so a retry POSTed the recorded body while checking
    the health of a route the run never used, re-registering a project the original
    attempt already bound, and writing a durable record that contradicted the bytes on
    the wire.
    """
    from ouroboros import subagents
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    # Attempt 1: indefinite failure leaves the invocation PENDING.
    state: dict = {}
    fake_route.start_error = ClaudexorUnavailable("daemon_unreachable", "boom", status_code=0)
    with pytest.raises(ClaudexorUnavailable):
        _run_session_directly(tmp_path, retry_state=state)
    pending = state["pending_invocation_id"]
    assert pending

    # The environment is RECONFIGURED between the attempts.
    monkeypatch.setenv(REVIEW_SESSION_ROUTE_ENV, "other-route=other-model:high")
    before = [len(i.registrations) for i in fake_route.instances]
    health_calls = []

    def _health_must_not_run(*args, **kwargs):
        health_calls.append((args, kwargs))
        raise AssertionError("route health is fresh admission, not idempotent recovery")

    monkeypatch.setattr(subagents, "route_health", _health_must_not_run)

    facts = _run_session_directly(tmp_path, retry_state=state)

    assert facts["idempotent_recovery"] is True
    # The replay ran the ORIGINAL route, not the reconfigured one.
    assert facts["route_id"] == "fake-review"
    retry_gateway = fake_route.instances[-1]
    assert retry_gateway.start_requests[0]["primaryHarness"] == "fake-review"
    assert retry_gateway.start_requests[0]["harnesses"] == ["fake-review"]
    # The original request already passed admission. A fresh health snapshot can
    # drift while the first POST's outcome is unknown, so recovery joins by the
    # stored body/key and never re-admits it.
    assert health_calls == []
    # No project lookup or registration happened on the retry: the original
    # attempt's project rides the record.
    assert retry_gateway.project_lookups == []
    assert retry_gateway.registrations == []
    assert sum(before) == sum(len(i.registrations) for i in fake_route.instances)
    # Same wire key, byte-identical body.
    assert retry_gateway.start_keys == [pending]


def test_retry_refuses_typed_when_the_stored_prompt_diverges(tmp_path, fake_route):
    """The replay sends the RECORDED bytes. If this call describes a different review,
    that is a typed refusal — never a silent review of something else."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_execution import ReviewRouteUnavailable

    state: dict = {}
    fake_route.start_error = ClaudexorUnavailable("daemon_unreachable", "boom", status_code=0)
    with pytest.raises(ClaudexorUnavailable):
        _run_session_directly(tmp_path, retry_state=state, prompt="review THIS")

    with pytest.raises(ReviewRouteUnavailable, match="prompt"):
        _run_session_directly(tmp_path, retry_state=state, prompt="review SOMETHING ELSE")
    with pytest.raises(ReviewRouteUnavailable, match="session root"):
        _run_session_directly(tmp_path, retry_state=state, prompt="review THIS",
                              root="/tmp/other-repo")


def test_definite_refusal_retires_the_registration_it_orphaned(tmp_path, fake_route):
    """A DEFINITE 4xx proves no run bound this registration, so the project this
    start created is retired. Only then: an unknown outcome must never destroy state
    a live run may still be using."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    # This start registers the project itself (nothing pre-existing to reuse).
    fake_route.project_unregistered = True
    fake_route.start_error = ClaudexorUnavailable("bad_request", "nope", status_code=400)
    state: dict = {}
    with pytest.raises(ClaudexorUnavailable):
        _run_session_directly(tmp_path, retry_state=state)

    gateway = fake_route.instances[-1]
    assert gateway.registrations == ["/tmp/fake-repo"]
    assert gateway.removals == ["proj-new"], gateway.removals
    # A definitely refused invocation is retired, never replayed.
    assert "pending_invocation_id" not in state


@pytest.mark.parametrize("code,status", [("daemon_unreachable", 0), ("daemon_busy", 503),
                                       ("daemon_unavailable", 503)])
def test_unknown_outcome_retains_the_registration_and_says_why(tmp_path, fake_route, code, status):
    """A transport error leaves the POST's fate UNKNOWN: a run may be live against
    this registration, so it is RETAINED and the durable row names the reason."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    fake_route.project_unregistered = True
    fake_route.start_error = ClaudexorUnavailable(code, "RPC unavailable", status_code=status)
    state: dict = {}
    with pytest.raises(ClaudexorUnavailable):
        _run_session_directly(tmp_path, retry_state=state)

    assert fake_route.instances[-1].removals == []
    assert [row["invocation_id"] for row in custody.pending_invocations(tmp_path)] == [state["pending_invocation_id"]]
    rows = [json.loads(ln) for ln in
            custody.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    failed = [r for r in rows if r.get("type") == custody.START_FAILED]
    assert failed and failed[-1]["project_retention_reason"] == (
        "start_outcome_unknown_run_may_exist"), failed[-1]


def test_started_run_reports_whether_its_custody_row_landed(tmp_path, fake_route,
                                                            monkeypatch):
    """record_started's answer is a FACT the caller needs: a run whose authoritative
    row did not land is custodied by this process alone, and reporting a plainly
    started run over that state is how a live run becomes unfindable."""
    import ouroboros.delegate_custody as custody_mod

    assert _run_session_directly(tmp_path)["custody_durable"] is True

    monkeypatch.setattr(custody_mod, "record_started", lambda *_a, **_k: False)
    assert _run_session_directly(tmp_path)["custody_durable"] is False


def test_session_is_never_restarted_for_format_repair(tmp_path, fake_route, monkeypatch):
    """5.5: a resend over bad output performs local extraction over the already
    collected transcript — the session is not relaunched."""
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    fake_route.manifest_capabilities = {}
    fake_route.detail = _terminal_detail("prose without a verdict")
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "0")
    llm = FakeLLM(reply="UNEXTRACTABLE")
    from datetime import datetime, timedelta, timezone

    deadline = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(deadline_at=deadline),
                         slot=_agent_slot(transport_timeout_sec=17),
                         call_id="c1", call_type="scope_review",
                         custody_root=tmp_path),
        llm=llm,
    )
    first = executor.execute()
    second = executor.execute()  # the coordinator's permitted resend
    assert len(fake_route.instances) == 1
    assert len(fake_route.instances[0].start_requests) == 1
    assert first.raw_text == second.raw_text == "prose without a verdict"
    assert len(llm.calls) == 2  # local extraction ran each time, no new session
    assert all(0 < call["timeout"] <= 17 for call in llm.calls)
    assert llm.calls[1]["timeout"] <= llm.calls[0]["timeout"]


def test_a_pool_exhausted_terminal_is_typed_like_a_spent_window(tmp_path, fake_route):
    """Cross-repo forward-compat (B1): a newer engine reports a spent credential POOL
    with its own RunFailureCode. A DATED pool gets the same timer-healing semantics and
    exception class — with the ORIGINAL code preserved, never relabelled. An unknown code stays the
    generic typed refusal (fail-open: old engines emit code:null and behave as today)."""
    from ouroboros.gateways.claudexor import (
        ClaudexorSubscriptionWindowExhausted, ClaudexorUnavailable)
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    detail = _exhausted_window_detail()
    detail["summary"]["failure"]["code"] = "credential_pool_exhausted"
    fake_route.detail = detail
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(),
                         call_id="c-pool", call_type="scope_review",
                         custody_root=tmp_path),
        llm=FakeLLM(),
    )
    with pytest.raises(ClaudexorSubscriptionWindowExhausted) as excinfo:
        executor.execute()
    assert excinfo.value.code == "credential_pool_exhausted"
    assert excinfo.value.reset_at == "2030-01-01T00:00:00Z"

    detail = _exhausted_window_detail()
    detail["summary"]["failure"]["code"] = "some_future_code"
    fake_route.detail = detail
    custody._CUSTODY.clear()
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(),
                         call_id="c-unknown", call_type="scope_review",
                         custody_root=tmp_path / "b"),
        llm=FakeLLM(),
    )
    with pytest.raises(ClaudexorUnavailable) as generic:
        executor.execute()
    assert not isinstance(generic.value, ClaudexorSubscriptionWindowExhausted)
    assert generic.value.code == "some_future_code"


def test_pending_retry_replays_the_stored_credential_pin(tmp_path, fake_route):
    """Phase D1 on the RECOVERY path: the stored request is the durable pin
    carrier. A pinned slot whose first attempt died mid-flight must replay as
    PINNED on the wire. Live route health is fresh admission and cannot block
    recovery of a POST whose provider outcome is still unknown."""
    import dataclasses

    from ouroboros import subagents
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.subagents import parse_subagent_harness

    pinned = dataclasses.replace(parse_subagent_harness("fake-review=fake-small"),
                                 profile_id="acct-pinned")

    # Attempt 1: pinned start dies indefinite; the invocation stays PENDING.
    state: dict = {}
    fake_route.start_error = ClaudexorUnavailable("daemon_unreachable", "boom", status_code=0)
    with pytest.raises(ClaudexorUnavailable):
        _run_session_directly(tmp_path, retry_state=state, session_route=pinned)
    assert state["pending_invocation_id"]

    # Attempt 2: the row now reads permanently unavailable (the agy shape).
    fake_route.start_error = None
    fake_route.catalog_entry["status"] = "unavailable"
    fake_route.catalog_entry["enabled"] = False
    health_calls = []

    def _health_must_not_run(*args, **kwargs):
        health_calls.append((args, kwargs))
        raise AssertionError("pending replay must not consult current route health")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subagents, "route_health", _health_must_not_run)
        facts = _run_session_directly(tmp_path, retry_state=state, session_route=pinned)

    assert facts["idempotent_recovery"] is True
    assert health_calls == []
    retry_starts = [r for inst in fake_route.instances for r in inst.start_requests]
    assert retry_starts[-1]["credentialProfileId"] == "acct-pinned"


def test_pending_invocation_checkpoint_precedes_provider_post(
    tmp_path, fake_route, monkeypatch,
):
    from ouroboros.delegate_custody import START_REQUESTED

    checkpoints = []
    original_start = FakeGateway.start_run

    def _start_after_checkpoint(self, request, *, idempotency_key=""):
        assert checkpoints == [idempotency_key]
        return original_start(self, request, idempotency_key=idempotency_key)

    monkeypatch.setattr(FakeGateway, "start_run", _start_after_checkpoint)

    facts = _run_session_directly(
        tmp_path,
        operation_id="op-checkpoint",
        pending_invocation_checkpoint=lambda invocation_id: checkpoints.append(
            invocation_id
        ),
    )

    assert facts["run_id"] == "run-1"
    assert checkpoints == [fake_route.instances[-1].start_keys[0]]
    requested = [
        row for row in _custody_rows(tmp_path)
        if row.get("type") == START_REQUESTED
    ][-1]
    assert requested["surface"] == "scope_review"
    assert requested["slot_id"] == "scope_slot_1"
    assert requested["operation_id"] == "op-checkpoint"

def test_retry_of_a_pinned_session_replays_without_fresh_account_health(
    tmp_path, fake_route, monkeypatch,
):
    """A pending retry replays its admitted pin without consulting live health."""
    from ouroboros import subagents
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.subagents import DelegationRoute

    pinned = DelegationRoute(route_id="fake-review", model="fake-small",
                             effort="low", profile_id="pinned-account")
    state: dict = {}
    fake_route.start_error = ClaudexorUnavailable("daemon_unreachable", "boom", status_code=0)
    with pytest.raises(ClaudexorUnavailable):
        _run_session_directly(tmp_path, retry_state=state, session_route=pinned)
    assert state["pending_invocation_id"]

    # The setting drifts to another route between the attempts.
    monkeypatch.setenv(REVIEW_SESSION_ROUTE_ENV, "other-route=other-model:high")
    health_calls = []

    def _health_must_not_run(*args, **kwargs):
        health_calls.append((args, kwargs))
        raise AssertionError("pending replay must not be blocked by current account health")

    monkeypatch.setattr(subagents, "route_health", _health_must_not_run)

    facts = _run_session_directly(tmp_path, retry_state=state)

    assert health_calls == []
    retry_gateway = fake_route.instances[-1]
    assert retry_gateway.start_requests[0]["credentialProfileId"] == "pinned-account"
    # And the fresh STARTED custody row carries the pin, symmetric with the
    # delegate lane, so the receipt line can disclose a requested-vs-ran drift.
    started = [r for r in _custody_rows(tmp_path) if r["type"] == custody.STARTED]
    assert started and started[-1]["profile_id"] == "pinned-account"
    assert facts["run_id"]


# ---------------------------------------------------------------------------
# #1547 link 3: a `succeeded` seen at the slot deadline is paid evidence
# ---------------------------------------------------------------------------


class _DeadlineWitnessLLM(FakeLLM):
    """Records what the stop axes say at the moment the extraction call is made."""

    def chat(self, **kwargs):
        import time

        from ouroboros.model_wait import dispatch_deadline_remaining_sec

        self.remaining_at_send = dispatch_deadline_remaining_sec()
        self.sent_at = time.monotonic()
        return super().chat(**kwargs)


def _late_success_executor(tmp_path, fake_route, monkeypatch, *, llm, seen_by, window=1.0):
    """A run still `running` through the whole slot budget whose natural
    `succeeded` is first seen either by the spent window's last read or by the
    verify read of the slot-deadline cancel (completion wins).

    The runtime's poll clock is test-local: it moves only by the runtime's own sleeps,
    each really slept so the slot's execution deadline does expire, and the run turns
    `succeeded` on the read the runtime itself makes with its window spent. No setup,
    custody or host stall can then move the success relative to the window. Returns the
    executor and the slot budget each of the runtime's reads was made with."""
    import time
    from types import SimpleNamespace

    from ouroboros import review_execution as rx
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    fake_route.nonterminal = True
    fake_route.manifest_capabilities = {}
    fake_route.detail = _terminal_detail("narrative first\n[]\nNO_FINDINGS")
    clock, reads, observe, original = [0.0], [], rx._observe_session, FakeGateway.get_run

    def sleep(seconds):
        time.sleep(seconds)
        clock[0] += seconds

    def observe_session(gateway, run_id, remaining, streak):
        reads.append(remaining)
        if seen_by == "spent_read" and remaining <= 0:
            FakeGateway.nonterminal = False  # it finished during the window's last sleep
        return observe(gateway, run_id, remaining, streak)

    def get_run(self, run_id, **kw):
        if seen_by == "verify_read" and self.cancels:
            FakeGateway.nonterminal = False
        return original(self, run_id, **kw)

    monkeypatch.setattr(rx, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(rx, "_observe_session", observe_session)
    monkeypatch.setattr(FakeGateway, "get_run", get_run)
    if seen_by == "spent_read":
        # The subject is what the deadline read DOES with a `succeeded` it sees, not whether a
        # loaded test host schedules the read's worker thread inside its 1 ms wall bound: the
        # fake answers synchronously (the bound itself is pinned by the delegate_progress tests).
        from ouroboros import delegate_progress

        monkeypatch.setattr(delegate_progress, "_strict_poll",
                            lambda gateway, run_id, timeout: gateway.get_run(run_id, timeout_sec=timeout))
    return AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(timeout_sec=window),
                         call_id="c-late", call_type="scope_review", custody_root=tmp_path),
        llm=llm), reads


@pytest.mark.parametrize("seen_by, cancels", [("spent_read", []), ("verify_read", ["review_slot_timeout"])])
def test_a_success_first_seen_at_the_slot_deadline_is_accepted_and_extracted_past_the_slot_deadline(
        tmp_path, fake_route, monkeypatch, seen_by, cancels):
    """The deadline discovers `succeeded`: the verdict is accepted
    (``late_success_accepted``) and its light-model extraction runs on the
    operation's own wait — the slot's expired execution deadline no longer says
    ``deadline`` to the send. When the cancel's verify read is what found it,
    the cancel was still written under ``review_slot_timeout``."""
    import time

    from ouroboros.model_wait import execution_deadline_scope

    llm = _DeadlineWitnessLLM()
    executor, reads = _late_success_executor(tmp_path, fake_route, monkeypatch, llm=llm, seen_by=seen_by)
    slot_deadline = time.monotonic() + 1.0
    with execution_deadline_scope(slot_deadline, review_slot_id="scope_slot_1"):
        result = executor.execute()
    assert reads[-1] == 0 and all(budget > 0 for budget in reads[:-1]), \
        "the run read `running` inside its window; the runtime's last read was its spent one"
    assert llm.calls, "the late verdict was extracted, not dropped"
    assert llm.sent_at >= slot_deadline, "the slot deadline had really expired by the extraction send"
    assert llm.remaining_at_send is None, "the expired SLOT deadline must not reach the paid verdict's extraction"
    assert result.usage["late_success_accepted"] is True and result.usage["extraction"]
    assert empty_array_is_verified_clean(result.raw_text)
    assert [reason for _rid, reason in fake_route.instances[0].cancels] == cancels


def test_the_calendar_deadline_still_interrupts_the_late_verdict_phase(tmp_path, fake_route, monkeypatch):
    """Only the slot's execution deadline is lifted for the paid verdict: an
    explicit calendar ``deadline_at`` stays a stop axis of the extraction send."""
    import time

    from ouroboros.model_wait import calendar_scope, execution_deadline_scope

    llm = _DeadlineWitnessLLM()
    executor, _reads = _late_success_executor(tmp_path, fake_route, monkeypatch, llm=llm, seen_by="verify_read")
    with calendar_scope("2000-01-01T00:00:00+00:00"), \
            execution_deadline_scope(time.monotonic() + 1.0, review_slot_id="scope_slot_1"):
        result = executor.execute()
    assert llm.remaining_at_send == 0.0, "a spent calendar deadline is still visible to the send"
    assert result.usage["late_success_accepted"] is True
    assert [reason for _rid, reason in fake_route.instances[0].cancels] == ["review_slot_timeout"]


def test_a_success_read_inside_the_slot_budget_is_not_marked_late(tmp_path, fake_route):
    """The ordinary path keeps its ordinary record: ``late_success_accepted`` is False."""
    result = run_review_request(_agent_request(), slots=[_agent_slot()], drive_root=tmp_path, llm=FakeLLM())
    assert result.actors[0]["status"] == "ok"
    assert result.actors[0]["usage"]["late_success_accepted"] is False
    assert fake_route.instances[0].cancels == []


def test_a_run_cancelled_at_the_slot_deadline_is_not_accepted_and_times_out_typed(tmp_path, fake_route, monkeypatch):
    """A `cancelled`/`failed` terminal the deadline read sees is never a late
    acceptance: the slot ends as the honest typed timeout, as today."""
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    fake_route.nonterminal = True
    original = FakeGateway.get_run

    def get_run(self, run_id, **kw):
        if self.cancels:
            return _terminal_detail("", state="cancelled")
        return original(self, run_id, **kw)

    monkeypatch.setattr(FakeGateway, "get_run", get_run)
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(timeout_sec=1),
                         call_id="c-cancelled", call_type="scope_review", custody_root=tmp_path),
        llm=FakeLLM())
    with pytest.raises(TimeoutError, match="host-cancelled"):
        executor.execute()
    assert "late_success_accepted" not in executor._session_usage


def test_a_started_run_blind_to_its_deadline_stays_in_flight_through_the_real_substrate(
        tmp_path, fake_route, monkeypatch):
    """#1547 link 2, end to end: the run starts, every read of it fails (daemon
    unreachable) through the whole slot budget and the deadline cancel cannot be
    verified either. The drain hands back the slot as ``in_flight`` at its
    deadline; the worker's own settlement, projected by the real substrate from
    ``ReviewPollUnavailable``, keeps that seat ``in_flight`` with its pending
    token and run id — never settled, never a $0 retry."""
    import threading
    from types import SimpleNamespace

    from ouroboros import review_custody as custody_mod
    from ouroboros import review_execution as rx
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_custody import _attempt_key

    def blind(self, run_id, **_kw):
        self.run_gets.append(run_id)
        raise ClaudexorUnavailable("daemon_unreachable", "connection refused", status_code=0)

    def unreachable_cancel(self, run_id, *, reason=""):
        self.cancels.append((run_id, reason))
        raise ClaudexorUnavailable("daemon_unreachable", "connection refused", status_code=0)

    settled, worker_actors = threading.Event(), []
    original_settle = custody_mod._settle_review_attempt

    def spy_settle(entry, slot, actor, **kwargs):
        worker_actors.append(actor)
        try:
            return original_settle(entry, slot, actor, **kwargs)
        finally:
            settled.set()

    monkeypatch.setattr(FakeGateway, "get_run", blind)
    monkeypatch.setattr(FakeGateway, "cancel_run", unreachable_cancel)
    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.05)
    monkeypatch.setattr(custody_mod, "_settle_review_attempt", spy_settle)
    ctx = SimpleNamespace(task_id="t-agent", drive_root=tmp_path, budget_drive_root=tmp_path,
                          pending_events=[], event_queue=None)
    request, slot = _agent_request(retry_key="scope:t-agent"), _agent_slot(timeout_sec=0.5)
    result = run_review_request(request, slots=[slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx)
    drained = result.actors[0]
    assert drained["operation_state"] == "in_flight" and drained["late_result_pending"]
    assert settled.wait(10), "the blind worker never settled"
    gateway, worker = fake_route.instances[0], worker_actors[0]
    assert gateway.start_requests and len(gateway.run_gets) >= 2, "started once, observed repeatedly"
    assert [reason for _rid, reason in gateway.cancels] == ["review_slot_timeout"]
    assert worker.status == "error" and worker.operation_state == "in_flight" and worker.late_result_pending
    assert worker.failure_code == "review_poll_unavailable"
    assert worker.usage["review_failure_phase"] == "delivery"
    assert worker.usage["delegated_run_id"] == "run-1" and worker.usage["pending_invocation_id"]
    key = _attempt_key(request, slot)
    assert ctx._review_pending_invocations[key]["pending_invocation_id"] == worker.usage["pending_invocation_id"]
    assert key not in (getattr(ctx, "_review_settled_attempts", None) or {}), \
        "an unobserved live run is not a replayable settled verdict"
