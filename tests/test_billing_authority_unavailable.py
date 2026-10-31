"""An unestablished billing authority is a technical fact, never money (owner 2026-10-14:
ab53 = A, 00a5 = A).

Each test drives an actual consumer under the safe_test boundary: the real ledger
reservation, ``run_llm_loop`` into the exact pause on the technical rail, the
supervisor park, the strict Resume grant and the restored loop, late session
settlement, the delegated START gate, and the model-wait control that may wait
only on positive contention. No provider, engine or live data is reached.
"""

from __future__ import annotations

import os
import queue as queue_mod
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.settings_integrity import task_settings_snapshot
from tests._store_shared import unreadable_result
from tests._usage_store_testing import ledger_rows
from tests.test_billing_group import _scope, _spend


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """The isolated ledger root ``test_billing_group`` uses."""
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    (root / "state").mkdir(parents=True)
    ua._reset_task_cache_splits()
    return root


BINDING = {"billing_group_id": "P", "billing_group_limit_usd": 20.0,
           "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "cap-1"}


def _request():
    return ua.AttemptRequest(model="m", provider="openai", reservation_usd=0.01)


def _kinds(root, kind):
    return [row for row in ledger_rows(root) if row.get("kind") == kind]


def test_a_startup_read_failure_binds_no_stand_in_and_every_send_rereads(data_root, monkeypatch):
    """Negative then positive through the scope ``handle_task`` actually binds: the startup
    read fails, so no group is bound (never ``unavailable:``/0). While the reader still
    fails, each send is refused typed and writes nothing; once it recovers, the SAME
    scope reads again and reserves in the original group P. On the old code the stand-in
    was cached and refused as money without any read."""
    from ouroboros import task_results
    from ouroboros.agent import OuroborosAgent
    from ouroboros.task_results import write_task_result

    write_task_result(data_root, "S", "running", billing_group=BINDING)
    real_read = task_results.load_task_result
    switch = {"denied": True}
    calls = []

    def reader(*args, **kwargs):
        calls.append(args[1])
        if switch["denied"]:
            raise PermissionError("synthetic unreadable billing authority")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", reader)
    monkeypatch.setattr("ouroboros.subagent_runtime.apply_task_start_settings_or_disclose",
                        lambda *_a, **_k: task_settings_snapshot({}, os.environ))
    monkeypatch.setattr("ouroboros.model_wait.task_model_wait_scope", lambda **_k: nullcontext())
    outcomes = []

    def scoped(_task):
        scope = ua.current_usage_scope()
        for denied in (True, False):
            switch["denied"] = denied
            try:
                reservation = ua.reserve_attempt(_request())
            except ua.BillingAuthorityUnavailable as exc:
                outcomes.append(("refused", exc.kind, len(_kinds(data_root, "attempt"))))
            else:
                ua.release_attempt(reservation, reason="test_no_send")
                outcomes.append(("reserved", reservation.scope.billing_group_id, len(calls) > 1))
        return scope

    from tests._agent_entry_helpers import task_entry_agent

    host = task_entry_agent(data_root, scoped)
    scope = OuroborosAgent.handle_task(host, {"id": "S", "type": "task"})
    assert scope.billing_group_id == "" and scope.billing_group_limit_usd is None  # nothing stands in
    assert outcomes == [("refused", "unreadable", 0), ("reserved", "P", True)]
    assert [row["billing_group_id"] for row in _kinds(data_root, "attempt")] == ["P"]


def test_corrupt_unreadable_and_refused_lineage_are_typed_while_real_exhaustion_stays_money(data_root):
    from ouroboros.task_results import write_task_result
    from ouroboros.usage_admission import task_billing_fields

    # A corrupt binding is an operational fault: unreadable, never a positive refusal.
    write_task_result(data_root, "C", "running", billing_group={"billing_group_id": "P"})
    with pytest.raises(ua.BillingAuthorityUnavailable) as corrupt:
        task_billing_fields({"id": "C"}, "C", None, data_root)
    assert (corrupt.value.kind, corrupt.value.cause) == ("unreadable", "malformed_binding")
    with pytest.raises(ua.BillingAuthorityUnavailable):
        _spend(data_root, _scope(data_root, "C", "C"), 0.01)
    # An older record's stand-in carried as a binding authorizes nothing.
    write_task_result(data_root, "L", "running", billing_group={"billing_group_id": "unavailable:L",
                                                                  "billing_group_limit_usd": 0.0})
    with pytest.raises(ua.BillingAuthorityUnavailable) as legacy:
        task_billing_fields({"id": "L"}, "L", None, data_root)
    assert (legacy.value.kind, legacy.value.cause) == ("refused", "stand_in_binding")
    # A descendant whose root has neither a row nor ledger history has no lineage at all.
    with pytest.raises(ua.BillingAuthorityUnavailable) as orphan:
        _spend(data_root, _scope(data_root, "child", "gone"), 0.01)
    assert (orphan.value.kind, orphan.value.cause) == ("refused", "root_lineage_missing")
    assert "(refused: root_lineage_missing)" in str(orphan.value)
    assert _kinds(data_root, "attempt") == []
    # Control: an actually reached group cap is still the monetary refusal.
    capped = dict(group="P", group_limit=1.0, root_limit=1.0)
    _spend(data_root, _scope(data_root, "P", "P", **capped), 1.0)
    with pytest.raises(ua.BudgetExceeded):
        _spend(data_root, _scope(data_root, "P", "P", **capped), 0.5)


def test_a_late_session_keeps_its_start_bucket_and_otherwise_stays_unrecorded_until_readable(data_root, monkeypatch):
    """Settlement never writes a stand-in, missing or own group. START custody carries
    the bucket the session was admitted under. With no readable bucket, the receipt
    is refused typed (its custody keeps it), then lands in P once the read recovers."""
    from ouroboros import delegate_custody as custody
    from ouroboros.task_results import write_task_result

    write_task_result(data_root, "S", "running", billing_group=BINDING)
    write_task_result(data_root, "R", "running", billing_group={**BINDING, "billing_group_id": "R"})
    assert custody.record_start_requested(data_root, task_id="S", root_task_id="S", invocation_id="s-start",
                                          request={"prompt": "work"}, idempotency_key="s-start")
    # An older start recorded the stand-in itself: it names no bucket.
    custody.emit(data_root, custody.START_REQUESTED, {
        "ts": "2026-10-14T00:00:00+00:00", "event_id": "legacy-standin", "run_id": "", "task_id": "R",
        "root_task_id": "R", "invocation_id": "r-start", "idempotency_key": "r-start",
        "billing_group": {"billing_group_id": "unavailable:R", "billing_group_limit_usd": 0.0}})
    unreadable_result(monkeypatch, "S")
    switch = unreadable_result(monkeypatch, "R")
    ua.record_subscription_session("late-s", drive_root=data_root, route="test", model="m",
                                   task_id="S", root_task_id="S", spend_usd=3.0)
    with pytest.raises(ua.BillingAuthorityUnavailable):
        ua.record_subscription_session("late-r", drive_root=data_root, route="test", model="m",
                                       task_id="R", root_task_id="R", spend_usd=2.0)
    sessions = _kinds(data_root, "subscription_session")
    assert [(row["task_id"], row["billing_group_id"]) for row in sessions] == [("S", "P")]
    switch["on"] = False  # the authority reads again: the same receipt now lands in its own bucket
    ua.record_subscription_session("late-r", drive_root=data_root, route="test", model="m",
                                   task_id="R", root_task_id="R", spend_usd=2.0)
    assert {row["billing_group_id"] for row in _kinds(data_root, "subscription_session")} == {"P", "R"}
    assert not any(str(row.get("billing_group_id") or "").startswith("unavailable:") for row in ledger_rows(data_root))


def test_no_delegated_session_starts_under_an_unreadable_bucket(data_root):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import task_snapshot, tree_billing_group
    from ouroboros.delegate_start_claims import claimed_start_request
    from ouroboros.task_store import patch_document
    from ouroboros.task_results import write_task_result

    write_task_result(data_root, "S", "running", billing_group={"billing_group_id": "P"})  # corrupt: unreadable
    with pytest.raises(ua.BillingAuthorityUnavailable):
        custody.record_start_requested(data_root, task_id="S", root_task_id="S", invocation_id="i-1",
                                       request={"prompt": "x"}, idempotency_key="i-1")
    requested, refusal = claimed_start_request(
        data_root, run_id="", task_id="S", root_task_id="S",
        invocation_id="i-2", request={"prompt": "x"}, idempotency_key="i-2")
    assert requested is False and refusal["reason"] == "billing_authority_unavailable"
    assert refusal["billing_authority"] == "unreadable" and "budget" not in refusal["detail"]
    assert task_snapshot(data_root, "S")["pending"] == []  # no START row, so no POST was possible
    patch_document(data_root, "S", lambda current: {**current, "billing_group": BINDING})
    assert custody.record_start_requested(data_root, task_id="S", root_task_id="S", invocation_id="i-3",
                                          request={"prompt": "x"}, idempotency_key="i-3")
    assert len(task_snapshot(data_root, "S")["pending"]) == 1
    assert tree_billing_group(data_root, "S")["billing_group_id"] == "P"  # the bucket its receipt will use


# ------------------------------------------------------------- the real loop and its pause

def _loop_driver(root, monkeypatch, task_id, *, exact_owner=True):
    from ouroboros import loop as loop_mod
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    ctx = registry._ctx
    ctx.task_id, ctx.budget_drive_root, ctx.task_attempt = task_id, str(root), 1
    ctx.task_started_at = time.time() - 60.0
    if exact_owner:
        ctx.owner_wait_callback = lambda *_a, **_k: "owner_input"
    sends = []

    def round_model(_call):  # the real admission refuses before any model bytes exist
        sends.append(task_id)
        return ua.reserve_attempt(_request())

    monkeypatch.setattr(loop_mod, "_call_round_model", round_model)
    return loop_mod, registry, sends


def _run(loop_mod, registry, root, task_id):
    return loop_mod.run_llm_loop(
        messages=[{"role": "user", "content": "keep this thought"}], tools=registry,
        llm=SimpleNamespace(default_model=lambda: "test-model"), drive_logs=root,
        emit_progress=lambda _text, *, incident=None: None, incoming_messages=queue_mod.Queue(),
        task_id=task_id, drive_root=root)


def _task_scope(root, task_id, root_task_id):
    """The scope startup binds when the authority could not be read: no group stands in."""
    return ua.UsageScope(drive_root=root, task_id=task_id, root_task_id=root_task_id, source="test")


def test_unreadable_authority_parks_the_same_thought_and_only_a_strict_reread_resumes_it(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_wait
    from ouroboros.gateway.state import _activity_pause_cause
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.task_store import patch_document
    from supervisor.events import _handle_budget_pause
    from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx, _supervisor_ctx

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    write_task_result(tmp_path, "tech", "running", billing_group={"billing_group_id": "tech"})  # unreadable cap
    observed = []
    real_observe = budget_pause.observe_task_runs

    def spy(root, task_id, **kwargs):
        observed.append(kwargs)
        return real_observe(root, task_id, **kwargs)

    monkeypatch.setattr(budget_pause, "observe_task_runs", spy)
    loop_mod, registry, sends = _loop_driver(tmp_path, monkeypatch, "tech")
    with ua.usage_scope(_task_scope(tmp_path, "tech", "tech")), pytest.raises(budget_pause.BudgetPauseRequested) as paused:
        _run(loop_mod, registry, tmp_path, "tech")
    pause = paused.value.pause
    assert sends == ["tech"] and _kinds(tmp_path, "attempt") == []
    assert (pause["reason"], pause["rail"], pause["scope"]) == ("authority", "billing_authority_unreadable", "task")
    assert pause["source_ref"] and "billing group authority unavailable" in pause["reason_text"]
    assert observed[-1]["stop_policy"] == "observe_only"  # sent runs finish in their own limits
    budget_pause.end_dispatch_fence("tech")

    # The supervisor parks the SAME id under the authority reason: no monetary root latch.
    task = {"id": "tech", "type": "task", "chat_id": 0, "root_task_id": "tech", "_attempt": 1}
    workers.RUNNING["tech"] = {"task": task, "worker_id": 0, "attempt": 1}
    workers.WORKERS[0] = SimpleNamespace(busy_task_id="tech")
    pushed = []
    _handle_budget_pause({**budget_pause.pause_event(task, pause), "worker_id": 0},
                         _supervisor_ctx(tmp_path, workers, queue, [], pushed))
    parked = workers.PENDING[0]
    assert parked["_budget_pause"]["reason"] == "authority" and "tech" not in queue.BUDGET_ROOT_FENCES
    row = load_task_result(tmp_path, "tech")
    assert row["reason_code"] == "authority_paused" and "billing authority unreadable" in row["result"]
    assert row["budget_pause"]["state"] == budget_pause.STATE_PAUSED
    assert [notice["reason"] for notice in row["pause_notices"].values()] == ["authority"]
    assert pushed[0]["pause_reason"] == "authority" and _activity_pause_cause(parked, {}) == "authority"

    # Resume rereads the authority strictly: still unreadable grants nothing, and is named so.
    refused = queue.resume_budget_paused_task("tech")
    assert (refused["error"], refused["billing_authority"]) == ("billing_authority_unavailable", "unreadable")
    assert "_budget_pause" in parked and "_budget_pause_resume" not in parked
    patch_document(tmp_path, "tech", lambda current: {**current, "billing_group": {**BINDING, "billing_group_id": "tech"}})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 0.0)  # real exhaustion still binds
    assert queue.resume_budget_paused_task("tech")["error"] == "budget_still_exhausted"
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    granted = queue.resume_budget_paused_task("tech")
    assert granted["ok"] is True and observed[-1]["request_stop"] is False  # Resume observes, never stops

    # The restored loop continues the same saved thought, told truthfully why it paused.
    ctx, _limit = _loop_ctx(tmp_path, "tech")
    ctx.budget_pause_resume = parked["_budget_pause_resume"]
    saved = budget_pause.load_budget_pause(ctx)
    assert saved["pause_id"] == pause["pause_id"] and saved["reason"] == "authority"
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    messages = []
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, messages, {}, {}, set(), budget_remaining_usd=5.0)
    first = messages[0]
    assert first["role"] == "user" and "keep this thought" in str(first["content"])  # the saved stack itself
    notice = messages[-1]["content"]
    assert "technical pause (billing authority unreadable)" in notice and "not_a_money_rail" in notice
    assert budget_pause.budget_pause_row(tmp_path, "tech")["state"] == budget_pause.STATE_RESUMED


def test_refused_lineage_ends_as_a_technical_interruption_without_pause_or_money_words(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.owner_continue import continuation_eligibility

    loop_mod, registry, sends = _loop_driver(tmp_path, monkeypatch, "orphan")
    with ua.usage_scope(_task_scope(tmp_path, "orphan", "gone-root")):
        text, usage, trace = _run(loop_mod, registry, tmp_path, "orphan")
    assert sends == ["orphan"] and (usage["execution_status"], usage["reason_code"]) == (
        "infra_failed", "billing_authority_refused")
    assert usage["billing_authority"]["kind"] == "refused" and trace["billing_authority"]["kind"] == "refused"
    assert "not a budget limit" in text and "neither waiting nor a budget change" in text
    assert "resource_limit" not in usage and budget_pause.budget_pause_row(tmp_path, "orphan") == {}
    assert continuation_eligibility({"status": "failed", "reason_code": "billing_authority_refused"}, "orphan") == {
        "eligible": False, "cause": "billing_authority_refused", "refusal": "billing_lineage_refused"}


def test_an_unreadable_authority_without_an_exact_owner_is_a_continuable_interruption(tmp_path, monkeypatch):
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "solo", "running", billing_group={"billing_group_id": "solo"})
    loop_mod, registry, _sends = _loop_driver(tmp_path, monkeypatch, "solo", exact_owner=False)
    with ua.usage_scope(_task_scope(tmp_path, "solo", "solo")):
        text, usage, _trace = _run(loop_mod, registry, tmp_path, "solo")
    assert usage["reason_code"] == "billing_authority_unreadable"
    assert usage["billing_authority"]["exact_pause_unavailable"] == "no_continuation_owner"
    assert "could not be read" in text and "not a budget limit" in text and "Increase" not in text
    assert continuation_eligibility({"status": "failed", "reason_code": "billing_authority_unreadable"}, "solo")["eligible"]


# ------------------------------------------------------------- advisory and presentation consumers

def test_review_wave_discloses_unknown_without_a_new_gate_and_names_the_authority_kind(data_root, monkeypatch):
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.review_helpers import review_wave_budget_gate

    write_task_result(data_root, "C", "running", billing_group={"billing_group_id": "P"})
    with ua.usage_scope(_scope(data_root, "C", "C")):
        wave = ua.review_wave_admission(data_root, root_task_id="C", task_id="C", models=["openai/gpt-5.2"],
                                        prompt_chars=4000, max_completion_tokens=1000)
        assert (wave["fits"], wave["reason"], wave["billing_authority"]) == (False, "billing_authority_unavailable", "unreadable")
        events = []
        monkeypatch.setattr("ouroboros.tools.review_helpers.emit_review_event", lambda _ctx, event: events.append(event))
        # No budget headline: each seat's reservation refuses typed and unsent instead.
        assert review_wave_budget_gate(SimpleNamespace(task_id="C"), surface="plan_review",
                                       models=["openai/gpt-5.2"], prompt_chars=4000) is None
        assert [event["type"] for event in events] == ["review_wave_billing_authority_unavailable"]
    monkeypatch.setattr(ua, "usage_projection", lambda *_a, **_k: (_ for _ in ()).throw(OSError("ledger unreadable")))
    with ua.usage_scope(_scope(data_root, "S", "S", **dict(group="S", group_limit=20.0, root_limit=20.0))):
        wave = ua.review_wave_admission(data_root, root_task_id="S", models=["openai/gpt-5.2"], prompt_chars=4000)
    assert wave["fits"] is None and wave["admission_unknown"] == "OSError"
    events.clear()
    with ua.usage_scope(_scope(data_root, "S", "S", group="S", group_limit=20.0, root_limit=20.0)):
        assert review_wave_budget_gate(SimpleNamespace(task_id="S"), surface="plan_review",
                                       models=["openai/gpt-5.2"], prompt_chars=4000) is None
    assert [event["type"] for event in events] == ["review_wave_admission_unavailable"]
    assert events[0]["error"] == "OSError"


def test_a_non_contention_launch_fault_surfaces_once_and_contention_still_waits(monkeypatch):
    from ouroboros import budget_pause, llm_attempt, loop_round_limits, owner_pause
    from ouroboros.model_wait import ModelWaitInterrupted

    monkeypatch.setattr(llm_attempt, "require_physical_dispatch_window", lambda: None)
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.0)
    probes = []

    @contextmanager
    def admission(_scope):
        reason = probes.pop(0)
        if reason:
            raise owner_pause.LaunchBusy(reason)
        yield

    monkeypatch.setattr(owner_pause, "launch_admission", admission)
    unsent = ModelWaitInterrupted("owner_launch_authority_unavailable", role="main")
    probes[:] = ["permission"]
    with pytest.raises(ModelWaitInterrupted) as raised:
        loop_round_limits._handle_model_wait_control(SimpleNamespace(), unsent)
    assert raised.value.control_reason == "owner_launch_authority_unavailable:permission"
    assert raised.value.previous_error is unsent and probes == []  # probed once, never looped
    probes[:] = ["contention", "contention", ""]
    assert loop_round_limits._handle_model_wait_control(
        SimpleNamespace(), ModelWaitInterrupted("owner_launch_authority_unavailable")) is None
    assert probes == []


def test_a_broken_store_ends_the_authority_rail_truthfully_instead_of_holding(tmp_path, monkeypatch):
    """No indefinite hold for a technical cause: when the pause record cannot be written the
    rail returns (no pause claimed, fence reopened) and the loop ends as an interruption."""
    from ouroboros import budget_pause
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "brk", "running", billing_group={"billing_group_id": "brk"})
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    monkeypatch.setattr(budget_pause, "set_budget_pause",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("store refuses writes")))
    loop_mod, registry, _sends = _loop_driver(tmp_path, monkeypatch, "brk")
    with ua.usage_scope(_task_scope(tmp_path, "brk", "brk")):
        text, usage, _trace = _run(loop_mod, registry, tmp_path, "brk")
    assert usage["reason_code"] == "billing_authority_unreadable"
    assert usage["billing_authority"]["exact_pause_unavailable"] == budget_pause.HOLD_PAUSE_RECORD_UNWRITABLE
    assert "budget_pause_hold" not in usage and not budget_pause.dispatch_fenced("brk")
    assert getattr(registry._ctx, "_budget_pausing", False) is False and "not a budget limit" in text


def test_resume_of_an_authority_pause_waits_for_its_runs_without_suggesting_cancellation(tmp_path, monkeypatch):
    from ouroboros import artifacts
    from supervisor.budget_resume import _resume_custody_refusal

    monkeypatch.setattr(artifacts, "read_actor_source_bytes", lambda *_a, **_k: b"{}")
    external = {"custody_read": "ok", "runs": [{"run_id": "r1", "state": "running", "stop_outcome": ""}]}
    for reason, action in (("authority", "wait_for_delegated_runs_to_settle"),
                           ("budget", "wait_for_delegated_runs_to_settle_or_cancel_them")):
        refusal = _resume_custody_refusal(tmp_path, "m2", {"reason": reason, "source_ref": {"path": "x"},
                                                           "pause_id": "p", "state": "paused"}, external)
        assert (refusal["error"], refusal["action"]) == ("external_runs_unsettled", action)


# ------------------------------------------------------------- retained runs: the existing body custody

def _authority_park(tmp_path, monkeypatch, *, critic=False):
    """A real authority park over one live ordinary run, accepted by the body before release."""
    from ouroboros import budget_pause as bp, delegate_custody as custody, sleep_custody as sc
    from supervisor.events_budget import install_exact_budget_pause
    from tests._budget_pause_exact_helpers import _supervisor_ctx
    from tests.test_cold_external_custody import _setup

    env = _setup(tmp_path, monkeypatch)
    if critic:  # a started panel's critic keeps its own controller and is not transferred
        assert custody.emit(tmp_path, custody.STARTED, {"run_id": "critic", "invocation_id": "invoke-critic",
            "task_id": "sleeper", "root_task_id": "sleeper", "route": "route", "source": "review_substrate",
            "access": "readonly"})
    accepted_before_release = []

    def host_tick(_ctx):
        sc.reconcile(env.q, runtime=env.runtime)
        if bp.budget_pause_row(tmp_path, "sleeper").get("external_custody", {}).get("state") == "accepted":
            accepted_before_release.append(True)
        return ""

    monkeypatch.setattr(bp, "_hold_control_reason", host_tick)
    with pytest.raises(bp.BudgetPauseRequested) as raised:
        bp.request_pause(env.limit, rail=bp.RAIL_BILLING_AUTHORITY, scope="task",
                         reason_text="billing authority unreadable", root_task_id="sleeper")
    row = raised.value.pause
    assert accepted_before_release and sc.accepted_receipt("sleeper", row)
    install_exact_budget_pause(_supervisor_ctx(tmp_path, env.workers, env.q, [], []), "sleeper",
                               bp.exact_pause_marker(row)["checkpoint"], evt=bp.pause_event(env.task, row))
    bp.end_dispatch_fence("sleeper")
    return env, bp.budget_pause_row(tmp_path, "sleeper")


def test_an_authority_park_hands_its_exact_runs_to_the_body_and_the_same_author_returns_over_them(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, delegate_custody_current as current, owner_wait
    from supervisor.sleep_wake import wake_ready_sleepers

    env, row = _authority_park(tmp_path, monkeypatch, critic=True)
    assert (row["reason"], row["rail"], row["state"]) == ("authority", "billing_authority_unreadable", "paused")
    receipt = row["external_custody"]
    assert receipt["runs"] == [["run-one", "invoke-one"]]  # the critic stayed with its panel
    assert receipt["binding"]["reason"] == "authority" and "sleep_id" not in receipt["binding"]
    assert env.runtime.calls and all(call[0] == [["run-one", "invoke-one"]] for call in env.runtime.calls)
    assert "sleeper" not in env.q.BUDGET_ROOT_FENCES  # member-only: no root money latch
    assert wake_ready_sleepers(env.q) == []  # nothing wakes an authority park by itself
    # The same author returns over its still-live run once the authority reads again.
    granted = env.q.resume_budget_paused_task("sleeper")
    assert granted["ok"] is True, granted
    assert not current.run(tmp_path, "run-one").settled, "Resume neither stops nor collects the run"
    ctx, _limit = _loop_ctx_for(tmp_path)
    ctx.budget_pause_resume = env.workers.PENDING[0]["_budget_pause_resume"]
    saved = bp.load_budget_pause(ctx)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    messages = []
    bp.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, messages, {}, {}, set(), budget_remaining_usd=5.0)
    assert messages[0]["content"] == "original intent"
    assert "technical pause (billing authority unreadable)" in messages[-1]["content"]


def _loop_ctx_for(tmp_path):
    from tests._budget_pause_exact_helpers import _loop_ctx
    return _loop_ctx(tmp_path, "sleeper")


def test_return_needs_the_body_to_hold_the_exact_runs_and_an_owner_pause_still_stops_them(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp
    from ouroboros.owner_pause import install_fence
    from supervisor.owner_pause_control import _record_settled_member

    env, _row = _authority_park(tmp_path, monkeypatch)
    env.runtime.active = False  # the body no longer answers for the run: no return, no stop asked
    refused = env.q.resume_budget_paused_task("sleeper")
    assert (refused["error"], refused["action"]) == ("external_runs_unsettled", "wait_for_delegated_runs_to_settle")
    # A real later owner Pause reaches the retained run through its accepted receipt.
    install_fence(tmp_path, "sleeper", request_id="owner-pause")
    calls = []
    monkeypatch.setattr(bp, "observe_task_runs", lambda _root, _task, **kw: calls.append(kw) or {
        "runs": [], "custody_read": "ok", "observed_at": time.time()})
    _record_settled_member(env.q, {"id": "sleeper", "root_task_id": "sleeper", "budget_drive_root": str(tmp_path)})
    assert [call["stop_policy"] for call in calls] == [bp.STOP_POLICY_TASK_OWNED]


def test_an_unbound_pending_start_keeps_the_authority_park_held_never_released(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, delegate_custody as custody
    from ouroboros.model_wait import ModelWaitInterrupted
    from tests._budget_pause_exact_helpers import _controls
    from tests.test_cold_external_custody import _setup

    env = _setup(tmp_path, monkeypatch)
    assert custody.record_start_requested(tmp_path, task_id="sleeper", root_task_id="sleeper",
                                          invocation_id="pending-1", request={"prompt": "x"},
                                          idempotency_key="pending-1")
    monkeypatch.setattr(bp, "_hold_control_reason", _controls("", "", "cancelled"))
    with pytest.raises(ModelWaitInterrupted) as ended:
        bp.request_pause(env.limit, rail=bp.RAIL_BILLING_AUTHORITY, scope="task", reason_text="unreadable",
                         root_task_id="sleeper")
    assert ended.value.control_reason == "cancelled"
    hold = env.limit.accumulated_usage["budget_pause_hold"]
    assert hold["hold_reason"] == bp.HOLD_COLD_WRITERS_UNSETTLED and "unbound external invocation" in hold["error"]
    assert bp.budget_pause_row(tmp_path, "sleeper")["state"] == bp.STATE_ABANDONED  # no pause was claimed


# ------------------------------------------------------------- control authority (claim 5)

def test_an_unreadable_dispatch_fence_parks_on_the_control_rail_not_as_money(tmp_path, monkeypatch):
    from ouroboros import budget_pause, queue_observation
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "fence", "running", billing_group={**BINDING, "billing_group_id": "fence"})
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "queue_dispatch.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(queue_observation, "dispatch_probe", lambda *_a, **_k: None)
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    loop_mod, registry, _sends = _loop_driver(tmp_path, monkeypatch, "fence")
    with ua.usage_scope(_task_scope(tmp_path, "fence", "fence")), pytest.raises(budget_pause.BudgetPauseRequested) as paused:
        _run(loop_mod, registry, tmp_path, "fence")
    pause = paused.value.pause
    assert (pause["reason"], pause["rail"], pause["scope"]) == ("authority", "control_authority_unreadable", "task")
    assert "root budget fence authority unavailable" in pause["reason_text"] and _kinds(tmp_path, "attempt") == []
    budget_pause.end_dispatch_fence("fence")


def test_an_unreadable_owner_pause_fence_is_saved_when_capture_works_and_ends_typed_when_it_cannot(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_pause
    from ouroboros.loop_round_limits import _handle_model_wait_control
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.task_results import write_task_result
    from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx

    write_task_result(tmp_path, "ctl", "running")
    ctx, limit_ctx = _loop_ctx(tmp_path, "ctl")
    _fast_hold(monkeypatch, budget_pause)
    monkeypatch.setattr(owner_pause, "read_fence", lambda *_a, **_k: (_ for _ in ()).throw(OSError("fence unreadable")))
    ctx.task_lifecycle_bound = True
    with pytest.raises(ModelWaitInterrupted) as boundary:
        budget_pause.enter_owner_pause(limit_ctx)
    assert boundary.value.control_reason == "owner_pause_authority_unreadable"
    with pytest.raises(budget_pause.BudgetPauseRequested) as saved:
        _handle_model_wait_control(limit_ctx, boundary.value)
    assert (saved.value.pause["reason"], saved.value.pause["rail"]) == ("authority", "control_authority_unreadable")
    budget_pause.end_dispatch_fence("ctl")
    # When the capture itself cannot be written, the typed cause ends the work; nothing holds or claims a pause.
    write_task_result(tmp_path, "ctl2", "running")
    ctx2, limit2 = _loop_ctx(tmp_path, "ctl2")
    monkeypatch.setattr(budget_pause, "set_budget_pause", lambda *_a, **_k: (_ for _ in ()).throw(OSError("store")))
    unread = ModelWaitInterrupted("owner_pause_authority_unreadable")
    with pytest.raises(ModelWaitInterrupted) as ended:
        _handle_model_wait_control(limit2, unread)
    assert ended.value is unread and limit2.accumulated_usage["exact_pause_unavailable"] == \
        budget_pause.HOLD_PAUSE_RECORD_UNWRITABLE
    assert not budget_pause.dispatch_fenced("ctl2") and "budget_pause_hold" not in limit2.accumulated_usage


def test_an_ambiguous_transfer_after_publication_holds_and_keeps_its_published_source(tmp_path, monkeypatch):
    """The prepared row may already be published when the canonical read fails: that is no
    definite failed save. The park holds (never the bare-seed infrastructure fallback) until
    the task's own control ends it, and the published source and receipt request survive."""
    from ouroboros import budget_pause as bp, sleep_custody as sc
    from ouroboros.model_wait import ModelWaitInterrupted
    from tests._budget_pause_exact_helpers import _controls
    from tests.test_cold_external_custody import _setup

    env = _setup(tmp_path, monkeypatch)

    def confirm(ctx, prepared, seed, *, control):
        sc._replace(tmp_path, "sleeper", seed, prepared)  # published, then the canonical read fails
        raise RuntimeError("canonical pause row unreadable after publication")

    monkeypatch.setattr(sc, "confirm_release", confirm)
    monkeypatch.setattr(bp, "_hold_control_reason", _controls("", "", "cancelled"))
    with pytest.raises(ModelWaitInterrupted) as ended:
        bp.request_pause(env.limit, rail=bp.RAIL_BILLING_AUTHORITY, scope="task", reason_text="unreadable",
                         root_task_id="sleeper")
    usage = env.limit.accumulated_usage
    assert ended.value.control_reason == "cancelled" and usage["exact_pause_unavailable"] == "hold_ended_by_control"
    assert usage["budget_pause_hold"]["hold_reason"] == bp.HOLD_COLD_WRITERS_UNSETTLED
    row = bp.budget_pause_row(tmp_path, "sleeper")
    assert row["state"] == bp.STATE_ABANDONED and row["abandon_reason"] == "hold_ended_by_control:cancelled"
    assert row["source_ref"] and row["external_custody"]["state"] == "requested"


def test_a_one_shot_control_during_the_transfer_propagates_as_itself_at_once(tmp_path, monkeypatch):
    """A control raised once inside the confirmation is the task's own Stop/deadline: the exact
    exception leaves at once, the published capture is closed as ended by that control, and
    nothing waits for the control to be reported a second time."""
    from ouroboros import budget_pause as bp, sleep_custody as sc
    from ouroboros.model_wait import ModelWaitInterrupted
    from tests.test_cold_external_custody import _setup

    env = _setup(tmp_path, monkeypatch)
    original = ModelWaitInterrupted("deadline")

    def confirm(ctx, prepared, seed, *, control):
        sc._replace(tmp_path, "sleeper", seed, prepared)
        raise original

    polls = []

    def quiet_controls(_ctx):  # the control is never reported again; a second loop would be a hang
        polls.append(1)
        assert len(polls) < 5, "the hold kept polling instead of ending on the raised control"
        return ""

    monkeypatch.setattr(sc, "confirm_release", confirm)
    monkeypatch.setattr(bp, "_hold_control_reason", quiet_controls)
    with pytest.raises(ModelWaitInterrupted) as ended:
        bp.request_pause(env.limit, rail=bp.RAIL_BILLING_AUTHORITY, scope="task", reason_text="unreadable",
                         root_task_id="sleeper")
    assert ended.value is original and len(polls) == 1
    row = bp.budget_pause_row(tmp_path, "sleeper")
    assert row["state"] == bp.STATE_ABANDONED and row["abandon_reason"] == "hold_ended_by_control:deadline"
    assert row["source_ref"] and row["external_custody"]["state"] == "requested"
    assert env.limit.accumulated_usage["exact_pause_unavailable"] == "hold_ended_by_control"
