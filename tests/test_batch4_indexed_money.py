"""Batch4 group consumers composed with the published strict writer and waits."""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import threading
import time

import pytest

from ouroboros import usage_accounting as ua
from ouroboros import usage_ledger as ledger
from ouroboros import usage_store
from tests.test_billing_group import _scope, _spend
from tests.test_physical_candidate_capture import data_root as data_root
from tests.test_usage_lock_continuity import held_lock, owner
from tests.test_usage_lock_continuity import short_acquisitions as short_acquisitions
from tests._usage_store_testing import request, write_journal
from tests._usage_store_testing import root as root

pytestmark = pytest.mark.serial


def group_scope(root, task="child", root_id="successor", *, cap=10, root_cap=None):
    return _scope(root, task, root_id, group="original", group_limit=cap, root_limit=root_cap,
                  parent_task_id=root_id if task != root_id else "",
                  billing_group_limit_source="initial_task_admission", billing_group_limit_revision="original-rev")


@pytest.mark.parametrize("prefix", [1, 2500])
def test_group_hot_writers_and_off_context_binding_never_scan_history(root, monkeypatch, prefix):
    from ouroboros import _usage_rows
    from ouroboros import usage_admission as admission

    # Pre-group original rows (journaled, imported once) join by root identity,
    # while each successor keeps its own identity.
    records = []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for index in range(prefix):
        row = dict(attempt_id=f"original-{index}", kind="attempt", task_id="original", root_task_id="original",
                   provider="local", model="stub", root_limit_usd=10, reservation_upper_bound_usd=.001, ts=stamp)
        for state in ("reserved", "dispatched", "settled"):
            records.append({**row, "state": state,
                            **({"cost_usd": .001, "cost_final": True} if state == "settled" else {})})
    write_journal(root, records)
    scope = group_scope(root)
    with ua.usage_scope(scope):
        late = ua.reserve_attempt(request(root, root_task_id="successor"))
        ua.mark_dispatched(late)
        ua.mark_unresolved(late, "receipt delayed")

    def forbidden(*a, **kw):
        pytest.fail("group monetary call walked historical rows")

    # The hot paths read summary rows and the addressed attempt only.
    monkeypatch.setattr(usage_store, "read_usage_records", forbidden)
    monkeypatch.setattr(usage_store, "migrate_from_journal", forbidden)
    for name in ("_summary", "_breakdown_bucket"):
        monkeypatch.setattr(_usage_rows, name, forbidden)
        monkeypatch.setattr(ua, name, forbidden, raising=False)
    for index in range(8):
        with ua.usage_scope(group_scope(root, task=f"sibling-{index}", root_id=f"sibling-{index}")):
            assert ua.execute_physical_attempt(request(root, task_id=f"sibling-{index}", root_task_id=f"sibling-{index}",
                                                      reservation_usd=.1), lambda: {"usage": {"cost": .1}})
    # No execution context or canonical result: the first ledger binding remains indexed authority.
    ua.record_subscription_session("late-session", drive_root=root, task_id="child", root_task_id="successor",
                                   route="subscription", spend_usd=.2)
    ua.settle_attempt(late, cost_usd=.3, cost_final=True)
    assert admission.ledger_billing_binding(root, "successor")["billing_group_limit_revision"] == "original-rev"
    assert admission.original_group_limit(root, "original") == {"limit_usd": 10.0, "source": "ledger_first_row"}
    with usage_store.read(root) as txn:
        assert txn.summary(billing_group_id="original")["accounted_usd"] == pytest.approx(prefix*.001+1.3)
        assert txn.summary("successor")["accounted_usd"] == .5
        assert txn.summary()["accounted_usd"] == txn.summary(billing_group_id="original")["accounted_usd"]


@pytest.mark.parametrize("axis", ["root", "group", "global"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_pre_send_wait_revalidates_late_charge_on_each_axis(root, monkeypatch, axis, asynchronous):
    from ouroboros import _usage_wait
    from ouroboros.task_results import write_task_result
    write_task_result(root, 'successor', 'running', root_task_id='successor')
    write_task_result(root, 'child', 'running', root_task_id='successor', parent_task_id='successor')
    scope = group_scope(root, cap=2 if axis == "group" else 20, root_cap=2 if axis == "root" else 20)
    other_root = "successor" if axis == "root" else "original" if axis == "group" else "foreign"
    other_scope = group_scope(root, task=other_root, root_id=other_root, cap=scope.billing_group_limit_usd)
    if axis == "global":
        other_scope = dataclasses.replace(other_scope, billing_group_id="foreign")
    with ua.usage_scope(other_scope):
        late = ua.reserve_attempt(request(root, task_id=other_root, root_task_id=other_root, reservation_usd=0))
        ua.mark_dispatched(late)
        ua.mark_unresolved(late, "late charge")
    sent, prepared, waits = [], [], []
    lock_owner = contextlib.ExitStack()
    original_hold = _usage_wait._hold
    def observed_wait(wait_owner, phase, *args):
        original_hold(wait_owner, phase, *args)
        waits.append(phase)
        if phase == "entered":
            # Release only after actual contention is observed. The real lock
            # keeps its production acquisition slice; setup/cleanup are not timed faults.
            lock_owner.close()
    monkeypatch.setattr(_usage_wait, "_hold", observed_wait)
    with contextlib.ExitStack() as stack:
        stack.callback(lock_owner.close)
        stack.enter_context(owner(root, task={"id": "child", "root_task_id": "successor"}))
        stack.enter_context(ua.usage_scope(scope))
        stack.enter_context(ua.physical_attempt_limit(1))
        def before(reservation):
            prepared.append(reservation)
            # The same writer receives the original/sibling's delayed receipt after reservation.
            with ua.usage_scope(None):
                # Its known price reaches the $2 cap exactly (#1487: equality refuses).
                ua.settle_attempt(late, cost_usd=2.0, cost_final=True)
            lock_owner.enter_context(held_lock(root))
        req = request(root, root_task_id="successor", global_limit_usd=2 if axis == "global" else 100)
        async def send():
            sent.append(1)
            return {"usage": {}}
        with pytest.raises(ua.BudgetExceeded) as error:
            if asynchronous:
                asyncio.run(ua.execute_physical_attempt_async(req, send, before_dispatch=before))
            else:
                ua.execute_physical_attempt(req, lambda: sent.append(1), before_dispatch=before)
        assert error.value.limit_scope == ("global" if axis == "global" else "root")
        assert error.value.root_task_id == "successor"
        assert error.value.physical_attempt_capture.state == "released"
        assert ua._PHYSICAL_LIMIT.get().used == 0
    assert not sent and len(prepared) == 1
    assert waits == ["entered", "ended"]
    assert ua.read_usage_records(root, final_only=True)[-1]["state"] == "released"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_pause_installs_during_money_wait_after_capture_without_blocking(root, short_acquisitions, monkeypatch,
                                                                              asynchronous):
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_batch4_repair_compositions import _running

    _, _, workers = _install_queue(root, monkeypatch)
    _running(root, workers, "successor")
    scope = group_scope(root, task="successor")
    pause_ack, errors, sent, prepared = [], [], [], []
    pause_threads = []
    # The 25 ms acquisition fixture exercises the PRE-SEND wait. Explicitly
    # finish its lock holder before cleanup; scheduler timing must not decide
    # whether this test instead exercises the bounded failed-release case.
    lock_owner = contextlib.ExitStack()
    release_attempt = ua.release_attempt
    def release_after_pause(*args, **kwargs):
        for thread in pause_threads:
            thread.join(3)
            assert not thread.is_alive()
        lock_owner.close()
        return release_attempt(*args, **kwargs)
    monkeypatch.setattr(ua, 'release_attempt', release_after_pause)
    captured = threading.Event()
    original_capture = ua._record_attempt_capture
    def capture(*a, **kw):
        result = original_capture(*a, **kw)
        if a[2] == "reserved":
            captured.set()
        return result
    monkeypatch.setattr(ua, "_record_attempt_capture", capture)
    with contextlib.ExitStack() as stack:
        stack.callback(lock_owner.close)
        stack.enter_context(ua.usage_scope(scope))
        stack.enter_context(owner(root, task={"id": "successor", "root_task_id": "successor"}))
        stack.enter_context(ua.physical_attempt_limit(1))
        def before(held):
            prepared.append(held)
            release = lock_owner.enter_context(held_lock(root))
            def pause():
                try:
                    assert captured.wait(2)
                    pause_ack.append(request_owner_pause("successor", request_id="during-accounting"))
                    assert pause_ack[0]["ok"]
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    release.set()
            thread = threading.Thread(target=pause)
            pause_threads.append(thread)
            thread.start()
            stack.callback(thread.join, 3)
        req = request(root, task_id="successor", root_task_id="successor")
        async def send():
            sent.append(1)
        with pytest.raises(PhysicalDispatchInterrupted) as failure:
            if asynchronous:
                asyncio.run(ua.execute_physical_attempt_async(req, send, before_dispatch=before))
            else:
                ua.execute_physical_attempt(req, lambda: sent.append(1), before_dispatch=before)
        assert failure.value.control_reason == "owner_pause"
        assert ua._PHYSICAL_LIMIT.get().used == 0
    assert not errors and not sent and pause_ack[0]["ok"] and len(prepared) == 1
    assert ua.read_usage_records(root, final_only=True)[-1]["state"] == "released"


@pytest.mark.parametrize("terminal", ["settled", "unresolved", "dispatched"])
@pytest.mark.parametrize("deadline", [False, True])
def test_group_real_async_response_cancellation_preserves_binding_and_capture(data_root, monkeypatch, terminal, deadline):
    from tests import test_usage_response_custody as incoming

    monkeypatch.setattr(incoming, "_scope", lambda root, task: group_scope(root, task=task))
    incoming.test_real_api_consumer_cancel_retains_response_capture_and_claim(data_root, monkeypatch, terminal, deadline)
    finals = ua.read_usage_records(data_root, final_only=True)
    assert len(finals) == 1
    row = finals[0]
    assert (row["task_id"], row["root_task_id"], row["parent_task_id"], row["billing_group_id"]) == (
        "task-async", "successor", "successor", "original")
    assert row["billing_group_limit_usd"] == 10
    assert row["billing_group_limit_source"] == "initial_task_admission"
    assert row["billing_group_limit_revision"] == "original-rev"
    assert row["local_answer_owner_pid"] == os.getpid()
    assert ua.usage_projection(data_root, billing_group_id="original")["accounted_usd"] == (0 if terminal == "settled" else .01)


@pytest.mark.parametrize("unlimited", [False, True])
def test_imported_aggregates_keep_group_cash_provenance_and_unpriced_liability(root, unlimited):
    from ouroboros.usage_admission import ledger_billing_binding, original_group_limit
    from tests.fixtures_usage_store import foldable_attempt_ids, fold_into_archive

    cap = None if unlimited else 10
    for task, rid in (("original", "original"), ("successor", "successor"), ("helper", "successor")):
        for _ in range(3):
            _spend(root, group_scope(root, task, rid, cap=cap, root_cap=20), .1234565, bound=.2)
    with ua.usage_scope(group_scope(root, cap=cap, root_cap=20)):
        unknown = ua.reserve_attempt(request(root, root_task_id="successor", reservation_usd=None,
                                             force_unknown_reservation=True, provider="unpriced"))
        ua.mark_dispatched(unknown)
        ua.mark_unresolved(unknown, "unknown provider price")
    before = ua.usage_projection(root, billing_group_id="original")
    assert before.get("limit_usd") == cap and before["unknown_unmetered"] == 1
    assert before["cost_final"] is False
    # The same money as an install upgraded from a compacted journal carries it.
    fold_into_archive(root, foldable_attempt_ids(root))
    assert ua.usage_projection(root, billing_group_id="original") == before
    usage_store.forget(root)  # a fresh process answers the same
    assert ua.usage_projection(root, billing_group_id="original") == before
    binding = ledger_billing_binding(root, "successor")
    assert binding["billing_group_limit_usd"] is cap or binding["billing_group_limit_usd"] == cap
    assert binding["billing_group_limit_revision"] == "original-rev"
    assert original_group_limit(root, "original")["limit_usd"] == cap
    ua.settle_attempt(unknown, cost_usd=2, cost_final=True)
    group = ua.refresh_root_accounting(root, "group:original", strict=True)
    real_root = ua.refresh_root_accounting(root, "successor", strict=True)
    assert group["accounted_usd"] == 3.111108 and group["root_limit_usd"] == cap
    assert real_root["accounted_usd"] == 2.740739 and real_root["root_limit_usd"] == 20
    assert ua.usage_projection(root)["accounted_usd"] == group["accounted_usd"]


def test_group_six_place_policy_and_nonfinite_cap_remain_honest(root):
    with usage_store.hold(root) as txn:
        txn.write({"kind": "external_unmetered", "attempt_id": "exact", "state": "settled",
                   "root_task_id": "original", "cost_usd": "2.4675885", "cost_final": True})
    with ua.usage_scope(group_scope(root, cap=2.467588002)):
        held = ua.reserve_attempt(request(root, root_task_id="successor", reservation_usd=0))
        ua.mark_dispatched(held)
        ua.release_attempt(held, "before_dispatch_failed:test", proven_unsent=True)
    with ua.usage_scope(group_scope(root, cap=2.467588001)), pytest.raises(ua.BudgetExceeded):
        ua.reserve_attempt(request(root, root_task_id="successor", reservation_usd=0))
    with pytest.raises(ledger.UsageNonFiniteMoney), usage_store.hold(root) as txn:
        txn.write({"kind": "external_unmetered", "attempt_id": "invalid",
                   "state": "settled", "cost_usd": None, "billing_group_limit_usd": "NaN"})
    with usage_store.read(root) as txn:
        assert txn.attempt("invalid") is None


@pytest.mark.parametrize("source", ["owner", "panic"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_cancel_intent_after_claim_still_stops_request_bytes(root, monkeypatch, source, asynchronous):
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.llm import LLMClient
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.model_wait import task_model_wait_scope
    from ouroboros.task_results import write_task_result
    from tests.test_physical_candidate_capture import _target

    write_task_result(root, "successor", "running", root_task_id="successor")
    original_mark, claims, sent = ua.mark_dispatched, [], []
    def mark(*a, **kw):
        result = original_mark(*a, **kw)
        claims.append(a[0].attempt_id)
        request_cancel(root, "successor", source=source, requested_by="owner", reason=source,
                       requested_stop_policy="immediate")
        return result
    monkeypatch.setattr(ua, "mark_dispatched", mark)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: .01)
    client = LLMClient(api_key="unused")
    payload = {"model": "gpt-5.2", "messages": [{"role": "user", "content": "once"}]}
    async def send(**kw):
        sent.append(kw)
    with ua.usage_scope(group_scope(root, task="successor")), ua.physical_attempt_limit(1), task_model_wait_scope(
            task={"id": "successor", "root_task_id": "successor"}, drive_root=root, event_queue=None, worker_slot_held=False):
        with pytest.raises(PhysicalDispatchInterrupted) as error:
            if asynchronous:
                asyncio.run(client._create_chat_completion_with_retries_async(send, payload, _target()))
            else:
                client._create_chat_completion_with_retries(lambda **kw: sent.append(kw), payload, _target())
        assert error.value.control_reason == "cancelled"
        assert error.value.physical_attempt_capture.state == "released"
        assert ua._PHYSICAL_LIMIT.get().used == 0
    assert not sent and len(claims) == 1
    row = ua.read_usage_records(root, final_only=True)[-1]
    assert row["state"] == "released" and row["billing_group_id"] == "original"
