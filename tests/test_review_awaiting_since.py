"""``since HH:MM`` is shown only where the host itself recorded the send moment.

Two-sided, per row. A reviewer row the host is still waiting on carries the wall
clock of the physical operation THIS process started — the released-early
``pending_dispatch`` row and the expired-window ``in_flight`` row alike. Every
other row carries none: a free replay of an already settled answer, a rejoin of
an operation an EARLIER process paid for, an answered row, and a stored row whose
value is not a timestamp. The projection never invents, infers or back-fills the
moment, and a row recorded before the field existed still loads.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.review_projection import _review_actor_projection
from ouroboros.review_records import ReviewActorRecord


def _instant(value: str) -> _dt.datetime:
    """Parse exactly as the projection does; a lie here must raise, not pass."""
    return _dt.datetime.fromisoformat(value)


def _error_actor(slot, error, operation_id="", operation_state="settled"):
    """The substrate's error-actor contract, as ``review_substrate`` supplies it."""
    return ReviewActorRecord(
        slot_id=slot.slot_id, model=slot.model, status="error", error=error,
        operation_id=operation_id, operation_state=operation_state,
        late_result_pending=operation_state in {"in_flight", "custody_lost"},
    )


def _run(tmp_path, request, slot, run_slot, ctx=None):
    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.usage_accounting import UsageScope

    return run_custodied_review_slots(
        request=request, slots=[slot],
        usage_ctx=ctx if ctx is not None else SimpleNamespace(_review_paid_stamp=lambda: None),
        task_id=request.task_id, usage_meta={},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id=request.task_id),
        run_slot=run_slot, error_actor=_error_actor,
    )


def _blocking_worker(release, entered, workers):
    def run_slot(slot, _operation_id, _retry_state, _deadline, _checkpoint):
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(10), "the test never released the review worker"
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="ok", raw_text="[]",
        )

    return run_slot


def test_a_released_slot_of_a_new_operation_records_when_the_host_sent_it(tmp_path):
    """The owner's question — since when? — is answered by the host's own clock."""
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    release, entered, workers = threading.Event(), threading.Event(), []
    request = ReviewRequest(
        surface="task_acceptance", goal="review", task_id="wait-clock-new",
        retry_key="acceptance:wait-clock-new", drain_deadline=time.monotonic(),
    )
    slot = ReviewSlot(slot_id="slot-1", model="test/model", timeout_sec=30)
    before = _dt.datetime.now(tz=_dt.timezone.utc)
    try:
        [actor] = _run(tmp_path, request, slot, _blocking_worker(release, entered, workers))
        after = _dt.datetime.now(tz=_dt.timezone.utc)
        assert actor.operation_state == "pending_dispatch"
        assert actor.late_result_pending is True
        assert before <= _instant(actor.awaiting_since) <= after
        projection = _review_actor_projection(dataclasses.asdict(actor), "task_acceptance")
        assert projection["awaiting_since"] == actor.awaiting_since
    finally:
        release.set()
        assert entered.wait(10), "the review worker never started"
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive(), "the review worker did not settle"


def test_an_expired_window_row_carries_the_same_recorded_moment(tmp_path):
    """The unresolved twin of the awaited row: same operation, same recorded send."""
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    release, entered, workers = threading.Event(), threading.Event(), []
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="wait-clock-expired",
        retry_key="commit_review:wait-clock-expired",
    )
    slot = ReviewSlot(slot_id="slot-1", model="test/model", timeout_sec=0.05)
    before = _dt.datetime.now(tz=_dt.timezone.utc)
    try:
        [actor] = _run(tmp_path, request, slot, _blocking_worker(release, entered, workers))
        after = _dt.datetime.now(tz=_dt.timezone.utc)
        assert actor.operation_state == "in_flight"
        assert before <= _instant(actor.awaiting_since) <= after
        projection = _review_actor_projection(dataclasses.asdict(actor), "multi_model_review")
        assert projection["awaiting_since"] == actor.awaiting_since
    finally:
        release.set()
        assert entered.wait(10), "the review worker never started"
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive(), "the review worker did not settle"


def test_an_answered_row_and_its_free_replay_carry_no_moment(tmp_path):
    """Nothing is awaited once the answer is here, and a $0 replay never waited."""
    from ouroboros.review_custody import prepare_frozen_review_reconciliation
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="wait-clock-answered",
        retry_key="commit_review:wait-clock-answered",
    )
    slot = ReviewSlot(slot_id="slot-1", model="cursor/test", timeout_sec=30)

    def answer(_slot, _operation_id, _retry_state, _deadline, _checkpoint):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="ok", raw_text="[]",
        )

    [answered] = _run(tmp_path, request, slot, answer)
    assert answered.status == "ok"
    assert answered.awaiting_since == ""
    assert "awaiting_since" not in _review_actor_projection(
        dataclasses.asdict(answered), "multi_model_review")

    ctx = SimpleNamespace(_review_paid_stamp=lambda: pytest.fail("a replay paid"))
    prepare_frozen_review_reconciliation(ctx, SimpleNamespace(triad_raw_results=[{
        "slot_id": "slot-1", "model_id": "cursor/test", "status": "ok",
        "raw_text": "[]", "operation_id": "op-settled", "operation_state": "settled",
        "late_result_pending": False,
    }], scope_raw_result={}))
    replay_request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="wait-clock-replay",
        retry_key="commit_review:wait-clock-replay", reconcile_only=True,
    )
    [replayed] = _run(
        tmp_path, replay_request, slot,
        lambda *_args: pytest.fail("a free replay dispatched a reviewer"), ctx=ctx,
    )
    assert replayed.operation_id == "op-settled"
    assert replayed.awaiting_since == ""
    assert "awaiting_since" not in _review_actor_projection(
        dataclasses.asdict(replayed), "multi_model_review")


def test_a_rejoined_operation_an_earlier_process_paid_for_states_no_moment(tmp_path):
    """This process did not send that request, so it does not know when it went."""
    import ouroboros.config as config
    from ouroboros.review_custody import prepare_frozen_review_reconciliation
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    config_margin = config.NESTED_SETTLEMENT_MARGIN_SEC
    release, entered, workers = threading.Event(), threading.Event(), []
    ctx = SimpleNamespace(_review_paid_stamp=lambda: pytest.fail("a rejoin paid"))
    prepare_frozen_review_reconciliation(ctx, SimpleNamespace(triad_raw_results=[{
        "slot_id": "slot-1", "model_id": "cursor/test", "status": "error",
        "operation_id": "op-existing", "operation_state": "in_flight",
        "late_result_pending": True, "pending_invocation_id": "inv-existing",
    }], scope_raw_result={}))
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="wait-clock-rejoin",
        retry_key="commit_review:wait-clock-rejoin", reconcile_only=True,
        drain_deadline=time.monotonic(),
    )
    slot = ReviewSlot(
        slot_id="slot-1", model="cursor/test", route=ReviewRouteKind.AGENT_SESSION,
    )
    assert config_margin > 0, "the settlement margin must leave the rejoin a window"
    try:
        [actor] = _run(tmp_path, request, slot,
                       _blocking_worker(release, entered, workers), ctx=ctx)
        assert actor.operation_id == "op-existing"
        assert actor.operation_state == "pending_dispatch"
        assert actor.awaiting_since == ""
        assert "awaiting_since" not in _review_actor_projection(
            dataclasses.asdict(actor), "multi_model_review")
    finally:
        release.set()
        assert entered.wait(10), "the rejoined review worker never started"
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive(), "the rejoined review worker did not settle"


@pytest.mark.parametrize("stored", ["", "since yesterday", "2026-13-45T99:99Z"])
def test_a_stored_value_that_is_not_an_instant_projects_no_moment(stored):
    """An unparseable record is a hole, never a rendered clock."""
    row = dataclasses.asdict(ReviewActorRecord(
        slot_id="slot-1", model="test/model", status="error",
        error="Pending dispatch; the physical review operation is in flight (window 60s)",
        operation_id="op-1", operation_state="pending_dispatch",
        late_result_pending=True, awaiting_since=stored,
    ))
    assert "awaiting_since" not in _review_actor_projection(row, "task_acceptance")
    # The guard fires in the other direction on the identical row.
    row["awaiting_since"] = "2026-09-19T08:41:00+00:00"
    assert _review_actor_projection(row, "task_acceptance")["awaiting_since"] == (
        "2026-09-19T08:41:00+00:00")


def test_a_settled_row_never_carries_a_moment_even_when_one_was_stored():
    """Only the two wait predicates publish the field; a settled row is an answer."""
    for state in ("settled", "late_settled", "not_dispatched"):
        row = dataclasses.asdict(ReviewActorRecord(
            slot_id="slot-1", model="test/model", status="error", error="run failed",
            operation_id="op-1", operation_state=state,
            awaiting_since="2026-09-19T08:41:00+00:00",
        ))
        assert "awaiting_since" not in _review_actor_projection(row, "task_acceptance"), state
    for state in ("pending_dispatch", "in_flight", "custody_lost"):
        row = dataclasses.asdict(ReviewActorRecord(
            slot_id="slot-1", model="test/model", status="error", error="waiting",
            operation_id="op-1", operation_state=state, late_result_pending=True,
            awaiting_since="2026-09-19T08:41:00+00:00",
        ))
        assert _review_actor_projection(row, "task_acceptance")["awaiting_since"] == (
            "2026-09-19T08:41:00+00:00"), state


def test_a_row_recorded_before_the_field_existed_still_loads():
    """Every rebuild path of a stored actor row survives the older shape."""
    from ouroboros.review_custody import _frozen_actor
    from ouroboros.tools.plan_ask import reply_facts_from_row
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor

    legacy = dataclasses.asdict(ReviewActorRecord(
        slot_id="slot-1", model="test/model", status="error", error="waiting",
        operation_id="op-1", operation_state="pending_dispatch", late_result_pending=True,
    ))
    legacy.pop("awaiting_since")
    assert "awaiting_since" not in legacy

    # 1. the persisted producer outcome, rebuilt by recover_review_producer
    assert ReviewActorRecord(**legacy).awaiting_since == ""
    # 2. the frozen wave row, rebuilt for a free replay
    frozen = _frozen_actor(legacy, SimpleNamespace(slot_id="slot-1", model="test/model"))
    assert frozen.awaiting_since == ""
    # 3. the plan record's participant facts, read off the plan row
    assert reply_facts_from_row(_plan_row_from_actor(legacy, None))["awaiting_since"] == ""
    # 4. the projection every surface reads
    assert "awaiting_since" not in _review_actor_projection(legacy, "plan")


def test_the_plan_participant_row_carries_the_moment_the_host_recorded(tmp_path):
    """The field reaches the plan record's participant row, not only the acceptance panel."""
    from ouroboros.tools.plan_ask import reply_facts_from_row
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor

    actor = dataclasses.asdict(ReviewActorRecord(
        slot_id="slot-1", model="test/model", status="error", error="waiting",
        operation_id="op-1", operation_state="pending_dispatch", late_result_pending=True,
        awaiting_since="2026-09-19T08:41:00+00:00",
    ))
    row = _plan_row_from_actor(actor, None)
    assert row["awaiting_since"] == "2026-09-19T08:41:00+00:00"
    facts = reply_facts_from_row(row)
    assert facts["status"] == "pending" and facts["awaiting_since"] == "2026-09-19T08:41:00+00:00"
    # A settled plan row keeps the honest empty value it was recorded with.
    settled = {**actor, "operation_state": "settled", "late_result_pending": False, "awaiting_since": "",
               "status": "ok", "error": "", "raw_text": '{"prose": "fine"}'}
    facts = reply_facts_from_row(_plan_row_from_actor(settled, None))
    assert facts["status"] == "answered" and facts["awaiting_since"] == ""
