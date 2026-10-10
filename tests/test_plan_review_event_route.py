"""``plan_task``'s event route (owner batch 2, Q2=A): a fresh dispatch returns at
the dispatch barrier, the reviewer workers settle into process-local custody, the
last settlement writes ONE system frame into the task mailbox, and a later $0
collection closes the wave and pays its cycle exactly once.

Two levels: the custody drain loop itself (``run_custodied_review_slots`` with a
held worker) and the whole engine through the REAL review substrate with a
blocking executor (no ``run_review_request`` stub, no custody stub).
"""

from __future__ import annotations

import pathlib
import threading
import time
from types import SimpleNamespace

import pytest


from tests.test_plan_review_engine import CLEAN, DECK_SPEC, _call, _control, _state
from tests.test_plan_review_engine import harness as _engine_harness

harness = _engine_harness  # noqa: F811 - pytest fixture re-export


def _mailbox_entries(drive, task_id):
    from ouroboros.owner_mailbox import drain_owner_entries

    return drain_owner_entries(pathlib.Path(drive), task_id, set())


def _custody_kwargs(tmp_path, *, surface, retry_key, slots, run_slot, ctx):
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest
    from ouroboros.usage_accounting import UsageScope

    request = ReviewRequest(
        surface=surface, goal="review", task_id="event-route", retry_key=retry_key,
        reconciliation_identity={"subject_hash": "f" * 64},
    )

    ctx.task_id = request.task_id

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
            late_result_pending=operation_state == "in_flight",
        )

    return request, dict(
        request=request, slots=slots, usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(drive_root=tmp_path),
        run_slot=run_slot, error_actor=error_actor,
    )


def _held_worker(slots, results, release, entered):
    from ouroboros.review_substrate import ReviewActorRecord

    calls = []

    def run_slot(slot, operation_id, _retry_state, _deadline, _checkpoint):
        calls.append(slot.slot_id)
        entered[slot.slot_id].set()
        assert release[slot.slot_id].wait(10), "test did not release the review worker"
        row = results.get(slot.slot_id) or {}
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model,
            status=row.get("status", "ok"), raw_text=row.get("raw_text", CLEAN),
            error=row.get("error", ""), operation_id=operation_id,
            operation_state=row.get("operation_state", "settled"),
        )

    return calls, run_slot


def test_drain_deadline_releases_pending_dispatch_rows_and_the_last_settlement_writes_one_frame(tmp_path):
    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewSlot

    slots = [ReviewSlot(slot_id="s1", model="m/a", timeout_sec=30.0),
             ReviewSlot(slot_id="s2", model="m/b", timeout_sec=30.0)]
    release = {s.slot_id: threading.Event() for s in slots}
    entered = {s.slot_id: threading.Event() for s in slots}
    calls, run_slot = _held_worker(slots, {}, release, entered)
    progress = []
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=progress.append)
    request, kwargs = _custody_kwargs(
        tmp_path, surface="plan_review", retry_key="plan_review:" + "f" * 64 + ":1",
        slots=slots, run_slot=run_slot, ctx=ctx)
    request.drain_deadline = time.monotonic()
    try:
        first = custody.run_custodied_review_slots(**kwargs)
        assert {a.operation_state for a in first} == {"pending_dispatch"}
        assert all(a.late_result_pending for a in first)
        assert all(a.error.startswith("Pending dispatch;") for a in first)
        assert all(entered[s.slot_id].wait(10) for s in slots)
        # Released, never timed out: the later settlement is not a ``late`` result.
        assert all(not e.timed_out and e.released_early for e in custody._ACTIVE.values())
        assert _mailbox_entries(tmp_path, request.task_id) == []
        release["s1"].set()
        deadline = time.time() + 10
        while not any("m/a answered" in line for line in progress) and time.time() < deadline:
            time.sleep(0.01)
        assert sum("m/a answered" in line and "reviewer" in line for line in progress) == 1
        assert _mailbox_entries(tmp_path, request.task_id) == []  # one slot still running
        release["s2"].set()
        while len(_mailbox_entries(tmp_path, request.task_id)) < 1 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        for event in release.values():
            event.set()
    frames = _mailbox_entries(tmp_path, request.task_id)
    assert len(frames) == 1
    assert frames[0]["provenance"] == "system" and frames[0]["kind"] == "task_message"
    assert "Plan review wave ffffffff: 2 of 2 reviewer slot(s) settled (2 ok, 0 failed)" in frames[0]["text"]
    assert "not yet collected" in frames[0]["text"]
    assert not custody._RELEASED_WAVES
    # Collection: the same cycle replays both settled actors with no second send.
    request.drain_deadline = time.monotonic()
    collected = custody.run_custodied_review_slots(**kwargs)
    assert [a.status for a in collected] == ["ok", "ok"]
    assert calls == ["s1", "s2"]
    assert len(_mailbox_entries(tmp_path, request.task_id)) == 1


def test_typed_zero_refusal_settled_after_release_replays_at_collection(tmp_path):
    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewSlot

    slots = [ReviewSlot(slot_id="s1", model="m/a", timeout_sec=30.0)]
    release = {"s1": threading.Event()}
    entered = {"s1": threading.Event()}
    calls, run_slot = _held_worker(
        slots, {"s1": {"status": "not_dispatched", "raw_text": "",
                       "error": "Owner deadline exhausted before physical review dispatch",
                       "operation_state": "not_dispatched"}}, release, entered)
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=lambda _m: None)
    request, kwargs = _custody_kwargs(
        tmp_path, surface="plan_review", retry_key="plan_review:" + "e" * 64 + ":1",
        slots=slots, run_slot=run_slot, ctx=ctx)
    request.drain_deadline = time.monotonic()
    try:
        [first] = custody.run_custodied_review_slots(**kwargs)
        assert first.operation_state == "pending_dispatch"
        assert entered["s1"].wait(10)
        release["s1"].set()
        deadline = time.time() + 10
        while not _mailbox_entries(tmp_path, request.task_id) and time.time() < deadline:
            time.sleep(0.01)
    finally:
        release["s1"].set()
    [collected] = custody.run_custodied_review_slots(**kwargs)
    assert collected.operation_state == "not_dispatched"  # typed $0, never custody_lost
    assert calls == ["s1"]


def test_requests_without_a_drain_deadline_and_other_surfaces_are_untouched(tmp_path):
    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewSlot

    slots = [ReviewSlot(slot_id="s1", model="m/a", timeout_sec=30.0)]
    release = {"s1": threading.Event()}
    entered = {"s1": threading.Event()}
    release["s1"].set()
    calls, run_slot = _held_worker(slots, {}, release, entered)
    progress = []
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=progress.append)
    request, kwargs = _custody_kwargs(
        tmp_path, surface="plan_review", retry_key="plan_review:" + "a" * 64 + ":1",
        slots=slots, run_slot=run_slot, ctx=ctx)
    assert request.drain_deadline is None
    [actor] = custody.run_custodied_review_slots(**kwargs)  # waits for the worker as before
    assert actor.status == "ok" and actor.operation_state == "settled"
    assert not custody._RELEASED_WAVES
    assert len(progress) == 2 and "reviewer m/a started" in progress[0] and "m/a answered" in progress[1]
    assert _mailbox_entries(tmp_path, request.task_id) == []
    # A non-plan surface released at a drain deadline gets pending rows but no frame.
    triad, triad_kwargs = _custody_kwargs(
        tmp_path, surface="multi_model_review", retry_key="commit_review:" + "b" * 64,
        slots=slots, run_slot=run_slot, ctx=ctx)
    triad.drain_deadline = time.monotonic()
    release["s1"].clear()
    try:
        [pending] = custody.run_custodied_review_slots(**triad_kwargs)
        assert pending.operation_state == "pending_dispatch"
    finally:
        release["s1"].set()
    deadline = time.time() + 5
    while custody._RELEASED_WAVES and time.time() < deadline:
        time.sleep(0.01)
    assert _mailbox_entries(tmp_path, request.task_id) == []
    # Every row is a reviewer row of its own surface; no plan frame reached the non-plan surface.
    assert all(line.startswith(("Plan reviewer m/a ", "Commit reviewer m/a ")) for line in progress)


class _HeldExecutor:
    """A plan-review api_chat executor whose sends block until the test releases them."""

    def __init__(self):
        self.execute_calls = 0
        self.release = threading.Event()

    def restore_custody(self, _state):
        return None

    def set_pending_invocation_checkpoint(self, _checkpoint):
        return None

    def prompt_payload(self):
        return {"messages": []}

    def prompt_chars(self):
        return 0

    def execute(self):
        from ouroboros.review_execution import ReviewAttemptResult

        self.execute_calls += 1
        assert self.release.wait(20), "test did not release the reviewer send"
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "physical_attempt_state": "settled"}
        return ReviewAttemptResult(message={"content": CLEAN}, usage=usage, raw_text=CLEAN)

    def failure_custody(self):
        return {}


def _install_real_substrate(monkeypatch):
    """No run_review_request stub and no custody stub: the REAL drain loop runs."""
    executor = _HeldExecutor()
    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor",
                        lambda *_a, **_k: executor)
    return executor


def _wait_until(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.02)
    return predicate()


def test_default_native_plan_reads_workspace_and_collects_exact_paid_wave(harness, monkeypatch):
    """Real pool slots (three natively retrieving api seats), native executor,
    inspection registry and collector."""
    from collections import Counter
    from ouroboros.tools.plan_review_runtime import plan_review_slots
    from ouroboros.usage_accounting import current_usage_scope
    from tests.review_pool_rosters import set_review_pool
    from tests.test_native_tool_round_executor import _tool_call

    set_review_pool(monkeypatch, delivery="native")
    harness.state['slots'] = plan_review_slots()
    assert len(harness.state['slots']) == 3
    assert all(slot.native_retrieval and slot.subagent_id == slot.slot_id for slot in harness.state['slots'])
    calls, release = [], threading.Event()

    def transport(_self, **kwargs):
        slot_id = current_usage_scope().review_slot_id
        calls.append(slot_id)
        assert kwargs['tools'] and release.wait(10)
        observed = [m for m in kwargs['messages'] if m.get('role') == 'tool']
        if observed:
            assert 'deck notes' in observed[-1]['content']
            answer = {'content': CLEAN}
        else:
            answer = {'tool_calls': [_tool_call('read_file', {'path': 'notes.md', 'root': 'active_workspace'})]}
        return answer, {'prompt_tokens': 10, 'completion_tokens': 5, 'cost': 0,
                        'physical_attempt_state': 'settled'}

    monkeypatch.setattr('ouroboros.llm.LLMClient.chat', transport)
    ctx = harness.make_ctx()
    try:
        assert not _control(_call(ctx))['closed']
        assert _wait_until(lambda: len(calls) == 3)
    finally:
        release.set()
    assert _wait_until(lambda: len(_mailbox_entries(harness.drive, ctx.task_id)) == 1)
    answer = _call(ctx)
    assert _control(answer) == {'outcome': 'GREEN', 'closed': True}, answer
    assert set(Counter(calls).values()) == {2} and len(calls) == 6
    assert _state(harness)['cycles_paid'] == 1
    assert _control(_call(ctx)) == {'outcome': 'GREEN', 'closed': True}
    assert len(calls) == 6


def test_fresh_dispatch_returns_at_the_barrier_and_the_resubmitted_envelope_collects_once(harness, monkeypatch):
    executor = _install_real_substrate(monkeypatch)
    ctx = harness.make_ctx()
    try:
        first = _call(ctx)
        assert _control(first) == {"outcome": "DEGRADED", "closed": False}
        state = _state(harness)
        wave = state["waves"][-1]
        assert wave["custody_pending"] is True and wave["paid"] is False
        assert state["cycles_paid"] == 0  # paid iff dispatched: nothing proven yet
        assert {a["operation_state"] for a in wave["actors"]} == {"pending_dispatch"}
        assert "REVIEW CUSTODY PENDING" in first
        assert _wait_until(lambda: executor.execute_calls == 3)  # every worker reached its send
        assert _mailbox_entries(harness.drive, "task-1") == []
        executor.release.set()
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, "task-1")) == 1)
    finally:
        executor.release.set()
    [frame] = _mailbox_entries(harness.drive, "task-1")
    assert frame["provenance"] == "system"
    assert frame["text"].startswith(f"Plan review wave {wave['request_fingerprint'][:8]}: 3 of 3 reviewer slot(s) settled (3 ok, 0 failed)")
    # The frame reaches the model as a system task message, never as an owner directive.
    from ouroboros.loop_round_limits import _drain_incoming_messages
    import queue

    messages, owner_ctx = [], SimpleNamespace()
    _drain_incoming_messages(messages, queue.Queue(), harness.drive, "task-1", None, set(), owner_ctx=owner_ctx)
    assert messages and messages[0]["content"].startswith("[System task message]\nPlan review wave")
    assert getattr(owner_ctx, "_owner_directives", []) == []
    # The identical envelope is the existing resume path: it collects the settled
    # slots (one cycle, no second send) and closes the wave.
    second = _call(ctx)
    assert _control(second) == {"outcome": "GREEN", "closed": True}, second
    state = _state(harness)
    assert state["cycles_paid"] == 1 and state["waves"][-1]["paid"] is True
    assert executor.execute_calls == 3
    # The final mailbox frame can precede another slot's progress callback.
    assert _wait_until(lambda: any("m/a answered" in line and "reviewer" in line for line in harness.progress))


def test_barrier_wave_replaces_a_stale_paid_predecessor_and_pays_only_at_collection(tmp_path):
    """The D2 narrowing (an unpaid wave never replaces a paid predecessor) does not
    swallow the barrier wave of a re-dispatched stale DEGRADED envelope."""
    from ouroboros.task_results import load_plan_review_state, record_plan_review_wave
    from tests.test_plan_review import _wave

    fp = "c" * 64
    record_plan_review_wave(tmp_path, "t", {**_wave(fp, aggregate="DEGRADED"), "actors": []})
    barrier = {**_wave(fp, aggregate="DEGRADED"), "cycle_index": 2, "paid": False, "custody_pending": True,
               "actors": [{"slot_id": "s1", "operation_state": "pending_dispatch"}]}
    record_plan_review_wave(tmp_path, "t", barrier)
    state = load_plan_review_state(tmp_path, "t")
    assert state["cycles_paid"] == 1 and state["waves"][-1]["custody_pending"] is True
    collected = {**barrier, "paid": True, "custody_pending": False, "aggregate": "GREEN", "closed": True,
                 "actors": [{"slot_id": "s1", "operation_state": "settled", "physical_attempt_state": "settled"}]}
    record_plan_review_wave(tmp_path, "t", collected)
    state = load_plan_review_state(tmp_path, "t")
    assert state["cycles_paid"] == 2 and state["waves"][-1]["closed"] is True


def test_in_flight_panels_count_toward_the_cycle_cap_at_dispatch(harness, monkeypatch):
    """Fix cycle 1, F1: a dispatched panel commits its cycle at the barrier. Under
    OUROBOROS_REVIEW_MAX_CYCLES=1 a revised envelope submitted while the first panel is
    still in flight buys no second panel: it is refused with the typed cap state."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    executor = _install_real_substrate(monkeypatch)
    ctx = harness.make_ctx()
    try:
        first = _call(ctx)
        assert _control(first) == {"outcome": "DEGRADED", "closed": False}
        assert _wait_until(lambda: executor.execute_calls == 3)
        first_fp = _state(harness)["waves"][-1]["request_fingerprint"]
        second = _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 6-slide deck"]})
        assert second.startswith("ERROR: PLAN_REVIEW_IN_FLIGHT:") and first_fp in second
        third = _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 7-slide deck"]})
        assert executor.execute_calls == 3, "no panel beyond the cap was dispatched"
        assert third.startswith("ERROR: PLAN_REVIEW_IN_FLIGHT:")
        state = _state(harness)
        assert state["cycles_paid"] == 0  # committed, not yet proven paid: nothing is written as spent
        assert state["current_attempt"]["fingerprint"] == first_fp  # the in-flight wave stays current
        assert not any(line.startswith("📐 Plan review: no review rounds left") for line in harness.progress)
    finally:
        executor.release.set()
    assert _wait_until(lambda: len(_mailbox_entries(harness.drive, "task-1")) == 1)


def test_the_barrier_records_no_failed_last_execution_for_running_slots(harness, monkeypatch, tmp_path):
    """Fix cycle 1, F3: a slot released at the dispatch barrier is running, not failed.
    The last-execution projection (Settings and the capabilities digest) is written when
    the slot settles, never at the barrier with an error status."""
    from ouroboros import reviewer_slot_config

    monkeypatch.setattr(reviewer_slot_config, "_last_execution_path", lambda: tmp_path / "last.json")
    executor = _install_real_substrate(monkeypatch)
    ctx = harness.make_ctx()
    try:
        _call(ctx)
        assert {a["operation_state"] for a in _state(harness)["waves"][-1]["actors"]} == {"pending_dispatch"}
        last = reviewer_slot_config.reviewer_slot_last_executions()
        assert not [sid for sid, row in last.items() if row.get("status") == "error"], last
        assert _wait_until(lambda: executor.execute_calls == 3)
        executor.release.set()
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, "task-1")) == 1)
    finally:
        executor.release.set()
    assert _control(_call(ctx)) == {"outcome": "GREEN", "closed": True}  # the collection
    last = reviewer_slot_config.reviewer_slot_last_executions()
    assert {sid: row["status"] for sid, row in last.items()} == {"s1": "ok", "s2": "ok", "s3": "ok"}


def test_a_slot_settling_during_the_barrier_release_never_splits_the_wave_into_two_frames(tmp_path, monkeypatch):
    """Fix cycle 2, 2b: the released roster is registered atomically before any
    slot can complete it. A slot that settles (an immediate typed refusal) while the
    coordinator is still minting the other released rows must not complete a one-slot
    roster and mint a second frame when the next slot settles."""
    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewSlot

    slots = [ReviewSlot(slot_id="s1", model="m/a", timeout_sec=30.0),
             ReviewSlot(slot_id="s2", model="m/b", timeout_sec=30.0)]
    release = {s.slot_id: threading.Event() for s in slots}
    entered = {s.slot_id: threading.Event() for s in slots}
    calls, run_slot = _held_worker(
        slots, {"s1": {"status": "not_dispatched", "raw_text": "",
                       "error": "daemon unreachable before physical review dispatch",
                       "operation_state": "not_dispatched"}}, release, entered)
    progress = []
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=progress.append)
    request, kwargs = _custody_kwargs(
        tmp_path, surface="plan_review", retry_key="plan_review:" + "d" * 64 + ":1",
        slots=slots, run_slot=run_slot, ctx=ctx)
    request.drain_deadline = time.monotonic()
    original = custody._late_or_timeout_actor

    def interleaved(slot, entry, timeout, error_actor, *, released_early=False):
        actor = original(slot, entry, timeout, error_actor, released_early=released_early)
        if released_early and slot.slot_id == "s1":
            # s1 settles right after its own row is minted, before s2's row exists.
            assert entered["s1"].wait(10)
            release["s1"].set()
            assert _wait_until(lambda: any("m/a wasn't sent" in line for line in progress))
        return actor

    monkeypatch.setattr(custody, "_late_or_timeout_actor", interleaved)
    try:
        first = custody.run_custodied_review_slots(**kwargs)
        assert {a.operation_state for a in first} == {"pending_dispatch"}
        assert _mailbox_entries(tmp_path, request.task_id) == []  # s2 still running: no frame yet
        assert entered["s2"].wait(10)
        release["s2"].set()
        assert _wait_until(lambda: len(_mailbox_entries(tmp_path, request.task_id)) >= 1)
        time.sleep(0.2)
    finally:
        for event in release.values():
            event.set()
    frames = _mailbox_entries(tmp_path, request.task_id)
    assert len(frames) == 1, [f["text"] for f in frames]
    assert "2 of 2 reviewer slot(s) settled (1 ok, 1 failed)" in frames[0]["text"]
    assert not custody._RELEASED_WAVES


def test_a_collection_records_the_dispatched_packet_not_one_rebuilt_from_the_live_corpus(harness, monkeypatch):
    """Owner-forwarded audit, finding 1: a $0 collection must not rewrite the history of
    what the reviewers saw. An owner directive that arrives AFTER every slot was dispatched
    is a live directive of the task, but it was in no physically sent packet: it must appear
    in no artifact of that wave and in no prior history the next paid cycle continues from."""
    import copy
    import json

    from ouroboros.tools.plan_review import _handle_plan_task
    from ouroboros.tools.plan_review_artifacts import authority_wave, continuation_inputs

    executor = _HeldExecutor()
    sent = []

    def factory(assignment, **_kw):
        if not assignment.request.reconcile_only:
            sent.append(copy.deepcopy(assignment.request.messages))
        return executor

    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor", factory)
    ctx = harness.make_ctx()
    ctx._owner_directives = [{"source": "initial_user", "content": "Original scope: make the chart blue."}]
    late = "OWNER CHANGE ARRIVED AFTER DISPATCH: make the chart red."
    try:
        _call(ctx)
        assert _wait_until(lambda: executor.execute_calls == 3)
        before = authority_wave(harness.drive, "task-1", _state(harness)["waves"][-1])
        fp = before["request_fingerprint"]
        before_messages = copy.deepcopy(before["reviewer_outputs"][0]["request_messages"])
        assert late not in json.dumps(sent) and late not in json.dumps(before_messages)
        ctx._owner_directives.append({"source": "owner_mailbox", "content": late, "msg_id": "later-1"})
        executor.release.set()
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, "task-1")) == 1)
        collected = _handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": []})
    finally:
        executor.release.set()
    assert _control(collected) == {"outcome": "GREEN", "closed": True}
    after = authority_wave(harness.drive, "task-1", _state(harness)["waves"][-1])
    after_messages = after["reviewer_outputs"][0]["request_messages"]
    assert after["request_fingerprint"] == fp and executor.execute_calls == 3
    assert late not in json.dumps(sent), "no dispatched reviewer saw the later owner directive"
    assert late not in json.dumps(after_messages), "the collection re-recorded the dispatched packet"
    assert after_messages == before_messages
    # The next paid cycle continues from that same recorded history, never from the rebuild.
    _slots, history, _threads, cause = continuation_inputs(
        harness.drive, "task-1", after, harness.state["slots"], user_content="Next paid review turn")
    assert cause == {} and history
    assert late not in json.dumps(history["s1"][:-2]), "the late directive entered the prior history"
    assert history["s1"][:-2] == before_messages


def test_the_open_wave_text_names_the_route_that_waits_for_the_settlement_frame(harness, monkeypatch):
    """Owner-forwarded audit, finding 5: the barrier return says paid operations are still
    in flight but never says how to wait for them. The auditor verified the working route
    (``wait_task`` on the task's OWN id returns on the settlement frame, then the $0
    collection) and that ``schedule_followup`` is the wrong one here: it mints a NEW root
    task whose collection of the old wave is refused as PLAN_REVIEW_DISPOSITION_UNBINDABLE."""
    from ouroboros.tools.plan_render import _next_step

    executor = _install_real_substrate(monkeypatch)
    ctx = harness.make_ctx()
    before = set(threading.enumerate())
    workers = []
    try:
        try:
            first = _call(ctx)
            assert _wait_until(lambda: executor.execute_calls == 3)
            workers = [t for t in threading.enumerate()
                       if t not in before and t.name.startswith("ouroboros-review-plan_review-")]
            assert sorted(t.name for t in workers) == [f"ouroboros-review-plan_review-s{i}" for i in (1, 2, 3)]
        finally:
            executor.release.set()
        wave = _state(harness)["waves"][-1]
        fp = wave["request_fingerprint"]
        for text in (first, _next_step(wave, enforcement="blocking", cap=2, cycles_paid=0)):
            # "paid" is not claimed: a slot released at the barrier is $0 until its row proves the send.
            assert "one or more reviewer operations are still in flight" in text and "paid reviewer" not in text
            assert ("The host writes ONE message into this task's mailbox when every released slot "
                    "settles: wait_task on this task's own id (wait_tasks while children run) "
                    "returns on it") in text
            assert f"plan_task(review_disposition={{review_fingerprint: '{fp}', items: []}})" in text
            assert "schedule_followup" not in text  # a new root task cannot collect this wave
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, "task-1")) == 1)
    finally:
        executor.release.set()
        # Even a failed assertion must not leave this test's released reviewers alive.
        # The positive path above still checks the exact settlement frame.
        retired = _wait_until(lambda: not any(
            t.is_alive() for t in threading.enumerate()
            if t not in before and t.name.startswith("ouroboros-review-plan_review-")))
    assert retired


@pytest.mark.parametrize("effort", ["low", "high", "none", "default"])
@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_neutral_collection_uses_the_paid_wave_not_schema_filled_overrides(harness, monkeypatch, effort, enforcement):
    """Real substrate and saved source: padded fields never buy or author a new wave."""
    import copy
    from ouroboros.tools import plan_review as pr
    from ouroboros.tools.plan_review_artifacts import authority_wave

    harness.state["enforcement"] = enforcement
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    executor = _install_real_substrate(monkeypatch)
    ctx = harness.make_ctx()
    try:
        _call(ctx)
        wave = _state(harness)["waves"][-1]
        fingerprint = wave["request_fingerprint"]
        exact = authority_wave(harness.drive, ctx.task_id, wave)
        assert _wait_until(lambda: executor.execute_calls == 3)
        for author in ({"disposition": "partial", "rationale": "Collect"},
                       {"disposition": "deferred", "rationale": ""}):
            payload = {"goal": "", "plan": "", "spec": {k: [] for k in DECK_SPEC},
                "reviewer_effort": effort, "review_disposition": {"review_fingerprint": fingerprint,
                "items": [], "author_action": "none", "author_disposition": author}}
            original = copy.deepcopy(payload)
            result = pr._handle_plan_task(ctx, **payload)
            assert _control(result) == {"outcome": "DEGRADED", "closed": False}
            pending = _state(harness)
            assert pending["waves"][-1]["custody_pending"]
            assert not pending["waves"][-1].get("author_disposition")
            assert not pending["current_attempt"].get("author_subject")
            assert payload == original and executor.execute_calls == 3
        executor.release.set()
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, ctx.task_id)) == 1)
        result = pr._handle_plan_task(ctx, **payload)
        assert _control(result) == {"outcome": "GREEN", "closed": True}
        state = _state(harness)
        collected = authority_wave(harness.drive, ctx.task_id, state["waves"][-1])
        assert executor.execute_calls == 3 and state["cycles_paid"] == 1
        assert len(state["waves"]) == 1 and collected["request_fingerprint"] == fingerprint
        assert collected["reviewer_effort"] == exact["reviewer_effort"] == ""
        assert collected["reviewer_config_fingerprint"] == exact["reviewer_config_fingerprint"]
        assert not collected.get("author_disposition") and payload == original
    finally:
        executor.release.set()
    assert _wait_until(lambda: len(_mailbox_entries(harness.drive, ctx.task_id)) == 1)
