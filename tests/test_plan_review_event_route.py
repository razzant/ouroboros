"""``plan_task``'s event route (owner batch 2, Q2=A) on the planning organ: an ask returns at
the dispatch barrier, the reviewer workers settle into process-local custody, every settlement
lands in the plan record through ``plan_ask.settle_released_reply``, and the last one writes
ONE system frame into the task mailbox; nothing is sent twice.

Two levels: the custody drain loop itself (``run_custodied_review_slots`` with a held worker,
over a planted plan record) and a ``plan_task`` ask through the REAL review substrate with a
held executor (no ``run_review_request`` stub, no custody stub).
"""

from __future__ import annotations

import pathlib
import threading
import time

from tests.test_plan_task_tool import CLEAN, _call, _record, _write
from tests.test_plan_task_tool import harness as _plan_task_harness

harness = _plan_task_harness  # noqa: F811 - pytest fixture re-export

# The held executor's default answer is the retired engine's clean findings array: the
# earlier-schema suites that still import these helpers read it; organ tests pass ``CLEAN``.
_LEGACY_ANSWER = "[]\nNO_FINDINGS"
_TASK = "event-route"


def _mailbox_entries(drive, task_id):
    from ouroboros.owner_mailbox import drain_owner_entries

    return drain_owner_entries(pathlib.Path(drive), task_id, set())


def _custody_kwargs(tmp_path, *, surface, retry_key, slots, run_slot, ctx, plan_ask_id=""):
    """One custodied request of ``slots``; ``plan_ask_id`` makes it the organ's ask (the
    reconciliation identity ``plan_ask.dispatch_ask`` sends), whose releases settle into
    the plan record."""
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest
    from ouroboros.usage_accounting import UsageScope

    request = ReviewRequest(
        surface=surface, goal="review", task_id=_TASK, retry_key=retry_key,
        reconciliation_identity={"subject_hash": "f" * 64, **({"plan_ask_id": plan_ask_id} if plan_ask_id else {})},
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
            parse_status="valid",  # fixture replaces the producer after its plan-reply parse
            error=row.get("error", ""), operation_id=operation_id,
            operation_state=row.get("operation_state", "settled"),
        )

    return calls, run_slot


def _plant_ask(root, slots, ask_id="a1"):
    """The plan record ``plan_ask.dispatch_ask`` leaves at the barrier: one revision and an
    ask whose seats were dispatched and are still pending."""
    from ouroboros import plan_record

    plan_record.append_revision(root, _TASK, plan_record.compose_revision(
        None, goal="Ship the deck", plan="Outline first.", done_when=[], affected_paths=[]), body=False)
    return plan_record.append_ask(root, _TASK, {"ask_id": ask_id, "revision": 1, "participants": [
        {"seat_id": str(slot.slot_id), "model": str(slot.model), "status": "pending", "dispatched": True,
         "owner_marked": True} for slot in slots]})


def _statuses(root):
    from ouroboros import plan_record

    return {p["seat_id"]: p["status"] for p in plan_record.load_plan_record(root, _TASK)["asks"][0]["participants"]}


def test_drain_deadline_releases_pending_dispatch_rows_and_the_last_settlement_writes_one_frame(tmp_path):
    import ouroboros.review_custody as custody
    from types import SimpleNamespace
    from ouroboros.review_substrate import ReviewSlot

    slots = [ReviewSlot(slot_id="s1", model="m/a", timeout_sec=30.0),
             ReviewSlot(slot_id="s2", model="m/b", timeout_sec=30.0)]
    _plant_ask(tmp_path, slots)
    release = {s.slot_id: threading.Event() for s in slots}
    entered = {s.slot_id: threading.Event() for s in slots}
    calls, run_slot = _held_worker(slots, {}, release, entered)
    progress = []
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=progress.append)
    request, kwargs = _custody_kwargs(
        tmp_path, surface="plan_review", retry_key=f"plan_ask:{_TASK}:a1",
        slots=slots, run_slot=run_slot, ctx=ctx, plan_ask_id="a1")
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
        while _statuses(tmp_path)["s1"] == "pending" and time.time() < deadline:
            time.sleep(0.01)
        assert _statuses(tmp_path) == {"s1": "answered", "s2": "pending"}  # each settlement lands on its own
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
    assert frames[0]["text"].startswith("Plan ask a1: every participant settled — 2 answered; no open objection")
    assert _statuses(tmp_path) == {"s1": "answered", "s2": "answered"}
    assert not custody._RELEASED_WAVES
    # A re-run of the same request replays both settled actors with no second send.
    request.drain_deadline = time.monotonic()
    collected = custody.run_custodied_review_slots(**kwargs)
    assert [a.status for a in collected] == ["ok", "ok"]
    assert calls == ["s1", "s2"]
    assert len(_mailbox_entries(tmp_path, request.task_id)) == 1


def test_typed_zero_refusal_settled_after_release_replays_at_collection(tmp_path):
    import ouroboros.review_custody as custody
    from types import SimpleNamespace
    from ouroboros.review_substrate import ReviewSlot

    slots = [ReviewSlot(slot_id="s1", model="m/a", timeout_sec=30.0)]
    _plant_ask(tmp_path, slots)
    release = {"s1": threading.Event()}
    entered = {"s1": threading.Event()}
    calls, run_slot = _held_worker(
        slots, {"s1": {"status": "not_dispatched", "raw_text": "",
                       "error": "Owner deadline exhausted before physical review dispatch",
                       "operation_state": "not_dispatched"}}, release, entered)
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=lambda _m: None)
    request, kwargs = _custody_kwargs(
        tmp_path, surface="plan_review", retry_key=f"plan_ask:{_TASK}:a1",
        slots=slots, run_slot=run_slot, ctx=ctx, plan_ask_id="a1")
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
    assert _statuses(tmp_path) == {"s1": "not_dispatched"}  # the plan record keeps the typed $0 fact


def test_requests_without_a_drain_deadline_and_other_surfaces_are_untouched(tmp_path):
    import ouroboros.review_custody as custody
    from types import SimpleNamespace
    from ouroboros.review_substrate import ReviewSlot

    slots = [ReviewSlot(slot_id="s1", model="m/a", timeout_sec=30.0)]
    _plant_ask(tmp_path, slots)
    release = {"s1": threading.Event()}
    entered = {"s1": threading.Event()}
    release["s1"].set()
    calls, run_slot = _held_worker(slots, {}, release, entered)
    progress = []
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=progress.append)
    request, kwargs = _custody_kwargs(
        tmp_path, surface="plan_review", retry_key=f"plan_ask:{_TASK}:a1",
        slots=slots, run_slot=run_slot, ctx=ctx, plan_ask_id="a1")
    assert request.drain_deadline is None
    [actor] = custody.run_custodied_review_slots(**kwargs)  # waits for the worker as before
    assert actor.status == "ok" and actor.operation_state == "settled"
    assert not custody._RELEASED_WAVES
    assert len(progress) == 2 and "reviewer m/a started" in progress[0] and "m/a answered" in progress[1]
    assert _mailbox_entries(tmp_path, request.task_id) == []
    # Not released, so not the hook's: the dispatcher records this answer itself.
    assert _statuses(tmp_path) == {"s1": "pending"}
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


def test_a_slot_settling_during_the_barrier_release_never_splits_the_wave_into_two_frames(tmp_path, monkeypatch):
    """Fix cycle 2, 2b: the released roster is registered atomically before any
    slot can complete it. A slot that settles (an immediate typed refusal) while the
    coordinator is still minting the other released rows must not complete a one-slot
    roster and mint a second frame when the next slot settles."""
    import ouroboros.review_custody as custody
    from types import SimpleNamespace
    from ouroboros.review_substrate import ReviewSlot

    slots = [ReviewSlot(slot_id="s1", model="m/a", timeout_sec=30.0),
             ReviewSlot(slot_id="s2", model="m/b", timeout_sec=30.0)]
    _plant_ask(tmp_path, slots)
    release = {s.slot_id: threading.Event() for s in slots}
    entered = {s.slot_id: threading.Event() for s in slots}
    calls, run_slot = _held_worker(
        slots, {"s1": {"status": "not_dispatched", "raw_text": "",
                       "error": "daemon unreachable before physical review dispatch",
                       "operation_state": "not_dispatched"}}, release, entered)
    progress = []
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=progress.append)
    request, kwargs = _custody_kwargs(
        tmp_path, surface="plan_review", retry_key=f"plan_ask:{_TASK}:a1",
        slots=slots, run_slot=run_slot, ctx=ctx, plan_ask_id="a1")
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
        assert _wait_until(lambda: _statuses(tmp_path)["s1"] == "not_dispatched")
        assert entered["s2"].wait(10)
        release["s2"].set()
        assert _wait_until(lambda: len(_mailbox_entries(tmp_path, request.task_id)) >= 1)
        time.sleep(0.2)
    finally:
        for event in release.values():
            event.set()
    frames = _mailbox_entries(tmp_path, request.task_id)
    assert len(frames) == 1, [f["text"] for f in frames]
    assert frames[0]["text"].startswith("Plan ask a1: every participant settled — 1 answered; no open objection")
    assert _statuses(tmp_path) == {"s1": "not_dispatched", "s2": "answered"}
    assert not custody._RELEASED_WAVES


class _HeldExecutor:
    """A plan-review api_chat executor whose sends block until the test releases them."""

    def __init__(self, answer=_LEGACY_ANSWER):
        self.execute_calls = 0
        self.release = threading.Event()
        self.answer = answer

    def restore_custody(self, _custody_state):
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
        return ReviewAttemptResult(message={"content": self.answer}, usage=usage, raw_text=self.answer)

    def failure_custody(self):
        return {}


def _install_real_substrate(monkeypatch, answer=_LEGACY_ANSWER):
    """No run_review_request stub and no custody stub: the REAL drain loop runs."""
    executor = _HeldExecutor(answer)
    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor",
                        lambda *_a, **_k: executor)
    return executor


def _wait_until(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.02)
    return predicate()


def test_an_ask_returns_at_the_barrier_and_its_seats_settle_into_the_plan_record_once(harness, monkeypatch):
    """Through the REAL substrate: the ask returns at the dispatch barrier with every seat
    pending, each settlement lands in the plan record through the custody hook, the last one
    writes ONE system frame, and a later READ sends nothing again."""
    executor = _install_real_substrate(monkeypatch, answer=CLEAN)
    ctx = harness.make_ctx()
    try:
        text, result = _write(ctx, ask="council")
        assert result.meta["plan_pending_asks"] == 1 and "Still working: s1, s2, s3" in text
        assert [p["status"] for p in _record(harness)["asks"][0]["participants"]] == ["pending"] * 3
        assert _wait_until(lambda: executor.execute_calls == 3)  # every worker reached its send
        assert _mailbox_entries(harness.drive, "task-1") == []
        executor.release.set()
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, "task-1")) == 1)
    finally:
        executor.release.set()
    [frame] = _mailbox_entries(harness.drive, "task-1")
    assert frame["provenance"] == "system"
    assert frame["text"].startswith("Plan ask a1: every participant settled — 3 answered; no open objection")
    assert [p["status"] for p in _record(harness)["asks"][0]["participants"]] == ["answered"] * 3
    text, result = _call(ctx)
    assert result.meta["plan_pending_asks"] == 0 and executor.execute_calls == 3


def test_a_seat_released_at_the_barrier_is_no_last_execution_while_a_settled_one_is():
    """The «runs as» projection (D22) skips a seat released at the dispatch barrier — it is
    running, not failed — and records a settled one."""
    from types import SimpleNamespace

    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.reviewer_slot_config import reviewer_slot_execution_rows

    slots = {sid: ReviewSlot(slot_id=sid, model=f"m/{sid}") for sid in ("running", "settled")}
    rows = reviewer_slot_execution_rows("plan_review", [
        SimpleNamespace(slot_id="running", status="error", error="Pending dispatch; released at the barrier",
                        usage={}, operation_state="pending_dispatch"),
        SimpleNamespace(slot_id="settled", status="ok", error="", usage={}, operation_state="settled"),
    ], slots)
    assert set(rows) == {"settled"} and rows["settled"]["status"] == "ok"
