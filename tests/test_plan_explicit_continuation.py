"""A stopped Plan participant continues only by ``plan_task(continue_seats=...)``, inside its own ask.

The participant's first ask stops (no answered row exists, so no earlier thread is known to the
next-ask continuation); its original START body holds the engine thread it ran on. The explicit
choice records one decision in the ask's ledger record, reopens the same participant of the same
ask and revision, and sends ONE new turn of that thread (never ``continueFrom``, which the engine
refuses on a thread turn); no new ask is prepared. Only the engine is scripted: the plan record,
substrate, custody, ledger and the released-reply hook are real.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

from ouroboros import plan_record
from ouroboros.review_ledger import load_record
from tests._review_session_route_shared import FakeGateway, _owned_gateway_uses_each_test_transport  # noqa: F401
from tests._review_session_route_shared import _terminal_detail, fake_route as _fake_route
from tests.test_plan_task_tool import CLEAN, _call, _write
from tests.test_plan_task_tool import harness as _harness

harness, fake_route = _harness, _fake_route
pytestmark = pytest.mark.serial
RESUMABLE = {"cause": "vendor_limit", "carriers": ["native"], "resetsAt": None, "session": None,
             "workspace": {"kind": "none", "root": None}}
OFFER = "Continuable stopped participants (plan_task(continue_seats=[{seat_id, operation_id}])): "


@pytest.fixture
def plan_session(harness, fake_route, monkeypatch):
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_review_runtime import PLAN_REVIEW_MAX_TOKENS

    harness.state["slots"] = [ReviewSlot(
        slot_id="s2", model="fake-review=fake-small", effort="high", role_hint="plan reviewer", subagent_id="s2",
        route=ReviewRouteKind.AGENT_SESSION, session_target="fake-review=fake-small",
        max_tokens=PLAN_REVIEW_MAX_TOKENS, default_temperature=0.2)]
    engine = SimpleNamespace(threads=[], turns=[], keys={}, fates={"run-1": "stop"}, runs=[])

    def create_thread(self, request, *, idempotency_key):
        engine.threads.append(dict(request))
        return {"id": "thread-1"}

    def start_thread_turn(self, thread_id, request, *, idempotency_key):
        engine.turns.append((thread_id, dict(request), str(idempotency_key)))
        run_id = engine.keys.setdefault(str(idempotency_key), f"run-{len(engine.keys) + 1}")
        return {"runId": run_id, "turnId": f"turn-{run_id}"}

    def get_run(self, run_id, **_kw):
        engine.runs.append(run_id)
        if engine.fates.get(run_id) == "stop":
            detail = _terminal_detail("PARTIAL PLAN NOTES", state="failed")
            detail["summary"].update(failure={"code": "vendor_limit", "safeMessage": "Vendor window reached"},
                                     resumable=dict(RESUMABLE))
            return detail
        return _terminal_detail(CLEAN)

    monkeypatch.setattr(FakeGateway, "create_thread", create_thread, raising=False)
    monkeypatch.setattr(FakeGateway, "start_thread_turn", start_thread_turn, raising=False)
    monkeypatch.setattr(FakeGateway, "get_thread", lambda self, thread_id: {"id": thread_id, "turns": [
        {"id": f"turn-{run}", "runId": run} for run in engine.keys.values()]}, raising=False)
    monkeypatch.setattr(FakeGateway, "get_run", get_run)
    harness.engine = engine
    return harness


def _participant(h, task_id="task-1"):
    ask = plan_record.load_plan_record(h.drive, task_id)["asks"][-1]
    return ask, ask["participants"][0]


def _eventually(read, ok, what):
    """The released-reply hook settles on its own thread: the participant first, then the ledger seat and the frame."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        value = read()
        if ok(value):
            return value
        time.sleep(0.05)
    raise AssertionError(f"{what} never happened: {read()}")


def _settled(h, status):
    return _eventually(lambda: _participant(h)[1], lambda participant: participant.get("status") == status,
                       f"participant reaching {status}")


def _offered(h, ctx):
    """The reply hook writes the participant first, its ledger seat next: only then does the read offer it."""
    stopped = _settled(h, "failed")
    text = _eventually(lambda: _call(ctx)[0], lambda read: OFFER in read, "the read offering the stopped participant")
    return stopped, json.loads(text.split(OFFER)[1])


def _hold_the_stopped_worker(monkeypatch):
    """The stopped seat's worker parks right after its reply hook (participant failed, ledger seat settled, first
    frame written), while custody still holds its live entry; ``release`` lets it finish its publication."""
    from ouroboros.review_continuation import PREFIX
    from ouroboros.tools import plan_ask

    held, release, publish = threading.Event(), threading.Event(), plan_ask.settle_released_reply

    def settle(usage_ctx, *, actor, **kw):
        publish(usage_ctx, actor=actor, **kw)
        if not str(getattr(actor, "operation_id", "") or "").startswith(PREFIX):
            held.set()
            release.wait(120)

    monkeypatch.setattr(plan_ask, "settle_released_reply", settle)
    return held, release


def _live_operations():
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK

    with _ACTIVE_LOCK:
        return {entry.operation_id for entry in _ACTIVE.values()}


def test_a_first_interrupted_ask_continues_as_a_new_turn_of_its_own_thread_in_the_same_ask(plan_session):
    h = plan_session
    ctx = h.make_ctx()
    _write(ctx, ask=["s2"])
    stopped, offers = _offered(h, ctx)
    ask, _ = _participant(h)
    assert len(h.engine.turns) == 1 and h.engine.turns[0][0] == "thread-1"
    assert stopped["failure_details"]["resumable"]["cause"] == "vendor_limit"
    assert not stopped.get("review_thread_id")  # a first ask that stopped records no answered thread
    assert [(o["seat_id"], o["operation_id"], o["ask_id"]) for o in offers] == [("s2", stopped["operation_id"], ask["ask_id"])]

    text, result = _call(ctx, continue_seats=[{"seat_id": "s2", "operation_id": stopped["operation_id"]}])
    assert result.status == "ok", text
    answered = _settled(h, "answered")
    (thread0, first, _key0), (thread1, successor, key1) = h.engine.turns
    assert thread1 == thread0 == "thread-1" and "continueFrom" not in successor  # a new turn of that very thread
    changed = {name for name in set(first) | set(successor) if first.get(name) != successor.get(name)}
    assert changed <= {"prompt", "maxSeconds"}, changed
    assert "stopped before it finished" in successor["prompt"]
    record = plan_record.load_plan_record(h.drive, "task-1")
    assert len(record["asks"]) == 1 and record["asks"][0]["ask_id"] == ask["ask_id"]  # no new ask
    assert answered["operation_id"] != stopped["operation_id"] and answered["superseded_operation_ids"] == [stopped["operation_id"]]
    assert answered["prior_operations"][0]["failure_code"] == "vendor_limit"
    seat = _eventually(lambda: load_record(h.drive, ask["record_id"])["rows"][0],
                       lambda row: row["status"] == "responded", "the ledger seat settling")
    assert seat["continuations"][0]["invocation_id"] == key1
    assert seat["continuations"][0]["thread_id"] == "thread-1" and seat["continuations"][0]["continues_run"] == "run-1"
    from ouroboros.owner_mailbox import drain_owner_entries

    own = f"plan-complete:task-1:{ask['ask_id']}:{answered['operation_id']}"  # its own frame, never the first
    frames = _eventually(lambda: [str(entry.get("msg_id") or "") for entry in drain_owner_entries(
        h.drive, "task-1", set(), 1, include_acknowledged=True) if str(entry.get("msg_id") or "").startswith("plan-complete:")],
        lambda found: own in found, "the continued completion frame")
    assert frames == [f"plan-complete:task-1:{ask['ask_id']}", own]

    _call(ctx, continue_seats=[{"seat_id": "s2", "operation_id": stopped["operation_id"]}])  # rejoin: nothing sent
    assert len(h.engine.turns) == 2


def test_a_participant_failed_before_its_ledger_seat_settled_is_not_offered_and_a_choice_decides_nothing(plan_session, monkeypatch):
    """Between the participant's write and its ledger seat's, the read offers nothing and naming the operation is
    refused at $0: an offer always means the seat's stop is recorded. Once it is, the same choice continues."""
    from ouroboros.tools import plan_ask

    h = plan_session
    ctx = h.make_ctx()
    reached, release, settle = threading.Event(), threading.Event(), plan_ask.settle_ledger_seat

    def held(*args, **kw):
        reached.set()
        release.wait(120)
        return settle(*args, **kw)

    monkeypatch.setattr(plan_ask, "settle_ledger_seat", held)
    try:
        _write(ctx, ask=["s2"])
        assert reached.wait(60), "the stopped participant never reached its ledger seat"
        stopped = _participant(h)[1]
        chosen = [{"seat_id": "s2", "operation_id": stopped["operation_id"]}]
        assert stopped["status"] == "failed" and OFFER not in _call(ctx)[0]
        text, result = _call(ctx, continue_seats=chosen)
        assert result.code == "TOOL_ARG_ERROR" and "PLAN_CONTINUATION_REFUSED" in text and len(h.engine.turns) == 1, text
    finally:
        release.set()
    _offered(h, ctx)
    text, result = _call(ctx, continue_seats=chosen)
    assert result.status == "ok", text
    assert _settled(h, "answered")["superseded_operation_ids"] == [stopped["operation_id"]] and len(h.engine.turns) == 2


def test_a_choice_made_while_the_stopped_worker_still_publishes_waits_for_it_then_sends_one_turn(plan_session, monkeypatch):
    """The stopped seat's reply has landed while its worker still holds custody's live entry, under the attempt key its
    successor shares. The choice waits for that publication to end, then decides and sends one turn; custody never
    meets it as a second operation to refuse as lost (``Exact review custody is unavailable``)."""
    from ouroboros import review_continuation

    h = plan_session
    ctx = h.make_ctx()
    held, release = _hold_the_stopped_worker(monkeypatch)
    waiting, wait = threading.Event(), review_continuation._await_publication
    monkeypatch.setattr(review_continuation, "_await_publication", lambda *args: (waiting.set(), wait(*args))[1])
    outcome = {}
    choice = threading.Thread(target=lambda: outcome.update(zip(("text", "result"), _call(ctx, continue_seats=[
        {"seat_id": "s2", "operation_id": stopped["operation_id"]}]))), daemon=True)
    try:
        _write(ctx, ask=["s2"])
        assert held.wait(60), "the stopped worker never published its reply"
        ask, stopped = _participant(h)
        assert stopped["status"] == "failed" and stopped["operation_id"] in _live_operations()
        choice.start()
        _eventually(lambda: waiting.is_set() or not choice.is_alive(), bool, "the choice meeting the publishing worker")
        now = _participant(h)[1]  # nothing decided while that worker publishes
        assert now == stopped, {key: now.get(key) for key in ("operation_id", "status", "operation_state", "error")}
        assert waiting.is_set() and choice.is_alive() and len(h.engine.turns) == 1  # it waits; nothing was sent
    finally:
        release.set()
        if choice.ident is not None:
            choice.join(60)
    assert outcome["result"].status == "ok", outcome["text"]
    answered = _settled(h, "answered")
    assert answered["superseded_operation_ids"] == [stopped["operation_id"]] and not answered.get("unresolved")
    assert [turn[0] for turn in h.engine.turns] == ["thread-1", "thread-1"]  # one successor turn of that thread
    seat = _eventually(lambda: load_record(h.drive, ask["record_id"])["rows"][0],
                       lambda row: row["status"] == "responded", "the ledger seat settling")
    assert seat["operation_id"] == answered["operation_id"] and seat["continuations"][0]["invocation_id"] == h.engine.turns[1][2]


def test_the_owners_pause_ends_a_choice_waiting_on_a_publishing_worker_at_zero_and_after_resume_it_continues(
        plan_session, monkeypatch):
    """The wait has no bound of its own: Pause (like Stop or the end of the original window) ends it, nothing
    decided or sent, and the publishing worker keeps its entry; after Resume the same choice sends one turn."""
    from ouroboros import owner_pause, review_continuation, review_pause

    h = plan_session
    ctx = h.make_ctx()
    monkeypatch.setattr(review_pause, "DETACH_POLL_SEC", 0.05)  # the drain slice the wait rereads its fences in
    held, release = _hold_the_stopped_worker(monkeypatch)
    waiting, wait = threading.Event(), review_continuation._await_publication
    monkeypatch.setattr(review_continuation, "_await_publication", lambda *args: (waiting.set(), wait(*args))[1])
    outcome = {}
    choice = threading.Thread(target=lambda: outcome.update(zip(("text", "result"), _call(ctx, continue_seats=[
        {"seat_id": "s2", "operation_id": stopped["operation_id"]}]))), daemon=True)
    try:
        _write(ctx, ask=["s2"])
        assert held.wait(60), "the stopped worker never published its reply"
        ask, stopped = _participant(h)
        choice.start()
        _eventually(waiting.is_set, bool, "the choice waiting on the publishing worker")
        owner_pause.install_fence(h.drive, "task-1", request_id="pause-while-it-publishes")
        choice.join(30)
        assert not choice.is_alive() and outcome["result"].code == "TOOL_ARG_ERROR", outcome.get("text")
        assert "stopped_or_paused" in outcome["text"], outcome["text"]
        assert _participant(h)[1] == stopped and len(h.engine.turns) == 1  # nothing decided, nothing sent
        assert stopped["operation_id"] in _live_operations()  # the publishing worker keeps its own entry
    finally:
        release.set()
        owner_pause.release_fence(h.drive, "task-1", reason="owner_resume")
    _eventually(lambda: stopped["operation_id"] not in _live_operations(), bool, "the stopped worker finishing")
    text, result = _call(ctx, continue_seats=[{"seat_id": "s2", "operation_id": stopped["operation_id"]}])
    assert result.status == "ok", text
    assert _settled(h, "answered")["superseded_operation_ids"] == [stopped["operation_id"]] and len(h.engine.turns) == 2


def test_a_plan_choice_whose_reopen_failed_after_its_start_is_not_first_sent_after_its_window(plan_session, monkeypatch):
    """START recorded, participant never reopened, so nothing went out. Past the original window the repeat sends no
    turn, settles the seat unsent and says so; the reply it would promise can never arrive."""
    from ouroboros import deadline_utils, delegate_custody
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.review_continuation import WINDOW_SPENT

    monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "600")
    h = plan_session
    ctx = h.make_ctx()
    _write(ctx, ask=["s2"])
    stopped, _offers = _offered(h, ctx)
    chosen = [{"seat_id": "s2", "operation_id": stopped["operation_id"]}]
    reopen = plan_record.reopen_participant
    monkeypatch.setattr(plan_record, "reopen_participant", lambda *_a, **_k: (_ for _ in ()).throw(
        OSError("process ended before the reopen")))
    text, result = _call(ctx, continue_seats=chosen)
    monkeypatch.setattr(plan_record, "reopen_participant", reopen)
    ask, participant = _participant(h)
    decision = load_record(h.drive, ask["record_id"])["rows"][0]["continuations"][0]
    assert result.code == "TOOL_ERROR" and "PLAN_CONTINUATION_UNSENT" in text, text
    assert delegate_custody.invocation_record(h.drive, decision["invocation_id"]) and len(h.engine.turns) == 1
    assert participant["operation_id"] == stopped["operation_id"] and decision["window_end"]

    later = parse_deadline_ts(decision["window_end"]) + timedelta(seconds=5)
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: later)
    text, result = _call(ctx, continue_seats=chosen)
    seat = load_record(h.drive, ask["record_id"])["rows"][0]
    assert len(h.engine.turns) == 1, h.engine.turns[1:]
    assert (seat["status"], seat["failure_code"]) == ("not_dispatched", WINDOW_SPENT), seat
    assert result.status == "ok" and "Continued in ask" not in text and "original review window" in text, text
    now = _participant(h)[1]  # the participant names what the ledger seat names; its stopped facts are history
    assert (now["operation_id"], now["status"], now["failure_code"]) == (decision["operation_id"], "not_dispatched", WINDOW_SPENT)
    assert now["prior_operations"][-1]["operation_id"] == stopped["operation_id"] and not now.get("pending_invocation_id")
    assert delegate_custody.invocation_record(h.drive, decision["invocation_id"])["state"] == "failed_definite"


def test_a_plan_choice_at_a_reached_budget_fence_sends_no_turn_and_with_room_it_continues(plan_session, monkeypatch):
    """The ordinary ask rides ``review_wave_budget_gate`` before dispatch; a continued participant's turn is new
    generation too. Known spend at the task's fence (no Pause anywhere): refused at $0, no turn, nothing decided."""
    from ouroboros import usage_accounting as ua

    h = plan_session
    ctx = h.make_ctx()
    _write(ctx, ask=["s2"])
    stopped, _offers = _offered(h, ctx)
    chosen = [{"seat_id": "s2", "operation_id": stopped["operation_id"]}]
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 0.5)  # the price catalog only

    def scope(fence):
        return ua.UsageScope(drive_root=h.drive, task_id="task-1", root_task_id="task-1", root_limit_usd=fence)
    with ua.usage_scope(scope(2.0)):
        row = ua.reserve_attempt(ua.AttemptRequest(model="fake-small", provider="test", source="main"))
        ua.mark_dispatched(row)
        ua.settle_attempt(row, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=2.0, cost_final=True)
    ask, _ = _participant(h)
    before = load_record(h.drive, ask["record_id"])
    with ua.usage_scope(scope(2.0)):
        text, result = _call(ctx, continue_seats=chosen)
    assert result.code == "TOOL_ARG_ERROR" and "budget_insufficient" in text and len(h.engine.turns) == 1, text
    assert load_record(h.drive, ask["record_id"]) == before and _participant(h)[1]["operation_id"] == stopped["operation_id"]
    with ua.usage_scope(scope(20.0)):
        text, result = _call(ctx, continue_seats=chosen)
    assert result.status == "ok", text
    assert _settled(h, "answered")["superseded_operation_ids"] == [stopped["operation_id"]] and len(h.engine.turns) == 2


def test_a_plan_turn_whose_launch_crosses_its_window_end_is_never_sent(plan_session, monkeypatch):
    """The send callback is shared by runs and thread turns: a launch delay after the window check that carries the
    successor's first turn past the window its START recorded refuses before ``start_thread_turn``. No turn is
    sent; the participant's outcome stays unknown."""
    from ouroboros import deadline_utils, delegate_custody
    from ouroboros.deadline_utils import parse_deadline_ts
    from tests.test_review_continuation_unsent import _launch_delayed_past, _shifted_clock

    monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "600")
    h = plan_session
    ctx = h.make_ctx()
    _write(ctx, ask=["s2"])
    stopped, _offers = _offered(h, ctx)
    started, cap = delegate_custody.run_timing(h.drive, "run-1")  # the stopped turn's own window
    end = parse_deadline_ts(started) + timedelta(seconds=cap)
    _shifted_clock(monkeypatch, (end - deadline_utils.utc_now()).total_seconds() - 3)  # three seconds of it left
    launched = _launch_delayed_past(monkeypatch, end)
    text, result = _call(ctx, continue_seats=[{"seat_id": "s2", "operation_id": stopped["operation_id"]}])
    assert result.status == "ok", text
    now = _eventually(lambda: _participant(h)[1], lambda p: p.get("operation_state") == "custody_lost",
                      "the successor's outcome settling as unknown")
    assert launched and launched[0] < end and len(h.engine.turns) == 1, (launched, h.engine.turns[1:])
    assert now.get("unresolved") and now["operation_id"] != stopped["operation_id"]


@pytest.mark.parametrize("delay", ["crosses_before_the_start", "crosses_after_the_start"])
def test_a_plan_preparation_delay_before_its_start_never_moves_the_original_window_end(plan_session, monkeypatch, delay):
    """The stopped turn's window has 10 s left when the projection reads it and its steps before the START take 15: the
    turn would begin after the window, so none is sent and the seat and participant settle as never sent, offered no
    more. Or 13 s are left, 5 s go before the cap is read and 5 more before the START is stamped, and the launch then
    crosses the end the START carries: no turn is sent, its outcome stays unknown."""
    from ouroboros import deadline_utils, delegate_custody
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.review_continuation import WINDOW_SPENT
    from tests import test_review_continuation_unsent as unsent

    monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "600")
    h = plan_session
    ctx = h.make_ctx()
    _write(ctx, ask=["s2"])
    stopped, _offers = _offered(h, ctx)
    started, cap = delegate_custody.run_timing(h.drive, "run-1")
    end = parse_deadline_ts(started) + timedelta(seconds=cap)
    left, takes, stamped_after = (10, 15, 0) if delay == "crosses_before_the_start" else (13, 5, 5)
    clock = unsent._shifted_clock(monkeypatch, (end - deadline_utils.utc_now()).total_seconds() - left)
    unsent._preparation_takes(monkeypatch, clock, takes)
    unsent._start_persisted_late(monkeypatch, clock, stamped_after)
    launched = unsent._launch_delayed_past(monkeypatch, end)
    text, result = _call(ctx, continue_seats=[{"seat_id": "s2", "operation_id": stopped["operation_id"]}])
    ask, _ = _participant(h)
    if delay == "crosses_after_the_start":
        assert result.status == "ok", text
        now = _eventually(lambda: _participant(h)[1], lambda p: p.get("operation_state") == "custody_lost",
                          "the successor's outcome settling as unknown")
        token = load_record(h.drive, ask["record_id"])["rows"][0]["continuations"][-1]["invocation_id"]
        assert launched and launched[0] < end and len(h.engine.turns) == 1, (launched, h.engine.turns[1:])
        assert delegate_custody.invocation_record(h.drive, token)["state"] == "pending" and now.get("unresolved")
        return
    seat = load_record(h.drive, ask["record_id"])["rows"][0]
    assert len(h.engine.turns) == 1 and not launched, h.engine.turns[1:]
    assert (seat["status"], seat["failure_code"]) == ("not_dispatched", WINDOW_SPENT), seat
    assert delegate_custody.invocation_record(h.drive, seat["continuations"][-1]["invocation_id"]) is None
    assert result.status == "ok" and "original review window" in text, text
    now = _participant(h)[1]
    assert (now["status"], now["failure_code"]) == ("not_dispatched", WINDOW_SPENT) and not now.get("pending_invocation_id")


def test_a_plan_turn_whose_known_spend_reaches_its_fence_during_the_launch_is_never_sent(plan_session, monkeypatch):
    """The send callback is shared by runs and thread turns: known spend another call settles at the task's fence while
    the successor turn's launch steps run stops its POST. No turn is sent; its outcome stays unknown."""
    from ouroboros import delegate_custody, review_execution
    from ouroboros import usage_accounting as ua

    h = plan_session
    ctx = h.make_ctx()
    _write(ctx, ask=["s2"])
    stopped, _offers = _offered(h, ctx)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 0.5)  # the price catalog only

    def scope(fence):
        return ua.UsageScope(drive_root=h.drive, task_id="task-1", root_task_id="task-1", root_limit_usd=fence)
    launch, launched = review_execution.run_operation, []

    def other_work():  # the task's other work, on its own thread, reaches the fence
        with ua.usage_scope(scope(2.0)):
            row = ua.reserve_attempt(ua.AttemptRequest(model="fake-small", provider="test", source="main"))
            ua.mark_dispatched(row)
            ua.settle_attempt(row, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=2.0, cost_final=True)

    def settled_meanwhile(source, function, *args, **kwargs):
        if not launched:
            other = threading.Thread(target=other_work)
            other.start()
            other.join(30)
        launched.append(True)  # the substrate's one retry of a refused send meets the same fence
        return launch(source, function, *args, **kwargs)
    monkeypatch.setattr(review_execution, "run_operation", settled_meanwhile)
    with ua.usage_scope(scope(2.0)):  # room when the choice is made and its START written
        text, result = _call(ctx, continue_seats=[{"seat_id": "s2", "operation_id": stopped["operation_id"]}])
    assert result.status == "ok", text
    now = _eventually(lambda: _participant(h)[1], lambda p: p.get("operation_state") == "custody_lost",
                      "the successor's outcome settling as unknown")
    assert launched and len(h.engine.turns) == 1, h.engine.turns[1:]
    ask, _ = _participant(h)
    token = load_record(h.drive, ask["record_id"])["rows"][0]["continuations"][-1]["invocation_id"]
    assert delegate_custody.invocation_record(h.drive, token)["state"] == "pending"  # key and exact bytes kept
    assert now.get("unresolved") and now["operation_id"] != stopped["operation_id"]


def test_a_stale_reply_of_the_stopped_operation_never_overwrites_the_continued_participant(plan_session, monkeypatch):
    from ouroboros.tools import plan_ask

    h = plan_session
    ctx = h.make_ctx()
    _write(ctx, ask=["s2"])
    stopped, _offers = _offered(h, ctx)
    _call(ctx, continue_seats=[{"seat_id": "s2", "operation_id": stopped["operation_id"]}])
    answered = _settled(h, "answered")
    ask, _ = _participant(h)
    _eventually(lambda: load_record(h.drive, ask["record_id"])["rows"][0], lambda row: row["status"] == "responded",
                "the ledger seat settling")  # the stale writes below meet a settled successor
    _eventually(lambda: answered["operation_id"] not in _live_operations(), bool,
                "the successor's worker finishing")  # its own "answered" line is not what is listened for below
    said = []
    monkeypatch.setattr(plan_ask, "_progress", lambda _ctx, text: said.append(text))
    for identity in ({"operation_id": stopped["operation_id"]}, {"operation_id": ""}, {}):  # stale, empty, missing
        stale = {"slot_id": "s2", "operation_state": "settled", "text": "", "error": "a late callback of the stopped run",
                 "model": "fake-review=fake-small", **identity}
        plan_ask.record_reply(ctx, state_root=h.drive, task_id="task-1", ask_id=ask["ask_id"], seat_id="s2", row=stale)
        plan_record.update_participant(h.drive, "task-1", ask_id=ask["ask_id"], seat_id="s2",
                                       facts={**identity, "status": "failed", "error": "late", "reply": None})
    _ask, after = _participant(h)
    assert (after["status"], after["operation_id"], after.get("reply")) == ("answered", answered["operation_id"], answered.get("reply"))
    assert not said, said  # nothing about the stopped operation reaches its author as news
    seat = load_record(h.drive, ask["record_id"])["rows"][0]
    assert seat["status"] == "responded" and seat["operation_id"] == answered["operation_id"]
