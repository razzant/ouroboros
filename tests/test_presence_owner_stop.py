"""#2012: the owner's public Stop reaches a live Presence turn, whose runner settles it.

The owner presses Stop through the real public route (``gateway.tasks.api_task_cancel``,
single and the app's cascade shape), which records the durable intent and drives the real
custody; the cancel-intent watchdog settles it later. The turn is a real Place focus on a
Host thread (``PresenceTurnExecutions.start_thread`` -> ``run_presence_turn``) with the
production agent bootstrap, loop, terminal pipeline and runner; only inference and review
transport are scripted (``install_bootstrap_harness``). Every scripted model call first
passes the production pre-dispatch window (``llm_attempt.require_physical_dispatch_window``),
the gate each physical send crosses, so a call refused there generated nothing. A call
already sent can still return the final answer; the answer's closing admission reads the
same open intent, so that answer is retained, never sent, and a Stop after it lets it stand.
A forced final call (the round limit's, a Wrap up's), an authored stop that a hard rail
publishes without a call and every other no-call exit (a real budget refusal's retained
answer) are admitted the same way.

Wrap up (``stop_policy=finalize_then_cancel``) reaches the same turn: the existing owner-stop
episode arms its ``finalize_now`` control, its loop drains it, and its one tool-less final turn
answers through the ordinary reply gate; Stop now hardens that same request.

Rows nothing witnesses keep their old answer; ``tests/test_cancel_intent_authority.py``
keeps that contract on synthetic rows. The parked author in another process is
``tests/test_presence_owner_stop_process.py``.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import cancel_intents as ci
from ouroboros.presence_runner import PresenceTurnError
from ouroboros.task_results import load_task_result
from tests.test_presence_continuation import review
from tests.test_presence_continuation_bootstrap import finish, install_bootstrap_harness, tool_call
from tests.test_presence_contact_wait import start
from tests.test_presence_owner_contact import contact_runtime as _contact_runtime, focus  # noqa: F401

contact_runtime = _contact_runtime
pytestmark = pytest.mark.serial

ANSWER = "The status report is ready."
SUMMARY = "Where it stands: the report is drafted; its numbers are not checked yet."
LATE_ANSWER = "The complete answer, from the call sent before Stop now."
FORCED_ANSWER = "The round limit's best answer: the report is drafted, its numbers are unchecked."
AUTHORED_STOP = "Stopping here: the report is drafted; checking its numbers remains."
HELD_ANSWER = "The complete report is ready, with the backup window still unchecked."


@pytest.fixture
def owner_stop(contact_runtime, monkeypatch):
    from ouroboros.gateway.tasks import api_task_cancel
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"])])
    app.state.drive_root = contact_runtime.root
    client = TestClient(app)

    def press(task_id, *, cascade, policy="immediate", action="owner-stop-1"):
        # The app's Stop now and Wrap up are the cascade shape (web/modules/task_control_menu.js); an empty body is single.
        body = {"cascade": True, "stop_policy": policy, "stop_action_id": action} if cascade else None
        return client.post(f"/api/tasks/{task_id}/cancel", **({"json": body} if body else {}))

    return contact_runtime, press


def _scripted(monkeypatch, r, during_first_call, after_first):
    """Round one hands the Stop to ``during_first_call``; nothing past the dispatch gate is counted unless sent."""
    from ouroboros.llm_attempt import require_physical_dispatch_window

    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")
    generated = []

    def model(_messages):
        require_physical_dispatch_window()  # the production gate before any physical send
        generated.append(len(generated) + 1)
        if len(generated) == 1:
            during_first_call(h.agents[0].tools._ctx.task_id)
            return after_first
        return finish("after-stop", message="A reply composed after the owner's Stop.")

    h.script = model
    return h, generated


def _outbound(root, task_id):
    from ouroboros import dialogue_store

    return [stored.row for stored in dialogue_store.task_rows(root, [task_id], stream="chat")
            if stored.row.get("direction") == "out" or stored.row.get("type") == "presence_delivery"]


def _incidents(root, task_id):
    path = root / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("task_id") == task_id and str(row.get("type", "")).startswith("task_cancel_cascade")]


def _round_limited(monkeypatch, r, during_final_call):
    """The production round limit (``OUROBOROS_MAX_ROUNDS=1``): round one asks for a tool, so the next
    round is the forced ``[ROUND_LIMIT]`` call, which runs ``during_final_call`` once it is already out."""
    from ouroboros.llm_attempt import require_physical_dispatch_window

    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")
    generated, inputs = [], []

    def model(messages):
        inputs.append(messages)
        require_physical_dispatch_window()  # the production gate before any physical send
        generated.append(len(generated) + 1)
        if len(generated) == 1:
            return {"content": None, "tool_calls": [tool_call("knowledge_list", {}, "look")]}
        during_final_call(h.agents[0].tools._ctx.task_id)
        return {"content": FORCED_ANSWER, "tool_calls": []}

    h.script = model
    return h, generated, inputs


def _kept_unsent(r, h, stopped, stop, answer):
    """Stop now met ``answer`` at its admission: kept on the cancelled row and never sent; the intent
    stays open for the watchdog, which settles it from that terminal with the app's one receipt.
    ``stopped`` is what the ended turn raised (``None`` if it returned a published result)."""
    from supervisor.owner_stop import owner_stop_ending
    from supervisor.task_lifecycle import sweep_cancel_intents

    tid = h.agents[0].tools._ctx.task_id
    row = load_task_result(r.root, tid)
    # The stored ending and what went out first: an answer that escaped the Stop names its own rail.
    assert (row["status"], row["reason_code"], [out.get("text") for out in _outbound(r.root, tid)]) == (
        "cancelled", "owner_requested_finalization", []), "no reply, send or delivery receipt after the owner's Stop"
    assert stop.status_code == 202, stop.text
    assert stop.json()["cancel_state"] == "pending" and stop.json()["stop_policy"] == "immediate"
    assert isinstance(stopped, PresenceTurnError), stopped
    assert stopped.code == "presence_turn_cancelled" and stopped.disposition == "retained"
    assert row["result"] == answer and owner_stop_ending(row) == ("stop_now", "stop")
    assert row["cancel_origin"]["stop_policy"] == "immediate"
    assert row["cancel_origin"]["request_id"] == stop.json()["request_id"]
    assert row["outcome_axes"]["lifecycle"]["status"] == "cancelled"
    assert not _incidents(r.root, tid)
    assert ci.active_intent(r.root, tid)
    _bus(r)  # what the turn emitted before its end; each caller states its own
    assert sweep_cancel_intents(now=time.time() + 3600) == {tid: "cancelled"}
    assert ci.active_intents(r.root) == {} and load_task_result(r.root, tid)["status"] == "cancelled"
    assert not _outbound(r.root, tid)
    assert [(e["system_type"], e["chat_id"]) for e in _bus(r)] == [("cancel_receipt", row["chat_id"])]
    assert not h.executions.live()


@pytest.mark.parametrize("round_one", ["tool_call", "final_answer"])
@pytest.mark.parametrize("cascade", [False, True], ids=["single", "cascade"])
def test_public_stop_reaches_the_live_turn_and_its_runner_settles_cancelled(owner_stop, monkeypatch, cascade, round_one):
    from supervisor.task_lifecycle import sweep_cancel_intents

    r, press = owner_stop
    stops = {}

    def stop_now(task_id):
        stops["first"] = press(task_id, cascade=cascade)

    # The owner presses Stop while round one's call is out: it asks for a tool (refused at the next
    # dispatch) or returns the final answer (read at that answer's closing admission).
    reply = ({"content": LATE_ANSWER, "tool_calls": []} if round_one == "final_answer"
             else {"content": None, "tool_calls": [tool_call("knowledge_list", {}, "look")]})
    h, generated = _scripted(monkeypatch, r, stop_now, reply)
    execution = start(r, h)
    with pytest.raises(PresenceTurnError) as stopped:
        execution.result.result(timeout=90)
    tid = h.agents[0].tools._ctx.task_id
    first = stops["first"]
    # Custody found a witnessed live turn: the Stop is pending, never "not active" or a fabricated row.
    assert first.status_code == 202, first.text
    assert first.json()["cancel_state"] == "pending" and first.json()["stop_policy"] == "immediate"
    # The runner observed the open intent at its next stop check: nothing past the gate was generated.
    assert generated == [1]
    assert stopped.value.code == "presence_turn_cancelled" and stopped.value.disposition == "retained"
    row = load_task_result(r.root, tid)
    assert row["status"] == "cancelled" and row["reason_code"] == "owner_requested_finalization"
    if round_one == "final_answer":
        assert row["result"] == LATE_ANSWER  # the retained answer stays on the row, unsent
    origin = row["cancel_origin"]
    assert origin["source"] == ("http_cascade" if cascade else "http_single")
    assert origin["stop_policy"] == "immediate" and origin["request_id"] == first.json()["request_id"]
    assert row["outcome_axes"]["lifecycle"]["status"] == "cancelled"
    assert not _outbound(r.root, tid), "no reply, send or delivery receipt after the owner's Stop"
    assert not _incidents(r.root, tid), "a cascade waiting on its live Presence root is pending, not a no-op"
    # The intent stayed open for the runner; the watchdog settles it from the runner's own terminal.
    assert ci.active_intent(r.root, tid)
    while not r.bus.empty():
        r.bus.get_nowait()
    swept = sweep_cancel_intents(now=time.time() + 3600)
    assert swept == {tid: "cancelled"}
    assert ci.active_intents(r.root) == {}
    assert load_task_result(r.root, tid)["status"] == "cancelled"
    assert not _outbound(r.root, tid)
    # Custody wrote nothing of its own; the app's tree Stop keeps its one receipt, a System row in
    # the turn's own room (the local chat surface, never a connector send). A single Stop has none.
    receipts = []
    while not r.bus.empty():
        receipts.append(r.bus.get_nowait())
    assert [(e["type"], e.get("system_type"), e["chat_id"], e.get("role")) for e in receipts] == (
        [("send_message", "cancel_receipt", row["chat_id"], "system")] if cascade else [])
    # Once settled, another Stop finds nothing to stop.
    assert press(tid, cascade=cascade).status_code == 404
    assert not h.executions.live()


@pytest.mark.parametrize("cascade", [False, True], ids=["single", "cascade"])
def test_a_terminal_written_before_the_stop_wins_while_its_runner_still_lives(owner_stop, monkeypatch, cascade):
    """Completion wins: the Stop lands after the turn's terminal, while its runner is still tearing down."""
    import supervisor.queue_transitions as transitions
    from ouroboros.presence_runner import presence_turn_is_live

    r, press = owner_stop
    stops = {}
    reconcile = transitions.reconcile_terminal_task_projections

    def stop_during_teardown(root, task_id):
        if task_id not in stops:  # the runner's own end: terminal written, still registered live
            assert load_task_result(root, task_id)["status"] == "completed"
            assert presence_turn_is_live(task_id, drive_root=root)
            stops[task_id] = press(task_id, cascade=cascade)
        return reconcile(root, task_id)

    monkeypatch.setattr(transitions, "reconcile_terminal_task_projections", stop_during_teardown)
    h, generated = _scripted(monkeypatch, r, lambda _tid: None, finish("final", message=ANSWER))
    execution = start(r, h)
    result = execution.result.result(timeout=90)
    tid = result.task_id
    assert result.text == ANSWER and generated == [1]
    # Nothing to stop: no intent is minted over the settled turn and the answer stands.
    assert stops[tid].status_code == 404, stops[tid].text
    assert ci.active_intents(r.root) == {}
    row = load_task_result(r.root, tid)
    assert row["status"] == "completed" and "cancel_origin" not in row


def test_a_stop_after_the_answer_closed_admission_lets_the_answer_stand(owner_stop, monkeypatch):
    """Completion wins past the last stop check: Stop now lands once the answer's admission has
    closed (at the reply check), before its terminal; the answer goes out and the intent settles."""
    from ouroboros import place_reply
    from supervisor.task_lifecycle import sweep_cancel_intents

    r, press = owner_stop
    stops = []
    prepare = place_reply.prepare_focus_reply

    def stop_at_the_reply_check(tool_ctx, *args, **kwargs):
        if not stops:  # the loop's check; the result event (build_presence_result_event) asks again
            stops.append(press(tool_ctx.task_id, cascade=False))
        return prepare(tool_ctx, *args, **kwargs)

    monkeypatch.setattr(place_reply, "prepare_focus_reply", stop_at_the_reply_check)
    h, generated = _scripted(monkeypatch, r, lambda _tid: None, {"content": ANSWER, "tool_calls": []})
    result = start(r, h).result.result(timeout=90)
    tid = result.task_id
    assert result.text == ANSWER and generated == [1]
    assert [stop.status_code for stop in stops] == [202], [stop.text for stop in stops]
    row = load_task_result(r.root, tid)
    assert row["status"] == "completed" and "cancel_origin" not in row
    assert [(out["direction"], out["text"]) for out in _outbound(r.root, tid)] == [("out", ANSWER)]
    assert sweep_cancel_intents(now=time.time() + 3600) == {tid: "already_settled"}
    assert ci.active_intents(r.root) == {} and load_task_result(r.root, tid)["status"] == "completed"


def test_public_stop_during_the_round_limit_final_call_keeps_its_answer_unsent(owner_stop, monkeypatch):
    """A forced answer is admitted as the ordinary one is: Stop now pressed while the round limit's one
    tool-less call is already out keeps that call's plain answer on the row, cancelled and unsent."""
    r, press = owner_stop
    stops = []
    h, generated, inputs = _round_limited(monkeypatch, r, lambda tid: stops.append(press(tid, cascade=True)))
    stopped = start(r, h).result.exception(timeout=90)
    assert "[ROUND_LIMIT]" in json.dumps(inputs[1], ensure_ascii=False)
    # Both calls crossed the gate before the Stop; no refresh, repair or later call was even attempted.
    assert generated == [1, 2] and h.calls == 2
    [stop] = stops
    _kept_unsent(r, h, stopped, stop, FORCED_ANSWER)


def test_a_stop_after_the_forced_answer_closed_admission_lets_it_stand(owner_stop, monkeypatch):
    """Completion wins past the forced answer's stop check too: Stop now lands at its reply check, after
    that admission read the intent; the round limit's answer goes out and the intent settles."""
    from ouroboros import place_reply
    from supervisor.task_lifecycle import sweep_cancel_intents

    r, press = owner_stop
    stops = []
    prepare = place_reply.prepare_focus_reply

    def stop_at_the_reply_check(tool_ctx, *args, **kwargs):
        if not stops:  # a forced exit's one reply check is the result event's (build_presence_result_event)
            stops.append(press(tool_ctx.task_id, cascade=False))
        return prepare(tool_ctx, *args, **kwargs)

    monkeypatch.setattr(place_reply, "prepare_focus_reply", stop_at_the_reply_check)
    h, generated, _inputs = _round_limited(monkeypatch, r, lambda _tid: None)
    result = start(r, h).result.result(timeout=90)
    tid = result.task_id
    assert result.text == FORCED_ANSWER and generated == [1, 2]
    assert [stop.status_code for stop in stops] == [202], [stop.text for stop in stops]
    row = load_task_result(r.root, tid)
    assert row["status"] == "completed" and row["reason_code"] == "round_limit" and "cancel_origin" not in row
    assert [(out["direction"], out["text"]) for out in _outbound(r.root, tid)] == [("out", FORCED_ANSWER)]
    assert sweep_cancel_intents(now=time.time() + 3600) == {tid: "already_settled"}
    assert ci.active_intents(r.root) == {} and load_task_result(r.root, tid)["status"] == "completed"


def test_public_stop_during_an_authored_stops_batch_at_a_hard_rail_keeps_it_unsent(owner_stop, monkeypatch):
    """The selected finish's neighbour: the authored stop is staged, then Stop now lands while a tool of
    the same batch runs. A hard rail (exhausted money, the existing seam
    ``authored_completion_budget_exhausted``) publishes that stop without a model call, so it reads the
    open intent first, as it already reads a direct turn's Stop control."""
    from ouroboros import loop_budget
    from ouroboros.tools import knowledge

    r, press = owner_stop
    monkeypatch.setattr(loop_budget, "authored_completion_budget_exhausted", lambda *_args: True)
    stops = []
    listing = knowledge._knowledge_list

    def listing_while_stopped(ctx, *args, **kwargs):
        stops.append(press(ctx.task_id, cascade=True))
        return listing(ctx, *args, **kwargs)

    monkeypatch.setattr(knowledge, "_knowledge_list", listing_while_stopped)
    authored = finish("stop", action="stop", rationale="Checking its numbers remains.", message=AUTHORED_STOP)
    batch = {"content": None, "tool_calls": [*authored["tool_calls"], tool_call("knowledge_list", {}, "look")]}
    h, generated = _scripted(monkeypatch, r, lambda _tid: None, batch)
    stopped = start(r, h).result.exception(timeout=90)
    assert generated == [1] and h.calls == 1
    [stop] = stops
    _kept_unsent(r, h, stopped, stop, AUTHORED_STOP)


def _budget_refused(monkeypatch, r, press, stop_at):
    """A held answer meets a real budget refusal: its result review FAILs, so the turn keeps the answer
    and asks once more, and that call's reservation (``usage_accounting.reserve_attempt`` on a zero global
    limit) is refused before a byte is sent; the budget rail ends the turn on that answer without a call.
    Stop now is pressed while the real transaction decides the refusal (``stop_at="refusal"``), at the
    budget answer's reply check (``"reply"``) or never (``""``)."""
    from ouroboros import place_reply
    from ouroboros.usage_accounting import AttemptRequest, BudgetExceeded, reserve_attempt
    from ouroboros.usage_store import Txn

    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="required")
    h.release.set()
    h.addressed, h.verdict = "", "FAIL"
    seen = SimpleNamespace(stops=[], refusals=[], admission_open=[])
    exceeds_limit, prepare = Txn.exceeds_limit, place_reply.prepare_focus_reply

    def deciding_the_refusal(txn, limit, **kwargs):
        if limit == 0 and stop_at == "refusal" and not seen.stops:  # the refused reservation's own predicate
            seen.admission_open.append(h.agents[0]._accepting_owner_messages)
            seen.stops.append(press(h.agents[0].tools._ctx.task_id, cascade=True))
        return exceeds_limit(txn, limit, **kwargs)

    def at_the_reply_check(tool_ctx, *args, **kwargs):
        if stop_at == "reply" and seen.refusals and not seen.stops:  # the budget answer's, after the refusal
            seen.stops.append(press(tool_ctx.task_id, cascade=False))
        return prepare(tool_ctx, *args, **kwargs)

    monkeypatch.setattr(Txn, "exceeds_limit", deciding_the_refusal)
    monkeypatch.setattr(place_reply, "prepare_focus_reply", at_the_reply_check)

    def model(_messages):
        if h.calls == 1:
            return review(answer=HELD_ANSWER)
        if h.calls == 2:
            return {"content": HELD_ANSWER, "tool_calls": []}
        try:
            reserve_attempt(AttemptRequest(model="budget-refusal", provider="local", reservation_usd=0,
                                           global_limit_usd=0, allow_live_fetch=False))
        except BudgetExceeded as refused:
            seen.refusals.append(refused.limit_scope)
            raise
        raise AssertionError("a zero global limit refuses this reservation")

    h.script = model
    return h, seen


@pytest.mark.parametrize("stop_at", ["", "refusal", "reply"], ids=["no-stop", "stop-at-refusal", "stop-after-admission"])
def test_public_stop_during_a_real_budget_refusal_keeps_the_held_answer_unsent(owner_stop, monkeypatch, stop_at):
    """The budget rail ends the turn on its held answer without a model call, so that answer is admitted
    where every no-call exit is: Stop now accepted while the real reservation is refused keeps it on the
    cancelled row, unsent. Without a Stop, or with one after that admission, the budget's answer stands."""
    from supervisor.task_lifecycle import sweep_cancel_intents

    r, press = owner_stop
    h, seen = _budget_refused(monkeypatch, r, press, stop_at)
    ended = start(r, h).result.exception(timeout=90)
    # The review's call, the held answer's, then one refused before any byte: no further call is bought.
    assert seen.refusals == ["global"] and h.calls == 3
    tid = h.agents[0].tools._ctx.task_id
    if stop_at == "refusal":
        assert seen.admission_open == [True]  # the held answer had not closed its admission
        [stop] = seen.stops
        _kept_unsent(r, h, ended, stop, HELD_ANSWER)
    else:
        row = load_task_result(r.root, tid)
        assert ended is None and "cancel_origin" not in row
        assert (row["status"], row["reason_code"], [out["text"] for out in _outbound(r.root, tid)]) == (
            "completed", "budget_exhausted", [HELD_ANSWER])
    if stop_at == "reply":
        assert [stop.status_code for stop in seen.stops] == [202], [stop.text for stop in seen.stops]
        assert sweep_cancel_intents(now=time.time() + 3600) == {tid: "already_settled"}
        assert ci.active_intents(r.root) == {} and load_task_result(r.root, tid)["status"] == "completed"
    elif not stop_at:
        assert not seen.stops
    # Whichever terminal stands, the refusal's own resource fact stays on the row; the Stop erases no money fact.
    limit = load_task_result(r.root, tid)["outcome_axes"]["execution"]["resource_limit"]
    assert (limit["scope"], limit["status"], limit["auto_resume"]) == ("global", "resource_limited", False)


@pytest.mark.parametrize("rail", ["retry_wall", "context_overflow", "output_exhaustion"])
@pytest.mark.parametrize("stopping", [False, True], ids=["no-stop", "stop-now"])
def test_a_no_call_rail_stamps_its_reason_before_stop_now_can_end_it(tmp_path, monkeypatch, rail, stopping):
    """The no-call rails that stamp their own reason do it before the shared composer, which ends any of
    them on Stop now, so no stamp overwrites that terminal afterwards; without a Stop each keeps its own.
    Rail level (the live turn is the budget case above): the composer, hard stop, durable intent, its
    reader and the retained candidate are production's; the wall's stamps are the real transport's."""
    from ouroboros import loop_output_exhaustion
    from ouroboros.loop_llm_call import RETRY_WALL_EXHAUSTED_KEY, call_llm_with_retry
    from tests.test_delivery_forced_finalization import _forced_test_context
    from tests.test_provider_wall_rail_contract import _TransientFailingLLM

    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    loop._replace_delivery_candidate(registry, limit_ctx, trace, HELD_ANSWER, control="replace")
    monkeypatch.setattr(loop, "_call_forced_model_once", lambda *_a, **_k: pytest.fail("no model call"))
    # An inline Presence turn: no worker slot of its own, so Stop now is its open immediate intent.
    registry._ctx.model_wait_context = SimpleNamespace(
        worker_slot_held=False, task={"_presence_turn": True}, canonical_root=tmp_path, task_id="parent1")
    if stopping:
        ci.request_cancel(tmp_path, "parent1", source="http_single", requested_stop_policy="immediate")
        assert ci.stop_policy(ci.active_intent(tmp_path, "parent1")) == ci.STOP_POLICY_IMMEDIATE
    usage = limit_ctx.accumulated_usage
    if rail == "retry_wall":
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        assert call_llm_with_retry(_TransientFailingLLM(), [{"role": "user", "content": "hi"}], "openai/gpt-5.5",
                                   None, "medium", 2, tmp_path, "parent1", 1, None, usage, "task", False)[0] is None
        assert usage[RETRY_WALL_EXHAUSTED_KEY] is True
        own = "provider_unavailable"
        text, usage, trace = loop._handle_provider_unavailable(limit_ctx)
    elif rail == "context_overflow":
        usage.update(reason_code="context_overflow", _last_llm_error_kind="context_overflow")
        own = "context_overflow"
        text, usage, trace = loop._handle_provider_unavailable(limit_ctx, error_kind="context_overflow")
    else:
        own = loop_output_exhaustion.REASON
        text, usage, trace = loop_output_exhaustion.finish(limit_ctx)
    assert (text, usage["terminal_origin"]) == (HELD_ANSWER, "model_final")
    if stopping:  # the hard stop's own terminal, which the live turn above settles cancelled and unsent
        assert (usage["execution_status"], usage["reason_code"]) == ("failed", "owner_requested_finalization")
        assert trace["forced_finalization"]["source"] == "owner_stop_retained_candidate"
        assert registry._ctx._skip_post_task_synthesis is True
    else:
        assert (usage["execution_status"], usage["reason_code"]) == ("infra_failed", own)
        assert trace["forced_finalization"]["source"] == "retained_candidate"


def _wrap_up(monkeypatch, r, press, *, during="tool", harden=""):
    """The owner presses Wrap up while the turn's tool runs (``during="tool"``) or during its model call;
    Stop now hardens it as the final turn is about to go out (``harden="gate"``) or once that final call
    is already out (``harden="call"``)."""
    import threading
    from ouroboros.llm_attempt import require_physical_dispatch_window
    from ouroboros.tools import knowledge

    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")
    generated, inputs, presses, listed = [], [], {}, []

    def wrap(task_id):
        presses["wrap"] = press(task_id, cascade=True, policy="finalize_then_cancel", action="owner-wrap-up")
        for thread in threading.enumerate():  # the ingress arms the episode off its request thread
            if thread.name.startswith("owner-stop-"):
                thread.join(30)

    listing = knowledge._knowledge_list

    def listing_while_wrapped(ctx, *args, **kwargs):
        listed.append(ctx.task_id)
        if during == "tool":
            wrap(ctx.task_id)
        return listing(ctx, *args, **kwargs)

    monkeypatch.setattr(knowledge, "_knowledge_list", listing_while_wrapped)

    def model(messages):
        tid = h.agents[0].tools._ctx.task_id
        if inputs and harden == "gate":
            presses["stop"] = press(tid, cascade=True, action="owner-stop-now")
        inputs.append(messages)
        require_physical_dispatch_window()
        generated.append(len(generated) + 1)
        if len(generated) == 1:
            if during == "call":
                wrap(tid)
            return {"content": None, "tool_calls": [tool_call("knowledge_list", {}, "look")]}
        if harden == "call":
            presses["stop"] = press(tid, cascade=True, action="owner-stop-now")
        return {"content": SUMMARY, "tool_calls": []}

    h.script = model
    return h, generated, inputs, presses, listed


def _bus(r):
    events = []
    while not r.bus.empty():
        events.append(r.bus.get_nowait())
    return events


def test_public_wrap_up_gives_the_live_turn_its_one_final_answer(owner_stop, monkeypatch):
    """Wrap up is the existing owner-stop episode: one tool-less turn asked to summarize, whose
    answer is the turn's ordinary reply through the reply gate; the watchdog then settles the intent."""
    from supervisor.owner_stop import owner_stop_ending
    from supervisor.task_lifecycle import sweep_cancel_intents

    r, press = owner_stop
    h, generated, inputs, presses, listed = _wrap_up(monkeypatch, r, press)
    result = start(r, h).result.result(timeout=90)
    tid = result.task_id
    wrap = presses["wrap"]
    assert wrap.status_code == 202, wrap.text
    assert wrap.json()["cancel_state"] == "pending" and wrap.json()["stop_policy"] == "finalize_then_cancel"
    assert listed == [tid] and generated == [1, 2]
    assert "[OWNER_STOP]" in json.dumps(inputs[1], ensure_ascii=False)
    assert result.text == SUMMARY
    row = load_task_result(r.root, tid)
    assert row["status"] == "completed" and row["reason_code"] == "owner_requested_finalization"
    assert row["cancel_origin"]["stop_policy"] == "finalize_then_cancel"
    assert owner_stop_ending(row) == ("wrap_up", "summary")
    assert [(out["direction"], out["text"]) for out in _outbound(r.root, tid)] == [("out", SUMMARY)]
    # The episode's one progress toast is a System row in the turn's own room, as for a direct turn.
    [toast] = _bus(r)
    assert (toast["type"], toast["system_type"], toast["chat_id"], toast["role"]) == (
        "send_message", "finalization_notice", row["chat_id"], "system")
    assert tid in toast["text"] and "Stop now remains available" in toast["text"]
    # The cascade intent settles on its postcondition; the answered turn stays completed and the
    # tree's receipt is suppressed (owner Q4=A), exactly as for a task's successful Wrap up.
    assert sweep_cancel_intents(now=time.time() + 3600) == {tid: "cancelled"}
    assert ci.active_intents(r.root) == {} and load_task_result(r.root, tid)["status"] == "completed"
    assert _bus(r) == [] and _outbound(r.root, tid)[-1]["text"] == SUMMARY
    forensic = (r.root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert "owner_stop_summary_suppressed" in forensic
    assert press(tid, cascade=True, policy="finalize_then_cancel", action="owner-wrap-up-2").status_code == 404
    assert not h.executions.live()


def test_stop_now_during_wrap_up_hardens_it_before_the_final_turn_goes_out(owner_stop, monkeypatch):
    from supervisor.owner_stop import owner_stop_ending
    from supervisor.task_lifecycle import sweep_cancel_intents

    r, press = owner_stop
    h, generated, _inputs, presses, _listed = _wrap_up(monkeypatch, r, press, harden="gate")
    with pytest.raises(PresenceTurnError) as stopped:
        start(r, h).result.result(timeout=90)
    tid = h.agents[0].tools._ctx.task_id
    assert presses["wrap"].status_code == 202, presses["wrap"].text
    stop = presses["stop"]
    assert stop.status_code == 202 and stop.json()["stop_policy"] == "immediate", stop.text
    # The armed final turn met the hardened intent at the dispatch gate: nothing more was generated.
    assert generated == [1]
    assert stopped.value.code == "presence_turn_cancelled"
    row = load_task_result(r.root, tid)
    assert row["status"] == "cancelled" and row["cancel_origin"]["stop_policy"] == "immediate"
    assert row["cancel_origin"]["request_id"] == stop.json()["request_id"]  # one request, hardened in place
    assert owner_stop_ending(row) == ("stop_now", "stop")
    assert not _outbound(r.root, tid), "no reply after Stop now"
    assert [event["system_type"] for event in _bus(r)] == ["finalization_notice"]
    assert sweep_cancel_intents(now=time.time() + 3600) == {tid: "cancelled"}
    assert ci.active_intents(r.root) == {} and load_task_result(r.root, tid)["status"] == "cancelled"
    assert [(e["system_type"], e["chat_id"]) for e in _bus(r)] == [("cancel_receipt", row["chat_id"])]


def test_stop_now_hardening_wrap_up_while_its_final_call_is_out_keeps_the_summary_unsent(owner_stop, monkeypatch):
    """The forced neighbour of the round limit: Wrap up's one final call is already out when Stop now
    hardens its request; the summary it returns is kept, cancelled and unsent, with no further call."""
    r, press = owner_stop
    h, generated, inputs, presses, _listed = _wrap_up(monkeypatch, r, press, harden="call")
    stopped = start(r, h).result.exception(timeout=90)
    assert presses["wrap"].status_code == 202, presses["wrap"].text
    assert "[OWNER_STOP]" in json.dumps(inputs[1], ensure_ascii=False)
    assert generated == [1, 2] and h.calls == 2
    stop = presses["stop"]
    tid = h.agents[0].tools._ctx.task_id
    assert [(e["system_type"], e["chat_id"]) for e in _bus(r)] == [
        ("finalization_notice", load_task_result(r.root, tid)["chat_id"])]
    _kept_unsent(r, h, stopped, stop, SUMMARY)


def test_wrap_up_landing_between_a_reply_and_its_tool_starts_nothing_new(owner_stop, monkeypatch):
    """The shared loop's existing ending, now reached by a Presence turn too: the tool the model
    asked for is not started, and the turn ends without a further model call, saying so."""
    from supervisor.owner_stop import owner_stop_ending

    r, press = owner_stop
    h, generated, _inputs, presses, listed = _wrap_up(monkeypatch, r, press, during="call")
    result = start(r, h).result.result(timeout=90)
    tid = result.task_id
    assert presses["wrap"].status_code == 202 and generated == [1] and listed == []
    row = load_task_result(r.root, tid)
    assert row["status"] == "failed" and row["reason_code"] == "owner_requested_finalization"
    assert row["cancel_origin"]["stop_policy"] == "finalize_then_cancel" and result.text == ""
    assert owner_stop_ending(row) == ("wrap_up", "stop")
    assert not _outbound(r.root, tid)


def test_an_unreadable_author_census_is_unknown_never_a_witness(tmp_path, monkeypatch, caplog):
    """No witness is invented: the census answers None and the log calls it unknown. What custody
    and the cascade incident do with that unknown is ``tests/test_presence_stop_owner.py``."""
    import logging

    from ouroboros import conversation_lease
    from ouroboros.presence_runner import presence_execution_witness

    def unreadable(**_kwargs):
        raise OSError("census unreadable")

    monkeypatch.setattr(conversation_lease, "live_authors", unreadable)
    with caplog.at_level(logging.WARNING, logger="ouroboros.presence_runner"):
        assert presence_execution_witness("presence-unknown", drive_root=tmp_path) is None
    assert "liveness unknown" in caplog.text
