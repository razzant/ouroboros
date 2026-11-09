"""A forced wrap-up is its loop's last read of place mail: its intake closes before that rail's first drain.

What landed before the close reaches the rail's one call; what arrives after it is refused and takes the Host's
ordinary route (a focus under the allowance), never accepted by a root no further call reads. A rail that sends nothing
reads nothing new either. A budget lookahead that only prices a wrap-up closes nothing: healthy work keeps reading.
The end-to-end case runs the real Host route, promotion, worker body (``run_worker_task`` -> ``root_intake`` ->
``run_llm_loop``) and round limit; the loop cases run the real forced and budget rails with the model and prices scripted.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import loop
from ouroboros.conversation_lease import author_intake_accepting, open_author_intake
from ouroboros.owner_mailbox import mail_read_state, mailbox_holds, write_task_message
from ouroboros.task_results import load_task_result
from tests.test_configured_session_restore import InputReached
from tests.test_place_conversation_work import (
    attention, call, dispatched, place_rows, post, promote, run_root, until,
)
from tests.test_place_conversation_work import host as host  # fixture

pytestmark = pytest.mark.usefixtures("provider_catalog_offline")
KEY = "slack:T1:C1:1.0"


def test_a_follow_up_during_a_roots_forced_wrap_up_is_refused_and_an_ordinary_focus_takes_it(host, monkeypatch):
    world = host()
    task = dispatched(world, promote(world))
    key = task["metadata"]["presence"]["event"]["conversation_key"]
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")  # one tool round, then the ONE forced wrap-up call
    seen, root_inputs = {}, []

    def tool_round(messages):
        root_inputs.append(json.dumps(messages, ensure_ascii=False))
        seen["early"] = post(world, text="The backup job too, please.")  # while the loop still reads
        return call(("chat_history", {}))

    def forced_call(messages):
        root_inputs.append(json.dumps(messages, ensure_ascii=False))
        seen["intake"] = author_intake_accepting(world.data, task["id"], key, attempt=1)
        seen["late"] = post(world, text="And the certificates.")
        return {"content": "Forced wrap-up: two of three services checked."}

    box = run_root(world, task, tool_round, forced_call)
    until(lambda: (load_task_result(world.data, task["id"]) or {}).get("status") in {"completed", "failed"},
          "the root's terminal")
    box["thread"].join(30)
    row = load_task_result(world.data, task["id"]) or {}
    assert row.get("status") == "completed" and len(root_inputs) == 2, (row.get("status"), len(root_inputs))
    assert "The backup job too, please." in root_inputs[1], "mail that landed before the close reaches the one call"
    assert attention(world, seen["early"][0])["recipient"] == task["id"]
    assert seen["intake"] is False, "closed before the forced rail's first drain"
    assert attention(world, seen["late"][0]) == {}, "refused by the closed root, and not plain heard"
    assert [r["place_arrival"]["source_event_id"] for r in place_rows(world.data, task["id"])] == [seen["early"][0]]
    until(lambda: any("And the certificates." in words for words in world.focus_inputs), "the ordinary focus")
    assert not any("And the certificates." in words for words in root_inputs)


def _door(root, task_id):
    return lambda: author_intake_accepting(root, task_id, KEY, attempt=1)


def _arrive(root, task_id, name, text):
    """The Host's work door: one locked append admitted only while the reader's durable intake is open."""
    return bool(write_task_message(root, text, task_id, source_task_id=f"arrival-{name}", provenance="place_event",
                                   msg_id=f"place-event:{name}", place_arrival={"source_event_id": name, "message": {},
                                                                                "wake": True},
                                   admit=_door(root, task_id)))


def _reader(ctx, task_id, *, presence=True, accepting=False):
    """The facts a promoted root (or an inline Presence author) reads by: its attempt-bound durable intake beside
    its mailbox, and the agent's local flag under its admission lock."""
    if presence:
        ctx.task_metadata = {**(getattr(ctx, "task_metadata", None) or {}),
                             "presence": {"binding_id": "b", "event": {"conversation_key": KEY}}}
        assert open_author_intake(ctx.drive_root, task_id, KEY, attempt=1)  # what ``root_intake`` opens
    ctx.owner_message_admission_lock = threading.Lock()
    ctx.owner_message_admission_agent = SimpleNamespace(_accepting_owner_messages=accepting, _busy=True,
                                                        _current_task_id=task_id)
    return ctx.owner_message_admission_agent


@pytest.mark.parametrize("arrive", ["during_forced_call", "after_forced_call_before_return"])
def test_the_round_limit_rail_reads_what_came_before_and_accepts_nothing_after(tmp_path, monkeypatch, arrive):
    from tests.test_completion_hold_observation import runner, tool_call

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    turn = runner(tmp_path, monkeypatch, task_id="root-task")
    ctx = turn.registry._ctx
    ctx.is_direct_chat, ctx.task_id, ctx.task_attempt = False, "root-task", 1
    agent = _reader(ctx, ctx.task_id, accepting=True)  # an inline author's flag is its place door too
    observed = {}

    def tool_round(*_a, **_kw):
        observed["before"] = _arrive(tmp_path, ctx.task_id, "before", "Landed while the loop read.")
        return "Synthetic history"

    turn.registry.override_handler("chat_history", tool_round)

    def forced_call():
        observed["local_at_call"] = agent._accepting_owner_messages
        if arrive == "during_forced_call":
            observed["late"] = _arrive(tmp_path, ctx.task_id, "late", "Arrived during the forced call.")
        return {"content": "Forced wrap-up final."}

    if arrive == "after_forced_call_before_return":
        real = loop._record_forced_finalization

        def record(*args, **kwargs):
            observed.setdefault("late", _arrive(tmp_path, ctx.task_id, "late", "Arrived before the rail returned."))
            return real(*args, **kwargs)

        monkeypatch.setattr(loop, "_record_forced_finalization", record)
    first = {"role": "assistant", "content": None, "tool_calls": [tool_call("chat_history", "read")]}
    _text, usage, _trace = turn.run([first, forced_call])
    words = [json.dumps(call_input, ensure_ascii=False) for call_input in turn.calls]
    assert usage.get("reason_code") == "round_limit" and len(words) == 2
    assert observed["before"] and "Landed while the loop read." in words[1], "before the close: in the one call"
    assert observed["late"] is False and not mailbox_holds(tmp_path, ctx.task_id, "place-event:late")
    assert observed["local_at_call"] is False and _door(tmp_path, ctx.task_id)() is False


def test_a_direct_turns_forced_rail_keeps_its_owner_route_open(tmp_path, monkeypatch):
    """No place reads a direct turn: its local flag is its owner's route and Stop door, and the rail leaves it."""
    from tests.test_completion_hold_observation import runner, tool_call

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    turn = runner(tmp_path, monkeypatch)
    agent = _reader(turn.registry._ctx, turn.registry._ctx.task_id, presence=False, accepting=True)
    turn.registry.override_handler("chat_history", lambda *_a, **_kw: "Synthetic history")
    seen = []
    first = {"role": "assistant", "content": None, "tool_calls": [tool_call("chat_history", "read")]}
    _text, usage, _trace = turn.run([first, lambda: seen.append(agent._accepting_owner_messages) or {"content": "Final."}])
    assert usage.get("reason_code") == "round_limit" and seen == [True] and agent._accepting_owner_messages is True


def test_a_direct_owner_turns_forced_rail_closes_its_durable_intake_and_keeps_its_owner_route(tmp_path, monkeypatch):
    """Main's door reads an owner turn's durable fact (``owner:<chat>``): the forced rail is the loop's last read, so it
    closes there before the rail's drain, as a final drain closes it; the local flag stays its route and Stop door."""
    from tests.test_completion_hold_observation import runner, tool_call

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    turn = runner(tmp_path, monkeypatch, task_id="main-turn")
    ctx = turn.registry._ctx
    ctx.task_id, ctx.task_attempt, ctx.task_metadata = "main-turn", None, {}
    agent = _reader(ctx, "main-turn", presence=False, accepting=True)
    assert open_author_intake(tmp_path, "main-turn", "owner:1", attempt=1)
    turn.registry.override_handler("chat_history", lambda *_a, **_kw: "Synthetic history")
    seen = []

    def forced_call():
        seen.append((agent._accepting_owner_messages,
                     author_intake_accepting(tmp_path, "main-turn", "owner:1", attempt=1)))
        return {"content": "Final."}

    first = {"role": "assistant", "content": None, "tool_calls": [tool_call("chat_history", "read")]}
    _text, usage, _trace = turn.run([first, forced_call])
    assert usage.get("reason_code") == "round_limit" and seen == [(True, False)]
    assert agent._accepting_owner_messages is True


def _budget_rail(tmp_path, monkeypatch, *, exact_fits, second_fits, accepting=False):
    """The real budget rails with their prices scripted: ``land`` the soft landing that always ends the task,
    ``check`` the explicit profile's lookahead after a round, which may also admit ordinary work again."""
    from ouroboros import budget_pause, loop_budget, task_pacing
    from tests.test_late_tool_result_delivery import TASK, _limit_ctx, _loop_registry
    from tests.test_tree_cost_ceiling import EXPLICIT_50

    registry = _loop_registry(tmp_path, monkeypatch)
    agent = _reader(registry._ctx, TASK, accepting=accepting)
    observed, priced, sent, exact = {"between": []}, [], [], object()

    def price(_ctx, prospective, **_kwargs):  # the exact candidate; an arrival lands right after the first pricing
        priced.append(json.loads(json.dumps(prospective)))
        observed.setdefault("door_at_lookahead", _door(tmp_path, TASK)())
        if len(priced) == 1:
            observed["between"].append(_arrive(tmp_path, TASK, "between", "Landed after the lookahead's drain."))
        return exact, prospective

    def send(_llm, messages, *_args, initial_messages=None, **_kwargs):
        observed["door_at_send"], observed["local_at_send"] = _door(tmp_path, TASK)(), agent._accepting_owner_messages
        sent.append(json.loads(json.dumps(messages if initial_messages is None else initial_messages)))
        raise InputReached

    monkeypatch.setattr(budget_pause, "request_pause", lambda *_a, **_k: None)  # the pause seed is not the subject
    monkeypatch.setattr(task_pacing, "prepared_wrapup_candidate", price)
    monkeypatch.setattr(task_pacing, "prospective_wrapup_attempt_request", lambda **_k: object())
    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits",
                        lambda **kw: exact_fits if kw.get("request") is exact else False)
    monkeypatch.setattr(loop_budget, "_second_reservation_fits", lambda *_a, **_k: second_fits)
    monkeypatch.setattr(loop_budget, "_wrapup_global_remaining", lambda: 1.0)
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_k: None)
    monkeypatch.setattr(loop, "call_llm_with_retry", send)
    ctx = _limit_ctx(registry, {"tool_calls": []}, [{"role": "user", "content": "go"}], tmp_path / "logs")
    ctx.accumulated_usage["_context_prompt_estimate"] = 1000
    registry._ctx._cost_stop_policy = {"authority": task_pacing.COST_STOP_EXPLICIT}  # what its start read
    exhausted, active = (task_pacing.resolve_cost_ceiling(100.0, EXPLICIT_50, root_cap_usd=cap) for cap in (0.5, 10.0))
    return SimpleNamespace(ctx=ctx, agent=agent, observed=observed, priced=priced, sent=sent, task_id=TASK,
                           land=lambda: loop._soft_land_exhausted_ceiling(ctx, exhausted),
                           check=lambda: loop._check_budget_limits(ctx, 5.0, active))


def test_an_admitted_budget_wrap_up_closes_at_its_entry_and_reprices_what_landed_after_its_lookahead(
        tmp_path, monkeypatch):
    rail = _budget_rail(tmp_path, monkeypatch, exact_fits=True, second_fits=False)
    with pytest.raises(InputReached):
        rail.land()
    assert rail.observed["door_at_lookahead"] is True and rail.observed["between"] == [True], "pricing closes nothing"
    assert rail.observed["door_at_send"] is False
    (request,) = rail.sent
    assert "Landed after the lookahead's drain." in json.dumps(request, ensure_ascii=False)
    assert len(rail.priced) == 2 and request == rail.priced[-1], "re-priced with it before the one admitted send"


def test_a_budget_rail_that_sends_nothing_closes_intake_and_claims_no_read(tmp_path, monkeypatch):
    rail = _budget_rail(tmp_path, monkeypatch, exact_fits=False, second_fits=False)
    real, late = loop._record_forced_finalization, []

    def record(*args, **kwargs):
        late.append(_arrive(tmp_path, rail.task_id, "late", "Arrived as the rail ended."))
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "_record_forced_finalization", record)
    _text, usage, trace = rail.land()
    assert trace["forced_finalization"]["source"] == "budget_wrapup_unaffordable" and rail.sent == []
    assert late == [False] and not mailbox_holds(tmp_path, rail.task_id, "place-event:late")
    assert rail.observed["between"] == [True] and mail_read_state(tmp_path, rail.task_id, "place-event:between") is False
    assert not any("Landed after the lookahead's drain." in json.dumps(row) for row in rail.ctx.messages)


def test_a_budget_lookahead_that_admits_ordinary_work_leaves_its_intake_open(tmp_path, monkeypatch):
    rail = _budget_rail(tmp_path, monkeypatch, exact_fits=True, second_fits=True, accepting=True)
    assert rail.check() is None and rail.sent == [] and len(rail.priced) == 1, "priced once, nothing committed"
    assert _door(tmp_path, rail.task_id)() is True and rail.agent._accepting_owner_messages is True
    assert rail.observed["between"] == [True] and mailbox_holds(tmp_path, rail.task_id, "place-event:between")
