"""#2570, owner 03.11 option 3: a place-carrying root whose final no connector reads hears that fact once.

The real Host turn that builds the speaker carrier, the real ``run_llm_loop`` with its finalization gates, working
checkpoints and recovery, the real scheduler and promotion (``fired_context``), the pipeline that publishes the result,
and an in-process ``Transport`` (``test_presence_explicit_final``) run; only the model is scripted. The connector's send
door is a declared ``effect: message`` tool (``records_room``'s declaration). Synthetic data; nothing leaves the
process and no provider is contacted. The host never sends the final itself: every place send below is the model's.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import extension_registry_state as registry_state
from ouroboros.place_reply import NO_SEND_KEY
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolEntry, ToolRegistry
from tests.test_presence_explicit_final import (
    ACK,
    FINAL,
    NOTE,
    REACT_SCHEMA,
    RECORDS_REACT,
    RECORDS_SEND,
    SEND_SCHEMA,
    finish,
    record_send,
    settle,
    turn,
)
from tests.test_presence_explicit_final import room as room  # fixture

pytestmark = [pytest.mark.usefixtures("provider_catalog_offline")]

MARK = "[ROUTELESS_FINAL]"
SCHEDULED = {"schedule_id": "s-routeless", "occurrence_id": "o-1"}


def call(name, arguments, identifier=None):
    return {"content": None, "tool_calls": [{"id": identifier or name, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}]}


def open_door(monkeypatch):
    """The connector declares its in-conversation send door and a reaction door (``gesture_room``'s publication)."""
    manifest = SimpleNamespace(raw_extra={"tools": [
        {"name": "send", "effect": "message", "target": ["conversation_id"], "conversation": "conversation_id",
         "thread": "thread_id"},
        {"name": "react", "effect": "gesture", "target": ["account_id", "conversation_id"],
         "conversation": "conversation_id", "message": "message_id"}]})
    bundle = registry_state._ExtensionRegistrations(generation_digest="g1", loaded_skill=SimpleNamespace(manifest=manifest))
    for name in (RECORDS_SEND, RECORDS_REACT):
        monkeypatch.setitem(registry_state._tools, name, {"skill": "bridge", "extension_generation": "g1"})
    monkeypatch.setitem(registry_state._extensions, "bridge", bundle)


class Crash(BaseException):
    """The worker dies inside this model call: nothing after it runs."""


def carrier(room):
    """The speaker carrier a real turn of this conversation leaves (the place's door is not yet declared)."""
    room.h.script = [finish(answer="I will look again later.")]
    body = turn(room)
    room.transport.submit("chat:a:1", body)
    return dict(load_task_result(room.data, body["turn_ref"])["metadata"]["presence"])


def run_root(room, monkeypatch, responses, *, metadata=None, root_id="routeless-root", attempt=None, recovery=None,
             presence=None, setup=None):
    """One place-carrying root in the real loop and pipeline; ``responses`` script its model (a callable reads the
    messages; ``Crash`` ends the worker there). Returns its trace, model inputs, stored row and pipeline events."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros import loop

    meta = metadata if metadata is not None else {"presence": presence or carrier(room), "delegation_role": "root",
                                                  "schedule_occurrence": dict(SCHEDULED)}
    open_door(monkeypatch)
    from ouroboros import context
    from tests.test_context_fit_v664 import _plan

    registry = ToolRegistry(repo_dir=room.h.repo, drive_root=room.data)
    ctx = registry._ctx
    ctx.task_id, ctx.task_metadata, ctx.is_direct_chat, ctx.task_attempt = root_id, meta, False, attempt
    # A restored attempt rebinds the route it saved (``rebind_restored_route``), as the agent's start would.
    ctx.active_model, ctx.active_effort, ctx.active_use_local, ctx.active_context_mode = "test-model", "high", False, "max"
    ctx.context_fit_plan, ctx.owner_wait_callback = _plan(window=500_000, known=True), (lambda *_: None)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="route-a")))
    if recovery is not None:
        ctx.working_recovery = recovery
    registry.register(ToolEntry(name=RECORDS_SEND, schema={**SEND_SCHEMA, "name": RECORDS_SEND},
                                handler=room.transport.tool, timeout_sec=30, safety_policy="skip"))
    registry.register(ToolEntry(name=RECORDS_REACT, schema=REACT_SCHEMA, handler=room.transport.react,
                                timeout_sec=30, safety_policy="skip"))
    if setup is not None:
        setup(ctx)
    task = {"id": root_id, "type": "task", "chat_id": 1, "text": "Look again", "metadata": meta,
            "delegation_role": "root", "_skip_post_task_synthesis": True}
    write_task_result(room.data, root_id, "running", metadata=meta, delegation_role="root", root_task_id=root_id,
                      chat_id=1)
    script, inputs = iter(responses), []

    def respond(_llm, messages, *_a, **_kw):
        inputs.append(json.loads(json.dumps(messages)))
        response = next(script, None)
        if response is Crash:
            raise Crash()
        return (response(messages) if callable(response) else response), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    sent_before, events = list(room.transport.provider), queue.Queue()
    try:
        text, usage, trace = loop.run_llm_loop(
            [{"role": "user", "content": "Look again"}], registry, SimpleNamespace(default_model=lambda: "test-model"),
            room.data / "logs", lambda *_a, **_kw: None, queue.Queue(), task_id=root_id, drive_root=room.data,
            event_queue=events)
    except Crash:
        return SimpleNamespace(id=root_id, inputs=inputs, crashed=True, ctx=ctx, meta=meta)
    pending = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=room.data, repo_dir=room.h.repo), None, None, pending,
                               task, text, usage, trace, 0.0, room.data / "logs", ctx=ctx)
    return SimpleNamespace(id=root_id, trace=trace, inputs=inputs, stored=load_task_result(room.data, root_id),
                           pending=pending, crashed=False, ctx=ctx, meta=meta, usage=usage,
                           sent=room.transport.provider[len(sent_before):], events=list(events.queue))


def notes(run):
    return [row["content"] for messages in run.inputs[-1:] for row in messages
            if row.get("role") == "user" and MARK in str(row.get("content"))]


def all_notes(run):
    return sum(json.dumps(messages, ensure_ascii=False).count(MARK) for messages in run.inputs[-1:])


# --- the one reminder and its choices -------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["selection", "prose"])
@pytest.mark.parametrize("then", ["send_and_silence", "silence", "record"])
def test_a_routeless_wordy_final_returns_once_and_the_mind_chooses(room, monkeypatch, first, then):
    """The schedule-born root ends with words nobody receives: one more round reads the fact, the doors and the two
    finishes; then it sends through the door, stays quiet, or keeps its words as the record. The host sends nothing."""
    opening = finish(answer=FINAL) if first == "selection" else {"content": FINAL, "tool_calls": []}
    after = {"send_and_silence": [{"content": None, "tool_calls": [*record_send(text=FINAL)["tool_calls"],
                                                                   *finish(silent=True, answer=NOTE)["tool_calls"]]}],
             "silence": [finish(silent=True, answer=NOTE)],
             "record": [finish(answer=FINAL)]}[then]
    run = run_root(room, monkeypatch, [opening, *after])
    assert len(run.inputs) == 2, "exactly one extra model round"
    [note] = notes(run)
    assert "Nobody receives this text" in note and "it started from a schedule" in note and RECORDS_SEND in note
    assert "the host sends nothing for you" in note and "silent=true sends nothing" in note
    assert all_notes(run) == 1 and FINAL in json.dumps(run.inputs[-1], ensure_ascii=False)  # the answer stays in view
    # The mind's own send after the note makes the lineage possibly speaking; the note stays spent.
    assert run.trace[NO_SEND_KEY] == {"sends": "possible" if then == "send_and_silence" else "none", "noted": True}
    sent = [row["text"] for row in run.sent]
    assert sent == ([FINAL] if then == "send_and_silence" else []), "only the model's own send reaches the place"
    if then == "record":
        assert run.stored["result"] == FINAL and run.stored["metadata"]["presence_outcome"] == "message"
    else:
        assert run.stored["result"] == NOTE and run.stored["metadata"]["presence_outcome"] == "silent"
    assert "FINISH_NOT_ACCEPTED" not in json.dumps(run.inputs, ensure_ascii=False)
    assert "No completion selection was made" not in json.dumps(run.inputs, ensure_ascii=False)


def test_the_note_after_prose_takes_prose_again_without_a_selection_round(room, monkeypatch):
    run = run_root(room, monkeypatch, [{"content": FINAL, "tool_calls": []}, {"content": FINAL, "tool_calls": []}])
    assert len(run.inputs) == 2 and run.stored["result"] == FINAL and run.sent == []


def test_known_reads_and_own_records_keep_the_lineage_silent(room, monkeypatch):
    """Reads, this mind's own memory and the local-room delivery family say nothing in the place: the reminder still
    comes. A gesture is not words either."""
    reads = {"content": None, "tool_calls": [
        *call("list_files", {"path": "."}, "read")["tool_calls"],
        *call("knowledge_list", {}, "memory")["tool_calls"],
        *call("update_scratchpad", {"content": "looked again"}, "scratch")["tool_calls"]]}
    run = run_root(room, monkeypatch, [reads, finish(answer=FINAL), finish(answer=FINAL)])
    assert len(run.inputs) == 3 and len(notes(run)) == 1 and run.trace[NO_SEND_KEY]["sends"] == "none"


@pytest.mark.parametrize("batch", ["door_send", "opaque_command", "browser", "handed_to_task"])
def test_a_call_that_might_have_spoken_suppresses_the_hint(room, monkeypatch, batch):
    """A send through the door, an opaque command, a browser act or words handed to another task may have reached the
    place: the optional hint stays away and the finish is ordinary. No tool is blocked."""
    step = {"door_send": record_send(text=ACK),
            "opaque_command": call("run_command", {"cmd": "echo checked"}),
            "browser": call("browser_action", {"action": "screenshot"}),
            "handed_to_task": call("tell_task", {"task_id": "someone-else", "note": FINAL})}[batch]
    run = run_root(room, monkeypatch, [step, finish(answer=FINAL)])
    assert len(run.inputs) == 2 and MARK not in json.dumps(run.inputs, ensure_ascii=False)
    assert run.trace[NO_SEND_KEY]["sends"] == "possible" and "noted" not in run.trace[NO_SEND_KEY]
    assert [row["text"] for row in run.sent] == ([ACK] if batch == "door_send" else [])
    assert run.stored["result"] == FINAL


def test_a_send_that_times_out_before_its_guard_is_possible_speech(room, monkeypatch):
    """The call's worker has not reached the place guard when its wait ends: the fact was set before submission."""
    import threading

    from ouroboros import loop_tool_execution as execution
    from ouroboros import place_guard

    release, real = threading.Event(), place_guard.dispatch_refusal

    def slow_guard(*args, **kwargs):
        release.wait(20)
        return real(*args, **kwargs)

    monkeypatch.setattr(place_guard, "dispatch_refusal", slow_guard)
    timeouts = execution._get_tool_timeout
    monkeypatch.setattr(execution, "_get_tool_timeout",
                        lambda tools, name, args=None: 1 if name == RECORDS_SEND else timeouts(tools, name, args))
    try:
        run = run_root(room, monkeypatch, [record_send(text=ACK), finish(answer=FINAL)])
    finally:
        release.set()
    assert any(row.get("tool") == RECORDS_SEND and row.get("is_error") for row in run.trace["tool_calls"])
    assert len(run.inputs) == 2 and MARK not in json.dumps(run.inputs, ensure_ascii=False)
    assert run.trace[NO_SEND_KEY]["sends"] == "possible"


# --- what buys no round ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ending", ["silent", "stop", "round_limit", "deadline", "budget"])
def test_silence_stop_and_every_rail_buy_no_round(room, monkeypatch, ending):
    from datetime import datetime, timedelta, timezone

    if ending == "round_limit":
        monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    if ending == "budget":
        monkeypatch.setattr("ouroboros.loop_budget._next_round_pays", lambda _ctx: True)
        monkeypatch.setattr("ouroboros.loop_budget._wrapup_global_remaining", lambda: 0.0)

    def finishing(_messages):
        if ending == "deadline":  # the task's own deadline lands inside the finalization reserve during this round
            run_ctx = getattr(finishing, "ctx")
            run_ctx.task_metadata["deadline_at"] = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
        # A tool-tail finish meets the deadline rail before its selection (``_maybe_early_finalize``): prose here.
        return {"silent": finish(silent=True, answer=NOTE),
                "stop": call("finish_task", {"action": "stop", "rationale": "Nothing to add until tomorrow.",
                                             "answer": FINAL}),
                "deadline": {"content": FINAL, "tool_calls": []}}.get(ending, finish(answer=FINAL))

    presence = carrier(room)
    meta = {"presence": presence, "delegation_role": "root", "schedule_occurrence": dict(SCHEDULED)}
    finishing.ctx = SimpleNamespace(task_metadata=meta)
    run = run_root(room, monkeypatch, [finishing, finish(answer=FINAL)], metadata=meta)
    assert len(run.inputs) == 1 and MARK not in json.dumps(run.inputs, ensure_ascii=False)
    assert "noted" not in run.trace[NO_SEND_KEY] and run.sent == []


def test_empty_words_and_unsettled_routes_get_nothing(room, monkeypatch):
    """A root admitted before handoffs were recorded is UNKNOWN, a turn is ROUTED: neither is told nobody hears it."""
    presence = carrier(room)
    unknown = run_root(room, monkeypatch, [finish(answer=FINAL)], root_id="unknown-root",
                       metadata={"presence": presence, "delegation_role": "root"})
    assert len(unknown.inputs) == 1 and MARK not in json.dumps(unknown.inputs, ensure_ascii=False)
    from ouroboros.place_reply import routeless_note

    candidate = SimpleNamespace(quiet=False)
    state = {NO_SEND_KEY: {"sends": "none"}}
    assert routeless_note(unknown.ctx, SimpleNamespace(), state, "   ", candidate) == ""
    assert routeless_note(unknown.ctx, SimpleNamespace(), state, FINAL, SimpleNamespace(quiet=True)) == ""


@pytest.mark.parametrize("state", ["silent", "observe", "no_door"])
def test_a_place_that_cannot_receive_a_send_gets_no_hint(room, monkeypatch, state):
    from ouroboros import places

    presence = carrier(room)
    meta = {"presence": presence, "delegation_role": "root", "schedule_occurrence": dict(SCHEDULED)}
    if state == "silent":
        places.edit_place(room.data, room.place.place_id, "set_state", author="owner", state="silent")
    if state == "observe":
        meta.update(initiator="consciousness", consciousness_autonomy="observe")
        monkeypatch.setattr("ouroboros.consciousness_authority.is_observe_origin", lambda _m: True)
    if state == "no_door":
        meta["presence"] = {**presence, "transport_skill": "another-connector"}
    run = run_root(room, monkeypatch, [finish(answer=FINAL), finish(answer=FINAL)], metadata=meta)
    assert len(run.inputs) == 1 and MARK not in json.dumps(run.inputs, ensure_ascii=False) and run.sent == []


# --- lineage: fresh, lost, retried, continued, restored -----------------------------------------------------------


@pytest.mark.parametrize("start", ["second_attempt", "lost_recovery", "timeout_retry", "continuation", "original_task_id"])
def test_a_start_that_is_not_positively_fresh_knows_no_silence(room, monkeypatch, start):
    presence = carrier(room)
    meta = {"presence": presence, "delegation_role": "root", "schedule_occurrence": dict(SCHEDULED)}
    if start == "timeout_retry":
        meta["timeout_retry_from"] = "earlier-root"
    if start == "continuation":
        meta["continuation"] = {"predecessor_task_id": "earlier-root"}
    if start == "original_task_id":
        meta["original_task_id"] = "earlier-root"
    run = run_root(room, monkeypatch, [finish(answer=FINAL), finish(answer=FINAL)], metadata=meta,
                   attempt=2 if start == "second_attempt" else None,
                   recovery={"source_ref": {"sha256": "0" * 64}, "source_task_id": "routeless-root", "from_attempt": 1}
                   if start == "lost_recovery" else None)
    assert NO_SEND_KEY not in run.trace and len(run.inputs) == 1
    assert MARK not in json.dumps(run.inputs, ensure_ascii=False)
    if start == "lost_recovery":
        assert "saved working state could not be restored" in json.dumps(run.inputs[0], ensure_ascii=False)


def _recover(room, task_id, attempt=1):
    from ouroboros.working_checkpoint import prepare_recovery

    handoff = prepare_recovery(room.data, task_id, from_attempt=attempt, cause="worker_crash")
    assert handoff, "the dead attempt left a working checkpoint"
    return handoff


@pytest.mark.parametrize("identity", ["same", "new"])
@pytest.mark.parametrize("before", ["read", "door_send"])
def test_a_retried_lineage_keeps_what_it_knew(room, monkeypatch, identity, before):
    """The dead attempt's coverage rides its working state into the retry, a new execution ID included."""
    presence = carrier(room)
    step = call("list_files", {"path": "."}, "read") if before == "read" else record_send(text=ACK)
    first = run_root(room, monkeypatch, [step, Crash], presence=presence, root_id="lineage-root")
    assert first.crashed
    handoff = _recover(room, "lineage-root")
    retry_id = "lineage-root" if identity == "same" else "lineage-retry"
    meta = dict(first.meta, **({} if identity == "same" else {"original_task_id": "lineage-root"}))
    second = run_root(room, monkeypatch, [finish(answer=FINAL), finish(answer=FINAL)], metadata=meta,
                      root_id=retry_id, attempt=2, recovery=handoff)
    told = MARK in json.dumps(second.inputs, ensure_ascii=False)
    assert told == (before == "read") and len(second.inputs) == (2 if before == "read" else 1)
    assert second.trace[NO_SEND_KEY]["sends"] == ("none" if before == "read" else "possible")


def test_an_unanswered_call_restored_from_its_pre_effect_state_is_possible_speech(room, monkeypatch):
    """The worker dies inside the send's own handler: the restored batch has no result for it, and it may have spoken."""
    from ouroboros.working_checkpoint import read_working

    presence = carrier(room)
    original = ToolRegistry.register

    def register(self, entry):
        if entry.name == RECORDS_SEND:
            def dying_send(ctx, **_kwargs):
                raise Crash()
            entry = ToolEntry(name=entry.name, schema=entry.schema, handler=dying_send, timeout_sec=30,
                              safety_policy="skip")
        return original(self, entry)

    monkeypatch.setattr(ToolRegistry, "register", register)
    first = run_root(room, monkeypatch, [record_send(text=ACK)], presence=presence, root_id="unanswered-root")
    assert first.crashed
    monkeypatch.setattr(ToolRegistry, "register", original)
    _bytes, saved = read_working(room.data, "unanswered-root", 1)
    assert saved["working"]["boundary"] == "pre_effect" and saved["trace"][NO_SEND_KEY]["sends"] == "possible"
    second = run_root(room, monkeypatch, [finish(answer=FINAL)], metadata=first.meta, root_id="unanswered-root",
                      attempt=2, recovery=_recover(room, "unanswered-root"))
    assert "execution state is UNKNOWN" in json.dumps(second.inputs[0], ensure_ascii=False)
    assert len(second.inputs) == 1 and MARK not in json.dumps(second.inputs, ensure_ascii=False)


@pytest.mark.parametrize("identity", ["same", "new"])
def test_a_restart_right_after_the_note_reads_it_before_any_final_and_never_a_second(room, monkeypatch, identity):
    """The note, the answer and the spent mark are one saved ``ready`` state: the restored attempt calls the model
    with that note first, accepts its next finish and adds no other note."""
    from ouroboros.working_checkpoint import read_working

    presence = carrier(room)
    first = run_root(room, monkeypatch, [finish(answer=FINAL), Crash], presence=presence, root_id="noted-root")
    assert first.crashed and len(first.inputs) == 2 and MARK in json.dumps(first.inputs[1], ensure_ascii=False)
    _bytes, saved = read_working(room.data, "noted-root", 1)
    assert saved["working"]["boundary"] == "ready" and saved["trace"][NO_SEND_KEY] == {"sends": "none", "noted": True}
    assert sum(MARK in json.dumps(row, ensure_ascii=False) for row in saved["messages"]) == 1
    retry_id = "noted-root" if identity == "same" else "noted-retry"
    second = run_root(room, monkeypatch, [finish(silent=True, answer=NOTE), finish(answer=FINAL)], metadata=first.meta,
                      root_id=retry_id, attempt=2, recovery=_recover(room, "noted-root"))
    assert len(second.inputs) == 1, "the restored note is read by a model call; no saved final is accepted first"
    assert json.dumps(second.inputs[0], ensure_ascii=False).count(MARK) == 1
    assert second.stored["result"] == NOTE and second.stored["metadata"]["presence_outcome"] == "silent"
    assert second.sent == []


def test_a_death_before_the_note_state_lands_shows_it_once_after_the_saved_candidate(room, monkeypatch):
    """The worker dies while saving the note: the last durable state is the candidate, so the restored finalization
    meets the reminder for the first time, shows it once and buys one round."""
    from ouroboros import working_checkpoint
    from ouroboros.working_checkpoint import read_working

    real = working_checkpoint.save_round

    def dying(limit_ctx, boundary):
        if boundary == "ready" and MARK in json.dumps(limit_ctx.messages[-1:], ensure_ascii=False):
            raise Crash()
        return real(limit_ctx, boundary)

    presence = carrier(room)
    monkeypatch.setattr(working_checkpoint, "save_round", dying)
    first = run_root(room, monkeypatch, [{"content": FINAL, "tool_calls": []}], presence=presence, root_id="dying-root")
    monkeypatch.setattr(working_checkpoint, "save_round", real)
    assert first.crashed and len(first.inputs) == 1
    _bytes, saved = read_working(room.data, "dying-root", 1)
    assert saved["working"]["boundary"] == "candidate" and "noted" not in saved["trace"][NO_SEND_KEY]
    second = run_root(room, monkeypatch, [finish(silent=True, answer=NOTE), finish(answer=FINAL)], metadata=first.meta,
                      root_id="dying-root", attempt=2, recovery=_recover(room, "dying-root"))
    assert len(second.inputs) == 1 and json.dumps(second.inputs[0], ensure_ascii=False).count(MARK) == 1
    assert second.stored["metadata"]["presence_outcome"] == "silent" and second.sent == []


def test_an_unwritable_note_state_shows_nothing_and_the_finish_proceeds(room, monkeypatch):
    from ouroboros import working_checkpoint

    real = working_checkpoint.save_round

    def failing(limit_ctx, boundary):
        if boundary == "ready" and MARK in json.dumps(limit_ctx.messages[-1:], ensure_ascii=False):
            raise OSError("disk full")
        return real(limit_ctx, boundary)

    monkeypatch.setattr(working_checkpoint, "save_round", failing)
    run = run_root(room, monkeypatch, [finish(answer=FINAL), finish(answer=FINAL)])
    assert len(run.inputs) == 1 and MARK not in json.dumps(run.inputs, ensure_ascii=False)
    assert "noted" not in run.trace[NO_SEND_KEY] and run.stored["result"] == FINAL and run.sent == []


# --- real producers: the scheduler and a second promotion ---------------------------------------------------------


def _team_turn(room, monkeypatch):
    """A real admitted turn in a second conversation of the same connector, its registry and the scheduler's queue."""
    import supervisor.queue as supervisor_queue
    from ouroboros import places
    from tests.test_presence_final_audience import THREAD, _source_turn

    monkeypatch.setattr("supervisor.consciousness_carriers.autonomous_work_enabled", lambda *_a, **_kw: True,
                        raising=False)
    places.connect_place(room.data, connector="bridge", provider="chat", account_id="team", wakes=["mention"])
    supervisor_queue.init(room.data)
    pending: list = []
    supervisor_queue.init_queue_refs(pending, {}, {"value": 0})
    registry, task = _source_turn(room.data, THREAD)
    # The scheduler's queue init published a pool without workers; the promote handler below stands for it.
    monkeypatch.setattr("ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    return registry, task, pending


def test_the_real_scheduler_and_a_second_promotion_produce_roots_that_hear_it(room, monkeypatch):
    """A follow-up the real ``schedule_followup`` and scheduler fire, and a turn's second ``start_task`` admitted by the
    supervisor's promote handler, are routeless by ``final_route``; each hears the one fact. The first handoff the
    turn published is routed and hears nothing."""
    from ouroboros.presence_context import ROUTED, ROUTELESS, final_route
    from tests.test_presence_final_audience import _promote
    from tests.test_presence_final_route import _ended, _fired

    registry, source, pending = _team_turn(room, monkeypatch)
    fired = _fired(room.data, pending, registry)
    first = _promote(room.data, monkeypatch, source, registry)
    second = _promote(room.data, monkeypatch, source, registry)
    _ended(room.data, source, registry)
    assert final_route(room.data, first["metadata"], first["id"])[0] == ROUTED
    for task, heard in ((fired, True), (second, True), (first, False)):
        if heard:
            assert final_route(room.data, task["metadata"], task["id"])[0] == ROUTELESS
        run = run_root(room, monkeypatch, [finish(answer=FINAL), finish(answer=FINAL)],
                       metadata=json.loads(json.dumps(task["metadata"])), root_id=task["id"])
        assert len(run.inputs) == (2 if heard else 1) and len(notes(run)) == int(heard), task["id"]
        assert run.sent == []


def test_no_note_after_the_loop_even_when_the_route_settles_later(room, monkeypatch):
    """A root whose route is unsettled while it runs finishes in one call; its source ending afterwards without
    passing it on makes it routeless, but nothing returns to the model and the pipeline publishes one result."""
    from ouroboros.presence_context import ROUTELESS, UNKNOWN, final_route
    from tests.test_presence_final_audience import _promote

    registry, source, _pending = _team_turn(room, monkeypatch)
    child = _promote(room.data, monkeypatch, source, registry)
    assert final_route(room.data, child["metadata"], child["id"])[0] == UNKNOWN
    run = run_root(room, monkeypatch, [finish(answer=FINAL), finish(answer=FINAL)],
                   metadata=json.loads(json.dumps(child["metadata"])), root_id=child["id"])
    write_task_result(room.data, source["id"], "completed", metadata=source["metadata"], result="Done.")
    assert final_route(room.data, child["metadata"], child["id"])[0] == ROUTELESS
    assert len(run.inputs) == 1 and MARK not in json.dumps(run.inputs, ensure_ascii=False)
    assert [event.get("type") for event in run.pending].count("presence_result") == 1 and run.sent == []


# --- the records-only focus and the refusal wording stay their own ----------------------------------------------


def test_a_refused_routeless_final_is_not_promised_posting(room, monkeypatch):
    """Where no door is declared the reminder never comes; a Safety refusal's facts say no connector reads the final."""
    def check(name, subject, **_kw):
        return (name != "place_reply" or subject["text"] != FINAL), "Reconsider the wording."

    monkeypatch.setattr("ouroboros.safety.check_safety", check)
    presence = {**carrier(room), "transport_skill": "doorless"}
    run = run_root(room, monkeypatch, [finish(answer=FINAL), finish(silent=True, answer=NOTE)], metadata={
        "presence": presence, "delegation_role": "root", "schedule_occurrence": dict(SCHEDULED)})
    told = json.dumps(run.inputs[-1], ensure_ascii=False)
    assert "[FOCUS_REPLY_REFUSED]" in told and "No connector reads this run's final" in told
    assert "it started from a schedule" in told and "is posted as a new message" not in told
    assert MARK not in told and run.stored["metadata"]["presence_outcome"] == "silent"


def test_a_focus_turn_keeps_its_own_record_note_and_no_routeless_fact(room, monkeypatch):
    """The records-only turn is routed: its one return is the record note, never this fact."""
    open_door(monkeypatch)
    room.h.script = [record_send(text=ACK), finish(answer=FINAL), finish(answer=FINAL)]
    view, stored = settle(room, turn(room))
    rounds = json.dumps(room.h.inputs, ensure_ascii=False)
    assert "[FOCUS_FINISH_IS_A_RECORD]" in rounds and MARK not in rounds and room.transport.texts() == [ACK]


# --- review round 2: the note's own durability, rails after a restart, explicit profiles, gestures, state ------


def _crash_after_note_save(monkeypatch):
    """The worker dies right after the note's own ``ready`` save landed: before ``_no_tool_final_answer`` returns, so
    before the loop's next ``save_ready`` or any request."""
    from ouroboros import loop_delivery

    real = loop_delivery.observe_completion_hold

    def dying(tools, ctx, origin, reason, subreason=""):
        if reason == "routeless_final":
            raise Crash()
        return real(tools, ctx, origin, reason, subreason)

    monkeypatch.setattr(loop_delivery, "observe_completion_hold", dying)
    return lambda: monkeypatch.setattr(loop_delivery, "observe_completion_hold", real)


def _noted_then_dead(room, monkeypatch, root_id, setup=None):
    from ouroboros.working_checkpoint import read_working

    presence = carrier(room)
    healed = _crash_after_note_save(monkeypatch)
    first = run_root(room, monkeypatch, [{"content": FINAL, "tool_calls": []}, Crash], presence=presence, root_id=root_id,
                     setup=setup)
    healed()
    assert first.crashed and len(first.inputs) == 1, "the worker died before any request after the note"
    _bytes, saved = read_working(room.data, root_id, 1)
    assert saved["working"]["boundary"] == "ready" and saved["trace"][NO_SEND_KEY] == {"sends": "none", "noted": True}
    assert [row["content"] for row in saved["messages"] if MARK in str(row.get("content"))][0].startswith(MARK)
    assert saved["delivery_candidate"]["finalization_control"] == "routeless_final_noted"
    return first


@pytest.mark.parametrize("save", ["raises", "writes_nothing"])
def test_a_note_no_durable_state_holds_is_not_shown(room, monkeypatch, save):
    """A failed or empty save of the note shows nothing: no note, no hold, no extra round, the ordinary final."""
    from ouroboros import working_checkpoint

    real = working_checkpoint.save_round

    def unsaved(limit_ctx, boundary):
        if boundary == "ready" and MARK in json.dumps(limit_ctx.messages[-1:], ensure_ascii=False):
            if save == "raises":
                raise OSError("disk full")
            return False
        return real(limit_ctx, boundary)

    monkeypatch.setattr(working_checkpoint, "save_round", unsaved)
    run = run_root(room, monkeypatch, [{"content": FINAL, "tool_calls": []}, finish(answer=FINAL)])
    assert len(run.inputs) == 1 and MARK not in json.dumps(run.inputs, ensure_ascii=False)
    assert "noted" not in run.trace[NO_SEND_KEY] and run.trace["delivery_candidate"]["finalization_control"] == "candidate"
    assert run.stored["result"] == FINAL and run.stored["metadata"]["presence_outcome"] == "message" and run.sent == []


@pytest.mark.parametrize("identity", ["same", "new"])
def test_a_death_right_after_the_note_save_restores_the_note_first_and_once(room, monkeypatch, identity):
    first = _noted_then_dead(room, monkeypatch, "atomic-root")
    retry_id = "atomic-root" if identity == "same" else "atomic-retry"
    second = run_root(room, monkeypatch, [finish(silent=True, answer=NOTE), finish(answer=FINAL)], metadata=first.meta,
                      root_id=retry_id, attempt=2, recovery=_recover(room, "atomic-root"))
    assert len(second.inputs) == 1, "the restored note is read by a request; no saved final is accepted first"
    assert json.dumps(second.inputs[0], ensure_ascii=False).count(MARK) == 1
    assert second.stored["metadata"]["presence_outcome"] == "silent" and second.sent == []


@pytest.mark.parametrize("identity", ["same", "new"])
@pytest.mark.parametrize("meanwhile", ["money", "deadline", "wrap_up", "explicit_last_fit", "explicit_exhausted",
                                       "explicit_unpriced", "explicit_no_balances", "explicit_second_unknown"])
def test_a_restored_note_meets_the_rails_first_and_keeps_the_ordinary_final(room, monkeypatch, meanwhile, identity):
    """While the worker was down the money ran out, the deadline came inside its reserve, or the supervisor asked to
    wrap up: the restored round is never requested and no wrap-up or pause is bought; the answer finishes as it was."""
    from datetime import datetime, timedelta, timezone

    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

    explicit = meanwhile.startswith("explicit_")
    prices = _prices(monkeypatch) if explicit else None
    if explicit:  # a funded explicit profile showed the note; the restart re-reads the same authority
        _restore_explicit(monkeypatch)
    first = _noted_then_dead(room, monkeypatch, "railed-root", setup=_explicit_profile if explicit else None)
    retry_id = "railed-root" if identity == "same" else "railed-retry"
    meta = json.loads(json.dumps(first.meta))
    if meanwhile == "money":
        monkeypatch.setattr("ouroboros.loop_budget._next_round_pays", lambda _ctx: True)
        monkeypatch.setattr("ouroboros.loop_budget._wrapup_global_remaining", lambda: 0.0)
    if meanwhile == "deadline":
        meta["deadline_at"] = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    if explicit:  # meanwhile the spend moved: the note's own request no longer fits, or cannot be priced
        prices.update({"explicit_last_fit": {"two": False}, "explicit_exhausted": {"fits": False},
                       "explicit_unpriced": {"fail": True}, "explicit_no_balances": {"wallet": None},
                       "explicit_second_unknown": {"two": None}}[meanwhile])
    if meanwhile == "wrap_up":
        write_owner_message(room.data, "Finalization grace: wrap up now.", task_id=retry_id, msg_id="grace",
                            kind=KIND_FINALIZE_NOW)
    second = run_root(room, monkeypatch, [finish(answer="A paid wrap-up."), finish(answer="A paid wrap-up.")],
                      metadata=meta, root_id=retry_id, attempt=2, recovery=_recover(room, "railed-root"))
    assert second.inputs == [], "no request: neither the note's round nor a forced wrap-up"
    assert second.stored["result"] == FINAL and second.stored["metadata"]["presence_outcome"] == "message"
    assert "forced_finalization" not in second.trace and not second.stored.get("budget_pause")
    assert second.usage.get("terminal_origin") == "model_final" and second.trace[NO_SEND_KEY]["withdrawn"] is True
    assert second.sent == []


def test_a_deadline_landing_during_the_notes_round_keeps_the_ordinary_final_live(room, monkeypatch):
    """No restart: the note is shown, and before its round dispatches the deadline arrives inside the reserve."""
    from datetime import datetime, timedelta, timezone

    from ouroboros import loop_delivery

    real = loop_delivery.observe_completion_hold
    state = {}

    def deadline_lands(tools, ctx, origin, reason, subreason=""):
        real(tools, ctx, origin, reason, subreason)
        if reason == "routeless_final":
            state["ctx"] = tools._ctx
            tools._ctx.task_metadata["deadline_at"] = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()

    monkeypatch.setattr(loop_delivery, "observe_completion_hold", deadline_lands)
    run = run_root(room, monkeypatch, [{"content": FINAL, "tool_calls": []}, finish(answer="A paid wrap-up.")])
    assert len(run.inputs) == 1 and run.stored["result"] == FINAL and "forced_finalization" not in run.trace
    assert run.trace[NO_SEND_KEY] == {"sends": "none", "noted": True, "withdrawn": True}


def _explicit_profile(ctx, authority="explicit"):
    """An active in-task ceiling under an explicit ``cost_hard_stop_pct`` (or a producer allowance), as resolved."""
    from ouroboros import task_pacing

    ctx._cost_ceiling = task_pacing.CostCeiling(state=task_pacing.COST_CEILING_ACTIVE, ceiling_usd=40.0,
                                                root_cap_usd=50.0, basis="test profile")
    ctx._cost_stop_policy = {"authority": getattr(task_pacing, f"COST_STOP_{authority.upper()}"), "profile": {},
                             "root_cap_usd": 50.0, "non_root_member": False, "root_ceiling_usd": 40.0,
                             "producer_allowance_lifted": False}


def _prices(monkeypatch, **initial):
    """The price oracle behind the budget owner's exact probe (``loop_budget._exact_reservations``) and its proxy:
    ``fits``/``two`` answer one and two reservations, ``fail`` makes the request unpriceable, ``wallet`` is the known
    global remainder (``None``: unknown). Every exact request it prices is kept in ``seen``. The comparisons, last-fit
    relaxation and rails that read it are the real ones."""
    from ouroboros import task_pacing

    prices = {"fits": True, "two": True, "fail": False, "wallet": 10.0, "seen": [], **initial}

    def request(*, messages, **_kwargs):
        if prices["fail"]:
            raise RuntimeError("no price for this route")
        prices["seen"].append(json.loads(json.dumps(messages)))
        return {"priced_rows": len(messages)}

    def reservation_fits(*, reservation_count=1, **_kwargs):
        return prices["fits"] if reservation_count == 1 else prices["two"]

    monkeypatch.setattr(task_pacing, "prospective_wrapup_attempt_request", request)
    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", reservation_fits)
    monkeypatch.setattr("ouroboros.loop_budget._wrapup_global_remaining", lambda: prices["wallet"])
    return prices


def _restore_explicit(monkeypatch, relaxed=False):
    from ouroboros import task_pacing

    def restored(ctx, _saved):
        _explicit_profile(ctx)
        ctx._budget_resume_last_fit_relaxed = relaxed  # an owner Resume's relaxed last fit, as the restart reads it
        return ctx._cost_ceiling

    monkeypatch.setattr(task_pacing, "restore_cost_ceiling", restored)


@pytest.mark.parametrize("case", ["funded", "funded_relaxed_last_fit", "last_fit", "exhausted", "unknown", "unpriced",
                                  "no_balances", "second_unknown", "second_unknown_relaxed"])
def test_an_explicit_profile_prices_the_notes_own_request(room, monkeypatch, case):
    """Under an active explicit ``cost_hard_stop_pct`` profile the note's round is priced as the exact request it
    would send (this transcript, the answer, the note) by the same reservation logic its own rail uses: funded, it is
    bought (an owner Resume's relaxed last-fit too); exhausted, past last fit, unknown or unpriceable, the final stays
    as it was, without a pause, a forced prompt or a reservation."""
    prices = _prices(monkeypatch, **{"funded": {}, "funded_relaxed_last_fit": {"two": False}, "last_fit": {"two": False},
                                     "exhausted": {"fits": False}, "unknown": {"fits": None},
                                     "unpriced": {"fail": True}, "no_balances": {"wallet": None},
                                     "second_unknown": {"two": None}, "second_unknown_relaxed": {"two": None}}[case])

    def profile(ctx):
        _explicit_profile(ctx)
        ctx._budget_resume_last_fit_relaxed = "relaxed" in case

    run = run_root(room, monkeypatch, [{"content": FINAL, "tool_calls": []}, finish(answer=FINAL)], setup=profile)
    bought = case.startswith("funded")
    assert len(run.inputs) == (2 if bought else 1) and (MARK in json.dumps(run.inputs, ensure_ascii=False)) is bought
    assert bool(prices["seen"]) is (case not in {"unpriced", "no_balances"}), "nothing is priced without balances"
    if prices["seen"]:
        priced = prices["seen"][0]
        assert priced[-2] == {"role": "assistant", "content": FINAL} and priced[-1]["content"].startswith(MARK), (
            "the probe priced the round's own request, note included")
        if bought:
            assert priced[-1]["content"] == notes(run)[0], "the request priced is the request sent"
    assert "forced_finalization" not in run.trace and not run.stored.get("budget_pause")
    assert run.stored["result"] == FINAL and run.sent == []


@pytest.mark.parametrize("identity", ["same", "new"])
@pytest.mark.parametrize("relaxed", [False, True], ids=["funded", "relaxed_last_fit"])
def test_a_funded_explicit_profile_restores_the_note_and_requests_it(room, monkeypatch, identity, relaxed):
    """Funded when the note was saved; after the restart either still funded, or past its last fit under an owner
    Resume's relaxation, which admits the one affordable call: the restored note is requested once."""
    prices = _prices(monkeypatch)
    _restore_explicit(monkeypatch, relaxed=relaxed)
    first = _noted_then_dead(room, monkeypatch, "explicit-root", setup=_explicit_profile)
    prices["two"] = not relaxed
    retry_id = "explicit-root" if identity == "same" else "explicit-retry"
    second = run_root(room, monkeypatch, [finish(silent=True, answer=NOTE), finish(answer=FINAL)], metadata=first.meta,
                      root_id=retry_id, attempt=2, recovery=_recover(room, "explicit-root"))
    assert len(second.inputs) == 1 and json.dumps(second.inputs[0], ensure_ascii=False).count(MARK) == 1
    priced = prices["seen"][-1]  # the restored transcript: the note, then the host's continuation notice
    assert [str(row.get("content")).startswith(MARK) for row in priced].count(True) == 1, "priced with its note"
    assert "continued from its working checkpoint" in priced[-1]["content"]
    assert second.stored["metadata"]["presence_outcome"] == "silent" and second.sent == []


@pytest.mark.parametrize("authority", ["explicit", "producer"])
def test_a_producer_allowance_far_from_its_ceiling_gets_the_round_and_an_explicit_one_is_priced(
        room, monkeypatch, authority):
    prices = _prices(monkeypatch)
    run = run_root(room, monkeypatch, [{"content": FINAL, "tool_calls": []}, finish(answer=FINAL)],
                   setup=lambda ctx: _explicit_profile(ctx, authority))
    assert len(run.inputs) == 2 and len(notes(run)) == 1
    assert bool(prices["seen"]) is (authority == "explicit"), "only the explicit profile's rail prices reservations"


def test_a_gesture_leaves_the_lineage_silent_and_is_named_beside_the_fact(room, monkeypatch):
    react = call(RECORDS_REACT, {"conversation_id": "r", "message_id": "chat:a:1", "emoji": "eyes"}, "react")
    run = run_root(room, monkeypatch, [react, finish(answer=FINAL), finish(answer=FINAL)])
    assert [(row["kind"], row["message_id"]) for row in run.sent] == [("reaction", "chat:a:1")]
    assert run.trace[NO_SEND_KEY]["sends"] == "none" and len(run.inputs) == 3
    [note] = notes(run)
    assert f"Gestures you made there, which are not words: {RECORDS_REACT} on message chat:a:1." in note


def test_a_local_room_message_leaves_the_lineage_silent(room, monkeypatch):
    """``send_user_message`` really queues its local-room frame, never the place: the reminder still comes."""
    local = call("send_user_message", {"text": "Local note for my human."}, "local")
    run = run_root(room, monkeypatch, [local, finish(answer=FINAL), finish(answer=FINAL)],
                   setup=lambda ctx: setattr(ctx, "current_chat_id", 1))  # the room the root was admitted in
    receipt = next(row for row in run.trace["tool_calls"] if row["tool"] == "send_user_message")
    assert not receipt.get("is_error"), receipt
    assert any(event.get("type") == "send_message" and event.get("text") == "Local note for my human."
               for event in run.events)
    assert run.trace[NO_SEND_KEY]["sends"] == "none" and len(notes(run)) == 1 and run.sent == []


@pytest.mark.parametrize("stored", ["absent", "not_a_mapping", "unknown_value", "null_value"])
def test_a_restored_legacy_or_corrupt_no_send_fact_is_unknown(room, monkeypatch, stored):
    """A working state from before this fact, or one whose fact is malformed, restores as unknown: no hint."""
    from ouroboros.working_checkpoint import checkpoint_path, read_working

    presence = carrier(room)
    first = run_root(room, monkeypatch, [call("list_files", {"path": "."}, "read"), Crash], presence=presence,
                     root_id="legacy-root")
    assert first.crashed
    _bytes, state = read_working(room.data, "legacy-root", 1)
    assert state["trace"][NO_SEND_KEY] == {"sends": "none"}
    if stored == "absent":
        state["trace"].pop(NO_SEND_KEY)
    else:
        state["trace"][NO_SEND_KEY] = {"not_a_mapping": "none", "unknown_value": {"sends": "maybe"},
                                       "null_value": {"sends": None}}[stored]
    checkpoint_path(room.data, "legacy-root", 1).write_text(json.dumps(state), encoding="utf-8")
    second = run_root(room, monkeypatch, [finish(answer=FINAL), finish(answer=FINAL)], metadata=first.meta,
                      root_id="legacy-root", attempt=2, recovery=_recover(room, "legacy-root"))
    assert len(second.inputs) == 1 and MARK not in json.dumps(second.inputs, ensure_ascii=False)
    assert second.stored["result"] == FINAL and second.sent == []


@pytest.mark.parametrize("delayed_guard", [False, True], ids=["delivered", "timeout-before-guard"])
def test_a_new_message_generation_after_pre_effect_is_not_no_send(room, monkeypatch, delayed_guard):
    """A published read is replaced before dispatch. Its actual message generation may speak even when its
    guard has not run by timeout; pre-effect's old declaration cannot justify a second model round."""
    import threading

    from ouroboros import loop_tool_execution, place_guard, working_checkpoint

    release = threading.Event()
    saved_states = []
    original_save = working_checkpoint.save_round
    original_guard = place_guard.dispatch_refusal
    original_timeout = loop_tool_execution._get_tool_timeout

    def publish(effect, generation):
        manifest = SimpleNamespace(raw_extra={"tools": [
            {"name": "send", "effect": effect, "target": ["conversation_id"],
             "conversation": "conversation_id", "thread": "thread_id"}]})
        bundle = registry_state._ExtensionRegistrations(
            generation_digest=generation, loaded_skill=SimpleNamespace(manifest=manifest))
        monkeypatch.setitem(registry_state._tools, RECORDS_SEND,
                            {"skill": "bridge", "extension_generation": generation})
        monkeypatch.setitem(registry_state._extensions, "bridge", bundle)

    def saved_before_replacement(limit_ctx, boundary):
        saved = original_save(limit_ctx, boundary)
        batch = limit_ctx.messages[-1].get("tool_calls") or []
        if boundary == "pre_effect" and any(row.get("function", {}).get("name") == RECORDS_SEND for row in batch):
            saved_states.append(dict(limit_ctx.llm_trace[NO_SEND_KEY]))
            publish("message", "g2")
        return saved

    def guard(*args, **kwargs):
        if delayed_guard:
            release.wait(20)
        return original_guard(*args, **kwargs)

    monkeypatch.setattr(working_checkpoint, "save_round", saved_before_replacement)
    monkeypatch.setattr(place_guard, "dispatch_refusal", guard)
    monkeypatch.setattr(loop_tool_execution, "_get_tool_timeout", lambda tools, name, args=None:
                        1 if delayed_guard and name == RECORDS_SEND else original_timeout(tools, name, args))
    try:
        run = run_root(room, monkeypatch, [record_send(text=ACK), finish(answer=FINAL), finish(answer=FINAL)],
                       setup=lambda _ctx: publish("read", "g1"))
    finally:
        release.set()
    assert saved_states == [{"sends": "none"}], "the first checkpoint really saw the earlier read generation"
    result = next(row for row in run.trace["tool_calls"] if row["tool"] == RECORDS_SEND)
    assert bool(result.get("is_error")) is delayed_guard, result
    if not delayed_guard:
        assert [row["text"] for row in run.sent] == [ACK], "the real transport already sent the acknowledgement"
    assert len(run.inputs) == 2 and not notes(run), "the old read must not buy a reminder after possible speech"
    assert run.trace[NO_SEND_KEY]["sends"] != "none"


def test_dispatch_observation_preserves_its_bound_message_generation(room, monkeypatch):
    from ouroboros.place_reply import note_dispatch
    from ouroboros.tools import extension_dispatch

    presence = carrier(room)
    ctx = SimpleNamespace(task_metadata={"presence": presence}, task_id="bound-root", drive_root=room.data)
    trace = {NO_SEND_KEY: {"sends": "none"}}
    open_door(monkeypatch)
    row = {"name": "send", "effect": "message", "target": ["conversation_id"], "conversation": "conversation_id"}
    descriptor = {"skill": "bridge", "extension_generation": "g2", registry_state.REVIEWED_DECLARATION:
                  {"extension_generation": "g2", "tool": row}}
    reads = []

    def resolve(context, name):
        reads.append((context, name))
        return descriptor, False

    monkeypatch.setattr(extension_dispatch, "_extension_dispatch_candidate", resolve)
    with extension_dispatch.extension_call_scope():
        extension_dispatch.timeout_dispatch_candidate(ctx, RECORDS_SEND)
        # The published declaration changes again after timeout binding. The bound handler remains a message.
        registry_state._extensions["bridge"].loaded_skill.manifest.raw_extra["tools"][0]["effect"] = "read"
        note_dispatch(ctx, trace, record_send(text=ACK)["tool_calls"][0], bound=True)
        assert extension_dispatch.bound_dispatch_candidate(ctx, RECORDS_SEND) == (descriptor, False)
    assert reads == [(ctx, RECORDS_SEND)], "observation must not consume or re-resolve the dispatch binding"
    assert trace[NO_SEND_KEY]["sends"] == "possible"


def test_interrupted_call_is_not_exonerated_by_a_new_read_declaration(room, monkeypatch):
    from ouroboros import working_checkpoint

    real_save, real_register = working_checkpoint.save_round, ToolRegistry.register

    def publish(effect, generation):
        manifest = SimpleNamespace(raw_extra={"tools": [
            {"name": "send", "effect": effect, "target": ["conversation_id"], "conversation": "conversation_id"}]})
        monkeypatch.setitem(registry_state._extensions, "bridge", registry_state._ExtensionRegistrations(
            generation_digest=generation, loaded_skill=SimpleNamespace(manifest=manifest)))
        monkeypatch.setitem(registry_state._tools, RECORDS_SEND,
                            {"skill": "bridge", "extension_generation": generation})

    def save(limit_ctx, boundary):
        saved = real_save(limit_ctx, boundary)
        if boundary == "pre_effect":
            publish("message", "g2")
        return saved

    def register(self, entry):
        if entry.name == RECORDS_SEND:
            def spoke_then_died(ctx, **args):
                room.transport.tool(ctx, **args)
                raise Crash()
            entry = ToolEntry(name=entry.name, schema=entry.schema, handler=spoke_then_died,
                              timeout_sec=30, safety_policy="skip")
        return real_register(self, entry)

    presence = carrier(room)
    monkeypatch.setattr(working_checkpoint, "save_round", save)
    monkeypatch.setattr(ToolRegistry, "register", register)
    first = run_root(room, monkeypatch, [record_send(text=ACK)], presence=presence, root_id="drift-crash",
                     setup=lambda _ctx: publish("read", "g1"))
    assert first.crashed and room.transport.provider[-1]["text"] == ACK
    _bytes, saved = working_checkpoint.read_working(room.data, "drift-crash", 1)
    assert saved["working"]["boundary"] == "pre_effect" and saved["trace"][NO_SEND_KEY] == {"sends": "none"}
    monkeypatch.setattr(working_checkpoint, "save_round", real_save)
    monkeypatch.setattr(ToolRegistry, "register", real_register)

    def finish_after_restore(_messages):
        publish("message", "g4")
        return finish(answer=FINAL)

    second = run_root(room, monkeypatch, [finish_after_restore, finish(answer=FINAL)], metadata=first.meta,
                      root_id="drift-crash", attempt=2, recovery=_recover(room, "drift-crash"),
                      setup=lambda _ctx: publish("read", "g3"))
    assert second.trace[NO_SEND_KEY]["sends"] == "possible"
    assert len(second.inputs) == 1 and not notes(second) and second.sent == []
