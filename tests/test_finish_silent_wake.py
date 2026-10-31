"""The quiet door: finish_task(silent=true) on a wake or a schedule-born root.

The scenarios run the real loop (scripted model responses) and the real terminal
pipeline: a clean silent wake publishes no final row and owes no outbox delivery,
while its words, activity, sends and cost stay on the record. Owner words admitted after the last live reply, other turns, and any
outcome but a clean one keep the ordinary final.
"""

import json
import time
import queue
from types import SimpleNamespace

import pytest

import ouroboros.agent_task_pipeline as pipeline
import ouroboros.loop as loop
from ouroboros.consciousness_authority import CONSCIOUSNESS_CATEGORY, CONSCIOUSNESS_TASK_CATEGORY
from ouroboros.owner_mailbox import write_owner_message
from ouroboros.task_results import load_task_result
from ouroboros.task_finalization import quiet_finish
from ouroboros.tools.registry import ToolRegistry
from tests._wake_summary_shared import seed_wake

TASK_ID = "wake-task"
WAKE = {"initiator": "consciousness", "usage_category": CONSCIOUSNESS_CATEGORY, "wake_reason": "heartbeat", "wake_cause": "own_clock"}
SCHEDULE = {"schedule_occurrence": {"schedule_id": "daily-check", "token": "occurrence-1"}}


def finish(answer=None, **kw):
    arguments = {"action": "finish", **kw}
    if answer is not None:
        arguments["answer"] = answer
    return call("finish_task", arguments)


def call(name, arguments, call_id=None):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id or name, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]}


@pytest.fixture
def wake(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.is_direct_chat = True
    registry._ctx.current_chat_id = 1
    registry._ctx.task_metadata = dict(WAKE)
    calls = []

    def run(responses, *, live=True, metadata=None, bus=None, messages=None):
        registry = state.registry  # a cold resume swaps in a fresh registry (a new process)
        if metadata is not None:
            registry._ctx.task_metadata = dict(metadata)
        iterator = iter(responses)

        def respond(_llm, messages, *_a, **_kw):
            calls.append(json.loads(json.dumps(messages)))
            response = next(iterator)
            return (response() if callable(response) else response), 0.0
        monkeypatch.setattr(loop, "call_llm_with_retry", respond)
        bus = bus if bus is not None else (queue.Queue() if live else None)
        messages = [{"role": "user", "content": "Wake: no one has asked for a task."}] if messages is None else messages
        text, usage, trace = loop.run_llm_loop(
            messages, registry,
            SimpleNamespace(default_model=lambda: "test-model"), tmp_path / "logs",
            lambda *_a, **_kw: None, queue.Queue(), task_id=TASK_ID, event_queue=bus, drive_root=tmp_path)
        return text, usage, trace, bus

    def owner_writes(text, then, msg_id):
        """A scripted response during whose generation the owner writes to the running wake."""
        def response():
            write_owner_message(tmp_path, text, TASK_ID, msg_id=msg_id)
            return then
        return response
    state = SimpleNamespace(registry=registry, ctx=registry._ctx, calls=calls, run=run, root=tmp_path,
                            owner_writes=owner_writes)
    return state


def emit(wake, text, usage, trace, *, bus=None, metadata=None, pending=None, task=None):
    """The production terminal pipeline after the agent's empty-final resolution."""
    from ouroboros.agent import _resolve_intentional_empty

    pending = [] if pending is None else pending
    replaced, _, text = _resolve_intentional_empty(wake.ctx, usage, text)
    if replaced:
        usage["terminal_origin"] = "host_notice"
    if task is None:
        task = {"id": TASK_ID, "type": "task", "chat_id": 1, "text": "wake", "_is_direct_chat": True,
                "metadata": dict(metadata or WAKE)}
    pipeline.emit_task_results(SimpleNamespace(drive_root=wake.root, repo_dir=wake.root), None, None,
                               pending, task, text, usage, trace, 0.0, wake.root / "logs",
                               ctx=wake.ctx, event_queue=bus)
    return pending, load_task_result(wake.root, TASK_ID) or {}


def sends(events):
    return [e for e in events if isinstance(e, dict) and e.get("type") == "send_message"]


def drained(bus):
    rows = []
    while bus is not None and not bus.empty():
        rows.append(bus.get_nowait())
    return rows


def owed_rows(root):
    from supervisor.terminal_delivery import pending_deliveries

    return [row for row in pending_deliveries(root, strict=True) if row.get("task_id") == TASK_ID]


@pytest.fixture(autouse=True)
def _no_post_task_model(monkeypatch):
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)


# --- clean silence: nothing published, everything retained ----------------------------

def test_a_quiet_run_with_open_delegated_custody_is_never_silenced(wake):
    """#1583: silence hides only a clean run. An own delegated run the current audit still
    holds is a debt the host states (its own notice, never the quiet words), beside the
    run's own cause: the stored reason stays the loop's, never the custody code."""
    from ouroboros import delegate_custody_current as current, store
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES

    with store.open_store(wake.root, write=True) as txn:
        txn.put_custody("leaf", {"run_id": "leaf", "task_id": TASK_ID, "source": "delegated_subagent"},
                        state=current.HELD)
    text, usage, trace, bus = wake.run([finish(INTERNAL, silent=True)])
    assert usage.get("wake_silent") is True
    pending, stored = emit(wake, text, usage, trace, bus=bus)
    shown = sends(pending)
    assert len(shown) == 1 and shown[0]["terminal_origin"] == "host_notice"
    assert shown[0]["text"].startswith("⚠️ This run chose to finish without a message, but it ended ok")
    assert TASK_CAUSE_PHRASES["delegated_custody_unreconciled"] in shown[0]["text"]
    assert INTERNAL not in _spoken(wake, pending, bus)
    assert stored["result"] == INTERNAL and stored["delegated_runs_unreconciled"] == ["leaf"]
    assert stored["reason_code"] != "delegated_custody_unreconciled"
    assert stored["outcome_axes"]["execution"]["status"] == "ok"


@pytest.mark.parametrize("answer", [None, "", "Read the journal; nothing changed my view."])
@pytest.mark.parametrize("cause", ["own_clock", "focus"])
def test_clean_silent_wake_publishes_nothing_and_keeps_its_record(wake, answer, cause):
    metadata = {**WAKE, "wake_cause": cause}
    if cause == "focus":  # the host stamps a focus with its place and its event
        metadata.update(focus={"place_id": "place-1", "conversation_key": "slack:c1"},
                        presence={"event": {"conversation_key": "slack:c1"}})
    text, usage, trace, bus = wake.run([finish(answer, silent=True)], metadata=metadata)
    assert len(wake.calls) == 1 and usage.get("wake_silent") is True
    assert text == (answer or "")
    pending, stored = emit(wake, text, usage, trace, bus=bus, metadata=metadata)
    assert sends(pending) == [] and sends(drained(bus)) == [], "no final row, live or buffered"
    assert owed_rows(wake.root) == [], "boot replay owes nothing"
    assert stored["status"] == "completed" and stored.get("wake_silent") is True
    assert quiet_finish(stored) and "owner_message_sent" not in stored
    assert stored["outcome_axes"]["execution"]["status"] == "ok"
    assert stored["result"] == (answer or ""), "internal words are retained, not erased"
    assert "​" not in json.dumps(stored, ensure_ascii=False)
    done = [e for e in pending if e.get("type") == "task_done"]
    metrics = [e for e in pending if e.get("type") == "task_metrics"]
    assert done and done[0]["status"] == "completed" and "cost_accounting_status" in done[0]
    assert metrics and "cost_accounting_status" in metrics[0], "cost stays on the record"


def test_internal_work_and_earlier_sends_remain(wake):
    wake.registry.override_handler("chat_history", lambda _ctx, **kw: "read result")
    note = call("send_user_message", {"text": "Found the regression source."})
    text, usage, trace, bus = wake.run([call("chat_history", {}), note, finish(silent=True)])
    assert usage.get("wake_silent") is True
    assert [c["tool"] for c in trace["tool_calls"]][:2] == ["chat_history", "send_user_message"]
    live = drained(bus)
    assert [e["text"] for e in sends(live)] == ["Found the regression source."], "the earlier live word stays"
    pending, stored = emit(wake, text, usage, trace, bus=bus)
    assert sends(pending) == [] and sends(drained(bus)) == []
    assert stored.get("wake_silent") is True
    assert usage["owner_message_sent"] is True and stored["owner_message_sent"] is True
    assert not quiet_finish(stored)


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("tool", ["send_links", "send_file"])
def test_send_family_prevents_a_quiet_finish(wake, live, tool):
    path = wake.root / "note.txt"
    path.write_text("A source for you.", encoding="utf-8")
    arguments = {"file_path": str(path)} if tool == "send_file" else {
        "title": "A source for you", "links": [{"label": "Source", "url": "https://example.org/"}]}
    text, usage, trace, bus = wake.run([call(tool, arguments), finish("Private note.", silent=True)], live=live)
    events = drained(bus) if live else wake.ctx.pending_events
    assert any(row.get("type") == ("send_document" if tool == "send_file" else "send_links") for row in events)
    _, stored = emit(wake, text, usage, trace, bus=bus)
    assert stored["wake_silent"] is True and stored.get("owner_message_sent") is True
    assert not quiet_finish(stored)


@pytest.mark.parametrize("cause", ["own_clock", "focus"])
@pytest.mark.parametrize("action,silent", [("finish", True), ("finish", False), ("stop", False)])
def test_wake_summary_titles_the_stored_wake_without_changing_its_words(wake, cause, action, silent):
    metadata = {**WAKE, "wake_cause": cause}
    summary = "Проверил состояние системы; вмешательство не требуется."
    text, usage, trace, bus = wake.run([finish(
        "Whole note.\n\nSecond paragraph stays as written.", action=action,
        silent=silent, rationale="The remaining probe can wait.", summary="  Проверил\n состояние\tсистемы; вмешательство не требуется.  ")], metadata=metadata)
    assert usage["wake_summary"] == summary
    seed_wake(wake.root, task_id=TASK_ID)
    _, stored = emit(wake, text, usage, trace, bus=bus, metadata=metadata)
    assert stored["wake_summary"] == stored["suggested_name"] == summary
    assert "summary" not in stored, "generic summary would override result in project dialogue"
    assert stored["result"] == "Whole note.\n\nSecond paragraph stays as written."


@pytest.mark.parametrize("summary", [None, "", " \n\t"])
def test_absent_summary_keeps_the_existing_title(wake, summary):
    text, usage, trace, bus = wake.run([finish(silent=True, **({"summary": summary} if summary is not None else {}))])
    before = seed_wake(wake.root, task_id=TASK_ID)
    _, stored = emit(wake, text, usage, trace, bus=bus)
    assert "wake_summary" not in usage and "wake_summary" not in stored
    assert stored["suggested_name"] == before["suggested_name"]


@pytest.mark.parametrize("arguments", [{}, {"summary": ""}, {"summary": None}])
def test_ordinary_finish_accepts_an_omitted_empty_or_null_summary(wake, arguments):
    from ouroboros.tools.control_runtime import _finish_task

    wake.ctx.task_metadata = {}
    result = _finish_task(wake.ctx, action="finish", answer="x", **arguments)
    assert "completion_requested" in result and "TOOL_ERROR" not in result
    assert wake.ctx._completion_request["answer"] == "x"
    assert "wake_summary" not in wake.ctx._completion_request


@pytest.mark.parametrize("cause,title", [("own_clock", "Own time"), ("focus", "Research room")])
@pytest.mark.parametrize("summary", ["", "S2"])
@pytest.mark.parametrize("reset_at_start", [False, True])
def test_retry_does_not_inherit_the_previous_attempts_summary(wake, cause, title, summary, reset_at_start):
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import patch_document

    metadata = {**WAKE, "wake_cause": cause, "focus": {"label": "Research room"}}
    seed_wake(wake.root, task_id=TASK_ID, status="running", task_attempt=1,
              metadata=metadata, suggested_name=title, title="")
    text, usage, trace, bus = wake.run([finish("First note", silent=True, summary="S1")], metadata=metadata)
    _, first = emit(wake, text, usage, trace, bus=bus, metadata=metadata)
    assert first["wake_summary"] == first["suggested_name"] == "S1"
    enriched = write_task_result(wake.root, TASK_ID, first["status"], cost_usd=0.1)
    assert enriched["wake_summary"] == enriched["suggested_name"] == "S1", "same-attempt enrichment keeps authorship"
    # Explicit reopen owns lifecycle; the merge writer must scope carried words itself.
    patch_document(wake.root, TASK_ID, lambda row: {**row, "status": "running"})
    if reset_at_start:
        running = write_task_result(wake.root, TASK_ID, "running", task_attempt=2)
        assert not running.get("wake_summary") and running["suggested_name"] == title
    text, usage, trace, bus = wake.run([finish("Second note", silent=True, summary=summary)], metadata=metadata)
    _, second = emit(wake, text, usage, trace, bus=bus, metadata=metadata)
    assert second.get("wake_summary") == (summary or None)
    assert second["suggested_name"] == (summary or title)
    assert second["result"] == "Second note" and not second.get("title") and "summary" not in second


@pytest.mark.parametrize("metadata,direct", [({}, True), ({**SCHEDULE, "presence": {"binding_id": "b1"}}, False),
    ({"initiator": "consciousness", "usage_category": CONSCIOUSNESS_TASK_CATEGORY}, False), (WAKE, False)])
@pytest.mark.parametrize("action", ["finish", "stop"])
def test_summary_refuses_runs_not_self_started_but_empty_summary_keeps_ordinary_completion(wake, metadata, direct, action):
    from ouroboros.tools.control_runtime import _finish_task

    wake.ctx.task_metadata, wake.ctx.is_direct_chat = metadata, direct
    wake.ctx._active_builtin_tool_result = None
    rejected = _finish_task(wake.ctx, action=action, answer="Answer", rationale="Unfinished", summary="A line")
    assert "COMPLETION_ARGUMENT: silent and summary are available only on a self-started run" in rejected
    assert wake.ctx._active_builtin_tool_result.code == "TOOL_ARG_ERROR"
    assert getattr(wake.ctx, "_completion_request", None) is None
    assert "completion_requested" in _finish_task(wake.ctx, action=action, answer="Answer", rationale="Unfinished", summary=" \n")
    assert "wake_summary" not in wake.ctx._completion_request


@pytest.mark.parametrize("action,silent", [("finish", True), ("finish", False), ("stop", False)])
def test_schedule_born_root_may_leave_one_sentence_for_its_human(wake, action, silent):
    """The same door as silence: a self-started run says what it did; its note stays its own."""
    from ouroboros.tools.control_runtime import _finish_task

    wake.ctx.task_metadata, wake.ctx.is_direct_chat = dict(SCHEDULE), False
    wake.ctx.task_id, wake.ctx.parent_task_id = TASK_ID, None
    wake.ctx._active_builtin_tool_result = None
    result = _finish_task(wake.ctx, action=action, answer="Internal check result.", rationale="The remaining probe can wait.",
                          summary="  Checked\n the disk:  30 GiB free. ", **({"silent": True} if silent else {}))
    assert "completion_requested" in result and "TOOL_ERROR" not in result
    request = wake.ctx._completion_request
    assert request["wake_summary"] == "Checked the disk: 30 GiB free."
    assert request.get("wake_silent") is (True if silent else None)
    assert request["answer"] == "Internal check result." and request["action"] == action


def test_finish_schema_offers_optional_summary():
    from ouroboros.tools.control import get_tools

    entry = next(entry for entry in get_tools() if entry.name == "finish_task")
    parameters = entry.schema["parameters"]
    assert parameters["properties"]["summary"]["type"] == "string"
    assert parameters["properties"]["summary"]["default"] == ""
    assert "summary" not in parameters["required"]
    assert "silent=true sends no message (any answer stays your internal note)" in entry.schema["description"]
    assert "no owner-facing line" not in entry.schema["description"]


def test_silent_terminal_ships_no_deferred_mid_turn_send_early(wake):
    """The post-task fallback scan would otherwise pick a buffered mid-turn send as 'the final'."""
    text, usage, trace, _ = wake.run([finish(silent=True)])
    buffered = {"type": "send_message", "chat_id": 1, "text": "deferred note", "task_id": TASK_ID,
                "system_type": None}
    bus = queue.Queue()
    pending, _stored = emit(wake, text, usage, trace, bus=bus, pending=[buffered])
    assert drained(bus) == [], "nothing goes live ahead of the buffered drain"
    assert sends(pending) == [buffered], "the buffered word still drains in its own order"


# --- owner words: an admitted row needs a live reply first ------------------------------

def test_unanswered_owner_words_refuse_silence_until_a_live_reply(wake):
    text, usage, trace, bus = wake.run([
        wake.owner_writes("Are you there?", call("chat_history", {}), "m1"),
        finish(silent=True),
        call("send_user_message", {"text": "Yes, here."}),
        finish(silent=True),
    ])
    reasons = [row["reason"] for row in trace.get("completion_refusals") or []]
    assert len(reasons) == 1 and reasons[0].startswith("owner_reply_owed")
    assert usage.get("wake_silent") is True and len(wake.calls) == 4


def test_deferred_or_failed_send_does_not_discharge_the_reply(wake):
    text, usage, trace, _ = wake.run([
        wake.owner_writes("Status?", call("chat_history", {}), "m1"),
        call("send_user_message", {"text": "Working on it."}),
        finish(silent=True),
        finish("Working on it; nothing else yet."),
    ], live=False)
    reasons = [row["reason"] for row in trace.get("completion_refusals") or []]
    assert reasons and reasons[0].startswith("owner_reply_owed")
    assert "wake_silent" not in usage and text == "Working on it; nothing else yet."
    pending, stored = emit(wake, text, usage, trace)
    assert [e["text"] for e in sends(pending) if not e.get("system_type")] == ["Working on it; nothing else yet."]
    assert "wake_silent" not in stored


def test_failed_live_put_does_not_discharge_the_reply(wake, monkeypatch):
    """A live owner-send whose queue put fails falls back to the buffer: no live reply was made."""
    class RefusingBus(queue.Queue):
        def put_nowait(self, item):
            if isinstance(item, dict) and item.get("type") == "send_message":
                raise OSError("worker queue refused the frame")
            return super().put_nowait(item)
    import ouroboros.tools.owner_delivery as delivery

    original, modes = delivery.deliver_owner_event, []
    monkeypatch.setattr(delivery, "deliver_owner_event",
                        lambda ctx, evt: modes.append(original(ctx, evt)) or modes[-1])
    text, usage, trace, _ = wake.run([
        wake.owner_writes("Status?", call("chat_history", {}), "m1"),
        call("send_user_message", {"text": "Working on it."}),
        finish(silent=True),
        finish("Working on it; nothing else yet."),
    ], bus=RefusingBus())
    assert modes == ["deferred"] and getattr(wake.ctx, "_owner_delivery_sticky_deferred", False) is True
    reasons = [row["reason"] for row in trace.get("completion_refusals") or []]
    assert reasons and reasons[0].startswith("owner_reply_owed")
    assert "wake_silent" not in usage and text == "Working on it; nothing else yet."
    assert not getattr(wake.ctx, "_owner_reply_revision", None)


def test_new_owner_words_after_the_reply_refuse_silence_again(wake):
    text, usage, trace, _ = wake.run([
        wake.owner_writes("Status?", call("chat_history", {}), "m1"),
        wake.owner_writes("And the tests?", call("send_user_message", {"text": "Building now."}), "m2"),
        finish(silent=True),
        finish("Tests pass."),
    ])
    reasons = [row["reason"] for row in trace.get("completion_refusals") or []]
    assert reasons and reasons[0].startswith("owner_reply_owed")
    assert "wake_silent" not in usage and text == "Tests pass."


def test_reply_already_sent_needs_no_ceremonial_second_final(wake):
    text, usage, trace, _ = wake.run([
        wake.owner_writes("Status?", call("chat_history", {}), "m1"),
        call("send_user_message", {"text": "All green."}),
        finish(silent=True),
    ])
    assert not trace.get("completion_refusals") and usage.get("wake_silent") is True


def test_owner_quiz_closure_counts_as_owner_words(wake):
    """A Main owner-actor quiz closure rides the owner corpus (with a reason); it is not swallowed."""
    from ouroboros.owner_mailbox import KIND_QUIZ_CLOSED

    def closed():
        write_owner_message(wake.root, "[Owner question closed without answer] quiz q1", TASK_ID,
                            msg_id="quiz_closed:q1", kind=KIND_QUIZ_CLOSED)
        return call("chat_history", {})
    _, usage, trace, _ = wake.run([closed, finish(silent=True), finish("Noted; dropping it.")])
    assert (trace.get("completion_refusals") or [{}])[0].get("reason", "").startswith("owner_reply_owed")
    assert "wake_silent" not in usage


# --- who may use the door ----------------------------------------------------------------

@pytest.mark.parametrize("origin", [{}, {
    "initiator": "consciousness", "usage_category": CONSCIOUSNESS_TASK_CATEGORY,
}], ids=["scheduled-root", "consciousness-followup-root"])
def test_schedule_born_root_may_finish_quietly(wake, origin):
    wake.ctx.is_direct_chat = False
    text, usage, trace, _ = wake.run([finish("Internal check result.", silent=True)],
                                      metadata={**origin, **SCHEDULE})
    assert usage.get("wake_silent") is True and text == "Internal check result."
    assert not trace.get("completion_refusals")


@pytest.mark.parametrize("extra,occurrence,parent", [
    ({"delegation_role": "subagent"}, SCHEDULE["schedule_occurrence"], None),
    ({}, SCHEDULE["schedule_occurrence"], "parent-task"),
    ({"root_task_id": "other-root"}, SCHEDULE["schedule_occurrence"], None),
    ({"presence": {"binding_id": "b1"}}, SCHEDULE["schedule_occurrence"], None),
    ({"presence": {}}, SCHEDULE["schedule_occurrence"], None),
    ({}, {"schedule_id": "daily-check"}, None),
    ({}, {"token": "occurrence-1"}, None),
    ({}, {"schedule_id": "daily-check", "token": " "}, None),
    ({}, "daily-check", None),
], ids=["delegated", "parent", "non-root", "presence", "empty-presence",
        "missing-token", "missing-schedule", "empty-token", "invalid-occurrence"])
def test_schedule_marker_does_not_bypass_quiet_authority(wake, extra, occurrence, parent):
    from ouroboros.tools.control_runtime import _finish_task

    wake.ctx.is_direct_chat = False
    wake.ctx.task_id = TASK_ID
    wake.ctx.parent_task_id = parent
    wake.ctx.task_metadata = {**extra, "schedule_occurrence": occurrence}
    assert "COMPLETION_ARGUMENT" in _finish_task(wake.ctx, action="finish", silent=True)
    assert "COMPLETION_ARGUMENT" in _finish_task(wake.ctx, action="finish", answer="Answer", summary="A line")
    assert getattr(wake.ctx, "_completion_request", None) is None
    assert "completion_requested" in _finish_task(wake.ctx, action="finish", answer="Ordinary answer.")


def test_presence_ceiling_alone_refuses_quiet_schedule_finish(wake):
    from ouroboros.tools.control_runtime import _finish_task

    wake.ctx.is_direct_chat = False
    wake.ctx.task_id = TASK_ID
    wake.ctx.task_metadata = dict(SCHEDULE)
    wake.ctx.task_contract = {"capability_ceiling": {}}
    assert "COMPLETION_ARGUMENT" in _finish_task(wake.ctx, action="finish", silent=True)
    wake.ctx.task_contract = {}
    assert "completion_requested" in _finish_task(wake.ctx, action="finish", silent=True)


def test_schedule_born_root_owes_a_reply_to_owner_words_mid_run(wake):
    wake.ctx.is_direct_chat = False
    _, usage, trace, _ = wake.run([
        wake.owner_writes("Anything new?", call("chat_history", {}), "scheduled-owner-message"),
        finish(silent=True),
        call("send_user_message", {"text": "Nothing new."}),
        finish(silent=True),
    ], metadata=SCHEDULE)
    reasons = [row["reason"] for row in trace.get("completion_refusals") or []]
    assert len(reasons) == 1 and reasons[0].startswith("owner_reply_owed")
    assert usage.get("wake_silent") is True and len(wake.calls) == 4


def test_quiet_project_schedule_through_delivery_and_notifications(wake, monkeypatch):
    import asyncio
    import shutil
    import subprocess
    from pathlib import Path

    from ouroboros.headless import prepare_terminal_task_files
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor.event_observation import ObservedEventQueue
    from supervisor.events import _handle_task_done
    from tests._dialogue import chat_rows
    from tests.test_task_done_durable_vs_transport import _mk_ctx
    from tests.test_telegram_notifications import _load, _Rec

    project = create_project(wake.root, "checks", name="Checks")
    bind_task_to_project(wake.root, TASK_ID, project["id"], project["chat_id"], origin={"absent": "system"})
    task = {"id": TASK_ID, "type": "task", "text": "Check for changes", "title": "Check for changes",
            "root_task_id": TASK_ID, "delegation_role": "root", "chat_id": project["chat_id"],
            "project_id": project["id"], "metadata": dict(SCHEDULE), "_skip_post_task_synthesis": True}
    wake.ctx.is_direct_chat = False
    wake.ctx.current_chat_id = project["chat_id"]
    bus = ObservedEventQueue()
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda **_kw: bus)
    internal = "Internal: no changes worth a message."
    text, usage, trace, _ = wake.run([finish(internal, silent=True)], metadata=SCHEDULE, bus=bus)
    pending, stored = emit(wake, text, usage, trace, bus=bus, task=task)
    assert stored["result"] == internal and stored["wake_silent"] is True
    assert sends(pending) == [] and sends(drained(bus)) == [] and owed_rows(wake.root) == []
    assert not prepare_terminal_task_files(wake.root, task)["error"]
    sent, pushed = [], []
    ctx = _mk_ctx(wake.root, sent, pushed)
    ctx.RUNNING[TASK_ID] = {"task": task}
    done_event = next(event for event in pending if event.get("type") == "task_done")
    _handle_task_done({**done_event, "_files_prepared_attempt": 1}, ctx)
    rows = chat_rows(wake.root)
    done = next(row for row in rows if row.get("summary_id") == f"task-terminal:{TASK_ID}")
    terminal = next(row for row in pushed if row.get("type") == "task_done")
    for row in (done, terminal):
        assert row.get("wake_silent") is True and row.get("quiet_effective") is True, row
    assert done["text"] == "Done · no message" and done["chat_id"] == project["chat_id"]
    assert not any(row.get("role") == "assistant" or row.get("direction") == "out" for row in rows)
    assert not any(row.get("type") == "project_completion_summary" for row in rows)
    assert not sent and sends(drained(bus)) == []
    assert internal not in json.dumps(rows)
    assert load_task_result(wake.root, TASK_ID)["result"] == internal

    notifier = _load()
    monkeypatch.setattr(notifier, "TelegramClient", _Rec)
    _Rec.sent = []
    state_dir = wake.root / "state" / "skills" / "telegram"
    api = SimpleNamespace(get_state_dir=lambda: str(state_dir), get_settings=lambda _keys: {}, log=lambda *a, **k: None)
    seen = {"notified_task_ids": []}
    asyncio.run(notifier._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, seen, "en"))
    assert _Rec.sent == [] and seen["notified_task_ids"] == [TASK_ID]

    node = shutil.which("node")
    assert node, "Node is required for the Python-to-web quiet completion witness"
    module = (Path(__file__).resolve().parents[1] / "web" / "modules" / "notifications.js").as_uri()
    probe = subprocess.run([node, "--input-type=module", "-e", f"""
        import {{ classifyLiveFrame }} from {json.dumps(module)};
        let input = ''; for await (const chunk of process.stdin) input += chunk;
        const [terminal, done] = JSON.parse(input);
        const frames = [classifyLiveFrame(terminal, {{ kind: 'log' }}),
            classifyLiveFrame({{ ...done, role: 'system', system_type: 'task_summary', content: done.text }}, {{ kind: 'chat' }})];
        process.stdout.write(JSON.stringify(frames));
    """], input=json.dumps([terminal, done]), text=True, capture_output=True, timeout=15, check=True)
    assert json.loads(probe.stdout) == [None, None]


@pytest.mark.parametrize("metadata,direct", [
    ({"initiator": "owner", "usage_category": "chat"}, True),
    ({"initiator": "consciousness", "usage_category": CONSCIOUSNESS_TASK_CATEGORY}, True),
    ({"initiator": "consciousness", "usage_category": CONSCIOUSNESS_TASK_CATEGORY}, False),
    ({"presence": {"binding_id": "b1"}, "usage_category": "presence"}, True),
], ids=["owner-turn", "wake-started-root", "wake-started-child", "presence"])
def test_other_turns_cannot_finish_silently(wake, metadata, direct):
    wake.ctx.is_direct_chat = direct
    text, usage, trace, _ = wake.run([finish(silent=True), finish("Ordinary answer.")], metadata=metadata)
    assert trace["tool_calls"][0]["is_error"] is True
    assert "wake_silent" not in usage and text == "Ordinary answer."


def test_silent_stop_is_refused_even_on_the_wake(wake):
    _, usage, trace, _ = wake.run([finish(silent=True, action="stop", rationale="x"), finish(silent=True)])
    assert trace["tool_calls"][0]["is_error"] is True and usage.get("wake_silent") is True


# --- outcomes that must stay visible ----------------------------------------------------

def _degrade(trace):
    trace["verification_events"] = [{"kind": "runtime_event", "services": [
        {"name": "svc", "artifact_output_failed": True, "artifact_outputs": "boom"}]}]


def _custody(wake, trace):
    from ouroboros import delegate_custody as custody
    from ouroboros.task_results import write_task_result

    assert custody.emit(wake.root, custody.STARTED, {"task_id": TASK_ID, "run_id": "run-1"})
    write_task_result(wake.root, TASK_ID, "running", delegated_runs_unreconciled=["run-1"])


def _fail(usage):
    usage["execution_status"] = "failed"


@pytest.mark.parametrize("internal", ["", "INTERNAL: probe svc failed; retry next wake."])
@pytest.mark.parametrize("outcome,status", [("failed", "failed"), ("degraded", "degraded"),
                                            ("custody", "ok")])
def test_nonclean_outcome_publishes_a_host_notice_never_the_internal_words(wake, internal, outcome, status):
    """Errors are not hidden behind silence, and words selected as internal never become speech."""
    from supervisor.terminal_delivery import build_completed_result_event
    from ouroboros.task_finalization import build_sealed_final_package, sealed_final_prompt_section

    text, usage, trace, bus = wake.run([finish(internal or None, silent=True)])
    if outcome == "failed":
        _fail(usage)
    elif outcome == "degraded":
        _degrade(trace)
    else:
        _custody(wake, trace)
    pending, stored = emit(wake, text, usage, trace, bus=bus)
    shown = sends(pending)
    assert len(shown) == 1 and shown[0]["role"] == "system", "one host-authored row, not the model's voice"
    assert shown[0]["text"].startswith("⚠️ This run chose to finish without a message, but it ended")
    assert f"ended {status} " in shown[0]["text"] and shown[0]["terminal_origin"] == "host_notice"
    # Speech is every chat send, live or buffered, and every owed outbox row (tool-call activity is not speech).
    spoken = json.dumps([shown, sends(drained(bus)), owed_rows(wake.root)], ensure_ascii=False)
    assert owed_rows(wake.root), "the notice is owed like any final"
    assert "\u200b" not in spoken and (not internal or internal not in spoken), "internal words never sent"
    assert stored["result"] == internal and stored.get("wake_silent") is True
    assert stored["terminal_origin"] == "model_final", "the retained words stay the model's, typed internal"
    assert shown[0]["text"] in stored["terminal_host_notice"]
    replay = build_completed_result_event(wake.root, {"chat_id": 1}, TASK_ID, stored)
    if outcome == "custody":
        assert replay["role"] == "system" and replay["terminal_origin"] == "host_notice"
        assert replay["text"] == "Open delegated execution: run-1."
        assert not internal or internal not in json.dumps(replay, ensure_ascii=False)
    else:
        assert replay is None
    package = build_sealed_final_package(stored, shown[0]["text"])
    assert package["final_result_text"] == "" and package["internal_terminal_text"] == internal
    assert shown[0]["text"] in package["terminal_host_notice"]
    assert "no model words were submitted" in sealed_final_prompt_section(package)


def test_owner_hard_stop_is_never_silent(wake):
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW
    from ouroboros.tools.control_runtime import _finish_task
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    def stop(ctx, **kw):
        result = _finish_task(ctx, **kw)
        write_owner_message(wake.root, REASON_OWNER_STOPPED_DIRECT_TURN, TASK_ID, kind=KIND_FINALIZE_NOW)
        return result
    wake.registry.override_handler("finish_task", stop)
    _, usage, _, _ = wake.run([finish("partial", silent=True, action="stop", rationale="x"),
                               finish(silent=True)])
    assert "wake_silent" not in usage


# --- held candidates, cold continuation and replay -------------------------------------

@pytest.mark.parametrize("hidden", ["", "hidden internal words"])
def test_owner_words_at_admission_close_void_quiet_bytes(wake, monkeypatch, hidden):
    """Owner words drained at the final admission close void the accepted selection; a plain
    (even empty) follow-up then replaces the held quiet bytes instead of publishing them."""
    import threading

    wake.ctx.owner_message_admission_lock = threading.Lock()
    wake.ctx.owner_message_admission_agent = SimpleNamespace(
        _accepting_owner_messages=True, _busy=True, _current_task_id=TASK_ID)
    original, written = loop._project_child_result_dispositions, []

    def owner_arrives(*args):
        if not written:  # after the selection was consumed, before admission closes
            written.append(write_owner_message(wake.root, "Wait, one question", TASK_ID, msg_id="m9"))
        return original(*args)
    monkeypatch.setattr(loop, "_project_child_result_dispositions", owner_arrives)
    text, usage, trace, _ = wake.run([finish(hidden, silent=True), {"content": ""}])
    assert written and len(wake.calls) == 2
    assert "wake_silent" not in usage and text == ""
    pending, stored = emit(wake, text, usage, trace)
    assert [e["text"] for e in sends(pending)] == ["⚠️ Model returned an empty response. Try rephrasing your request."]
    assert "wake_silent" not in stored


def test_cold_continuation_carries_the_quiet_selection_and_reply_mark(wake):
    from ouroboros.loop_delivery import DeliveryCandidate, _prepare_delivery_candidate
    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    ctx = wake.ctx
    ctx.task_id, ctx.task_attempt = TASK_ID, 1
    ctx._delivery_candidate = DeliveryCandidate(full_text="internal", content_sha256="a" * 64, revision=1,
                                                evidence_revision=0, evidence_fingerprint="",
                                                acceptance_binding={}, quiet=True)
    ctx._completion_selected = {"action": "finish", "answer": "internal", "wake_silent": True,
                                "wake_summary": "Checked the journal."}
    ctx._owner_reply_revision = 2
    state = json.loads(json.dumps(continuation_state(ctx, [], {}, {}, 3, [], set())))
    ctx._delivery_candidate = ctx._completion_selected = ctx._owner_reply_revision = None
    restore_continuation_state(wake.registry, state, [], {}, {}, set())
    assert ctx._delivery_candidate.quiet is True and ctx._completion_selected["wake_silent"] is True
    assert ctx._completion_selected["wake_summary"] == "Checked the journal."
    assert ctx._owner_reply_revision == 2
    # A resumed boundary without the selection (a new execution id) re-asks instead of publishing.
    ctx._completion_selected = None
    ready, _ = _prepare_delivery_candidate("internal", wake.registry, SimpleNamespace(messages=[]), {},
                                           explicit=False, resumed=True)
    assert ready is False


class _ProcessDied(BaseException):
    """The worker dies right after its continuation was saved (nothing in the loop catches it)."""


def _park_after_selection(wake, monkeypatch, responses):
    """Run the wake until its silent selection is consumed, save the loop's own owner-wait
    continuation at that exact boundary, then die; nothing is checkpointed by hand."""
    from ouroboros.owner_wait import checkpoint_owner_wait
    from ouroboros.task_results import write_task_result

    write_task_result(wake.root, TASK_ID, "running")
    wake.ctx.task_attempt, saved = 1, {}
    original = loop._project_child_result_dispositions

    def park(limit_ctx, llm_trace):
        ctx = wake.ctx
        if not saved and (getattr(ctx, "_completion_selected", None) or {}).get("wake_silent"):
            saved["wait"] = checkpoint_owner_wait(
                ctx, limit_ctx.messages, llm_trace, limit_ctx.accumulated_usage, limit_ctx.round_idx,
                loop.initial_tool_schemas(wake.registry), set(limit_ctx.owner_msg_seen or ()))
            raise _ProcessDied()
        return original(limit_ctx, llm_trace)
    monkeypatch.setattr(loop, "_project_child_result_dispositions", park)
    with pytest.raises(_ProcessDied):
        wake.run(responses)
    monkeypatch.setattr(loop, "_project_child_result_dispositions", original)
    return saved["wait"]


def _resume_in_new_process(wake, monkeypatch, wait):
    from ouroboros import context
    from ouroboros.owner_wait import set_owner_wait
    from tests.test_context_fit_v664 import _plan

    set_owner_wait(wake.root, TASK_ID, {**wait, "state": "waiting"})
    fresh = ToolRegistry(repo_dir=wake.root, drive_root=wake.root)
    resumed = fresh._ctx
    resumed.is_direct_chat, resumed.current_chat_id, resumed.task_metadata = True, 1, dict(WAKE)
    resumed.task_id, resumed.task_attempt = TASK_ID, 1
    resumed.owner_wait_resume = {**wait, "restart_transaction_id": "observed-restart"}
    resumed.owner_wait_callback = lambda *_: None
    resumed.context_fit_plan = _plan(window=500_000, known=True)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="route-a")))
    wake.registry, wake.ctx = fresh, resumed


INTERNAL = "INTERNAL: journal read, nothing new."


@pytest.mark.parametrize("summary", ["", "Checked the journal."])
@pytest.mark.parametrize("sent", [False, True])
def test_cold_resume_completes_the_restored_quiet_selection_silently(wake, monkeypatch, summary, sent):
    responses = [call("send_user_message", {"text": "A visible progress update."})] if sent else []
    wait = _park_after_selection(wake, monkeypatch, [*responses, finish(INTERNAL, silent=True, summary=summary)])
    assert bool(getattr(wake.ctx, "_owner_message_sent", False)) is sent
    _resume_in_new_process(wake, monkeypatch, wait)
    text, usage, trace, bus = wake.run([{"content": ""}])
    assert usage.get("wake_silent") is True and text == INTERNAL
    pending, stored = emit(wake, text, usage, trace, bus=bus)
    assert sends(pending) == [] and sends(drained(bus)) == [] and owed_rows(wake.root) == []
    assert stored["result"] == INTERNAL and stored.get("wake_silent") is True
    assert stored.get("wake_summary", "") == summary and usage.get("wake_summary", "") == summary
    assert "summary" not in stored
    assert bool(stored.get("owner_message_sent")) is sent
    assert quiet_finish(stored) is (not sent)


@pytest.mark.parametrize("summary", ["", "Checked the tests."])
def test_cold_resume_new_owner_words_void_the_quiet_selection_without_leaking(wake, monkeypatch, summary):
    wait = _park_after_selection(wake, monkeypatch, [
        wake.owner_writes("Status?", call("chat_history", {}), "m1"),
        call("send_user_message", {"text": "Quiet day."}),
        finish(INTERNAL, silent=True, summary="Stale journal line.")])
    _resume_in_new_process(wake, monkeypatch, wait)
    write_owner_message(wake.root, "Actually, what about the tests?", TASK_ID, msg_id="m2")
    text, usage, trace, bus = wake.run([{"content": ""}, finish(silent=True), {"content": ""},
                                        finish("Tests pass.", summary=summary)])
    assert wake.ctx._owner_reply_revision == 1, "the restored mark covers m1, not the new m2"
    reasons = [row["reason"] for row in trace.get("completion_refusals") or []]
    assert any(reason.startswith("owner_reply_owed") for reason in reasons)
    assert "wake_silent" not in usage and text == "Tests pass."
    pending, stored = emit(wake, text, usage, trace, bus=bus)
    spoken = json.dumps([sends(pending), sends(drained(bus)), owed_rows(wake.root)], ensure_ascii=False)
    assert "Tests pass." in spoken and INTERNAL not in spoken
    assert usage.get("wake_summary", "") == stored.get("wake_summary", "") == summary
    assert "Stale journal line." not in spoken and "summary" not in stored


def test_cold_resume_restored_reply_mark_still_answers_earlier_owner_words(wake, monkeypatch):
    """Parked after an answered owner row: the restored mark lets the resumed wake finish silently."""
    wait = _park_after_selection(wake, monkeypatch, [
        wake.owner_writes("Status?", call("chat_history", {}), "m1"),
        call("send_user_message", {"text": "Quiet day."}),
        finish(INTERNAL, silent=True)])
    _resume_in_new_process(wake, monkeypatch, wait)
    text, usage, trace, _ = wake.run([finish(silent=True)])  # a fresh selection after the restored one
    assert usage.get("wake_silent") is True and not trace.get("completion_refusals")


@pytest.mark.parametrize("hard_rail", [False, True])
def test_authored_stop_publishes_a_summary_only_when_accepted(wake, monkeypatch, hard_rail):
    from ouroboros import loop_budget

    fallback, rails = loop._forced_fallback_result, []

    def record_rail(*args, **kwargs):
        rails.append(kwargs["source"])
        return fallback(*args, **kwargs)

    monkeypatch.setattr(loop, "_forced_fallback_result", record_rail)
    monkeypatch.setattr(loop_budget, "authored_completion_budget_exhausted", lambda *a: hard_rail)
    text, usage, trace, bus = wake.run([finish("Unfinished note.", action="stop", rationale="One probe remains.",
                                               summary="Checked what I could.")])
    assert len(wake.calls) == 1
    _, stored = emit(wake, text, usage, trace, bus=bus)
    if hard_rail:
        assert rails == ["authored_stop_at_hard_rail"] and trace["forced_finalization"]
        assert "wake_summary" not in usage and "wake_summary" not in stored
    else:
        assert not rails and "forced_finalization" not in trace
        assert stored["wake_summary"] == stored["suggested_name"] == "Checked what I could."
    assert "summary" not in stored


def test_forced_exit_discards_a_carried_sentence_even_with_retained_candidate(wake, monkeypatch):
    original = loop._record_forced_finalization

    def carried_sentence(ctx, *args, **kwargs):
        ctx.accumulated_usage["wake_summary"] = "A stale checkpoint's sentence."
        return original(ctx, *args, **kwargs)

    monkeypatch.setattr(loop, "_record_forced_finalization", carried_sentence)
    _control_at_admission_close(wake, monkeypatch, "deadline", expire=True)
    text, usage, trace, bus = wake.run([finish(INTERNAL, silent=True, summary="Unaccepted sentence.")])
    _, stored = emit(wake, text, usage, trace, bus=bus)
    assert trace["forced_finalization"] and "wake_summary" not in usage and "wake_summary" not in stored


# --- forced rails after a quiet selection -------------------------------------------------

def _control_at_admission_close(wake, monkeypatch, control, *, expire=False, stale=False):
    """A mailbox control (and optionally an elapsed deadline or newer evidence) lands after the
    silent selection was consumed, before admission closes, so a forced rail decides the final."""
    import threading
    import ouroboros.loop_delivery as delivery
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW

    wake.ctx.owner_message_admission_lock = threading.Lock()
    wake.ctx.owner_message_admission_agent = SimpleNamespace(
        _accepting_owner_messages=True, _busy=True, _current_task_id=TASK_ID)
    original, injected = loop._project_child_result_dispositions, []

    def arrive(limit_ctx, llm_trace):
        if not injected and (getattr(wake.ctx, "_completion_selected", None) or {}).get("wake_silent"):
            injected.append(True)
            if expire:
                limit_ctx.deadline_ts = time.time() - 1
            write_owner_message(wake.root, control, TASK_ID, kind=KIND_FINALIZE_NOW)
        return original(limit_ctx, llm_trace)
    monkeypatch.setattr(loop, "_project_child_result_dispositions", arrive)
    if stale:  # newer evidence reaches the loop as the wrap-up opens, after the selection
        wrap_up = loop._forced_final_answer

        def newer_evidence(*args, **kwargs):
            monkeypatch.setattr(delivery, "observed_delivery_evidence", lambda *_a, **_kw: "newer-evidence")
            return wrap_up(*args, **kwargs)
        monkeypatch.setattr(loop, "_forced_final_answer", newer_evidence)
    return injected


def _spoken(wake, pending, bus):
    return json.dumps([sends(pending), sends(drained(bus)), owed_rows(wake.root)], ensure_ascii=False)


@pytest.mark.parametrize("internal", ["", INTERNAL])
@pytest.mark.parametrize("control,expire,source,reason", [
    ("deadline", True, "retained_candidate", "finalization_grace"),
    ("owner_stopped_direct_chat_turn", False, "owner_stop_retained_candidate", "owner_requested_finalization"),
], ids=["deadline", "owner-stop"])
def test_forced_rail_retaining_quiet_bytes_publishes_only_the_host_notice(wake, monkeypatch, internal,
                                                                           control, expire, source, reason):
    """The forced rail keeps the quiet candidate's exact bytes: they stay internal, while the
    failed outcome is still announced in the host's own words."""
    from ouroboros.task_finalization import build_sealed_final_package

    injected = _control_at_admission_close(wake, monkeypatch, control, expire=expire)
    text, usage, trace, bus = wake.run([finish(internal or None, silent=True, summary="Checked the probe.")])
    assert injected and len(wake.calls) == 1, "no paid wrap-up call"
    assert trace["forced_finalization"]["source"] == source
    assert text == internal and usage.get("wake_silent") is True
    pending, stored = emit(wake, text, usage, trace, bus=bus)
    shown = sends(pending)
    assert len(shown) == 1 and shown[0]["role"] == "system" and shown[0]["terminal_origin"] == "host_notice"
    assert shown[0]["text"].startswith("⚠️ This run chose to finish without a message, but it ended")
    assert f"({reason})" in shown[0]["text"] and "ended ok" not in shown[0]["text"]
    spoken = _spoken(wake, pending, bus)
    assert owed_rows(wake.root) and (not internal or internal not in spoken), "the notice is owed, never the words"
    assert stored["result"] == internal and stored.get("wake_silent") is True
    assert "wake_summary" not in usage and "wake_summary" not in stored
    assert build_sealed_final_package(stored, shown[0]["text"])["final_result_text"] == ""


def test_forced_wrap_up_with_new_ordinary_words_speaks_them(wake, monkeypatch):
    """A wrap-up the model authors afresh replaces the quiet candidate and is ordinary speech."""
    _control_at_admission_close(wake, monkeypatch, "deadline")
    text, usage, trace, bus = wake.run([finish(INTERNAL, silent=True, summary="Checked the probe."),
                                        {"role": "assistant", "content": "Wrap-up: the probe is still red."}])
    assert len(wake.calls) == 2 and trace["forced_finalization"]["source"] == "model"
    assert "wake_silent" not in usage and text == "Wrap-up: the probe is still red."
    pending, stored = emit(wake, text, usage, trace, bus=bus)
    spoken = _spoken(wake, pending, bus)
    assert "Wrap-up: the probe is still red." in spoken and INTERNAL not in spoken
    assert "wake_silent" not in stored
    assert "wake_summary" not in usage and "wake_summary" not in stored


def test_forced_retention_of_stale_quiet_bytes_stays_internal(wake, monkeypatch):
    """Retaining the quiet bytes after newer evidence republishes them as a stale candidate;
    that replacement keeps their provenance, and the stale disclosure stays on the record."""
    import hashlib

    _control_at_admission_close(wake, monkeypatch, "deadline", stale=True)
    keep = json.dumps({"action": "finish", "answer_sha256": hashlib.sha256(INTERNAL.encode()).hexdigest()})
    text, usage, trace, bus = wake.run([finish(INTERNAL, silent=True), {"role": "assistant", "content": keep}])
    assert len(wake.calls) == 2
    assert trace["forced_finalization"]["source"] == "model_control_retained_stale_evidence_resume_required"
    assert text == INTERNAL and usage.get("wake_silent") is True
    pending, stored = emit(wake, text, usage, trace, bus=bus)
    shown = sends(pending)
    assert len(shown) == 1 and shown[0]["text"].startswith("⚠️ This run chose to finish without a message")
    assert INTERNAL not in _spoken(wake, pending, bus)
    assert "STALE-EVIDENCE NOTICE" in stored["terminal_host_notice"] and stored["result"] == INTERNAL


def _batch(*calls):
    return {"role": "assistant", "content": None, "tool_calls": [c["tool_calls"][0] for c in calls]}


def _refused_quiet_selection(internal):
    """One batch reads and selects silence: the selection is refused as unobserved, its bytes held."""
    return _batch(call("chat_history", {}), finish(internal or None, silent=True))


def _host_notice_only(wake, text, usage, trace, bus, internal, reason):
    pending, stored = emit(wake, text, usage, trace, bus=bus)
    shown = sends(pending)
    assert len(shown) == 1 and shown[0]["role"] == "system" and shown[0]["terminal_origin"] == "host_notice"
    assert shown[0]["text"].startswith("⚠️ This run chose to finish without a message") and reason in shown[0]["text"]
    spoken = _spoken(wake, pending, bus)
    assert "empty response" not in spoken and (not internal or internal not in spoken)
    assert stored["result"] == internal and stored.get("wake_silent") is True
    assert stored["terminal_origin"] == "model_final"


@pytest.mark.parametrize("internal", ["", "INTERNAL: probe read.\nFINAL ANSWER: internal pick"])
def test_refused_quiet_selection_then_owner_stop_stays_internal(wake, internal):
    """Held quiet bytes reach the hard-stop rail: no internal words, no false empty-response notice,
    and the quiet words latch no fallback answer for the budget rail either."""
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    def read(_ctx, **_kw):
        write_owner_message(wake.root, REASON_OWNER_STOPPED_DIRECT_TURN, TASK_ID, kind=KIND_FINALIZE_NOW)
        return "read result"
    wake.registry.override_handler("chat_history", read)
    text, usage, trace, bus = wake.run([_refused_quiet_selection(internal)])
    assert (trace.get("completion_refusals") or [{}])[0].get("reason", "").startswith("unobserved_tool_results")
    assert len(wake.calls) == 1 and trace["forced_finalization"]["source"] == "owner_stop_retained_candidate"
    assert "best_valid_final_answer" not in trace
    assert text == internal and usage.get("wake_silent") is True
    _host_notice_only(wake, text, usage, trace, bus, internal, "(owner_requested_finalization)")


def test_provider_death_after_refused_quiet_selection_keeps_words_internal(wake):
    """The provider-salvage rail retains the held quiet bytes; only its failure is announced."""
    wake.registry.override_handler("chat_history", lambda _ctx, **_kw: "read result")
    text, usage, trace, bus = wake.run([_refused_quiet_selection(INTERNAL), None, None, None])
    assert trace["forced_finalization"]["source"] == "retained_candidate"
    assert text == INTERNAL and usage.get("wake_silent") is True
    _host_notice_only(wake, text, usage, trace, bus, INTERNAL, "(provider_unavailable)")


def test_completion_wins_quiet_row_speaks_only_its_custody_line(wake):
    """A completion-wins teardown that left runs open discloses them for a silent wake too —
    as the host's own row, never the internal words."""
    from supervisor.terminal_delivery import build_completed_result_event

    text, usage, trace, _ = wake.run([finish(INTERNAL, silent=True)])
    _, stored = emit(wake, text, usage, trace)
    assert build_completed_result_event(wake.root, {"chat_id": 1}, TASK_ID, stored) is None
    event = build_completed_result_event(wake.root, {"chat_id": 1}, TASK_ID, stored, unreconciled_runs=["run-7"])
    assert event["role"] == "system" and event["terminal_origin"] == "host_notice"
    assert "run-7" in event["text"] and INTERNAL not in json.dumps(event, ensure_ascii=False)


def test_history_replay_and_completion_wins_owe_no_silent_row(wake):
    from supervisor.terminal_delivery import build_completed_result_event
    from ouroboros.task_finalization import build_sealed_final_package, sealed_final_prompt_section

    text, usage, trace, _ = wake.run([finish("internal note", silent=True)])
    _, stored = emit(wake, text, usage, trace)
    assert build_completed_result_event(wake.root, {"chat_id": 1}, TASK_ID, stored) is None
    package = build_sealed_final_package(stored, "")
    assert package["final_result_text"] == "" and package["internal_terminal_text"] == "internal note"
    section = sealed_final_prompt_section(package)
    assert "finished silently" in section and "never sent to the owner" in section
