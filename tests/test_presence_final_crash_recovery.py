"""A Presence root's Project final is owed before its result is stored; a death between them must not make that
final its conversation's speech.

Recovery finalizes such a root from its owed final (``working_checkpoint.adopt_registered_answer``); the delivery its
Presence result stamped before the death (``presence_delivery``) rides that final through the outbox (owed, delivered
with its receipt, or given up) and becomes the recovered row's own. Promoted roots and inline turns run through the real
loop, pipeline, place and Host routes of ``tests.test_presence_explicit_final``; nothing leaves the process.
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from ouroboros import places
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_runner import run_presence_turn
from ouroboros.task_results import load_task_result
from tests.test_presence_explicit_final import (
    FINAL, HEADERS, NOTE, WorkerDeath, finish, promoted_root, room_speech, supervise, turn)
from tests.test_presence_explicit_final import room as room_fixture

room = room_fixture  # the real Host, place, transport and scripted model


# Where a death between the owed final and the result store leaves it: authorized, held at its finish (its place open
# again by the read) or held by the place's state since, each owed, delivered or given up; an explicit silent finish
# owes no final at all.
CRASH_CASES = [*((case, basis) for case in ("speech", "held", "paused_since") for basis in ("owed", "delivered", "exhausted")),
               ("silent", "none")]


@pytest.mark.parametrize("case,basis", CRASH_CASES)
def test_a_death_before_the_result_store_keeps_the_conversations_own_delivery(room, monkeypatch, basis, case):
    """A Project root's final is owed before its result is stored. A death in between finalizes from that final
    (``adopt_registered_answer``), whose words are the Project's and never its conversation's speech. Owed, delivered
    with its receipt or given up, the recovered row keeps what the Presence result stamped before the death: a held
    reply's explicit silence and empty text, an authorized reply's exact words and output identity. The authenticated
    work poll, the cached turn projection and the place's adapter read it, the held reply's place open again; the
    place's current state still holds an authorized reply. The Project keeps its final either way. An explicit silent
    finish owes nothing: nothing is adopted, heard or kept in any room."""
    from ouroboros.presence_runner import _cached_result
    from ouroboros.projects_registry import create_project
    from ouroboros.working_checkpoint import adopt_registered_answer
    from supervisor import terminal_delivery as td
    from tests._cancel_intents_shared import _CaptureQueue

    project = create_project(room.data, "report", name="Report")
    answer = "The compiled report is attached above."

    def pause_then_finish(_messages):
        places.edit_place(room.data, room.place.place_id, "set_state", author="owner", state="paused")
        return finish(answer=answer)

    final = {"held": pause_then_finish, "silent": finish(silent=True, answer=NOTE)}.get(case, finish(answer=answer))
    root = promoted_root(room, monkeypatch, [final], chat_id=project["chat_id"], project=project, crash=True)
    result = next(event for event in root.pending if event.get("type") == "presence_result")
    spoken = ("message", answer) if case in {"speech", "paused_since"} else ("silent", "")
    assert (result["outcome"], result["text"]) == spoken and bool(result["output_ref"]) == (spoken[0] == "message")
    assert root.stored["status"] == "running" and "presence_outcome" not in root.stored["metadata"]  # died first
    owed = [row for row in td.pending_deliveries(room.data, strict=True) if row.get("task_id") == root.id]
    if case == "silent":
        assert owed == [] and adopt_registered_answer(room.data, dict(root.stored), root.id, attempt=1,
                                                      cause="worker_crash") is None
        assert _silent_after_death(room, root.id, project["chat_id"], before=room.transport.texts())
        return
    assert [(row["chat_id"], row["text"]) for row in owed] == [(project["chat_id"], answer)]
    if basis == "delivered":  # the live copy left before the death and its receipt was retained
        supervise(room.data, monkeypatch, [event for event in root.pending if event.get("type") == "send_message"])
    elif basis == "exhausted":
        td._retire_exhausted(room.data, owed)
    assert td.terminal_answer_receipts(room.data, root.id)["state"] == basis
    adopt_registered_answer(room.data, dict(root.stored), root.id, attempt=1, cause="worker_crash")
    stored = load_task_result(room.data, root.id)
    assert (stored["status"], stored["result"], stored["finalization_recovery"]["answer"]) == ("failed", answer, basis)
    assert ({key: stored["metadata"].get(key, "absent") for key in ("presence_outcome", "presence_result_text")}
            == dict(zip(("presence_outcome", "presence_result_text"), spoken)))
    assert stored["metadata"].get("presence_output_ref", "") == result["output_ref"]
    if case != "speech":  # held: open again, so its stamped silence alone keeps the final unheard there
        places.edit_place(room.data, room.place.place_id, "set_state", author="owner",
                          state="active" if case == "held" else "paused")
    heard = spoken if case == "speech" else ("silent", "")
    view = room.client.get(f"/presence/work/{root.id}", headers=HEADERS).json()
    cached = _cached_result(room.data, root.id)
    assert (view["outcome"], view["text"]) == (cached.outcome, cached.text) == heard
    before = room.transport.texts()
    room.transport.work[root.id] = "chat:a:1"
    room.transport.poll()
    room.transport.poll()
    assert room.transport.texts()[len(before):] == ([answer] if case == "speech" else [])  # once, when it may speak
    if basis == "owed":  # the outbox replay still sends the Project its final
        monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0.0)
        bus = _CaptureQueue()
        td.replay_pending_deliveries(room.data, event_queue=bus)
        supervise(room.data, monkeypatch, bus.events)
    assert room_speech(room.data, project["chat_id"], root.id) == ([] if basis == "exhausted" else [answer])
    assert answer not in room_speech(room.data, 1, root.id)


def _silent_after_death(room, task_id, project_chat, *, before):
    """An explicit silent finish that died before its result was stored: no final was owed, so the work poll names
    no speech, the place hears nothing and no room keeps its note."""
    view = room.client.get(f"/presence/work/{task_id}", headers=HEADERS).json()
    room.transport.work[task_id] = "chat:a:1"
    room.transport.poll()
    return (view.get("outcome") != "message" and not view.get("text") and room.transport.texts() == before
            and room_speech(room.data, project_chat, task_id) == room_speech(room.data, 1, task_id) == [])


@pytest.mark.parametrize("case,basis", CRASH_CASES)
def test_an_inline_turn_that_dies_before_its_result_store_keeps_its_conversations_own_delivery(room, monkeypatch,
                                                                                              case, basis):
    """The inline branch of the same death: a turn whose work became a Project's mid-flight dies, with the Host that
    ran it, after its Project final was owed (left owed, sent by the live bus first, or given up) and before its
    result was stored. The live Host finalizes it from that final (``adopt_registered_answer``) and keeps the delivery
    its Presence result stamped: exact words, output identity and reply anchor when authorized, explicit silence and
    empty text when held, the place's current state over an authorized reply. The transport's fresh work poll and its
    retried submission of the same event (the cached turn, no model run) agree. An explicit silent finish owes no
    final: nothing is adopted or heard."""
    import queue

    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.working_checkpoint import adopt_registered_answer
    from supervisor import terminal_delivery as td
    from tests._cancel_intents_shared import _CaptureQueue

    project = create_project(room.data, "report", name="Report")
    bus = queue.Queue()
    if basis == "delivered":  # the Host's own runner passes the supervisor bus: the final leaves before the store
        made = room.h.factory
        monkeypatch.setattr(room.h, "factory", lambda **kwargs: made(**{**kwargs, "event_queue": bus}))
    stamped, build = [], pipeline.build_presence_result_event
    monkeypatch.setattr(pipeline, "build_presence_result_event", lambda *a, **k: stamped.append(build(*a, **k)) or stamped[-1])

    def bind_then_finish(_messages):
        bind_task_to_project(room.data, room.h.agents[-1].tools._ctx.task_id, project["id"], project["chat_id"],
                             origin={"absent": "system"})
        if case == "held":
            places.edit_place(room.data, room.place.place_id, "set_state", author="owner", state="paused")
        return finish(silent=True, answer=NOTE) if case == "silent" else finish(answer=FINAL)

    def die(*_a, **_k):
        raise WorkerDeath()

    room.h.script = [bind_then_finish]
    monkeypatch.setattr(pipeline, "_store_task_result", die)
    dying = TestClient(create_host_service_app(room.data, presence_runner=lambda **kw: run_presence_turn(
        repo_dir=room.h.repo, drive_root=room.data, agent_factory=room.h.factory, **kw))).__enter__()
    with pytest.raises(WorkerDeath):  # that Host dies with the turn; nothing closes it
        turn(room, client=dying)
    tid, calls, result = room.h.agents[-1].tools._ctx.task_id, room.h.calls, stamped[-1]
    spoken = ("message", FINAL) if case in {"speech", "paused_since"} else ("silent", "")
    assert (result["outcome"], result["text"]) == spoken and bool(result["output_ref"]) == (spoken[0] == "message")
    stored = load_task_result(room.data, tid)
    assert stored["status"] == "running" and "presence_outcome" not in stored["metadata"]  # died first
    owed = [row for row in td.pending_deliveries(room.data, strict=True) if row.get("task_id") == tid]
    if case == "silent":
        assert owed == [] and adopt_registered_answer(room.data, dict(stored), tid, attempt=1, cause="worker_crash") is None
        assert _silent_after_death(room, tid, project["chat_id"], before=[])
        return
    assert [(row["chat_id"], row["text"]) for row in owed] == [(project["chat_id"], FINAL)]
    if basis == "delivered":
        live = [event for event in iter(lambda: bus.get_nowait() if not bus.empty() else None, None)
                if event.get("type") == "send_message" and event.get("task_id") == tid]
        assert [event["delivery_id"] for event in live] == [owed[0]["delivery_id"]]
        supervise(room.data, monkeypatch, live)
    elif basis == "exhausted":
        td._retire_exhausted(room.data, owed)
    assert td.terminal_answer_receipts(room.data, tid)["state"] == basis
    adopt_registered_answer(room.data, dict(stored), tid, attempt=1, cause="worker_crash")
    stored = load_task_result(room.data, tid)
    assert (stored["status"], stored["result"], stored["finalization_recovery"]["answer"]) == ("failed", FINAL, basis)
    keys = ("presence_outcome", "presence_result_text", "presence_output_ref", "presence_reply_anchor")
    assert {key: stored["metadata"].get(key, "absent") for key in keys} == {
        "presence_outcome": result["outcome"], "presence_result_text": result["text"],
        "presence_output_ref": result["output_ref"] or "absent", "presence_reply_anchor": result.get("reply_anchor", "absent")}
    if case != "speech":  # held: open again, so its stamped silence alone keeps the final unheard there
        places.edit_place(room.data, room.place.place_id, "set_state", author="owner",
                          state="active" if case == "held" else "paused")
    heard = spoken if case == "speech" else ("silent", "")
    view = room.client.get(f"/presence/work/{tid}", headers=HEADERS).json()
    body = turn(room)  # the transport's retry of the same event
    assert (view["outcome"], view["text"]) == (body["outcome"], body["text"]) == heard and room.h.calls == calls
    # Its output identity travels in the work poll, held or not (the version-0 turn body keeps its legacy shape).
    assert view["output_ref"] == result["output_ref"] and "output_ref" not in body
    room.transport.submit("chat:a:1", body)
    assert room.transport.texts() == ([FINAL] if case == "speech" else [])  # once, when it may speak
    if basis == "owed":  # the outbox replay still sends the Project its final
        monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0.0)
        replayed = _CaptureQueue()
        td.replay_pending_deliveries(room.data, event_queue=replayed)
        supervise(room.data, monkeypatch, replayed.events)
    assert room_speech(room.data, project["chat_id"], tid) == ([] if basis == "exhausted" else [FINAL])
    assert FINAL not in room_speech(room.data, 1, tid)
