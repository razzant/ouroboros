"""A final the mind selects after it already spoke is new speech; silence and handoffs stay explicit.

The real Host routes, skill token, place admission, turn gate, production agent bootstrap,
registered ``finish_task``, loop finalization, current place state and Safety check, terminal
pipeline, replay and work polls run with a scripted model. ``Transport`` mirrors the installed
Telegram adapter without importing Hub code: its tool queues a provider send that its worker
delivers and reports through the Host's own receipt writer (an interim acknowledgement, a
multipart or formatted text, a file without a caption, a reaction), ``submit`` is
``custody.record_submission`` and ``poll`` is ``runtime.process_one_work`` over ``host.poll``.
Each output identity is sent once. Nothing leaves the process; no provider is contacted.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import places
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_runner import _read_previous_turn, run_presence_turn
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolEntry

TOKEN = "explicit-final-token"
HEADERS = {"X-Skill-Token": TOKEN}
KEY = "chat:a:r:0"
ACK = "On it: checking the three services now."
FINAL = "All three services are healthy; the nightly backup finished at 03:10."
RESULT = "Report: all three services are healthy."
NOTE = "Sent the full report through the transport; nothing more to say here."
CHILD = "work-explicit-final-child"
CHILD_ANSWER = "The independent audit found no drift."
SEND_SCHEMA = {"name": "bridge_send", "description": "Queue one provider send in this conversation.",
               "parameters": {"type": "object", "properties": {
                   "text": {"type": "string"}, "kind": {"type": "string"}, "parts": {"type": "integer"},
                   "format": {"type": "string"}, "state": {"type": "string"}, "thread_id": {"type": "string"},
                   "foreign_task": {"type": "string"}, "request_id": {"type": "string"}}}}


class Transport:
    """The installed adapter's custody contract, in process; ``provider`` is every physical send."""

    def __init__(self):
        self.client = self.recorder = None
        self.provider: list[dict] = []
        self.work: dict[str, str] = {}
        self.sent: set[str] = set()

    def tool(self, ctx, text="", kind="message", parts=1, format="plain", state="delivered",
             thread_id="", foreign_task="", request_id="send"):
        """The connector's tool queues; its outbox worker sends and reports (here before returning)."""
        event = ctx.task_metadata["presence"]["event"]
        chunks = text.split("|") if parts > 1 else [text]
        message = ({"kind": "operation", "method": "setMessageReaction", "reaction": [{"type": "emoji", "emoji": "\U0001F44D"}]}
                   if kind == "reaction" else {"kind": kind, **({"file_name": "report.pdf"} if kind == "document" else {})})
        for part, chunk in enumerate(chunks):
            self.provider.append({"origin": "tool", "kind": kind, "text": chunk, "thread_id": thread_id})
            self.recorder.record("bridge", receipt(f"tool:{request_id}", str(part), chunk, state=state, fmt=format,
                                                   message=message, thread_id=thread_id, origin="tool",
                                                   task_id=foreign_task or ctx.task_id, source=event["source_event_id"]))
        return json.dumps({"ok": True, "state": "queued", "request_id": request_id})

    def deliver(self, identity, text, task_id, source):
        """One automatic send per output identity, reported through the real receipt route."""
        if identity in self.sent:
            return
        self.sent.add(identity)
        self.provider.append({"origin": "automatic", "kind": "message", "text": text, "thread_id": ""})
        reply = self.client.post("/presence/delivery", headers=HEADERS, json=receipt(
            identity, "0", text, origin="automatic", task_id=task_id, source=source))
        assert reply.status_code == 200 and reply.json()["recorded"] is True, reply.text

    def submit(self, event_id, body):
        assert body["ok"] is True and body["outcome"] in {"message", "silent", "tool_delivered", "deferred"}
        if body["text"] and body["outcome"] in {"message", "deferred"}:
            self.deliver(body.get("output_ref") or f"presence:{event_id}:turn", body["text"], body["turn_ref"], event_id)
        observed = body.get("continuation_ref") or (body["work_ref"] if body["outcome"] == "deferred" else "")
        assert observed or body["outcome"] != "deferred", "deferred submission requires work_ref"
        if observed:
            self.work[observed] = event_id
        if body.get("continuation_ref") and body["work_ref"] and body["work_ref"] != body["continuation_ref"]:
            self.work[body["work_ref"]] = event_id

    def poll(self):
        for ref, event_id in list(self.work.items()):
            reply = self.client.get(f"/presence/work/{ref}", headers=HEADERS)
            assert reply.status_code in {200, 202}, reply.text
            body = reply.json()
            outputs = list(body.get("outputs") or [])
            if body.get("output_ref") and body.get("text"):
                outputs.append(body)
            for output in outputs:
                if output["outcome"] in {"message", "deferred"} and output.get("text"):
                    self.deliver(output["output_ref"], output["text"], ref, event_id)
            child = next((value for value in (body.get("child_work_ref"), body.get("work_ref"))
                          if value and value not in {ref, body.get("turn_ref")}), "")
            if child:
                self.work.setdefault(child, event_id)
            if body["status"] in {"completed", "failed", "cancelled"}:
                if body["outcome"] in {"message", "deferred"} and body.get("text") and not body.get("output_ref"):
                    self.deliver(f"presence-work:{ref}", body["text"], ref, event_id)
                del self.work[ref]

    def texts(self, origin=None):
        return [row["text"] for row in self.provider if origin in (None, row["origin"])]


def receipt(delivery_id, part, text, *, state="delivered", fmt="plain", message=None, thread_id="",
            origin="tool", task_id="", source=""):
    return {"schema_version": 1, "delivery_id": delivery_id, "part_id": part, "state": state,
            "provider": "chat", "account_id": "a", "conversation_id": "r", "thread_id": thread_id,
            "text": text, "format": fmt, "message": message or {"kind": "message"},
            "origin": {"kind": origin, "task_id": task_id, "source_event_id": source}}


def call(name, arguments, identifier=None):
    return {"content": None, "tool_calls": [{"id": identifier or name, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}]}


def send(**arguments):
    return call("bridge_send", arguments, "send-" + str(arguments.get("request_id", "send")))


def finish(**arguments):
    return call("finish_task", {"action": "finish", **arguments}, "finish")


@pytest.fixture
def room(tmp_path, monkeypatch):
    from tests.test_host_service_api import _seed_token
    from tests.test_presence_continuation_bootstrap import install_bootstrap_harness

    data = tmp_path / "data"
    h = install_bootstrap_harness(monkeypatch, data, review_mode="off")
    _seed_token(data, skill="bridge", token=TOKEN, permissions=["presence"], manifest_permissions=["presence"])
    place = places.connect_place(data, connector="bridge", provider="chat", account_id="a",
                                 conversation_id="r", wakes=["mention"])
    transport = Transport()

    def factory(**kwargs):
        agent = h.factory(**kwargs)
        agent.tools.register(ToolEntry(name="bridge_send", schema=SEND_SCHEMA, handler=transport.tool,
                                       timeout_sec=30, safety_policy="skip"))
        return agent

    app = create_host_service_app(data, presence_runner=lambda **kw: run_presence_turn(
        repo_dir=h.repo, drive_root=data, agent_factory=factory, **kw))
    with TestClient(app) as client:
        transport.client, transport.recorder = client, app.state.host_service_context.presence_deliveries
        yield SimpleNamespace(h=h, data=data, place=place, client=client, transport=transport)


def turn(room, event_id="chat:a:1", version=0, reporting=1, text="Please check the three services."):
    payload = {"event": {"source_event_id": event_id, "provider": "chat", "account_id": "a",
                         "conversation_id": "r", "thread_id": "", "conversation_key": "ignored-by-host",
                         "actor": {"platform_actor_id": "person"}, "conversation": {},
                         "message": {"message_id": event_id}, "text": text, "facts": ["mention"]},
               **({"delivery_reporting_version": 1} if reporting else {}),
               **({"continuation_version": version} if version else {})}
    reply = room.client.post("/presence/turn", json=payload, headers=HEADERS)
    assert reply.status_code == 200, reply.text
    return reply.json()


def settle(room, body, event_id="chat:a:1"):
    """Submit like the adapter, then poll until every observed reference has settled."""
    import time

    room.transport.submit(event_id, body)
    deadline = time.monotonic() + 60
    while room.transport.work and CHILD not in room.transport.work and time.monotonic() < deadline:
        room.transport.poll()
        time.sleep(0.02)
    assert not set(room.transport.work) - {CHILD}, "an author's reference never settled"
    view = room.client.get(f"/presence/work/{body['turn_ref']}", headers=HEADERS).json()
    return view, load_task_result(room.data, body["turn_ref"])


def pointer_of(room, ref, seconds=60):
    """The turn thread writes its pointer just after its terminal row, which a version-2 poll can see first."""
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pointer = _read_previous_turn(room.data, KEY) or {}
        if pointer.get("task_id") == ref and not pointer.get("continuing"):
            return pointer
        time.sleep(0.02)
    raise AssertionError("the turn's pointer never landed")


def receipts(room, task_id):
    from ouroboros import dialogue_store

    return [entry.row for entry in dialogue_store.iter_rows(room.data, "chat")
            if entry.row.get("type") == "presence_delivery" and entry.row.get("task_id") == task_id]


INTERIM = {
    "ack": {"text": ACK},
    "multipart": {"text": "Checking the API|then the database", "parts": 2},
    "formatted": {"text": "<b>Checking</b> the services", "format": "html"},
    "document": {"text": "", "kind": "document"},
    "reaction": {"text": "", "kind": "reaction"},
    "accepted": {"text": ACK, "state": "accepted"},
    "uncertain": {"text": ACK, "state": "uncertain"},
}


@pytest.mark.parametrize("interim,version", [
    *((name, 0) for name in INTERIM), ("ack", 1), ("ack", 2), ("reaction", 2)])
def test_a_final_after_interim_speech_is_new_speech_once(room, interim, version):
    room.h.script = [send(**INTERIM[interim]), finish(answer=FINAL)]
    body = turn(room, version=version)
    if version == 2:  # the early envelope is custody, never this author's speech
        assert (body["status"], body["outcome"], body["text"], body["continuation_ref"]) == (
            "continuing", "deferred", "", body["turn_ref"])
    else:
        assert (body["status"], body["outcome"], body["text"]) == ("completed", "message", FINAL)
    view, stored = settle(room, body)
    assert (view["status"], view["outcome"], view["text"]) == ("completed", "message", FINAL)
    assert stored["metadata"]["presence_outcome"] == "message"
    assert stored["metadata"]["presence_result_text"] == FINAL
    interim_texts = INTERIM[interim]["text"].split("|") if INTERIM[interim].get("parts") else [INTERIM[interim]["text"]]
    assert room.transport.texts() == [*interim_texts, FINAL], "the final is spoken once, after the interim send"
    calls = room.h.calls
    assert turn(room, version=version) == body and room.h.calls == calls == 2  # replay: no generation, no resend
    settle(room, body)
    assert room.transport.texts().count(FINAL) == 1
    pointer = pointer_of(room, body["turn_ref"])
    confirmed = [text for text in interim_texts if text and INTERIM[interim].get("state") != "uncertain"]
    # A version-2 transport can report the final's own receipt before the turn writes this pointer.
    assert pointer["message"] == FINAL and [text for text in pointer["transport_sends"] if text != FINAL] == confirmed


@pytest.mark.parametrize("version", [0, 1, 2])
def test_explicit_silence_after_a_tool_delivered_the_result_adds_nothing(room, version):
    room.h.script = [send(text=RESULT), finish(silent=True, answer=NOTE)]
    body = turn(room, version=version)
    view, stored = settle(room, body)
    assert (view["outcome"], view["text"], view["output_ref"]) == ("silent", "", "")
    assert room.transport.texts() == [RESULT]
    assert stored["result"] == NOTE and stored["wake_silent"] is True  # the words stay internal
    pointer = pointer_of(room, body["turn_ref"])
    assert (pointer["outcome"], pointer["message"], pointer["finish_note"]) == ("silent", "", NOTE)
    assert pointer["transport_sends"] == [RESULT]


@pytest.mark.parametrize("version", [0, 1, 2])
def test_a_quiet_finish_keeps_an_admitted_handoff_pollable(room, version):
    from ouroboros.tools.control_routing import _finish_swarm_handoff

    def admit_then_finish(_messages):
        # Promotion admission is outside this fixture: retain its scheduled child and receipt.
        ctx = room.h.agents[-1].tools._ctx
        write_task_result(room.data, CHILD, "scheduled", delegation_role="root", root_task_id=CHILD,
                          description="Independent audit", metadata={"presence": dict(ctx.task_metadata["presence"])})
        _finish_swarm_handoff(ctx, {"task_id": CHILD}, "Admission receipt", status="scheduled")
        return finish(silent=True, answer=NOTE)

    room.h.script = [admit_then_finish]
    body = turn(room, version=version)
    if version == 2:
        assert (body["status"], body["outcome"], body["text"]) == ("continuing", "deferred", "")
    else:  # versions 0 and 1 learn of the child only from this terminal body
        assert (body["status"], body["outcome"], body["text"], body["work_ref"]) == ("completed", "deferred", "", CHILD)
    view, stored = settle(room, body)
    assert (view["outcome"], view["text"], view["child_work_ref"]) == ("deferred", "", CHILD)
    assert stored["metadata"]["presence_outcome"] == "deferred" and stored["metadata"]["presence_result_text"] == ""
    assert CHILD in room.transport.work and room.transport.texts() == []
    room.transport.poll()  # the scheduled child is still pending
    assert room.transport.texts() == []
    write_task_result(room.data, CHILD, "completed", result=CHILD_ANSWER, terminal_origin="model_final",
                      metadata=load_task_result(room.data, CHILD)["metadata"])
    room.transport.poll()
    room.transport.poll()
    assert room.transport.texts() == [CHILD_ANSWER] and not room.transport.work
    pointer = pointer_of(room, body["turn_ref"])
    assert (pointer["outcome"], pointer["message"], pointer["finish_note"], pointer["work_ref"]) == (
        "deferred", "", NOTE, CHILD)


@pytest.mark.parametrize("version", [0, 1, 2])
def test_a_hold_at_replay_keeps_admitted_work_observable(room, version):
    """A lost first answer, then a pause: the retry's speech stays held, the child's result still arrives."""
    import time

    from ouroboros.tools.control_routing import _finish_swarm_handoff

    def admit_then_answer(_messages):
        ctx = room.h.agents[-1].tools._ctx
        write_task_result(room.data, CHILD, "scheduled", delegation_role="root", root_task_id=CHILD,
                          description="Independent audit", metadata={"presence": dict(ctx.task_metadata["presence"])})
        _finish_swarm_handoff(ctx, {"task_id": CHILD}, "Admission receipt", status="scheduled")
        return finish(answer=FINAL)

    room.h.script = [admit_then_answer]
    ref = turn(room, version=version)["turn_ref"]  # this answer never reaches the transport's custody
    deadline = time.monotonic() + 60
    while (load_task_result(room.data, ref) or {}).get("status") != "completed":
        assert time.monotonic() < deadline, "the author never settled"
        time.sleep(0.02)
    places.edit_place(room.data, room.place.place_id, "set_state", author="owner", state="paused")
    replay = turn(room, version=version)
    if version != 2:
        assert (replay["outcome"], replay["text"], replay["work_ref"]) == ("deferred", "", CHILD)
    room.transport.submit("chat:a:1", replay)
    for _ in range(3):
        room.transport.poll()
    assert room.transport.texts() == []
    places.edit_place(room.data, room.place.place_id, "set_state", author="owner", state="active")
    write_task_result(room.data, CHILD, "completed", result=CHILD_ANSWER, terminal_origin="model_final",
                      metadata=load_task_result(room.data, CHILD)["metadata"])
    for _ in range(3):
        room.transport.poll()
    assert room.transport.texts() == [CHILD_ANSWER] and room.h.calls == 1


@pytest.mark.parametrize("hold", ["silent", "paused", "advises", "safety"])
def test_current_place_state_and_safety_still_govern_a_final_after_an_ack(room, monkeypatch, hold):
    from ouroboros import dialogue_store

    checked = []
    if hold == "advises":  # the level the owner's sheet calls Advises, stamped when the focus starts
        monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "observe")

    def check(name, subject, **_kw):
        if name != "place_reply":  # tool-call checks stay the harness's
            return True, ""
        checked.append(subject["text"])
        return (hold != "safety" or subject["text"] != FINAL), "Reconsider: the backup time is unverified."

    monkeypatch.setattr("ouroboros.safety.check_safety", check)

    def change_state_then_finish(_messages):
        if hold in {"silent", "paused"}:
            places.edit_place(room.data, room.place.place_id, "set_state", author="owner", state=hold)
        return finish(answer=FINAL)

    revised = "All three services are healthy."
    room.h.script = [send(text=ACK), change_state_then_finish, finish(answer=revised)]
    view, stored = settle(room, turn(room))
    if hold == "safety":  # the flag returned to the mind; its revision is the reply
        assert (view["outcome"], view["text"]) == ("message", revised)
        assert checked == [FINAL, revised] and room.transport.texts() == [ACK, revised]
        assert "Reconsider: the backup time is unverified." in json.dumps(room.h.inputs[-1])
        return
    assert (view["outcome"], view["text"]) == ("silent", "") and room.transport.texts() == [ACK]
    assert checked == [] and room.h.calls == 2
    held = [entry.row for entry in dialogue_store.iter_rows(room.data, "chat") if entry.row.get("held")]
    assert [row["held_text"] for row in held] == [FINAL]
    assert held[0]["text"].startswith("Reply held:")
    assert stored["metadata"]["presence_outcome"] == "silent"


def test_receipts_of_another_task_or_thread_neither_select_nor_count(room):
    room.h.script = [send(text="Elsewhere in the thread", thread_id="9", request_id="thread"),
                     send(text="Another task's send", foreign_task="other-task", request_id="other"),
                     finish(answer=FINAL)]
    body = turn(room)
    view, _stored = settle(room, body)
    assert (view["outcome"], view["text"]) == ("message", FINAL)
    assert room.transport.texts() == ["Elsewhere in the thread", "Another task's send", FINAL]
    assert [row["text"] for row in receipts(room, body["turn_ref"]) if row["transport"]["origin"]["kind"] == "tool"] == [
        "Elsewhere in the thread"]  # attributed to the task, but another conversation key
    assert pointer_of(room, body["turn_ref"])["transport_sends"] == []


def test_late_and_unknown_confirmations_never_resend_or_rewrite_the_answer(room):
    room.h.script = [send(text=ACK, state="uncertain", request_id="ack"), finish(answer=FINAL)]
    body = turn(room, version=1)
    view, _stored = settle(room, body)
    assert (view["outcome"], view["text"]) == ("message", FINAL) and room.transport.texts() == [ACK, FINAL]
    ref = body["turn_ref"]
    late = receipt("tool:ack", "0", ACK, task_id=ref, source="chat:a:1")
    for _ in range(2):  # the uncertain acknowledgement is confirmed late, then reported again
        assert room.client.post("/presence/delivery", headers=HEADERS, json=late).status_code == 200
    calls = room.h.calls
    assert turn(room, version=1) == body
    assert room.client.get(f"/presence/work/{ref}", headers=HEADERS).json() == view
    room.transport.submit("chat:a:1", body)
    room.transport.poll()
    assert room.h.calls == calls and room.transport.texts() == [ACK, FINAL]


def promoted_root(room, monkeypatch, responses):
    """A root promoted from this conversation keeps the speaker carrier: the real loop and pipeline run it.

    A response may be a callable of the messages, like a turn's script; ``None`` is a provider that returns nothing."""
    import queue

    from ouroboros import agent_task_pipeline as pipeline, loop
    from ouroboros.tools.registry import ToolRegistry

    room.h.script = [finish(answer="I will compile the report as separate work.")]
    body = turn(room)
    room.transport.submit("chat:a:1", body)
    root_id = "promoted-place-root"
    presence = dict(load_task_result(room.data, body["turn_ref"])["metadata"]["presence"])
    metadata = {"presence": presence, "delegation_role": "root"}
    registry = ToolRegistry(repo_dir=room.h.repo, drive_root=room.data)
    registry._ctx.task_id, registry._ctx.task_metadata, registry._ctx.is_direct_chat = root_id, metadata, False
    registry.register(ToolEntry(name="bridge_send", schema=SEND_SCHEMA, handler=room.transport.tool,
                                timeout_sec=30, safety_policy="skip"))
    task = {"id": root_id, "type": "task", "chat_id": 1, "text": "Compile the report", "metadata": metadata,
            "delegation_role": "root", "_skip_post_task_synthesis": True}
    write_task_result(room.data, root_id, "running", metadata=metadata, delegation_role="root", root_task_id=root_id)
    responses, inputs = iter(responses), []

    def respond(_llm, messages, *_a, **_kw):
        inputs.append(json.loads(json.dumps(messages)))
        response = next(responses, None)
        return (response(messages) if callable(response) else response), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    text, usage, trace = loop.run_llm_loop(
        [{"role": "user", "content": "Compile the report"}], registry, SimpleNamespace(default_model=lambda: "test-model"),
        room.data / "logs", lambda *_a, **_kw: None, queue.Queue(), task_id=root_id, drive_root=room.data)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    pipeline.emit_task_results(SimpleNamespace(drive_root=room.data, repo_dir=room.h.repo), None, None, [], task,
                               text, usage, trace, 0.0, room.data / "logs", ctx=registry._ctx)
    return SimpleNamespace(id=root_id, trace=trace, inputs=inputs, stored=load_task_result(room.data, root_id))


@pytest.mark.parametrize("silent", [False, True])
def test_a_promoted_place_root_speaks_a_new_final_after_an_interim_send_or_stays_silent(room, monkeypatch, silent):
    """Promoted work answers its own conversation through the poll; it chose silence or new words."""
    root = promoted_root(room, monkeypatch, [
        send(text=RESULT if silent else ACK, request_id="root"),
        finish(silent=True, answer=NOTE) if silent else finish(answer="The compiled report is attached above.")])
    assert not any(row.get("is_error") for row in root.trace["tool_calls"]), root.trace["tool_calls"]
    before = room.transport.texts()
    room.transport.work[root.id] = "chat:a:1"
    room.transport.poll()
    if silent:
        assert root.stored["metadata"]["presence_outcome"] == "silent" and root.stored["result"] == NOTE
        assert room.transport.texts() == before
    else:
        assert root.stored["metadata"]["presence_outcome"] == "message"
        assert room.transport.texts()[len(before):] == ["The compiled report is attached above."]
    assert not room.transport.work


def test_a_refused_place_root_is_never_told_the_frozen_completion_names(room, monkeypatch):
    """The real loop's refusal round: the flag returns and the next finish is ordinary ``finish_task``."""
    def check(name, subject, **_kw):
        return (name != "place_reply" or subject["text"] != FINAL), "Reconsider: the backup time is unverified."

    monkeypatch.setattr("ouroboros.safety.check_safety", check)
    root = promoted_root(room, monkeypatch, [finish(answer=FINAL), finish(silent=True, answer=NOTE)])
    assert "Reconsider: the backup time is unverified." in json.dumps(root.inputs[-1])
    assert "presence_finish" not in json.dumps(root.inputs) and "tool_delivered" not in json.dumps(root.inputs)
    assert root.stored["metadata"]["presence_outcome"] == "silent"


def test_a_refused_silent_finish_stays_silent_when_the_provider_then_fails(room, monkeypatch):
    """The refused selection leaves no accepted request, only its quiet candidate; the forced exit
    keeps that candidate quiet, so its words are the finish note and never the conversation's."""
    monkeypatch.setattr("ouroboros.loop_model_call._route_candidates", lambda *_a: [])  # no fallback route
    root = promoted_root(room, monkeypatch, [  # finishing beside an unread result is refused
        {"content": None, "tool_calls": [*send(text=RESULT, request_id="root")["tool_calls"],
                                         *finish(silent=True, answer=NOTE)["tool_calls"]]},
        None])
    assert [row["reason"].split(":")[0] for row in root.trace["completion_refusals"]] == ["unobserved_tool_results"]
    assert root.stored["wake_silent"] is True and root.stored["result"] == NOTE
    before = room.transport.texts()
    room.transport.work[root.id] = "chat:a:1"
    room.transport.poll()
    assert (root.stored["metadata"]["presence_outcome"], root.stored["metadata"]["presence_result_text"]) == (
        "silent", "")
    assert room.transport.texts() == before and before[-1] == RESULT and NOTE not in before


def test_a_fresh_forced_answer_after_a_silent_selection_is_new_speech(room, monkeypatch):
    """A selected silent finish meets the supervisor's finalization grace before its exit: the
    wrap-up's fresh words replace the quiet candidate, so the terminal is ordinary speech, once."""
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

    wrapped = "Out of time: the report is compiled; two figures are unverified."

    def finish_as_the_deadline_arrives(_messages):
        write_owner_message(room.data, "deadline", task_id="promoted-place-root", msg_id="grace",
                            kind=KIND_FINALIZE_NOW)
        return finish(silent=True, answer=NOTE)

    root = promoted_root(room, monkeypatch, [finish_as_the_deadline_arrives, {"content": wrapped}])
    assert root.trace["forced_finalization"]["reason_code"] == "finalization_grace"
    assert root.stored.get("wake_silent") is not True and root.stored["result"] == wrapped
    room.transport.work[root.id] = "chat:a:1"
    room.transport.poll()
    assert (root.stored["metadata"]["presence_outcome"], root.stored["metadata"]["presence_result_text"]) == (
        "message", wrapped)
    assert room.transport.texts().count(wrapped) == 1 and NOTE not in room.transport.texts()


@pytest.mark.parametrize("form", ["native", "text"])
def test_a_place_root_at_its_round_limit_keeps_the_silence_it_selected(room, monkeypatch, form):
    """A forced rail reads finish_task's one contract: the retained carrier's silence, as a call or the
    same request in its text, stays internal words and no speech; the rail stays the recorded cause."""
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    request = {"action": "finish", "silent": True, "answer": NOTE}
    root = promoted_root(room, monkeypatch, [
        send(text=RESULT, request_id="root"),
        finish(silent=True, answer=NOTE) if form == "native" else {"content": json.dumps(request)}])
    assert root.trace["forced_completion_request"]["status"] == "selected"
    assert (root.stored["wake_silent"], root.stored["result"]) == (True, NOTE)
    assert (root.stored["reason_code"], root.stored["terminal_origin"]) == ("round_limit", "model_final")
    assert (root.stored["metadata"]["presence_outcome"], root.stored["metadata"]["presence_result_text"]) == (
        "silent", "")
    before = room.transport.texts()
    room.transport.work[root.id] = "chat:a:1"
    room.transport.poll()
    assert room.transport.texts() == before and before[-1] == RESULT and NOTE not in before


@pytest.mark.parametrize("form", ["native", "text"])
def test_a_place_root_gets_no_sentence_for_its_human_at_its_round_limit(room, monkeypatch, form):
    """Silence is the carrier's; the sentence stays a self-started run's, on a forced rail as in the handler."""
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    request = {"action": "finish", "silent": True, "summary": "A line for the human.", "answer": NOTE}
    root = promoted_root(room, monkeypatch, [
        send(text=RESULT, request_id="root"),
        finish(silent=True, summary=request["summary"], answer=NOTE) if form == "native"
        else {"content": json.dumps(request)}])
    fact = root.trace["forced_completion_request"]
    assert fact["status"] == "refused" and fact["reason"].startswith("invalid_completion_request: silent and summary")
    assert root.stored.get("wake_silent") is not True and not root.stored.get("wake_summary")
    assert NOTE not in room.transport.texts() and NOTE not in root.stored["metadata"]["presence_result_text"]


def test_a_voided_place_final_points_at_finish_task_and_frozen_registries_keep_their_words(tmp_path, monkeypatch):
    from ouroboros.loop_delivery import _void_presence_completion
    from ouroboros.place_reply import prepare_focus_reply
    from ouroboros.presence_context import presence_finish_not_accepted_note

    monkeypatch.setattr("ouroboros.tool_access.canonical_data_root", lambda _ctx: tmp_path)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    place = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="a", conversation_id="r")
    # The real setter on a promoted root's retained carrier, which names no wake cause.
    carrier = {"binding_id": "b1", "place_id": place.place_id, "transport_skill": "bridge",
               "event": {"conversation_key": KEY, "source_event_id": "event1"}}
    ctx = SimpleNamespace(task_metadata={"presence": carrier}, task_id="promoted", drive_root=tmp_path)
    assert prepare_focus_reply(ctx, FINAL)[0] and ctx._presence_completion["outcome"] == "message"
    messages = []
    _void_presence_completion(ctx, messages)
    note = messages[-1]["content"]
    assert note.startswith("[FINISH_NOT_ACCEPTED]") and "finish_task" in note and "silence is valid" in note
    assert "presence_finish" not in note and "tool_delivered" not in note
    assert ctx._presence_completion is None and ctx._presence_completion_accepted is False
    legacy = SimpleNamespace(task_id="frozen", drive_root=tmp_path, task_metadata={"presence": {
        "binding_id": "b1", "event": {"conversation_key": KEY, "source_event_id": "event1"}}})
    frozen = presence_finish_not_accepted_note(legacy, {"outcome": "message"})
    assert frozen.startswith("[PRESENCE_FINISH_NOT_ACCEPTED]") and "tool_delivered" in frozen


@pytest.mark.parametrize("origin,refused,handoff", [
    ("host_notice", False, False), ("host_notice", False, True),
    ("model_final", True, False), ("model_final", True, True)])
def test_forced_or_refused_words_never_ride_speech_or_a_handoff(tmp_path, monkeypatch, origin, refused, handoff):
    """A host-authored final is never speech or a note; a Safety refusal at a forced exit holds its words."""
    from ouroboros.presence_delivery import PresenceDeliveryRecorder
    from ouroboros.presence_runner import build_presence_result_event
    from tests.test_place_focus import task_for

    place = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="a", conversation_id="r")
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_runner import PresenceTurnEvent
    admission = admit_presence_turn(drive_root=tmp_path, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r")
    event = PresenceTurnEvent("event1", "chat", "a", "r", "", KEY, {"platform_actor_id": "person"}, {}, {}, "hi",
                              facts=("mention",))
    monkeypatch.setattr("ouroboros.presence_context._own_work_section", lambda *_args: "")
    task = task_for((tmp_path, place, admission, event))
    write_task_result(tmp_path, task["id"], "running", metadata=task["metadata"])
    PresenceDeliveryRecorder(tmp_path).record("bridge", receipt("tool:ack", "0", ACK, task_id=task["id"], source="event1"))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (not refused, "Reconsider"))
    ctx = SimpleNamespace(task_metadata=task["metadata"], task_id=task["id"], drive_root=tmp_path,
                          _swarm_handoff_attempt={"status": "scheduled", "task_id": CHILD} if handoff else None)
    monkeypatch.setattr("ouroboros.tool_access.canonical_data_root", lambda _ctx: tmp_path)
    row = build_presence_result_event(task, "Words that must not be spoken.", ctx, terminal_origin=origin,
                                      retain_scheduled_handoff=handoff)
    assert (row["outcome"], row["text"], row["work_ref"]) == (("deferred", "", CHILD) if handoff else ("silent", "", ""))
    assert "finish_note" not in row and task["metadata"]["presence_result_text"] == ""
