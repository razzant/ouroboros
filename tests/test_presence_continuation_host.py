"""#1536 through the real Host boundary: /presence/turn, /presence/work and initiate_presence.

The harness of ``test_presence_continuation`` supplies the real loop, acceptance
coordinator, direct wait, mailbox settlement and pipeline terminal; here the Host's
own gate, execution registry, admission, in-flight budget and endpoints decide.
"""

from __future__ import annotations

import functools
import json
import time

import pytest
from starlette.testclient import TestClient

from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_runner import run_presence_turn
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_host_service_api import _seed_presence_behavior, _seed_token
from tests.test_presence_continuation import ANSWER, NEW_WORDS, finish, harness as harness, wait_for

_TOKEN = "presence-token"
_HEADERS = {"X-Skill-Token": _TOKEN}
pytestmark = pytest.mark.serial


class _Quick:
    """A same-conversation turn that answers at once (no effects, no review)."""

    def __init__(self, root):
        self.root = root

    def handle_task(self, task):
        write_task_result(self.root, task["id"], "completed", metadata=task["metadata"], result="Noted.",
                          terminal_origin="model_final")
        return [{"type": "presence_result", "outcome": "message", "text": "Noted.", "work_ref": ""}]


@pytest.fixture
def host(harness, monkeypatch):
    h = harness
    monkeypatch.setenv("OUROBOROS_PRESENCE_MAX_ACTIVE", "1")
    _seed_token(h.data, skill="telegram-bot", token=_TOKEN, permissions=["presence"], manifest_permissions=["presence"])
    h.binding = _seed_presence_behavior(h.data, account_wide=True)  # any room/thread of the account
    h.quick = set()

    def runner(**kwargs):
        quick = kwargs["event"].source_event_id in h.quick
        return run_presence_turn(repo_dir=h.repo, drive_root=h.data,
                                 agent_factory=lambda **_kw: _Quick(h.data) if quick else h.Agent(), **kwargs)

    with TestClient(create_host_service_app(h.data, presence_runner=runner)) as client:
        h.client = client
        yield h


def turn(h, number, text="Please prepare the status report", version=1, thread="topic-1", reporting=0, transport_queue=None):
    payload = {"binding_id": h.binding, "event": {
        "source_event_id": f"telegram:bot-1:{number}", "provider": "telegram", "account_id": "bot-1",
        "conversation_id": "room-1", "thread_id": thread, "conversation_key": "ignored",
        "actor": {"platform_actor_id": "user-7", "username": "alex"}, "conversation": {"title": "Community"},
        "message": {"message_id": str(number)}, "text": text}}
    if version is not None:
        payload["continuation_version"] = version
    if reporting:
        payload["delivery_reporting_version"] = reporting
    if transport_queue is not None:
        payload["event"]["conversation"]["transport_queue"] = transport_queue
    return h.client.post("/presence/turn", headers=_HEADERS, json=payload)


def poll(h, ref):
    return h.client.get(f"/presence/work/{ref}", params={"binding_id": h.binding}, headers=_HEADERS)


def test_identity_advertises_the_continuation_version(host):
    assert host.client.get("/identity", headers=_HEADERS).json()["presence_continuation_version"] == 1


def test_v1_consumer_gets_the_envelope_the_next_event_runs_and_the_late_result_polls_once(host):
    h = host

    def scripted(messages):
        if any("[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")) for row in messages):
            assert any(NEW_WORDS in str(row.get("content")) for row in messages)
            return finish("final", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256)
        return finish("nominate", message=ANSWER)

    h.script = scripted
    first = turn(h, 42).json()
    assert first["status"] == "continuing" and first["continuation_version"] == 1
    assert (first["outcome"], first["text"], first["output_ref"]) == ("deferred", "", "")
    assert first["continuation_ref"] == first["turn_ref"] and first["work_ref"] == ""
    pending = poll(h, first["continuation_ref"])
    assert pending.status_code == 202 and pending.json()["status"] == "pending" and pending.json()["outputs"] == []
    # Cap 1: the next event of the same conversation is answered while the author waits.
    h.quick.add("telegram:bot-1:43")
    second = turn(h, 43, NEW_WORDS).json()
    assert (second["status"], second["text"]) == ("completed", "Noted.")
    assert turn(h, 42).json() == first  # replay: the identical write-once envelope, nothing rerun
    h.release.set()
    wait_for(lambda: poll(h, first["continuation_ref"]).status_code == 200, timeout=60, what="the late result")
    late = poll(h, first["continuation_ref"]).json()
    assert (late["status"], late["outcome"], late["text"]) == ("completed", "message", ANSWER)
    assert late["output_ref"].startswith("presence-output-") and late["child_work_ref"] == ""
    assert poll(h, first["continuation_ref"]).json() == late  # a repeated poll is the same fact
    assert turn(h, 42).json() == first  # still the envelope after the author ended
    assert h.calls == 2 and len(h.reviews) == 1


def test_v0_consumer_keeps_the_legacy_shape_and_waits_for_the_terminal(host):
    import threading

    h = host
    h.script = lambda messages: (finish("final", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256)
                                 if any("[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")) for row in messages)
                                 else finish("nominate", message=ANSWER))
    box = {}
    worker = threading.Thread(target=lambda: box.update(body=turn(h, 42, version=None).json()), daemon=True)
    worker.start()
    wait_for(lambda: (load_task_result(h.data, _turn_id(h, 42)) or {}).get("presence_continuation"), what="the park")
    # The legacy request still waits, but the conversation and the active slot are free.
    h.quick.add("telegram:bot-1:43")
    assert turn(h, 43, NEW_WORDS, version=None).json()["text"] == "Noted."
    assert "body" not in box
    h.release.set()
    worker.join(60)
    assert box["body"] == {"ok": True, "status": "completed", "outcome": "message", "text": ANSWER,
                           "turn_ref": _turn_id(h, 42), "work_ref": "", "delivery_reporting_version": 0}


def _turn_id(h, number):
    from ouroboros.presence_runner import presence_turn_task_id

    return presence_turn_task_id(h.binding, f"telegram:bot-1:{number}")


def test_six_parked_same_skill_turns_return_their_in_flight_reservations(host):
    h = host
    h.script = lambda messages: (finish("final", message=ANSWER)
                                 if any("[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")) for row in messages)
                                 else finish("nominate", message=ANSWER))
    # Six authors of one skill (its in-flight budget is five) all park; none needs a 429 retry.
    bodies = [turn(h, 100 + index, thread=f"topic-{index}").json() for index in range(6)]
    assert [body["status"] for body in bodies] == ["continuing"] * 6
    h.release.set()
    for body in bodies:
        wait_for(lambda ref=body["continuation_ref"]: poll(h, ref).status_code == 200, timeout=90, what="each result")
        assert poll(h, body["continuation_ref"]).json()["text"] == ANSWER


def test_promoted_child_and_continuing_parent_keep_two_independent_refs(host, monkeypatch):
    h = host
    original = h.Agent.handle_task

    def with_child(self, task):
        write_task_result(h.data, "work-9", "scheduled", delegation_role="root", root_task_id="work-9",
                          description="Compile Q2", metadata={"presence": dict(task["metadata"]["presence"])})
        return original(self, task)

    monkeypatch.setattr(h.Agent, "handle_task", with_child)

    def scripted(messages):
        h.ctxs[0]._swarm_handoff_attempt = {"status": "scheduled", "task_id": "work-9"}
        if any("[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")) for row in messages):
            note = next(str(row["content"]) for row in messages if "[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")))
            assert "your promoted work work-9 is scheduled" in note
            return finish("final", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256)
        return finish("nominate", message=ANSWER)

    h.script = scripted
    first = turn(h, 42).json()
    assert first["status"] == "continuing" and first["work_ref"] == "work-9"
    assert first["continuation_ref"] == first["turn_ref"] != "work-9"
    assert poll(h, "work-9").json()["status"] == "pending" and "outputs" not in poll(h, "work-9").json()
    h.release.set()
    wait_for(lambda: poll(h, first["continuation_ref"]).status_code == 200, timeout=60, what="the parent tail")
    tail = poll(h, first["continuation_ref"]).json()
    assert tail["child_work_ref"] == "work-9" and tail["text"] == ANSWER
    assert poll(h, "work-9").json()["status"] == "pending"  # the child is still owed, polled on its own
    write_task_result(h.data, "work-9", "completed", result="Q2 done.", terminal_origin="model_final",
                      metadata=load_task_result(h.data, "work-9")["metadata"])
    assert poll(h, "work-9").json()["status"] == "completed"


def test_a_crashed_continuing_author_is_shown_interrupted_and_never_rerun(host):
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_runner import _build_task, presence_event_identity
    from tests.test_presence_continuation import event as turn_event

    h = host
    admission = admit_presence_turn(drive_root=h.data, authenticated_transport_skill="telegram-bot",
                                    binding_id=h.binding, global_max_rounds=12)
    event, task_id = turn_event(), _turn_id(h, 42)
    task = _build_task(admission, event, drive_root=h.data, staged_files=(), physical_task_id=task_id)
    identity = presence_event_identity(admission.binding_id, event)
    envelope = {"status": "continuing", "outcome": "deferred", "text": "", "output_ref": "", "work_ref": ""}
    write_task_result(h.data, task_id, "running", metadata=task["metadata"], chat_id=task["chat_id"],
                      source="presence", presence_continuation={
                          "version": 1, "continuation_ref": task_id, "event_identity": identity,
                          "source_event_id": event.source_event_id, "conversation_key": event.conversation_key,
                          "initial": envelope, "outputs": [], "lent_at": "2026-10-06T00:00:00+00:00"})
    # The host's orphan reconciler marks the lost stack; nothing restarts it.
    write_task_result(h.data, task_id, "failed", reason_code="orphaned_running_after_worker_restart",
                      status_reconciled_from="running")
    h.script = lambda _messages: pytest.fail("a crashed author was regenerated")
    replay = turn(h, 42).json()
    assert replay["status"] == "continuing" and replay["continuation_ref"] == task_id
    view = poll(h, task_id)
    assert view.status_code == 200 and view.json()["status"] == "interrupted" and view.json()["text"] == ""
    legacy = turn(h, 42, version=None)
    assert legacy.status_code == 409 and legacy.json()["code"] == "presence_attempt_outcome_unknown"
    assert h.calls == 0


def _initiate(h, *args):
    from ouroboros.tools.presence import get_tools
    from ouroboros.tools.registry import ToolContext

    initiate = next(item for item in get_tools() if item.name == "initiate_presence")
    return json.loads(initiate.handler(ToolContext(repo_dir=h.repo, drive_root=h.data), h.binding, *args))


_INITIATION_FIELDS = {"ok", "status", "outcome", "delivered", "text", "turn_ref", "work_ref", "continuation_ref",
                      "output_ref", "continuation_version"}


def test_initiated_turn_outlives_the_callers_operation_bound(host, monkeypatch):
    import threading

    from ouroboros import usage_accounting
    from ouroboros.model_wait import _CALENDAR, calendar_scope, execution_deadline_scope, monotonic_now

    h = host
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr("ouroboros.tools.presence._INITIATION_MARGIN_SEC", 0.5)
    monkeypatch.setattr("ouroboros.presence_runner.run_presence_turn",
                        functools.partial(run_presence_turn, agent_factory=lambda **_kw: h.Agent()))
    gate = threading.Event()
    callers_scope = usage_accounting.UsageScope(task_id="the-initiating-task", root_task_id="the-initiating-task")
    seen = {}

    def slow(_messages):
        from ouroboros.model_wait import dispatch_deadline_remaining_sec

        assert gate.wait(30)
        # On the turn's own thread, after the caller's bound passed: its one-second operation
        # deadline was never adopted (only the explicit calendar bounds the turn), and the
        # turn's own controls (Stop, calendar, ceiling) report nothing.
        assert dispatch_deadline_remaining_sec() > 86_400
        assert h.ctxs[0].model_wait_context.control_reason() is None
        # The caller's money attribution stays with the caller; its explicit calendar carries.
        seen.update(scope=usage_accounting.current_usage_scope(), calendar=_CALENDAR.get())
        return finish("final", message="Hello from the initiated cycle.")

    h.script = [slow]
    with usage_accounting.usage_scope(callers_scope), calendar_scope("2099-01-01T00:00:00+00:00"), \
            execution_deadline_scope(monotonic_now() + 1.0):
        result = _initiate(h, "Say hello.", "wake")
    # The bounded caller wait expired: the actual refs of the live turn, in the success shape.
    assert result["ok"] is True and result["status"] == "running" and result["delivered"] is False
    assert result["turn_ref"] == result["continuation_ref"] == _proactive_id(h, "Say hello.", "wake")
    assert set(result) == _INITIATION_FIELDS and result["continuation_version"] == 1
    time.sleep(1.0)  # the caller's own operation deadline has now passed
    gate.set()
    from ouroboros.presence_runner import PROACTIVE_TURNS

    wait_for(lambda: not PROACTIVE_TURNS.live(), timeout=30, what="the initiated turn")
    stored = load_task_result(h.data, result["turn_ref"])
    assert stored["status"] == "completed" and stored["metadata"]["presence_result_text"] == "Hello from the initiated cycle."
    assert h.calls == 1
    # The caller's money attribution stayed with the caller; its explicit calendar carried.
    assert seen["scope"] is not callers_scope and "2099-01-01T00:00:00+00:00" in seen["calendar"]


def _proactive_id(h, prompt, dedupe):
    import hashlib

    from ouroboros.presence_runner import presence_turn_task_id

    stable = "\0".join((h.binding, "background", dedupe, prompt))
    return presence_turn_task_id(h.binding, "presence-initiate:" + hashlib.sha256(stable.encode()).hexdigest()[:32])


def test_an_initiated_advisory_release_reaches_the_initiating_call(host, monkeypatch):
    """Finding 4: the initiating call is a version-1 consumer; an early release is not discarded."""
    h = host
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr("ouroboros.presence_runner.run_presence_turn",
                        functools.partial(run_presence_turn, agent_factory=lambda **_kw: h.Agent()))

    def scripted(messages):
        if any("[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")) for row in messages):
            return finish("again", answer_sha256=h.ctxs[0]._presence_released[0]["sha256"])
        return finish("nominate", message=ANSWER, pending_review="finish")

    h.script = scripted
    result = _initiate(h, "Report the status.", "release")
    assert (result["status"], result["outcome"], result["text"]) == ("continuing", "message", ANSWER)
    assert set(result) == _INITIATION_FIELDS
    assert result["output_ref"].startswith("presence-output-") and result["continuation_ref"] == result["turn_ref"]
    record = load_task_result(h.data, result["turn_ref"])["presence_continuation"]
    assert record["outputs"] == [{"output_ref": result["output_ref"], "outcome": "message", "text": ANSWER}]
    h.release.set()
    from ouroboros.presence_runner import PROACTIVE_TURNS

    wait_for(lambda: not PROACTIVE_TURNS.live(), timeout=60, what="the initiated author")
    stored = load_task_result(h.data, result["turn_ref"])
    # The author re-selected its released answer: the terminal says nothing new.
    assert stored["status"] == "completed" and stored["metadata"]["presence_result_text"] == ""
    assert h.calls == 2 and len(h.reviews) == 1


def test_reporting_v1_release_is_receipt_owned_and_reaches_the_resuming_author(host, monkeypatch):
    h = host
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")

    def scripted(messages):
        notes = [str(row["content"]) for row in messages if "[PRESENCE CONVERSATION RESUMED]" in str(row.get("content"))]
        if not notes:
            return finish("nominate", message=ANSWER, pending_review="finish")
        assert "delivery delivered for this turn" in notes[-1] and ANSWER in notes[-1]
        return finish("again", answer_sha256=h.ctxs[0]._presence_released[0]["sha256"])

    h.script = scripted
    first = turn(h, 42, reporting=1).json()
    assert (first["status"], first["outcome"], first["text"]) == ("continuing", "message", ANSWER)
    assert first["delivery_reporting_version"] == 1 and first["output_ref"]
    rows = [json.loads(line) for line in (h.data / "logs" / "chat.jsonl").read_text().splitlines()]
    assert not [row for row in rows if row.get("direction") == "out"]  # mode 1: receipts own outgoing history
    receipt = {"schema_version": 1, "delivery_id": f"auto:{first['output_ref']}", "part_id": "0",
               "state": "delivered", "provider": "telegram", "account_id": "bot-1", "conversation_id": "room-1",
               "thread_id": "topic-1", "text": ANSWER, "format": "markdown", "message": {"provider_message_id": "9"},
               "origin": {"kind": "automatic", "task_id": first["turn_ref"], "source_event_id": "telegram:bot-1:42"}}
    assert h.client.post("/presence/delivery", headers=_HEADERS, json=receipt).status_code == 200
    h.release.set()
    wait_for(lambda: poll(h, first["continuation_ref"]).status_code == 200, timeout=60, what="the terminal")
    late = poll(h, first["continuation_ref"]).json()
    assert (late["outcome"], late["text"], late["output_ref"]) == ("silent", "", "")  # never sent twice
    assert late["outputs"] == [{"output_ref": first["output_ref"], "outcome": "message", "text": ANSWER}]
