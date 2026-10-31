"""#1536 through the real Host boundary: /presence/turn, /presence/work and initiate_presence.

The harness of ``test_presence_continuation`` supplies the real loop, review coordinator,
direct wait, mailbox settlement and pipeline terminal, and its kept trigger: the author
reviews its own result, leaves that review for an addressed message and parks on it. Here
the Host's own gate, execution registry, admission, in-flight budget and endpoints decide;
the admitted behavior selects ``review_change``, so the Host's admission is the authority
that lets the author review its result.
"""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import functools
import json
import pathlib
import time

import pytest
from starlette.testclient import TestClient

from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_runner import run_presence_turn
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_host_service_api import _seed_token
from tests.test_presence_continuation import (
    ANSWER, NEW_WORDS, author, deliver, finish, harness as harness, parked, resumed, wait_for,
)

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


def _seed_reviewing_behavior(data_dir: pathlib.Path) -> str:
    """``test_host_service_api``'s fixture behavior, bound to any room of the account, whose owner also
    selected ``review_change``: its turns may review their own result (a result review takes no workspace
    binding, so no resource is selected)."""
    from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint
    from tests.frozen_presence_bindings import new_presence_binding_id, save_presence_binding
    from tests.frozen_presence_capabilities import (
        PresenceSelection, PresenceState, PresenceToolTarget, presence_state_fingerprint, save_presence_state,
    )
    from tests.frozen_presence_profile import parse_presence_profile, presence_request_fingerprint
    from ouroboros.skill_loader import SkillReviewState, load_skill, save_enabled, save_review_state

    skill_dir = data_dir / "skills" / "external" / "community-helper"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: community-helper\ndescription: Neutral helper.\nversion: 0.1.0\n"
        "type: instruction\npresence:\n  instructions: Participate helpfully.\n"
        "  capability_requests:\n    - id: history\n      kind: tool\n      required: true\n"
        "      purpose: Read history.\n    - id: review\n      kind: tool\n      required: true\n"
        "      purpose: Review my own result.\n---\n# Helper\n",
        encoding="utf-8",
    )
    loaded = load_skill(skill_dir, data_dir)
    assert loaded is not None
    save_enabled(data_dir, loaded.name, True)
    save_review_state(data_dir, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    profile = parse_presence_profile(loaded.manifest, skill_dir)
    assert profile is not None
    targets = {"history": PresenceToolTarget("builtin", "chat_history"),
               "review": PresenceToolTarget("builtin", "review_change")}
    save_presence_state(
        data_dir, loaded.name,
        PresenceState(tuple(PresenceSelection(presence_request_fingerprint(request), targets[request.request_id])
                            for request in profile.capability_requests)),
        expected_state_fingerprint=presence_state_fingerprint(PresenceState()),
    )
    binding = save_presence_binding(data_dir, PresenceBinding(
        new_presence_binding_id(), "telegram-bot", loaded.name, PresenceEndpoint("telegram", "bot-1", "*", ""),
        PresenceEndpoint("telegram", "bot-1", "room-1", "topic-1")))
    return binding.binding_id


@pytest.fixture
def host(harness, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("supervisor.consciousness_carriers.autonomous_work_enabled", lambda *_: True, raising=False)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available", "remaining_usd": 10})
    h = harness
    monkeypatch.setenv("OUROBOROS_PRESENCE_MAX_ACTIVE", "1")
    _seed_token(h.data, skill="telegram-bot", token=_TOKEN, permissions=["presence"], manifest_permissions=["presence"])
    h.binding = _seed_reviewing_behavior(h.data)  # any room/thread of the account
    h.quick = set()

    def runner(**kwargs):
        quick = kwargs["event"].source_event_id in h.quick
        return run_presence_turn(repo_dir=h.repo, drive_root=h.data,
                                 agent_factory=lambda **_kw: _Quick(h.data) if quick else h.Agent(), **kwargs)

    with TestClient(create_host_service_app(h.data, presence_runner=runner)) as client:
        h.client = client
        try:
            yield h
        finally:
            h.release.set()
            executions = client.app.state.host_service_context.presence_turns
            wait_for(lambda: not executions.live(), timeout=90, what="Host author teardown")


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
    identity = host.client.get("/identity", headers=_HEADERS).json()
    assert identity["presence_continuation_version"] == 1  # deployed consumers compare the scalar exactly
    assert identity["presence_continuation_versions"] == [1, 2]


def test_v2_early_answer_lends_nothing_and_the_real_park_returns_the_reservation_once(host, monkeypatch):
    import threading
    from ouroboros.conversation_lease import live_author

    h = host
    original = h.Agent.handle_task

    def opens_intake(self, task):
        # ``OuroborosAgent._handle_task_scoped`` fires it once its prepared loop opens intake; this harness
        # agent runs the loop directly, so it fires it here, twice: a repeat answers nothing again.
        callback = getattr(self, "intake_opened_callback", None)
        for _ in range(2):
            callback(task)
        return original(self, task)

    monkeypatch.setattr(h.Agent, "handle_task", opens_intake)
    proceed, step = threading.Event(), author(deliver)

    def held(messages):
        assert proceed.wait(60), "the test never let the author's model call return"
        return step(messages)

    h.script = held
    reservations = h.client.app.state.host_service_context._inflight
    first = turn(h, 42, version=2).json()
    ref = first["continuation_ref"]
    assert (first["status"], first["continuation_version"], first["turn_ref"]) == ("continuing", 2, ref)
    record = load_task_result(h.data, ref)["presence_continuation"]
    assert "lent_at" not in record and "first_lent_at" not in record and record["continuation_version"] == 2
    assert reservations["telegram-bot:presence"] == 1 and live_author(record["conversation_key"], drive_root=h.data) == ref
    proceed.set()
    wait_for(lambda: parked(h, ref), what="the review park")
    wait_for(lambda: reservations["telegram-bot:presence"] == 0, what="the park's return of the reservation")
    record = load_task_result(h.data, ref)["presence_continuation"]
    assert record["lent_at"] and record["first_lent_at"] and record["initial"] == {
        "status": "continuing", "outcome": "deferred", "text": "", "output_ref": "", "work_ref": ""}
    h.quick.add("telegram:bot-1:43")  # the lent conversation and slot run the next event at once
    assert turn(h, 43, NEW_WORDS, version=2).json()["text"] == "Noted."
    assert turn(h, 42, version=2).json() == first  # still the write-once envelope
    h.release.set()
    wait_for(lambda: poll(h, ref).status_code == 200, timeout=60, what="the late result")
    late = poll(h, ref).json()
    assert (late["text"], late["continuation_version"]) == (ANSWER, 2)
    # The resumed author's final model send carried event 43 whole in its reentry note: that is the last
    # input its answer covers, so the reply anchors there, with 43's exact captured message.
    assert "(event telegram:bot-1:43, taken by turn" in str(h.inputs[-1]) and NEW_WORDS in str(h.inputs[-1])
    assert late["reply_anchor"] == {"source_event_id": "telegram:bot-1:43", "message": {"message_id": "43"}}
    wait_for(lambda: reservations["telegram-bot:presence"] == 0, what="no second return")
    assert reservations["telegram-bot:presence"] == 0


def test_v1_consumer_gets_the_envelope_the_next_event_runs_and_the_late_result_polls_once(host):
    h = host

    def after_wake(messages):
        assert NEW_WORDS in resumed(messages)
        return deliver()

    h.script = author(after_wake)
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
    assert h.calls == 3 and len(h.reviews) == 1


def test_v0_consumer_keeps_the_legacy_shape_and_waits_for_the_terminal(host):
    import threading

    h = host
    h.script = author(deliver)
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
    h.script = author(deliver)
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

    def after_wake(messages):
        assert "your promoted work work-9 is scheduled" in resumed(messages)
        return deliver()

    step = author(after_wake)

    def scripted(messages):
        h.ctxs[0]._swarm_handoff_attempt = {"status": "scheduled", "task_id": "work-9"}
        return step(messages)

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
    assert view.status_code == 200 and view.json()["status"] == "failed" and view.json()["text"] == ""
    assert view.json()["code"] == "presence_attempt_outcome_unknown" and view.json()["work_ref"] == task_id
    legacy = turn(h, 42, version=None)
    assert legacy.status_code == 409 and legacy.json()["code"] == "presence_attempt_outcome_unknown"
    assert h.calls == 0


def test_a_parked_author_still_live_in_this_process_stays_pending_and_is_never_rerun(host, monkeypatch):
    """A positive live observation wins (``presence_recovery.attempt_ended``): even when the park's controller
    witness reads as another process session, an author still live in this process is pending, never shown
    ended; a replay returns the write-once envelope and nothing reruns it. A really lost process is the
    child-Host suite's (``test_presence_continuation_process``), the dead-witness rule ``test_presence_attempt_end``'s."""
    from ouroboros.review_operation import controller_identity

    h = host
    h.script = author(deliver)
    first = turn(h, 42).json()
    ref = first["continuation_ref"]
    wait_for(lambda: parked(h, ref), what="the review park")
    assert load_task_result(h.data, ref)["presence_continuation"]["author_controller"] == controller_identity()
    assert poll(h, ref).status_code == 202  # its controller is this live process: still pending
    with monkeypatch.context() as replaced:
        replaced.setattr("ouroboros.process_custody.current_custody_session_id", lambda: "a-new-process-session")
        from ouroboros.presence_continuation import continuation_status

        assert continuation_status(load_task_result(h.data, ref), h.data) == "running"
        assert poll(h, ref).status_code == 202
        assert turn(h, 42).json() == first  # replay returns the write-once envelope; nothing reruns the author
    assert h.calls == 2
    # The projection wrote nothing: the (live) author finishes as usual.
    h.release.set()
    wait_for(lambda: poll(h, ref).status_code == 200, timeout=60, what="the author")
    assert (poll(h, ref).json()["status"], poll(h, ref).json()["text"]) == ("completed", ANSWER) and h.calls == 3


def _initiate(h, *args):
    from tests.presence_legacy_helpers import legacy_tools as get_tools
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
    monkeypatch.setattr("tests.presence_legacy_helpers._INITIATION_MARGIN_SEC", 0.5)
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


def test_an_initiated_continuing_envelope_reaches_the_initiating_call(host, monkeypatch):
    """Finding 4: the initiating call is a version-1 consumer; a continuing author's first envelope reaches it."""
    h = host
    monkeypatch.setattr("ouroboros.presence_runner.run_presence_turn",
                        functools.partial(run_presence_turn, agent_factory=lambda **_kw: h.Agent()))
    h.script = author(deliver)
    result = _initiate(h, "Report the status.", "release")
    assert (result["status"], result["outcome"], result["text"], result["output_ref"]) == ("continuing", "deferred", "", "")
    assert set(result) == _INITIATION_FIELDS and result["delivered"] is False
    assert result["continuation_ref"] == result["turn_ref"] == _proactive_id(h, "Report the status.", "release")
    record = load_task_result(h.data, result["turn_ref"])["presence_continuation"]
    assert record["initial"]["status"] == "continuing" and record["outputs"] == []
    h.release.set()
    from ouroboros.presence_runner import PROACTIVE_TURNS

    wait_for(lambda: not PROACTIVE_TURNS.live(), timeout=60, what="the initiated author")
    stored = load_task_result(h.data, result["turn_ref"])
    # Nothing was spoken while it waited: the same author's terminal speaks its answer.
    assert stored["status"] == "completed" and stored["metadata"]["presence_result_text"] == ANSWER
    assert h.calls == 3 and len(h.reviews) == 1


def test_reporting_v1_receipts_own_outgoing_history_and_reach_the_resuming_author(host):
    h = host
    second_ref = _turn_id(h, 43)

    def after_wake(messages):
        note = resumed(messages)
        # The receipt of the newer turn's reply, reported while this author waited, is a fact of its reentry.
        assert f"delivery delivered for turn {second_ref}" in note and "Noted." in note
        return deliver()

    h.script = author(after_wake)
    first = turn(h, 42, reporting=1).json()
    assert (first["status"], first["outcome"], first["text"], first["output_ref"]) == ("continuing", "deferred", "", "")
    assert first["delivery_reporting_version"] == 1
    h.quick.add("telegram:bot-1:43")
    second = turn(h, 43, NEW_WORDS, reporting=1).json()
    assert (second["status"], second["text"], second["turn_ref"]) == ("completed", "Noted.", second_ref)
    assert second["delivery_reporting_version"] == 1
    chat = h.data / "logs" / "chat.jsonl"
    assert not [row for row in dialogue_test.log_rows(chat) if row.get("direction") == "out"]  # mode 1: receipts own it

    def receipt(output_ref, task_id, source_event_id, text):
        return {"schema_version": 1, "delivery_id": f"auto:{output_ref}", "part_id": "0", "state": "delivered",
                "provider": "telegram", "account_id": "bot-1", "conversation_id": "room-1", "thread_id": "topic-1",
                "text": text, "format": "markdown", "message": {"provider_message_id": source_event_id[-2:]},
                "origin": {"kind": "automatic", "task_id": task_id, "source_event_id": source_event_id}}

    # The stub turn mints no output ref of its own (it never reaches the pipeline): its receipt names the turn.
    sent = receipt(second["output_ref"] or second_ref, second_ref, "telegram:bot-1:43", "Noted.")
    assert h.client.post("/presence/delivery", headers=_HEADERS, json=sent).status_code == 200
    h.release.set()
    wait_for(lambda: poll(h, first["continuation_ref"]).status_code == 200, timeout=60, what="the terminal")
    late = poll(h, first["continuation_ref"]).json()
    assert (late["outcome"], late["text"]) == ("message", ANSWER) and late["output_ref"].startswith("presence-output-")
    assert late["delivery_reporting_version"] == 1 and late["outputs"] == []  # nothing was released while it waited
    # The late terminal's speech is not host-written outgoing history either; only receipts are.
    assert [row["text"] for row in dialogue_test.log_rows(chat) if row.get("direction") == "out"] == ["Noted."]
    final = receipt(late["output_ref"], first["turn_ref"], "telegram:bot-1:42", ANSWER)
    assert h.client.post("/presence/delivery", headers=_HEADERS, json=final).status_code == 200
    assert [row["text"] for row in dialogue_test.log_rows(chat) if row.get("direction") == "out"] == ["Noted.", ANSWER]
