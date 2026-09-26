from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


from ouroboros.presence_admission import PresenceAdmission
from ouroboros.presence_authority import (
    PresenceCapabilityCeiling,
    PresenceToolGrant,
    presence_ceiling_payload,
)
from ouroboros.presence_bindings import PresenceEndpoint
from ouroboros.presence_context import build_presence_context_section
from ouroboros.presence_runner import (
    PresenceTurnEvent,
    PresenceTurnGate,
    run_presence_turn,
)
from ouroboros.task_results import write_task_result


def _terminal(drive_root, task, reply):
    """The real pipeline's durable terminal: the Host reads it back, the envelope alone never answers."""
    write_task_result(pathlib.Path(drive_root), task["id"], "completed", metadata=task["metadata"], result=reply)


def _admission() -> PresenceAdmission:
    endpoint = PresenceEndpoint("telegram", "bot-1", "room-1", "topic-1")
    ceiling = PresenceCapabilityCeiling(
        skill_name="community-helper",
        skill_content_hash="a" * 64,
        profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64,
        selection_fingerprint="d" * 64,
        model_slot="main",
        inline_max_rounds=10,
        tool_grants=(PresenceToolGrant("chat_history"),),
        resource_grants=(),
        digest="0" * 64,
    )
    payload = presence_ceiling_payload(ceiling)
    ceiling = PresenceCapabilityCeiling(**{**ceiling.__dict__, "digest": payload["digest"]})
    return PresenceAdmission(
        binding_id="1" * 32,
        transport_skill="telegram-bot",
        behavior_skill="community-helper",
        origin=endpoint,
        destination=endpoint,
        instructions="Participate helpfully.",
        context_topics=("social-memory",),
        model_slot="main",
        inline_max_rounds=10,
        skill_content_hash="a" * 64,
        profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64,
        selection_fingerprint="d" * 64,
        capability_ceiling=ceiling,
    )


def _event() -> PresenceTurnEvent:
    return PresenceTurnEvent(
        source_event_id="telegram:bot-1:42",
        provider="telegram",
        account_id="bot-1",
        conversation_id="room-1",
        thread_id="topic-1",
        conversation_key="telegram:bot-1:room-1:topic-1",
        actor={"platform_actor_id": "user-7", "username": "alex"},
        conversation={"title": "Community"},
        message={"message_id": "42"},
        text="Hello",
    )


def test_runner_builds_bounded_fresh_task_and_logs_shared_dialogue(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    captured = {}

    class Agent:
        def handle_task(self, task):
            captured.update(task)
            _terminal(data, task, "Hi")
            return [{"type": "presence_result", "outcome": "message", "text": "Hi", "work_ref": ""}]

    result = run_presence_turn(
        admission=_admission(),
        event=_event(),
        repo_dir=repo,
        drive_root=data,
        agent_factory=lambda **_kwargs: Agent(),
        gate=PresenceTurnGate(2),
    )

    assert result.outcome == "message"
    assert result.text == "Hi"
    assert captured["_presence_turn"] is True
    assert captured["metadata"]["inline_max_rounds"] == 10
    assert captured["metadata"]["presence"]["binding_id"] == "1" * 32
    assert captured["task_contract"]["capability_ceiling"]["digest"] == _admission().capability_ceiling.digest
    rows = [json.loads(line) for line in (data / "logs" / "chat.jsonl").read_text().splitlines()]
    assert [row["direction"] for row in rows] == ["in", "out"]
    assert rows[0]["transport"]["actor"]["platform_actor_id"] == "user-7"
    assert rows[0]["presence_provenance"] == {
        "binding_id": "1" * 32,
        "transport_skill": "telegram-bot",
        "behavior_skill": "community-helper",
        "profile_fingerprint": "b" * 64,
        "state_fingerprint": "c" * 64,
        "selection_fingerprint": "d" * 64,
        "source_event_id": "telegram:bot-1:42",
        "conversation_key": "telegram:bot-1:room-1:topic-1",
        "provider": "telegram",
        "account_id": "bot-1",
        "conversation_id": "room-1",
        "thread_id": "topic-1",
        "actor_id": "user-7",
    }
    assert rows[1]["presence_provenance"] == rows[0]["presence_provenance"]


def test_presence_initial_attachment_rejection_defaults_to_partial_staging(tmp_path):
    """В25c (capinv-447): one bad attachment no longer discards the whole set —
    the staged sibling rides into the task and the rejected row stays disclosed."""
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    staged_source = tmp_path / "available.txt"
    staged_source.write_text("available", encoding="utf-8")
    seen_tasks = []

    class Agent:
        def handle_task(self, task):
            seen_tasks.append(task)
            _terminal(data, task, "ok")
            return [{"type": "presence_result", "outcome": "message", "text": "ok"}]

    result = run_presence_turn(
        admission=_admission(),
        event=_event(),
        repo_dir=repo,
        drive_root=data,
        staged_files=(staged_source, tmp_path / "missing.txt"),
        agent_factory=lambda **_kwargs: Agent(),
        gate=PresenceTurnGate(2),
    )

    assert result is not None
    assert seen_tasks, "partial staging must let the turn proceed"
    manifest = seen_tasks[0]["attachments"]
    assert [row["status"] for row in manifest] == ["staged", "rejected"]
    assert manifest[1]["reason"] == "source_missing"
    assert pathlib.Path(manifest[0]["abs_path"]).is_file()


def test_presence_context_loads_declared_topic_and_completion_contract(tmp_path):
    topic = tmp_path / "memory" / "knowledge" / "social-memory.md"
    topic.parent.mkdir(parents=True)
    topic.write_text("Alex prefers concise replies.", encoding="utf-8")
    section = build_presence_context_section(
        tmp_path,
        {
            "behavior_skill": "community-helper",
            "profile_fingerprint": "b" * 64,
            "instructions": "Participate helpfully.",
            "context_topics": ["social-memory"],
            "event": {"source_event_id": "event-1"},
        },
    )
    assert "Participate helpfully." in section
    assert "Alex prefers concise replies." in section
    assert "presence_finish" in section


def test_configured_presence_parallelism_is_bounded(monkeypatch, tmp_path):
    import ouroboros.presence_runner as runner

    monkeypatch.setenv("OUROBOROS_PRESENCE_MAX_ACTIVE", "3")
    runner._GATES.clear()
    assert runner._configured_gate(tmp_path) is runner._configured_gate(tmp_path)
    assert runner._configured_gate(tmp_path)._max_active == 3


def test_presence_gate_coordinates_distinct_instances_via_install_state(tmp_path):
    active = 0
    maximum = 0
    guard = threading.Lock()
    start = threading.Barrier(6)
    gates = [PresenceTurnGate(2, state_root=tmp_path) for _ in range(6)]

    def run(index):
        nonlocal active, maximum
        start.wait()

        def callback():
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with guard:
                active -= 1
            return PresenceTurnResult("silent", "", f"task-{index}")

        return gates[index].run(f"conversation-{index}", callback)

    from ouroboros.presence_runner import PresenceTurnResult

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, range(6)))
    assert maximum == 2
    assert len(results) == 6


def test_presence_gate_serializes_same_conversation_across_instances(tmp_path):
    active = 0
    maximum = 0
    guard = threading.Lock()
    start = threading.Barrier(4)
    gates = [PresenceTurnGate(4, state_root=tmp_path) for _ in range(4)]

    def run(index):
        nonlocal active, maximum
        start.wait()

        def callback():
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return None

        gates[index].run("same-conversation", callback)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(4)))
    assert maximum == 1


def test_presence_turn_is_live_for_liveness_readers_but_never_an_owner_target(monkeypatch, tmp_path):
    """The presence-local live set shields a running turn from orphan reconcile; the owner
    census (update drain, steer roots, /api/state, settings idle, wake gating) never sees it."""
    from types import SimpleNamespace

    from ouroboros import presence_runner
    from ouroboros.consciousness import BackgroundConsciousness
    from ouroboros.gateway.settings import _has_running_agent_tasks, _has_started_agent_tasks
    from ouroboros.gateway.state import _direct_turns_snapshot_safe
    from ouroboros.server_routing_context import _active_direct_roots
    from ouroboros.task_status import _is_stale_orphan_running_task
    from ouroboros.utils import append_jsonl
    from supervisor import workers
    from supervisor.active_activity import get_direct_activity_registry

    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "queue_snapshot.json").write_text(
        '{"ts": "2027-01-15T08:00:00+00:00", "pending": [], "running": []}', encoding="utf-8")
    append_jsonl(tmp_path / "logs" / "events.jsonl", {"ts": "2026-05-28T00:00:02+00:00", "type": "worker_boot"})
    orphan_row = {"status": "running", "ts": "2026-05-28T00:00:00+00:00"}
    registry = get_direct_activity_registry()
    registry.clear()
    mind = object.__new__(BackgroundConsciousness)
    mind._last_wake_task_id = ""
    main_actor = SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(), _busy=True, _current_task_id="main-turn",
        _accepting_owner_messages=True, _current_task_metadata={}, _current_task_text="owner text",
        _task_started_ts=1.0, _current_chat_id=1)
    armed, seen = [], {}

    def observe(task_id):
        return {
            "orphan": _is_stale_orphan_running_task(tmp_path, task_id, orphan_row),
            "drain": workers.drain_repo_writers(timeout=0), "roots": [row["task_id"] for row in _active_direct_roots(None)],
            "turns": [turn["id"] for turn in workers.direct_chat_turns()],
            "state": [row["activity_id"] for row in _direct_turns_snapshot_safe()],
            "busy": (_has_running_agent_tasks(), _has_started_agent_tasks()), "wakes": mind.live_turns(),
        }

    class Agent:  # shaped like a real native actor: only the absent census entry hides it
        _owner_message_admission_lock = threading.Lock()
        _busy = _accepting_owner_messages = True
        _current_task_metadata, _current_task_text, _task_started_ts, _current_chat_id = {}, "hi", 1.0, 1

        def handle_task(self, task):
            self._current_task_id = task["id"]
            seen.update(live=presence_runner.presence_turn_is_live(task["id"]), registered=registry.get(task["id"]),
                        armed=workers.arm_direct_chat_turn(task["id"], armed.append), alone=observe(task["id"]))
            registry.register("main-turn", 1, actor=main_actor)  # positive control: an owner turn IS census-visible
            try:
                seen["with_main"] = observe(task["id"])
            finally:
                registry.unregister("main-turn")
            _terminal(tmp_path, task, "")
            return [{"type": "presence_result", "outcome": "silent", "text": "", "work_ref": ""}]

    try:
        result = run_presence_turn(admission=_admission(), event=_event(), repo_dir=tmp_path, drive_root=tmp_path,
                                   agent_factory=lambda **_kwargs: Agent(), gate=PresenceTurnGate(1))
        assert seen["live"] is True and seen["registered"] is None and seen["armed"] is None and armed == []
        assert seen["alone"] == {"orphan": False, "drain": [], "roots": [], "turns": [], "state": [],
                                 "busy": (False, False), "wakes": ("", False)}
        assert seen["with_main"] == {"orphan": False, "drain": ["main-turn"], "roots": ["main-turn"],
                                     "turns": ["main-turn"], "state": ["main-turn"], "busy": (True, True),
                                     "wakes": ("", True)}
        # Once the turn returns the set is empty and the same orphan evidence reconciles.
        assert presence_runner.presence_turn_is_live(result.task_id) is False
        assert result.task_id not in presence_runner._LIVE_PRESENCE_TASKS
        assert _is_stale_orphan_running_task(tmp_path, result.task_id, orphan_row) is True
    finally:
        registry.clear()


def _pointer_turn(tmp_path, event_id, row, *, thread="topic-1", version=0, captured=None, during=None,
                  status="completed", terminal_origin="model_final"):
    """One executed turn with a durable terminal row, as the real pipeline leaves it."""
    from dataclasses import replace

    from ouroboros.presence_bindings import conversation_key
    from ouroboros.task_results import write_task_result

    event = replace(_event(), source_event_id=event_id, thread_id=thread, delivery_reporting_version=version,
                    conversation_key=conversation_key("telegram", "bot-1", "room-1", thread))

    class Agent:
        def handle_task(self, task):
            if captured is not None:
                captured.append(task)
            write_task_result(tmp_path, task["id"], "running", metadata=task["metadata"])
            if during is not None:
                during(task)
            write_task_result(tmp_path, task["id"], status, result=row.get("text", ""),
                              terminal_origin=terminal_origin, metadata={
                                  **task["metadata"], "presence_outcome": row["outcome"],
                                  "presence_result_text": row.get("text", ""),
                                  "presence_work_ref": row.get("work_ref", "")})
            return [{"type": "presence_result", "task_id": task["id"], **row}]

    return run_presence_turn(admission=_admission(), event=event, repo_dir=tmp_path, drive_root=tmp_path,
                             agent_factory=lambda **_kwargs: Agent(), gate=PresenceTurnGate(1))


def test_previous_turn_pointer_names_message_deferred_and_silent_turns(tmp_path):
    captured: list = []
    first = _pointer_turn(tmp_path, "e1", {"outcome": "message", "text": "Hi there", "message": "Hi there"},
                          captured=captured)
    _pointer_turn(tmp_path, "e2", {"outcome": "deferred", "text": "On it", "message": "On it",
                                   "work_ref": "presence-work-9"}, captured=captured)
    _pointer_turn(tmp_path, "e3", {"outcome": "silent", "text": "", "message": ""}, captured=captured)
    _pointer_turn(tmp_path, "e4", {"outcome": "silent", "text": ""}, captured=captured)
    contexts = [task["metadata"]["presence"] for task in captured]
    assert "previous_turn" not in contexts[0]
    previous = contexts[1]["previous_turn"]
    assert (previous["task_id"], previous["outcome"], previous["message"], previous["delivery"]) == (
        first.task_id, "message", "Hi there", "unknown")  # a v0 transport never confirms delivery
    assert (contexts[2]["previous_turn"]["outcome"], contexts[2]["previous_turn"]["work_ref"]) == (
        "deferred", "presence-work-9")
    sections = [build_presence_context_section(tmp_path, context) for context in contexts]
    assert "Previous turn" not in sections[0]
    assert f"Previous turn in this conversation (task {first.task_id}, finished " in sections[1]
    assert 'UTC, outcome message, delivery unknown): "Hi there".' in sections[1]
    assert '"On it". Its deferred work (task presence-work-9) has no task row.' in sections[2]
    assert "outcome silent, delivery unknown): nothing sent." in sections[3]


def test_previous_turn_is_per_conversation_and_never_rewritten_by_a_replay(tmp_path):
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import _previous_turn_path

    captured: list = []
    older = _pointer_turn(tmp_path, "e1", {"outcome": "message", "text": "Room reply"})
    _pointer_turn(tmp_path, "e2", {"outcome": "message", "text": "Other"}, thread="topic-2", captured=captured)
    assert "previous_turn" not in captured[-1]["metadata"]["presence"]  # another thread's pointer is not read
    room = _previous_turn_path(tmp_path, conversation_key("telegram", "bot-1", "room-1", "topic-1"))
    misplaced = _previous_turn_path(tmp_path, conversation_key("telegram", "bot-1", "room-1", "topic-3"))
    misplaced.write_bytes(room.read_bytes())  # a pointer naming another conversation is ignored
    _pointer_turn(tmp_path, "e3", {"outcome": "silent", "text": ""}, thread="topic-3", captured=captured)
    assert "previous_turn" not in captured[-1]["metadata"]["presence"]
    newer = _pointer_turn(tmp_path, "e4", {"outcome": "message", "text": "Newer"}, captured=captured)
    assert captured[-1]["metadata"]["presence"]["previous_turn"]["task_id"] == older.task_id
    pointer = room.read_bytes()
    assert json.loads(pointer)["task_id"] == newer.task_id
    # The older event's retry is a cached replay: no execution, and the newest pointer stays.
    replay = _pointer_turn(tmp_path, "e1", {"outcome": "message", "text": "Must not run"}, captured=captured)
    assert replay == older and len(captured) == 3 and room.read_bytes() == pointer


def test_previous_turn_shows_what_a_transport_tool_delivered(tmp_path):
    from ouroboros.presence_delivery import PresenceDeliveryRecorder

    recorder = PresenceDeliveryRecorder(tmp_path)

    def send(task):
        recorder.record("telegram-bot", {
            "schema_version": 1, "delivery_id": f"send:{task['id']}", "part_id": "0", "state": "delivered",
            "provider": "telegram", "account_id": "bot-1", "conversation_id": "room-1", "thread_id": "topic-1",
            "text": "Schedule: Mon 10:00", "format": "markdown", "message": {"provider_message_id": "7"},
            "origin": {"kind": "tool", "task_id": task["id"]},
        })

    captured: list = []
    _pointer_turn(tmp_path, "e1", {"outcome": "tool_delivered", "text": "", "message": "Sent the schedule"},
                  version=1, during=send)
    _pointer_turn(tmp_path, "e2", {"outcome": "message", "text": "Anything else?", "message": "Anything else?"},
                  version=1, captured=captured)
    _pointer_turn(tmp_path, "e3", {"outcome": "tool_delivered", "text": ""}, captured=captured)
    _pointer_turn(tmp_path, "e4", {"outcome": "silent", "text": ""}, captured=captured)
    delivered, authored, unrecorded = (task["metadata"]["presence"]["previous_turn"] for task in captured)
    assert (delivered["transport_sends"], delivered["delivery"], delivered["message"]) == (
        ["Schedule: Mon 10:00"], "confirmed", "Sent the schedule")
    assert authored["delivery"] == "authored"  # a v1 reply's receipt arrives only after the turn returns
    assert (unrecorded["transport_sends"], unrecorded["delivery"]) == ([], "unknown")
    sections = [build_presence_context_section(tmp_path, task["metadata"]["presence"]) for task in captured]
    assert 'delivery confirmed): "Schedule: Mon 10:00" / finish note "Sent the schedule".' in sections[0]
    assert "delivered via transport tool (content unrecorded)." in sections[2]
    # An early ack through a tool plus a final reply through the adapter: only the ack is confirmed so far.
    _pointer_turn(tmp_path, "e5", {"outcome": "message", "text": "Final: 10:00", "message": "Final: 10:00"},
                  version=1, during=send)
    _pointer_turn(tmp_path, "e6", {"outcome": "silent", "text": ""}, version=1, captured=captured)
    partly = captured[-1]["metadata"]["presence"]["previous_turn"]
    assert (partly["transport_sends"], partly["message"], partly["delivery"]) == (
        ["Schedule: Mon 10:00"], "Final: 10:00", "partly confirmed")
    assert 'delivery partly confirmed): "Schedule: Mon 10:00" / "Final: 10:00".' in build_presence_context_section(
        tmp_path, captured[-1]["metadata"]["presence"])


def test_deferred_handoff_keeps_internal_finish_note_out_of_prior_speech(tmp_path):
    """A tool-send note stays context even when owed work makes the turn deferred."""
    note = "helper failed; the table was already sent"
    captured = []
    first = _pointer_turn(tmp_path, "note-e1", {"outcome": "deferred", "text": "", "message": "",
                                                 "finish_note": note, "work_ref": "owed-task"})
    assert (first.outcome, first.text, first.work_ref) == ("deferred", "", "owed-task")
    _pointer_turn(tmp_path, "note-e2", {"outcome": "silent", "text": ""}, captured=captured)
    pointer = captured[-1]["metadata"]["presence"]["previous_turn"]
    assert pointer["message"] == "" and pointer["finish_note"] == note
    section = build_presence_context_section(tmp_path, captured[-1]["metadata"]["presence"])
    assert 'finish note "helper failed; the table was already sent"' in section
    assert '): "helper failed; the table was already sent"' not in section


def test_legacy_deferred_pointer_is_checked_against_canonical_reply_before_quoting(tmp_path):
    """Old deferred tool-send notes shared `message` with speech; source separates them."""
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import _previous_turn_path

    captured = []
    note = "helper failed, result already sent"
    _pointer_turn(tmp_path, "old-note", {"outcome": "deferred", "text": "", "message": note,
                                         "work_ref": "owed"})
    path = _previous_turn_path(tmp_path, conversation_key("telegram", "bot-1", "room-1", "topic-1"))
    historical = path.read_bytes()
    _pointer_turn(tmp_path, "after-note", {"outcome": "silent", "text": ""}, captured=captured)
    previous = captured[-1]["metadata"]["presence"]["previous_turn"]
    assert (previous["message"], previous["finish_note"]) == ("", note)
    assert b'"finish_note"' not in historical  # source remains unchanged; this is a read projection
    assert 'finish note "helper failed, result already sent"' in build_presence_context_section(
        tmp_path, captured[-1]["metadata"]["presence"])

    _pointer_turn(tmp_path, "old-speech", {"outcome": "deferred", "text": "Still working", "message": "Still working",
                                           "work_ref": "owed"})
    _pointer_turn(tmp_path, "after-speech", {"outcome": "silent", "text": ""}, captured=captured)
    spoken = captured[-1]["metadata"]["presence"]["previous_turn"]
    assert spoken["message"] == "Still working" and "finish_note" not in spoken


def test_previous_turn_reports_the_fate_of_its_deferred_work(tmp_path):
    """"Work continues" only while the child runs; a finished child's answer or failure is stated instead."""
    from ouroboros.task_results import write_task_result

    captured: list = []
    _pointer_turn(tmp_path, "e1", {"outcome": "deferred", "text": "Looking into it", "work_ref": "presence-work-1"})
    write_task_result(tmp_path, "presence-work-1", "running", metadata={"source": "presence"})
    _pointer_turn(tmp_path, "e2", {"outcome": "silent", "text": ""}, captured=captured)
    assert "Work continues as task presence-work-1 (status running)." in build_presence_context_section(
        tmp_path, captured[-1]["metadata"]["presence"])
    _pointer_turn(tmp_path, "e3", {"outcome": "deferred", "text": "On it", "work_ref": "presence-work-2"})
    write_task_result(tmp_path, "presence-work-2", "completed", result="Report ready", terminal_origin="model_final",
                      metadata={"source": "presence", "presence_outcome": "message", "presence_result_text": "Report ready"})
    _pointer_turn(tmp_path, "e4", {"outcome": "silent", "text": ""}, captured=captured)
    previous = captured[-1]["metadata"]["presence"]["previous_turn"]
    assert (previous["work_status"], previous["work_result"]) == ("completed", "Report ready")
    assert 'Its deferred work (task presence-work-2) completed and answered: "Report ready".' in (
        build_presence_context_section(tmp_path, captured[-1]["metadata"]["presence"]))
    _pointer_turn(tmp_path, "e5", {"outcome": "deferred", "text": "Trying", "work_ref": "presence-work-3"})
    write_task_result(tmp_path, "presence-work-3", "failed", result="boom", metadata={"source": "presence"})
    _pointer_turn(tmp_path, "e6", {"outcome": "silent", "text": ""}, captured=captured)
    section = build_presence_context_section(tmp_path, captured[-1]["metadata"]["presence"])
    assert "Its deferred work (task presence-work-3) ended failed." in section and "boom" not in section
    # A host-authored terminal (salvage) is never spoken to the correspondent, but the record is shown.
    _pointer_turn(tmp_path, "e7", {"outcome": "deferred", "text": "Digging", "work_ref": "presence-work-4"})
    write_task_result(tmp_path, "presence-work-4", "completed", result="Salvaged: " + "x" * 400,
                      terminal_origin="host_salvage", metadata={"source": "presence"})
    _pointer_turn(tmp_path, "e8", {"outcome": "silent", "text": ""}, captured=captured)
    previous = captured[-1]["metadata"]["presence"]["previous_turn"]
    assert previous["work_result"] == "" and previous["work_record"].endswith(" …(truncated)")
    section = build_presence_context_section(tmp_path, captured[-1]["metadata"]["presence"])
    assert "completed silently; the host recorded an undelivered result: \"Salvaged: xxx" in section


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_previous_turn_pointer_is_rebuilt_by_the_replay_of_a_turn_that_lost_it(tmp_path, monkeypatch, status):
    """A turn killed between its terminal write and its pointer write: the retry replays and repairs the
    pointer; an authored reply on a failed task is speech too and repairs the same way."""
    from ouroboros import presence_runner
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import _previous_turn_path

    room = _previous_turn_path(tmp_path, conversation_key("telegram", "bot-1", "room-1", "topic-1"))
    older = _pointer_turn(tmp_path, "e1", {"outcome": "message", "text": "Old answer"}, version=1)
    real_write, skipped = presence_runner._write_previous_turn, []

    def killed_before_pointer_write(*args, **kwargs):  # the process died right after the terminal write
        if not skipped:
            skipped.append(args[2])  # (drive_root, conversation_key, task_id, ...)
            return None
        real_write(*args, **kwargs)

    monkeypatch.setattr(presence_runner, "_write_previous_turn", killed_before_pointer_write)
    _pointer_turn(tmp_path, "e2", {"outcome": "message", "text": "New answer"}, version=1, status=status)
    lost = json.loads(room.read_text(encoding="utf-8"))
    assert lost["task_id"] == older.task_id and skipped  # the durable row completed, the pointer did not follow
    captured: list = []
    replay = _pointer_turn(tmp_path, "e2", {"outcome": "message", "text": "Must not run"}, version=1, captured=captured)
    assert replay.text == "New answer" and captured == []  # a cached replay, no execution
    repaired = json.loads(room.read_text(encoding="utf-8"))
    assert (repaired["task_id"], repaired["message"], repaired["delivery"]) == (replay.task_id, "New answer", "authored")
    assert repaired["finished_at"] > lost["finished_at"]  # completion order, from the durable row's stamp
    _pointer_turn(tmp_path, "e3", {"outcome": "silent", "text": ""}, version=1, captured=captured)
    assert captured[-1]["metadata"]["presence"]["previous_turn"]["task_id"] == replay.task_id
    newest = room.read_bytes()
    # The older turn's replay finds a newer pointer and leaves it alone.
    assert _pointer_turn(tmp_path, "e1", {"outcome": "message", "text": "Must not run"}, version=1,
                         captured=captured) == older
    assert room.read_bytes() == newest and len(captured) == 1


def test_pointer_write_failure_does_not_fail_an_answered_turn(tmp_path, monkeypatch, caplog):
    """The pointer is a projection: a full disk under it never turns a delivered answer into a 400."""
    from ouroboros import presence_runner
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import _previous_turn_path

    room = _previous_turn_path(tmp_path, conversation_key("telegram", "bot-1", "room-1", "topic-1"))
    real_write = presence_runner.atomic_write_json

    def full_disk(path, payload):
        if pathlib.Path(path) == room:
            raise OSError("disk full")
        real_write(path, payload)

    monkeypatch.setattr(presence_runner, "atomic_write_json", full_disk)
    with caplog.at_level(logging.WARNING, logger="ouroboros.presence_runner"):
        answered = _pointer_turn(tmp_path, "e1", {"outcome": "message", "text": "Still delivered"})
    assert answered.text == "Still delivered" and not room.exists()
    assert any("previous-turn pointer not written" in record.getMessage() for record in caplog.records)
    monkeypatch.setattr(presence_runner, "atomic_write_json", real_write)
    captured: list = []
    _pointer_turn(tmp_path, "e2", {"outcome": "silent", "text": ""}, captured=captured)
    assert "previous_turn" not in captured[0]["metadata"]["presence"]  # the gap is a gap, not an invented fact
    assert json.loads(room.read_text(encoding="utf-8"))["task_id"] != answered.task_id


@pytest.mark.parametrize("origin", ["", "host_notice", "host_salvage"])
def test_a_host_authored_failure_never_repairs_the_pointer(tmp_path, monkeypatch, origin):
    """A failed row the host wrote (no reply of the model's own) replays silent and is not a turn to point at."""
    from ouroboros import presence_runner
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import _previous_turn_path

    room = _previous_turn_path(tmp_path, conversation_key("telegram", "bot-1", "room-1", "topic-1"))
    older = _pointer_turn(tmp_path, "e1", {"outcome": "message", "text": "Old answer"}, version=1)
    before = room.read_bytes()
    monkeypatch.setattr(presence_runner, "_write_previous_turn", lambda *a, **k: None)  # killed before the write
    _pointer_turn(tmp_path, "e2", {"outcome": "message", "text": "Host text"}, version=1, status="failed",
                  terminal_origin=origin)
    monkeypatch.undo()
    captured: list = []
    replay = _pointer_turn(tmp_path, "e2", {"outcome": "message", "text": "Must not run"}, version=1, captured=captured)
    assert (replay.outcome, replay.text, captured) == ("silent", "", [])  # cached silence, no execution
    assert room.read_bytes() == before and json.loads(before)["task_id"] == older.task_id
