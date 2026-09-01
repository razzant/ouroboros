from __future__ import annotations

import json
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor


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
