"""Presence call mode: one live audio session as one admitted Presence turn."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace

import pytest

from ouroboros import presence_call
from ouroboros.call_protocol import OPENED_BY_SKILL, CallSpec
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_call import (
    MEETING_AUDIO_LABEL,
    PresenceCallOpened,
    PresenceCallRefused,
    open_presence_call,
    presence_call_run,
    record_call_utterance,
)
from ouroboros.presence_capabilities import load_presence_state, presence_state_fingerprint, save_presence_state
from ouroboros.presence_gate import _configured_gate, admit_configured_gate
from ouroboros.presence_runner import _read_previous_turn, presence_chat_id
from ouroboros.presence_runtime import PresenceRuntimeOverrides
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_presence_admission import _binding, _install_behavior, _select_history

KEY = "telegram:public-bot:room-42:thread-7"


def _event_payload(source_event_id: str = "zoom:meeting-1:join", **over) -> dict:
    payload = {
        "source_event_id": source_event_id, "provider": "telegram", "account_id": "public-bot",
        "conversation_id": "room-42", "thread_id": "thread-7", "conversation_key": "ignored by the host",
        "actor": {"platform_actor_id": "meeting-1", "display_name": "Weekly sync"},
        "conversation": {"title": "Weekly sync"}, "message": {}, "text": "Ouroboros joined the meeting",
    }
    payload.update(over)
    return payload


class _Hub:
    """The audio lane's registry seam, recorded: what was opened and closed, in order."""

    def __init__(self, monkeypatch, *, fail_open: bool = False):
        self.opened: list[CallSpec] = []
        self.closed: list[str] = []
        self.order: list[str] = []
        self.fail_open = fail_open
        monkeypatch.setattr(presence_call, "_open_hub", self.open)
        monkeypatch.setattr(presence_call, "_close_hub", self.close)

    def open(self, spec):
        if self.fail_open:
            raise RuntimeError("no audio stack")
        self.opened.append(spec)
        self.order.append("hub_open")

    def close(self, call_id):
        self.closed.append(call_id)
        self.order.append("hub_close")


# A guard against a call that never ends, not a speed claim: a run that admits the turn, opens the
# hub and settles the lease takes several seconds on a loaded machine.
CALL_WAIT_S = 60


class _Driver:
    """A fake live driver: what the agent does once the task carries ``live_call``."""

    def __init__(self, drive_root, hub: _Hub, *, hold: threading.Event | None = None,
                 ending: str = "message"):
        self.drive_root, self.hub, self.hold, self.ending = drive_root, hub, hold, ending
        self.task = None
        self.started = threading.Event()

    def factory(self, **_kwargs):
        return self

    def handle_task(self, task):
        self.task = task
        self.hub.order.append("turn")
        self.started.set()
        if self.hold is not None:
            assert self.hold.wait(CALL_WAIT_S), "the fake call never got its end signal"
        if self.ending == "crash":
            raise RuntimeError("realtime socket died")
        if self.ending == "stop":
            # The owner's Stop reached the driver: it ended the session and the loop
            # recorded the cancelled terminal before returning.
            write_task_result(self.drive_root, task["id"], "cancelled", metadata=task["metadata"],
                              result="Stopped by owner.")
            return []
        write_task_result(self.drive_root, task["id"], "completed", metadata=task["metadata"],
                          result="The call ended.")
        return [{"type": "presence_result", "outcome": "silent", "text": "", "work_ref": ""}]


def _setup(tmp_path, *, execution: str = "call", override: str | None = None):
    drive_root, repo = tmp_path / "data", tmp_path / "repo"
    repo.mkdir()
    skill_dir = _install_behavior(drive_root, execution=execution)
    _select_history(drive_root, skill_dir)
    if override is not None:
        state = load_presence_state(drive_root, "community-helper")
        save_presence_state(drive_root, "community-helper",
                            replace(state, runtime_overrides=PresenceRuntimeOverrides(execution=override)),
                            expected_state_fingerprint=presence_state_fingerprint(state))
    return drive_root, repo, _binding(drive_root)


def _open(drive_root, repo, binding, hub, driver, **over):
    kwargs = dict(drive_root=drive_root, repo_dir=repo, skill_name=binding.transport_skill,
                  binding_id=binding.binding_id, event=_event_payload(), label="Weekly sync",
                  agent_factory=driver.factory)
    kwargs.update(over)
    return open_presence_call(**kwargs)


def _wait_ended(opened: PresenceCallOpened, run) -> None:
    assert run is not None and run.ended.wait(CALL_WAIT_S), f"call {opened.call_id} did not end"


# --- opening -------------------------------------------------------------------------------------

def test_a_call_needs_the_execution_call_runtime(tmp_path, monkeypatch):
    hub = _Hub(monkeypatch)
    drive_root, repo, binding = _setup(tmp_path, execution="")
    with pytest.raises(PresenceCallRefused) as refused:
        _open(drive_root, repo, binding, hub, _Driver(drive_root, hub))
    assert refused.value.code == "presence_call_not_enabled"
    assert "execution" in refused.value.message and hub.opened == []
    # Nothing was taken: the conversation admits a text turn at once.
    assert _configured_gate(drive_root).try_acquire(KEY) is not None


@pytest.mark.parametrize("profile_execution, override", [("call", None), ("", "call")])
def test_an_enabled_binding_opens_one_live_turn_under_its_ceiling(tmp_path, monkeypatch, profile_execution, override):
    monkeypatch.setenv("OUROBOROS_CALL_MODEL_GOOGLE", "gemini-live-test")
    monkeypatch.setenv("OUROBOROS_CALL_VOICE_GOOGLE", "Kore")
    monkeypatch.setenv("OUROBOROS_PRESENCE_CALL_MAX_MINUTES", "45")
    hub = _Hub(monkeypatch)
    drive_root, repo, binding = _setup(tmp_path, execution=profile_execution, override=override)
    driver = _Driver(drive_root, hub)

    opened = _open(drive_root, repo, binding, hub, driver)
    run = presence_call_run(opened.call_id)
    assert isinstance(opened, PresenceCallOpened) and opened.call_id.startswith("call-")
    assert run is not None and run.thread.name == f"presence-call-{opened.call_id}"
    _wait_ended(opened, run)
    assert run.error is None and run.result is not None and run.result.task_id == opened.task_id

    # The hub opened before the turn started and closed once the turn ended.
    assert hub.order == ["hub_open", "turn", "hub_close"]
    assert hub.closed == [opened.call_id]
    spec = hub.opened[0]
    task = driver.task
    admission = admit_presence_turn(drive_root=drive_root, authenticated_transport_skill=binding.transport_skill,
                                    binding_id=binding.binding_id, global_max_rounds=None)
    assert spec == CallSpec(
        call_id=opened.call_id, chat_id=presence_chat_id(KEY), engine="google", model="gemini-live-test",
        voice="Kore", opened_by=OPENED_BY_SKILL, presence_binding_id=binding.binding_id, max_minutes=45,
        label="Weekly sync", client_message_id="zoom:meeting-1:join",
    )
    assert task["chat_id"] == spec.chat_id and task["id"] == opened.task_id
    assert task["metadata"]["live_call"] == spec.as_metadata()
    assert task["metadata"]["inline_max_rounds"] == admission.inline_max_rounds
    assert task["task_contract"]["capability_ceiling"]["digest"] == admission.capability_ceiling.digest
    assert task["task_contract"]["capability_ceiling"]["runtime"]["execution"] == "call"

    # The lease is released, the run collected, and the previous-turn pointer names the call.
    lease = _configured_gate(drive_root).try_acquire(KEY)
    assert lease is not None
    lease.release()
    assert presence_call_run(opened.call_id) is None
    pointer = _read_previous_turn(drive_root, KEY)
    assert pointer["task_id"] == opened.task_id and pointer["outcome"] == "silent"
    rows = [json.loads(line) for line in (drive_root / "logs" / "chat.jsonl").read_text().splitlines()]
    assert [row["direction"] for row in rows] == ["in"]
    assert rows[0]["text"] == "Ouroboros joined the meeting" and rows[0]["task_id"] == opened.task_id


def test_max_minutes_ships_empty_and_refuses_a_malformed_value(tmp_path, monkeypatch):
    hub = _Hub(monkeypatch)
    drive_root, repo, binding = _setup(tmp_path)
    monkeypatch.delenv("OUROBOROS_PRESENCE_CALL_MAX_MINUTES", raising=False)
    driver = _Driver(drive_root, hub)
    opened = _open(drive_root, repo, binding, hub, driver)
    _wait_ended(opened, presence_call_run(opened.call_id))
    assert hub.opened[0].max_minutes is None

    monkeypatch.setenv("OUROBOROS_PRESENCE_CALL_MAX_MINUTES", "soon")
    with pytest.raises(PresenceCallRefused) as refused:
        _open(drive_root, repo, binding, hub, _Driver(drive_root, hub), event=_event_payload("zoom:meeting-2:join"))
    assert refused.value.code == "presence_call_invalid_setting" and len(hub.opened) == 1


@pytest.mark.parametrize("change, code", [
    ({"skill_name": "other-transport"}, "presence_binding_wrong_transport"),
    ({"binding_id": "f" * 32}, "presence_binding_not_found"),
    ({"event": _event_payload(account_id="someone-else")}, "presence_origin_mismatch"),
    ({"event": _event_payload(conversation_id="room-99")}, "presence_origin_mismatch"),
    ({"event": {"text": "no identity"}}, "presence_event_invalid"),
    ({"event": _event_payload(actor={})}, "presence_identity_missing"),
    ({"event": _event_payload(source_event_id="")}, "presence_identity_missing"),
])
def test_binding_transport_and_event_refusals_take_nothing(tmp_path, monkeypatch, change, code):
    hub = _Hub(monkeypatch)
    drive_root, repo, binding = _setup(tmp_path)
    with pytest.raises(PresenceCallRefused) as refused:
        _open(drive_root, repo, binding, hub, _Driver(drive_root, hub), **change)
    assert refused.value.code == code
    assert hub.opened == [] and hub.closed == []
    assert _configured_gate(drive_root).try_acquire(KEY) is not None


def test_a_hub_that_cannot_open_releases_the_lease(tmp_path, monkeypatch):
    hub = _Hub(monkeypatch, fail_open=True)
    drive_root, repo, binding = _setup(tmp_path)
    with pytest.raises(PresenceCallRefused) as refused:
        _open(drive_root, repo, binding, hub, _Driver(drive_root, hub))
    assert refused.value.code == "presence_call_hub_unavailable"
    assert _configured_gate(drive_root).try_acquire(KEY) is not None
    assert not (drive_root / "task_results").exists() or not list((drive_root / "task_results").glob("*.json"))


# --- the gate: busy is answered now, the lease spans the call ------------------------------------

def test_a_busy_conversation_or_cap_refuses_at_once_with_the_active_count(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_PRESENCE_MAX_ACTIVE", "1")
    hub = _Hub(monkeypatch)
    drive_root, repo, binding = _setup(tmp_path)
    gate = _configured_gate(drive_root)
    held = gate.acquire(KEY)  # a text turn of this conversation is live
    try:
        started = time.monotonic()
        with pytest.raises(PresenceCallRefused) as refused:
            _open(drive_root, repo, binding, hub, _Driver(drive_root, hub))
        assert time.monotonic() - started < 1.0, "the door parked in the gate queue"
        assert refused.value.code == "presence_call_busy"
        assert "1 of 1 active" in refused.value.message
    finally:
        held.release()
    other = gate.acquire("telegram:public-bot:elsewhere:")  # the only slot, taken by another conversation
    try:
        with pytest.raises(PresenceCallRefused) as refused:
            _open(drive_root, repo, binding, hub, _Driver(drive_root, hub))
        assert refused.value.code == "presence_call_busy" and "1 of 1 active" in refused.value.message
    finally:
        other.release()
    assert hub.opened == []


@pytest.mark.parametrize("ending", ["message", "stop", "crash"])
def test_the_lease_spans_the_call_and_returns_on_end_stop_or_crash(tmp_path, monkeypatch, ending):
    hub = _Hub(monkeypatch)
    drive_root, repo, binding = _setup(tmp_path)
    hold = threading.Event()
    driver = _Driver(drive_root, hub, hold=hold, ending=ending)
    opened = _open(drive_root, repo, binding, hub, driver)
    run = presence_call_run(opened.call_id)
    assert driver.started.wait(CALL_WAIT_S)
    gate = _configured_gate(drive_root)

    # Live: the conversation is held; a second call and a text turn both meet the lock.
    assert gate.try_acquire(KEY) is None and run.lease.held()
    with pytest.raises(PresenceCallRefused) as refused:
        _open(drive_root, repo, binding, hub, _Driver(drive_root, hub), event=_event_payload("zoom:meeting-1:again"))
    assert refused.value.code == "presence_call_busy"

    async def text_turn_waits():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(admit_configured_gate(drive_root, KEY), 0.3)
    asyncio.run(text_turn_waits())

    hold.set()  # the call ends: normally, by the owner's Stop, or by a driver crash
    _wait_ended(opened, run)
    assert not run.lease.held()
    assert hub.closed == [opened.call_id] and len(hub.opened) == 1
    lease = gate.try_acquire(KEY)
    assert lease is not None, "the lease outlived the call"
    lease.release()
    stored = load_task_result(drive_root, opened.task_id)
    if ending == "message":
        assert run.error is None and run.result.outcome == "silent"
        assert stored["status"] == "completed"
    elif ending == "stop":
        assert run.result is None and getattr(run.error, "code", "") == "presence_attempt_outcome_unknown"
        assert stored["status"] == "cancelled"
    else:
        assert isinstance(run.error, RuntimeError) and stored["status"] == "running"
        # The lost attempt keeps its unknown outcome: the same event is not re-joined,
        # and no second paid session opens for it.
        with pytest.raises(PresenceCallRefused) as again:
            _open(drive_root, repo, binding, hub, _Driver(drive_root, hub))
        assert again.value.code == "presence_attempt_outcome_unknown"
        assert len(hub.opened) == 1


def test_a_settled_event_is_not_reopened(tmp_path, monkeypatch):
    hub = _Hub(monkeypatch)
    drive_root, repo, binding = _setup(tmp_path)
    opened = _open(drive_root, repo, binding, hub, _Driver(drive_root, hub))
    _wait_ended(opened, presence_call_run(opened.call_id))
    with pytest.raises(PresenceCallRefused) as refused:
        _open(drive_root, repo, binding, hub, _Driver(drive_root, hub))
    assert refused.value.code == "presence_call_settled" and len(hub.opened) == 1


# --- dialogue facts and the terminal -------------------------------------------------------------

def test_utterances_become_dialogue_rows_of_the_call(tmp_path, monkeypatch):
    hub = _Hub(monkeypatch)
    drive_root, repo, binding = _setup(tmp_path)
    hold = threading.Event()
    driver = _Driver(drive_root, hub, hold=hold)
    opened = _open(drive_root, repo, binding, hub, driver)
    assert driver.started.wait(CALL_WAIT_S)
    task = driver.task
    assert record_call_utterance(drive_root, task, direction="in", text="Can you summarize the agenda?")
    assert record_call_utterance(drive_root, task, direction="in", text="I agree with Alex.", speaker_label="Dana")
    assert record_call_utterance(drive_root, task, direction="out", text="Sure: three items today.")
    assert not record_call_utterance(drive_root, task, direction="in", text="   ")
    assert not record_call_utterance(drive_root, task, direction="sideways", text="x")
    hold.set()
    _wait_ended(opened, presence_call_run(opened.call_id))

    rows = [json.loads(line) for line in (drive_root / "logs" / "chat.jsonl").read_text().splitlines()]
    assert [(row["direction"], row["text"]) for row in rows] == [
        ("in", "Ouroboros joined the meeting"),
        ("in", "Can you summarize the agenda?"),
        ("in", "I agree with Alex."),
        ("out", "Sure: three items today."),
    ]
    assert rows[1]["sender_label"] == MEETING_AUDIO_LABEL and rows[2]["sender_label"] == "Dana"
    assert rows[1]["transport"]["actor"]["display_name"] == MEETING_AUDIO_LABEL
    assert rows[1]["user_id"] != rows[2]["user_id"] and rows[3]["user_id"] == 0
    assert {row["task_id"] for row in rows} == {opened.task_id}
    assert {row["chat_id"] for row in rows} == {presence_chat_id(KEY)}
    # Every row keeps the call event's identity, so a replay of that event recognizes them.
    assert len({row["presence_event_identity"] for row in rows}) == 1
    assert rows[3]["transport"]["delivery"] == {"state": "authored"}


def test_a_host_notice_terminal_is_silent_speech(tmp_path):
    from ouroboros.presence_runner import _presence_delivery, presence_result_from_stored
    from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE, TERMINAL_ORIGIN_MODEL_FINAL

    assert _presence_delivery("message", "The call ended after 45 minutes.", TERMINAL_ORIGIN_HOST_NOTICE) == ("silent", "")
    assert _presence_delivery("message", "Thanks, bye.", TERMINAL_ORIGIN_MODEL_FINAL) == ("message", "Thanks, bye.")
    stored = {"status": "completed", "terminal_origin": TERMINAL_ORIGIN_HOST_NOTICE,
              "result": "The call ended after 45 minutes.",
              "metadata": {"presence_outcome": "message", "presence": {"delivery_reporting_version": 0}}}
    projected = presence_result_from_stored(stored, "presence-call-task")
    assert (projected.outcome, projected.text) == ("silent", "")


# --- the ceiling governs the live tool path --------------------------------------------------------

@pytest.mark.parametrize("with_ceiling", [True, False])
def test_the_ceiling_governs_schemas_and_dispatch_on_the_live_path(tmp_path, with_ceiling):
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls
    from ouroboros.presence_authority import presence_ceiling_payload
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    drive_root, repo, binding = _setup(tmp_path)
    (repo / "notes.txt").write_text("private notes", encoding="utf-8")
    admission = admit_presence_turn(drive_root=drive_root, authenticated_transport_skill=binding.transport_skill,
                                    binding_id=binding.binding_id, global_max_rounds=None)
    contract = {"capability_ceiling": presence_ceiling_payload(admission.capability_ceiling)} if with_ceiling else {}
    ctx = ToolContext(repo_dir=repo, drive_root=drive_root, task_contract=contract, task_id="presence-call-task")
    registry = ToolRegistry(repo_dir=repo, drive_root=drive_root)
    registry.set_context(ctx)

    # What the live driver may offer the realtime engine: the schemas the task's registry permits.
    offered = {schema["function"]["name"] for schema in initial_tool_schemas(registry, "nano", ["read_file", "chat_history"])}
    # What a tool call from the engine does: the same registry dispatch a text turn uses.
    call = [{"id": "call-1", "type": "function",
             "function": {"name": "read_file", "arguments": json.dumps({"root": "active_workspace", "path": "notes.txt"})}}]
    messages: list = []
    (drive_root / "logs").mkdir(exist_ok=True)
    stateful = StatefulToolExecutor()
    try:
        handle_tool_calls(call, registry, drive_root / "logs", ctx.task_id, stateful, messages,
                          {"tool_calls": []}, lambda _text: None)
    finally:
        stateful.shutdown(wait=True)
    content = messages[-1]["content"]
    if with_ceiling:
        assert "read_file" not in offered and "chat_history" in offered
        assert "PRESENCE_CAPABILITY_BLOCKED" in content and "private notes" not in content
    else:  # the negative control: without the ceiling the same path reads the file
        assert "read_file" in offered
        assert "private notes" in content
