"""Server integration seams of the in-flight direct turn indicator.

Pins the three seams the browser contract depends on:

1. the lane admission (``supervisor.workers.handle_chat_direct``) writes the turn's RUNNING
   row, which the census (``/api/state`` ``active_direct_turns``) lists with the correct
   ``kind``/``client_message_id``/``project_id`` for as long as a carrier holds it.
2. ``supervisor.events._handle_typing_start`` stamps ``kind`` and
   ``client_message_id`` from the registry onto the typing action — and leaves
   them empty for untracked tasks. No in-repo client reads the stamp (wire
   compatibility): only the /api/state census inserts into the header live-set, so an empty
   kind exempts nothing from that census's deletion authority.
3. ``supervisor.message_bus.MessageBus.send_chat_action`` carries the typed
   fields on the broadcast ``typing`` frame, omitting absent optionals.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from supervisor.active_activity import get_direct_activity_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_direct_activity_registry()
    registry.clear()
    yield
    registry.clear()


# ---------------------------------------------------------------------------
# Seam 1: the lane admission <-> the census (the RUNNING row, as /api/state reads it)
# ---------------------------------------------------------------------------


def _census():
    from ouroboros.gateway.state import _direct_turns_snapshot_safe

    return _direct_turns_snapshot_safe()


class _CensusProbeAgent:
    """Snapshots the census while the turn is 'running' in its carrier."""

    def __init__(self):
        self.task = None
        self.census_during = None

    def handle_task(self, task):
        self.task = task
        self.census_during = _census()
        return [{"type": "task_done", "task_id": task["id"], "chat_id": task["chat_id"]}]


def _patch_workers(monkeypatch, tmp_path):
    import queue

    import supervisor.message_bus as message_bus
    import supervisor.workers as workers
    from tests.conversation_carrier_support import isolate_lane_pool

    drive = tmp_path / "drive"
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    # These tests pin the census/bridge seams, not the process-lifetime event bus: a local
    # queue keeps the turn off get_event_q() (same isolation precedent as
    # tests/test_promote_event_transport.py::_isolate_event_bus_shutdown_latch).
    bus = queue.Queue()
    [spare] = isolate_lane_pool(monkeypatch, drive, event_q=bus)
    # Capture the turn's start announce (bridge typing frame) instead of
    # touching the real singleton bridge.
    bridge_probe = _BridgeProbe()
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge_probe)
    return workers, bridge_probe, spare, bus


def test_admitted_turn_is_in_the_census_for_its_duration(tmp_path, monkeypatch):
    from tests.conversation_carrier_support import carry

    workers, bridge, spare, bus = _patch_workers(monkeypatch, tmp_path)
    workers.handle_chat_direct(7, "hello", None, task_metadata={"origin_message_ref": {"client_message_id": "cmid-42"}})
    # The admission wrote the row; the census lists it from the row (no process-local actor).
    [turn] = _census()
    assert turn["chat_id"] == 7 and turn["client_message_id"] == "cmid-42" and turn["kind"] == "direct_chat"
    assert turn["phase"] == "working" and turn["carrier"] == "worker"  # handed to the lane spare
    assert get_direct_activity_registry().snapshot() == []
    agent = _CensusProbeAgent()
    carry(workers, spare, agent)
    assert agent.task["id"] == turn["activity_id"]
    assert [row["activity_id"] for row in agent.census_during] == [turn["activity_id"]]
    # The carrier's task_done ends the row (the supervisor's pooled custody pops it).
    done = [event for event in list(bus.queue) if event.get("type") == "task_done"]
    assert [event["task_id"] for event in done] == [turn["activity_id"]]
    # The authoritative start is announced immediately (2A): the client's
    # `Sending...` retires on this frame, which also carries the
    # activity<->client_message_id link for the keyed conclusion.
    announces = [c for c in bridge.calls if c["action"] == "typing"]
    assert len(announces) == 1
    assert announces[0]["chat_id"] == 7
    assert announces[0]["activity_id"] == turn["activity_id"]
    assert announces[0]["client_message_id"] == "cmid-42"
    assert announces[0]["kind"] == "direct_chat"
    assert announces[0]["phase"] == "thinking"


def test_admitted_turn_census_row_carries_its_project_id(tmp_path, monkeypatch):
    workers, bridge, spare, _bus = _patch_workers(monkeypatch, tmp_path)
    workers.handle_chat_direct(3, "decide", None, task_metadata={"client_message_id": "cmid-flat", "project_id": "proj-x"})
    [turn] = _census()
    assert turn["kind"] == "direct_chat"
    # Flat metadata key is the fallback when origin_message_ref is absent.
    assert turn["client_message_id"] == "cmid-flat"
    assert turn["project_id"] == "proj-x"
    assert get_direct_activity_registry().snapshot() == []
    announces = [c for c in bridge.calls if c["action"] == "typing"]
    assert len(announces) == 1
    assert announces[0]["kind"] == "direct_chat"


def test_a_carrier_failure_is_the_turns_own_terminal_and_the_start_announce_links_the_submission(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from tests.conversation_carrier_support import carry

    workers, bridge, spare, _bus = _patch_workers(monkeypatch, tmp_path)
    drive = workers.DRIVE_ROOT

    class _BoomAgent:
        env = SimpleNamespace(drive_root=drive, drive_path=lambda name: drive / name)

        def handle_task(self, task):
            self.task = task
            raise RuntimeError("deliberate failure")

    workers.handle_chat_direct(5, "boom", None, task_metadata={"origin_message_ref": {"client_message_id": "cmid-err"}})
    agent = _BoomAgent()
    # The carrier records the exception as the turn's own terminal before it unwinds (owner
    # decision 1A, 11.10.2026); the registry keeps no ghost activity that would pin "Thinking...".
    with pytest.raises(RuntimeError, match="deliberate failure"):
        carry(workers, spare, agent)
    assert get_direct_activity_registry().snapshot() == []
    stored = load_task_result(drive, agent.task["id"])
    assert stored["status"] == "failed" and "deliberate failure" in stored["result"]
    # The start announce carried the activity<->client_message_id link the keyed conclusion needs.
    announces = [c for c in bridge.calls if c["action"] == "typing"]
    assert len(announces) == 1
    assert announces[0]["activity_id"] == str(agent.task["id"])
    assert announces[0]["client_message_id"] == "cmid-err"
    assert announces[0]["kind"] == "direct_chat"


# ---------------------------------------------------------------------------
# Seam 2: _handle_typing_start stamping
# ---------------------------------------------------------------------------


class _BridgeProbe:
    def __init__(self):
        self.calls = []

    def send_chat_action(self, chat_id, action="typing", **kwargs):
        self.calls.append({"chat_id": chat_id, "action": action, **kwargs})
        return True


class _CtxProbe:
    def __init__(self):
        self.bridge = _BridgeProbe()


def test_typing_start_stamps_kind_for_registry_tracked_turn():
    from supervisor.events import _handle_typing_start

    registry = get_direct_activity_registry()
    registry.register(
        "act-typed",
        chat_id=9,
        client_message_id="cmid-9",
        kind="direct_chat",
        phase="thinking",
    )

    ctx = _CtxProbe()
    _handle_typing_start(
        {"type": "typing_start", "chat_id": 9, "task_id": "act-typed", "phase": "thinking"},
        ctx,
    )

    assert len(ctx.bridge.calls) == 1
    call = ctx.bridge.calls[0]
    assert call["chat_id"] == 9
    assert call["activity_id"] == "act-typed"
    assert call["client_message_id"] == "cmid-9"
    assert call["kind"] == "direct_chat"
    assert call["phase"] == "thinking"


def test_typing_start_leaves_kind_empty_for_untracked_managed_task():
    from supervisor.events import _handle_typing_start

    ctx = _CtxProbe()
    _handle_typing_start(
        {"type": "typing_start", "chat_id": 1, "task_id": "managed-task-1", "phase": "thinking"},
        ctx,
    )

    assert len(ctx.bridge.calls) == 1
    call = ctx.bridge.calls[0]
    assert call["activity_id"] == "managed-task-1"
    # No registry entry => no kind stamp (this ctx carries no RUNNING row to
    # stamp "managed_task" from). No in-repo client reads the stamp; only the
    # /api/state census inserts into the header live-set, so an absent kind
    # exempts nothing.
    assert call["kind"] == ""
    assert call["client_message_id"] == ""


def test_typing_start_without_drive_root_or_running_still_sends():
    """Binding resolution is fail-soft: a ctx that carries neither DRIVE_ROOT
    nor a RUNNING table still delivers the indicator to the chat the event
    names — and 0 is the panel, a destination, not an absence."""
    from supervisor.events import _handle_typing_start

    ctx = _CtxProbe()
    _handle_typing_start(
        {"type": "typing_start", "chat_id": 0, "task_id": "act-rootless", "phase": "thinking"},
        ctx,
    )

    assert len(ctx.bridge.calls) == 1
    assert ctx.bridge.calls[0]["chat_id"] == 0


# ---------------------------------------------------------------------------
# Seam 3: send_chat_action broadcast frame
# ---------------------------------------------------------------------------


def _bus_with_probe():
    from supervisor.message_bus import LocalChatBridge

    bus = LocalChatBridge(settings={})
    sent = []
    bus._broadcast_fn = sent.append
    return bus, sent


def test_send_chat_action_broadcasts_typed_fields():
    bus, sent = _bus_with_probe()
    ok = bus.send_chat_action(
        4,
        "typing",
        activity_id="act-4",
        client_message_id="cmid-4",
        phase="thinking",
        kind="direct_chat",
    )
    assert ok is True
    assert len(sent) == 1
    frame = sent[0]
    assert frame["type"] == "typing"
    assert frame["chat_id"] == 4
    assert frame["activity_id"] == "act-4"
    assert frame["client_message_id"] == "cmid-4"
    assert frame["phase"] == "thinking"
    assert frame["kind"] == "direct_chat"


def test_send_chat_action_omits_absent_optionals():
    bus, sent = _bus_with_probe()
    bus.send_chat_action(4, "typing")
    assert len(sent) == 1
    frame = sent[0]
    # Optionals are omitted, not emitted as empty strings (contract shape).
    assert "activity_id" not in frame
    assert "client_message_id" not in frame
    assert "kind" not in frame
    assert frame["phase"] == "thinking"
