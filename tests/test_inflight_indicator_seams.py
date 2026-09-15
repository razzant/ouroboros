"""Server integration seams of the in-flight direct turn indicator.

Pins the three seams the browser contract depends on:

1. ``supervisor.workers._run_chat_task`` tracks the turn in the
   ``DirectActivityRegistry`` through preparation, ``agent.handle_task`` and event delivery
   with the correct ``kind``/``client_message_id``/``project_id``.
2. ``supervisor.events._handle_typing_start`` stamps ``kind`` and
   ``client_message_id`` from the registry onto the typing action — and leaves
   them empty for untracked tasks. No in-repo client reads the stamp (wire
   compatibility): only the /api/state census inserts into the header live-set, so an empty
   kind exempts nothing from that census's deletion authority.
3. ``supervisor.message_bus.MessageBus.send_chat_action`` carries the typed
   fields on the broadcast ``typing`` frame, omitting absent optionals.
"""

from __future__ import annotations

import pytest

from supervisor.active_activity import get_direct_activity_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_direct_activity_registry()
    registry.clear()
    yield
    registry.clear()


# ---------------------------------------------------------------------------
# Seam 1: _run_chat_task <-> DirectActivityRegistry
# ---------------------------------------------------------------------------


class _RegistryProbeAgent:
    """Snapshots the global registry while the turn is 'running'."""

    def __init__(self):
        self.task = None
        self.snapshot_during = None

    def handle_task(self, task):
        self.task = task
        self.snapshot_during = get_direct_activity_registry().snapshot()
        return []


def _patch_workers(monkeypatch, tmp_path):
    import queue

    import supervisor.message_bus as message_bus
    import supervisor.workers as workers

    drive = tmp_path / "drive"
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
    # These tests pin the registry/bridge seams, not the process-lifetime event
    # bus. Use a local queue so the turn never touches get_event_q(): a prior
    # TestClient lifespan in the same xdist worker leaves _EVENT_Q_SHUTDOWN
    # latched (server shutdown calls shutdown_event_q()), and the real
    # get_event_q() then raises before agent.handle_task ever runs (seen on the
    # Windows CI shard; same isolation precedent as
    # tests/test_promote_event_transport.py::_isolate_event_bus_shutdown_latch).
    monkeypatch.setattr(workers, "get_event_q", lambda: queue.Queue())
    # Capture the turn's start announce (bridge typing frame) instead of
    # touching the real singleton bridge.
    bridge_probe = _BridgeProbe()
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge_probe)
    return workers, bridge_probe


def test_run_chat_task_tracks_direct_turn_for_its_duration(tmp_path, monkeypatch):
    workers, bridge = _patch_workers(monkeypatch, tmp_path)

    agent = _RegistryProbeAgent()
    workers._run_chat_task(
        agent,
        7,
        "hello",
        None,
        task_metadata={
            "origin_message_ref": {"client_message_id": "cmid-42"},
        },
    )

    assert agent.task is not None
    snap = agent.snapshot_during
    assert isinstance(snap, list) and len(snap) == 1
    turn = snap[0]
    assert turn["activity_id"] == str(agent.task["id"])
    assert turn["chat_id"] == 7
    assert turn["client_message_id"] == "cmid-42"
    assert turn["kind"] == "direct_chat"
    assert turn["phase"] == "thinking"
    # Cleared once the turn concluded.
    assert get_direct_activity_registry().snapshot() == []
    # The authoritative start is announced immediately (2A): the client's
    # `Sending...` retires on this frame, which also carries the
    # activity<->client_message_id link for the keyed conclusion.
    announces = [c for c in bridge.calls if c["action"] == "typing"]
    assert len(announces) == 1
    assert announces[0]["chat_id"] == 7
    assert announces[0]["activity_id"] == str(agent.task["id"])
    assert announces[0]["client_message_id"] == "cmid-42"
    assert announces[0]["kind"] == "direct_chat"
    assert announces[0]["phase"] == "thinking"


def test_run_chat_task_tracks_direct_turn_with_project_id(tmp_path, monkeypatch):
    workers, bridge = _patch_workers(monkeypatch, tmp_path)

    agent = _RegistryProbeAgent()
    workers._run_chat_task(
        agent,
        3,
        "decide",
        None,
        task_metadata={
            "client_message_id": "cmid-flat",
            "project_id": "proj-x",
        },
    )

    snap = agent.snapshot_during
    assert isinstance(snap, list) and len(snap) == 1
    turn = snap[0]
    assert turn["kind"] == "direct_chat"
    # Flat metadata key is the fallback when origin_message_ref is absent.
    assert turn["client_message_id"] == "cmid-flat"
    assert turn["project_id"] == "proj-x"
    assert get_direct_activity_registry().snapshot() == []
    announces = [c for c in bridge.calls if c["action"] == "typing"]
    assert len(announces) == 1
    assert announces[0]["kind"] == "direct_chat"


def test_run_chat_task_unregisters_and_keys_error_final_when_agent_raises(tmp_path, monkeypatch):
    workers, bridge = _patch_workers(monkeypatch, tmp_path)
    sent = []
    monkeypatch.setattr(
        workers, "send_with_budget", lambda *a, **k: sent.append((a, k))
    )

    class _BoomAgent:
        def handle_task(self, task):
            self.task = task
            raise RuntimeError("deliberate failure")

    agent = _BoomAgent()
    # _run_chat_task swallows the error (reports to chat); the registry must
    # not leak a ghost activity that would pin "Thinking..." forever.
    workers._run_chat_task(
        agent,
        5,
        "boom",
        None,
        task_metadata={"origin_message_ref": {"client_message_id": "cmid-err"}},
    )
    assert get_direct_activity_registry().snapshot() == []
    # The error final is keyed with the turn's activity id so the client
    # concludes exactly this turn (4A) instead of relying on an unkeyed sweep.
    assert len(sent) == 1
    args, kwargs = sent[0]
    assert args[0] == 5
    assert kwargs.get("task_id") == str(agent.task["id"])
    assert kwargs.get("progress_meta") == {"task_terminal_status": "failed"}
    # The failed turn re-announces the activity<->client_message_id link right
    # before the keyed final, so the client retires the `Sending...` submission
    # even when the start announce never reached it.
    announces = [c for c in bridge.calls if c["action"] == "typing"]
    assert len(announces) == 2  # start announce + pre-final announce
    assert announces[-1]["activity_id"] == str(agent.task["id"])
    assert announces[-1]["client_message_id"] == "cmid-err"
    assert announces[-1]["kind"] == "direct_chat"


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
