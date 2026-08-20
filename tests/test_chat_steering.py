"""Multi-task chat steering: choosing a target and delivering to it once.

Split verbatim out of ``tests/test_promote_chat_flow.py`` by theme. This module owns
the steer_task transport event, the host manifest that makes a project-bound or busy
direct root addressable without promotion, the manual target a closed admission
returns, the single delivery to a running task, the visible notice on a stale target,
and the running-task metadata a decision turn is given.
"""

from __future__ import annotations

import types


from tests._promote_chat_shared import _isolated_projects_root  # noqa: F401  (autouse fixture applies on import)


# --- WS1: multi-task chat steering (steer_task + current_chat.running_tasks) ---

def test_steer_task_tool_emits_event_with_target_and_client_id(tmp_path):
    """The agent's steer_task choice emits a transport event (target + message +
    chat + originating message id); the supervisor performs the actual delivery."""
    from ouroboros.tools.control import _steer_task

    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={"client_message_id": "cm-42"},
    )
    out = _steer_task(ctx, "abc12345", "also add the benchmarks slide")
    assert out.startswith("⚠️ STEER_UNCONFIRMED")
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "steer_task"
    assert evt["target_task_id"] == "abc12345"
    assert evt["message"] == "also add the benchmarks slide"
    assert evt["chat_id"] == 1
    assert evt["client_message_id"] == "cm-42"
    assert evt["allow_global_root"] is False
    assert ctx._typed_routing_action_emitted == "steer_task"


def test_main_steer_can_address_project_bound_root_from_host_manifest(tmp_path, monkeypatch):
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.tools.control import _steer_task
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    emitted = []
    tool_ctx = types.SimpleNamespace(
        pending_events=emitted,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={
            "client_message_id": "main-42",
            "routing_contract": {"source_lane": "main"},
        },
    )
    _steer_task(tool_ctx, "project-root", "continue from Main")
    assert emitted[0]["allow_global_root"] is True

    supervisor_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "project-root": {
                "task": {"id": "project-root", "chat_id": 42, "project_id": "racer"},
                "started_at": 1.0,
            },
        },
    )
    _handle_steer_task(emitted[0], supervisor_ctx)
    assert drain_owner_messages(tmp_path, "project-root") == ["continue from Main"]


def test_busy_direct_main_root_is_manifested_and_steerable_without_promotion(tmp_path):
    import threading

    import server
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.tools.control import _steer_task
    from supervisor.events import _handle_steer_task

    direct_agent = types.SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(),
        _accepting_owner_messages=True,
        _busy=True,
        _current_task_id="direct-root",
        _current_chat_id=1,
        _current_task_text="Build the AIRI research report",
        _current_task_metadata={"client_message_id": "initial-1"},
        _task_started_ts=10.0,
    )
    routing_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
    )
    metadata = server._decision_turn_metadata(routing_ctx, 1, "followup-1", {})
    root = metadata["main_routing_manifest"]["root_tasks"][0]
    assert root["task_id"] == "direct-root"
    assert root["direct_chat"] is True
    assert root["objective"] == "Build the AIRI research report"

    emitted = []
    tool_ctx = types.SimpleNamespace(
        pending_events=emitted,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={
            "client_message_id": "followup-1",
            "routing_contract": metadata["routing_contract"],
        },
    )
    _steer_task(tool_ctx, "direct-root", "Use FusionBrain images too")
    assert [event["type"] for event in emitted] == ["steer_task"]

    event_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
    )
    _handle_steer_task(emitted[0], event_ctx)
    assert drain_owner_messages(tmp_path, "direct-root") == ["Use FusionBrain images too"]


def test_direct_turn_closed_admission_returns_manual_target(tmp_path):
    import threading

    from supervisor.events import _handle_steer_task

    direct_agent = types.SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(),
        _accepting_owner_messages=False,
        _busy=True,
        _current_task_id="direct-root",
        _current_chat_id=1,
        _current_task_metadata={},
    )
    receipts = []

    class Bridge:
        def send_routing_ack(self, *args, **kwargs):
            receipts.append((args, kwargs))

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
        bridge=Bridge(),
    )
    _handle_steer_task({
        "target_task_id": "direct-root",
        "message": "too late",
        "chat_id": 1,
        "client_message_id": "followup-late",
        "allow_global_root": True,
    }, ctx)
    assert receipts[-1][1]["status"] == "needs_manual_target"


def test_steer_task_tool_requires_args(tmp_path):
    from ouroboros.tools.control import _steer_task

    ctx = types.SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1, task_metadata={})
    assert "TOOL_ARG_ERROR" in _steer_task(ctx, "", "msg")
    assert "TOOL_ARG_ERROR" in _steer_task(ctx, "t1", "")
    assert not ctx.pending_events


def test_handle_steer_task_delivers_once_to_running_task(tmp_path, monkeypatch):
    """The handler writes the running task's owner-mailbox on its active drive, and
    a retry with the same client_message_id+target does NOT double-deliver."""
    import supervisor.queue as queue
    from supervisor.events import _handle_steer_task
    from ouroboros.owner_mailbox import drain_owner_entries

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"t1": {"task": {"id": "t1", "chat_id": 1}, "started_at": 1.0}},
        send_with_budget=lambda *a, **k: None,
    )
    evt = {"type": "steer_task", "target_task_id": "t1", "message": "steer me",
           "chat_id": 1, "client_message_id": "cm-1"}
    _handle_steer_task(evt, ctx)
    _handle_steer_task(evt, ctx)  # retry — same client id + target -> stable msg_id
    entries = drain_owner_entries(tmp_path, "t1")  # dedups by msg_id
    assert [e["text"] for e in entries] == ["steer me"]  # delivered exactly once


def test_handle_steer_task_stale_target_notifies_visibly(tmp_path, monkeypatch):
    """A target no longer RUNNING (or in another chat / a subagent) fails VISIBLY
    with a chat notice and writes NO mailbox — never silently dropped or respawned."""
    import supervisor.queue as queue
    from supervisor.events import _handle_steer_task
    from ouroboros.owner_mailbox import drain_owner_entries

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    notices = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "other": {"task": {"id": "other", "chat_id": 999}},  # different chat
            "sub": {"task": {"id": "sub", "chat_id": 1, "delegation_role": "subagent"}},
        },
        send_with_budget=lambda cid, text, *a, **k: notices.append(text),
    )
    _handle_steer_task({"target_task_id": "gone", "message": "a", "chat_id": 1}, ctx)   # not running
    _handle_steer_task({"target_task_id": "other", "message": "b", "chat_id": 1}, ctx)  # wrong chat
    _handle_steer_task({"target_task_id": "sub", "message": "c", "chat_id": 1}, ctx)    # subagent
    assert len(notices) == 3 and all("Couldn't steer task" in n for n in notices)
    assert drain_owner_entries(tmp_path, "gone") == []
    assert drain_owner_entries(tmp_path, "other") == []
    assert drain_owner_entries(tmp_path, "sub") == []


def test_chat_running_tasks_lists_same_chat_pooled_only(tmp_path):
    """The structural snapshot lists the chat's pooled RUNNING root tasks (so the
    decision turn can pick a steer target) and excludes direct/subagent/other-chat."""
    import server

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "a": {"task": {"id": "a", "chat_id": 1, "objective": "build racer"}, "started_at": 1.0},
            "b": {"task": {"id": "b", "chat_id": 1, "title": "Docs", "objective": "write docs"}, "started_at": 2.0},
            "direct": {"task": {"id": "direct", "chat_id": 1, "_is_direct_chat": True}},
            "sub": {"task": {"id": "sub", "chat_id": 1, "delegation_role": "subagent"}},
            "elsewhere": {"task": {"id": "elsewhere", "chat_id": 7}},
        },
    )
    rows = server._chat_running_tasks(ctx, 1)
    assert {r["task_id"] for r in rows} == {"a", "b"}
    assert all(r["steerable"] for r in rows)
    by_id = {r["task_id"]: r for r in rows}
    assert by_id["a"]["objective"] == "build racer"
    assert by_id["b"]["title"] == "Docs"


def test_decision_turn_metadata_injects_running_tasks_and_client_id(tmp_path):
    """The chat-turn metadata is enriched with current_chat.running_tasks + the
    originating message id, so build_runtime_section can surface them (P5 — state
    only; the agent still chooses)."""
    import server

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"a": {"task": {"id": "a", "chat_id": 1, "objective": "x"}, "started_at": 1.0}},
    )
    md = server._decision_turn_metadata(ctx, 1, "cm-9", {"project_id": "p"})
    assert md["project_id"] == "p"  # preserved
    assert md["client_message_id"] == "cm-9"
    assert md["current_chat"]["chat_id"] == 1
    assert [t["task_id"] for t in md["current_chat"]["running_tasks"]] == ["a"]
    # No running tasks + no client id -> metadata returned unchanged.
    empty_ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={})
    assert server._decision_turn_metadata(empty_ctx, 1, "", {"k": "v"}) == {"k": "v"}
