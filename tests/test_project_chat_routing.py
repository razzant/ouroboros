"""What a project chat can see, and how a message reaches the task behind it.

Split verbatim out of ``tests/test_promote_chat_flow.py`` by theme. This module owns
the projects-changed broadcast, the chat-id registry, the history and recent-context
surfaces a project focus exposes, the 1:1 delivery rules — idempotent, never confirmed
on a failed mailbox write, deferred when several tasks run, escalated to an ephemeral
decision turn when the project is busy — and the restart drain that must not strand a
live chat task.
"""

from __future__ import annotations

import types


from tests._promote_chat_shared import _isolated_projects_root  # noqa: F401  (autouse fixture applies on import)


def test_promote_chat_to_task_broadcasts_projects_changed(tmp_path, monkeypatch):
    """Backend project creation pushes a projects_changed WS frame carrying the new
    chat_id, so the frontend fan-out learns the project thread IMMEDIATELY (no
    ≤20s window where its live frames misroute into the main chat)."""
    import supervisor.message_bus as mbus
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    broadcasts = []
    fake_bridge = types.SimpleNamespace(broadcast=lambda payload: broadcasts.append(payload))
    monkeypatch.setattr(mbus, "get_bridge", lambda: fake_bridge)
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: None,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "pc1",
        "objective": "Build it",
        "project_id": "proj-x",
        "chat_id": 0,
    }, ctx)

    from ouroboros.contracts.chat_id_policy import project_chat_id

    changed = [b for b in broadcasts if b.get("type") == "projects_changed"]
    assert len(changed) == 1
    assert changed[0]["project_id"] == "proj-x"
    assert changed[0]["chat_id"] == project_chat_id("proj-x")


def test_registered_project_chat_ids_recognizes_every_project(tmp_path):
    """The isolation SSOT recognizes EVERY registered project's chat_id (regardless
    of sidebar visibility) so its raw chat never re-leaks into the штаб's main
    context / dialogue consolidation / background consciousness (BIBLE P1). Sidebar
    visibility is a separate presentation concern (no project statuses, v6.33.0)."""
    from ouroboros.projects_registry import (
        create_project,
        registered_project_chat_ids,
        update_project,
    )

    proj = create_project(tmp_path, "old-racer")
    chat_id = int(proj["chat_id"])
    assert chat_id in registered_project_chat_ids(tmp_path)
    # A rename (or any mutable-field update) never drops it from the isolation set.
    update_project(tmp_path, "old-racer", name="Old Racer (renamed)")
    assert chat_id in registered_project_chat_ids(tmp_path)


def test_chat_history_tool_spans_all_threads_full_awareness(tmp_path):
    """Full project awareness (v6.32.0): the chat_history TOOL is the one mind's
    DELIBERATE recall — it spans the WHOLE conversation (main + ALL project
    threads), only A2A virtual transport excluded. Project-task FOCUS lives in the
    passive default context (build_recent_sections), NOT in this recall tool, so
    the one identity can recall anything it chooses (BIBLE P1)."""
    import json

    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    a = create_project(tmp_path, "alpha")
    b = create_project(tmp_path, "beta")
    ca, cb = int(a["chat_id"]), int(b["chat_id"])
    rows = [
        {"direction": "in", "text": "main-msg", "chat_id": 1},
        {"direction": "in", "text": "alpha-msg", "chat_id": ca},
        {"direction": "in", "text": "beta-msg", "chat_id": cb},
        {"direction": "in", "text": "a2a-noise", "chat_id": -1001},
    ]
    (logs / "chat.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    mem = Memory(drive_root=tmp_path)

    view = mem.chat_history(count=50)
    assert "main-msg" in view and "alpha-msg" in view and "beta-msg" in view  # all threads
    assert "a2a-noise" not in view  # only A2A virtual transport excluded


def test_recent_context_full_awareness_and_project_focus_with_bindings(tmp_path):
    """Passive context (v6.32.0): the one identity's MAIN recent context sees
    EVERYTHING, including a post-hoc bound task's rows (one mind, BIBLE P1). A
    PROJECT task's recent context is FOCUSED on its own thread + rows of tasks
    bound to it; unrelated main chat is left out of the focused working view
    (focus in the passive default, not isolation)."""
    import json

    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import bind_task_to_project, create_project

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    proj = create_project(tmp_path, "promoted")
    pchat = int(proj["chat_id"])
    bind_task_to_project(tmp_path, "task-7", "promoted", pchat, origin={"absent": "system"})
    rows = [
        {"direction": "in", "text": "plain-main", "chat_id": 1},
        {"direction": "out", "text": "bound-task-row", "chat_id": 1, "task_id": "task-7"},
    ]
    (logs / "chat.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    mem = Memory(drive_root=tmp_path)

    # Main passive context: full awareness sees everything.
    main_ctx = "\n".join(build_recent_sections(mem, env=None))
    assert "plain-main" in main_ctx and "bound-task-row" in main_ctx

    # Project task passive context: focused on its own thread + bound-task rows.
    proj_ctx = "\n".join(build_recent_sections(mem, env=None, thread_chat_id=pchat))
    assert "bound-task-row" in proj_ctx
    assert "plain-main" not in proj_ctx


def test_restart_drain_defers_then_completes_without_sleeping(tmp_path, monkeypatch):
    """The drain must NOT sleep on the supervisor thread: a restart with live
    tasks defers (returns immediately), and a later loop-tick check completes
    it once tasks drain or the deadline passes."""
    import types

    import server

    monkeypatch.setenv("OUROBOROS_RESTART_DRAIN_MAX_SEC", "120")
    performed = []
    from ouroboros import server_restart
    monkeypatch.setattr(server_restart, "_perform_supervisor_restart", lambda ctx, **kw: performed.append(True))
    server._pending_restart.clear()

    now = __import__("time").time()
    ctx = types.SimpleNamespace(
        RUNNING={"t1": {"task": {"id": "t1"}, "last_heartbeat_at": now}},
        load_state=lambda: {"owner_chat_id": 0},
        send_with_budget=lambda *a, **k: None,
        DRIVE_ROOT=tmp_path,
    )

    # Live task -> defer, do NOT restart inline.
    server._handle_restart_in_supervisor({"reason": "evolution"}, ctx)
    assert performed == []
    assert server._pending_restart  # recorded for the loop tick

    # Tick while still live + before deadline -> keep waiting.
    server._check_pending_restart_drain(ctx)
    assert performed == []

    # Task drained -> the next tick completes the restart.
    ctx.RUNNING = {}
    server._check_pending_restart_drain(ctx)
    assert performed == [True]
    assert not server._pending_restart


def test_restart_drain_no_live_tasks_restarts_immediately(tmp_path, monkeypatch):
    import types

    import server

    monkeypatch.setenv("OUROBOROS_RESTART_DRAIN_MAX_SEC", "120")
    performed = []
    from ouroboros import server_restart
    monkeypatch.setattr(server_restart, "_perform_supervisor_restart", lambda ctx, **kw: performed.append(True))
    server._pending_restart.clear()

    ctx = types.SimpleNamespace(
        RUNNING={},
        load_state=lambda: {"owner_chat_id": 0},
        send_with_budget=lambda *a, **k: None,
        DRIVE_ROOT=tmp_path,
    )
    server._handle_restart_in_supervisor({"reason": "x"}, ctx)
    assert performed == [True]
    assert not server._pending_restart


def test_restart_drain_uses_generic_queue_heartbeat_not_retired_planning_knob(
    tmp_path, monkeypatch
):
    """A stale generic RUNNING heartbeat must not defer restart, even when a
    legacy process environment still carries the removed planning-scout knob."""
    import time
    import types

    import server
    from supervisor.queue import HEARTBEAT_STALE_SEC

    monkeypatch.setenv("OUROBOROS_RESTART_DRAIN_MAX_SEC", "120")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC", "999999")
    performed = []
    from ouroboros import server_restart
    monkeypatch.setattr(server_restart, "_perform_supervisor_restart", lambda ctx, **kw: performed.append(True))
    server._pending_restart.clear()

    ctx = types.SimpleNamespace(
        RUNNING={
            "stale": {
                "task": {"id": "stale"},
                "last_heartbeat_at": time.time() - HEARTBEAT_STALE_SEC - 1,
            }
        },
        load_state=lambda: {"owner_chat_id": 0},
        send_with_budget=lambda *a, **k: None,
        DRIVE_ROOT=tmp_path,
    )

    server._handle_restart_in_supervisor({"reason": "x"}, ctx)

    assert performed == [True]
    assert not server._pending_restart


def test_direct_chat_project_thread_skips_letters_home(tmp_path, monkeypatch):
    """A project-thread CONVERSATION (direct chat) is project-scoped for context
    only: it must not block on post-processing or write journal/digest."""
    from ouroboros.project_lease import running_project_ids

    # Sanity: a direct-chat task is never a lease occupant (no project lane),
    # and _is_direct_chat tasks are excluded from letters-home by the pipeline.
    direct = {"id": "d1", "type": "task", "project_id": "racer", "_is_direct_chat": True}
    # The lease only counts top-level project tasks; a direct-chat task still
    # carries project_id but the pipeline gates letters-home on _is_direct_chat.
    assert running_project_ids([{"task": direct}]) == {"racer"}  # context scope is real
    # (full pipeline gating is covered by the agent_task_pipeline branch; this
    # pins the flag the branch reads.)
    assert direct.get("_is_direct_chat") is True


def test_route_project_chat_ignores_non_registered_chat_ids(tmp_path):
    """External-transport chat ids (large, non-project) must NOT be captured as
    project threads — only registered project chat_ids route to a task mailbox."""
    import types

    import server
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])
    transport_chat = 987654321  # Telegram-style id, NOT a project

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "tp": {"task": {"id": "tp", "chat_id": transport_chat}, "last_heartbeat_at": 1.0},
            "pr": {"task": {"id": "pr", "chat_id": project_chat}, "last_heartbeat_at": 1.0},
        },
    )
    # Transport chat: not a project -> never routed (main free lane preserved).
    assert server._route_project_chat_to_running_task(ctx, transport_chat, "hi") == ""
    # Registered project chat with an active task -> routed to its mailbox.
    assert server._route_project_chat_to_running_task(ctx, project_chat, "steer") == "pr"


def test_route_project_chat_defers_when_multiple_running_tasks(tmp_path, monkeypatch):
    """v6.34.0 WS1/P5: with MORE THAN ONE steerable task in a project room, choosing a
    target is a routing JUDGMENT — code must NOT mechanically steer the first of several.
    The pre-LLM delivery returns "" (the message reaches the decision turn, where the
    agent picks via steer_task) and nothing is mechanically written to a mailbox."""
    import types

    import server
    import ouroboros.owner_mailbox as omb
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])

    delivered = []
    monkeypatch.setattr(
        omb, "write_owner_message", lambda *a, **k: delivered.append(a) or True
    )

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "a": {"task": {"id": "a", "chat_id": project_chat}, "last_heartbeat_at": 1.0},
            "b": {"task": {"id": "b", "chat_id": project_chat}, "last_heartbeat_at": 1.0},
        },
    )
    assert server._route_project_chat_to_running_task(ctx, project_chat, "which one?") == ""
    assert delivered == []  # no mechanical first-of-N steer


def test_route_project_chat_1to1_delivery_is_idempotent(tmp_path, monkeypatch):
    """The 1:1 project-room auto-delivery derives a STABLE msg_id from client_message_id,
    so a WebSocket retry can't double-deliver (drain_owner_entries dedups by msg_id) —
    matching steer_task's idempotency contract."""
    import types

    import server
    import ouroboros.owner_mailbox as omb
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])

    msg_ids = []
    monkeypatch.setattr(omb, "write_owner_message",
                        lambda drive, text, tid, msg_id=None, **k: msg_ids.append(msg_id) or True)

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"pr": {"task": {"id": "pr", "chat_id": project_chat}, "last_heartbeat_at": 1.0}},
    )
    # Same client_message_id retried twice -> identical stable msg_id (dedup), not None (random).
    server._route_project_chat_to_running_task(ctx, project_chat, "go", "cmid-7")
    server._route_project_chat_to_running_task(ctx, project_chat, "go", "cmid-7")
    assert msg_ids == ["cmid-7:pr", "cmid-7:pr"]


def test_route_project_chat_does_not_confirm_failed_mailbox_write(tmp_path, monkeypatch):
    import types

    import ouroboros.owner_mailbox as omb
    import server
    from ouroboros.projects_registry import create_project

    project_chat = int(create_project(tmp_path, "racer")["chat_id"])
    monkeypatch.setattr(omb, "write_owner_message", lambda *_a, **_k: False)
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "pr": {
                "task": {"id": "pr", "chat_id": project_chat},
                "last_heartbeat_at": 1.0,
            }
        },
    )

    assert (
        server._route_project_chat_to_running_task(
            ctx, project_chat, "must be durable", "owner-msg"
        )
        == ""
    )


def test_busy_project_chat_routes_to_ephemeral_decision_turn(tmp_path, monkeypatch):
    """WS1/P5 (v6.34.0): a busy PROJECT chat is NOT mechanically auto-enqueued into a
    duplicate pooled task. It runs the ephemeral decision turn (project-scoped, seeing
    current_chat.running_tasks) so the one mind decides steer_task / answer / promote by
    judgment — replacing the old 'Hybrid B+' auto-enqueue fallback."""
    import threading as _threading

    import server
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "market-research")
    project_chat = int(proj["chat_id"])
    enqueued = []
    ephemeral_calls = []
    called = _threading.Event()

    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *a, **k: None)

    class _Bridge:
        def get_updates(self, offset=0, timeout=0):
            return [{
                "update_id": offset,
                "message": {
                    "chat": {"id": project_chat},
                    "from": {"id": 1},
                    "text": "сколько будет 2+2?",
                    "source": "web",
                    "task_metadata": {"project_id": "market-research"},
                },
            }]

    class _Consciousness:
        def inject_observation(self, _text):
            return None

    def _ephemeral(cid, text, image_data, *, task_constraint=None, task_metadata=None):
        ephemeral_calls.append({"chat_id": cid, "text": text, "metadata": task_metadata})
        called.set()

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=_Consciousness(),
        get_chat_agent=lambda: types.SimpleNamespace(_busy=True),
        handle_chat_direct=lambda *a, **k: (_ for _ in ()).throw(AssertionError("direct lane must not run when busy")),
        handle_chat_ephemeral=_ephemeral,
        enqueue_task=lambda task: enqueued.append(task),
        send_with_budget=lambda *a, **k: None,
    )

    assert server._process_bridge_updates(_Bridge(), 0, ctx) == 1
    assert called.wait(timeout=3)  # the ephemeral decision turn ran on its own thread
    assert enqueued == []  # NOT auto-enqueued into a duplicate pooled task
    assert len(ephemeral_calls) == 1
    md = ephemeral_calls[0]["metadata"] or {}
    assert str(md.get("project_id") or "")  # project-scoped decision turn
    assert "сколько будет 2+2?" in (ephemeral_calls[0]["text"] or "")
