"""promote_chat_to_task + project chat routing (multi-project, v6.32.0)."""

from __future__ import annotations

import types


def test_promote_tool_emits_event_with_chat_and_project(tmp_path):
    from ouroboros.tools.control import _promote_chat_to_task

    events = []
    ctx = types.SimpleNamespace(
        pending_events=events,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
    )
    out = _promote_chat_to_task(ctx, "Build the racer prototype", project_id="racer")
    assert out.startswith("OK: promoted to supervised task")
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "promote_chat_to_task"
    assert evt["objective"] == "Build the racer prototype"
    assert evt["project_id"] == "racer"
    assert evt["chat_id"] == 1
    assert evt["task_id"]


def test_promote_tool_rejects_dirty_project_id(tmp_path):
    from ouroboros.tools.control import _promote_chat_to_task

    ctx = types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path,
    )
    out = _promote_chat_to_task(ctx, "x", project_id="Bad Name!")
    assert "TOOL_ARG_ERROR" in out
    assert not ctx.pending_events


def test_promote_tool_project_name_creates_named_project_event(tmp_path):
    """LLM-first 'create a named project and work there' (v6.33.0): project_name
    derives a clean id, carries the human display name, and rides title."""
    from ouroboros.tools.control import _promote_chat_to_task

    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1, drive_root=tmp_path,
    )
    out = _promote_chat_to_task(
        ctx, "research everything about the airi institute",
        project_name="Airi Research", title="Airi Research",
    )
    assert out.startswith("OK: promoted to supervised task")
    assert "new project 'Airi Research'" in out
    evt = events[0]
    assert evt["project_name"] == "Airi Research"
    assert evt["project_id"] == "airi-research"   # derived, filesystem-clean
    assert evt["title"] == "Airi Research"


def test_project_id_from_display_name_handles_non_ascii():
    """A Cyrillic-only display name must still yield a usable (hash) id, not '' —
    so the named-project feature works for the Russian-speaking owner."""
    from ouroboros.project_facts import project_id_from_display_name

    assert project_id_from_display_name("airi research") == "airi-research"
    assert project_id_from_display_name("Динозавры").startswith("proj_")
    # Deterministic: re-asking for the same name resolves to the same project.
    assert project_id_from_display_name("Динозавры") == project_id_from_display_name("Динозавры")
    assert project_id_from_display_name("") == ""


def test_promote_tool_cyrillic_project_name_still_creates(tmp_path):
    """promote_chat_to_task(project_name=<cyrillic>) must NOT fail — it derives a
    hash id while keeping the Cyrillic display name (Workflow-caught regression)."""
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.tools.control import _promote_chat_to_task

    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1, drive_root=tmp_path,
    )
    out = _promote_chat_to_task(ctx, "исследуй динозавров", project_name="динозавры", title="динозавры")
    assert "TOOL_ARG_ERROR" not in out
    assert out.startswith("OK: promoted")
    evt = events[0]
    assert evt["project_name"] == "динозавры"
    assert evt["project_id"] == project_id_from_display_name("динозавры")
    assert evt["project_id"].startswith("proj_")  # ASCII-clean hash fallback


def test_promote_event_names_project_from_display_name(tmp_path, monkeypatch):
    """The handler creates the project with the human display name (not the bare
    id) and persists the task title (v6.33.0)."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import get_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        load_state=lambda: {"owner_chat_id": 1},
    )
    workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "air01",
        "objective": "research the airi institute",
        "project_id": "airi-research",
        "project_name": "Airi Research",
        "title": "Airi Research",
        "chat_id": 0,
    }, ctx)

    project = get_project(tmp_path, "airi-research")
    assert project is not None
    assert project["name"] == "Airi Research"      # human name, not the bare id
    assert enqueued[0]["title"] == "Airi Research"  # persisted on the task


def test_derive_project_name_prefers_title(tmp_path):
    """_derive_project_name uses the model-coined short title over the objective
    so a converted card never shows a truncated sentence or a bare id (v6.33.0)."""
    from ouroboros.gateway.projects import _derive_project_name
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(
        tmp_path, "tt01", STATUS_RUNNING,
        title="Tic-tac-toe game",
        objective="make an html page with a tic-tac-toe game that tracks score",
    )
    assert _derive_project_name(tmp_path, "tt01") == "Tic-tac-toe game"


def test_promote_event_enqueues_first_class_task(tmp_path, monkeypatch):
    """The supervisor handler enqueues a pooled OWNER task (not a subagent),
    registers the project, and carries the chat thread."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        load_state=lambda: {"owner_chat_id": 1},
    )
    evt = {
        "type": "promote_chat_to_task",
        "task_id": "abc12345",
        "objective": "Research the market",
        "expected_output": "A summary",
        "project_id": "research-1",
        "chat_id": 0,  # falls back to owner chat
    }
    workers.promote_chat_to_task(evt, ctx)

    assert len(enqueued) == 1
    task = enqueued[0]
    assert task["id"] == "abc12345"
    assert task["type"] == "task"
    assert task["project_id"] == "research-1"
    assert "delegation_role" not in task
    assert "_is_direct_chat" not in task
    assert "Expected output: A summary" in task["text"]
    # The project got registered as a side effect, and the promoted task runs in
    # the PROJECT thread: its chat_id is the project's deterministic chat_id (not
    # the main/owner fallback), so its live card + owner mailbox route to the panel.
    from ouroboros.contracts.chat_id_policy import project_chat_id
    from ouroboros.projects_registry import get_project

    project = get_project(tmp_path, "research-1")
    assert project is not None
    assert task["chat_id"] == project["chat_id"] == project_chat_id("research-1")
    assert task["chat_id"] != 1  # not the owner-chat fallback
    # P2: the promoted task is BOUND to its project, so /api/state's all_task_bindings
    # surfaces it and the frontend never offers a stray "turn into project" button.
    from ouroboros.projects_registry import all_task_bindings
    assert all_task_bindings(tmp_path).get("abc12345") == project["chat_id"]


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
    bind_task_to_project(tmp_path, "task-7", "promoted", pchat)
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
    monkeypatch.setattr(server, "_perform_supervisor_restart", lambda ctx: performed.append(True))
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
    monkeypatch.setattr(server, "_perform_supervisor_restart", lambda ctx: performed.append(True))
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
    monkeypatch.setattr(omb, "write_owner_message", lambda *a, **k: delivered.append(a))

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
                        lambda drive, text, tid, msg_id=None, **k: msg_ids.append(msg_id))

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"pr": {"task": {"id": "pr", "chat_id": project_chat}, "last_heartbeat_at": 1.0}},
    )
    # Same client_message_id retried twice -> identical stable msg_id (dedup), not None (random).
    server._route_project_chat_to_running_task(ctx, project_chat, "go", "cmid-7")
    server._route_project_chat_to_running_task(ctx, project_chat, "go", "cmid-7")
    assert msg_ids == ["cmid-7:pr", "cmid-7:pr"]


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


def test_project_from_task_endpoint_creates_binding(tmp_path):
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import get_project, project_binding_for_task

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "abc123", "id": "task-abc123", "name": "Research thread"}

    resp = asyncio.run(api_project_from_task(_Req()))
    payload = json.loads(resp.body)
    assert resp.status_code == 200
    assert payload["project"]["id"] == "task-abc123"
    assert payload["project"]["name"] == "Research thread"
    assert payload["binding"]["task_id"] == "abc123"
    assert get_project(tmp_path, "task-abc123") is not None
    assert project_binding_for_task(tmp_path, "abc123")["project_id"] == "task-abc123"


def test_project_from_task_auto_names_from_objective(tmp_path):
    """One-click convert (owner P1): with NO name supplied the project name is
    derived from the task's own objective, not the live progress headline, and
    long objectives are collapsed/truncated. No human input, no extra LLM call."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import get_project
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    long_objective = "Собрать конкурентный обзор   рынка облачных GPU\nи свести в таблицу за квартал"
    write_task_result(tmp_path, "obj01", STATUS_RUNNING, objective=long_objective)

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "obj01", "id": "task-obj01"}  # no name → derive

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    name = payload["project"]["name"]
    assert "\n" not in name and "  " not in name  # whitespace collapsed
    assert name.startswith("Собрать конкурентный обзор")
    assert len(name) <= 60
    assert name != "task-obj01"  # not the bare id fallback
    assert get_project(tmp_path, "task-obj01")["name"] == name


def test_project_from_task_uses_neutral_name_when_nothing_derivable(tmp_path):
    """Nothing derivable (no title/objective/description) → a NEUTRAL 'New project'
    name, never the bare task id (the owner explicitly rejects task-… names)."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "noobj", "id": "task-noobj"}

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    assert payload["project"]["name"] == "New project"
    assert payload["project"]["name"] != "task-noobj"


def test_project_from_task_names_skill_lifecycle_task(tmp_path):
    """A skill-lifecycle (non-human-text) task carries no owner request, so naming
    derives a human label from the synthetic skill_lifecycle_<kind>_<target>_<job>
    id instead of dead-ending at 'New project' (P1)."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(
                state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
            )

        async def json(self):
            return {
                "task_id": "skill_lifecycle_install_travel-planner-notion-ai-obsidian_job1",
                "id": "task-skl1",
            }

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    name = payload["project"]["name"]
    assert name != "New project"
    assert name.startswith("Install skill")
    assert "travel-planner" in name
    assert len(name) <= 60


def test_project_from_task_uses_objective_hint_for_in_progress_direct_chat(tmp_path):
    """A still in-progress DIRECT chat task has no server-side title/objective/queue
    source, so the frontend's objective_hint (the owner's original request) names
    the project — not 'New project' or the bare id (P1, scope-review fix)."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "live9", "id": "task-live9",
                    "objective_hint": "исследуй рынок облачных GPU и собери таблицу"}

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    name = payload["project"]["name"]
    assert name.startswith("исследуй рынок облачных GPU")
    assert name not in ("New project", "task-live9")
    assert len(name) <= 60


def test_project_from_task_auto_names_from_live_queue_snapshot(tmp_path):
    """An in-progress conversion (no task_result objective written yet) derives the
    name from the LIVE queue snapshot, not the bare task id (F1 — fixes the observed
    task-id fallback when converting a still-running card)."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text(
        json.dumps({
            "running": [{"id": "live01", "task": {"id": "live01", "objective": "Изучить рынок облачных GPU и собрать таблицу"}}],
            "pending": [],
        }),
        encoding="utf-8",
    )

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "live01", "id": "task-live01"}  # no name, no task_result

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    name = payload["project"]["name"]
    assert name.startswith("Изучить рынок облачных GPU")
    assert name != "task-live01"  # not the bare id fallback


def test_all_task_project_bindings_exposes_project_id(tmp_path):
    """F4: the richer binding map carries project_id (not just chat_id) so a bound
    main-chat card can render a pointer that opens the bound project's panel."""
    from ouroboros.projects_registry import (
        all_task_project_bindings,
        bind_task_to_project,
        create_project,
    )

    proj = create_project(tmp_path, "market-thread", name="Market thread")
    bind_task_to_project(tmp_path, "tk9", "market-thread", proj["chat_id"])
    mapping = all_task_project_bindings(tmp_path)
    assert mapping["tk9"]["project_id"] == "market-thread"
    assert mapping["tk9"]["chat_id"] == int(proj["chat_id"])


def test_bound_project_history_backfills_task_progress(tmp_path):
    """A task converted into a project after it started keeps its original log
    rows, but project history resolves them through the binding."""
    import asyncio
    import json

    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "bound-progress", name="Bound progress")
    project_chat = int(project["chat_id"])
    bind_task_to_project(tmp_path, "task-1", "bound-progress", project_chat)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    with open(logs / "chat.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": "2026-01-01T00:00:00Z", "direction": "out", "text": "final answer", "chat_id": 1, "task_id": "task-1"}) + "\n")
        fh.write(json.dumps({"ts": "2026-01-01T00:00:01Z", "direction": "in", "text": "raw project chat", "chat_id": project_chat}) + "\n")
    with open(logs / "progress.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": "2026-01-01T00:00:02Z", "type": "send_message", "content": "working", "text": "working", "is_progress": True, "chat_id": 1, "task_id": "task-1", "format": "markdown"}) + "\n")

    api = make_chat_history_endpoint(tmp_path)

    class _Req:
        def __init__(self, params):
            self.query_params = params

    project_resp = json.loads(asyncio.run(api(_Req({"chat_id": str(project_chat)}))).body)
    project_texts = [m["text"] for m in project_resp["messages"]]
    assert "final answer" in project_texts
    assert "working" in project_texts
    assert "raw project chat" in project_texts

    main_resp = json.loads(asyncio.run(api(_Req({}))).body)
    main_texts = [m["text"] for m in main_resp["messages"]]
    assert "working" in main_texts  # main mirrors sanitized progress
    assert "raw project chat" not in main_texts
    # The bound task's RAW final-answer row (still stored with main chat_id 1) is
    # project-owned via the binding and must NOT leak into the штаб's main history.
    assert "final answer" not in main_texts


def test_bound_task_heartbeat_routes_to_project_panel(tmp_path):
    """A post-hoc bound task's heartbeat routes to its PROJECT panel: the durable
    binding takes PRECEDENCE over the task's original (main) chat_id, matching the
    send_message/log handlers (UI routing for a "Turn into project" running task)."""
    import time

    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor.events import _handle_task_heartbeat

    project = create_project(tmp_path, "hb-proj")
    project_chat = int(project["chat_id"])
    bind_task_to_project(tmp_path, "task-hb", "hb-proj", project_chat)

    pushed = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"task-hb": {"task": {"id": "task-hb", "type": "task", "chat_id": 1}, "started_at": time.time()}},
        bridge=types.SimpleNamespace(push_log=lambda payload: pushed.append(payload)),
    )
    _handle_task_heartbeat({"task_id": "task-hb", "phase": "running"}, ctx)
    assert pushed
    assert pushed[0]["chat_id"] == project_chat  # binding precedence, not the original main 1


def test_bound_task_media_routes_to_project_panel(tmp_path):
    """A post-hoc bound task's media (send_photo/send_video) routes to its PROJECT
    panel via the durable binding, not the task's original (main) chat_id —
    same precedence as the send_message/log/heartbeat handlers."""
    import base64

    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor.events import _handle_send_photo, _handle_send_video

    project = create_project(tmp_path, "media-proj")
    project_chat = int(project["chat_id"])
    bind_task_to_project(tmp_path, "task-m", "media-proj", project_chat)

    photo_sent, video_sent = [], []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        append_jsonl=lambda *a, **k: None,
        bridge=types.SimpleNamespace(
            send_photo=lambda cid, data, caption="", mime="": (photo_sent.append(cid) or (True, "")),
            send_video=lambda cid, data, caption="", mime="": (video_sent.append(cid) or (True, "")),
        ),
    )
    blob = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()
    _handle_send_photo({"task_id": "task-m", "chat_id": 1, "image_base64": blob, "mime": "image/png"}, ctx)
    _handle_send_video({"task_id": "task-m", "chat_id": 1, "video_base64": blob, "mime": "video/mp4"}, ctx)
    assert photo_sent == [project_chat]  # binding precedence, not the original main 1
    assert video_sent == [project_chat]


def test_bound_task_send_message_routes_future_events_to_project(tmp_path):
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor.events import _handle_send_message

    project = create_project(tmp_path, "future-events")
    project_chat = int(project["chat_id"])
    bind_task_to_project(tmp_path, "task-9", "future-events", project_chat)
    sent = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        send_with_budget=lambda *args, **kwargs: sent.append((args, kwargs)),
        append_jsonl=lambda *a, **k: None,
    )
    _handle_send_message({
        "chat_id": 1,
        "task_id": "task-9",
        "text": "future progress",
        "is_progress": True,
        "format": "markdown",
    }, ctx)
    assert sent
    assert sent[0][0][0] == project_chat


def test_chat_history_filters_by_thread(tmp_path):
    """api_chat_history returns only the requested thread's rows."""
    import asyncio
    import json

    from ouroboros.gateway.history import make_chat_history_endpoint

    # Register a project so its chat_id partitions out of the main view; a
    # large NON-project chat_id (transport mirror) must STAY in the main view.
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])
    transport_chat = 555000111

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    rows = [
        {"ts": "2026-06-13T00:00:01Z", "direction": "in", "text": "main hello", "chat_id": 1},
        {"ts": "2026-06-13T00:00:02Z", "direction": "out", "text": "main reply", "chat_id": 1},
        {"ts": "2026-06-13T00:00:03Z", "direction": "in", "text": "project hello", "chat_id": project_chat},
        {"ts": "2026-06-13T00:00:033Z", "direction": "system", "type": "task_summary", "text": "project summary", "chat_id": project_chat, "task_id": "pt"},
        {"ts": "2026-06-13T00:00:035Z", "direction": "in", "text": "transport mirror", "chat_id": transport_chat},
        {"ts": "2026-06-13T00:00:04Z", "direction": "out", "text": "a2a noise", "chat_id": -1001},
        {"ts": "2026-06-13T00:00:05Z", "direction": "out", "text": "legacy row (no chat_id)"},
    ]
    with open(logs / "chat.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    api = make_chat_history_endpoint(tmp_path)

    class _Req:
        def __init__(self, params):
            self.query_params = params

    main = json.loads(asyncio.run(api(_Req({}))).body)
    main_texts = [m["text"] for m in main["messages"]]
    assert "main hello" in main_texts and "main reply" in main_texts
    assert "legacy row (no chat_id)" in main_texts  # legacy rows are main-chat
    assert "transport mirror" in main_texts  # non-project transport stays visible
    assert "project hello" not in main_texts  # registered project partitions out
    assert "project summary" in main_texts  # штаб mirrors project summaries/progress
    assert "a2a noise" not in main_texts

    proj_resp = json.loads(asyncio.run(api(_Req({"chat_id": str(project_chat)}))).body)
    proj_texts = [m["text"] for m in proj_resp["messages"]]
    assert proj_texts and "project hello" in proj_texts
    assert "project summary" in proj_texts
    assert "main hello" not in proj_texts
    assert "transport mirror" not in proj_texts


def test_project_media_and_typing_broadcasts_carry_chat_id():
    """Photo/video/typing WS frames must carry chat_id so the client fan-out
    routes project-thread media to its panel (default-to-main would hide them)."""
    from supervisor.message_bus import LocalChatBridge

    bridge = LocalChatBridge()
    frames = []
    bridge._broadcast_fn = lambda payload: frames.append(payload)

    project_chat = 1234  # positive project-range id (not A2A, which is negative)
    bridge.send_chat_action(project_chat, "typing")
    bridge.send_photo(project_chat, b"img-bytes", caption="shot")
    bridge.send_video(project_chat, b"vid-bytes", caption="clip", mime="video/mp4")

    by_type = {f.get("type"): f for f in frames}
    assert by_type["typing"]["chat_id"] == project_chat
    assert by_type["photo"]["chat_id"] == project_chat
    assert by_type["video"]["chat_id"] == project_chat


def test_journal_write_rejects_over_limit_instead_of_truncating(tmp_path, monkeypatch):
    """A durable journal entry is never silently sliced: over-limit writes are
    rejected (the workpad_write contract), so cognitive memory stays whole."""
    import types

    # Project store paths resolve via config.DATA_DIR (NOT ctx.drive_root); isolate
    # it to tmp_path so a plain local pytest run (no OUROBOROS_DATA_DIR set) never
    # writes into the real data dir.
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.tools.project_journal import _MAX_TEXT_CHARS, _journal_read, _journal_write

    ctx = types.SimpleNamespace(project_id="journal-reject-test", task_id="t1", drive_root=tmp_path)
    assert _journal_write(ctx, "note", "hello milestone", "").startswith("OK:")
    over = _journal_write(ctx, "note", "Z" * (_MAX_TEXT_CHARS + 50), "")
    assert "TOOL_ARG_ERROR" in over and "exceeds" in over
    body = _journal_read(ctx, "", 30)
    assert "hello milestone" in body
    assert "Z" * 200 not in body  # the rejected over-limit text was never stored


# --- WS1: multi-task chat steering (steer_task + current_chat.running_tasks) ---

def test_steer_task_tool_emits_event_with_target_and_client_id(tmp_path):
    """The agent's steer_task choice emits a transport event (target + message +
    chat + originating message id); the supervisor performs the actual delivery."""
    from ouroboros.tools.control import _steer_task

    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1,
        task_metadata={"client_message_id": "cm-42"},
    )
    out = _steer_task(ctx, "abc12345", "also add the benchmarks slide")
    assert out.startswith("✉️ Steering task abc12345")
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "steer_task"
    assert evt["target_task_id"] == "abc12345"
    assert evt["message"] == "also add the benchmarks slide"
    assert evt["chat_id"] == 1
    assert evt["client_message_id"] == "cm-42"


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
