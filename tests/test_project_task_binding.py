"""Binding a task to a project, and the events that follow the binding.

Split verbatim out of ``tests/test_promote_chat_flow.py`` by theme. This module owns
the project-from-task endpoint and the names it derives, the binding inventory, and
the routing of a bound task's history, heartbeat, media and messages into the project
panel — including the thread filter on chat history and the journal write that refuses
rather than truncates.
"""

from __future__ import annotations

import types


from tests._promote_chat_shared import _isolated_projects_root  # noqa: F401  (autouse fixture applies on import)


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
    bind_task_to_project(tmp_path, "tk9", "market-thread", proj["chat_id"], origin={"absent": "system"})
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
    bind_task_to_project(tmp_path, "task-1", "bound-progress", project_chat, origin={"absent": "system"})
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
    bind_task_to_project(tmp_path, "task-hb", "hb-proj", project_chat, origin={"absent": "system"})

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
    bind_task_to_project(tmp_path, "task-m", "media-proj", project_chat, origin={"absent": "system"})

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
    bind_task_to_project(tmp_path, "task-9", "future-events", project_chat, origin={"absent": "system"})
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
