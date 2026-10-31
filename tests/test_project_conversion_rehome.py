"""Converting a task into a Project adds a read lens without copying dialogue.

- the owner's ORIGINAL canonical row is projected into the Project room;
- a subagent's progress classifies into the root project's thread by lineage;
- the main chat keeps one owner row and no physical mirror bubble.
"""

from __future__ import annotations

from ouroboros import dialogue_store
from tests import _dialogue as dialogue_test

import asyncio
import json
from types import SimpleNamespace


def _request(tmp_path, body):
    async def _json():
        return body

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
        json=_json,
    )


def _history(tmp_path, chat_id):
    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    resp = asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": str(chat_id), "limit": "50"})))
    return json.loads(resp.body.decode("utf-8"))["messages"]


def test_conversion_projects_canonical_owner_request_and_rehomes_subagents(tmp_path):
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import project_binding_for_task, project_chat_for_task
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    dialogue_store.append(
        tmp_path, "chat",
        # The owner's original request lives in the MAIN chat (chat_id 1). It is
        # logged at receive time with NO task_id. The immutable binding stores a
        # source reference so the Project lens can render this same row.
        {"ts": "2026-06-18T00:00:00Z", "direction": "in", "chat_id": 1,
         "client_message_id": "owner-1", "text": "build cyberpunk racing"},
    )
    # v6.73.0: the origin identity is captured at ingress and persisted on the
    # TASK RECORD; post-hoc conversion reads it from there (never a content lookup).
    write_task_result(
        tmp_path, "root", STATUS_RUNNING,
        objective="build cyberpunk racing",
        origin_message_ref=build_owner_message_ref(
            chat_id=1, client_message_id="owner-1",
            ts="2026-06-18T00:00:00Z", text="build cyberpunk racing",
        ),
        origin_message_text="build cyberpunk racing",
    )

    resp = asyncio.run(api_project_from_task(_request(
        tmp_path,
        {"task_id": "root", "id": "task-root", "objective_hint": "build cyberpunk racing"},
    )))
    payload = json.loads(resp.body.decode("utf-8"))
    proj_chat = int(payload["project"]["chat_id"])
    assert proj_chat == project_chat_for_task(tmp_path, "root") > 0
    binding = project_binding_for_task(tmp_path, "root")
    assert binding["source_ref"]["client_message_id"] == "owner-1"
    assert binding["source_text"] == "build cyberpunk racing"

    # A subagent emits progress; its rows carry main chat_id but lineage roots at "root".
    dialogue_store.append(tmp_path, "progress", {
        "ts": "2026-06-18T00:01:00Z", "content": "child working",
        "chat_id": 1, "task_id": "child", "parent_task_id": "root",
        "root_task_id": "root", "subagent_event": "progress",
    })

    project_view = _history(tmp_path, proj_chat)
    # The canonical owner request is the first Project message...
    assert any(m.get("role") == "user" and "cyberpunk racing" in m.get("text", "") for m in project_view)
    # ...and the subagent progress re-homes into the project thread by lineage (C4.4).
    assert any(m.get("task_id") == "child" for m in project_view)
    # BUG 1 (ordering): the owner's row sorts to the TOP using its ORIGINAL send ts
    # (00:00:00), ahead of the working bubble (00:01:00) — not stamped 'now' at the
    # bottom. History replay sorts purely by ts.
    owner_idx = next(i for i, m in enumerate(project_view)
                     if m.get("role") == "user" and "cyberpunk racing" in m.get("text", ""))
    child_idx = next(i for i, m in enumerate(project_view) if m.get("task_id") == "child")
    assert owner_idx < child_idx, f"owner row must precede the working bubble (got {owner_idx} vs {child_idx})"
    assert str(project_view[owner_idx].get("ts", "")).startswith("2026-06-18T00:00:00"), \
        project_view[owner_idx].get("ts")

    main_view = _history(tmp_path, 1)
    # Main still renders that same canonical row exactly once.
    owner_user_rows = [m for m in main_view if m.get("role") == "user" and "cyberpunk racing" in m.get("text", "")]
    assert len(owner_user_rows) == 1
    raw_rows = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    assert len([row for row in raw_rows if "cyberpunk racing" in row.get("text", "")]) == 1
    # The subagent's raw chat stays out of main; only sanitized progress may mirror.
    assert all(not (m.get("task_id") == "child" and m.get("role") == "user") for m in main_view)


def test_repeat_conversion_does_not_duplicate_owner_message(tmp_path):
    """A repeated conversion keeps one canonical row and one immutable ref."""
    from ouroboros.gateway.projects import api_project_from_task

    dialogue_store.append(
        tmp_path, "chat", {
            "ts": "2026-06-18T00:00:00Z",
            "direction": "in",
            "chat_id": 1,
            "client_message_id": "owner-repeat",
            "text": "build cyberpunk racing",
        },
    )

    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(
        tmp_path, "root", STATUS_RUNNING,
        objective="build cyberpunk racing",
        origin_message_ref=build_owner_message_ref(
            chat_id=1, client_message_id="owner-repeat",
            ts="2026-06-18T00:00:00Z", text="build cyberpunk racing",
        ),
        origin_message_text="build cyberpunk racing",
    )
    body = {"task_id": "root", "id": "task-root", "objective_hint": "build cyberpunk racing"}
    asyncio.run(api_project_from_task(_request(tmp_path, body)))
    asyncio.run(api_project_from_task(_request(tmp_path, body)))

    rows = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
    canonical = [r for r in rows if r.get("direction") == "in" and "cyberpunk racing" in r.get("text", "")]
    assert len(canonical) == 1


def test_conversion_reuses_proactive_suggested_name(tmp_path):
    """Cluster B: turn-into-project reuses the LLM name the proactive card namer already
    coined (suggested_name on the running task result) — no extra LLM call, never a bare
    'task-…' id. An explicit caller name still wins; absent a preset it would fall to the
    inline LLM namer / heuristic."""
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(
        tmp_path, "root", STATUS_RUNNING,
        suggested_name="Cyber Racing Game",
        objective="build a cyberpunk racing game",
    )

    resp = asyncio.run(api_project_from_task(_request(
        tmp_path, {"task_id": "root", "id": "task-root", "objective_hint": "build a cyberpunk racing game"},
    )))
    payload = json.loads(resp.body.decode("utf-8"))
    assert payload["project"]["name"] == "Cyber Racing Game", payload["project"]["name"]
