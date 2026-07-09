from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ouroboros.gateway.history import make_chat_history_endpoint


def test_chat_history_preserves_subagent_lane_group_metadata(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-06-05T00:00:00Z",
                "content": "subagent queued",
                "task_id": "child1",
                "subagent_event": "scheduled",
                "model_lane": "review",
                "requested_model_lane": "review",
                "effective_model_lane": "review",
                "model": "review-a",
                "task_group_id": "group1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "child1")
    assert rec["model_lane"] == "review"
    assert rec["requested_model_lane"] == "review"
    assert rec["effective_model_lane"] == "review"
    assert rec["model"] == "review-a"
    assert rec["task_group_id"] == "group1"


def test_chat_history_replays_delivered_document_row(tmp_path):
    """A persisted document chat row is replayed as a msg_type=document record so
    the frontend rebuilds the file bubble on reload from the durable URL."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-07-09T00:00:00Z",
                "direction": "out",
                "chat_id": 1,
                "user_id": 7,
                "text": "quarterly numbers",
                "type": "document",
                "filename": "report.pdf",
                "mime": "application/pdf",
                "download_url": "/api/files/download?path=Desktop/report.pdf",
                "caption": "quarterly numbers",
                "task_id": "t-doc",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("msg_type") == "document")
    assert rec["role"] == "assistant"
    assert rec["filename"] == "report.pdf"
    assert rec["mime"] == "application/pdf"
    assert rec["download_url"] == "/api/files/download?path=Desktop/report.pdf"
    assert rec["caption"] == "quarterly numbers"


def test_chat_history_backfills_from_rotated_archive(tmp_path):
    """The live chat.jsonl is rotated to archive/chat_<ts>.jsonl at ~800KB. History
    replay must backfill from the most recent archive(s) so a rotation does not
    silently erase the visible conversation — including delivered file bubbles —
    that scrolled just before it (BIBLE P1: no silent loss)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()

    # Older conversation + a delivered document, now rotated into the archive.
    (archive / "chat_20260709T165729.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-07-09T16:00:00Z",
                "direction": "in",
                "chat_id": 1,
                "user_id": 1,
                "text": "older message before the rotation",
            }
        )
        + "\n"
        + json.dumps(
            {
                "ts": "2026-07-09T16:05:00Z",
                "direction": "out",
                "chat_id": 1,
                "user_id": 7,
                "text": "here is the old pdf",
                "type": "document",
                "filename": "archived_report.pdf",
                "mime": "application/pdf",
                "download_url": "/api/files/download?path=Desktop/archived_report.pdf",
                "caption": "here is the old pdf",
                "task_id": "t-old-doc",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Small live file written after the rotation.
    (logs / "chat.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-07-09T17:29:00Z",
                "direction": "in",
                "chat_id": 1,
                "user_id": 1,
                "text": "newest live message",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "50"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    texts = [item.get("text", "") for item in payload]
    # The archived human message survives the rotation.
    assert "older message before the rotation" in texts
    assert "newest live message" in texts
    # The archived delivered-document row is replayed as a document bubble.
    doc = next(item for item in payload if item.get("msg_type") == "document")
    assert doc["filename"] == "archived_report.pdf"
    assert doc["download_url"] == "/api/files/download?path=Desktop/archived_report.pdf"
    # Chronological reassembly: archived rows precede the newer live row.
    assert texts.index("older message before the rotation") < texts.index("newest live message")


def test_chat_history_backfill_quota_is_thread_aware(tmp_path):
    """Regression for the v6.58.5 review finding: the archive-backfill human-row
    quota must be counted with the SAME thread filter used at render time. A
    project-thread request whose LIVE file already holds `want` unrelated
    main-chat rows must still read the archive so rotated PROJECT rows/documents
    are recovered (they used to be skipped because the quota counted every live
    human row before the thread filter)."""
    from ouroboros import projects_registry
    from ouroboros.contracts.chat_id_policy import project_chat_id

    # A registered project so its chat_id classifies as a project thread.
    projects_registry.create_project(tmp_path, "proj_demo", name="Demo")
    pc = project_chat_id("proj_demo")

    logs = tmp_path / "logs"
    logs.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()

    # Rotated archive holds a PROJECT-thread delivered document.
    (archive / "chat_20260709T150000.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-07-09T14:00:00Z",
                "direction": "out",
                "chat_id": pc,
                "user_id": 7,
                "text": "project pdf",
                "type": "document",
                "filename": "project_report.pdf",
                "mime": "application/pdf",
                "download_url": "/api/files/download?path=Desktop/project_report.pdf",
                "caption": "project pdf",
                "task_id": "t-proj-doc",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Live file: only UNRELATED main-chat rows (chat_id defaults to 1).
    (logs / "chat.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "ts": f"2026-07-09T17:0{i}:00Z",
                    "direction": "in" if i % 2 else "out",
                    "chat_id": 1,
                    "user_id": 1,
                    "text": f"main chat row {i}",
                }
            )
            for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    # want=2 (< the 4 unrelated live rows): old quota would stop before reading
    # the archive; thread-aware quota reads it because 0 live rows match `pc`.
    response = asyncio.run(
        endpoint(SimpleNamespace(query_params={"chat_id": str(pc), "n_human": "2"}))
    )
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    doc = next(item for item in payload if item.get("msg_type") == "document")
    assert doc["filename"] == "project_report.pdf"
    # And unrelated main-chat rows do NOT leak into the project thread.
    assert not any(item.get("text", "").startswith("main chat row") for item in payload)


def test_chat_history_preserves_subagent_accept_markers(tmp_path):
    """WS8 accept/count markers must survive chat-history replay (gateway contract)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-06-08T00:00:00Z",
                "content": "subagent queued",
                "task_id": "child2",
                "subagent_event": "scheduled",
                "accepted": True,
                "active_subagent_count": 3,
                "max_active_subagents": 6,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "child2")
    assert rec["accepted"] is True
    assert rec["active_subagent_count"] == 3
    assert rec["max_active_subagents"] == 6


def test_chat_history_preserves_subagent_reconciliation_metadata(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-06-27T00:00:00Z",
                "content": "subagent queued behind active cap",
                "task_id": "child3",
                "subagent_event": "scheduled",
                "queued_behind_active_cap": True,
                "required_capabilities": ["shell", "vcs"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "child3")
    assert rec["queued_behind_active_cap"] is True
    assert rec["required_capabilities"] == ["shell", "vcs"]
