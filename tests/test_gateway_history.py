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


def test_chat_history_replays_typed_direct_error_terminal_status(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-27T00:00:00Z",
            "direction": "out",
            "chat_id": 1,
            "user_id": 7,
            "text": "error",
            "task_id": "failed-task",
            "task_terminal_status": "failed",
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "failed-task")
    assert rec["task_terminal_status"] == "failed"


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
                "size_bytes": 4096,
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
    assert rec["task_id"] == "t-doc"
    assert rec["size_bytes"] == 4096


def test_chat_history_replays_links_with_task_grouping_fields(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    actions = [
        {"label": "Report", "url": "https://example.com/report"},
        {"label": "Dashboard", "url": "http://example.com/dashboard"},
    ]
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-30T00:00:00Z",
            "direction": "out",
            "chat_id": 1,
            "user_id": 7,
            "text": "References",
            "type": "links",
            "title": "References",
            "actions": actions,
            "task_id": "task-links",
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(
        query_params={"limit": "10"},
    )))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("msg_type") == "links")
    assert rec["role"] == "assistant"
    assert rec["title"] == "References"
    assert rec["actions"] == actions
    assert rec["task_id"] == "task-links"


def test_chat_history_replays_durable_photo_and_keeps_legacy_row_as_text(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [
        {
            "ts": "2026-08-21T00:00:00Z", "direction": "out", "chat_id": 1,
            "user_id": 7, "text": "durable shot", "type": "photo",
            "mime": "image/png", "caption": "durable shot", "task_id": "media-task",
            "download_url": "/api/tasks/media-task/artifacts/chat-media-" + "a" * 64 + ".png",
        },
        {
            "ts": "2026-08-21T00:01:00Z", "direction": "out", "chat_id": 1,
            "user_id": 7, "text": "durable clip", "type": "video",
            "mime": "video/mp4", "caption": "durable clip", "task_id": "media-task",
            "download_url": "/api/tasks/media-task/artifacts/chat-media-" + "b" * 64 + ".mp4",
        },
        {
            "ts": "2026-08-21T00:02:00Z", "direction": "out", "chat_id": 1,
            "user_id": 7, "text": "legacy shot", "type": "photo", "mime": "image/png",
            "task_id": "still-running",
        },
    ]
    (logs / "chat.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    durable = next(item for item in payload if item["text"] == "durable shot")
    video = next(item for item in payload if item["text"] == "durable clip")
    legacy = next(item for item in payload if item["text"] == "legacy shot")
    assert durable["msg_type"] == "photo"
    assert durable["download_url"].endswith(".png")
    assert durable["task_id"] == "media-task"
    assert video["msg_type"] == "video"
    assert video["download_url"].endswith(".mp4")
    assert video["task_id"] == "media-task"
    assert "msg_type" not in legacy
    assert legacy["task_id"] == "still-running"


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


def test_chat_history_marks_malformed_rows_incomplete(tmp_path):
    """A skipped JSONL row must not make the bounded window claim completeness."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_bytes(
        json.dumps({
            "ts": "2026-08-21T18:00:00Z",
            "direction": "in",
            "chat_id": 1,
            "text": "surviving row",
        }).encode("utf-8")
        + b"\n{malformed row\n"
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))

    assert any(item.get("text") == "surviving row" for item in payload["messages"])
    assert payload["window"]["complete"] is False
    assert "chat_malformed_jsonl" in payload["window"]["truncated_by"]


def test_history_gap_metadata_keeps_reader_failures_fail_soft(tmp_path, monkeypatch):
    """Adding gap metadata must not turn the existing best-effort reader into a 500."""
    from ouroboros.gateway import history

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic reader failure")

    monkeypatch.setattr(history, "_read_chat_history_entries", fail)
    rows, quota, gaps = history._collect_chat_rows(
        tmp_path / "chat.jsonl",
        tmp_path / "archive",
        10,
        lambda *_args: True,
        {},
        include_gaps=True,
    )

    assert rows == []
    assert quota == 0
    assert gaps == set()


def test_history_gap_metadata_keeps_legacy_reader_call_shape(tmp_path, monkeypatch):
    """The opt-in metadata path must not change existing private helper callers."""
    from ouroboros.gateway import history

    row = {
        "direction": "in",
        "chat_id": 1,
        "text": "legacy helper row",
        "ts": "2026-08-21T18:01:00Z",
    }

    def legacy(_live, _archive, _want, _predicate):
        return [row]

    monkeypatch.setattr(history, "_read_chat_history_entries", legacy)
    rows, quota = history._collect_chat_rows(
        tmp_path / "chat.jsonl",
        tmp_path / "archive",
        10,
        lambda *_args: True,
        {},
    )

    assert rows[0]["text"] == "legacy helper row"
    assert quota == 1


def test_rotated_reader_legacy_mode_keeps_parser_call_shape(tmp_path, monkeypatch):
    """Non-metadata callers retain the old iterator signature and cost path."""
    from ouroboros.gateway import _helpers

    logs = tmp_path / "logs"
    logs.mkdir()
    live = logs / "chat.jsonl"
    live.write_text("{}\n", encoding="utf-8")

    def legacy(path, *, tail_bytes=None):
        assert tail_bytes is None or isinstance(tail_bytes, int)
        return iter([{"text": str(path), "direction": "in"}])

    monkeypatch.setattr(_helpers, "iter_jsonl_objects", legacy)
    entries = _helpers.read_rotated_jsonl_entries(
        live,
        tmp_path / "archive",
        "chat",
        10,
        lambda _entry: True,
    )

    assert isinstance(entries, list)
    assert entries[0]["direction"] == "in"


def test_progress_history_backfills_from_rotated_archive(tmp_path):
    """progress.jsonl now rotates to archive/progress_<ts>.jsonl like chat. History
    replay must backfill rotated progress rows (newest-first, until the n_progress
    quota) so a rotation does not silently erase live task cards (BIBLE P1)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()

    (archive / "progress_20260808T010000.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-08T00:30:00Z",
            "content": "archived step",
            "task_id": "rotated-task",
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-08T01:30:00Z",
            "content": "live step",
            "task_id": "live-task",
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "chat.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"n_progress": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    texts = [item.get("text", "") for item in payload]
    assert "archived step" in texts
    assert "live step" in texts
    # Chronological reassembly: archived rows precede the newer live row.
    assert texts.index("archived step") < texts.index("live step")


def test_progress_archive_not_read_when_live_satisfies_quota(tmp_path):
    """Archive backfill stops once the live window already satisfies the filtered
    quota — the rotated segments are not touched on the common path."""
    logs = tmp_path / "logs"
    logs.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "progress_20260808T010000.jsonl").write_text(
        json.dumps({"ts": "2026-08-08T00:30:00Z", "content": "old archived", "task_id": "t-old"}) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        "\n".join(
            json.dumps({"ts": f"2026-08-08T01:00:{i:02d}Z", "content": f"live-{i}", "task_id": "t1"})
            for i in range(5)
        ) + "\n",
        encoding="utf-8",
    )
    (logs / "chat.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"n_progress": "3"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    texts = [item.get("text", "") for item in payload]
    assert "old archived" not in texts  # quota satisfied by the live window
    assert texts == ["live-2", "live-3", "live-4"]


def test_history_annotation_after_quota_resolves_in_window_card(tmp_path):
    """Post-quota annotation behavior change (v6.90.x P2): a terminal task whose
    truth-anchor rows fell OUTSIDE the emitted window (here: its task_summary chat
    row evicted by the n_human quota) still resolves its surviving in-window
    progress card with the full terminal truth. Pre-change, the summary row seen
    during the full parse suppressed the progress-row anchor and the truth was
    applied only to rows the quota then dropped."""
    logs = tmp_path / "logs"
    logs.mkdir()
    chat_rows = [
        json.dumps({
            "ts": "2026-08-08T00:00:00Z",
            "direction": "system",
            "type": "task_summary",
            "task_id": "windowed",
            "chat_id": 1,
            "text": "Task windowed finished.",
            "tool_calls": 1,
            "rounds": 1,
        })
    ]
    # Newer human rows push the summary row out of a small n_human window.
    chat_rows += [
        json.dumps({
            "ts": f"2026-08-08T00:10:{i:02d}Z",
            "direction": "in" if i % 2 else "out",
            "chat_id": 1,
            "text": f"newer human {i}",
        })
        for i in range(4)
    ]
    (logs / "chat.jsonl").write_text("\n".join(chat_rows) + "\n", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-08T00:05:00Z",
            "content": "still visible progress",
            "task_id": "windowed",
        }) + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "windowed.json").write_text(
        json.dumps({
            "_schema_version": 1,
            "task_id": "windowed",
            "status": "completed",
            "cost_usd": 0.42,
            "cost_final": True,
            "reason_code": "",
        }),
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(
        endpoint(SimpleNamespace(query_params={"n_human": "2", "n_progress": "10"}))
    )
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    assert not any(item.get("system_type") == "task_summary" for item in payload)
    rec = next(item for item in payload if item.get("task_id") == "windowed")
    assert rec["task_terminal_status"] == "completed"
    # Full terminal truth (cost) landed on the in-window progress anchor —
    # ABI-3: the stored legacy spelling replays under the honest name only.
    assert rec["accounted_upper_bound_usd"] == 0.42
    assert "cost_usd" not in rec
    assert rec["cost_final"] is True


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


def test_chat_history_task_summary_row_passes_flat_cost_fields_through(tmp_path):
    """v6.82 P1: agent_task_pipeline writes the pre-synthesis cost snapshot onto
    the task_summary chat row; history replay must pass those flat fields
    through so a reload still shows the card's cost (no result file present)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    # A STORED legacy snapshot (retired with-children spelling): ABI-3 replay
    # converts it to the honest name instead of copying the alias out.
    row_cost = {
        "cost_accounting_status": "available",
        "cost_final": False,
        "cost_usd_with_children": 1.234567,
        "cost_with_children_partial": True,
        "reserved_usd": 0.25,
        "unresolved_upper_bound_usd": 0.5,
        "unknown_unmetered": 0,
    }
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-07-29T00:00:00Z",
            "direction": "system",
            "type": "task_summary",
            "task_id": "cost-summary",
            "chat_id": 1,
            "text": "Task cost-summary finished.",
            "tool_calls": 2,
            "rounds": 3,
            **row_cost,
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "cost-summary")
    # openness fields pass through; the legacy pair converts (deprecated-wins)
    for field in ("cost_accounting_status", "cost_final",
                  "cost_with_children_partial", "reserved_usd",
                  "unresolved_upper_bound_usd", "unknown_unmetered"):
        assert rec.get(field) == row_cost[field]
    assert rec["accounted_upper_bound_usd_with_children"] == 1.234567
    assert "cost_usd_with_children" not in rec
    # The snapshot honestly lacks an own-cost amount — replay must not
    # fabricate one under either spelling.
    assert "cost_usd" not in rec and "accounted_upper_bound_usd" not in rec


def test_chat_history_replays_task_summary_finality_without_task_result(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-09-03T00:00:00Z",
            "direction": "system",
            "type": "task_summary",
            "task_id": "open-summary",
            "chat_id": 1,
            "text": "Narrative written before finalization.",
            "outcome_phase": "warn",
            "outcome_final": False,
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "open-summary")
    assert rec["outcome_phase"] == "warn"
    assert rec["outcome_final"] is False
    assert "task_terminal_status" not in rec


def test_chat_history_settled_result_overrides_pre_final_summary_finality(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-09-03T00:00:00Z",
            "direction": "system",
            "type": "task_summary",
            "task_id": "settled-summary",
            "chat_id": 1,
            "text": "Narrative written before finalization.",
            "outcome_phase": "working",
            "outcome_final": False,
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "settled-summary.json").write_text(
        json.dumps({"_schema_version": 1, "task_id": "settled-summary", "status": "completed"}),
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "settled-summary")
    assert rec["outcome_phase"] == "done"
    assert rec["outcome_final"] is True


def test_chat_history_attaches_terminal_cost_truth_from_task_result(tmp_path):
    """v6.82 P1: a terminal task_results/<id>.json carries the final cost truth;
    it is attached to the surviving progress anchor on replay. ABI-3
    (fix-round-2 conversion of this OLD-ABI contract test): the stored row is
    LEGACY-spelled, and the outbound frame carries the honest names only."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-07-29T00:00:00Z",
            "content": "working on it",
            "task_id": "cost-terminal",
            "chat_id": 1,
        }) + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "cost-terminal.json").write_text(
        json.dumps({
            "_schema_version": 1,
            "task_id": "cost-terminal",
            "status": "completed",
            "cost_usd": 1.5,
            "cost_accounting_status": "available",
            "cost_final": True,
            "cost_usd_with_children": 2.75,
            "cost_with_children_partial": False,
        }),
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "cost-terminal")
    assert rec["task_terminal_status"] == "completed"
    assert rec["accounted_upper_bound_usd"] == 1.5
    assert rec["cost_final"] is True
    assert rec["accounted_upper_bound_usd_with_children"] == 2.75
    assert rec["cost_with_children_partial"] is False
    assert rec["cost_accounting_status"] == "available"
    assert "cost_usd" not in rec and "cost_usd_with_children" not in rec


def test_chat_history_terminal_cost_truth_overrides_row_embedded_snapshot(tmp_path):
    """v6.82 P1 precedence: the persisted task result's cost fields OVERRIDE the
    row-embedded (pre-synthesis, non-final) task_summary snapshot on replay."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-07-29T00:00:00Z",
            "direction": "system",
            "type": "task_summary",
            "task_id": "cost-override",
            "chat_id": 1,
            "text": "Task cost-override finished.",
            "tool_calls": 1,
            "rounds": 2,
            "cost_accounting_status": "available",
            "cost_final": False,
            "cost_usd_with_children": 1.0,
            "cost_with_children_partial": True,
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "cost-override.json").write_text(
        json.dumps({
            "_schema_version": 1,
            "task_id": "cost-override",
            "status": "completed",
            "cost_usd": 0.9,
            "cost_accounting_status": "available",
            "cost_final": True,
            "cost_usd_with_children": 2.5,
            "cost_with_children_partial": False,
        }),
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "cost-override")
    # ABI-3: both the row-embedded snapshot and the stored result are
    # legacy-spelled; the override lands under the honest names only.
    assert rec["cost_final"] is True
    assert rec["accounted_upper_bound_usd_with_children"] == 2.5
    assert rec["cost_with_children_partial"] is False
    assert rec["accounted_upper_bound_usd"] == 0.9
    assert "cost_usd" not in rec and "cost_usd_with_children" not in rec


def test_chat_history_preserves_nullable_cost_status_and_bounds(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    cost_meta = {
        "cost_usd": None,
        "cost_accounting_status": "unavailable",
        "cost_accounting_error": "ledger_unavailable",
        "cost_final": False,
        "cost_usd_with_children": None,
        "cost_with_children_partial": True,
        "reserved_usd": None,
        "unresolved_upper_bound_usd": None,
        "unknown_unmetered": None,
    }
    (logs / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-07-14T00:00:00Z",
            "content": "terminal accounting status",
            "task_id": "cost-unavailable",
            **cost_meta,
        }) + "\n",
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "cost-unavailable")
    # ABI-3: the legacy null pair converts to the honest names (still null —
    # never fabricated into $0); every openness marker survives verbatim.
    assert rec["accounted_upper_bound_usd"] is None
    assert rec["accounted_upper_bound_usd_with_children"] is None
    assert "cost_usd" not in rec and "cost_usd_with_children" not in rec
    for field, expected in cost_meta.items():
        if field in ("cost_usd", "cost_usd_with_children"):
            continue
        assert field in rec
        assert rec[field] == expected


def test_chat_history_preserves_cancelable_marker(tmp_path):
    """v6.82 (P5): the supervisor's host-attested `cancelable` progress-meta marker
    must survive history replay, or a reloaded live root card would lose its
    "Cancel run" action while the task is still running."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-07-29T00:00:00Z",
                "content": "Scheduled task root1: do the thing",
                "task_id": "root1",
                "is_progress": True,
                "cancelable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]

    rec = next(item for item in payload if item.get("task_id") == "root1")
    assert rec["cancelable"] is True


def test_main_history_keeps_only_host_project_root_completion_while_project_keeps_detail(
    tmp_path,
):
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "launch", name="Launch 🚀")
    project_chat = int(project["chat_id"])
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "chat.jsonl").write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "ts": "2026-08-21T00:00:02Z",
                    "direction": "system",
                    "chat_id": project_chat,
                    "type": "task_summary",
                    "task_id": "root-project",
                    "text": "Detailed Project summary",
                    "tool_calls": 3,
                    "rounds": 4,
                },
                {
                    "ts": "2026-08-21T00:00:03Z",
                    "direction": "system",
                    "chat_id": 1,
                    "type": "project_completion_summary",
                    "task_id": "root-project",
                    "project_id": "launch",
                    "project_name": "Launch 🚀",
                    "target_label": "Launch 🚀 › Ship release",
                    "status": "completed",
                    "text": "Launch 🚀 › Ship release · Completed",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-08-21T00:00:01Z",
                "chat_id": project_chat,
                "content": "Project-only progress",
                "task_id": "root-project",
                "is_progress": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)
    main = json.loads(
        asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "1"}))).body
    )["messages"]
    project_rows = json.loads(
        asyncio.run(
            endpoint(SimpleNamespace(query_params={"chat_id": str(project_chat)}))
        ).body
    )["messages"]

    assert [row["system_type"] for row in main if row.get("system_type")] == [
        "project_completion_summary"
    ]
    completion = main[0]
    assert completion["task_id"] == "root-project"
    assert completion["project_id"] == "launch"
    assert completion["project_name"] == "Launch 🚀"
    assert completion["target_label"] == "Launch 🚀 › Ship release"
    assert completion["status"] == "completed"
    assert any(row.get("is_progress") for row in project_rows)
    assert any(row.get("system_type") == "task_summary" for row in project_rows)


def test_main_history_admits_project_started_row_and_project_thread_excludes_it(
    tmp_path,
):
    """B1 admission: the durable `project_started` row is Main-only — admitted
    to the Main view with its event-time metadata passthrough (same keys as the
    completion row), excluded from the Project's own thread view, and stable
    across a replayed read."""
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "launch", name="Launch 🚀")
    project_chat = int(project["chat_id"])
    bind_task_to_project(
        tmp_path, "root-project", project["id"], project_chat,
        origin={"absent": "system"},
    )
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "chat.jsonl").write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "ts": "2026-08-21T00:00:01Z",
                    "direction": "system",
                    "chat_id": 1,
                    "type": "project_started",
                    "task_id": "root-project",
                    "project_id": "launch",
                    "project_name": "Launch 🚀",
                    "target_label": "Launch 🚀 › Ship release",
                    "text": "Launch 🚀 › Ship release · Started\nWork is running in this Project.",
                },
                {
                    "ts": "2026-08-21T00:00:02Z",
                    "direction": "system",
                    "chat_id": project_chat,
                    "type": "task_summary",
                    "task_id": "root-project",
                    "text": "Detailed Project summary",
                    "tool_calls": 3,
                    "rounds": 4,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    endpoint = make_chat_history_endpoint(tmp_path)

    def _messages(chat_id):
        return json.loads(
            asyncio.run(
                endpoint(SimpleNamespace(query_params={"chat_id": str(chat_id)}))
            ).body
        )["messages"]

    main = _messages(1)
    assert [row["system_type"] for row in main if row.get("system_type")] == [
        "project_started"
    ]
    started = next(row for row in main if row["system_type"] == "project_started")
    assert started["role"] == "system"
    assert started["task_id"] == "root-project"
    assert started["project_id"] == "launch"
    assert started["project_name"] == "Launch 🚀"
    assert started["target_label"] == "Launch 🚀 › Ship release"
    assert "status" not in started  # the started row carries no outcome yet

    project_rows = _messages(project_chat)
    assert all(
        row.get("system_type") != "project_started" for row in project_rows
    )
    assert any(row.get("system_type") == "task_summary" for row in project_rows)

    # Replay-safe: a second read of the same durable stream is identical.
    assert _messages(1) == main
