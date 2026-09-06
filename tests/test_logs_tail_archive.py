"""api_logs_tail bounded-tail + archive-backfill tests (v6.90.x P2)."""

from __future__ import annotations

import json

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.logs import api_logs_tail


def _app(data):
    app = Starlette(routes=[Route("/api/logs/{name}", endpoint=api_logs_tail, methods=["GET"])])
    app.state.drive_root = data
    return app


def test_logs_tail_backfills_from_rotated_progress_archive(tmp_path):
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    archive = data / "archive"
    archive.mkdir()
    (archive / "progress_20260101T000100.jsonl").write_text(
        "\n".join(
            json.dumps({"ts": f"2026-01-01T00:00:{i:02d}Z", "content": f"archived-{i}", "task_id": "t1"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:02:00Z", "content": "live-row", "task_id": "t1"}) + "\n",
        encoding="utf-8",
    )

    payload = TestClient(_app(data)).get("/api/logs/progress?limit=10").json()

    contents = [row["content"] for row in payload["entries"]]
    assert contents == ["archived-0", "archived-1", "archived-2", "live-row"]


def test_logs_tail_backfills_from_rotated_events_archive(tmp_path):
    """v6.109.29: events.jsonl rotates on the supervisor tick (server.py). The
    generalized archive-aware reader (api_logs_tail -> read_rotated_jsonl_entries)
    already passes the log `name` as the archive prefix, so /api/logs/events
    picks up archive/events_*.jsonl automatically — no code change needed in
    logs.py beyond the stale comment update."""
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    archive = data / "archive"
    archive.mkdir()
    (archive / "events_20260101T000100.jsonl").write_text(
        "\n".join(
            json.dumps({
                "ts": f"2026-01-01T00:00:{i:02d}Z",
                "type": "llm_call",
                "task_id": "t1",
                "content": f"archived-event-{i}",
            })
            for i in range(2)
        ) + "\n",
        encoding="utf-8",
    )
    (logs / "events.jsonl").write_text(
        json.dumps({
            "ts": "2026-01-01T00:02:00Z",
            "type": "llm_call",
            "task_id": "t1",
            "content": "live-event",
        }) + "\n",
        encoding="utf-8",
    )

    payload = TestClient(_app(data)).get("/api/logs/events?limit=10").json()

    contents = [row["content"] for row in payload["entries"]]
    assert contents == ["archived-event-0", "archived-event-1", "live-event"]


def test_logs_tail_backfills_from_rotated_tools_archive(tmp_path):
    """v6.109.29: tools.jsonl rotates alongside events.jsonl; same archive-aware
    reader contract — /api/logs/tools merges archive/tools_*.jsonl with the live
    file in chronological order."""
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    archive = data / "archive"
    archive.mkdir()
    (archive / "tools_20260101T000100.jsonl").write_text(
        json.dumps({
            "ts": "2026-01-01T00:00:00Z",
            "type": "tool_call",
            "task_id": "t1",
            "tool": "read_file",
            "content": "archived-tool",
        }) + "\n",
        encoding="utf-8",
    )
    (logs / "tools.jsonl").write_text(
        json.dumps({
            "ts": "2026-01-01T00:02:00Z",
            "type": "tool_call",
            "task_id": "t1",
            "tool": "write_file",
            "content": "live-tool",
        }) + "\n",
        encoding="utf-8",
    )

    payload = TestClient(_app(data)).get("/api/logs/tools?limit=10").json()

    contents = [row["content"] for row in payload["entries"]]
    assert contents == ["archived-tool", "live-tool"]


def test_logs_tail_skips_archives_when_live_satisfies_limit(tmp_path):
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    archive = data / "archive"
    archive.mkdir()
    (archive / "progress_20260101T000100.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "content": "archived", "task_id": "t1"}) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        "\n".join(
            json.dumps({"ts": f"2026-01-01T00:02:{i:02d}Z", "content": f"live-{i}", "task_id": "t1"})
            for i in range(5)
        ) + "\n",
        encoding="utf-8",
    )

    payload = TestClient(_app(data)).get("/api/logs/progress?limit=3").json()

    contents = [row["content"] for row in payload["entries"]]
    assert contents == ["live-2", "live-3", "live-4"]  # newest live tail, no archive read


def test_logs_tail_task_filter_counts_toward_backfill_quota(tmp_path):
    """The archive is consulted when the LIVE file lacks enough rows MATCHING the
    task filter, even if it holds plenty of unrelated rows."""
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    archive = data / "archive"
    archive.mkdir()
    (archive / "progress_20260101T000100.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "content": "wanted-archived", "task_id": "target"}) + "\n",
        encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text(
        "\n".join(
            json.dumps({"ts": f"2026-01-01T00:02:{i:02d}Z", "content": f"noise-{i}", "task_id": "other"})
            for i in range(5)
        ) + "\n",
        encoding="utf-8",
    )
    (data / "task_results").mkdir()
    (data / "task_results" / "target.json").write_text(
        json.dumps({"task_id": "target", "status": "completed", "result": "done"}),
        encoding="utf-8",
    )
    (data / "state").mkdir()
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")

    payload = TestClient(_app(data)).get("/api/logs/progress?task_id=target&limit=2").json()

    contents = [row["content"] for row in payload["entries"]]
    assert contents == ["wanted-archived"]


def test_logs_follow_dedupe_marker_stable_across_polls_on_large_log(tmp_path):
    """Regression (P2 review, GPT live probe): `_line` is window-relative since the
    bounded tail read, so on a >512KB log the window shifts with every append and a
    positional dedupe marker re-prints (or skips) already-seen rows on each poll.
    The CLI `logs follow` marker must be content identity: across two polls with an
    append in between, exactly the appended row is new."""
    from ouroboros.cli import _log_row_identity

    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    pad = "x" * 2000
    with (logs / "progress.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(400):  # ~800KB, past the 512KB first window
            handle.write(json.dumps({
                "ts": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
                "content": f"row-{i}",
                "pad": pad,
                "task_id": "t1",
            }) + "\n")

    client = TestClient(_app(data))
    first = client.get("/api/logs/progress?limit=100").json()["entries"]
    assert len(first) == 100
    seen = {_log_row_identity(entry) for entry in first}

    with (logs / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "ts": "2026-01-01T00:09:00Z",
            "content": "row-400",
            "pad": pad,
            "task_id": "t1",
        }) + "\n")

    second = client.get("/api/logs/progress?limit=100").json()["entries"]
    fresh = [entry for entry in second if _log_row_identity(entry) not in seen]
    assert [entry["content"] for entry in fresh] == ["row-400"]


def test_logs_tail_task_discovery_does_zero_artifact_work(tmp_path, monkeypatch):
    """The task_id root/children discovery is a False-projection read: no artifact
    collection or copy may run on this endpoint (materialize_artifacts contract)."""
    import ouroboros.artifacts as artifacts_mod

    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    (logs / "progress.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "content": "row", "task_id": "target"}) + "\n",
        encoding="utf-8",
    )
    (data / "task_results").mkdir()
    (data / "task_results" / "target.json").write_text(
        json.dumps({"task_id": "target", "status": "completed", "result": "done"}),
        encoding="utf-8",
    )
    (data / "state").mkdir()
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(artifacts_mod, "collect_task_artifact_records", lambda *a, **k: calls.append("collect") or [])
    monkeypatch.setattr(artifacts_mod, "copy_file_to_task_artifacts", lambda *a, **k: calls.append("copy") or {})

    payload = TestClient(_app(data)).get("/api/logs/progress?task_id=target&limit=10").json()

    assert [row["content"] for row in payload["entries"]] == ["row"]
    assert calls == []
