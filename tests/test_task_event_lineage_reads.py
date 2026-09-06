"""Failed result reads disclose uncertainty without inventing or losing lineage."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from ouroboros.gateway.task_events import _TaskEventCursorFollower, _TaskEventFollower
from ouroboros.task_results import write_task_result
from tests.test_task_event_cursor import append, content, seed


def child_follower(tmp_path):
    root = seed(tmp_path / "root")
    child = seed(tmp_path / "child", "child")
    write_task_result(root, "child", "running", delegation_role="subagent",
                      parent_task_id="root", child_drive_root=str(child))
    append(child, "first", task_id="child")
    follower = _TaskEventCursorFollower(root, "root", {"v": 2, "seq": 0, "view": "", "positions": {}})
    follower.refresh_view()
    assert content(list(follower.read_events())) == ["first"]
    return root, child, follower, root / "task_results" / "child.json"


@pytest.mark.parametrize("fault", ["malformed", "permission"])
def test_living_follower_retains_only_its_prior_proof_and_reports_failed_read(tmp_path, fault):
    root, child, follower, path = child_follower(tmp_path)
    original, stat = path.read_bytes(), path.stat()
    checkpoint = follower.checkpoint()
    if fault == "malformed":
        path.write_bytes(b'{"unfinished":')
    else:
        if os.name == "nt":
            pytest.skip("POSIX mode permissions; malformed case is platform-independent")
        path.chmod(0)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    try:
        if fault == "permission":
            try:
                path.read_bytes()
            except PermissionError:
                pass
            else:
                pytest.skip("test user can bypass file read permission")
        append(child, "pending", task_id="child")
        append(root, "root-continues")
        assert follower.refresh_view() is None
        assert child in follower.roots and follower.view == checkpoint["view"]
        rows = list(follower.read_events())
        gap = rows[0]
        assert gap["type"] == "history_gap" and gap["reason"] == "lineage_incomplete"
        assert gap["data"] == {"failed_result_reads": 1, "retained_result_bindings": 1,
                               "unknown_result_membership": 0}
        assert gap["cursor"] == checkpoint  # diagnostic acknowledges no log row
        assert set(content(rows)) == {"pending", "root-continues"}
        follower.refresh_view()
        assert list(follower.read_events()) == []  # no warning storm per poll
        fresh = _TaskEventCursorFollower(root, "root", checkpoint)
        assert fresh.refresh_view()["type"] == "cursor_replay"
        assert child not in fresh.roots  # cursor paths cannot supply proof
        fresh_rows = list(fresh.read_events())
        assert fresh_rows[0]["data"]["unknown_result_membership"] == 1
        assert "pending" not in content(fresh_rows)
    finally:
        path.chmod(stat.st_mode & 0o777)
        path.write_bytes(original)
    follower.refresh_view()
    assert child in follower.roots and list(follower.read_events()) == []


@pytest.mark.parametrize("change", ["parent", "role", "schema", "delete"])
def test_confirmed_membership_change_still_retires_the_old_proof(tmp_path, change):
    _root, child, follower, path = child_follower(tmp_path)
    row = json.loads(path.read_text())
    if change == "delete":
        path.unlink()
    else:
        if change == "parent":
            row.update(parent_task_id="elsewhere", root_task_id="elsewhere")
        elif change == "role":
            row["delegation_role"] = "root"
        else:
            row["_schema_version"] = 999
        path.write_text(json.dumps(row), encoding="utf-8")
    assert follower.refresh_view()["type"] == "cursor_replay"
    assert child not in follower.roots and "child" not in follower.task_filter_ids
    assert not follower._seen_result_names
    assert all(row["type"] != "history_gap" for row in follower.read_events())


def test_unrelated_broken_result_discloses_unknown_membership_without_stopping_logs(tmp_path):
    root, child, follower, _path = child_follower(tmp_path)
    (root / "task_results" / "other.json").write_bytes(b'{"unfinished":')
    append(root, "root-continues")
    append(child, "child-continues", task_id="child")
    assert follower.refresh_view() is None
    rows = list(follower.read_events())
    assert rows[0]["data"] == {"failed_result_reads": 1, "retained_result_bindings": 0,
                               "unknown_result_membership": 1}
    assert set(content(rows)) == {"root-continues", "child-continues"}
    assert not any(row["type"] == "error" for row in rows)


def test_legacy_remerge_keeps_live_proof_and_does_not_insert_diagnostics_into_history_rank(tmp_path):
    root, child, _cursor_follower, path = child_follower(tmp_path)
    follower = _TaskEventFollower(root, "root")
    before = follower.full_merge()
    path.write_bytes(b'{"unfinished":')
    follower.poll()
    assert child in follower.roots
    assert follower._lineage_notice["type"] == "history_gap"
    after = follower.full_merge()
    assert [(row["type"], row["seq"]) for row in before] == [(row["type"], row["seq"]) for row in after]
    assert child in follower.roots


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_live_http_stream_warns_without_cli_stopping_error_or_losing_child(tmp_path, method):
    from starlette.requests import Request
    from ouroboros.gateway.task_events import api_task_events

    root, child, _follower, path = child_follower(tmp_path)
    original = path.read_bytes()
    async def receive():
        return {"type": "http.request", "body": b'{"v":2,"wait":1}', "more_body": False}
    request = Request({"type": "http", "method": method, "path": "/api/tasks/root/events",
        "path_params": {"task_id": "root"}, "query_string": b"wait=1", "headers": [],
        "app": SimpleNamespace(state=SimpleNamespace(drive_root=root))}, receive)
    async def consume():
        response = await api_task_events(request)
        events, faulted = [], False
        async for frame in response.body_iterator:
            for line in frame.splitlines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                events.append(event)
                if not faulted and event.get("data", {}).get("content") == "first":
                    faulted = True
                    path.write_bytes(b'{"unfinished":')
                    append(child, "pending", task_id="child", ts="2030-01-01T00:00:00Z")
                elif event.get("data", {}).get("content") == "pending":
                    path.write_bytes(original)
        return events
    events = asyncio.run(consume())
    assert "pending" in content(events)
    gaps = [event for event in events if event["type"] == "history_gap"]
    assert len(gaps) == 1 and gaps[0]["reason"] == "lineage_incomplete"
    assert gaps[0]["data"]["retained_result_bindings"] == 1
    assert not any(event["type"] == "error" for event in events)
