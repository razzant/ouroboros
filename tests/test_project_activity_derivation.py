"""Room activity and unread are dialogue facts; no second archive scan or registry rewrite."""
from __future__ import annotations

import pytest

from ouroboros import dialogue_import as di, dialogue_store as ds, maintenance
from ouroboros import projects_registry as pr
from tests._dialogue import import_all, seed


def _summary(root, project):
    return pr.projects_summary(root, chat_id=project["chat_id"])[0]


@pytest.mark.parametrize("stream", ["chat", "progress"])
def test_imported_activity_uses_room_facts_and_never_rewrites_the_registry(tmp_path, stream):
    active = pr.create_project(tmp_path, "legacy", origin="reconcile")
    quiet = pr.create_project(tmp_path, "quiet", origin="reconcile")
    seed(tmp_path, archives={f"{stream}_20260101T000000.jsonl": [{"chat_id": active["chat_id"], "text": "old words"}]})
    before = (tmp_path / "state/projects.json").read_bytes()
    with pytest.raises(ds.StoreMigrating, match="still importing"):
        _summary(tmp_path, active)
    import_all(tmp_path)
    assert _summary(tmp_path, active)["has_thread_activity"] is True
    assert _summary(tmp_path, quiet)["has_thread_activity"] is False
    assert (tmp_path / "state/projects.json").read_bytes() == before
    assert "activity_backfill" not in maintenance.JOBS


def test_new_progress_is_activity_but_not_unread_and_read_marker_is_not_activity(tmp_path):
    active = pr.create_project(tmp_path, "telemetry", origin="reconcile")
    quiet = pr.create_project(tmp_path, "quiet", origin="reconcile")
    ds.mark_room_read(tmp_path, quiet["chat_id"], 0)
    assert not _summary(tmp_path, quiet)["has_thread_activity"]
    ds.append(tmp_path, "progress", {"chat_id": active["chat_id"], "text": "working"})
    assert _summary(tmp_path, active)["has_thread_activity"]
    assert _summary(tmp_path, active)["visible_revision"] == 0


def test_owner_created_and_bound_rooms_stay_visible_without_dialogue(tmp_path):
    fresh = pr.create_project(tmp_path, "fresh", origin="owner_ui")
    bound = pr.create_project(tmp_path, "bound", origin="reconcile")
    assert _summary(tmp_path, fresh)["has_thread_activity"]
    assert not _summary(tmp_path, bound)["has_thread_activity"]
    pr.bind_task_to_project(tmp_path, "task", "bound", bound["chat_id"], origin={"absent": "system"})
    assert _summary(tmp_path, bound)["has_thread_activity"]


def test_revision_and_order_follow_sql_commits_without_touching_registry(tmp_path):
    older = pr.create_project(tmp_path, "older", origin="reconcile")
    newer = pr.create_project(tmp_path, "newer", origin="reconcile")
    pr.update_project(tmp_path, "older", last_active_at="2000-01-01T00:00:00Z")
    pr.update_project(tmp_path, "newer", last_active_at="2000-01-02T00:00:00Z")
    before = (tmp_path / "state/projects.json").read_bytes()
    ds.append(tmp_path, "chat", {"chat_id": older["chat_id"], "direction": "out", "text": "new answer",
                                 "ts": "2026-10-17T00:00:00Z"}, feed=True)
    assert pr.projects_summary(tmp_path)[0]["id"] == "older"
    assert _summary(tmp_path, older)["visible_revision"] == 1
    assert pr.get_project(tmp_path, "older")["visible_revision"] == 1
    assert _summary(tmp_path, newer)["visible_revision"] == 0
    assert (tmp_path / "state/projects.json").read_bytes() == before


@pytest.mark.parametrize("damage", ["invalid", "mismatch"])
def test_unknown_history_never_proves_room_inactive(tmp_path, damage):
    project = pr.create_project(tmp_path, "unknown", origin="reconcile")
    name = "chat_20260101T000000.jsonl"
    seed(tmp_path, archives={name: ["{broken\n"]})
    ds.ensure(tmp_path)
    if damage == "mismatch":
        (tmp_path / "archive" / name).write_text("changed after cutover\n", encoding="utf-8")
    di.run_dialogue_import(tmp_path)
    with pytest.raises(ds.StoreError, match="gaps|could not be imported"):
        _summary(tmp_path, project)
    # Positive current evidence remains readable even when older coverage is unknown.
    ds.append(tmp_path, "chat", {"chat_id": project["chat_id"], "direction": "in", "text": "known words"})
    assert _summary(tmp_path, project)["has_thread_activity"]
