"""Appearance certification waits for authority and never skips older-body rows.

Regression witnesses from the independent membership review. Only disposable stores are used;
PERSIST contention is real, rather than inferred from a mocked result or exception message.
"""
from __future__ import annotations

import pytest

from ouroboros import dialogue_store as ds, room_appearances as ra
from ouroboros.chat_chain import room_rows
from ouroboros.projects_registry import RoomResolver, create_project
from ouroboros.task_results import write_task_result


def _world(root):
    chat = create_project(root, "nova", name="Nova")["chat_id"]
    write_task_result(root, "nova-root", "running", chat_id=chat, title="Nova root")
    ds.append(root, "chat", {"chat_id": 1, "direction": "in", "ts": "seed", "text": "prime dialogue store"})
    return chat


def _notice():
    return {"ts": "2026-10-27T01:00:00Z", "direction": "out", "chat_id": 1,
            "type": "main_notice", "task_id": "nova-root", "text": "Notice before restored archive"}


def test_real_task_store_contention_preserves_accepted_notice_for_catchup(tmp_path, monkeypatch):
    import sqlite3
    from ouroboros import projects_registry as pr, store
    from supervisor.message_ingress import log_chat

    monkeypatch.setattr(store, "_journal_mode", lambda *_args: "PERSIST")
    chat = _world(tmp_path)
    pr._TASK_LINEAGE.pop((str(tmp_path), "nova-root"), None)
    pr._ROOM_ANSWERS.clear()
    before = ra.mark(tmp_path)
    blocker = sqlite3.connect(store.store_path(tmp_path), isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        with store.acquisition_budget(0.05):
            stored = log_chat("out", 1, 0, "Notice during real task-store contention", record_type="main_notice",
                              task_id="nova-root", drive_root=tmp_path, require_write=True, feed=True)
    finally:
        blocker.rollback()
        blocker.close()
    assert stored, "accepting speech must not depend on deriving its secondary appearance"
    assert ra.mark(tmp_path) == before and ra.owed(tmp_path)
    assert RoomResolver(tmp_path)("nova-root") == chat
    assert ra.run(tmp_path, lambda: False)["state"] == "done"
    assert stored["text"] in [entry.row.get("text") for entry in room_rows(tmp_path, chat)]
    assert not ra.owed(tmp_path)


def test_unavailable_project_registry_leaves_derivation_pending(tmp_path, monkeypatch):
    from ouroboros import projects_registry as pr
    from supervisor.message_ingress import log_chat

    chat = _world(tmp_path)
    before = ra.mark(tmp_path)
    # The production reader distinguishes an unavailable registry from a known empty one.
    # Its strict mode is what lets appearance certification retain that distinction.
    original = pr.reserved_project_chat_ids

    def unavailable(root, *, strict=False):
        if strict:
            raise RuntimeError("registry temporarily unavailable")
        return set()

    with monkeypatch.context() as blocked:
        blocked.setattr(pr, "reserved_project_chat_ids", unavailable)
        stored = log_chat("out", 1, 0, "Notice during registry unavailability", record_type="main_notice",
                          task_id="nova-root", drive_root=tmp_path, require_write=True, feed=True)
        assert stored and ra.mark(tmp_path) == before and ra.owed(tmp_path)
        with pytest.raises(RuntimeError):
            ra.run(tmp_path, lambda: False)
        assert ra.mark(tmp_path) == before
    assert original(tmp_path) == {chat}
    assert ra.run(tmp_path, lambda: False)["state"] == "done"
    assert stored["text"] in [entry.row.get("text") for entry in room_rows(tmp_path, chat)]


def test_restored_archive_import_keeps_preceding_unindexed_live_rows_owed(tmp_path):
    from tests._dialogue import seed, import_all

    chat = _world(tmp_path)
    before = ra.mark(tmp_path)
    row = _notice()
    # An older writer appends a row without advancing the appearance mark.
    stored = ds.append(tmp_path, "chat", row, feed=True, appearances=None)
    assert ra.mark(tmp_path) == before and ra.owed(tmp_path)
    seed(tmp_path, archives={"chat_20261026T000000.jsonl": [
        {"ts": "2026-10-26T01:00:00Z", "direction": "in", "chat_id": 1, "text": "restored older conversation"}]})
    import_all(tmp_path)
    assert ra.mark(tmp_path) == before and ra.owed(tmp_path)
    assert ra.run(tmp_path, lambda: False)["state"] == "done"
    assert row["text"] in [entry.row.get("text") for entry in room_rows(tmp_path, chat)]
    assert ds.row_at(tmp_path, "chat", stored.seq).row == stored.row
    assert not ra.owed(tmp_path)
