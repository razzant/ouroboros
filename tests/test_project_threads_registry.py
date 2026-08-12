"""Project threads — registry storage, canonical projection, chat-id invariant.

Phase T0 of the project-threads work. Pins the three claims the rest of the
feature stands on:

* thread #0 IS the project's existing chat (no migration, no row rewriting);
* the canonical projection synthesizes thread #0 from the project row, and a
  LEGACY row (no ``threads`` key at all) reads as a healthy one-thread project;
* chat-id reservation is registry-WIDE — a project can never be minted onto a
  chat id an existing project OR thread already owns.
"""

from __future__ import annotations

import json

import pytest

from ouroboros.contracts.chat_id_policy import (
    MAIN_THREAD_ID,
    is_project_chat_id,
    project_chat_id,
    thread_chat_id,
)
from ouroboros.projects_registry import (
    _registry_path,
    create_project,
    create_thread,
    duplicate_chat_ids,
    fork_thread,
    get_thread,
    increment_project_visible_revision,
    project_threads,
    projects_summary,
    rename_thread,
    reserved_project_chat_ids,
    resolve_chat_binding,
)


def test_thread_zero_is_exactly_todays_project_chat_id():
    """B2: the whole zero-migration claim. If this ever drifts, every existing
    history row, task binding and unread counter of every project is orphaned."""
    for pid in ("racer", "proj_abc123", "a-b.c_d"):
        assert thread_chat_id(pid, MAIN_THREAD_ID) == project_chat_id(pid)
        assert thread_chat_id(pid) == project_chat_id(pid)
    # Non-primary threads are distinct, deterministic and still project chats.
    first = thread_chat_id("racer", 1)
    assert first != project_chat_id("racer")
    assert thread_chat_id("racer", 1) == first
    assert thread_chat_id("racer", 2) != first
    assert is_project_chat_id(first)
    # An unusable project id degrades to main, exactly like project_chat_id.
    assert thread_chat_id("", 3) == 1


def test_legacy_row_without_threads_projects_one_thread(tmp_path):
    """A registry written before threads existed must read as a one-thread
    project — and must NOT be rewritten on read."""
    path = _registry_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = {
        "projects": [{
            "id": "legacy",
            "name": "Legacy",
            "chat_id": project_chat_id("legacy"),
            "working_dir": "",
            "origin": "owner",
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_active_at": "2026-01-02T00:00:00+00:00",
            "lifecycle": "active",
            "visible_revision": 7,
        }],
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")

    rows = projects_summary(tmp_path)
    assert len(rows) == 1
    threads = rows[0]["threads"]
    assert len(threads) == 1
    assert threads[0] == {
        "id": MAIN_THREAD_ID,
        "chat_id": project_chat_id("legacy"),
        "name": "Legacy",
        "created_at": "2026-01-01T00:00:00+00:00",
        "visible_revision": 7,
        # T3: thread #0 IS the project, so its lifecycle MIRRORS the project row
        # rather than being a second state that could disagree with it.
        "lifecycle": "active",
        "archived_at": "",
        "delete_error": "",
    }
    # Compatibility alias preserved.
    assert rows[0]["chat_id"] == threads[0]["chat_id"]
    # Untouched on disk: a read must never rewrite the row.
    assert json.loads(path.read_text(encoding="utf-8")) == legacy


def test_stored_thread_zero_is_ignored(tmp_path):
    """Thread #0 is synthesized, never stored: a hand-edited/corrupt row that
    claims id 0 must not become a second, disagreeing truth."""
    path = _registry_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"projects": [{
        "id": "p", "name": "P", "chat_id": project_chat_id("p"), "lifecycle": "active",
        "threads": [
            {"id": 0, "chat_id": 424242, "name": "impostor"},
            {"id": 1, "chat_id": thread_chat_id("p", 1), "name": "real"},
            {"id": 1, "chat_id": 999, "name": "duplicate id"},
            {"id": "bogus", "chat_id": 5},
            "not a dict",
        ],
    }]}), encoding="utf-8")

    threads = project_threads(projects_summary(tmp_path)[0])
    assert [t["id"] for t in threads] == [0, 1]
    assert threads[0]["chat_id"] == project_chat_id("p")
    assert threads[1]["name"] == "real"
    assert 424242 not in reserved_project_chat_ids(tmp_path)


def test_create_fork_and_rename_threads(tmp_path):
    create_project(tmp_path, "racer", name="Cyber Racer")
    thread = create_thread(tmp_path, "racer", name="Tuning")

    assert thread["id"] == 1
    assert thread["chat_id"] == thread_chat_id("racer", 1)
    assert "fork_of_chat_id" not in thread

    fork = fork_thread(tmp_path, "racer", thread["id"])
    # D2: plain English auto-name, no model call.
    assert fork["name"] == "Copy of Tuning"
    # A3: a CURSOR, never a row copy.
    assert fork["fork_of_chat_id"] == thread["chat_id"]
    assert fork["fork_before_ts"]

    assert rename_thread(tmp_path, "racer", thread["id"], "Tuned")["name"] == "Tuned"
    assert get_thread(tmp_path, "racer", thread["id"])["name"] == "Tuned"
    # Renaming thread #0 renames the project itself (it IS the project row).
    assert rename_thread(tmp_path, "racer", MAIN_THREAD_ID, "Racer II")["name"] == "Racer II"
    assert projects_summary(tmp_path)[0]["name"] == "Racer II"

    with pytest.raises(ValueError):
        fork_thread(tmp_path, "racer", 999)


def test_thread_ids_are_never_reused(tmp_path):
    """A durable high-water mark, not `max(live ids) + 1`. The moment threads
    become removable, a reused id would mint a chat id the removed thread's
    history rows still carry — silently merging two conversations."""
    import json

    create_project(tmp_path, "racer")
    first = create_thread(tmp_path, "racer", name="one")
    second = create_thread(tmp_path, "racer", name="two")
    assert (first["id"], second["id"]) == (1, 2)

    # Simulate a future removal: drop the live rows, keep the persisted mark.
    path = _registry_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["projects"][0]["thread_seq"] == 2
    data["projects"][0]["threads"] = []
    path.write_text(json.dumps(data), encoding="utf-8")

    third = create_thread(tmp_path, "racer", name="three")
    assert third["id"] == 3
    assert third["chat_id"] not in {first["chat_id"], second["chat_id"]}


def test_reserved_chat_ids_and_binding_cover_every_thread(tmp_path):
    create_project(tmp_path, "racer", name="Cyber Racer")
    thread = create_thread(tmp_path, "racer", name="Tuning")

    reserved = reserved_project_chat_ids(tmp_path)
    assert {project_chat_id("racer"), thread["chat_id"]} <= reserved

    binding = resolve_chat_binding(tmp_path, thread["chat_id"])
    assert binding["project_id"] == "racer"
    assert binding["thread_id"] == thread["id"]
    assert binding["lifecycle"] == "active"

    zero = resolve_chat_binding(tmp_path, project_chat_id("racer"))
    assert zero["thread_id"] == MAIN_THREAD_ID
    # Main chat / unknown transport ids are NOT project bindings.
    assert resolve_chat_binding(tmp_path, 1) == {}
    assert resolve_chat_binding(tmp_path, 0) == {}
    assert resolve_chat_binding(tmp_path, "junk") == {}


def test_binding_index_refreshes_after_a_write(tmp_path):
    """The index is memoized on the registry file's stat — a thread created
    after a lookup must be visible immediately, not on the next process."""
    create_project(tmp_path, "racer")
    assert resolve_chat_binding(tmp_path, project_chat_id("racer"))["thread_id"] == 0
    thread = create_thread(tmp_path, "racer", name="later")
    assert resolve_chat_binding(tmp_path, thread["chat_id"])["thread_id"] == thread["id"]


def test_project_creation_refuses_a_chat_id_owned_by_a_thread(tmp_path, monkeypatch):
    """X1: the collision class is registry-WIDE. A later PROJECT colliding with
    an existing THREAD must be refused loudly, never silently merged."""
    create_project(tmp_path, "racer")
    thread = create_thread(tmp_path, "racer", name="Tuning")

    import ouroboros.projects_registry as registry

    monkeypatch.setattr(registry, "project_chat_id", lambda pid: thread["chat_id"])
    with pytest.raises(ValueError, match="already reserved"):
        create_project(tmp_path, "colliding")


def test_thread_minting_retries_past_a_reserved_chat_id(tmp_path):
    """X1's retry: a thread id is opaque, so a collision walks to the next id."""
    create_project(tmp_path, "racer")
    # Reserve exactly the chat id thread #1 would take, on another project.
    taken = thread_chat_id("racer", 1)

    # The mint lives in the thread module, so that is where the contract call
    # this test bends is resolved.
    import ouroboros.project_threads_registry as registry

    real = registry.thread_chat_id
    thread = create_thread(tmp_path, "racer", name="first")
    assert thread["chat_id"] == taken

    # Now force thread #2's mint to collide with thread #1 once.
    def _colliding(pid, tid):
        return taken if int(tid) == 2 else real(pid, tid)

    registry.thread_chat_id = _colliding
    try:
        second = create_thread(tmp_path, "racer", name="second")
    finally:
        registry.thread_chat_id = real
    assert second["id"] == 3
    assert second["chat_id"] == real("racer", 3)


def test_reconcile_skips_a_colliding_store_instead_of_merging(tmp_path, monkeypatch, caplog):
    from ouroboros.projects_registry import list_projects, reconcile_projects

    create_project(tmp_path, "kept")
    (tmp_path / "projects" / "legacy-store" / "knowledge").mkdir(parents=True)

    import ouroboros.projects_registry as registry

    monkeypatch.setattr(registry, "project_chat_id", lambda pid: project_chat_id("kept"))
    with caplog.at_level("ERROR"):
        assert reconcile_projects(tmp_path) == 0
    assert "already reserved" in caplog.text
    assert {p["id"] for p in list_projects(tmp_path)} == {"kept"}


def test_duplicate_chat_ids_detected_on_load(tmp_path, caplog):
    path = _registry_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shared = project_chat_id("a")
    path.write_text(json.dumps({"projects": [
        {"id": "a", "name": "A", "chat_id": shared, "lifecycle": "active"},
        {"id": "b", "name": "B", "chat_id": shared, "lifecycle": "active"},
    ]}), encoding="utf-8")

    import ouroboros.project_threads_registry as registry

    registry._DUPLICATE_CHAT_ID_REPORTED.clear()
    with caplog.at_level("ERROR"):
        clashes = duplicate_chat_ids(tmp_path)
    assert clashes == {shared: [("a", 0), ("b", 0)]}
    assert "COLLISION" in caplog.text


def test_visible_revision_advances_thread_and_project_aggregate(tmp_path):
    create_project(tmp_path, "racer")
    thread = create_thread(tmp_path, "racer", name="Tuning")

    increment_project_visible_revision(tmp_path, chat_id=thread["chat_id"])
    row = projects_summary(tmp_path)[0]
    # The project aggregate is what today's FLAT project_seen_revision cursor
    # reads, so thread activity must never be invisible to it.
    assert row["visible_revision"] == 1
    by_id = {t["id"]: t for t in row["threads"]}
    assert by_id[thread["id"]]["visible_revision"] == 1
    # ...but thread #0 has its OWN counter, and a SIBLING's activity is not its
    # activity. Projecting the aggregate here marked the project's main thread
    # unread every time any other thread received a message.
    assert by_id[MAIN_THREAD_ID]["visible_revision"] == 0

    increment_project_visible_revision(tmp_path, chat_id=project_chat_id("racer"))
    row = projects_summary(tmp_path)[0]
    assert row["visible_revision"] == 2
    by_id = {t["id"]: t for t in row["threads"]}
    assert by_id[thread["id"]]["visible_revision"] == 1
    assert by_id[MAIN_THREAD_ID]["visible_revision"] == 1


def test_thread0_revision_seeds_from_a_legacy_aggregate(tmp_path):
    """A registry written before thread #0 got its own counter carries only the
    aggregate. While a project had ONE thread the two were the SAME fact, so the
    projection seeds from it — a legacy project must not read as freshly-unread
    (revision 0) after the upgrade."""
    from ouroboros.utils import atomic_write_json, read_json_dict

    create_project(tmp_path, "legacy")
    for _ in range(3):
        increment_project_visible_revision(tmp_path, project_id="legacy")
    data = read_json_dict(_registry_path(tmp_path))
    for entry in data["projects"]:
        entry.pop("thread0_visible_revision", None)   # pre-split on-disk shape
    atomic_write_json(_registry_path(tmp_path), data)

    row = projects_summary(tmp_path)[0]
    assert row["visible_revision"] == 3
    assert project_threads(row)[0]["visible_revision"] == 3


def test_normalization_preserves_unknown_thread_keys(tmp_path):
    """T3 stores a ``worktree`` binding on a branched-off thread. Normalization
    runs on EVERY read, so rebuilding a fresh dict of known keys would delete
    such a field from an untouched registry — a loss indistinguishable from
    never having written it."""
    from ouroboros.utils import atomic_write_json, read_json_dict

    create_project(tmp_path, "racer")
    thread = create_thread(tmp_path, "racer", name="Branched")
    data = read_json_dict(_registry_path(tmp_path))
    for entry in data["projects"]:
        for row in entry.get("threads") or []:
            if int(row["id"]) == int(thread["id"]):
                row["worktree"] = {"path": "/w/racer-2", "branch": "thread/racer__2"}
                row["future_field"] = "kept"
    atomic_write_json(_registry_path(tmp_path), data)

    stored = get_thread(tmp_path, "racer", thread["id"])
    assert stored["worktree"] == {"path": "/w/racer-2", "branch": "thread/racer__2"}
    assert stored["future_field"] == "kept"

    # A HALF-written fork cursor is still dropped whole: an ancestry walk must
    # never inherit a bound without a parent (or the reverse).
    data = read_json_dict(_registry_path(tmp_path))
    for entry in data["projects"]:
        for row in entry.get("threads") or []:
            row["fork_of_chat_id"] = 12345      # no fork_before_ts
    atomic_write_json(_registry_path(tmp_path), data)
    reread = get_thread(tmp_path, "racer", thread["id"])
    assert "fork_of_chat_id" not in reread and "fork_before_ts" not in reread


def test_threads_cannot_be_added_to_a_fenced_project(tmp_path):
    from ouroboros.projects_registry import begin_project_deletion

    create_project(tmp_path, "racer")
    begin_project_deletion(tmp_path, "racer")
    with pytest.raises(ValueError, match="deleting"):
        create_thread(tmp_path, "racer", name="too late")
