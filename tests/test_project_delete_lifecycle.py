"""Project delete orchestration: fence, cancel, quiesce, tombstone, resume."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest


def _request(drive_root, project_id: str = ""):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=drive_root)),
        path_params={"project_id": project_id},
    )


def _wait_for_lifecycle(drive_root, project_id: str, expected: str) -> dict:
    from ouroboros.projects_registry import get_reserved_project

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        row = get_reserved_project(drive_root, project_id) or {}
        if row.get("lifecycle") == expected:
            return row
        time.sleep(0.01)
    pytest.fail(f"Project {project_id} did not reach {expected}")


@pytest.fixture
def isolated_project_queue(tmp_path, monkeypatch):
    import ouroboros.gateway.projects as gateway
    import supervisor.queue as queue
    import supervisor.task_lifecycle as lifecycle
    import supervisor.workers as workers

    pending: list[dict] = []
    running: dict[str, dict] = {}
    broadcasts: list[tuple[str, object]] = []
    cancelled: list[str] = []

    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", running)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", running)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(
        lifecycle,
        "_broadcast_projects_changed",
        lambda project_id, chat_id: broadcasts.append((project_id, chat_id)),
    )
    monkeypatch.setattr(
        gateway,
        "_broadcast_projects_changed",
        lambda project_id, chat_id: broadcasts.append((project_id, chat_id)),
    )

    def cancel_task(task_id: str, *, cascade: bool = False) -> bool:
        assert cascade is True
        cancelled.append(task_id)
        for index, task in enumerate(list(pending)):
            if str(task.get("id") or "") == task_id:
                pending.pop(index)
                return True
        if task_id in running:
            running.pop(task_id)
            return True
        return False

    monkeypatch.setattr(queue, "cancel_task_by_id", cancel_task)
    with lifecycle._PROJECT_DELETE_WORKERS_LOCK:
        lifecycle._PROJECT_DELETE_WORKERS.clear()
    yield SimpleNamespace(
        pending=pending,
        running=running,
        broadcasts=broadcasts,
        cancelled=cancelled,
        queue=queue,
        lifecycle=lifecycle,
    )
    deadline = time.monotonic() + 3
    while lifecycle._PROJECT_DELETE_WORKERS and time.monotonic() < deadline:
        time.sleep(0.01)
    with lifecycle._PROJECT_DELETE_WORKERS_LOCK:
        lifecycle._PROJECT_DELETE_WORKERS.clear()


def test_delete_cancels_bound_root_and_descendants_then_preserves_tombstone(
    tmp_path, isolated_project_queue
):
    from ouroboros.gateway.projects import api_project_delete
    from ouroboros.projects_registry import (
        bind_task_to_project,
        create_project,
        get_reserved_project,
        project_binding_for_task,
        reconcile_projects,
        update_project,
    )

    project = create_project(tmp_path, "alpha", name="Alpha")
    folder = tmp_path / "owner-folder"
    folder.mkdir()
    memory = tmp_path / "projects" / "alpha"
    memory.mkdir(parents=True)
    update_project(tmp_path, "alpha", working_dir=str(folder))
    bind_task_to_project(tmp_path, "root-bound", "alpha", origin={"absent": "system"})

    isolated_project_queue.pending.extend([
        {"id": "root-bound", "root_task_id": "root-bound"},
        {
            "id": "child-pending",
            "parent_task_id": "root-bound",
            "root_task_id": "root-bound",
        },
        {"id": "root-stored", "root_task_id": "root-stored", "project_id": "alpha"},
        {"id": "unrelated", "root_task_id": "unrelated", "project_id": "beta"},
    ])
    isolated_project_queue.running["grandchild-running"] = {
        "task": {
            "id": "grandchild-running",
            "parent_task_id": "child-pending",
            "root_task_id": "root-bound",
        }
    }

    # Route ids are compatibility-sanitized; cancellation must use the
    # canonical registry id rather than missing this task on a case variant.
    response = asyncio.run(api_project_delete(_request(tmp_path, "ALPHA")))
    assert response.status_code == 200
    response_body = json.loads(response.body)
    assert response_body["ok"] is True
    assert response_body["project_id"] == "alpha"
    tombstone = _wait_for_lifecycle(tmp_path, "alpha", "tombstoned")

    assert set(isolated_project_queue.cancelled) == {
        "root-bound",
        "child-pending",
        "grandchild-running",
        "root-stored",
    }
    assert isolated_project_queue.cancelled.index("child-pending") < isolated_project_queue.cancelled.index("root-bound")
    assert [task["id"] for task in isolated_project_queue.pending] == ["unrelated"]
    assert isolated_project_queue.running == {}
    assert folder.is_dir() and memory.is_dir()
    assert project_binding_for_task(tmp_path, "root-bound")["project_id"] == "alpha"
    assert tombstone["chat_id"] == project["chat_id"]
    # The worker persists the tombstone before broadcasting that new state.
    # Observing the durable lifecycle can therefore win this intentional,
    # tiny scheduling window; wait for the asynchronous notification itself.
    broadcast_deadline = time.monotonic() + 3
    while len(isolated_project_queue.broadcasts) < 2 and time.monotonic() < broadcast_deadline:
        time.sleep(0.01)
    assert len(isolated_project_queue.broadcasts) >= 2  # deleting, then tombstoned

    # Boot reconcile sees the preserved memory store but the reserved id prevents
    # resurrection; the immutable binding remains available for history routing.
    assert reconcile_projects(tmp_path) == 0
    assert get_reserved_project(tmp_path, "alpha")["lifecycle"] == "tombstoned"


def test_delete_failure_stays_fenced_with_visible_error(tmp_path, isolated_project_queue, monkeypatch):
    from ouroboros.projects_registry import begin_project_deletion, create_project, get_reserved_project

    project = create_project(tmp_path, "stuck", name="Stuck")
    isolated_project_queue.pending.append({"id": "stuck-root", "project_id": "stuck"})
    monkeypatch.setattr(
        isolated_project_queue.queue,
        "cancel_task_by_id",
        lambda _task_id, **_kwargs: False,
    )

    begin_project_deletion(tmp_path, "stuck")
    isolated_project_queue.lifecycle.run_project_deletion(
        tmp_path, "stuck", project["chat_id"]
    )

    row = get_reserved_project(tmp_path, "stuck")
    assert row["lifecycle"] == "deleting"
    assert "did not quiesce" in row["delete_error"]
    assert isolated_project_queue.pending == [{"id": "stuck-root", "project_id": "stuck"}]
    assert isolated_project_queue.broadcasts[-1] == ("stuck", project["chat_id"])


def test_supervisor_retries_deleting_project_with_prior_error(
    tmp_path, isolated_project_queue, monkeypatch,
):
    from ouroboros.projects_registry import begin_project_deletion, create_project, get_reserved_project

    project = create_project(tmp_path, "retry", name="Retry")
    isolated_project_queue.pending.append({"id": "retry-root", "project_id": "retry"})
    working_cancel = isolated_project_queue.queue.cancel_task_by_id
    monkeypatch.setattr(
        isolated_project_queue.queue,
        "cancel_task_by_id",
        lambda _task_id, **_kwargs: False,
    )
    begin_project_deletion(tmp_path, "retry")
    isolated_project_queue.lifecycle.run_project_deletion(
        tmp_path, "retry", project["chat_id"],
    )
    assert get_reserved_project(tmp_path, "retry")["delete_error"]

    monkeypatch.setattr(isolated_project_queue.queue, "cancel_task_by_id", working_cancel)
    assert isolated_project_queue.lifecycle.resume_project_deletions(tmp_path) == 1
    _wait_for_lifecycle(tmp_path, "retry", "tombstoned")
    assert isolated_project_queue.cancelled == ["retry-root"]


def test_supervisor_resumes_deleting_project_after_restart(tmp_path, isolated_project_queue):
    from ouroboros.projects_registry import begin_project_deletion, create_project, reconcile_projects

    create_project(tmp_path, "resume", name="Resume")
    (tmp_path / "projects" / "resume").mkdir(parents=True)
    isolated_project_queue.pending.append({"id": "resume-root", "project_id": "resume"})
    begin_project_deletion(tmp_path, "resume")

    # A fresh process has no in-memory marker. Supervisor startup, not a UI GET,
    # resumes the durable deleting row for headless parity.
    with isolated_project_queue.lifecycle._PROJECT_DELETE_WORKERS_LOCK:
        isolated_project_queue.lifecycle._PROJECT_DELETE_WORKERS.clear()
    assert isolated_project_queue.lifecycle.resume_project_deletions(tmp_path) == 1
    _wait_for_lifecycle(tmp_path, "resume", "tombstoned")
    assert isolated_project_queue.cancelled == ["resume-root"]
    assert reconcile_projects(tmp_path) == 0


# --------------------------------------------------------------------------- #
# P1 — the sweep must not acknowledge work the owner never saw
# --------------------------------------------------------------------------- #

def _git(cwd, *args, check=True):
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check
    )


@pytest.fixture
def owner_repo(tmp_path):
    root = tmp_path / "owner_folder"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "app.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "seed")
    return root


def test_the_sweep_reinspects_and_keeps_a_newly_at_risk_checkout(
    tmp_path, owner_repo, monkeypatch
):
    """P1: `remove_project_thread_worktrees` hardcoded `acknowledge_unmerged=True`.

    Reproduced end to end on the merged tree: the checkout is CLEAN when
    `api_project_delete` inspects it, so nothing refuses; the route's own removal
    correctly answers `project_busy` because a task is still writing there; the
    still-running task then commits work that exists nowhere else and edits a
    tracked file; the post-quiescence sweep destroyed both with no re-inspection
    and no fresh consent.

    The sweep now re-asks the SAME judge the route asked, so a checkout that became
    at-risk after the owner looked comes back `unmerged_work` and is KEPT.
    """
    import pathlib

    from ouroboros.project_threads_registry import create_thread
    from ouroboros.projects_registry import create_project
    from ouroboros.thread_branching import branch_off_thread
    from ouroboros.thread_worktrees import (
        checkout_work_at_risk,
        get_thread_worktree,
        inspect_thread_worktree,
    )
    from supervisor.task_lifecycle import _sweep_project_checkouts

    monkeypatch.setenv("OUROBOROS_THREAD_WORKTREE_ROOT", str(tmp_path / "wts"))
    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer", working_dir=str(owner_repo))
    thread = create_thread(drive, "racer", name="Side quest")
    out = branch_off_thread(
        drive, "racer", thread["id"], data_dir=drive, worktree_root=tmp_path / "wts"
    )
    assert out["ok"] is True
    checkout = pathlib.Path(out["path"])

    # What the still-running task produced after the owner's pre-fence look.
    (checkout / "only_copy.txt").write_text("exists nowhere else\n", encoding="utf-8")
    _git(checkout, "add", "-A")
    _git(
        checkout, "-c", "user.email=t@example.com", "-c", "user.name=T",
        "commit", "-m", "add only_copy.txt",
    )
    (checkout / "app.txt").write_text("owner edit, uncommitted\n", encoding="utf-8")

    risk = checkout_work_at_risk(
        inspect_thread_worktree(get_thread_worktree(drive, "racer", thread["id"]))
    )
    assert risk["at_risk"] is True
    assert risk["unmerged_commits"] == 1
    assert risk["tracked_files"]

    note = _sweep_project_checkouts(drive, "racer")

    assert checkout.is_dir(), "the sweep destroyed work the owner never acknowledged"
    assert (checkout / "only_copy.txt").is_file()
    assert (checkout / "app.txt").read_text(encoding="utf-8") == "owner edit, uncommitted\n"
    assert get_thread_worktree(drive, "racer", thread["id"]) is not None
    # ...and it did not happen silently: the survivor is named, WITH its folder.
    assert "unmerged_work" in note
    assert str(checkout) in note
    assert out["branch"] in note


def test_rebuildable_dirt_is_still_swept_without_a_second_prompt(
    tmp_path, owner_repo, monkeypatch
):
    """The re-inspection must not turn the sweep into a wall.

    `checkout_work_at_risk` is deliberately narrower than the removal's own "would
    be destroyed": an ignored build directory is not work at risk, the owner
    already confirmed the deletion, and refusing over it would resurrect the
    three-step detour H-ter closed.
    """
    import pathlib

    from ouroboros.project_threads_registry import create_thread
    from ouroboros.projects_registry import create_project
    from ouroboros.thread_branching import branch_off_thread
    from ouroboros.thread_worktrees import get_thread_worktree
    from supervisor.task_lifecycle import _sweep_project_checkouts

    monkeypatch.setenv("OUROBOROS_THREAD_WORKTREE_ROOT", str(tmp_path / "wts"))
    drive = tmp_path / "drive"
    (owner_repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    _git(owner_repo, "add", "-A")
    _git(owner_repo, "commit", "-m", "ignore logs")
    create_project(drive, "racer", name="Racer", working_dir=str(owner_repo))
    thread = create_thread(drive, "racer", name="Side quest")
    out = branch_off_thread(
        drive, "racer", thread["id"], data_dir=drive, worktree_root=tmp_path / "wts"
    )
    checkout = pathlib.Path(out["path"])
    (checkout / "build.log").write_text("noise\n", encoding="utf-8")
    (checkout / "never_added.txt").write_text("untracked\n", encoding="utf-8")

    note = _sweep_project_checkouts(drive, "racer")

    assert note == ""
    assert not checkout.exists()
    assert get_thread_worktree(drive, "racer", thread["id"]) is None


def test_a_checkout_that_survives_is_disclosed_on_the_tombstone_and_to_the_owner(
    tmp_path, owner_repo, monkeypatch
):
    """P1, second half: a `removal_failed` used to end with `lifecycle` tombstoned,
    the checkout on disk, the registry row present and `delete_error == ""` — the
    sweep's `log.warning` reaching no surface at all.

    Keeping the project UNTOMBSTONED was rejected as owner direction (§I M2), so
    the answer is disclosure: the survivors ride the tombstoned row and the owner
    is told where they are.
    """
    import pathlib

    from ouroboros.project_threads_registry import create_thread
    from ouroboros.projects_registry import (
        begin_project_deletion,
        create_project,
        get_reserved_project,
    )
    from ouroboros.thread_branching import branch_off_thread
    import ouroboros.thread_worktrees as twt
    import supervisor.task_lifecycle as lifecycle

    monkeypatch.setenv("OUROBOROS_THREAD_WORKTREE_ROOT", str(tmp_path / "wts"))
    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer", working_dir=str(owner_repo))
    thread = create_thread(drive, "racer", name="Side quest")
    out = branch_off_thread(
        drive, "racer", thread["id"], data_dir=drive, worktree_root=tmp_path / "wts"
    )
    checkout = pathlib.Path(out["path"])
    assert checkout.is_dir()

    # Removal cannot take it — a git lock, a read-only parent, a busy file.
    monkeypatch.setattr(
        twt, "remove_thread_worktree",
        lambda **kw: {"removed": False, "reason": "removal_failed", "inspection": {}},
    )
    monkeypatch.setattr(lifecycle, "_live_project_task_ids", lambda *a, **k: [])
    monkeypatch.setattr(lifecycle, "_broadcast_projects_changed", lambda *a, **k: None)
    told: list[tuple[str, str]] = []
    monkeypatch.setattr(
        lifecycle, "_tell_owner_about_orphaned_checkouts",
        lambda pid, note: told.append((pid, note)),
    )

    begin_project_deletion(drive, "racer")
    lifecycle.run_project_deletion(drive, "racer", 1234)

    entry = get_reserved_project(drive, "racer")
    # The tombstone still happens — that is the owner's decision, not a defect.
    assert entry["lifecycle"] == "tombstoned"
    assert checkout.is_dir()
    # ...but the owner learns WHAT was left behind and WHERE.
    assert "removal_failed" in entry["delete_error"]
    assert str(checkout) in entry["delete_error"]
    assert out["branch"] in entry["delete_error"]
    assert told and told[0][0] == "racer"
    assert str(checkout) in told[0][1]


def test_a_disclosure_too_long_to_fit_says_what_it_left_out(tmp_path):
    """P1: the one text that may NOT lose part of itself in silence.

    `_orphaned_checkouts_note` names each surviving folder and branch because a
    tombstoned project is on no surface that could show them — and then cut the
    joined lines with a flat `[:2000]`. Reproduced at 30 survivors: 13 checkouts
    disappeared from the only record naming them and the sentence ended mid-word,
    with nothing saying anything had been dropped. A cap over an owner-facing
    disclosure about lost work is a bound, not a licence to lose the tail.
    """
    from supervisor.task_lifecycle import _orphaned_checkouts_note

    kept = [
        {"thread_id": i,
         "path": f"/Users/owner/Ouroboros/thread_worktrees/bigproject__{i}",
         "branch": f"thread/feature-{i}", "reason": "unmerged_work"}
        for i in range(1, 31)
    ]
    note = _orphaned_checkouts_note(kept)

    named = [row for row in kept if f"thread {row['thread_id']}: {row['path']}" in note]
    dropped = [row for row in kept if row not in named]
    assert dropped, "this reproduction needs the budget to actually bite"
    # Whatever is named is named WHOLE — the bound falls on an entry boundary, so
    # no checkout is half-identified by a path cut in the middle.
    for row in named:
        assert f"{row['path']} (branch {row['branch']}) — unmerged_work" in note
    # ...and everything else is DECLARED: how many, which threads, and where the
    # unabridged list is.
    assert f"{len(dropped)} more are still on disk and NOT named above" in note
    for row in dropped:
        assert str(row["thread_id"]) in note
    assert "in the app log" in note
    # The total is still stated up front, so the count never shrinks to what fit.
    assert note.startswith(f"{len(kept)} thread checkouts could not be removed")


def test_a_bounded_delete_error_reaches_the_row_with_its_omission_marker(tmp_path):
    """The registry's own backstop bounds `delete_error`; it must not cut silently
    either, and it must be roomy enough not to re-cut a note that already declared
    its own omissions."""
    from ouroboros.projects_registry import (
        begin_project_deletion,
        complete_project_deletion,
        create_project,
    )

    drive = tmp_path / "drive"
    folder = tmp_path / "folder"
    folder.mkdir()
    create_project(drive, "verbose", name="Verbose", working_dir=str(folder))
    begin_project_deletion(drive, "verbose")
    original = "x" * 12000
    entry = complete_project_deletion(drive, "verbose", delete_error=original)

    stored = entry["delete_error"]
    assert len(stored) < len(original), "the field is still bounded"
    assert "OMISSION NOTE" in stored, "...but the bound announces itself"
    assert str(len(original)) in stored, "the original length is part of the disclosure"
