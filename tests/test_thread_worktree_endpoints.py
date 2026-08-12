"""HTTP contract for the thread branching / checkout routes (T3).

    GET  /api/projects/{project_id}/threads/{thread_id}/branch-bases
    POST /api/projects/{project_id}/threads/{thread_id}/branch-off
    POST /api/projects/{project_id}/threads/{thread_id}/merge-back
    GET  /api/projects/{project_id}/threads/{thread_id}/worktree
    POST /api/projects/{project_id}/threads/{thread_id}/worktree/remove
    GET  /api/projects/{project_id}/threads/{thread_id}/diff

Owner surfaces, gateway-only and deliberately NOT LLM-callable tools: these
gestures touch the owner's own folder and history.

Hermetic against a REAL git repository. What is pinned is the transport
contract — statuses, the shared refusal envelope, and the fact that a refusal
carries a typed reason the UI can branch on rather than a stack trace.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.project_threads import (
    api_thread_branch_bases,
    api_thread_branch_off,
    api_thread_diff,
    api_thread_merge_back,
    api_thread_worktree_inspect,
    api_thread_worktree_remove,
)
from ouroboros.project_threads_registry import create_thread
from ouroboros.projects_registry import create_project


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True)


@pytest.fixture(autouse=True)
def worktrees_under_tmp(tmp_path, monkeypatch):
    """Keep every provisioned checkout inside the test's own tmp tree."""
    monkeypatch.setenv("OUROBOROS_THREAD_WORKTREE_ROOT", str(tmp_path / "thread_worktrees"))


@pytest.fixture(autouse=True)
def quiet_broadcast(monkeypatch):
    import ouroboros.gateway.project_threads as gw

    monkeypatch.setattr(gw, "_broadcast_thread_change", lambda *a, **k: None)


@pytest.fixture()
def folder(tmp_path):
    root = tmp_path / "owner_folder"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "app.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "seed")
    return root


def _client(drive_root: pathlib.Path) -> TestClient:
    base = "/api/projects/{project_id}/threads/{thread_id}"
    app = Starlette(routes=[
        Route(f"{base}/branch-bases", api_thread_branch_bases, methods=["GET"]),
        Route(f"{base}/branch-off", api_thread_branch_off, methods=["POST"]),
        Route(f"{base}/merge-back", api_thread_merge_back, methods=["POST"]),
        Route(f"{base}/worktree", api_thread_worktree_inspect, methods=["GET"]),
        Route(f"{base}/worktree/remove", api_thread_worktree_remove, methods=["POST"]),
        Route(f"{base}/diff", api_thread_diff, methods=["GET"]),
    ])
    app.state.drive_root = drive_root
    app.state.repo_dir = drive_root
    return TestClient(app)


@pytest.fixture()
def wired(tmp_path, folder):
    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer", working_dir=str(folder))
    thread = create_thread(drive, "racer", name="Side quest")
    return _client(drive), drive, thread["id"], folder


def test_branch_bases_lists_the_offer(wired):
    client, _drive, tid, folder = wired
    _git(folder, "tag", "v1")

    body = client.get(f"/api/projects/racer/threads/{tid}/branch-bases").json()

    assert body["project_id"] == "racer"
    assert body["thread_id"] == tid
    assert body["current_branch"] == "main"
    assert body["location"]["where"] == "project_folder"
    assert "v1" in [row["ref"] for row in body["bases"]]
    assert body["snapshot"]["ref"] == "@snapshot"


def test_branch_off_then_the_checkout_diff_shows_that_checkouts_work(wired):
    """A13/X9: the per-task diff route structurally cannot answer this, so the
    thread route does — with the same envelope, same statuses, same patch bytes."""
    client, _drive, tid, _folder = wired

    empty = client.get(f"/api/projects/racer/threads/{tid}/diff").json()
    assert empty["status"] == "blocked"
    assert empty["blockers"] == ["thread_not_branched"]
    assert empty["source"] == "thread_checkout"

    branched = client.post(f"/api/projects/racer/threads/{tid}/branch-off", json={}).json()
    assert branched["ok"] is True, branched
    checkout = pathlib.Path(branched["path"])

    clean = client.get(f"/api/projects/racer/threads/{tid}/diff").json()
    assert clean["status"] == "empty"
    assert clean["patch"] == ""

    (checkout / "app.txt").write_text("changed by the thread\n", encoding="utf-8")
    (checkout / "brand_new.txt").write_text("new file\n", encoding="utf-8")

    ready = client.get(f"/api/projects/racer/threads/{tid}/diff").json()
    assert ready["status"] == "ready"
    assert ready["project_id"] == "racer" and ready["thread_id"] == tid
    assert ready["source"] == "thread_checkout"
    assert ready["base_commit"] == branched["base_sha"]
    # Unsaved edits AND untracked new files, because that is what the owner sees
    # when they open that folder.
    assert "changed by the thread" in ready["patch"]
    assert "brand_new.txt" in ready["patch"]
    assert ready["patch_sha256"]


def test_diff_of_an_unknown_thread_is_the_only_404(wired):
    client, _drive, _tid, _folder = wired
    assert client.get("/api/projects/racer/threads/999/diff").status_code == 404


def test_merge_back_conflict_is_a_409_with_its_paths(wired):
    client, _drive, tid, folder = wired
    branched = client.post(f"/api/projects/racer/threads/{tid}/branch-off", json={}).json()
    checkout = pathlib.Path(branched["path"])
    (checkout / "app.txt").write_text("thread version\n", encoding="utf-8")
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "T")
    _git(checkout, "commit", "-qam", "thread edit")
    (folder / "app.txt").write_text("owner version\n", encoding="utf-8")
    _git(folder, "commit", "-qam", "owner edit")

    response = client.post(f"/api/projects/racer/threads/{tid}/merge-back", json={})

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False
    assert body["reason"] == "merge_conflict"
    assert body["conflicts"] == ["app.txt"]
    assert body["message"]
    # The thread is still where it was, with its branch intact.
    assert body["branch"]
    assert (folder / "app.txt").read_text(encoding="utf-8") == "owner version\n"


def test_removal_refuses_unmerged_work_until_the_owner_acknowledges_it(wired):
    """A10, end to end: the inspection is SHOWN, the refusal names the stakes,
    and the acknowledgement is the only way through."""
    client, _drive, tid, _folder = wired
    branched = client.post(f"/api/projects/racer/threads/{tid}/branch-off", json={}).json()
    checkout = pathlib.Path(branched["path"])
    (checkout / "app.txt").write_text("unsaved thread work\n", encoding="utf-8")

    inspected = client.get(f"/api/projects/racer/threads/{tid}/worktree").json()
    assert inspected["inspection"]["dirty"] is True
    assert inspected["location"]["where"] == "worktree"

    refused = client.post(f"/api/projects/racer/threads/{tid}/worktree/remove", json={})
    assert refused.status_code == 409
    body = refused.json()
    assert body["removed"] is False
    assert body["reason"] == "unmerged_work"
    assert "Removing it deletes that work" in body["message"]
    assert checkout.is_dir()

    removed = client.post(
        f"/api/projects/racer/threads/{tid}/worktree/remove",
        json={"acknowledge_unmerged": True},
    )
    assert removed.status_code == 200
    assert removed.json()["removed"] is True
    assert removed.json()["location"]["where"] == "project_folder"
    assert not checkout.exists()


def test_removal_refuses_while_a_task_is_running_in_the_project(wired, monkeypatch):
    """T3R2-H5 at the route: removal deletes a folder a task may be writing in,
    and the route has a sentence for that reason instead of a bare fallback."""
    from supervisor import workers

    client, _drive, tid, folder = wired
    branched = client.post(f"/api/projects/racer/threads/{tid}/branch-off", json={}).json()
    checkout = pathlib.Path(branched["path"])
    monkeypatch.setitem(
        workers.RUNNING, "live",
        {"task": {"id": "live", "project_id": "racer", "workspace_root": str(checkout)}},
    )
    try:
        refused = client.post(
            f"/api/projects/racer/threads/{tid}/worktree/remove",
            json={"acknowledge_unmerged": True},
        )
    finally:
        workers.RUNNING.pop("live", None)

    assert refused.status_code == 409
    body = refused.json()
    assert body["removed"] is False
    assert body["reason"] == "project_busy"
    assert "until that task finishes" in body["message"]
    assert checkout.is_dir()
    # A WAIT, not a dead end: with the task gone it removes.
    assert client.post(
        f"/api/projects/racer/threads/{tid}/worktree/remove",
        json={"acknowledge_unmerged": True},
    ).json()["removed"] is True


def test_a_folderless_project_refuses_with_a_typed_reason_not_a_500(tmp_path):
    drive = tmp_path / "drive"
    create_project(drive, "placeless", name="Placeless")
    thread = create_thread(drive, "placeless", name="Side quest")
    client = _client(drive)

    response = client.post(
        f"/api/projects/placeless/threads/{thread['id']}/branch-off", json={},
    )

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False and body["reason"] == "no_project_folder"
    assert body["message"]


def test_an_unknown_base_is_a_400(wired):
    client, _drive, tid, _folder = wired

    response = client.post(
        f"/api/projects/racer/threads/{tid}/branch-off", json={"base_ref": "nope"},
    )

    assert response.status_code == 400
    assert response.json()["reason"] == "unknown_base"


# --------------------------------------------------------------------------- #
# Lifecycle (D4 / X10) over HTTP
# --------------------------------------------------------------------------- #

def _lifecycle_client(drive_root: pathlib.Path) -> TestClient:
    from ouroboros.gateway.project_threads import (
        api_thread_archive,
        api_thread_delete,
        api_thread_restore,
    )

    base = "/api/projects/{project_id}/threads/{thread_id}"
    app = Starlette(routes=[
        Route(f"{base}/archive", api_thread_archive, methods=["POST"]),
        Route(f"{base}/restore", api_thread_restore, methods=["POST"]),
        Route(f"{base}/delete", api_thread_delete, methods=["POST"]),
    ])
    app.state.drive_root = drive_root
    app.state.repo_dir = drive_root
    return TestClient(app)


def test_archive_and_restore_round_trip_over_http(tmp_path):
    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer")
    thread = create_thread(drive, "racer", name="Side quest")
    client = _lifecycle_client(drive)

    archived = client.post(f"/api/projects/racer/threads/{thread['id']}/archive", json={}).json()
    assert archived["ok"] is True
    assert archived["lifecycle"] == "archived"
    assert archived["archived_at"]
    # Nothing is running, so nothing is being kept visible against the owner's wish.
    assert archived["visible_until_terminal"] is False

    restored = client.post(f"/api/projects/racer/threads/{thread['id']}/restore", json={}).json()
    assert restored["lifecycle"] == "active"


def test_archiving_thread_zero_is_a_409_that_says_where_the_operation_lives(tmp_path):
    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer")
    client = _lifecycle_client(drive)

    response = client.post("/api/projects/racer/threads/0/archive", json={})

    assert response.status_code == 409
    body = response.json()
    assert body["reason"] == "thread_zero_is_the_project"
    assert "project" in body["message"].lower()


def test_delete_takes_a_CLEAN_checkout_with_it_and_says_so(tmp_path, folder, monkeypatch):
    """T3R2-M2, owner-directed: deleting a thread must delete its worktree too.

    A tombstoned thread is invisible on every surface, `list_thread_worktrees` has
    no route and no UI consumer, and branch/merge refuse `thread_not_live` — so a
    checkout left behind is a folder AND a branch that A10's explicit removal can
    no longer reach, on durable state exempt from every GC. A CLEAN one (nothing
    uncommitted, no commit the project folder lacks) is exactly what A10/D4
    already offer one-click removal for, so it goes with the thread and the answer
    says it did. X10 is unchanged: the fence is up, the tombstone is not.
    """
    started: list = []
    monkeypatch.setattr(
        "supervisor.task_lifecycle.start_thread_deletion",
        lambda drive_root, pid, tid, chat_id: started.append((pid, tid, chat_id)) or True,
    )
    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer", working_dir=str(folder))
    thread = create_thread(drive, "racer", name="Doomed")
    branched = _client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/branch-off", json={},
    ).json()
    assert branched["ok"] is True

    body = _lifecycle_client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/delete", json={},
    ).json()

    assert body["ok"] is True
    # Fenced, NOT yet tombstoned: its tasks are still being cancelled.
    assert body["lifecycle"] == "deleting"
    assert started and started[0][1] == int(thread["id"])
    assert body["journal_rows_retained"] is True
    # The checkout AND its branch went with it — disclosed, never silent.
    assert body["worktree_removed"] is True
    assert body["worktree_kept"] is False
    assert body["branch"] == branched["branch"]
    assert body["branch_removed"] is True
    assert not pathlib.Path(branched["path"]).exists()
    assert branched["branch"] not in _git(folder, "branch", "--list").stdout


def test_delete_REFUSES_while_the_checkout_still_holds_work(tmp_path, folder, monkeypatch):
    """Work AT RISK still stops the delete dead, and the refusal names the route.

    "At risk" is the narrow set: commits that exist nowhere else, and changes to
    files git is TRACKING. Both are exercised here — an edited tracked file, then
    that edit committed onto the thread's branch — because the two are separate
    causes with separate sentences and only one of them survives a `git commit`.
    Neither is answerable: `acknowledge_unmerged` must NOT open this door, or the
    narrowing would have turned into an override for the thing it must not touch.
    """
    started: list = []
    monkeypatch.setattr(
        "supervisor.task_lifecycle.start_thread_deletion",
        lambda drive_root, pid, tid, chat_id: started.append(tid) or True,
    )
    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer", working_dir=str(folder))
    thread = create_thread(drive, "racer", name="Doomed")
    branched = _client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/branch-off", json={},
    ).json()
    checkout = pathlib.Path(branched["path"])
    # A TRACKED file, edited. Its previous contents are in history; this edit is
    # nowhere else at all.
    (checkout / "app.txt").write_text("hours of work\n", encoding="utf-8")

    for body_sent in ({}, {"acknowledge_unmerged": True}):
        response = _lifecycle_client(drive).post(
            f"/api/projects/racer/threads/{thread['id']}/delete", json=body_sent,
        )
        body = response.json()

        assert response.status_code == 409, body
        assert body["ok"] is False
        assert body["reason"] == "checkout_holds_work"
        # The refusal NAMES what is at stake and where to go next.
        assert "git is tracking" in body["message"]
        assert "Remove checkout" in body["message"]
        # ...and is not a question, so no menu may render a confirm for it.
        assert body["acknowledgeable"] is False
        assert body["inspection"]["dirty"] is True
        # Nothing happened: not fenced, not tombstoned, checkout intact.
        assert started == []
        assert (checkout / "app.txt").read_text(encoding="utf-8") == "hours of work\n"
        from ouroboros.projects_registry import get_thread

        assert get_thread(drive, "racer", thread["id"])["lifecycle"] == "active"

    # Commit it: the tree is clean now, but the commit exists nowhere else.
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "T")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-m", "the only copy")

    committed = _lifecycle_client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/delete", json={"acknowledge_unmerged": True},
    )

    assert committed.status_code == 409
    assert committed.json()["reason"] == "checkout_holds_work"
    assert "exist nowhere else" in committed.json()["message"]
    assert started == []
    assert checkout.is_dir()


def test_delete_takes_a_checkout_holding_only_IGNORED_files_in_TWO_steps(
    tmp_path, folder, monkeypatch,
):
    """T3R2-M2 follow-up, owner-directed: `сложно разве ворктрии снести?`

    H3 made the inspection count ignored files as dirt — correct, a one-click
    removal was destroying `.env`/`local.db` silently. M2 then made ANY unclean
    inspection refuse the DELETE. Together, a checkout holding nothing but
    `node_modules/` and a `build.log` made deleting the thread refuse, with a
    three-step escape through merge-back and an acknowledged removal. That is
    friction over files that can be rebuilt by running a command.

    So deletion asks a narrower question, and the rebuildable case rides the same
    acknowledgement shape the rest of this surface uses: refuse naming the files,
    then confirm. Two steps, and the checkout goes with the thread.
    """
    started: list = []
    monkeypatch.setattr(
        "supervisor.task_lifecycle.start_thread_deletion",
        lambda drive_root, pid, tid, chat_id: started.append(tid) or True,
    )
    (folder / ".gitignore").write_text("node_modules/\n*.log\n", encoding="utf-8")
    _git(folder, "add", "-A")
    _git(folder, "commit", "-m", "ignore build artefacts")

    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer", working_dir=str(folder))
    thread = create_thread(drive, "racer", name="Doomed")
    branched = _client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/branch-off", json={},
    ).json()
    checkout = pathlib.Path(branched["path"])
    (checkout / "node_modules").mkdir()
    (checkout / "node_modules" / "left-pad.js").write_text("//\n", encoding="utf-8")
    (checkout / "build.log").write_text("ok\n", encoding="utf-8")

    # H3 is untouched: the INSPECTION still sees them, because removal's own
    # prompt has to say what it would destroy.
    inspection = _client(drive).get(
        f"/api/projects/racer/threads/{thread['id']}/worktree",
    ).json()["inspection"]
    assert inspection["dirty"] is True
    assert any("build.log" in line for line in inspection["dirty_files"])

    # STEP ONE — refused, and it says exactly what is in there.
    first = _lifecycle_client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/delete", json={},
    )
    body = first.json()
    assert first.status_code == 409
    assert body["reason"] == "checkout_holds_rebuildable_files"
    assert "git was told to ignore" in body["message"]
    assert "nothing committed" in body["message"]
    # It is a QUESTION, so a menu can render the second call.
    assert body["acknowledgeable"] is True
    assert body["inspection"]["dirty"] is True
    assert started == []
    assert checkout.is_dir()

    # STEP TWO — the owner's yes, in the same field name the removal route uses.
    second = _lifecycle_client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/delete",
        json={"acknowledge_unmerged": True},
    ).json()

    assert second["ok"] is True
    assert second["lifecycle"] == "deleting"
    assert started == [int(thread["id"])]
    assert second["worktree_removed"] is True
    assert second["worktree_kept"] is False
    assert not checkout.exists(), "the checkout goes with the thread"
    # ...and so does the branch: it held no commits, so keeping it would have
    # left a `thread/<name>` nothing can reach once the thread is tombstoned.
    assert second["branch_removed"] is True
    assert branched["branch"] not in _git(folder, "branch", "--list").stdout


def test_delete_takes_a_checkout_holding_only_UNTRACKED_files_after_confirming(
    tmp_path, folder, monkeypatch,
):
    """Same door for content git has simply never been told about.

    An untracked file is not on any branch and no history has it, so it cannot be
    "unmerged" — but it can be the owner's scratch note, so it is NAMED and
    confirmed rather than either destroyed silently or made a wall.
    """
    monkeypatch.setattr(
        "supervisor.task_lifecycle.start_thread_deletion",
        lambda drive_root, pid, tid, chat_id: True,
    )
    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer", working_dir=str(folder))
    thread = create_thread(drive, "racer", name="Doomed")
    branched = _client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/branch-off", json={},
    ).json()
    checkout = pathlib.Path(branched["path"])
    (checkout / "scratch.txt").write_text("notes\n", encoding="utf-8")

    refused = _lifecycle_client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/delete", json={},
    ).json()
    assert refused["reason"] == "checkout_holds_rebuildable_files"
    assert "git is not tracking" in refused["message"]
    assert refused["acknowledgeable"] is True
    assert checkout.is_dir()

    done = _lifecycle_client(drive).post(
        f"/api/projects/racer/threads/{thread['id']}/delete",
        json={"acknowledge_unmerged": True},
    ).json()
    assert done["ok"] is True
    assert done["worktree_removed"] is True
    assert not checkout.exists()


def test_a_delete_that_will_be_REFUSED_never_removes_the_checkout_first(tmp_path, folder, monkeypatch):
    """Self-review of the M2 ordering. The checkout is removed BEFORE the fence so
    a refusal leaves the thread exactly as it was — which means a transition the
    registry would refuse must be caught HERE, or the folder is destroyed on the
    way to a 409. Thread #0 is the project (`thread_zero_is_the_project`), and a
    tombstone is terminal."""
    started: list = []
    monkeypatch.setattr(
        "supervisor.task_lifecycle.start_thread_deletion",
        lambda drive_root, pid, tid, chat_id: started.append(tid) or True,
    )
    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer", working_dir=str(folder))
    branched = _client(drive).post("/api/projects/racer/threads/0/branch-off", json={}).json()
    assert branched["ok"] is True, branched

    response = _lifecycle_client(drive).post("/api/projects/racer/threads/0/delete", json={})

    assert response.status_code == 409
    assert response.json()["reason"] == "thread_zero_is_the_project"
    assert started == []
    assert pathlib.Path(branched["path"]).is_dir(), "a refused delete must destroy nothing"


def test_branch_bases_carries_the_honest_queue_notice(wired, monkeypatch):
    """A14: the sentence says QUEUED, not rejected, and offers branching."""
    from supervisor import workers

    client, _drive, tid, folder = wired

    quiet = client.get(f"/api/projects/racer/threads/{tid}/branch-bases").json()
    # T3R2-L5: the response contract declares `ok` and the refusal path sets it,
    # so a client reading `body.ok` first — which the shared envelope asks of it —
    # read every SUCCESSFUL bases list as a refusal.
    assert quiet["ok"] is True
    assert quiet["queue_notice"]["queued"] is False
    assert quiet["queue_notice"]["message"] == ""

    monkeypatch.setitem(
        workers.RUNNING, "t1",
        {"task": {"id": "t1", "project_id": "racer", "workspace_root": str(folder)}},
    )
    try:
        busy = client.get(f"/api/projects/racer/threads/{tid}/branch-bases").json()
    finally:
        workers.RUNNING.pop("t1", None)

    notice = busy["queue_notice"]
    assert notice["queued"] is True
    assert "QUEUED" in notice["message"]
    assert "rejected" in notice["message"]  # ...as the thing it explicitly is NOT
    assert "is not rejected" in notice["message"]
    assert notice["remedy"] == "branch_off"


def test_the_queue_notice_names_the_FOLDER_not_a_guess_about_who_holds_it(wired, monkeypatch):
    """T3R-15. After T0R2-5 the writer lane is keyed on the FOLDER alone, across
    projects and threads alike — so whatever is holding it may belong to another
    thread, another project, or no project at all.

    "Another thread in this project" was a guess about the occupant, and a wrong
    guess sends the owner looking for a room that is not the one making them wait.
    """
    from ouroboros.thread_branching import QUEUE_NOTICE
    from supervisor import workers

    client, _drive, tid, folder = wired
    # The occupant belongs to a DIFFERENT project, in the same folder.
    monkeypatch.setitem(
        workers.RUNNING, "other",
        {"task": {"id": "other", "project_id": "someone-else", "workspace_root": str(folder)}},
    )
    try:
        notice = client.get(
            f"/api/projects/racer/threads/{tid}/branch-bases"
        ).json()["queue_notice"]
    finally:
        workers.RUNNING.pop("other", None)

    assert notice["queued"] is True
    assert notice["message"] == QUEUE_NOTICE
    assert "Another task is working in this folder" in notice["message"]
    assert "this project" not in notice["message"], "the occupant's project is not known here"


def test_a_broken_queue_notice_never_takes_the_BASES_LIST_down_with_it(wired, monkeypatch):
    """T3R-13. The notice is the least important thing on this answer — an
    advisory beside the list of bases the owner actually came for.

    Its fail-open guard covered only the queue READ. The `project_lease` import
    and the `candidate_is_leasable` call sat OUTSIDE it — and that call raises
    TypeError by contract on a malformed lane — so anything wrong there 500'd the
    whole route and the owner lost their bases list to a sentence about waiting.
    """
    import ouroboros.project_lease as lease

    client, _drive, tid, folder = wired
    monkeypatch.setitem(
        workers_module().RUNNING, "t1",
        {"task": {"id": "t1", "project_id": "racer", "workspace_root": str(folder)}},
    )

    def _explode(*_a, **_kw):
        raise TypeError("candidate_is_leasable expects lane keys")

    monkeypatch.setattr(lease, "candidate_is_leasable", _explode)
    try:
        response = client.get(f"/api/projects/racer/threads/{tid}/branch-bases")
    finally:
        workers_module().RUNNING.pop("t1", None)

    assert response.status_code == 200
    body = response.json()
    assert body["queue_notice"] == {"queued": False, "reason": "", "message": "", "remedy": ""}
    # The answer the owner came for is intact.
    assert body["current_branch"] == "main"
    assert body["snapshot"]["ref"] == "@snapshot"


def test_a_queue_notice_whose_own_IMPORT_fails_is_still_only_a_missing_sentence(wired, monkeypatch):
    """The import itself was outside the guard too, so a `project_lease` that
    would not load took the route with it."""
    import sys

    from ouroboros.thread_branching import queue_notice

    client, drive, tid, _folder = wired
    monkeypatch.setitem(sys.modules, "ouroboros.project_lease", None)

    assert queue_notice(drive, "racer", tid) == {
        "queued": False, "reason": "", "message": "", "remedy": "",
    }
    assert client.get(f"/api/projects/racer/threads/{tid}/branch-bases").status_code == 200


def workers_module():
    from supervisor import workers

    return workers


def test_a_failing_thread_route_never_answers_in_a_refusal_shape(monkeypatch):
    """A REFUTATION, pinned so it stays true.

    `project_thread_actions.js::typedAnswer` turns a thrown `error.body` into a
    typed refusal when it carries a `reason` string or `ok === false`, and two
    independent reviewers have now argued that a JSON 500/502 with `ok:false`
    would therefore be rendered as an owner-answerable decision — an outage
    dressed up as something the owner chose.

    Nothing in this stack produces that shape. Every one of the eight routes
    `threadOps` calls guards its whole body and answers a failure through
    `gateway._helpers.json_exception`, whose payload is `{"error": ...}` — no
    `ok`, no `reason` — so `typedAnswer` re-throws and the menu reports an error.
    Anything escaping the guard is Starlette's plain-text 500, which
    `fetchJson` cannot parse and turns into `{error: "non-json response (HTTP
    500)"}`: again no `ok` and no `reason`. `_refusal_status` never returns 5xx,
    and the app installs no exception handler or middleware that could rewrite a
    body.

    So the fix for that finding is this test rather than a speculative client
    guard: if a route ever starts answering a failure with `ok` or `reason`, the
    refutation stops being true and this fails loudly.
    """
    import asyncio
    import json as _json

    import ouroboros.gateway.project_threads as pt

    routes = [
        "api_thread_branch_bases", "api_thread_branch_off", "api_thread_merge_back",
        "api_thread_worktree_inspect", "api_thread_worktree_remove",
        "api_thread_archive", "api_thread_restore", "api_thread_delete",
    ]

    def _boom(*_a, **_k):
        raise RuntimeError("the drive went away mid-request")

    monkeypatch.setattr(pt, "request_drive_root", _boom)

    async def _drive(handler):
        scope = {
            "type": "http", "method": "POST", "path": "/x", "headers": [],
            "path_params": {"project_id": "p", "thread_id": "1"}, "query_string": b"",
            "app": None,
        }

        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        from starlette.requests import Request

        return await handler(Request(scope, receive))

    for name in routes:
        response = asyncio.run(_drive(getattr(pt, name)))
        assert response.status_code >= 500, name
        body = _json.loads(bytes(response.body).decode("utf-8"))
        assert "ok" not in body, f"{name} answered a 5xx wearing a refusal's clothes: {body}"
        assert "reason" not in body, f"{name} answered a 5xx with a typed reason: {body}"
        assert body.get("error"), name
