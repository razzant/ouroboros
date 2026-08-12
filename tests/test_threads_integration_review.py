"""Integration-review findings on the merged project-threads tree (I1-I19).

These are CROSS-STREAM defects: each one lives in the seam between two phases
that were built against different foundations, so none of them is visible from
inside either stream's own suite. Every test here reproduces the defect against
REAL git and the REAL supervisor queue where the defect involves them — mocking
either would pin our beliefs about them rather than their behaviour, and the
gestures under test destroy an owner's folder when they are wrong.

Numbering follows the review packet so a finding can be traced from the report to
the guard that keeps it closed.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import subprocess
import time
from types import SimpleNamespace

import pytest

from ouroboros.project_threads_registry import create_thread
from ouroboros.projects_registry import create_project
from ouroboros.thread_branching import (
    BASE_SNAPSHOT,
    REASON_PROJECT_BUSY,
    branch_off_thread,
)


def _git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check)


@pytest.fixture()
def folder(tmp_path):
    """A real git repository standing in for the owner's project folder."""
    root = tmp_path / "owner_folder"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "app.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "seed")
    return root


@pytest.fixture()
def drive(tmp_path):
    return tmp_path / "drive"


@pytest.fixture()
def wt_root(tmp_path):
    return tmp_path / "thread_worktrees"


def _project(drive, folder, pid="racer"):
    create_project(drive, pid, name="Racer", working_dir=str(folder))
    return create_thread(drive, pid, name="Side quest")


def _commit_in(root, name, text):
    path = pathlib.Path(root) / name
    path.write_text(text, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", f"add {name}")


def _delete_request(drive_root, project_id):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=drive_root)),
        path_params={"project_id": project_id},
    )


@pytest.fixture()
def quiet_project_delete(monkeypatch):
    """Keep the delete route's cancellation worker out of these tests.

    The worker is a real thread that tombstones the project; every assertion here
    is about what the ROUTE did to the checkouts before it started, and a
    background tombstone racing the assertions would make them flaky. The sweep
    the worker performs has its own test.
    """
    import supervisor.task_lifecycle as lifecycle

    started: list = []
    # The route imports this name at CALL time, so the module attribute is the
    # seam. Patching `gateway.projects` would silently do nothing.
    monkeypatch.setattr(
        lifecycle, "start_project_deletion", lambda *a, **k: started.append(a) or True,
    )
    return started


# --------------------------------------------------------------------------- #
# I1 — `Delete project…` used to orphan every branched thread's checkout
# --------------------------------------------------------------------------- #

def test_project_delete_refuses_while_a_thread_checkout_holds_work(
    drive, folder, wt_root, monkeypatch, quiet_project_delete,
):
    """I1 BLOCKER. Every clause `api_thread_delete` gives for taking a thread's
    checkout with the thread is equally true of the PROJECT, and nothing applied
    it: the route fenced and tombstoned while N checkouts and N `thread/*`
    branches stayed on disk, reachable from no surface at all.

    A checkout holding work the project folder never received must therefore
    REFUSE, before anything is fenced — a project on its way to a tombstone is
    the one state from which the owner cannot get that work back.
    """
    monkeypatch.setenv("OUROBOROS_THREAD_WORKTREE_ROOT", str(wt_root))
    thread = _project(drive, folder)
    out = branch_off_thread(drive, "racer", thread["id"], data_dir=drive, worktree_root=wt_root)
    assert out["ok"] is True
    _commit_in(out["path"], "only_copy.txt", "exists nowhere else\n")

    from ouroboros.gateway.projects import api_project_delete
    from ouroboros.projects_registry import get_reserved_project
    from ouroboros.thread_worktrees import get_thread_worktree

    response = asyncio.run(api_project_delete(_delete_request(drive, "racer")))

    assert response.status_code == 409
    body = json.loads(response.body)
    assert body["ok"] is False
    assert body["reason"] == "threads_hold_checkouts"
    # It NAMES the thread and reuses the single-thread sentence, so the two
    # gestures explain the identical fact identically.
    assert f"Thread {thread['id']}" in body["message"]
    assert "exist nowhere else" in body["message"]
    assert "Remove checkout" in body["message"]
    assert [row["thread_id"] for row in body["threads"]] == [thread["id"]]
    # NOTHING happened: not the fence, not the folder.
    assert (get_reserved_project(drive, "racer") or {})["lifecycle"] == "active"
    assert get_thread_worktree(drive, "racer", thread["id"]) is not None
    assert (pathlib.Path(out["path"]) / "only_copy.txt").is_file()


def test_project_delete_takes_its_threads_clean_checkouts_and_branches_with_it(
    drive, folder, wt_root, monkeypatch, quiet_project_delete,
):
    """I1, the other half: a checkout with nothing at risk goes WITH the project
    and is DISCLOSED, because a tombstoned project leaves no surface that could
    reach the folder or its `thread/*` branch (D4's "never silently" has to mean
    "never orphaned" too).
    """
    monkeypatch.setenv("OUROBOROS_THREAD_WORKTREE_ROOT", str(wt_root))
    thread = _project(drive, folder)
    out = branch_off_thread(drive, "racer", thread["id"], data_dir=drive, worktree_root=wt_root)
    assert out["ok"] is True
    checkout = pathlib.Path(out["path"])
    branch = out["branch"]

    from ouroboros.gateway.projects import api_project_delete
    from ouroboros.thread_worktrees import get_thread_worktree

    response = asyncio.run(api_project_delete(_delete_request(drive, "racer")))

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["worktrees_removed"] == [thread["id"]]
    assert body["branches_removed"] == [branch]
    assert body["worktrees_pending"] == []
    assert not checkout.exists()
    assert get_thread_worktree(drive, "racer", thread["id"]) is None
    assert _git(folder, "rev-parse", "--verify", branch, check=False).returncode != 0
    # The owner's own folder is untouched, exactly as the answer says.
    assert (folder / "app.txt").read_text(encoding="utf-8") == "one\n"


def test_a_checkout_a_task_is_still_writing_in_is_swept_after_quiescence(
    drive, folder, wt_root, monkeypatch,
):
    """I1's remaining gap: the removal's own busy judge refuses while a task is
    in the folder, so the route cannot take that checkout yet. It says so
    (`worktrees_pending`) and the cancellation worker takes it once the project
    quiesces — no path leaves an orphan silently.
    """
    monkeypatch.setenv("OUROBOROS_THREAD_WORKTREE_ROOT", str(wt_root))
    from supervisor import workers

    thread = _project(drive, folder)
    out = branch_off_thread(drive, "racer", thread["id"], data_dir=drive, worktree_root=wt_root)
    checkout = pathlib.Path(out["path"])

    from ouroboros.gateway.projects import api_project_delete
    from ouroboros.thread_worktrees import get_thread_worktree
    from supervisor.task_lifecycle import _sweep_project_checkouts

    monkeypatch.setitem(
        workers.RUNNING, "t-live",
        {"task": {"id": "t-live", "project_id": "racer", "workspace_root": str(checkout)}},
    )
    try:
        import supervisor.task_lifecycle as lifecycle

        monkeypatch.setattr(lifecycle, "start_project_deletion", lambda *a, **k: True)
        response = asyncio.run(api_project_delete(_delete_request(drive, "racer")))
        body = json.loads(response.body)
        assert response.status_code == 200
        assert body["worktrees_removed"] == []
        assert [row["thread_id"] for row in body["worktrees_pending"]] == [thread["id"]]
        assert body["worktrees_pending"][0]["reason"] == "project_busy"
        assert checkout.is_dir()
    finally:
        workers.RUNNING.pop("t-live", None)

    # ...and once the tasks are gone, the sweep the deletion worker runs takes it.
    _sweep_project_checkouts(drive, "racer")
    assert not checkout.exists()
    assert get_thread_worktree(drive, "racer", thread["id"]) is None


# --------------------------------------------------------------------------- #
# I3 — "the folders are unknown" is not "no project has one"
# --------------------------------------------------------------------------- #

def test_project_working_dirs_tells_unreadable_apart_from_empty(tmp_path):
    """I3. No exception is required to lose the fail-safe: `_load` fails open to
    `{"projects": []}`, so a truncated write, a partial `atomic_write` on a full
    disk or a hand-edit produced an EMPTY map — indistinguishable from "no project
    has a folder", which the lane is entitled to narrow on.
    """
    from ouroboros.projects_registry import _registry_path, project_working_dirs

    # No registry at all: "this install has no projects" is a fact.
    assert project_working_dirs(tmp_path) == {}

    folder = tmp_path / "alpha"
    folder.mkdir()
    create_project(tmp_path, "alpha", working_dir=str(folder))
    assert project_working_dirs(tmp_path) == {"alpha": str(folder.resolve())}

    # The file is THERE and holds no project list.
    path = _registry_path(tmp_path)
    path.write_text('{"projects": "truncat', encoding="utf-8")
    assert project_working_dirs(tmp_path) is None
    path.write_text("{}", encoding="utf-8")
    assert project_working_dirs(tmp_path) is None


def test_an_unreadable_folder_map_never_admits_a_second_writer():
    """I3, both directions, restoring T0's deleted `WILDCARD_WORKSPACE` fail-safe
    without resurrecting the wildcard.

    Under the empty map a second writer entered `/w/alpha` from either side: a
    folder-bearing candidate matched no narrow lane, and a placeless RUNNING
    holder stopped blocking a folder-bearing candidate. `None` — the honest value
    for an unreadable registry — queues both. `{}` keeps the honest narrow key for
    the genuinely file-less project.
    """
    from ouroboros.project_lease import candidate_is_leasable, running_project_lanes

    placeless_holder = {"id": "t1", "project_id": "alpha"}
    folder_candidate = {"id": "t2", "project_id": "alpha", "workspace_root": "/w/alpha"}

    # Direction 1: the HOLDER names the folder, the candidate does not.
    held_folder = running_project_lanes([{"task": folder_candidate}], None)
    assert candidate_is_leasable(placeless_holder, held_folder, None) is False
    assert candidate_is_leasable(placeless_holder, held_folder, {}) is True   # honest narrow key

    # Direction 2: the HOLDER is placeless, the candidate names the folder it may
    # well be writing in.
    held_narrow = running_project_lanes([{"task": placeless_holder}], None)
    assert held_narrow == {("alpha", "")}
    assert candidate_is_leasable(folder_candidate, held_narrow, None) is False
    assert candidate_is_leasable(folder_candidate, held_narrow, {}) is True   # honest narrow key

    # An unscoped task never serializes, and nothing held means nothing to queue
    # behind.
    assert candidate_is_leasable({"id": "t4"}, held_narrow, None) is True
    assert candidate_is_leasable(placeless_holder, set(), None) is True


def test_an_unresolved_project_lane_blocks_every_folder_bearing_candidate():
    """The half of I3 that was left open: only the candidate's OWN project key
    was compared against the narrow lanes.

    A narrow `(project_id, "")` lane means "this RUNNING writer's folder could
    not be read". Asking whether it is the folder a candidate names is exactly
    as unanswerable — the holder's registered `working_dir` may BE that folder,
    and two projects may share one, which is why the lane is folder-keyed at
    all. So a candidate naming `/w/shared` under ANOTHER project id, and a
    PROJECTLESS candidate naming it, were both admitted straight into a folder a
    live writer may hold, while the same-project candidate beside them queued.
    Two writers in one folder is the one thing the lane exists to prevent.
    """
    from ouroboros.project_lease import candidate_is_leasable, running_project_lanes

    placeless_holder = {"id": "t1", "project_id": "alpha"}
    held_narrow = running_project_lanes([{"task": placeless_holder}], None)
    assert held_narrow == {("alpha", "")}

    cross_project = {"id": "t2", "project_id": "beta", "workspace_root": "/w/shared"}
    projectless = {"id": "t3", "workspace_root": "/w/shared"}
    assert candidate_is_leasable(cross_project, held_narrow, None) is False
    assert candidate_is_leasable(projectless, held_narrow, None) is False
    # ...including a folder that has nothing to do with alpha: the map is
    # unreadable, so "nothing to do with" is not a fact anything here holds.
    assert candidate_is_leasable(
        {"id": "t5", "project_id": "beta", "workspace_root": "/w/beta"}, held_narrow, None,
    ) is False

    # It is still not a wildcard. With a REAL map (even an empty one) the narrow
    # key is honest again and a different folder runs...
    assert candidate_is_leasable(cross_project, held_narrow, {}) is True
    assert candidate_is_leasable(projectless, held_narrow, {}) is True
    # ...and an unreadable map with only RESOLVED folder lanes held blocks only
    # the folder it actually names.
    held_folder = running_project_lanes(
        [{"task": {"id": "r", "project_id": "alpha", "workspace_root": "/w/alpha"}}], None,
    )
    assert held_folder == {("", os.path.normcase("/w/alpha"))}
    assert candidate_is_leasable(
        {"id": "t6", "project_id": "beta", "workspace_root": "/w/beta"}, held_folder, None,
    ) is True
    assert candidate_is_leasable(
        {"id": "t7", "project_id": "beta", "workspace_root": "/w/alpha"}, held_folder, None,
    ) is False


def test_assign_tasks_hands_the_lease_None_when_the_registry_cannot_be_read(tmp_path, monkeypatch):
    """The scheduler's own except-arm must carry the new value, or I3's fix is
    inert at the one call site that decides who runs."""
    import supervisor.workers as workers

    src = pathlib.Path(workers.__file__).read_text(encoding="utf-8")
    block = src.split("assign_tasks: project working_dir map unavailable")[1][:200]
    assert "_project_workspaces = None" in block, (
        "assign_tasks still fails open to {}, which the lane reads as "
        "'no project has a folder' and narrows on (I3)"
    )


# --------------------------------------------------------------------------- #
# I4 — a stale lane pin must not survive a requeue
# --------------------------------------------------------------------------- #

def test_a_requeued_task_does_not_carry_its_previous_lane_pin(tmp_path, monkeypatch):
    """I4. `ensure_workers_healthy` re-enqueues the very dict `assign_tasks`
    stamped, and `enqueue_task`'s field-stripping allowlist did not include the
    pin — so attempt 2 held the lane attempt 1 was pinned into. With the folder
    attached between attempts it froze `('alpha','')` while writing `/w/alpha`,
    and a room task in that folder read the folder as free.

    `persist_queue_snapshot` was already safe (an allowlist that omits the pin);
    the in-process retry was not, and the fix belongs in the enqueue SSOT so every
    requeue path is covered at once.
    """
    import supervisor.queue as queue
    from ouroboros.project_lease import (
        LANE_PIN_FIELD,
        candidate_is_leasable,
        pin_task_lane,
        running_project_lanes,
    )

    pending: list = []
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "queue_snapshot.json")
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})

    attempt1 = {"id": "t1", "type": "task", "project_id": "alpha", "_attempt": 1}
    assert pin_task_lane(attempt1, {}) == ("alpha", "")   # no folder registered yet
    assert attempt1[LANE_PIN_FIELD] == ["alpha", ""]

    queue.enqueue_task(dict(attempt1), front=True)
    assert pending and LANE_PIN_FIELD not in pending[0]

    # The folder is attached between attempts, so attempt 2 must pin the FOLDER.
    folders = {"alpha": "/w/alpha"}
    attempt2 = dict(pending[0])
    assert pin_task_lane(attempt2, folders) == ("", os.path.normcase("/w/alpha"))
    lanes = running_project_lanes([{"task": attempt2}], folders)
    assert candidate_is_leasable(
        {"id": "room", "project_id": "alpha", "workspace_root": "/w/alpha"}, lanes, folders,
    ) is False


# --------------------------------------------------------------------------- #
# I5 — ONE answer to "is this folder occupied"
# --------------------------------------------------------------------------- #

def test_project_is_busy_sees_a_non_task_holder(monkeypatch):
    """I5. The merge unioned `reserved_folder_lane` into `running_project_lanes`
    so the SCHEDULER sees a merge-back holding the folder, while
    `project_is_busy` — the SSOT for every owner gesture's precondition — read
    only the two task queries. During a merge-back a second merge-back, a checkout
    removal and a thread delete were all told IDLE, and a second holder was in
    fact admitted.
    """
    import threading

    import ouroboros.project_lease as lease
    import ouroboros.thread_branching as branching
    from supervisor import workers

    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "PENDING", [])

    entered = threading.Event()
    release = threading.Event()
    answers: dict = {}

    def _holder():
        with lease.reserved_folder_lane("/w/alpha"):
            entered.set()
            # From INSIDE its own reservation the holder must not be refused by its
            # own claim, or merge-back could never run at all.
            answers["own"] = branching.project_is_busy("alpha", "/w/alpha")
            release.wait(3)

    worker = threading.Thread(target=_holder, daemon=True)
    worker.start()
    assert entered.wait(3)
    try:
        assert lease.running_project_lanes([]) == {("", os.path.normcase("/w/alpha"))}
        # ANOTHER gesture asking about the same folder is told BUSY.
        assert branching.project_is_busy("alpha", "/w/alpha") is True
        # A different folder is untouched.
        assert branching.project_is_busy("alpha", "/w/beta") is False
    finally:
        release.set()
        worker.join(3)

    assert answers["own"] is False
    # Released in a `finally`: the folder is schedulable again.
    assert branching.project_is_busy("alpha", "/w/alpha") is False


def test_removing_a_checkout_sees_a_merge_holding_the_project_folder(
    drive, folder, wt_root, monkeypatch,
):
    """I5, applied to the gesture the reviewer named: removal asked only about the
    CHECKOUT, so a merge-back holding the PROJECT folder — rewriting the same
    repository's worktree metadata and reading the very `thread/<name>` branch this
    would delete — read as idle."""
    import ouroboros.project_lease as lease
    from ouroboros.thread_worktrees import remove_thread_worktree
    from supervisor import workers

    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "PENDING", [])
    thread = _project(drive, folder)
    out = branch_off_thread(drive, "racer", thread["id"], data_dir=drive, worktree_root=wt_root)
    assert out["ok"] is True

    import threading

    entered, release = threading.Event(), threading.Event()

    def _merger():
        with lease.reserved_folder_lane(folder):
            entered.set()
            release.wait(3)

    worker = threading.Thread(target=_merger, daemon=True)
    worker.start()
    assert entered.wait(3)
    try:
        refused = remove_thread_worktree(
            data_dir=drive, project_id="racer", thread_id=thread["id"],
            worktree_root=wt_root,
        )
    finally:
        release.set()
        worker.join(3)

    assert refused["removed"] is False
    assert refused["reason"] == "project_busy"
    assert pathlib.Path(out["path"]).is_dir()


# --------------------------------------------------------------------------- #
# I6 — BRANCH OFF's snapshot commit is a WRITE and needs the same guard
# --------------------------------------------------------------------------- #

def test_the_snapshot_base_refuses_while_a_task_writes_in_the_folder(
    drive, folder, wt_root, monkeypatch,
):
    """I6. `merge_back_thread` guards its write twice — it holds the folder's lane
    and asks `project_is_busy`. `branch_off_thread` with `base_ref="@snapshot"`
    runs `git add -A` + commit in the SAME folder and did neither, so a live
    task's half-written scratch file became a commit on the owner's branch while
    `project_is_busy` was answering True for that exact folder one line earlier.
    """
    from supervisor import workers

    thread = _project(drive, folder)
    head_before = _git(folder, "rev-parse", "HEAD").stdout.strip()
    (folder / "half_written.tmp").write_text("a task is mid-write\n", encoding="utf-8")

    monkeypatch.setitem(
        workers.RUNNING, "t-live",
        {"task": {"id": "t-live", "project_id": "racer", "workspace_root": str(folder)}},
    )
    try:
        refused = branch_off_thread(
            drive, "racer", thread["id"], base_ref=BASE_SNAPSHOT,
            data_dir=drive, worktree_root=wt_root,
        )
        assert refused["ok"] is False
        assert refused["reason"] == REASON_PROJECT_BUSY
        assert "half-written" in refused["message"]
        # NOTHING was committed and the owner's index is untouched.
        assert _git(folder, "rev-parse", "HEAD").stdout.strip() == head_before
        status = _git(folder, "status", "--porcelain").stdout
        assert "?? half_written.tmp" in status

        # Every OTHER base reads a commit-ish and writes nothing to the project
        # folder, so it must keep working while that task runs.
        ok = branch_off_thread(
            drive, "racer", thread["id"], base_ref="main",
            data_dir=drive, worktree_root=wt_root,
        )
        assert ok["ok"] is True, ok
    finally:
        workers.RUNNING.pop("t-live", None)


# --------------------------------------------------------------------------- #
# I7 — a branched thread's CHAT lane must read the folder its TASKS write
# --------------------------------------------------------------------------- #

def test_the_chat_lens_follows_the_room_into_its_own_checkout(drive, folder, wt_root):
    """I7. `room_chat_lens_dir` took no room chat id and answered
    `project.working_dir` unconditionally, so for a branched thread the TASK
    workspace was the checkout while the CHAT lens was the project folder — the
    same fact/affordance split `thread_checkout_for_room`'s own docstring calls
    "the robot-room incident".
    """
    from ouroboros.projects_registry import get_thread
    from ouroboros.workspace_admission import resolve_room_workspace, room_chat_lens_dir

    thread = _project(drive, folder)
    out = branch_off_thread(drive, "racer", thread["id"], data_dir=drive, worktree_root=wt_root)
    assert out["ok"] is True
    thread_chat = int(thread["chat_id"])
    zero_chat = int(get_thread(drive, "racer", 0)["chat_id"])

    lens, note = room_chat_lens_dir(drive, "racer", thread_chat)
    assert note == ""
    assert lens == str(pathlib.Path(out["path"]).resolve())
    # ...and it is the SAME folder admission gives that room's tasks.
    task_root, error, _decision = resolve_room_workspace(
        drive_root=drive, system_repo_dir=pathlib.Path(__file__).resolve().parents[1],
        project_id="racer", room_chat_id=thread_chat,
    )
    assert error == ""
    assert pathlib.Path(task_root).resolve() == pathlib.Path(lens).resolve()

    # Thread #0 keeps the project folder — its siblings are not moved (A7/I8).
    assert room_chat_lens_dir(drive, "racer", zero_chat)[0] == str(folder.resolve())


def test_the_room_fact_names_the_folder_it_actually_resolved(drive, folder, wt_root, monkeypatch):
    """I7's doc-truth half: `context.py` stated to the MODEL as fact that "the
    promoted task inherits this folder as its workspace" while naming
    `working_dir` — false for a branched thread, whose promoted tasks land in its
    checkout instead."""
    import ouroboros.config as config
    from ouroboros.context import build_runtime_section

    thread = _project(drive, folder)
    out = branch_off_thread(drive, "racer", thread["id"], data_dir=drive, worktree_root=wt_root)
    assert out["ok"] is True
    (drive / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATA_DIR", drive)

    env = SimpleNamespace(
        repo_dir=pathlib.Path(__file__).resolve().parents[1],
        drive_root=drive,
        drive_path=lambda rel: drive / rel,
    )
    rendered = build_runtime_section(env, {
        "id": "t1", "project_id": "racer", "_is_direct_chat": True,
        "chat_id": int(thread["chat_id"]),
    })

    assert "room_dir" in rendered
    assert str(pathlib.Path(out["path"]).resolve()) in rendered
    assert "inherits room_dir as its workspace" in rendered


def test_every_room_chat_lens_dir_call_site_names_the_room(tmp_path):
    """A guard against the NEXT lens caller forgetting the room, in the same shape
    the `resolve_room_workspace` guard already has.

    `room_chat_id` has a DEFAULT — the resolver is also asked project-wide
    questions — so a call site that omits it compiles, passes every behavioural
    test it writes, and silently reintroduces I7. The fact under test is "no call
    site forgets", which no single behavioural test can express.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    call = re.compile(r"room_chat_lens_dir\s*\(", re.MULTILINE)
    offenders: list[str] = []
    inspected = 0
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if rel.parts[0] in {"tests", ".git", "build", "dist"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in call.finditer(text):
            depth, index = 0, match.end() - 1
            while index < len(text):
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            if "def room_chat_lens_dir" in text[max(0, match.start() - 40):match.start()]:
                continue
            args = text[match.end():index]
            inspected += 1
            # Three arguments, or the keyword: either spelling names the room.
            top_level = 0
            commas = 0
            for char in args:
                if char in "([{":
                    top_level += 1
                elif char in ")]}":
                    top_level -= 1
                elif char == "," and top_level == 0:
                    commas += 1
            if "room_chat_id" not in args and commas < 2:
                offenders.append(f"{rel}:{text[:match.start()].count(chr(10)) + 1}")
    assert inspected >= 2, (
        "the scan found no lens call sites — the pattern went stale and this guard "
        "is now asserting nothing"
    )
    assert not offenders, (
        "these room_chat_lens_dir call sites do not name the room, so a branched "
        "thread's chat reads the project folder while its tasks write its "
        f"checkout (I7): {offenders}"
    )


# --------------------------------------------------------------------------- #
# I9 / I17 — the removal route's refusals
# --------------------------------------------------------------------------- #

def test_the_unmerged_work_refusal_declares_that_it_can_be_answered(
    drive, folder, wt_root, monkeypatch,
):
    """I9. `acknowledge_unmerged` IS the answer to `unmerged_work`, and the payload
    never declared `acknowledgeable` — so a client reading that field (the shared
    envelope's own name for it) bailed on the one refusal whose sentence ends "or
    confirm you want it gone"."""
    monkeypatch.setenv("OUROBOROS_THREAD_WORKTREE_ROOT", str(wt_root))
    from ouroboros.gateway.project_threads import api_thread_worktree_remove

    thread = _project(drive, folder)
    out = branch_off_thread(drive, "racer", thread["id"], data_dir=drive, worktree_root=wt_root)
    assert out["ok"] is True
    _commit_in(out["path"], "kept.txt", "unmerged\n")

    async def _json():
        return {}

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=drive)),
        path_params={"project_id": "racer", "thread_id": str(thread["id"])},
        json=_json,
    )
    response = asyncio.run(api_thread_worktree_remove(request))

    assert response.status_code == 409
    body = json.loads(response.body)
    assert body["reason"] == "unmerged_work"
    assert body["acknowledgeable"] is True
    assert "confirm you want it gone" in body["message"]


def test_removal_failed_gets_a_sentence_instead_of_something_went_wrong():
    """I17. It was the ONE refusal with no copy: the cause sat in the log and
    `branch_kept_reason` carried a usable sentence `describeOutcome` never reads on
    a refusal."""
    from ouroboros.gateway.project_threads import _removal_message

    message = _removal_message(
        "removal_failed",
        {"error": "the checkout is not on disk: /w/gone"},
        {"branch_kept_reason": "the checkout survived removal, so its branch still points at it"},
    )

    assert "still on disk" in message
    assert "/w/gone" in message
    assert "safe to retry" in message
    assert "its branch still points at it" in message
    assert message != "The checkout could not be removed."


# --------------------------------------------------------------------------- #
# I11 / I12 / I1's copy — client facts that live in the DOM half
# --------------------------------------------------------------------------- #

def _web(rel: str) -> str:
    return (pathlib.Path(__file__).resolve().parents[1] / "web" / rel).read_text(encoding="utf-8")


def test_the_sidebar_fingerprint_covers_the_fields_the_paint_reads():
    """I11. The per-thread tuple omitted exactly the two fields the merge added
    consumers for, so a rewritten `delete_error` never reached the `Retry delete`
    row and an `active -> deleting` transition from another tab left this tab
    painting an ordinary full-menu row. Reproduced over a real `projects_summary`:
    both edits left the fingerprint byte-identical.
    """
    app = _web("app.js")
    tuple_src = app[app.index("const json = JSON.stringify(rows.map(p => ["):][:400]
    assert "t.lifecycle" in tuple_src and "t.delete_error" in tuple_src, tuple_src


def test_a_thread_on_its_way_out_is_not_a_thread_to_open():
    """I12. `threadRowPresentation` states the rule and implemented it for
    `draggable`/`showsUnread` only: the row kept its click handler, `openThread`
    guards only the PROJECT lifecycle, and the admission fence then answered a
    typed refusal the chat rendered as "Project is unavailable" — while
    `server.py`'s own comment says a thread can be fenced inside a perfectly
    healthy project.
    """
    threads = _web("modules/project_threads.js")
    assert "row.disabled = !paint.draggable;" in threads
    chat = _web("modules/chat.js")
    assert "return 'This room is no longer available'" in chat
    assert "'Project is unavailable'" not in chat
    # ...and the centre stage closes when the open thread leaves the projection.
    app = _web("app.js")
    after = app[app.index("if (openThreadRow) threadStage.setTitle(active, openThreadRow);"):][:600]
    assert "closeProjectPanel();" in after


def test_the_project_delete_copy_mentions_the_checkouts_it_removes():
    """I1's client half: the confirm dialog promised "its id, chat history, task
    bindings, memory, and working folder are preserved" with no mention of the N
    checkouts and N `thread/*` branches the gesture destroys."""
    menu = _web("modules/project_create.js")
    body = menu[menu.index("title: 'Delete project'"):][:900]
    assert "checkouts its threads branched off" in body
    assert "thread/" in body
    assert "the delete stops" in body


def test_an_unresolvable_lane_is_not_pinned_so_the_outage_does_not_outlive_itself():
    """I3, probed adversarially against the fix itself.

    Pinning ``(pid, "")`` because the registry was unreadable at that instant would
    survive the outage: once the map reads again the CANDIDATE check has a real
    answer and stops applying the conservative rule, while the holder still carries
    the narrow key — so a folder-bearing candidate matches nothing and becomes a
    second writer in the very folder the holder is in. Left unpinned, occupancy
    resolves from the record the moment the registry can be read.
    """
    from ouroboros.project_lease import (
        LANE_PIN_FIELD,
        candidate_is_leasable,
        pin_task_lane,
        running_project_lanes,
    )

    placeless = {"id": "t1", "type": "task", "project_id": "alpha"}
    assert pin_task_lane(placeless, None) is None
    assert LANE_PIN_FIELD not in placeless

    # The registry is readable again on the NEXT pass.
    folders = {"alpha": "/w/alpha"}
    lanes = running_project_lanes([{"task": placeless}], folders)
    assert lanes == {("", os.path.normcase("/w/alpha"))}
    assert candidate_is_leasable(
        {"id": "t2", "project_id": "alpha", "workspace_root": "/w/alpha"}, lanes, folders,
    ) is False
    # A real map — even an empty one — still pins, because it is an ANSWER.
    assert pin_task_lane(dict(placeless), {}) == ("alpha", "")
