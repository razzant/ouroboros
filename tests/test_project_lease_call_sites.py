"""The writer lane's five CALL SITES, asserted through the folder map (I2).

The merge's central cross-stream fix — "the defect neither stream could see" — is
that ``pin_task_lane``/``running_project_lanes``/``candidate_is_leasable`` must be
given the SAME ``project_workspaces`` map the admission check used. A task admitted
as ``("", registered_folder)`` and pinned as ``(pid, "")`` reads the folder as
unheld to the very next candidate: the pin, whose whole purpose is to stop a live
writer drifting out of its lane, becomes the thing that drifts it.

All five sites pass the map, and NONE of them had a test. The two that looked like
guards structurally could not see it: ``test_project_lease.py`` pins the
FUNCTION's contract by calling ``pin_task_lane`` itself, and
``test_project_lease_ui_conversion.py``'s lane assertion had the map on NEITHER
side, so both sides reduced to ``(pid, "")`` and it returned the same answer
either way. Five mutations dropping the map survived the entire relevant suite.

So every test here goes through the PRODUCTION entry point and asserts the value
the site is obliged to produce. The mutation each one kills is named in its
docstring, because that is the fact under test — not "the lane works", but "this
call site cannot be reverted with a green suite".
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.project_lease import LANE_PIN_FIELD, normalize_workspace_root
from ouroboros.projects_registry import create_project


@pytest.fixture()
def folder(tmp_path):
    root = tmp_path / "alpha_folder"
    root.mkdir()
    return root


def _running(task_id="tlive", project_id=""):
    """A live task that names NO folder of its own — the only shape the map matters
    for. A task carrying its own ``workspace_root`` resolves without the map, so a
    test built on one cannot see the defect at all."""
    return {"task": {"id": task_id, "type": "task", "project_id": project_id}}


# --------------------------------------------------------------------------- #
# Site 1 — supervisor/workers.py, assign_tasks -> pin_task_lane
# --------------------------------------------------------------------------- #

def test_assign_tasks_pins_the_registered_folder_not_the_project(tmp_path, monkeypatch, folder):
    """MUTATION KILLED: ``pin_task_lane(running_record)`` in ``assign_tasks``.

    This is the site the pin exists for. A placeless project-scoped candidate is
    ADMITTED as ``("", registered_folder)`` two loops earlier — the assignment pass
    reads the map for exactly that — so pinning without it freezes ``(pid, "")``
    and the folder the task is now writing in reads as free.
    """
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path, 600, 1800)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    workers.WORKERS.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    queue.init_queue_refs(workers.PENDING, workers.RUNNING, workers.QUEUE_SEQ_COUNTER_REF)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})

    create_project(tmp_path, "alpha", name="Alpha", working_dir=str(folder))
    workers.WORKERS[0] = SimpleNamespace(
        wid=0, busy_task_id=None, reaping=False,
        in_q=SimpleNamespace(put=lambda task: None),
    )
    # Placeless: only the promote/room path stamps `workspace_root`, and a task
    # scoped post-hoc carries the project id alone.
    workers.PENDING.append({"id": "t1", "type": "task", "chat_id": 0, "project_id": "alpha"})

    workers.assign_tasks()

    assert "t1" in workers.RUNNING, "the candidate was not assigned; the test premise is gone"
    assert workers.RUNNING["t1"]["task"][LANE_PIN_FIELD] == [
        "", normalize_workspace_root(folder)
    ]
    workers.RUNNING.clear()
    workers.WORKERS.clear()
    workers.PENDING[:] = []


# --------------------------------------------------------------------------- #
# Site 2 — supervisor/workers.py, ensure_project_scope -> mark_task_project
# --------------------------------------------------------------------------- #

def test_ensure_project_scope_pins_the_registered_folder(tmp_path, monkeypatch, folder):
    """MUTATION KILLED: ``mark_task_project(running, pending, tid, pid)`` in
    ``ensure_project_scope``.

    The mid-flight self-scope is the live case the write-once pin admits: the task
    ACQUIRES a lane it did not hold. Marked without the map it freezes ``(pid, "")``
    while every later candidate for the same project resolves to
    ``("", folder)`` and is admitted into that folder alongside it.
    """
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    create_project(tmp_path, "alpha", name="Alpha", working_dir=str(folder))
    ctx = SimpleNamespace(RUNNING={"t1": _running("t1")}, PENDING=[])

    workers.ensure_project_scope(
        {"task_id": "t1", "project_id": "alpha", "project_name": "Alpha"}, ctx,
    )

    assert ctx.RUNNING["t1"]["task"]["project_id"] == "alpha"
    assert ctx.RUNNING["t1"]["task"][LANE_PIN_FIELD] == [
        "", normalize_workspace_root(folder)
    ]


# --------------------------------------------------------------------------- #
# Site 3 — ouroboros/gateway/projects.py, api_project_from_task
# --------------------------------------------------------------------------- #

def test_ui_conversion_pins_the_projects_registered_folder(tmp_path, monkeypatch, folder):
    """MUTATION KILLED: ``mark_task_project(RUNNING, PENDING, task_id, pid)`` in
    ``api_project_from_task``.

    The sibling assertion in ``test_project_lease_ui_conversion.py`` reads the lane
    with the map on NEITHER side, so both reduce to ``(pid, "")`` and it returns
    the same answer with the fix and without it — it looks like it guards the lane
    and structurally cannot see it. This one asserts the pinned VALUE.
    """
    import supervisor.queue as queue
    import supervisor.workers as workers
    from ouroboros.gateway.projects import api_project_from_task

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
    snap = tmp_path / "state" / "queue_snapshot.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", snap)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)

    # `create_project` is idempotent by design and returns the existing row, so a
    # conversion that names an existing project adopts its folder. That is what
    # makes a PLACELESS running task meet a project that HAS a folder — the exact
    # pairing the map exists for.
    create_project(tmp_path, "task-tlive", name="Alpha", working_dir=str(folder))
    monkeypatch.setitem(workers.RUNNING, "tlive", _running("tlive"))

    async def _json():
        return {"task_id": "tlive", "id": "task-tlive", "objective_hint": "build it"}

    response = asyncio.run(api_project_from_task(SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)), json=_json,
    )))
    body = json.loads(response.body)
    assert body["project"]["id"] == "task-tlive"
    assert body["project"]["working_dir"] == str(folder.resolve())

    assert workers.RUNNING["tlive"]["task"][LANE_PIN_FIELD] == [
        "", normalize_workspace_root(folder)
    ]
    workers.RUNNING.pop("tlive", None)


# --------------------------------------------------------------------------- #
# Site 4 — ouroboros/thread_branching.py, queue_notice
# --------------------------------------------------------------------------- #

def test_the_queue_notice_reads_the_same_map_the_scheduler_does(tmp_path, monkeypatch):
    """MUTATION KILLED: ``running_project_lanes(running)`` /
    ``candidate_is_leasable(candidate, lanes)`` in ``queue_notice``.

    A14's honesty runs in BOTH directions. Here a placeless task holds the project
    FOLDER and the thread asking has branched off into its own checkout, so the
    truthful answer is "nothing waits". Read without the map the holder keys on
    ``(project_id, "")``, the folder is unresolvable, and the notice warns the owner
    about a wait that does not exist — one line above a Branch off… offer that
    already happened.
    """
    import subprocess

    import ouroboros.thread_branching as branching
    from ouroboros.project_threads_registry import create_thread
    from ouroboros.thread_worktrees import provision_thread_worktree
    from supervisor import workers

    repo = tmp_path / "owner_folder"
    repo.mkdir()
    for args in (
        ("init", "-b", "main"), ("config", "user.email", "t@e.com"), ("config", "user.name", "T"),
    ):
        subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)
    (repo / "app.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=str(repo), check=True, capture_output=True)

    drive = tmp_path / "drive"
    create_project(drive, "racer", name="Racer", working_dir=str(repo))
    thread = create_thread(drive, "racer", name="Side quest")
    provision_thread_worktree(
        repo_dir=repo, project_id="racer", thread_id=int(thread["id"]),
        data_dir=drive, worktree_root=tmp_path / "wt",
    )

    monkeypatch.setattr(workers, "PENDING", [])
    holder = {"id": "t1", "type": "task", "project_id": "racer"}   # placeless
    monkeypatch.setattr(workers, "RUNNING", {"t1": {"task": holder}})

    notice = branching.queue_notice(drive, "racer", int(thread["id"]), data_dir=drive)

    assert notice["queued"] is False, (
        "the branched thread was told it would wait behind a task in a folder it "
        "does not share — the holder's lane was read without the folder map (I2)"
    )
    # ...and the SAME read still warns the thread that really does share the folder.
    zero = branching.queue_notice(drive, "racer", 0, data_dir=drive)
    assert zero["queued"] is True
    assert zero["remedy"] == "branch_off"


# --------------------------------------------------------------------------- #
# Site 5 — supervisor/events.py, _project_lane_wait_suffix
# --------------------------------------------------------------------------- #

def test_the_scheduled_task_notice_reads_the_same_map_the_scheduler_does(tmp_path, folder):
    """MUTATION KILLED: ``running_project_lanes(rows)`` /
    ``candidate_is_leasable(task, lanes)`` in ``_project_lane_wait_suffix``.

    Same fact, other surface: the sentence a scheduled task carries. The holder is
    placeless in the project whose registered folder is ``folder``; the scheduled
    task names a DIFFERENT folder, so it does not wait. Read without the map the
    holder's folder is unknown and the owner is told their task is queued behind
    something that is not in their way — the false warning this function's own
    docstring forbids.
    """
    from supervisor.events import _project_lane_wait_suffix

    create_project(tmp_path, "alpha", name="Alpha", working_dir=str(folder))
    holder = {"t1": {"task": {"id": "t1", "type": "task", "project_id": "alpha"}}}
    elsewhere = {
        "id": "t2", "type": "task", "project_id": "alpha",
        "workspace_root": str(tmp_path / "some_other_checkout"),
    }

    assert _project_lane_wait_suffix(elsewhere, holder, tmp_path) == "", (
        "the scheduled task was warned about a folder it does not share — the "
        "holder's lane was read without the folder map (I2)"
    )
    # ...and a task that DOES name the project's registered folder is still warned.
    same = {"id": "t3", "type": "task", "project_id": "alpha", "workspace_root": str(folder)}
    assert _project_lane_wait_suffix(same, holder, tmp_path) != ""


# --------------------------------------------------------------------------- #
# The map argument itself: a source guard, because the default is silent
# --------------------------------------------------------------------------- #

def test_every_lane_call_site_hands_over_the_folder_map():
    """A guard against the SIXTH call site, in the shape the room-workspace guard
    already has.

    ``project_workspaces`` has a default, so a site that omits it compiles and
    passes its own behavioural tests. Since I3 that default means "the folders are
    unknown" and behaves conservatively, which is safe but wrong: the site would
    queue work that should run and warn about waits that do not exist. Reading the
    SOURCE is the only way to express "no site forgets".
    """
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    watched = ("pin_task_lane", "running_project_lanes", "candidate_is_leasable", "mark_task_project")
    call = re.compile(rf"\b({'|'.join(watched)})\s*\(", re.MULTILINE)
    offenders: list[str] = []
    inspected = 0
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if rel.parts[0] in {"tests", ".git", "build", "dist"} or rel.name == "project_lease.py":
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
            args = text[match.end():index]
            if "import" in text[max(0, match.start() - 200):match.start()].rsplit("\n", 1)[-1]:
                continue
            inspected += 1
            # Any of the four spellings of "the map went in": the local name the
            # supervisor reads it into, the registry function itself, or the map
            # variable the branching/notice paths build.
            if not any(
                token in args
                for token in ("_project_workspaces", "project_working_dirs", "folders", "project_workspaces")
            ):
                offenders.append(f"{rel}:{text[:match.start()].count(chr(10)) + 1}: {match.group(1)}")
    assert inspected >= 5, (
        f"the scan found only {inspected} lane call sites — the pattern went stale "
        "and this guard is now asserting nothing"
    )
    assert not offenders, (
        "these writer-lane call sites do not hand over the project->folder map, so "
        "the lane they compute is not the lane the scheduler compares against "
        f"(I2): {offenders}"
    )


def test_normalize_workspace_root_is_the_one_spelling_the_pin_uses(folder):
    """The assertions above compare against `normalize_workspace_root`, so that
    equality is the thing they actually pin. Kept explicit: a pin stored in some
    other spelling would make every one of them pass while the lane still split."""
    assert normalize_workspace_root(folder) == os.path.normcase(os.path.normpath(str(folder))).casefold() \
        or normalize_workspace_root(folder) == os.path.normcase(os.path.normpath(str(folder)))
