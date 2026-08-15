"""The WHOLE road a remote task travels: queue -> supervisor -> REAL worker -> target.

Every half of this road already had tests, and the defect lived in the JOIN — the
same shape as `test_registry_remote_dispatch.py`, one layer out. The existing
suites ran the broker, the dispatch and the "worker" in ONE process and called
`broker.admit_workspace(task_id=...)` themselves in a fixture, which is precisely
the wire production never laid. So they were green while a real remote project
could not execute a single tool: the worker arrived at the broker with a task id no
session had ever been bound to and every tool answered
``REMOTE_EXECUTION_UNAVAILABLE: task_session_unbound``.

What this file drives, with nothing about the target simulated:

* a sealed placement produced by the real admission door
  (``workspace_admission.admit_remote_placement``) against a real execd over real
  OpenSSH in Docker;
* the real supervisor assignment step (``supervisor.workers.assign_tasks``), which
  is where the session binding is laid;
* a REAL separate worker process holding nothing but a ``multiprocessing`` Pipe
  proxy — the production handle — driving the real ``ToolRegistry`` dispatch;
* the broker in this process, answering over that pipe and talking to the target.

Two cases, because the two hide behind each other:

* **RWSB-01** ``queued_remote_task_executes_through_a_real_worker_process`` — the
  live-QA scenario exactly: project admitted, task queued, worker runs a tool.
* **RWSB-02** ``assignment_rebinds_after_the_project_session_is_gone`` — the same
  road when the broker holds NO session for the project (what a restart leaves
  behind). Plumbing a project id down to the broker would make RWSB-01 pass and
  this one still hang on a typed refusal forever; only a binding laid at
  assignment re-admits.
* **RWSB-03** ``subagent_task_shares_the_parent_project_session`` — two tasks of
  one project hold two bindings on ONE session, not two sessions.
* **RWSB-04** ``the_root_matrix_holds_over_the_real_wire`` — a Home-native root on a
  live remote task is answered by HOME, compared by BYTES against a same-named file on
  the target, because a wrong-file read returns plausible content and no label to doubt.
* **RWSB-05** ``a_clean_target_worktree_is_an_empty_result_not_a_remote_failure`` — the
  clean/dirty pair of `vcs_status`/`vcs_diff` beside the `run_command` spelling of the
  same fact, which is how the two placements were caught disagreeing about one worktree.

The last two both come from a live server rather than from a test, which is the reason
they are here and not only on the fake wire.

Serial + explicitly gated: it builds a container, spawns real processes and
mutates the supervisor's module-global queue.
"""

from __future__ import annotations

import multiprocessing
import os
import pathlib
import queue as queue_mod
import uuid
from typing import Any

import pytest

# The Docker/OpenSSH fixture is REUSED rather than re-implemented: a second copy of
# an sshd container recipe is a second definition of what "the target" is.
from tests.test_remote_workspace_ssh import (  # noqa: F401 -- fixture import
    _REMOTE_WORKSPACE,
    _broker_capability_manifest,
    docker_ssh_host,
)

pytestmark = [
    pytest.mark.serial,
    pytest.mark.skipif(
        os.environ.get("OUROBOROS_RUN_REMOTE_SSH_TESTS") != "1",
        reason=(
            "real Docker/OpenSSH lane; set OUROBOROS_RUN_REMOTE_SSH_TESTS=1 "
            "to run explicitly"
        ),
    ),
]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PROJECT_ID = "project-real"


def _remote_tool_child(
    proxy: Any,
    task: dict,
    repo_dir: str,
    drive_root: str,
    result_q: Any,
    calls: Any = None,
) -> None:
    """A REAL worker process: a pipe proxy, a task dict, and the real dispatch.

    Deliberately a faithful slice of `worker_main` rather than a call into it: the
    agent loop needs a model provider, and what is under test is the placement
    road, not the LLM. Everything the road touches is production code — the proxy
    is the one the supervisor mints, the context is built from the task's SEALED
    metadata, and the tool goes through `ToolRegistry.execute`.

    ``calls`` is a sequence of ``(tool, args)`` run in order through ONE dispatch
    context, because the facts that matter for the root matrix are comparative: the
    same task reading a Home-native root and the target-native one, and the same
    worktree answered clean and then dirty. Two separate worker processes would be
    two contexts and could not compare them.
    """

    try:
        from ouroboros.remote_workspace import set_remote_workspace_service

        set_remote_workspace_service(proxy)
        import ouroboros.safety as safety

        safety.check_safety = lambda *a, **k: (True, None)
        from ouroboros.tools.registry import ToolContext, ToolRegistry

        repo = pathlib.Path(repo_dir)
        drive = pathlib.Path(drive_root)
        ctx = ToolContext(
            repo_dir=repo,
            drive_root=drive,
            task_id=str(task.get("id") or ""),
            workspace_root=str(task.get("workspace_root") or ""),
            workspace_mode="external",
            project_id=str(task.get("project_id") or ""),
        )
        ctx.task_metadata.update(dict(task.get("metadata") or {}))
        registry = ToolRegistry(repo_dir=repo, drive_root=drive)
        registry.set_context(ctx)
        plan = list(calls or [("list_files", {"root": "active_workspace", "path": "."})])
        texts = [registry.execute(tool, dict(args)) for tool, args in plan]
        result_q.put({"ok": True, "text": texts[0], "texts": texts})
    except BaseException as exc:  # noqa: BLE001 -- the child's only channel is the queue
        result_q.put({"ok": False, "text": f"{type(exc).__name__}: {exc}", "texts": []})


class _StubWorkerProc:
    """The supervisor's Worker slot without a real child: assignment only reads
    `pid`/`is_alive` and puts the task on `in_q`, and the REAL process is spawned
    by the test from that dispatched task."""

    pid = None

    def is_alive(self) -> bool:
        return False


@pytest.fixture()
def lane(docker_ssh_host, tmp_path, monkeypatch):  # noqa: F811
    """A live broker + owner connection store + an initialized supervisor queue."""

    from ouroboros import config
    from ouroboros.connection_store import add_connection
    from ouroboros.remote_workspace import (
        RemoteSessionBroker,
        set_remote_workspace_service,
    )
    from supervisor import state as sup_state, workers as sup_workers

    drive = tmp_path / "drive"
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    (drive / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        config, "REMOTE_CONNECTIONS_PATH", drive / "state" / "remote_connections.json"
    )
    # The owner's store is the ONLY place the binding may learn a connection from —
    # the ssh alias is the container's, so this row really does reach the target.
    row = add_connection(name="Real SSH", ssh_alias="ouroboros-real-test")

    generation = f"bind-generation-{uuid.uuid4().hex}"
    broker = RemoteSessionBroker(
        drive,
        generation,
        _broker_capability_manifest(),
        bundle_dir=docker_ssh_host.bundle_dir,
        ssh_binary=str(docker_ssh_host.ssh_wrapper),
    )
    broker.start()
    set_remote_workspace_service(broker)

    sup_state.init(drive, 1000.0)
    sup_state.init_state()
    sup_workers.init(
        repo_dir=_REPO_ROOT,
        drive_root=drive,
        max_workers=2,
        soft_timeout=600,
        hard_timeout=1800,
        total_budget_limit=1000.0,
    )
    sup_workers.PENDING[:] = []
    sup_workers.RUNNING.clear()
    sup_workers.WORKERS.clear()
    for wid in (0, 1):
        sup_workers.WORKERS[wid] = sup_workers.Worker(
            wid=wid, proc=_StubWorkerProc(), in_q=queue_mod.Queue()
        )
    try:
        yield {
            "broker": broker,
            "connection_id": str(row["id"]),
            "drive": drive,
            "workers": sup_workers,
        }
    finally:
        sup_workers.PENDING[:] = []
        sup_workers.RUNNING.clear()
        sup_workers.WORKERS.clear()
        set_remote_workspace_service(None)
        broker.close(timeout_sec=5)


def _sealed_task(ref: Any, drive: pathlib.Path, *, task_id: str, **extra: Any) -> dict:
    """A task shaped exactly as `/api/tasks` seals a remote one."""

    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY

    task = {
        "id": task_id,
        "type": "task",
        "text": "list the target worktree",
        "description": "list the target worktree",
        "project_id": _PROJECT_ID,
        "workspace_root": ref.remote_root,
        "workspace_mode": "external",
        "memory_mode": "forked",
        "drive_root": str(drive),
        "metadata": {SEALED_WORKSPACE_REF_KEY: ref.to_payload()},
    }
    task.update(extra)
    return task


def _admit_project_placement(connection_id: str) -> Any:
    """The real project-creation door: the target allocates the workspace identity."""

    from ouroboros.workspace_admission import admit_remote_placement

    return admit_remote_placement(
        connection_id=connection_id,
        remote_root=_REMOTE_WORKSPACE,
        project_id=_PROJECT_ID,
    )


def _assign_and_run(lane: dict, task: dict, calls: Any = None) -> dict:
    """Assign through the real supervisor step, then run the tool in a REAL process."""

    workers = lane["workers"]
    workers.PENDING[:] = [task]
    workers.assign_tasks()
    dispatched = None
    for worker in workers.WORKERS.values():
        try:
            dispatched = worker.in_q.get_nowait()
        except queue_mod.Empty:
            continue
        break
    assert dispatched is not None, (
        "the supervisor never dispatched the remote task; PENDING="
        f"{[t.get('id') for t in workers.PENDING]}"
    )
    proxy = lane["broker"].create_worker_pipe_proxy("worker:0")
    mp_ctx = multiprocessing.get_context("spawn")
    result_q = mp_ctx.Queue()
    proc = mp_ctx.Process(
        target=_remote_tool_child,
        args=(proxy, dispatched, str(_REPO_ROOT), str(lane["drive"]), result_q, calls),
    )
    proc.start()
    proxy.close_parent_copy()
    try:
        outcome = result_q.get(timeout=180)
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)
    return outcome


def _assert_reached_the_target(outcome: dict) -> None:
    text = str(outcome.get("text") or "")
    assert outcome.get("ok"), text
    for marker in (
        "REMOTE_EXECUTION_UNAVAILABLE",
        "task_session_unbound",
        "task_session_mismatch",
        "remote_session_disconnected",
        "SSH_EXECUTOR_UNAVAILABLE",
        "TOOL_ERROR",
    ):
        assert marker not in text, text
    assert "README.md" in text, text


def test_rwsb_01_queued_remote_task_executes_through_a_real_worker_process(lane):
    ref = _admit_project_placement(lane["connection_id"])
    task = _sealed_task(ref, lane["drive"], task_id=f"task-{uuid.uuid4().hex[:12]}")
    _assert_reached_the_target(_assign_and_run(lane, task))
    # The binding is the task's own, recorded against THIS generation.
    state = task["_remote_session_bind"]
    assert state["status"] == "bound"
    assert state["server_generation"] == lane["broker"].server_generation
    rows = lane["broker"].status()["connections"]
    assert [row["active_task_count"] for row in rows] == [1], rows


def test_rwsb_02_assignment_rebinds_after_the_project_session_is_gone(lane):
    ref = _admit_project_placement(lane["connection_id"])
    # What a server restart leaves behind: a sealed placement on disk and a broker
    # holding no session for it. Nothing on the queue road re-opens one unless the
    # assignment step admits.
    assert lane["broker"].close_project_session(ref.to_payload(), project_id=_PROJECT_ID)
    assert lane["broker"].status()["connections"] == []
    task = _sealed_task(ref, lane["drive"], task_id=f"task-{uuid.uuid4().hex[:12]}")
    _assert_reached_the_target(_assign_and_run(lane, task))


def test_rwsb_03_subagent_task_shares_the_parent_project_session(lane):
    ref = _admit_project_placement(lane["connection_id"])
    parent_id = f"task-{uuid.uuid4().hex[:12]}"
    parent = _sealed_task(ref, lane["drive"], task_id=parent_id)
    _assert_reached_the_target(_assign_and_run(lane, parent))
    child = _sealed_task(
        ref,
        lane["drive"],
        task_id=f"task-{uuid.uuid4().hex[:12]}",
        delegation_role="subagent",
        parent_task_id=parent_id,
        root_task_id=parent_id,
    )
    # The parent still occupies its worker slot, so the child needs the second one.
    lane["workers"].WORKERS[0].busy_task_id = parent_id
    _assert_reached_the_target(_assign_and_run(lane, child))
    rows = lane["broker"].status()["connections"]
    assert len(rows) == 1, rows
    assert rows[0]["active_task_count"] == 2, rows


def test_rwsb_04_the_root_matrix_holds_over_the_real_wire(lane):
    """RWSB-04: on a LIVE remote task, a Home-native root is answered by HOME.

    The defect this pins was found on a real server, not by a test: the dispatch asked
    only whether the TOOL has a native counterpart and never whether THIS CALL's root
    lives on the target, so `read_file(root='artifact_store')` was prepared and executed
    on the target — which does not model `root`, resolved the name in its own worktree,
    and answered with a different file under an `active_workspace:` label. Both hosts
    hold a `README.md` here on purpose: only the BYTES can tell which one answered, and
    a routing bug that returns plausible content is exactly the failure a label cannot
    reveal.
    """
    ref = _admit_project_placement(lane["connection_id"])
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    artifacts = lane["drive"] / "task_results" / "artifacts" / task_id
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "README.md").write_text("home-artifact-only\n", encoding="utf-8")

    task = _sealed_task(ref, lane["drive"], task_id=task_id)
    outcome = _assign_and_run(lane, task, calls=[
        ("read_file", {"root": "artifact_store", "path": "README.md"}),
        ("read_file", {"root": "active_workspace", "path": "README.md"}),
        ("write_file", {"root": "task_drive", "path": "scratch.txt", "content": "home scratch\n"}),
    ])
    assert outcome["ok"], outcome["text"]
    home, target, wrote = outcome["texts"]

    assert "home-artifact-only" in home, home
    assert "remote-only" not in home, home
    # …and the target-native root in the SAME task still answers with the target's bytes.
    assert "remote-only" in target, target
    assert "home-artifact-only" not in target, target
    # A Home-native WRITE lands on Home, so the task's scratch does not evaporate with
    # the session — and it is not on the target either.
    assert "⚠️" not in wrote, wrote
    scratch = lane["drive"] / "task_drives" / task_id / "scratch.txt"
    assert scratch.read_text(encoding="utf-8") == "home scratch\n"


def test_rwsb_05_a_clean_target_worktree_is_an_empty_result_not_a_remote_failure(lane):
    """RWSB-05: `vcs_status` on a clean remote worktree, end to end.

    Reproduced on a live server: `vcs_status` answered
    ``REMOTE_EXECUTION_FAILED: remote_result_empty`` while
    ``run_command(["git","status","--porcelain"])`` on the SAME target returned data —
    one clean worktree, two placements, two different stories. A clean porcelain emits
    no bytes, the target's envelope carried an empty `text`, and Home read "no text" as
    "no result". Both spellings are asked here, in one task, so they can be compared:
    the Home handler answers `''` for a clean tree, and one voice means the target's
    clean tree answers `''` too.
    """
    ref = _admit_project_placement(lane["connection_id"])
    task = _sealed_task(ref, lane["drive"], task_id=f"task-{uuid.uuid4().hex[:12]}")
    restore = ("run_command", {"cmd": ["git", "checkout", "--", "."]})
    outcome = _assign_and_run(lane, task, calls=[
        # Normalize to the committed tree first — the container is module-scoped, and what
        # is under test is the CLEAN answer, not which test ran before this one. `git
        # checkout` rather than a `write_file` of the original bytes: an exact restore,
        # and it does not meet the accidental-truncation shrink guard on the way.
        restore,
        ("vcs_status", {}),
        ("vcs_diff", {}),
        ("run_command", {"cmd": ["git", "status", "--porcelain"]}),
        # …then dirty the same tracked file, so "empty" is proven to be an ANSWER and not
        # a swallowed one.
        ("write_file", {
            "root": "active_workspace", "path": "README.md",
            "content": "remote-only\nplus one dirty line\n",
        }),
        ("vcs_status", {}),
        ("vcs_diff", {}),
        restore,
    ])
    assert outcome["ok"], outcome["text"]
    (
        normalized, clean_status, clean_diff, porcelain,
        dirtied, dirty_status, dirty_diff, restored,
    ) = outcome["texts"]

    assert "exit_code=0" in normalized, normalized
    assert clean_status == "", repr(clean_status)
    assert clean_diff == "", repr(clean_diff)
    # The shell spelling of the same fact on the same target: it worked before the fix
    # too, which is how the two placements were caught disagreeing.
    assert "exit_code=0" in porcelain, porcelain
    assert "README.md" not in porcelain, porcelain

    assert "⚠️" not in dirtied, dirtied
    assert "README.md" in dirty_status, dirty_status
    assert "README.md" in dirty_diff, dirty_diff
    for text in (clean_status, clean_diff, dirty_status, dirty_diff):
        assert "remote_result_empty" not in text, text
        assert "REMOTE_EXECUTION_FAILED" not in text, text
    # The worktree is handed back committed, so a later test in this module inherits the
    # fixture's state rather than this test's.
    assert "exit_code=0" in restored, restored
