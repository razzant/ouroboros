"""Off-loop workspace preparation of acting subagents (claim 9; PREP1-PREP4).

Real queue maps, real reservation/receipt code, real Git copies held at their
snapshot step (a long tree walk; there is no shared copy lock). Process-global
supervisor state is touched, so the module runs in the serial lane.
"""
from __future__ import annotations

import contextlib
import pathlib
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import store
from ouroboros import subagent_worktrees as trees
from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
from ouroboros.task_results import STATUS_REQUESTED, STATUS_SCHEDULED, load_task_result, write_task_result
from supervisor import queue, task_lifecycle
from supervisor import workspace_preparation as prep
from supervisor.events import dispatch_event

pytestmark = pytest.mark.serial

_WAIT_SEC = 30.0  # bounds only how long a broken run takes to report; passing runs never wait on it


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def make_repo(path, files):
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Fixture")
    git(path, "config", "user.email", "fixture@example.invalid")
    for name, content in files.items():
        (path / name).write_bytes(content)
    git(path, "add", "-A")
    git(path, "commit", "-qm", "fixture")
    return path


class SupervisorCtx:
    """The event context the supervisor loop hands to handlers, over the real queue maps."""

    def __init__(self, data, system):
        self.DRIVE_ROOT, self.REPO_DIR = data, system
        self.PENDING, self.RUNNING = queue.PENDING, queue.RUNNING
        self.WORKERS = {0: SimpleNamespace(busy_task_id=None)}
        self.bridge = SimpleNamespace(push_log=lambda _event: None)
        self.sent, self.snapshots = [], []

    def load_state(self):
        return {"owner_chat_id": 7}

    def send_with_budget(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))

    def enqueue_task(self, task):
        return queue.enqueue_task(task)

    def persist_queue_snapshot(self, reason=""):
        self.snapshots.append(reason)
        return True

    def append_jsonl(self, path, row):
        from ouroboros.utils import append_jsonl
        append_jsonl(path, row)


class Env(SimpleNamespace):
    def request(self, tid, *, root="root", parent="parent", surface="self_worktree", source=None,
                base_sha="", write=True):
        """The producer's durable REQUESTED row and the schedule_subagent event it emits."""
        child_drive = str(self.data / "state" / "headless_tasks" / tid / "data")
        workspace = str(source or self.source)
        facts = {"objective": f"change {tid}", "expected_output": "a patch", "role": "builder",
                 "parent_task_id": parent, "root_task_id": root, "delegation_role": "subagent",
                 "memory_mode": "forked", "drive_root": child_drive, "child_drive_root": child_drive,
                 "budget_drive_root": str(self.data), "workspace_root": workspace, "workspace_mode": "external",
                 "task_constraint": {"mode": "acting_subagent", "surface": surface}, "chat_id": 7}
        if write:
            write_task_result(self.data, tid, STATUS_REQUESTED, description=facts["objective"],
                              result="Subagent request queued. Awaiting supervisor acceptance.", **facts)
        return {**facts, "type": "schedule_subagent", "task_id": tid, "description": facts["objective"],
                "depth": 1, **({"base_sha": base_sha} if base_sha else {})}

    def readonly(self, tid, *, root="other-root"):
        event = self.request(tid, root=root, parent=root)
        event["task_constraint"] = {"mode": "local_readonly_subagent"}
        return event

    def row(self, tid):
        return load_task_result(self.data, tid, strict=True) or {}

    def phase(self, tid):
        return (self.row(tid).get("workspace_preparation") or {}).get("phase")

    def wait_phase(self, tid, phase):
        deadline = time.monotonic() + _WAIT_SEC
        while time.monotonic() < deadline:
            if self.phase(tid) == phase:
                return
            time.sleep(0.01)
        raise AssertionError(f"{tid} never reached {phase}: {self.row(tid).get('workspace_preparation')}")

    def dispatch_returns(self, event):
        """Dispatch through the real handler on a stand-in loop thread; did it return promptly?"""
        done = threading.Event()
        thread = threading.Thread(target=lambda: (dispatch_event(event, self.ctx), done.set()),
                                  name="test-supervisor-loop", daemon=True)
        thread.start()
        returned = done.wait(_WAIT_SEC)
        self.loops.append(thread)
        return returned

    def pending_ids(self):
        with queue._queue_lock:
            return [str(row.get("id") or "") for row in queue.PENDING]

    @contextlib.contextmanager
    def copy_blocked(self):
        """Hold every private-copy provision at its snapshot step (before it records or
        makes anything, where a large tree spends its time) until the block exits."""
        release = threading.Event()
        real = trees.snapshot_commit

        def blocked(*args, **kwargs):
            assert release.wait(_WAIT_SEC * 4), "a held copy was never released"
            return real(*args, **kwargs)

        trees.snapshot_commit = blocked
        try:
            yield
        finally:
            trees.snapshot_commit = real
            release.set()


@pytest.fixture
def env(tmp_path, monkeypatch):
    system = make_repo(tmp_path / "system", {"system.txt": b"own body\n"})
    source = make_repo(tmp_path / "project", {"a.txt": b"base\n"})
    data, copies = tmp_path / "data", tmp_path / "copies"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(copies))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv("OUROBOROS_BOOT_RUNTIME_MODE", raising=False)
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "")
    monkeypatch.setattr(queue, "DRIVE_ROOT", data)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    fences = {name: dict(getattr(task_lifecycle, name)) for name in ("BUDGET_ROOT_FENCES", "CANCELLED_ROOT_FENCES")}
    for name in fences:
        getattr(task_lifecycle, name).clear()
    prep._reset_for_tests()
    ctx = SupervisorCtx(data, system)
    generation_stop, server_stop = threading.Event(), threading.Event()
    # The composite stop the server passes: this generation's token, or an explicit server stop/restart.
    generation = prep.start_generation(stop=lambda: generation_stop.is_set() or server_stop.is_set())
    state = Env(data=data, system=system, source=source, copies=copies, ctx=ctx, gen=generation, loops=[],
                server_stop=server_stop)
    try:
        yield state
    finally:
        prep.close_generation(generation)
        assert generation.wait_idle(_WAIT_SEC * 4)
        for thread in state.loops:
            thread.join(_WAIT_SEC)
        prep._reset_for_tests()
        for name, saved in fences.items():
            getattr(task_lifecycle, name).clear()
            getattr(task_lifecycle, name).update(saved)


def spy_provisions(monkeypatch):
    calls = []
    real = trees.provision_worktree

    def counted(**kwargs):
        calls.append(kwargs["task_id"])
        return real(**kwargs)

    monkeypatch.setattr(trees, "provision_worktree", counted)
    return calls


def test_blocked_copy_keeps_event_dispatch_responsive_then_admits_once(env, monkeypatch):
    calls = spy_provisions(monkeypatch)
    acting = env.request("act1", root="root-a")
    with env.copy_blocked():
        assert env.dispatch_returns(acting), "the event loop waited on the workspace copy"
        env.wait_phase("act1", "copying")  # the FIFO thread is inside the copy, held at its snapshot
        # Unrelated work and owner controls progress through the real handler meanwhile.
        assert env.dispatch_returns(env.readonly("ro1"))
        env.wait_phase("ro1", "admitted")  # the admission lane never waits behind the held copy
        assert env.row("ro1")["status"] == STATUS_SCHEDULED and "ro1" in env.pending_ids()
        assert env.dispatch_returns({"type": "budget_root_fence", "task_id": "other-task",
                                     "resource_limit": {"scope": "root", "root_task_id": "root-c"}})
        assert task_lifecycle.BUDGET_ROOT_FENCES["root-c"]["status"] == "paused"
        assert env.row("act1")["status"] == STATUS_REQUESTED and "act1" not in env.pending_ids()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert env.pending_ids().count("act1") == 1 and calls == ["act1"]
    row = env.row("act1")
    assert row["status"] == STATUS_SCHEDULED
    assert row["delegation_admission"]["status"] == "accepted"
    assert row["workspace_preparation"]["phase"] == "admitted"
    assert row["workspace_preparation"]["resource"]["path"] == row["workspace_root"]
    assert (pathlib.Path(row["workspace_root"]) / "a.txt").read_bytes() == b"base\n"
    assert queue.ADMISSION_RESERVATIONS == {}


def test_duplicate_arrivals_coalesce_into_one_copy_and_one_admission(env, monkeypatch):
    calls = spy_provisions(monkeypatch)
    event = env.request("dup1")
    with env.copy_blocked():
        assert env.dispatch_returns(event)
        env.wait_phase("dup1", "copying")
        assert env.dispatch_returns(dict(event)) and env.dispatch_returns(dict(event))
    assert env.gen.wait_idle(_WAIT_SEC)
    assert calls == ["dup1"]
    assert env.pending_ids() == ["dup1"]
    assert env.row("dup1")["status"] == STATUS_SCHEDULED
    assert not [text for _chat, text, _kw in env.ctx.sent if "rejected" in text.lower() or "not scheduled" in text]


def test_exact_child_stop_during_copy_prevents_admission_and_frees_only_its_copy(env, monkeypatch):
    from ouroboros import cancel_intents
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {})
    event = env.request("stop1")
    with env.copy_blocked():
        assert env.dispatch_returns(event)
        env.wait_phase("stop1", "copying")
        cancel_intents.request_cancel(env.data, "stop1", reason="owner Stop", source="test")
        queue.cancel_task_custody("stop1")
        assert env.row("stop1")["status"] == "cancelled"
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row("stop1")
    assert row["status"] == "cancelled" and "admission_outcome" not in row  # never overwritten
    assert row["workspace_preparation"]["phase"] == "copying"  # its own claim was not rewritten either
    assert "stop1" not in env.pending_ids() and queue.ADMISSION_RESERVATIONS == {}
    assert not trees.copy_path(trees._resolve_root(), "stop1").exists()
    assert not [entry for entry in trees.list_worktrees() if entry.get("task_id") == "stop1"]


_CONTROL_REASONS = {"root_stop": "root_cancelled", "owner_pause": "root_owner_paused",
                    "generation_close": "preparation_interrupted_by_generation_exit",
                    "server_restart": "preparation_interrupted_by_generation_exit"}


@pytest.mark.parametrize("control", sorted(_CONTROL_REASONS))
def test_control_during_copy_refuses_typed_and_disposes_its_copy_only_after_the_receipt(env, monkeypatch, control):
    from supervisor.events_budget import _set_root_budget_pause_locked

    removals = []
    real_remove = trees.remove_worktree

    def remove_after_receipt(**kwargs):
        row = env.row("ctl1")  # the refusal receipt must already be durable
        removals.append((row.get("status"), bool(row.get("_admission_refusal_token"))))
        return real_remove(**kwargs)

    monkeypatch.setattr(trees, "remove_worktree", remove_after_receipt)
    event = env.request("ctl1", root="root-x")
    with env.copy_blocked():
        assert env.dispatch_returns(event)
        env.wait_phase("ctl1", "copying")
        if control == "root_stop":
            with queue._queue_lock:
                task_lifecycle.CANCELLED_ROOT_FENCES["root-x"] = "2026-10-13T00:00:00Z"
        elif control == "owner_pause":
            with queue._queue_lock:
                _set_root_budget_pause_locked("root-x", {"cause": "owner_pause", "fence_id": "pause-1"})
        elif control == "server_restart":
            env.server_stop.set()  # an explicit server stop/restart request, not a close() call
        else:
            prep.close_generation(env.gen)
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row("ctl1")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["reason_code"] == _CONTROL_REASONS[control] and row["artifact_status"] == "failed"
    marker = row["workspace_preparation"]
    assert marker["phase"] == "refused" and marker["resource_disposition"] == "after_refusal_receipt"
    assert removals == [("failed", True)]
    assert not pathlib.Path(marker["resource"]["path"]).exists()
    assert "ctl1" not in env.pending_ids() and queue.ADMISSION_RESERVATIONS == {}
    notices = [kw.get("progress_meta") or {} for _chat, _text, kw in env.ctx.sent]
    assert any(meta.get("subagent_event") == "rejected" and meta.get("subagent_task_id") == "ctl1" for meta in notices)


def test_queued_preparation_never_starts_once_its_generation_closes(env, monkeypatch):
    calls = spy_provisions(monkeypatch)
    with env.copy_blocked():
        assert env.dispatch_returns(env.request("first", root="r1"))
        env.wait_phase("first", "copying")
        assert env.dispatch_returns(env.request("second", root="r2"))
        assert env.phase("second") is None and "second" in queue.ADMISSION_RESERVATIONS  # claimed, not started
        prep.close_generation(env.gen)
    assert env.gen.wait_idle(_WAIT_SEC)
    assert calls == ["first"]  # the queued claim never started a copy
    for tid in ("first", "second"):
        row = env.row(tid)
        assert row["status"] == "failed" and row["reason_code"] == "preparation_interrupted_by_generation_exit"
        assert row["admission_outcome"] == "never_admitted"
    assert "resource" not in env.row("second")["workspace_preparation"]
    assert env.pending_ids() == [] and not trees.list_worktrees()


def test_preparation_captures_the_current_tree_when_it_runs_not_when_requested(env):
    import os

    src = env.source
    (src / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")
    (src / "tool.sh").write_bytes(b"#!/bin/sh\necho hi\n")
    os.chmod(src / "tool.sh", 0o755)
    git(src, "add", "crlf.txt")
    git(src, "add", "--chmod=+x", "tool.sh")  # Windows has no exec bit to read: the mode is git's
    git(src, "commit", "-qm", "modes and line endings")
    (src / "a.txt").write_bytes(b"base\nedited before the request\n")
    (src / "notes.txt").write_bytes(b"untracked eligible input\n")
    head = git(src, "rev-parse", "HEAD").strip().decode()
    with env.copy_blocked():
        assert env.dispatch_returns(env.request("blocker", root="rb"))
        env.wait_phase("blocker", "copying")
        assert env.dispatch_returns(env.request("timed", root="rt", base_sha=head))
        assert env.phase("timed") is None and "timed" in queue.ADMISSION_RESERVATIONS  # claimed, not started
        (src / "a.txt").write_bytes(b"base\nedited while the request was queued\n")  # same HEAD, new dirty bytes
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row("timed")
    assert row["status"] == STATUS_SCHEDULED
    copy = pathlib.Path(row["workspace_root"])
    assert (copy / "a.txt").read_bytes() == b"base\nedited while the request was queued\n"
    assert (copy / "crlf.txt").read_bytes() == b"one\r\ntwo\r\n"
    assert git(copy, "ls-files", "-s", "tool.sh").startswith(b"100755 ")
    assert os.name == "nt" or (copy / "tool.sh").stat().st_mode & 0o111  # Windows reports 0o100666
    assert (copy / "notes.txt").read_bytes() == b"untracked eligible input\n"
    binding = row["workspace_copy"]
    assert binding["target_head"] == head and binding["baseline_sha"] != head  # synthetic dirty baseline
    assert git(src, "rev-parse", "HEAD").strip().decode() == head


def test_head_moved_while_queued_keeps_the_typed_base_refusal(env):
    src = env.source
    head = git(src, "rev-parse", "HEAD").strip().decode()
    with env.copy_blocked():
        assert env.dispatch_returns(env.request("blocker", root="rb"))
        env.wait_phase("blocker", "copying")
        assert env.dispatch_returns(env.request("based", root="rt", base_sha=head))
        (src / "a.txt").write_bytes(b"moved\n")
        git(src, "commit", "-qam", "move HEAD")
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row("based")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert "source HEAD differs from the requested base_sha" in row["result"]
    assert "based" not in env.pending_ids()
    assert not [entry for entry in trees.list_worktrees() if entry.get("task_id") == "based"]


def test_producer_requested_row_flows_through_claim_copy_enqueue_and_receipt(env, monkeypatch):
    from ouroboros.tools.control_scheduling import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests.test_task_status_flow import _configure_test_subagent

    _configure_test_subagent(monkeypatch)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(env.source.parent))
    parent = ToolContext(repo_dir=env.system, drive_root=env.data, task_id="parent",
                         workspace_root=env.source, workspace_mode="external")
    out = _schedule_task(parent, subagent_id="api-scout", objective="Change source", expected_output="patch",
                         workspace_root=str(env.source), write_surface="self_worktree")
    assert "queued" in out, out
    event = parent.pending_events[-1]
    tid = event["task_id"]
    requested = env.row(tid)
    assert requested["status"] == STATUS_REQUESTED and "workspace_preparation" not in requested
    assert env.dispatch_returns(event)
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row(tid)
    assert row["status"] == STATUS_SCHEDULED and row["delegation_admission"]["status"] == "accepted"
    marker = row["workspace_preparation"]
    assert marker["phase"] == "admitted" and marker["resource"]["path"] == row["workspace_root"]
    [task] = [item for item in queue.PENDING if item.get("id") == tid]
    assert task["_admission_owner_token"] == marker["token"] and task["workspace_root"] == row["workspace_root"]
    assert queue.ADMISSION_RESERVATIONS == {}


@pytest.mark.parametrize("held", ["foreign_claim", "foreign_reservation", "progressed"])
def test_requested_row_owned_elsewhere_is_never_claimed_copied_or_overwritten(env, monkeypatch, held):
    from tests._store_shared import result_signature

    calls = spy_provisions(monkeypatch)
    event = env.request("owned1")
    if held == "foreign_claim":
        write_task_result(env.data, "owned1", STATUS_REQUESTED, workspace_preparation={
            "token": "another-token", "phase": "copying", "owner": {"pid": 1, "birth": "x"}})
    elif held == "foreign_reservation":
        queue.ADMISSION_RESERVATIONS["owned1"] = "another-token"
    else:
        write_task_result(env.data, "owned1", "cancelled", result="stopped before preparation")
    before = result_signature(env.data, "owned1")
    assert env.dispatch_returns(event)
    assert env.gen.wait_idle(_WAIT_SEC)
    assert calls == [] and result_signature(env.data, "owned1") == before
    assert env.pending_ids() == []
    assert queue.ADMISSION_RESERVATIONS == ({"owned1": "another-token"} if held == "foreign_reservation" else {})


def test_disposal_removes_only_the_exact_recorded_copy(env):
    first = trees.provision_worktree(repo_dir=env.source, task_id="exact1", source_is_system_repo=False)
    recorded = {"surface": "self_worktree", "path": first.path, "branch": first.branch,
                "created_at": first.created_at, "base_sha": first.base_sha}
    # A later holder at the same path is not this token's copy.
    trees.remove_worktree(path=first.path)
    successor = trees.provision_worktree(repo_dir=env.source, task_id="exact1", source_is_system_repo=False)
    assert successor.created_at != first.created_at
    assert prep._dispose_resource("exact1", recorded) == "retained_not_owned"
    assert pathlib.Path(successor.path).is_dir() and [entry["path"] for entry in trees.list_worktrees()] == [successor.path]
    assert prep._dispose_resource("other-task", {**recorded, "created_at": successor.created_at}) == "retained_not_owned"
    own = {**recorded, "created_at": successor.created_at, "branch": successor.branch}
    assert prep._dispose_resource("exact1", own) == "removed"
    assert not pathlib.Path(successor.path).exists() and trees.list_worktrees() == []
    genesis = trees.provision_genesis_project(repo_dir=env.system, task_id="gen1")
    project = pathlib.Path(genesis.path)
    (project / "work.txt").write_bytes(b"child work\n")
    git(project, "add", "work.txt")
    git(project, "-c", "user.name=f", "-c", "user.email=f@example.invalid", "commit", "-qm", "child commit")
    moved = {"surface": "genesis", "path": genesis.path, "base_sha": genesis.base_sha}
    assert prep._dispose_resource("gen1", moved) == "retained_not_owned" and project.is_dir()
    current = git(project, "rev-parse", "HEAD").strip().decode()
    assert prep._dispose_resource("gen1", {**moved, "base_sha": current}) == "removed"
    assert not project.exists()


def _crash_child(data, system, event, stop_at, log_path):
    """A real supervisor process that dies at one preparation boundary (spawned)."""
    import contextlib
    import os
    import traceback

    with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), \
            contextlib.redirect_stderr(log):
        try:
            queue.init(pathlib.Path(data))
            real_transition, real_git = prep._transition, trees._git

            def transition(drive_root, task_id, token, marker, *, expected):
                if (stop_at, marker["phase"]) in {("before_copy", "copying"), ("after_copy", "admitting")}:
                    os._exit(17)
                landed = real_transition(drive_root, task_id, token, marker, expected=expected)
                if stop_at == "after_admission_entry" and marker["phase"] == "admitting" and landed is True:
                    os._exit(17)
                return landed

            def git_step(repo_dir, *args, **kwargs):
                if stop_at == "during_copy" and args[:1] == ("reset",):
                    os._exit(17)  # the copy is registered and checked out, not yet populated
                return real_git(repo_dir, *args, **kwargs)

            prep._transition, trees._git = transition, git_step
            prep.start_generation()
            dispatch_event(event, SupervisorCtx(pathlib.Path(data), pathlib.Path(system)))
            time.sleep(_WAIT_SEC * 4)
        except BaseException:
            traceback.print_exc(file=log)
            raise


def crash_at(env, tid, stop_at):
    import multiprocessing

    log = env.data.parent / f"{tid}-child.log"
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_child, args=(str(env.data), str(env.system), env.request(tid), stop_at, str(log)))
    process.start()
    try:
        process.join(_WAIT_SEC * 4)
        assert process.exitcode == 17, log.read_text(encoding="utf-8") if log.exists() else "no child log"
    finally:
        if process.is_alive():
            process.kill()
            process.join(_WAIT_SEC)
        process.close()


@pytest.mark.parametrize("stop_at, phase", [("before_copy", "queued"), ("during_copy", "copying"),
                                            ("after_copy", "ready")])
def test_crash_before_admission_entry_settles_never_admitted_without_retry(env, monkeypatch, stop_at, phase):
    from ouroboros.tools.control import _wait_for_task

    calls = spy_provisions(monkeypatch)
    crash_at(env, "crash1", stop_at)
    left = env.row("crash1")
    assert left["status"] == STATUS_REQUESTED and left["workspace_preparation"]["phase"] == phase
    recorded = left["workspace_preparation"].get("resource")
    report = prep.recover_interrupted_preparations(env.data)
    assert report["settled"] == ["crash1"], report
    row = env.row("crash1")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["reason_code"] == "preparation_interrupted_by_restart"
    marker = row["workspace_preparation"]
    assert marker["phase"] == "refused" and marker["token"] == left["workspace_preparation"]["token"]
    assert calls == [] and env.pending_ids() == []  # nothing re-copied, retried or dispatched
    assert _wait_for_task(SimpleNamespace(drive_root=env.data), "crash1", timeout_sec=0).startswith(
        "Task wait completed")
    if stop_at == "after_copy":
        assert recorded and marker["resource_disposition"] == "after_refusal_receipt"
        assert not pathlib.Path(recorded["path"]).exists() and trees.list_worktrees() == []
    elif stop_at == "during_copy":
        assert recorded is None and marker["resource_disposition"] == "unidentified_residue_retained"
        assert [entry["task_id"] for entry in trees.list_worktrees()] == ["crash1"]  # residue kept for GC
    else:
        assert recorded is None and marker["resource_disposition"] == "none" and trees.list_worktrees() == []
    assert prep.recover_interrupted_preparations(env.data)["settled"] == []


def test_crash_after_admission_entry_stays_unresolved_and_is_never_refused(env):
    from ouroboros import obligations

    crash_at(env, "crash2", "after_admission_entry")
    left = env.row("crash2")["workspace_preparation"]
    assert left["phase"] == "admitting" and left["resource"]["path"]
    report = prep.recover_interrupted_preparations(env.data)
    assert report["uncertain"] == ["crash2"] and report["settled"] == []
    row = env.row("crash2")
    assert row["status"] == STATUS_REQUESTED and "admission_outcome" not in row  # absence of PENDING proves nothing
    assert obligations.members(env.data, "unknowns")["task:crash2"]["reason"] == prep.ADMISSION_UNCERTAIN
    assert pathlib.Path(left["resource"]["path"]).is_dir() and env.pending_ids() == []
    write_task_result(env.data, "crash2", "cancelled", result="owner Stop settled it")
    prep.recover_interrupted_preparations(env.data)
    assert "task:crash2" not in obligations.members(env.data, "unknowns")


def test_recovery_leaves_a_live_preparation_alone(env):
    with env.copy_blocked():
        assert env.dispatch_returns(env.request("live1"))
        env.wait_phase("live1", "copying")
        report = prep.recover_interrupted_preparations(env.data)
        assert report["live"] == ["live1"] and env.phase("live1") == "copying"
    assert env.gen.wait_idle(_WAIT_SEC)
    assert env.row("live1")["status"] == STATUS_SCHEDULED


def _unwritable(*_args, **_kwargs):
    raise OSError("store unavailable")


def test_old_generation_claim_that_could_not_settle_is_recovered_by_its_successor(env, monkeypatch):
    from supervisor import task_admission

    with monkeypatch.context() as patch:
        patch.setattr(task_admission, "persist_never_admitted_refusal", _unwritable)
        with env.copy_blocked():
            assert env.dispatch_returns(env.request("old1", root="ro"))
            env.wait_phase("old1", "copying")
            prep.close_generation(env.gen)
        assert env.gen.wait_idle(_WAIT_SEC)
    left = env.row("old1")["workspace_preparation"]
    assert env.row("old1")["status"] == STATUS_REQUESTED and left["phase"] == "ready"
    assert pathlib.Path(left["resource"]["path"]).is_dir()  # an unconfirmed refusal keeps the copy
    successor = prep.start_generation()
    report = prep.recover_interrupted_preparations(env.data)
    assert report["settled"] == ["old1"], report
    row = env.row("old1")
    assert row["reason_code"] == "preparation_interrupted_by_generation_exit"
    assert row["workspace_preparation"]["generation"] == env.gen.token != successor.token
    assert not pathlib.Path(left["resource"]["path"]).exists()


def test_current_generation_job_that_could_not_record_its_outcome_is_recovered_typed(env, monkeypatch):
    from supervisor import task_admission

    real = prep._transition

    def admitting_unwritable(drive_root, task_id, token, marker, *, expected):
        if marker["phase"] == "admitting":
            return None
        return real(drive_root, task_id, token, marker, expected=expected)

    with monkeypatch.context() as patch:
        patch.setattr(prep, "_transition", admitting_unwritable)
        patch.setattr(task_admission, "persist_never_admitted_refusal", _unwritable)
        assert env.dispatch_returns(env.request("unw1"))
        assert env.gen.wait_idle(_WAIT_SEC)
    assert env.row("unw1")["status"] == STATUS_REQUESTED and env.phase("unw1") == "ready"
    assert "unw1" not in env.pending_ids()
    report = prep.recover_interrupted_preparations(env.data)
    assert report["settled"] == ["unw1"], report
    assert env.row("unw1")["reason_code"] == "preparation_marker_unwritable"


def test_unreadable_receipt_keeps_queue_custody_and_recovery_never_refuses_it(env, monkeypatch):
    from ouroboros import obligations
    from supervisor import task_admission

    real_write, real_load = task_admission.write_task_result, task_admission.load_task_result
    receipt = []

    def receipt_write_fails(root, task_id, status, **fields):
        if status == STATUS_SCHEDULED:
            receipt.append(task_id)
            raise OSError("receipt write lost")
        return real_write(root, task_id, status, **fields)

    def readback_unreadable(root, task_id, **kwargs):
        if receipt:
            raise OSError("receipt readback unreadable")
        return real_load(root, task_id, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(task_admission, "write_task_result", receipt_write_fails)
        patch.setattr(task_admission, "load_task_result", readback_unreadable)
        assert env.dispatch_returns(env.request("unc1"))
        assert env.gen.wait_idle(_WAIT_SEC)
    assert env.pending_ids() == ["unc1"] and "schedule_subagent_receipt_unconfirmed" in env.ctx.snapshots
    assert env.phase("unc1") == "admitting"
    report = prep.recover_interrupted_preparations(env.data)
    assert report["admitted"] == ["unc1"]  # the queue owns it; the claim is retired, not refused
    row = env.row("unc1")
    assert row["status"] == STATUS_REQUESTED and row["workspace_preparation"]["phase"] == "admitted"
    with queue._queue_lock:
        queue.PENDING.clear()  # a restart that never persisted the row: still not proof it was unsent
    report = prep.recover_interrupted_preparations(env.data)
    assert report["uncertain"] == ["unc1"] and "admission_outcome" not in env.row("unc1")
    assert "task:unc1" in obligations.members(env.data, "unknowns")


def test_restart_without_preparation_markers_changes_nothing(env):
    from ouroboros import obligations
    from tests._store_shared import result_signature

    write_task_result(env.data, "legacy-requested", STATUS_REQUESTED, result="queued by an older body")
    write_task_result(env.data, "scheduled1", STATUS_SCHEDULED, result="accepted")
    write_task_result(env.data, "running1", "running", result="working")
    write_task_result(env.data, "paused1", STATUS_SCHEDULED, budget_pause={"state": "paused", "pause_id": "p1"})
    write_task_result(env.data, "refused-old", "failed", admission_outcome="never_admitted",
                      workspace_preparation={"token": "old", "phase": "refused"})
    ids = ("legacy-requested", "scheduled1", "running1", "paused1", "refused-old")
    before = {tid: result_signature(env.data, tid) for tid in ids}
    report = prep.recover_interrupted_preparations(env.data)
    assert report["live"] == ["legacy-requested"]  # held: nothing proves its request cannot arrive
    assert not any(value for key, value in report.items() if key != "live"), report
    assert {tid: result_signature(env.data, tid) for tid in ids} == before
    assert obligations.members(env.data, "unknowns") == {}


def test_unreadable_requested_rows_settle_nothing(env, monkeypatch):
    from ouroboros import task_store

    write_task_result(env.data, "unread1", STATUS_REQUESTED, workspace_preparation={
        "token": "t", "phase": "queued", "owner": {"pid": 1, "birth": "x"}})
    with monkeypatch.context() as patch:
        patch.setattr(task_store, "list_documents", _unwritable)
        report = prep.recover_interrupted_preparations(env.data)
    assert report["errors"] and not report["settled"]
    assert env.row("unread1")["status"] == STATUS_REQUESTED


def _budget_probe(monkeypatch):
    """A recording stand-in for the store's per-context acquisition budget (absent on older stores)."""
    import contextvars

    from ouroboros import store

    budget = contextvars.ContextVar("test_acquisition_budget", default=())

    @contextlib.contextmanager
    def acquisition_budget(seconds):
        token = budget.set(budget.get() + (seconds,))
        try:
            yield
        finally:
            budget.reset(token)

    monkeypatch.setattr(store, "acquisition_budget", acquisition_budget, raising=False)
    return lambda: list(budget.get())


def test_entry_claim_and_marker_never_wait_for_storage_on_the_event_loop(env, monkeypatch):
    from supervisor import task_admission

    active = _budget_probe(monkeypatch)
    seen = []
    real_claim, real_transition = task_admission.claim_requested_child, prep._transition

    def claim(*args, **kwargs):
        seen.append(("claim", threading.current_thread().name, active()))
        return real_claim(*args, **kwargs)

    def transition(drive_root, task_id, token, marker, *, expected, **kwargs):
        seen.append((marker["phase"], threading.current_thread().name, active()))
        return real_transition(drive_root, task_id, token, marker, expected=expected, **kwargs)

    monkeypatch.setattr(task_admission, "claim_requested_child", claim)
    monkeypatch.setattr(prep, "_transition", transition)
    opened = []
    real_connection = store.connection

    @contextlib.contextmanager
    def connection(*args, **kwargs):
        opened.append(threading.current_thread().name)
        with real_connection(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(store, "connection", connection)
    assert env.dispatch_returns(env.request("budget1"))
    assert env.gen.wait_idle(_WAIT_SEC)
    entry = [row for row in seen if row[1] == "test-supervisor-loop"]
    assert [row[0] for row in entry] == ["claim"] and "test-supervisor-loop" not in opened  # memory only
    off_loop = [row for row in seen if row[1] != "test-supervisor-loop"]
    assert [(row[0], row[1]) for row in off_loop] == [("queued", "workspace-preparation"), ("copying", "workspace-preparation"),
                                                      ("ready", "workspace-preparation"),
                                                      ("admitting", "workspace-preparation-validation"),
                                                      ("admitted", "workspace-preparation-validation")]
    assert all(row[2] == [ACQUISITION_SLICE_SEC] for row in off_loop)  # one slice per attempt, off Q
    assert env.row("budget1")["status"] == STATUS_SCHEDULED


@pytest.mark.parametrize("busy", ["marker", "marker_and_refusal"])
def test_busy_store_at_entry_is_typed_and_starts_no_preparation(env, monkeypatch, busy):
    from ouroboros import headless
    from ouroboros.store import StoreBusy
    from supervisor import task_admission
    from tests._store_shared import result_signature

    calls = spy_provisions(monkeypatch)
    drive_cleanup = []
    monkeypatch.setattr(headless, "remove_subagent_task_drive",
                        lambda *_a, **_k: drive_cleanup.append(threading.current_thread().name))
    event = env.request("busy1")
    before = result_signature(env.data, "busy1")

    def store_busy(*_args, **_kwargs):
        raise StoreBusy("current store busy: database is locked")

    monkeypatch.setattr(prep, "write_task_result", store_busy)
    if busy == "marker_and_refusal":
        monkeypatch.setattr(task_admission, "persist_never_admitted_refusal", store_busy)
    assert env.dispatch_returns(event)
    assert env.gen.wait_idle(_WAIT_SEC)
    assert calls == [] and env.pending_ids() == [] and queue.ADMISSION_RESERVATIONS == {}
    row = env.row("busy1")
    texts = [text for _chat, text, _kw in env.ctx.sent]
    if busy == "marker":
        assert row["status"] == "failed" and row["reason_code"] == "preparation_marker_unwritable"
        assert row["admission_outcome"] == "never_admitted" and row["artifact_status"] == "failed"
        assert drive_cleanup == ["workspace-preparation"]  # refusal cleanup left the event thread
    else:
        assert result_signature(env.data, "busy1") == before and "workspace_preparation" not in row
        assert any("scheduled_result_authority_unknown" in text or "unreadable" in text for text in texts)
        assert drive_cleanup == []


@pytest.mark.parametrize("change", ["rebind", "deletion"])
@pytest.mark.parametrize("phase", ["copy", "admission"])
def test_project_change_during_copy_is_revalidated_at_admission(env, monkeypatch, change, phase):
    from ouroboros import projects_registry as registry, settings_integrity
    from ouroboros.startup_migrations import prepare_startup_state

    prepare_startup_state(env.data)
    registry.create_project(env.data, "proj", name="Proj")
    event = env.request("projchild", root="proj-root")
    event["project_id"] = "proj"  # the parent's Project, frozen when the request is read
    def change_project():
        if change == "rebind":
            registry.update_project(env.data, "proj", working_dir=str(env.data.parent / "elsewhere"))
        else:
            registry.begin_project_deletion(env.data, "proj")

    original_read, reads = settings_integrity.live_subagents_mode, []
    def read(**kwargs):
        reads.append(kwargs)
        if phase == "admission" and len(reads) == 2:
            change_project()
        return original_read(**kwargs)
    monkeypatch.setattr(settings_integrity, "live_subagents_mode", read)
    with env.copy_blocked():
        assert env.dispatch_returns(event)
        env.wait_phase("projchild", "copying")
        if phase == "copy":
            change_project()
    assert env.gen.wait_idle(_WAIT_SEC)
    if phase == "admission":
        assert len(reads) == 2
    row = env.row("projchild")
    marker = row["workspace_preparation"]
    if change == "rebind":
        # A completed folder rebind cannot redirect the request: it is admitted on its frozen basis.
        assert row["status"] == STATUS_SCHEDULED and env.pending_ids() == ["projchild"]
        basis = row["_project_admission"]
        assert basis["project_id"] == "proj" and basis["project"]["working_dir"] == ""
        assert marker["phase"] == "admitted" and pathlib.Path(marker["resource"]["path"]).is_dir()
        return
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert str(row["reason_code"]).startswith("project_routing_fence"), row["reason_code"]
    assert marker["phase"] == "refused" and not pathlib.Path(marker["resource"]["path"]).exists()
    assert "projchild" not in env.pending_ids() and queue.ADMISSION_RESERVATIONS == {}


def test_closed_generation_callback_never_admits_or_touches_its_successor(env):
    old = env.gen
    successor = None
    try:
        with env.copy_blocked():
            assert env.dispatch_returns(env.request("old2", root="r-old"))
            env.wait_phase("old2", "copying")
            prep.close_generation(old)
            successor = prep.start_generation()  # an in-process revival
            queue.ADMISSION_RESERVATIONS.clear()  # the successor's queue init clears the map
            assert prep.recover_interrupted_preparations(env.data)["live"] == ["old2"]
            assert env.dispatch_returns(env.request("old2"))  # a replay finds the old claim's marker
            assert env.dispatch_returns(env.request("new2", root="r-new"))
            env.wait_phase("new2", "copying")
        assert old.wait_idle(_WAIT_SEC) and successor.wait_idle(_WAIT_SEC)
        stale = env.row("old2")
        assert stale["status"] == "failed" and stale["reason_code"] == "preparation_interrupted_by_generation_exit"
        assert stale["workspace_preparation"]["generation"] == old.token
        assert not pathlib.Path(stale["workspace_preparation"]["resource"]["path"]).exists()
        fresh = env.row("new2")
        assert fresh["status"] == STATUS_SCHEDULED and fresh["workspace_preparation"]["generation"] == successor.token
        assert pathlib.Path(fresh["workspace_root"]).is_dir()
        assert [entry["task_id"] for entry in trees.list_worktrees()] == ["new2"]
        assert env.pending_ids() == ["new2"]
    finally:
        if successor is not None:
            successor.close()
            successor.wait_idle(_WAIT_SEC)
