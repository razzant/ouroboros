"""An existing folder's preparation never waits behind another task's workspace copy.

Copies and new projects keep one FIFO lane of the preparation generation; an
``external_workspace`` claim, whose resolver only validates the folder and pins
its HEAD, runs on the second lane. Admission, controls, generation close,
recovery and cleanup keep their meaning on both. Real dispatch, queue maps,
reservations, receipts and real copies held at their snapshot step.
"""
from __future__ import annotations

import pathlib
import threading
import time

import pytest

from ouroboros import subagent_worktrees as trees
from ouroboros.task_results import STATUS_REQUESTED, STATUS_SCHEDULED
from supervisor import events_subagent_admission as admission
from supervisor import queue
from supervisor import workspace_preparation as prep
from tests.test_workspace_preparation import _WAIT_SEC, spy_provisions
from tests.test_workspace_preparation import env as env


_COPY_THREAD, _VALIDATION_THREAD = "workspace-preparation", "workspace-preparation-validation"


def wait_for(predicate, what):
    deadline = time.monotonic() + _WAIT_SEC
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def held_validation(monkeypatch, env):
    """Hold every external_workspace resolver at its HEAD read until released."""
    reached, release, executors = threading.Event(), threading.Event(), []
    real_head = admission._external_workspace_head

    def head(path):
        executors.append((threading.current_thread().name, env.gen.on_executor()))
        reached.set()
        assert release.wait(_WAIT_SEC)
        return real_head(path)

    monkeypatch.setattr(admission, "_external_workspace_head", head)
    return reached, release, executors


def test_existing_folder_is_admitted_while_another_task_copy_waits(env, monkeypatch):
    calls = spy_provisions(monkeypatch)
    with env.copy_blocked():
        assert env.dispatch_returns(env.request("slowcopy", root="slow-root"))
        env.wait_phase("slowcopy", "copying")
        assert env.dispatch_returns(env.readonly("readcontrol"))  # a readonly request rides the admission lane
        wait_for(lambda: env.phase("readcontrol") == "admitted", "the readonly child's admission")
        assert env.pending_ids() == ["readcontrol"] and env.row("readcontrol")["status"] == STATUS_SCHEDULED
        assert env.dispatch_returns(env.request("existing", root="fast-root", surface="external_workspace"))
        # The receipt (SCHEDULED, ``admitting``) precedes the put, which retires the marker to ``admitted``.
        wait_for(lambda: env.phase("existing") == "admitted", "the existing folder's admission")
        row = env.row("existing")
        assert row["workspace_preparation"]["phase"] == "admitted"
        assert row["workspace_root"] == str(env.source.resolve()) and "resource" not in row["workspace_preparation"]
        assert env.row("slowcopy")["status"] == STATUS_REQUESTED and env.phase("slowcopy") == "copying"
        assert env.pending_ids() == ["readcontrol", "existing"]
    assert env.gen.wait_idle(_WAIT_SEC)
    assert calls == ["slowcopy"]
    assert sorted(env.pending_ids()) == ["existing", "readcontrol", "slowcopy"]  # each admitted exactly once
    assert all(env.row(tid)["status"] == STATUS_SCHEDULED for tid in ("existing", "readcontrol", "slowcopy"))
    assert queue.ADMISSION_RESERVATIONS == {} and not prep._LIVE


def test_each_surface_runs_on_its_lane(env, monkeypatch):
    seen = {}
    real = prep._transition

    def transition(drive_root, task_id, token, marker, *, expected):
        if marker["phase"] in {"copying", "admitting"}:
            seen.setdefault(task_id, set()).add((marker["phase"], threading.current_thread().name))
        return real(drive_root, task_id, token, marker, expected=expected)

    monkeypatch.setattr(prep, "_transition", transition)
    for tid, surface in (("copy1", "self_worktree"), ("gen1", "genesis"), ("ext1", "external_workspace")):
        assert env.dispatch_returns(env.request(tid, root=f"r-{tid}", surface=surface))
        assert env.gen.wait_idle(_WAIT_SEC)  # one at a time
    assert env.dispatch_returns(env.readonly("ro1")) and env.gen.wait_idle(_WAIT_SEC)
    # Copies stay on the copy lane; every admission step (the ``admitting`` boundary, the receipt, the
    # put) runs on the admission lane, an existing folder's validation and a readonly child included.
    assert seen == {"copy1": {("copying", _COPY_THREAD), ("admitting", _VALIDATION_THREAD)},
                    "gen1": {("copying", _COPY_THREAD), ("admitting", _VALIDATION_THREAD)},
                    "ext1": {("copying", _VALIDATION_THREAD), ("admitting", _VALIDATION_THREAD)},
                    "ro1": {("admitting", _VALIDATION_THREAD)}}
    assert all(env.row(tid)["status"] == STATUS_SCHEDULED for tid in seen)


def test_closing_the_generation_settles_both_lanes_and_admits_nothing(env, monkeypatch):
    calls = spy_provisions(monkeypatch)
    reached, release, executors = held_validation(monkeypatch, env)
    try:
        with env.copy_blocked():
            assert env.dispatch_returns(env.request("copy1", root="r1"))
            env.wait_phase("copy1", "copying")
            assert env.dispatch_returns(env.request("copy2", root="r2"))
            assert env.dispatch_returns(env.request("ext1", root="r3", surface="external_workspace"))
            assert reached.wait(_WAIT_SEC), "the existing folder waited behind the held copy"
            assert env.dispatch_returns(env.request("ext2", root="r4", surface="external_workspace"))
            # Queued behind a lane's running job: claimed in memory, nothing written yet.
            assert env.phase("copy2") is None and env.phase("ext2") is None
            assert sorted(claim.task_id for claim in prep._LIVE.values()) == ["copy1", "copy2", "ext1", "ext2"]
            assert executors == [(_VALIDATION_THREAD, True)] and not env.gen.on_executor()
            assert sorted(prep.recover_interrupted_preparations(env.data)["live"]) == ["copy1", "copy2", "ext1", "ext2"]
            prep.close_generation(env.gen)
            release.set()
            wait_for(lambda: not any(claim.task_id.startswith("ext") for claim in prep._LIVE.values()),
                     "the validation lane to settle while the copy is still held")
            assert env.phase("copy1") == "copying"
    finally:
        release.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert calls == ["copy1"] and len(executors) == 1  # no queued claim of either lane started its resolver
    for tid in ("copy1", "copy2", "ext1", "ext2"):
        row = env.row(tid)
        assert row["status"] == "failed" and row["reason_code"] == "preparation_interrupted_by_generation_exit"
        assert row["admission_outcome"] == "never_admitted"
    assert env.pending_ids() == [] and queue.ADMISSION_RESERVATIONS == {} and not trees.list_worktrees()
    assert not any(env.gen._threads.values()) and not any(env.gen._jobs.values())


def test_stop_during_existing_folder_validation_beside_a_held_copy_is_never_admitted(env, monkeypatch):
    from ouroboros import cancel_intents
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {})
    reached, release, _executors = held_validation(monkeypatch, env)
    try:
        with env.copy_blocked():
            assert env.dispatch_returns(env.request("copy1", root="r1"))
            env.wait_phase("copy1", "copying")
            assert env.dispatch_returns(env.request("stopext", root="r2", surface="external_workspace"))
            assert reached.wait(_WAIT_SEC)
            cancel_intents.request_cancel(env.data, "stopext", reason="owner Stop", source="test")
            queue.cancel_task_custody("stopext")
            release.set()
            wait_for(lambda: not any(claim.task_id == "stopext" for claim in prep._LIVE.values()),
                     "the stopped validation to end")
            assert env.row("stopext")["status"] == "cancelled" and "stopext" not in env.pending_ids()
    finally:
        release.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row("stopext")
    assert row["status"] == "cancelled" and "admission_outcome" not in row  # never overwritten
    assert env.pending_ids() == ["copy1"] and env.row("copy1")["status"] == STATUS_SCHEDULED
    assert queue.ADMISSION_RESERVATIONS == {}


def test_validation_lane_that_cannot_start_refuses_only_its_own_request(env, monkeypatch):
    real_start = threading.Thread.start

    def cannot_start(thread):
        if thread.name == _VALIDATION_THREAD:
            raise OSError("thread start failed")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", cannot_start)
    assert env.dispatch_returns(env.request("ext1", root="r1", surface="external_workspace"))
    assert env.dispatch_returns(env.request("copy1", root="r2"))
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row("ext1")
    assert row["status"] == "failed" and row["reason_code"] == "preparation_executor_unavailable"
    assert row["admission_outcome"] == "never_admitted"
    assert env.pending_ids() == ["copy1"] and env.row("copy1")["status"] == STATUS_SCHEDULED


def test_stop_while_a_task_copy_is_still_provisioning_frees_it_only_after_it_returns(env, monkeypatch):
    from ouroboros import cancel_intents
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {})
    parked, release = threading.Event(), threading.Event()
    real_check_out = trees._check_out

    def park_after_checkout(*args, **kwargs):
        real_check_out(*args, **kwargs)
        parked.set()  # recorded and checked out; provisioning has not returned
        assert release.wait(_WAIT_SEC)

    monkeypatch.setattr(trees, "_check_out", park_after_checkout)
    try:
        assert env.dispatch_returns(env.request("half1"))
        assert parked.wait(_WAIT_SEC)
        [entry] = [row for row in trees.list_worktrees() if row.get("task_id") == "half1"]
        copy = pathlib.Path(entry["path"])
        assert (copy / "a.txt").read_bytes() == b"base\n"  # the files are there; the copy is not ready
        assert env.row("half1")["status"] == STATUS_REQUESTED and env.phase("half1") == "copying"
        assert "half1" not in env.pending_ids()
        assert prep.recover_interrupted_preparations(env.data)["live"] == ["half1"]
        assert trees.prune_orphans()["removed"] == 0 and copy.is_dir()  # retention keeps a live copy
        cancel_intents.request_cancel(env.data, "half1", reason="owner Stop", source="test")
        queue.cancel_task_custody("half1")
        assert copy.is_dir()  # Stop never deletes a copy that has not returned
    finally:
        release.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row("half1")
    assert row["status"] == "cancelled" and row["workspace_preparation"]["phase"] == "copying"
    assert "half1" not in env.pending_ids() and queue.ADMISSION_RESERVATIONS == {}
    assert not copy.exists() and not [e for e in trees.list_worktrees() if e.get("task_id") == "half1"]


def _overlapping_sources(tmp_path):
    """A data drive inside a runtime folder, a task drive copy in it, the system repository and an ordinary folder."""
    runtime, repo, outside = tmp_path / "runtime", tmp_path / "repo", tmp_path / "work"
    data = runtime / "data"
    copy_ = data / "task_drives" / "root-task" / "candidate"
    for folder in (copy_, repo, outside):
        folder.mkdir(parents=True)
    return runtime, data, copy_, repo, outside


def test_a_native_writing_child_source_in_runtime_data_hears_what_works_in_the_guards_own_geometry(tmp_path, monkeypatch):
    """The overlap guard still refuses a writing child's source that lies in runtime data or holds it, and the refusal
    names what works instead (C11, 27.11: three fixers died on the bare overlap message), worded for the guard's own
    geometry and carried by its typed reason, never by reading the path again (Astra rev1 C2): an address the
    validator cannot resolve keeps its structured refusal, surrounding spaces keep the hint, a source overlapping the
    system repository keeps its plain refusal, and an ordinary folder outside both is admitted."""
    from types import SimpleNamespace

    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "1")
    runtime, data, copy_, repo, outside = _overlapping_sources(tmp_path)
    ctx = SimpleNamespace(REPO_DIR=repo, DRIVE_ROOT=data)

    def refusal(surface, path):
        return admission._resolve_subagent_constraint(
            ctx, tid="child", requested_constraint={"mode": "acting_subagent", "surface": surface,
                                                    "write_root": path if surface == "external_workspace" else ""},
            workspace_root=path if surface == "self_worktree" else "", workspace_mode="", base_sha="",
            parent_task_id="parent")[3]

    assert refusal("external_workspace", f"{tmp_path}/bad\x00name") == (
        "Subagent rejected: invalid external workspace: embedded null byte")
    spaced = admission._validate_external_workspace(ctx, f"  {copy_}  ")  # the validator strips them: so does the hint
    assert "input_files" in spaced and spaced == admission._validate_external_workspace(ctx, str(copy_))
    assert admission._validate_external_workspace(ctx, str(runtime)) != spaced  # a folder holding the drive: its own words
    inside = refusal("self_worktree", str(copy_))  # the C11 fixers' own path
    around = refusal("external_workspace", str(runtime))
    assert "overlap the Ouroboros data drive. A folder in runtime data (a task drive copy" in inside
    assert ("since the child's Git records (and, for a child working in place, its writes) would land in runtime data"
            in inside)  # a worktree child's files are written outside runtime data (R1F D4, REV3F C-F1)
    assert "prepare_self_change" in inside and "input_files" in inside
    assert "overlap the Ouroboros data drive. A folder that holds runtime data" in around and "input_files" in around
    assert refusal("external_workspace", str(repo / "sub")) == (
        "Subagent rejected: invalid external workspace: workspace_root must not overlap the Ouroboros system repo")
    assert refusal("external_workspace", str(outside)) == ""


def test_a_delegated_writer_hears_what_works_and_an_investigation_keeps_its_readonly_repair(tmp_path):
    """The delegated folder entry refuses the same sources with the same words for a writer, in the guard's own
    geometry; an evidence-only investigation keeps its one repair, the explicit readonly lowering."""
    from types import SimpleNamespace

    from ouroboros.delegate_directory import prepare_directory_execution

    runtime, data, copy_, repo, _outside = _overlapping_sources(tmp_path)
    ctx = SimpleNamespace(repo_dir=repo, drive_root=data, budget_drive_root=data)

    def refusal(path, *, investigation=False):
        with pytest.raises(ValueError) as refused:
            prepare_directory_execution(ctx, None, str(path), SimpleNamespace(mode="execute", investigation=investigation),
                                        None, None)
        return str(refused.value)

    assert refusal(f" {copy_} ") == refusal(copy_) and "input_files" in refusal(copy_)
    assert refusal(runtime) != refusal(copy_)  # its own geometry: a folder holding the drive is not a copy kept in it
    assert refusal(copy_).startswith("workspace_root must not overlap the Ouroboros data drive. A folder in runtime data")
    assert refusal(runtime).startswith("workspace_root must not overlap the Ouroboros data drive. A folder that holds")
    assert refusal(copy_, investigation=True) == (
        "workspace_root must not overlap the Ouroboros data drive; an evidence-only investigation has no private copy "
        "for this folder, so lower this call with access='readonly' to read it in place")
    assert refusal(repo / "sub") == "workspace_root must not overlap the Ouroboros system repo"
