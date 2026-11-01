"""C10: a plain working copy past retention goes only with proof its task is over (the supervisor's
liveness probe, a terminal row, no lineage process); a checkout that survives its removal keeps its
record and pin; a record file is written only when its copy changed, and a pass locks only the
records it selected; ``only`` leaves other copies alone."""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

from ouroboros import body_candidate
from ouroboros import subagent_worktrees as sw
from ouroboros.platform_layer import _lock_identity
from ouroboros.task_results import write_task_result


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=True)


def _init_repo(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    (path / "a.txt").write_text("hi\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo, data, root = tmp_path / "repo", tmp_path / "data", tmp_path / "wtroot"
    _init_repo(repo)
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    return repo, data, root


def _copy(world, task_id):
    repo, data, root = world
    return pathlib.Path(sw.provision_worktree(repo_dir=repo, task_id=task_id, worktree_root=root, data_dir=data).path)


def _set(data, task_id=None, **fields):
    """Rewrite the record of every copy (or of ``task_id``'s) with ``fields``."""
    for row in sw._records(data):
        if task_id is None or row.get("task_id") == task_id:
            sw._put_record({**row, **fields}, data)


def _age(data, *, created_at=1.0):
    _set(data, created_at=created_at)


def _prune(world, **kwargs):
    repo, data, root = world
    return sw.prune_orphans(worktree_root=root, data_dir=data, **kwargs)


def _records_state(data):
    """Every record file's name, bytes and mtime: what a pass that changed nothing leaves alone."""
    return sorted((f.name, f.read_bytes(), f.stat().st_mtime_ns)
                  for f in (data / "state" / "private_copies").glob("*.json") if not f.name.startswith("."))


def _lock_held(path, data):
    return bool(_lock_identity(sw._record_file(path, data).with_suffix(".lock")))


@pytest.mark.parametrize("retention_days", [7, 0])
def test_an_old_copy_stays_while_its_task_runs_is_pending_unknown_or_unreadable(world, monkeypatch, retention_days):
    """L3, the keeping half: age is a courtesy, never a proof."""
    repo, data, root = world
    cases = {
        "running": ("running", True), "pending": ("", True), "unknown": ("completed", None),
        "unreadable": (None, False), "process": ("completed", False), "done": ("completed", False),
    }
    copies = {name: _copy(world, name) for name in cases}
    for name, (status, _probe) in cases.items():
        if status:
            write_task_result(data, name, status)
    _age(data)
    answers = {name: probe for name, (_status, probe) in cases.items()}
    monkeypatch.setattr(body_candidate, "live_lineage_processes",
                        lambda _root, task_id: [4242] if task_id == "process" else [])

    assert _prune(world, retention_days=retention_days) == {"removed": 0, "kept": 6}, "without a probe nothing goes"
    assert all(path.is_dir() for path in copies.values())
    report = _prune(world, retention_days=retention_days, live=lambda task_id: answers[task_id])
    assert report == {"removed": 1, "kept": 5}, "only the proven-over copy goes"
    assert not copies["done"].exists() and all(path.is_dir() for name, path in copies.items() if name != "done")


@pytest.mark.parametrize("retention_days", [7, 0])
def test_an_old_copy_goes_once_its_task_is_terminal_and_proven_not_live(world, retention_days):
    """L3, the removing half: only the retired copy's record goes."""
    repo, data, root = world
    done, fresh = _copy(world, "done"), _copy(world, "fresh")
    write_task_result(data, "done", "completed")
    write_task_result(data, "fresh", "completed")
    _age(data)
    if retention_days:
        _set(data, "fresh", created_at=9e12)  # inside the wait

    report = _prune(world, retention_days=retention_days, live=lambda _task: False)
    expected = 1 if retention_days else 2
    assert report["removed"] == expected
    assert not done.exists() and fresh.exists() != bool(retention_days) or fresh.exists() == bool(retention_days)
    assert _git(repo, "branch", "--list", "subagent/done*").stdout.strip() == ""
    assert "done" not in {row["task_id"] for row in sw.list_worktrees(data_dir=data)}


def test_a_missing_checkout_keeps_its_row_and_branch_until_ownership_is_proven_over(world, monkeypatch):
    repo, data, root = world
    gone = _copy(world, "gone")
    shutil.rmtree(gone)
    before = _records_state(data)
    for status, answer in (("running", False), ("completed", True), ("completed", None)):
        write_task_result(data, "gone", status)
        assert _prune(world, retention_days=9999, live=lambda _task: answer) == {"removed": 0, "kept": 1}
        assert _records_state(data) == before
        assert _git(repo, "branch", "--list", "subagent/gone*").stdout.strip()
    assert _prune(world, retention_days=9999) == {"removed": 0, "kept": 1}
    with monkeypatch.context() as process:
        process.setattr(body_candidate, "live_lineage_processes", lambda *_args: [4242])
        assert _prune(world, retention_days=9999, live=lambda _task: False) == {"removed": 0, "kept": 1}
    assert _prune(world, retention_days=9999, live=lambda _task: False) == {"removed": 1, "kept": 0}
    assert _git(repo, "branch", "--list", "subagent/gone*").stdout == ""


def test_only_restricts_the_pass_and_an_unchanged_copy_keeps_its_record_untouched(world):
    repo, data, root = world
    first, second = _copy(world, "first"), _copy(world, "second")
    for name in ("first", "second"):
        write_task_result(data, name, "completed")
    _age(data)
    before = _records_state(data)
    assert _prune(world, retention_days=0, live=lambda _task: False, only=["nobody"]) == {"removed": 0, "kept": 2}
    assert _records_state(data) == before, "nothing changed: no record is rewritten"
    assert _prune(world, retention_days=0, live=lambda _task: False, only=["second"]) == {"removed": 1, "kept": 1}
    assert first.is_dir() and not second.exists()
    assert _records_state(data) == [entry for entry in before if entry[0] != sw._record_file(second, data).name]


def _pass_cost(monkeypatch):
    """Count the record locks a pass takes (with the copy each names) and its full listings."""
    counts = {"locked": [], "listings": 0}
    real_lock, real_records = sw._record_lock, sw._records

    def lock(path, *args, **kwargs):
        counts["locked"].append(pathlib.Path(str(path)).name)
        return real_lock(path, *args, **kwargs)

    def records(*args, **kwargs):
        counts["listings"] += 1
        return real_records(*args, **kwargs)

    monkeypatch.setattr(sw, "_record_lock", lock)
    monkeypatch.setattr(sw, "_records", records)
    return counts


@pytest.mark.parametrize("count", [100, 800])
@pytest.mark.parametrize("only", [None, ["absent"], ["task-0"]])
def test_a_pass_locks_only_the_records_it_selected_and_rewrites_none(world, monkeypatch, count, only):
    repo, data, root = world
    for i in range(count):
        (root / str(i)).mkdir(parents=True)
        sw._put_record({"task_id": f"task-{i}", "kind": "delegated_exec" if i % 2 else "plain",
                        "path": str(root / str(i)), "created_at": 9e12}, data)
    before = _records_state(data)
    costs = _pass_cost(monkeypatch)
    assert _prune(world, retention_days=7, only=only, live=lambda _: False) == {"removed": 0, "kept": count}
    plain = {str(i) for i in range(0, count, 2)}  # the copies this pass owns; delegated ones are custody's
    expected = plain if only is None else ({"0"} if only == ["task-0"] else set())
    assert sorted(costs["locked"]) == sorted(expected) and costs["listings"] <= 2
    assert _records_state(data) == before


def test_a_selected_old_copy_still_goes_among_many_unrelated_records(world, monkeypatch):
    repo, data, root = world
    path = _copy(world, "selected")
    write_task_result(data, "selected", "completed")
    _age(data)
    unrelated = [{"task_id": f"task-{i}", "kind": "delegated_exec", "path": str(root / str(i))}
                 for i in range(400)]
    for row in unrelated:
        sw._put_record(row, data)
    costs = _pass_cost(monkeypatch)
    assert _prune(world, retention_days=0, only=["selected"], live=lambda _: False) == {"removed": 1, "kept": 400}
    assert set(costs["locked"]) == {path.name} and costs["listings"] <= 2  # its decision, then its removal
    assert not path.exists()
    assert sorted(sw.list_worktrees(data), key=lambda row: row["path"]) == sorted(unrelated, key=lambda row: row["path"])


def test_a_checkout_surviving_its_removal_keeps_its_row_and_is_retried(world, monkeypatch):
    """L5: a forced removal that leaves the directory marks the row instead of forgetting it."""
    repo, data, root = world
    stubborn = _copy(world, "stubborn")
    write_task_result(data, "stubborn", "completed")
    _age(data)
    with monkeypatch.context() as failed:
        failed.setattr(sw, "_force_rmtree", lambda _path: None)  # the removal fails to delete anything
        failed.setattr(sw, "_git_quiet", lambda *_args: None)
        report = _prune(world, retention_days=0, live=lambda _task: False)
    assert report == {"removed": 0, "kept": 1}
    [row] = sw.list_worktrees(data_dir=data)
    assert row["task_id"] == "stubborn" and row["removal_failed"]
    assert stubborn.is_dir()
    assert _git(repo, "branch", "--list", "subagent/stubborn*").stdout.strip(), "the branch is kept with the row"

    # Restore only deletion faults, keeping the fixture's canonical install selected.
    (stubborn / "released").touch()  # a changed identity, no unique work left behind
    (stubborn / "released").unlink()
    assert _prune(world, retention_days=0, live=lambda _task: False) == {"removed": 1, "kept": 0}
    assert not stubborn.exists() and sw.list_worktrees(data_dir=data) == []


def test_remove_worktree_keeps_the_row_when_the_directory_survives(world, monkeypatch):
    repo, data, root = world
    path = _copy(world, "held")
    with monkeypatch.context() as failed:  # restore only the deletion faults: the fixture's install stays selected
        failed.setattr(sw, "_force_rmtree", lambda _path: None)
        failed.setattr(sw, "_git_quiet", lambda *_args: None)
        assert sw.remove_worktree(task_id="held", worktree_root=root, data_dir=data) is False
    [row] = sw.list_worktrees(data_dir=data)
    assert row["removal_failed"] and path.is_dir()
    (path / "released").touch()
    assert sw.remove_worktree(task_id="held", worktree_root=root, data_dir=data) is True
    assert sw.list_worktrees(data_dir=data) == [] and not path.exists()


def test_a_snapshot_surviving_its_removal_keeps_its_row_and_pin_and_only_selects_snapshots(world, monkeypatch):
    repo, data, root = world
    write_task_result(data, "t1", "completed")
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    one = sw.provision_execution_snapshot(target_root=repo, task_id="t1", snapshot_id="snapOne", worktree_root=root, data_dir=data)
    two = sw.provision_execution_snapshot(target_root=repo, task_id="t2", snapshot_id="snapTwo", worktree_root=root, data_dir=data)
    with monkeypatch.context() as failed:
        failed.setattr(sw, "_force_rmtree", lambda _path: None)
        report = sw.prune_execution_snapshots(set(), worktree_root=root, data_dir=data, only={"snapOne"})
    assert report == {"removed": [], "kept": ["snapOne"]}
    assert sw.find_execution_snapshot("snapOne", data_dir=data)["removal_failed"]
    assert pathlib.Path(one.path).is_dir() and pathlib.Path(two.path).is_dir()
    assert _git(repo, "rev-parse", "--verify", one.baseline_ref).stdout.strip(), "the pin stays with the row"
    (pathlib.Path(one.path) / "released").touch()
    assert sw.prune_execution_snapshots(set(), worktree_root=root, data_dir=data, only={"snapOne"}) == {"removed": ["snapOne"], "kept": []}
    assert not pathlib.Path(one.path).exists() and pathlib.Path(two.path).is_dir()
    assert sw.find_execution_snapshot("snapTwo", data_dir=data) is not None


@pytest.mark.parametrize("kind", ["plain", "body", "snapshot"])
def test_pruning_locks_only_the_retired_copy_and_preserves_a_concurrent_registration(world, monkeypatch, kind):
    repo, data, root = world
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    if kind == "snapshot":
        handle = sw.provision_execution_snapshot(
            target_root=repo, task_id="done", snapshot_id="closed", worktree_root=root, data_dir=data)
        path = pathlib.Path(handle.path)
    else:
        path = _copy(world, "done")
        if kind == "body":
            _set(data, "done", kind="body_candidate")
    write_task_result(data, "done", "completed")
    _age(data)
    real_remove, real_drop = sw._force_rmtree, sw._drop_record
    drops, deletions, arrivals = [], [], []

    def remove(checkout):
        assert _lock_held(checkout, data), "a copy's tree goes under its own record lock"
        deletions.append(checkout)
        arrivals.append(_copy(world, "arrived"))  # a real provision interleaves with this deletion
        real_remove(checkout)

    def drop(where, data_dir=None):
        drops.append((pathlib.Path(str(where)), _lock_held(where, data_dir)))
        real_drop(where, data_dir)

    monkeypatch.setattr(sw, "_force_rmtree", remove)
    monkeypatch.setattr(sw, "_drop_record", drop)
    if kind == "snapshot":
        assert sw.prune_execution_snapshots(set(), worktree_root=root, data_dir=data)["removed"] == ["closed"]
    else:
        assert _prune(world, retention_days=0, live=lambda _: False)["removed"] == 1
    assert deletions == [path] and not path.exists()
    assert drops == [(path, True)], "its record is dropped under that same lock"
    assert not _lock_held(arrivals[0], data), "the interleaved provision never waited for it"
    [row] = sw.list_worktrees(data_dir=data)
    assert row["task_id"] == "arrived" and pathlib.Path(row["path"]).is_dir()


def test_snapshot_gc_requires_terminal_task_negative_liveness_and_readable_custody(world, monkeypatch):
    from ouroboros import delegate_custody as custody

    repo, data, root = world
    handle = sw.provision_execution_snapshot(target_root=repo, task_id="owner", snapshot_id="proof",
                                             worktree_root=root, data_dir=data)
    probe, processes = [False], []
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: probe[0])
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_args: processes)

    def kept(open_ids=()):
        before = _records_state(data)
        assert sw.prune_execution_snapshots(set(open_ids), worktree_root=root, data_dir=data) == {
            "removed": [], "kept": ["proof"]}
        assert _records_state(data) == before
        assert _git(repo, "rev-parse", handle.baseline_ref).stdout.strip() == handle.baseline_sha

    kept()  # no task record yet: the provisioning window
    write_task_result(data, "owner", "running")
    kept()
    write_task_result(data, "owner", "completed")
    for probe[0] in (True, None):
        kept()
    probe[0] = False
    processes.append(4242)
    kept()
    processes.clear()
    with monkeypatch.context() as unreadable:
        unreadable.setattr(custody, "custody_log_unreadable", lambda _root: True)
        kept()
    with monkeypatch.context() as broken:
        def fail(_task):
            raise OSError("unreadable liveness")
        broken.setattr("supervisor.queue.task_settlement_liveness", fail)
        kept()
    shutil.rmtree(handle.path)
    kept({"proof"})  # missing checkout does not release open custody or its pin
    assert sw.prune_execution_snapshots(set(), worktree_root=root, data_dir=data) == {
        "removed": ["proof"], "kept": []}
    assert _git(repo, "for-each-ref", handle.baseline_ref).stdout == ""


def test_snapshot_gc_rechecks_custody_between_removals(world, monkeypatch):
    from ouroboros import delegate_custody as custody

    repo, data, root = world
    handles = [sw.provision_execution_snapshot(target_root=repo, task_id="done", snapshot_id=name,
                                               worktree_root=root, data_dir=data) for name in ("first", "second")]
    write_task_result(data, "done", "completed")
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    real_remove = sw.remove_execution_snapshot

    def remove(snapshot_id, **kwargs):
        removed = real_remove(snapshot_id, **kwargs)
        custody.record_started(data, custody.RunCustody(
            run_id="late-start", task_id="done", snapshot_id="second", execution_root=handles[1].path))
        return removed

    monkeypatch.setattr(sw, "remove_execution_snapshot", remove)
    assert sw.prune_execution_snapshots(set(), worktree_root=root, data_dir=data) == {
        "removed": ["first"], "kept": ["second"]}
    assert not pathlib.Path(handles[0].path).exists() and pathlib.Path(handles[1].path).is_dir()
    assert _git(repo, "rev-parse", handles[1].baseline_ref).stdout.strip() == handles[1].baseline_sha


@pytest.mark.parametrize("kind", ["plain", "body", "snapshot"])
def test_failed_removal_waits_for_changed_directory_identity_without_rewriting_first_stamp(world, monkeypatch, kind):
    repo, data, root = world
    if kind == "snapshot":
        handle = sw.provision_execution_snapshot(target_root=repo, task_id="done", snapshot_id="stubborn",
                                                 worktree_root=root, data_dir=data)
        path = pathlib.Path(handle.path)
    else:
        path = _copy(world, "done")
        if kind == "body":
            _set(data, "done", kind="body_candidate")
    write_task_result(data, "done", "completed")
    _age(data)
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    attempts, verdicts, commands = [], [], []
    real_remove, real_verdict, real_git = sw._force_rmtree, body_candidate.retention_verdict, sw._git_quiet
    monkeypatch.setattr(sw, "_force_rmtree", lambda checkout: attempts.append(checkout))

    def verdict(*args, **kwargs):
        verdicts.append(args[0]["path"])
        return real_verdict(*args, **kwargs)

    def git(*args):
        commands.append(args)
        return real_git(*args)

    monkeypatch.setattr(body_candidate, "retention_verdict", verdict)
    monkeypatch.setattr(sw, "_git_quiet", git)

    def prune():
        if kind == "snapshot":
            return sw.prune_execution_snapshots(set(), worktree_root=root, data_dir=data)["removed"]
        return _prune(world, retention_days=0, live=lambda _task: False)["removed"]

    assert not prune() and path.is_dir()
    [first] = sw.list_worktrees(data)
    stamp = first["removal_failed"]
    record = sw._record_file(path, data)
    before, mtime, judged = record.read_bytes(), record.stat().st_mtime_ns, len(verdicts)
    for _ in range(2):
        assert not prune()
        assert record.read_bytes() == before and record.stat().st_mtime_ns == mtime
        assert attempts == [path] and len(verdicts) == judged and commands == []

    (path / "changed").touch()
    (path / "changed").unlink()  # a changed root identity authorizes another attempt, not elapsed time
    assert not prune() and attempts == [path, path]
    assert sw.list_worktrees(data)[0]["removal_failed"] == stamp
    monkeypatch.setattr(sw, "_force_rmtree", real_remove)
    (path / "changed_again").touch()
    (path / "changed_again").unlink()
    assert prune() and not path.exists() and sw.list_worktrees(data) == []


_RETRY_IN_NEW_PROCESS = """
import json, pathlib, sys
from ouroboros import subagent_worktrees as wt
from supervisor import queue
data, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
kind, failing = sys.argv[3], sys.argv[4] == "fail"
queue.task_settlement_liveness = lambda _: False
attempts = []
remove = wt._force_rmtree
def attempt(path):
    attempts.append(str(path))
    if not failing:
        remove(path)
wt._force_rmtree = attempt
removed = []
records = data / "state" / "private_copies"
for _ in range(3):
    if kind == "snapshot":
        result = len(wt.prune_execution_snapshots(set(), worktree_root=root, data_dir=data)["removed"])
    else:
        result = wt.prune_orphans(worktree_root=root, data_dir=data, retention_days=0, live=lambda _: False)["removed"]
    removed.append(result)
    state = sorted((f.name, f.read_bytes(), f.stat().st_mtime_ns) for f in records.glob("*.json"))
    if len(removed) == 1:
        first = state
    else:
        assert state == first, "unchanged failure rewrote its record on a later pass"
rows = wt.list_worktrees(data)
print(json.dumps({"attempts": len(attempts), "removed": removed,
                  "stamp": rows[0].get("removal_failed") if rows else None}))
"""


@pytest.mark.parametrize("kind", ["plain", "body", "snapshot"])
def test_a_deep_survivor_retries_once_in_each_new_process(world, monkeypatch, kind):
    repo, data, root = world
    if kind == "snapshot":
        path = pathlib.Path(sw.provision_execution_snapshot(
            target_root=repo, task_id="done", snapshot_id="deep", worktree_root=root, data_dir=data).path)
    else:
        path = _copy(world, "done")
        if kind == "body":
            _set(data, "done", kind="body_candidate")
    write_task_result(data, "done", "completed")
    _age(data)
    held = path / "nested" / "deep" / "held.txt"
    held.parent.mkdir(parents=True)
    held.write_text("held by another process")
    monkeypatch.setattr(sw, "_force_rmtree", lambda _: None)
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _: False)
    monkeypatch.setattr("ouroboros.copy_capture_proof.plain_copy_verdict",  # this test's subject is the survivor
                        lambda *_a, **_k: {"remove": True, "keep_branch": False, "reason": "no_unique_work"})
    if kind == "snapshot":
        assert sw.prune_execution_snapshots(set(), worktree_root=root, data_dir=data)["removed"] == []
    else:
        assert _prune(world, retention_days=0, live=lambda _: False)["removed"] == 0
    [failed] = sw.list_worktrees(data)
    held.unlink()  # release below the root does not change its saved stat identity
    seen = path.stat()
    assert failed["removal_failed_identity"] == [seen.st_dev, seen.st_ino, seen.st_mtime_ns, seen.st_ctime_ns]

    def restarted(action):
        child = subprocess.run([sys.executable, "-c", _RETRY_IN_NEW_PROCESS, str(data), str(root), kind, action],
                               capture_output=True, text=True, timeout=120)
        assert child.returncode == 0, child.stderr
        return json.loads(child.stdout)

    again = restarted("fail")
    assert again == {"attempts": 1, "removed": [0, 0, 0], "stamp": failed["removal_failed"]}
    assert restarted("remove") == {"attempts": 1, "removed": [1, 0, 0], "stamp": None}
    assert not path.exists() and sw.list_worktrees(data) == []
