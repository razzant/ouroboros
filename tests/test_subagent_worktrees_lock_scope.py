"""Private copies never wait for one another (#1241).

One delegated snapshot of a project with 67,692 untracked files once held a
machine-wide ``.worktree_ops.lock`` for 40 minutes, and every other mutating
``delegate_start`` on the machine waited 120 s and was refused. There is no
shared lock any more: each copy has its own record file, written BEFORE the pin
and the worktree it names (so a crash at any later point leaves a copy custody GC
can name), and a record is locked only while that one copy changes. A held
record lock is a typed refusal that names its holder and blocks nothing else; a
dead holder is evicted at once; a refused provision is definitely unrun and
carries git's own words.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import threading
import time

import pytest

from ouroboros import body_candidate
from ouroboros import subagent_worktrees as wt
from ouroboros.platform_layer import _lock_identity, pid_is_alive
from ouroboros.unindexed import COPIES
from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401
from tests.test_delegated_full_access import full_run  # noqa: F401
from tests.test_delegated_run_isolation import _git, _nanny_ctx, _seed_target

REPO = pathlib.Path(__file__).resolve().parents[1]


def _provision(target, snaps, data, snapshot_id="snap1", task_id="t1"):
    return wt.provision_execution_snapshot(
        target_root=target, task_id=task_id, snapshot_id=snapshot_id, worktree_root=snaps, data_dir=data)


def _record_lock_path(path, data) -> pathlib.Path:
    return wt._record_file(path, data).with_suffix(".lock")


def _lock_held(path, data) -> bool:
    return bool(_lock_identity(_record_lock_path(path, data)))


def _nothing_left(target, snaps, data, snapshot_id):
    assert wt.find_execution_snapshot(snapshot_id, data_dir=data) is None
    assert _git(target, "for-each-ref", "refs/ouroboros/").stdout == ""
    assert not [*snaps.glob("dlg_*"), *snaps.glob(f"{COPIES}/dlg_*")]
    assert not list((target / ".git" / "worktrees").glob("dlg_*"))
    assert not list((data / "state" / "private_copies").glob("*.json"))


def test_the_record_is_written_before_the_pin_and_the_worktree_it_names(tmp_path, monkeypatch):
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    order: list = []
    real_git, real_put = wt._git, wt._put_record

    def spy_git(repo_dir, *args, **kw):
        if args[:1] in (("update-ref",), ("reset",)) or args[:2] == ("worktree", "add"):
            order.append(args[0] if args[0] != "worktree" else "worktree add")
        return real_git(repo_dir, *args, **kw)

    def spy_put(row, data_dir=None):
        order.append(f"record:{row['provisioning_sec'] > 0}")
        return real_put(row, data_dir)

    monkeypatch.setattr(wt, "_git", spy_git)
    monkeypatch.setattr(wt, "_put_record", spy_put)
    handle = _provision(target, snaps, data)

    assert order == ["record:False", "update-ref", "worktree add", "reset", "record:True"]
    assert handle.provisioning_sec > 0
    assert wt.find_execution_snapshot("snap1", data_dir=data)["provisioning_sec"] == handle.provisioning_sec
    assert not _lock_held(handle.path, data), "provisioning a new copy takes no lock at all"


@pytest.mark.parametrize("same_target", [False, True])
def test_a_parked_provision_does_not_block_another(tmp_path, monkeypatch, same_target):
    target_a = _seed_target(tmp_path)
    (tmp_path / "other").mkdir()
    target_b = target_a if same_target else _seed_target(tmp_path / "other")
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    parked, release = threading.Event(), threading.Event()
    real_snapshot = wt.snapshot_commit

    def parking_snapshot(*args, **kw):
        made = real_snapshot(*args, **kw)  # the tree walk and hashing are done; nothing is pinned yet
        if threading.current_thread().name == "provision-a":
            parked.set()
            assert release.wait(timeout=60), "the parked provision was never released"
        return made

    monkeypatch.setattr(wt, "snapshot_commit", parking_snapshot)
    errors: list = []

    def run_a():
        try:
            _provision(target_a, snaps, data, snapshot_id="snapA", task_id="ta")
        except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
            errors.append(exc)

    thread_a = threading.Thread(target=run_a, name="provision-a")
    thread_a.start()
    assert parked.wait(timeout=60)
    index_before = (target_b / ".git" / "index").read_bytes()
    head_before = _git(target_b, "rev-parse", "HEAD").stdout

    b_done = threading.Event()

    def run_b():
        _provision(target_b, snaps, data, snapshot_id="snapB", task_id="tb")
        b_done.set()

    thread_b = threading.Thread(target=run_b, name="provision-b")
    thread_b.start()
    assert b_done.wait(timeout=120), "provision B did not finish while A was parked"
    assert wt.find_execution_snapshot("snapA", data_dir=data) is None, "A was still parked when B finished"
    release.set()
    thread_a.join(timeout=120)
    thread_b.join(timeout=120)
    assert not errors and not thread_a.is_alive()
    assert (target_b / ".git" / "index").read_bytes() == index_before
    assert _git(target_b, "rev-parse", "HEAD").stdout == head_before
    rows = {row["snapshot_id"]: row for row in wt.list_worktrees(data_dir=data)}
    assert set(rows) == {"snapA", "snapB"}
    assert all(pathlib.Path(rows[s]["path"]).is_dir() for s in rows)


_HOLDER = """
import os, pathlib, sys
sys.path.insert(0, sys.argv[2])
from ouroboros.platform_layer import acquire_exclusive_file_lock
fd = acquire_exclusive_file_lock(
    pathlib.Path(sys.argv[1]), timeout_sec=5, stale_sec=600,
    metadata=f"pid={os.getpid()} task=t-holder op=capture_result since=2026-09-24T00:00:00Z target=/tmp/with space")
assert fd is not None
print("HELD", os.getpid(), flush=True)
sys.stdin.readline()  # hold until the test closes our stdin (or kills us)
"""


def _hold_lock(lock_path: pathlib.Path) -> "tuple[subprocess.Popen, int]":
    """A live lock holder in another process; returns it with the pid the holder
    itself wrote into the lock (on Windows a venv ``python.exe`` is a launcher whose
    CHILD is the interpreter, so ``proc.pid`` is not that pid)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(lock_path), str(REPO)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    banner = proc.stdout.readline().split()
    assert banner[:1] == ["HELD"], banner
    return proc, int(banner[1])


def _release_holder(proc: subprocess.Popen, holder_pid: int) -> None:
    proc.kill()
    proc.stdin.close()  # the interpreter behind a launcher exits on EOF too
    proc.wait()
    # Wait for the HOLDER (not only the launcher) to be gone: it may still hold
    # the OS-level lock for a moment after EOF, and the dead-holder phase below
    # rewrites the lock file by hand.
    deadline = time.monotonic() + 10
    while pid_is_alive(holder_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_is_alive(holder_pid)


def test_a_held_record_lock_refuses_only_its_copy_typed_and_a_dead_holder_is_evicted(tmp_path, monkeypatch):
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    held = _provision(target, snaps, data, snapshot_id="held")
    free = _provision(target, snaps, data, snapshot_id="free")
    monkeypatch.setattr(wt, "_LOCK_TIMEOUT_SEC", 0.5)
    lock_path = _record_lock_path(held.path, data)
    holder, holder_pid = _hold_lock(lock_path)
    try:
        started = time.monotonic()
        with pytest.raises(wt.WorktreeOpsLockBusy) as info:
            wt.remove_execution_snapshot("held", worktree_root=snaps, data_dir=data)
        assert time.monotonic() - started < 10
        # Another copy's change does not wait for that lock.
        assert wt.remove_execution_snapshot("free", worktree_root=snaps, data_dir=data) is True
        third = _provision(target, snaps, data, snapshot_id="third")
    finally:
        _release_holder(holder, holder_pid)
    busy = info.value
    assert busy.holder == {"pid": str(holder_pid), "task": "t-holder", "op": "capture_result",
                           "since": "2026-09-24T00:00:00Z", "target": "/tmp/with space"}
    assert busy.waited_sec > 0 and "t-holder" in str(busy) and str(lock_path) in str(busy)
    assert pathlib.Path(held.path).is_dir() and wt.find_execution_snapshot("held", data_dir=data) is not None
    assert _git(target, "rev-parse", held.baseline_ref).stdout.strip() == held.baseline_sha
    assert not pathlib.Path(free.path).exists() and wt.find_execution_snapshot("free", data_dir=data) is None
    assert pathlib.Path(third.path).is_dir()

    # A holder that died (SIGKILL, panic) left its lock file behind: the owner-aware
    # stale check evicts it at once instead of refusing for _LOCK_STALE_SEC.
    lock_path.write_text(f"pid={holder.pid} task=t-dead op=remove since=x target=y", encoding="utf-8")
    started = time.monotonic()
    assert wt.remove_execution_snapshot("held", worktree_root=snaps, data_dir=data) is True
    assert time.monotonic() - started < 30  # far below _LOCK_STALE_SEC=600
    assert not pathlib.Path(held.path).exists() and not lock_path.exists()


def test_a_provisioning_refusal_is_definitely_unrun_and_carries_git_s_words(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_shared import REFUSAL_FACT_KEYS, delegate_payload
    from ouroboros.subagent_bootstrap import _startup_refusal_definite
    from ouroboros.tools.delegate_integration import _provision_snapshot

    target = _seed_target(tmp_path)
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    drive = custody.custody_root(ctx)
    _git(target, "update-ref", "refs/ouroboros/delegated", "HEAD")  # git cannot write the pin under it

    handle, refusal = _provision_snapshot(ctx, drive, str(target), "inv-git")
    payload = delegate_payload(refusal)
    assert handle is None and payload["reason"] == "execution_snapshot_failed"
    assert payload["definitely_unrun"] is True and _startup_refusal_definite(payload)
    assert "GitFailed" in payload["detail"] and "cannot lock ref" in payload["detail"], payload["detail"]
    assert not set(REFUSAL_FACT_KEYS) & set(payload), "a git failure carries no lock facts"
    assert not [*(tmp_path / "snaps").glob("dlg_*"), *(tmp_path / "snaps").glob(f"{COPIES}/dlg_*")]

    def broken(**kw):
        raise RuntimeError("git exploded")

    monkeypatch.setattr(wt, "provision_execution_snapshot", broken)
    _handle, refusal = _provision_snapshot(ctx, drive, str(target), "inv-broken")
    payload = delegate_payload(refusal)
    assert payload["definitely_unrun"] is True and "cause" not in payload and "git exploded" in payload["detail"]
    assert _startup_refusal_definite(payload)


def test_a_provisioning_refusal_leaves_a_durable_start_failed_row(full_run, monkeypatch):  # noqa: F811
    """The incident's two bootstrap refusals existed only inside the child's first
    prompt: no custody row, no event. A refused provision now settles its
    invocation durably, carries git's own words, and never reaches the daemon."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_shared import delegate_payload
    from ouroboros.tools import delegate

    ctx, _target, facts = full_run

    def refused(**_kw):
        raise wt.GitFailed(128, ["git", "worktree", "add"], "", "fatal: '/copies/dlg' already exists")

    monkeypatch.setattr(wt, "provision_execution_snapshot", refused)
    payload = delegate_payload(delegate._delegate_start(ctx, "Implement the fixture."))

    assert payload["reason"] == "execution_snapshot_failed" and payload["definitely_unrun"] is True
    assert "already exists" in payload["detail"]
    assert facts["requests"] == [], "a refused provision must never POST a run"
    rows = [json.loads(line) for line in custody.event_log_path(custody.custody_root(ctx))
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    failed = [row for row in rows if row.get("type") == custody.START_FAILED]
    assert len(failed) == 1 and failed[0]["definite"] is True and failed[0]["invocation_id"]
    assert failed[0]["reason"] == "execution_snapshot_failed" and failed[0]["run_id"] == ""
    assert "git worktree add exited 128: fatal: '/copies/dlg' already exists" in failed[0]["detail"]


def test_configured_child_bootstrap_keeps_the_refusal_facts(tmp_path, monkeypatch):
    """The host pre-start of a configured leaf (the incident's first refusal) ends the
    child at $0 AND keeps the producer's facts: the $0 terminal names the lock holder,
    the availability row and the acceptance evidence carry the detail."""
    from ouroboros import subagent_bootstrap, subagent_runtime
    from ouroboros.agent_dispatch import executor_blocked_outcome
    from ouroboros.delegate_shared import _fail
    from ouroboros.subagents import SubagentExecutorResolution

    busy = wt.WorktreeOpsLockBusy(tmp_path / "lock", 120.0, {"pid": "4242", "task": "t-other", "op": "provision"})
    refusal = _fail("delegate_start", "execution_snapshot_failed",
                    f"A private execution snapshot could not be provisioned ({busy}).", definitely_unrun=True,
                    cause="lock_busy", holder=dict(busy.holder), retryable=True, waited_sec=120.0,
                    retry_hint="wait for the holder and retry delegate_start")
    monkeypatch.setattr(subagent_runtime, "exact_start", lambda *_a, **_kw: refusal)
    monkeypatch.setattr(subagent_runtime, "current_subagent_alternatives", lambda selected: [])

    class _Ctx:
        task_id = "t-child"

    ctx = _Ctx()
    task = {"configured_subagent": {"selected_subagent_id": "codex=gpt-6-astra/xhigh"}}
    wake = subagent_bootstrap._pre_start_leaf(ctx, task, {})

    assert wake == ""  # a definite refusal: the child ends unrun at $0, no model round
    stash = ctx._configured_startup_refusal
    assert stash["reason"] == "execution_snapshot_failed" and stash["cause"] == "lock_busy"
    assert stash["holder"]["task"] == "t-other" and "t-other" in stash["detail"]
    availability = task["subagent_availability"]
    assert availability["status"] == "unavailable" and availability["holder"]["pid"] == "4242"
    text, usage = executor_blocked_outcome(
        SubagentExecutorResolution(requested="harness", executor="blocked", reason=stash["reason"]),
        availability=availability)
    assert "t-other" in text and usage["reason_code"] == "subagent_executor_unavailable"


def test_acting_worktree_add_and_remove_change_only_their_own_record(tmp_path, monkeypatch):
    """The acting self_worktree lane uses the same primitive: a branch-pinned linked
    worktree with no checkout hooks, removed under that copy's own record lock only."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    bystander = wt.provision_worktree(repo_dir=target, task_id="bystander", worktree_root=snaps, data_dir=data)
    seen: dict = {}
    real_git, real_rmtree = wt._git, wt._force_rmtree

    def spy_git(repo_dir, *args, **kw):
        if "worktree" in args and "add" in args:
            seen["worktree_add"] = "--no-checkout" in args
        if "reset" in args:
            seen["reset"] = True
        return real_git(repo_dir, *args, **kw)

    def spy_rmtree(path):
        seen["rmtree"] = (_lock_held(path, data), _lock_held(bystander.path, data))
        return real_rmtree(path)

    monkeypatch.setattr(wt, "_git", spy_git)
    monkeypatch.setattr(wt, "_force_rmtree", spy_rmtree)
    handle = wt.provision_worktree(repo_dir=target, task_id="acting1", worktree_root=snaps, data_dir=data)
    assert seen == {"worktree_add": True, "reset": True}
    assert (pathlib.Path(handle.path) / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"  # Current source content, including the parent edit
    assert _git(target, "rev-parse", "--verify", handle.branch).returncode == 0
    assert any(row.get("task_id") == "acting1" for row in wt.list_worktrees(data_dir=data))

    assert wt.remove_worktree(task_id="acting1", worktree_root=snaps, data_dir=data)
    assert seen["rmtree"] == (True, False), "its own record is locked while it goes, nobody else's"
    assert not pathlib.Path(handle.path).exists()
    assert _git(target, "rev-parse", "--verify", handle.branch, check=False).returncode != 0
    assert not any(row.get("task_id") == "acting1" for row in wt.list_worktrees(data_dir=data))
    assert pathlib.Path(bystander.path).is_dir() and any(
        row.get("task_id") == "bystander" for row in wt.list_worktrees(data_dir=data))
    assert not _lock_held(handle.path, data)


def test_a_record_naming_the_snapshot_root_itself_deletes_nothing(tmp_path):
    """A record is durable state; a malformed one whose path IS the root must not
    wipe every live sibling snapshot (the root holds them all). Nor is it forgotten:
    like any existing tree the root may not delete (#1239), its record stays with a
    removal stamp, the refs it names stay pinned, and this process does not retry it."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    keep = _provision(target, snaps, data, snapshot_id="keep")
    row = wt.find_execution_snapshot("keep", data_dir=data)
    wt._put_record({**row, "snapshot_id": "bad", "path": str(snaps.resolve()),
                    "baseline_ref": "refs/ouroboros/delegated/bad"}, data)
    _git(target, "update-ref", "refs/ouroboros/delegated/bad", keep.baseline_sha)
    assert wt.remove_execution_snapshot("bad", worktree_root=snaps, data_dir=data) is False
    assert pathlib.Path(keep.path).is_dir() and wt.find_execution_snapshot("keep", data_dir=data) is not None
    assert _git(target, "rev-parse", keep.baseline_ref).stdout.strip() == keep.baseline_sha
    bad = wt.find_execution_snapshot("bad", data_dir=data)
    assert bad is not None and bad.get("removal_failed") and snaps.is_dir()
    assert _git(target, "rev-parse", "refs/ouroboros/delegated/bad").stdout.strip() == keep.baseline_sha
    assert wt.remove_execution_snapshot("bad", worktree_root=snaps, data_dir=data) is False  # not handled again
    assert not wt._deletable(snaps, snaps) and not wt._deletable(pathlib.Path("/"), snaps)
    assert wt._deletable(pathlib.Path(keep.path), snaps)


@pytest.mark.parametrize("failing", ["update-ref", "worktree", "reset"])
def test_a_failure_after_the_record_leaves_nothing(tmp_path, monkeypatch, failing):
    """The provisional record is written first; a failed pin, admin dir or checkout
    right after it must discard the record too, not only later phases."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    real_git = wt._git

    def failing_git(repo_dir, *args, **kw):
        if failing in args and (failing != "worktree" or "add" in args):
            raise wt.GitFailed(128, ["git", *args], "", f"fatal: injected {failing} failure")
        return real_git(repo_dir, *args, **kw)

    monkeypatch.setattr(wt, "_git", failing_git)
    with pytest.raises(wt.GitFailed, match=f"injected {failing} failure"):
        _provision(target, snaps, data, snapshot_id="snapLockB")
    _nothing_left(target, snaps, data, "snapLockB")
    assert not _lock_held(snaps / COPIES / "dlg_t1_snapLockB", data)


def test_a_failure_after_the_checkout_leaves_nothing_and_a_crash_is_gc_reclaimable(tmp_path, monkeypatch):
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    real_check_out = wt._check_out

    def boom(*args, **kwargs):
        real_check_out(*args, **kwargs)
        raise OSError("checkout failed")

    monkeypatch.setattr(wt, "_check_out", boom)
    with pytest.raises(OSError, match="checkout failed"):
        _provision(target, snaps, data, snapshot_id="snapFail")
    _nothing_left(target, snaps, data, "snapFail")

    # A worker killed between the provisional record and the final one: the record
    # names the copy, custody never opened -> GC removes it once the task is proven over.
    monkeypatch.setattr(wt, "_discard_snapshot_checkout", lambda *a, **k: None)
    with pytest.raises(OSError, match="checkout failed"):
        _provision(target, snaps, data, snapshot_id="snapCrash")
    assert wt.find_execution_snapshot("snapCrash", data_dir=data) is not None
    assert list(snaps.glob(f"{COPIES}/dlg_*")) and "refs/ouroboros/delegated/snapCrash" in _git(
        target, "for-each-ref", "refs/ouroboros/").stdout
    report = wt.prune_execution_snapshots(set(), worktree_root=snaps, data_dir=data, live=lambda _task: False)
    assert report == {"removed": [], "kept": ["snapCrash"]}, "not before its task is proven over"
    from ouroboros.task_results import write_task_result
    write_task_result(data, "t1", "failed")
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_args: [])
    report = wt.prune_execution_snapshots(set(), worktree_root=snaps, data_dir=data, live=lambda _task: False)
    assert report["removed"] == ["snapCrash"]
    _nothing_left(target, snaps, data, "snapCrash")


@pytest.mark.parametrize("autocrlf", [False, True])
def test_populate_matches_worktree_add_and_runs_no_target_hook(tmp_path, monkeypatch, autocrlf):
    """``worktree add --no-checkout`` + ``reset --hard --no-recurse-submodules`` is what
    git's own ``worktree add`` runs — a target with ``submodule.recurse=true`` must
    still snapshot — minus the target's post-checkout hook, which no longer executes
    project-authored code at provision. The copy holds git's checkout of the snapshot:
    under ``core.autocrlf=true`` (Git for Windows' default) its text files have CRLF
    line ends where the source's work tree has LF, and ``git status`` is still clean."""
    # Both legs pin the setting explicitly: on windows-latest the system config
    # already says true, so an unset "False" leg would not be the working side.
    config = tmp_path / "gitconfig"
    config.write_text(f"[core]\n\tautocrlf = {'true' if autocrlf else 'false'}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    sub = tmp_path / "sub"
    sub.mkdir()
    _git(sub, "init", "-q")
    (sub / "s.txt").write_text("s\n", encoding="utf-8")
    _git(sub, "add", "s.txt")
    _git(sub, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "sub")
    target = _seed_target(tmp_path)
    _git(target, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "vendored")
    _git(target, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "vendored")
    _git(target, "config", "submodule.recurse", "true")
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "post-checkout").write_text("#!/bin/sh\ntouch .hook_ran\n", encoding="utf-8")
    (hooks / "post-checkout").chmod(0o755)
    _git(target, "config", "core.hooksPath", str(hooks))
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    # Positive control: the hook DOES fire on a plain `worktree add`, so its absence
    # below is the populate path's doing, not a dead fixture.
    control = tmp_path / "control-wt"
    _git(target, "-c", "submodule.recurse=false", "worktree", "add", "--detach", str(control), "HEAD")
    assert (control / ".hook_ran").exists()
    _git(target, "worktree", "remove", "--force", str(control))

    handle = _provision(target, snaps, data)

    exec_root = pathlib.Path(handle.path)
    assert (exec_root / "vendored").is_dir() and (exec_root / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"
    assert not (exec_root / ".hook_ran").exists() and not (target / ".hook_ran").exists()
    assert _git(exec_root, "status", "--porcelain").stdout == ""
    assert (exec_root / ".gitmodules").read_text(encoding="utf-8") == (target / ".gitmodules").read_text(encoding="utf-8")
    assert (b"\r\n" in (exec_root / "tracked.txt").read_bytes()) is autocrlf
    assert b"\r\n" not in (target / "tracked.txt").read_bytes()
    assert "160000" in _git(exec_root, "ls-files", "-s", "vendored").stdout

    # The guard: without --no-recurse-submodules the populate fails on this target.
    real_git = wt._git

    def recursing_git(repo_dir, *args, **kw):
        return real_git(repo_dir, *tuple(a for a in args if a != "--no-recurse-submodules"), **kw)

    monkeypatch.setattr(wt, "_git", recursing_git)
    with pytest.raises(subprocess.CalledProcessError):
        _provision(target, snaps, data, snapshot_id="snapRecurse")


def test_snapshot_facts_ride_the_start_receipt_as_disclosure_only(tmp_path):
    from ouroboros.tools.delegate import _snapshot_facts

    target = _seed_target(tmp_path)
    (target / "blob.bin").write_bytes(b"\x7fELF\x00" * 10)
    handle = _provision(target, tmp_path / "snaps", tmp_path / "data")
    facts = _snapshot_facts(handle)
    assert set(facts) == {"entries", "provisioning_sec"}
    # tracked.txt, staged.txt and the untracked untracked.txt, .env and blob.bin
    assert facts["entries"] == handle.entry_count == 5 and facts["provisioning_sec"] == handle.provisioning_sec
    assert _snapshot_facts(None) == {}
    # Facts, not a threshold: no runtime module consumes them to refuse or truncate.
    consumers = sorted(
        path.relative_to(REPO).as_posix() for path in (REPO / "ouroboros").rglob("*.py")
        if "provisioning_sec" in path.read_text(encoding="utf-8"))
    assert consumers == ["ouroboros/subagent_worktrees.py", "ouroboros/tools/delegate_integration.py"]


def test_an_empty_snapshot_names_the_nested_repository_it_left_out(tmp_path):
    """The shape of the Claudexor project folder: a Git repository with an empty tree whose
    real work tree is an untracked nested clone. The snapshot leaves the clone out; the
    receipt said only ``entries: 0``, so the nanny had to discover why by hand."""
    from ouroboros.tools.delegate import _snapshot_facts

    target = tmp_path / "project"
    target.mkdir()
    _git(target, "init")
    _git(target, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty", "-m", "empty")
    clone = target / "claudexor"
    clone.mkdir()
    _git(clone, "init")
    (clone / "engine.ts").write_text("export {}\n", encoding="utf-8")
    _git(clone, "add", "-A")
    _git(clone, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed")

    handle = _provision(target, tmp_path / "snaps", tmp_path / "data")
    facts = _snapshot_facts(handle)
    assert facts["entries"] == 0 and facts["nested_repositories_excluded"] == ["claudexor"]
    assert not (pathlib.Path(handle.path) / "claudexor" / "engine.ts").exists()
    assert wt.find_execution_snapshot("snap1", data_dir=tmp_path / "data")["excluded_untracked"] == [
        {"path": "claudexor", "reason": "nested_repository"}]
    # A continuation in the same snapshot names it too; a snapshot with none says nothing extra.
    from ouroboros.delegate_continuation import AdoptedSnapshot

    reused = AdoptedSnapshot("snap1", handle.baseline_sha, handle.path, "run-first", handle.excluded_untracked)
    facts = _snapshot_facts(reused)
    assert set(facts) == {"reused_from_run", "nested_repositories_excluded", "nested_repositories_note"}
    assert facts["nested_repositories_excluded"] == ["claudexor"] and "captures changes there" in facts["nested_repositories_note"]
    assert _snapshot_facts(AdoptedSnapshot("s", "b", "/p", "run-x")) == {"reused_from_run": "run-x"}
    # Only nested repositories are named, the first twenty with the total beside them.
    from types import SimpleNamespace

    many = SimpleNamespace(entry_count=3, excluded_untracked=(
        {"path": "x.sqlite-shm", "reason": "sqlite_wal_index"},
        {"path": "fifo", "reason": "special_file"},
        *({"path": f"repo{n:02d}", "reason": "nested_repository"} for n in range(25))))
    facts = _snapshot_facts(many)
    assert facts["nested_repositories_excluded"] == [f"repo{n:02d}" for n in range(20)]
    assert facts["nested_repositories_excluded_count"] == 25 and facts["entries"] == 3
    assert "baseline manifest" in facts["nested_repositories_note"]
