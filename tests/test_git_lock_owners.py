"""An ``index.lock`` belongs to whichever process holds it; its age proves nothing.

No reader, retry or boot step removes it: a live writer finishes its own write, and a
step Git refuses returns, in Git's words, as the failed bootstrap or restart it belonged
to (a refused restart abandons the adoption it carried; a failed bootstrap is told in the
owner chat, one line). A replace that Git refused at its first tree step is abandoned as
unapplied, and a Restore whose one reset it refused ends as an aborted update, unrestarted; every
other update refusal keeps the fenced rollback, which runs no destructive command once its
whole target is verified present, so a held lock traps no Update; only stashed owner work
waits, gated, until a start finds Git can apply it. A store-less target moves only through the
store switch. Only an index whose content Git reports corrupt is backed up and rebuilt, never
while the lock exists; a rebuild Git refuses puts the original back. Real Git and real writer
processes throughout.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import supervisor.git_ops as git_ops

pytestmark = pytest.mark.serial  # real writer processes and pinned module roots
_real_sleep = time.sleep  # the fixtures' own polling; the code under test runs with `_no_retry_sleep`


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """No test here times the retry schedule: `_run_git_resilient`'s pauses between refused
    attempts (one second each) would only add seconds to every refusal."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


def _git(repo, *args, input=None):
    proc = subprocess.run(["git", *args], cwd=repo, input=input, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"fixture git {args!r} failed: {proc.stderr}"
    return proc.stdout.strip()


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "ouroboros")
    # Real merge/cherry-pick commands need identity too, not only fixture commits.
    for key, value in {"user.name": "Fixture", "user.email": "fixture@example.invalid",
                       "commit.gpgsign": "false"}.items():
        _git(repo, "config", "--local", key, value)
    return repo


def _repo(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "seed.txt").write_text("committed seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "seed")
    return repo


def _backdate(path, seconds=60):
    """Older than any age gate. Windows refuses to re-time a file another process holds
    open; such a lock stays young there, which changes nothing when no age is read."""
    old = time.time() - seconds
    try:
        os.utime(path, (old, old))
    except PermissionError:
        if sys.platform != "win32":
            raise


def _orphan_lock(repo):
    lock = repo / ".git" / "index.lock"
    lock.write_bytes(b"left by a crashed process\n")
    _backdate(lock)
    return lock


@contextlib.contextmanager
def _live_writer(repo):
    """A real ``git update-index`` that holds ``index.lock`` until its stdin closes."""
    blob = _git(repo, "hash-object", "-w", "--stdin", input="first writer payload\n")
    lock = repo / ".git" / "index.lock"
    proc = subprocess.Popen(["git", "update-index", "--index-info"], cwd=repo,  # bytes: no CRLF in the path
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    writer = SimpleNamespace(lock=lock, stderr="")

    def finish():
        writer.stderr = proc.communicate(timeout=10)[1].decode("utf-8", "replace")
        return proc.returncode

    writer.finish = finish
    try:
        proc.stdin.write(f"100644 {blob}\tfirst.txt\n".encode())
        proc.stdin.flush()
        deadline = time.monotonic() + 10
        while not lock.exists() and proc.poll() is None and time.monotonic() < deadline:
            _real_sleep(0.01)
        assert lock.exists() and proc.poll() is None, "the fixture writer never took the index lock"
        yield writer
    finally:
        if proc.returncode is None:  # not collected by the test
            proc.kill()
            proc.communicate(timeout=10)


def _rows(data, kind):
    path = data / "logs" / "supervisor.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [row for row in map(json.loads, lines) if row.get("type") == kind]


def _worker_env(repo):
    return SimpleNamespace(repo_dir=repo, repo_path=lambda rel: repo / rel)


def test_worker_dirty_check_leaves_a_live_writers_old_lock(tmp_path):
    from ouroboros import agent_startup_checks as checks

    repo = _repo(tmp_path)
    with _live_writer(repo) as writer:
        _backdate(writer.lock)  # older than any age gate
        inode = writer.lock.stat().st_ino
        result, _issues = checks.check_uncommitted_changes(_worker_env(repo))
        assert writer.lock.stat().st_ino == inode
        assert writer.finish() == 0, writer.stderr
    assert result["status"] != "error", result
    assert {"seed.txt", "first.txt"} <= set(_git(repo, "ls-files").split())


def test_worker_dirty_check_does_not_refresh_the_index(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks as checks

    # Production default; the suite's child-environment wrapper would otherwise pin "0".
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    repo = _repo(tmp_path)
    seed = repo / "seed.txt"
    info = seed.stat()
    os.utime(seed, ns=(info.st_atime_ns, info.st_mtime_ns - 10_000_000_000))  # same bytes, stale stat cache
    index = repo / ".git" / "index"
    before = index.read_bytes()

    assert checks.check_uncommitted_changes(_worker_env(repo)) == ({"status": "ok"}, 0)
    assert index.read_bytes() == before, "the worker's status took the optional index lock"


def test_generic_git_capture_leaves_a_live_writers_old_lock(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    blob = _git(repo, "hash-object", "-w", "--stdin", input="contender payload\n")
    with _live_writer(repo) as writer:
        _backdate(writer.lock)
        rc, _out, err = git_ops.git_capture(
            ["git", "update-index", "--add", "--cacheinfo", f"100644,{blob},contender.txt"])
        assert rc != 0 and "index.lock" in err
        assert writer.lock.exists()
        assert writer.finish() == 0, writer.stderr
    entries = set(_git(repo, "ls-files").split())
    assert "first.txt" in entries and "contender.txt" not in entries


@pytest.fixture
def managed_boot(tmp_path, monkeypatch):
    """Real bootstrap -> serialized restart -> safe_restart -> checkout_and_reset -> Git.

    Dependency sync, the import test and remotes are inert."""
    import server

    repo, data = _repo(tmp_path), tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    for name, value in {"REPO_DIR": repo, "DRIVE_ROOT": data, "BRANCH_DEV": "ouroboros",
                        "BRANCH_STABLE": "ouroboros-stable"}.items():
        monkeypatch.setattr(git_ops, name, value)
    monkeypatch.setattr(git_ops, "init", lambda **_kw: None)  # roots pinned above
    monkeypatch.setattr(git_ops, "update_state", lambda mutate, **_kw: mutate({}))
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda **_kw: (True, "fixture"))
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": True})
    for name, value in {"REPO_DIR": repo, "DATA_DIR": data, "_LAUNCHER_MANAGED": True,
                        "_LAUNCHER_MANAGED_REPO_DIR": str(repo)}.items():
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *_a: None)
    monkeypatch.setattr(server, "_runtime_branch_defaults", lambda: ("ouroboros", "ouroboros-stable"))
    monkeypatch.setattr(server, "_has_active_evolution_transaction", lambda: False)
    monkeypatch.delenv("OUROBOROS_DISABLE_MANAGED_UPDATES", raising=False)
    return SimpleNamespace(server=server, repo=repo, data=data)


@pytest.mark.parametrize("holder", ["crashed", "live"])
def test_bootstrap_returns_a_refused_checkout_and_keeps_the_lock(managed_boot, holder):
    boot = managed_boot
    head = _git(boot.repo, "rev-parse", "HEAD")
    with contextlib.ExitStack() as stack:
        writer = stack.enter_context(_live_writer(boot.repo)) if holder == "live" else None
        lock = writer.lock if writer else _orphan_lock(boot.repo)
        _backdate(lock)

        ok, message = boot.server._bootstrap_supervisor_repo({}, git_ops)

        assert ok is False and lock.exists()
        assert "git reset --hard HEAD" in message and "index.lock" in message
        assert "no checkout step had changed the working tree" in message  # Git refused the FIRST step
        assert writer is None or writer.finish() == 0
    assert _git(boot.repo, "rev-parse", "HEAD") == head
    assert holder == "crashed" or "first.txt" in _git(boot.repo, "ls-files").split()
    [row] = _rows(boot.data, "checkout_step_failed")
    assert row["reason"] == "bootstrap" and row["returncode"] != 0 and "index.lock" in row["stderr"]


@pytest.mark.parametrize("boot_kind", ["managed_updates_disabled", "local_dev"])
def test_a_held_lock_does_not_fail_a_bootstrap_that_writes_no_index(managed_boot, monkeypatch, boot_kind):
    lock = _orphan_lock(managed_boot.repo)
    if boot_kind == "managed_updates_disabled":
        monkeypatch.setenv("OUROBOROS_DISABLE_MANAGED_UPDATES", "1")
    else:
        monkeypatch.setattr(managed_boot.server, "_LAUNCHER_MANAGED", False)

    ok, message = managed_boot.server._bootstrap_supervisor_repo({}, git_ops)

    assert ok is True, message
    assert lock.exists()


def test_owner_restart_reports_a_refused_checkout_and_does_not_exit(managed_boot, monkeypatch):
    from ouroboros import server_restart

    def must_not_run(*_a, **_kw):
        raise AssertionError("a refused checkout must not proceed to the restart")

    monkeypatch.setattr(server_restart, "_stop_owned_work", must_not_run)
    monkeypatch.setattr(server_restart, "_request_restart_exit", must_not_run)
    with _live_writer(managed_boot.repo) as writer:
        ok, message = server_restart._perform_owner_restart(SimpleNamespace(safe_restart=git_ops.safe_restart))
        assert ok is False and "index.lock" in message
        assert writer.finish() == 0, writer.stderr
    assert "first.txt" in _git(managed_boot.repo, "ls-files").split()


def test_a_refused_restart_checkout_abandons_the_authorized_adoption(tmp_path, monkeypatch):
    """`bind_restart` abandons an adoption on a returned refusal; a raised one left it authorized."""
    from ouroboros import body_adoption, body_candidate
    from tests.body_candidate_support import isolate, make_ctx, make_serving, restart_receipt, rich_commit

    serving, data = make_serving(tmp_path), isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    ctx = make_ctx(serving, data, "root-refused-checkout")
    body_candidate.prepare(ctx)
    cand = rich_commit(ctx.repo_dir)
    body_candidate.record_reviewed_commit(ctx, cand)
    body_adoption.authorize(ctx, cand, reason="adopt it")
    restart_receipt(data, cand, "adopt it")
    for name, value in {"REPO_DIR": serving, "DRIVE_ROOT": data, "BRANCH_DEV": "ouroboros",
                        "BRANCH_STABLE": "ouroboros-stable"}.items():
        monkeypatch.setattr(git_ops, name, value)
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda **_kw: (True, "fixture"))
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": True})
    lock = _orphan_lock(serving)

    ok, message = body_adoption.bind_restart(git_ops.safe_restart, data, "adopt it")(
        reason="agent_restart_request", unsynced_policy="rescue_and_block")

    assert ok is False and "was not armed" in message and "index.lock" in message
    assert body_adoption.read(data) == {} and lock.exists()  # abandoned: archived, no live handoff


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX modes as a non-root user")
def test_an_unreadable_valid_index_is_reported_not_rebuilt(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    (repo / "staged.txt").write_text("unique staged intent\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    index = repo / ".git" / "index"
    original = index.read_bytes()
    index.chmod(0)
    try:
        rc, _out, err = git_ops.git_capture(["git", "status", "--porcelain"])
    finally:
        index.chmod(0o644)

    assert rc != 0 and "index file open failed" in err
    assert index.read_bytes() == original
    assert not list((repo / ".git").glob("index.corrupt.*.bak"))
    assert _git(repo, "diff", "--cached", "--name-only").split() == ["staged.txt"]


def test_a_corrupt_index_beside_a_held_lock_is_left_untouched_by_every_read(tmp_path, monkeypatch):
    """No backup, move or relink while the lock exists (a passive status read used to leave one
    more backup each time); Git's own failure returns, and the repair runs once the lock is gone."""
    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    index = repo / ".git" / "index"
    index.write_bytes(b"invalid index")
    inode = index.stat().st_ino
    lock = _orphan_lock(repo)

    for _read in range(3):
        rc, out, err = git_ops.git_capture(["git", "status", "--porcelain"])
        assert (rc != 0, out) == (True, "") and "index file smaller than expected" in err
    assert (index.read_bytes(), index.stat().st_ino) == (b"invalid index", inode)
    assert lock.read_bytes() == b"left by a crashed process\n"
    assert not list((repo / ".git").glob("index.corrupt.*.bak"))

    lock.unlink()  # the owner (or the agent) resolves the lock; the ordinary repair then proceeds
    assert git_ops.git_capture(["git", "status", "--porcelain"])[0] == 0
    assert _git(repo, "ls-files").split() == ["seed.txt"]
    [backup] = (repo / ".git").glob("index.corrupt.*.bak")
    assert backup.read_bytes() == b"invalid index"


@pytest.mark.parametrize("race", ["lock_taken_before_rebuild", "index_published_before_relink"])
def test_a_rebuild_refused_by_a_lock_taken_meanwhile_never_leaves_a_missing_or_clobbered_index(
        tmp_path, monkeypatch, race):
    """Retrying against the moved-aside index read every tracked file as deleted (a later commit
    recorded an empty tree). Another process takes the lock between the check and the rebuild:
    the original bytes come back beside their backup, and an index that process published in
    the meantime is never replaced."""
    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    index = repo / ".git" / "index"
    index.write_bytes(b"invalid index")
    published = b"index another writer published"
    real_warning = git_ops.log.warning

    def contender_moves(message, *args, **kwargs):
        if str(message).startswith("Backed up corrupt git index"):
            _orphan_lock(repo)  # Git now refuses the rebuild
            if race == "index_published_before_relink":
                index.write_bytes(published)
        return real_warning(message, *args, **kwargs)

    monkeypatch.setattr(git_ops.log, "warning", contender_moves)
    rc, out, err = git_ops.git_capture(["git", "status", "--porcelain"])

    assert (rc != 0, out) == (True, "") and "index file smaller than expected" in err  # the original failure
    assert index.read_bytes() == (published if race == "index_published_before_relink" else b"invalid index")
    [backup] = (repo / ".git").glob("index.corrupt.*.bak")
    assert backup.read_bytes() == b"invalid index"


def test_a_malformed_index_is_still_backed_up_and_rebuilt(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    (repo / ".git" / "index").write_bytes(b"invalid index")

    rc, _out, err = git_ops.git_capture(["git", "status", "--porcelain"])

    assert rc == 0, err
    [backup] = (repo / ".git").glob("index.corrupt.*.bak")
    assert backup.read_bytes() == b"invalid index"
    assert _git(repo, "ls-files").split() == ["seed.txt"]


# --------------------------------------------------------------------------- #
# A UI replace refused before its first tree step is abandoned as unapplied;
# every other refusal keeps the fenced rollback (and names Git's own words).
# --------------------------------------------------------------------------- #
def _commit(repo, message, files=(), delete=()):
    for rel, text in dict(files).items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    for rel in delete:
        (repo / rel).unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def replace_scene(tmp_path, monkeypatch):
    """The real update doors (``post``; ``apply`` is replace): preflight prepare (rescue, keep ref,
    intent), tx, checkout and the fenced rollback. Only the fetching plan, the worker fence/revival
    and the restart callback are recorded. The local branch carries one owner commit the target
    lacks, so a keep ref is made."""
    from ouroboros.gateway import control
    from supervisor import update_merge

    repo = _init_repo(tmp_path)
    root = _commit(repo, "root", {"BIBLE.md": "constitution\n", "ouroboros/store.py": "# store\n",
                                  "ouroboros/usage_store.py": "# usage\n", "seed.txt": "seed\n"})
    base = _commit(repo, "owner work", {"local.txt": "owner commit\n"})
    _git(repo, "checkout", "-q", "-b", "official", root)
    target = _commit(repo, "official release", {"official.txt": "release\n"})
    storeless = _commit(repo, "store-less release", delete=("ouroboros/store.py",))
    _git(repo, "checkout", "-q", "ouroboros")
    (repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(
        json.dumps({"managed_local_branch": "ouroboros"}), encoding="utf-8")
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    for name, value in {"REPO_DIR": repo, "DRIVE_ROOT": data, "BRANCH_DEV": "ouroboros"}.items():
        monkeypatch.setattr(git_ops, name, value)
    monkeypatch.setattr(git_ops, "update_state", lambda mutate, **_kw: mutate({}))
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda *_a: ("", "", "official"))
    events = {"fenced": [], "respawned": 0, "restarts": []}
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda _reason: [])
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update",
                        lambda: events.__setitem__("respawned", events["respawned"] + 1))
    real_fenced = control._rollback_fenced_update
    monkeypatch.setattr(control, "_rollback_fenced_update",
                        lambda reason, error, **extra: events["fenced"].append(reason) or real_fenced(reason, error, **extra))
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    app = Starlette(routes=[Route("/api/update/apply", control.api_update_apply, methods=["POST"])])
    app.state.request_restart = lambda **kw: events["restarts"].append(kw)
    scene = SimpleNamespace(repo=repo, data=data, base=base, target=target, storeless=storeless, events=events)

    def post(strategy, to, **plan_fields):
        plan = {"kind": "clean", "available": True, "base_sha": scene.base, "target_sha": to, "target_ref": "official",
                "update_channel": "stable", "local_dirty_count": 0, **plan_fields}
        monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **_kw: dict(plan))
        monkeypatch.setattr(git_ops, "_resolve_managed_update_target", lambda *_a: ("official", to, ""))
        with TestClient(app) as client:
            response = client.post("/api/update/apply", json={"strategy": strategy, "confirm_recovery": True,
                                                            "expected_base_sha": scene.base, "expected_target_sha": to})
        return response.status_code, response.json()

    scene.post, scene.apply = post, lambda to: post("replace", to)
    return scene


def test_a_replace_refused_at_its_first_tree_step_by_a_held_lock_is_abandoned_unapplied(replace_scene):
    """Not a gate_blocked trap: the tx and intent are cleared, writers reopen, Git's refusal is
    named, and the rescue snapshot and keep ref made before the checkout stay (and are reported)."""
    from supervisor import update_merge

    scene = replace_scene
    with _live_writer(scene.repo) as writer:
        inode = writer.lock.stat().st_ino
        status, body = scene.apply(scene.target)

        outcome = (status, body.get("reason"), body.get("abandoned"), body.get("rolled_back"))
        assert outcome == (409, "replace_not_applied", True, False), body
        assert "index.lock" in body["error"] and "File exists" in body["error"]
        assert scene.events == {"fenced": [], "respawned": 1, "restarts": []}
        assert update_merge.read_update_tx_strict()[0] == "absent" and not git_ops._read_update_intent()
        assert _git(scene.repo, "rev-parse", "HEAD") == scene.base and not (scene.repo / "official.txt").exists()
        assert writer.lock.stat().st_ino == inode
        assert writer.finish() == 0, writer.stderr
    assert "first.txt" in _git(scene.repo, "ls-files").split()
    assert body["keep_branch"] and _git(scene.repo, "rev-parse", body["keep_branch"]) == scene.base  # retained
    assert isinstance(body["rescue"], dict)  # the preflight rescue snapshot is retained and reported
    [row] = _rows(scene.data, "managed_update_replace_apply_abandoned")
    assert row["cause"] == "index_lock_refused" and row["cmd"] == "git reset --hard HEAD" and row["head"] == scene.base


@pytest.mark.parametrize("locked_after", ["git reset --hard HEAD", "git checkout -B"])
def test_a_lock_that_refuses_a_later_tree_step_keeps_the_fenced_rollback(replace_scene, monkeypatch, locked_after):
    """A refusal after an earlier tree step ran proves nothing about the tree: no abandon. The
    fenced rollback then reads its own target: still present after a mere `reset --hard HEAD`
    (rolled back in place), gone once `checkout -B` moved HEAD (the same lock refuses the
    rollback, the transaction stays gate_blocked, and Git restores it once the lock is gone)."""
    from supervisor import update_merge

    scene, real = replace_scene, git_ops._run_git_resilient

    def lock_between_steps(cmd, **kwargs):
        result = real(cmd, **kwargs)
        if " ".join(cmd).startswith(locked_after) and not (scene.repo / ".git" / "index.lock").exists():
            _orphan_lock(scene.repo)  # another process takes the lock between two checkout steps
        return result

    monkeypatch.setattr(git_ops, "_run_git_resilient", lock_between_steps)
    status, body = scene.apply(scene.target)

    assert scene.events["fenced"] == ["replace_checkout_exception"] and "abandoned" not in body
    assert "index.lock" in body["error"] and "earlier checkout steps may already have" in body["error"]
    assert not _rows(scene.data, "managed_update_replace_apply_abandoned")
    if locked_after == "git reset --hard HEAD":
        assert (status, body["rolled_back"]) == (409, True) and "needed no reset/checkout" in body["rollback"]
        assert update_merge.read_update_tx_strict()[0] == "absent" and scene.events["respawned"] == 1
        assert _git(scene.repo, "rev-parse", "HEAD") == scene.base
        return
    assert status == 500 and body["restart_required"] is True  # the same lock refuses the rollback
    assert update_merge.read_update_tx_strict()[1]["phase"] == update_merge.GATE_BLOCKED_PHASE
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.target  # a moved checkout is never called restored
    assert not _rows(scene.data, "managed_update_rolled_back")

    (scene.repo / ".git" / "index.lock").unlink()  # its holder is gone; the next boot's recovery can work
    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert result["rolled_back"] is True and _git(scene.repo, "rev-parse", "HEAD") == scene.base
    [row] = _rows(scene.data, "managed_update_rolled_back")
    assert "checkout_already_restored" not in row  # Git did the restoring this time


def test_a_store_switch_deferral_moves_nothing_and_is_untouched_by_a_held_lock(replace_scene):
    from ouroboros import store_switch
    from supervisor import update_merge

    scene = replace_scene
    with _live_writer(scene.repo) as writer:
        status, body = scene.apply(scene.storeless)
        assert (status, body["status"]) == (200, "ok") and scene.events["fenced"] == []
        assert update_merge.read_update_tx_strict()[1]["phase"] == "pending_boot_smoke"
        assert store_switch.read_switch_intent(scene.data)["target_sha"] == scene.storeless
        assert _git(scene.repo, "rev-parse", "HEAD") == scene.base and writer.lock.exists()
        assert writer.finish() == 0, writer.stderr
    assert len(scene.events["restarts"]) == 1


@pytest.mark.parametrize("case", ["every_attempt_refused", "mixed_attempts", "foreign_lock", "not_a_locking_command",
                                  "repair_between"])
def test_only_a_lock_refusal_on_every_attempt_of_a_locking_command_is_typed(tmp_path, monkeypatch, case):
    from supervisor import git_ops_reset

    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    owner = tmp_path / "other" if case == "foreign_lock" else repo
    refusal = f"fatal: Unable to create '{owner}/.git/index.lock': File exists.\n\nAnother git process seems to be running"
    results = [(1, "error: unrelated failure")] if case == "mixed_attempts" else []
    results += [(128, refusal)] * (5 - len(results))
    cmd = ["git", "clean", "-fd"] if case == "not_a_locking_command" else ["git", "reset", "--hard", "HEAD"]
    def fake_run(argv, **_kw):
        returncode, stderr = results.pop(0)
        return subprocess.CompletedProcess(argv, returncode, "", stderr)

    monkeypatch.setattr(git_ops_reset.subprocess, "run", fake_run)
    repaired = iter([case == "repair_between"])
    monkeypatch.setattr(git_ops, "_maybe_repair_git_index", lambda *_a, **_k: next(repaired, False))

    with pytest.raises(subprocess.CalledProcessError) as raised:
        git_ops._run_git_resilient(cmd, cwd=str(repo), check=True)

    expected = git_ops_reset.IndexLockRefusal if case == "every_attempt_refused" else subprocess.CalledProcessError
    assert type(raised.value) is expected and results == []
    assert getattr(raised.value, "before_tree_effects", False) is False  # only checkout's first step sets it


@pytest.mark.parametrize("owner_chat", [42, 0])
def test_a_failed_bootstrap_is_told_once_in_the_owner_chat_and_a_deliberate_deferral_is_not(managed_boot, monkeypatch, owner_chat):
    from supervisor import evolution_lifecycle, message_bus, state, update_merge

    sent = []
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": owner_chat})
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat, text, **kw: sent.append((chat, text, kw)))
    lock = _orphan_lock(managed_boot.repo)

    ok, message = managed_boot.server._bootstrap_supervisor_repo({}, git_ops)

    assert ok is False and len(sent) == 1
    chat, text, kw = sent[0]
    assert (chat, kw) == (owner_chat, {"role": "system", "system_type": "startup_notice"})
    # One owner line: the step, Git's own first line and what is known of the tree. Git's whole
    # output stays in the returned (and logged) message and the checkout_step_failed row.
    headline, full = message.split("\n")
    assert text == f"Startup bootstrap failed: {headline}"  # no decorative glyph to strip at render
    assert "index.lock" in headline and "no checkout step had changed the working tree" in headline
    assert full.startswith("Git: fatal: Unable to create")
    [row] = _rows(managed_boot.data, "checkout_step_failed")
    assert full == f"Git: {row['stderr']}"

    sent.clear()
    lock.unlink()
    monkeypatch.setattr(evolution_lifecycle, "disable_evolution_projection", lambda: None)
    monkeypatch.setattr(evolution_lifecycle, "pause_evolution_campaign", lambda *_a: None)
    head = _git(managed_boot.repo, "rev-parse", "HEAD")
    update_merge.write_update_tx({"phase": "assisted_resolution", "task_id": "resolver",
                                  "pre_update_sha": head, "pre_update_branch": "ouroboros"})
    ok, message = managed_boot.server._bootstrap_supervisor_repo({}, git_ops)
    assert ok is False and "still being resolved" in message and sent == []  # deliberate deferral

    update_merge.clear_update_tx()
    assert managed_boot.server._bootstrap_supervisor_repo({}, git_ops)[0] is True and sent == []

    # A reset blocked during an Evolution transaction kept the owner's files on purpose: not "failed".
    monkeypatch.setattr(managed_boot.server, "_has_active_evolution_transaction", lambda: True)
    (managed_boot.repo / "owner-draft.txt").write_text("unsaved\n", encoding="utf-8")
    ok, message = managed_boot.server._bootstrap_supervisor_repo({}, git_ops)
    assert ok is False and message.startswith("Failed checkout ouroboros: Reset blocked (")
    assert [text for _chat, text, _kw in sent] == [f"Startup kept the current files: {message.partition(': ')[2]}"]
    assert (managed_boot.repo / "owner-draft.txt").exists()


@pytest.mark.parametrize("owner_chat", [None, "", -5, "not-an-id"])
def test_bootstrap_notice_never_invents_an_owner_address(monkeypatch, owner_chat):
    from ouroboros.server_restart import _noticed_bootstrap
    from supervisor import message_bus, state

    sent = []
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": owner_chat})
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *a, **kw: sent.append((a, kw)))
    assert _noticed_bootstrap(lambda **kw: (False, "Git refusal"))() == (False, "Git refusal")
    assert sent == []


def test_landed_update_keeps_its_own_boot_authority_after_refused_bootstrap(managed_boot, monkeypatch):
    """Real bootstrap and boot-finalizer consumers; deps/import/native probing and feed refresh are inert.

    A usable supervisor plus the landed update identity must not initiate lock-caused
    rollback. Adoption still observes the unsuccessful bootstrap separately.
    """
    from ouroboros import body_adoption, update_letter
    from supervisor import update_merge

    boot = managed_boot
    head = _git(boot.repo, "rev-parse", "HEAD")
    lock = _orphan_lock(boot.repo)
    ok, message = boot.server._bootstrap_supervisor_repo({}, git_ops)
    assert ok is False and "index.lock" in message
    monkeypatch.setattr(boot.server, "_bootstrap_ok", ok)
    monkeypatch.setattr(boot.server, "_wait_for_supervisor_update_finalize", lambda: True)
    monkeypatch.setattr(git_ops, "compute_managed_update_status", lambda **kw: {})
    monkeypatch.setattr(update_letter, "refresh_after_check", lambda *a, **kw: None)
    monkeypatch.setattr(boot.server, "broadcast_ws_sync", lambda *a: None)
    monkeypatch.setattr(update_merge, "external_host_restart_smoke", lambda: {"ok": True})
    adoption = []
    monkeypatch.setattr(body_adoption, "settle_on_boot", lambda *a, **kw: adoption.append(kw))
    update_merge.write_update_tx({"phase": "pending_boot_smoke", "merge_commit": head,
                                  "pre_update_sha": head, "pre_update_branch": "ouroboros"})

    boot.server._boot_managed_update_tasks()

    assert update_merge.read_update_tx_strict()[0] == "absent"
    assert adoption == [{"supervisor_ready": False}]
    assert lock.read_bytes() == b"left by a crashed process\n"
    assert not _rows(boot.data, "managed_update_rolled_back")
    assert not _rows(boot.data, "managed_update_rollback_after_failed_boot")


def test_deferred_replace_records_git_refusal_after_export_and_retirement(replace_scene, monkeypatch):
    """The store-less target defers checkout until exit; its real owner still owes Git's cause."""
    from ouroboros import store, store_switch
    from ouroboros.store_import import ensure_current_store
    from supervisor import message_bus

    scene = replace_scene
    ensure_current_store(scene.data)
    assert store.store_ready(scene.data)
    lock = _orphan_lock(scene.repo)
    status, body = scene.apply(scene.storeless)
    assert status == 200 and body["status"] == "ok"
    outcome = store_switch.run_switch_seam(scene.data, repo_dir=scene.repo)
    assert outcome["status"] == "switch_failed" and "index.lock" in outcome["message"]
    receipt = json.loads((scene.data / store_switch.RECEIPT_REL).read_text(encoding="utf-8"))
    assert receipt["export"]["status"] == "completed" and receipt["retired"]
    assert receipt["switched"]["ok"] is False and "index.lock" in receipt["switched"]["message"]
    assert store_switch.read_switch_intent(scene.data) is None
    assert lock.exists() and _git(scene.repo, "rev-parse", "HEAD") == scene.base
    # The current generation re-imports its completed export; the old intent is not replayed.
    assert store_switch.resume_pending_switch(scene.data, repo_dir=scene.repo)["status"] == "nothing_pending"
    ensure_current_store(scene.data)
    sent = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat, text, **kw: sent.append((chat, text, kw)))
    store_switch.report_outcomes(scene.data, 42)
    assert len(sent) == 1 and "index.lock" in sent[0][1]
    assert sent[0][2]["system_type"] == "storage_switch_notice"


# --------------------------------------------------------------------------- #
# An ordinary Update meets a held lock at its first tree step: the fenced rollback
# reads its whole target, finds it present and runs no Git command over it, so the
# core converses and its Restart is no longer deferred (that restart's own reset still
# meets the lock); anything short of that target stays fenced.
# --------------------------------------------------------------------------- #
_LANE_ROLLBACK = {"auto_merge": "merge_apply_failed", "assisted": "assisted_materialize_failed"}


@pytest.fixture
def update_lanes(replace_scene, monkeypatch):
    """Both smart-update lanes through the real door, Git and writer-admission latch; model
    budget, the resolver's boot, the owner-control preload and the smokes are inert."""
    from ouroboros.tools import review_admission
    from supervisor import state, update_merge, worker_chat_lane, workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", replace_scene.data)
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")  # restored for the next test
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 10.0)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(review_admission, "managed_update_wave_estimate", lambda _room: {"type": "wave_fixture"})
    monkeypatch.setattr(update_merge, "ensure_assisted_resolver_ready", lambda *_a, **_k: True)
    monkeypatch.setattr(update_merge, "enqueue_assisted_resolution_task", lambda _tx: True)
    monkeypatch.setattr(worker_chat_lane, "preload_owner_control_path", lambda: None)
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": True})
    scene = replace_scene

    def post(lane):
        fields = ({"merge_commit": scene.target} if lane == "auto_merge"
                  else {"kind": "conflicting", "code_conflict_paths": ["official.txt"]})
        return scene.post(lane, scene.target, local_snapshot=scene.base, **fields)

    return post


@pytest.mark.parametrize("holder", ["crashed", "live"])
@pytest.mark.parametrize("lane", ["auto_merge", "assisted"])
def test_an_update_refused_at_its_first_step_rolls_back_in_place_and_keeps_the_core_usable(
        replace_scene, update_lanes, lane, holder):
    from ouroboros import server_restart
    from supervisor import update_merge, worker_chat_lane, workers

    scene = replace_scene
    with contextlib.ExitStack() as stack:
        writer = stack.enter_context(_live_writer(scene.repo)) if holder == "live" else None
        lock = writer.lock if writer else _orphan_lock(scene.repo)
        inode = lock.stat().st_ino
        status, body = update_lanes(lane)

        assert (status, body.get("rolled_back"), body.get("restart_required")) == (409, True, None), body
        assert "index.lock" in body["error"] and "needed no reset/checkout" in body["rollback"]
        assert scene.events["fenced"] == [_LANE_ROLLBACK[lane]] and scene.events["respawned"] == 1
        assert update_merge.read_update_tx_strict()[0] == "absent"
        assert workers.repo_writer_admission_closed() == ""  # writers are admitted again...
        assert worker_chat_lane.owner_conversation_admitted(42)  # ...and so is the owner's Main turn
        assert server_restart._safe_restart_serialized(  # the in-app Restart is no longer deferred
            lambda **_kw: (True, "restarting"), reason="owner_restart", unsynced_policy="rescue_and_reset",
        ) == (True, "restarting")
        assert _git(scene.repo, "rev-parse", "HEAD") == scene.base and not (scene.repo / "official.txt").exists()
        assert lock.stat().st_ino == inode  # nobody removed or replaced the lock
        assert writer is None or writer.finish() == 0, writer.stderr
    assert holder == "crashed" or "first.txt" in _git(scene.repo, "ls-files").split()  # the live write landed
    [row] = _rows(scene.data, "managed_update_rolled_back")
    assert row["checkout_already_restored"] is True and row["pre_update_sha"] == scene.base
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["reason"] == "no pending update"


@pytest.mark.parametrize("lane", ["auto_merge", "assisted"])
def test_a_matching_head_over_a_changed_tree_is_not_taken_for_the_restored_target(
        replace_scene, update_lanes, monkeypatch, lane):
    """HEAD alone proves nothing: a file that appeared beside the held lock keeps the transaction
    gate_blocked with Git's refusal and stays on disk; once the lock is gone, boot recovery rescues
    it and restores the target with Git."""
    from supervisor import update_merge

    scene, real = replace_scene, git_ops.git_capture
    late = scene.repo / "late-edit.txt"

    def edit_and_lock_at_the_first_step(cmd, **kwargs):
        if cmd == ["git", "reset", "--hard", "HEAD"] and not late.exists():
            late.write_text("written while the update ran\n", encoding="utf-8")
            _orphan_lock(scene.repo)
        return real(cmd, **kwargs)

    monkeypatch.setattr(git_ops, "git_capture", edit_and_lock_at_the_first_step)
    status, body = update_lanes(lane)

    assert (status, body["rolled_back"], body["restart_required"]) == (500, False, True), body
    assert body["rollback"].startswith("rollback reset failed before checkout: ") and "index.lock" in body["rollback"]
    assert update_merge.read_update_tx_strict()[1]["phase"] == update_merge.GATE_BLOCKED_PHASE
    assert late.exists() and _git(scene.repo, "rev-parse", "HEAD") == scene.base
    assert not _rows(scene.data, "managed_update_rolled_back")

    monkeypatch.setattr(git_ops, "git_capture", real)
    (scene.repo / ".git" / "index.lock").unlink()
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["rolled_back"] is True
    [row] = _rows(scene.data, "managed_update_rolled_back")
    assert "checkout_already_restored" not in row and row.get("rescue_path")  # rescued, then Git restored
    assert not late.exists() and _git(scene.repo, "rev-parse", "HEAD") == scene.base


def test_owner_work_beside_a_held_lock_is_refused_before_anything_moves(replace_scene, update_lanes):
    """The shared stash prologue needs the lock first: Git's refusal returns, no transaction is left,
    writers revive and the owner's edit stays where it was."""
    from supervisor import update_merge, workers

    scene = replace_scene
    (scene.repo / "seed.txt").write_text("owner edit in progress\n", encoding="utf-8")
    lock = _orphan_lock(scene.repo)
    status, body = update_lanes("auto_merge")

    error = body["error"].partition("could not preserve local changes before the update: ")
    # Git's own words, or the fallback when Git said nothing (Apple Git 2.39 refuses this push silently).
    assert status == 409 and error[1] and (
        "index.lock" in error[2] or error[2] in {"git stash push failed", "error: could not write index"}
    ), body
    assert update_merge.read_update_tx_strict()[0] == "absent" and scene.events["respawned"] == 1
    assert workers.repo_writer_admission_closed() == "" and scene.events["fenced"] == []
    assert (scene.repo / "seed.txt").read_text(encoding="utf-8") == "owner edit in progress\n"
    assert lock.read_bytes() == b"left by a crashed process\n" and _git(scene.repo, "rev-parse", "HEAD") == scene.base


@pytest.mark.parametrize("lock_at_next_boot", ["released", "still_held"])
def test_owner_work_stashed_before_a_lock_appeared_waits_gated_and_returns_once_git_can_apply_it(
        replace_scene, update_lanes, monkeypatch, lock_at_next_boot):
    """The lock arrives after `git stash push`: the rollback target is present (the stash took the
    owner's edits), but applying the stash needs the lock. The restore does not start, so its
    one-shot marker is not spent on Git's refusal: the work stays in its recorded stash and pin and
    the transaction keeps the writer gate and the in-app Restart closed until a boot finds the lock
    gone, which then restores the work through the same marker owner and finishes the rollback."""
    from ouroboros import server_restart
    from supervisor import update_merge, worker_chat_lane, workers

    scene, real = replace_scene, git_ops.git_capture
    (scene.repo / "seed.txt").write_text("owner edit in progress\n", encoding="utf-8")
    (scene.repo / "draft.txt").write_text("untracked owner draft\n", encoding="utf-8")

    def lock_after_the_stash(cmd, **kwargs):
        result = real(cmd, **kwargs)
        if cmd[:3] == ["git", "stash", "push"] and result[0] == 0:
            _orphan_lock(scene.repo)
        return result

    def held_until_safe():
        tx = update_merge.read_update_tx_strict()[1]
        assert tx["phase"] in {update_merge.GATE_BLOCKED_PHASE, "rolling_back"} and "stash_restore" not in tx
        assert tx["stash_sha"] == stash and _git(scene.repo, "rev-parse", f"rescue-local-{stash[:12]}") == stash
        assert _git(scene.repo, "show", f"{stash}:seed.txt") == "owner edit in progress"
        assert (scene.repo / "seed.txt").read_text(encoding="utf-8") == "seed\n"
        assert not (scene.repo / "draft.txt").exists()
        gate = workers.repo_writer_admission_closed()
        assert gate.startswith("managed_update_tx:") and not worker_chat_lane.conversation_admitted_during_update(gate)
        assert server_restart._safe_restart_serialized(
            lambda **_kw: (True, "restarting"), reason="owner_restart", unsynced_policy="rescue_and_reset")[0] is False
        assert not _rows(scene.data, "managed_update_rolled_back")

    monkeypatch.setattr(git_ops, "git_capture", lock_after_the_stash)
    status, body = update_lanes("auto_merge")
    monkeypatch.setattr(git_ops, "git_capture", real)
    lock = scene.repo / ".git" / "index.lock"
    stash = update_merge.read_update_tx_strict()[1]["stash_sha"]

    assert (status, body["rolled_back"], body["restart_required"]) == (500, False, True), body
    assert "index lock is held" in body["stash_note"] and stash[:12] in body["stash_note"]
    assert "quit and reopen the app" in body["stash_note"] and "git stash apply" not in body["stash_note"]
    held_until_safe()
    if lock_at_next_boot == "still_held":  # every boot that still meets the lock waits the same way
        result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
        assert result["rolled_back"] is False and "index lock is held" in result["msg"], result
        held_until_safe()
        assert lock.read_bytes() == b"left by a crashed process\n"
    lock.unlink()  # its holder is gone

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert result["rolled_back"] is True and "local changes restored" in result["msg"], result
    [row] = _rows(scene.data, "managed_update_rolled_back")
    assert row["stash_restore_status"] == "restored" and row["checkout_already_restored"] is True
    assert (scene.repo / "seed.txt").read_text(encoding="utf-8") == "owner edit in progress\n"
    assert (scene.repo / "draft.txt").read_text(encoding="utf-8") == "untracked owner draft\n"
    assert update_merge.read_update_tx_strict()[1]["phase"] == update_merge.MARKER_CLEANUP_RETRY_PHASE
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["finalized"] is True
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")  # the restored process's own latch
    assert update_merge.read_update_tx_strict()[0] == "absent" and workers.repo_writer_admission_closed() == ""
    assert _git(scene.repo, "rev-parse", stash) == stash  # an update stash is never dropped automatically
    assert _git(scene.repo, "rev-parse", f"rescue-local-{stash[:12]}") == stash


@pytest.mark.parametrize("lane", ["auto_merge", "assisted"])
def test_without_a_lock_both_update_lanes_still_land(replace_scene, update_lanes, lane):
    from supervisor import update_merge

    scene = replace_scene
    status, body = update_lanes(lane)

    assert scene.events["fenced"] == [] and not _rows(scene.data, "managed_update_rolled_back")
    tx = update_merge.read_update_tx_strict()[1]
    if lane == "auto_merge":
        assert (status, body["status"], body["restarting"], tx["phase"]) == (200, "ok", True, "pending_boot_smoke")
        assert _git(scene.repo, "rev-parse", "HEAD") == scene.target and len(scene.events["restarts"]) == 1
    else:
        assert (status, body["status"], tx["phase"]) == (200, "assisted_started", "assisted_resolution")
        assert _git(scene.repo, "rev-parse", "MERGE_HEAD") == scene.target


# --------------------------------------------------------------------------- #
# A store-less target moves only through the store switch. When the switch failed
# (or was refused), the next boot neither replays it outside the seam nor finalizes
# a target that never booted, whether or not the lock is still held.
# --------------------------------------------------------------------------- #
def _next_boot(scene, monkeypatch):
    """The next process over the same roots: the store's boot door, the managed bootstrap (real
    Git) and the boot finalizer with passing smokes; deps, import test and remotes are inert."""
    import server
    from ouroboros import store_switch
    from ouroboros.store_import import ensure_current_store
    from supervisor import update_merge, workers

    store_switch.resume_pending_switch(scene.data, repo_dir=scene.repo)
    ensure_current_store(scene.data)
    for name, value in {"REPO_DIR": scene.repo, "DATA_DIR": scene.data, "_LAUNCHER_MANAGED": True,
                        "_LAUNCHER_MANAGED_REPO_DIR": str(scene.repo)}.items():
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *_a: None)
    monkeypatch.setattr(server, "_runtime_branch_defaults", lambda: ("ouroboros", "ouroboros-stable"))
    monkeypatch.setattr(server, "_has_active_evolution_transaction", lambda: False)
    monkeypatch.setattr(git_ops, "init", lambda **_kw: None)
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda **_kw: (True, "fixture"))
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": True})
    monkeypatch.setattr(workers, "DRIVE_ROOT", scene.data)
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": True})
    monkeypatch.setattr(update_merge, "external_host_restart_smoke", lambda: {"ok": True})
    booted = server._bootstrap_supervisor_repo({}, git_ops)
    return booted, update_merge.finalize_managed_update_on_boot(supervisor_ready=True)


@pytest.mark.parametrize("lock_at_next_boot", ["released", "still_held"])
@pytest.mark.parametrize("history", ["side_release", "older_ancestor"])
@pytest.mark.parametrize("seam", ["switch_failed", "refused", "switch_not_performed"])
def test_a_store_less_target_the_switch_did_not_reach_is_never_replayed_or_finalized(
        replace_scene, monkeypatch, seam, history, lock_at_next_boot):
    from ouroboros import store_switch
    from ouroboros.store_import import ensure_current_store
    from supervisor import message_bus, state, update_merge

    scene = replace_scene
    if history == "older_ancestor":  # a real downgrade: the running checkout descends from the old release
        _git(scene.repo, "checkout", "-q", "-B", "ouroboros", scene.storeless)
        _commit(scene.repo, "the store arrives", {"ouroboros/store.py": "# store\n"})
        scene.base = _commit(scene.repo, "owner work", {"local.txt": "owner commit\n"})
    ensure_current_store(scene.data)
    lock = _orphan_lock(scene.repo)
    status, body = scene.apply(scene.storeless)
    assert (status, body["status"]) == (200, "ok"), body
    if seam == "switch_not_performed":  # the process died after the intent was consumed, before its owner ran
        def died(_intent):
            raise KeyboardInterrupt("stopped after consuming the intent")

        with pytest.raises(KeyboardInterrupt):
            store_switch.run_switch_seam(scene.data, repo_dir=scene.repo, owner=died)
    else:
        outcome = store_switch.run_switch_seam(scene.data, repo_dir=scene.repo,
                                               blockers=["worker:42"] if seam == "refused" else ())
        assert outcome["status"] == seam, outcome
    assert git_ops._read_update_intent()["target_sha"] == scene.storeless  # still armed past the restart exit
    if lock_at_next_boot == "released":
        lock.unlink()  # e.g. the owner followed Git's advice before reopening the app
    sent = []
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr(message_bus, "send_with_budget",
                        lambda _chat, text, **kw: sent.append((kw["system_type"], text)))

    (ok, message), result = _next_boot(scene, monkeypatch)

    assert _git(scene.repo, "rev-parse", "HEAD") == scene.base  # never the store-less target
    assert ok is (lock_at_next_boot == "released"), message
    assert (result.get("finalized"), result.get("rolled_back")) == (False, True), result
    assert not _rows(scene.data, "managed_update_finalized") and not git_ops._read_update_intent()
    [row] = _rows(scene.data, "managed_update_rolled_back")
    assert row["checkout_already_restored"] is True and row["pre_update_sha"] == scene.base
    assert update_merge.read_update_tx_strict()[1]["phase"] == update_merge.MARKER_CLEANUP_RETRY_PHASE
    # The restored generation only clears its handoff; the owner learns the switch did not happen.
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["finalized"] is True
    assert update_merge.read_update_tx_strict()[0] == "absent"
    store_switch.report_outcomes(scene.data, 7)
    [switch_notice] = [text for kind, text in sent if kind == "storage_switch_notice"]
    assert "did not happen" in switch_notice and "\n" not in switch_notice
    assert seam != "switch_failed" or ("index.lock" in switch_notice and "Another git process" not in switch_notice)
    startup = [text for kind, text in sent if kind == "startup_notice"]
    assert startup == ([] if ok else [f"Startup bootstrap failed: {message.splitlines()[0]}"])
    assert lock.exists() is (lock_at_next_boot == "still_held")


def test_the_store_switch_itself_still_reaches_a_store_less_target(replace_scene):
    from ouroboros import store_switch
    from ouroboros.store_import import ensure_current_store

    scene = replace_scene
    ensure_current_store(scene.data)
    assert scene.apply(scene.storeless)[0] == 200
    assert store_switch.run_switch_seam(scene.data, repo_dir=scene.repo)["status"] == "switched"
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.storeless


def test_bootstrap_still_replays_an_armed_intent_whose_target_has_the_store(replace_scene, monkeypatch):
    """The replay guard is narrow: a replace that armed its intent and died before its checkout
    still lands a target WITH the store at the next bootstrap and finalizes after its smokes."""
    from supervisor import update_merge

    scene = replace_scene
    monkeypatch.setattr(git_ops, "_resolve_managed_update_target", lambda *_a: ("official", scene.target, ""))
    ok, payload = git_ops.prepare_managed_update("replace", expected_base_sha=scene.base,
                                                 expected_target_sha=scene.target)
    assert ok, payload
    update_merge.write_update_tx({"phase": "applying_replace", "pre_update_sha": scene.base,
                                  "pre_update_branch": "ouroboros", "target_sha": scene.target,
                                  "merge_commit": scene.target, "strategy": "replace"})

    (ok, message), result = _next_boot(scene, monkeypatch)

    assert ok, message
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.target and result["finalized"] is True, result


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("operation", ["cherry-pick", "revert"])
def test_a_clean_index_with_an_unfinished_operation_is_not_a_finished_rollback(
        replace_scene, update_lanes, held, operation):
    """An empty operation leaves a clean index but a real Git continuation; reset must not be skipped."""
    from supervisor import update_merge

    scene = replace_scene
    if operation == "revert":
        to_reverse = _commit(scene.repo, "second", {"seed.txt": "second\n"})
        _commit(scene.repo, "third", {"seed.txt": "third\n"})
    else:
        _git(scene.repo, "commit", "--allow-empty", "-qm", "empty")
        to_reverse = "HEAD"
    scene.base = _git(scene.repo, "rev-parse", "HEAD")
    result = subprocess.run(["git", "-C", str(scene.repo), operation, to_reverse], capture_output=True)
    assert result.returncode == 1
    if operation == "revert":
        _git(scene.repo, "checkout", "HEAD", "--", "seed.txt")  # resolved to the current tree, not committed
    marker = scene.repo / ".git" / ("CHERRY_PICK_HEAD" if operation == "cherry-pick" else "REVERT_HEAD")
    assert marker.exists() and _git(scene.repo, "status", "--porcelain") == ""
    lock = _orphan_lock(scene.repo) if held else None
    update_merge.write_update_tx({"phase": "gate_blocked", "pre_update_sha": scene.base,
                                  "pre_update_branch": "ouroboros", "target_sha": scene.target})
    ok, message = update_merge.rollback_managed_update(reason="unfinished-operation")
    if held:
        assert not ok, message
        assert marker.exists() and lock.exists()
        assert not _rows(scene.data, "managed_update_rolled_back")
    else:
        assert ok, message
        assert not marker.exists()
        [row] = _rows(scene.data, "managed_update_rolled_back")
        assert not row.get("checkout_already_restored")


@pytest.mark.parametrize("operation", ["cherry-pick", "rebase"])
def test_a_stopped_sequence_or_rebase_denies_only_the_skip_not_the_restored_target(replace_scene, operation):
    """Reset ends an empty cherry-pick or revert but leaves a stopped multi-commit sequence or a
    rebase in place; such a marker forces Git's restore, and the target check after it does not
    require the marker gone (each boot would otherwise fail the same unnamed verification)."""
    from supervisor import update_merge

    scene = replace_scene
    _git(scene.repo, "checkout", "-q", "-b", "side", scene.base)
    _commit(scene.repo, "side one", {"seed.txt": "side\n"})
    picks = [_git(scene.repo, "rev-parse", "HEAD"), _commit(scene.repo, "side two", {"side.txt": "two\n"})]
    _git(scene.repo, "checkout", "-q", "ouroboros")
    pre = _commit(scene.repo, "ours", {"seed.txt": "ours\n"})
    args = ["cherry-pick", *picks] if operation == "cherry-pick" else ["rebase", "side"]
    assert subprocess.run(["git", "-C", str(scene.repo), *args], capture_output=True).returncode != 0
    marker = scene.repo / ".git" / ("sequencer" if operation == "cherry-pick" else "rebase-merge")
    assert marker.is_dir()
    update_merge.write_update_tx({"phase": "gate_blocked", "pre_update_sha": pre,
                                  "pre_update_branch": "ouroboros", "target_sha": scene.target})

    ok, message = update_merge.rollback_managed_update(reason="stopped-operation")

    assert ok, message
    assert _git(scene.repo, "rev-parse", "HEAD") == pre and _git(scene.repo, "status", "--porcelain") == ""
    assert _git(scene.repo, "rev-parse", "--abbrev-ref", "HEAD") == "ouroboros"
    assert marker.is_dir()  # Git's reset left it, as it always did
    [row] = _rows(scene.data, "managed_update_rolled_back")
    assert not row.get("checkout_already_restored")


# --------------------------------------------------------------------------- #
# Restore (the UI button; the agent's vcs_rollback) runs one reset after its keep
# branch and rescue. Git refusing that reset at the held index lock means the reset
# changed nothing: no restart, and the aborted-update readmission decides, under its
# own guards, whether writers return. Every other failure may have moved the tree
# and stays fenced for a restart.
# --------------------------------------------------------------------------- #
@pytest.fixture
def restore_door(tmp_path, monkeypatch):
    """``rollback_through_door`` over real Git, the real repo-mutation lock and the real writer
    gate; draining, killing and pool start are recorded, and saved-work handoff succeeds."""
    from ouroboros import process_custody
    from ouroboros.gateway import control
    from ouroboros.tools import services
    from supervisor import restart_retention, workers

    repo = _init_repo(tmp_path)
    storeless = _commit(repo, "before the store", {"BIBLE.md": "constitution\n", "seed.txt": "storeless\n",
                                                   "ouroboros/usage_store.py": "# usage\n"})
    old = _commit(repo, "the store arrives", {"ouroboros/store.py": "# store\n", "seed.txt": "old\n"})
    head = _commit(repo, "owner work", {"seed.txt": "new\n", "new.txt": "new\n"})
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    for name, value in {"REPO_DIR": repo, "DRIVE_ROOT": data, "BRANCH_DEV": "ouroboros"}.items():
        monkeypatch.setattr(git_ops, name, value)
    monkeypatch.setattr(git_ops, "update_state", lambda mutate, **_kw: mutate({}))
    for name, value in {"DRIVE_ROOT": data, "_repo_writer_gate_reason": "", "RUNNING": {}, "WORKERS": {},
                        "PENDING": []}.items():
        monkeypatch.setattr(workers, name, value)
    events = {"pool": [], "restarts": [], "gate_at_reset": []}
    monkeypatch.setattr(workers, "drain_repo_writers", lambda: [])
    monkeypatch.setattr(workers, "kill_workers_for_update", lambda **_kw: [])
    monkeypatch.setattr(services, "kill_all_services", lambda *_a, **_kw: [])
    monkeypatch.setattr(process_custody, "quiesce_custodied_services", lambda *_a: (True, []))
    monkeypatch.setattr(restart_retention, "_update_returns", {})  # the stop's captured cohort, this test's only
    monkeypatch.setattr(restart_retention, "recover_aborted_update", lambda *_a: True)
    monkeypatch.setattr(workers, "ensure_worker_pool_started", lambda **kw: events["pool"].append(kw) or True)
    real_capture = git_ops.git_capture

    def observed(cmd, **kwargs):
        if cmd[:3] == ["git", "reset", "--hard"]:
            events["gate_at_reset"].append(workers.repo_writer_admission_closed())
        return real_capture(cmd, **kwargs)

    monkeypatch.setattr(git_ops, "git_capture", observed)

    def restore(target, door="ui_rollback"):
        return control.rollback_through_door(
            target, door=door, request_restart=lambda: events["restarts"].append(target) or True)

    return SimpleNamespace(repo=repo, data=data, storeless=storeless, old=old, head=head, events=events,
                           restore=restore)


@pytest.mark.parametrize("door", ["ui_rollback", "vcs_rollback"])
@pytest.mark.parametrize("holder", ["crashed", "live"])
def test_a_restore_git_refused_at_the_index_lock_returns_writers_without_a_restart(restore_door, holder, door):
    from supervisor import worker_chat_lane, workers

    scene = restore_door
    with contextlib.ExitStack() as stack:
        writer = stack.enter_context(_live_writer(scene.repo)) if holder == "live" else None
        lock = writer.lock if writer else _orphan_lock(scene.repo)
        inode = lock.stat().st_ino
        status, body = scene.restore(scene.old, door)

        assert (status, body.get("reason")) == (409, "restore_not_applied") and "restart_required" not in body, body
        assert "index.lock" in body["error"] and "File exists" in body["error"]
        assert scene.events["gate_at_reset"] == ["managed_update:manual_rollback"]  # writers were stopped...
        assert workers.repo_writer_admission_closed() == ""  # ...and are admitted again,
        assert worker_chat_lane.owner_conversation_admitted(42)  # the owner's Main turn too
        assert scene.events["pool"] == [{"allow_disabled_restart": True}] and scene.events["restarts"] == []
        assert _git(scene.repo, "rev-parse", "HEAD") == scene.head
        assert (scene.repo / "seed.txt").read_text(encoding="utf-8") == "new\n"
        assert lock.stat().st_ino == inode  # nobody removed or replaced the lock
        assert writer is None or writer.finish() == 0, writer.stderr
    assert holder == "crashed" or "first.txt" in _git(scene.repo, "ls-files").split()  # the live write landed
    [row] = _rows(scene.data, "checkout_step_failed")
    assert row["reason"] == {"ui_rollback": "ui_rollback", "vcs_rollback": "agent_rollback_tool"}[door]
    assert row["cmd"] == f"git reset --hard {scene.old}" and row["returncode"] == 128 and "index.lock" in row["stderr"]
    assert _git(scene.repo, "rev-parse", row["keep_branch"]) == scene.head  # made before the reset, retained
    assert os.path.isdir(row["rescue_path"])
    assert not _rows(scene.data, "manual_rollback")


@pytest.mark.parametrize("guard", ["update_transaction", "saved_work_handoff"])
def test_a_refused_restore_readmits_writers_only_through_the_aborted_update_guards(restore_door, monkeypatch, guard):
    """The same exact index-lock refusal of the reset, but a readmission guard holds: a durable update
    transaction (the door's entry refuses one, so here it appears while the restore runs) or a stopped
    work handoff that cannot be confirmed. The gate stays shut and no pool starts; the answer still
    describes only the reset, and the transaction stays for its owner."""
    from supervisor import restart_retention, update_merge, workers

    scene = restore_door
    lock = _orphan_lock(scene.repo)
    handoffs = []
    monkeypatch.setattr(restart_retention, "recover_aborted_update",
                        lambda *args: handoffs.append(args) or guard != "saved_work_handoff")
    if guard == "update_transaction":
        observed = git_ops.git_capture

        def transaction_appears(cmd, **kwargs):
            if cmd[:3] == ["git", "reset", "--hard"] and not update_merge.active_update_tx():
                update_merge.write_update_tx({"phase": update_merge.GATE_BLOCKED_PHASE, "pre_update_sha": scene.head,
                                              "pre_update_branch": "ouroboros"})
            return observed(cmd, **kwargs)

        monkeypatch.setattr(git_ops, "git_capture", transaction_appears)

    status, body = scene.restore(scene.old)

    assert (status, sorted(body), body["reason"]) == (409, ["error", "reason"], "restore_not_applied"), body
    assert body["error"].startswith("the restore was not applied: Git refused its reset: ")
    assert "index.lock" in body["error"] and scene.events["gate_at_reset"] == ["managed_update:manual_rollback"]
    assert workers.repo_writer_admission_closed() == "managed_update:manual_rollback"  # not reopened
    assert scene.events["pool"] == [] and scene.events["restarts"] == []
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.head and lock.read_bytes() == b"left by a crashed process\n"
    [row] = _rows(scene.data, "checkout_step_failed")
    assert row["cmd"] == f"git reset --hard {scene.old}" and row["returncode"] == 128
    if guard == "update_transaction":
        assert handoffs == []  # the transaction guard holds first
        assert update_merge.read_update_tx_strict()[1]["phase"] == update_merge.GATE_BLOCKED_PHASE  # kept
    else:
        assert [args[0] for args in handoffs] == [scene.data]  # asked once; its refusal kept the gate shut
        assert update_merge.read_update_tx_strict()[0] == "absent"


def test_without_a_lock_the_restore_lands_and_requests_its_restart(restore_door):
    from supervisor import workers

    scene = restore_door
    status, body = scene.restore(scene.old)

    assert (status, body["status"], body["restarting"], body["switch_deferred"]) == (200, "ok", True, False), body
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.old and scene.events["restarts"] == [scene.old]
    assert workers.repo_writer_admission_closed() == "managed_update:manual_rollback"  # the restart follows
    assert scene.events["pool"] == [] and not _rows(scene.data, "checkout_step_failed")
    [row] = _rows(scene.data, "manual_rollback")
    assert row["new_sha"] == scene.old


def test_a_reset_refused_after_it_moved_the_tree_stays_fenced_for_a_restart(restore_door):
    """A held branch-ref lock: Git takes the index lock, rewrites the tree and index, then fails
    to move HEAD. Its message names a lock and HEAD is unchanged, yet the tree moved: no
    readmission, the restart stays required."""
    from supervisor import workers

    scene = restore_door
    (scene.repo / ".git" / "refs" / "heads" / "ouroboros.lock").write_bytes(b"")
    status, body = scene.restore(scene.old)

    assert (status, body["restart_required"]) == (500, True), body
    assert body["error"].startswith("git reset failed: ") and "ouroboros.lock': File exists" in body["error"]
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.head  # HEAD did not move...
    assert (scene.repo / "seed.txt").read_text(encoding="utf-8") == "old\n"  # ...but the tree did
    assert not (scene.repo / "new.txt").exists()
    assert workers.repo_writer_admission_closed() == "managed_update:manual_rollback"
    assert scene.events["pool"] == [] and scene.events["restarts"] == []
    assert not _rows(scene.data, "checkout_step_failed") and not _rows(scene.data, "manual_rollback")


def test_a_deferred_restore_refused_at_the_restart_exit_is_a_recorded_failed_switch(restore_door):
    """The store-less target's seam calls the same Restore owner at the restart exit; Git's typed
    refusal becomes the seam's failed switch in Git's words, never an exception after retirement."""
    from ouroboros import store_switch
    from ouroboros.store_import import ensure_current_store

    scene = restore_door
    ensure_current_store(scene.data)
    status, body = scene.restore(scene.storeless)
    assert (status, body["switch_deferred"]) == (200, True), body
    lock = _orphan_lock(scene.repo)

    outcome = store_switch.run_switch_seam(scene.data, repo_dir=scene.repo)

    assert outcome["status"] == "switch_failed" and "index.lock" in outcome["message"]
    assert "no checkout step had changed the working tree" in outcome["message"].splitlines()[0]
    receipt = json.loads((scene.data / store_switch.RECEIPT_REL).read_text(encoding="utf-8"))
    assert receipt["switched"]["ok"] is False and receipt["switched"]["head"] == scene.head
    assert lock.exists() and _git(scene.repo, "rev-parse", "HEAD") == scene.head


def test_restore_does_not_label_a_repaired_retry_as_an_unchanged_refusal(restore_door, monkeypatch):
    """A corrupt index appears after rescue; a different writer takes the lock after
    a successful rebuild. The final lock error alone cannot certify the whole reset.
    Restore must use one physical attempt, leaving ordinary capture repair unchanged.
    """
    from supervisor import workers

    scene = restore_door
    real_run = subprocess.run
    hard_attempts = []
    repairs = []

    def concurrent_index_changes(cmd, *args, **kwargs):
        if cmd[:3] == ["git", "reset", "--hard"]:
            hard_attempts.append(list(cmd))
            if len(hard_attempts) == 1:
                (scene.repo / ".git" / "index").write_bytes(b"broken")
        result = real_run(cmd, *args, **kwargs)
        if cmd[:3] == ["git", "reset", "--mixed"] and result.returncode == 0:
            repairs.append(list(cmd))
            _orphan_lock(scene.repo)
        return result

    monkeypatch.setattr(git_ops.subprocess, "run", concurrent_index_changes)
    status, body = scene.restore(scene.old)

    assert status == 500 and body["restart_required"], body
    assert "index file smaller than expected" in body["error"]
    assert len(hard_attempts) == 1 and not repairs
    assert (scene.repo / ".git" / "index").read_bytes() == b"broken"
    assert not _rows(scene.data, "checkout_step_failed")
    assert workers.repo_writer_admission_closed() == "managed_update:manual_rollback"
    assert scene.events["pool"] == [] and scene.events["restarts"] == []
