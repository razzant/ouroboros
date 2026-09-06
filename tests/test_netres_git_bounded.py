"""Lane C1 contracts: bounded network git honors an explicit cwd, kills a hung
process tree on timeout, and the routed callers keep their result shapes."""

from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import time
from types import SimpleNamespace

import pytest

from supervisor import git_ops, update_source

# The hung-git tests fake `git` with a #!/bin/sh PATH shim; that shim is not
# executable on Windows (the real git would run and the assertions would
# lie), so they are POSIX-only. The pure-monkeypatch shape tests below stay
# cross-platform.
_posix_shim = pytest.mark.skipif(
    os.name == "nt", reason="uses a #!/bin/sh PATH shim; POSIX-only"
)


def _git(repo: pathlib.Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"},
    )
    return res.stdout.strip()


def _seed_repo(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(path, "add", "seed.txt")
    _git(path, "commit", "-qm", "seed")
    return path


def test_git_network_bounded_honors_explicit_cwd(tmp_path, monkeypatch):
    """An explicit cwd selects the repository; the system repo is untouched."""
    upstream = _seed_repo(tmp_path / "upstream")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(upstream), str(clone)],
        capture_output=True, text=True, check=True,
    )
    (upstream / "seed.txt").write_text("advanced\n", encoding="utf-8")
    _git(upstream, "commit", "-qam", "advance")
    new_tip = _git(upstream, "rev-parse", "HEAD")

    # A non-repo default proves the explicit cwd (not REPO_DIR) was used.
    sentinel = tmp_path / "not-a-repo"
    sentinel.mkdir()
    monkeypatch.setattr(git_ops, "REPO_DIR", sentinel)

    rc, _out, err = update_source._git_network_bounded(
        ["fetch", "origin"], cwd=clone, timeout=60,
    )

    assert rc == 0, err
    assert _git(clone, "rev-parse", "origin/main") == new_tip


def test_git_network_bounded_default_cwd_remains_system_repo(tmp_path, monkeypatch):
    repo = _seed_repo(tmp_path / "repo")
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    captured = {}

    def fake_bounded(cmd, *, timeout, cwd=None, env=None, text=True):
        captured["cwd"] = cwd
        return 0, "", ""

    monkeypatch.setattr(git_ops, "_run_git_process_bounded", fake_bounded)
    rc, _out, _err = update_source._git_network_bounded(["fetch", "origin"])
    assert rc == 0
    assert captured["cwd"] == repo


def test_git_network_bounded_rejects_missing_cwd(tmp_path):
    rc, out, err = update_source._git_network_bounded(
        ["fetch", "origin"], cwd=tmp_path / "does-not-exist",
    )
    assert rc != 0
    assert out == ""
    assert "cwd" in err


@_posix_shim
def test_git_network_bounded_timeout_kills_tree_and_repo_stays_operable(tmp_path, monkeypatch):
    """A hung network git is killed together with its children (kill + reap)
    and returns the typed timeout shape. The shim drops a lockfile under the
    clone's ``.git`` before sleeping: SIGKILL leaves such stale lockfiles
    behind — the runner does no cleanup, and clearing them is left to git's
    own per-file tolerance. The contract asserted here is that the repository
    stays operable: a real follow-up ``git fetch origin`` in the same clone
    succeeds."""
    upstream = _seed_repo(tmp_path / "upstream")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(upstream), str(clone)],
        capture_output=True, text=True, check=True,
    )
    (upstream / "seed.txt").write_text("advanced\n", encoding="utf-8")
    _git(upstream, "commit", "-qam", "advance")
    new_tip = _git(upstream, "rev-parse", "HEAD")

    original_path = os.environ.get("PATH", "")
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    stale_lock = clone / ".git" / "config.lock"
    fake_git = shim_dir / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'echo $$ > "$NETRES_PID_DIR/parent.pid"\n'
        'touch "$NETRES_STALE_LOCK"\n'
        "sleep 300 &\n"
        'echo $! > "$NETRES_PID_DIR/child.pid"\n'
        "wait\n",
        encoding="utf-8",
    )
    fake_git.chmod(fake_git.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{original_path}")
    monkeypatch.setenv("NETRES_PID_DIR", str(pid_dir))
    monkeypatch.setenv("NETRES_STALE_LOCK", str(stale_lock))

    rc, out, err = update_source._git_network_bounded(
        ["fetch", "origin"], cwd=clone, timeout=1.0,
    )

    assert rc == update_source.FETCH_TIMEOUT_RC
    assert out == ""
    assert "exceeded" in err

    pids = []
    for name in ("parent.pid", "child.pid"):
        raw = (pid_dir / name).read_text(encoding="utf-8").strip()
        assert raw, f"{name} was never written — shim did not run"
        pids.append(int(raw))
    deadline = time.monotonic() + 5
    for pid in pids:
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"process {pid} survived the bounded timeout kill")

    # SIGKILL leaves the stale lockfile behind — honesty pin: the runner does
    # NOT clean it up.
    assert stale_lock.exists()

    # Contract: the repository stays operable — a real follow-up network git
    # command in the same clone succeeds despite the stale lockfile.
    monkeypatch.setenv("PATH", original_path)
    rc2, _out2, err2 = update_source._git_network_bounded(
        ["fetch", "origin"], cwd=clone, timeout=60,
    )
    assert rc2 == 0, err2
    assert _git(clone, "rev-parse", "origin/main") == new_tip


def test_push_to_remote_timeout_surfaces_as_todays_failure_shape(monkeypatch):
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name: True)
    monkeypatch.setattr(
        git_ops,
        "_git_network_bounded",
        lambda _cmd, **_kw: (git_ops.FETCH_TIMEOUT_RC, "", "git push exceeded 300s and was terminated"),
    )
    ok, message = git_ops.push_to_remote("feature")
    assert ok is False
    assert message.startswith("git push failed:")
    assert "exceeded" in message


def test_push_to_remote_tags_timeout_stays_best_effort(monkeypatch):
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name: True)
    results = iter([
        (0, "", ""),
        (git_ops.FETCH_TIMEOUT_RC, "", "git push exceeded 300s and was terminated"),
    ])
    monkeypatch.setattr(
        git_ops, "_git_network_bounded", lambda _cmd, **_kw: next(results),
    )
    ok, message = git_ops.push_to_remote("feature", push_tags=True)
    assert ok is True
    assert "Pushed feature to origin" in message
    assert "tags push failed" in message


def test_ff_pull_fetch_is_bounded_with_repo_cwd_and_keeps_error_shape(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools

    upstream = _seed_repo(tmp_path / "upstream")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(upstream), str(clone)],
        capture_output=True, text=True, check=True,
    )
    captured = {}

    def fake_bounded(args, *, cwd=None, timeout=None):
        captured["args"] = list(args)
        captured["cwd"] = cwd
        return 0, "", ""

    monkeypatch.setattr(update_source, "_git_network_bounded", fake_bounded)
    result = git_tools._ff_pull(clone)
    assert captured["args"][0] == "fetch"
    assert captured["cwd"] == clone
    assert "Already up to date" in result

    monkeypatch.setattr(
        update_source,
        "_git_network_bounded",
        lambda args, **_kw: (1, "", "fatal: could not read from remote repository"),
    )
    result = git_tools._ff_pull(clone)
    assert result.startswith("⚠️ PULL_ERROR: git fetch failed:")


def test_ci_push_branch_is_bounded_with_repo_cwd_and_keeps_shape(tmp_path, monkeypatch):
    from ouroboros.tools import ci

    repo = _seed_repo(tmp_path / "repo")
    captured = {}

    def fake_bounded(args, *, cwd=None, timeout=None):
        captured["args"] = list(args)
        captured["cwd"] = cwd
        return 0, "pushed", ""

    monkeypatch.setattr(update_source, "_git_network_bounded", fake_bounded)
    ok, message = ci._push_branch(str(repo), "feature")
    assert ok is True
    assert message == "pushed"
    assert captured["args"] == ["push", "-u", "origin", "feature"]
    assert captured["cwd"] == repo

    monkeypatch.setattr(
        update_source,
        "_git_network_bounded",
        lambda args, **_kw: (update_source.FETCH_TIMEOUT_RC, "", "git push exceeded 300s and was terminated"),
    )
    ok, message = ci._push_branch(str(repo), "feature")
    assert ok is False
    assert "exceeded" in message


def test_run_git_network_cmd_failure_reports_stdout_when_stderr_is_empty(tmp_path, monkeypatch):
    """Some git failures report only on stdout; the run_cmd-shaped error must
    carry that text instead of an empty STDERR-only message."""
    from ouroboros.tools import git as git_tools

    monkeypatch.setattr(
        update_source,
        "_git_network_bounded",
        lambda args, **_kw: (1, "remote: rejected by hook", ""),
    )
    with pytest.raises(RuntimeError) as excinfo:
        git_tools._run_git_network_cmd(["git", "fetch", "origin"], cwd=tmp_path)
    message = str(excinfo.value)
    assert "Command failed: git fetch origin" in message
    assert "remote: rejected by hook" in message


def test_commit_path_auto_push_timeout_is_best_effort_warning(tmp_path, monkeypatch):
    """Through the commit path: a push timeout in the real ``_auto_push`` →
    real ``push_to_remote`` → bounded-runner chain leaves the commit itself
    successful, with the push-failed warning carried in the result."""
    from ouroboros.tools import git as git_module

    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path / "drive",
        branch_dev="ouroboros",
        current_task_type="task",
        task_id="t1",
        task_metadata={},
        last_push_succeeded=True,
        pending_events=[],
    )

    def fake_run(cmd, cwd=None, **_kw):
        if cmd[:2] == ["git", "commit"]:
            return ""
        if cmd[:2] == ["git", "rev-parse"]:
            return "a" * 40 + "\n"
        return ""

    def fake_stage_cycle(_ctx, _msg, _start, **_kw):
        return {
            "status": "passed", "message": "",
            "pre_fingerprint": {"fingerprint": "x"},
            "post_fingerprint": {"fingerprint": "x", "binding": {}},
        }

    monkeypatch.setattr(git_module, "run_cmd", fake_run)
    monkeypatch.setattr(git_module, "_run_reviewed_stage_cycle", fake_stage_cycle)
    monkeypatch.setattr(git_module, "_task_attributed_commit_paths",
                        lambda _ctx, paths: (paths, None, "", None))
    monkeypatch.setattr(git_module, "_check_overlapping_review_attempt", lambda _ctx: "")
    monkeypatch.setattr(git_module, "_prepare_review_commit_worktree",
                        lambda _ctx, _tx: (False, ""))
    monkeypatch.setattr(git_module, "_verify_reviewed_commit_binding",
                        lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr(git_module, "_managed_post_commit_tests_gate",
                        lambda *_a, **_kw: "")
    monkeypatch.setattr(git_module, "_auto_tag_on_version_bump", lambda *_a, **_kw: "")
    monkeypatch.setattr(git_module, "_post_commit_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(git_module, "_record_commit_attempt", lambda *_a, **_kw: None)
    monkeypatch.setattr(git_module, "_acquire_git_lock", lambda _ctx: tmp_path / "git.lock")
    monkeypatch.setattr(git_module, "_release_git_lock", lambda _path: None)

    from supervisor import update_merge

    monkeypatch.setattr(update_merge, "managed_assisted_tx_for",
                        lambda _task_id, _meta: (None, ""))

    # The push leg stays REAL: _auto_push → push_to_remote → bounded runner.
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name: True)
    monkeypatch.setattr(
        git_ops,
        "_git_network_bounded",
        lambda _cmd, **_kw: (git_ops.FETCH_TIMEOUT_RC, "", "git push exceeded 300s and was terminated"),
    )

    result = git_module._repo_commit_push(ctx, "netres: best-effort push test")

    assert result.startswith("OK: committed to ouroboros:")
    assert "[push skipped: git push failed:" in result
    assert "exceeded" in result
    assert ctx.last_push_succeeded is False
