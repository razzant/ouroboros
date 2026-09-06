import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import supervisor.git_ops as git_ops

from tests._git_ops_recovery_shared import (
    _git,
    _history_repo,
)


def test_manual_rollback_pins_previous_head_before_reset(tmp_path, monkeypatch):
    repo, first, second = _history_repo(tmp_path)
    git_ops.init(repo, tmp_path / "data", "")
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name=None: False)
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    monkeypatch.setattr(git_ops, "save_state", lambda _state: None)

    ok, message = git_ops.rollback_to_version(first, reason="test")

    assert ok, message
    assert _git(repo, "rev-parse", "HEAD") == first
    keep_branches = _git(repo, "branch", "--list", "rollback-keep-*").splitlines()
    assert len(keep_branches) == 1
    keep_branch = keep_branches[0].lstrip("* ")
    assert _git(repo, "rev-parse", keep_branch) == second
    assert keep_branch in message


def test_promotion_push_uses_captured_sha_when_dev_advances(tmp_path, monkeypatch):
    repo, first, second = _history_repo(tmp_path)
    _git(repo, "branch", "ouroboros-stable", first)
    git_ops.init(repo, tmp_path / "data", "")
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name=None: True)
    pushed = []

    def fake_push(args, **_kwargs):
        pushed.append(list(args))
        (repo / "value.txt").write_text("three\n", encoding="utf-8")
        _git(repo, "commit", "-qam", "three")
        return 0, "", ""

    monkeypatch.setattr(git_ops, "_git_network_bounded", fake_push)

    ok, result = git_ops.promote_branch_exact(
        "ouroboros", "ouroboros-stable", push_remote=True
    )

    assert ok, result
    assert result["sha"] == second
    assert _git(repo, "rev-parse", "ouroboros-stable") == second
    assert _git(repo, "rev-parse", "ouroboros") != second
    assert pushed == [[
        "push", "origin", f"{second}:refs/heads/ouroboros-stable"
    ]]


def test_event_promotion_refuses_while_managed_update_is_active(monkeypatch):
    import supervisor.events as events
    import supervisor.update_merge as update_merge

    token = object()
    released = []
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: token)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {"phase": "pending_boot_smoke"})
    monkeypatch.setattr(update_merge, "release_update_lock", lambda value: released.append(value))
    monkeypatch.setattr(
        git_ops,
        "promote_branch_exact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("promotion must be fenced")),
    )
    ctx = SimpleNamespace(
        BRANCH_DEV="ouroboros", BRANCH_STABLE="ouroboros-stable", load_state=lambda: {}
    )

    events._handle_promote_to_stable({}, ctx)

    assert released == [token]


def test_git_capture_repairs_corrupt_index(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "index").write_text("broken", encoding="utf-8")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)

    calls = {"status": 0, "rebuild": 0}

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None,
                 timeout=None):
        if cmd == ["git", "status", "--porcelain"]:
            calls["status"] += 1
            if calls["status"] == 1:
                return subprocess.CompletedProcess(
                    cmd,
                    128,
                    stdout="",
                    stderr="fatal: .git/index: index file smaller than expected\n",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout=" M changed.py\n", stderr="")
        if cmd == ["git", "reset", "--mixed", "HEAD"]:
            calls["rebuild"] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    rc, stdout, stderr = git_ops.git_capture(["git", "status", "--porcelain"])

    assert rc == 0
    assert stdout == "M changed.py"
    assert stderr == ""
    assert calls["status"] == 2
    assert calls["rebuild"] == 1
    assert any(path.name.startswith("index.corrupt.") for path in git_dir.iterdir())


def test_bounded_git_capture_bounds_corrupt_index_rebuild(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "index").write_text("broken", encoding="utf-8")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)

    calls = []

    def fake_bounded(cmd, *, timeout, cwd=None, env=None, text=True):
        calls.append((cmd, timeout, cwd, text))
        if cmd == ["git", "status", "--porcelain"] and len(calls) == 1:
            return 128, "", "fatal: .git/index: index file smaller than expected"
        if cmd == ["git", "reset", "--mixed", "HEAD"]:
            return 0, "", ""
        if cmd == ["git", "status", "--porcelain"]:
            return 0, " M changed.py\n", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "_run_git_process_bounded", fake_bounded)

    rc, stdout, stderr = git_ops.git_capture(
        ["git", "status", "--porcelain"], timeout=17,
    )

    assert (rc, stdout, stderr) == (0, "M changed.py", "")
    assert [call[0] for call in calls] == [
        ["git", "status", "--porcelain"],
        ["git", "reset", "--mixed", "HEAD"],
        ["git", "status", "--porcelain"],
    ]
    assert all(call[1] == 17 for call in calls)


@pytest.mark.serial
def test_git_capture_times_out_instead_of_hanging(monkeypatch, tmp_path):
    """Issue #182: a hung git process (fsmonitor deadlock, an unresponsive
    filesystem) must not stall the rescue/rollback graph indefinitely. A
    genuinely slow process under a small timeout returns a typed failure
    quickly rather than blocking for its full runtime."""
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)

    started = time.monotonic()
    rc, stdout, stderr = git_ops.git_capture(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2,
    )
    elapsed = time.monotonic() - started

    assert rc == git_ops.FETCH_TIMEOUT_RC
    assert stdout == ""
    assert "timed out" in stderr
    assert elapsed < 4  # well under the 5s sleep; proves it did not wait it out


def test_git_capture_default_timeout_is_unbounded(monkeypatch, tmp_path):
    """Every call site other than the rescue graph passes no timeout at all;
    confirm that shape delegates without adding a subprocess timeout."""
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    rc, stdout, stderr = git_ops.git_capture(["git", "status", "--porcelain"])

    assert rc == 0
    assert stdout == ""
    assert stderr == ""
    assert "timeout" not in captured


def test_bounded_git_process_kills_tree_and_is_panic_tracked(monkeypatch):
    import ouroboros.platform_layer as platform_layer
    from ouroboros.tools import shell

    killed = []

    class HungProcess:
        returncode = -9

        def __init__(self):
            self.pid = 12345
            self.calls = 0

        def communicate(self, input=None, timeout=None):
            assert self in shell._active_subprocesses
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["git", "status"], timeout)
            return "", "partial diagnostic"

    proc = HungProcess()
    monkeypatch.setattr(git_ops.subprocess, "Popen", lambda *_a, **_k: proc)
    monkeypatch.setattr(
        platform_layer, "kill_process_tree", lambda child: killed.append(child),
    )

    rc, stdout, stderr = git_ops._run_git_process_bounded(
        ["git", "status"], timeout=0.01,
    )

    assert rc == git_ops.FETCH_TIMEOUT_RC
    assert stdout == ""
    assert "timed out" in stderr
    assert "partial diagnostic" in stderr
    assert killed == [proc]
    assert proc not in shell._active_subprocesses


def test_rescue_git_capture_bounds_with_rescue_timeout(monkeypatch):
    """The rescue wrapper forwards the one settings-owned timeout authority."""
    captured = {}

    def fake_git_capture(cmd, *, timeout=None):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return 0, "", ""

    from ouroboros import update_channels

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)
    monkeypatch.setattr(update_channels, "get_rescue_git_timeout_sec", lambda: 321)

    rc, stdout, stderr = git_ops.rescue_git_capture(["git", "status", "--porcelain"])

    assert captured["cmd"] == ["git", "status", "--porcelain"]
    assert captured["timeout"] == 321
    assert (rc, stdout, stderr) == (0, "", "")


def test_collect_repo_sync_state_uses_rescue_bounded_capture(monkeypatch):
    """The rescue/rollback graph must go through the bounded wrapper, not the
    unbounded default: this is what actually closes #182 end to end."""
    calls = []

    def fake_rescue_git_capture(cmd):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "remote"]:
            return 0, "origin", ""
        if cmd == [
            "git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}",
        ]:
            return 0, "origin/ouroboros", ""
        if cmd == ["git", "log", "--oneline", "origin/ouroboros..HEAD"]:
            return 0, "", ""
        return 0, "", ""

    monkeypatch.setattr(git_ops, "rescue_git_capture", fake_rescue_git_capture)
    monkeypatch.setattr(
        git_ops, "git_capture",
        lambda cmd, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"unbounded capture escaped rescue graph: {cmd}")
        ),
    )

    git_ops._collect_repo_sync_state()

    assert ["git", "rev-parse", "--abbrev-ref", "HEAD"] in calls
    assert ["git", "status", "--porcelain"] in calls
    assert ["git", "remote"] in calls
    assert ["git", "log", "--oneline", "origin/ouroboros..HEAD"] in calls
