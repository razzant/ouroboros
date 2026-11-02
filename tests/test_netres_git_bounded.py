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


@pytest.mark.serial
def test_checkout_reset_fetch_uses_configured_bound_and_keeps_local_head(tmp_path, monkeypatch):
    """The real reset/wrapper chain keeps its local reset after a timed-out fetch."""
    repo = _seed_repo(tmp_path / "repo")
    _git(repo, "remote", "add", "origin", str(tmp_path / "upstream"))
    original_head = _git(repo, "rev-parse", "HEAD")
    (repo / "seed.txt").write_text("dirty\n", encoding="utf-8")
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setenv("OUROBOROS_MANAGED_UPDATE_FETCH_TIMEOUT_SEC", "117")
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {})
    monkeypatch.setattr(git_ops, "_read_update_intent", lambda: {})
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    state, captured, events = {}, {}, []
    monkeypatch.setattr(git_ops, "save_state", state.update)
    monkeypatch.setattr(git_ops, "update_state", lambda mutator, **_kw: mutator(state) or state)
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, row: events.append(row))
    real_process = git_ops._run_git_process_bounded

    def fake_process(cmd, *, timeout, cwd, env, text):
        if "fetch" not in cmd:  # the reset's own local Git shares this runner and runs for real
            return real_process(cmd, timeout=timeout, cwd=cwd, env=env, text=text)
        captured.update(cmd=cmd, timeout=timeout, cwd=cwd, env=env)
        return git_ops.FETCH_TIMEOUT_RC, "", "timed out"

    monkeypatch.setattr(git_ops, "_run_git_process_bounded", fake_process)

    assert git_ops.checkout_and_reset("main", reason="restart", unsynced_policy="ignore") == (True, "ok")
    assert captured["cmd"] == [
        "git", "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=30", "fetch", "origin",
    ]
    assert captured["timeout"] == 117
    assert captured["cwd"] == repo
    assert captured["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert state == {"current_branch": "main", "current_sha": original_head}
    assert (repo / "seed.txt").read_text(encoding="utf-8") == "seed\n"
    assert events[0]["type"] == "reset_fetch_failed"
    assert events[0]["error"] == "git fetch origin failed: git fetch origin exceeded 117s and was terminated"
    assert events[0]["continuing_local_reset"] is True


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

    # First execution of a new shim can spend several seconds in macOS's
    # executable scan. Reach the hanging child before testing tree teardown.
    rc, out, err = update_source._git_network_bounded(
        ["fetch", "origin"], cwd=clone, timeout=10.0,
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


def test_push_to_remote_timeout_keeps_diagnostics_in_logs(monkeypatch, caplog):
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name: True)
    monkeypatch.setattr(
        git_ops,
        "_git_network_bounded",
        lambda _cmd, **_kw: (git_ops.FETCH_TIMEOUT_RC, "", "git push exceeded 300s and was terminated"),
    )
    ok, message = git_ops.push_to_remote("feature")
    assert ok is False
    assert message == "Branch publication failed or is unconfirmed; see Logs. No automatic retry"
    assert "exceeded 300s" in caplog.text and "exceeded" not in message


def test_push_to_remote_tags_timeout_stays_best_effort(tmp_path, monkeypatch, caplog):
    repo = _seed_repo(tmp_path / "repo")
    _git(repo, "branch", "feature")
    _git(repo, "tag", "-a", "v1", "-m", "release")
    commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name: True)
    results = iter([
        (0, "", ""),
        (git_ops.FETCH_TIMEOUT_RC, "", "git push exceeded 300s and was terminated"),
    ])
    monkeypatch.setattr(
        git_ops, "_git_network_bounded", lambda _cmd, **_kw: next(results),
    )
    ok, message = git_ops.push_to_remote("feature", push_tags=True, created_tag="v1", expected_commit_sha=commit)
    assert ok is True  # the compatibility boolean describes the successful branch push
    assert "Pushed feature to origin" in message
    assert "partial publication: tag v1" in message
    assert "no automatic retry" in message
    assert "exceeded 300s" in caplog.text and "exceeded" not in message


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
    assert "[push skipped: Branch publication failed or is unconfirmed; see Logs. No automatic retry]" in result
    assert "exceeded" not in result
    assert ctx.last_push_succeeded is False


def _ci_note(monkeypatch, runs, jobs=None):
    """The post-push CI note for one GitHub Actions answer; no network, no git."""
    import io
    import json
    import urllib.request

    from ouroboros.tools import git as git_module

    sha = "a" * 40
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPO", "example/project")
    monkeypatch.setattr(git_module, "run_cmd",
                        lambda cmd, cwd=None, **_kw: "ouroboros\n" if "--abbrev-ref" in cmd else sha + "\n")

    def fake_urlopen(request, timeout=None):
        if isinstance(runs, Exception):
            raise runs
        if request.full_url.endswith("/jobs"):
            return io.BytesIO(json.dumps({"jobs": jobs or []}).encode("utf-8"))
        assert f"event=push&head_sha={sha}" in request.full_url
        listed = [{"head_sha": sha, "run_number": index, "html_url": f"https://example.test/{index}",
                   "jobs_url": f"https://example.test/{index}/jobs", **run}
                  for index, run in enumerate(runs, start=1)]
        return io.BytesIO(json.dumps({"workflow_runs": listed}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return git_module._check_ci_status_after_push(pathlib.Path("."))


_DONE = {"status": "completed", "conclusion": "success"}


def test_ci_note_names_every_push_workflow_and_none_stands_for_the_others(monkeypatch):
    """A repository may run several push workflows, or none.
    A finished canary run listed first is not "CI passed"."""
    note = _ci_note(monkeypatch, [
        {"name": "Provider canaries", **_DONE},
        {"name": "CI", "status": "in_progress", "conclusion": None},
        {"name": "UI browser (ouroboros push)", "status": "queued", "conclusion": None},
    ])
    assert note.startswith("\n\n⏳ CI: push runs in progress — ") and "✅" not in note
    # The hint names the repository the push went to: a bare SHA is read in the Project's repository.
    assert note.endswith(f". Read the results later: github_read(method='checks', target='{'a' * 40}', repo='example/project').")
    assert "Provider canaries: success; CI: in progress; UI browser (ouroboros push): queued" in note

    note = _ci_note(monkeypatch, [{"name": "Provider canaries", **_DONE}, {"name": "CI", **_DONE}])
    assert note == ("\n\n✅ CI: registered push runs passed for this commit — "
                    "Provider canaries: success; CI: success.")
    assert _ci_note(monkeypatch, []) == "\n\nCI: No push workflow run is registered for this commit."


def test_ci_note_attributes_a_failure_to_its_workflow(monkeypatch):
    runs = [
        {"name": "CI", **_DONE},
        {"name": "Provider canaries", "status": "completed", "conclusion": "failure"},
    ]
    jobs = [{"name": "integration-test / integration-test", "conclusion": "failure",
             "steps": [{"name": "Run integration tests", "conclusion": "failure"}]}]
    note = _ci_note(monkeypatch, runs, jobs)
    assert "⚠️ CI STATUS: Provider canaries FAILED for this commit (run #2)" in note
    assert "  Workflows: CI: success; Provider canaries: failure\n" in note
    assert "  Failed: integration-test / integration-test → Run integration tests\n" in note
    assert note.endswith("  URL: https://example.test/2")

    # The same commit pushed again: the API lists the newest run of a workflow first.
    rerun = _ci_note(monkeypatch, [{"name": "Provider canaries", **_DONE}, *runs])
    assert rerun.startswith("\n\n✅") and "failure" not in rerun
    # A finished failure is reported while another workflow still runs, and every red workflow is named.
    racing = _ci_note(monkeypatch, [
        {"name": "CI", "status": "in_progress", "conclusion": None},
        {"name": "Provider canaries", "status": "completed", "conclusion": "failure"},
        {"name": "UI browser (ouroboros push)", "status": "completed", "conclusion": "failure"},
    ])
    assert "⚠️ CI STATUS: Provider canaries, UI browser (ouroboros push) FAILED" in racing and "⏳" not in racing
    assert "CI: in progress" in racing
    skipped = _ci_note(monkeypatch, [{"name": "CI", **_DONE},
                                     {"name": "Sync mirror", "status": "completed", "conclusion": "skipped"}])
    assert skipped.startswith("\n\n✅") and "Sync mirror: skipped" in skipped
    cancelled = _ci_note(monkeypatch, [{"name": "CI", "status": "completed", "conclusion": "cancelled"}])
    assert "⚠️ CI STATUS: CI CANCELLED for this commit (run #1)" in cancelled
    # A cancelled workflow beside a failed one: only the failure is called FAILED, and its jobs are read.
    mixed = _ci_note(monkeypatch, [{"name": "CI", "status": "completed", "conclusion": "cancelled"}, runs[1]], jobs)
    assert "⚠️ CI STATUS: Provider canaries FAILED for this commit (run #2)" in mixed
    assert "CI: cancelled" in mixed and "  Failed: integration-test / integration-test" in mixed


def test_ci_note_stays_empty_when_github_is_unreachable(monkeypatch):
    assert _ci_note(monkeypatch, OSError("network down")) == ""
