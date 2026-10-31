"""All update doors observe the effective Git lock before they stop any work."""

import contextlib
import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.gateway import control
from supervisor import git_ops, git_ops_reset, workers
from tests.test_git_lock_owners import _git, _repo, _live_writer


@pytest.mark.parametrize("topology", ["checkout", "linked-worktree", "alternate-index"])
@pytest.mark.parametrize("holder", ["orphan", "live"])
@pytest.mark.parametrize("door", ["smart", "replace", "ui_rollback", "vcs_rollback"])
def test_lock_refusal_precedes_every_stop(tmp_path, monkeypatch, topology, holder, door):
    repo = _repo(tmp_path)
    if topology == "linked-worktree":
        linked = tmp_path / "linked"
        _git(repo, "worktree", "add", "-qb", "linked", str(linked))
        repo = linked
    elif topology == "alternate-index":
        alternate = tmp_path / "custom-index"
        alternate.write_bytes((repo / ".git" / "index").read_bytes())
        monkeypatch.setenv("GIT_INDEX_FILE", str(alternate))
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    stopped = []
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: stopped.append(reason) or [])
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: pytest.fail("nothing stopped"))
    path = pathlib.Path(str(git_ops_reset.effective_index_path()) + ".lock")
    with contextlib.ExitStack() as stack:
        writer = stack.enter_context(_live_writer(repo)) if holder == "live" else None
        if writer is None:
            path.write_bytes(b"orphaned index writer\n")
        before = path.read_bytes(), path.stat().st_ino
        if door in {"ui_rollback", "vcs_rollback"}:
            status, body = control.rollback_through_door("HEAD", door=door,
                request_restart=lambda: pytest.fail("a refusal cannot restart"))
        else:
            apply = control._apply_smart_update_fenced if door == "smart" else control._apply_replace_recovery_fenced
            response = apply(SimpleNamespace(), expected_base_sha="a" * 40, expected_target_sha="b" * 40)
            status, body = response.status_code, json.loads(response.body)
        assert (status, body["reason"], body["outcome"]) == (409, "index_lock_present", "not_applied")
        assert body["index_lock"]["path"] == str(path) and body["index_lock"]["exists"] is True
        assert stopped == [] and workers.repo_writer_admission_closed() == ""
        assert (path.read_bytes(), path.stat().st_ino) == before
        if writer:
            assert writer.finish() == 0, writer.stderr


def test_absent_lock_is_observed_without_creating_one(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    assert control._index_lock_precondition() is None
    assert not (repo / ".git" / "index.lock").exists()


@pytest.mark.parametrize("door", ["smart", "replace", "ui_rollback", "vcs_rollback"])
def test_failed_git_observation_is_not_applied_before_any_stop(tmp_path, monkeypatch, door):
    from supervisor import update_merge

    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    assert control._index_lock_precondition() is None
    # A broken checkout must keep Git's actual failure, even when the caller is Apply.
    (repo / ".git").rename(repo / "unavailable-git-dir")
    rc, _, error = git_ops.git_capture(["git", "rev-parse", "--path-format=absolute", "--git-path", "index"])
    assert rc != 0 and "not a git repository" in error
    stopped = []
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: stopped.append(reason) or [])
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: pytest.fail("nothing stopped"))
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: pytest.fail("no mutation admitted"))
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: pytest.fail("no mutation admitted"))
    monkeypatch.setattr(control.update_progress, "begin", lambda **kwargs: pytest.fail("no attempt admitted"))
    if door in {"ui_rollback", "vcs_rollback"}:
        status, body = control.rollback_through_door("HEAD", door=door,
            request_restart=lambda: pytest.fail("a refusal cannot restart"))
    else:
        apply = control._apply_smart_update_fenced if door == "smart" else control._apply_replace_recovery_fenced
        response = apply(SimpleNamespace(), expected_base_sha="a" * 40, expected_target_sha="b" * 40)
        status, body = response.status_code, json.loads(response.body)
    assert (status, body["reason"], body["outcome"]) == (409, "index_lock_check_failed", "not_applied")
    assert error in body["error"] and "No work was stopped" in body["error"]
    assert stopped == [] and workers.repo_writer_admission_closed() == ""


@pytest.mark.parametrize("topology", ["checkout", "linked-worktree", "alternate-index", "relative-index"])
@pytest.mark.parametrize("output", ["modern", "echoed-flag", "relative", "empty"])
def test_index_path_fallback_observes_the_real_lock(tmp_path, monkeypatch, topology, output):
    repo = _repo(tmp_path)
    if topology == "linked-worktree":
        linked = tmp_path / "linked"
        _git(repo, "worktree", "add", "-qb", "linked", str(linked))
        repo = linked
    elif topology in {"alternate-index", "relative-index"}:
        monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "alternate") if topology == "alternate-index" else "alternate")
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    expected = pathlib.Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    original = git_ops.git_capture

    def capture(cmd, **kwargs):
        if "--path-format=absolute" in cmd and output != "modern":
            rc, path, error = original([part for part in cmd if part != "--path-format=absolute"], **kwargs)
            if output == "empty":
                return 0, "", ""
            return rc, ("--path-format=absolute\n" if output == "echoed-flag" else "") + path, error
        return original(cmd, **kwargs)

    monkeypatch.setattr(git_ops, "git_capture", capture)
    assert git_ops_reset.effective_index_path() == expected
    assert control._index_lock_precondition() is None
    lock = pathlib.Path(str(expected) + ".lock")
    lock.write_bytes(b"external writer\n")
    before = lock.read_bytes(), lock.stat().st_ino
    response = control._index_lock_precondition()
    assert response.status_code == 409
    body = json.loads(response.body)
    assert body["reason"] == "index_lock_present" and body["index_lock"]["path"] == str(lock)
    command = ["git", "reset", "--hard", "HEAD"]
    rc, _, stderr = git_ops.git_capture(command)
    assert rc == 128
    assert git_ops_reset._index_lock_refused(command, SimpleNamespace(returncode=rc, stderr=stderr))
    assert (lock.read_bytes(), lock.stat().st_ino) == before
