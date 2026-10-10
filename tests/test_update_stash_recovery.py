"""Real-Git preservation and crash boundaries of the managed-update stash owner."""

import json
import sys

import pytest

from ouroboros.gateway import control
from supervisor import git_ops, update_candidate, update_merge
from tests.test_update_dirty_stash import _git, _init_repo, _point_at
from tests.test_update_merge_assisted import _stub_worker_gates

pytestmark = pytest.mark.serial


class Crash(BaseException):
    """A process interruption must escape ordinary exception recovery."""


def _fixture(tmp_path, monkeypatch, consumer="boot_finalize"):
    repo, branch = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, branch)
    _stub_worker_gates(monkeypatch)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("owner work\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("owner scratch\n", encoding="utf-8")
    status, sha, error = update_merge.stash_local_changes_for_update("fault-test")
    assert status == "ok" and sha, error
    assert update_merge.create_rescue_local_ref(sha)
    tx = {"pre_update_sha": pre, "pre_update_branch": branch, "stash_sha": sha,
          "target_sha": pre, "attempt_id": "fault-test"}
    if consumer in {"boot_finalize", "boot_diverged", "rollback"}:
        (repo / "official.txt").write_text("official\n", encoding="utf-8")
        assert _git(repo, "add", "-A").returncode == 0
        assert _git(repo, "commit", "-qm", "official").returncode == 0
        current = _git(repo, "rev-parse", "HEAD").stdout.strip()
        tx.update(target_sha=current, merge_commit=current, pre_restart_smoke="passed")
    if consumer == "boot_diverged":
        # An unrelated local descendant, not the update's merge commit.
        tx.pop("merge_commit")
        tx["target_sha"] = "f" * 40
        tx["phase"] = "assisted_resolution"
    elif consumer == "rollback":
        tx["phase"] = "rolling_back"
    elif consumer == "boot_finalize":
        tx["phase"] = "pending_boot_smoke"
    else:
        tx["phase"] = "stashing_local_work"
    update_merge.write_update_tx(tx)
    return repo, tx


def _drive(consumer, tx):
    if consumer == "gateway_unwind":
        return {"stash_note": control._unwind_stashed_update(tx, "test_unwind")}
    if consumer == "rollback":
        ok, note = update_merge.rollback_managed_update("test_rollback")
        return {"rolled_back": ok, "stash_note": note}
    return update_merge.finalize_managed_update_on_boot(supervisor_ready=True)


def _events(tmp_path):
    path = tmp_path / "data" / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.mark.parametrize("consumer", ["rollback", "boot_finalize", "boot_diverged", "boot_preapply", "gateway_unwind"])
def test_each_consumer_retains_failed_marker_and_replay_preserves_later_edits(tmp_path, monkeypatch, consumer):
    repo, tx = _fixture(tmp_path, monkeypatch, consumer)
    real_write = update_merge.write_update_tx

    def fail_final_marker(payload):
        if payload.get("stash_restored"):
            raise OSError("injected completion write failure")
        real_write(payload)

    monkeypatch.setattr(update_merge, "write_update_tx", fail_final_marker)
    result = _drive(consumer, tx)
    assert "unconfirmed" in result["stash_note"], result
    assert result.get("rolled_back") is not True
    assert result.get("finalized") is not True
    saved = update_merge.read_update_tx()
    assert saved["stash_restore"]["status"] == "applying"
    assert saved["_schema_version"] == 2 and not saved.get("stash_restored")
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"
    assert tx["stash_sha"] in _git(repo, "stash", "list", "--format=%H").stdout
    assert not any(event["type"] == "managed_update_stash_restored" for event in _events(tmp_path))

    (repo / "a.txt").write_text("owner work plus later edits\n", encoding="utf-8")
    (repo / "later.txt").write_text("later untracked\n", encoding="utf-8")
    monkeypatch.setattr(update_merge, "write_update_tx", real_write)
    real_capture, calls = git_ops.git_capture, []

    def capture(cmd):
        calls.append(cmd)
        return real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", capture)
    resumed = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert "unconfirmed" in resumed["stash_note"]
    if consumer == "rollback":
        assert update_merge.read_update_tx()["phase"] == update_merge.MARKER_CLEANUP_RETRY_PHASE
        resumed = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
        assert resumed["finalized"] is True and not resumed.get("rolled_back")
    assert update_merge.read_update_tx() == {}
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work plus later edits\n"
    assert (repo / "later.txt").read_text(encoding="utf-8") == "later untracked\n"
    assert not any(cmd[1] in {"reset", "clean", "checkout"} or cmd[:3] == ["git", "stash", "apply"] for cmd in calls)


def test_crash_between_apply_and_marker_uses_old_reader_refusal_then_new_reader_handoff(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "rollback")
    real_capture = git_ops.git_capture

    def crash_after_apply(cmd):
        result = real_capture(cmd)
        if cmd[:3] == ["git", "stash", "apply"]:
            assert result[0] == 0
            raise Crash()
        return result

    monkeypatch.setattr(git_ops, "git_capture", crash_after_apply)
    with pytest.raises(Crash):
        update_merge.rollback_managed_update("crash")
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == tx["pre_update_sha"]
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"
    marker = update_merge._update_tx_marker_path()
    before = marker.read_bytes()
    monkeypatch.setattr(update_merge, "UPDATE_TX_SCHEMA_VERSION", 1)
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: pytest.fail(f"old reader ran Git: {cmd}"))
    assert update_merge.read_update_tx_strict()[0] == "future"
    assert update_merge.rollback_managed_update("old-reader")[0] is False
    refused = update_merge.finalize_managed_update_on_boot()
    assert "newer version" in refused["reason"]
    assert marker.read_bytes() == before
    monkeypatch.setattr(update_merge, "UPDATE_TX_SCHEMA_VERSION", 2)
    monkeypatch.setattr(git_ops, "git_capture", real_capture)
    (repo / "a.txt").write_text("later owner work\n", encoding="utf-8")
    resumed = update_merge.finalize_managed_update_on_boot()
    assert resumed["rolled_back"] is True
    assert "unconfirmed" in resumed["stash_note"]
    assert (repo / "a.txt").read_text(encoding="utf-8") == "later owner work\n"


def test_rollback_checkout_before_restore_refuses_unsafe_old_recovery(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "rollback")
    real_capture = git_ops.git_capture

    def crash_after_checkout(cmd):
        result = real_capture(cmd)
        if cmd[:3] == ["git", "checkout", "-B"]:
            raise Crash()
        return result

    monkeypatch.setattr(git_ops, "git_capture", crash_after_checkout)
    with pytest.raises(Crash):
        update_merge.rollback_managed_update("before-restore")
    saved = update_merge.read_update_tx()
    assert saved["_schema_version"] == 2 and "stash_restore" not in saved
    before = update_merge._update_tx_marker_path().read_bytes()
    monkeypatch.setattr(update_merge, "UPDATE_TX_SCHEMA_VERSION", 1)
    assert update_merge.read_update_tx_strict()[0] == "future"
    assert update_merge.rollback_managed_update("old-reader")[0] is False
    assert update_merge._update_tx_marker_path().read_bytes() == before
    monkeypatch.setattr(update_merge, "UPDATE_TX_SCHEMA_VERSION", 2)
    monkeypatch.setattr(git_ops, "git_capture", real_capture)
    assert update_merge.rollback_managed_update("resume")[0] is True
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"


def test_write_ahead_failure_does_not_apply_or_clear_and_retry_still_restores(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "boot_preapply")
    real_write = update_merge.write_update_tx
    monkeypatch.setattr(update_merge, "write_update_tx", lambda payload: (_ for _ in ()).throw(OSError("read-only disk")))
    result = update_merge.finalize_managed_update_on_boot()
    assert result["reason"] == "stash_recovery_incomplete"
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base\n"
    assert update_merge.read_update_tx()["stash_sha"] == tx["stash_sha"]
    monkeypatch.setattr(update_merge, "write_update_tx", real_write)
    result = update_merge.finalize_managed_update_on_boot()
    assert result["stash_restore_status"] == "restored"
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"


def _conflict(tmp_path, monkeypatch):
    repo, branch = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, branch)
    path = "conflict 'named' file.txt" if sys.platform == "win32" else " conflict\nname.txt "
    (repo / path).write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "conflict base")
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / path).write_text("owner\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("owner scratch\n", encoding="utf-8")
    status, sha, _ = update_merge.stash_local_changes_for_update("conflict")
    assert status == "ok"
    (repo / path).write_text("official\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "official")
    target = _git(repo, "rev-parse", "HEAD").stdout.strip()
    tx = {"phase": "pending_boot_smoke", "pre_update_sha": pre, "pre_update_branch": branch,
          "target_sha": target, "merge_commit": target, "pre_restart_smoke": "passed", "stash_sha": sha}
    update_merge.write_update_tx(tx)
    return repo, tx, path


def test_real_conflict_census_and_stdout_are_recorded_before_cleanup_and_reach_notice(tmp_path, monkeypatch):
    repo, tx, path = _conflict(tmp_path, monkeypatch)
    real_capture = git_ops.git_capture
    before_cleanup = []

    def capture(cmd):
        if cmd[:3] == ["git", "reset", "--hard"]:
            record = update_merge.read_update_tx()["stash_restore"]
            assert record["status"] == "cleanup_pending"
            assert record["conflict_paths"] == [path]
            assert record["apply_returncode"] == 1
            assert record["stdout"] and not record["stderr"]
            assert (repo / "scratch.txt").exists()
            before_cleanup.append(record)
        return real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", capture)
    result = update_merge.finalize_managed_update_on_boot()
    assert before_cleanup and result["finalized"] is True
    assert result["stash_restore_status"] == "preserved"
    assert json.dumps(path) in result["stash_note"]
    assert "Git diagnostics:" in result["stash_note"]
    assert "git stash apply " + tx["stash_sha"] in result["stash_note"]
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert tx["stash_sha"] in _git(repo, "stash", "list", "--format=%H").stdout
    event = next(row for row in _events(tmp_path) if row["type"] == "managed_update_stash_restore_failed")
    assert event["conflict_paths"] == [path] and event["cleanup"]["verified_clean"] is True
    assert event["stdout"] == before_cleanup[0]["stdout"]
    assert not update_merge.active_update_tx()


def test_cleanup_failure_is_not_success_and_replay_preserves_unmerged_files(tmp_path, monkeypatch):
    from supervisor import worker_chat_lane, workers

    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    repo, tx, path = _conflict(tmp_path, monkeypatch)
    real_capture = git_ops.git_capture

    def fail_reset(cmd):
        if cmd[:3] == ["git", "reset", "--hard"]:
            return 1, "", "index write denied"
        return real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fail_reset)
    result = update_merge.finalize_managed_update_on_boot()
    assert result["finalized"] is False and "cleanup failed" in result["stash_note"]
    assert update_candidate.live_unmerged_paths() == [path]
    partial = (repo / path).read_bytes()
    (repo / "late.txt").write_text("later work\n", encoding="utf-8")
    monkeypatch.setattr(git_ops, "git_capture", real_capture)
    resumed = update_merge.finalize_managed_update_on_boot()
    assert resumed["stash_restore_status"] == "preserved"
    assert "unconfirmed" in resumed["stash_note"] and "cleanup failed" in resumed["stash_note"]
    assert (repo / path).read_bytes() == partial
    assert (repo / "late.txt").read_text(encoding="utf-8") == "later work\n"
    assert update_candidate.live_unmerged_paths() == [path]
    assert update_merge.active_update_tx() == {}  # normal owner repair is available
    assert workers.repo_writer_admission_closed() == ""
    assert worker_chat_lane.owner_conversation_admitted(1) is True


def test_missing_stash_list_entry_uses_pinned_object_and_missing_object_stays_incomplete(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "boot_preapply")
    assert _git(repo, "stash", "drop", "stash@{0}").returncode == 0
    result = update_merge.finalize_managed_update_on_boot()
    assert result["stash_restore_status"] == "restored"
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"
    assert _git(repo, "rev-parse", "rescue-local-" + tx["stash_sha"][:12]).stdout.strip() == tx["stash_sha"]
    update_merge.write_update_tx({**tx, "stash_sha": "f" * 40})
    missing = update_merge.finalize_managed_update_on_boot()
    assert missing["reason"] == "stash_recovery_incomplete"
    assert update_merge.active_update_tx()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"


def test_pin_failed_gateway_unwind_still_returns_local_work(tmp_path, monkeypatch):
    repo, branch = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, branch)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("owner work\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("owner scratch\n", encoding="utf-8")
    real_capture, failed_pins = git_ops.git_capture, []

    def fail_pin(cmd):
        if cmd[:3] == ["git", "branch", "-f"]:
            failed_pins.append(cmd)
            return 1, "", "ref write denied"
        return real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fail_pin)
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    tx, response = control._stash_local_work_fenced(
        branch=branch, base_sha=pre, target_sha=pre, plan={},
    )
    body = json.loads(response.body)
    assert tx is None and response.status_code == 409 and "pin" in body["error"]
    assert "local changes restored" in body["stash_note"]
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"
    assert (repo / "scratch.txt").read_text(encoding="utf-8") == "owner scratch\n"
    assert len(failed_pins) == 1  # restore did not impose another ref-write gate
    sha = _git(repo, "stash", "list", "--format=%H").stdout.strip()
    assert sha and _git(repo, "rev-parse", "--verify", "rescue-local-" + sha[:12]).returncode != 0
    assert not update_merge.active_update_tx()


def test_interrupted_restore_preserves_exact_stash_without_repin(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "boot_preapply")
    assert _git(repo, "branch", "-D", "rescue-local-" + tx["stash_sha"][:12]).returncode == 0
    tx["stash_restore"] = {"status": "applying", "stash_sha": tx["stash_sha"]}
    update_merge.write_update_tx(tx)
    (repo / "late.txt").write_text("later work\n", encoding="utf-8")
    monkeypatch.setattr(update_merge, "create_rescue_local_ref", lambda *_a: pytest.fail("restore tried to repin"))
    result = update_merge.finalize_managed_update_on_boot()
    assert result["stash_restore_status"] == "preserved" and "unconfirmed" in result["stash_note"]
    assert "git stash apply " + tx["stash_sha"] in result["stash_note"]
    assert "rescue-local-" not in result["stash_note"]
    assert (repo / "late.txt").read_text(encoding="utf-8") == "later work\n"
    assert tx["stash_sha"] in _git(repo, "stash", "list", "--format=%H").stdout
    assert not update_merge.active_update_tx()


def test_event_append_failure_retries_only_disclosure(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "boot_preapply")
    real_log = update_merge._log_supervisor
    monkeypatch.setattr(update_merge, "_log_supervisor", lambda row: False)
    failed = update_merge.finalize_managed_update_on_boot()
    assert failed["reason"] == "stash_recovery_incomplete"
    assert update_merge.read_update_tx()["stash_restored"] is True
    (repo / "a.txt").write_text("later edits\n", encoding="utf-8")
    monkeypatch.setattr(update_merge, "_log_supervisor", real_log)
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: pytest.fail(f"disclosure replay ran Git: {cmd}"))
    resumed = update_merge.finalize_managed_update_on_boot()
    assert resumed["stash_restore_status"] == "restored"
    assert (repo / "a.txt").read_text(encoding="utf-8") == "later edits\n"
    assert not update_merge.active_update_tx()


def test_secondary_completion_event_failure_keeps_the_deliverable_note(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "boot_preapply")
    real_append = update_merge.append_jsonl

    def append(path, row, **kwargs):
        if row["type"] == "managed_update_stash_recovered_on_boot":
            raise OSError("secondary event failure")
        return real_append(path, row, **kwargs)

    monkeypatch.setattr(update_merge, "append_jsonl", append)
    result = update_merge.finalize_managed_update_on_boot()
    assert result["stash_restore_status"] == "restored"
    assert "update stash was retained" in result["stash_note"]
    assert not update_merge.active_update_tx()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"
    assert any(row["type"] == "managed_update_stash_restored" for row in _events(tmp_path))


@pytest.mark.parametrize("census", [[], None])
def test_failed_apply_without_census_does_not_invent_conflict_names(tmp_path, monkeypatch, census):
    repo, tx = _fixture(tmp_path, monkeypatch, "boot_preapply")
    real_capture = git_ops.git_capture
    monkeypatch.setattr(update_candidate, "live_unmerged_paths", lambda: census)

    def fail_apply(cmd):
        return (1, "apply output", "apply failure") if cmd[:3] == ["git", "stash", "apply"] else real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fail_apply)
    result = update_merge.finalize_managed_update_on_boot()
    assert result["stash_restore_status"] == "preserved"
    assert "apply output" in result["stash_note"] and "apply failure" in result["stash_note"]
    assert ("inventory could not be read" if census is None else "without tracked conflicts") in result["stash_note"]
    event = next(row for row in _events(tmp_path) if row["type"] == "managed_update_stash_restore_failed")
    assert event["conflict_paths"] == census


def test_gateway_exception_after_restore_does_not_enter_destructive_rollback(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "gateway_unwind")
    real_active = update_merge.active_update_tx
    first_check = [True]

    def active():
        if first_check:
            first_check.pop()
            return {}
        return real_active()

    monkeypatch.setattr(update_merge, "active_update_tx", active)
    monkeypatch.setattr(update_merge, "clear_update_tx", lambda: False)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: [])
    monkeypatch.setattr(control, "_stash_local_work_fenced", lambda **kw: (tx, None))
    monkeypatch.setattr(control, "_rollback_fenced_update", lambda *a, **kw: pytest.fail("destructive generic rollback"))
    plan = {"available": True, "kind": "clean", "base_sha": tx["pre_update_sha"],
            "target_sha": tx["target_sha"]}

    def plan_then_fail(**kwargs):
        if kwargs.get("build"):
            control._unwind_stashed_update(tx, "exception_test")
            raise RuntimeError("response delivery failed")
        return plan

    monkeypatch.setattr(update_merge, "plan_managed_update_merge", plan_then_fail)
    response = control._apply_smart_update_fenced(
        object(), expected_base_sha=tx["pre_update_sha"], expected_target_sha=tx["target_sha"],
    )
    body = json.loads(response.body)
    assert response.status_code == 500 and body["reason"] == "stash_recovery_incomplete"
    assert "stash" in body["stash_note"] and body["restart_required"] is True
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"
    assert real_active()["stash_restored"] is True


@pytest.mark.parametrize("ok", [True, False])
def test_gateway_rollback_disclosure_uses_the_actual_stash_note_consumer(monkeypatch, ok):
    note = "rolled back to abc; conflicting paths: draft.txt; recover with git stash apply def"
    monkeypatch.setattr(update_merge, "rollback_managed_update", lambda reason: (ok, note))
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    monkeypatch.setattr(update_merge, "mark_update_tx_gate_blocked", lambda *a: None)
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    response = control._rollback_fenced_update("test", "smoke failed")
    body = json.loads(response.body)
    assert body["rolled_back"] is ok
    assert body["stash_note"] == note == body["rollback"]


@pytest.mark.parametrize("restore_status", [None, "applying", "restored", "preserved"])
def test_restart_keeps_ordinary_boot_smoke_but_never_resets_pending_restore(tmp_path, monkeypatch, restore_status):
    from ouroboros.server_restart import _safe_restart_serialized

    repo, tx = _fixture(tmp_path, monkeypatch)
    if restore_status:
        # Both the uncertain window and a recorded restore awaiting event/tx
        # cleanup may hold returned work. The current reader must refuse too.
        tx["stash_restore"] = {"status": restore_status, "stash_sha": tx["stash_sha"]}
        update_merge.write_update_tx(tx)
        assert _git(repo, "stash", "apply", tx["stash_sha"]).returncode == 0
        (repo / "a.txt").write_text("returned work plus later edits\n", encoding="utf-8")
    before = (repo / "a.txt").read_bytes()
    calls = []

    def destructive_restart(**kwargs):
        calls.append(kwargs)
        assert _git(repo, "reset", "--hard", "HEAD").returncode == 0
        assert _git(repo, "clean", "-fd").returncode == 0
        return True, "restarted"

    ok, message = _safe_restart_serialized(
        destructive_restart, reason="owner_restart", unsynced_policy="rescue_and_reset",
    )
    if restore_status:
        assert ok is False and not calls
        assert "Quit and reopen" in message and "server process" in message
        assert (repo / "a.txt").read_bytes() == before
    else:
        assert ok is True and calls == [{"reason": "owner_restart", "unsynced_policy": "rescue_and_reset"}]


def _checkout_state(tmp_path, monkeypatch, tx):
    from supervisor import state

    data = tmp_path / "data"
    for name, path in {"DRIVE_ROOT": data, "STATE_PATH": data / "state/state.json",
                       "STATE_LAST_GOOD_PATH": data / "state/state.last_good.json",
                       "STATE_LOCK_PATH": data / "locks/state.lock"}.items():
        monkeypatch.setattr(state, name, path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    assert state.init_state().quality == "current"
    state.update_state(lambda row: row.update(current_sha=tx["target_sha"],
                                             current_branch=tx["pre_update_branch"], evolution_cycle=17))
    return state


def _boot_chain(tmp_path, monkeypatch, *, fixture=None):
    """Real Git/bootstrap/finalizer with process exit, feeds and dependencies inert."""
    import server
    from ouroboros import update_letter
    from supervisor import evolution_lifecycle

    repo, tx = fixture or _fixture(tmp_path, monkeypatch, "rollback")
    (repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(json.dumps({
        "managed_local_branch": tx["pre_update_branch"], "managed_local_stable_branch": "stable",
    }), encoding="utf-8")
    state = _checkout_state(tmp_path, monkeypatch, tx)
    restarts = []
    monkeypatch.setattr(server, "REPO_DIR", repo)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(server, "_runtime_branch_defaults", lambda: (tx["pre_update_branch"], "stable"))
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *_a: None)
    monkeypatch.setattr(server, "_launcher_managed_repo_matches", lambda: True)
    monkeypatch.setattr(server, "_has_active_evolution_transaction", lambda: False)
    monkeypatch.setattr(server, "_wait_for_supervisor_update_finalize", lambda: True)
    monkeypatch.setattr(server, "_request_restart_exit", lambda: restarts.append(True))
    monkeypatch.setattr(server, "broadcast_ws_sync", lambda *_a: None)
    monkeypatch.setattr(git_ops, "init", lambda **_kw: None)  # roots already pinned
    monkeypatch.setattr(git_ops, "_has_remote", lambda *_a: False)
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda **_kw: (True, "fixture"))
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": True})
    monkeypatch.setattr(git_ops, "compute_managed_update_status", lambda **_kw: {"available": False})
    monkeypatch.setattr(update_letter, "refresh_after_check", lambda *_a: None)
    monkeypatch.setattr(update_letter, "runtime_status", lambda value: value)
    monkeypatch.setattr(evolution_lifecycle, "disable_evolution_projection", lambda: None)
    monkeypatch.setattr(evolution_lifecycle, "pause_evolution_campaign", lambda *_a: None)
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.delenv("OUROBOROS_DISABLE_MANAGED_UPDATES", raising=False)
    return server, repo, tx, state, restarts


def test_boot_rollback_restored_work_survives_its_required_restart(tmp_path, monkeypatch):
    from supervisor import worker_pool_lifecycle, workers

    server, repo, tx, state, restarts = _boot_chain(tmp_path, monkeypatch)
    assert state.load_state()["current_sha"] == tx["target_sha"]
    server._boot_managed_update_tasks()
    assert restarts == [True]
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"
    assert (repo / "scratch.txt").read_text(encoding="utf-8") == "owner scratch\n"
    handoff = update_merge.read_update_tx()
    assert handoff["phase"] == update_merge.MARKER_CLEANUP_RETRY_PHASE
    assert handoff["gate_blocked_reason"] == "rollback_restart_pending"
    assert handoff["_schema_version"] == 1 and handoff["stash_restored"] is True
    assert "stash_restore" not in handoff and handoff["stash_sha"] == tx["stash_sha"]

    # The new process first bootstraps, THEN finishes the handoff. A missing
    # marker here used to let actual managed bootstrap reset/clean both files.
    (repo / "a.txt").write_text("owner work plus later edit\n", encoding="utf-8")
    bootstrap_ok, _ = server._bootstrap_supervisor_repo({})
    assert bootstrap_ok is False  # the existing phase gate skips checkout/reset
    server._boot_managed_update_tasks()
    assert restarts == [True] and not update_merge.active_update_tx()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work plus later edit\n"
    assert (repo / "scratch.txt").read_text(encoding="utf-8") == "owner scratch\n"
    assert tx["stash_sha"] in _git(repo, "stash", "list", "--format=%H").stdout
    saved = json.loads(state.STATE_PATH.read_text(encoding="utf-8"))
    assert saved["current_sha"] == tx["pre_update_sha"] and saved["current_branch"] == tx["pre_update_branch"]
    assert saved["evolution_cycle"] == 17
    monkeypatch.setattr(workers, "REPO_DIR", repo)
    assert worker_pool_lifecycle._worker_sha_relation(saved["current_sha"], tx["pre_update_sha"])["relation"] == "equal"

    # After the pending handoff closes, unrelated later manual Restart keeps
    # its established reset policy. This fix grants no permanent dirty bypass.
    assert server._bootstrap_supervisor_repo({})[0] is True
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base\n"
    assert not (repo / "scratch.txt").exists()


def test_assisted_boot_with_pre_update_head_reexecs_once_after_rollback(tmp_path, monkeypatch):
    from ouroboros import server_process
    from tests.test_update_merge_assisted import _materialized_conflict_tx

    repo, _branch, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    _stub_worker_gates(monkeypatch)
    tx["resolution_attempts"] = 4
    update_merge.write_update_tx(tx)
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == plan["base_sha"]
    assert update_merge._merge_head_sha() == plan["target_sha"]
    assert _git(repo, "status", "--porcelain").stdout.strip()
    assert (repo / "a.txt").read_text(encoding="utf-8") != "base\n"
    monkeypatch.setattr(server_process, "_source_baseline", plan["base_sha"])
    server, repo, tx, _state, restarts = _boot_chain(tmp_path, monkeypatch, fixture=(repo, tx))
    server._boot_managed_update_tasks()
    assert restarts == [True] and update_merge.active_update_tx()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base\n"
    assert update_merge._merge_head_sha() == ""
    assert server._bootstrap_supervisor_repo({})[0] is False
    server._boot_managed_update_tasks()
    assert restarts == [True] and not update_merge.active_update_tx()


def test_failed_rollback_does_not_stamp_an_unreached_checkout(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "rollback")
    state = _checkout_state(tmp_path, monkeypatch, tx)
    before = state.STATE_PATH.read_bytes()
    real_capture = git_ops.git_capture

    def fail_checkout(cmd):
        return (1, "", "checkout denied") if cmd[:3] == ["git", "checkout", "-B"] else real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fail_checkout)
    assert update_merge.rollback_managed_update("checkout-failure")[0] is False
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == tx["target_sha"]
    assert state.STATE_PATH.read_bytes() == before


def test_unavailable_state_is_disclosed_without_blocking_verified_rollback(tmp_path, monkeypatch, caplog):
    from supervisor.state import StateUnavailable

    repo, tx = _fixture(tmp_path, monkeypatch, "rollback")
    state = _checkout_state(tmp_path, monkeypatch, tx)
    before = state.STATE_PATH.read_bytes()

    def unavailable(_mutate):
        raise StateUnavailable("lock_timeout", "injected")

    monkeypatch.setattr(git_ops, "update_state", unavailable)
    ok, _note = update_merge.rollback_managed_update("state-unavailable", reopen_writer_admission=False)
    assert ok is True and "Checkout facts not recorded in state" in caplog.text
    assert state.STATE_PATH.read_bytes() == before
    assert (repo / "a.txt").read_text(encoding="utf-8") == "owner work\n"
    assert update_merge.read_update_tx()["gate_blocked_reason"] == "rollback_restart_pending"


@pytest.mark.parametrize("fail_stage", ["event", "handoff"])
def test_rollback_does_not_release_custody_before_handoff_is_durable(tmp_path, monkeypatch, fail_stage):
    repo, tx = _fixture(tmp_path, monkeypatch, "rollback")
    real_log, real_write = update_merge._log_supervisor, update_merge.write_update_tx

    def log(row):
        return False if row["type"] == "managed_update_rolled_back" else real_log(row)

    def write(payload):
        if payload.get("gate_blocked_reason") == "rollback_restart_pending":
            raise OSError("handoff write failed")
        real_write(payload)

    monkeypatch.setattr(update_merge, "_log_supervisor", log if fail_stage == "event" else real_log)
    monkeypatch.setattr(update_merge, "write_update_tx", write if fail_stage == "handoff" else real_write)
    ok, message = update_merge.rollback_managed_update("test", reopen_writer_admission=False)
    assert ok is False and "could not be saved" in message
    saved = update_merge.read_update_tx()
    assert saved["_schema_version"] == 2 and saved["stash_restored"] is True
    assert saved["phase"] == "rolling_back"
    (repo / "a.txt").write_text("later work\n", encoding="utf-8")
    monkeypatch.setattr(update_merge, "_log_supervisor", real_log)
    monkeypatch.setattr(update_merge, "write_update_tx", real_write)
    assert update_merge.rollback_managed_update("retry", reopen_writer_admission=False)[0] is True
    assert update_merge.read_update_tx()["_schema_version"] == 1
    assert (repo / "a.txt").read_text(encoding="utf-8") == "later work\n"


def test_completed_preservation_handoff_never_claims_a_successful_apply(tmp_path, monkeypatch):
    repo, tx = _fixture(tmp_path, monkeypatch, "rollback")
    real_capture = git_ops.git_capture

    def fail_apply(cmd):
        return (1, "", "apply failed") if cmd[:3] == ["git", "stash", "apply"] else real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fail_apply)
    ok, message = update_merge.rollback_managed_update("test", reopen_writer_admission=False)
    assert ok is True and "apply failed" in message
    handoff = update_merge.read_update_tx()
    assert handoff["phase"] == update_merge.MARKER_CLEANUP_RETRY_PHASE
    assert handoff["_schema_version"] == 1 and not handoff.get("stash_restored")
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: pytest.fail(f"handoff replay mutated Git: {cmd}"))
    result = update_merge.finalize_managed_update_on_boot()
    assert "apply failed" in result["stash_note"] and not result.get("rolled_back")
    assert not update_merge.active_update_tx()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base\n"
