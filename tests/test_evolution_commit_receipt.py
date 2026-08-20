"""The exact commit receipt: what binds it, what may read it, and what may never erase it.

Split out of ``tests/test_evolution_state_integrity_v3.py`` by theme: the receipt bound to
campaign, transaction and task; the second commit blocked before review; the receipt race
ahead of the git commit; revoked authority; the rescue link and campaign sidecar that share
the CAS; and the stale, terminal and panicking writers that must not overwrite it.
"""

from __future__ import annotations

import pathlib
import threading
from types import SimpleNamespace

import pytest

from tests._evolution_state_shared import (
    _active_transaction,
    _patch_commit_seam,
)


def test_exact_commit_receipt_is_bound_to_campaign_transaction_and_task(tmp_path):
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
    }
    assert evolution_lifecycle.check_evolution_authority(**claim)["ok"] is True

    receipt = evolution_lifecycle.record_evolution_commit(**claim, commit_sha="a" * 40)

    assert receipt["ok"] is True
    assert receipt["commit_sha"] == "a" * 40
    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["commit_receipt"] == receipt
    assert evolution_lifecycle.check_evolution_authority(
        **claim, commit_sha="b" * 40,
    )["reason"] == "commit_receipt_mismatch"

    campaign_state = evolution_lifecycle._read_evolution_campaign()
    campaign_state["active_transaction"].pop("commit_receipt")
    assert evolution_lifecycle._write_evolution_campaign(campaign_state) is True
    assert evolution_lifecycle.check_evolution_authority(
        **claim, commit_sha="a" * 40,
    )["reason"] == "commit_receipt_missing"


def test_second_evolution_commit_is_blocked_before_review(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], "a" * 40,
    )["ok"] is True
    review_calls = []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    _patch_commit_seam(monkeypatch, "_run_reviewed_stage_cycle",
        lambda *a, **k: review_calls.append(True) or {"status": "passed"},
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        current_task_type="evolution",
        task_id=tx["task_id"],
        task_metadata={"evolution_transaction": tx},
    )

    result = git_tools._repo_commit_push(ctx, "second commit")

    assert "transaction_already_committed" in result
    assert "No reviewer was called" in result
    assert review_calls == []


def test_receipt_race_blocks_evolution_before_git_commit(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        current_task_type="evolution",
        task_id=tx["task_id"],
        task_metadata={"evolution_transaction": tx},
    )
    _patch_commit_seam(monkeypatch, "_record_commit_attempt", lambda *a, **k: None)
    claim, error = git_tools._check_evolution_commit_stage(
        ctx, "commit", 0.0, phase="pre_review_authority",
    )
    assert error == ""
    assert evolution_lifecycle.record_evolution_commit(
        **claim, commit_sha="b" * 40,
    )["ok"] is True

    _claim, error = git_tools._check_evolution_commit_stage(
        ctx, "commit", 0.0, phase="pre_commit_authority",
    )

    assert "transaction_already_committed" in error
    assert "Nothing was committed" in error


def test_revoked_authority_leaves_commit_unrecorded(tmp_path):
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    live = state.load_state()
    live["evolution_mode_enabled"] = False
    live["evolution_owner_stopped"] = True
    state.save_state(live)

    receipt = evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], "c" * 40,
    )

    assert receipt == {"ok": False, "reason": "owner_stopped", "commit_sha": "c" * 40}
    assert evolution_lifecycle._read_evolution_campaign()["active_transaction"]["commit_sha"] == ""


def test_exact_receipt_remains_authority_after_post_task_autostop(tmp_path):
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    sha = "9" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
    }
    assert evolution_lifecycle.record_evolution_commit(**claim, commit_sha=sha)["ok"] is True
    state.update_state(lambda live: live.update(
        evolution_mode_enabled=False,
        post_task_autostop=False,
    ))

    assert evolution_lifecycle.check_evolution_authority(
        **claim, commit_sha=sha,
    )["ok"] is True
    assert evolution_lifecycle.check_evolution_authority(**claim)["reason"] == "evolution_disabled"


@pytest.mark.parametrize("held_lock", ["state", "campaign"])
def test_rescue_link_uses_shared_campaign_cas_and_preserves_commit_receipt(
    tmp_path, monkeypatch, held_lock,
):
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )
    from ouroboros.utils import atomic_write_json
    from supervisor import evolution_lifecycle, git_ops, state

    campaign, tx = _active_transaction(tmp_path)
    sha = "3" * 40
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    monkeypatch.setattr(evolution_lifecycle, "EVOLUTION_CAMPAIGN_CAS_TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    if held_lock == "state":
        lock_path = tmp_path / "locks" / "state.lock"
        lock_fd = state.acquire_file_lock(lock_path, timeout_sec=1.0)
        release = state.release_file_lock
    else:
        lock_path = campaign_path.with_name(campaign_path.name + ".lock")
        lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
        release = release_exclusive_file_lock
    assert lock_fd is not None
    done = threading.Event()

    def _link() -> None:
        git_ops._link_rescue_to_evolution_transaction(
            {"rescue_ref": "rescue/test", "path": "/tmp/rescue-test"},
            "test",
        )
        done.set()

    thread = threading.Thread(target=_link, daemon=True)
    thread.start()
    try:
        assert done.wait(0.1) is False
        current = evolution_lifecycle._read_evolution_campaign()
        current["active_transaction"]["interleaved"] = held_lock
        atomic_write_json(campaign_path, current, trailing_newline=True)
    finally:
        release(lock_path, lock_fd)
    assert done.wait(2.0) is True
    thread.join(timeout=1.0)

    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["commit_sha"] == sha
    assert stored["commit_receipt"]["commit_sha"] == sha
    assert stored["rescue_ref"] == "rescue/test"
    assert stored["interleaved"] == held_lock


def test_commit_receipt_uses_campaign_sidecar_before_rescue(tmp_path, monkeypatch):
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    monkeypatch.setattr(evolution_lifecycle, "EVOLUTION_CAMPAIGN_CAS_TIMEOUT_SEC", 1.0)
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    lock_path = campaign_path.with_name(campaign_path.name + ".lock")
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
    assert lock_fd is not None
    done = threading.Event()
    result = {}

    def _record() -> None:
        result.update(evolution_lifecycle.record_evolution_commit(
            campaign["id"], tx["transaction_id"], tx["task_id"], "4" * 40,
        ))
        done.set()

    thread = threading.Thread(target=_record, daemon=True)
    thread.start()
    try:
        assert done.wait(0.1) is False
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)
    assert done.wait(2.0) is True
    thread.join(timeout=1.0)
    assert result["ok"] is True
    assert evolution_lifecycle._read_evolution_campaign()["active_transaction"][
        "commit_receipt"
    ]["commit_sha"] == "4" * 40


def test_campaign_sidecar_contention_releases_state_lock_quickly(tmp_path):
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    sidecar = campaign_path.with_name(campaign_path.name + ".lock")
    sidecar_fd = acquire_exclusive_file_lock(sidecar, timeout_sec=1.0)
    assert sidecar_fd is not None
    try:
        result = evolution_lifecycle.record_evolution_commit(
            campaign["id"], tx["transaction_id"], tx["task_id"], "6" * 40,
        )
        assert result["ok"] is False
        state_fd = state.acquire_file_lock(state.STATE_LOCK_PATH, timeout_sec=0.2)
        assert state_fd is not None
        state.release_file_lock(state.STATE_LOCK_PATH, state_fd)
    finally:
        release_exclusive_file_lock(sidecar, sidecar_fd)


def test_sent_owner_report_clear_cannot_erase_concurrent_commit_receipt(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue

    campaign, tx = _active_transaction(tmp_path)
    report = {"cycle_outcome": "absorbed", "task_id": "previous"}
    current = evolution_lifecycle._read_evolution_campaign()
    current["pending_owner_report"] = report
    assert evolution_lifecycle._write_evolution_campaign(current) is True

    def _send_then_record(*args, **kwargs):
        receipt = evolution_lifecycle.record_evolution_commit(
            campaign["id"], tx["transaction_id"], tx["task_id"], "5" * 40,
        )
        assert receipt["ok"] is True

    monkeypatch.setattr(queue, "notify_owner_cycle_outcome", _send_then_record)

    queue._deliver_pending_owner_report()

    stored = evolution_lifecycle._read_evolution_campaign()
    assert "pending_owner_report" not in stored
    assert stored["active_transaction"]["commit_receipt"]["commit_sha"] == "5" * 40


def test_terminal_campaign_cannot_be_resurrected_by_a_stale_writer(tmp_path):
    from supervisor import evolution_lifecycle

    campaign, _ = _active_transaction(tmp_path)
    stale = dict(campaign)
    evolution_lifecycle.complete_evolution_campaign("owner stop", cleanup_worktree=False)
    stale["status"] = "active"

    assert evolution_lifecycle._write_evolution_campaign(stale) is False
    assert evolution_lifecycle._read_evolution_campaign()["status"] == "stopped"


def test_stale_campaign_cannot_overwrite_a_new_campaign(tmp_path):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    first = evolution_lifecycle.start_evolution_campaign("First", source="test")
    stale = dict(first)
    evolution_lifecycle.complete_evolution_campaign("done", cleanup_worktree=False)
    second = evolution_lifecycle.start_evolution_campaign("Second", source="test")

    stale["status"] = "active"
    assert evolution_lifecycle._write_evolution_campaign(stale) is False
    assert evolution_lifecycle._read_evolution_campaign()["id"] == second["id"]


def test_panic_campaign_close_uses_nonblocking_state_lock(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    timeouts = []
    monkeypatch.setattr(
        state,
        "acquire_file_lock",
        lambda path, timeout_sec=4.0, **kw: timeouts.append(timeout_sec) or None,
    )

    evolution_lifecycle.complete_evolution_campaign(
        "panic stop", status="stopped", cleanup_worktree=False,
    )

    assert timeouts == [0.001]
