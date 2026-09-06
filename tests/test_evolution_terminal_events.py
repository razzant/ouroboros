"""The terminal event write: which campaign it may touch, and what it must leave alone.

Split out of ``tests/test_evolution_state_integrity_v3.py`` by theme: the terminal that
cannot write into a different campaign, the metadata-less one that cannot mutate the
active campaign, the duplicates that resume pending cleanup and a missing restart request,
the exact model reason a restart preserves, the serialized concurrent pause, and the
exception and rejection paths that leave no lifecycle side effects.
"""

from __future__ import annotations

import json
import pathlib
import threading
from types import SimpleNamespace

from tests._evolution_state_shared import (
    _CaptureQueue,
    _active_transaction,
)


def test_terminal_event_cannot_write_into_a_different_campaign(tmp_path):
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path, task_id="same-task")
    stale = {
        **tx,
        "campaign_id": "old-campaign",
        "transaction_id": "old-transaction",
    }

    result = evolution_lifecycle.update_evolution_campaign_after_task(
        "same-task",
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}},
        rounds=1,
        transaction=stale,
    )

    assert result == {
        "accepted": False,
        "persisted": False,
        "replay": False,
        "reason": "transaction_mismatch",
        "transaction": {},
    }
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["id"] == campaign["id"]
    assert stored["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert stored.get("history", []) == []


def test_metadata_less_terminal_cannot_mutate_active_campaign(tmp_path):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")

    result = evolution_lifecycle.update_evolution_campaign_after_task(
        "stale-task",
        cost_usd=1.25,
        outcome_axes={"execution": {"status": "failed"}},
        rounds=1,
    )

    assert result == {
        "accepted": False,
        "persisted": False,
        "replay": False,
        "reason": "transaction_missing",
        "transaction": {},
    }
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["id"] == campaign["id"]
    assert stored["cycles_done"] == 0
    assert stored["budget_spent_usd"] == 0.0
    assert stored.get("history", []) == []


def test_duplicate_terminal_resumes_pending_cleanup_and_owner_report(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle

    _campaign, tx = _active_transaction(tmp_path)
    assert evolution_lifecycle.update_evolution_transaction(
        tx["task_id"], rescue_ref="refs/ouroboros/rescue/test",
    )
    real_resume = evolution_lifecycle._resume_evolution_terminal_effects
    monkeypatch.setattr(
        evolution_lifecycle,
        "_resume_evolution_terminal_effects",
        lambda _campaign_id, _task_id, value: dict(value),
    )

    first = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "failed"}},
        rounds=1,
        transaction=tx,
    )

    assert first["persisted"] is True
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["history"][0]["transaction"]["cleanup_status"] == "pending"
    assert stored["pending_owner_report"]["cycle_outcome"] == "abandoned"

    cleanup_calls = []
    reports = []

    def _cleanup(value, *_args, **_kwargs):
        cleanup_calls.append(value["transaction_id"])
        value["cleanup_status"] = "already_clean"

    monkeypatch.setattr(evolution_lifecycle, "_resume_evolution_terminal_effects", real_resume)
    monkeypatch.setattr(evolution_lifecycle, "_cleanup_worktree_after_cycle", _cleanup)
    monkeypatch.setattr(
        evolution_lifecycle,
        "notify_owner_cycle_outcome",
        lambda campaign, value: reports.append((campaign["id"], value["cycle_outcome"])),
    )

    replay = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "failed"}},
        rounds=1,
        transaction=tx,
    )

    assert replay["replay"] is True
    assert cleanup_calls == [tx["transaction_id"]]
    assert reports == [(_campaign["id"], "abandoned")]
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["history"][0]["transaction"]["cleanup_status"] == "already_clean"
    assert "pending_owner_report" not in stored


def test_duplicate_terminal_resumes_missing_restart_request(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle

    _campaign, tx = _active_transaction(tmp_path)
    receipt = evolution_lifecycle.record_evolution_commit(
        campaign_id=tx["campaign_id"],
        transaction_id=tx["transaction_id"],
        task_id=tx["task_id"],
        commit_sha="a" * 40,
    )
    assert receipt["ok"] is True
    real_resume = evolution_lifecycle._resume_evolution_terminal_effects
    monkeypatch.setattr(
        evolution_lifecycle,
        "_resume_evolution_terminal_effects",
        lambda _campaign_id, _task_id, value: dict(value),
    )

    first = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}},
        rounds=1,
        transaction=tx,
    )

    assert first["transaction"]["cycle_outcome"] == "waiting_for_restart"
    restart_calls = []
    monkeypatch.setattr(evolution_lifecycle, "_resume_evolution_terminal_effects", real_resume)
    monkeypatch.setattr(
        evolution_lifecycle,
        "request_evolution_restart",
        lambda drive_root, value, log=None: restart_calls.append(
            (pathlib.Path(drive_root), value["commit_sha"])
        ),
    )

    replay = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}},
        rounds=1,
        transaction=tx,
    )

    assert replay["replay"] is True
    assert restart_calls == [(tmp_path, "a" * 40)]


def test_terminal_restart_preserves_exact_model_reason(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, workers

    campaign, tx = _active_transaction(tmp_path)
    sha = "c" * 40
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    current_tx = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({
        "expected_sha": sha,
        "reason": "apply reviewed evolution",
        "evolution_claim": claim,
    }))
    events = _CaptureQueue()
    monkeypatch.setenv("OUROBOROS_EVOLUTION_AUTO_RESTART", "true")
    monkeypatch.setattr(workers, "get_event_q", lambda: events)

    evolution_lifecycle.request_evolution_restart(tmp_path, current_tx)

    assert json.loads(marker.read_text())["reason"] == "apply reviewed evolution"
    assert len(events.items) == 1
    assert events.items[0]["reason"] == "apply reviewed evolution"
    assert events.items[0]["evolution_restart"] is True


def test_auto_restart_off_still_writes_the_exact_restart_marker(tmp_path, monkeypatch):
    """W4-F3 (owner 5 = A, 2026-09-04): ``OUROBOROS_EVOLUTION_AUTO_RESTART`` off skips
    ONLY the restart. The exact claim marker is still written, so the owner's manual
    restart verifies the cycle by exact claim instead of the markerless reconcile."""
    from supervisor import evolution_lifecycle, workers

    campaign, tx = _active_transaction(tmp_path)
    sha = "d" * 40
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    current_tx = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    events = _CaptureQueue()
    monkeypatch.setenv("OUROBOROS_EVOLUTION_AUTO_RESTART", "false")
    monkeypatch.setattr(workers, "get_event_q", lambda: events)

    evolution_lifecycle.request_evolution_restart(tmp_path, current_tx)

    marker = json.loads((tmp_path / "state" / "pending_restart_verify.json").read_text())
    assert marker["expected_sha"] == sha
    assert marker["reason"] == "supervisor_auto_evolution_restart"
    assert marker["evolution_claim"] == {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert events.items == []  # only the restart itself is skipped


def test_both_restart_marker_writers_share_one_schema(tmp_path):
    """The agent's ``restart`` tool and the supervisor write the marker through one
    helper: the same keys, values stripped of the git newline, and the claim key
    only for an exact claim — an empty one would read as a claim mismatch at boot."""
    from supervisor.evolution_lifecycle import write_pending_restart_marker

    sha = "e" * 40
    path = write_pending_restart_marker(
        tmp_path, expected_sha=sha + "\n", expected_branch="ouroboros\n", reason="agent_requested_restart",
    )
    assert path == tmp_path / "state" / "pending_restart_verify.json"
    plain = json.loads(path.read_text())
    assert set(plain) == {"ts", "expected_sha", "expected_branch", "reason"}
    assert (plain["expected_sha"], plain["expected_branch"]) == (sha, "ouroboros")
    claim = {"campaign_id": "c", "transaction_id": "t", "task_id": "k", "commit_sha": sha}
    write_pending_restart_marker(tmp_path, expected_sha=sha, expected_branch="ouroboros", reason="r",
                                 evolution_claim=claim)
    assert json.loads(path.read_text())["evolution_claim"] == claim
    write_pending_restart_marker(tmp_path, expected_sha=sha, expected_branch="ouroboros", reason="r",
                                 evolution_claim={})
    assert "evolution_claim" not in json.loads(path.read_text())


def test_terminal_write_serializes_concurrent_campaign_pause(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle

    _campaign, tx = _active_transaction(tmp_path)
    real_write = evolution_lifecycle._write_evolution_campaign
    entered = threading.Event()
    release = threading.Event()
    terminal_result = {}
    pause_result = {}

    def _hold_terminal_write(data, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return real_write(data, **kwargs)

    monkeypatch.setattr(evolution_lifecycle, "_write_evolution_campaign", _hold_terminal_write)
    monkeypatch.setattr(
        evolution_lifecycle,
        "_cleanup_worktree_after_cycle",
        lambda tx, *_a, **_k: tx.update(cleanup_status="already_clean"),
    )

    def _terminal():
        terminal_result.update(evolution_lifecycle.update_evolution_campaign_after_task(
            tx["task_id"],
            cost_usd=1.0,
            outcome_axes={"execution": {"status": "ok"}},
            rounds=1,
            transaction=tx,
        ))

    terminal_thread = threading.Thread(target=_terminal)
    terminal_thread.start()
    assert entered.wait(timeout=2)

    def _pause():
        pause_result.update(evolution_lifecycle.pause_evolution_campaign("concurrent pause"))

    pause_thread = threading.Thread(target=_pause)
    pause_thread.start()
    pause_thread.join(timeout=0.05)
    assert pause_thread.is_alive()
    release.set()
    terminal_thread.join(timeout=2)
    pause_thread.join(timeout=2)

    assert terminal_result["persisted"] is True
    assert pause_result["status"] == "paused"
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["status"] == "paused"
    assert stored["pause_reason"] == "concurrent pause"
    assert stored["history"][0]["task_id"] == tx["task_id"]


def test_terminal_write_exception_has_no_lifecycle_side_effects(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle

    _campaign, tx = _active_transaction(tmp_path)
    side_effects = []
    monkeypatch.setattr(
        evolution_lifecycle,
        "_write_evolution_campaign",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        evolution_lifecycle,
        "_cleanup_worktree_after_cycle",
        lambda *_a, **_k: side_effects.append("cleanup"),
    )
    monkeypatch.setattr(
        evolution_lifecycle,
        "notify_owner_cycle_outcome",
        lambda *_a, **_k: side_effects.append("notify"),
    )

    result = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}},
        rounds=1,
        transaction=tx,
    )

    assert result["persisted"] is False
    assert result["reason"] == "campaign_write_failed"
    assert side_effects == []


def test_rejected_terminal_does_not_consume_global_evolution_state(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, state
    from supervisor.events import _handle_evolution_task_done

    state.init(tmp_path)
    state.update_state(lambda live: live.update(
        evolution_mode_enabled=True,
        post_task_autostop=True,
        evolution_consecutive_failures=4,
    ))
    monkeypatch.setattr(
        evolution_lifecycle,
        "update_evolution_campaign_after_task",
        lambda *_a, **_k: {
            "accepted": True, "persisted": False, "replay": False,
            "reason": "campaign_write_refused", "transaction": {},
        },
    )
    checkpoints = []
    monkeypatch.setattr(
        "ouroboros.evolution_checkpoints.append_evolution_checkpoint",
        lambda *_a, **_k: checkpoints.append(True),
    )
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, REPO_DIR=tmp_path)
    task = {"metadata": {"evolution_transaction": {"transaction_id": "stale"}}}

    _handle_evolution_task_done(
        ctx,
        evt={},
        task_id="stale",
        task=task,
        task_done_event={"status": "failed"},
        outcome_axes={"execution": {"status": "failed"}},
        cost=1.0,
        rounds=1,
    )

    live = state.load_state()
    assert live["evolution_mode_enabled"] is True
    assert live["post_task_autostop"] is True
    assert live["evolution_consecutive_failures"] == 4
    assert checkpoints == []
