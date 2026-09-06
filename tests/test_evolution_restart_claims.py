"""The restart claim: who may take it, who must wait, and what boot reconciliation may revive.

Split out of ``tests/test_evolution_state_integrity_v3.py`` by theme: the exact active
receipt a restart requires, the v2 claim verified only after a new generation, the losers
that wait instead of bypassing, the dead claim reclaimed, the write failures that restore
the claim for retry, the owner-stopped campaign boot cannot resurrect, and the supervisor
rechecks that gate an evolution restart against a stale marker or a moved head.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from types import SimpleNamespace

import pytest

from tests._evolution_state_shared import _active_transaction, _patch_commit_seam


def test_restart_requires_the_exact_active_commit_receipt(tmp_path, monkeypatch):
    from ouroboros.tools import control_runtime as control
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    sha = "e" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
    }
    assert evolution_lifecycle.record_evolution_commit(**claim, commit_sha=sha)["ok"] is True
    monkeypatch.setattr(
        control,
        "run_cmd",
        lambda cmd, cwd=None: "" if cmd[:2] == ["git", "status"] else sha,
    )
    ctx = SimpleNamespace(
        current_task_type="evolution",
        repo_dir=tmp_path,
        task_id=tx["task_id"],
        task_metadata={"evolution_transaction": tx},
        last_reviewed_commit_sha=sha,
    )
    assert control._evolution_restart_block_reason(ctx) == ""

    live = state.load_state()
    live["evolution_owner_stopped"] = True
    live["evolution_mode_enabled"] = False
    state.save_state(live)
    assert "owner_stopped" in control._evolution_restart_block_reason(ctx)


def test_boot_restart_verifies_exact_v2_claim_only_after_new_generation(
    tmp_path, monkeypatch,
):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "server-a"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    assert tx["schema_version"] == 2
    sha = "8" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)

    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert int(stored.get("absorbed_cycles_done") or 0) == 0
    assert marker.is_file()

    generation["value"] = "server-b"
    agent_startup_checks.verify_restart(env, sha)

    stored = evolution_lifecycle._read_evolution_campaign()
    assert "active_transaction" not in stored
    assert stored["transaction_history"][-1]["cycle_outcome"] == "absorbed"
    assert stored["last_boot_reconcile_gen"] == "server-b"
    assert not marker.exists()


def test_boot_restart_rejects_mismatched_claim_without_loser_bypass(
    tmp_path, monkeypatch,
):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    sha = "9" * 40
    exact = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**exact)["ok"] is True
    stale = {**exact, "transaction_id": "stale-transaction"}
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({"expected_sha": sha, "evolution_claim": stale}))
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "boot-gen")
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)
    agent_startup_checks.verify_restart(env, sha)  # a rename loser must not reconcile it

    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert stored["active_transaction"]["restart_authority_error"] == "restart_claim_mismatch"
    assert int(stored.get("absorbed_cycles_done") or 0) == 0
    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "restart_verify"
    assert event["error"] == "restart_claim_mismatch"


def test_boot_markerless_v2_missing_receipt_stays_unresolved(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    stored = evolution_lifecycle._read_evolution_campaign()
    stored["active_transaction"]["commit_sha"] = "a" * 40
    assert evolution_lifecycle._write_evolution_campaign(stored) is True
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "boot-gen")
    monkeypatch.setattr(
        agent_startup_checks.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, "head")

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert current["active_transaction"]["restart_authority_error"] == "commit_receipt_missing"
    assert int(current.get("absorbed_cycles_done") or 0) == 0
    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "evolution_tx_reconcile_blocked"
    assert event["reason"] == "commit_receipt_missing"


def test_boot_rename_loser_waits_for_claim_winner(tmp_path):
    from ouroboros import agent_startup_checks
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    sha = "b" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    claimed = tmp_path / "state" / f"pending_restart_verify.claimed.{os.getpid()}.json"
    claimed.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert int(current.get("absorbed_cycles_done") or 0) == 0


def test_boot_reclaims_dead_restart_claim(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, platform_layer, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "server-a"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    sha = "7" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    claimed = tmp_path / "state" / "pending_restart_verify.claimed.999999999.json"
    claimed.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: False)
    generation["value"] = "server-b"
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)

    current = evolution_lifecycle._read_evolution_campaign()
    assert "active_transaction" not in current
    assert current["transaction_history"][-1]["cycle_outcome"] == "absorbed"
    assert list((tmp_path / "state").glob("pending_restart_verify.claimed.*.json")) == []


def test_new_campaign_is_stamped_for_same_generation_worker_respawns(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle, queue, state

    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "same-server")
    state.init(tmp_path)
    queue.init(tmp_path)
    queue.init_queue_refs([], {}, {"value": 0})
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    assert campaign["last_boot_reconcile_gen"] == "same-server"
    state.update_state(lambda live: live.update(evolution_mode_enabled=True))
    tx = evolution_lifecycle.begin_evolution_transaction("respawn", cycle=1, campaign=campaign)
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": "6" * 40,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, claim["commit_sha"])

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert int(current.get("absorbed_cycles_done") or 0) == 0


def test_boot_reconcile_cannot_resurrect_owner_stopped_campaign(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "before-restart"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    sha = "4" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    generation["value"] = "after-restart"
    reached = threading.Event()
    release = threading.Event()

    def delayed_merge_base(*args, **kwargs):
        reached.set()
        assert release.wait(timeout=2)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(agent_startup_checks.subprocess, "run", delayed_merge_base)
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )
    worker = threading.Thread(target=agent_startup_checks.verify_restart, args=(env, "5" * 40))
    worker.start()
    assert reached.wait(timeout=2)
    evolution_lifecycle.complete_evolution_campaign("owner stop", cleanup_worktree=False)
    release.set()
    worker.join(timeout=2)

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["status"] == "stopped"
    assert current["completion_reason"] == "owner stop"
    assert "active_transaction" not in current


def test_owner_stop_preserves_prior_boot_reconciliation_evidence(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "before-restart"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    sha = "3" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    generation["value"] = "after-restart"
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )
    agent_startup_checks.verify_restart(env, sha)
    evolution_lifecycle.complete_evolution_campaign("owner stop", cleanup_worktree=False)

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["status"] == "stopped"
    assert current["absorbed_cycles_done"] == 1
    assert current["last_boot_reconcile_gen"] == "after-restart"
    assert current["transaction_history"][-1]["cycle_outcome"] == "absorbed"


def test_boot_restart_writers_obey_live_root_fuse(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    sha = "2" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    before = campaign_path.read_bytes()
    monkeypatch.setenv("OUROBOROS_PYTEST_ACTIVE", "1")
    monkeypatch.setenv("OUROBOROS_TEST_LIVE_DATA_ROOT", str(tmp_path))
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        agent_startup_checks.verify_restart(env, sha)

    assert campaign_path.read_bytes() == before
    assert list((tmp_path / "state").glob("pending_restart_verify.claimed.*.json")) == []


def test_boot_restart_write_failure_restores_claim_for_retry(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "server-a"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    sha = "c" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    pending = tmp_path / "state" / "pending_restart_verify.json"
    pending.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    real_write = agent_startup_checks.atomic_write_json
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("temporary write failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(agent_startup_checks, "atomic_write_json", fail_once)
    generation["value"] = "server-b"
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)
    assert pending.is_file()
    assert list((tmp_path / "state").glob("pending_restart_verify.claimed.*.json")) == []

    agent_startup_checks.verify_restart(env, sha)
    current = evolution_lifecycle._read_evolution_campaign()
    assert "active_transaction" not in current
    assert current["transaction_history"][-1]["cycle_outcome"] == "absorbed"


def test_boot_exact_claim_never_passes_without_active_transaction(tmp_path):
    from ouroboros import agent_startup_checks

    (tmp_path / "state").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)
    sha = "d" * 40
    claim = {
        "campaign_id": "campaign",
        "transaction_id": "transaction",
        "task_id": "task",
        "commit_sha": sha,
    }
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    campaign_path.write_text("{bad")
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)

    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "restart_verify"
    assert event["ok"] is False
    assert event["error"] == "transaction_missing"
    assert marker.exists() is False
    assert campaign_path.read_text() == "{bad"


def test_evolution_restart_write_failure_does_not_become_generic_restart(tmp_path, monkeypatch):
    from ouroboros.tools import control_runtime as control

    monkeypatch.setattr(control, "_evolution_restart_block_reason", lambda ctx: "")
    monkeypatch.setattr(
        control, "run_cmd",
        lambda cmd, cwd=None: "f" * 40 if cmd[-1] == "HEAD" else "ouroboros",
    )
    # The marker write lives in the shared writer helper (W4-F3: one schema for
    # the tool and the supervisor), so the disk failure is injected at its seam.
    from supervisor import evolution_lifecycle

    monkeypatch.setattr(
        evolution_lifecycle, "atomic_write_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    ctx = SimpleNamespace(
        current_task_type="evolution",
        repo_dir=tmp_path,
        drive_root=tmp_path,
        drive_path=lambda name: tmp_path / name,
        task_id="evo",
        task_metadata={"evolution_transaction": {}},
        pending_restart_reason=None,
        last_push_succeeded=True,
        last_reviewed_commit_sha="f" * 40,
    )

    result = control._request_restart(ctx, "apply reviewed evolution")

    assert "RESTART_BLOCKED" in result
    assert ctx.pending_restart_reason is None


def test_supervisor_rechecks_evolution_claim_immediately_before_restart(tmp_path):
    import server
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    sha = "1" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({
        "reason": "evolution restart",
        "expected_sha": sha,
        "evolution_claim": claim,
    }))
    live = state.load_state()
    live.update({"evolution_mode_enabled": False, "evolution_owner_stopped": True})
    state.save_state(live)
    restarted = []
    messages = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        load_state=state.load_state,
        safe_restart=lambda **k: restarted.append(k) or (True, "ok"),
        send_with_budget=lambda *a: messages.append(a),
    )

    server._perform_supervisor_restart(
        ctx, restart_reason="evolution restart", evolution_restart=True,
    )

    assert restarted == []
    assert "owner_stopped" in messages[0][1]


def test_supervisor_blocks_evolution_restart_if_marker_disappears_during_drain(tmp_path):
    import server

    restarted = []
    messages = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        load_state=lambda: {"owner_chat_id": 1},
        safe_restart=lambda **k: restarted.append(k) or (True, "ok"),
        send_with_budget=lambda *a: messages.append(a),
    )

    server._perform_supervisor_restart(
        ctx,
        restart_reason="agent_requested_restart",
        evolution_restart=True,
    )

    assert restarted == []
    assert "receipt is missing" in messages[0][1]


def test_generic_restart_ignores_stale_evolution_marker(tmp_path, monkeypatch):
    import server

    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({
        "reason": "agent_requested_restart",
        "evolution_claim": {"campaign_id": "stale"},
    }))
    restarted = []
    exited = []
    monkeypatch.setattr(server, "_request_restart_exit", lambda: exited.append(True))
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        load_state=lambda: {},
        safe_restart=lambda **k: restarted.append(k) or (True, "ok"),
        kill_workers=lambda **k: None,
        save_state=lambda state: None,
        persist_queue_snapshot=lambda **k: None,
    )

    server._perform_supervisor_restart(
        ctx, restart_reason="agent_requested_restart", evolution_restart=False,
    )

    assert restarted
    assert exited == [True]


def test_supervisor_blocks_restart_when_head_moved_after_receipt(tmp_path):
    import server
    from supervisor import evolution_lifecycle

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "file.txt").write_text("current\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "current"], cwd=repo, check=True, capture_output=True)
    campaign, tx = _active_transaction(tmp_path)
    reviewed_sha = "2" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": reviewed_sha,
    }
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], reviewed_sha,
    )["ok"] is True
    (tmp_path / "state" / "pending_restart_verify.json").write_text(json.dumps({
        "reason": "evolution restart",
        "expected_sha": reviewed_sha,
        "evolution_claim": claim,
    }))
    restarted = []
    messages = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        REPO_DIR=repo,
        load_state=lambda: {"owner_chat_id": 1},
        safe_restart=lambda **k: restarted.append(k) or (True, "ok"),
        send_with_budget=lambda *a: messages.append(a),
    )

    server._perform_supervisor_restart(
        ctx, restart_reason="evolution restart", evolution_restart=True,
    )

    assert restarted == []
    assert "no longer matches" in messages[0][1]


def _boot_env(tmp_path):
    return SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )


def _crash_after_reviewed_commit(tmp_path, monkeypatch, *, head_sha):
    """Drive the reviewed evolution commit and die before its SHA receipt.

    The reviewed commit lands on HEAD and the process never reaches
    ``record_evolution_commit`` — the exact W4-F1 crash window.
    """
    import pathlib

    from ouroboros.tools import git as git_tools
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    binding = {"tree_sha": "7" * 40, "parents": ["9" * 40]}
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    _patch_commit_seam(monkeypatch, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    _patch_commit_seam(monkeypatch, "_verify_reviewed_commit_binding", lambda *a, **k: (True, ""))
    _patch_commit_seam(monkeypatch, "_run_reviewed_stage_cycle",
        lambda *a, **k: {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post", "binding": dict(binding)},
        },
    )
    committed = {}

    def _run_cmd(cmd, cwd=None):
        if cmd[:2] == ["git", "commit"]:
            stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
            committed["intent_at_commit_time"] = dict(stored.get("commit_intent") or {})
            return ""
        return head_sha if cmd[:3] == ["git", "rev-parse", "HEAD"] else ""

    _patch_commit_seam(monkeypatch, "run_cmd", _run_cmd)

    def _crash(**kwargs):
        raise RuntimeError("process died before the SHA receipt")

    monkeypatch.setattr(evolution_lifecycle, "record_evolution_commit", _crash)
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        current_task_type="evolution",
        task_id=tx["task_id"],
        task_metadata={"evolution_transaction": tx},
    )

    with pytest.raises(RuntimeError):
        git_tools._repo_commit_push(ctx, "reviewed commit")

    return campaign, tx, binding, committed


def test_boot_attributes_the_commit_a_crash_left_without_a_receipt(tmp_path, monkeypatch):
    """W4-F1: the commit-vs-receipt crash window is recovered from the intent."""
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle, git_ops

    head = "d" * 40
    campaign, tx, binding, committed = _crash_after_reviewed_commit(
        tmp_path, monkeypatch, head_sha=head,
    )
    # Phase one is durable BEFORE `git commit` runs, not after it.
    assert committed["intent_at_commit_time"]["tree_sha"] == binding["tree_sha"]
    assert committed["intent_at_commit_time"]["parents"] == binding["parents"]
    crashed = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert str(crashed.get("commit_sha") or "") == ""

    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "boot-gen")
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd, **k: (
        (0, binding["tree_sha"], "") if cmd[1] == "rev-parse"
        else (0, f"{head} {binding['parents'][0]}", "")
    ))

    agent_startup_checks.verify_restart(_boot_env(tmp_path), head)

    current = evolution_lifecycle._read_evolution_campaign()
    assert "active_transaction" not in current
    resolved = current["transaction_history"][-1]
    assert resolved["commit_sha"] == head
    assert resolved["cycle_outcome"] == "absorbed"
    assert resolved["commit_receipt"]["reason"] == "recovered_from_commit_intent"
    assert int(current.get("absorbed_cycles_done") or 0) == 1


def test_boot_refuses_to_attribute_a_head_that_is_not_the_reviewed_material(
    tmp_path, monkeypatch,
):
    """Recovery is structural: a HEAD whose tree/parents differ is never adopted."""
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle, git_ops

    head = "d" * 40
    campaign, tx, binding, _committed = _crash_after_reviewed_commit(
        tmp_path, monkeypatch, head_sha=head,
    )
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "boot-gen")
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd, **k: (
        (0, "e" * 40, "") if cmd[1] == "rev-parse"
        else (0, f"{head} {binding['parents'][0]}", "")
    ))

    agent_startup_checks.verify_restart(_boot_env(tmp_path), head)

    current = evolution_lifecycle._read_evolution_campaign()
    assert str(current["active_transaction"].get("commit_sha") or "") == ""
    assert int(current.get("absorbed_cycles_done") or 0) == 0


def test_boot_backfills_the_cycle_outcome_row_a_crash_lost(tmp_path, monkeypatch):
    """W4-F2: the absorb write and the ledger append are not one transaction."""
    from ouroboros import agent_startup_checks, evolution_checkpoints, process_custody
    from ouroboros.evolution_checkpoints import build_solve_capability_digest
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    sha = "c" * 40
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    generation = {"value": "gen-1"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )

    def _crash(*a, **k):
        raise RuntimeError("process died between the campaign write and the ledger")

    monkeypatch.setattr(
        evolution_checkpoints, "append_cycle_outcome_checkpoint", _crash,
    )
    agent_startup_checks.verify_restart(_boot_env(tmp_path), sha)

    ledger = tmp_path / "state" / "evolution_checkpoints.jsonl"
    absorbed = evolution_lifecycle._read_evolution_campaign()["transaction_history"][-1]
    assert absorbed["cycle_outcome"] == "absorbed"
    assert not ledger.exists()  # the campaign says absorbed; the ledger says nothing

    monkeypatch.undo()
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: "gen-2",
    )
    agent_startup_checks.verify_restart(_boot_env(tmp_path), sha)

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    tagged = [row for row in rows if row.get("kind") == "cycle_outcome"]
    assert [(row["task_id"], row["cycle_outcome"], row["source"]) for row in tagged] == [
        (tx["task_id"], "absorbed", "boot_backfill"),
    ]
    assert "absorbed=1" in build_solve_capability_digest(tmp_path)

    agent_startup_checks.verify_restart(_boot_env(tmp_path), sha)
    assert len(ledger.read_text().splitlines()) == len(rows)  # idempotent


def _cycle_outcome_rows(ledger):
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    return [
        (row["task_id"], row["cycle_outcome"], row["source"])
        for row in rows if row.get("kind") == "cycle_outcome"
    ]


def test_boot_backfill_waits_for_the_reconcile_tag_instead_of_duplicating_it(
    tmp_path, monkeypatch,
):
    """S22: two booters, one ledger row. Actor 1 (this thread) reconciles the
    dangling commit; actor 2 (a thread) boots exactly while actor 1 sits between
    its campaign write and its ledger tag. Actor 2's backfill must block on
    ``locks/state.lock`` — an unlocked scan in that gap sees ``absorbed`` with no
    row and appends the duplicate ``boot_backfill`` row CI caught."""
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    sha = "e" * 40
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "gen-2")
    ledger = tmp_path / "state" / "evolution_checkpoints.jsonl"
    real_tag = agent_startup_checks.append_cycle_outcome_tag
    second_booter = threading.Thread(
        target=agent_startup_checks.verify_restart, args=(_boot_env(tmp_path), sha),
    )
    seen_in_gap = {}

    def _tag_with_a_second_booter_in_the_gap(*args, **kwargs):
        resolved = evolution_lifecycle._read_evolution_campaign()["transaction_history"][-1]
        assert resolved["cycle_outcome"] == "absorbed"  # campaign resolved, tag pending
        second_booter.start()
        second_booter.join(timeout=0.5)
        seen_in_gap["second_booter_blocked"] = second_booter.is_alive()
        seen_in_gap["rows_written_in_gap"] = _cycle_outcome_rows(ledger)
        real_tag(*args, **kwargs)

    monkeypatch.setattr(
        agent_startup_checks, "append_cycle_outcome_tag", _tag_with_a_second_booter_in_the_gap,
    )
    agent_startup_checks.verify_restart(_boot_env(tmp_path), sha)
    second_booter.join(timeout=10)
    assert not second_booter.is_alive()

    assert seen_in_gap == {"second_booter_blocked": True, "rows_written_in_gap": []}
    assert _cycle_outcome_rows(ledger) == [(tx["task_id"], "absorbed", "boot_reconcile")]
    current = evolution_lifecycle._read_evolution_campaign()
    assert [
        row["cycle_outcome"] for row in current["transaction_history"]
        if row.get("transaction_id") == tx["transaction_id"]
    ] == ["absorbed"]
    assert current["absorbed_cycles_done"] == 1


def test_boot_skips_the_backfill_when_the_state_lock_is_unavailable(
    tmp_path, monkeypatch, caplog,
):
    """No lock, no row: an unlocked backfill IS the duplicate writer, so the
    repair waits for a later boot instead — and the boot itself does not fail."""
    from ouroboros import agent_startup_checks, evolution_checkpoints, process_custody
    from supervisor import evolution_lifecycle
    from supervisor import state as supervisor_state

    campaign, tx = _active_transaction(tmp_path)
    sha = "f" * 40
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "gen-1")

    def _crash(*a, **k):
        raise RuntimeError("process died between the campaign write and the ledger")

    monkeypatch.setattr(evolution_checkpoints, "append_cycle_outcome_checkpoint", _crash)
    agent_startup_checks.verify_restart(_boot_env(tmp_path), sha)
    ledger = tmp_path / "state" / "evolution_checkpoints.jsonl"
    assert evolution_lifecycle._read_evolution_campaign()["transaction_history"][-1][
        "cycle_outcome"
    ] == "absorbed"
    assert not ledger.exists()

    monkeypatch.undo()
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "gen-2")
    lock_path = tmp_path / "locks" / "state.lock"
    real_acquire = supervisor_state.acquire_file_lock
    held_elsewhere = real_acquire(lock_path)  # another actor holds the state lock
    assert held_elsewhere is not None
    monkeypatch.setattr(
        supervisor_state, "acquire_file_lock",
        lambda path, **kwargs: real_acquire(path, timeout_sec=0.2),
    )
    try:
        with caplog.at_level(logging.WARNING, logger="ouroboros.agent_startup_checks"):
            agent_startup_checks.verify_restart(_boot_env(tmp_path), sha)
    finally:
        supervisor_state.release_file_lock(lock_path, held_elsewhere)
    assert not ledger.exists()
    assert any(
        "Skipped cycle-outcome backfill" in record.getMessage() for record in caplog.records
    )

    agent_startup_checks.verify_restart(_boot_env(tmp_path), sha)  # lock free: repair replays
    assert _cycle_outcome_rows(ledger) == [(tx["task_id"], "absorbed", "boot_backfill")]


def test_containment_disowns_the_commit_intent_so_boot_cannot_adopt_it(
    tmp_path, monkeypatch,
):
    """A commit the authority path refused must stay unattributable at boot."""
    from ouroboros import agent_startup_checks, process_custody
    from ouroboros.tools import git_evolution
    from supervisor import evolution_lifecycle, git_ops

    head = "d" * 40
    campaign, tx, binding, _committed = _crash_after_reviewed_commit(
        tmp_path, monkeypatch, head_sha=head,
    )
    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["commit_intent"]["tree_sha"] == binding["tree_sha"]

    ctx = SimpleNamespace(repo_dir=tmp_path, task_id=tx["task_id"])
    assert "CONTAINMENT_FAILED" in git_evolution._preserve_evolution_orphan(ctx, head)
    assert evolution_lifecycle._read_evolution_campaign()[
        "active_transaction"
    ]["commit_intent"] == {}

    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "boot-gen")
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd, **k: (
        (0, binding["tree_sha"], "") if cmd[1] == "rev-parse"
        else (0, f"{head} {binding['parents'][0]}", "")
    ))

    agent_startup_checks.verify_restart(_boot_env(tmp_path), head)

    current = evolution_lifecycle._read_evolution_campaign()
    assert str(current["active_transaction"].get("commit_sha") or "") == ""
    assert int(current.get("absorbed_cycles_done") or 0) == 0
