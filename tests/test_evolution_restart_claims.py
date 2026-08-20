"""The restart claim: who may take it, who must wait, and what boot reconciliation may revive.

Split out of ``tests/test_evolution_state_integrity_v3.py`` by theme: the exact active
receipt a restart requires, the v2 claim verified only after a new generation, the losers
that wait instead of bypassing, the dead claim reclaimed, the write failures that restore
the claim for retry, the owner-stopped campaign boot cannot resurrect, and the supervisor
rechecks that gate an evolution restart against a stale marker or a moved head.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from types import SimpleNamespace

import pytest

from tests._evolution_state_shared import _active_transaction


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
    monkeypatch.setattr(
        control, "atomic_write_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    ctx = SimpleNamespace(
        current_task_type="evolution",
        repo_dir=tmp_path,
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
    from ouroboros import server_restart

    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({
        "reason": "agent_requested_restart",
        "evolution_claim": {"campaign_id": "stale"},
    }))
    restarted = []
    exited = []
    monkeypatch.setattr(server_restart, "_request_restart_exit", lambda: exited.append(True))
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
