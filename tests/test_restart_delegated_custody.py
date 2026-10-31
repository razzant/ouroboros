"""Portable S12 composition: saved-work recovery and delegated startup custody.

The Linux scenario owns SIGKILL and /proc proof. Here the real checkpoint,
queue restore, SQLite store and boot sweep meet the real gateway client over
the loopback fake. A crash preserves the leaf; a deliberate Stop cancels it.
"""
from __future__ import annotations

import hashlib

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._usage_store_testing import ledger_rows
from tests.system_e2e.harness import ArtifactOracle
from tests.system_e2e.interfaces import FAKE_HANG_MARKER, FakeClaudexorDaemon
from tests.test_restart_saved_work import _working_run

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("owner_stop", [False, True], ids=["natural-completion", "owner-stop"])
def test_saved_parent_boot_retains_then_settles_original_run(tmp_path, monkeypatch, owner_stop):
    from ouroboros import claudexor_daemon, post_task_checkpoint, process_custody, server_maintenance
    from ouroboros import delegate_custody as custody
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at
    from supervisor import active_activity

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_maintenance, "_installed_skill_names", lambda: set())
    monkeypatch.setattr(process_custody, "reap_orphaned_processes", lambda *a, **kw: [])
    monkeypatch.setattr(post_task_checkpoint, "POST_TASK_SYNTHESIS_INFLIGHT", {})
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", active_activity.DirectActivityRegistry())
    monkeypatch.setattr(custody, "_CUSTODY", {})
    task_id = "saved-nanny"
    _working_run(tmp_path, workers, task_id)
    assert queue.persist_queue_snapshot(reason="before_crash")
    oracle = ArtifactOracle(tmp_path)

    with FakeClaudexorDaemon() as daemon:
        daemon.install(tmp_path / "claudexor")
        endpoint = discover_daemon_at(tmp_path / "claudexor")
        monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway",
                            lambda **kw: ClaudexorGateway(endpoint))
        with ClaudexorGateway(endpoint) as gateway:
            gateway.handshake()
            project_id = gateway.register_project(str(tmp_path))
            handle = gateway.start_run({
                "prompt": FAKE_HANG_MARKER, "mode": "ask", "access": "readonly",
                "authPreference": "subscription", "maxSeconds": 60,
                "scope": {"kind": "project", "root": str(tmp_path)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
            }, idempotency_key="original-invocation")
        run_id = handle["runId"]
        assert custody.record_started(tmp_path, custody.RunCustody(
            run_id=run_id, task_id=task_id, route_id=daemon.harness_id, model="mock-model",
            project_id=project_id, project_owned=True,
            root_task_id=task_id, ledger_root=str(tmp_path), access="readonly", mode="ask",
            invocation_id="original-invocation", idempotency_key="original-invocation"))

        # No graceful kill/terminal write: only the saved snapshot and store survive.
        workers.RUNNING.clear()
        custody._CUSTODY.clear()
        terminalized = []
        assert queue.restore_pending_from_snapshot(terminalized=terminalized) == 1
        assert not terminalized
        [held] = workers.PENDING
        assert held["id"] == task_id and held["_working_recovery"]["from_attempt"] == 1
        assert oracle.task_result(task_id)["reason_code"] == "saved_work_hold"
        assert server_maintenance._live_task_ids() == set()

        for _ in range(2):
            server_maintenance._startup_custody_sweep()
        reconciled = oracle.events(custody.RECONCILED)
        assert [row["action"] for row in reconciled] == ["left_live", "left_live"]
        assert all(row["run_id"] == run_id and row["state"] == "running" for row in reconciled)
        assert {run.run_id for run in custody.open_runs(tmp_path)} == {run_id}
        assert not oracle.events(custody.CANCEL_OUTCOME) and not oracle.events(custody.SETTLED)
        assert not daemon.calls("POST", f"/v2/runs/{run_id}/control")

        if owner_stop:
            from ouroboros.cancel_intents import request_cancel

            request_cancel(tmp_path, task_id, reason="owner stop", source="owner", requested_by="owner")
            assert queue.cancel_task_custody(task_id, deliver=False) == "cancelled"
            assert oracle.task_result(task_id)["status"] == "cancelled"
        else:
            with daemon._lock:
                daemon.runs[run_id]["hang"] = False

        # The boot/periodic reconciler now observes either the deliberate owner
        # terminal or the engine's natural completion; neither starts another run.
        custody._CUSTODY.clear()
        server_maintenance._startup_custody_sweep()
        server_maintenance._startup_custody_sweep()
        expected = "cancelled" if owner_stop else "succeeded"
        settled = oracle.events(custody.SETTLED)
        assert len(settled) == 1 and settled[0]["run_id"] == run_id and settled[0]["state"] == expected
        assert len(oracle.events(custody.LEDGER_RECORDED)) == 1
        [session] = [row for row in ledger_rows(tmp_path) if row.get("kind") == "subscription_session"]
        assert session["task_id"] == task_id
        assert session["session_id_sha256"] == hashlib.sha256(run_id.encode()).hexdigest()
        assert session["cost_final"] is True and session["cost_usd"] == 0.0
        stored = custody.replay(tmp_path)[run_id]
        assert stored.settled and stored.ledger_recorded and stored.terminal_state == expected
        assert not custody.open_runs(tmp_path)
        assert len(daemon.run_start_posts()) == 1
        controls = daemon.calls("POST", f"/v2/runs/{run_id}/control")
        cancels = oracle.events(custody.CANCEL_OUTCOME)
        if owner_stop:
            assert len(controls) == len(cancels) == 1 and cancels[0]["outcome"] == "confirmed"
            # Stop retains its verified read-back; recovery never repeats the
            # cancellation or money. Output remains with the body collector.
            after_stop = oracle.events(custody.RECONCILED)[len(reconciled):]
            assert [row["action"] for row in after_stop] == ["cancelled"]
            assert after_stop[0]["outcome"] == "confirmed"
            from ouroboros.delegate_collection import collect_terminal
            assert stored.terminal_source and not stored.output_source
            assert collect_terminal(tmp_path, run_id)["status"] == "collected"
            stored = custody.replay(tmp_path)[run_id]
            assert stored.terminal_source and stored.output_source and stored.output_complete
            assert stored.output_terminal_sha == stored.terminal_source["sha256"]
            assert not stored.output_consumed
        else:
            assert not controls and not cancels
            assert oracle.events(custody.RECONCILED)[-1]["settled"] is True
            assert oracle.task_result(task_id)["reason_code"] == "saved_work_hold"
