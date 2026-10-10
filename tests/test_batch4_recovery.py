"""Recovery regressions: actual admissions, launch races and durable custody."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import _interrupted, NONCE
from tests._usage_store_testing import ledger_rows


def test_publication_failure_cannot_dispatch_and_replay_repairs_same_action(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor import continuation_admission as ca
    from supervisor.queue_transitions import budget_pause_fact

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    from ouroboros import task_results as tr
    real_write = tr.write_task_result
    def fail(*args, **kwargs):
        if "continuation_admission" in kwargs:
            raise OSError("result disk unavailable")
        return real_write(*args, **kwargs)
    monkeypatch.setattr(tr, "write_task_result", fail)
    first = ca.admit_continuation("pred-1", action_nonce=NONCE)
    assert first["unconfirmed"] and not first["ok"]
    tid = first["successor_task_id"]
    assert budget_pause_fact(workers.PENDING[0]) is not None
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert tid in json.dumps(snapshot) and "_continuation_prepared" in json.dumps(snapshot)
    # Actual snapshot restore must preserve the preparation hold.
    workers.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert budget_pause_fact(workers.PENDING[0]) is not None
    monkeypatch.setattr(tr, "write_task_result", real_write)
    replay = ca.admit_continuation("pred-1", action_nonce=NONCE)
    assert replay["ok"] and replay["successor_task_id"] == tid
    assert not workers.PENDING[0].get("_continuation_prepared")
    assert load_task_result(tmp_path, tid)["continuation_admission"]["binding"]["action_nonce"] == NONCE


def test_no_initial_cap_continues_under_the_configured_cap_disclosed(tmp_path, monkeypatch):
    """A legacy predecessor that recorded no cap stays its own group under today's cap, named as such."""
    from ouroboros.task_results import task_result_path
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    path = task_result_path(tmp_path, "pred-1")
    row = json.loads(path.read_text())
    row.pop("billing_group")
    path.write_text(json.dumps(row))
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "999")
    assert admit_continuation("pred-1", action_nonce=NONCE)["ok"]
    binding = workers.PENDING[0]["metadata"]["continuation"]
    assert (binding["billing_group_id"], binding["billing_group_limit_usd"],
            binding["billing_group_limit_source"]) == ("pred-1", 999.0, "legacy_default")


@pytest.mark.parametrize("payload", ['[]\n', '{torn\n', '\udcff'])
def test_bad_mail_is_never_deleted_or_silently_dropped(tmp_path, monkeypatch, payload):
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.task_custody import settle_task_mailbox
    from supervisor.continuation_admission import admit_continuation

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, root_phase_checkpoint={}, child_ref_promotion={})
    path = _mailbox_path(tmp_path, "pred-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("utf-8", errors="surrogateescape"))
    assert not settle_task_mailbox(tmp_path, "pred-1", tmp_path)
    assert path.exists()
    assert not admit_continuation("pred-1", action_nonce=NONCE)["ok"]


def test_owner_mail_retention_failure_keeps_acknowledged_source(tmp_path, monkeypatch):
    from ouroboros import task_custody as tc
    from ouroboros.owner_mailbox import write_owner_message, acknowledge_transcript_entry, _mailbox_path

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, root_phase_checkpoint={}, child_ref_promotion={})
    write_owner_message(tmp_path, "exact correction", "pred-1", msg_id="owner-x")
    acknowledge_transcript_entry(tmp_path, "pred-1", {"msg_id": "owner-x"})
    monkeypatch.setattr(tc, "_write_custody_fields", lambda *a, **kw: None)  # lost publication
    assert not tc.settle_task_mailbox(tmp_path, "pred-1", tmp_path)
    assert "exact correction" in _mailbox_path(tmp_path, "pred-1").read_text()


def test_paused_or_terminal_child_with_unbound_start_holds_continue(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as dc
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    write_task_result(tmp_path, "old-child", "completed", root_task_id="pred-1")
    assert dc.record_start_requested(tmp_path, task_id="old-child", root_task_id="pred-1",
                                    invocation_id="unanswered-start", idempotency_key="key", route="test", request={"prompt": "work"})
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] and ack["held"]
    assert any(b.get("task_id") == "old-child" for b in ack["blockers"])
    assert len(workers.PENDING) == 1


@pytest.mark.parametrize("pause_before_claim", [True, False])
def test_physical_claim_linearizes_with_pause_without_network_lock(tmp_path, monkeypatch, pause_before_claim):
    from ouroboros import usage_accounting as ua
    from ouroboros.owner_pause import install_fence
    from ouroboros.task_results import write_task_result

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    write_task_result(tmp_path, "root", "running")
    at_boundary, release = threading.Event(), threading.Event()
    sent = []
    def prepare(_reservation):
        if pause_before_claim:
            at_boundary.set()
            assert release.wait(4)
    def send():
        sent.append(True)
        if not pause_before_claim:
            at_boundary.set()
            assert release.wait(4)
        return "response"
    def work():
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")):
            return ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="test", reservation_usd=0.01),
                       send, before_dispatch=prepare, extractor=lambda _: ({}, 0.01, True))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(work)
        assert at_boundary.wait(4)
        try:
            # If the transport held the admission lock this would time out.
            fence, _ = install_fence(tmp_path, "root", request_id="pause")
            assert fence["state"] == "requested"
        finally:
            release.set()
        if pause_before_claim:
            with pytest.raises(Exception):
                future.result(timeout=4)
            assert not sent
        else:
            assert future.result(timeout=4) == "response" and sent
    with ua._locked(tmp_path):
        rows = list({row['attempt_id']: row for row in ledger_rows(tmp_path)}.values())
    assert rows[-1]["state"] == ("released" if pause_before_claim else "settled")


def test_tool_body_is_owned_before_pause_and_until_its_return(tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.owner_pause import install_fence
    from ouroboros.task_results import write_task_result, load_task_result

    write_task_result(tmp_path, "root", "running")
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    entered, release = threading.Event(), threading.Event()
    def body(*a, **kw):
        entered.set()
        assert release.wait(4)
        return "done"
    registry.override_handler("knowledge_read", body)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(registry.execute_result, "knowledge_read", {"topic": "x"})
        assert entered.wait(4)
        try:
            install_fence(tmp_path, "root", request_id="pause")
            assert load_task_result(tmp_path, "root")["launch_handoffs"]
        finally:
            release.set()
        assert future.result(timeout=4).text == "done"
    assert not load_task_result(tmp_path, "root")["launch_handoffs"]
    assert registry.execute_result("knowledge_read", {"topic": "x"}).code == "OWNER_PAUSE_NOT_STARTED"


def test_sleep_string_cannot_bypass_model_selection_policy(tmp_path, monkeypatch):
    from supervisor.budget_resume import grant_exact_budget_resume
    from supervisor.events_budget import budget_resume_dispatch_allowed

    q, _, _ = _install_queue(tmp_path, monkeypatch)
    assert grant_exact_budget_resume({}, {}, selected_by="sleep_wake")["error"] == "sleep_policy_required"
    forged = {"id": "child", "root_task_id": "root", "_budget_pause_resume": {"selected_by": "sleep_wake"}}
    assert not budget_resume_dispatch_allowed(q, forged)


def test_warm_sleep_refuses_selected_root_blocked_by_its_project_lease(tmp_path):
    from tests.test_model_sleep import _ctx, _result
    from ouroboros.model_sleep import selectors, request_sleep

    ctx = _ctx(tmp_path)
    ctx.project_id = "project"
    _result(tmp_path, "other-root", "scheduled", project_id="project")
    chosen = selectors(ctx, tasks=["other-root"])
    with pytest.raises(ValueError, match="project lease"):
        request_sleep(ctx, chosen, "warm")


def test_cold_sleep_retains_unknown_child_start_after_result_collection(tmp_path):
    from tests.test_model_sleep import _ctx, _result
    from ouroboros import delegate_custody as dc
    from ouroboros.model_sleep import request_sleep

    _result(tmp_path, "sleeper")
    assert dc.record_start_requested(tmp_path, task_id="collected-child", root_task_id="sleeper",
                                    invocation_id="uncertain", idempotency_key="key", route="test",
                                    request={"prompt": "work"})
    with pytest.raises(ValueError, match="member_custody collected-child"):
        request_sleep(_ctx(tmp_path), {"tasks": [], "runs": [], "senders": [], "wake_at": ""}, "cold")


def test_assignment_observes_ready_sleep_without_holding_queue_lock(tmp_path, monkeypatch):
    from tests.test_model_sleep import _cold_park, _mail, _result
    from ouroboros import budget_pause
    from supervisor.worker_assignment import assign_tasks

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _result(tmp_path, "peer")
    _cold_park(tmp_path, monkeypatch, workers, senders=["peer"])
    _mail(tmp_path, "sleeper", "ready", sender="peer")
    observed, release = threading.Event(), threading.Event()
    real_observe = budget_pause.observe_task_runs
    def observe(*args, **kwargs):
        if kwargs.get("reason") == "sleep_wake_check":
            assert kwargs.get("request_stop") is False
            observed.set()
            assert release.wait(4)
        return real_observe(*args, **kwargs)
    monkeypatch.setattr(budget_pause, "observe_task_runs", observe)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(assign_tasks)
        assert observed.wait(4)
        acquired = queue._queue_lock.acquire(timeout=1)
        try:
            assert acquired, "the actual assignment consumer held queue authority over observation"
        finally:
            if acquired:
                queue._queue_lock.release()
            release.set()
        future.result(timeout=4)
    assert workers.PENDING[0]["_budget_pause_resume"]["authority"] == "sleep_readiness"


def test_zero_dispatch_hold_survives_malformed_snapshot_fences(tmp_path, monkeypatch):
    from supervisor import queue_snapshot as qs
    from supervisor.events_budget import budget_hold_fact
    from ouroboros.task_results import write_task_result, load_task_result

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "zero", "scheduled")
    row = {"id": "zero", "type": "task", "admitted_dispatch": "none",
           "_budget_pause_hold": {"reason": "owner_restart_hold", "selected": False}}
    assert qs._refuse_restore_invalid_fences([row]) == 1
    assert budget_hold_fact(workers.PENDING[0])
    assert load_task_result(tmp_path, "zero")["status"] == "scheduled"


def test_corrupt_pause_authority_refuses_repeated_real_tool_launches(tmp_path):
    from ouroboros.owner_pause import install_fence
    from ouroboros.task_results import write_task_result, task_result_path
    from ouroboros.tools.registry import ToolRegistry

    write_task_result(tmp_path, "root", "running")
    install_fence(tmp_path, "root", request_id="pause")
    task_result_path(tmp_path, "root").write_text('{broken')
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry.override_handler("knowledge_read", lambda *a, **kw: pytest.fail("body launched"))
    for _ in range(2):
        assert registry.execute_result("knowledge_read", {"topic": "x"}).code == "OWNER_PAUSE_NOT_STARTED"


def test_worker_claim_is_durable_before_handoff_and_restore_is_not_unrun(tmp_path, monkeypatch):
    from supervisor.worker_assignment import _claim_worker_launch
    from supervisor.events_budget import budget_hold_fact
    from ouroboros.task_results import write_task_result

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "scheduled")
    task = {"id": "root", "type": "task", "chat_id": 0, "root_task_id": "root", "admitted_dispatch": "none"}
    workers.PENDING.append(task)
    handed_off = []
    def put(candidate):
        snapshot = json.loads((tmp_path / "state" / "queue_snapshot.json").read_text())
        assert snapshot["pending"][0]["task"]["admitted_dispatch"] == "possible"
        handed_off.append(candidate)
    assert _claim_worker_launch(q, task, SimpleNamespace(in_q=SimpleNamespace(put=put)))
    assert handed_off == [task] and task["admitted_dispatch"] == "possible"
    workers.PENDING.clear()  # crash before transport outcome became observable
    assert q.restore_pending_from_snapshot() == 1
    assert budget_hold_fact(workers.PENDING[0])["reason"] == "dispatch_outcome_unknown"
    assert not q.resume_budget_paused_task("root")["ok"]
