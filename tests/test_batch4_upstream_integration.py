"""Published Batch2 + Batch4: stop causes through the kill and recovery consumers."""
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_batch4_repair_compositions import _running
from tests.test_processing_transport import transport  # noqa: F401 - fake physical transport
from tests.test_restart_retention import _done_ids, _pool_events
from tests.test_review_operation_lifetime import env  # noqa: F401 - isolated operation fixture
from tests.test_send_clock import ticking  # noqa: F401 - deterministic clock samples
from tests._usage_store_testing import ledger_rows

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("door,expected", [
    ("crash", "saved_sleep_recovery"), ("restart", "owner_restart_hold"), ("panic", "panic_hold"),
])
@pytest.mark.parametrize("conversion_fails", [False, True])
def test_kill_saved_warm_sleep_keeps_cause_before_and_after_restore(tmp_path, monkeypatch, door, expected,
                                                                  conversion_fails):
    from ouroboros import budget_pause, model_sleep, owner_wait
    from ouroboros.task_results import load_task_result
    from supervisor import restart_retention
    from supervisor.events_budget import budget_hold_fact

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    events = _pool_events(workers, monkeypatch)
    _running(tmp_path, workers)
    ctx, limit = _loop_ctx(tmp_path, "root")
    model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=1), "warm")
    checkpoint = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, {}, 1, [], set())
    owner_wait.set_owner_wait(tmp_path, "root", {**checkpoint, "state": "waiting"})
    # A fake worker gives the real kill consumer a signal boundary to cross.
    signalled = []
    workers.WORKERS[0] = SimpleNamespace(wid=0, proc=SimpleNamespace(
        pid=None, is_alive=lambda: not signalled, terminate=lambda: signalled.append(True),
        join=lambda **_kw: None))
    read_checkpoint = owner_wait.saved_sleep_checkpoint
    def after_signal(*args, **kwargs):
        assert signalled, "Panic must signal before retention reads or writes"
        return read_checkpoint(*args, **kwargs)
    monkeypatch.setattr(owner_wait, "saved_sleep_checkpoint", after_signal)
    if door == "panic":
        (tmp_path / "state" / "panic_stop.flag").write_text("panic")
    # Restart's explicit door must work even when its flag was not written.
    with monkeypatch.context() as failing:
        if conversion_fails:
            def cannot_publish(*_a, **_kw):
                raise OSError("conversion write failed at kill")
            failing.setattr(budget_pause, "set_budget_pause", cannot_publish)
        assert workers.kill_workers(force=True, archive_service_logs=False,
                                    reconcile_delegate_custody=False, reconcile_review_custody=False,
                                    hold_never_started=door == "restart",
                                    stop_source="owner_restart" if door == "restart" else "")
    assert not _done_ids(events) and not workers.RUNNING
    parked = workers.PENDING[0]
    assert budget_hold_fact(parked)["reason"] == expected
    assert bool(parked.get("_budget_pause")) is not conversion_fails
    assert restart_retention.held_ids(workers.PENDING) == (["root"] if door == "restart" else [])
    workers.PENDING.clear()
    assert q.restore_pending_from_snapshot() == 1
    parked = workers.PENDING[0]
    assert budget_hold_fact(parked)["reason"] == expected
    assert load_task_result(tmp_path, "root")["status"] != "cancelled"
    assert not parked.get("_budget_pause_resume"), "snapshot null is not a Resume grant"
    if door == "panic":
        (tmp_path / "state" / "panic_stop.flag").unlink()
    assert q.resume_budget_paused_task("root")["ok"]
    ctx.budget_pause_resume = parked["_budget_pause_resume"]
    assert budget_pause.load_budget_pause(ctx)["messages"] == limit.messages


@pytest.mark.parametrize("consumer", ["kill", "restore"])
@pytest.mark.parametrize("door,expected", [("restart", "owner_restart_hold"), ("panic", "panic_hold")])
def test_markerless_saved_sleep_recovery_is_upgraded_by_actual_stop(tmp_path, monkeypatch, consumer, door, expected):
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import budget_hold_fact, hold_budget_row
    from supervisor.restart_retention import held_ids

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    task = {"id": "root", "root_task_id": "root", "type": "task", "chat_id": 0, "_attempt": 1}
    hold_budget_row(task, reason="saved_sleep_recovery", detail="conversion previously failed", result_root=tmp_path)
    workers.PENDING.append(task)
    flag = tmp_path / "state" / ("owner_restart_no_resume.flag" if door == "restart" else "panic_stop.flag")
    flag.write_text("owner_restart_no_resume" if door == "restart" else "panic")
    if consumer == "kill":
        assert workers.kill_workers(archive_service_logs=False, reconcile_delegate_custody=False,
                                    reconcile_review_custody=False, hold_never_started=door == "restart")
    else:
        assert q.persist_queue_snapshot(reason="previous-failed-conversion")
        workers.PENDING.clear()
        assert q.restore_pending_from_snapshot() == 1
    assert budget_hold_fact(workers.PENDING[0])["reason"] == expected
    assert held_ids(workers.PENDING) == (["root"] if door == "restart" else [])
    assert "_budget_pause_resume" not in workers.PENDING[0]


def test_sent_review_settles_after_pause_and_author_close_with_original_group(env, monkeypatch):  # noqa: F811
    import dataclasses
    import json
    import threading

    from ouroboros import review_operation
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted, require_physical_dispatch_window
    from ouroboros.model_wait import model_waitable
    from ouroboros.review_dispatch import collect_task_acceptance_run
    from ouroboros.review_substrate import run_review_request
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_billing_group import _scope, _spend
    from tests.test_review_operation_lifetime import TASK, _parent, _request, _slot, until

    f = env
    _, _, workers = _install_queue(f.root, monkeypatch)
    _running(f.root, workers, TASK)
    _spend(f.root, _scope(f.root, "original", "original", group="original", group_limit=2.0), 0.8)
    sent, release = threading.Event(), threading.Event()
    observations = []

    class SentModel:
        @model_waitable
        def chat(self, messages, model, model_poll_control=None, **kwargs):
            require_physical_dispatch_window()
            reservation = ua.reserve_attempt(ua.AttemptRequest(model=model, provider="test", reservation_usd=1.2))
            ua.mark_dispatched(reservation)
            sent.set()
            assert release.wait(10)
            # The callback was bound in the operation, not in its now-closed author.
            observations.append(model_poll_control())
            # A ledger "dispatched" row is not a physical handoff: this fake sender
            # never handed bytes to an executor, so it earned no Pause exception.
            with pytest.raises(PhysicalDispatchInterrupted, match="owner_pause"):
                require_physical_dispatch_window()
            ua.settle_attempt(reservation, {"prompt_tokens": 3, "completion_tokens": 2},
                              cost_usd=1.2, cost_final=True)
            return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "settled"})}, {}

    try:
        with ua.usage_scope(_scope(f.root, TASK, TASK, group="original", group_limit=2.0,
                                  root_limit=2.0, global_limit_usd=100)), _parent(f) as parent:
            result = run_review_request(_request(), slots=[_slot()], drive_root=f.root,
                                        usage_ctx=f.ctx, llm=SentModel())
            assert sent.wait(5)
            operation = next(op for op in review_operation._LIVE.values() if op.task_id == TASK)
            assert request_owner_pause(TASK, request_id="pause-after-send")["ok"]
            # The author's own context carries no review episode: it stays fenced.
            with pytest.raises(PhysicalDispatchInterrupted, match="owner_pause"):
                require_physical_dispatch_window()
        write_task_result(f.root, TASK, "completed", result="author finished")
        assert parent.closed
    finally:
        release.set()
    until(lambda: operation.closed)
    from ouroboros.task_results import load_task_result

    until(lambda: load_task_result(f.root, TASK)["review_operations"][operation.owner_id]["state"] == "unpublished")
    collected = collect_task_acceptance_run(json.loads(json.dumps(dataclasses.asdict(result))),
                                            drive_root=f.root, usage_ctx=f.ctx)
    assert collected.actors[0]["parsed"]["verdict"] == "PASS"
    assert observations == [None]
    assert ua.usage_projection(f.root, billing_group_id="original")["accounted_usd"] == pytest.approx(2.0)
    rows = ledger_rows(f.root)
    review_rows = [r for r in rows if r.get("task_id") == TASK]
    assert {r["root_task_id"] for r in review_rows} == {TASK}
    assert {r["billing_group_id"] for r in review_rows} == {"original"}
    # One attempt (one current row), settled: collection buys nothing.
    assert [r["state"] for r in review_rows] == ["settled"], "collection buys nothing"
    with pytest.raises(ua.BudgetExceeded):
        _spend(f.root, _scope(f.root, "original", "original", group="original", group_limit=2.0), 0.1)


def test_new_physical_clocks_preserve_group_and_sealed_bytes(transport, ticking):  # noqa: F811
    from ouroboros import send_clock as sc
    from ouroboros import usage_accounting as ua
    from ouroboros.task_results import write_task_result
    from tests.test_billing_group import _scope, _spend
    from tests.test_send_clock import MOSCOW, _digest, _ledger, _sealed

    root, client, sent = transport
    write_task_result(root, "successor", "running", root_task_id="successor")
    scope = _scope(root, "successor", "successor", group="original", group_limit=2.0,
                   root_limit=5.0, global_limit_usd=100)
    with ua.usage_scope(scope), sc.MainSendClock(MOSCOW).bound() as clock:
        for _ in range(2):
            client.chat(messages=[{"role": "user", "content": "same logical text"}],
                        model="openai::test-model", model_role="main", max_tokens=123)
    assert len(sent) == 2 and len(set(clock.notes)) == 2
    settled = [r for r in _ledger(root) if r["state"] == "settled"]
    for row, candidate in zip(settled, sent):
        assert row["billing_group_id"] == "original" and row["root_task_id"] == "successor"
        assert row["candidate_raw_sha256"] == _digest(candidate) == _sealed(row)
    _spend(root, _scope(root, "sibling", "sibling", group="original", group_limit=2.0), 2.0)
    with ua.usage_scope(scope), sc.MainSendClock(MOSCOW).bound(), pytest.raises(ua.BudgetExceeded):
        client.chat(messages=[{"role": "user", "content": "no group remainder"}],
                    model="openai::test-model", model_role="main", max_tokens=123)
    assert len(sent) == 2, "repreparation cannot give a successor a fresh group budget"
