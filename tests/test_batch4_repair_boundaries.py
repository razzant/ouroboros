"""Adversarial consumers at the Batch4 launch and sleep boundaries."""
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_batch4_repair_compositions import _running
from tests.test_restart_retention import _pool_events, _restart_door



def _next_generation(monkeypatch):
    """The old stop latches end with its process; the transaction remains for the new boot."""
    import threading
    from ouroboros import delegate_recovery, server_restart

    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    monkeypatch.setattr(server_restart, "_owner_restart_requested", threading.Event())
    monkeypatch.setattr(delegate_recovery, "_PROCESS_GENERATION", "next-boot")


@pytest.mark.parametrize("direct", [False, True])
def test_warm_restart_restores_after_failed_conversion(tmp_path, monkeypatch, direct):
    from ouroboros import owner_wait, model_sleep, budget_pause, server_restart
    from ouroboros.task_results import load_task_result
    from supervisor import active_activity
    from supervisor.events_budget import budget_hold_fact
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _running(tmp_path, workers)
    ctx, limit = _loop_ctx(tmp_path, "root", direct=direct)
    model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=3600), "warm")
    checkpoint = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, {}, 1, [], set())
    owner_wait.set_owner_wait(tmp_path, "root", {**checkpoint, "state": "waiting"})
    if direct:
        workers.RUNNING.clear()
        monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: SimpleNamespace(
            snapshot=lambda: [{"activity_id": "root"}], actors=lambda: []))
        monkeypatch.setattr(q, "PRIOR_DIRECT_ROOTS", {"task_ids": ["root"]})
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    assert "root" not in server_restart._owned_live_task_ids(SimpleNamespace(RUNNING=workers.RUNNING))
    with monkeypatch.context() as failing:
        failing.setattr(budget_pause, "set_budget_pause", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk")))
        _restart_door(tmp_path, monkeypatch, workers)
    if direct:
        monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: SimpleNamespace(snapshot=lambda: [], actors=lambda: []))
    _next_generation(monkeypatch)
    workers.PENDING.clear()
    assert q.restore_pending_from_snapshot() == 1
    task = workers.PENDING[0]
    assert task["id"] == "root" and budget_hold_fact(task)
    assert load_task_result(tmp_path, "root")["status"] != "cancelled"
    assert q.resume_budget_paused_task("root")["ok"]
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    saved = budget_pause.load_budget_pause(ctx)
    assert saved["messages"] == limit.messages and saved["sleep"] == checkpoint["sleep"]
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_kw: (None, "max"))
    messages = []
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, messages, {}, {}, set(),
                                    budget_remaining_usd=5)
    assert budget_pause.budget_pause_row(tmp_path, "root")["state"] == budget_pause.STATE_RESUMED
    assert load_task_result(tmp_path, "root")["owner_wait"]["state"] == "retained"
    assert messages[:len(limit.messages)] == limit.messages


@pytest.mark.parametrize("invalid", ["source", "attempt", "quiz", "review", "stop"])
def test_warm_retention_requires_current_sleep_proof(tmp_path, monkeypatch, invalid):
    from ouroboros import owner_wait, model_sleep
    from supervisor.restart_retention import pause_retention
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    ctx, limit = _loop_ctx(tmp_path, "root")
    model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=3600), "warm")
    row = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, {}, 1, [], set())
    if invalid == "source":
        row["source_ref"] = {**row["source_ref"], "sha256": "0" * 64}
    elif invalid == "attempt":
        row["task_attempt"] = 2
    elif invalid in {"quiz", "review"}:
        row["reason"] = "owner" if invalid == "quiz" else "review"
    elif invalid == "stop":
        from ouroboros.cancel_intents import request_cancel
        request_cancel(tmp_path, "root", reason="stop", source="owner", requested_by="owner")
    owner_wait.set_owner_wait(tmp_path, "root", {**row, "state": "waiting"})
    assert not pause_retention(tmp_path, "root", 1)


from tests.test_llm_claudexor import setup as model_setup  # noqa: F401 - pytest fixture


def test_delegated_start_pause_after_claim_prevents_post(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, delegate_custody
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx, _LiveRunStub, _transport_snapshot
    from ouroboros import subagent_runtime, subagents
    from supervisor.owner_pause_control import request_owner_pause
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    gateway = _LiveRunStub()
    starts = []
    monkeypatch.setattr(gateway, "start_run", lambda *_a, **_kw: starts.append(True) or {"runId": "unwanted"})
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: gateway)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    original = delegate.claimed_start_request
    def claim(*args, **kwargs):
        value = original(*args, **kwargs)
        assert value[0]
        assert request_owner_pause("root", request_id="delegate-final")["ok"]
        return value
    monkeypatch.setattr(delegate, "claimed_start_request", claim)
    # A read-only child (the stub serves readonly only); #1609: technical access alone
    # decides the run shape, so the selected row says so.
    token = subagent_runtime._EXACT_START_SELECTION.set(
        {"snapshot": _transport_snapshot(subagents.get_subagent_harness(), access="readonly")})
    try:
        outcome = delegate._delegate_start(_nanny_ctx(tmp_path, "root"), "work")
    finally:
        subagent_runtime._EXACT_START_SELECTION.reset(token)
    assert not starts and '"reason": "owner_pause"' in outcome.text
    assert not delegate_custody.pending_invocations(tmp_path)
