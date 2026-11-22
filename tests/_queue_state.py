"""Isolated queue and ledger inputs for the real state and task-control handlers."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """Real gateway handlers and source readers; no supervisor lifespan or workers."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config, usage_store
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.state import api_state
    from ouroboros.gateway.tasks import api_task_resume
    from supervisor import active_activity, queue, state, workers

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(config, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(state, "DRIVE_ROOT", tmp_path)
    for name, relative in (("STATE_PATH", "state/state.json"),
                           ("STATE_LAST_GOOD_PATH", "state/state.last_good.json"),
                           ("STATE_LOCK_PATH", "locks/state.lock")):
        monkeypatch.setattr(state, name, tmp_path / relative)
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    state.save_state({"owner_id": 1, "current_branch": "fixture", "current_sha": "a" * 40,
                      "bg_consciousness_enabled": False, "evolution_mode_enabled": False})
    usage_store.migrate_from_journal(tmp_path)
    pending, running = [], {}
    for module in (queue, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", running)
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", active_activity.DirectActivityRegistry())
    app = Starlette(routes=[Route("/api/state", api_state),
                            Route("/api/chat/history", make_chat_history_endpoint(tmp_path)),
                            Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])
    app.state.drive_root = tmp_path
    app.state.app_start = 0.0
    with TestClient(app) as client:
        yield SimpleNamespace(root=tmp_path, client=client, pending=pending, running=running)


def read_state(gateway):
    response = gateway.client.get("/api/state")
    assert response.status_code == 200, response.text
    return response.json()


def assignment_with_an_evolution_row(data_root, monkeypatch, *, settle_at):
    """The real ``assign_tasks`` over a real ledger: a $5 limit, one lane whose $4 hold is
    in flight (the warmed snapshot still says $5: a hold is not spending, #1487) and then
    settles at ``settle_at`` — the exact read must see that known charge."""
    from ouroboros import usage_accounting as ua
    from supervisor import queue, state, workers
    from tests._usage_store_testing import request
    from tests.worker_pool_support import affirm_ready

    state.init(data_root, total_budget_limit=5.0)
    queue.init(data_root)
    for module in (workers, queue):
        monkeypatch.setattr(module, "DRIVE_ROOT", data_root)
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda: None)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(workers, "_evolution_assignment_error", lambda _task: "")
    pending, running, pool = [], {}, {}
    for name, value in (("PENDING", pending), ("RUNNING", running), ("WORKERS", pool)):
        monkeypatch.setattr(workers, name, value)
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    queue.init_queue_refs(pending, running, workers.QUEUE_SEQ_COUNTER_REF)
    reservation = ua.reserve_attempt(request(
        data_root, model="openai/gpt-5.2", provider="openai", task_id="task", root_task_id="root",
        reservation_usd=4.0, global_limit_usd=5.0, source="test",
    ))
    ua.mark_dispatched(reservation)
    assert state.budget_remaining({}, strict=True, allow_stale=True) == pytest.approx(5.0)  # warms the memo
    ua.settle_attempt(reservation, {"prompt_tokens": 5, "completion_tokens": 2}, cost_usd=settle_at, cost_final=True)
    sent: list = []
    pool[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                              in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    pending.append({"id": "evo-1", "type": "evolution", "chat_id": 0, "priority": 1})
    reasons: list = []
    real_persist = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot",
                        lambda reason="", **kw: (reasons.append(reason), real_persist(reason=reason, **kw))[1])
    return workers, pending, reasons
