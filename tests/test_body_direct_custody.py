"""Direct actors transfer through the host adapter before any park event exists."""
from __future__ import annotations

import copy
import threading
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _supervisor_ctx
from tests.test_cold_external_custody import _setup



@pytest.fixture
def direct_env(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, delegate_collection as body
    from ouroboros import server_maintenance as sm, sleep_custody as sc
    from supervisor.active_activity import get_direct_activity_registry

    root = tmp_path / "actor-store"
    env = _setup(root, monkeypatch, direct=True)
    env.root = root
    env.ctx.task_attempt = env.task["_attempt"] = 7
    env.host_root = tmp_path / "host-default"
    env.host_root.mkdir()
    monkeypatch.setattr(sm, "_supervisor_stop", threading.Event())
    monkeypatch.setattr(sm, "_restart_requested", threading.Event())
    monkeypatch.setattr(sc, "_runtime", body.body_runtime)

    class Gateway:
        def get_run(self, rid, **kwargs):
            return {"summary": {"runId": rid, "state": "running", "effectiveAccess": "readonly"}, "lastSeq": 1}
        def close(self):
            pass

    env.runtime = body.start_body_runtime(root, gateway_factory=Gateway)
    registry = get_direct_activity_registry()
    registry.register("sleeper", 0, project_id="project",
                      actor=SimpleNamespace(tools=SimpleNamespace(_ctx=env.ctx)))
    try:
        yield env
    finally:
        registry.unregister("sleeper")
        bp.end_dispatch_fence("sleeper")
        env.runtime.close(wait=True)


def _arm_host_transfer(env, monkeypatch, *, stop_event=None):
    from ouroboros import budget_pause as bp, model_sleep, server_maintenance as sm

    accepted = []
    def host_tick(_ctx):
        assert not env.workers.RUNNING and not env.workers.PENDING, "no park event has been delivered"
        sm._reconcile_cold_custody(env.host_root, stop_event)
        row = bp.budget_pause_row(env.root, "sleeper")
        if row.get("external_custody", {}).get("state") == "accepted":
            accepted.append(copy.deepcopy(row))
        return ""

    monkeypatch.setattr(bp, "_hold_control_reason", host_tick)
    model_sleep.request_sleep(env.ctx, model_sleep.selectors(env.ctx, runs=["run-one"]), "cold")
    return accepted


@pytest.mark.parametrize("root_source", ["budget", "execution"])
def test_host_discovers_direct_transfer_in_its_actual_root_before_park(direct_env, monkeypatch, root_source):
    from ouroboros import budget_pause as bp, sleep_custody as sc
    from supervisor.events_budget import install_exact_budget_pause

    env = direct_env
    if root_source == "budget":
        env.ctx.drive_root = env.host_root
    else:
        env.ctx.budget_drive_root = None
    accepted = _arm_host_transfer(env, monkeypatch)
    with pytest.raises(bp.BudgetPauseRequested) as raised:
        bp.enter_cold_sleep(env.limit)
    row = raised.value.pause
    assert accepted and sc.accepted_receipt("sleeper", row)
    receipt = row["external_custody"]
    assert receipt["runtime"] == env.runtime.identity()
    assert receipt["runs"] == [["run-one", "invoke-one"]]
    assert receipt["binding"]["task_id"] == receipt["binding"]["root_task_id"] == "sleeper"
    assert receipt["binding"]["project_id"] == "project" and receipt["binding"]["task_attempt"] == 7
    assert bp.budget_pause_row(env.host_root, "sleeper") == {}, "host default did not acquire this actor"
    assert not env.workers.PENDING
    sup = _supervisor_ctx(env.root, env.workers, env.q, [], [])
    install_exact_budget_pause(sup, "sleeper", bp.exact_pause_marker(row)["checkpoint"],
                               evt=bp.pause_event(env.task, row))
    carrier = env.workers.PENDING[0]
    assert carrier["id"] == "sleeper" and carrier["project_id"] == "project"
    assert carrier["budget_drive_root"] == str(env.root)
    assert carrier["_budget_pause"]["checkpoint"]["task_attempt"] == 7


def test_mismatched_actor_root_cannot_borrow_the_host_roots_request(direct_env, monkeypatch):
    from ouroboros import budget_pause as bp, routing_wait, server_maintenance as sm
    from supervisor.active_activity import get_direct_activity_registry

    env = direct_env
    # The real actor saves in its root; the registry now names a different owner
    # context. A host-default fallback must not pick up the first request by id.
    activity = get_direct_activity_registry().get("sleeper")
    activity.actor = SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace(
        budget_drive_root=env.host_root, drive_root=env.root)))
    host_root = env.root
    monkeypatch.setattr(routing_wait, "_confirm_wait_sec", lambda _bound: 0)
    _arm_host_transfer(env, monkeypatch)
    monkeypatch.setattr(bp, "_hold_control_reason", lambda _ctx: (sm._reconcile_cold_custody(host_root), "")[1])
    bp.enter_cold_sleep(env.limit)
    row = bp.budget_pause_row(env.root, "sleeper")
    assert row["state"] == "abandoned" and row["external_custody"]["state"] == "requested"
    assert env.ctx._model_sleep is None and not env.workers.PENDING
    assert bp.budget_pause_row(env.host_root, "sleeper") == {}


def test_closing_host_generation_after_body_ensure_cannot_accept_late(direct_env, monkeypatch):
    from ouroboros import budget_pause as bp, routing_wait, server_maintenance as sm

    env, stop = direct_env, threading.Event()
    original, ensured = env.runtime.ensure_runs, []
    def close_generation(pairs, *, task_id=""):
        answer = original(pairs, task_id=task_id)
        ensured.append(answer)
        stop.set()
        return answer
    monkeypatch.setattr(env.runtime, "ensure_runs", close_generation)
    monkeypatch.setattr(routing_wait, "_confirm_wait_sec", lambda _bound: 0)
    accepted = _arm_host_transfer(env, monkeypatch, stop_event=stop)
    bp.enter_cold_sleep(env.limit)
    assert ensured == [True] and not accepted
    row = bp.budget_pause_row(env.root, "sleeper")
    assert row["state"] == "abandoned" and row["external_custody"]["state"] == "requested"
    assert env.runtime.active, "closing the supervisor does not end the lifespan body"
    assert sm._reconcile_cold_custody(env.host_root, stop) == set()
    assert not env.workers.PENDING and env.ctx._model_sleep is None
