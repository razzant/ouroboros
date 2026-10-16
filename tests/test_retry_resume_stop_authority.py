"""Real retry -> Pause -> pending Stop -> public Resume, across all grant doors.

Cancellation is written by request_cancel, never replaced by a boolean double.
The common fixtures double OS/model boundaries; warm park saves through the real
serializer and worker event handler, interrupted only after its acknowledgement.
"""
import copy
import sys
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, cancel_intents, owner_pause, owner_wait
from supervisor import queue, workers
from supervisor.owner_pause_control import request_owner_pause, refresh_owner_pause_tree
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_execution_clock_resume_handover import clock as clock
from tests.test_retry_pause_generations import _park, _running_actor
from tests.test_retry_pause_identity import (
    _assign, _fresh_plan, _loop_once, _ordinary_retry, _reap, reaper as reaper,
)
from tests.test_swarm_host_admission import host as host

pytestmark = pytest.mark.serial


class Parked(BaseException):
    pass


def paused_retry(host, monkeypatch, clock, reaper, door):  # noqa: F811 - imported pytest fixtures
    task = (_ordinary_retry(host, monkeypatch, clock, reaper) if door == "queued"
            else _running_actor(host, monkeypatch, clock, reaper, "retry"))
    child = queue.enqueue_task({"id": "unselected", "type": "task", "text": "child work", "chat_id": 0,
                               "root_task_id": "held", "parent_task_id": "held", "depth": 1,
                               "delegation_role": "subagent"})
    assert not child.get("_admission_blocked")
    assert request_owner_pause(task["id"], request_id="pause")["ok"]
    plan = _fresh_plan(monkeypatch)
    if door == "exact":
        _park(host, task, plan)
    elif door in {"warm", "unsaved"}:
        ctx, limit = _loop_ctx(host.root, task["id"], attempt=task["_attempt"])
        ctx.task_metadata = {**task, "budget_drive_root": str(host.root)}
        ctx.active_model, ctx.context_fit_plan = plan.model, plan
        if door == "warm":
            from supervisor.worker_owner_wait import handle_owner_wait

            slot = workers.WORKERS[0]
            slot.proc = SimpleNamespace(pid=12345)
            def park(_ctx, checkpoint):
                handle_owner_wait({"phase": "park", "task_id": task["id"], "worker_id": 0,
                    "pid": 12345, "task_attempt": task["_attempt"], "wait_id": checkpoint["wait_id"],
                    "checkpoint": checkpoint}, workers)
                assert workers.RUNNING[task["id"]]["owner_wait"]["state"] == "waiting"
                raise Parked
            ctx.owner_wait_callback = park
            with pytest.raises(Parked):
                owner_wait.park_owner_pause_warm(limit, ctx, fence=owner_pause.read_fence(host.root, "held"),
                                                detached=[])
        else:
            # Death after the real first-boundary seed, before any source for THIS attempt.
            limit.round_idx = 0
            with monkeypatch.context() as fault:
                def die(*_a, **_k):
                    raise Parked
                fault.setattr(budget_pause, "_exact_continuation_row", die)
                with pytest.raises(Parked):
                    budget_pause.enter_owner_pause(limit)
            budget_pause.end_dispatch_fence(task["id"])
            _reap(clock, reaper, task["id"])
            task = next(row for row in host.pending if row["id"] == task["id"])
            assert task["_budget_pause_hold"]["reason"] == "unsaved_pause_intent"
        refresh_owner_pause_tree("held")
    return task, child, plan


@pytest.mark.parametrize("door", ["warm", "exact", "queued", "unsaved"])
@pytest.mark.parametrize("stop", ["logical_graceful", "logical_immediate", "physical", "healthy"])
def test_pending_stop_vetoes_retry_resume(host, monkeypatch, clock, reaper, door, stop):  # noqa: F811
    task, child, plan = paused_retry(host, monkeypatch, clock, reaper, door)
    task_id = task["id"]
    if stop != "healthy":
        logical = stop.startswith("logical")
        intent = cancel_intents.request_cancel(host.root, "held" if logical else task_id,
            source="web", requested_by="owner", reason="owner stop", allow_settled_target=True,
            scope="cascade" if logical else "single",
            requested_stop_policy="finalize_then_cancel" if stop == "logical_graceful" else "immediate")
        assert intent["task_id"] == ("held" if logical else task_id), intent
        assert cancel_intents.has_active_intent(host.root, intent["task_id"], strict=True)
        if logical:
            assert not cancel_intents.has_active_intent(host.root, task_id, strict=True)
    before = copy.deepcopy(owner_pause.read_fence(host.root, "held"))
    latch = copy.deepcopy(queue.BUDGET_ROOT_FENCES)
    pause = copy.deepcopy(budget_pause.budget_pause_row(host.root, task_id))
    result = queue.resume_budget_paused_task(task_id)
    print("Stop/Resume", {"door": door, "stop": stop, "task": task_id, "result": result})
    if stop != "healthy":
        assert not result["ok"], result
        assert result["error"] in {"cancel_intent_active", "selection_owner_held"}, result
        assert owner_pause.read_fence(host.root, "held") == before
        assert queue.BUDGET_ROOT_FENCES == latch
        assert budget_pause.budget_pause_row(host.root, task_id) == pause
        assert not task.get("_budget_pause_resume")
    else:
        assert result["ok"], result
        if door != "warm":
            [sent] = _assign(host, monkeypatch)
            assert sent["id"] == task_id
            if door == "exact":
                assert _loop_once(host, monkeypatch, sent, plan=plan)["kind"] == "pause"
        assert owner_pause.read_fence(host.root, "held")["state"] == "released"
        assert "held" not in queue.BUDGET_ROOT_FENCES
        assert child in host.pending and not child.get("_budget_pause_hold", {}).get("selected")
        assert not _assign(host, monkeypatch), "old children need explicit selection"


@pytest.mark.parametrize("door", ["warm", "exact", "queued", "unsaved"])
def test_unknown_root_identity_refuses_without_member_grant(host, monkeypatch, clock, reaper, door):  # noqa: F811
    task, _child, _plan = paused_retry(host, monkeypatch, clock, reaper, door)
    from ouroboros.task_store import patch_document
    # The retired root can be terminal while its reciprocal physical successor lives.
    patch_document(host.root, "held", lambda row: {**row, "status": "failed"})
    before = copy.deepcopy(owner_pause.read_fence(host.root, "held"))
    real, reads = cancel_intents._validated_single_cancel_target, []
    def transient(*args, **kwargs):
        # Only the grant's classification fails; actual source/custody readers stay real.
        if sys._getframe(1).f_code.co_name == "root_actor" and sys._getframe(2).f_code.co_name in {
                "resume_warm_owner_pause_root", "_grant_exact_resume", "resume_budget_paused_task"}:
            reads.append(args[1])
            raise OSError("transient lineage read failure")
        return real(*args, **kwargs)
    with monkeypatch.context() as fault:
        fault.setattr(cancel_intents, "_validated_single_cancel_target", transient)
        result = queue.resume_budget_paused_task(task["id"])
    print("Unknown identity", {"door": door, "reads": reads, "result": result})
    assert reads == ["held"]
    # Queued selection already refuses via its existing owner-fence authority;
    # warm/exact grant issuers must not reinterpret unknown as member authority.
    expected = "selection_owner_held" if door in {"queued", "unsaved"} else "owner_pause_custody_unreadable"
    assert result == {"ok": False, "error": expected}, result
    assert owner_pause.read_fence(host.root, "held") == before
    assert queue.resume_budget_paused_task(task["id"])["ok"]


@pytest.mark.parametrize("door", ["warm", "exact", "queued", "unsaved"])
@pytest.mark.parametrize("veto", ["unreadable_cancel", "legacy_cancel", "panic", "restart"])
def test_unknown_cancel_and_existing_stop_carriers_keep_the_pause(host, monkeypatch, clock, reaper, door, veto):  # noqa: F811
    from ouroboros.task_store import patch_document

    task, _child, _plan = paused_retry(host, monkeypatch, clock, reaper, door)
    if veto == "legacy_cancel":
        patch_document(host.root, "held", lambda row: {**row, "status": "cancel_requested"})
    elif veto in {"panic", "restart"}:
        (host.root / "state" / ("panic_stop.flag" if veto == "panic" else "owner_restart_no_resume.flag")).touch()
    before = copy.deepcopy(owner_pause.read_fence(host.root, "held"))
    latch = copy.deepcopy(queue.BUDGET_ROOT_FENCES)
    with monkeypatch.context() as fault:
        if veto == "unreadable_cancel":
            real = cancel_intents.active_intents
            def unavailable(*args, **kwargs):
                if kwargs.get("strict"):
                    raise OSError("cancel projection unavailable")
                return real(*args, **kwargs)
            fault.setattr(cancel_intents, "active_intents", unavailable)
        result = queue.resume_budget_paused_task(task["id"])
    print("Resume veto", {"door": door, "veto": veto, "result": result})
    assert not result["ok"], result
    assert result["error"] in {"cancellation_authority_unavailable", "cancel_intent_active",
                               "selection_observation_unavailable", "selection_owner_held", "restart_no_resume"}
    assert owner_pause.read_fence(host.root, "held") == before and queue.BUDGET_ROOT_FENCES == latch
    assert not task.get("_budget_pause_resume")
