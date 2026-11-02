"""C1 acceptance: the owner's anonymized parked trees, through custody, Resume and gateway."""

from contextlib import nullcontext
import os
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, owner_pause, owner_wait
from ouroboros.control_actor import mind_control
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tool_custody import owner_proof
from supervisor import owner_pause_control as control
from supervisor.events_budget import install_exact_budget_pause
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests._work_projection_shared import state_fields
from tests.test_direct_model_wait_activity import _js_pause_cards
from tests.test_local_custody_recovery import _dead_process_identity
from tests.test_owner_pause import _live_tree, _owner_park


def _tree(root, monkeypatch):
    queue, state, workers = _install_queue(root, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _live_tree(root, workers)
    workers.PENDING.clear()
    for task_id, meta in workers.RUNNING.items():
        meta["task"]["chat_id"] = 7
        write_task_result(root, task_id, "running", **{k: v for k, v in meta["task"].items() if k != "id"})
    return queue, workers


def _park(root, monkeypatch, queue, workers, task_id, runs=()):
    saved = _owner_park(root, monkeypatch, task_id, list(runs))
    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=queue.sort_pending,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(ctx, task_id, budget_pause.exact_pause_marker(saved)["checkpoint"])
    return next(t for t in workers.PENDING if t["id"] == task_id)


def _card(root, *, tail=""):
    body = state_fields(root)
    [work] = body["works"]
    assert (work["activity"], work["transition"], work["live"]) == ("paused", "none", False)
    assert {"action": "resume", "target": "root-1"} in work["controls"]
    # The full body comes from Python's census and projector, not from a JS status fixture.
    [rendered] = _js_pause_cards([body])
    assert rendered["phase"] == "Paused" and rendered["resume"]
    assert rendered["mark"] == "pause/ink"
    if tail:
        assert tail in rendered["line"]
    else:
        assert work["tails"] == []
    return work


@pytest.mark.parametrize("case, outcome", [("root-a", "succeeded"), ("root-b", "failed"),
                                           ("root-c", "cancelled"), ("member", "succeeded")])
def test_c1_settled_external_park_is_paused_and_resumable(tmp_path, monkeypatch, case, outcome):
    from ouroboros import delegate_custody

    queue, workers = _tree(tmp_path, monkeypatch)
    member = "child-run" if case == "member" else "root-1"
    assert delegate_custody.emit(tmp_path, delegate_custody.STARTED,
                                {"run_id": "closed-run", "task_id": member, "root_task_id": "root-1"})
    assert control.request_owner_pause("root-1", request_id="pause-a")["ok"]
    for task_id in ("child-run", "root-1"):
        _park(tmp_path, monkeypatch, queue, workers, task_id,
              [{"run_id": "closed-run", "state": "running"}] if task_id == member else [])
    assert delegate_custody.emit(tmp_path, delegate_custody.SETTLED,
                                {"run_id": "closed-run", "task_id": member, "state": outcome})
    control._record_settled_member(queue, next(t for t in workers.PENDING if t["id"] == member))
    assert control.refresh_owner_pause_tree("root-1") == "paused"
    pause = budget_pause.budget_pause_row(tmp_path, member)
    assert (pause["state"], pause["settlement"]) == ("pausing", "settled")
    assert not pause.get("paused_confirmed_at")
    work = _card(tmp_path)
    assert work["members"]["states"][member]["activity"] == "paused"
    assert queue.resume_budget_paused_task("root-1")["ok"]
    # The root's exact grant releases the owner fence only when its worker consumes it.
    assert queue.resume_budget_paused_task("child-run")["error"] == "root_still_paused"
    task = next(t for t in workers.PENDING if t["id"] == "root-1")
    workers.PENDING.remove(task)
    workers.RUNNING["root-1"] = {"task": task, "attempt": 1}
    ctx, limit = _loop_ctx(tmp_path, "root-1")
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    saved = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_k: None)
    budget_pause.resume_paused_loop(limit.tools, saved, list(limit.messages), {}, {}, set(), budget_remaining_usd=5.0)
    assert owner_pause.read_fence(tmp_path, "root-1")["state"] == "released"
    child = queue.resume_budget_paused_task("child-run")
    assert child["ok"], child
    assert budget_pause.budget_pause_row(tmp_path, member)["state"] == (
        "resume_granted" if member == "child-run" else "resumed")


def _model_recipients(root, monkeypatch, *, ended):
    from ouroboros import model_wait, platform_layer, usage_accounting as ua

    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    pid, birth = _dead_process_identity() if ended else (os.getpid(), platform_layer.process_start_time(os.getpid()))
    proof = {} if ended else {"local_answer_owner_proof": owner_proof()}  # this live image's session
    for _ in range(4):
        receiver = model_wait.TaskModelWait(task={"id": "child-run", "_attempt": 1}, drive_root=root,
                                            event_queue=None, worker_slot_held=True)
        monkeypatch.setattr(receiver, "bind_answer_consumer", lambda receiver=receiver: {
            "local_answer_consumer_id": receiver.answer_consumer_id, "local_answer_owner_pid": pid,
            "local_answer_owner_birth": birth, "local_answer_task_attempt": 1, **proof})
        with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="child-run", root_task_id="root-1")):
            with model_wait.operation_wait_scope(receiver):
                attempt = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="t", reservation_usd=.2))
                ua.mark_dispatched(attempt, local_answer_owner_pid=os.getpid())
                ua.mark_unresolved(attempt, "receipt_unknown")


@pytest.mark.parametrize("ended", [False, True])
@pytest.mark.parametrize("mind", [False, True])
def test_c1_four_bound_model_receivers_are_unknown_until_addressed_resume(tmp_path, monkeypatch, ended, mind):
    from ouroboros import usage_accounting as ua
    from supervisor.continuation_admission import model_handoffs

    queue, workers = _tree(tmp_path, monkeypatch)
    _model_recipients(tmp_path, monkeypatch, ended=ended)
    assert len(model_handoffs(tmp_path, "root-1")) == 4
    assert control.request_owner_pause("root-1", request_id="pause-b")["ok"]
    for task_id in ("child-run", "root-1"):
        _park(tmp_path, monkeypatch, queue, workers, task_id)
    assert control.refresh_owner_pause_tree("root-1") == "requested"
    before = ua.read_usage_records(tmp_path, final_only=False)
    work = _card(tmp_path, tail="a model request has no recorded result")
    assert [(t["code"], t["state"], len(t["ids"])) for t in work["tails"]] == [("model_handoff", "unknown", 4)]
    assert not load_task_result(tmp_path, "child-run").get("retired_model_consumers"), "GET never repairs"
    with mind_control("controller", "Resume the saved tree") if mind else nullcontext():
        result = queue.resume_budget_paused_task("root-1")
    assert result["ok"] is ended, result
    if ended:
        assert len(load_task_result(tmp_path, "child-run")["retired_model_consumers"]) == 4
        assert model_handoffs(tmp_path, "root-1") == []
    else:
        assert result["error"] == "owner_pause_effects_unsettled"
        assert len([b for b in result["blockers"] if b["kind"] == "model_handoff"]) == 4
        assert "a model answer for task child-run is awaited by a live worker" in result["message"]
        assert "acceptable_unknown" not in result, "a live receiver is never offered to go on past"
    assert ua.read_usage_records(tmp_path, final_only=False) == before, "repair cannot re-execute or settle unknown cost"


@pytest.mark.parametrize("recipient", ["dead", "live", "none"])
@pytest.mark.parametrize("mind", [False, True])
def test_c1_warm_author_and_completed_child_custody(tmp_path, monkeypatch, recipient, mind):
    from ouroboros import delegate_custody, platform_layer
    from supervisor.continuation_admission import conflicting_writers

    queue, workers = _tree(tmp_path, monkeypatch)
    workers.RUNNING.pop("child-run")
    write_task_result(tmp_path, "child-run", "completed", result="Answer delivered")
    if recipient != "none":
        pid, birth = _dead_process_identity() if recipient == "dead" else (
            os.getpid(), platform_layer.process_start_time(os.getpid()))
        write_task_result(tmp_path, "child-run", "completed", launch_handoffs={"verify": {
            "tool": "verify_and_record", "task_id": "child-run", "root_task_id": "root-1", "state": "claimed",
            "local_owner": {"pid": pid, "process_birth": birth, "task_attempt": 1},
            **({"owner_proof": owner_proof()} if recipient == "live" else {})}})
    assert control.request_owner_pause("root-1", request_id="pause-c")["ok"]
    # Already-launched reviews have completed; the retained author stack executes nothing.
    for event, extra in ((delegate_custody.STARTED, {}), (delegate_custody.SETTLED, {"state": "succeeded"})):
        assert delegate_custody.emit(tmp_path, event, {"run_id": "finished-review", "task_id": "root-1",
            "root_task_id": "root-1", "source": "review_substrate", **extra})
    wait = owner_wait.set_owner_wait(tmp_path, "root-1", {"state": "waiting", "reason": "owner_pause"})
    workers.RUNNING["root-1"]["owner_wait"] = wait
    assert control.refresh_owner_pause_tree("root-1") == ("paused" if recipient == "none" else "requested")
    fence = owner_pause.read_fence(tmp_path, "root-1")
    blockers = conflicting_writers(queue, "root-1", owner_pause_fence_id=fence["fence_id"])
    assert [(b["kind"], b["task_id"]) for b in blockers] == (
        [] if recipient == "none" else [("tool_handoff", "child-run")])
    _card(tmp_path)  # no model or external execution is invented for an unresolved tool receipt
    assert not load_task_result(tmp_path, "child-run").get("retired_tool_invocations")
    with mind_control("controller", "Resume the saved author") if mind else nullcontext():
        result = queue.resume_budget_paused_task("root-1")
    assert result["ok"] is (recipient != "live"), result
    child = load_task_result(tmp_path, "child-run")
    assert child["status"] == "completed"
    if recipient == "dead":
        assert child["retired_tool_invocations"]["verify"]["replay_authorized"] is False
    elif recipient == "live":
        assert result["error"] == "owner_pause_effects_unsettled"
        assert any(b["kind"] == "tool_handoff" and b["task_id"] == "child-run" for b in result["blockers"])
        # The live recipient is worded by what is known, never as a call that «has not returned».
        assert "verify_and_record call of task child-run is in a live worker" in result["message"]
        assert "has not returned" not in result["message"]
    assert all(t["id"] != "child-run" for t in workers.PENDING), "Resume never restarts the completed author"
