"""Explicit Resume revalidates the same native wait after either boot read gap."""
import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_recovery, owner_wait, projects_registry, store, task_results
from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from ouroboros.gateway.state import _chat_activities_snapshot_safe
from supervisor import queue, state, workers
from ouroboros.gateway.state import _pending_resume_available
from supervisor.queue_transitions import pending_member_replay_safe
from ouroboros.work_stop import continuation_refusal
from tests._store_shared import read_result, seed_result, unreadable_result
from tests.test_owner_wait_restart import (
    acknowledge, first_cleanup, restart_case as restart_case, restore_project_wait, restore_stale_snapshot,
)
from tests.test_project_hold_recovery import worker

pytestmark = pytest.mark.serial

EARLY = "restore_refused:owner_wait_source_unavailable"
LATE = "restore_refused:pause_record_unreadable"


def _held(case, monkeypatch, *, stage="early", project=False):
    state.init(case.root)
    if project:
        restored = restore_project_wait(case, monkeypatch, registry_outage=False)
    else:
        expected_ids = {row["id"] for row in workers.PENDING} | {case.task_id}
        first_cleanup(case)
        acknowledge(case, monkeypatch, "launcher")
        assert restore_stale_snapshot(case) == len(expected_ids)
        [restored] = [row for row in workers.PENDING if row["id"] == case.task_id]
    carrier = copy.deepcopy(restored["_owner_wait_resume"])
    exact = read_actor_source_bytes(case.root, case.task_id, carrier["source_ref"])
    assert queue.persist_queue_snapshot(reason="native-wait-gap-test")
    expected_ids = {row["id"] for row in workers.PENDING}
    workers.PENDING.clear()
    reads = []
    with monkeypatch.context() as fault:
        if stage == "early":
            unreadable_result(fault, case.task_id, root=case.root)
        else:
            # The native owner keeps its real imported result reader. The later
            # boot result read imports this boundary and encounters the gap.
            original = task_results.load_task_result

            def unavailable(root, task_id, **kwargs):
                if str(task_id) == case.task_id:
                    reads.append(task_id)
                    raise store.StoreError("later boot result read unavailable")
                return original(root, task_id, **kwargs)

            fault.setattr(task_results, "load_task_result", unavailable)
        assert queue.restore_pending_from_snapshot() == len(expected_ids)
    assert {row["id"] for row in workers.PENDING} == expected_ids
    [kept] = [row for row in workers.PENDING if row["id"] == case.task_id]
    assert continuation_refusal(kept)["reason"] == (EARLY if stage == "early" else LATE)
    assert {k: v for k, v in kept["_owner_wait_resume"].items() if k != "refusal"} == carrier and kept["_attempt"] == case.attempt
    assert not kept.get("_terminalization_retry")
    assert task_results.load_task_result(case.root, case.task_id)["status"] != "failed"
    assert stage == "early" or reads
    sent = worker(SimpleNamespace(root=case.root), monkeypatch)
    return SimpleNamespace(case=case, task=kept, carrier=carrier, exact=exact, sent=sent)


def _activity(env):
    return next(row for row in _chat_activities_snapshot_safe(env.case.root)
                if row["activity_id"] == env.case.task_id)


def _assert_original_dispatch(env):
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in env.sent] == [env.case.task_id]
    launched = env.sent[0]
    assert launched["_attempt"] == env.case.attempt
    assert launched["_owner_wait_resume"] == env.carrier
    assert launched["_owner_wait_resume"]["started_at"] == env.case.started
    saved = owner_wait.load_owner_wait(env.case.ctx, launched["_owner_wait_resume"])
    original = json.loads(env.exact)
    assert saved["messages"] == original["messages"]
    assert saved["trace"] == original["trace"] and saved["round_idx"] == 7
    assert saved["messages"][-1] == {
        "role": "tool", "tool_call_id": "save-1", "content": "Saved draft object 42"}
    assert read_actor_source_bytes(env.case.root, env.case.task_id, env.carrier["source_ref"]) == env.exact


@pytest.mark.parametrize("stage", ["early", "late"])
@pytest.mark.parametrize("project", [False, True], ids=["root", "project-root"])
def test_recovered_native_wait_needs_explicit_resume_then_dispatches_once(restart_case, monkeypatch, stage, project):
    env = _held(restart_case, monkeypatch, stage=stage, project=project)
    workers.assign_tasks()
    workers.assign_tasks()
    assert not env.sent and continuation_refusal(env.task) and not env.task.get("_resume_selection")
    assert not env.task.get("_project_admission_restore_hold")
    assert _pending_resume_available(queue, env.task, owner_paused=False)
    assert _activity(env)["can_resume"] is True
    resumed = queue.resume_budget_paused_task(env.case.task_id)
    assert resumed["ok"] and resumed["same_generation"], resumed
    assert not continuation_refusal(env.task) and not env.task.get("_budget_pause_resume")
    released = copy.deepcopy(env.task)
    again = queue.resume_budget_paused_task(env.case.task_id)
    assert not again["ok"] and env.task == released
    assert not env.sent  # Selection itself starts no worker and replays no tool.
    _assert_original_dispatch(env)


@pytest.mark.parametrize("fault", ["result", "source", "transaction", "unacknowledged", "spent",
                                   "terminal", "cancel", "owner_restart", "deadline"])
def test_explicit_resume_keeps_current_native_authority_refusals(restart_case, monkeypatch, fault):
    from ouroboros.cancel_intents import request_cancel

    env = _held(restart_case, monkeypatch)
    case, task = env.case, env.task
    if fault == "result":
        unreadable_result(monkeypatch, case.task_id, root=case.root)
    elif fault == "source":
        (task_artifact_dir_path(case.root, case.task_id) / env.carrier["source_ref"]["path"]).write_bytes(b"{torn")
    elif fault == "transaction":
        delegate_recovery._restart_transaction_path(case.root, case.transaction_id).write_bytes(b"{torn")
    elif fault == "unacknowledged":
        transaction = delegate_recovery._read_restart_transaction(case.root, case.transaction_id)
        delegate_recovery._write_restart_transaction(case.root, {**transaction, "status": "prepared"})
    elif fault == "spent":
        owner_wait.set_owner_wait(case.root, case.task_id, {**case.wait, "state": "resumed"},
                                  expected_wait_id=case.wait["wait_id"])
    elif fault == "terminal":
        task_results.write_task_result(case.root, case.task_id, "completed", result="Already finished")
    elif fault == "cancel":
        request_cancel(case.root, case.task_id, reason="owner Stop")
    elif fault == "owner_restart":
        from tests._work_stop_helpers import live_restart
        live_restart(case.root, monkeypatch)
    else:
        task["deadline_at"] = "2000-01-01T00:00:00+00:00"
    before = copy.deepcopy(task)
    refused = queue.resume_budget_paused_task(case.task_id)
    assert not refused["ok"], refused
    assert workers.PENDING == [before] and not env.sent
    assert {k: v for k, v in task["_owner_wait_resume"].items() if k != "refusal"} == env.carrier and continuation_refusal(task) and not task.get("_resume_selection")


@pytest.mark.parametrize("carrier,fallback", [(None, "working"), ({}, "zero"),
                                               ("not-a-locator", "working"), ({"wait_id": "other"}, "zero")])
def test_read_gap_without_valid_native_authority_never_falls_back(restart_case, monkeypatch, carrier, fallback):
    env = _held(restart_case, monkeypatch)
    task = env.task
    if fallback == "working":
        from ouroboros import working_checkpoint as wc
        from tests._budget_pause_exact_helpers import _loop_ctx

        # Freeze real prior work before restoring the native park that governs it.
        original = read_result(env.case.root, env.case.task_id)
        seed_result(env.case.root, env.case.task_id, {key: value for key, value in original.items()
                                                    if key != "owner_wait"})
        try:
            _, limit = _loop_ctx(env.case.root, env.case.task_id, attempt=env.case.attempt)
            wc.save_round(limit, "ready")
            task["_attempt"] = env.case.attempt + 1
            assert wc.attach_recovery(env.case.root, task, source_task_id=env.case.task_id,
                                      from_attempt=env.case.attempt, cause="worker_crash")
        finally:
            seed_result(env.case.root, env.case.task_id, original)
        assert read_result(env.case.root, env.case.task_id) == original
    else:
        from ouroboros import usage_store

        seed_result(env.case.root, env.case.task_id, {"status": "scheduled", "total_rounds": 0})
        usage_store.migrate_from_journal(env.case.root)
        task.update(_attempt=1, admitted_dispatch="none")
    no_native = {key: value for key, value in task.items() if key != "_owner_wait_resume"}
    assert pending_member_replay_safe(queue, no_native)[0], "the unrelated fallback must actually be valid"
    # The malformed locator retains its refusal, on that same source kind.
    refusal = continuation_refusal(task)
    task["_owner_wait_resume"] = {**(copy.deepcopy(carrier) if isinstance(carrier, dict) else
                                    {"source_ref": carrier}), "refusal": refusal}
    refused = queue.resume_budget_paused_task(env.case.task_id)
    assert not refused["ok"] and continuation_refusal(task) and not task.get("_resume_selection")
    assert not env.sent and workers.PENDING == [task]


def test_native_selection_does_not_release_the_project_verification_hold(restart_case, monkeypatch):
    env = _held(restart_case, monkeypatch, project=True)
    path = projects_registry._registry_path(env.case.root)
    exact = path.read_bytes()
    path.write_text("{torn", encoding="utf-8")
    workers.assign_tasks()
    hold = copy.deepcopy(env.task["_project_admission_restore_hold"])
    resumed = queue.resume_budget_paused_task(env.case.task_id)
    assert resumed["ok"], resumed  # Intent does not certify the Project or dispatch a worker.
    assert not continuation_refusal(env.task)
    assert env.task["_project_admission_restore_hold"] == hold
    workers.assign_tasks()
    assert not env.sent and env.task["_project_admission_restore_hold"]
    path.write_bytes(exact)
    _assert_original_dispatch(env)
    assert not env.task.get("_project_admission_restore_hold")


def test_failed_selection_publication_keeps_the_native_wait_unselected(restart_case, monkeypatch):
    env = _held(restart_case, monkeypatch)
    before = copy.deepcopy(env.task)
    persist = queue.persist_queue_snapshot
    with monkeypatch.context() as fault:
        fault.setattr(queue, "persist_queue_snapshot", lambda reason="":
                      False if reason == "budget_hold_selected" else persist(reason=reason))
        refused = queue.resume_budget_paused_task(env.case.task_id)
    assert refused["ok"] is False and refused["error"] == "snapshot_not_persisted"
    assert env.task == before and not env.sent
    workers.assign_tasks()
    assert not env.sent and continuation_refusal(env.task) and not env.task.get("_resume_selection")
    assert queue.resume_budget_paused_task(env.case.task_id)["ok"]
    assert not continuation_refusal(env.task)
    _assert_original_dispatch(env)


def test_technical_refusal_text_does_not_override_the_native_source_kind(restart_case, monkeypatch):
    env = _held(restart_case, monkeypatch)
    continuation_refusal(env.task)["reason"] = "restore_refused:invalid_acceptance_fence_snapshot"
    assert _pending_resume_available(queue, env.task, owner_paused=False)
    assert _activity(env)["can_resume"] is True
    assert queue.resume_budget_paused_task(env.case.task_id)["ok"]
    assert not continuation_refusal(env.task) and not env.task.get("_resume_selection")
    _assert_original_dispatch(env)


@pytest.mark.parametrize("stage", ["early", "late"])
def test_technical_resume_rejoins_native_work_without_monetary_selection(restart_case, monkeypatch, stage):
    from ouroboros import usage_accounting as accounting
    from supervisor import events_budget

    case = restart_case
    state.init(case.root)
    case.task.update(root_task_id=case.task_id, delegation_role="root")
    sibling = {"id": "unstarted-child", "type": "task", "chat_id": 1, "depth": 1,
               "root_task_id": case.task_id, "parent_task_id": case.task_id,
               "delegation_role": "subagent", "text": "Independent work"}
    assert not queue.enqueue_task(sibling).get("_admission_blocked")
    task_results.write_task_result(case.root, "spent-child", "failed", root_task_id=case.task_id,
                                   parent_task_id=case.task_id, reason_code="budget_exhausted")
    events_budget._handle_budget_root_fence({
        "type": "budget_root_fence", "task_id": "spent-child", "task_type": "task",
        "resource_limit": {"scope": "root", "root_task_id": case.task_id},
    }, SimpleNamespace(DRIVE_ROOT=case.root, RUNNING=workers.RUNNING,
                       persist_queue_snapshot=queue.persist_queue_snapshot,
                       bridge=SimpleNamespace(push_log=lambda _event: None)))
    fence = copy.deepcopy(queue.BUDGET_ROOT_FENCES[case.task_id])
    env = _held(case, monkeypatch, stage=stage)
    siblings = copy.deepcopy([row for row in workers.PENDING if row["id"] != case.task_id])
    assert [row["id"] for row in siblings] == [sibling["id"]]
    monkeypatch.setattr(state, "budget_remaining", lambda *_args, **_kwargs: 100.0)
    workers.assign_tasks()
    assert not env.sent and continuation_refusal(env.task) and not env.task.get("_resume_selection")
    # Assignment independently rechecks the sibling before Resume touches the root.
    [revalidated] = [row for row in workers.PENDING if row["id"] != case.task_id]
    assert siblings[0]["_project_admission_restore_hold"]["reason"] == "project_routing_fence_lookup_failed"
    assert revalidated["_project_admission_restore_hold"]["reason"] == "project_dispatch_unconfirmed"
    assert revalidated == {**siblings[0],
                           "_project_admission_restore_hold": revalidated["_project_admission_restore_hold"]}
    siblings = copy.deepcopy([revalidated])
    assert queue.resume_budget_paused_task(case.task_id)["ok"]
    assert not continuation_refusal(env.task) and not env.task.get("_budget_pause_resume")
    assert env.task["_owner_wait_resume"] == env.carrier
    assert events_budget.live_root_resume_grant(queue, case.task_id, case.root) == {}
    released = copy.deepcopy(env.task)
    # This second click now meets only the monetary fence. Native restore proof
    # must not turn an ordinary money Resume into permission to spend or select children.
    duplicate = queue.resume_budget_paused_task(case.task_id)
    assert not duplicate["ok"] and env.task == released
    assert not env.sent and queue.BUDGET_ROOT_FENCES == {case.task_id: fence}
    assert [row for row in workers.PENDING if row["id"] != case.task_id] == siblings
    _assert_original_dispatch(env)
    assert workers.PENDING == siblings and queue.BUDGET_ROOT_FENCES == {case.task_id: fence}
    assert events_budget.live_root_resume_grant(queue, case.task_id, case.root) == {}
    assert json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))["budget_root_fences"] == [fence]
    fresh = queue.enqueue_task({**sibling, "id": "fresh-child"})
    assert fresh["_admission_blocked"] == "root_budget_fence"
    with accounting.usage_scope(accounting.UsageScope(
            drive_root=case.root, task_id=case.task_id, root_task_id=case.task_id,
            global_limit_usd=100.0, root_limit_usd=100.0)):
        with pytest.raises(accounting.BudgetExceeded) as refused:
            accounting.reserve_attempt(accounting.AttemptRequest(
                model="fixture", provider="openai", reservation_usd=1.0))
        assert refused.value.limit_scope == "root"


def test_owner_pause_after_technical_release_cannot_resume_native_work_as_unstarted(restart_case, monkeypatch):
    from ouroboros.owner_pause import fence_closed, read_fence
    from supervisor.events_budget import live_root_resume_grant
    from supervisor.owner_pause_control import request_owner_pause

    case = restart_case
    state.init(case.root)
    case.task.update(root_task_id=case.task_id, delegation_role="root")
    sibling = {"id": "pending-sibling", "type": "task", "chat_id": 1, "depth": 1,
               "root_task_id": case.task_id, "parent_task_id": case.task_id,
               "delegation_role": "subagent", "text": "Independent work"}
    assert not queue.enqueue_task(sibling).get("_admission_blocked")
    env = _held(case, monkeypatch)
    assert queue.resume_budget_paused_task(case.task_id)["ok"]
    assert not continuation_refusal(env.task) and not env.sent
    assert env.task["_owner_wait_resume"] == env.carrier
    paused = request_owner_pause(case.task_id, request_id="pause-native-pending")
    assert paused["ok"], paused
    latch = copy.deepcopy(queue.BUDGET_ROOT_FENCES[case.task_id])
    durable = read_fence(case.root, case.task_id)
    assert latch["cause"] == "owner_pause" and fence_closed(durable)
    pending = copy.deepcopy(workers.PENDING)
    assert {row["id"] for row in pending} == {case.task_id, sibling["id"]}
    refused = queue.resume_budget_paused_task(case.task_id)
    assert not refused["ok"] and refused["error"] == "dispatch_outcome_unknown", refused
    assert not refused.get("never_started")
    assert workers.PENDING == pending and queue.BUDGET_ROOT_FENCES == {case.task_id: latch}
    assert read_fence(case.root, case.task_id) == durable
    assert live_root_resume_grant(queue, case.task_id, case.root) == {}
    workers.assign_tasks()
    workers.assign_tasks()
    assert not env.sent and not workers.RUNNING
    assert fence_closed(read_fence(case.root, case.task_id))
    assert env.task["_owner_wait_resume"] == env.carrier
    assert read_actor_source_bytes(case.root, case.task_id, env.carrier["source_ref"]) == env.exact
