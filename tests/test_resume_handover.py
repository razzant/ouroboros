"""#1716: an exact Resume hands its restored cognition to the attempt's rolling
working checkpoint before it consumes the grant, so an application stop on
either side of that consumption continues exact-or-newer cognition — never an
older saved-work locator the attempt carried from an earlier recovery.

The real producers run: Project/Main admission, the agent's pause handler and
checkpoint retirement, the exact park, Resume grants, assignment, the loop's
loaders and restorers, the application-stop teardown and the boot restore.
Process signals, worker queues and model calls are doubled; a ``BaseException``
models the loss of the process at one exact boundary.
"""
from __future__ import annotations

import copy
import json
import queue as queue_lib
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import agent, budget_pause, loop, working_checkpoint as wc
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, route_binding
from supervisor import queue, workers
from supervisor.events_budget import budget_hold_fact, install_exact_budget_pause
from supervisor.owner_pause_control import request_owner_pause
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_context_fit_v664 import _plan
from tests.test_loop_transport_wait import _loop_kwargs
from tests.test_project_hold_recovery import worker
from tests.test_restart_saved_work import _ack
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.test_working_checkpoint import _registry
from tests.test_working_recovery_admission import _running

pytestmark = pytest.mark.serial

MARK = "EXACT_R29_COGNITION"


class ProcessLost(BaseException):
    """The application process ends at this exact boundary; no handler runs."""


class _DeadProcess:
    """No OS process: the stop's kill makes it dead and its join verifies that."""
    pid = 2147483001

    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        assert not self.alive

    def kill(self, pid):
        assert pid == self.pid
        self.alive = False


def _stop_boot(monkeypatch, *, returning=False):
    """The application's own stop teardown, then the next generation's boot restore."""
    from supervisor.restart_retention import prepare_restart_returns

    if returning:  # an acknowledged Restart names what it returns before stopping
        prepare_restart_returns(queue.DRIVE_ROOT, workers.RUNNING, workers.PENDING, transaction_id="tx-1716")
        _ack(queue.DRIVE_ROOT, "tx-1716")
    process = _DeadProcess()
    workers.WORKERS.clear()
    workers.WORKERS[0] = SimpleNamespace(wid=0, proc=process, busy_task_id=next(iter(workers.RUNNING), None),
                                         reaping=False)
    monkeypatch.setattr(workers, "kill_worker_tree", process.kill)
    monkeypatch.setattr(workers, "_kill_survivors", lambda: None)
    assert workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                                stop_source="server_shutdown", retain_saved_work=True)
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    return queue.restore_pending_from_snapshot()


def _start_record(root, task):
    agent.OuroborosAgent._persist_running_record(SimpleNamespace(
        env=SimpleNamespace(drive_root=root), _task_started_ts=time.time()), task)


def _dispatch_one(host, monkeypatch):  # noqa: F811
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert len(sent) == 1 and not host.pending, (sent, host.pending)
    _start_record(host.root, sent[0])
    return sent[0]


def _pause_through_agent(host, task, *, rail, tail, plan=None):  # noqa: F811
    """Round 29 of the recovered attempt meets a pause; the real agent handler parks it."""
    if rail == budget_pause.RAIL_OWNER_PAUSE:
        assert request_owner_pause("held", request_id="handover-29")["ok"]
    ctx, limit = _loop_ctx(host.root, "held", attempt=2)
    if plan is not None:  # the route the attempt runs on (a fallback the chain installed)
        ctx.active_model, ctx.context_fit_plan = plan.model, plan
    limit.round_idx, limit.budget_tail = 29, tail
    limit.messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": MARK}]
    if tail == "tool":  # an interrupted batch: one call answered, one with no result
        limit.messages[1] = {"role": "assistant", "content": MARK, "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "y", "arguments": "{}"}}]}
        limit.messages.append({"role": "tool", "tool_call_id": "call_a", "content": "done a"})
    wc.save_round(limit, "post_batch")  # the attempt had its own rolling state before the pause
    owner = object.__new__(agent.OuroborosAgent)
    owner.env = agent.Env(repo_dir=host.root, drive_root=host.root)
    owner.tools = SimpleNamespace(_ctx=ctx)
    owner._owner_message_admission_lock = threading.RLock()
    owner._incoming_messages = queue_lib.Queue()
    owner._event_queue = None
    owner._emit_live_log = lambda *a, **k: None
    owner._start_task_heartbeat_loop = lambda *a: None

    def prepare(*_args):
        # Context construction is irrelevant to the pause protocol: the real producer
        # raises into the real agent's BudgetPauseRequested handler.
        if rail == budget_pause.RAIL_OWNER_PAUSE:
            budget_pause.enter_owner_pause(limit)
        budget_pause.request_pause(limit, rail=rail, scope="global", reason_text="money gone")
        pytest.fail("the pause did not raise")

    owner._prepare_task_context = prepare
    events = owner._handle_task_scoped(task)
    budget_pause.end_dispatch_fence("held")
    assert any(event.get("type") == "budget_pause" for event in events)
    assert not wc.checkpoint_path(host.root, "held", 2).exists(), "pause retirement discarded the rolling file"
    pause = budget_pause.budget_pause_row(host.root, "held")
    install_exact_budget_pause(SimpleNamespace(
        DRIVE_ROOT=host.root, RUNNING=workers.RUNNING, PENDING=workers.PENDING, WORKERS=workers.WORKERS,
        sort_pending=queue.sort_pending, persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None),
        "held", budget_pause.exact_pause_marker(pause)["checkpoint"])
    return pause


def _recovered_then_paused(host, tmp_path, monkeypatch, scope, *,  # noqa: F811
                           rail=budget_pause.RAIL_OWNER_PAUSE, tail="tool", plan=None):
    """a1/r4 saved → stop → boot → Resume → a2 loads r4 → exact pause at r29 → Resume granted."""
    _running(host, tmp_path, monkeypatch, scope)
    assert _stop_boot(monkeypatch) == 1
    assert queue.resume_budget_paused_task("held")["ok"]
    attempt2 = _dispatch_one(host, monkeypatch)
    assert attempt2["_attempt"] == 2 and attempt2["_working_recovery"]["from_attempt"] == 1
    ctx, _limit = _loop_ctx(host.root, "held", attempt=2)
    ctx.working_recovery = attempt2["_working_recovery"]
    assert loop._load_working_recovery(ctx, [])["round_idx"] == 4
    pause = _pause_through_agent(host, attempt2, rail=rail, tail=tail, plan=plan)
    granted = queue.resume_budget_paused_task("held")
    assert granted["ok"] and granted["exact_continuation"], granted
    return pause


def _run_loop(host, monkeypatch, task, cut, *, waiter=False, plan=None):  # noqa: F811
    """The real loop of one dispatched attempt, until the process is lost at ``cut``."""
    registry = _registry(host.root, monkeypatch, task["id"], task["_attempt"])
    if plan is not None:  # the new worker's initial plan; its route's account evidence is synthetic
        registry._ctx.context_fit_plan = plan
        monkeypatch.setattr("ouroboros.context._context_fit_route", _account_route)
    registry._ctx.model_wait_context = None
    if waiter:
        from ouroboros.model_wait import TaskModelWait

        registry._ctx.model_wait_context = TaskModelWait(task=task, drive_root=host.root, event_queue=None,
                                                         worker_slot_held=False)
    registry._ctx.task_started_at = float((task.get("_budget_pause_resume") or task.get("_working_recovery")
                                           or {}).get("started_at") or time.time())
    agent._restore_saved_clocks(registry._ctx, task)
    real_resume, real_ready, real_cas = loop._resume_continuation, wc.save_ready, budget_pause.set_budget_pause
    real_reopen = budget_pause._reopen_owner_fence
    seen = {"ctx": registry._ctx}

    def resumed(*args, **kwargs):
        answer = real_resume(*args, **kwargs)
        saved, pause, working = args[1]
        state = saved or pause or working
        seen.update(kind="pause" if pause else "working" if working else "none",
                    round_idx=state.get("round_idx"), task_attempt=state.get("task_attempt"),
                    boundary=(state.get("working") or {}).get("boundary"),
                    messages=copy.deepcopy(args[2]), tails=answer[-3:])
        if cut == "after_resume":
            raise ProcessLost("resumed, before the first ordinary save")
        return answer

    def lose(*_args, **_kwargs):
        raise ProcessLost(cut)

    def cas(root, task_id, row, *args, **kwargs):
        if row.get("state") == budget_pause.STATE_RESUMED:
            raise ProcessLost("bridge written, grant not yet consumed")
        return real_cas(root, task_id, row, *args, **kwargs)

    def reopen_then_lose(*args, **kwargs):
        real_reopen(*args, **kwargs)
        raise ProcessLost("fence reopened, nothing saved since")

    def ready(limit):
        real_ready(limit)
        seen["saved"] = wc.read_working(host.root, task["id"], task["_attempt"])[1]
        raise ProcessLost("after the first ordinary save")

    with monkeypatch.context() as fault:
        fault.setattr(loop, "_resume_continuation", resumed)
        fault.setattr(wc, "save_ready", ready)
        fault.setattr(loop, "_dispatch_round_model", lambda *a, **k: pytest.fail("model dispatch before the cut"))
        if cut == "before_bridge":
            fault.setattr(wc, "save_resumed", lose)
        elif cut == "after_bridge":
            fault.setattr(budget_pause, "set_budget_pause", cas)
        elif cut == "after_cas":
            fault.setattr(budget_pause, "_reopen_owner_fence", lose)
        elif cut == "after_fence":
            fault.setattr(budget_pause, "_reopen_owner_fence", reopen_then_lose)
        elif cut == "finisher":
            for name in ("_finish_tool_round_budget", "_finish_no_tool_round_budget"):
                fault.setattr(loop, name, lambda limit, *_a, _name=name, **_k: (
                    seen.update(finisher=(_name, limit.round_idx)), lose()))
        kwargs = _loop_kwargs(host.root, registry, [])
        kwargs["task_id"] = task["id"]
        with pytest.raises(ProcessLost):
            loop.run_llm_loop(**kwargs)
    return seen


def _carried(monkeypatch):
    """Record every non-empty carry of an older saved-work locator."""
    real, carried = wc._carry_recovery, []

    def carry(*args, **kwargs):
        answer = real(*args, **kwargs)
        if answer:
            carried.append(answer)
        return answer

    monkeypatch.setattr(wc, "_carry_recovery", carry)
    return carried


def _has_mark(messages):
    return any(row.get("content") == MARK for row in messages)


def _account_route(task, **_kwargs):
    """Advertised evidence for the account the role's route binds; no provider is asked."""
    model, local, account = route_binding(task["model"], task["use_local_model"], task["model_role"])
    return ({"model": model, "provider": "claudexor", "use_local": local}, SimpleNamespace(
        status="confirmed", stale=False, window_tokens=500_000, route_fp="route-" + account, source_id="codex",
        source="advertised", credential_profile_id=account, account_fingerprint="fingerprint-" + account))


# --- the reproduced sequence (issue #1716 qualification) -----------------------------------------

@pytest.mark.parametrize("cut", ["after_resume", "after_save"])
@pytest.mark.parametrize("scope", ["project", "main"])
def test_consumed_exact_resume_then_application_stop_never_restores_older_cognition(
    host, tmp_path, monkeypatch, scope, cut,  # noqa: F811
):
    pause = _recovered_then_paused(host, tmp_path, monkeypatch, scope)
    resumed = _dispatch_one(host, monkeypatch)
    assert resumed["_attempt"] == 2 and resumed["_budget_pause_resume"]["pause_id"] == pause["pause_id"]
    seen = _run_loop(host, monkeypatch, resumed, cut)
    assert seen["kind"] == "pause" and seen["round_idx"] == 29 and _has_mark(seen["messages"])
    consumed = budget_pause.budget_pause_row(host.root, "held")
    assert consumed["state"] == budget_pause.STATE_RESUMED and consumed["grant"]["consumed_at"]
    carried = _carried(monkeypatch)
    assert _stop_boot(monkeypatch) == 1
    [successor] = host.pending
    assert successor["_attempt"] == 3 and "_budget_pause_resume" not in successor and budget_hold_fact(successor)
    assert queue.resume_budget_paused_task("held")["ok"]
    final = _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "after_resume")
    restored = (final["kind"], final["task_attempt"], final["round_idx"], _has_mark(final["messages"]))
    assert restored == ("working", 2, 29 if cut == "after_resume" else 30, True), "an older saved locator won"
    assert not carried, "the older attempt-1 locator was carried past the attempt's own state"
    assert final["boundary"] == ("resumed" if cut == "after_resume" else "ready")
    if cut == "after_save":  # an ordinary ready restore keeps its own flags
        assert final["tails"] == (False, False, True)
    unknown = [row for row in final["messages"] if row.get("tool_call_id") == "call_b"]
    assert len(unknown) == 1 and "UNKNOWN" in unknown[0]["content"], "closed once, never re-executed"
    assert not host.attempts


# --- every cut of the handover, through the real stop, boot and Resume doors -------------------

@pytest.mark.parametrize("cut", ["before_bridge", "after_bridge", "after_cas", "after_fence"])
@pytest.mark.parametrize("scope", ["project", "main"])
def test_every_cut_of_the_handover_continues_exact_or_newer_cognition(
    host, tmp_path, monkeypatch, scope, cut,  # noqa: F811
):
    from ouroboros.owner_pause import fence_closed, read_fence

    _recovered_then_paused(host, tmp_path, monkeypatch, scope)
    _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), cut)
    consumed = cut in {"after_cas", "after_fence"}
    assert (budget_pause.budget_pause_row(host.root, "held")["state"] == budget_pause.STATE_RESUMED) is consumed
    assert wc.checkpoint_path(host.root, "held", 2).exists() is (cut != "before_bridge"), "written before the CAS"
    assert fence_closed(read_fence(host.root, "held")) is (cut != "after_fence")
    carried = _carried(monkeypatch)
    assert _stop_boot(monkeypatch) == 1
    [row] = host.pending
    if not consumed:
        # The stop revoked the unconsumed grant: the live pause still owns recovery
        # and dominates a handover written before the CAS.
        assert row["_attempt"] == 2 and row["_budget_pause"]["reason"] == "owner"
        assert budget_pause.budget_pause_row(host.root, "held")["grant"]["revoked_at"]
        assert not wc.prepare_recovery(host.root, "held", from_attempt=2, cause="app_stop")
        assert queue.resume_budget_paused_task("held")["exact_continuation"]
        expected = ("pause", 2, 29, True)
    else:
        assert row["_attempt"] == 3 and row["_working_recovery"]["boundary"] == "resumed"
        assert queue.resume_budget_paused_task("held")["ok"]
        expected = ("working", 2, 29, True)
    final = _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "after_resume")
    assert (final["kind"], final["task_attempt"], final["round_idx"], _has_mark(final["messages"])) == expected
    assert not fence_closed(read_fence(host.root, "held")), "the owner's Resume opened the root"
    assert not carried and not host.attempts


def test_acknowledged_return_of_a_root_with_a_closed_consumed_fence_waits_for_resume(
    host, tmp_path, monkeypatch,  # noqa: F811
):
    from ouroboros.owner_pause import fence_closed, read_fence

    _recovered_then_paused(host, tmp_path, monkeypatch, "project")
    _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "after_cas")
    assert _stop_boot(monkeypatch, returning=True) == 1
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and fence_closed(read_fence(host.root, "held")), "boot never releases a Pause by itself"
    assert queue.resume_budget_paused_task("held")["ok"]
    final = _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "after_resume")
    assert (final["kind"], final["round_idx"], _has_mark(final["messages"])) == ("working", 29, True)


# --- a member of a terminal root: its consumed grant's own fence selection -----------------------

def _member_resumed_until(tmp_path, monkeypatch, cut):
    """An owner-selected child of a terminal root consumes its grant; the process ends at ``cut``.

    The child is admitted, claimed and started by the real producers, so its row
    and result carry the host's own scope and lineage facts.
    """
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
    from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _mock_pause_observation
    from tests.test_budget_pause_lineage import CHILD, ROOT, _member_ctx, _park, _sup, _workers

    queue_mod, state, workers_mod = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, ROOT, STATUS_RUNNING, chat_id=0, root_task_id=ROOT)
    workers_mod.RUNNING[ROOT] = {"task": {"id": ROOT, "type": "task", "chat_id": 0, "root_task_id": ROOT},
                                 "worker_id": 0, "attempt": 1}
    admitted = queue_mod.enqueue_task({"id": CHILD, "type": "task", "chat_id": 0, "root_task_id": ROOT, "depth": 1,
                                       "parent_task_id": ROOT, "delegation_role": "subagent", "text": "child work"})
    assert admitted.get("_project_scope_none") is True, "fresh host admission attests no Project scope"
    started = _workers(workers_mod)
    workers_mod.assign_tasks()
    _start_record(tmp_path, started[0])
    assert request_owner_pause(ROOT, request_id="p")["ok"]
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    ctx, limit = _member_ctx(tmp_path)
    limit.messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": MARK}]
    pause = _park(limit)
    install_exact_budget_pause(_sup(tmp_path, queue_mod, workers_mod), CHILD,
                               budget_pause.exact_pause_marker(pause)["checkpoint"])
    workers_mod.RUNNING.pop(ROOT)
    write_task_result(tmp_path, ROOT, STATUS_COMPLETED, chat_id=0, root_task_id=ROOT)
    sent = _workers(workers_mod)
    assert queue_mod.resume_budget_paused_task(CHILD)["ok"]
    workers_mod.assign_tasks()
    ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    blob = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    real_reopen = budget_pause._reopen_owner_fence

    def reopen_then_lose(*args, **kwargs):
        real_reopen(*args, **kwargs)
        raise ProcessLost("selected on its fence, nothing saved since")

    with monkeypatch.context() as fault:
        if cut == "after_cas":
            fault.setattr(budget_pause, "_reopen_owner_fence", lambda *_a, **_k: (_ for _ in ()).throw(ProcessLost()))
        else:
            fault.setattr(budget_pause, "_reopen_owner_fence", reopen_then_lose)
        with pytest.raises(ProcessLost):
            budget_pause.resume_paused_loop(limit.tools, blob, [], {}, {}, set(), budget_remaining_usd=5.0)
    return ctx, blob["_pause_row"]["grant"]["grant_id"]


@pytest.mark.parametrize("returning", [False, True])
@pytest.mark.parametrize("cut", ["after_cas", "after_fence"])
def test_a_member_resume_consumed_before_its_fence_selection_completes_at_the_owner_resume(
    tmp_path, monkeypatch, cut, returning,
):
    from ouroboros.owner_pause import fence_closed, member_fence, read_fence
    from tests.test_budget_pause_lineage import CHILD, ROOT, _member_ctx, _workers

    _ctx, grant_id = _member_resumed_until(tmp_path, monkeypatch, cut)
    fence = read_fence(tmp_path, ROOT)
    assert fence_closed(fence) and (CHILD in (fence.get("selected_members") or {})) is (cut == "after_fence")
    assert _stop_boot(monkeypatch, returning=returning) == 1
    [row] = workers.PENDING
    assert row["_attempt"] == 2 and row["_working_recovery"]["boundary"] == "resumed"
    sent = _workers(workers)
    workers.assign_tasks()
    assert not sent, "the closed fence still refuses the saved member"
    answer = queue.resume_budget_paused_task(CHILD)
    assert answer["ok"], answer
    after = read_fence(tmp_path, ROOT)
    assert fence_closed(after) and after["fence_id"] == fence["fence_id"]
    assert after["selected_members"][CHILD]["grant_id"] == grant_id, "the consumed grant's own selection"
    sibling, _limit = _member_ctx(tmp_path, "lineage-sibling")
    assert fence_closed(member_fence(sibling)), "siblings stay fenced"
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD], "the owner's Resume reaches a worker, held or returned"
    seen = _run_loop(SimpleNamespace(root=tmp_path), monkeypatch, sent[0], "after_resume")
    restored = (seen["kind"], seen["task_attempt"], seen["round_idx"], seen["boundary"], _has_mark(seen["messages"]))
    assert restored == ("working", 1, 3, wc.RESUMED, True), "the member continues its exact handover"


@pytest.mark.parametrize("change", ["none", "root", "revoked", "other_attempt", "other_fence", "already_selected"])
def test_member_resume_completion_needs_its_own_consumed_grant_and_fence(tmp_path, monkeypatch, change):
    from ouroboros.owner_pause import consumed_member_resume, read_fence, set_fence_state
    from tests.test_budget_pause_lineage import CHILD, ROOT

    _member_resumed_until(tmp_path, monkeypatch, "after_cas")
    row, fence = budget_pause.budget_pause_row(tmp_path, CHILD), read_fence(tmp_path, ROOT)
    task_id, attempt = CHILD, 1
    if change == "root":
        task_id = ROOT
    elif change == "revoked":
        budget_pause.set_budget_pause(tmp_path, CHILD, {**row, "grant": {**row["grant"], "revoked_at": "now"}})
    elif change == "other_attempt":
        attempt = 2
    elif change == "other_fence":  # the closed fence is not the one the consumed grant names
        budget_pause.set_budget_pause(tmp_path, CHILD, {**row, "grant": {**row["grant"],
                                                                         "owner_pause_fence_id": "older-fence"}})
    elif change == "already_selected":
        set_fence_state(tmp_path, ROOT, fence_id=fence["fence_id"], state=fence["state"],
                        selected_members={CHILD: {"grant_id": row["grant"]["grant_id"]}})
    before = read_fence(tmp_path, ROOT)
    found = consumed_member_resume(tmp_path, ROOT, task_id, attempt=attempt)
    assert found == ({"fence_id": fence["fence_id"], "grant_id": row["grant"]["grant_id"]} if change == "none" else {})
    assert read_fence(tmp_path, ROOT) == before, "a read never changes the fence"


@pytest.mark.parametrize("returning", [False, True])
@pytest.mark.parametrize("refusal", ["stop", "restart_no_resume"])
def test_a_refused_owner_resume_never_completes_the_member_fence(tmp_path, monkeypatch, refusal, returning):
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_pause import read_fence
    from tests.test_budget_pause_lineage import CHILD, ROOT, _workers

    _member_resumed_until(tmp_path, monkeypatch, "after_cas")
    assert _stop_boot(monkeypatch, returning=returning) == 1
    if refusal == "stop":
        request_cancel(tmp_path, CHILD, reason="owner stop", source="owner", requested_by="owner")
    else:
        from tests._work_stop_helpers import live_restart
        live_restart(tmp_path, monkeypatch)
    before = read_fence(tmp_path, ROOT)
    answer = queue.resume_budget_paused_task(CHILD)
    assert not answer["ok"] and read_fence(tmp_path, ROOT) == before, answer
    sent = _workers(workers)
    workers.assign_tasks()
    assert not sent


@pytest.mark.parametrize("returning", [False, True])
def test_an_unpublished_member_selection_leaves_the_fence_and_the_next_resume_completes_it(
    tmp_path, monkeypatch, returning,
):
    from ouroboros.owner_pause import launch_lock, read_fence
    from tests.test_budget_pause_lineage import CHILD, ROOT, _workers

    _member_resumed_until(tmp_path, monkeypatch, "after_cas")
    assert _stop_boot(monkeypatch, returning=returning) == 1
    before = read_fence(tmp_path, ROOT)
    with monkeypatch.context() as fault:
        fault.setattr(queue, "persist_queue_snapshot", lambda reason="": False)
        assert queue.resume_budget_paused_task(CHILD)["error"] == "snapshot_not_persisted"
    assert read_fence(tmp_path, ROOT) == before, "an unpublished selection leaves the fence exactly as it was"
    with launch_lock(tmp_path, ROOT):  # another launch decision of the tree: never awaited under the queue lock
        started = time.monotonic()
        assert not queue.resume_budget_paused_task(CHILD)["ok"]
        assert time.monotonic() - started < 2.0
    assert read_fence(tmp_path, ROOT) == before
    sent = _workers(workers)
    workers.assign_tasks()
    assert not sent
    assert queue.resume_budget_paused_task(CHILD)["ok"], "the next explicit Resume completes it"
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD]


def _fail_member_release(monkeypatch, failure):
    """The fence release raises right after the queue selection was published; records that order."""
    from ouroboros import owner_pause

    published = []

    def fail_release(*_args, **_kwargs):
        published.append(any(row.get("_resume_selection") for row in workers.PENDING))
        raise failure

    monkeypatch.setattr(owner_pause, "select_member_resume", fail_release)
    return published


@pytest.mark.parametrize("fault", ["refused", "unwritable"])
def test_a_member_release_failing_after_its_queue_selection_stays_continuable(tmp_path, monkeypatch, fault):
    from ouroboros.owner_pause import OwnerPauseRefused, read_fence
    from tests.test_budget_pause_lineage import CHILD, ROOT, _workers

    _member_resumed_until(tmp_path, monkeypatch, "after_cas")
    assert _stop_boot(monkeypatch) == 1
    before = read_fence(tmp_path, ROOT)
    with monkeypatch.context() as patched:
        published = _fail_member_release(patched, OwnerPauseRefused("owner_pause_fence_changed")
                                         if fault == "refused" else OSError("read-only drive"))
        assert queue.resume_budget_paused_task(CHILD)["error"] == "selection_owner_held"
    assert published == [True], "the queue selection was published before the member's release"
    assert read_fence(tmp_path, ROOT) == before
    [row] = workers.PENDING
    assert budget_hold_fact(row), "the selection was withdrawn: the row is held, never selected over a closed fence"
    sent = _workers(workers)
    assert queue.resume_budget_paused_task(CHILD)["ok"], "the next explicit Resume completes it"
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD]


def test_a_process_lost_between_the_member_queue_selection_and_its_release_continues_after_boot(
    tmp_path, monkeypatch,
):
    from ouroboros.owner_pause import fence_closed, member_fence, read_fence
    from tests.test_budget_pause_lineage import CHILD, ROOT, _member_ctx, _workers

    _member_resumed_until(tmp_path, monkeypatch, "after_cas")
    assert _stop_boot(monkeypatch) == 1
    before = read_fence(tmp_path, ROOT)
    with monkeypatch.context() as patched:
        published = _fail_member_release(patched, ProcessLost())
        with pytest.raises(ProcessLost):
            queue.resume_budget_paused_task(CHILD)
    assert published == [True] and read_fence(tmp_path, ROOT) == before
    child, _limit = _member_ctx(tmp_path)
    assert fence_closed(member_fence(child)), "launch authority stays closed without the member's release"
    sent = _workers(workers)
    workers.assign_tasks()
    assert not sent, "a published selection alone starts nothing"
    assert _stop_boot(monkeypatch) == 1
    [row] = workers.PENDING
    assert budget_hold_fact(row)["reason"] == "saved_work_hold", "the host-unscoped row is held again for Resume"
    assert queue.resume_budget_paused_task(CHILD)["ok"], "the next explicit Resume completes it"
    sent = _workers(workers)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD] and sent[0]["_working_recovery"]["boundary"] == wc.RESUMED


# --- a handover that cannot be written, or a grant that is no longer live ------------------------

def _granted(tmp_path, monkeypatch, task_id):
    from tests.test_budget_pause_exact_resume import _granted_loop

    _module, ctx, state_blob, row = _granted_loop(tmp_path, monkeypatch, task_id)
    return ctx, state_blob, row


def test_an_unwritable_handover_holds_the_unconsumed_grant_then_consumes_once(tmp_path, monkeypatch):
    ctx, state_blob, _row = _granted(tmp_path, monkeypatch, "handover-hold")
    real, attempts, published = wc.save_resumed, [], []

    def flaky(*args, **kwargs):
        attempts.append(budget_pause.budget_pause_row(tmp_path, "handover-hold")["state"])
        if len(attempts) < 3:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(wc, "save_resumed", flaky)
    monkeypatch.setattr(budget_pause, "_publish_hold", lambda _ctx, hold: published.append(dict(hold)))
    acked = []
    ctx._pending_content_acks = [lambda: acked.append(1)]
    messages, usage = [], {}
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, messages, {}, usage, set(),
                                    budget_remaining_usd=5.0)
    assert attempts == [budget_pause.STATE_RESUME_GRANTED] * 3, "nothing consumed before the handover landed"
    assert [hold["hold_reason"] for hold in published] == [wc.HOLD_UNWRITABLE]
    assert published[0]["state"] == budget_pause.STATE_RESUME_GRANTED
    consumed = budget_pause.budget_pause_row(tmp_path, "handover-hold")
    assert consumed["state"] == budget_pause.STATE_RESUMED and "budget_pause_hold" not in usage
    saved = wc.read_working(tmp_path, "handover-hold", 1)[1]
    assert saved["working"]["boundary"] == wc.RESUMED and saved["messages"] == messages
    assert [row.get("tool_call_id") for row in messages if row.get("role") == "tool"] == ["call_a", "call_b"]
    assert sum("continued from its budget pause" in str(row.get("content")) for row in messages) == 1
    assert saved["usage"]["budget_pause_resume"]["paused_duration_sec"] == consumed["paused_duration_sec"]
    assert not acked and len(ctx._pending_content_acks) == 1, "the handover acknowledges no mail"


@pytest.mark.parametrize("control", ["cancelled", "deadline"])
def test_a_handover_hold_ended_by_control_consumes_and_starts_nothing(tmp_path, monkeypatch, control):
    from ouroboros.model_wait import ModelWaitInterrupted
    from tests._budget_pause_exact_helpers import _controls

    ctx, state_blob, row = _granted(tmp_path, monkeypatch, "handover-stop")
    monkeypatch.setattr(wc, "save_resumed", lambda *_a, **_k: (_ for _ in ()).throw(OSError("read-only drive")))
    monkeypatch.setattr(budget_pause, "_hold_control_reason", _controls("", control))
    usage = {}
    with pytest.raises(ModelWaitInterrupted) as raised:
        budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, [], {}, usage, set(),
                                        budget_remaining_usd=5.0)
    assert raised.value.control_reason == control
    durable = budget_pause.budget_pause_row(tmp_path, "handover-stop")
    assert durable["state"] == budget_pause.STATE_RESUME_GRANTED and not durable["grant"].get("consumed_at")
    assert durable["source_ref"] == row["source_ref"] and not wc.checkpoint_path(tmp_path, "handover-stop", 1).exists()
    hold = usage["budget_pause_hold"]
    assert (hold["hold_reason"], hold["ended_by"]) == (wc.HOLD_UNWRITABLE, control)


def test_a_failed_handover_in_the_real_loop_runs_no_model_or_tool(host, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros.model_wait import ModelWaitInterrupted

    _recovered_then_paused(host, tmp_path, monkeypatch, "main")
    task = _dispatch_one(host, monkeypatch)
    registry = _registry(host.root, monkeypatch, "held", 2)
    registry._ctx.model_wait_context = None
    agent._restore_saved_clocks(registry._ctx, task)
    monkeypatch.setattr(wc, "save_resumed", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "cancelled")
    monkeypatch.setattr(loop, "_dispatch_round_model",
                        lambda *a, **k: pytest.fail("model call after a failed handover"))
    for name in ("execute", "execute_result"):
        monkeypatch.setattr(registry, name, lambda *a, **k: pytest.fail("tool call after a failed handover"))
    kwargs = _loop_kwargs(host.root, registry, [])
    kwargs["task_id"] = "held"
    with pytest.raises(ModelWaitInterrupted):
        loop.run_llm_loop(**kwargs)
    assert budget_pause.budget_pause_row(host.root, "held")["state"] == budget_pause.STATE_RESUME_GRANTED


def test_a_grant_revoked_before_consumption_reparks_and_retires_the_handover(tmp_path, monkeypatch):
    ctx, state_blob, row = _granted(tmp_path, monkeypatch, "handover-revoked")
    real = wc.save_resumed

    def revoked_after_write(*args, **kwargs):
        real(*args, **kwargs)
        live = budget_pause.budget_pause_row(tmp_path, "handover-revoked")
        budget_pause.set_budget_pause(tmp_path, "handover-revoked", {
            **live, "state": budget_pause.STATE_PAUSED, "grant": {**live["grant"], "revoked_at": time.time()}})

    monkeypatch.setattr(wc, "save_resumed", revoked_after_write)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, [], {}, {}, set(),
                                        budget_remaining_usd=5.0)
    assert raised.value.pause["pause_id"] == row["pause_id"]
    assert wc.checkpoint_path(tmp_path, "handover-revoked", 1).exists()
    agent._retire_working_checkpoint(ctx, {"id": "handover-revoked", "_attempt": 1})
    assert not wc.checkpoint_path(tmp_path, "handover-revoked", 1).exists(), "the live pause supersedes it"


# --- the exact program counter, route, last-fit permission and clocks survive the handover -------

@pytest.mark.parametrize("tail", ["tool", "no_tool"])
def test_the_handover_rejoins_the_exact_budget_tail_before_any_model_call(
    host, tmp_path, monkeypatch, tail,  # noqa: F811
):
    _recovered_then_paused(host, tmp_path, monkeypatch, "project", tail=tail)
    exact = _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "finisher")
    assert _stop_boot(monkeypatch) == 1
    assert queue.resume_budget_paused_task("held")["ok"]
    handed = _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "finisher")
    assert (exact["kind"], handed["kind"], handed["boundary"]) == ("pause", "working", wc.RESUMED)
    assert handed["tails"] == exact["tails"] == ((True, False, False) if tail == "tool" else (False, True, True))
    finisher = "_finish_tool_round_budget" if tail == "tool" else "_finish_no_tool_round_budget"
    assert handed["finisher"] == exact["finisher"] == (finisher, 29)


@pytest.mark.parametrize("rail", [budget_pause.RAIL_GRACEFUL_CEILING, budget_pause.RAIL_GLOBAL_EXHAUSTED])
def test_last_fit_permission_original_start_and_paused_time_cross_the_handover(
    host, tmp_path, monkeypatch, rail,  # noqa: F811
):
    pause = _recovered_then_paused(host, tmp_path, monkeypatch, "main", rail=rail)
    first = _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "after_resume", waiter=True)
    consumed = budget_pause.budget_pause_row(host.root, "held")
    paused = consumed["paused_duration_sec"]
    assert paused == consumed["grant"]["paused_duration_sec"] > 0, "no second interval adder"
    saved = wc.read_working(host.root, "held", 2)[1]
    assert saved["started_at"] == pause["started_at"] == first["ctx"].task_started_at
    assert saved["model_wait"]["budget_paused_sec"] == paused
    assert saved["usage"]["budget_pause_resume"]["last_fit_relaxed"] is (rail in budget_pause.GRACEFUL_RAILS)
    assert _stop_boot(monkeypatch) == 1
    assert queue.resume_budget_paused_task("held")["ok"]
    handed = _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "after_resume", waiter=True)
    ctx = handed["ctx"]
    assert ctx._budget_resume_last_fit_relaxed is first["ctx"]._budget_resume_last_fit_relaxed
    assert ctx._budget_resume_last_fit_relaxed is (rail in budget_pause.GRACEFUL_RAILS)
    assert ctx.task_started_at == pause["started_at"]
    assert ctx._budget_paused_sec == ctx.model_wait_context.budget_paused_sec == paused


@pytest.mark.parametrize(("role", "route"), [
    ("fallback:0", ("fallback:0", "account-b")), ("main", ("main", "account-a")),
    ("", ("main", "account-a")),  # no recorded role: each worker derives it from its own plan
])
def test_the_handover_keeps_the_route_role_its_pause_ran_on(host, tmp_path, monkeypatch, role, route):  # noqa: F811
    """Main and fallback share one model on different accounts; every new worker starts on Main."""
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b"]}))
    fresh = replace(_plan(), model="claudexor::codex=route-fixture-model")
    _recovered_then_paused(host, tmp_path, monkeypatch, "main", plan=replace(fresh, model_role=role))
    first = _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "after_resume", plan=fresh)
    saved = wc.read_working(host.root, "held", 2)[1]
    assert _stop_boot(monkeypatch) == 1
    assert queue.resume_budget_paused_task("held")["ok"]
    handed = _run_loop(host, monkeypatch, _dispatch_one(host, monkeypatch), "after_resume", plan=fresh)
    plans = [seen["ctx"].context_fit_plan for seen in (first, handed)]
    assert [(plan.model_role, plan.model_route["credentialProfileId"]) for plan in plans] == [route, route]
    assert (handed["kind"], handed["boundary"], saved["context_model_role"]) == ("working", wc.RESUMED, role)


def test_a_sleep_resume_folds_grant_to_consumption_once_into_the_handover(tmp_path, monkeypatch):
    from ouroboros.model_wait import TaskModelWait

    ctx, state_blob, _row = _granted(tmp_path, monkeypatch, "handover-sleep")
    row = state_blob["_pause_row"]
    granted_at = time.time() - 50.0
    state_blob["_pause_row"] = {**row, "reason": budget_pause.REASON_SLEEP,
                                "grant": {**row["grant"], "granted_at_ts": granted_at, "paused_duration_sec": 10.0}}
    ctx.model_wait_context = TaskModelWait(task={"id": "handover-sleep", "_budget_pause_resume": {
        "paused_duration_sec": 10.0, "sleep_exclusion_since": granted_at}}, drive_root=tmp_path,
        event_queue=None, worker_slot_held=False)
    assert ctx.model_wait_context.sleep_started_monotonic is not None
    ctx.model_wait_context.clocks[""].elapsed, ctx.model_wait_context.revision = 37.0, 3  # confirmed quota waits
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, [], {}, {}, set(), budget_remaining_usd=5.0)
    consumed = budget_pause.budget_pause_row(tmp_path, "handover-sleep")
    folded = 10.0 + consumed["grant"]["consumed_at"] - granted_at
    assert consumed["paused_duration_sec"] == pytest.approx(folded, abs=1e-6)
    saved = wc.read_working(tmp_path, "handover-sleep", 1)[1]
    assert saved["model_wait"]["budget_paused_sec"] == consumed["paused_duration_sec"]
    assert ctx.model_wait_context.budget_paused_sec == consumed["paused_duration_sec"]
    assert ctx.model_wait_context.sleep_started_monotonic is None, "the sleep interval is subtracted once"
    # The quota-wait union rides the same handover; recovery restores both carriers, adding neither again.
    handoff = wc.prepare_recovery(tmp_path, "handover-sleep", from_attempt=1, cause="app_stop")
    assert handoff["model_wait_quota_clock"]["revision"] == 3 and handoff["model_wait_quota_clock"]["elapsed_sec"] >= 37
    restored = TaskModelWait(task={"id": "handover-sleep", "_attempt": 2}, drive_root=tmp_path, event_queue=None,
                             worker_slot_held=False)
    restored.restore_continuation(saved["model_wait"], started_at=saved["started_at"])
    assert restored.budget_paused_sec == consumed["paused_duration_sec"] and restored.revision == 3
    assert restored.clocks[""].duration(time.monotonic()) == pytest.approx(37.0, abs=1.0)


def test_a_retained_delivery_candidate_crosses_the_handover_without_a_finalization_pass(tmp_path, monkeypatch):
    from ouroboros.loop_delivery import DeliveryCandidate, resume_delivery_candidate
    from tests import _budget_pause_exact_helpers as helpers

    original = helpers._loop_ctx

    def with_candidate(*args, **kwargs):
        context, limits = original(*args, **kwargs)
        context._delivery_candidate = DeliveryCandidate(
            full_text="draft answer", content_sha256="0" * 64, revision=2, evidence_revision=1,
            evidence_fingerprint="fp", acceptance_binding={})
        return context, limits

    monkeypatch.setattr(helpers, "_loop_ctx", with_candidate)
    ctx, state_blob, _row = _granted(tmp_path, monkeypatch, "handover-candidate")
    assert state_blob["delivery_candidate"]["full_text"] == "draft answer"
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, [], {}, {}, set(), budget_remaining_usd=5.0)
    saved = wc.read_working(tmp_path, "handover-candidate", 1)[1]
    assert saved["delivery_candidate"]["full_text"] == "draft answer"
    assert not resume_delivery_candidate(SimpleNamespace(_ctx=ctx), saved, []), "exact Resume starts no finalization"


def test_a_newer_pause_during_the_handover_stays_closed_through_consumption_stop_and_boot(
    host, tmp_path, monkeypatch,  # noqa: F811
):
    from ouroboros.owner_pause import fence_closed, read_fence

    _recovered_then_paused(host, tmp_path, monkeypatch, "project")
    resumed = _dispatch_one(host, monkeypatch)
    granted = read_fence(host.root, "held")
    assert request_owner_pause("held", request_id="pause-again")["ok"]
    newer = read_fence(host.root, "held")
    assert newer["fence_id"] != granted["fence_id"] and newer["supersedes_fence_id"] == granted["fence_id"]
    seen = _run_loop(host, monkeypatch, resumed, "after_resume")
    assert seen["round_idx"] == 29 and budget_pause.budget_pause_row(host.root, "held")["state"] == "resumed"
    assert read_fence(host.root, "held") == newer, "the consumed grant releases nothing it did not name"
    assert _stop_boot(monkeypatch) == 1
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and fence_closed(read_fence(host.root, "held")), "boot never releases the newer Pause"
    [row] = host.pending
    assert row["_working_recovery"]["boundary"] == wc.RESUMED and budget_hold_fact(row)


def test_a_held_sleep_handover_keeps_its_live_exclusion_for_the_task_controls(tmp_path, monkeypatch):
    import time as real_time

    from ouroboros import config, model_wait

    real_control = budget_pause._hold_control_reason
    ctx, state_blob, _row = _granted(tmp_path, monkeypatch, "handover-sleep-hold")
    monkeypatch.setattr(budget_pause, "_hold_control_reason", real_control)  # the task's real controls end a hold
    now, row = real_time.time(), state_blob["_pause_row"]
    state_blob["_pause_row"] = {**row, "reason": budget_pause.REASON_SLEEP, "grant": {
        **row["grant"], "granted_at_ts": now - 60.0, "paused_duration_sec": 900.0}}
    skew = {"sec": 0.0}  # wall and monotonic clocks advanced together by a long unwritable spell
    clock = SimpleNamespace(time=lambda: real_time.time() + skew["sec"], sleep=real_time.sleep,
                            monotonic=lambda: real_time.monotonic() + skew["sec"], perf_counter=real_time.perf_counter)
    monkeypatch.setattr(model_wait, "time", clock)
    monkeypatch.setattr(budget_pause, "time", clock)
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 100.0)
    real_save, attempts = wc.save_resumed, []

    def unwritable_for_a_while(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            skew["sec"] += 500.0
            raise OSError("disk full")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(wc, "save_resumed", unwritable_for_a_while)
    task = {"id": "handover-sleep-hold", "_attempt": 1, "_budget_pause_resume": {
        "paused_duration_sec": 900.0, "sleep_exclusion_since": now - 60.0}}
    with model_wait.task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None,
                                          worker_slot_held=False) as waiter:
        waiter.restore_continuation({}, started_at=now - 1000.0, budget_paused_sec=900.0)
        ctx.model_wait_context = waiter
        assert waiter.executed_seconds() == pytest.approx(40.0, abs=1.0)
        budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, [], {}, {}, set(),
                                        budget_remaining_usd=5.0)
        assert len(attempts) == 2, "held through the spell, never ended as an exhausted lifetime"
        consumed = budget_pause.budget_pause_row(tmp_path, "handover-sleep-hold")
        assert consumed["paused_duration_sec"] == pytest.approx(900.0 + consumed["grant"]["consumed_at"] - (now - 60.0))
        assert waiter.budget_paused_sec == consumed["paused_duration_sec"] and waiter.sleep_started_monotonic is None
        assert waiter.executed_seconds() == pytest.approx(40.0, abs=1.0), "the held spell was paused time, once"
