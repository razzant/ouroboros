"""#1729 / #1730: the stored work of a historical idle-timeout retry keeps Pause, Resume and saved work.

No runtime producer kills silent work or mints a retry now (#1376); the reaper only ends hard
bounds. A pre-upgrade retry still lives in stored results and restored queue rows: a new-ID
successor of a root, a same-id successor of a child, or the same id held as pause intent over a
seed that saved nothing (``tests._silence_reaper_support``). The real readers run: API admission
and assignment, saved work, the owner Pause through the agent handler
(``enter_owner_pause``/``request_pause``), the Resume grant and its consumption, assignment
(``_claim_worker_launch``/``launch_admission``) and the actual worker loader or a real
tool-registry launch. Doubled: OS processes, model calls and the worker queue's ``put``.

The contract (owner 8 Oct quiz 1b1a2d93, 13 Oct G4 release), for that stored work:
- pause intent never dispatches by itself; the owner's Resume starts only what was saved, or the
  root's request, and refuses over an effect of unknown outcome; a member names cancel or a new run;
- a Pause on the visible retry is its LOGICAL root's tree Pause: every launch gate agrees, the
  exact grant stays on the physical row, one owner Resume releases it, siblings are not selected;
- only the current attempt of a retry chain acts for its root.
"""
from __future__ import annotations

import queue as queue_lib
import queue as stdqueue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import agent, budget_pause
from ouroboros import working_checkpoint as wc
from ouroboros.owner_pause import OwnerPauseRefused, launch_admission, member_fence, read_fence, root_actor
from ouroboros.task_results import load_task_result
from supervisor import queue, workers
from supervisor.events_budget import budget_hold_fact, install_exact_budget_pause
from supervisor.owner_pause_control import request_owner_pause
from tests import test_execution_clock_resume_handover as handover
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_execution_clock_resume_handover import (  # noqa: F401
    MARK,
    ProcessLost,
    _admit,
    _attempt_ctx,
    _dispatch_one,
    _fresh_plan,
    _granted_after_capacity,
    _loop_once,
    _save_first_attempt,
    _supervisor_events,
    _working_then_paused,
    clock,
)
from tests.test_project_hold_recovery import worker
from tests.test_resume_handover import _DeadProcess as _HandoverProcess
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.worker_pool_support import affirm_ready
from tests._silence_reaper_support import historical_timeout_retry, historical_unsaved_pause_hold

pytestmark = pytest.mark.serial


class _WorkerDied(BaseException):
    """The worker is lost after its durable pausing seed and before its continuation source."""


class _DeadProcess(_HandoverProcess):
    """The public pool's death proof includes the Process exit code."""
    @property
    def exitcode(self):
        return None if self.alive else -9


@pytest.fixture
def reaper(host, monkeypatch):  # noqa: F811
    """The reaper's OS doubles; the queue's admission latch keeps its production identity."""
    # ``queue.init`` clears ONE latch map in place that queue and task_lifecycle share; the shared
    # host fixture rebinds only the queue's name, which would hide the enqueue latch.
    monkeypatch.setattr("supervisor.task_lifecycle.BUDGET_ROOT_FENCES", queue.BUDGET_ROOT_FENCES)
    monkeypatch.setattr("supervisor.task_lifecycle.CANCELLED_ROOT_FENCES", {})
    real_metadata = handover._metadata
    monkeypatch.setattr(handover, "_metadata", lambda task: {**real_metadata(task), **{
        key: task[key] for key in ("root_task_id", "original_task_id", "timeout_retry_from") if task.get(key)}})
    jobs = stdqueue.Queue()
    monkeypatch.setattr(queue, "_reap_queue", jobs)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(workers, "respawn_worker", lambda *_a, **_k: None)
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 3)
    monkeypatch.setattr(queue, "QUEUE_MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(queue, "reconstruct_task_cost", lambda *_a, **_k: {
        "cost_accounting_status": "available", "cost_final": True})

    def arm():
        """The busy slot's process becomes a double the reaper's kill marks dead."""
        process = _DeadProcess()
        workers.WORKERS[0].proc = process
        monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree", lambda pid, **_k: process.kill(pid))
        return jobs

    return SimpleNamespace(arm=arm, jobs=jobs)


def _successors(host):  # noqa: F811
    return [t for t in host.pending if t.get("timeout_retry_from") and t.get("id") != t.get("timeout_retry_from")]


def _assign(host, monkeypatch):  # noqa: F811
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    return sent


def _consumed(host, monkeypatch, clock):  # noqa: F811
    """Attempt 2 was Paused by the owner, Resumed, and its loop consumed the grant."""
    fresh = _fresh_plan(monkeypatch)
    pause = _working_then_paused(host, monkeypatch, clock, None, fresh)
    resumed = _granted_after_capacity(host, monkeypatch, clock, pause)
    return fresh, pause, resumed


def _handler_pause(host, ctx, limit, task, *, request_id, rail="owner", first_boundary=False):  # noqa: F811
    """A real boundary meets a Pause; the agent handler writes the exact pause and the
    supervisor's park never runs (its event is lost with the stalled worker). ``first_boundary``:
    the Pause meets the attempt before its first tool batch, so it never produced a checkpoint."""
    if first_boundary:
        limit.round_idx, limit.budget_tail = 0, "no_tool"
        limit.messages = [{"role": "user", "content": "do it"}]
    else:
        limit.round_idx, limit.budget_tail = 31, "tool"
        limit.messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": MARK}]
        wc.save_round(limit, "post_batch")
    if rail != "owner":
        with pytest.raises(budget_pause.BudgetPauseRequested):
            budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                       scope=rail, reason_text="money gone", root_task_id="held")
        budget_pause.end_dispatch_fence("held")
        return budget_pause.budget_pause_row(host.root, "held")
    assert request_owner_pause("held", request_id=request_id)["ok"]
    owner = object.__new__(agent.OuroborosAgent)
    owner.env = agent.Env(repo_dir=host.root, drive_root=host.root)
    owner.tools = SimpleNamespace(_ctx=ctx)
    owner._owner_message_admission_lock = threading.RLock()
    owner._incoming_messages = queue_lib.Queue()
    owner._event_queue = None
    owner._emit_live_log = lambda *a, **k: None
    owner._start_task_heartbeat_loop = lambda *a: None

    def prepare(*_args):
        budget_pause.enter_owner_pause(limit)
        pytest.fail("the owner pause did not park")

    owner._prepare_task_context = prepare
    owner._handle_task_scoped(task)
    budget_pause.end_dispatch_fence("held")
    return budget_pause.budget_pause_row(host.root, "held")


# --- B4: a historical pause intent over a seed that saved nothing --------------------------------

def _pause_at_a_boundary(host, monkeypatch, fresh, ctx, limit, task, rail, *, seedless,  # noqa: F811
                         first_boundary=False):
    """The handler's Pause; ``seedless``: the worker is lost after the durable seed, before its source."""
    ctx.active_model, ctx.context_fit_plan = fresh.model, fresh
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.0)
    with monkeypatch.context() as fault:
        if not seedless:
            _handler_pause(host, ctx, limit, task, request_id=f"pause-{rail}", rail=rail)
            return budget_pause.budget_pause_row(host.root, "held")

        def die(*_a, **_k):
            raise _WorkerDied

        fault.setattr(budget_pause, "_exact_continuation_row", die)
        with pytest.raises(_WorkerDied):
            _handler_pause(host, ctx, limit, task, request_id=f"pause-{rail}", rail=rail,
                           first_boundary=first_boundary)
        budget_pause.end_dispatch_fence("held")
    return budget_pause.budget_pause_row(host.root, "held")


def _first_attempt_seed(host, monkeypatch, rail, *, first_boundary=False, before_pause=None):  # noqa: F811
    fresh = _fresh_plan(monkeypatch)
    _admit(host, None)
    task = _dispatch_one(host, monkeypatch)
    ctx, limit, _waiter = _attempt_ctx(host, task, monkeypatch, attempt=1)
    if before_pause is not None:
        before_pause()
    return fresh, _pause_at_a_boundary(host, monkeypatch, fresh, ctx, limit, task, rail, seedless=True,
                                       first_boundary=first_boundary)


def _latch_pending_pauses(host, monkeypatch):  # noqa: F811
    """Every owner Pause here meets the root's busy launch lock: the real public ``latch_pending``
    answer leaves only the durable fence, no queue latch (another thread holds the real lock)."""
    from tests.test_acceptance_fence_cold_resume import _pause_with_latch_pending

    monkeypatch.setattr(f"{__name__}.request_owner_pause", lambda task_id, request_id: _pause_with_latch_pending(
        host, monkeypatch, task_id, request_id))


@pytest.mark.parametrize("latch", ["latched", "latch_pending"])
def test_a_historical_pause_intent_over_an_unsaved_seed_waits_for_the_owners_resume(  # noqa: F811
        host, monkeypatch, clock, reaper, latch):  # noqa: F811
    """``latch_pending``: the owner's Resume completes the missing latch from the closed fence its
    held-selection observation revalidated, as a replay of that Pause would, and releases the root
    alone; a child admitted meanwhile waits for explicit selection (Q9)."""
    from supervisor.events_budget import HOLD_UNSAVED_PAUSE

    if latch == "latch_pending":
        _latch_pending_pauses(host, monkeypatch)

    fresh, seed = _first_attempt_seed(host, monkeypatch, "owner", first_boundary=True)
    assert not wc.checkpoint_path(host.root, "held", 1).exists(), "its first boundary came before any save"
    assert ("held" in queue.BUDGET_ROOT_FENCES) is (latch == "latched")
    historical_unsaved_pause_hold(clock)
    assert not _successors(host) and "held" not in workers.RUNNING
    [held] = [t for t in host.pending if t["id"] == "held"]
    hold = budget_hold_fact(held)
    assert hold["reason"] == HOLD_UNSAVED_PAUSE and hold["pause_id"] == seed["pause_id"]
    assert hold["detail"].endswith("the owner's Resume starts it again from its request"), hold
    assert hold["pause_generation"] == seed["pause_generation"] and held["_attempt"] == 2
    assert "_budget_pause" not in held and "_working_recovery" not in held
    row = budget_pause.budget_pause_row(host.root, "held")
    assert {k: row.get(k) for k in ("state", "pause_id", "pause_generation", "task_attempt", "source_ref")} == {
        k: seed.get(k) for k in ("state", "pause_id", "pause_generation", "task_attempt", "source_ref")}
    stored = load_task_result(host.root, "held")
    assert (stored["status"], stored["reason_code"]) == ("scheduled", "owner_paused")
    # A child a delegate event drains now: the standing latch refuses it; without one it is admitted.
    admitted = queue.enqueue_task({"id": "child", "type": "task", "chat_id": 0, "text": "child work",
                                   "root_task_id": "held", "parent_task_id": "held", "depth": 1,
                                   "delegation_role": "subagent"})
    assert (admitted.get("_admission_blocked") == "root_owner_paused") is (latch == "latched"), admitted
    assert not _assign(host, monkeypatch), "pause intent never dispatches by itself"
    resumed = queue.resume_budget_paused_task("held")
    # The owner's own Resume of its Pause authorizes the next attempt, as for its held retry.
    assert resumed["ok"] and read_fence(host.root, "held")["state"] == "released", resumed
    assert "held" not in queue.BUDGET_ROOT_FENCES
    for queued_child in [t for t in host.pending if t["id"] == "child"]:
        assert budget_hold_fact(queued_child)["reason"] == "root_fence_lifted_pending_selection"
    sent = _assign(host, monkeypatch)
    assert [(t["id"], t["_attempt"]) for t in sent] == [("held", 2)], "Root Resume selects the root alone"
    started = _loop_once(host, monkeypatch, sent[0], plan=fresh)
    assert (started["kind"], started["round_idx"]) == ("none", None), "nothing saved, nothing fabricated"
    assert started["messages"][0]["content"] == "go", "the original prompt starts it again"


@pytest.mark.parametrize("latch", ["latched", "latch_pending"])
def test_an_unsaved_owner_intent_over_an_effect_of_unknown_outcome_stays_held(host, monkeypatch, clock,  # noqa: F811
                                                                             reaper, latch):
    """No checkpoint is not proof of no effect: the dead attempt's handed operation still holds it."""
    from ouroboros.owner_pause import operation_start, tool_handoff
    from supervisor.events_budget import HOLD_UNSAVED_PAUSE

    if latch == "latch_pending":
        _latch_pending_pauses(host, monkeypatch)

    def handed_and_lost():  # the attempt handed an operation whose outcome is unknown, then saved nothing
        source = SimpleNamespace(task_id="held", root_task_id="held", drive_root=host.root)
        with pytest.raises(TimeoutError):
            with tool_handoff(source, "external_write"):
                with operation_start(source):
                    raise TimeoutError("request sent, response unknown")

    _first_attempt_seed(host, monkeypatch, "owner", first_boundary=True, before_pause=handed_and_lost)
    historical_unsaved_pause_hold(clock)
    [held] = host.pending
    assert budget_hold_fact(held)["reason"] == HOLD_UNSAVED_PAUSE
    refused = queue.resume_budget_paused_task("held")
    assert refused["error"] == "owner_pause_effects_unsettled", refused
    assert [blocker["kind"] for blocker in refused["blockers"]] == ["tool_handoff"]
    assert not _assign(host, monkeypatch), "the original request is not started over an unknown effect"
    assert budget_hold_fact(held)["reason"] == HOLD_UNSAVED_PAUSE and read_fence(host.root, "held")
    # The refused Resume leaves the accepted Pause's latch complete, as that Pause's own replay would.
    assert queue.BUDGET_ROOT_FENCES["held"]["fence_id"] == read_fence(host.root, "held")["fence_id"]


def test_a_resume_refusal_past_twenty_unsettled_effects_carries_them_all(host, monkeypatch, clock,  # noqa: F811
                                                                         reaper):
    """The public Resume of an unsaved pause intent whose attempt left 21 delegated runs it started
    and an operation of unknown outcome refuses with every one of them, nothing cut at twenty: the
    owner's view reduces them to a few facts and a true remainder (web ``heldWorkFacts``)."""
    from ouroboros import delegate_custody as dc
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.owner_pause import operation_start, tool_handoff
    from tests.test_continue_blocker_diagnostics import _http

    def started_then_lost():
        for n in range(21):
            assert dc.record_started(host.root, dc.RunCustody(
                run_id=f"run-{n:02d}", task_id="held", route_id="codex", model="leaf-model",
                snapshot_id=f"snap-{n:02d}", authority_source="active_workspace", ledger_root=str(host.root)))
        dc._CUSTODY.clear()
        source = SimpleNamespace(task_id="held", root_task_id="held", drive_root=host.root)
        with pytest.raises(TimeoutError):
            with tool_handoff(source, "external_write"):
                with operation_start(source):
                    raise TimeoutError("request sent, response unknown")

    _first_attempt_seed(host, monkeypatch, "owner", first_boundary=True, before_pause=started_then_lost)
    historical_unsaved_pause_hold(clock)
    status, refusal = _http(api_task_resume, "held")
    kinds = [b["kind"] for b in refusal.get("blockers") or []]
    assert status == 409 and refusal["error"] == "owner_pause_effects_unsettled", refusal
    assert len(kinds) > 20 and kinds.count("delegated_run") == 21 and "tool_handoff" in kinds, kinds
    assert not _assign(host, monkeypatch)


# --- B5: an owner Pause over a hung attempt -------------------------------------------------------

def _hung_first_attempt(host, monkeypatch, clock):  # noqa: F811
    _admit(host, None)
    first = _dispatch_one(host, monkeypatch)
    _save_first_attempt(host, first, monkeypatch, clock)
    workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id="held", reaping=False, active_capacity=True)
    return first


def test_an_owner_paused_hung_root_keeps_one_held_retry_until_one_resume(host, monkeypatch, clock,  # noqa: F811
                                                                        reaper):
    from supervisor.events_budget import observe_held_budget_selection
    from supervisor.queue_transitions import _resume_unstarted_owner_paused_root

    _hung_first_attempt(host, monkeypatch, clock)
    first = request_owner_pause("held", request_id="hung")
    assert first["ok"] and first["root_task_id"] == "held"
    historical_timeout_retry(clock)  # stored before the upgrade, under the Pause it was taken in
    [retry] = _successors(host)
    assert queue.BUDGET_ROOT_FENCES["held"]["cause"] == "owner_pause"
    # Preservation, not dispatch: assignment skips it and its own launch gate reads the closed fence.
    assert not _assign(host, monkeypatch)
    with pytest.raises(OwnerPauseRefused):
        with launch_admission(SimpleNamespace(task_id=retry["id"], root_task_id="held", budget_drive_root=host.root)):
            pytest.fail("a launch under the accepted Pause")
    stale = observe_held_budget_selection(queue, retry["id"])
    resumed = queue.resume_budget_paused_task(retry["id"])
    assert resumed["ok"] and resumed["owner_pause_released"] and resumed["root_task_id"] == "held"
    assert read_fence(host.root, "held")["state"] == "released"
    assert not queue.resume_budget_paused_task(retry["id"])["ok"], "a replayed Resume is not a second release"
    # A newer Pause before its dispatch holds it again; the stale observation cannot release it.
    assert request_owner_pause(retry["id"], request_id="newer")["root_task_id"] == "held"
    assert not _assign(host, monkeypatch)
    with queue._queue_lock:
        row = next(t for t in host.pending if t["id"] == retry["id"])
        stale_resume = _resume_unstarted_owner_paused_root(
            queue, row, queue.BUDGET_ROOT_FENCES["held"], {**stale, "owner_pause_blockers": []})
    assert stale_resume == {"ok": False, "error": "selection_authority_changed"}
    assert queue.resume_budget_paused_task(retry["id"])["ok"]
    assert [t["id"] for t in _assign(host, monkeypatch)] == [retry["id"]]


# --- #1730: the Pause of a new-ID retry is its logical root's Pause ---------------------------------

def _ordinary_retry(host, monkeypatch, clock, reaper):  # noqa: F811
    """A root's historical new-ID retry; ``reaper`` keeps the shared latch identity its readers need."""
    _hung_first_attempt(host, monkeypatch, clock)
    historical_timeout_retry(clock)
    [retry] = _successors(host)
    return retry


def test_pause_of_a_queued_retry_fences_its_logical_root_and_one_resume_releases_it(host, monkeypatch, clock,  # noqa: F811
                                                                                   reaper):
    retry = _ordinary_retry(host, monkeypatch, clock, reaper)
    assert root_actor(host.root, "held", retry["id"]) and not root_actor(host.root, "held", "held-child")
    accepted = request_owner_pause(retry["id"], request_id="retry-pause")
    assert accepted["ok"] and (accepted["task_id"], accepted["root_task_id"]) == (retry["id"], "held")
    assert sorted(queue.BUDGET_ROOT_FENCES) == ["held"] and read_fence(host.root, "held")["state"] == "paused"
    assert not read_fence(host.root, retry["id"]), "no fence a launch gate would never read"
    assert not _assign(host, monkeypatch)
    resumed = queue.resume_budget_paused_task(retry["id"])
    assert resumed["ok"] and resumed["root_task_id"] == "held"
    assert [t["id"] for t in _assign(host, monkeypatch)] == [retry["id"]]


def test_pause_of_a_running_retry_stops_its_new_effects_and_one_resume_reopens_the_tree(  # noqa: F811
        host, monkeypatch, clock, reaper):  # noqa: F811
    from concurrent.futures import ThreadPoolExecutor
    from ouroboros.owner_pause import review_episode, submit_tool
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    fresh = _fresh_plan(monkeypatch)
    _ordinary_retry(host, monkeypatch, clock, reaper)
    [running] = _assign(host, monkeypatch)
    ctx = ToolContext(repo_dir=workers.REPO_DIR, drive_root=host.root, task_id=running["id"],
                      budget_drive_root=str(host.root))
    ctx.task_metadata = {key: running[key] for key in ("root_task_id", "original_task_id", "timeout_retry_from")}
    ctx.task_metadata["budget_drive_root"] = str(host.root)
    ctx.task_lifecycle_bound = True
    with ThreadPoolExecutor(max_workers=1) as executor, review_episode("held") as armed:
        # The real handoff arms this episode; calling its marker alone is not a launch.
        assert submit_tool(ctx, "review probe", executor.submit, lambda: "started").result(timeout=2) == "started"
    assert request_owner_pause(running["id"], request_id="running-retry")["root_task_id"] == "held"
    assert member_fence(ctx).get("state") == "requested"
    registry = ToolRegistry(repo_dir=workers.REPO_DIR, drive_root=host.root)
    registry.set_context(ctx)
    out = registry.execute("write_file", {"root": "task_drive", "path": "sentinel.txt", "content": "EFFECT"})
    assert "OWNER_PAUSE_NOT_STARTED" in out
    assert not (host.root / "task_drives" / running["id"] / "sentinel.txt").exists()
    from ouroboros import owner_pause
    token = owner_pause._REVIEW_EPISODE.set(armed)
    try:
        assert member_fence(ctx) == {}, "the review that already started finishes its check"
    finally:
        owner_pause._REVIEW_EPISODE.reset(token)
    with review_episode("held"):  # a NEW review episode is refused like any launch
        assert member_fence(ctx).get("state") == "requested"
    # The retry's own boundary parks its exact pause on ITS physical row under the logical fence.
    rctx, limit = _loop_ctx(host.root, running["id"], attempt=running["_attempt"])
    rctx.task_metadata = dict(ctx.task_metadata)
    rctx.active_model, rctx.context_fit_plan = fresh.model, fresh
    limit.messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": MARK}]
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit)
    budget_pause.end_dispatch_fence(running["id"])
    install_exact_budget_pause(SimpleNamespace(
        DRIVE_ROOT=host.root, RUNNING=workers.RUNNING, PENDING=workers.PENDING, WORKERS=workers.WORKERS,
        sort_pending=queue.sort_pending, persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None),
        running["id"], budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    assert budget_pause.budget_pause_row(host.root, running["id"])["root_task_id"] == "held"
    granted = queue.resume_budget_paused_task(running["id"])
    assert granted["ok"] and granted["exact_continuation"] and granted["root_task_id"] == "held"
    assert "held" not in queue.BUDGET_ROOT_FENCES, "the root actor's Resume lifts the logical latch"
    [again] = _assign(host, monkeypatch)
    assert again["id"] == running["id"] and again["_budget_pause_resume"]["root_fence_id"] == ""
    seen = _loop_once(host, monkeypatch, again, plan=fresh)
    assert seen["kind"] == "pause"
    assert read_fence(host.root, "held")["state"] == "released", "consumption reopened the logical tree"


def test_only_the_current_attempt_of_the_chain_acts_for_the_root(host, monkeypatch, clock, reaper):  # noqa: F811
    first_retry = _ordinary_retry(host, monkeypatch, clock, reaper)
    [running] = _assign(host, monkeypatch)
    assert running["id"] == first_retry["id"]
    historical_timeout_retry(clock, running["id"])  # the chain's second historical attempt
    [second] = [t for t in host.pending if t.get("timeout_retry_from") == running["id"]]
    assert root_actor(host.root, "held", second["id"])
    assert not root_actor(host.root, "held", running["id"]), "a superseded attempt never reopens the fence"
    assert request_owner_pause(second["id"], request_id="second")["root_task_id"] == "held"


@pytest.mark.parametrize("prior", ["no_root_grant", "superseded_root_grant"])
def test_a_resumed_retry_selects_its_child_under_its_own_grant(host, monkeypatch, clock, reaper,  # noqa: F811
                                                                prior):
    """The root's live grant is the CURRENT attempt's, read through its verified retry chain: the
    model's child selection binds it and dispatch rechecks it; the logical row's own grant (here
    one a superseded attempt consumed) is neither missing authority nor a binding."""
    from supervisor.events_budget import _handle_budget_resume_child, live_root_resume_grant

    if prior == "no_root_grant":
        fresh = _fresh_plan(monkeypatch)
        _ordinary_retry(host, monkeypatch, clock, reaper)
    else:
        fresh, _pause, resumed = _consumed(host, monkeypatch, clock)
        _loop_once(host, monkeypatch, resumed, plan=fresh, events=_supervisor_events(host))
        historical_timeout_retry(clock)
    [running] = _assign(host, monkeypatch)
    assert running["id"] != "held" and root_actor(host.root, "held", running["id"])
    child = queue.enqueue_task({"id": "held-child", "type": "task", "chat_id": 0, "text": "child work",
                                "root_task_id": "held", "parent_task_id": running["id"], "depth": 1,
                                "delegation_role": "subagent"})
    assert not child.get("_admission_blocked"), child
    assert request_owner_pause(running["id"], request_id="tree")["root_task_id"] == "held"
    rctx, limit = _loop_ctx(host.root, running["id"], attempt=running["_attempt"])
    rctx.task_metadata = {key: running[key] for key in ("root_task_id", "original_task_id", "timeout_retry_from")}
    rctx.task_metadata["budget_drive_root"] = str(host.root)
    rctx.active_model, rctx.context_fit_plan = fresh.model, fresh
    limit.messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": MARK}]
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit)
    budget_pause.end_dispatch_fence(running["id"])
    pushed = []
    sctx = SimpleNamespace(DRIVE_ROOT=host.root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                           WORKERS=workers.WORKERS, sort_pending=queue.sort_pending,
                           persist_queue_snapshot=queue.persist_queue_snapshot,
                           bridge=SimpleNamespace(push_log=pushed.append))
    install_exact_budget_pause(sctx, running["id"], budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    assert queue.resume_budget_paused_task(running["id"])["ok"]
    [again] = _assign(host, monkeypatch)
    assert again["id"] == running["id"], "the root's Resume releases the root alone"
    assert _loop_once(host, monkeypatch, again, plan=fresh)["kind"] == "pause"  # it consumed its grant
    own = budget_pause.budget_pause_row(host.root, running["id"])["grant"]["grant_id"]
    _handle_budget_resume_child({"type": "budget_resume_child", "task_id": "held-child",
                                 "requested_by": running["id"], "reason": "still needed"}, sctx)
    outcome = pushed[-1]
    assert outcome["type"] == "budget_resume_child_outcome" and outcome["ok"] is True, (
        outcome.get("error"), outcome.get("detail"))
    [held] = [row for row in host.pending if row["id"] == "held-child"]
    selection = held["_resume_selection"]
    assert (selection["selected_by"], selection["root_grant_id"]) == (
        running["id"], own), "the selection names the current attempt's grant"
    assert live_root_resume_grant(queue, "held", host.root)["grant_id"] == own
    if prior == "superseded_root_grant":  # an unreadable chain names no grant, never the logical row's
        from ouroboros import task_results

        real_load = task_results.load_task_result
        with monkeypatch.context() as fault:
            fault.setattr(task_results, "load_task_result", lambda root, tid, *a, **k: (
                (_ for _ in ()).throw(OSError("unreadable")) if tid == running["id"] else real_load(root, tid, *a, **k)))
            assert live_root_resume_grant(queue, "held", host.root) == {}
    assert [row["id"] for row in _assign(host, monkeypatch)] == ["held-child"], "dispatch rechecks that grant"
    # A newer Pause the current attempt parks on ends its grant; nothing older stands in.
    assert request_owner_pause(running["id"], request_id="newer")["root_task_id"] == "held"
    rctx, limit = _loop_ctx(host.root, running["id"], attempt=again["_attempt"])
    rctx.task_metadata = dict(again.get("metadata") or {}, root_task_id="held", budget_drive_root=str(host.root))
    rctx.active_model, rctx.context_fit_plan = fresh.model, fresh
    limit.messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": MARK}]
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit)
    budget_pause.end_dispatch_fence(running["id"])
    install_exact_budget_pause(sctx, running["id"], budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    assert live_root_resume_grant(queue, "held", host.root) == {}


# --- a child's same-ID retry under the owner's Pause stays in the held tree -------------------------

def _two_slots(host, monkeypatch):  # noqa: F811
    """Two idle worker slots whose queue ``put`` is the one recorded double."""
    sent = worker(host, monkeypatch)
    workers.WORKERS[1] = affirm_ready(SimpleNamespace(wid=1, busy_task_id=None, reaping=False,
        in_q=SimpleNamespace(put=lambda row: sent.append(dict(row)) if row.get("type") != "worker_quiescence" else None)))
    return sent


def _running_child(host, monkeypatch, *, saved):  # noqa: F811
    """The root runs on slot 0 and its subagent on slot 1, both through real admission and assignment;
    ``saved``: the child saved one round of its own before it hung."""
    from tests.test_resume_handover import _start_record

    fresh = _fresh_plan(monkeypatch)
    _admit(host, None)
    root = _dispatch_one(host, monkeypatch)
    child = queue.enqueue_task({"id": "held-child", "type": "task", "chat_id": 0, "text": "child work",
                                "root_task_id": "held", "parent_task_id": "held", "depth": 1,
                                "delegation_role": "subagent"})
    assert not child.get("_admission_blocked"), child
    sent = _two_slots(host, monkeypatch)
    workers.WORKERS[0].busy_task_id = "held"
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held-child"] and "held-child" in workers.RUNNING
    _start_record(host.root, sent[0])
    if saved:
        cctx, climit = _loop_ctx(host.root, "held-child", attempt=1)
        cctx.root_task_id, cctx.active_model, cctx.context_fit_plan = "held", fresh.model, fresh
        cctx.task_metadata = {"root_task_id": "held", "parent_task_id": "held", "delegation_role": "subagent"}
        climit.messages = [{"role": "user", "content": "child work"}, {"role": "assistant", "content": MARK}]
        wc.save_round(climit, "post_batch")
    return fresh, root


@pytest.mark.parametrize("saved", [True, False], ids=["saved_work", "nothing_saved"])
def test_a_child_retry_under_the_owner_pause_waits_for_its_own_selection(host, monkeypatch, clock,  # noqa: F811
                                                                         reaper, saved):
    from supervisor.events_budget import HOLD_ROOT_FENCE_LIFTED
    from supervisor.owner_pause_control import refresh_owner_pause_tree

    fresh, root = _running_child(host, monkeypatch, saved=saved)
    ctx, limit, _waiter = _attempt_ctx(host, root, monkeypatch, attempt=1)
    handover._pause_through_agent(host, root, ctx, limit, plan=fresh)  # the owner's Pause parks the root
    assert read_fence(host.root, "held")["state"] == "requested", "its child still runs"
    historical_timeout_retry(clock, "held-child")  # a subagent's historical retry kept its own id
    # Not dispatched: the same id, its next attempt, held by the owner's latch.
    [child] = [row for row in host.pending if row["id"] == "held-child"]
    assert child["_attempt"] == 2 and not child.get("_admission_blocked")
    assert bool(child.get("_working_recovery")) is saved
    assert refresh_owner_pause_tree("held") == "paused", "a queued retry is not a running member"
    assert not _assign(host, monkeypatch)
    # The root's Resume releases the root alone; the child is never auto-dispatched.
    assert queue.resume_budget_paused_task("held")["ok"]
    assert budget_hold_fact(child)["reason"] == HOLD_ROOT_FENCE_LIFTED
    sent = _two_slots(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"]
    assert _loop_once(host, monkeypatch, sent[0], plan=fresh)["kind"] == "pause"  # the root consumes its grant
    assert read_fence(host.root, "held")["state"] == "released"
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"], "the reopened tree still dispatches no unselected child"
    selected = queue.resume_budget_paused_task("held-child")  # the owner's explicit selection
    if not saved:
        # Nothing of its own to continue: a typed refusal, never a silent restart or a Failed row.
        assert not selected["ok"] and selected["action"] == "cancel_or_new_run", selected
        workers.assign_tasks()
        assert [row["id"] for row in sent] == ["held"] and budget_hold_fact(child)["reason"] == HOLD_ROOT_FENCE_LIFTED
        return
    assert selected["ok"], selected
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held", "held-child"]
    seen = _loop_once(host, monkeypatch, sent[1], plan=fresh)
    assert seen["kind"] == "working" and any(MARK in str(m.get("content")) for m in seen["messages"])


def test_a_member_seed_with_nothing_saved_names_only_the_actions_it_has(host, monkeypatch, clock,  # noqa: F811
                                                                       reaper):
    """A child's own owner-Pause seed whose attempt saved nothing is pause intent like the root's, but
    no root Resume restarts a member and its selection continues only saved work: its owner-visible
    detail names cancel or a new run, which the typed refusal of its selection confirms (#1729)."""
    from supervisor.events_budget import HOLD_UNSAVED_PAUSE

    fresh, root = _running_child(host, monkeypatch, saved=False)
    ctx, limit, _waiter = _attempt_ctx(host, root, monkeypatch, attempt=1)
    handover._pause_through_agent(host, root, ctx, limit, plan=fresh)  # the owner's Pause parks the root
    cctx, climit = _loop_ctx(host.root, "held-child", attempt=1)
    cctx.root_task_id, cctx.active_model, cctx.context_fit_plan = "held", fresh.model, fresh
    cctx.task_metadata = {"root_task_id": "held", "parent_task_id": "held", "delegation_role": "subagent"}
    climit.round_idx, climit.budget_tail = 0, "no_tool"
    climit.messages = [{"role": "user", "content": "child work"}]

    def die(*_a, **_k):
        raise _WorkerDied

    with monkeypatch.context() as fault:  # the child meets the Pause, then its worker is lost
        fault.setattr(budget_pause, "_HOLD_POLL_SEC", 0.0)
        fault.setattr(budget_pause, "_exact_continuation_row", die)
        with pytest.raises(_WorkerDied):
            budget_pause.enter_owner_pause(climit)
    budget_pause.end_dispatch_fence("held-child")
    seed = budget_pause.budget_pause_row(host.root, "held-child")
    assert (seed["reason"], seed["state"], seed["source_ref"]) == ("owner", budget_pause.STATE_PAUSING, None), seed
    historical_unsaved_pause_hold(clock, "held-child")
    [child] = [row for row in host.pending if row["id"] == "held-child"]
    hold = budget_hold_fact(child)
    assert hold["reason"] == HOLD_UNSAVED_PAUSE and "_working_recovery" not in child, hold
    assert hold["detail"].endswith("nothing was saved to continue: cancel it or start a new run"), hold
    assert queue.resume_budget_paused_task("held")["ok"]  # the root's own Resume: the member stays held
    sent = _two_slots(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"] and budget_hold_fact(child) == hold
    assert _loop_once(host, monkeypatch, sent[0], plan=fresh)["kind"] == "pause"  # the root consumes its grant
    selected = queue.resume_budget_paused_task("held-child")
    assert not selected["ok"] and selected["action"] == "cancel_or_new_run", selected
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"] and budget_hold_fact(child) == hold


# --- shared with the retained-reap Stop tests -------------------------------------------------------

def _paused_hung_attempt(host, monkeypatch, clock):  # noqa: F811
    fresh = _fresh_plan(monkeypatch)
    _admit(host, None)
    task = _dispatch_one(host, monkeypatch)
    ctx, limit, _waiter = _attempt_ctx(host, task, monkeypatch, attempt=1)
    ctx.active_model, ctx.context_fit_plan = fresh.model, fresh
    _handler_pause(host, ctx, limit, task, request_id="race")
    return fresh


def _owner_stop(host, task_id):  # noqa: F811
    """The owner's Stop through the real HTTP handler (``POST /api/tasks/{id}/cancel``, body ``{}``)."""
    import asyncio
    import json

    from starlette.requests import Request

    from ouroboros.gateway.tasks import api_task_cancel

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": f"/api/tasks/{task_id}/cancel",
                       "path_params": {"task_id": task_id}, "headers": [(b"content-type", b"application/json")],
                       "app": SimpleNamespace(state=SimpleNamespace(drive_root=host.root, repo_dir=workers.REPO_DIR))},
                      receive)
    response = asyncio.run(api_task_cancel(request))
    return response.status_code, json.loads(response.body)
