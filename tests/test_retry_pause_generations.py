"""A new owner Pause outlives an assigned retry's unconsumed physical Resume.

Use a historical retry's stored state, Pause/Resume acceptance, worker assignment, grant
consumption, launch gate and next safe boundary. Only the shared fixtures' OS
processes/model calls are doubled. Request replay must still be idempotent.
"""
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, owner_pause
from supervisor import queue, workers
from supervisor.events_budget import install_exact_budget_pause
from supervisor.owner_pause_control import request_owner_pause
from tests._budget_pause_exact_helpers import _loop_ctx
from tests._silence_reaper_support import historical_timeout_retry
from tests.test_execution_clock_resume_handover import clock  # noqa: F401
from tests.test_retry_pause_identity import (  # noqa: F401
    MARK, _assign, _fresh_plan, _hung_first_attempt, _loop_once, _ordinary_retry, reaper,
)
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial


def _running_actor(host, monkeypatch, clock, reaper, actor):  # noqa: F811
    if actor == "root":
        _hung_first_attempt(host, monkeypatch, clock)
        return dict(workers.RUNNING["held"]["task"])
    _ordinary_retry(host, monkeypatch, clock, reaper)
    if actor == "retry_chain":
        [prior] = _assign(host, monkeypatch)
        historical_timeout_retry(clock, prior["id"])
    [running] = _assign(host, monkeypatch)
    assert owner_pause.root_actor(host.root, "held", running["id"])
    return running


def _park(host, running, plan, *, event=False):  # noqa: F811 -- helper shares the imported fixture's name
    ctx, limit = _loop_ctx(host.root, running["id"], attempt=running["_attempt"])
    ctx.task_metadata = {key: running[key] for key in (
        "root_task_id", "original_task_id", "timeout_retry_from") if running.get(key)}
    ctx.task_metadata["budget_drive_root"] = str(host.root)
    ctx.active_model, ctx.context_fit_plan = plan.model, plan
    limit.messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": MARK}]
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit)
    budget_pause.end_dispatch_fence(running["id"])
    pool = SimpleNamespace(
        DRIVE_ROOT=host.root, RUNNING=workers.RUNNING, PENDING=workers.PENDING, WORKERS=workers.WORKERS,
        sort_pending=queue.sort_pending, persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    if event:  # the live worker's own park event through its supervisor handler
        from supervisor.events_budget import _handle_budget_pause

        _handle_budget_pause(budget_pause.pause_event(running, raised.value.pause), pool)
    else:
        install_exact_budget_pause(pool, running["id"], budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    return ctx, limit


@pytest.mark.parametrize("actor", ["root", "retry", "retry_chain"])
@pytest.mark.parametrize("action", ["ordinary", "duplicate", "newer"])
def test_assigned_resume_releases_only_its_pause_generation(host, monkeypatch, clock, reaper, actor, action):  # noqa: F811
    running = _running_actor(host, monkeypatch, clock, reaper, actor)
    plan = _fresh_plan(monkeypatch)
    first = request_owner_pause(running["id"], request_id="first")
    assert first["ok"] and first["root_task_id"] == "held"
    # An ordinary second press before any Resume coalesces with the closed fence.
    coalesced = request_owner_pause(running["id"], request_id="still-paused")
    assert coalesced["duplicate"] and coalesced["fence_id"] == first["fence_id"]
    ctx, limit = _park(host, running, plan)
    assert queue.resume_budget_paused_task(running["id"])["ok"]
    [again] = _assign(host, monkeypatch)
    row = budget_pause.budget_pause_row(host.root, running["id"])
    assert row["state"] == budget_pause.STATE_RESUME_GRANTED and not row["grant"].get("consumed_at")
    assert row["grant"]["owner_pause_fence_id"] == first["fence_id"]
    accepted = first
    if action != "ordinary":
        accepted = request_owner_pause(running["id"], request_id="first" if action == "duplicate" else "second")
        assert accepted["ok"]
    # Consume through the real worker loop, then probe both consumers before asserting identities.
    seen = _loop_once(host, monkeypatch, again, plan=plan)
    assert seen["kind"] == "pause"
    ctx = limit.tools._ctx = seen["ctx"]
    effects = []
    try:
        with owner_pause.launch_admission(ctx):
            effects.append("launched")
    except owner_pause.OwnerPauseRefused:
        pass
    parked = False
    try:
        budget_pause.enter_owner_pause(limit)
    except budget_pause.BudgetPauseRequested:
        parked = True
    finally:
        budget_pause.end_dispatch_fence(running["id"])
    fence = owner_pause.read_fence(host.root, "held")
    facts = {"actor": actor, "action": action, "duplicate": accepted["duplicate"],
             "same_fence": accepted["fence_id"] == first["fence_id"], "state": fence["state"],
             "effects": effects, "next_boundary_parked": parked}
    print("assigned Resume/Pause witness:", facts)
    if action == "newer":
        assert effects == [] and parked, facts
        assert not accepted["duplicate"] and accepted["fence_id"] != first["fence_id"]
        assert fence["generation"] == 2 and fence["supersedes_fence_id"] == first["fence_id"]
        replay = request_owner_pause(running["id"], request_id="first")
        assert replay["duplicate"] and replay["state"] == "released" and replay["fence_id"] == first["fence_id"]
        assert owner_pause.read_fence(host.root, "held")["fence_id"] == accepted["fence_id"]
    else:
        assert effects == ["launched"] and not parked and fence["state"] == "released", facts
        assert accepted["fence_id"] == first["fence_id"]
        assert accepted["duplicate"] is (action == "duplicate")


@pytest.mark.parametrize("invalid", ["unreadable_chain", "broken_reciprocal", "stale_attempt", "child", "delegated"])
def test_pause_keeps_the_current_reciprocal_root_actor_requirement(host, monkeypatch, clock, reaper, invalid):  # noqa: F811
    from ouroboros import task_results
    from ouroboros.task_store import patch_document

    running = _running_actor(host, monkeypatch, clock, reaper, "retry_chain")
    assert request_owner_pause(running["id"], request_id="first")["ok"]
    before = owner_pause.read_fence(host.root, "held")
    target = running["id"]
    expected = "retry_lineage_unverified"
    if invalid == "unreadable_chain":
        real_load = task_results.load_task_result

        def unreadable(root, task_id, *args, **kwargs):
            if task_id == target:
                raise OSError("physical result unreadable")
            return real_load(root, task_id, *args, **kwargs)

        monkeypatch.setattr(task_results, "load_task_result", unreadable)
    elif invalid == "broken_reciprocal":
        patch_document(host.root, target, lambda row: {**row, "supersedes_task_id": "wrong"})
    elif invalid == "stale_attempt":
        target = running["timeout_retry_from"]
        prior = task_results.load_task_result(host.root, target, strict=True)
        workers.RUNNING[target] = {"task": {**prior, "id": target}, "worker_id": 0}
    else:
        task = workers.RUNNING[target]["task"]
        task["delegation_role"] = "subagent"
        if invalid == "child":
            task["parent_task_id"] = "held"
        expected = "not_a_root_task"
    refused = request_owner_pause(target, request_id="invalid")
    assert not refused["ok"] and refused["error"] == expected, refused
    assert owner_pause.read_fence(host.root, "held") == before


@pytest.mark.parametrize("fault", ["changed_chain", "unreadable_grant"])
def test_pause_revalidates_physical_authority_inside_the_logical_launch_lock(
        host, monkeypatch, clock, reaper, fault):  # noqa: F811
    import inspect
    from ouroboros import task_results

    running = _running_actor(host, monkeypatch, clock, reaper, "retry")
    assert request_owner_pause(running["id"], request_id="first")["ok"]
    _park(host, running, _fresh_plan(monkeypatch))
    assert queue.resume_budget_paused_task(running["id"])["ok"]
    _assign(host, monkeypatch)
    before = owner_pause.read_fence(host.root, "held")
    real, injected = task_results.load_task_result, []

    def changed(root, task_id, *args, **kwargs):
        row = real(root, task_id, *args, **kwargs)
        stack = {frame.function for frame in inspect.stack()}
        if task_id == running["id"] and "install_fence" in stack:
            if fault == "changed_chain":
                injected.append(fault)
                return {**row, "supersedes_task_id": "wrong"}
            if "root_actor" not in stack:
                injected.append(fault)
                raise OSError("physical grant became unreadable after lineage validation")
        return row

    monkeypatch.setattr(task_results, "load_task_result", changed)
    refused = request_owner_pause(running["id"], request_id="second")
    assert injected and not refused["ok"]
    assert refused["error"] == ("retry_lineage_unverified" if fault == "changed_chain" else "pause_record_unwritable")
    assert owner_pause.read_fence(host.root, "held") == before


@pytest.mark.parametrize("actor", ["retry", "retry_chain"])
def test_physical_retry_resume_survives_fence_publication_contention(
        host, monkeypatch, clock, reaper, actor):  # noqa: F811
    from ouroboros.store import StoreBusy

    running = _running_actor(host, monkeypatch, clock, reaper, actor)
    plan = _fresh_plan(monkeypatch)
    first = request_owner_pause(running["id"], request_id="first")
    assert first["ok"]
    _park(host, running, plan)
    assert queue.resume_budget_paused_task(running["id"])["ok"]
    [again] = _assign(host, monkeypatch)
    release, finish = owner_pause.release_fence, owner_pause._finish_resume_publication
    publications, failures = [], []

    def contended(*args, **kwargs):
        publications.append(True)
        if len(publications) == 1:
            # The writer proved rollback; this is not an unknown effect or a read failure.
            error = StoreBusy("proved-rolled-back fence publication", reason="contention")
            error.rollback_confirmed = True
            raise error
        return release(*args, **kwargs)

    def observed_finish(*args, **kwargs):
        try:
            return finish(*args, **kwargs)
        except Exception as exc:
            failures.append(str(exc))
            raise

    monkeypatch.setattr(owner_pause, "release_fence", contended)
    monkeypatch.setattr(owner_pause, "_finish_resume_publication", observed_finish)
    seen = _loop_once(host, monkeypatch, again, plan=plan)
    assert not failures, failures
    assert seen["kind"] == "pause" and len(publications) == 2
    assert owner_pause.read_fence(host.root, "held")["state"] == "released"
    assert budget_pause.budget_pause_row(host.root, running["id"])["grant"]["consumed_at"]
    with owner_pause.launch_admission(seen["ctx"]):
        effect = "allowed only after publication"
    assert effect


@pytest.mark.parametrize("change", ["pause", "stop_root", "stop_retry", "grant", "lineage"])
def test_retry_fence_publication_rechecks_controls_after_rollback(
        host, monkeypatch, clock, reaper, change):  # noqa: F811
    from ouroboros import cancel_intents
    from ouroboros.store import StoreBusy
    from ouroboros.task_store import patch_document

    running = _running_actor(host, monkeypatch, clock, reaper, "retry")
    first = request_owner_pause(running["id"], request_id="first")
    assert first["ok"]
    _park(host, running, _fresh_plan(monkeypatch))
    assert queue.resume_budget_paused_task(running["id"])["ok"]
    _assign(host, monkeypatch)
    pause = budget_pause.budget_pause_row(host.root, running["id"])
    release, attempts, waits = owner_pause.release_fence, [], []

    def contended(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            error = StoreBusy("proved rollback before outside control", reason="contention")
            error.rollback_confirmed = True
            raise error
        return release(*args, **kwargs)

    def outside(_seconds):
        with owner_pause.launch_lock(host.root, "held", wait=False):
            pass  # No root launch lock is held across the wait.
        waits.append(True)
        if change == "pause":
            owner_pause.install_fence(host.root, "held", request_id="new", actor_id=running["id"])
        elif change.startswith("stop_"):
            cancel_intents.request_cancel(host.root, "held" if change == "stop_root" else running["id"],
                                         source="http_single", reason="owner Stop")
        elif change == "grant":
            budget_pause.set_budget_pause(host.root, running["id"], {
                **pause, "grant": {**pause["grant"], "grant_id": "another-grant"}})
        else:
            patch_document(host.root, running["id"], lambda row: {**row, "supersedes_task_id": "wrong"})

    monkeypatch.setattr(owner_pause, "release_fence", contended)
    timer = owner_pause.time
    monkeypatch.setattr(owner_pause, "time", SimpleNamespace(
        monotonic=timer.monotonic, time=timer.time, sleep=outside))

    def resume():
        owner_pause.reopen_for_resume(host.root, "held", running["id"],
                                     fence_id=first["fence_id"], grant_id=pause["grant"]["grant_id"])

    if change == "pause":
        resume()
    else:
        refusal = (cancel_intents.CancelIntentLineageIndeterminate if change == "lineage"
                   else owner_pause.OwnerPauseRefused)
        with pytest.raises(refusal):
            resume()
    assert waits == [True] and attempts == [True]
    fence = owner_pause.read_fence(host.root, "held")
    assert owner_pause.fence_closed(fence)
    assert (fence["fence_id"] != first["fence_id"]) is (change == "pause")


@pytest.mark.parametrize("stop", ["logical_cascade", "physical_single"])
def test_a_pending_stop_of_either_retry_identity_vetoes_the_pause(host, monkeypatch, clock, reaper, stop):  # noqa: F811
    from ouroboros import cancel_intents

    running = _running_actor(host, monkeypatch, clock, reaper, "retry")
    if stop == "logical_cascade":  # the graceful cascade ingress keeps its intent on the logical root
        cancel_intents.request_cancel(
            host.root, "held", reason="owner requested finalize-then-stop", source="http_graceful",
            requested_by="owner", requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE,
            scope=cancel_intents.SCOPE_CASCADE, allow_settled_target=True)
    else:
        cancel_intents.request_cancel(host.root, running["id"], source="http_single", reason="owner Stop")
    keyed = {key: cancel_intents.has_active_intent(host.root, key, strict=True) for key in ("held", running["id"])}
    assert keyed == {"held": stop == "logical_cascade", running["id"]: stop == "physical_single"}, keyed
    before, latch = owner_pause.read_fence(host.root, "held"), dict(queue.BUDGET_ROOT_FENCES)
    refused = request_owner_pause(running["id"], request_id="under-stop")
    assert not refused["ok"] and refused["error"] == "cancel_pending", refused
    assert owner_pause.read_fence(host.root, "held") == before and dict(queue.BUDGET_ROOT_FENCES) == latch


@pytest.mark.parametrize("latch", ["latched", "latch_pending"])
def test_a_live_park_over_an_unknown_current_fence_projects_none(host, monkeypatch, clock, reaper, latch):  # noqa: F811
    """A live park is never replayed (a failed event is dropped), so an unreadable current fence
    keeps the standing latch and mints none; the durable fence still refuses every launch, and one
    ordinary Resume dispatches the parked root."""
    import inspect

    from tests.test_acceptance_fence_cold_resume import _pause_with_latch_pending

    running = _running_actor(host, monkeypatch, clock, reaper, "root")
    plan = _fresh_plan(monkeypatch)
    paused = (request_owner_pause(running["id"], request_id="first") if latch == "latched"
              else _pause_with_latch_pending(host, monkeypatch, running["id"], "first"))
    assert paused["ok"] and bool(paused.get("latch_pending")) is (latch == "latch_pending"), paused
    standing = (queue.BUDGET_ROOT_FENCES.get("held") or {}).get("fence_id")
    real, injected = owner_pause.read_fence, []

    def unreadable(*args, **kwargs):  # only the park's own read of the current fence
        if "_owner_park_fence_id" in {frame.function for frame in inspect.stack()}:
            injected.append(True)
            raise ValueError("owner_pause_authority_unreadable")
        return real(*args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(owner_pause, "read_fence", unreadable)
        _park(host, running, plan, event=True)
    [parked] = [t for t in workers.PENDING if t["id"] == "held"]
    facts = {"latch": latch, "injected": len(injected), "fence": paused["fence_id"], "standing": standing,
             "after": (queue.BUDGET_ROOT_FENCES.get("held") or {}).get("fence_id"),
             "marker": parked["_budget_pause"].get("fence_id"), "running": "held" in workers.RUNNING}
    print("unknown-fence live park witness:", facts)
    assert injected and facts["after"] == standing == facts["marker"] and not facts["running"], facts
    assert standing == (paused["fence_id"] if latch == "latched" else None), facts
    assert not _assign(host, monkeypatch), "the durable fence holds the parked root"
    assert queue.resume_budget_paused_task("held")["ok"]
    assert "held" not in queue.BUDGET_ROOT_FENCES
    [sent] = _assign(host, monkeypatch)
    row = budget_pause.budget_pause_row(host.root, "held")
    assert sent["id"] == "held" and row["grant"]["owner_pause_fence_id"] == paused["fence_id"], row


@pytest.mark.parametrize("park_read", ["readable", "unknown"])
@pytest.mark.parametrize("latch", ["latched", "latch_pending"])
def test_root_resume_holds_queued_children_for_selection_without_a_projected_latch(
        host, monkeypatch, clock, reaper, latch, park_read):  # noqa: F811
    """Two queued children under a live root; the Pause's latch stands or is ``latch_pending``, and the
    park's own read of the current fence is readable or meets real SQLite contention. A Resume whose
    fresh read still fails changes nothing; the next ordinary Resume binds the closed current fence:
    the census holds the never-started children, it selects the root alone, and only the child the
    root then selects runs. A newer Pause still refuses the other one."""
    import inspect

    from supervisor.events_budget import budget_hold_fact
    from tests.test_acceptance_fence_cold_resume import _pause_with_latch_pending
    from tests.test_admission_contention import database_reader

    running = _running_actor(host, monkeypatch, clock, reaper, "root")
    plan = _fresh_plan(monkeypatch)
    children = ["child-selected", "child-unselected"]
    for tid in children:
        admitted = queue.enqueue_task({"id": tid, "type": "task", "chat_id": 0, "text": "child work",
                                       "root_task_id": "held", "parent_task_id": "held", "depth": 1,
                                       "delegation_role": "subagent"})
        assert not admitted.get("_admission_blocked"), admitted

    def pause(request_id):
        answer = (request_owner_pause("held", request_id=request_id) if latch == "latched"
                  else _pause_with_latch_pending(host, monkeypatch, "held", request_id))
        assert answer["ok"] and bool(answer.get("latch_pending")) is (latch == "latch_pending"), answer
        return answer

    first = pause("first")
    real = owner_pause.read_fence

    def contended(*args, **kwargs):  # a competing SQLite writer holds the store for the park's read only
        if "_owner_park_fence_id" in {frame.function for frame in inspect.stack()}:
            with database_reader(host.root, exclusive=True):
                return real(*args, **kwargs)
        return real(*args, **kwargs)

    with monkeypatch.context() as fault:
        if park_read == "unknown":
            fault.setattr(owner_pause, "read_fence", contended)
        _park(host, running, plan, event=True)
    projected = (queue.BUDGET_ROOT_FENCES.get("held") or {}).get("fence_id")
    assert projected == (None if (latch, park_read) == ("latch_pending", "unknown") else first["fence_id"])
    assert not _assign(host, monkeypatch), "nothing launches under the Pause"

    def unreadable(*args, **kwargs):  # the Resume's own fresh read still fails
        if "resume_budget_paused_task" in {frame.function for frame in inspect.stack()}:
            raise ValueError("owner_pause_authority_unreadable")
        return real(*args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(owner_pause, "read_fence", unreadable)
        refused = queue.resume_budget_paused_task("held")
    rows = {t["id"]: t for t in workers.PENDING}
    assert refused["ok"] is False and refused["error"] == "owner_pause_custody_unreadable", refused
    assert rows["held"].get("_budget_pause") and not any(budget_hold_fact(rows[c]) for c in children)
    assert (queue.BUDGET_ROOT_FENCES.get("held") or {}).get("fence_id") == projected
    resumed = queue.resume_budget_paused_task("held")
    print("queued-children Resume:", {"latch": latch, "park_read": park_read, "projected": projected,
                                      "resume": {k: resumed.get(k) for k in ("ok", "error", "held_siblings")}})
    assert resumed["ok"] and sorted(resumed["held_siblings"]) == sorted(children), resumed
    [root] = _assign(host, monkeypatch)
    assert root["id"] == "held", "Root Resume selects the root alone"
    assert _loop_once(host, monkeypatch, root, plan=plan)["kind"] == "pause"
    assert queue.resume_budget_paused_task(children[0], selected_by="held")["ok"]
    assert [t["id"] for t in _assign(host, monkeypatch)] == [children[0]], "only the selected child runs"
    [sibling] = [t for t in workers.PENDING if t["id"] == children[1]]
    assert budget_hold_fact(sibling) and "_resume_selection" not in sibling
    newer = pause("newer")  # a replacement Pause: the older Resume selects nothing more
    assert newer["fence_id"] != first["fence_id"]
    late = queue.resume_budget_paused_task(children[1], selected_by="held")
    assert not late["ok"], late
    assert not _assign(host, monkeypatch), "the newer Pause refuses the unselected child"


WARM_CHILDREN = ("child-selected", "child-unselected")


def _warm_root_over_children(host, monkeypatch, clock, reaper, latch, *, before_pause=None):  # noqa: F811
    """The root parks WARM under the owner's Pause (its worker keeps the stack while a critic it
    started finishes) over two never-started queued children; the latch stands or is ``latch_pending``.
    Doubled: the warm park's durably waiting RUNNING row and the worker queues' ``put``."""
    from tests.test_acceptance_fence_cold_resume import _pause_with_latch_pending
    from tests.test_project_hold_recovery import worker

    running = _running_actor(host, monkeypatch, clock, reaper, "root")
    for tid in WARM_CHILDREN:
        admitted = queue.enqueue_task({"id": tid, "type": "task", "chat_id": 0, "text": "child work",
                                       "root_task_id": "held", "parent_task_id": "held", "depth": 1,
                                       "delegation_role": "subagent"})
        assert not admitted.get("_admission_blocked"), admitted
    if before_pause is not None:
        before_pause()

    def pause(request_id, mode=latch):
        answer = (request_owner_pause("held", request_id=request_id) if mode == "latched"
                  else _pause_with_latch_pending(host, monkeypatch, "held", request_id))
        assert answer["ok"] and bool(answer.get("latch_pending")) is (mode == "latch_pending"), answer
        return answer

    first = pause("first")
    workers.RUNNING["held"]["owner_wait"] = {"wait_id": "warm-1", "state": "waiting", "reason": "owner_pause",
                                             "task_attempt": int(running["_attempt"])}
    sent = worker(host, monkeypatch)
    workers.WORKERS[0].busy_task_id = "held"  # the warm root keeps its worker; a second one is free
    workers.WORKERS[1] = affirm_ready(SimpleNamespace(wid=1, busy_task_id=None, reaping=False,
                                                     in_q=workers.WORKERS[0].in_q))

    def assign():
        workers.assign_tasks()
        workers.assign_tasks()
        return [row["id"] for row in sent]

    assert not assign(), "nothing launches under the Pause"
    assert (queue.BUDGET_ROOT_FENCES.get("held") or {}).get("fence_id") == (
        first["fence_id"] if latch == "latched" else None)
    return first, pause, assign


@pytest.mark.parametrize("latch", ["latched", "latch_pending"])
def test_warm_root_resume_holds_queued_children_without_a_projected_latch(
        host, monkeypatch, clock, reaper, latch):  # noqa: F811
    """The warm root's Resume completes a missing latch from the closed fence it read BEFORE releasing
    that fence, so its never-started children wait for explicit selection (Q9) exactly as after a
    projected latch: it selects the root alone, only the child the root selects runs, and a newer Pause
    still refuses the other one."""
    from supervisor.events_budget import budget_hold_fact

    first, pause, assign = _warm_root_over_children(host, monkeypatch, clock, reaper, latch)
    resumed = queue.resume_budget_paused_task("held")
    holds = {t["id"]: (budget_hold_fact(t) or {}).get("reason") for t in workers.PENDING}
    print("warm root Resume:", {"latch": latch, "resume": resumed, "holds": holds})
    assert resumed["ok"] and resumed["warm"] and sorted(resumed["held_siblings"]) == sorted(WARM_CHILDREN), resumed
    assert holds == dict.fromkeys(WARM_CHILDREN, "root_fence_lifted_pending_selection"), holds
    assert owner_pause.read_fence(host.root, "held")["state"] == "released"
    assert "held" not in queue.BUDGET_ROOT_FENCES and not assign(), "Root Resume selects the root alone"
    assert queue.resume_budget_paused_task(WARM_CHILDREN[0], selected_by="held")["ok"]
    assert assign() == [WARM_CHILDREN[0]], "only the selected child runs"
    newer = pause("newer")
    assert newer["fence_id"] != first["fence_id"]
    assert not queue.resume_budget_paused_task(WARM_CHILDREN[1], selected_by="held")["ok"]
    assert assign() == [WARM_CHILDREN[0]], "the newer Pause refuses the unselected child"


@pytest.mark.parametrize("control", ["unknown_effect", "pending_stop", "generation_changed"])
def test_warm_root_resume_without_a_latch_keeps_every_refusal(host, monkeypatch, clock, reaper, control):  # noqa: F811
    """``latch_pending``: an effect of unknown outcome and a pending Stop refuse before any latch is
    completed; a fence generation replaced between the Resume's read and its release refuses too and
    withdraws the latch it completed for the dead generation. Nothing launches either way, and the
    current generation's own Resume then holds the children for selection."""
    import inspect

    from ouroboros import cancel_intents
    from ouroboros.owner_pause import operation_start, tool_handoff

    def handed_and_lost():  # the root handed an operation whose outcome is unknown
        source = SimpleNamespace(task_id="held", root_task_id="held", drive_root=host.root)
        with pytest.raises(TimeoutError):
            with tool_handoff(source, "external_write"):
                with operation_start(source):
                    raise TimeoutError("request sent, response unknown")

    first, pause, assign = _warm_root_over_children(
        host, monkeypatch, clock, reaper, "latch_pending",
        before_pause=handed_and_lost if control == "unknown_effect" else None)
    real, replaced = owner_pause.launch_lock, []

    def replaced_meanwhile(*args, **kwargs):  # once, as the Resume reaches its release
        if not replaced and "resume_warm_owner_pause_root" in {f.function for f in inspect.stack()}:
            replaced.append((queue.BUDGET_ROOT_FENCES.get("held") or {}).get("fence_id"))
            owner_pause.release_fence(host.root, "held", reason="owner_resume")
            replaced.append(pause("newer")["fence_id"])
        return real(*args, **kwargs)

    if control == "pending_stop":
        cancel_intents.request_cancel(host.root, "held", source="http_single", reason="owner Stop")
    with monkeypatch.context() as race:
        if control == "generation_changed":
            race.setattr(owner_pause, "launch_lock", replaced_meanwhile)
        refused = queue.resume_budget_paused_task("held")
    facts = {"control": control, "refused": refused, "replaced": replaced,
             "latch": (queue.BUDGET_ROOT_FENCES.get("held") or {}).get("fence_id"),
             "fence": owner_pause.read_fence(host.root, "held")}
    print("warm root Resume control:", facts)
    expected = {"unknown_effect": "owner_pause_effects_unsettled", "pending_stop": "cancel_intent_active",
                "generation_changed": "selection_authority_changed"}[control]
    assert not refused["ok"] and refused["error"] == expected, facts
    if control == "unknown_effect":
        assert [b["kind"] for b in refused["blockers"]] == ["tool_handoff"], facts
    assert facts["latch"] is None and owner_pause.fence_closed(facts["fence"]), facts
    assert not assign(), "nothing launches over a refused Resume"
    if control == "generation_changed":
        assert replaced == [first["fence_id"], facts["fence"]["fence_id"]], facts
        assert facts["fence"]["fence_id"] != first["fence_id"]
        resumed = queue.resume_budget_paused_task("held")
        assert resumed["ok"] and sorted(resumed["held_siblings"]) == sorted(WARM_CHILDREN), resumed
        assert not assign(), "the current generation's Resume still selects the root alone"
