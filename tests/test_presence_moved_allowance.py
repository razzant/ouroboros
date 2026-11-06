"""The owner's choice of 2026-11-02 for a moved Presence turn's producer allowance (#2566, quiz f43a393a).

Only the owner's own Resume door moves the turn off its allowance onto the ordinary project budget: a mind's, a model's or an
automatic Resume never does, and no repeat, rollback or restart widens or resets that decision. Spend so far, the tree and
group caps, the shared Total budget, deadlines, Stop/Panic and unknown outcomes stay the grant's own checks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _parked
from tests.test_presence_moved_lifecycle import CEILING, PROJECT, _seed_and_built, root as root
from tests._cases import case_folders


def _moved_paused(root, monkeypatch, built, task_id, *, metadata_extra=None, move=True):
    """A moved autonomous Place turn with its producer allowance, saved by an exact pause, the queue keeping it."""
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result

    metadata = {**built["metadata"], "root_cost_ceiling_usd": CEILING, **(metadata_extra or {})}
    write_task_result(root, task_id, "running", metadata=metadata, chat_id=built["chat_id"], _is_direct_chat=True,
                      source="presence", task_contract=built["task_contract"], result="Task is running.")
    if move:
        bind_task_to_project(root, task_id, PROJECT, origin=_owner_task_origin(root, task_id))
    record = {key: value for key, value in built.items() if key not in {"id", "root_task_id", "_attempt"}}
    # The row a park keeps is a saved record (``saved_turn_record``'s own fact).
    task, _row = _parked(root, monkeypatch, task_id=task_id,
                         extra={**record, "metadata": {**metadata, "presence_request_waits": False}})
    return task


def _resume_over_http(task_id, body=None):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_task_resume

    app = Starlette(routes=[Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])
    with TestClient(app) as client:
        if body is None:
            return client.post(f"/api/tasks/{task_id}/resume")
        return client.post(f"/api/tasks/{task_id}/resume", json=body)


def _grant(root, task_id):
    from ouroboros.budget_pause import budget_pause_row

    return budget_pause_row(root, task_id).get("grant") or {}


def _row(task_id):
    from supervisor import workers

    return next(task for task in workers.PENDING if task["id"] == task_id)


def test_only_the_owners_own_resume_door_moves_a_moved_turn_off_its_producer_allowance(root, monkeypatch):
    from ouroboros.control_actor import mind_control, owner_door
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from supervisor import queue

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    # A mind's resume_task (its control binds ``mind_control``), even inside an owner door: the grant stands, the
    # allowance stays.
    _moved_paused(root, monkeypatch, built, tid + "-mind")
    with owner_door("api_task_resume"), mind_control("mind-1", "the owner asked me to resume it"):
        assert queue.resume_budget_paused_task(tid + "-mind")["ok"] is True
    assert "producer_allowance_lifted" not in _grant(root, tid + "-mind")
    assert recorded_producer_ceiling(_row(tid + "-mind")) == CEILING
    # A model's resume_child_task selection: no lift whatever its outcome.
    _moved_paused(root, monkeypatch, built, tid + "-child")
    with mind_control("parent-1", "selected"):
        queue.resume_budget_paused_task(tid + "-child", selected_by="parent-1")
    assert recorded_producer_ceiling(_row(tid + "-child")) == CEILING
    assert "producer_allowance_lifted" not in _grant(root, tid + "-child")
    # An unbound call (every automatic path: a restart's return, a sleep's wake, an internal resume) never reads as
    # the owner.
    _moved_paused(root, monkeypatch, built, tid + "-auto")
    assert queue.resume_budget_paused_task(tid + "-auto")["ok"] is True
    assert "producer_allowance_lifted" not in _grant(root, tid + "-auto")
    assert recorded_producer_ceiling(_row(tid + "-auto")) == CEILING
    # The owner's door on an unmoved turn, or on owner work at entry (C8), lifts nothing: no moved autonomous work.
    _moved_paused(root, monkeypatch, built, tid + "-unmoved", move=False)
    _moved_paused(root, monkeypatch, built, tid + "-owner-work", metadata_extra={"work_kind": "owner"})
    for suffix in ("-unmoved", "-owner-work"):
        assert _resume_over_http(tid + suffix).status_code == 200
        assert recorded_producer_ceiling(_row(tid + suffix)) == CEILING
        assert "producer_allowance_lifted" not in _grant(root, tid + suffix)
    # The owner's own Resume door: the decision is recorded once with the grant, and the queue row keeps it in place
    # of the allowance, so every later start, restore, member and Continue reads no producer ceiling.
    owner = tid + "-owner"
    before = _moved_paused(root, monkeypatch, built, owner)
    before_metadata = dict(before["metadata"])
    kept = {key: before.get(key) for key in ("task_contract", "deadline_at", "execution_timeout_sec", "chat_id")}
    response = _resume_over_http(owner)
    assert response.status_code == 200 and response.json()["exact_continuation"] is True
    decision = _grant(root, owner)["producer_allowance_lifted"]
    assert decision["ceiling_usd"] == CEILING and decision["by"]["actor"] == "owner"
    assert decision["by"]["door"] == "api_task_resume"
    # The durable row keeps it too: a Stop before the start, then a Continue, reads no number (the others keep theirs).
    durable = load_task_result(root, owner)
    assert recorded_producer_ceiling(durable) is None and durable["metadata"]["producer_allowance_lifted"] == decision
    for kept_by in ("-mind", "-child", "-auto", "-unmoved", "-owner-work"):
        assert recorded_producer_ceiling(load_task_result(root, tid + kept_by)) == CEILING
    row = _row(owner)
    assert recorded_producer_ceiling(row) is None and row["metadata"]["producer_allowance_lifted"] == decision
    assert {key: row.get(key) for key in kept} == kept  # contract, deadline, lifetime and room unchanged
    unchanged = {key: value for key, value in before_metadata.items() if key != "root_cost_ceiling_usd"}
    assert {key: value for key, value in row["metadata"].items() if key != "producer_allowance_lifted"} == unchanged


def test_the_owners_lift_is_once_rolls_back_with_its_grant_and_never_bypasses_money(root, monkeypatch):
    from ouroboros.control_actor import owner_door
    from ouroboros.presence_carrier import lift_allowance, owner_allowance_lift, saved_turn_record
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import patch_document
    from supervisor import queue, state

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    # Money first: a still-exhausted wallet refuses the owner's Resume and lifts nothing.
    broke = tid + "-broke"
    _moved_paused(root, monkeypatch, built, broke)
    with monkeypatch.context() as patched:
        patched.setattr(state, "budget_remaining", lambda _st, **_k: 0.0)
        assert _resume_over_http(broke).json()["error"] == "budget_still_exhausted"
    assert recorded_producer_ceiling(_row(broke)) == CEILING and "_budget_pause" in _row(broke)
    # A grant whose snapshot never landed is rolled back with its lift: the Resume did not happen.
    lost = tid + "-lost"
    _moved_paused(root, monkeypatch, built, lost)
    with monkeypatch.context() as patched:
        patched.setattr(queue, "publish_queue_snapshot", lambda **_k: False)
        assert _resume_over_http(lost).json()["error"] == "snapshot_not_persisted"
    row = _row(lost)
    assert recorded_producer_ceiling(row) == CEILING and "producer_allowance_lifted" not in row["metadata"]
    assert "_budget_pause" in row and "_budget_pause_resume" not in row
    stored = load_task_result(root, lost)  # in the same write as the grant's own rollback
    assert recorded_producer_ceiling(stored) == CEILING and "producer_allowance_lifted" not in stored["metadata"]
    assert "grant" not in stored["budget_pause"]
    # Lifted once: no later Resume of the same work finds an allowance to lift again, so none widens or resets it.
    once = tid + "-once"
    _moved_paused(root, monkeypatch, built, once)
    assert _resume_over_http(once).status_code == 200
    lifted = _row(once)
    with owner_door("api_task_resume"):
        assert owner_allowance_lift(lifted, root) == {}
    assert lift_allowance(lifted, {}) == {} and lifted["metadata"]["producer_allowance_lifted"]["ceiling_usd"] == CEILING
    # A restart's return rebuilds a sparse row from the durable record, here one older than the decision and still
    # carrying the number: the queue row's decision is kept, never the number re-armed.
    patch_document(root, once, lambda row: {**row, "metadata": {**lifted["metadata"], "root_cost_ceiling_usd": CEILING}})
    seed_row = {"id": once, "_attempt": 1, "metadata": dict(lifted["metadata"]), "chat_id": built["chat_id"]}
    rebuilt = saved_turn_record(seed_row, root)
    assert recorded_producer_ceiling(rebuilt) is None
    assert rebuilt["metadata"]["producer_allowance_lifted"] == lifted["metadata"]["producer_allowance_lifted"]


def test_a_lifted_turns_cost_stop_reads_no_producer_ceiling_in_its_live_stack_or_its_restore(root):
    from dataclasses import asdict

    from ouroboros import task_pacing
    from ouroboros.presence_carrier import lift_live_allowance
    from ouroboros.usage_accounting import usage_scope
    from ouroboros.usage_origin import UsageScope

    scope = UsageScope(drive_root=root, task_id="warm-1", root_task_id="warm-1", root_limit_usd=5.0,
                       root_cost_ceiling_usd=CEILING)
    with usage_scope(scope):
        ctx = SimpleNamespace(task_id="warm-1", task_metadata={"root_cost_ceiling_usd": CEILING}, task_contract={})
        assert task_pacing._cost_stop_inputs(ctx)["authority"] == task_pacing.COST_STOP_PRODUCER  # the control
        saved = task_pacing.resolve_task_cost_ceiling(ctx, None)
        assert saved.state == task_pacing.COST_CEILING_ACTIVE and saved.ceiling_usd == CEILING
        # A warm Resume's lift where the stack continues (its scope still carries the number): no producer stop.
        lift_live_allowance(ctx, {"by": {"actor": "owner", "door": "api_task_resume"}, "ceiling_usd": CEILING})
        inputs = task_pacing._cost_stop_inputs(ctx)
        assert inputs["authority"] == task_pacing.COST_STOP_NONE and inputs["root_cap_usd"] == 5.0
        assert "root_cost_ceiling_usd" not in ctx.task_metadata
        assert ctx._cost_ceiling.state != task_pacing.COST_CEILING_ACTIVE
        # A cold restore of the same work says what became of its saved stop.
        restored = task_pacing.restore_cost_ceiling(ctx, asdict(saved))
        assert "lifted by the owner's Resume" in restored.basis


def test_the_grantless_releases_lift_through_the_owners_door_only_and_a_lane_seed_is_judged_on_its_record(root, monkeypatch):
    """A held row, a never-started root and a start that never ran have no grant to record the decision on: the same one
    helper decides on the row itself, or on the record its start wrote when the row is a lane seed, and keeps it durably
    first. What a door's rollback restores: ``test_a_grantless_resume_whose_snapshot_never_landed_decided_nothing``."""
    from ouroboros.control_actor import mind_control, owner_door
    from ouroboros.presence_carrier import owner_resume_lift, saved_turn_record
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    complete = _moved_paused(root, monkeypatch, built, tid + "-complete")
    assert owner_resume_lift(complete, root) == {} and recorded_producer_ceiling(complete) == CEILING  # unbound
    with owner_door("api_task_resume"), mind_control("mind-1", "the owner said so"):
        assert owner_resume_lift(complete, root) == {} and recorded_producer_ceiling(complete) == CEILING
    assert recorded_producer_ceiling(load_task_result(root, tid + "-complete")) == CEILING  # neither of those wrote anything
    with owner_door("api_task_resume"):
        assert owner_resume_lift(complete, root) == {}
    assert recorded_producer_ceiling(complete) is None
    assert complete["metadata"]["producer_allowance_lifted"]["by"]["door"] == "api_task_resume"
    durable = load_task_result(root, tid + "-complete")
    assert recorded_producer_ceiling(durable) is None
    assert durable["metadata"]["producer_allowance_lifted"] == complete["metadata"]["producer_allowance_lifted"]
    # A returned lane seed carries no contract and no number: its durable record decides, and its rebuild keeps the lift.
    seeded = tid + "-seed"
    _moved_paused(root, monkeypatch, built, seeded)
    seed_row = {"id": seeded, "_attempt": 1, "chat_id": built["chat_id"],
                "metadata": {key: value for key, value in built["metadata"].items() if key != "root_cost_ceiling_usd"}}
    with owner_door("api_task_resume"):
        assert owner_resume_lift(seed_row, root) == {}
    assert seed_row["metadata"]["producer_allowance_lifted"]["ceiling_usd"] == CEILING
    assert recorded_producer_ceiling(load_task_result(root, seeded)) is None
    rebuilt = saved_turn_record(seed_row, root)
    assert rebuilt["task_contract"] == built["task_contract"] and recorded_producer_ceiling(rebuilt) is None


def test_a_warm_resumes_decision_reaches_its_live_stack_once_where_the_stack_continues(root):
    """A warm Resume leaves the row with its worker, so its grant on the fence carries the owner's decision to the stack
    (``consume_warm_resume``); a warm Resume with no decision, a mind's or an automatic one, changes nothing."""
    from ouroboros import owner_pause, task_pacing
    from ouroboros.owner_wait import consume_warm_resume
    from ouroboros.task_results import write_task_result
    from ouroboros.usage_accounting import usage_scope
    from ouroboros.usage_origin import UsageScope

    def released(task_id, **decision):
        write_task_result(root, task_id, "running", metadata={"root_cost_ceiling_usd": CEILING}, result="Task is running.")
        fence = owner_pause.install_fence(root, task_id, request_id="pause-" + task_id)[0]
        owner_pause.set_fence_state(root, task_id, fence_id=fence["fence_id"], state=owner_pause.FENCE_RELEASED,
                                    release_reason="owner_resume", resume_grant={"warm": True, "grant_id": "g", **decision})
        return SimpleNamespace(task_id=task_id, root_task_id=task_id, drive_root=root, budget_drive_root=root, task_attempt=1,
                               task_metadata={"root_cost_ceiling_usd": CEILING}, task_contract={})

    def scope(task_id):
        return UsageScope(drive_root=root, task_id=task_id, root_task_id=task_id, root_limit_usd=5.0,
                          root_cost_ceiling_usd=CEILING)

    with usage_scope(scope("warm-mind")):  # no decision on the grant: the producer's stop stays
        unlifted = released("warm-mind")
        assert consume_warm_resume(unlifted) is True and unlifted.task_metadata["root_cost_ceiling_usd"] == CEILING
        assert task_pacing._cost_stop_inputs(unlifted)["authority"] == task_pacing.COST_STOP_PRODUCER
    decision = {"by": {"actor": "owner", "door": "api_task_resume", "at": "t1"}, "ceiling_usd": CEILING}
    with usage_scope(scope("warm-owner")):
        lifted = released("warm-owner", producer_allowance_lifted=decision)
        assert consume_warm_resume(lifted) is True
        assert lifted.task_metadata == {"producer_allowance_lifted": decision}
        inputs = task_pacing._cost_stop_inputs(lifted)
        assert inputs["authority"] == task_pacing.COST_STOP_NONE and inputs["root_cap_usd"] == 5.0
        assert lifted._cost_ceiling.state != task_pacing.COST_CEILING_ACTIVE
        # Spent once: a second read finds the grant consumed and writes nothing new.
        assert consume_warm_resume(lifted) is True and lifted.task_metadata == {"producer_allowance_lifted": decision}


class _Repaused(BaseException):
    """The next budget check paused on the allowance: it ends the loop under test past its exception rails."""


def _warm_parked_turn(root, monkeypatch, door):
    """A real ``run_llm_loop`` of a moved turn over its allowance, under a settled owner Pause beside a running critic: its warm
    wait calls the real Resume door through ``door``, as the pool's wait would while the stack is parked."""
    from ouroboros import budget_pause, loop, owner_pause
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result
    from ouroboros.usage_origin import UsageScope
    from supervisor import queue, state, workers
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    metadata = {**built["metadata"], "root_cost_ceiling_usd": CEILING}
    write_task_result(root, tid, "running", metadata=metadata, chat_id=built["chat_id"], _is_direct_chat=True, source="presence",
                      task_contract=built["task_contract"], result="Task is running.")
    bind_task_to_project(root, tid, PROJECT, origin=_owner_task_origin(root, tid))
    fence = owner_pause.install_fence(root, tid, request_id="pause-warm")[0]
    owner_pause.set_fence_state(root, tid, fence_id=fence["fence_id"], state=owner_pause.FENCE_PAUSED)
    # The lane's row: the seed's sparse metadata (no number, the same event identity), parked warm with its slot.
    lane_metadata = {key: built["metadata"][key] for key in ("presence", "presence_event_identity")}
    meta = {"task": {"id": tid, "type": "presence", "source": "presence", "chat_id": built["chat_id"], "_is_direct_chat": True,
                     "_presence_turn": True, "root_task_id": tid, "metadata": lane_metadata},
            "worker_id": 90, "attempt": 1, "owner_wait": {"state": "waiting", "reason": "owner_pause"}}
    workers.RUNNING[tid] = meta
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    outcomes = []

    def wait(_ctx, _checkpoint):
        with door():
            outcomes.append(queue.resume_budget_paused_task(tid))
        assert outcomes[-1]["ok"] and outcomes[-1]["warm"], outcomes
        return "control:owner_resume"

    registry = _registry(root, monkeypatch, tid, 1)
    ctx = registry._ctx
    ctx.task_metadata, ctx.owner_wait_callback, ctx.review_wait_callback = dict(metadata), wait, None
    ctx.root_task_id, ctx.budget_drive_root, ctx._cost_ceiling = tid, root, None  # the loop resolves its start's stop
    monkeypatch.setattr("ouroboros.review_pause.live_detached_operations",
                        lambda task_id: [{"owner_id": "critic-1"}] if not outcomes else [])
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_k: {"settled_usd": 0.5})  # known spend is past the allowance
    real_pause = budget_pause.request_pause

    def pause(limit, **kwargs):
        if kwargs["rail"] == budget_pause.RAIL_GRACEFUL_CEILING:
            raise _Repaused(kwargs["reason_text"])
        return real_pause(limit, **kwargs)

    monkeypatch.setattr(budget_pause, "request_pause", pause)
    rounds = []

    def dispatch(_call, _disposition, **_kwargs):
        rounds.append(1)
        if len(rounds) == 1:
            call = {"id": "call-1", "type": "function", "function": {"name": "repo_read", "arguments": "{}"}}
            return {"role": "assistant", "content": "", "tool_calls": [call]}, 0.0
        return {"role": "assistant", "content": "Finished on the project budget.", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(registry, "execute_result", lambda *_a, **_k: "ok")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    args = {**_loop_kwargs(root, registry, []), "task_id": tid, "budget_remaining_usd": 10.0}
    args["llm"] = SimpleNamespace(default_model=lambda: "m")
    scope = UsageScope(drive_root=root, task_id=tid, root_task_id=tid, root_limit_usd=5.0, root_cost_ceiling_usd=CEILING)
    return tid, built, metadata, meta, args, scope, loop


def test_the_loops_next_budget_check_and_every_later_record_read_the_owners_warm_resume(root, monkeypatch):
    """Real door, real warm consumption, real loop: the stack over its $0.40 allowance runs on, and the durable row, the
    running row, its published snapshot and any rebuild keep the decision, never the number; a member reads it too."""
    from ouroboros import task_pacing
    from ouroboros.control_actor import owner_door
    from ouroboros.presence_carrier import saved_turn_record
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.usage_accounting import usage_scope

    tid, built, metadata, meta, args, scope, loop = _warm_parked_turn(root, monkeypatch, lambda: owner_door("api_task_resume"))
    with usage_scope(scope):
        text, _usage, _trace = loop.run_llm_loop(**args)
    assert text == "Finished on the project budget."
    stored = load_task_result(root, tid)
    decision = stored["metadata"]["producer_allowance_lifted"]
    assert decision["ceiling_usd"] == CEILING and decision["by"]["door"] == "api_task_resume"
    assert task_pacing.recorded_producer_ceiling(stored) is None
    assert stored["owner_pause"]["resume_grant"]["producer_allowance_lifted"] == decision  # one write, one decision
    assert meta["task"]["metadata"]["producer_allowance_lifted"] == decision
    assert task_pacing.recorded_producer_ceiling(meta["task"]) is None
    [published] = [row for row in load_queue_document(root)["running"] if row["id"] == tid]
    assert published["task"]["metadata"]["producer_allowance_lifted"] == decision
    # A restart rebuilds from the durable record, whatever an older row it holds still says.
    rebuilt = saved_turn_record({"id": tid, "metadata": dict(metadata)}, root)
    assert task_pacing.recorded_producer_ceiling(rebuilt) is None
    assert rebuilt["metadata"]["producer_allowance_lifted"] == decision
    # A later park carries the stack's own older task dict: the running row's decision wins over its number.
    carried = saved_turn_record(meta["task"], root, carried={**built, "metadata": dict(metadata)})
    assert task_pacing.recorded_producer_ceiling(carried) is None
    # The stack's terminal writes its own older task dict over the row: the decision stays for a Continue to read.
    write_task_result(root, tid, "completed", metadata=dict(metadata), result="Done.", root_cost_ceiling_usd=CEILING)
    ended = load_task_result(root, tid)
    assert task_pacing.recorded_producer_ceiling(ended) is None and ended["metadata"]["producer_allowance_lifted"] == decision
    # A member of the tree still carrying the number reads its root's row, which no longer authors it.
    assert task_pacing._root_cost_stop_authority(scope) == (task_pacing.COST_STOP_NONE, None)


@pytest.mark.parametrize("door", ["mind", "automatic"])
def test_a_minds_or_an_automatic_warm_resume_keeps_the_allowance_and_the_next_check_pauses_on_it(root, monkeypatch, door):
    import contextlib

    from ouroboros import task_pacing
    from ouroboros.control_actor import mind_control, owner_door
    from ouroboros.task_results import load_task_result
    from ouroboros.usage_accounting import usage_scope

    @contextlib.contextmanager
    def minds():
        with owner_door("api_task_resume"), mind_control("mind-1", "the owner asked me to resume it"):
            yield

    opened = minds if door == "mind" else contextlib.nullcontext
    tid, _built, _metadata, meta, args, scope, loop = _warm_parked_turn(root, monkeypatch, opened)
    with usage_scope(scope), pytest.raises(_Repaused):
        loop.run_llm_loop(**args)
    stored = load_task_result(root, tid)
    assert task_pacing.recorded_producer_ceiling(stored) == CEILING
    assert "producer_allowance_lifted" not in stored["metadata"]
    assert "producer_allowance_lifted" not in stored["owner_pause"]["resume_grant"]
    assert "producer_allowance_lifted" not in meta["task"]["metadata"]
    assert task_pacing._root_cost_stop_authority(scope) == (task_pacing.COST_STOP_PRODUCER, CEILING)


def test_a_newer_pause_before_the_stack_took_up_the_owners_lift_leaves_it_with_the_next_warm_grant(root, monkeypatch):
    """The decision lands on the durable row with its release, so a later warm grant finds no allowance to lift. The stack
    a newer Pause parked before it consumed the first grant still takes the decision up from the next one, whoever gives it."""
    from ouroboros import owner_pause, task_pacing
    from ouroboros.control_actor import mind_control, owner_door
    from ouroboros.owner_wait import consume_warm_resume
    from ouroboros.task_results import load_task_result
    from ouroboros.usage_accounting import usage_scope
    from supervisor import queue

    tid, _built, metadata, _meta, _args, scope, _loop = _warm_parked_turn(root, monkeypatch, lambda: owner_door("api_task_resume"))
    with owner_door("api_task_resume"):
        assert queue.resume_budget_paused_task(tid)["warm"] is True
    decision = load_task_result(root, tid)["metadata"]["producer_allowance_lifted"]
    # A newer Pause closes the fence before the parked stack consumed that grant: the stack parks again, unlifted.
    ctx = SimpleNamespace(task_id=tid, root_task_id=tid, drive_root=root, budget_drive_root=root, task_attempt=1,
                          task_metadata=dict(metadata), task_contract={})
    fence = owner_pause.install_fence(root, tid, request_id="pause-newer")[0]
    owner_pause.set_fence_state(root, tid, fence_id=fence["fence_id"], state=owner_pause.FENCE_PAUSED)
    with usage_scope(scope):
        assert consume_warm_resume(ctx) is False and ctx.task_metadata["root_cost_ceiling_usd"] == CEILING
        with mind_control("mind-1", "resume it"):
            assert queue.resume_budget_paused_task(tid)["warm"] is True
        assert consume_warm_resume(ctx) is True
        assert ctx.task_metadata["producer_allowance_lifted"] == decision
        assert ctx._cost_ceiling.state == task_pacing.COST_CEILING_DISABLED


# --- The decision persists: Stop, Continue and restore read the durable row, whichever door released the row. ---

GRANTLESS = ("held", "unstarted_root", "start_never_ran")
_ACTORS = ("owner", "mind", "automatic")


class _Crash(BaseException):
    """The process dies at this write: nothing after it runs, past every exception rail."""


def _focus_built(root, monkeypatch):
    """A Place focus's own built task, from the real Place admission and ``_build_task``: its allowance, its kind and the
    opening a Continue reads are the producer's."""
    from dataclasses import replace

    from ouroboros import places
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_runner import PresenceTurnEvent, _build_task

    places.connect_place(root, connector="bridge", provider="chat", account_id="a", conversation_id="r", wakes=["mention"])
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance",
                        lambda _root: {"status": "available", "remaining_usd": CEILING, "settled_usd": 1})
    monkeypatch.setattr("ouroboros.presence_context._own_work_section", lambda *_args: "")
    event = PresenceTurnEvent("event1", "chat", "a", "r", "", "chat:a:r", {"platform_actor_id": "person", "display_name": "Person"},
                              {}, {}, "Words of the room", facts=("mention",))
    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="bridge", provider="chat", account_id="a",
                                    conversation_id="r")
    built = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": CEILING}), event, drive_root=root,
                        staged_files=())
    assert built["metadata"]["root_cost_ceiling_usd"] == CEILING and built["metadata"]["work_kind"] == "consciousness"
    return built


def _stored_turn(root, built, task_id, status="scheduled", *, move=True):
    """That turn's durable row, moved by the owner to a Project (``move=False``: never moved)."""
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result

    metadata = dict(built["metadata"])
    write_task_result(root, task_id, status, metadata=metadata, chat_id=built["chat_id"], _is_direct_chat=True,
                      source="presence", root_task_id=task_id, task_contract=built["task_contract"], result="Accepted.")
    if move:
        bind_task_to_project(root, task_id, PROJECT, origin=_owner_task_origin(root, "no-captured-origin"))
    return metadata


def _saved_record_row(built, task_id, metadata, **facts):
    """The complete record a queue keeps of that turn (``saved_turn_record``'s own fact), its allowance still on it."""
    record = {key: value for key, value in built.items() if key not in {"id", "root_task_id", "_attempt"}}
    return {**record, "id": task_id, "root_task_id": task_id, "_attempt": 1, "_presence_turn": True,
            "metadata": {**metadata, "presence_request_waits": False}, **facts}


def _behind_grantless_door(root, built, task_id, door, *, move=True):
    """A never-started moved turn the queue keeps behind one of the releases that mint no grant: a held row, a root its
    owner Pause caught while queued, and a start parked before it ran its body."""
    from supervisor import workers
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.restart_retention import hold_stopped_queue

    row = _saved_record_row(built, task_id, _stored_turn(root, built, task_id, move=move), admitted_dispatch="none")
    if door == "start_never_ran":
        row["_budget_pause"] = {"status": "paused_before_dispatch", "scope": "global", "replay_safe": True,
                                "physical_calls": 0, "auto_resume": False}
    workers.PENDING.append(row)
    if door == "held":
        hold_stopped_queue(row, root, "app_stop")
    elif door == "unstarted_root":
        assert request_owner_pause(task_id, request_id="pause-" + task_id)["ok"]
    return row


def _resume_as(actor, task_id):
    """The Resume of ``actor``: the owner's real door, a mind's control inside it, or an unbound automatic call."""
    from ouroboros.control_actor import mind_control, owner_door
    from supervisor import queue

    if actor == "owner":
        return _resume_over_http(task_id).json()
    if actor == "owner_in_process":  # its door, past the HTTP client's own exception rails
        with owner_door("api_task_resume"):
            return queue.resume_budget_paused_task(task_id)
    if actor == "mind":
        with owner_door("api_task_resume"), mind_control("mind-1", "the owner asked me to resume it"):
            return queue.resume_budget_paused_task(task_id)
    return queue.resume_budget_paused_task(task_id)


def _start_authority(root, row):
    """Who authors the cost stop this row's own start resolves, by the real scope and resolver of a worker's start."""
    from ouroboros import task_pacing
    from ouroboros.usage_accounting import usage_scope
    from ouroboros.usage_origin import task_usage_scope

    scope = task_usage_scope(row, drive_root=root, root_task_id=row["id"], parent_task_id="", billing={"root_limit_usd": 5.0})
    ctx = SimpleNamespace(task_metadata=dict(row["metadata"]), task_contract={})
    with usage_scope(scope):
        task_pacing.resolve_task_cost_ceiling(ctx, 1000.0)
        assert task_pacing._cost_stop_inputs(ctx)["root_cap_usd"] == 5.0, "the tree's cap is the grant's own check: it stays"
        return task_pacing.cost_stop_authority(ctx)


def _technical_stop(root, task_id):
    """A Stop of the queued row a Continue may follow: the owner's Restart recorded this cause (an HTTP Stop refuses one)."""
    from ouroboros.cancel_intents import request_cancel
    from supervisor import queue

    request_cancel(root, task_id, reason="Owner restart", source="owner_restart", requested_by="owner",
                   requested_stop_policy="immediate")
    assert queue.cancel_task_custody(task_id) == queue.CANCEL_CANCELLED


def _continue(task_id):
    from supervisor import workers
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    ack = admit_continuation(task_id, action_nonce=NONCE)
    assert ack["ok"] is True, ack
    return next(row for row in workers.PENDING if row["id"] == ack["successor_task_id"])


@pytest.mark.parametrize("actor", _ACTORS)
@pytest.mark.parametrize("door", GRANTLESS)
def test_a_grantless_resume_decides_for_the_stop_and_continue_that_follow_it(root, monkeypatch, door, actor):
    """The real door, Stop custody and Continue admission: the owner's decision is on the durable row before any start, so
    the Continue of a turn stopped still queued finds no number; a mind's or an automatic Resume keeps it to the end."""
    from ouroboros import task_pacing
    from ouroboros.task_results import load_task_result
    from supervisor import state

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    row = _behind_grantless_door(root, built, tid, door)
    resumed = _resume_as(actor, tid)
    assert resumed.get("ok") is True, resumed
    owner = actor == "owner"
    durable = load_task_result(root, tid)
    assert (task_pacing.recorded_producer_ceiling(durable) is None) is owner
    assert (task_pacing.recorded_producer_ceiling(row) is None) is owner
    if owner:
        decision = durable["metadata"]["producer_allowance_lifted"]
        assert decision["ceiling_usd"] == CEILING and decision["by"]["door"] == "api_task_resume"
        assert row["metadata"]["producer_allowance_lifted"] == decision
    else:
        assert "producer_allowance_lifted" not in durable["metadata"] and "producer_allowance_lifted" not in row["metadata"]
    # A member of the tree reads its root's durable row, which no longer authors a number (or still does).
    authored = task_pacing._root_cost_stop_authority(SimpleNamespace(drive_root=root, root_task_id=tid))
    assert authored == ((task_pacing.COST_STOP_NONE, None) if owner else (task_pacing.COST_STOP_PRODUCER, CEILING))
    expected = task_pacing.COST_STOP_NONE if owner else task_pacing.COST_STOP_PRODUCER
    assert _start_authority(root, row) == expected

    _technical_stop(root, tid)
    stopped = load_task_result(root, tid)
    assert stopped["status"] == "cancelled" and stopped["cancel_origin"]["source"] == "owner_restart"
    assert (task_pacing.recorded_producer_ceiling(stopped) is None) is owner

    successor = _continue(tid)
    assert task_pacing.recorded_producer_ceiling(successor) == (None if owner else CEILING)
    assert _start_authority(root, successor) == expected
    stored_successor = load_task_result(root, successor["id"])
    assert task_pacing.recorded_producer_ceiling(stored_successor) == (None if owner else CEILING)


@pytest.mark.parametrize("door,work", [(door, work) for door in GRANTLESS for work in ("unmoved", "owner_work")
                                       if (door, work) != ("unstarted_root", "unmoved")])  # an unmoved turn has no Pause
def test_the_owners_grantless_resume_of_ordinary_work_decides_nothing(root, monkeypatch, door, work):
    """The owner's own door is not enough: an unmoved turn, or owner work at entry (C8), is no moved autonomous work."""
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from supervisor import state

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    if work == "owner_work":
        built["metadata"]["work_kind"] = "owner"
    row = _behind_grantless_door(root, built, tid, door, move=work != "unmoved")
    resumed = _resume_as("owner", tid)
    assert resumed.get("ok") is True, resumed
    stored = load_task_result(root, tid)
    assert recorded_producer_ceiling(row) == CEILING and "producer_allowance_lifted" not in row["metadata"]
    assert recorded_producer_ceiling(stored) == CEILING and "producer_allowance_lifted" not in stored["metadata"]


def test_an_owners_http_stop_keeps_the_decision_and_its_continue_stays_refused(root, monkeypatch):
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from supervisor import queue, state
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    _behind_grantless_door(root, built, tid, "held")
    assert _resume_as("owner", tid)["ok"] is True
    request_cancel(root, tid, reason="Stop", source="http_single", requested_by="owner", requested_stop_policy="immediate")
    assert queue.cancel_task_custody(tid) == queue.CANCEL_CANCELLED
    stopped = load_task_result(root, tid)
    assert recorded_producer_ceiling(stopped) is None and stopped["metadata"]["producer_allowance_lifted"]["ceiling_usd"] == CEILING
    assert admit_continuation(tid, action_nonce=NONCE)["error"] == "stopped_by_owner"


def _still_behind_its_door(root, row, door, tid):
    """Nothing moved, anywhere: the row, its durable record and what keeps it from dispatch are as the Resume found them."""
    from ouroboros import owner_pause
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from supervisor import queue
    from supervisor.events_budget import SELECTION_KEY
    from supervisor.queue_transitions import budget_pause_fact

    durable = load_task_result(root, tid)
    assert recorded_producer_ceiling(row) == CEILING and "producer_allowance_lifted" not in row["metadata"]
    assert recorded_producer_ceiling(durable) == CEILING and "producer_allowance_lifted" not in durable["metadata"]
    assert budget_pause_fact(row) is not None and SELECTION_KEY not in row
    if door == "unstarted_root":
        assert owner_pause.fence_closed(owner_pause.read_fence(root, tid)) and tid in queue.BUDGET_ROOT_FENCES


@pytest.mark.parametrize("door", ["held", "unstarted_root"])
def test_a_grantless_resume_whose_snapshot_never_landed_decided_nothing(root, monkeypatch, door):
    from supervisor import queue, state

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    row = _behind_grantless_door(root, built, tid, door)
    with monkeypatch.context() as patched:
        patched.setattr(queue, "publish_queue_snapshot", lambda **_k: False)
        assert _resume_as("owner", tid)["error"] == "snapshot_not_persisted"
    _still_behind_its_door(root, row, door, tid)


@pytest.mark.parametrize("door", GRANTLESS)
def test_a_store_that_refuses_the_owners_decision_refuses_the_resume_and_the_next_one_decides(root, monkeypatch, door):
    """No release stands without its decision and no decision without its release: a refused write leaves both as they were,
    the snapshot included, and answers the owner typed; the same Resume repeated then lands both."""
    from ouroboros import task_store
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from supervisor import state

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    row = _behind_grantless_door(root, built, tid, door)

    def refused(*_a, **_k):
        raise OSError("store refused")

    with monkeypatch.context() as patched:
        patched.setattr(task_store, "patch_document", refused)
        answer = _resume_as("owner", tid)
    assert answer.get("ok") is not True, answer
    assert answer["error"] in {"grant_not_recorded", "owner_pause_fence_unwritable"}
    _still_behind_its_door(root, row, door, tid)
    for saved in [item for item in load_queue_document(root).get("pending") or [] if item["id"] == tid]:
        assert recorded_producer_ceiling(saved["task"]) == CEILING, saved
    assert _resume_as("owner", tid)["ok"] is True
    assert recorded_producer_ceiling(row) is None
    assert recorded_producer_ceiling(load_task_result(root, tid)) is None


def _boot_from_disk(queue, workers):
    """The next process: nothing in memory, the snapshot as the last publication left it."""
    workers.PENDING.clear()
    workers.RUNNING.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    queue.restore_pending_from_snapshot()
    return list(workers.PENDING)


def _warm_parked_saved_record(tmp_path, monkeypatch, *, saved_record=True):
    """A moved turn parked warm under the owner's Pause, the snapshot on disk holding its RUNNING row with the number: the real
    park (``test_warm_restart_retention``); its row a complete record a crash keeps as it is, or its lane's seed."""
    from ouroboros import owner_pause
    from ouroboros.queue_observation import load_queue_document
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_restart_retention import _pool_events
    from tests.test_warm_restart_retention import _park

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _pool_events(workers, monkeypatch)
    built = _focus_built(tmp_path, monkeypatch)
    _ctx, _limit, _wait, task = _park(tmp_path, workers, "owner_pause")
    tid = task["id"]
    metadata = _stored_turn(tmp_path, built, tid, status="running")
    owner_pause.set_fence_state(tmp_path, tid, fence_id=owner_pause.read_fence(tmp_path, tid)["fence_id"],
                                state=owner_pause.FENCE_PAUSED)
    if saved_record:
        task.update(_saved_record_row(built, tid, metadata))
    else:
        task.update(type="presence", source="presence", _presence_turn=True, _is_direct_chat=True, chat_id=built["chat_id"],
                    task_contract={"objective": "a lane's default contract"},
                    metadata={key: built["metadata"][key] for key in ("presence", "presence_event_identity")})
    workers.RUNNING[tid]["owner_wait"] = {"state": "waiting", "reason": "owner_pause"}
    assert queue.persist_queue_snapshot(reason="before_the_release")
    assert [row["id"] for row in load_queue_document(tmp_path)["running"]] == [tid]
    return queue, workers, built, tid


def test_a_crash_between_the_warm_release_and_its_snapshot_restores_the_durable_decision(subtests, tmp_path, monkeypatch):
    """Real warm door, real restore over the snapshot the crash left. A complete record is no rebuild's input, so it takes the
    owner's decision from the durable row before any gate reads it; a lane's seed rebuilds from that row. A mind's or an
    automatic Resume left none, and the number stands. The next restart, start, Stop and Continue read the same."""
    case_paths = case_folders(tmp_path)
    for record in ["saved_record", "lane_seed"]:
        for actor in _ACTORS:
            tmp_path = next(case_paths)
            with subtests.test(record=record, actor=actor), monkeypatch.context() as monkeypatch:
                from ouroboros import owner_pause, task_pacing
                from ouroboros.control_actor import owner_door
                from ouroboros.presence_carrier import _saved_record
                from ouroboros.task_results import load_task_result

                queue, workers, built, tid = _warm_parked_saved_record(tmp_path, monkeypatch, saved_record=record == "saved_record")
                owner = actor == "owner"

                def dies(**_k):
                    raise _Crash()

                with monkeypatch.context() as patched, pytest.raises(_Crash):
                    patched.setattr(queue, "publish_queue_snapshot", dies)
                    if owner:
                        with owner_door("api_task_resume"):
                            queue.resume_budget_paused_task(tid)
                    else:
                        _resume_as(actor, tid)
                durable = load_task_result(tmp_path, tid)
                assert owner_pause.read_fence(tmp_path, tid)["state"] == owner_pause.FENCE_RELEASED
                assert (task_pacing.recorded_producer_ceiling(durable) is None) is owner

                admission = queue.enqueue_task
                gated = []

                def admitted(task, **kwargs):  # what the restore's own admission gate reads
                    gated.append(task_pacing.recorded_producer_ceiling(task))
                    return admission(task, **kwargs)

                with monkeypatch.context() as patched:
                    patched.setattr(queue, "enqueue_task", admitted)
                    [row] = [task for task in _boot_from_disk(queue, workers) if task["id"] == tid]
                assert gated == [None if owner else CEILING], "reconciled before the gate, not after it"
                assert _saved_record(row)
                assert row["task_contract"]["objective"] == built["task_contract"]["objective"]
                assert task_pacing.recorded_producer_ceiling(row) == (None if owner else CEILING)
                if owner:
                    assert row["metadata"]["producer_allowance_lifted"] == durable["metadata"]["producer_allowance_lifted"]
                else:
                    assert "producer_allowance_lifted" not in row["metadata"]
                expected = task_pacing.COST_STOP_NONE if owner else task_pacing.COST_STOP_PRODUCER
                assert _start_authority(tmp_path, row) == expected
                assert queue.persist_queue_snapshot(reason="after_the_restore")
                [again] = [task for task in _boot_from_disk(queue, workers) if task["id"] == tid]
                assert task_pacing.recorded_producer_ceiling(again) == (None if owner else CEILING)

                _technical_stop(tmp_path, tid)
                successor = _continue(tid)
                assert task_pacing.recorded_producer_ceiling(successor) == (None if owner else CEILING)
                assert _start_authority(tmp_path, successor) == expected


def test_a_crash_between_the_exact_grant_and_its_snapshot_leaves_the_decision_and_a_minds_next_resume_runs_on_it(root, monkeypatch):
    """The exact grant writes the decision in its own compare-and-set. Its snapshot never landed, so the restored row is the
    older record: it takes the decision; the orphaned grant is revoked at the next Resume, a mind's, which decides nothing of
    its own and starts on the owner's decision. An automatic crash in the same window left no decision to take."""
    from ouroboros import budget_pause, task_pacing
    from ouroboros.control_actor import mind_control, owner_door
    from ouroboros.task_results import load_task_result
    from supervisor import queue, state, workers

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _seed, built = _seed_and_built(root)
    tid, unbound = built["id"], built["id"] + "-unbound"
    for task_id in (tid, unbound):
        task = _moved_paused(root, monkeypatch, built, task_id)
        task["_presence_turn"] = True
    assert queue.persist_queue_snapshot(reason="before_the_grant")

    def dies(**_k):
        raise _Crash()

    with monkeypatch.context() as patched:
        patched.setattr(queue, "publish_queue_snapshot", dies)
        with owner_door("api_task_resume"), pytest.raises(_Crash):
            queue.resume_budget_paused_task(tid)
        with pytest.raises(_Crash):
            queue.resume_budget_paused_task(unbound)
    stored = load_task_result(root, tid)
    decision = stored["metadata"]["producer_allowance_lifted"]
    assert stored["budget_pause"]["grant"]["producer_allowance_lifted"] == decision
    assert task_pacing.recorded_producer_ceiling(load_task_result(root, unbound)) == CEILING

    restored = {task["id"]: task for task in _boot_from_disk(queue, workers)}
    assert task_pacing.recorded_producer_ceiling(restored[tid]) is None
    assert restored[tid]["metadata"]["producer_allowance_lifted"] == decision
    assert task_pacing.recorded_producer_ceiling(restored[unbound]) == CEILING
    with mind_control("mind-1", "resume it"):
        outcome = queue.resume_budget_paused_task(tid)
    assert outcome["ok"] is True, outcome
    grant = budget_pause.budget_pause_row(root, tid)["grant"]
    assert "producer_allowance_lifted" not in grant, "the mind's own grant decides nothing"
    assert task_pacing.recorded_producer_ceiling(restored[tid]) is None
    assert _start_authority(root, restored[tid]) == task_pacing.COST_STOP_NONE
    assert load_task_result(root, tid)["metadata"]["producer_allowance_lifted"] == decision


@pytest.mark.parametrize("door", GRANTLESS)
def test_a_crash_after_a_grantless_decision_restores_it_on_the_saved_record_the_last_snapshot_kept(root, monkeypatch, door):
    """The decision is durable before the queue row takes it, so the snapshot may be older than the decision: the restore
    reads the durable row. An automatic Resume in the same window finds no decision to take."""
    from ouroboros.presence_carrier import _saved_record
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from supervisor import queue, state, workers

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    row = _behind_grantless_door(root, built, tid, door)
    assert queue.persist_queue_snapshot(reason="before_the_resume")
    published = []

    def older(**_k):  # the release's publication lands, nothing later does; the file stays older than the decision
        published.append(_k)
        return True

    with monkeypatch.context() as patched:
        patched.setattr(queue, "publish_queue_snapshot", older)
        patched.setattr(queue, "persist_queue_snapshot", older)
        assert _resume_as("owner", tid)["ok"] is True
    decision = load_task_result(root, tid)["metadata"]["producer_allowance_lifted"]
    assert row["metadata"]["producer_allowance_lifted"] == decision
    [saved] = [item for item in load_queue_document(root)["pending"] if item["id"] == tid]
    assert recorded_producer_ceiling(saved["task"]) == CEILING, "the file is older than the decision"
    [restored] = [task for task in _boot_from_disk(queue, workers) if task["id"] == tid]
    assert _saved_record(restored) and recorded_producer_ceiling(restored) is None
    assert restored["metadata"]["producer_allowance_lifted"] == decision


@pytest.mark.parametrize("door", GRANTLESS)
def test_a_crash_before_a_grantless_decision_is_durable_decided_nothing_and_the_owner_can_still_decide(root, monkeypatch, door):
    """The write that would carry the decision never happened: no row claims it, whatever the last snapshot says of the
    release, and the owner's next Resume decides as if this were the first."""
    from ouroboros import task_store
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from supervisor import queue, state, workers
    from supervisor.queue_transitions import budget_pause_fact

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    _behind_grantless_door(root, built, tid, door)
    assert queue.persist_queue_snapshot(reason="before_the_resume")

    def dies(*_a, **_k):
        raise _Crash()

    with monkeypatch.context() as patched, pytest.raises(_Crash):
        patched.setattr(task_store, "patch_document", dies)
        _resume_as("owner_in_process", tid)
    assert recorded_producer_ceiling(load_task_result(root, tid)) == CEILING
    [saved] = [item for item in load_queue_document(root)["pending"] if item["id"] == tid]
    assert recorded_producer_ceiling(saved["task"]) == CEILING and "producer_allowance_lifted" not in saved["task"]["metadata"]

    [row] = [task for task in _boot_from_disk(queue, workers) if task["id"] == tid]
    assert recorded_producer_ceiling(row) == CEILING and "producer_allowance_lifted" not in row["metadata"]
    assert budget_pause_fact(row) is not None, "nothing starts on a Resume no one was told of"
    resumed = _resume_as("owner", tid)
    assert resumed.get("ok") is True, resumed
    assert recorded_producer_ceiling(row) is None
    assert recorded_producer_ceiling(load_task_result(root, tid)) is None


def test_the_restores_reconciliation_reads_only_its_own_events_durable_decision_and_needs_no_binding(root, monkeypatch):
    """A complete row takes the decision whatever else this boot can read of it (an unreadable binding is not a move, but the
    decision is its own fact), never another event's, and a durable row with no decision changes nothing."""
    from ouroboros.presence_carrier import adopt_saved_records
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_store import patch_document

    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    metadata = _stored_turn(root, built, tid)
    decision = {"by": {"actor": "owner", "door": "api_task_resume", "at": "t1"}, "ceiling_usd": CEILING}

    def durable(*, identity=None):
        kept = {key: value for key, value in metadata.items() if key != "root_cost_ceiling_usd"}
        kept.update(producer_allowance_lifted=decision, **({"presence_event_identity": identity} if identity else {}))
        patch_document(root, tid, lambda row: {**row, "metadata": kept})

    row = _saved_record_row(built, tid, metadata)
    untouched = dict(row)
    adopt_saved_records([row], root)
    assert row == untouched, "no decision recorded: nothing to take"
    durable(identity="another-event")
    adopt_saved_records([row], root)
    assert recorded_producer_ceiling(row) == CEILING and "producer_allowance_lifted" not in row["metadata"]
    durable()
    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.projects_registry.project_binding_for_task",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("registry unreadable")))
        adopt_saved_records([row], root)
    assert recorded_producer_ceiling(row) is None and row["metadata"]["producer_allowance_lifted"] == decision
    adopt_saved_records([row], root)
    assert row["metadata"]["producer_allowance_lifted"] == decision


@pytest.mark.parametrize("door", GRANTLESS)
def test_a_models_selection_of_a_grantless_row_decides_nothing_whatever_it_answers(root, monkeypatch, door):
    from ouroboros.control_actor import mind_control
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from supervisor import queue, state

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    row = _behind_grantless_door(root, built, tid, door)
    with mind_control("parent-1", "selected"):
        queue.resume_budget_paused_task(tid, selected_by="parent-1")
    stored = load_task_result(root, tid)
    assert recorded_producer_ceiling(row) == CEILING and "producer_allowance_lifted" not in row["metadata"]
    assert recorded_producer_ceiling(stored) == CEILING and "producer_allowance_lifted" not in stored["metadata"]


@pytest.mark.parametrize("door,veto", [(door, "stop_pending") for door in GRANTLESS] + [("held", "dispatch_unknown")])
def test_a_resume_its_door_refuses_decides_nothing(root, monkeypatch, door, veto):
    """The decision follows the door's own checks, so whatever kept this work from running still does: a Stop already recorded,
    or, for a held row, a dispatch nobody can prove never happened."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result
    from supervisor import state

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    built = _focus_built(root, monkeypatch)
    tid = built["id"]
    row = _behind_grantless_door(root, built, tid, door)
    if veto == "stop_pending":
        request_cancel(root, tid, reason="Stop", source="http_single", requested_by="owner", requested_stop_policy="immediate")
    else:
        row["admitted_dispatch"] = "possible"
    answer = _resume_as("owner", tid)
    assert answer.get("ok") is not True, answer
    stored = load_task_result(root, tid)
    assert recorded_producer_ceiling(row) == CEILING and "producer_allowance_lifted" not in row["metadata"]
    assert recorded_producer_ceiling(stored) == CEILING and "producer_allowance_lifted" not in stored["metadata"]
