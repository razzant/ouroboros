"""The members of a moved Presence turn's tree that the owner's Pause parked warm, after the owner's own Resume (#2566).

The owner's Resume moves the moved turn off its producer allowance (``test_presence_moved_allowance``). A member that
resolved that allowance before the Resume and kept its stack warm reads its root's decision where it continues
(``owner_wait.consume_warm_resume``), as its cold restore would; its own explicit limit, and any Resume that is not the
owner's own, keep the stop.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_presence_moved_allowance import _Repaused, _warm_parked_turn
from tests.test_presence_moved_lifecycle import CEILING, root as root


@pytest.mark.parametrize("wake", ["warm_root_grant", "cold_root_release", "selected_member"])
def test_an_already_warm_member_reads_its_roots_owner_lift_where_it_continues_and_keeps_its_own_limit(
        root, monkeypatch, wake):
    """Review R2 F2 (``.review-evidence/2566-review2-astra/test_warm_descendant_allowance.py``): a member that
    resolved its root's producer allowance before the owner's Resume continues on the decision the root's row keeps,
    whichever path wakes it: the warm root's own grant (the real door and loop), the release a cold root's
    consumption writes with no grant on the fence, or the owner's selection of the member under a fence that stays
    closed. Its stop is read again from the root's row, never labelled as its own lift; an explicit limit of its own
    stays as it was."""
    from ouroboros import owner_pause, task_pacing
    from ouroboros.control_actor import owner_door
    from ouroboros.loop_budget import _resolve_task_cost_ceiling
    from ouroboros.owner_wait import consume_warm_resume
    from ouroboros.presence_carrier import lifted_record
    from ouroboros.task_results import load_task_result, stamp_task_result_schema
    from ouroboros.task_store import patch_document
    from ouroboros.usage_accounting import usage_scope
    from ouroboros.usage_origin import UsageScope

    tid, _built, _metadata, _meta, args, root_scope, loop = _warm_parked_turn(
        root, monkeypatch, lambda: owner_door("api_task_resume"))
    members = {}
    for task_id, contract in (("warm-child", {}), ("warm-explicit", {"budget_profile": {"cost_hard_stop_pct": 50}})):
        ctx = SimpleNamespace(task_id=task_id, root_task_id=tid, task_attempt=1, drive_root=root,
                              budget_drive_root=root, task_metadata={"root_task_id": tid}, task_contract=contract)
        scope = UsageScope(drive_root=root, task_id=task_id, root_task_id=tid, root_limit_usd=5.0,
                           root_cost_ceiling_usd=CEILING)
        with usage_scope(scope):
            members[task_id] = (ctx, scope, _resolve_task_cost_ceiling(ctx, 10.0))  # its start's stop, kept warm
    inherited, explicit = members["warm-child"][2], members["warm-explicit"][2]
    assert (inherited.state, inherited.ceiling_usd, inherited.basis) == (task_pacing.COST_CEILING_ACTIVE, CEILING,
                                                                         "root_resolved_ceiling")
    assert explicit.state == task_pacing.COST_CEILING_ACTIVE and explicit.basis != inherited.basis
    if wake == "warm_root_grant":
        with usage_scope(root_scope):
            assert loop.run_llm_loop(**args)[0] == "Finished on the project budget."
        assert owner_pause.read_fence(root, tid)["resume_grant"]["producer_allowance_lifted"]
    else:
        # A cold root's exact grant kept the owner's decision on its row (``owner_resume_lift``); its consumption
        # released the fence with no grant on it, or the fence stays closed and the owner selected the members.
        decision = {"by": {"actor": "owner", "door": "api_task_resume", "at": "t1"}, "ceiling_usd": CEILING}
        patch_document(root, tid, lambda row: stamp_task_result_schema(lifted_record(row, decision)))
        fence = owner_pause.read_fence(root, tid)
        if wake == "cold_root_release":
            owner_pause.release_fence(root, tid, reason="owner_resume_consumed")
        else:
            selected = {task_id: {"grant_id": "g-" + task_id, "selected_at": "t1", "warm": True} for task_id in members}
            owner_pause.set_fence_state(root, tid, fence_id=fence["fence_id"], state=fence["state"],
                                        expected_state=fence["state"], selected_members=selected)
        assert "resume_grant" not in owner_pause.read_fence(root, tid)
    assert load_task_result(root, tid)["metadata"]["producer_allowance_lifted"]["ceiling_usd"] == CEILING
    for ctx, scope, _held in members.values():
        with usage_scope(scope):
            assert consume_warm_resume(ctx) is True
    child = members["warm-child"][0]
    assert child._cost_ceiling.state == task_pacing.COST_CEILING_DISABLED
    assert child.task_metadata == {"root_task_id": tid}
    assert members["warm-explicit"][0]._cost_ceiling == explicit


@pytest.mark.parametrize("door", ["owner", "mind", "automatic"])
def test_a_warm_members_real_loop_runs_on_its_roots_owner_lift_and_any_other_resume_keeps_the_allowance(
        root, monkeypatch, door):
    """The same member's real loop over the $0.40 allowance (the review's second probe): the owner's own Resume of
    the warm root lets it finish on the project budget; a mind's or an automatic Resume of the root leaves the
    allowance, and its next budget check pauses on it."""
    import contextlib

    from ouroboros.control_actor import mind_control, owner_door
    from ouroboros.task_results import write_task_result
    from ouroboros.usage_accounting import usage_scope
    from ouroboros.usage_origin import UsageScope
    from supervisor import workers

    @contextlib.contextmanager
    def minds():
        with owner_door("api_task_resume"), mind_control("mind-1", "the owner asked me to resume it"):
            yield

    opened = {"owner": lambda: owner_door("api_task_resume"), "mind": minds, "automatic": contextlib.nullcontext}[door]
    tid, _built, _metadata, _meta, args, _root_scope, loop = _warm_parked_turn(root, monkeypatch, opened)
    child_id = "warm-child-loop"
    ctx = args["tools"]._ctx
    ctx.task_id, ctx.root_task_id, ctx.task_metadata = child_id, tid, {"root_task_id": tid}
    args["task_id"] = child_id
    write_task_result(root, child_id, "running", root_task_id=tid, task_contract={}, metadata=ctx.task_metadata,
                      result="Task is running.")
    workers.RUNNING[child_id] = {
        "task": {"id": child_id, "root_task_id": tid, "type": "task", "chat_id": 0},
        "worker_id": 91, "attempt": 1, "owner_wait": {"state": "waiting", "reason": "owner_pause"}}
    scope = UsageScope(drive_root=root, task_id=child_id, root_task_id=tid, root_limit_usd=5.0,
                       root_cost_ceiling_usd=CEILING)
    with usage_scope(scope):
        if door == "owner":
            assert loop.run_llm_loop(**args)[0] == "Finished on the project budget."
        else:
            with pytest.raises(_Repaused):
                loop.run_llm_loop(**args)
