"""Resume a queued root through its existing owner-Pause fence and queue latch."""

from __future__ import annotations

import pathlib
import time
import uuid
from typing import Any, Dict

from ouroboros.control_actor import control_actor, resume_author
from ouroboros.utils import utc_now_iso


def _has_resume_choices(blockers, *, allow_starts):
    from ouroboros.work_stop import acceptable_unknown

    # A census has ids for local holds; only decide() can expose a delegated start choice.
    return bool(acceptable_unknown(blockers) or (allow_starts and any(b.get("ordinary_start") for b in blockers)))


def observe_admitted_retry(q, observation, *, gaps):
    """Observe custody and, for a selectable outcome, ordinary Resume authority off the queue lock."""
    from supervisor.budget_resume import _global_money_refusal, _restart_or_update_refusal, _root_money_refusal
    from supervisor.continuation_admission import action_writers
    from supervisor.state import budget_remaining

    authority = observation["authority"]
    task = authority["task"]
    task_id = task["id"]
    tree = task.get("root_task_id") or task_id
    root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    try:
        blockers = action_writers(q, tree, drive_root=root, owner_pause_fence_id=authority["owner_fence"]["fence_id"],
                                  gaps=gaps)
    except Exception as exc:
        blockers = [{"kind": "tree_census_unreadable", "detail": str(exc)[:200]}]
    observation["owner_pause_blockers"] = blockers
    safe = observation["safe"]
    if (safe or "_working_recovery" in task) and _has_resume_choices(blockers, allow_starts=safe):
        observation["choice_resume_refusal"] = (
            _restart_or_update_refusal(q, task, root, task_id)
            or _global_money_refusal(q, budget_remaining, [task])
            or _root_money_refusal(root, task, tree)
        )


def _retry_clock_refusal(task):
    from ouroboros.model_wait import execution_elapsed_seconds
    from supervisor.budget_resume import resume_clock_refusal
    from supervisor.worker_assignment import _settled_running_task

    # Project the exact clock assignment would hand over. An old retry without
    # a handoff keeps its original clock; newer carries exclude time in the queue.
    candidate = _settled_running_task(task, task.get("_working_recovery") or {})
    now = time.time()
    elapsed = execution_elapsed_seconds({**(candidate.get("_working_recovery") or {}), "task": candidate}, now)
    return resume_clock_refusal(task, elapsed, now)


def _resume_unstarted_owner_paused_root(q: Any, task: Dict[str, Any], fence: Dict[str, Any],
                                         observation: Dict[str, Any]) -> Dict[str, Any]:
    """The owner's Resume of a root the Pause caught while it was only queued.

    A never-started root has no members. An admitted retry needs a fresh tree
    census without unsettled effects (owner_pause_blockers); its attempt and
    dispatch evidence stay. As every other root Resume, this one lends the root
    one grant, bound to the observed fence generation and root basis: its held
    members bind to it, and the released fence carries it, so the resumed root
    or the owner selects each child explicitly (Q9) and nothing fans out;
    selection and dispatch revalidate it against the current fence and basis.
    Reopen both durable fence and queue latch or refuse: reopening only the
    latch would fail every launch. The latch names the LOGICAL root; task may
    be its current retry (#1730). A Continue whose earlier attempt's writers
    all still run leaves the Pause for its own writer hold, which starts it
    once they settle (owner, 1 Nov); an unknown outcome still refuses. If the
    replay check refused an admitted root, persist that typed refusal and its
    observed basis with the fence release, lending nothing, so the root's own
    Resume addresses it next instead of children hidden behind it. A named local
    unknown can release that fence, but only a usable source can take a named
    delegated-start risk. Queue lock held.
    """
    from ouroboros.owner_pause import FENCE_RELEASED, launch_lock, read_fence, set_fence_state
    import copy
    from supervisor.events_budget import (
        BUDGET_HOLD_KEY, hold_continuation_writers, hold_root_resume_descendants, project_writer_wait,
    )
    from ouroboros.work_stop import (
        clear_refusal, current_stop, refuse_continuation, release_refusal, stop_record, unsafe_replay_response,
    )

    task_id, root_id = str(task.get("id") or ""), str(fence.get("root_task_id") or task.get("id") or "")
    predecessor = str(((task.get("metadata") or {}).get("continuation") or {}).get("predecessor_task_id") or "")
    admitted = "owner_pause_blockers" in observation
    safe, error = observation["safe"], observation["unsafe_error"]
    if not safe and not admitted:
        return unsafe_replay_response(observation)
    refused = error if not safe and "_working_recovery" in task else ""
    evidence = {"refusal_basis": observation["refusal_basis"]} if refused and observation.get("refusal_basis") else {}
    if refusal := release_refusal(current_stop(task, {
            "owner": observation["authority"]["owner_fence"], "budget": q.BUDGET_ROOT_FENCES}), control_actor()):
        return refusal
    writers = list(observation.get("blockers") or ())
    if writers:
        from supervisor.continuation_admission import only_running_writers, writers_unsettled_refusal

        if not only_running_writers(writers):
            refusal = writers_unsettled_refusal(writers)
            if admitted and "_working_recovery" in task:
                refusal["selection_scope_mismatch"] = True  # this saved episode cannot accept a predecessor's choice
            return refusal
    root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    blockers = observation.get("owner_pause_blockers") or []
    from ouroboros.delegate_resume import decide, decisions_in_use, names_start
    own_episode = admitted and (not predecessor or "_working_recovery" in task)
    if names_start() and not (own_episode and safe):
        return {"ok": False, "error": "selection_authority_changed"}
    has_choices = _has_resume_choices(blockers, allow_starts=safe)
    if own_episode and (safe or (refused and has_choices)):
        if "choice_resume_refusal" in observation:
            if refusal := observation["choice_resume_refusal"] or _retry_clock_refusal(task):
                return refusal
        elif has_choices:
            return {"ok": False, "error": "selection_observation_required"}
        if refusal := decide(root, root_id, task_id, blockers, tree_actor=True, allow_starts=safe,
                             fence_id=str(observation["authority"]["owner_fence"]["fence_id"])):
            return refusal
        blockers = []
    if blockers:
        return {"ok": False, "error": "owner_pause_effects_unsettled", "action": "wait_for_effect_settlement",
                "blockers": blockers}
    members = [row for row in q.PENDING if admitted and row is not task
               and str(row.get("root_task_id") or "") == root_id]
    prior = [(row, copy.deepcopy(row)) for row in [task, *members]]

    def rollback() -> None:
        for row, before in prior:
            row.clear()
            row.update(before)
        q.BUDGET_ROOT_FENCES[root_id] = fence

    with launch_lock(root, root_id):
        authority = observation["authority"]
        if read_fence(root, root_id) != authority["owner_fence"]:
            return {"ok": False, "error": "selection_authority_changed"}
        # Minted before any queue change, from the revalidated observation; held members bind
        # to it below and the fence release records the same grant, or rollback drops both.
        grant = {"by": resume_author(), "grant_id": uuid.uuid4().hex, "granted_at": utc_now_iso(),
                 "granted_at_ts": time.time(), "authority": "explicit_resume", "selected_by": "owner",
                 "generation": int(authority["owner_fence"].get("generation") or 0),
                 "root_resume_basis": authority["root_resume_basis"],
                 **({"accepted_start_risks": risks} if (risks := decisions_in_use(task_id)) else {})}
        # Publish queue eligibility first, while durable launch authority is still
        # closed. A crash or failed second write cannot start the unrun root.
        q.BUDGET_ROOT_FENCES.pop(root_id, None)
        task.pop("_budget_pause", None)
        task.pop(BUDGET_HOLD_KEY, None)
        clear_refusal(task)
        if refused:  # rollback restores the row: the refusal stands only with this published Resume
            refuse_continuation(task, stop_record(refused, held_at=utc_now_iso(), by=control_actor(),
                                                  binding=evidence))
        wait = hold_continuation_writers(task, predecessor, writers) if writers and not refused else None
        if members:
            hold_root_resume_descendants(q, root_id, fence, grant, actor_id=task_id)
        if not q.publish_queue_snapshot(reason="owner_pause_root_resumed"):  # durable while launch authority is closed
            rollback()
            return {"ok": False, "error": "snapshot_not_persisted"}
        from ouroboros.presence_carrier import lift_allowance, lifted_record, owner_resume_decision

        try:
            # The owner's own Resume of a moved Presence turn (#2566) is decided in the one write that releases it:
            # neither stands without the other. The row takes it below, before any dispatch; a restore reads that write.
            lift = owner_resume_decision(task, root)
            set_fence_state(root, root_id, fence_id=str(fence.get("fence_id") or ""),
                            state=FENCE_RELEASED, release_reason="owner_resume_unstarted_root",
                            resume_actor=control_actor(), resume_grant=grant,
                            row_update=(lambda row: lifted_record(row, lift)) if lift else None)
        except Exception as exc:
            rollback()
            q.publish_queue_snapshot(reason="owner_pause_root_resume_held")
            return {"ok": False, "error": "owner_pause_fence_unwritable", "detail": str(exc)[:200]}
        lift_allowance(task, lift)
    held = {"held": True, "blockers": wait["blockers"]} if wait else {}
    if wait:
        project_writer_wait(root, task_id, wait)
    # ok=True: the Pause is released, nothing dispatched; the row's typed refusal says what is next. Only this
    # append-only event keeps that this Resume kept it once the row is gone; the answer gains no key.
    q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                   {"ts": utc_now_iso(), "type": "owner_pause_resumed", "task_id": task_id,
                    "root_task_id": root_id, "fence_id": str(fence.get("fence_id") or ""),
                    "grant_id": grant["grant_id"], "same_generation": True, "never_started": not admitted, **held,
                    **({"continuation_refused": refused, **evidence} if refused else {})})
    return {"ok": True, "task_id": task_id, "root_task_id": root_id, "same_generation": True,
            "owner_pause_released": True, "never_started": not admitted, "grant_id": grant["grant_id"], **held}
