"""Same-task owner waiting at a completed-tool boundary.

The queue owns admission and active worker capacity. This module preserves the
native continuation through the existing source store. Pooled workers wait on
their command queue; direct actors use the same mailbox and task controls without
holding pooled capacity. A warm wake continues the original stack and browser;
a confirmed planned-restart handoff can load a cold continuation. A validated
model sleep may also transfer to the existing exact budget-pause owner on stop;
its owner-wait state becomes ``retained`` and explicit Resume is required.
Quiz/review waits keep their existing authority. Source bytes outlive grants.

An optional bound (``escalate(max_wait_minutes=N)``) rides the checkpoint as an
ABSOLUTE stamp (``wait_deadline_at``), so a planned restart resumes the same
bound instead of starting it over. Both callbacks report a typed wake cause
(``answer``, ``owner_text``, ``hurry``, ``mail:<task_id>``, ``timeout`` or
``control:<reason>``; ``control:owner_pause`` wakes a warm wait so the owner's
Pause reaches its boundary without a model round). Stop, cancel, deadline and absolute ceiling are checked
first; a timeout is not an answer and creates no new wait state, and a hurry
is a request to finish sooner, never an answer.
"""

from __future__ import annotations

import json
import logging
import pathlib
import queue
import time
import uuid
from dataclasses import asdict
from typing import Any

from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
from ouroboros.observability import without_finalization_timing
from ouroboros.owner_mailbox import OwnerMailboxPeek
from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def classify_wake(entries: list[dict], quiz_id: str) -> str:
    """The typed cause of a wake; a wake is never an answer by default.

    Closed vocabulary (TZ-2 B2): ``answer`` (this card), ``owner_text`` (the
    owner's own words, or an answer to another card), ``hurry`` (the owner's
    typed acceleration control: a request to finish sooner, never an answer),
    ``mail:<task_id>`` (task mail, naming its sender; ``mail:unknown`` when it
    names none), ``unknown`` (nothing observed); the callers add ``timeout``
    and ``control:<reason>``.
    Precedence: control, this card's answer, owner words, hurry, mail.
    """
    from ouroboros.owner_mailbox import (
        KIND_FINALIZE_NOW,
        KIND_HURRY,
        KIND_OWNER_PAUSE,
        KIND_OWNER_TEXT,
        KIND_QUIZ_ANSWER,
        KIND_TASK_MESSAGE,
    )

    kinds = [str(row.get("kind") or KIND_OWNER_TEXT) for row in entries]
    if KIND_FINALIZE_NOW in kinds:
        return "control:finalize_now"
    if KIND_OWNER_PAUSE in kinds:
        return "control:owner_pause"
    if any(kind == KIND_QUIZ_ANSWER and row.get("msg_id") == f"quiz_answer:{quiz_id}"
           for kind, row in zip(kinds, entries)):
        return "answer"
    if any(kind in {KIND_OWNER_TEXT, KIND_QUIZ_ANSWER} for kind in kinds):
        return "owner_text"
    if KIND_HURRY in kinds:
        return "hurry"
    for kind, row in zip(kinds, entries):
        if kind == KIND_TASK_MESSAGE:  # the first mail is the one that woke the task
            source = str(row.get("source_task_id") or "").strip()
            return f"mail:{source}" if source else "mail:unknown"
    return "mail:unknown" if entries else "unknown"


def _wait_entries(ctx: Any) -> list[dict]:
    """Observe on a private seen-set; only the loop may ACK or deliver."""
    from ouroboros.owner_mailbox import drain_owner_entries

    return drain_owner_entries(
        pathlib.Path(ctx.drive_root), ctx.task_id,
        set(getattr(ctx, "_loop_mailbox_seen_ids", None) or ()), getattr(ctx, "task_attempt", None) or 1,
    )


def _fresh_wake(ctx: Any, quiz_id: str, outcome: str) -> str:
    """An owner answer can land while a pooled wait is awaiting capacity.

    Owner authority (this card's answer, owner words, a control) replaces the
    reason the resume was requested with; anything else replaces only
    ``unknown``, so a bound that ended the wait keeps saying so.
    """
    observed = classify_wake(_wait_entries(ctx), quiz_id)
    if observed in {"answer", "owner_text", "control:finalize_now", "control:owner_pause"}:
        return observed
    return observed if outcome == "unknown" else outcome


class OwnerWaitSuperseded(ValueError):
    """A compare-and-set lost: the wait row changed identity or state since its read."""


def set_owner_wait(root: Any, task_id: str, wait: dict,
                   expected_wait_id: str | None = None, *, expected_state: str | None = None) -> dict:
    """Update only the existing continuation projection, preserving siblings."""
    from ouroboros.task_results import (
        require_writable_task_result_schema,
        stamp_task_result_schema,
        task_result_path,
    )
    from ouroboros.utils import update_json_locked

    def update(current: dict) -> dict:
        require_writable_task_result_schema(current)
        if not current.get("status") or current.get("task_id") != task_id:
            raise ValueError("owner wait requires its lifecycle owner's task result")
        if current.get("status") in _TRULY_TERMINAL_STATUSES:
            raise ValueError("a terminal task cannot continue owner waiting")
        old = current.get("owner_wait") or {}
        if expected_wait_id is not None and old.get("wait_id") != expected_wait_id:
            raise OwnerWaitSuperseded("owner wait identity changed")
        if expected_state is not None and old.get("state") != expected_state:
            raise OwnerWaitSuperseded("owner wait state changed")
        return stamp_task_result_schema({**current, "owner_wait": dict(wait)})

    update_json_locked(task_result_path(root, task_id), update, strict_existing_dict=True)
    return dict(wait)


def _wait_bound_fields(ctx: Any) -> dict:
    """The optional bound, as an ABSOLUTE instant plus the minutes it named.

    Absent unless a bound was requested: an unbounded wait must not carry a
    field that reads as one. The stamp (not a remaining count) is what lets a
    planned restart resume the SAME bound instead of granting it again.
    """
    deadline = str(getattr(ctx, "_owner_wait_deadline_at", "") or "")
    if not deadline:
        return {}
    return {"wait_deadline_at": deadline,
            "wait_max_minutes": int(getattr(ctx, "_owner_wait_max_minutes", 0) or 0)}


def continuation_state(ctx: Any, messages: list, trace: dict, usage: dict,
                       round_idx: int, tool_schemas: list, seen: set) -> dict:
    """The loop's exact continuation values, never Python handles.

    ONE serializer for every same-ID continuation (owner wait, acceptance park,
    budget pause): the carried fields are the loop's cognition — transcript,
    trace, usage, route, delivery candidate, acceptance identities, owner
    directives — so a second serializer could only drift from this one.
    """
    candidate = getattr(ctx, "_delivery_candidate", None)
    cost_ceiling = getattr(ctx, "_cost_ceiling", None)
    model_wait = getattr(ctx, "model_wait_context", None)
    model_state = model_wait.continuation_state() if model_wait is not None else {}
    return {
        "task_id": ctx.task_id, "task_attempt": int(getattr(ctx, "task_attempt", None) or 1),
        "started_at": getattr(ctx, "task_started_at", None),
        # A successor of this source records its attempt before worker handoff;
        # an absent legacy dispatch mark can only belong to the original run.
        "retained_work_dispatch_protocol": 1,
        "messages": messages, "trace": trace,
        # Live timing cannot retain a monotonic origin across a reboot.
        "usage": without_finalization_timing(usage),
        "cost_ceiling": asdict(cost_ceiling) if cost_ceiling is not None else None,
        "model_wait": model_state,
        "context_model_role": getattr(getattr(ctx, "context_fit_plan", None), "model_role", ""),
        # Exposure facts gate automatic compaction; a resumed task keeps them.
        "context_observations": {key: getattr(ctx, key) for key in (
            "_last_context_observation", "_inspected_context_view", "_pending_compaction",
            "_historical_author_inputs",
        ) if getattr(ctx, key, None) is not None},
        "round_idx": round_idx, "tool_schemas": tool_schemas,
        "seen": sorted(seen), "owner_directives": getattr(ctx, "_owner_directives", []),
        "route": {key: getattr(ctx, key, None) for key in (
            "active_model", "active_effort", "active_use_local", "active_context_mode",
            "active_model_override", "active_effort_override", "active_use_local_override",
            "active_role_override", "route_wait_on_primary", "primary_route", "_route_facts_pending",
        )},
        "delivery_candidate": asdict(candidate) if candidate is not None else None,
        "delivery": {key: getattr(ctx, key, None) for key in (
            "_delivery_candidate_revision", "_delivery_control_required",
            "_delivery_evidence_revision", "_delivery_evidence_fingerprint",
            "_delivery_effective_criteria", "_delivery_material_tool_indices",
            "_acceptance_ack_source_sha256", "_completion_request", "_completion_selected",
            "_completion_observation", "_completion_held_sha256", "_presence_completion",
            "_presence_completion_owner_revision", "_acceptance_observation",
            "_presence_forced_declaration", "_presence_forced_pending", "_presence_completion_accepted",
            "_presence_selection_seq", "_presence_released", "_presence_release",
            "_task_acceptance_sealed_fence_token", "_task_acceptance_sealed_fence_generation",

        ) if getattr(ctx, key, None) is not None},
        "acceptance": {
            "_task_acceptance_improvement_passes": int(getattr(ctx, "_task_acceptance_improvement_passes", 0)),
            "_task_acceptance_reviewed": bool(getattr(ctx, "_task_acceptance_reviewed", False)),
            "_task_acceptance_pending": str(getattr(ctx, "_task_acceptance_pending", "")),
            "_task_acceptance_reviewed_subject": str(getattr(ctx, "_task_acceptance_reviewed_subject", "")),
        },
    }


def store_continuation_source(ctx: Any, state: dict, source_id: str) -> dict:
    """Persist one continuation state through the existing actor source store."""
    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    return store_actor_source_bytes(root, ctx.task_id, category="context_checkpoints",
                                    source_id=source_id,
                                    data=json.dumps(state, ensure_ascii=False).encode(), extension="json")


REASON_OWNER_PAUSE_WARM = "owner_pause"


def checkpoint_owner_wait(ctx: Any, messages: list, trace: dict, usage: dict,
                          round_idx: int, tool_schemas: list, seen: set,
                          *, review_binding: str = "", pause: dict | None = None) -> dict:
    """Capture only the live loop's continuation values, never Python handles.

    ``pause`` names the owner Pause a member parks WARM under (its fence id and
    generation, the critics it detached from): reason ``owner_pause``, woken by
    that fence's release, never by ordinary mail.
    """
    wait_id = uuid.uuid4().hex
    sleep = getattr(ctx, "_model_sleep", None) if not (review_binding or pause) else None
    reason = (REASON_OWNER_PAUSE_WARM if pause else "review" if review_binding
              else ("sleep" if sleep else "owner"))
    quiz_id = "" if (sleep or pause) else getattr(ctx, "_owner_wait_requested", "")
    bound = {} if pause else _wait_bound_fields(ctx)
    state = {
        **continuation_state(ctx, messages, trace, usage, round_idx, tool_schemas, seen),
        "wait_id": wait_id, "quiz_id": quiz_id,
        **bound,
        "reason": reason, **({"sleep": dict(sleep)} if sleep else {}),
        **({"owner_pause": dict(pause)} if pause else {}),
        "review_binding": review_binding,
    }
    source = store_continuation_source(ctx, state, "owner-wait-" + wait_id)
    model_state = state.get("model_wait") or {}
    return {
        "wait_id": wait_id, "quiz_id": quiz_id,
        **bound,
        # A model sleep (``model_sleep``): only its selected sources wake it.
        "reason": reason, **({"sleep": dict(sleep),
                                "sleep_started_at": float(getattr(ctx, "_model_sleep_started", 0) or time.time())}
                               if sleep else {}),
        **({"owner_pause": dict(pause)} if pause else {}),
        "review_binding": review_binding,
        # When THIS owner wait began: readers date the wait by it, never by the
        # task's ``started_at`` below (which the lifetime clocks own). The whole
        # row rides every later write, so a planned restart keeps the stamp. A
        # review-bound park is dated by its review operation, not here.
        **({} if review_binding else {"parked_at": utc_now_iso()}),
        "source_ref": source, "task_attempt": int(ctx.task_attempt or 1),
        "execution_drive_root": str(ctx.drive_root),
        "started_at": getattr(ctx, "task_started_at", None),
        "model_wait_quota_clock": model_state.get("quota_clock", {}),
        # The SAME two carriers every finite-lifetime reader subtracts: the quota
        # union above and the budget-paused interval (#1196, F5). Without it the
        # row's own lifetime check would count a pause as execution.
        "budget_paused_sec": float(model_state.get("budget_paused_sec") or 0.0),
    }


def saved_sleep_checkpoint(root: Any, task_id: str, attempt: int | None = None) -> tuple[dict, dict]:
    """Validate an exact warm-sleep source before shutdown promises retention."""
    row = load_task_result(root, task_id, strict=True) or {}
    wait = row.get("owner_wait") or {}
    if (row.get("status") in _TRULY_TERMINAL_STATUSES or wait.get("state") != "waiting"
            or wait.get("reason") != "sleep" or not wait.get("source_ref")
            or not isinstance(wait.get("sleep"), dict) or not wait["sleep"].get("sleep_id")
            or wait["sleep"].get("mode") != "warm"):
        return {}, {}
    state = json.loads(read_actor_source_bytes(root, task_id, wait["source_ref"]))
    if (state.get("task_id") != task_id or not wait.get("wait_id")
            or state.get("wait_id") != wait["wait_id"] or state.get("reason") != "sleep"
            or state.get("sleep") != wait["sleep"]
            or int(state.get("task_attempt") or 0) != int(wait.get("task_attempt") or 0)
            or not wait.get("task_attempt")
            or attempt is not None and int(wait["task_attempt"]) != int(attempt)):
        return {}, {}
    return wait, state


def load_owner_wait(ctx: Any, handoff: dict | None = None) -> dict:
    """Resolve a selected handoff; a stale snapshot cannot revive a spent wait."""
    handoff = handoff or getattr(ctx, "owner_wait_resume", None)
    if not handoff:
        return {}
    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    row = load_task_result(root, ctx.task_id, strict=True) or {}
    current = row.get("owner_wait") or {}
    if (row.get("status") in _TRULY_TERMINAL_STATUSES
            or current.get("state") != "waiting"
            or current.get("wait_id") != handoff.get("wait_id")
            or not handoff.get("restart_transaction_id")):
        raise ValueError("owner wait continuation is not an active planned-restart handoff")
    state = json.loads(read_actor_source_bytes(root, ctx.task_id, current["source_ref"]))
    if (state.get("task_id") != ctx.task_id
            or state.get("wait_id") != current.get("wait_id")
            or state.get("task_attempt") != int(ctx.task_attempt or 1)):
        raise ValueError("owner wait continuation identity mismatch")
    from ouroboros.task_pacing import restore_cost_ceiling

    # The start's own authority, never the saved number alone (#1128).
    ctx._cost_ceiling = restore_cost_ceiling(ctx, state.get("cost_ceiling"))
    return state


def restore_owner_wait_allowed(root: Any, task: dict, *, strict: bool = False) -> bool:
    """Current wait and acknowledged restart authorize a locator; strict preserves read failures."""
    import time

    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.config import get_task_abs_ceiling_sec
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from ouroboros.delegate_recovery import (
        _ack_direct_exec_successor,
        _read_restart_transaction,
        _restart_transaction_path,
    )
    from ouroboros.model_wait import execution_elapsed_seconds

    handoff = task.get("_owner_wait_resume")
    if not isinstance(handoff, dict):
        return False
    root = pathlib.Path(root)
    try:  # Panic never returns work; the owner's Restart returns only what its acknowledged transaction names
        if (root / "state" / "panic_stop.flag").read_text(encoding="utf-8").strip() == "panic":
            return False
    except FileNotFoundError:
        pass
    except OSError:
        return False
    _ack_direct_exec_successor(root)
    task_id = str(task.get("id") or "")
    transaction_id = str(handoff.get("restart_transaction_id") or "")
    transaction = {} if strict else _read_restart_transaction(root, transaction_id)
    if not strict and (transaction.get("status") != "normal_exit_acknowledged"
                       or task_id not in transaction.get("task_ids", [])):
        return False
    row = load_task_result(root, task_id, strict=True) or {}
    wait = row.get("owner_wait") or {}
    if (row.get("status") in _TRULY_TERMINAL_STATUSES
            or wait.get("state") != "waiting" or wait.get("wait_id") != handoff.get("wait_id")):
        return False
    if has_active_intent(root, task_id, strict=True):
        return False
    deadline = parse_deadline_ts(task.get("deadline_at") or (task.get("task_contract") or {}).get("deadline_at"))
    if deadline is not None and deadline <= utc_now():
        return False
    started = float(handoff.get("started_at") or wait.get("started_at") or 0)
    now = time.time()
    ceiling = get_task_abs_ceiling_sec()  # None = no lifetime bound to have outlived
    # ONE shared clock (``model_wait.execution_elapsed_seconds``): wall time minus
    # the quota union minus the budget-paused carrier. A task that was paused and
    # then parked in an owner wait must not have that paused time charged to its
    # finite lifetime by this reader alone (#1196, F5).
    # The durable row is the authority whenever it carries the field (0.0 included);
    # the handoff is the fallback for a row written before it existed.
    paused_carrier = wait.get("budget_paused_sec")
    if paused_carrier is None:
        paused_carrier = handoff.get("budget_paused_sec") or 0.0
    executed = execution_elapsed_seconds(
        {"started_at": started,
         "model_wait_quota_clock": wait.get("model_wait_quota_clock") or {},
         "budget_paused_sec": paused_carrier}, now)
    if started and ceiling is not None and executed >= ceiling:
        return False
    # Independent controls apply even when restart/replay evidence is unreadable.
    if strict:
        transaction = json.loads(_restart_transaction_path(root, transaction_id).read_text(encoding="utf-8"))
    if not isinstance(transaction, dict):
        raise ValueError("Owner-wait restart transaction is unreadable")
    if transaction.get("status") != "normal_exit_acknowledged" or task_id not in transaction.get("task_ids", []):
        return False
    read_actor_source_bytes(root, task_id, wait["source_ref"])
    return True


def worker_owner_wait(wid: int, in_q: Any, out_q: Any, ctx: Any,
                      checkpoint: dict) -> str:
    """Keep the original task process asleep until the pool grants capacity.

    A bound is a second reason to ask for the SAME resume the mailbox already
    triggers — no new scheduler: the request must sit inside the ``parked``
    gate, because the supervisor ignores a resume for a wait that is not
    durably waiting. Disclosed: a resume is a REQUEST, not a guarantee (the
    grant can be refused under a cancel intent or the repo-writer gate); the
    hard axes still end the task in that case.
    """
    import os

    from ouroboros.deadline_utils import parse_deadline_ts, utc_now

    identity = {"type": "owner_wait", "worker_id": wid, "pid": os.getpid(),
                "task_id": ctx.task_id, "task_attempt": int(ctx.task_attempt or 1),
                "wait_id": checkpoint["wait_id"]}
    # The existing deferred buffer must reach the supervisor before parking.
    # Remove only a successfully submitted prefix; its final flush cannot repeat it.
    while ctx.pending_events:
        out_q.put({**ctx.pending_events[0], "worker_id": wid})
        del ctx.pending_events[0]
    out_q.put({**identity, "phase": "park", "checkpoint": checkpoint})
    peek = OwnerMailboxPeek()
    # The bound's absolute instant; None = an unbounded wait.
    deadline = parse_deadline_ts((checkpoint or {}).get("wait_deadline_at"))
    parked = resume_requested = False
    outcome = "unknown"
    while True:
        try:
            command = in_q.get(timeout=1.0)
        except queue.Empty:
            command = None
        if isinstance(command, dict) and command.get("type") == "owner_wait":
            if all(command.get(key) == identity[key] for key in ("task_id", "task_attempt", "wait_id")):
                phase = command.get("phase")
                if phase == "parked":
                    parked = True
                elif phase == "resume_granted":
                    if checkpoint.get("reason") != REASON_OWNER_PAUSE_WARM:  # the Pause's own wake cause stands
                        outcome = _fresh_wake(ctx, str(checkpoint.get("quiz_id") or ""), outcome)
                    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
                    wait = (load_task_result(root, ctx.task_id, strict=True) or {}).get("owner_wait") or {}
                    if wait.get("wait_id") == checkpoint["wait_id"] and wait.get("state") == "resumed":
                        set_owner_wait(root, ctx.task_id, {**wait, "resume_reason": outcome}, wait["wait_id"])
                    return outcome
                elif phase == "refused":
                    raise RuntimeError(str(command.get("reason") or "owner wait refused"))
        if parked and not resume_requested:
            woke = _sleep_wake(ctx, checkpoint)
            if woke is not None:
                if woke:  # a sleep's own selected source (or the owner) is ready
                    outcome = woke
                    out_q.put({**identity, "phase": "resume", "resume_reason": outcome})
                    resume_requested = True
            elif checkpoint.get("reason") == REASON_OWNER_PAUSE_WARM:
                woke = _owner_pause_wake(ctx, checkpoint)
                if woke:  # the fence opened, or a stop control arrived
                    outcome = woke
                    out_q.put({**identity, "phase": "resume", "resume_reason": outcome})
                    resume_requested = True
            elif peek.pending(
                    pathlib.Path(ctx.drive_root), ctx.task_id,
                    set(getattr(ctx, "_loop_mailbox_seen_ids", set())), ctx.task_attempt or 1):
                outcome = classify_wake(_wait_entries(ctx), str(checkpoint.get("quiz_id") or ""))
                out_q.put({**identity, "phase": "resume", "resume_reason": outcome})
                resume_requested = True
            elif deadline is not None and utc_now() >= deadline:
                outcome = "timeout"
                out_q.put({**identity, "phase": "resume", "resume_reason": outcome})
                resume_requested = True


def _sleep_wake(ctx: Any, checkpoint: dict) -> str | None:
    """``None`` for an owner/review wait (any mail wakes it); for a model sleep, the
    reason its selected sources give NOW (``""`` = keep sleeping), read from the
    canonical records on every poll — the recheck after the park that closes the
    race with an event landing between the tool's check and the park."""
    chosen = (checkpoint or {}).get("sleep")
    if not isinstance(chosen, dict):
        return None
    from ouroboros.model_sleep import wake_reason

    try:
        return wake_reason(ctx, chosen)
    except Exception:
        log.warning("Sleep readiness unreadable for %s; still sleeping", ctx.task_id, exc_info=True)
        return ""


def _owner_pause_wake(ctx: Any, checkpoint: dict) -> str:
    """What ends a WARM owner-Pause park, read on every poll: the root fence's
    release (or this member's explicit selection) for ``control:owner_resume``,
    or the task's own ending controls (Stop/cancel, Panic, deadline, lifetime),
    exactly the ones that end a cold pause's hold. The owner's words, a Wrap up
    and task mail wait for Resume, as every paused member's do. An unreadable
    fence keeps the park: unknown authority never wakes a paused task."""
    from ouroboros.budget_pause import _hold_control_reason
    from ouroboros.owner_pause import member_fence

    try:
        fence = member_fence(ctx)
    except Exception:
        fence = {"state": "unknown"}
    if not fence:
        return "control:owner_resume"
    control = _hold_control_reason(ctx)
    return f"control:{control}" if control else ""


def park_owner_pause_warm(limit_ctx: Any, ctx: Any, *, fence: dict, detached: list) -> str:
    """Park the SAME author stack warm under the owner's Pause (full variant, 2026-10-08).

    The member whose launched reviewers the Pause lets finish keeps its worker:
    the exact continuation is stored through the one serializer, the pool
    lends the worker's capacity (``worker_owner_wait``) or the direct actor
    holds its stack (``direct_owner_wait``), and the fence's release returns
    the same stack with a notice naming the reviews that kept running. Nothing
    is cancelled, bought or re-dispatched here. Returns the wake cause.
    """
    callback = getattr(ctx, "owner_wait_callback", None)
    if not callable(callback):
        raise RuntimeError("owner pause warm park has no worker continuation owner")
    messages, trace = limit_ctx.messages, limit_ctx.llm_trace if isinstance(limit_ctx.llm_trace, dict) else {}
    usage_for_state = {key: value for key, value in limit_ctx.accumulated_usage.items()
                       if key not in ("execution_status", "reason_code", "_best_effort_extracted",
                                      "budget_pause_hold", "_llm_round_started")}
    from ouroboros.external_runs import STOP_POLICY_TASK_OWNED, observe_task_runs

    try:  # the member's OWN runs stop exactly as at a cold Pause; its reviewers' runs are spared
        from ouroboros import delegate_custody as custody

        external = observe_task_runs(custody.custody_root(ctx), str(ctx.task_id),
                                     reason="owner_pause_warm_park", stop_policy=STOP_POLICY_TASK_OWNED)
    except Exception as exc:
        external = {"runs": [], "custody_read": "failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    pause = {"fence_id": str(fence.get("fence_id") or ""), "generation": int(fence.get("generation") or 0),
             "root_task_id": str(fence.get("root_task_id") or ""), "detached_reviews": list(detached),
             "external_runs": external, "parked_at": utc_now_iso()}
    from ouroboros import model_sleep

    # Reuse the live paused interval and fold it into the same cumulative carrier
    # once, including capacity reacquisition. Reviewer owners keep their own clocks.
    model_sleep.begin(ctx)
    try:
        checkpoint = checkpoint_owner_wait(ctx, messages, trace, usage_for_state, int(limit_ctx.round_idx),
                                           list(limit_ctx.tool_schemas or []),
                                           set(limit_ctx.owner_msg_seen or ()), pause=pause)
        outcome = str(callback(ctx, checkpoint) or "")
    finally:
        model_sleep.end(ctx)
    if outcome == "control:owner_resume":
        messages.append(owner_pause_resume_notice(ctx, checkpoint, outcome))
    return outcome


def consume_warm_resume(ctx: Any) -> bool:
    """Spend this member's explicit Resume where its stack actually continues.

    Under the root's launch lock: an open fence carrying the root's warm
    ``resume_grant`` gets ``consumed_at`` (once — a second reader finds it
    spent); an owner-selected member of a closed fence spends its selection.
    Any other open fence (a cold root's own consumption released it) needs
    nothing. A fence closed again by a newer Pause returns False: the stack
    parks again. A write failure raises; the caller holds and retries.
    """
    from ouroboros.owner_pause import _member_coordinates, fence_closed, launch_lock, read_fence, set_fence_state

    root_drive, root_task_id, task_id = _member_coordinates(ctx)
    with launch_lock(root_drive, root_task_id):
        fence = read_fence(root_drive, root_task_id)
        stamp = {"consumed_at": utc_now_iso(), "consumed_by": task_id,
                 "task_attempt": int(getattr(ctx, "task_attempt", 1) or 1)}
        if fence_closed(fence):
            selected = dict(fence.get("selected_members") or {})
            mine = selected.get(task_id)
            if not isinstance(mine, dict):
                return False
            if not mine.get("consumed_at"):
                selected[task_id] = {**mine, **stamp}
                set_fence_state(root_drive, root_task_id, fence_id=str(fence.get("fence_id") or ""),
                                state=str(fence.get("state") or ""), expected_state=str(fence.get("state") or ""),
                                selected_members=selected)
            return True
        grant = fence.get("resume_grant") if isinstance(fence.get("resume_grant"), dict) else {}
        if (grant.get("warm") and task_id == root_task_id and not grant.get("consumed_at")
                and not grant.get("revoked_at")):
            set_fence_state(root_drive, root_task_id, fence_id=str(fence.get("fence_id") or ""),
                            state=str(fence.get("state") or ""), expected_state=str(fence.get("state") or ""),
                            resume_grant={**grant, **stamp})
        return True


def owner_pause_resume_notice(ctx: Any, checkpoint: dict, outcome: str) -> dict:
    """The host frame a warm-parked member reads when the owner's Pause ends.

    States only what this process knows about each review it detached from:
    still ``running``, ``closed`` here (its workers settled and published), or
    ``unknown``. Re-calling the same review with an unchanged subject rejoins
    or replays that attempt in this process; nothing is sent again by itself.
    """
    from ouroboros.review_pause import review_operation_state

    pause = checkpoint.get("owner_pause") if isinstance(checkpoint.get("owner_pause"), dict) else {}
    lines = []
    for item in pause.get("detached_reviews") or []:
        if not isinstance(item, dict):
            continue
        state = review_operation_state(str(item.get("owner_id") or ""))
        lines.append(f"- {item.get('surface') or 'review'} operation {item.get('owner_id')}: {state}")
    reviews = ("\n".join(lines) if lines else "- none")
    abandoned = list(getattr(ctx, "_abandoned_model_attempts", None) or [])
    ctx._abandoned_model_attempts = []
    late = (f"The Pause interrupted {len(abandoned)} model request(s) already sent ({', '.join(abandoned)}): "
            "the provider may still have finished and charged them; any late answer was recorded but NOT used "
            "and its tools were not run. " if abandoned else "")
    return {"role": "user", "content": (
        "[SYSTEM NOTICE]\nThe owner paused this task and has now resumed it. Reviewers you had already "
        f"launched kept running while you were paused:\n{reviews}\nYour review tool returned a pending "
        "result for them; nothing was committed, applied or published after the Pause. Calling the same "
        "review again with an unchanged subject rejoins or replays that attempt in this process without a "
        "new dispatch; a changed subject is a new review. A reviewer whose state is unknown was not resent. "
        f"{late}Nothing you asked for after the Pause ran. Cumulative spend, rounds and elapsed time were "
        "not reset; prior tool results remain recorded; do not repeat completed effects.")}


def direct_owner_wait(ctx: Any, checkpoint: dict) -> str:
    """Retain a registered chat actor's stack; it holds no pooled capacity.

    The existing mailbox still owns input and its loop still owns delivery.
    TaskModelWait supplies the same Stop/deadline clocks as native model calls;
    this owner wait does not enter a quota pause or grant cold restart authority.
    An optional bound releases the loop only AFTER those controls are consulted,
    so Stop, the task deadline and the absolute ceiling keep precedence.
    """
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now

    control = ctx.model_wait_context
    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    # Without a queue (a Presence turn may have none) the buffer keeps its events for the final flush.
    while ctx.pending_events and getattr(ctx, "event_queue", None) is not None:
        ctx.event_queue.put(dict(ctx.pending_events[0]))
        del ctx.pending_events[0]
    wait = set_owner_wait(root, ctx.task_id, {**checkpoint, "state": "waiting"})
    peek = OwnerMailboxPeek()
    deadline = parse_deadline_ts((checkpoint or {}).get("wait_deadline_at"))  # None = unbounded
    outcome = "unknown"
    warm_pause = checkpoint.get("reason") == REASON_OWNER_PAUSE_WARM
    while not control.control_reason():
        woke = _sleep_wake(ctx, checkpoint)
        if woke:
            outcome = woke
            break
        if warm_pause:
            woke = _owner_pause_wake(ctx, checkpoint)
            if woke:
                outcome = woke
                break
        elif woke is None and peek.pending(
                pathlib.Path(ctx.drive_root), ctx.task_id,
                set(getattr(ctx, "_loop_mailbox_seen_ids", set())), ctx.task_attempt or 1):
            break
        if deadline is not None and utc_now() >= deadline:
            outcome = "timeout"
            break
        time.sleep(1.0)
    control_reason = control.control_reason()
    outcome = (f"control:{control_reason}" if control_reason else outcome if warm_pause else
               _fresh_wake(ctx, str(checkpoint.get("quiz_id") or ""), outcome))
    set_owner_wait(root, ctx.task_id,
                   {**wait, "state": "resumed", "resume_reason": outcome}, wait["wait_id"])
    if not outcome.startswith("control:"):
        announce_wait_ended(root, ctx.task_id, str(checkpoint.get("quiz_id") or ""),
                            int(getattr(ctx, "current_chat_id", 0) or 0))
    return outcome


def announce_wait_ended(root: Any, task_id: str, quiz_id: str, chat_id: int) -> None:
    """A bound closed and the turn resumed: the card's projection and the live card both
    stop saying "waiting" while the question stays answerable. The direct lane calls it
    here; the pool's supervisor grant calls the same two seams (best effort, never raises)."""
    if not quiz_id:
        return
    try:
        from ouroboros.owner_quiz import mark_wait_ended

        changed = mark_wait_ended(root, task_id, quiz_id)
    except Exception:
        log.debug("owner-wait end not recorded on quiz %s", quiz_id, exc_info=True)
        return
    if not changed:  # an answered card must never be broadcast as open
        return
    try:
        from supervisor.message_bus import get_bridge

        get_bridge().send_quiz_state(quiz_id, task_id, "open", chat_id=chat_id, wait_for_answer=False)
    except Exception:
        log.debug("owner-wait end not broadcast for quiz %s", quiz_id, exc_info=True)


def owner_wait_ended_notice(ctx: Any, checkpoint: dict, reason: str = "timeout",
                            *, woke_by: str = "") -> dict:
    """Host frame for a wait that ended without THIS question being answered.

    ``reason`` is ``timeout`` (the bound closed) or ``not_answered`` (something
    else woke the task — task mail, an owner hurry request, unconfirmed input —
    and ``woke_by`` names it). A host notice, never owner-marked content: the
    model must not read it as something the owner said. The card stays open,
    so the honest instruction depends on whether an assumption was recorded —
    without one, silence is explicitly NOT consent.
    """
    if reason == "not_answered":
        return {"role": "user", "content": (
            f"[SYSTEM NOTICE]\nThe wait on question {(checkpoint or {}).get('quiz_id')} ended "
            f"after {woke_by or 'unconfirmed input'}, not a confirmed owner answer. "
            "The question remains open and answerable. Continue by your judgment; "
            "ask again only if you must wait.")}
    if reason != "timeout":
        raise ValueError(f"unknown owner wait end reason: {reason!r}")
    minutes = int((checkpoint or {}).get("wait_max_minutes") or 0)
    window = f"within {minutes} minutes" if minutes > 0 else "within the requested window"
    assumption = ""
    try:
        from ouroboros.owner_quiz import quiz_states

        root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
        block = quiz_states(root, ctx.task_id).get(str((checkpoint or {}).get("quiz_id") or "")) or {}
        assumption = str(block.get("assumption") or "").strip()
    except Exception:
        assumption = ""
    stance = (f"Proceed under your stated assumption: {assumption}" if assumption
              else "No assumption was recorded; no answer is not consent")
    return {"role": "user", "content": (
        f"[SYSTEM NOTICE]\nNo owner answer arrived {window}. The question card stays open — "
        "a later answer reaches you as an ordinary owner message. "
        f"{stance} — or finish and say what is unresolved.")}


def owner_wait_timeout_notice(ctx: Any, checkpoint: dict) -> dict:
    """The bound's notice under its original name."""
    return owner_wait_ended_notice(ctx, checkpoint, "timeout")


def _wake_description(outcome: str, entries: list[dict]) -> str:
    """What woke the task, for the notice: a hurry is named as what it is."""
    if outcome == "hurry":
        return "an owner hurry request (a request to finish sooner, not an answer)"
    sources = sorted({str(row.get("source_task_id") or row.get("kind") or "mail") for row in entries})
    return f"task mail from {', '.join(sources)}" if sources else "unconfirmed input"


def _rendered_owner_authority(entries: list[dict]) -> bool:
    """Whether something drained reaches the transcript AS owner authority (the
    owner's words, a quiz answer, a principal task's message) and so explains
    the wake by itself. A hurry is owner authority too, but it is applied
    structurally and never rendered — the notice must name it."""
    from ouroboros.loop_messages import owner_authority_kinds
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, KIND_HURRY

    return any(kind not in {KIND_FINALIZE_NOW, KIND_HURRY} for kind in owner_authority_kinds(entries))


def append_wake_notice(ctx: Any, checkpoint: dict, outcome: Any, messages: list) -> None:
    """Explain a non-answer wake to the model, warm and cold alike.

    Only a wake that answered nothing gets a notice: this card's answer or the
    owner's words reach the transcript through the ordinary drain, and a
    control reason is acted on by the loop. An answer may have landed after
    the wake but before this round — never claim silence then. The acceptance
    park is not an owner question and gets no notice.
    """
    outcome = str(outcome or "")
    if checkpoint.get("reason") == REASON_OWNER_PAUSE_WARM:
        return  # ``park_owner_pause_warm`` writes the Resume notice itself
    if checkpoint.get("review_binding") or not (
            outcome in {"timeout", "hurry", "unknown"} or outcome.startswith("mail")):
        return
    from ouroboros.owner_quiz import quiz_states

    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    block = quiz_states(root, ctx.task_id).get(str(checkpoint.get("quiz_id") or "")) or {}
    if block.get("state") == "answered":
        return
    if outcome == "timeout":
        messages.append(owner_wait_ended_notice(ctx, checkpoint, "timeout"))
        return
    entries = _wait_entries(ctx)
    if _rendered_owner_authority(entries):
        return  # the owner's or a principal's words explain the wake themselves
    messages.append(owner_wait_ended_notice(ctx, checkpoint, "not_answered",
                                            woke_by=_wake_description(outcome, entries)))


def wait_after_tools(ctx: Any, messages: list, trace: dict, usage: dict,
                     round_idx: int, tool_schemas: list, seen: set,
                     *, review_binding: str = "") -> None:
    """Yield only after complete tool results; no model polling or terminal path."""
    if not getattr(ctx, "_owner_wait_requested", "") and not review_binding:
        return
    sleep = getattr(ctx, "_model_sleep", None)
    if isinstance(sleep, dict) and sleep.get("mode") == "cold" and not review_binding:
        # A cold sleep ends this process at the next round boundary
        # (``budget_pause.enter_cold_sleep``), not in a warm park here.
        ctx._owner_wait_requested = ""
        ctx._owner_wait_deadline_at = ""
        return
    callback = getattr(ctx, "owner_wait_callback", None)
    # A Presence author has only the narrow review-wait owner (``presence_continuation``): it
    # parks for its panel and never gains owner quizzes, sleeps or budget pauses through it.
    review_callback = getattr(ctx, "review_wait_callback", None) if review_binding else None
    if not callable(callback) and not callable(review_callback):
        raise RuntimeError("required owner wait has no worker continuation owner")
    checkpoint = checkpoint_owner_wait(ctx, messages, trace, usage, round_idx, tool_schemas, seen,
                                       review_binding=review_binding)
    sleep = checkpoint.get("sleep")
    if not callable(callback):
        review_callback(ctx, checkpoint, messages)
    elif sleep:
        from ouroboros import model_sleep

        model_sleep.begin(ctx)
        try:
            outcome = callback(ctx, checkpoint)
        finally:
            slept = model_sleep.end(ctx)  # through capacity reacquisition: the task runs again now
        messages.append(model_sleep.wake_notice(sleep, str(outcome or ""), slept, ctx))
        ctx._model_sleep = None
    else:
        outcome = callback(ctx, checkpoint)
        append_wake_notice(ctx, checkpoint, outcome, messages)
    ctx._owner_wait_requested = ""
    ctx._owner_wait_deadline_at = ""
    ctx._owner_wait_max_minutes = 0


def restore_continuation_state(tools: Any, state: dict, messages: list, trace: dict,
                               usage: dict, seen: set) -> None:
    """Rebind the saved cognition onto the live loop objects (shared by every
    same-ID continuation). Python handles (browser, executors, services) are
    NOT restored: they died with the previous process and stay invalidated."""
    from ouroboros.loop_delivery import DeliveryCandidate
    from ouroboros.model_wait import budget_paused_seconds

    ctx = tools._ctx
    for key, value in (state.get("context_observations") or {}).items():
        if key in {"_last_context_observation", "_inspected_context_view", "_pending_compaction", "_historical_author_inputs"}:
            setattr(ctx, key, value)
    messages[:] = state["messages"]
    trace.update(state["trace"])
    usage.update(state["usage"])
    seen.update(state["seen"])
    ctx._loop_mailbox_seen_ids = seen
    ctx._owner_directives = state["owner_directives"]
    # The cumulative budget-paused carrier rides EVERY same-ID continuation
    # (#1196, F5): a cold owner-wait restore of a task that had been budget
    # paused keeps it, so a later pause row and the delegate clock start from
    # the same cumulative value; a budget grant overrides it with its own.
    ctx._budget_paused_sec = budget_paused_seconds(state.get("model_wait") or {})
    for key, value in {**state["route"], **state["delivery"], **state["acceptance"]}.items():
        setattr(ctx, key, value)
    candidate = state.get("delivery_candidate")
    ctx._delivery_candidate = DeliveryCandidate(**{key: value for key, value in candidate.items()
        if key != "repair_attempted"}) if candidate else None
    if ctx._delivery_candidate is not None:
        value = ctx._delivery_candidate
        if value.control_episode_seen or value.finalization_control not in {"candidate", "owner_revision_required"}:
            ctx._delivery_control_required = True
            value.control_episode_seen = True


def rebind_restored_route(tools: Any, state: dict, messages: list) -> tuple:
    """Rebind the restored route's context-fit plan; returns ``(plan, mode)``."""
    from ouroboros.loop import _rebind_context_fit_plan, get_context_mode
    from ouroboros.model_slots import task_model_binding

    ctx = tools._ctx
    model_wait = getattr(ctx, "model_wait_context", None)
    role, account = task_model_binding(
        {"model_role": state.get("context_model_role"), "task_metadata": ctx.task_metadata},
        context_fit_plan=ctx.context_fit_plan,
        overrides=model_wait.overrides if model_wait is not None else None,
    )
    return _rebind_context_fit_plan(
        ctx.context_fit_plan, tools, messages, model=ctx.active_model,
        use_local=ctx.active_use_local, preferred_mode=get_context_mode(),
        tool_schemas=state["tool_schemas"],
        model_role=role, model_route={}, credential_profile_id=account,
    )


def resume_native_loop(tools: Any, state: dict, messages: list, trace: dict,
                       usage: dict, seen: set) -> tuple:
    """Restore the selected cold continuation and await its ordinary input grant."""
    ctx = tools._ctx
    restore_continuation_state(tools, state, messages, trace, usage, seen)
    cold_checkpoint = ctx.owner_wait_resume
    outcome = ctx.owner_wait_callback(ctx, cold_checkpoint)
    ctx.owner_wait_resume = None
    plan, mode = rebind_restored_route(tools, state, messages)
    messages.append({"role": "user", "content": (
        "[SYSTEM NOTICE]\nThis task continued from its saved owner wait after a planned restart. "
        "Prior tool results remain recorded; do not repeat completed effects. "
        "The restart ended the previous browser process and task-local services; "
        "their recorded results remain evidence, not proof they are still running.")})
    # The bound survived the restart as an absolute stamp, so the cold
    # continuation can ALSO end on it — and a peer wake or a hurry is no more
    # an answer cold than warm: the same notice seam says so.
    append_wake_notice(ctx, cold_checkpoint or {}, outcome, messages)
    return (ctx.active_model, ctx.active_effort, ctx.active_use_local,
            mode, state["round_idx"], plan)


def prepare_owner_wait_handoffs(root: Any, running: dict, transaction_id: str) -> set[str]:
    """Select only parked native continuations for the planned restart owner."""
    selected = set()
    for task_id, meta in running.items():
        task = meta.get("task") or {}
        row = load_task_result(root, task_id, strict=True) or {}
        wait = row.get("owner_wait") or {}
        if wait.get("state") != "waiting" or not wait.get("source_ref"):
            continue
        read_actor_source_bytes(root, task_id, wait["source_ref"])
        task["_owner_wait_resume"] = {**wait, "restart_transaction_id": transaction_id}
        selected.add(task_id)
    return selected
