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
from ouroboros.transcript_prefix import OBSERVER_STATE_ATTRS as _SEND_OBSERVER_KEYS  # a resumed task keeps the send observer's state
from ouroboros.observability import without_finalization_timing
from ouroboros.owner_mailbox import OwnerMailboxPeek
from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
from ouroboros.utils import utc_now_iso
from ouroboros.task_store import patch_document

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
        KIND_CARD_INPUT, KIND_FINALIZE_NOW, KIND_HURRY, KIND_OWNER_PAUSE, KIND_OWNER_TEXT, KIND_QUIZ_ANSWER,
        KIND_QUIZ_CLOSED, KIND_TASK_MESSAGE,
    )

    kinds = [str(row.get("kind") or KIND_OWNER_TEXT) for row in entries]
    if KIND_FINALIZE_NOW in kinds:
        return "control:finalize_now"
    if KIND_OWNER_PAUSE in kinds:
        return "control:owner_pause"
    # This card's answer: ``quiz_answer:{asker}:{quiz}``, or saved ``quiz_answer:{quiz}`` mail.
    answers = [str(row.get("msg_id") or "") for kind, row in zip(kinds, entries) if kind == KIND_QUIZ_ANSWER]
    if quiz_id and any(msg_id.startswith("quiz_answer:") and msg_id.rsplit(":", 1)[-1] == quiz_id
                       for msg_id in answers):
        return "answer"
    if any(kind == KIND_QUIZ_CLOSED and row.get("msg_id") == f"quiz_closed:{quiz_id}"
           for kind, row in zip(kinds, entries)):
        return "question_closed"
    if any(kind in {KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_CARD_INPUT} for kind in kinds):
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
        set(getattr(ctx, "_loop_mailbox_seen_ids", None) or ()), ctx.task_attempt or 1,
    )


def _fresh_wake(ctx: Any, quiz_id: str, outcome: str) -> str:
    """An owner answer can land while a pooled wait is awaiting capacity.

    Owner authority (this card's answer, owner words, a control) replaces the
    reason the resume was requested with; anything else replaces only
    ``unknown``, so a bound that ended the wait keeps saying so.
    """
    try:
        observed = classify_wake(_wait_entries(ctx), quiz_id)
    except Exception:
        if outcome == "unknown":
            raise  # A failed refresh cannot establish a wake on its own.
        log.warning("Wake refresh unreadable for %s; retaining %s", ctx.task_id, outcome, exc_info=True)
        return outcome
    if observed in {"answer", "owner_text", "control:finalize_now", "control:owner_pause"}:
        return observed
    if _question_closed(ctx, {"quiz_id": quiz_id}):
        return "question_closed"
    return observed if outcome == "unknown" else outcome


class OwnerWaitSuperseded(ValueError):
    """A compare-and-set lost: the wait row changed identity or state since its read."""


def set_owner_wait(root: Any, task_id: str, wait: Any,
                   expected_wait_id: str | None = None, *, expected_state: str | None = None) -> dict:
    """Update only the existing continuation projection, preserving siblings.

    A callable ``wait`` derives the new row from the CURRENT one inside this same write: a
    transition (``resumed``) never drops what another writer recorded on that wait meanwhile."""
    from ouroboros.task_results import (
        require_writable_task_result_schema,
        stamp_task_result_schema,
    )

    written: dict = {}

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
        written.clear()
        written.update(wait(dict(old)) if callable(wait) else wait)
        return stamp_task_result_schema({**current, "owner_wait": dict(written)})

    patch_document(root, task_id, update)
    return dict(written)


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

    ONE serializer for every same-ID continuation (owner wait, review park,
    budget pause): the carried fields are the loop's cognition — transcript,
    trace, usage, route, delivery candidate, admission-seal and owner-source
    identities, owner directives — so a second serializer could only drift from this one.
    """
    candidate = getattr(ctx, "_delivery_candidate", None)
    cost_ceiling = getattr(ctx, "_cost_ceiling", None)
    model_wait = getattr(ctx, "model_wait_context", None)
    model_state = model_wait.continuation_state() if model_wait is not None else {}
    from ouroboros.settings_integrity import captured_cognition

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
        "cognition": captured_cognition(),
        "nanny_activity": {key: getattr(ctx, key, None) for key in (
            "_nanny_metered_progress", "_nanny_delegate_baseline",
        )},
        "context_model_role": getattr(getattr(ctx, "context_fit_plan", None), "model_role", ""),
        # Exposure facts gate automatic compaction; a resumed task keeps them.
        "context_observations": {key: getattr(ctx, key) for key in (
            "_last_context_observation", "_inspected_context_view", "_pending_compaction",
            "_historical_author_inputs", "_pending_owner_dialogue", *_SEND_OBSERVER_KEYS,
        ) if getattr(ctx, key, None) is not None},
        "round_idx": round_idx, "tool_schemas": tool_schemas,
        **({"actor_loaded_tool_names": sorted(ctx._actor_loaded_tool_names)}
           if hasattr(ctx, "_actor_loaded_tool_names") else {}),
        "seen": sorted(seen), "owner_directives": getattr(ctx, "_owner_directives", []),
        "sleep_interaction_seen": dict(getattr(ctx, "_sleep_interaction_seen", None) or {}),
        "held_dependencies_seen": dict(getattr(ctx, "_held_dependencies_seen", None) or {}),
        "held_dependencies_order": dict(getattr(ctx, "_held_dependencies_order", None) or {}),
        "held_dependency_sources": dict(getattr(ctx, "_held_dependency_sources", None) or {}),
        "held_owner_messages": dict(getattr(ctx, "_held_owner_messages", None) or {}),
        "open_relays": dict(getattr(ctx, "_open_relays", None) or {}),
        # Own routing acts still unconfirmed: a resumed turn reports their outcomes too.
        "unconfirmed_routing_acts": list(getattr(ctx, "unconfirmed_routing_acts", None) or []),
        "restart": {
            "request": dict(getattr(ctx, "pending_restart_request", None) or {}),
            "reason": getattr(ctx, "pending_restart_reason", None),
            "evolution": bool(getattr(ctx, "pending_restart_is_evolution", False)),
        },
        "route": {key: getattr(ctx, key, None) for key in (
            "active_model", "active_effort", "active_use_local", "active_context_mode",
            "active_model_override", "active_effort_override", "active_use_local_override",
            "active_role_override", "route_wait_on_primary", "primary_route", "_route_facts_pending",
            "route_choice",  # following its role, or the task's own (route_choice)
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
            "_presence_selection_seq", "_presence_released",
            "_task_acceptance_sealed_fence_token", "_task_acceptance_sealed_fence_generation", "_owner_reply_revision",
            "_owner_message_sent",
        ) if getattr(ctx, key, None) is not None},
        # The block keeps its stored name; the task-acceptance flags it once carried are gone
        # (a pre-fold continuation still has them: ``restore_continuation_state`` ignores them).
        "acceptance": {
            # The validated wait/finish choice outlives its consumed selection; turn
            # start resets it, so only the current choice rides.
            "_acceptance_pending_review_choice": str(getattr(ctx, "_acceptance_pending_review_choice", "") or ""),
            # A left commit/change review wait, by value; its owner token says whether a
            # restored copy still has live custody in the resuming process.
            "_left_review_wait": (dict(getattr(ctx, "_left_review_wait", None))
                                  if isinstance(getattr(ctx, "_left_review_wait", None), dict) else None),
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
        # A model sleep uses model_sleep's source and question readiness contract.
        "reason": reason, **({"sleep": dict(sleep),
                                "sleep_started_at": float(getattr(ctx, "_model_sleep_started", 0) or time.time())}
                               if sleep else {}),
        **({"owner_pause": dict(pause)} if pause else {}),
        "review_binding": review_binding,
        # Date this wait independently of task lifetime; the stamp survives restart.
        # A review-bound park is dated by its review operation instead.
        **({} if review_binding else {"parked_at": utc_now_iso()}),
        "source_ref": source, "task_attempt": int(ctx.task_attempt or 1),
        "execution_drive_root": str(ctx.drive_root),
        "started_at": getattr(ctx, "task_started_at", None),
        "model_wait_quota_clock": model_state.get("quota_clock", {}),
        # The SAME two carriers every finite-lifetime reader subtracts: the quota
        # union above and the budget-paused interval (#1196, F5). Without it the
        # row's own lifetime check would count a pause as execution.
        **({"budget_paused_sec": model_state["budget_paused_sec"]}
           if "budget_paused_sec" in model_state else {}),
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


def governing_continuation(task: dict) -> str:
    """Name the ONE carrier the loop resumes: owner wait, then exact pause, then saved work.

    The loop loads the first present carrier and never falls back when it fails,
    so admission validates that carrier alone. A later carrier legitimately
    retains the earlier saved-work locator; a valid newer one never reads it, and
    an invalid one never falls back to it. Falsy wait/pause fields are absent;
    working recovery is selected by presence, even with an unusable locator.
    """
    if task.get("_owner_wait_resume"):
        return "owner_wait"
    if task.get("_budget_pause_resume"):
        return "exact_pause"
    return "working" if "_working_recovery" in task else ""


def fresh_input(task: dict) -> bool:
    """The task's own first attempt: no governing continuation, retry or reissue of another task."""
    return not (governing_continuation(task) or int(task.get("_attempt") or 1) != 1
                or task.get("original_task_id") or task.get("timeout_retry_from"))


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


def restore_owner_wait_allowed(root: Any, task: dict, *, strict: bool = False, expiring: bool = False) -> bool:
    """Current wait and acknowledged restart authorize a locator; strict preserves read failures.

    ``expiring``: the caller is settling the task's explicit deadline (#939), so only that
    elapsed deadline is not a refusal; every other check and read still binds."""
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from ouroboros.delegate_recovery import _ack_direct_exec_successor, _read_restart_transaction, _restart_transaction_path
    from ouroboros.config import resolve_task_execution_timeout_sec
    from ouroboros.model_wait import execution_elapsed_seconds
    import time

    handoff = task.get("_owner_wait_resume")
    if not isinstance(handoff, dict):
        return False
    root = pathlib.Path(root)
    from ouroboros.delegate_recovery import owner_restart_in_progress
    if owner_restart_in_progress(root):
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
    if deadline is not None and deadline <= utc_now() and not expiring:
        return False
    started = float(handoff.get("started_at") or wait.get("started_at") or 0)
    now = time.time()
    ceiling = resolve_task_execution_timeout_sec(task)  # None = no execution bound to have outlived
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
        {"task": task, "started_at": started,
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
    """Keep this stack until the pool grants capacity. Mail and the wait's own
    bound request Resume only after park; hard controls also end a pre-ACK wait.
    A request grants nothing: current cancellation and writer gates still bind.
    """
    import os

    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from ouroboros.budget_pause import _hold_control_reason

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
    from ouroboros import custody_owed

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
        if custody_owed.pending():  # this park's own poll services what the worker owes (its wake lands here too)
            custody_owed.drain()
        if not parked and (reason := _hold_control_reason(ctx)):
            from ouroboros.model_wait import ModelWaitInterrupted

            raise ModelWaitInterrupted(reason)
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


_CLOSURE_READS: dict = {}  # (task_id, quiz_id) -> (task record signature, closed)


def _question_closed(ctx: Any, checkpoint: dict) -> bool:
    """Only this addressed canonical question can release its paired wait.

    Closure requests the ordinary capacity grant. It grants no owner authority
    and does not mutate the waiting projection before the supervisor's grant.
    A Pause or sleep park carries no quiz id. The parked poll re-reads the
    record only when its store signature (``write_seq``, ``updated_at``) changed.
    """
    quiz_id = str((checkpoint or {}).get("quiz_id") or "")
    if not quiz_id or checkpoint.get("review_binding") or checkpoint.get("sleep"):
        return False
    from ouroboros.owner_quiz import quiz_states
    from ouroboros.store import StoreError
    from ouroboros.task_store import signature

    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    key = (str(ctx.task_id), quiz_id)
    try:
        stamp = signature(root, ctx.task_id)
    except StoreError:
        stamp = None
    cached = _CLOSURE_READS.get(key)
    if stamp is not None and cached is not None and cached[0] == stamp:
        return cached[1]
    closed = (quiz_states(root, ctx.task_id).get(quiz_id) or {}).get("state") == "closed"
    if len(_CLOSURE_READS) >= 256:
        _CLOSURE_READS.clear()
    if stamp is not None:
        _CLOSURE_READS[key] = (stamp, closed)
    return closed


def _sleep_wake(ctx: Any, checkpoint: dict) -> str | None:
    """``None`` for an owner/review wait (any mail wakes it; ``review_settled`` once a left
    review's wave settled) unless its own question was explicitly closed; for a model sleep, the
    reason its selected sources give NOW (``""`` = keep sleeping), read from the
    canonical records on every poll — the recheck after the park that closes the
    race with an event landing between the tool's check and the park."""
    chosen = (checkpoint or {}).get("sleep")
    if not isinstance(chosen, dict):
        # A left review's park also ends at its wave's settlement, notice written or not.
        left = getattr(ctx, "_left_review_wait", None)
        binding = (checkpoint or {}).get("review_binding")
        if binding and isinstance(left, dict) and left.get("binding") == binding:
            from ouroboros.review_waiting import wake_reason
            reason = wake_reason(ctx, left)
            if reason:
                return "helper_result" if reason.startswith(("task:", "run:")) else reason
            return str(left.get("wake_reason") or "review_settled") if left.get("done") else ""
        return "question_closed" if _question_closed(ctx, checkpoint) else None
    from ouroboros.model_sleep import wake_reason
    from ouroboros.store import acquisition_budget

    wait_id = checkpoint.get("wait_id") or id(chosen)
    try:
        with acquisition_budget(0):  # store contention is retried by the next poll, never waited on
            reason = wake_reason(ctx, chosen)
        ctx._sleep_read_failures = None
        return reason
    except Exception:
        previous = getattr(ctx, "_sleep_read_failures", None)
        count = previous[1] + 1 if previous and previous[0] == wait_id else 1
        ctx._sleep_read_failures = (wait_id, count)
        log.warning("Sleep readiness unreadable for %s; still sleeping (consecutive failures: %d)",
                    ctx.task_id, count, exc_info=count == 1)
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
    from ouroboros.external_runs import STOP_POLICY_TASK_OWNED, episode_fields, observe_task_runs

    try:  # the member's OWN runs stop exactly as at a cold Pause; its reviewers' runs are spared
        from ouroboros import delegate_custody as custody

        external = observe_task_runs(custody.custody_root(ctx), str(ctx.task_id),
                                     reason="owner_pause_warm_park", stop_policy=STOP_POLICY_TASK_OWNED)
    except Exception as exc:
        external = {"runs": [], "custody_read": "failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    pause = {"fence_id": str(fence.get("fence_id") or ""), "generation": int(fence.get("generation") or 0),
             "root_task_id": str(fence.get("root_task_id") or ""), "detached_reviews": list(detached),
             "external_runs": external, **episode_fields({}, external, starts=True), "parked_at": utc_now_iso()}
    from ouroboros import model_sleep

    # Reuse the live paused interval and fold it into the same cumulative carrier
    # once, including capacity reacquisition. Reviewer owners keep their own clocks.
    model_sleep.begin(ctx)
    try:
        checkpoint = checkpoint_owner_wait(ctx, messages, trace, usage_for_state, int(limit_ctx.round_idx),
                                           list(limit_ctx.tool_schemas or []),
                                           set(limit_ctx.owner_msg_seen or ()), pause=pause)
        checkpoint["sleep_started_at"] = ctx._model_sleep_started
        outcome = str(callback(ctx, checkpoint) or "")
    finally:
        model_sleep.end(ctx)
    if outcome == "control:owner_resume":
        messages.append(owner_pause_resume_notice(ctx, checkpoint, outcome))
    model_sleep.announce_resumed(ctx, outcome)
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
        if fence_closed(fence) or ((fence.get("resume_grant") or {}).get("tree_only") and task_id != root_task_id):
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


def owner_pause_resume_notice(ctx: Any, checkpoint: dict, outcome: str, *, cold: bool = False) -> dict:
    """The host frame a warm-parked member reads when the owner's Pause ends.

    States only what this process knows about each review it detached from:
    still ``running``, ``closed`` here (its workers settled and published), or
    ``unknown``. Re-calling the same review with an unchanged subject rejoins
    or replays that attempt in this process; nothing is sent again by itself.
    """
    from ouroboros.review_pause import review_operation_state

    pause = checkpoint.get("owner_pause") if isinstance(checkpoint.get("owner_pause"), dict) else {}
    if cold:
        return {"role": "user", "content": (
            "[SYSTEM NOTICE]\nThe application stopped while this owner Pause was warm. "
            "Detached review operations recorded at the pause: "
            + json.dumps(pause.get("detached_reviews") or [], ensure_ascii=False)
            + ". Their former process-local state is not current liveness or a verdict. "
            "Collect the original review records before deciding what remains; "
            "an unknown paid outcome is not permission to dispatch it again.")}
    lines = []
    for item in pause.get("detached_reviews") or []:
        if not isinstance(item, dict):
            continue
        state = review_operation_state(str(item.get("owner_id") or ""))
        lines.append(f"- {item.get('surface') or 'review'} operation {item.get('owner_id')}: {state}")
    reviews = ("\n".join(lines) if lines else "- none")
    from ouroboros.external_runs import paused_runs_notice

    runs = paused_runs_notice(ctx, _parked_owner_pause(ctx, checkpoint, pause))
    abandoned = list(getattr(ctx, "_abandoned_model_attempts", None) or [])
    ctx._abandoned_model_attempts = []
    late = (f"The Pause interrupted {len(abandoned)} model request(s) already sent ({', '.join(abandoned)}): "
            "the provider may still have finished and charged them; any late answer was recorded but NOT used "
            "and its tools were not run. " if abandoned else "")
    return {"role": "user", "content": (
        "[SYSTEM NOTICE]\nThe owner paused this task and has now resumed it. Reviewers you had already "
        f"launched kept running while you were paused:\n{reviews}\nWhat became of the delegated runs this Pause "
        f"stopped or observed (current custody; engine facts are evidence, never a grant):\n{runs}\n"
        "Your review tool returned a pending "
        "result for them; nothing was committed, applied or published after the Pause. Calling the same "
        "review again with an unchanged subject rejoins or replays that attempt in this process without a "
        "new dispatch; a changed subject is a new review. A reviewer whose state is unknown was not resent. "
        f"{late}Nothing you asked for after the Pause ran. Cumulative spend, rounds and elapsed time were "
        "not reset; prior tool results remain recorded; do not repeat completed effects.")}


def _parked_owner_pause(ctx: Any, checkpoint: dict, pause: dict) -> dict:
    """This park's ``owner_pause`` with the participants its settle tick recorded since (same wait
    only); a park without a whole list stays partial (``external_runs.episode_fields``)."""
    from ouroboros.external_runs import PARTIAL_KEY, episode_fields

    try:
        wait = (load_task_result(pathlib.Path(ctx.budget_drive_root or ctx.drive_root), ctx.task_id, strict=True)
                or {}).get("owner_wait") or {}
    except Exception:
        return pause  # the park's own list stands; the notice still reads current custody
    later = (wait.get("owner_pause") or {}) if wait.get("wait_id") == checkpoint.get("wait_id") else {}
    merged = {**pause, **episode_fields(pause, {"runs": later.get("episode_runs")})}
    return {**merged, PARTIAL_KEY: True} if later.get(PARTIAL_KEY) is True else merged


def _admit_direct_continuation_wake(task_id: str, wait: dict, root: pathlib.Path) -> bool:
    """Clear park mark under the repo-writer gate lock; re-park if admission is closed."""
    from supervisor import workers
    from supervisor.active_activity import get_direct_activity_registry

    registry = get_direct_activity_registry()
    with workers._repo_writer_gate_lock:
        registry.clear_parked(task_id)
        if workers.repo_writer_admission_closed():
            registry.mark_parked(task_id, wait, drive_root=root)
            return False
        return True


def direct_owner_wait(ctx: Any, checkpoint: dict) -> str:
    """Retain a registered chat actor's stack; it holds no pooled capacity.

    The existing mailbox still owns input and its loop still owns delivery.
    TaskModelWait supplies the same Stop/deadline clocks as native model calls;
    this owner wait does not enter a quota pause or grant cold restart authority.
    An optional bound releases the loop only AFTER those controls are consulted,
    so Stop, the task deadline and the absolute ceiling keep precedence.
    """
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from supervisor.active_activity import get_direct_activity_registry

    control = ctx.model_wait_context
    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    if getattr(ctx, "event_queue", None) is None:
        from ouroboros.presence_owner_contact import flush_owner_events
        flush_owner_events(root, ctx.pending_events)
    # Without a queue (a Presence turn may have none) the buffer keeps its events for the final flush.
    while ctx.pending_events and getattr(ctx, "event_queue", None) is not None:
        ctx.event_queue.put(dict(ctx.pending_events[0]))
        del ctx.pending_events[0]
    wait = set_owner_wait(root, ctx.task_id, {**checkpoint, "state": "waiting", "is_direct_chat": True})
    get_direct_activity_registry().mark_parked(ctx.task_id, wait, drive_root=root)
    peek = OwnerMailboxPeek()
    deadline = parse_deadline_ts((checkpoint or {}).get("wait_deadline_at"))  # None = unbounded
    outcome = "unknown"
    warm_pause = checkpoint.get("reason") == REASON_OWNER_PAUSE_WARM
    control_reason = ""
    try:
        while True:
            control_reason = str(control.control_reason() or "")
            if control_reason:
                if control_reason not in {"finalize_requested", "finalize_now"}:
                    break
                if _admit_direct_continuation_wake(ctx.task_id, wait, root):
                    break
                time.sleep(0.05)
                continue
            woke = _sleep_wake(ctx, checkpoint)
            if woke:
                if _admit_direct_continuation_wake(ctx.task_id, wait, root):
                    outcome = woke
                    break
                time.sleep(0.05)
                continue
            if warm_pause:
                woke = _owner_pause_wake(ctx, checkpoint)
                if woke:
                    if (woke.startswith("control:")
                            and woke not in {"control:owner_resume", "control:owner_pause",
                                             "control:finalize_requested", "control:finalize_now"}):
                        outcome = woke
                        break
                    if _admit_direct_continuation_wake(ctx.task_id, wait, root):
                        outcome = woke
                        break
                    time.sleep(0.05)
                    continue
            elif woke is None and peek.pending(
                    pathlib.Path(ctx.drive_root), ctx.task_id,
                    set(getattr(ctx, "_loop_mailbox_seen_ids", set())), ctx.task_attempt or 1):
                if _admit_direct_continuation_wake(ctx.task_id, wait, root):
                    break
                time.sleep(0.05)
                continue
            if deadline is not None and utc_now() >= deadline:
                if _admit_direct_continuation_wake(ctx.task_id, wait, root):
                    outcome = "timeout"
                    break
                time.sleep(0.05)
                continue
            time.sleep(1.0)
    finally:
        get_direct_activity_registry().clear_parked(ctx.task_id)
    if not control_reason:
        # Every continuation exit above was already admitted (or is terminal); a late
        # control only joins it. Re-checking admission here could re-park a stack that
        # is already resuming and hide it from the update drain (#2191).
        control_reason = str(control.control_reason() or "")
    outcome = (f"control:{control_reason}" if control_reason else outcome if warm_pause else
               _fresh_wake(ctx, str(checkpoint.get("quiz_id") or ""), outcome))
    # The current same wait, in one write: a settle tick's facts recorded meanwhile stay (#1562).
    set_owner_wait(root, ctx.task_id, lambda current: {**current, "state": "resumed", "resume_reason": outcome},
                   wait["wait_id"])
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
        from ouroboros.owner_quiz import mark_wait_ended, quiz_states
        from ouroboros.presence_owner_contact import private_main_contact

        changed = mark_wait_ended(root, task_id, quiz_id)
        block = quiz_states(root, task_id).get(quiz_id) or {}
        chat_id = int(block.get("publication_chat_id", 1 if private_main_contact(block, task_id) else chat_id))
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
    the wake but before this round — never claim silence then. A review park
    is not an owner question and gets no notice.
    """
    outcome = str(outcome or "")
    if checkpoint.get("reason") == REASON_OWNER_PAUSE_WARM:
        return  # ``park_owner_pause_warm`` writes the Resume notice itself
    if checkpoint.get("review_binding") and (outcome in {"helper_result", "review_result", "timeout"} or outcome.startswith("run_question:")):
        from ouroboros.review_waiting import append_helper_notice
        append_helper_notice(ctx, messages)
    if checkpoint.get("review_binding") or not (
            outcome in {"timeout", "hurry", "unknown", "question_closed"} or outcome.startswith("mail")):
        return
    from ouroboros.owner_quiz import quiz_states

    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    block = quiz_states(root, ctx.task_id).get(str(checkpoint.get("quiz_id") or "")) or {}
    if block.get("state") == "answered":
        return
    if block.get("state") == "closed":
        if block.get("closed_actor") == "owner" and any(
                row.get("kind") == "quiz_closed" for row in _wait_entries(ctx)):
            return  # The typed closure frame reaches the ordinary mailbox drain.
        actor = "the owner" if block.get("closed_actor") == "owner" else f"task {block.get('closed_by_task_id')}"
        messages.append({"role": "user", "content": (
            f"[SYSTEM NOTICE]\nQuestion {checkpoint.get('quiz_id')} was explicitly closed "
            f"by {actor}: {block.get('closed_reason')}. "
            "Its paired wait has ended without an owner answer. Closure is not owner approval.")})
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
    from ouroboros.model_detach import preserve_owner_question
    preserve_owner_question(ctx, messages)
    sleep = getattr(ctx, "_model_sleep", None)
    if isinstance(sleep, dict) and sleep.get("mode") == "cold" and not review_binding:
        # A cold sleep ends this process at the next round boundary
        # (``budget_pause.enter_cold_sleep``), not in a warm park here.
        ctx._owner_wait_requested = ""
        ctx._owner_wait_deadline_at = ""
        return
    callback = getattr(ctx, "owner_wait_callback", None)
    # Presence validates its own review/private question at this narrow seam;
    # the callback grants no sleep, detach, owner Pause or budget-pause capability.
    review_callback = getattr(ctx, "review_wait_callback", None)
    if not callable(callback) and not callable(review_callback):
        raise RuntimeError("required owner wait has no worker continuation owner")
    from ouroboros import model_sleep

    if not review_binding and model_sleep.withdraw_for_new_holds(ctx, messages) and not ctx._owner_wait_requested:
        return  # a stop begun since arming reaches the mind before the park; the same request then sleeps
    checkpoint = checkpoint_owner_wait(ctx, messages, trace, usage, round_idx, tool_schemas, seen,
                                       review_binding=review_binding)
    if not callable(callback):
        outcome = review_callback(ctx, checkpoint, messages)
        append_wake_notice(ctx, checkpoint, outcome, messages)
    elif sleep := checkpoint.get("sleep"):
        model_sleep.begin(ctx)
        checkpoint["sleep_started_at"] = ctx._model_sleep_started
        try:
            outcome = callback(ctx, checkpoint)
        finally:
            slept = model_sleep.end(ctx)  # through capacity reacquisition: the task runs again now
        messages.append(model_sleep.wake_notice(sleep, str(outcome or ""), slept))
        model_sleep.append_question_notice(ctx, str(outcome or ""), messages)
        ctx._model_sleep = None
    else:
        outcome = callback(ctx, checkpoint)
        append_wake_notice(ctx, checkpoint, outcome, messages)
    ctx._owner_wait_requested, ctx._owner_wait_deadline_at, ctx._owner_wait_max_minutes = "", "", 0
    if callable(callback):
        model_sleep.announce_resumed(ctx, outcome)  # the same attempt prepares again, with no new start


def park_for_left_review(tools: Any, limit_ctx: Any, content: Any, emit_progress: Any,
                         origin: str = "prose") -> bool:
    """Plain prose with no completion choice while a commit/change review the author left
    for an addressed message is still its intention: the prose stays in the transcript as
    narration and the same wait resumes on the review park, with no model call until the
    settlement (``_sleep_wake``) or another addressed event. A notice written but not yet read
    gets one round, and one that could not be written is said inline; a wait restored from
    another process holds no live custody here, so the author gets one round with that fact
    instead of a park nothing would end. A new commit or review_change act, an explicit
    completion, a new turn or a read notice ends it. Each case that holds the prose records
    its ``left_review_wait`` hold itself (``origin`` from the caller); a park before it waits."""
    from ouroboros.review_pause import PROCESS_TOKEN

    def held(case: str) -> bool:
        from ouroboros.loop_delivery import observe_completion_hold

        observe_completion_hold(tools, limit_ctx, origin, "left_review_wait", case)
        return True

    ctx = tools._ctx
    wait = getattr(ctx, "_left_review_wait", None)
    if not isinstance(wait, dict):
        return False
    restored = wait.get("owner") != PROCESS_TOKEN
    if str(wait.get("msg_id") or "") in limit_ctx.owner_msg_seen:
        ctx._review_wait_selection = dict(wait)
        ctx._left_review_wait = None
        return False  # its notice was read: that wave is over
    from ouroboros.loop_messages import _emit_round_progress

    if str(content or "").strip():
        limit_ctx.messages.append({"role": "assistant", "content": str(content)})
        _emit_round_progress(content, {"content": content}, emit_progress, limit_ctx.llm_trace)
    if restored:
        from ouroboros.review_pause import left_review_custody_note

        ctx._left_review_wait = None
        limit_ctx.messages.append({"role": "user", "content": left_review_custody_note(wait)})
        return held("custody_restored")
    parked = not wait.get("done")
    if parked:
        from ouroboros.review_waiting import wake_reason
        if reason := wake_reason(ctx, wait):
            append_wake_notice(ctx, {"review_binding": wait.get("binding")},
                               "helper_result" if reason.startswith(("task:", "run:")) else reason, limit_ctx.messages)
            return held("wake_notice")
        held("parked")  # before the wait: the hold is readable while it lasts and after a wait that ended it
        wait_after_tools(ctx, limit_ctx.messages, limit_ctx.llm_trace, limit_ctx.accumulated_usage,
                         limit_ctx.round_idx, limit_ctx.tool_schemas, limit_ctx.owner_msg_seen,
                         review_binding=str(wait.get("binding") or ""))
    if wait.get("done") and not wait.get("noticed"):  # settled, but its notice could not be written
        ctx._left_review_wait = None
        if wait.get("notice"):
            limit_ctx.messages.append({"role": "user", "content": str(wait["notice"])})
    return True if parked else held("settled")


def restore_continuation_state(tools: Any, state: dict, messages: list, trace: dict,
                               usage: dict, seen: set) -> None:
    """Rebind the saved cognition onto the live loop objects (shared by every
    same-ID continuation). Python handles (browser, executors, services) are
    NOT restored: they died with the previous process and stay invalidated."""
    from ouroboros.loop_delivery import DeliveryCandidate

    from ouroboros.model_wait import adopt_handoff_clock, budget_paused_seconds
    from ouroboros.settings_integrity import restore_cognition

    ctx = tools._ctx
    restore_cognition(state.get("cognition"))
    # Missing old observations are unknown, not this new process's startup seed.
    # None is the known fresh state; an empty mark has no historical anchor.
    for key in ("_nanny_metered_progress", "_nanny_delegate_baseline"):
        setattr(ctx, key, (state.get("nanny_activity") or {}).get(key, {}))
    for key, value in (state.get("context_observations") or {}).items():
        if key in {"_last_context_observation", "_inspected_context_view", "_pending_compaction", "_historical_author_inputs", "_pending_owner_dialogue", *_SEND_OBSERVER_KEYS}:
            setattr(ctx, key, value)
    messages[:] = state["messages"]
    # This set is also bound into enable_tools; never replace the closure's object.
    loaded = getattr(ctx, "_actor_loaded_tool_names", None)
    if loaded is not None:
        loaded.update(state.get("actor_loaded_tool_names") or ())
    trace.update(state["trace"])
    usage.update(state["usage"])
    seen.update(state["seen"])
    ctx._loop_mailbox_seen_ids = seen
    ctx._sleep_interaction_seen = dict(state.get("sleep_interaction_seen") or {})
    # A legacy state without this map shows each current hold once more before its wait arms.
    ctx._held_dependencies_seen = dict(state.get("held_dependencies_seen") or {})
    # Each record's place: an older source read whole later never replaces a newer one (``model_sleep._record_holds``).
    ctx._held_dependencies_order = dict(state.get("held_dependencies_order") or {})
    # A shortened hold view's exact source stays readable to completion (``model_sleep.credit_hold_source``).
    ctx._held_dependency_sources = dict(state.get("held_dependency_sources") or {})
    ctx._owner_directives = state["owner_directives"]
    ctx._held_owner_messages = dict(state.get("held_owner_messages") or {})
    ctx._open_relays = dict(state.get("open_relays") or {})
    ctx.unconfirmed_routing_acts = [dict(act) for act in state.get("unconfirmed_routing_acts") or [] if isinstance(act, dict)]
    restart = state.get("restart") or {}
    request = restart.get("request") or {}
    # Cold Pause/Resume keeps the requester's obligation without sending again.
    # A new task or attempt inherits no authority to finish this Restart.
    if (request.get("task_id"), request.get("task_attempt")) == (
            ctx.task_id, int(getattr(ctx, "task_attempt", None) or 1)):
        ctx.pending_restart_request = dict(request)
        ctx.pending_restart_reason = restart.get("reason")
        ctx.pending_restart_is_evolution = bool(restart.get("evolution"))
    # Match the agent's native handoff clock; a budget grant overrides it separately.
    ctx._budget_paused_sec = budget_paused_seconds(adopt_handoff_clock(
        state.get("model_wait") or {}, getattr(ctx, "owner_wait_resume", None)))
    # A pre-fold continuation's ``acceptance`` block also carries the deleted task-acceptance
    # flags (``_task_acceptance_pending`` and friends); they are not restored.
    review = {key: value for key, value in state["acceptance"].items()
              if key in {"_acceptance_pending_review_choice", "_left_review_wait"}}
    for key, value in {**state["route"], **state["delivery"], **review}.items():
        setattr(ctx, key, value)
    from ouroboros.route_choice import restored_choice

    ctx.route_choice = restored_choice(ctx, state)  # a saved choice, else the legacy classification (D5)
    ctx._route_binding = None  # the first round compares to the restored acting route (route_choice.round_route)
    candidate = state.get("delivery_candidate")
    ctx._delivery_candidate = DeliveryCandidate(**{key: value for key, value in candidate.items()
        if key != "repair_attempted"}) if candidate else None
    if ctx._delivery_candidate is not None:
        value = ctx._delivery_candidate
        if value.control_episode_seen or value.finalization_control not in {"candidate", "owner_revision_required"}:
            ctx._delivery_control_required = True
            value.control_episode_seen = True


def rebind_restored_route(tools: Any, state: dict, messages: list) -> tuple:
    """Rebind the restored route's context-fit plan; returns ``(plan, mode)``.

    The actor's captured owner choice survives restart, separately from the physically
    achieved mode. Legacy sources without that choice keep the previous fallback."""
    from ouroboros.loop import _rebind_context_fit_plan, get_context_mode
    from ouroboros.memory_floor import MODES
    from ouroboros.model_slots import task_model_binding
    from ouroboros.continuation_tools import reconcile_restored_tools

    ctx = tools._ctx
    model_wait = getattr(ctx, "model_wait_context", None)
    role, account = task_model_binding(
        {"model_role": state.get("context_model_role"), "task_metadata": ctx.task_metadata},
        context_fit_plan=ctx.context_fit_plan,
        overrides=model_wait.overrides if model_wait is not None else None,
    )
    owner, ran = get_context_mode(), getattr(ctx, "active_context_mode", None)
    route = dict(model=ctx.active_model,
        use_local=ctx.active_use_local, preferred_mode=owner,
        tool_schemas=state["tool_schemas"],
        model_role=role, model_route={}, credential_profile_id=account,
    )
    plan, mode = _rebind_context_fit_plan(ctx.context_fit_plan, tools, messages, **route,
        start_mode=ran if owner in MODES and ran in MODES[MODES.index(owner):] else None)
    if reconcile_restored_tools(tools, state, messages, mode):
        plan, mode = _rebind_context_fit_plan(plan, tools, messages, **route, start_mode=mode)
    return plan, mode


def resume_native_loop(tools: Any, state: dict, messages: list, trace: dict,
                       usage: dict, seen: set) -> tuple:
    """Restore the selected cold continuation and await its ordinary input grant."""
    from ouroboros.review_pause import wake_restored_left_review
    from ouroboros.model_wait import native_sleep_started_at
    from ouroboros import model_sleep

    ctx = tools._ctx
    restore_continuation_state(tools, state, messages, trace, usage, seen)
    cold_checkpoint = ctx.owner_wait_resume
    wake_restored_left_review(ctx)  # departed review custody is disclosed and wakes its author
    started = native_sleep_started_at(cold_checkpoint)
    if started is not None:
        model_sleep.begin(ctx, started_at=started)
    try:
        outcome = ctx.owner_wait_callback(ctx, cold_checkpoint)
    finally:
        if started is not None:
            model_sleep.end(ctx)
    model_sleep.announce_resumed(ctx, outcome)
    ctx.owner_wait_resume = None
    plan, mode = rebind_restored_route(tools, state, messages)
    messages.append({"role": "user", "content": (
        "[SYSTEM NOTICE]\nThis task continued from its saved owner wait after a planned restart. "
        "Prior tool results remain recorded; do not repeat completed effects. "
        "The restart ended the previous browser process and task-local services; "
        "their recorded results remain evidence, not proof they are still running.")})
    left = getattr(ctx, "_left_review_wait", None)
    if isinstance(left, dict) and left.get("done") and not left.get("noticed") and left.get("notice"):
        messages.append({"role": "user", "content": str(left["notice"])})
        ctx._left_review_wait = None
    # The same absolute bound and wake semantics survive a cold return.
    if (cold_checkpoint or {}).get("reason") == "sleep":
        messages.append(model_sleep.wake_notice(cold_checkpoint.get("sleep") or {}, str(outcome or ""),
                                                max(0.0, time.time() - started) if started is not None else None))
        model_sleep.append_question_notice(ctx, str(outcome or ""), messages)
    else:
        append_wake_notice(ctx, cold_checkpoint or {}, outcome, messages)
    return (ctx.active_model, ctx.active_effort, ctx.active_use_local,
            mode, state["round_idx"], plan)


def prepare_owner_wait_handoffs(root: Any, running: dict, transaction_id: str, *,
                                include_sleep: bool = True, direct: Any = None) -> set[str]:
    """Prepare returning waits; owner-Pause keeps its own retention.

    Planned restarts and updates may return a sleeper to its existing wait.
    Manual Restart excludes sleep: it stays held until explicit Resume.
    """
    selected = set()
    reasons = {"owner", "review", "sleep"} if include_sleep else {"owner", "review"}
    if direct is None:
        try:
            from supervisor.workers import direct_chat_turns

            direct = direct_chat_turns()
        except Exception:
            direct = ()
    candidates = [
        *((str(task_id or ""), (meta or {}).get("task") or {}) for task_id, meta in (running or {}).items()),
        *((str((task or {}).get("id") or ""), task if isinstance(task, dict) else {}) for task in (direct or ())),
    ]
    for task_id, task in candidates:
        if not task_id or task_id in selected:
            continue
        try:
            row = load_task_result(root, task_id, strict=True) or {}
        except (OSError, ValueError, TypeError, RuntimeError):
            log.warning("Restart handoff could not read task %s; its retention stays unresolved", task_id, exc_info=True)
            continue
        wait = row.get("owner_wait") or {}
        if (wait.get("state") != "waiting" or wait.get("reason") not in reasons
                or not wait.get("source_ref")):
            continue
        read_actor_source_bytes(root, task_id, wait["source_ref"])
        task["_owner_wait_resume"] = {**wait, "restart_transaction_id": transaction_id}
        selected.add(task_id)
    return selected
