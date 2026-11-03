"""The root admission seal: the queue-owned compare-and-seal a root takes at an author stop.

Sealing closes the root's admission — no new subtask under it, and an owner message
that arrived between "decided" and "delivered" is caught by generation instead of
being answered by an earlier decision (#406). The supervisor owns the record under
its queue lock; this module is the worker's client: begin, inspect, end, and the
release when an owner follow-up reopens a sealed root.

The mechanism was born as the task-acceptance fence. Its persisted names stay as
they were stored — ``acceptance_fences`` in the queue snapshot, the
``acceptance_fence`` event, ``state/acceptance_fence_acks/``, the ``*_acceptance_fence``
transport seam on the context and the ``_task_acceptance_*fence*`` context fields a
same-ID continuation carries — so this client keeps the begin/end names that match
them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict

log = logging.getLogger("ouroboros.loop")


def _loop():
    """The parent loop module, read at call time (tests rebind members there)."""
    from ouroboros import loop

    return loop


@dataclass(frozen=True)
class FenceOutcome:
    """Typed answer to one queue-fence request; truthy iff the supervisor said yes.

    ``refused``: it answered no (``reason``). ``unknown``: no answer arrived within
    ``waited_sec`` — a gap, never a refusal, a verdict or an owner message.
    """

    status: str = "ok"
    op: str = ""
    reason: str = ""
    waited_sec: float = 0.0

    def __bool__(self) -> bool:
        return self.status == "ok"


def _root_task_id(ctx: Any, task_id: str) -> str:
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    return str(meta.get("root_task_id") or getattr(ctx, "root_task_id", "") or task_id)


def _settle_fence_outcome(ctx: Any, outcome: FenceOutcome) -> FenceOutcome:
    """Expose the outcome on ctx; a refusal or a gap leaves ONE durable worker-side row."""
    ctx._task_acceptance_fence_outcome = outcome
    if not outcome:
        from ouroboros import task_pacing
        from ouroboros.utils import append_jsonl, utc_now_iso

        task_id = str(getattr(ctx, "task_id", "") or "")
        try:
            append_jsonl(task_pacing.acceptance_timing_events_path(ctx), {
                "ts": utc_now_iso(), "type": "supervisor_ack_unavailable", "task_id": task_id,
                "root_task_id": _root_task_id(ctx, task_id), "op": outcome.op, "outcome": outcome.status,
                "reason": outcome.reason, "waited_sec": outcome.waited_sec,
            })
        except Exception:
            log.warning("supervisor_ack_unavailable row could not be written for %s", task_id, exc_info=True)
    return outcome


def _fence_request(ctx: Any, op: str, callback: Callable[..., Any], **kwargs: Any) -> tuple[FenceOutcome, Any]:
    """One request to the queue-owned fence: ok | refused(reason) | unknown(waited_sec)."""
    started = time.monotonic()
    try:
        response = callback(**kwargs)
        if not (isinstance(response, dict) and not response.get("ok", True)):
            return _settle_fence_outcome(ctx, FenceOutcome(op=op)), response
        outcome = FenceOutcome("refused", op, str(response.get("error") or response.get("status") or ""))
    except RuntimeError as exc:
        outcome = FenceOutcome("refused", op, str(exc))
    except Exception as exc:
        outcome = FenceOutcome("unknown", op, type(exc).__name__, round(time.monotonic() - started, 3))
    return _settle_fence_outcome(ctx, outcome), None


def _drop_fence_binding(ctx: Any) -> None:
    """The queue owns the fence; a token whose state is unproven is never retained."""
    ctx._task_acceptance_fence_token = None
    ctx._task_acceptance_fence_generation = None


def _begin_task_acceptance_fence(ctx: Any, task_id: str) -> tuple[FenceOutcome, Any]:
    """Optional seam implemented by the supervisor under its queue lock."""
    admission_lock = getattr(ctx, "owner_message_admission_lock", None)
    admission_agent = getattr(ctx, "owner_message_admission_agent", None)
    if admission_lock is not None and admission_agent is not None:
        with admission_lock:
            ctx._task_acceptance_owner_generation = int(getattr(admission_agent, "_owner_message_generation", 0) or 0)
    existing = getattr(ctx, "_task_acceptance_fence_token", None)
    inspect = getattr(ctx, "inspect_acceptance_fence", None)
    if existing is not None and not callable(inspect):
        return FenceOutcome(op="begin"), existing
    if existing is not None:
        outcome, refreshed = _fence_request(ctx, "inspect", inspect, token=str(existing))
        if outcome and isinstance(refreshed, dict):
            ctx._task_acceptance_fence_generation = int(refreshed.get("owner_message_generation") or 0)
        if outcome or outcome.status == "unknown":
            return outcome, existing  # no answer is a gap: the binding and its known generation stay
        _drop_fence_binding(ctx)  # refused: the row is gone — rebind through the idempotent begin
    callback = getattr(ctx, "begin_acceptance_fence", None)
    if not callable(callback):
        return FenceOutcome(op="begin"), None  # one-minor/direct-context compatibility
    outcome, response = _fence_request(
        ctx, "begin", callback, root_task_id=_root_task_id(ctx, task_id), task_id=str(task_id))
    answer = response if isinstance(response, dict) else {"token": response}
    if outcome and answer.get("token") in (None, False, ""):
        outcome = _settle_fence_outcome(ctx, FenceOutcome("refused", "begin", "no_token"))
    if not outcome:
        return outcome, None
    # Never None: ``end`` omits ``expected_generation`` for None, and a seal without the queue's
    # compare-and-seal is the blind seal (#406). A fresh fence starts at 0; a wrong 0 refuses.
    ctx._task_acceptance_fence_generation = int(answer.get("owner_message_generation") or 0)
    ctx._task_acceptance_fence_token = answer["token"]
    return outcome, answer["token"]


def _end_task_acceptance_fence(ctx: Any, *, outcome: str, admission_locked: bool = False) -> FenceOutcome:
    token = getattr(ctx, "_task_acceptance_fence_token", None)
    if token is None and str(outcome) in {"revision", "author_stop"}:
        token = getattr(ctx, "_task_acceptance_sealed_fence_token", None)
    callback = getattr(ctx, "end_acceptance_fence", None)
    admission_lock = getattr(ctx, "owner_message_admission_lock", None)
    admission_agent = getattr(ctx, "owner_message_admission_agent", None)
    acquired = False
    try:
        if admission_lock is not None and admission_agent is not None and not admission_locked:
            admission_lock.acquire()
            acquired = True
        expected_owner_generation = getattr(ctx, "_task_acceptance_owner_generation", None)
        from ouroboros.loop_messages import owner_source_sha256
        from ouroboros.loop_transport import _owner_signal_pending

        def owner_mail_pending() -> bool:
            return _owner_signal_pending(
                getattr(ctx, "_acceptance_observation_incoming", None), getattr(ctx, "drive_root", None),
                str(getattr(ctx, "task_id", "") or ""), getattr(ctx, "_loop_mailbox_seen_ids", None),
                getattr(ctx, "task_attempt", None) or 1, owner_authority_only=True,
            )

        acknowledged_source = getattr(ctx, "_acceptance_ack_source_sha256", "")
        generation_mismatch = bool(
            (acknowledged_source and (acknowledged_source != owner_source_sha256(ctx) or owner_mail_pending()))
            or (expected_owner_generation is not None and admission_agent is not None
                and int(getattr(admission_agent, "_owner_message_generation", 0) or 0) != int(expected_owner_generation))
        )
        effective_outcome = "revision" if generation_mismatch else str(outcome)
        if token is None or not callable(callback):
            ctx._task_acceptance_fence_generation_mismatch = generation_mismatch
            return FenceOutcome(op="end")
        expected_queue_generation = getattr(ctx, "_task_acceptance_fence_generation", None)
        if expected_queue_generation is None and token == getattr(ctx, "_task_acceptance_sealed_fence_token", None):
            expected_queue_generation = getattr(ctx, "_task_acceptance_sealed_fence_generation", None) or 0
        result, response = _fence_request(
            ctx, "end", callback, token=token, outcome=effective_outcome,
            **({} if expected_queue_generation is None else {"expected_generation": int(expected_queue_generation)}),
        )
        status = str(response.get("status") or "") if isinstance(response, dict) else ""
        generation_mismatch = generation_mismatch or bool(isinstance(response, dict) and response.get("generation_mismatch"))
        if result and status == "released" and effective_outcome != "revision" and not generation_mismatch:
            # Only ``sealed`` is a seal. An unexplained release may echo a lost ``released +
            # generation_mismatch`` answer: owner mail is durably written before the generation
            # moves, so the local mailbox decides here, never a blind seal.
            generation_mismatch = owner_mail_pending()
    finally:
        if acquired:
            admission_lock.release()
    _drop_fence_binding(ctx)  # also after a refusal or a gap: the next begin re-adopts or reopens
    ctx._task_acceptance_fence_generation_mismatch = generation_mismatch  # local owner facts stand whatever the queue answered
    if result:
        sealed = status == "sealed" or (not status and effective_outcome != "revision")
        ctx._task_acceptance_sealed_fence_token = token if sealed else None
        ctx._task_acceptance_sealed_fence_generation = expected_queue_generation if sealed else None
    return result


def _release_for_owner_followup(ctx: Any, llm_trace: Dict[str, Any], *, admission_locked: bool = False) -> FenceOutcome:
    """Reopen a sealed root for Main to consume new owner input; the selected completion is void."""
    llm_trace.pop("task_completion", None)
    released = _loop()._end_task_acceptance_fence(ctx, outcome="revision", admission_locked=admission_locked)
    ctx._task_acceptance_fence_generation_mismatch = False
    return released


def _owner_generation_changed(ctx: Any) -> bool:
    """Whether the owner wrote since the retained answer's source: direct and queue-owned
    generations, read without closing the seal."""
    from ouroboros.loop_messages import owner_source_sha256, queue_inspection_unknown

    known_source = (getattr(ctx, "_acceptance_ack_source_sha256", "")
                    or getattr(getattr(ctx, "_delivery_candidate", None), "owner_source_sha256", ""))
    if known_source and known_source != owner_source_sha256(ctx):
        return True
    expected_owner = getattr(ctx, "_task_acceptance_owner_generation", None)
    admission_agent = getattr(ctx, "owner_message_admission_agent", None)
    if (
        expected_owner is not None
        and admission_agent is not None
        and int(getattr(admission_agent, "_owner_message_generation", 0) or 0)
        != int(expected_owner)
    ):
        return True
    expected_queue = getattr(ctx, "_task_acceptance_fence_generation", None)
    token = getattr(ctx, "_task_acceptance_fence_token", None)
    inspect = getattr(ctx, "inspect_acceptance_fence", None)
    if expected_queue is None or token is None or not callable(inspect):
        return False
    try:
        state = inspect(token=str(token))
        return bool(
            isinstance(state, dict)
            and int(state.get("owner_message_generation") or 0) != int(expected_queue)
        )
    except Exception as exc:
        # Unknown queue state is not evidence that a new message arrived; the
        # inspection failure is disclosed by the shared stamp.
        queue_inspection_unknown(ctx, exc)
        return False
