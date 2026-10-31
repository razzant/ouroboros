"""One rolling working checkpoint per task attempt (#1563, consolidated #1543).

Every same-ID continuation already serializes the loop's cognition through ONE
serializer (``owner_wait.continuation_state``), but only when the task parks.
This module writes that same state DURING ordinary work, so a worker crash, an
automatic retry or an application stop costs at most the work since the last
boundary instead of the whole line of thought:

- ``ready`` — incorporated owner/peer mail and context changes, before the
  model call (skipped when nothing changed since the last save);
- ``pre_effect`` — the accepted assistant message and its tool batch, before
  the first tool: a write failure HOLDS the batch visibly (typed, interruptible
  by the task's own Stop/Panic/deadline); no unrecorded effect starts;
- ``post_batch`` — the completed batch and its results;
- ``candidate`` — a no-tool answer published as the delivery candidate, before
  its review or delivery;
- ``resumed`` — an exact Resume's restored continuation and its budget tail,
  written before the grant is consumed (``save_resumed``): the pause retired
  this file, and consumption ends the pause's own recovery ownership.

The file (``task_results/artifacts/<task>/source_handles/working/working_checkpoint-a<attempt>.json``,
never a deliverable: every artifact/evidence scanner skips ``source_handles``)
has exactly one writer, its attempt's loop, and is replaced atomically (temp +
``os.replace``: a process crash leaves the old or the new state, never a torn
one; no fsync, so no power-loss promise). It is deleted only once the
attempt's result, owed final or exact pause is durable, or once the attempt
that continues it loaded the frozen copy (never at the freeze itself, whose
locator is not durable yet); storage stays one state per live attempt.
Only a real recovery FREEZES its exact bytes into the existing immutable
continuation-source store, so a retry or a restart restores a verified
source, never the mutable file. A live park (owner wait, exact pause, sleep)
keeps its own newer source: recovery never prefers this file over it.

Recovery restores cognition only. A tool call of the interrupted batch with no
recorded result is closed as UNKNOWN and never re-executed; content mail
drained but not yet in a saved state (a ``ready`` save, a returned result's
final save, an exact pause's published source) was never acknowledged, so it
is delivered again; an answer the host already owes or delivered (the terminal
delivery registry) is not answered twice — that task is not recovered here.
"""

from __future__ import annotations

import hashlib
import contextlib
import contextvars
import json
import logging
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

FILE_PREFIX = "working_checkpoint-a"
# What a ``_working_recovery`` locator names: a rolling working state, or a parked
# question/review wait's own continuation source (each read by its own contract).
SOURCE_WORKING = "working"
SOURCE_OWNER_WAIT = "owner_wait"
BOUNDARIES = ("ready", "pre_effect", "post_batch", "candidate")
RESUMED = "resumed"
_MODEL_CALL = contextvars.ContextVar("working_model_call", default=None)
HOLD_UNWRITABLE = "working_checkpoint_unwritable"
_TRANSIENT_USAGE = ("execution_status", "reason_code", "_best_effort_extracted", "budget_pause_hold",
                    "_llm_round_started", "working_checkpoint")


def _working_dir(root: Any, task_id: str, *, create: bool = False) -> pathlib.Path:
    """Under ``source_handles``: task-private continuation material, never a deliverable
    (every artifact, evidence and patch-capture scanner already skips that subtree)."""
    from ouroboros.artifacts import task_artifact_dir_path

    directory = task_artifact_dir_path(pathlib.Path(root), str(task_id), create=create) / "source_handles" / "working"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def checkpoint_path(root: Any, task_id: str, attempt: int, *, create: bool = False) -> pathlib.Path:
    return _working_dir(root, task_id, create=create) / f"{FILE_PREFIX}{int(attempt)}.json"


def _root(ctx: Any) -> pathlib.Path:
    return pathlib.Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root)


def _signature(limit_ctx: Any) -> tuple:
    messages = limit_ctx.messages or []
    last = messages[-1] if messages else None
    tail = hashlib.sha256(repr(last).encode("utf-8", "replace")).hexdigest() if last is not None else ""
    ctx = getattr(getattr(limit_ctx, "tools", None), "_ctx", None)
    candidate = getattr(ctx, "_delivery_candidate", None)
    return (int(getattr(ctx, "task_attempt", 1) or 1), int(limit_ctx.round_idx), id(messages), len(messages),
            tail, len(limit_ctx.owner_msg_seen or ()), getattr(candidate, "revision", None))


def save_round(limit_ctx: Any, boundary: str) -> bool:
    """Replace this attempt's working checkpoint with the loop's state NOW.

    Returns False when an unchanged ``ready`` state was skipped. Raises on a
    write failure; the caller decides whether that holds (``pre_effect``) or is
    retried at the next boundary.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown working checkpoint boundary: {boundary!r}")
    ctx = getattr(getattr(limit_ctx, "tools", None), "_ctx", None)
    if ctx is None or not str(getattr(ctx, "task_id", "") or "") or not (
            getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None)) or not _root(ctx).is_dir():
        return False  # no durable task root: nothing here could be recovered from it
    if boundary == "ready" and getattr(ctx, "_working_checkpoint_signature", None) == _signature(limit_ctx):
        return False
    _write(ctx, limit_ctx, boundary)
    return True


def save_resumed(tools: Any, state: Dict[str, Any], messages: list, trace: dict, usage: dict, seen: set) -> None:
    """An exact Resume's restored continuation, before ``resume_paused_loop`` consumes the grant.

    It never skips: it returns once the replace landed and raises otherwise, so
    the caller holds the unconsumed grant. It acknowledges no mail. It runs before
    the route rebinds, so it keeps the source's model role, not the new worker's.
    """
    from types import SimpleNamespace

    _write(tools._ctx, SimpleNamespace(
        tools=tools, messages=messages, llm_trace=trace, accumulated_usage=usage, owner_msg_seen=seen,
        round_idx=int(state["round_idx"]), tool_schemas=state["tool_schemas"],
        budget_tail=str((state.get("resume_point") or {}).get("budget_tail") or "tool"),
        context_model_role=str(state.get("context_model_role") or "")), RESUMED)


def _write(ctx: Any, limit_ctx: Any, boundary: str, *, model_call: Any = None) -> None:
    from ouroboros.budget_pause import pending_tool_call_ids
    from ouroboros.owner_wait import continuation_state
    from ouroboros.utils import utc_now_iso, write_bytes_atomic

    signature = _signature(limit_ctx)
    started = time.perf_counter()
    usage = limit_ctx.accumulated_usage
    seq = int(getattr(ctx, "_working_checkpoint_seq", 0) or 0) + 1
    state = continuation_state(
        ctx, limit_ctx.messages, limit_ctx.llm_trace if isinstance(limit_ctx.llm_trace, dict) else {},
        {key: value for key, value in usage.items() if key not in _TRANSIENT_USAGE},
        int(limit_ctx.round_idx), list(limit_ctx.tool_schemas or []), set(limit_ctx.owner_msg_seen or ()))
    if model_call is not None:  # a provisional alternative can checkpoint before route adoption
        from ouroboros.route_choice import after_fallback

        state["messages"] = model_call.messages
        state["route"].update(active_model=model_call.active_model, active_model_override=model_call.active_model,
                              active_use_local=model_call.active_use_local, active_use_local_override=model_call.active_use_local,
                              active_context_mode=model_call.active_context_mode, active_role_override=model_call.model_role,
                              route_choice=after_fallback(ctx, not str(model_call.model_role or "").startswith("fallback:")))
        state["context_model_role"] = model_call.model_role or state["context_model_role"]
    if boundary == RESUMED:  # written before the rebind: the plan is still the new worker's
        state["context_model_role"] = limit_ctx.context_model_role
    state["working"] = {"boundary": boundary, "seq": seq, "saved_at": utc_now_iso(),
                        "budget_tail": getattr(limit_ctx, "budget_tail", "tool"),
                        "pending_tool_call_ids": pending_tool_call_ids(limit_ctx.messages)}
    degraded: List[str] = []

    def opaque(value: Any) -> Dict[str, str]:
        # A live-only handle (never cognition) cannot cross a process; it is named,
        # never invented, and recovery discloses that the state carried it.
        degraded.append(type(value).__name__)
        return {"__unserializable__": type(value).__name__}

    data = json.dumps(state, ensure_ascii=False, default=opaque).encode("utf-8")
    if degraded:
        state["working"]["opaque_values"] = sorted(set(degraded))
        data = json.dumps(state, ensure_ascii=False, default=opaque).encode("utf-8")
    encoded = time.perf_counter()
    write_bytes_atomic(checkpoint_path(_root(ctx), ctx.task_id, int(getattr(ctx, "task_attempt", None) or 1),
                                       create=True), data)
    written = time.perf_counter()
    ctx._working_checkpoint_seq, ctx._working_checkpoint_signature = seq, signature
    stats = usage.setdefault("working_checkpoint", {"saves": 0, "max_bytes": 0, "encode_ms": 0.0, "write_ms": 0.0})
    stats.update(saves=stats["saves"] + 1, last_bytes=len(data), max_bytes=max(stats["max_bytes"], len(data)),
                 last_boundary=boundary, encode_ms=round(stats["encode_ms"] + (encoded - started) * 1000, 3),
                 write_ms=round(stats["write_ms"] + (written - encoded) * 1000, 3))


def save_or_log(limit_ctx: Any, boundary: str) -> bool:
    """A non-gating boundary: a failed write is disclosed and retried at the next one."""
    try:
        return save_round(limit_ctx, boundary)
    except Exception:
        log.warning("Working checkpoint (%s) unwritten for %s", boundary,
                    getattr(getattr(getattr(limit_ctx, "tools", None), "_ctx", None), "task_id", ""), exc_info=True)
        return False


def save_before_effects(limit_ctx: Any) -> None:
    """The pre-effect boundary: no tool of this batch starts until its state is saved."""
    from ouroboros.budget_pause import _hold_until

    ctx = limit_ctx.tools._ctx
    _hold_until(ctx, limit_ctx.accumulated_usage, HOLD_UNWRITABLE,
                lambda: save_round(limit_ctx, "pre_effect"), rail="working_checkpoint")


@contextlib.contextmanager
def model_call_scope(limit_ctx: Any):
    """Bind the loop's existing checkpoint writer for write-ahead call allowances."""
    token = _MODEL_CALL.set(limit_ctx)
    try:
        yield
    finally:
        _MODEL_CALL.reset(token)


def save_model_call(*, model_call: Any = None) -> None:
    """Usage-only transition: bypass ready's cognition signature, hold on write failure.

    The accepted-response checkpoint later replaces this ready state. Until then
    recovery sees an unconfirmed dispatch, even if the live receiver got an answer.
    Direct calls without a recoverable loop have no working checkpoint.
    """
    from ouroboros.budget_pause import _hold_until

    limit = _MODEL_CALL.get()
    if limit is not None:
        _hold_until(limit.tools._ctx, limit.accumulated_usage, HOLD_UNWRITABLE,
                    lambda: _write(limit.tools._ctx, limit, "ready", model_call=model_call), rail="working_checkpoint")


def defer_content_ack(ctx: Any, ack: Any) -> None:
    """Keep a drained content entry's ACK until a saved state holds it."""
    pending = getattr(ctx, "_pending_content_acks", None)
    if not isinstance(pending, list):
        pending = []
        ctx._pending_content_acks = pending
    pending.append(ack)


def flush_content_acks(ctx: Any) -> None:
    """Acknowledge content only after its incorporated state was saved."""
    pending = list(getattr(ctx, "_pending_content_acks", None) or [])
    ctx._pending_content_acks = []
    for ack in pending:
        try:
            ack()
        except Exception:
            log.warning("Owner mail ACK unwritten for %s; it will be delivered again", getattr(ctx, "task_id", ""),
                        exc_info=True)


def save_ready(limit_ctx: Any) -> None:
    """The ready boundary, then the content ACKs it makes safe."""
    ctx = limit_ctx.tools._ctx
    saved = save_or_log(limit_ctx, "ready")
    if saved or getattr(ctx, "_working_checkpoint_signature", None) == _signature(limit_ctx):
        flush_content_acks(ctx)


def save_terminal(limit_ctx: Any) -> None:
    """The loop's ``finally`` for a RETURNED result: content a finalization drain took
    after the last save is saved, then ACKed, before the caller's terminal write
    captures unread mail. A result the loop's own handlers return counts (their
    exception is no longer handled here). The exception that ``finally`` handles
    (a failure exit) saves and ACKs nothing, as would a caller's own handled one;
    a failed save leaves the mail unread."""
    ctx = getattr(getattr(limit_ctx, "tools", None), "_ctx", None)
    if ctx is None:
        return
    from ouroboros.owner_relay import flush_held_state

    flush_held_state(ctx)  # owner sources a busy store refused during the turn: carried on either exit
    if not getattr(ctx, "_pending_content_acks", None) or sys.exc_info()[0] is not None:
        return
    try:
        save_ready(limit_ctx)
    except Exception:
        log.warning("Final working state of %s unsaved; its drained mail stays unread",
                    getattr(ctx, "task_id", ""), exc_info=True)


def discard(root: Any, task_id: str) -> None:
    """Remove a task's working states once its result, pause or recovery source is durable."""
    try:
        for path in _working_dir(root, task_id).glob(f"{FILE_PREFIX}*.json"):
            path.unlink(missing_ok=True)
    except Exception:
        log.debug("Working checkpoint of %s not removed", task_id, exc_info=True)


# --- recovery: supervisor freezes, the next start restores ---------------------------

def read_working(root: Any, task_id: str, attempt: int) -> tuple:
    """``(bytes, state)`` of one attempt's working checkpoint; ``(b"", {})`` when it has none."""
    try:
        data = checkpoint_path(root, task_id, attempt).read_bytes()
    except FileNotFoundError:
        return b"", {}
    state = json.loads(data.decode("utf-8"))
    if (not isinstance(state, dict) or state.get("task_id") != str(task_id)
            or int(state.get("task_attempt") or 0) != int(attempt) or not isinstance(state.get("working"), dict)):
        raise ValueError("working checkpoint identity mismatch")
    return data, state


def live_park(root: Any, task_id: str, attempt: Optional[int] = None, *, verify_source: bool = False) -> str:
    """A park naming a source at least as new as the working attempt, or ``""``.

    With ``attempt``, a park recorded by an OLDER attempt (already continued by
    this one) does not dominate this attempt's working state. A wait a stop
    ``retained`` into an exact pause is that pause's own source: it parks only
    while that pause is live. Its Resume continues the SAME attempt, so after the
    consumption the attempt's newer working state is the source, not the old wait.
    """
    from ouroboros.budget_pause import LIVE_PAUSE_STATES
    from ouroboros.task_results import load_task_result

    def current(park: Dict[str, Any]) -> bool:
        return attempt is None or int(park.get("task_attempt") or 0) >= int(attempt)

    row = load_task_result(pathlib.Path(root), str(task_id), strict=True) or {}
    wait = row.get("owner_wait") if isinstance(row.get("owner_wait"), dict) else {}
    pause = row.get("budget_pause") if isinstance(row.get("budget_pause"), dict) else {}
    live_pause = pause.get("state") in LIVE_PAUSE_STATES
    retained = (wait.get("state") == "retained" and live_pause and bool(wait.get("wait_id"))
                and pause.get("retained_owner_wait_id") == wait.get("wait_id"))
    kind = ("owner_wait" if (wait.get("state") == "waiting" or retained) and wait.get("source_ref") and current(wait)
            else "budget_pause" if live_pause and pause.get("source_ref") and current(pause) else "")
    if kind and verify_source:
        from ouroboros.artifacts import read_actor_source_bytes

        park, key = row[kind], "wait_id" if kind == "owner_wait" else "pause_id"
        state = json.loads(read_actor_source_bytes(pathlib.Path(root), task_id, park["source_ref"]))
        if (state.get("task_id") != task_id or state.get("task_attempt") != park.get("task_attempt")
                or not park.get(key) or state.get(key) != park[key]):
            raise ValueError("park source identity mismatch")
    return kind


def final_answer_registered(root: Any, task_id: str) -> bool:
    """An answer the host owes or recorded as sent must not be generated again.

    A legacy delivered id proves no exact answer bytes, but still prevents a
    second generation. An unreadable registry also cannot authorize a replay.
    """
    from supervisor.terminal_delivery import terminal_answer_receipts

    receipts = terminal_answer_receipts(root, task_id)
    return (receipts.get("state") in {"delivered", "owed", "exhausted"}
            or bool(receipts.get("unverified_delivery_ids")) or receipts.get("registry") == "unreadable")


REASON_FINALIZATION_INCOMPLETE = "finalization_incomplete"


def registered_answer(root: Any, task_id: str) -> Dict[str, Any]:
    """The answer the host already holds for this task, read from the outbox owner; ``{}`` when none.

    ``basis`` names the custody: ``owed`` (a pending outbox row the replay still
    sends — its exact text travels), ``exhausted`` (the host gave up sending it; its
    text and any preserved copy's ``preserved_path`` travel), ``delivered`` (a retained
    receipt whose bytes are re-read; a legacy id proves no bytes, so ``text`` may be
    empty) or ``unreadable`` (the registry could not be read: an answer may exist, which
    is not permission to regenerate). Never the task result's prose.
    """
    from supervisor.terminal_delivery import terminal_answer_receipts

    receipts = terminal_answer_receipts(root, task_id)
    if receipts.get("registry") == "unreadable":
        return {"basis": "unreadable", "text": "", "delivery_id": "", "terminal_origin": ""}
    for basis in ("owed", "exhausted"):  # a row the replay still sends outranks one it gave up
        rows = [row for row in (receipts.get(basis) or []) if isinstance(row, dict)]
        if rows:
            row = max(rows, key=lambda item: str(item.get("registered_at") or ""))
            return {"basis": basis, "text": str(row.get("text") or ""), "delivery_id": str(row.get("delivery_id") or ""),
                    "terminal_origin": str(row.get("terminal_origin") or ""),
                    **({"preserved_path": str(row["preserved_path"])} if row.get("preserved_path") else {})}
    delivered = [row for row in (receipts.get("delivered") or []) if isinstance(row, dict)]
    if delivered:
        receipt = max(delivered, key=lambda item: str(item.get("sent_at") or ""))
        text = origin = ""
        try:
            from ouroboros.artifacts import read_actor_source_bytes

            record = json.loads(read_actor_source_bytes(root, task_id, receipt.get("source_ref")).decode("utf-8"))
            if isinstance(record, dict):
                text, origin = str(record.get("text") or ""), str(record.get("terminal_origin") or "")
        except Exception:
            log.debug("Retained terminal answer of %s could not be re-read", task_id, exc_info=True)
        return {"basis": "delivered", "text": text, "delivery_id": str(receipt.get("delivery_id") or ""),
                "terminal_origin": origin}
    unverified = [str(did) for did in (receipts.get("unverified_delivery_ids") or [])]
    if unverified:
        return {"basis": "delivered", "text": "", "delivery_id": unverified[-1], "terminal_origin": ""}
    return {}


def adopt_registered_answer(root: Any, task: Dict[str, Any], task_id: str, *, attempt: int, cause: str,
                            registry_root: Any = None) -> Optional[Dict[str, Any]]:
    """Finalize a dead attempt whose answer the host already holds, instead of running its prompt again.

    Timeout reaping and dead-worker recovery consult this before admitting a fresh
    model attempt, cancellation custody before settling a direct boot fence. With
    the answer out before the result store (owner 1A(1)) a death inside the store is
    an ordinary window; the authored answer must not be generated a second time.
    A registered answer — owed, exhausted,
    delivered, or merely unreadable evidence — terminalizes the attempt in the
    lifecycle-fault vocabulary: lifecycle ``failed``, execution ``infra_failed``,
    reason ``finalization_incomplete``, the outbox's own text as the result and a
    host notice naming what stayed incomplete. An owed row keeps replaying through
    the outbox; an exhausted one is named as given up. Returns the stored terminal row, or None when no answer is
    registered (the ordinary retry proceeds). Storage errors propagate; the
    monotonic writer may instead return an already-settled row. Neither permits a new model run.
    """
    answer = registered_answer(registry_root if registry_root is not None else root, task_id)
    if not answer:
        return None
    from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
    from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_SALVAGE
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    # Only causes whose phrase claims no retry name themselves; the rest stay typed.
    ended = _CAUSES[cause] if cause in {"worker_crash", "app_stop"} else f"its attempt was stopped ({cause or 'unknown cause'})"
    copy = f"; its copy is kept at {answer['preserved_path']}" if answer.get("preserved_path") else ""
    held = {"owed": "was authored and is owed to the chat (the outbox delivers it)",
            "exhausted": f"was authored, but sending it to the chat was given up{copy}",
            "delivered": "was authored and delivered to the chat",
            "unreadable": "may already exist (the delivery registry could not be read)"}[answer["basis"]]
    notice = (f"Finalization did not complete: the final answer {held}, then {ended} before the "
              "result, review evidence, artifacts and cost accounting were stored. The prompt was "
              "not run again; those facts are incomplete.")
    fields: Dict[str, Any] = {
        "reason_code": REASON_FINALIZATION_INCOMPLETE, "terminal_host_notice": notice,
        "finalization_recovery": {"cause": str(cause or ""), "answer": answer["basis"],
                                  "delivery_id": answer["delivery_id"], "attempt": int(attempt or 0)},
        "outcome_axes": terminal_outcome_axes(
            lifecycle=STATUS_FAILED, execution=EXECUTION_INFRA_FAILED,
            reason_code=REASON_FINALIZATION_INCOMPLETE, review_trigger="supervisor_terminal"),
        "result": answer["text"] or notice,
    }
    origin = answer.get("terminal_origin") or ""
    if origin == TERMINAL_ORIGIN_HOST_SALVAGE:
        # The owed text is the short receipt; the full bytes the host preserved before the
        # send are the result an ordinary store would have written, with their path.
        from ouroboros.observability import preserved_salvage_path

        path = preserved_salvage_path(pathlib.Path(registry_root if registry_root is not None else root), task_id)
        try:
            if path:
                fields.update(terminal_salvage_path=path, result=pathlib.Path(path).read_text(encoding="utf-8"))
        except OSError:
            log.warning("Preserved host salvage of %s could not be read; the receipt stays the result", task_id)
    if answer["text"] and origin:  # the producer's stamp travels; an unstamped legacy row claims none
        fields["terminal_origin"] = origin
    log.warning("Task %s attempt %s: finalizing from its registered answer (%s) instead of a retry (%s)",
                task_id, attempt, answer["basis"], cause)
    return write_task_result(pathlib.Path(root), task_id, STATUS_FAILED, **fields)


def prepare_recovery(root: Any, task_id: str, *, from_attempt: int, cause: str) -> Dict[str, Any]:
    """Freeze the dead attempt's exact working bytes; ``{}`` when there is nothing to continue.

    Raises when a checkpoint exists but cannot be read or frozen: the caller
    keeps its existing (original-prompt or terminal) path rather than guess.
    """
    from ouroboros.artifacts import store_actor_source_bytes

    data, state = read_working(root, task_id, from_attempt)
    if not state or final_answer_registered(root, task_id) or live_park(root, task_id, from_attempt):
        return {}
    working = state["working"]
    ref = store_actor_source_bytes(pathlib.Path(root), str(task_id), category="context_checkpoints",
                                   source_id=f"working-a{int(from_attempt)}-s{int(working.get('seq') or 0)}",
                                   data=data, extension="json")
    return {"source_kind": SOURCE_WORKING, "source_ref": ref, "source_task_id": str(task_id),
            "from_attempt": int(from_attempt), "boundary": str(working.get("boundary") or ""),
            "seq": int(working.get("seq") or 0), "saved_at": str(working.get("saved_at") or ""),
            # These cumulative facts belong to the original start, including
            # a new-ID retry. They do not date this worker's idle/progress clock.
            **({"started_at": state["started_at"],
                "model_wait_quota_clock": (state.get("model_wait") or {}).get("quota_clock", {}),
                "budget_paused_sec": (state.get("model_wait") or {}).get("budget_paused_sec", 0.0)}
               if state.get("started_at") else {}),
            "cause": str(cause)}


def attach_recovery(root: Any, task: Dict[str, Any], *, source_task_id: str, from_attempt: int,
                    cause: str, prior_task: Optional[Dict[str, Any]] = None) -> bool:
    """An EXISTING automatic retry continues the dead attempt's saved work (owner 2026-10-08,
    quiz 1b1a2d93): eligibility, count and cadence stay the caller's. On success the
    frozen source rides ``task["_working_recovery"]``. Without prior saved work the
    retry keeps its original-prompt start; a failed relay keeps its old locator
    held instead of silently abandoning the last saved cognition.

    The rolling file STAYS: this locator lives in memory until the caller's own
    queue/snapshot publication, so a crash or refused publication here must find
    the same file and freeze the same bytes again. Only the consumer that loaded
    the frozen source removes it (``consume_recovery``); terminal cleanup removes
    whatever is left (``discard``)."""
    prior = dict(prior_task or {})
    # A failed relay retains the old locator as a non-admissible carrier. Losing
    # the last source must not silently turn a continuation into a fresh prompt.
    if "_working_recovery" not in prior:
        task.pop("_working_recovery", None)
    try:
        handoff = prepare_recovery(root, source_task_id, from_attempt=from_attempt, cause=cause)
        if not handoff and prior.get("_working_recovery"):
            handoff = _carry_recovery(root, prior, str(task.get("id") or source_task_id),
                                      source_task_id, from_attempt, cause)
    except Exception:
        log.warning("Working checkpoint of %s attempt %s could not be frozen for its retry",
                    source_task_id, from_attempt, exc_info=True)
        return False
    if not handoff:
        return False
    task["_working_recovery"] = handoff
    return True


def _carry_recovery(root: Any, prior: Dict[str, Any], target_id: str, task_id: str,
                    attempt: int, cause: str) -> Dict[str, Any]:
    """A dead receiver with no newer checkpoint carries its exact source one step.

    Only the existing retry/restore producer calls this, after capturing its
    prior attempted task. The source's identity and bytes are never rewritten.
    """
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    if (prior.get("id") != task_id or prior.get("_attempt") != attempt
            or not recovery_source_for_task(root, prior)
            or checkpoint_path(root, task_id, attempt).exists()
            or final_answer_registered(root, task_id) or live_park(root, task_id, attempt)):
        return {}
    stored = load_task_result(pathlib.Path(root), task_id, strict=True) or {}
    mark = stored.get("admitted_dispatch_attempt")
    if (stored.get("task_id") != task_id or type(mark) is not int or mark != attempt
            or stored.get("status") in _TRULY_TERMINAL_STATUSES):
        return {}
    return {**prior["_working_recovery"], "target_task_id": target_id, "target_attempt": attempt + 1,
            "continued_from_task_id": task_id, "continued_from_attempt": attempt, "cause": cause}


def complete_pause_from_working(root: Any, task_id: str, row: Dict[str, Any], attempt: int) -> Dict[str, Any]:
    """A ``pausing`` seed whose worker died before its exact source existed (#1543).

    The same attempt's working state becomes that pause's source: its exact
    cognition plus the pause's own program counter (the interrupted batch is
    UNKNOWN, never re-run) and a fresh custody observation under the pause's
    own stop policy. Returns the completed durable row, or ``{}`` when there
    is no usable working state (the caller keeps its fenced terminal path).
    """
    import time as _time

    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.budget_pause import (
        RESUME_POLICY, STATE_PAUSING, STOP_POLICY_ALL, STOP_POLICY_OBSERVE, STOP_POLICY_TASK_OWNED,
        episode_fields, observe_task_runs, set_budget_pause, task_owned_runs_open,
    )
    from ouroboros.model_wait import budget_paused_seconds

    _data, state = read_working(root, task_id, attempt)
    if not state or row.get("state") != STATE_PAUSING or row.get("source_ref"):
        return {}
    working = state["working"]
    pending = list(working.get("pending_tool_call_ids") or [])
    point = {"round_idx": int(state.get("round_idx") or 0),
             "phase": "partial_tool_batch_unknown" if pending else "boundary",
             "unanswered_tool_call_ids": pending, "unanswered_policy": "not_re_executed_execution_unknown",
             "abandoned_model_attempts": [], "budget_tail": str(working.get("budget_tail") or "tool"),
             "recovered_from_working": {"boundary": working.get("boundary"), "seq": working.get("seq")}}
    reason = str(row.get("reason") or "budget")
    policy = (STOP_POLICY_TASK_OWNED if reason == "owner" else STOP_POLICY_ALL if reason == "budget"
              else STOP_POLICY_OBSERVE)
    external = observe_task_runs(root, task_id, reason="budget_pause_uncovered_cost", stop_policy=policy)
    local = {"dispatch_fence": "closed", "quiescent": True, "basis": "worker_process_dead"}
    started = state.get("started_at") or row.get("started_at")
    model_state = state.get("model_wait") or {}
    if not started:
        # Legacy sources with no original clock cannot deduct old exclusions
        # from a fresh start. Keep the source and Resume's row on the same clock.
        model_state = {**model_state, "quota_clock": {}, "budget_paused_sec": 0.0}
    source = {**state, "started_at": started, "model_wait": model_state,
              "pause_id": row.get("pause_id"), "reason": reason, "rail": row.get("rail"),
              "scope": row.get("scope"), "resume_point": point, "external_runs": external, "local_producers": local}
    ref = store_actor_source_bytes(pathlib.Path(root), str(task_id), category="context_checkpoints",
                                   source_id="budget-pause-" + str(row.get("pause_id") or ""),
                                   data=json.dumps(source, ensure_ascii=False).encode("utf-8"), extension="json")
    completed = {**row, "source_ref": ref, "resume_point": point, "external_runs": external,
                 "local_producers": local, "paused_at": _time.time(), "execution_drive_root": str(root),
                 "cost_ceiling": state.get("cost_ceiling"), "physical_calls": None, "exact_continuation": True,
                 "replay_safe": False, "auto_resume": False, "resume_policy": RESUME_POLICY,
                 "started_at": started, "paused_duration_sec": budget_paused_seconds(model_state),
                 "model_wait_quota_clock": model_state.get("quota_clock", {}),
                 # The seed keeps no list: stops its dead worker issued may have settled unseen (partial).
                 **({"settlement": "external_writers_running" if task_owned_runs_open(external) else "settled",
                     **episode_fields(row, external)} if reason == "owner" else {})}
    set_budget_pause(pathlib.Path(root), str(task_id), completed, expected_pause_id=str(row.get("pause_id") or ""),
                     expected_state=STATE_PAUSING)
    return completed


def load_recovery(ctx: Any, handoff: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The loop side: the frozen source named by this start's ``_working_recovery``."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.runtime_limits import explicit_execution_timeout_sec

    handoff = handoff or getattr(ctx, "working_recovery", None)
    if not isinstance(handoff, dict) or not handoff.get("source_ref"):
        return {}
    source_task = str(handoff.get("source_task_id") or ctx.task_id)
    kind = str(handoff.get("source_kind") or SOURCE_WORKING)
    state = json.loads(read_actor_source_bytes(_root(ctx), source_task, handoff["source_ref"]))
    if (kind not in {SOURCE_WORKING, SOURCE_OWNER_WAIT} or state.get("task_id") != source_task
            or int(state.get("task_attempt") or 0) != int(handoff.get("from_attempt") or -1)):
        raise ValueError("working recovery identity mismatch")
    # Each source keeps its OWN reader contract: a rolling state carries its
    # ``working`` block; a parked question/review wait is the owner-wait
    # continuation source, bound to its ``wait_id`` (``owner_wait.load_owner_wait``).
    if kind == SOURCE_WORKING and not isinstance(state.get("working"), dict):
        raise ValueError("working recovery identity mismatch")
    if kind == SOURCE_OWNER_WAIT and (not handoff.get("wait_id") or state.get("wait_id") != handoff.get("wait_id")):
        raise ValueError("owner wait recovery identity mismatch")
    if not handoff.get("started_at"):
        # Legacy working sources lack the original clock origin. Keep their
        # existing fresh-start fallback, but never deduct OLD exclusions from
        # that new clock (including when the loop next saves/parks this state).
        state["model_wait"] = {**(state.get("model_wait") or {}),
                               "quota_clock": {}, "budget_paused_sec": 0.0}
    elif explicit_execution_timeout_sec(getattr(ctx, "task_metadata", {}) or {}) is not None:
        from ouroboros.model_wait import adopt_handoff_clock

        # The loop's next Pause must keep the same queue exclusion as agent start.
        state["model_wait"] = adopt_handoff_clock(state.get("model_wait") or {}, handoff)
    return {**state, "_working_handoff": dict(handoff)}


def recovery_source_for_task(root: Any, task: Dict[str, Any]) -> Dict[str, Any]:
    """Verify the frozen source and its exact successor, without consuming either.

    The queue owns dispatch permission; this reader binds one proposed attempt
    to its immediate predecessor and exact saved cognition (which can precede
    that receiver when it died before saving). A new-id idle retry must carry
    the retry owner's explicit predecessor relation as well.
    """
    from types import SimpleNamespace

    handoff = task.get("_working_recovery")
    if not isinstance(handoff, dict) or not isinstance(handoff.get("source_ref"), dict):
        return {}
    task_id, source_id = str(task.get("id") or ""), str(handoff.get("source_task_id") or "")
    attempt, previous = task.get("_attempt"), handoff.get("from_attempt")
    if (not task_id or not source_id or type(attempt) is not int or type(previous) is not int
            or previous < 1):
        return {}
    predecessor = source_id
    if any(key in handoff for key in ("target_task_id", "target_attempt", "continued_from_task_id", "continued_from_attempt")):
        preceding = handoff.get("continued_from_attempt")
        predecessor = handoff.get("continued_from_task_id")
        if (handoff.get("target_task_id") != task_id or type(handoff.get("target_attempt")) is not int
                or handoff["target_attempt"] != attempt or type(preceding) is not int
                or preceding < previous or attempt != preceding + 1 or not isinstance(predecessor, str)
                or not predecessor):
            return {}
    elif attempt != previous + 1:
        return {}
    if predecessor != task_id and not (task.get("original_task_id") == predecessor
                                       and task.get("timeout_retry_from") == predecessor):
        return {}
    return load_recovery(SimpleNamespace(task_id=task_id, drive_root=root,
                                         budget_drive_root=task.get("budget_drive_root") or root), handoff)


def consume_recovery(ctx: Any, handoff: Dict[str, Any]) -> None:
    """The consumer captured the frozen source: retire what it replaces (best effort).

    A rolling source removes the dead attempt's mutable file; an owner-wait source
    marks that wait ``resumed`` (an answer is still accepted as a message), so the
    old park never dominates this attempt's own later working state.
    """
    from ouroboros.owner_wait import set_owner_wait
    from ouroboros.task_results import load_task_result

    source_task = str(handoff.get("source_task_id") or ctx.task_id)
    try:
        if str(handoff.get("source_kind") or SOURCE_WORKING) == SOURCE_WORKING:
            checkpoint_path(_root(ctx), source_task, int(handoff.get("from_attempt") or 0)).unlink(missing_ok=True)
            return
        wait = (load_task_result(_root(ctx), source_task, strict=True) or {}).get("owner_wait") or {}
        if wait.get("wait_id") == handoff.get("wait_id") and wait.get("state") == "waiting":
            set_owner_wait(_root(ctx), source_task, {**wait, "state": "resumed",
                                                      "resume_reason": "recovered_after_stop"},
                           expected_wait_id=str(handoff["wait_id"]))
    except Exception:
        log.warning("Recovered source of %s was not retired; terminal cleanup retries", source_task, exc_info=True)


def close_unanswered_calls(messages: List[Dict[str, Any]], kind: str) -> List[str]:
    """Close the last batch's calls that have no recorded result as UNKNOWN, never re-run."""
    from ouroboros.budget_pause import pending_tool_call_ids

    pending = pending_tool_call_ids(messages)
    for call_id in pending:
        # Transcript validity for the provider AND the honest fact: unknown, not
        # "did not run" and not "ran". The host never re-executes it.
        messages.append({"role": "tool", "tool_call_id": call_id, "content": (
            f"[HOST NOTICE] No result for this tool call was recorded before the {kind}. "
            "Its execution state is UNKNOWN: it may have run and produced effects, or not run at all. "
            "It was NOT re-executed. Verify from authoritative state (files, git, services, custody) "
            "before repeating it.")})
    return pending


_CAUSES = {"worker_crash": "its worker process ended unexpectedly",
           "idle_timeout": "its stalled attempt was stopped and retried",
           "restart": "the application restarted", "app_stop": "the application stopped",
           "update_aborted": "the managed update was aborted before restart"}


def resume_from_working(tools: Any, state: Dict[str, Any], messages: list, trace: dict,
                        usage: dict, seen: set) -> tuple:
    """Restore a frozen working checkpoint; ``(model, effort, local, mode, round, plan, tool_tail)``.

    Same ID: the whole continuation (usage, delivery candidate, review and
    owner-source state) continues. A NEW execution ID (an idle retry) restores the
    transcript, route and cumulative execution clock: its own accounting starts
    here (spend stays on the ledger's tree), and it borrows no review or delivery authority
    from the attempt it continues. Nothing is re-executed; runs are disclosed.
    """
    from ouroboros.budget_pause import observe_task_runs
    from ouroboros.owner_wait import rebind_restored_route, restore_continuation_state

    ctx = tools._ctx
    handoff = state.get("_working_handoff") or {}
    same_id = str(handoff.get("source_task_id") or ctx.task_id) == str(ctx.task_id)
    restore_continuation_state(tools, state if same_id else {
        **state, "usage": {}, "delivery": {key: value for key, value in state.get("delivery", {}).items()
            if key in {"_delivery_candidate_revision", "_delivery_evidence_revision", "_delivery_evidence_fingerprint",
                       "_delivery_effective_criteria", "_delivery_material_tool_indices"}},
        "acceptance": {}}, messages, trace, usage, seen)
    # Retry lineage carries the remaining allowance, never a fresh grant or old money. A new ID keeps
    # each failure row's own owner (its sources stay the predecessor's) and the captions its request had.
    from ouroboros.loop_llm_call import TIMEOUT_REPEAT_KEY
    if record := (state.get("usage") or {}).get(TIMEOUT_REPEAT_KEY):
        source = str(handoff.get("source_task_id") or ctx.task_id)
        usage[TIMEOUT_REPEAT_KEY] = {**record, "recovered": True, **({} if same_id else {
            row: {"task_id": source, **record[row]} for row in ("first", "repeat") if isinstance(record.get(row), dict)})}
        if not same_id:
            usage.setdefault("_vision_caption_memo", dict(state["usage"].get("_vision_caption_memo") or {}))
    from ouroboros.loop_output_exhaustion import KEY as OUTPUT_RECOVERY_KEY
    if record := (state.get("usage") or {}).get(OUTPUT_RECOVERY_KEY):
        usage[OUTPUT_RECOVERY_KEY] = dict(record)
    # Saved prose is evidence, never permission to skip today's finalization:
    # a restored attempt holds no seal of the interrupted one.
    ctx._task_acceptance_sealed_fence_token = None
    ctx._task_acceptance_sealed_fence_generation = None
    if ctx._delivery_candidate is not None:
        # It holds no verdict and says so explicitly: an empty binding let a superseded PASS stay current.
        from ouroboros.loop_delivery import _unaccepted_delivery_binding

        ctx._delivery_candidate.acceptance_binding = _unaccepted_delivery_binding(
            tools, ctx._delivery_candidate.content_sha256)
    if not same_id:
        for run in trace.get("review_runs") or []:
            if isinstance(run, dict) and run.get("authority") == "host_root":
                run.update(superseded_by_revision=True, superseded_reason="working_recovery_new_execution")
        # ``acceptance_decision`` is written no more; a pre-fold checkpoint's own does not carry over.
        for key in ("acceptance_decision", "review_decision", "root_phase_checkpoint", "task_completion"):
            trace.pop(key, None)
    from ouroboros.task_pacing import restore_cost_ceiling

    # Every restore re-reads this start's authority, including a formerly
    # unreadable root. A saved number alone is never monetary authority.
    ctx._cost_ceiling = restore_cost_ceiling(ctx, state.get("cost_ceiling"))
    usage["_task_attempt"] = getattr(ctx, "task_attempt", None)
    ctx.working_recovery = None
    plan, mode = rebind_restored_route(tools, state, messages)
    pending = close_unanswered_calls(messages, "interruption")
    try:
        from ouroboros import delegate_custody as custody

        runs = observe_task_runs(custody.custody_root(ctx), str(handoff.get("source_task_id") or ctx.task_id),
                                 reason="working_recovery_disclosure", request_stop=False)
    except Exception:
        runs = {"custody_read": "failed", "runs": []}
    run_lines = "".join(f"\n- run {run.get('run_id') or run.get('invocation_id')}: {run.get('state')}"
                        for run in runs.get("runs") or [] if isinstance(run, dict)) or (
        "\n- none" if runs.get("custody_read") == "ok" else "\n- custody unreadable: treat every run as possibly live")
    boundary = str(handoff.get("boundary") or "")
    opaque = (state.get("working") or {}).get("opaque_values") or []
    question = (" It was waiting for the owner's answer to its question; that question stays open and an "
                "answer that arrives is delivered to this task as a message — never assume one you have not "
                "received." if handoff.get("source_kind") == SOURCE_OWNER_WAIT else "")
    messages.append({"role": "user", "content": (
        f"[SYSTEM NOTICE]\nThis task continued from its working checkpoint ({boundary}, saved "
        f"{handoff.get('saved_at') or 'earlier'}) because {_CAUSES.get(str(handoff.get('cause') or ''), 'it was interrupted')}"
        + ("" if same_id else f"; this is a new execution of task {handoff.get('source_task_id')} — its earlier "
           "review state does not carry over")
        + ". Work after that point may be lost. Cumulative spend, rounds and elapsed time were NOT reset; prior "
        "tool results remain recorded; do not repeat completed effects. The previous browser process and "
        "task-local services ended; their recorded results are evidence, not proof they still run. Re-read "
        "files before building on them."
        + (f" {len(pending)} tool call(s) of the interrupted batch have no recorded result and were NOT "
           "re-executed (host rows above); their execution state is unknown." if pending else "")
        + (f" Live-only objects ({', '.join(opaque)}) could not be saved with it." if opaque else "") + question
        + f"\nDelegated runs of this task:{run_lines}\nAn unknown or merely requested stop is not proof of "
        "termination: never start a second writer over such a run; the host never continues one by itself.")})
    # A batch rejoins its tool tail; ready state repeats its logical round.
    # A saved candidate rejoins finalization before another author call.
    # A Resume handover rejoins its pause's tail, under the last-fit permission
    # that Resume already spent (its receipt rides a same-ID usage).
    resumed = boundary == RESUMED
    if resumed:
        ctx._budget_resume_last_fit_relaxed = bool((usage.get("budget_pause_resume") or {}).get("last_fit_relaxed"))
    return (ctx.active_model, ctx.active_effort, ctx.active_use_local, mode, int(state["round_idx"]), plan,
            boundary in {"pre_effect", "post_batch"}
            or resumed and (state.get("working") or {}).get("budget_tail") == "tool")
