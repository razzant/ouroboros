"""Owner-facing delivery of worker chat events: text, media, and typing.

One owner for what reaches the owner's chat and for the bound chat id every
other handler routes onto. Final answers are deduplicated twice - an in-memory
deque for the fast path and the durable delivery registry that survives a
restart - because the worker deliberately sends the terminal answer over both
the live queue and the buffered return.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import pathlib
import threading
import time
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict
from ouroboros.utils import utc_now_iso
from ouroboros.observability import emit_finalization_timing, timed_phase
from ouroboros.store import acquisition_budget
from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
from ouroboros.owner_stop_receipt import EVENT_FIELD as RETAINED_WORK_FIELD

log = logging.getLogger(__name__)


# A progress frame's ``task_id`` is a ROUTING address — it says which live card the
# line lands on, NOT who wrote the line. The supervisor narrates a task's terminal
# path (grace requested, grace withdrawn) onto that task's own card, so those frames
# carry the task's id while the task itself did nothing. Host-authored frames set
# this key; ``_handle_send_message`` refuses to count them as the task's work.
# Without it the supervisor's own voice answers its own question — the grace toast
# stamped last_progress_at, the next 0.5s tick read the task as resumed, and the
# episode it had just opened was withdrawn before the worker could ever drain it.
from supervisor.log_addressing import bound_project_chat_id as _bound_project_chat_id
from supervisor.message_bus import notification_chat_route
from ouroboros.subagent_messages import delegated_activity_meta, executor_observation_meta, subagent_message_meta


HOST_NARRATION = "host_narration"


def _on_consumer(handler):
    """A consumer handler writes under the acquisition slice; the shared writers keep callers' waits."""
    @functools.wraps(handler)
    def run(evt: Dict[str, Any], ctx: Any) -> Any:
        with acquisition_budget(ACQUISITION_SLICE_SEC):
            return handler(evt, ctx)
    return run


def _handle_typing_start(evt: Dict[str, Any], ctx: Any) -> None:
    try:
        task_id = str(evt.get("task_id") or "")
        phase = str(evt.get("phase") or "thinking")
        client_msg_id = ""
        kind = ""
        task_row: Dict[str, Any] = {}
        if task_id:
            # One read of the RUNNING row: it carries both the lineage the
            # project binding is resolved by and the root check the kind stamp
            # needs.
            try:
                running = getattr(ctx, "RUNNING", None)
                meta = running.get(task_id) if isinstance(running, dict) else None
                row = meta.get("task") if isinstance(meta, dict) else None
                if isinstance(row, dict):
                    task_row = row
            except Exception:
                log.debug("RUNNING row read failed for %s", task_id, exc_info=True)
        # Membership, not truthiness: absence skips the indicator, an explicit
        # id — the hidden partition included — is a destination. The binding is
        # resolved AT EMISSION and OUTRANKS the origin chat, because a task
        # bound to a project after admission keeps the chat it was born in on
        # its row (same order as _handle_task_heartbeat and task_done).
        chat_id = notification_chat_route(
            _bound_project_chat_id(
                ctx, task_id, task_row.get("parent_task_id"), task_row.get("root_task_id")
            )
            or None,
            evt.get("chat_id"),
        )
        if task_id:
            try:
                from supervisor.active_activity import get_direct_activity_registry
                # A registry hit identifies a direct/ephemeral turn; queued
                # managed tasks also emit typing_start but are not tracked here,
                # so their frames go out without a kind stamp. The stamp is
                # kept for wire compatibility only: no in-repo client reads a
                # typing frame's kind, and it carries no authority over any
                # client-side set.
                entry = get_direct_activity_registry().get(task_id)
                if entry:
                    client_msg_id = entry.client_message_id
                    kind = entry.kind
            except Exception:
                pass
        if not kind and task_row:
            # A RUNNING queue ROOT is stamped "managed_task" like the /api/state
            # activity census names it (that census lists queue roots). The stamp
            # is kept for wire compatibility only: no in-repo client reads it, and
            # the web header never admits a typing frame into its live-activity
            # set, so an empty kind is not an exemption from anything. The census still
            # does not enumerate children by design — their own task cards do.
            try:
                from ouroboros.task_results import resolve_task_lineage

                lineage = resolve_task_lineage(
                    task_id,
                    metadata=task_row.get("metadata"),
                    root_task_id=task_row.get("root_task_id"),
                    parent_task_id=task_row.get("parent_task_id"),
                    delegation_role=task_row.get("delegation_role"),
                    original_task_id=task_row.get("original_task_id"),
                    timeout_retry_from=task_row.get("timeout_retry_from"),
                )
                if lineage["is_root_task"]:
                    kind = "managed_task"
            except Exception:
                log.debug("managed typing kind resolution failed for %s", task_id, exc_info=True)
        if chat_id is not None:
            ctx.bridge.send_chat_action(
                chat_id,
                "typing",
                activity_id=task_id,
                client_message_id=client_msg_id,
                phase=phase,
                kind=kind,
            )
    except Exception:
        log.debug("Failed to send typing action to chat", exc_info=True)
        pass


_DELIVERED_MESSAGE_IDS: deque = deque(maxlen=256)  # (id/token, actual destination facts); text stays in the outbox
_DELIVERY_LOCK = threading.Lock()
_ACTIVE_DELIVERIES: set = set()  # physical handler ownership, released only after send/receipt returns
_DEFERRED_DELIVERIES: set = set()  # wake hints only; payload authority remains terminal_deliveries.json


def _delivery_key(root: Any, event: dict) -> tuple:
    return str(pathlib.Path(root).resolve()) if root else "", str(event.get("delivery_id") or "")


def _remembered_delivery(event: dict) -> Any:
    did, token = str(event.get("delivery_id") or ""), str(event.get("terminal_projection_token") or "")
    identity = f"{did}:{token}" if token else did
    return next((facts for key, facts in tuple(_DELIVERED_MESSAGE_IDS) if key == identity), None)


def _defer_delivery(ctx: Any, event: dict) -> None:
    from ouroboros.dialogue_retry import request_drain
    from supervisor.terminal_delivery import register_pending_delivery

    if event.get("delivery_id") and getattr(ctx, "DRIVE_ROOT", None):
        with _DELIVERY_LOCK:
            _DEFERRED_DELIVERIES.add(_delivery_key(ctx.DRIVE_ROOT, event))
        register_pending_delivery(ctx.DRIVE_ROOT, {**event, "delivery_deferred": True})
        request_drain()


def replay_deferred_deliveries(root: Any, rows: list, *, event_queue: Any = None) -> list:
    """The existing duty continues failed handlers here, never through control events.

    Bus membership protects queued originals. The handler's physical claim also
    protects direct continuations from duplicate handlers and concurrent sweeps.
    Volatile hints cover a busy registry write; durable deferred/sent facts survive restart.
    Receipt repairs (a recorded or remembered send, or a superseded or given-up row's returned
    one) never send and run first, so what one retires (a Stop receipt's superseded host notice)
    is not resent by the same sweep.
    """
    from supervisor import message_bus, queue, workers, terminal_delivery as td
    from supervisor.event_observation import delivery_outstanding
    from ouroboros.utils import append_jsonl

    event_queue = event_queue if event_queue is not None else workers.get_event_q(create=False)
    ctx = SimpleNamespace(DRIVE_ROOT=pathlib.Path(root), RUNNING=queue.RUNNING, EVENT_Q=event_queue,
                          send_with_budget=message_bus.send_with_budget, append_jsonl=append_jsonl)
    continued = []
    with _DELIVERY_LOCK:
        present = {_delivery_key(root, row) for row in rows}
        _DEFERRED_DELIVERIES.difference_update(
            key for key in _DEFERRED_DELIVERIES - present if key[0] == str(pathlib.Path(root).resolve()))
        hinted = set(_DEFERRED_DELIVERIES)
    for row in sorted(rows, key=lambda row: not (row.get("emitted") or _remembered_delivery(row) is not None)):
        key = _delivery_key(root, row)
        if not row.get("delivery_deferred") and key not in hinted:
            continue
        with _DELIVERY_LOCK:
            if key in _ACTIVE_DELIVERIES:
                continue  # direct continuations hold physical ownership without bus membership
        if row.get("transport_failed") and row.get("terminal_projection_token"):
            row["delivery_deferred"] = False  # canonical Main retains its own transport retry clock
            with _DELIVERY_LOCK:
                _DEFERRED_DELIVERIES.discard(key)
            continue
        row["delivery_deferred"] = True  # also excludes the hint from ordinary aged queue replay
        if delivery_outstanding(event_queue, key[1]) is not False:
            continue
        # Positive send evidence only needs receipt repair, even at the cap. Busy
        # pre-send storage has spent nothing; every recorded entry (including an
        # interrupted/unknown outcome) keeps the same backoff and transport bound.
        if not row.get("terminal_projection_token") and not row.get("emitted") and _remembered_delivery(row) is None:
            attempts = td._dispatch_attempts(row)
            if attempts is None or ((attempts or row.get("transport_failed")) and not td._replay_due(row)):
                continue
            if attempts >= td._PENDING_MAX_REPLAYS:
                td._retire_exhausted(root, [row])
                continue
        if _handle_send_message({**row, "type": "send_message", "_delivery_replay": True}, ctx):
            continued.append(key[1])
    return continued


def _register_delivered(ctx: Any, delivery_id: str, emitted: "Dict[str, Any] | None" = None, *, task_id: str = '') -> bool:
    """Atomically mark one id delivered and clear its pending-outbox row.

    ``emitted`` is what THIS handler just sent (exact text, routed chat, task):
    a terminal answer's receipt of its bytes, never inferred elsewhere.
    """
    if not getattr(ctx, "DRIVE_ROOT", None):
        return True  # rootless transports retain their historical send-only contract
    try:
        from supervisor.terminal_delivery import register_delivery, record_cancel_receipt_delivery

        _relay_room_result(ctx.DRIVE_ROOT, delivery_id, emitted)
        if emitted and emitted.get("system_type") == "cancel_receipt":
            record_cancel_receipt_delivery(ctx.DRIVE_ROOT, str(emitted.get("task_id") or ""),
                                           delivery_id, emitted["chat_id"], strict=True)
        register_delivery(ctx.DRIVE_ROOT, delivery_id, emitted=emitted, strict=True)
        _handoff_delivered_review(ctx, task_id or str((emitted or {}).get('task_id') or ''), delivery_id)
        with _DELIVERY_LOCK:
            _DEFERRED_DELIVERIES.discard(_delivery_key(ctx.DRIVE_ROOT, {"delivery_id": delivery_id}))
        return True
    except Exception:
        log.debug("durable delivery registration failed", exc_info=True)
        return False


def _relay_room_result(root: Any, delivery_id: str, emitted: "Dict[str, Any] | None") -> None:
    """A delivered terminal answer is context for the room's other live direct turns.

    Run before acknowledging delivery: a refused mailbox write keeps the sent
    outbox row owed, so replay retries this handoff without sending chat twice.
    """
    from ouroboros.owner_mailbox import mailbox_drain_ended, mailbox_holds, write_task_message
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.terminal_delivery import TERMINAL_MESSAGE_ID
    from supervisor.workers import direct_chat_turn

    match = TERMINAL_MESSAGE_ID.fullmatch(delivery_id)
    if not emitted or emitted.get("is_progress") or match is None:
        return
    source = str(emitted.get("task_id") or "")
    if match.group("task") != source or not emitted.get("text"):
        return
    chat_id, root = int(emitted["chat_id"]), pathlib.Path(root)
    for activity in get_direct_activity_registry().actors():
        recipient = activity.activity_id
        if activity.chat_id != chat_id or recipient == source:
            continue
        # The actor closes message admission under this same lock. Recheck its
        # identity and room there so a completed/reused actor receives nothing.
        lock = getattr(activity.actor, "_owner_message_admission_lock", None)
        with lock if lock is not None else contextlib.nullcontext():
            turn = direct_chat_turn(recipient)
            if turn is None or turn["chat_id"] != chat_id or mailbox_drain_ended(root, recipient):
                continue
            msg_id = f"room-result:{delivery_id}:{recipient}"
            if mailbox_holds(root, recipient, msg_id):
                continue
            # Every terminal family rides (final, cancel, cascade): a turn waiting on a stopped task
            # must learn that too, so the label names the ending, not a "final result".
            if not write_task_message(root, f"Task {source} in this room has ended; its terminal message:\n{emitted['text']}",
                                      recipient, source_task_id=source, provenance="system", msg_id=msg_id):
                raise RuntimeError(f"Result mailbox delivery deferred for {recipient}")


def _handoff_delivered_review(ctx: Any, task_id: str, delivery_id: str) -> None:
    """After a delivery is registered (a real send, or a copy suppressed as already delivered):
    settle-on-ack — the Done card held behind this very debt follows the answer now (owner
    Fork 2). ``prepare=False``: readiness is task_done's to mint, so an ack that lands before
    the row is terminal settles nothing and task_done settles it."""
    if not task_id:
        return
    try:
        from ouroboros.terminal_projection import settle_terminal_projection

        settle_terminal_projection(ctx.DRIVE_ROOT, task_id, event_queue=getattr(ctx, "EVENT_Q", None), prepare=False)
    except Exception:
        log.debug("terminal projection settlement after delivery deferred", exc_info=True)


def _handle_send_message(evt: Dict[str, Any], ctx: Any) -> bool:
    """Own this physical send until it returns; contention leaves the exact outbox debt."""
    key = _delivery_key(getattr(ctx, "DRIVE_ROOT", None), evt)
    claimed = bool(key[1]) and not (evt.get("terminal_custody_notice") or evt.get(RETAINED_WORK_FIELD))
    with _DELIVERY_LOCK:
        if claimed and key in _ACTIVE_DELIVERIES:
            return False
        if claimed:
            _ACTIVE_DELIVERIES.add(key)
    try:
        with acquisition_budget(0 if evt.get("is_progress") and not key[1] else ACQUISITION_SLICE_SEC):
            return _send_message(evt, ctx) is not False
    finally:
        if claimed:
            with _DELIVERY_LOCK:
                _ACTIVE_DELIVERIES.discard(key)


def bundled_rows(evt: Dict[str, Any]) -> tuple:
    """The answer without the facts it carries, and those facts as their own typed card rows.

    Issue #1006: open delegated execution is a typed row OF the task's card, so it is split off
    before the host notice and never carries the answer's phase: a custody row concludes or
    relabels nothing. Wrap up's receipt (#1564) is a second such row under the Stop's own identity,
    so whichever terminal message carries it, it lands once. The outbox owes these same rows for a
    notice a Stop receipt replaced before any handler split it (``supervisor.terminal_supersession``).
    """
    answer = dict(evt)
    custody = str(answer.pop("terminal_custody_notice", "") or "")
    retained = answer.pop(RETAINED_WORK_FIELD, None)
    retained = retained if isinstance(retained, dict) and retained.get("text") else {}
    meta = evt.get("progress_meta") if isinstance(evt.get("progress_meta"), dict) else {}
    # Derived from the ANSWER's id, so a re-split never grows a suffix. An
    # answer that reaches this seam unregistered gets the same canonical
    # identity its owed registration would mint; with no task to key it,
    # the custody text stays on the joined host notice rather than
    # minting a task-independent id the delivered registry would then
    # suppress for every later task.
    base_id = str(evt.get("delivery_id") or "")
    if not base_id and str(evt.get("task_id") or ""):
        from supervisor.terminal_delivery import delivery_id_for

        base_id = delivery_id_for(str(evt.get("task_id")), str(evt.get("text") or ""))

    def _row(text: str, row_id: str, **marks: Any) -> Dict[str, Any]:
        row = {**answer, "text": text, "log_text": text, "role": "system", "system_type": "custody_notice"}
        if row_id:
            row.update(delivery_id=row_id, progress_meta={
                **{key: value for key, value in meta.items() if key not in ("task_phase", "task_terminal_status")},
                "card_row": "timeline", "card_row_id": row_id, **marks})
        for key in ("terminal_host_notice", "terminal_origin", "_finalization_timing",
                    "terminal_projection_token"):
            row.pop(key, None)
        return row

    if not base_id:
        # Nothing to key the custody row by: the fact still reaches the
        # chat as its OWN typed row instead of riding a host-notice
        # field no consumer reads, and it mints no task-independent
        # delivery id the delivered registry would suppress for every
        # later task.
        answer.pop("terminal_host_notice", None)
    rows = [_row(custody, base_id and base_id + ":custody_notice")] if custody else []
    if retained:
        # Typed as a terminal receipt: a transport that leaves card rows to
        # its progress toggle still sends this one (Telegram).
        rows.append(_row(str(retained["text"]), str(retained.get("id") or "")
                         or (base_id and base_id + ":retained_work"), terminal_receipt="retained_work"))
    return answer, rows


def _send_message(evt: Dict[str, Any], ctx: Any) -> Any:
    try:
        if evt.get("terminal_custody_notice") or evt.get(RETAINED_WORK_FIELD):
            from supervisor.terminal_delivery import register_pending_delivery

            answer, rows = bundled_rows(evt)
            # Owe the rows before the answer send clears its bundled outbox row.
            for row in rows:
                if row.get("delivery_id"):
                    register_pending_delivery(ctx.DRIVE_ROOT, row)
            _handle_send_message(answer, ctx)
            for row in rows:
                _handle_send_message(row, ctx)
            return
        delivery_id = str(evt.get("delivery_id") or "")
        pending: Dict[str, Any] = {}
        if delivery_id and getattr(ctx, "DRIVE_ROOT", None):
            from supervisor.terminal_delivery import pending_deliveries

            pending = next((row for row in pending_deliveries(ctx.DRIVE_ROOT)
                            if row["delivery_id"] == delivery_id), {})
            evt.setdefault("ts", pending.get("ts") or utc_now_iso())
            emitted = evt.get("emitted") or pending.get("emitted")
            if (isinstance(emitted, dict)
                    and emitted.get("terminal_projection_token", "") == str(evt.get("terminal_projection_token") or "")):
                if not _register_delivered(ctx, delivery_id, emitted=emitted):
                    _defer_delivery(ctx, {**evt, "emitted": emitted})
                    return False
                return True
            if evt.get("_delivery_replay") and not pending and not evt.get("terminal_projection_token"):
                return  # a withdrawn/exhausted stale queue copy cannot recreate debt
        if delivery_id and not evt.get("is_progress"):
            from supervisor.terminal_delivery import TERMINAL_MESSAGE_ID

            # The task's own answer (or Stop/cascade receipt) is handed off BEFORE this send
            # and before the worker's slow result store: post-answer work stays alive but no
            # longer prepares a reply (DESIGN §9, Telegram typing). A copy suppressed below
            # was handed off by its first copy.
            match = TERMINAL_MESSAGE_ID.fullmatch(delivery_id)
            if match is not None and match.group("task") == str(evt.get("task_id") or ""):
                from supervisor.active_activity import mark_reply_handed_off

                mark_reply_handed_off(match.group("task"), getattr(ctx, "RUNNING", None))
        publication_token = str(evt.get("terminal_projection_token") or "")
        if publication_token:
            from ouroboros.terminal_projection import main_delivery_current
            if not main_delivery_current(ctx.DRIVE_ROOT, str(evt.get("task_id") or ""), publication_token):
                return
        memory_id = f"{delivery_id}:{publication_token}" if publication_token else delivery_id
        remembered = _remembered_delivery(evt)
        if delivery_id and remembered is not None:
            log.debug("send_message suppressed as duplicate (delivery_id=%s)", delivery_id)
            # This copy is suppressed because the FIRST one was sent, so record
            # that durably: it also clears any pending-outbox row, which would
            # otherwise be replayed (and suppressed again) until it gave up.
            emitted = {**evt, **remembered}
            if not _register_delivered(ctx, delivery_id, emitted=emitted, task_id=str(evt.get('task_id') or '')):
                _defer_delivery(ctx, {**evt, "emitted": emitted})
                return False
            return
        if delivery_id:
            try:
                from supervisor.terminal_delivery import already_delivered
                from supervisor.terminal_supersession import superseded_by

                if already_delivered(ctx.DRIVE_ROOT, delivery_id, task_id=str(evt.get("task_id") or ""),
                                     publication_token=publication_token or None):
                    log.debug(
                        "send_message suppressed as durably delivered (delivery_id=%s)",
                        delivery_id,
                    )
                    _register_delivered(ctx, delivery_id, task_id=str(evt.get('task_id') or ''))
                    return
                # A buffered original of a host notice a delivered Stop receipt replaced (no longer owed)
                # keeps its outbox bytes and is not sent; this process's own send of it was recorded above.
                if not pending and not publication_token and superseded_by(ctx.DRIVE_ROOT, delivery_id):
                    log.debug("send_message suppressed as superseded (delivery_id=%s)", delivery_id)
                    return
            except Exception:
                # Fail open toward delivery — never lose an answer to a dedupe read.
                log.debug("durable delivery dedupe read failed", exc_info=True)
        log_text = evt.get("log_text")
        fmt = str(evt.get("format") or "")
        is_progress = bool(evt.get("is_progress"))
        raw_ts = evt.get("ts")
        task_id = str(evt.get("task_id") or "")
        # Real-progress signal (activity model): a progress narration line is genuine work,
        # so stamp the EMITTING task's last_progress_at. (A productively-waiting parent is
        # kept alive separately by _subtree_progressing detecting fresh DESCENDANT progress,
        # not by re-stamping its own last_progress_at from child narration.) HOST_NARRATION
        # frames are addressed to the task's card but authored by the supervisor, so they
        # are narration ABOUT the task, never work BY it.
        progress_meta = evt.get("progress_meta") if isinstance(evt.get("progress_meta"), dict) else None
        if is_progress and evt.get("_is_direct_chat") is True:
            # Stamped by value on the turn's own queue (TurnEventQueue); the live
            # frame is built from progress_meta, so the fact rides along.
            progress_meta = dict(progress_meta or {})
            progress_meta["_is_direct_chat"] = True
        _running = getattr(ctx, "RUNNING", None)
        task_row: Dict[str, Any] = {}
        if task_id and isinstance(_running, dict):
            _m = _running.get(task_id)
            # Mutate in place (see _handle_llm_usage): no write-back, so a cross-thread
            # cancel that popped this task is never resurrected.
            if isinstance(_m, dict):
                if is_progress and not evt.get(HOST_NARRATION):
                    _m["last_progress_at"] = time.time()
                task_row = _m.get("task") if isinstance(_m.get("task"), dict) else {}
                child_meta = subagent_message_meta(task_row, task_id=task_id)
                if child_meta:
                    event_meta = dict(progress_meta or {})
                    progress_meta = dict(child_meta)
                    progress_meta.update(event_meta)
            if is_progress and isinstance(_m, dict):
                # Host-attested RUNNING lineage is the cancel authority.
                progress_meta = dict(progress_meta or {})
                for lineage_key in ("root_task_id", "parent_task_id", "delegation_role"):
                    value = str(task_row.get(lineage_key) or "").strip()
                    if value and not progress_meta.get(lineage_key):
                        progress_meta[lineage_key] = value
                try:
                    from ouroboros.task_results import resolve_task_lineage

                    lineage = resolve_task_lineage(
                        task_id,
                        metadata=task_row.get("metadata"),
                        root_task_id=task_row.get("root_task_id"),
                        parent_task_id=task_row.get("parent_task_id"),
                        delegation_role=task_row.get("delegation_role"),
                        original_task_id=task_row.get("original_task_id"),
                        timeout_retry_from=task_row.get("timeout_retry_from"),
                    )
                    if bool(lineage["is_root_task"]):
                        progress_meta["cancelable"] = True
                except Exception:
                    log.debug("cancelable lineage resolution failed for %s", task_id, exc_info=True)
            if is_progress and not isinstance(_m, dict):
                # The in-process direct-chat turn has no RUNNING row; it is a
                # root by construction and owner-addressable through the same
                # ownership seam the cancel ingress uses (workers.direct_chat_turn),
                # so its live card carries the same host-attested marker.
                from supervisor import workers

                if workers.direct_chat_turn(task_id) is not None:
                    progress_meta = dict(progress_meta or {})
                    progress_meta["cancelable"] = True
        result = {}
        if task_id and not is_progress and not task_row and not subagent_message_meta(progress_meta, task_id=task_id):
            try:
                from ouroboros.task_status import load_effective_task_result

                result = load_effective_task_result(ctx.DRIVE_ROOT, task_id, materialize_artifacts=False)
                child_meta = subagent_message_meta(result, task_id=task_id)
                if child_meta:
                    event_meta = dict(progress_meta or {})
                    progress_meta = dict(child_meta)
                    progress_meta.update(event_meta)
            except Exception:
                log.debug("final lineage recovery failed for %s", task_id, exc_info=True)
        if (evt.get("system_type") == "project_completion_summary"
                and not isinstance((progress_meta or {}).get("completion_notice"), dict)):
            from ouroboros.terminal_projection import completion_event_view

            if not result:
                from ouroboros.task_results import load_task_result

                try:
                    result = load_task_result(ctx.DRIVE_ROOT, task_id, strict=True) or {}
                except Exception:
                    log.debug("legacy completion metadata unavailable for %s; keeping retained text", task_id, exc_info=True)
                    result = {}
            progress_meta = completion_event_view(ctx.DRIVE_ROOT, {**evt, "progress_meta": progress_meta or {}}, result).get("progress_meta")
        if progress_meta and "executor_observation" in progress_meta:
            progress_meta = dict(progress_meta)
            observation = executor_observation_meta(
                progress_meta.pop("executor_observation"), task_id=task_id,
                task_attempt=task_row.get("_attempt"),
            )
            if observation:
                progress_meta["executor_observation"] = observation
        if progress_meta and "delegated_activity" in progress_meta:
            progress_meta = dict(progress_meta)
            activity = delegated_activity_meta(progress_meta.pop("delegated_activity"), task_id=task_id)
            if activity:
                progress_meta["delegated_activity"] = activity
        meta = progress_meta or {}
        bound_chat = _bound_project_chat_id(
            ctx, task_id,
            meta.get("parent_task_id") or evt.get("parent_task_id"),
            meta.get("root_task_id") or evt.get("root_task_id"),
        )
        system_type = str(evt.get("system_type") or "")
        from ouroboros.project_dialogue import ROOM_FIXED_ROW_TYPES

        # Project lifecycle rows and a root's Main notice pin Main; others keep lineage routing.
        chat_id = int(evt["chat_id"]) if system_type in ROOM_FIXED_ROW_TYPES else bound_chat or int(evt["chat_id"])
        if delivery_id:
            progress_meta = dict(progress_meta or {})
            progress_meta["delivery_id"] = delivery_id
        with timed_phase("sender", timing=evt.get("_finalization_timing") or {}):
            send_facts = ctx.send_with_budget(
                chat_id,
                str(evt.get("text") or ""),
                log_text=(str(log_text) if isinstance(log_text, str) else None),
                fmt=fmt,
                is_progress=is_progress,
                task_id=task_id,
                progress_meta=progress_meta,
                ts=(str(raw_ts) if raw_ts else None),
                # S3 (Q4): a typed system receipt keeps its role/type end to end.
                role=str(evt.get("role") or ""),
                system_type=system_type,
                **({"require_write": True} if delivery_id else {}),
                **({"replay_delivery_id": delivery_id}
                   if delivery_id and evt.get("_delivery_replay") and not publication_token else {}),
            )
            if send_facts is False:
                return False  # replay was no longer owed/allowed; no send or receipt
        # Register only after send; a failed first copy must not suppress retry.
        if delivery_id:
            # The send seam owns the final room lookup. Retain its selection,
            # never a new binding read after transport (placement can race both).
            # Legacy send-only adapters return None and keep the passed address.
            if isinstance(send_facts, dict):
                chat_id = int(send_facts["chat_id"])
            emitted = {
                "text": str(evt.get("text") or ""), "chat_id": chat_id, "task_id": task_id, "format": fmt,
                "is_progress": is_progress,
                "sent_at": utc_now_iso(),
                "role": str(evt.get("role") or ""), "system_type": system_type,
                "terminal_origin": str(evt.get("terminal_origin") or ""),
                "terminal_projection_token": publication_token,
                "routing": {"basis": "terminal_sender_bound_project", "intended_chat_id": int(evt["chat_id"]),
                            "routed_chat_id": chat_id} if bound_chat == chat_id or chat_id != int(evt["chat_id"]) else None}
            _DELIVERED_MESSAGE_IDS.append((memory_id, {k: v for k, v in emitted.items() if k != "text"}))
            # Retain the physical-send fact before slower capture/ack. A failed registry
            # write is covered by the existing bounded in-process dedupe until replay.
            _defer_delivery(ctx, {**evt, "emitted": emitted, "transport_failed": False})
            if not _register_delivered(ctx, delivery_id, emitted=emitted):
                return False
        emit_finalization_timing(evt, getattr(ctx, "DRIVE_ROOT", None))
    except Exception as e:
        from supervisor.message_bus import MessageTransportError

        failed_transport = isinstance(e, MessageTransportError)
        _defer_delivery(ctx, {**evt, "transport_failed": failed_transport})
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "send_message_event_error", "error": repr(e),
            },
        )
        return False



def _delivery_chat_id(evt: Dict[str, Any], ctx: Any) -> "int | None":
    """The bound delivery chat for a media/links/quiz frame (upstream unification):
    post-hoc bound media stays in the project panel even when the task retained
    its original chat id; chat 0 is a real hidden session (same contract across
    photo/video/file/links)."""
    bound_chat = _bound_project_chat_id(
        ctx, evt.get("task_id"), evt.get("parent_task_id"), evt.get("root_task_id")
    )
    raw_chat_id = evt.get("chat_id")
    if not bound_chat and (raw_chat_id is None or raw_chat_id == ""):
        return None
    return bound_chat or int(raw_chat_id)


def _log_error(ctx: Any, event_type: str, **fields: Any) -> None:
    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {"ts": utc_now_iso(), "type": event_type, **fields},
    )


@_on_consumer
def _handle_send_photo(evt: Dict[str, Any], ctx: Any) -> None:
    """Send a photo to the owner's chat."""
    import base64 as b64mod
    try:
        # Binding precedence matches text delivery: post-hoc bound media stays
        # in the project panel even when the task retained its original chat id.
        # chat 0 is a real hidden session (same contract as video/file/links).
        chat_id = _delivery_chat_id(evt, ctx)
        image_b64 = str(evt.get("image_base64") or "")
        caption = str(evt.get("caption") or "")
        mime = str(evt.get("mime") or "image/png")
        if chat_id is None or not image_b64:
            return
        photo_bytes = b64mod.b64decode(image_b64)
        ok, err = ctx.bridge.send_photo(
            chat_id, photo_bytes, caption=caption, mime=mime,
            task_id=str(evt.get("task_id") or ""),
        )
        if not ok:
            _log_error(ctx, "send_photo_error", chat_id=chat_id, error=err)
    except Exception as e:
        _log_error(ctx, "send_photo_event_error", error=repr(e))


@_on_consumer
def _handle_send_video(evt: Dict[str, Any], ctx: Any) -> None:
    """Send a video to the owner's chat."""
    import base64 as b64mod
    try:
        chat_id = _delivery_chat_id(evt, ctx)
        video_b64 = str(evt.get("video_base64") or "")
        caption = str(evt.get("caption") or "")
        mime = str(evt.get("mime") or "video/mp4")
        if chat_id is None or not video_b64:
            return
        video_bytes = b64mod.b64decode(video_b64)
        ok, err = ctx.bridge.send_video(
            chat_id, video_bytes, caption=caption, mime=mime,
            task_id=str(evt.get("task_id") or ""),
        )
        if not ok:
            _log_error(ctx, "send_video_error", chat_id=chat_id, error=err)
    except Exception as e:
        _log_error(ctx, "send_video_event_error", error=repr(e))


@_on_consumer
def _handle_send_document(evt: Dict[str, Any], ctx: Any) -> None:
    """Send an arbitrary document/file to the owner's chat."""
    import base64 as b64mod
    try:
        chat_id = _delivery_chat_id(evt, ctx)
        file_b64 = str(evt.get("file_base64") or "")
        file_ref = evt.get("file_ref") if isinstance(evt.get("file_ref"), dict) else None
        if chat_id is None or not (file_b64 or file_ref):
            return
        ok, err = ctx.bridge.send_document(
            chat_id,
            b64mod.b64decode(file_b64),
            filename=str(evt.get("filename") or "file"),
            caption=str(evt.get("caption") or ""),
            mime=str(evt.get("mime") or "application/octet-stream"),
            download_url=str(evt.get("download_url") or ""),
            task_id=str(evt.get("task_id") or ""),
            file_ref=file_ref,
            download_url_compat=str(evt.get("download_url_compat") or ""),
        )
        if not ok:
            _log_error(ctx, "send_document_error", chat_id=chat_id, error=err)
    except Exception as e:
        _log_error(ctx, "send_document_event_error", error=repr(e))


@_on_consumer
def _handle_send_links(evt: Dict[str, Any], ctx: Any) -> None:
    """Send structured HTTP(S) actions to the owner's chat."""
    try:
        chat_id = _delivery_chat_id(evt, ctx)
        actions = evt.get("actions")
        if chat_id is None or not isinstance(actions, list) or not actions:
            return
        ok, err = ctx.bridge.send_links(
            chat_id,
            actions,
            title=str(evt.get("title") or ""),
            task_id=str(evt.get("task_id") or ""),
        )
        if not ok:
            _log_error(ctx, "send_links_error", chat_id=chat_id, error=err)
    except Exception as e:
        _log_error(ctx, "send_links_event_error", error=repr(e))


@_on_consumer
def _handle_send_card(evt: Dict[str, Any], ctx: Any) -> None:
    """Publish a stored card: its chat row, or a ``card_state`` update for a replaced body."""
    from ouroboros.card_store import load_card, record_publication
    from supervisor.card_delivery import send_card, send_card_state

    try:
        chat_id = _delivery_chat_id(evt, ctx)
        record = load_card(ctx.DRIVE_ROOT, evt.get("card_id"))
        if chat_id is None or record is None:
            return
        if evt.get("replace") is True:
            send_card_state(ctx.bridge, record)
            return
        record = record_publication(ctx.DRIVE_ROOT, record["card_id"], chat_id) or record
        ok, err = send_card(ctx.bridge, record, chat_id, task_id=str(evt.get("task_id") or ""),
                            interactive=evt.get("interactive") is True)
        if not ok:
            _log_error(ctx, "send_card_error", chat_id=chat_id, error=err)
    except Exception as e:
        _log_error(ctx, "send_card_event_error", error=repr(e))


@_on_consumer
def _handle_send_quiz(evt: Dict[str, Any], ctx: Any) -> None:
    """Attempt the canonical publication debt with its persisted retry clock."""
    from ouroboros.owner_quiz import claim_quiz_delivery, quiz_states, record_quiz_delivery_failure
    from supervisor.log_addressing import resolve_quiz_publication

    tid, qid = str(evt.get("task_id") or ""), str(evt.get("quiz_id") or "")
    try:
        block = quiz_states(ctx.DRIVE_ROOT, tid).get(qid) or {}
        if (block.get("delivery") or {}).get("kind") != "publication":
            return  # A stale ask must not consume the answer/closure retry clock.
        if not claim_quiz_delivery(ctx.DRIVE_ROOT, tid, qid):
            return
        try:
            chat_id = resolve_quiz_publication(ctx.DRIVE_ROOT, tid, block, evt.get("chat_id"),
                str(evt.get("parent_task_id") or ""), str(evt.get("root_task_id") or ""))
        except (OSError, ValueError) as exc:
            reason = str(exc) if str(exc) in {"chat_unresolved", "publication_address_unknown"} else "project_source_unknown"
            record_quiz_delivery_failure(ctx.DRIVE_ROOT, tid, qid, reason)
            return
        options = evt.get("options")
        if chat_id is None or not isinstance(options, list):
            record_quiz_delivery_failure(ctx.DRIVE_ROOT, tid, qid,
                                         "chat_unresolved" if chat_id is None else "publication_payload_invalid")
            return
        from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
        if is_a2a_chat_id(chat_id):
            record_quiz_delivery_failure(ctx.DRIVE_ROOT, tid, qid, "a2a_chat")
            return
        ok, err = ctx.bridge.send_quiz(
            chat_id,
            quiz_id=str(evt.get("quiz_id") or ""),
            question=str(evt.get("question") or ""),
            options=options,
            stake=str(evt.get("stake") or ""),
            assumption=str(evt.get("assumption") or ""),
            state=str(evt.get("state") or "open"),
            task_id=str(evt.get("task_id") or ""),
            host_facts=str(evt.get("host_facts") or ""),
            **({"wait_for_answer": True} if evt.get("wait_for_answer") is True else {}),
        )
        if not ok:
            record_quiz_delivery_failure(ctx.DRIVE_ROOT, tid, qid, "publication_send_failed")
            _log_error(ctx, "send_quiz_error", chat_id=chat_id, error=err)
    except Exception as exc:
        record_quiz_delivery_failure(ctx.DRIVE_ROOT, tid, qid, "publication_write_failed")
        _log_error(ctx, "send_quiz_event_error", error=repr(exc))


@_on_consumer
def _handle_send_quiz_state(evt: Dict[str, Any], ctx: Any) -> None:
    """An explicit closure discharges only after its reason reaches durable history."""
    from ouroboros.gateway.task_decision import _late_answer_destination
    from ouroboros.owner_quiz import (
        claim_quiz_delivery, mark_discharged, mark_quiz_history_written,
        question_delivery_projection, quiz_states, record_quiz_delivery_failure,
    )
    from supervisor.message_bus import log_chat

    tid, qid = str(evt.get("task_id") or ""), str(evt.get("quiz_id") or "")
    block = quiz_states(ctx.DRIVE_ROOT, tid).get(qid) or {}
    if not block or block.get("state") != "closed":
        return
    block = claim_quiz_delivery(ctx.DRIVE_ROOT, tid, qid, force=evt.get("owner_retry") is True)
    if not block:
        return
    try:
        if (block.get("delivery") or {}).get("publication_pending") and "publication_chat_id" not in block:
            from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
            from ouroboros.owner_quiz import record_publication
            from ouroboros.task_results import load_task_result
            from supervisor.log_addressing import resolve_quiz_publication

            task = load_task_result(ctx.DRIVE_ROOT, tid, strict=True) or {}
            chat_id = resolve_quiz_publication(ctx.DRIVE_ROOT, tid, block, block.get("chat_id"),
                str(task.get("parent_task_id") or ""), str(task.get("root_task_id") or ""))
            if is_a2a_chat_id(chat_id):
                record_quiz_delivery_failure(ctx.DRIVE_ROOT, tid, qid, "a2a_chat")
                return
            block = record_publication(ctx.DRIVE_ROOT, tid, qid, chat_id)
        chat_id, reason = _late_answer_destination(ctx.DRIVE_ROOT, tid, block)
        if chat_id is None:  # only the history row waits on the address; a live author still learns
            record_quiz_delivery_failure(ctx.DRIVE_ROOT, tid, qid, reason)
        elif not (block.get("delivery") or {}).get("history_at"):
            log_chat("system", chat_id, 0, str(block.get("closed_reason") or ""),
                     task_id=tid, record_type="quiz_closed", quiz=block, ts=block.get("closed_at"),
                     source="owner_question_close" if block.get("closed_actor") == "owner" else "model_question_close",
                     client_message_id=f"quiz_closed:{tid}:{qid}", drive_root=ctx.DRIVE_ROOT,
                     require_write=True)
            mark_quiz_history_written(ctx.DRIVE_ROOT, tid, qid)
        if block.get("closed_actor") == "owner" and not (block.get("delivery") or {}).get("mailbox_at"):
            from ouroboros.gateway.task_decision import _mail_holder
            from ouroboros.owner_mailbox import KIND_QUIZ_CLOSED, write_owner_message
            from supervisor.queue import _task_drive_for_task

            task, _refusal = _mail_holder(tid)  # a paused author learns it when it continues
            if task is not None:
                frame = (f"[Owner question closed without answer] quiz {qid} — "
                         f"closed {block['closed_at']}. Reason: {block['closed_reason']}\n"
                         "This closes the question; no option was chosen and no approval was given.")
                if not write_owner_message(_task_drive_for_task(task, tid), frame, tid,
                                           msg_id=f"quiz_closed:{qid}", kind=KIND_QUIZ_CLOSED):
                    record_quiz_delivery_failure(ctx.DRIVE_ROOT, tid, qid, "mailbox_write_failed")
                    return
                mark_quiz_history_written(ctx.DRIVE_ROOT, tid, qid, receipt="mailbox_at")
        if chat_id is None:
            return
        block = mark_discharged(ctx.DRIVE_ROOT, tid, qid) or block
        ctx.bridge.send_quiz_state(quiz_id=qid, task_id=tid, state="closed", chat_id=chat_id,
                                  closed_reason=str(block.get("closed_reason") or ""),
                                  closed_at=block.get("closed_at"), closed_actor=block.get("closed_actor"),
                                  closed_request_id=block.get("closed_request_id"),
                                  closed_by_task_id=block.get("closed_by_task_id"),
                                  question_delivery=question_delivery_projection(block))
    except Exception:
        record_quiz_delivery_failure(ctx.DRIVE_ROOT, tid, qid, "closure_write_failed")
        raise


# Merged into supervisor.events.EVENT_HANDLERS (the `**_CEH` pattern): the
# structured chat-delivery event registry this leaf owns.
EVENT_HANDLERS = {
    "send_photo": _handle_send_photo,
    "send_video": _handle_send_video,
    "send_document": _handle_send_document,
    "send_links": _handle_send_links,
    "send_card": _handle_send_card,
    "send_quiz": _handle_send_quiz,
    "send_quiz_state": _handle_send_quiz_state,
}
