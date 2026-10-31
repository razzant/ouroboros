"""Finalization phase telemetry carried by the live answer delivery.

Task-local clocks follow the last model answer through the existing event queue
and journal. Durable copies omit the sidecar because monotonic origins cannot
survive a reboot. Observability re-exports the public helpers for existing callers.

Latency stamps are telemetry, never gates. Upstream of this module:
``chat.jsonl`` inbound rows carry ``message_accepted_at``, ``task_received``
carries ``activity_emitted_at``, and request/response manifests and ``llm_round``
carry ``first_request_at``/``first_answer_at``. This module adds the finalization
phases, which ride the live final event, the supervisor consumer's dequeue stamp,
and one ``task_finalization_timing`` row in ``events.jsonl`` written after the
sender returns.
"""

from __future__ import annotations

import copy
import functools
import contextvars
import logging
import pathlib
import time
from contextlib import contextmanager

from ouroboros.utils import utc_now_iso


_TASK_TIMING = contextvars.ContextVar("task_finalization_timing", default=None)
_TIMING_PHASES = contextvars.ContextVar("task_timing_phases", default=())


@contextmanager
def task_timing_scope(*, reuse=False):
    """Attempt-local phase fields, carried by usage and the existing final event.

    No history reads, store, worker or policy. Each returned main-model response
    resets the tail; nested spans overlap and must not be summed as a partition.
    Context isolation covers native threads and pooled workers alike. A loop
    shares its enclosing task scope so post-loop child lookups remain counted.
    """
    if reuse and _TASK_TIMING.get() is not None:
        yield
        return
    token = _TASK_TIMING.set({})
    phases_token = _TIMING_PHASES.set(())
    try:
        yield
    finally:
        _TASK_TIMING.reset(token)
        _TIMING_PHASES.reset(phases_token)


def task_timing():
    return _TASK_TIMING.get()


def without_finalization_timing(fields):
    """Copy durable fields without the live, boot-relative timing sidecar."""
    return {key: value for key, value in fields.items() if key != "_finalization_timing"}


def mark_last_answer(usage):
    """Stamp the returned model response, before response persistence/accounting."""
    timing = task_timing()
    if timing is None:
        return
    timing.clear()
    timing.update(last_answer_at=utc_now_iso(), _origin=time.monotonic(), phases={})
    usage["_finalization_timing"] = timing


@contextmanager
def timed_phase(name, *, timing=None, within=""):
    """Accumulate observed calls and monotonic seconds; optionally end at acquire.

    The yielded callback closes a lock-wait span immediately after acquisition.
    Exceptions still propagate unchanged. A later response withdraws old spans.
    """
    timing = task_timing() if timing is None else timing
    if not timing or "_origin" not in timing or (within and within not in _TIMING_PHASES.get()):
        yield lambda: None
        return
    origin, started = timing["_origin"], time.monotonic()
    row = timing["phases"].setdefault(name, {
        "count": 0, "seconds": 0.0, "errors": 0,
        "started_at": utc_now_iso(), "started_sec": max(0.0, started - origin),
    })
    row["count"] += 1
    finished = False

    def finish():
        nonlocal finished
        if not finished and timing.get("_origin") == origin:
            ended = time.monotonic()
            row.update(finished_at=utc_now_iso(), finished_sec=max(0.0, ended - origin))
            row["seconds"] += max(0.0, ended - started)
        finished = True

    token = _TIMING_PHASES.set((*_TIMING_PHASES.get(), name))
    try:
        yield finish
    except BaseException:
        if not finished:
            row["errors"] += 1
        raise
    finally:
        finish()
        _TIMING_PHASES.reset(token)


def stamp_finalization_enqueue(event):
    """Snapshot at the actual queue handoff; keep the buffered copy independent.

    Monotonic anchors travel only on live IPC, never in the durable outbox:
    replay after a reboot cannot compare clocks or reconstruct missing phases.
    """
    timing = event.get("_finalization_timing")
    if not timing or "_origin" not in timing:
        return event
    timing = copy.deepcopy(timing)
    timing.update(enqueued_at=utc_now_iso(), enqueued_sec=max(0.0, time.monotonic() - timing["_origin"]))
    return {**event, "_finalization_timing": timing}


def stamp_finalization_dequeue(event):
    """Stamp the moment the supervisor consumer took a live final off the bus.

    It splits the queue wait (enqueue -> dequeue) from the handler's work before the
    send (dequeue -> sender) on the row. The origin is the worker's monotonic stamp, the
    same basis as ``sender.started_sec`` (the monotonic clock is system-wide on the
    supported hosts); the event's own sidecar is replaced, never mutated in place. A
    malformed sidecar is left as it came: telemetry never costs the delivery.
    """
    timing = event.get("_finalization_timing") if isinstance(event, dict) else None
    if not isinstance(timing, dict) or "_origin" not in timing:
        return
    try:
        dequeued_sec = max(0.0, time.monotonic() - float(timing["_origin"]))
    except (TypeError, ValueError):
        return
    event["_finalization_timing"] = {**timing, "dequeued_at": utc_now_iso(), "dequeued_sec": dequeued_sec}


def emit_finalization_timing(event, drive_root):
    """One best-effort journal row after the deduplicated sender has returned.

    This is a host send-handler receipt, not proof of client display. A missing
    phase was not observed; failed sends/crashes and durable replay add no row.
    ``enqueue_to_sender_sec`` splits into ``enqueue_to_dequeue_sec`` (bus wait) and
    ``dequeue_to_sender_sec`` (the handler before the send); None when unstamped.
    The sender's existing delivery identity, not a new registry, deduplicates it.
    """
    timing = event.get("_finalization_timing")
    if drive_root is None or not timing or "sender" not in timing.get("phases", {}):
        return
    try:
        from ouroboros.utils import append_jsonl

        sender, dequeued = timing["phases"]["sender"], timing.get("dequeued_sec")
        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            **{key: value for key, value in timing.items() if not key.startswith("_")},
            "type": "task_finalization_timing", "ts": sender["finished_at"],
            "task_id": event.get("task_id"), "delivery_id": event.get("delivery_id"),
            "basis": "send_handler_returned", "total_sec": sender["finished_sec"],
            "enqueue_to_sender_sec": (max(0.0, sender["started_sec"] - timing["enqueued_sec"])
                                      if "enqueued_sec" in timing else None),
            "enqueue_to_dequeue_sec": (max(0.0, dequeued - timing["enqueued_sec"])
                                       if dequeued is not None and "enqueued_sec" in timing else None),
            "dequeue_to_sender_sec": max(0.0, sender["started_sec"] - dequeued) if dequeued is not None else None,
        })
    except Exception:
        logging.getLogger(__name__).debug("Finalization timing could not be recorded", exc_info=True)


def emit_post_answer_timing(usage, drive_logs, task, delivery_id):
    """One best-effort row for the packaging that ran AFTER the answer left (owner 1A(1)).

    The live copy's snapshot was frozen at its enqueue, so the sender's row cannot
    hold the later ``result_store`` span. This correlated row (same task and
    delivery id, ``basis: post_answer_store``) carries every span the worker
    observed up to the store's end; readers keep the two rows apart by basis.
    """
    timing = (usage or {}).get("_finalization_timing")
    phases = (timing or {}).get("phases") if isinstance(timing, dict) else None
    if drive_logs is None or not isinstance(phases, dict) or "result_store" not in phases:
        return
    try:
        from ouroboros.utils import append_jsonl

        store = phases["result_store"]
        append_jsonl(pathlib.Path(drive_logs) / "events.jsonl", {
            **{key: value for key, value in timing.items() if not key.startswith("_")},
            "type": "task_finalization_timing", "ts": store.get("finished_at") or utc_now_iso(),
            "task_id": task.get("id"), "task_attempt": task.get("_attempt"),
            "delivery_id": delivery_id, "basis": "post_answer_store",
            "total_sec": store.get("finished_sec"),
        })
    except Exception:
        logging.getLogger(__name__).debug("Post-answer finalization timing could not be recorded", exc_info=True)


_FINALIZATION_CHILDREN = contextvars.ContextVar("finalization_children", default=None)


@contextmanager
def finalization_child_scope():
    """One child projection per root during a synchronous finalization pass.

    End before model waits or later finalization passes: children may advance.
    Nested reducers reuse the pass, with artifact-materialization modes separate.
    """
    if _FINALIZATION_CHILDREN.get() is not None:
        yield
        return
    token = _FINALIZATION_CHILDREN.set({})
    try:
        yield
    finally:
        _FINALIZATION_CHILDREN.reset(token)


def cached_finalization_children(function):
    @functools.wraps(function)
    def children(drive_root, **kwargs):
        cache = _FINALIZATION_CHILDREN.get()
        if cache is None:
            return function(drive_root, **kwargs)
        parent, root = (str(kwargs.get(key) or "").strip() for key in ("parent_task_id", "root_task_id"))
        materialize = kwargs.get("materialize_artifacts", True)
        key = (str(pathlib.Path(drive_root).resolve()), parent, root, materialize)
        if key not in cache:
            cache[key] = function(drive_root, parent_task_id=parent, root_task_id=root,
                                  scope="subtree", materialize_artifacts=materialize)
        direct = str(kwargs.get("scope") or "subtree").strip().lower() == "direct"
        excluded = str(kwargs.get("exclude_task_id") or "").strip()
        return copy.deepcopy([row for row in cache[key] if row.get("task_id") != excluded
                              and (not direct or bool(parent) and row.get("parent_task_id") == parent)])
    return children
