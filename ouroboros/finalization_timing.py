"""Finalization phase telemetry carried by the live answer delivery.

Task-local clocks follow the last model answer through the existing event queue
and journal. Durable copies omit the sidecar because monotonic origins cannot
survive a reboot. Observability re-exports the public helpers for existing callers.

Latency stamps are telemetry, never gates. Upstream of this module:
``chat.jsonl`` inbound rows carry ``message_accepted_at``, ``task_received``
carries ``activity_emitted_at``, and request/response manifests and ``llm_round``
carry ``first_request_at``/``first_answer_at``. This module adds the finalization
phases, which ride the live final event, and one ``task_finalization_timing`` row
in ``events.jsonl`` written after the sender returns.
"""

from __future__ import annotations

import copy
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


def emit_finalization_timing(event, drive_root):
    """One best-effort journal row after the deduplicated sender has returned.

    This is a host send-handler receipt, not proof of client display. A missing
    phase was not observed; failed sends/crashes and durable replay add no row.
    The sender's existing delivery identity, not a new registry, deduplicates it.
    """
    timing = event.get("_finalization_timing")
    if drive_root is None or not timing or "sender" not in timing.get("phases", {}):
        return
    try:
        from ouroboros.utils import append_jsonl

        sender = timing["phases"]["sender"]
        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            **{key: value for key, value in timing.items() if not key.startswith("_")},
            "type": "task_finalization_timing", "ts": sender["finished_at"],
            "task_id": event.get("task_id"), "delivery_id": event.get("delivery_id"),
            "basis": "send_handler_returned", "total_sec": sender["finished_sec"],
            "enqueue_to_sender_sec": (max(0.0, sender["started_sec"] - timing["enqueued_sec"])
                                      if "enqueued_sec" in timing else None),
        })
    except Exception:
        logging.getLogger(__name__).debug("Finalization timing could not be recorded", exc_info=True)
