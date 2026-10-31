"""Verdict-neutral storage and accounting for the Safety Supervisor."""

import json
import logging
import pathlib
import threading
from typing import Any, Dict, Optional

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_CHECK_CHIPS = ("checked", "warned", "sent_despite", "held", "not_checked")
_CHECK_TOOL_CHARS = 160
_RECORDED_CHIPS: dict[tuple[pathlib.Path, str], set[str]] = {}
_CHIP_LOCK = threading.RLock()


def coalesce_verdicts(previous, incoming):
    """Retain distinct evidence in order; repeats carry a count instead of another row."""
    def identity(verdict):
        return tuple(verdict.get(key) for key in ("stage", "status", "reason"))

    rows = {identity(verdict): dict(verdict) for verdict in previous}
    for verdict in incoming:
        key = identity(verdict)
        if key in rows:
            rows[key]["repeat_count"] = rows[key].get("repeat_count", 0) + 1
        else:
            rows[key] = {**verdict, "repeat_count": 0}
    return tuple(rows.values())


def verdict_history_bytes(history):
    """Charge distinct rows and their counters once; counter growth is not new evidence."""
    evidence = {key: {**row, "verdicts": [
        {**verdict, "repeat_count": 0} for verdict in row["verdicts"]]}
        for key, row in history.items()}
    return len(json.dumps(evidence, ensure_ascii=False).encode("utf-8", "replace"))


def _record_task_checks(ctx: Optional[Any], event: Dict[str, Any]) -> None:
    """Keep one display fact per chip in the existing task, never retry authority.

    Five entries at most, with bounded tool names; the full audit stays in events.
    The task store merges under its write lock, so parallel checks cannot lose chips
    or overwrite a terminal transition. Missing tasks are never manufactured here.
    Confirmed chips are remembered for this process; repeats still reach the audit.
    """
    task_id = str(event.get("task_id") or "")
    if not task_id:
        return
    kind = event.get("type")
    chips = {"safety_sent_despite_check": ("sent_despite",), "safety_hold": ("held",),
             "safety_mode_skip": ("not_checked",), "safety_declared_read_skip": ("not_checked",)}.get(kind, ())
    if kind == "safety_verdict":
        chips = tuple(dict.fromkeys("checked" if verdict["status"] == "SAFE" else "warned"
                                   for verdict in event.get("verdicts", ()) if verdict.get("status")))
    if not chips:
        return
    from ouroboros.task_store import patch_document
    from ouroboros.tool_access import canonical_data_root

    root = canonical_data_root(ctx) if getattr(ctx, "drive_root", None) else _safety_drive_root(ctx)
    tool = str(event.get("tool") or "")[:_CHECK_TOOL_CHARS]

    def merge(current):
        if not current:
            return None
        previous = current.get("safety_checks") or []
        checks = {item["chip"]: item for item in previous if item.get("chip") in _CHECK_CHIPS}
        for chip in chips:
            checks.setdefault(chip, {"chip": chip, "tool": tool})
        result = list(checks.values())
        return {**current, "safety_checks": result} if result != previous else None

    cache_key = (pathlib.Path(root).resolve(), task_id)
    with _CHIP_LOCK:
        if set(chips) <= _RECORDED_CHIPS.get(cache_key, set()):
            return
        document = patch_document(root, task_id, merge)
        if document:
            _RECORDED_CHIPS[cache_key] = {item["chip"] for item in document.get("safety_checks", ())
                                        if item.get("chip") in _CHECK_CHIPS}


def _safety_drive_root(ctx: Optional[Any]) -> pathlib.Path:
    """Where this safety call's observability records belong.

    The context owns the answer. Without one, the process's configured data root
    is the SSOT — never the old cwd-relative ``../data``, which names whatever
    directory happens to sit beside the current working directory and only
    resolves to the real root by coincidence of the dev layout (a safety call
    made with ``ctx=None`` — the extension/MCP dispatch shape — wrote its
    observability rows there). Read late off the module so test isolation and
    runtime rebinding are honored, the same resolution order the review
    surfaces already use.
    """
    root = getattr(ctx, "drive_root", None) if ctx is not None else None
    if root:
        return pathlib.Path(root)
    from ouroboros import config

    return pathlib.Path(config.DATA_DIR)


def _record_safety_usage(ctx: Optional[Any], usage_payload: Dict[str, Any]) -> None:
    """Charge a safety call that had no event queue to report on.

    The queue is the normal path; this is the fallback. The ledger writer is
    INJECTED by the context when it has one (the ``update_budget_from_usage``
    attribute the supervisor event context already carries), so a caller that
    owns its own accounting is charged where it lives. A context without one
    falls back to this process's own supervisor state, imported AT CALL TIME so
    the safety module — which runs inside every worker and on every tool call —
    keeps no import-time dependency on the supervisor package, exactly as the
    six sibling call sites of this writer already do.
    """
    sink = getattr(ctx, "update_budget_from_usage", None) if ctx is not None else None
    if callable(sink):
        sink(usage_payload)
        return
    from supervisor.state import update_budget_from_usage

    update_budget_from_usage(usage_payload)


def _emit_durable_safety_event(ctx: Optional[Any], event: Dict[str, Any]) -> None:
    """Emit a safety audit event durably AT THE MOMENT of the decision (v6.54.3).

    The canonical durable pattern (mirrors control._emit_swarm_fanout): a DIRECT
    append into the drive's events.jsonl — the per-worker log sink forwards
    appended lines to the live dashboard, so no separate queue put is needed and
    a queued-but-undrained event can never be lost to a worker death (review
    round 9, P3). The event_queue is only the last resort for contexts with no
    drive_logs at all."""
    payload = {
        "task_id": str(getattr(ctx, "task_id", "") or "") if ctx is not None else "",
        "ts": utc_now_iso(),
        **event,
    }
    try:
        _record_task_checks(ctx, payload)
    except Exception:
        log.error("failed to retain task safety checks for %s", payload["task_id"], exc_info=True)
    try:
        drive_logs = getattr(ctx, "drive_logs", None) if ctx is not None else None
        if callable(drive_logs):
            from ouroboros.utils import append_jsonl

            if append_jsonl(drive_logs() / "events.jsonl", payload):
                return
            # append_jsonl answers whether the row landed (its documented contract for
            # important events); a False here means the durable half of the disclosure
            # is missing, so fall through to the queue rather than returning silently.
            log.error(
                "durable safety event %s not persisted; falling back to event queue",
                event.get("type") or "?",
            )
        eq = getattr(ctx, "event_queue", None) if ctx is not None else None
        if eq is not None:
            eq.put_nowait(payload)
    except Exception:
        # The durable audit is a P3 invariant for waved-through safety decisions —
        # a lost emit must not be TRULY silent. Never fatal (a failed audit write
        # must not break a task), but escalate to error so the loss is itself
        # traced (adversarial review r1 #21).
        log.error("failed to emit durable safety event %s", event.get("type") or "?", exc_info=True)


def _emit_safety_mode_skip(ctx: Optional[Any], tool_name: str, mode: str, policy: str) -> None:
    """Durable audit for a safety LLM check waved through by a non-full owner mode.

    P3: an advisory/off mode is legitimate ONLY while every decision it waves
    through leaves a loud, durable trace at the moment it happens (review round 1)."""
    _emit_durable_safety_event(ctx, {
        "type": "safety_mode_skip",
        "tool": tool_name,
        "safety_mode": mode,
        "policy": policy,
    })
