"""Parse, retain and account for reported Safety verdicts without assigning them execution authority."""

import contextlib
import json
import logging
import pathlib
import threading
import time
import uuid
from typing import Any, Dict, Optional

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_CHECK_CHIPS = ("checked", "warned", "sent_despite", "held", "not_checked")
_CHECK_TOOL_CHARS = 160
_RECORDED_CHIPS: dict[tuple[pathlib.Path, str], set[str]] = {}
_CHIP_LOCK = threading.RLock()


def parse_safety_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse the ``{"status", "reason", "speech", "quote"}`` verdict from a Safety model response.

    ``speech`` is kept only as a boolean and ``quote`` only as non-empty text; both are the checker's claims.
    Models occasionally wrap the JSON in prose or code fences despite the prompt.
    The whole reply and every outermost ``{...}`` in it are candidates; the
    strictest recognized status wins (DANGEROUS > SUSPICIOUS > SAFE, first of
    equals), so a quoted SAFE cannot outrank a direct flag. Only when no candidate
    carries a verdict is the whole reply read as ``{"answer": "<verdict JSON>"}``
    with no other key: that string is decoded once, never searched. Otherwise
    None: the caller fails closed.
    """
    clean = text.replace("```json", "").replace("```", "").strip()
    candidates = [clean]
    depth = 0
    start = -1
    in_string = False
    escape = False
    for idx, ch in enumerate(clean):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(clean[start:idx + 1])
                start = -1
    best: Dict[str, Any] | None = None
    rank = {"SAFE": 1, "SUSPICIOUS": 2, "DANGEROUS": 3}

    def verdict_of(raw: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(raw)
        status = str(obj.get("status") or "").upper() if isinstance(obj, dict) else ""
        if status not in rank:
            return None
        quote = obj.get("quote")
        return {"status": status, "reason": str(obj.get("reason") or ""),
                **({"speech": obj["speech"]} if isinstance(obj.get("speech"), bool) else {}),
                **({"quote": quote} if isinstance(quote, str) and quote.strip() else {})}

    for candidate in candidates:
        try:
            found = verdict_of(candidate)
        except json.JSONDecodeError:
            continue
        if found and (best is None or rank[found["status"]] > rank[best["status"]]):
            best = found
    if best is None:
        try:
            envelope = json.loads(clean)
            if isinstance(envelope, dict) and list(envelope) == ["answer"] and isinstance(envelope["answer"], str):
                best = verdict_of(envelope["answer"])
        except (ValueError, RecursionError):  # a degenerate payload stays unparseable, as before
            pass
    return best



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
        {**verdict, "repeat_count": 0} for verdict in row.get("verdicts", ())]}
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
             "safety_mode_skip": ("not_checked",), "safety_declared_read_skip": ("not_checked",),
             "safety_sent_without_assessment": ("not_checked",)}.get(kind, ())
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


def _canonical_logs(ctx: Optional[Any], logs: pathlib.Path) -> Optional[pathlib.Path]:
    """The canonical shared ``logs`` directory when ``logs`` belongs to a per-task drive, else None.

    A per-task drive is pruned two days after its task; the shared log keeps the audit. When the
    drive is the canonical root the one row already landed there, so nothing is written twice."""
    from ouroboros.preflight_receipts import canonical_data_root

    root = canonical_data_root(ctx)
    if root is None:
        return None
    canonical = (pathlib.Path(root) / "logs").resolve(strict=False)
    return None if canonical == pathlib.Path(logs).resolve(strict=False) else canonical


def _emit_durable_safety_event(ctx: Optional[Any], event: Dict[str, Any]) -> None:
    """Emit a safety audit event durably AT THE MOMENT of the decision (v6.54.3).

    The canonical durable pattern (mirrors control._emit_swarm_fanout): a DIRECT
    append into the drive's events.jsonl — the per-worker log sink forwards
    appended lines to the live dashboard, so no separate queue put is needed and
    a queued-but-undrained event can never be lost to a worker death (review
    round 9, P3). A per-task drive's row is also appended to the canonical shared
    log (``_canonical_logs``), which outlives the drive's pruning. The event_queue
    is only the last resort for contexts with no drive_logs at all."""
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

            logs = drive_logs()
            if append_jsonl(logs / "events.jsonl", payload):
                if (shared := _canonical_logs(ctx, logs)) is not None and not append_jsonl(shared / "events.jsonl", payload):
                    log.error("durable safety event %s not copied to the shared log", event.get("type") or "?")
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


@contextlib.contextmanager
def safety_check_wait(ctx: Optional[Any], tool_name: str):
    """One host wait episode in the task's progress stream while a tool call waits on its Safety check
    (``checkpoint_waits``): ``entered`` before the first attempt, ``ended`` however the check ends (an answer,
    a failure or a cancellation). Presentation only: it never changes or fails the check."""
    episode_id, started = uuid.uuid4().hex, time.monotonic()

    def say(phase):
        queue_, task_id = getattr(ctx, "event_queue", None), str(getattr(ctx, "task_id", "") or "")
        if queue_ is None or not task_id:
            return
        try:
            from ouroboros.loop_messages import _emit_checkpoint_event

            detail = "Checking safety" if phase == "entered" else "Safety check ended"
            drive_root, chat_id = getattr(ctx, "drive_root", None), getattr(ctx, "current_chat_id", None)
            _emit_checkpoint_event(queue_, task_id, pathlib.Path(drive_root) / "logs" if drive_root else None, {
                "checkpoint_kind": "safety_check_wait", "owner_visible": True, "tool": tool_name,
                "phase": phase, "episode_id": episode_id, "elapsed_sec": time.monotonic() - started,
                "detail": detail, "content": detail, "text": detail, "role": "system", "system_type": "task_checkpoint",
                **({"chat_id": chat_id} if chat_id is not None else {}),
            })
        except Exception:
            log.debug("Could not disclose the Safety check wait", exc_info=True)

    say("entered")
    try:
        yield
    finally:
        say("ended")
