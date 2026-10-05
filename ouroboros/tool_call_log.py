"""The durable ``tools.jsonl`` call record: one host invocation, up to three rows (#1316).

The loop freezes ONE ``invocation_id`` per tool call at wrapper entry, BEFORE the
call is handed to an executor, together with the task attempt, execution, round,
LLM call and provider ``tool_call_id`` it answers (providers reuse those ids, so
the provider id alone is not an identity). Every row of that call carries the
same frozen facts:

- ``tool_call_started`` — the host began processing the call (an executor wait
  included). It is NOT evidence that the handler ran or that any effect happened.
- ``tool_call`` — the settlement: the handler's result (or a host refusal after
  the start), with the host-measured ``elapsed_ms`` of the whole call, distinct
  from a process's own ``duration_ms``.
- ``tool_call_timeout`` — the caller's wait ended; the worker may still settle
  later. Timeout and settlement are independent facts and may land in either order.

Readers count invocations, never rows. A row without ``invocation_id`` is legacy
and stands alone (never joined by guesswork); a start with no later row is
``unknown`` — never running, never failed. Appends are ordinary appends with no
power-loss promise; each target's outcome is returned so a missed start can be
disclosed on the later rows. Nothing here vetoes execution.
"""

from __future__ import annotations

import pathlib
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from ouroboros.utils import append_jsonl

CALL_STARTED = "tool_call_started"
CALL_SETTLED = "tool_call"
CALL_WAIT_ENDED = "tool_call_timeout"
_FROZEN_KEYS = ("invocation_id", "tool_call_id", "task_attempt", "execution_id", "round_id", "llm_call_id", "args_source_ref", "args_source_status")


def new_invocation(tool_call_id: Any, correlation: Dict[str, Any], task_attempt: Any) -> Dict[str, Any]:
    """The immutable identity of one call, captured before submission."""
    frozen = {
        "invocation_id": uuid.uuid4().hex,
        "tool_call_id": str(tool_call_id or ""),
        "task_attempt": task_attempt,
        "execution_id": correlation.get("execution_id"),
        "round_id": correlation.get("round_id"),
        "llm_call_id": correlation.get("llm_call_id"),
    }
    return {**{key: value for key, value in frozen.items() if value not in (None, "")},
            "_started_mono": time.monotonic(), "_settlement_lock": threading.Lock()}


def invocation_fields(invocation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The frozen facts a row carries (never the private monotonic stamp)."""
    return {key: invocation[key] for key in _FROZEN_KEYS if invocation and key in invocation}


def persist_dispatch_source(meta: Dict[str, Any], drive_logs: pathlib.Path, task_id: str,
                            invocation: Dict[str, Any], tool: str, arguments: Any) -> None:
    """Full redacted dispatch input, in canonical observability before any execution.

    The preview sanitizer is intentionally lossy; this existing CAS/manifest is
    not. Readback verifies the actual bytes, including False/no-op writer failures.
    Failure is evidence loss, never an execution veto or a fabricated source ref.
    """
    from ouroboros.observability import persist_call, read_call_payload

    try:
        root = pathlib.Path(meta.get("budget_drive_root") or pathlib.Path(drive_logs).parent).resolve()
        call_id = f"tool_dispatch_{invocation['invocation_id']}"
        persist_call(root, task_id=task_id, call_id=call_id, call_type="tool_dispatch",
                     payload={"tool": tool, "arguments": arguments, **invocation_fields(invocation)},
                     manifest={"tool": tool, **invocation_fields(invocation)}, keep_raw=False)
        _manifest, payload, verified = read_call_payload(root, task_id=task_id, call_id=call_id)
        if payload.get("invocation_id") != invocation["invocation_id"]:
            raise ValueError("dispatch source identity mismatch")
        invocation["args_source_ref"] = verified["manifest_ref"]
        invocation["args_source_status"] = "ready"
    except Exception as exc:
        invocation["args_source_status"] = f"unavailable:{type(exc).__name__}"


def elapsed_ms(invocation: Optional[Dict[str, Any]]) -> Optional[int]:
    started = (invocation or {}).get("_started_mono")
    return None if started is None else int((time.monotonic() - started) * 1000)


def append_call_row(meta: Dict[str, Any], drive_logs: pathlib.Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Append one row to the task's log and its canonical copy; return each outcome.

    ``meta`` is the task metadata (lineage, ``budget_drive_root``). The canonical
    copy is ALWAYS attempted, even when the task-log append failed, and the result
    names ``{"task_log": ok, "canonical": ok | None}`` (None: no separate copy)."""
    from ouroboros.task_results import resolve_task_lineage

    if task_id := str(payload.get("task_id") or "").strip():
        # ONE lineage resolver (a direct root is its own root), so every task row
        # carries root_task_id/delegation_role for the task log stream readers.
        lineage = resolve_task_lineage(task_id, metadata=meta)
        payload["root_task_id"] = lineage["root_task_id"]
        if role := lineage["delegation_role"] or ("root" if lineage["is_root_task"] else ""):
            payload["delegation_role"] = role
    for key in ("parent_task_id", "task_depth"):
        if meta.get(key) not in (None, ""):
            payload[key] = meta.get(key)
    local = pathlib.Path(drive_logs) / "tools.jsonl"
    targets = {"task_log": local}
    outcome: Dict[str, Any] = {"canonical": None}
    root = str(meta.get("budget_drive_root") or "").strip()
    if root:
        try:
            candidate = pathlib.Path(root).resolve(strict=False) / "logs" / "tools.jsonl"
            if candidate != local.resolve(strict=False):
                targets["canonical"] = candidate
        except Exception:
            outcome["canonical"] = False
    # Every target gets its own attempt and outcome, including False returns.
    for name, path in targets.items():
        try:
            outcome[name] = bool(append_jsonl(path, payload))
        except Exception:
            outcome[name] = False
    return outcome


def append_failed(outcome: Optional[Dict[str, Any]]) -> bool:
    return bool(outcome) and (outcome.get("task_log") is False or outcome.get("canonical") is False)


def start_log_field(invocation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``{"start_log": outcome}`` when the call's start append failed — the ONE field every
    later row (settlement, host refusal, wait end) discloses it in; else nothing."""
    outcome = (invocation or {}).get("start_log")
    return {"start_log": outcome} if append_failed(outcome) else {}


def claim_settlement(invocation: Optional[Dict[str, Any]]) -> bool:
    """Claim operation settlement once. A wait end never owns this fact."""
    if invocation is None:
        return True
    lock = invocation.setdefault("_settlement_lock", threading.Lock())
    with lock:
        first = not invocation.get("_settled")
        invocation["_settled"] = True
        return first


def counts_as_call(row: Dict[str, Any]) -> bool:
    """A per-row LOWER bound for sizing a read window: a start, or a legacy row.

    A settlement or wait end whose start is missing (a lost append, or a start
    outside the window) is not counted here, so a window only grows; the exact
    count is ``len(logical_calls(rows))`` over the rows read, where such an
    orphan is one call and a start with its later rows is one call."""
    return row.get("type") == CALL_STARTED or not row.get("invocation_id")


def logical_calls(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group rows into calls in first-seen order.

    Each call names its ``started``/``settled``/``wait_ended`` rows and ``state``:
    ``settled`` (a result exists, in either order with a timeout), ``wait_ended``
    (the caller stopped waiting, no result recorded), ``unknown`` (only a start),
    ``legacy`` (a pre-identity row) or ``orphan`` (a result whose start row lies
    outside the rows given). A duplicate settlement (pre-guard data) never
    replaces a real one with a host ``host_error`` row."""
    calls: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    slot = {CALL_STARTED: "started", CALL_SETTLED: "settled", CALL_WAIT_ENDED: "wait_ended"}
    for row in rows:
        if not isinstance(row, dict):
            continue
        invocation_id = str(row.get("invocation_id") or "")
        if not invocation_id:
            calls.append({"state": "legacy", "tool": row.get("tool"), "args": row.get("args"), "settled": row})
            continue
        call = by_id.get(invocation_id)
        if call is None:
            call = by_id[invocation_id] = {"invocation_id": invocation_id, "tool": row.get("tool"),
                                           "args": row.get("args")}
            calls.append(call)
        key = slot.get(str(row.get("type") or ""), "settled")
        held = call.get(key)
        if held is None or (key == "settled" and held.get("status") == "host_error"
                            and row.get("status") != "host_error"):
            call[key] = row
    for call in calls:
        if "state" not in call:
            call["state"] = ("settled" if "settled" in call and "started" in call
                             else "orphan" if "settled" in call
                             else "wait_ended" if "wait_ended" in call else "unknown")
    return calls


def replay_evidence(drive_root: pathlib.Path, task_id: str, want: int = 200, *, iter_objects=None) -> Dict[str, Any]:
    """Bounded canonical invocation facts for history, independent of frozen wait metrics."""
    from ouroboros.memory import Memory
    from ouroboros.tool_capabilities import routing_action_for_tool

    rows, coverage = Memory(drive_root).read_task_recent("tools.jsonl", task_id, want, iter_objects=iter_objects)
    observations = []
    legacy = {"calls": 0, "errors": 0, "wait_ended": False, "unknown": False}
    for call in logical_calls(rows):
        if not call.get("invocation_id"):
            row = call.get("settled") or {}
            legacy["calls"] += 1
            legacy["errors"] += int(row.get("type") == CALL_SETTLED and bool(row.get("is_error")))
            legacy["wait_ended"] |= row.get("type") == CALL_WAIT_ENDED
            legacy["unknown"] |= row.get("type") == CALL_STARTED
            continue  # separate observations, never guessed joins to modern/live calls
        base = {"key": f"tool:{task_id}:{call['invocation_id']}", "tool": call.get("tool"),
                "receipt": bool(routing_action_for_tool(call.get("tool")))}
        for slot, fact in (("started", "started"), ("wait_ended", "wait_ended"), ("settled", "settled")):
            row = call.get(slot)
            if row is not None:
                observations.append({**base, "fact": fact, "live": False,
                    "receipt": base["receipt"] or row.get("completion_control") is True,
                    "status": ("error" if row.get("is_error") else "ok") if slot == "settled" else "unknown",
                    "hostError": row.get("status") == "host_error"})
    return {"observations": observations, "legacy": legacy, "coverage": coverage}


def replay_evidence_for_tasks(drive_root: pathlib.Path, task_ids: Iterable[str]) -> Dict[str, dict]:
    """Share exact parsed windows within ONE history read, keeping per-task quotas.

    Only selected tasks are retained. Tail windows still double and archive
    backfill stays bounded by the existing reader; no cache survives this request.
    """
    from ouroboros.utils import iter_jsonl_objects

    selected = dict.fromkeys(task_ids)
    windows = {}

    def parse(path, *, tail_bytes=None, gap_reasons=None):
        info = path.stat()
        stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if windows.get(path, (None,))[0] != stamp:
            windows[path] = stamp, {}
        parsed = windows[path][1]
        if tail_bytes not in parsed:
            gaps: set = set()
            rows = [row for row in iter_jsonl_objects(path, tail_bytes=tail_bytes, gap_reasons=gaps)
                    if str(row.get("task_id") or "").strip() in selected]
            parsed[tail_bytes] = rows, gaps
        rows, gaps = parsed[tail_bytes]
        if gap_reasons is not None:
            gap_reasons.update(gaps)
        return iter(rows)

    return {task_id: replay_evidence(drive_root, task_id, iter_objects=parse) for task_id in selected}
