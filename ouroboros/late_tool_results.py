"""A timed-out tool call's own later settlement: its identity, its reader, its next-turn locator.

When the loop's wait for a tool call ends (``tool_call_timeout``) the worker keeps running
and may still settle. Its complete result is retained the ordinary way, by the worker's own
``tool_call`` manifest written before its Future resolves. Owner decision (27 Oct 2026,
#1376, option A): the next turn the work is ALREADY authorized to take — the same running
task's next round, its Resume, a retry restoring its trace, or a Continue successor — gets a
short exact locator and reads the result through ``get_task_result``. Arrival itself starts
no model, resumes no Pause or Stop, replays no handler and rewrites no finished outcome; the
timeout stays the recorded fact of that call.

Identity. One tool invocation has one ``invocation_id`` (``tool_call_log``) and one address
per record: ``tool_timeout_<tool>_<id>`` for the wait end and ``tool_<tool>_<id>`` for the
result (``observability.new_call_id``), both carrying the invocation id and attempt. A pair
counts only when both manifests of the SAME task name the same invocation, tool, provider
call id and attempt, so a provider id reused by a later call is never joined to it and a
manifest merely present under a guessed name is never admitted (``late_call_ref``). A result
that was never retained, and a manifest written before invocation ids were recorded, stay
unknown: nothing is announced and the reader refuses.

Watching. A trace entry whose wait ended keeps ``late_watch`` (``late_watch_fields``; a
handler's own TOOL_TIMEOUT settled inside the wait and is not watched), so a continuation
restoring the saved trace keeps watching. A Continue successor walks its whole Continue
ancestry before its first request (``_walk_lineage``): each ancestor's own wait-end manifests
and the predecessor its result row names, each read once, so a timeout older than any window,
or owed through successors that died before their first turn, reaches that first request.
Admission of a Continue holds while a predecessor's timed-out call still runs (its
``launch_handoffs`` claim, released after the result manifest is written), so an ancestor's
late result landed before the successor existed or never will: it is probed once. The
task's own calls and a retry-linked owner's are probed every round until they settle; a
call whose worker died keeps that one existence check per root for the context's life.
Probes read only the outstanding invocations' two known addresses, never a log; every
settlement is announced in the round about to be sent, once per context (``DELIVERED_KEY``,
saved with the trace), and a later successor is a new context and is told again. A read
that fails now is no absence: that ancestor or address is read again before the next
request, and the reader names the failure rather than refusing the call's identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ouroboros.observability import call_manifest_path, new_call_id

log = logging.getLogger(__name__)

WATCH_KEY = "late_tool_watch"
DELIVERED_KEY = "late_tool_results"
NOTE_KIND = "late_tool_result"
_MANIFEST_BYTES = 1024 * 1024
_INVOCATION = re.compile(r"[0-9a-f]{32}")


def record_call_id(prefix: str, invocation_id: str) -> str:
    """The manifest id of one invocation's ``prefix`` record (``tool_<tool>`` result, ``tool_timeout_<tool>``
    wait end), derived from the invocation so its readers can address it; random only where that id would
    replace the invocation's own ``tool_dispatch_<id>`` source (a call to a tool named ``dispatch``)."""
    call_id = new_call_id(prefix, invocation_id)
    return new_call_id(prefix) if call_id == f"tool_dispatch_{invocation_id}" else call_id


def late_watch_fields(exec_result: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """``{"late_watch": ...}`` for a trace entry whose OUTER wait ended, its wait-end manifest
    persisted (``_make_timeout_result``), else ``{}``: TOOL_TIMEOUT alone is also a handler's
    own timely report, which has no wait end to pair and nothing left to settle."""
    invocation_id = str(exec_result.get("invocation_id") or "")
    wait_end = (exec_result.get("trace_ref") or {}).get("manifest_ref") or {}
    if not (_INVOCATION.fullmatch(invocation_id) and wait_end.get("call_id")
            == record_call_id(f"tool_timeout_{exec_result.get('fn_name')}", invocation_id)):
        return {}
    return {"late_watch": {"task_id": str(getattr(ctx, "task_id", "") or ""), "invocation_id": invocation_id,
                           "task_attempt": exec_result.get("task_attempt")}}


def _manifest_at(roots: Iterable[pathlib.Path], task_id: str, call_id: str,
                 admit: Optional[Callable[[pathlib.Path], None]] = None) -> Optional[Tuple[dict, dict]]:
    """The first loose manifest of ``call_id`` among ``roots``: ``(manifest, ref)``, else ``None``.

    Only ``admit``'s own refusal propagates as ``PermissionError``. A file system failure is no
    absence: when no root yields the manifest and one could not be read, ``OSError`` says so."""
    unreadable = None
    for root in roots:
        try:
            path = call_manifest_path(root, task_id, call_id)
        except ValueError:  # not an absolute drive root: nothing can live under it
            continue
        if admit is not None:
            admit(path)
        try:
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                raw = stream.read(_MANIFEST_BYTES + 1)
        except OSError as exc:
            unreadable = exc
            continue
        try:
            manifest = json.loads(raw.decode("utf-8")) if len(raw) <= _MANIFEST_BYTES else None
        except ValueError:
            continue
        if isinstance(manifest, dict) and manifest.get("task_id") == task_id and manifest.get("call_id") == call_id:
            return manifest, {"path": str(path), "call_id": call_id, "sha256": hashlib.sha256(raw).hexdigest()}
    if unreadable is not None:
        raise OSError(f"call manifest {call_id} is unreadable now: {type(unreadable).__name__}") from unreadable
    return None


def _pair(timeout: dict, result: dict, task_id: str, tool: str, invocation_id: str) -> bool:
    """Both records of one invocation of this task, nothing joined by provider id alone."""
    return (timeout.get("call_type") == "tool_timeout" and result.get("call_type") == "tool_call"
            and all(row.get("task_id") == task_id and row.get("tool") == tool
                    and row.get("invocation_id") == invocation_id for row in (timeout, result))
            and timeout.get("tool_call_id") == result.get("tool_call_id")
            and timeout.get("task_attempt") == result.get("task_attempt"))


def late_settlement(roots: Iterable[pathlib.Path], task_id: str, tool: str, invocation_id: str) -> Optional[dict]:
    """The verified late settlement of one timed-out invocation, ``None`` while there is none,
    ``OSError`` while either record cannot be read."""
    roots = list(dict.fromkeys(pathlib.Path(root) for root in roots if root))
    if not (task_id and tool and _INVOCATION.fullmatch(str(invocation_id or ""))):
        return None
    # The result address first: while the worker still runs, a round costs one existence check per root.
    result = _manifest_at(roots, task_id, record_call_id(f"tool_{tool}", invocation_id))
    timeout = _manifest_at(roots, task_id, record_call_id(f"tool_timeout_{tool}", invocation_id)) if result else None
    if timeout is None or not _pair(timeout[0], result[0], task_id, tool, invocation_id):
        return None
    manifest, ref = result
    return {"task_id": task_id, "tool": tool, "invocation_id": invocation_id,
            "tool_call_id": manifest.get("tool_call_id"), "task_attempt": manifest.get("task_attempt"),
            "timeout_sec": timeout[0].get("timeout_sec"), "settled_at": manifest.get("created_at"),
            "status": manifest.get("status"), "tool_code": manifest.get("tool_code"),
            "call_id": ref["call_id"], "manifest_sha256": ref["sha256"], "path": ref["path"]}


def late_call_ref(root: pathlib.Path, task_id: str, call_id: str, digest: str,
                  *, admit: Optional[Callable[[pathlib.Path], None]] = None) -> dict:
    """The manifest ref of ``call_id`` at exactly ``digest`` when it is the later settlement of one
    of ``task_id``'s timed-out invocations; ``OSError`` while neither record can be read now and
    ``PermissionError`` for anything else."""
    from ouroboros.observability import read_call_manifest_ref
    from ouroboros.source_retention import retained_task_roots

    root = pathlib.Path(root)
    roots, unreadable = [root, *retained_task_roots(root, task_id)], None
    for candidate in roots:
        ref = {"call_id": call_id, "sha256": digest, "path": str(call_manifest_path(candidate, task_id, call_id))}
        try:  # a loose original, its retained child placement or its exact retained version
            manifest = read_call_manifest_ref(root, ref, task_id=task_id, admit=admit,
                                              max_bytes=_MANIFEST_BYTES, call_types=("tool_call",))
        except PermissionError:
            raise
        except (FileNotFoundError, ValueError):
            continue
        except OSError as exc:
            unreadable = exc
            continue
        tool, invocation_id = str(manifest.get("tool") or ""), str(manifest.get("invocation_id") or "")
        timeout = (_manifest_at(roots, task_id, record_call_id(f"tool_timeout_{tool}", invocation_id), admit)
                   if _INVOCATION.fullmatch(invocation_id) and call_id == record_call_id(f"tool_{tool}", invocation_id)
                   else None)
        if timeout is not None and _pair(timeout[0], manifest, task_id, tool, invocation_id):
            return ref
        unreadable = None  # a record that was read and does not pair is refused
        break
    if unreadable is not None:
        raise OSError(f"call manifest {call_id} is unreadable now: {type(unreadable).__name__}") from unreadable
    raise PermissionError("call identity is neither published by this task result "
                          "nor the later settlement of one of its timed-out calls")


def _watches(llm_trace: Dict[str, Any]) -> List[dict]:
    rows = []
    for entry in llm_trace.get("tool_calls") or []:
        watch = entry.get("late_watch") if isinstance(entry, dict) else None
        if isinstance(watch, dict):
            rows.append({**watch, "tool": entry.get("tool"),
                         "timeout_ref": (entry.get("trace_ref") or {}).get("manifest_ref") or {}})
    stored = llm_trace.get(WATCH_KEY)
    rows.extend(row for row in (stored.get("invocations") or [] if isinstance(stored, dict) else [])
                if isinstance(row, dict))
    return rows


def _recorded_root(ref: Any) -> Optional[pathlib.Path]:
    """The drive root a recorded ``observability/calls/<task>/<call>.json`` path lives in."""
    path = pathlib.Path(str((ref or {}).get("path") or "")) if isinstance(ref, dict) else None
    if path is None or len(path.parts) < 5 or path.parent.parent.name != "calls":
        return None
    return path.parents[3] if path.parents[2].name == "observability" else None


def _predecessor(meta: Any) -> str:
    continuation = meta.get("continuation") if isinstance(meta, dict) else None
    return str(continuation.get("predecessor_task_id") or "") if isinstance(continuation, dict) else ""


def _own_timeouts(roots: Iterable[pathlib.Path], task_id: str) -> List[dict]:
    """``task_id``'s own wait-end manifests that carry an invocation identity, read from its call folders.

    An unreadable folder or file raises (the caller retries it); a malformed or pre-identity record is
    not a timeout this reader can pair, exactly as ``late_call_ref`` would refuse it."""
    found: Dict[str, dict] = {}
    for root in roots:
        directory = call_manifest_path(root, task_id, "probe").parent
        try:
            names = sorted(entry.name for entry in os.scandir(directory))
        except FileNotFoundError:
            continue
        for name in names:
            if not (name.startswith("tool_timeout_") and name.endswith(".json")):
                continue
            raw = (directory / name).read_bytes()
            try:
                manifest = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            if not isinstance(manifest, dict):
                continue
            tool, invocation_id = str(manifest.get("tool") or ""), str(manifest.get("invocation_id") or "")
            if (manifest.get("call_type") == "tool_timeout" and manifest.get("task_id") == task_id
                    and _INVOCATION.fullmatch(invocation_id)
                    and name[:-5] == record_call_id(f"tool_timeout_{tool}", invocation_id)):
                found.setdefault(invocation_id, {"task_id": task_id, "invocation_id": invocation_id, "tool": tool,
                                                 "timeout_ref": {"path": str(directory / name), "call_id": name[:-5]}})
    return list(found.values())


def _walk_lineage(ctx: Any, llm_trace: Dict[str, Any]) -> None:
    """A Continue successor watches every timed-out call of its whole Continue ancestry.

    The walk finishes before the first request: each ancestor is read once, with the reciprocal
    retry predecessors and retained placements its sources already follow
    (``source_retention._source_locations``), from those owners' wait-end manifests and its own
    result row (which names its predecessor). The cursor rides the trace, so a Resume carries it
    and later rounds read nothing; an ancestor that cannot be read now stops the walk there and
    stays first in line, read again before the next request, never dropped. An ancestor's own
    call is marked ``settled_or_never`` (Continue admission held while it ran)."""
    from ouroboros.source_retention import _source_locations
    from ouroboros.task_results import load_task_result, validate_task_id

    state = llm_trace.get(WATCH_KEY)
    if not isinstance(state, dict):
        first = _predecessor(getattr(ctx, "task_metadata", None))
        if not first:
            return
        state = llm_trace[WATCH_KEY] = {"pending": [first], "walked": [], "invocations": []}
    root = pathlib.Path(getattr(ctx, "budget_drive_root", "") or ctx.drive_root)
    pending = state.setdefault("pending", [])
    while pending:
        task_id = str(pending[0])
        try:
            validate_task_id(task_id)
        except ValueError:  # no record can live under it: a named gap, not a retry every round
            state.setdefault("gaps", []).append({"task_id": task_id, "reason": "invalid_task_id"})
            pending.pop(0)
            continue
        try:
            found = [{**item, **({"settled_or_never": True} if owner == task_id else {})}
                     for place, owner in _source_locations(root, task_id) for item in _own_timeouts([place], owner)]
            row = load_task_result(root, task_id)
        except Exception as exc:  # unreadable now is not "none timed out": read it again next round
            state["retry"] = {"task_id": task_id, "reason": type(exc).__name__}
            log.warning("Late-result lineage of %s unreadable at %s", getattr(ctx, "task_id", ""), task_id,
                        exc_info=True)
            return
        known = {(item.get("task_id"), item.get("invocation_id")) for item in state.setdefault("invocations", [])}
        for item in found:
            if (item["task_id"], item["invocation_id"]) not in known:
                known.add((item["task_id"], item["invocation_id"]))
                state["invocations"].append(item)
        pending.pop(0)
        walked = state.setdefault("walked", [])
        walked.append(task_id)
        state.pop("retry", None)
        previous = _predecessor((row or {}).get("metadata"))
        if previous and previous not in walked and previous not in pending:
            pending.append(previous)


def _note(rows: List[dict]) -> str:
    many = len(rows) > 1
    lines = [f"[LATE_TOOL_RESULT] {'These tool calls' if many else 'This tool call'} returned TOOL_TIMEOUT and "
             f"settled afterwards. {'Each' if many else 'The'} timeout stands as recorded and nothing was run again; "
             f"{'each' if many else 'the'} complete result is retained at its get_task_result source address:"]
    for row in rows:
        outcome = "/".join(str(row[key]) for key in ("status", "tool_code") if row.get(key))
        lines.append(
            f"- {row['tool']} (provider call {row.get('tool_call_id')}, attempt {row.get('task_attempt')}, "
            f"wait ended after {row.get('timeout_sec')}s; settled {row.get('settled_at') or 'at an unrecorded time'}"
            f"{', ' + outcome if outcome else ''}): task_id={row['task_id']!r}, call_id={row['call_id']!r}, "
            f"manifest_sha256={row['manifest_sha256']!r}")
    return "\n".join(lines)


def deliver_late_results(messages: List[Dict[str, Any]], ctx: Any, llm_trace: Optional[Dict[str, Any]],
                         drive_logs: Any, append: Callable[..., None]) -> bool:
    """Announce, once, every outstanding timed-out call that has settled, in the round about to be sent.

    The address is stated, never a verdict on this actor's reach: ``get_task_result``'s own
    dispatch admits or refuses the read. A probe that cannot read now keeps its watch."""
    if not isinstance(llm_trace, dict):
        return False
    _walk_lineage(ctx, llm_trace)
    lineage = llm_trace.get(WATCH_KEY) or {}
    unavailable = [lineage["retry"]] if lineage.get("retry") else []
    seen = {(row.get("task_id"), row.get("invocation_id"))
            for row in llm_trace.get(DELIVERED_KEY) or [] if isinstance(row, dict)}
    settled, closed = [], set()
    base = [pathlib.Path(drive_logs).parent if drive_logs else None, getattr(ctx, "drive_root", None),
            getattr(ctx, "budget_drive_root", None)]
    for watch in _watches(llm_trace):
        key = (watch.get("task_id"), watch.get("invocation_id"))
        if key in seen:
            continue
        seen.add(key)
        try:
            found = late_settlement([_recorded_root(watch.get("timeout_ref")), *base],
                                    str(key[0] or ""), str(watch.get("tool") or ""), str(key[1] or ""))
        except OSError as exc:
            log.warning("Late result of %s/%s unreadable now; probed again next round", *key, exc_info=True)
            unavailable.append({"task_id": key[0], "invocation_id": key[1], "reason": type(exc).__name__})
            continue
        if found is not None:
            settled.append(found)
        elif watch.get("settled_or_never"):
            closed.add(key)
    state = llm_trace.get(WATCH_KEY)
    if closed and isinstance(state, dict):
        state["invocations"] = [row for row in state.get("invocations") or []
                                if (row.get("task_id"), row.get("invocation_id")) not in closed]
    # A round may finish: a failed discovery must be visible in this request, not
    # only in logs or a hypothetical later round. Keep each gap once per context.
    announced = llm_trace.setdefault("late_tool_unavailable", [])
    new_gaps = [row for row in unavailable if row not in announced]
    if new_gaps:
        append(messages, "[LATE_TOOL_RESULT_UNAVAILABLE] Late-result discovery could not read these sources. "
               "Whether a late result exists there is unknown; the watch remains for the next authorized turn. "
               "No tool was repeated. Owning task/invocation: " + json.dumps(new_gaps, ensure_ascii=False),
               kind="late_tool_result_unavailable")
        announced.extend(new_gaps)
    if not settled:
        return bool(new_gaps)
    append(messages, _note(settled), kind=NOTE_KIND)
    llm_trace.setdefault(DELIVERED_KEY, []).extend(
        {key: row[key] for key in ("task_id", "invocation_id", "tool", "tool_call_id", "call_id", "manifest_sha256")}
        for row in settled)
    return True
