"""Retryable Project/Main terminal publication, with no cognition or timer.

The first root terminal transition records provenance; readiness precedes all
effects and the Project receipt retains Main disposition after readiness retires.
A separate short-lived projection
lock serializes callbacks, never holding the result lock across reentrant IO.
The chat row's token heals append/receipt-write crashes. Main uses the existing
bounded outbox; its external send/register crash gap remains at-least-once.

The same rows are the source of a memory page's host stamp (``stamp_facts``,
``part_stamp``): read and copied, never written or interpreted here.
"""
from __future__ import annotations

import json
import logging
import pathlib
import uuid
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterable

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.task_results import (
    _TRULY_TERMINAL_STATUSES,
    is_reconciled_presence_placeholder, load_task_result, resolve_task_lineage, task_result_path, write_task_result,
)
from ouroboros.utils import jsonl_chain_handles, utc_now_iso

log = logging.getLogger(__name__)
SETTLEMENT_NONE, SETTLEMENT_DEFERRED, SETTLEMENT_SETTLED = "none", "deferred", "settled"


def _settled(row: dict) -> bool:
    """Settled for publication: a host-reconciled presence placeholder is not a result (the event
    re-runs), so it owes no terminal projection even when an earlier release already recorded readiness."""
    return (row.get("status") in _TRULY_TERMINAL_STATUSES and not is_reconciled_presence_placeholder(row)
            and row.get("admission_outcome") != "never_admitted")


def _lineage(tid: str, row: dict) -> dict:
    return resolve_task_lineage(tid, metadata=row.get("metadata"), **{
        key: row.get(key) for key in (
            "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from",
        )
    })


# Shared with the executor occurrence fact and wake's publication witness.
from ouroboros.terminal_time import task_attempt_witness as _attempt, terminal_time_fact


def _witness(row: dict) -> dict:
    return {"attempt": _attempt(row), "status": row.get("status"),
            "checkpoint": row.get("root_phase_checkpoint"),
            "artifact_status": row.get("artifact_status"),
            "artifact_bundle": row.get("artifact_bundle")}


def _files_ready(root: Any, tid: str, row: dict) -> bool:
    from ouroboros.headless import terminal_task_files_ready

    return terminal_task_files_ready(pathlib.Path(root), {**row, "id": tid}, row)


def _open(row: dict) -> bool:
    checkpoint = row.get("root_phase_checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("post_task_synthesis"):
        return False
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    return post_task_synthesis_is_open(checkpoint.get("post_task_synthesis"))


@contextmanager
def _publication_lock(root: Any, tid: str):
    path = task_result_path(root, tid, create=False).with_suffix(".projection.lock")
    fd = acquire_exclusive_file_lock(path, timeout_sec=0.01, poll_sec=0.005)
    try:
        yield fd is not None
    finally:
        if fd is not None:
            release_exclusive_file_lock(path, fd)


def _prepare(root: Any, tid: str, task: dict, event: dict) -> dict:
    """Durably record readiness from canonical terminal authority, before IO."""
    def prepare(current: dict, _patch: dict):
        if not _settled(current):
            return None
        effective = {**task, **current}
        if not _lineage(tid, effective)["is_root_task"]:
            return None
        ready = current.get("canonical_terminal_projection_ready")
        if (not isinstance(ready, dict)
                and current.get("canonical_terminal_projection_origin") != "terminal_transition"):
            # Terminal status alone is historical, not proof of new debt.
            # Existing readiness remains eligible across protocol upgrades.
            return None
        marker = current.get("canonical_terminal_projection")
        if isinstance(ready, dict) and ready.get("attempt") == _attempt(current):
            return None
        if isinstance(marker, dict) and not isinstance(ready, dict):
            # Pre-protocol Project receipts belonged to the legacy Main sender.
            # Do not replay those historical rows after delivered IDs age out.
            if "attempt" not in marker or marker["attempt"] == _attempt(current):
                return None
        chat_id = effective.get("chat_id")
        if chat_id is None:
            chat_id = event.get("chat_id", 0)
        return {"status": current["status"], "canonical_terminal_projection_ready": {
            "summary_id": f"task-terminal:{tid}", "token": uuid.uuid4().hex,
            "attempt": _attempt(current), "task_done_ts": utc_now_iso(),
            "terminal_time": terminal_time_fact(current),
            "chat_id": int(chat_id or 0),
        }}

    # Absence is not a terminal authority, and must never create a completed row.
    stored = load_task_result(root, tid, strict=True)
    if not stored or not _settled(stored):
        return stored or {}
    return write_task_result(root, tid, stored["status"], strict_existing_dict=True,
                             _field_projector=prepare)


def _project_row(tid: str, row: dict, event: dict, ready: dict) -> dict:
    from ouroboros import project_dialogue as dialogue
    from ouroboros.project_facts import resolve_project_id

    lineage = _lineage(tid, row)
    is_root = lineage["is_root_task"]
    role = str(row.get("role") or ("root" if is_root else "child"))
    parent = str(lineage["parent_task_id"] or "")
    phase = dialogue.outcome_phase(row, event)
    outcome = dialogue.OUTCOME_PHASE_HEADLINE[phase]
    chat_id = int(ready.get("chat_id", row.get("chat_id", event.get("chat_id", 0))) or 0)
    text = (f"{outcome}. Root task {tid}." if is_root
            else f"{outcome}. {role} (child {tid} of {parent or 'unknown'}).")
    verdict = dialogue._completion_verdict(row, event)
    excerpt = dialogue._completion_excerpt(row, chat_id=chat_id, salvage_only=True)
    text += "".join(f" {part}" for part in (verdict, excerpt) if part)
    from ouroboros.dialogue_provenance import presence_provenance_fields

    result = {
        "ts": utc_now_iso(),
        "terminal_time": ready.get("terminal_time") or terminal_time_fact(row),
        "direction": "system", "type": "task_summary",
        **presence_provenance_fields(row),  # a presence room labels its terminal row like every other row
        "summary_id": f"task-terminal:{tid}",
        "summary_kind": "terminal_root_projection" if is_root else "terminal_result_projection",
        "task_id": tid, "parent_task_id": parent, "root_task_id": lineage["root_task_id"],
        "project_id": resolve_project_id(row), "chat_id": chat_id,
        "delegation_role": str(row.get("delegation_role") or ""), "role": role,
        "status": row["status"], "outcome": outcome, "outcome_phase": phase, "outcome_final": True,
        "outcome_authority": "canonical_task_result_after_finalization",
        "outcome_axes": row.get("outcome_axes") or {}, "reason_code": str(row.get("reason_code") or ""),
        "result_ref": {"kind": "task_result", "task_id": tid, "reader": "get_task_result"}, "text": text,
        **({"reason_detail": verdict} if verdict else {}),
        **({"terminal_projection_token": ready["token"]} if ready.get("token") else {}),
    }
    if isinstance(row.get("model_execution"), dict):
        result["model_execution"] = dict(row["model_execution"])
    from ouroboros.history_retention import retention_summary

    retention = retention_summary(row)
    if retention:
        result["history_retention"] = retention
    return result


def _already_in_chat(root: Any, row: dict) -> dict | None:
    path = pathlib.Path(root) / "logs" / "chat.jsonl"
    # Pin the live inode before enumerating archives: rotation between append
    # and receipt persistence must not turn an existing row into an absence.
    with jsonl_chain_handles(path, strict=True) as handles:
        for segment, stream in handles:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    log.warning("Skipping malformed terminal projection history row at %s:%s", segment, line_number)
                    continue
                if not isinstance(entry, dict):
                    log.warning("Skipping non-object terminal projection history row at %s:%s", segment, line_number)
                    continue
                if (entry.get("summary_id") == row["summary_id"]
                        and entry.get("terminal_projection_token") == row.get("terminal_projection_token")):
                    return entry
    return None


def _append_project(root: Any, tid: str, task: dict, event: dict) -> bool:
    from ouroboros import project_dialogue as dialogue

    stored = _prepare(root, tid, task, event)
    if not stored or not _settled(stored):
        return False
    effective = {**task, **stored}
    is_root = _lineage(tid, effective)["is_root_task"]
    ready = stored.get("canonical_terminal_projection_ready") or {}
    marker = stored.get("canonical_terminal_projection")
    if isinstance(marker, dict) and (not ready or marker.get("token") == ready.get("token")):
        return False
    if is_root and (not ready or _open(stored) or not _files_ready(root, tid, effective)):
        return False
    row = _project_row(tid, effective, {key: event[key] for key in ("ts", "chat_id") if key in event}, ready)
    appended = False
    existing_row = _already_in_chat(root, row) if is_root else None
    if existing_row:
        row = existing_row  # append/receipt crash: retain actual publication time
    else:
        appended = dialogue.append_canonical_task_summary(root, row)
        if not appended:
            return False

    def receipt(current: dict, _patch: dict):
        if (_witness(current) != _witness(stored)
                or current.get("canonical_terminal_projection_ready") != stored.get("canonical_terminal_projection_ready")):
            return None
        return {"status": current["status"], "canonical_terminal_projection": {
            "summary_id": row["summary_id"], "summary_kind": row["summary_kind"],
            "written_at": row["ts"], "chat_id": row["chat_id"],
            "attempt": _attempt(stored), **({"token": ready["token"]} if ready.get("token") else {}),
        }}

    write_task_result(root, tid, stored["status"], strict_existing_dict=True, _field_projector=receipt)
    return appended


def append_terminal_projection(root: Any, tid: str, task: dict, event: dict, *, result: dict | None = None) -> bool:
    with _publication_lock(root, tid) as acquired:
        if not acquired:
            return False
        # Compatibility callers may hand over the first terminal observation.
        # Persist it before readiness/effects; never overwrite existing authority.
        if result and _settled(result) and load_task_result(root, tid, strict=True) is None:
            fields = {**(task or {}), **result}
            status = fields.pop("status")
            fields.pop("task_id", None)
            write_task_result(root, tid, status, create_only=True, strict_existing_dict=True,
                              _terminal_time_source=result, **fields)
        return _append_project(root, tid, task or {}, event or {})


def clear_terminal_projection_obligation(root: Any, tid: str, expected: dict, disposition: str) -> bool:
    """CAS the complete readiness token and canonical attempt/checkpoint witness."""
    if disposition not in {"owed", "ineligible"}:
        return False
    cleared = False

    def retire(current: dict, _patch: dict):
        nonlocal cleared
        ready = expected.get("canonical_terminal_projection_ready")
        marker = current.get("canonical_terminal_projection")
        if (not isinstance(ready, dict) or not isinstance(marker, dict)
                or current.get("canonical_terminal_projection_ready") != ready
                or marker != expected.get("canonical_terminal_projection")
                or _witness(current) != _witness(expected) or _open(current)
                or not _files_ready(root, tid, current)
                or marker.get("token") != ready.get("token")):
            return None
        cleared = True
        return {"status": current["status"], "canonical_terminal_projection_ready": None,
                "canonical_terminal_projection": {**marker, "main_disposition": disposition}}

    write_task_result(root, tid, str(expected.get("status") or "completed"),
                      strict_existing_dict=True, _field_projector=retire)
    return cleared


def settle_terminal_projection(drive_root: Any, task_id: str, *, task: dict | None = None,
                               event: dict | None = None) -> str:
    from ouroboros import project_dialogue as dialogue

    tid = str(task_id or "").strip()
    if not tid:
        return SETTLEMENT_NONE
    try:
        with _publication_lock(drive_root, tid) as acquired:
            if not acquired:
                return SETTLEMENT_DEFERRED
            stored = _prepare(drive_root, tid, task or {}, event or {})
            if (not isinstance(stored.get("canonical_terminal_projection_ready"), dict)
                    or is_reconciled_presence_placeholder(stored)):  # readiness an earlier release recorded
                return SETTLEMENT_NONE
            if (_open(stored) or not _settled(stored)
                    or not _files_ready(drive_root, tid, {**(task or {}), **stored})):
                return SETTLEMENT_DEFERRED
            _append_project(drive_root, tid, task or {}, event or {})
            # IO may have advanced canonical state. Render Main from the NEW
            # result, never the task_done snapshot passed to this continuation.
            stored = load_task_result(drive_root, tid, strict=True) or {}
            ready = stored.get("canonical_terminal_projection_ready")
            marker = stored.get("canonical_terminal_projection")
            if (not isinstance(ready, dict) or not isinstance(marker, dict)
                    or marker.get("token") != ready.get("token") or _open(stored)
                    or not _settled(stored)
                    or not _files_ready(drive_root, tid, {**(task or {}), **stored})):
                return SETTLEMENT_DEFERRED
            effective = {**(task or {}), **stored, "id": tid}
            done = {**(event or {}), **stored, "ts": ready["task_done_ts"], "chat_id": ready["chat_id"]}
            retired = False

            def retire_owed() -> bool:
                nonlocal retired
                retired = clear_terminal_projection_obligation(drive_root, tid, stored, "owed")
                return retired

            disposition, _queued = dialogue.project_completion_delivery_outcome(
                drive_root, done, tid, effective, stored, done, _on_owed=retire_owed)
            if retired or clear_terminal_projection_obligation(drive_root, tid, stored, disposition):
                return SETTLEMENT_SETTLED
            return SETTLEMENT_DEFERRED
    except Exception:
        log.warning("Terminal projection deferred for %s", tid, exc_info=True)
        return SETTLEMENT_DEFERRED


def terminal_projection_owed(task_id: str, row: dict) -> bool:
    """Select publication debt, including readiness already recorded before a crash.

    Both recovery scans use this before entering the writer's mailbox capture.
    Open synthesis still owns its continuation; this predicate never settles it.
    """
    if not row or not _settled(row):
        return False
    ready = row.get("canonical_terminal_projection_ready")
    if (not isinstance(ready, dict)
            and row.get("canonical_terminal_projection_origin") != "terminal_transition"):
        return False
    marker = row.get("canonical_terminal_projection")
    if isinstance(marker, dict) and not isinstance(ready, dict):
        if "attempt" not in marker or marker["attempt"] == _attempt(row):
            return False
    return _lineage(task_id, row)["is_root_task"] and not _open(row)


def reconcile_terminal_projections(drive_root: Any) -> int:
    """Discover the terminal-write/readiness-write crash gap on the existing pass.

    Never re-enter synthesis or infer debt from historical terminal status. Read
    each authority strictly: one bad sibling must not stall others or quarantine
    unknown bytes via a tolerant scan.
    """
    from ouroboros.obligations import result_rows

    settled = 0
    for row in result_rows(drive_root, "terminal_projection"):
        try:
            if terminal_projection_owed(row["task_id"], row):
                settled += settle_terminal_projection(drive_root, row["task_id"]) == SETTLEMENT_SETTLED
        except Exception:
            log.warning("Terminal projection reconciliation deferred for %s", row["task_id"], exc_info=True)
    return settled


# --- the host stamp of a memory page ------------------------------------------------------------

_TERMINAL_SUMMARIES = frozenset({"terminal_root_projection", "terminal_result_projection"})


def _source_address(address: Any) -> str | None:
    """The text address of the stamp's source row; an entry without a well-formed address has none."""
    from ouroboros.chat_chain import format_address

    try:
        return format_address(address)
    except (KeyError, TypeError, ValueError):
        return None


def _stamp_entry(task_id: str, row: dict, source: str, address: Any = None) -> dict:
    """One task's stamp: the keys every page carries, then optional facts copied from the same source.

    Copied, never interpreted: ``outcome_final`` and ``reason_code`` as the source
    holds them, ``objective_status`` from its ``outcome_axes``; the axes themselves
    stay with their source (one axis can be kilobytes of policy denials).
    """
    entry = {"task_id": task_id, "status": str(row.get("status") or ""), "outcome": str(row.get("outcome") or ""),
             "outcome_phase": str(row.get("outcome_phase") or ""), "source": source,
             "result_ref": row.get("result_ref") or {"kind": "task_result", "task_id": task_id,
                                                      "reader": "get_task_result"}}
    if row.get("reason_detail"):
        entry["review_verdict"] = str(row["reason_detail"])
    if isinstance(row.get("outcome_final"), bool):
        entry["outcome_final"] = row["outcome_final"]
    if row.get("reason_code"):
        entry["reason_code"] = str(row["reason_code"])
    axes = row.get("outcome_axes")
    objective = axes.get("objective") if isinstance(axes, dict) else None
    if isinstance(objective, dict) and objective.get("status"):
        entry["objective_status"] = str(objective["status"])
    where = _source_address(address) if address is not None else None
    if where:
        entry["source_address"] = where
    return entry


def _result_entry(root: pathlib.Path, task_id: str) -> dict:
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE, outcome_phase

    try:
        result = load_task_result(root, task_id, strict=True)  # strict: never moves the file
    except (OSError, ValueError):
        result = None
    if not isinstance(result, dict) or not result:
        return {"task_id": task_id, "status": "not_recorded"}
    try:
        phase = outcome_phase(result, {})
    except (KeyError, TypeError, ValueError, AttributeError):
        phase = ""
    return _stamp_entry(task_id, {**result, "outcome_phase": phase,
                                  "outcome": OUTCOME_PHASE_HEADLINE.get(phase, "")}, "task_results")


def stamp_facts(root: Any, task_ids: Iterable[Any], *, rows: Iterable[Any] = ()) -> dict[str, dict]:
    """The host's facts about each task, keyed by task id in the order asked: a page's stamp.

    Per task, first match wins: its terminal projection row, then its host facts row
    (both among ``rows``, the page's already-read ``(address, row[, pos])`` entries;
    a repeated row of one task takes the last), then the strict task result (the
    file never moves), else ``not_recorded``. There is no pass over the chat chain.
    """
    terminal: dict[str, tuple] = {}
    host_facts: dict[str, tuple] = {}
    for entry in rows:
        address, row = entry[0], entry[1]
        task = str(row.get("task_id") or "")
        if row.get("type") != "task_summary" or not task:
            continue
        if row.get("summary_kind") in _TERMINAL_SUMMARIES:
            terminal[task] = (address, row)
        elif row.get("summary_kind") == "host_task_facts":
            host_facts[task] = (address, row)
    facts: dict[str, dict] = {}
    for task in dict.fromkeys(str(t) for t in task_ids if str(t or "")):
        if task in terminal:
            address, row = terminal[task]
            facts[task] = _stamp_entry(task, row, str(row["summary_kind"]), address)
        elif task in host_facts:
            address, row = host_facts[task]
            facts[task] = _stamp_entry(task, row, "host_task_facts", address)
        else:
            facts[task] = _result_entry(pathlib.Path(root), task)
    return facts


def part_stamp(page_stamps: Iterable[Any]) -> dict:
    """A part's stamp from its member pages' stamps, losing no failure.

    ``counts`` holds how many tasks there are per ``(source, outcome_phase)``;
    ``tasks`` keeps in full every task whose phase is not ``done`` or whose stamp
    did not come from a terminal projection. A task stamped on two pages counts
    once, by its later stamp.
    """
    latest: dict[str, dict] = {}
    for stamp in page_stamps:
        for entry in (stamp.get("tasks") if isinstance(stamp, dict) else None) or []:
            if isinstance(entry, dict) and str(entry.get("task_id") or ""):
                latest[str(entry["task_id"])] = entry
    counts = Counter((str(entry.get("source") or ""), str(entry.get("outcome_phase") or ""))
                     for entry in latest.values())
    kept = [entry for entry in latest.values()
            if entry.get("outcome_phase") != "done" or entry.get("source") not in _TERMINAL_SUMMARIES]
    return {"tasks": kept,
            "counts": [{"source": source, "outcome_phase": phase, "tasks": n}
                       for (source, phase), n in sorted(counts.items())],
            "computed_at": utc_now_iso()}
