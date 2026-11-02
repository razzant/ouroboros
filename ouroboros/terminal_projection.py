"""Retryable Project/Main terminal publication, with no cognition or timer.

The first root terminal transition records provenance; readiness precedes all
effects and the Project receipt retains Main disposition after readiness retires.
A separate short-lived projection
lock serializes callbacks, never holding the result lock across reentrant IO.
The chat row's token heals append/receipt-write crashes. Main retains its exact addressed event and retry schedule in readiness until
the strict sender acknowledges it. The existing bounded outbox remains a
transport cache; its eviction/retry limits never discharge canonical debt.
External send/register crash gaps remain at-least-once.

The same rows are the source of a memory page's host stamp (``stamp_facts``,
``part_stamp``): read and copied, never written or interpreted here.
"""
from __future__ import annotations

import logging
import pathlib
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.task_results import (
    _TRULY_TERMINAL_STATUSES,
    is_reconciled_presence_placeholder, load_task_result, resolve_task_lineage, task_result_path, write_task_result,
)
from ouroboros.utils import utc_now_iso
from ouroboros.store import StoreError

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
    from ouroboros.install_identity import require_owned_root
    require_owned_root(root)
    path = task_result_path(root, tid, create=False).with_suffix(".projection.lock")
    fd = acquire_exclusive_file_lock(path, timeout_sec=0.01, poll_sec=0.005)
    try:
        yield fd is not None
    finally:
        if fd is not None:
            release_exclusive_file_lock(path, fd)


def _delivery_disposition(root: Any, tid: str) -> dict:
    """What the outbox says about this task's terminal message (owner Fork 2, 20:48).

    ``state`` is ``owed`` while any terminal message (model final, Stop/reap
    salvage receipt, cascade digest) is still owed — debt outranks every older or
    technical receipt — ``exhausted`` when the host gave up delivering one,
    ``unknown`` when the delivery registry cannot be read, ``delivered`` with
    proven bytes, else ``""`` (no terminal message known). Only ``owed`` holds the
    Done card; unknown and exhausted are SHOWN on it, never hidden. Nothing here is
    recomputed from stored prose, and no read failure turns into a silent hold.
    A settle that runs between the terminal row and the send sees ``owed`` and waits;
    the delivery acknowledgement re-enters ``settle_terminal_projection``
    (``events_chat_delivery._handoff_delivered_review``), so the held card follows its
    answer without a sweep.
    """
    from supervisor.terminal_delivery import terminal_answer_receipts

    receipts = terminal_answer_receipts(root, tid)
    if receipts.get("registry") == "unreadable":
        return {"state": "unknown"}
    state = str(receipts.get("state") or "")
    disposition = {"state": state if state in {"owed", "exhausted", "delivered"} else ""}
    if state == "exhausted":
        row = max(receipts["exhausted"], key=lambda item: str(item.get("exhausted_at") or ""))
        disposition["delivery_id"] = str(row.get("delivery_id") or "")
        disposition["preserved_path"] = str(row.get("preserved_path") or "")
    return disposition


def _prepare(root: Any, tid: str, task: dict, event: dict) -> tuple[dict, bool]:
    """Durably record readiness from canonical terminal authority, before IO.

    Returns ``(stored, minted)``: ``minted`` says THIS call recorded the readiness,
    so no chat row can carry its token yet and the duplicate scan is skipped.
    """
    from ouroboros.install_identity import require_owned_root, require_task_roots
    require_owned_root(root)
    minted = False

    def prepare(current: dict, _patch: dict):
        nonlocal minted
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
        minted = True
        return {"status": current["status"], "canonical_terminal_projection_ready": {
            "summary_id": f"task-terminal:{tid}", "token": uuid.uuid4().hex,
            "attempt": _attempt(current), "task_done_ts": utc_now_iso(),
            "terminal_time": terminal_time_fact(current),
            "chat_id": int(chat_id or 0),
        }}

    # Absence is not a terminal authority, and must never create a completed row.
    stored = load_task_result(root, tid, strict=True)
    if not stored or not _settled(stored):
        return stored or {}, False
    require_task_roots({**task, **stored})
    return write_task_result(root, tid, stored["status"], strict_existing_dict=True,
                             _field_projector=prepare), minted


def completion_notice(record: dict) -> dict:
    """Compact transport view computed before projection loses custody/cancel facts.

    Neutral means a finished, execution-ok record has only technical caveats,
    never success or reviewer approval. Unknown/partial rows retain their causes.
    """
    from ouroboros import project_dialogue as dialogue
    axes = record.get("outcome_axes") or {}
    execution, objective = axes.get("execution") or {}, axes.get("objective") or {}
    delivery = record.get("terminal_delivery") or {}
    detail = dialogue._join_cause_clauses([
        dialogue._completion_verdict(record, {}, include_review_details=False),
        dialogue.terminal_delivery_clause(delivery.get("state", ""), delivery.get("preserved_path", "")),
    ])
    full, phase = dialogue._completion_verdict(record, {}), dialogue.outcome_phase(record, {})
    neutral = (not detail and bool(full) and execution.get("status") == "ok"
               and objective.get("status") in {"pass", "not_evaluated"} and phase in {"done", "warn"})
    return {"neutral": neutral, "detail": detail or ("" if neutral else full or str(record.get("reason_detail") or "")), "phase": phase}


def completion_event_view(root: Any, event: dict, record: dict) -> dict:
    """Add transport metadata to a pre-upgrade Main event, never rewrite its bytes.

    Both canonical readiness and the outbox can replay an old event. Its loaded
    result supplies the salvage distinction; missing source keeps the old view
    rather than guessing that the retained text contains only a status.
    """
    meta = event.get("progress_meta") if isinstance(event.get("progress_meta"), dict) else {}
    if (event.get("system_type") != "project_completion_summary"
            or isinstance(meta.get("completion_notice"), dict) or not record):
        return event
    from ouroboros.project_dialogue import _completion_excerpt

    try:
        # Source delivery is independent registry truth, not a field of the result.
        source_delivery = _delivery_disposition(root, str(event.get("task_id") or ""))
        salvage = _completion_excerpt(record, chat_id=event.get("chat_id"), salvage_only=True)
        return {**event, "progress_meta": {**meta, "completion_notice": {
            **completion_notice({**record, "terminal_delivery": source_delivery}), "salvage": salvage}}}
    except Exception:
        log.debug("legacy completion metadata unavailable; keeping retained text", exc_info=True)
        return event


def _project_row(tid: str, row: dict, event: dict, ready: dict, *, delivery: dict | None = None) -> dict:
    from ouroboros import project_dialogue as dialogue
    from ouroboros.project_facts import resolve_project_id
    from ouroboros.task_finalization import quiet_finish

    quiet = quiet_finish(row)
    lineage = _lineage(tid, row)
    is_root = lineage["is_root_task"]
    role = str(row.get("role") or ("root" if is_root else "child"))
    parent = str(lineage["parent_task_id"] or "")
    chat_id = int(ready.get("chat_id", row.get("chat_id", event.get("chat_id", 0))) or 0)
    delivery = {key: str(value or "") for key, value in (delivery or {}).items()
                if key in ("state", "delivery_id", "preserved_path")}
    if delivery.get("state") in {"exhausted", "unknown"}:
        # Honest delivery state (owner 20:48): an exhausted or unknown delivery is a
        # ROW-LEVEL warning — the same objective.warning the browser reads for its
        # pill — never a durable change to the task's own axes (the registry may be
        # repaired, the copy delivered later), plus one plain sentence and the
        # saved-answer pointer when a copy exists.
        from ouroboros.outcomes import _merge_objective_warning

        axes = dict(row.get("outcome_axes") or {}) if isinstance(row.get("outcome_axes"), dict) else {}
        objective = dict(axes.get("objective") or {}) if isinstance(axes.get("objective"), dict) else {}
        _merge_objective_warning(objective, f"terminal_delivery_{delivery['state']}")
        row = {**row, "outcome_axes": {**axes, "objective": objective}}
    else:
        delivery = {}
    phase = dialogue.outcome_phase(row, event)
    outcome = dialogue.OUTCOME_PHASE_HEADLINE[phase]
    text = (f"{outcome}. Root task {tid}." if is_root
            else f"{outcome}. {role} (child {tid} of {parent or 'unknown'}).")
    verdict = dialogue._join_cause_clauses([
        dialogue._completion_verdict(row, event),
        dialogue.terminal_delivery_clause(delivery.get("state", ""), delivery.get("preserved_path", "")),
    ])
    notice = completion_notice({**event, **row, "terminal_delivery": delivery})
    excerpt = dialogue._completion_excerpt(row, chat_id=chat_id, salvage_only=True)
    text = dialogue.QUIET_RUN_HEADLINE if quiet else text + "".join(f" {part}" for part in (verdict, excerpt) if part)
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
        "wake_silent": row.get("wake_silent") is True, "quiet_effective": quiet, **{key: row[key] for key in ("suggested_name", "wake_summary", "schedule_name") if row.get(key)},
        "outcome_authority": "canonical_task_result_after_finalization",
        # Memory stamps copy the recorded objective; readers interpret it using answer provenance.
        "outcome_axes": row.get("outcome_axes") or {}, "reason_code": str(row.get("reason_code") or ""),
        "_result_text_present": (isinstance(row.get("result"), str) and bool(row["result"]))
        if "result" in row else row.get("_result_text_present") is True,
        "terminal_origin": str(row.get("terminal_origin") or ""),
        **({"cancel_origin": dict(row["cancel_origin"])} if isinstance(row.get("cancel_origin"), dict) else {}),
        "result_ref": {"kind": "task_result", "task_id": tid, "reader": "get_task_result"}, "text": text,
        **({"reason_detail": verdict} if verdict else {}),
        "completion_notice": notice,
        **({"terminal_projection_token": ready["token"]} if ready.get("token") else {}),
        # The typed delivery disposition the browser card renders (log_events.terminalDeliveryClause).
        **({"terminal_delivery": delivery} if delivery else {}),
    }
    if isinstance(row.get("model_execution"), dict):
        result["model_execution"] = dict(row["model_execution"])
    from ouroboros.history_retention import retention_summary

    retention = retention_summary(row)
    if retention:
        result["history_retention"] = retention
    return result


def _already_in_chat(root: Any, row: dict, *, since_ts: str = "") -> dict | None:
    """The task's earlier publication by summary id and readiness token.

    Row timestamps never limit this lookup: delayed worker rows can be backdated.
    The indexed task probe replaces archive-time bounds; ``since_ts`` remains a
    caller-compatible argument. A miss during import is unknown and keeps the debt.
    """
    from ouroboros import dialogue_store as ds

    task_id = str(row.get("task_id") or "")
    if not task_id:
        raise ValueError("a terminal projection must name its task")
    for entry in ds.iter_probes(root, "chat", [("task_id", task_id)], newest_first=True):
        if (entry.row.get("summary_id") == row["summary_id"]
                and entry.row.get("terminal_projection_token") == row.get("terminal_projection_token")):
            return entry.row
    ds.require_imported(root, "chat")
    return None


def _append_project(root: Any, tid: str, task: dict, event: dict, *,
                    prepare: bool = True, minted_now: bool = False) -> bool:
    from ouroboros import project_dialogue as dialogue

    if prepare:
        stored, minted = _prepare(root, tid, task, event)
        minted_now = minted_now or minted
    else:
        stored = load_task_result(root, tid, strict=True) or {}
    if not stored or not _settled(stored):
        return False
    effective = {**task, **stored}
    is_root = _lineage(tid, effective)["is_root_task"]
    ready = stored.get("canonical_terminal_projection_ready") or {}
    marker = stored.get("canonical_terminal_projection")
    if isinstance(marker, dict) and (not ready or marker.get("token") == ready.get("token")):
        return False
    # The Done card follows the answer: the outbox's own debt holds it (owner Fork 2).
    delivery = _delivery_disposition(root, tid) if is_root else {}
    if is_root and (not ready or _open(stored) or delivery.get("state") == "owed"
                    or not _files_ready(root, tid, effective)):
        return False
    row = _project_row(tid, effective, {key: event[key] for key in ("ts", "chat_id") if key in event}, ready,
                       delivery=delivery)
    appended = False
    # A readiness minted in this very call has no row yet: nothing to scan for.
    existing_row = None if minted_now or not is_root else _already_in_chat(root, row)
    if existing_row:
        row = existing_row  # append/receipt crash: retain actual publication time
    else:
        # A child mints no readiness, so nothing settles it again: its row waits in the retry buffer.
        appended = dialogue.append_canonical_task_summary(root, row, **({} if is_root else {"retained": True}))
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


def retain_main_delivery(root: Any, tid: str, event: dict) -> dict:
    """Keep the exact Main publication and its address in canonical readiness.

    The bounded transport outbox is a cache. This existing result obligation
    owns recovery until the sender confirms the durable chat append.
    """
    stored = load_task_result(root, tid, strict=True) or {}
    ready = stored.get("canonical_terminal_projection_ready")
    if not isinstance(ready, dict):
        return event  # legacy caller; normal terminal settlement has readiness
    saved = ready.get("main_delivery")
    if isinstance(saved, dict) and isinstance(saved.get("event"), dict):
        return completion_event_view(root, dict(saved["event"]), stored)
    retained = {**event, "ts": event.get("ts") or utc_now_iso(), "terminal_projection_token": ready["token"]}

    def retain(current, _patch):
        if current.get("canonical_terminal_projection_ready") != ready:
            return None
        return {"status": current["status"], "canonical_terminal_projection_ready": {
            **ready, "main_delivery": {"event": retained, "attempts": 0}}}

    updated = write_task_result(root, tid, stored["status"], strict_existing_dict=True,
                                _field_projector=retain)
    saved = (updated.get("canonical_terminal_projection_ready") or {}).get("main_delivery")
    if not isinstance(saved, dict) or not isinstance(saved.get("event"), dict):
        raise RuntimeError("Main publication readiness changed before retention")
    return dict(saved["event"])


def claim_main_delivery(root: Any, event: dict) -> bool:
    """Persist retry timing before queue exposure; caps delay, never erases debt.

    This schedule survives outbox eviction, restart and five failed transport
    attempts. A failed queue put also consumes its slot, preventing busy loops.
    """

    token = event.get("terminal_projection_token")
    if not token:
        return True
    tid = str(event.get("task_id") or "")
    claimed = False

    def claim(current, _patch):
        nonlocal claimed
        ready = current.get("canonical_terminal_projection_ready")
        delivery = ready.get("main_delivery") if isinstance(ready, dict) else None
        if (not isinstance(delivery, dict) or ready.get("token") != token
                or delivery.get("event", {}).get("delivery_id") != event.get("delivery_id")):
            return None
        attempts = int(delivery.get("attempts") or 0)
        last = str(delivery.get("last_attempt_at") or "")
        if last:
            stamped = datetime.fromisoformat(last.replace("Z", "+00:00"))
            if stamped.tzinfo is None:
                stamped = stamped.replace(tzinfo=timezone.utc)
            delay = min(3600, 60 * 2 ** min(max(attempts - 1, 0), 6))
            if (datetime.now(timezone.utc) - stamped).total_seconds() < delay:
                return None
        claimed = True
        return {"status": current["status"], "canonical_terminal_projection_ready": {
            **ready, "main_delivery": {**delivery, "attempts": attempts + 1,
                                      "last_attempt_at": utc_now_iso()}}}

    stored = load_task_result(root, tid, strict=True) or {}
    if stored:
        write_task_result(root, tid, stored["status"], strict_existing_dict=True, _field_projector=claim)
    return claimed


def main_delivery_recorded(root: Any, delivery_id: str, *, task_id: Any, publication_token: Any = None) -> bool:
    """Canonical receipt outlives the bounded delivered-id registry; the caller names the task
    (the event's typed ``task_id``), never the outbox parsing an id."""
    if not str(task_id or ""):
        return False
    row = load_task_result(root, str(task_id), strict=True) or {}
    marker = row.get("canonical_terminal_projection")
    return (isinstance(marker, dict) and marker.get("main_disposition") == "delivered"
            and marker.get("main_delivery_id") == delivery_id
            and (publication_token is None or marker.get("token") == publication_token))


def main_delivery_current(root: Any, task_id: str, publication_token: str) -> bool:
    """An obsolete queued attempt cannot publish into a newer logical outcome."""
    row = load_task_result(root, task_id, strict=True) or {}
    ready = row.get("canonical_terminal_projection_ready")
    current = ready if isinstance(ready, dict) else row.get("canonical_terminal_projection")
    return isinstance(current, dict) and current.get("token") == publication_token


def acknowledge_main_delivery(root: Any, delivery_id: str, *, task_id: Any, chat_id: Any = None,
                              publication_token: Any = None) -> None:
    """Retire canonical debt only after the strict sender returned successfully."""
    tid = str(task_id or "")
    if not tid or chat_id not in (None, 1) or not publication_token:
        return
    stored = load_task_result(root, tid, strict=True) or {}
    if not stored:
        return

    def acknowledge(current, _patch):
        ready, marker = current.get("canonical_terminal_projection_ready"), current.get("canonical_terminal_projection")
        delivery = ready.get("main_delivery") if isinstance(ready, dict) else None
        if (not isinstance(delivery, dict) or not isinstance(marker, dict)
                or marker.get("token") != ready.get("token")
                or ready.get("token") != publication_token
                or delivery.get("event", {}).get("delivery_id") != delivery_id):
            return None
        return {"status": current["status"], "canonical_terminal_projection_ready": None,
                "canonical_terminal_projection": {**marker, "main_disposition": "delivered",
                    "main_delivery_id": delivery_id, "main_delivered_at": utc_now_iso(), "main_chat_id": 1}}

    write_task_result(root, tid, stored["status"], strict_existing_dict=True, _field_projector=acknowledge)


def clear_terminal_projection_obligation(root: Any, tid: str, expected: dict, disposition: str) -> bool:
    """CAS the complete readiness token and canonical attempt/checkpoint witness."""
    if disposition != "ineligible":
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


def _deferral(root: Any, tid: str, task: dict | None, stored: dict) -> str:
    """What keeps a recorded readiness from publishing now, '' when nothing does: the
    canonical row's own facts first, the delivery registry only once they all hold."""
    if _open(stored):
        return "synthesis_open"
    if not _settled(stored):
        return "not_settled"
    if not _files_ready(root, tid, {**(task or {}), **stored}):
        return "child_files"
    return "terminal_delivery" if _delivery_disposition(root, tid).get("state") == "owed" else ""


def delivery_registry_stamp(root: Any) -> list | None:
    """The identity of ``supervisor.terminal_delivery``'s registry file, ``None`` when absent: every
    write replaces the file (``update_json_locked``), so an equal stamp is an unchanged registry."""
    try:
        info = (pathlib.Path(root) / "state" / "terminal_deliveries.json").stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return ["unreadable", exc.errno]
    return [info.st_ino, info.st_size, info.st_mtime_ns]


def rest_deferral(rest: Any, task_id: str, waits: dict) -> None:
    """Rest a projection member on the cause ``settle_terminal_projection`` recorded in WAITS
    (``obligations.DutyRest``); a deferral without one (a held lock, an owed Main delivery this
    call re-queued, a failure) is examined again by the next pass."""
    if rest is not None and waits.get("on"):
        rest.wait(task_id, waits["on"], inputs=("terminal_deliveries",) if waits["on"] == "terminal_delivery" else ())


def settle_terminal_projection(drive_root: Any, task_id: str, *, task: dict | None = None,
                               event: dict | None = None, event_queue: Any = None,
                               prepare: bool = True, waits: dict | None = None) -> str:
    """Run both halves of a root's owed terminal projection once its conditions hold.

    ``prepare=False`` is the delivery lanes' word (a delivery acknowledged or given
    up on): readiness is task_done's to mint, so the settle only proceeds when an
    earlier terminal transition already recorded it. ``waits`` receives ``on``, the
    cause of a deferral only a change of the row (or of the delivery registry, for
    ``terminal_delivery``) can lift, for a duty's rest (``rest_deferral``).
    """
    from ouroboros import project_dialogue as dialogue

    from ouroboros.install_identity import StoreForeign, require_task_roots

    tid = str(task_id or "").strip()
    waits = {} if waits is None else waits
    if not tid:
        return SETTLEMENT_NONE
    try:
        with _publication_lock(drive_root, tid) as acquired:
            if not acquired:
                return SETTLEMENT_DEFERRED
            if prepare:
                stored, minted_now = _prepare(drive_root, tid, task or {}, event or {})
            else:
                stored, minted_now = load_task_result(drive_root, tid, strict=True) or {}, False
                require_task_roots({**(task or {}), **stored})
            if (not isinstance(stored.get("canonical_terminal_projection_ready"), dict)
                    or is_reconciled_presence_placeholder(stored)):  # readiness an earlier release recorded
                waits["on"] = "nothing_owed"
                return SETTLEMENT_NONE
            cause = _deferral(drive_root, tid, task, stored)
            if cause:
                waits["on"] = cause
                return SETTLEMENT_DEFERRED
            _append_project(drive_root, tid, task or {}, event or {}, prepare=False, minted_now=minted_now)
            # IO may have advanced canonical state. Render Main from the NEW
            # result, never the task_done snapshot passed to this continuation.
            stored = load_task_result(drive_root, tid, strict=True) or {}
            ready = stored.get("canonical_terminal_projection_ready")
            marker = stored.get("canonical_terminal_projection")
            source_delivery = _delivery_disposition(drive_root, tid)
            if (not isinstance(ready, dict) or not isinstance(marker, dict)
                    or marker.get("token") != ready.get("token") or _open(stored)
                    or not _settled(stored)
                    or source_delivery.get("state") == "owed"
                    or not _files_ready(drive_root, tid, {**(task or {}), **stored})):
                return SETTLEMENT_DEFERRED
            effective = {**(task or {}), **stored, "id": tid}
            done = {**(event or {}), **stored, "ts": ready["task_done_ts"], "chat_id": ready["chat_id"]}
            from ouroboros.task_finalization import quiet_finish

            if quiet_finish(stored):
                disposition = dialogue.MAIN_MIRROR_INELIGIBLE
            else:
                disposition, _queued = dialogue.project_completion_delivery_outcome(
                    drive_root, done, tid, effective, stored, done, event_queue=event_queue,
                    source_delivery=source_delivery)
            if clear_terminal_projection_obligation(drive_root, tid, stored, disposition):
                return SETTLEMENT_SETTLED
            # Owed remains discoverable until the strict sender acknowledges it.
            if main_delivery_recorded(drive_root, f"project-completion:{tid}", task_id=tid):
                return SETTLEMENT_SETTLED
            return SETTLEMENT_DEFERRED
    except StoreForeign:
        raise
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
    unknown bytes via a tolerant scan. A member whose deferral has a recorded cause
    rests (``rest_deferral``) until its row or the delivery registry changes.
    """
    from ouroboros.obligations import DutyRest, result_rows
    from ouroboros.install_identity import StoreForeign, rest_foreign_member

    rest = DutyRest(drive_root, "terminal_projection", name="terminal_projection",
                    inputs={"terminal_deliveries": delivery_registry_stamp(drive_root)})
    settled = 0
    for row in result_rows(drive_root, "terminal_projection", ids=rest.examine):
        waits = {"on": "nothing_owed"}
        try:
            if terminal_projection_owed(row["task_id"], row):
                waits = {}
                settled += settle_terminal_projection(drive_root, row["task_id"], waits=waits) == SETTLEMENT_SETTLED
        except StoreForeign as exc:
            rest_foreign_member(rest, row["task_id"], row, exc)
            continue
        except Exception:
            waits = {}
            log.warning("Terminal projection reconciliation deferred for %s", row["task_id"], exc_info=True)
        rest_deferral(rest, row["task_id"], waits)
    rest.commit()
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

    Copied, never interpreted: ``outcome_final``, ``outcome_authority`` and ``reason_code``
    as the source holds them, ``objective_status`` from its ``outcome_axes``; the axes
    themselves stay with their source (one axis can be kilobytes of policy denials).
    """
    entry = {"task_id": task_id, "status": str(row.get("status") or ""), "outcome": str(row.get("outcome") or ""),
             "outcome_phase": str(row.get("outcome_phase") or ""), "source": source,
             "result_ref": row.get("result_ref") or {"kind": "task_result", "task_id": task_id,
                                                      "reader": "get_task_result"}}
    if row.get("reason_detail"):
        entry["review_verdict"] = str(row["reason_detail"])
    if isinstance(row.get("outcome_final"), bool):
        entry["outcome_final"] = row["outcome_final"]
    if isinstance(row.get("outcome_authority"), str) and row["outcome_authority"].strip():
        entry["outcome_authority"] = row["outcome_authority"].strip()
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
    except (OSError, ValueError, StoreError):
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
