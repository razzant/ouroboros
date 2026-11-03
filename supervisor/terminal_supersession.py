"""A delivered Stop receipt supersedes its task's older host notices in that chat (#1276).

Moved from ``supervisor/terminal_delivery.py`` at its module ceiling. The outbox still decides under
its own registry lock, reaching the shared seams through the ``terminal_delivery`` module object at
call time. Two writes supersede: ``register_delivery`` in the write that retires the Stop receipt's
own owed row, and the replay sweep for a Stop receipt recorded without that row (the previous
writer's, or one recorded after its row left), proven from what the task result retains. Both
supersede only a notice the send handler would route now to the chat the receipt reached
(``_routed_chat``). The send handler reads ``superseded_by``.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Tuple

from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE

log = logging.getLogger(__name__)

# Outbox bookkeeping on an owed row; the rest is the event as it was owed.
_BOOKKEEPING = ("registered_at", "replay_attempts", "dispatch_attempts", "last_replay_at",
                "delivery_deferred", "transport_failed")


def _td():
    from supervisor import terminal_delivery

    return terminal_delivery


def superseded_by(drive_root: Any, delivery_id: str) -> str:
    """The delivered Stop receipt that replaced this host notice, '' when none; unreadable fails open ('')."""
    td = _td()
    try:
        row = td._mapping_rows(td._read_registry(drive_root), "superseded").get(str(delivery_id or ""))
    except Exception:
        log.debug("terminal-delivery registry read failed for %s", delivery_id, exc_info=True)
        return ""
    return str(row.get("superseded_by") or "") if isinstance(row, dict) else ""


def _replaceable(did: str, row: Any, task_id: str) -> bool:
    """The task's exact ``final:`` row stamped ``host_notice``, with no recorded send and no Main publication."""
    return (isinstance(row, dict) and bool(task_id) and did.startswith(f"final:{task_id}:")
            and _td()._TERMINAL_ANSWER_ID.fullmatch(did) is not None and str(row.get("task_id")) == task_id
            and row.get("terminal_origin") == TERMINAL_ORIGIN_HOST_NOTICE
            and not row.get("emitted") and not row.get("terminal_projection_token"))


def _routed_chat(drive_root: Any, row: Dict[str, Any]) -> str:
    """The chat the send handler routes this owed row to now (``events_chat_delivery._send_message``):
    a room-fixed type keeps its own, any other goes to its task tree's current Project room (the shared
    resolver, with the row's lineage) before the chat it was born in."""
    from ouroboros.project_dialogue import ROOM_FIXED_ROW_TYPES
    from supervisor.log_addressing import resolve_project_chat

    meta = row.get("progress_meta") if isinstance(row.get("progress_meta"), dict) else {}
    room = 0 if str(row.get("system_type") or "") in ROOM_FIXED_ROW_TYPES else resolve_project_chat(
        drive_root, row.get("task_id"), meta.get("parent_task_id") or row.get("parent_task_id"),
        meta.get("root_task_id") or row.get("root_task_id"))
    return str(room or row.get("chat_id"))


def _supersede(notice: str, stop_id: str, rows: List[str], pending: Dict[str, Any],
               exhausted: Dict[str, Any], superseded: Dict[str, Any]) -> Dict[str, Any]:
    """Move one owed notice to ``superseded`` as it was owed (payload and send facts) and owe what it bundled.

    The custody notice and Wrap up receipt it carries are facts of their own: the rows the send handler
    splits off (``bundled_rows``) are owed under their own ids in the same write, unless already owed,
    recorded or given up."""
    from supervisor.events_chat_delivery import bundled_rows

    td, row = _td(), pending.pop(notice)
    superseded[notice] = {**row, "superseded_by": stop_id, "superseded_at": td.utc_now_iso()}
    event = {key: value for key, value in row.items() if key not in _BOOKKEEPING}
    for extra in bundled_rows({**event, "delivery_id": notice})[1]:
        rid = str(extra.get("delivery_id") or "")
        if rid and rid not in rows and not any(rid in marks for marks in (pending, exhausted, superseded)):
            pending[rid] = td._owed_row(extra)
    return {**superseded[notice], "delivery_id": notice}


def supersede_host_notices(drive_root: Any, did: str, owed: Dict[str, Any], emitted: Any, rows: List[str],
                           pending: Dict[str, Any], exhausted: Dict[str, Any],
                           superseded: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Move what a delivered Stop receipt (``cancel:``/``cascade:``) replaced from ``pending`` to ``superseded``.

    Replaced are its task's exact ``final:`` rows stamped ``host_notice``, owed since before the receipt
    was, that the send handler would route now to the chat it reached (``_routed_chat``: a Project room the
    task became after the Stop never showed it). Whatever earlier sends did (refused, never returned,
    unknown), the row keeps its payload and send facts, gains ``superseded_by`` and is never sent again. A
    recorded ``emitted`` send (receipt repair) and Main's debt stay owed. A send already begun may still
    land; its real receipt retires the mark."""
    td = _td()
    stop, sent = td.TERMINAL_MESSAGE_ID.fullmatch(did), emitted if isinstance(emitted, dict) else {}
    tid = stop.group("task") if stop is not None and not did.startswith("final:") else ""
    since = td._instant(owed.get("registered_at"))
    if not tid or since is None or sent.get("system_type") != "cancel_receipt" or str(sent.get("task_id")) != tid:
        return []
    moved: List[Dict[str, Any]] = []
    for notice, row in list(pending.items()):
        stamped = td._instant(row.get("registered_at")) if isinstance(row, dict) else None
        if (stamped is not None and stamped < since and notice not in rows and _replaceable(notice, row, tid)
                and _routed_chat(drive_root, row) == str(sent.get("chat_id"))):
            moved.append(_supersede(notice, did, rows, pending, exhausted, superseded))
    return moved


def _retained_proof(drive_root: Any, task_id: str) -> Tuple[str, Any, Any]:
    """``(stop id, chat it reached, Stop request instant)`` from the task result's ``cancel_receipt``;
    ``("", None, None)`` when any of them is not recorded there or cannot be read."""
    from ouroboros.task_results import load_task_result

    td = _td()
    try:
        receipt = (load_task_result(pathlib.Path(drive_root), task_id) or {}).get("cancel_receipt")
    except Exception:
        log.debug("retained Stop receipt of %s unreadable; its notices stay owed", task_id, exc_info=True)
        receipt = None
    stop = str(receipt.get("delivery_id") or "") if isinstance(receipt, dict) else ""
    match, since = td.TERMINAL_MESSAGE_ID.fullmatch(stop), None
    if match is not None and match.group("task") == task_id and not stop.startswith("final:"):
        since = td._instant(receipt.get("stop_requested_at"))
    if since is None or receipt.get("delivered_chat_id") is None:
        return "", None, None
    return stop, receipt["delivered_chat_id"], since


def supersede_recorded_stops(drive_root: Any, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Before any dispatch, supersede the notices a Stop receipt replaced when it was recorded without its owed row.

    Nothing of that receipt is owed any more, so the proof is what its task result retains: the
    ``cancel_receipt`` names the Stop id (which the outbox records as delivered), the chat it reached,
    which must be where the notice is routed now (``_routed_chat``), and when the Stop was requested,
    which the notice's owing must precede. A missing, unreadable or different fact keeps the notice
    owed. Reads the current owed rows and one result per task with such a notice; the registry lock
    rechecks every row. Returns the rows still owed."""
    td, proofs, proven = _td(), {}, {}
    for row in rows:
        did, tid = row["delivery_id"], str(row.get("task_id") or "")
        if not _replaceable(did, row, tid):
            continue
        if tid not in proofs:
            proofs[tid] = _retained_proof(drive_root, tid)
        stop, chat, since = proofs[tid]
        stamped = td._instant(row.get("registered_at"))
        if stop and stamped is not None and stamped < since and str(chat) == _routed_chat(drive_root, row):
            proven[did] = (stop, row)
    if not proven:
        return rows
    moved: List[Dict[str, Any]] = []
    evicted: List[Dict[str, Any]] = []

    def _mutate(current: Dict[str, Any]) -> Any:
        moved.clear()
        evicted.clear()
        delivered = td._delivered_rows(current, strict=True)
        pending = td._mapping_rows(current, "pending", strict=True)
        exhausted, superseded = (td._mapping_rows(current, key, strict=True) for key in ("exhausted", "superseded"))
        for notice, (stop, read) in proven.items():
            row = pending.get(notice)
            if (stop in delivered and notice not in delivered and isinstance(row, dict)
                    and {**row, "delivery_id": notice} == read):
                moved.append(_supersede(notice, stop, delivered, pending, exhausted, superseded))
        if not moved:
            return None
        evicted.extend(td._evict_over_cap(pending, exhausted))
        return td._registry_document(current, delivered, pending, exhausted=exhausted, superseded=superseded)

    try:
        td.update_json_locked(td._registry_path(drive_root), _mutate, strict_existing_dict=True,
                              timeout_sec=td.bounded_wait(4.0))
    except ValueError:
        td._log_registry_corrupt(drive_root, ",".join(proven), op="supersede_recorded_stops")
        return rows
    except Exception:
        log.warning("superseding notices of recorded Stop receipts deferred to the next sweep", exc_info=True)
        return rows
    disclose(drive_root, moved, evicted, basis="task_result_cancel_receipt")
    gone = {row["delivery_id"] for row in moved + evicted}
    return [row for row in rows if row["delivery_id"] not in gone]


def disclose(drive_root: Any, moved: List[Dict[str, Any]], evicted: List[Dict[str, Any]], *, basis: str = "") -> None:
    """After the write: one typed event per superseded notice; capacity evictions take their usual disclosure."""
    td = _td()
    for row in moved:
        td._record_event(drive_root, {"type": "terminal_delivery_superseded", "task_id": row.get("task_id"),
                                      "delivery_id": row["delivery_id"], "superseded_by": row["superseded_by"],
                                      "chat_id": row.get("chat_id"), **({"basis": basis} if basis else {})})
    for row in evicted:
        td._disclose_exhausted_delivery(drive_root, row, reason="outbox_capacity")
