"""Notice receipts addressed by notice id, written by chat publication.

The lifecycle import admits already imported upgrade notices. A missing receipt
is checked against its chat in the dialogue store; pending archive import is
unknown and defers the notice, without blocking the lifecycle import. New
publication stores the receipt immediately after the canonical append, before
any state-marker write or task-result acknowledgement. An append/receipt crash
remains at-least-once (an extra notice, never a loss). A pause-notice receipt
lives only from its chat append to the task's acknowledgement
(``ouroboros/pause_notices.py``).
"""
import logging

from ouroboros import obligations as o

log = logging.getLogger(__name__)

UPGRADE_TYPES = frozenset({"reviewer_default_notice", "optional_bounds_notice", "legacy_memory_notice"})
PAUSE_TYPE = "task_pause_notice"


def notice_id(chat_id, notice_type):
    return f"{int(chat_id)}:{notice_type}"


def recorded(root, chat_id, notice_type):
    if o.members(root, "upgrade_notices").get(notice_id(chat_id, notice_type), {}).get("recorded") is True:
        return True
    from ouroboros import dialogue_store as ds

    # Historical receipts can arrive after the lifecycle store was published: the notice's own
    # rows (its type is an indexed fact), never a walk of the owner's whole chat.
    for entry in ds.iter_probes(root, "chat", [("kind", notice_type)], newest_first=True):
        row = entry.row
        if (row.get("direction") == "system" and row.get("type") == notice_type
                and isinstance(row.get("chat_id"), int) and row["chat_id"] == int(chat_id)):
            return True
    ds.require_imported(root, "chat")
    return False


def pause_receipt_id(chat_id, identity):
    return f"{int(chat_id)}:{identity}"


def record(root, row):
    """After the chat row landed: a receipt that cannot be written never turns the landed row into a
    failure (the notice may repeat once; the next import re-reads the receipts)."""
    kind = row.get("type")
    if row.get("direction") != "system":
        return
    try:
        if kind in UPGRADE_TYPES and isinstance(row.get("chat_id"), int):
            o.add(root, "upgrade_notices", notice_id(row["chat_id"], kind),
                  {"chat_id": row["chat_id"], "type": kind, "ts": row.get("ts"), "recorded": True})
        elif kind == PAUSE_TYPE and row.get("card_row_id") and isinstance(row.get("chat_id"), int):
            o.add(root, "pause_notice_receipts", pause_receipt_id(row["chat_id"], row["card_row_id"]),
                  {"task_id": row.get("task_id"), "ts": row.get("ts")})
    except Exception:
        log.warning("Notice receipt for %s could not be recorded", kind, exc_info=True)


def import_upgrade_receipts(root):
    """Admit positive receipts only; ``recorded`` owns the coverage check on a miss."""
    from ouroboros import dialogue_store as ds

    records = {}
    for entry in ds.iter_probes(root, "chat", [("kind", kind) for kind in sorted(UPGRADE_TYPES)]):
        row = entry.row
        kind = row.get("type")
        # One malformed historical row is skipped, never a failed import.
        if row.get("direction") == "system" and kind in UPGRADE_TYPES and isinstance(row.get("chat_id"), int):
            records[notice_id(row["chat_id"], kind)] = {"chat_id": row["chat_id"], "type": kind,
                                                      "ts": row.get("ts"), "recorded": True}
    return records
