"""Notice receipts addressed by notice id, written by chat publication.

The lifecycle import reads historical chat once for upgrade notices. New
publication stores the receipt immediately after the canonical append, before
any state-marker write or task-result acknowledgement. An append/receipt crash
remains at-least-once (an extra notice, never a loss). A pause-notice receipt
lives only from its chat append to the task's acknowledgement
(``ouroboros/pause_notices.py``).
"""
from ouroboros import obligations as o

UPGRADE_TYPES = frozenset({"reviewer_default_notice", "optional_bounds_notice", "legacy_memory_notice"})
PAUSE_TYPE = "task_pause_notice"


def notice_id(chat_id, notice_type):
    return f"{int(chat_id)}:{notice_type}"


def recorded(root, chat_id, notice_type):
    return o.members(root, "upgrade_notices").get(notice_id(chat_id, notice_type), {}).get("recorded") is True


def pause_receipt_id(chat_id, identity):
    return f"{int(chat_id)}:{identity}"


def record(root, row):
    """After the chat row landed: a receipt that cannot be written never turns the landed row into a
    failure (the notice may repeat once; the next start rebuilds the sets)."""
    kind = row.get("type")
    if row.get("direction") != "system":
        return
    if kind in UPGRADE_TYPES and isinstance(row.get("chat_id"), int):
        o._bookkeeping(root, "upgrade notice receipt", o.add, root, "upgrade_notices", notice_id(row["chat_id"], kind),
                       {"chat_id": row["chat_id"], "type": kind, "ts": row.get("ts"), "recorded": True})
    elif kind == PAUSE_TYPE and row.get("card_row_id") and isinstance(row.get("chat_id"), int):
        o._bookkeeping(root, "pause notice receipt", o.add, root, "pause_notice_receipts",
                       pause_receipt_id(row["chat_id"], row["card_row_id"]),
                       {"task_id": row.get("task_id"), "ts": row.get("ts")})


def import_upgrade_receipts(root):
    from pathlib import Path
    from ouroboros.utils import jsonl_chain_handles, iter_jsonl_objects
    records = {}
    with jsonl_chain_handles(Path(root) / "logs/chat.jsonl", strict=True) as handles:
        rows = [row for path, handle in handles for row in iter_jsonl_objects(path, _handle=handle)]
    for row in rows:
        kind = row.get("type")
        # One malformed historical row is skipped, never a failed import.
        if row.get("direction") == "system" and kind in UPGRADE_TYPES and isinstance(row.get("chat_id"), int):
            records[notice_id(row["chat_id"], kind)] = {"chat_id": row["chat_id"], "type": kind,
                                                      "ts": row.get("ts"), "recorded": True}
    return records
