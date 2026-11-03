"""Resolve owner chat/quiz/mailbox membership; action meaning belongs to Main.

No skill permissions or lifecycle grants are inferred here. Callers validate their
actual actor and selected target before interpreting this exact source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def answered_quiz_block(drive_root: Any, task_id: str, quiz_id: Any) -> dict[str, Any]:
    """The answered card ``quiz_id`` of ``task_id`` exactly as its projection recorded it, else ``{}``.

    Scope is the caller's: ``resolve_owner_source`` admits only the calling task's own
    card; Continue reads only the card identity the producer carried into a start.
    """
    from ouroboros.owner_quiz import STATE_ANSWERED, quiz_states

    row = quiz_states(drive_root, task_id).get(str(quiz_id or ""), {})
    if row.get("state") != STATE_ANSWERED or not row.get("request_id"):
        return {}
    index = row.get("answered_index")
    options = row.get("options") or []
    label = options[index] if type(index) is int and 0 <= index < len(options) else ""
    return row if label or str(row.get("comment") or "").strip() else {}


def resolve_owner_source(ctx: Any, source: Any) -> dict[str, Any]:
    """Resolve an actual owner source belonging to this task's conversation."""
    from ouroboros.tool_access import canonical_data_root

    if not isinstance(source, dict):
        return {}
    drive = canonical_data_root(ctx)
    task_id = str(getattr(ctx, "task_id", "") or "")
    kind = source.get("kind")
    if kind == "chat":
        from ouroboros.project_dialogue import build_owner_message_ref, owner_message_ref_is_valid, resolve_owner_message_source
        from ouroboros.task_status import load_effective_task_result

        ref = source.get("ref")
        if not owner_message_ref_is_valid(ref):
            return {}
        task = load_effective_task_result(drive, task_id, materialize_artifacts=False) if task_id else {}
        chat_id = getattr(ctx, "current_chat_id", None)
        if ref.get("chat_id") != chat_id and ref != (task or {}).get("origin_message_ref"):
            return {}
        row = resolve_owner_message_source(drive, ref)
        if not row or row.get("direction") != "in" or not str(row.get("text") or "").strip():
            return {}
        # System/skill injections cannot become owner authority merely by
        # retaining an inbound-looking row or a copied owner reference.
        if row.get("system_type") or row.get("presence") or row.get("source") == "skill_repair":
            return {}
        # Canonicalize selectors to the actual row so omitting a client id or
        # adding caller fields cannot turn one owner message into two grants.
        actual_ref = build_owner_message_ref(chat_id=row.get("chat_id", 1),
            client_message_id=row.get("client_message_id"), ts=row["ts"], text=row["text"])
        return {"kind": kind, "ref": actual_ref, "ts": row["ts"], "text": row["text"],
                "attachments": list(row.get("attachments") or [])}
    if not task_id or source.get("task_id") != task_id:
        return {}
    if kind == "quiz":
        from ouroboros.owner_quiz import late_answer_owner_text, recorded_answer_frame

        row = answered_quiz_block(drive, task_id, source.get("quiz_id"))
        if not row:
            return {}
        return {"kind": kind, "task_id": task_id, "quiz_id": row["quiz_id"],
                "request_id": row["request_id"], "ts": row.get("answered_at"), "text": late_answer_owner_text(row),
                "question_frame": recorded_answer_frame(row)}
    if kind == "mailbox":
        from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries

        roots = dict.fromkeys((Path(ctx.drive_root), drive))
        for root in roots:
            for row in drain_owner_entries(root, task_id, include_acknowledged=True):
                if row.get("msg_id") == source.get("msg_id") and row.get("kind") == KIND_OWNER_TEXT:
                    return {"kind": kind, "task_id": task_id, "msg_id": row["msg_id"],
                            "ts": row["ts"], "text": row["text"]}
    return {}
