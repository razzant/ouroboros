"""Resolve owner chat/quiz/mailbox membership; action meaning belongs to Main.

No skill permissions or lifecycle grants are inferred here. Callers validate their
actual actor and selected target before interpreting this exact source.
"""
from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Any, Optional


def _valid_answered_quiz_block(row: Any) -> dict[str, Any]:
    """One answered card record if it carries an accepted answer and its asking request id, else ``{}``."""
    from ouroboros.chat_uploads import answer_evidence
    from ouroboros.owner_quiz import STATE_ANSWERED

    if not isinstance(row, dict) or row.get("state") != STATE_ANSWERED or not row.get("request_id"):
        return {}
    index = row.get("answered_index")
    options = row.get("options") or []
    label = options[index] if type(index) is int and 0 <= index < len(options) else ""
    # A Reply's file alone answers too (its stored upload or why none arrived); an empty entry does not.
    return row if label or str(row.get("comment") or "").strip() or answer_evidence(row.get("attachments")) else {}


def answered_quiz_block(drive_root: Any, task_id: str, quiz_id: Any) -> dict[str, Any]:
    """The answered card ``quiz_id`` of ``task_id`` exactly as its projection recorded it, else ``{}``.

    Scope is the caller's: ``resolve_owner_source`` admits only the calling task's own
    card; Continue reads only the card identity the producer carried into a start.
    """
    from ouroboros.owner_quiz import quiz_states

    return _valid_answered_quiz_block(quiz_states(drive_root, task_id).get(str(quiz_id or ""), {}))


def _recorded_answered_quiz(drive_root: pathlib.Path, task_id: str, quiz_id: str) -> tuple[dict[str, Any], Optional[dict]]:
    """The canonical ``quiz_answer`` history row of an evicted ordinary card, without borrowing ``delivery``."""
    from ouroboros import dialogue_store as ds
    from ouroboros.owner_quiz import STATE_ANSWERED
    from ouroboros.question_answers import answer_message_id

    source_id = answer_message_id(task_id, quiz_id)
    candidates = [
        entry.row for entry in ds.probe_rows(drive_root, "chat", [("client_message_id", source_id)], limit=8)
        if isinstance(entry.row, dict) and entry.row.get("client_message_id") == source_id
    ]
    if len(candidates) != 1:
        return {}, None
    row = candidates[0]
    kept = row.get("quiz")
    if (row.get("type") == "quiz_answer" and row.get("source") == "owner_quiz_answer"
            and row.get("direction") == "system" and row.get("task_id") == task_id
            and isinstance(kept, dict) and kept.get("quiz_id") == quiz_id
            and kept.get("state") == STATE_ANSWERED
            and not (kept.get("secret_request") or kept.get("secret_outcome"))):
        cleaned = _valid_answered_quiz_block({key: value for key, value in kept.items() if key != "delivery"})
        if cleaned:
            return cleaned, row
    return {}, None


def _recorded_row(drive: Any, row: Any, recorded: dict[str, Any]) -> bool:
    """Whether ``row`` is physically the origin a task's start recorded: by its digest, or, for an origin recorded
    before digests, by the one readable original that older reference resolves to (an ambiguous one names none)."""
    from ouroboros.chat_chain import source_row_id
    from ouroboros.project_dialogue import resolve_owner_message_source

    if not row:
        return False
    digest = recorded.get("row_sha256")
    if digest is None:
        original = resolve_owner_message_source(drive, recorded)
        digest = source_row_id(original) if original else None
    return digest is not None and digest == source_row_id(row)


def resolve_owner_source(ctx: Any, source: Any) -> dict[str, Any]:
    """Resolve an actual owner source belonging to this task's conversation."""
    from ouroboros.tool_access import canonical_data_root

    if not isinstance(source, dict):
        return {}
    drive = canonical_data_root(ctx)
    task_id = str(getattr(ctx, "task_id", "") or "")
    kind = source.get("kind")
    if kind == "chat":
        from ouroboros.chat_chain import _row_chat_id
        from ouroboros.project_dialogue import build_owner_message_ref, owner_message_ref_is_valid, resolve_owner_message_source
        from ouroboros.task_status import load_effective_task_result

        ref = source.get("ref")
        if not owner_message_ref_is_valid(ref):
            return {}
        task = load_effective_task_result(drive, task_id, materialize_artifacts=False) if task_id else {}
        chat_id = getattr(ctx, "current_chat_id", None)
        recorded = (task or {}).get("origin_message_ref")
        if ref.get("chat_id") != chat_id and not owner_message_ref_is_valid(recorded):
            return {}
        row = resolve_owner_message_source(drive, ref)
        # Another room's message is this task's own only as the physical row its start recorded, however many of
        # its fields the selector names: one readable original each, compared by identity, never by dictionary.
        if ref.get("chat_id") != chat_id and not _recorded_row(drive, row, recorded):
            return {}
        # System/skill injections and a place participant cannot become owner authority merely by
        # retaining an inbound-looking row or a copied owner reference (``owner_words.owner_row``).
        from ouroboros.owner_words import owner_row

        if not owner_row(row, drive):
            return {}
        # Canonicalize selectors to the actual row so omitting a client id or
        # adding caller fields cannot turn one owner message into two grants.
        actual_ref = build_owner_message_ref(chat_id=_row_chat_id(row),
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
                "question_frame": recorded_answer_frame(row, asker=task_id, reader=task_id)}  # its own card
    if kind == "mailbox":
        from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries

        roots = dict.fromkeys((Path(ctx.drive_root), drive))
        for root in roots:
            for row in drain_owner_entries(root, task_id, include_acknowledged=True):
                if row.get("msg_id") == source.get("msg_id") and row.get("kind") == KIND_OWNER_TEXT:
                    return {"kind": kind, "task_id": task_id, "msg_id": row["msg_id"],
                            "ts": row["ts"], "text": row["text"]}
    return {}


def _retained_start_quote(root: pathlib.Path, drafter: str, address: str) -> Optional[dict]:
    """A drafter's exact addressed relay can outlive the room row, including a quote made from a carried field."""
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_relay import KIND_RELAY, owner_row_address
    from ouroboros.task_custody import own_child_drives
    from ouroboros.task_store import read_document

    held = ((read_document(root, drafter) or {}).get("held_owner_messages") or {}).get(address)
    if isinstance(held, dict) and owner_row_address(held) == address:
        return held
    for drive in [root, *own_child_drives(root, drafter)]:
        for entry in drain_owner_entries(drive, drafter, include_acknowledged=True, kinds={KIND_RELAY}):
            if not entry.get("host_carried"):
                for quoted in entry.get("owner_rows") or []:
                    if isinstance(quoted, dict) and owner_row_address(quoted) == address:
                        return quoted
    return None


def start_answer_source(card: dict, row: dict) -> Optional[str]:
    """The type the accepted answer proves, including any citation carried on the answer."""
    from ouroboros.answer_facts import OWNER_PRESS, answer_source
    from ouroboros.chronicle_quotes import quote_in
    from ouroboros.owner_quiz import late_answer_owner_text

    words = row["content"]
    if late_answer_owner_text(card) != words:
        return None
    source = answer_source(card)
    if "cited" in row or row.get("basis") == "cited":
        cite = row.get("cited")
        if not isinstance(cite, str) or not cite.strip() or row.get("basis") != "cited":
            return None
        if source == OWNER_PRESS:
            if cite.strip() != str(card["options"][card["answered_index"]]).strip():
                return None
        elif not quote_in(cite, words):
            return None
    return source


def start_answer(root: pathlib.Path, task_id: str, row: dict, drafter: str) -> Optional[dict]:
    """Read a relay by its exact chat address, or a direct answer by its asker's delivery."""
    from ouroboros.answer_facts import answer_source, answered_card, answered_card_reply, row_source
    from ouroboros.chat_chain import resolve_row
    from ouroboros.chat_uploads import answer_evidence
    from ouroboros.owner_relay import owner_row_identity, quotable_owner_row
    from ouroboros.owner_words import answered_card_source
    from ouroboros.question_answers import answer_message_id
    from ouroboros.store import StoreError

    asker, quiz_id, words, msg_id = (row.get(key) for key in ("source_task_id", "quiz_id", "content", "msg_id"))
    if not isinstance(msg_id, str) or not isinstance(words, str):
        return None
    physical = row.get("row_sha256")
    if msg_id.startswith("row:"):
        source_root = row.get("relay_source_root") or root
        try:
            quoted, _ = resolve_row(source_root, msg_id)
            quoted = quoted or _retained_start_quote(root, drafter, msg_id)
        except (StoreError, OSError, ValueError, TypeError):
            return None
        if not quoted or not quotable_owner_row(quoted, source_root) or row_source(quoted, "owner_mailbox", source_root) != row["source"]:
            return None
        if physical and physical != owner_row_identity(quoted)["row_sha256"]:
            return None
        block, asker = answered_card_source(quoted, source_root)
        quiz_id = (block or {}).get("quiz_id")
        if not block and isinstance(carried := quoted.get("corpus_answer"), dict):
            asker, quiz_id = carried.get("source_task_id"), carried.get("quiz_id")
            block = answered_quiz_block(source_root, asker, quiz_id) if asker and quiz_id else {}
            if (not block or answer_source(block) != row["source"]
                    or answered_card_reply(block, asker) != carried.get("reply_to")):
                return None
    else:
        if not all(isinstance(x, str) and x.strip() for x in (asker, quiz_id, msg_id)) or asker == task_id:
            return None
        from ouroboros.task_store import read_document

        try:
            document = read_document(root, asker)
        except (StoreError, OSError, ValueError, TypeError):
            return None
        if not isinstance(document, dict) or not isinstance(quizzes := document.get("owner_quiz"), dict):
            return None
        if quiz_id in quizzes:
            block = _valid_answered_quiz_block(quizzes.get(quiz_id))
            if not block:
                return None
            delivery = block.get("delivery") if isinstance(block.get("delivery"), dict) else {}
            if msg_id not in (answer_message_id(asker, quiz_id), delivery.get("msg_id")):
                return None
            if physical not in (None, ""):
                answer_row = delivery.get("answer_row")
                if (not isinstance(answer_row, dict) or not answer_row
                        or physical != owner_row_identity(answer_row)["row_sha256"]):
                    return None
        else:
            try:
                block, history_row = _recorded_answered_quiz(root, asker, quiz_id)
            except (StoreError, OSError, ValueError, TypeError):
                return None
            if not block or msg_id != answer_message_id(asker, quiz_id):
                return None
            if physical not in (None, "") and (not isinstance(history_row, dict)
                                                or physical != owner_row_identity(history_row)["row_sha256"]):
                return None
    card = answered_card(block, words)
    if (not card or not asker or not quiz_id or row.get("source_task_id", asker) != asker
            or row.get("quiz_id", quiz_id) != quiz_id
            or "reply_to" in row and row["reply_to"] != answered_card_reply(block, asker)):
        return None
    if (source := start_answer_source(card, row)) is None:
        return None
    files = answer_evidence(block.get("attachments"))  # the winner's own record, never the carried row's refs
    return {"source": source, "content": words, "msg_id": msg_id, "source_task_id": asker,
            **({"attachments": files} if files else {}),
            **{key: row[key] for key in ("basis", "cited", "row_sha256", "relay_source_root") if row.get(key)},
            "quiz_id": quiz_id, "card": card, **({"request_id": block["request_id"]} if block.get("request_id") else {})}


def start_citations(rows: list) -> tuple[list, dict]:
    """Fold an older citation's display address into the unique native quiz row it names.
    The native row still has to prove its delivery or relay in start_answer."""
    from ouroboros.chronicle_quotes import quote_in
    from ouroboros.owner_words import row_ref

    rows = [dict(row) if isinstance(row, dict) else row for row in rows]
    native = [row for row in rows if isinstance(row, dict) and row.get("source") == "owner_quiz_answer"
              and row.get("source_task_id") and row.get("quiz_id") and isinstance(row.get("msg_id"), str)]
    kept, aliases = [], {}
    for row in rows:
        if (isinstance(row, dict) and row.get("source") == "owner_quiz_answer" and row.get("basis") == "cited"
                and row.get("carried") is True and isinstance(row.get("cited"), str) and row["cited"].strip()
                and isinstance(row.get("content"), str) and quote_in(row["cited"], row["content"])
                and set(row) <= {"source", "content", "msg_id", "cited", "basis", "carried"}):
            matches = [item for item in native if row_ref(item) == row.get("msg_id")]
            if len(matches) == 1 and matches[0].get("content") == row["content"]:
                matches[0].update(basis="cited", cited=row["cited"])
                aliases[row["msg_id"]] = (row["content"], True)
                continue
        kept.append(row)
    return kept, aliases
