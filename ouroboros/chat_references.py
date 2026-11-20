"""Host-owned message addresses and Reply facts, shared by every owner ingress.

References describe what the sender pointed at. They do not select work or grant
authority. Hash the persisted row before adding any history/view fields. A row the
dialogue store cannot read leaves an ordinary Reply unresolved. A selected passage
requires an exact source at acceptance and remains a sender-reported citation.
"""
from __future__ import annotations

from ouroboros.chat_chain import format_address, parse_address, resolve_row, row_address
from ouroboros.chat_uploads import web_attachment_caption
from ouroboros.store import StoreError


def _card_display(root, row: dict) -> dict:
    """A late card's displayed text may describe an act or combine its label and submitted fields."""
    if row.get("direction") == "in" and (row.get("card_press") or row.get("late_answer")):
        from ouroboros.answer_facts import card_input

        return card_input(row, root)
    return {}


def _card_display_only(row: dict, card: dict) -> bool:
    # A quiz comment occupies its own bubble. A living form's bubble mixes the title/labels and values.
    return bool(card and (row.get("card_press") or card["source"] != "owner_mailbox"))


def card_reply_display(root, row: dict) -> str:
    """Host context for a whole-row Reply to a mixed or wordless card display, not the owner's speech."""
    card = _card_display(root, row)
    if not _card_display_only(row, card):
        return ""
    from ouroboros.answer_facts import press_record
    from ouroboros.owner_words import OWNER_PRESS

    if card["source"] == OWNER_PRESS:
        return press_record(card["content"], card.get("source_task_id"), who="the owner")
    if card["source"] == "owner_mailbox":
        return f"The owner submitted values on a card of task {card['source_task_id']}:\n{card['content']}"
    return card["content"]


def projection_fields(row: dict, root=None) -> dict:
    """Additive fields for live frames and history, from the same stored row.

    An outgoing chat row saved under its own name (a relay's ``relay:<id>:<task>``) carries that name as
    ``row_client_message_id``: on a live assistant/system frame ``client_message_id`` names the owner's
    submission the row answers, so the row's own name travels under a field of its own. A progress
    record is not named: it merges its frame's meta, owner correlation included.
    """
    fields = {key: row[key] for key in ("reply_to", "relay", "room_revisions") if isinstance(row.get(key), dict)}
    if row.get("reply_to_relay"):
        fields["reply_to_relay"] = row["reply_to_relay"]
    # A host-fact ingress row (a call's opening) is the host's own row named by that id, as history keys it.
    if (row.get("direction") != "in" or isinstance(row.get("host_fact"), dict)) and not row.get("is_progress") \
            and row.get("client_message_id"):
        fields["row_client_message_id"] = str(row["client_message_id"])
    fields["row_address"] = format_address(row_address(row))
    if _card_display_only(row, _card_display(root, row)):
        fields["reply_excerpt_disabled"] = True
    return fields


def _snippet(text) -> str:
    """A Reply's one-line preview: at most 160 characters, a longer text visibly cut; the address reads it all."""
    words = " ".join(str(text or "").split())
    return words if len(words) <= 160 else words[:159].rstrip() + "…"


def _preview(text) -> dict:
    """``snippet`` and, when it cuts the text, ``chars``: the whole text's length, which only the model's reading of
    the reference states (``dialogue_provenance.relation_text``); the stored snippet stays the contract's."""
    full = str(text or "")
    return {"snippet": _snippet(full), **({"chars": len(full)} if len(" ".join(full.split())) > 160 else {})}


def project_answer_text(row: dict) -> str:
    """Authored words displayed instead of a Project lifecycle notice (project_answer.js).

    The stored System row still owns its identity; replies and search share only this presentation.
    """
    return str(row.get("completion_answer") or "") if row.get("type") in {
        "project_started", "project_handoff", "project_completion_summary"} else ""


REPLY_ROLES = ("user", "assistant", "system")


def reply_role(row: dict) -> str:
    """Whose words a replied row holds, from the stored row alone: the one actor identity every reader of it
    shares, whatever label each shows for that person. A lifecycle row carrying an authored answer is that answer
    in Ouroboros's voice wherever Main shows it (project_answer.js::appendProjectRow)."""
    return "assistant" if project_answer_text(row) else {"in": "user", "out": "assistant"}.get(
        row.get("direction"), "system")


def reference_from_row(row: dict, root=None) -> dict:
    kind = str(row.get("type") or "message")
    # The address still hashes the raw System row of an authored lifecycle answer.
    answer = project_answer_text(row)
    kind = ("question" if kind == "quiz" else "completion" if kind in
            {"project_completion_summary", "task_summary"} else
            "system" if row.get("direction") == "system" else "message")
    role = reply_role(row)
    body = card_reply_display(root, row)
    if body:
        role = kind = "system"
    from ouroboros.room_membership import reply_target_home
    address = row_address(row)
    return {"address": format_address(address), "row_sha256": address["row_sha256"], "chat_id": address["chat_id"],
            "role": role, "author_label": (str(row.get("sender_label") or "You") if role == "user"
                                             else {"assistant": "Ouroboros", "system": "System"}[role]),
            "kind": kind, **_preview(body or answer or row.get("text") or row.get("caption")),
            "ts": str(row.get("ts") or ""), "task_id": str(row.get("task_id") or ""),
            "target_home": reply_target_home(row)}


def resolve_reply(root, address: str) -> dict:
    """Resolve a transport-neutral Reply address once, at acceptance."""
    if not isinstance(address, str):
        raise ValueError("reply_to must be an address string")
    if not address:
        return {}
    if address.startswith("task:"):
        from ouroboros.task_status import load_effective_task_result
        task_id = address[5:]
        task = load_effective_task_result(root, task_id, materialize_artifacts=False) if task_id else None
        if task:
            return {"address": address, "chat_id": int(task.get("chat_id") or 1), "role": "assistant",
                    "author_label": "Ouroboros", "kind": "work_card",
                    **_preview(task.get("title") or task.get("text")),
                    "ts": str(task.get("updated_at") or task.get("started_at") or ""), "task_id": task_id,
                    "target_home": "work"}
    else:
        try:
            row, _resolution = resolve_row(root, address)
            if row is None:
                row = resolve_progress_row(root, address)
        except (ValueError, StoreError):
            row = None
        if row:
            return reference_from_row(row, root)
    return {"address": address, "unresolved": True}


class ReplyExcerptUnavailable(ValueError):
    """Fresh acceptance could not resolve the exact authored source of a selected passage."""


class ReplyExcerptNotAuthored(ValueError):
    """The resolved source has no eligible authored body or caption."""


def reply_identity(metadata: dict) -> dict:
    """The sender's part of a Reply. Never accept a client author, snippet, kind or routing fact."""
    address = metadata.get("reply_to", "")
    if not isinstance(address, str):
        raise ValueError("reply_to must be an address string")
    ref = {"address": address} if address else {}
    if "reply_excerpt" in metadata:
        excerpt = metadata["reply_excerpt"]
        if not isinstance(excerpt, dict) or not address.startswith("row:"):
            raise ValueError("reply_excerpt needs an addressed message")
        text, start = excerpt.get("text"), excerpt.get("start_utf16")
        if (not isinstance(text, str) or not text.strip() or type(start) is not int
                or not 0 <= start <= 2**53 - 1 or not _valid_unicode(text)):
            raise ValueError("reply_excerpt needs nonempty Unicode text and a nonnegative UTF-16 position")
        ref["excerpt"] = {"text": text, "start_utf16": start}
    return ref


def _valid_unicode(text: str) -> bool:
    try:
        text.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def same_reply(root, stored: dict, requested: dict) -> bool:
    """A replay names the accepted source and passage, including every digest digit the sender supplied."""
    if stored.get("excerpt") != requested.get("excerpt"):
        return False
    saved, wanted = stored.get("address"), requested.get("address")
    if saved == wanted and not requested.get("row_sha256"):
        return True
    if not saved or not wanted or stored.get("unresolved"):
        return False
    try:
        left, right = parse_address(saved), parse_address(wanted)
    except ValueError:  # task addresses and unresolved free-form addresses only compare literally
        return saved == wanted
    if (left["chat_id"], left["ts"]) != (right["chat_id"], right["ts"]):
        return False
    # Older accepted references only stored the display prefix. Read their source when a retry
    # supplies more digits than the acceptance recorded; never accept an unverified suffix.
    digest = stored.get("row_sha256") or left["row_sha256"]
    prefix = requested.get("row_sha256") or right["row_sha256"]
    if len(prefix) > len(digest):
        _, resolution = resolve_history_row(root, saved)
        digest = resolution.get("address", {}).get("row_sha256", "")
    return digest.startswith(prefix)


def media_caption(row: dict) -> str:
    """An explicit delivery caption, never the history row's generated fallback text."""
    caption = row.get("caption")
    return caption if row.get("type") in ("photo", "video", "document") and isinstance(caption, str) else ""


def excerpt_source_text(row: dict) -> str:
    """The authored source of a selected reply, not the generated preview for a whole-row reply."""
    body = web_attachment_caption(row) if row.get("direction") == "in" else str(row.get("text") or "")
    return project_answer_text(row) or media_caption(row) or body


def ingress_reply(root, metadata: dict, *, accepted: dict | None = None) -> dict:
    """Fresh selections require a real authored row. A duplicate compares with its original acceptance."""
    ref = reply_identity(metadata)
    if accepted is not None:
        return ref  # rejoin_or_refuse compares it before using the saved resolution
    address = ref.get("address", "")
    if "excerpt" not in ref:
        return resolve_reply(root, address)
    try:
        row, _ = resolve_row(root, address)
    except (ValueError, StoreError, OSError) as exc:
        raise ReplyExcerptUnavailable("The quoted message is unavailable. Retry or clear the quote.") from exc
    if row is None:  # resolve_row checks the room, timestamp and every supplied digest digit
        raise ReplyExcerptUnavailable("The quoted message is unavailable. Retry or clear the quote.")
    authored = (row.get("direction") in {"in", "out"} and row.get("role") != "system"
                and not any(row.get(key) for key in ("relay", "quiz", "host_fact", "text_placeholder", "is_progress")))
    plain = (row.get("system_type") or row.get("type")) in {
        None, "", "message", "text", "main_notice", "relay_reply", "proactive_message"}
    body = plain and (web_attachment_caption(row) if row.get("direction") == "in" else row.get("text"))
    if not project_answer_text(row) and not (authored and (body or media_caption(row).strip())
                                             and not _card_display_only(row, _card_display(root, row))):
        raise ReplyExcerptNotAuthored("Only a passage of an authored chat message can be quoted")
    reference = reference_from_row(row, root)
    reference.pop("chars", None)
    return {**reference, **_preview(excerpt_source_text(row)), "excerpt": ref["excerpt"]}


def origin_reply(root, task_id: str, metadata=None) -> dict:
    """The answer's host reference to the owner message that began its turn.

    Best-effort enrichment: a busy or unreadable store omits the reference, never the message."""
    if not task_id or not root:
        return {}
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.project_dialogue import resolve_owner_message_source
    try:
        task = metadata or load_effective_task_result(root, task_id, materialize_artifacts=False) or {}
        ref = task.get("origin_message_ref") or (task.get("metadata") or {}).get("origin_message_ref")
        row = resolve_owner_message_source(root, ref) if ref else None
    except StoreError:
        return {}
    return reference_from_row(row, root) if row else {}


def resolve_progress_row(root, address):
    """Progress uses the same persisted-row hash, found by its indexed chat and time.

    A missing or ambiguous row answers None, as a chat row that does not resolve does."""
    return resolve_row(root, address, stream="progress")[0]


def resolve_history_row(root, address):
    """Exact navigation reads BOTH typed sources; ingress's unresolved preview is not a verdict.

    A readable match survives the other source's import/failure. Only two complete readable
    misses prove absence; conflicting matches and an ambiguous prefix remain ambiguous. A
    same-time different row beside unreadable lines proves nothing here, though generic
    ``resolve_row`` callers keep that ``row_mismatch`` and its candidates.
    """
    from ouroboros import dialogue_positions as dp
    from ouroboros.chat_chain import source_row_id
    from ouroboros.store import StoreMigrating

    matches, states, errors = {}, set(), []
    for source in ("chat", "progress"):
        try:
            row, resolution = resolve_row(root, address, stream=source)
            gap = resolution["status"] == "row_mismatch" and dp.skipped_between(root, source)
            states.add("row_unreadable" if gap else resolution["status"])
            if row is not None:
                matches.setdefault(source_row_id(row), (row, {**resolution, "source": source}))
        except (OSError, StoreError) as exc:
            errors.append(exc)
    if len(matches) > 1 or "row_ambiguous" in states:
        return None, {"state": "ambiguous"}
    if matches:
        return next(iter(matches.values()))
    if errors or "row_unreadable" in states:
        if errors and all(isinstance(exc, StoreMigrating) for exc in errors) and "row_unreadable" not in states:
            raise StoreMigrating("Dialogue history is still importing")
        raise StoreError("Dialogue history source unavailable")
    return None, {"state": "absent"}


def history_row_destination(root, row, thread_id, project_ids, source_refs, rooms):
    """Prefer a real occurrence in this room; otherwise the row's home, without source-ref copies.

    Only row-derived candidates enter the shared membership rule. Retained origins, the quoting
    task and the reference's old chat label cannot supply a home. Unreadable authority raises.
    """
    from ouroboros.chat_chain import _row_chat_id
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.project_dialogue import bound_room_chat, room_membership

    written = _row_chat_id(row)
    def belongs(chat, refs):
        return room_membership(chat, project_ids, refs, rooms)(written, row)

    if thread_id in {1, *project_ids} and belongs(thread_id, source_refs):
        home = thread_id
    else:
        work = bound_room_chat(rooms, row)
        candidates = {1, written, work} & {1, *project_ids}
        homes = [chat for chat in candidates if belongs(chat, [])]
        if len(homes) > 1 and work in homes:  # work speech shown where it was written too: its work's room is home,
            homes = [1 if private_main_row(dict(row)) else work]  # private contact's is Main, its address
        if len(homes) != 1:
            return {"state": "ambiguous" if homes else "unavailable"}
        home = homes[0]
    # Every room's identity, then only the home's routing: a malformed unrelated room blocks nothing.
    if home != 1 and reserved_project_for_chat(root, home).get("lifecycle") != "active":
        return {"state": "unavailable"}
    return {"state": "found" if home == thread_id else "moved", "chat_id": home}


def origin_projection(root, ref):
    """A projected origin keeps its real source address, never hashes view fields."""
    from ouroboros.project_dialogue import resolve_owner_message_source
    try:
        row = resolve_owner_message_source(root, ref)
    except StoreError:
        return {}
    return projection_fields(row) if row else {}
