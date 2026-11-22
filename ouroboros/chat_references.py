"""Host-owned message addresses and Reply facts, shared by every owner ingress.

References describe what the sender pointed at. They do not select work or grant
authority: only a confirmed Project Reply from the bundled owner Telegram bridge
opens its own Project turn. Unresolved ordinary quotes do not block acceptance;
a transient store error during the Telegram decision permits a safe retry.
Selected passages require an authored source and remain sender-reported citations.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Dict, Mapping

from ouroboros.chat_chain import format_address, parse_address, resolve_row, row_address
from ouroboros.chat_uploads import web_attachment_caption
from ouroboros.store import StoreBusy, StoreError, StoreMigrating

TELEGRAM_SOURCE = "skill:telegram"  # the bundled owner bridge's authenticated ingress identity
# The host's mark on a proven Telegram address (``proven_address``), never an ingress's: a room's binding drops it
# and ``exact_reply`` (``LocalChatBridge.activate_update_transport``).
ADDRESS, REPLY_ADDRESS, ROOT_ADDRESS = "host_address", "reply", "root"

# A selected passage's source as acceptance captured it (words and files). It rides the delivery's ``reply_reference``
# by value, so a work reads it when history cannot (``owner_words.reply_target``), and is never stored on the owner row.
SOURCE_KEYS = ("text", "attachments")


def stored_reply(reply):
    """A Reply reference as the owner row stores it and every client reads it: without the captured source."""
    return {key: value for key, value in reply.items() if key not in SOURCE_KEYS} if isinstance(reply, dict) else reply


def has_source(reply) -> bool:
    """Whether a reference carries its selected passage's source (words or files), not only its preview."""
    return isinstance(reply, dict) and bool(reply.get("text") or reply.get("attachments"))


def _same_selection(prior: Mapping[str, Any], citation: Mapping[str, Any]) -> bool:
    """Two carriers' records of one selected reply: the exact address and passage (offsets refer to the rendered
    body, so the selections are compared to each other, never to raw Markdown) and one source actor. Each reader
    labels a person its own way (``You`` and ``User [source=web]``), so the host's role for the replied row decides
    when both recorded it, else the label itself; an identity is never read out of a label."""
    actor = "role" if prior.get("role") and citation.get("role") else "author"
    return all(prior.get(key) == citation.get(key) for key in ("address", "excerpt", actor))


def reply_target(drive_root: Any, reference: Any) -> Dict[str, Any]:
    """The whole message my human's words reply to, under its author, by the address the door resolved: one
    sha-checked read. Unreadable: the address alone (a selected passage that carried its source keeps it); a reply
    to a work card names work, not anyone's words. Its
    author is signed without the places' owner snapshot on purpose: a target is context, never my human's words."""
    reference = reference if isinstance(reference, Mapping) else {}
    address = str(reference.get("address") or "")
    if not address or address.startswith("task:"):
        return {}
    citation = {"excerpt": reference["excerpt"]} if reference.get("excerpt") else {}
    if reference.get("unresolved"):
        return {"address": address, "author": "", "text": "", **citation}
    recorded_author = str((reference.get("author") if citation else "") or reference.get("author_label") or "")
    from ouroboros.dialogue_provenance import quote_words, render_row_text, row_author

    # The host's role for the replied row rides with a selection: the actor its carriers agree on (``_same_selection``).
    recorded_role = {"role": reference["role"]} if citation and reference.get("role") in REPLY_ROLES else {}
    try:
        from ouroboros import chat_chain

        row = chat_chain.resolve_row(drive_root, address)[0] if drive_root is not None else None
    except (StoreError, OSError, ValueError, TypeError):
        row = None
    if not row:
        # A selected passage that arrived with its source by value (a relay's quote) keeps it when history cannot be read.
        held = reference.get("text")
        by_value = ({"text": held, **({"attachments": reference["attachments"]} if reference.get("attachments") else {})}
                    if citation and isinstance(held, str) and held else {})
        return {"address": address, "author": recorded_author, "text": "", **by_value, **citation, **recorded_role}
    if display := card_reply_display(drive_root, row):
        return {"address": address, "author": "System", "text": display, **citation}
    if citation:
        files = excerpt_source_attachments(row)
        return {"address": address, "author": "Ouroboros" if project_answer_text(row) else row_author(row)["label"],
                "text": excerpt_source_text(row), **citation, "role": reply_role(row),
                **({"attachments": files} if files else {})}
    return {"address": address, "author": row_author(row)["label"],
            "text": quote_words(row) if (row.get("reply_to") if isinstance(row.get("reply_to"), Mapping) else {}).get("excerpt") else render_row_text(row)}


def selected_reply_target(root, source, matching_deliveries):
    """Use the selected row's retained target, or later deliveries of that same address and passage."""
    held = source.get("_owner_reply_target")
    target = held if isinstance(held, Mapping) and held else reply_target(root, source.get("reply_to"))
    if not target.get("address") or target.get("text"):
        return target
    return next((prior for prior in matching_deliveries if isinstance(prior, Mapping) and prior.get("text")
                 and all(prior.get(key) == target.get(key) for key in ("address", "excerpt"))), target)


def delivery_reply(stored, held):
    """The saved row's reference as delivery carries it: the source acceptance captured stays beside it while ``held``
    names the same address and passage, never beside another's."""
    if not (isinstance(stored, dict) and isinstance(held, dict) and stored.get("excerpt")
            and held.get("address") == stored.get("address") and held.get("excerpt") == stored.get("excerpt")):
        return stored
    return {**stored, **{key: held[key] for key in SOURCE_KEYS if key in held}}


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
    fields["reply_author_label"] = _reply_author_label(row)
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


def _reply_author_label(row: dict) -> str:
    from ouroboros.dialogue_provenance import recorded_speaker_name, row_provenance

    if reply_role(row) == "user":
        return recorded_speaker_name(row) or ("author not recorded" if row_provenance(row) == "place" else "You")
    return "Ouroboros" if reply_role(row) == "assistant" else "System"


def reply_role(row: dict) -> str:
    """Whose words a replied row holds, from the stored row alone: the one actor identity every reader of it
    shares, whatever label each shows for that person. A lifecycle row carrying an authored answer is that answer
    in Ouroboros's voice wherever Main shows it (project_answer.js::appendProjectRow). A row stored as ``role: system``
    is the host's even when it entered by the owner's door (a call's opening: ``direction: in``), as history shows it."""
    if project_answer_text(row):
        return "assistant"
    return "system" if row.get("role") == "system" else {"in": "user", "out": "assistant"}.get(
        row.get("direction"), "system")


def reference_from_row(row: dict, root=None) -> dict:
    kind = str(row.get("type") or "message")
    # The address still hashes the raw System row of an authored lifecycle answer.
    answer = project_answer_text(row)
    # My human's recorded answer to a card reads as their choice, never the host frame the System row stores;
    # its role and address stay the row's, which room membership and saved stamps read.
    from ouroboros.owner_relay import owner_text, quotable_owner_row
    chose = owner_text(row) if kind == "quiz_answer" and quotable_owner_row(row) else ""
    kind = ("question" if kind == "quiz" else "completion" if kind in
            {"project_completion_summary", "task_summary"} else
            "system" if row.get("direction") == "system" else "message")
    role = reply_role(row)
    body = card_reply_display(root, row)  # the host's account of a card display, for its preview alone: the row is still its sender's
    from ouroboros.room_membership import reply_target_home
    address = row_address(row)
    return {"address": format_address(address), "row_sha256": address["row_sha256"], "chat_id": address["chat_id"],
            **({"quote_role": "user"} if chose else {}),
            "role": role, "author_label": str(row.get("sender_label") or "You") if chose else _reply_author_label(row),
            "kind": kind, **_preview("" if row.get("text_placeholder") else body or answer or chose or row.get("text") or row.get("caption")),
            "ts": str(row.get("ts") or ""), "task_id": str(row.get("task_id") or ""),
            "target_home": reply_target_home(row)}


def resolve_reply(root, address: str, *, strict_transient: bool = False) -> dict:
    """Resolve a transport-neutral Reply address once, at acceptance."""
    if not isinstance(address, str):
        raise ValueError("reply_to must be an address string")
    if not address:
        return {}
    if address.startswith("task:"):
        from ouroboros.task_status import load_effective_task_result
        task_id = address[5:]
        try:
            task = load_effective_task_result(root, task_id, materialize_artifacts=False) if task_id else None
        except (StoreBusy, StoreMigrating):
            if strict_transient:
                raise
            task = None
        except (OSError, StoreError):
            task = None
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
        except (StoreBusy, StoreMigrating):
            if strict_transient:
                raise
            row = None
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


def excerpt_source_attachments(row: dict) -> list:
    """Files beside a selected source. Delivered documents record their download facts on the row, not in uploads."""
    files = copy.deepcopy(row["attachments"]) if isinstance(row.get("attachments"), list) else []
    if row.get("direction") == "out" and row.get("type") == "document" and row.get("filename"):
        files.append({"name": row["filename"], "mime": row.get("mime"), "size": row.get("size_bytes"),
                      **({"download_url": row["download_url"]} if row.get("download_url") else {})})
    return files


def ingress_reply(root, metadata: dict, *, accepted: dict | None = None, strict_transient: bool = False) -> dict:
    """Fresh selections require a real authored row. A duplicate compares with its original acceptance."""
    ref = reply_identity(metadata)
    if accepted is not None:
        return ref  # rejoin_or_refuse compares it before using the saved resolution
    address = ref.get("address", "")
    if "excerpt" not in ref:
        return resolve_reply(root, address, strict_transient=strict_transient)
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
    source = excerpt_source_text(row)  # captured now, before the owner row is saved: delivery needs no later history read
    files = excerpt_source_attachments(row)
    return {**reference, **_preview(source), "excerpt": ref["excerpt"], "text": source,
            **({"attachments": files} if files else {})}


_BINDING_SOURCES = (None, {})


def _checksum_only_source(source: dict) -> bool:
    """A legacy ref that names its source only by its room and the words' checksum (no client id, no time)."""
    digest = source.get("text_sha256")
    return bool(isinstance(source.get("chat_id"), int) and not isinstance(source.get("chat_id"), bool)
                and not source.get("client_message_id") and not source.get("ts")
                and isinstance(digest, str) and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest))


def _origin_bindings(bindings, ts, client_id="", chat_id=None):
    """Narrow the immutable registry snapshot to candidate bindings before checking their source and room.

    A legacy ref can name its source by client id without a timestamp; one that names only its source room and
    the words' checksum is admitted for a lookup that names that same source room. A candidate is not a match:
    the batched physical resolver and the visibility/membership predicates decide."""
    global _BINDING_SOURCES
    snapshot, by_source = _BINDING_SOURCES
    if snapshot is not bindings:
        by_source = {}
        for task, binding in bindings.items():
            source = binding.get("source_ref") if isinstance(binding, dict) else None
            if not isinstance(source, dict):
                continue
            if source.get("ts"):
                key = ("ts", str(source["ts"]))
            elif source.get("client_message_id"):
                key = ("client_message_id", str(source["client_message_id"]))
            elif _checksum_only_source(source):
                key = ("chat_id", source["chat_id"])
            else:
                continue
            by_source.setdefault(key, {})[task] = binding
        _BINDING_SOURCES = bindings, by_source
    candidates = by_source.get(("ts", str(ts or "")), {})
    if client_id:
        candidates = candidates | by_source.get(("client_message_id", client_id), {})
    if chat_id is not None:
        candidates = candidates | by_source.get(("chat_id", chat_id), {})
    return candidates


def visible_source_reply(root, ref, chat_id, *, reply=None, row=None, display_row=None):
    """Main can quote its owner's local Project original. Other rooms need membership or a permitted copy.

    A saved reply is a preview of the addressed row. Its words alone cannot establish room membership.
    """
    from ouroboros.chat_chain import _row_chat_id, parse_address
    from ouroboros.owner_relay import quotable_owner_row
    from ouroboros.project_dialogue import (owner_message_ref_is_valid, project_origin_rows,
                                            resolve_owner_message_source, source_refs_for_project)
    from ouroboros.projects_registry import RoomResolver, reserved_project_chat_ids
    from ouroboros.room_membership import binding_origin_allowed, room_membership, row_rooms

    if row is None and not ref and not (isinstance(reply, dict) and reply.get("address")):
        return {}
    try:
        supplied_row = row is not None
        if row is None:
            row = resolve_owner_message_source(root, ref) if ref else resolve_row(root, reply["address"])[0]
        rooms = RoomResolver(root, strict=True)
        projects = reserved_project_chat_ids(root, strict=True)
        # A shared live frame has one preview for all its appearances.
        destinations = {chat_id}
        if display_row is not None:
            destinations.update(row_rooms(chat_id, display_row, projects, rooms))
        if row is not None:
            written = _row_chat_id(row)
            for room in destinations - row_rooms(written, row, projects, rooms):
                if room == 1 and supplied_row:
                    # A carried quote alone cannot establish a home in this store.
                    row, _ = resolve_row(root, row_address(row))
                    if row is None:
                        return {}
                if not binding_origin_allowed(written, row):
                    return {}
                if (room == 1 and (written == 1 or written in projects) and quotable_owner_row(row)
                        and history_row_destination(root, row, 1, projects, [], rooms)["state"] == "moved"):
                    continue
                if room not in projects:
                    return {}
                bindings = _origin_bindings(rooms.bindings, row.get("ts"), str(row.get("client_message_id") or ""), written)
                refs = source_refs_for_project(root, room, bindings=bindings)
                if not room_membership(room, projects, refs, rooms)(written, row):
                    return {}
            return reference_from_row(row, root)
        if isinstance(reply, dict) and destinations <= projects:
            # A verified copy from another drive is visible only where its binding projects it.
            candidates = _origin_bindings(rooms.bindings, parse_address(reply.get("address"))["ts"])
            bindings = [b for b in candidates.values()
                        if owner_message_ref_is_valid(source := b.get("source_ref")) and source.get("row_sha256")
                        and (not ref or source == ref) and format_address(source) == reply.get("address")]
            if all(project_origin_rows(root, room, bindings=bindings) for room in destinations):
                return dict(reply)
    except (OSError, ValueError, StoreError):
        pass  # an unknown room or source cannot attest its visibility
    return {}


def projected_reply_fields(root, chat_id, row, fields, *, visibility=None):
    """Recheck a generated quote at its history or pointer destination. Explicit relay replies keep their source."""
    reply = fields.get("reply_to")
    if (not isinstance(reply, dict) or not isinstance(reply.get("address"), str)
            or row.get("reply_to_relay") or row.get("type") == "relay_entry"):
        return fields
    # Older task speech and file/quiz rows kept the host's owner quote without the origin marker.
    origin = (row.get("origin_reply") or row.get("type") == "project_completion_summary" or
              (row.get("task_id") and row.get("direction") in {"out", "system"}
               and (reply.get("quote_role") or reply.get("role")) == "user"))
    if origin:
        cache = visibility if visibility is not None else {}
        address = reply["address"]
        if address not in cache:
            cache[address] = visible_source_reply(root, None, chat_id, reply=reply)
        if cache[address]:
            fields["reply_to"] = cache[address]
        else:
            fields.pop("reply_to")
    return fields


def origin_reply(root, task_id: str, metadata=None, *, chat_id=1, binding_fallback=False, display_row=None) -> dict:
    """Host reference to the work's origin, when its final destination can show it."""
    if not task_id or not root:
        return {}
    from ouroboros.task_status import load_effective_task_result
    try:
        supplied = metadata if isinstance(metadata, dict) else {}
        task = supplied if isinstance(supplied.get("metadata"), dict) else (
            load_effective_task_result(root, task_id, materialize_artifacts=False) or {})
        held = task.get("metadata") or {}
        saved = task.get("origin_message_ref") or held.get("origin_message_ref")
        ref = supplied.get("origin_message_ref") or saved
        if not ref and binding_fallback:
            from ouroboros.projects_registry import project_binding_for_task
            ref = (project_binding_for_task(root, task_id, strict=True) or {}).get("source_ref")
        if isinstance(ref, dict) and ref:
            snapshot = held.get("origin_message_snapshot") or {}
            reply = snapshot.get("reply") if ref == saved and isinstance(snapshot, dict) and snapshot.get("projectable") is True else None
            return visible_source_reply(root, ref, chat_id, reply=reply, display_row=display_row)
    except (OSError, ValueError, StoreError):
        pass
    return {}


def _origin_row(root, task_id: str, metadata=None):
    if not task_id or not root:
        return None
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.project_dialogue import resolve_owner_message_source
    try:
        task = metadata or load_effective_task_result(root, task_id, materialize_artifacts=False) or {}
        ref = task.get("origin_message_ref") or (task.get("metadata") or {}).get("origin_message_ref")
        return resolve_owner_message_source(root, ref) if ref else None
    except StoreError:
        return None


def telegram_reply_room(root, ref: dict, ingress_chat: int) -> tuple:
    """Where an accepted owner-bot Reply runs, ``(room, decided)``: ``0`` keeps the ordinary turn of its own
    conversation, a Project's chat id opens a separate turn there, ``None`` refuses it without routing it anywhere.

    Only a confirmed Project target opens a turn: the room its work lives in (``RoomResolver``: its binding, else
    its admission) or, for a row no work owns, the room it was written in, as ``room_membership.reply_target_room``
    decides it. A retired or deleted Project there refuses the Reply, never sends it to Main. Everything else stays
    the ordinary turn of this conversation: the owner's own words, Main and this conversation, a question (its card
    is its answer's door), a row no room owns (``target_home`` ``none``). The private chat's physical id
    (``ingress_chat``) and a local room id are different facts that can be equal: an id they share is this
    conversation's unless that Project's work is proven (``_project_work_here``). A target the store cannot find or
    read now is an ordinary turn too, but ``decided`` False: nothing records a room it could not name. Unreadable
    registry or origin state raises."""
    if not isinstance(ref, dict) or not ref or ref.get("unresolved"):
        return 0, False
    if ref.get("role") == "user" or ref.get("kind") == "question":
        return 0, True
    from ouroboros.contracts.chat_id_policy import is_project_chat_id
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.projects_registry import RoomResolver
    from ouroboros.room_membership import TARGET_ABSENT, TARGET_ROOM, TARGET_UNKNOWN, TARGET_WORK, reply_target

    rooms = RoomResolver(root, strict=True)
    home, task, chat = reply_target(rooms, ref)
    if home in (TARGET_ABSENT, TARGET_UNKNOWN):
        return 0, False
    room = int(rooms(task) or 0) if home == TARGET_WORK and task else chat if home == TARGET_ROOM else 0
    if room == 1 or not is_project_chat_id(room):
        return 0, True
    if room == ingress_chat:  # the number this conversation and a Project share: the Project's only when proven
        proven = _project_work_here(root, rooms, task, room) if home == TARGET_WORK and task else False
        if not proven:
            return 0, proven is not None
    project = reserved_project_for_chat(root, room)
    if not project and home == TARGET_ROOM:
        return 0, True  # a written connector conversation is not itself a Project reservation
    return (room if project.get("lifecycle") == "active" else None), True


def _project_work_here(root, rooms, task: str, chat: int):
    """Whether a work whose room bears the owner's private chat number is proven that Project's: a Project binding on
    its lineage, or a root begun by a word written in that room or by the owner bot's Reply the host gave that very
    room; never this conversation's own turn (False). None when the message that began it cannot be named: nothing
    proves either room. The owner message is read by its host reference; unreadable raises."""
    from ouroboros.project_dialogue import resolve_owner_message_source
    from ouroboros.projects_registry import project_chat_for_task

    lineage = rooms.lineage(task)
    top = str(lineage.get("root_task_id") or task)
    if any(project_chat_for_task(root, tid) == chat
           for tid in {task, str(lineage.get("parent_task_id") or ""), top} if tid):
        return True
    ref = _origin_ref(root, top)
    row = resolve_owner_message_source(root, ref) if ref else None
    if not row:
        return None
    if not telegram_chat_row(row.get("source"), row.get("chat_id"), row.get("client_message_id")):
        return True
    reply = row.get("reply_to")
    return isinstance(reply, dict) and reply.get("telegram_room_chat_id") == chat


def telegram_chat_row(source, chat_id, client_message_id) -> bool:
    """Whether an owner message is one the Telegram bridge took in its own chat (``tg:[<bot>:]<chat>:<message>``):
    its chat id is that Telegram chat's physical address, never a room's, whatever room bears the same number.
    A message the host delivers into a room on the owner's behalf (a late answer, a card press) is the room's."""
    parts = str(client_message_id or "").split(":")
    return source == TELEGRAM_SOURCE and parts[0] == "tg" and len(parts) in (3, 4) and parts[-2] == str(chat_id)


def bundled_bridge(root) -> bool:
    """Whether the one skill named ``telegram`` here is the bundled bridge, seeded from the application's own
    release (``skill_loader._classify_skill_source``: ``native``): a same-named skill from any other source, or two
    of that name, is an ordinary transport whose Replies are ordinary references."""
    from pathlib import Path

    from ouroboros.config import SKILL_SOURCE_NATIVE
    from ouroboros.skill_loader import discover_skill_identity

    try:
        found = discover_skill_identity(Path(root), "telegram")
    except Exception:
        return False
    return len(found) == 1 and found[0].source == SKILL_SOURCE_NATIVE and not found[0].identity_collision


def exact_reply_ingress(root, source, chat_id, client_message_id, transport) -> bool:
    """Whether an accepted owner Reply has its room decided (``telegram_reply_room``): the bundled bridge's
    (``bundled_bridge``), under the bot-qualified address it proves (``proven_address``). Any other Reply, the
    bridge's older botless ids included, stays an ordinary reference."""
    row = {"source": source, "direction": "in", "transport": transport, "client_message_id": client_message_id,
           "chat_id": chat_id}
    return source == TELEGRAM_SOURCE and bool(proven_address(row)) and bundled_bridge(root)


def keeps_binding(msg: dict) -> bool:
    """Whether an owner message leaves its room's binding as it was: the host's delivery of the owner's Telegram act (a
    late answer to a question, a press on a living card) into the room that act belongs to is no word written
    there, so it neither binds that room to Telegram nor clears another connector's binding (its turn's own address:
    ``turn_reply_transport``)."""
    meta = msg.get("task_metadata") if isinstance(msg.get("task_metadata"), dict) else {}
    reply = (msg.get("accepted_source_row") or {}).get("reply_to")
    return msg.get("source") == TELEGRAM_SOURCE and bool(
        meta.get("late_answer") or meta.get("card_press") or
        (isinstance(reply, dict) and type(reply.get("telegram_room_chat_id")) is int and reply["telegram_room_chat_id"] > 0))


def proven_address(row: dict) -> dict:
    """Where the bundled bridge accepted one owner row: ``kind``, ``conversation_id`` and ``bot_id``, never an anchor.

    The row's own facts must agree exactly: its ``tg:<bot>:<chat>:<message>`` id, the chat it was accepted in and
    its transport (whose ``bot_id`` a row accepted before the bridge sent it may lack). A botless id proves no bot:
    ``{}``, as for every other row, and nothing is guessed."""
    transport, parts = row.get("transport"), str(row.get("client_message_id") or "").split(":")
    if not (row.get("source") == TELEGRAM_SOURCE and row.get("direction") == "in" and isinstance(transport, dict)
            and transport.get("kind") == "telegram" and len(parts) == 4 and parts[0] == "tg" and parts[1].isdigit()):
        return {}
    chat = str(transport.get("conversation_id") or "")
    if parts[2] != chat or str(row.get("chat_id")) != chat or transport.get("bot_id", parts[1]) != parts[1]:
        return {}
    return {"kind": "telegram", "conversation_id": chat, "bot_id": parts[1]}


def exact_reply_transport(row: dict) -> dict:
    """The address one accepted exact Reply gives its own turn, with its strict anchor; ``{}`` for any other row.

    Only the bundled bridge's owner row the host sent to a Project (``telegram_room_chat_id``) qualifies: its
    ``proven_address``, marked ``reply``, and in ``exact_reply`` the owner's message id with the card he pressed
    or the helper his Reply was about (skills/telegram/SKILL.md). The room's own binding stays as it is."""
    reply, transport, address = row.get("reply_to"), row.get("transport"), proven_address(row)
    room = reply.get("telegram_room_chat_id") if isinstance(reply, dict) else None
    owner = transport.get("owner_message_id") if address else None
    if (type(room) is not int or room <= 0 or type(owner) is not int or owner <= 0
            or str(row["client_message_id"]).rsplit(":", 1)[1] != str(owner)):
        return {}
    exact, card = {"owner_message_id": owner}, transport.get("reply_card_message_id")
    child = transport.get("reply_child_task_id")
    if type(card) is int and card > 0:
        exact["card_message_id"] = card
    if isinstance(child, str) and child:
        exact["child_task_id"] = child
    return {**address, ADDRESS: REPLY_ADDRESS, "exact_reply": exact}


def decided_reply_address(row: dict) -> dict:
    """Where a routing notice about an accepted owner-bot Reply the host gave a Project, or refused, answers it: the bot
    and chat that took it, marked ``reply``, without an anchor; ``{}`` for any other row (its binding stays)."""
    reply = row.get("reply_to")
    address = proven_address(row) if isinstance(reply, dict) and reply.get("telegram_room_chat_id", 0) != 0 else {}
    return {**address, ADDRESS: REPLY_ADDRESS} if address else {}


def _origin_ref(root, task_id: str):
    """The owner message that began ``task_id``: as its live turn or running task holds it, else its stored result."""
    from ouroboros.task_status import load_effective_task_result
    from supervisor import queue
    from supervisor.active_activity import get_direct_activity_registry

    activity, running = get_direct_activity_registry().get(task_id), None
    if activity is not None:
        ref = activity.origin_message_ref
    else:
        with queue._queue_lock:
            running = queue.RUNNING.get(task_id)
        ref = ((running or {}).get("task") or {}).get("origin_message_ref")
        if running is None:
            task = load_effective_task_result(root, task_id, materialize_artifacts=False) or {}
            ref = task.get("origin_message_ref") or (task.get("metadata") or {}).get("origin_message_ref")
    return ref if isinstance(ref, dict) else None


TURN_ADDRESS = "exact_reply_transport"  # the turn's own: the host's at acceptance, never a sender's


def with_turn_address(metadata, row: dict) -> dict:
    """The task metadata of the turn ``row`` begins, carrying that row's ``exact_reply_transport`` (and no other value
    under its key): every delivery of the turn reads it from the turn itself (``turn_reply_transport``)."""
    meta = {key: value for key, value in (metadata or {}).items() if key != TURN_ADDRESS}
    exact = exact_reply_transport(row)
    return {**meta, TURN_ADDRESS: exact} if exact else meta


def _turn_task(root, task_id: str, *, live_only: bool = False) -> dict:
    """The live task, or for durable delivery its stored result, including its host-owned lineage."""
    from ouroboros.task_status import load_effective_task_result
    from supervisor import queue

    with queue._queue_lock:
        running = queue.RUNNING.get(task_id)
    if running is not None:
        return running.get("task") or {}
    if live_only:
        from supervisor.workers import direct_chat_turn

        return direct_chat_turn(task_id) or {}
    return load_effective_task_result(root, task_id, materialize_artifacts=False) or {}


def _turn_metadata(task: dict) -> dict:
    meta = task.get("metadata")
    return meta if isinstance(meta, dict) else {}


def turn_reply_transport(root, task_id: str, *, anchor: bool = True, strict: bool = False, live_only: bool = False) -> dict:
    """Where everything ``task_id``'s turn sends goes (its answer, questions, media, cards, typing; progress and its
    end's notice without the anchor, ``anchor`` False), when the owner's Telegram chat began it: an accepted exact
    Reply's address with its strict anchor, as the host accepted it and the turn carries it from its start
    (``TURN_ADDRESS``), so no read of the source row is needed; his late answer to a question there, the bot and
    chat that answer came from (``late_answer_surface``). A helper inherits its root's surface without the root's
    anchor. ``{}`` for every other turn: the room's binding, as before.
    Ordinary host progress relies on live lineage only, not a full saved result (``live_only``).
    A record that cannot be read answers ``{}`` too, or raises for ``strict``."""
    if not root or not task_id:
        return {}
    try:
        task = _turn_task(root, task_id, live_only=live_only)
        address = reply_transport_from_metadata(root, _turn_metadata(task), anchor=anchor)
        if not address and task.get("delegation_role") == "subagent":
            root_id = str(task.get("root_task_id") or task.get("parent_task_id") or "")
            if root_id and root_id != task_id:
                parent = _turn_task(root, root_id, live_only=live_only)
                address = reply_transport_from_metadata(root, _turn_metadata(parent), anchor=False)
        return address
    except (OSError, ValueError, TypeError, StoreError):
        if strict:
            raise
        return {}


def reply_transport_from_metadata(root, metadata: dict, *, anchor: bool = True) -> dict:
    """Read the host's per-turn surface by value, separate from the room's binding."""
    carried = metadata.get(TURN_ADDRESS)
    if not (isinstance(carried, dict) and carried.get(ADDRESS) == REPLY_ADDRESS):
        carried = late_answer_surface(root, metadata.get("late_answer"))
    if not (carried.get("kind") == "telegram" and str(carried.get("conversation_id") or "").lstrip("-").isdigit()
            and str(carried.get("bot_id") or "").isdigit()):
        return {}
    address = {key: carried[key] for key in ("kind", "conversation_id", "bot_id", ADDRESS)}
    if anchor and isinstance(carried.get("exact_reply"), dict):
        address["exact_reply"] = dict(carried["exact_reply"])
    return address


def late_answer_surface(root, late) -> dict:
    """The Telegram bot and chat the owner's late answer came from, as its question's record keeps it
    (``answer_surface``, written with the answer the bundled bridge relayed), marked ``reply``, without an anchor;
    ``{}`` for an answer from anywhere else, and for one recorded before the record kept its surface."""
    from ouroboros.owner_quiz import late_answer_ref, resolved_late_quiz

    ref = late_answer_ref(late)
    try:
        block = resolved_late_quiz(root, ref) if ref else {}
    except (OSError, ValueError, TypeError, StoreError):
        return {}
    surface = block.get("answer_surface") if block.get("answer_source") == TELEGRAM_SOURCE else None
    if not (isinstance(surface, dict) and surface.get("kind") == "telegram"
            and str(surface.get("bot_id") or "").isdigit() and str(surface.get("conversation_id") or "")):
        return {}
    return {"kind": "telegram", "conversation_id": str(surface["conversation_id"]), "bot_id": str(surface["bot_id"]),
            ADDRESS: REPLY_ADDRESS}


def delivery_transport(root, bound: dict, task_id) -> dict:
    """What a live question, medium, card or typing of ``task_id`` carries: the address of an accepted exact Reply's
    own turn first (it wins on that turn and leaves ``bound``, the room's binding, as it is), else the binding.
    Its answer takes the same address at the send seam (``answer_reference``)."""
    return turn_reply_transport(root, str(task_id or "")) or bound


def answer_reference(root, meta: dict, task_id: str, *, is_progress: bool, chat_id=1, display_row=None) -> None:
    """The send seam's quote and address for one answer of ``task_id`` (``meta``, set only where absent): it quotes
    the owner message that began its turn (``origin_reply``, best effort), and answers where that turn's own address
    says (``turn_reply_transport``), its own row read or not. A row naming another source keeps the quote it names and
    the room's binding; a host receipt names its address itself (``host_operation_reply_kwargs``); progress is
    ``live_transport``'s."""
    if is_progress:
        return
    explicit = "reply_to" in meta
    if not explicit and (reply := origin_reply(root, task_id, meta, chat_id=chat_id, display_row=display_row)):
        meta.update(reply_to=reply, origin_reply=True)
    if (not explicit or not meta["reply_to"]) and (exact := turn_reply_transport(root, task_id)):
        meta.setdefault("transport", exact)
    elif ((row := _origin_row(root, task_id, meta if meta.get("origin_message_ref") else None))
          and telegram_chat_row(row.get("source"), row.get("chat_id"), row.get("client_message_id"))
          and not (row.get("reply_to") or {}).get("telegram_room_chat_id")):
        meta["logical_room_chat_id"] = 1  # physical owner turn, independent of a coincident Project number


def live_transport(root, meta: dict, room_binding: Callable[[], dict], task_id: str, *, role: str, system_type: str,
                   is_progress: bool) -> dict:
    """What a row's live frame carries, after its rows are written (never stored): a helper's card or notice, its
    root's address (``helper_card_address``), which alone needs the room's binding read. With a binding, add the address
    beside it (``telegram_root``, which only the bundled bridge reads), so another connector bound there keeps the row;
    an exact Reply's turn's progress, where its answer goes but without the anchor, so Silent mode replaces it as
    before; ``{}`` for any other row."""
    address = helper_card_address(root, meta, role=role, system_type=system_type, is_progress=is_progress)
    if address:
        bound = room_binding()
        return {**bound, "telegram_root": address} if bound else address
    # An ordinary host status has a room even without a live turn. Unlike a helper card or
    # a turn's own speech, it need not reload completed work to reconstruct a Telegram route.
    return turn_reply_transport(root, task_id, anchor=False, live_only=role == "system") if is_progress and task_id else {}


def helper_card_address(root, meta: dict, *, role: str, system_type: str, is_progress: bool) -> dict:
    """``telegram_root_address`` of the root for a helper's card (``subagent_event``) or a separate row about
    the helper (its host notice: ``terminal_incident`` or a card row; its typed message), so the phone gets
    them in a room without its own Telegram binding. Ordinary narration and a helper's final report get none."""
    child = meta.get("delegation_role") == "subagent"
    notice = child and (role == "system" and (bool(meta.get("card_row")) or system_type == "terminal_incident")
                        or (not is_progress and role != "system" and bool(system_type)))
    if not (meta.get("subagent_event") or notice):
        return {}
    root_id = str(meta.get("root_task_id") or meta.get("parent_task_id") or "")
    return turn_reply_transport(root, root_id, anchor=False) or telegram_root_address(root, root_id)


def telegram_root_address(root, root_task_id: str) -> dict:
    """Where the bundled bridge accepted the owner message that began ``root_task_id``: its ``proven_address``,
    marked ``root``, never that message's anchor; ``{}`` when it came another way or proves no bot and chat."""
    if not root or not root_task_id:
        return {}
    from ouroboros.project_dialogue import resolve_owner_message_source

    try:
        ref = _origin_ref(root, root_task_id)
        row = resolve_owner_message_source(root, ref) if ref else None
    except (OSError, ValueError, TypeError, StoreError):
        return {}
    address = proven_address(row) if row else {}
    return {**address, ADDRESS: ROOT_ADDRESS} if address else {}


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


def quiz_question_row(root, answer, *, answer_seq):
    """The stored question displaying this exact answer, by asker and quiz id, never a source-ref copy."""
    from ouroboros import dialogue_positions as dp, dialogue_store as ds

    task, quiz = str(answer.get("task_id") or ""), answer.get("quiz")
    if answer.get("type") != "quiz_answer" or not task or not isinstance(quiz, dict) or not quiz.get("quiz_id"):
        return None
    for entry in ds.iter_probes(root, "chat", [("task_id", task)]):
        if (entry.row.get("type") == "quiz" and isinstance(entry.row.get("quiz"), dict)
                and entry.row["quiz"].get("quiz_id") == quiz["quiz_id"]):
            return entry
    # On a miss, the question could still be in the unread history preceding its answer.
    ds.require_imported(root, "chat", upper=answer_seq)
    if dp.skipped_between(root, "chat", upper=answer_seq):
        raise StoreError("Question history source unavailable")
    return None


def history_row_anchor(root, address, thread_id, project_ids, source_refs, rooms):
    """Resolve the destination and keep the same question receipt for paging at an exact answer."""
    row, target = resolve_history_row(root, address)
    if row is None:
        return target, None
    if row.get("type") == "quiz_answer":
        question = quiz_question_row(root, row, answer_seq=target["seq"] if target["source"] == "chat" else None)
        if question is None:
            return {"state": "unavailable"}, None
        row, source_refs = question.row, []  # a Project selecting the answer does not host its card
        target["question"] = question
    try:
        destination = history_row_destination(root, row, thread_id, project_ids, source_refs, rooms)
    except ValueError:
        raise StoreError("Project room authority unavailable") from None
    return destination, target


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
    from ouroboros.room_membership import binding_origin_allowed
    return projection_fields(row, root) if row and binding_origin_allowed(ref["chat_id"], row) else {}
