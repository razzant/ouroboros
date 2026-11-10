"""Exact owner quotes and contextual notes share one idempotent mailbox entry.

Quote authority is possession of a host-retained ingress row, never the teller's
lane. The mailbox is the durable delivery owner; ACK follows saved cognition.
Return addresses are retained by the recipient independently of the teller.
How a member comes to hold my human's rows lives here too: the host's carriage of a
work's rows down its tree at receipt (``carry_at_receipt``) and each drain's catch-up
(``carry_to_members``), its birth set from its record and room (``task_words_text``, files
staged by ``stage_birth_files``), and the identity read of a text already sent
(``words_unseen_in`` over ``owner_words.words_stamp`` lines), which carriage and placement
use to render a row once.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from ouroboros.chat_chain import format_address, row_address
from ouroboros.context_budget import OWNER_WORDS_KIND

KIND_RELAY = "relay_entry"
# A carried row's ``went_to`` when no work was delivered it: a head of the room routed it. A typed marker, never a
# task id (an id carries no colon) and never the empty value that means ``reached this work``.
ROUTED_BY_HEAD = "routed:head"


def owner_row_address(row) -> str:
    """Canonical row address, excluding local source-authority annotations."""
    clean = {key: value for key, value in (row or {}).items() if key != "_owner_source_root"}
    return format_address(row_address(clean))


def quotable_owner_row(row, drive_root=None, *, accounts=None) -> bool:
    """An attested owner row (``owner_words.owner_row``, over the caller's one place snapshot when given) or an
    owner's answered card."""
    from ouroboros.owner_quiz import secret_outcome_block
    from ouroboros.owner_words import owner_row
    # A secret request's answer row is the host's outcome frame: never a quotable owner row.
    return owner_row(row, drive_root, accounts=accounts) or bool(row and row.get("type") == "quiz_answer"
        and row.get("source") == "owner_quiz_answer" and (row.get("quiz") or {}).get("state") == "answered"
        and secret_outcome_block(row) is None)


def owner_text(row) -> str:
    if row.get("type") == "quiz_answer":
        from ouroboros.owner_quiz import late_answer_owner_text
        return late_answer_owner_text(row["quiz"])
    return "" if row.get("text_placeholder") else str(row.get("text") or "")


def held_owner_messages(ctx) -> dict:
    held = getattr(ctx, "_held_owner_messages", None)
    if not isinstance(held, dict):
        held = {}
        ctx._held_owner_messages = held
        metadata = getattr(ctx, "task_metadata", None) or {}
        # Continue carries host-retained rows; it does not acquire arbitrary history.
        inherited = (metadata.get("continuation") or {}).get("owner_sources") or {}
        if not hasattr(ctx, "_open_relays"):
            ctx._open_relays = copy.deepcopy(inherited.get("open_relays") or {})
        held.update(copy.deepcopy(inherited.get("held_owner_messages") or metadata.get("held_owner_messages") or {}))
        from ouroboros.project_dialogue import resolve_owner_message_source
        from ouroboros.tool_access import canonical_data_root
        if ref := metadata.get("origin_message_ref"):
            from ouroboros.store import StoreError
            try:
                row = resolve_owner_message_source(canonical_data_root(ctx), ref)
            except StoreError:  # its archive is still importing: nothing held yet, never a failed turn
                row = None
            if quotable_owner_row(row, canonical_data_root(ctx)):
                held[owner_row_address(row)] = {**row, "_owner_source_root": str(canonical_data_root(ctx))}
        persist_held_state(ctx)
    return held


def retain_owner_message(ctx, row=None, *, client_message_id="", accounts=None, source_root=None):
    """Hold one owner row of this turn by its address and return it (``None`` when nothing was held), so its corpus
    row can name the same room and time; ``accounts``: the caller's one place snapshot for its rows. A row named
    by client id is looked up in the context's chat, then in the room the room rule names for its tree: a task
    bound to a Project after admission keeps its admitted chat (``control_relay._teller_room``). ``source_root``:
    the data root the row was read from (a teller's, for a relayed row), kept beside it as ``_owner_source_root``."""
    if ctx is None:
        return None
    if row is None and client_message_id and getattr(ctx, "drive_root", None):
        from supervisor.message_ingress import accepted_chat_message
        from ouroboros.projects_registry import project_chat_for_task_tree
        from ouroboros.tool_access import canonical_data_root
        root, meta = canonical_data_root(ctx), getattr(ctx, "task_metadata", None) or {}
        chat_id = getattr(ctx, "current_chat_id", None)  # a context may carry no chat: Main's lookup
        chat = 1 if chat_id is None else int(chat_id)
        bound = project_chat_for_task_tree(root, str(getattr(ctx, "task_id", "") or ""), meta.get("parent_task_id", ""),
                                           meta.get("root_task_id", ""))
        row = accepted_chat_message(root, chat, client_message_id) or (
            accepted_chat_message(root, bound, client_message_id) if bound and bound != chat else None)
    from ouroboros.tool_access import canonical_data_root
    root = source_root or (row or {}).get("_owner_source_root") or (
        canonical_data_root(ctx) if getattr(ctx, "drive_root", None) else None)
    if quotable_owner_row(row, root, accounts=accounts):
        held_owner_messages(ctx)[owner_row_address(row)] = {
            **copy.deepcopy(row), "_owner_source_root": str(root) if root else ""}
        persist_held_state(ctx)
        return row
    return None


def persist_held_state(ctx) -> None:
    """Retain sources and return addresses on an existing canonical task record.

    The context's own maps are the turn's authority; the record is their durable copy for a
    later Continue. Maps this context already persisted open no write transaction at all (the
    write gate is the convoy's own contention point); a record that already holds them is not
    rewritten. A store too busy to
    take the write (``StoreBusy`` after a confirmed rollback: a reader convoy, never a lost
    write) must not fail the owner's turn over bookkeeping: the maps stay, the gap is logged
    once and marked, and the next persist or the loop's terminal save carries them. Other
    store errors still raise.
    """
    if not (getattr(ctx, "task_id", "") and getattr(ctx, "drive_root", None)):
        return
    from ouroboros.store import StoreBusy
    from ouroboros.task_store import patch_document
    from ouroboros.tool_access import canonical_data_root
    held = copy.deepcopy(getattr(ctx, "_held_owner_messages", None) or {})
    relays = copy.deepcopy(getattr(ctx, "_open_relays", None) or {})
    if getattr(ctx, "_held_state_persisted", None) == (held, relays) and not getattr(ctx, "_held_state_dirty", False):
        return

    def _mutate(current):
        if not current or (current.get("held_owner_messages") == held and current.get("open_relays") == relays):
            return None  # nothing to write: an unchanged record is not rewritten
        return {**current, "held_owner_messages": held, "open_relays": relays}

    try:
        patch_document(canonical_data_root(ctx), ctx.task_id, _mutate)
    except StoreBusy as exc:
        if getattr(exc, "rollback_confirmed", False) is not True:
            raise
        if not getattr(ctx, "_held_state_dirty", False):
            logging.getLogger(__name__).warning(
                "Held owner sources of %s not persisted (store busy, %s); kept in memory until the next "
                "persist or the terminal save", ctx.task_id, exc.reason)
        ctx._held_state_dirty = True
        return
    ctx._held_state_dirty = False
    ctx._held_state_persisted = (held, relays)


def flush_held_state(ctx) -> None:
    """Carry a held state a busy store refused earlier (``working_checkpoint.save_terminal``)."""
    if getattr(ctx, "_held_state_dirty", False):
        try:
            persist_held_state(ctx)
        except Exception:
            logging.getLogger(__name__).warning("Held owner sources of %s still unpersisted at the terminal save",
                                                getattr(ctx, "task_id", ""), exc_info=True)


def quote_rows(ctx, handles) -> tuple[list, str]:
    if handles == []:
        return [], ""  # An unquoted note reads no owner source or account snapshot.
    held = held_owner_messages(ctx)
    from ouroboros.owner_words import owner_place_accounts
    from ouroboros.tool_access import canonical_data_root
    root = canonical_data_root(ctx)
    accounts_by_root = {}
    current = {}
    for handle, row in held.items():
        source_root = row.get("_owner_source_root") or root
        cache_key = str(source_root)
        accounts = accounts_by_root.get(cache_key)
        if accounts is None:
            accounts = accounts_by_root[cache_key] = owner_place_accounts(source_root)
        if quotable_owner_row(row, source_root, accounts=accounts):
            current[handle] = row
    if not isinstance(handles, list) or any(not isinstance(h, str) or h not in current for h in handles):
        return [], "QUOTE_NOT_HELD: choose only held owner handles: " + json.dumps(list(current), ensure_ascii=False)
    return [owner_quote(root, current[handle], address=handle) for handle in dict.fromkeys(handles)], ""


def owner_quote(root, row: dict, *, address: str = "") -> dict:
    """One retained owner row as a quote: exact words, attachments and where/when."""
    from ouroboros.owner_words import answered_card_source
    from ouroboros.projects_registry import list_reserved_projects
    root = row.get("_owner_source_root") or root
    labels = {int(p["chat_id"]): p["name"] for p in list_reserved_projects(root)}
    chat_id = int(row.get("chat_id", 1))
    quote = {"text": owner_text(row), "attachments": copy.deepcopy(row.get("attachments") or []),
             "source": {"chat_id": chat_id, "label": labels.get(chat_id, "Main"), "ts": str(row.get("ts") or ""),
                        "address": address or owner_row_address(row),
                        "client_message_id": str(row.get("client_message_id") or "")},
             "owner_row": copy.deepcopy(row)}
    if isinstance(row.get("reply_to"), dict) and row["reply_to"]:  # what the owner was replying to, as the host resolved it
        quote["reply_to"] = copy.deepcopy(row["reply_to"])
    if isinstance(answered := answered_card_source(row)[0], dict):  # a quiz row's card or a late answer's snapshot
        from ouroboros.owner_quiz import recorded_answer_frame
        quote["question_frame"] = recorded_answer_frame(answered)
    return quote


def relay_envelope(relay_id: str, quotes: list, *, chat_id: int, summary: str = "", label: str = "Main") -> dict:
    # The singular UI shape carries the first quote; the additive list retains
    # every selected source independently (never splice different owner rows).
    # `summary` is the teller's one line for the owner, which the room shows in place of the note.
    visible = [{k: copy.deepcopy(v) for k, v in q.items() if k != "owner_row"} for q in quotes]
    first = visible[0] if visible else {"text": "", "attachments": [], "source": {
        "chat_id": chat_id, "label": label, "ts": "", "client_message_id": "", "address": ""}}
    return {"relay_id": relay_id, **first, "quotes": visible, **({"summary": summary} if summary else {})}


def relay_sender_facts(root, teller_task_id: str) -> dict:
    """Where a note was written, for the receiving room's note card: the teller's room, else its name ("" for a
    nameless work: the card says `Another work`)."""
    from ouroboros.task_results import load_task_result
    task = load_task_result(root, teller_task_id) if teller_task_id else None
    if not task:
        return {}
    facts = relay_target_facts(root, teller_task_id, task)
    name = (facts["target_titles"] or [""])[0] if facts["project_chat_id"] == 0 else facts["project_name"]
    return {"task_id": teller_task_id, "chat_id": facts["project_chat_id"], "name": name}


def _source_label(root, chat_id: int) -> str:
    """The room a note without owner quotes came from, for its card: Main, else that Project's name. A registry
    that cannot be read names no Project ("Project"): presentation never refuses the owner's words."""
    if int(chat_id) == 1:
        return "Main"
    from ouroboros.projects_registry import list_reserved_projects
    try:
        return next((str(p.get("name") or "Project") for p in list_reserved_projects(root) if p.get("chat_id") == chat_id),
                    "Main")
    except Exception:
        logging.getLogger(__name__).debug("Relay source label of chat %s unreadable", chat_id, exc_info=True)
        return "Project"


def write_relay(root, task_id: str, *, relay_id: str, note: str, quotes: list, chat_id: int,
                teller_task_id: str, canonical_root=None, sender_origin=None, client_surface=None,
                summary: str = "", host_carried=None) -> tuple[dict, bool]:
    """Mail lock: dedupe then append; caller bumps generation only on a new quote. ``host_carried`` (a dict naming
    where the row went, ``went_to``: "" for a row addressed to the receiver) marks a row the host carries to a
    member of its tree (``carry_to_members``): held and shown there, in no corpus, with no receipt and no return."""
    from ouroboros.owner_mailbox import _mailbox_path, mailbox_lines
    from ouroboros.task_custody import task_mail_lock
    from ouroboros.utils import append_jsonl, utc_now_iso
    path = _mailbox_path(Path(root), task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    label = {} if quotes else {"label": _source_label(canonical_root or root, chat_id)}
    quote_roots = {str(q.get("owner_row", {}).get("_owner_source_root") or "") for q in quotes
                   if isinstance(q, dict)}
    source_root = next(iter(quote_roots)) if len(quote_roots) == 1 and next(iter(quote_roots)) else str(canonical_root or root)
    entry = {"kind": KIND_RELAY, "msg_id": f"relay:{relay_id}:{task_id}", "ts": utc_now_iso(),
             "text": note, "relay": relay_envelope(relay_id, quotes, chat_id=chat_id, summary=summary, **label),
             "owner_rows": [q["owner_row"] for q in quotes], "source_task_id": teller_task_id,
             "relay_canonical_root": source_root}
    if client_surface:
        entry["client_surface"] = copy.deepcopy(client_surface)
    if sender_origin:
        entry["sender_origin"] = copy.deepcopy(sender_origin)
    if isinstance(host_carried, dict):
        entry["host_carried"] = dict(host_carried)
    with task_mail_lock(root, task_id) as locked:
        if not locked:
            raise OSError("relay mailbox lock unavailable")
        if path.exists():
            for line in mailbox_lines(path.read_text(encoding="utf-8")):
                prior = json.loads(line)
                if prior.get("msg_id") == entry["msg_id"]:
                    if any(prior.get(k) != entry[k] for k in ("text", "relay", "owner_rows", "source_task_id")):
                        raise ValueError("relay identity already names different words")
                    return prior, False
        if not append_jsonl(path, entry, require_lock=True, ensure_record_boundary=True):
            raise OSError("relay mailbox append failed")
    return entry, True


def carried_quote_line(quote: dict, *, address: str, fact=None) -> str:
    """The one line over an owner row's quoted words, for a relay's quote and a carried row alike: the pinned
    prefix, the time and the room the quote names (a part the host does not have is left out, never invented),
    the carriage fact of a carried row (``reached this work`` | ``went to <task>`` | ``in the project room; routed by
    a head``, for ``ROUTED_BY_HEAD``) and the row's handle."""
    source = quote.get("source") if isinstance(quote, dict) and isinstance(quote.get("source"), dict) else {}
    parts = ["Owner words quoted verbatim", str(source.get("ts") or ""), str(source.get("label") or "")]
    if isinstance(fact, dict):
        went = str(fact.get("went_to") or "")
        where = ("in the project room; routed by a head" if went == ROUTED_BY_HEAD else f"went to {went}" if went
                 else "reached this work")
        parts.append(f"carried to this work by the host ({where})")
    return "[" + "; ".join(part for part in [*parts, f"handle {address}"] if part) + "]"


def _quote_body(row: dict, line: str) -> str:
    """The quoted row under its line: its words whole, then the message it replies to, the card it answers and
    the files it carries, each as a host frame beside my human's words."""
    from ouroboros.owner_words import answered_card_source

    body = f"{line}\n{owner_text(row)}"
    if isinstance(row.get("reply_to"), dict) and row["reply_to"]:
        body += ("\n[Host reference: the message these owner words reply to; context, not the owner's words]\n"
                 + json.dumps(row["reply_to"], ensure_ascii=False))
    if isinstance(answered := answered_card_source(row)[0], dict):  # the question it answers, for the model too
        from ouroboros.owner_quiz import recorded_answer_frame
        body += "\n[Host question frame]\n" + recorded_answer_frame(answered)
    if row.get("attachments"):
        body += "\n[Host-retained attachments]\n" + json.dumps(row["attachments"], ensure_ascii=False)
    return body


def _quotes_by_row(entry: dict) -> list:
    """``(row, its quote)`` for each owner row of a relay entry (the envelope lists the quotes in the rows' order;
    an entry built without one gives an empty quote)."""
    quotes = (entry.get("relay") or {}).get("quotes") if isinstance(entry.get("relay"), dict) else None
    quotes = quotes if isinstance(quotes, list) else []
    return [(row, quotes[index] if index < len(quotes) and isinstance(quotes[index], dict) else {})
            for index, row in enumerate(entry.get("owner_rows") or [])]


def retained_relay(entry: dict, delivered: set | None = None) -> dict:
    """The internal return/ACK envelope names only injected quotes and their stores; the visible source is unchanged."""
    relay = copy.deepcopy(entry.get("relay") or {})
    rows, roots, kept = entry.get("owner_rows") or [], {}, []
    quotes = relay.get("quotes") or []
    for index, quote in enumerate(quotes):
        if delivered is not None and index not in delivered:
            continue
        kept.append(quote)
        row = rows[index] if index < len(rows) else {}
        if root := row.get("_owner_source_root") or entry.get("relay_canonical_root"):
            roots[str(quote.get("source", {}).get("address") or "")] = str(root)
    if delivered is not None and quotes:
        if not kept:
            return {}  # the peer note entered, but none of its owner quotes did
        for key in quotes[0]:
            relay.pop(key, None)
        relay.update(kept[0], quotes=kept)
    if roots:
        relay["_owner_source_roots"] = roots
    return relay


def inject_relay(ctx, entry: dict, messages: list) -> list:
    """Only exact owner rows enter the owner corpus, carried from the teller (no reply owed); the note has its
    own voice. A ``host_carried`` entry takes the carried mode (``inject_carried``) instead. Returns the addresses
    of carried rows shown here under the fact ``reached this work`` (the drain's closing control); ``[]`` otherwise."""
    from ouroboros import loop
    from ouroboros.owner_words import answered_card_reply, answered_card_source, owner_place_accounts, reply_target
    from ouroboros.tool_access import canonical_data_root

    if isinstance(entry.get("host_carried"), dict):
        return inject_carried(ctx, entry, messages)
    root = canonical_data_root(ctx) if getattr(ctx, "drive_root", None) else None
    # The teller may have read the quoted place from a different canonical data
    # root than the recipient. Each forwarded row retains its source; the
    # envelope root is the fallback for older rows, not a new source for all of them.
    fallback_root = entry.get("relay_canonical_root") or root
    accounts_by_root = {}
    delivered = set()
    for index, (row, quote) in enumerate(_quotes_by_row(entry)):
        source_root = row.get("_owner_source_root") or fallback_root
        key = str(source_root)
        if key not in accounts_by_root:
            accounts_by_root[key] = owner_place_accounts(source_root)
        accounts = accounts_by_root[key]
        # A held place row is not permanent authority: account revocation between
        # relay creation and delivery must prevent stale words from entering the
        # recipient's corpus or message frame.
        if not quotable_owner_row(row, source_root, accounts=accounts):
            continue
        retain_owner_message(ctx, row, accounts=accounts, source_root=source_root)
        address = owner_row_address(row)
        # An answer replies to its card (a quiz row's own, or a late answer's snapshot), else to its chat target.
        replied = answered_card_reply(*answered_card_source(row)) or reply_target(source_root, row.get("reply_to")) or None
        loop._record_owner_directive(ctx, source="owner_mailbox", content=owner_text(row), msg_id=address, carried=True,
                                     origin={"source_task_id": entry.get("source_task_id"),
                                             "client_message_id": row.get("client_message_id"),
                                             "chat_id": row.get("chat_id"), "ts": row.get("ts")},
                                     reply_to=replied)
        content = quote_content(ctx, entry, row, _quote_body(row, carried_quote_line(quote, address=address)))
        loop._append_or_merge_user_message(messages, content, slot=ctx, kind=OWNER_WORDS_KIND)
        delivered.add(index)
    note = str(entry.get("text") or "")
    relay = entry["relay"]
    loop._append_or_merge_user_message(messages,
        f"[Message from Ouroboros in another turn; context, relay {relay['relay_id']}]\n{note}", slot=ctx)
    # The deferred ACK describes this injection, not a later account snapshot. In particular, revocation after
    # injection cannot erase a real read, while a row refused before it earns no read or return context.
    entry["_relay_delivery"] = retained_relay(entry, delivered)
    if ctx is not None:
        relays = getattr(ctx, "_open_relays", None)
        if not isinstance(relays, dict):
            ctx._open_relays = relays = {}
        if entry["_relay_delivery"]:
            relays[relay["relay_id"]] = copy.deepcopy(entry["_relay_delivery"])
        else:
            relays.pop(relay["relay_id"], None)
        persist_held_state(ctx)
    return []


def first_request_stamps(messages: list) -> dict:
    """``{reference: source}`` of the owner-words stamps in a member's first request: everything before its first
    assistant turn, read as it was sent (``stamped_sources``). ``{}`` when it holds no stamp or cannot
    be read: every row is then new (a duplicate is the cheaper error than a loss)."""
    stamps: dict = {}
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get("role") == "assistant":
            break
        content = message.get("content")
        blocks = [content] if isinstance(content, str) else content if isinstance(content, list) else []
        for block in blocks:
            stamps.update(stamped_sources(block if isinstance(block, str) else (block or {}).get("text")
                                          if isinstance(block, (str, dict)) else ""))
    return stamps


def _stamp_parts(line: str) -> list[str]:
    """The parts of one owner-words stamp line as ``words_stamp`` prints it (``[… · owner · …]``); ``[]`` for any
    other line, a reply target's stamp among them. The one parser of a stamp for every reader of one."""
    line = line.strip()
    parts = line[1:-1].split(" · ") if line.startswith("[") and line.endswith("]") else []
    return parts if "owner" in parts[:2] else []


def stamped_sources(text: Any) -> dict[str, str]:
    """``{reference: source}`` of every owner-words stamp in a rendered text (a review subject, a brief, a member's
    first request): the last part of each stamp line, whole, so one reference is never found inside a longer one,
    and the source part the stamp names before it (``room_message``, ``owner_mailbox of task …``)."""
    stamps: dict[str, str] = {}
    for line in str(text or "").split("\n"):
        if parts := _stamp_parts(line):
            stamps[parts[-1]] = parts[parts.index("owner") + 1] if len(parts) > 2 else ""
    return stamps


def stamped_refs(text: Any) -> set:
    """The reference tokens of every owner-words stamp in a rendered text (``stamped_sources``)."""
    return set(stamped_sources(text))


def _stamp_blocks(text: str) -> list[tuple[str, str]]:
    """``(time, words)`` under every owner-words stamp line of a rendered text: the stamp's time part (``""`` when
    undated) and the lines beneath it up to the next stamp line, as a row with no reference is told
    (``words_unseen_in``)."""
    blocks: list[tuple[str, list[str]]] = []
    for line in text.split("\n"):
        if parts := _stamp_parts(line):
            blocks.append(("" if parts[0] == "owner" else parts[0], []))
        elif blocks:
            blocks[-1][1].append(line)
    return [(when, "\n".join(lines).strip()) for when, lines in blocks]


def words_unseen_in(ctx: Any, text: Any) -> list[dict[str, Any]]:
    """My human's rows of this run's set (``work_words``; never a reply target, another author's words) whose
    identity a rendered ``text`` does not hold. Identity is the stamp's reference token, matched whole, under
    either reference the row has printed (``_earlier_refs``); a row with no reference is told by its words under
    a stamp with its time (``_stamp_blocks``: the words whole at the block's head, never a substring of the text).
    Never the whole stamp: a row's source, and a referenced row's date, may change for a reader that did see it.
    A text with no stamp at all (unreadable, or rendered without words) holds no row: every row is unseen."""
    from ouroboros.owner_words import REPLY_TARGET, _earlier_refs, work_words

    rows = [row for row in work_words(ctx)[0] if row.get("source") != REPLY_TARGET]
    text = str(text or "")
    seen, earlier, blocks = stamped_refs(text), _earlier_refs(ctx), _stamp_blocks(text)

    def held(row: dict) -> bool:
        if row["ref"]:
            return row["ref"] in seen or earlier.get(row["ref"]) in seen
        words = row["text"].strip()
        return bool(words) and any(when == row["ts"] and (block == words or block.startswith(words + "\n"))
                                   for when, block in blocks)

    return [row for row in rows if not (seen and held(row))]


def inject_carried(ctx, entry: dict, messages: list) -> list:
    """The carried mode: each row is held by its address with the carriage fact beside it (persisted for a
    Continue) and shown once under the carried line, typed as my human's words; it enters no corpus, owes no
    reply, opens no return and prints no teller line. A row held under the same standing renders nothing (an
    addressed row is never downgraded by a context entry; a context row addressed since flips, and is shown); a row
    whose reference already stands in the member's first request renders no words, only, for a row addressed to
    this work that stood there under its room, the one host fact line. Attachments are staged either way (by saved
    hash, idempotent). Returns the addresses shown under ``reached this work``."""
    from ouroboros import loop
    from ouroboros.owner_words import _chat_ref

    fact_of = entry["host_carried"]
    went = str(fact_of.get("went_to") or "")
    held = held_owner_messages(ctx) if ctx is not None else {}
    stamps = first_request_stamps(messages)
    addressed = []
    for row, quote in _quotes_by_row(entry):
        source = quote.get("source") if isinstance(quote.get("source"), dict) else {}
        address = str(source.get("address") or "") or owner_row_address(row)
        prior = held.get(address)
        if isinstance(prior, dict):
            prior_fact = prior.get("carried") if isinstance(prior.get("carried"), dict) else {}
            if went or not str(prior_fact.get("went_to") or ""):
                continue  # the same standing, or addressed already: nothing more to show
        held[address] = {**copy.deepcopy(row), "carried": {"from": str(entry.get("source_task_id") or ""),
                                                            "went_to": went, "at": str(entry.get("ts") or "")}}
        refs = {address, _chat_ref(row) if row.get("client_message_id") else "", str(fact_of.get("earlier_ref") or "")}
        seen = [stamps[ref] for ref in refs - {""} if ref in stamps]
        if not seen:
            body = _quote_body(row, carried_quote_line(quote, address=address, fact={"went_to": went}))
            loop._append_or_merge_user_message(messages, quote_content(ctx, entry, row, body), slot=ctx,
                                               kind=OWNER_WORDS_KIND)
        elif row.get("attachments") and ctx is not None:
            stage_quoted(ctx, entry, row)  # its words stand in the first request; its bytes are on this drive
        if seen and (went or not any(part.startswith("room_message") for part in seen)):
            continue  # shown at birth under the standing it has now: nothing more
        if seen:
            loop._append_or_merge_user_message(
                messages, f"[Host fact: the row {address} in your starting words was addressed to this work]", slot=ctx)
        if not went:
            addressed.append(address)
    if ctx is not None:
        persist_held_state(ctx)
    return addressed


def stage_quoted(ctx, entry, row) -> list:
    """Stage a quoted row's files on the recipient's drive from their saved uploads, by their saved hashes
    (idempotent: a copy already there is kept); the manifest rows, staged or rejected under their names. The
    uploads are read from the root the row was read from (``_owner_source_root``), else the entry's root."""
    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.chat_uploads import uploads_dir, upload_id
    from ouroboros.tool_access import canonical_data_root
    root = row.get("_owner_source_root") or entry.get("relay_canonical_root") or canonical_data_root(ctx)
    return stage_task_attachments(ctx.drive_root, ctx.task_id, [
        {**ref, "path": str(uploads_dir(root) / upload_id(ref.get("upload"))) if upload_id(ref.get("upload")) else "",
         "label": str(ref.get("name") or "attachment")} for ref in row["attachments"]])


def task_words_text(drive_root: Any, task: dict, *, audience: str = "child", indent: str = "",
                    stage: bool = False, marks: Any = None) -> str:
    """A task's set rendered from its record at its start (a helper's birth), ``owner_words``'s set in its rendering:
    the field it was scheduled with and the rows its record holds, then the live rows of its room, read once here;
    without ``drive_root`` the record alone. ``stage`` (a birth): the files the rows carry are staged on the task's
    own drive and named by path.
    ``marks``: the acting marks the caller's capture holds, so the marked rows are cut from them (``_live_words``)."""
    from ouroboros.owner_words import (_deduped, _held_words, _live_words, _mapping, _room_of, owner_place_accounts,
                                       render_owner_words, render_schedule_text, task_governing_words,
                                       task_schedule_items)

    record = _mapping(task)
    meta = _mapping(record.get("metadata"))
    task_id = str(record.get("id") or record.get("task_id") or "")
    root_id = str(record.get("root_task_id") or meta.get("root_task_id") or "")
    rows, absent = task_governing_words(record)
    gaps: list[str] = []
    if drive_root is not None:
        accounts = owner_place_accounts(drive_root)
        held = record.get("held_owner_messages") or meta.get("held_owner_messages")
        live, gaps = _live_words(drive_root, _room_of(
            drive_root, task_id, record.get("parent_task_id") or meta.get("parent_task_id"), root_id), accounts, marks)
        rows = _deduped([*rows, *(row for context in (False, True) for row in _held_words(
            held, drive_root, task_id, context=context, accounts=accounts, read=True)), *live])
        if stage and task_id:
            stage_birth_files(drive_root, record, task_id, rows)
    words = render_owner_words(rows, absent, audience=audience, root_task_id=root_id, indent=indent, gaps=gaps)
    schedule = render_schedule_text(task_schedule_items(record), audience=audience, indent=indent)
    return "\n\n".join(part for part in (words, schedule) if part)


def stage_birth_files(drive_root: Any, record: dict, task_id: str, rows: list[dict[str, Any]]) -> None:
    """At a helper's birth the files its set's rows carry are staged on its own drive from their saved uploads
    (``stage_quoted``, as a relay stages at a drain: by saved hash, idempotent), and each file's line
    names the staged path it opens, or the host's reason it could not be staged; a declared helper and a
    grandchild stage the same way. A staging that fails leaves the file named as before. Never raises."""
    from types import SimpleNamespace

    from ouroboros.owner_words import _mapping
    from supervisor.queue_timeouts import _task_drive_for_task

    meta = _mapping(record.get("metadata"))
    # The record's drive fields, else the drive the mailbox writer resolves for the same task (``carry_to_members``).
    drive = str(record.get("child_drive_root") or record.get("drive_root") or meta.get("child_drive_root") or "") or (
        _task_drive_for_task(dict(record), task_id, canonical=drive_root) if drive_root is not None else "")
    ctx = SimpleNamespace(drive_root=Path(drive) if drive else drive_root, task_id=task_id)
    for row in rows:
        if not row.get("attachments"):
            continue
        try:
            manifest = stage_quoted(ctx, {"relay_canonical_root": str(drive_root)}, row)
        except Exception as exc:  # the file stays named; the birth block is never refused over a copy
            logging.getLogger(__name__).warning("owner words: files of a row not staged at the birth of %s (%s: %s)",
                                                task_id, type(exc).__name__, exc)
            continue
        for ref, item in zip(row["attachments"], manifest if isinstance(manifest, list) else []):
            item = _mapping(item)
            ref["staged"] = (f"read_file(root='artifact_store', path='{item.get('relpath')}')"
                             if item.get("status") == "staged" else f"not staged: {item.get('reason') or 'unknown'}")


def quote_content(ctx, entry, row, body):
    """The quoted row's content for the model: its body and, for a row with files, their copies staged here."""
    if not row.get("attachments") or ctx is None:
        return body
    from ouroboros.artifacts import attachment_manifest_projection
    from ouroboros.context import build_incoming_user_content
    from ouroboros.gateway.tasks import _render_attachment_lines
    manifest = stage_quoted(ctx, entry, row)
    body += "\n[Host attachment delivery]\n" + _render_attachment_lines(
        attachment_manifest_projection(ctx.drive_root, ctx.task_id, manifest))
    return build_incoming_user_content({"text": body, "attachment_manifest": manifest}, ctx.drive_root, ctx.task_id)


def quoted_inputs(root, refs: list) -> list:
    """The owner files a relay's quoted rows carried (their saved chat uploads, which ``quote_content`` stages on
    delivery), as rows another task's own copy reads (``artifacts.materialize_inherited_attachment_manifest``): each
    upload verified against its saved identity and named by its unique upload id, so two same-named files stay two. One
    gone or changed is a rejected ``source_missing`` row under its name, never other bytes; a secret-named one is
    refused as staging refuses it (``secret_source``)."""
    import re

    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.chat_uploads import file_media, upload_id, uploads_dir
    from ouroboros.workspace_patch_rules import _sensitive_untracked_reason

    rows = []
    for ordinal, ref in enumerate(refs):
        uid = upload_id(ref.get("upload"))
        source_root = ref.get("_owner_source_root") or root
        path, original = uploads_dir(source_root) / uid, re.sub(r"^[0-9a-f]{32}_", "", uid)
        row = {"ordinal": ordinal, "label": str(ref.get("name") or "attachment"),
               **{key: ref[key] for key in ("mime", "size", "sha256") if key in ref}}
        if uid and (reason := _sensitive_untracked_reason(original)):
            rows.append({**row, "status": "rejected", "reason": "secret_source",
                         "rule": f"uploaded file name {original!r}: {reason}"})
            continue
        try:
            if not uid:
                raise OSError("the quoted row names no saved upload")
            stream_artifact_file(path, expected=ref)
            mime, kind = file_media(path)
        except OSError:
            rows.append({**row, "status": "rejected", "reason": "source_missing"})
            continue
        rows.append({**row, "status": "staged", "root": "artifact_store", "relpath": f"attachments/{uid}",
                     "abs_path": str(path), "mime": mime, "is_image": kind == "image"})
    return rows


def select_return(ctx, relay_id: str) -> tuple[dict, str]:
    relays = getattr(ctx, "_open_relays", None) or {}
    if relay_id and relay_id in relays:
        return relays[relay_id], ""
    if not relay_id and len(relays) == 1:
        return next(iter(relays.values())), ""
    return {}, "RELAY_RETURN_UNRESOLVED: open relay ids: " + json.dumps(list(relays), ensure_ascii=False)


def acknowledge_relay(root, target: str, entry: dict) -> None:
    """Called only after the mailbox's durable content acknowledgement."""
    from ouroboros.project_dialogue import append_chat_annotation
    relay = entry["_relay_delivery"] if "_relay_delivery" in entry else retained_relay(entry)
    for quote in relay.get("quotes") or []:
        message_id = quote["source"].get("client_message_id")
        if message_id:
            source_root = relay.get("_owner_source_roots", {}).get(quote["source"].get("address", "")) or root
            append_chat_annotation(source_root, message_id, action="steer_task", target=target, status="read",
                                   routing_token=entry["relay"]["relay_id"],
                                   **relay_target_facts(root, target))


def work_display_name(task) -> str:
    """The work's name as the UI shows it (room_work.js: its suggested name, then its title); never its id or its text."""
    return str((task or {}).get("suggested_name") or (task or {}).get("title") or "").strip()


def relay_target_facts(root, target: str, task=None) -> dict:
    from ouroboros.projects_registry import project_chat_for_task, list_reserved_projects
    from ouroboros.task_results import load_task_result
    task = task or load_task_result(root, target) or {}
    chat_id = project_chat_for_task(root, target) or task.get("chat_id", 1)
    project = next((p for p in list_reserved_projects(root) if p["chat_id"] == chat_id), {})
    name = work_display_name(task)  # never the work's text or id: a nameless work is the receipt's `the work`
    return {"project_id": str(project.get("id") or ""), "project_name": str(project.get("name") or ("Main" if chat_id else "Hidden")),
            "project_chat_id": int(chat_id), "target_task_ids": [target], "target_titles": [name] if name else []}


def publish_relay_ack(root, target: str, relay: dict, bridge) -> None:
    """The receiver's saved ACK updates the same source annotation live, with the row as saved (#968),
    never a second read of the work. A read row the sidecar could not take still goes out from the
    work's facts, as before: the mailbox ACK is the authority."""
    if bridge is None:
        return
    from ouroboros.project_dialogue import ROUTING_RECEIPT_FIELDS, chat_annotation_receipt
    for quote in relay.get("quotes") or []:
        source = quote["source"]
        if source.get("client_message_id"):
            try:
                source_root = relay.get("_owner_source_roots", {}).get(source.get("address", "")) or root
                saved = chat_annotation_receipt(source_root, source["client_message_id"], relay["relay_id"])
            except OSError:  # an unreadable sidecar sends the ack from the work's facts, as before
                saved = {}
            fields = ({key: saved[key] for key in ROUTING_RECEIPT_FIELDS if key in saved} if saved.get("status") == "read"
                      else {"action": "steer_task", "target": target, "status": "read",
                            "routing_token": relay["relay_id"], **relay_target_facts(root, target)})
            bridge.send_routing_ack(source["chat_id"], client_message_id=source["client_message_id"], **fields)


def publish_relay(root, target: str, entry: dict, bridge, task=None) -> None:
    """One target chat row, even after an append-before-receipt crash. The delivery is the quoted rows' receipt:
    they ride on down the target's tree at once (``carry_relay_at_receipt``)."""
    if entry.get("owner_rows") and not entry.get("host_carried"):
        carry_relay_at_receipt(root, target, entry)
    from supervisor.message_ingress import _INGRESS_LOCK
    from supervisor.message_bus import log_chat
    from ouroboros import dialogue_store
    from ouroboros.chat_references import projection_fields
    from ouroboros.task_results import load_task_result
    task = task or load_task_result(root, target) or {}
    facts = relay_target_facts(root, target, task)
    with _INGRESS_LOCK:
        # The saved row of this relay id, by its indexed client id: an append-before-receipt crash
        # finds it here (a relay saved before the upgrade, once its archive segment is imported).
        stored = next((dict(found.row) for found in dialogue_store.rows_by_client_id(root, entry["msg_id"])
                       if found.row.get("type") == KIND_RELAY), None)
        if stored is None:
            # The saved row also names the room the note came from and the work it was written for, by the name
            # the UI showed then (the mailbox entry keeps its identity bytes; Main's note card reads the receiver
            # from it, and says `this work` for a work without a readable name).
            sender = relay_sender_facts(root, str(entry.get("source_task_id") or ""))
            title = work_display_name(task)
            receiver = {"task_id": target, **({"title": title} if title else {})}
            stored = log_chat("out", facts["project_chat_id"], 0, entry["text"], ts=entry["ts"],
                              task_id=target, record_type=KIND_RELAY, client_message_id=entry["msg_id"],
                              message_meta={"relay": {**entry["relay"], **({"sender": sender} if sender else {}),
                                                      "target": receiver}},
                              drive_root=root, require_write=True)
    if bridge is not None and stored["chat_id"]:
        try:
            bridge.send_message(stored["chat_id"], stored["text"], task_id=target, ts=stored["ts"],
                                system_type=KIND_RELAY,
                                progress_meta={"kind": KIND_RELAY, **projection_fields(stored)})
        except Exception:
            logging.getLogger(__name__).warning("Relay %s saved; live publication failed", entry["msg_id"], exc_info=True)


# --- carriage: my human's rows reach every live member of the tree -------------------------------------------

def _words_identity(row: dict) -> Any:
    """A row of the set told by its reference, else by its words and its time (``owner_words._deduped``)."""
    from ouroboros.project_dialogue import _text_sha256
    return row["ref"] or ("words", _text_sha256(row["text"]), str(row.get("ts") or ""))


def _pristine_form(ctx, row: dict) -> tuple:
    """``(address, chat row)`` for a row of the set this run holds no chat row for (a direct turn's input): its
    words, time, room and client id as the corpus recorded them, in the shape a quote is built from."""
    ref, client = str(row.get("ref") or ""), ""
    chat_id = int(getattr(ctx, "current_chat_id", None) or 1)
    if ref.startswith("chat "):
        head, _sep, client = ref[5:].partition(" / ")
        chat_id = int(head) if head.strip().isdigit() else chat_id
    elif ref.startswith("message "):
        client = ref[8:]
    pristine = {"direction": "in", "chat_id": chat_id, "ts": str(row.get("ts") or ""), "text": row["text"],
                **({"client_message_id": client} if client else {}),
                **({"attachments": copy.deepcopy(row["attachments"])} if row.get("attachments") else {})}
    return format_address(row_address(pristine)), pristine


def _rows_to_carry(ctx, root) -> list:
    """``(identities, address, quote, went_to, earlier reference)`` for each row of my human's this run carries
    down: the tree's rows and the rows it holds as context (``owner_words``), never a reply target (another
    author's words ride inside its row's quote). The quote is built from the chat row the run holds for it (its
    carriage fact stripped), else from the row's recorded form; a row's identities are its printed references
    (now and before its room rode with it) and its address."""
    from ouroboros.owner_words import REPLY_TARGET, _chat_ref, _deduped, _earlier_refs, _held_words, _tree_words

    held = held_owner_messages(ctx)
    by_ref = {}
    for key, value in held.items():
        if isinstance(value, dict):
            fact = value.get("carried") if isinstance(value.get("carried"), dict) else {}
            by_ref[_chat_ref(value) if value.get("client_message_id") else str(key)] = (
                str(key), {k: v for k, v in value.items() if k != "carried"}, str(fact.get("went_to") or ""))
    rows, _root, accounts = _tree_words(ctx)
    rows = _deduped(rows + _held_words(held, root, str(ctx.task_id), context=True, accounts=accounts))
    earlier, carried = _earlier_refs(ctx), []
    for row in rows:
        if row["source"] == REPLY_TARGET:
            continue
        address, pristine, went = by_ref.get(row["ref"]) or (*_pristine_form(ctx, row), "")
        before = earlier.get(row["ref"], "") if row["ref"] else ""
        identities = {_words_identity(row), address, before} - {""}
        carried.append((identities, address, owner_quote(root, pristine, address=address), went, before))
    return carried


def _member_has(drive, task_id: str, member: dict) -> tuple:
    """What a member already holds, read once: the identities of its inherited field and the ids of its mailbox
    (append-only until its settlement; an id counts whatever its bytes)."""
    from ouroboros.owner_mailbox import _mail_text, _mailbox_path, mailbox_lines
    from ouroboros.owner_words import REPLY_TARGET, task_governing_words

    path, ids = _mailbox_path(Path(drive), task_id), set()
    if path.exists():
        for line in mailbox_lines(_mail_text(path)):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                ids.add(str(parsed.get("msg_id") or ""))
    return {_words_identity(row) for row in task_governing_words(member)[0] if row["source"] != REPLY_TARGET}, ids


def _carry_to_member(root, drive, task_id: str, member: dict, rows: list, teller: str) -> None:
    """One ``host_carried`` entry per row the member holds under none of its references: ``carry:<address>`` for a
    row addressed to this work, ``carry-context:<address>`` for a row held as context; an addressed entry stands
    for both (a context row the head then addressed is written once more, and the member's fact flips)."""
    field, ids = _member_has(drive, task_id, member)
    for identities, address, quote, went, before in rows:
        if identities & field or f"relay:carry:{address}:{task_id}" in ids or (
                went and f"relay:carry-context:{address}:{task_id}" in ids):
            continue
        write_relay(drive, task_id, relay_id=f"{'carry-context' if went else 'carry'}:{address}", note="",
                    quotes=[quote], chat_id=int(quote["source"]["chat_id"]), teller_task_id=teller,
                    canonical_root=root, host_carried={"went_to": went, **({"earlier_ref": before} if before else {})})


def carry_to_members(ctx, *, delivered: bool) -> None:
    """Reconcile my human's rows with every live member below this run at the end of a drain, the catch-up of the
    host's carriage at receipt (``carry_at_receipt``): the tree's members for a root, the direct children for a
    helper; a member whose record and mailbox hold a row under none of its references gets it once
    (``_carry_to_member``), teller the tree's root, so two drains, and the host before them, write identical bytes.
    No delivery state: the only memory is a skip memo. A helper's quiet drain (nothing delivered, no helper
    scheduled since its last pass, no failed write, not the attempt's first) reads no member; a root's drain
    reads its census and skips the members' reads when the live set is unchanged and the drain was quiet. A
    write that fails for one member stops no other; the next drain writes what is missing, exactly once."""
    if ctx is None or not getattr(ctx, "drive_root", None) or not getattr(ctx, "task_id", ""):
        return
    log = logging.getLogger(__name__)
    metadata = getattr(ctx, "task_metadata", None) or {}
    task_id, helper = str(ctx.task_id), metadata.get("delegation_role") == "subagent"
    memo = getattr(ctx, "_carriage_memo", None)
    if not isinstance(memo, dict):  # ``attempt`` starts as no attempt's value: the attempt's first pass is due
        memo = ctx._carriage_memo = {"attempt": (), "members": None, "failed": False, "scheduled": 0}
    scheduled, attempt = len(getattr(ctx, "_last_scheduled_subagents", None) or []), getattr(ctx, "task_attempt", None)
    due = delivered or memo["failed"] or memo["attempt"] != attempt or scheduled != memo["scheduled"]
    if helper and not due:
        return
    from ouroboros.task_status import FINAL_STATUSES, find_child_tasks
    from ouroboros.tool_access import canonical_data_root
    from supervisor.queue_timeouts import _task_drive_for_task

    root = canonical_data_root(ctx)
    memo.update(attempt=attempt, scheduled=scheduled)
    try:
        census = (find_child_tasks(root, parent_task_id=task_id, scope="direct", materialize_artifacts=False) if helper
                  else find_child_tasks(root, root_task_id=str(metadata.get("root_task_id") or task_id),
                                        exclude_task_id=task_id, materialize_artifacts=False))
        live = {str(row.get("task_id") or ""): row for row in census
                if str(row.get("task_id") or "") and str(row.get("status") or "") not in FINAL_STATUSES}
        rows = _rows_to_carry(ctx, root) if live and (due or set(live) != memo["members"]) else []
        if not rows:
            memo.update(members=set(live), failed=False)
            return
    except Exception:
        log.warning("Carriage: the members of %s could not be read; the next drain reads them", task_id, exc_info=True)
        memo["failed"] = True
        return
    failed = False
    for tid, member in live.items():
        try:
            _carry_to_member(root, _task_drive_for_task(member, tid, canonical=root), tid, member, rows,
                             str(metadata.get("root_task_id") or task_id))
        except Exception:
            log.warning("Carriage to member %s of %s failed; the next drain writes it", tid, task_id, exc_info=True)
            failed = True
    memo.update(members=set(live), failed=failed)


def carry_at_receipt(drive_root, work_id: str, quotes: list, *, went_to: str = "") -> Any:
    """The host carries my human's rows a work just received down its tree at receipt (inv. 26), outside every
    focus's turn and seat: each live member of the tree gets each row in its mail now (``_carry_to_member``, teller
    the work: the bytes its drain's catch-up would write, a sleeping member woken as by any relay), and each open
    session of the tree a frame (``delegate_interactions.place_at_receipt``). ``quotes``: each row's quote as the
    work holds the row; ``went_to``: where a context row went. Returns the placement thread, or None. Never raises:
    what fails here, the work's drain and its starters' round tops carry."""
    from ouroboros.owner_words import _chat_ref
    from ouroboros.task_status import FINAL_STATUSES, find_child_tasks
    from supervisor.queue_timeouts import _task_drive_for_task

    log = logging.getLogger(__name__)
    rows = [({address, _chat_ref(quote["owner_row"]) if quote["owner_row"].get("client_message_id") else ""} - {""},
             address, quote, went_to, "") for quote in quotes for address in [quote["source"]["address"]]]
    try:
        census = find_child_tasks(drive_root, root_task_id=work_id, exclude_task_id=work_id, materialize_artifacts=False)
    except Exception:
        log.warning("Carriage at receipt: the members of %s could not be read; its drain carries", work_id, exc_info=True)
        census = []
    for member in census:
        tid = str(member.get("task_id") or "")
        if tid and str(member.get("status") or "") not in FINAL_STATUSES:
            try:
                _carry_to_member(drive_root, _task_drive_for_task(member, tid, canonical=drive_root), tid, member, rows,
                                 work_id)
            except Exception:
                log.warning("Carriage at receipt to %s of %s failed; the next drain writes it", tid, work_id, exc_info=True)
    from ouroboros.delegate_interactions import place_at_receipt
    return place_at_receipt(drive_root, work_id, rows) if rows else None


def carry_row_at_receipt(drive_root, work_id: str, chat_id: int, client_message_id: str) -> Any:
    """``carry_at_receipt`` of the accepted chat row the room delivered to ``work_id`` (``server_owner_routing``),
    quoted as the work's drain will hold it (``retain_owner_message``: beside its source root)."""
    from supervisor.message_ingress import accepted_chat_message

    try:
        row = accepted_chat_message(drive_root, int(chat_id), str(client_message_id or ""))
        if not quotable_owner_row(row, drive_root):
            return None
        held = {**row, "_owner_source_root": str(Path(drive_root).resolve(strict=False))}
        quote = owner_quote(drive_root, held, address=owner_row_address(row))
    except Exception:
        logging.getLogger(__name__).warning("Carriage at receipt: row %s of room %s unreadable", client_message_id,
                                            chat_id, exc_info=True)
        return None
    return carry_at_receipt(drive_root, work_id, [quote])


def carry_relay_at_receipt(root, target: str, entry: dict) -> Any:
    """``carry_at_receipt`` of a relay's quoted rows still quotable now (a teller's ``tell_task``), each quoted as
    the target's drain will hold it (``inject_relay``: beside the root it was read from)."""
    quotes = []
    try:
        for row, _quote in _quotes_by_row(entry):
            source_root = row.get("_owner_source_root") or entry.get("relay_canonical_root") or root
            if quotable_owner_row(row, source_root):
                held = {**copy.deepcopy(row), "_owner_source_root": str(source_root)}
                quotes.append(owner_quote(root, held, address=owner_row_address(row)))
    except Exception:
        logging.getLogger(__name__).warning("Carriage at receipt: relay rows to %s unreadable", target, exc_info=True)
        return None
    return carry_at_receipt(root, target, quotes) if quotes else None


def carry_room_context(drive_root, chat_id: int, client_message_id: str, roots: list, went_to: str) -> None:
    """At a Project room's ingress, after the routing outcome: one carried context entry of the accepted chat row
    to every other live root of the room (``addressable_roots``), naming where the row went (the work it was
    delivered to, else ``ROUTED_BY_HEAD``: a head of the room routed it), and down each root's tree at once
    (``carry_at_receipt``; its drain is the catch-up). A write that fails stops no other root and never the message."""
    from supervisor.message_ingress import accepted_chat_message
    from supervisor.queue_timeouts import _task_drive_for_task

    log = logging.getLogger(__name__)
    try:
        row = accepted_chat_message(drive_root, int(chat_id), str(client_message_id or ""))
        if not quotable_owner_row(row, drive_root):
            return
        address = owner_row_address(row)
        quote = owner_quote(drive_root, row, address=address)
    except Exception:
        log.warning("Carriage: room %s row %s unreadable at ingress", chat_id, client_message_id, exc_info=True)
        return
    for root in roots:
        tid = str((root or {}).get("task_id") or "")
        if not tid or tid == went_to:
            continue
        try:
            write_relay(_task_drive_for_task(root, tid, canonical=drive_root), tid, relay_id=f"carry-context:{address}",
                        note="", quotes=[quote], chat_id=int(chat_id), teller_task_id=tid, canonical_root=drive_root,
                        host_carried={"went_to": went_to or ROUTED_BY_HEAD})
        except Exception:
            log.warning("Carriage: room %s row %s not written to root %s", chat_id, address, tid, exc_info=True)
            continue
        carry_at_receipt(drive_root, tid, [quote], went_to=went_to or ROUTED_BY_HEAD)
