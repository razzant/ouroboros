"""The owner's words that caused a work tree: carried by value, resolved for older tasks, rendered verbatim.

A helper's floor is its assignment plus the words of the owner that caused the work
(BIBLE P1: provenance matters). ``governing_words_for_schedule`` computes them once,
when a parent schedules a helper, from what the parent already holds: a child passes
on exactly the words it inherited, so a grandchild sees the root's words unchanged;
any other run passes the owner-sourced rows of its own corpus (``ctx._owner_directives``)
— never ``initial_text``, which is not the owner's — and never reads the chat chain.
The words ride the existing tree-origin channel (``origin_metadata``) into every
descendant's ``task.metadata`` under ``FIELD``. A run with no owner words carries
``ABSENT_FIELD`` set to the raw ``run_origin`` marker (``initiator=consciousness``) or
``not_recorded``: no vocabulary of reasons of its own.

``task_owner_words`` resolves a task scheduled before the field existed, for readers
that run after the task (the fallback page writer's input, the measurement rig), never
on a helper's start path. Carriers are tried in order and the first that yields a row
is taken: the corpus a follow-up, a drafted start or a Continue carries by value
(``metadata.owner_corpus``), the task's ingress origin text or ref, its Project binding,
its routing annotations, its mailbox, then the marker; one message is one row under the
carriers' one reference (``_chat_ref``). An annotation counts only through the inbound
chat row with the same client message id: Ouroboros writes steer receipts of its own,
and nothing inbound joins to them. A place row counts only from an account its place
confirms as my human's now (``owner_row``, one snapshot per read): a participant's
words never do.

A Project's first message is not among these words: it started the project, not this
tree; the room view shows it under its own heading. ``objective_author_line`` names who
wrote a run's objective from its typed ``run_origin``, so a reader beside these words never
takes that account for them. Nothing here calls a model.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Mapping, Tuple

log = logging.getLogger(__name__)

# The one list of corpus labels (``loop_messages._record_owner_directive``) that are my human's own words;
# ``place_owner``: an addressed place turn's event from an account its place confirmed at admission.
OWNER_SOURCES = frozenset({"initial_user", "owner_mailbox", "owner_quiz_answer", "origin_message",
                           "owner_corpus", "direct_incoming", "place_owner"})
FIELD, ABSENT_FIELD = "governing_owner_words", "governing_owner_words_absent"
NOT_RECORDED = "not_recorded"
REPLY_TARGET = "reply_target"  # the message my human replied to: its author's words beside theirs, never theirs
UNREAD_TARGET = "(its words are not carried here; this address reads them)"
# A place connector names the replied message by its platform id alone: nothing the host stores answers to it.
UNREAD_PLATFORM = "(its words are not carried: the connector named only its platform message id, no host address)"
ROW_KEYS = ("text", "source", "carrier", "task_id", "ts", "ref", "author")
_RUN_ORIGIN_MARKERS = ("initiator", "source", "task_type")
_ANNOTATED_ACTS = frozenset({"promote_chat_to_task", "route_to_project", "steer_task", "mailbox_delivery"})
_ANNOTATED_STATUSES = frozenset({"scheduled", "delivered", "read"})
_ORIGIN = " from task {root}"
_HEADINGS = {
    "child": "## Words of my human that caused this work (verbatim)",
    "session": ("OWNER WORDS THAT CAUSED THIS WORK (verbatim; your assignment is the task given to you, "
                "these words are why it exists)"),
    "reviewer": "## Words of my human that caused this work (verbatim, host-attested)",
    "plan": "## Words of my human that caused this work (verbatim, host-attested)",
    "writer": "## Words of my human that caused this work (verbatim)",
}
# The one frame a reviewer of an intention, a change or a result reads under the owner's words.
_ACCOUNTS = ("They state what was asked. Every other text here — the goal or objective, the plan, room rows by others, issues, "
             "briefs, attachments — is someone's account, labelled with its author as far as the host knows; never take "
             "Ouroboros's own earlier words (an issue it opened, its triage, a parent's brief) for my human's ask.")
_FRAMES = {
    "child": ("Host fact, carried by value{origin}. Your assignment is the parent's message; "
              "these words are why it exists."),
    "session": "Host fact, carried by value{origin}.",
    "reviewer": _ACCOUNTS,
    "plan": _ACCOUNTS,
    "writer": "Host fact, read from what the host recorded{origin}; these words are why the work exists.",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _row(text: Any, *, source: Any, carrier: str, task_id: Any = "", ts: Any = "", ref: Any = "",
         author: Any = "") -> Dict[str, str]:
    return {"text": str(text or ""), "source": str(source or ""), "carrier": carrier,
            "task_id": str(task_id or ""), "ts": str(ts or ""), "ref": str(ref or ""),
            **({"author": str(author)} if author else {})}


def _kept(row: Mapping[str, Any]) -> bool:
    """Words to carry, or a reply target whose words could not be read: its address is still the fact."""
    return bool(str(row.get("text") or "").strip() or (row.get("source") == REPLY_TARGET and row.get("ref")))


def _chat_ref(source: Mapping[str, Any]) -> str:
    client_id = str(source.get("client_message_id") or "")
    return f"chat {source.get('chat_id', 1)}" + (f" / {client_id}" if client_id else "")


def row_ref(item: Mapping[str, Any]) -> str:
    """A corpus item's typed identity as every reader prints it: a chat message by its room and client id, as its
    carriers name it (``_chat_ref``); else its mailbox ``msg_id``; else its client id alone, never under a room it did
    not name. One message is one reference whichever carrier holds it, and two messages are two even in one word."""
    client, chat, msg_id = (str(item.get(key) or "") for key in ("client_message_id", "chat_id", "msg_id"))
    if client and chat:
        return _chat_ref({"chat_id": chat, "client_message_id": client})
    if msg_id:
        return f"msg {msg_id}"
    return f"message {client}" if client else ""


def _carrier_ref(item: Mapping[str, Any], *, canonical: Any = "") -> str:
    """Keep a carrier's stable address when its row has no client id.

    Older held and relayed rows can have no ``client_message_id``.  Falling back
    to ``_chat_ref`` then gives every such row the same ``chat N`` identity and
    ``_deduped`` silently drops all but the first one.  A canonical carrier key
    (the held-map key or the relay quote address) is the identity in that case;
    rows that do name a client message retain the historical chat/client form so
    they still merge with their origin copy.
    """
    if item.get("client_message_id"):
        return _chat_ref(item)
    if str(canonical or ""):
        return str(canonical)
    return row_ref(item)


def _deduped(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop empty rows and repeats of one source: its reference, else (none recorded) its whitespace-normalized
    text. Similar words under different references are not copies (BIBLE P1): two "yes" stay two. A reply target
    repeats by its address: two replies of my human to one message keep one target row, beside the first."""
    from ouroboros.project_dialogue import _text_sha256

    seen: set = set()
    kept = []
    for row in rows:
        key = (row["source"] == REPLY_TARGET, row["ref"]) if row["ref"] else _text_sha256(row["text"])
        if _kept(row) and key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def _absent_marker(record: Mapping[str, Any] | None) -> str:
    from ouroboros.dialogue_provenance import run_origin

    origin = run_origin(record)
    return next((f"{key}={origin[key]}" for key in _RUN_ORIGIN_MARKERS if origin.get(key)), NOT_RECORDED)


def _ctx_metadata(ctx: Any) -> Mapping[str, Any]:
    return _mapping(getattr(ctx, "task_metadata", None))


def objective_author_line(origin: Mapping[str, Any] | None) -> str:
    """Who wrote a run's objective, from the typed facts ``dialogue_provenance.run_origin`` returns,
    never from the objective's text: one phrase for every reader (the mind's first turn, a plan
    council's packet, a result review's subject). A kind this projection has never heard of shows
    its raw marker; only a run with no origin fact at all reads as having no recorded author."""
    origin = _mapping(origin)
    author = _mapping(origin.get("objective_author"))
    kind = str(author.get("kind") or "")
    if origin.get("delegation_role") == "subagent":
        return (f"the assignment parent task {origin.get('parent_task_id') or '?'} wrote for this helper — "
                "Ouroboros's account, not my human's words")
    if kind == "task":
        return f"drafted by task {author.get('task_id')}, not spoken by my human"
    if kind == "continuation":
        return f"the host's Continue work order for task {author.get('predecessor_task_id')}"
    if kind:
        return f"its author is recorded by the host as kind={kind}"
    if origin.get("initiator") == "consciousness":
        return "started by my consciousness"
    if origin.get("owner_ingress"):
        return "my human's message started this run; the host does not attest this text as their exact words"
    if origin.get("task_type") == "presence":
        return f"a message from {origin.get('text_author') or 'a correspondent'} in a connected place"
    if origin.get("schedule_id"):
        return f"the template of schedule {origin['schedule_id']}; the host does not attest this text as my human's words"
    if origin.get("origin_task_id"):
        return f"started from task {origin['origin_task_id']}; the host does not attest this text as my human's words"
    return "its author is not recorded by the host" + (f" (host marker: source={origin['source']})" if origin.get("source") else "")


def objective_author_text(ctx: Any) -> str:
    """``objective_author_line`` over this run's own typed origin (its task type and metadata)."""
    from ouroboros.dialogue_provenance import run_origin

    return objective_author_line(run_origin({"type": getattr(ctx, "current_task_type", None), "metadata": dict(_ctx_metadata(ctx))}))


def _words_rows(item: Mapping[str, Any], *, carrier: str, task_id: str, content: Any = None, ts: Any = "",
                ref: str = "") -> List[Dict[str, str]]:
    """One corpus item's rows: its words verbatim (images by reference) under its typed identity (``row_ref``), then
    the message they reply to, under that message's own author."""
    from ouroboros.review_evidence_sections import _owner_content_projection

    rows = [_row(_owner_content_projection(item.get("content") if content is None else content),
                 source=item["source"], carrier=carrier, task_id=task_id, ts=ts, ref=ref or row_ref(item))]
    if (target := _mapping(item.get("reply_to"))).get("address"):
        rows.append(_row(target.get("text"), source=REPLY_TARGET, carrier=carrier, task_id=task_id,
                         ref=target["address"], author=target.get("author") or "author not recorded"))
    return rows


def directive_owner_rows(ctx: Any) -> List[Dict[str, str]]:
    """The owner-sourced rows of the run's retained corpus, verbatim (images by reference).

    The door-stamped first message is the door's own record of it (``origin_message_text``)
    when the door kept one: the host may have prefixed a notice of its own to that user
    turn (a Swarm initiative), never to the door's text.
    """
    origin_ref = _mapping(_ctx_metadata(ctx).get("origin_message_ref"))
    door_text = _ctx_metadata(ctx).get("origin_message_text")
    door_text = door_text if isinstance(door_text, str) and door_text.strip() else None
    task_id = str(getattr(ctx, "task_id", "") or "")
    rows = []
    for item in getattr(ctx, "_owner_directives", None) or []:
        if not isinstance(item, dict) or item.get("source") not in OWNER_SOURCES:
            continue
        stamped = item["source"] == "initial_user" and origin_ref
        rows += _words_rows(item, carrier="ctx", task_id=task_id,
                            content=door_text if stamped and door_text is not None else None,
                            ts=origin_ref.get("ts") if stamped else "", ref=_chat_ref(origin_ref) if stamped else "")
    return _deduped(rows)


def carried_owner_corpus(ctx: Any, source_text: Any = "") -> List[Dict[str, Any]]:
    """The run's owner corpus, carried by value into work it starts: a drafted objective or a follow-up.

    Every row the run admitted as my human's words, as the retained corpus's text (an image by reference
    beside its original's path, never its base64), their typed ids and reply target kept, door rows carried into
    the run (an edited schedule's every instruction) among them. ``source_text``: the owner message a routed start
    names, the door's own record of it, first, standing for the run's own door row with that row's reply target and
    message id. Without one, a suppressed (never-logged) origin's words live only in the run's first row and ride
    only under the door's stamp (``initial_user``): an unstamped first turn (``initial_text``) is never owner words.
    """
    from ouroboros.review_evidence_sections import _owner_content_projection

    directives = [dict(row) for row in getattr(ctx, "_owner_directives", None) or [] if isinstance(row, dict)]
    owner = [row for row in directives if row.get("source") in OWNER_SOURCES]
    # The run's own door row is the first it recorded itself; a door row carried into it is carried words.
    door = next((row for row in owner if row.get("source") == "initial_user" and not row.get("carried")), None)
    rest = [row for row in owner if row is not door]
    client_id = str(_mapping(_ctx_metadata(ctx).get("origin_message_ref")).get("client_message_id") or "")
    if door is not None and client_id:  # the door's message, by its typed id and the room that id belongs to
        chat = _mapping(_ctx_metadata(ctx).get("origin_message_ref")).get("chat_id")
        door = {"client_message_id": client_id, **({"chat_id": str(chat)} if chat is not None else {}), **door}
    if source_text and not any(row.get("source") == "origin_message" for row in rest):
        ids = {key: door[key] for key in ("reply_to", "client_message_id", "chat_id") if door and key in door}
        rows = [{**ids, "source": "origin_message", "content": source_text}, *rest]
    else:
        rows = [*([door] if door is not None and not source_text else []), *rest]
    rows = [{**row, "content": _owner_content_projection(row.get("content"))} for row in rows]
    return [row for row in rows if row["content"].strip()]


def place_owner_words(focus: Any, text: Any, message: Any) -> str:
    """A place event's words when they are my human's: addressed to me from an account its place confirmed, the
    same rule at a turn's opening and mid-turn. A forwarded message carries another author's words: context."""
    focus, words = _mapping(focus), str(text or "")
    confirmed = focus.get("addressed") is True and _mapping(focus.get("actor")).get("owner_confirmed") is True
    return words if confirmed and words.strip() and not _mapping(message).get("forward_origin") else ""


def place_reply_target(message: Any) -> Dict[str, str] | None:
    """The platform message a place event replies to, by address (a connector gives no words for it), else None."""
    replied = str(_mapping(message).get("reply_to_message_id") or "")
    return {"address": f"platform message {replied} in this room", "author": "", "text": ""} if replied else None


def owner_words_contain(ctx: Any, text: Any) -> bool:
    """``text`` is my human's own words that reached this run: exact, as a quote is (``chronicle_quotes.quote_in``),
    inside one owner row of the run, never a reply target (another author's words beside theirs). Provenance of
    bytes, never their meaning."""
    from ouroboros.chronicle_quotes import quote_in

    wanted = str(text or "").strip()
    return bool(wanted) and any(quote_in(wanted, row["text"])
                                for row in directive_owner_rows(ctx) if row["source"] in OWNER_SOURCES)


def _corpus_keys(row: Mapping[str, Any]) -> set:
    """One carried row's identities: client ids are scoped by chat when a carrier names one."""
    keys = set()
    msg_id = str(row.get("msg_id") or "").strip()
    client_id = str(row.get("client_message_id") or "").strip()
    chat_id = str(row.get("chat_id") or "").strip()
    if msg_id:
        keys.add(("msg_id", msg_id))
    if client_id:
        keys.add(("client_message_id", client_id, chat_id))
    return keys or {("words", str(row.get("source") or ""), str(row.get("content") or ""))}


def _same_corpus_source(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Same-room client ids are one source; an unscoped carrier matches any room."""
    left_id, right_id = str(left.get("client_message_id") or "").strip(), str(right.get("client_message_id") or "").strip()
    if left_id and left_id == right_id:
        left_chat, right_chat = str(left.get("chat_id") or "").strip(), str(right.get("chat_id") or "").strip()
        return not left_chat or not right_chat or left_chat == right_chat
    return bool(_corpus_keys(left) & _corpus_keys(right))


def merged_corpus(kept: Any, added: Any) -> List[Dict[str, Any]]:
    """Merge carried rows without collapsing same client ids from different rooms."""
    merged: List[Dict[str, Any]] = []
    for row in [*(kept if isinstance(kept, list) else []), *(added if isinstance(added, list) else [])]:
        if isinstance(row, Mapping) and not any(_same_corpus_source(row, prior) for prior in merged):
            merged.append(dict(row))
    return merged


# What a successor reads of an answered card beside the owner's words (``answered_card``).
CARD_FIELDS = ("quiz_id", "question", "options", "option_details", "stake", "assumption", "host_facts", "materials",
               "answered_index", "comment", "asked_at", "answered_at")


def answered_card(block: Any, words: str) -> Dict[str, Any] | None:
    """What a successor reads of the card an owner answer replies to, or None.

    ``block`` must be an answered card whose recorded owner words are exactly ``words``.
    The question and options are the asking task's words; they ride beside the answer
    and never enter the owner corpus.
    """
    import copy

    from ouroboros.owner_quiz import STATE_ANSWERED, late_answer_owner_text

    if not (isinstance(block, dict) and block.get("state") == STATE_ANSWERED
            and late_answer_owner_text(block) == words):
        return None
    return {key: copy.deepcopy(block[key]) for key in CARD_FIELDS if key in block}


def carded(row: Dict[str, Any], source: Any) -> Dict[str, Any]:
    """``row`` with the card its words answer beside them (``answered_card_source(source)``) when that card records
    exactly these words: the asking task, the card's id and the card a Continue successor reads."""
    block, asker = answered_card_source(source)
    card = answered_card(block, row["content"])
    if card and all(isinstance(value, str) and value for value in (asker, card.get("quiz_id"))):
        row.update(source_task_id=asker, quiz_id=card["quiz_id"], card=card)
    return row


def answered_card_source(row: Any) -> tuple:
    """``(block, asking task)`` of the card a row of my human's words answers, else ``(None, None)``: a relayed quiz row's
    ``quiz``, a live delivery's ``quiz_delivery`` snapshot, or a late answer's typed snapshot (``late_answer_ref``)."""
    row = _mapping(row)
    if row.get("type") == "quiz_answer":
        return row.get("quiz"), row.get("task_id")
    if delivery := _mapping(row.get("quiz_delivery")):
        return delivery.get("answer_snapshot"), delivery.get("task_id")
    from ouroboros.owner_quiz import late_answer_ref

    late = late_answer_ref(row.get("late_answer")) or {}
    return _mapping(late.get("answer_snapshot")).get("quiz"), late.get("task_id")


def answered_card_reply(block: Any, task_id: Any) -> Dict[str, str]:
    """The card my human answered, as the message their answer replies to (a corpus row's ``reply_to``), or {}.

    The whole subject the card showed (question, options with details, stake, assumption, materials) is the asking
    task's words: it rides beside the owner's answer for every reader of the corpus (Safety, reviewers, children,
    Continue), never as the owner's words; the host's record of the pressed option closes it, so a choice survives
    beside a comment. A secret request's form has none."""
    block = _mapping(block)
    if block.get("secret_request") or not str(block.get("question") or "").strip():
        return {}
    options = block.get("options") if isinstance(block.get("options"), list) else []
    details = block.get("option_details") if isinstance(block.get("option_details"), list) else []
    lines = [str(block["question"]), *(f"{n}. {label}" + (f" ({details[n - 1]})" if n <= len(details) and str(
        details[n - 1]).strip() else "") for n, label in enumerate(options, 1))]
    lines += [f"{name}: {block[key]}" for key, name in (("stake", "Stake"), ("assumption", "If unanswered"),
                                                        ("host_facts", "Host facts")) if str(block.get(key) or "").strip()]
    materials = [f"{m.get('name')} (task {m.get('task_id') or task_id})" for m in block.get("materials") or []
                 if isinstance(m, Mapping) and m.get("name")]
    lines += [f"Materials: {', '.join(materials)}"] * bool(materials)
    if type(block.get("answered_index")) is int:
        lines.append(f"(host record: my human pressed option {block['answered_index'] + 1})")
    return {"address": f"quiz {block.get('quiz_id')} of task {task_id}", "author": f"Ouroboros, task {task_id}",
            "text": "\n".join(lines)}


def reply_target(drive_root: Any, reference: Any) -> Dict[str, str]:
    """The whole message my human's words reply to, under its author, by the address the door resolved: one
    sha-checked read. Unreadable: the address alone; a reply to a work card names work, not anyone's words. Its
    author is signed without the places' owner snapshot on purpose: a target is context, never my human's words."""
    reference = _mapping(reference)
    address = str(reference.get("address") or "")
    if not address or address.startswith("task:"):
        return {}
    if reference.get("unresolved"):
        return {"address": address, "author": "", "text": ""}
    from ouroboros.chat_chain import resolve_row
    from ouroboros.dialogue_provenance import render_row_text, row_author
    from ouroboros.store import StoreError

    try:
        row, _resolution = resolve_row(drive_root, address)
    except (StoreError, OSError, ValueError, TypeError):
        row = None
    if not row:
        return {"address": address, "author": str(reference.get("author_label") or ""), "text": ""}
    return {"address": address, "author": row_author(row)["label"], "text": render_row_text(row)}


def task_governing_words(task: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """``(rows, absence marker)`` carried by value, read at the top level or under ``metadata``."""
    source = _mapping(task)
    nested = _mapping(source.get("metadata"))
    holder = source if (FIELD in source or ABSENT_FIELD in source) else nested
    raw = holder.get(FIELD)
    rows = [_row(item.get("text"), source=item.get("source"), carrier=str(item.get("carrier") or ""),
                 task_id=item.get("task_id"), ts=item.get("ts"), ref=item.get("ref"), author=item.get("author"))
            for item in (raw if isinstance(raw, list) else []) if isinstance(item, Mapping)]
    return [row for row in rows if _kept(row)], str(holder.get(ABSENT_FIELD) or "")


def governing_words_for_schedule(ctx: Any) -> Dict[str, Any]:
    """``{FIELD: rows}`` or ``{ABSENT_FIELD: marker}`` for the helper this run is about to schedule: a child's
    inherited words unchanged, any other run's own corpus beside the words it carries itself."""
    metadata = _ctx_metadata(ctx)
    if str(metadata.get("delegation_role") or "") == "subagent":
        rows, absent = task_governing_words(metadata)
    else:
        rows, absent = _deduped(task_governing_words(metadata)[0] + directive_owner_rows(ctx)), ""
    if rows:
        return {FIELD: rows}
    return {ABSENT_FIELD: absent or _absent_marker({"type": getattr(ctx, "current_task_type", None),
                                                    "metadata": dict(metadata)})}


def _inbound_rows(drive_root: Any, client_ids: Mapping[str, str], accounts: Any = None) -> List[Dict[str, Any]]:
    """Owner rows addressed by client message id, in canonical order, judged over one place snapshot (the
    caller's, else one read here)."""
    from ouroboros import dialogue_store as ds

    ds.require_imported(drive_root, "chat")
    rows = [entry.row for entry in ds.iter_probes(drive_root, "chat", [("client_message_id", key) for key in client_ids])]
    accounts = owner_place_accounts(drive_root) if accounts is None else accounts
    return [row for row in rows if owner_row(row, drive_root, accounts=accounts)]


def owner_place_accounts(drive_root: Any) -> Mapping[str, Any]:
    """One snapshot of whose place rows are my human's now (``places.place_owner_accounts``) for an operation that
    judges many rows; empty without a drive or when unreadable, so no place row counts."""
    from ouroboros.places import PlaceError, place_owner_accounts  # D17->D20 is lazy-only

    try:
        return place_owner_accounts(drive_root) if drive_root is not None else {}
    except PlaceError:
        return {}


def owner_row(row: Mapping[str, Any] | None, drive_root: Any = None, *, accounts: Any = None) -> bool:
    """The owner-source cut: inbound, non-empty, not a system or skill-repair injection, and a place row only
    from an account its place confirms as my human's now (``places.owner_confirmed_row`` over ``accounts``, the
    operation's one snapshot, else one read of ``drive_root``'s; unreadable: not)."""
    if not (row and row.get("direction") == "in" and str(row.get("text") or "").strip()
            and not (row.get("system_type") or row.get("source") == "skill_repair")):
        return False
    from ouroboros.dialogue_provenance import row_provenance
    from ouroboros.places import owner_confirmed_row  # D17->D20 is lazy-only

    if row_provenance(row) != "place":
        return True
    return owner_confirmed_row(row, owner_place_accounts(drive_root) if accounts is None else accounts)


def _message_rows(drive_root: Any, task_id: str, text: Any, ref: Any, carrier: str,
                  accounts: Any = None) -> List[Dict[str, str]]:
    ref = _mapping(ref)
    from ouroboros.project_dialogue import resolve_owner_message_source

    row = resolve_owner_message_source(drive_root, dict(ref)) if ref else None
    if isinstance(text, str) and text.strip():
        target = reply_target(drive_root, _mapping(row).get("reply_to"))
        return _words_rows({"source": "origin_message", "content": text, "reply_to": target}, carrier=carrier,
                           task_id=task_id, ts=ref.get("ts"), ref=_chat_ref(ref) if ref else "")
    if not owner_row(row, drive_root, accounts=accounts):
        return []
    return _words_rows({**row, "source": "origin_message", "content": row["text"],
                        "reply_to": reply_target(drive_root, row.get("reply_to"))}, carrier=carrier,
                       task_id=task_id, ts=row.get("ts"), ref=_chat_ref(row))


def _corpus_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any], accounts: Any) -> List[Dict[str, str]]:
    """The words a follow-up, a drafted start or a Continue carries by value, as its producer recorded them; its
    ``origin_message`` row at the record's own origin address and time, as the door's row is stamped."""
    corpus = _mapping(record.get("metadata")).get("owner_corpus") or record.get("owner_corpus")
    origin = _mapping(record.get("origin_message_ref") or _mapping(record.get("metadata")).get("origin_message_ref"))
    stamp = {"ts": origin.get("ts"), "ref": _chat_ref(origin)} if origin else {}
    return [row for item in (corpus if isinstance(corpus, list) else [])
            if isinstance(item, Mapping) and item.get("source") in OWNER_SOURCES
            for row in _words_rows(item, carrier="owner_corpus", task_id=task_id,
                                   **(stamp if item.get("source") == "origin_message" else {}))]


def _origin_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any], accounts: Any) -> List[Dict[str, str]]:
    nested = _mapping(record.get("metadata"))
    return _message_rows(drive_root, task_id,
                         record.get("origin_message_text") or nested.get("origin_message_text"),
                         record.get("origin_message_ref") or nested.get("origin_message_ref"), "origin_message", accounts)


def _binding_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any], accounts: Any) -> List[Dict[str, str]]:
    from ouroboros.projects_registry import project_task_bindings

    binding = _mapping(project_task_bindings(drive_root).get(task_id))
    return _message_rows(drive_root, task_id, binding.get("source_text"), binding.get("source_ref"), "binding",
                         accounts)


def _annotation_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any],
                        accounts: Any) -> List[Dict[str, str]]:
    from ouroboros.project_dialogue import _ANNOTATIONS_NAME, _latest_annotations_by_token

    acts: Dict[str, str] = {}
    for (client_id, _token), row in _latest_annotations_by_token(
            pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME).items():
        if (row.get("action") in _ANNOTATED_ACTS and row.get("status") in _ANNOTATED_STATUSES
                and str(row.get("target") or "") == task_id):
            acts.setdefault(client_id, str(row["action"]))
    rows = []
    for row in (_inbound_rows(drive_root, acts, accounts) if acts else []):
        rows.extend(_words_rows({**row, "source": acts[str(row["client_message_id"])], "content": row["text"],
                                 "reply_to": reply_target(drive_root, row.get("reply_to"))},
                                carrier="annotation", task_id=task_id, ts=row.get("ts"), ref=_chat_ref(row)))
    return rows


def _mailbox_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any], accounts: Any) -> List[Dict[str, str]]:
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries
    from ouroboros.owner_relay import owner_text, quotable_owner_row

    try:
        entries = drain_owner_entries(pathlib.Path(drive_root), task_id, include_acknowledged=True)
    except (OSError, ValueError):
        return []
    rows: List[Dict[str, str]] = []
    accounts_by_root = {str(drive_root): accounts}
    for entry in entries:
        if entry.get("kind") != KIND_OWNER_TEXT:
            continue
        item = {**entry, "source": "owner_mailbox", "content": entry.get("text")}
        if target := reply_target(drive_root, entry.get("reply_reference")):
            item["reply_to"] = target
        rows.extend(_words_rows(item, carrier="mailbox", task_id=task_id, ts=entry.get("ts"),
                                ref=f"msg {entry.get('msg_id')}"))
    for entry in entries:
        if entry.get("kind") != "relay_entry":
            continue
        fallback_root = entry.get("relay_canonical_root") or drive_root
        quotes = entry.get("relay", {}).get("quotes") if isinstance(entry.get("relay"), Mapping) else []
        for index, row in enumerate(entry.get("owner_rows") or []):
            source_root = row.get("_owner_source_root") or fallback_root
            source_key = str(source_root)
            source_accounts = accounts_by_root.get(source_key)
            if source_accounts is None:
                source_accounts = accounts_by_root[source_key] = owner_place_accounts(source_root)
            if not quotable_owner_row(row, source_root, accounts=source_accounts):
                continue
            quote = quotes[index] if isinstance(quotes, list) and index < len(quotes) else {}
            source = quote.get("source") if isinstance(quote, Mapping) else {}
            canonical = source.get("address") if isinstance(source, Mapping) else ""
            canonical = canonical or f"{entry.get('msg_id') or 'relay'}#{index}"
            item = {**row, "source": "owner_mailbox", "content": owner_text(row)}
            if target := reply_target(source_root, row.get("reply_to")):
                item["reply_to"] = target
            rows.extend(_words_rows(item, carrier="relay", task_id=task_id, ts=row.get("ts"),
                                    ref=_carrier_ref(row, canonical=canonical)))
    return rows


def _task_record(drive_root: Any, task_id: str) -> Dict[str, Any]:
    if not task_id:
        return {}
    from ouroboros.task_status import load_effective_task_result

    try:
        record = load_effective_task_result(drive_root, task_id, materialize_artifacts=False)
    except Exception:
        log.debug("owner words: task record unreadable for %s", task_id, exc_info=True)
        return {}
    return dict(record) if isinstance(record, dict) else {}


def task_owner_words(drive_root: Any, task_id: str, *,
                     task: Mapping[str, Any] | None = None) -> Tuple[List[Dict[str, str]], str]:
    """The words for a task: its carried field, else the first carrier of its root that yields a row; a message the
    task holds (``owner_relay.held_owner_messages``) under the carriers' reference, so one message is one row."""
    record = dict(task) if isinstance(task, Mapping) else _task_record(drive_root, task_id)
    from ouroboros.owner_relay import owner_text, quotable_owner_row

    accounts = owner_place_accounts(drive_root)  # one snapshot for every row this read judges
    accounts_by_root = {str(drive_root): accounts}
    held = []
    for handle, row in (record.get("held_owner_messages") or {}).items():
        if not isinstance(row, Mapping):
            continue
        source_root = row.get("_owner_source_root") or drive_root
        source_key = str(source_root)
        source_accounts = accounts_by_root.get(source_key)
        if source_accounts is None:
            source_accounts = accounts_by_root[source_key] = owner_place_accounts(source_root)
        if not quotable_owner_row(row, source_root, accounts=source_accounts):
            continue
        item = {**row, "source": "owner_mailbox", "content": owner_text(row)}
        if target := reply_target(source_root, row.get("reply_to")):
            item["reply_to"] = target
        held.extend(_words_rows(item, carrier="held", task_id=task_id, ts=row.get("ts"),
                                ref=_carrier_ref(row, canonical=handle)))
    rows, absent = task_governing_words(record)
    if rows or absent:
        return _deduped(rows + held), "" if rows or held else absent
    task_id = str(task_id or "")
    root_id = str(record.get("root_task_id") or _mapping(record.get("metadata")).get("root_task_id") or task_id)
    if root_id != task_id:
        task_id, record = root_id, (_task_record(drive_root, root_id) or record)
    for carrier in (_corpus_carrier, _origin_carrier, _binding_carrier, _annotation_carrier, _mailbox_carrier):
        rows = _deduped(carrier(drive_root, task_id, record, accounts))
        if rows:
            return _deduped(rows + held), ""
    return _deduped(held), "" if held else _absent_marker(record)


def render_owner_words(rows: Any, absent: str = "", *, audience: str, root_task_id: str = "", indent: str = "") -> str:
    """One verbatim section for a child, a session, a reviewer, a plan review or the page writer.

    ``indent`` prefixes every line of the words (the memory view's two spaces), so a line of
    the owner's own that starts with ``## `` never reads as a section of the request.
    """
    if audience not in _HEADINGS:
        raise ValueError(f"unknown owner-words audience {audience!r}; expected one of {sorted(_HEADINGS)}")
    rows = [row for row in (rows if isinstance(rows, list) else []) if isinstance(row, Mapping) and _kept(row)]
    if not rows:
        return f"No words of my human are recorded for this work (host marker: {absent})." if absent else ""
    origin = _ORIGIN.format(root=root_task_id) if root_task_id else ""
    blocks = []
    for row in rows:
        text = str(row.get("text") or "") or (unread_target(row) if row.get("source") == REPLY_TARGET else "")
        words = "\n".join(indent + line if line else line for line in text.split("\n"))
        blocks.append(words_stamp(row) + "\n" + words)
    return f"{_HEADINGS[audience]}\n{_FRAMES[audience].format(origin=origin)}\n" + "\n\n".join(blocks)


def unread_target(row: Mapping[str, Any]) -> str:
    """What stands for a reply target's words the host did not carry: the host address that reads them, or the
    honest fact that the connector gave only a platform message id."""
    return UNREAD_TARGET if str(row.get("ref") or "").startswith("row:") else UNREAD_PLATFORM


def words_stamp(row: Mapping[str, Any]) -> str:
    """One row's stamp in every form, whole or by reference: ``[ts · owner · source of task · ref]``, or a reply
    target's own author, never my human (``REPLY_TARGET``)."""
    source, task_id, ref = str(row.get("source") or ""), str(row.get("task_id") or ""), str(row.get("ref") or "")
    if source == REPLY_TARGET:  # context beside the words, under its own author: never my human's words
        stamp = [f"{row.get('author') or 'author not recorded'}: the message my human replied to, "
                 "not my human's words", ref]
    else:
        stamp = [str(row.get("ts") or ""), "owner", f"{source} of task {task_id}" if task_id else source, ref]
    return "[" + " · ".join(part for part in stamp if part) + "]"


def owner_words_text(ctx: Any, *, audience: str = "reviewer") -> str:
    """``render_owner_words`` over the words this run would hand a helper."""
    words = governing_words_for_schedule(ctx)
    root = str(_ctx_metadata(ctx).get("root_task_id") or getattr(ctx, "task_id", "") or "")
    return render_owner_words(words.get(FIELD) or [], str(words.get(ABSENT_FIELD) or ""),
                              audience=audience, root_task_id=root)


def reviewer_words_material() -> list:
    """Review-contract material (``commit_gate.commit_review_contract_fingerprint``): a reviewer's frame of the
    owner's words over a placeholder row, and the absence marker. The words and stamps are data."""
    row = {"text": "⟨words⟩", "source": "⟨source⟩", "task_id": "⟨task⟩", "ts": "⟨ts⟩", "ref": "⟨ref⟩"}
    return [render_owner_words([row], audience="reviewer", root_task_id="⟨root⟩"),
            *(render_owner_words([], marker, audience="reviewer") for marker in (NOT_RECORDED, "⟨marker⟩"))]
