"""My human's words for a work tree: one set, read where the rows already are, dated, rendered verbatim.

A helper's floor is its assignment plus my human's words for the work (BIBLE P1:
provenance matters). One resolver serves every handoff, in two projections of the same
rows. ``governing_words_for_schedule`` computes the tree's own rows by value when a run
schedules a helper, from what that run already holds: the field it inherited, the
owner-sourced rows of its own corpus (``ctx._owner_directives`` — never ``initial_text``,
which is not the owner's) and the rows it holds that reached this work
(``owner_relay.held_owner_messages``). A helper passes on what it inherited and what it
received since; that computation reads no dialogue. The rows ride the existing
tree-origin channel (``origin_metadata``) into every descendant's ``task.metadata`` under
``FIELD``. A run with no owner words carries ``ABSENT_FIELD`` set to the raw ``run_origin``
marker (``initiator=consciousness``) or ``not_recorded``: no vocabulary of reasons of its own.

``work_words`` is the set a handoff renders (a helper's birth, a reviewer's ask, a direct
start) and the mind reads itself (``memory_read(owner_words=true)``): the tree's rows first,
then the rows the run holds as context, the bound Project room's rows read once at that
render, the Project's retained origin rows and the rows that marks in force target. One
message is one row whichever carrier holds it — its room and client id; a row with no id
at all by its words and its time — and the first carrier keeps
the row. Every row carries the time the room accepted it, or ``received``, when the host
got it, or none: never an invented date. An answer travels with the card it answers, under
the asking task's name; a press without words is their act (``OWNER_PRESS``), rendered as
the host's record of it, never as their words. A room or a mark store that cannot be read is one gap line beside
the rows that could. A Main tree has no room term: Main's rows reach a work by the mind's
relay.

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

A Project's first message is one of its room's rows (else its retained origin row) and is
stamped as such; it never enters the field a schedule computes. ``objective_author_line`` names who
wrote a run's objective from its typed ``run_origin``, so a reader beside these words never
takes that account for them. Nothing here calls a model.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Mapping, Tuple

from ouroboros.answer_facts import (OWNER_CARD_INPUT, OWNER_PRESS, answered_card, answered_card_reply,
                                    answered_card_source, carded as carded, is_answered_card)
from ouroboros.chat_references import _same_selection as _same_selection, reply_target as reply_target, selected_reply_target

log = logging.getLogger(__name__)

# The one list of corpus labels (``loop_messages._record_owner_directive``) that are my human's own input: their words,
# and ``OWNER_PRESS``, their press of an offered option without words (``answer_facts``): an act, counted, owed a reply
# and read by Safety's card check like their words, never cited or rendered as words (``OWNER_WORD_SOURCES``);
# ``place_owner``: their words from a place that a run cited for the work it started (``basis: cited``, verified by
# ``cited_owner_basis``); older records carry the same label from the retired host verdict.
OWNER_SOURCES = frozenset({"initial_user", "owner_mailbox", "owner_quiz_answer", "origin_message",
                           "owner_corpus", "direct_incoming", "place_owner", OWNER_PRESS, OWNER_CARD_INPUT})
OWNER_WORD_SOURCES = OWNER_SOURCES - {OWNER_PRESS, OWNER_CARD_INPUT}
# A place event from an account its place confirms as my human's: their words, said where others can hear. Provenance
# only (outside ``OWNER_SOURCES``): whom they are for, and whether they ask me for work, is my reading; an order exists
# only where I cite them and the host verifies the bytes.
PLACE_CONFIRMED = "place_confirmed"
# A schedule's text, beside my human's words and never theirs: each field of it (its objective, its context) under the
# supplier the host recorded for that field and when (``objective_author``/``context_author``, written only by the
# schedule writers in ``supervisor/schedule_lifecycle.py``). Text that came through the Schedules API
# (``POST /api/schedules``: the owner's CLI, or any program holding that access) is the work's assignment and its
# conditions, not my human's personal word, and gives no permission only they can give. Corpus items carry it by value
# (outside ``OWNER_SOURCES``) and ``SCHEDULE_FIELD`` hands the run's items to every helper it schedules.
SCHEDULE_TEXT = "schedule_text"
SCHEDULE_FIELD = "governing_schedule_text"
# A schedule item's typed ids, each a string when present (``scheduled_text_items``).
SCHEDULE_ITEM_IDS = ("schedule_id", "field", "supplier", "supplier_task_id", "supplied_at")
_TEXT_AUTHOR_KEYS = (("objective", "objective_author"), ("context", "context_author"))
CITED = "cited"
FIELD, ABSENT_FIELD = "governing_owner_words", "governing_owner_words_absent"
# A session task's record alone: the rows ``_live_words`` read at its scheduling (the bound room's rows, the
# Project's origin rows and the marked rows), by value, read by its work order only (never inherited, carried or
# read by a Safety check); its presence tells a work order rendered after it.
ROOM_FIELD = "work_order_room_words"
NOT_RECORDED = "not_recorded"
REPLY_TARGET = "reply_target"  # the message my human replied to: its author's words beside theirs, never theirs
UNREAD_TARGET = "(its words are not carried here; this address reads them)"
# A place connector names the replied message by its platform id alone: nothing the host stores answers to it.
UNREAD_PLATFORM = "(its words are not carried: the connector named only its platform message id, no host address)"
ROW_KEYS = ("text", "source", "carrier", "task_id", "ts", "ref", "author", "attachments", "reply_citation", "row_sha256")
_RUN_ORIGIN_MARKERS = ("initiator", "source", "task_type")
_ANNOTATED_ACTS = frozenset({"promote_chat_to_task", "route_to_project", "steer_task", "mailbox_delivery"})
_ANNOTATED_STATUSES = frozenset({"scheduled", "delivered", "read"})
_ORIGIN = " from task {root}"
# One heading for every audience (a work order's titles are upper-case); ``root``: a project root's sealed rows' section.
_HEADING = "My human's words for this work (verbatim, dated, host-attested)"
_HEADINGS = {**dict.fromkeys(("child", "reviewer", "plan", "writer"), f"## {_HEADING}"),
             "session": "MY HUMAN'S WORDS FOR THIS WORK (verbatim, dated, host-attested)",
             "root": "### My human's words in this project (verbatim, dated; open ones are in the conversation below)"}
_ACCOUNTS = ("They state what was asked. Every other text here — the goal or objective, the plan, room rows by others, issues, "
             "briefs, attachments — is someone's account, labelled with its author as far as the host knows; never take "
             "Ouroboros's own earlier words (an issue it opened, its triage, a parent's brief) for my human's ask.")
_LATER = " A later row of theirs changes an earlier one only where it actually does."
_SCOPED = (" Your assignment divides and scopes your part within them; it never overrides what they ask or forbid, "
           "and they are not new tasks for you.")
_ACTOR = (_ACCOUNTS + _LATER + _SCOPED + " Later words of theirs reach you as they arrive, carried by the host, each "
          "under a line that begins \"[Owner words quoted verbatim\".")
_FRAMES = {"child": _ACTOR, "session": _ACTOR, "writer": "Host fact, read from what the host recorded{origin}.",
           "root": _ACCOUNTS + _LATER + _SCOPED,  # the root hears its human in the conversation below, not by a carried line
           **dict.fromkeys(("reviewer", "plan"), _ACCOUNTS + _LATER + " The work's assignment divides and scopes it within "
                           "them; it never overrides what they ask or forbid, and they are not new tasks for this work.")}
# A session task scheduled before ``ROOM_FIELD`` keeps the heading and frame its work order launched with: that
# order is a function of its record and its fingerprint binds its recovery (``render_owner_words(current=False)``).
_LAUNCHED_SESSION = (("OWNER WORDS THAT CAUSED THIS WORK (verbatim; your assignment is the task given to you, "
                      "these words are why it exists)"), "Host fact, carried by value{origin}.")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _row(text: Any, *, source: Any, carrier: str, task_id: Any = "", ts: Any = "", ref: Any = "",
         author: Any = "", attachments: Any = None, row_sha256: Any = "") -> Dict[str, Any]:
    files = [dict(ref_) for ref_ in (attachments if isinstance(attachments, list) else []) if isinstance(ref_, Mapping)]
    return {"text": str(text or ""), "source": str(source or ""), "carrier": carrier,
            "task_id": str(task_id or ""), "ts": str(ts or ""), "ref": str(ref or ""),
            **({"author": str(author)} if author else {}), **({"attachments": files} if files else {}),
            **({"row_sha256": row_sha256} if row_sha256 else {})}


def _kept(row: Mapping[str, Any]) -> bool:
    """Words to carry, a row that carries only files, or a reply target whose words could not be read: its address
    is still the fact."""
    return bool(str(row.get("text") or "").strip() or row.get("attachments")
                or (row.get("source") == REPLY_TARGET and row.get("ref")))


def _chat_ref(source: Mapping[str, Any]) -> str:
    client_id = str(source.get("client_message_id") or "")
    from ouroboros.chat_chain import _row_chat_id
    return f"chat {_row_chat_id(source)}" + (f" / {client_id}" if client_id else "")


def row_ref(item: Mapping[str, Any]) -> str:
    """A corpus item's typed identity as every reader prints it: a chat message by its room and client id, as its
    carriers name it (``_chat_ref``); else its mailbox ``msg_id``; else its client id alone, never under a room it did
    not name. One message is one reference whichever carrier holds it, and two messages are two even in one word."""
    client, msg_id = (str(item.get(key) or "") for key in ("client_message_id", "msg_id"))
    if client and item.get("chat_id") not in (None, ""):
        return _chat_ref(item)
    if msg_id:
        return msg_id if msg_id.startswith("row:") else f"msg {msg_id}"
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


def _held_origin(ref: Mapping[str, Any], held: Any) -> Mapping[str, Any] | None:
    """Resolve an identified origin from rows already retained, without guessing from equal words alone."""
    if not (ref.get("client_message_id") or ref.get("ts")) or not ref.get("text_sha256"):
        return None
    from ouroboros.project_dialogue import entry_matches_source_ref
    from ouroboros.owner_relay import owner_row_identity

    matches = {owner_row_identity(row)["row_sha256"]: row for row in _mapping(held).values()
               if isinstance(row, dict) and entry_matches_source_ref(row, [dict(ref)])}
    return next(iter(matches.values())) if len(matches) == 1 else None


def _reply_body_key(address: Any, view: Any = None, author: Any = None) -> tuple:
    """A stored row can supply a whole-row notice and a distinct authored body (or no readable body)."""
    return str(address or ""), str(view or ""), str(author or "author not recorded") if view else ""


def _reply_bodies(rows: List[Dict[str, Any]]) -> dict:
    return {_reply_body_key(row["ref"], row.get("reply_view"), row.get("author")): row
            for row in rows if row["source"] == REPLY_TARGET}


def _carried_selection(citation: Any, bodies: dict, *, attachments: bool = False) -> dict:
    """Rejoin one reply's citation to its by-value source view, never to an ordinary Reply's display."""
    citation = _mapping(citation)
    if not (address := citation.get("address")):
        return {}
    # An older field did not distinguish the target's views; never infer authored words from that row.
    author, view = citation.get("author") or "", citation.get("reply_view")
    body = bodies.get(_reply_body_key(address, view, author), {}) if view else {}
    return {"address": address, "author": author, "excerpt": citation.get("excerpt"), "text": body.get("text", ""),
            **({"attachments": body["attachments"]} if attachments and body.get("attachments") else {})}


def _deduped(rows: List[Dict[str, Any]], *, addressed: bool = False) -> List[Dict[str, Any]]:
    """Drop empty rows and repeats of one source: its reference, else (none recorded) its whitespace-normalized
    text and its time; the text alone only when no time was recorded either. Similar words under different
    references or times are not copies (BIBLE P1): two "yes" stay two. The first row of one identity keeps its
    carrier and its date, and takes a later twin's files when it has none (a corpus row records a relayed row's
    words; its held chat row keeps the upload references): a row never loses a file to its own identity. A reply
    target repeats by its address and view: two replies to the same form keep one target row, beside the first."""
    from ouroboros.project_dialogue import _text_sha256

    hashes: Dict[str, dict] = {}
    openings = set()
    for row in rows:
        if row.get("row_sha256") and row["ref"]:
            hashes.setdefault(row["ref"], {})[row["row_sha256"]] = row
            if row["source"] == "origin_message":
                openings.add(row["row_sha256"])
    seen: Dict[Any, Dict[str, Any]] = {}  # one identity -> the row kept for it
    kept = []
    for row in rows:
        if not _kept(row):
            continue
        digest = row.get("row_sha256")
        if row["source"] == REPLY_TARGET:
            key = _reply_body_key(row["ref"], row.get("reply_view"), row.get("author"))
        else:
            key = (False, row["ref"]) if row["ref"] else (_text_sha256(row["text"]), str(row.get("ts") or ""))
            candidates = hashes.get(row["ref"], {})
            if not digest and len(candidates) == 1:
                twin = next(iter(candidates.values()))
                compatible = twin["row_sha256"] not in openings or (row["ts"] and row["text"] == twin["text"])
                if compatible and (not row["ts"] or row["ts"] == twin["ts"]) and (
                        not row.get("attachments") or row["attachments"] == twin.get("attachments")):
                    key = ("row_sha256", twin["row_sha256"])
            if digest:
                key = ("row_sha256", digest)
                if len(candidates) > 1 and row["ref"].startswith("chat "):
                    room = row["ref"][5:].partition(" / ")[0]
                    row = {**row, "ref": f"row:{room}@{row['ts']}#{digest[:12]}"}
        if key not in seen:
            kept.append(seen.setdefault(key, row))
            continue
        first = seen[key]
        if first["text"] != row["text"] and not (first.get("row_sha256") and row.get("row_sha256")):
            continue  # an older capture under the same name cannot take the changed row's files or address
        if (digest and first.get("row_sha256") == digest and row.get("answer_input") is not None
                and row["carrier"] in {"origin", "origin_message", "owner_corpus", "binding"}):
            old, new = first.get("answer_input"), row["answer_input"]
            if old is None or (first["source"] == OWNER_CARD_INPUT and row["source"] != OWNER_CARD_INPUT
                               or "submitted_values" in new and "submitted_values" not in old):
                first.update(text=row["text"], source=row["source"], answer_input=new)
        if digest:
            first.setdefault("row_sha256", digest)
        if citation := _mapping(row.get("reply_citation")):
            prior = first.setdefault("reply_citation", citation)
            # Two carriers of the same reply can differ only in whether they could read the original.
            if (row["ref"] and row["text"] == first["text"] and prior.get("reply_view") == "unread"
                    and citation.get("reply_view") == "authored" and citation.get("address") and citation.get("excerpt")
                    and _same_selection(prior, citation)):
                first["reply_citation"] = citation
        if row.get("attachments") and not first.get("attachments"):
            first["attachments"] = row["attachments"]
        # Only the birth capture requests coordinates. A normalized-text match
        # without a ref does not identify a twin, and a changed row cannot
        # supply a reader for the earlier words even with the same client id.
        if (addressed and row["ref"] and row["ts"] == first["ts"] and row["text"] == first["text"]
                and row.get("author") == first.get("author")
                and (not first.get("address") or not row.get("address") or first["address"] == row["address"])):
            for field in ("address", "room_id", "origin_ref"):
                if field in row:
                    first.setdefault(field, row[field])
    # A second carrier of the same reply can know less (or more); don't print its unused source view by itself.
    cited = {_reply_body_key(citation["address"], citation["reply_view"], citation.get("author")) for row in kept
             if (citation := _mapping(row.get("reply_citation"))).get("reply_view") and citation.get("address")}
    return [row for row in kept if not row.get("reply_view") or
            _reply_body_key(row["ref"], row["reply_view"], row.get("author")) in cited]


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
    if kind == "schedule_api":
        when = f" at {author['at']}" if author.get("at") else ""
        return (f"supplied through the Schedules API{when} (the owner's CLI, or any program with that access): "
                "the assignment and its conditions, not my human's words")
    if kind == "ui_control":
        return (f"the host's prompt for my human's press of {author.get('control') or 'a control'}: "
                "the press is theirs, these words are not")
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


def scheduled_text_items(metadata: Any, contract: Any) -> List[Dict[str, Any]]:
    """A schedule's run's own corpus items for its schedule's text (``SCHEDULE_TEXT``): each field its contract holds
    (the objective and the context, as the contract keeps them) with the schedule, the field and the supplier the
    host recorded for it (``supplier``: the author's kind; ``supplier_task_id``; ``supplied_at``), each absent when
    the host recorded none: an unknown supplier is never my human. A run no schedule admitted has none."""
    metadata, contract = _mapping(metadata), _mapping(contract)
    schedule_id = str(metadata.get("schedule_id") or "")
    items: List[Dict[str, Any]] = []
    for field, key in _TEXT_AUTHOR_KEYS if schedule_id else ():
        text, author = contract.get(field), _mapping(metadata.get(key))
        if isinstance(text, str) and text.strip():
            items.append({"source": SCHEDULE_TEXT, "content": text, "schedule_id": schedule_id, "field": field,
                          **{name: str(author[value]) for name, value in (("supplier", "kind"), ("supplier_task_id", "task_id"),
                                                                           ("supplied_at", "at")) if author.get(value)}})
    return items


def schedule_items(ctx: Any) -> List[Dict[str, Any]]:
    """The schedule text this run holds: its own schedule's and its tree's (``_initialize_owner_directives`` records
    both into the corpus), as items."""
    return [{key: row[key] for key in ("source", "content", *SCHEDULE_ITEM_IDS) if row.get(key)}
            for row in getattr(ctx, "_owner_directives", None) or []
            if isinstance(row, dict) and row.get("source") == SCHEDULE_TEXT and str(row.get("content") or "").strip()]


def task_schedule_items(task: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The schedule text a helper was scheduled with (``SCHEDULE_FIELD``), at the top level or under ``metadata``."""
    source = _mapping(task)
    raw = source[SCHEDULE_FIELD] if SCHEDULE_FIELD in source else _mapping(source.get("metadata")).get(SCHEDULE_FIELD)
    return [dict(item) for item in (raw if isinstance(raw, list) else [])
            if isinstance(item, Mapping) and item.get("source") == SCHEDULE_TEXT and str(item.get("content") or "").strip()]


def _schedule_rows(items: Any, task_id: str = "") -> List[Dict[str, Any]]:
    """Items as rows of a section: the words, the supplier's label (``author``) and when, under ``schedule <id> <field>``."""
    rows = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, Mapping) and item.get("source") == SCHEDULE_TEXT:
            kind = str(item.get("supplier") or "")
            label = {"schedule_api": "the Schedules API",
                     "task": f"task {item.get('supplier_task_id') or '(id not recorded)'}"}.get(kind, kind and f"kind={kind}")
            rows.append(_row(item.get("content"), source=SCHEDULE_TEXT, carrier="schedule", task_id=task_id,
                             ts=item.get("supplied_at"), ref=f"schedule {item.get('schedule_id')} {item.get('field')}",
                             author=label or "a supplier the host did not record"))
    return _deduped(rows)


_SCHEDULE_HEADING = "The schedule's text for this work (verbatim, with the supplier the host recorded)"
_SCHEDULE_FRAME = ("It is the work's assignment and its conditions, not my human's words: text supplied through the Schedules "
                   "API (the owner's CLI, or any program with that access) gives no permission only my human can give.")


def render_schedule_text(items: Any, *, audience: str, indent: str = "") -> str:
    """The schedule text section for the audiences of ``render_owner_words``; ``""`` when there is none."""
    if audience not in _HEADINGS:
        raise ValueError(f"unknown owner-words audience {audience!r}; expected one of {sorted(_HEADINGS)}")
    rows = _schedule_rows(items)
    if not rows:
        return ""
    heading = (_SCHEDULE_HEADING.upper() if audience == "session"
               else f"{'###' if audience == 'root' else '##'} {_SCHEDULE_HEADING}")
    return f"{heading}\n{_SCHEDULE_FRAME}\n" + "\n\n".join(words_block(row, indent) for row in rows)


def _item_time(item: Mapping[str, Any]) -> str:
    """A corpus item's time as its stamp prints it: the time its room accepted it, else ``received``, when its
    mailbox entry was written or a drain read it; else none."""
    return str(item.get("ts") or "") or (f"received {item['received']}" if item.get("received") else "")


def _words_rows(item: Mapping[str, Any], *, carrier: str, task_id: str, content: Any = None, ts: Any = "",
                ref: str = "", attachments: Any = None) -> List[Dict[str, Any]]:
    """One corpus item's rows: its words verbatim (images by reference), a press without words as the host's record
    of the act (``answer_facts.press_record``), under its typed identity (``row_ref``) and its time (``_item_time``),
    then the message they reply to, under that message's own author."""
    from ouroboros.answer_facts import press_record
    from ouroboros.review_evidence_sections import _owner_content_projection

    words = _owner_content_projection(item.get("content") if content is None else content)
    shown = words
    if item["source"] == OWNER_PRESS:  # an act: the host's record stands where words would; its card follows
        words = press_record(words, item.get("source_task_id") or answered_card_source(item)[1])
    rows = [_row(words, source=item["source"], carrier=carrier, task_id=task_id, ts=ts or _item_time(item),
                 ref=ref or row_ref(item), attachments=item.get("attachments") if attachments is None else attachments,
                 row_sha256=item.get("row_sha256"))]
    target = _mapping(item.get("reply_to"))
    if item["source"] in (OWNER_CARD_INPUT, OWNER_PRESS) or is_answered_card(target) or target.get("card_id"):
        rows[0]["answer_input"] = {**({"shown": shown} if item["source"] == OWNER_PRESS else {}),
                                   "reply_address": target.get("address", ""),
                                   **{k: item[k] for k in ("source_task_id", "quiz_id") if item.get(k)},
                                   **{k: target[k] for k in ("card_id", "pressed_label", "submitted_values") if target.get(k)}}
    if target.get("address") and (target.get("text") or target.get("excerpt")
                                  or not (is_answered_card(target) or target.get("card_id"))):
        replied = _row(target.get("text"), source=REPLY_TARGET, carrier=carrier, task_id=task_id,
                       ref=target["address"], author=target.get("author") or "author not recorded",
                       attachments=target.get("attachments"))
        if target.get("excerpt"):  # each reply selects its own words; the associated source body can be shared
            replied["reply_view"] = "authored" if replied["text"] else "unread"
            rows[0]["reply_citation"] = {"address": target["address"], "author": target.get("author") or "",
                                         "excerpt": target["excerpt"], "reply_view": replied["reply_view"],
                                         **({"role": target["role"]} if target.get("role") else {})}
        rows.append(replied)
    return rows


def _held_twin(item: Mapping[str, Any], held: Mapping[str, Any], *, root=None, opening="") -> Mapping[str, Any]:
    """The chat row this focus holds for a corpus item recorded without its room (before the drain named it, or
    when no chat row was found then): by the client id an owner mail or a direct delivery kept, or by the card
    an answer names (``answered_card_source``: a quiz row, a live delivery, a late answer). ``{}``: none held."""
    quiz, asker = str(item.get("quiz_id") or ""), str(item.get("source_task_id") or "")
    from ouroboros.chat_chain import _row_chat_id
    from ouroboros.owner_relay import origin_files, owner_row_address, owner_row_identity, owner_text, source_words
    matches = {}
    address = str(item.get("msg_id") or "")
    client = str(item.get("client_message_id") or "")
    if not client and item.get("source") == "direct_incoming" and not address.startswith("row:"):
        client = address
    for row in held.values():
        if not isinstance(row, Mapping):
            continue
        if item.get("chat_id") not in (None, "") and str(item["chat_id"]) != str(_row_chat_id(row)):
            continue
        if (item.get("ts") and item["ts"] != row.get("ts")) or (
                address.startswith("row:") and address != owner_row_address(row)):
            continue
        digest = owner_row_identity(row)["row_sha256"]
        if item.get("row_sha256") and item["row_sha256"] != digest:
            continue
        block, task = answered_card_source(row) if quiz else (None, None)
        if client and client != str(row.get("client_message_id") or ""):
            continue
        if quiz and (quiz != str(_mapping(block).get("quiz_id") or "") or asker != str(task or "")):
            continue
        if item.get("row_sha256") or address.startswith("row:") or client or quiz and asker:
            matches[digest] = row
    if len(matches) != 1:
        return {}
    digest, twin = next(iter(matches.items()))
    # A partial old note can describe its own held message, but a client id alone cannot make it the opening.
    if digest == opening and not (item.get("row_sha256") or address.startswith("row:") or quiz and asker
                                  or item.get("ts") and item.get("chat_id") not in (None, "")):
        return {}
    # Content can contradict an identity, never choose between candidates.
    if item.get("words", item.get("content")) not in (owner_text(twin), source_words(twin)):
        return {}
    if "attachments" in item and item["attachments"] not in (twin.get("attachments") or [], origin_files(twin, root)):
        return {}
    return twin


def _opening(item: Mapping[str, Any], origin_ref: Mapping[str, Any], *, first: bool = False) -> bool:
    """The run's first message as the owner door stamped it (``origin_message_ref``): its own ``initial_user`` row, or
    its origin row recorded without an identity of its own. A door row carried in by value (``carried``) is another
    run's first message: it keeps its own identity, even none, and never takes this run's."""
    source = item.get("source")
    return bool(origin_ref) and not item.get("carried") and (
        source == "initial_user" or first and source == "origin_message" and not row_ref(item) and not item.get("row_sha256"))


def _opening_identity(ctx: Any, *, read: bool = False) -> Dict[str, str]:
    """This run's own first message as typed ids a carried row keeps by value: the room and client id its stamp
    names, else the canonical address of the held row the stamp names (``_held_origin``), with the stamp's time;
    ``{}`` when the door stamped no message (a suppressed origin) or none of it can be read."""
    from ouroboros.chat_chain import _row_chat_id
    from ouroboros.owner_relay import owner_row_address, owner_row_identity

    origin = _mapping(_ctx_metadata(ctx).get("origin_message_ref"))
    stamp = {"ts": str(origin["ts"])} if origin.get("ts") else {}
    if origin.get("row_sha256"):
        stamp["row_sha256"] = origin["row_sha256"]
    if origin.get("row_sha256") and not origin.get("client_message_id"):
        from ouroboros.chat_chain import format_address
        return {"msg_id": format_address(origin), **stamp}
    if origin.get("client_message_id"):
        return {"client_message_id": str(origin["client_message_id"]), "chat_id": str(_row_chat_id(origin)), **stamp}
    known = _held_origin(origin, _mapping(getattr(ctx, "_held_owner_messages", None))
                         or _ctx_metadata(ctx).get("held_owner_messages")) if origin else None
    if not known and read and origin.get("ts") and origin.get("text_sha256") and (root := _data_root(ctx)):
        from ouroboros.project_dialogue import resolve_owner_message_source
        from ouroboros.store import StoreError
        try:
            known = resolve_owner_message_source(root, dict(origin))
        except (OSError, StoreError):
            pass
    return {"msg_id": owner_row_address(known), "row_sha256": owner_row_identity(known)["row_sha256"], **stamp} if known else {}


def _door_ref(ctx: Any) -> str:
    """The run's first message under the door's stamp (``origin_message_ref``): the reference the held origin row
    keeps when the run holds one (``_held_origin``: a legacy door without a client id keeps its carrier's address),
    else its room and client id. The copy a helper inherits carries the same reference."""
    from ouroboros.owner_relay import owner_row_address

    origin_ref = _mapping(_ctx_metadata(ctx).get("origin_message_ref"))
    held = _mapping(getattr(ctx, "_held_owner_messages", None))
    known = _held_origin(origin_ref, held or _ctx_metadata(ctx).get("held_owner_messages"))
    if known:
        return _carrier_ref(known, canonical=owner_row_address(known))
    return _opening_identity(ctx).get("msg_id") or _chat_ref(origin_ref)


def _dated_directives(ctx: Any):
    """``(item, time, reference, door-stamped)`` for each owner-sourced item of the run's corpus: the door's stamp
    for its first message, and for its origin row recorded without an identity of its own, under the door's
    reference (``_door_ref``); else the item's own room, client id and time (empty here: ``_words_rows`` reads them);
    an item without a room reads under the chat row this focus holds for it (``_held_twin``), so one message is
    one row whichever carrier names it, and an answer recorded without its card takes the card its twin answers,
    so the card renders beside the answer."""
    origin_ref = _mapping(_ctx_metadata(ctx).get("origin_message_ref"))
    held = _mapping(getattr(ctx, "_held_owner_messages", None))
    reference = _door_ref(ctx)
    for index, item in enumerate(getattr(ctx, "_owner_directives", None) or []):
        if not isinstance(item, dict) or item.get("source") not in OWNER_SOURCES:
            continue
        if _opening(item, origin_ref, first=index == 0):
            yield {**item, **_opening_identity(ctx)}, str(origin_ref.get("ts") or ""), reference, item["source"] == "initial_user"
            continue
        twin = (_held_twin(item, held, root=_data_root(ctx), opening=origin_ref.get("row_sha256"))
                if held and not item.get("row_sha256") else {})
        if twin:
            from ouroboros.owner_relay import owner_row_identity
            item = {**item, "row_sha256": owner_row_identity(twin)["row_sha256"]}
        if twin and not _mapping(item.get("reply_to")).get("address") and (
                card := answered_card_reply(*answered_card_source(twin))):
            item = {**item, "reply_to": card}
        yield item, str(twin.get("ts") or ""), _carrier_ref(twin, canonical=item.get("msg_id")) if twin else "", False


def directive_owner_rows(ctx: Any) -> List[Dict[str, Any]]:
    """The owner-sourced rows of the run's retained corpus, verbatim (images by reference), each under its
    reference and time (``_dated_directives``).

    The door-stamped first message is the door's own record of it (``origin_message_text``)
    when the door kept one: the host may have prefixed a notice of its own to that user
    turn (a Swarm initiative), never to the door's text.
    """
    door_text = _ctx_metadata(ctx).get("origin_message_text")
    door_text = door_text if isinstance(door_text, str) and door_text.strip() else None
    task_id = str(getattr(ctx, "task_id", "") or "")
    rows, carried = [], []
    for item, ts, ref, door in _dated_directives(ctx):
        snapshot = _mapping(_ctx_metadata(ctx).get("origin_message_snapshot")) if door else {}
        if snapshot.get("placeholder_text"):
            item = {**item, "content": "", "attachments": snapshot.get("attachments") or item.get("attachments")}
        found = _words_rows(item, carrier="ctx", task_id=task_id,
                            content=door_text if door and not (snapshot.get("placeholder_text") or _card_input_row(item)) else None,
                            ts=ts, ref=ref)
        carried += [(item, found[0])] * bool(item.get("carried") and item.get("attachments"))
        rows += found
    rows = _deduped(rows)
    if carried:  # a message carried in beside the one this work answers keeps its files named, not given
        from ouroboros.owner_relay import mark_context_files, owner_row_identity
        mark_context_files(carried, _opening_identity(ctx), {owner_row_identity(row)["row_sha256"] for row in _mapping(
            getattr(ctx, "_held_owner_messages", None)).values() if isinstance(row, Mapping)})
    return rows


def _earlier_refs(ctx: Any) -> Dict[str, str]:
    """The reference a corpus row prints now -> the mailbox reference (``msg <id>``) it printed before its room
    rode with it: one identity under either, so a reader that holds the older print has seen the row."""
    return {now: f"msg {item['msg_id']}" for item, _ts, ref, _door in _dated_directives(ctx)
            if item.get("msg_id") and (now := ref or row_ref(item)) != f"msg {item['msg_id']}"}


def _card_input_row(row: Mapping[str, Any]) -> bool:
    return (row.get("source") in (OWNER_PRESS, OWNER_CARD_INPUT, "owner_quiz_answer")
            or bool(row.get("quiz_id") or _mapping(row.get("reply_to")).get("card_id")))


def carried_owner_corpus(ctx: Any, source_text: Any = "", *, source: Mapping[str, Any] | None = None
                         ) -> List[Dict[str, Any]]:
    """The run's owner corpus, carried by value into work it starts: a drafted objective or a follow-up.

    Every row the run admitted as my human's words, as the retained corpus's text (an image by reference
    beside its original's path, never its base64), their typed ids and reply target kept, door rows carried into
    the run (an edited schedule's every instruction) among them. ``source_text``: the owner message a routed start
    names, the door's own record of it, first, standing for the run's own door row with that row's reply target and
    message id. Without one, a suppressed (never-logged) origin's words live only in the run's first row and ride
    only under the door's stamp (``initial_user``): an unstamped first turn (``initial_text``) is never owner words.
    ``source``: a held row the start selected (``start_task`` ``owner_source``) opens it instead, under its own room,
    client id and time (``_selected_rows``). The schedule text the run holds (``schedule_items``) rides after them
    under its own label, never as theirs.
    """
    from ouroboros.review_evidence_sections import _owner_content_projection

    directives = [dict(row) for row in getattr(ctx, "_owner_directives", None) or [] if isinstance(row, dict)]
    owner = [row for row in directives if row.get("source") in OWNER_SOURCES]
    metadata = _ctx_metadata(ctx)
    # The run's own door row is the first it recorded itself; a door row carried into it is carried words.
    door = next((row for row in owner if row.get("source") == "initial_user" and not row.get("carried")), None)
    rest = [row for row in owner if row is not door]
    snapshot = _mapping(metadata.get("origin_message_snapshot"))
    origin = _mapping(metadata.get("origin_message_ref"))
    if door is not None:
        door = {**door, **_opening_identity(ctx)}
    if source is not None:
        rows = _selected_rows(ctx, door, rest, source)
    elif source_text or snapshot.get("placeholder_text"):
        opening = _opening_identity(ctx)
        current = next((row for index, row in enumerate(rest) if (row.get("source") == "origin_message" or _card_input_row(row)) and (
            _opening(row, origin, first=door is None and index == 0) or (
                opening and _same_corpus_source(row, opening)))), None)
        first = {**(current or door or {}), **opening}
        from ouroboros.project_dialogue import opening_input, opening_words

        if (not _card_input_row(first) or first.get("source") == OWNER_CARD_INPUT) and (
                typed := opening_input(None, None, retained=snapshot.get("input"))):
            first.update(opening_words({"ref": dict(origin), "text": source_text, "source_input": typed}))
        elif not _card_input_row(first):
            first.update(source="origin_message", content="" if snapshot.get("placeholder_text") else source_text)
        if snapshot.get("attachments"):
            first["attachments"] = snapshot["attachments"]
        rows = [first, *(row for row in rest if row is not current)]
    else:
        rows = [*([door] if door is not None and not source_text else []), *rest]
    rows = [{**row, "content": _owner_content_projection(row.get("content"))} for row in rows]
    return [row for row in rows if row["content"].strip() or row.get("attachments")] + schedule_items(ctx)


def _selected_rows(ctx: Any, door: Mapping[str, Any] | None, rest: List[Dict[str, Any]],
                   source: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The selected physical opening first, then the other words with their own identities and file descriptions.

    The previous opening keeps its door stamp (or the suppressed origin's turn id); an unnamed door stays carried
    context, never the opening. A duplicate delivery of the selected row adds no second set of words."""
    from ouroboros.chat_chain import _row_chat_id
    from ouroboros.answer_facts import retained_card_input
    from ouroboros.owner_relay import origin_files, owner_row_address, owner_row_identity, owner_text

    metadata = _ctx_metadata(ctx)
    identity = owner_row_identity(source)
    ids = {**({"client_message_id": str(source["client_message_id"])} if source.get("client_message_id")
              else {"msg_id": owner_row_address(source)}), "chat_id": str(identity["chat_id"]),
           "row_sha256": identity["row_sha256"], **({"ts": identity["ts"]} if identity["ts"] else {})}
    held = {**_mapping(getattr(ctx, "_held_owner_messages", None)), owner_row_address(source): source}
    physical = {owner_row_identity(row)["row_sha256"]: row for row in held.values() if isinstance(row, Mapping)}
    prior = ([dict(door)] if door is not None else []) + rest
    root = _data_root(ctx)
    files = origin_files(source, root)

    def selected(row):
        if row.get("row_sha256"):
            return row["row_sha256"] == ids["row_sha256"]
        twin = _held_twin(row, held, root=root, opening=ids["row_sha256"])
        return bool(twin and owner_row_identity(twin)["row_sha256"] == ids["row_sha256"])
    source_root = source.get("_owner_source_root") or root
    item = retained_card_input(source, source_root)
    if item.get("source") == OWNER_CARD_INPUT:
        known = next((row for row in rest if row.get("row_sha256") == ids["row_sha256"]
                      and _card_input_row(row) and row.get("source") in (OWNER_PRESS, "owner_mailbox", "owner_quiz_answer")), None)
        if known:
            item = {key: known[key] for key in ("source", "content", "source_task_id", "quiz_id", "reply_to") if key in known}
    item = item or {"source": "origin_message", "content": owner_text(source)}
    reply = item.get("reply_to") or selected_reply_target(source_root, source, (row.get("reply_to") for row in prior if selected(row)))
    rows = [{**item, **ids, **({"reply_to": dict(reply)} if reply else {}), **({"attachments": files} if files else {})}]
    origin, record = _mapping(metadata.get("origin_message_ref")), metadata.get("origin_message_text")
    own = _opening_identity(ctx)
    if door is not None and not own and metadata.get("client_message_id"):  # a suppressed origin's turn id
        own = {"client_message_id": str(metadata["client_message_id"]),
               "chat_id": str(_row_chat_id({"chat_id": getattr(ctx, "current_chat_id", None)}))}
    for index, row in enumerate(prior):
        if row.get("source") == "initial_user" and not row.get("carried") or _opening(row, origin, first=index == 0):
            door_words = (row.get("source") == "initial_user" and isinstance(record, str) and record.strip()
                          and not _card_input_row(row)
                          and not _mapping(metadata.get("origin_message_snapshot")).get("placeholder_text"))
            row = ({key: row[key] for key in ("reply_to", "attachments", "source_task_id", "quiz_id") if row.get(key)}
                   | ({**own, "source": row["source"] if _card_input_row(row) else "origin_message"}
                      if own else {"source": row["source"], "carried": True})
                   | {"content": record if door_words else row.get("content")})
        if selected(row):
            continue
        if not row.get("attachments") and (held_row := physical.get(row.get("row_sha256"))):
            if descriptors := origin_files(held_row, root):
                row = {**row, "attachments": [{k: v for k, v in file.items() if k != "upload"} for file in descriptors]}
        rows.append(row)
    return rows


def place_confirmed_words(confirmed: bool, text: Any, message: Any) -> str:
    """A place event's words as provenance when its author is an account the place confirms as my human's (and the
    connector does not report it unauthenticated), the same at a turn's opening and mid-turn. A forwarded message
    carries another author's words: context. Never a verdict about whom the words are for."""
    from ouroboros.places import forwarded_message

    words = str(text or "")
    return words if confirmed is True and words.strip() and not forwarded_message(message) else ""


def place_row_facts(conversation_id: Any, message: Any, origin: Mapping[str, Any] | None = None, *,
                    source: str = "") -> Dict[str, Any]:
    """A confirmed place event's corpus-row facts beside its words: the message it replies to (by address; with
    ``source``, the exact reader of the event record that carried its words) and its provider identity
    (``own_sends.message_identity``), so a later reader tells my own post through my human's account once that
    post's receipt is known; ``origin``: the row's other typed ids."""
    from ouroboros.own_sends import message_identity

    found = message_identity(conversation_id, message)
    identity = {"conversation_id": found[0], "message_id": found[1]} if found else {}
    return {"reply_to": place_reply_target(message, source), "origin": {**(origin or {}), **identity}}


def _my_post(ctx: Any, item: Mapping[str, Any]) -> bool:
    """A confirmed place row whose provider identity my send receipts now name: my own echo, never my human's."""
    identity = (str(item.get("conversation_id") or ""), str(item.get("message_id") or ""))
    if not all(identity) or not getattr(ctx, "drive_root", None):
        return False
    from ouroboros.own_sends import sent_identities
    from ouroboros.tool_access import canonical_data_root

    return identity in sent_identities(canonical_data_root(ctx))


def cited_owner_basis(ctx: Any, owner_words: Any, *, call_speech: bool = False) -> Tuple[Dict[str, Any] | None, str]:
    """``(basis, error)`` for words a run cites as my human's order for the work it starts.

    The host verifies provenance and nothing else: the exact bytes (``chronicle_quotes.quote_in``, emphasis
    markers aside) inside one message that reached THIS run and is my human's by provenance: an owner-door row
    of their words (``OWNER_WORD_SOURCES``: a press is an act, its label never theirs) or a place event from
    an account its place confirms (``PLACE_CONFIRMED``: confirmed,
    not unauthenticated, not forwarded). Absent words: ``(None, "")``, the work is my own. Words found: the row
    the new root carries (``basis: cited``) holds the whole verified message as ``content`` (never a host cut of my
    human's words) with the cited span as ``cited``, labelled by its real source: the owner door it came through,
    or ``place_owner`` for a confirmed account's place words, read again at the cite: a row my send receipt now names
    is my own echo (``_my_post``). Words not found: a typed refusal, nothing started.
    A wordless choice can instead cite its complete pressed label. Its original typed act and card ride together.
    The citing mind judges whether the words or choice apply to the work.
    The owner's decision of 11 November 2026 admits their own call speech only for a model choice:
    its caller opts in with ``call_speech`` and carries the returned ``live_call`` span only on the lead,
    never as the new work's owner corpus (the caption rules of 7 November otherwise stand)."""
    from ouroboros.answer_facts import word_spans
    from ouroboros.chronicle_quotes import quote_in

    wanted = str(owner_words or "").strip()
    if not wanted:
        return None, ""
    for row in directive_owner_rows(ctx):  # owner-door rows that reached this run (never a reply target's words)
        fact = _mapping(row.get("answer_input"))
        if row["source"] in OWNER_WORD_SOURCES and quote_in(wanted, word_spans(row["text"], fact)):
            if row["source"] == "owner_quiz_answer" or fact.get("card_id"):
                for item, _, ref, _ in _dated_directives(ctx):
                    if (item["source"] == row["source"] and item.get("content") == row["text"]
                            and (ref or row_ref(item)) == row["ref"]):
                        return {**item, "cited": wanted, "basis": CITED, "carried": True}, ""
            return {"source": row["source"], "content": row["text"], "cited": wanted, "basis": CITED, "carried": True,
                    **({"msg_id": row["ref"]} if row.get("ref") else {})}, ""
    for item, _, _, _ in _dated_directives(ctx):
        if item["source"] != OWNER_PRESS:
            continue
        # A quiz bubble has the host's one-based display prefix. A living action's label is already whole.
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if item.get("quiz_id") or is_answered_card(item.get("reply_to")):
            number, sep, label = content.partition(". ")
            if not sep or not number.isascii() or not number.isdecimal() or number.startswith("0"):
                continue
        else:
            label = _mapping(item.get("reply_to")).get("pressed_label")
        if isinstance(label, str) and wanted == label.strip():
            return {**item, "cited": label, "basis": CITED, "carried": True}, ""
    for item in getattr(ctx, "_owner_directives", None) or []:  # confirmed place events, with what they replied to
        if (isinstance(item, dict) and item.get("source") == PLACE_CONFIRMED
                and quote_in(wanted, str(item.get("content") or "")) and not _my_post(ctx, item)):
            return {"source": "place_owner", "content": str(item.get("content") or ""), "cited": wanted, "basis": CITED,
                    "carried": True, **{key: item[key] for key in ("msg_id", "reply_to") if item.get(key)}}, ""
    # Only the own-call owner-caption branch records live_call directives; heard speech and
    # Presence calls do not. A model choice may cite it without granting authority to the work.
    for item in (getattr(ctx, "_owner_directives", None) or []) if call_speech else ():
        if (isinstance(item, dict) and item.get("source") == "live_call"
                and quote_in(wanted, str(item.get("content") or ""))):
            return {"source": "live_call", "cited": wanted,
                    **({"msg_id": item["msg_id"]} if item.get("msg_id") else {})}, ""
    return None, ("owner_words were not found verbatim in any message of your human that reached this run (their "
                  "confirmed account, not forwarded). Copy their exact words or the complete label they pressed, or "
                  "omit owner_words to start the work as your own.")


def place_reply_target(message: Any, source: str = "") -> Dict[str, str] | None:
    """The replied message's provider address and any words the connector carried, as context under its author."""
    from ouroboros.presence_message_facts import reply_target

    return reply_target(message, source=source)


def owner_words_contain(ctx: Any, text: Any) -> bool:
    """``text`` is my human's own words that reached this run: exact, as a quote is (``chronicle_quotes.quote_in``),
    inside one row of their words in the run, never a reply target (another author's words beside theirs) or a
    press (an act). Provenance of bytes, never their meaning."""
    from ouroboros.answer_facts import word_spans
    from ouroboros.chronicle_quotes import quote_in

    wanted = str(text or "").strip()
    return bool(wanted) and any(quote_in(wanted, word_spans(row["text"], _mapping(row.get("answer_input"))))
                                for row in directive_owner_rows(ctx) if row["source"] in OWNER_WORD_SOURCES)


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
    """Prefer the complete row identity. Older carriers may name only their delivery or scoped client id."""
    if left.get("row_sha256") or right.get("row_sha256"):
        return left.get("row_sha256") == right.get("row_sha256")
    if left.get("ts") and right.get("ts") and left["ts"] != right["ts"]:
        return False
    left_id, right_id = str(left.get("client_message_id") or "").strip(), str(right.get("client_message_id") or "").strip()
    if left_id and left_id == right_id:
        left_chat, right_chat = str(left.get("chat_id") or "").strip(), str(right.get("chat_id") or "").strip()
        return not left_chat or not right_chat or left_chat == right_chat
    return bool(_corpus_keys(left) & _corpus_keys(right))


def _same_form_citation(prior: Mapping, row: Mapping) -> bool:
    """Attach a field citation only to its own identified submission, with the same recorded field boundaries."""
    from ouroboros.answer_facts import word_spans
    from ouroboros.chronicle_quotes import quote_in

    target = _mapping(prior.get("reply_to"))
    cited_target = _mapping(row.get("reply_to"))
    if (row.get("source") != "owner_mailbox" or not target.get("card_id")
            or target["card_id"] != cited_target.get("card_id")
            or prior.get("source_task_id") != row.get("source_task_id")
            or target.get("submitted_values") != cited_target.get("submitted_values")):
        return False
    if not (prior.get("msg_id") or prior.get("client_message_id")):
        return False  # equal words without a delivery identity do not establish a submission
    if any(prior.get(key) and row.get(key) and str(prior[key]) != str(row[key])
           for key in ("msg_id", "client_message_id", "chat_id")):
        return False
    return isinstance(row["cited"], str) and isinstance(prior.get("content"), str) and quote_in(
        row["cited"], word_spans(prior["content"], target))


def merged_corpus(kept: Any, added: Any) -> List[Dict[str, Any]]:
    """Merge carried rows without collapsing same client ids from different rooms."""
    merged: List[Dict[str, Any]] = []
    for row in [*(kept if isinstance(kept, list) else []), *(added if isinstance(added, list) else [])]:
        if not isinstance(row, Mapping):
            continue
        prior = next((item for item in merged if _same_corpus_source(row, item)), None)
        if prior is None:
            merged.append(dict(row))
        elif (row.get("basis") == CITED and row.get("cited") and prior.get("source") == row.get("source")
              and prior.get("content") == row.get("content")
              and (row.get("source") in (OWNER_PRESS, "owner_quiz_answer") or _same_form_citation(prior, row))):
            prior.update(basis=CITED, cited=row["cited"])
    return merged


def task_governing_words(task: Mapping[str, Any], *, addressed: bool = False) -> Tuple[List[Dict[str, str]], str]:
    """``(rows, absence marker)`` carried by value, read at the top level or under ``metadata``. ``addressed`` (an
    inspection's capture alone): a carried row keeps the coordinates it was carried with."""
    source = _mapping(task)
    nested = _mapping(source.get("metadata"))
    holder = source if (FIELD in source or ABSENT_FIELD in source) else nested
    raw = holder.get(FIELD)
    rows = []
    for item in raw if isinstance(raw, list) else ():
        if not isinstance(item, Mapping):
            continue
        row = _row(item.get("text"), source=item.get("source"), carrier=str(item.get("carrier") or ""),
                   task_id=item.get("task_id"), ts=item.get("ts"), ref=item.get("ref"), author=item.get("author"),
                   attachments=item.get("attachments"), row_sha256=item.get("row_sha256"))
        if _kept(row):
            for key in ("answer_input", "reply_citation"):
                if isinstance(fact := item.get(key), Mapping):
                    row[key] = dict(fact)
            if row["source"] == REPLY_TARGET and item.get("reply_view") in ("authored", "unread"):
                row["reply_view"] = item["reply_view"]
            if addressed and isinstance(item.get("address"), str):
                row["address"] = item["address"]
            if addressed and isinstance(origin := item.get("origin_ref"), Mapping):
                row["origin_ref"] = {key: origin[key] for key in ("chat_id", "client_message_id", "ts", "text_sha256", "row_sha256")
                                     if key in origin}
            rows.append(row)
    return rows, str(holder.get(ABSENT_FIELD) or "")


def _chat_words(row: Mapping[str, Any], *, key: str, carrier: str, source: str, task_id: str = "",
                root: Any = None, addressed: bool = False, attachments: Any = None) -> List[Dict[str, Any]]:
    """One chat row of my human's (held by a focus, or read from its room) as rows of the set: its words under the
    carriers' one reference (its room and client id; ``key``, its address, for a row with no client id), its files
    named beside them, then the card it answers, else the message it replies to, under that message's own author
    (``answered_card_reply``: the one card projection). ``root`` reads a chat reply target; without one the target
    rides by its address."""
    from ouroboros.answer_facts import card_input, row_source
    from ouroboros.owner_relay import owner_row_identity, owner_text

    reference, (block, asker) = _mapping(row.get("reply_to")), answered_card_source(row, root)
    living = card_input(row, root, source)
    replied = living.get("reply_to") or answered_card_reply(block, asker) or row.get("_owner_reply_target") or reply_target(
        root, reference if root is not None or reference.get("excerpt") else {**reference, "unresolved": True})
    carried = _mapping(row.get("corpus_answer"))
    item = living or {"source": row_source(row, source, root), "content": owner_text(row),
                      "reply_to": carried.get("reply_to") or replied,
                      "source_task_id": asker or carried.get("source_task_id"),
                      "quiz_id": _mapping(block).get("quiz_id") or carried.get("quiz_id")}
    item["row_sha256"] = owner_row_identity(row)["row_sha256"]
    found = _words_rows(item, carrier=carrier, task_id=task_id, ts=row.get("ts"), attachments=row.get("attachments") if attachments is None else attachments,
                       ref=_chat_ref(row) if row.get("client_message_id") else key)
    if addressed:
        found[0]["address"] = key
    return found


def _held_words(held: Any, drive_root: Any, task_id: str, *, context: Any = None, accounts: Any = None,
                read: bool = False, addressed: bool = False) -> List[Dict[str, Any]]:
    """The rows a focus holds by address (``owner_relay.held_owner_messages``) as rows of the set. ``context``
    selects by the carriage fact beside a held row: False, the rows that reached this work (a held value without
    a ``carried`` fact counts as one); True, the rows held as context, which went to another work; None, every
    row. A context row's carrier names where it went. A row is judged, and its reply target read, against the
    data root it was read from (``_owner_source_root``: a teller's root may differ from this work's), with that
    root's place snapshot; ``accounts`` is ``drive_root``'s."""
    from ouroboros.owner_relay import origin_files, quotable_owner_row

    rows: List[Dict[str, Any]] = []
    accounts_by_root = {str(drive_root): accounts}
    for key, row in _mapping(held).items():
        fact = _mapping(row.get("carried")) if isinstance(row, Mapping) else {}
        went = str(fact.get("went_to") or "")
        if not isinstance(row, Mapping) or (context is not None and bool(went) is not context):
            continue
        source_root = row.get("_owner_source_root") or drive_root
        if (source_accounts := accounts_by_root.get(str(source_root))) is None:
            source_accounts = accounts_by_root[str(source_root)] = owner_place_accounts(source_root)
        if quotable_owner_row(row, source_root, accounts=source_accounts):
            found = _chat_words(row, key=str(key), carrier=f"went to {went}" if went else "held",
                                source="owner_mailbox", task_id=task_id,
                                root=source_root if read or row.get("late_answer") or row.get("card_press") else None,
                                addressed=addressed, attachments=origin_files(row, drive_root))
            if found and (birth_ref := fact.get("birth_ref")):  # this row supplies the source, not new words
                found[0]["ref"] = birth_ref
            rows += found
    return rows


def _live_words(drive_root: Any, room: int, accounts: Any, marks: Any = None, *,
                addressed: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    """``(rows, gap lines)`` read live at a handoff: my human's rows of the bound Project room (carrier ``room``),
    the Project's retained origin rows (``origin``), then the rows that marks in force target (``marked``: the
    room's and the global marks; Main's for a tree outside every Project). ``marks``: every room's acting marks
    the caller's one capture already holds (a helper's birth inside ``memory_view.capture_memory_view``), cut here
    as the store cuts them for the room, so the chronicle is not entered once more; None (a review subject, a
    session's scheduling read) reads them from the store. Each store that cannot be read is one typed gap line,
    as is a room read that may lack rows (``room_appearances.read_gaps``); the rows that could be read stay, and
    nothing else is withheld. ``addressed`` (``project_words``): a chat row's words carry its chat address too."""
    from ouroboros.chat_chain import format_address, resolve_row, room_rows, row_address
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.owner_relay import quotable_owner_row
    from ouroboros.project_dialogue import opening_words, project_origin_rows
    from ouroboros.room_appearances import GAP_DETAILS

    def words(row: Any, carrier: str, source: str) -> List[Dict[str, Any]]:
        if not quotable_owner_row(row, drive_root, accounts=accounts):
            return []
        found = _chat_words(row, key=format_address(row_address(row)), carrier=carrier, source=source,
                            root=drive_root, addressed=addressed)
        if addressed and carrier == "room":  # only this row passed room_rows; its quoted target did not
            found[0]["room_id"] = str(room)
        return found

    rows: List[Dict[str, Any]] = []
    gaps: List[str] = []
    lacking: set = set()
    if room:
        try:
            for entry in room_rows(drive_root, room, gaps=lacking):
                rows += words(entry.row, "room", "room_message")
        except Exception as exc:  # a room that cannot be read is a named gap, never a withheld set
            log.warning("owner words: room %s unreadable at a handoff (%s: %s)", room, type(exc).__name__, exc)
            gaps.append(f"[Host fact: my human's rows of room {room} could not be read here "
                        f"({type(exc).__name__}); chat_history reads that room.]")
        if lacking:
            gaps.append(f"[Host fact: my human's rows of room {room} read here may lack some "
                        f"({'; '.join(GAP_DETAILS.get(kind, kind) for kind in sorted(lacking))}); chat_history reads that room.]")
        try:  # the registry's retained copies of the Project's first words: read whether or not the room read failed
            origin_gaps = set()
            for origin in project_origin_rows(drive_root, room, gaps=origin_gaps):
                ref = _mapping(origin.get("ref"))
                found = _words_rows(opening_words(origin), carrier="origin", task_id="",
                                    ref=_chat_ref(ref) if ref.get("client_message_id") else (
                                        format_address(ref) if ref.get("row_sha256") else ""))
                if addressed:
                    found[0]["origin_ref"] = ref
                rows.extend(found)
            gaps.extend(f"[Host fact: {gap}]" for gap in sorted(origin_gaps))
        except Exception as exc:
            log.warning("owner words: origin rows of room %s unreadable at a handoff (%s: %s)", room, type(exc).__name__, exc)
            gaps.append(f"[Host fact: the origin rows the Project retained for room {room} could not be read here "
                        f"({type(exc).__name__}); chat_history reads that room.]")
    try:
        key = str(room or WEB_UI_CHAT_ID)
        if marks is None:
            store = ChronicleStore(drive_root)
            marks = store.active_marks(key) if store.log_path.exists() else []
        else:  # the store's own cut of every room's acting marks: this room's and the global ones
            marks = [mark for mark in marks if str(mark.get("room_id", "")) == key or mark.get("scope", "room") == "global"]
        for mark in marks:
            target = _mapping(mark.get("target_ref"))
            if target.get("kind") == "chat_row" and target.get("row_sha256"):
                rows += words(resolve_row(drive_root, dict(target))[0], "marked", "marked_message")
    except Exception as exc:  # unreadable marks hide no row the tree or the room already gave
        log.warning("owner words: marks unreadable at a handoff (%s: %s)", type(exc).__name__, exc)
        gaps.append(f"[Host fact: the rows that marks in force target could not be read here "
                    f"({type(exc).__name__}); memory_read reads a mark and its target.]")
    return rows, gaps


def _data_root(ctx: Any) -> Any:
    """The canonical data root this run's chat rows live under, or None for a context that names no drive."""
    if not getattr(ctx, "drive_root", None):
        return None
    from ouroboros.tool_access import canonical_data_root

    return canonical_data_root(ctx)


def _tree_words(ctx: Any, *, read: bool = False) -> Tuple[List[Dict[str, Any]], Any, Any]:
    """``(rows, data root, place snapshot)`` addressed to this run's tree, by value: the field it inherited, the
    owner-sourced rows of its own corpus, and the rows it holds that reached this work (never a context row).
    Reads no dialogue; ``read`` (a render) also reads a held row's chat reply target."""
    held = _mapping(getattr(ctx, "_held_owner_messages", None))
    root = _data_root(ctx) if held or read else None
    accounts = owner_place_accounts(root) if root is not None else {}  # one snapshot for every row judged
    rows = task_governing_words(_ctx_metadata(ctx))[0] + directive_owner_rows(ctx) + _held_words(
        held, root, str(getattr(ctx, "task_id", "") or ""), context=False, accounts=accounts, read=read)
    return _deduped(rows), root, accounts


def _absent(ctx: Any) -> str:
    metadata = _ctx_metadata(ctx)
    return task_governing_words(metadata)[1] or _absent_marker(
        {"type": getattr(ctx, "current_task_type", None), "metadata": dict(metadata)})


def governing_words_for_schedule(ctx: Any, *, session: bool = False) -> Dict[str, Any]:
    """``{FIELD: rows}`` or ``{ABSENT_FIELD: marker}`` for the helper this run is about to schedule: the rows
    addressed to the tree, by value (``_tree_words``). A helper passes on what it inherited and what reached it
    since; the room, the origins and the marked rows are read live at the helper's own start (``work_words``).
    ``session`` (the helper compiles an external work order): the rows ``_live_words`` reads at this handoff (the
    bound room's rows, the Project's origin rows and the marked rows) ride beside them under ``ROOM_FIELD``, by
    value (an empty list when none), so that work order stays a function of the helper's record; a store that
    cannot be read gives the rows that could be, and its gap is logged. The schedule text the run holds rides beside
    them under ``SCHEDULE_FIELD``, by value: the helper's assignment never drops its tree's conditions."""
    rows = _tree_words(ctx)[0]
    facts: Dict[str, Any] = {FIELD: rows} if rows else {ABSENT_FIELD: _absent(ctx)}
    if schedule := schedule_items(ctx):
        facts[SCHEDULE_FIELD] = schedule
    if session:
        root, metadata, task_id = _data_root(ctx), _ctx_metadata(ctx), str(getattr(ctx, "task_id", "") or "")
        facts[ROOM_FIELD] = [] if root is None else _live_words(root, _room_of(
            root, task_id, metadata.get("parent_task_id"), metadata.get("root_task_id")), owner_place_accounts(root))[0]
    return facts


def _room_of(drive_root: Any, task_id: Any, parent_id: Any, root_id: Any) -> int:
    from ouroboros.projects_registry import project_chat_for_task_tree

    return int(project_chat_for_task_tree(drive_root, task_id, parent_id, root_id) or 0)


def work_words(ctx: Any) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """``(rows, absence marker, gap lines)``: my human's rows for this run's work as a handoff renders them. In
    order: the tree's rows (``_tree_words``), the rows held as context, the bound Project room's rows read once
    here, the Project's origin rows, the marked rows; the first row of one identity keeps its carrier and date."""
    rows, root, accounts = _tree_words(ctx, read=True)
    gaps: List[str] = []
    if root is not None:
        metadata, task_id = _ctx_metadata(ctx), str(getattr(ctx, "task_id", "") or "")
        live, gaps = _live_words(root, _room_of(root, task_id, metadata.get("parent_task_id"),
                                                metadata.get("root_task_id")), accounts)
        rows = _deduped(rows + _held_words(getattr(ctx, "_held_owner_messages", None), root, task_id, context=True,
                                           accounts=accounts, read=True) + live)
    return rows, "" if rows else _absent(ctx), gaps


def source_refs(item: Mapping[str, Any]) -> set:
    """Every reference a row names its message by: its typed identity (``row_ref``) and the one a carrier printed
    into its ``msg_id`` (``_chat_ref``'s ``chat N / client``), so one message under two formats is one source. A row
    that names no message has none: an empty reference never makes two rows one, and coincident words never do."""
    if item.get("row_sha256"):
        return {f"row-sha256:{item['row_sha256']}"}
    return {row_ref(item), str(item.get("msg_id") or "")} - {""}

def _listed_refs(row: Mapping[str, Any]) -> set:
    """``source_refs`` of a row Safety lists. A row without a digest that names its message by a room which is no number
    keeps the reference older carriers printed for that room (``chat c-1 / x``): never ``chat 1`` (``_row_chat_id``'s
    fallback), which is another room's, and no alias beside it."""
    chat, client = row.get("chat_id"), str(row.get("client_message_id") or "")
    if row.get("row_sha256") or not client or chat in (None, ""):
        return source_refs(row)
    try:
        int(chat)
    except (TypeError, ValueError):
        return {f"chat {chat} / {client}", str(row.get("msg_id") or "")} - {""}
    return source_refs(row)


def safety_sources(ctx: Any, *, _private: bool = False) -> List[Dict[str, Any]]:
    """The rows the Safety check reads for this run's work, shaped as its corpus rows (``source``, ``content``,
    ``msg_id`` = the row's reference, ``reply_to`` = the card or message it answers, ``ts`` = its time where one was
    recorded): the corpus, the field it inherited (a reply target folded into its row), then the rows it holds that
    reached this work; one row per source (``source_refs``), the first kept with the card and time a later copy
    adds, and the opening message dated by its door (``_opening``). Never a room, context or marked row: a
    permission my human gave another work is not this one's. ``_private`` (``safety_owner_ids``) keeps each row's exact
    ``row_sha256`` for the identities; it is never part of what the check shows. Reads the context's attributes and writes nothing."""
    from ouroboros.answer_facts import card_input, row_source
    from ouroboros.chat_chain import _FULL_DIGEST, format_address
    from ouroboros.dialogue_provenance import excerpt_citation
    from ouroboros.owner_relay import owner_row_address, owner_row_identity, owner_text

    rows: List[Dict[str, Any]] = []
    kept: Dict[str, Dict[str, Any]] = {}  # reference -> the listed row of that source
    origin = _mapping(_ctx_metadata(ctx).get("origin_message_ref"))
    fields = task_governing_words(_ctx_metadata(ctx))[0]

    def reaches(row: Any) -> bool:  # a held row that reached this work, not one held as context
        return isinstance(row, Mapping) and isinstance(row.get("carried"), dict) and not row["carried"].get("went_to")
    # Each canonical address with every full digest known for it: the field's typed rows (a room their reference names as
    # a number, with the time and digest the producer carried) and the rows held for this work. A corpus row recorded
    # without a digest takes one only when its whole msg_id is such an address and exactly one digest answers to it.
    known: Dict[str, set] = {}
    for row in fields:
        ref, digest = str(row["ref"]), row.get("row_sha256")
        room = ref[4:].split("@", 1)[0] if ref.startswith("row:") else ref[5:].split(" / ", 1)[0] if ref.startswith("chat ") else ""
        digits = room.removeprefix("-")
        if row["ts"] and isinstance(digest, str) and _FULL_DIGEST.fullmatch(digest) and digits.isascii() and digits.isdigit():
            known.setdefault(format_address({"chat_id": int(room), "ts": row["ts"], "row_sha256": digest}), set()).add(digest)
    for row in _mapping(getattr(ctx, "_held_owner_messages", None)).values():
        if reaches(row):
            known.setdefault(owner_row_address(row), set()).add(owner_row_identity(row)["row_sha256"])

    def add(row: Mapping[str, Any]) -> Dict[str, Any]:  # the listed copy of this source, gaps filled by a later one
        target = _mapping(row.get("reply_to"))
        if "excerpt" in target:
            row = {**row, "reply_to": {k: v for k, v in target.items() if k not in ("excerpt", "role")},
                   "reply_citation": excerpt_citation(target)}
        refs = _listed_refs(row)
        listed = next((kept[ref] for ref in refs if ref in kept), None)
        if listed is None:
            rows.append(listed := dict(row))
            kept.update(dict.fromkeys(refs, listed))
        listed.update({key: row[key] for key in ("reply_to", "ts") if row.get(key) and not listed.get(key)})
        if (row.get("reply_citation") and not listed.get("reply_citation") and target.get("address")
                and _mapping(listed.get("reply_to")).get("address") == target["address"]):
            listed.update(reply_citation=row["reply_citation"], reply_to=row["reply_to"])
        return listed
    for item in getattr(ctx, "_owner_directives", None) or []:
        if isinstance(item, Mapping):
            found = () if item.get("row_sha256") else known.get(str(item.get("msg_id") or ""), ())
            listed = add({**item, "row_sha256": next(iter(found))} if len(found) == 1 else item)
            if _opening(item, origin) and origin.get("ts") and not listed.get("ts"):
                listed["ts"] = str(origin["ts"])
    last: Dict[str, Any] | None = None  # the field row a following reply target belongs to
    bodies = _reply_bodies(fields)
    for row in fields:
        if row["source"] == REPLY_TARGET:
            if row.get("reply_view"):
                continue  # selected targets are paired by their citation, never by adjacency
            if last is not None and not last.get("reply_to"):
                last["reply_to"] = {"address": row["ref"], "author": row.get("author") or "", "text": row["text"]}
            last = None
            continue
        target = _carried_selection(row.get("reply_citation"), bodies)
        last = add({"source": row["source"], "content": row["text"], "msg_id": row["ref"],
                    **({"row_sha256": row["row_sha256"]} if row.get("row_sha256") else {}),
                    **({"ts": row["ts"]} if row["ts"] else {}), **({"reply_to": target} if target else {})})
    root = _data_root(ctx)
    for key, row in _mapping(getattr(ctx, "_held_owner_messages", None)).items():
        if reaches(row):
            source_root = row.get("_owner_source_root") or root
            item = card_input(row, source_root)
            replied = (item.get("reply_to") or answered_card_reply(*answered_card_source(row, source_root))
                       or row.get("_owner_reply_target") or reply_target(
                source_root, row.get("reply_to")))
            add({"source": item.get("source") or row_source(row, "owner_mailbox", source_root),
                 "content": item["content"] if item else owner_text(row), "msg_id": str(key),
                 "row_sha256": owner_row_identity(row)["row_sha256"],
                 **({"ts": when} if (when := _item_time(row)) else {}), **({"reply_to": replied} if replied else {})})
    return rows if _private else [{key: value for key, value in row.items() if key != "row_sha256"} for row in rows]


def safety_owner_ids(ctx: Any) -> List[set]:
    """One identity per row of my human's that ``safety_sources`` lists (``OWNER_SOURCES`` and ``PLACE_CONFIRMED``):
    the references it names its message by (``source_refs``), the opening message's with the door's (``_door_ref``),
    which its copy in a helper carries too; a row naming no message, its words and time. Two rows are one message
    when their identities meet. Reads the context's attributes and writes nothing."""
    from ouroboros.project_dialogue import _text_sha256

    origin = _mapping(_ctx_metadata(ctx).get("origin_message_ref"))
    door = source_refs(origin) if origin.get("row_sha256") else ({_door_ref(ctx)} if origin else set())
    return [_listed_refs(row) | (door if _opening(row, origin) else set())
            or {f"words {_text_sha256(row.get('content'))} {row.get('ts') or ''}"}
            for row in safety_sources(ctx, _private=True) if row.get("source") in OWNER_SOURCES | {PLACE_CONFIRMED}]


def _inbound_rows(drive_root: Any, client_ids: Mapping[str, str], accounts: Any = None) -> List[Dict[str, Any]]:
    """Owner rows addressed by client message id, in canonical order, judged over one place snapshot (the
    caller's, else one read here)."""
    from ouroboros import dialogue_store as ds
    from ouroboros.chat_chain import source_row_id

    rows = {}
    for entry in ds.iter_probes(drive_root, "chat", [("client_message_id", key) for key in client_ids]):
        row = entry.row
        if row.get("direction") == "in":
            rows[source_row_id(row)] = row
    accounts = owner_place_accounts(drive_root) if accounts is None else accounts
    return [row for row in rows.values() if owner_row(row, drive_root, accounts=accounts)]


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
    operation's one snapshot, else one read of ``drive_root``'s; unreadable: not) whose words are not another
    author's (``places.forwarded_message``: the account is still theirs, the words are not)."""
    if not (row and row.get("direction") == "in" and str(row.get("text") or "").strip()
            and not (row.get("system_type") or row.get("source") == "skill_repair")):
        return False
    from ouroboros.dialogue_provenance import row_provenance
    from ouroboros.places import forwarded_message, owner_confirmed_row  # D17->D20 is lazy-only

    if row_provenance(row) != "place":
        return True
    if forwarded_message(_mapping(row.get("transport")).get("message")):
        return False
    return owner_confirmed_row(row, owner_place_accounts(drive_root) if accounts is None else accounts)


def _message_rows(drive_root: Any, task_id: str, text: Any, ref: Any, carrier: str,
                  accounts: Any = None, held: Any = None, *, snapshot: Any = None) -> List[Dict[str, str]]:
    ref = _mapping(ref)
    snapshot = _mapping(snapshot)
    from ouroboros.answer_facts import retained_card_input
    from ouroboros.project_dialogue import opening_input, opening_words, resolve_owner_message_source
    from ouroboros.store import StoreError

    try:
        row = _held_origin(ref, held) or (resolve_owner_message_source(drive_root, dict(ref)) if ref else None)
    except (StoreError, OSError):
        if not (isinstance(text, str) and text.strip() or _mapping(snapshot).get("placeholder_text")):
            raise
        log.debug("Reply source unreadable; retaining recorded owner words", exc_info=True)
        row = None
    from ouroboros.owner_relay import origin_files, owner_row_address, owner_text
    from ouroboros.chat_chain import format_address
    reference = _carrier_ref(row, canonical=owner_row_address(row)) if row else (_chat_ref(ref) if ref else "")
    if not row and not ref.get("client_message_id") and ref.get("row_sha256"):
        reference = format_address(ref)
    source_root = _mapping(row).get("_owner_source_root") or drive_root
    files = snapshot.get("attachments") or (origin_files(row, drive_root) if row else [])
    live = retained_card_input(row, source_root) if row else {}
    if typed := opening_input(None, drive_root, retained=snapshot.get("input")):
        if not live or live["source"] == OWNER_CARD_INPUT and typed["source"] != OWNER_CARD_INPUT:
            live = opening_words({"ref": dict(ref), "text": text, "source_input": typed})
    if live:
        return _words_rows({**live, "row_sha256": ref.get("row_sha256")}, carrier=carrier, task_id=task_id, ref=reference,
                           ts=ref.get("ts") or _mapping(row).get("ts"), attachments=files)
    if _mapping(row).get("text_placeholder") or _mapping(snapshot).get("placeholder_text"):
        text = ""
    elif not (isinstance(text, str) and text.strip()):
        from ouroboros.owner_relay import quotable_owner_row
        if not quotable_owner_row(row, source_root, accounts=accounts if str(source_root) == str(drive_root) else None):
            return []
        text = owner_text(row)
    block, asker = answered_card_source(row, source_root)
    card = answered_card(block, text)
    target = (answered_card_reply(card, asker) if card else {}) or _mapping(row).get("_owner_reply_target") or reply_target(
        source_root, _mapping(row).get("reply_to"))
    return _words_rows({"source": "origin_message", "content": text, "reply_to": target, "row_sha256": ref.get("row_sha256")}, carrier=carrier,
                       task_id=task_id, ts=ref.get("ts") or _mapping(row).get("ts"), ref=reference, attachments=files)


def _corpus_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any], accounts: Any) -> List[Dict[str, str]]:
    """The words a follow-up, a drafted start or a Continue carries by value, as its producer recorded them; its
    ``origin_message`` row at the record's own origin address and time, as the door's row is stamped. An
    ``origin_message`` row naming another message (an earlier start's opening, carried on) keeps its own identity."""
    corpus = _mapping(record.get("metadata")).get("owner_corpus") or record.get("owner_corpus")
    origin = _mapping(record.get("origin_message_ref") or _mapping(record.get("metadata")).get("origin_message_ref"))
    stamp = {"ts": origin.get("ts"), "ref": _chat_ref(origin)} if origin else {}
    if origin.get("row_sha256") and not origin.get("client_message_id"):
        from ouroboros.chat_chain import format_address
        stamp["ref"] = format_address(origin)
    elif origin and not origin.get("client_message_id"):
        from ouroboros.project_dialogue import resolve_owner_message_source
        from ouroboros.owner_relay import owner_row_address
        from ouroboros.store import StoreError
        try:
            saved = (_held_origin(origin, record.get("held_owner_messages"))
                     or resolve_owner_message_source(drive_root, dict(origin)))
        except (StoreError, OSError):
            saved = None  # its carried words remain; an unreadable source supplies no new identity
        if saved:
            stamp["ref"] = _carrier_ref(saved, canonical=owner_row_address(saved))
    rows = []
    for index, item in enumerate(corpus if isinstance(corpus, list) else []):
        if not isinstance(item, Mapping) or item.get("source") not in OWNER_SOURCES:
            continue
        opening = item.get("source") == "origin_message" and _opening(item, origin, first=index == 0)
        if opening and origin.get("row_sha256"):
            item = {**item, "row_sha256": origin["row_sha256"]}
        rows += _words_rows(item, carrier="owner_corpus", task_id=task_id, **(stamp if opening else {}))
    nested = _mapping(record.get("metadata"))
    snapshot = _mapping(nested.get("origin_message_snapshot"))
    if origin and snapshot.get("input"):
        rows += _message_rows(drive_root, task_id, record.get("origin_message_text") or nested.get("origin_message_text"),
                              origin, "owner_corpus", accounts, record.get("held_owner_messages"), snapshot=snapshot)
    return rows


def _origin_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any], accounts: Any) -> List[Dict[str, str]]:
    nested = _mapping(record.get("metadata"))
    return _message_rows(drive_root, task_id,
                         record.get("origin_message_text") or nested.get("origin_message_text"),
                         record.get("origin_message_ref") or nested.get("origin_message_ref"), "origin_message", accounts,
                         record.get("held_owner_messages"), snapshot=nested.get("origin_message_snapshot"))


def _binding_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any], accounts: Any) -> List[Dict[str, str]]:
    from ouroboros.projects_registry import project_task_bindings

    binding = _mapping(project_task_bindings(drive_root).get(task_id))
    words = "" if binding.get("source_text_placeholder") else binding.get("source_text")
    return _message_rows(drive_root, task_id, words, binding.get("source_ref"), "binding",
                         accounts, record.get("held_owner_messages"), snapshot={
                             "placeholder_text": binding.get("source_text_placeholder"), "attachments": binding.get("source_attachments"),
                             "input": binding.get("source_input")})


def _annotation_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any],
                        accounts: Any) -> List[Dict[str, str]]:
    from ouroboros.owner_relay import owner_row_identity
    from ouroboros.chat_annotations import addressed_chat_annotations
    from ouroboros.chat_chain import format_address

    acts = {}
    for (_room, client_id, address, _token), row in addressed_chat_annotations(drive_root, by_token=True).items():
        if (row.get("action") in _ANNOTATED_ACTS and row.get("status") in _ANNOTATED_STATUSES
                and str(row.get("target") or "") == task_id):
            acts.setdefault((client_id, address), str(row["action"]))
    rows = []
    for row in (_inbound_rows(drive_root, {client for client, _ in acts}, accounts) if acts else []):
        action = acts.get((row["client_message_id"], format_address(owner_row_identity(row))))
        if action:
            rows.extend(_chat_words(row, key=_chat_ref(row), carrier="annotation", task_id=task_id, source=action,
                                    root=drive_root))
    return rows


def mailbox_owner_input(drive_root: Any, entry: Mapping[str, Any]) -> dict:
    """The owner input in an owner-text or card mailbox entry, with its card or chat reply target."""
    from ouroboros.answer_facts import card_input, row_source

    if submitted := card_input(entry, drive_root):
        return submitted
    block, asker = answered_card_source(entry)
    target = answered_card_reply(block, asker) or reply_target(drive_root, entry.get("reply_reference"))
    return {"source": row_source(entry, "owner_mailbox"), "content": entry.get("text"),
            **({"source_task_id": asker, "quiz_id": block["quiz_id"]} if block and block.get("quiz_id") else {}),
            **({"reply_to": target} if target else {})}


def _mailbox_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any], accounts: Any) -> List[Dict[str, str]]:
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, KIND_OWNER_TEXT, drain_owner_entries
    from ouroboros.owner_relay import origin_files, quotable_owner_row

    try:
        entries = drain_owner_entries(pathlib.Path(drive_root), task_id, include_acknowledged=True)
    except (OSError, ValueError):
        return []
    rows: List[Dict[str, str]] = []
    accounts_by_root = {str(drive_root): accounts}
    for entry in entries:
        if entry.get("kind") not in (KIND_OWNER_TEXT, KIND_CARD_INPUT):
            continue
        item = {**entry, **mailbox_owner_input(drive_root, entry)}
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
            rows.extend(_chat_words(row, key=_carrier_ref(row, canonical=canonical), carrier="relay", task_id=task_id,
                                    source="owner_mailbox", root=source_root,
                                    attachments=origin_files({**row, "_owner_source_root": source_root}, drive_root)))
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
    task holds (``owner_relay.held_owner_messages``) under the carriers' reference, so one message is one row, with
    the card it answers beside it."""
    record = dict(task) if isinstance(task, Mapping) else _task_record(drive_root, task_id)

    accounts = owner_place_accounts(drive_root)  # one snapshot for every row this read judges
    held = _held_words(record.get("held_owner_messages"), drive_root, task_id, accounts=accounts, read=True)
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


def render_owner_words(rows: Any, absent: str = "", *, audience: str, root_task_id: str = "", indent: str = "",
                       gaps: Any = (), current: bool = True) -> str:
    """One verbatim section for a child, a session, a reviewer, a plan review or the page writer. ``indent``
    prefixes every line of the words (the memory view's two spaces), so a line of the owner's own that starts with
    ``## `` never reads as a section of the request. A row's files are named on a host line under its words;
    ``gaps`` (``work_words``) close the section. ``current=False``: a session order whose record predates
    ``ROOM_FIELD`` keeps the literals it launched with (``_LAUNCHED_SESSION``)."""
    if audience not in _HEADINGS:
        raise ValueError(f"unknown owner-words audience {audience!r}; expected one of {sorted(_HEADINGS)}")
    rows = [row for row in (rows if isinstance(rows, list) else []) if isinstance(row, Mapping) and _kept(row)]
    gaps = [str(line) for line in gaps or () if str(line).strip()]
    if not rows:
        none = f"No words of my human are recorded for this work (host marker: {absent})." if absent else ""
        return "\n".join([*([none] if none else []), *gaps])
    origin = _ORIGIN.format(root=root_task_id) if root_task_id else ""
    heading, frame = _LAUNCHED_SESSION if audience == "session" and not current else (_HEADINGS[audience], _FRAMES[audience])
    return f"{heading}\n{frame.format(origin=origin)}\n" + "\n\n".join([*(words_block(row, indent) for row in rows), *gaps])


def words_block(row: Mapping[str, Any], indent: str = "") -> str:
    """One row as every section prints it: its stamp, its words (a reply target's unread fact) and its files line."""
    text = str(row.get("text") or "") or (unread_target(row) if row.get("source") == REPLY_TARGET else "")
    from ouroboros.answer_facts import submitted_fields_line
    from ouroboros.dialogue_provenance import excerpt_citation
    text = "\n".join(part for part in (text, submitted_fields_line(_mapping(row.get("answer_input"))),
                                     attachments_line(row.get("attachments")),
                                     excerpt_citation(_mapping(row.get("reply_citation")))) if part)
    return words_stamp(row) + "\n" + "\n".join(indent + line if line else line for line in text.split("\n"))


def project_words(drive_root: Any, room: int, *, open_rows: Any = (), indent: str = "") -> List[Dict[str, Any]]:
    """A project root's rows of its human that left the open conversation: the set's room rows (``_live_words``,
    carrier ``room``; never a reply target) whose reference or chat address no shown row (``open_rows``: lane 1, the origins) holds,
    oldest first, whole (``line``) and as the dated address the floor folds it into (``short``: its stamp, its size
    and the address that reads it; never a retelling); a store gap closes the list as a line with no address."""
    rows, gaps = _live_words(drive_root, room, owner_place_accounts(drive_root), [], addressed=True)
    held = {key for item in open_rows for key in (item.get("ref"), item.get("address")) if key}
    return [{"address": row["address"], "ref": row["ref"], "ts": row["ts"], "line": words_block(row, indent),
             "short": f"{words_stamp(row)} (words, {len(row['text'])} chars — read by address {row['address']})"}
            for row in _deduped(rows) if row.get("carrier") == "room" and row.get("source") != REPLY_TARGET
            and not {row["ref"], row["address"]} & held] + [{"line": line} for line in gaps]


def attachments_line(files: Any) -> str:
    """Files named by the host under the row's words: name, type, size, staged path or delivered download URL,
    or the fact that the file was not given to this work. ``""`` for a row with none."""
    named = []
    for ref in files if isinstance(files, list) else []:
        if isinstance(ref, Mapping):
            facts = ([str(ref["mime"])] * bool(ref.get("mime")) + [f"{ref['size']} bytes"] * (ref.get("size") is not None)
                     + [str(ref[key]) for key in ("staged", "download_url") if ref.get(key)])
            if ref.get("unavailable"):
                facts.append("unavailable")
            named.append(str(ref.get("name") or "attachment") + (f" ({', '.join(facts)})" if facts else ""))
    return f"[Host-retained attachments: {'; '.join(named)}]" if named else ""


def unread_target(row: Mapping[str, Any]) -> str:
    """What stands for a reply target's words the host did not carry: the host address that reads them, or the
    honest fact that the connector gave only a platform message id."""
    return UNREAD_TARGET if str(row.get("ref") or "").startswith("row:") else UNREAD_PLATFORM


def words_stamp(row: Mapping[str, Any]) -> str:
    """One row's stamp in every form, whole or by reference: ``[ts · owner · source of task · ref]``; a row held as
    context adds where it went (``went to <task>``) before its reference, the last part (a room, origin or marked
    row's source names its carrier); a reply target's own author, never my human (``REPLY_TARGET``)."""
    source, task_id, ref = str(row.get("source") or ""), str(row.get("task_id") or ""), str(row.get("ref") or "")
    carrier = str(row.get("carrier") or "")
    if source == REPLY_TARGET:  # context beside the words, under its own author: never my human's words
        stamp = [f"{row.get('author') or 'author not recorded'}: the message my human replied to, "
                 "not my human's words", ref]
    elif source == SCHEDULE_TEXT:  # the work's assignment under its recorded supplier: never my human's words
        stamp = [str(row.get("ts") or ""), ref, f"supplied by {row.get('author') or 'a supplier the host did not record'}",
                 "not my human's words"]
    else:
        stamp = [str(row.get("ts") or ""), "owner", f"{source} of task {task_id}" if task_id else source,
                 carrier if carrier.startswith("went to ") else "", ref]
    return "[" + " · ".join(part for part in stamp if part) + "]"


def work_words_text(ctx: Any, *, audience: str = "reviewer") -> str:
    """``render_owner_words`` over this run's set at this handoff (``work_words``): my human's words alone. Reviewers
    and the plan council read it with the schedule text appended; ``memory_read(owner_words=true)`` returns it whole."""
    rows, absent, gaps = work_words(ctx)
    root = str(_ctx_metadata(ctx).get("root_task_id") or getattr(ctx, "task_id", "") or "")
    return render_owner_words(rows, absent, audience=audience, root_task_id=root, gaps=gaps)


def owner_words_text(ctx: Any, *, audience: str = "reviewer") -> str:
    """``work_words_text``, then the schedule text the run holds (``render_schedule_text``)."""
    words = work_words_text(ctx, audience=audience)
    schedule = render_schedule_text(schedule_items(ctx), audience=audience)
    return "\n\n".join(part for part in (words, schedule) if part)


def reviewer_words_material() -> list:
    """Review-contract material (``commit_gate.commit_review_contract_fingerprint``): a reviewer's frame of the
    owner's words over a placeholder row, and the absence marker. The words and stamps are data."""
    row = {"text": "⟨words⟩", "source": "⟨source⟩", "task_id": "⟨task⟩", "ts": "⟨ts⟩", "ref": "⟨ref⟩"}
    return [render_owner_words([row], audience="reviewer", root_task_id="⟨root⟩"),
            *(render_owner_words([], marker, audience="reviewer") for marker in (NOT_RECORDED, "⟨marker⟩"))]
