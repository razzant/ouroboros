"""Which verified messages of my human carry cited words, and when each was written.

``owner_cites`` yields every basis ``owner_words.cited_owner_basis`` would accept, in its order. After a Panic a
mind's release needs my human's own message written after that Panic began (Emergency Stop Invariant, read
21 Nov 2026), so ``cited_owner_message`` dates each basis by its ONE message, read by identity from its canonical
record, and keeps the newest written. The host checks who wrote and when, never what the words mean:

- a chat message (the run's first one, a delivered one, an answer): its accepted row (``dialogue_store``), never a
  copy. A row a skill injected is store-and-forward: its writing time is the carrier's own send time (``sent_at``,
  Telegram's ``message.date``) when kept and not later than the host accepted it (``ts``); none, or a later one,
  proves no writing time (a bridge that reports none, a messenger's card tap). A window row's acceptance is its
  writing (``ts``);
- a confirmed place message: its event's own send time (``sent_at`` on its corpus item), else unknown;
- anything that cannot be resolved: unknown. Unknown is no authority after a Panic; the words still serve
  everything else.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

UNDATED = ("The host has no proven writing time for that message of your human (a bridge that did not report when "
           "it was sent, or a card tap through a messenger), so it gives no authority after a Panic. A new message "
           "from them through the window or an up-to-date bridge, or their own press, does.")


def owner_cites(ctx: Any, wanted: str, *, call_speech: bool = False) -> Iterator[Dict[str, Any]]:
    """Every verified basis of ``wanted`` in ``owner_words.cited_owner_basis``'s order (its first is that basis, whose
    docstring says what verified means)."""
    from ouroboros.answer_facts import word_spans
    from ouroboros.chronicle_quotes import quote_in
    from ouroboros.owner_words import (CITED, OWNER_PRESS, OWNER_WORD_SOURCES, PLACE_CONFIRMED, _dated_directives,
                                       _mapping, _my_post, directive_owner_rows, is_answered_card, row_ref)

    for row in directive_owner_rows(ctx):  # owner-door rows that reached this run (never a reply target's words)
        fact = _mapping(row.get("answer_input"))
        if row["source"] in OWNER_WORD_SOURCES and quote_in(wanted, word_spans(row["text"], fact)):
            if row["source"] == "owner_quiz_answer" or fact.get("card_id"):
                found = next((item for item, _, ref, _ in _dated_directives(ctx) if item["source"] == row["source"]
                              and item.get("content") == row["text"] and (ref or row_ref(item)) == row["ref"]), None)
                if found is not None:
                    yield {**found, "cited": wanted, "basis": CITED, "carried": True}
                    continue
            yield {"source": row["source"], "content": row["text"], "cited": wanted, "basis": CITED, "carried": True,
                    **({"msg_id": row["ref"]} if row.get("ref") else {})}
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
            yield {**item, "cited": label, "basis": CITED, "carried": True}
    for item in getattr(ctx, "_owner_directives", None) or []:  # confirmed place events, with what they replied to
        if (isinstance(item, dict) and item.get("source") == PLACE_CONFIRMED
                and quote_in(wanted, str(item.get("content") or "")) and not _my_post(ctx, item)):
            yield {"source": "place_owner", "content": str(item.get("content") or ""), "cited": wanted, "basis": CITED,
                    "carried": True, **{key: item[key] for key in ("msg_id", "reply_to") if item.get(key)}}
    # Only the own-call owner-caption branch records live_call directives; heard speech and
    # Presence calls do not. A model choice may cite it without granting authority to the work.
    for item in (getattr(ctx, "_owner_directives", None) or []) if call_speech else ():
        if (isinstance(item, dict) and item.get("source") == "live_call"
                and quote_in(wanted, str(item.get("content") or ""))):
            yield {"source": "live_call", "cited": wanted,
                    **({"msg_id": item["msg_id"]} if item.get("msg_id") else {})}




def _instant(value: Any) -> Optional[float]:
    from ouroboros.work_stop import _instant as parse

    return parse(value) if value else None


def _identity(ctx: Any, basis: Mapping[str, Any]) -> Optional[Tuple[Any, str]]:
    """``(chat_id, client_message_id)`` of the chat message a basis names, from the basis itself, the run's
    door-stamped first message, the corpus item or the held row that prints the same reference."""
    from ouroboros.owner_words import _chat_ref, _ctx_metadata, _door_ref, _mapping, row_ref

    if basis.get("chat_id") not in (None, "") and basis.get("client_message_id"):
        return basis["chat_id"], str(basis["client_message_id"])
    ref = str(basis.get("msg_id") or "")
    origin = _mapping(_ctx_metadata(ctx).get("origin_message_ref"))
    candidates = [origin] if ref and ref == _door_ref(ctx) else []
    candidates += [item for item in getattr(ctx, "_owner_directives", None) or []
                   if isinstance(item, Mapping) and ref in (row_ref(item), str(item.get("msg_id") or ""))]
    candidates += [row for row in _mapping(getattr(ctx, "_held_owner_messages", None)).values()
                   if isinstance(row, Mapping) and _chat_ref(row) == ref]
    for item in candidates if ref else ():
        if item.get("chat_id") not in (None, "") and item.get("client_message_id"):
            return item["chat_id"], str(item["client_message_id"])
    return None


def written_at(ctx: Any, basis: Mapping[str, Any]) -> str:
    """When the ONE message under ``basis`` was written, as above; ``""`` unknown."""
    if basis.get("source") == "place_owner":
        item = next((item for item in getattr(ctx, "_owner_directives", None) or [] if isinstance(item, Mapping)
                     and item.get("msg_id") == basis.get("msg_id") and item.get("content") == basis.get("content")), {})
        sent, seen = _instant(item.get("sent_at")), _instant(item.get("ts") or item.get("received"))
        return str(item["sent_at"]) if sent is not None and (seen is None or sent <= seen) else ""
    found = _identity(ctx, basis)
    if found is None:
        return ""
    from ouroboros import dialogue_store
    from ouroboros.owner_words import _data_root

    try:
        row = dialogue_store.accepted_row(_data_root(ctx), int(found[0]), found[1])
    except Exception:
        return ""
    row = dict(getattr(row, "row", None) or {})
    accepted = _instant(row.get("ts"))
    if accepted is None:
        return ""
    if not str(row.get("source") or "").startswith("skill:"):
        return str(row["ts"])
    sent = _instant(row.get("sent_at"))
    return str(row["sent_at"]) if sent is not None and sent <= accepted else ""


def cited_owner_message(ctx: Any, owner_words: Any) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """``(basis, written_at, error)``: of every verified message carrying the words, the one written newest; a found
    message whose time is unknown gives ``UNDATED``; none found gives ``cited_owner_basis``'s refusal."""
    from ouroboros.owner_words import cited_owner_basis

    wanted = str(owner_words or "").strip()
    dated = [(basis, written_at(ctx, basis)) for basis in owner_cites(ctx, wanted)] if wanted else []
    timed = [pair for pair in dated if _instant(pair[1]) is not None]
    if timed:
        basis, when = max(timed, key=lambda pair: _instant(pair[1]))
        return basis, when, ""
    if dated:
        return dated[0][0], "", UNDATED
    return None, "", cited_owner_basis(ctx, wanted)[1]
