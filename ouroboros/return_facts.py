"""Facts at my human's word: what happened elsewhere since my human's previous word to me.

At my human's word the host gives the focus that receives it facts, not text (memory design §2.15): when my
human's previous word to me was and where; their words elsewhere since; which places and people moved, with exact
readers of each person's rows and, beside them, each person's newest words whole; the works that ended, the rooms that
started and the cards opened to my human, by their own titles; which knowledge pages any focus changed, with the
writer's reason and the version's reader; and what is not known (rows still importing, reads that failed, connected
places with no arrival from anyone else, whose hearing nothing records). The mind decides what deserves a sentence;
nothing here asks it to say anything. My own messages and notices are not repeated: the view and the transcript hold
them.

A pure function of what the install holds: the dialogue store, the knowledge journals and the place registry; nothing
is stored. The lower bound is my human's previous word to me (``previous_word``: their row outside places and A2A that
``owner_speech`` confirms, card answers included; a late click's two rows are one word); a caller whose view reaches
further back or later (``since``) starts the rows where that view ends and page changes at the earlier of the two, no
view holding an edit as a change. ``Own`` is what the receiver already holds: at a turn start its room
(``## This room``), in a running root its own rows and page writes. With no connected place and nothing in the interval, no block; quiet connected places alone make
one line.

Three forms (``render``): full, the structure and then each person's newest words whole (a count alone updates
nothing: memory audit X1); middle, the structure, those words behind the readers on their place lines; last, the
interval and its counts beside the retained full text's reader, for the physical window only. At a turn start the
newest words are their own host section (``WORDS_ID``) beside the structure; both stand whatever a target or a selected
Low asks, and only where the window would otherwise lose the books or a telling do the words step to their readers,
then the structure to its last form (``memory_floor.TELLING_SPARED``; owner 18.11.2026, 1A: a count alone updated
nothing in the audit, the words did). Reads are bounded by the interval: one reverse read to the previous word, one
forward read, journal tails back to its start (an untouched journal is not opened), one indexed seek per quiet place.

A running root hears my human's later words through its drains (``place_before_words``): its facts start at what it
last saw (``REACH_KEY``: its view's capture, then the upper bound of the last facts it received; relayed or carried
words move nothing) and stand in one typed row before those words: one line, the interval and its counts, beside the
reader of the full facts retained in that work's artifacts (``_line``; owner 18.11.2026, 3A: a full row cost up to 2.4
million tokens a day on the owner's data, the line about 18 thousand; whether to read the rest is the mind's).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ouroboros import dialogue_store as ds
from ouroboros.chat_chain import format_address, row_address
from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.deadline_utils import ago_text
from ouroboros.dialogue_provenance import RoomLabelResolver, owner_speech, recorded_person, row_author, row_provenance
from ouroboros.knowledge_versions import JOURNAL_NAME, is_claim_commit
from ouroboros.text_sentences import first_sentence

SECTION_ID = "return_facts"  # = ``context_budget.RETURN_FACTS_KIND``, the running row's kind
HEADING = "## Since my human's previous word"
REACH_KEY = "_return_facts_reach"  # on a running root's context: ``{"seq", "ts"}`` its view and facts reach to
_BLOCK = 1 << 20
_LATE = "quiz_late_answer:"


@dataclass(frozen=True)
class Own:
    """What the receiving focus already holds: a room's rows (``## This room``), or a running root's own rows and
    page writes (its transcript). Its children's acts and other foci's are never its own."""

    room: Optional[int] = None
    root_task_id: str = ""

    def row(self, row: Mapping[str, Any]) -> bool:
        if self.root_task_id:
            return str(row.get("task_id") or "") == self.root_task_id
        return self.room is not None and _chat(row) == self.room

    def write(self, focus: Mapping[str, Any]) -> bool:
        return bool(self.root_task_id) and str(focus.get("task_id") or "") == self.root_task_id


@dataclass
class _Place:
    label: str
    namespace: str
    people: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)  # (id, name) -> count, rooms, newest
    counts: Dict[str, List[Any]] = field(default_factory=dict)  # automation / my own account / my replies / imports
    first: str = ""
    last: str = ""


def _chat(row: Mapping[str, Any]) -> Optional[int]:
    try:
        return int(row.get("chat_id"))
    except (TypeError, ValueError):
        return None


def _when(ts: Any) -> str:
    ts = str(ts or "")
    return ts[8:10] + "." + ts[5:7] + " " + ts[11:16] if len(ts) >= 16 else ts


def _seconds(ts: Any) -> Optional[float]:
    try:
        moment = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=_dt.timezone.utc)).timestamp()


def _words(text: Any) -> str:
    return " ".join(str(text or "").split())


def _title(text: Any) -> str:
    """A card's own title: its opening bold sentence as its author wrote it, else its first line."""
    text = str(text or "").strip()
    if text.startswith("**") and "**" in text[2:]:
        return _words(text[2:text.index("**", 2)])
    return _words(text.splitlines()[0] if text else "")


def _reader(first: ds.Row, last: ds.Row) -> str:
    return (f"memory_read(room_id='{_chat(last.row)}', rows=true, from='{format_address(row_address(first.row))}', "
            f"to='{format_address(row_address(last.row))}')")


def _answers(row: Mapping[str, Any], cards: Any) -> bool:
    """Whether a row is my human's answer to one of these cards (its answer row or its late delivery)."""
    return bool(cards) and (_late(row) or str(row.get("type") or "") == "quiz_answer") and _quiz_id(row) in cards


def _late(row: Mapping[str, Any]) -> bool:
    return str(row.get("client_message_id") or "").startswith(_LATE)


def _quiz_id(row: Mapping[str, Any]) -> str:
    """The card a late click or a card answer answers (``quiz_late_answer:<task>:<quiz>``, or its ``quiz`` block)."""
    client = str(row.get("client_message_id") or "")
    if client.startswith(_LATE):
        return client.rsplit(":", 1)[-1]
    quiz = row.get("quiz") if isinstance(row.get("quiz"), Mapping) else {}
    return str(quiz.get("quiz_id") or "")


def word_to_me(row: Mapping[str, Any], accounts: Any = None) -> bool:
    """My human's word to me: their row outside places and A2A traffic (Main, a room, a bound chat, a card answer)."""
    return (row_provenance(row) != "place" and not is_a2a_chat_id(row.get("chat_id"))
            and owner_speech(row, row_author(row, owner_accounts=accounts)))


def previous_word(root: Any, word: ds.Row, accounts: Any = None, skip: Any = ()) -> Optional[ds.Row]:
    """My human's newest word to me before ``word``; the other row of the same late click is the same word, and the
    words delivered with it (``skip``) are not before it."""
    before, card = word.seq, _quiz_id(word.row) if str(word.row.get("client_message_id") or "").startswith(_LATE) else ""
    while True:
        candidates = ds.newest_owner_candidates(root, before=before, limit=32)
        for entry in candidates:
            if (word_to_me(entry.row, accounts) and not (card and _quiz_id(entry.row) == card) and entry.seq not in skip
                    and (word.length or str(entry.row.get("ts") or "") < str(word.row.get("ts") or ""))):
                return entry  # a word with no row stands at the stream's end: one after its time is not before it
        if len(candidates) < 32:
            return None
        before = candidates[-1].seq


def journal_rows_between(journal: pathlib.Path, lower: str, upper: str, gaps: List[str]) -> List[Dict[str, Any]]:
    """A journal's rows stamped strictly between two instants, oldest first, read backwards from its end until the
    first row at or before ``lower``; a journal untouched since ``lower`` is not opened. A missing journal holds no
    writes; an unreadable one and rows that are not JSON are named in ``gaps``, never skipped silently."""
    name = f"{journal.parent.name}/{journal.name}"
    try:
        stat = journal.stat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        gaps.append(f"knowledge journal {name} unreadable: {type(exc).__name__}")
        return []
    lower_at = _seconds(lower)
    if lower_at is not None and stat.st_mtime < lower_at:
        return []
    found, position, tail, bad = [], stat.st_size, b"", 0
    try:
        with journal.open("rb") as handle:
            while position > 0:
                start = max(0, position - _BLOCK)
                handle.seek(start)
                lines = (handle.read(position - start) + tail).split(b"\n")
                tail, body = (lines[0], lines[1:]) if start else (b"", lines)
                for line in reversed(body):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        bad += 1
                        continue
                    stamp = str(row.get("ts") or "") if isinstance(row, dict) else ""
                    if stamp and stamp <= lower:
                        position = 0
                        break
                    if stamp and stamp < upper:
                        found.append(row)
                else:
                    position = start
    except OSError as exc:
        gaps.append(f"knowledge journal {name} unreadable: {type(exc).__name__}")
    if bad:
        gaps.append(f"knowledge journal {name}: {bad} row{'s' if bad != 1 else ''} not JSON")
    return found[::-1]


def reach(root: Any) -> Optional[Dict[str, Any]]:
    """Where a view captured now reaches: the chat stream's next position and this instant (read before the view, so
    no row falls between it and the facts a later word brings); ``None`` when the store cannot say."""
    try:
        return {"seq": ds.stream_end(root, "chat"), "ts": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    except Exception:  # no reach: a later word is told from my human's previous word, more rather than less
        logging.getLogger(__name__).debug("return facts: no reach recorded", exc_info=True)
        return None


def facts(root: Any, word: ds.Row, own: Own, *, upper: Optional[ds.Row] = None,
          since: Optional[Mapping[str, Any]] = None, delivered: Any = (), cards: Any = (),
          told: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """What happened elsewhere from the receiver's lower bound to ``upper`` (default ``word``, the first of the words
    it now receives); ``since`` is a running root's reach; ``delivered`` the positions of the words it receives with
    the block, which it holds already, and ``cards`` the cards those words answer: a card answer's other row (a late
    click writes two) is the same word, never one elsewhere. ``None`` when there is nothing to say."""
    from ouroboros.places import place_owner_accounts, place_rows

    root = pathlib.Path(root)
    try:
        accounts = place_owner_accounts(root)
    except Exception:  # an unreadable registry confirms no place account: place rows stay other people's
        accounts = None
    end = upper or word
    held = {word.seq, *delivered}
    answered = {card for card in (_quiz_id(word.row), _quiz_id(end.row), *cards) if card}
    previous = previous_word(root, word, accounts, skip=held)
    starts = ([(int(since["seq"]), str(since.get("ts") or ""))] if since else []) + (
        [(previous.seq + 1, str(previous.row.get("ts") or ""))] if previous is not None else [])
    if not starts:
        return None
    # A running root's frozen view holds the rows up to what it last saw: rows from there; page changes from the
    # earlier of that and my human's previous word, which no view holds as changes. A turn's start: both from the word.
    lower = starts[0][0]
    begin = min(starts)[1]
    labels, gaps = RoomLabelResolver(root), []
    coverage = ds.range_coverage(root, "chat", lower=lower, upper=end.seq)
    if not coverage.get("imported") or coverage.get("gaps"):
        gaps.append(f"chat rows of this interval: {coverage.get('pending', 0)} segments still importing, "
                    f"{coverage.get('mismatch', 0)} not importable, {coverage.get('gaps', 0)} lines unreadable")
    rows = [entry for entry in ds.iter_rows(root, "chat", lower=lower, upper=end.seq)
            if not own.row(entry.row) and entry.seq not in held and not _answers(entry.row, answered)]
    later = None  # an act without a row stands at the stream's end: rows after its time wait for the next facts
    if not end.length:
        cut = str(end.row.get("ts") or "")
        later = min((entry.seq for entry in rows if str(entry.row.get("ts") or "") > cut), default=None)
        rows = [entry for entry in rows if later is None or entry.seq < later]
    if told is not None:  # where the next facts begin, though these may say nothing
        told["later"] = later
    chats = {chat for chat in (_chat(entry.row) for entry in [previous, *rows] if entry is not None) if chat}
    transports = ds.latest_inbound_transport(root, chats)
    labels.warm(chats)  # rooms no registry names: their latest inbound transport, in one read
    found: Dict[str, Any] = {"previous": previous, "since": since, "begin": begin, "word": word, "end": end, "later": later,
                             "labels": labels, "mine": [], "places": {}, "ended": [], "started": [], "cards": [],
                             "heard": set(), "gaps": gaps}
    for entry in rows:
        _sort(found, entry, transports, accounts)
    found["pages"] = _page_writes(root, begin, str(end.row.get("ts") or ""), own, labels, gaps)
    try:  # a Silent place still hears; a Paused one hears nothing by its human's choice
        registry = [place for place in place_rows(root).values() if place.state in {"active", "silent"}]
    except Exception as exc:
        registry = []
        gaps.append(f"connected places unreadable: {type(exc).__name__}")
    quiet = [place for place in registry if place.place_id not in found["heard"]]
    found["quiet"] = _newest_arrivals(root, quiet, gaps, before=str(end.row.get("ts") or ""))
    if not any(found[key] for key in ("mine", "places", "ended", "started", "cards", "pages", "quiet", "gaps")):
        return None
    people = {key for place in found["places"].values() for key in place.people}
    found["keys"] = _keys(root) if people else {}
    return found


def _newest_arrivals(root: pathlib.Path, places: List[Any], gaps: List[str], *, before: str) -> List[Tuple[str, str]]:
    """``(label, newest recorded arrival before the upper bound)`` of each quiet place, one indexed seek each: the one
    hearing fact step 1 holds."""
    from ouroboros.place_arrivals import newest_arrivals

    try:
        newest = newest_arrivals(root, [place.place_id for place in places], before=before) if places else {}
    except Exception as exc:
        gaps.append(f"newest arrivals of quiet places unreadable: {type(exc).__name__}")
        newest = {}
    return sorted((place.label, newest.get(place.place_id, "")) for place in places)


def _sort(found: Dict[str, Any], entry: ds.Row, transports: Mapping[int, Any], accounts: Any) -> None:
    """One row of the interval into its fact: my human's words, a place's line, a room's event, or nothing."""
    row, chat = entry.row, _chat(entry.row)
    author = row_author(row, owner_accounts=accounts)
    if owner_speech(row, author) and not is_a2a_chat_id(chat):
        card = _quiz_id(row)
        if card and any(_quiz_id(seen.row) == card and (_late(seen.row) or _late(row)) for seen in found["mine"]):
            return  # a late click's two rows are one word (as ``previous_word``)
        found["mine"].append(entry)  # my human's words since: whole, theirs; their place was heard
        provenance = row.get("presence_provenance") if isinstance(row.get("presence_provenance"), dict) else {}
        found["heard"].update(str(value) for value in (row.get("place_id"), provenance.get("place_id")) if value)
        return
    transport = row.get("transport") if isinstance(row.get("transport"), dict) else transports.get(chat) or {}
    provider, kind = str(transport.get("provider") or ""), str(row.get("type") or "")
    if provider:
        if kind == "task_summary":
            return  # a place turn's terminal: its reply, if any, is among my replies
        conversation = transport.get("conversation") if isinstance(transport.get("conversation"), dict) else {}
        key = (provider, str(transport.get("account_id") or ""), str(transport.get("conversation_id") or chat))
        name = next((value for value in (conversation.get(field) for field in ("full_name", "name", "title"))
                     if isinstance(value, str) and value.strip()), "")  # scalar names only, as ``_room_details``
        place = found["places"].setdefault(key, _Place(_words(name) or found["labels"].label(row), key[1]))
        place.first, place.last = place.first or str(row.get("ts") or ""), str(row.get("ts") or "")
        actor = transport.get("actor") if isinstance(transport.get("actor"), dict) else {}
        mine = row.get("direction") == "out" or author.get("kind") == "ouroboros" or author.get("via") == "own_account"
        if not mine:  # someone else's row: this place was heard in the interval
            provenance = row.get("presence_provenance") if isinstance(row.get("presence_provenance"), dict) else {}
            found["heard"].update(str(value) for value in (row.get("place_id"), provenance.get("place_id")) if value)
        if isinstance(row.get("historical_import"), dict):
            label = "imported earlier messages"
        elif mine:  # an event of the account I send as that no work of mine sent is its own fact
            label = "my replies" if row.get("direction") == "out" or author.get("via") != "own_account" else "my own account"
        else:
            if author.get("is_bot"):
                label = "automation"
            elif not recorded_person(author):
                label = "unattributed arrivals"
            else:
                actor_id = str(actor.get("platform_actor_id") or actor.get("id") or "")  # as ``dialogue_provenance``
                name = _words(actor.get("display_name") or actor.get("username") or row.get("sender_label"))
                held = place.people.setdefault(actor_id or name, {"count": 0, "rooms": {}, "id": actor_id, "name": name})
                held["count"] += 1  # one person by their id: a name they changed meanwhile is their newest
                held["newest"], held["name"] = entry, name or held["name"]
                held["forward"] = bool(author.get("forwarded"))  # its sender's row, another author's words
                span = held["rooms"].setdefault(chat, [entry, entry])
                span[1] = entry
                return
        held = place.counts.setdefault(label, [0, entry])
        held[0], held[1] = held[0] + 1, entry
        return
    if isinstance(row.get("historical_import"), dict):
        return
    if kind == "task_summary" and row.get("summary_kind") == "terminal_root_projection":
        found["ended"].append(entry)
    elif kind == "project_started":
        found["started"].append(entry)
    elif kind == "quiz":
        found["cards"].append(entry)


def _keys(root: pathlib.Path) -> Dict[Tuple[Any, Any, Any], Dict[str, List[str]]]:
    from ouroboros.knowledge import key_pages

    try:
        return key_pages(root)
    except (OSError, ValueError):  # an unreadable shelf joins nothing: the keys are still printed
        return {}


def _page_writes(root: pathlib.Path, begin: str, end: str, own: Own, labels: RoomLabelResolver,
                 gaps: List[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Page writes in the interval, one entry per page in first-write order: count, writers, last, reason, version."""
    journals = [("global", root / "memory" / JOURNAL_NAME)]
    try:
        journals += sorted((f"project:{path.parent.name}", path) for path in root.glob(f"projects/*/{JOURNAL_NAME}"))
    except OSError as exc:
        gaps.append(f"project knowledge journals unreadable: {type(exc).__name__}")
    written = [(scope, row) for scope, journal in journals for row in journal_rows_between(journal, begin, end, gaps)
               if isinstance(row, dict) and row.get("topic") and not is_claim_commit(row)
               and not own.write(row.get("focus") if isinstance(row.get("focus"), Mapping) else {})]
    labels.warm({_chat(row["focus"]) for _scope, row in written if isinstance(row.get("focus"), Mapping)} - {None})
    pages: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for scope, row in written:
        focus = row.get("focus") if isinstance(row.get("focus"), Mapping) else {}
        where = labels.label({"chat_id": focus["chat_id"]}) if focus.get("chat_id") not in (None, "") else ""
        writer = " in ".join(part for part in (str(focus.get("role") or row.get("writer") or "unknown"), where) if part)
        page = pages.setdefault((scope, str(row["topic"])), {"count": 0, "writers": [], "last": "", "why": "",
                                                              "why_by": "", "why_at": "", "revision": ""})
        page["count"] += 1
        if writer not in page["writers"]:
            page["writers"].append(writer)
        page["last"], page["revision"] = str(row.get("ts") or ""), str(row.get("new_sha256") or "")
        if _words(row.get("reason")):  # a reason stays its own write's: a later write without one never inherits it
            page["why"], page["why_by"], page["why_at"] = _words(row.get("reason")), writer, str(row.get("ts") or "")
    return pages


def _said(row: Mapping[str, Any]) -> str:
    """My human's word as they gave it: a card answer by its choice and words beside the card's own title (the
    host's frame around it is not theirs), anything else whole."""
    quiz = row.get("quiz") if isinstance(row.get("quiz"), Mapping) else None
    if quiz is None or row.get("type") not in {"quiz_answer", "quiz_closed"}:
        return f"«{str(row.get('text') or '').strip()}»"
    options, index = quiz.get("options") if isinstance(quiz.get("options"), list) else [], quiz.get("answered_index")
    chosen = options[index] if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(options) else None
    label = _words(chosen.get("label") if isinstance(chosen, Mapping) else chosen)
    parts = ([f"chose «{label}»"] if label else []) + ([f"with the words «{str(quiz['comment']).strip()}»"]
                                                        if _words(quiz.get("comment")) else [])
    return f"card answer: {'; '.join(parts) or 'closed it'}; the card: «{_title(quiz.get('question'))}»"


def _utc(ts: Any) -> str:
    return str(ts or "")[:16].replace("T", " ") + " UTC"


def _word_line(labels: RoomLabelResolver, row: Mapping[str, Any], at: float) -> str:
    source = str(row.get("source") or "")
    return (f"{labels.label(row)}{'' if source in ('', 'web') else ' via ' + source}, "
            f"{str(row.get('ts') or '')[:16].replace('T', ' ')} UTC, {ago_text(at - (_seconds(row.get('ts')) or at))}")


WORDS_ID = "return_facts_words"
WORDS_HEADING = "## Their newest words since my human's previous word"


def render(found: Mapping[str, Any]) -> Tuple[str, str, str]:
    """The three forms of one set of facts: full (the structure, then each person's newest words whole), middle (the
    structure: those words behind the readers on their place lines) and last (the interval and its counts, for the
    physical window only, beside the retained full text's reader)."""
    structure, words, totals = _forms(found)
    return (f"{structure}\n\n{words}" if words else structure), structure, totals


def _forms(found: Mapping[str, Any]) -> Tuple[str, str, str]:
    return _structure(found), _newest_words(found), _totals(found)


def _head(found: Mapping[str, Any]) -> List[str]:
    labels, word, previous, since = found["labels"], found["word"], found["previous"], found["since"]
    at = _seconds(word.row.get("ts")) or 0
    lines = [HEADING]
    if previous is not None:
        lines.append(f"My human's previous word: {_word_line(labels, previous.row, at)}.")
    if since:  # a running root: what its view and earlier facts reach, and the spans this row covers
        lines.append(f"This work's view and earlier facts reach to {_utc(since['ts'])}; rows from then, page changes "
                     f"from {_utc(found['begin'])}, to my human's word of {_utc(found['end'].row.get('ts'))}.")
    return lines


def _people(found: Mapping[str, Any]):
    """Each place's people, newest first, with the place they wrote in: ``(place key, place, (actor, name), held)``."""
    for key, place in sorted(found["places"].items(), key=lambda kv: kv[1].last, reverse=True):
        for held in sorted(place.people.values(), key=lambda held: held["newest"].seq, reverse=True):
            yield key, place, (held["id"], held["name"]), held


def _persons(found: Mapping[str, Any], *, speaking: bool = False) -> set:
    """Each person once per provider and account, by their id, else their name; ``speaking``: only people whose
    newest words have text."""
    return {(key[0], key[1], person[0] or person[1]) for key, _place, person, held in _people(found)
            if not speaking or str(held["newest"].row.get("text") or "").strip()}


def _structure(found: Mapping[str, Any]) -> str:
    labels = found["labels"]
    lines = _head(found)
    mine = [f"- {labels.label(entry.row)}, {_when(entry.row.get('ts'))}: {_said(entry.row)}; {_reader(entry, entry)}"
            for entry in found["mine"]]
    lines += ["My human's words elsewhere since then:", *mine] if mine else []
    people: Dict[Tuple[str, str, str], List[str]] = {}
    for (provider, namespace, conversation), _place, (actor, name), held in _people(found):
        pages = (found["keys"].get((provider, namespace, actor)) or {}).get("active") or []
        people.setdefault((provider, namespace, conversation), []).append(
            f"{name or actor} ({actor}){''.join(' → ' + reader for reader in pages)} ×{held['count']}, newest "
            f"{_when(held['newest'].row.get('ts'))}, " + ", ".join(_reader(first, last) for first, last in held["rooms"].values()))
    places = []
    for key, place in sorted(found["places"].items(), key=lambda kv: kv[1].last, reverse=True):
        parts = people.get(key, []) + [f"{label} ×{count}, newest {_reader(newest, newest)}"
                                       for label, (count, newest) in place.counts.items()]
        named = f" [{key[1]}]" if key[1] and place.people else ""  # a person's key: provider, this, their id
        places.append(f"- {key[0]} · {place.label}{named}, {_when(place.first)}–{_when(place.last)}: " + "; ".join(parts))
    lines += ["Places since then:", *places] if places else []
    works = [f"- ended: {labels.label(e.row)} · {e.row.get('target_label') or e.row.get('suggested_name') or e.row.get('task_id')}: "
             f"{e.row.get('outcome') or e.row.get('status')} {_when(e.row.get('ts'))}"
             + (f" «{first_sentence(_words(e.row['wake_summary']))}»" if e.row.get("wake_summary") else "")
             + f"; get_task_result(task_id='{e.row.get('task_id')}')" for e in found["ended"]]
    works += [f"- started: {_title(e.row.get('text'))} {_when(e.row.get('ts'))}; {_reader(e, e)}" for e in found["started"]]
    works += [f"- card to my human in {labels.label(e.row)}, {_when(e.row.get('ts'))}: «{_title(e.row.get('text'))}»; "
              f"{_reader(e, e)}" for e in found["cards"]]
    lines += ["Works since then:", *works] if works else []
    pages = [f"- knowledge {scope}/{topic}: {page['count']} write{'s' if page['count'] != 1 else ''} by "
             f"{', '.join(page['writers'])}; last {_when(page['last'])}" + (
                 "" if not page["why"] else f"; why: «{page['why']}»" if page["why_at"] == page["last"]
                 else f"; the last write gave no reason; earlier, {page['why_by']} {_when(page['why_at'])}: «{page['why']}»")
             + f"; knowledge_read(topic={topic!r}, scope={scope!r}, version={page['revision']!r})"
             for (scope, topic), page in found["pages"].items()]
    lines += ["Pages changed since then:", *pages] if pages else []
    unknown = list(found["gaps"])
    if found["quiet"]:
        count = len(found["quiet"])
        alone = not any(found[key] for key in ("mine", "places", "ended", "started", "cards", "pages"))
        unknown.append(f"no arrival from anyone else recorded since then in {count} connected place"
                       f"{'s' if count != 1 else ''}; nothing records how far a place was heard: "
                       "configure_presence(action='list')" if alone else
                       "connected places with no arrival from anyone else recorded since then (nothing records how far "
                       "a place was heard), each with its newest recorded arrival before my human's word: "
                       + "; ".join(f"{label} ({_when(newest) or 'none recorded'})" for label, newest in found["quiet"]))
    lines += ["Not known:", *(f"- {line}" for line in unknown)] if unknown else []
    return "\n".join(lines)


def _newest_words(found: Mapping[str, Any]) -> str:
    """Each person's newest words whole, one line each with the place and time the structure names."""
    lines = [f"- {key[0]} · {place.label} · {name or actor} ({actor}), {_when(held['newest'].row.get('ts'))}"
             f"{' (a forward: another author' + chr(39) + 's words)' if held.get('forward') else ''}: "
             f"«{str(held['newest'].row.get('text') or '').strip()}»"
             for key, place, (actor, name), held in _people(found) if str(held["newest"].row.get("text") or "").strip()]
    return "\n".join([WORDS_HEADING, *lines]) if lines else ""


def _counted(count: int, one: str, many: str = "") -> str:
    return f"{count} {one if count == 1 else many or one + 's'}"


def _totals(found: Mapping[str, Any]) -> str:
    """The last form: the interval and what it holds, counted; the rest stands at the retained text's reader."""
    people = [held for _key, _place, _person, held in _people(found)]
    persons = _persons(found)
    others = sum(count for place in found["places"].values() for count, _newest in place.counts.values())
    writes = sum(page["count"] for page in found["pages"].values())
    parts = [_counted(len(found["mine"]), "word") + " of my human elsewhere" if found["mine"] else "",
             f"{_counted(len(found['places']), 'place')}: {_counted(len(persons), 'person', 'people')} "
             f"({_counted(sum(held['count'] for held in people), 'message')}), {_counted(others, 'other row')}"
             if found["places"] else "",
             _counted(len(found["ended"]), "work") + " ended" if found["ended"] else "",
             _counted(len(found["started"]), "room") + " started" if found["started"] else "",
             _counted(len(found["cards"]), "card") + " to my human" if found["cards"] else "",
             f"{_counted(writes, 'page write')} on {_counted(len(found['pages']), 'page')}" if found["pages"] else "",
             _counted(len(found["quiet"]), "connected place") + " with no arrival from anyone else" if found["quiet"] else "",
             _counted(len(found["gaps"]), "line") + " not known" if found["gaps"] else ""]
    return "\n".join([*_head(found), "Since then, counted: " + "; ".join(part for part in parts if part) + "."])


def turn_sections(task: Mapping[str, Any], root: Any, ctx: Any = None, seen: Optional[Mapping[str, Any]] = None) -> List[Any]:
    """The sections of a turn my human's word starts (the facts, then the people's newest words), or none. On ``ctx``
    it records where this root's view and facts reach (``REACH_KEY``): the bound of the facts a later word of my
    human brings it while it runs; for a turn my human's word starts, that word (what follows it is told later, never
    lost between). A read that fails is named in the section; the turn goes on."""
    from ouroboros.section_floor import HostSection

    if ctx is not None and seen:  # the reach read before this turn's memory view; none read, none recorded (no
        setattr(ctx, REACH_KEY, dict(seen))  # second read after the view: a later word is told from the previous word)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    ref = next((value for value in (task.get("origin_message_ref"), metadata.get("origin_message_ref"))
                if isinstance(value, Mapping) and value.get("client_message_id")), {})
    if not task.get("_is_direct_chat") or not ref:
        return []
    try:
        chat = _chat(ref) if ref.get("chat_id") is not None else _chat(task)
        word = ds.accepted_row(root, chat, str(ref["client_message_id"]))
        if word is None or not word_to_me(word.row):
            return []
        if ctx is not None:
            setattr(ctx, REACH_KEY, {"seq": word.seq + 1, "ts": str(word.row.get("ts") or "")})
        found = facts(root, word, Own(room=chat))
        if found is None:
            return []
        structure, words, totals = _forms(found)
    except Exception as exc:  # unknown is never "nothing happened": the gap is named, the turn goes on
        return [HostSection(SECTION_ID, f"{HEADING}\nNot known: these facts could not be read ({type(exc).__name__}).")]
    people = len(_persons(found, speaking=True))
    return [HostSection(SECTION_ID, structure, totals), *([HostSection(WORDS_ID, words, (
        f"{WORDS_HEADING}\nThe newest words of {people} {'person' if people == 1 else 'people'} stand at the readers "
        "on their place lines above."))] if words else [])]


def _line(root: pathlib.Path, task_id: str, found: Mapping[str, Any]) -> str:
    """A running root's row: the interval and its counts, beside the reader of the full facts retained now in this
    work's artifacts (``section_floor.retain_section``; written once, digest-named, kept with the work's sources). A
    copy that cannot be kept leaves the counts and names the gap; the full form never stands here."""
    from ouroboros.section_floor import HostSection, retain_section

    full, _middle, totals = render(found)
    try:
        held = retain_section(HostSection(SECTION_ID, full, totals), root=str(root), task_id=task_id)
    except (OSError, ValueError) as exc:
        return f"{totals}\nNot known: the full facts could not be kept for reading ({type(exc).__name__})."
    return f"{totals}\nThe full facts (places, people and their newest words, works, pages, readers): {held['reader']}."


# My human's inputs this drain carried: their words, and their acts typed apart from words (a press without words,
# a card submission no longer readable), which equally owe a reply after what happened since.
_HEARD = frozenset({"direct_incoming", "owner_mailbox", "owner_quiz_answer", "owner_press", "owner_card_input"})


def _word_row(root: pathlib.Path, origin: Mapping[str, Any], chat: Optional[int], task_id: str = "") -> Optional[ds.Row]:
    """The dialogue row of one delivered word, by the room and client id its delivery recorded; a card's answer or
    closure delivered as ``<kind>:<quiz>`` by its canonical row ``<kind>:<task>:<quiz>`` (``dialogue_evidence``)."""
    client = str(origin.get("client_message_id") or "")
    room = _chat(origin) if origin.get("chat_id") not in (None, "") else chat
    msg = str(origin.get("msg_id") or "")
    kind = next((kind for kind in ("quiz_closed", "quiz_answer") if msg.startswith(kind + ":")), "")
    if not client and kind and task_id:
        client, room = f"{kind}:{task_id}:{msg[len(kind) + 1:]}", None
    if not client:
        return None
    found = [entry for entry in ds.probe_rows(root, "chat", [("client_message_id", client)], limit=8)
             if room is None or _chat(entry.row) == room]
    return found[0] if found else None


def _pressed(root: pathlib.Path, origin: Mapping[str, Any]) -> Optional[ds.Row]:
    """My human's act that wrote no chat row (a press on a live card, a card closed: ``card_input``, ``quiz_closed``
    and late answers deliver only the act): their word at the chat stream's end, at the time its delivery recorded,
    length 0; the facts then count rows up to that time (``facts``)."""
    at = str(origin.get("ts") or origin.get("received") or "")
    if origin.get("client_message_id") or not at:
        return None
    return ds.Row("chat", ds.stream_end(root, "chat"), {"ts": at, "type": "quiz_answer", "direction": "system",
                                                        "quiz": {"quiz_id": str(origin.get("quiz_id") or "")}}, None, 0)


def place_before_words(ctx: Any, messages: List[Dict[str, Any]], start: int, heard_before: int) -> None:
    """One row of facts before the first of my human's words this drain appended to a running root: from what the
    root last saw (``REACH_KEY``) or my human's previous word, the earlier, to the last of those words, as one line
    beside the reader of the full facts (``_line``); then its reach moves there. Nothing sent is rewritten: a later
    drain brings its own row before its own words, and renewal keeps the newest run of words with these rows whole
    (``context_renewal._cut_units``). A child or a delivery without its word's row gets none; a failed read is named."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND, RETURN_FACTS_KIND

    if ctx is None or (getattr(ctx, "task_metadata", None) or {}).get("delegation_role") == "subagent":
        return
    first = next((index for index in range(start, len(messages))
                  if messages[index].get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND), None)
    directives = [row for row in list(getattr(ctx, "_owner_directives", None) or []) if isinstance(row, dict)]
    fresh = [row for row in directives[heard_before:] if row.get("source") in _HEARD and not row.get("carried")]
    if first is None or not fresh:
        return
    root = pathlib.Path(getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", ""))
    chat = getattr(ctx, "current_chat_id", None)
    try:
        task_id = str(getattr(ctx, "task_id", "") or "")
        words = [row for row in (_word_row(root, origin, chat, task_id) or _pressed(root, origin) for origin in fresh)
                 if row is not None and word_to_me(row.row)]
        if not words:
            return
        told: Dict[str, Any] = {}
        found = facts(root, words[0], Own(root_task_id=task_id), upper=words[-1], since=getattr(ctx, REACH_KEY, None),
                      delivered=[row.seq for row in words], cards=[_quiz_id(row.row) for row in words], told=told)
        text = _line(root, task_id, found) if found is not None else ""
        later = told.get("later")
        setattr(ctx, REACH_KEY, {"seq": later if later is not None else words[-1].seq + (1 if words[-1].length else 0),
                                 "ts": str(words[-1].row.get("ts") or "")})
    except Exception as exc:  # unknown is never "nothing happened": named, and the words still reach the work
        text = f"{HEADING}\nNot known: these facts could not be read ({type(exc).__name__})."
    if text:  # renewal keeps the row with my human's words (``context_renewal._cut_units``)
        messages.insert(first, {"role": "user", HOST_CONTEXT_KIND_KEY: RETURN_FACTS_KIND, "content": text})
