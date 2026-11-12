"""My other conversations in every integrating focus: ``## Other conversations``, after ``## This room``.

Every focus that integrates -- a Main turn, a root, a wake of consciousness, a reply in a place -- holds every
other conversation of mine at some granularity (BIBLE P1 "The subject of continuity"); the focus's role decides
nothing here, a delegated child or nanny carries none of it (``ViewSpec.other_conversations``). One group per
conversation -- a place conversation without its thread, a Project, Main -- in a stable order by its key; the
focus's own room is ``## This room``. A group names where it is and who reads it (``RoomLabelResolver``), its
running work, and each of its rooms that has open rows, a note of mine or running work:

- a place room with a recorded living person's words or my outward reply (sealed rows too), or any non-place
  room, is **engaged**: its open words stay whole, notes stay whole, and tasks carry their host facts and first gist
  sentence. Without my human's confirmed words or forward or my outward reply it is **overheard**: line forms
  count its unread words with the conversation's other overheard rooms. A bot inside a conversation keeps its
  words too; children never travel as reports;
- arrivals from bots or my own unconfirmed account alone are **events**: the host names their first text,
  last recorded author, counts and reader. A tighter ordinary form counts such rooms per conversation.
  Arrivals without a recorded author are counted in the event line (owner choice A). Notes remain mine
  and stay whole even beside events; running work remains named in the group. These host forms owe no telling.

A writer's terminal rows its page links below it (``memory_sealing.linked_rows``) are that page's, never open
here. A room's later delivery of words it already holds (the same ``text_sha256``, not my human's; by the same
recorded speaker, or by none, with the same connector facts), and one row several rooms hold by membership beyond the one keeping it whole
(``_once``), is one line naming that earlier copy and its reader; its text is not read.
The capture keeps consideration and author facts. A replacement is my telling of the
same source printed in this block beside its address, a host fact, or a recorded
window/mode emergency. ``memory_forms.STAGES`` orders remote ordinary forms first,
then this room's intermediate replies, remote untold words, this room's final replies,
remote last-gist counts and story/notes; owner words stay whole until the physical floor.
The clean-start guard excludes emergencies when ordinary forms fit the complete frame.
Rendering and source classification live in ``memory_view_conversation_forms``. Texts are read
only for the open words a group may show, in one read;
sizes come from stored ``text_chars``.
"""
from __future__ import annotations

import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from ouroboros import chat_chain, memory_inventory, memory_view_events as events
from ouroboros.context_timing import timed_section
from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
from ouroboros.dialogue_provenance import (owner_speech, place_row_facts, recorded_person, recorded_speaker, row_class,
                                           warm_lineage)

_RANKS = ("retold", "considered", "nobody considered")  # F6w folds the first first
MAIN = str(WEB_UI_CHAT_ID)
Entry = Tuple[Dict[str, Any], Dict[str, Any], int]


def _instant(value: Any) -> Optional[datetime.datetime]:
    try:
        moment = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)


def conversation_of(room: str, entries: List[Entry], notes: List[Dict[str, Any]],
                    projects: Iterable[int]) -> Tuple[str, Dict[str, Any]]:
    """``(key, place facts)``: a place conversation is its provider, account and conversation, without the thread;
    Main and each Project are their own; any other room is its own too."""
    if room == MAIN:
        return "main", {}
    if room.lstrip("-").isdigit() and int(room) in projects:
        return f"project:{room}", {}
    for _address, meta, _pos in reversed(entries):
        facts = place_row_facts(meta)
        if facts.get("provider") and facts.get("conversation_id"):
            return ":".join(str(facts.get(key) or "") for key in ("provider", "account_id", "conversation_id")), facts
    key = next((str(note["origin"].get("conversation_key")) for note in notes
                if note.get("origin", {}).get("conversation_key")), "")
    return (key.rsplit(":", 1)[0], {}) if key.count(":") >= 3 else (f"room:{room}", {})


def _task_line(root_id: str, group: List[Tuple[Entry, Dict[str, Any]]], summaries: Mapping[str, Any]) -> Dict[str, Any]:
    """One line for a task's lane-2 rows in another conversation: the host's word for its ending from its stored
    facts, the root's closing sentence and its reader; children by count, never their words."""
    from ouroboros.memory_view import _HOST_FACTS, _KINDLESS, _TERMINAL, _minute
    from ouroboros.memory_view_work import first_sentence

    def chosen(kinds: frozenset, kind: str = "task_summary") -> Optional[Entry]:  # ``memory_view._task_line``'s order
        return next((entry for entry, _cls in reversed(group) if entry[1].get("type") == kind
                     and entry[1].get("summary_kind") in kinds and str(entry[1].get("task_id") or "") == root_id), None)

    terminal = chosen(_TERMINAL) or chosen(_KINDLESS, "project_completion_summary") or chosen(_HOST_FACTS)
    # The card's stored word ("Stopped", "Done"), as this task's own room reads it; a row without one keeps its status.
    status = (str(terminal[1].get("outcome") or terminal[1].get("status") or "status not recorded") if terminal is not None
              else "running or unreported")
    sentence = first_sentence((summaries.get(root_id) or {}).get("summary", ""))
    children = {str(entry[1].get("task_id")) for entry, _cls in group
                if entry[1].get("task_id") and str(entry[1].get("task_id")) != root_id}
    first, last = group[0][0], group[-1][0]
    parts = ([f"children {len(children)}"] if children else []) + [f"rows {len(group)}", f"get_task_result(task_id='{root_id}')"]
    head = f"[{_minute(first[1].get('ts') or '?')}; host; task {root_id}] "
    return {"task": root_id, "line": head + "; ".join([status, *([f"«{sentence}»"] if sentence else []), *parts]),
            "line_without_gist": head + "; ".join([status, *parts]),
            "pos": first[2], "last_pos": last[2], "chars": sum(int(e[1].get("text_chars") or 0) for e, _c in group)}


@timed_section("memory_view.conversations")
def capture(store: Any, root: Any, spec: Any, label: Callable[..., str], by_room: Mapping[str, List[Entry]],
            notes: Mapping[str, List[Dict[str, Any]]], lineage: Mapping[str, Any], *, quiet: Mapping[str, List[Entry]],
            sealed: Mapping[str, List[Entry]], gaps: Mapping[str, set], projects: Any,
            summaries: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """The other conversations, by key. The full catalogue includes the focus room when calculating event readers;
    the visible groups then exclude it (``spec.room_id``). From the rows ``memory_inventory.open_rows_by_room`` read for this capture."""
    from ouroboros import memory_view as mv
    from ouroboros.memory_room_activity import considered_through, running_by_room
    from ouroboros.memory_sealing import linked_rows
    from ouroboros.memory_view_work import gists

    summaries = gists(by_room, chat_chain.format_address, quiet, sealed) if summaries is None else summaries
    membership = memory_inventory.membership_facts(root)
    running = running_by_room(root, by_room, notes, membership.bindings, projects or [])
    rooms = sorted(set(by_room) | set(notes) | {room for room, kinds in gaps.items() if kinds}
                   | {room for room, tasks in running.items() if tasks})
    linked = linked_rows(root, store, by_room, projects or ())
    here = {entry[0]["row_sha256"] for entry in by_room.get(str(spec.room_id), ())} if spec.room_lanes else set()
    # Labels come from the rows in hand (open and sealed): a room's newest row that came over a transport, whose tail
    # metadata 7 keeps its conversation's name and audience. Only rooms none of whose rows carries them, and their
    # conversations, read the newest inbound row's transport, all in one read (``warm``), never a row per room.
    in_hand = {room: [*sealed.get(room, ()), *by_room.get(room, ())] for room in rooms}
    samples = {room: next((meta for _address, meta, _pos in reversed(rows) if meta.get("transport")
                           or meta.get("presence_provenance")), None) for room, rows in in_hand.items()}
    unnamed = [room for room, meta in samples.items() if _scalars([meta] if meta else []) is None]
    places = (conversation_of(room, in_hand[room], notes.get(room, []), membership.project_chat_ids)[1]
              for room in unnamed)
    getattr(label, "warm", lambda _rooms: None)([*unnamed, *(_conversation_chat(place) for place in places if place)])
    warm_lineage(lineage, ((meta, pos) for entries in sealed.values() for _address, meta, pos in entries))
    groups: Dict[str, Dict[str, Any]] = {}
    shown: List[Tuple[Dict[str, Any], Entry, Dict[str, Any], Any]] = []  # (room, row, author, earlier copy or None)
    for room in rooms:
        held = by_room.get(room, [])
        entries = [entry for entry in held if entry[0]["row_sha256"] not in linked]
        if held and not entries and not notes.get(room) and not running.get(room) and not gaps.get(room):
            continue  # all that is open here are pages' own later rows, linked below them: nothing waits here
        classed = [(entry, row_class(entry[1], pos=entry[2], **lineage)) for entry in entries]
        spoken = [(entry, cls) for entry, cls in classed if cls["lane"] == 1]
        key, place = conversation_of(room, held or sealed.get(room, []), notes.get(room, []), membership.project_chat_ids)
        own = max((entry[2] for entry, cls in spoken if _outward(cls["author"])), default=-1)  # §3.4 (i)
        ours = any(_ours(entry[1], cls["author"]) for entry, cls in spoken)
        participated = not place or own >= 0 or any(cls["author"].get("owner_confirmed") is True for _entry, cls in spoken)
        first = entries[0][2] if entries else -1
        for _address, meta, pos in reversed(sealed.get(room, ())):
            if ours and participated and (pos < first or first < 0):  # older rows can add neither fact nor consideration
                break
            author = row_class(meta, pos=pos, **lineage)
            if author["lane"] == 1:
                participated |= author["author"].get("owner_confirmed") is True
            if author["lane"] == 1 and _outward(author["author"]):
                ours, own = True, max(own, pos) if pos > first else own
                participated = True
            elif author["lane"] == 1 and _ours(meta, author["author"]):
                ours = True
        engaged = not place or ours
        fold = max(((pos, meta) for _a, meta, pos in held if meta.get("fold_read") is True and meta.get("wake_summary")),
                   key=lambda pair: pair[0], default=None)  # S5: a finish that read here folded it into its sentence
        considered = max(own, considered_through(quiet.get(room, ()), lineage), fold[0] if fold else -1)
        noted = max((_instant(note.get("ts")) for note in notes.get(room, ()) if _instant(note.get("ts"))), default=None)
        retold = max((pos for _a, meta, pos in held if meta.get("wake_summary")), default=-1)
        sample = samples[room]
        group = groups.setdefault(key, {"key": key, "rooms": [], "running": [], "latest": "", "place": bool(place)})
        if place and "label" not in group:  # a conversation; its rooms carry their own ids
            group["label"] = conversation_label(label, place, [meta for _a, meta, _p in in_hand[room]])
        facts = memory_inventory.room_facts(store, room, entries, lineage, len(notes.get(room, ())),
                                            classes=[cls for _entry, cls in classed])
        latest = str(facts.get("latest") or "")
        group["latest"] = max(group["latest"], latest)
        group["running"] = sorted({*group["running"], *running.get(room, ())})
        newest = max(entries, key=lambda entry: entry[2]) if entries else None
        item: Dict[str, Any] = {"room_id": room, "label": label(room, sample), "engaged": engaged, "facts": facts,
                                "overheard": bool(place) and engaged and not participated,
                                "first": first, "notes": [], "words": [], "tasks": [], "silent": "", "folded": {},
                                "newest": chat_chain.format_address(newest[0]) if newest else "",
                                "last": newest[2] if newest else -1}
        group.setdefault("label", item["label"])
        group["rooms"].append(item)
        if not engaged:
            previous = [(e, row_class(e[1], pos=e[2], **lineage)) for e in sealed.get(room, ())]
            item["events"] = events.capture([*previous, *spoken])
        item["notes"] = list(notes.get(room, ()))
        again = _repeats(sorted((*sealed.get(room, ()), *held), key=lambda entry: entry[2]))
        for entry, cls in (pair for pair in spoken if pair[0][0]["row_sha256"] not in here):  # once, in this room
            if room == str(spec.room_id) or not engaged and not (
                cls["author"]["kind"] == "place_fact" and entry[1].get("direction") == "out"
            ):
                continue
            # My human's words stay whole, and so does a relay row, which may carry them quoted (``relay_quotes``):
            # a fold of what a finish read stands for its room's other words only.
            if (fold and entry[2] <= fold[0] and not owner_speech(entry[1], cls["author"])
                    and entry[1].get("type") != "relay_entry"):
                folded = item["folded"] or {"sentence": str(fold[1]["wake_summary"]), "ts": str(fold[1].get("ts") or ""),
                                            "task": str(fold[1].get("task_id") or ""),
                                            "count": 0, "chars": 0, "from": chat_chain.format_address(entry[0])}
                item["folded"] = {**folded, "count": folded["count"] + 1, "to": chat_chain.format_address(entry[0]),
                                  "chars": folded["chars"] + int(entry[1].get("text_chars") or 0)}
                continue
            rank = 0 if entry[2] <= retold else 1 if not _unconsidered(entry, cls["author"], considered) or (
                noted is not None and (_instant(entry[1].get("ts")) or noted) <= noted) else 2
            shown.append((item, entry, {**cls["author"], "rank": rank},
                          None if owner_speech(entry[1], cls["author"]) else again.get(entry[0]["row_sha256"])))
        tasks: Dict[str, List[Tuple[Entry, Dict[str, Any]]]] = {}
        for entry, cls in classed:
            task = str(entry[1].get("root_task_id") or entry[1].get("task_id") or "")
            if cls["lane"] == 2 and task:
                tasks.setdefault(task, []).append((entry, cls))
        item["tasks"] = sorted((_task_line(task, rows, summaries) for task, rows in tasks.items()), key=lambda line: line["pos"])
    focus = _running_focus(root, {task for group in groups.values() for task in group["running"]})
    for group in groups.values():
        group["running"] = [f"get_task_result(task_id='{task}')" + (f": {focus[task]}" if task in focus else "")
                            for task in group["running"]]
        group["rooms"].sort(key=lambda room: (room["first"] if room["first"] >= 0 else 1 << 62, room["room_id"]))
    shown = _once(shown, [room for key in sorted(groups) for room in groups[key]["rooms"]])
    event_rooms = [room for group in groups.values() for room in group["rooms"] if room.get("events")]
    texts = mv._row_texts(root, [entry for _item, entry, _author, earlier in shown if earlier is None]
                         + [entry for room in event_rooms for entry in events.text_entries(room["events"])])
    for room in event_rooms:
        events.finish_capture(room["events"], texts)
    for item, entry, author, earlier in shown:
        word = mv._spoken(entry, author, texts)
        if earlier is not None:  # the earlier copy's words: one line naming it, this row's text never read
            first, where = chat_chain.format_address(earlier[0]), earlier[1] or item["room_id"]
            same = "the same words as" if earlier[1] is None else f"the same row as in room {where}:"
            word.update(again=first, line=f"{word['head']} delivered again: {same} {first}; "
                        f"memory_read(room_id='{where}', rows=true, from='{first}', to='{first}')")
        item["words"].append({**word, "id": f"{item['room_id']}:{word['address']}", "rank": author["rank"]})
    ordered = [groups[key] for key in sorted(groups)]
    if event_rooms or any(room["overheard"] or len(group["rooms"]) > 1 for group in ordered for room in group["rooms"]):
        events.locate_catalogue(ordered)  # event, unread and several-room headers read their place in it
    for group in ordered:
        group["rooms"] = [room for room in group["rooms"] if room["room_id"] != str(spec.room_id)]
        group["latest"] = max((str(room["facts"].get("latest") or "") for room in group["rooms"]), default="")
        visible = sorted({task for room in group["rooms"] for task in running.get(room["room_id"], ())})
        group["running"] = [f"get_task_result(task_id='{task}')" + (f": {focus[task]}" if task in focus else "")
                            for task in visible]
    return [group for group in ordered if group["rooms"]]


def _once(shown: List[Tuple[Dict[str, Any], Entry, Dict[str, Any], Any]],
          order: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Entry, Dict[str, Any], Any]]:
    """Each canonical row (its ``row_sha256``) whole once in the block: a row several of my rooms hold by membership
    (``memory_inventory.rooms_of_row``: where it was written, the Projects its binding names) keeps its words in the
    room it was written in when that room shows it, else in the first room shown; every other room keeps its entry
    as one line naming that room and the row's reader (``(address, room)``). Membership, attribution and counts stay;
    only the same row, never equal words of another row (``_repeats``) or similar words."""
    rank = {id(item): n for n, item in enumerate(order)}
    holder: Dict[str, Tuple[Tuple[int, int], Dict[str, Any]]] = {}
    for item, entry, _author, earlier in shown:
        if earlier is None:
            key = (0 if item["room_id"] == str(chat_chain._row_chat_id(entry[1])) else 1, rank.get(id(item), 1 << 30))
            held = holder.get(entry[0]["row_sha256"])
            if held is None or key < held[0]:
                holder[entry[0]["row_sha256"]] = (key, item)
    marked = []
    for item, entry, author, earlier in shown:
        if earlier is not None:  # a later copy of words this room already holds (``_repeats``)
            earlier = (earlier, None)
        elif holder[entry[0]["row_sha256"]][1] is not item:
            earlier = (entry[0], holder[entry[0]["row_sha256"]][1]["room_id"])
        marked.append((item, entry, author, earlier))
    return marked


def conversation_label(label: Callable[..., str], place: Mapping[str, Any], rows: List[Mapping[str, Any]]) -> str:
    """A place conversation's own name, as each of its rooms' labels begins (``RoomLabelResolver``): its provider,
    account and conversation without a thread, named by the newest of ``rows`` that carries the conversation's scalars."""
    flow = {**{key: place.get(key) for key in ("provider", "account_id", "conversation_id")}, "thread_id": ""}
    scalars = _scalars(rows)
    named = label(_conversation_chat(place), {"transport": {**flow, **({"conversation": scalars} if scalars else {})}})
    return named.rsplit(" [chat_id=", 1)[0]


def _scalars(rows: List[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """The conversation's own name and audience (tail metadata 7) from the newest of ``rows`` that carries them."""
    carried = (row["transport"] for row in reversed(rows) if isinstance(row.get("transport"), Mapping))
    return next((transport["conversation"] for transport in carried if isinstance(transport.get("conversation"), Mapping)),
                None)


def _conversation_chat(place: Mapping[str, Any]) -> int:
    """A place conversation's own chat id, without a thread: the room ``conversation_label`` names it by."""
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    keys = [place.get(key) for key in ("provider", "account_id", "conversation_id")]
    return presence_chat_id(conversation_key(*keys, ""))


def _repeats(rows: List[Entry]) -> Dict[str, Dict[str, Any]]:
    """``row_sha256 -> the earlier copy's address`` of a room's later deliveries of words it already holds, its rows
    (sealed and open) in position order, the same non-empty words (``text_sha256``, whitespace aside): a row with no
    recorded speaker repeats the room's earliest copy, a recorded speaker's row only that speaker's own earliest copy
    (never an unrecorded one), and the same connector-facts digest from row_meta. Differing facts keep their own
    row and reader; edited words and wordless rows are none. No similarity, prefix or length."""
    first: Dict[Tuple[str, str, str], Dict[str, Any]] = {}  # (words, speaker or "" for any, facts) -> earliest copy
    found: Dict[str, Dict[str, Any]] = {}
    for address, meta, _pos in rows:
        digest = meta.get("text_sha256") if int(meta.get("text_chars") or 0) else ""
        if digest:
            speaker, facts = recorded_speaker(meta), str(meta.get("message_facts_sha256") or "")
            earlier = first.get((digest, speaker, facts))
            if earlier is not None and earlier["row_sha256"] != address["row_sha256"]:
                found[address["row_sha256"]] = earlier
            first.setdefault((digest, "", facts), address)
            first.setdefault((digest, speaker, facts), address)
    return found


def _ours(row: Mapping[str, Any], author: Mapping[str, Any]) -> bool:
    """A recorded living person or my outward reply, including before the open frontier."""
    return events.starts_conversation(author)


def _outward(author: Mapping[str, Any]) -> bool:
    """My own outward row: a reply or delivery of mine, which considers what came before it; an arrival from my own
    account (``via == "own_account"``: an operator session or a ``gh`` call publishing as me) considers nothing."""
    return author.get("kind") == "ouroboros" and author.get("via") != "own_account"


def _unconsidered(entry: Entry, author: Mapping[str, Any], considered: int) -> bool:
    """A recorded person's words after the considered point, not my human's: words nobody considered (a silent
    room's count, ``F6w``'s last rank). A reaction, a row with no recorded speaker, my words and my account's are not."""
    return recorded_person(author) and entry[2] > considered and not owner_speech(entry[1], author)


def nobody_considered(spoken: List[Tuple[Entry, Mapping[str, Any]]], considered: int) -> int:
    """How many of a room's words nobody considered (``_unconsidered``)."""
    return sum(1 for entry, cls in spoken if _unconsidered(entry, cls["author"], considered))


def _running_focus(root: Any, tasks: Set[str]) -> Dict[str, str]:
    """The focus each running task authored for itself, with its time (one read of their display documents)."""
    import json

    from ouroboros.focus import compact_focus
    from ouroboros.memory_view import _minute
    from ouroboros.task_store import display_documents

    found = {}
    for task, text in display_documents(root, tasks).items() if tasks else ():
        line = compact_focus(json.loads(text).get("focus"))
        if line and line.get("author_task_id") == task:
            found[task] = f"{' '.join(str(line['text']).split())} ({_minute(line['authored_at'])})"
    return found


from ouroboros.memory_view_conversation_forms import (  # noqa: F401 -- compatibility exports for existing readers
    source_index as source_index,
    _gist_lines as _gist_lines,
    _unread_line as _unread_line,
    _fold_line as _fold_line,
    group_form as group_form,
    floor_steps as floor_steps,
    identity_line as identity_line,
    render_group as render_group,
    pieces as pieces,
    owed as owed,
    folded_rooms as folded_rooms,
    owner_items as owner_items,
)
