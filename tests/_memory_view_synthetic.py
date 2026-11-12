"""Synthetic memory-view snapshots of real shape and chosen size, for the floor and mode tests.

A snapshot is plain data (``memory_view.MemoryViewSnapshot``), so the floor can be pinned
at the sizes measured on the owner's copy without a chronicle: 377 retold-memory pointers
in 78 rooms, twelve marks, a dozen other conversations, the current room's retold page, its open
conversation and one line per task. Shapes follow ``capture_memory_view`` exactly; a
capture-shape test in ``test_memory_view_floor`` keeps the two from drifting.
"""
from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Dict, List, Optional

from ouroboros import memory_view as mv

SHA = "0123456789ab"
_PROSE = {  # plain words of each kind of text; ASCII without quotes, so a text's JSON characters are its length
    "page": "The floor walk prices each form from the snapshot's pieces. My human's words stay whole while the window "
            "holds them, and my other conversations fold first. ",
    "mine": "I read the floor walk and the tail facts; the replies of a finished work can stand for its summary. ",
    "theirs": "Could you look at the release notes and say whether the memory fix landed in this build? ",
    "owner": "Check why the Nova decisions keep slipping out of view and keep my requirements whole. ",
}


def prose(chars: int, kind: str = "page") -> str:
    """``chars`` characters of plain sentences of one kind of text, never one repeated letter."""
    return (_PROSE[kind] * (chars // len(_PROSE[kind]) + 1))[:chars]


def _ts(day: int, minute: int = 0) -> str:
    moment = datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=day - 1,
                                                                                             minutes=minute)
    return moment.isoformat()


def _address(room: str, ts: str, n: int) -> str:
    return f"row:{room}@{ts}#{(SHA + format(n, 'x'))[-12:]}"


def _label(room: int) -> str:
    return f"Project {'long name of a past project ' * 3}[chat_id={room}]"


def pointers(count: int = 377, rooms: int = 78) -> List[Dict[str, Any]]:
    """Retold-memory pointers in story order (block, then room); block ``b`` spans days 5b+1..5b+4."""
    out = []
    for i in range(count):
        block, room = divmod(i, rooms)
        start, end = _ts(5 * block + 1), _ts(5 * block + 4, 23 * 60)
        out.append({"kind": "legacy", "id": f"legacy-b{block:02d}-r{1000 + room}", "room_id": str(1000 + room),
                    "block": block, "label": _label(1000 + room), "period": f"{mv._minute(start)} → {mv._minute(end)}",
                    "span": [mv._minute(start), mv._minute(end)], "rows": 40, "gap": ""})
    return out


def pages(count: int, chars: int, room: str = "1") -> List[Dict[str, Any]]:
    return [{"kind": "page", "id": f"page-{i:03d}", "room_id": room, "label": "Main",
             "period": f"{mv._minute(_ts(40 + i))} → {mv._minute(_ts(40 + i, 600))}", "text": f"Page {i}. " + prose(chars),
             "status": "", "stamp": "", "fixes": []} for i in range(count)]


def marks(count: int = 12) -> List[Dict[str, Any]]:
    return [{"id": f"mark-{i:02d}", "scope": "global" if i % 3 else "room", "room": "Main",
             "text": f"Keep in view {i}: " + "a nomination of a durable lesson " * 4, "by": "the old dialogue writer",
             "date": "2026-09-01", "target": f"node legacy-b00-r{1000 + i}", "quote": ""} for i in range(count)]


def conversation(i: int, *, notes: int = 0, words: int = 0, word_chars: int = 300, owner: bool = False,
                 engaged: bool = True) -> Dict[str, Any]:
    """One other conversation of one room in the shape ``memory_view_conversations.capture`` returns: ``words``
    people's (or, with ``owner``, my human's) words whole; a silent room is one line."""
    room = str(2000 + i)
    first, last = _ts(60 + i), _ts(60 + i, 300)
    spoken = []
    for n in range(words if engaged else 0):
        ts, address = _ts(60 + i, 10 + n), _address(room, _ts(60 + i, 10 + n), i * 100 + n)
        head = f"[{ts}; {'Owner' if owner else 'Sasha'}; {address}]"
        spoken.append({"line": f"{head} " + prose(word_chars, "owner" if owner else "theirs"), "head": head,
                       "kind": "human", "owner_side": owner, "address": address, "ts": ts, "chars": word_chars,
                       "pos": 20_000 + i * 100 + n, "id": f"{room}:{address}", "rank": 2})
    label = f"Project room {i} [chat_id={room}]"
    item = {"room_id": room, "label": label, "engaged": engaged, "first": 20_000 + i * 100,
            "facts": facts(rows=3, people=words or 1, mine=1, first=first, last=last, notes=notes), "words": spoken,
            "notes": [{"id": f"note-{i}-{n}", "date": "2026-09-01", "ts": first, "role": "root",
                       "text": "What this project is."} for n in range(notes)],
            "tasks": [], "silent": "" if engaged else f"last {mv._minute(last)}; 1 words nobody considered; "
                                                      f"memory_read(room_id='{room}', rows=true)"}
    return {"key": f"project:{room}", "label": label, "latest": last, "running": [], "place": False, "rooms": [item]}


def facts(*, rows: int = 3, people: int = 1, mine: int = 1, chars: int = 2_400, first: str = "", last: str = "",
          notes: int = 0, pages: int = 0, last_covered: str = "") -> Dict[str, Any]:
    """A room's standing facts in the shape ``memory_inventory.room_facts`` returns (times ISO, as captured)."""
    return {"rows": rows, "people": people, "mine": mine, "task_facts": rows - people - mine, "chars": chars,
            "earliest": first or _ts(60), "latest": last or _ts(60, 300), "notes": notes, "pages": pages,
            "last_covered": last_covered}


def spoken(i: int, kind: str, chars: int, room: str = "1") -> Dict[str, Any]:
    ts = _ts(70, i)
    label = "Ouroboros" if kind == "ouroboros" else "Owner"
    head = f"[{ts}; {label}; {_address(room, ts, 50_000 + i)}]"
    return {"line": f"{head} " + prose(chars, "mine" if kind == "ouroboros" else "owner"), "head": head, "kind": kind,
            "address": _address(room, ts, 50_000 + i), "ts": ts, "chars": chars, "pos": 30_000 + i}


def task_line(i: int, room: str = "1", chars: int = 400) -> Dict[str, Any]:
    ts, last_ts = _ts(69, 2 * i), _ts(69, 2 * i + 1)
    first, last = _address(room, ts, 60_000 + 2 * i), _address(room, last_ts, 60_001 + 2 * i)
    line = f"[{ts}; host; task t{i:03d}] completed: " + prose(chars, "mine")
    return {"line": line, "line_without_gist": line, "task": f"t{i:03d}", "ts": ts,
            "pos": 25_000 + 2 * i, "first": first, "last_ts": last_ts, "last": last, "last_pos": 25_001 + 2 * i}


def room(room_id: str = "1", *, label: str = "Main", legacy: int = 0, legacy_chars: int = 0, under: int = 0,
         origins: int = 0, lane1: Optional[List[Dict[str, Any]]] = None, lane2: int = 0,
         notes: int = 0) -> Dict[str, Any]:
    return {"room_id": room_id, "label": label, "head": 7,
            "facts": facts(rows=len(lane1 or []) + 2 * lane2, people=sum(item["kind"] == "human" for item in lane1 or []),
                           mine=sum(item["kind"] == "ouroboros" for item in lane1 or []), notes=notes),
            "legacy": [{"id": f"legacy-b{b:02d}-r{room_id}", "period": f"{mv._minute(_ts(b + 1))} → "
                        f"{mv._minute(_ts(b + 1, 600))}", "text": f"Retold {b}. " + prose(legacy_chars)}
                       for b in range(legacy)],
            "under_parts": [{"id": f"page-u{n}", "kind": "page", "part": "part-0", "period": f"{mv._minute(_ts(30 + n))} "
                             f"→ {mv._minute(_ts(30 + n, 60))}", "text": "Under a part. " + prose(800)} for n in range(under)],
            "notes": [{"id": f"note-here-{n}", "date": "2026-09-02", "role": "root", "text": "Where this room stands."}
                      for n in range(notes)],
            "origins": [{"ts": _ts(1), "text": "Start this project. " + prose(1_000, "owner"), "ref": "chat 1 / origin-a"}
                        for _n in range(origins)],
            "since": mv._minute(_ts(70)), "lane1": list(lane1 or []), "lane2": [task_line(i, room_id) for i in range(lane2)]}


def snapshot(*, role: str = "integrator", room_facts: Optional[Dict[str, Any]] = None, story: Any = (),
             conversations: Any = (), mark_count: int = 0, owner_words: str = "",
             other_conversations: Optional[bool] = None) -> mv.MemoryViewSnapshot:
    """A snapshot of ``role``'s default spec (``other_conversations`` overrides it) over synthetic facts."""
    default = mv.ROLE_DEFAULTS[role].other_conversations
    spec = dataclasses.replace(mv.ROLE_DEFAULTS[role], room_id=room_facts["room_id"] if room_facts else None,
                               other_conversations=default if other_conversations is None else other_conversations)
    status = {"folded": 0, "total": 23, "pages_by_me": 0, "latest_by_me": "", "helper_pages": 0, "parts": 0,
              "open_records": 377, "open_rows": 9_422, "open_chars": 2_001_746, "helper_route": "light"} if spec.story else {}
    return mv.MemoryViewSnapshot(spec=spec, store_status={"state": "active"}, frontier={"status": "exact", "pos": 12_400},
                                 story=tuple(story) if spec.story else (), room=room_facts,
                                 conversations=tuple(conversations) if spec.other_conversations else (),
                                 marks=tuple(marks(mark_count)), legacy_blocks=status, owner_words=owner_words)


def live_scale(groups: int = 520, words: int = 33, word_chars: int = 40) -> mv.MemoryViewSnapshot:
    """Main's memory at a start of 07.11.2026 (68d1a491: 521 other conversations holding ~17,000 words of people,
    a story of 80 long pages), without any of my human's words; ``word_chars`` sizes a word (live ones held ~1,000)."""
    lane = [spoken(i, "ouroboros", 2_000) for i in range(6)]
    return snapshot(room_facts=room("1", lane1=lane, lane2=3), story=pages(80, 6_000), mark_count=12,
                    conversations=[conversation(i, words=words, word_chars=word_chars) for i in range(groups)])


# --- what each kind of memory is, by its words, in the shapes the capture gives it ---------------------------------

def said(i: int, kind: str, text: str, room: str = "1", **facts: Any) -> Dict[str, Any]:
    """A lane-1 row by its words (``spoken``'s shape); ``facts`` are what the capture holds beside them."""
    row = spoken(i, kind, len(text), room)
    return {**row, "line": f"{row['head']} {text}", **facts}


def work_line(i: int, task: str, status: str) -> Dict[str, Any]:
    """A task's lane-2 line: the host's status, a finished root's closing sentence in it as the capture writes it."""
    return {**task_line(i), "task": task, "line": f"[{_ts(69, 2 * i)}; host; task {task}] {status}; rows 3"}


def work_room() -> Dict[str, Any]:
    """This room's works (``task_id``: the work a reply belongs to, a fact the view reads nowhere yet): a finished one
    with a closing sentence (an intermediate and a final reply), a finished one without (one long reply), a running
    one, one whose closing sentence another room holds; my human's words before them and after them, unanswered,
    answering my final reply (``reply_to``)."""
    final = said(2, "ouroboros", "Found it: under a tight window my replies became bare stubs before old pages "
                 "did. A finished work should stand by its summary instead.", task_id="t101")
    lane1 = [said(0, "human", "Why do the Nova decisions keep slipping? Check the memory floor.", owner_side=True),
             said(1, "ouroboros", "Reading the floor walk now: my replies yield before old pages.", task_id="t101"),
             final, said(3, "ouroboros", prose(20_000, "mine"), task_id="t102"),
             said(4, "ouroboros", "Started the Nova audit; the first counts come within the hour.", task_id="t103"),
             said(5, "ouroboros", "The Nova root finished; its summary is in that project's room.", task_id="t104"),
             said(6, "human", "Yes, keep the summaries.", owner_side=True, immune=True, reply_to=final["address"])]
    return {**room("1", lane1=lane1), "lane2": [
        work_line(0, "t101", "completed — summary: «Replies became stubs before old pages; summaries keep them.»"),
        work_line(1, "t102", "completed"), work_line(2, "t103", "running or unreported; last row a reply by Ouroboros")]}


def told_story() -> List[Dict[str, Any]]:
    """My story by kind: an old retelling (``legacy``), a part and pages with a lead their writer was asked for
    (``lead``, a mark no code reads yet); pages without one: a topical first sentence before an untold decision, one
    the old code wrote between two of the new, one with no first sentence, a late one of a focus whose request held
    the old instruction."""
    texts = [("Memory is one subject with many foci. Every focus reads the same story first.", True),
             ("Nova memory audit, November. Decision: my human's words stay whole and my replies fold into the "
              "summaries of their works.", False),
             ("Rooms and projects, the October work. Each project kept its root and its room.", False),
             ("The tail facts count only new rows. The stored tail keeps its own count.", True),
             ("- kept whole: my human's words\n- folded: replies of finished works\n- open: the Nova summaries", False),
             ("A late page of a focus whose request held the old instruction. Its first sentence was never asked to "
              "lead.", False)]
    story = [{**page, "text": text, "floor_order": [n, n], **({"lead": True} if lead else {})}
             for n, (page, (text, lead)) in enumerate(zip(pages(len(texts), 0), texts))]
    part = {**story[0], "id": "part-000", "kind": "part", "floor_order": [-1, 0], "lead": True,
            "text": "October in one part: rooms became conversations. Each project kept its root and its room."}
    old = {**pointers(1, 1)[0], "text": "Retold before the update: the October work on rooms and projects, " + prose(300)}
    return [old, part, *story]


def place(i: int, words: List[tuple], summary: str = "") -> Dict[str, Any]:
    """A place thread in ``conversation``'s shape: ``words`` as ``(kind, author, text, rank)`` (rank 2: nobody
    considered them), and a finished task's line with its closing sentence."""
    group = conversation(i)
    item, spoken_words = group["rooms"][0], []
    for n, (kind, author, text, rank) in enumerate(words):
        ts = _ts(60 + i, 10 + n)
        address = _address(item["room_id"], ts, i * 100 + n)
        spoken_words.append({"line": f"[{ts}; {author}; {address}] {text}", "head": f"[{ts}; {author}; {address}]",
                             "kind": kind, "owner_side": False, "address": address, "ts": ts, "chars": len(text),
                             "pos": 20_000 + i * 100 + n, "id": f"{item['room_id']}:{address}", "rank": rank})
    item.update(words=spoken_words, tasks=[{"line": f"[{_ts(60 + i)}; host; task t{i}] completed; «{summary}»; rows 3",
                                            "pos": item["first"], "last_pos": item["first"] + 9, "chars": 900}]
                if summary else [])
    return {**group, "key": f"github:ouroboros-agent:thread-{i}", "place": True}


def place_threads() -> List[Dict[str, Any]]:
    """Place conversations: one with two finished threads, each with its own closing sentence; a thread of my own
    account's arrival alone, one with a bot's words, one with a stranger's reply, an arrival without a recorded
    author, and a thread of arrivals nobody considered, which the capture leaves silent."""
    mine, own = ("ouroboros", "Ouroboros"), ("ouroboros", "the account I send as: ouroboros-agent")
    issue = (*own, "Opened an issue: replies yield before old pages.", 1)
    both = place(1, [(*mine, "Fixed: the stored tail counts only new rows.", 1)], "The tail counts only new rows.")
    other = place(2, [(*mine, "Answered the review: the fact keeps index runs.", 1)], "The fact keeps index runs.")
    return [{**both, "rooms": [*both["rooms"], *other["rooms"]]},
            place(3, [(*own, "Published the release notes for 7.10.2.", 1)]),
            place(4, [issue, ("human", "github-actions[bot]", "Labels added: memory, context.", 2)]),
            place(5, [issue, ("human", "Sasha Petrova", "I see the same stubs on my install. Is there a fix?", 2)]),
            place(6, [("human", "author not recorded", "The description was edited.", 1)]),
            {**conversation(7, engaged=False), "key": "github:ouroboros-agent:thread-7", "place": True}]


OWNER_WORDS = "## Owner words that caused this work\n\nHost fact, carried by value from task root.\n" + prose(800, "owner")


def actor(name: str) -> mv.MemoryViewSnapshot:
    """The memory of one actor at the sizes measured on the owner's copy (estimator tokens, not o200k)."""
    main_lane = [spoken(0, "human", 400), spoken(1, "ouroboros", 5_600), spoken(2, "human", 1_600),
                 spoken(3, "ouroboros", 1_200)]
    project_lane = [spoken(i, "human" if i % 2 else "ouroboros", 1_100) for i in range(5)]
    live12 = [conversation(i) for i in range(12)]
    main_room = room("1", legacy=23, legacy_chars=8_400, lane1=main_lane, lane2=19)
    if name == "main":
        return snapshot(room_facts=main_room, story=pointers(), conversations=live12, mark_count=12)
    if name == "project":
        return snapshot(room_facts=room("257912875", label="Project X [chat_id=257912875]", legacy=8,
                                        legacy_chars=4_900, origins=1, lane1=project_lane, lane2=2),
                        story=pointers(), conversations=live12, mark_count=12)
    if name == "consciousness":  # Main's own view, and the words of people in my other conversations
        return snapshot(role="consciousness", room_facts=main_room, story=pointers(),
                        conversations=[conversation(i, words=2, word_chars=1_100) for i in range(12)], mark_count=12)
    if name in ("child_project", "nanny"):
        return snapshot(role="child" if name == "child_project" else "nanny", story=pointers(),
                        room_facts=room("257912875", label="Project X [chat_id=257912875]", legacy=8,
                                        legacy_chars=4_900, origins=1), mark_count=2, owner_words=OWNER_WORDS)
    assert name == "child_main", name
    return snapshot(role="child", story=pointers(), room_facts=room("1", legacy=23, legacy_chars=8_400), mark_count=2,
                    owner_words=OWNER_WORDS)


# The request without its memory, by mode, in estimator tokens (measured on the owner's copy: tools, books by mode,
# identity, knowledge, base sections, the task); a helper's books are always the navigation.
FIXED = {
    "main": {"max": 337_554, "low": 114_375, "nano": 72_895},
    "project": {"max": 341_319, "low": 118_140, "nano": 76_660},
    "consciousness": {"max": 340_148, "low": 116_969, "nano": 75_489},
    "child_project": {"max": 50_428, "low": 50_428, "nano": 31_921},
    "child_main": {"max": 50_428, "low": 50_428, "nano": 31_921},
    "nanny": {"max": 50_428, "low": 50_428, "nano": 31_921},
}


def conversations_text(groups, gone) -> str:
    """``## Other conversations`` as its pieces join, for a test's reading."""
    from ouroboros import memory_view_conversations as conversations

    groups = list(groups)
    return "\n\n".join(conversations.pieces(groups, gone, lambda _key, make: make())) if groups else ""
