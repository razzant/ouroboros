"""Synthetic memory-view snapshots of real shape and chosen size, for the floor and mode tests.

A snapshot is plain data (``memory_view.MemoryViewSnapshot``), so the floor can be pinned
at the sizes measured on the owner's copy without a chronicle: 377 retold-memory pointers
in 78 rooms, twelve marks, a dozen live rooms, the current room's retold page, its open
conversation and one line per task. Shapes follow ``capture_memory_view`` exactly; a
capture-shape test in ``test_memory_view_floor`` keeps the two from drifting.
"""
from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Dict, List, Optional

from ouroboros import memory_view as mv

SHA = "0123456789ab"


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
             "period": f"{mv._minute(_ts(40 + i))} → {mv._minute(_ts(40 + i, 600))}", "text": f"Page {i}. " + "w" * chars,
             "status": "", "revision": "", "stamp": "", "fixes": []} for i in range(count)]


def marks(count: int = 12) -> List[Dict[str, Any]]:
    return [{"id": f"mark-{i:02d}", "scope": "global" if i % 3 else "room", "room": "Main",
             "text": f"Keep in view {i}: " + "a nomination of a durable lesson " * 4, "by": "the old dialogue writer",
             "date": "2026-09-01", "target": f"node legacy-b00-r{1000 + i}", "quote": ""} for i in range(count)]


def live_room(i: int, *, notes: int = 0, words: int = 0, word_chars: int = 300) -> Dict[str, Any]:
    room = str(2000 + i)
    first, last = _ts(60 + i), _ts(60 + i, 300)
    spoken = []
    for n in range(words):
        ts = _ts(60 + i, 10 + n)
        head = f"[{ts}; Owner; {_address(room, ts, i * 100 + n)}]"
        spoken.append({"line": f"{head} " + "s" * word_chars, "head": head, "kind": "human",
                       "address": _address(room, ts, i * 100 + n), "ts": ts, "chars": word_chars,
                       "pos": 20_000 + i * 100 + n})
    return {"room_id": room, "label": f"Project room {i} [chat_id={room}]", "key": 20_000 + i * 100,
            "first": mv._minute(first), "last": mv._minute(last), "people": words or 1, "rows": 3, "mine": 1,
            "facts": 1, "words": spoken,
            "notes": [{"id": f"note-{i}-{n}", "date": "2026-09-01", "role": "root", "text": "What this project is."}
                      for n in range(notes)]}


def spoken(i: int, kind: str, chars: int, room: str = "1") -> Dict[str, Any]:
    ts = _ts(70, i)
    label = "Ouroboros" if kind == "ouroboros" else "Owner"
    head = f"[{ts}; {label}; {_address(room, ts, 50_000 + i)}]"
    return {"line": f"{head} " + ("m" if kind == "ouroboros" else "p") * chars, "head": head, "kind": kind,
            "address": _address(room, ts, 50_000 + i), "ts": ts, "chars": chars, "pos": 30_000 + i}


def task_line(i: int, room: str = "1", chars: int = 400) -> Dict[str, Any]:
    ts, last_ts = _ts(69, 2 * i), _ts(69, 2 * i + 1)
    first, last = _address(room, ts, 60_000 + 2 * i), _address(room, last_ts, 60_001 + 2 * i)
    return {"line": f"[{ts}; host; task t{i:03d}] completed: " + "f" * chars, "task": f"t{i:03d}", "ts": ts,
            "pos": 25_000 + 2 * i, "first": first, "last_ts": last_ts, "last": last, "last_pos": 25_001 + 2 * i}


def room(room_id: str = "1", *, label: str = "Main", legacy: int = 0, legacy_chars: int = 0, under: int = 0,
         origins: int = 0, lane1: Optional[List[Dict[str, Any]]] = None, lane2: int = 0,
         notes: int = 0) -> Dict[str, Any]:
    return {"room_id": room_id, "label": label, "head": 7,
            "legacy": [{"id": f"legacy-b{b:02d}-r{room_id}", "period": f"{mv._minute(_ts(b + 1))} → "
                        f"{mv._minute(_ts(b + 1, 600))}", "text": f"Retold {b}. " + "r" * legacy_chars}
                       for b in range(legacy)],
            "under_parts": [{"id": f"page-u{n}", "kind": "page", "part": "part-0", "period": f"{mv._minute(_ts(30 + n))} "
                             f"→ {mv._minute(_ts(30 + n, 60))}", "text": "Under a part. " + "u" * 800} for n in range(under)],
            "notes": [{"id": f"note-here-{n}", "date": "2026-09-02", "role": "root", "text": "Where this room stands."}
                      for n in range(notes)],
            "origins": [{"ts": _ts(1), "text": "Start this project. " + "o" * 1_000, "ref": "chat 1 / origin-a"}
                        for _n in range(origins)],
            "since": mv._minute(_ts(70)), "lane1": list(lane1 or []), "lane2": [task_line(i, room_id) for i in range(lane2)]}


def snapshot(*, role: str = "integrator", room_facts: Optional[Dict[str, Any]] = None, story: Any = (),
             live: Any = (), mark_count: int = 0, owner_words: str = "", live_rooms: str = "lines") -> mv.MemoryViewSnapshot:
    spec = dataclasses.replace(mv.ROLE_DEFAULTS[role], live_rooms=live_rooms,
                               room_id=room_facts["room_id"] if room_facts else None)
    status = {"folded": 0, "total": 23, "pages_by_me": 0, "open_records": 377, "open_rows": 9_422,
              "open_chars": 2_001_746, "helper_route": "light"} if spec.story else {}
    return mv.MemoryViewSnapshot(spec=spec, store_status={"state": "active"}, frontier={"status": "exact", "pos": 12_400},
                                 story=tuple(story) if spec.story else (), room=room_facts, live_rooms=tuple(live),
                                 marks=tuple(marks(mark_count)), legacy_blocks=status, owner_words=owner_words)


OWNER_WORDS = "## Owner words that caused this work\n\nHost fact, carried by value from task root.\n" + "w" * 800


def actor(name: str) -> mv.MemoryViewSnapshot:
    """The memory of one actor at the sizes measured on the owner's copy (estimator tokens, not o200k)."""
    main_lane = [spoken(0, "human", 400), spoken(1, "ouroboros", 5_600), spoken(2, "human", 1_600),
                 spoken(3, "ouroboros", 1_200)]
    project_lane = [spoken(i, "human" if i % 2 else "ouroboros", 1_100) for i in range(5)]
    live12 = [live_room(i) for i in range(12)]
    if name == "main":
        return snapshot(room_facts=room("1", legacy=23, legacy_chars=8_400, lane1=main_lane, lane2=19),
                        story=pointers(), live=live12, mark_count=12)
    if name == "project":
        return snapshot(room_facts=room("257912875", label="Project X [chat_id=257912875]", legacy=8,
                                        legacy_chars=4_900, origins=1, lane1=project_lane, lane2=2),
                        story=pointers(), live=live12, mark_count=12)
    if name == "consciousness":
        return snapshot(role="consciousness", story=pointers(), live=[live_room(i) for i in range(13)], mark_count=12)
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
