"""The production-shaped stand: the owner's install of 01.11 as a fixture, counted.

PLAN_BASE §(e). ``FULL`` is that install's live set: 50 running and 20 owner-paused roots (86 members), 61 pending, 122 held
custody runs (113 settled), 35 schedules, 2,000 reconcile markers, 500 live rooms, a story of about 125,000 tokens, 5 concurrent
owner turns and a burst of 10 Presence events. ``REDUCED`` keeps every axis above the laws' jitter and runs in CI.

Two halves share these shapes. In the booted driver process (tests/_cost_driver.py, ``plan: stand``; tests/test_stand_counts.py)
``measure`` lays the live set (tests/_live_set.py), then the rooms and the story, and takes the owner's turn up to its first model
request alone and beside others: E2, E3, E4, E6 and E8 are equalities of what the meter counts, on WAL and on PERSIST. A real server
with real workers, the scripted model at a fixed latency and the fake engine (tests/system_e2e/test_stand_server.py) holds the laws
only processes can: E1, E5, E7, E9, E10. Wall-clock figures in either half are reported as facts, never as a verdict.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import threading
from typing import Any, Dict, List, NamedTuple

from tests._cost_driver import (
    CLOCKS,
    _boot_census,
    _boundary,
    _event_ctx,
    _FirstRequest,
    _lane_rows_ended,
    _mark,
    _owner_turn,
    _summary,
    no_carrier_can_start,
    refused,
)
from tests._cost_meter import tag_of, tagged
from tests._live_set import FULL as FULL_SET
from tests._live_set import Shape
from tests._live_set import measure as measure_live_set

CHARS_PER_TOKEN = 4  # the plan's "~125k-token story": what a page's text weighs, read by no law as a limit
FIRST_ROOM = 6000
# The others each start in a conversation of their own: two owner messages in one chat are one owner turn while
# it reads (``server_owner_routing._route_main_followup``), so a turn beside others is one beside other chats' turns.
OTHER_CHAT = 7000
HOLD_SEC = 300.0  # a transport bound: a turn the measurement forgot fails the driver, never hangs the lane


class Stand(NamedTuple):
    live: Shape
    rooms: int  # live rooms with a story
    story_tokens: int  # across those rooms' pages
    turns: int  # concurrent owner turns
    burst: int  # Presence events (the real server's half)
    real_tasks: int  # roots at work with a real worker each (the real server's half; a worker costs a process)


REDUCED = Stand(live=Shape(running=6, paused=3, members=14, pending=8, held=12, settled=10, schedules=6, markers=12),
                rooms=20, story_tokens=5_000, turns=3, burst=3, real_tasks=2)
FULL = Stand(live=FULL_SET, rooms=500, story_tokens=125_000, turns=5, burst=10, real_tasks=FULL_SET.running)
SHAPES = {"reduced": REDUCED, "full": FULL}


def room_needle(chat_id: int) -> str:
    return f"the story of room {chat_id} so far"


def lay_rooms(root: pathlib.Path, rooms: int, story_tokens: int) -> Dict[str, Any]:
    """Live rooms with open rows, each with a page of story, the story's weight divided among them; every room's page
    opens with its own needle, so a request that lost a room says which."""
    from ouroboros.chronicle_store import ChronicleStore
    from tests import _turn_shapes as shapes

    per_page = max(80, story_tokens * CHARS_PER_TOKEN // max(1, rooms))
    open_rows: List[Dict[str, Any]] = []
    paged: Dict[int, List[Dict[str, Any]]] = {}
    for index in range(rooms):
        chat_id = FIRST_ROOM + index
        open_rows += shapes._rows(chat_id, shapes.OPEN_ROWS, "open")
        paged[chat_id] = shapes._rows(chat_id, 2, "paged")
    shapes.append_rows(root, open_rows)
    ids = shapes.append_rows(root, [row for rows in paged.values() for row in rows])
    filler = "What this room settled, and why it matters to the rest of the work. "
    records = []
    for index, chat_id in enumerate(paged):
        text = room_needle(chat_id) + ": " + (filler * (per_page // len(filler) + 1))[:per_page]
        records.append(shapes._page(chat_id, ids[index * 2:index * 2 + 2], text))
    published = ChronicleStore(root).publish(records)
    if not published.ok:
        raise RuntimeError(f"the rooms' story did not land: {published}")
    return {"rooms": rooms, "pages": len(records), "story_chars": sum(len(record["text"]) for record in records)}


def _messages_text(messages: Any) -> List[str]:
    texts: List[str] = []
    for message in messages or ():
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(str(block.get("text") or "") for block in content if isinstance(block, dict))
    return texts


class _Requests:
    """What each measured turn's first model request carried; a turn tagged ``T:other:*`` waits at its request, live,
    until ``release``: the others the owner's turn runs beside."""

    def __init__(self, result: Dict[str, Any], rooms: int) -> None:
        self.result, self.rooms = result, rooms
        self.release = threading.Event()
        self.waiting: Dict[str, threading.Event] = {}
        self.seen: Dict[str, Dict[str, Any]] = {}

    def wrap(self, refuse: Any) -> Any:
        def chat(llm_self, *args, **kwargs):
            tag = tag_of()
            if tag.startswith("T:other:"):
                self.waiting.setdefault(tag, threading.Event()).set()
                if not self.release.wait(HOLD_SEC):
                    self.result["errors"].append(f"{tag} was never released")
                raise _FirstRequest(tag)
            if tag.startswith("T:stand:") and tag not in self.seen:
                self.seen[tag] = self._facts(kwargs.get("messages") or (args[0] if args else ()))
            return refuse(llm_self, *args, **kwargs)
        return chat

    def _facts(self, messages: Any) -> Dict[str, Any]:
        texts = _messages_text(messages)
        whole = "\n".join(texts)
        rooms = [FIRST_ROOM + index for index in range(self.rooms)]
        story = [text for text in texts if room_needle(rooms[0]) in text] if rooms else []
        start = story[0].find(room_needle(rooms[0])) if story else -1
        return {"rooms_carried": sum(room_needle(chat_id) in whole for chat_id in rooms), "chars": len(whole),
                "story_block_sha": hashlib.sha256(story[0][start:].encode("utf-8")).hexdigest() if story else ""}


def _newest_view_receipt(root: pathlib.Path) -> Dict[str, Any]:
    """The newest ``context_memory_view`` receipt in the event log (one per build, this process's newest turn)."""
    found: Dict[str, Any] = {}
    for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        if '"context_memory_view"' in line:
            try:
                found = json.loads(line)
            except ValueError:
                continue
    return found


def _turns(meter: Any, srv: Any, result: Dict[str, Any], root: pathlib.Path, facts: Dict[str, Any], stand: Stand) -> Dict[str, Any]:
    """The history and steady state B10's turns meet, the rooms and the story; then the owner's turn up to its first model
    request: alone, again (a second focus over the same chronicle), and beside ``turns - 1`` others held at their requests."""
    from ouroboros import llm, skill_identity, utils
    from supervisor.active_activity import get_direct_activity_registry
    from tests import _turn_shapes as shapes

    # As in B10 (``CLOCKS``): a file read within a clock tick of its write is read again.
    utils._RACY_NS = skill_identity._RACY_NS = int(CLOCKS["racy_ns"])
    no_carrier_can_start()
    ctx, k = _event_ctx(srv), int(facts.get("k") or 1)
    out: Dict[str, Any] = {"history": shapes.turn_history(root, k), "steady": shapes.steady_state(root, refused=refused, window=shapes.WIDE_WINDOW)}
    _boot_census(srv, root)
    out["rooms"] = lay_rooms(root, stand.rooms, stand.story_tokens)
    requests = _Requests(result, stand.rooms)
    llm.LLMClient.chat = requests.wrap(llm.LLMClient.chat)

    def turn(tag: str, message_id: str) -> None:
        _boundary(meter, result, tag, lambda: _owner_turn(srv, ctx, 1, message_id), timeout=HOLD_SEC)
        _mark(result, tag)

    turn("T:stand:first", "stand-first")
    _lane_rows_ended()
    turn("T:stand:alone", "stand-alone")
    _lane_rows_ended()
    others = [f"T:other:{index}" for index in range(stand.turns - 1)]
    for index, tag in enumerate(others):
        requests.waiting[tag] = threading.Event()
        tagged(tag, lambda index=index: _owner_turn(srv, ctx, OTHER_CHAT + index, f"stand-other-{index}"))
    arrived = [tag for tag in others if requests.waiting[tag].wait(HOLD_SEC)]
    out["others_held"] = len(arrived)
    out["registry_live"] = len(get_direct_activity_registry().actors())
    turn("T:stand:beside", "stand-beside")
    requests.release.set()
    for tag in others:
        meter.join_spawned(tag, timeout=HOLD_SEC)
    out["requests"] = requests.seen
    out["memory_view"] = _newest_view_receipt(root)
    for tag in ("T:stand:first", "T:stand:alone", "T:stand:beside"):
        out[tag] = result.get(f"{tag}@request") or _summary(meter, tag)
    return out


def measure(meter: Any, srv: Any, client: Any, result: Dict[str, Any], root: pathlib.Path, spec: Dict[str, Any],
            stubs: Dict[str, Any]) -> None:
    stand = SHAPES[spec["stand"]]
    measure_live_set(meter, srv, client, result, root, {**spec, "shape": stand.live._asdict()}, stubs)
    result["stand"] = _turns(meter, srv, result, root, spec["facts"], stand)


# ---- the real server's root ------------------------------------------------------------------------------------------------

def build_root(root: pathlib.Path, shape: str, wake_due: bool = True) -> Dict[str, Any]:
    """The published scale root (tests/_scale_root.py) with the shape's standing set: what a process that restarts over the
    owner's install finds. The queue a boot restores is the work: its first rows become the roots at work, each on a real worker,
    and the rest stand behind them. ``wake_due``: consciousness on and its alarm past due, so the wake falls due at its floor."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.dialogue_import import run_dialogue_import
    from supervisor import queue, state
    from tests._live_set import grow_held, grow_markers, grow_paused, grow_pending, grow_schedules
    from tests._scale_root import build

    stand = SHAPES[shape]
    live = stand.live
    facts = build(root, 1)
    imported = run_dialogue_import(root).get("state")
    ChronicleStore(root).ensure_activated(wait=True)
    laid = {"paused": grow_paused(root, 0, live.paused, live.members), "pending": grow_pending(root, 0, live.pending + stand.real_tasks),
            "held": grow_held(root, 0, live.held, live.settled, facts), "schedules": grow_schedules(root, 0, live.schedules),
            "markers": grow_markers(root, live.markers), "rooms": lay_rooms(root, stand.rooms, stand.story_tokens)}
    if not queue.persist_queue_snapshot(reason="pre_restart_exit"):
        raise RuntimeError("the stand's queue snapshot was not published")
    if wake_due:
        state.update_state(lambda st: st.update(bg_consciousness_enabled=True, consciousness_next_wake_at=1.0))
    return {**facts, "laid": laid, "import": imported}


def main(argv: List[str]) -> int:
    from tests._scale_root import force_store_journal

    root, shape, out = pathlib.Path(argv[0]), argv[1], pathlib.Path(argv[2])
    force_store_journal()
    out.write_text(json.dumps(build_root(root, shape), indent=1, sort_keys=True, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
