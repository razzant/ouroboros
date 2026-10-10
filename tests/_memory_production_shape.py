"""A memory installation in the shape a live one took, in miniature.

It is ``tests._memory_inventory_shared.world`` (Main, Projects alpha and beta, a transport chat,
twenty stream rows, two blocks of old memory over rows 0-9) plus the forms a fixture with one room
per block and no correction never showed. Each was found on a live installation:

- **Old memory folded one part per retold record**, drafted by a nanny and accepted by the mind, the
  rooms of a block newest first. Both blocks hold several rooms; Alpha takes the tail of block zero
  and Beta the tail of block one, so the period a part records (its block's) starts before its
  room's first row, and the parts of one block record one shared position, in an order that runs
  against the stream.
- **A part corrected by an addition**: the correction adds a fact and repeats none of the part's
  words. Beta's part carries two, the second by a wake.
- **A nested fold**: my page, a part over it, a correction of the page, a second part over the first.
- **A page with a gap**: sealed by task, it covers two rows of Main and leaves the row between open.
- **A note not yet sealed** in a Project room.
- **An answer to a question card in a Project room**, written with the host's own frame
  (``owner_quiz.quiz_answer_frame``): the question, longer than 500 characters, comes before the choice.

``production(root)`` builds it and returns the ids and rows a test names; the texts are constants, so
a test can look for an original and its correction apart. ``page``, ``part`` and ``correct`` are the
writers it uses, for a test that adds a record of its own.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ouroboros import chat_chain
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared

MIND = shared.MIND
WAKE = {"kind": "mind", "task_id": "wake0001", "focus": {"role": "consciousness", "task_id": "wake0001"}}
NANNY = {"kind": "helper", "task_id": "nan00001", "route": {}, "focus": {"role": "nanny", "task_id": "nan00001"}}

ADDITION = "Also settled then: alpha ships only after its inventory is counted twice."
FIRST_FIX = "A source note: the retelling names no owner decision about beta."
SECOND_FIX = "Narrowing my note: the owner did ask about beta once, on 2 September."
PAGE = "The room started and the owner asked me to go on."
INNER = "The first hours of Main after the update."
PAGE_FIX = "One more fact for that page: the room had been started by the host, not by the owner."
OUTER = "Main after the update, told once more."
GAP_PAGE = "Task t2: I answered at length and the host recorded it completed."
NOTE = "Alpha still waits for the owner's word on the inventory."
QUESTION = ("Which inventory should alpha count first? " + "The warehouse list is older and longer, the shop list "
            "is newer and already half counted, and whichever goes first decides what the report can say. " * 4)
OPTIONS = ["Count the warehouse first", "Count the shop first"]
COMMENT = "The shop, and tell me before you touch the warehouse."
ASKED, ANSWERED = "2026-09-04T00:00:00+00:00", "2026-09-04T00:05:00+00:00"


def part_text(unit_id: str) -> str:
    return f"A nanny's account of the old retelling {unit_id}."


def page(root: pathlib.Path, room: Any, *, first: Optional[int] = None, last: Optional[int] = None,
         task_ids: Optional[List[str]] = None, author: Dict[str, Any] = MIND, text: str = "") -> str:
    """A page over the room's rows between two stream positions, or of the named tasks, stamped by the host."""
    from ouroboros.tools.chronicle import host_stamp, page_covers

    if task_ids is None:
        addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(root)}
        read = page_covers(root, room, from_addr=addresses[first], to_addr=addresses[last])
    else:
        read = page_covers(root, room, task_ids=task_ids)
    result = ChronicleStore(root).publish_page(
        room_id=room, text=text or f"Page of room {room}.", covers=read["covers"], author=author,
        host_stamp=host_stamp(root, read["covers"]["task_ids"], rows=read["rows"]))
    assert result.ok, result
    return result.record["id"]


def part(root: pathlib.Path, room: Any, members: List[str], *, author: Dict[str, Any] = MIND,
         text: str = "A part of my story.") -> str:
    store = ChronicleStore(root)
    result = store.publish_part(room_id=room, text=text, member_ids=members, author=author,
                                expected_sequence=store.room_head(room))
    assert result.ok, result
    if author.get("kind") == "helper":
        assert store.decide(result.record["id"], True, MIND, "Read against its members; accepted.").ok
    return result.record["id"]


def correct(root: pathlib.Path, target: str, text: str, *, author: Dict[str, Any] = MIND,
            expected_revision: Optional[str] = None) -> str:
    store = ChronicleStore(root)
    result = store.correct(target, text, author, expected_revision=expected_revision,
                           expected_sequence=store.room_head(store.get(target)["room_id"]))
    assert result.ok, result
    return result.record["id"]


def quiz_rows(alpha: int) -> List[Dict[str, Any]]:
    """The question card and the owner's answer in the Project room, as the host writes them."""
    from ouroboros.owner_quiz import quiz_answer_frame

    card = {"quiz_id": "q-inventory", "question": QUESTION, "options": OPTIONS, "asked_at": ASKED, "chat_id": alpha}
    answered = {**card, "answered_index": 1, "comment": COMMENT, "answered_at": ANSWERED}
    return [shared.msg(ASKED, "", chat_id=alpha, direction="out", type="quiz", task_id="ta2", quiz=card),
            shared.msg(ANSWERED, quiz_answer_frame(answered, 1, COMMENT), chat_id=alpha, direction="system",
                       type="quiz_answer", task_id="ta2", source="owner_quiz_answer",
                       client_message_id="quiz_answer:ta2:q-inventory", quiz=answered)]


def production(root: pathlib.Path) -> SimpleNamespace:
    """Build the installation; the namespace names what a test asks about."""
    rooms = shared.world(root)
    alpha, beta = str(rooms["alpha"]), str(rooms["beta"])
    # One part per retold record, as one background fold wrote them: within a block the last room first.
    folds = [("777", "legacy-b00-r777"), (alpha, f"legacy-b00-r{alpha}"), ("1", "legacy-b00-r1"),
             (beta, f"legacy-b01-r{beta}"), (alpha, f"legacy-b01-r{alpha}"), ("1", "legacy-b01-r1")]
    parts = {unit: part(root, room, [unit], author=NANNY, text=part_text(unit)) for room, unit in folds}
    corrected = parts[f"legacy-b00-r{alpha}"]
    addition = correct(root, corrected, ADDITION)
    twice = parts[f"legacy-b01-r{beta}"]
    first_fix = correct(root, twice, FIRST_FIX)
    second_fix = correct(root, twice, SECOND_FIX, author=WAKE, expected_revision=first_fix)
    nested_page = page(root, "1", first=10, last=11, text=PAGE)
    inner = part(root, "1", [nested_page], text=INNER)
    page_fix = correct(root, nested_page, PAGE_FIX)
    outer = part(root, "1", [inner], text=OUTER)
    gap_page = page(root, "1", task_ids=["t2"], text=GAP_PAGE)
    note = ChronicleStore(root).write_note(room_id=alpha, task_id="ta", text=NOTE, author=MIND)
    assert note.ok, note
    card, answer = quiz_rows(rooms["alpha"])
    shared.append(root / "logs" / "chat.jsonl", card, answer)
    return SimpleNamespace(rooms=rooms, alpha=alpha, beta=beta, parts=parts, corrected=corrected, addition=addition,
                           twice=twice, first_fix=first_fix, second_fix=second_fix, page=nested_page, inner=inner,
                           page_fix=page_fix, outer=outer, gap_page=gap_page, note=note.record["id"], card=card,
                           answer=answer)
