"""One host line in Main per Low episode of fresh starts that memory lowered to Low (owner A, 07.11.2026, which
replaced the line per start of 1B, 29.10).

The start's own Max room decides (``memory_floor.crowded_by_memory``); the line, without numbers, goes through the
terminal outbox under one id per task attempt when no episode is open; a start that holds Max closes the episode
silently; a route, tool schemas that tip a start alone, a delegated child or nanny, the owner's Low or Nano and an
unknown window neither send nor touch the episode.
"""
import json
from types import SimpleNamespace

import pytest

from ouroboros import loop
from ouroboros.context_view_facts import low_start_line
from supervisor import schedule_notes, terminal_delivery as outbox
from tests.test_memory_max_room import _core, _crowded_window, _fact, _start


def _room(cause="crowded", *, window=1_000_000, zero_need=900_000, levels=1):
    return {"cause": cause, "window": window, "allowance": 40_000, "floor_need": zero_need - 70_000,
            "schemas": 70_000, "zero_need": zero_need, "need": zero_need + 150_000, "story": 110_000,
            "unfolded": 60_000, "room": 40_000, "reserve": 65_536, "work": 125_000 * levels, "levels": levels,
            "parts": [["books", 400_000], ["core", 30_000]]}


def _plan(room, *, role="integrator"):
    facts = {"role": role, **({"max_room": room} if room else {})}
    return SimpleNamespace(projection=lambda _mode: SimpleNamespace(memory_facts=facts))


@pytest.fixture
def sent(monkeypatch):
    rows = []
    monkeypatch.setattr(outbox, "enqueue_terminal_delivery_outcome",
                        lambda root, event, event_queue=None: rows.append((root, event)) or "queued")
    monkeypatch.setattr(schedule_notes, "owner_chat_id", lambda _root: 7)
    return rows


def test_a_start_memory_lowered_to_low_says_so_without_numbers_or_advice():
    line = low_start_line(_plan(_room()), "low", "rootA")
    assert line == "Task rootA started in Low with chapter maps: memory did not fit beside the full books."
    assert not any(char.isdigit() for char in line) and len(line.splitlines()) == 1  # numbers stay in the task's facts
    assert not any(word in line for word in ("may bring", "should", "must", "fold", "episode", "again"))


@pytest.mark.parametrize("plan", [
    _plan(_room("route")),                             # the books cannot fit this route: its task's facts say so
    _plan(None),                                       # the owner's Low or Nano, or an unknown window: no Max room
    None,
])
def test_every_other_start_gets_no_line(plan):
    assert low_start_line(plan, "low", "t") == ""


def test_a_measured_crowded_start_gets_its_line(monkeypatch):
    core = _core()
    held = _fact(_start(core, 2_000_000, monkeypatch))
    plan = _start(core, _crowded_window(held), monkeypatch)
    assert plan.initial_mode == "low" and _fact(plan)["cause"] == "crowded"
    assert low_start_line(plan, plan.initial_mode, "wake").startswith("Task wake started in Low with chapter maps:")
    assert low_start_line(_start(core, 2_000_000, monkeypatch), "max", "wake") == ""


def test_the_line_goes_through_the_outbox_once_per_low_episode(tmp_path, sent):
    """Low, Low: one line; Max closes the episode silently; Low again: a new line. A delegated child or nanny, the
    owner's Low or Nano, an unknown window, a route and tool schemas that tip a start alone send nothing and leave the
    episode as it is, open or closed."""
    ctx = SimpleNamespace(task_metadata={"budget_drive_root": str(tmp_path)}, drive_root=tmp_path / "drive",
                          task_attempt=None)
    bit = tmp_path.resolve() / "state" / "low_start_episode.json"

    def start(task, room, mode="low", **plan):
        loop._note_low_start(ctx, _plan(room, **plan), mode, task, None)
        return json.loads(bit.read_text(encoding="utf-8")) if bit.exists() else None

    assert start("rootA", _room())["task"] == "rootA"
    [(root, event)] = sent
    assert root == tmp_path.resolve() and event["chat_id"] == 7 and event["task_id"] == ""
    assert event["role"] == "system" and event["system_type"] == "low_start_notice" and event["type"] == "send_message"
    assert event["delivery_id"] == "low_start:rootA:1" and event["text"].startswith("Task rootA started in Low")
    opened = start("rootB", _room())
    assert opened["task"] == "rootA" and len(sent) == 1  # Low holds: no second line
    for task, room, mode, plan in (("kid", _room(), "low", {"role": "child"}), ("nan", _room("held"), "max", {"role": "nanny"}),
                                   ("ownerlow", None, "low", {}), ("route", _room("route"), "low", {}),
                                   ("schemas", _room(zero_need=1_100_000), "low", {})):
        assert start(task, room, mode, **plan) == opened and len(sent) == 1  # untouched, open
    assert start("rootC", _room("held"), "max") == {} and len(sent) == 1  # Max held: closed, no return line
    for task, room, mode, plan in (("kid", _room(), "low", {"role": "child"}), ("ownerlow", None, "low", {})):
        assert start(task, room, mode, **plan) == {} and len(sent) == 1  # untouched, closed
    assert start("rootD", _room())["task"] == "rootD"
    assert [event["delivery_id"] for _root, event in sent] == ["low_start:rootA:1", "low_start:rootD:1"]


def test_only_a_start_with_the_opening_work_room_that_holds_max_closes_the_episode(tmp_path, sent):
    """Opus RV1 F2: a root keeps two eighths of its window for work, a chat turn or wake one. Memory in the band
    between lets a turn hold Max where a root cannot; that proves no room for the next root, so the episode stays open
    and the next root's Low adds no line. A root that holds Max closes it, and a turn's own episode closes on a turn."""
    ctx = SimpleNamespace(task_metadata={"budget_drive_root": str(tmp_path)}, drive_root=tmp_path / "drive",
                          task_attempt=None)
    starts = (("rootA", "crowded", 2), ("turnA", "held", 1), ("rootB", "crowded", 2),  # a turn's Max: still open
              ("rootC", "held", 2), ("rootD", "crowded", 2),  # a root held Max: the next Low is a new episode
              ("rootE", "held", 2), ("turnB", "crowded", 1),  # a chat turn opens its own episode
              ("turnC", "held", 1), ("turnD", "crowded", 1))  # and a chat turn's Max closes it
    for task, cause, levels in starts:
        loop._note_low_start(ctx, _plan(_room(cause, levels=levels)), "max" if cause == "held" else "low", task, None)
    assert [event["delivery_id"] for _root, event in sent] == [
        "low_start:rootA:1", "low_start:rootD:1", "low_start:turnB:1", "low_start:turnD:1"]
    bit = json.loads((tmp_path.resolve() / "state" / "low_start_episode.json").read_text(encoding="utf-8"))
    assert (bit["task"], bit["levels"]) == ("turnD", 1)


def test_a_failing_outbox_never_fails_the_start_and_keeps_the_episode_closed(tmp_path, monkeypatch, sent):
    queued = {"outcome": "unavailable"}
    monkeypatch.setattr(outbox, "enqueue_terminal_delivery_outcome", lambda root, event, event_queue=None: (
        sent.append((root, event)) or queued["outcome"]))
    ctx = SimpleNamespace(task_metadata={}, drive_root=tmp_path)
    loop._note_low_start(ctx, _plan(_room()), "low", "t", None)  # the queue refused the line: nothing is open
    monkeypatch.setattr(outbox, "enqueue_terminal_delivery_outcome",
                        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("store busy")))
    loop._note_low_start(ctx, _plan(_room()), "low", "t", None)  # a raising outbox fails no start either
    monkeypatch.setattr(outbox, "enqueue_terminal_delivery_outcome", lambda root, event, event_queue=None: (
        sent.append((root, event)) or "queued"))
    loop._note_low_start(ctx, _plan(_room()), "low", "u", None)
    loop._note_low_start(ctx, _plan(_room()), "low", "v", None)
    assert [event["delivery_id"] for _root, event in sent] == ["low_start:t:1", "low_start:u:1"]
