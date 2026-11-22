"""One host line in Main per open period of each cause of fresh starts that the window took from whole books in my
human's Max (owner A, 07.11.2026, which replaced the line per start of 1B, 29.10; every cause since 22.11.2026, a
period per cause by the coordinator's option C, 22.11.2026).

The start's own Max room decides (``memory_floor.lowered_cause``: memory, what the request itself carries, the route);
the line, without numbers, goes through the terminal outbox under one id per task attempt when its cause has no open
period; a start of the opener's kind (its room and work room) that holds Max closes its periods silently; a delegated
child or nanny, the owner's Low or Nano and an unknown window neither send nor touch the periods.
"""
import json
from types import SimpleNamespace

import pytest

from ouroboros import loop
from ouroboros.context_view_facts import low_start_line
from supervisor import schedule_notes, terminal_delivery as outbox
from tests.test_memory_max_room import _core, _crowded_window, _fact, _start


def _room(cause="crowded", *, window=1_000_000, zero_need=900_000, floor_need=None, levels=1, room_id=None):
    if cause == "route" and floor_need is None:  # the books, core and work room alone exceed this route's window
        floor_need, zero_need = window + 10_000, window + 80_000
    floor_need = zero_need - 70_000 if floor_need is None else floor_need
    return {"cause": cause, "window": window, "allowance": 40_000, "floor_need": floor_need,
            "schemas": zero_need - floor_need, "zero_need": zero_need, "need": zero_need + 150_000, "story": 110_000,
            "unfolded": 60_000, "room": 40_000, "reserve": 65_536, "work": 125_000 * levels, "levels": levels,
            "room_id": room_id, "parts": [["books", 400_000], ["core", 30_000]]}


def _plan(room, *, role="integrator", model=""):
    facts = {"role": role, **({"max_room": room} if room else {})}
    return SimpleNamespace(projection=lambda _mode: SimpleNamespace(memory_facts=facts), model=model)


@pytest.fixture
def sent(monkeypatch):
    rows = []
    monkeypatch.setattr(outbox, "enqueue_terminal_delivery_outcome",
                        lambda root, event, event_queue=None: rows.append((root, event)) or "queued")
    monkeypatch.setattr(schedule_notes, "owner_chat_id", lambda _root: 7)
    return rows


def test_a_start_memory_lowered_to_low_says_so_without_numbers_or_advice():
    line = low_start_line(_plan(_room()), "low", "rootA")
    assert line == ("Task rootA runs in Low for now, not in your Max: the whole books did not fit beside my memory even "
                    "in its shortest tellings. Your setting stays Max.")
    assert not any(char.isdigit() for char in line) and len(line.splitlines()) == 1  # numbers stay in the task's facts
    assert not any(word in line for word in ("may bring", "should", "must", "fold", "episode", "again"))
    # A start that protected words took past the maps frame to Nano is named Nano, never Max (``context_fit``)
    assert low_start_line(_plan(_room()), "nano", "rootA") == (
        "Task rootA runs in Nano for now, not in your Max: the whole books did not fit beside my memory even in its "
        "shortest tellings; even the chapter maps did not fit. Your setting stays Max.")


def test_what_the_request_carries_and_the_route_send_their_own_lines():
    """Consilium of 22.11.2026 (P6, P7): the opening line goes for every cause, each in its own words, no numbers."""
    request = _room(zero_need=1_100_000, floor_need=950_000)  # the books fit without the tool schemas, not with them
    assert low_start_line(_plan(request, model="opus"), "low", "rootA") == (  # a period per model: the line names it
        "Task rootA runs in Low for now, not in your Max: even with no memory, what the request itself carries tips "
        "this start over the window on opus. Your setting stays Max.")
    assert "over the window on this route. Your" in low_start_line(_plan(request), "low", "rootA")  # no model named
    assert low_start_line(_plan(_room("route"), model="opus"), "low", "rootA") == (
        "Task rootA runs in Low for now, not in your Max: on opus the window cannot hold the whole books beside the rest "
        "of this start and the work's room even with no memory; folding my memory cannot change that. Your setting stays Max.")
    assert "on this route the window" in low_start_line(_plan(_room("route")), "low", "rootA")  # no model named


@pytest.mark.parametrize("plan", [
    _plan(_room("held")),                              # Max held
    _plan(None),                                       # the owner's Low or Nano, or an unknown window: no Max room
    _plan(_room(), role="child"),                      # a delegated child
    None,
])
def test_every_other_start_gets_no_line(plan):
    assert low_start_line(plan, "low", "t") == ""


def test_a_measured_crowded_start_gets_its_line(monkeypatch):
    core = _core()
    held = _fact(_start(core, 2_000_000, monkeypatch))
    plan = _start(core, _crowded_window(held), monkeypatch)
    assert plan.initial_mode == "low" and _fact(plan)["cause"] == "crowded"
    assert low_start_line(plan, plan.initial_mode, "wake").startswith("Task wake runs in Low for now, not in your Max:")
    assert low_start_line(_start(core, 2_000_000, monkeypatch), "max", "wake") == ""


def test_the_line_goes_through_the_outbox_once_per_low_period(tmp_path, sent):
    """Low, Low: one line; Max closes the period silently; Low again: a new line. A delegated child or nanny, the
    owner's Low or Nano, an unknown window and a held room in a lower mode send nothing and leave the periods as they
    are, open or closed."""
    ctx = SimpleNamespace(task_metadata={"budget_drive_root": str(tmp_path)}, drive_root=tmp_path / "drive",
                          task_attempt=None)
    bit = tmp_path.resolve() / "state" / "low_start_episode.json"

    def start(task, room, mode="low", **plan):
        loop._note_low_start(ctx, _plan(room, **plan), mode, task, None)
        return json.loads(bit.read_text(encoding="utf-8")) if bit.exists() else None

    assert start("rootA", _room())["memory"]["task"] == "rootA"
    [(root, event)] = sent
    assert root == tmp_path.resolve() and event["chat_id"] == 7 and event["task_id"] == ""
    assert event["role"] == "system" and event["system_type"] == "low_start_notice" and event["type"] == "send_message"
    assert event["delivery_id"] == "low_start:rootA:1" and event["text"].startswith("Task rootA runs in Low for now")
    opened = start("rootB", _room())
    assert opened["memory"]["task"] == "rootA" and len(sent) == 1  # Low holds: no second line
    for task, room, mode, plan in (("kid", _room(), "low", {"role": "child"}), ("nan", _room("held"), "max", {"role": "nanny"}),
                                   ("ownerlow", None, "low", {}),
                                   ("voice", _room("held"), "low", {})):  # a held room whose envelope fit lower (Astra 8)
        assert start(task, room, mode, **plan) == opened and len(sent) == 1  # untouched, open
    assert start("rootC", _room("held"), "max") == {} and len(sent) == 1  # Max held: closed, no return line
    for task, room, mode, plan in (("kid", _room(), "low", {"role": "child"}), ("ownerlow", None, "low", {})):
        assert start(task, room, mode, **plan) == {} and len(sent) == 1  # untouched, closed
    assert start("rootD", _room())["memory"]["task"] == "rootD"
    assert [event["delivery_id"] for _root, event in sent] == ["low_start:rootA:1", "low_start:rootD:1"]


def test_only_a_start_with_the_opening_work_room_that_holds_max_closes_the_period(tmp_path, sent):
    """Opus RV1 F2: a root keeps two eighths of its window for work, a chat turn or wake one. Memory in the band
    between lets a turn hold Max where a root cannot; that proves no room for the next root, so the period stays open
    and the next root's Low adds no line. A root that holds Max closes it, and a turn's own period closes on a turn."""
    ctx = SimpleNamespace(task_metadata={"budget_drive_root": str(tmp_path)}, drive_root=tmp_path / "drive",
                          task_attempt=None)
    starts = (("rootA", "crowded", 2), ("turnA", "held", 1), ("rootB", "crowded", 2),  # a turn's Max: still open
              ("rootC", "held", 2), ("rootD", "crowded", 2),  # a root held Max: the next Low is a new period
              ("rootE", "held", 2), ("turnB", "crowded", 1),  # a chat turn opens its own period
              ("turnC", "held", 1), ("turnD", "crowded", 1))  # and a chat turn's Max closes it
    for task, cause, levels in starts:
        loop._note_low_start(ctx, _plan(_room(cause, levels=levels)), "max" if cause == "held" else "low", task, None)
    assert [event["delivery_id"] for _root, event in sent] == [
        "low_start:rootA:1", "low_start:rootD:1", "low_start:turnB:1", "low_start:turnD:1"]
    bit = json.loads((tmp_path.resolve() / "state" / "low_start_episode.json").read_text(encoding="utf-8"))
    assert (bit["memory"]["task"], bit["memory"]["levels"]) == ("turnD", 1)


def test_each_cause_has_its_own_period_closed_by_a_held_start_of_its_openers_room_and_work_room(tmp_path, sent):
    """The coordinator's option C, 22.11.2026: one period per cause, ``memory`` for the installation and
    ``request:<model>`` and ``route:<model>`` per model. A route start never silences the memory line and a second start
    of an open cause is silent. Fable's seat (22.11): a Max held in another room, or beside another work room, closes
    nothing; one of the opener's room and work room closes its periods silently, the others stay open, and the next
    start of that cause sends its line again. The single episode written before periods silences no first line."""
    ctx = SimpleNamespace(task_metadata={"budget_drive_root": str(tmp_path)}, drive_root=tmp_path / "drive",
                          task_attempt=None)
    bit = tmp_path.resolve() / "state" / "low_start_episode.json"
    bit.parent.mkdir(parents=True)
    bit.write_text(json.dumps({"since": "2026-11-21T10:00:00+00:00", "task": "old", "levels": 1}), encoding="utf-8")
    request = {"zero_need": 1_100_000, "floor_need": 950_000}  # the books fit without the tool schemas, not with them
    starts = (("rootA", "crowded", 1, "1", "", {}),  # the old episode holds no period: the memory line
              ("rootB", "route", 1, "1", "opus", {}),  # the open memory period never silences the route line
              ("rootC", "crowded", 1, "1", "", {}), ("rootD", "route", 1, "1", "opus", {}),  # both open: silent
              ("rootE", "route", 1, "nova", "glm", {}),  # another model's route: its own period, in another room
              ("rootF", "crowded", 1, "1", "opus", request),  # what the request carries: its own period
              ("turnG", "held", 1, "3", "", {}), ("rootH", "held", 2, "1", "", {}),  # another room or work room
              ("rootI", "crowded", 1, "1", "", {}),  # memory still open: silent
              ("rootJ", "held", 1, "1", "", {}),  # the opener's room and work room: closed silently, nova's stays
              ("rootK", "crowded", 1, "1", "", {}), ("rootL", "route", 1, "1", "glm", {}))  # a new line; glm still open
    for task, cause, levels, room, model, numbers in starts:
        plan = _plan(_room(cause, levels=levels, room_id=room, **numbers), model=model)
        loop._note_low_start(ctx, plan, "max" if cause == "held" else "low", task, None)
    assert [event["delivery_id"] for _root, event in sent] == [
        "low_start:rootA:1", "low_start:rootB:1", "low_start:rootE:1", "low_start:rootF:1", "low_start:rootK:1"]
    assert "on opus the window cannot hold" in sent[1][1]["text"] and "what the request itself" in sent[3][1]["text"]
    periods = json.loads(bit.read_text(encoding="utf-8"))
    assert sorted(periods) == ["memory", "route:glm"]
    assert [(periods[key]["task"], periods[key]["room"]) for key in ("memory", "route:glm")] == [("rootK", "1"), ("rootE", "nova")]


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
