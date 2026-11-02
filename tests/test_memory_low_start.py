"""One host line in Main for each fresh start that memory lowered to Low (owner decision 1B, 29.10).

No episode and no line when Max returns: the start's own Max room decides (``memory_floor.crowded_by_memory``), the
line goes through the terminal outbox under one id per task attempt, and a route, tool schemas that tip a start
alone, a held start, a delegated child or nanny and the owner's Low or Nano get none.
"""
from types import SimpleNamespace

import pytest

from ouroboros import loop
from ouroboros.context_view_facts import low_start_line
from supervisor import schedule_notes, terminal_delivery as outbox
from tests.test_memory_max_room import _core, _crowded_window, _fact, _start


def _room(cause="crowded", *, window=1_000_000, zero_need=900_000):
    return {"cause": cause, "window": window, "allowance": 40_000, "floor_need": zero_need - 70_000,
            "schemas": 70_000, "zero_need": zero_need, "need": zero_need + 150_000, "story": 110_000,
            "unfolded": 60_000, "room": 40_000, "reserve": 65_536, "work": 125_000, "levels": 1,
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


def test_a_start_memory_lowered_to_low_names_the_hosts_measure_without_advice():
    room = _room()
    line = low_start_line(_plan(room), "low", "root1")
    assert line == (f"Task root1 started in Low with chapter maps: its memory ({room['story'] + room['room']:,} tokens) "
                    f"did not fit the {room['allowance']:,} tokens left beside the full books.")
    assert len(line.splitlines()) == 1 and "my memory" not in line
    assert not any(word in line for word in ("may bring", "should", "must", "fold", "episode", "again"))


@pytest.mark.parametrize("plan", [
    _plan(_room("route")),                             # the books cannot fit this route: its task's facts say so
    _plan(_room(zero_need=1_100_000)),                 # tool schemas tip it alone: nothing memory could fold
    _plan(_room("held")),                              # Max held
    _plan(_room(), role="child"), _plan(_room(), role="nanny"),
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


def test_the_line_goes_through_the_outbox_once_per_task_attempt(tmp_path, sent):
    ctx = SimpleNamespace(task_metadata={"budget_drive_root": str(tmp_path)}, drive_root=tmp_path / "drive",
                          task_attempt=None)
    loop._note_low_start(ctx, _plan(_room()), "low", "root1", None)
    [(root, event)] = sent
    assert root == tmp_path.resolve() and event["chat_id"] == 7 and event["task_id"] == ""
    assert event["role"] == "system" and event["system_type"] == "low_start_notice" and event["type"] == "send_message"
    assert event["delivery_id"] == "low_start:root1:1" and event["text"].startswith("Task root1 started in Low")
    loop._note_low_start(SimpleNamespace(**{**vars(ctx), "task_attempt": 2}), _plan(_room()), "low", "root1", None)
    loop._note_low_start(ctx, _plan(_room()), "low", "root2", None)
    assert [event["delivery_id"] for _root, event in sent] == ["low_start:root1:1", "low_start:root1:2",
                                                               "low_start:root2:1"]
    loop._note_low_start(ctx, _plan(_room("held")), "max", "root3", None)  # Max held: no line, no return line
    assert len(sent) == 3


def test_a_failing_outbox_never_fails_the_start(tmp_path, monkeypatch):
    monkeypatch.setattr(outbox, "enqueue_terminal_delivery_outcome",
                        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("store busy")))
    monkeypatch.setattr(schedule_notes, "owner_chat_id", lambda _root: 7)
    loop._note_low_start(SimpleNamespace(task_metadata={}, drive_root=tmp_path), _plan(_room()), "low", "t", None)
