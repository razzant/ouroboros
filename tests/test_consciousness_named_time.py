"""How the alarm keeps a named time (``set_next_wakeup(at)``, ``consciousness_clock``): the earlier of it and the
interval while it lies ahead, never sooner than MIN nor later than MAX after a wake's end, and consumed by the
settlement of a wake appointed after it was stored. The alarm's fixture and its other behaviour are
``test_consciousness.py``'s; the tool's arguments and receipts are pinned in ``test_consciousness_wake_lane.py``, the
facts' lines in ``test_consciousness_clock.py``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import consciousness as clock_module
from ouroboros.consciousness import INTERVAL_STATE_KEY, LAST_WAKE_STATE_KEY, NEXT_WAKE_STATE_KEY, BackgroundConsciousness
from ouroboros.consciousness_clock import WAKE_TARGET_STATE_KEY
from tests.test_consciousness import AVAILABLE, CEILING, DEFAULT, FLOOR, T0, _iso, _launched
from tests.test_consciousness import clock as clock  # the alarm over an in-memory state: the fixture these tests share


# --- a named time (set_next_wakeup(at), consciousness_clock) ------------------------


def _call_clock_tool(monkeypatch, *, now, at=None, seconds=None):
    """``set_next_wakeup`` as the mind calls it at ``now``: from inside a wake or from a Main turn."""
    from ouroboros.tools import control_runtime

    monkeypatch.setattr(clock_module.time, "time", lambda: now)
    return control_runtime._set_next_wakeup(SimpleNamespace(task_id="caller"), seconds=seconds,
                                            at=None if at is None else _iso(at))


def _wake(clock, monkeypatch, task_id, *, start, end, ok=True, seconds=None, at=None):
    """One own wake launched at ``start`` and settled at ``end``; it may call the tool in between."""
    clock.receipt["task_id"] = task_id
    finished = _launched(clock, now=start)
    if seconds is not None or at is not None:
        _call_clock_tool(monkeypatch, now=start + 10, seconds=seconds, at=at)
    monkeypatch.setattr(clock_module.time, "time", lambda: end)
    finished(task_id, ok)


def _named(at, *, standing=None):
    return {"at": _iso(at), "stored_at": _iso(at - 3600), "standing_appointment": standing}


def test_a_named_time_ahead_sets_the_next_wake_and_the_wake_it_appointed_consumes_it(clock, monkeypatch):
    start = T0 + FLOOR + 1
    target = start + 60 + 1800
    _wake(clock, monkeypatch, "wake0001", start=start, end=start + 60, at=target)
    assert clock.clock.next_wake_at == target, "the earlier of the interval (3300 s) and the named time"
    assert clock.store[WAKE_TARGET_STATE_KEY]["at"] == _iso(target), "a wake ending before it leaves it standing"
    _wake(clock, monkeypatch, "wake0002", start=target, end=target + 60)
    assert WAKE_TARGET_STATE_KEY not in clock.store and clock.clock.next_wake_at == target + 60 + DEFAULT
    # Never sooner than MIN after a wake's end, however close the named time.
    _wake(clock, monkeypatch, "wake0003", start=clock.clock.next_wake_at, end=clock.clock.next_wake_at + 60,
          at=clock.clock.next_wake_at + 300)
    assert clock.clock.next_wake_at - clock.store[LAST_WAKE_STATE_KEY] == FLOOR


def test_a_named_time_beyond_max_is_reached_through_intermediate_wakes_that_set_seconds(clock, monkeypatch):
    start = T0 + FLOOR + 1
    # More than MAX (14400 s) after the first wake's end: one default step (3300 s), then 1800 s steps of 60 s wakes.
    target = start + 60 + (DEFAULT + 60) + 7 * (1800 + 60) + 1200
    _wake(clock, monkeypatch, "wake0001", start=start, end=start + 60, at=target)
    starts, index = [], 1
    while WAKE_TARGET_STATE_KEY in clock.store:
        index += 1
        begin = clock.clock.next_wake_at
        assert FLOOR <= begin - clock.store[LAST_WAKE_STATE_KEY] <= CEILING and index < 30
        starts.append(begin)
        _wake(clock, monkeypatch, f"wake{index:04d}", start=begin, end=begin + 60, seconds=1800)  # the habitual call
    assert starts[-1] == target and all(begin < target for begin in starts[:-1]), "consumed by the wake it appointed"
    assert clock.store[INTERVAL_STATE_KEY] == 1800 and clock.clock.next_wake_at == target + 60 + 1800


@pytest.mark.parametrize("ok", [True, False])
def test_a_wake_appointed_after_the_time_consumes_it_at_its_end_success_or_failure(clock, monkeypatch, ok):
    start = T0 + FLOOR + 1
    target = start + 60 + 1500
    _wake(clock, monkeypatch, "wake0001", start=start, end=start + 60, at=target)
    _wake(clock, monkeypatch, "wake0002", start=target, end=target + 60, ok=ok)
    assert WAKE_TARGET_STATE_KEY not in clock.store
    assert clock.clock.next_wake_at == target + 60 + (DEFAULT if ok else DEFAULT * 2)


def test_a_wake_pending_when_the_time_was_stored_keeps_it_for_the_next(clock, monkeypatch):
    start = T0 + FLOOR + 1
    _wake(clock, monkeypatch, "wake0001", start=start, end=start + 60)
    pending = clock.clock.next_wake_at
    receipt = _call_clock_tool(monkeypatch, now=start + 120, at=pending + 600)  # a Main turn between the wakes
    assert f"appointed for {clock_module._iso(pending)[:19]}Z keeps its time and does not consume it" in receipt
    assert clock.store[WAKE_TARGET_STATE_KEY]["standing_appointment"] == pending
    _wake(clock, monkeypatch, "wake0002", start=pending, end=pending + 900)  # it ran past the named time
    assert clock.store[WAKE_TARGET_STATE_KEY]["at"] == _iso(pending + 600), "the wake that stood appointed keeps it"
    assert clock.clock.next_wake_at == pending + 900 + DEFAULT, "a passed named time shortens nothing"
    _wake(clock, monkeypatch, "wake0003", start=clock.clock.next_wake_at, end=clock.clock.next_wake_at + 60)
    assert WAKE_TARGET_STATE_KEY not in clock.store


def test_a_refused_admission_retries_on_the_ordinary_path_shortened_only_by_a_time_ahead(clock):
    now = T0 + FLOOR + 1
    clock.store[WAKE_TARGET_STATE_KEY] = passed = _named(now - 60)
    clock.receipt.update({"admitted": False, "task_id": "", "reason": "budget_exhausted"})
    assert clock.clock.tick(now) == "rejected:budget_exhausted"
    assert clock.clock.next_wake_at == now + DEFAULT, "a passed named time never pins retries to MIN"
    assert clock.store[WAKE_TARGET_STATE_KEY] == passed, "a refusal never consumes it"
    clock.store[WAKE_TARGET_STATE_KEY], clock.clock._next_wake_at = _named(now + 1200), now
    assert clock.clock.tick(now) == "rejected:budget_exhausted" and clock.clock.next_wake_at == now + 1200


def test_after_a_failed_wake_a_named_time_still_fires_once_then_the_backoff_resumes(clock, monkeypatch):
    start = T0 + FLOOR + 1
    target = start + 60 + 2000
    _wake(clock, monkeypatch, "wake0001", start=start, end=start + 60, ok=False, at=target)
    assert clock.clock.next_wake_at == target, "earlier than the doubled interval (6600 s)"
    _wake(clock, monkeypatch, "wake0002", start=target, end=target + 60, ok=False)
    assert WAKE_TARGET_STATE_KEY not in clock.store and clock.clock.next_wake_at == target + 60 + DEFAULT * 4


@pytest.mark.parametrize("ahead", [7200, 1200])  # beyond the interval (3300 s) after the wake's end, and nearer
def test_a_newer_named_time_written_during_a_settlement_survives_and_sets_the_next_wake(clock, monkeypatch, ahead):
    """The settlement decides on the state its own write holds: a Main turn's ``at`` landing between the alarm's read
    and that write is kept (stored during this wake, not consumed by it) and is the one that sets the next wake."""
    from supervisor import state

    start = T0 + FLOOR + 1
    target = start + 60 + 1000
    _wake(clock, monkeypatch, "wake0001", start=start, end=start + 60, at=target)
    newer = _named(target + 60 + ahead, standing=target)
    settle_write = state.update_state

    def racing(mutator, **kwargs):
        if getattr(mutator, "__name__", "") == "settle":  # a Main turn's at lands between the read and the write
            clock.store[WAKE_TARGET_STATE_KEY] = newer
        return settle_write(mutator, **kwargs)

    monkeypatch.setattr(state, "update_state", racing)
    _wake(clock, monkeypatch, "wake0002", start=target, end=target + 60)
    assert clock.store[WAKE_TARGET_STATE_KEY] == newer
    assert clock.clock.next_wake_at == clock.store[NEXT_WAKE_STATE_KEY] == target + 60 + min(ahead, DEFAULT)


def test_turning_consciousness_on_comes_first_at_a_time_named_while_it_was_off(clock, monkeypatch):
    """"Stored for when it is enabled": a named time still ahead comes before the appointment the stop left, never
    sooner than MIN after the last wake; one after that appointment leaves it, and its own wake consumes it."""
    from ouroboros.consciousness import APPOINTMENT_STATE_KEY

    start = T0 + FLOOR + 1
    _wake(clock, monkeypatch, "wake0001", start=start, end=start + 60)
    end, kept = start + 60, clock.clock.next_wake_at

    def turned_on(named):
        clock.clock.stop(source=clock_module.OWNER_STOP)  # the owner's door saves the stop the tool reads
        clock.clock._next_wake_at = kept  # the appointment the stop left, each time
        clock.store.pop(WAKE_TARGET_STATE_KEY, None)
        if named is not None:
            receipt = _call_clock_tool(monkeypatch, now=end + 60, at=named)
            assert "Consciousness is off: it is stored for when it is enabled." in receipt
        clock.clock.start()
        return clock.clock.next_wake_at

    for named, due in ((end + 1500, end + 1500), (kept + 600, kept), (end + 120, end + FLOOR), (None, kept)):
        assert turned_on(named) == clock.store[APPOINTMENT_STATE_KEY] == due, (named, due)
    turned_on(end + 1500)
    _wake(clock, monkeypatch, "wake0002", start=end + 1500, end=end + 1560)
    assert WAKE_TARGET_STATE_KEY not in clock.store, "the wake it appointed consumes it"


def test_turning_consciousness_on_keeps_min_after_the_last_skip(clock, monkeypatch):
    """The tool's contract: no wake-up starts sooner than MIN after the last wake-up, boot or skip. A skip (here an
    exhausted allowance waiting for its reset) floors a time named while off; one later than every floor keeps it."""
    start = T0 + FLOOR + 1
    _wake(clock, monkeypatch, "wake0001", start=start, end=start + 60)
    skipped = clock.clock.next_wake_at
    resets = skipped + 6 * 3600
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_read: {
        **AVAILABLE, "status": "exhausted", "remaining_usd": 0.0, "resets_at": _iso(resets)})
    assert clock.clock.tick(skipped) == "skipped:allowance_exhausted" and clock.clock.next_wake_at == resets
    for named, due in ((skipped + 120, skipped + FLOOR), (skipped + FLOOR + 600, skipped + FLOOR + 600)):
        clock.clock.stop(source=clock_module.OWNER_STOP)
        clock.clock._next_wake_at = resets  # the attempt the skip stored, each time
        _call_clock_tool(monkeypatch, now=skipped + 60, at=named)
        clock.clock.start()
        assert clock.clock.next_wake_at == due, (named, due)


def test_a_restart_keeps_the_named_time_and_the_appointment_that_stood_when_it_was_stored(clock, monkeypatch):
    start = T0 + FLOOR + 1
    _wake(clock, monkeypatch, "wake0001", start=start, end=start + 60)
    pending = clock.clock.next_wake_at
    _call_clock_tool(monkeypatch, now=start + 120, at=pending + 600)
    alarm = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=pending - 100)  # a new process
    assert alarm.next_wake_at == pending + 800, "the boot floor still holds"
    clock.receipt["task_id"] = "wake0002"
    assert alarm.tick(alarm.next_wake_at) == "launched"
    monkeypatch.setattr(clock_module.time, "time", lambda: pending + 860)
    clock.finish("wake0002", True, alarm=alarm)
    assert clock.store[WAKE_TARGET_STATE_KEY]["at"] == _iso(pending + 600), "it stood appointed before the restart"
    clock.receipt["task_id"] = "wake0003"
    assert alarm.tick(alarm.next_wake_at) == "launched"
    monkeypatch.setattr(clock_module.time, "time", lambda: alarm.next_wake_at + 60)
    clock.finish("wake0003", True, alarm=alarm)
    assert WAKE_TARGET_STATE_KEY not in clock.store


@pytest.mark.parametrize("stored", ["named", "zone", "neither"])
def test_a_launched_wake_shows_the_own_clock_of_the_state_it_was_admitted_with(clock, stored):
    """``_launch`` carries the named time and my human's zone as it read them, this wake's appointment and the
    owner's bounds; the carrier's message shows them as the "Own clock" section only while either fact stands."""
    from ouroboros.consciousness import APPOINTMENT_STATE_KEY
    from ouroboros.consciousness_clock import OWNER_ZONE_STATE_KEY

    now = T0 + FLOOR + 1
    facts = {"named": {WAKE_TARGET_STATE_KEY: _named(now + 1800)},
             "zone": {OWNER_ZONE_STATE_KEY: {"zone": "Europe/Moscow", "at": _iso(now - 600)}}, "neither": {}}[stored]
    clock.store.update(facts)
    assert clock.clock.tick(now) == "launched"
    launch = clock.launches[-1]
    assert launch["prepare"]["clock"] == {**facts, APPOINTMENT_STATE_KEY: 0.0, "floor": FLOOR, "ceiling": CEILING}
    projection = launch["task"]["metadata"]["wake_observation"]["projection_text"]
    for text in (launch["text"], projection):
        assert ("\nOwn clock (3):\n- named next wake-up time: " in text) is (stored != "neither")
        assert (f"- named next wake-up time: {_iso(now + 1800)}, in 30 min;" in text) is (stored == "named")
        assert ("- my human's last reported time zone: Europe/Moscow (UTC+03:00 now)" in text) is (stored == "zone")
