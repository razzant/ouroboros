"""The own clock's named time (``ouroboros/consciousness_clock.py``): an explicit instant, consumed only by the
settlement of a wake appointed after it was stored that ends at or after it.

How the alarm uses a named time is pinned in ``test_consciousness.py``; the tool's arguments and receipts in
``test_consciousness_wake_lane.py``.
"""
from __future__ import annotations

import datetime as dt

import pytest

from ouroboros import consciousness_clock as clock

NOW = dt.datetime(2026, 11, 21, 2, 0, tzinfo=dt.timezone.utc).timestamp()


@pytest.mark.parametrize(("raw", "problem"), [
    ("2026-11-21T09:00:00", "carries no UTC offset"),  # never read as UTC: on a UTC server 9:00 is not his morning
    ("2026-11-21", "carries no UTC offset"),
    ("tomorrow at nine", "is not an ISO 8601 time"),
    ("", "is not an ISO 8601 time"),
    (900, "is not an ISO 8601 time"),
    ("2026-11-21T04:59:00+03:00", "is not in the future"),
    ("2026-11-21T02:00:00Z", "is not in the future"),  # this very second is not ahead
])
def test_a_named_time_needs_an_explicit_offset_and_the_future(raw, problem):
    instant, why = clock.named_time(raw, now=NOW)
    assert instant is None and problem in why
    assert ("e.g. 2026-11-21T09:00:00+03:00" in why) is (problem != "is not in the future")  # how to write it


@pytest.mark.parametrize("raw", ["2026-11-21T09:00:00+03:00", "2026-11-21T06:00:00Z", " 2026-11-21T06:00:00+00:00 "])
def test_a_named_time_with_its_offset_is_that_instant(raw):
    instant, why = clock.named_time(raw, now=NOW)
    assert why == "" and instant.timestamp() == NOW + 4 * 3600


def test_only_a_wake_appointed_after_the_time_was_stored_and_ending_at_or_after_it_consumes_it():
    record = {"at": "2026-11-21T06:00:00Z", "stored_at": "2026-11-21T02:00:00Z", "standing_appointment": NOW - 60}
    target = NOW + 4 * 3600
    assert clock.consumed_by(record, appointment_at=NOW + 3600, now=target)
    assert clock.consumed_by(record, appointment_at=NOW + 3600, now=target + 600)
    assert not clock.consumed_by(record, appointment_at=NOW + 3600, now=target - 1), "not yet reached"
    assert not clock.consumed_by(record, appointment_at=NOW - 60, now=target + 600), "the wake that stood appointed"
    stored_while_off = {**record, "standing_appointment": None}
    assert clock.consumed_by(stored_while_off, appointment_at=NOW - 60, now=target)
    assert not clock.consumed_by(None, appointment_at=NOW, now=target) and clock.target_ts(None) is None

