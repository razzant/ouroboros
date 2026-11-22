"""The own clock's facts beside the alarm (``ouroboros/consciousness_clock.py``): a named time is an explicit
instant, my human's last reported zone comes only from their own zoned message at ingress, and the "Own clock"
lines a wake reads say both with this wake's appointment and the owner's bounds, as facts, not instructions.

How the alarm uses a named time is pinned in ``test_consciousness_named_time.py``; the tool's arguments and receipts
in ``test_consciousness_wake_lane.py``.
"""
from __future__ import annotations

import datetime as dt

import pytest

from ouroboros import consciousness_clock as clock
from ouroboros.consciousness import APPOINTMENT_STATE_KEY

NOW = dt.datetime(2026, 11, 21, 2, 0, tzinfo=dt.timezone.utc).timestamp()
# "is appointed": a skip (an exhausted allowance waits for its reset) can start it later than MAX after the end
BOUNDS_LINE = ("- each next wake-up is appointed no sooner than MIN 300 s and no later than MAX 3600 s after a "
               "wake-up ends (owner settings); a named time beyond MAX is reached through intermediate wake-ups")


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


def test_own_clock_lines_without_a_named_time_or_a_reported_zone():
    lines = clock.own_clock_lines({APPOINTMENT_STATE_KEY: NOW - 30}, now=NOW, floor=300, ceiling=3600)
    assert lines == [
        "- named next wake-up time: none; the interval alone sets the next wake-up",
        "- this wake's appointment: 2026-11-21T01:59:30Z (whether it came on time: the reason line)",
        BOUNDS_LINE,
        "- my human's last reported time zone: unknown",
    ]
    assert clock.own_clock_lines({}, now=NOW, floor=300, ceiling=3600)[1] == BOUNDS_LINE  # no appointment, no line


def test_own_clock_lines_say_how_far_ahead_whether_this_wake_consumes_it_and_the_zone_offset_now():
    record = {"at": "2026-11-21T09:00:00+03:00", "stored_at": "2026-11-21T01:00:00Z", "standing_appointment": NOW - 30}
    zone = {"zone": "Europe/Moscow", "at": "2026-11-20T22:42:11.500000+00:00",
            "message": {"chat_id": 1, "client_message_id": "c1"}}
    state = {APPOINTMENT_STATE_KEY: NOW - 30, clock.WAKE_TARGET_STATE_KEY: record, clock.OWNER_ZONE_STATE_KEY: zone}
    lines = clock.own_clock_lines(state, now=NOW, floor=300, ceiling=3600)
    assert lines[0] == ("- named next wake-up time: 2026-11-21T09:00:00+03:00, in 4 h 0 min; the settlement of a "
                        "wake-up appointed after it was stored consumes it once that wake-up ends at or after it; "
                        "this wake was appointed before it was stored, so its end does not")
    assert lines[1:3] == ["- this wake's appointment: 2026-11-21T01:59:30Z (whether it came on time: the reason line)",
                          BOUNDS_LINE]
    assert lines[3] == ("- my human's last reported time zone: Europe/Moscow (UTC+03:00 now), "
                        "from their message at 2026-11-20T22:42:11Z")
    appointed_after = clock.own_clock_lines({**state, APPOINTMENT_STATE_KEY: NOW + 0.5}, now=NOW, floor=300, ceiling=3600)
    assert appointed_after[0].endswith("consumes it once that wake-up ends at or after it")
    passed = clock.own_clock_lines({**state, clock.WAKE_TARGET_STATE_KEY: {**record, "at": "2026-11-21T04:30:00+03:00"}},
                                   now=NOW, floor=300, ceiling=3600)
    assert passed[0].startswith("- named next wake-up time: 2026-11-21T04:30:00+03:00, passed 30 min ago;")


@pytest.mark.parametrize(("month", "offset"), [(11, "UTC-05:00"), (7, "UTC-04:00")])
def test_the_zone_line_prints_the_offset_the_zone_has_at_the_wake(month, offset):
    now = dt.datetime(2026, month, 21, 12, tzinfo=dt.timezone.utc).timestamp()
    fact = {"zone": "America/New_York", "at": "2026-11-20T12:00:00Z"}
    line = clock.own_clock_lines({clock.OWNER_ZONE_STATE_KEY: fact}, now=now, floor=300, ceiling=3600)[-1]
    assert line == f"- my human's last reported time zone: America/New_York ({offset} now), from their message at 2026-11-20T12:00:00Z"


# --- the zone fact at owner ingress -------------------------------------------------------------------------------


class _Bridge:
    def __init__(self, messages):
        self._messages = list(messages)

    def get_updates(self, offset=0, timeout=1):
        return [{"update_id": offset + index, "message": message} for index, message in enumerate(self._messages)]

    def broadcast(self, _payload):
        pass


class _Ctx:
    """The supervisor context the owner intake stamps activity through: an in-memory state."""

    def __init__(self, state):
        self.state, self.sent, self.consciousness, self.kill_workers = dict(state), [], None, None

    def load_state(self):
        return dict(self.state)

    def update_state(self, mutator):
        live = dict(self.state)
        mutator(live)
        self.state = live
        return live

    def send_with_budget(self, chat_id, text, **_kwargs):
        self.sent.append((chat_id, text))


BOUND = {"owner_id": 1, "owner_chat_id": 1, "owner_external_id": 7, "owner_external_chat_id": 42}


def _message(zone, *, chat=1, sender=1, source="web", name="m1"):
    surface = {"client_surface": {"timezone": zone, "received_at": "2026-11-21T02:00:00+00:00"}} if zone else {}
    return {"chat": {"id": chat}, "from": {"id": sender}, "source": source, "text": "hello",
            "client_message_id": name, "task_metadata": surface}


def _intake(monkeypatch, ctx, *messages):
    import server
    import supervisor.message_bus as message_bus

    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_route_owner_message", lambda bridge, ctx, incoming: None)
    server._process_bridge_updates(_Bridge(messages), 0, ctx)
    return ctx.state.get(clock.OWNER_ZONE_STATE_KEY)


def test_my_humans_own_web_message_with_a_valid_zone_writes_the_dated_fact(monkeypatch):
    fact = _intake(monkeypatch, _Ctx(BOUND), _message("Europe/Moscow", chat=5, name="web-1"))
    assert fact["zone"] == "Europe/Moscow" and dt.datetime.fromisoformat(fact["at"])
    assert fact["message"]["chat_id"] == 5 and fact["message"]["client_message_id"] == "web-1"


def test_an_invalid_zone_or_a_sender_other_than_my_human_leaves_the_fact(monkeypatch):
    earlier = {"zone": "Asia/Dubai", "at": "2026-11-19T10:00:00+00:00"}
    ctx = _Ctx({**BOUND, clock.OWNER_ZONE_STATE_KEY: earlier})
    assert _intake(monkeypatch, ctx, _message("Mars/Olympus_Mons", name="bad")) == earlier
    stranger = _message("Europe/Berlin", chat=99, sender=8, source="skill:telegram-bridge", name="x1")
    assert _intake(monkeypatch, ctx, stranger) == earlier, "another chat of an external transport is not my human"
    owner_chat = _message("Europe/Berlin", chat=42, sender=7, source="skill:telegram-bridge", name="x2")
    assert _intake(monkeypatch, ctx, owner_chat)["zone"] == "Europe/Berlin", "the bound owner chat is my human"


def test_an_unidentified_external_sender_is_not_my_human_where_no_external_owner_is_bound(monkeypatch):
    """With no external owner bound the binding pair reads (0, 0), and an unidentified external sender is (0, 0) too
    (the Host Service's sentinel): only a known, positive bound pair is my human, as at the Panic door beside it."""
    ctx = _Ctx({"owner_id": 1, "owner_chat_id": 1})
    unidentified = _message("Asia/Tokyo", chat=0, sender=0, source="skill:telegram-bridge", name="u1")
    assert _intake(monkeypatch, ctx, unidentified) is None
    assert _intake(monkeypatch, ctx, _message("Asia/Tokyo", name="w1"))["zone"] == "Asia/Tokyo", "the web owner's own"


def test_a_retry_takes_the_zone_and_its_time_from_the_row_it_repeats(monkeypatch):
    """A same-id web retry dispatches the first acceptance's row (``accepted_source_row``) beside its own metadata:
    a retry that sends no client_surface (Send again after reload) records the original's zone at the original's
    time, and one from another zone never moves it; a fresh message, and a non-web one without an accepted row, keep
    reading their own metadata."""
    from ouroboros.project_dialogue import build_owner_message_ref

    ts = "2026-11-21T08:00:00+00:00"

    def retry(metadata):
        ref = build_owner_message_ref(chat_id=1, client_message_id="r1", ts=ts, text="hello")
        row = {"direction": "in", "source": "web", "chat_id": 1, "client_message_id": "r1", "ts": ts, "text": "hello",
               "client_surface": {"timezone": "Europe/Moscow"}}
        return {**_message(None, name="r1"), "task_metadata": metadata, "accepted_source_ref": ref,
                "accepted_source_row": row}

    fact = _intake(monkeypatch, _Ctx(BOUND), retry({}))
    assert fact["zone"] == "Europe/Moscow" and fact["at"] == fact["message"]["ts"] == ts
    moved = _intake(monkeypatch, _Ctx(BOUND), retry({"client_surface": {"timezone": "Asia/Tokyo"}}))
    assert moved["zone"] == "Europe/Moscow" and moved["at"] == ts, "the retry's own zone is not the original's"
    assert _intake(monkeypatch, _Ctx(BOUND), _message("Asia/Tokyo", name="fresh"))["zone"] == "Asia/Tokyo"
    telegram = _message("Europe/Berlin", chat=42, sender=7, source="skill:telegram-bridge", name="tg1")
    assert _intake(monkeypatch, _Ctx(BOUND), telegram)["zone"] == "Europe/Berlin", "no accepted row: its metadata"


def test_a_named_delivery_whose_row_recorded_no_surface_takes_the_zone_its_metadata_carries(tmp_path, monkeypatch):
    """Named external ingress (``message_ingress.accept_local_message``) accepts the row without the surface the
    bound external owner supplied and queues it beside that metadata: the zone then comes from the metadata, dated by
    the accepted row's time; a row that recorded a surface still wins (the retry test above)."""
    import server
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_route_owner_message", lambda bridge, ctx, incoming: None)
    bridge = message_bus.LocalChatBridge()
    row, rejoined = message_bus.accept_local_message(
        bridge, tmp_path, "hello", chat_id=42, user_id=7, client_message_id="tg:42:1", source="skill:telegram-bridge",
        task_metadata={"client_surface": {"timezone": "Europe/Berlin"}})
    assert not rejoined and not row.get("client_surface"), "the named row keeps no surface"
    ctx = _Ctx({**BOUND, clock.OWNER_ZONE_STATE_KEY: {"zone": "UTC", "at": "2020-01-01T00:00:00+00:00"}})
    server._process_bridge_updates(bridge, 0, ctx)
    fact = ctx.state[clock.OWNER_ZONE_STATE_KEY]
    assert fact["zone"] == "Europe/Berlin" and fact["at"] == fact["message"]["ts"] == row["ts"]


def test_a_display_default_sender_in_the_bound_owner_chat_is_not_my_human(monkeypatch):
    """The zone takes the bridge's positive identities, as a task's owner binding does (``bound_external_owner``): in
    a private chat bound as (42, 42), a message without a sender id reads as the chat's id only by display default."""
    ctx = _Ctx({**BOUND, "owner_external_id": 42})
    nameless = {**_message("Europe/Berlin", chat=42, source="skill:telegram-bridge", name="d1"), "from": {}}
    assert _intake(monkeypatch, ctx, nameless) is None
    owner = _message("Europe/Berlin", chat=42, sender=42, source="skill:telegram-bridge", name="d2")
    assert _intake(monkeypatch, ctx, owner)["zone"] == "Europe/Berlin", "the bound owner's own positive pair"


def test_a_message_without_a_zone_keeps_the_earlier_dated_fact(monkeypatch):
    earlier = {"zone": "Europe/Moscow", "at": "2026-11-20T22:42:11+00:00", "message": {"client_message_id": "old"}}
    ctx = _Ctx({**BOUND, clock.OWNER_ZONE_STATE_KEY: earlier})
    assert _intake(monkeypatch, ctx, _message(None, name="no-zone"),
                   _message(None, chat=42, sender=7, source="skill:telegram-bridge", name="tg")) == earlier
    assert ctx.state["last_owner_message_at"], "the stamp it rides in still ran"


def test_the_zone_is_dated_by_its_own_message_and_an_older_one_never_replaces_a_newer_report(monkeypatch):
    """A message accepted earlier and processed later keeps the time it came in its reference (``accepted_source_ref``):
    that time dates the fact, so it never replaces a report dated later, and a newer one replaces it; a message whose
    reference carries no time is dated at ingress, as before."""
    from ouroboros.project_dialogue import build_owner_message_ref

    def accepted(zone, ts, name):
        ref = build_owner_message_ref(chat_id=1, client_message_id=name, ts=ts, text="hello")
        message = _message(zone, name=name)  # the row keeps the surface its acceptance reported (``accept_chat``)
        row = {"direction": "in", "source": "web", "chat_id": 1, "client_message_id": name, "ts": ts, "text": "hello",
               **message["task_metadata"]}
        return {**message, "accepted_source_ref": ref, "accepted_source_row": row}

    newer = {"zone": "Asia/Dubai", "at": "2026-11-21T09:00:00+00:00"}
    ctx = _Ctx({**BOUND, clock.OWNER_ZONE_STATE_KEY: newer})
    assert _intake(monkeypatch, ctx, accepted("Europe/Moscow", "2026-11-21T08:00:00+00:00", "a8")) == newer
    fact = _intake(monkeypatch, ctx, accepted("Europe/Berlin", "2026-11-21T09:30:00+00:00", "a930"))
    assert fact["zone"] == "Europe/Berlin" and fact["at"] == fact["message"]["ts"] == "2026-11-21T09:30:00+00:00"
    live = {clock.OWNER_ZONE_STATE_KEY: newer}
    clock.note_owner_zone(live, {"client_surface": {"timezone": "Asia/Tokyo"}}, own=True,
                          message={"chat_id": 1, "client_message_id": "c2"}, at="2026-11-21T10:00:00+00:00")
    assert live[clock.OWNER_ZONE_STATE_KEY] == {"zone": "Asia/Tokyo", "at": "2026-11-21T10:00:00+00:00",
                                                "message": {"chat_id": 1, "client_message_id": "c2"}}
