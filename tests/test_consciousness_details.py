"""The Dashboard and state fallback share control truth and every clock status."""

import asyncio
import datetime as dt
import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from ouroboros import consciousness_view as view
from ouroboros.consciousness import BackgroundConsciousness
from ouroboros.gateway.state import api_consciousness


NOW = dt.datetime(2027, 1, 15, 12, tzinfo=dt.timezone.utc).timestamp()


@pytest.mark.parametrize("facts,status", [
    ({}, "sleeping"),
    ({"live_wake_task_id": "own-1"}, "thinking"),
    ({"live_wake_task_id": "own-1", "live_wake_held": True}, "wake_paused"),
    ({"last_wake_outcome": "skipped:waiting_for_first_conversation"}, "waiting_for_first_conversation"),
    ({"last_wake_outcome": "skipped:allowance_exhausted"}, "allowance_exhausted"),
    ({"last_wake_outcome": "skipped:allowance_unknown"}, "allowance_unknown"),
    ({"last_wake_outcome": "rejected:budget_exhausted"}, "wake_rejected"),
    ({"last_wake_outcome": "paused"}, "wake_paused"),
    ({"last_wake_outcome": "pausing"}, "wake_paused"),
    ({"last_wake_outcome": "unknown"}, "wake_outcome_unknown"),
    ({"last_wake_outcome": "failed"}, "wake_failed"),
    ({"next_wake_at": "2027-01-15T11:59:59+00:00"}, "due"),
    ({"next_wake_at": "2027-01-15T11:59:59+00:00", "live_wake_task_id": "own-1"}, "thinking"),
])
def test_primary_endpoint_and_state_fallback_share_status(monkeypatch, tmp_path, facts, status):
    import server
    from supervisor import state

    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: NOW))
    snapshot = {"enabled": True, "next_wake_at": "2027-01-15T12:30:00+00:00",
                "allowance_resets_at": "2027-01-15T18:00:00+00:00", "spent_24h_usd": 21.5,
                "daily_usd": 20, "unknown_unmetered": 2, "integrity_degraded": True,
                "places_brief": [{"place_id": "mail", "label": "Inbox"}], **facts}
    clock = object.__new__(BackgroundConsciousness)
    clock._drive_root = tmp_path
    monkeypatch.setattr(clock, "status_snapshot", lambda: dict(snapshot))
    monkeypatch.setattr(server, "_consciousness", clock)
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True})
    def wakes(root, places, *, live_wake_task_id, ids):
        assert live_wake_task_id == snapshot.get("live_wake_task_id", "")
        assert places == snapshot["places_brief"]
        assert ids == ["wake"]
        return [{"task_id": "wake"}]
    monkeypatch.setattr(view, "wake_ids", lambda *args: ["wake"])
    monkeypatch.setattr(view, "recent_wakes", wakes)
    response = asyncio.run(api_consciousness(Request({"type": "http", "app": server.app.app})))
    panel = json.loads(response.body)
    fallback = server.app.app.state.describe_bg_consciousness_state(True)
    assert panel["status"] == status and panel["detail"]
    assert {key: panel[key] for key in fallback} == fallback
    assert clock.details_snapshot(True) == panel
    assert "places" not in panel and panel["recent_wakes"] == [{"task_id": "wake"}]
    assert panel["recent_wakes_coverage"] == "wakes that reached a model call in the last 48 h"
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("wake,held", [("", False), ("wake", False), ("wake", True)])
@pytest.mark.parametrize("appointment", ["", "2027-01-15T11:30:00Z"])
@pytest.mark.parametrize("deadline", ["2027-01-15T11:59:59Z", "2027-01-15T12:00:00Z",
                                     "2027-01-15T12:00:01Z", "", "invalid"])
def test_due_headline_requires_a_past_attempt_without_a_live_wake(monkeypatch, wake, held, appointment, deadline):
    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: NOW))
    monkeypatch.setattr(view, "_DISPLAY_ZONE", dt.timezone.utc)
    facts = {"next_wake_at": deadline, "live_wake_task_id": wake,
             "live_wake_held": held, "appointment_at": appointment}
    result = view.describe_consciousness_state(facts, True)
    due = deadline in {"2027-01-15T11:59:59Z", "2027-01-15T12:00:00Z"}
    if wake:
        assert result["status"] == ("wake_paused" if held else "thinking")
        assert ":" not in result["detail"]
        if not held:
            assert result["detail"] == "Wake-up wake is running."
    elif due:
        assert result["status"] == "due"
        assert result["detail"] == f"Due{' since 11:30' if appointment else ''}; the wake-up has not started yet."
    else:
        assert result["status"] == "sleeping"
    assert {key: result[key] for key in facts} == facts


@pytest.mark.parametrize("enabled,status", [(None, "unknown"), (False, "disabled")])
@pytest.mark.parametrize("held", [False, True])
def test_due_work_never_overrides_unknown_or_disabled_control(monkeypatch, enabled, status, held):
    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: NOW))
    result = view.describe_consciousness_state({"next_wake_at": "2027-01-15T11:59:59Z",
                                              "live_wake_task_id": "wake", "live_wake_held": held}, enabled)
    assert result["status"] == status


@pytest.mark.parametrize("outcome", ["skipped:allowance_unknown", "skipped:allowance_exhausted",
    "rejected:budget_exhausted", "paused", "pausing", "unknown", "failed"])
@pytest.mark.parametrize("hour,words", [(11, "due since 11:00"), (13, "at 13:00")])
def test_explained_outcomes_label_future_and_overdue_attempts(monkeypatch, outcome, hour, words):
    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: NOW))
    monkeypatch.setattr(view, "_DISPLAY_ZONE", dt.timezone.utc)
    result = view.describe_consciousness_state({"last_wake_outcome": outcome,
        "next_wake_at": f"2027-01-15T{hour}:00:00Z", "appointment_at": "2027-01-15T10:00:00Z"}, True)
    assert words in result["detail"] and "10:00" not in result["detail"]
    assert result["status"] not in {"due", "sleeping"}


@pytest.mark.parametrize("constructed", [False, True])
@pytest.mark.parametrize("enabled,status", [(None, "unknown"), (False, "disabled"), (True, "stopped")])
def test_stored_control_and_unconstructed_clock_are_visible(monkeypatch, tmp_path, constructed, enabled, status):
    import server
    from supervisor import state

    clock = object.__new__(BackgroundConsciousness)
    clock._drive_root = tmp_path
    monkeypatch.setattr(clock, "status_snapshot", lambda: {"enabled": not enabled})
    monkeypatch.setattr(view, "place_views", lambda root: [])
    monkeypatch.setattr(view, "recent_wakes", lambda root, places, **kw: [])
    monkeypatch.setattr(server, "_consciousness", clock if constructed else None)
    monkeypatch.setattr(state, "load_state", lambda: {} if enabled is None else {"bg_consciousness_enabled": enabled})
    response = asyncio.run(api_consciousness(Request({"type": "http", "app": server.app.app})))
    panel = json.loads(response.body)
    fallback = server.app.app.state.describe_bg_consciousness_state(enabled)
    assert panel["status"] == ("sleeping" if constructed and enabled else status)
    assert panel["enabled"] is enabled
    assert {key: panel[key] for key in fallback} == fallback
    assert ("not constructed" in panel["detail"]) == (not constructed and enabled is True)


@pytest.mark.parametrize("selected_count", [0, 49, 50])
def test_recent_wake_coverage_names_the_selection_cap_even_with_pruned_documents(monkeypatch, tmp_path, selected_count):
    from tests._wake_summary_shared import seed_wake

    document = seed_wake(tmp_path)
    ids = [document["task_id"], *[f"pruned-{i}" for i in range(selected_count - 1)]] if selected_count else []
    reads = []
    monkeypatch.setattr(view, "wake_ids", lambda *args: reads.append(args) or ids)
    clock = object.__new__(BackgroundConsciousness)
    clock._drive_root = tmp_path
    monkeypatch.setattr(clock, "status_snapshot", lambda: {"places_brief": []})
    panel = clock.details_snapshot(True)
    assert len(reads) == 1 and len(panel["recent_wakes"]) == bool(selected_count)
    prefix = "the latest 50 " if selected_count == 50 else ""
    assert panel["recent_wakes_coverage"] == prefix + "wakes that reached a model call in the last 48 h"


@pytest.mark.parametrize("own,places", [(50, 0), (0, 50), (49, 1), (25, 25), (1, 49), (24, 25)])
def test_recent_wake_coverage_names_actual_shares_only_on_a_full_mixed_page(monkeypatch, tmp_path, own, places):
    wakes = [{"task_id": str(i), "cause": "own_clock" if i < own else "focus"} for i in range(own + places)]
    monkeypatch.setattr(view, "wake_ids", lambda *args: [wake["task_id"] for wake in wakes])
    monkeypatch.setattr(view, "recent_wakes", lambda *args, **kwargs: wakes)
    clock = object.__new__(BackgroundConsciousness)
    clock._drive_root = tmp_path
    monkeypatch.setattr(clock, "status_snapshot", lambda: {"places_brief": []})
    coverage = clock.details_snapshot(True)["recent_wakes_coverage"]
    if own + places < 50:
        assert coverage == "wakes that reached a model call in the last 48 h"
    elif own and places:
        assert coverage == (f"the latest {own} of its own wakes and {places} wakes in places "
                            "that reached a model call in the last 48 h")
    else:
        assert coverage == "the latest 50 wakes that reached a model call in the last 48 h"


@pytest.mark.parametrize("offset", [0, 3])
@pytest.mark.parametrize("day,words", [(14, "Jan 14 10:00"), (15, "10:00"), (16, "Jan 16 10:00")])
def test_headlines_date_instants_outside_the_server_local_day(monkeypatch, offset, day, words):
    zone = dt.timezone(dt.timedelta(hours=offset))
    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: NOW))
    monkeypatch.setattr(view, "_DISPLAY_ZONE", zone)
    instant = dt.datetime(2027, 1, day, 10, tzinfo=zone).isoformat()
    assert view._clock_of(instant) == words
    facts = {"next_wake_at": instant, "appointment_at": instant}
    shown = view.describe_consciousness_state(facts, True)
    assert shown["detail"] == (f"Sleeping until {words}." if day == 16
                               else f"Due since {words}; the wake-up has not started yet.")
    exhausted = view.describe_consciousness_state({**facts, "allowance_resets_at": instant,
                                                  "last_wake_outcome": "skipped:allowance_exhausted"}, True)
    assert f"next check {'at' if day == 16 else 'due since'} {words}." in exhausted["detail"]


def test_host_date_follows_local_midnight_not_the_utc_day(monkeypatch):
    monkeypatch.setattr(view, "_DISPLAY_ZONE", dt.timezone(dt.timedelta(hours=3)))
    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: dt.datetime(2027, 1, 15, 22, tzinfo=dt.timezone.utc).timestamp()))
    assert view._clock_of("2027-01-15T21:30:00Z") == "00:30"
    assert view._clock_of("2027-01-15T20:30:00Z") == "Jan 15 23:30"
    assert view._clock_of("") == "?"
