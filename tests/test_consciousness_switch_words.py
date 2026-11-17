"""The consciousness switch's words (owner 15.11, decision 5A): one home for every surface, true for the install's facts.

The app shows these sentences from the server's answers, the /bg replies and the mind's fact use them directly; each
block says only what the install's facts support and is absent without them.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import consciousness_view as V


def place(pid, label, state="active", provider="telegram"):
    return {"place_id": pid, "label": label, "state": state, "provider": provider}


PLACES = [place("a", "Telegram · Reading Club"), place("s", "Slack", provider="slack"),
          place("b", "Telegram · Lab chat · Наука/Lab"), place("e", "Email · Minutes", "silent", "email"),
          place("x", "Telegram · Strangers", "paused"), place("g", "GitHub · razzant/ouroboros", provider="github")]
PEOPLE = {"by_place": {"a": 129, "s": 84, "b": 34, "e": 3, "g": 7}, "total": 250, "capped": True}


def test_the_positions_are_named_by_meaning_and_unknown_is_never_off():
    assert V.position_words(False) == "Only when you write to it"
    assert V.position_words(True) == "Also on its own and in places"
    assert V.position_words(None) == ""  # an unknown control says nothing, never Off
    # An install that holds no place is not told of places; an unread place store keeps the whole words.
    assert V.position_words(True, []) == "Also on its own" and V.position_words(True, None) == V.position_words(True)
    assert V.position_words(True, [place("x", "X", "paused")]) == V.position_words(True)  # Paused is still a place
    assert V.describe_consciousness_state({"places_brief": []}, True)["position"] == "Also on its own"
    described = V.describe_consciousness_state({"places_brief": PLACES}, False)
    assert described["position"] == V.position_words(False) and described["gloss"] == V.SWITCH_GLOSS


def test_off_says_how_many_places_hear_and_a_paused_place_is_not_among_them():
    assert V.heard_sentence(PLACES) == "5 places are heard, not answered."
    assert V.heard_sentence([place("a", "A")]) == "1 place is heard, not answered."
    assert V.heard_sentence([place("x", "X", "paused")]) == V.heard_sentence([]) == "Nothing wakes it between your messages."
    assert V.heard_sentence(None) == "Its places are heard, not answered."  # an unread store claims no count
    assert V.describe_consciousness_state({"places_brief": None}, False)["detail"] == V.heard_sentence(None)


def test_turning_off_names_the_busiest_places_and_the_people_of_the_week():
    preview = V.switch_preview("off", PLACES, people=PEOPLE)
    names = [row["label"] for row in preview["places"]]
    assert names == ["Reading Club", "Slack", "Lab chat · Наука/Lab", "razzant/ouroboros", "Minutes"]  # busiest first
    said = " ".join(preview["lines"])
    assert "5 places will be heard, not answered: Reading Club, Slack, Lab chat · Наука/Lab and 2 more." in preview["lines"]
    assert "At least 250 people wrote there in the last 7 days." in preview["lines"]  # a capped read says «at least»
    assert "Your chats, your tasks and the schedules you asked for keep working." in preview["lines"]
    assert not any("stop" in line for line in preview["lines"])  # nothing running: nothing said to end
    assert "Strangers" not in said and all(row["place_id"] != "x" for row in preview["places"])  # Paused hears nothing


def test_turning_off_with_no_place_names_nothing_and_an_unread_store_claims_no_count():
    empty = V.switch_preview("off", [])
    assert empty["places"] == [] and not any("place" in line for line in empty["lines"])  # the app asks nothing then
    unread = V.switch_preview("off", None)
    assert "Its places will be heard, not answered; which ones could not be read." in unread["lines"]
    assert unread["places"] is None  # the app still asks
    quiet = V.switch_preview("off", PLACES, people={"by_place": {}, "total": 0, "capped": False})
    assert not any("people" in line for line in quiet["lines"])  # no one wrote: no sentence about people


def test_turning_on_says_where_it_may_answer_and_what_happens_to_what_waited():
    on = V.switch_preview("on", PLACES, people=PEOPLE, autonomy="full", daily_usd=200.0)["lines"]
    assert "It may answer in 4 places: Reading Club, Slack, Lab chat · Наука/Lab and 1 more." in on
    assert "In 1 Silent place it only reads." in on  # the list's other places are said, not left to the count
    assert any("at its first wake it sees what is unread there and decides what to read and answer" in line for line in on)
    advises = " ".join(V.switch_preview("on", PLACES, autonomy="observe", daily_usd=20.0)["lines"])
    assert "Advises: in 4 places" in advises and "may answer" not in advises
    broke = " ".join(V.switch_preview("on", PLACES, autonomy="act", daily_usd=0)["lines"])
    assert "cannot wake until you raise it in Costs" in broke and "may answer" not in broke
    alone = V.switch_preview("on", [], autonomy="act", daily_usd=20.0)
    assert alone["places"] == [] and len(alone["lines"]) == 1  # nothing waited, nowhere to answer


def test_the_bg_replies_say_the_same_facts_after_the_turn():
    stored = [SimpleNamespace(place_id=p["place_id"], label=p["label"], state=p["state"], provider=p["provider"]) for p in PLACES]
    # It promises no answer after the turn: a Silent place still sends nothing then (review r1).
    assert V.off_sentence(stored) == ("Consciousness off: Ouroboros acts only when you write to it. 5 places are heard, "
                                      "not answered.")
    assert V.off_sentence([]) == "Consciousness off: Ouroboros acts only when you write to it."
    assert "Its places are heard, not answered" in V.off_sentence(None)
    on = V.on_sentence("15:14", stored, "act")
    assert on.startswith("Consciousness on: Ouroboros also acts on its own; next wake at 15:14.") and "4 places" in on
    advises = V.on_sentence("15:14", stored, "observe")
    assert "Advises" in advises and "may answer" not in advises  # Advises answers nowhere


@pytest.mark.parametrize(("state", "switch", "autonomy", "allowance", "cause"), [
    ("active", "off", "full", "available", "off"),
    ("silent", "off", "full", "available", "off"),
    ("active", "panic", "full", "available", "panic"),
    ("active", "on", "full", "exhausted", "allowance"),
    ("silent", "on", "full", "exhausted", "allowance"),
    ("active", "on", "observe", "available", "advises"),
    ("active", "on", "observe", "exhausted", "allowance"),  # nothing wakes it before there is room again
    ("silent", "on", "observe", "available", None),  # Silent already holds: its own word says it
    ("active", "on", "full", "available", None),  # it speaks: not quiet
    ("paused", "off", "full", "available", None),  # Paused says Paused
    ("active", "unknown", "full", "available", None),  # an unknown switch is no claim
])
def test_a_place_is_quiet_for_one_cause_first_match_wins(state, switch, autonomy, allowance, cause):
    quiet = V.place_quiet(state, switch=switch, autonomy=autonomy, allowance=allowance, limit_usd=20.0)
    assert (quiet or {}).get("cause") == cause
    if quiet:
        assert quiet["word"] and quiet["note"] and quiet["door"] in ("consciousness", "costs")


def test_the_quiet_note_says_what_it_does_here_now_and_promises_only_waking():
    # Off promises the wake, never an answer: a Silent (or Advises) place still sends nothing once it is on (review r1).
    off = V.place_quiet("silent", switch="off")
    assert off["note"] == "Consciousness is off: it only hears here, and nothing here wakes it until you turn it on."
    assert off["word"] == "Hears only · consciousness off" and "answer" not in off["note"]
    assert "waits for you in its conversation here" in V.place_quiet("active", switch="on", autonomy="observe")["note"]
    # A spent allowance names no clock (the oldest spend leaving the window need not leave room) and no read; a $0 one
    # has no reset at all: only the owner opens it (review r1).
    spent = V.place_quiet("active", switch="on", allowance="exhausted", limit_usd=3.0)
    assert spent["note"] == ("Its allowance for the last 24 hours is spent: nothing here starts a new wake until there is room "
                             "in it again.") and spent["door"] == "costs"  # Costs may not be able to say when (review r4)
    zero = V.place_quiet("active", switch="on", allowance="exhausted", limit_usd=0.0)
    assert zero["note"] == "Its allowance is $0.00: nothing here starts a new wake until you raise it in Costs."
    assert zero["door"] == "costs"


def test_connecting_a_place_says_the_truth_for_the_switch_as_it_stands():
    assert "until you make it Active. While consciousness is off, it only hears." in V.connect_note("off")
    assert V.connect_note("panic") == V.connect_note("off")
    assert "hears and thinks there" in V.connect_note("on")


def test_the_mind_gets_the_switch_as_a_fact_only_while_its_places_only_hear():
    fact = V.mind_fact(PLACES, enabled=False, stop_source="owner")
    assert fact.startswith("Consciousness is off (my human turned it off):") and "In my 5 connected places what arrives" in fact
    assert "recorded and wakes nothing" in fact and "I do not" not in fact  # a fact, never a rule of conduct
    assert "the schedules my human asked for, still reach me" in fact  # his own work is not said to stop
    assert "the schedules I set for myself have stopped" in fact  # and its own are (supervisor autonomous_work_enabled)
    assert "my human pressed Panic" in V.mind_fact(PLACES, enabled=False, stop_source="panic")
    assert "a Panic waits" in V.mind_fact(PLACES, enabled=True, panic=True)
    # Absent the evidence, no block: on, unknown, or no place that hears (a default install's request is unchanged).
    assert V.mind_fact(PLACES, enabled=True) == V.mind_fact(PLACES, enabled=None) == ""
    assert V.mind_fact([], enabled=False) == V.mind_fact([place("x", "X", "paused")], enabled=False) == ""


def test_the_first_run_offer_carries_the_switch_words():
    from ouroboros.settings_setup_contract import consciousness_offer

    words = consciousness_offer({}, enabled=False)["words"]
    assert words == V.FIRST_RUN and words["choice"] == V.position_words(True)
    assert "places you connect are heard, not answered" in words["meaning"]


def test_recent_people_counts_each_person_once_across_places_and_skips_bots(tmp_path):
    from ouroboros import dialogue_store as ds
    from ouroboros.place_arrivals import recent_people

    def arrival(pid, actor, *, bot=False, ts="2026-11-17T10:00:00+00:00"):
        ds.append(tmp_path, "chat", {
            "ts": ts, "direction": "in", "type": "place_arrival", "chat_id": 9, "text": "hi", "source": "presence:slack",
            "sender_label": actor, "sender_session_id": actor, "client_message_id": f"{pid}:{actor}:{ts}",
            "place_id": pid, "transport": {"provider": "slack", "actor": {"platform_actor_id": actor, "is_bot": bot}}})

    arrival("p1", "U1"); arrival("p1", "U2"); arrival("p2", "U1"); arrival("p2", "BOT", bot=True)
    arrival("p2", "U3", ts="2026-11-01T10:00:00+00:00")  # outside the week
    from datetime import datetime
    now = datetime.fromisoformat("2026-11-17T12:00:00+00:00").timestamp()
    people = recent_people(tmp_path, ["p1", "p2", "p3"], days=7, now=now)
    assert people == {"by_place": {"p1": 2, "p2": 1, "p3": 0}, "total": 2, "capped": False, "capped_places": []}


def test_a_turn_off_says_what_it_ends_at_once_only_when_something_runs():
    running = {"live_wake_task_id": "w1", "live_wake_held": False, "tasks_running": 2}
    lines = V.switch_preview("off", PLACES, running=running)["lines"]
    # The stop asks: no instant end and no saved result is promised (review r3).
    assert "The wake running now and its 2 own tasks are asked to stop now." in lines
    assert "The wake running now is asked to stop now." in V.switch_preview("off", PLACES, running={**running, "tasks_running": 0})["lines"]
    assert "Its 1 own task is asked to stop now." in V.switch_preview("off", PLACES, running={"tasks_running": 1})["lines"]
    assert not any("at once" in line or "saves" in line for line in lines)
    held = {"live_wake_task_id": "w1", "live_wake_held": True}
    assert not any("stop" in line for line in V.switch_preview("off", PLACES, running=held)["lines"])  # a held wake is not running


def test_a_place_is_named_as_its_row_names_it():
    named = V.switch_preview("off", PLACES, circles={"b": "Наука/Lab"})["places"]
    assert "Lab chat" in [row["label"] for row in named]  # its circle's name is the row's head, not its name


def test_the_bg_status_reply_names_the_position_and_the_level_while_on():
    on = V.status_reply({"enabled": True, "position": V.position_words(True), "level": "observe", "detail": "Sleeping until 15:14."})
    assert on == "🧠 Consciousness: Also on its own and in places (Advises) — Sleeping until 15:14."
    off = V.status_reply(V.describe_consciousness_state({"places_brief": PLACES, "stop_source": "panic"}, False))
    assert off == "🧠 Consciousness: Only when you write to it — Panic turned it off. 5 places are heard, not answered."


def test_the_level_names_are_the_web_strips_own_words():
    """One owner's word per level (21.10): the server's /bg reply and the Dashboard's strip say the same names."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "web/modules/consciousness_autonomy.js").read_text(encoding="utf-8")
    pairs = dict(re.findall(r"value: '(\w+)', label: '(\w+)'", source))
    assert pairs == V.LEVEL_NAMES


def test_places_that_could_not_be_read_are_not_no_places(tmp_path, monkeypatch):
    """A place store this read cannot open says so, and the app still asks (``places: None``); no places at all asks
    nothing (``places: []``)."""
    off, on = V.switch_preview("off", None), V.switch_preview("on", None)
    assert off["places"] is None and "Its places will be heard, not answered; which ones could not be read." in off["lines"]
    assert on["places"] is None and "Where it may answer could not be read." in on["lines"]
    assert V.switch_preview("off", [])["places"] == [] and V.switch_preview("on", [])["places"] == []
    assert "Where it may answer could not be read." not in V.switch_preview("on", [])["lines"]

    import ouroboros.places as places_module

    def unreadable(_root):
        raise OSError("places.json unreadable")

    monkeypatch.setattr(places_module, "place_records", unreadable)
    assert V.read_switch_preview(tmp_path, "off")["places"] is None


def test_the_preview_route_reads_the_turn_and_a_plain_read_stays_the_panel(tmp_path, monkeypatch):
    """``GET /api/consciousness?switch=off|on`` answers the turn's preview (never cached), turning off with the state
    line's running work; any other value is the panel's ordinary read."""
    import asyncio
    import json

    from starlette.requests import Request

    from ouroboros.gateway.state import api_consciousness

    asked = []

    def preview(root, to, running=None):
        asked.append((root, to, running))
        return {"to": to, "lines": [f"turn {to}"], "places": []}

    monkeypatch.setattr(V, "read_switch_preview", preview)
    line = {"status": "sleeping", "live_wake_task_id": "w1"}
    app = SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path, describe_bg_consciousness_state=lambda _on: line,
                                                describe_consciousness=None))

    def get(query):
        return asyncio.run(api_consciousness(Request({"type": "http", "query_string": query, "app": app})))

    off = get(b"switch=off")
    assert json.loads(off.body) == {"to": "off", "lines": ["turn off"], "places": []}
    assert off.headers["cache-control"] == "no-store"
    on = get(b"switch=on")
    assert json.loads(on.body)["to"] == "on"
    assert asked == [(tmp_path, "off", line), (tmp_path, "on", None)]
    assert get(b"switch=maybe").status_code == 503 and len(asked) == 2  # the panel's own read, no preview


def test_recent_people_reads_at_most_its_cap_per_place_and_says_so(tmp_path, monkeypatch):
    """A busy place is read to ``PEOPLE_ROW_CAP`` newest rows, and the count then says «at least» (``capped``); the
    same raw id on two services is two people."""
    from datetime import datetime

    import ouroboros.place_arrivals as arrivals
    from ouroboros import dialogue_store as ds

    def arrival(pid, actor, ts, provider="slack"):
        ds.append(tmp_path, "chat", {
            "ts": ts, "direction": "in", "type": "place_arrival", "chat_id": 9, "text": "hi", "source": f"presence:{provider}",
            "sender_label": actor, "sender_session_id": actor, "client_message_id": f"{pid}:{actor}:{ts}",
            "place_id": pid, "transport": {"provider": provider, "actor": {"platform_actor_id": actor, "is_bot": False}}})

    for minute in range(5):
        arrival("busy", f"U{minute}", f"2026-11-17T10:0{minute}:00+00:00")
    arrival("gh", "U4", "2026-11-17T10:00:00+00:00", provider="github")  # U4 is also among the busy place's newest
    now = datetime.fromisoformat("2026-11-17T12:00:00+00:00").timestamp()
    monkeypatch.setattr(arrivals, "PEOPLE_ROW_CAP", 3)
    people = arrivals.recent_people(tmp_path, ["busy", "gh"], days=7, now=now,
                                    services={"busy": "slack", "gh": "github"})
    assert people == {"by_place": {"busy": 3, "gh": 1}, "total": 4, "capped": True, "capped_places": ["busy"]}
    assert V._people_line(people) == "At least 4 people wrote there in the last 7 days."


def test_the_preview_counts_people_by_service(tmp_path, monkeypatch):
    """The real preview hands each place's service to the people count (review r2: without it two services merge)."""
    import ouroboros.place_arrivals as arrivals
    import ouroboros.places as places_module

    stored = {"t": SimpleNamespace(place_id="t", label="Telegram · Lab", state="active", provider="telegram"),
              "g": SimpleNamespace(place_id="g", label="GitHub · repo", state="active", provider="github")}
    monkeypatch.setattr(places_module, "place_records",
                        lambda _root: SimpleNamespace(places=stored, circle_context=lambda _pid: None))
    seen = {}

    def people(_root, ids, *, days, now=None, services=None):
        seen.update(ids=list(ids), days=days, services=services)
        return {"by_place": {"t": 1, "g": 1}, "total": 2, "capped": False}

    monkeypatch.setattr(arrivals, "recent_people", people)
    preview = V.read_switch_preview(tmp_path, "off")
    assert seen == {"ids": ["t", "g"], "days": 7, "services": {"t": "telegram", "g": "github"}}
    assert "2 people wrote there in the last 7 days." in preview["lines"]



def test_the_turn_on_receipt_says_what_the_dialog_said_and_names_no_wake_at_zero():
    """`/bg start` on Telegram and the CLI hears the dialog's own sentences (names, Silent, Advises); a $0 allowance
    names no wake (review r3)."""
    stored = [SimpleNamespace(place_id=p["place_id"], label=p["label"], state=p["state"], provider=p["provider"]) for p in PLACES]
    on = V.on_sentence("15:14", stored, "act", 20.0)
    assert on.startswith("Consciousness on: Ouroboros also acts on its own; next wake at 15:14.")
    assert "It may answer in 4 places: Lab chat · Наука/Lab, razzant/ouroboros, Reading Club and 1 more." in on  # by name
    assert "In 1 Silent place it only reads." in on and "sees what is unread there" in on
    zero = V.on_sentence("15:14", stored, "act", 0.0)
    assert zero.startswith("Consciousness on, but its allowance is $0.00: it cannot wake until you raise it in Costs.")
    assert "next wake" not in zero and "may answer" not in zero
    advises = V.on_sentence("15:14", stored, "observe", 20.0)
    assert "Advises: in 4 places it reads and thinks; what it would say waits for you." in advises
    assert "sees what is unread there" in advises


def test_a_reaction_with_no_words_is_no_writing_and_a_capped_place_says_at_least(tmp_path):
    from datetime import datetime

    from ouroboros import dialogue_store as ds
    from ouroboros.place_arrivals import recent_people

    def arrival(pid, actor, text="hi", reaction=False):
        message = {"reaction": {"schema_version": 1, "values": [{"emoji": "+1"}]}} if reaction else {}
        ds.append(tmp_path, "chat", {
            "ts": "2026-11-17T10:00:00+00:00", "direction": "in", "type": "place_arrival", "chat_id": 9, "text": text,
            "source": "presence:slack", "sender_label": actor, "sender_session_id": actor,
            "client_message_id": f"{pid}:{actor}:{text}:{reaction}", "place_id": pid,
            "transport": {"provider": "slack", "message": message, "actor": {"platform_actor_id": actor, "is_bot": False}}})

    arrival("p1", "U1"); arrival("p1", "U2", text="", reaction=True); arrival("p1", "U3", text="nice", reaction=True)
    now = datetime.fromisoformat("2026-11-17T12:00:00+00:00").timestamp()
    assert recent_people(tmp_path, ["p1"], days=7, now=now)["by_place"] == {"p1": 2}  # U1 and U3 wrote; U2 only reacted
    rows = V._ranked([{"place_id": "p1", "label": "Ops", "state": "active"}],
                     {"by_place": {"p1": 2000}, "total": 2000, "capped": True, "capped_places": ["p1"]})
    assert rows[0]["people_capped"] is True


def test_an_unconfirmed_stop_source_names_no_panic(tmp_path, monkeypatch):
    """Off confirmed with its source awaiting confirmation is a plain Off, never «Panic» (review r3)."""
    import supervisor.state as state_module
    from ouroboros import presence_admission

    monkeypatch.setattr("ouroboros.consciousness.panic_blocks_wake", lambda _root: False)
    projections = {
        "confirmed": {"bg_consciousness_enabled": False, "bg_consciousness_stop_source": "panic"},
        "unconfirmed": {"bg_consciousness_enabled": False, "bg_consciousness_stop_source": "panic",
                        state_module.STATE_READ_KEY: {"quality": "current", "unconfirmed": ["bg_consciousness_stop_source"]}},
    }
    for name, expected in (("confirmed", "panic"), ("unconfirmed", "off")):
        monkeypatch.setattr(state_module, "read_state",
                            lambda _root, p=projections[name]: SimpleNamespace(projection=lambda: dict(p)))
        assert presence_admission.switch_cause(tmp_path) == expected, name


def test_building_context_never_runs_the_place_import(tmp_path):
    """The mind's fact reads the place store only once its legacy import has run elsewhere (review r3)."""
    import json

    from ouroboros.context_sections import _places_imported

    assert _places_imported(tmp_path) is False
    assert not (tmp_path / "state" / "places.json").exists()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "places.json").write_text(json.dumps({"schema_version": 1, "places": {}, "migration": {}}))
    assert _places_imported(tmp_path) is False
    (tmp_path / "state" / "places.json").write_text(
        json.dumps({"schema_version": 1, "places": {}, "migration": {"presence_bindings_v1": True}}))
    assert _places_imported(tmp_path) is True


def test_a_saved_panic_stays_named_after_a_restart(tmp_path, monkeypatch):
    """A restarted process has no stop of its own: the status names the confirmed saved source (review r3)."""
    import supervisor.state as state_module
    from ouroboros.consciousness import BackgroundConsciousness

    clock = object.__new__(BackgroundConsciousness)
    clock._drive_root, clock._stop_source = tmp_path, ""
    saved = {"bg_consciousness_enabled": False, "bg_consciousness_stop_source": "panic"}
    monkeypatch.setattr(state_module, "read_state", lambda _root: SimpleNamespace(projection=lambda: dict(saved)))
    assert clock._saved_stop_source() == "panic"
    clock._stop_source = "agent_tool"  # the agent stops an already stopped clock: the saved Panic keeps its name
    assert clock._saved_stop_source() == "panic"
    saved.pop("bg_consciousness_stop_source")
    assert clock._saved_stop_source() == "agent_tool"
    saved["bg_consciousness_stop_source"] = "owner"  # a Panic waiting to be saved outranks an older saved stop (review r4)
    monkeypatch.setattr("ouroboros.consciousness.panic_blocks_wake", lambda _root: True)
    assert clock._saved_stop_source() == "panic"


def test_the_receipt_names_no_wake_while_the_window_is_spent():
    stored = [SimpleNamespace(place_id="a", label="Lab", state="active", provider="slack")]
    spent = V.on_sentence("15:14", stored, "act", 3.0, "exhausted")
    assert spent.startswith("Consciousness on, but its allowance for the last 24 hours is spent: it wakes once there is "
                            "room in it again.") and "next wake" not in spent
    unread = V.on_sentence("15:14", stored, "act", 3.0, "allowance_unknown")
    assert unread.startswith("Consciousness on, but its spending could not be read") and "next wake" not in unread


def test_the_endpoint_index_names_the_switch_query():
    from ouroboros.gateway.endpoint_index import HTTP_SHAPES

    assert HTTP_SHAPES["GET /api/consciousness"].query == "switch"

