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
    assert "Its places will be heard, not answered." in unread["lines"] and unread["places"] == []
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
    assert V.off_sentence(stored) == ("Consciousness off: Ouroboros acts only when you write to it. 5 places are heard, "
                                      "not answered, until you turn it on.")
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
    ("active", "on", "observe", "exhausted", "allowance"),  # nothing is read before the window resets
    ("silent", "on", "observe", "available", None),  # Silent already holds: its own word says it
    ("active", "on", "full", "available", None),  # it speaks: not quiet
    ("paused", "off", "full", "available", None),  # Paused says Paused
    ("active", "unknown", "full", "available", None),  # an unknown switch is no claim
])
def test_a_place_is_quiet_for_one_cause_first_match_wins(state, switch, autonomy, allowance, cause):
    quiet = V.place_quiet(state, switch=switch, autonomy=autonomy, allowance=allowance,
                          resets_at="2026-11-17T16:20:00+00:00")
    assert (quiet or {}).get("cause") == cause
    if quiet:
        assert quiet["word"] and quiet["note"] and quiet["door"] in ("consciousness", "costs")


def test_the_quiet_note_says_what_it_does_here_now_and_until_when():
    off = V.place_quiet("silent", switch="off")
    assert "answers nothing until you turn it on" in off["note"] and off["word"] == "Hears only · consciousness off"
    assert "waits for you in its conversation here" in V.place_quiet("active", switch="on", autonomy="observe")["note"]
    spent = V.place_quiet("active", switch="on", allowance="exhausted", resets_at="")
    assert "until its reset" in spent["note"] and spent["door"] == "costs"


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
    assert people == {"by_place": {"p1": 2, "p2": 1, "p3": 0}, "total": 2, "capped": False}


def test_a_turn_off_says_what_it_ends_at_once_only_when_something_runs():
    running = {"live_wake_task_id": "w1", "live_wake_held": False, "tasks_running": 2}
    lines = V.switch_preview("off", PLACES, running=running)["lines"]
    assert "The wake running now and 2 own tasks stop at once." in lines
    assert "The wake running now stops at once." in V.switch_preview("off", PLACES, running={**running, "tasks_running": 0})["lines"]
    assert "Its 1 own task stop at once." not in V.switch_preview("off", PLACES, running={"tasks_running": 1})["lines"]
    assert "Its 1 own task stops at once." in V.switch_preview("off", PLACES, running={"tasks_running": 1})["lines"]
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
