"""A wake keeps its start floor while its fallback form carries my human's card answers whole: when that projection
is physically unfit too, the first build delivers the bare pointer its producer bound, the plan keeps the whole input,
a round boundary measures the projection as well before it swaps one in, a route that holds more gets it back, and
every delivery record names the form the request carried. A wake's text counts my human's messages (my view carries
them: ``## This room``, ``## Other conversations``) and lists place messages whole; the source keeps each line.

Built on the review's reproduction (RV1W Astra F3): the same memory, window and words that refused Max and Low
before now start; an unreadable source still rescues nothing.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from ouroboros.context_input_selection import wake_input_checkpoint, wake_projection_inputs, whole_wake_input
from tests.test_consciousness_wake import T0, _chat, _iso, _observe, _owner
from tests.test_wake_observation_fit import FULL, WAKE, _ctx, _first_build, _record_events, _start_wake

LARGE = "An ordinary paragraph of the owner's answer. " * 600


def _card(index):
    """My human's answer to a card, in their own words: a wake's text and its projection carry it whole."""
    block = {"quiz_id": f"q{index}", "state": "answered", "asked_at": _iso(T0 - 200 + index),
             "answered_at": _iso(T0 - 100 + index), "options": ["A", "B"], "comment": LARGE + str(index),
             "question": "Which way?"}
    return {"ts": _iso(T0 - 100 + index), "direction": "system", "type": "quiz_answer", "chat_id": 1,
            "task_id": f"asker{index}", "quiz": block}


def _large_owner_wake(tmp_path, monkeypatch):
    from ouroboros.consciousness_wake import bind_wake_observation

    env, memory, task, _observation, registry = _start_wake(tmp_path, monkeypatch, lines=0)
    _chat(env.drive_root, *[_card(index) for index in range(20)])
    observation = _observe(env.drive_root)
    assert observation.counts() == {"card_answer": 20}

    def render(events):
        return "[Wake-up] " + events

    task["text"] = render(observation.full_text())
    bind_wake_observation(env.drive_root, task, observation, render)
    registry._ctx.task_metadata = dict(task["metadata"])
    return env, memory, task, registry


def _delivery_records(env):
    path = env.drive_root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [(row.get("checkpoint_kind"), row.get("round"), row.get("form")) for row in rows
            if str(row.get("checkpoint_kind") or "").startswith("wake_input_")]


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_a_large_owner_window_still_starts_on_the_bare_pointer(tmp_path, monkeypatch, mode):
    env, memory, task, registry = _large_owner_wake(tmp_path, monkeypatch)
    wake = task["metadata"]["wake_observation"]
    assert LARGE in wake["projection_text"] and LARGE not in wake["pointer_text"], "my human's words ride the projection"
    messages, _capture = _first_build(env, memory, task, registry, monkeypatch, mode)
    assert messages[1] == {"role": "user", "content": wake["pointer_text"]}, "the projection proved unfit too"
    assert json.loads(registry._ctx.context_fit_plan.user_content_json) == task["text"], "the plan keeps the whole input"
    assert _delivery_records(env) == [("wake_input_by_source", 0, "pointer")]


@pytest.mark.parametrize("mode", ["max", "nano"])
def test_an_unreadable_source_rescues_neither_form(tmp_path, monkeypatch, mode):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.context_fit import MinimumViewUnfit

    env, memory, task, registry = _large_owner_wake(tmp_path, monkeypatch)
    wake = task["metadata"]["wake_observation"]
    assert len(wake_projection_inputs(task)) == 2
    stored = task_artifact_dir_path(env.drive_root, task["id"], create=False) / wake["source"]["path"]
    stored.write_bytes(b"{}\n" + stored.read_bytes()[3:])  # the same size, other bytes
    assert wake_projection_inputs(task) == []
    if mode == "nano":
        messages, _capture = _first_build(env, memory, task, registry, monkeypatch, mode)
        assert messages[1]["content"].startswith("[Exact task input source]"), "owner-Nano keeps its own last rescue"
    else:
        with pytest.raises(MinimumViewUnfit):
            _first_build(env, memory, task, registry, monkeypatch, mode)
    assert _delivery_records(env) == []


def test_a_round_after_a_pointer_start_never_swaps_an_unfit_projection_back(tmp_path, monkeypatch):
    """The round boundary measures the projection as it measures the whole input: on the route that refused both
    the bare pointer stays, a route that holds my human's words gets them back, one that holds everything the whole
    input, each with its form named."""
    from ouroboros.loop_model_call import _project_wake_input

    events = _record_events(monkeypatch)
    words = "My human's words, whole: " + "w" * 260_000  # ~65K tokens beside the 65K answer reserve
    pointer = "Wake facts: the complete observation is stored at <source>."
    whole, projection = FULL + "\n" + words, pointer + " " + words
    ctx, _ = _ctx(tmp_path, window=120_000, text=pointer,
                  wake={**WAKE, "projection_text": projection, "pointer_text": pointer})
    ctx.context_fit_plan = replace(ctx.context_fit_plan, user_content_json=json.dumps(whole))
    assert _project_wake_input(ctx) is False and ctx.messages[1]["content"] == pointer
    ctx.context_fit_plan = replace(ctx.context_fit_plan, window_tokens=180_000)
    assert _project_wake_input(ctx) is True and ctx.messages[1]["content"] == projection
    ctx.context_fit_plan = replace(ctx.context_fit_plan, window_tokens=500_000)
    assert _project_wake_input(ctx) is True and ctx.messages[1]["content"] == whole
    assert [(event["checkpoint_kind"], event["form"]) for event in events] == [
        ("wake_input_by_source", "projection"), ("wake_input_inline", "whole")]


def test_a_pointer_start_returns_to_the_whole_input_on_a_route_that_holds_it():
    wake = {"projection_text": "pointer + my human's words", "pointer_text": "bare pointer"}
    plan = type("Plan", (), {"user_content_json": json.dumps("the whole observation")})()
    for shown in (wake["projection_text"], wake["pointer_text"]):
        assert whole_wake_input({"role": "user", "content": shown}, plan, wake)["content"] == "the whole observation"
    assert whole_wake_input({"role": "user", "content": "other"}, plan, wake)["content"] == "other"


def test_every_delivery_record_names_the_form_the_request_carried():
    bound = {"projection_text": "pointer + my human's words", "pointer_text": "bare pointer", "composition": {},
             "source": {"sha256": "s"}}
    plain = {"projection_text": "bare pointer", "composition": {}, "source": {"sha256": "s"}}

    def form(wake, content):
        return wake_input_checkpoint(wake, content, round_idx=1, after_overflow=False)["form"]

    assert form(bound, None) == "whole" and form(bound, bound["projection_text"]) == "projection"
    assert form(bound, bound["pointer_text"]) == "pointer", "nothing inline: no coverage claim applies"
    busy = {**bound, "owner_projection_text": "pointer + my human's words, the others counted"}
    assert form(busy, busy["owner_projection_text"]) == "projection", "my human's words inline here too"
    assert form(busy, busy["pointer_text"]) == "pointer"
    assert form(plain, plain["projection_text"]) == "pointer", "a projection with no words of mine is the bare pointer"
    assert wake_input_checkpoint(bound, None, round_idx=1, after_overflow=False)["checkpoint_kind"] == "wake_input_inline"


# --- RV2W: the round boundary chooses among the forms smaller than the whole input (Astra R4, Opus N3) ----------

ALL_WORDS = "My human's words, whole: " + "w" * 400_000  # ~100K tokens: almost the whole window is theirs
ALL_WHOLE = "Wake facts:\n- owner message: " + ALL_WORDS
ALL_POINTER = "Wake facts: the complete observation is stored at <source>; read the ranges you need."
ALL_PROJECTION = ALL_POINTER + "\nMy human's words in this window, whole:\n- owner message: " + ALL_WORDS


def test_a_real_owner_wake_gets_its_whole_input_back_on_a_route_that_holds_it(tmp_path, monkeypatch):
    """Astra R4: the projection of a window that is almost all my human's words is longer than the whole input; after
    a pointer start a route that holds everything still gets the whole input back."""
    from ouroboros import loop
    from ouroboros.loop_model_call import _project_wake_input

    env, _memory, task, _registry = _large_owner_wake(tmp_path, monkeypatch)
    wake = task["metadata"]["wake_observation"]
    whole, projection, pointer = task["text"], wake["projection_text"], wake["pointer_text"]
    assert len(projection) > len(whole) > len(pointer), "the projection saves nothing here"
    events = _record_events(monkeypatch)
    ctx, _ = _ctx(env.drive_root, window=1_000_000, text=whole, wake=wake)
    measured = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    assert measured.estimated_input_tokens + measured.response_reserve_tokens < measured.capacity_total_tokens
    ctx.messages[1]["content"] = pointer  # the start rescued it with the bare pointer
    ctx.context_fit_plan = replace(ctx.context_fit_plan, user_content_json=json.dumps(whole))
    assert _project_wake_input(ctx) is True and ctx.messages[1]["content"] == whole
    assert [(event["checkpoint_kind"], event["form"]) for event in events] == [("wake_input_inline", "whole")]


@pytest.mark.parametrize("route", ["smaller", "larger"])
def test_a_projection_that_saves_nothing_never_stops_the_pointer_or_the_whole_input(tmp_path, monkeypatch, route):
    """Opus N3, both edges: on a route too small for the whole input the bare pointer goes, never the longer
    projection; after a pointer start a route that holds everything gets the whole input back."""
    from ouroboros.loop_model_call import _project_wake_input

    events = _record_events(monkeypatch)
    wake = {**WAKE, "projection_text": ALL_PROJECTION, "pointer_text": ALL_POINTER}
    if route == "smaller":
        ctx, _ = _ctx(tmp_path, window=120_000, text=ALL_WHOLE, wake=wake)
        expected, form = ALL_POINTER, ("wake_input_by_source", "pointer")
    else:
        ctx, _ = _ctx(tmp_path, window=1_000_000, text=ALL_POINTER, wake=wake)
        ctx.context_fit_plan = replace(ctx.context_fit_plan, user_content_json=json.dumps(ALL_WHOLE))
        expected, form = ALL_WHOLE, ("wake_input_inline", "whole")
    assert _project_wake_input(ctx) is True and ctx.messages[1]["content"] == expected
    assert [(event["checkpoint_kind"], event["form"]) for event in events] == [form]


def test_a_busy_window_lists_the_places_messages_and_starts_with_my_humans_words_in_my_view(tmp_path, monkeypatch):
    """Opus N2's window: thousands of short messages of a participant beside one of my human's. The wake lists the
    participant's messages whole and counts my human's (the source keeps every line), so its whole input is unfit and
    the bare pointer starts; my human's words reach the first request in my view. The participant's room is a
    conversation (a recorded person spoke there) with no word of my human or reply of mine: its oldest unread words
    yield first to one host line that counts them and quotes how the latest begins, the newest stay whole."""
    from ouroboros import dialogue_store as ds, places
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.consciousness_wake import bind_wake_observation, observe_wake

    env, memory, task, _observation, registry = _start_wake(tmp_path, monkeypatch, lines=0)
    place = places.connect_place(env.drive_root, connector="bridge", provider="chat", account_id="qa", label="Team")
    words = "Please stop the deploy and keep the database as it is until I say otherwise."
    ds.append(env.drive_root, "chat", _owner(words, T0 - 3500, cmid="owner-1"))
    for index in range(6000):
        ts = _iso(T0 - 3000 + index * 0.4)
        ds.append(env.drive_root, "chat", {
            "ts": ts, "direction": "in", "chat_id": 2 ** 41 + 23, "text": f"message number {index} " + "s" * 60,
            "source": "presence:chat", "sender_label": "Sasha", "client_message_id": f"ev-sasha-{ts}",
            "transport": {"provider": "chat", "account_id": "qa", "conversation_id": "c",
                          "actor": {"platform_actor_id": "sasha", "display_name": "Sasha"}},
            "presence_provenance": {"place_id": place.place_id}, "place_id": place.place_id})
    observation = observe_wake(env.drive_root, boundary=None, since=T0 - 3600, now=T0, reason="heartbeat")

    def render(events):
        return "[Wake-up] " + events

    task["text"] = render(observation.full_text())
    bind_wake_observation(env.drive_root, task, observation, render)
    registry._ctx.task_metadata = dict(task["metadata"])
    wake = task["metadata"]["wake_observation"]
    assert "1 owner message (counted, not listed" in task["text"] and "6000 messages in places;" in task["text"]
    assert words not in task["text"] and "message number 17 " in task["text"]  # the place's messages: listed whole
    assert not {"owner_projection_text", "pointer_text"} & set(wake)  # no card answer: the projection is the pointer
    stored = read_actor_source_bytes(env.drive_root, wake["source"]["task_id"], wake["source"]).decode("utf-8")
    assert words in stored and "message number 5999 " in stored  # every line whole at the source
    messages, _capture = _first_build(env, memory, task, registry, monkeypatch, "max")
    assert messages[1]["content"] == wake["projection_text"], "the whole input is unfit: the bare pointer starts"
    view = json.dumps(messages[0], ensure_ascii=False)
    assert words in view, "my human's words reach the first request, in my view"
    # Decision 1A and commit 4: a count line, its opening bounded, for the oldest yielded words; every word is either.
    yielded = int(re.search(r"No recorded consideration: (\d+) words; \[", view)[1])
    opening = re.search(r"latest by Sasha[^\]]*\]\] «([^»]*)»", view)[1]
    assert "message number 17 " not in view and "message number 5999 " in view and 0 < yielded < 6000
    assert opening == f"message number {yielded - 1} " + "s" * 60 and len(opening) <= 160
    assert view.count("message number ") == 6000 - yielded + 1
    assert _delivery_records(env) == [("wake_input_by_source", 0, "pointer")]
