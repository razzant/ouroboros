"""A wake keeps its start floor while its fallback form carries my human's words whole: when that projection is
physically unfit too, the first build delivers the bare pointer its producer bound, the plan keeps the whole input,
a round boundary measures the projection as well before it swaps one in, a route that holds more gets it back, and
every delivery record names the form the request carried.

Built on the review's reproduction (RV1W Astra F3): the same memory, window and words that refused Max and Low
before now start; an unreadable source still rescues nothing.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ouroboros.context_input_selection import wake_input_checkpoint, wake_projection_inputs, whole_wake_input
from tests.test_consciousness_wake import T0, _chat, _observe, _owner
from tests.test_wake_observation_fit import FULL, WAKE, _ctx, _first_build, _record_events, _start_wake

LARGE = "An ordinary paragraph of the owner's message. " * 600


def _large_owner_wake(tmp_path, monkeypatch):
    from ouroboros.consciousness_wake import bind_wake_observation

    env, memory, task, _observation, registry = _start_wake(tmp_path, monkeypatch, lines=0)
    _chat(env.drive_root, *[_owner(LARGE + str(index), T0 - 100 + index, cmid=f"owner-{index}") for index in range(20)])
    observation = _observe(env.drive_root)

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
    assert form(plain, plain["projection_text"]) == "pointer", "a projection with no words of mine is the bare pointer"
    assert wake_input_checkpoint(bound, None, round_idx=1, after_overflow=False)["checkpoint_kind"] == "wake_input_inline"
