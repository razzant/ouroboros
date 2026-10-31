"""Rendered Presence context distinguishes conversation routes without new authority."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from ouroboros.presence_context import build_presence_context_section


def _value(*, provider="custom", current_room="direct-room", current_thread="reply-thread",
           origin_room="*", proactive_room="configured-room"):
    return {
        "transport_skill": "custom-transport",
        "behavior_skill": "community-profile",
        "profile_fingerprint": "a" * 64,
        "instructions": "Use the selected tools with judgment.",
        "context_topics": [],
        "event": {
            "source_event_id": "event-1", "provider": provider, "account_id": "account-1",
            "conversation_id": current_room, "thread_id": current_thread,
            "conversation_key": f"{provider}:account-1:{current_room}:{current_thread}",
            "actor": {"platform_actor_id": "person-7", "display_name": "A correspondent"},
            "conversation": {"title": "Current conversation"},
            "message": {"message_id": "message-9", "reply_to_message_id": "message-8"},
            "origin": {"transport": provider, "account_id": "account-1",
                       "conversation_id": origin_room, "thread_id": ""},
            "destination": {"transport": provider, "account_id": "account-1",
                            "conversation_id": proactive_room, "thread_id": "proactive-thread"},
        },
    }


def _render(tmp_path, value):
    section = build_presence_context_section(tmp_path, value)
    raw = section.split("## Current presence event (host-authored facts)\n\n", 1)[1]
    return json.JSONDecoder().raw_decode(raw)[0], section


@pytest.mark.parametrize("provider", ["telegram", "slack", "email", "custom"])
def test_current_route_never_uses_wildcard_or_proactive_room(tmp_path, provider):
    value = _value(provider=provider)
    before = deepcopy(value)
    rendered, _ = _render(tmp_path, value)
    communication = rendered["communication"]

    assert communication["current_reply_route"] == {
        "provider": provider, "account_id": "account-1",
        "conversation_id": "direct-room", "thread_id": "reply-thread",
    }
    assert communication["binding_origin_filter"] == value["event"]["origin"]
    assert communication["proactive_destination"] == value["event"]["destination"]
    assert communication["transport_skill"] == value["transport_skill"]
    # Existing authority argument-binding paths retain their original values.
    assert rendered["event"] == before["event"]
    assert value == before


def test_proactive_cycle_projects_its_actual_target_without_claiming_a_send(tmp_path):
    value = _value(current_room="configured-room", current_thread="proactive-thread")
    value["event"]["actor"] = {"id": "ouroboros", "kind": "proactive_initiation"}
    value["event"]["conversation"] = {"kind": "configured_presence_destination"}
    value["event"]["message"] = {"kind": "proactive_initiation"}
    rendered, _ = _render(tmp_path, value)

    assert rendered["communication"]["current_reply_route"]["conversation_id"] == "configured-room"
    assert rendered["communication"]["current_reply_route"]["thread_id"] == "proactive-thread"
    assert rendered["event"]["actor"] == value["event"]["actor"]
    assert "delivered" not in rendered["communication"]


@pytest.mark.parametrize("marker", [None, "management_group"])
def test_room_marker_stays_a_fact_not_an_owner_or_tool_grant(tmp_path, marker):
    value = _value()
    if marker:
        value["event"]["conversation"]["configured_room"] = marker
    value["event"]["actor"]["claimed_role"] = "owner"
    original = deepcopy(value)
    rendered, _ = _render(tmp_path, value)

    assert rendered["event"] == original["event"]
    assert rendered["communication"]["current_reply_route"]["conversation_id"] == "direct-room"
    assert set(rendered["communication"]) == {
        "transport_skill", "current_reply_route", "binding_origin_filter",
        "proactive_destination", "route_meanings", "speaking_during_work",
    }
    assert "is_owner" not in rendered and "capability_ceiling" not in rendered
    assert "not proof of system ownership" in rendered["communication"]["route_meanings"]


def test_missing_route_fact_is_not_filled_from_proactive_binding(tmp_path):
    value = _value()
    del value["event"]["conversation_id"]
    del value["transport_skill"]
    rendered, _ = _render(tmp_path, value)

    assert rendered["communication"]["current_reply_route"]["conversation_id"] is None
    assert rendered["communication"]["transport_skill"] is None
    assert rendered["communication"]["proactive_destination"]["conversation_id"] == "configured-room"


def test_early_reply_guidance_preserves_profile_topics_and_completion(tmp_path):
    value = _value()
    value["context_topics"] = ["context-notes"]
    topic = tmp_path / "memory" / "knowledge" / "context-notes.md"
    topic.parent.mkdir(parents=True)
    topic.write_text("Current working knowledge.", encoding="utf-8")
    rendered, section = _render(tmp_path, value)
    guidance = rendered["communication"]["speaking_during_work"]

    assert "what your participation adds" in guidance
    assert "social warmth do not require a mention" in guidance
    assert "Observation or private consideration may stay silent" in guidance
    assert "When you undertake long work that calls for a response here" in guidance
    assert "available selected transport send tool" in guidance
    assert "then continue the work" in guidance
    assert "queued is not delivered" in guidance
    assert "early acknowledgement is not the final result" in guidance
    assert "finish_task" in rendered["completion"]
    assert value["instructions"] in section and "Current working knowledge." in section
    assert rendered["profile"]["profile_fingerprint"] == "a" * 64


def test_ordinary_task_without_presence_still_has_no_presence_section(tmp_path):
    assert build_presence_context_section(tmp_path, None) == ""
    assert build_presence_context_section(tmp_path, {"instructions": "ordinary"}) == ""
