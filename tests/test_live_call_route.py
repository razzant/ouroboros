"""The realtime route of a live call (Gemini Live): its documented facts, the plan the fit receives, the first tools.

Calls run on Gemini Live only. The fit measures against the realtime route itself: its
input limit (131,072) and a speech-sized reply allowance (8,192), never the provider's
65,536 output maximum, which the first release subtracted from the window.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import live_call_route
from ouroboros.call_protocol import CallSpec
from ouroboros.gateways.realtime_engine import ENGINE_GOOGLE


def _spec(**overrides) -> CallSpec:
    fields = dict(call_id="call-1", chat_id=1, engine=ENGINE_GOOGLE, model="gemini-3.8-live", voice="Kore",
                  opened_by="owner", presence_binding_id="", max_minutes=0, label="Test call", client_message_id="cm-1")
    fields.update(overrides)
    return CallSpec(**fields)


def _schema(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}


def test_the_route_window_is_recorded_as_sourced_evidence_before_the_fit(tmp_path):
    route = live_call_route.resolve_call_route(_spec(), tmp_path)
    assert route.window_tokens == 131_072 and route.fact.session_cap_minutes is None
    assert (route.engine.input_rate, route.engine.output_rate) == (16_000, 24_000)  # the Gemini Live wire
    assert route.fact.reply_allowance_tokens == 8_192 and route.fact.max_output_tokens == 65_536
    evidence = live_call_route.window_evidence(tmp_path, _spec())
    # A shipped documented fact is asserted evidence with its source and read date,
    # never metadata the provider confirmed on the wire.
    assert evidence["window_tokens"] == 131_072 and evidence["status"] == "asserted"
    assert evidence["source"] == "documented_route_fact" and evidence["response_limit"]["source"] == "documented_route_fact"
    assert "ai.google.dev" in evidence["detail"] and "read 2026-10-16" in evidence["detail"]
    assert evidence["response_limit"]["max_output_tokens"] == 65_536  # recorded as documentation only
    unknown = live_call_route.resolve_call_route(_spec(model="gemini-live-future"), tmp_path)
    assert unknown.fact is None and unknown.window_tokens is None
    assert live_call_route.window_evidence(tmp_path, _spec(model="gemini-live-future")) is None


def test_call_context_plan_binds_the_fit_to_the_realtime_route_and_the_call_view(monkeypatch, tmp_path):
    from ouroboros.capability_evidence import route_fingerprint

    monkeypatch.setattr(live_call_route, "initial_call_schemas", lambda tools, spec=None: [_schema("web_search")])
    assert live_call_route.call_context_plan({"id": "t", "metadata": {}}, None, tmp_path) is None
    task = {"id": "t", "metadata": {"live_call": _spec().as_metadata()}}
    plan = live_call_route.call_context_plan(task, None, tmp_path)
    fingerprint = route_fingerprint(provider=ENGINE_GOOGLE, model="gemini-3.8-live")
    # The realtime route's own input limit and speech-sized reply allowance, never its 65,536 output
    # maximum (v1 passed that and fitted Gemini into half its window), and its own identity.
    assert plan.fit_kwargs == {"known_window_tokens": 131_072, "known_output_reserve_tokens": 8_192, "call_view": True,
                               "known_route": {"provider": ENGINE_GOOGLE, "model": "gemini-3.8-live",
                                               "route_fp": fingerprint, "cold_density": 1.25}}
    assert [s["function"]["name"] for s in plan.tool_schemas] == ["web_search"]
    unknown = {"id": "t", "metadata": {"live_call": _spec(model="gemini-live-future").as_metadata()}}
    without = live_call_route.call_context_plan(unknown, None, tmp_path)
    assert "known_window_tokens" not in without.fit_kwargs and without.fit_kwargs["call_view"] is True
    assert without.fit_kwargs["known_output_reserve_tokens"] == 8_192 and without.spec.call_id == "call-1"


@pytest.mark.parametrize("raw, expected", [("1", True), ("off", False), ("", True)])
def test_native_search_follows_the_owners_setting(monkeypatch, raw, expected):
    monkeypatch.setenv("OUROBOROS_CALL_NATIVE_SEARCH", raw)
    assert live_call_route.native_search_enabled(_spec()) is expected


def test_the_first_tool_view_reads_its_pointers_and_hands_long_work_to_a_task():
    from ouroboros.tool_capabilities import COGNITIVE_MEMORY_TOOL_NAMES

    # Every retained host section in the call view is a runtime_data read, and a Gemini session
    # reopens to gain a tool: read_file must be there from the first open.
    assert set(live_call_route.CALL_INITIAL_TOOL_NAMES) == set(COGNITIVE_MEMORY_TOOL_NAMES) | {
        "web_search", "send_user_message", "read_file", "start_task"}


def test_the_first_tool_view_names_exist_in_the_registry(tmp_path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "data")
    registry.set_context(ToolContext(repo_dir=repo, drive_root=tmp_path / "data", task_id="t"))
    names = {schema["function"]["name"] for schema in live_call_route.initial_call_schemas(registry, _spec())}
    assert {"read_file", "start_task", "knowledge_read", "memory_read"} <= names


def test_a_presence_call_opens_with_its_whole_ceiling_and_an_owner_call_with_the_nano_view():
    registry = SimpleNamespace(schemas=lambda: [_schema("notes_read"), _schema("web_search"), _schema("enable_tools")])
    owner = [s["function"]["name"] for s in live_call_route.initial_call_schemas(registry, _spec())]
    presence = [s["function"]["name"] for s in live_call_route.initial_call_schemas(
        registry, _spec(presence_binding_id="b" * 32))]
    assert "notes_read" not in owner and {"web_search", "enable_tools"} <= set(owner)
    assert presence == ["notes_read", "web_search", "enable_tools"]


def test_the_first_releases_call_keys_are_retired_where_settings_are_read():
    from ouroboros.config import normalize_settings_raw

    gone = {"OUROBOROS_CALL_ENGINE": "openai", "OUROBOROS_CALL_MODEL_OPENAI": "gpt-realtime-2.1",
            "OUROBOROS_CALL_VOICE_OPENAI": "marin"}
    read = normalize_settings_raw({**gone, "OUROBOROS_CALL_MODEL_GOOGLE": "gemini-3.8-live"})
    assert not set(gone) & set(read) and read["OUROBOROS_CALL_MODEL_GOOGLE"] == "gemini-3.8-live"
