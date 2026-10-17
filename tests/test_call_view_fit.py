"""A live call's view: the ordinary fit against the realtime route, with every host section in its operative form.

A live session's window always fills, so in a call every host section that has an operative
view enters in that form from the first open; tier zero stays whole, the call starts in Nano,
and the other live rooms are one more such section. A text turn with the same capture and window keeps every section whole.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as fit, memory_view
from ouroboros.section_floor import HostSection, capture_sections
from tests import _memory_view_synthetic as syn

RESERVE = 8_192
WINDOW = 131_072


@pytest.fixture(autouse=True)
def capacity_facts(monkeypatch):
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 65_536)
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density",
                        lambda *_a: (1.0, "cold_estimate"))


def _service(name, size=2_000):
    return HostSection(name, f"## {name}\n\n" + "observed details " * size, f"## {name}\n\nOperative state remains.")


def _core(root, *, snapshot=None, dynamic=(), head=(), semi=()):
    semi_parts = [HostSection("identity", "## Identity\n\n" + "WHOLE IDENTITY " * 50), *semi]
    return fit.ContextCore(base_prompt="SYSTEM WHOLE", bible_md="BIBLE WHOLE", architecture_md="# Architecture",
        development_md="# Development", docs_need_development=False,
        semi_stable_text="\n\n".join(semi_parts), semi_stable_sections=capture_sections(semi_parts, prefix="semi"),
        dynamic_text="\n\n".join(dynamic), dynamic_head_text="\n\n".join(head),
        user_content_json=json.dumps("[Live call] A live voice call is open in this room."),
        dynamic_sections=capture_sections(dynamic, prefix="dynamic"),
        dynamic_head_sections=capture_sections(head, prefix="head"),
        section_root=str(root), section_task_id="call-task",
        memory_view_json=memory_view.snapshot_json(snapshot) if snapshot else "")


def _call(core, *, window=WINDOW, preferred="max", **kwargs):
    route = {"provider": "google", "model": "gemini-3.8-live", "route_fp": "realtime-fp"}
    resolver = lambda *_a, **_kw: pytest.fail("a call never resolves Main's route")  # noqa: E731
    return fit.build_context_fit_plan(SimpleNamespace(drive_root=Path(core.section_root)), core, {"id": "call-task"},
        preferred_mode=preferred, route_resolver=resolver, known_window_tokens=window,
        known_output_reserve_tokens=RESERVE, known_route=route, call_view=True, **kwargs)


def _text_turn(core, *, window=WINDOW, preferred="max"):
    return fit.build_context_fit_plan(SimpleNamespace(drive_root=Path(core.section_root)), core, {"id": "call-task"},
        preferred_mode=preferred, known_window_tokens=window,
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "main"}, SimpleNamespace(
            route_fp="main-fp", status="confirmed", stale=False, window_tokens=window, source="test")))


def _system(plan, mode=None):
    return "\n".join(block["text"] for block in plan.messages_for(mode or plan.initial_mode)[0]["content"])


def _addressed(plan, mode=None):
    return {row["section_id"]: row for row in
            plan.projection(mode or plan.initial_mode).memory_facts.get("host_sections_addressed", [])}


def _snapshot():
    live = [syn.live_room(i, notes=1) for i in range(3)]
    for item in live:
        item["notes"][0]["text"] = "My note about what this room still waits for. " * 20
    return syn.snapshot(room_facts=syn.room("1", lane1=[syn.spoken(1, "human", 600)]), story=syn.pages(2, 800),
                        live=live, mark_count=3)


def test_every_operative_section_enters_in_that_form_from_the_first_open(tmp_path):
    health, runtime, world = _service("health_invariants"), _service("runtime"), _service("world")
    core = _core(tmp_path, snapshot=_snapshot(), dynamic=[health, runtime, "protected dynamic text"],
                 head=[HostSection("knowledge_index", "## Knowledge base\n\nWHOLE INDEX")], semi=[world])
    call = _call(core)
    assert call.call_view and call.initial_mode == "nano"
    text = _system(call)
    assert set(_addressed(call)) == {"health_invariants", "runtime", "world", "live_rooms"}
    for section in (health, runtime, world):
        assert section.operative_view in text and section.full_text not in text
        row = _addressed(call)[section.section_id]
        assert (tmp_path / row["path"]).read_text(encoding="utf-8") == section.full_text  # the pointer reads back
        assert row["reader"] in text
    # The marker names the call's view as the cause, never a physical window that did not ask for it.
    assert "omitted from this call's view" in text and "omitted for the physical window" not in text
    # Tier zero and unviewed parts stay whole; the window alone would not have asked for any of this.
    for whole in ("SYSTEM WHOLE", "BIBLE WHOLE", "WHOLE IDENTITY " * 50, "WHOLE INDEX", "protected dynamic text"):
        assert whole in text
    turn = _text_turn(core)
    assert not _addressed(turn) and turn.initial_mode == "max"
    assert health.full_text in _system(turn) and runtime.full_text in _system(turn)


def test_under_window_pressure_a_text_turn_relocates_service_blocks_and_keeps_its_stable_blocks(tmp_path):
    health, runtime, world = _service("health_invariants"), _service("runtime"), _service("world")
    core = _core(tmp_path, dynamic=[health, runtime], semi=[world])
    window = 65_536 + 4_000  # Main's reserve plus less than any one block whole
    turn = _text_turn(core, window=window)
    assert set(_addressed(turn)) & {"health_invariants", "runtime"}  # the window asked the host floor
    assert "world" not in _addressed(turn) and world.full_text in _system(turn)  # a text turn's stable block stays
    assert "world" in _addressed(_call(core, window=window))  # a call addresses it from its first request


def test_the_other_live_rooms_keep_names_and_state_and_their_notes_are_one_read_away(tmp_path):
    snapshot = _snapshot()
    call, turn = _call(_core(tmp_path, snapshot=snapshot)), _text_turn(_core(tmp_path, snapshot=snapshot))
    note = snapshot.live_rooms[0]["notes"][0]["text"]
    assert note in _system(turn) and note not in _system(call)
    for item in snapshot.live_rooms:
        assert f"### {item['label']} — " in _system(call) and f"memory_read(room_id='{item['room_id']}'" in _system(call)
    row = _addressed(call)["live_rooms"]
    retained = (tmp_path / row["path"]).read_text(encoding="utf-8")
    assert retained.startswith("## Live rooms") and note in retained
    assert call.projection("nano").memory_facts["blocks"]["live_rooms"] == 3
    # The marks I keep in view and this room are untouched by the move.
    for mark in snapshot.marks:
        assert mark["text"] in _system(call)
    assert "## This room (Main)" in _system(call)


def test_the_call_is_fitted_to_the_realtime_routes_identity_and_window(tmp_path, monkeypatch):
    witnessed = []
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda root, fp, model: witnessed.append((fp, model)) or 1.25)
    call = _call(_core(tmp_path, snapshot=_snapshot()))
    assert witnessed == [("realtime-fp", "gemini-3.8-live")]
    assert (call.route_fp, call.model, call.provider, call.model_role) == ("realtime-fp", "gemini-3.8-live",
                                                                           "google", "live_call")
    assert call.window_tokens == WINDOW and call.output_reserve_tokens == RESERVE
    assert call.projection("nano").calibration_ratio == 1.25
    assert call.projection("nano").memory_facts["floor"]["window_tokens"] == WINDOW


def test_a_call_view_reprojection_keeps_the_operative_forms(tmp_path):
    core = _core(tmp_path, snapshot=_snapshot(), dynamic=[_service("health_invariants")], semi=[_service("world")])
    call = _call(core)
    again = call.reproject_for_route(window_tokens=WINDOW, known_window=True, ratio=1.0, output_reserve=RESERVE,
                                     tool_schemas=[], start_mode="nano", refused_tokens=1)
    assert set(_addressed(again, "nano")) >= set(_addressed(call, "nano"))
    assert "observed details" not in _system(again, "nano")


def test_parts_render_byte_identical_to_the_joined_texts_when_nothing_is_relocated(tmp_path):
    core = _core(tmp_path, snapshot=_snapshot(), dynamic=[_service("runtime"), ""], head=["head text", ""],
                 semi=[_service("world"), ""])
    legacy = dataclasses.replace(core, semi_stable_sections=(), dynamic_sections=(), dynamic_head_sections=())
    for preferred in ("max", "low", "nano"):
        selected, baseline = _text_turn(core, window=2_000_000, preferred=preferred), \
            _text_turn(legacy, window=2_000_000, preferred=preferred)
        for mode in ("max", "low", "nano"):
            assert selected.projection(mode).system_content_json == baseline.projection(mode).system_content_json


def test_a_calls_selected_schemas_count_in_its_fit_as_sent(tmp_path, monkeypatch):
    from ouroboros import memory_floor

    seen = {}
    original = memory_floor.mode_views

    def spy(snapshot, **kwargs):
        seen.update(fixed=dict(kwargs["fixed_tokens_by_mode"]), tools=kwargs["tool_names"])
        return original(snapshot, **kwargs)

    monkeypatch.setattr(memory_floor, "mode_views", spy)
    read_file = {"type": "function", "function": {"name": "read_file", "description": "Read a retained file. " * 20,
                                                  "parameters": {"type": "object", "properties": {}}}}
    _call(_core(tmp_path, snapshot=_snapshot()), tool_schemas=[read_file])
    with_schema = dict(seen)
    _call(_core(tmp_path, snapshot=_snapshot()), tool_schemas=[])
    # The driver sends every schema the call selected, so the Nano view is fitted beside all of them.
    assert "read_file" in with_schema["tools"]["nano"]
    assert with_schema["fixed"]["nano"] > seen["fixed"]["nano"]


def test_a_first_call_counts_in_its_routes_cold_density_and_a_witness_replaces_it(tmp_path, monkeypatch):
    route = {"provider": "google", "model": "gemini-3.8-live", "route_fp": "realtime-fp", "cold_density": 1.25}

    def call():
        return fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), _core(tmp_path), {"id": "call-task"},
            preferred_mode="max", route_resolver=lambda *_a, **_kw: pytest.fail("never Main's route"),
            known_window_tokens=WINDOW, known_output_reserve_tokens=RESERVE, known_route=route, call_view=True)

    assert call().projection("nano").calibration_ratio == 1.25  # cold store: the route's documented density
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density",
                        lambda *_a: (1.12, "fresh_route_usage"))
    assert call().projection("nano").calibration_ratio == 1.12  # the route's own witness wins
