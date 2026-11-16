"""The Max room of a start: one fact beside whole books, its cause and the words that carry it."""
import dataclasses
import json
import math

import pytest

from ouroboros import capability_evidence as ce, context_fit as cf, memory_floor as mf, memory_view as mv
from ouroboros.context_view_facts import memory_facts
from ouroboros.llm_messages import project_wire_messages
from ouroboros.tool_policy import select_tool_schemas
from tests import _memory_view_synthetic as syn


def _core(*, pages=40, page_chars=4_000, books=40_000):
    snap = syn.snapshot(role="consciousness", story=syn.pages(pages, page_chars), room_facts=syn.room(lane2=3))
    snap = dataclasses.replace(snap, spec=dataclasses.replace(snap.spec, work_room_levels=2))
    return cf.ContextCore("Required core", "Constitution", "b" * books, "", "Identity", "f" * 1_000, '"Assignment"',
                          False, memory_view_json=mv.snapshot_json(snap))


def _start(core, window, monkeypatch):
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    return cf.build_context_fit_plan(None, core, {"id": "room01", "usage_category": "consciousness"},
        preferred_mode="max", route_resolver=lambda *_a, **_kw: (
            {"provider": "test", "model": "test"}, ce.CapabilityEvidence(window, "confirmed", "metadata", "room-route")))


def _fact(plan):
    return plan.projection(plan.initial_mode).memory_facts["max_room"]


def _crowded_window(held):
    """A window whose Max frame leaves half of my memory's ordinary forms: the books fit, the memory does not."""
    fixed = sum(tokens for _name, tokens in held["parts"]) + held["schemas"]
    return math.ceil(((held["story"] + held["room"]) / 2 + held["reserve"] + fixed) / (1 - held["levels"] / 8))


def test_a_roomy_window_holds_max_and_names_the_measured_parts(monkeypatch):
    plan = _start(_core(), 2_000_000, monkeypatch)
    fact = _fact(plan)
    assert plan.initial_mode == "max" and fact["cause"] == "held" and fact["cuts"] == []
    assert fact["headroom"] == fact["window"] - fact["need"] > 0 and "short" not in fact
    assert fact["levels"] == 2 and fact["window"] == 2_000_000 and fact["floor_need"] <= fact["zero_need"] < fact["need"]
    assert fact["unfolded"] and fact["unfolded"] <= fact["story"]  # 40 pages, none in a part or account yet
    assert dict(fact["parts"])["books"] > dict(fact["parts"])["knowledge, scratchpad and runtime facts"] and "tool schemas" not in dict(fact["parts"])
    line = memory_facts(plan, "max")  # every start, Max held too: the measure part by part, no advice
    assert line.startswith("Max room in this captured build: window 2,000,000; books, core, knowledge, scratchpad, runtime facts, "
                           "connections, assignment, reply reserve and work room")
    assert f"(books {dict(fact['parts'])['books']:,}, " in line and f"tool schemas {fact['schemas']:,}; " in line
    assert f"story {fact['story']:,} (40 pages no part or account holds yet, {fact['unfolded']:,} tokens, most in " in line
    assert f"beside a Max room of {fact['allowance']:,}; Max held;" in line
    assert "; marks in view 0" in line and "drafts awaiting" not in line and "shown only by address" not in line


def test_memory_or_schemas_that_take_the_books_room_are_measured_without_advice(monkeypatch):
    core = _core()
    held = _fact(_start(core, 2_000_000, monkeypatch))
    plan = _start(core, _crowded_window(held), monkeypatch)
    fact = _fact(plan)
    assert plan.initial_mode == "low" and fact["cause"] == "crowded" and fact["cuts"] == []
    assert fact["floor_need"] <= fact["window"] and fact["short"] == fact["need"] - fact["window"] > 0
    shown = plan.low_projection.system_content_json
    assert "Low with chapter maps. The host's measure of this start: window " in shown and "b" * 40_000 not in shown
    assert not any(words in shown for words in ("may bring the books back", "Folding memory"))
    assert f"(pages no part or account holds yet {fact['unfolded']:,})" in shown
    assert not any(words in shown for words in ("failure", "crowds the books out", "cannot make room", "did not fit beside"))
    line = memory_facts(plan, "low")
    assert f"beside a Max room of {fact['allowance']:,}; Low; " in line
    assert not any(words in line for words in ("may bring", "Folding", "failure"))
    # The fact rides in the receipt the task trace keeps, and a selection keeps the start's frame.
    assert mf.trace_facts(plan.low_projection.memory_facts)["max_room"] == fact
    selected = mf.select_view(plan, "low", [])
    assert selected.low_projection.memory_facts["max_room"] == fact
    assert "The host's measure of this start" in selected.low_projection.system_content_json


def test_reprojected_restore_names_its_captured_build_not_an_unknown_original_start(monkeypatch):
    built = _start(_core(), 2_000_000, monkeypatch)
    restored = cf.restored_start(built, None)
    moved = restored.reproject_for_route(window_tokens=128_600, known_window=True, ratio=1.0,
        output_reserve=built.output_reserve_tokens, tool_schemas=[], start_mode=restored.initial_mode, admitted=True)
    assert moved.start == cf.UNKNOWN_START
    assert _fact(moved) == _fact(built)  # the retained room belongs to the capture, not the new route
    assert moved.window_tokens == 128_600
    text = memory_facts(moved, moved.initial_mode)
    assert "Max room in this captured build: window 2,000,000" in text
    assert "Max room at the start" not in text


def test_a_route_whose_window_cannot_hold_the_books_floor_is_a_disclosed_fact(monkeypatch):
    plan = _start(_core(pages=1, page_chars=100, books=900_000), 200_000, monkeypatch)
    fact = _fact(plan)
    assert plan.initial_mode == "low" and fact["cause"] == "route" and fact["floor_need"] > fact["window"]
    assert fact["parts"][0][0] == "books"  # largest first
    shown = plan.low_projection.system_content_json
    assert ("Low with chapter maps, a disclosed fact of this route: its window cannot hold the books beside my core, "
            f"assignment, reply reserve and the work's room (needs {fact['floor_need']:,} > 200,000: books ") in shown
    assert "folding memory or dropping tool schemas cannot make room here." in shown and "may bring" not in shown
    line = memory_facts(plan, "low")
    assert "; Low, a disclosed fact of this route: folding memory or dropping tool schemas cannot make room here" in line


def test_a_directly_required_book_at_the_max_boundary_never_lowers_the_start(monkeypatch):
    """The first window holding this start's Max, where the required handbook's own room note once chose Low alone
    (Astra's counterexample): the book joins that Max with its room told after every choice, never inside one."""
    core = dataclasses.replace(_core(), development_md="# Development\n\n## Review\n\n" + "rule " * 2_000)
    lo, hi = 100_000, 2_000_000
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            lo, hi = (lo, mid) if _start(core, mid, monkeypatch).initial_mode == "max" else (mid + 1, hi)
        except cf.MinimumViewUnfit:
            lo = mid + 1
    plain, held = (_start(dataclasses.replace(core, development_required=flag), hi, monkeypatch) for flag in (False, True))
    assert plain.initial_mode == held.initial_mode == "max"
    assert held.max_projection.memory_facts["floor"] == plain.max_projection.memory_facts["floor"]
    blocks = [block["text"] for block in json.loads(held.max_projection.system_content_json)]
    assert any(text.startswith("## DEVELOPMENT.md\n") for text in blocks)
    assert "Start room is smaller than requested: this task's directly required handbook takes" in blocks[-1]


WIDE = [{"type": "function", "function": {"name": "read_file", "description": "Read a fixture.", "parameters": {
    "type": "object", "properties": {f"field_{i}": {"type": "string", "description": "A fixture value."} for i in range(4000)}}}}]
# Small schemas a custom catalog can carry (``openai_chat_custom``: one this wide exceeds its description limit).
SMALL = [{"type": "function", "function": {"name": f"tool_{i}", "description": "Do one small fixture thing.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string", "description": "A path."}}, "required": ["path"]}}}
    for i in range(60)]


def _wide_start(window, monkeypatch, *, provider="test", effort="", tools=WIDE):
    """A start whose call schemas outweigh its memory: one schema of 4,000 fields beside a whole small book."""
    snap = syn.snapshot(room_facts=syn.room("1"), conversations=[], story=syn.pages(1, 300))
    snap = dataclasses.replace(snap, spec=dataclasses.replace(snap.spec, work_room_levels=2))
    core = cf.ContextCore("Core", "BIBLE", "# Architecture\n\n" + "Full book. " * 2000, "", "Identity", "Facts",
                          '"Assignment"', False, memory_view_json=mv.snapshot_json(snap))
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    return cf.build_context_fit_plan(None, core, {"id": "wide01", "chat_id": 1}, preferred_mode="max", tool_schemas=tools,
        reasoning_effort=effort, route_resolver=lambda *_a, **_kw: (
            {"provider": provider, "model": "test"}, ce.CapabilityEvidence(window, "confirmed", "metadata", "wide-route")))


def test_the_first_window_holding_max_is_where_the_request_it_sends_fits(monkeypatch):
    """Max holds exactly where the request this start sends (messages and schemas in the one count a round measures,
    nothing host-only), its reply reserve and work room fit; one token of window less stays Low. 15.11.2026: the
    schemas, priced apart with separator spaces, read about 5,000 tokens over this request and started it Low."""
    lo, hi = 150_000, 260_000
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if _wide_start(mid, monkeypatch).initial_mode == "max" else (mid + 1, hi)
    held, short = _wide_start(hi, monkeypatch), _wide_start(hi - 1, monkeypatch)
    fact, over = _fact(held), _fact(short)
    sent = list(select_tool_schemas(WIDE, context_mode="max").schemas)
    messages = held.messages_for("max")
    request = cf.estimate_context_prompt_tokens(messages, sent)
    assert request == cf.estimate_context_prompt_tokens(project_wire_messages(messages), sent)  # the wire's count
    assert held.initial_mode == "max" and fact["need"] == request + fact["reserve"] + fact["work"] <= hi
    assert held.projected_tokens_with_tools("max", sent) == request  # density 1: a route's projection reads the same
    assert fact["schemas"] == cf.tool_schema_tokens(sent) == math.ceil(len(json.dumps(
        sent, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) / 4)
    assert short.initial_mode == "low" and over["cause"] == "crowded" and over["need"] > over["window"] == hi - 1
    assert short.core == held.core and short.output_reserve_tokens == held.output_reserve_tokens  # one window apart


def test_a_direct_openai_reasoning_start_measures_the_shape_it_sends(monkeypatch):
    """Direct OpenAI with reasoning sends the schemas as its custom catalog (``openai_chat_dispatch``): a start told
    that effort measures the longer shape the first round will send and names it as the schemas' part."""
    plain = _wide_start(2_000_000, monkeypatch, provider="openai", tools=SMALL)
    reasoning = _wide_start(2_000_000, monkeypatch, provider="openai", effort="high", tools=SMALL)
    sent = list(select_tool_schemas(SMALL, context_mode="max").schemas)
    assert plain.max_projection.system_content_json == reasoning.max_projection.system_content_json  # the same view
    for plan, effort in ((plain, ""), (reasoning, "high")):
        fact, shape = _fact(plan), {"provider": "openai", "reasoning_effort": effort}
        assert fact["need"] == cf.estimate_context_prompt_tokens(plan.messages_for("max"), sent, **shape) + (
            fact["reserve"] + fact["work"])
        assert fact["schemas"] == cf.tool_schema_tokens(sent, **shape)
    assert _fact(reasoning)["schemas"] > _fact(plain)["schemas"] and _fact(reasoning)["need"] > _fact(plain)["need"]


def test_the_connections_map_is_a_part_of_its_own_beside_the_assignment(monkeypatch):
    """The connections map Max shows beside the core (``section_floor._system_blocks``) is its own measured part; the
    assignment part keeps only the task's messages. 15.11.2026: a real map of 213,668 characters read as assignment."""
    connections = ("## Connections\n\n" + "- mcp_fixture: one server's tool line\n" * 400, "## Connections\n\nmcp_fixture")
    bare = _start(_core(), 2_000_000, monkeypatch)
    mapped = _start(dataclasses.replace(_core(), connections=connections), 2_000_000, monkeypatch)
    bare_parts, mapped_parts = dict(_fact(bare)["parts"]), dict(_fact(mapped)["parts"])
    assert bare.initial_mode == mapped.initial_mode == "max" and "connections" not in bare_parts
    assert any(connections[0] in block["text"] for block in json.loads(mapped.max_projection.system_content_json))
    assert mapped_parts["connections"] == mf.view_tokens(connections[0]) > 2_000
    assert abs(mapped_parts["assignment"] - bare_parts["assignment"]) <= 1  # the map's separator and rounding only
    assert f"connections {mapped_parts['connections']:,}" in memory_facts(mapped, "max")


def test_no_fact_without_a_known_window_or_the_owners_max(tmp_path, monkeypatch):
    from tests.test_memory_view_modes import _built

    _core_, build = _built(tmp_path, monkeypatch, ratio=1)
    unknown = build(1, status="failed")
    assert unknown.initial_mode == "max" and "max_room" not in unknown.max_projection.memory_facts
    assert "Max room" not in memory_facts(unknown, "max")
    for owner in ("low", "nano"):
        chosen = build(1_000_000, preferred=owner)
        assert "max_room" not in chosen.projection(owner).memory_facts


@pytest.mark.parametrize("mode,floor_need,cause", [("max", 900, "held"), ("low", 900, "crowded"),
                                                    ("low", 1_001, "route")])
def test_the_settled_fact_names_cause_cuts_and_gap(mode, floor_need, cause):
    # floor_need <= window < zero_need: tool schemas tipped this start, which is crowded, never the route's fact
    fact = {"window": 1_000, "allowance": 0, "floor_need": floor_need, "schemas": 200, "zero_need": 1_100,
            "need": 1_230, "story": 60, "room": 70, "unfolded": 20, "levels": 1, "room_id": "1", "reserve": 10,
            "work": 125, "folds": 0, "parts": [["books", 600], ["core", 40]]}
    facts = {"floor": {"steps": {"F1": 2, "F6w": 5, "F6a": 1, "F2": 4}}}
    settled = mf.with_max_room(facts, fact, mode)["max_room"]
    assert settled["cause"] == cause and settled["short"] == 230 and "headroom" not in settled
    assert settled["cuts"] == (["F6w", "F2", "F6a"] if cause == "held" else [])  # the ladder's order, F6a last
    assert mf.with_max_room(facts, None, mode) == facts
    # Memory took the books' room only when everything else fits: here the tool schemas tip it (zero_need > window).
    assert not mf.crowded_by_memory(settled) and not mf.crowded_by_memory(None)
    assert mf.crowded_by_memory({**settled, "zero_need": 1_000}) is (cause == "crowded")
    words = mf.lowered_line("low", "max", fact) + mf.measure_text(fact, parts=True)
    assert "with no memory shown over by 100" in words
    assert not any(word in words for word in ("must", "should", "please", "failure", "may bring"))  # measures only
