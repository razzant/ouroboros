"""The Max room of a start: one fact beside whole books, its cause and the words that carry it."""
import dataclasses
import json
import math

from types import SimpleNamespace

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
    assert dict(fact["parts"])["books"] > dict(fact["parts"])["runtime facts"] and "tool schemas" not in dict(fact["parts"])
    line = memory_facts(plan, "max")  # every start, Max held too: the measure part by part, no advice
    assert line.startswith(f"Max room in this captured build: window 2,000,000; the gate's need {fact['need']:,}, headroom "
                           f"{2_000_000 - fact['need']:,}; the ordinary need {fact['need']:,}; with no memory ")  # no telling priced
    assert (f"; mine: my memory at the gate's level {fact['story'] + fact['room']:,}; what the request itself carries: tool "
            f"schemas {fact['schemas']:,}, runtime facts {dict(fact['parts'])['runtime facts']:,}; my human's: books "
            f"{dict(fact['parts'])['books']:,}, reply reserve {fact['reserve']:,}, work room {fact['work']:,}; core ") in line
    assert f"story {fact['story']:,} (40 pages no part or account holds yet, {fact['unfolded']:,} tokens, most in " in line
    assert f"beside a Max room of {fact['allowance']:,}; Max held;" in line
    assert "; marks in view 0" in line and "drafts awaiting" not in line and "shown only by address" not in line


def test_the_measure_reads_each_host_section_by_whose_part_and_adds_up(monkeypatch):
    """The split by whose part reads the host's sections by id (``memory_floor.max_room_fact``): knowledge, the source
    registry, scratchpad and held note heads are mine; runtime facts, skills (their lines and always-in-context bodies,
    review round 1, Astra 2) and subagents what the request itself carries; the books my human's; core and the
    assignment alone. With the tool schemas and memory they are the start's
    whole need at density 1, and a calibrated density is named beside them: the needs apply it, the parts do not."""
    from ouroboros.section_floor import HostSection

    stable = (HostSection("stable_0", "Identity " * 50), HostSection("subagents", "s" * 800),
              HostSection("always_on_skills", "a" * 700))
    head = (HostSection("knowledge_index", "k" * 1_200),)
    dynamic = (HostSection("runtime", "f" * 1_000), HostSection("registry", "r" * 600), HostSection("scratchpad", "p" * 400),
               HostSection("installed_skills", "i" * 300), HostSection("held_note:plans", "h" * 500))
    core = dataclasses.replace(_core(), semi_stable_text="\n\n".join(stable), semi_stable_sections=stable,
                               dynamic_head_text="\n\n".join(head), dynamic_head_sections=head,
                               dynamic_text="\n\n".join(dynamic), dynamic_sections=dynamic)
    plan, tok = _start(core, 2_000_000, monkeypatch), mf.view_tokens
    fact = _fact(plan)
    parts = dict(fact["parts"])
    assert (parts["knowledge"], parts["source registry"], parts["scratchpad"], parts["held note heads"]) == (
        tok("k" * 1_200), tok("r" * 600), tok("p" * 400), tok("h" * 500))
    assert parts["skills and subagents"] == tok("s" * 800) + tok("i" * 300) + tok("a" * 700)
    assert parts["runtime facts"] == tok(core.dynamic_text) - tok("r" * 600) - tok("p" * 400) - tok("i" * 300) - tok("h" * 500)
    assert parts["core"] == (tok(core.base_prompt) + tok(core.bible_md) + tok(core.semi_stable_text) - tok("s" * 800)
                             - tok("a" * 700))
    assert sum(parts.values()) + fact["schemas"] + fact["story"] + fact["room"] + fact["reserve"] + fact["work"] == fact["need"]
    line = memory_facts(plan, "max")
    assert (f"mine: my memory at the gate's level {fact['story'] + fact['room']:,}, knowledge {parts['knowledge']:,}, source "
            f"registry {parts['source registry']:,}, scratchpad {parts['scratchpad']:,}, held note heads "
            f"{parts['held note heads']:,}; what the request itself carries: tool schemas {fact['schemas']:,}, runtime facts "
            f"{parts['runtime facts']:,}, skills and subagents {parts['skills and subagents']:,};") in line
    assert fact["density"] == 1.0 and "parts before density" not in line
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.3)
    dense = cf.build_context_fit_plan(None, core, {"id": "room01", "usage_category": "consciousness"},
        preferred_mode="max", route_resolver=lambda *_a, **_kw: (
            {"provider": "test", "model": "test"}, ce.CapabilityEvidence(2_000_000, "confirmed", "metadata", "room-route")))
    # Review round 2 (Astra advisory): the clause names what it covers; reply reserve and work room are window tokens
    assert _fact(dense)["density"] == 1.3 and "; parts before density 1.30 except reply reserve and work room; " in (
        memory_facts(dense, "max"))


def test_memory_or_schemas_that_take_the_books_room_are_measured_without_advice(monkeypatch):
    core = _core()
    held = _fact(_start(core, 2_000_000, monkeypatch))
    plan = _start(core, _crowded_window(held), monkeypatch)
    fact = _fact(plan)
    assert plan.initial_mode == "low" and fact["cause"] == "crowded" and fact["cuts"] == []
    assert fact["floor_need"] <= fact["window"] and fact["short"] == fact["need"] - fact["window"] > 0
    gate = fact["telling_need"]  # the gate's need, printed beside the ordinary one (owner 21.11.2026)
    assert fact["zero_need"] <= fact["window"] < gate <= fact["need"] and 0 < fact["telling"] <= fact["story"] + fact["room"]
    shown = plan.low_projection.system_content_json
    assert (f"Low, temporarily, for the window; my human's mode is Max. Whole books, my memory at the gate's level and the "
            f"work's room need {gate:,}; the window holds {fact['window']:,}; short by {gate - fact['window']:,}. Only the "
            "window cuts my memory here; Low's own target does not bind. The host's measure of this start: window "
            f"{fact['window']:,}; the gate's need {gate:,}, short by ") in shown and "b" * 40_000 not in shown
    assert f"; the ordinary need {fact['need']:,}; with no memory {fact['zero_need']:,}; mine: " in shown
    assert f"; mine: my memory at the gate's level {fact['telling']:,}; what the request itself carries: " in shown
    assert not any(words in shown for words in ("may bring the books back", "Folding memory", "Max with chapter maps"))
    assert f"(pages no part or account holds yet {fact['unfolded']:,})" in shown
    assert not any(words in shown for words in ("failure", "crowds the books out", "cannot make room", "did not fit beside", "Low with"))
    line = memory_facts(plan, "low")
    assert f"beside a Max room of {fact['allowance']:,}; Low, temporarily, for the window, cause memory; " in line
    from ouroboros.context_view_facts import mode_words
    assert mode_words(plan, "low") == ("low, temporarily, for the window (my human's mode: max); cause memory; "
                                       f"short {gate - fact['window']:,}")
    assert not any(words in line for words in ("may bring", "Folding", "failure"))
    # The fact rides in the receipt the task trace keeps, and a selection keeps the start's frame.
    assert mf.trace_facts(plan.low_projection.memory_facts)["max_room"] == fact
    selected = mf.select_view(plan, "low", [])
    assert selected.low_projection.memory_facts["max_room"] == fact
    assert "The host's measure of this start" in selected.low_projection.system_content_json


def test_a_max_start_the_window_took_to_nano_is_named_nano_in_its_facts(tmp_path, monkeypatch):
    """Review round 1 (Astra 4): the owner's shape at 92K leaves even the maps frame no room, so its Max start runs in
    Nano; its captured facts name Nano, temporarily, for the window, never the Low words, while a start the window took
    to Low keeps them."""
    from tests import _memory_scale_fixture as scale

    nano = scale.build(tmp_path, monkeypatch)
    assert nano.initial_mode == "nano" and _fact(nano)["cause"] in ("crowded", "route")
    line = memory_facts(nano, "nano")
    assert f"; Nano, temporarily, for the window, cause {mf.lowered_cause(_fact(nano))}" in line and "Low, temp" not in line
    core = _core()
    low = _start(core, _crowded_window(_fact(_start(core, 2_000_000, monkeypatch))), monkeypatch)
    assert low.initial_mode == "low" and "; Low, temporarily, for the window, cause memory; " in memory_facts(low, "low")
    # Review round 2 (Astra 4): a Low start a smaller route later takes to Nano names its captured build's mode.
    crowded = next(plan for plan in (scale.build(tmp_path, monkeypatch, window=window) for window in (
        300_000, 200_000, 150_000, 120_000, 100_000)) if plan.initial_mode == "low"
        and mf.crowded_by_memory(plan.projection("low").memory_facts.get("max_room")))
    record = nano.window_record  # as ``loop_model_call`` rebinds: the smaller route's own record
    taken = dataclasses.replace(crowded, window_record=record, window_tokens=record.total_equivalent).reproject_for_route(
        window_tokens=record.total_equivalent, known_window=True, ratio=1.0, output_reserve=crowded.output_reserve_tokens,
        tool_schemas=scale.tools(), start_mode="low", current_messages=crowded.messages_for("low"))
    assert taken.initial_mode == "nano" and taken.start == crowded.start == ("low", "memory")
    line = memory_facts(taken, "nano")
    assert "; Low, temporarily, for the window, cause memory; " in line and "Nano, temp" not in line


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
    assert fact["telling_need"] >= fact["zero_need"] >= fact["floor_need"]  # the gate's need is printed for every cause
    shown = plan.low_projection.system_content_json
    assert ("Low, temporarily, for the window; my human's mode is Max. With no memory and no tool schemas, whole books beside "
            f"the rest of this start (my core, the host's sections and the assignment) and the work's room need "
            f"{fact['floor_need']:,}; the window holds 200,000; short by "
            f"{fact['floor_need'] - 200_000:,}, a disclosed fact of this route: folding my memory or dropping tool schemas "
            "cannot make room here.") in shown and "may bring" not in shown
    line = memory_facts(plan, "low")
    assert ("; Low, temporarily, for the window, cause route, a disclosed fact of this route: folding memory or dropping "
            "tool schemas cannot make room here") in line
    from ouroboros.context_view_facts import mode_words
    assert mode_words(plan, "low").startswith("low, temporarily, for the window (my human's mode: max); cause route; short ")


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
    assert mf.lowered_cause(fact) == ("route" if floor_need > fact["window"] else "request")  # by its numbers alone
    assert ("the window holds 1,000; short by 1" if floor_need > 1_000 else "need 1,100; the window holds 1,000; short by 100") in words
    assert "; the ordinary need 1,230; with no memory 1,100; mine: my memory at the gate's level 130; " in words
    assert not any(word in words for word in ("must", "should", "please", "failure", "may bring"))  # measures only


def test_three_causes_name_what_took_the_whole_books(monkeypatch):
    """Owner 21.11.2026 and the consilium of 22.11 (P7): a measure, not blame. ``memory`` when the request would fit
    with no memory, ``request`` when the books fit without the tool schemas but not with them (F1 #7-#9), ``route``
    when even that does not fit; each start records its own, and ``window`` stays for other narrowings."""
    core = _core()
    memory = _start(core, _crowded_window(_fact(_start(core, 2_000_000, monkeypatch))), monkeypatch)
    route = _start(_core(pages=1, page_chars=100, books=900_000), 200_000, monkeypatch)
    roomy = _fact(_wide_start(2_000_000, monkeypatch))  # two eighths of work room: the request band's middle
    request = _wide_start((4 * (roomy["floor_need"] - roomy["work"]) + 2 * roomy["schemas"]) // 3, monkeypatch)
    from ouroboros.context_view_facts import mode_words
    for plan, cause in ((memory, "memory"), (request, "request"), (route, "route")):
        fact = _fact(plan)
        assert plan.start == ("low", cause) and mf.start_cause(plan.low_projection.memory_facts) == cause
        assert mf.lowered_cause(fact) == cause and fact["telling_need"] > fact["window"]
        # Review round 1 (Astra 5, Carbon 2): the facts row's shortfall names its basis, and only beside its own cause:
        # a cold restore's capture that lost the books for another cause than the recorded one names none.
        basis = {"memory": "", "request": " with no memory", "route": " with no memory and no tool schemas"}[cause]
        said = f"low, temporarily, for the window (my human's mode: max); cause {cause}"
        assert mode_words(plan, "low") == f"{said}; short {mf.lowered_need(fact)[1] - fact['window']:,}{basis}"
        other = "memory" if cause == "route" else "route"
        assert mode_words(dataclasses.replace(plan, start=("low", other)), "low") == said.replace(cause, other)
    fact = _fact(request)
    assert fact["floor_need"] <= fact["window"] < fact["zero_need"]
    assert (f"With no memory, whole books, what the request itself carries and the work's room need {fact['zero_need']:,}; "
            f"the window holds {fact['window']:,}; short by {fact['zero_need'] - fact['window']:,}.") in (
        request.low_projection.system_content_json)
    assert cf.carried_start(["low", "request"]) == ("low", "request") and cf.carried_start(["max", "unmeasured"]) == (
        "max", "unmeasured")


def test_a_start_without_a_known_window_asks_once_more_and_is_never_a_held_max(monkeypatch):
    """F1 #10: a start whose route named no window was recorded as a held Max. Before its first send it asks the
    route's window evidence once more; a window then known is walked like any answer, and one still unknown is a Max
    not measured, never held. No bound is guessed and the Low episode is left alone (no Max room). Review round 2
    (Astra 2): that second answer is used unless it failed, one that read the route without a window too."""
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    unknown = SimpleNamespace(route_fp="r", status="failed", stale=True, window_tokens=0)
    known = ce.CapabilityEvidence(2_000_000, "confirmed", "metadata", "room-route")

    def build(*answers):
        asked = []

        def resolver(*_a, **_kw):
            asked.append(1)
            answer = answers[min(len(asked), len(answers)) - 1]
            if isinstance(answer, Exception):
                raise answer
            return {"provider": "test", "model": "test"}, answer
        plan = cf.build_context_fit_plan(None, _core(), {"id": "room01", "usage_category": "consciousness"},
                                         preferred_mode="max", route_resolver=resolver)
        return plan, len(asked)

    plan, asked = build(unknown)
    assert asked == 2 and plan.initial_mode == "max" and plan.start == ("max", "unmeasured")
    assert "max_room" not in plan.max_projection.memory_facts
    from ouroboros.context_view_facts import mode_words
    assert mode_words(plan, "max") == "max (window not known: Max not measured)"
    learned, asked = build(unknown, known)
    assert asked == 2 and learned.start == ("max", "") and _fact(learned)["cause"] == "held"
    assert mode_words(learned, "max") == "max"
    # A second answer that read the route without a window (an account, a reply limit) is used, still unmeasured ...
    read = ce.CapabilityEvidence(0, "unprobeable", "none", "r2", account_fingerprint="acct",
                                 response_limit={"max_output_tokens": 4_096})
    adopted, asked = build(unknown, read)
    assert asked == 2 and adopted.start == ("max", "unmeasured") and (adopted.route_fp, adopted.status) == ("r2", "unprobeable")
    assert adopted.output_reserve_tokens == 4_096 < plan.output_reserve_tokens
    # ... and a second ask that failed, answered or raised, keeps the first: its route and evidence, never the failed read's.
    for failed in (SimpleNamespace(route_fp="r3", status="failed", stale=True, window_tokens=0), RuntimeError("read failed")):
        kept, asked = build(unknown, failed)
        assert asked == 2 and kept.start == ("max", "unmeasured") and kept.route_fp == "r" and kept.model == "test"
        assert kept.output_reserve_tokens == plan.output_reserve_tokens
    held, asked = build(known)
    assert asked == 1 and held.start == ("max", "")  # a known window is asked once


def test_max_rebound_onto_an_unknown_window_is_not_measured_as_its_window_text_says(tmp_path, monkeypatch):
    """Review round 2 (Carbon 1, Opus 1, Astra 3): a route rebind onto a window nothing names drops the plan's window
    record (``reproject_for_route``); the facts row's mode reads the window as its window text does
    (``context_view_facts.row_window``), so a held start and an unmeasured start rebound there both say Max is not
    measured, and rebound onto a known window both say Max. A window learned later keeps the start."""
    from ouroboros import context, loop
    from ouroboros.context_view_facts import mode_words, window_fact
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    unknown = SimpleNamespace(route_fp="rebind-unknown", status="failed", stale=True, window_tokens=0)
    known = ce.CapabilityEvidence(2_000_000, "confirmed", "metadata", "rebind-known")

    def start(evidence):
        return cf.build_context_fit_plan(None, _core(), {"id": "room01", "usage_category": "consciousness"},
            preferred_mode="max", route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"}, evidence))

    def rebind(plan, evidence):
        tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        tools._ctx.task_id = "rebind"
        monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_kw: ({"provider": "test", "model": "test"}, evidence))
        return loop._rebind_context_fit_plan(plan, tools, plan.messages_for("max"), model="test", use_local=False,
                                             tool_schemas=[])[0]

    held, unmeasured = start(known), start(unknown)
    assert held.start == ("max", "") and unmeasured.start == ("max", "unmeasured")
    for plan in (held, unmeasured):
        moved, back = rebind(plan, unknown), rebind(plan, known)
        assert (moved.initial_mode, moved.start, back.initial_mode, back.start) == ("max", plan.start, "max", plan.start)
        assert mode_words(moved, "max") == "max (window not known: Max not measured)"
        assert window_fact(moved, moved.output_reserve_tokens).startswith("window unknown;")  # the two labels agree
        assert mode_words(back, "max") == "max"
        assert window_fact(back, back.output_reserve_tokens).startswith("window 2,000,000 total:")


@pytest.mark.parametrize("ask", ["direct", "beside the build", "at the usual point"])
def test_an_unknown_window_whose_read_failed_is_read_anew_not_from_its_record(monkeypatch, ask):
    """Review round 1 (Carbon 1): the second ask of a start whose route named no window reads the route anew when the
    first read failed, live (``read`` failed) or as the probe's ten-minute record of that failure (``status`` failed),
    whichever way the fit asks: its own resolver, a question answered beside the build (a Claudexor route) or one asked
    at the usual point (another route). A route with no metadata source (unprobeable, no failed read) is not read
    again: its record is its answer."""
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(cf, "_route_key", lambda _task: "engine-route" if ask == "beside the build" else None)
    task = {"id": "room01", "usage_category": "consciousness"}
    for status, read, again in (("failed", "", True), ("unprobeable", "failed", True), ("unprobeable", "", False)):
        asked = []

        def resolver(asked_task, *, allow_fetch, status=status, read=read):
            asked.append(bool(asked_task.get("route_read_again")))
            return {"provider": "test", "model": "test"}, ce.CapabilityEvidence(0, status, "none", "r", read=read)
        route = resolver if ask == "direct" else cf.RouteQuestion(task, resolver)
        plan = cf.build_context_fit_plan(None, _core(), task, preferred_mode="max", route_resolver=route)
        assert plan.start == ("max", "unmeasured") and asked == [False, again], (status, read)
