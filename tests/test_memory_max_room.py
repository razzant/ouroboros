"""The Max room of a start: one fact beside whole books, its cause and the words that carry it."""
import dataclasses
import math

import pytest

from ouroboros import capability_evidence as ce, context_fit as cf, memory_floor as mf, memory_view as mv
from ouroboros.context_view_facts import memory_facts
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
    assert line.startswith("Max room at the start: window 2,000,000; books, core, knowledge, scratchpad, runtime facts, "
                           "assignment, reply reserve and work room")
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
    facts = {"floor": {"steps": {"F1": 2, "F1b": 1, "F6a": 3, "F2": 4}}}
    settled = mf.with_max_room(facts, fact, mode)["max_room"]
    assert settled["cause"] == cause and settled["short"] == 230 and "headroom" not in settled
    assert settled["cuts"] == (["F6a", "F2"] if cause == "held" else [])
    assert mf.with_max_room(facts, None, mode) == facts
    # Memory took the books' room only when everything else fits: here the tool schemas tip it (zero_need > window).
    assert not mf.crowded_by_memory(settled) and not mf.crowded_by_memory(None)
    assert mf.crowded_by_memory({**settled, "zero_need": 1_000}) is (cause == "crowded")
    words = mf.lowered_line("low", "max", fact) + mf.measure_text(fact, parts=True)
    assert "with no memory shown over by 100" in words
    assert not any(word in words for word in ("must", "should", "please", "failure", "may bring"))  # measures only
