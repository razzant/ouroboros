"""Start book choice and subsequent route recovery from one captured view."""
import dataclasses
import pytest
from ouroboros import context_budget as cb, memory_floor as mf
from ouroboros.context_fit import MinimumViewUnfit
from tests._context_fit_fixtures import _built



def _on_route(plan, window):
    return plan.reproject_for_route(window_tokens=window, known_window=True, ratio=1,
                                   output_reserve=65_536, tool_schemas=None, start_mode=plan.initial_mode)



def test_honest_max_selects_low_before_folding_story_and_keeps_window_budget(tmp_path, monkeypatch):
    core, build = _built(tmp_path, monkeypatch, books=600_000, ratio=1)
    roomy = build(1_000_000)
    tight = build(200_000)
    # Unified Body Map (BIBLE P1): Max carries the compact Body Map, never whole architecture_md,
    # so 200K holds Max cleanly without a spurious Low hop; only a physically tiny window steps to Nano.
    assert roomy.initial_mode == "max" and core.architecture_md not in roomy.max_projection.system_content_json.replace("\\n", "\n")
    assert tight.initial_mode == "max" and tight.preferred_mode == "max"
    assert roomy.projection("max").memory_facts["max_room"]["cause"] == "held"
    assert tight.projection("max").memory_facts["max_room"]["cause"] == "held"
    assert "word word word word" not in tight.max_projection.system_content_json
    assert tight.core_sha256 == roomy.core_sha256
    tiny = build(65_537)
    assert tiny.initial_mode == "nano" and tiny.preferred_mode == "max"
    floor = tiny.projection("nano").memory_facts["floor"]
    assert floor["target_tokens"] is None and floor["budget_allowance_tokens"] is None
    assert floor["mode_switch"] == {"from": "max", "to": "nano"}
    assert tiny.projection("nano").memory_facts["max_room"]["cause"] == "route"


@pytest.mark.parametrize("mode,target", [("low", cb.OWNER_LOW_TARGET_TOKENS), ("nano", cb.OWNER_NANO_TARGET_TOKENS)])
def test_owner_choices_keep_targets_and_unknown_window_does_not_invent_one(tmp_path, monkeypatch, mode, target):
    _, build = _built(tmp_path, monkeypatch, ratio=1)
    for window, status in ((1_000_000, "asserted"), (1, "failed")):
        plan = build(window, status=status, preferred=mode)
        assert plan.initial_mode == mode
        assert plan.projection(mode).memory_facts["floor"]["target_tokens"] == target
    unknown = build(1, status="failed")
    assert unknown.initial_mode == "max" and unknown.max_projection.memory_facts["floor"]["steps"] == {}
    with pytest.raises(MinimumViewUnfit, match="Core, assignment, resident control tools and work room"):
        build(1)


def test_larger_or_fitting_route_keeps_bytes_and_smaller_route_uses_same_snapshot(tmp_path, monkeypatch):
    _, build = _built(tmp_path, monkeypatch, ratio=1)
    plan = build(1_000_000)
    monkeypatch.setattr("ouroboros.memory_view.capture_memory_view", lambda *_a, **_kw: pytest.fail("recaptured memory"))
    assert _on_route(plan, 900_000).max_projection.system_content_json == plan.max_projection.system_content_json
    narrow = _on_route(plan, 65_537)
    assert narrow.initial_mode == "nano"
    shown = narrow.projection(narrow.initial_mode)
    floor = shown.memory_facts["floor"]
    assert floor["steps"] == {}  # The compact books and untouched memory already fit.
    assert floor["allowance_tokens"] == floor["physical_allowance_tokens"]
    reserve = cb.context_mode_limits(narrow.initial_mode, narrow.preferred_mode, narrow.output_reserve_tokens)[1]
    assert shown.calibrated_tokens + reserve <= narrow.window_tokens
    enlarged = _on_route(narrow, 1_000_000)
    assert enlarged.initial_mode == narrow.initial_mode
    assert enlarged.projection(enlarged.initial_mode).system_content_json == shown.system_content_json
    assert narrow.core_sha256 == plan.core_sha256
    assert narrow.core.memory_view_json == plan.core.memory_view_json


def test_a_narrowing_keeps_what_the_starts_request_took_with_no_memory(tmp_path, monkeypatch):
    """3A: Low's missed target names what the start's request took with no memory (``floor.no_memory_tokens``), the
    start's fact. A route that narrows the shown view keeps it, as it keeps the start's Max room, so the mode chooser can
    warn while later sends still miss Low, until a new start measures again; a start without it gets none."""
    _core, build = _built(tmp_path, monkeypatch, ratio=1, page_chars=400_000)
    plan = build(1_000_000)

    def missed(projection):
        facts = projection.memory_facts
        return dataclasses.replace(projection, memory_facts={**facts, "floor": {**facts["floor"], "no_memory_tokens": 315_536}})
    marked = dataclasses.replace(plan, **{f"{mode}_projection": missed(plan.projection(mode)) for mode in ("max", "low", "nano")})
    narrow = _on_route(marked, 65_537)
    assert narrow.projection(narrow.initial_mode).memory_facts["floor"]["no_memory_tokens"] == 315_536
    plain = _on_route(plan, 65_537)
    assert "no_memory_tokens" not in plain.projection(plain.initial_mode).memory_facts["floor"]


def test_smaller_route_addresses_memory_only_when_the_compact_request_exceeds_its_room(tmp_path, monkeypatch):
    from ouroboros import memory_view as mv

    core, build = _built(tmp_path, monkeypatch, ratio=1, page_chars=400_000)
    plan = build(1_000_000)
    assert plan.initial_mode == "max" and plan.max_projection.memory_facts["floor"]["steps"] == {}
    monkeypatch.setattr(mv, "capture_memory_view", lambda *_a, **_kw: pytest.fail("recaptured memory"))
    narrow = _on_route(plan, 65_537)
    shown = narrow.projection(narrow.initial_mode)
    floor = shown.memory_facts["floor"]
    snapshot = mv.snapshot_from_json(core.memory_view_json)
    whole_memory = mf.view_tokens(mv.render_story(snapshot)) + mf.view_tokens(mv.render_room(snapshot))
    assert whole_memory > floor["physical_allowance_tokens"]
    assert narrow.initial_mode == "nano" and floor["steps"]["F5l"] == 1
    # The start's Max room outlives the narrowing, in the view fact and in what a task trace keeps of it.
    assert shown.memory_facts["max_room"] == plan.max_projection.memory_facts["max_room"]
    assert mf.trace_facts(shown.memory_facts)["max_room"]["cause"] == "held"
    reserve = cb.context_mode_limits(narrow.initial_mode, narrow.preferred_mode, narrow.output_reserve_tokens)[1]
    assert shown.calibrated_tokens + reserve <= narrow.window_tokens
    enlarged = _on_route(narrow, 1_000_000)
    assert enlarged.initial_mode == narrow.initial_mode
    assert enlarged.projection(enlarged.initial_mode).system_content_json == shown.system_content_json
    assert narrow.core_sha256 == plan.core_sha256
    assert narrow.core.memory_view_json == plan.core.memory_view_json


def test_start_uses_one_floor_walk_and_calibrates_before_book_choice(tmp_path, monkeypatch):
    core, build = _built(tmp_path, monkeypatch, books=600_000, ratio=1)
    big = dataclasses.replace(core, semi_stable_sections=(),
                              semi_stable_text=core.semi_stable_text + "\n" + "steady " * 5_000)
    def build_big(window):
        from ouroboros import capability_evidence as ce, context_fit as cf
        return cf.build_context_fit_plan(None, big, {"id": "cal-mode"}, preferred_mode="max",
            route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "fixture"},
                ce.CapabilityEvidence(window, "asserted", "metadata", "fixture")), tool_schemas=[])
    est = build_big(1_000_000).max_projection.estimated_tokens
    window = est + 65_536 + 500  # fits Max at ratio=1; calibrated ratio=4/2.7 overflows Max before any Max floor walk
    render, walks = mf.render_view_for_mode, []
    def measured(*args, **kwargs):
        walks.append(bool(kwargs.get("shortest") or kwargs.get("level")))  # True: a host step's price, never a chosen view
        return render(*args, **kwargs)
    monkeypatch.setattr(mf, "render_view_for_mode", measured)
    assert build_big(window).initial_mode == "max"
    assert walks.count(False) == 1 and walks[-1] is False  # price candidates, then walk the selected view once
    monkeypatch.setattr("ouroboros.context_fit._route_calibration_ratio", lambda *_a, **_kw: 4 / 2.7)
    assert build_big(window).initial_mode == "nano"
    assert walks.count(False) == 2  # calibrated before mode selection: Nano's one walk, never a Max walk first
