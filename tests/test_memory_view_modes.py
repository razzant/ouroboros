"""Start book choice and subsequent route recovery from one captured view."""
import dataclasses
import pytest
from ouroboros import context_budget as cb, memory_floor as mf
from ouroboros.context_fit import MinimumViewUnfit

_TASK = {"type": "task", "text": "hi", "id": "tmain", "chat_id": 1}


def _built(tmp_path, monkeypatch, *, books=80_000, ratio=None, page_chars=0):
    """``(core, plan(window, **kw))`` on the shared installation; Max's books grown by ``books`` characters
    so that Max's fixed part exceeds Low's (the test repository's own books are smaller than its nav map)."""
    from types import SimpleNamespace

    from ouroboros import context, context_fit
    from tests._memory_view_context import world

    env, memory, _rooms = world(tmp_path)
    if page_chars:
        from ouroboros.chronicle_store import ChronicleStore
        from tests.test_common_story import _page

        ChronicleStore(memory.drive_root).ensure_activated()
        _page(memory.drive_root, "1", 10, 11, text="Story. " * (page_chars // 7))
    core = context._capture_context_core(env, memory, dict(_TASK), None, None)
    core = dataclasses.replace(core, architecture_md=core.architecture_md + "\n\n" + "word " * (books // 5))
    if ratio is not None:
        monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: ratio)

    def plan(window, *, status="asserted", preferred="max", tool_schemas=None):
        evidence = SimpleNamespace(route_fp="r", status=status, stale=status != "asserted", window_tokens=window)
        return context._build_context_fit_plan(env, core, dict(_TASK), preferred_mode=preferred, tool_schemas=tool_schemas,
                                               route_resolver=lambda *_a, **_kw: ({"model": "m", "provider": "p"}, evidence))
    return core, plan


def _on_route(plan, window):
    return plan.reproject_for_route(window_tokens=window, known_window=True, ratio=1,
                                   output_reserve=65_536, tool_schemas=None, start_mode=plan.initial_mode)



def test_honest_max_selects_low_before_folding_story_and_keeps_window_budget(tmp_path, monkeypatch):
    core, build = _built(tmp_path, monkeypatch, books=600_000, ratio=1)
    roomy = build(1_000_000)
    tight = build(200_000)
    assert roomy.initial_mode == "max" and core.architecture_md in roomy.max_projection.system_content_json.replace("\\n", "\n")
    assert tight.initial_mode == "low" and tight.preferred_mode == "max"
    floor = tight.projection("low").memory_facts["floor"]
    assert floor["target_tokens"] is None and floor["budget_allowance_tokens"] is None
    assert floor["mode_switch"] == {"from": "max", "to": "low"}
    assert "Low with chapter maps, a disclosed fact of this route" in tight.low_projection.system_content_json
    assert tight.projection("low").memory_facts["max_room"]["cause"] == "route"  # the books alone overflow 200k
    assert roomy.projection("max").memory_facts["max_room"]["cause"] == "held"
    assert "word word word word" not in tight.low_projection.system_content_json
    assert tight.core_sha256 == roomy.core_sha256


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
    _, build = _built(tmp_path, monkeypatch, books=600_000, ratio=1)
    walk = mf.fit_memory_view
    calls = []
    def measured(*args, **kwargs):
        calls.append(1)
        return walk(*args, **kwargs)
    monkeypatch.setattr(mf, "fit_memory_view", measured)
    assert build(330_000).initial_mode == "max"
    assert len(calls) == 1
    monkeypatch.setattr("ouroboros.context_fit._route_calibration_ratio", lambda *_a, **_kw: 4 / 2.7)
    assert build(330_000).initial_mode == "low"
    assert len(calls) == 2
