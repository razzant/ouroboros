"""Both early refusal sites retain the ordinary fitter's numeric evidence."""
from dataclasses import asdict
import json
import math
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as fit, memory_view
from ouroboros.contracts.realtime import RouteFacts
from tests import _memory_view_synthetic as syn


FACTS = RouteFacts(conversation_window_tokens=92_000, working_reserve_tokens=10_000,
                   source="synthetic retention measurement")


def build_minimum_view(root, monkeypatch, shape, *, ratio=1.0, task=None):
    """Real common fit over synthetic protected words; no provider or owner data."""
    words = {**syn.spoken(1, "human", 340_000 if shape == "memory" else 100),
             "owner_side": True, "immune": True}
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=[words]))
    core = fit.ContextCore(base_prompt="SYSTEM", bible_md="BIBLE", architecture_md="# Architecture",
        development_md="# Development", docs_need_development=False,
        semi_stable_text="IDENTITY " + "x" * {"fixed": 400_000, "work_room": 300_000}.get(shape, 0),
        dynamic_text="", user_content_json=json.dumps("Open the call."),
        memory_view_json=memory_view.snapshot_json(snapshot), section_root=str(root))
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: ratio)
    evidence = SimpleNamespace(route_fp="synthetic-account", status="confirmed", stale=False,
                               window_tokens=92_000, route_facts=asdict(FACTS), source=FACTS.source)
    return fit.build_context_fit_plan(SimpleNamespace(drive_root=root), core,
        task or {"id": "minimum-view", "_is_direct_chat": True}, preferred_mode="nano", tool_schemas=[],
        route_resolver=lambda *_a, **_kw: ({"provider": "synthetic", "model": "synthetic"}, evidence))


@pytest.mark.parametrize("shape", ["fixed", "memory"])
@pytest.mark.parametrize("ratio", [1.0, 1.333])
def test_both_raise_sites_name_their_measured_input_reserve_window_and_deficit(tmp_path, monkeypatch, shape, ratio):
    with pytest.raises(fit.MinimumViewUnfit) as raised:
        build_minimum_view(tmp_path, monkeypatch, shape, ratio=ratio)
    error = raised.value
    assert str(error).startswith("Core, assignment" if shape == "fixed" else "Protected owner words")
    assert error.physical
    assert error.fixed_input_tokens > 0 and error.memory_view_tokens > 0
    assert error.estimated_input_tokens == error.fixed_input_tokens + error.memory_view_tokens
    assert error.estimated_input_tokens == math.ceil(error.raw_input_tokens * ratio)
    assert error.response_reserve_tokens == 10_000 and error.window_tokens == 92_000
    assert error.deficit_tokens == error.estimated_input_tokens + 10_000 - 92_000 > 0
    for label, value in (("fixed input", error.fixed_input_tokens), ("memory view", error.memory_view_tokens),
                         ("reply/working reserve", 10_000), ("window", 92_000), ("deficit", error.deficit_tokens)):
        assert f"{label}={value}" in str(error)


def test_a_work_room_only_shortfall_starts_with_the_existing_notice(tmp_path, monkeypatch):
    plan = build_minimum_view(tmp_path, monkeypatch, "work_room")
    shown = plan.projection(plan.initial_mode)
    margin = 92_000 - 10_000 - shown.calibrated_tokens
    assert 0 <= margin < 92_000 // 8, f'physical margin={margin}; requested work room={92_000 // 8}'
    assert 'Start room is smaller than requested' in shown.system_content_json
