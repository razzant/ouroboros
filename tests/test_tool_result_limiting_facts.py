"""The measured delivery explains cuts and its retained reader recovers the bytes."""
from __future__ import annotations

import copy

import pytest

from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.core_file_tools import delivered_source_prefix
from tests._tool_result_delivery_shared import measured_fit
from tests.test_tool_result_delivery import _ctx, _deliver, _row, _view_json


def test_automatic_empty_view_names_target_pressure_and_can_be_read(tmp_path):
    ctx = _ctx(tmp_path)
    reg = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    reg.set_context(ctx)
    text = "An exact process observation.\r\n" * 3000
    candidates = []
    measure = measured_fit(window=100_000, target=2_000, reserve=500)

    def fit(messages, schemas):
        candidates.append(copy.deepcopy(messages))
        return measure(messages, schemas)

    messages, trace = _deliver(ctx, [_row("c1", "run_command", text)], fit,
                              messages=[{"role": "user", "content": "Prior context. " * 2000}])
    delivered = messages[-1]["content"]
    view = _view_json(delivered)
    assert view["shown_ranges"] == []
    diagnostic = view.get("delivery", {})
    assert "target_pressure" in diagnostic.get("limiters", []), diagnostic
    assert "physical_pressure" not in diagnostic["limiters"], diagnostic
    assert diagnostic["capacity_tokens"] == 100_000
    assert any(frame[-1]["content"] == delivered for frame in candidates)
    assert measure(messages, []) in [measure(frame, []) for frame in candidates]
    reader = dict(view["read_omitted"])
    name = reader.pop("tool")
    reread = reg.execute(name, reader)
    recovered = delivered_source_prefix(ctx.last_read_view, reread, len(reread))
    assert recovered == text.replace("\r\n", "\n")
    retained = ctx.drive_root / "task_results/artifacts" / ctx.task_id / reader["path"]
    assert retained.read_bytes() == text.encode()


@pytest.mark.parametrize("window,target,args,prefix,expected", [
    (100_000, None, {}, 0, {"unsolicited_share"}),
    (100_000, None, {"view_head_chars": 0, "view_tail_chars": 0}, 0, {"explicit_view"}),
    (2_000, 100_000, {}, 2000, {"physical_pressure", "minimum_view_unfit"}),
    (None, 2_000, {}, 2000, {"capacity_unknown", "target_pressure"}),
    (None, None, {"view_head_chars": 0, "view_tail_chars": 0}, 0, {"capacity_unknown", "explicit_view"}),
    (2_000, None, {"view_head_chars": 100_000, "view_tail_chars": 0}, 0, {"physical_pressure"}),
])
def test_each_measured_limiter_is_carried_inside_the_actual_view(tmp_path, window, target, args, prefix, expected):
    ctx = _ctx(tmp_path)
    measured = []
    fit = measured_fit(window=window, target=target, reserve=500)

    def capture(messages, schemas):
        measured.append(copy.deepcopy(messages))
        return fit(messages, schemas)

    messages, _ = _deliver(ctx, [_row("c1", "run_command", "x" * 100_000, args=args)], capture,
                           messages=[{"role": "user", "content": "Prior context. " * prefix}])
    view = _view_json(messages[-1]["content"])
    diagnostic = view["delivery"]
    assert set(diagnostic["limiters"]) == expected, diagnostic
    assert any(frame[-1]["content"] == messages[-1]["content"] for frame in measured)
    if args.get("view_head_chars") == 0:
        assert view["shown_ranges"] == [] and diagnostic["selection"] == "explicit"


def test_source_unavailable_and_full_output_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.tool_result_delivery.persist_result_source", lambda *_a: {})
    ctx = _ctx(tmp_path)
    messages, _ = _deliver(ctx, [_row("c1", "run_command", "x" * 100_000),
                                _row("c2", "run_command", "small\r\nexact")], measured_fit(window=20_000))
    assert _view_json(messages[0]["content"])["delivery"]["source_status"] == "source_unavailable"
    assert "FULL_RESULT_SOURCE_UNAVAILABLE=true" in messages[0]["content"]
    assert messages[1]["content"] == "small\r\nexact"


def test_parallel_automatic_views_disclose_the_shared_allowance(tmp_path):
    messages, _ = _deliver(_ctx(tmp_path), [_row(str(i), "run_command", str(i) * 100_000) for i in range(2)],
                           measured_fit(window=100_000))
    for message in messages:
        diagnostic = _view_json(message["content"])["delivery"]
        assert diagnostic["unasked_results"] == 2 and diagnostic["batch_share_tokens"] == 12_500
        assert diagnostic["limiters"] == ["unsolicited_share"]


def test_failed_source_minimum_uses_actual_envelope_not_standin(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.tool_result_delivery.persist_result_source", lambda *_a: {})
    ctx = _ctx(tmp_path)
    crossed_boundary = False
    for window in range(600, 1600, 50):
        fit = measured_fit(window=window, reserve=500)
        messages, _ = _deliver(ctx, [_row("c1", "run_command", "x" * 100_000)], fit)
        view = _view_json(messages[-1]["content"])
        assert view["delivery"]["source_status"] == "source_unavailable"
        if not view["shown_ranges"] and fit(messages, [])["estimated_input_tokens"] + 500 <= window:
            crossed_boundary = True
            assert "minimum_view_unfit" not in view["delivery"]["limiters"], (window, view)
    assert crossed_boundary, "the sweep must exercise an actually fitting empty view"
