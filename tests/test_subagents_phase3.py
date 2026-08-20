from __future__ import annotations

import json
import queue


def test_no_lane_fans_out_and_depth_does_not_downgrade(monkeypatch):
    """A lane names STRENGTH, and strength is one model, so every lane resolves to exactly
    one model at every depth.

    Two behaviors died in v6.87.7 and this pins both. The `review`/`scope` lanes fanned out
    across the configured reviewer slots — a TOPOLOGY smuggled in through a strength
    parameter, which no review surface ever used (they read their slots from config and run
    on the review substrate). And the capability-depth cap collapsed a nested child to Light
    regardless of what its parent asked for. Depth bounds how deep delegation NESTS, never
    how strong a descendant is. v6.87.28 removed the last trace of the fan-out: the slot
    LIST, which had had exactly one member per lane since the lanes went.
    """
    from ouroboros.subagents import (
        SUBAGENT_MODEL_LANES,
        normalize_subagent_model_lane,
        resolve_subagent_lane,
    )
    import pytest

    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "light-model")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "heavy-model")

    for lane in ("review", "scope"):
        assert lane not in SUBAGENT_MODEL_LANES
        with pytest.raises(ValueError, match="model_lane must be one of"):
            normalize_subagent_model_lane(lane)

    # Depth is not an input to the lane resolver at all (the dead parameter was
    # removed — XG-2R.4); "at every depth alike" now holds by construction.
    assert resolve_subagent_lane("heavy").model == "heavy-model"
    # v6.87.26: an omitted lane inherits the parent's.
    inherited = resolve_subagent_lane("auto", parent_lane="heavy")
    assert inherited.effective_lane == "heavy"


def test_schedule_subagent_emits_intent_only_and_no_task_group(monkeypatch, tmp_path):
    """One request schedules one child, and states what was ASKED FOR.

    The lane, the model and the effort are DERIVED at dispatch (v6.87.28), so the
    scheduling event carries the request and the parent's own lane — the fact an
    omitted lane inherits — and nothing that would need live availability to know."""
    from ouroboros.task_results import STATUS_REQUESTED
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "heavy-model")
    event_queue: queue.Queue = queue.Queue()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = event_queue
    ctx.task_metadata = {"root_task_id": "root1", "session_id": "sess1",
                         "effective_model_lane": "main"}

    result = _schedule_task(
        ctx,
        objective="Review the design",
        expected_output="One findings list",
        role="reviewer",
        model_lane="heavy",
    )

    assert "TOOL_ARG_ERROR" not in result
    event = event_queue.get_nowait()
    assert event_queue.empty()
    assert event["requested_model_lane"] == "heavy"
    assert event["parent_model_lane"] == "main"
    assert "effective_model_lane" not in event and "model" not in event
    # One request, one child: the lane fan-out that a group id existed for has not
    # been reachable since v6.87.7, so no group is minted and none is claimed.
    assert "task_group_id" not in event and "task_group" not in event
    assert event["subagent_envelope"]["task_group_id"] == ""
    assert event["subagent_envelope"]["status"] == STATUS_REQUESTED
    assert event["subagent_envelope"]["lineage"]["root_task_id"] == "root1"

    path = tmp_path / "task_results" / f"{event['task_id']}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["requested_model_lane"] == "heavy"
    assert data["parent_model_lane"] == "main"
    assert "model" not in data and "effective_model_lane" not in data
    assert "task_group_id" not in data


def test_schedule_subagent_drive_failure_is_fail_closed(monkeypatch, tmp_path):
    """A drive that cannot be prepared leaves NOTHING behind: no event, no durable record,
    no half-provisioned state directory."""
    import ouroboros.tools.control_scheduling as control
    from ouroboros.headless import HEADLESS_TASKS_DIR
    from ouroboros.tools.registry import ToolContext

    def fake_prepare(_root, _tid, _mode):
        raise RuntimeError("boom")

    monkeypatch.setattr(control, "prepare_task_drive", fake_prepare)
    event_queue: queue.Queue = queue.Queue()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = event_queue
    ctx.task_metadata = {"root_task_id": "root1", "session_id": "sess1"}

    result = control._schedule_task(
        ctx,
        objective="Review the design",
        expected_output="One findings list",
        role="reviewer",
    )

    assert "SUBTASK_DRIVE_ERROR" in result
    assert event_queue.empty()
    assert not any((tmp_path / "task_results").glob("*.json"))
    assert not any((tmp_path / HEADLESS_TASKS_DIR).glob("*"))
