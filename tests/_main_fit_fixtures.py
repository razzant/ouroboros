"""Shared serial Main fit fixtures: the scripted fit plan and the model-round context built on it.

Moved from ``tests/test_loop_compaction.py`` so test modules do not import each other
(Development 02, "Documentation contract").
"""
from types import SimpleNamespace
from unittest.mock import Mock


def _plan(*, preferred="max"):
    def reproject(messages, mode):
        rebuilt = list(messages)
        rebuilt[0] = {"role": "system", "content": f"{mode.upper()}_SYSTEM"}
        return rebuilt

    plan = SimpleNamespace(
        model="same-model", route_fp="route-a", preferred_mode=preferred, core_sha256="a" * 64,
        status="confirmed", stale=False,  # evidence-shaped, as ``is_known`` reads the real plan
        window_tokens=500_000, output_reserve_tokens=65_536,
        projection=lambda _mode: SimpleNamespace(calibration_ratio=1.0),
        reproject_transcript=reproject, reproject_for_route=Mock(),
    )
    # The rebound plan names the selected mode, as the real projection does.
    plan.reproject_for_route.return_value = SimpleNamespace(**vars(plan), initial_mode="low")
    return plan


def _ctx(tmp_path, *, preferred="max", mode="max"):
    from ouroboros import loop

    inner = SimpleNamespace(
        messages=[],
        active_context_mode=mode,
        task_metadata={},
        _context_overflow_retries=set(),
    )
    tools = SimpleNamespace(_ctx=inner)
    messages = [
        {"role": "system", "content": f"{mode.upper()}_SYSTEM"},
        {"role": "user", "content": "go"},
    ]
    inner.messages = messages
    return loop._RoundModelCallContext(
        llm=SimpleNamespace(),
        messages=messages,
        tools=tools,
        context_fit_plan=_plan(preferred=preferred),
        active_model="same-model",
        tool_schemas=[],
        active_effort="high",
        max_retries=3,
        drive_logs=tmp_path / "logs",
        task_id="task",
        round_idx=1,
        event_queue=None,
        accumulated_usage={"execution_id": "exec"},
        task_type="task",
        active_use_local=False,
        active_context_mode=mode,
        drive_root=tmp_path,
    )
