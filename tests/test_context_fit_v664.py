"""Ordinary-Main target-versus-capacity fit regressions."""

from __future__ import annotations

import json

import pytest



def _projection(mode: str):
    from ouroboros.context_fit import ContextFitProjection

    return ContextFitProjection(
        mode=mode,
        system_content_json=json.dumps(f"{mode.upper()}_SYSTEM"),
        estimated_tokens=10,
        calibrated_tokens=10,
        calibration_ratio=1.0,
    )


def _plan(*, preferred: str = "max", window: int = 0, known: bool = False):
    from ouroboros.context_fit import ContextFitPlan

    return ContextFitPlan(
        core_sha256="a" * 64,
        preferred_mode=preferred,
        initial_mode=preferred,
        model="openai/test-model",
        provider="openai",
        route_fp="route-a",
        status="confirmed" if known else "unprobeable",
        stale=False,
        window_tokens=window,
        output_reserve_tokens=65_536,
        user_content_json=json.dumps("go"),
        max_projection=_projection("max"),
        low_projection=_projection("low"),
    )


def test_route_switch_without_immutable_core_fails_loudly(tmp_path):
    from ouroboros import loop
    from ouroboros.tools.registry import ToolRegistry

    with pytest.raises(RuntimeError, match="CONTEXT_FIT_REBUILD_FAILED"):
        loop._rebind_context_fit_plan(
            None,
            ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
            [{"role": "user", "content": "go"}],
            model="new-model",
            use_local=False,
            preferred_mode="max",
            tool_schemas=[],
        )
