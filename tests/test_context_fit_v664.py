"""Ordinary-Main target-versus-capacity fit regressions."""

from __future__ import annotations

import pytest

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
