"""Regression test for Issue #203:
Promoted managed tasks (self_modification profile) must be able to search runtime_data.
"""

from types import SimpleNamespace
from ouroboros.tool_access import active_tool_profile, _POLICY, _TOP_LEVEL_PRINCIPAL_POLICY


def test_top_level_policy_includes_search_for_runtime_data():
    assert "search" in _TOP_LEVEL_PRINCIPAL_POLICY["runtime_data"]
    assert "search" in _POLICY["self_modification"]["runtime_data"]
    assert "search" in _POLICY["workspace_task"]["runtime_data"]
    assert "search" in _POLICY["external_workspace_task"]["runtime_data"]


def test_promoted_task_active_profile_resolution():
    ctx = SimpleNamespace(
        is_workspace_mode=lambda: False,
        is_direct_chat=False,
        task_constraint=None,
    )
    profile = active_tool_profile(ctx)
    assert profile == "self_modification"
    assert "search" in _POLICY[profile]["runtime_data"]


def test_workspace_task_active_profile_resolution():
    ctx = SimpleNamespace(
        is_workspace_mode=lambda: True,
        workspace_mode="internal",
        is_direct_chat=False,
        task_constraint=None,
    )
    profile = active_tool_profile(ctx)
    assert profile == "workspace_task"
    assert "search" in _POLICY[profile]["runtime_data"]
