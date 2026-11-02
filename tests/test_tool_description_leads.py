"""Rendered contracts that text refactoring must preserve.

Purpose and wording are reviewed, not frozen into prose assertions. These
checks cover alias contract delivery and actor-specific schema projection.
"""
from __future__ import annotations

import pytest


def _rendered(registry):
    from ouroboros.tool_policy import initial_tool_schemas

    return {schema["function"]["name"]: schema["function"] for schema in initial_tool_schemas(registry)}


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    from ouroboros.tools.registry import ToolRegistry

    root = tmp_path_factory.mktemp("descriptions")
    return _rendered(ToolRegistry(repo_dir=root, drive_root=root))


def test_older_commit_name_keeps_the_complete_contract():
    """A caller granted only the old spelling receives the same complete workflow."""
    from ouroboros.tools import git

    entries = {entry.name: entry for entry in git.get_tools()}
    current, older = entries["commit_reviewed"], entries["vcs_commit_reviewed"]
    assert older.handler is current.handler
    assert older.schema["parameters"] == current.schema["parameters"]
    contract = current.schema["description"].split("workflow: ", 1)[1]
    assert contract and older.schema["description"].endswith(contract)


def test_readonly_browser_projection_matches_its_actual_actions(rendered, tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(
        repo_dir=tmp_path, drive_root=tmp_path,
        task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
    ))
    readonly = _rendered(registry)
    full_actions = rendered["browser_action"]["parameters"]["properties"]["action"]["enum"]
    readonly_actions = readonly["browser_action"]["parameters"]["properties"]["action"]["enum"]
    assert readonly_actions == [action for action in full_actions if action != "evaluate"]
    for schema, actions in ((rendered["browser_action"], full_actions),
                            (readonly["browser_action"], readonly_actions)):
        for action in actions:
            assert f"{action} (" in schema["description"], action
    assert "evaluate (" not in readonly["browser_action"]["description"]
    assert "JavaScript evaluate is unavailable" in readonly["browser_action"]["description"]
