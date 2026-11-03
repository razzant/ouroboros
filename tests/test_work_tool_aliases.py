"""Saved names retain their restriction and schema-selection meaning."""

from types import SimpleNamespace

import pytest

from ouroboros.contracts.task_contract import normalize_disabled_tools
from ouroboros.contracts.task_contract import TOOL_NAME_ALIASES
from ouroboros.tool_policy import select_tool_schemas
from ouroboros.tools.registry_guards import _disabled_tools


@pytest.mark.parametrize("old,new", TOOL_NAME_ALIASES.items())
def test_saved_restriction_is_preserved_without_normalization(old, new):
    ctx = SimpleNamespace(task_metadata={"disabled_tools": [old, "route_to_project", "web_search"]})
    assert {old, new, "route_to_project", "web_search"} <= _disabled_tools(ctx)
    assert "read_file" not in _disabled_tools(ctx)
    assert normalize_disabled_tools([old, old, new]) == [old, new]


@pytest.mark.parametrize("old,new", TOOL_NAME_ALIASES.items())
def test_saved_profile_selects_successor(old, new):
    schemas = [{"function": {"name": name}} for name in (new, "read_file")]
    selected = select_tool_schemas(schemas, context_mode="nano", schema_names=[old])
    assert selected.chosen == (new,)
    assert selected.unknown == ()
    assert select_tool_schemas(schemas, context_mode="nano", schema_names=["route_to_project"]).unknown == ("route_to_project",)


def test_existing_retired_tool_restrictions_stay_intact():
    ctx = SimpleNamespace(task_metadata={"task_contract": {
        "disabled_tools": ["claude_code_edit", "advisory_review"],
    }})
    assert _disabled_tools(ctx) == frozenset({
        "claude_code_edit", "delegate_start", "advisory_review", "preflight_review",
    })
