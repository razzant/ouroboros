"""Issue #203: promoted managed tasks can search ordinary runtime data."""

from __future__ import annotations

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import active_tool_profile, filesystem_affordance_map
from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_MODE
from ouroboros.tools.registry import ToolContext, ToolRegistry


@pytest.mark.parametrize("runtime_mode", ["light", "advanced"])
def test_promoted_task_can_read_search_read_runtime_data(
    tmp_path, monkeypatch, runtime_mode,
):
    from ouroboros import config

    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", runtime_mode, raising=True)
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    logs = data / "logs"
    repo.mkdir()
    logs.mkdir(parents=True)
    marker = "ISSUE_203_RUNTIME_SEARCH_MARKER"
    (logs / "events.jsonl").write_text(
        f'{{"event":"before"}}\n{{"event":"{marker}"}}\n',
        encoding="utf-8",
    )

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    ctx = ToolContext(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)

    assert active_tool_profile(ctx) == "self_modification"
    first_read = registry.execute(
        "read_file",
        {"root": "runtime_data", "path": "logs/events.jsonl", "start_line": 1, "max_lines": 1},
    )
    search = registry.execute(
        "search_code",
        {"root": "runtime_data", "path": "logs", "query": marker},
    )
    second_read = registry.execute(
        "read_file",
        {"root": "runtime_data", "path": "logs/events.jsonl", "start_line": 2, "max_lines": 1},
    )

    assert '"event":"before"' in first_read
    assert marker in search
    assert marker in second_read

    affordances = filesystem_affordance_map(ctx, runtime_mode=runtime_mode)
    assert "runtime_data" in affordances["searchable_roots"]
    assert "task_drive" in affordances["searchable_roots"]
    assert "artifact_store" in affordances["searchable_roots"]


def test_specialized_child_searches_ordinary_runtime_data_but_not_owner_settings(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    logs = data / "logs"
    repo.mkdir()
    logs.mkdir(parents=True)
    (logs / "events.jsonl").write_text("CHILD_MUST_NOT_FIND\n", encoding="utf-8")

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=data,
        task_constraint=TaskConstraint(mode=LOCAL_READONLY_SUBAGENT_MODE),
    )
    registry.set_context(ctx)

    result = registry.execute(
        "search_code",
        {"root": "runtime_data", "path": "logs", "query": "CHILD_MUST_NOT_FIND"},
    )

    assert "CHILD_MUST_NOT_FIND" in result
    # The read→search closure does not widen the per-file owner-state/secret gate.
    (data / "settings.json").write_text('{"OPENAI_API_KEY":"CHILD_OWNER_SECRET"}', encoding="utf-8")
    blocked = registry.execute("search_code", {"root": "runtime_data", "path": "settings.json",
                                                "query": "CHILD_OWNER_SECRET"})
    assert "CHILD_OWNER_SECRET" not in blocked
    assert "Found 1 match" not in blocked
