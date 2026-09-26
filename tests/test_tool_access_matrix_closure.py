"""TZ-1 cluster E: the access matrix is closed under read⇒list,search and write⇒edit.

A root a profile may READ it may also LIST and SEARCH — the search tool reads
the same bytes through the same per-file guards and masks a child's matches, so
withholding it bought no protection and sent the model into a probe loop over
roots it could already read file by file — and a root it may WRITE it may also
EDIT (an exact replacement is a narrower write). The closure adds no write-like
operation anywhere, so every read-only ceiling holds: the read-only child writes
nothing, the orchestrator read-only roots stay {read, list, search} for every
profile, and the acting child's only write surface is still its isolated
active_workspace. Both directions are pinned: the property over the whole
matrix, and the rows the closure actually opened, end to end through the
registry.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import (
    _ALL_ROOTS,
    _POLICY,
    _READONLY_RESOURCE_ROOTS,
    _WRITE_LIKE_OPS,
    decide_tool_access,
    filesystem_affordance_map,
)
from ouroboros.tool_access_types import _close_operations
from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_MODE
from ouroboros.tools.registry import ToolContext, ToolRegistry

# --- the property, over every profile and root ---------------------------------

def test_every_row_is_closed_under_read_and_write_implications():
    for profile, matrix in _POLICY.items():
        for root, ops in matrix.items():
            if "read" in ops:
                assert {"list", "search"} <= ops, (profile, root, sorted(ops))
            if "write" in ops:
                assert "edit" in ops, (profile, root, sorted(ops))


def test_the_closure_added_no_write_like_operation_anywhere():
    """Read-only ceilings: the read-only child mutates nothing; the orchestrator
    read-only roots are exactly {read, list, search} wherever they appear; the
    acting child's write surface is still only its isolated workspace."""
    readonly_child = _POLICY["local_readonly_subagent"]
    assert not any(ops & _WRITE_LIKE_OPS for ops in readonly_child.values()), readonly_child
    for profile, matrix in _POLICY.items():
        for root in _READONLY_RESOURCE_ROOTS:
            if root in matrix:
                assert matrix[root] == {"read", "list", "search"}, (profile, root)
    acting = _POLICY["acting_subagent"]
    assert {root for root, ops in acting.items() if ops & _WRITE_LIKE_OPS} == {"active_workspace"}
    assert "system_repo" not in acting and "user_files" not in acting and "skill_payload" not in acting


def test_close_operations_adds_only_the_implied_read_family_and_edit():
    assert _close_operations({"read"}) == {"read", "list", "search"}
    assert _close_operations({"write"}) == {"write", "edit"}
    assert _close_operations({"list"}) == {"list"}
    assert _close_operations({"edit"}) == {"edit"}
    assert _close_operations({"vcs"}) == {"vcs"}
    closed = _close_operations({"read", "write", "vcs"})
    assert closed == {"read", "list", "search", "write", "edit", "vcs"}
    assert _close_operations(closed) == closed  # idempotent
    assert not (_close_operations({"read"}) & _WRITE_LIKE_OPS)


def test_the_rows_the_closure_opened():
    """The cells that were narrower than the rule: named, so a revert is loud."""
    for profile in ("workspace_task", "external_workspace_task", "self_modification"):
        for root in ("task_drive", "artifact_store"):
            assert decide_tool_access(profile=profile, root=root, operation="search").allow, (profile, root)
        assert decide_tool_access(profile=profile, root="artifact_store", operation="edit").allow, profile
    for profile in ("local_readonly_subagent", "acting_subagent"):
        for root in ("runtime_data", "task_drive", "artifact_store"):
            assert decide_tool_access(profile=profile, root=root, operation="search").allow, (profile, root)
            for op in ("write", "edit", "shell", "service"):
                if profile == "local_readonly_subagent":
                    assert not decide_tool_access(profile=profile, root=root, operation=op).allow, (root, op)
    # What stays closed stays closed: a child never reaches the roots it never had.
    for profile in ("local_readonly_subagent", "acting_subagent"):
        for root in ("subagent_projects", "user_files"):
            for op in ("read", "list", "search"):
                assert not decide_tool_access(profile=profile, root=root, operation=op).allow, (profile, root, op)
    assert set(_POLICY["operator_control"]) == _ALL_ROOTS


# --- end to end through the registry --------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    home, repo, data = tmp_path / "home", tmp_path / "repo", tmp_path / "data"
    for path in (home, repo, data):
        path.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    (repo / "README.md").write_text("repo readme\n", encoding="utf-8")
    (data / "logs").mkdir()
    (data / "logs" / "events.jsonl").write_text('{"event":"CLOSURE_LOG_MARKER"}\n', encoding="utf-8")
    (data / "settings.json").write_text('{"OPENAI_API_KEY": "CLOSURE_SECRET_MARKER"}\n', encoding="utf-8")
    task_drive = data / "task_drives" / "t1"
    task_drive.mkdir(parents=True)
    (task_drive / "notes.txt").write_text("CLOSURE_TASK_DRIVE_MARKER here\n", encoding="utf-8")
    child_drive = data / "task_drives" / "c1"
    child_drive.mkdir()
    (child_drive / "own.txt").write_text("CLOSURE_CHILD_DRIVE_MARKER\n", encoding="utf-8")
    return SimpleNamespace(home=home, repo=repo, data=data, task_drive=task_drive)


def _top_level(world):
    ctx = ToolContext(repo_dir=world.repo, drive_root=world.data, task_id="t1")
    registry = ToolRegistry(repo_dir=world.repo, drive_root=world.data)
    registry.set_context(ctx)
    return registry, ctx


def _readonly_child(world):
    ctx = ToolContext(
        repo_dir=world.repo, drive_root=world.data, task_id="c1",
        task_constraint=TaskConstraint(mode=LOCAL_READONLY_SUBAGENT_MODE),
    )
    registry = ToolRegistry(repo_dir=world.repo, drive_root=world.data)
    registry.set_context(ctx)
    return registry, ctx


def test_affordance_map_lists_the_task_roots_as_searchable(world):
    _registry, ctx = _top_level(world)
    affordance = filesystem_affordance_map(ctx)
    assert {"runtime_data", "task_drive", "artifact_store"} <= set(affordance["searchable_roots"])


def test_top_level_task_searches_its_task_drive_and_artifact_store(world):
    registry, _ctx = _top_level(world)
    wrote = registry.execute("write_file", {"root": "artifact_store", "path": "report.txt",
                                            "content": "CLOSURE_ARTIFACT_MARKER\n"})
    assert wrote.startswith("OK:"), wrote
    drive = registry.execute("search_code", {"root": "task_drive", "path": ".", "query": "CLOSURE_TASK_DRIVE"})
    store = registry.execute("search_code", {"root": "artifact_store", "path": ".", "query": "CLOSURE_ARTIFACT"})
    assert "task_drive:notes.txt:1:" in drive and "CLOSURE_TASK_DRIVE_MARKER" in drive, drive
    assert "artifact_store:report.txt:1:" in store and "CLOSURE_ARTIFACT_MARKER" in store, store


def test_top_level_task_edits_the_artifact_it_wrote(world):
    registry, _ctx = _top_level(world)
    registry.execute("write_file", {"root": "artifact_store", "path": "report.txt", "content": "draft v1\n"})
    edited = registry.execute("edit_text", {"root": "artifact_store", "path": "report.txt",
                                            "old_str": "v1", "new_str": "v2"})
    assert edited.startswith("OK: edited artifact_store:report.txt"), edited
    assert (world.data / "task_results" / "artifacts" / "t1" / "report.txt").read_text(encoding="utf-8") == "draft v2\n"


def test_readonly_child_searches_runtime_data_through_its_read_guards(world):
    """The child may now search what it could already read file by file — and the
    per-file secret guard the read applies still hides the owner's settings."""
    registry, _ctx = _readonly_child(world)
    result = registry.execute("search_code", {"root": "runtime_data", "path": ".", "query": "CLOSURE_"})
    assert "CLOSURE_LOG_MARKER" in result, result
    assert "CLOSURE_SECRET_MARKER" not in result and "settings.json" not in result, result
    drive = registry.execute("search_code", {"root": "task_drive", "path": ".", "query": "CLOSURE_CHILD"})
    assert "task_drive:own.txt:1:" in drive and "CLOSURE_CHILD_DRIVE_MARKER" in drive, drive
    enum = registry.get_schema_by_name("search_code")["function"]["parameters"]["properties"]["root"]["enum"]
    assert {"runtime_data", "task_drive", "artifact_store"} <= set(enum), enum
    assert "user_files" not in enum and "subagent_projects" not in enum, enum


def test_readonly_child_refusal_still_names_the_roots_it_can_search(world):
    registry, _ctx = _readonly_child(world)
    result = registry.execute("search_code", {"root": "user_files", "path": ".", "query": "CLOSURE_"})
    assert result.startswith("⚠️ TOOL_ACCESS_BLOCKED"), result
    assert "Roots your profile can search:" in result
    for root in ("active_workspace", "system_repo", "runtime_data", "task_drive", "artifact_store"):
        assert root in result, (root, result)
    for tool, args in (
        ("write_file", {"root": "runtime_data", "path": "logs/x.txt", "content": "x"}),
        ("edit_text", {"root": "task_drive", "path": "notes.txt", "old_str": "a", "new_str": "b"}),
    ):
        assert registry.execute(tool, args).startswith("⚠️"), tool
