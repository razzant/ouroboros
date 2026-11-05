"""D16 marker-aware payload authority and selected manifestless recovery."""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import (
    build_resolved_resource_binding,
    filesystem_affordance_map,
)
from ouroboros.tools.registry import ToolContext, ToolRegistry

_HUB_CATALOG = "https://raw.githubusercontent.com/ExampleOwner/ExampleHub/main/catalog.json"


def _skill(
    data: pathlib.Path,
    name: str,
    *,
    bucket: str = "native",
    seeded: bool = False,
) -> pathlib.Path:
    payload = data / "skills" / bucket / name
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    (payload / "notes.txt").write_text("old\n", encoding="utf-8")
    if seeded:
        (payload / ".seed-origin").write_text("launcher-seed\n", encoding="utf-8")
    return payload


def _registry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
) -> tuple[ToolRegistry, pathlib.Path, pathlib.Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_a, **_kw: (True, ""),
    )
    return ToolRegistry(repo_dir=repo, drive_root=data), repo, data



@pytest.mark.parametrize("mode", ["advanced", "pro"])
def test_legacy_runtime_data_native_alias_uses_marker_ownership(
    mode,
    tmp_path,
    monkeypatch,
):
    registry, _repo, data = _registry(tmp_path, monkeypatch, mode=mode)
    user_payload = _skill(data, "legacy-user")
    seeded_payload = _skill(data, "legacy-seeded", seeded=True)

    user_write = registry.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": "skills/native/legacy-user/new.txt",
            "content": "new\n",
        },
    )
    user_edit = registry.execute(
        "edit_text",
        {
            "root": "runtime_data",
            "path": "skills/native/legacy-user/notes.txt",
            "old_str": "old",
            "new_str": "new",
        },
    )
    seeded_write = registry.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": "skills/native/legacy-seeded/new.txt",
            "content": "bad\n",
        },
    )
    seeded_edit = registry.execute(
        "edit_text",
        {
            "root": "runtime_data",
            "path": "skills/native/legacy-seeded/notes.txt",
            "old_str": "old",
            "new_str": "bad",
        },
    )

    assert user_write.startswith("OK:"), user_write
    assert user_edit.startswith("OK: edited"), user_edit
    assert (user_payload / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert "read/review only" in seeded_write
    assert "read/review only" in seeded_edit
    assert not (seeded_payload / "new.txt").exists()
    assert (seeded_payload / "notes.txt").read_text(encoding="utf-8") == "old\n"


@pytest.mark.parametrize(
    "constraint",
    [
        TaskConstraint(mode="local_readonly_subagent"),
        TaskConstraint(
            mode="skill_repair",
            skill_name="user-native",
            payload_root="skills/external/user-native",
        ),
    ],
)
def test_external_alias_does_not_widen_child_or_repair_profiles(
    constraint,
    tmp_path,
):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    _skill(data, "user-native")
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_constraint=constraint)

    refusal = "cannot select skill location=native" if constraint.mode == "local_readonly_subagent" else "selected a different skill payload"
    with pytest.raises(ValueError, match=refusal):
        build_resolved_resource_binding(
            ctx,
            root="skill_payload",
            operation="read",
            path="notes.txt",
            bucket="external",
            skill_name="user-native",
        )


def test_logical_external_refuses_physical_external_native_collision(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    _skill(data, "same", bucket="external")
    _skill(data, "same", bucket="native")
    ctx = ToolContext(repo_dir=repo, drive_root=data)

    for operation in ("read", "write"):
        with pytest.raises(ValueError, match="collision"):
            build_resolved_resource_binding(
                ctx,
                root="skill_payload",
                operation=operation,
                path="SKILL.md",
                bucket="external",
                skill_name="same",
            )


def _publish_registry(
    repo: pathlib.Path,
    data: pathlib.Path,
    checkout: pathlib.Path,
    skill: str,
    monkeypatch: pytest.MonkeyPatch,
) -> ToolRegistry:
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_a, **_kw: (True, ""),
    )
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=data,
        current_task_type="skill_publish",
        task_metadata={
            "skill_publish_target": {
                "skill": skill,
                "repository": "ExampleOwner/ExampleHub",
            }
        },
    )
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)
    return registry


@pytest.mark.parametrize(
    ("manifest_name", "manifest"),
    [
        (
            "SKILL.md",
            "---\nname: draft\ndescription: Draft skill.\nversion: 1.0.0\ntype: instruction\n---\n# Draft\n",
        ),
        (
            "skill.json",
            json.dumps(
                {
                    "name": "draft",
                    "description": "Draft skill.",
                    "version": "1.0.0",
                    "type": "instruction",
                }
            )
            + "\n",
        ),
    ],
)
def test_selected_manifestless_preflight_to_manifest_and_fresh_review(
    manifest_name,
    manifest,
    tmp_path,
    monkeypatch,
):
    from ouroboros.gateway.skill_publish import build_skill_publish_preflight
    from ouroboros.skill_loader import load_review_state
    from tests.test_skill_review_persist_guard import _clean_wave

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    checkout = tmp_path / "checkout"
    leaf = checkout / "draft"
    repo.mkdir()
    data.mkdir()
    leaf.mkdir(parents=True)
    (leaf / "notes.txt").write_text("INSPECT_TOKEN\n", encoding="utf-8")

    outcome = build_skill_publish_preflight(
        data,
        "draft",
        repo_path=str(checkout),
        catalog_url=_HUB_CATALOG,
        github_token_configured=True,
    )
    assert outcome.status_code == 200
    assert outcome.payload["reason_code"] == "snapshot_manifest_missing"
    assert outcome.payload["task_start_allowed"] is True

    registry = _publish_registry(repo, data, checkout, "draft", monkeypatch)
    affordance = filesystem_affordance_map(registry._ctx, runtime_mode="light")
    assert "omit bucket" in affordance["skill_payload_selector"]
    assert "draft" in affordance["skill_payload_selector"]
    selector = {"root": "skill_payload", "skill_name": "draft"}
    assert "notes.txt" in registry.execute("list_files", {**selector, "path": "."})
    assert "INSPECT_TOKEN" in registry.execute("read_file", {**selector, "path": "notes.txt"})
    assert "INSPECT_TOKEN" in registry.execute("search_code", {**selector, "path": ".", "query": "INSPECT_TOKEN"})

    nested = registry.execute(
        "write_file",
        {**selector, "path": f"nested/{manifest_name}", "content": "# wrong\n"},
    )
    wrong_bucket = registry.execute(
        "write_file",
        {
            **selector,
            "bucket": "external",
            "path": manifest_name,
            "content": "# parallel\n",
        },
    )
    assert "not found" in nested.lower() or "requires bucket" in nested.lower()
    assert "not in requested" in wrong_bucket
    assert not (data / "skills" / "external" / "draft").exists()

    created = registry.execute(
        "write_file",
        {**selector, "path": manifest_name, "content": manifest},
    )
    assert created.startswith("OK:"), created
    assert (leaf / manifest_name).read_text(encoding="utf-8") == manifest
    omitted_after_manifest = registry.execute("read_file", {**selector, "path": "notes.txt"})
    assert "bucket may be omitted only" in omitted_after_manifest
    assert "INSPECT_TOKEN" in registry.execute(
        "read_file",
        {**selector, "bucket": "user_repo", "path": "notes.txt"},
    )

    _clean_wave(monkeypatch)  # the golden pool's seats answer clean through the real custody layer
    preflight = json.loads(registry.execute("skill_preflight", {"skill": "draft"}))
    assert preflight.get("ok") is True, preflight
    registry.execute("skill_review", {"skill": "draft"})
    persisted = load_review_state(data, "draft")
    assert persisted.status == "clean"


def test_selected_manifestless_refuses_grouping_unknown_collision_and_escape(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    checkout = tmp_path / "checkout"
    repo.mkdir()
    data.mkdir()
    child = checkout / "group" / "child"
    child.mkdir(parents=True)
    (child / "SKILL.md").write_text("# child\n", encoding="utf-8")

    grouping = _publish_registry(repo, data, checkout, "group", monkeypatch)
    grouping_list = grouping.execute(
        "list_files",
        {"root": "skill_payload", "skill_name": "group", "path": "."},
    )
    grouping_write = grouping.execute(
        "write_file",
        {
            "root": "skill_payload",
            "skill_name": "group",
            "path": "SKILL.md",
            "content": "# group\n",
        },
    )
    assert "not found" in grouping_list.lower()
    assert "not found" in grouping_write.lower()
    assert not (checkout / "group" / "SKILL.md").exists()

    unknown = _publish_registry(repo, data, checkout, "ghost", monkeypatch)
    unknown_result = unknown.execute(
        "write_file",
        {
            "root": "skill_payload",
            "skill_name": "ghost",
            "path": "SKILL.md",
            "content": "# ghost\n",
        },
    )
    assert "not found" in unknown_result.lower()
    assert not (checkout / "ghost").exists()

    leaf = checkout / "draft"
    leaf.mkdir()
    (leaf / "notes.txt").write_text("safe\n", encoding="utf-8")
    collision = _skill(data, "draft", bucket="external")
    registry = _publish_registry(repo, data, checkout, "draft", monkeypatch)
    collision_result = registry.execute(
        "list_files",
        {"root": "skill_payload", "skill_name": "draft", "path": "."},
    )
    assert "collision" in collision_result.lower()
    assert collision.is_dir()

    escape_leaf = checkout / "escape"
    escape_leaf.mkdir()
    (escape_leaf / "notes.txt").write_text("safe\n", encoding="utf-8")
    escape_registry = _publish_registry(repo, data, checkout, "escape", monkeypatch)
    escape_result = escape_registry.execute(
        "read_file",
        {"root": "skill_payload", "skill_name": "escape", "path": "../outside"},
    )
    assert "traversal is not allowed" in escape_result.lower()
