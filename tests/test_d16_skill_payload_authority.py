"""D16 marker-aware payload authority and selected manifestless recovery."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.presence_authority import build_presence_capability_ceiling, presence_ceiling_payload
from ouroboros.presence_capabilities import (
    PresenceProfileResolution,
    PresenceResourceTarget,
    PresenceSelection,
    PresenceToolTarget,
)
from ouroboros.presence_runtime import ResolvedPresenceRuntime
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


def _seeded_native_payload(data: pathlib.Path, name: str = "seeded-native") -> pathlib.Path:
    payload = _skill(data, name, seeded=True)
    (payload / ".clawhub.json").write_text('{"origin":"catalog"}\n', encoding="utf-8")
    (payload / "node_modules" / "dep").mkdir(parents=True)
    (payload / "node_modules" / "dep" / "index.js").write_text(
        "export const dependency = true;\n", encoding="utf-8"
    )
    return payload


@pytest.mark.parametrize("mode", ["light", "advanced", "pro"])
def test_markerless_native_is_user_managed_in_every_runtime_mode(
    mode,
    tmp_path,
    monkeypatch,
):
    registry, _repo, data = _registry(tmp_path, monkeypatch, mode=mode)
    user_payload = _skill(data, "user-native")
    seeded_payload = _skill(data, "seeded-native", seeded=True)
    selector = {
        "root": "skill_payload",
        "bucket": "external",
        "skill_name": "user-native",
    }

    catalogue = json.loads(registry.execute("list_skills", {}))
    sources = {row["name"]: row["source"] for row in catalogue["skills"]}
    assert sources["user-native"] == "external"
    assert sources["seeded-native"] == "native"

    read = registry.execute("read_file", {**selector, "path": "notes.txt"})
    written = registry.execute(
        "write_file",
        {**selector, "path": "written.txt", "content": "written\n"},
    )
    edited = registry.execute(
        "edit_text",
        {
            **selector,
            "path": "notes.txt",
            "old_str": "old",
            "new_str": "edited",
        },
    )
    shell = registry.execute(
        "run_command",
        {
            "cwd": "skill_payload",
            "bucket": "external",
            "skill_name": "user-native",
            "cmd": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('shell.txt').write_text('shell\\n')",
            ],
        },
    )

    assert "old" in read
    assert written.startswith("OK:"), written
    assert "Replaced" in edited
    assert "SHELL_CWD_BLOCKED" not in shell, shell
    assert (user_payload / "written.txt").read_text(encoding="utf-8") == "written\n"
    assert (user_payload / "notes.txt").read_text(encoding="utf-8") == "edited\n"
    assert (user_payload / "shell.txt").read_text(encoding="utf-8") == "shell\n"

    sidecar_write = registry.execute(
        "write_file",
        {**selector, "path": ".seed-origin", "content": "forged\n"},
    )
    (user_payload / ".clawhub.json").write_text('{"origin":"real"}\n', encoding="utf-8")
    sidecar_edit = registry.execute(
        "edit_text",
        {
            **selector,
            "path": ".clawhub.json",
            "old_str": "real",
            "new_str": "forged",
        },
    )
    sidecar_shell = registry.execute(
        "run_command",
        {
            "cwd": "skill_payload",
            "bucket": "external",
            "skill_name": "user-native",
            "cmd": ["touch", ".seed-origin"],
        },
    )
    existing_sidecar_shell = registry.execute(
        "run_command",
        {
            "cwd": "skill_payload",
            "bucket": "external",
            "skill_name": "user-native",
            "cmd": ["touch", ".clawhub.json"],
        },
    )
    assert "BLOCKED" in sidecar_write
    assert "BLOCKED" in sidecar_edit
    assert "SKILL_PAYLOAD_BLOCKED" in sidecar_shell
    assert "SKILL_PAYLOAD_BLOCKED" in existing_sidecar_shell
    assert not (user_payload / ".seed-origin").exists()
    assert "real" in (user_payload / ".clawhub.json").read_text(encoding="utf-8")

    seeded_selector = {
        "root": "skill_payload",
        "bucket": "native",
        "skill_name": "seeded-native",
    }
    assert "old" in registry.execute("read_file", {**seeded_selector, "path": "notes.txt"})
    for result in (
        registry.execute(
            "write_file",
            {**seeded_selector, "path": "bad.txt", "content": "bad\n"},
        ),
        registry.execute(
            "edit_text",
            {
                **seeded_selector,
                "path": "notes.txt",
                "old_str": "old",
                "new_str": "bad",
            },
        ),
        registry.execute(
            "run_command",
            {
                "cwd": "skill_payload",
                "bucket": "native",
                "skill_name": "seeded-native",
                "cmd": [sys.executable, "-c", "print('must not run')"],
            },
        ),
    ):
        assert "read/review only" in result, result
    assert not (seeded_payload / "bad.txt").exists()
    assert (seeded_payload / "notes.txt").read_text(encoding="utf-8") == "old\n"


def test_direct_operator_can_read_native_payload_but_not_mutate_or_forge_sidecars(
    tmp_path,
    monkeypatch,
):
    registry, repo, data = _registry(tmp_path, monkeypatch, mode="advanced")
    payload = _seeded_native_payload(data)
    registry._ctx = ToolContext(
        repo_dir=repo,
        drive_root=data,
        is_direct_chat=True,
    )
    selector = {
        "root": "skill_payload",
        "bucket": "native",
        "skill_name": "seeded-native",
    }

    read = registry.execute("read_file", {**selector, "path": ".seed-origin"})
    dependency = registry.execute(
        "read_file",
        {**selector, "path": "node_modules/dep/index.js"},
    )
    listing = registry.execute("list_files", {**selector, "path": "."})
    search = registry.execute(
        "search_code",
        {**selector, "path": ".", "query": "seeded-native"},
    )

    assert "launcher-seed" in read
    assert "dependency" in dependency
    assert "node_modules/" in listing and ".seed-origin" in listing
    assert "SKILL.md" in search

    ordinary_write = registry.execute(
        "write_file",
        {**selector, "path": "new.txt", "content": "must not write\n"},
    )
    sidecar_write = registry.execute(
        "write_file",
        {**selector, "path": ".seed-origin", "content": "forged\n"},
    )
    assert "SKILL_PAYLOAD_ARG_ERROR" in ordinary_write
    assert "SKILL_PAYLOAD_ARG_ERROR" in sidecar_write
    assert not (payload / "new.txt").exists()
    assert (payload / ".seed-origin").read_text(encoding="utf-8") == "launcher-seed\n"


def test_presence_bucket_ceiling_cannot_be_bypassed_by_native_read_overlay(tmp_path, monkeypatch):
    registry, repo, data = _registry(tmp_path, monkeypatch, mode="advanced")
    _seeded_native_payload(data, "seeded-native")
    resolution = PresenceProfileResolution(
        active=(
            PresenceSelection("1" * 64, PresenceToolTarget("builtin", "read_file")),
            PresenceSelection(
                "2" * 64,
                PresenceResourceTarget(
                    "skill_payload",
                    ("read",),
                    ".",
                    bucket="external",
                    skill_name="seeded-native",
                ),
            ),
        ),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="presence-native-overlay-test",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=resolution,
    )
    registry.set_context(
        ToolContext(
            repo_dir=repo,
            drive_root=data,
            task_constraint=TaskConstraint(mode="local_readonly_subagent"),
            task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
        )
    )

    result = registry.execute(
        "read_file",
        {
            "root": "skill_payload",
            "bucket": "native",
            "skill_name": "seeded-native",
            "path": "SKILL.md",
        },
    )
    assert "PRESENCE_RESOURCE_BLOCKED" in result


@pytest.mark.parametrize(
    "constraint",
    [
        TaskConstraint(mode="skill_repair", skill_name="seeded-native", payload_root="skills/native/seeded-native"),
        TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root="/tmp/acting-native"),
    ],
)
def test_native_payload_reads_follow_parent_while_repair_constraint_stays(
    constraint,
    tmp_path,
):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    _seeded_native_payload(data)
    if constraint.mode == "acting_subagent":
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        constraint = TaskConstraint(
            mode="acting_subagent",
            surface="external_workspace",
            write_root=str(workspace),
        )
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_constraint=constraint)

    arguments = dict(root="skill_payload", operation="read", path="SKILL.md",
                     bucket="native", skill_name="seeded-native")
    if constraint.mode == "acting_subagent":
        binding = build_resolved_resource_binding(ctx, **arguments)
        assert binding.target_path == data / "skills" / "native" / "seeded-native" / "SKILL.md"
        with pytest.raises(ValueError):
            build_resolved_resource_binding(ctx, **{**arguments, "operation": "write"})
    else:
        with pytest.raises(ValueError):
            build_resolved_resource_binding(ctx, **arguments)



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


@pytest.mark.parametrize("mode", ["advanced", "pro"])
@pytest.mark.parametrize(
    "sidecar",
    [
        ".clawhub.json",
        ".ouroboroshub.json",
        ".self_authored.json",
        ".seed-origin",
        "SKILL.openclaw.md",
    ],
)
def test_legacy_runtime_data_native_sidecars_remain_control_plane(
    mode,
    sidecar,
    tmp_path,
    monkeypatch,
):
    registry, _repo, data = _registry(tmp_path, monkeypatch, mode=mode)
    protected_payload = _skill(data, "legacy-control")
    ordinary_payload = _skill(data, "legacy-ordinary")
    protected_file = protected_payload / sidecar
    protected_file.write_text("control-old\n", encoding="utf-8")

    sidecar_edit = registry.execute(
        "edit_text",
        {
            "root": "runtime_data",
            "path": f"skills/native/legacy-control/{sidecar}",
            "old_str": "control-old",
            "new_str": "control-edited",
        },
    )
    sidecar_write = registry.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": f"skills/native/legacy-control/{sidecar}",
            "content": "control-overwritten\n",
        },
    )
    ordinary_edit = registry.execute(
        "edit_text",
        {
            "root": "runtime_data",
            "path": "skills/native/legacy-ordinary/notes.txt",
            "old_str": "old",
            "new_str": "edited",
        },
    )
    sidecar_read = registry.execute(
        "read_file",
        {
            "root": "runtime_data",
            "path": f"skills/native/legacy-control/{sidecar}",
        },
    )
    payload_listing = registry.execute(
        "list_files",
        {
            "root": "runtime_data",
            "path": "skills/native/legacy-control",
        },
    )

    assert "BLOCKED" in sidecar_edit
    assert "BLOCKED" in sidecar_write
    assert protected_file.read_text(encoding="utf-8") == "control-old\n"
    assert ordinary_edit.startswith("OK: edited"), ordinary_edit
    assert ordinary_payload.joinpath("notes.txt").read_text(encoding="utf-8") == "edited\n"
    assert "control-old" in sidecar_read
    assert sidecar in payload_listing


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
    from tests.test_skill_review_persist_guard import _pass_actor

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

    monkeypatch.setattr(
        "ouroboros.tools.review._handle_multi_model_review",
        lambda *_a, **_kw: json.dumps({"results": [_pass_actor("fake/a"), _pass_actor("fake/b")]}),
    )
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
