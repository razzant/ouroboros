import json
from pathlib import Path

import pytest

from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.presence_authority import (
    PresenceAuthorityError,
    build_presence_capability_ceiling,
    presence_ceiling_allows_binding,
    presence_ceiling_allows_delegated_read,
    presence_ceiling_allows_delegated_surface,
    presence_ceiling_allows_tool,
    presence_ceiling_from_payload,
    presence_ceiling_payload,
)
from ouroboros.presence_capabilities import (
    PresenceArgumentBinding,
    PresenceProfileResolution,
    PresenceResourceTarget,
    PresenceScriptTarget,
    PresenceSelection,
    PresenceState,
    PresenceToolTarget,
    presence_state_fingerprint,
)
from ouroboros.presence_runtime import ResolvedPresenceRuntime
from ouroboros.tool_access import ResolvedResourceBinding
from ouroboros.tools.registry import ToolContext, ToolEntry, ToolRegistry


def _resolution(*targets, required=True):
    selections = tuple(
        PresenceSelection(str(index + 1) * 64, target) for index, target in enumerate(targets)
    )
    return PresenceProfileResolution(
        active=selections,
        missing_required=() if required else (object(),),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=required,
    )


def test_ceiling_compiles_exact_tools_scripts_resources_and_digest():
    state = PresenceState()
    ceiling = build_presence_capability_ceiling(
        skill_name="moderator",
        skill_content_hash="c" * 64,
        state_fingerprint=presence_state_fingerprint(state),
        resolution=_resolution(
            PresenceToolTarget("builtin", "chat_history"),
            PresenceScriptTarget("calendar", "scripts/create.py"),
            PresenceResourceTarget("active_workspace", ("read", "write"), "shared"),
        ),
    )

    assert [grant.name for grant in ceiling.tool_grants] == ["chat_history", "skill_exec"]
    script = next(grant for grant in ceiling.tool_grants if grant.name == "skill_exec")
    assert [(item.argument_path, item.static_value) for item in script.bindings] == [
        (("skill",), "calendar"),
        (("script",), "scripts/create.py"),
    ]
    payload = presence_ceiling_payload(ceiling)
    assert payload["digest"] == ceiling.digest
    assert len(payload["digest"]) == 64
    assert presence_ceiling_from_payload(payload) == ceiling
    assert presence_ceiling_allows_tool(ceiling, "chat_history")
    assert not presence_ceiling_allows_tool(ceiling, "run_command")


def test_missing_required_selection_refuses_admission():
    with pytest.raises(PresenceAuthorityError) as caught:
        build_presence_capability_ceiling(
            skill_name="moderator",
            skill_content_hash="c" * 64,
            state_fingerprint="d" * 64,
            resolution=_resolution(required=False),
        )
    assert caught.value.code == "presence_authority_missing_required"


def test_resolved_binding_must_fit_root_operation_and_prefix(tmp_path):
    ceiling = build_presence_capability_ceiling(
        skill_name="moderator",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=_resolution(
            PresenceResourceTarget("active_workspace", ("read",), "shared")
        ),
    )
    allowed = ResolvedResourceBinding(
        profile="operator_control",  # type: ignore[arg-type]
        root="active_workspace",
        operation="read",
        base_path=tmp_path,
        target_path=tmp_path / "shared" / "note.md",
        source="active_workspace",
        skill_name="",
        state_drive_root=Path(tmp_path),
    )
    wrong_prefix = ResolvedResourceBinding(
        **{**allowed.__dict__, "target_path": tmp_path / "private" / "note.md"}
    )
    wrong_operation = ResolvedResourceBinding(
        **{**allowed.__dict__, "operation": "write"}
    )

    assert presence_ceiling_allows_binding(ceiling, allowed)
    assert not presence_ceiling_allows_binding(ceiling, wrong_prefix)
    assert not presence_ceiling_allows_binding(ceiling, wrong_operation)


def test_resolved_skill_binding_must_fit_granted_bucket(tmp_path):
    ceiling = build_presence_capability_ceiling(
        skill_name="moderator",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=_resolution(
            PresenceResourceTarget(
                "skill_payload",
                ("read",),
                ".",
                bucket="external",
                skill_name="alpha",
            )
        ),
    )
    allowed = ResolvedResourceBinding(
        profile="operator_control",  # type: ignore[arg-type]
        root="skill_payload",
        operation="read",
        base_path=tmp_path,
        target_path=tmp_path / "SKILL.md",
        source="external",
        skill_name="alpha",
        state_drive_root=Path(tmp_path),
    )
    wrong_bucket = ResolvedResourceBinding(**{**allowed.__dict__, "source": "native"})

    assert presence_ceiling_allows_binding(ceiling, allowed)
    assert not presence_ceiling_allows_binding(ceiling, wrong_bucket)


def test_mutative_descendants_require_matching_selected_surface_root(tmp_path):
    ceiling = build_presence_capability_ceiling(
        skill_name="moderator",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=_resolution(
            PresenceResourceTarget("system_repo", ("write",), ".")
        ),
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}

    assert presence_ceiling_allows_delegated_surface(ctx, "self_worktree")
    assert not presence_ceiling_allows_delegated_surface(ctx, "external_workspace")
    assert not presence_ceiling_allows_delegated_surface(ctx, "genesis")


def test_payload_digest_and_shape_are_fail_closed():
    ceiling = build_presence_capability_ceiling(
        skill_name="moderator",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=_resolution(PresenceToolTarget("builtin", "chat_history")),
    )
    payload = presence_ceiling_payload(ceiling)
    payload["tools"][0]["name"] = "run_command"
    with pytest.raises(PresenceAuthorityError) as caught:
        presence_ceiling_from_payload(payload)
    assert caught.value.code == "presence_authority_digest_mismatch"


def test_task_contract_preserves_only_verified_ceiling():
    ceiling = build_presence_capability_ceiling(
        skill_name="moderator",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=_resolution(PresenceToolTarget("builtin", "chat_history")),
    )
    payload = presence_ceiling_payload(ceiling)
    contract = build_task_contract({"text": "hello", "task_contract": {"capability_ceiling": payload}})
    assert contract["capability_ceiling"] == payload

    tampered = json.loads(json.dumps(payload))
    tampered["tools"][0]["name"] = "run_command"
    with pytest.raises(PresenceAuthorityError):
        build_task_contract({"text": "hello", "task_contract": {"capability_ceiling": tampered}})


def test_registry_filters_schema_dispatch_and_resolved_targets(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / "shared").mkdir(parents=True)
    data.mkdir()
    (repo / "shared" / "allowed.txt").write_text("ok", encoding="utf-8")
    (repo / "private.txt").write_text("no", encoding="utf-8")
    ceiling = build_presence_capability_ceiling(
        skill_name="moderator",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=_resolution(
            PresenceToolTarget("builtin", "read_file"),
            PresenceResourceTarget("active_workspace", ("read",), "shared"),
        ),
    )
    contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_contract=contract)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)

    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert names == {"presence_finish", "presence_cancel_work", "read_file"}
    assert "PRESENCE_CAPABILITY_BLOCKED" in registry.execute(
        "run_command", {"command": "pwd"}
    )
    assert "PRESENCE_RESOURCE_BLOCKED" in registry.execute(
        "read_file", {"root": "active_workspace", "path": "private.txt"}
    )
    assert registry.execute(
        "read_file", {"root": "active_workspace", "path": "shared/allowed.txt"}
    ).endswith("\nok")


def test_presence_argument_bindings_override_model_supplied_values(tmp_path):
    selection = PresenceSelection(
        "1" * 64,
        PresenceToolTarget("builtin", "bound_tool"),
        (
            PresenceArgumentBinding(("person_id",), "actor", source_path=("platform_actor_id",)),
            PresenceArgumentBinding(("destination",), "destination", source_path=("conversation_id",)),
            PresenceArgumentBinding(("mode",), "static", static_value="append"),
        ),
    )
    resolution = PresenceProfileResolution(
        active=(selection,),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="moderator",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=resolution,
    )
    ctx = ToolContext(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
        task_metadata={
            "presence": {
                "event": {
                    "actor": {"platform_actor_id": "u-7"},
                    "destination": {"conversation_id": "room-1"},
                }
            }
        },
    )
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.register(ToolEntry(
        name="bound_tool",
        schema={
            "name": "bound_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "string"},
                    "destination": {"type": "string"},
                    "mode": {"type": "string"},
                },
                "required": ["person_id", "destination", "mode"],
            },
        },
        handler=lambda _ctx, **kwargs: json.dumps(kwargs, sort_keys=True),
    ))
    registry.set_context(ctx)

    raw_result = registry.execute(
        "bound_tool",
        {"person_id": "spoof", "destination": "wrong", "mode": "overwrite"},
    )
    # #447 H1: host notes (safety reminder) trail the payload now.
    result = json.loads(raw_result.rsplit("\n---\n", 1)[-1].splitlines()[0])
    assert result == {"destination": "room-1", "mode": "append", "person_id": "u-7"}


def test_presence_delegate_read_requires_the_whole_active_repo_root(tmp_path):
    def context(resource):
        ceiling = build_presence_capability_ceiling(
            skill_name="moderator",
            skill_content_hash="c" * 64,
            state_fingerprint="d" * 64,
            resolution=_resolution(resource),
        )
        return ToolContext(
            repo_dir=tmp_path,
            drive_root=tmp_path,
            task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
        )

    assert presence_ceiling_allows_delegated_read(
        context(PresenceResourceTarget("system_repo", ("read",), "."))
    )
    assert not presence_ceiling_allows_delegated_read(
        context(PresenceResourceTarget("system_repo", ("read",), "docs"))
    )
    assert not presence_ceiling_allows_delegated_read(
        context(PresenceResourceTarget("active_workspace", ("read",), "."))
    )
