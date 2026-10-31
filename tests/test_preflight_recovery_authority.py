"""A run_script citation is usable by the caller that received it."""

import json

import pytest

from ouroboros.presence_authority import (
    PresenceCapabilityCeiling,
    PresenceResourceGrant,
    PresenceToolGrant,
    presence_ceiling_payload,
)
from ouroboros.task_results import write_task_result
from ouroboros.tool_call_log import invocation_scope, new_invocation, persist_dispatch_source
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _presence_contract(*tool_names):
    ceiling = PresenceCapabilityCeiling(
        skill_name="script-reader", skill_content_hash="a" * 64,
        profile_fingerprint="b" * 64, state_fingerprint="c" * 64,
        selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=3,
        tool_grants=tuple(PresenceToolGrant(name) for name in tool_names),
        resource_grants=(
            PresenceResourceGrant("task_drive", ("read", "shell")),
            PresenceResourceGrant("runtime_data", ("read",)),
        ),
        digest="0" * 64,
    )
    return {"capability_ceiling": presence_ceiling_payload(ceiling)}


@pytest.mark.parametrize("presence", [False, True])
def test_script_source_recipe_follows_registered_get_task_result_authority(tmp_path, monkeypatch, presence):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    task_id = "presence-script" if presence else "ordinary-script"
    metadata = {"presence": {"binding_id": "bound-room"}} if presence else {}
    write_task_result(data, task_id, "completed", result="done", metadata=metadata)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(
        repo_dir=repo, drive_root=data, budget_drive_root=str(data),
        task_id=task_id, task_metadata=metadata,
        task_contract=_presence_contract("run_script", "get_task_result") if presence else {},
    ))
    arguments = {"script": "print('source-marker')", **({"cwd": "task_drive"} if presence else {})}
    invocation = new_invocation("source-call", {}, 1)
    persist_dispatch_source(
        {"budget_drive_root": str(data)}, data / "logs", task_id,
        invocation, "run_script", arguments,
    )
    with invocation_scope(invocation):
        script_result = registry.execute_result("run_script", arguments)
    assert script_result.status == "ok", script_result.text
    fact = script_result.meta["script_source"]
    assert fact["status"] == "ready"
    recipe = fact["source_read"]
    assert recipe["tool"] == "get_task_result"
    assert recipe["arguments"].get("presence_scope") == ("own_binding" if presence else None)
    assert all(f"{key}={value!r}" in script_result.text for key, value in recipe["arguments"].items())

    if presence:
        unscoped = {key: value for key, value in recipe["arguments"].items() if key != "presence_scope"}
        own = registry.execute_result("get_task_result", unscoped)
        assert own.status == "ok"  # existing task-result authority admits this own task
        write_task_result(
            data, "foreign-script", "completed", result="private",
            metadata={"presence": {"binding_id": "another-room"}},
        )
        foreign = registry.execute_result("get_task_result", {
            **recipe["arguments"], "task_id": "foreign-script",
        })
        assert foreign.status == "blocked" and "another-room" not in foreign.text
        registry._ctx.task_contract = _presence_contract("run_script")
        no_grant = registry.execute_result(recipe["tool"], recipe["arguments"])
        assert no_grant.status == "blocked" and "PRESENCE_CAPABILITY_BLOCKED" in no_grant.text
        registry._ctx.task_contract = _presence_contract("run_script", "get_task_result")

    metadata_result = registry.execute_result(recipe["tool"], recipe["arguments"])
    assert metadata_result.status == "ok", metadata_result.text
    complete_chars = json.loads(metadata_result.text)["call_source"]["complete_chars"]
    bounded = {
        **recipe["arguments"],
        "source_start_char": 0,
        "source_end_char": min(complete_chars, 8192),
    }
    recovered = registry.execute_result(recipe["tool"], bounded)
    assert recovered.status == "ok", recovered.text
    assert "source-marker" in json.loads(recovered.text)["call_source"]["text"]
