"""Registered regressions for retained targets and descriptor eligibility."""
import ast

import pytest

from tests.test_canonical_runtime_reads import actor, world as _world

world = _world


@pytest.mark.parametrize("profile", ["root", "local_readonly_subagent", "acting_subagent"])
def test_directory_recipe_keeps_canonical_target(world, profile):
    _, registry = actor(world, profile)
    for base, name in ((world.drive_root, "canonical-only.txt"), (world.fork, "execution-only.txt")):
        (base / "only").mkdir()
        (base / "only" / name).write_text("marker")
    response = registry.execute_result("read_file", {"path": str(world.drive_root / "only")})
    recipe = next(line.strip() for line in response.text.splitlines() if line.strip().startswith("list_files("))
    call = ast.parse(recipe, mode="eval").body
    args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    listed = registry.execute_result("list_files", args)
    assert listed.status == "ok" and "canonical-only.txt" in listed.text
    assert "execution-only.txt" not in listed.text


def test_all_schema_readers_agree_on_network_gate(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.tools.tool_discovery import enable_tools
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-token")
    reg = ToolRegistry(tmp_path / "repo", tmp_path / "data")
    reg.set_context(ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data", task_id="actor"))
    for allowed in (False, True):
        reg._ctx.task_contract = {"allowed_resources": {"network": allowed}}
        name = "github_read"
        for core_only in (False, True):
            assert (name in {row["function"]["name"] for row in reg.schemas(core_only=core_only)}) is allowed
        assert (reg.get_schema_by_name(name) is not None) is allowed
        assert (reg.policy_hidden_reason(name) is None) is allowed
        answer = enable_tools(reg, reg._ctx, name, [])
        assert ("resource_blocked" in answer) is not allowed
