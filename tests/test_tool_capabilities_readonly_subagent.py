"""What a local read-only subagent may reach.

Split verbatim out of ``tests/test_tool_capabilities.py`` by theme. This
module owns the read-only subagent profile boundary: forbidden tools at
execute time, the enabled extension tool it may still call, the
allowed-resources block on web/external tools, and the secret-file,
task-drive and skill-payload filters on its data and repo reads.
"""
import os
import pathlib


def test_local_readonly_subagent_execute_blocks_forbidden_tools(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    import ouroboros.mcp_client as mcp_client

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(
        ToolContext(
            repo_dir=tmp_path,
            drive_root=tmp_path,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    assert registry.get_schema_by_name("write_file") is None
    assert registry.get_schema_by_name("enable_tools") is None
    assert registry.get_schema_by_name("schedule_subagent") is not None
    # switch_model changes COGNITIVE POWER, not authority: a child that started cheap and
    # found the work harder raises its own strength, and nothing about its sandbox moves.
    # It was on the blocked list until v6.87.7 purely because power and authority were
    # conflated; a read-only child stays read-only at any model.
    assert registry.get_schema_by_name("switch_model") is not None
    assert "LOCAL_READONLY_SUBAGENT_BLOCKED" not in registry.execute("switch_model", {})
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MCP touched")))
    assert "LOCAL_READONLY_SUBAGENT_BLOCKED" not in registry.execute("list_files", {"path": "."})
    assert registry.get_schema_by_name("vcs_status") is not None
    assert "TOOL_ACCESS_BLOCKED" not in registry.execute("vcs_status", {"root": "system_repo"})
    blocked_tools = [
        "write_file",
        "edit_text",
        "knowledge_write",
        "update_scratchpad",
        "update_identity",
        "commit_reviewed",
        "advisory_review",
        "task_acceptance_review",
        "skill_review",
        "request_restart",
        "enable_tools",
        "run_command",
        "skill_exec",
        "list_skills",
    ]
    for name in blocked_tools:
        assert registry.get_schema_by_name(name) is None
        assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in registry.execute(name, {})


def test_local_readonly_subagent_allows_enabled_extension_tool(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from tests._shared import clean_extension_runtime_state
    from tests.test_extension_loader import _mark_isolated_deps_installed, _prepare_extension

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    plugin = (
        "def _lookup(ctx, query=''):\n"
        "    return 'external-ok:' + query\n"
        "def register(api):\n"
        "    api.register_tool('lookup', _lookup, description='External lookup', "
        "schema={'type': 'object', 'properties': {'query': {'type': 'string'}}}, timeout_sec=5)\n"
    )
    loaded, skills_repo, parent_drive = _prepare_extension(
        tmp_path,
        "research",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _mark_isolated_deps_installed(parent_drive, loaded)
    child_drive = tmp_path / "child-drive"
    child_drive.mkdir()
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=parent_drive)
    assert err is None, err
    tool_name = extension_loader.extension_surface_name("research", "lookup")
    assert extension_loader.is_extension_live("research", parent_drive, repo_path=str(skills_repo))
    assert not extension_loader.is_extension_live("research", child_drive, repo_path=str(skills_repo))
    assert extension_loader.get_tool(tool_name)["out_of_process"] is True
    repo_dir = pathlib.Path(__file__).resolve().parents[1]
    registry = ToolRegistry(repo_dir=repo_dir, drive_root=child_drive)
    try:
        registry.set_context(
            ToolContext(
                repo_dir=repo_dir,
                drive_root=child_drive,
                task_metadata={"budget_drive_root": str(parent_drive)},
                task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
            )
        )
        assert registry.get_schema_by_name(tool_name) is not None
        assert "external-ok:budget-root" in registry.execute(tool_name, {"query": "budget-root"})
    finally:
        clean_extension_runtime_state()


def test_allowed_resources_block_web_and_external_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from ouroboros import extension_loader
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    task_contract = build_task_contract({
        "id": "task-resources",
        "allowed_resources": {"web": "false", "network": "false"},
    })
    tool_name = extension_loader.extension_surface_name("research", "lookup")
    with extension_loader._lock:
        extension_loader._tools[tool_name] = {
            "name": tool_name,
            "handler": lambda ctx, **kwargs: "external-ok",
            "description": "External lookup",
            "schema": {"type": "object", "properties": {}},
            "timeout_sec": 5,
            "skill": "research",
        }
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
    try:
        registry.set_context(
            ToolContext(
                repo_dir=tmp_path / "repo",
                drive_root=tmp_path / "data",
                task_contract=task_contract,
                task_metadata={"task_contract": task_contract},
            )
        )
        assert task_contract["allowed_resources"] == {"web": False, "network": False}
        assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute("web_search", {"query": "x"})
        # VLM tools are first-class vision tools, not web egress. Benchmark isolation
        # withholds them by name via disabled_tools instead of relying on web=false.
        assert "RESOURCE_CONSTRAINT_BLOCKED" not in registry.execute("vlm_query", {"prompt": "x"})
        assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute(
            "vlm_query", {"prompt": "x", "image_url": "https://example.com/a.png"}
        )
        assert registry.get_schema_by_name(tool_name) is None
        assert tool_name not in {schema["function"]["name"] for schema in registry.schemas()}
        assert any(item.get("surface") == "extensions" and item.get("reason") == "resource_blocked" for item in registry.capability_omissions())
        blocked = registry.execute(tool_name, {})
        assert "RESOURCE_CONSTRAINT_BLOCKED" in blocked
        assert "network=false" in blocked

        alias_contract = build_task_contract({
            "id": "task-resource-aliases",
            "allowed_resources": {"allow_network": "false"},
        })
        registry.set_context(
            ToolContext(
                repo_dir=tmp_path / "repo",
                drive_root=tmp_path / "data",
                task_contract=alias_contract,
                task_metadata={"task_contract": alias_contract},
            )
        )
        assert alias_contract["allowed_resources"] == {"allow_network": False}
        assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute("web_search", {"query": "x"})
    finally:
        with extension_loader._lock:
            extension_loader._tools.pop(tool_name, None)


def test_local_readonly_subagent_data_read_denies_secret_files(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    (tmp_path / "settings.json").write_text('{"OPENROUTER_API_KEY":"secret"}', encoding="utf-8")
    (tmp_path / "settings.tmp").write_text('{"OPENROUTER_API_KEY":"secret"}', encoding="utf-8")
    (tmp_path / ".settings.json.tmp.123").write_text('{"OPENROUTER_API_KEY":"secret"}', encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "prod.env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "state" / "skills" / "weather").mkdir(parents=True)
    (tmp_path / "state" / "skills" / "weather" / "grants.json").write_text("{}", encoding="utf-8")
    (tmp_path / "state" / "skills" / "weather" / ".grants.json.tmp.123").write_text("{}", encoding="utf-8")
    (tmp_path / "state" / "skills" / "weather" / "review.json.lock").write_text("{}", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "events.jsonl").write_text("{}", encoding="utf-8")
    try:
        os.symlink("settings.json", tmp_path / "alias.txt")
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(tmp_path / "settings.json", tmp_path / "hardlink.txt")
    except (OSError, NotImplementedError):
        pass

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(
        ToolContext(
            repo_dir=tmp_path,
            drive_root=tmp_path,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    blocked = registry.execute("read_file", {"root": "runtime_data", "path": "settings.json"})
    assert "DATA_READ_BLOCKED" in blocked
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": "settings.tmp"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": ".settings.json.tmp.123"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": ".env.local"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": "prod.env"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": "state/skills/weather/.grants.json.tmp.123"})
    assert "DATA_READ_BLOCKED" in registry.execute("read_file", {"root": "runtime_data", "path": "state/skills/weather/review.json.lock"})
    alias_result = registry.execute("read_file", {"root": "runtime_data", "path": "alias.txt"})
    if (tmp_path / "alias.txt").exists():
        assert "DATA_READ_BLOCKED" in alias_result
    hardlink_result = registry.execute("read_file", {"root": "runtime_data", "path": "hardlink.txt"})
    if (tmp_path / "hardlink.txt").exists():
        assert "DATA_READ_BLOCKED" in hardlink_result
    listing = registry.execute("list_files", {"root": "runtime_data", "path": "."})
    assert "settings.json" not in listing
    assert "settings.tmp" not in listing
    assert ".settings.json.tmp.123" not in listing
    assert ".env.local" not in listing
    assert "prod.env" not in listing
    assert "alias.txt" not in listing
    assert "hardlink.txt" not in listing
    assert "secret/control" in listing
    skill_state_listing = registry.execute("list_files", {"root": "runtime_data", "path": "state/skills/weather"})
    assert "grants.json" not in skill_state_listing
    assert ".grants.json.tmp.123" not in skill_state_listing
    assert "review.json.lock" not in skill_state_listing
    assert "secret/control" in skill_state_listing
    assert "DATA_LIST_BLOCKED" in registry.execute("list_files", {"root": "runtime_data", "path": "state/skills/weather/grants.json"})
    assert "DATA_LIST_BLOCKED" in registry.execute("list_files", {"root": "runtime_data", "path": "state/skills/weather/.grants.json.tmp.123"})
    readable = registry.execute("read_file", {"root": "runtime_data", "path": "logs/events.jsonl"})
    assert "{}" in readable


def test_local_readonly_subagent_repo_read_denies_secret_files(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / ".git").mkdir(parents=True)
    data.mkdir()
    (repo / ".git" / "credentials").write_text("https://token@example.invalid\n", encoding="utf-8")
    (repo / ".git" / "config").write_text("[credential]\n", encoding="utf-8")
    (repo / ".env.local").write_text("TOKEN=secret\nLEAK_MARKER=env\n", encoding="utf-8")
    (repo / "auth_token.json").write_text('{"token":"TOKEN_LEAK"}\n', encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "public.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "src" / "skill_token.py").write_text("TOKEN_NAME = 'safe source symbol'\n", encoding="utf-8")
    try:
        os.symlink(".git/credentials", repo / "alias.txt")
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(repo / ".git" / "credentials", repo / "hardlink.txt")
    except (OSError, NotImplementedError):
        pass

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(
        ToolContext(
            repo_dir=repo,
            drive_root=data,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    assert "REPO_READ_BLOCKED" in registry.execute("read_file", {"path": ".git/credentials"})
    assert "READ_FILE_BLOCKED" in registry.execute("read_file", {"root": "system_repo", "path": ".git/credentials"})
    assert "REPO_READ_BLOCKED" in registry.execute("read_file", {"path": ".git/config"})
    assert "READ_FILE_BLOCKED" in registry.execute("read_file", {"root": "system_repo", "path": ".git/config"})
    assert "REPO_READ_BLOCKED" in registry.execute("read_file", {"path": ".env.local"})
    assert "REPO_READ_BLOCKED" in registry.execute("read_file", {"path": "auth_token.json"})
    alias_result = registry.execute("read_file", {"path": "alias.txt"})
    if (repo / "alias.txt").exists():
        assert "REPO_READ_BLOCKED" in alias_result
    hardlink_result = registry.execute("read_file", {"path": "hardlink.txt"})
    if (repo / "hardlink.txt").exists():
        assert "REPO_READ_BLOCKED" in hardlink_result
    listing = registry.execute("list_files", {"path": "."})
    assert ".git/" not in listing
    assert ".env.local" not in listing
    assert "auth_token.json" not in listing
    assert "alias.txt" not in listing
    assert "hardlink.txt" not in listing
    assert "src/" in listing
    assert "secret/control" in listing
    system_listing = registry.execute("list_files", {"root": "system_repo", "path": "."})
    assert ".git/" not in system_listing
    assert "auth_token.json" not in system_listing
    assert "secret/control" in system_listing
    assert "REPO_LIST_BLOCKED" in registry.execute("list_files", {"path": ".git"})
    readable = registry.execute("read_file", {"path": "src/public.py"})
    assert "print('ok')" in readable
    source_with_token_name = registry.execute("read_file", {"path": "src/skill_token.py"})
    assert "safe source symbol" in source_with_token_name
    secret_search = registry.execute("search_code", {"query": "TOKEN_LEAK"})
    assert "No matches found" in secret_search
    assert "auth_token.json:" not in secret_search
    assert "SEARCH_BLOCKED" in registry.execute("search_code", {"query": "TOKEN_LEAK", "path": "auth_token.json"})
    public_search = registry.execute("search_code", {"query": "safe source symbol"})
    assert "src/skill_token.py" in public_search
    digest = registry.execute("query_code", {"op": "digest"})
    assert "auth_token.json" not in digest
    assert ".env.local" not in digest
    assert "src/skill_token.py" in digest
    cached = list((data / "state" / "code_intel").glob("*/inventory.json"))
    assert not cached


def test_local_readonly_subagent_task_drive_and_skill_payload_filters(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (data / "settings.json").write_text('{"OPENROUTER_API_KEY":"secret"}', encoding="utf-8")
    (data / "skills" / "external" / "alpha").mkdir(parents=True)
    (data / "skills" / "external" / "alpha" / "SKILL.md").write_text("hello", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(
        ToolContext(
            repo_dir=repo,
            drive_root=data,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    assert "READ_FILE_BLOCKED" in registry.execute("read_file", {"root": "task_drive", "path": "settings.json"})
    traversal = registry.execute(
        "read_file",
        {"root": "skill_payload", "bucket": "external", "skill_name": "../../settings.json", "path": "."},
    )
    assert "TOOL_ACCESS_BLOCKED" in traversal or "READ_FILE_ERROR" in traversal or "TOOL_ARG_ERROR" in traversal
    skill_payload_read = registry.execute(
        "read_file",
        {"root": "skill_payload", "bucket": "external", "skill_name": "alpha", "path": "SKILL.md"},
    )
    # v6.70.0 (owner-approved): read-only scouts may READ skill payloads — a scout
    # sent to review a skill used to be structurally blind to it. Mutation stays
    # blocked (pinned in test_owner_facing_honesty.py).
    assert "TOOL_ACCESS_BLOCKED" not in skill_payload_read
    assert "hello" in skill_payload_read
