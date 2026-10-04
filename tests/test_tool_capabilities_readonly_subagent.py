"""What a local read-only subagent may reach.

Split verbatim out of ``tests/test_tool_capabilities.py`` by theme. This
module owns the read-only subagent profile boundary: forbidden tools at
execute time, the enabled extension tool it may still call, the
allowed-resources block on web/external tools, parent-equivalent file reads,
and task-drive and skill-payload admission.
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
    # Memory written in the child's own name is open (signed with its focus), chronicle
    # pages only as its drafts (tests/test_child_chronicle_drafts.py); identity and
    # scratchpad stay with the integrating parent.
    for name in ("knowledge_write", "memory_mark", "memory_read", "chronicle_write"):
        assert registry.get_schema_by_name(name) is not None
    blocked_tools = [
        "write_file",
        "edit_text",
        "update_scratchpad",
        "update_identity",
        "commit_reviewed",
        "preflight_review",
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


def test_local_readonly_subagent_data_reads_and_listings_match_parent(tmp_path):
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

    parent = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    for path in ("settings.json", "settings.tmp", ".settings.json.tmp.123", ".env.local", "prod.env",
                 "state/skills/weather/grants.json", "state/skills/weather/.grants.json.tmp.123",
                 "state/skills/weather/review.json.lock", "logs/events.jsonl", "alias.txt", "hardlink.txt"):
        args = {"root": "runtime_data", "path": path}
        child_result = registry.execute("read_file", args)
        assert child_result == parent.execute("read_file", args)
        if path in {"settings.json", "settings.tmp", ".settings.json.tmp.123", ".env.local", "prod.env"}:
            assert "secret" in child_result
    for path in (".", "state/skills/weather", "state/skills/weather/grants.json"):
        args = {"root": "runtime_data", "path": path}
        assert registry.execute("list_files", args) == parent.execute("list_files", args)
    listing = registry.execute("list_files", {"root": "runtime_data", "path": "."})
    for name in ("settings.json", "settings.tmp", ".settings.json.tmp.123", ".env.local", "prod.env"):
        assert name in listing
    assert "hidden from this subagent" not in listing


def test_local_readonly_subagent_repo_reads_include_git_and_credential_named_files(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / ".git").mkdir(parents=True)
    data.mkdir()
    (repo / ".git" / "credentials").write_text("https://token@example.invalid\n", encoding="utf-8")
    (repo / ".git" / "config").write_text("[credential]\n", encoding="utf-8")
    (repo / ".env.local").write_text("TOKEN=secret\nLEAK_MARKER=env\n", encoding="utf-8")
    (repo / "auth_token.json").write_text('{"token":"PROJECT_TOKEN_REPORT"}\n', encoding="utf-8")
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

    for root in ("active_workspace", "system_repo"):
        assert "https://token@example.invalid" in registry.execute("read_file", {"root": root, "path": ".git/credentials"})
        assert "[credential]" in registry.execute("read_file", {"root": root, "path": ".git/config"})
        assert "TOKEN=secret" in registry.execute("read_file", {"root": root, "path": ".env.local"})
        listing = registry.execute("list_files", {"root": root, "path": "."})
        assert ".git/" in listing and ".env.local" in listing and "auth_token.json" in listing
        assert "secret/control" not in listing
    for path in ("alias.txt", "hardlink.txt"):
        if (repo / path).exists():
            assert "https://token@example.invalid" in registry.execute("read_file", {"path": path})
    assert "credentials" in registry.execute("list_files", {"path": ".git"})
    assert "PROJECT_TOKEN_REPORT" in registry.execute("read_file", {"path": "auth_token.json"})
    assert "print('ok')" in registry.execute("read_file", {"path": "src/public.py"})
    assert ".env.local:" in registry.execute("search_code", {"query": "LEAK_MARKER"})
    assert "auth_token.json:" in registry.execute("search_code", {"query": "PROJECT_TOKEN_REPORT", "path": "auth_token.json"})
    assert "src/skill_token.py" in registry.execute("search_code", {"query": "safe source symbol"})
    digest = registry.execute("query_code", {"op": "digest"})
    assert "src/skill_token.py" in digest
    assert not list((data / "state" / "code_intel").glob("*/inventory.json"))
    parent = ToolRegistry(repo_dir=repo, drive_root=data)
    assert digest == parent.execute("query_code", {"op": "digest"})


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

    from ouroboros.tool_access import resource_root_path

    task_root = resource_root_path(registry._ctx, "task_drive")
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "settings.json").write_text("ordinary task settings", encoding="utf-8")
    (task_root / "auth_token.json").write_text("ordinary task output", encoding="utf-8")
    assert "ordinary task settings" in registry.execute("read_file", {"root": "task_drive", "path": "settings.json"})
    assert "ordinary task output" in registry.execute("read_file", {"root": "task_drive", "path": "auth_token.json"})
    admitted = registry.execute_result("read_file", {"root": "runtime_data", "path": "settings.json"})
    assert admitted.status == "ok" and '"secret"' in admitted.text
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
