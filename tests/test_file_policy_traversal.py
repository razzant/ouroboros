"""File walks prepare locations once while keeping target identity live."""
from __future__ import annotations

from collections import Counter

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools import core_secret_paths
from tests.test_file_resource_consistency import environment as environment

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("mode", ["local_readonly_subagent", "acting_subagent"])
@pytest.mark.parametrize("tool,args", [
    ("list_files", {"path": "src", "max_entries": 200}),
    ("search_code", {"path": "src", "query": "visible_function"}),
    ("query_code", {"op": "symbols", "path": "src/module_0.py"}),
    ("query_code", {"op": "structural", "query": "FunctionDef", "path": "src"}),
])
def test_each_walk_prepares_locations_once_and_next_call_refreshes(environment, monkeypatch, mode, tool, args):
    from ouroboros import credential_shapes, tool_access

    reg, ctx, _home, repo, _data = environment
    ctx.task_constraint = TaskConstraint(mode=mode, surface="external_workspace", write_root=str(repo))
    (repo / "src").mkdir()
    counts = Counter()

    def counted(module, name):
        original = getattr(module, name)
        def run(*positional, **keywords):
            key = positional[1] if name == "resource_root_path" else name
            counts[key] += 1
            return original(*positional, **keywords)
        monkeypatch.setattr(module, name, run)

    counted(credential_shapes, "owner_credential_locations")
    counted(core_secret_paths, "restricted_data_roots")
    counted(tool_access, "resource_root_path")
    for total in (12, 40):
        for index in range(total):
            (repo / "src" / f"module_{index}.py").write_text(
                f"def visible_function_{index}():\n    return {index}\n", encoding="utf-8",
            )
        counts.clear()
        result = reg.execute(tool, args)
        assert "BLOCKED" not in result and "ERROR" not in result, result
        assert "module_0.py" in result
        assert counts["owner_credential_locations"] == counts["restricted_data_roots"] == 1, counts
        assert counts["task_drive"] == counts["artifact_store"] == 1, counts


def test_prepared_check_keeps_per_target_identity_and_all_data_roots(environment, monkeypatch):
    from ouroboros import config
    from ouroboros.tool_access import resource_root_path

    _reg, ctx, _home, repo, data = environment
    canonical, configured = repo / "canonical", repo / "configured"
    ctx.task_metadata = {"budget_drive_root": str(canonical)}
    monkeypatch.setattr(config, "DATA_DIR", configured)
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    ordinary = repo / "src" / "auth" / "normal.py"
    public = repo / "src" / "public.pem"
    nested = repo / "src" / "deploy" / "credentials.json"
    secret = repo / ".env"
    for path in (ordinary, public, nested, secret):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture", encoding="utf-8")
    owner_files = []
    for root in (data, canonical, configured):
        path = root / "state" / "skills" / "demo" / "grants.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        owner_files.append(path)
    output = resource_root_path(ctx, "task_drive") / "credentials.json"
    output.parent.mkdir(parents=True)
    output.write_text("ordinary requested output", encoding="utf-8")
    alias = repo / "src" / "old-alias.txt"
    alias.hardlink_to(secret)
    state_alias = data / "uploads" / "state-copy.txt"
    state_alias.parent.mkdir()
    state_alias.hardlink_to(owner_files[0])
    check = core_secret_paths.make_subagent_secret_target_check(repo, ctx=ctx)
    assert not check(ordinary) and not check(public) and not check(output)
    assert check(nested) and check(secret) and check(alias) and check(state_alias)
    assert all(check(path) for path in owner_files)

    # A prepared list of candidate names must not cache their inode identities.
    secret.unlink()
    secret.write_text("replacement fixture", encoding="utf-8")
    fresh_alias = repo / "src" / "new-alias.txt"
    fresh_alias.hardlink_to(secret)
    assert not check(alias) and check(fresh_alias)


def test_prepared_check_resolves_each_new_symlink_target(environment):
    _reg, ctx, _home, repo, _data = environment
    ordinary, secret, alias = repo / "normal.txt", repo / "credentials.json", repo / "link.txt"
    ordinary.write_text("ordinary", encoding="utf-8")
    secret.write_text("synthetic fixture", encoding="utf-8")
    try:
        alias.symlink_to(ordinary)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is not supported by this test host")
    check = core_secret_paths.make_subagent_secret_target_check(repo, ctx=ctx)
    assert not check(alias)
    alias.unlink()
    alias.symlink_to(secret)
    assert check(alias)


def test_query_cached_facts_do_not_cache_read_permission(environment, monkeypatch):
    from ouroboros import code_intelligence

    reg, ctx, _home, repo, data = environment
    source = repo / "visible.py"
    source.write_text("def visible_function():\n    return 1\n", encoding="utf-8")
    code_intelligence.build_code_inventory(repo, drive_root=data)
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    secret = repo / ".env"
    secret.hardlink_to(source)

    def must_use_cached_facts(*_args, **_kwargs):
        pytest.fail("unchanged digest should reuse the existing source facts")

    monkeypatch.setattr(code_intelligence, "_file_fact", must_use_cached_facts)
    hidden = reg.execute("query_code", {"op": "symbols", "path": "visible.py"})
    assert "No results" in hidden and "visible_function" not in hidden, hidden
    secret.unlink()
    visible = reg.execute("query_code", {"op": "symbols", "path": "visible.py"})
    assert "visible_function" in visible, visible
