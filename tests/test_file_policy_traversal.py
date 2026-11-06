"""File reads avoid credential-shape filters; command controls remain separate."""
from __future__ import annotations

from collections import Counter

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools import core_secret_paths
from tests.test_file_resource_consistency import environment as environment



@pytest.mark.parametrize(("tool", "args", "rg_failure", "mode"), [
    pytest.param("list_files", {"path": "src", "max_entries": 200}, None, "local_readonly_subagent", id='list_files-args0-None-local_readonly_subagent'),
    pytest.param("search_code", {"path": "src", "query": "visible_function"}, None, "acting_subagent", id='search_code-args1-None-acting_subagent'),
    pytest.param("search_code", {"path": "src", "query": "visible_function"}, FileNotFoundError, "local_readonly_subagent", id='search_code-args2-FileNotFoundError-local_readonly_subagent'),
    pytest.param("search_code", {"path": "src", "query": "visible_function"}, RuntimeError, "acting_subagent", id='search_code-args3-RuntimeError-acting_subagent'),
    pytest.param("search_code", {"path": "src/module_0.py", "query": "visible_function"}, None, "local_readonly_subagent", id='search_code-args4-None-local_readonly_subagent'),
    pytest.param("query_code", {"op": "symbols", "path": "src/module_0.py"}, None, "acting_subagent", id='query_code-args5-None-acting_subagent'),
    pytest.param("query_code", {"op": "structural", "query": "FunctionDef", "path": "src"}, None, "local_readonly_subagent", id='query_code-args6-None-local_readonly_subagent'),
])
def test_file_walks_do_not_prepare_credential_shape_filters(environment, monkeypatch, mode, tool, args, rg_failure):
    from ouroboros import code_search_rg, credential_shapes, tool_access

    if rg_failure is not None:
        def unavailable(*_args, **_kwargs):
            raise rg_failure("controlled search backend failure")
        monkeypatch.setattr(code_search_rg, "search_with_rg", unavailable)

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
    counted(core_secret_paths, "make_runtime_secret_target_check")
    counted(core_secret_paths, "runtime_data_roots")
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
        assert counts["owner_credential_locations"] == counts["make_runtime_secret_target_check"] == 0, counts
        assert counts["runtime_data_roots"] in {1, 2}, counts


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
    check = core_secret_paths.make_runtime_secret_target_check(repo, ctx=ctx)
    assert not check(ordinary) and not check(public) and not check(output) and not check(nested)
    assert check(secret) and check(alias) and check(state_alias)
    assert all(check(path) for path in owner_files)

    # A prepared list of candidate names must not cache their inode identities.
    secret.unlink()
    secret.write_text("replacement fixture", encoding="utf-8")
    fresh_alias = repo / "src" / "new-alias.txt"
    fresh_alias.hardlink_to(secret)
    assert not check(alias) and check(fresh_alias)


def test_prepared_check_resolves_each_new_symlink_target(environment):
    _reg, ctx, home, repo, _data = environment
    # The protected destination is the owner's physical store, not a project
    # file merely named credentials.json. The shared fixture binds Path.home().
    ordinary, secret, alias = repo / "credentials.json", home / ".ssh" / "id_rsa", repo / "link.txt"
    ordinary.write_text("ordinary", encoding="utf-8")
    secret.parent.mkdir()
    secret.write_text("synthetic fixture", encoding="utf-8")
    try:
        alias.symlink_to(ordinary)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is not supported by this test host")
    check = core_secret_paths.make_runtime_secret_target_check(repo, ctx=ctx)
    assert not check(alias)
    alias.unlink()
    alias.symlink_to(secret)
    assert check(alias)
    alias.unlink()
    alias.symlink_to(ordinary)
    assert not check(alias)


def test_query_cached_facts_remain_visible_with_credential_named_alias(environment, monkeypatch):
    from ouroboros import code_intelligence

    reg, ctx, _home, repo, data = environment
    source = repo / "visible.py"
    source.write_text("def visible_function():\n    return 1\n", encoding="utf-8")
    code_intelligence.build_code_inventory(repo, drive_root=data)
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    secret = repo / ".env"
    secret.hardlink_to(source)

    cached = {path: path.read_bytes() for path in (data / "state" / "code_intel").rglob("*") if path.is_file()}
    first = reg.execute("query_code", {"op": "symbols", "path": "visible.py"})
    assert "visible_function" in first, first
    secret.unlink()
    visible = reg.execute("query_code", {"op": "symbols", "path": "visible.py"})
    assert "visible_function" in visible, visible
    assert cached == {path: path.read_bytes() for path in (data / "state" / "code_intel").rglob("*") if path.is_file()}
