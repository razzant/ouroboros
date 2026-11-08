"""Extension normalization and honest outline coverage through the public tool."""
import json

import pytest

from ouroboros import code_intelligence as ci
from ouroboros.tools.registry import ToolRegistry


@pytest.fixture
def project(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    return repo, data, ToolRegistry(repo_dir=repo, drive_root=data)


def test_unchanged_inventory_skips_serialization_but_edits_and_head_publish(project, monkeypatch):
    repo, data, _registry = project
    source = repo / "module.py"
    source.write_text("def original(): pass\n", encoding="utf-8")
    monkeypatch.setattr(ci, "utc_now_iso", lambda: "first build")
    ci.build_code_inventory(repo, drive_root=data)
    path = ci.inventory_cache_path(repo, data)
    before = path.read_text(encoding="utf-8")
    assert before == json.dumps(json.loads(before), ensure_ascii=False, separators=(",", ":")) + "\n"
    monkeypatch.setattr(ci, "utc_now_iso", lambda: "later build")
    with monkeypatch.context() as unchanged:
        unchanged.setattr(ci.CodeInventory, "to_json", lambda self: pytest.fail("unchanged inventory dumped"))
        ci.build_code_inventory(repo, drive_root=data)
    assert path.read_text(encoding="utf-8") == before
    source.write_text("def edited(): pass\n", encoding="utf-8")
    ci.build_code_inventory(repo, drive_root=data)
    changed = json.loads(path.read_text(encoding="utf-8"))
    assert changed["created_at"] == "later build"
    assert changed["files"][0]["symbols"][0]["name"] == "edited"
    monkeypatch.setattr(ci, "_git_head", lambda root: "new-head")
    ci.build_code_inventory(repo, drive_root=data)
    assert ci.load_cached_inventory(repo, data).git_head == "new-head"


@pytest.mark.parametrize("extension,language", [
    ("mjs", "javascript"), ("mts", "typescript"), ("cts", "typescript"),
])
def test_module_extensions_share_outline_call_and_import_grammars(project, extension, language):
    repo, data, registry = project
    target = f"dep.{extension}"
    importer = f"use.{extension}"
    (repo / target).write_text("function helper() {}\n", encoding="utf-8")
    importing = (f"const helper = require('./{target}');" if extension == "cjs"
                 else f"import {{ helper }} from './{target}';")
    (repo / importer).write_text(importing + "\nfunction run() { return helper(); }\n", encoding="utf-8")

    def query(op, **kwargs):
        return registry.execute("query_code", {"op": op, "lang": language, **kwargs})

    symbols = query("symbols", query="run", path=importer)
    assert f"{importer}:2 function" in symbols and "[outline]" in symbols
    callers = query("callers", query="helper", path=target)
    assert f"{importer}:2:" in callers and "call_expression.function call?" in callers
    callees = query("callees", query="run", path=importer)
    assert f"{importer}:2:" in callees and "helper()" in callees
    digest = query("digest", path=importer)
    assert "Calls:" in digest and "helper" in digest
    impact = query("impact", query=target)
    assert f"{importer}:1:" in impact and importing in impact
    assert f"unique candidate (relative: {target}) hop 1" in impact
    inventory = ci.load_cached_inventory(repo, data)
    assert inventory is not None
    assert {file.language for file in inventory.files} == {language}
    assert all(file.disposition == "indexed" for file in inventory.files)


@pytest.mark.parametrize("extension,language", [
    ("mjs", "javascript"), ("mts", "typescript"), ("cts", "typescript"),
])
def test_cached_old_extension_facts_rebuild_without_a_source_edit(project, extension, language):
    repo, data, registry = project
    path = f"module.{extension}"
    (repo / path).write_text("function visible() {}\n", encoding="utf-8")
    inventory = ci.build_code_inventory(repo, drive_root=data)
    stale = inventory.to_json()
    stale["files"][0].update(language=extension, disposition="indexed", symbols=[], call_sites=[])
    ci.inventory_cache_path(repo, data).write_text(json.dumps(stale), encoding="utf-8")

    result = registry.execute("query_code", {"op": "symbols", "query": "visible", "path": path})
    assert f"{path}:1 function" in result
    # A correct scoped answer must not overwrite the shared full inventory.
    assert json.loads(ci.inventory_cache_path(repo, data).read_text(encoding="utf-8")) == stale
    ci.build_code_inventory(repo, drive_root=data)
    refreshed = ci.load_cached_inventory(repo, data)
    assert refreshed is not None and refreshed.files[0].language == language
    assert [symbol.name for symbol in refreshed.files[0].symbols] == ["visible"]
    assert refreshed.files[0].sha256 == stale["files"][0]["sha256"]


@pytest.mark.parametrize("path,language", [("settings.cfg", "cfg"), ("NOTES", "text")])
def test_unknown_outline_reports_no_parse_including_old_cached_facts(project, path, language):
    repo, data, registry = project
    (repo / path).write_text("needle = enabled\n", encoding="utf-8")
    inventory = ci.build_code_inventory(repo, drive_root=data)
    assert inventory.files[0].disposition == f"structural_unavailable:{language}"
    # The previous schema-3 implementation silently called unknown files indexed.
    stale = inventory.to_json()
    stale["files"][0]["disposition"] = "indexed"
    ci.inventory_cache_path(repo, data).write_text(json.dumps(stale), encoding="utf-8")

    symbols = registry.execute("query_code", {"op": "symbols", "path": path})
    assert f"coverage: structural_unavailable:{language}=1" in symbols
    assert "coverage: indexed=" not in symbols
    digest = registry.execute("query_code", {"op": "digest", "path": path})
    assert f"Status: structural_unavailable:{language}" in digest
    references = registry.execute("query_code", {"op": "references", "query": "needle"})
    assert f"{path}:1:" in references and "text" in references
    assert f"no grammar: {language}" in references
