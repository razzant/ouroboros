import json
import pathlib
import subprocess

import ouroboros.code_intelligence as ci
from ouroboros.code_intelligence import build_code_inventory, render_codebase_digest


def test_code_inventory_indexes_python_symbols_imports_and_no_raw_source_cache(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "pkg" / "main.py").write_text(
        "import pkg.helper\n\n"
        "from .helper import VALUE\n\n"
        "from . import helper\n\n"
        "CONST = 'INVENTORY_RAW_SOURCE_SENTINEL'\n\n"
        "class Worker:\n"
        "    pass\n\n"
        "async def run():\n"
        "    return CONST\n",
        encoding="utf-8",
    )

    inventory = build_code_inventory(repo, drive_root=data, persist=True)
    files = {file.path: file for file in inventory.files}
    main = files["pkg/main.py"]

    assert main.sha256
    assert main.language == "python"
    assert {symbol.name for symbol in main.symbols} >= {"Worker", "run", "CONST"}
    assert "pkg.helper" in main.imports
    assert not hasattr(main, "resolved_import_paths")
    assert not hasattr(main, "references")

    digest = render_codebase_digest(inventory)
    assert "pkg/main.py" in digest
    assert "Worker" in digest

    cache_files = list((data / "state" / "code_intel").glob("*/inventory.json"))
    assert len(cache_files) == 1
    cached = json.loads(cache_files[0].read_text(encoding="utf-8"))
    rendered_cache = json.dumps(cached)
    assert "INVENTORY_RAW_SOURCE_SENTINEL" not in rendered_cache
    assert "return CONST" not in rendered_cache
    assert cached["schema_version"] == 5


def test_code_inventory_classifies_sensitive_and_symlink_escape(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    outside = tmp_path / "outside.txt"
    repo.mkdir()
    outside.write_text("external", encoding="utf-8")
    (repo / ".env").write_text("OPENAI_API_KEY=thisisaverylongsecretvalue123456", encoding="utf-8")
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "escape").symlink_to(outside)

    inventory = build_code_inventory(repo, drive_root=data, persist=True)
    files = {file.path: file for file in inventory.files}

    assert files[".env"].disposition == "sensitive"
    assert files[".env"].sha256 == ""
    assert files[".env"].size == 0
    assert files["escape"].disposition == "path_escape"

    cache_files = list((data / "state" / "code_intel").glob("*/inventory.json"))
    cached = json.loads(cache_files[0].read_text(encoding="utf-8"))
    rendered_cache = json.dumps(cached)
    assert "thisisaverylongsecretvalue" not in rendered_cache


def test_code_inventory_rebuilds_old_cache(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    (repo / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    build_code_inventory(repo, drive_root=data, persist=True)
    cache_file = next((data / "state" / "code_intel").glob("*/inventory.json"))
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    cached["schema_version"] = 1
    for file in cached["files"]:
        file.pop("call_sites", None)
        file.pop("references", None)
    cache_file.write_text(json.dumps(cached), encoding="utf-8")

    rebuilt = build_code_inventory(repo, drive_root=data, persist=True)
    assert rebuilt.schema_version == 5
    rebuilt_cache = json.loads(cache_file.read_text(encoding="utf-8"))
    assert rebuilt_cache["schema_version"] == 5
    app = {file.path: file for file in rebuilt.files}["app.py"]
    assert hasattr(app, "call_sites")


def test_v2_cache_cannot_reintroduce_cross_file_joins(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("import helper\nhelper.run()\n")
    inventory = ci.build_code_inventory(repo, drive_root=tmp_path / "data")
    cache = ci.inventory_cache_path(repo, tmp_path / "data")
    raw = inventory.to_json()
    raw["schema_version"] = 2
    raw["files"][0]["references"] = [{"name": "stale", "line": 1}]
    raw["files"][0]["resolved_import_paths"] = ["deleted.py"]
    cache.write_text(json.dumps(raw))
    rebuilt = ci.build_code_inventory(repo, drive_root=tmp_path / "data")
    assert rebuilt.schema_version == 5
    assert "references" not in rebuilt.to_json()["files"][0]
    assert "resolved_import_paths" not in rebuilt.to_json()["files"][0]
    assert rebuilt.files[0].imports == ["helper"]
    assert [call.name for call in rebuilt.files[0].call_sites] == ["run"]


def test_nested_git_uses_own_ignores_and_visibility_before_entering(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    nested = repo / "vendor" / "child"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / ".gitignore").write_text("ignored.py\n")
    (nested / "app.py").write_text("def visible(): pass\n")
    (nested / "ignored.py").write_text("def invisible(): pass\n")
    inventory = ci.build_code_inventory(repo, persist=False)
    paths = {file.path for file in inventory.files}
    assert "vendor/child/app.py" in paths
    assert "vendor/child/ignored.py" not in paths
    assert inventory.nested_roots == ["vendor/child"]
    assert inventory.enumeration_limits == []

    called_roots = []
    original = ci.subprocess.run

    def record(*args, **kwargs):
        called_roots.append(kwargs.get("cwd"))
        return original(*args, **kwargs)

    monkeypatch.setattr(ci.subprocess, "run", record)
    filtered = ci.build_code_inventory(
        repo, persist=False,
        path_allowed=lambda path: not path.is_relative_to(nested),
    )
    assert str(nested) not in called_roots
    assert filtered.nested_roots == []
    assert not any(file.path.startswith("vendor/child/") for file in filtered.files)


def test_non_git_walk_discovers_nested_git_and_never_follows_directory_symlinks(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "child"
    outside = tmp_path / "outside"
    nested.mkdir(parents=True)
    outside.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / "app.py").write_text("def run(): pass\n")
    (nested / ".gitignore").write_text("ignored.py\n")
    (nested / "ignored.py").write_text("def ignored(): pass\n")
    (outside / "outside.py").write_text("def outside(): pass\n")
    (repo / "link").symlink_to(outside, target_is_directory=True)
    inventory = ci.build_code_inventory(repo, persist=False)
    assert "child/app.py" in {file.path for file in inventory.files}
    assert not any("outside.py" in file.path or "ignored.py" in file.path for file in inventory.files)
    assert inventory.nested_roots == ["child"]
    assert "filesystem_fallback_without_git_ignores" in inventory.enumeration_limits
    assert "directory_symlinks_not_followed" in inventory.enumeration_limits


def test_enumeration_cap_is_explicit_and_cache_round_trips_metadata(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (repo / name).write_text("def run(): pass\n")
    monkeypatch.setattr(ci, "_MAX_ENUMERATION_PATHS", 2)
    inventory = ci.build_code_inventory(repo, drive_root=tmp_path / "data")
    assert len(inventory.files) == 2
    assert "enumeration_entry_limit:2" in inventory.enumeration_limits
    loaded = ci.load_cached_inventory(repo, tmp_path / "data")
    assert loaded.enumeration_limits == inventory.enumeration_limits
    assert loaded.nested_roots == inventory.nested_roots


def test_oversized_and_sensitive_files_never_opened_for_cache_hash(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    large = repo / "large.py"
    secret = repo / ".env"
    large.write_bytes(b"x" * 80)
    secret.write_text("SECRET=must-not-read")
    monkeypatch.setattr(ci, "_MAX_INDEX_FILE_BYTES", 40)
    original_open = pathlib.Path.open

    def guarded_open(path, *args, **kwargs):
        assert path not in {large, secret}, "classified files must not be read to validate cache"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", guarded_open)
    inventory = ci.build_code_inventory(repo, persist=False)
    files = {file.path: file for file in inventory.files}
    assert files["large.py"].disposition == "oversized"
    assert files[".env"].disposition == "sensitive"


def test_digest_entries_and_rollups_respect_scope_without_hiding_unavailable_files():
    inventory = ci.CodeInventory(3, "/repo", "", "", [
        ci.FileFact("pkg/a.py", "", 10, "python", 20, symbols=[ci.SymbolFact("run", "function", 1, 2)]),
        ci.FileFact("pkg/b.rs", "", 10, "rs", 20, disposition="structural_unavailable:rs"),
        ci.FileFact("elsewhere.py", "", 10, "python", 20),
    ], {})
    totals = ci.digest_rollups(inventory, "pkg")
    assert totals["files"] == 2
    assert totals["indexed"] == 1
    assert totals["symbols"] == 1
    assert "structural_unavailable:rs" in ci.render_digest_entry(inventory.files[1])
    assert ci.digest_rollups(inventory, "pkg/a.py")["files"] == 1
    assert ci.digest_rollups(inventory, "pkg/a.py/missing")["files"] == 0


def test_relevant_files_filters_scope_before_ranking_and_supports_deep_pages():
    files = [ci.FileFact(f"unrelated/worker{i}.py", "", 10, "python", 20,
                        symbols=[ci.SymbolFact("worker", "function", 1, 2)]) for i in range(220)]
    files += [ci.FileFact(f"target/worker{i:03d}.py", "", 10, "python", 20) for i in range(220)]
    inventory = ci.CodeInventory(3, "/repo", "", "", files, {})
    selected = ci.relevant_files(inventory, "worker", path="target", limit=None)
    assert len(selected) == 220
    assert selected[205][0].path == "target/worker205.py"
    assert [row[0].path for row in ci.relevant_files(inventory, "worker", path="target", limit=1)] == ["target/worker000.py"]


def test_missing_grammar_still_supports_path_relevance():
    file = ci.FileFact("worker.go", "", 10, "go", 20, disposition="structural_unavailable:go")
    inventory = ci.CodeInventory(3, "/repo", "", "", [file], {})
    assert ci.relevant_files(inventory, "worker", limit=None)[0][0] is file


def test_path_escape_through_symlink_ancestor_is_classified_before_read(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (outside / "app.py").write_text("def external(): pass\n")
    (repo / "link").symlink_to(outside, target_is_directory=True)
    original = pathlib.Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name != "app.py", "escaped source must not be read"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", guarded_open)
    assert ci._file_fact(repo, repo / "link" / "app.py").disposition == "path_escape"


def test_visibility_callback_preserves_full_cache_but_never_overwrites_with_partial(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def a(): pass\n")
    (repo / "b.py").write_text("def b(): pass\n")
    drive = tmp_path / "data"
    ci.build_code_inventory(repo, drive_root=drive, path_allowed=lambda path: True)
    cache = ci.inventory_cache_path(repo, drive)
    assert cache.exists()
    before = cache.read_bytes()
    filtered = ci.build_code_inventory(repo, drive_root=drive, path_allowed=lambda path: path.name != "b.py")
    assert [file.path for file in filtered.files] == ["a.py"]
    assert cache.read_bytes() == before


def test_scoped_directory_starts_before_unrelated_enumeration_budget(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    selected = repo / "z_selected"
    selected.mkdir(parents=True)
    for index in range(8):
        (repo / f"a_unrelated{index}.py").write_text("def unrelated(): pass\n")
    (selected / "target.py").write_text("def selected(): pass\n")
    monkeypatch.setattr(ci, "_MAX_ENUMERATION_PATHS", 2)
    inventory = ci.build_code_inventory(repo, scope="z_selected", persist=False)
    assert [file.path for file in inventory.files] == ["z_selected/target.py"]
    assert not any("entry_limit" in limit for limit in inventory.enumeration_limits)


def test_scoped_explicit_file_uses_literal_git_pathspec_and_preserves_full_cache(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for index in range(8):
        (repo / f"a_unrelated{index}.py").write_text("def unrelated(): pass\n")
    (repo / "z_target[1].py").write_text("def selected(): pass\n")
    drive = tmp_path / "data"
    ci.build_code_inventory(repo, drive_root=drive)
    cache = ci.inventory_cache_path(repo, drive)
    before = cache.read_bytes()
    monkeypatch.setattr(ci, "_MAX_ENUMERATION_PATHS", 1)
    inventory = ci.build_code_inventory(repo, scope="z_target[1].py", drive_root=drive)
    assert [file.path for file in inventory.files] == ["z_target[1].py"]
    assert inventory.enumeration_limits == []
    assert cache.read_bytes() == before


def test_scoped_nested_repository_keeps_ancestor_metadata_and_ignore_rules(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "child"
    selected = nested / "pkg"
    selected.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / ".gitignore").write_text("pkg/ignored.py\n")
    (selected / "app.py").write_text("def run(): pass\n")
    (selected / "ignored.py").write_text("def hidden(): pass\n")
    inventory = ci.build_code_inventory(repo, scope="child/pkg", persist=False)
    assert [file.path for file in inventory.files] == ["child/pkg/app.py"]
    assert inventory.nested_roots == ["child"]
    ignored = ci.build_code_inventory(repo, scope="child/pkg/ignored.py", persist=False)
    assert ignored.files == []
    assert ignored.nested_roots == ["child"]


def test_scoped_directory_checks_visibility_before_git_invocation(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    selected = repo / "hidden"
    selected.mkdir(parents=True)
    (selected / "app.py").write_text("def hidden(): pass\n")
    original = ci.subprocess.run

    def guarded(*args, **kwargs):
        assert kwargs.get("cwd") != str(selected)
        return original(*args, **kwargs)

    monkeypatch.setattr(ci.subprocess, "run", guarded)
    inventory = ci.build_code_inventory(repo, scope="hidden", persist=False,
                                        path_allowed=lambda path: not path.is_relative_to(selected))
    assert inventory.files == []
