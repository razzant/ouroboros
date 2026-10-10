"""Local import facts avoid full warm parses while retaining current evidence."""

from collections import Counter
from dataclasses import asdict
import json

import pytest

from ouroboros import code_intelligence as ci, code_occurrences as co
from ouroboros.tools.registry import ToolRegistry


@pytest.fixture
def project(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    return repo, data, ToolRegistry(repo_dir=repo, drive_root=data)


def write(repo, path, source):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def body(result):
    return result.split("\n\n", 1)[1] if "\n\n" in result else ""


@pytest.mark.parametrize("path,source,expected", [
    ("pkg/main.py", "import a.b as renamed, c\nfrom . import child as chosen\nfrom ..pkg import helper\n",
     [(1, 1, "ast.Import", "a.b"), (1, 1, "ast.Import", "c"),
      (2, 1, "ast.ImportFrom", "."), (2, 1, "ast.ImportFrom", ".child"),
      (3, 1, "ast.ImportFrom", "..pkg"), (3, 1, "ast.ImportFrom", "..pkg.helper")]),
    ("main.py", "from pkg import (\n child as chosen,\n other,\n)\nfrom pkg import *\n",
     [(1, 1, "ast.ImportFrom", "pkg"), (1, 1, "ast.ImportFrom", "pkg.child"),
      (1, 1, "ast.ImportFrom", "pkg.other"), (5, 1, "ast.ImportFrom", "pkg")]),
    ("main.ts", 'import type {\n T\n} from "./util";\nexport {\n T as U\n} from "./util";\n',
     [(1, 1, "import_statement", "./util"), (4, 1, "export_statement", "./util")]),
    ("main.ts", 'import util = require("./util");\n',
     [(1, 1, "import_statement", "./util")]),
    ("main.js", 'import "./side";\nconst a = require("./util");\nimport("./lazy");\n',
     [(1, 1, "import_statement", "./side"), (2, 11, "call_expression", "./util"),
      (3, 1, "call_expression", "./lazy")]),
    ("main.js", 'obj.require("./x"); require(variable); import(`./${name}`);\n', []),
    ("main.go", 'package main\nimport (\n "fmt"\n renamed "example.org/pkg"\n)\n',
     [(3, 2, "import_spec", "fmt"), (4, 2, "import_spec", "example.org/pkg")]),
    ("Main.java", 'import pkg.Util;\nimport static pkg.Util.run;\n',
     [(1, 1, "import_declaration", "pkg.Util"), (2, 1, "import_declaration", "pkg.Util"),
      (2, 1, "import_declaration", "pkg.Util.run")]),
    ("main.rs", 'use crate::util::{self, helper as chosen, nested::{A, B}};\n',
     [(1, 1, "use_declaration", spec) for spec in
      ("crate::util", "crate::util::helper", "crate::util::nested::A", "crate::util::nested::B")]),
    ("main.rs", 'use foo::*;\nuse foo::{bar, *};\n',
     [(1, 1, "use_declaration", "foo"), (2, 1, "use_declaration", "foo"),
      (2, 1, "use_declaration", "foo::bar")]),
])
def test_cached_import_syntax_roundtrip_keeps_specifiers_and_anchors(
    project, monkeypatch, path, source, expected,
):
    repo, data, _ = project
    write(repo, path, source)
    cold = ci.build_code_inventory(repo, drive_root=data)
    cached = ci.load_cached_inventory(repo, data)
    assert cached is not None
    assert cached.to_json()["files"] == cold.to_json()["files"]
    file = cached.files[0]
    assert file.import_facts_complete and file.import_facts_method
    assert [(f.line, f.column, f.slot, f.specifier) for f in file.import_facts] == expected
    assert all(set(asdict(f)) == {"specifier", "line", "column", "slot", "enclosing"}
               for f in file.import_facts)

    def no_parser(*args, **kwargs):
        pytest.fail("complete hash-identical import facts must not reparse source")

    monkeypatch.setattr(ci, "_file_fact", no_parser)
    monkeypatch.setattr(ci, "_ts_parser", no_parser)
    warm = ci.build_code_inventory(repo, drive_root=data)
    scan = co.scan_occurrences(warm, None, mode="imports")
    assert not scan.incomplete and not scan.limits
    assert scan.files_parsed == 0
    assert "hash-validated" in scan.summary()
    assert [(r.line, r.column, r.slot, r.specifier) for r in scan.rows] == expected
    assert all(r.source == source.splitlines()[r.line - 1] for r in scan.rows)


@pytest.mark.parametrize("path,source,slot", [
    ("main.py", "class Outer:\n    def run(self):\n        label = 'é'; import target\n", "ast.Import"),
    ("main.js", "class Outer {\n  run() {\n    const label = 'é'; require('./target');\n  }\n}\n", "call_expression"),
])
def test_cached_imports_keep_enclosing_names_and_utf8_byte_columns(project, path, source, slot):
    repo, data, _ = project
    write(repo, path, source)
    inventory = ci.build_code_inventory(repo, drive_root=data)
    scan = co.scan_occurrences(inventory, None, mode="imports")
    assert len(scan.rows) == 1
    row = scan.rows[0]
    needle = "import target" if path.endswith(".py") else "require("
    line = source.splitlines()[2]
    assert (row.line, row.column, row.slot) == (3, len(line[:line.index(needle)].encode()) + 1, slot)
    assert row.enclosing == "Outer / run"
    assert row.source == line


@pytest.mark.parametrize("query", ["target.py", "rare_symbol"])
def test_warm_public_impact_parses_only_symbol_matching_files(project, monkeypatch, query):
    repo, data, registry = project
    write(repo, "target.py", "def rare_symbol(): pass\n")
    write(repo, "use.py", "from target import rare_symbol as chosen\nchosen()\n")
    for index in range(40):
        write(repo, f"unrelated{index}.py", "import target\n")
    ci.build_code_inventory(repo, drive_root=data)
    original = ci._ts_parser
    parsed = Counter()

    class CountingParser:
        def __init__(self, parser):
            self.parser = parser

        def parse(self, raw):
            parsed[raw] += 1
            assert b"rare_symbol" in raw, "unrelated imports must not trigger a full parse"
            return self.parser.parse(raw)

    monkeypatch.setattr(ci, "_ts_parser", lambda grammar: CountingParser(original(grammar)))
    monkeypatch.setattr(ci, "_file_fact", lambda *args: pytest.fail("unchanged inventory must not rebuild"))
    for _ in range(2):
        parsed.clear()
        result = registry.execute("query_code", {"op": "impact", "query": query, "limit": 200})
        assert "QUERY_CODE_ERROR" not in result and "QUERY_CODE_TRUNCATED" not in result, result
        assert "use.py:1:1" in body(result) and "unique candidate" in body(result), result
        assert "from target import rare_symbol as chosen" in body(result), result
        assert "hash-validated" in result, result
        assert sum(parsed.values()) == (0 if query == "target.py" else 2)


def test_cached_python_relative_submodules_use_current_targets(project, monkeypatch):
    repo, data, registry = project
    write(repo, "pkg/use.py", "from . import child as chosen\n")
    write(repo, "upper.py", "import pkg.use\n")
    ci.build_code_inventory(repo, drive_root=data)
    original = ci._file_fact
    rebuilt = []

    def record(root, path):
        rebuilt.append(path.relative_to(root).as_posix())
        return original(root, path)

    monkeypatch.setattr(ci, "_file_fact", record)
    # Python's existing AST inventory supplies all imports here. No query-time
    # tree-sitter parse is needed, including after adding/deleting target paths.
    monkeypatch.setattr(ci, "_ts_parser", lambda grammar: pytest.fail("unexpected full import parse"))
    request = {"op": "impact", "query": "pkg/child.py", "depth": 2}
    absent = registry.execute("query_code", request)
    assert "pkg/use.py:1:" not in body(absent)
    write(repo, "pkg/child.py", "VALUE = 1\n")
    added = registry.execute("query_code", request)
    assert "unique candidate (relative: pkg/child.py) hop 1" in body(added), added
    assert "upper.py:1:1" in body(added) and "hop 2" in body(added), added
    write(repo, "pkg/child/__init__.py", "VALUE = 1\n")
    ambiguous = registry.execute("query_code", request)
    assert "ambiguous candidate" in body(ambiguous), ambiguous
    assert "pkg/child/__init__.py" in body(ambiguous), ambiguous
    assert "upper.py:1:" not in body(ambiguous), ambiguous
    (repo / "pkg/child/__init__.py").unlink()
    (repo / "pkg/child.py").unlink()
    removed = registry.execute("query_code", request)
    assert "pkg/use.py:1:" not in body(removed)
    assert rebuilt == ["pkg/child.py", "pkg/child/__init__.py"]


def test_changed_import_source_after_inventory_parses_the_same_bytes(project, monkeypatch):
    repo, data, _ = project
    write(repo, "main.ts", "import './old';\n")
    inventory = ci.build_code_inventory(repo, drive_root=data)
    write(repo, "main.ts", "// changed\nexport { value } from './new';\n")
    scan = co.scan_occurrences(inventory, None, mode="imports")
    assert scan.files_parsed == 1
    assert [(r.line, r.specifier, r.source) for r in scan.rows] == [
        (2, "./new", "export { value } from './new';")]
    # Scanning never modifies the old inventory or persists a replacement.
    assert inventory.files[0].import_facts[0].specifier == "./old"
    assert ci.load_cached_inventory(repo, data).files[0].import_facts[0].specifier == "./old"


@pytest.mark.parametrize("bound,value", [("_MAX_LOCAL_IMPORT_FACTS", 1), ("_MAX_LOCAL_IMPORT_CHARS", 1)])
def test_bounded_local_import_overflow_falls_back_without_losing_imports(project, monkeypatch, bound, value):
    repo, data, _ = project
    write(repo, "main.ts", "import './one';\nexport { x } from './two';\n")
    monkeypatch.setattr(ci, bound, value)
    inventory = ci.build_code_inventory(repo, drive_root=data)
    file = inventory.files[0]
    assert not file.import_facts_complete and not file.import_facts
    scan = co.scan_occurrences(inventory, None, mode="imports")
    assert scan.files_parsed == 1
    assert [r.specifier for r in scan.rows] == ["./one", "./two"]
    assert not scan.incomplete


def test_partial_python_import_syntax_stays_available(project):
    repo, data, _ = project
    write(repo, "main.py", "import target\ndef broken(:\n    pass\n")
    inventory = ci.build_code_inventory(repo, drive_root=data)
    assert inventory.files[0].syntax_error
    scan = co.scan_occurrences(inventory, None, mode="imports")
    assert any(r.specifier == "target" and r.line == 1 and r.column == 1 for r in scan.rows)
    assert any("syntax error" in reason for reason in scan.limits)


@pytest.mark.parametrize("version", [3, 4])
def test_old_or_missing_import_projection_rebuilds_once(project, monkeypatch, version):
    repo, data, _ = project
    write(repo, "main.py", "from . import child\n")
    inventory = ci.build_code_inventory(repo, drive_root=data)
    cache = ci.inventory_cache_path(repo, data)
    raw = inventory.to_json()
    raw["schema_version"] = version
    for file in raw["files"]:
        for key in ("import_facts", "import_facts_complete", "import_facts_method"):
            file.pop(key)
    cache.write_text(json.dumps(raw), encoding="utf-8")
    rebuilt = ci.build_code_inventory(repo, drive_root=data)
    assert rebuilt.schema_version == 5
    assert [f.specifier for f in rebuilt.files[0].import_facts] == [".", ".child"]

    monkeypatch.setattr(ci, "_file_fact", lambda *args: pytest.fail("warm cache should be reused"))
    monkeypatch.setattr(ci, "_ts_parser", lambda *args: pytest.fail("warm imports should not parse"))
    warm = ci.build_code_inventory(repo, drive_root=data)
    assert co.scan_occurrences(warm, None, mode="imports").files_parsed == 0


def test_cached_import_reuse_obeys_read_policy(project, monkeypatch):
    repo, data, _ = project
    write(repo, "main.py", "import target\n")
    inventory = ci.build_code_inventory(repo, drive_root=data)
    monkeypatch.setattr(co, "search_skip_reason", lambda path: pytest.fail("denied path reached source checks"))
    scan = co.scan_occurrences(inventory, None, mode="imports", path_allowed=lambda path: False)
    assert not scan.rows and scan.files_scanned == 0
    assert scan.limits["policy excluded"] == 1
