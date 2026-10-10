"""Impact keeps source anchors useful without spending quotas on import joins."""

from collections import Counter
import subprocess

import pytest

from ouroboros.tools.registry import ToolRegistry


@pytest.fixture
def project(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo, ToolRegistry(repo_dir=repo, drive_root=tmp_path / "data")


def write(repo, path, source):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def body(result):
    return result.split("\n\n", 1)[1] if "\n\n" in result else ""


def small_row_cap(monkeypatch, cap):
    from ouroboros import code_navigation as navigation

    monkeypatch.setattr(navigation, "MAX_EVIDENCE_ROWS", cap)
    original = navigation.scan_occurrences

    def scan(*args, **kwargs):
        kwargs["max_rows"] = cap
        return original(*args, **kwargs)

    monkeypatch.setattr(navigation, "scan_occurrences", scan)


@pytest.mark.parametrize("query,expected_count", [("target.py", 1), ("rare_symbol", 3)])
def test_irrelevant_import_specifiers_do_not_spend_impact_result_cap(
    project, monkeypatch, query, expected_count,
):
    repo, registry = project
    for name in ("unrelated_one", "unrelated_two", "unrelated_three"):
        write(repo, f"{name}.py", "VALUE = 1\n")
    for index in range(8):
        write(repo, f"a{index}.py", "import unrelated_one\nimport unrelated_two\nimport unrelated_three\n")
    write(repo, "target.py", "def rare_symbol(): pass\n")
    write(repo, "z_use.py", "import target\ntarget.rare_symbol()\n")
    # A tiny selected-result cap reproduces the former 20,000-intermediate-row
    # failure without an expensive timing assertion or an oversized fixture.
    # All 24 irrelevant specifiers have real path candidates; only target
    # selection, rather than dropping unresolved imports, makes the quota fit.
    small_row_cap(monkeypatch, 3)

    result = registry.execute("query_code", {"op": "impact", "query": query})

    assert "z_use.py:1:" in body(result), result
    assert "unique candidate (suffix: target.py) hop 1" in body(result), result
    assert f"{expected_count} of {expected_count}" in result, result
    assert "selected row cap" not in result and "QUERY_CODE_TRUNCATED" not in result, result
    assert not any(f"a{index}.py:" in body(result) for index in range(8)), result


def test_selected_impact_results_still_obey_the_cap(project, monkeypatch):
    repo, registry = project
    write(repo, "target.py", "def rare_symbol(): pass\n")
    for index in range(4):
        write(repo, f"use{index}.py", "import target\n")
    small_row_cap(monkeypatch, 2)

    result = registry.execute("query_code", {"op": "impact", "query": "target.py"})

    assert "2 of at least 2" in result, result
    assert "selected row cap 2" in result, result
    assert body(result).count("hop 1") == 2, result


@pytest.mark.parametrize("comment_count", [1, 8])
def test_symbol_impact_representative_keeps_call_after_leading_comment(
    project, monkeypatch, comment_count,
):
    repo, registry = project
    write(repo, "target.py", "def rare_symbol(): pass\n")
    write(repo, "use.py", "# rare_symbol is mentioned before its use\n" * comment_count
          + "def run():\n    return rare_symbol()\n")
    write(repo, "notes.cfg", "rare_symbol is a configured name\n")
    small_row_cap(monkeypatch, 3)

    result = registry.execute("query_code", {"op": "impact", "query": "rare_symbol", "path": "target.py"})

    rows = body(result)
    assert f"use.py:{comment_count + 2}:" in rows and "call?" in rows, result
    assert "return rare_symbol()" in rows, result
    assert f"symbol occurrence evidence ({comment_count + 1} rows; representative per file)" in rows, result
    assert "notes.cfg:1:" in rows and "text" in rows, result
    assert "not bound to rare_symbol" in result, result
    assert "3 of 3" in result and "selected row cap" not in result, result


@pytest.mark.parametrize("op,query,selector", [
    ("impact", "helper", "target.py"),
    ("references", "chosen", "use.py"),
    ("callers", "chosen", "use.py"),
])
def test_each_request_normalizes_candidate_paths_once_including_alias_followup(
    project, monkeypatch, op, query, selector,
):
    from ouroboros import code_import_candidates as candidates

    repo, registry = project
    write(repo, "target.py", "def helper(): pass\n")
    write(repo, "use.py", "from target import helper as chosen\n"
          "def run(): return chosen()\n"
          "def shadow(chosen): return chosen()\n")
    # These paths never serve as an importer or import specifier. Their only
    # _safe_path calls are normalization of the current candidate universe.
    sentinels = {f"unused/files/entry{index}.py" for index in range(12)}
    for path in sentinels:
        write(repo, path, "VALUE = 1\n")
    for index in range(9):
        write(repo, f"imports{index}.py", f"import missing{index}\nimport target\n")
    normalized = Counter()
    original = candidates._safe_path

    def observe(path):
        if path in sentinels:
            normalized[path] += 1
        return original(path)

    monkeypatch.setattr(candidates, "_safe_path", observe)
    result = registry.execute("query_code", {"op": op, "query": query, "path": selector})

    assert normalized == Counter({path: 1 for path in sentinels}), normalized
    assert "time limit" not in result and "QUERY_CODE_TRUNCATED" not in result, result
    if op == "impact":
        assert "use.py:1:" in body(result) and "unique candidate" in body(result), result
        assert "use.py:2:" not in body(result), result  # No invented helper -> chosen binding.
    else:
        assert "use.py:2:" in body(result) and "use.py:3:" in body(result), result
        assert "call?" in body(result) and "order: heuristic" in result, result


def test_cached_importer_joins_added_and_removed_current_targets(project, monkeypatch):
    from ouroboros import code_intelligence as intelligence

    repo, registry = project
    write(repo, "pkg/main.ts", "import { x } from './target';\n")
    write(repo, "upper.ts", "import './pkg/main';\n")
    request = {"op": "impact", "query": "pkg/target.ts", "depth": 2, "lang": "typescript"}
    absent = registry.execute("query_code", request)
    assert "pkg/main.ts:1:" not in body(absent), absent

    rebuilt = []
    original = intelligence._file_fact

    def record_rebuild(root, path):
        rebuilt.append(path.relative_to(root).as_posix())
        return original(root, path)

    monkeypatch.setattr(intelligence, "_file_fact", record_rebuild)
    write(repo, "pkg/target.ts", "export const x = 1;\n")
    added = registry.execute("query_code", request)
    assert "pkg/main.ts:1:" in body(added), added
    assert "upper.ts:1:" in body(added) and "hop 2" in body(added), added

    write(repo, "pkg/target.js", "export const x = 1;\n")
    ambiguous = registry.execute("query_code", request)
    assert "ambiguous candidate" in body(ambiguous) and "pkg/target.js" in body(ambiguous), ambiguous
    assert "upper.ts:1:" not in body(ambiguous), ambiguous
    (repo / "pkg/target.js").unlink()
    unique_again = registry.execute("query_code", request)
    assert "unique candidate" in body(unique_again) and "upper.ts:1:" in body(unique_again), unique_again

    (repo / "pkg/target.ts").unlink()
    removed = registry.execute("query_code", request)
    assert "pkg/main.ts:1:" not in body(removed), removed
    assert "target absent from current inventory" in removed, removed
    assert "pkg/main.ts" not in rebuilt and "upper.ts" not in rebuilt, rebuilt
