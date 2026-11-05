"""Impact keeps source anchors useful without spending quotas on import joins."""

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
