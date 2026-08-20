"""The search_code tool: classification, registration and behavior.

Split verbatim out of ``tests/test_tool_capabilities.py`` by theme. This
module owns everything search_code: its capability-set membership and
result limit, its schema/registry visibility, the literal/regex/filter
semantics, and the ripgrep path filters, fallback and symlink fence.
"""
import os
import pathlib

import pytest
import sys
import tempfile


# ---------------------------------------------------------------------------
# search_code classification tests
# ---------------------------------------------------------------------------


def test_search_code_in_core_tools():
    """search_code must be in CORE_TOOL_NAMES."""
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    assert "search_code" in CORE_TOOL_NAMES


def test_search_code_is_parallel_safe():
    """search_code must be in READ_ONLY_PARALLEL_TOOLS."""
    from ouroboros.tool_capabilities import READ_ONLY_PARALLEL_TOOLS
    assert "search_code" in READ_ONLY_PARALLEL_TOOLS


def test_search_code_has_result_limit():
    """search_code must have an explicit result size limit."""
    from ouroboros.tool_capabilities import TOOL_RESULT_LIMITS
    assert "search_code" in TOOL_RESULT_LIMITS
    from ouroboros.tool_capabilities import UNTRUNCATED_TOOL_RESULTS
    assert "plan_task" in UNTRUNCATED_TOOL_RESULTS
    # Child-handoff tools stay transport-uncapped: wait_task/get_task_result are
    # FULL by contract, and wait_tasks' compact projection must not additionally
    # be char-capped (child_result_sha256 pins the exact result text seen).
    for _handoff_tool in ("wait_task", "wait_tasks", "get_task_result"):
        assert _handoff_tool in UNTRUNCATED_TOOL_RESULTS
    from ouroboros.tool_capabilities import FOREGROUND_MUTATIVE_TOOLS
    # D10 retired claude_code_edit — the only foreground-mutative tool. The
    # CLASS stays wired (an empty set) so a successor lands as one entry.
    assert FOREGROUND_MUTATIVE_TOOLS == frozenset()


# ---------------------------------------------------------------------------
# search_code tool behavior tests
# ---------------------------------------------------------------------------


def _make_ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext
    from unittest.mock import MagicMock
    ctx = MagicMock(spec=ToolContext)
    ctx.repo_dir = tmp_path
    ctx.repo_path = lambda p: tmp_path / p
    return ctx


def _populate_repo(tmp_path):
    """Create a mini repo structure for search tests."""
    (tmp_path / "foo.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (tmp_path / "bar.py").write_text("import os\ndef hello_bar():\n    pass\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "baz.py").write_text("class MyClass:\n    hello = True\n", encoding="utf-8")
    # Binary-like file (should be skipped)
    (tmp_path / "data.png").write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
    # Cache dir (should be skipped)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "foo.cpython-310.pyc").write_bytes(b'\x00' * 50)


def test_code_search_literal(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "hello")
    assert "foo.py:1:" in result
    assert "bar.py:2:" in result
    assert "sub/baz.py:2:" in result


def test_code_search_regex(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, r"def \w+\(\)", regex=True)
    assert "foo.py:1:" in result
    assert "bar.py:2:" in result


def test_code_search_scoped_path(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "hello", path="sub")
    assert "sub/baz.py" in result
    assert "foo.py" not in result


def test_code_search_include_filter(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    (tmp_path / "readme.md").write_text("hello from markdown\n", encoding="utf-8")
    result = _code_search(ctx, "hello", include="*.md")
    assert "readme.md" in result
    assert "foo.py" not in result


def test_code_search_no_matches(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "zzz_nonexistent_zzz")
    assert "No matches found" in result


def test_code_search_skips_binaries(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "PNG")
    # .png file should be skipped even though it contains "PNG" bytes
    assert "data.png" not in result


def test_code_search_skips_cache_dirs(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    _populate_repo(tmp_path)
    result = _code_search(ctx, "foo")
    assert "__pycache__" not in result


def test_code_search_max_results(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    # Create many matching lines
    lines = "\n".join(f"match_line_{i}" for i in range(50))
    (tmp_path / "many.py").write_text(lines, encoding="utf-8")
    result = _code_search(ctx, "match_line", max_results=10)
    assert "truncated at 10" in result


def test_code_search_empty_query(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    result = _code_search(ctx, "")
    assert "SEARCH_ERROR" in result


def test_code_search_invalid_regex(tmp_path):
    from ouroboros.tools.core import _code_search
    ctx = _make_ctx(tmp_path)
    result = _code_search(ctx, "[invalid", regex=True)
    assert "SEARCH_ERROR" in result


# ---------------------------------------------------------------------------
# Initial tool visibility
# ---------------------------------------------------------------------------


def test_search_code_in_initial_schemas():
    """search_code must appear in initial tool schemas."""
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tool_policy import initial_tool_schemas
    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    names = {s["function"]["name"] for s in initial_tool_schemas(registry)}
    assert "search_code" in names


def test_search_code_registered():
    """search_code must be registered in the tool registry."""
    from ouroboros.tools.registry import ToolRegistry
    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    available = {t["function"]["name"] for t in registry.schemas()}
    assert "search_code" in available


def test_search_code_ripgrep_path_filters_protected_files(tmp_path, monkeypatch):
    """The rg fast path must receive only files that passed Ouroboros gates."""
    import json
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_MODE

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    (repo / "safe.py").write_text("needle public\n", encoding="utf-8")
    (repo / "auth").mkdir()
    (repo / "auth" / "secret.py").write_text("needle secret\n", encoding="utf-8")
    seen = tmp_path / "seen.json"
    fake_rg_py = tmp_path / "fake_rg.py"
    fake_rg_py.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "needle=args[args.index('--')+1]\n"
        "paths=args[args.index('--')+2:]\n"
        f"pathlib.Path({str(seen)!r}).write_text(json.dumps(paths))\n"
        "for p in paths:\n"
        "    text=pathlib.Path(p).read_text(errors='replace')\n"
        "    if needle in text:\n"
        "        print(json.dumps({'type':'match','data':{'path':{'text':p},'line_number':1,'lines':{'text':text.splitlines()[0]+'\\\\n'}}}))\n",
        encoding="utf-8",
    )
    fake_rg_py.chmod(0o755)
    if os.name == "nt":
        fake_rg = tmp_path / "fake_rg.cmd"
        fake_rg.write_text(f"@echo off\r\n\"{sys.executable}\" \"{fake_rg_py}\" %*\r\n", encoding="utf-8")
    else:
        fake_rg = fake_rg_py
    monkeypatch.setattr("ouroboros.code_search_rg._rg_binary", lambda: str(fake_rg))

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.task_constraint = TaskConstraint(mode=LOCAL_READONLY_SUBAGENT_MODE)
    result = registry.execute("search_code", {"query": "needle"})
    assert "safe.py" in result
    assert "auth/secret.py" not in result
    assert all("auth/secret.py" not in path for path in json.loads(seen.read_text(encoding="utf-8")))


def test_search_code_ripgrep_fallback_when_unavailable(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    (repo / "safe.py").write_text("needle public\n", encoding="utf-8")
    monkeypatch.setattr("ouroboros.code_search_rg._rg_binary", lambda: "")

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    result = registry.execute("search_code", {"query": "needle"})
    assert "safe.py" in result
    assert "files searched" in result


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_search_code_does_not_follow_symlink_outside_root(tmp_path, monkeypatch):
    """A symlink inside the workspace that points OUTSIDE the resource root must not
    be read by search_code (rg path resolved-containment + is_search_skippable)."""
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("needle CONFIDENTIAL_OUTSIDE\n", encoding="utf-8")
    (repo / "in_root.txt").write_text("needle in_root_ok\n", encoding="utf-8")
    (repo / "escape.txt").symlink_to(outside)  # symlink whose target escapes the root

    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    # rg path
    result = registry.execute("search_code", {"query": "needle"})
    assert "in_root_ok" in result
    assert "CONFIDENTIAL_OUTSIDE" not in result
    # python fallback path (rg unavailable) must also refuse the symlink
    monkeypatch.setattr("ouroboros.code_search_rg._rg_binary", lambda: "")
    fallback = registry.execute("search_code", {"query": "needle"})
    assert "CONFIDENTIAL_OUTSIDE" not in fallback
