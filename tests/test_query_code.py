import pathlib

from ouroboros.tools.query_code import _query_code
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "helper.py").write_text(
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (repo / "pkg" / "main.py").write_text(
        "from .helper import helper\n\n"
        "class Worker:\n"
        "    def run(self):\n"
        "        return helper()\n",
        encoding="utf-8",
    )
    (repo / "web.js").write_text("import {x} from './x.js';\nfunction start(){ return x(); }\n", encoding="utf-8")
    return repo


def test_query_code_symbols_references_callers_and_impact(tmp_path):
    repo = _repo(tmp_path)
    data = tmp_path / "data"
    ctx = ToolContext(repo_dir=repo, drive_root=data)

    symbols = _query_code(ctx, op="symbols", query="Worker")
    assert "pkg/main.py" in symbols
    assert "class Worker" in symbols

    refs = _query_code(ctx, op="references", query="helper", path="pkg/helper.py")
    assert "pkg/main.py" in refs

    callers = _query_code(ctx, op="callers", query="helper", path="pkg/helper.py")
    assert "pkg/main.py" in callers
    assert "helper" in callers
    callees = _query_code(ctx, op="callees", query="run", path="pkg/main.py")
    assert "helper" in callees

    impact = _query_code(ctx, op="impact", query="pkg/helper.py")
    assert "pkg/main.py" in impact


def test_query_code_relevant_and_structural(tmp_path):
    repo = _repo(tmp_path)
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data")

    relevant = _query_code(ctx, op="relevant_files", query="worker run helper")
    assert "pkg/main.py" in relevant

    structural = _query_code(ctx, op="structural", query="FunctionDef", lang="python", path="pkg")
    assert "pkg/helper.py" in structural or "pkg/main.py" in structural

    js_symbols = _query_code(ctx, op="definition", query="start", path="web.js")
    assert "web.js" in js_symbols


def test_query_code_structural_polyglot_go_rust_and_filter(tmp_path):
    """CW11 (v6.34.0): op=structural is polyglot via tree-sitter, lang-filtered, and
    never literal-matches — a node-type query must not echo a matching comment."""
    repo = _repo(tmp_path)
    (repo / "svc.go").write_text(
        "package main\n\n// function_declaration appears in this comment\n"
        "func Serve() int { return 1 }\n",
        encoding="utf-8",
    )
    (repo / "lib.rs").write_text("pub struct Widget { n: i32 }\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data")

    go = _query_code(ctx, op="structural", query="function_declaration", lang="go")
    assert "svc.go" in go  # the Go function found via tree-sitter (node), not the comment
    assert "appears in this comment" not in go  # no literal/text fallback
    assert "pkg/helper.py" not in go and "lib.rs" not in go  # lang=go scopes to Go only

    rs = _query_code(ctx, op="structural", query="struct_item", lang="rust")
    assert "lib.rs" in rs  # Rust struct_item via tree-sitter

    py = _query_code(ctx, op="structural", query="FunctionDef", lang="python", path="pkg")
    assert "pkg/helper.py" in py or "pkg/main.py" in py  # Python ast path unchanged


def test_query_code_structural_unavailable_marker(tmp_path, monkeypatch):
    """A missing tree-sitter grammar surfaces a visible structural_unavailable marker,
    never a silent text guess."""
    repo = _repo(tmp_path)
    (repo / "svc.go").write_text("func Serve() int { return 1 }\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data")
    # Force the Go grammar to look unavailable (_structural imports _ts_parser from
    # code_intelligence at call time, so patch it there).
    monkeypatch.setattr("ouroboros.code_intelligence._ts_parser", lambda grammar: None)
    out = _query_code(ctx, op="structural", query="function_declaration", lang="go")
    assert "structural_unavailable:go" in out


def test_query_code_structural_schema_enum_is_polyglot():
    from ouroboros.tools.query_code import get_tools

    enum = get_tools()[0].schema["parameters"]["properties"]["lang"]["enum"]
    for lang in ("go", "rust", "java", "ruby", "c", "cpp"):
        assert lang in enum


def test_query_code_registered_and_policy_wired(tmp_path):
    from ouroboros.safety import POLICY_SKIP, TOOL_POLICY
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
        READ_ONLY_PARALLEL_TOOLS,
        TOOL_RESULT_LIMITS,
    )

    registry = ToolRegistry(repo_dir=_repo(tmp_path), drive_root=tmp_path / "data")
    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert "query_code" in names
    assert TOOL_POLICY["query_code"] == POLICY_SKIP
    assert "query_code" in CORE_TOOL_NAMES
    assert "query_code" in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert "query_code" in ACTING_SUBAGENT_TOOL_NAMES
    assert "query_code" in READ_ONLY_PARALLEL_TOOLS
    assert TOOL_RESULT_LIMITS["query_code"] == 80_000


def test_query_code_workspace_and_subagent_schema_roots(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE

    repo = _repo(tmp_path)
    registry = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "data")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data", workspace_root=repo, workspace_mode="external")
    registry.set_context(ctx)
    assert "query_code" in {schema["function"]["name"] for schema in registry.schemas()}

    readonly = ToolContext(
        repo_dir=repo,
        drive_root=tmp_path / "data",
        task_constraint=TaskConstraint(mode=LOCAL_READONLY_SUBAGENT_MODE),
    )
    registry.set_context(readonly)
    schema = registry.get_schema_by_name("query_code")["function"]
    assert schema["parameters"]["properties"]["root"]["enum"] == ["active_workspace", "system_repo"]

    acting = ToolContext(
        repo_dir=repo,
        drive_root=tmp_path / "data",
        task_constraint=TaskConstraint(mode=ACTING_SUBAGENT_MODE, surface="self_worktree"),
    )
    registry.set_context(acting)
    schema = registry.get_schema_by_name("query_code")["function"]
    assert schema["parameters"]["properties"]["root"]["enum"] == ["active_workspace"]
    blocked = _query_code(acting, op="symbols", query="Worker", root="system_repo")
    assert "TOOL_ACCESS_BLOCKED" in blocked


def test_query_code_subagent_name_shape_is_not_authorization(tmp_path):
    """#447 A1 conversion: this test used to pin the REFUSAL (an ordinary source
    file named secret.py was invisible to children). The positive contract is
    pinned instead: a credential-shaped NAME on an ordinary source file at an
    allowed location is readable/queryable by children (the suffix carve);
    only the conventional credential LOCATION (an auth/ directory) stays
    denied, and no code-intel cache is written for a restricted child."""
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_MODE

    repo = _repo(tmp_path)
    (repo / "pkg" / "secret.py").write_text(
        "def issue_token():\n    return 'ordinary auth logic'\n", encoding="utf-8")
    (repo / "auth").mkdir()
    (repo / "auth" / "service.py").write_text("def hidden():\n    return 2\n", encoding="utf-8")
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=tmp_path / "data",
        task_constraint=TaskConstraint(mode=LOCAL_READONLY_SUBAGENT_MODE),
    )

    result = _query_code(ctx, op="structural", query="FunctionDef", path="pkg")
    assert "pkg/secret.py" in result  # the NAME alone does not hide the file
    located = _query_code(ctx, op="structural", query="FunctionDef", path="auth")
    assert "auth/service.py" not in located  # the conventional LOCATION still does
    cache_files = list((tmp_path / "data" / "state" / "code_intel").glob("*/inventory.json"))
    assert cache_files == []
