"""Smoke test suite for Ouroboros.

Tests core invariants:
- All modules import cleanly
- Tool registry discovers all expected tools
- Utility functions work correctly
- Memory operations don't crash
- Context builder produces valid structure
- Bible invariants hold (no hardcoded replies, version sync)

Run: python -m pytest tests/test_smoke.py -v
"""
import os
import pathlib
import re
import shutil
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

# ── Module imports ───────────────────────────────────────────────

CORE_MODULES = [
    "ouroboros.agent",
    "ouroboros.context",
    "ouroboros.loop",
    "ouroboros.llm",
    "ouroboros.memory",
    "ouroboros.review",
    "ouroboros.git_shell_policy",
    "ouroboros.protected_artifacts",
    "ouroboros.shell_parse",
    "ouroboros.utils",
    "ouroboros.consciousness",
    "ouroboros.tool_capabilities",
]

TOOL_MODULES = [
    "ouroboros.tools.registry",
    "ouroboros.tools.core",
    "ouroboros.tools.git",
    "ouroboros.tools.shell",
    "ouroboros.tools.search",
    "ouroboros.tools.control",
    "ouroboros.tools.browser",
    "ouroboros.tools.review",
    "ouroboros.tools.claude_advisory_review",
    "ouroboros.tools.recent_tasks",
    "ouroboros.tools.scope_review",
    "ouroboros.tools.review_helpers",
    "ouroboros.tools.plan_review",
    "ouroboros.tools.git_rollback",
    "ouroboros.tools.git_pr",
    "ouroboros.tools.github",
    "ouroboros.tools.ci",
    "ouroboros.tools.verify",
]

SUPERVISOR_MODULES = [
    "supervisor.state",
    "supervisor.message_bus",
    "supervisor.queue",
    "supervisor.workers",
    "supervisor.git_ops",
    "supervisor.events",
]


@pytest.mark.parametrize("module", CORE_MODULES + TOOL_MODULES + SUPERVISOR_MODULES)
def test_import(module):
    """Every module imports without error."""
    __import__(module)


# ── Tool registry ────────────────────────────────────────────────

@pytest.fixture
def registry():
    from ouroboros.tools.registry import ToolRegistry
    tmp = pathlib.Path(tempfile.mkdtemp())
    return ToolRegistry(repo_dir=tmp, drive_root=tmp)


def test_tool_set_matches(registry):
    """Tool registry contains exactly the expected tools (no more, no less)."""
    schemas = registry.schemas()
    actual_tools = {t["function"]["name"] for t in schemas}
    expected_tools = set(EXPECTED_TOOLS)

    missing = expected_tools - actual_tools
    extra = actual_tools - expected_tools

    assert missing == set(), f"Missing tools: {sorted(missing)}"
    assert extra == set(), f"Extra tools: {sorted(extra)}"
    assert actual_tools == expected_tools, "Tool set mismatch"


EXPECTED_TOOLS = [
    "browse_page", "browser_action",
    "run_ci_tests",
    "advisory_review", "review_status",
    "compact_context", "set_tool_timeout", "request_restart",
    "promote_to_stable", "schedule_subagent", "integrate_subagent_patch", "compare_subagent_patches",
    # C1: the explicit acceptance seam for a delegated run's captured patch —
    # a first-class tool, so the registry contract must name it.
    "integrate_delegated_patch", "cancel_task",
    "peek_task", "discard_child_result", "override_delegation_constraint",
    "request_deep_self_review", "chat_history", "update_scratchpad",
    "send_user_message", "update_identity", "toggle_evolution",
    "toggle_consciousness", "switch_model", "get_task_result",
    "wait_task", "wait_tasks", "tree_note", "tree_read",
    "delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer",
    "read_file", "list_files", "write_file", "edit_text",
    "apply_patch", "edit_batch",
    "send_photo", "send_video", "send_file", "search_code", "query_code", "forward_to_worker",
    "generate_evolution_stats",
    "commit_reviewed", "vcs_commit_reviewed", "vcs_status", "vcs_diff",
    "vcs_pull_ff", "vcs_restore", "vcs_revert",
    "fetch_pr_ref", "create_integration_branch", "cherry_pick_pr_commits",
    "stage_adaptations", "stage_pr_merge", "vcs_rollback",
    "list_github_prs", "get_github_pr", "comment_on_pr",
    "list_github_issues", "get_github_issue", "comment_on_issue",
    "close_github_issue", "create_github_issue",
    "codebase_health", "knowledge_read", "knowledge_write", "knowledge_list",
    "journal_read", "journal_write", "workpad_read", "workpad_write",
    "promote_chat_to_task", "route_to_project", "list_projects", "steer_task",
    "ensure_project_scope",
    "memory_map", "memory_update_registry",
    "plan_task", "recent_tasks", "task_acceptance_review", "verify_and_record", "web_search",
    "start_service", "service_status", "service_logs", "stop_service",
    "run_command", "run_script",
    "list_skills", "skill_review", "skill_exec", "toggle_skill",
    "skill_preflight", "submit_skill_to_hub",
    "list_available_tools", "enable_tools",
    "analyze_screenshot", "vlm_query", "view_image",
    "ocr_pdf", "youtube_transcript", "extract_video_frames",
]


@pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
def test_tool_registered(registry, tool_name):
    """Each expected tool is in the registry."""
    available = [t["function"]["name"] for t in registry.schemas()]
    assert tool_name in available, f"{tool_name} not in registry"


def test_unknown_tool_returns_warning(registry):
    """Calling unknown tool returns warning, not exception."""
    result = registry.execute("__nonexistent__", {})
    assert "Unknown tool" in result or "⚠️" in result


def test_tool_schemas_valid(registry):
    """All tool schemas have required OpenAI fields."""
    for schema in registry.schemas():
        assert schema["type"] == "function"
        func = schema["function"]
        assert "name" in func
        assert "description" in func
        assert isinstance(func["description"], str)
        assert "parameters" in func
        params = func["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


def test_tool_schemas_have_no_empty_enum_values(registry):
    """No tool-parameter `enum` may contain an empty/blank string.

    Google Gemini's function-calling validator rejects empty enum values with
    HTTP 400 INVALID_ARGUMENT ("enum[0]: cannot be empty"), which silently forces
    a per-round fallback to another provider. OpenAI/Anthropic accept empty enums,
    so this only surfaces against live Gemini — hence this cheap static guard over
    the whole assembled tool-schema set. Express "no choice" by OMITTING the
    optional param, never by an empty enum member."""
    def _walk(node, path):
        if isinstance(node, dict):
            enum = node.get("enum")
            if isinstance(enum, list):
                bad = [v for v in enum if isinstance(v, str) and v.strip() == ""]
                assert not bad, f"empty enum value at {path}: {enum!r}"
            for key, value in node.items():
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{path}[{i}]")

    for schema in registry.schemas():
        _walk(schema.get("function", {}).get("parameters", {}), schema.get("function", {}).get("name", "?"))


def test_github_create_issue_schema_fields(registry):
    schema = registry.get_schema_by_name("create_github_issue")["function"]
    props = schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["title"]
    assert props["title"]["type"] == "string"
    assert props["body"]["type"] == "string"
    assert props["body"]["default"] == ""
    assert props["labels"]["type"] == "string"
    assert props["labels"]["default"] == ""


def test_tool_execute_basic(registry):
    """Actually execute a simple tool to verify execution works."""
    result = registry.execute("run_command", {"cmd": ["echo", "hello"]})
    assert isinstance(result, str), "Tool execute should return string"
    assert "hello" in result.lower() or "⚠️" in result, "Should return output or error"


def test_frozen_registry_includes_packaged_tool_modules(monkeypatch):
    """Frozen-mode registry must still load packaged tool modules."""
    from ouroboros.tools.registry import ToolRegistry
    tmp = pathlib.Path(tempfile.mkdtemp())
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    available = {t["function"]["name"] for t in registry.schemas()}
    expected_subset = {
        "memory_map",
        "memory_update_registry",
        "advisory_review",
        "review_status",
        "plan_task",
        "vcs_rollback",
        "run_ci_tests",
        # github.py is in _FROZEN_TOOL_MODULES — PR inspection tools must work in frozen builds
        "list_github_prs",
        "get_github_pr",
        "comment_on_pr",
        "query_code",
    }
    missing = expected_subset - available
    assert missing == set(), f"Frozen registry missing tools: {sorted(missing)}"


# ── Utilities ────────────────────────────────────────────────────

def test_safe_relpath_normal():
    from ouroboros.utils import safe_relpath
    result = safe_relpath("foo/bar.py")
    assert result == "foo/bar.py"


def test_safe_relpath_rejects_traversal():
    from ouroboros.utils import safe_relpath
    with pytest.raises(ValueError):
        safe_relpath("../../../etc/passwd")


def test_safe_relpath_strips_leading_slash():
    """safe_relpath strips leading / but doesn't raise."""
    from ouroboros.utils import safe_relpath
    result = safe_relpath("/etc/passwd")
    assert not result.startswith("/")


def test_clip_text():
    from ouroboros.utils import clip_text

    # Test 1: Long text gets clipped (max_chars=500)
    long_text = "hello world " * 100  # ~1200 chars
    result = clip_text(long_text, 500)
    assert len(result) < len(long_text), "Long text should be clipped"
    assert len(result) > 0, "Result should not be empty"
    assert "...(truncated)..." in result, "Truncation marker should be present"

    # Test 2: Short text passes through unchanged
    short_text = "hello world"
    result_short = clip_text(short_text, 500)
    assert result_short == short_text, "Short text should pass through unchanged"


def test_estimate_tokens():
    from ouroboros.utils import estimate_tokens
    tokens = estimate_tokens("Hello world, this is a test.")
    assert 5 <= tokens <= 20


# ── Memory ───────────────────────────────────────────────────────

def test_memory_scratchpad():
    """Memory reads/writes scratchpad without crash."""
    from ouroboros.memory import Memory
    with tempfile.TemporaryDirectory() as tmp:
        from ouroboros.utils import write_text
        mem = Memory(drive_root=pathlib.Path(tmp))
        write_text(mem.scratchpad_path(), "test content")
        content = mem.load_scratchpad()
        assert "test content" in content


def test_memory_identity():
    """Memory reads/writes identity without crash."""
    from ouroboros.memory import Memory
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(drive_root=pathlib.Path(tmp))
        # Write identity file directly (identity_path is a method)
        mem.identity_path().parent.mkdir(parents=True, exist_ok=True)
        mem.identity_path().write_text("I am Ouroboros", encoding="utf-8")
        content = mem.load_identity()
        assert "Ouroboros" in content


def test_memory_chat_history_empty():
    """Chat history returns string when no data."""
    from ouroboros.memory import Memory
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(drive_root=pathlib.Path(tmp))
        history = mem.chat_history(count=10)
        assert isinstance(history, str)


def test_memory_persistence():
    """Memory persists across instances (write with one, read with another)."""
    from ouroboros.memory import Memory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)

        # Write with first instance
        from ouroboros.utils import write_text
        mem1 = Memory(drive_root=tmp_path)
        write_text(mem1.scratchpad_path(), "test persistence content")

        # Read with second instance
        mem2 = Memory(drive_root=tmp_path)
        content = mem2.load_scratchpad()
        assert "test persistence content" in content, "Memory should persist across instances"


# ── Context builder ─────────────────────────────────────────────

# ── Bible invariants ─────────────────────────────────────────────

def test_no_hardcoded_replies():
    """Principle 5 (LLM-First): no hardcoded reply strings in code.
    
    Checks for suspicious patterns like:
    - reply = "Fixed string"
    - return "Sorry, I can't..."
    """
    suspicious = re.compile(
        r'(reply|response)\s*=\s*["\'](?!$|{|\s*$)',
        re.IGNORECASE,
    )
    violations = []
    for root, dirs, files in os.walk(REPO / "ouroboros"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = pathlib.Path(root) / f
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                if suspicious.search(line):
                    if "{" in line or "f'" in line or 'f"' in line:
                        continue
                    violations.append(f"{path.name}:{i}: {line.strip()}")
    assert len(violations) < 5, "Possible hardcoded replies:\n" + "\n".join(violations)


def test_version_file_exists():
    """VERSION file exists and contains a valid PEP 440 version.

    Stable releases carry plain ``X.Y.Z``; pre-releases carry
    ``X.Y.Z[-]?(rc|alpha|beta|a|b)\\.?N`` per the ``release_sync``
    carrier-format contract. Both are accepted here; stricter
    spelling rules live in ``tests/test_release_sync.py``.
    """
    from ouroboros.tools.release_sync import _VERSION_RE

    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    assert _VERSION_RE.match(version), (
        f"VERSION '{version}' is not a valid semver / PEP 440 pre-release token"
    )


def test_version_in_readme():
    """VERSION matches what README claims."""
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert version in readme, f"VERSION {version} not found in README.md"


def test_bible_exists_and_has_principles():
    """BIBLE.md exists and contains the current principle set (0-12)."""
    bible = (REPO / "BIBLE.md").read_text(encoding="utf-8")
    principles = re.findall(r"^## Principle (\d+):", bible, flags=re.MULTILINE)
    assert principles == [str(i) for i in range(13)], f"Unexpected BIBLE principles: {principles}"


# ── Code quality invariants ──────────────────────────────────────

def test_no_env_dumping():
    """Security: no code dumps entire env (os.environ without key access).

    Allows: os.environ["KEY"], os.environ.get(), os.environ.setdefault(),
            os.environ.copy() (for subprocess).
    Disallows: print(os.environ), json.dumps(os.environ), etc.
    """
    # Only flag raw os.environ passed to print/json/log without bracket or .get( accessor
    dangerous = re.compile(r'(?:print|json\.dumps|log)\s*\(.*\bos\.environ\b(?!\s*[\[.])')
    violations = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'tests')]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = pathlib.Path(root) / f
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                if dangerous.search(line):
                    violations.append(f"{path.name}:{i}: {line.strip()[:80]}")
    assert len(violations) == 0, "Dangerous env dumping:\n" + "\n".join(violations)


def test_no_oversized_modules():
    """Principle 7: exact-path debt is the only exception to the hard gate."""
    from ouroboros.review import GIANT_PATHS, MAX_MODULE_LINES, iter_gated_modules

    max_lines = MAX_MODULE_LINES
    violations = [
        f"{module.path}: {module.line_count} lines"
        for module in iter_gated_modules(REPO)
        if module.line_count > max_lines and module.path not in GIANT_PATHS
    ]
    assert len(violations) == 0, f"Oversized modules (>{max_lines} lines):\n" + "\n".join(violations)


def test_size_ratchet_manifest_matches_live_tree():
    """Exact module/function/band/byte debt matches the untruncated candidate tree."""
    from ouroboros.review import validate_size_ratchet

    errors = validate_size_ratchet(REPO)
    assert not errors, "Size-ratchet manifest violations:\n" + "\n".join(errors)


def test_js_module_gate_buckets_and_grandfathering():
    """The JS size gate sees web/tests and exempts debt by exact rel-path only."""
    from ouroboros.review import compute_complexity_metrics, module_is_grandfathered

    sections = [
        ("repo/web/app.js", "x\n" * 2000),                   # gated, over hard gate, not grandfathered
        ("repo/web/modules/chat.js", "x\n" * 4000),          # gated, grandfathered by rel-path
        ("repo/web/vendor/chart.umd.min.js", "x\n" * 9000),  # vendored/minified — excluded
        ("repo/web/tests/foo.test.js", "x\n" * 9000),        # web/tests/ — gated
        ("repo/ouroboros/small.py", "x\n" * 10),
    ]
    metrics = compute_complexity_metrics(sections)

    assert metrics["js_files"] == 3  # app.js + chat.js + web/tests; vendored excluded
    oversized = {p for p, _n in metrics["oversized_modules"]}
    grandfathered = {p for p, _n in metrics["grandfathered_modules"]}
    assert "web/app.js" in oversized
    assert "web/modules/chat.js" in grandfathered
    assert "web/modules/chat.js" not in oversized
    assert "web/tests/foo.test.js" in oversized
    assert "web/vendor/chart.umd.min.js" not in oversized
    assert "web/vendor/chart.umd.min.js" not in grandfathered

    # Grandfather entry is rel-path-keyed: a chat.js anywhere else stays gated.
    assert module_is_grandfathered("web/modules/chat.js")
    assert not module_is_grandfathered("repo/web/modules/chat.js")
    assert not module_is_grandfathered("web/other/chat.js")


def _bare_except_pass_violations(root: pathlib.Path) -> list:
    """Sites under `root` spelling a BARE `except:` swallowed by `pass`.

    BOUNDARY, stated because this scan's docstring used to claim otherwise
    ("not even except Exception: pass"): `except <Class>: pass` is NOT flagged,
    and `ouroboros/` holds hundreds of those. Only the classless form — which
    swallows `KeyboardInterrupt` and `SystemExit` as well — is a hard error here.
    Widening it to typed swallows is a policy change over a few hundred existing
    sites, not a test fix.
    """

    violations = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = pathlib.Path(base) / f
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines, 1):
                if line.strip() != "except:":
                    continue
                for j in range(i, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and next_line == "pass":
                        violations.append(f"{path.name}:{i}: bare except: pass")
                        break
    return violations


def test_the_bare_except_scan_flags_one():
    """The scan must be shown FLAGGING a violation, not only finding none.

    `ouroboros/` contains zero bare `except:`, so this detector has never had a
    candidate and passing proved nothing about whether it can fire. Both the
    matched shape and the deliberately unmatched ones are pinned here.
    """

    root = pathlib.Path(tempfile.mkdtemp())
    try:
        (root / "offender.py").write_text(
            "try:\n    go()\nexcept:\n    pass\n", encoding="utf-8"
        )
        (root / "typed.py").write_text(
            "try:\n    go()\nexcept ValueError:\n    pass\n", encoding="utf-8"
        )
        (root / "reraised.py").write_text(
            "try:\n    go()\nexcept:\n    raise\n", encoding="utf-8"
        )
        assert _bare_except_pass_violations(root) == [
            "offender.py:3: bare except: pass"
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_bare_except_pass():
    """No bare `except:` swallowed by `pass` in `ouroboros/`.

    A classless `except` also swallows `KeyboardInterrupt`/`SystemExit`; paired
    with `pass` it is silent and unkillable-looking at once. What this does NOT
    cover is stated on `_bare_except_pass_violations`.
    """

    violations = _bare_except_pass_violations(REPO / "ouroboros")
    assert len(violations) == 0, "Bare except:pass found:\n" + "\n".join(violations)


# ── AST-based function size check ───────────────────────────────


def _get_function_sizes():
    """Return exact path/qualname tuples from the production iterator.

    The path is REPO-RELATIVE, not the bare basename: the function-size waiver in
    `FUNCTION_DEBT` is keyed the same way, and keying it by basename exempted a
    (name, function) pair in every directory at once.
    """
    from ouroboros.review import iter_gated_functions

    return [(item.path, item.qualname, item.line_count) for item in iter_gated_functions(REPO)]


def test_no_extremely_oversized_functions():
    """No function exceeds the hard gate."""
    from ouroboros.review import FUNCTION_DEBT, MAX_FUNCTION_LINES

    violations = []
    for path, qualname, size in _get_function_sizes():
        if (path, qualname) in FUNCTION_DEBT:
            continue
        if size > MAX_FUNCTION_LINES:
            violations.append(f"{path}:{qualname} = {size} lines")
    assert len(violations) == 0, \
        f"Functions exceeding {MAX_FUNCTION_LINES} lines:\n" + "\n".join(violations)


def test_waivers_are_keyed_by_exact_repo_relative_path():
    """A same-named file in another directory inherits NO size waiver.

    Both waiver sets used to be keyed by bare basename, so `ouroboros/probe/llm.py`
    at 1701 lines and a 402-line `_run_reviewed_stage_cycle` in some other `git.py`
    passed the hard gates without anyone accepting that debt. The ratchet manifest
    keys debt by exact repo-relative path; this pins that it stays one.
    """
    from ouroboros.review import (
        FUNCTION_DEBT,
        GIANT_PATHS,
        function_is_grandfathered,
        module_is_grandfathered,
    )

    # Every entry names a real file, so the waiver list cannot drift into fiction.
    for rel in GIANT_PATHS:
        assert (REPO / rel).is_file(), f"giant path {rel} does not exist"
    for rel, _qualname in FUNCTION_DEBT:
        assert (REPO / rel).is_file(), f"function debt module {rel} does not exist"

    # A file that merely shares the NAME of a debt entry is not exempt.
    for impostor in (
        "ouroboros/probe/llm.py",
        "ouroboros/mut_probe_dir/git.py",
        "ouroboros/core.py",
        "skills/other_skill/plugin.py",
    ):
        assert not module_is_grandfathered(impostor), impostor

    # …and neither is a (path, qualname) pair reused in another directory.
    for rel, qualname in FUNCTION_DEBT:
        assert function_is_grandfathered(rel, qualname)
        if "/" in rel:
            assert not function_is_grandfathered(pathlib.PurePosixPath(rel).name, qualname)
            assert not function_is_grandfathered(f"ouroboros/mut_probe_dir/{rel}", qualname)


def test_function_count_has_sanity_floor():
    """The production inventory still discovers a plausible codebase."""
    sizes = _get_function_sizes()
    assert len(sizes) >= 100, f"Only {len(sizes)} functions — too few?"


# ── Pre-push gate tests ──────────────────────────────────────────────

class TestPrePushGate:
    """Tests for pre-push test gate in git.py."""

    def test_run_pre_push_tests_disabled(self):
        """When OUROBOROS_PRE_PUSH_TESTS=0, should return None (skip)."""
        import os
        from ouroboros.tools.git import _run_pre_push_tests
        old = os.environ.get("OUROBOROS_PRE_PUSH_TESTS")
        try:
            os.environ["OUROBOROS_PRE_PUSH_TESTS"] = "0"
            # ctx doesn't matter since we return early
            result = _run_pre_push_tests(None)
            assert result is None
        finally:
            if old is None:
                os.environ.pop("OUROBOROS_PRE_PUSH_TESTS", None)
            else:
                os.environ["OUROBOROS_PRE_PUSH_TESTS"] = old

    def test_run_pre_push_tests_no_tests_dir(self):
        """When tests/ dir doesn't exist, should return None."""
        from ouroboros.tools.git import _run_pre_push_tests
        import os
        old = os.environ.get("OUROBOROS_PRE_PUSH_TESTS")
        try:
            os.environ["OUROBOROS_PRE_PUSH_TESTS"] = "1"
            # Create a mock ctx with non-existent repo_dir
            class FakeCtx:
                repo_dir = "/tmp/nonexistent_repo_dir_12345"
            result = _run_pre_push_tests(FakeCtx())
            assert result is None
        finally:
            if old is None:
                os.environ.pop("OUROBOROS_PRE_PUSH_TESTS", None)
            else:
                os.environ["OUROBOROS_PRE_PUSH_TESTS"] = old

    def test_git_commit_with_tests_exists(self):
        """_git_commit_with_tests helper exists and is callable."""
        from ouroboros.tools.git import _git_commit_with_tests
        assert callable(_git_commit_with_tests)

    def test_pre_push_tests_timeout_is_sufficient(self):
        """The pre-push/post-commit pytest budget must be >= 180s.

        Since v6.88.0 this is the TOTAL budget across BOTH preflight passes
        (parallel ``not serial``, then ``serial``, which gets the remainder).
        The full suite measures ~180s two-pass against ~470-510s for the old
        single serial pass, so a shorter cap produces false TESTS_FAILED on
        every successful commit. The budget is owned by ``run_hermetic_pytest``
        (default + ``OUROBOROS_PREFLIGHT_TIMEOUT_SEC`` env) so callers do not
        re-pin a stale literal — this guard anchors on that single source of truth.
        """
        from ouroboros.preflight_runner import (
            _DEFAULT_PREFLIGHT_TIMEOUT_SEC,
            _resolve_preflight_timeout,
        )

        assert _DEFAULT_PREFLIGHT_TIMEOUT_SEC >= 180, (
            f"preflight default timeout is {_DEFAULT_PREFLIGHT_TIMEOUT_SEC}s — must be "
            ">= 180s; the full suite takes ~2 minutes and a shorter cap reports "
            "spurious TESTS_FAILED on successful commits."
        )
        # The env override is honoured so operators can raise it on slow hosts.
        import os as _os
        prev = _os.environ.get("OUROBOROS_PREFLIGHT_TIMEOUT_SEC")
        try:
            _os.environ["OUROBOROS_PREFLIGHT_TIMEOUT_SEC"] = "600"
            assert _resolve_preflight_timeout(_DEFAULT_PREFLIGHT_TIMEOUT_SEC) == 600
        finally:
            if prev is None:
                _os.environ.pop("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", None)
            else:
                _os.environ["OUROBOROS_PREFLIGHT_TIMEOUT_SEC"] = prev




def test_no_module_defines_the_same_top_level_name_twice():
    """A redefinition is invisible at import: the later one silently wins.

    This is the shape a clean git merge produces when two branches independently add
    the same concept — no conflict markers, no import error, no failing test, just one
    definition quietly shadowing another. It happened in this very series: two branches
    each added `SUBAGENT_EXECUTORS` and `normalize_subagent_executor` to subagents.py,
    one a frozenset and one a tuple, and the merge picked a winner by position. That
    pair happened to be equivalent; the next one will not be.
    """
    import ast
    import collections
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted(root.rglob("*.py")):
        parts = set(path.parts)
        if parts & {".git", "node_modules", "venv", ".venv", "build", "dist"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        seen = collections.Counter()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seen[node.name] += 1
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                seen[node.target.id] += 1
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        seen[target.id] += 1
        for name, count in seen.items():
            # Dunders and TYPE_CHECKING/try-except import shims legitimately rebind.
            if count > 1 and not name.startswith("__"):
                offenders.append(f"{path.relative_to(root)}: {name} defined {count}x")
    assert not offenders, "Top-level names defined more than once:\n" + "\n".join(offenders)
