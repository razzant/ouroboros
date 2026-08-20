"""Tests for repo_read slicing and per-tool truncation limits.

Also covers the core governance artifact invariants introduced in the
'Core Governance Artifacts' section of docs/DEVELOPMENT.md:
  - repo_read default max_lines raised to 2000 so ARCHITECTURE.md is
    readable in a single call.
  - BackgroundConsciousness._build_context() includes ARCHITECTURE.md.
  - Triad review prompt includes ARCHITECTURE.md even when not touched.
  - DEVELOPMENT.md contains the core governance artifact invariant rule.
"""

from unittest.mock import MagicMock


def _make_ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext
    ctx = MagicMock(spec=ToolContext)
    ctx.repo_dir = tmp_path
    def _repo_path(p):
        import ouroboros.utils as u
        return tmp_path / u.safe_relpath(p)
    ctx.repo_path.side_effect = _repo_path
    return ctx


def test_repo_read_full_file_has_header(tmp_path):
    from ouroboros.tools.core_file_tools import _repo_read
    f = tmp_path / "hello.py"
    f.write_text("line1\nline2\nline3\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    result = _repo_read(ctx, "hello.py")
    assert result.startswith("# hello.py — lines 1–3 of 3\n")


def test_repo_read_max_lines_slice(tmp_path):
    from ouroboros.tools.core_file_tools import _repo_read
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 101)) + "\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    result = _repo_read(ctx, "big.py", max_lines=10)
    assert result.startswith("# big.py — lines 1–10 of 100\n")
    assert "line11" not in result


def test_data_read_memory_file_never_truncated():
    from ouroboros.loop_tool_execution import _truncate_tool_result
    big = "m" * 70000
    result = _truncate_tool_result(big, "read_file", {"path": "memory/scratchpad.md"})
    assert result == big


def test_data_read_cold_start_returns_sentinel(tmp_path):
    from ouroboros.tools.core_file_tools import _data_read

    ctx = MagicMock()
    ctx.drive_path.side_effect = lambda p: tmp_path / p

    result = _data_read(ctx, "memory/knowledge/patterns.md")
    assert "DATA_NOT_YET_CREATED" in result
    assert "memory/knowledge/patterns.md" in result
    assert "lazily on first write" in result


def test_data_read_existing_file_still_read_verbatim(tmp_path):
    from ouroboros.tools.core_file_tools import _data_read

    target = tmp_path / "memory" / "scratchpad.md"
    target.parent.mkdir(parents=True)
    target.write_text("real scratchpad content\n", encoding="utf-8")

    ctx = MagicMock()
    ctx.drive_path.side_effect = lambda p: tmp_path / p

    assert _data_read(ctx, "memory/scratchpad.md") == "real scratchpad content\n"


def test_data_read_propagates_non_filenotfound_errors(tmp_path, monkeypatch):
    import pytest
    import ouroboros.tools.core_file_tools as core_mod
    from ouroboros.tools.core_file_tools import _data_read

    ctx = MagicMock()
    ctx.drive_path.side_effect = lambda p: tmp_path / p

    def _raise_permission(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(core_mod, "read_text", _raise_permission)
    with pytest.raises(PermissionError):
        _data_read(ctx, "memory/scratchpad.md")

    def _raise_is_dir(path):
        raise IsADirectoryError(21, "Is a directory", str(path))

    monkeypatch.setattr(core_mod, "read_text", _raise_is_dir)
    with pytest.raises(IsADirectoryError):
        _data_read(ctx, "memory/knowledge/")


def test_data_read_toctou_race_handled_by_sentinel(tmp_path, monkeypatch):
    import ouroboros.tools.core_file_tools as core_mod
    from ouroboros.tools.core_file_tools import _data_read

    target = tmp_path / "memory" / "racy.md"
    target.parent.mkdir(parents=True)
    target.write_text("content that is about to vanish\n", encoding="utf-8")

    ctx = MagicMock()
    ctx.drive_path.side_effect = lambda p: tmp_path / p

    def _raise_file_not_found(path):
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(core_mod, "read_text", _raise_file_not_found)

    result = _data_read(ctx, "memory/racy.md")
    assert "DATA_NOT_YET_CREATED" in result
    assert "memory/racy.md" in result


def test_data_read_sentinel_narrower_for_non_memory_paths(tmp_path):
    from ouroboros.tools.core_file_tools import _data_read

    ctx = MagicMock()
    ctx.drive_path.side_effect = lambda p: tmp_path / p

    mem_result = _data_read(ctx, "memory/knowledge/patterns.md")
    assert "DATA_NOT_YET_CREATED" in mem_result
    assert "lazily on first write" in mem_result

    non_mem_result = _data_read(ctx, "logs/nonexistent.jsonl")
    assert "DATA_NOT_YET_CREATED" in non_mem_result
    assert "lazily on first write" not in non_mem_result
    assert "not guaranteed" in non_mem_result


def test_repo_read_prompt_file_never_truncated():
    from ouroboros.loop_tool_execution import _truncate_tool_result
    big = "p" * 90000
    result = _truncate_tool_result(big, "read_file", {"path": "prompts/SYSTEM.md"})
    assert result == big


def test_repo_commit_results_never_truncated():
    from ouroboros.loop_tool_execution import _truncate_tool_result
    big = "r" * 90000
    assert _truncate_tool_result(big, "commit_reviewed") == big
    assert _truncate_tool_result(big, "commit_reviewed") == big
    assert _truncate_tool_result(big, "task_acceptance_review") == big
    assert _truncate_tool_result(big, "skill_review") == big


def test_self_check_returns_bool_and_interval_15():
    from ouroboros.loop_nudges import _maybe_inject_self_check
    messages = []
    usage = {"cost": 0}
    progress_calls = []
    assert _maybe_inject_self_check(14, 200, messages, usage, progress_calls.append) is False
    assert _maybe_inject_self_check(15, 200, messages, usage, progress_calls.append) is True
    assert "CHECKPOINT" in messages[0]["content"]


def test_advisory_pre_review_results_never_truncated():
    """advisory_pre_review results must not be truncated (full JSON needed)."""
    from ouroboros.loop_tool_execution import _truncate_tool_result
    big = "a" * 90000
    assert _truncate_tool_result(big, "advisory_review") == big


def test_review_status_results_never_truncated():
    """review_status results must not be truncated (full JSON needed)."""
    from ouroboros.loop_tool_execution import _truncate_tool_result
    big = "b" * 90000
    assert _truncate_tool_result(big, "review_status") == big


def test_child_task_handoff_results_never_truncated():
    """Child-task handoff tools must return the full result to the parent."""
    from ouroboros.loop_tool_execution import _truncate_tool_result
    big = "c" * 90000
    assert _truncate_tool_result(big, "get_task_result") == big
    assert _truncate_tool_result(big, "wait_task") == big
    assert _truncate_tool_result(big, "wait_tasks") == big


# ---------------------------------------------------------------------------
# Core governance artifact invariants
# ---------------------------------------------------------------------------

def test_repo_read_default_max_lines_is_2000(tmp_path):
    """Default max_lines must be 2000 so ARCHITECTURE.md (~1285 lines) fits in one call."""
    import inspect
    from ouroboros.tools.core_file_tools import _repo_read
    sig = inspect.signature(_repo_read)
    default = sig.parameters["max_lines"].default
    assert default == 2000, (
        f"repo_read default max_lines should be 2000, got {default}. "
        "ARCHITECTURE.md is longer than 1050 lines and must be readable in a single call."
    )


def test_repo_read_schema_default_is_2000():
    """Tool schema for repo_read must advertise default 2000 for max_lines."""
    from ouroboros.tools.core import get_tools
    tools = {t.name: t for t in get_tools()}
    assert "read_file" in tools
    schema = tools["read_file"].schema
    ml_param = schema["parameters"]["properties"]["max_lines"]
    assert ml_param["default"] == 2000, (
        f"repo_read schema default for max_lines should be 2000, got {ml_param['default']}."
    )


def test_repo_read_can_read_architecture_md_in_one_call(tmp_path):
    """A file of ARCHITECTURE.md length (~1285 lines) is returned fully with default max_lines."""
    from ouroboros.tools.core_file_tools import _repo_read
    # Simulate a file slightly longer than the old 1050-line default
    n_lines = 1300
    content = "\n".join(f"line {i}" for i in range(1, n_lines + 1)) + "\n"
    arch = tmp_path / "docs"
    arch.mkdir()
    (arch / "ARCHITECTURE.md").write_text(content, encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    result = _repo_read(ctx, "docs/ARCHITECTURE.md")
    # Header declares full file was read
    assert f"lines 1\u2013{n_lines} of {n_lines}" in result
    assert f"line {n_lines}" in result


def test_consciousness_context_includes_architecture_md(tmp_path):
    """BackgroundConsciousness._build_context must include ARCHITECTURE.md section."""
    import queue
    from unittest.mock import patch, MagicMock

    # Minimal fake repo tree
    repo_dir = tmp_path / "repo"
    (repo_dir / "docs").mkdir(parents=True)
    (repo_dir / "BIBLE.md").write_text("# BIBLE", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text(
        "# ARCHITECTURE\n\nThis is the architecture doc.", encoding="utf-8"
    )
    (repo_dir / "prompts" / "CONSCIOUSNESS.md").mkdir(parents=True, exist_ok=True) if False else None
    # Consciousness prompt (optional — can be missing)

    drive_root = tmp_path / "data"
    (drive_root / "logs").mkdir(parents=True)
    (drive_root / "state").mkdir(parents=True)
    (drive_root / "state" / "state.json").write_text("{}", encoding="utf-8")
    (drive_root / "memory").mkdir(parents=True)

    eq = queue.Queue()

    from ouroboros.consciousness import BackgroundConsciousness

    with patch.object(BackgroundConsciousness, "_build_registry", return_value=MagicMock()), \
         patch("ouroboros.consciousness.build_memory_sections", return_value=[]), \
         patch("ouroboros.consciousness.build_health_invariants", return_value=""), \
         patch("ouroboros.consciousness.build_runtime_section", return_value="## Runtime\n\nok"), \
         patch("ouroboros.consciousness.build_recent_sections", return_value=[]):
        bc = BackgroundConsciousness(
            drive_root=drive_root,
            repo_dir=repo_dir,
            event_queue=eq,
            owner_chat_id_fn=lambda: 1,
        )
        context = bc._build_context()

    assert "## ARCHITECTURE.md" in context, (
        "BackgroundConsciousness._build_context() must include a '## ARCHITECTURE.md' section. "
        "This is a core governance artifact — see docs/DEVELOPMENT.md."
    )
    assert "This is the architecture doc." in context


def test_consciousness_context_architecture_before_knowledge_base(tmp_path):
    """ARCHITECTURE.md section must come before knowledge base in consciousness context."""
    import queue
    from unittest.mock import patch, MagicMock

    repo_dir = tmp_path / "repo"
    (repo_dir / "docs").mkdir(parents=True)
    (repo_dir / "BIBLE.md").write_text("# BIBLE", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text("# ARCH CONTENT", encoding="utf-8")

    drive_root = tmp_path / "data"
    (drive_root / "logs").mkdir(parents=True)
    (drive_root / "state").mkdir(parents=True)
    (drive_root / "state" / "state.json").write_text("{}", encoding="utf-8")
    (drive_root / "memory").mkdir(parents=True)
    kb = drive_root / "memory" / "knowledge"
    kb.mkdir(parents=True)
    (kb / "index-full.md").write_text("# Knowledge base index", encoding="utf-8")

    eq = queue.Queue()
    from ouroboros.consciousness import BackgroundConsciousness

    with patch.object(BackgroundConsciousness, "_build_registry", return_value=MagicMock()), \
         patch("ouroboros.consciousness.build_memory_sections", return_value=[]), \
         patch("ouroboros.consciousness.build_health_invariants", return_value=""), \
         patch("ouroboros.consciousness.build_runtime_section", return_value="## Runtime\n\nok"), \
         patch("ouroboros.consciousness.build_recent_sections", return_value=[]):
        bc = BackgroundConsciousness(
            drive_root=drive_root,
            repo_dir=repo_dir,
            event_queue=eq,
            owner_chat_id_fn=lambda: 1,
        )
        context = bc._build_context()

    arch_pos = context.find("## ARCHITECTURE.md")
    kb_pos = context.find("## Knowledge base")
    assert arch_pos != -1, "ARCHITECTURE.md section not found in consciousness context"
    if kb_pos != -1:
        assert arch_pos < kb_pos, (
            "ARCHITECTURE.md must appear before the knowledge base in consciousness context"
        )


def test_triad_review_prompt_includes_architecture_md(tmp_path):
    """Triad review prompt must include ARCHITECTURE.md even when it is not in touched files."""
    from ouroboros.tools.review import (
        _REVIEW_PROMPT_TEMPLATE_DYNAMIC,
        _REVIEW_PROMPT_TEMPLATE_STABLE,
    )
    from ouroboros.tools.review_helpers import CRITICAL_FINDING_CALIBRATION, load_governance_doc

    # Write a fake ARCHITECTURE.md
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    arch_content = "# ARCHITECTURE TEST CONTENT UNIQUE_MARKER_12345"
    (docs_dir / "ARCHITECTURE.md").write_text(arch_content, encoding="utf-8")

    arch_text = load_governance_doc(tmp_path, "docs/ARCHITECTURE.md", on_missing="explicit")
    assert arch_text == arch_content, (
        "load_governance_doc should read the full file content"
    )

    # Verify the STABLE template (the cache-marked governance prefix) carries
    # the {architecture_section} placeholder.
    assert "{architecture_section}" in _REVIEW_PROMPT_TEMPLATE_STABLE, (
        "_REVIEW_PROMPT_TEMPLATE_STABLE must contain {architecture_section} placeholder. "
        "ARCHITECTURE.md must be a first-class section in the triad review prompt."
    )

    # Render both template halves and verify the content appears
    rendered = _REVIEW_PROMPT_TEMPLATE_STABLE.format(
        preamble="PREAMBLE",
        critical_calibration=CRITICAL_FINDING_CALIBRATION,
        json_contract="JSON",
        anti_pattern_lock_guard="LOCK",
        checklist_section="CHECKLIST",
        dev_guide_text="DEVGUIDE",
        architecture_section=arch_text,
    ) + _REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(
        goal_section="GOAL",
        scope_section="",
        current_files_section="FILES",
        rebuttal_section="",
        review_history_section="",
        diff_text="DIFF",
        changed_files="changed_file.py",
    )
    assert "UNIQUE_MARKER_12345" in rendered, (
        "ARCHITECTURE.md content must appear in the rendered triad review prompt"
    )
    assert "## ARCHITECTURE.md" in rendered


def test_governance_doc_load_emits_explicit_omission_marker_on_missing(tmp_path):
    """Loading ARCHITECTURE.md for the triad prompt must emit a visible
    ``[⚠️ OMISSION: ...]`` marker (not a silent empty string) when absent.

    DEVELOPMENT.md "No silent truncation" forbids the silent-empty-string
    fallback — invisible omission of a core governance artifact in a
    triad-review prompt would let reviewers PASS without ever seeing the
    architectural rationale they are supposed to grade against. The triad
    prompt builder routes through the SSOT ``review_helpers.load_governance_doc``
    with ``on_missing="explicit"`` which returns
    ``[⚠️ OMISSION: docs/ARCHITECTURE.md not found at <path>]``.
    """
    from ouroboros.tools.review_helpers import load_governance_doc
    result = load_governance_doc(tmp_path, "docs/ARCHITECTURE.md", on_missing="explicit")
    assert result.startswith("[⚠️ OMISSION:"), (
        f"Missing ARCHITECTURE.md should yield an explicit omission marker, got: {result!r}"
    )
    assert "docs/ARCHITECTURE.md" in result
    # Never raises — the function still degrades gracefully.


def test_consciousness_logs_warning_when_architecture_md_missing(tmp_path):
    """BackgroundConsciousness._build_context must log a warning when ARCHITECTURE.md is absent.

    Per the Core Governance Artifacts invariant in docs/DEVELOPMENT.md:
    'Log a warning if the file is missing or unavailable — do not silently skip.'

    Uses a fresh MagicMock() per getLogger call to avoid mutating real logger singletons.
    """
    import queue
    from unittest.mock import patch, MagicMock

    repo_dir = tmp_path / "repo"
    (repo_dir / "docs").mkdir(parents=True)
    (repo_dir / "BIBLE.md").write_text("# BIBLE", encoding="utf-8")
    # Deliberately do NOT create docs/ARCHITECTURE.md

    drive_root = tmp_path / "data"
    (drive_root / "logs").mkdir(parents=True)
    (drive_root / "state").mkdir(parents=True)
    (drive_root / "state" / "state.json").write_text("{}", encoding="utf-8")
    (drive_root / "memory").mkdir(parents=True)

    eq = queue.Queue()
    from ouroboros.consciousness import BackgroundConsciousness

    # A single shared mock logger so all getLogger(name) calls return the same object.
    # This avoids mutating any real logger singleton.
    mock_logger = MagicMock()

    with patch.object(BackgroundConsciousness, "_build_registry", return_value=MagicMock()), \
         patch("ouroboros.consciousness.build_memory_sections", return_value=[]), \
         patch("ouroboros.consciousness.build_health_invariants", return_value=""), \
         patch("ouroboros.consciousness.build_runtime_section", return_value="## Runtime\n\nok"), \
         patch("ouroboros.consciousness.build_recent_sections", return_value=[]), \
         patch("logging.getLogger", return_value=mock_logger):
        bc = BackgroundConsciousness(
            drive_root=drive_root,
            repo_dir=repo_dir,
            event_queue=eq,
            owner_chat_id_fn=lambda: 1,
        )
        context = bc._build_context()

    # 1. ARCHITECTURE.md section must be absent from context (file doesn't exist)
    assert "## ARCHITECTURE.md" not in context, (
        "ARCHITECTURE.md section should not appear when file is missing"
    )

    # 2. mock_logger.warning must have been called at least once with ARCHITECTURE.md in the message
    warning_messages = [
        str(call_args)
        for call_args in mock_logger.warning.call_args_list
    ]
    arch_warnings = [w for w in warning_messages if "ARCHITECTURE.md" in w]
    assert arch_warnings, (
        "BackgroundConsciousness._build_context() must call logger.warning with 'ARCHITECTURE.md' "
        "when the file is missing. Core Governance Artifacts invariant in DEVELOPMENT.md. "
        f"All warning calls: {warning_messages}"
    )
    assert any("not found" in w or "empty" in w for w in arch_warnings), (
        f"Warning message must indicate the file is missing/empty, got: {arch_warnings}"
    )


def test_development_md_contains_core_governance_invariant():
    """docs/DEVELOPMENT.md must contain the core governance artifact invariant rule."""
    import pathlib
    dev_md = pathlib.Path(__file__).resolve().parent.parent / "docs" / "DEVELOPMENT.md"
    assert dev_md.exists(), "docs/DEVELOPMENT.md must exist"
    content = dev_md.read_text(encoding="utf-8")

    required_phrases = [
        "Core Governance Artifacts",
        "BIBLE.md",
        "docs/ARCHITECTURE.md",
        "first-class context",
    ]
    for phrase in required_phrases:
        assert phrase in content, (
            f"docs/DEVELOPMENT.md must contain '{phrase}' as part of the "
            "core governance artifact invariant. "
            "This ensures the rule is documented and checkable."
        )
