"""Tests for repo_read slicing and per-tool truncation limits.

Also covers the core governance artifact invariants introduced in the
'Core Governance Artifacts' section of docs/DEVELOPMENT.md:
  - repo_read default max_lines raised to 2000 so ARCHITECTURE.md is
    readable in a single call.
  - the wake-up's Main context (build_llm_messages) includes ARCHITECTURE.md.
  - Triad review prompt includes ARCHITECTURE.md even when not touched.
  - DEVELOPMENT.md contains the core governance artifact invariant rule.
"""

from unittest.mock import MagicMock
from tests._governance_docs_shared import development_text


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
    from ouroboros.tools.core import _repo_read
    f = tmp_path / "hello.py"
    f.write_text("line1\nline2\nline3\n", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    result = _repo_read(ctx, "hello.py")
    assert result.startswith("# hello.py — lines 1–3 of 3\n")


def test_repo_read_max_lines_slice(tmp_path):
    from ouroboros.tools.core import _repo_read
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
    from ouroboros.tools.core import _data_read

    ctx = MagicMock()
    ctx.drive_path.side_effect = lambda p: tmp_path / p

    result = _data_read(ctx, "memory/knowledge/patterns.md")
    assert "DATA_NOT_YET_CREATED" in result
    assert "memory/knowledge/patterns.md" in result
    assert "lazily on first write" in result


def test_data_read_existing_file_still_read_verbatim(tmp_path):
    from ouroboros.tools.core import _data_read

    target = tmp_path / "memory" / "scratchpad.md"
    target.parent.mkdir(parents=True)
    target.write_text("real scratchpad content\n", encoding="utf-8")

    ctx = MagicMock()
    ctx.drive_path.side_effect = lambda p: tmp_path / p

    assert _data_read(ctx, "memory/scratchpad.md") == "real scratchpad content\n"


def test_data_read_propagates_non_filenotfound_errors(tmp_path, monkeypatch):
    import pytest
    import ouroboros.tools.core_file_tools as core_mod
    from ouroboros.tools.core import _data_read

    ctx = MagicMock()
    ctx.drive_path.side_effect = lambda p: tmp_path / p

    def _raise_permission(path, extent=None):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(core_mod, "_read_source_text", _raise_permission)
    with pytest.raises(PermissionError):
        _data_read(ctx, "memory/scratchpad.md")

    def _raise_is_dir(path, extent=None):
        raise IsADirectoryError(21, "Is a directory", str(path))

    monkeypatch.setattr(core_mod, "_read_source_text", _raise_is_dir)
    with pytest.raises(IsADirectoryError):
        _data_read(ctx, "memory/knowledge/")


def test_data_read_toctou_race_handled_by_sentinel(tmp_path, monkeypatch):
    import ouroboros.tools.core_file_tools as core_mod
    from ouroboros.tools.core import _data_read

    target = tmp_path / "memory" / "racy.md"
    target.parent.mkdir(parents=True)
    target.write_text("content that is about to vanish\n", encoding="utf-8")

    ctx = MagicMock()
    ctx.drive_path.side_effect = lambda p: tmp_path / p

    def _raise_file_not_found(path, extent=None):
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(core_mod, "_read_source_text", _raise_file_not_found)

    result = _data_read(ctx, "memory/racy.md")
    assert "DATA_NOT_YET_CREATED" in result
    assert "memory/racy.md" in result


def test_data_read_sentinel_narrower_for_non_memory_paths(tmp_path):
    from ouroboros.tools.core import _data_read

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
    from ouroboros.loop import _maybe_inject_self_check
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
    """Default max_lines must be 2000 so ARCHITECTURE.md fits in one call."""
    import inspect
    from ouroboros.tools.core import _repo_read
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
    """A file longer than the historical 1050-line default is returned fully with default max_lines."""
    from ouroboros.tools.core import _repo_read
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


def _wake_context(tmp_path):
    """The system text a consciousness wake-up gets: Main's own builder over a wake-shaped task."""
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory

    class FakeEnv:
        def drive_path(self, p):
            return tmp_path / "data" / p

        def repo_path(self, p):
            return tmp_path / "repo" / p

        @property
        def repo_dir(self):
            return tmp_path / "repo"

        @property
        def drive_root(self):
            return tmp_path / "data"

    drive_root = tmp_path / "data"
    for rel in ("logs", "state", "memory"):
        (drive_root / rel).mkdir(parents=True, exist_ok=True)
    (drive_root / "state" / "state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repo" / "prompts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
    task = {"id": "wake1", "type": "task", "text": "[Wake-up · heartbeat]", "_is_direct_chat": True,
            "metadata": {"initiator": "consciousness", "usage_category": "consciousness",
                         "consciousness_autonomy": "act"}}
    messages, _cap = build_llm_messages(env=FakeEnv(), memory=Memory(drive_root=drive_root, repo_dir=tmp_path / "repo"), task=task)
    return "\n\n".join(block["text"] for block in messages[0]["content"])


def test_wake_context_includes_architecture_md(tmp_path):
    """A consciousness wake-up reads the same governance artifacts as any Main turn."""
    repo_dir = tmp_path / "repo"
    (repo_dir / "docs").mkdir(parents=True)
    (repo_dir / "BIBLE.md").write_text("# BIBLE", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text(
        "# ARCHITECTURE\n\nThis is the architecture doc.", encoding="utf-8"
    )

    context = _wake_context(tmp_path)

    assert "## ARCHITECTURE.md" in context, (
        "the wake-up's Main context must include a '## ARCHITECTURE.md' section. "
        "This is a core governance artifact — see docs/DEVELOPMENT.md."
    )
    assert "This is the architecture doc." in context


def test_wake_context_architecture_before_knowledge_base(tmp_path):
    """ARCHITECTURE.md section must come before knowledge base in the wake-up's context."""
    repo_dir = tmp_path / "repo"
    (repo_dir / "docs").mkdir(parents=True)
    (repo_dir / "BIBLE.md").write_text("# BIBLE", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text("# ARCH CONTENT", encoding="utf-8")
    kb = tmp_path / "data" / "memory" / "knowledge"
    kb.mkdir(parents=True)
    (kb / "index-full.md").write_text("# Knowledge base index", encoding="utf-8")

    context = _wake_context(tmp_path)

    arch_pos = context.find("## ARCHITECTURE.md")
    kb_pos = context.find("## Knowledge base")
    assert arch_pos != -1, "ARCHITECTURE.md section not found in the wake-up's context"
    if kb_pos != -1:
        assert arch_pos < kb_pos, (
            "ARCHITECTURE.md must appear before the knowledge base in the wake-up's context"
        )


def test_triad_review_prompt_reaches_architecture_md_by_navigation():
    """The triad packet no longer inlines the reference books whole.

    Owner decision 2026-09-17: one SSOT (`tools/governance_context.py`) tiers the
    governance corpus. The architecture map is delivered as book navigation for
    every reviewer, with the sections that name a touched file selected for a
    packet row. Every chapter remains named with line ranges, without promising
    tools to a packet recipient. The templates carry no whole-book placeholder, and
    nothing may be omitted silently."""
    from ouroboros.tools.governance_context import governance_context
    from ouroboros.tools.review import (
        _REVIEW_PROMPT_TEMPLATE_DYNAMIC,
        _REVIEW_PROMPT_TEMPLATE_STABLE,
    )
    from ouroboros.tools.review_helpers import REPO_ROOT

    for placeholder in ("{architecture_section}", "{dev_guide_text}", "{design_text}"):
        assert placeholder not in _REVIEW_PROMPT_TEMPLATE_STABLE, (
            f"{placeholder} inlines a whole reference book into every api row; the "
            "governance tiers deliver the map as navigation instead."
        )
        assert placeholder not in _REVIEW_PROMPT_TEMPLATE_DYNAMIC

    context = governance_context(
        REPO_ROOT,
        surface="triad",
        touched_paths=["web/modules/chat.js"],
        usable_window_tokens=200_000,
        delivery="packet",
        checklist_section_text="## Change Review Checklist\n",
        already_inline=("BIBLE.md", "docs/CHECKLISTS_ARCHIVE.md"),
    )
    # The map is named, addressable and never inlined whole.
    assert "docs/ARCHITECTURE.md" in [row["path"] for row in context.manifest]
    assert 'read_file(root="system_repo"' not in context.navigation
    assert "index of sources not inlined" in context.navigation
    assert "docs/architecture/03-web-ui-pages-and-buttons.md" in context.navigation
    inlined_whole = [row["path"] for row in context.manifest
                     if row["disposition"] == "inline" and row["tier"] == 3 and "#" not in row["path"]]
    assert inlined_whole == []
    # Every governance document has a disposition and a reason: no silent drop.
    assert all(row["reason"] and row["disposition"] in ("inline", "navigation")
               for row in context.manifest)


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


def test_wake_context_logs_warning_when_architecture_md_missing(tmp_path, caplog):
    """The Main context builder must log a warning when ARCHITECTURE.md is absent — the
    wake-up (an ordinary Main turn) inherits it.

    Per the Core Governance Artifacts invariant in docs/DEVELOPMENT.md:
    'Log a warning if the file is missing or unavailable — do not silently skip.'
    """
    import logging

    repo_dir = tmp_path / "repo"
    (repo_dir / "docs").mkdir(parents=True)
    (repo_dir / "BIBLE.md").write_text("# BIBLE", encoding="utf-8")
    # Deliberately do NOT create docs/ARCHITECTURE.md

    with caplog.at_level(logging.WARNING, logger="ouroboros.context"):
        context = _wake_context(tmp_path)

    # 1. ARCHITECTURE.md section must be absent from context (file doesn't exist)
    assert "## ARCHITECTURE.md" not in context, (
        "ARCHITECTURE.md section should not appear when file is missing"
    )

    # 2. a warning naming ARCHITECTURE.md must have been logged
    arch_warnings = [record.getMessage() for record in caplog.records
                     if record.levelno >= logging.WARNING and "ARCHITECTURE.md" in record.getMessage()]
    assert arch_warnings, (
        "the Main context builder must log a warning naming 'ARCHITECTURE.md' when the file "
        "is missing. Core Governance Artifacts invariant in DEVELOPMENT.md. "
        f"All records: {[r.getMessage() for r in caplog.records]}"
    )
    assert any("not found" in w or "empty" in w or "unavailable" in w for w in arch_warnings), (
        f"Warning message must indicate the file is missing/unavailable, got: {arch_warnings}"
    )
def test_development_md_contains_core_governance_invariant():
    """docs/DEVELOPMENT.md must keep the Core Governance Artifacts chapter naming the governance files."""
    import pathlib
    dev_md = pathlib.Path(__file__).resolve().parent.parent / "docs" / "DEVELOPMENT.md"
    assert dev_md.exists(), "docs/DEVELOPMENT.md must exist"
    content = development_text()

    required_phrases = [
        "Core Governance Artifacts",
        "BIBLE.md",
        "docs/ARCHITECTURE.md",
    ]
    for phrase in required_phrases:
        assert phrase in content, (
            f"docs/DEVELOPMENT.md must name '{phrase}' in the "
            "core governance artifact chapter."
        )
