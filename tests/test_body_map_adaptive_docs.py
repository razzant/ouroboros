"""Body Map + Reference Atlas, adaptive body contact points, two-tier size rule, and audited fixes."""
from __future__ import annotations

import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def test_unified_body_map_is_byte_identical_across_modes_and_cache_invariant():
    from types import SimpleNamespace
    from ouroboros.context_layout import body_map_section, reference_doc_sections
    from ouroboros.reference_books import load_reference_book
    from ouroboros.section_floor import HostSection

    arch_book = load_reference_book(REPO, "architecture")
    dev_book = load_reference_book(REPO, "development")
    env = SimpleNamespace(repo_dir=REPO, repo_path=lambda rel: REPO / rel)

    rendered = {}
    for mode in ("max", "low", "nano"):
        parts, dev = reference_doc_sections(
            env, context_mode=mode, include_development=False,
            books=(arch_book, dev_book), required_book=False,
        )
        assert dev == ""
        assert len(parts) == 1
        assert isinstance(parts[0], HostSection) and parts[0].section_id == "body_map"
        rendered[mode] = str(parts[0])

    assert rendered["max"] == rendered["low"] == rendered["nano"]
    text = rendered["max"]
    assert "## Chapters" in text and "## Reference Atlas" in text
    assert "docs/DEVELOPMENT.md" in text and "docs/DESIGN.md" in text and "docs/PERSISTENCE.md" in text
    assert "— lines " not in text
    assert "# Ouroboros v" not in text

    bm = body_map_section(arch_book)
    assert isinstance(bm, HostSection) and bm.operative_view is not None
    assert len(bm.operative_view) < len(bm.full_text)
    assert "docs/architecture/06-1-loop-tools-and-safety.md" in bm.operative_view

    # Explicit context_requires_development=true still loads DEVELOPMENT whole in any mode.
    for mode in ("max", "low", "nano"):
        _parts, dev = reference_doc_sections(
            env, context_mode=mode, include_development=True,
            books=(arch_book, dev_book), required_book=True,
        )
        assert dev.startswith("## DEVELOPMENT.md\n\n") and "Role and authority" in dev


def test_two_tier_size_rule_and_staged_index_balance(tmp_path):
    from ouroboros.reference_books import (
        BOOK_CHAPTER_MAX_LINES,
        body_map_bytes,
        book_balances,
        chapter_band_overages,
    )
    from ouroboros.review import diff_pulse

    assert chapter_band_overages(REPO, "architecture") == ()
    assert chapter_band_overages(REPO, "development") == ()
    assert (body_map_bytes(REPO) or 0) > 1000

    # Synthetic repo with Body Map + Reference Atlas and staged vs worktree drift (Bug #3).
    repo = tmp_path / "repo"
    (repo / "docs/architecture").mkdir(parents=True)
    entry = (
        "# Body Map\n\nIntro.\n\n## Chapters\n\n"
        "- [1. One](architecture/01-one.md)\n\n"
        "## Reference Atlas\n\n- `docs/DESIGN.md` — UI\n"
    )
    ch1 = "# 1. One\n\nShort opening lead.\n\nBody paragraph.\n"
    (repo / "docs/ARCHITECTURE.md").write_text(entry, encoding="utf-8", newline="\n")
    (repo / "docs/architecture/01-one.md").write_text(ch1, encoding="utf-8", newline="\n")
    (repo / "docs/DESIGN.md").write_text("# Design\n\nRule.\n", encoding="utf-8", newline="\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    _git(repo, "remote", "add", "canonical", "https://github.com/razzant/ouroboros.git")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")

    # Growing chapter body inside band: Tier B reports factual delta with owed == 0; Body Map unchanged -> owed == 0.
    staged_ch1 = ch1 + "Extra rationale line.\n"
    (repo / "docs/architecture/01-one.md").write_text(staged_ch1, encoding="utf-8", newline="\n")
    _git(repo, "add", "docs/architecture/01-one.md")
    # Dirty worktree AFTER staging: staged=True pulse must read staged bytes, not worktree bytes!
    (repo / "docs/architecture/01-one.md").write_text(staged_ch1 + "UNSTAGED WORKTREE EXTRA.\n", encoding="utf-8", newline="\n")
    staged_balances = {b.book_id: b for b in book_balances(repo, ["docs/architecture/01-one.md"], source="index")}
    wt_balances = {b.book_id: b for b in book_balances(repo, ["docs/architecture/01-one.md"], source="worktree")}
    assert staged_balances["architecture"].vs_head == len("Extra rationale line.\n")
    assert wt_balances["architecture"].vs_head == len("Extra rationale line.\nUNSTAGED WORKTREE EXTRA.\n")
    assert staged_balances["architecture"].owed == 0
    assert staged_balances["body_map"].vs_head == 0 and staged_balances["body_map"].owed == 0

    pulse = diff_pulse(repo, staged=True)
    staged_delta = len("Extra rationale line.\n")
    wt_delta = len("Extra rationale line.\nUNSTAGED WORKTREE EXTRA.\n")
    assert f"+{staged_delta} B vs HEAD" in pulse
    assert f"+{wt_delta} B vs HEAD" not in pulse

    # Growing chapter opening grows Tier A body_map -> owed > 0.
    (repo / "docs/architecture/01-one.md").write_text(
        "# 1. One\n\nShort opening lead made noticeably longer.\n\nBody paragraph.\n",
        encoding="utf-8", newline="\n",
    )
    bm_bal = {b.book_id: b for b in book_balances(repo, ["docs/architecture/01-one.md"])}["body_map"]
    assert bm_bal.owed > 0

    # Standalone spec inside band has owed == 0; over-band standalone (>80 KB) carries shrink-only debt.
    (repo / "docs/DESIGN.md").write_text("# Design\n\nRule.\nAnother rule.\n", encoding="utf-8", newline="\n")
    design_bal = book_balances(repo, ["docs/DESIGN.md"])[0]
    assert design_bal.vs_head == len("Another rule.\n") and design_bal.tier == "manual" and design_bal.owed == 0

    giant = "# Design\n\n" + ("R" * 100 + "\n") * 900
    (repo / "docs/DESIGN.md").write_text(giant, encoding="utf-8", newline="\n")
    _git(repo, "add", "docs/DESIGN.md")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "giant")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    (repo / "docs/DESIGN.md").write_text(giant + "Extra giant byte.\n", encoding="utf-8", newline="\n")
    giant_bal = {b.book_id: b for b in book_balances(repo, None)}["docs/DESIGN.md"]
    assert giant_bal.tier == "shrink_only" and giant_bal.owed == len("Extra giant byte.\n")

    # Oversized chapter triggers physical one-read band overage.
    (repo / "docs/architecture/01-one.md").write_text(
        "# 1. One\n\nOpening.\n\n" + ("line\n" * (BOOK_CHAPTER_MAX_LINES + 5)),
        encoding="utf-8", newline="\n",
    )
    over = chapter_band_overages(repo, "architecture")
    assert len(over) == 1 and over[0][0] == "docs/architecture/01-one.md" and over[0][1] > BOOK_CHAPTER_MAX_LINES


def test_read_file_body_contact_note_is_own_body_code_only_and_preserves_read_view(tmp_path):
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.tools.tool_result import _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar

    ctx = ToolContext(repo_dir=REPO, drive_root=tmp_path / "data", task_id="t-read")
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        out = _read_file(ctx, "ouroboros/review.py", root="system_repo", max_lines=10)
        published = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert "[Body manual: docs/architecture/06-4-self-change-and-review.md" in out
    assert "docs/architecture/10-key-invariants.md" in out
    assert ctx.last_read_view is not None and ctx.last_read_view["complete_sha256"]
    assert published is not sentinel and published.text == out and not published.host_annotations
    # Full registry dispatch preserves the exact same footer cleanly.
    reg = ToolRegistry(repo_dir=REPO, drive_root=tmp_path / "data2")
    reg_out = reg.execute("read_file", {"path": "ouroboros/review.py", "root": "system_repo", "max_lines": 10})
    assert "[Body manual: docs/architecture/06-4-self-change-and-review.md" in reg_out

    # Docs and tests never get a [Body manual:] footer.
    doc_out = _read_file(ctx, "docs/ARCHITECTURE.md", root="system_repo", max_lines=10)
    assert "[Body manual:" not in doc_out
    test_out = _read_file(ctx, "tests/test_smoke.py", root="system_repo", max_lines=10)
    assert "[Body manual:" not in test_out

    # Foreign repo that happens to have ouroboros/review.py never gets an own-body note.
    foreign = tmp_path / "foreign"
    (foreign / "ouroboros").mkdir(parents=True)
    (foreign / "ouroboros/review.py").write_text('"""Foreign."""\nX = 1\n', encoding="utf-8")
    f_ctx = ToolContext(repo_dir=REPO, drive_root=tmp_path / "data", task_id="t-foreign",
                        workspace_root=foreign, workspace_mode="external")
    f_out = _read_file(f_ctx, "ouroboros/review.py", root="active_workspace", max_lines=10)
    assert "[Body manual:" not in f_out


def test_diff_pulse_ownership_invariants_and_relocated_prose_set_diff(tmp_path):
    from ouroboros.review import diff_pulse

    repo = tmp_path / "repo"
    (repo / "ouroboros").mkdir(parents=True)
    (repo / "docs/architecture").mkdir(parents=True)
    (repo / "ouroboros/domains.toml").write_text(
        '[domains]\nD01 = "Core"\n[references]\nD01 = ["docs/architecture/01-core.md"]\n'
        '[modules]\n"ouroboros/core_mod.py" = "D01"\n',
        encoding="utf-8", newline="\n",
    )
    (repo / "ouroboros/core_mod.py").write_text('"""Core."""\nX = 1\n', encoding="utf-8", newline="\n")
    (repo / "docs/ARCHITECTURE.md").write_text(
        "# Map\n\nIntro.\n\n## Chapters\n\n"
        "- [1. Core](architecture/01-core.md)\n"
        "- [10. Invariants](architecture/10-key-invariants.md)\n",
        encoding="utf-8", newline="\n",
    )
    (repo / "docs/architecture/01-core.md").write_text(
        "# 1. Core\n\nCore lead.\n\nKept line.\nMoved line.\nDeleted rationale line.\n",
        encoding="utf-8", newline="\n",
    )
    (repo / "docs/architecture/10-key-invariants.md").write_text(
        "# 10. Key Invariants\n\nLead.\n\n1. **Core invariant.** `core_mod.py` never drops state.\n",
        encoding="utf-8", newline="\n",
    )
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")

    (repo / "ouroboros/core_mod.py").write_text('"""Core."""\nX = 2\n', encoding="utf-8", newline="\n")
    (repo / "docs/architecture/01-core.md").write_text(
        "# 1. Core\n\nCore lead.\n\nKept line.\n", encoding="utf-8", newline="\n",
    )
    (repo / "docs/architecture/10-key-invariants.md").write_text(
        "# 10. Key Invariants\n\nLead.\n\n1. **Core invariant.** `core_mod.py` never drops state.\nMoved line.\n",
        encoding="utf-8", newline="\n",
    )
    pulse = diff_pulse(repo, ["ouroboros/core_mod.py", "docs/architecture/01-core.md", "docs/architecture/10-key-invariants.md"])
    assert "Owning manuals: docs/architecture/01-core.md (touched); matching invariants: §10#1." in pulse
    assert "Deleted rationale line." in pulse
    assert "Moved line." not in pulse  # relocated verbatim -> zero false noise


def test_governance_context_carries_key_invariants_selectable_pre_heading_and_domain_modules():
    from ouroboros.tools.governance_context import (
        BODY_INVARIANTS_CHAPTER,
        BODY_LAYER,
        PACKET_DELIVERY,
        RETRIEVING_DELIVERY,
        governance_context,
    )

    pkt = governance_context(
        REPO, surface="commit", touched_paths=["ouroboros/review.py"],
        usable_window_tokens=200_000, delivery=PACKET_DELIVERY, layer=BODY_LAYER,
    )
    assert BODY_INVARIANTS_CHAPTER in pkt.inline_whole_documents
    assert "10.2 Closed roads" in pkt.inline_whole_documents[BODY_INVARIANTS_CHAPTER]
    assert any(row["path"] == "ouroboros/domains.toml#affected_domains" and row["disposition"] == "inline"
               for row in pkt.manifest)

    ret = governance_context(
        REPO, surface="commit", touched_paths=["ouroboros/review.py"],
        usable_window_tokens=200_000, delivery=RETRIEVING_DELIVERY, layer=BODY_LAYER,
    )
    assert any(row["path"] == BODY_INVARIANTS_CHAPTER and row["disposition"] == "required"
               for row in ret.manifest)


def test_plan_packet_retrieving_body_plan_lists_all_architecture_chapters():
    from ouroboros.reference_books import load_reference_book
    from ouroboros.tools.plan_packet import build_stable_prompt

    book = load_reference_book(REPO, "architecture")
    chapters = tuple(ch.source_path for ch in book.chapters)
    prompt = build_stable_prompt(
        checklist_section="Plan checklist", body=True,
        bible_text="BIBLE", architecture_text="ARCH",
        governance_by_retrieval=True,
        architecture_chapter_locators=chapters,
    )
    assert "docs/architecture/06-4-self-change-and-review.md" in prompt
    assert "docs/architecture/10-key-invariants.md" in prompt


def test_prepare_self_change_child_ok_and_closed_roads_at_body_gate(tmp_path):
    from ouroboros.body_candidate import closed_roads_note
    from ouroboros.tools.control_runtime import _prepare_self_change
    from ouroboros.tools.registry import ToolContext

    # Body-gate Closed roads helper loads real 10.2 Closed roads.
    ctx = ToolContext(repo_dir=REPO, drive_root=tmp_path / "data", task_id="t-parent")
    roads = closed_roads_note(ctx)
    assert "10.2 Closed roads" in roads and "10-key-invariants.md" in roads

    # Bug #4: ordinary unbound subagent gets non-failure OK receipt; resume= stays refused.
    child_ctx = ToolContext(
        repo_dir=REPO, drive_root=tmp_path / "data", task_id="t-child",
        task_constraint={"mode": "acting_subagent", "parent_task_id": "t-parent"},
    )
    ok_text = _prepare_self_change(child_ctx)
    assert ok_text.startswith("OK: no separate body candidate prepared")
    refused = _prepare_self_change(child_ctx, resume="cand-123")
    assert "⚠️ CANDIDATE_NOT_APPLICABLE" in refused


def test_audited_tool_schema_and_ui_observation_contracts():
    from ouroboros.openai_chat_custom import _compact_schema
    from ouroboros.physical_attempt_context import shown_mode
    from ouroboros.tools.control import _MESSAGE_DESTINATION_DESCRIPTION, _SCHEDULE_SUBAGENT_DESCRIPTION
    from ouroboros.tools import subagent_integration

    # Bug #5: enum compacting omits redundant string type, keeping schedule_subagent under 4096.
    compacted = _compact_schema({"type": "string", "enum": ["a", "b"], "description": "d"})
    assert "type" not in compacted and compacted["enum"] == ["a", "b"]
    assert "workspace_root" in _SCHEDULE_SUBAGENT_DESCRIPTION
    # Bug #6 & #7 descriptions
    assert "already in Main" in _MESSAGE_DESTINATION_DESCRIPTION
    integrate = next(t for t in subagent_integration.get_tools() if t.name == "integrate_delegated_patch")
    assert "protected_paths_grant" in integrate.schema["description"]
    # Unified body_map observation form
    assert shown_mode("max") == ("max", "body_map")
    assert shown_mode("low") == ("low", "body_map")
    assert shown_mode("nano") == ("nano", "body_map")


def test_expand_contract_version_carrier_and_cache_invariant_render():
    from ouroboros.reference_books import body_map_book, load_reference_book
    from ouroboros.tools.release_sync import extract_architecture_header_version, version_carrier_desyncs

    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    arch_disk = (REPO / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert extract_architecture_header_version(arch_disk) == version
    assert version_carrier_desyncs(version, arch_text=arch_disk) == []
    # Tolerant forward check allows future headerless retirement while still catching wrong versions.
    assert version_carrier_desyncs(version, arch_text="# Ouroboros — Body Map & Architecture Reference\n") == []
    assert version_carrier_desyncs(version, arch_text="# Ouroboros v0.0.1 — Body Map\n") != []
    # Rendered Block A' Body Map strips the transitional version token completely.
    rendered = body_map_book(load_reference_book(REPO, "architecture")).text
    assert f"v{version}" not in rendered and rendered.startswith("# Ouroboros — Body Map & Architecture Reference\n")


def test_contact_one_bound_candidate_broken_project_folder_symlink_and_exact_mentions(tmp_path):
    from ouroboros.body_candidate import own_body_reference_note
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.registry import ToolContext

    # 1) Bound candidate (serving != candidate): system_repo read still delivers Contact #1.
    cand = tmp_path / "cand"
    (cand / "ouroboros/tools").mkdir(parents=True)
    (cand / "docs/architecture").mkdir(parents=True)
    (cand / "ouroboros/domains.toml").write_text((REPO / "ouroboros/domains.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (cand / "ouroboros/review.py").write_text('"""Review."""\nX = 1\n', encoding="utf-8")
    ctx = ToolContext(
        repo_dir=cand, drive_root=tmp_path / "data", task_id="t-bound",
        serving_repo_dir=REPO, system_repo_dir=cand,
        task_metadata={"body_candidate": {"state": "bound", "path": str(cand), "repo_dir": str(REPO)}},
    )
    out = _read_file(ctx, "ouroboros/review.py", root="system_repo", max_lines=10)
    assert "[Body manual: docs/architecture/06-4-self-change-and-review.md" in out

    # 2) Symlink escaping candidate root gets no own-body note.
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 1\n", encoding="utf-8")
    escape = cand / "ouroboros/escape.py"
    escape.symlink_to(outside)
    assert own_body_reference_note(ctx, escape, "system_repo") == ""

    # 3) Broken-folder Project room: explicit system_repo read succeeds without crashing on active_repo_dir_for.
    broken = ToolContext(repo_dir=REPO, drive_root=tmp_path / "data", task_id="t-broken")
    broken._project_room_note = "project folder unavailable"
    broken._project_room_dir = None
    broken_out = _read_file(broken, "ouroboros/review.py", root="system_repo", max_lines=5)
    assert "[Body manual: docs/architecture/06-4-self-change-and-review.md" in broken_out

    # 4) Exact relative-path mention union: control_subagent_spec.py (D07) also surfaces 06-1 where it is named.
    spec_out = _read_file(broken, "ouroboros/tools/control_subagent_spec.py", root="system_repo", max_lines=5)
    assert "docs/architecture/06-1-loop-tools-and-safety.md" in spec_out


def test_contact_two_dedup_acting_child_and_direct_chapter_ten_read(tmp_path):
    from ouroboros.reference_books import closed_roads_section
    from ouroboros.tools.control_runtime import _prepare_self_change
    from ouroboros.tools.registry import ToolContext

    # 1) Direct read of 10-key-invariants.md works even when an unrelated Architecture chapter is missing.
    mini = tmp_path / "mini"
    (mini / "docs/architecture").mkdir(parents=True)
    (mini / "docs/ARCHITECTURE.md").write_text(
        "# Map\n\nIntro.\n\n## Chapters\n\n- [1. Missing](architecture/01-missing.md)\n- [10. Inv](architecture/10-key-invariants.md)\n",
        encoding="utf-8",
    )
    (mini / "docs/architecture/10-key-invariants.md").write_text(
        "# 10. Key Invariants\n\nLead.\n\n### 10.2 Closed roads\n\n- Never reopen road A.\n",
        encoding="utf-8",
    )
    assert "Never reopen road A." in closed_roads_section(mini)

    # 2) Acting child gets Closed roads once (never duplicated on repeat prepare_self_change).
    child = ToolContext(
        repo_dir=REPO, drive_root=tmp_path / "data", task_id="t-child",
        task_constraint={"mode": "acting_subagent", "parent_task_id": "t-parent"},
    )
    first = _prepare_self_change(child)
    second = _prepare_self_change(child)
    assert "10.2 Closed roads" in first and "10.2 Closed roads" not in second


def test_contact_three_single_file_pulse_invariant_precision_and_small_packet(tmp_path):
    from ouroboros.code_shape_facts import body_ownership_pulse
    from ouroboros.review import diff_pulse
    from ouroboros.tools.governance_context import (
        BODY_INVARIANTS_CHAPTER,
        BODY_LAYER,
        PACKET_DELIVERY,
        RETRIEVING_DELIVERY,
        governance_context,
    )

    # 1) Single-file diff_pulse sees earlier-touched owning manual in the same diff as (touched).
    repo = tmp_path / "repo"
    (repo / "ouroboros").mkdir(parents=True)
    (repo / "docs/architecture").mkdir(parents=True)
    (repo / "ouroboros/domains.toml").write_text(
        '[domains]\nD01 = "Core"\n[references]\nD01 = ["docs/architecture/01-core.md"]\n'
        '[modules]\n"ouroboros/core_mod.py" = "D01"\n',
        encoding="utf-8", newline="\n",
    )
    (repo / "ouroboros/core_mod.py").write_text('"""Core."""\nX = 1\n', encoding="utf-8", newline="\n")
    (repo / "docs/ARCHITECTURE.md").write_text(
        "# Map\n\nIntro.\n\n## Chapters\n\n- [1. Core](architecture/01-core.md)\n",
        encoding="utf-8", newline="\n",
    )
    (repo / "docs/architecture/01-core.md").write_text("# 1. Core\n\nLead.\n\nBody.\n", encoding="utf-8", newline="\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    (repo / "docs/architecture/01-core.md").write_text("# 1. Core\n\nLead.\n\nUpdated.\n", encoding="utf-8", newline="\n")
    (repo / "ouroboros/core_mod.py").write_text('"""Core."""\nX = 2\n', encoding="utf-8", newline="\n")
    single_pulse = diff_pulse(repo, ["ouroboros/core_mod.py"])
    assert "docs/architecture/01-core.md (touched)" in single_pulse

    # 2) No English-word false positive for supervisor/state.py; qualified loop._finalize_loop_candidate matches §10#22.
    state_line = body_ownership_pulse(REPO, ["supervisor/state.py"], staged=False)
    assert "§10#4" not in state_line and "§10#6" not in state_line
    loop_line = body_ownership_pulse(REPO, ["ouroboros/loop.py"], staged=False)
    assert "§10#22" in loop_line

    # 3) Small 24k packet NEVER starves 10-key-invariants.md; retrieving nav names REQUIRED; no duplicate manifest rows.
    small = governance_context(
        REPO, surface="commit", touched_paths=["ouroboros/tools/review.py"],
        usable_window_tokens=24_000, delivery=PACKET_DELIVERY, layer=BODY_LAYER,
    )
    assert BODY_INVARIANTS_CHAPTER in small.inline_whole_documents
    paths_seen = [row["path"] for row in small.manifest]
    assert len(paths_seen) == len(set(paths_seen)), f"duplicate manifest rows: {paths_seen}"
    # Touched late-sorting module (ouroboros/tools/review.py is #84 of 104 in D06) appears in affected-domain map with omission count.
    normal = governance_context(
        REPO, surface="commit", touched_paths=["ouroboros/tools/review.py"],
        usable_window_tokens=128_000, delivery=PACKET_DELIVERY, layer=BODY_LAYER,
    )
    dom_sec = next(text for title, text in normal.inline_sections if "Affected domain module map" in title)
    assert "ouroboros/tools/review.py" in dom_sec and "more module(s) omitted" in dom_sec

    ret = governance_context(
        REPO, surface="commit", touched_paths=["ouroboros/tools/review.py"],
        usable_window_tokens=200_000, delivery=RETRIEVING_DELIVERY, layer=BODY_LAYER,
    )
    assert "REQUIRED" in ret.navigation and BODY_INVARIANTS_CHAPTER in ret.navigation

    # 4) Pre-heading pipe_table (Ch.4 HTTP route table) is selectable for router.py + task_model.py.
    gw = governance_context(
        REPO, surface="commit", touched_paths=["ouroboros/gateway/router.py", "ouroboros/gateway/task_model.py"],
        usable_window_tokens=200_000, delivery=PACKET_DELIVERY, layer=BODY_LAYER,
    )
    assert any("04-server-api-endpoints.md#opening block" in title and "| Method | Path |" in text
               for title, text in gw.inline_sections)


def test_module_map_confinement_multiline_js_headers_and_single_plan_capture(tmp_path, monkeypatch):
    import pytest
    from ouroboros.code_intelligence_architecture import DomainManifest, module_map_rows
    from ouroboros.tools import plan_ask, plan_packet

    # 1) DomainManifest.references defaults to dict, not tuple.
    m = DomainManifest(domains={}, modules={}, graph_allowed=(), lazy_only=())
    assert isinstance(m.references, dict)

    # 2) Path traversal rejected; web/ skips web/tests; multiline JS header extracted.
    with pytest.raises(ValueError, match="escapes repository root"):
        module_map_rows(REPO, "web/../..")
    with pytest.raises(ValueError, match="repository-relative"):
        module_map_rows(REPO, "/etc")
    web_rows = module_map_rows(REPO, "web/modules/")
    assert not any(row.startswith("web/tests/") for row in web_rows)
    card_row = next(row for row in web_rows if row.startswith("web/modules/card_model.js"))
    assert " — " in card_row

    # 3) Bug #1: plan_ask.stable_prompt_pair loads Architecture ONCE across packet + retrieving prompts.
    from ouroboros import reference_books as rb
    calls = []
    real_load = rb.load_reference_book
    monkeypatch.setattr(rb, "load_reference_book", lambda *a, **kw: (calls.append(a[1]), real_load(*a, **kw))[1])
    pkt, ret = plan_ask.stable_prompt_pair(None, body=True, system_root=REPO)
    assert calls == ["architecture"]
    assert "10.2 Closed roads" in pkt and "docs/architecture/10-key-invariants.md" in ret

    # Missing chapter in multi-chapter Architecture fails closed with PlanPacketError.
    broken = tmp_path / "broken_arch"
    (broken / "docs/architecture").mkdir(parents=True)
    (broken / "BIBLE.md").write_text("# Bible\n", encoding="utf-8")
    (broken / "docs/ARCHITECTURE.md").write_text(
        "# Map\n\nIntro.\n\n## Chapters\n\n- [1. Gone](architecture/01-gone.md)\n", encoding="utf-8",
    )
    with pytest.raises(plan_packet.PlanPacketError):
        plan_packet.load_architecture_book_for_plan(broken)


def test_bug_five_real_registry_schedule_subagent_and_optional_dev_isolation(tmp_path):
    from types import SimpleNamespace
    from ouroboros.context import _capture_context_core
    from ouroboros.context_fit import ContextCore, _integrating
    from ouroboros.memory import Memory
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    # 1) Bug #5: real full ToolRegistry schedule_subagent <= 4096 chars on OpenAI custom wire; readonly pops write args.
    reg = ToolRegistry(repo_dir=REPO, drive_root=tmp_path / "d")
    full_schemas = reg.schemas()
    projected = project_function_tools_to_openai_custom(full_schemas)
    sched = next(t for t in projected.wire_tools() if t["custom"]["name"] == "schedule_subagent")
    assert len(sched["custom"]["description"]) <= 4096
    reg.set_context(ToolContext(
        repo_dir=REPO, drive_root=tmp_path / "d", task_id="t-ro",
        task_constraint={"mode": "local_readonly_subagent", "parent_task_id": "p"},
    ))
    ro_schemas = reg.schemas()
    ro_sched = next(t for t in ro_schemas if t["function"]["name"] == "schedule_subagent")
    ro_props = ro_sched["function"]["parameters"]["properties"]
    for forbidden in ("workspace", "directory_strategy", "scope_paths"):
        assert forbidden not in ro_props

    # 2) Opus F-C: delegated_focus=True stays non-integrating even without a memory snapshot role.
    c = ContextCore("b", "bi", "a", "", "", "", '""', False, delegated_focus=True)
    assert _integrating(c, None) is False

    # 3) Astra F19: unreadable optional DEVELOPMENT.md does not pollute ordinary greeting core errors.
    fake_repo = tmp_path / "repo"
    (fake_repo / "docs/architecture").mkdir(parents=True)
    (fake_repo / "prompts").mkdir(parents=True)
    (fake_repo / "BIBLE.md").write_text("# Bible\n", encoding="utf-8")
    (fake_repo / "prompts/SYSTEM.md").write_text("System.\n", encoding="utf-8")
    (fake_repo / "docs/ARCHITECTURE.md").write_text(
        "# Map\n\nIntro.\n\n## Chapters\n\n- [1. One](architecture/01-one.md)\n\n## Reference Atlas\n\n- `docs/DESIGN.md` — UI\n",
        encoding="utf-8",
    )
    (fake_repo / "docs/architecture/01-one.md").write_text("# 1. One\n\nLead.\n\nBody.\n", encoding="utf-8")
    (fake_repo / "docs/DEVELOPMENT.md").write_text(
        "# Dev\n\nIntro.\n\n## Chapters\n\n- [1. Missing](development/01-missing.md)\n", encoding="utf-8",
    )
    drive = tmp_path / "drive"
    env = SimpleNamespace(
        repo_dir=fake_repo, drive_root=drive, budget_drive_root=drive, branch_dev="ouroboros",
        repo_path=lambda rel: fake_repo / rel, drive_path=lambda rel: drive / rel,
    )
    mem = Memory(drive_root=drive, repo_dir=fake_repo)
    core = _capture_context_core(env, mem, {"id": "t1", "type": "chat", "text": "hello"}, None, None)
    assert core.reference_book_errors == ()
    core_req = _capture_context_core(
        env, mem,
        {"id": "t2", "type": "task", "text": "dev", "task_contract": {"context_requires_development": True}},
        None, None,
    )
    assert any("DEVELOPMENT.md" in err for err in core_req.reference_book_errors)


def test_r2_query_code_web_module_map_and_contact_one_metadata_quiet(tmp_path):
    from ouroboros.code_intelligence import CodeInventory, FileFact
    from ouroboros.code_intelligence_architecture import architecture_fact_rows
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.registry import ToolContext

    # R2-1: real CodeInventory([FileFact(path=...)]) yields live JS/TS module_map rows.
    inv = CodeInventory(
        schema_version=1,
        repo_root=str(REPO),
        git_head="HEAD",
        created_at="2026-11-26T00:00:00Z",
        files=[FileFact(path="web/modules/card_model.js", sha256="a", size=100, language="javascript", token_estimate=25)],
        coverage={},
    )
    rows = architecture_fact_rows(REPO, "module_map web/modules/", inventory=lambda: inv)
    assert rows and "(1 JS/TS modules)" in rows[0]
    assert any(r.startswith("web/modules/card_model.js — ") for r in rows[1:])

    # Contact #1 stays quiet on governing/administrative root files (BIBLE.md, VERSION, README.md).
    ctx = ToolContext(repo_dir=REPO, drive_root=tmp_path / "data", task_id="t-meta")
    for meta_path in ("BIBLE.md", "VERSION", "README.md"):
        assert "[Body manual:" not in _read_file(ctx, meta_path, root="system_repo", max_lines=5)


def test_r2_retrieving_brief_distinct_roots_broken_sibling_and_candidate_new_module(tmp_path):
    from ouroboros.tools.governance_context import (
        BODY_INVARIANTS_CHAPTER,
        BODY_LAYER,
        PACKET_DELIVERY,
        RETRIEVING_DELIVERY,
        governance_context,
    )
    from ouroboros.tools.review_brief_coupling import BriefInputs, build_retrieving_brief

    # R2-5: broken sibling chapter in Architecture still delivers intact 10-key-invariants.md.
    broken = tmp_path / "broken_gov"
    (broken / "docs/architecture").mkdir(parents=True)
    (broken / "docs/ARCHITECTURE.md").write_text(
        "# Map\n\nIntro.\n\n## Chapters\n\n- [1. Missing](architecture/01-missing.md)\n"
        "- [10. Invariants](architecture/10-key-invariants.md)\n",
        encoding="utf-8",
    )
    (broken / "docs/architecture/10-key-invariants.md").write_text(
        "# 10. Key Invariants\n\nLead.\n\n### 10.2 Closed roads\n\n- Keep road closed.\n",
        encoding="utf-8",
    )
    pkt_broken = governance_context(
        broken, surface="commit", touched_paths=["ouroboros/review.py"],
        usable_window_tokens=128_000, delivery=PACKET_DELIVERY, layer=BODY_LAYER,
    )
    assert BODY_INVARIANTS_CHAPTER in pkt_broken.inline_whole_documents
    ret_broken = governance_context(
        broken, surface="commit", touched_paths=["ouroboros/review.py"],
        usable_window_tokens=128_000, delivery=RETRIEVING_DELIVERY, layer=BODY_LAYER,
    )
    assert any(r["path"] == BODY_INVARIANTS_CHAPTER and r["disposition"] == "required" for r in ret_broken.manifest)

    # R2-2: isolated candidate modifying 10-key-invariants.md gets system_repo row (native) and materialized
    # review-governance artifact (delegated) with exact gov_root SHA.
    cand = tmp_path / "cand"
    (cand / "docs/architecture").mkdir(parents=True)
    (cand / "ouroboros").mkdir(parents=True)
    (cand / "docs/architecture/10-key-invariants.md").write_text(
        "# 10. Key Invariants\n\nCANDIDATE_MODIFIED_COPY\n", encoding="utf-8",
    )
    (cand / "ouroboros/domains.toml").write_text(
        (REPO / "ouroboros/domains.toml").read_text(encoding="utf-8").replace(
            "[modules]\n", '[modules]\n"ouroboros/brand_new_r2.py" = "D06"\n', 1,
        ),
        encoding="utf-8",
    )
    (cand / "ouroboros/brand_new_r2.py").write_text('"""Brand new candidate module lead."""\nX = 1\n', encoding="utf-8")
    _git(cand, "init", "-q")
    _git(cand, "add", "-A")
    _git(cand, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")

    src_root = tmp_path / "sources"
    native_brief = BriefInputs(
        checklist_section="chk", governance_repo_dir=REPO,
        touched_paths=("docs/architecture/10-key-invariants.md",), required_sources=[],
        delegated=False, source_root=str(src_root), task_id="t-nat",
    )
    _, nat_manifest = build_retrieving_brief(cand, native_brief)
    nat_gov = [r for r in nat_manifest["native_required_sources"] if r.get("disposition") == "governance_required"]
    assert len(nat_gov) == 1 and nat_gov[0]["root"] == "system_repo" and nat_gov[0]["path"] == BODY_INVARIANTS_CHAPTER

    del_brief = BriefInputs(
        checklist_section="chk", governance_repo_dir=REPO,
        touched_paths=("docs/architecture/10-key-invariants.md",), required_sources=[],
        delegated=True, source_root=str(src_root), task_id="t-del",
    )
    _, del_manifest = build_retrieving_brief(cand, del_brief)
    del_gov = [r for r in del_manifest["native_required_sources"] if r.get("disposition") == "governance_required"]
    assert len(del_gov) == 1 and del_gov[0].get("governance_of") == BODY_INVARIANTS_CHAPTER
    assert del_gov[0].get("root") == "session_root"
    assert str(del_gov[0].get("path") or "").startswith(".review-drive/t-del/")
    assert del_manifest["native_data_root"] == str(src_root)

    # R2-3: candidate-only brand-new module still produces affected-domain module map in packet seats.
    pkt_new = governance_context(
        REPO, surface="commit", touched_paths=["ouroboros/brand_new_r2.py"],
        usable_window_tokens=200_000, delivery=PACKET_DELIVERY, layer=BODY_LAYER, module_map_root=cand,
    )
    dom_sec = next(text for title, text in pkt_new.inline_sections if "Affected domain module map" in title)
    assert "ouroboros/brand_new_r2.py — Brand new candidate module lead." in dom_sec


def test_r2_nano_continuation_preserves_compact_body_map_and_telling_vs_schemas(tmp_path):
    from ouroboros import context_fit, held_notes, memory_floor
    from tests import _memory_view_synthetic as syn
    from tests.test_section_floor import _core, _host_facts, _plan, booked

    # R2-4: stepped Nano compact Body Map survives continuation_capture -> restored_capture -> cold Nano fit.
    import dataclasses
    core = dataclasses.replace(
        booked(_core(tmp_path, snapshot=syn.snapshot()), tmp_path, body="Mechanism text. " * 60),
        development_required=True, development_md="## DEVELOPMENT.md\n\nHandbook body. " * 200,
    )
    roomy = _plan(core, 2_000_000, "nano")
    stepped = memory_floor.step_book_map(roomy, "nano", maps=True)
    assert stepped is not None
    assert [r["section_id"] for r in _host_facts(stepped, "nano")] == ["body_map"]
    cap = held_notes.continuation_capture(stepped, "nano")
    assert cap and cap[held_notes.CAPTURE_KEY]["book_maps"]
    restored = held_notes.restored_capture(cap)
    cold = _plan(dataclasses.replace(core, held_note_resume=restored), 2_000_000, "nano")
    assert [r["section_id"] for r in _host_facts(cold, "nano")] == ["body_map"]
    assert "## Body (parts & addresses)" in cold.messages_for("nano")[0]["content"][0]["text"]

    # R2-6: integrating Max start chooses Nano standing schemas to preserve 20/20 authored leads
    # rather than erasing tellings to keep reloadable tool schemas.
    from ouroboros.chronicle_lead import LEAD_RULE_VERSION
    pages = [
        dict(p, lead_rule=LEAD_RULE_VERSION, text=(f"Room {i} authored lead sentence " + "detail " * 25 + ". ") + syn.prose(2_500))
        for i, p in enumerate(syn.pages(30, 2_500))
    ]
    snap = syn.snapshot(story=pages)
    core_t = booked(_core(tmp_path / "t", snapshot=snap), tmp_path / "t", body="Short. ")
    standing = [
        {"type": "function", "function": {"name": name, "description": "Standing tool.", "parameters": {"type": "object", "properties": {}}}}
        for name in ("read_file", "enable_tools", "memory_read", "finish_task")
    ]
    extra = [
        {"type": "function", "function": {"name": f"extra_tool_{i}", "description": "Reloadable syntax. " * 220, "parameters": {"type": "object", "properties": {}}}}
        for i in range(40)
    ]
    all_tools = standing + extra
    empty_snap = syn.snapshot(story=[])
    core_empty = booked(_core(tmp_path / "te", snapshot=empty_snap), tmp_path / "te", body="Short. ")
    max_fixed = context_fit.estimate_context_prompt_tokens(
        _plan(core_empty, 2_000_000, "max").messages_for("max"), all_tools,
    )
    nano_fixed = context_fit.estimate_context_prompt_tokens(
        _plan(core_empty, 2_000_000, "nano").messages_for("nano"), standing,
    )
    assert max_fixed > nano_fixed + 10_000
    telling_toks = memory_floor.tokens_of(memory_floor.mv.story_chars(snap, memory_floor.telling_level(snap)))
    assert telling_toks > 1_000
    # Window fits Max with emergency memory (no tellings), overflows Max with tellings, and fits Nano with tellings.
    tight_win = max_fixed + 8_192 + max(400, telling_toks // 3)
    tight_plan = context_fit.build_context_fit_plan(
        None, core_t, {"id": "r2-6", "chat_id": 1}, preferred_mode="max",
        route_resolver=lambda *_a, **_kw: (
            {"provider": "test", "model": "fixture"},
            type("Ev", (), {"route_fp": "r2-6", "status": "asserted", "stale": False, "window_tokens": tight_win})(),
        ),
        tool_schemas=all_tools,
    )
    assert tight_plan.initial_mode == "nano"
    tight_level = memory_floor.kept_level(snap, tight_plan.projection("nano").memory_facts)
    assert memory_floor.level_is_telling(snap, tight_level)


def test_r2_corrupt_domains_manifest_preflight_and_staged_module_line_count(tmp_path):
    from ouroboros.review import diff_pulse
    from ouroboros.tools import review as review_tools

    # R2-9: corrupt staged domains.toml fails preflight instead of silently falling back.
    repo = tmp_path / "bad_dom"
    (repo / "ouroboros").mkdir(parents=True)
    (repo / "docs/architecture").mkdir(parents=True)
    (repo / "ouroboros/domains.toml").write_text('[domains]\nD01 = "Core"\n[modules]\n', encoding="utf-8")
    (repo / "docs/ARCHITECTURE.md").write_text("# Map\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    (repo / "ouroboros/domains.toml").write_text("[modules\nBROKEN_TOML =", encoding="utf-8")
    (repo / "ouroboros/new_mod.py").write_text('"""Lead."""\nX = 1\n', encoding="utf-8")
    (repo / "docs/ARCHITECTURE.md").write_text("# Map\n\nUpdated.\n", encoding="utf-8")
    _git(repo, "add", "-A")
    res = review_tools._preflight_check("add mod", "A  ouroboros/new_mod.py\nM  docs/ARCHITECTURE.md", str(repo))
    assert res is not None and "could not be parsed" in res

    # Staged diff_pulse reports staged module line count, not dirty worktree line count.
    repo2 = tmp_path / "staged_lines"
    (repo2 / "ouroboros").mkdir(parents=True)
    (repo2 / "ouroboros/tiny.py").write_text('"""Tiny."""\nX = 1\n', encoding="utf-8", newline="\n")
    _git(repo2, "init", "-q")
    _git(repo2, "add", "-A")
    _git(repo2, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    (repo2 / "ouroboros/tiny.py").write_text('"""Tiny."""\nX = 1\nY = 2\n', encoding="utf-8", newline="\n")
    _git(repo2, "add", "ouroboros/tiny.py")
    (repo2 / "ouroboros/tiny.py").write_text('"""Tiny."""\n' + "Z = 1\n" * 15, encoding="utf-8", newline="\n")
    pulse = diff_pulse(repo2, ["ouroboros/tiny.py"], staged=True)
    assert "ouroboros/tiny.py 3 (+1/-0)" in pulse
