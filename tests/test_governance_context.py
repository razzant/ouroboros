"""The governance-tier SSOT: which documents a reviewer receives in full.

Every case runs against a small fake repository that mirrors the real chapter
layout (a book entrypoint with a `## Chapters` membership list plus
`docs/<book>/NN-*.md` members), so the tiers are pinned by behaviour rather than
by the current size of the production books.
"""

from __future__ import annotations

import pathlib

import pytest

from ouroboros.runtime_limits import REVIEW_GOVERNANCE_INLINE_SHARE
from ouroboros.tools.governance_context import (
    REVIEW_PROTOCOL_CHAPTER,
    SHARED_CHECKLIST_SECTION,
    GovernanceContext,
    governance_context,
)
from ouroboros.tools.review_helpers import load_checklist_section
from ouroboros.utils import estimate_tokens

CHECKLIST_SECTION = "## Change Review Checklist\n\n1. item\n"
SHARED_PATH = f"docs/CHECKLISTS.md#{SHARED_CHECKLIST_SECTION}"
# The shared section comes from the review code's own checklist, never from the
# reviewed tree, whose copy (written by the fixture) says something else.
SHARED_SECTION = load_checklist_section(SHARED_CHECKLIST_SECTION)
TREE_SHARED_RULE = "The reviewed tree's own copy of the shared rule."
SHARED_RENDERED = f"## {SHARED_PATH}\n\n{SHARED_SECTION}"

# Chapter bodies are addressed by what they MENTION: the selector looks for an
# exact touched file name, so `review_project_dialogue.py` must not satisfy a
# change to `project_dialogue.py`.
DEV_CHAPTERS = {
    "01-naming.md": ("Naming", "Names live here. See ouroboros/project_dialogue.py twice: "
                               "ouroboros/project_dialogue.py."),
    "02-substrings.md": ("Substrings", "This chapter only names review_project_dialogue.py "
                                       "and chat.jsx, neither of which is touched."),
    "05-review-and-commit-protocol.md": ("Protocol", "How a commit is reviewed."),
    "06-widgets.md": ("Widgets", "Widget rules mention chat.js once."),
}
ARCH_CHAPTERS = {
    "01-core.md": [("Core loop", "The loop mentions ouroboros/project_dialogue.py."),
                   ("Unrelated", "Nothing touched here.")],
    "02-web.md": [("Web surfaces", "The chat page is web/modules/chat.js and chat.js again.")],
}


@pytest.fixture()
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "docs" / "development").mkdir(parents=True)
    (tmp_path / "docs" / "architecture").mkdir(parents=True)
    (tmp_path / "BIBLE.md").write_text("# Constitution\n\nP1 Continuity.\n", encoding="utf-8", newline="\n")
    (tmp_path / "docs" / "CHECKLISTS_ARCHIVE.md").write_text(
        "# Archive\n\nA standing disclosure.\n", encoding="utf-8", newline="\n")
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        f"# Checklists\n\n{CHECKLIST_SECTION}\n## {SHARED_CHECKLIST_SECTION}\n\n{TREE_SHARED_RULE}\n",
        encoding="utf-8", newline="\n")
    (tmp_path / "docs" / "DESIGN.md").write_text(
        "# Design\n\nThe design system for web/ work.\n", encoding="utf-8", newline="\n")

    for name, (title, body) in DEV_CHAPTERS.items():
        (tmp_path / "docs" / "development" / name).write_text(
            f"# {title}\n\nAn authored introduction.\n\n## {title} rules\n\n{body}\n",
            encoding="utf-8", newline="\n")
    (tmp_path / "docs" / "DEVELOPMENT.md").write_text(
        "# Development\n\nThe handbook entrypoint.\n\n## Chapters\n\n"
        + "\n".join(f"- [{name}](development/{name})" for name in DEV_CHAPTERS)
        + "\n", encoding="utf-8", newline="\n")

    for name, sections in ARCH_CHAPTERS.items():
        body = "\n\n".join(f"## {heading}\n\n{text}" for heading, text in sections)
        (tmp_path / "docs" / "architecture" / name).write_text(
            f"# {name}\n\nAn authored introduction.\n\n{body}\n", encoding="utf-8", newline="\n")
    (tmp_path / "docs" / "ARCHITECTURE.md").write_text(
        "# Architecture\n\nThe map entrypoint.\n\n## Chapters\n\n"
        + "\n".join(f"- [{name}](architecture/{name})" for name in ARCH_CHAPTERS)
        + "\n", encoding="utf-8", newline="\n")
    return tmp_path


def _context(repo: pathlib.Path, **kwargs) -> GovernanceContext:
    return governance_context(
        repo,
        surface=kwargs.pop("surface", "triad"),
        touched_paths=kwargs.pop("touched_paths", ["ouroboros/project_dialogue.py"]),
        usable_window_tokens=kwargs.pop("usable_window_tokens", 1_000_000),
        delivery=kwargs.pop("delivery", "packet"),
        checklist_section_text=kwargs.pop("checklist_section_text", CHECKLIST_SECTION),
        **kwargs,
    )


def _rows(context: GovernanceContext, disposition: str = "", tier: int = 0) -> list[dict]:
    return [row for row in context.manifest
            if (not disposition or row["disposition"] == disposition)
            and (not tier or row["tier"] == tier)]


def _paths(context: GovernanceContext, disposition: str = "", tier: int = 0) -> list[str]:
    return [row["path"] for row in _rows(context, disposition, tier)]


# --- tier 1 -------------------------------------------------------------------

def test_tier_one_rules_are_always_inline(repo):
    context = _context(repo)

    assert _paths(context, "inline", tier=1) == [
        "docs/CHECKLISTS.md", SHARED_PATH, "BIBLE.md", "docs/CHECKLISTS_ARCHIVE.md"]
    assert "P1 Continuity." in context.stable_inline
    assert "A standing disclosure." in context.stable_inline
    # The caller's checklist section is declared, never re-rendered: the surface
    # owns where its own section sits in the prompt. The shared section is the
    # one this module loads itself, and it leads the stable text.
    assert CHECKLIST_SECTION not in context.stable_inline
    assert context.stable_inline.startswith(SHARED_RENDERED)
    assert TREE_SHARED_RULE not in context.stable_inline
    assert [row["chars"] for row in _rows(context, tier=1)][:2] == [
        len(CHECKLIST_SECTION), len(SHARED_SECTION)]


def test_the_body_layer_has_no_switch_that_drops_the_shared_section():
    """The only surface that ever opted out of `Shared Contract Ownership` (the
    skill advisory prompt) is gone with PR-3; a body-layer reviewer of this
    repository's code always carries the section, and nothing can ask otherwise."""
    import inspect

    from ouroboros.tools import governance_context as module

    assert "repository_rules" not in inspect.signature(module.governance_context).parameters
    assert "repository_rules" not in inspect.getsource(module)


def test_a_non_body_subject_runs_the_core_layer_without_the_shared_section(repo):
    """Another repository is not Ouroboros's body: the core layer names the
    shared section `not_applicable` and inlines no text of it; the supplied
    checklist section is the only tier-1 inline row."""
    context = _context(repo, layer="core")

    assert SHARED_PATH in _paths(context, "not_applicable")
    assert SHARED_CHECKLIST_SECTION not in context.stable_inline + context.navigation
    assert _paths(context, "inline", tier=1) == ["docs/CHECKLISTS.md"]
    assert context.layer == "core" and _context(repo).layer == "body"


def test_an_unloadable_shared_section_is_named_not_claimed(repo, monkeypatch):
    def _missing(name, checklist_path=None):
        raise ValueError(f"Section '## {name}' not found")

    monkeypatch.setattr("ouroboros.tools.review_helpers.load_checklist_section", _missing)
    context = _context(repo)

    assert f"[⚠️ OMISSION: {SHARED_PATH} could not be loaded" in context.stable_inline
    assert "the section that applies to this review is inlined above." in context.navigation
    assert f"`{SHARED_CHECKLIST_SECTION}` section" not in context.navigation


def test_tier_one_is_the_same_text_whatever_the_change_touches(repo):
    """The cache-marked prefix must not move when the change class moves."""
    web = _context(repo, touched_paths=["web/modules/chat.js"])
    core = _context(repo, touched_paths=["ouroboros/project_dialogue.py"])

    assert web.stable_inline == core.stable_inline
    assert web.selected_inline != core.selected_inline


def test_a_document_the_surface_already_inlines_is_declared_not_duplicated(repo):
    context = _context(repo, already_inline=("BIBLE.md", "docs/CHECKLISTS_ARCHIVE.md"))

    assert "P1 Continuity." not in context.stable_inline
    assert context.stable_inline == SHARED_RENDERED
    carried = {row["path"]: row for row in _rows(context, "inline", tier=1)}
    assert carried["BIBLE.md"]["chars"] == len("# Constitution\n\nP1 Continuity.\n")
    assert "carried by this surface" in carried["BIBLE.md"]["reason"]


def test_a_carrying_surface_may_state_its_own_delivery_mechanism(repo):
    """The mapping identifies an actual inline location, never a future read."""
    mechanism = "inlined whole in this row's constitutional system message"
    context = _context(repo, already_inline={"BIBLE.md": mechanism})

    bible = next(row for row in context.manifest if row["path"] == "BIBLE.md")
    assert bible["disposition"] == "inline" and bible["reason"] == mechanism
    assert bible["chars"] == len("# Constitution\n\nP1 Continuity.\n")
    assert "P1 Continuity." not in context.stable_inline
    # The archive was not declared, so this surface still receives it inline.
    assert "A standing disclosure." in context.stable_inline
    # A blank reason falls back to the default wording rather than to no reason.
    blank = _context(repo, already_inline={"BIBLE.md": ""})
    assert next(row for row in blank.manifest
                if row["path"] == "BIBLE.md")["reason"] == "carried by this surface's own delivery"


def test_a_surface_with_no_checklist_section_says_so_instead_of_claiming_one(repo):
    """A deep self-review supplies no section. Recording `inline` with zero
    characters, and telling the reviewer a section is inlined above, would both
    be false (BIBLE P1); the shared section it does receive is named as such."""
    context = _context(repo, checklist_section_text="")

    row = next(row for row in context.manifest if row["path"] == "docs/CHECKLISTS.md")
    assert row["disposition"] == "navigation" and row["chars"] == 0
    assert row["reason"] == "this surface supplies no checklist section"
    assert f"its `{SHARED_CHECKLIST_SECTION}` section is inlined above." in context.navigation
    assert "the section that applies to this review" not in context.navigation
    assert context.stable_inline.startswith(SHARED_RENDERED)
    bare = _context(repo, checklist_section_text="", layer="core")
    assert "NO section of `docs/CHECKLISTS.md` is inlined for this review" in bare.navigation
    assert "is inlined above" not in bare.navigation
    # A surface that does supply one keeps the inline row and the pointer.
    supplied = _context(repo)
    assert next(r for r in supplied.manifest
                if r["path"] == "docs/CHECKLISTS.md")["disposition"] == "inline"
    assert (f"the section that applies to this review and its `{SHARED_CHECKLIST_SECTION}` "
            "section are inlined above.") in supplied.navigation


def test_an_unreadable_tier_one_document_is_named_not_silently_skipped(repo):
    (repo / "BIBLE.md").unlink()
    context = _context(repo)

    assert "[⚠️ OMISSION:" in context.stable_inline
    assert "BIBLE.md" in _paths(context, "inline", tier=1)


# --- tier 2 -------------------------------------------------------------------

def test_design_is_inline_only_when_a_touched_path_is_under_web(repo):
    web = _context(repo, touched_paths=["web/modules/chat.js"])
    core = _context(repo, touched_paths=["ouroboros/project_dialogue.py"])

    assert "docs/DESIGN.md" in _paths(web, "inline", tier=2)
    assert "The design system for web/ work." in web.selected_inline
    assert "docs/DESIGN.md" in _paths(core, "navigation", tier=2)
    assert "The design system for web/ work." not in core.selected_inline


def test_development_chapters_are_selected_by_exact_file_name(repo):
    context = _context(repo, touched_paths=["ouroboros/project_dialogue.py"])

    inline = _paths(context, "inline", tier=2)
    assert "docs/development/01-naming.md" in inline
    # A longer name that merely CONTAINS the touched basename is not a mention.
    assert "docs/development/02-substrings.md" in _paths(context, "navigation", tier=2)
    assert "docs/development/06-widgets.md" in _paths(context, "navigation", tier=2)
    naming = next(row for row in context.manifest
                  if row["path"] == "docs/development/01-naming.md")
    # One touched file, mentioned twice — a full path and its basename in the
    # same sentence are one mention, not two.
    assert naming["reason"] == "mentions 1 touched file(s), 2 time(s)"


def test_a_directory_prefix_selects_no_chapter(repo):
    """`web/` or `ouroboros/` would match nearly every chapter, so only exact
    file names and repo-relative paths are searched. The DESIGN change class is
    a separate, deliberate prefix rule and keeps working."""
    context = _context(repo, touched_paths=["ouroboros/"])

    assert _paths(context, "inline", tier=2) == [REVIEW_PROTOCOL_CHAPTER]
    assert _rows(context, "inline", tier=3) == []


def test_the_review_protocol_chapter_is_inline_even_with_nothing_touched(repo):
    context = _context(repo, touched_paths=[])

    assert _paths(context, "inline", tier=2) == [REVIEW_PROTOCOL_CHAPTER]
    assert "How a commit is reviewed." in context.selected_inline


def test_an_unassemblable_book_is_disclosed_instead_of_delivered(repo):
    (repo / "docs" / "development" / "01-naming.md").unlink()
    context = _context(repo)

    book = next(row for row in context.manifest if row["path"] == "docs/DEVELOPMENT.md")
    assert book["disposition"] == "navigation"
    assert "01-naming.md" in book["reason"]
    assert _paths(context, "inline", tier=2) == []


# --- the share budget ---------------------------------------------------------

def test_change_class_rules_are_admitted_protocol_then_design_then_relevance(repo):
    context = _context(repo, touched_paths=["ouroboros/project_dialogue.py", "web/modules/chat.js"])

    # 01-naming mentions its touched file twice, 06-widgets mentions one once.
    assert _paths(context, "inline", tier=2) == [
        REVIEW_PROTOCOL_CHAPTER, "docs/DESIGN.md",
        "docs/development/01-naming.md", "docs/development/06-widgets.md"]


def test_the_share_budget_bounds_the_selection_and_names_the_rest(repo):
    window = 200
    budget = int(window * REVIEW_GOVERNANCE_INLINE_SHARE)
    context = _context(repo, touched_paths=["ouroboros/project_dialogue.py", "web/modules/chat.js"],
                       usable_window_tokens=window)

    # The admission cost is the rendered section, and a token estimate depends
    # only on length, so the manifest's own chars reproduce it exactly.
    spent = sum(estimate_tokens(f"## {row['path']}\n\n" + "x" * row["chars"])
                for row in _rows(context, "inline") if row["tier"] > 1)
    assert spent <= budget, f"tier 2/3 inline {spent} tokens exceeds the {budget}-token share"
    assert _paths(context, "inline", tier=2) == [REVIEW_PROTOCOL_CHAPTER]
    exhausted = [row for row in context.manifest if "inline share exhausted" in row["reason"]]
    assert exhausted, "an over-budget candidate must be named as a pointer"
    assert all(row["disposition"] == "navigation" for row in exhausted)


def test_a_zero_window_inlines_no_change_class_rules_and_names_all_of_them(repo):
    context = _context(repo, usable_window_tokens=0)

    assert _rows(context, "inline", tier=2) == []
    assert _rows(context, "inline", tier=3) == []
    assert REVIEW_PROTOCOL_CHAPTER in _paths(context, "navigation", tier=2)
    assert context.stable_inline  # tier 1 is never budgeted away


# --- tier 3 -------------------------------------------------------------------

def test_the_architecture_map_is_never_inlined_whole(repo):
    context = _context(repo, touched_paths=["web/modules/chat.js"])

    whole_arch = [path for path in _paths(context, "inline")
                  if path.startswith("docs/architecture/") and "#" not in path]
    assert whole_arch == []
    assert "docs/ARCHITECTURE.md" in _paths(context, "navigation", tier=3)
    assert "docs/architecture/02-web.md" in context.navigation


def test_a_packet_row_receives_the_sections_that_name_a_touched_file(repo):
    context = _context(repo, touched_paths=["web/modules/chat.js"], delivery="packet")

    assert _paths(context, "inline", tier=3) == ["docs/architecture/02-web.md#Web surfaces"]
    assert "The chat page is web/modules/chat.js" in context.selected_inline
    assert "01-core.md#Unrelated" not in context.selected_inline


def test_a_retrieving_row_receives_the_navigation_and_no_sections(repo):
    context = _context(repo, touched_paths=["web/modules/chat.js"], delivery="retrieving")

    assert _rows(context, "inline", tier=3) == []
    assert "docs/architecture/02-web.md" in context.navigation
    arch = next(row for row in context.manifest if row["path"] == "docs/ARCHITECTURE.md")
    assert "reads sections on demand" in arch["reason"]


# --- the whole result ---------------------------------------------------------

def test_the_navigation_tells_the_reviewer_how_to_read_what_it_did_not_receive(repo):
    context = _context(repo, delivery="retrieving")

    assert 'read_file(root="system_repo"' in context.navigation
    assert "docs/CHECKLISTS.md" in context.navigation
    assert "docs/DESIGN.md" in context.navigation  # a pointer when not inlined
    packet = _context(repo, delivery="packet")
    assert "read_file" not in packet.navigation
    assert "This row has no repository tools" in packet.navigation
    assert "NOT delivered" in packet.navigation


def test_inline_whole_documents_carry_only_whole_documents(repo):
    # Reference-book sources retain exact line endings, including on hosts
    # where a plain read_text would silently translate CRLF back to LF.
    chapter = repo / REVIEW_PROTOCOL_CHAPTER
    chapter.write_bytes(chapter.read_bytes().replace(b"\n", b"\r\n"))
    context = _context(repo, touched_paths=["web/modules/chat.js"])

    assert all("#" not in path for path in context.inline_whole_documents)
    assert context.inline_whole_documents[REVIEW_PROTOCOL_CHAPTER] == chapter.read_bytes().decode("utf-8")
    assert context.inline_whole_documents["docs/DESIGN.md"] == (repo / "docs" / "DESIGN.md").read_bytes().decode("utf-8")
    # A duplicate-suppressing caller compares bytes, so the map must be exact.
    for path, text in context.inline_whole_documents.items():
        assert (repo / path).read_bytes().decode("utf-8") == text


def test_the_result_is_deterministic_for_one_tree_and_one_change(repo):
    first = _context(repo, touched_paths=["web/modules/chat.js", "ouroboros/project_dialogue.py"])
    second = _context(repo, touched_paths=["web/modules/chat.js", "ouroboros/project_dialogue.py"])

    assert first == second
    assert first.tokens_estimate == estimate_tokens(
        "\n\n".join(part for part in (first.stable_inline, first.selected_inline, first.navigation)
                    if part.strip()))


def test_every_document_is_dispositioned_exactly_once(repo):
    context = _context(repo, touched_paths=["web/modules/chat.js"])

    paths = _paths(context)
    assert len(paths) == len(set(paths))
    for expected in ("BIBLE.md", "docs/CHECKLISTS.md", SHARED_PATH, "docs/CHECKLISTS_ARCHIVE.md",
                     "docs/DESIGN.md", "docs/DEVELOPMENT.md", "docs/ARCHITECTURE.md",
                     *(f"docs/development/{name}" for name in DEV_CHAPTERS),
                     *(f"docs/architecture/{name}" for name in ARCH_CHAPTERS)):
        assert expected in paths
    assert all(set(row) == {"path", "tier", "disposition", "chars", "reason"}
               for row in context.manifest)


# --- the triad wiring ---------------------------------------------------------

def test_the_triad_packet_declares_bible_and_the_archive_as_already_delivered(repo, monkeypatch):
    """`review_multi_model.triad_api_messages` prepends BIBLE.md to every api row
    and `_load_checklist_section` appends the standing disclosures, so the triad
    must not send either a second time."""
    from ouroboros.review_records import ReviewSlot
    from ouroboros.tools import review

    monkeypatch.setattr(review, "reviewer_context_window", lambda *_a, **_k: 200_000)
    ctx = type("_Ctx", (), {"repo_dir": str(repo)})()
    context = review._triad_governance_context(
        ctx, ["web/modules/chat.js"], CHECKLIST_SECTION,
        ["openai/packet"], [ReviewSlot(slot_id="triad_slot_1", model="openai/packet")])

    assert context.stable_inline == SHARED_RENDERED
    assert all("carried by this surface" in row["reason"]
               for row in context.manifest if row["path"] in ("BIBLE.md", "docs/CHECKLISTS_ARCHIVE.md"))
    assert "docs/DESIGN.md" in [row["path"] for row in context.manifest
                                if row["disposition"] == "inline"]


def test_a_panel_with_no_api_row_asks_for_no_packet_governance(repo):
    from ouroboros.tools import review

    ctx = type("_Ctx", (), {"repo_dir": str(repo)})()
    context = review._triad_governance_context(ctx, ["a.py"], CHECKLIST_SECTION, [], [])

    assert context == GovernanceContext()
    assert context.manifest == [] and context.navigation == ""


def test_retrieving_triad_receives_shared_tiers_in_its_actual_task(repo, monkeypatch):
    from ouroboros.review_records import ReviewSlot
    from ouroboros.tools import review
    from ouroboros.tools.review_subject import build_triad_session_task

    monkeypatch.setattr(review, "reviewer_context_window", lambda *_a, **_k: 200_000)
    ctx = type("_Ctx", (), {"repo_dir": str(repo)})()
    archive = (repo / "docs/CHECKLISTS_ARCHIVE.md").read_text(encoding="utf-8")
    checklist = CHECKLIST_SECTION + archive
    context = review._triad_governance_context(
        ctx, ["web/modules/chat.js"], checklist, ["openai/native"],
        [ReviewSlot(slot_id="triad_1", model="openai/native")], delivery="retrieving")
    # The builder the two-part brief calls (review_brief_coupling.build_retrieving_brief).
    task = build_triad_session_task(
        governance_repo_dir=repo, goal_section="goal", scope_section="scope", checklist_section=checklist,
        rebuttal_section="", review_history_section="", governance=context)

    assert task.count("P1 Continuity.") == 1
    assert task.count("A standing disclosure.") == 1
    assert task.count(SHARED_SECTION) == 1
    assert "The design system for web/ work." in task
    assert "How a commit is reviewed." in task
    assert "docs/architecture/02-web.md" in task
    assert "The chat page is web/modules/chat.js" not in task
    assert 'read_file(root="system_repo"' in task
