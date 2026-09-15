"""A relocated chapter must be read exactly as the monolith it came out of.

The split turned one physical governance file into an entrypoint plus chapters.
Every reader that used to get ~1 MB of prose from one `read_text()` now gets a
membership list unless it asks for the BOOK, and every predicate keyed on the
path used to answer for the whole corpus with one exact string. This module
pins the four places that difference is load-bearing: the full-book loader, the
canonical-corpus predicate, the pack's duplicate suppression, and the
untruncated-read guarantee -- plus the documentation gate, which must accept the
chapter that actually documents a new module.
"""

import pathlib

import pytest

from ouroboros.reference_books import BOOK_ENTRYPOINTS, book_entrypoint_for, book_path_role

REPO = pathlib.Path(__file__).resolve().parents[1]


def _chaptered_corpus(root: pathlib.Path, *, body: str = "The exact chapter body.\n") -> dict:
    files = {}
    for book_id, entrypoint in BOOK_ENTRYPOINTS.items():
        files[entrypoint] = (
            f"# {book_id.title()}\n\nThe authored orientation.\n\n## Chapters\n\n"
            f"- [Only chapter]({book_id}/only.md)\n"
        )
        files[f"docs/{book_id}/only.md"] = f"# Only chapter\n\nWhy it exists.\n\n## Detail\n\n{body}"
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return files


# --- path role -------------------------------------------------------------

@pytest.mark.parametrize("path,role", [
    ("docs/ARCHITECTURE.md", "entrypoint"),
    ("docs/DEVELOPMENT.md", "entrypoint"),
    ("docs/architecture/06-agent-core.md", "chapter"),
    ("docs/development/14-build-and-ci.md", "chapter"),
    ("docs/reference-books-migration.md", ""),
    ("docs/CHECKLISTS.md", ""),
    ("docs/architecture/notes.txt", ""),
    ("", ""),
])
def test_book_path_role_answers_by_shape_without_reading_the_tree(path, role):
    assert book_path_role(path) == role


def test_book_entrypoint_for_maps_a_chapter_back_to_its_own_book():
    assert book_entrypoint_for("docs/development/09-process-custody-rule.md") == "docs/DEVELOPMENT.md"
    assert book_entrypoint_for("docs/ARCHITECTURE.md") == "docs/ARCHITECTURE.md"
    assert book_entrypoint_for("BIBLE.md") == ""


# --- the full-book loader --------------------------------------------------

def test_load_governance_doc_delivers_the_composed_book_not_the_membership_page(tmp_path):
    from ouroboros.tools.review_helpers import load_governance_doc

    _chaptered_corpus(tmp_path, body="THE-ONLY-CHAPTER-BODY\n")
    for entrypoint in BOOK_ENTRYPOINTS.values():
        text = load_governance_doc(tmp_path, entrypoint)
        assert "THE-ONLY-CHAPTER-BODY" in text, entrypoint
        assert "## Chapters" in text, entrypoint
    # A non-book governance document is untouched by the book branch.
    (tmp_path / "BIBLE.md").write_text("# Constitution\n", encoding="utf-8")
    assert load_governance_doc(tmp_path, "BIBLE.md") == "# Constitution\n"


def test_an_unassemblable_book_is_a_named_omission_not_a_short_entrypoint(tmp_path):
    from ouroboros.tools.review_helpers import load_governance_doc

    _chaptered_corpus(tmp_path)
    (tmp_path / "docs/architecture/only.md").unlink()
    explicit = load_governance_doc(tmp_path, "docs/ARCHITECTURE.md")
    assert "OMISSION" in explicit and "only.md" in explicit, explicit
    assert "## Chapters" not in explicit, "a failure must not look like a delivered book"
    assert load_governance_doc(tmp_path, "docs/ARCHITECTURE.md", on_missing="silent") == ""


def test_the_live_books_reach_the_packed_review_surfaces_whole():
    from ouroboros.reference_books import compose_book, load_reference_book
    from ouroboros.tools.review_helpers import load_governance_doc

    for book_id, entrypoint in BOOK_ENTRYPOINTS.items():
        book = load_reference_book(REPO, book_id)
        delivered = load_governance_doc(REPO, entrypoint)
        assert delivered == compose_book(book)
        assert len(delivered) > 10 * len(book.entrypoint.text)


# --- the canonical corpus --------------------------------------------------

def test_a_chapter_is_canonical_exactly_as_its_entrypoint_was(tmp_path):
    from ouroboros.tools.review_context_atlas import atlas_required_beyond_diff
    from ouroboros.tools.review_helpers import (
        canonical_governance_sources,
        is_canonical_governance_path,
    )

    assert is_canonical_governance_path("docs/architecture/06-agent-core.md")
    assert is_canonical_governance_path("docs/DEVELOPMENT.md")
    assert not is_canonical_governance_path("docs/reference-books-migration.md")
    # Owed in full regardless of the change: the staged diff is never a
    # substitute for a canonical artifact, chapter or entrypoint.
    assert atlas_required_beyond_diff("docs/development/06-rules-by-change-class.md")

    _chaptered_corpus(tmp_path)
    (tmp_path / "BIBLE.md").write_text("# Constitution\n", encoding="utf-8")
    sources = canonical_governance_sources(tmp_path)
    assert "docs/architecture/only.md" in sources and "docs/development/only.md" in sources
    assert "BIBLE.md" in sources
    assert "docs/CHECKLISTS.md" not in sources, "an absent document is not claimed"
    assert len(sources) == len(set(sources))


def test_an_unreadable_book_contributes_its_entrypoint_alone(tmp_path):
    from ouroboros.tools.review_helpers import canonical_governance_sources

    _chaptered_corpus(tmp_path)
    (tmp_path / "docs/development/only.md").unlink()
    sources = canonical_governance_sources(tmp_path)
    assert "docs/DEVELOPMENT.md" in sources
    assert not any(s.startswith("docs/development/") for s in sources), (
        "a book that cannot be read must not widen the claimed coverage"
    )


# --- pack duplicate suppression -------------------------------------------

def test_a_touched_chapter_is_withheld_when_the_prefix_carries_its_book(tmp_path):
    from ouroboros.tools.review_file_pack import triad_pack_exclusions
    from ouroboros.tools.review_helpers import load_governance_doc

    _chaptered_corpus(tmp_path)
    composed = load_governance_doc(tmp_path, "docs/ARCHITECTURE.md")
    touched = ["docs/architecture/only.md", "docs/development/only.md"]
    excluded, note = triad_pack_exclusions(
        tmp_path, touched, prefix_texts={"docs/ARCHITECTURE.md": composed},
    )
    assert "docs/architecture/only.md" in excluded
    assert "docs/development/only.md" not in excluded, (
        "only the book whose composed copy IS in the prefix may be withheld"
    )
    assert "docs/architecture/only.md" in note

    # A chapter edited after the prefix was rendered keeps its full text.
    (tmp_path / "docs/architecture/only.md").write_text(
        "# Only chapter\n\nWhy it exists.\n\n## Detail\n\nEdited after the prefix.\n",
        encoding="utf-8",
    )
    excluded_after, _ = triad_pack_exclusions(
        tmp_path, touched, prefix_texts={"docs/ARCHITECTURE.md": composed},
    )
    assert "docs/architecture/only.md" not in excluded_after


# --- the documentation gate ------------------------------------------------

def test_the_new_module_gate_takes_the_chapter_that_documents_the_module():
    from ouroboros.tools import review

    staged = "A  ouroboros/brand_new_owner.py\n"
    blocked = review._preflight_check("add an owner", staged, REPO)
    assert blocked and "Architecture book" in blocked

    for documented in ("docs/ARCHITECTURE.md", "docs/architecture/01-high-level-architecture.md"):
        assert review._preflight_check(
            "add an owner", staged + f"M  {documented}\n", REPO,
        ) is None, documented

    assert review._preflight_check(
        "add an owner", staged + "M  docs/reference-books-migration.md\n", REPO,
    ) is not None, "a docs file outside the book documents no module"


# --- the untruncated-read guarantee ---------------------------------------

def test_a_chapter_read_keeps_the_untruncated_guarantee_the_monolith_had():
    from ouroboros.loop_tool_execution import _path_is_cognitive_artifact, _truncate_tool_result
    from ouroboros.tool_capabilities import TOOL_RESULT_LIMITS

    limit = TOOL_RESULT_LIMITS["read_file"]
    oversized = "x" * (limit + 5_000)
    for rel in ("docs/architecture/06-agent-core.md", "docs/development/06-rules-by-change-class.md"):
        assert _path_is_cognitive_artifact("read_file", {"path": rel}), rel
        assert _truncate_tool_result(oversized, "read_file", {"path": rel}) == oversized, rel
    # The guarantee is a book-source class, not every markdown file under docs/.
    assert not _path_is_cognitive_artifact("read_file", {"path": "docs/reference-books-migration.md"})
    assert len(_truncate_tool_result(oversized, "read_file",
                                     {"path": "docs/reference-books-migration.md"})) < len(oversized)


def test_every_chapter_over_the_read_cap_is_covered_by_the_prefix_guarantee():
    """The chapters that actually need it, measured rather than assumed."""
    from ouroboros.loop_tool_execution import _path_is_cognitive_artifact
    from ouroboros.reference_books import load_reference_book
    from ouroboros.tool_capabilities import TOOL_RESULT_LIMITS

    oversized = []
    for book_id in BOOK_ENTRYPOINTS:
        for chapter in load_reference_book(REPO, book_id).chapters:
            assert _path_is_cognitive_artifact("read_file", {"path": chapter.source_path})
            if len(chapter.raw) > TOOL_RESULT_LIMITS["read_file"]:
                oversized.append(chapter.source_path)
    assert oversized, "if no chapter exceeds the cap, this guarantee needs a different proof"


# --- the mandatory-read corpus --------------------------------------------

def test_the_mandatory_read_pointer_measures_the_book_not_its_membership_page():
    from ouroboros.tools import preflight_review_prompt as prompt

    measured = prompt._mandatory_read_corpus_chars(REPO)
    entrypoints = sum(len((REPO / rel).read_text(encoding="utf-8"))
                      for rel in BOOK_ENTRYPOINTS.values())
    assert measured > 20 * entrypoints, (
        "a retrieving reviewer told to read the books in full must be budgeted "
        f"for their chapters; measured {measured} against {entrypoints} of membership"
    )
    assert set(prompt._MANDATORY_READ_DOCS) >= set(BOOK_ENTRYPOINTS.values())


# --- physical reference and coverage readers -------------------------------

def test_the_triad_session_task_addresses_chapters_never_the_membership_page():
    """A session receives no assembled evidence, so its map IS its addressing.
    Built from the supplied composed text against the entrypoint path it would
    hand out offsets into a 20-line file."""
    from ouroboros.tools.review_helpers import load_governance_doc
    from ouroboros.tools.review_subject import build_triad_session_task

    sections = dict(
        goal_section="## Goal\n\nx", scope_section="## Scope\n\nx",
        checklist_section="## Checklist\n\nx", rebuttal_section="",
        review_history_section="",
        dev_guide_text=load_governance_doc(REPO, "docs/DEVELOPMENT.md"),
        architecture_text=load_governance_doc(REPO, "docs/ARCHITECTURE.md"),
    )
    with_book = build_triad_session_task(governance_repo_dir=REPO, **sections)
    assert "Source: `docs/architecture/06-agent-core.md`" in with_book
    assert "Source: `docs/development/14-build-and-ci.md`" in with_book
    assert "## ARCHITECTURE.md (navigation map)" not in with_book

    # No governance root: an explicitly historical or synthetic input is still
    # mapped, as the one source it was handed.
    without = build_triad_session_task(**sections)
    assert "## ARCHITECTURE.md (navigation map)" in without


def test_a_non_constitutional_plan_pointer_maps_the_chapters(tmp_path):
    from ouroboros.tools.plan_review_runtime import _architecture_navigation

    mapped = _architecture_navigation(REPO, "unused")
    assert "Source: `docs/architecture/01-high-level-architecture.md`" in mapped
    assert "Devtools boundary" in mapped
    # An unreadable book falls back to mapping the supplied text rather than
    # dropping the architecture pointer entirely.
    fallback = _architecture_navigation(tmp_path, "# Doc\n\n## Section\n\nBody\n")
    assert "## ARCHITECTURE.md (navigation map)" in fallback and "Section" in fallback


def test_the_scope_session_governance_map_addresses_chapters():
    from ouroboros.tools.scope_review_session import governance_nav_maps

    maps = governance_nav_maps(REPO, ("docs/ARCHITECTURE.md", "docs/CHECKLISTS.md"))
    assert "Source: `docs/architecture/10-key-invariants.md`" in maps
    # A non-book governance document keeps the single-source map.
    assert "## docs/CHECKLISTS.md (navigation map)" in maps


def test_a_mandatory_full_read_pointer_enumerates_the_chapter_closure(tmp_path):
    from ouroboros.reference_books import load_reference_book
    from ouroboros.tools.claude_advisory_review import _mandatory_read_pointer

    pointer = _mandatory_read_pointer(REPO, "docs/DEVELOPMENT.md")
    chapters = load_reference_book(REPO, "development").chapters
    assert "membership page, NOT the book" in pointer
    for chapter in chapters:
        assert str((REPO / chapter.source_path).resolve()) in pointer, chapter.source_path
    assert f"({len(chapters[0].raw):,} bytes)" in pointer

    # A non-book document and a sectioned pointer keep their existing form.
    assert "membership page" not in _mandatory_read_pointer(REPO, "BIBLE.md")
    sectioned = _mandatory_read_pointer(REPO, "docs/CHECKLISTS.md", section="Repo Commit Checklist")
    assert "'## Repo Commit Checklist' section" in sectioned

    # An unassemblable book says its coverage is unknown; it never reports a
    # membership page as the whole book.
    _chaptered_corpus(tmp_path)
    (tmp_path / "docs/development/only.md").unlink()
    broken = _mandatory_read_pointer(tmp_path, "docs/DEVELOPMENT.md")
    assert "coverage is UNKNOWN" in broken
