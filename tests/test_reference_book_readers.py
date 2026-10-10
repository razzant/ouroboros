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
    from ouroboros.tools.review_helpers import (
        canonical_governance_sources,
        is_canonical_governance_path,
    )

    assert is_canonical_governance_path("docs/architecture/06-agent-core.md")
    assert is_canonical_governance_path("docs/DEVELOPMENT.md")
    assert not is_canonical_governance_path("docs/reference-books-migration.md")

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


# --- chapter reads under the measured first show ---------------------------

def test_a_chapter_read_is_an_ordinary_result_under_the_measured_frame():
    """Owner Q4: the monolith exempted book chapters from any delivery bound; the
    books stay whole where they are REQUIRED (the system prompt), while a tool
    READ of a chapter is a result like any other -- whole when the measured frame
    holds it, a head+tail range with its exact source when it does not. No path
    class decides that; the compat renderer invents no cap without a measurement."""
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import UNTRUNCATED_REPO_READ_PREFIXES, requested_result_view
    from ouroboros.tool_result_delivery import RESULT_VIEW_MARKER

    oversized = "x" * 85_000 + "\nLAST LINE"
    for rel in ("docs/architecture/06-agent-core.md", "docs/development/06-rules-by-change-class.md",
                "docs/reference-books-migration.md"):
        args = {"path": rel}
        assert requested_result_view(args) is None  # a plain read asks for no particular form
        assert _truncate_tool_result(oversized, "read_file", args) == oversized, rel
        view = _truncate_tool_result(oversized, "read_file", args, allowance_chars=6_000)
        assert len(view) < len(oversized) and "LAST LINE" in view and RESULT_VIEW_MARKER in view, rel
        assert "or page this tool (offset/limit) for the omitted range" in view or "FULL_RESULT_SOURCE_UNAVAILABLE" in view
    # The book-source class still exists as classification (prompts, both books).
    assert "docs/architecture/" in UNTRUNCATED_REPO_READ_PREFIXES


def test_chapters_exceed_a_single_turn_so_range_delivery_is_load_bearing():
    """Measured rather than assumed: some chapters are longer than the retired
    80,000-char read page, so a chapter read can only ever be whole under a frame
    that actually holds it."""
    from ouroboros.reference_books import load_reference_book
    from ouroboros.tool_capabilities import TOOL_RESULT_LIMITS, UNTRUNCATED_REPO_READ_PREFIXES

    oversized = []
    for book_id in BOOK_ENTRYPOINTS:
        for chapter in load_reference_book(REPO, book_id).chapters:
            assert chapter.source_path.startswith(UNTRUNCATED_REPO_READ_PREFIXES)
            if len(chapter.raw) > TOOL_RESULT_LIMITS["read_file"]:
                oversized.append(chapter.source_path)
    assert oversized, "if no chapter exceeds the page, range delivery needs a different proof"


# --- the mandatory-read corpus --------------------------------------------

def test_the_governance_tiers_account_for_the_chapters_not_the_membership_page():
    """A reviewer budgeted for a book must be budgeted for its CHAPTERS: the
    governance tiers select, measure and disclose chapter sources, so an
    entrypoint's own membership page never stands in for the book it lists."""
    from ouroboros.reference_books import book_path_role
    from ouroboros.tools.governance_context import governance_context

    rows = {row["path"]: row for row in governance_context(
        REPO, surface="preflight", touched_paths=["ouroboros/loop.py"], delivery="retrieving").manifest}
    entrypoints = sum(len((REPO / rel).read_text(encoding="utf-8"))
                      for rel in BOOK_ENTRYPOINTS.values())
    chapters = [row for path, row in rows.items() if book_path_role(path) == "chapter"]
    measured = sum(int(row["chars"]) for row in chapters)
    assert measured > 20 * entrypoints, (
        f"measured {measured} of chapters against {entrypoints} of membership"
    )
    for rel in BOOK_ENTRYPOINTS.values():
        assert rows[rel]["disposition"] == "navigation", rel


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
    # An unreadable book falls back to mapping the supplied text rather than
    # dropping the architecture pointer entirely.
    fallback = _architecture_navigation(tmp_path, "# Doc\n\n## Section\n\nBody\n")
    assert "## ARCHITECTURE.md (navigation map)" in fallback and "Section" in fallback


def test_the_scope_session_governance_map_addresses_chapters():
    from ouroboros.tools.governance_context import governance_context

    maps = governance_context(
        REPO, surface="scope", touched_paths=(), delivery="retrieving",
        checklist_section_text="(scope checklist)").navigation
    assert "Source: `docs/architecture/10-key-invariants.md`" in maps
    # The checklist book arrives as its applicable section plus a pointer to the
    # rest, so the navigation names it without mapping it.
    assert "`docs/CHECKLISTS.md` — the complete checklist book" in maps


def test_a_crlf_checkout_still_withholds_the_chapter_its_composed_book_carries(tmp_path):
    """Windows: a checkout that rewrote the chapter files with CRLF (no LF pin, or a
    fixture written with the platform newline) composes the book from those exact
    bytes, so the pack's duplicate check must compare the touched chapter's exact
    bytes too — a newline-translating read never finds it inside the composition."""
    from ouroboros.tools.review_file_pack import triad_pack_exclusions
    from ouroboros.tools.review_helpers import load_governance_doc

    files = _chaptered_corpus(tmp_path)
    for rel, text in files.items():
        (tmp_path / rel).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    composed = load_governance_doc(tmp_path, "docs/ARCHITECTURE.md")
    assert composed == "\n\n".join(
        files[rel].replace("\n", "\r\n") for rel in ("docs/ARCHITECTURE.md", "docs/architecture/only.md")
    )
    excluded, _note = triad_pack_exclusions(
        tmp_path, ["docs/architecture/only.md"], prefix_texts={"docs/ARCHITECTURE.md": composed},
    )
    assert "docs/architecture/only.md" in excluded
    # A plain document's prefix copy came through a newline-translating read:
    # the CRLF file on disk is still the same text and is still withheld.
    (tmp_path / "BIBLE.md").write_bytes(b"# Constitution\r\nOne law.\r\n")
    excluded, _note = triad_pack_exclusions(
        tmp_path, ["BIBLE.md"], prefix_texts={"BIBLE.md": "# Constitution\nOne law.\n"},
    )
    assert "BIBLE.md" in excluded
