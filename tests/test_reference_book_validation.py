"""Candidate-exact book admission shared by CI and preflight callers."""

import hashlib
import pathlib
import subprocess

import pytest

from ouroboros.reference_books import (
    BOOK_ENTRYPOINTS,
    load_reference_book,
    read_book_section,
    validate_reference_books,
)


def _corpus():
    corpus = {}
    for book_id, entrypoint in BOOK_ENTRYPOINTS.items():
        corpus[entrypoint] = (
            f"# {book_id.title()}\n\nThe authored orientation.\n\n## Chapters\n\n"
            f"- [Useful knowledge]({book_id}/a.md)\n"
        ).encode()
        corpus[f"docs/{book_id}/a.md"] = (
            "# Useful knowledge\n\nWHY this exists and where to look.\n\n"
            "## Exact section\n\nComplete source.\n"
        ).encode()
    return corpus


def _validate(corpus, *, tracked=None, require_chaptered=True):
    return validate_reference_books(
        pathlib.Path("/no-filesystem-source"),
        tracked_paths=corpus if tracked is None else tracked,
        require_chaptered=require_chaptered,
        read_bytes=corpus.__getitem__,
    )


def test_complete_candidate_and_exact_named_section_keep_physical_source():
    corpus = _corpus()
    assert _validate(corpus) == ()
    book = load_reference_book(pathlib.Path("/unused"), "architecture", corpus.__getitem__)
    view = read_book_section(book, "Exact section")
    ref = view.sources[0]
    assert ref.path == "docs/architecture/a.md"
    raw = corpus[ref.path]
    assert ref.sha256 == hashlib.sha256(raw).hexdigest()
    assert raw[ref.span.start_byte:ref.span.end_byte].decode() == view.text
    assert ref.span.start_line == 5


def test_exact_index_bytes_override_worktree_and_population(tmp_path):
    corpus = _corpus()
    for path in corpus:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Deliberately invalid working-tree copy")
    assert validate_reference_books(
        tmp_path, tracked_paths=corpus, require_chaptered=True,
        read_bytes=corpus.__getitem__,
    ) == ()
    assert validate_reference_books(tmp_path, tracked_paths=corpus, require_chaptered=True)


def test_orphan_and_untracked_members_remain_separate_missing_source_facts():
    corpus = _corpus()
    tracked = set(corpus) | {"docs/architecture/orphan.md"}
    tracked.remove("docs/development/a.md")
    problems = _validate(corpus, tracked=tracked)
    assert any("unlisted chapter docs/architecture/orphan.md" in p for p in problems)
    assert any("chapter absent from candidate tree docs/development/a.md" in p for p in problems)
    assert not any("Complete source" in p for p in problems)


def test_short_entrypoint_without_membership_cannot_satisfy_final_admission():
    corpus = {path: b"# Short entrypoint\n\nSome orientation only.\n" for path in BOOK_ENTRYPOINTS.values()}
    assert _validate(corpus, require_chaptered=False) == ()
    problems = _validate(corpus, require_chaptered=True)
    assert len(problems) == 2
    assert all("chaptered source required" in p for p in problems)


@pytest.mark.parametrize("edit,expected", [
    (lambda c: c.pop("docs/architecture/a.md"), "docs/architecture/a.md"),
    (lambda c: c.update({"docs/architecture/a.md": b""}), "nonempty H1"),
    (lambda c: c.update({"docs/architecture/a.md": b"# Chapter\n\n## Details\n\nBody\n"}), "introductory paragraph"),
    (lambda c: c.update({"docs/ARCHITECTURE.md": b"# Book\n\nOrientation.\n\n## Chapters\n\n"}), "nonempty, flat Markdown list"),
    (lambda c: c.update({"docs/ARCHITECTURE.md": c["docs/ARCHITECTURE.md"] + b"- [Again](architecture/a.md)\n"}), "duplicate book chapter"),
    (lambda c: c.update({"docs/architecture/a.md": b"---\nx: [unterminated\n---\n# Body\n"}), "expected"),
])
def test_invalid_candidate_reports_failure_instead_of_partial_success(edit, expected):
    corpus = _corpus()
    edit(corpus)
    problems = _validate(corpus)
    assert problems and any(expected in p for p in problems)


def test_duplicate_named_sections_are_ambiguous_instead_of_first_match():
    corpus = _corpus()
    corpus["docs/architecture/a.md"] += b"\n## Exact section\n\nDifferent source.\n"
    book = load_reference_book(pathlib.Path("/unused"), "architecture", corpus.__getitem__)
    with pytest.raises(ValueError, match="found 2"):
        read_book_section(book, "Exact section")


def test_current_reference_books_pass_the_final_chaptered_admission():
    """The production caller of the validator: the tracked tree, chaptered.

    This is the docs-lane check item 8 asks for -- a missing chapter, an
    unlisted one, or a chapter whose authored introduction was lost fails here
    rather than at the next review that assembles a book.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    assert validate_reference_books(root, tracked_paths=tracked, require_chaptered=True) == ()


def test_docs_sync_reads_full_chapters_and_never_loses_residue_between_files(monkeypatch):
    import tests.test_docs_sync as docs

    corpus = _corpus()
    original_load = load_reference_book
    monkeypatch.setattr(docs, "load_reference_book", lambda root, book_id: original_load(root, book_id, corpus.__getitem__))
    assert "WHY this exists" in docs._read("docs/ARCHITECTURE.md")
    assert "Complete source" in docs._architecture_section("Exact section")
    corpus["docs/development/a.md"] += b"\n### Documentation contract\n\nThe word previously is quoted here.\n"
    extra = "docs/development/b.md"
    corpus[extra] = b"# Next chapter\n\nThis previously behaved differently.\n\n## Details\n\nBody.\n"
    corpus["docs/DEVELOPMENT.md"] += b"- [Next](development/b.md)\n"
    with pytest.raises(AssertionError, match="b.md.*residue grew"):
        docs.test_resident_docs_residue_only_shrinks()
    counts = docs.doc_residue_counts("docs/ARCHITECTURE.md", "# Chapter (v1.2)\n", is_entrypoint=False)
    assert counts["(preamble)"]["version_stamp"] == 1
