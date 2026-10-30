"""Compressed canon points to real book sections, including the release carrier list."""
import pathlib
import re

import pytest

from ouroboros.reference_books import load_reference_book, read_book_section

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _assert_pointers(text):
    for path in re.findall(r"docs/[\w./-]+\.md", text):
        assert (ROOT / path).is_file(), path
    books = {name.upper(): load_reference_book(ROOT, name) for name in ("architecture", "development")}
    for name, heading in re.findall(r'`docs/(ARCHITECTURE|DEVELOPMENT)\.md`\s+"([^"]+)"', text):
        assert read_book_section(books[name], " ".join(heading.split())).text
    for name, heading in re.findall(r'\b(ARCHITECTURE|DEVELOPMENT)\s+"([^"]+)"', text):
        assert read_book_section(books[name], " ".join(heading.split())).text
    for name, heading in re.findall(r"docs/(ARCHITECTURE|DEVELOPMENT)\.md § ([A-Za-z][^\n)]+)", text):
        assert read_book_section(books[name], heading).text
    for name, number, invariant in re.findall(
            r"`docs/(ARCHITECTURE|DEVELOPMENT)\.md` §(\d+), invariant\s+(\d+)", text):
        book = books[name]
        chapter = next(c for c in book.chapters if c.text.startswith(f"# {number}. "))
        assert re.search(rf"^{invariant}\. ", chapter.text, re.M)
    if "release-sync section" in text:
        assert read_book_section(books["DEVELOPMENT"], "Release sync").text


@pytest.mark.parametrize("path", ["BIBLE.md", "prompts/SYSTEM.md"])
def test_canon_document_paths_and_named_book_sections_resolve(path):
    _assert_pointers((ROOT / path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("pointer", [
    "`docs/ARCHITECTURE.md` \"Review delivery\"",
    "docs/DEVELOPMENT.md § Design System",
    "`docs/ARCHITECTURE.md` §10, invariant 2",
    "`docs/CREATING_SKILLS.md`",
    '(DEVELOPMENT "Capability placement")',
    '(ARCHITECTURE "Review delivery")',
])
def test_dangling_paths_headings_and_invariants_fail_while_real_ones_resolve(pointer):
    _assert_pointers(pointer)
    missing = (pointer.replace("Review delivery", "Missing section")
               .replace("Design System", "Missing section").replace("invariant 2", "invariant 999")
               .replace("Capability placement", "Missing section")
               .replace("CREATING_SKILLS.md", "NO_SUCH_DOC.md"))
    with pytest.raises((AssertionError, ValueError, StopIteration)):
        _assert_pointers(missing)


def test_relocated_release_and_map_rules_have_concrete_destinations():
    book = load_reference_book(ROOT, "architecture")
    for layer in ("Structural", "Operational", "Rationale", "complete registry"):
        assert layer in book.entrypoint.text
    invariant = read_book_section(book, "10. Key Invariants").text.split("\n3.")[0]
    assert "VERSION_CARRIER_SPANS" in invariant
    assert "Release Invariant names (BIBLE)" not in invariant
    fitting = read_book_section(book, "Context fitting, retry, and compaction").text
    assert "owner_nano" in fitting and "rendered_mode=nano" in fitting
