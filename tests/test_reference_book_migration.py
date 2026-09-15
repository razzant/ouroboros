"""The chapter split moved bytes, not meaning: a reversible byte proof.

Each chapter file was written as a two-line prologue plus the old `##` section
body:

    # <the old section title>
    <blank>
    <one authored introductory paragraph>
    <the old section body, byte for byte, starting with its own newline>

So the move is INVERTIBLE, and the inverse is this module's whole method:

    first  = raw.index(b"\\n\\n")               -> the H1 line is raw[:first]
    second = raw.index(b"\\n\\n", first + 2)    -> the introduction is between them
    section = b"## " + raw[2:first] + raw[second:]

Concatenating those sections in membership order must reproduce the old
monolith from its first `## ` heading to EOF, byte for byte.

That is a claim about HISTORY, so it is checked against history: the base
commit's monolith against the migration commit's chapters, both resolved from
Git (`quick-test` and `full-test` check out with `fetch-depth: 0`). The
migration commit is found by content — the commit that added the transfer
table — so a rebase cannot strand the proof on a rewritten SHA, and a later
ordinary edit to a chapter cannot turn a historical fact red. The structural
half below has no such bound and holds on the working tree forever.

The entrypoint preamble is deliberately NOT part of the byte proof: it was
replaced by one merged/authored paragraph plus the `## Chapters` membership
list, and `docs/reference-books-migration.md` records what it said before. Its
H1 line IS pinned here, because that line is the release version carrier.
"""

import hashlib
import pathlib
import subprocess

import pytest

from ouroboros.reference_books import BOOK_ENTRYPOINTS, load_reference_book

REPO = pathlib.Path(__file__).resolve().parents[1]

# The integration base this migration was cut from.
MIGRATION_BASE = "5585133db86419c1a28673e498de4fb13c6b2d1e"

# Found by content, not pinned: the commit that ADDED the operator transfer
# table is the split commit.
TRANSFER_TABLE = "docs/reference-books-migration.md"

# Recorded at the base commit by the split itself (and mirrored in the transfer
# table): the whole old file, and the part of it that moved -- everything from
# the first `## ` heading to EOF.
OLD_MONOLITHS = {
    "architecture": {
        "old_bytes": 724691,
        "old_sha256": "5db278f8ef5060c4aff5ee1e8743c279661ddd975a311a9bafb5858f32b080de",
        "preamble_bytes": 610,
        "moved_bytes": 724081,
        "moved_sha256": "f1f054c700a15533687e0cf81cf19ac53ffcf022eb179f84c0cbccacbb1d305e",
        "h1": "# Ouroboros v7.0.0 — Architecture & Reference",
    },
    "development": {
        "old_bytes": 275551,
        "old_sha256": "50eb460602f1501915195e3ad1918366312e6292b0dccec6f7ef53f5a5302f3b",
        "preamble_bytes": 60,
        "moved_bytes": 275491,
        "moved_sha256": "bffc00227bc5e91f054b38eaed63acd256c2ddf111e231754bfbe92c7dd122e3",
        "h1": "# DEVELOPMENT.md — Development Principles & Module Guide",
    },
}


def _git(*args: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _migration_commit() -> str | None:
    out = _git("log", "--diff-filter=A", "--format=%H", "--", TRANSFER_TABLE)
    lines = [line for line in (out or b"").decode().split("\n") if line.strip()]
    return lines[-1] if lines else None


def _restore_section(raw: bytes) -> bytes:
    """Undo one chapter's prologue and return the old `## ` section bytes."""
    first = raw.index(b"\n\n")
    second = raw.index(b"\n\n", first + 2)
    h1 = raw[:first]
    assert h1.startswith(b"# "), h1[:40]
    introduction = raw[first + 2:second]
    assert introduction.strip() and b"\n\n" not in introduction, h1
    return b"## " + h1[2:] + raw[second:]


@pytest.mark.parametrize("book_id", sorted(BOOK_ENTRYPOINTS))
def test_the_migration_commit_reconstructs_the_base_monolith_byte_for_byte(book_id):
    recorded = OLD_MONOLITHS[book_id]
    rel = BOOK_ENTRYPOINTS[book_id]
    split = _migration_commit()
    old = _git("show", f"{MIGRATION_BASE}:{rel}")
    if split is None or old is None:
        pytest.fail(
            f"the migration proof needs Git history: base {MIGRATION_BASE} "
            f"{'unreachable' if old is None else 'ok'}, the commit that added "
            f"{TRANSFER_TABLE} {'unreachable' if split is None else 'ok'}. "
            "CI checks out with fetch-depth: 0 for exactly this."
        )

    assert len(old) == recorded["old_bytes"]
    assert hashlib.sha256(old).hexdigest() == recorded["old_sha256"]
    assert old[:recorded["preamble_bytes"]].startswith(recorded["h1"].encode("utf-8"))

    book = load_reference_book(
        REPO, book_id,
        read_bytes=lambda path: _git("show", f"{split}:{path}"),
    )
    assert not book.legacy, f"{book_id} was not chaptered by the migration commit"
    moved = b"".join(_restore_section(chapter.raw) for chapter in book.chapters)

    assert len(moved) == recorded["moved_bytes"]
    assert hashlib.sha256(moved).hexdigest() == recorded["moved_sha256"]
    assert moved == old[recorded["preamble_bytes"]:]


@pytest.mark.parametrize("book_id", sorted(BOOK_ENTRYPOINTS))
def test_every_base_section_is_still_exactly_one_chapter(book_id):
    """The structural half, unbounded in time: the base's `##` titles are the
    chapters' H1 titles, in order, one each. A later edit to a chapter body
    cannot make this red, and a lost, merged or re-titled chapter still can."""
    old = _git("show", f"{MIGRATION_BASE}:{BOOK_ENTRYPOINTS[book_id]}")
    if old is None:
        pytest.skip(f"base commit {MIGRATION_BASE} is unreachable in this checkout")
    in_fence = False
    base_titles = []
    for line in old.decode("utf-8").split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("## "):
            base_titles.append(line[3:].strip())

    book = load_reference_book(REPO, book_id)
    chapter_titles = [chapter.headings[0].title for chapter in book.chapters]
    assert chapter_titles == base_titles


@pytest.mark.parametrize("book_id", sorted(BOOK_ENTRYPOINTS))
def test_entrypoint_keeps_its_h1_line_and_carries_only_the_membership(book_id):
    rel = BOOK_ENTRYPOINTS[book_id]
    raw = (REPO / rel).read_bytes()
    lines = raw.decode("utf-8").split("\n")
    assert lines[0] == OLD_MONOLITHS[book_id]["h1"]
    book = load_reference_book(REPO, book_id)
    # One `## Chapters` heading and nothing else: the entrypoint orients, the
    # chapters carry the book.
    assert [h.title for h in book.entrypoint.headings] == [lines[0][2:], "Chapters"]
    members = [chapter.source_path for chapter in book.chapters]
    assert members == sorted(members), "membership order must be the reading order"
    for chapter in book.chapters:
        assert chapter.source_path.startswith(f"docs/{book_id}/")
        assert f"]({chapter.source_path[len('docs/'):]})" in book.entrypoint.text


def test_no_chapter_body_was_reheaded_into_a_duplicate_title():
    """Relocation kept `###`/`####` levels, so a section title still names one
    physical place across the whole book -- what `read_book_section` needs."""
    for book_id in BOOK_ENTRYPOINTS:
        book = load_reference_book(REPO, book_id)
        titles = [h.title for source in (book.entrypoint, *book.chapters) for h in source.headings]
        duplicates = sorted({t for t in titles if titles.count(t) > 1})
        assert not duplicates, f"{book_id}: ambiguous section titles {duplicates}"


def test_the_transfer_table_is_committed_and_is_not_a_book_member():
    table = REPO / TRANSFER_TABLE
    assert table.is_file(), "the operator transfer table must be reviewable"
    text = table.read_text(encoding="utf-8")
    assert MIGRATION_BASE in text
    for book_id in BOOK_ENTRYPOINTS:
        book = load_reference_book(REPO, book_id)
        assert TRANSFER_TABLE not in [c.source_path for c in book.chapters]
        for chapter in book.chapters:
            assert f"`{chapter.source_path}`" in text, chapter.source_path
        assert OLD_MONOLITHS[book_id]["moved_sha256"] in text
