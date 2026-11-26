"""Shared source text for reference-book balance and context-view tests.

Keep these fixtures outside test modules so tests can reuse the same book shape
without creating a test-to-test import dependency.
"""

from __future__ import annotations

import pathlib
import subprocess

from ouroboros.reference_books import BOOK_ENTRYPOINTS

CHAPTER = "docs/architecture/01-one.md"
ENTRY = "# Architecture\n\nThe map.\n\n## Chapters\n\n- [One](architecture/01-one.md)\n"
BODY = "# One\n\nThe first chapter.\n\nP1 honest and long enough to shorten.\n"


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                                   cwd=root).decode("utf-8").strip()


def _book_repo(root: pathlib.Path, *, official: bool = True) -> pathlib.Path:
    for path, text in {BOOK_ENTRYPOINTS["architecture"]: ENTRY, CHAPTER: BODY,
                       "ouroboros/reference_books.py": "# the book contract\n", "ouroboros/loop.py": "x = 1\n",
                       "notes.md": "notes\n"}.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    if official:
        _git(root, "remote", "add", "canonical", "https://github.com/razzant/ouroboros.git")
        _git(root, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    return root


def _registry(tmp_path: pathlib.Path, *, external: bool):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system = _book_repo(tmp_path / "system")
    drive = tmp_path / "drive"
    drive.mkdir()
    registry = ToolRegistry(repo_dir=system, drive_root=drive)
    if external:  # a user's own project that happens to carry the same docs layout
        project = _book_repo(tmp_path / "project")
        registry.set_context(ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive,
                                         workspace_root=project, workspace_mode="external"))
    return registry


def booked(core, root: pathlib.Path, *, headings: int = 150, body: str = "", development_title: str = "Change"):
    """``core`` with two real chaptered books: the unified Body Map + Reference Atlas is a host section whose
    parts-and-addresses list is one retained read away (``context_layout.body_map_section``); ``headings`` and
    ``body`` size both the Body Map leads and the full chapters."""
    from dataclasses import replace

    from ouroboros.reference_books import compose_book, load_reference_book

    repo = pathlib.Path(root) / "books"
    chapters = max(10, min(24, headings // 25))
    lead_pad = ("Its mechanism and contract boundary in my body. " * max(3, headings // 10))
    chapter_links = "\n".join(f"- [Part {i:02d}](architecture/part-{i:02d}.md)" for i in range(chapters))
    atlas = (
        "\n## Reference Atlas\n\n"
        f"- `docs/DEVELOPMENT.md` — {development_title} handbook ({headings} rules). {lead_pad[:120]}\n"
        "- `docs/DESIGN.md` — UI design bible and visual craft.\n"
    )
    files = {
        "docs/ARCHITECTURE.md": f"# Body\n\nThe body map.\n\n## Chapters\n\n{chapter_links}\n{atlas}",
        "docs/DEVELOPMENT.md": f"# Engineering\n\nHow I change.\n\n## Chapters\n\n- [{development_title}](development/change.md)\n",
        "docs/development/change.md": f"# {development_title}\n\nHow changes stay coherent.\n\n" + "".join(
            f"## Rule {i}\n\nIts practice. {body}\n\n" for i in range(headings)),
    }
    for i in range(chapters):
        files[f"docs/architecture/part-{i:02d}.md"] = (
            f"# Part {i:02d}\n\n"
            f"How part {i:02d} works for {development_title} ({headings} rules): {lead_pad}{body[:240]}\n\n"
            + "".join(f"## Mechanism {j}\n\nIts facts. {body}\n\n" for j in range(max(4, headings // chapters)))
        )
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8", newline="\n")  # LF on every OS, as .gitattributes keeps books
    books = tuple(load_reference_book(repo, book_id) for book_id in ("architecture", "development"))
    return replace(core, reference_books=books, architecture_md=compose_book(books[0]), development_md=compose_book(books[1]))


def map_saving(book, section_id: str) -> int:
    """What one book map saves at its compact address list beside a retained reader's receipt (``relocated_text``)."""
    from ouroboros import context_layout, memory_floor
    from ouroboros.section_floor import relocated_text

    section = (context_layout.body_map_section(book) if section_id == "body_map"
               else context_layout.book_map_section(book, section_id))
    reader = f"read_file(root='artifact_store', path='source_handles/context_checkpoints/host-{section_id}-{'0' * 64}.md')"
    return memory_floor.view_tokens(section.full_text) - memory_floor.view_tokens(relocated_text(section, reader))
