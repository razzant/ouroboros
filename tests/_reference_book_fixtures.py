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
    """``core`` with two real chaptered books and the resident Body Map the real renderer gives them
    (``reference_books.body_map``): ``body_map`` whole and ``body_map_parts``, its parts and addresses, which a
    lowered frame shows with the Map's full text one retained read away (``context_layout.body_map_section``).
    The Architecture chapter's authored opening names ``headings`` mechanisms (``Mechanism {i}: its facts.``, each
    with ``body``), so they size the Map and stand only in its whole form; the Development chapter carries
    ``headings`` rules (``## Rule {i}``, each with ``body``), which only the whole handbook holds. The entrypoint's
    Reference Atlas gives the Map one Atlas row (``DESIGN.md``)."""
    from dataclasses import replace

    from ouroboros.reference_books import body_map, compose_book, load_reference_book

    repo = pathlib.Path(root) / "books"
    opening = "How the core works.\n" + "\n".join(f"Mechanism {i}: its facts. {body}".rstrip() for i in range(headings))
    for path, text in {
            "docs/ARCHITECTURE.md": ("# Body\n\nThe body map.\n\n## Chapters\n\n- [Core](architecture/core.md)\n\n"
                                     "## Reference Atlas\n\n- [DESIGN.md](DESIGN.md) — how the screens look.\n"),
            "docs/DEVELOPMENT.md": "# Engineering\n\nHow I change.\n\n## Chapters\n\n- [Change](development/change.md)\n",
            "docs/architecture/core.md": f"# Core\n\n{opening}\n\n## Manual\n\nThe engineering manual.\n",
            "docs/development/change.md": f"# {development_title}\n\nHow changes stay coherent.\n\n" + "".join(
                f"## Rule {i}\n\nIts practice. {body}\n\n" for i in range(headings))}.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8", newline="\n")  # LF on every OS, as .gitattributes keeps books
    books = tuple(load_reference_book(repo, book_id) for book_id in ("architecture", "development"))
    view, parts = body_map(*books)
    return replace(core, reference_books=books, body_map=view.text, body_map_parts=parts,
                   development_md=compose_book(books[1]))


def map_saving(core) -> int:
    """What the Body Map of ``core`` saves at its parts and addresses beside a retained reader's receipt
    (``relocated_text``): the lawful step of a lowered frame (``memory_floor.lowered_maps``)."""
    from ouroboros import context_layout, memory_floor
    from ouroboros.section_floor import relocated_text

    section = context_layout.body_map_section(core.body_map, core.body_map_parts, steppable=True)
    reader = f"read_file(root='artifact_store', path='source_handles/context_checkpoints/host-body_map-{'0' * 64}.md')"
    return memory_floor.view_tokens(section.full_text) - memory_floor.view_tokens(relocated_text(section, reader))
