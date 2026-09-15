"""The ONE reader tests use for a governance document's complete text.

`docs/ARCHITECTURE.md` and `docs/DEVELOPMENT.md` are reference-book
entrypoints: an orientation paragraph and an ordered `## Chapters` membership
list. Reading either with `Path.read_text()` returns that page and none of the
book, so a substring pin over it stops testing anything while still passing --
which is worse than failing. Every test that asserts something about a
governance document's CONTENT resolves it here instead, and a book resolves to
its composed text exactly as the review surfaces receive it.
"""

from __future__ import annotations

import pathlib

from ouroboros.reference_books import BOOK_ENTRYPOINTS, compose_book, load_reference_book

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def governance_doc_text(rel_path: str, repo_root: pathlib.Path | None = None) -> str:
    """The complete text of one governance document, composed when it is a book."""
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    book_id = next((key for key, entry in BOOK_ENTRYPOINTS.items() if entry == rel_path), None)
    if book_id is not None:
        return compose_book(load_reference_book(root, book_id))
    return (root / rel_path).read_text(encoding="utf-8")


def architecture_text(repo_root: pathlib.Path | None = None) -> str:
    return governance_doc_text(BOOK_ENTRYPOINTS["architecture"], repo_root)


def development_text(repo_root: pathlib.Path | None = None) -> str:
    return governance_doc_text(BOOK_ENTRYPOINTS["development"], repo_root)
