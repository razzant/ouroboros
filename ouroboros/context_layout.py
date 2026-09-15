"""Reference-book views for Main, children and background cognition.

Main Max keeps the full Architecture capability/WHY map for every task class;
its Development handbook remains tied to the active self-body binding. Low and
Nano carry both books' authored chapter introductions and physical pointers.
Children use that compact orientation while retaining their shared biography;
parent-selected details arrive through ordinary source reads and working views.

Source bytes belong to ReferenceBook; this module only renders the selected
view. Stable full prefixes contain no cwd, task identity or revision counters.
SYSTEM, BIBLE and the common identity/narrative core remain caller-owned.
"""

from __future__ import annotations

import pathlib
from typing import Any, List

from ouroboros.reference_books import ReferenceBook, compose_book, load_reference_book, overview_book

# Protected core: always rendered in full, in every context mode. Encoded as
# data so a drift-guard test can assert no future change demotes it.
TIER0_ALWAYS_FULL = frozenset({
    "system",
    "bible",
    "identity",
    "scratchpad",
    "knowledge_index",
    "recent_dialogue",
})


def _read_doc(env: Any, rel_path: str) -> str:
    try:
        return env.repo_path(rel_path).read_text(encoding="utf-8")
    except Exception:
        return ""


def generate_doc_nav_map(text: str, *, title: str, rel_path: str, instructions: bool = True) -> str:
    """Build a compact, fence-aware navigation map of a markdown doc.

    Lists every ``##`` through ``####`` heading with its inclusive line range
    so the agent knows what exists and where, and can pull the full section on demand via
    ``read_file(root="system_repo", path=rel_path, start_line=A, max_lines=N)``.
    A parent's range includes its complete descendant group, so parent and child
    ranges intentionally overlap. This is a lossless index
    (P1: no silent truncation) — the single canonical file on disk is unchanged.
    """
    lines = text.splitlines()
    total = len(lines)
    headings: List[tuple[int, str, int]] = []  # (level, title, 1-based line)
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.startswith("## "):
            headings.append((2, line[3:].strip(), i))
        elif line.startswith("### "):
            headings.append((3, line[4:].strip(), i))
        elif line.startswith("#### "):
            headings.append((4, line[5:].strip(), i))

    out = [f"## {title} (navigation map)", ""]
    if instructions:
        out += [
            f"Full text is NOT inlined to keep the working context window fit. Read any "
            f"section on demand with `read_file(root=\"system_repo\", path=\"{rel_path}\", "
            f"start_line=A, max_lines=N)` (untruncated). Ranges are inclusive; a "
            f"parent includes its complete descendant group, and `max_lines=B-A+1` "
            f"for `lines A-B`. Sections:",
            "",
        ]
    if not headings:
        out.append(f"- (no `##`/`###`/`####` headings; read `{rel_path}` directly)")
    for idx, (level, htitle, lineno) in enumerate(headings):
        end = total
        for later_level, _later_title, later_lineno in headings[idx + 1:]:
            if later_level <= level:
                end = later_lineno - 1
                break
        indent = "  " * (level - 2)
        out.append(f"{indent}- {htitle} — lines {lineno}-{end}")
    return "\n".join(out)


def book_navigation(book: ReferenceBook) -> str:
    """The compact, CHAPTER-ADDRESSED view of one reference book.

    The authored introductions are the overview (there is no second editable
    summary corpus), and beside each one sits that chapter's own heading index
    with line ranges into the PHYSICAL chapter file. A composed-book line
    number is never a `read_file` locator: the composition exists only in the
    prompt, so an offset taken from it would address the wrong bytes of the
    entrypoint. The read instruction is stated once, by `overview_book`, rather
    than repeated under every chapter.

    The one place a source's SHAPE still decides anything: an exact historical
    revision is one physical file, and the single-source mapper is the right
    map for it. That is a fact about the revision, not a migration switch —
    callers declare the VIEW they want and both shapes answer.
    """
    if book.legacy:
        return generate_doc_nav_map(
            book.entrypoint.text,
            title=pathlib.PurePosixPath(book.entrypoint.source_path).name,
            rel_path=book.entrypoint.source_path,
        )
    return overview_book(
        book,
        lambda chapter: generate_doc_nav_map(
            chapter.text,
            title=chapter.headings[0].title if chapter.headings else chapter.source_path,
            rel_path=chapter.source_path,
            instructions=False,
        ),
    ).text


def architecture_context_section(
    env: Any,
    *,
    context_mode: str,
    text: str | None = None,
    book: ReferenceBook | None = None,
) -> str:
    """ARCHITECTURE.md: full in max, navigation map in low. Empty if unreadable."""
    if book is None and text is None:
        try:
            book = load_reference_book(env.repo_dir, "architecture", lambda path: env.repo_path(path).read_bytes())
        except (OSError, ValueError) as exc:
            return f"Reference book source unavailable: docs/ARCHITECTURE.md. {exc}. Full context is not established."
    if book is not None:
        # View intent, not a migration flag: a compact mode asks for the
        # navigable overview and a full mode asks for the composition. Both
        # answers are defined for a legacy revision too.
        if context_mode in {"low", "nano"}:
            return book_navigation(book)
        if text is None:
            text = compose_book(book)
    if text is None:
        text = _read_doc(env, "docs/ARCHITECTURE.md")
    if not text.strip():
        return ""
    if context_mode in {"low", "nano"}:
        return generate_doc_nav_map(
            text, title="ARCHITECTURE.md", rel_path="docs/ARCHITECTURE.md"
        )
    return "## ARCHITECTURE.md\n\n" + text


def reference_doc_sections(
    env: Any,
    *,
    context_mode: str,
    include_development: bool,
    architecture_text: str | None = None,
    development_text: str | None = None,
    books: tuple[ReferenceBook, ...] = (),
) -> List[str]:
    """Render one captured book pair; missing bodies remain named and readable.

    Max's handbook inclusion follows the caller's active-repository decision.
    Compact modes orient the mind to both books and retain exact physical
    pointers; they never turn a composed-book line into a file address.
    """
    parts: List[str] = []
    on_demand: List[str] = []
    by_id = {book.book_id: book for book in books}

    arch_section = architecture_context_section(
        env,
        context_mode=context_mode,
        text=architecture_text,
        book=by_id.get("architecture"),
    )
    if arch_section:
        parts.append(arch_section)

    development_book = by_id.get("development")
    if development_book is None and development_text is None:
        try:
            development_book = load_reference_book(env.repo_dir, "development", lambda path: env.repo_path(path).read_bytes())
        except (OSError, ValueError) as exc:
            development_text = f"Reference book source unavailable: docs/DEVELOPMENT.md. {exc}. Full context is not established."
    dev_text = (
        development_text
        if development_text is not None
        else _read_doc(env, "docs/DEVELOPMENT.md")
    )
    if development_book is not None and development_text is None:
        dev_text = compose_book(development_book)
    if dev_text.strip():
        if context_mode in {"low", "nano"}:
            parts.append(book_navigation(development_book) if development_book is not None
                         else generate_doc_nav_map(dev_text, title="DEVELOPMENT.md", rel_path="docs/DEVELOPMENT.md"))
        elif include_development:
            parts.append("## DEVELOPMENT.md\n\n" + dev_text)
        else:
            on_demand.append("docs/DEVELOPMENT.md")

    # README (user-facing) and CHECKLISTS (reviewers load their own copy) are not
    # inlined in the agent context in any mode.
    on_demand.extend(["README.md", "docs/CHECKLISTS.md"])

    if on_demand:
        listing = ", ".join(f"`{p}`" for p in on_demand)
        parts.append(
            "## Reference docs available on demand\n\n"
            f"Not inlined in the working context: {listing}. "
            "Read them in full (untruncated) with `read_file(root=\"system_repo\", path=...)` when relevant."
        )
    return parts
