"""Two reference books composed from one explicit Markdown membership list.

Each entrypoint carries its H1, one authored introduction and the ordered
`## Chapters` list; each member carries its own H1, one authored introduction
and its subject body. An exact historical revision that predates the split is
still read as the one physical source it was. Chapter bytes are never copied to
a second editable corpus, full composition has no runtime paths or revision
stamps, and selected views carry physical refs separately.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import unquote, urlsplit

from ouroboros.markdown_source import MarkdownSource, SourceRange, parse_markdown_source


BOOK_ENTRYPOINTS = {
    "architecture": "docs/ARCHITECTURE.md",
    "development": "docs/DEVELOPMENT.md",
}


def book_path_role(path: str) -> str:
    """``"entrypoint"``, ``"chapter"`` or ``""`` for a repository path.

    Pure path shape and no I/O, so every consumer that must treat a relocated
    chapter exactly as it treated the monolith it came out of — canonical
    requiredness, untruncated reads, pack duplicate suppression, the
    new-module documentation gate — asks ONE question instead of carrying its
    own copy of the chapter population.
    """
    normalized = str(path or "").replace("\\", "/").lstrip("./")
    if normalized in set(BOOK_ENTRYPOINTS.values()):
        return "entrypoint"
    for book_id in BOOK_ENTRYPOINTS:
        if normalized.startswith(f"docs/{book_id}/") and normalized.endswith(".md"):
            return "chapter"
    return ""


def book_entrypoint_for(path: str) -> str:
    """The entrypoint of the book this path belongs to, else ``""``."""
    normalized = str(path or "").replace("\\", "/").lstrip("./")
    role = book_path_role(normalized)
    if role == "entrypoint":
        return normalized
    if role == "chapter":
        return BOOK_ENTRYPOINTS[normalized.split("/")[1]]
    return ""


@dataclass(frozen=True)
class ReferenceBook:
    book_id: str
    entrypoint: MarkdownSource
    chapters: tuple[MarkdownSource, ...]
    legacy: bool


@dataclass(frozen=True)
class BookSourceRef:
    book_id: str
    path: str
    sha256: str
    span: SourceRange


@dataclass(frozen=True)
class BookView:
    """A physical-source view, not evidence of whole-book/reviewer coverage."""

    text: str
    sources: tuple[BookSourceRef, ...]
    source_complete: bool


def _ref(book: ReferenceBook, source: MarkdownSource, span: SourceRange) -> BookSourceRef:
    return BookSourceRef(book.book_id, source.source_path, source.sha256, span)


def _whole(source: MarkdownSource) -> SourceRange:
    return SourceRange(0, len(source.raw), 1, source.body_span.end_line)


def _preamble(source: MarkdownSource) -> SourceRange:
    """The authored introduction: the FIRST paragraph after a source's H1.

    It must open the chapter — a source whose H1 is followed straight by a
    subsection has no introduction and is refused, which is the property that
    keeps an overview from quoting body prose as if someone had written it for
    that purpose. What it deliberately does NOT require is that the
    introduction be the only paragraph before the first subsection: a chapter
    carries its relocated section body at the heading level that body already
    had, and most sections open with prose, so demanding a single paragraph
    would force either a rewritten heading level or an invented sub-heading.
    The overview says in its own words that it holds introductions, not
    complete chapters, and carries each chapter's physical path beside them.
    """
    first = source.headings[0] if source.headings else None
    if first is None or first.level != 1 or not first.title:
        raise ValueError(f"{source.source_path}: chapter needs a nonempty H1")
    stop = next((h.span.start_byte for h in source.headings[1:]), len(source.raw))
    intro = next((p for p in source.paragraphs if first.span.end_byte <= p.start_byte < stop), None)
    if intro is None or not source.text_at(intro).strip():
        raise ValueError(f"{source.source_path}: chapter needs an authored introductory paragraph under its H1")
    return intro


def _member_paths(entrypoint: MarkdownSource, book_id: str) -> tuple[str, ...] | None:
    sections = [h for h in entrypoint.headings if h.level == 2 and h.title == "Chapters"]
    if not sections:
        return None
    if len(sections) != 1:
        raise ValueError(f"{entrypoint.source_path}: expected one Chapters section")
    section = sections[0]
    lists = [s for s in entrypoint.lists if section.span.end_byte <= s.start_byte < section.section.end_byte]
    if len(lists) != 1:
        raise ValueError(f"{entrypoint.source_path}: Chapters needs one nonempty, flat Markdown list")
    membership = lists[0]
    items = [s for s in entrypoint.list_items if membership.start_byte <= s.start_byte < membership.end_byte]
    paths: list[str] = []
    for item in items:
        links = [link for link in entrypoint.links if item.start_byte <= link.span.start_byte < item.end_byte]
        if len(links) != 1:
            raise ValueError(f"{entrypoint.source_path}:{item.start_line}: each chapter item needs one Markdown link")
        target = urlsplit(links[0].destination)
        if target.scheme or target.netloc or target.query or target.fragment:
            raise ValueError("book membership must link to a whole local chapter")
        path = posixpath.normpath(posixpath.join(posixpath.dirname(entrypoint.source_path), unquote(target.path)))
        if not path.startswith(f"docs/{book_id}/") or not path.endswith(".md"):
            raise ValueError(f"book chapter is outside docs/{book_id}/: {path}")
        if path in paths:
            raise ValueError(f"duplicate book chapter: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"{entrypoint.source_path}: empty Chapters list")
    return tuple(paths)


def load_reference_book(
    repo_root: Path,
    book_id: str,
    read_bytes: Callable[[str], bytes] | None = None,
) -> ReferenceBook:
    """Read one source revision; the callback may read the index or an exact ref."""
    if book_id not in BOOK_ENTRYPOINTS:
        raise ValueError(f"unknown reference book: {book_id}")
    reader = read_bytes if read_bytes is not None else lambda path: (Path(repo_root) / path).read_bytes()
    entry_path = BOOK_ENTRYPOINTS[book_id]
    entrypoint = parse_markdown_source(reader(entry_path), entry_path)
    members = _member_paths(entrypoint, book_id)
    chapters = tuple(parse_markdown_source(reader(path), path) for path in members or ())
    book = ReferenceBook(book_id, entrypoint, chapters, members is None)
    problems = validate_reference_book(book)
    if problems:
        raise ValueError("; ".join(problems))
    return book


def validate_reference_book(book: ReferenceBook) -> tuple[str, ...]:
    """Structural facts only; introductory meaning and WHY remain review work."""
    problems: list[str] = []
    if not book.entrypoint.text.strip():
        problems.append(f"{book.entrypoint.source_path}: empty book entrypoint")
    try:
        members = _member_paths(book.entrypoint, book.book_id)
        if book.legacy != (members is None) or tuple(s.source_path for s in book.chapters) != (members or ()):
            problems.append("book sources do not match the entrypoint's complete ordered membership")
    except ValueError as exc:
        problems.append(str(exc))
    if not book.legacy:
        for chapter in (book.entrypoint, *book.chapters):
            try:
                _preamble(chapter)
            except ValueError as exc:
                problems.append(str(exc))
    return tuple(problems)


def validate_reference_books(
    repo_root: Path, *, tracked_paths: Iterable[str], require_chaptered: bool,
    read_bytes: Callable[[str], bytes] | None = None,
) -> tuple[str, ...]:
    """Check the exact supplied tree for CI or preflight, including docs-only work.

    The caller selects the migration phase explicitly and supplies the candidate's
    tracked population and byte reader (index/ref/checkout). No Git invocation,
    implicit chapter discovery or semantic documentation judgment happens here.
    """
    tracked = set(tracked_paths)
    problems: list[str] = []
    for book_id, entrypoint in BOOK_ENTRYPOINTS.items():
        if entrypoint not in tracked:
            problems.append(f"{entrypoint}: entrypoint is absent from the candidate tree")
        try:
            book = load_reference_book(repo_root, book_id, read_bytes)
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"{entrypoint}: {exc}")
            continue
        if require_chaptered and book.legacy:
            problems.append(f"{entrypoint}: chaptered source required; missing Chapters membership")
        expected = {chapter.source_path for chapter in book.chapters}
        population = {path for path in tracked
                      if path.startswith(f"docs/{book_id}/") and path.endswith(".md")}
        problems.extend(f"{entrypoint}: unlisted chapter {path}" for path in sorted(population - expected))
        problems.extend(f"{entrypoint}: chapter absent from candidate tree {path}" for path in sorted(expected - population))
        for source in (book.entrypoint, *book.chapters):
            spans = [source.body_span, *source.paragraphs, *source.lists, *source.list_items,
                     *source.code_blocks, *(link.span for link in source.links)]
            spans.extend(span for heading in source.headings for span in (heading.span, heading.section))
            if source.frontmatter_span:
                spans.append(source.frontmatter_span)
            for span in spans:
                if not (0 <= span.start_byte <= span.end_byte <= len(source.raw)
                        and span.start_line == source.raw.count(b"\n", 0, span.start_byte) + 1
                        and span.end_line == source.raw.count(b"\n", 0, max(span.start_byte, span.end_byte - 1)) + 1):
                    problems.append(f"{source.source_path}: invalid physical source range {span}")
                    break
    return tuple(problems)


def compose_book(book: ReferenceBook) -> str:
    """Legacy bytes stay exact; chapter source bytes stay exact within the result."""
    return "\n\n".join(source.text for source in (book.entrypoint, *book.chapters))


def overview_book(
    book: ReferenceBook,
    chapter_navigation: Callable[[MarkdownSource], str] | None = None,
) -> BookView:
    """The compact view: authored introductions plus physical source addresses.

    ``chapter_navigation`` is INJECTED rather than imported, because the one
    heading mapper lives in the doc-layout owner above this module. A compact
    view that lost the subsection index the monolith's map carried would be a
    capability regression for every reader that navigates before reading.
    """
    read_instruction = (
        'Full chapter text is available through `read_file(root="system_repo", path=...)`; '
        'use the physical Source path listed below, with `start_line` and `max_lines` '
        'for a selected range. This overview contains introductions, not complete chapters.'
    )
    if book.legacy:
        source = book.entrypoint
        rows = [f"# {book.book_id.title()} (source navigation)",
                f"Full source: `{source.source_path}`", ""]
        for heading in source.headings:
            rows.append(f"- {heading.title}: `{source.source_path}` lines {heading.section.start_line}-{heading.section.end_line}")
        return BookView("\n".join(rows), tuple(_ref(book, source, h.span) for h in source.headings), False)
    rows = [book.entrypoint.text, read_instruction]
    refs = [_ref(book, book.entrypoint, _whole(book.entrypoint))]
    for chapter in book.chapters:
        preamble = _preamble(chapter)
        rows.extend((f"# {chapter.headings[0].title}",
                     f"Source: `{chapter.source_path}`", chapter.text_at(preamble)))
        navigation = chapter_navigation(chapter) if chapter_navigation is not None else ""
        if navigation.strip():
            rows.append(navigation)
        refs.extend((_ref(book, chapter, chapter.headings[0].span), _ref(book, chapter, preamble)))
    return BookView("\n\n".join(rows), tuple(refs), False)


def read_book_range(
    book: ReferenceBook, chapter_path: str, start_line: int = 1,
    max_lines: int | None = None,
) -> BookView:
    sources = {source.source_path: source for source in (book.entrypoint, *book.chapters)}
    if chapter_path not in sources:
        raise ValueError(f"source is not a member of {book.book_id}: {chapter_path}")
    source = sources[chapter_path]
    span = source.lines(start_line, max_lines)
    return BookView(source.text_at(span), (_ref(book, source, span),),
                    span.start_byte == 0 and span.end_byte == len(source.raw))


def read_book_section(book: ReferenceBook, title: str) -> BookView:
    """Read one exact named section, retaining its physical source and revision."""
    matches = [(source, heading) for source in (book.entrypoint, *book.chapters)
               for heading in source.headings if heading.title == title]
    if len(matches) != 1:
        raise ValueError(f"{book.book_id}: expected one section {title!r}; found {len(matches)}")
    source, heading = matches[0]
    span = heading.section
    return read_book_range(book, source.source_path, span.start_line,
                           span.end_line - span.start_line + 1)
