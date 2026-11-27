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
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import unquote, urlsplit

from ouroboros.markdown_source import MarkdownSource, SourceRange, parse_markdown_source
from ouroboros.repo_remotes import OFFICIAL_REPO, remote_fetch_urls
from ouroboros.update_channels import UPDATE_CHANNEL_BRANCHES


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


@lru_cache(maxsize=64)
def _book_source(raw: bytes, source_path: str) -> MarkdownSource:
    """Bounded parser reuse by exact bytes AND physical path, never mtime or TTL.

    Only this book loader uses the memo; mutable YAML metadata is detached below before
    a source escapes. Every load still reads current bytes and current membership.
    """
    return parse_markdown_source(raw, source_path)


def _read_book_source(reader: Callable[[str], bytes], path: str) -> MarkdownSource:
    source = _book_source(reader(path), path)
    return replace(source, frontmatter=deepcopy(source.frontmatter)) if source.frontmatter is not None else source


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
    entrypoint = _read_book_source(reader, entry_path)
    members = _member_paths(entrypoint, book_id)
    chapters = tuple(_read_book_source(reader, path) for path in members or ())
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
    *, instructions: bool = True,
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
    ) if instructions else "This overview contains introductions, not complete chapters."
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


BODY_MAP_ATLAS = "Reference Atlas"
BODY_MAP_READER = 'Each `Source:` path and Atlas address reads with `read_file(root="system_repo", path=...)`.'
#: What stands in the reader's place for an actor whose capability ceiling cannot follow the addresses (BIBLE P1: a
#: disclosure names a source its reader can resolve, or says that it cannot).
BODY_MAP_NO_READER = ("This task's capability ceiling holds no reader for these addresses: they say where each "
                      "manual lives, for a focus that has `read_file` on the system repository.")


def _atlas_rows(entrypoint: MarkdownSource) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The Reference Atlas items as ``(text, linked paths)``: each Markdown link replaced by the repository path
    it resolves to, and those paths apart.

    A link in the entrypoint is relative to ``docs/``; a reader tool takes a repository path, so the resident
    Map carries that. Spans come from the parsed source: no pattern guesses where a link starts or ends.
    """
    section = next((h for h in entrypoint.headings if h.level == 2 and h.title == BODY_MAP_ATLAS), None)
    if section is None:
        return ()
    base = posixpath.dirname(entrypoint.source_path)
    rows = []
    for item in entrypoint.list_items:
        if not section.span.end_byte <= item.start_byte < section.section.end_byte:
            continue
        at, parts, paths = item.start_byte, [], []
        for link in entrypoint.links:
            if item.start_byte <= link.span.start_byte and link.span.end_byte <= item.end_byte:
                target = urlsplit(link.destination)
                if target.scheme or target.netloc:
                    continue  # an outside address stays as written
                paths.append(posixpath.normpath(posixpath.join(base, unquote(target.path))))
                parts += [entrypoint.raw[at:link.span.start_byte].decode("utf-8"), f"`{paths[-1]}`"]
                at = link.span.end_byte
        rows.append(("".join([*parts, entrypoint.raw[at:item.end_byte].decode("utf-8")]).strip(), tuple(paths)))
    return tuple(rows)


def _under_folders(entries: Iterable[tuple[str, str]]) -> list[str]:
    """``(path, label)`` rows with each folder, and an extension all its files share, named once above them: every
    address stays exact as that line with the row's name in place of ``<name>``, and a row keeps its label."""
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    for path, label in entries:
        folder, _, name = path.rpartition("/")
        if not groups or groups[-1][0] != folder:
            groups.append((folder, []))
        groups[-1][1].append((name, label))
    rows = []
    for folder, files in groups:
        shared = {posixpath.splitext(name)[1] for name, _label in files}
        suffix = shared.pop() if len(shared) == 1 else ""
        rows.append(f"`{folder + '/' if folder else ''}<name>{suffix}`:")
        rows += [f"{name[:len(name) - len(suffix)]}: {label}" if label else name[:len(name) - len(suffix)]
                 for name, label in files]
    return rows


def body_map(architecture: ReferenceBook, development: ReferenceBook | None = None, *,
             readable: bool = True) -> tuple[BookView, str]:
    """The resident Body Map and its parts-and-addresses form (BIBLE P1), both rendered from the book sources.

    The Map is the Architecture entrypoint's authored introduction, each chapter's title, physical path and
    authored opening, the Reference Atlas and the Development chapters by title and path. It carries no line
    number, count or release version, so an edit inside a chapter's body or a release leaves its bytes alone;
    the file on disk keeps its version header (``release_sync``). The second value is what stands when a
    physical window cannot hold the Map: every chapter of both books by title and every Atlas address, each
    folder and shared extension named once above its files (``_under_folders``), nothing else. A linked Atlas address is taken from its
    parsed link; a row with no link keeps its address side as written. It is ``""`` for a legacy single-file
    revision, whose only view is its source navigation. ``readable`` false (a Presence turn whose ceiling cannot
    read the system repository) replaces each form's reader line by ``BODY_MAP_NO_READER``.
    """
    if architecture.legacy:  # one physical file: its navigation is its only view; the other book stays one address away
        view = overview_book(architecture)
        return replace(view, text=f"{view.text}\n\nEngineering handbook, read on demand: `{BOOK_ENTRYPOINTS['development']}`."), ""
    entry = architecture.entrypoint
    chapters = [chapter for book in (architecture, development) if book is not None and not book.legacy
                for chapter in book.chapters]
    handbook = [] if development is None or development.legacy else [
        f"- {chapter.headings[0].title}: `{chapter.source_path}`" for chapter in development.chapters]
    atlas = _atlas_rows(entry)
    rows = ["## Body map", entry.text_at(_preamble(entry)).strip(), BODY_MAP_READER if readable else BODY_MAP_NO_READER]
    refs = [_ref(architecture, entry, _preamble(entry))]
    for chapter in architecture.chapters:
        opening = _preamble(chapter)
        rows.append(f"### {chapter.headings[0].title}\nSource: `{chapter.source_path}`\n{chapter.text_at(opening).strip()}")
        refs.extend((_ref(architecture, chapter, chapter.headings[0].span), _ref(architecture, chapter, opening)))
    if atlas:
        rows.append(f"### {BODY_MAP_ATLAS}\n" + "\n".join(text for text, _paths in atlas))
    if handbook:
        rows.append("### Development handbook chapters\n" + "\n".join(handbook))
    parts = ["## Body map (parts and addresses)",
             "\n".join(_under_folders((chapter.source_path, chapter.headings[0].title) for chapter in chapters))]
    if atlas:
        linked = _under_folders((path, "") for _text, paths in atlas for path in paths)
        unlinked = [text.split(" — ", 1)[0] for text, paths in atlas if not paths]  # its address side, as written
        parts.append("\n".join([f"{BODY_MAP_ATLAS}:", *linked, *unlinked]))
    parts.append('Each reads with `read_file(root="system_repo", path=...)`, its name in place of `<name>`.'
                 if readable else BODY_MAP_NO_READER)
    return BookView("\n\n".join(rows), tuple(refs), False), "\n\n".join(parts)


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


# --------------------------------------------------------------------------- text balance
# A measurement, never a gate (BIBLE P3 c5). The size lane needs an explicit
# contribution base; ordinary manual/tag CI cannot certify per-change growth.
#
# The size rule (owner decision 2A, 2026-11-25) follows what a mind carries. What every mind carries in every
# request ends no larger than at its contribution base: BIBLE.md, the prompts as one text (owner decision 3A,
# 2026-10-28: words moved between prompt files cost nothing, a new prompt file is growth) and the rendered Body
# Map. A book chapter is read on demand, so its limit is physical: one default ``read_file``; a chapter that
# outgrows it is split by subject with its reasons kept. The large standalone documents have no size rule: their
# delta is shown as a fact. Each text has three states in a revision: absent (a fact), there but unreadable
# (``UnmeasuredText``: unknown, never absent and never smaller) and measured. The lane renders a base's Map with
# that revision's own code (``revision_body_map_sizes``), so growth of the renderer's own text is growth; a
# revision without a renderer has no earlier Map. The balance shown on a book edit renders every side with the
# running code: that edit changes sources, and a change to the renderer is code the lane measures.

BOOK_GROWTH_RULE = (
    "BIBLE.md, the prompts (prompts/*.md as one text) and the rendered Body Map (the Architecture entry, each "
    "chapter's opening, the Reference Atlas and the Development chapters by title and path) ride every request, "
    "so each must end no larger than at its contribution base; added text is paid in the same text by a statement "
    "that leaves, never by a lost fact (DEVELOPMENT \"Paying down a size cap\"). Manual/tag CI without that base "
    "does not verify this rule. The owner's book-growth label can approve an exception; local commits are not blocked."
)
# One default ``read_file``: its line window (the tool's ``max_lines`` default) and its result limit in characters
# (``tool_capabilities.TOOL_RESULT_LIMITS``), both pinned to their owners by ``tests/test_reference_book_readers.py``.
CHAPTER_READ_LINES, CHAPTER_READ_CHARS = 2000, 80_000
CHAPTER_BAND_RULE = (
    f"A book chapter is read on demand in one default read_file ({CHAPTER_READ_LINES:,} lines, {CHAPTER_READ_CHARS:,} "
    "characters); a chapter that outgrows it is split by subject with its reasons kept, never trimmed to fit."
)

RULED_TEXTS = {"constitution": "BIBLE.md", "prompts": "prompts"}
BODY_MAP_TEXT = "body_map"
# Large documents read on demand outside the two books: their size delta is a fact in the balance, under no rule.
STANDALONE_DOCS = ("docs/DESIGN.md", "docs/PERSISTENCE.md", "docs/CLIENT_CONTRACT.md", "docs/CREATING_SKILLS.md",
                   "docs/CHECKLISTS.md")
_TEXT_NAMES = {"constitution": "BIBLE.md", "prompts": "Prompts", BODY_MAP_TEXT: "Body Map"}
# Everything a balance reads in a Git revision: one listing and one blob batch cover it.
_MEASURED_PATHS = (*BOOK_ENTRYPOINTS.values(), *(f"docs/{book_id}" for book_id in BOOK_ENTRYPOINTS),
                   RULED_TEXTS["constitution"], RULED_TEXTS["prompts"], *STANDALONE_DOCS)


def ruled_text_of(path: str) -> str:
    """``"constitution"``, ``"prompts"`` or ``""`` for a repository path; pure path shape."""
    normalized = str(path or "").replace("\\", "/").lstrip("./")
    if normalized == RULED_TEXTS["constitution"]:
        return "constitution"
    folder, _, name = normalized.rpartition("/")
    return "prompts" if folder == RULED_TEXTS["prompts"] and name.endswith(".md") else ""


def text_name(text_id: str) -> str:
    """How a book, a ruled text or a standalone document is named in a balance line."""
    return _TEXT_NAMES.get(text_id) or (text_id if text_id in STANDALONE_DOCS else f"{text_id.title()} book")


class UnmeasuredText(OSError):
    """A text that is there but cannot be read or rendered: its size is unknown, never absent and never smaller."""


@dataclass(frozen=True)
class BookBalance:
    book_id: str
    size: int                              # composed UTF-8 bytes in the working tree
    vs_head: int | None                    # net change against HEAD; None when HEAD has no such book
    vs_upstream: int | None                # contribution delta against the official target's merge-base
    upstream: str                          # local official development ref, "" when unavailable
    changed: tuple[tuple[str, int], ...]   # sources whose bytes differ from HEAD, with their delta
    merge_base: str = ""                   # exact measured commit, never the feature's tracking tip
    ruled: bool = True                     # False: a book or standalone document, whose size is a fact under no rule
    over_read: tuple[tuple[str, int, int], ...] = ()  # a book's chapters past one default read: (path, lines, chars)
    unmeasured: str = ""                   # why the text, there, could not be measured; its sizes are then unknown

    @property
    def owed(self) -> int:
        """Known growth of a ruled text for warning consumers: contribution when measured, else HEAD only."""
        delta = self.vs_upstream if self.vs_upstream is not None else self.vs_head
        return max(0, delta or 0) if self.ruled else 0


_OFFLINE_GIT = {"GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": ""}


def _git_bytes(root: Path, *args: str, stdin: bytes | None = None) -> bytes | None:
    """stdout of one git read, ``None`` when it fails; a measurement fetches nothing.

    A partial clone fetches an object it lacks once a read needs it. Here that read fails instead
    and its measure stays unknown: Git 2.44+ starts no fetch (``GIT_NO_LAZY_FETCH``), and the fetch
    of an older Git stops before it opens a transport, because the empty ``GIT_ALLOW_PROTOCOL`` allow-list denies every protocol.
    """
    import os
    import subprocess

    try:
        done = subprocess.run(["git", *args], cwd=str(root), capture_output=True, timeout=30, input=stdin,
                              env={**os.environ, **_OFFLINE_GIT})
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _official_fetch_url(url: str) -> bool:
    """Recognize the official GitHub repository, not a remote's conventional name."""
    if "://" not in url:  # Git's scp-like SSH spelling.
        host, colon, path = url.partition(":")
        url = f"ssh://{host}/{path}" if colon else ""
    try:
        parsed = urlsplit(url)
        ports = {"https": 443, "http": 80, "ssh": 22, "git": 9418}
        return (parsed.scheme in ports and parsed.hostname == "github.com"
                and parsed.port in (None, ports[parsed.scheme]) and not parsed.query and not parsed.fragment
                and parsed.path.lower().rstrip("/").removesuffix(".git") == f"/{OFFICIAL_REPO.lower()}")
    except ValueError:
        return False


def _official_contribution_target(root: Path) -> str:
    """Use a cached official development ref; no fetch, tracking-branch or settings fallback.

    Git supplies shared remotes/refs for linked worktrees too. If multiple official
    remotes have this ref, the first by name wins and is shown in every measurement.
    """
    for name, url in remote_fetch_urls(lambda *args: _git_bytes(root, *args)):
        if _official_fetch_url(url):
            ref = f"refs/remotes/{name}/{UPDATE_CHANNEL_BRANCHES['development']}"
            if _git_bytes(root, "rev-parse", "--verify", f"{ref}^{{commit}}"):
                return ref
    return ""


def contribution_base(root: Path) -> tuple[str, str]:
    """``(official ref, HEAD's merge-base with it)``, each ``""`` when unavailable.

    The one base every contribution measurement uses (the book balance and
    ``review.diff_pulse``): a branch's own earlier commits stay inside what it contributes.
    """
    upstream = _official_contribution_target(root)
    base = (_git_bytes(root, "merge-base", "HEAD", upstream) or b"").decode().strip() if upstream else ""
    return upstream, base


class _Revision:
    """The measured sources of one revision: the working tree, the index (``staged``) or ``ref``.

    A Git revision is listed once and every measured blob is read in one ``git cat-file --batch`` the first time any
    is needed, so one operation reads each (revision, path) once. ``FileNotFoundError``: absent there, a fact;
    ``UnmeasuredText``: there but unreadable (an unlisted tree, an object the store lacks, a denied file).
    """

    def __init__(self, root: Path, ref: str = "", staged: bool = False) -> None:
        self.root, self.ref, self.git = Path(root), ref, bool(ref or staged)
        self.where = ref or ("the index" if staged else "the working tree")
        self._blobs: dict[str, bytes | None] | None = None

    def _held(self) -> dict[str, bytes | None]:
        if self._blobs is None:
            listing = (_git_bytes(self.root, "ls-tree", "-r", "-z", "--full-tree", self.ref, "--", *_MEASURED_PATHS)
                       if self.ref else _git_bytes(self.root, "ls-files", "-s", "-z", "--", *_MEASURED_PATHS))
            if listing is None:
                raise UnmeasuredText(f"{self.where} cannot be listed")
            oids = {}  # path -> blob
            for row in listing.decode("utf-8", "replace").split("\0"):
                meta, _, path = row.partition("\t")
                fields = meta.split()
                if not path or len(fields) != 3:
                    continue
                if not self.ref and fields[2] != "0":  # an unmerged path has no single staged form
                    raise UnmeasuredText(f"{self.where} holds {path} unmerged")
                if not self.ref or fields[1] == "blob":
                    oids[path] = fields[2] if self.ref else fields[1]
            out = _git_bytes(self.root, "cat-file", "--batch", stdin="".join(f"{o}\n" for o in oids.values()).encode()
                             ) if oids else b""
            blobs: dict[str, bytes | None] = {}
            at = 0
            for path, oid in oids.items():  # `<oid> blob <size>\n<bytes>\n`, or `<oid> missing\n`, in input order
                end = (out or b"").find(b"\n", at)
                header = out[at:end].split() if out and end >= 0 else []
                if len(header) == 3 and header[0].decode("ascii", "replace") == oid and header[2].isdigit():
                    blobs[path], at = out[end + 1:end + 1 + int(header[2])], end + 2 + int(header[2])
                else:
                    blobs[path], at = None, (end + 1 if end >= 0 else at)
            self._blobs = blobs
        return self._blobs

    def holds(self, path: str) -> bool:
        if self.git:
            return path in self._held()
        try:
            (self.root / path).stat()
        except (FileNotFoundError, NotADirectoryError):
            return False
        except OSError as exc:
            raise UnmeasuredText(f"{path}: {exc}") from exc
        return True

    def paths(self) -> list[str]:
        """Every measured path a Git revision holds."""
        return list(self._held())

    def read(self, path: str) -> bytes:
        if not self.git:
            try:
                return (self.root / path).read_bytes()
            except (FileNotFoundError, NotADirectoryError):
                raise
            except OSError as exc:
                raise UnmeasuredText(f"{path}: {exc}") from exc
        held = self._held()
        if path not in held:
            raise FileNotFoundError(f"{self.where}:{path}")
        blob = held[path]
        if blob is None:
            raise UnmeasuredText(f"{self.where}:{path} is listed but its object cannot be read here")
        return blob

    def size(self, path: str) -> int:
        if not self.git:
            try:
                return (self.root / path).stat().st_size
            except (FileNotFoundError, NotADirectoryError):
                raise
            except OSError as exc:
                raise UnmeasuredText(f"{path}: {exc}") from exc
        return len(self.read(path))


def _unmeasured(text_id: str, revision: _Revision, exc: Exception) -> UnmeasuredText:
    return exc if isinstance(exc, UnmeasuredText) else UnmeasuredText(f"{text_name(text_id)} at {revision.where}: {exc}")


def book_source_sizes(root: Path, book_id: str, ref: str = "", *, staged: bool = False,
                      revision: _Revision | None = None) -> dict[str, int] | None:
    """UTF-8 bytes of each book source, entrypoint first, in the working tree, the index (``staged``) or at ``ref``.

    ``None`` where the book's entrypoint is absent; ``UnmeasuredText`` where the book is there but unreadable.
    """
    revision = revision or _Revision(root, ref, staged)
    entry = BOOK_ENTRYPOINTS[book_id]
    if not revision.holds(entry):
        return None
    try:
        raw = revision.read(entry)
        members = _member_paths(parse_markdown_source(raw, entry), book_id) or ()
        return {entry: len(raw), **{path: revision.size(path) for path in members}}
    except (OSError, ValueError, KeyError, IndexError, UnicodeDecodeError) as exc:
        raise _unmeasured(book_id, revision, exc) from exc


def composed_size(sizes: dict[str, int]) -> int:
    """``len(compose_book(book).encode())`` from source sizes: sources joined by one blank line."""
    return sum(sizes.values()) + 2 * (len(sizes) - 1)


def ruled_text_sizes(root: Path, text_id: str, ref: str = "", *, staged: bool = False,
                     revision: _Revision | None = None) -> dict[str, int] | None:
    """UTF-8 bytes of each file of a ruled text, in the working tree, the index (``staged``) or at ``ref``;
    ``None`` when absent, ``UnmeasuredText`` when there but unreadable."""
    revision = revision or _Revision(root, ref, staged)
    root, where = revision.root, RULED_TEXTS[text_id]
    try:
        if revision.git:
            sizes = {path: revision.size(path) for path in revision.paths() if ruled_text_of(path) == text_id}
        else:  # a Windows checkout may hold CRLF for these unpinned files: count the committed LF form
            files = [root / where] if text_id == "constitution" else sorted((root / where).glob("*.md"))
            sizes = {file.relative_to(root).as_posix(): len(file.read_bytes().replace(b"\r\n", b"\n"))
                     for file in files if file.is_file()}
    except (OSError, ValueError, IndexError) as exc:  # a file denied or gone mid-read: unmeasured, never absent
        raise _unmeasured(text_id, revision, exc) from exc
    return sizes or None


def body_map_sizes(root: Path, text_id: str = BODY_MAP_TEXT, ref: str = "", *, staged: bool = False,
                   revision: _Revision | None = None) -> dict[str, int] | None:
    """UTF-8 bytes of the Body Map this code renders from the books in the working tree, the index or at ``ref``.

    ``None`` where that revision holds no Architecture book (no Map, a fact); a Development book it does not hold
    adds no handbook chapters (a fact as well). ``UnmeasuredText`` where either book is there but cannot be read:
    an unreadable handbook is unknown, never a Map smaller by its chapter list."""
    revision = revision or _Revision(root, ref, staged)
    if not revision.holds(BOOK_ENTRYPOINTS["architecture"]):
        return None
    try:
        architecture = load_reference_book(revision.root, "architecture", revision.read)
        development = (load_reference_book(revision.root, "development", revision.read)
                       if revision.holds(BOOK_ENTRYPOINTS["development"]) else None)
        return {text_id: len(body_map(architecture, development)[0].text.encode("utf-8"))}
    except (OSError, ValueError, KeyError) as exc:
        raise _unmeasured(BODY_MAP_TEXT, revision, exc) from exc


# Run by ``revision_body_map_sizes`` in a child interpreter: every ``ouroboros`` module comes from the revision's own
# tree (the package itself is a bare namespace, so no ``__init__`` runs), and that revision's ``body_map_sizes``
# renders its own books read through Git. It prints the size, or ``null`` where the revision has no renderer.
_REVISION_MAP = r'''
import importlib.abc, importlib.util, json, subprocess, sys
root, ref = sys.argv[1], sys.argv[2]

class Revision(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == "ouroboros":
            return importlib.util.spec_from_loader(name, self, is_package=True)
        if not name.startswith("ouroboros."):
            return None
        stem = name.replace(".", "/")
        for file, package in ((stem + ".py", False), (stem + "/__init__.py", True)):
            done = subprocess.run(["git", "show", f"{ref}:{file}"], cwd=root, capture_output=True, timeout=30)
            if done.returncode == 0:
                spec = importlib.util.spec_from_loader(name, self, origin=f"{ref}:{file}", is_package=package)
                spec.loader_state = done.stdout
                return spec
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__spec__.loader_state is not None:
            exec(compile(module.__spec__.loader_state, module.__spec__.origin, "exec"), module.__dict__)

sys.meta_path.insert(0, Revision())
books = importlib.import_module("ouroboros.reference_books")
sizes = books.body_map_sizes(root, ref=ref) if hasattr(books, "body_map") else None
print(json.dumps(None if sizes is None else sum(sizes.values())))
'''


def revision_body_map_sizes(root: Path, ref: str, text_id: str = BODY_MAP_TEXT) -> dict[str, int] | None:
    """The Body Map revision ``ref`` renders with its OWN code from its own books: what that revision sent.

    ``None`` where the revision has no renderer (no ``ouroboros/reference_books.py`` with ``body_map``): it had no
    Map, so nothing is compared. ``UnmeasuredText`` where its tree cannot be listed or its renderer fails. One child
    interpreter with a deadline and offline Git; only the size lane pays it."""
    import json
    import os
    import subprocess
    import sys

    listing = _git_bytes(Path(root), "ls-tree", "-z", "--full-tree", ref, "--", "ouroboros/reference_books.py")
    if listing is None:
        raise UnmeasuredText(f"{ref}: its tree cannot be listed")
    if not listing:
        return None
    try:
        done = subprocess.run([sys.executable, "-I", "-B", "-c", _REVISION_MAP, str(root), ref], cwd=str(root),
                              capture_output=True, timeout=120, env={**os.environ, **_OFFLINE_GIT})
        size = json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1]) if done.returncode == 0 else ...
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise UnmeasuredText(f"{ref}: its Body Map renderer did not answer: {exc}") from exc
    if size is ... or not (size is None or isinstance(size, int)):
        why = (done.stderr.decode("utf-8", "replace").strip().splitlines() or ["no output"])[-1]
        raise UnmeasuredText(f"{ref}: its Body Map renderer failed: {why}")
    return None if size is None else {text_id: size}


def chapters_over_one_read(root: Path, book_id: str, ref: str = "", *, staged: bool = False,
                           revision: _Revision | None = None) -> tuple[tuple[str, int, int], ...]:
    """``(path, lines, characters)`` of each chapter of ``book_id`` that one default ``read_file`` cannot hold.
    Raises where the book cannot be read: an unmeasured chapter is unknown, never within the band."""
    revision = revision or _Revision(root, ref, staged)
    book = load_reference_book(revision.root, book_id, revision.read)
    sized = ((chapter.source_path, len(chapter.text.splitlines()), len(chapter.text)) for chapter in book.chapters)
    return tuple(row for row in sized if row[1] > CHAPTER_READ_LINES or row[2] > CHAPTER_READ_CHARS)


def standalone_sizes(root: Path, path: str, ref: str = "", *, staged: bool = False,
                     revision: _Revision | None = None) -> dict[str, int] | None:
    """UTF-8 bytes of one standalone document there, ``None`` when absent, ``UnmeasuredText`` when unreadable."""
    revision = revision or _Revision(root, ref, staged)
    return {path: revision.size(path)} if revision.holds(path) else None


def ruled_text_paths(root: Path) -> list[str]:
    """The working-tree files of every ruled text, for a caller that measures everything (it names an unreadable one)."""
    root = Path(root)
    files = [root / RULED_TEXTS["constitution"], *sorted((root / RULED_TEXTS["prompts"]).glob("*.md"))]
    return [file.relative_to(root).as_posix() for file in files if file.is_file()]


def _texts_of(paths: list[str] | None) -> list[tuple[str, Callable[..., dict[str, int] | None], bool]]:
    """``(text id, its measure, ruled)`` of every text ``paths`` touch, all of them when ``None``: the ruled texts,
    the Body Map whenever a book source is touched, the books, the standalone documents."""
    books = list(BOOK_ENTRYPOINTS) if paths is None else sorted(
        {book_id for book_id, entry in BOOK_ENTRYPOINTS.items() for path in paths if book_entrypoint_for(path) == entry})
    texts = list(RULED_TEXTS) if paths is None else sorted({ruled_text_of(path) for path in paths} - {""})
    return ([(text_id, ruled_text_sizes, True) for text_id in texts]
            + ([(BODY_MAP_TEXT, body_map_sizes, True)] if books else [])
            + [(book_id, book_source_sizes, False) for book_id in books]
            + [(path, standalone_sizes, False) for path in STANDALONE_DOCS if paths is None or path in paths])


def book_balances(root: Path, paths: Iterable[str] | None = None, *, staged: bool = False) -> list[BookBalance]:
    """The balance of every text ``paths`` touch (all of them when ``None``), measured in the working tree or,
    ``staged``, in the index a commit would hold; never raises. Ruled texts (``BIBLE.md``, the prompts, and the
    Body Map whenever a book source is touched) carry what they owe; a book and a standalone document carry their
    size as a fact, a book also its chapters past one read. A text absent here is left out (the caller names it); a
    text there but unreadable is a balance whose ``unmeasured`` says why. Each revision is read once."""
    measures = _texts_of(None if paths is None else list(paths))
    if not measures:
        return []
    upstream, base = contribution_base(root)
    now_at, head_at = _Revision(root, staged=staged), _Revision(root, "HEAD")
    base_at = _Revision(root, base) if base else None

    def measured(sizes_at, text_id: str, revision: _Revision | None) -> tuple[dict[str, int] | None, str]:
        try:
            return (sizes_at(root, text_id, revision=revision) if revision is not None else None), ""
        except (OSError, ValueError, KeyError) as exc:
            return None, str(exc) or type(exc).__name__

    balances: list[BookBalance] = []
    for text_id, sizes_at, ruled in measures:
        total = composed_size if text_id in BOOK_ENTRYPOINTS else (lambda sizes: sum(sizes.values()))
        now, why = measured(sizes_at, text_id, now_at)
        over: tuple[tuple[str, int, int], ...] = ()
        if now is not None and text_id in BOOK_ENTRYPOINTS:
            try:
                over = chapters_over_one_read(root, text_id, revision=now_at)
            except (OSError, ValueError, KeyError) as exc:  # sized but unreadable as a book: never within the band
                now, why = None, str(_unmeasured(text_id, now_at, exc))
        if now is None:
            if why:
                balances.append(BookBalance(text_id, 0, None, None, upstream, (), base, ruled, (), why))
            continue
        head, at_base = measured(sizes_at, text_id, head_at)[0], measured(sizes_at, text_id, base_at)[0]
        changed = tuple((path, now.get(path, 0) - (head or {}).get(path, 0))
                        for path in dict.fromkeys([*now, *(head or {})])
                        if now.get(path, 0) != (head or {}).get(path, 0))
        balances.append(BookBalance(
            text_id, total(now),
            total(now) - total(head) if head is not None else None,
            total(now) - total(at_base) if at_base is not None else None,
            upstream, changed if text_id != BODY_MAP_TEXT else (), base, ruled, over))
    return balances


def _signed(value: int | None) -> str:
    return "n/a" if value is None else f"{value:+,} B"


def render_book_balance(balance: BookBalance) -> str:
    """One line: worktree size, HEAD delta, contribution delta and its actual cached base."""
    if balance.unmeasured:
        return f"{text_name(balance.book_id)} balance unavailable: {balance.unmeasured}"
    line = f"{text_name(balance.book_id)} {balance.size:,} B: {_signed(balance.vs_head)} vs HEAD"
    if balance.upstream:
        line += f", {_signed(balance.vs_upstream)} vs {balance.upstream} (merge-base {balance.merge_base or 'unavailable'})"
    if balance.vs_upstream is None:
        line += "; contribution unknown"
    if balance.changed:
        line += " (" + ", ".join(f"{Path(path).stem} {delta:+,}" for path, delta in balance.changed) + ")"
    if not balance.ruled:
        line += " [a fact: read on demand, under no size rule]"
    if balance.over_read:
        line += " [past one read: " + ", ".join(
            f"{Path(path).name} {lines:,} lines/{chars:,} characters" for path, lines, chars in balance.over_read) + "]"
    return line


def book_balance_note(root: Path, paths: Iterable[str], *, staged: bool = False) -> str:
    """The footer for a write to Ouroboros's own body; ``""`` when no book, ruled text or standalone document was written.
    ``staged`` measures the index (a staged commit's subject), otherwise the working tree.

    Callers decide that ``root`` is the body (the system repository or its bound candidate):
    a user's project with a ``docs/architecture`` folder never receives this note.
    """
    touched = [path for path in paths if book_path_role(path) or ruled_text_of(path) or path in STANDALONE_DOCS]
    if not touched:
        return ""
    balances = book_balances(root, touched, staged=staged)
    available = {balance.book_id for balance in balances}
    absent = [f"{text_name(text_id)} is absent here" for text_id, _, _ in _texts_of(touched) if text_id not in available]
    lines = [render_book_balance(balance) for balance in balances] + absent
    rules = ([BOOK_GROWTH_RULE] if any(balance.owed for balance in balances) else []) + (
        [CHAPTER_BAND_RULE] if any(balance.over_read for balance in balances) else [])
    if rules:
        return "ℹ️ Text growth:\n" + "\n".join([*lines, *rules])
    measured = all(balance.vs_upstream is not None and not balance.unmeasured for balance in balances)
    return "ℹ️ Text growth: " + "; ".join(lines) + ("; nothing owed." if measured else ".")
