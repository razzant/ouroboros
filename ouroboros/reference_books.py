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


BODY_MAP_ATLAS_HEADING = "Reference Atlas"
# Physical one-default-`read_file` band for pulled book chapters (BIBLE P7, Fork 2=A):
# chapters stay inside one default read (2,000 lines / 80,000 chars) and split by subject
# rather than deleting rationale under a whole-book shrink-only cap.
BOOK_CHAPTER_MAX_LINES = 2000
BOOK_CHAPTER_MAX_CHARS = 80_000
STANDALONE_REFERENCE_DOCS = (
    "docs/DESIGN.md",
    "docs/PERSISTENCE.md",
    "docs/CLIENT_CONTRACT.md",
    "docs/CREATING_SKILLS.md",
    "docs/CHECKLISTS.md",
)


def has_body_map_atlas(book: ReferenceBook) -> bool:
    """Whether ``book``'s entrypoint carries the Reference Atlas section."""
    return any(h.level == 2 and h.title == BODY_MAP_ATLAS_HEADING for h in book.entrypoint.headings)


def body_map_book(book: ReferenceBook) -> BookView:
    """The cache-invariant resident Body Map + Reference Atlas (zero line numbers, zero counts).

    Entrypoint (first-person orientation, Chapters list, Reference Atlas) plus each chapter's
    H1 title, physical `Source:` path, and authored first-person opening paragraph. Identical
    across all modes, rooms, wakes, Presence, children, and nannies; edits inside chapter bodies
    never shift its bytes. Legacy single-file revisions fall back to ``overview_book``.
    """
    if book.legacy:
        return overview_book(book, instructions=False)
    rows = [book.entrypoint.text]
    refs = [_ref(book, book.entrypoint, _whole(book.entrypoint))]
    for chapter in book.chapters:
        preamble = _preamble(chapter)
        rows.extend((f"### {chapter.headings[0].title}",
                     f"Source: `{chapter.source_path}`\n{chapter.text_at(preamble)}"))
        refs.extend((_ref(book, chapter, chapter.headings[0].span), _ref(book, chapter, preamble)))
    return BookView("\n\n".join(rows), tuple(refs), False)


def closed_roads_section(
    repo_root: Path,
    read_bytes: Callable[[str], bytes] | None = None,
) -> str:
    """The authored ``10.2 Closed roads`` section from ``docs/architecture/10-key-invariants.md``."""
    try:
        book = load_reference_book(repo_root, "architecture", read_bytes)
        return read_book_section(book, "10.2 Closed roads").text.strip()
    except (OSError, ValueError, KeyError):
        return ""


def load_domain_references(
    repo_root: Path,
    read_bytes: Callable[[str], bytes] | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, ...]]]:
    """``(domains, modules, references)`` from ``ouroboros/domains.toml``; empty dicts on failure."""
    try:  # Python 3.11+
        import tomllib
    except ImportError:  # pragma: no cover - 3.10 venvs ship tomli
        import tomli as tomllib

    reader = read_bytes if read_bytes is not None else lambda path: (Path(repo_root) / path).read_bytes()
    try:
        data = tomllib.loads(reader("ouroboros/domains.toml").decode("utf-8"))
    except Exception:
        return {}, {}, {}
    domains = {str(k): str(v) for k, v in (data.get("domains") or {}).items()}
    modules = {str(k): str(v) for k, v in (data.get("modules") or {}).items()}
    references = {
        str(k): tuple(str(v) for v in vals)
        for k, vals in (data.get("references") or {}).items()
        if isinstance(vals, list)
    }
    return domains, modules, references


def resolve_body_references(
    repo_root: Path,
    paths: Iterable[str],
    *,
    read_bytes: Callable[[str], bytes] | None = None,
    include_mentions: bool = False,
) -> dict[str, object]:
    """Single deterministic lookup: repo paths -> domains + owning manuals.

    Combines ``ouroboros/domains.toml`` (exact ``[modules]`` -> ``[references]`` and longest
    non-Python prefix match in ``[references]``) and optionally chapter text mentions in
    Architecture and Development books. Pure data lookup, never a heuristic router.
    """
    import re

    root = Path(repo_root)
    domains, modules, references = load_domain_references(root, read_bytes)
    prefix_keys = sorted((k for k in references if k not in domains), key=len, reverse=True)
    by_path: dict[str, dict[str, object]] = {}
    ordered_manuals: list[str] = []
    matched_domains: set[str] = set()
    norm_paths: list[str] = []

    for raw in paths:
        norm = str(raw or "").strip().replace("\\", "/").lstrip("./")
        if not norm or norm in by_path:
            continue
        norm_paths.append(norm)
        dom = modules.get(norm, "")
        manuals: list[str] = []
        if dom:
            matched_domains.add(dom)
            manuals.extend(references.get(dom, ()))
        else:
            for prefix in prefix_keys:
                if norm == prefix.rstrip("/") or norm.startswith(prefix):
                    manuals.extend(references.get(prefix, ()))
                    break
        deduped = tuple(dict.fromkeys(manuals))
        by_path[norm] = {
            "domain": dom,
            "domain_title": domains.get(dom, ""),
            "manuals": deduped,
        }
        for manual in deduped:
            if manual not in ordered_manuals:
                ordered_manuals.append(manual)

    if include_mentions and norm_paths:
        tokens = []
        for norm in norm_paths:
            base = norm.rsplit("/", 1)[-1]
            tokens.append((norm,) if base == norm else (norm, base))
        for book_id in ("architecture", "development"):
            try:
                book = load_reference_book(root, book_id, read_bytes)
            except (OSError, ValueError, KeyError):
                continue
            for chapter in book.chapters:
                if chapter.source_path in ordered_manuals:
                    continue
                for group in tokens:
                    if any(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", chapter.text)
                           for name in group):
                        ordered_manuals.append(chapter.source_path)
                        break

    return {
        "by_path": by_path,
        "manuals": tuple(ordered_manuals),
        "domains": tuple(sorted(matched_domains)),
    }


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


# --------------------------------------------------------------------------- book balance
# A measurement, never a gate (BIBLE P3 c5). Two-tier size rule (BIBLE P7, Fork 2=A):
# - Tier A (always-resident texts): BIBLE.md, prompts/*.md (as one text), and the rendered
#   Body Map + Reference Atlas obey strict pairwise `<= +0 B` vs contribution base.
# - Tier B (pulled engineering manuals): book chapters have no whole-book shrink-only cap;
#   each chapter stays within one default `read_file` (2,000 lines / 80,000 chars) and splits
#   by subject rather than deleting rationale. Standalone >80 KB specs report factual deltas.

BOOK_GROWTH_RULE = (
    "Always-resident texts — BIBLE.md, the prompts (prompts/*.md as one text), and the rendered "
    "Body Map + Reference Atlas — must end no larger than at the contribution base; added text is "
    "paid in the same text by a statement that leaves, never by a lost fact (DEVELOPMENT \"Paying "
    "down a size cap\"). Pulled book chapters have no whole-book shrink-only cap: keep each chapter "
    f"within one default read ({BOOK_CHAPTER_MAX_LINES:,} lines / {BOOK_CHAPTER_MAX_CHARS:,} chars) "
    "and split by subject with rationale intact. Manual/tag CI without a base skips comparison. "
    "The owner's book-growth label can approve an exception; local commits are not blocked."
)
BOOK_CHAPTER_BAND_RULE = (
    f"Every pulled book chapter must fit inside one default read ({BOOK_CHAPTER_MAX_LINES:,} lines "
    f"and {BOOK_CHAPTER_MAX_CHARS:,} chars); split by subject with rationale intact rather than "
    "deleting reasons (BIBLE P7). The owner's book-growth label can approve an exception."
)

# The constitution, prompts, and rendered Body Map stay always-resident (Tier A).
RULED_TEXTS = {"constitution": "BIBLE.md", "prompts": "prompts"}
_TEXT_NAMES = {"constitution": "BIBLE.md", "prompts": "Prompts", "body_map": "Body Map + Atlas"}


def ruled_text_of(path: str) -> str:
    """``"constitution"``, ``"prompts"`` or ``""`` for a repository path; pure path shape."""
    normalized = str(path or "").replace("\\", "/").lstrip("./")
    if normalized == RULED_TEXTS["constitution"]:
        return "constitution"
    folder, _, name = normalized.rpartition("/")
    return "prompts" if folder == RULED_TEXTS["prompts"] and name.endswith(".md") else ""


def standalone_reference_of(path: str) -> str:
    """Canonical standalone >80 KB reference path if ``path`` names one, else ``""``."""
    normalized = str(path or "").replace("\\", "/").lstrip("./")
    return normalized if normalized in STANDALONE_REFERENCE_DOCS else ""


def text_name(text_id: str) -> str:
    """How a book, resident text or standalone manual is named in a balance line."""
    if text_id in _TEXT_NAMES:
        return _TEXT_NAMES[text_id]
    if text_id in STANDALONE_REFERENCE_DOCS:
        return text_id
    return f"{text_id.title()} book"


@dataclass(frozen=True)
class BookBalance:
    book_id: str
    size: int                              # UTF-8 bytes in the measured source (worktree or staged index)
    vs_head: int | None                    # net change against HEAD; None when HEAD has no such text
    vs_upstream: int | None                # contribution delta against the official target's merge-base
    upstream: str                          # local official development ref, "" when unavailable
    changed: tuple[tuple[str, int], ...]   # sources whose bytes differ from HEAD, with their delta
    merge_base: str = ""                   # exact measured commit, never the feature's tracking tip
    tier: str = "resident"                 # "resident" (Tier A <= +0 B) | "manual" (Tier B factual + band)
    chapter_overages: tuple[tuple[str, int, int], ...] = ()  # (path, lines, chars) over one-read band

    @property
    def owed(self) -> int:
        """Known growth owed under Tier A; Tier B manuals carry no whole-book shrink-only debt."""
        if self.tier != "resident":
            return 0
        delta = self.vs_upstream if self.vs_upstream is not None else self.vs_head
        return max(0, delta or 0)


_OFFLINE_GIT = {"GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": ""}


def _git_bytes(root: Path, *args: str) -> bytes | None:
    """stdout of one git read, ``None`` when it fails; a measurement fetches nothing.

    A partial clone fetches an object it lacks once a read needs it. Here that read fails instead
    and its measure stays unknown: Git 2.44+ starts no fetch (``GIT_NO_LAZY_FETCH``), and the fetch
    of an older Git stops before it opens a transport, because the empty ``GIT_ALLOW_PROTOCOL`` allow-list denies every protocol.
    """
    import os
    import subprocess

    try:
        done = subprocess.run(["git", *args], cwd=str(root), capture_output=True, timeout=30,
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


def _source_reader(root: Path, ref: str = "", *, source: str = "worktree") -> Callable[[str], bytes]:
    """Byte reader for a commit ``ref``, the Git ``"index"``, or the ``"worktree"``."""
    root = Path(root)
    if ref:
        def _at_ref(path: str) -> bytes:
            raw = _git_bytes(root, "show", f"{ref}:{path}")
            if raw is None:
                raise FileNotFoundError(f"{ref}:{path}")
            return raw
        return _at_ref
    if source == "index":
        def _at_index(path: str) -> bytes:
            raw = _git_bytes(root, "show", f":{path}")
            if raw is None:
                raise FileNotFoundError(f":{path}")
            return raw
        return _at_index
    return lambda path: (root / path).read_bytes().replace(b"\r\n", b"\n")


def book_source_sizes(
    root: Path, book_id: str, ref: str = "", *, source: str = "worktree",
) -> dict[str, int] | None:
    """UTF-8 bytes of each book source, entrypoint first, in the working tree, index or at ``ref``.

    Only the entrypoint is parsed; chapter bytes are sizes, so a measurement costs one
    `git show` plus one `git ls-tree` per ref. ``None`` when the book is unreadable there.
    """
    root, entry = Path(root), BOOK_ENTRYPOINTS[book_id]
    try:
        if ref:
            raw = _git_bytes(root, "show", f"{ref}:{entry}")
            listing = _git_bytes(root, "ls-tree", "-r", "-l", "-z", "--full-tree", ref, "--", f"docs/{book_id}")
            if raw is None or listing is None:
                return None
            blobs = {}
            for row in listing.decode("utf-8").split("\0"):
                meta, _, path = row.partition("\t")
                if path and meta.split()[1] == "blob":
                    blobs[path] = int(meta.split()[3])
            size_of = blobs.__getitem__
        elif source == "index":
            raw = _git_bytes(root, "show", f":{entry}")
            if raw is None:
                return None

            def size_of(path: str) -> int:
                blob = _git_bytes(root, "show", f":{path}")
                if blob is None:
                    raise FileNotFoundError(f":{path}")
                return len(blob)
        else:
            raw = (root / entry).read_bytes()
            size_of = lambda path: (root / path).stat().st_size  # noqa: E731
        members = _member_paths(parse_markdown_source(raw, entry), book_id) or ()
        return {entry: len(raw), **{path: size_of(path) for path in members}}
    except (OSError, ValueError, KeyError, IndexError, UnicodeDecodeError):
        return None


def composed_size(sizes: dict[str, int]) -> int:
    """``len(compose_book(book).encode())`` from source sizes: sources joined by one blank line."""
    return sum(sizes.values()) + 2 * (len(sizes) - 1)


def body_map_bytes(
    root: Path, ref: str = "", *, source: str = "worktree", require_atlas_at_ref: bool = True,
) -> int | None:
    """UTF-8 bytes of the rendered resident Body Map + Reference Atlas; ``None`` when absent.

    When ``ref`` is supplied and ``require_atlas_at_ref`` is True, a historical commit that
    predates the Body Map + Atlas layout (no ``## Reference Atlas`` section in
    ``docs/ARCHITECTURE.md``) returns ``None`` so the migration commit itself has no legacy
    half-map base to compare against while every post-migration change is measured strictly.
    """
    try:
        book = load_reference_book(Path(root), "architecture", _source_reader(Path(root), ref, source=source))
    except (OSError, ValueError, KeyError):
        return None
    if not has_body_map_atlas(book):
        return None if (ref and require_atlas_at_ref) or not ref else len(body_map_book(book).text.encode("utf-8"))
    return len(body_map_book(book).text.encode("utf-8"))


def chapter_band_overages(
    root: Path, book_id: str, ref: str = "", *, source: str = "worktree",
) -> tuple[tuple[str, int, int], ...]:
    """Chapters of ``book_id`` exceeding one default read (``BOOK_CHAPTER_MAX_LINES`` or ``BOOK_CHAPTER_MAX_CHARS``)."""
    try:
        book = load_reference_book(Path(root), book_id, _source_reader(Path(root), ref, source=source))
    except (OSError, ValueError, KeyError):
        return ()
    overages: list[tuple[str, int, int]] = []
    for chapter in book.chapters:
        lines = len(chapter.text.splitlines())
        chars = len(chapter.text)
        if lines > BOOK_CHAPTER_MAX_LINES or chars > BOOK_CHAPTER_MAX_CHARS:
            overages.append((chapter.source_path, lines, chars))
    return tuple(overages)


def standalone_doc_size(
    root: Path, relpath: str, ref: str = "", *, source: str = "worktree",
) -> dict[str, int] | None:
    """UTF-8 bytes of one standalone reference document; ``None`` when absent."""
    try:
        raw = _source_reader(Path(root), ref, source=source)(relpath)
        return {relpath: len(raw)}
    except (OSError, ValueError):
        return None


def ruled_text_sizes(
    root: Path, text_id: str, ref: str = "", *, source: str = "worktree",
) -> dict[str, int] | None:
    """UTF-8 bytes of each file of a ruled text, in the working tree, index or at ``ref``; ``None`` when absent."""
    root, where = Path(root), RULED_TEXTS[text_id]
    try:
        if ref:
            listing = _git_bytes(root, "ls-tree", "-l", "-z", "--full-tree", ref, "--",
                                 where if text_id == "constitution" else f"{where}/")
            rows = [row.partition("\t") for row in (listing or b"").decode("utf-8", "replace").split("\0")]
            sizes = {path: int(meta.split()[3]) for meta, _, path in rows
                     if path and meta.split()[1] == "blob" and ruled_text_of(path) == text_id}
        elif source == "index":
            if text_id == "constitution":
                blob = _git_bytes(root, "show", f":{where}")
                sizes = {where: len(blob)} if blob is not None else {}
            else:
                listing = _git_bytes(root, "ls-files", "-z", "--cached", "--", f"{where}/")
                paths = [p for p in (listing or b"").decode("utf-8", "replace").split("\0")
                         if p and ruled_text_of(p) == text_id]
                sizes = {}
                for path in sorted(paths):
                    blob = _git_bytes(root, "show", f":{path}")
                    if blob is not None:
                        sizes[path] = len(blob)
        else:  # a Windows checkout may hold CRLF for these unpinned files: count the committed LF form
            files = [root / where] if text_id == "constitution" else sorted((root / where).glob("*.md"))
            sizes = {file.relative_to(root).as_posix(): len(file.read_bytes().replace(b"\r\n", b"\n"))
                     for file in files if file.is_file()}
    except (OSError, ValueError, IndexError):  # a file gone mid-read: unmeasured, like an unreadable book
        return None
    return sizes or None


def ruled_text_paths(root: Path) -> list[str]:
    """The working-tree files of every ruled text, for a caller that measures everything."""
    return [path for text_id in RULED_TEXTS for path in (ruled_text_sizes(root, text_id) or {})]


def book_balances(
    root: Path, paths: Iterable[str] | None = None, *, source: str = "worktree",
) -> list[BookBalance]:
    """The balance of every touched book, resident text or standalone manual (all when ``None``); never raises."""
    paths = None if paths is None else list(paths)
    wanted = list(BOOK_ENTRYPOINTS) if paths is None else sorted(
        {book_id for book_id, entry in BOOK_ENTRYPOINTS.items()
         for path in paths if book_entrypoint_for(path) == entry})
    texts = list(RULED_TEXTS) if paths is None else sorted({ruled_text_of(path) for path in paths} - {""})
    standalones = ([] if paths is None else sorted({standalone_reference_of(path) for path in paths} - {""}))
    if not wanted and not texts and not standalones:
        return []
    upstream, base = contribution_base(root)
    measures: list[tuple[str, str, Callable, Callable]] = [
        (book_id, "manual",
         lambda r, tid, ref="", src=source: book_source_sizes(r, tid, ref, source=src),
         composed_size)
        for book_id in wanted
    ]
    measures += [
        (text_id, "resident",
         lambda r, tid, ref="", src=source: ruled_text_sizes(r, tid, ref, source=src),
         lambda sizes: sum(sizes.values()))
        for text_id in texts
    ]
    measures += [
        (rel, "manual",
         lambda r, tid, ref="", src=source: standalone_doc_size(r, tid, ref, source=src),
         lambda sizes: sum(sizes.values()))
        for rel in standalones
    ]
    balances: list[BookBalance] = []
    for text_id, tier, sizes_at, total in measures:
        now = sizes_at(root, text_id)
        if now is None:
            continue
        head = sizes_at(root, text_id, "HEAD")
        at_base = sizes_at(root, text_id, base) if base else None
        changed = tuple((path, now.get(path, 0) - (head or {}).get(path, 0))
                        for path in dict.fromkeys([*now, *(head or {})])
                        if now.get(path, 0) != (head or {}).get(path, 0))
        overages = chapter_band_overages(root, text_id, source=source) if text_id in BOOK_ENTRYPOINTS else ()
        balances.append(BookBalance(
            text_id, total(now),
            total(now) - total(head) if head is not None else None,
            total(now) - total(at_base) if at_base is not None else None,
            upstream, changed, base, tier=tier, chapter_overages=overages))
    if "architecture" in wanted:
        bm_now = body_map_bytes(root, source=source)
        if bm_now is not None:
            bm_head = body_map_bytes(root, "HEAD")
            bm_base = body_map_bytes(root, base) if base else None
            balances.append(BookBalance(
                "body_map", bm_now,
                bm_now - bm_head if bm_head is not None else None,
                bm_now - bm_base if bm_base is not None else None,
                upstream, (), base, tier="resident"))
    return balances


def _signed(value: int | None) -> str:
    return "n/a" if value is None else f"{value:+,} B"


def render_book_balance(balance: BookBalance) -> str:
    """One line: size, HEAD delta, contribution delta and its actual cached base."""
    line = f"{text_name(balance.book_id)} {balance.size:,} B: {_signed(balance.vs_head)} vs HEAD"
    if balance.upstream:
        line += f", {_signed(balance.vs_upstream)} vs {balance.upstream} (merge-base {balance.merge_base or 'unavailable'})"
    if balance.vs_upstream is None:
        line += "; contribution unknown"
    if balance.changed:
        line += " (" + ", ".join(f"{Path(path).stem} {delta:+,}" for path, delta in balance.changed) + ")"
    if balance.chapter_overages:
        ov = ", ".join(f"{Path(p).name} {lines:,}L/{chars:,}c" for p, lines, chars in balance.chapter_overages)
        line += f" [OVER one-read band: {ov}]"
    return line


def book_balance_note(root: Path, paths: Iterable[str], *, source: str = "worktree") -> str:
    """The footer for a write to Ouroboros's own body; ``""`` when no tracked governance text was written.

    Callers decide that ``root`` is the body (the system repository or its bound candidate):
    a user's project with a ``docs/architecture`` folder never receives this note.
    """
    touched = [path for path in paths
               if book_path_role(path) or ruled_text_of(path) or standalone_reference_of(path)]
    if not touched:
        return ""
    balances = book_balances(root, touched, source=source)
    wanted = {book_entrypoint_for(path) for path in touched}
    available = {balance.book_id for balance in balances}
    unavailable = [f"{text_name(book_id)} balance unavailable"
                   for book_id, entry in BOOK_ENTRYPOINTS.items()
                   if entry in wanted and book_id not in available]
    unavailable += [f"{text_name(text_id)} balance unavailable"
                    for text_id in sorted({ruled_text_of(path) for path in touched} - {""} - available)]
    lines = [render_book_balance(balance) for balance in balances] + unavailable
    rules: list[str] = []
    if any(balance.owed for balance in balances):
        rules.append(BOOK_GROWTH_RULE)
    if any(balance.chapter_overages for balance in balances):
        rules.append(BOOK_CHAPTER_BAND_RULE)
    if rules:
        return "ℹ️ Text growth:\n" + "\n".join(lines) + "\n" + "\n".join(rules)
    measured = not unavailable and all(balance.vs_upstream is not None for balance in balances)
    return "ℹ️ Text growth: " + "; ".join(lines) + ("; nothing owed." if measured else ".")
