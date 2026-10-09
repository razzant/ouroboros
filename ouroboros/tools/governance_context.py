"""The ONE decision about which governance documents a reviewer receives in full.

Owner decision (2026-09-17, batch 2 question 1 = A): every review surface —
triad, scope, advisory, deep self-review — asks this module which rules it must
carry inline and which it reaches through a navigation map. Three tiers:

1. **Rules, always inline.** The surface's applicable ``docs/CHECKLISTS.md``
   section (supplied by the caller, which owns its own section name), the
   shared section every reviewer of this repository's code applies
   (:data:`SHARED_CHECKLIST_SECTION`, loaded here), ``BIBLE.md`` whole and
   ``docs/CHECKLISTS_ARCHIVE.md`` whole. This tier is byte-stable across
   commits, so the caller puts it in its cache-marked prefix.
2. **Rules by change class, within a budget share.** ``docs/DESIGN.md`` when the
   change touches ``web/``; the review-and-commit protocol chapter always; and
   every DEVELOPMENT chapter whose text mentions a touched file NAME. Bounded by
   :data:`~ouroboros.runtime_limits.REVIEW_GOVERNANCE_INLINE_SHARE` of the
   reviewer's usable window, largest-relevance-first.
3. **Map, never whole.** ``docs/ARCHITECTURE.md`` arrives as book navigation for
   every delivery. A PACKET reviewer (a row with no tools of its own)
   additionally receives the top-level sections whose text mentions a touched
   file name, from the same budget; a RETRIEVING row gets the navigation and
   reads what it needs with its own ``read_file``.

Every document that is not inlined is NAMED in the returned manifest and in the
navigation text, never silently dropped (BIBLE P1), and the manifest is the
disclosure record the durable review evidence carries.

The tiers above are the **body layer**: they govern a change to Ouroboros's own
body. A change review of another repository runs the **core layer**
(``layer="core"``; `review_body_fact.layer_for` decides): tier 1 is the
supplied universal checklist alone, the body's constitution, handbook, design
system, architecture map and standing disclosures are recorded
``not_applicable`` (named, never silently dropped), and the navigation indexes
the SUBJECT's own documents (``subject_root``) plus its required-source
manifest. The shared-contract section is a rule of this repository's CODE and
travels with the body layer only.

Choosing which reference chapter to inline from an exact file-name mention is
context ASSEMBLY, not behaviour selection: no verdict, routing decision or
reviewer judgment depends on the match, and the reviewer may read anything else
it wants. No LLM call; one tree plus one touched-path list gives one result.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ouroboros.runtime_limits import REVIEW_GOVERNANCE_INLINE_SHARE
from ouroboros.utils import estimate_tokens

BIBLE_PATH = "BIBLE.md"
CHECKLISTS_PATH = "docs/CHECKLISTS.md"
# One home in CHECKLISTS for a rule the triad, scope, advisory and deep review
# all apply whatever their own section; plan and skill review never receive it.
SHARED_CHECKLIST_SECTION = "Shared Contract Ownership"
CHECKLISTS_ARCHIVE_PATH = "docs/CHECKLISTS_ARCHIVE.md"
DESIGN_PATH = "docs/DESIGN.md"
DEVELOPMENT_BOOK_ID = "development"
ARCHITECTURE_BOOK_ID = "architecture"
# Always relevant: it is the protocol the reviewer is executing right now.
REVIEW_PROTOCOL_CHAPTER = "docs/development/05-review-and-commit-protocol.md"
# The one change class that decides a whole document: DESIGN governs web/ work.
DESIGN_CHANGE_CLASS_PREFIX = "web/"

PACKET_DELIVERY = "packet"
RETRIEVING_DELIVERY = "retrieving"
_CARRIED_REASON = "carried by this surface's own delivery"

CORE_LAYER = "core"
BODY_LAYER = "body"
GOVERNANCE_LAYERS = (CORE_LAYER, BODY_LAYER)
# A body document in a core-layer manifest: named, not delivered, by rule.
NOT_APPLICABLE_DISPOSITION = "not_applicable"
_CORE_NOT_APPLICABLE = "core layer: governs Ouroboros's own body, not this subject"
# Bounds of the core layer's index of the subject's own documents (root
# Markdown files and docs/**): listed by name and size, the first few mapped.
SUBJECT_DOCS_MAX_LISTED = 24
SUBJECT_DOCS_MAX_MAPPED = 6
SUBJECT_DOC_MAP_MAX_BYTES = 120_000


@dataclass(frozen=True)
class GovernanceContext:
    """What one reviewer receives, and the record of what it did not receive.

    ``stable_inline``/``selected_inline`` render the same sections for the
    caller's two prompt regions: tier 1 belongs in the cache-marked prefix, tier
    2/3 in the change-relative tail. ``inline_whole_documents`` maps a path to
    the exact text inlined here, so a caller that also packs touched files
    suppresses the duplicate instead of sending one document twice."""

    inline_sections: list = field(default_factory=list)   # [(title, text)] in prompt order
    navigation: str = ""
    manifest: list = field(default_factory=list)          # [{path, tier, disposition, chars, reason}]
    tokens_estimate: int = 0
    stable_inline: str = ""
    selected_inline: str = ""
    inline_whole_documents: dict = field(default_factory=dict)
    layer: str = BODY_LAYER


def _normalize(path: Any) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def _touched_names(touched_paths: Optional[Iterable[Any]]) -> tuple[tuple[str, ...], ...]:
    """One name group per touched file: its repo-relative path and its basename.

    Directory prefixes are deliberately absent — ``web/`` or ``ouroboros/``
    would match nearly every chapter and so select nothing."""
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for raw in touched_paths or ():
        rel = _normalize(raw)
        if not rel or rel.endswith("/") or rel in seen:
            continue
        seen.add(rel)
        base = rel.rsplit("/", 1)[-1]
        groups.append((rel,) if base == rel else (rel, base))
    return tuple(groups)


def _mentions(text: str, groups: Iterable[tuple[str, ...]]) -> tuple[int, int]:
    """``(touched files mentioned, mentions)`` of those files in ``text``.

    A name matches only as its own token: ``git.py`` inside ``review_git.py`` is
    not a mention, while ``ouroboros/git.py`` is one (a path separator may
    precede a basename). The largest per-spelling count wins, so a full path
    beside its own basename is one mention, not two."""
    files = mentions = 0
    for group in groups:
        found = max(
            len(re.findall(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", text))
            for name in group)
        if found:
            files += 1
            mentions += found
    return files, mentions


def _carried_reasons(already_inline: Any) -> dict:
    """``{path: reason}`` for the documents the CALLER's delivery already carries.

    A plain iterable of paths takes the default wording; a mapping lets the
    caller state the existing inline location, such as a constitutional head.
    A pointer or an instruction to read later is not inline delivery."""
    if isinstance(already_inline, Mapping):
        return {_normalize(path): str(reason or _CARRIED_REASON) or _CARRIED_REASON
                for path, reason in already_inline.items()}
    return {_normalize(path): _CARRIED_REASON for path in already_inline or ()}


def _row(path: str, tier: int, disposition: str, chars: int, reason: str) -> dict:
    return {"path": path, "tier": tier, "disposition": disposition,
            "chars": int(chars), "reason": reason}


def _render(title: str, text: str) -> str:
    return f"## {title}\n\n{text}"


def _join(parts: Iterable[str]) -> str:
    return "\n\n".join(part for part in parts if str(part or "").strip())


def _load_book(repo_dir: Path, book_id: str):
    """``(book, "")`` or ``(None, reason)`` — an unreadable book is disclosed,
    never rendered as a delivered one."""
    from ouroboros.reference_books import load_reference_book

    try:
        return load_reference_book(repo_dir, book_id), ""
    except (OSError, ValueError, KeyError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _book_sources(book) -> tuple:
    """A chaptered book's chapters; a legacy revision has none to select from."""
    return () if book is None or book.legacy else tuple(book.chapters)


def _top_sections(source) -> tuple:
    """The chapter's own top-level sections, heading plus complete subtree.

    A chapter keeps its relocated body at the heading level that body already
    had, so the monolith's ``##`` entries sit at ``###`` in a split chapter and
    at ``##`` in one authored after the split: the shallowest level below the
    chapter's H1 selects the same sections in both shapes. A chapter with no
    subsections contributes none and stays addressable through the navigation."""
    levels = [heading.level for heading in source.headings if heading.level > 1]
    if not levels:
        return ()
    top = min(levels)
    return tuple(heading for heading in source.headings if heading.level == top)


def _book_navigation(repo_dir: Path, book_id: str, book, load_doc, *, instructions: bool = True) -> str:
    from ouroboros.context_layout import book_navigation, generate_doc_nav_map
    from ouroboros.reference_books import BOOK_ENTRYPOINTS

    entrypoint = BOOK_ENTRYPOINTS[book_id]
    if book is not None:
        return book_navigation(book, instructions=instructions)
    # No book: map whatever the entrypoint reader returns, so the document stays
    # named and addressable even when its membership cannot be assembled.
    text = load_doc(repo_dir, entrypoint, on_missing="explicit")
    return generate_doc_nav_map(
        text, title=entrypoint.rsplit("/", 1)[-1], rel_path=entrypoint, instructions=instructions)


def _subject_documents(subject_root: Path) -> list:
    """The subject's own Markdown documents: root files (README, CONTRIBUTING
    first), then ``docs/**``. Two bounded roots, never the whole tree."""
    def _rank(path: Path):
        name = path.name.lower()
        return (0 if name.startswith("readme") else 1 if name.startswith("contributing") else 2, name)

    try:
        root_docs = sorted((p for p in subject_root.glob("*.md") if p.is_file()), key=_rank)
        docs_dir = subject_root / "docs"
        nested = sorted(p for p in docs_dir.rglob("*.md") if p.is_file()) if docs_dir.is_dir() else []
    except OSError:
        return []
    return root_docs + nested


def _subject_navigation(subject_root: Any, *, packet: bool) -> tuple[str, list]:
    """The core layer's index of the SUBJECT's documents, with its manifest rows."""
    from ouroboros.context_layout import generate_doc_nav_map

    rows: list[dict] = []
    if subject_root is None:
        return ("### Subject documents\n\nNo subject root was supplied: the subject's own documents "
                "were not indexed for this review" + (
                    " and were not delivered to this packet." if packet
                    else "; read them with your own tools."), rows)
    root = Path(subject_root)
    docs = _subject_documents(root)
    if not docs:
        return (f"### Subject documents\n\nThe subject keeps no Markdown document at its root or "
                f"under `docs/` (root: `{root}`).", rows)
    lines = [f"### Subject documents (root: `{root}`)", ""]
    for index, path in enumerate(docs[:SUBJECT_DOCS_MAX_LISTED]):
        rel = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        rows.append(_row(f"subject:{rel}", 3, "navigation", size,
                         "the subject's own document: evidence of what it promises, not a rule"))
        if index < SUBJECT_DOCS_MAX_MAPPED and 0 < size <= SUBJECT_DOC_MAP_MAX_BYTES:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if text.strip():
                lines.append(generate_doc_nav_map(text, title=rel, rel_path=rel, instructions=False))
                lines.append("")
                continue
        lines.append(f"- `{rel}` ({size:,} bytes)")
    if len(docs) > SUBJECT_DOCS_MAX_LISTED:
        lines.append(f"- … {len(docs) - SUBJECT_DOCS_MAX_LISTED} more Markdown document(s) under "
                     "the same roots, not listed")
    return "\n".join(lines).rstrip(), rows


def _core_layer_context(*, surface: str, touched_paths: Optional[Iterable[Any]], delivery: str,
                        checklist_section_text: str, subject_root: Any) -> GovernanceContext:
    """The core layer: the subject is not the Ouroboros body.

    Tier 1 is the supplied universal checklist alone; the body's constitution,
    handbook, design system, architecture map and standing disclosures are
    recorded ``not_applicable`` — named, never silently dropped (P1). The
    navigation indexes the SUBJECT's own documents and its required-source
    manifest (empty by rule: no body inventory applies to another repository)."""
    from ouroboros.tools.scope_required_sources import render_required_sources, scope_required_sources

    packet = str(delivery or PACKET_DELIVERY) == PACKET_DELIVERY
    checklist_text = str(checklist_section_text or "")
    rows: list[dict] = []
    if checklist_text.strip():
        rows.append(_row(CHECKLISTS_PATH, 1, "inline", len(checklist_text),
                         f"applicable {surface} checklist section (core layer), supplied by the surface"))
    else:
        rows.append(_row(CHECKLISTS_PATH, 1, "navigation", 0,
                         "this surface supplies no checklist section"))
    for path, tier in ((f"{CHECKLISTS_PATH}#{SHARED_CHECKLIST_SECTION}", 1), (BIBLE_PATH, 1),
                       (CHECKLISTS_ARCHIVE_PATH, 1), ("docs/DEVELOPMENT.md", 2), (DESIGN_PATH, 2),
                       ("docs/ARCHITECTURE.md", 3)):
        rows.append(_row(path, tier, NOT_APPLICABLE_DISPOSITION, 0, _CORE_NOT_APPLICABLE))
    subject_nav, subject_rows = _subject_navigation(subject_root, packet=packet)
    rows.extend(subject_rows)
    pairs = [("M", _normalize(path)) for path in (touched_paths or ()) if _normalize(path)]
    # The scope brief states its own required-source manifest (with its identity)
    # in its tail; every other surface receives the layer's statement here.
    required = (scope_required_sources(subject_root, pairs, layer=CORE_LAYER)
                if subject_root is not None and surface != "scope" else [])
    rule_set = ("The checklist section inlined above is the whole rule set for this review"
                if checklist_text.strip() else
                f"NO section of `{CHECKLISTS_PATH}` is inlined for this review (this surface "
                "supplies none); the universal change-review rules are the whole rule set")
    navigation = _join([
        "## Governance navigation (core layer)",
        f"{rule_set}: the subject "
        "is not the Ouroboros body, so Ouroboros's constitution, engineering handbook, design "
        "system, architecture map and standing disclosures do not govern it and are not "
        "delivered. The subject's own documents are evidence of what it promises, not law: use "
        "them to judge behavioural documentation, public contracts and release metadata." + (
            " This row has no repository tools: the documents named below were NOT delivered; "
            "state any resulting uncertainty rather than claim to have read them." if packet else
            " Everything named below is complete on disk in the subject repository; read any "
            "range you need with your own tools."),
        subject_nav,
        render_required_sources(required, layer=CORE_LAYER) if surface != "scope" else "",
    ])
    return GovernanceContext(navigation=navigation, manifest=rows,
                             tokens_estimate=estimate_tokens(navigation), layer=CORE_LAYER)


def governance_context(
    repo_dir: Any,
    *,
    surface: str,
    touched_paths: Optional[Iterable[Any]] = None,
    usable_window_tokens: int = 0,
    delivery: str = PACKET_DELIVERY,
    checklist_section_text: str = "",
    already_inline: Any = (),
    layer: str = BODY_LAYER,
    subject_root: Any = None,
) -> GovernanceContext:
    """Tier the governance corpus for ONE reviewer of ONE change.

    ``repo_dir`` is the governance root — always the installed system
    repository, whose rules execute. ``surface`` labels the review surface in
    the disclosure record. ``checklist_section_text`` is the tier-1 section
    that surface already loaded; a surface supplying none has
    ``docs/CHECKLISTS.md`` recorded as read on demand, never as an inlined
    section of zero characters. ``already_inline`` names what the caller's OWN
    delivery carries in full (:func:`_carried_reasons` states the mechanism).
    ``layer`` is the ONE external switch: ``body`` (the subject is Ouroboros's
    body) runs the three tiers; ``core`` (any other subject) runs
    :func:`_core_layer_context` over ``subject_root``."""
    from ouroboros.tools.review_helpers import load_checklist_section, load_governance_doc

    if layer not in GOVERNANCE_LAYERS:
        raise ValueError(f"unknown governance layer {layer!r}; expected one of {GOVERNANCE_LAYERS}")
    if layer == CORE_LAYER:
        return _core_layer_context(surface=surface, touched_paths=touched_paths, delivery=delivery,
                                   checklist_section_text=checklist_section_text,
                                   subject_root=subject_root)

    root = Path(repo_dir)
    names = _touched_names(touched_paths)
    carried = _carried_reasons(already_inline)
    touched = {_normalize(path) for path in (touched_paths or ())}
    rows: list[dict] = []
    tier1: list[tuple[str, str]] = []
    selected: list[tuple[str, str]] = []
    inline_texts: dict[str, str] = {}

    # --- tier 1: rules, always inline ------------------------------------
    checklist_text = str(checklist_section_text or "")
    if checklist_text.strip():
        rows.append(_row(CHECKLISTS_PATH, 1, "inline", len(checklist_text),
                         f"applicable {surface} checklist section, supplied by the surface"))
    else:
        rows.append(_row(CHECKLISTS_PATH, 1, "navigation", 0,
                         "this surface supplies no checklist section"))
    shared_path = f"{CHECKLISTS_PATH}#{SHARED_CHECKLIST_SECTION}"
    shared_inline = False
    try:
        # Read where every surface reads its own section: the executing
        # review code's checklist, never the reviewed tree's copy, so a
        # contributor proposal cannot rewrite the rule it is judged by.
        shared = load_checklist_section(SHARED_CHECKLIST_SECTION)
        shared_inline = True
    except (OSError, ValueError) as exc:
        shared = f"[⚠️ OMISSION: {shared_path} could not be loaded: {exc}]"
    tier1.append((shared_path, shared))
    rows.append(_row(shared_path, 1, "inline", len(shared),
                     "tier 1: every reviewer of this repository's code carries it"))
    for path in (BIBLE_PATH, CHECKLISTS_ARCHIVE_PATH):
        if path in carried:
            already = load_governance_doc(root, path, on_missing="silent")
            rows.append(_row(path, 1, "inline", len(already), carried[path]))
            continue
        text = load_governance_doc(root, path, on_missing="explicit")
        tier1.append((path, text))
        inline_texts[path] = text
        rows.append(_row(path, 1, "inline", len(text), "tier 1: every reviewer carries it"))

    # --- the change-class budget: ONE share for tier 2 and tier 3 together,
    # rules by change class drawing first, architecture sections from the rest.
    budget = max(0, int(int(usable_window_tokens or 0) * REVIEW_GOVERNANCE_INLINE_SHARE))
    remaining = budget

    def _admit(path: str, title: str, text: str, tier: int, reason: str) -> bool:
        nonlocal remaining
        cost = estimate_tokens(_render(title, text))
        if cost > remaining:
            rows.append(_row(path, tier, "navigation", len(text),
                             f"{reason}; inline share exhausted "
                             f"({cost} tokens needed, {remaining} of {budget} left)"))
            return False
        remaining -= cost
        selected.append((title, text))
        if "#" not in path:
            inline_texts[path] = text
        rows.append(_row(path, tier, "inline", len(text), reason))
        return True

    # --- tier 2: rules by change class ------------------------------------
    dev_book, dev_problem = _load_book(root, DEVELOPMENT_BOOK_ID)
    dev_chapters = _book_sources(dev_book)
    rows.append(_row("docs/DEVELOPMENT.md", 2, "navigation", 0,
                     "book navigation; its chapters are selected by change class" if dev_chapters
                     else dev_problem or "no chaptered membership: navigation only"))

    candidates: list[tuple[tuple, str, str, str]] = []  # (order, path, title, reason)
    for chapter in dev_chapters:
        path = chapter.source_path
        files, mentions = _mentions(chapter.text, names)
        if path == REVIEW_PROTOCOL_CHAPTER:
            candidates.append(((0, 0, path), path, path, "the review protocol this reviewer executes"))
        elif files:
            candidates.append(((2, -files * 1000 - mentions, path), path, path,
                               f"mentions {files} touched file(s), {mentions} time(s)"))
        else:
            rows.append(_row(path, 2, "navigation", len(chapter.text),
                             "no touched file mentioned"))

    design_touched = any(path.startswith(DESIGN_CHANGE_CLASS_PREFIX) for path in touched)
    design_text = ""
    if design_touched:
        design_text = load_governance_doc(root, DESIGN_PATH, on_missing="explicit")
        candidates.append(((1, 0, DESIGN_PATH), DESIGN_PATH, DESIGN_PATH,
                           f"a touched path is under {DESIGN_CHANGE_CLASS_PREFIX}"))
    else:
        rows.append(_row(DESIGN_PATH, 2, "navigation", 0,
                         f"no touched path under {DESIGN_CHANGE_CLASS_PREFIX}"))

    chapter_text = {chapter.source_path: chapter.text for chapter in dev_chapters}
    chapter_text[DESIGN_PATH] = design_text
    for order, path, title, reason in sorted(candidates):
        _admit(path, title, chapter_text.get(path, ""), 2, reason)

    # --- tier 3: the map, never whole -------------------------------------
    arch_book, arch_problem = _load_book(root, ARCHITECTURE_BOOK_ID)
    arch_chapters = _book_sources(arch_book)
    packet = str(delivery or PACKET_DELIVERY) == PACKET_DELIVERY
    arch_reason = "the map is delivered as navigation, never whole; " + (
        "the sections that name a touched file are selected below" if packet
        else "this reviewer reads sections on demand with its own tools")
    rows.append(_row("docs/ARCHITECTURE.md", 3, "navigation", 0, arch_reason if arch_chapters
                     else arch_problem or "no chaptered membership: navigation only"))
    sections: list[tuple[tuple, str, str, str]] = []
    for chapter in arch_chapters:
        rows.append(_row(chapter.source_path, 3, "navigation", len(chapter.text),
                         "the map is never inlined whole; chapter navigation above"))
        if not packet or not names:
            continue
        for heading in _top_sections(chapter):
            body = chapter.text_at(heading.section)
            files, mentions = _mentions(body, names)
            if not files:
                continue
            path = f"{chapter.source_path}#{heading.title}"
            sections.append(((-files * 1000 - mentions, chapter.source_path, heading.span.start_byte),
                             path, path, f"mentions {files} touched file(s), {mentions} time(s)"))
            chapter_text[path] = body
    for order, path, title, reason in sorted(sections):
        _admit(path, title, chapter_text.get(path, ""), 3, reason)

    # --- navigation -------------------------------------------------------
    inlined = [label for label, present in (
        ("the section that applies to this review", bool(checklist_text.strip())),
        (f"its `{SHARED_CHECKLIST_SECTION}` section", shared_inline)) if present]
    checklist_pointer = (f"{' and '.join(inlined)} {'are' if len(inlined) > 1 else 'is'} inlined above."
                         if inlined else "NO section of it is inlined for this review.")
    pointers = [f"- `{CHECKLISTS_PATH}` — the complete checklist book; {checklist_pointer}"]
    if not design_touched:
        pointers.append(f"- `{DESIGN_PATH}` — the design system; not inlined because no "
                        f"touched path is under `{DESIGN_CHANGE_CLASS_PREFIX}`.")
    navigation = _join([
        "## Governance navigation (index of sources not inlined)" if packet else
        "## Governance navigation (read on demand)",
        ("The supplied rules and selected sections are the governance evidence this packet "
         "delivers. This row has no repository tools: the map below identifies sources and "
         "ranges that were NOT delivered, not additional evidence. State any resulting "
         "uncertainty; do not claim to have read the named sources." if packet else
        "The rules above are the ones this change activates. Everything below is complete "
        "on disk and untruncated: read any range with "
        '`read_file(root="system_repo", path=..., start_line=A, max_lines=N)`. Ranges are '
        "inclusive and a parent heading includes its descendant group. A document named here "
        "and not inlined was NOT dropped — it is one read away, and the review's governance "
        "manifest records the disposition of each one."),
        "\n".join(pointers),
        _book_navigation(root, DEVELOPMENT_BOOK_ID, dev_book, load_governance_doc, instructions=not packet),
        _book_navigation(root, ARCHITECTURE_BOOK_ID, arch_book, load_governance_doc, instructions=not packet),
    ])

    stable_inline = _join(_render(title, text) for title, text in tier1)
    selected_inline = _join(_render(title, text) for title, text in selected)
    return GovernanceContext(
        inline_sections=tier1 + selected,
        navigation=navigation,
        manifest=rows,
        tokens_estimate=estimate_tokens(_join([stable_inline, selected_inline, navigation])),
        stable_inline=stable_inline,
        selected_inline=selected_inline,
        inline_whole_documents=inline_texts,
    )
