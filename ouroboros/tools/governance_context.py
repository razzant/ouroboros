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
   change touches ``web/`` (its design bible alone when the whole book does not fit
   the share), its design bible alone for changed words elsewhere, and its numbered
   principles alone where the share cannot hold the whole bible;
   the review-and-commit protocol chapter always; and
   every DEVELOPMENT chapter whose text mentions a touched file NAME. Bounded by
   :data:`~ouroboros.runtime_limits.REVIEW_GOVERNANCE_INLINE_SHARE` of the
   reviewer's usable window, largest-relevance-first. Whenever DESIGN is not
   inlined, every body reviewer receives its generated heading map.
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

A skill payload runs the **skill layer** (``layer="skill"``, owner 21.10 1A/F1):
tier 1 inline is the Skill Review Checklist section the surface supplies plus
BIBLE whole; tier 2 is required of every skill reviewer — the three architecture
chapters that bind what a skill may touch and the module-size handbook chapter —
plus, by the manifest's own facts (``manifest_facts``: ``type`` and what it
declares), the PluginAPI contract and the extension UI validator; tier 3 is both
books' navigation and the CREATING_SKILLS heading map. A packet seat receives
tiers 1–2 inline and tier 3 as the index of what was NOT delivered; a retrieving
seat receives tier 1 inline, tier 2 as required rows ("one read away") and tier
3 as navigation. Selecting a contract by a manifest fact is context assembly,
never a gate: nothing the reviewer decides depends on the match.

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
_DESIGN_BIBLE_PATH = f"{DESIGN_PATH}#Design bible"
_DESIGN_PRINCIPLES_PATH = f"{_DESIGN_BIBLE_PATH} principles"
_DESIGN_REVIEW_MAP = {
    DESIGN_PATH: "Navigation only: no DESIGN rules are inlined as governance.",
    _DESIGN_BIBLE_PATH: "Inlined above: the whole design bible section; the rest is indexed only.",
    _DESIGN_PRINCIPLES_PATH: "Inlined above: the design bible's opening and numbered principles only.",
}
DEVELOPMENT_BOOK_ID = "development"
ARCHITECTURE_BOOK_ID = "architecture"
# Always relevant: it is the protocol the reviewer is executing right now.
REVIEW_PROTOCOL_CHAPTER = "docs/development/05-review-and-commit-protocol.md"
# Always relevant for body changes: cross-cutting invariants and Closed roads (matching SKILL_LAYER).
BODY_INVARIANTS_CHAPTER = "docs/architecture/10-key-invariants.md"

PACKET_DELIVERY = "packet"
RETRIEVING_DELIVERY = "retrieving"
_CARRIED_REASON = "carried by this surface's own delivery"

CORE_LAYER = "core"
BODY_LAYER = "body"
SKILL_LAYER = "skill"
GOVERNANCE_LAYERS = (CORE_LAYER, BODY_LAYER, SKILL_LAYER)
# The skill layer's tier 2 (owner 21.10 F1): read in full by every skill reviewer.
SKILL_REQUIRED_CHAPTERS = (
    "docs/architecture/10-key-invariants.md",
    "docs/architecture/12-host-service-companions-and-chat-ids.md",
    "docs/architecture/13-external-skills-layer.md",
    "docs/development/03-module-size-and-complexity.md",
)
SKILL_PLUGIN_API_PATH = "ouroboros/contracts/plugin_api.py"
SKILL_UI_VALIDATION_PATH = "ouroboros/extension_ui_validation.py"
CREATING_SKILLS_PATH = "docs/CREATING_SKILLS.md"
# Manifest facts that make a host contract a required read: the PluginAPI contract for an
# extension or any manifest that declares a host-mediated capability; the UI validator for
# an extension (widgets may be registered dynamically from plugin.py) or a declared UI surface.
SKILL_PLUGIN_API_FACTS = ("env_from_settings", "permissions", "subscribe_events", "companion_processes",
                         "scheduled_tasks", "plugin_api")
SKILL_UI_FACTS = ("ui_tab", "contributes", "page_module")
# What a seat is told about the governance it was given and owes, by layer and delivery ("every" reaches both):
# operative words of the review contract (``commit_gate.commit_review_contract_fingerprint``), never the
# documents they name. Each layer's navigation below is rendered from its own entry.
NAVIGATION_WORDS = {
    CORE_LAYER: {
        # (heading, rule set with a supplied section, rule set without one, what the subject is and is not)
        "every": (
            "## Governance navigation (core layer)",
            "The checklist section inlined above is the whole rule set for this review",
            f"NO section of `{CHECKLISTS_PATH}` is inlined for this review (this surface supplies none); the "
            "universal change-review rules are the whole rule set",
            ": the subject is not the Ouroboros body, so Ouroboros's constitution, engineering handbook, design "
            "system, architecture map and standing disclosures do not govern it and are not delivered. The "
            "subject's own documents are evidence of what it promises, not law: use them to judge behavioural "
            "documentation, public contracts and release metadata."),
        PACKET_DELIVERY: (" This row has no repository tools: the documents named below were NOT delivered; state "
                          "any resulting uncertainty rather than claim to have read them.",),
        RETRIEVING_DELIVERY: (" Everything named below is complete on disk in the subject repository; read any "
                              "range you need with your own tools.",),
    },
    BODY_LAYER: {
        PACKET_DELIVERY: (
            "## Governance navigation (index of sources not inlined)",
            "The supplied rules and selected sections are the governance evidence this packet delivers. This row "
            "has no repository tools: any ranges only indexed below were NOT delivered as governance. "
            "State the uncertainty that remains after reading the supplied evidence."),
        RETRIEVING_DELIVERY: (
            "## Governance navigation (read on demand)",
            "The rules above are the ones this change activates. Available sources below remain complete on disk "
            "and untruncated; missing sources are named. Read any range with `read_file(root=\"system_repo\", "
            "path=..., start_line=A, max_lines=N)`. Ranges are inclusive and a parent heading includes its "
            "descendant group. A document named here and not inlined was NOT dropped — it is one read away, and "
            "the review's governance manifest records the disposition of each one."),
    },
    SKILL_LAYER: {
        PACKET_DELIVERY: (
            "## Governance navigation (index of sources not inlined)",
            "The checklist section, BIBLE.md and the host contracts inlined above are the governance evidence "
            "this packet delivers. This row has no repository tools: the maps below identify sources that were "
            "NOT delivered, not additional evidence. State any resulting uncertainty; do not claim to have read "
            "the named sources."),
        RETRIEVING_DELIVERY: (
            "## Governance navigation (read on demand)",
            "The checklist section and BIBLE.md are inlined above. The sources named REQUIRED in your manifest "
            "({required}) bind what a skill may touch: read each in full with `read_file(root=\"system_repo\", "
            "path=..., start_line=A, max_lines=N)` before you judge. Everything else below is one read away, "
            "complete on disk and untruncated; a document named here and not inlined was NOT dropped, and the "
            "review's governance manifest records the disposition of each one."),
    },
}
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


def _selectable_blocks(source) -> list[tuple[str, int, str]]:
    """Individually selectable units of a chapter: pre-subheading blocks PLUS top subsections.

    Pre-subheading paragraphs, lists, and code blocks (between the chapter's H1 and its first
    subsection, or the entire body of a heading-less chapter) are each selectable on their own
    so content placed before the first ``##``/``###`` heading (such as numbered invariants or
    tables) is never invisible to packet reviewers (Bug #2).
    """
    blocks: list[tuple[str, int, str]] = []
    first_h1 = source.headings[0] if source.headings and source.headings[0].level == 1 else None
    pre_start = first_h1.span.end_byte if first_h1 is not None else 0
    sub_headings = [h for h in source.headings if h.level > 1]
    pre_end = sub_headings[0].span.start_byte if sub_headings else len(source.raw)
    pre_spans = sorted(
        (span for span in (*source.paragraphs, *source.lists, *source.code_blocks, *source.tables)
         if pre_start <= span.start_byte and span.end_byte <= pre_end),
        key=lambda s: s.start_byte,
    )
    # Skip nested spans (e.g. paragraphs inside list blocks already covered by the list span).
    outer_pre: list[Any] = []
    for span in pre_spans:
        if outer_pre and span.start_byte < outer_pre[-1].end_byte:
            continue
        outer_pre.append(span)
    for idx, span in enumerate(outer_pre, 1):
        body = source.text_at(span).strip()
        if body:
            blocks.append((f"opening block {idx} (lines {span.start_line}-{span.end_line})", span.start_byte, body))
    for heading in _top_sections(source):
        body = source.text_at(heading.section)
        if body.strip():
            blocks.append((heading.title, heading.span.start_byte, body))
    return blocks


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


# The core layer's frame of the subject's own documents (the documents and their maps are data).
_SUBJECT_DOCS = {
    "unsupplied": ("### Subject documents\n\nNo subject root was supplied: the subject's own documents were not "
                   "indexed for this review"),
    PACKET_DELIVERY: " and were not delivered to this packet.", RETRIEVING_DELIVERY: "; read them with your own tools.",
    "none": "### Subject documents\n\nThe subject keeps no Markdown document at its root or under `docs/` (root: `{root}`).",
    "listed": "### Subject documents (root: `{root}`)", "item": "- `{rel}` ({size:,} bytes)",
    "more": "- … {count} more Markdown document(s) under the same roots, not listed",
}


def _checklist_pointer(rules: bool, shared: bool) -> str:
    """The body navigation's line on the checklist book: which of its sections are inlined above."""
    inlined = [label for label, present in (("the section that applies to this review", rules),
                                            (f"its `{SHARED_CHECKLIST_SECTION}` section", shared)) if present]
    pointer = (f"{' and '.join(inlined)} {'are' if len(inlined) > 1 else 'is'} inlined above."
               if inlined else "NO section of it is inlined for this review.")
    return f"- `{CHECKLISTS_PATH}` — the complete checklist book; {pointer}"


def navigation_material(layer: str, delivery: str) -> list:
    """Review-contract material (``commit_gate.commit_review_contract_fingerprint``): a layer's governance navigation
    as its renderer composes it for ``delivery``, over placeholders (the subject's documents, the sources a skill
    owes). The documents it names, their headings and the guides' own text are not here."""
    from ouroboros.context_layout import design_navigation
    from ouroboros.tools.review_helpers import GOVERNANCE_OMISSION

    words = NAVIGATION_WORDS.get(layer, {})
    if layer != CORE_LAYER:
        pointers = [_checklist_pointer(rules, shared) for rules in (True, False) for shared in (True, False)]
        omissions = [text.format(rel_path="⟨path⟩", path="⟨path⟩", error="⟨error⟩") for text in GOVERNANCE_OMISSION.values()]
        if layer == BODY_LAYER:
            pointers.extend(design_navigation("", source="⟨source⟩", disclosure=note)
                            for note in _DESIGN_REVIEW_MAP.values())
        return [*(text.format(required="⟨required⟩") for text in words.get(delivery, ())), *omissions,
                *(pointers if layer == BODY_LAYER else ())]
    heading, with_rules, without_rules, subject_words = words["every"]
    (tail,) = words[delivery]
    return [heading, *(rules + subject_words + tail for rules in (with_rules, without_rules)),
            _SUBJECT_DOCS["unsupplied"] + _SUBJECT_DOCS[delivery], _SUBJECT_DOCS["none"].format(root="⟨root⟩"),
            _SUBJECT_DOCS["listed"].format(root="⟨root⟩"), _SUBJECT_DOCS["item"].format(rel="⟨path⟩", size=1),
            _SUBJECT_DOCS["more"].format(count=1)]


def _subject_navigation(subject_root: Any, *, packet: bool) -> tuple[str, list]:
    """The core layer's index of the SUBJECT's documents, with its manifest rows."""
    from ouroboros.context_layout import generate_doc_nav_map

    rows: list[dict] = []
    if subject_root is None:
        return _SUBJECT_DOCS["unsupplied"] + _SUBJECT_DOCS[PACKET_DELIVERY if packet else RETRIEVING_DELIVERY], rows
    root = Path(subject_root)
    docs = _subject_documents(root)
    if not docs:
        return _SUBJECT_DOCS["none"].format(root=root), rows
    lines = [_SUBJECT_DOCS["listed"].format(root=root), ""]
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
        lines.append(_SUBJECT_DOCS["item"].format(rel=rel, size=size))
    if len(docs) > SUBJECT_DOCS_MAX_LISTED:
        lines.append(_SUBJECT_DOCS["more"].format(count=len(docs) - SUBJECT_DOCS_MAX_LISTED))
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
    words = NAVIGATION_WORDS[CORE_LAYER]
    heading, with_rules, without_rules, subject_words = words["every"]
    (tail,) = words[PACKET_DELIVERY if packet else RETRIEVING_DELIVERY]
    navigation = _join([
        heading, (with_rules if checklist_text.strip() else without_rules) + subject_words + tail, subject_nav,
        render_required_sources(required, layer=CORE_LAYER) if surface != "scope" else "",
    ])
    return GovernanceContext(navigation=navigation, manifest=rows,
                             tokens_estimate=estimate_tokens(navigation), layer=CORE_LAYER)


def _contract_text(root: Path, path: str) -> str:
    """A host contract rendered for a reviewer: a Python source in a fence, a document as it is."""
    from ouroboros.tools.review_helpers import load_governance_doc
    from ouroboros.tools.review_prompt_text import format_prompt_code_block

    text = load_governance_doc(root, path, on_missing="explicit")
    return format_prompt_code_block(text, "python") if path.endswith(".py") and not text.startswith("[⚠️") else text


def _skill_layer_context(repo_dir: Any, *, surface: str, delivery: str, checklist_section_text: str,
                         manifest_facts: Any) -> GovernanceContext:
    """The skill layer (module docstring): tiers by the manifest's facts, never by a verdict."""
    from ouroboros.context_layout import generate_doc_nav_map
    from ouroboros.tools.review_helpers import load_governance_doc

    root = Path(repo_dir)
    packet = str(delivery or PACKET_DELIVERY) == PACKET_DELIVERY
    facts = dict(manifest_facts or {}) if isinstance(manifest_facts, Mapping) else {}
    kind = str(facts.get("type") or "")
    declares = {str(item) for item in (facts.get("declares") or ())}
    rows: list[dict] = []
    tier1: list[tuple[str, str]] = []
    selected: list[tuple[str, str]] = []
    inline_texts: dict[str, str] = {}
    checklist_text = str(checklist_section_text or "")
    rows.append(_row(CHECKLISTS_PATH, 1, "inline", len(checklist_text),
                     f"applicable {surface} checklist section (skill layer), supplied by the surface")
                if checklist_text.strip() else
                _row(CHECKLISTS_PATH, 1, "navigation", 0, "this surface supplies no checklist section"))
    rows.append(_row(f"{CHECKLISTS_PATH}#{SHARED_CHECKLIST_SECTION}", 1, NOT_APPLICABLE_DISPOSITION, 0,
                     "a rule of this repository's own code; a skill payload is not it"))
    bible = load_governance_doc(root, BIBLE_PATH, on_missing="explicit")
    tier1.append((BIBLE_PATH, bible))
    inline_texts[BIBLE_PATH] = bible
    rows.append(_row(BIBLE_PATH, 1, "inline", len(bible), "tier 1: every reviewer carries it"))
    rows.append(_row(CHECKLISTS_ARCHIVE_PATH, 1, NOT_APPLICABLE_DISPOSITION, 0,
                     "standing disclosures of the body's own changes; not a skill rule"))
    required = [(path, "tier 2: binds what a skill may touch (the chapters the checklist names)")
                for path in SKILL_REQUIRED_CHAPTERS]
    plugin_facts = sorted(declares & set(SKILL_PLUGIN_API_FACTS))
    ui_facts = sorted(declares & set(SKILL_UI_FACTS))
    conditional = (
        (SKILL_PLUGIN_API_PATH, kind == "extension" or bool(plugin_facts),
         "the host contract of " + ("an extension" if kind == "extension" else f"a manifest declaring {', '.join(plugin_facts)}"),
         "not an extension and the manifest declares no host-mediated capability"),
        (SKILL_UI_VALIDATION_PATH, kind == "extension" or bool(ui_facts),
         "the host's widget and contribution validator for " + (
             "an extension (widgets may be registered from plugin.py)" if kind == "extension" else f"a manifest declaring {', '.join(ui_facts)}"),
         "not an extension and the manifest declares no UI surface"),
    )
    for path, reason in required:
        text = _contract_text(root, path)
        if packet:
            selected.append((path, text))
            inline_texts[path] = text
            rows.append(_row(path, 2, "inline", len(text), reason))
        else:
            rows.append(_row(path, 2, "required", len(text), reason + "; read it in full"))
    for path, needed, reason, skipped in conditional:
        text = _contract_text(root, path)
        if not needed:
            rows.append(_row(path, 2, "navigation", len(text), f"tier 2 by manifest facts: {skipped}"))
        elif packet:
            selected.append((path, text))
            inline_texts[path] = text
            rows.append(_row(path, 2, "inline", len(text), f"tier 2 by manifest facts: {reason}"))
        else:
            rows.append(_row(path, 2, "required", len(text), f"tier 2 by manifest facts: {reason}; read it in full"))
    maps: list[str] = []
    for book_id, entrypoint in ((DEVELOPMENT_BOOK_ID, "docs/DEVELOPMENT.md"), (ARCHITECTURE_BOOK_ID, "docs/ARCHITECTURE.md")):
        book, problem = _load_book(root, book_id)
        rows.append(_row(entrypoint, 3, "navigation", 0,
                         "book navigation; a chapter not required above is one read away" if _book_sources(book)
                         else problem or "no chaptered membership: navigation only"))
        for chapter in _book_sources(book):
            if chapter.source_path not in SKILL_REQUIRED_CHAPTERS:
                rows.append(_row(chapter.source_path, 3, "navigation", len(chapter.text), "the map is never inlined whole"))
        maps.append(_book_navigation(root, book_id, book, load_governance_doc, instructions=not packet))
    creating = load_governance_doc(root, CREATING_SKILLS_PATH, on_missing="explicit")
    rows.append(_row(CREATING_SKILLS_PATH, 3, "navigation", len(creating), "the skill author's guide: its heading map"))
    maps.append(generate_doc_nav_map(creating, title=CREATING_SKILLS_PATH.rsplit("/", 1)[-1], rel_path=CREATING_SKILLS_PATH,
                                     instructions=not packet) if not creating.startswith("[⚠️") else creating)
    required_names = [row["path"] for row in rows if row["disposition"] == "required"]
    heading, words = NAVIGATION_WORDS[SKILL_LAYER][PACKET_DELIVERY if packet else RETRIEVING_DELIVERY]
    navigation = _join([heading, words.format(required=", ".join(f"`{name}`" for name in required_names)), *maps])
    stable_inline = _join(_render(title, text) for title, text in tier1)
    selected_inline = _join(_render(title, text) for title, text in selected)
    return GovernanceContext(
        inline_sections=tier1 + selected, navigation=navigation, manifest=rows,
        tokens_estimate=estimate_tokens(_join([stable_inline, selected_inline, navigation])),
        stable_inline=stable_inline, selected_inline=selected_inline, inline_whole_documents=inline_texts,
        layer=SKILL_LAYER)


def _affected_domain_module_map(
    map_root: Path,
    touched: set[str],
    touched_domains: tuple[str, ...],
    domain_by_path: Mapping[str, str],
    *,
    read_bytes: Any = None,
    max_neighbor_domains: int = 4,
    max_rows_per_domain: int = 45,
) -> tuple[str, tuple[str, ...], str]:
    """Build live module leads: all touched modules across all touched domains (plus touched web/ files) first, then neighbors."""
    from ouroboros import work_domain
    from ouroboros.code_intelligence_architecture import _module_lead, module_map_rows

    dom_lines: list[str] = []
    shown_domains: list[str] = []
    omitted_neighbor_doms: list[str] = []
    try:
        for idx, dom in enumerate(touched_domains):
            full_rows = module_map_rows(map_root, dom, read_bytes=read_bytes)
            if not full_rows:
                continue
            header_row, module_rows = full_rows[0], full_rows[1:]
            touched_first: list[str] = []
            rest: list[str] = []
            seen_touched: set[str] = set()
            for row_text in module_rows:
                mod_path = row_text.split(" — ", 1)[0].strip()
                if mod_path in touched:
                    touched_first.append(row_text)
                    seen_touched.add(mod_path)
                else:
                    rest.append(row_text)
            for rel in sorted(touched - seen_touched):
                if rel.endswith((".py", ".js", ".mjs", ".ts")) and domain_by_path.get(rel) == dom:
                    lead = _module_lead(map_root, rel, read_bytes=read_bytes)
                    if lead or (read_bytes is None and (map_root / rel).is_file()):
                        touched_first.append(f"{rel} — {lead}" if lead else rel)
            if idx < max_neighbor_domains:
                ordered_mod_rows = [*touched_first, *rest]
                cap = max(len(touched_first), max_rows_per_domain - 1)
                dom_lines.append(header_row)
                dom_lines.extend(ordered_mod_rows[:cap])
                omitted_mods = len(ordered_mod_rows) - min(len(ordered_mod_rows), cap)
                if omitted_mods > 0:
                    dom_lines.append(
                        f"(+{omitted_mods} more module(s) omitted in {dom}; "
                        f"use query_code(op='architecture', query='module_map {dom}', root='system_repo'))"
                    )
                shown_domains.append(dom)
            elif touched_first:
                # Beyond the neighbor-domain cap, still deliver every touched module's lead so
                # tool-less packet seats never lose an edited file's purpose across large PRs.
                dom_lines.append(header_row)
                dom_lines.extend(touched_first)
                if rest:
                    dom_lines.append(
                        f"(+{len(rest)} untouched neighbor module(s) omitted in {dom}; "
                        f"use query_code(op='architecture', query='module_map {dom}', root='system_repo'))"
                    )
                shown_domains.append(dom)
            else:
                omitted_neighbor_doms.append(dom)
        if omitted_neighbor_doms:
            dom_lines.append(
                f"(+{len(omitted_neighbor_doms)} more touched domain(s) omitted: {', '.join(omitted_neighbor_doms)})"
            )
        touched_web = sorted(p for p in touched if p.startswith("web/") and p.endswith((".js", ".mjs", ".ts")))
        web_rows: list[str] = []
        for rel in touched_web:
            lead = _module_lead(map_root, rel, read_bytes=read_bytes)
            if lead or (read_bytes is None and (map_root / rel).is_file()):
                web_rows.append(f"{rel} — {lead}" if lead else rel)
            elif read_bytes is not None:
                try:
                    read_bytes(rel); web_rows.append(rel)
                except OSError:
                    pass
        if web_rows:
            dom_lines.extend((f"web/ ({len(web_rows)} touched JS/TS module(s))", *web_rows))
            shown_domains.append("web/")
    except work_domain.WorkDomainUnsettled:
        raise
    except (OSError, ValueError) as exc:
        return "", tuple(shown_domains), f"affected domain module map unavailable: {type(exc).__name__}: {exc}"
    return "\n".join(dom_lines), tuple(shown_domains), ""


def _resolve_body_subject_domains(
    gov_root: Path,
    map_root: Path,
    touched_paths: Iterable[Any],
    *,
    read_bytes: Any = None,
) -> tuple[set[str], tuple[str, ...], dict[str, str]]:
    """Resolve owning manuals from executing ``gov_root`` and affected domains from both ``map_root`` and ``gov_root``."""
    from ouroboros.reference_books import load_domain_references, resolve_body_references

    body_refs = resolve_body_references(gov_root, touched_paths, include_mentions=True)
    manuals: set[str] = set(body_refs.get("manuals") or ())
    domains: list[str] = list(body_refs.get("domains") or ())
    domain_by_path: dict[str, str] = {
        rel: str(info.get("domain") or "")
        for rel, info in (body_refs.get("by_path") or {}).items()
        if info.get("domain")
    }
    if map_root.resolve(strict=False) != gov_root.resolve(strict=False) or read_bytes is not None:
        _, gov_mods, gov_refs = load_domain_references(gov_root)
        _, cand_mods, _ = load_domain_references(map_root, read_bytes=read_bytes)
        for raw in touched_paths:
            rel = _normalize(raw)
            dom = cand_mods.get(rel) or gov_mods.get(rel) or ""
            if dom:
                domain_by_path[rel] = dom
                if dom not in domains:
                    domains.append(dom)
                for m in gov_refs.get(dom, ()):
                    manuals.add(m)
    return manuals, tuple(domains), domain_by_path


def _read_body_invariants_text(root: Path, arch_chapters: Iterable[Any]) -> Optional[str]:
    """Read required §10 invariants from ``arch_chapters``, falling back to direct disk read if a sibling broke."""
    ch = next((c for c in arch_chapters if c.source_path == BODY_INVARIANTS_CHAPTER), None)
    if ch is not None:
        return ch.text
    direct = root / BODY_INVARIANTS_CHAPTER
    try:
        if direct.is_file():
            return direct.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return None


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
    manifest_facts: Any = None,
    review_words: Any = None,
    module_map_root: Any = None,
    module_map_source: str = "worktree",
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
    :func:`_core_layer_context` over ``subject_root``; ``skill`` (a skill payload)
    runs :func:`_skill_layer_context` over ``manifest_facts`` (``type`` and the
    manifest's ``declares``)."""
    from ouroboros.tools.review_helpers import load_checklist_section, load_governance_doc

    if layer not in GOVERNANCE_LAYERS:
        raise ValueError(f"unknown governance layer {layer!r}; expected one of {GOVERNANCE_LAYERS}")
    if layer == CORE_LAYER:
        return _core_layer_context(surface=surface, touched_paths=touched_paths, delivery=delivery,
                                   checklist_section_text=checklist_section_text,
                                   subject_root=subject_root)
    if layer == SKILL_LAYER:
        return _skill_layer_context(repo_dir, surface=surface, delivery=delivery,
                                    checklist_section_text=checklist_section_text, manifest_facts=manifest_facts)

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

    # --- tier 2: required body invariants + rules by change class ---------
    packet = str(delivery or PACKET_DELIVERY) == PACKET_DELIVERY
    arch_book, arch_problem = _load_book(root, ARCHITECTURE_BOOK_ID)
    arch_chapters = _book_sources(arch_book)
    inv_text = _read_body_invariants_text(root, arch_chapters)
    if inv_text is not None:
        if packet:
            # Required immune-system source (matching SKILL_LAYER): always delivered whole to
            # packet seats ahead of discretionary tier-2 chapters so small windows never starve it.
            remaining = max(0, remaining - estimate_tokens(_render(BODY_INVARIANTS_CHAPTER, inv_text)))
            selected.append((BODY_INVARIANTS_CHAPTER, inv_text))
            inline_texts[BODY_INVARIANTS_CHAPTER] = inv_text
            rows.append(_row(BODY_INVARIANTS_CHAPTER, 2, "inline", len(inv_text),
                             "required cross-cutting invariants and Closed roads for body review"))
        else:
            rows.append(_row(BODY_INVARIANTS_CHAPTER, 2, "required", len(inv_text),
                             "required cross-cutting invariants and Closed roads; read in full"))
    elif (root / "docs/ARCHITECTURE.md").is_file() or arch_problem:
        rows.append(_row(BODY_INVARIANTS_CHAPTER, 2, "navigation", 0,
                         "required cross-cutting invariants and Closed roads unavailable at this path"))

    map_root = Path(module_map_root) if module_map_root else (Path(subject_root) if subject_root else root)
    from ouroboros.reference_books import source_reader as _source_reader
    map_reader = _source_reader(map_root, source="index") if module_map_source == "index" else None
    ref_manuals, touched_domains, domain_by_path = _resolve_body_subject_domains(
        root, map_root, touched_paths or (), read_bytes=map_reader)

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
        elif files or path in ref_manuals:
            reason = (f"mentions {files} touched file(s), {mentions} time(s)" if files
                      else "owning manual via domains.toml [references]")
            candidates.append(((2, -(files or 1) * 1000 - mentions, path), path, path, reason))
        else:
            rows.append(_row(path, 2, "navigation", len(chapter.text),
                             "no touched file mentioned"))

    web_touched = any(path.startswith("web/") for path in touched)
    has_words = review_words is not None and bool(review_words.changes or review_words.gaps)
    design_text = load_governance_doc(root, DESIGN_PATH, on_missing="explicit")
    chapter_text = {chapter.source_path: chapter.text for chapter in dev_chapters}
    chapter_text[DESIGN_PATH] = design_text
    design_missing = design_text.startswith("[⚠️ OMISSION") or not design_text.strip()
    # The design bible alone: its whole section, principles, seed table and recipes, the
    # very slice its author carries (``context_layout.design_bible_section``).
    from ouroboros.context_layout import design_bible_section, design_navigation

    bible_path, principles_path = _DESIGN_BIBLE_PATH, _DESIGN_PRINCIPLES_PATH
    bible = None if design_missing else design_bible_section(design_text)
    principles = None
    if bible:
        bible_title = f"{bible_path} (excerpt: the whole section)"
        chapter_text[bible_path] = bible
        # A share too small for the whole section still carries its opening and numbered principles, the slice
        # reviewers had before (RV1), cut from that same section and labelled as only that.
        listed = re.search(r"(?ms)\A.*?^1\. .*?(?=^(?!\d+\. )\S|\Z)", bible)
        if listed and listed[0].strip() != bible:
            last = re.findall(r"(?m)^(\d+)\. ", listed[0])[-1]
            principles = f"{bible_path} (excerpt: principles 1–{last} only)"
            chapter_text[principles_path] = listed[0].strip()
    bible_absent = "design bible excerpt unavailable; the source has no design bible section"

    def _admit_bible(reason: str) -> None:  # the whole section, else its principles alone
        if not _admit(bible_path, bible_title, chapter_text[bible_path], 2, reason) and principles:
            _admit(principles_path, principles, chapter_text[principles_path], 2,
                   f"{reason}; the whole section does not fit the inline share, its principles only")
    if design_missing:
        rows.append(_row(DESIGN_PATH, 2, "navigation", 0,
                         design_text or "DESIGN source is empty; content not delivered"))
    elif web_touched:
        candidates.append(((1, 0, DESIGN_PATH), DESIGN_PATH, DESIGN_PATH,
                           "a touched path is under web/"))
    elif has_words:
        if bible:
            candidates.append(((1, 0, bible_path), bible_path, bible_title,
                               "changed source words; design bible excerpt only"))
        else:
            rows.append(_row(bible_path, 2, "navigation", 0, bible_absent))
        rows.append(_row(DESIGN_PATH, 2, "navigation", len(design_text),
                         "heading map for the whole book; only its bible is eligible for changed source words"))
    else:
        rows.append(_row(DESIGN_PATH, 2, "navigation", len(design_text),
                         "heading map only; no frontend path or changed source words"))

    for order, path, title, reason in sorted(candidates):
        if path == bible_path:
            _admit_bible(reason)
        elif not _admit(path, title, chapter_text.get(path, ""), 2, reason) and path == DESIGN_PATH:
            # The whole book does not fit this reviewer's share. Its bible is what a
            # change under web/ is judged by, so it is carried before any chapter chosen
            # by a file name; the heading map still indexes the rest.
            if bible:
                _admit_bible("a touched path is under web/ and the whole book does not fit the inline share; "
                             "design bible excerpt")
            else:
                rows.append(_row(bible_path, 2, "navigation", 0, bible_absent))

    # --- tier 3: the map, never whole (invariants delivered above) --------
    arch_reason = "the map is delivered as navigation, never whole; " + (
        "key invariants (with Closed roads) and relevant sections are selected below" if packet
        else "reads sections on demand; key invariants (with Closed roads) are required")
    rows.append(_row("docs/ARCHITECTURE.md", 3, "navigation", 0, arch_reason if arch_chapters
                     else arch_problem or "no chaptered membership: navigation only"))
    sections: list[tuple[tuple, str, str, str]] = []
    for chapter in arch_chapters:
        if chapter.source_path == BODY_INVARIANTS_CHAPTER:
            continue
        if not packet:
            rows.append(_row(chapter.source_path, 3, "navigation", len(chapter.text),
                             "the map is never inlined whole; chapter navigation above"))
            continue
        matched_any = False
        if names:
            for block_title, start_byte, body in _selectable_blocks(chapter):
                files, mentions = _mentions(body, names)
                if not files:
                    continue
                matched_any = True
                path = f"{chapter.source_path}#{block_title}"
                sections.append(((-files * 1000 - mentions, chapter.source_path, start_byte),
                                 path, path, f"mentions {files} touched file(s), {mentions} time(s)"))
                chapter_text[path] = body
        if not matched_any and chapter.source_path in ref_manuals:
            sections.append(((0, chapter.source_path, 0),
                             chapter.source_path, chapter.source_path,
                             "owning manual via domains.toml [references]"))
            chapter_text[chapter.source_path] = chapter.text
        else:
            rows.append(_row(chapter.source_path, 3, "navigation", len(chapter.text),
                             "the map is never inlined whole; chapter navigation above"))
    for order, path, title, reason in sorted(sections):
        _admit(path, title, chapter_text.get(path, ""), 3, reason)
    if packet and (touched_domains or any(p.startswith("web/") and p.endswith((".js", ".mjs", ".ts")) for p in touched)):
        dom_text, shown_domains, dom_err = _affected_domain_module_map(
            map_root, touched, touched_domains, domain_by_path, read_bytes=map_reader)
        if dom_text:
            _admit("ouroboros/domains.toml#affected_domains",
                   "Affected domain module map (projected from domains.toml & docstrings)",
                   dom_text, 3,
                   f"live module leads for touched domain(s): {', '.join(shown_domains)}")
        elif dom_err:
            rows.append(_row("ouroboros/domains.toml#affected_domains", 3, "navigation", 0, dom_err))

    # --- navigation -------------------------------------------------------
    pointers = [_checklist_pointer(bool(checklist_text.strip()), shared_inline)]
    if not packet:
        required_nav = [row["path"] for row in rows if row["disposition"] == "required"]
        if required_nav:
            pointers.append(
                "REQUIRED governance read(s) before judging (read each in full via `read_file` using its listed root, "
                "e.g. `system_repo` or `session_root` for materialized copies): "
                + ", ".join(f"`{p}`" for p in required_nav) + "."
            )
    design_map = ""
    if DESIGN_PATH not in inline_texts:
        delivered = next((row["path"] for row in rows if row["disposition"] == "inline"
                          and row["path"] in (bible_path, principles_path)), DESIGN_PATH)
        design_map = design_navigation(
            None if design_missing else design_text, source=str(root / DESIGN_PATH),
            disclosure=_DESIGN_REVIEW_MAP[delivered])
    nav_heading, nav_words = NAVIGATION_WORDS[BODY_LAYER][PACKET_DELIVERY if packet else RETRIEVING_DELIVERY]
    navigation = _join([
        nav_heading, nav_words,
        "\n".join(pointers),
        design_map,
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
