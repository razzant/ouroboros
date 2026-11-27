"""The resident Body Map, the DESIGN slice and reference-book views.

Every mode (Max, Low, Nano) and every focus (roots, wakes, Presence, children, nannies, voice) carries one
resident Body Map rendered from the Architecture book's sources (``reference_books.body_map``): what each
part gives me, how I reach it, which limit changes a choice and where its manuals live. The manuals behind
it (the chapters of both books and the documents its Reference Atlas names) are read on demand; a task's own
direct requirement adds the whole Development handbook in any mode. ``book_navigation`` remains the
chapter-addressed view reviewers and plan seats use.

Source bytes belong to ReferenceBook; this module only renders the selected
view. Stable full prefixes contain no cwd, task identity or revision counters.
SYSTEM, BIBLE and the common identity/narrative core remain caller-owned.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, List, Mapping, Optional

from ouroboros.reference_books import ReferenceBook, overview_book

# Protected core: always rendered in full, in every context mode. Encoded as
# data so a drift-guard test can assert no future change demotes it. The marks I
# keep in view never become addresses (``memory_floor`` has no step for them);
# conversations are the memory view's, whose physical floor alone degrades them.
# The shared-understanding overview rides whole too: the owner chose on 20.10.2026
# that the window never shortens it (its headings are no substitute for it).
TIER0_ALWAYS_FULL = frozenset({
    "system",
    "bible",
    "identity",
    "scratchpad",
    "memory_marks",
    "shared_understanding",
})


# Resident where present; only the physical window may choose its outline: the knowledge
# indexes and, last, the hand-written registry (owner 2.11.2026: whole in every mode).
INDEX_STEPS = ("project_knowledge_index", "knowledge_index")
WHOLE_OR_OUTLINE_STEPS = (*INDEX_STEPS, "registry")
TIER0_WHOLE_OR_OUTLINE = frozenset(WHOLE_OR_OUTLINE_STEPS)


def generate_doc_nav_map(text: str, *, title: str, rel_path: str, instructions: bool = True) -> str:
    """Build a compact, fence-aware navigation map of a markdown doc.

    Lists every ``##`` through ``####`` heading with its inclusive line range
    so the agent knows what exists and where, and can pull the full section on demand via
    ``read_file(root="system_repo", path=rel_path, start_line=A, max_lines=N)``.
    A parent's range includes its complete descendant group, so parent and child
    ranges intentionally overlap. This is a lossless index
    (P1: no silent truncation) — the single canonical file on disk is unchanged.
    """
    entries = _nav_entries(text)
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
    if not entries:
        out.append("- (no `##`/`###`/`####` headings; "
                   + (f"read `{rel_path}` directly)" if instructions else f"source: `{rel_path}`)"))
    return "\n".join([*out, *_nav_lines(entries)])


def _nav_entries(text: str) -> List[tuple[int, str, int, int]]:
    """Each ``##``-``####`` heading outside code fences: ``(level, title, first line, last line)``, 1-based."""
    lines = text.splitlines()
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
    return [(level, title, lineno, next((later - 1 for later_level, _title, later in headings[idx + 1:]
                                         if later_level <= level), len(lines)))
            for idx, (level, title, lineno) in enumerate(headings)]


def _nav_lines(entries: List[tuple[int, str, int, int]]) -> List[str]:
    return [f"{'  ' * (level - 2)}- {title} — lines {first}-{last}" for level, title, first, last in entries]


def _design_bible(text: str) -> Optional[tuple[str, int, int]]:
    """DESIGN's ``## Design bible`` section verbatim with its 1-based inclusive lines, or ``None``: from the heading
    titled exactly so (a subtitle only after ``: ``) to the next level-1 or level-2 heading outside a code fence
    (fences read as ``generate_doc_nav_map`` reads them). Every reader sees its lines with ``\\n`` endings, however
    the file was written; a digest still names the file's own bytes."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    first, in_fence = None, False
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif in_fence:
            continue
        elif first is None and re.fullmatch(r"## Design bible(?:: .*)?", line.rstrip()):
            first = number
        elif first is not None and line.startswith(("# ", "## ")):
            return "".join(lines[first - 1:number - 1]).strip(), first, number - 1
    return None if first is None else ("".join(lines[first - 1:]).strip(), first, len(lines))


def design_bible_section(text: str) -> Optional[str]:
    """The one DESIGN slice an author, a session and a reviewer of an interface change each carry whole: the
    design bible section, its principles, seed table and recipes; ``None`` when the book has none."""
    found = _design_bible(text)
    return found[0] if found else None


# The first line of an installed book read for work whose own tree is not established as the body (Fable F3, RVD1 N2).
INSTALLED_DESIGN = ("These are the interface rules of Ouroboros itself, from its installed body; this work's own tree is "
                    "not established as that body.")


def design_bible_view(text: Optional[str], *, source: str, installed: bool = False) -> str:
    """An author's DESIGN: its design bible whole, the rest of the book by headings and ranges, the exact source.

    Only the file's bytes and its path shape it (no time, counter or service state), so the block it rides keeps its
    bytes through a return. A source that cannot be read, or a book without the section, keeps the navigation's own
    honest form (``design_navigation``). ``installed``: the book is the installed body's for work whose own tree is
    not established as that body (``author_design_source``), and its first line says whose interface it governs
    (``INSTALLED_DESIGN``), in every form."""
    whose = [INSTALLED_DESIGN, ""] if installed else []
    found = None if text is None else _design_bible(text)
    if found is None:
        navigation = design_navigation(text, source=source)
        return "\n".join([*whose, navigation if text is None else navigation
                          + " This source has no `## Design bible` section."])
    section, first, last = found
    rest = [(level, f"{title} (whole above)", first, last) if line == first else (level, title, line, end)
            for level, title, line, end in _nav_entries(text) if not first < line <= last]
    return "\n".join([*whose, section, "", "## DESIGN.md (navigation map)", "", *_nav_lines(rest), "",
                      f"Source: {source}. Use this exact source path when reading; relative paths may address a "
                      f"candidate. Ranges are inclusive. Inlined above: its design bible whole (lines {first}-{last}); "
                      "every other section by its heading and range only. Availability is not evidence of reading or "
                      "application. Select relevant original sections; a participant without source-reading tools "
                      "needs those excerpts in its packet."])


def design_navigation(text: str | None, *, source: str, disclosure: str = "") -> str:
    """Index DESIGN headings; a review supplies its actual inline coverage and its own reader guidance."""
    locator = f"Source: {source}."
    if text is None:
        return f"## DESIGN.md source unavailable\n\n{locator} No DESIGN content or navigation delivered."
    if disclosure:
        locator += f" Ranges are inclusive. {disclosure} Availability is not evidence of reading or application."
    else:
        locator += (" Use this exact source path when reading; relative paths may address a candidate. "
                    "Ranges are inclusive. Navigation only: DESIGN content is not inlined; "
                    "availability is not evidence of reading or application. Select relevant original sections; "
                    "a participant without source-reading tools needs those excerpts in its packet.")
    return (
        generate_doc_nav_map(text, title="DESIGN.md", rel_path="docs/DESIGN.md", instructions=False)
        + f"\n\n{locator}"
    )


def task_design_source(ctx: Any, task: Mapping[str, Any]) -> pathlib.Path | None:
    """Select the author's body source by the same identity as workspace/review tools.

    A default false self-body flag cannot hide a recognized body. Conversely an
    explicit true flag never turns a known foreign workspace into Ouroboros.
    Folderless self-body work can address system_repo without reclassifying scratch.
    """
    return _design_source(ctx, task)[0]


def _design_source(ctx: Any, task: Mapping[str, Any]) -> tuple[pathlib.Path | None, str]:
    """``task_design_source`` and the active root's body fact it read (``review_body_fact.BODY_*``)."""
    from ouroboros import body_candidate
    from ouroboros.context_input_selection import _explicit_self_body_docs_flag
    from ouroboros.review_body_fact import BODY_FALSE, BODY_TRUE, body_fact
    from ouroboros.tool_access_roots import folderless_scratch_dir
    from ouroboros.tools.tool_resolution import active_repo_dir_for, system_repo_dir_for
    from ouroboros.workspace_copies import copy_binding, same_directory

    active, system = active_repo_dir_for(ctx), system_repo_dir_for(ctx)
    binding = copy_binding(task) or copy_binding(ctx)
    if binding and binding.get("source_is_system_repo") is not False and any(
            same_directory(parent, binding.get("execution_root")) for parent in (active, *active.parents)):
        return pathlib.Path(binding["execution_root"]), BODY_TRUE
    # A false copy flag means a separate Git directory, not necessarily a foreign
    # project: body_fact also recognizes the configured upstream/fork chain.
    if body_candidate.is_bound(ctx) and folderless_scratch_dir(ctx) is not None:
        # Explicit preparation/restoration selects the body while scratch stays
        # the default cwd. A foreign workspace/room still follows its own identity.
        return pathlib.Path(body_candidate.descriptor(ctx)["path"]), BODY_TRUE
    fact = body_fact(active, system_repo=system,
                     data_dir=getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None))
    if fact.body == BODY_TRUE:
        if fact.how == "dir":
            return system, BODY_TRUE
        # Read-only investigators may start below a body's Git worktree root.
        # Identity is already established; Git metadata locates its document.
        return next((parent for parent in (active, *active.parents)
                     if (parent / ".git").exists()), active), BODY_TRUE
    if fact.body == BODY_FALSE:
        return None, BODY_FALSE
    if _explicit_self_body_docs_flag(dict(task)) is True and folderless_scratch_dir(ctx) is not None:
        return system, fact.body
    return None, fact.body


def author_design_source(ctx: Any, task: Mapping[str, Any]) -> tuple[pathlib.Path | None, bool]:
    """Whose DESIGN an author's own context carries, and whether it is the installed body's for a tree not
    established as that body (the owner's item 97 of 21.11.2026; council C13, R2).

    The recognized body's book first (``task_design_source``). Otherwise the installed one, never an unadopted
    candidate the context is bound to (``body_candidate.serving_repo_dir_for``): for work the task itself declares
    Ouroboros's (``context_requires_self_body_docs`` true, from a foreign folder too), and for work whose body is
    unknown yet not proven foreign (a project room without a folder: missing facts never narrow it). A false or
    absent declaration
    narrows nothing: the contract writes false for silence. Proven-foreign work without the declaration gets none;
    children, nannies and background work of the body keep it (no on-demand surface narrows it).
    """
    from ouroboros.body_candidate import serving_repo_dir_for
    from ouroboros.context_input_selection import _explicit_self_body_docs_flag
    from ouroboros.review_body_fact import BODY_FALSE, BODY_TRUE

    root, body = _design_source(ctx, task)
    if body == BODY_TRUE:
        return root, False
    if body != BODY_FALSE or _explicit_self_body_docs_flag(dict(task)) is True:
        return serving_repo_dir_for(ctx), True
    return None, False


def read_design_source(path: pathlib.Path) -> tuple[bytes, str] | None:
    """One scoped read supplies both the UTF-8 view and the original bytes a session binds by digest."""
    from subprocess import TimeoutExpired

    from ouroboros import work_domain
    from ouroboros.workspace_executor_paths import ManagedPathRefused

    try:
        raw = work_domain.read_bytes(path)
        return raw, raw.decode("utf-8")
    except (OSError, UnicodeError, work_domain.WorkDomainRefused, ManagedPathRefused, TimeoutExpired):
        return None


def task_design_navigation(ctx: Any, task: Mapping[str, Any]) -> str:
    """Capture once before context fitting; the task-specific root stays out of A'.

    The author's view is a host section whose operative form is the book's navigation. Its whole excerpt stays
    while the physical request can hold it beside my shortest tellings, in every mode."""
    root, installed = author_design_source(ctx, task)
    if root is None:
        return ""
    path = root / "docs/DESIGN.md"
    _, text = read_design_source(path) or (None, None)
    source = f"`{path}`"
    view = design_bible_view(text, source=source, installed=installed)
    if text is None or design_bible_section(text) is None:
        return view  # no section, nothing shorter to yield to
    from ouroboros.section_floor import HostSection  # section_floor reads TIER0_ALWAYS_FULL from here

    navigation = "\n\n".join([*([INSTALLED_DESIGN] if installed else []), design_navigation(text, source=source)])
    return HostSection("design_bible", view, navigation)


def book_navigation(book: ReferenceBook, *, instructions: bool = True) -> str:
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
            instructions=instructions,
        )
    return overview_book(
        book,
        lambda chapter: generate_doc_nav_map(
            chapter.text,
            title=chapter.headings[0].title if chapter.headings else chapter.source_path,
            rel_path=chapter.source_path,
            instructions=False,
        ),
        instructions=instructions,
    ).text


def body_map_section(body_map: str, parts: str = "", *, steppable: bool) -> str:
    """The resident Body Map as the governance block carries it (BIBLE P1).

    The owner's Max holds it whole: there it is a plain part with nothing shorter to take. A frame the window
    lowered, and the owner's own Low or Nano, carry the same bytes as a host section whose operative view is the
    Map's parts and addresses, which only a physical window takes (``memory_floor.BOOK_MAPS``): never a mode's
    target, a start's desired room or a requested saving. Without ``parts`` (a legacy single-file revision)
    there is no shorter form.
    """
    from ouroboros.section_floor import HostSection, report_text  # section_floor reads TIER0_ALWAYS_FULL from here

    if not body_map.strip():
        return ""
    if steppable and parts.strip():
        return HostSection("body_map", body_map, parts)
    return report_text(body_map, "body_map")


def whole_development_book(text: str) -> str:
    """The composed handbook as its own context block: entrypoint and every chapter."""
    from ouroboros.section_floor import report_text

    return report_text("## DEVELOPMENT.md\n\n" + text, "development")


def reference_doc_sections(
    *,
    context_mode: str,
    body_map: str,
    body_map_parts: str = "",
    development_text: str = "",
    required_book: bool = False,
) -> tuple[List[str], str]:
    """The common reference sections and the separate handbook block.

    Every mode and focus carries the one resident Body Map; no mode or binding preloads a manual behind it. A
    task's own direct requirement (``required_book``) adds the whole Development handbook as its own block in
    any mode, so every other task keeps one shared prefix. Where no Map could be rendered the addresses of the
    two books stand in its place beside the capture's own disclosure of the gap.
    """
    section = body_map_section(body_map, body_map_parts, steppable=context_mode != "max")
    parts: List[str] = [section] if section else [
        "## Reference docs available on demand\n\n"
        "Not inlined in the working context: `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, `README.md`, "
        "`docs/CHECKLISTS.md`. Read them in full (untruncated) with "
        "`read_file(root=\"system_repo\", path=...)` when relevant."]
    development = whole_development_book(development_text) if required_book and development_text.strip() else ""
    return parts, development
