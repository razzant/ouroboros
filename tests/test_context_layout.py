"""Unit tests for the context_layout doc-layout SSOT (low/max)."""

from pathlib import Path

from ouroboros import context_layout as cl


def test_tier0_protected_core_declared():
    """The protected always-full core is a data invariant; future context-mode
    work must not silently demote any of these (BIBLE P1 / P4). The marks I keep
    in view joined it when conversations became the memory view's, whose physical
    floor alone turns old lines into addresses: the recent-dialogue horizon left."""
    kept = {"system", "bible", "identity", "scratchpad"}
    assert kept <= set(cl.TIER0_ALWAYS_FULL)
    assert "memory_marks" in cl.TIER0_ALWAYS_FULL
    # The hand-written registry joined on 2.11 by the owner's choice: whole, outlined only physically.
    assert cl.TIER0_WHOLE_OR_OUTLINE == {"project_knowledge_index", "knowledge_index", "registry"}
    assert not cl.TIER0_WHOLE_OR_OUTLINE & cl.TIER0_ALWAYS_FULL
    assert "recent_dialogue" not in cl.TIER0_ALWAYS_FULL
    # The shared-understanding overview joined on 22.10 by the owner's choice (20.10): never shortened.
    assert set(cl.TIER0_ALWAYS_FULL) == kept | {"memory_marks", "shared_understanding"}


def test_index_floor_order_and_whole_or_outline_fact_have_one_owner():
    from ouroboros import memory_floor

    assert memory_floor.WHOLE_OR_OUTLINE_STEPS is cl.WHOLE_OR_OUTLINE_STEPS
    assert cl.TIER0_WHOLE_OR_OUTLINE == frozenset(cl.WHOLE_OR_OUTLINE_STEPS)
    assert cl.WHOLE_OR_OUTLINE_STEPS == (*cl.INDEX_STEPS, "registry")  # the registry outline comes last
    assert set(cl.INDEX_STEPS) == {"project_knowledge_index", "knowledge_index"}
    assert not cl.TIER0_WHOLE_OR_OUTLINE & cl.TIER0_ALWAYS_FULL
    assert not cl.TIER0_WHOLE_OR_OUTLINE & set(memory_floor.HOST_STEPS)


def test_the_memory_floor_never_takes_the_marks_it_keeps_in_tier_zero():
    """``memory_marks`` in the tier-0 core is true only while no floor step addresses a mark."""
    from ouroboros import memory_floor
    from tests._memory_view_synthetic import actor

    view = actor("main")
    assert view.marks, "the synthetic Main view carries marks"
    elements = memory_floor.floor_elements(view)
    assert elements, "the floor has elements to take on this view"
    mark_ids = {mark["id"] for mark in view.marks}
    assert not mark_ids & {ident for _step, ident, _whole, _short in elements}
    assert all(mark["text"] not in whole for mark in view.marks for _step, _ident, whole, _short in elements)


def test_nav_map_lists_h2_through_h4_as_inclusive_complete_subtrees():
    text = "\n".join([
        "# Title",                   # 1
        "intro",                     # 2
        "## Alpha",                  # 3
        "alpha BODYSENT",            # 4
        "### Shared",                # 5
        "sub intro",                 # 6
        "#### Child one",            # 7
        "child one body",            # 8
        "#### Child two",            # 9
        "child two body",            # 10
        "### Shared",                # 11 (duplicate title)
        "second shared body",        # 12
        "#### Last child",            # 13
        "last body",                 # 14
        "## Beta",                   # 15
        "beta body",                 # 16
        "### Final",                 # 17
        "final body",                # 18 (EOF)
    ])
    m = cl.generate_doc_nav_map(text, title="ARCHITECTURE.md", rel_path="docs/ARCHITECTURE.md")
    assert m == "\n".join([
        "## ARCHITECTURE.md (navigation map)",
        "",
        "Full text is NOT inlined to keep the working context window fit. Read any "
        "section on demand with `read_file(root=\"system_repo\", "
        "path=\"docs/ARCHITECTURE.md\", start_line=A, max_lines=N)` (untruncated). "
        "Ranges are inclusive; a parent includes its complete descendant group, "
        "and `max_lines=B-A+1` for `lines A-B`. Sections:",
        "",
        "- Alpha — lines 3-14",
        "  - Shared — lines 5-10",
        "    - Child one — lines 7-8",
        "    - Child two — lines 9-10",
        "  - Shared — lines 11-14",
        "    - Last child — lines 13-14",
        "- Beta — lines 15-18",
        "  - Final — lines 17-18",
    ])
    # Structure only — the section bodies are NOT inlined.
    assert "BODYSENT" not in m
    assert "final body" not in m


def test_nav_map_is_fence_aware_at_every_supported_depth():
    """Supported heading forms inside a backtick fence stay out of the map."""
    text = (
        "## Real\n\n```markdown\n## fake-h2\n### fake-h3\n#### fake-h4\n"
        "```\n\n### Real child\n\n#### Real grandchild\n"
    )
    m = cl.generate_doc_nav_map(text, title="X", rel_path="x.md")
    assert "- Real — lines 1-11" in m
    assert "  - Real child — lines 9-11" in m
    assert "    - Real grandchild — lines 11-11" in m
    assert "fake-h2" not in m
    assert "fake-h3" not in m
    assert "fake-h4" not in m


def test_the_real_architecture_book_navigation_exposes_all_h4_groups():
    """The compact view must not lose the subsection index the monolith's map
    carried: every `###`/`####` group still appears, addressed to the PHYSICAL
    chapter it lives in, with the parent range containing its children."""
    from ouroboros.reference_books import load_reference_book

    repo = Path(__file__).resolve().parents[1]
    book = load_reference_book(repo, "architecture")
    m = cl.book_navigation(book)

    assert "## 6.1 Agent Core — Loop, Tools & Safety (navigation map)" in m
    assert "Source: `docs/architecture/06-1-loop-tools-and-safety.md`" in m
    assert "Source: `docs/architecture/06-5-planning-memory-and-skills.md`" in m
    # A composed-book offset is never a locator; the read instruction is stated
    # once, over the physical Source paths.
    assert 'read_file(root="system_repo", path=...)' in m
    assert m.count("Full text is NOT inlined") == 0

    tool_children = (
        "Web access mechanisms (three distinct paths — do not conflate)",
        "Context fitting, retry, and compaction",
        "Vision and local image evidence",
        "Background consciousness and Evolution",
    )
    planning_children = (
        "Plan construction and review",
        "Deep self-review",
        "Post-task work",
        "Durable memory and project focus",
    )
    for title in (*tool_children, *planning_children):
        assert f"    - {title} — lines " in m, title

    def _range(indent: str, title: str) -> tuple[int, int]:
        prefix = f"{indent}- {title} — lines "
        row = next(line for line in m.splitlines() if line.startswith(prefix))
        start, end = row.removeprefix(prefix).split("-", 1)
        return int(start), int(end)

    for parent_title, children in (
        ("Tool capability and execution", tool_children),
        ("Planning, deep review, post-task work, memory", planning_children),
    ):
        parent_start, parent_end = _range("  ", parent_title)
        child_ranges = [_range("    ", title) for title in children]
        assert all(parent_start < start <= end <= parent_end for start, end in child_ranges)
        assert parent_end == child_ranges[-1][1]

    # Those ranges address the chapter file, so they resolve there and nowhere
    # else: the entrypoint has one section and cannot answer for them.
    chapter = next(c for c in book.chapters if c.source_path.endswith("06-1-loop-tools-and-safety.md"))
    start, end = _range("  ", "Tool capability and execution")
    lines = chapter.text.split("\n")
    assert lines[start - 1].startswith("### Tool capability and execution")
    assert end <= len(lines)


def test_a_legacy_revision_navigates_as_the_single_source_it_is():
    from ouroboros.reference_books import load_reference_book

    monolith = b"# Book\n\nOrientation.\n\n## Runtime\n\nBody.\n\n### Startup\n\nMore.\n"
    book = load_reference_book(Path("unused"), "architecture", lambda _: monolith)
    nav = cl.book_navigation(book)
    assert "docs/ARCHITECTURE.md" in nav and "Runtime" in nav and "Startup" in nav


def test_nav_map_no_heading_fallback_names_all_supported_depths():
    m = cl.generate_doc_nav_map("# Title\nbody", title="X", rel_path="x.md")
    assert "(no `##`/`###`/`####` headings; read `x.md` directly)" in m


def test_reference_doc_sections_carry_one_body_map_and_the_handbook_only_when_required():
    """Every mode carries the one resident Body Map (whole in Max; a host section whose operative view is its parts
    in Low and Nano) and no manual behind it; the whole handbook rides only as a task's own requirement, in any mode.
    Where no Map could be rendered, both books stay named as addresses, never silently."""
    from ouroboros.section_floor import HostSection

    body_map = "## Body map\n\nMAPBODY\n"
    parts = "## Body map (parts and addresses)\n\n- Core: `docs/architecture/core.md`"
    dev = "## Dev A\n\nDEVBODY\n"
    for mode in ("max", "low", "nano"):
        sections, development = cl.reference_doc_sections(context_mode=mode, body_map=body_map,
                                                          body_map_parts=parts, development_text=dev)
        [section] = sections
        assert section == body_map and development == "" and "DEVBODY" not in "\n\n".join(sections), mode
        if mode == "max":  # nothing shorter to take in the owner's Max
            assert not isinstance(section, HostSection) or section.operative_view is None
        else:
            assert isinstance(section, HostSection) and section.section_id == "body_map"
            assert section.operative_view == parts
        _sections, required = cl.reference_doc_sections(context_mode=mode, body_map=body_map, body_map_parts=parts,
                                                         development_text=dev, required_book=True)
        assert "DEVBODY" in required and required == cl.whole_development_book(dev), mode
    [plain], _ = cl.reference_doc_sections(context_mode="low", body_map=body_map)  # a Map with no shorter form
    assert plain == body_map and (not isinstance(plain, HostSection) or plain.operative_view is None)
    [pointer], _ = cl.reference_doc_sections(context_mode="max", body_map="")
    assert "docs/ARCHITECTURE.md" in pointer and "docs/DEVELOPMENT.md" in pointer
