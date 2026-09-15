"""Unit tests for the context_layout doc-layout SSOT (low/max)."""

from pathlib import Path

from ouroboros import context_layout as cl


def test_tier0_protected_core_declared():
    """The protected always-full core is a data invariant; future context-mode
    work must not silently demote any of these (BIBLE P1 / P4)."""
    expected = {
        "system",
        "bible",
        "identity",
        "scratchpad",
        "knowledge_index",
        "recent_dialogue",
    }
    assert expected <= set(cl.TIER0_ALWAYS_FULL)


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

    assert "## 6. Agent Core (navigation map)" in m
    assert "Source: `docs/architecture/06-agent-core.md`" in m
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
        "Post-task reflection",
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
        ("Planning, deep review, reflection, memory", planning_children),
    ):
        parent_start, parent_end = _range("  ", parent_title)
        child_ranges = [_range("    ", title) for title in children]
        assert all(parent_start < start <= end <= parent_end for start, end in child_ranges)
        assert parent_end == child_ranges[-1][1]

    # Those ranges address the chapter file, so they resolve there and nowhere
    # else: the entrypoint has one section and cannot answer for them.
    chapter = next(c for c in book.chapters if c.source_path.endswith("06-agent-core.md"))
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


def test_reference_doc_sections_preserve_max_and_orient_both_books_in_low():
    """Max retains its full bodies; Low names both complete retrievable books."""
    arch = "## Arch A\n\nARCHBODY\n"
    dev = "## Dev A\n\nDEVBODY\n"

    def _render(mode, include_dev):
        parts = cl.reference_doc_sections(
            None,
            context_mode=mode,
            include_development=include_dev,
            architecture_text=arch,
            development_text=dev,
        )
        return "\n\n".join(parts)

    max_no_dev = _render("max", False)
    assert "ARCHBODY" in max_no_dev  # ARCH full in max even without dev context
    assert "DEVBODY" not in max_no_dev
    assert "docs/DEVELOPMENT.md" in max_no_dev  # pointer, never silent

    max_dev = _render("max", True)
    assert "ARCHBODY" in max_dev and "DEVBODY" in max_dev

    low_dev = _render("low", True)
    assert "ARCHBODY" not in low_dev  # nav map in low
    assert "navigation map" in low_dev
    assert "DEVBODY" not in low_dev
    assert "## DEVELOPMENT.md (navigation map)" in low_dev

    low_no_dev = _render("low", False)
    assert "DEVBODY" not in low_no_dev
    assert "docs/DEVELOPMENT.md" in low_no_dev
