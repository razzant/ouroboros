"""CPL-2 verify half (plan §7.2): the three generated inventories match a
fresh regeneration and their resolution invariants hold — staleness = red.

The generator half is ``python scripts/regenerate_inventories.py``:

- ``docs/inventories/FROZEN_CONTRACTS_INVENTORY.md`` — ARCHITECTURE §11.1 rows,
  every referenced owner/anchor path resolved against the tree, plus the
  ``ouroboros/contracts/`` package-coverage gap list (pinned here: growth of
  the gap is red even after regeneration);
- ``docs/inventories/DATA_LAYOUT_INVENTORY.md`` — the ARCHITECTURE "Data layout"
  tree probed entry-by-entry against tracked paths / runtime source literals
  (zero UNRESOLVED entries pinned here);
- ``docs/inventories/FACADE_INVENTORY.md`` — the AST-derived ``noqa: F401``
  re-export facade inventory over the domain manifest population.

Synthetic tests prove the red branches (missing file, unresolvable entry,
marker detection) actually fire.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "regenerate_inventories", REPO_ROOT / "scripts" / "regenerate_inventories.py")
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)


@pytest.fixture(scope="module")
def frozen():
    return inv.build_frozen_inventory()


@pytest.fixture(scope="module")
def layout():
    return inv.build_layout_inventory()


@pytest.fixture(scope="module")
def facades():
    return inv.build_facade_inventory()


# ---------------------------------------------------------------------------
# Byte-identity: regeneration must not change the committed inventories
# ---------------------------------------------------------------------------

def _assert_identical(out_path: pathlib.Path, rendered: str):
    rendered += "\n" if not rendered.endswith("\n") else ""
    assert out_path.is_file(), (
        f"{out_path.name} is missing — run `python scripts/regenerate_inventories.py`")
    assert out_path.read_text(encoding="utf-8") == rendered, (
        f"{out_path.name} is stale (regeneration changes it) — run "
        "`python scripts/regenerate_inventories.py`")


def test_frozen_contracts_inventory_is_byte_identical(frozen):
    _assert_identical(inv.FROZEN_OUT, frozen[0])


def test_data_layout_inventory_is_byte_identical(layout):
    _assert_identical(inv.LAYOUT_OUT, layout[0])


def test_facade_inventory_is_byte_identical(facades):
    _assert_identical(inv.FACADE_OUT, facades[0])


def test_ui_control_text_inventory_is_byte_identical():
    _assert_identical(inv.UI_CONTROLS_OUT, inv.build_ui_control_inventory()[0])


def test_ui_control_scan_lists_fixed_text_and_only_counts_the_rest():
    from scripts.ui_control_inventory import scan_controls

    rows, interpolated = scan_controls(
        """
        <button class="btn btn-default btn-sm" type="button">Retry</button>
        <button class="chat-attach-btn" aria-label="Attach file"><svg viewBox="0 0 1 1"></svg></button>
        <button class="btn ${tone}" type="button"><span>Save &amp; close</span></button>
        <button class="btn">${escapeHtml(label)}</button>
        <button class="icon-only"><svg></svg></button>
        <button class="btn btn-default" type="button"
            ${rows.length >= MAX_ROWS ? 'disabled' : ''}>Add subagent</button>
        const go = document.createElement('button');
        go.className = 'btn btn-xs btn-default';
        go.textContent = 'Turn into project';
        const dynamic = document.createElement('button');
        dynamic.textContent = labelFor(row);
        """
    )
    # Fixed text is listed with its classes; an icon-only control is named by its fixed
    # accessible name; an interpolated class is shown as a gap, never guessed.
    assert rows == [
        ("Retry", "btn btn-default btn-sm"),
        ("[icon] Attach file", "chat-attach-btn"),
        ("Save & close", "btn …"),
        # A comparison inside a templated attribute is not the end of the tag.
        ("Add subagent", "btn btn-default"),
        ("Turn into project", "btn btn-xs btn-default"),
    ]
    # Run-time text, a nameless icon and a computed label are counted, not invented.
    assert interpolated == 3


def test_ui_control_inventory_keeps_siblings_adjacent_and_names_the_one_raiser():
    rendered = inv.build_ui_control_inventory()[0]
    table = [line for line in rendered.splitlines() if line.startswith("| ") and not line.startswith("| text")]
    refresh = [index for index, line in enumerate(table) if line.startswith("| Refresh |")]
    # The point of the sort: every control that says the same thing is one contiguous block.
    assert len(refresh) >= 2 and refresh == list(range(refresh[0], refresh[0] + len(refresh)))
    # An owner intent with a door is raised by exactly one module.
    assert "| `ouro:open-project` | modules/project_reference.js |" in rendered


# ---------------------------------------------------------------------------
# Resolution invariants
# ---------------------------------------------------------------------------

def test_every_frozen_contract_reference_resolves(frozen):
    assert frozen[1] == [], (
        "ARCHITECTURE §11.1 references files that no longer exist — fix §11.1 "
        f"(and regenerate): {frozen[1]}")


def test_frozen_table_is_nonempty_and_covers_known_owners(frozen):
    doc = frozen[0]
    assert "`ouroboros/contracts/tool_context.py` (ok)" in doc
    assert "`ouroboros/contracts/plugin_api.py` (ok)" in doc
    assert "`ouroboros/gateway/contracts.py`" in doc


def test_contracts_package_coverage_gap_is_pinned(frozen):
    """§11.1 rows cover the whole contracts package; a NEW contracts-package module
    that never gets a §11.1 row must turn this red even after regeneration."""
    doc = frozen[0]
    known_gaps: set[str] = set()
    listed = {line[3:-1] for line in doc.splitlines()
              if line.startswith("- `ouroboros/contracts/")}
    assert listed == known_gaps, (
        "the §11.1 package-coverage gap changed: a new frozen-package module "
        "needs a §11.1 row (or an explicit owner decision recorded by updating "
        f"this pin). now-listed={sorted(listed)} pinned={sorted(known_gaps)}")


def test_every_data_layout_entry_resolves(layout):
    assert layout[1] == [], (
        "ARCHITECTURE Data-layout tree entries no longer resolve against the "
        f"tree/runtime sources — fix the tree (and regenerate): {layout[1]}")


def test_data_layout_probes_key_durable_files(layout):
    doc = layout[0]
    for token in ("settings.json", "queue_snapshot.json", "usage.sqlite",
                  "terminal_deliveries.json", "chat.jsonl"):
        assert f"`{token}`" in doc, f"layout inventory lost the `{token}` entry"


def test_facade_inventory_finds_the_known_big_facades(facades):
    doc = facades[0]
    for facade in ("ouroboros/config.py", "ouroboros/llm.py", "ouroboros/loop.py",
                   "supervisor/queue.py", "supervisor/events.py",
                   "ouroboros/tools/registry.py"):
        assert f"| `{facade}` |" in doc, (
            f"facade inventory lost `{facade}` — either its re-export markers "
            "vanished (a facade-surface change) or the scanner regressed")


# ---------------------------------------------------------------------------
# Synthetic red-branch coverage
# ---------------------------------------------------------------------------

def test_frozen_row_parser_extracts_owner_and_anchor_paths():
    section = (
        "\nprose `ouroboros/gateway/contracts.py` here.\n\n"
        "| Contract | File | Anchored by |\n"
        "|----------|------|-------------|\n"
        "| `Thing` — words | `ouroboros/contracts/tool_abi.py` | "
        "`tests/test_contracts.py::test_x` and `helper()` |\n")
    rows = inv.parse_frozen_rows(section)
    assert rows == [{
        "label": "Thing",
        "owners": ["ouroboros/contracts/tool_abi.py"],
        "anchors": ["tests/test_contracts.py::test_x"],
    }]


def test_frozen_row_parser_handles_escaped_pipes():
    section = (
        "| Contract | File | Anchored by |\n"
        "|---|---|---|\n"
        "| `SkillManifest` (`a \\| b \\| c`) | `ouroboros/contracts/skill_manifest.py` | prose |\n")
    rows = inv.parse_frozen_rows(section)
    assert rows[0]["owners"] == ["ouroboros/contracts/skill_manifest.py"]


def test_layout_parser_extracts_entries_and_probe_tokens():
    block = (
        "\n~/Ouroboros/\n"
        "├── data/\n"
        "│   ├── settings.json   ← User settings\n"
        "│   └── state/\n"
        "│       └── code_intel/<repo_key>/inventory.json ← facts\n"
        "└── <only placeholders>/\n")
    entries = inv.parse_layout_entries(block)
    assert entries == ["data/", "settings.json", "state/",
                       "code_intel/<repo_key>/inventory.json", "<only placeholders>/"]
    assert inv._probe_token("code_intel/<repo_key>/inventory.json") == "inventory.json"
    assert inv._probe_token("<only placeholders>/") is None


def test_noqa_f401_marker_detection():
    import ast
    src = ("from ouroboros import config  # noqa: F401\n"
           "from ouroboros import loop  # noqa: E501\n"
           "from ouroboros import llm  # noqa\n"
           "from ouroboros import agent\n")
    lines = src.splitlines()
    nodes = ast.parse(src).body
    flags = [inv._statement_has_noqa_f401(lines, n) for n in nodes]
    assert flags == [True, False, True, False]


def _chapter_book():
    from ouroboros.reference_books import load_reference_book

    frozen = (
        "## 11.1 What is frozen\n\n"
        "| Contract | File | Anchored by |\n|---|---|---|\n"
        "| `Thing` | `ouroboros/contracts/tool_abi.py` | `tests/test_contracts.py` |\n"
    )
    layout = (
        "## Data layout (`~/Ouroboros/`)\n\n````\n~/Ouroboros/\n"
        "├── data/\n│   ├── settings.json\n│   └── queue_snapshot.json\n````\n"
    )
    # Filenames carry no topic meaning; membership and actual headings own it.
    corpus = {
        "docs/ARCHITECTURE.md": b"# Book\n\nThe body and its reasons.\n\n## Chapters\n\n- [First](architecture/alpha.md)\n- [Second](architecture/omega.md)\n",
        "docs/architecture/alpha.md": ("# Contracts\n\nWHY: old consumers must retain their contract.\n\n" + frozen).encode(),
        "docs/architecture/omega.md": ("# Storage\n\nWHY: retained sources must remain discoverable.\n\n" + layout).encode(),
    }
    book = load_reference_book(pathlib.Path("/unused"), "architecture", corpus.__getitem__)
    legacy_text = "# Book\n\nSame complete sources.\n\n" + frozen + "\n" + layout
    legacy = load_reference_book(pathlib.Path("/unused"), "architecture", lambda _: legacy_text.encode())
    return book, legacy, corpus


@pytest.mark.parametrize("builder", ["build_frozen_inventory", "build_layout_inventory"])
def test_chapter_migration_preserves_complete_inventory_and_physical_provenance(builder):
    import hashlib

    book, legacy, corpus = _chapter_book()
    chapter_text, findings = getattr(inv, builder)(book)
    old_text, old_findings = getattr(inv, builder)(legacy)
    assert findings == old_findings == []
    without_source = chapter_text.splitlines(keepends=True)
    index = next(i for i, line in enumerate(without_source) if line.startswith("Source: "))
    del without_source[index:index + 2]
    assert "".join(without_source) == old_text
    title = "11.1 What is frozen" if builder == "build_frozen_inventory" else "Data layout (`~/Ouroboros/`)"
    view = inv.read_book_section(book, title)
    ref = view.sources[0]
    raw = corpus[ref.path]
    assert ref.sha256 == hashlib.sha256(raw).hexdigest()
    assert raw[ref.span.start_byte:ref.span.end_byte].decode() == view.text
    assert f"`{ref.path}`, physical LF lines {ref.span.start_line}-{ref.span.end_line}" in chapter_text
    assert ref.sha256 in chapter_text


def test_data_layout_cannot_borrow_a_fence_from_another_section():
    with pytest.raises(ValueError, match="one fenced tree"):
        inv.layout_block("# Book\n\n## Data layout (`~/Ouroboros/`)\n\nMissing.\n\n## Other\n\n```\n├── wrong.json\n```\n")


def test_frozen_section_uses_markdown_headings_not_fenced_examples():
    text = (
        "# Book\n\n```md\n### 11.1 What is frozen\nFake content\n```\n\n"
        "## 11.1 What is frozen\n\nActual content\n\n## Next\n\nOutside\n"
    )
    section = inv.frozen_section_text(text)
    assert "Actual content" in section
    assert "Fake content" not in section and "Outside" not in section


def test_missing_member_refuses_generator_before_partial_inventory(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ARCHITECTURE.md").write_text(
        "# Book\n\nPurpose.\n\n## Chapters\n\n- [Missing](architecture/missing.md)\n"
    )
    monkeypatch.setattr(inv, "REPO_ROOT", tmp_path)
    for builder in (inv.build_frozen_inventory, inv.build_layout_inventory):
        with pytest.raises(FileNotFoundError, match="missing.md"):
            builder()
