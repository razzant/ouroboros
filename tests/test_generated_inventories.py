"""Source invariants for frozen contracts, the layout tree and facade bindings.

Only the UI control snapshot is pinned to a file. The other reports project
current book and code sources; historical protected IDs now check those sources
instead of comparing committed copies.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import sys
from collections import Counter

import pytest

from ouroboros.code_intelligence_architecture import reexported_names

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
    scan = inv.scan_facade_bindings
    bindings = None

    def capture(*args, **kwargs):
        nonlocal bindings
        assert bindings is None, "facade inventory must scan the sources once"
        # Capture the full stream even if the renderer stops consuming it early.
        bindings = list(scan(*args, **kwargs))
        return iter(bindings)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inv, "scan_facade_bindings", capture)
        report, findings = inv.build_facade_inventory()
    assert bindings
    return report, findings, Counter((facade, leaf) for facade, _, leaf in bindings)


# ---------------------------------------------------------------------------
# The tracked UI snapshot and source-derived report invariants
# ---------------------------------------------------------------------------

def _assert_identical(out_path: pathlib.Path, rendered: str):
    rendered += "\n" if not rendered.endswith("\n") else ""
    regenerate = f"run scripts/regenerate_inventories.py with this test's Python ({sys.executable})"
    assert out_path.is_file(), f"{out_path.name} is missing: {regenerate}"
    assert out_path.read_text(encoding="utf-8") == rendered, (
        f"{out_path.name} is stale: {regenerate} and commit docs/inventories/ with the changes it describes, "
        "alone when they are already committed; untracked files count, and another task's uncommitted work "
        "owes its own regeneration.")


def test_data_layout_inventory_is_byte_identical(layout):
    """Protected node: every live Data layout entry still resolves and is reported."""
    view = inv.read_book_section(inv.load_reference_book(REPO_ROOT, "architecture"), inv.LAYOUT_SECTION)
    entries = inv.parse_layout_entries(inv.layout_block(view.text))
    report, findings = layout
    assert not findings, f"Data layout probes no longer resolve: {findings}"
    assert entries
    assert [line.split("`", 2)[1] for line in report.splitlines() if line.startswith("| `")] == [
        entry.replace("|", "\\|") for entry in entries]


def _facade_facts(report):
    facts = {}
    for line in report.splitlines():
        if not line.startswith("| `"):
            continue
        path, domain, bindings, cells = (item.strip() for item in line[1:-1].split("|"))
        path = path.strip("`")
        assert path not in facts, path
        leaves = {}
        for cell in cells.split("<br>"):
            match = re.fullmatch(r"`([^`]+)` \(([1-9][0-9]*)(?: ✗(\S+))?\)", cell)
            assert match, cell
            leaf, count, cross_domain = match.groups()
            assert leaf not in leaves, (path, leaf)
            leaves[leaf] = (int(count), cross_domain)
        facts[path] = (domain, int(bindings), leaves)
    return facts


def test_facade_inventory_is_byte_identical(facades):
    """Protected node: live report facts agree with source without rescanning all ASTs."""
    report, findings, expected_counts = facades
    assert not findings
    facts = _facade_facts(report)
    modules = inv.load_manifest(REPO_ROOT / "ouroboros/domains.toml").modules
    assert {(path, leaf): count for path, (_, _, leaves) in facts.items()
            for leaf, (count, _) in leaves.items()} == expected_counts
    for path, (domain, bindings, leaves) in facts.items():
        assert domain == modules[path], path
        assert bindings == sum(count for count, _ in leaves.values()), path
        for leaf, (_, cross_domain) in leaves.items():
            assert cross_domain == (None if modules[leaf] == domain else modules[leaf]), (path, leaf)


def test_ui_control_text_inventory_is_byte_identical():
    _assert_identical(inv.UI_CONTROLS_OUT, inv.build_ui_control_inventory()[0])


def test_fixed_translated_selection_and_menu_actions_are_in_the_inventory():
    text = inv.build_ui_control_inventory()[0]
    assert "| Reply to quote | `btn btn-float btn-sm ui-hit-area chat-reply-selection-action`" in text
    assert "| Select text | `(no class)` | modules/chat_row_menu.js |" in text


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
        go.className = 'btn btn-sm btn-default';
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
        ("Turn into project", "btn btn-sm btn-default"),
    ]
    # Run-time text, a nameless icon and a computed label are counted, not invented.
    assert interpolated == 3


def test_ui_control_scan_lists_a_modules_own_button_factory_by_its_literal_calls():
    from scripts.ui_control_inventory import scan_controls

    rows, interpolated = scan_controls(
        """
        const button = (label, data, tone = 'btn-default', key = '') => `<button type="button" class="btn ${tone} btn-sm" ${data} data-key="${attr(data)}"${key ? ` data-i18n-fmt="${attr(key)}"` : ''}>${text(label)}</button>`;
        ${button('Show what can go', 'data-free-preview')}
        ${button('Free', 'data-free-go', 'btn-danger', 'storage.free.go')}
        ${button(goLabel(free, checked), 'data-free-go', 'btn-danger')}
        const pill = (label) => `<button class="pill">Pin</button>`;
        ${pill('Unpin')}
        """
    )
    # The factory's template is no control; each call with a literal label is one, with the classes its
    # arguments and the factory's defaults resolve to. An arrow whose button text is fixed is not a factory.
    assert rows == [
        ("Pin", "pill"),
        ("Show what can go", "btn btn-default btn-sm"),
        ("Free", "btn btn-danger btn-sm"),
    ]
    assert interpolated == 1, "a call whose label is computed is counted, not invented"


# ---------------------------------------------------------------------------
# Resolution invariants
# ---------------------------------------------------------------------------

def test_every_frozen_contract_reference_resolves(frozen):
    assert frozen[1] == [], (
        f"ARCHITECTURE §11.1 references files that no longer exist: {frozen[1]}")


def test_frozen_table_is_nonempty_and_covers_known_owners(frozen):
    doc = frozen[0]
    assert "`ouroboros/contracts/tool_context.py` (ok)" in doc
    assert "`ouroboros/contracts/plugin_api.py` (ok)" in doc
    ws_http = next(line for line in doc.splitlines() if "| `gateway/contracts.py` |" in line)
    for leaf in ("contracts", "ui_i18n_contracts", "client_contracts", "management_contracts", "health_contracts"):
        assert f"`ouroboros/gateway/{leaf}.py` (ok)" in ws_http


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
    prose, (row,) = inv.parse_frozen_rows(section)
    assert prose == ("ouroboros/gateway/contracts.py",)
    assert (row.label, row.owners, row.anchors) == (
        "Thing", ("ouroboros/contracts/tool_abi.py",), ("tests/test_contracts.py::test_x",))


def test_frozen_row_parser_handles_escaped_pipes():
    section = (
        "| Contract | File | Anchored by |\n"
        "|---|---|---|\n"
        "| `SkillManifest` (`a \\| b \\| c`) | `ouroboros/contracts/skill_manifest.py` | prose |\n")
    _, (row,) = inv.parse_frozen_rows(section)
    assert row.owners == ("ouroboros/contracts/skill_manifest.py",)


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
    src = ("from ouroboros import config  # noqa: F401\n"
           "from ouroboros import loop  # noqa: E501\n"
           "from ouroboros import llm  # noqa\n"
           "from ouroboros import agent\n")
    bindings = reexported_names(ast.parse(src), src.splitlines(), "ouroboros/example.py")
    assert [(base, alias.name) for base, alias in bindings] == [("ouroboros", "config"), ("ouroboros", "llm")]


def test_facade_report_keeps_both_leaf_domains_and_shadowed_bindings(tmp_path, monkeypatch):
    pkg = tmp_path / "ouroboros"
    pkg.mkdir()
    (pkg / "domains.toml").write_text(
        '[domains]\nDA = "Alpha"\nDB = "Beta"\n[modules]\n'
        '"ouroboros/facade.py" = "DA"\n"ouroboros/local.py" = "DA"\n"ouroboros/foreign.py" = "DB"\n',
        encoding="utf-8",
    )
    (pkg / "facade.py").write_text(
        "from .local import alpha, beta as chosen  # noqa: F401\n"
        "from .foreign import gamma as chosen  # noqa: F401\n", encoding="utf-8",
    )
    (pkg / "local.py").write_text("alpha = 1\nbeta = 2\n", encoding="utf-8")
    (pkg / "foreign.py").write_text("gamma = 3\n", encoding="utf-8")

    monkeypatch.setattr(inv, "REPO_ROOT", tmp_path)
    report, findings = inv.build_facade_inventory()
    assert not findings
    assert _facade_facts(report) == {
        "ouroboros/facade.py": ("DA", 3, {"ouroboros/local.py": (2, None), "ouroboros/foreign.py": (1, "DB")})
    }


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


def test_data_layout_report_refuses_a_tree_without_entries():
    text = "# Book\n\n## Data layout (`~/Ouroboros/`)\n\n```\n~/Ouroboros/\n```\n"
    book = inv.load_reference_book(REPO_ROOT, "architecture", lambda _: text.encode())
    with pytest.raises(ValueError, match="Data layout tree has no entries"):
        inv.build_layout_inventory(book)


def test_frozen_section_uses_markdown_headings_not_fenced_examples():
    text = (
        "# Book\n\n```md\n### 11.1 What is frozen\nFake content\n```\n\n"
        "## 11.1 What is frozen\n\nActual content\n\n## Next\n\nOutside\n"
    )
    book = inv.load_reference_book(REPO_ROOT, "architecture", lambda _: text.encode())
    section = inv.read_book_section(book, inv.FROZEN_SECTION).text
    assert "Actual content" in section
    assert "Fake content" not in section and "Outside" not in section


def test_the_inventories_read_a_candidate_the_same_staged_or_not(tmp_path, monkeypatch):
    """The commit gate stages a candidate and unstages it after a red, and Ouroboros cannot stage by hand:
    new and deleted modules, anchors and folders read the same staged or not, and every row follows the disk."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                        "-c", "commit.gpgsign=false", *args], cwd=tmp_path, check=True, capture_output=True)

    def write(path, text):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")

    book = ("# Book\n\n## 11.1 What is frozen\n\n| Contract | File | Anchored by |\n|---|---|---|\n"
            "| `A` | `ouroboros/contracts/a.py` | `tests/test_a.py` |\n\n## Data layout (`~/Ouroboros/`)\n\n"
            "````\n~/Ouroboros/\n└── data/\n    ├── state/\n{}    └── gone.json\n````\n")
    write("docs/ARCHITECTURE.md", book.format(""))
    write("ouroboros/contracts/a.py", "A = 1\n")
    write("tests/test_a.py", "def test_a():\n    pass\n")
    write("ouroboros/keeps.py", "ROOT, STATE = 'data', 'state'\n")
    write("ouroboros/gone.py", "STORE = 'gone.json'\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "base")
    monkeypatch.setattr(inv, "REPO_ROOT", tmp_path)
    assert inv.build_frozen_inventory()[1] == inv.build_layout_inventory()[1] == []
    write("docs/ARCHITECTURE.md", book.format("    ├── fresh.json\n"))
    write("ouroboros/fresh.py", "STORE = 'fresh.json'\n")
    write("ouroboros/contracts/b.py", "B = 1\n")
    write("skills/demo/state/notes.md", "a folder named like a layout entry\n")
    (tmp_path / "ouroboros" / "gone.py").unlink()
    (tmp_path / "tests" / "test_a.py").unlink()
    frozen, layout = inv.build_frozen_inventory(), inv.build_layout_inventory()
    git("add", "-A")
    assert (frozen, layout) == (inv.build_frozen_inventory(), inv.build_layout_inventory())
    assert "- `ouroboros/contracts/b.py`" in frozen[0] and "`tests/test_a.py` (MISSING)" in frozen[0]
    for row in ("| `data/` | `data` | code-ref |", "| `state/` | `state` | repo-dir |",
                "| `fresh.json` | `fresh.json` | code-ref |", "| `gone.json` | `gone.json` | UNRESOLVED |"):
        assert row in layout[0], row
    assert len(frozen[1]) == len(layout[1]) == 1


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
