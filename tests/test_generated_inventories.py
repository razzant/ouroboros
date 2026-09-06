"""CPL-2 verify half (plan §7.2): the three generated inventories match a
fresh regeneration and their resolution invariants hold — staleness = red.

The generator half is ``python scripts/regenerate_inventories.py``:

- ``docs/v7next/FROZEN_CONTRACTS_INVENTORY.md`` — ARCHITECTURE §11.1 rows,
  every referenced owner/anchor path resolved against the tree, plus the
  ``ouroboros/contracts/`` package-coverage gap list (pinned here: growth of
  the gap is red even after regeneration);
- ``docs/v7next/DATA_LAYOUT_INVENTORY.md`` — the ARCHITECTURE "Data layout"
  tree probed entry-by-entry against tracked paths / runtime source literals
  (zero UNRESOLVED entries pinned here);
- ``docs/v7next/FACADE_INVENTORY.md`` — the AST-derived ``noqa: F401``
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
    """The two known §11.1 gaps are pinned; a NEW contracts-package module
    that never gets a §11.1 row must turn this red even after regeneration."""
    doc = frozen[0]
    known_gaps = {
        "ouroboros/contracts/skill_payload_policy.py",
        "ouroboros/contracts/task_constraint.py",
    }
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
    for token in ("settings.json", "queue_snapshot.json", "usage_attempts.jsonl",
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
