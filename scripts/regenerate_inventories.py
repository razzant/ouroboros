#!/usr/bin/env python3
"""Regenerate the three CPL-2 gen/verify inventories (plan §7.2).

Each inventory is a generated document whose staleness turns CI red
(``tests/test_generated_inventories.py`` pins byte-identity against a fresh
in-memory regeneration, plus the resolution invariants below):

1. ``docs/v7next/FROZEN_CONTRACTS_INVENTORY.md`` — machine extraction of the
   ARCHITECTURE §11.1 frozen-contracts table: per row the contract label, the
   owner files, and the anchoring suites, every referenced repo path resolved
   against the tree (a row whose owner or anchor file disappeared = red), plus
   the ``ouroboros/contracts/`` package coverage (a contracts module never
   referenced by §11.1 is listed as a gap — growth of that list = red).
2. ``docs/v7next/DATA_LAYOUT_INVENTORY.md`` — machine extraction of the
   ARCHITECTURE "Data layout (`~/Ouroboros/`)" tree (the closest thing this
   tree has to the reference's PERSISTENCE_OWNERS carrier): every entry is
   probed against reality — repo entries must exist as tracked paths, data-
   plane entries must appear as a literal in the runtime sources that
   construct them (a renamed/removed durable file whose tree row survived =
   red).
3. ``docs/v7next/FACADE_INVENTORY.md`` — AST-derived facade inventory: every
   runtime module whose top-level ``from <population module> import ...``
   statements carry the ``noqa: F401`` re-export marker (the codebase's
   declared "this binding exists for compatibility" convention, per the
   reference FACADE_CONSUMERS method), with its leaves, name counts and
   domain from ``ouroboros/domains.toml``.

Convention follows the ratchet/domain-manifest pairs: generator in scripts/,
deterministic output (no timestamps, no HEAD SHAs), verify test in tests/.
``--check`` exits 1 when any on-disk inventory differs from regeneration.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.domain_graph import (  # noqa: E402
    load_manifest,
    module_name,
    tracked_population,
)

ARCHITECTURE = REPO_ROOT / "docs" / "ARCHITECTURE.md"
OUT_DIR = REPO_ROOT / "docs" / "v7next"
FROZEN_OUT = OUT_DIR / "FROZEN_CONTRACTS_INVENTORY.md"
LAYOUT_OUT = OUT_DIR / "DATA_LAYOUT_INVENTORY.md"
FACADE_OUT = OUT_DIR / "FACADE_INVENTORY.md"

_PATH_SPAN = re.compile(
    r"^(?:ouroboros|supervisor|web|tests|scripts|docs|prompts|skills)/[\w./-]+\.\w+")
_CODE_SPAN = re.compile(r"`([^`]+)`")


def _tracked_all() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return set(out.split())


def _split_row(line: str) -> list[str]:
    """Split one markdown table row on unescaped pipes (adoption-validator
    convention)."""
    body = line.strip().strip("|")
    cells, cur, escaped = [], [], False
    for ch in body:
        if escaped:
            cur.append(ch)
            escaped = False
        elif ch == "\\":
            cur.append(ch)
            escaped = True
        elif ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


# ---------------------------------------------------------------------------
# 1. Frozen contracts (ARCHITECTURE §11.1)
# ---------------------------------------------------------------------------

def frozen_section_text(arch_text: str) -> str:
    m = re.search(r"^### 11\.1 What is frozen$(.*?)^### 11\.2 ",
                  arch_text, re.MULTILINE | re.DOTALL)
    if not m:
        raise ValueError("ARCHITECTURE.md has no `### 11.1 What is frozen` section")
    return m.group(1)


def _row_paths(cell: str) -> list[str]:
    paths = []
    for span in _CODE_SPAN.findall(cell):
        span = span.replace("\\|", "|")
        base = span.split("::", 1)[0]
        if _PATH_SPAN.match(base):
            paths.append(span)
    return paths


def _row_label(cell: str) -> str:
    m = _CODE_SPAN.search(cell)
    if m:
        return m.group(1)
    return cell.split(" — ", 1)[0].split(".", 1)[0].strip()[:80]


def parse_frozen_rows(section: str) -> list[dict]:
    rows = []
    lines = section.splitlines()
    in_table = False
    for line in lines:
        if line.startswith("|"):
            cells = _split_row(line)
            if [c.lower() for c in cells[:3]] == ["contract", "file", "anchored by"]:
                in_table = True
                continue
            if in_table and set(line.replace("|", "").strip()) <= {"-", " "}:
                continue
            if in_table and len(cells) >= 3:
                rows.append({
                    "label": _row_label(cells[0]),
                    "owners": _row_paths(cells[1]),
                    "anchors": _row_paths(cells[2]),
                })
        else:
            in_table = False
    return rows


def build_frozen_inventory() -> tuple[str, list[str]]:
    """Returns (rendered document, hard findings)."""
    arch_text = ARCHITECTURE.read_text(encoding="utf-8")
    section = frozen_section_text(arch_text)
    rows = parse_frozen_rows(section)
    tracked = _tracked_all()
    findings: list[str] = []

    # The prose paragraph above the table declares the browser-envelope ABI.
    prose = section.split("| Contract |", 1)[0]
    prose_paths = sorted({p for p in _row_paths(prose)})

    def resolve(span: str) -> str:
        base = span.split("::", 1)[0]
        if base in tracked:
            return "ok"
        findings.append(f"frozen: §11.1 references missing file: {span}")
        return "MISSING"

    contracts_pkg = sorted(p for p in tracked_population(REPO_ROOT)
                           if p.startswith("ouroboros/contracts/")
                           and not p.endswith("__init__.py"))
    referenced = set()
    for p in contracts_pkg:
        if p in section:
            referenced.add(p)
    uncovered = [p for p in contracts_pkg if p not in referenced]

    L: list[str] = []
    L.append("# Frozen-contracts inventory (generated)")
    L.append("")
    L.append("Machine extraction of `docs/ARCHITECTURE.md` §11.1 (the frozen-ABI"
             " SSOT), regenerated by `python scripts/regenerate_inventories.py`."
             " Do not edit — edit §11.1 and regenerate;"
             " `tests/test_generated_inventories.py` pins byte-identity and the"
             " resolution invariants (a §11.1 row whose owner or anchor file"
             " disappeared from the tree = red).")
    L.append("")
    L.append(f"- table rows: **{len(rows)}**")
    L.append(f"- browser-envelope prose owners: {', '.join(f'`{p}`' for p in prose_paths)}")
    L.append("")
    L.append("| # | contract | owner files | anchored by |")
    L.append("|---:|---|---|---|")
    for i, row in enumerate(rows, 1):
        owners = "<br>".join(f"`{p}` ({resolve(p)})" for p in row["owners"]) or "—"
        anchors = "<br>".join(f"`{p}` ({resolve(p)})" for p in row["anchors"]) or "—"
        label = row["label"].replace("|", "\\|")
        L.append(f"| {i} | `{label}` | {owners} | {anchors} |")
    for p in prose_paths:
        resolve(p)
    L.append("")
    L.append("## Package coverage")
    L.append("")
    L.append(f"`ouroboros/contracts/` modules (excluding `__init__.py`): {len(contracts_pkg)};"
             f" referenced by §11.1: {len(referenced)}.")
    if uncovered:
        L.append("")
        L.append("Contracts-package modules NOT referenced by §11.1 (a frozen-package"
                 " module without a frozen-table row — every new one must either get"
                 " a §11.1 row or be a deliberate, here-visible gap):")
        L.append("")
        for p in uncovered:
            L.append(f"- `{p}`")
    else:
        L.append("")
        L.append("Every contracts-package module is referenced by §11.1.")
    L.append("")
    return "\n".join(L), findings


# ---------------------------------------------------------------------------
# 2. Data layout (ARCHITECTURE "Data layout" tree)
# ---------------------------------------------------------------------------

_TREE_ENTRY = re.compile(r"[├└]──\s+(.+?)(?:\s+←.*)?$")


def layout_block(arch_text: str) -> str:
    m = re.search(r"^### Data layout \(`~/Ouroboros/`\)$.*?```(.*?)```",
                  arch_text, re.MULTILINE | re.DOTALL)
    if not m:
        raise ValueError("ARCHITECTURE.md has no Data layout fenced tree")
    return m.group(1)


def parse_layout_entries(block: str) -> list[str]:
    entries = []
    for line in block.splitlines():
        m = _TREE_ENTRY.search(line)
        if m:
            entries.append(m.group(1).strip())
    return entries


def _probe_token(entry: str) -> str | None:
    segments = [s for s in entry.strip("/").split("/") if s]
    literal = [s for s in segments if "<" not in s and ">" not in s]
    return literal[-1] if literal else None


def build_layout_inventory() -> tuple[str, list[str]]:
    arch_text = ARCHITECTURE.read_text(encoding="utf-8")
    entries = parse_layout_entries(layout_block(arch_text))
    tracked = _tracked_all()
    tracked_dirs: set[str] = set()
    for p in tracked:
        parts = p.split("/")
        for i in range(1, len(parts)):
            tracked_dirs.add("/".join(parts[:i]))

    runtime_blob = "\n".join(
        (REPO_ROOT / p).read_text(encoding="utf-8", errors="replace")
        for p in sorted(tracked_population(REPO_ROOT)))

    findings: list[str] = []
    rows: list[tuple[str, str, str]] = []
    for entry in entries:
        probe = _probe_token(entry)
        if probe is None:
            rows.append((entry, "—", "placeholder"))
            continue
        clean = probe.strip("/")
        if any(f == clean or f.endswith("/" + clean) for f in tracked) :
            rows.append((entry, clean, "repo-path"))
        elif clean in tracked_dirs or any(d.endswith("/" + clean) for d in tracked_dirs):
            rows.append((entry, clean, "repo-dir"))
        elif clean in runtime_blob:
            rows.append((entry, clean, "code-ref"))
        else:
            rows.append((entry, clean, "UNRESOLVED"))
            findings.append(
                f"layout: tree entry `{entry}` (probe `{clean}`) resolves neither as a"
                " tracked repo path nor as a literal in the runtime sources")

    L: list[str] = []
    L.append("# Data-layout inventory (generated)")
    L.append("")
    L.append("Machine extraction of the `docs/ARCHITECTURE.md` \"Data layout"
             " (`~/Ouroboros/`)\" tree — the durable-file orientation carrier"
             " (this tree's counterpart of the reference PERSISTENCE_OWNERS"
             " derivation checklist) — regenerated by"
             " `python scripts/regenerate_inventories.py`. Do not edit."
             " Every entry is probed against reality: repo entries must exist as"
             " tracked paths; data-plane entries must appear as a literal in the"
             " runtime sources that construct them. A durable file renamed or"
             " removed in code while its tree row survives = red"
             " (`tests/test_generated_inventories.py`).")
    L.append("")
    n_kinds = {}
    for _, _, kind in rows:
        n_kinds[kind] = n_kinds.get(kind, 0) + 1
    L.append(f"- entries: **{len(rows)}** ({', '.join(f'{k}: {v}' for k, v in sorted(n_kinds.items()))})")
    L.append("")
    L.append("| entry | probe | resolution |")
    L.append("|---|---|---|")
    for entry, probe, kind in rows:
        esc = entry.replace("|", "\\|")
        L.append(f"| `{esc}` | `{probe}` | {kind} |")
    L.append("")
    return "\n".join(L), findings


# ---------------------------------------------------------------------------
# 3. Facade inventory (AST, noqa: F401 re-export marker)
# ---------------------------------------------------------------------------

_NOQA_F401 = re.compile(r"#\s*noqa(?::[^#]*\bF401\b|\s*$|:\s*$)")


def _statement_has_noqa_f401(source_lines: list[str], node: ast.ImportFrom) -> bool:
    end = getattr(node, "end_lineno", node.lineno) or node.lineno
    for lineno in range(node.lineno, end + 1):
        if lineno - 1 < len(source_lines) and _NOQA_F401.search(source_lines[lineno - 1]):
            return True
    return False


def build_facade_inventory() -> tuple[str, list[str]]:
    manifest = load_manifest()
    mod_by_name = {module_name(p): p for p in manifest.modules}

    def resolve(dotted: str) -> str | None:
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if cand in mod_by_name:
                return mod_by_name[cand]
        return None

    findings: list[str] = []
    facades: list[dict] = []
    for path in sorted(manifest.modules):
        f = REPO_ROOT / path
        if not f.is_file():
            continue
        source = f.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=path)
        leaves: dict[str, int] = {}
        n_names = 0
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if not _statement_has_noqa_f401(lines, node):
                continue
            base = ("." * node.level) + (node.module or "")
            importer = module_name(path)
            if base.startswith("."):
                from scripts.domain_graph import resolve_relative
                base = resolve_relative(base, importer, path.endswith("__init__.py"))
            for alias in node.names:
                if alias.name == "*":
                    leaf = resolve(base)
                else:
                    leaf = resolve(f"{base}.{alias.name}") or resolve(base)
                if leaf is None or leaf == path:
                    continue
                leaves[leaf] = leaves.get(leaf, 0) + 1
                n_names += 1
        if leaves:
            facades.append({"path": path, "leaves": leaves, "names": n_names})

    L: list[str] = []
    L.append("# Facade inventory (generated)")
    L.append("")
    L.append("AST-derived inventory of compatibility facades, regenerated by"
             " `python scripts/regenerate_inventories.py`. Do not edit."
             " A facade row is any runtime module whose top-level"
             " `from <population module> import ...` statements carry the"
             " `noqa: F401` re-export marker — the codebase's declared"
             " \"this binding exists for its binding, not for this module's own"
             " use\" convention (reference FACADE_CONSUMERS method). Leaf"
             " domains come from `ouroboros/domains.toml`; a leaf outside the"
             " facade's domain is marked ✗ (that edge also appears in the"
             " manifest's pinned direction matrix)."
             " `tests/test_generated_inventories.py` pins byte-identity, so any"
             " re-export surface change must regenerate this file.")
    L.append("")
    n_cross = sum(1 for row in facades for leaf in row["leaves"]
                  if manifest.modules[leaf] != manifest.modules[row["path"]])
    L.append(f"- facade modules: **{len(facades)}**;"
             f" marked re-export bindings: **{sum(r['names'] for r in facades)}**;"
             f" cross-domain facade→leaf pairs: **{n_cross}**")
    L.append("")
    L.append("| facade | domain | bindings | leaves |")
    L.append("|---|---|---:|---|")
    for row in facades:
        dom = manifest.modules[row["path"]]
        leaf_cells = []
        for leaf in sorted(row["leaves"]):
            ldom = manifest.modules[leaf]
            mark = "" if ldom == dom else f" ✗{ldom}"
            leaf_cells.append(f"`{leaf}` ({row['leaves'][leaf]}{mark})")
        L.append(f"| `{row['path']}` | {dom} | {row['names']} | {'<br>'.join(leaf_cells)} |")
    L.append("")
    return "\n".join(L), findings


# ---------------------------------------------------------------------------

BUILDERS = {
    FROZEN_OUT: build_frozen_inventory,
    LAYOUT_OUT: build_layout_inventory,
    FACADE_OUT: build_facade_inventory,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the on-disk inventories match regeneration (no writes)")
    args = ap.parse_args(argv)

    stale: list[str] = []
    all_findings: list[str] = []
    for out_path, builder in BUILDERS.items():
        try:
            rendered, findings = builder()
        except (OSError, ValueError) as exc:
            print(f"structural failure building {out_path.name}: {exc}", file=sys.stderr)
            return 2
        rendered += "\n" if not rendered.endswith("\n") else ""
        all_findings.extend(findings)
        rel = out_path.relative_to(REPO_ROOT)
        if args.check:
            on_disk = out_path.read_text(encoding="utf-8") if out_path.is_file() else None
            if on_disk != rendered:
                stale.append(str(rel))
        else:
            out_path.write_text(rendered, encoding="utf-8")
            print(f"wrote {rel}")

    for f in all_findings:
        print(f"RED: {f}", file=sys.stderr)
    if args.check and stale:
        for rel in stale:
            print(f"RED: stale inventory (regeneration changes it): {rel}", file=sys.stderr)
    return 1 if (all_findings or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
