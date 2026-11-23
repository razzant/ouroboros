#!/usr/bin/env python3
"""Regenerate the tracked UI control inventory or show a source-derived inventory.

--check verifies the UI snapshot and validates the three source inventories
without writing. --show renders one of those inventories to stdout: their
book and code sources, not saved copies of the reports, are authoritative.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ouroboros.code_intelligence_architecture import (  # noqa: E402
    FROZEN_SECTION,
    parse_frozen_rows,
    scan_facade_bindings,
)
from ouroboros.markdown_source import parse_markdown_source  # noqa: E402
from ouroboros.reference_books import (  # noqa: E402
    BookView,
    ReferenceBook,
    load_reference_book,
    read_book_section,
)
from scripts.domain_graph import load_manifest, tracked_population  # noqa: E402
from scripts.ui_control_inventory import UI_CONTROLS_OUT, build_ui_control_inventory  # noqa: E402

LAYOUT_SECTION = "Data layout (`~/Ouroboros/`)"
_TREE_ENTRY = re.compile(r"[├└]──\s+(.+?)(?:\s+←.*)?$")


def _tracked_all() -> set[str]:
    """Present candidate files, including unignored files not staged yet."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         cwd=REPO_ROOT, capture_output=True, check=True).stdout
    return {p for p in out.decode("utf-8", "surrogateescape").split("\0") if p and (REPO_ROOT / p).is_file()}


def _candidate_population() -> list[str]:
    return sorted(p for p in tracked_population(REPO_ROOT, untracked=True) if (REPO_ROOT / p).is_file())


def _chapter_source_note(book: ReferenceBook, view: BookView) -> list[str]:
    if book.legacy:
        return []
    ref = view.sources[0]
    return [f"Source: `{ref.path}`, physical LF lines {ref.span.start_line}-{ref.span.end_line}; "
            f"UTF-8 SHA-256 `{ref.sha256}`.", ""]


def build_frozen_inventory(book: ReferenceBook | None = None) -> tuple[str, list[str]]:
    """Render §11.1 and check every named file, including prose owners."""
    book = book or load_reference_book(REPO_ROOT, "architecture")
    view = read_book_section(book, FROZEN_SECTION)
    prose_owners, rows = parse_frozen_rows(view.text)
    tracked = _tracked_all()
    findings: list[str] = []

    def resolve(span: str) -> str:
        if span.split("::", 1)[0] in tracked:
            return "ok"
        findings.append(f"frozen: §11.1 references missing file: {span}")
        return "MISSING"

    contracts_pkg = [p for p in _candidate_population()
                     if p.startswith("ouroboros/contracts/") and not p.endswith("__init__.py")]
    uncovered = [p for p in contracts_pkg if p not in view.text]
    lines = ["# Frozen-contracts inventory (generated)", "",
             "Source-derived report of ARCHITECTURE §11.1 (owners and anchoring tests).", "",
             *_chapter_source_note(book, view),
             f"- table rows: **{len(rows)}**",
             "- browser-envelope prose owners: " + ", ".join(f"`{p}`" for p in prose_owners), "",
             "| # | contract | owner files | anchored by |", "|---:|---|---|---|"]
    for i, row in enumerate(rows, 1):
        owners = "<br>".join(f"`{p}` ({resolve(p)})" for p in row.owners) or "—"
        anchors = "<br>".join(f"`{p}` ({resolve(p)})" for p in row.anchors) or "—"
        label = row.label.replace("|", "\\|")
        lines.append(f"| {i} | `{label}` | {owners} | {anchors} |")
    for p in prose_owners:
        resolve(p)
    lines.extend(["", "## Package coverage", "",
                  f"`ouroboros/contracts/` modules (excluding `__init__.py`): {len(contracts_pkg)};"
                  f" referenced by §11.1: {len(contracts_pkg) - len(uncovered)}.", ""])
    if uncovered:
        lines.extend(["Contracts-package modules NOT referenced by §11.1:", ""])
        lines.extend(f"- `{p}`" for p in uncovered)
    else:
        lines.append("Every contracts-package module is referenced by §11.1.")
    return "\n".join(lines) + "\n", findings


def layout_block(arch_text: str) -> str:
    source = parse_markdown_source(arch_text.encode("utf-8"), "docs/ARCHITECTURE.md")
    sections = [h for h in source.headings if h.title == LAYOUT_SECTION]
    if len(sections) != 1:
        raise ValueError("Architecture needs one Data layout section")
    section = sections[0].section
    blocks = [b for b in source.code_blocks if section.start_byte <= b.start_byte < section.end_byte]
    if len(blocks) != 1:
        raise ValueError("Architecture Data layout needs one fenced tree")
    return source.text_at(blocks[0])


def parse_layout_entries(block: str) -> list[str]:
    return [m.group(1).strip() for line in block.splitlines() if (m := _TREE_ENTRY.search(line))]


def _probe_token(entry: str) -> str | None:
    literal = [s for s in entry.strip("/").split("/") if s and "<" not in s and ">" not in s]
    return literal[-1] if literal else None


def probe_layout_entries(entries: list[str]) -> list[tuple[str, str, str]]:
    """Probe each last literal segment against candidate paths or runtime text."""
    tracked = _tracked_all()
    tracked_dirs = {parent.name for path in tracked for parent in pathlib.PurePosixPath(path).parents}
    tracked_names = {pathlib.PurePosixPath(path).name for path in tracked}
    runtime_blob = "\n".join(
        (REPO_ROOT / p).read_text(encoding="utf-8", errors="replace") for p in _candidate_population())
    rows = []
    for entry in entries:
        probe = _probe_token(entry)
        if probe is None:
            rows.append((entry, "—", "placeholder"))
        elif probe in tracked_names:
            rows.append((entry, probe, "repo-path"))
        elif probe in tracked_dirs:
            rows.append((entry, probe, "repo-dir"))
        elif probe in runtime_blob:
            rows.append((entry, probe, "code-ref"))
        else:
            rows.append((entry, probe, "UNRESOLVED"))
    return rows


def build_layout_inventory(book: ReferenceBook | None = None) -> tuple[str, list[str]]:
    book = book or load_reference_book(REPO_ROOT, "architecture")
    view = read_book_section(book, LAYOUT_SECTION)
    entries = parse_layout_entries(layout_block(view.text))
    if not entries:
        raise ValueError("Architecture Data layout tree has no entries")
    rows = probe_layout_entries(entries)
    findings = [f"layout: tree entry `{entry}` (probe `{probe}`) resolves neither as a"
                " tracked repo path nor as a literal in the runtime sources"
                for entry, probe, kind in rows if kind == "UNRESOLVED"]
    lines = ["# Data-layout inventory (generated)", "",
             "Source-derived report from the ARCHITECTURE Data layout tree.", "",
             *_chapter_source_note(book, view)]
    counts: dict[str, int] = {}
    for _, _, kind in rows:
        counts[kind] = counts.get(kind, 0) + 1
    lines.append(f"- entries: **{len(rows)}** ({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))})")
    lines.extend(["", "| entry | probe | resolution |", "|---|---|---|"])
    for entry, probe, kind in rows:
        escaped = entry.replace("|", "\\|")
        lines.append(f"| `{escaped}` | `{probe}` | {kind} |")
    return "\n".join(lines) + "\n", findings


def build_facade_inventory() -> tuple[str, list[str]]:
    manifest = load_manifest(REPO_ROOT / "ouroboros/domains.toml")
    facades: dict[str, dict[str, int]] = {}
    for facade, _, leaf in scan_facade_bindings(REPO_ROOT, manifest.modules, strict_syntax=True):
        counts = facades.setdefault(facade, {})
        counts[leaf] = counts.get(leaf, 0) + 1
    domains = manifest.modules
    bindings = sum(sum(leaves.values()) for leaves in facades.values())
    cross = sum(domains[leaf] != domains[facade] for facade, leaves in facades.items() for leaf in leaves)
    lines = ["# Facade inventory (generated)", "",
             "Top-level `noqa: F401` re-exports, derived from current Python sources and the domain manifest.", "",
             f"- facade modules: **{len(facades)}**; marked re-export bindings: **{bindings}**;"
             f" cross-domain facade→leaf pairs: **{cross}**", "",
             "| facade | domain | bindings | leaves |", "|---|---|---:|---|"]
    for facade, leaves in facades.items():
        domain = domains[facade]
        cells = []
        for leaf, count in sorted(leaves.items()):
            mark = "" if domains[leaf] == domain else f" ✗{domains[leaf]}"
            cells.append(f"`{leaf}` ({count}{mark})")
        lines.append(f"| `{facade}` | {domain} | {sum(leaves.values())} | {'<br>'.join(cells)} |")
    return "\n".join(lines) + "\n", []


_REPORTS = {"frozen": build_frozen_inventory, "data-layout": build_layout_inventory,
            "facades": build_facade_inventory}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = ap.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="check source reports and the UI snapshot, without writes")
    action.add_argument("--write", action="store_true", help="regenerate the UI snapshot (also the default)")
    action.add_argument("--show", choices=_REPORTS, help="render a source inventory to stdout without writes")
    args = ap.parse_args(argv)
    findings: list[str] = []
    try:
        if args.show:
            rendered, findings = _REPORTS[args.show]()
            print(rendered, end="")
        else:
            rendered, findings = build_ui_control_inventory()
            if args.check:
                on_disk = UI_CONTROLS_OUT.read_text(encoding="utf-8") if UI_CONTROLS_OUT.is_file() else None
                if on_disk != rendered:
                    findings.append(f"stale inventory: {UI_CONTROLS_OUT.relative_to(REPO_ROOT)}")
                for build in _REPORTS.values():
                    findings.extend(build()[1])
            else:
                UI_CONTROLS_OUT.parent.mkdir(parents=True, exist_ok=True)
                UI_CONTROLS_OUT.write_text(rendered, encoding="utf-8", newline="\n")
                print(f"wrote {UI_CONTROLS_OUT.relative_to(REPO_ROOT)}")
    except (OSError, SyntaxError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"structural failure building inventory: {exc}", file=sys.stderr)
        return 2
    for finding in findings:
        print(f"RED: {finding}", file=sys.stderr)
    return int(bool(findings))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    sys.exit(main())
