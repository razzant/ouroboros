#!/usr/bin/env python3
"""Domain manifest gate (plan §7.1, CPL-1) over ``ouroboros/domains.toml``.

Checks (exit 0 green, 1 findings, 2 structurally broken inputs):

1.  **Completeness** — the manifest's ``[modules]`` section equals the tracked
    population (every runtime module has exactly one domain row; a tracked
    module without a row = red, a stale row = red).
2.  **Direction matrix** — the strict (unconditional module-level) cross-domain
    dependency pairs of the live tree equal the pinned ``[graph].allowed``
    baseline. The baseline is today's FACTUAL matrix, not an aspiration
    (plan §7.1: current reality = baseline; tightening = a separate owner
    decision). A new direction = red until deliberately regenerated, so the
    diff of this data file is where the decision becomes visible.
3.  **Cycles** — the strict quotient's cycle groups (SCCs > 1 domain) equal
    the pinned ``[graph].cycle_groups`` ceiling. A cycle group that grows, or
    a new group, is red with its exact witnesses; the terminal target is
    ``cycle_groups = []`` — zero cycles on domain nodes.
4.  **Lazy/dynamic classification** — cross-domain pairs reachable ONLY
    through function-level imports (``lazy_only``) and through resolved
    dynamic imports (``dynamic_pairs``) equal their pinned inventories.
5.  **Literal-copy ban** — normalized function bodies appearing in more than
    one domain equal the pinned ``[duplicates].allowed`` baseline. "Fixing" a
    cycle by copying the body across the boundary instead of importing it is
    exactly what turns this red.
6.  **DOMAIN_MAP.md** — the generated map equals what the manifest renders
    (gen/verify pair: regeneration must not change the committed file).

``--write`` regenerates the manifest's generated sections and
``docs/DOMAIN_MAP.md`` from the live tree; the resulting diff is the reviewed
decision surface. ``tests/test_domain_manifest.py`` runs the same checks as
the CI verify half of the pair.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.domain_graph import (  # noqa: E402
    DOMAIN_MAP_PATH,
    MANIFEST_PATH,
    Manifest,
    build_import_graph,
    computed_graph_sections,
    duplicate_bodies,
    load_manifest,
    tracked_population,
)

GENERATED_BEGIN = "# --- BEGIN GENERATED (python scripts/check_domains.py --write) ---"
GENERATED_END = "# --- END GENERATED ---"


def _toml_str_list(values: list[str], indent: str = "  ") -> str:
    if not values:
        return "[]"
    body = "\n".join(f'{indent}"{v}",' for v in values)
    return f"[\n{body}\n]"


def _toml_group_list(groups: list[list[str]]) -> str:
    if not groups:
        return "[]"
    rows = []
    for group in groups:
        inner = ", ".join(f'"{d}"' for d in group)
        rows.append(f"  [{inner}],")
    return "[\n" + "\n".join(rows) + "\n]"


def render_generated_block(sections: dict[str, list], duplicates: list[str]) -> str:
    L: list[str] = [GENERATED_BEGIN]
    L.append("# Factual dependency data of the live tree, pinned as baseline data.")
    L.append("# Regenerate with `python scripts/check_domains.py --write`; the diff of")
    L.append("# this block is the reviewed decision surface for every new direction,")
    L.append("# cycle change, hidden coupling, or cross-domain literal copy.")
    L.append("")
    L.append("[graph]")
    L.append("# Strict cross-domain dependency directions (from->to). Absent = forbidden.")
    L.append(f"allowed = {_toml_str_list(sections['allowed'])}")
    L.append("# Domains locked in strict-quotient cycles (SCC ceiling; the target is []).")
    L.append(f"cycle_groups = {_toml_group_list(sections['cycle_groups'])}")
    L.append("# Cross-domain pairs reachable ONLY through function-level (lazy) imports —")
    L.append("# hidden coupling, classified out of the strict graph.")
    L.append(f"lazy_only = {_toml_str_list(sections['lazy_only'])}")
    L.append("# Cross-domain pairs created by resolved dynamic imports (importlib/__import__).")
    L.append(f"dynamic_pairs = {_toml_str_list(sections['dynamic_pairs'])}")
    L.append("")
    L.append("[duplicates]")
    L.append("# Literal-copy baseline: normalized function bodies (>= 10 normalized lines)")
    L.append("# present in modules of MORE THAN ONE domain. A new row is a banned")
    L.append('# literal-copy "fix" of a dependency edge; shrinking this list is progress.')
    L.append(f"allowed = {_toml_str_list(duplicates)}")
    L.append(GENERATED_END)
    return "\n".join(L) + "\n"


def manifest_with_generated_block(raw_text: str, block: str) -> str:
    begin = raw_text.find(GENERATED_BEGIN)
    if begin == -1:
        base = raw_text.rstrip("\n")
        return f"{base}\n\n{block}"
    end = raw_text.find(GENERATED_END, begin)
    if end == -1:
        raise ValueError("manifest has BEGIN GENERATED marker but no END marker")
    end += len(GENERATED_END)
    tail = raw_text[end:].lstrip("\n")
    head = raw_text[:begin].rstrip("\n")
    out = f"{head}\n\n{block}"
    if tail:
        out += "\n" + tail
    return out


def render_domain_map(manifest: Manifest) -> str:
    """docs/DOMAIN_MAP.md — rendered from the manifest ONLY (no live-tree
    reads), so it drifts exactly when the manifest drifts."""
    mods_by_domain: dict[str, list[str]] = {d: [] for d in sorted(manifest.domains)}
    for path in sorted(manifest.modules):
        mods_by_domain[manifest.modules[path]].append(path)

    L: list[str] = []
    L.append("# Domain map — v7next")
    L.append("")
    L.append("Generated from `ouroboros/domains.toml` by `python scripts/check_domains.py"
             " --write`. Do not edit — edit the manifest and regenerate;"
             " `tests/test_domain_manifest.py` pins byte-identity.")
    L.append("")
    L.append("The manifest is the SSOT of the module→domain assignment (1:1, complete"
             " over the tracked runtime population) and pins today's factual"
             " cross-domain dependency data as baseline; the gate contract is"
             " described in `scripts/check_domains.py`.")
    L.append("")
    L.append("## Index")
    L.append("")
    L.append("| domain | name | modules | proposed |")
    L.append("|---|---|---:|---:|")
    total = 0
    total_prop = 0
    for d in sorted(manifest.domains):
        n = len(mods_by_domain[d])
        n_prop = sum(1 for p in mods_by_domain[d] if p in manifest.proposed)
        total += n
        total_prop += n_prop
        L.append(f"| {d} | {manifest.domains[d]} | {n} | {n_prop} |")
    L.append(f"| **total** | | **{total}** | **{total_prop}** |")
    L.append("")

    used = sorted(manifest.domains)
    allowed_pairs = {tuple(p.split("->")) for p in manifest.graph_allowed}
    L.append("## Dependency direction matrix (strict, pinned)")
    L.append("")
    L.append("Rows may import columns (`[graph].allowed`). `·` = forbidden direction.")
    L.append("")
    L.append("| ↓ imports → | " + " | ".join(used) + " |")
    L.append("|---|" + "---|" * len(used))
    for d1 in used:
        row = [f"| **{d1}**"]
        for d2 in used:
            row.append("✓" if (d1, d2) in allowed_pairs else "·")
        L.append(" | ".join(row) + " |")
    L.append("")

    L.append("## Cycle status")
    L.append("")
    if not manifest.cycle_groups:
        L.append("The strict domain quotient is **acyclic** (`cycle_groups = []`).")
    else:
        L.append(f"{len(manifest.cycle_groups)} pinned cycle group(s) — the SCC ceiling;"
                 " the target is zero. Witness-level detail lives in"
                 " `docs/v7next/DOMAIN_QUOTIENT_REPORT.md`.")
        L.append("")
        for i, group in enumerate(manifest.cycle_groups, 1):
            L.append(f"- group {i} ({len(group)} domains): {' ⇄ '.join(group)}")
    L.append("")

    L.append("## Hidden coupling (classified out of the strict graph)")
    L.append("")
    L.append(f"- lazy-only cross-domain pairs: **{len(manifest.lazy_only)}**")
    for p in manifest.lazy_only:
        L.append(f"  - {p}")
    L.append(f"- dynamic-import cross-domain pairs: **{len(manifest.dynamic_pairs)}**")
    for p in manifest.dynamic_pairs:
        L.append(f"  - {p}")
    L.append("")

    L.append("## Literal-copy baseline")
    L.append("")
    if not manifest.duplicates_allowed:
        L.append("No function body (≥ 10 normalized lines) is shared verbatim across"
                 " domains. New occurrences turn the gate red.")
    else:
        L.append(f"{len(manifest.duplicates_allowed)} pinned cross-domain literal copies"
                 " (baseline; shrinking is progress, new rows are red):")
        L.append("")
        for row in manifest.duplicates_allowed:
            digest, _, locs = row.partition(" ")
            L.append(f"- `{digest}`: " + " · ".join(f"`{loc}`" for loc in locs.split()))
    L.append("")

    L.append("## Modules by domain")
    L.append("")
    L.append("`*` marks a `classification=proposed` row (owner review pending).")
    for d in sorted(manifest.domains):
        L.append("")
        L.append(f"### {d} — {manifest.domains[d]}")
        L.append("")
        for path in mods_by_domain[d]:
            mark = " *" if path in manifest.proposed else ""
            L.append(f"- `{path}`{mark}")
    L.append("")
    return "\n".join(L)


def run_checks(root: pathlib.Path = REPO_ROOT) -> tuple[list[str], list[str]]:
    """Returns (findings, notes). Structural problems raise."""
    manifest = load_manifest()
    findings: list[str] = []
    notes: list[str] = []

    bad_domains = sorted({d for d in manifest.modules.values() if d not in manifest.domains})
    if bad_domains:
        findings.append(f"manifest names unknown domains: {bad_domains}")
        return findings, notes

    tracked = tracked_population(root)
    for path in sorted(tracked - set(manifest.modules)):
        findings.append(f"completeness: tracked module has no manifest row: {path}")
    for path in sorted(set(manifest.modules) - tracked):
        findings.append(f"completeness: manifest row is not a tracked module: {path}")
    if findings:
        # The graph below is only meaningful over a complete population.
        return findings, notes

    graph = build_import_graph(manifest, root)
    sections = computed_graph_sections(manifest, graph)

    actual_allowed = set(sections["allowed"])
    pinned_allowed = set(manifest.graph_allowed)
    strict_pairs = graph.quotient()
    for pair in sorted(actual_allowed - pinned_allowed):
        d1, d2 = pair.split("->")
        witnesses = strict_pairs.get((d1, d2), [])[:5]
        wit = "; ".join(f"{s} -> {t}" for s, t in witnesses)
        findings.append(
            f"direction: NEW cross-domain dependency {pair} is not in [graph].allowed"
            f" (witnesses: {wit}) — a new direction is an owner decision; regenerate"
            " with --write to make it visible in the manifest diff")
    for pair in sorted(pinned_allowed - actual_allowed):
        findings.append(
            f"direction: [graph].allowed pins {pair} but the tree no longer has it —"
            " regenerate with --write to shrink the baseline")

    actual_cycles = sections["cycle_groups"]
    if actual_cycles != manifest.cycle_groups:
        findings.append(
            f"cycles: computed cycle groups {actual_cycles} != pinned"
            f" {manifest.cycle_groups} — growth is red (break the new edge instead);"
            " shrinkage must be banked with --write")
    if not actual_cycles:
        notes.append("cycles: the strict domain quotient is acyclic (target reached)")
    else:
        n = sum(len(g) for g in actual_cycles)
        notes.append(f"cycles: {len(actual_cycles)} pinned group(s), {n} domains still cyclic")

    if sections["lazy_only"] != manifest.lazy_only:
        findings.append(
            f"classification: lazy_only pairs drifted (computed {len(sections['lazy_only'])}"
            f" != pinned {len(manifest.lazy_only)}) — regenerate with --write")
    if sections["dynamic_pairs"] != manifest.dynamic_pairs:
        findings.append(
            f"classification: dynamic_pairs drifted (computed {len(sections['dynamic_pairs'])}"
            f" != pinned {len(manifest.dynamic_pairs)}) — regenerate with --write")

    dups = duplicate_bodies(manifest, root)
    pinned_dups = set(manifest.duplicates_allowed)
    for row in dups:
        if row not in pinned_dups:
            findings.append(
                f"literal-copy: cross-domain duplicate body not in baseline: {row} —"
                " import the single owner instead of copying the body across domains")
    for row in sorted(pinned_dups - set(dups)):
        findings.append(
            f"literal-copy: baseline row no longer observed: {row} — regenerate with"
            " --write to bank the shrink")

    rendered = render_domain_map(manifest)
    if not DOMAIN_MAP_PATH.is_file():
        findings.append(f"domain map: {DOMAIN_MAP_PATH.name} is missing — run --write")
    elif DOMAIN_MAP_PATH.read_text(encoding="utf-8") != rendered:
        findings.append(
            "domain map: docs/DOMAIN_MAP.md is stale (regeneration changes it) — run --write")
    return findings, notes


def write_generated(root: pathlib.Path = REPO_ROOT) -> None:
    manifest = load_manifest()
    tracked = tracked_population(root)
    missing = sorted(tracked - set(manifest.modules))
    stale = sorted(set(manifest.modules) - tracked)
    if missing or stale:
        raise SystemExit(
            "--write refuses on an incomplete [modules] section (the assignment is a"
            f" human decision, never generated): missing rows {missing}, stale rows {stale}")
    graph = build_import_graph(manifest, root)
    sections = computed_graph_sections(manifest, graph)
    dups = duplicate_bodies(manifest, root)
    block = render_generated_block(sections, dups)
    text = manifest.path.read_text(encoding="utf-8")
    manifest.path.write_text(manifest_with_generated_block(text, block), encoding="utf-8")
    # Re-load so the map renders the freshly pinned data.
    DOMAIN_MAP_PATH.write_text(render_domain_map(load_manifest()), encoding="utf-8")
    print(f"wrote {manifest.path.relative_to(root)} generated sections and "
          f"{DOMAIN_MAP_PATH.relative_to(root)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="regenerate the manifest's generated sections and docs/DOMAIN_MAP.md")
    args = ap.parse_args(argv)

    if not MANIFEST_PATH.is_file():
        print(f"missing manifest: {MANIFEST_PATH}", file=sys.stderr)
        return 2

    if args.write:
        write_generated()
        return 0

    try:
        findings, notes = run_checks()
    except (OSError, SyntaxError, ValueError, KeyError) as exc:
        print(f"structural failure: {exc}", file=sys.stderr)
        return 2
    for note in notes:
        print(note)
    if findings:
        for f in findings:
            print(f"RED: {f}", file=sys.stderr)
        return 1
    print("OK: domain manifest complete; directions, cycles, classification,"
          " literal-copy baseline and DOMAIN_MAP.md all match the tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
