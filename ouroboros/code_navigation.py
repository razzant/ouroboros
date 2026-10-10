"""Inventory views over current source evidence; no symbol identity inference.

Local outlines remain useful orientation. Occurrence views deliberately retain
same-name collisions and expose aliases for another query. Import paths are
filesystem candidates at every hop, never compiler-resolved dependency edges.
"""
from __future__ import annotations

import pathlib
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable

from ouroboros.code_import_candidates import ImportCandidateLookup, import_candidates
from ouroboros.code_intelligence import (
    CodeInventory,
    relevant_files,
    render_digest_entry,
    symbol_definitions,
)
from ouroboros.code_occurrences import MAX_EVIDENCE_ROWS, EvidenceScan, grammar_id, in_scope, scan_occurrences


@dataclass
class NavigationView:
    rows: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    incomplete: bool = False

    def scanned(self, scan: EvidenceScan) -> None:
        self.notes.append(scan.summary())
        self.limits.extend(f"{reason}: {count}" for reason, count in sorted(scan.limits.items()))
        self.incomplete |= scan.incomplete


def _candidate_label(kind: str, hits: list[str]) -> str:
    return (f"unique candidate ({kind}: {hits[0]})" if len(hits) == 1 else
            f"ambiguous candidate ({kind}, {len(hits)}: {', '.join(hits)})")


def _impact(view, inventory, query, scope, selector, depth, options):
    paths = {file.path for file in inventory.files}
    is_file = query in paths or "/" in query or pathlib.PurePosixPath(query).suffix in {
        ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
        ".go", ".rs", ".java", ".rb", ".c", ".h", ".cpp", ".cs",
    }
    if is_file:
        if selector and selector != query:
            raise ValueError("impact target comes from query; file path conflicts with that target")
        frontier = {query}
        view.notes.append(f"target: file {query}; import specifier candidates, not semantic dependencies")
        if query not in paths:
            view.limits.append("target absent from current inventory; no filesystem candidate can name it")
    else:
        definition_inventory = replace(inventory, files=[f for f in inventory.files
            if options["lang"] == "any" or grammar_id(f.language) == grammar_id(options["lang"])])
        definitions = symbol_definitions(definition_inventory, query, path=selector or scope)
        frontier = {file.path for file, _ in definitions}
        occurrence_counts = {}

        def occurrence_file(rows):
            if not rows:
                return []
            occurrence_counts[rows[0].path] = len(rows)
            # A file summary must not bury a useful call behind a leading
            # comment/import. Keep its actual syntax/source anchor, not a link
            # inferred from the selected definition.
            priority = {"call?": 0, "import?": 1, "def?": 2}
            return [min(rows, key=lambda r: (priority.get(r.hint, 4 if r.slot == "text" else 3),
                                            r.line, r.column))]

        scan = scan_occurrences(inventory, query, scope=scope, select_rows=occurrence_file, **options)
        view.scanned(scan)
        for row in scan.rows:
            count = occurrence_counts[row.path]
            view.rows.append(row.render(f"symbol occurrence evidence ({count} rows; representative per file); not bound to {query}"))
        view.notes.append(f"target: symbol {query}; {len(definitions)} outline definitions; occurrence files plus import candidates")
        if not definitions:
            view.notes.append("no outline definition found; import expansion has no initial definition file")
    if not frontier:
        return
    # Recompute against the entire current path set, even when importer search is
    # scoped. Creating/removing a target cannot leave an unchanged importer stale.
    lookup = ImportCandidateLookup(paths)
    candidates_by_target = {}
    observed = unresolved = 0

    def join_imports(rows):
        nonlocal observed, unresolved
        for row in rows:
            if time.monotonic() >= options["deadline"]:
                break  # scan_occurrences discloses the row-selection deadline
            kind, hits = import_candidates(row.specifier, row.path, lookup, row.language)
            observed += 1
            unresolved += not hits
            for target in hits:
                candidates_by_target.setdefault(target, []).append((row, kind, hits))
        # Intermediate import facts are not selected impact results. They spend
        # the existing source/time budget, not the selected-view row quota.
        return []

    imports = scan_occurrences(inventory, None, mode="imports", scope=scope,
                               select_rows=join_imports, **options)
    view.scanned(imports)
    view.notes.append(f"import specifiers: {observed} observed, {unresolved} without filesystem candidates; unsupported syntax may be absent")
    visited = set(frontier)
    seen_rows = set()
    expanded = 0
    for hop in range(1, max(1, min(5, depth)) + 1):
        if not frontier:
            break
        expanded = hop
        next_frontier = set()
        candidates = {}
        for target in sorted(frontier):
            for row, kind, hits in candidates_by_target.get(target, ()):
                candidates[(row.path, row.line, row.column, row.specifier)] = (row, kind, hits)
        for identity, (row, kind, hits) in sorted(candidates.items()):
            if time.monotonic() >= options["deadline"]:
                view.limits.append("import candidate expansion time limit")
                view.incomplete = True
                break
            via = sorted(set(hits) & frontier)
            if not via:
                continue
            if identity not in seen_rows:
                if len(view.rows) >= MAX_EVIDENCE_ROWS:
                    view.limits.append(f"selected row cap {MAX_EVIDENCE_ROWS}")
                    view.incomplete = True
                    return
                label = f"{_candidate_label(kind, hits)} hop {hop} via {', '.join(via)} specifier={row.specifier!r}"
                view.rows.append(row.render(label))
                seen_rows.add(identity)
            if len(hits) == 1 and row.path not in visited:
                next_frontier.add(row.path)
        visited.update(next_frontier)
        frontier = next_frontier
    view.notes.append(f"depth={max(1, min(5, depth))}; {expanded} hops examined; only unique candidates expand, every hop remains heuristic")


def inventory_view(inventory: CodeInventory, *, op: str, query: str = "", path: str = "",
                   lang: str = "any", kind: str = "any", depth: int = 1,
                   wall_seconds: float = 45, deadline: float | None = None,
                   path_allowed: Callable | None = None) -> NavigationView:
    root = pathlib.Path(inventory.repo_root)
    universe = inventory
    selector = path if path and (root / path).is_file() else ""
    scope = "" if selector and op in {"references", "callers", "impact"} else path
    files = [f for f in inventory.files if lang == "any" or grammar_id(f.language) == grammar_id(lang)]
    inventory = replace(inventory, files=files)
    scoped = replace(inventory, files=[f for f in files if in_scope(f.path, scope)])
    view = NavigationView()
    view.notes.append(f"scope: {scope or '.'} ({'file' if scope == selector and selector else 'directory/root'}); lang={lang}")
    view.notes.append("universe: inventory paths (Git ignores and inventory exclusions; nested repositories expanded)")
    view.limits.extend(inventory.enumeration_limits)
    view.incomplete = any("limit" in note or "error" in note for note in inventory.enumeration_limits)
    nested = [p for p in inventory.nested_roots if in_scope(p, scope) or in_scope(scope, p)]
    if nested:
        view.notes.append(f"nested repositories: {len(nested)} included ({', '.join(nested[:20])})")
    options = dict(lang=lang, deadline=deadline if deadline is not None else time.monotonic() + wall_seconds,
                   path_allowed=path_allowed)
    if selector and op in {"references", "callers", "impact"}:
        view.notes.append(f"file selector: {selector}; occurrence/importer search from root; no binding inferred")
    if op in {"digest", "relevant_files", "symbols", "definition"}:
        coverage = Counter(f.disposition for f in scoped.files)
        view.notes.append(f"files in scope: {len(scoped.files)}; coverage: " + ", ".join(f"{k}={v}" for k, v in sorted(coverage.items())))
        view.limits.extend(f"{k}: {v}" for k, v in sorted(coverage.items()) if k != "indexed")
        errors = sum(bool(f.syntax_error) for f in scoped.files)
        if errors:
            view.limits.append(f"syntax errors: {errors}; outline may be partial or unavailable")
    if op == "digest":
        view.notes.append(f"method: local outline/imports/routes/calls; head={inventory.git_head[:12] or 'unknown'}")
        groups = {}
        for file in scoped.files:
            relative = file.path[len(scope.rstrip('/') + '/'):] if scope else file.path
            if "/" in relative:
                groups.setdefault(relative.split("/", 1)[0], []).append(file)
        for name, group in sorted(groups.items())[:50]:
            languages = ",".join(sorted({f.language for f in group}))
            dispositions = ",".join(f"{k}={v}" for k, v in sorted(Counter(f.disposition for f in group).items()))
            view.notes.append(f"directory {name}/: {len(group)} files, {sum(len(f.symbols) for f in group)} symbols; {languages}; {dispositions}")
        if len(groups) > 50:
            view.limits.append("directory rollups capped at 50; flat file entries remain pageable")
        view.rows = [render_digest_entry(f) for f in sorted(scoped.files, key=lambda f: f.path)]
    elif op == "relevant_files":
        view.notes.append("method: path/symbol/import/route word ranking; scope applied before scoring")
        for i, (file, score, reason) in enumerate(relevant_files(scoped, query, limit=None), 1):
            names = ", ".join(s.name for s in file.symbols[:5])
            view.rows.append(f"{i}. {file.path} score={score:.2f} reason={reason}{' symbols=' + names if names else ''}")
    elif op in {"symbols", "definition"}:
        definitions = symbol_definitions(scoped, query, kind=kind)
        view.rows = [f"{f.path}:{s.line_start} {s.kind} {s.signature or s.name} [outline]" for f, s in definitions]
        view.notes.append(f"method: Python ast / tree-sitter outline; {len(definitions)} outline rows; grammar-qualified declaration kinds only")
        if op == "definition" and kind == "any":
            scan = scan_occurrences(scoped, query, mode="definition", **options)
            covered = {(f.path, s.line_start) for f, s in definitions}
            candidates = [r for r in scan.rows if (r.path, r.line) not in covered]
            view.rows.extend(r.render("name-field candidate, not a declaration assertion") for r in candidates)
            view.notes.append(f"{len(candidates)} additional name-field candidates")
            view.scanned(scan)
    elif op in {"references", "callers", "callees"}:
        ranges = None
        token = query
        if op == "callees":
            definitions = symbol_definitions(scoped, query)
            ranges = {}
            for file, symbol in definitions:
                ranges.setdefault(file.path, []).append((symbol.line_start, symbol.line_end))
                view.notes.append(f"definition range: {file.path}:{symbol.line_start}-{symbol.line_end} {symbol.name}")
            view.notes.append(f"{len(definitions)} outline definitions; calls inside their source ranges")
            token = None
        scan = scan_occurrences(inventory, token, mode=op, scope=scope, ranges=ranges, **options)
        view.scanned(scan)
        if op == "callers":
            view.notes.append(f"syntactic callee position, not resolved binding; {scan.occurrences} token occurrences observed before caller filtering")
        if selector and op != "callees":
            definitions = symbol_definitions(inventory, query)
            competing = {f.path for f, _ in definitions}
            view.notes.append(f"{len(definitions)} outline declarations share this name; selected file {selector}")
            # Rank only; never drop competing-name evidence or synthesize aliases.
            likely = set()
            current_paths = ImportCandidateLookup(f.path for f in universe.files)
            ordering_stopped = False
            for file in files:
                if time.monotonic() >= options["deadline"]:
                    ordering_stopped = True
                    break
                for spec in file.imports:
                    if time.monotonic() >= options["deadline"]:
                        ordering_stopped = True
                        break
                    if selector in import_candidates(spec, file.path, current_paths, file.language)[1]:
                        likely.add(file.path)
                if ordering_stopped:
                    break
            if ordering_stopped:
                view.limits.append("import candidate ordering time limit; partial heuristic order")
            scan.rows.sort(key=lambda r: (0 if r.path == selector else 1 if r.path in likely else 2 if r.path in competing else 3,
                                          r.path, r.line, r.column))
            view.notes.append("order: heuristic (selected file, raw import candidates, other outline declarations, others)")
            view.notes.append(f"{sum(r.path != selector for r in scan.rows)} rows outside selected file")
        view.rows = [row.render() for row in scan.rows]
    elif op == "impact":
        # lang scopes evidence rows, not the candidate universe: hiding a JS
        # sibling during a TS query would falsely promote ambiguity to unique.
        _impact(view, universe, query, scope, selector, depth, options)
    return view
