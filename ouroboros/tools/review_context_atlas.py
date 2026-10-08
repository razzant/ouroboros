"""The compact repository index a retrieving reviewer's brief carries.

A reviewer that reaches the repository with its own read-only tools needs a MAP,
not a copy: where every tracked path is, which class it belongs to, and — for
the paths this change touches — their structural facts and the files that import
them. Bodies are never rendered here; the reviewer opens any path in full with
``read_file``.

Two parts, both deterministic and both derived from the same per-file facts:

* the COVERAGE INDEX — one row for every tracked path: a bare ``path`` for the
  ordinary ``indexed`` class (the overwhelming majority, so the constant label is
  not repeated a thousand times), ``disposition<TAB>path`` for every other class,
  with the policy-excluded classes (tests, non-agent-logic directories, binary
  media, vendored/minified files) collapsed to one
  ``disposition<TAB>directory/ (N files)`` row per directory, so the whole tree
  is accounted for in a few thousand rows instead of a full-detail dump;
* the CHANGE-RELATIVE ROWS — size, digest, language, symbols, imports and the
  direct importers, for the touched paths and for the importers themselves. The
  importer list is the brief's answer to "who calls this", bounded per touched
  path with the total disclosed, so a hub module cannot flood the index.

No budgets, no admission, no model call and no writes: the index scales with the
repository, is the same text for the same tree, and never decides what a
reviewer may see.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from ouroboros.tools.review_helpers import (
    _FULL_REPO_BINARY_EXTENSIONS,
    _FULL_REPO_SKIP_DIR_PREFIXES,
    _MAX_FULL_REPO_FILE_BYTES,
    _SENSITIVE_EXTENSIONS,
    _SENSITIVE_NAMES,
    _VENDORED_NAMES,
    _VENDORED_SUFFIXES,
    _is_probably_binary,
    format_prompt_code_block,
    list_git_tracked_paths,
)
from ouroboros.utils import estimate_tokens

REPOSITORY_INDEX_SCHEMA_VERSION = 1

# Per-row caps. Symbols and imports are orientation, not an inventory; the
# importer cap keeps a hub module (a helpers leaf with hundreds of importers)
# from crowding the brief out of its first send. Every cap discloses its total.
INDEX_MAX_ROW_SYMBOLS = 12
INDEX_MAX_ROW_IMPORTS = 12
INDEX_MAX_TOUCHED_IMPORTERS = 40

# Dispositions whose per-path rows carry no orientation value: the coverage
# index collapses them to one `disposition<TAB>directory/ (N files)` row per
# directory. A touched path and a listed importer keep their own row whatever
# their class.
_COLLAPSED_INDEX_DISPOSITIONS = frozenset({
    "excluded_test",
    "excluded_dir",
    "binary_media",
    "vendored_minified",
})

_PYTHON_SUFFIX = ".py"
_JS_SUFFIXES = (".js", ".mjs", ".ts", ".tsx", ".jsx")


@dataclass
class _FileFacts:
    """One tracked path's index facts. Bodies are read, never kept."""

    rel_path: str
    disposition: str = "indexed"
    size_bytes: int = 0
    sha256: str = ""
    language: str = ""
    symbols: tuple[str, ...] = ()
    symbol_count: int = 0
    imports: tuple[str, ...] = ()
    import_count: int = 0
    # Tracked paths this file imports, resolved from the same facts. The edge
    # set is what makes `imported_by` computable without a second walk.
    import_targets: tuple[str, ...] = ()
    reason: str = ""


def repository_index(
    repo_dir: pathlib.Path,
    *,
    touched_paths: Sequence[str],
    tracked_paths: Sequence[str] | None = None,
) -> tuple[str, dict]:
    """``(index text, manifest)`` for one tree and one change.

    ``touched_paths`` decides which paths get change-relative rows; it never
    changes what the coverage index accounts for. ``tracked_paths`` overrides
    the ``git ls-files`` walk (tests and callers that already hold the list).
    """
    repo_dir = pathlib.Path(repo_dir)
    touched = tuple(
        path for path in dict.fromkeys(_normalize_path(rel) for rel in touched_paths) if path
    )
    tracked = [
        rel
        for rel in dict.fromkeys(
            _normalize_path(path)
            for path in (
                tracked_paths if tracked_paths is not None else list_git_tracked_paths(repo_dir)
            )
        )
        if rel
    ]
    touched_set = frozenset(touched)
    tracked_set = frozenset(tracked)

    facts_by_path = {
        rel: _build_file_facts(repo_dir, rel, exempt=rel in touched_set) for rel in tracked
    }
    _resolve_import_targets(facts_by_path, tracked_set)

    importers_by_path: dict[str, list[str]] = {}
    for rel in sorted(facts_by_path):
        for target in facts_by_path[rel].import_targets:
            importers_by_path.setdefault(target, []).append(rel)

    listed_importers: dict[str, tuple[str, ...]] = {}
    importer_totals: dict[str, int] = {}
    for rel in touched:
        every = importers_by_path.get(rel) or []
        importer_totals[rel] = len(every)
        listed_importers[rel] = tuple(every[:INDEX_MAX_TOUCHED_IMPORTERS])

    importer_paths = tuple(
        sorted({rel for listed in listed_importers.values() for rel in listed} - touched_set)
    )
    detail_paths = touched_set | set(importer_paths)
    # A listed importer is related to the change, so the two POLICY exclusions
    # lift for it exactly as they lift for a touched path — the row the index
    # prints and the class it prints cannot disagree.
    for rel in importer_paths:
        facts = facts_by_path[rel]
        if facts.disposition in {"excluded_test", "excluded_dir"}:
            facts.disposition = "indexed"
            facts.reason = "imports a touched path"

    touched_rows = [
        _index_row(
            facts_by_path.get(rel) or _FileFacts(
                rel_path=rel, disposition="untracked",
                reason="touched path is not in the tracked set of this tree"),
            imported_by=listed_importers.get(rel, ()),
            imported_by_total=importer_totals.get(rel, 0),
        )
        for rel in touched
    ]
    importer_rows = [
        _index_row(
            facts_by_path[rel],
            imported_by=tuple(
                (importers_by_path.get(rel) or [])[:INDEX_MAX_TOUCHED_IMPORTERS]
            ),
            imported_by_total=len(importers_by_path.get(rel) or []),
        )
        for rel in importer_paths
    ]

    counts = Counter(facts.disposition for facts in facts_by_path.values())
    text = _render_index_text(
        facts_by_path, detail_paths=detail_paths,
        touched_rows=touched_rows, importer_rows=importer_rows,
    )
    manifest = {
        "schema_version": REPOSITORY_INDEX_SCHEMA_VERSION,
        "strategy": "repository_index",
        "tracked_count": len(facts_by_path),
        "dispositions": dict(sorted(counts.items())),
        "touched_count": len(touched_rows),
        "importer_count": len(importer_rows),
        "importer_cap_per_touched_path": INDEX_MAX_TOUCHED_IMPORTERS,
        "touched": touched_rows,
        "importers": importer_rows,
        "index_chars": len(text),
        "tokens_estimate": int(estimate_tokens(text)),
    }
    return text, manifest


def _build_file_facts(repo_dir: pathlib.Path, rel: str, *, exempt: bool) -> _FileFacts:
    """One path's disposition and structural facts.

    ``exempt`` lifts the two POLICY exclusions (wider tests, non-agent-logic
    directories) for a path the change touches: a touched file is related to the
    change by definition, so the index describes it per path. The physical
    classes (sensitive, binary, vendored, oversized) are facts about the file
    and are reported as they are.

    Import facts are extracted for every readable source file whatever its
    index class, because the importer edges the brief needs come from files the
    coverage index collapses (a test, a devtools script) as well as from the
    indexed ones.
    """
    facts = _FileFacts(
        rel_path=rel, language=pathlib.PurePosixPath(rel).suffix.lstrip(".")
    )
    path = repo_dir / rel
    try:
        path.resolve().relative_to(repo_dir.resolve())
    except (OSError, ValueError):
        facts.disposition = "path_escape"
        facts.reason = "path escapes repository root"
        return facts
    if not path.is_file():
        facts.disposition = "missing"
        facts.reason = "tracked path is not a regular file"
        return facts
    try:
        raw = path.read_bytes()
    except OSError as exc:
        facts.disposition = "read_error"
        facts.reason = f"read error: {exc}"
        return facts
    facts.size_bytes = len(raw)
    facts.sha256 = hashlib.sha256(raw).hexdigest()[:16]

    suffix = pathlib.PurePosixPath(rel).suffix.lower()
    fname = pathlib.PurePosixPath(rel).name.lower()
    if fname in _SENSITIVE_NAMES or suffix in _SENSITIVE_EXTENSIONS:
        facts.disposition = "sensitive"
        facts.reason = "sensitive filename or extension"
        facts.size_bytes, facts.sha256 = 0, ""
        return facts
    if suffix in _FULL_REPO_BINARY_EXTENSIONS:
        facts.disposition = "binary_media"
        facts.reason = "binary/media extension"
        return facts
    if fname in _VENDORED_NAMES or any(fname.endswith(tail) for tail in _VENDORED_SUFFIXES):
        facts.disposition = "vendored_minified"
        facts.reason = "vendored or minified file"
        return facts
    if facts.size_bytes > _MAX_FULL_REPO_FILE_BYTES:
        facts.disposition = "oversized"
        facts.reason = f">{_MAX_FULL_REPO_FILE_BYTES // 1024}KB"
        return facts
    if _is_probably_binary(path):
        facts.disposition = "binary_media"
        facts.reason = "binary content"
        return facts

    if rel.startswith("tests/") and not exempt:
        facts.disposition = "excluded_test"
        facts.reason = "wider tests carry no per-path index row"
    elif rel.startswith(_FULL_REPO_SKIP_DIR_PREFIXES) and not exempt:
        facts.disposition = "excluded_dir"
        facts.reason = "non-agent-logic directory"
    else:
        facts.reason = "text source file"

    if suffix == _PYTHON_SUFFIX or suffix in _JS_SUFFIXES:
        content = raw.decode("utf-8", errors="replace")
        if suffix == _PYTHON_SUFFIX:
            _extract_python_facts(rel, content, facts)
        else:
            _extract_js_facts(rel, content, facts)
    return facts


def _extract_python_facts(rel: str, content: str, facts: _FileFacts) -> None:
    """Top-level symbols, imported modules, and the import edge candidates.

    The displayed imports are the module names the file names; the edge
    candidates additionally carry ``from package import module`` (the dominant
    form in this repository), so an importer of ``a.b.c`` is found whether it
    wrote ``import a.b.c`` or ``from a.b import c``.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return
    symbols: set[str] = set()
    imports: set[str] = set()
    candidates: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            symbols.update(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id.isupper()
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper():
                symbols.add(node.target.id)
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                candidates.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from_module(rel, node)
            if module:
                imports.add(module)
                candidates.add(module)
                candidates.update(f"{module}.{alias.name}" for alias in node.names)
    facts.symbols = tuple(sorted(symbols)[:INDEX_MAX_ROW_SYMBOLS])
    facts.symbol_count = len(symbols)
    facts.imports = tuple(sorted(imports)[:INDEX_MAX_ROW_IMPORTS])
    facts.import_count = len(imports)
    facts.import_targets = tuple(sorted(candidates))


def _extract_js_facts(rel: str, content: str, facts: _FileFacts) -> None:
    """Relative module specifiers resolved to the paths they could name."""
    from ouroboros.code_intelligence import extract_js_imports

    parent = pathlib.PurePosixPath(rel).parent
    found: set[str] = set()
    for spec in extract_js_imports(content):
        if not spec.startswith("."):
            continue
        base = (parent / spec).as_posix()
        found.add(base)
        if not pathlib.PurePosixPath(base).suffix:
            found.update({f"{base}.js", f"{base}.mjs", f"{base}/index.js"})
    facts.imports = tuple(sorted(found)[:INDEX_MAX_ROW_IMPORTS])
    facts.import_count = len(found)
    facts.import_targets = tuple(sorted(found))


def _resolve_import_targets(
    facts_by_path: dict[str, _FileFacts], tracked: frozenset[str]
) -> None:
    """Rewrite every file's edge candidates to the TRACKED paths they name.

    A Python candidate resolves through the module-name map; a JS candidate is
    already a path. Anything that names nothing tracked (a third-party package,
    a generated bundle) is dropped, so the importer rows only ever point at
    files the reviewer can open.
    """
    module_to_path = {}
    for rel in facts_by_path:
        module = _python_module_name(rel)
        if module:
            module_to_path[module] = rel
    for facts in facts_by_path.values():
        resolved = {
            module_to_path.get(candidate, candidate if candidate in tracked else "")
            for candidate in facts.import_targets
        }
        facts.import_targets = tuple(sorted(rel for rel in resolved if rel and rel != facts.rel_path))


def _python_module_name(rel: str) -> str:
    if not rel.endswith(".py"):
        return ""
    path = rel[:-3]
    if path.endswith("/__init__"):
        path = path[: -len("/__init__")]
    return path.replace("/", ".")


def _resolve_import_from_module(rel: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    current = _python_module_name(rel)
    package_parts = current.split(".")[:-1] if current else []
    base = package_parts[: max(0, len(package_parts) - (node.level - 1))]
    if node.module:
        return ".".join([*base, *node.module.split(".")])
    if node.names:
        return ".".join([*base, node.names[0].name])
    return ".".join(base)


def _index_row(
    facts: _FileFacts, *, imported_by: tuple[str, ...], imported_by_total: int
) -> dict:
    """One change-relative row: the facts a body cannot supply on its own."""
    row = {
        "path": facts.rel_path,
        "disposition": facts.disposition,
        "reason": facts.reason,
        "size": facts.size_bytes,
        "sha256": facts.sha256,
        "language": facts.language,
        "imported_by": list(imported_by),
        "imported_by_total": imported_by_total,
    }
    if facts.symbols:
        row["symbols"] = list(facts.symbols)
        row["symbols_total"] = facts.symbol_count
    if facts.imports:
        row["imports"] = list(facts.imports)
        row["imports_total"] = facts.import_count
    return row


def _render_row(row: dict) -> str:
    lines = [
        f"- {row['path']} — {int(row['size']):,} bytes"
        + (f" — {row['language']}" if row["language"] else "")
        + (f" — sha256={row['sha256']}" if row["sha256"] else "")
        + f" — {row['disposition']}"
    ]
    for key, label in (("symbols", "symbols"), ("imports", "imports")):
        listed = row.get(key) or []
        if listed:
            total = int(row.get(f"{key}_total") or len(listed))
            more = f" (+{total - len(listed)} more of {total})" if total > len(listed) else ""
            lines.append(f"  {label}: {', '.join(listed)}{more}")
    listed = row.get("imported_by") or []
    total = int(row.get("imported_by_total") or 0)
    if listed:
        more = f" (+{total - len(listed)} more of {total})" if total > len(listed) else ""
        lines.append(f"  imported_by: {', '.join(listed)}{more}")
    else:
        lines.append("  imported_by: (no tracked importer)")
    return "\n".join(lines)


def _render_index_text(
    facts_by_path: dict[str, _FileFacts],
    *,
    detail_paths: set[str],
    touched_rows: list[dict],
    importer_rows: list[dict],
) -> str:
    detailed: list[tuple[str, str]] = []
    collapsed: Counter[tuple[str, str]] = Counter()
    for rel, facts in facts_by_path.items():
        if facts.disposition in _COLLAPSED_INDEX_DISPOSITIONS and rel not in detail_paths:
            parent = str(pathlib.PurePosixPath(rel).parent)
            collapsed[(facts.disposition, "" if parent == "." else parent)] += 1
        else:
            detailed.append((facts.disposition, rel))
    # The default class is the whole point of the map, so it is the bare path;
    # only a non-default class spends a label on its row.
    index_lines = [rel if disposition == "indexed" else f"{disposition}\t{rel}"
                   for disposition, rel in sorted(detailed)]
    index_lines += [
        f"{disposition}\t{directory}/ ({count} files)"
        for (disposition, directory), count in sorted(collapsed.items())
    ]

    parts = [
        "## Repository index",
        "",
        "Candidate structural facts; bodies via `read_file`.",
        "",
        "### Coverage index",
        "",
        f"{len(facts_by_path):,} tracked paths: bare = readable `indexed`; else "
        "`disposition<TAB>path`. excluded_test/excluded_dir/binary_media/"
        "vendored_minified collapse to directory/count rows, except touched "
        "paths and listed importers. Read any path; this is not an access filter.",
        "",
        format_prompt_code_block("\n".join(index_lines), "text"),
        "",
        "### Touched paths",
        "",
        "Size/SHA/language, top-level symbols, imports, tracked importers "
        f"(limit {INDEX_MAX_TOUCHED_IMPORTERS}; totals disclosed).",
        "",
        "\n".join(_render_row(row) for row in touched_rows) or "(no touched paths)",
        "",
        "### Direct importers of the touched paths",
        "",
        "\n".join(_render_row(row) for row in importer_rows)
        or "(no tracked file imports a touched path)",
        "",
    ]
    return "\n".join(parts)


def _normalize_path(path: str) -> str:
    cleaned = str(path or "").strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return pathlib.PurePosixPath(cleaned).as_posix() if cleaned else ""
