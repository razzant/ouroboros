"""Every MIGRATION_v7.md row that calls its move verbatim is held to it byte-for-byte.

A "verbatim" semantic-delta note is a claim about source text: the declaration that
lived at the old identity in the merge base is the same bytes as the declaration at
the new identity now. Reviewers proved that mechanically for one SHA; this pins the
property so a later split cannot drift (reflowed lines, dropped comments, widened
annotations) while the ledger still says verbatim. A row that intentionally changes
text must say so in its note instead of "verbatim" — that is the point.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import re
import subprocess
import textwrap

REPO = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v7_migration_verbatim", REPO / "scripts" / "v7_migration.py")
assert SPEC is not None and SPEC.loader is not None
v7_migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v7_migration)

_VERBATIM_NOTE = re.compile(r"\bverbatim\b", re.IGNORECASE)


def _split_ref(ref: str) -> tuple[str, str]:
    path, _, symbol = ref.partition("::")
    return path.strip(), symbol.strip()


def _source_at_ref(ref: str, path: str, cache: dict[tuple[str, str], str | None]) -> str | None:
    key = (ref, path)
    if key not in cache:
        proc = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True, text=True, encoding="utf-8")
        cache[key] = proc.stdout if proc.returncode == 0 else None
    return cache[key]


def _python_declaration_lines(source: str, qualname: str) -> str | None:
    """Whole-line text of the exactly-one declaration bound to ``qualname`` (decorators included)."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    found: list[ast.AST] = []

    def walk(body: list[ast.stmt], scope: tuple[str, ...], in_class: bool = False) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ".".join((*scope, node.name)) == qualname:
                    found.append(node)
                walk(node.body, (*scope, node.name), isinstance(node, ast.ClassDef))
            elif (not scope or in_class) and isinstance(node, (ast.Assign, ast.AnnAssign)):
                # Module-scope assignments AND class-body attribute assignments are
                # declarations (the llm mixin split relocated 15 class attributes);
                # function-body assignments never are. Duplicates fail via len(found).
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                found.extend(node for name in names if ".".join((*scope, name)) == qualname)

    walk(tree.body, (), False)
    if len(found) != 1:
        return None
    node = found[0]
    first = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
    return "".join(lines[first - 1 : node.end_lineno])


def _javascript_declaration_text(source: str, qualname: str) -> str | None:
    node = v7_migration._js_declaration_node(source, qualname)
    return node.text.decode("utf-8", "replace") if node is not None else None


def _declaration_text(path: str, source: str, symbol: str) -> str | None:
    if path.endswith(".py"):
        return _python_declaration_lines(source, symbol)
    if path.endswith(".js"):
        return _javascript_declaration_text(source, symbol)
    return None


def test_every_verbatim_ledger_row_moves_byte_identical_source() -> None:
    rows = v7_migration._parse_migration(REPO / "MIGRATION_v7.md")
    cache: dict[tuple[str, str], str | None] = {}
    checked = 0
    checked_javascript = 0
    problems: list[str] = []
    for row in rows:
        try:
            note = str(json.loads(row["semantic delta"]).get("note", ""))
        except (ValueError, AttributeError):
            continue
        if not _VERBATIM_NOTE.search(note):
            continue
        old_ref, new_ref = row["old path/symbol"], row["new owner/path"]
        if new_ref.startswith(("retired:", "external:")):
            continue
        old_path, old_symbol = _split_ref(old_ref)
        new_path, new_symbol = _split_ref(new_ref)
        if not old_symbol or not new_symbol or (old_path, old_symbol) == (new_path, new_symbol):
            continue  # facade-only or identity rows carry no moved text
        old_source = _source_at_ref(v7_migration.MERGE_BASE_SHA, old_path, cache)
        new_file = REPO / new_path
        if old_source is None or not new_file.is_file():
            continue  # resolution and existence are validate_migration's job
        old_text = _declaration_text(old_path, old_source, old_symbol)
        new_text = _declaration_text(new_path, new_file.read_text(encoding="utf-8"), new_symbol)
        if old_text is None or new_text is None:
            continue  # unresolvable declarations are reported by check-migration, not here
        checked += 1
        checked_javascript += new_path.endswith(".js")
        # A declaration that left a class body for module scope legitimately loses one
        # indentation level; everything else about its text must survive the move.
        if textwrap.dedent(old_text) != textwrap.dedent(new_text):
            problems.append(f"{old_ref} -> {new_ref}: note says verbatim but the declaration text differs")
    # Both resolvers must stay live: a silent regression in either one turns its rows
    # into skips, and a single total would let the JavaScript half vanish unnoticed.
    # Floors sit just under the measured coverage (2610 compared in total at the
    # v6.104.0 merge adoption: 2522 Python + 88 JS), not two orders below it: a
    # resolver regression that silently turned most rows into skips would otherwise
    # stay green. `checked` gates the TOTAL, `checked_javascript` the JS subset.
    # The ledger only grows, so raise these when coverage grows; lower them only
    # with a phase that deliberately shrinks the ledger.
    assert checked > 2500, f"the verbatim pin should cover the extraction rows; only {checked} compared"
    assert checked_javascript > 80, f"the JavaScript rows stopped resolving; only {checked_javascript} compared"
    assert problems == [], "\n".join(problems)
