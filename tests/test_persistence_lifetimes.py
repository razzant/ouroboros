"""Every durable plane in docs/PERSISTENCE.md names its lifetime (owner decision 2026-10-17, 2A).

A row's Retention cell starts with its class:

- ``[record]``: kept as history or authority;
- ``[bounded]``: its size cannot grow with work (overwritten in place, capped, rotated);
- ``[disposable: package.module.qualname]``: a copy, cache or scratch that grows with work,
  and the production function that retires it.

A disposable plane that nothing retires yet says ``[disposable: none]`` and is listed in
``DISPOSABLE_WITHOUT_RETIRER``, which may only shrink except for the owner-approved shared
package-download cache (2026-10-21, quiz 44dd72): no homemade background cleaner; maintenance
uses package-manager commands, with no indefinite-retention or economy promise. Other new
growing planes name what ends them. Retirers resolve through source AST, never imports.
"""
from __future__ import annotations

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "PERSISTENCE.md"
_LIFETIME_RE = re.compile(r"^\[(record|bounded|disposable: ([\w.]+))\]")

# Disposable planes nothing retires yet: only removals, except the cache decision named above.
DISPOSABLE_WITHOUT_RETIRER = frozenset({
    "state/presence_turn_gate/last-<sha256>.json",
    "state/betterleaks/",
    "state/skills/<name>/jobs/<job>/",
    "playwright-browsers/",
    # Owner choice A (quiz 44dd72, 2026-10-21): one shared download cache, no bespoke cleaner; it is
    # pruned only explicitly, with the package manager (uv cache prune, pip cache purge, npm cache clean).
    "package-downloads/<manager>/",
})


def _cells(line: str) -> list:
    cells, current, code = [], [], False
    for char in line.strip().strip("|"):
        if char == "`":
            code = not code
        if char == "|" and not code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    return [*cells, "".join(current).strip()]


def inventory_rows(text: str) -> list:
    """``(key, Retention cell)`` of every body row of a table whose header names Retention,
    however its first cell starts; the key is that cell's first code span (its first path)."""
    rows, retention = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            retention = None
            continue
        cells = _cells(stripped)
        if retention is None:  # a header row
            retention = cells.index("Retention") if "Retention" in cells else None
        elif not set(stripped) <= set("|-: "):  # not the separator row
            spans = cells[0].split("`")
            rows.append((spans[1] if len(spans) > 2 else cells[0], cells[retention]))
    return rows


def lifetime(cell: str):
    """``(class, retirer)`` named at the start of a Retention cell, or ``None``."""
    match = _LIFETIME_RE.match(cell)
    if not match:
        return None
    return ("disposable", match.group(2)) if match.group(2) else (match.group(1), None)


def resolves(dotted: str) -> bool:
    """A dotted path names a module of this repository and a def or class inside it."""
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        module = REPO.joinpath(*parts[:cut]).with_suffix(".py")
        if not module.is_file():
            continue
        scope = ast.parse(module.read_text(encoding="utf-8"))
        for name in parts[cut:]:
            scope = next((node for node in getattr(scope, "body", ()) if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name), None)
            if scope is None:
                return False
        return True
    return False


ROWS = inventory_rows(DOC.read_text(encoding="utf-8"))


def test_every_row_names_its_lifetime():
    from ouroboros.code_intelligence_architecture import _persistence_rows

    assert len(ROWS) == len(_persistence_rows(REPO)) > 100, "every inventory row is read, prose-led ones too"
    missing = [path for path, cell in ROWS if lifetime(cell) is None]
    assert not missing, ("start these rows' Retention cell with [record], [bounded] or "
                         f"[disposable: package.module.function]: {missing}")


def test_every_retirer_is_a_real_function():
    unresolved = [(path, retirer) for path, cell in ROWS
                  for kind, retirer in [lifetime(cell) or ("", None)]
                  if kind == "disposable" and retirer != "none" and not resolves(retirer)]
    assert not unresolved, f"retirers that name no function in this repository: {unresolved}"


def test_disposable_planes_without_a_retirer_only_shrink():
    debts = {path for path, cell in ROWS if lifetime(cell) == ("disposable", "none")}
    assert debts == DISPOSABLE_WITHOUT_RETIRER, (
        "a plane that grows with work names the function that ends it; when one gains a retirer, "
        f"drop it from DISPOSABLE_WITHOUT_RETIRER. Added: {sorted(debts - DISPOSABLE_WITHOUT_RETIRER)}; "
        f"retired: {sorted(DISPOSABLE_WITHOUT_RETIRER - debts)}")


def test_a_row_whose_first_cell_starts_with_prose_is_read():
    table = ("| Path | Writer | Retention | Reset |\n|---|---|---|---|\n"
             "| task documents: `tasks` rows | store | [record] kept | lost |\n| `a/b` | w | none | r |\n\n"
             "| Name | Notes |\n|---|---|\n| `x` | no retention column |\n")
    assert inventory_rows(table) == [("tasks", "[record] kept"), ("a/b", "none")]


def test_a_fabricated_row_is_caught_and_a_real_retirer_resolves():
    assert lifetime("kept indefinitely") is None
    kind, retirer = lifetime("[disposable: ouroboros.review_checking_prune.no_such_function] copies")
    assert kind == "disposable" and not resolves(retirer)
    assert resolves("ouroboros.review_checking_prune.prune_review_checkouts")
    assert resolves("ouroboros.review_checking.CheckingAreaUnavailable")
    assert not resolves("ouroboros.no_such_module.prune")
    assert lifetime("[record] chat history") == ("record", None)


def test_exact_cut_and_room_reader_have_size_and_retention_rows():
    text = DOC.read_text()
    for token in ('<digest>-<sha256>.json` (physical cut', 'room-<digest>-<sha256>.json'):
        row = next(line for line in text.splitlines() if token in line)
        assert '[record]' in row and 'pack' in row and 'MB' in row
        assert 'no independent remover' in row
