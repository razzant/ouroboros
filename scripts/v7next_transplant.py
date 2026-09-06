#!/usr/bin/env python3
"""Mechanical transplant tool for the v7next module split (D18/D33 handle idiom).

Given (a) an upstream source file, (b) the symbols that move to a leaf module,
and (c) the declared parent-owned name set for that leaf, emit the leaf source
whose moved bodies are byte-identical to upstream except that every call-time
``Load`` of a declared name is rewritten ``NAME`` -> ``_handle().NAME`` — plus a
machine-checkable PROOF that nothing else changed:

* AST proof: inverse-normalizing the emitted spans (``_handle().NAME`` ->
  ``NAME``) yields code AST-equal (``ast.dump`` without attributes) to the
  original extracted upstream spans;
* token proof: every token outside the rewritten references is byte-identical
  to its upstream counterpart (comments, strings and whitespace-carrying
  INDENT tokens included). A pre-3.12 f-string is ONE string token, so a token
  holding rewritten references is compared after removing exactly as many
  ``_handle().`` prefixes as it holds reference sites;
* tree-inverse proof: collapsing ``_handle().NAME`` back to ``NAME`` in the
  emitted PARSE TREE reproduces the upstream tree. The text inverse cannot see
  a literal CPython derives from the rewritten bytes — ``f"{X=}"`` prints the
  expression source, so ``f"{_h().X=}"`` inverts byte-perfectly while its
  output changed; the tree carries that Constant, so it refuses.

Fail-closed on: names that are neither leaf-local, imported in the leaf,
declared, nor builtins (reported per symbol — that report is how the declared
set gets recalculated per leaf); wildcard imports; import-time reads of
declared names (module level, class bodies, decorators, default arguments);
``global``/``del``/store of declared names; symbols the AST cannot round-trip;
any emitted span that fails one of the three proofs above.

CLI:
  emit : python scripts/v7next_transplant.py --upstream supervisor/queue.py \
             --symbols persist_queue_snapshot,parse_iso_to_ts \
             --declared PENDING,RUNNING,... --handle _queue \
             --parent-module supervisor.queue [--preamble-file hdr.py] \
             [--out leaf.py] [--json]
  check: python scripts/v7next_transplant.py --check --upstream ... \
             --leaf leaf.py --symbols ... --declared ... --handle _queue [--json]

Stdlib-only (ast, symtable, tokenize). Python >= 3.10.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import dataclasses
import io
import json
import symtable
import sys
import tokenize
from typing import Any, Dict, List, Optional, Set, Tuple

_BUILTIN_NAMES = frozenset(dir(builtins)) | {
    "__name__", "__file__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__annotations__", "__cached__",
}

_COMP_NAMES = {
    ast.ListComp: "listcomp", ast.SetComp: "setcomp",
    ast.DictComp: "dictcomp", ast.GeneratorExp: "genexpr",
}

_PREAMBLE = "<preamble>"

_DEFAULT_PREAMBLE = (
    '"""Leaf module extracted by scripts/v7next_transplant.py (v7 module-handle split)."""\n'
    "\n"
    "from __future__ import annotations\n"
)


class TransplantError(Exception):
    """Fail-closed refusal with a machine-readable report."""

    def __init__(self, kind: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.details = details

    def report(self) -> Dict[str, Any]:
        return {"ok": False, "kind": self.kind, "message": self.message, **self.details}


@dataclasses.dataclass
class Span:
    """One extracted top-level symbol: exact byte span of the upstream file."""

    name: str                 # primary requested name
    names: Tuple[str, ...]    # every name the statement binds
    start: int                # 1-based first line (decorator-inclusive)
    end: int                  # 1-based last line
    text: str                 # exact bytes of lines start..end
    node: ast.stmt


@dataclasses.dataclass
class _Site:
    line: int
    col: int
    name: str
    symbol: str


@dataclasses.dataclass
class TransplantResult:
    leaf_source: str
    symbols: List[str]
    rewrites: List[_Site]
    proof: Dict[str, Any]
    annotation_names: List[str]


# ---------------------------------------------------------------------------
# extraction


def _unfold_target(target: ast.expr) -> Tuple[List[str], bool]:
    """All Names bound by an assignment target, at any nesting depth, plus a
    flag for any non-Name leaf (attribute/subscript store — a target the
    leaf-gate must treat as complex, wave-2 conformance review)."""
    names: List[str] = []
    complex_leaf = False
    stack = [target]
    while stack:
        el = stack.pop()
        if isinstance(el, ast.Starred):
            stack.append(el.value)
        elif isinstance(el, (ast.Tuple, ast.List)):
            stack.extend(el.elts)
        elif isinstance(el, ast.Name):
            names.append(el.id)
        else:
            complex_leaf = True
    return names, complex_leaf


def _stmt_binding_names(node: ast.stmt) -> List[str]:
    """Names a top-level statement binds (defs, classes, plain/annotated assigns)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        out: List[str] = []
        for target in node.targets:
            out.extend(_unfold_target(target)[0])
        return out
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _iter_module_stmts(body: List[ast.stmt]):
    """Module-level statements, descending into shallow If/Try branches."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, ast.If):
            yield from _iter_module_stmts(stmt.body)
            yield from _iter_module_stmts(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            yield from _iter_module_stmts(stmt.body)
            for handler in stmt.handlers:
                yield from _iter_module_stmts(handler.body)
            yield from _iter_module_stmts(stmt.orelse)
            yield from _iter_module_stmts(stmt.finalbody)


def _module_bindings(tree: ast.Module) -> Set[str]:
    """Names bound at module level (the v7 leaf-binding notion, incl. If/Try)."""
    bound: Set[str] = set()
    for stmt in _iter_module_stmts(tree.body):
        bound.update(_stmt_binding_names(stmt))
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                bound.add(alias.asname or alias.name.split(".")[0])
    return bound


def _check_no_wildcard(tree: ast.Module, where: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            raise TransplantError(
                "wildcard_import",
                f"{where} contains a wildcard import at line {node.lineno}; "
                "name resolution is unsound under `import *` — fail closed",
            )


def extract_spans(source: str, symbols: List[str]) -> Dict[str, Span]:
    """Extract requested top-level symbols as exact byte spans, fail-closed.

    A span runs from the first decorator line to ``end_lineno`` and is
    round-trip checked: parsed standalone it must be a single statement whose
    ``ast.dump`` equals the node seen in the full-file parse.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    binders: Dict[str, List[ast.stmt]] = {}
    for stmt in tree.body:
        for name in _stmt_binding_names(stmt):
            binders.setdefault(name, []).append(stmt)
    wanted = list(dict.fromkeys(symbols))
    missing = [s for s in wanted if s not in binders]
    if missing:
        raise TransplantError(
            "extraction",
            f"symbols not bound by a direct top-level def/class/assign: {missing} "
            "(conditional or nested definitions are not transplantable)",
            missing=missing,
        )
    multi = [s for s in wanted if len(binders[s]) > 1]
    if multi:
        raise TransplantError(
            "extraction",
            f"symbols bound more than once at top level (redefinition/fallback): {multi}",
            multiple=multi,
        )
    spans: Dict[str, Span] = {}
    by_node: Dict[int, Span] = {}
    for sym in wanted:
        node = binders[sym][0]
        if id(node) in by_node:
            spans[sym] = by_node[id(node)]
            continue
        bound = tuple(_stmt_binding_names(node))
        stray = [n for n in bound if n not in wanted]
        if stray:
            raise TransplantError(
                "extraction",
                f"moving {sym!r} would also move {stray} (same statement binds them); "
                "request every bound name or split the statement upstream",
                statement_binds=list(bound),
            )
        deco = getattr(node, "decorator_list", []) or []
        start = min([node.lineno] + [d.lineno for d in deco])
        end = node.end_lineno or node.lineno
        text = "".join(lines[start - 1:end])
        if not text.endswith("\n"):
            text += "\n"
        try:
            standalone = ast.parse(text)
        except SyntaxError as exc:
            raise TransplantError(
                "round_trip", f"extracted span for {sym!r} does not parse standalone: {exc}",
            ) from exc
        if len(standalone.body) != 1 or ast.dump(standalone.body[0]) != ast.dump(node):
            raise TransplantError(
                "round_trip",
                f"extracted span for {sym!r} does not round-trip through the AST "
                "(statement shares physical lines with a neighbour?)",
            )
        span = Span(sym, bound, start, end, text, node)
        by_node[id(node)] = span
        spans[sym] = span
    return spans


def _unique_spans(spans: Dict[str, Span]) -> List[Span]:
    out: List[Span] = []
    seen: Set[int] = set()
    for span in spans.values():
        if id(span) not in seen:
            seen.add(id(span))
            out.append(span)
    return out


# ---------------------------------------------------------------------------
# scope analysis (symtable-backed) and rewrite collection


class _Scope:
    """A symtable table plus a cursor over its children, entered in visit order."""

    def __init__(self, table: "symtable.SymbolTable") -> None:
        self.table = table
        self.kind = table.get_type()  # "module" | "function" | "class"
        self._children = list(table.get_children())
        self._next = 0

    def enter(self, name: str, lineno: int) -> "_Scope":
        if self._next >= len(self._children):
            raise TransplantError(
                "scope_alignment",
                f"symtable ran out of child scopes entering {name!r} at line {lineno}",
            )
        child = self._children[self._next]
        self._next += 1
        if child.get_name() != name or child.get_lineno() != lineno:
            raise TransplantError(
                "scope_alignment",
                f"symtable child {child.get_name()!r}@{child.get_lineno()} does not match "
                f"AST scope {name!r}@{lineno} — exotic construct, fail closed",
            )
        return _Scope(child)

    def resolve(self, name: str) -> str:
        """'local' | 'enclosing' | 'class_local' | 'module' for a name used here."""
        if self.kind == "module":
            return "module"
        try:
            sym = self.table.lookup(name)
        except KeyError as exc:
            raise TransplantError(
                "scope_alignment", f"{name!r} missing from symtable scope {self.table.get_name()!r}",
            ) from exc
        if self.kind == "class":
            if sym.is_local():
                return "class_local"
            if sym.is_free():
                return "enclosing"
            return "module" if sym.is_global() else "enclosing"
        if sym.is_parameter() or sym.is_local():
            return "local"
        if sym.is_free():
            return "enclosing"
        return "module" if sym.is_global() else "enclosing"


class _Analysis:
    def __init__(self) -> None:
        self.rewrites: List[_Site] = []
        self.unresolved: Dict[str, Set[str]] = {}
        self.violations: List[str] = []
        self.annotation_names: Set[str] = set()


class _Walker:
    """AST walk paired with symtable scopes; collects rewrite sites and problems.

    ``calltime`` is True only inside function bodies — the D18 handle is a
    call-time read, so declared-name uses that execute at import (module
    level, class bodies, decorators, default arguments) fail closed.
    """

    def __init__(self, declared: frozenset, handle: str, module_bound: Set[str],
                 regions: List[Tuple[int, int, str]]) -> None:
        self.declared = declared
        self.handle = handle
        self.module_bound = module_bound
        self.regions = regions
        self.out = _Analysis()

    def label(self, lineno: int) -> str:
        for start, end, label in self.regions:
            if start <= lineno <= end:
                return label
        return self.regions[-1][2] if self.regions else _PREAMBLE

    def run(self, tree: ast.Module, module_table: "symtable.SymbolTable") -> _Analysis:
        scope = _Scope(module_table)
        for stmt in tree.body:
            self._visit(stmt, scope, False)
        return self.out

    # -- annotation handling: the leaf requires `from __future__ import
    # annotations`, so annotation expressions are inert strings at runtime.
    # They are skipped for rewriting, and their names surfaced as a warning
    # (the operator adds `if TYPE_CHECKING:` imports for them).
    def _ann(self, node: Optional[ast.expr]) -> None:
        if node is None:
            return
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                self.out.annotation_names.add(sub.id)

    def _visit(self, node: ast.AST, scope: _Scope, calltime: bool) -> None:
        if isinstance(node, ast.Name):
            self._name(node, scope, calltime)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for d in args.defaults:
                self._visit(d, scope, calltime)
            for d in args.kw_defaults:
                if d is not None:
                    self._visit(d, scope, calltime)
            for a in (args.posonlyargs + args.args + args.kwonlyargs
                      + ([args.vararg] if args.vararg else [])
                      + ([args.kwarg] if args.kwarg else [])):
                self._ann(a.annotation)
            self._ann(node.returns)
            for dec in node.decorator_list:
                self._visit(dec, scope, calltime)
            inner = scope.enter(node.name, node.lineno)
            for stmt in node.body:
                self._visit(stmt, inner, True)
            return
        if isinstance(node, ast.Lambda):
            for d in node.args.defaults:
                self._visit(d, scope, calltime)
            for d in node.args.kw_defaults:
                if d is not None:
                    self._visit(d, scope, calltime)
            inner = scope.enter("lambda", node.lineno)
            self._visit(node.body, inner, True)
            return
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                self._visit(base, scope, calltime)
            for kw in node.keywords:
                self._visit(kw.value, scope, calltime)
            for dec in node.decorator_list:
                self._visit(dec, scope, calltime)
            inner = scope.enter(node.name, node.lineno)
            for stmt in node.body:
                self._visit(stmt, inner, calltime)  # class body runs when the stmt runs
            return
        if type(node) in _COMP_NAMES:
            gens = node.generators
            self._visit(gens[0].iter, scope, calltime)  # outermost iterable: enclosing scope
            inner = scope.enter(_COMP_NAMES[type(node)], node.lineno)
            self._visit(gens[0].target, inner, calltime)
            for cond in gens[0].ifs:
                self._visit(cond, inner, calltime)
            for gen in gens[1:]:
                self._visit(gen.iter, inner, calltime)
                self._visit(gen.target, inner, calltime)
                for cond in gen.ifs:
                    self._visit(cond, inner, calltime)
            if isinstance(node, ast.DictComp):
                self._visit(node.key, inner, calltime)
                self._visit(node.value, inner, calltime)
            else:
                self._visit(node.elt, inner, calltime)
            return
        if isinstance(node, ast.Global):
            label = self.label(node.lineno)
            for name in node.names:
                if name in self.declared:
                    self.out.violations.append(
                        f"{label}: `global {name}` at line {node.lineno} — a declared "
                        "parent-owned name cannot be rebound through the handle")
                elif name not in self.module_bound:
                    self.out.violations.append(
                        f"{label}: `global {name}` at line {node.lineno} rebinds module "
                        "state whose defining assignment did not move with the leaf")
            return
        if isinstance(node, ast.AnnAssign):
            self._ann(node.annotation)
            if node.value is not None:
                self._visit(node.value, scope, calltime)
            self._visit(node.target, scope, calltime)
            return
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                name = node.target.id
                if scope.resolve(name) == "module":
                    label = self.label(node.lineno)
                    if name in self.declared:
                        self.out.violations.append(
                            f"{label}: augmented assignment to declared name {name!r} "
                            f"at line {node.lineno}")
                    elif name not in self.module_bound and name not in _BUILTIN_NAMES:
                        self.out.unresolved.setdefault(label, set()).add(name)
            else:
                self._visit(node.target, scope, calltime)
            self._visit(node.value, scope, calltime)
            return
        for child in ast.iter_child_nodes(node):
            self._visit(child, scope, calltime)

    def _name(self, node: ast.Name, scope: _Scope, calltime: bool) -> None:
        name = node.id
        res = scope.resolve(name)
        label = self.label(node.lineno)
        if isinstance(node.ctx, ast.Load):
            if res == "module":
                if name in self.declared:
                    if label == _PREAMBLE:
                        self.out.violations.append(
                            f"{label}: preamble reads declared name {name!r} at line {node.lineno}")
                    elif not calltime:
                        self.out.violations.append(
                            f"{label}: import-time read of declared name {name!r} at line "
                            f"{node.lineno} (module level / class body / decorator / default "
                            "argument) — a call-time handle cannot carry it")
                    else:
                        self.out.rewrites.append(_Site(node.lineno, node.col_offset, name, label))
                elif name in self.module_bound or name in _BUILTIN_NAMES:
                    pass
                else:
                    self.out.unresolved.setdefault(label, set()).add(name)
            elif res == "class_local" and name in self.declared:
                self.out.violations.append(
                    f"{label}: declared name {name!r} is bound in a class body and also read "
                    f"there (line {node.lineno}); which binding a read hits is not decidable "
                    "mechanically — fail closed")
        else:
            if res == "module" and name in self.declared and scope.kind != "module":
                self.out.violations.append(
                    f"{label}: declared name {name!r} is written/deleted at line {node.lineno}; "
                    "stores cannot go through the call-time handle")


# ---------------------------------------------------------------------------
# assembly and rewriting


def generate_handle_def(handle: str, parent_module: str) -> str:
    parts = parent_module.split(".")
    if len(parts) == 1:
        imp, ref = f"import {parent_module}", parent_module
    else:
        imp, ref = f"from {'.'.join(parts[:-1])} import {parts[-1]}", parts[-1]
    return (
        f"def {handle}():\n"
        f'    """The parent module, read at call time.\n'
        f"\n"
        f"    The parent owns the rebindable module state and the members tests\n"
        f"    monkeypatch there; reading them through the module at each call keeps\n"
        f"    one binding, where a from-import would freeze the value this leaf saw\n"
        f'    at import time (the owner-approved D18/D33 mechanical exception).\n'
        f'    """\n'
        f"    {imp}\n"
        f"\n"
        f"    return {ref}\n"
    )


def _assemble(preamble: str, spans: List[Span]) -> Tuple[str, List[Tuple[int, int, str]]]:
    head = preamble.rstrip("\n") + "\n"
    parts = [head]
    regions: List[Tuple[int, int, str]] = [(1, head.count("\n"), _PREAMBLE)]
    cur = head.count("\n")
    for span in spans:
        parts.append("\n\n")
        cur += 2
        n = span.text.count("\n")
        regions.append((cur + 1, cur + n, span.name))
        parts.append(span.text)
        cur += n
    return "".join(parts), regions


def _apply_rewrites(source: str, sites: List[_Site], handle: str) -> str:
    lines = source.splitlines(keepends=True)
    for site in sorted(sites, key=lambda s: (s.line, s.col), reverse=True):
        line = lines[site.line - 1]
        seg = line[site.col:site.col + len(site.name)]
        before = line[site.col - 1] if site.col else ""
        after = line[site.col + len(site.name):site.col + len(site.name) + 1]
        if (seg != site.name
                or (before and (before.isalnum() or before in "._"))
                or (after and (after.isalnum() or after == "_"))):
            raise TransplantError(
                "position_mismatch",
                f"cannot rewrite {site.name!r} at {site.line}:{site.col} — source text does "
                "not match the AST position (fail closed rather than corrupt bytes)",
            )
        lines[site.line - 1] = line[:site.col] + handle + "()." + line[site.col:]
    return "".join(lines)


# ---------------------------------------------------------------------------
# declared-set recalculation hints


def suggest_resolutions(upstream_source: str, names: Set[str]) -> Dict[str, Dict[str, str]]:
    """Classify unresolved names against the upstream module (the recalc workflow)."""
    tree = ast.parse(upstream_source)
    imported: Dict[str, str] = {}
    bound: Set[str] = set()
    for stmt in _iter_module_stmts(tree.body):
        bound.update(_stmt_binding_names(stmt))
        if isinstance(stmt, ast.Import):
            for a in stmt.names:
                key = a.asname or a.name.split(".")[0]
                imported[key] = f"import {a.name}" + (f" as {a.asname}" if a.asname else "")
        elif isinstance(stmt, ast.ImportFrom):
            mod = "." * stmt.level + (stmt.module or "")
            for a in stmt.names:
                if a.name == "*":
                    continue
                imported[a.asname or a.name] = (
                    f"from {mod} import {a.name}" + (f" as {a.asname}" if a.asname else ""))
    global_bound = {n for g in ast.walk(tree) if isinstance(g, ast.Global) for n in g.names}
    out: Dict[str, Dict[str, str]] = {}
    for name in sorted(names):
        if name in bound or name in global_bound:
            out[name] = {"kind": "parent_global", "hint": (
                "bound at module scope of the upstream parent (rebindable state or member): "
                "add it to the declared set so the leaf reads it through the handle, or "
                "re-create it in the leaf preamble if it is leaf-owned (e.g. a logger)")}
        elif name in imported:
            out[name] = {"kind": "parent_import", "import": imported[name], "hint": (
                f"imported by the upstream parent via `{imported[name]}`: copy that import "
                "into the leaf preamble, or add the name to the declared set if tests "
                "rebind it on the parent")}
        else:
            out[name] = {"kind": "unknown", "hint": (
                "not found at upstream module scope — dynamic binding or a genuine bug; "
                "resolve manually")}
    return out


# ---------------------------------------------------------------------------
# the transform


def transplant(upstream_source: str, symbols: List[str], declared: Any, handle: str,
               parent_module: Optional[str] = None, preamble: Optional[str] = None,
               ) -> TransplantResult:
    """Emit the leaf source for ``symbols`` and prove the transform, fail-closed."""
    declared = frozenset(declared)
    if handle in declared:
        raise TransplantError("handle_collision", f"handle {handle!r} cannot be a declared name")
    up_tree = ast.parse(upstream_source)
    _check_no_wildcard(up_tree, "upstream module")
    spans = extract_spans(upstream_source, symbols)
    unique = _unique_spans(spans)
    for span in unique:
        for sub in ast.walk(span.node):
            if isinstance(sub, ast.Name) and sub.id == handle:
                raise TransplantError(
                    "handle_collision",
                    f"upstream symbol {span.name!r} already uses the name {handle!r} "
                    f"(line {sub.lineno}); pick a different handle")
    moved_names = {n for span in unique for n in span.names}

    preamble_text = preamble if preamble is not None else _DEFAULT_PREAMBLE
    try:
        pre_tree = ast.parse(preamble_text)
    except SyntaxError as exc:
        raise TransplantError("preamble", f"preamble does not parse: {exc}") from exc
    _check_no_wildcard(pre_tree, "preamble")
    has_future = any(
        isinstance(s, ast.ImportFrom) and s.module == "__future__"
        and any(a.name == "annotations" for a in s.names)
        for s in pre_tree.body)
    if not has_future:
        raise TransplantError(
            "preamble",
            "leaf preamble must carry `from __future__ import annotations`: the tool treats "
            "annotations as inert strings (they are skipped for handle rewriting), which is "
            "only sound under deferred annotations")
    handle_defined = any(
        isinstance(s, ast.FunctionDef) and s.name == handle for s in pre_tree.body)
    if not handle_defined:
        if not parent_module:
            raise TransplantError(
                "preamble",
                f"preamble does not define {handle}() and no --parent-module was given "
                "to generate it")
        preamble_text = preamble_text.rstrip("\n") + "\n\n\n" + generate_handle_def(handle, parent_module)

    assembled, regions = _assemble(preamble_text, unique)
    try:
        tree = ast.parse(assembled)
    except SyntaxError as exc:
        raise TransplantError("round_trip", f"assembled leaf does not parse: {exc}") from exc
    module_bound = _module_bindings(tree)
    overlap = sorted((module_bound - moved_names) & declared)
    if overlap:
        raise TransplantError(
            "preamble",
            f"preamble binds declared names {overlap}; a declared name may only be bound "
            "in the leaf by a moved symbol (the re-export pattern) — otherwise reads are "
            "ambiguous")
    table = symtable.symtable(assembled, "<leaf>", "exec")
    analysis = _Walker(declared, handle, module_bound, regions).run(tree, table)
    if analysis.violations:
        raise TransplantError(
            "violation", "transform refused:\n  " + "\n  ".join(analysis.violations),
            violations=analysis.violations)
    if analysis.unresolved:
        flat = sorted({n for names in analysis.unresolved.values() for n in names})
        raise TransplantError(
            "unresolved_names",
            "names used by the moved code but neither leaf-local, imported in the leaf, "
            "declared, nor builtins — recalculate the declared set / preamble imports:\n  "
            + "\n  ".join(
                f"{label}: {sorted(names)}" for label, names in sorted(analysis.unresolved.items())),
            unresolved={k: sorted(v) for k, v in analysis.unresolved.items()},
            suggestions=suggest_resolutions(upstream_source, set(flat)))
    leaf_source = _apply_rewrites(assembled, analysis.rewrites, handle)
    unused = sorted(declared - {s.name for s in analysis.rewrites})
    if unused:
        raise TransplantError(
            "unused_declared",
            f"declared names never read by the moved symbols: {unused} — drop them from "
            "the declared set for this leaf (or include the symbols that read them)",
            unused=unused)
    try:
        ast.parse(leaf_source)
    except SyntaxError as exc:  # pragma: no cover - guarded by position checks
        raise TransplantError("round_trip", f"emitted leaf does not parse: {exc}") from exc
    proof = verify_transplant(upstream_source, leaf_source, list(spans), declared, handle)
    if not proof["ok"]:
        raise TransplantError(
            "proof", "internal error: emitted leaf failed its own proof", proof=proof)
    interesting_ann = sorted(
        n for n in analysis.annotation_names
        if n not in module_bound and n not in _BUILTIN_NAMES)
    return TransplantResult(
        leaf_source=leaf_source,
        symbols=[s.name for s in unique],
        rewrites=sorted(analysis.rewrites, key=lambda s: (s.line, s.col)),
        proof=proof,
        annotation_names=interesting_ann,
    )


# ---------------------------------------------------------------------------
# the proof


def _tokens(text: str) -> List[tokenize.TokenInfo]:
    if not text.endswith("\n"):
        text += "\n"
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    return [t for t in toks if t.type not in (tokenize.ENCODING, tokenize.ENDMARKER)]


def _lockstep_tokens(up_text: str, leaf_text: str, sites: List[ast.Attribute],
                     handle: str) -> Tuple[bool, Optional[str]]:
    """Byte-compare tokens outside the rewritten references (spec proof #2).

    Before 3.12 an f-string is a SINGLE string token, so a reference rewritten
    inside one is not a token of its own: such a token is compared after
    stripping the ``handle().`` prefixes it carries, and only when it carries
    exactly as many as it holds reference sites (an upstream literal that
    already spelled ``handle().`` fails closed rather than being un-spelled).
    """
    try:
        up, lf = _tokens(up_text), _tokens(leaf_text)
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:  # pragma: no cover
        return False, f"tokenize failed: {exc}"
    site_at = {(s.lineno, s.col_offset): s.attr for s in sites}
    i = j = 0
    while i < len(up) or j < len(lf):
        if j < len(lf):
            t = lf[j]
            if t.start in site_at and t.type == tokenize.NAME and t.string == handle:
                attr = site_at[t.start]
                tail = [x.string for x in lf[j + 1:j + 5]]
                if tail[:3] != ["(", ")", "."] or len(tail) < 4 or tail[3] != attr:
                    return False, f"malformed handle read at leaf {t.start}"
                if i >= len(up) or up[i].type != tokenize.NAME or up[i].string != attr:
                    got = up[i].string if i < len(up) else "<eof>"
                    return False, f"upstream token at site {t.start} is {got!r}, expected {attr!r}"
                j += 5
                i += 1
                continue
        if i >= len(up) or j >= len(lf):
            return False, "token streams have different lengths outside rewritten references"
        a, b = up[i], lf[j]
        if a.type != b.type or a.string != b.string:
            inner = [s for s in sites if b.start <= (s.lineno, s.col_offset) < b.end]
            prefix = f"{handle}()."
            if not (inner and a.type == b.type == tokenize.STRING
                    and b.string.count(prefix) == len(inner)
                    and b.string.replace(prefix, "") == a.string):
                return False, (
                    f"token mismatch outside rewritten references: upstream {a.start} "
                    f"{a.string!r} vs leaf {b.start} {b.string!r}")
        i += 1
        j += 1
    return True, None


def verify_transplant(upstream_source: str, leaf_source: str, symbols: List[str],
                      declared: Any, handle: str,
                      leaf_owned: Optional[Set[str]] = None) -> Dict[str, Any]:
    """The PROOF: per moved symbol, inverse-normalize the leaf span and require
    (1) AST equality with the upstream span (ast.dump, no attributes),
    (2) byte-identical tokens outside the rewritten references,
    (3) a byte-identical round trip, AND (4) tree-inverse equality: collapsing
    the handle reads in the leaf's own PARSE TREE reproduces the upstream tree.
    (4) is not implied by (3): the text inverse restores bytes CPython also
    derived a literal from — ``f"{X=}"`` prints its expression source, so
    ``f"{_h().X=}"`` inverts byte-perfectly with a changed output; the leaf tree
    carries that Constant and refuses. Beyond the per-symbol spans it validates the
    WHOLE leaf as a runnable module (F0 phase review, audit 2026-08-30): the
    handle must be defined exactly once with the canonical body, no name may be
    both declared and preamble-bound (ambiguous ownership), every declared name
    must actually be read, and nothing unexpected may sit at top level.
    """
    declared = frozenset(declared)
    up_spans = extract_spans(upstream_source, symbols)
    leaf_spans = extract_spans(leaf_source, symbols)
    report: Dict[str, Any] = {"ok": True, "symbols": {}, "handle_reads": [],
                              "unread_declared": [], "leaf_invariants": []}
    reads: Set[str] = set()
    for span in _unique_spans(leaf_spans):
        up = up_spans[span.name]
        entry: Dict[str, Any] = {"ast_equal": False, "tokens_equal": False,
                                 "byte_identical": False, "ast_inverse_equal": False,
                                 "handle_reads": [], "detail": None}
        report["symbols"][span.name] = entry
        leaf_tree = ast.parse(span.text)
        sites: List[ast.Attribute] = []
        for node in ast.walk(leaf_tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == handle
                    and not node.value.args and not node.value.keywords):
                sites.append(node)
        handle_names = {(n.lineno, n.col_offset) for n in ast.walk(leaf_tree)
                        if isinstance(n, ast.Name) and n.id == handle}
        site_funcs = {(s.value.func.lineno, s.value.func.col_offset) for s in sites}
        problems: List[str] = []
        if handle_names - site_funcs:
            problems.append(f"{handle!r} used outside the `{handle}().NAME` read form")
        for site in sites:
            if site.attr not in declared:
                problems.append(f"handle read of undeclared name {site.attr!r} "
                                f"(line {site.lineno})")
            if site.lineno != site.end_lineno:
                problems.append(f"handle read of {site.attr!r} spans multiple lines "
                                f"(line {site.lineno}); cannot invert")
        recovered = None
        if not problems:
            lines = span.text.splitlines(keepends=True)
            for site in sorted(sites, key=lambda s: (s.lineno, s.col_offset), reverse=True):
                line = lines[site.lineno - 1]
                seg = line[site.col_offset:site.end_col_offset]
                if seg != f"{handle}().{site.attr}":
                    problems.append(f"cannot invert handle read at {site.lineno}:"
                                    f"{site.col_offset}: source is {seg!r}")
                    break
                lines[site.lineno - 1] = (
                    line[:site.col_offset] + site.attr + line[site.end_col_offset:])
            else:
                recovered = "".join(lines)
        if recovered is not None:
            try:
                entry["ast_equal"] = (
                    ast.dump(ast.parse(recovered)) == ast.dump(ast.parse(up.text)))
            except SyntaxError as exc:
                problems.append(f"inverse-normalized span does not parse: {exc}")
            if not entry["ast_equal"] and not problems:
                problems.append("inverse-normalized span is not AST-equal to upstream")
            ok, why = _lockstep_tokens(up.text, span.text, sites, handle)
            entry["tokens_equal"] = ok
            if not ok:
                problems.append(why or "token mismatch")
            entry["byte_identical"] = recovered == up.text
            if not entry["byte_identical"] and not problems:
                # MANDATORY (audit 2026-08-30): token equality alone is blind to
                # inter-token whitespace (`x  + y` == `x + y` as tokens). The
                # round trip must reproduce the upstream span BYTE-exactly.
                for k, (a, b) in enumerate(zip(recovered, up.text)):
                    if a != b:
                        problems.append(
                            "inverse-normalized span is not byte-identical to "
                            f"upstream (first divergence at offset {k}: "
                            f"{recovered[k:k+20]!r} != {up.text[k:k+20]!r})")
                        break
                else:
                    problems.append(
                        "inverse-normalized span is not byte-identical to "
                        "upstream (length differs: "
                        f"{len(recovered)} != {len(up.text)})")
            class _InverseReads(ast.NodeTransformer):
                def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                    self.generic_visit(node)
                    call = node.value
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                            and call.func.id == handle and not call.args
                            and not call.keywords):
                        return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
                    return node

            inverse_tree = _InverseReads().visit(leaf_tree)
            entry["ast_inverse_equal"] = (
                ast.dump(inverse_tree) == ast.dump(ast.parse(up.text)))
            if not entry["ast_inverse_equal"]:
                problems.append("inverse-normalized parse tree is not AST-equal to upstream")
        entry["handle_reads"] = sorted({s.attr for s in sites})
        reads.update(s.attr for s in sites)
        if problems:
            entry["detail"] = "; ".join(problems)
            report["ok"] = False
        elif not (entry["ast_equal"] and entry["tokens_equal"]
                  and entry["byte_identical"] and entry["ast_inverse_equal"]):
            report["ok"] = False
    report["handle_reads"] = sorted(reads)
    report["unread_declared"] = sorted(declared - reads)
    _flag_undeclared_top_level(leaf_source, symbols, handle, report, leaf_owned)
    _validate_leaf_invariants(leaf_source, symbols, declared, handle, reads, report,
                              leaf_owned)
    return report


def _own_returns(fn: ast.FunctionDef) -> List[ast.Return]:
    """Return statements of fn's OWN body, not of functions nested inside it."""
    out: List[ast.Return] = []
    stack: List[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Return):
            out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _validate_leaf_invariants(leaf_source: str, symbols: List[str], declared: frozenset,
                              handle: str, reads: Set[str], report: Dict[str, Any],
                              leaf_owned: Optional[Set[str]]) -> None:
    """Whole-leaf invariants a byte-faithful span proof cannot see (audit 2026-08-30
    CRITICAL): the leaf must be a runnable module, not just faithful fragments."""
    owned = set(leaf_owned or _PREAMBLE_OK_ASSIGN_DEFAULT)
    problems: List[str] = []
    tree = ast.parse(leaf_source)
    # (a) when the leaf reads anything through the handle (or declares names),
    # the handle must be defined exactly once, as a canonical SYNC parameterless
    # function whose own body returns a module reference (a Name or dotted
    # Attribute — generate_handle_def shape), never a constant, and never only
    # from a nested function. A projection-only leaf (zero handle reads, zero
    # declared) may omit it.
    if any(isinstance(n, ast.AsyncFunctionDef) and n.name == handle for n in tree.body):
        problems.append(f"handle {handle!r} must be a sync def, not async")
    handle_defs = [n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == handle]
    handle_required = bool(reads or declared)
    if handle_required and len(handle_defs) != 1:
        problems.append(f"handle {handle!r} defined {len(handle_defs)} times, expected 1")
    elif len(handle_defs) > 1:
        problems.append(f"handle {handle!r} defined {len(handle_defs)} times, expected at most 1")
    elif handle_defs:
        hd = handle_defs[0]
        a = hd.args
        if a.args or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
            problems.append(f"handle {handle!r} must take no parameters")
        returns = _own_returns(hd)
        ok_ret = lambda v: isinstance(v, (ast.Name, ast.Attribute))  # noqa: E731
        if not returns or not all(r.value is not None and ok_ret(r.value) for r in returns):
            problems.append(
                f"handle {handle!r} body must directly return a module reference "
                "(Name/Attribute), never a constant, bare return, or only via a "
                "nested function")
    # (b) declared names and preamble-bound names must be DISJOINT (ambiguous
    # ownership: a name both imported locally and read through the handle).
    preamble_bound: Set[str] = set()
    span_names = set(symbols) | {handle}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                preamble_bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name not in span_names:
                preamble_bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id not in span_names:
                    preamble_bound.add(t.id)
    overlap = declared & preamble_bound
    if overlap:
        problems.append(f"names both declared and preamble-bound (ambiguous ownership): "
                        f"{sorted(overlap)}")
    # (c) every declared name must actually be read through the handle.
    if declared - reads:
        problems.append(f"declared but never read through {handle}(): "
                        f"{sorted(declared - reads)}")
    # (d) leaf-owned allowlist must not silently absorb a declared name.
    if owned & declared:
        problems.append(f"leaf-owned allowlist overlaps declared set: {sorted(owned & declared)}")
    report["leaf_invariants"] = problems
    if problems:
        report["ok"] = False


_PREAMBLE_OK_ASSIGN_DEFAULT = frozenset({"log"})


def _flag_undeclared_top_level(leaf_source: str, symbols: List[str], handle: str,
                               report: Dict[str, Any],
                               leaf_owned: Optional[Set[str]] = None) -> None:
    """MANDATORY (audit 2026-08-30): a leaf must contain NOTHING at top level
    beyond the verified symbol spans and a recognizable preamble — otherwise an
    extra definition or an import-time side effect rides an ok=true proof.
    Allowed outside the requested symbols: the module docstring, __future__ and
    ordinary imports, the handle def itself, and simple assignments to
    leaf-owned names (default: {'log'}; extend deliberately, never silently).
    """
    owned = set(leaf_owned or _PREAMBLE_OK_ASSIGN_DEFAULT) | {handle}
    requested = set(symbols)
    extras: List[str] = []
    tree = ast.parse(leaf_source)
    for idx, node in enumerate(tree.body):
        if idx == 0 and isinstance(node, ast.Expr) and isinstance(
                getattr(node, "value", None), ast.Constant) and isinstance(
                node.value.value, str):
            continue  # module docstring
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in requested or node.name in owned:
                continue
            extras.append(f"{type(node).__name__} {node.name!r} (line {node.lineno})")
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            # Unfold to ANY depth: `A, B = ...`, `A, (X, Y) = ...`; a non-Name
            # leaf (obj.attr, d[k]) marks the whole statement complex — it can
            # never be a faithfully moved binding (wave-2 conformance review).
            names: Set[str] = set()
            has_complex = False
            for t in targets:
                got, complex_leaf = _unfold_target(t)
                names.update(got)
                has_complex = has_complex or complex_leaf
            if not has_complex and names and names <= (requested | owned):
                continue
            extras.append(f"assignment to {sorted(names) or '<complex target>'}"
                          f"{' (complex target)' if has_complex else ''} "
                          f"(line {node.lineno})")
            continue
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) \
                and node.test.id == "TYPE_CHECKING":
            continue
        extras.append(f"{type(node).__name__} (line {node.lineno})")
    report["undeclared_top_level"] = extras
    if extras:
        report["ok"] = False


# ---------------------------------------------------------------------------
# CLI


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v7next_transplant",
        description="Mechanical D18/D33 module-handle transplant with a built-in proof.")
    parser.add_argument("--upstream", required=True, help="upstream source file")
    parser.add_argument("--symbols", required=True, help="comma-separated symbols to move")
    parser.add_argument("--declared", default="", help="comma-separated declared parent names")
    parser.add_argument("--handle", required=True, help="handle function name, e.g. _queue")
    parser.add_argument("--parent-module", help="dotted parent module (for the generated handle)")
    parser.add_argument("--preamble-file", help="leaf header: docstring, imports, handle def")
    parser.add_argument("--check", action="store_true",
                        help="verify an existing leaf instead of emitting one")
    parser.add_argument("--leaf", help="leaf file to verify (with --check)")
    parser.add_argument("--out", help="write the emitted leaf here (default: stdout)")
    parser.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    args = parser.parse_args(argv)

    upstream_source = open(args.upstream, encoding="utf-8").read()
    symbols = _split_csv(args.symbols)
    declared = frozenset(_split_csv(args.declared))

    try:
        if args.check:
            if not args.leaf:
                parser.error("--check requires --leaf")
            leaf_source = open(args.leaf, encoding="utf-8").read()
            report = verify_transplant(upstream_source, leaf_source, symbols, declared, args.handle)
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                for name, entry in sorted(report["symbols"].items()):
                    status = "OK" if entry["ast_equal"] and entry["tokens_equal"] else "FAIL"
                    extra = f" ({entry['detail']})" if entry["detail"] else ""
                    print(f"{status} {name}: ast={entry['ast_equal']} tokens={entry['tokens_equal']} "
                          f"bytes={entry['byte_identical']} reads={entry['handle_reads']}{extra}",
                          file=sys.stderr)
                print(f"handle reads: {report['handle_reads']}", file=sys.stderr)
                if report["unread_declared"]:
                    print(f"declared but unread here: {report['unread_declared']}", file=sys.stderr)
            return 0 if report["ok"] else 2
        preamble = None
        if args.preamble_file:
            preamble = open(args.preamble_file, encoding="utf-8").read()
        result = transplant(upstream_source, symbols, declared, args.handle,
                            parent_module=args.parent_module, preamble=preamble)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(result.leaf_source)
        else:
            sys.stdout.write(result.leaf_source)
        summary = {
            "ok": True,
            "symbols": result.symbols,
            "rewrites": [dataclasses.asdict(s) for s in result.rewrites],
            "proof_ok": result.proof["ok"],
            "handle_reads": result.proof["handle_reads"],
            "annotation_names": result.annotation_names,
        }
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"transplanted {len(result.symbols)} symbols, "
                  f"{len(result.rewrites)} handle rewrites, proof ok", file=sys.stderr)
            if result.annotation_names:
                print("annotation-only names (add `if TYPE_CHECKING:` imports as needed): "
                      f"{result.annotation_names}", file=sys.stderr)
        return 0
    except TransplantError as exc:
        if args.json:
            print(json.dumps(exc.report(), indent=2, sort_keys=True, default=sorted))
        print(f"FAIL[{exc.kind}]: {exc.message}", file=sys.stderr)
        if exc.kind == "unresolved_names":
            for name, info in exc.details.get("suggestions", {}).items():
                print(f"  {name}: {info['kind']} — {info['hint']}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
