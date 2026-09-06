#!/usr/bin/env python3
"""Shared import-graph core for the domain manifest tools (plan §7.1, CPL-1).

Single home of the machinery that both the report generator
(``scripts/v7next_domain_report.py``) and the gate checker
(``scripts/check_domains.py``) consume, so the two tools cannot drift apart —
the same discipline the checker itself enforces on runtime code (the
literal-copy ban).

Provides:

- manifest loading (``ouroboros/domains.toml``: human sections + generated
  ``[graph]``/``[duplicates]`` sections);
- the module-level import collector with the strict / lazy / guarded /
  TYPE_CHECKING / dynamic classification (an import is STRICT only when it
  executes unconditionally at import time);
- the domain quotient (cross-domain edges keyed by domain pair, with exact
  module-edge witnesses) and Tarjan SCC over domain nodes;
- the literal-copy scan: normalized function-body source segments appearing
  in more than one domain (the span-normalization approach follows
  ``scripts/v7next_transplant.py``: exact source segments, not name matching).

This is analysis tooling, not runtime code: nothing under ``ouroboros/``
imports it.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib
import subprocess
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - the 3.10 venv ships tomli
    import tomli as tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "ouroboros" / "domains.toml"
DOMAIN_MAP_PATH = REPO_ROOT / "docs" / "DOMAIN_MAP.md"

STRICT, TYPE_ONLY, LAZY, DYNAMIC = "strict", "type_checking", "lazy", "dynamic"
# Executed at import time but failure-tolerant / entrypoint-only (F0 review F4):
# a `try: import x except ImportError/Exception` or an import under
# `if __name__ == "__main__"` must not stand as a strict cycle witness.
GUARDED = "guarded"

# A function body shorter than this many normalized source lines is too small
# to stand as literal-copy evidence (getters, one-line delegates).
DUPLICATE_MIN_LINES = 10


def module_name(path: str) -> str:
    parts = path[:-3].split("/")  # drop .py
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def is_main_guard_test(test: ast.expr) -> bool:
    """`if __name__ == "__main__":` — the body never runs on import."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1):
        return False
    sides = (test.left, test.comparators[0])
    has_name = any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
    has_main = any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides)
    return has_name and has_main


_SWALLOWING = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def try_swallows_import_failure(node: ast.Try) -> bool:
    """True when at least one handler catches import failure (or everything)
    AND does not re-raise. A handler whose body contains a top-level ``raise``
    may propagate the failure (``except ImportError: raise``), so it is not a
    swallow — misclassifying it as guarded would hide a strict cycle witness
    (F0 review round 2). A conditional re-raise nested in an ``if`` still
    counts as re-raising here: erring toward STRICT is the safe direction."""
    for h in node.handlers:
        reraises = any(isinstance(s, ast.Raise) for s in ast.walk(h))
        if reraises:
            continue
        if h.type is None:  # bare except
            return True
        types = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
        for t in types:
            if isinstance(t, ast.Name) and t.id in _SWALLOWING:
                return True
            if isinstance(t, ast.Attribute) and t.attr in _SWALLOWING:
                return True
    return False


class ImportCollector(ast.NodeVisitor):
    """Collect (kind, raw dotted target or ImportFrom base+names, lineno)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, tuple[str, ...], int]] = []
        # each record: (kind, base_or_module, aliases (() for plain import), lineno)
        self._depth = 0  # function nesting depth
        self._tc = 0     # TYPE_CHECKING nesting depth
        self._guard = 0  # __main__-guard / failure-swallowing-try nesting depth

    def _kind(self) -> str:
        if self._tc:
            return TYPE_ONLY
        if self._depth:
            return LAZY
        if self._guard:
            return GUARDED
        return STRICT

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_If(self, node: ast.If) -> None:
        tc = is_type_checking_test(node.test)
        mg = is_main_guard_test(node.test)
        if tc:
            self._tc += 1
        if mg:
            self._guard += 1
        for child in node.body:
            self.visit(child)
        if tc:
            self._tc -= 1
        if mg:
            self._guard -= 1
        for child in node.orelse:
            self.visit(child)

    def visit_Try(self, node: ast.Try) -> None:
        swallows = try_swallows_import_failure(node)
        if swallows:
            self._guard += 1
        for child in node.body:
            self.visit(child)
        if swallows:
            self._guard -= 1
        for part in (node.handlers, node.orelse, node.finalbody):
            for child in part:
                self.visit(child)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.records.append((self._kind(), alias.name, (), node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = tuple(a.name for a in node.names)
        base = ("." * node.level) + (node.module or "")
        self.records.append((self._kind(), base, names, node.lineno))

    def visit_Call(self, node: ast.Call) -> None:
        target = None
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "import_module":
            target = "?"
        elif isinstance(f, ast.Name) and f.id == "__import__":
            target = "?"
        if target is not None:
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                target = node.args[0].value
            else:
                target = "<unresolved>"
            self.records.append((DYNAMIC, target, (), node.lineno))
        self.generic_visit(node)


def resolve_relative(base: str, importer: str, is_pkg: bool) -> str:
    level = len(base) - len(base.lstrip("."))
    tail = base[level:]
    parts = importer.split(".")
    if not is_pkg:
        parts = parts[:-1]
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    prefix = ".".join(parts)
    return f"{prefix}.{tail}" if tail else prefix


@dataclass
class Manifest:
    """Parsed ``ouroboros/domains.toml``."""

    path: pathlib.Path
    raw_bytes: bytes
    meta: dict[str, str]
    domains: dict[str, str]
    modules: dict[str, str]                 # path -> domain id
    proposed: set[str]
    graph_allowed: list[str] = field(default_factory=list)      # "D01->D02"
    cycle_groups: list[list[str]] = field(default_factory=list)
    lazy_only: list[str] = field(default_factory=list)
    dynamic_pairs: list[str] = field(default_factory=list)
    duplicates_allowed: list[str] = field(default_factory=list)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()


def load_manifest(path: pathlib.Path = MANIFEST_PATH) -> Manifest:
    raw = path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    graph = data.get("graph", {})
    return Manifest(
        path=path,
        raw_bytes=raw,
        meta=data.get("meta", {}),
        domains=data["domains"],
        modules=data["modules"],
        proposed=set(data.get("classification", {}).get("proposed", [])),
        graph_allowed=list(graph.get("allowed", [])),
        cycle_groups=[list(g) for g in graph.get("cycle_groups", [])],
        lazy_only=list(graph.get("lazy_only", [])),
        dynamic_pairs=list(graph.get("dynamic_pairs", [])),
        duplicates_allowed=list(data.get("duplicates", {}).get("allowed", [])),
    )


def tracked_population(root: pathlib.Path = REPO_ROOT) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "ouroboros/**/*.py", "ouroboros/*.py",
         "supervisor/*.py", "supervisor/**/*.py", "server.py", "launcher.py"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.split()
    return set(out)


@dataclass
class ImportGraph:
    """The classified module-level import graph over the manifest population."""

    edges: dict[str, dict[tuple[str, str], list[int]]]
    dynamic_unresolved: list[tuple[str, int]]
    dom_of_path: dict[str, str]

    def quotient(self, kind: str = STRICT) -> dict[tuple[str, str], list[tuple[str, str]]]:
        """Cross-domain edges of one kind, keyed by (from_domain, to_domain)."""
        pairs: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for (src, dst) in sorted(self.edges[kind]):
            d1, d2 = self.dom_of_path[src], self.dom_of_path[dst]
            if d1 != d2:
                pairs[(d1, d2)].append((src, dst))
        return pairs


def build_import_graph(manifest: Manifest, root: pathlib.Path = REPO_ROOT,
                       population: set[str] | None = None) -> ImportGraph:
    modules = manifest.modules
    mod_by_name = {module_name(p): p for p in modules}

    def resolve(dotted: str) -> str | None:
        """Longest population module matching the dotted name."""
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if cand in mod_by_name:
                return mod_by_name[cand]
        return None

    edges: dict[str, dict[tuple[str, str], list[int]]] = {
        STRICT: defaultdict(list), TYPE_ONLY: defaultdict(list),
        LAZY: defaultdict(list), DYNAMIC: defaultdict(list),
        GUARDED: defaultdict(list),
    }
    dynamic_unresolved: list[tuple[str, int]] = []
    pop = set(modules) if population is None else (set(modules) & population)

    for path in sorted(pop):
        src = root / path
        tree = ast.parse(src.read_text(encoding="utf-8"), filename=path)
        importer = module_name(path)
        is_pkg = path.endswith("__init__.py")
        col = ImportCollector()
        col.visit(tree)
        for kind, base, names, lineno in col.records:
            if kind == DYNAMIC and base == "<unresolved>":
                dynamic_unresolved.append((path, lineno))
                continue
            base_abs = resolve_relative(base, importer, is_pkg) if base.startswith(".") else base
            targets = []
            if names:
                for name in names:
                    if name == "*":
                        targets.append(base_abs)
                        continue
                    sub = f"{base_abs}.{name}" if base_abs else name
                    targets.append(sub if resolve(sub) else base_abs)
            else:
                targets.append(base_abs)
            for dotted in targets:
                dst = resolve(dotted)
                if dst is None or dst == path:
                    continue
                edges[kind][(path, dst)].append(lineno)

    return ImportGraph(edges=edges, dynamic_unresolved=dynamic_unresolved,
                       dom_of_path=dict(modules))


def strongly_connected(domains: list[str],
                       dom_edges: dict[tuple[str, str], list[tuple[str, str]]],
                       ) -> list[list[str]]:
    """Tarjan SCC over domain nodes; returns every component (sorted members)."""
    graph: dict[str, set[str]] = defaultdict(set)
    for (d1, d2) in dom_edges:
        graph[d1].add(d2)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []
    counter = [0]

    def strongconnect(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph.get(v, ())):
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(sorted(comp))

    for v in sorted(domains):
        if v not in index:
            strongconnect(v)
    return sccs


def cycle_groups_of(domains: list[str],
                    dom_edges: dict[tuple[str, str], list[tuple[str, str]]],
                    ) -> list[list[str]]:
    """Only the SCCs with more than one domain (the actual cycles), sorted."""
    return sorted(c for c in strongly_connected(domains, dom_edges) if len(c) > 1)


def pair_key(d1: str, d2: str) -> str:
    return f"{d1}->{d2}"


def _normalize_body(segment: str) -> str:
    """Normalize a function's source segment for literal-copy comparison:
    dedent, strip trailing whitespace, drop blank lines. Version-stable
    (pure text), unlike ``ast.dump`` whose field set moves between Pythons."""
    lines = [ln.rstrip() for ln in textwrap.dedent(segment).splitlines()]
    return "\n".join(ln for ln in lines if ln)


def duplicate_bodies(manifest: Manifest, root: pathlib.Path = REPO_ROOT,
                     min_lines: int = DUPLICATE_MIN_LINES) -> list[str]:
    """Literal-copy scan: normalized function bodies appearing in modules of
    MORE THAN ONE domain. Returns sorted rows
    ``"<digest16> <path>::<qualname> <path>::<qualname> ..."`` — the format
    pinned in the manifest ``[duplicates].allowed`` baseline."""
    by_digest: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for path in sorted(manifest.modules):
        src_path = root / path
        if not src_path.is_file():
            continue
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path)

        def walk(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = f"{prefix}{child.name}"
                    segment = ast.get_source_segment(source, child)
                    if segment is not None:
                        norm = _normalize_body(segment)
                        if norm.count("\n") + 1 >= min_lines:
                            digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
                            by_digest[digest].append((path, qual))
                    walk(child, f"{qual}.")
                elif isinstance(child, ast.ClassDef):
                    walk(child, f"{prefix}{child.name}.")

        walk(tree, "")

    rows: list[str] = []
    for digest, locs in by_digest.items():
        domains = {manifest.modules[p] for p, _ in locs}
        if len(domains) < 2:
            continue
        parts = " ".join(f"{p}::{q}" for p, q in sorted(set(locs)))
        rows.append(f"{digest} {parts}")
    return sorted(rows)


def computed_graph_sections(manifest: Manifest, graph: ImportGraph) -> dict[str, list]:
    """The generated-section values derived from the live tree."""
    strict_pairs = graph.quotient(STRICT)
    lazy_pairs = graph.quotient(LAZY)
    dynamic_pairs = graph.quotient(DYNAMIC)
    allowed = sorted(pair_key(d1, d2) for (d1, d2) in strict_pairs)
    cycles = cycle_groups_of(sorted(manifest.domains), strict_pairs)
    lazy_only = sorted(pair_key(d1, d2) for (d1, d2) in lazy_pairs
                       if (d1, d2) not in strict_pairs)
    dyn = sorted(pair_key(d1, d2) for (d1, d2) in dynamic_pairs)
    return {
        "allowed": allowed,
        "cycle_groups": cycles,
        "lazy_only": lazy_only,
        "dynamic_pairs": dyn,
    }
