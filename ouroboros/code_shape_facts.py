"""Shape facts of a change's Python sources for the diff pulse: module handles, re-exports, test patches.

Pure functions over source text. ``review.diff_pulse`` reads both sides of each changed Python
file and the tree's module names; ``pulse_line`` states what the change added and removed, a
fact for the mind and never a gate (BIBLE P5, P7; DEVELOPMENT "Module Size & Complexity").
Counted per file and side:

- a module handle: a module-level zero-argument function whose body, after a docstring, is one
  import and the ``return`` of the module it binds, a leaf's call-time reach into its parent
  (``tests/test_module_handle_extraction.py``); and each ``handle().name`` read through one;
- a re-export: a name a top-level ``from ... import`` binds under the ``noqa: F401`` marker
  (``code_intelligence_architecture.reexport_imports``, the facade inventory's convention);
- in a test source (under a ``tests`` directory): each product-module name it patches
  (``monkeypatch.setattr``/``delattr`` with a dotted string or a module, ``mock.patch``,
  ``patch.object``) and each ``git init`` it runs (an argument list or a ``git`` helper call
  whose subcommand is ``init``, or a command string handed to ``subprocess`` or ``os.system``).

A side that does not parse, or was not read, leaves its file unknown, never zero.
"""

from __future__ import annotations

import ast
import collections
import pathlib
import re
from typing import Iterable, Iterator

from ouroboros.code_intelligence import _resolve_relative_import
from ouroboros.code_intelligence_architecture import reexport_imports

# The product modules a test may patch: the domain manifest's runtime population.
PRODUCT_ROOTS = frozenset({"ouroboros", "supervisor", "server", "launcher"})
_GIT_VALUE_OPTIONS = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--config-env"})
_SHELL_RUNNERS = frozenset({"run", "call", "check_call", "check_output", "Popen", "system", "getoutput",
                            "getstatusoutput"})
_SHELL_GIT_INIT = re.compile(r"(?:^|[\s;&|(])git\s+init\b")
# The line's (label, item kind) pairs per group: product sources (False), test sources (True).
_LINE_FACTS = {False: (("module handles", "handle"), ("calls through them", "call"), ("re-exports", "re-export")),
               True: (("patched product-module names", "patch"), ("git init", "git init"))}


def module_names(paths: Iterable[str]) -> frozenset[str]:
    """Dotted names of the Python modules among repository paths, a package by its ``__init__.py``."""
    names = set()
    for path in paths:
        posix = pathlib.PurePosixPath(path)
        if posix.suffix == ".py":
            parts = posix.with_suffix("").parts
            names.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    names.discard("")
    return frozenset(names)


def is_test_source(path: str) -> bool:
    """Python under a ``tests`` directory, the suite's sources as the pulse groups them."""
    return "tests" in pathlib.PurePosixPath(path).parts[:-1]


def _string(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _call_name(func: ast.AST | None) -> str:
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


def _bindings(node: ast.Import | ast.ImportFrom, path: str) -> Iterator[tuple[str, str]]:
    """``(dotted target, bound name)`` for each name an import statement binds."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            target = alias.name if alias.asname else alias.name.split(".")[0]
            yield target, alias.asname or target
        return
    base = _resolve_relative_import(pathlib.PurePosixPath(path), node.module or "", int(node.level or 0))
    for alias in node.names:
        if alias.name != "*":
            yield (f"{base}.{alias.name}" if base else alias.name), alias.asname or alias.name


def _is_handle(node: ast.stmt, path: str, modules: frozenset[str]) -> bool:
    if not isinstance(node, ast.FunctionDef):
        return False
    args = node.args
    if args.posonlyargs or args.args or args.vararg or args.kwonlyargs or args.kwarg:
        return False
    documented = isinstance(node.body[0], ast.Expr) and _string(node.body[0].value) is not None
    body = node.body[1:] if documented else node.body
    if len(body) != 2 or not isinstance(body[0], (ast.Import, ast.ImportFrom)):
        return False
    returned = body[1].value if isinstance(body[1], ast.Return) else None
    return isinstance(returned, ast.Name) and any(
        bound == returned.id and target in modules for target, bound in _bindings(body[0], path))


def _product_items(tree: ast.Module, path: str, lines: list[str], modules: frozenset[str]) -> Iterator[tuple]:
    handles = [node.name for node in tree.body if _is_handle(node, path, modules)]
    yield from (("handle", name) for name in handles)
    names = set(handles)
    for node in ast.walk(tree):
        call = node.value if isinstance(node, ast.Attribute) else None
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in names
                and not call.args and not call.keywords):
            yield "call", call.func.id, node.attr
    for node in reexport_imports(tree, lines):
        source = "." * int(node.level or 0) + (node.module or "")
        yield from (("re-export", source, alias.name, alias.asname or "") for alias in node.names)


def _git_subcommand(args: list[ast.expr]) -> str | None:
    """The subcommand of a ``git`` argument list after its global options; ``None`` when not literal."""
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        value = _string(arg)
        if value is None:
            return None
        if value in _GIT_VALUE_OPTIONS:
            skip = True
        elif not value.startswith("-"):
            return value
    return None


def _test_items(tree: ast.Module, path: str, modules: frozenset[str]) -> Iterator[tuple]:
    bound = {name: target for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
             for target, name in _bindings(node, path)}

    def dotted(node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute):
            owner = dotted(node.value)
            return f"{owner}.{node.attr}" if owner else None
        return bound.get(node.id) if isinstance(node, ast.Name) else None

    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            if node.elts and _string(node.elts[0]) == "git" and _git_subcommand(node.elts[1:]) == "init":
                yield ("git init",)
            continue
        if not isinstance(node, ast.Call):
            continue
        name, args = _call_name(node.func), node.args
        if "git" in name.lower().split("_"):
            rest = list(args)
            while rest and _string(rest[0]) is None and not isinstance(rest[0], ast.Starred):
                rest.pop(0)  # the repository or path a helper takes before its git arguments
            if _git_subcommand(rest) == "init":
                yield ("git init",)
        elif name in _SHELL_RUNNERS and args and _SHELL_GIT_INIT.search(_string(args[0]) or ""):
            yield ("git init",)
        monkeypatch = name in {"setattr", "delattr"} and isinstance(node.func, ast.Attribute)
        target = _string(args[0]) if args and (monkeypatch or name == "patch") else None
        if target is None and len(args) > 1 and _string(args[1]) is not None and (
                monkeypatch or name == "object" and _call_name(getattr(node.func, "value", None)) == "patch"):
            owner = dotted(args[0])
            target = f"{owner}.{args[1].value}" if owner in modules else None
        if target and "." in target and target.split(".", 1)[0] in PRODUCT_ROOTS:
            yield "patch", target


def source_shape(source: str, path: str, modules: frozenset[str]) -> collections.Counter:
    """The shape items of one Python source, by the path's role: a product source's handles, the
    calls through them and its re-exports; a test source's patched product-module names and its
    ``git init`` runs. ``modules`` are the dotted module names of the source's tree. Raises
    ``SyntaxError`` or ``ValueError`` when the text does not parse."""
    tree = ast.parse(source, filename=path)
    if is_test_source(path):
        return collections.Counter(_test_items(tree, path, modules))
    return collections.Counter(_product_items(tree, path, source.splitlines(), modules))


def pulse_line(
    rows: Iterable[tuple[str, str | None, str | None]],
    base_modules: frozenset[str] | None,
    current_modules: frozenset[str] | None,
    *,
    unreadable: Iterable[str] = (),
    limit: int = 5,
) -> str:
    """The pulse's shape line over ``(path, base text, current text)`` rows of changed Python sources,
    ``""`` without any; a text is ``None`` where the file is absent on that side. Module names are
    those of each side's tree, ``None`` when unknown. Added and removed are the items one side holds
    beyond the other; a file named ``unreadable`` (a side that exists but was not read), or whose
    side does not parse, is named and not counted."""
    added = {False: collections.Counter(), True: collections.Counter()}
    removed = {False: collections.Counter(), True: collections.Counter()}
    files, unread, skipped = collections.Counter(), {False: [], True: []}, set(unreadable)
    for path, before, after in rows:
        test = is_test_source(path)
        files[test] += 1
        try:
            if path in skipped or base_modules is None or current_modules is None:
                raise ValueError(path)
            old = collections.Counter() if before is None else source_shape(before, path, base_modules)
            new = collections.Counter() if after is None else source_shape(after, path, current_modules)
        except (SyntaxError, ValueError, RecursionError):
            unread[test].append(path)
            continue
        for item, count in (new - old).items():
            added[test][item[0]] += count
        for item, count in (old - new).items():
            removed[test][item[0]] += count
    parts = []
    for test, facts in _LINE_FACTS.items():
        if not files[test]:
            continue
        missed = unread[test]
        named = ", ".join(missed[:limit]) + (f" (+{len(missed) - limit} more)" if len(missed) > limit else "")
        if len(missed) == files[test]:
            text = f"{', '.join(name for name, _kind in facts)} unknown ({len(missed)} unreadable: {named})"
        else:
            text = ", ".join(f"{name} +{added[test][kind]}/-{removed[test][kind]}" for name, kind in facts)
            text += f", {len(missed)} unreadable not counted: {named}" if missed else ""
        parts.append(("tests: " if test else "") + text)
    return f"Code shape: {'; '.join(parts)}." if parts else ""
