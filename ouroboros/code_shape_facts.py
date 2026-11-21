"""Shape facts of a change's Python sources for the diff pulse: module handles, re-exported names, test patches.

Pure functions over source text. ``review.diff_pulse`` reads both sides of each changed Python
file and the tree's module names; ``pulse_line`` states what the change added and removed, a
fact for the mind and never a gate (BIBLE P5, P7; DEVELOPMENT "Module Size & Complexity").
A name resolves in its lexical scope: the module's bindings, then each enclosing function's own
(a parameter or other assignment hides an outer name; ``importlib.import_module("literal")``
binds that module). Counted per file and side:

- a module handle: a module-level zero-argument function whose body, after a docstring, is one
  import and the ``return`` of the module it binds, a leaf's call-time reach into its parent
  (``tests/test_module_handle_extraction.py``); and each ``handle().name`` read through one (the
  extraction table's unit: a local holding the module, ``q = handle()``, is not followed);
- a re-exported name: each name a top-level ``from ... import`` binds under the ``noqa: F401``
  marker (``code_intelligence_architecture.reexported_names``, the facade inventory's unit);
- in a test source (under a ``tests`` directory): each patch of a name in a product module's
  namespace (``monkeypatch.setattr``/``delattr`` or ``mock.patch`` with a dotted string, or with
  an object that resolves to such a name, a module or a class, and ``patch.object``), and each
  call site that runs ``git init``: a command runner (a ``subprocess`` call, ``os.system``, or a
  function the module defines that hands its first parameter to one) given
  ``["git", ..., "init"]`` or a ``git init`` command string; a Git helper given ``"init"`` (one
  the module defines around ``["git", *args]``, or any callable named for git); a ``for`` over
  literal argument lists that a runner takes, once per list that makes ``git init``. Test data
  (``pytest.param(["git", "init"])``) is never a run.

A side that does not parse, or was not read, leaves its file unknown, never zero.
"""

from __future__ import annotations

import ast
import collections
import pathlib
import re
from typing import Callable, Iterable, Iterator, Mapping, Optional, Tuple

from ouroboros.code_intelligence import _resolve_relative_import
from ouroboros.code_intelligence_architecture import reexported_names

# The product modules a test may patch: the domain manifest's runtime population.
PRODUCT_ROOTS = frozenset({"ouroboros", "supervisor", "server", "launcher"})
_GIT_VALUE_OPTIONS = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--config-env"})
# The calls that run a command, given its argument list or a string the shell reads.
_COMMAND_RUNNERS = frozenset({"os.system", "os.popen"} | {f"subprocess.{name}" for name in (
    "run", "call", "check_call", "check_output", "Popen", "getoutput", "getstatusoutput")})
_SHELL_GIT_INIT = re.compile(r"(?:^|[\s;&|(])git\s+init\b")
# The line's (label, item kind) pairs per group: product sources (False), test sources (True).
_LINE_FACTS = {False: (("module handles", "handle"), ("calls through them", "call"), ("re-exported names", "re-export")),
               True: (("patched product-module names", "patch"), ("git init call sites", "git init"))}
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (*_DEFS, ast.ClassDef, ast.Lambda)
# What a scope binds a name to: ("import", dotted name) by an import or a literal importlib call,
# ("handle", name) a module handle, ("git", name) a Git helper or ("run", name) a command runner the
# module defines; None any other binding, which hides an outer one.
Binding = Optional[Tuple[str, str]]


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


def _scope_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of one scope, without the bodies of the functions, classes and lambdas it defines."""
    nodes, pending = [], list(scope.body) if isinstance(scope.body, list) else [scope.body]
    while pending:
        node = pending.pop()
        nodes.append(node)
        if not isinstance(node, _SCOPES):
            pending.extend(ast.iter_child_nodes(node))
    return nodes


def _dotted(node: ast.AST, seen: Mapping[str, Binding]) -> str | None:
    """The dotted name an expression reaches through an import binding, ``None`` otherwise."""
    if isinstance(node, ast.Attribute):
        owner = _dotted(node.value, seen)
        return f"{owner}.{node.attr}" if owner else None
    binding = seen.get(node.id) if isinstance(node, ast.Name) else None
    return binding[1] if binding and binding[0] == "import" else None


def _through(node: ast.AST, seen: Mapping[str, Binding]) -> str | None:
    """The handle a ``handle()`` call reads its module through, ``None`` for any other expression."""
    named = isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args and not node.keywords
    binding = seen.get(node.func.id) if named else None
    return binding[1] if binding and binding[0] == "handle" else None


def _assigned(value: ast.AST, seen: Mapping[str, Binding]) -> Binding:
    """The binding an assigned value carries: a copied name's or dotted import's, or a literal
    ``importlib.import_module``'s."""
    if isinstance(value, ast.Name):
        return seen.get(value.id)
    if isinstance(value, ast.Attribute):
        dotted = _dotted(value, seen)
        return ("import", dotted) if dotted else None
    module = _string(value.args[0]) if isinstance(value, ast.Call) and value.args else None
    if module and not module.startswith(".") and _dotted(value.func, seen) == "importlib.import_module":
        return "import", module
    return None


Mark = Callable[[ast.AST, ast.AST, Mapping[str, Binding]], Binding]


def _own_bindings(scope: ast.AST, outer: Mapping[str, Binding], path: str, mark: Mark) -> dict[str, Binding]:
    """The names one scope binds itself (its parameters, imports, assignments and definitions);
    ``mark(scope, function, seen)`` binds a function that is more than a name hiding an outer one."""
    nodes = _scope_nodes(scope)
    args = getattr(scope, "args", None)
    names: dict[str, Binding] = {} if args is None else {
        arg.arg: None for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg) if arg}
    for node in nodes:
        name = (node.id if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                else node.name if isinstance(node, (*_DEFS, ast.ClassDef, ast.ExceptHandler)) else None)
        if name:
            names[name] = None
    names.update((bound, ("import", target)) for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom))
                 for target, bound in _bindings(node, path))
    seen = collections.ChainMap(names, outer)
    for node in nodes:
        if isinstance(node, _DEFS) and (binding := mark(scope, node, seen)):
            names[node.name] = binding
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            value = _assigned(node.value, seen)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update((target.id, value) for target in targets if value is not None and isinstance(target, ast.Name))
    return names


def _scoped(tree: ast.Module, path: str, mark: Mark) -> Iterator[tuple[ast.AST, Mapping[str, Binding]]]:
    """Each node with the bindings its lexical scope sees: the module's, then each enclosing
    function's own; a class body's names stay in that body."""
    module = collections.ChainMap(_own_bindings(tree, {}, path, mark))
    pending = [(node, module, module) for node in tree.body]
    while pending:
        node, seen, nested = pending.pop()
        yield node, seen
        if isinstance(node, (*_DEFS, ast.Lambda)):
            inner = nested.new_child(_own_bindings(node, nested, path, mark))
            outside = [*getattr(node, "decorator_list", ()), *node.args.defaults, *filter(None, node.args.kw_defaults)]
            body = node.body if isinstance(node.body, list) else [node.body]
            pending += [(child, seen, nested) for child in outside] + [(child, inner, inner) for child in body]
        elif isinstance(node, ast.ClassDef):
            inner = seen.new_child(_own_bindings(node, seen, path, mark))
            outside = [*node.decorator_list, *node.bases, *node.keywords]
            pending += [(child, seen, nested) for child in outside] + [(child, inner, nested) for child in node.body]
        else:
            pending += [(child, seen, nested) for child in ast.iter_child_nodes(node)]


def _product_items(tree: ast.Module, path: str, lines: list[str], modules: frozenset[str]) -> Iterator[tuple]:
    handles = {node.name for node in tree.body if _is_handle(node, path, modules)}
    yield from (("handle", name) for name in sorted(handles))

    def mark(scope: ast.AST, node: ast.AST, _seen: Mapping[str, Binding]) -> Binding:
        return ("handle", node.name) if scope is tree and node.name in handles else None

    for node, seen in _scoped(tree, path, mark):
        if isinstance(node, ast.Attribute) and (handle := _through(node.value, seen)):
            yield "call", handle, node.attr
    yield from (("re-export", base, alias.name, alias.asname or "")
                for base, alias in reexported_names(tree, lines, path))


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


def _git_argv(node: ast.AST) -> list[ast.expr] | None:
    """The elements after ``"git"`` of a literal ``["git", ...]`` argument list, ``None`` otherwise."""
    elements = node.elts if isinstance(node, (ast.List, ast.Tuple)) else []
    return elements[1:] if elements and _string(elements[0]) == "git" else None


def _runner(call: ast.Call, seen: Mapping[str, Binding]) -> str:
    """``"command"`` for a call that runs the command its first argument holds (a ``subprocess``
    function, ``os.system``, a wrapper of one defined here), ``"git"`` for a Git helper (one defined
    here around ``["git", *args]``, or named for git), else ``""``."""
    dotted = _dotted(call.func, seen)
    kind = (seen.get(call.func.id) or ("",))[0] if isinstance(call.func, ast.Name) else ""
    if dotted in _COMMAND_RUNNERS or kind == "run":
        return "command"
    name = dotted or (call.func.id if isinstance(call.func, ast.Name) else "")
    return "git" if kind == "git" or "git" in name.rsplit(".", 1)[-1].lower().split("_") else ""


def _commands(node: ast.AST, seen: Mapping[str, Binding]) -> list[ast.expr]:
    """The first argument of each command-runner call within ``node``."""
    return [call.args[0] for call in ast.walk(node)
            if isinstance(call, ast.Call) and call.args and _runner(call, seen) == "command"]


def _git_helper(_scope: ast.AST, node: ast.AST, seen: Mapping[str, Binding]) -> Binding:
    """A function defined here that runs ``git`` with the arguments it is given (``["git", *args]``,
    or a ``["git", ...]`` list beside a ``*args`` parameter), or that hands its first parameter to a
    command runner."""
    first = ([*node.args.posonlyargs, *node.args.args][:1] or [None])[0]
    lists, hands = [], False
    for item in () if node.name.startswith("test_") else ast.walk(node):  # a test is collected, never called
        if (argv := _git_argv(item)) is not None:
            lists.append(argv)
        elif first and isinstance(item, ast.Call) and item.args and getattr(item.args[0], "id", "") == first.arg:
            hands = hands or _runner(item, seen) == "command"
    if lists and (node.args.vararg or any(isinstance(element, ast.Starred) for argv in lists for element in argv)):
        return "git", node.name
    return ("run", node.name) if hands else None


def _git_inits(node: ast.AST, seen: Mapping[str, Binding]) -> int:
    """The ``git init`` call sites one node is: a command runner given ``["git", ..., "init"]`` or a
    ``git init`` string; a Git helper given ``"init"`` after the repository or path it takes first; a
    ``for`` over literal argument lists whose variable a command runner takes (as the argv, or
    spread into ``["git", ..., *variable]``), once per list that makes ``git init``."""
    if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and isinstance(node.iter, (ast.List, ast.Tuple)):
        name, commands = node.target.id, _commands(node, seen)
        whole = any(isinstance(argv, ast.Name) and argv.id == name for argv in commands)
        spread = any(isinstance(element, ast.Starred) and getattr(element.value, "id", "") == name
                     for argv in commands for element in _git_argv(argv) or ())
        return sum(1 for item in node.iter.elts
                   if whole and _git_subcommand(_git_argv(item) or []) == "init"
                   or spread and isinstance(item, (ast.List, ast.Tuple)) and _git_subcommand(item.elts) == "init")
    kind = _runner(node, seen) if isinstance(node, ast.Call) else ""
    if kind == "command":
        argv = node.args[0] if node.args else None
        return int(_git_subcommand(_git_argv(argv) or []) == "init" or bool(_SHELL_GIT_INIT.search(_string(argv) or "")))
    rest = list(node.args) if kind == "git" else []
    while rest and _string(rest[0]) is None and not isinstance(rest[0], ast.Starred):
        rest.pop(0)  # the repository or path a helper takes before its git arguments
    return int(_git_subcommand(rest) == "init")


def _patch_target(call: ast.Call, seen: Mapping[str, Binding], modules: frozenset[str]) -> str | None:
    """The product name a patch call replaces: a dotted string, or an object in a product module's
    namespace (the module, a class it defines) and the attribute named beside it."""
    name, args = _call_name(call.func), call.args
    monkeypatch = name in {"setattr", "delattr"} and isinstance(call.func, ast.Attribute)
    target = _string(args[0]) if args and (monkeypatch or name == "patch") else None
    if target is None and len(args) > 1 and _string(args[1]) is not None and (
            monkeypatch or name == "object" and _call_name(getattr(call.func, "value", None)) == "patch"):
        parts = (_dotted(args[0], seen) or "").split(".")
        if any(".".join(parts[:end]) in modules for end in range(len(parts), 0, -1)):
            target = f"{'.'.join(parts)}.{args[1].value}"
    return target if target and "." in target and target.split(".", 1)[0] in PRODUCT_ROOTS else None


def _test_items(tree: ast.Module, path: str, modules: frozenset[str]) -> Iterator[tuple]:
    for node, seen in _scoped(tree, path, _git_helper):
        yield from (("git init",) for _ in range(_git_inits(node, seen)))
        if isinstance(node, ast.Call) and (target := _patch_target(node, seen, modules)):
            yield "patch", target


def source_shape(source: str, path: str, modules: frozenset[str]) -> collections.Counter:
    """The shape items of one Python source, by the path's role: a product source's handles, the
    calls through them and its re-exported names; a test source's patched product names and its
    ``git init`` call sites. ``modules`` are the dotted module names of the source's tree. Raises
    ``SyntaxError``, ``ValueError``, ``RecursionError`` or ``MemoryError`` when the text does not
    parse (``MemoryError`` is CPython's "too complex to parse")."""
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
        except (SyntaxError, ValueError, RecursionError, MemoryError):
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
