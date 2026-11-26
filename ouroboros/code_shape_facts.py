"""Shape facts of a change's Python sources for the diff pulse: module handles, re-exported names, test patches.

Pure functions over source text. ``review.diff_pulse`` reads both sides of each changed Python
file and the tree's module names; ``pulse_line`` states what the change added and removed, a
fact for the mind and never a gate (BIBLE P5, P7; DEVELOPMENT "Module Size & Complexity").
A name resolves in its lexical scope: the module's bindings, then each enclosing function's own,
each scope holding a name's last binding in source order (a parameter or other assignment hides an
outer name, a comprehension's targets bind inside it alone; ``importlib.import_module("literal")``
binds that module). Counted per file and side:

- a module handle: a module-level zero-argument function whose body, after a docstring, is one
  import and the ``return`` of the module it binds, a leaf's call-time reach into its parent
  (``tests/test_module_handle_extraction.py``); and each ``handle().name`` read through one (the
  extraction table's unit: a local holding the module, ``q = handle()``, is not followed);
- a re-exported name: each name a top-level ``from ... import`` binds under the ``noqa: F401``
  marker (``code_intelligence_architecture.reexported_names``, the facade inventory's unit), by
  its source and the name it binds (``x as x`` is ``x``);
- in a test source (under a ``tests`` directory): each patch of a name in a product module's
  namespace (``monkeypatch.setattr``/``delattr`` or ``mock.patch`` with a dotted string, or with
  an object that resolves to such a name, a module or a class, and ``patch.object``; resolved from
  the source alone, so a patch through a helper's return value or a parameter is not counted), and each
  call site that runs ``git init``: a command runner (a ``subprocess`` call, ``os.system``, or a
  function the module defines that hands its first parameter to one) given
  ``["git", ..., "init"]`` (as its first argument or ``args=``) or a ``git init`` command string;
  a Git helper given ``"init"`` (one the module defines that hands a runner ``["git", ...,
  *<its varargs>]`` with the varargs at the subcommand's place, or a callable whose body the module
  does not hold, imported or a parameter, named for git); a ``for`` over literal argument lists that
  a runner takes, once per list that makes ``git init``. Test data (``pytest.param(["git", "init"])``)
  and a helper that only builds an argument list are never a run.

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
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
# What a scope binds a name to: ("import", dotted name) by an import or a literal importlib call,
# ("handle", name) a module handle, ("git", name) a Git helper, ("run", name) a command runner or
# ("def", name) any other function or class the module defines; None any other binding, which hides
# an outer one.
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
    """The nodes of one scope in source order, without the bodies of the functions, classes and lambdas it
    defines or the comprehensions it holds (they bind their own targets; a first iterable runs here)."""
    nodes, pending = [], list(reversed(scope.body if isinstance(scope.body, list) else [scope.body]))
    while pending:
        node = pending.pop()
        nodes.append(node)
        if isinstance(node, _COMPREHENSIONS):
            pending.append(node.generators[0].iter)
        elif not isinstance(node, _SCOPES):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))
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
    """The names one scope binds itself (its parameters, imports, assignments, definitions and other
    targets), each as its last binding in source order leaves it: a rebinding to a value the module
    cannot name hides an earlier import; ``mark(scope, function, seen)`` binds a function that is more
    than a name."""
    args = getattr(scope, "args", None)
    names: dict[str, Binding] = {} if args is None else {
        arg.arg: None for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg) if arg}
    seen, assigned = collections.ChainMap(names, outer), set()
    for node in _scope_nodes(scope):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((bound, ("import", target)) for target, bound in _bindings(node, path))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            value = _assigned(node.value, seen)
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    names[target.id] = value
                    assigned.add(id(target))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in assigned:
            names[node.id] = None
        elif isinstance(node, _DEFS):
            names[node.name] = mark(scope, node, seen) or ("def", node.name)
        elif isinstance(node, (ast.ClassDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names[node.name] = ("def", node.name) if isinstance(node, ast.ClassDef) else None
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
        elif isinstance(node, _COMPREHENSIONS):  # its targets bind inside it alone; its first iterable runs outside
            first, *rest = node.generators
            inner = nested.new_child({name.id: None for generator in node.generators
                                      for name in ast.walk(generator.target) if isinstance(name, ast.Name)})
            within = [child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.comprehension)]
            pending += [(first.iter, seen, nested)] + [(child, inner, inner) for child in
                                                       (*within, first.target, *first.ifs, *rest)]
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
    yield from (("re-export", base, alias.name, alias.asname or alias.name)
                for base, alias in reexported_names(tree, lines, path))


def _subcommand(args: list[ast.expr]) -> ast.expr | None:
    """The element at a ``git`` argument list's subcommand place, after its global options."""
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        value = _string(arg)
        if value is None or not value.startswith("-"):
            return arg
        skip = value in _GIT_VALUE_OPTIONS
    return None


def _git_argv(node: ast.AST | None) -> list[ast.expr] | None:
    """The elements after ``"git"`` of a literal ``["git", ...]`` argument list, ``None`` otherwise."""
    elements = node.elts if isinstance(node, (ast.List, ast.Tuple)) else []
    return elements[1:] if elements and _string(elements[0]) == "git" else None


def _initializes(args: list[ast.expr] | None) -> bool:
    """Whether a ``git`` argument list's subcommand is the literal ``init``."""
    return _string(_subcommand(args or [])) == "init"


def _argv(call: ast.Call) -> ast.expr | None:
    """The command a runner call is given: its first argument, or its ``args=``."""
    return call.args[0] if call.args else next((item.value for item in call.keywords if item.arg == "args"), None)


def _runner(call: ast.Call, seen: Mapping[str, Binding]) -> str:
    """``"command"`` for a call that runs the command it is given (a ``subprocess`` function,
    ``os.system``, a wrapper of one defined here), ``"git"`` for a Git helper (one defined here whose
    body passes its own arguments to ``git`` as the subcommand, or a callable whose body is not here,
    imported or a parameter, named for git), else ``""``."""
    dotted = _dotted(call.func, seen)
    kind = (seen.get(call.func.id) or ("",))[0] if isinstance(call.func, ast.Name) else ""
    if dotted in _COMMAND_RUNNERS or kind == "run":
        return "command"
    name = dotted or getattr(call.func, "id", "")
    return "git" if kind == "git" or kind != "def" and "git" in name.rsplit(".", 1)[-1].lower().split("_") else ""


def _commands(node: ast.AST, seen: Mapping[str, Binding]) -> list[ast.expr]:
    """The command of each command-runner call within ``node``."""
    return [argv for call in ast.walk(node) if isinstance(call, ast.Call) and _runner(call, seen) == "command"
            and (argv := _argv(call)) is not None]


def _passes_through(argv: ast.expr, rest: str, named: bool) -> bool:
    """Whether a command list hands ``*rest`` to ``git`` as its subcommand: ``["git", <global options>,
    *rest, ...]`` (another starred name there holds options), or, for a function named for git, a list
    that a starred constant prefix (``[*GIT, *rest]``) starts."""
    def spread(item: ast.AST | None) -> bool:
        return isinstance(item, ast.Starred) and getattr(item.value, "id", "") == rest

    head = _git_argv(argv)
    if head is not None:
        return spread(_subcommand([item for item in head if spread(item) or not isinstance(item, ast.Starred)]))
    items = argv.elts if isinstance(argv, (ast.List, ast.Tuple)) else []
    return named and bool(items) and isinstance(items[0], ast.Starred) and any(map(spread, items))


def _git_helper(node: ast.AST, seen: Mapping[str, Binding], path: str) -> Binding:
    """A function defined here that runs ``git`` with the subcommand it is given (it hands a command
    runner its varargs, ``_passes_through``), or that hands its first parameter to a command runner;
    one that only builds an argument list runs nothing. Its body is read with its own imports too."""
    seen = collections.ChainMap({bound: ("import", target) for item in _scope_nodes(node) if isinstance(
        item, (ast.Import, ast.ImportFrom)) for target, bound in _bindings(item, path)}, seen)
    first = ([*node.args.posonlyargs, *node.args.args][:1] or [None])[0]
    rest, named, hands = getattr(node.args.vararg, "arg", None), "git" in node.name.lower().split("_"), False
    for argv in () if node.name.startswith("test_") else _commands(node, seen):  # a test is collected, never called
        if rest and _passes_through(argv, rest, named):
            return "git", node.name
        hands = hands or first is not None and getattr(argv, "id", "") == first.arg
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
        return sum(1 for item in node.iter.elts if whole and _initializes(_git_argv(item))
                   or spread and isinstance(item, (ast.List, ast.Tuple)) and _initializes(item.elts))
    kind = _runner(node, seen) if isinstance(node, ast.Call) else ""
    if kind == "command":
        argv = _argv(node)
        return int(_initializes(_git_argv(argv)) or bool(_SHELL_GIT_INIT.search(_string(argv) or "")))
    rest = list(node.args) if kind == "git" else []
    while rest and _string(rest[0]) is None and not isinstance(rest[0], ast.Starred):
        rest.pop(0)  # the repository or path a helper takes before its git arguments
    return int(_initializes(rest))


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
    for node, seen in _scoped(tree, path, lambda _scope, node, seen: _git_helper(node, seen, path)):
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


def _is_governance_markdown(path: str) -> bool:
    """Whether ``path`` is a tracked governance or specification Markdown document."""
    from ouroboros.reference_books import book_path_role, ruled_text_of, standalone_reference_of

    norm = str(path or "").replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    if book_path_role(norm) or ruled_text_of(norm) or standalone_reference_of(norm):
        return True
    return norm.startswith("docs/") and norm.endswith(".md")


def body_ownership_pulse(
    root: pathlib.Path,
    shown: list[str],
    staged: bool,
    *,
    all_changed: Iterable[str] | None = None,
) -> str:
    """Factual ownership line: touched code/prompt/skill paths -> owning manuals & matching invariants.

    Reads ``ouroboros/domains.toml`` ``[references]`` unioned with exact-path chapter mentions and
    all numbered invariants in ``docs/architecture/10-key-invariants.md`` (matching exact paths,
    extension-qualified basenames, snake_case stems, or qualified ``stem.``/``stem::`` symbols).
    Purely diagnostic, never a gate.
    """
    from ouroboros.reference_books import _git_bytes, resolve_body_references, source_reader

    code_paths = [p for p in shown if not str(p).startswith(("docs/", "tests/"))]
    if not code_paths:
        return ""
    diff_universe = shown if all_changed is None else all_changed
    changed_set = {str(p).replace("\\", "/").removeprefix("./") for p in diff_universe}
    idx_reader = source_reader(root, source="index") if staged else None
    dirty_wt: set[str] = set()
    if staged:
        wt_raw = _git_bytes(root, "--no-optional-locks", "diff", "--name-only", "-z")
        untracked_raw = _git_bytes(root, "--no-optional-locks", "ls-files", "--others", "--exclude-standard", "-z", "--", "docs/architecture", "docs/development")
        if wt_raw is None or untracked_raw is None:
            dirty_wt.add("*")
        else:
            dirty_wt = {p.replace("\\", "/").removeprefix("./") for raw in (wt_raw, untracked_raw) for p in raw.decode("utf-8", "replace").split("\0") if p}
    index_needed = changed_set | dirty_wt

    def _fast_staged_reader(rel: str) -> bytes:
        norm_rel = str(rel).replace("\\", "/").removeprefix("./")
        if idx_reader is not None and ("*" in index_needed or norm_rel in index_needed):
            return idx_reader(norm_rel)
        return (root / norm_rel).read_bytes()

    reader = _fast_staged_reader if (staged and ("*" in index_needed or any(
        p in {"ouroboros/domains.toml", "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md"} or p.startswith(("docs/architecture/", "docs/development/"))
        for p in index_needed
    ))) else None
    refs = resolve_body_references(root, code_paths, read_bytes=reader, include_mentions=True)
    manuals = refs.get("manuals") or ()
    if not manuals:
        return ""
    manual_tags = [f"{m} ({'touched' if m in changed_set else 'untouched in diff'})" for m in manuals]
    matched_invariants: list[str] = []
    try:
        inv_bytes = (_fast_staged_reader if staged else (lambda p: (root / p).read_bytes()))("docs/architecture/10-key-invariants.md")
        inv_text = inv_bytes.decode("utf-8", errors="replace")
        exact_tokens: set[str] = set()
        qualified_stems: set[str] = set()
        for p in code_paths:
            norm = str(p).replace("\\", "/").removeprefix("./")
            base = norm.rsplit("/", 1)[-1]
            stem = base.rsplit(".", 1)[0] if "." in base else base
            exact_tokens.add(norm)
            if "." in base:
                exact_tokens.add(base)
            if "_" in stem:
                exact_tokens.add(stem)
            elif len(stem) >= 3:
                qualified_stems.add(stem)
        for line in inv_text.splitlines():
            m = re.match(r"^(\d+)\.\s+", line)
            if not m:
                continue
            hit = any(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(tok)}(?![A-Za-z0-9_-])", line) for tok in exact_tokens)
            if not hit and qualified_stems:
                hit = any(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(stem)}(?:\.|::)", line) for stem in qualified_stems)
            if hit:
                matched_invariants.append(f"§10#{m.group(1)}")
    except (OSError, ValueError):
        pass
    inv_part = f"; matching invariants: {', '.join(matched_invariants)}" if matched_invariants else ""
    return f"Owning manuals: {', '.join(manual_tags)}{inv_part}."


def removed_governance_prose_pulse(
    root: pathlib.Path,
    base: str,
    staged: bool,
    shown: list[str],
    change_texts_fn: Callable[..., tuple[dict[str, str | None], dict[str, str | None]]],
    *,
    all_changed: Iterable[str] | None = None,
) -> str:
    """P5-clean set-diff of removed prose lines in touched governance Markdown vs ``base``.

    Excludes lines that still appear verbatim in ANY changed governance doc across ``all_changed``
    in the current tree/index so splits and relocations across separate edits produce zero noise.
    Zero keyword heuristics.
    """
    gov_shown = [p for p in shown if _is_governance_markdown(p)]
    if not gov_shown:
        return ""
    diff_universe = shown if all_changed is None else all_changed
    gov_all = list(dict.fromkeys([*gov_shown, *(p for p in diff_universe if _is_governance_markdown(p))]))
    at_base, current = change_texts_fn(root, base, staged, gov_shown, gov_all)
    surviving_lines = {
        line.strip()
        for p in gov_all
        for line in (current.get(p) or "").splitlines()
        if line.strip()
    }
    removed: list[tuple[str, int, str]] = []
    for p in gov_shown:
        old = at_base.get(p)
        if not old:
            continue
        in_fence = False
        for lineno, line in enumerate(old.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or not stripped or stripped.startswith(("#", "---", "|---", "<!--")):
                continue
            if stripped not in surviving_lines:
                removed.append((p, lineno, stripped))
    if not removed:
        return ""
    cap = 12
    sample = [f"  - {p}:{ln}: {line[:140]}{'…' if len(line) > 140 else ''}" for p, ln, line in removed[:cap]]
    diff_flag = "--cached " if staged else ""
    more = (
        f"\n  (+{len(removed) - cap} more removed governance prose line(s); "
        f"inspect full diff via `git diff {diff_flag}{base[:12]} -- {' '.join(gov_shown[:4])}`)"
        if len(removed) > cap else ""
    )
    return f"Removed governance prose lines ({len(removed)} not relocated verbatim):\n" + "\n".join(sample) + more


def diff_pulse_tail_parts(
    root: pathlib.Path,
    base: str,
    staged: bool,
    shown: list[str],
    change_texts_fn: Callable[..., tuple[dict[str, str | None], dict[str, str | None]]],
    all_changed: Iterable[str] | None,
    tail_question: str,
) -> tuple[str, ...]:
    """Assemble diagnostic diff_pulse tail lines (ownership, removed governance prose, book balance, prompt)."""
    from ouroboros.reference_books import book_balance_note

    diff_paths = shown if all_changed is None else list(all_changed)
    return (
        body_ownership_pulse(root, shown, staged, all_changed=diff_paths),
        removed_governance_prose_pulse(root, base, staged, shown, change_texts_fn, all_changed=diff_paths),
        book_balance_note(root, shown, source="index" if staged else "worktree"),
        tail_question,
    )
