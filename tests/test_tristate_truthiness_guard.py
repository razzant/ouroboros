"""A tri-state capability reader answers True, False or None: never test it for truth.

``supports_vision`` returns ``None`` when no evidence of the exact route says
whether it accepts images. ``None`` means "unknown", and unknown never withholds
the owner's input: the image goes as it is and the route answers. A truthiness
test (``if supports_vision(m):``, ``not ...``, ``a or b``, ``bool(...)``, a
ternary, ``... or False``) silently turns that unknown into "no", which is how the
transport builder kept replacing images of every unlisted model with a
placeholder after the oracle already said "unknown" for subscription routes.

This is a SOURCE lint over the runtime packages, not a runtime gate. It flags a
reader call (bare name, import alias including function-local imports, or a
qualified ``module.supports_vision(...)``) and a local variable assigned from one,
whenever the IMMEDIATE AST parent consumes its truth value. A call whose parent is
a comparison such as ``is False`` / ``is not False`` / ``is None`` is the honest
form and stays allowed even inside an ``if``, ``and``/``or`` or ternary.

Out of reach (disclosed, not pretended): a reader value stored on an attribute or
container and tested later, a wrapper function that returns the reader's value
(add the wrapper to ``TRISTATE_READERS``), and ``any()``/``all()`` over reader
calls. ``ALLOWED`` is the residual disclosure: an entry states why that one site
may treat unknown as no.
"""

from __future__ import annotations

import ast
import functools
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
SCAN_DIRS = ("ouroboros", "supervisor")
SCAN_FILES = ("server.py",)

# Readers whose None means "unknown", not "no". A successor reader joins this set
# in the commit that introduces it: ``image_input_from_row`` is the one catalog-row
# parser and ``_image_input_verdict`` the policy's fail-soft read (``vision_routing``).
# ``route_image_input`` returns an evidence record whose ``verdict`` attribute is out
# of this lint's reach (see the module docstring).
TRISTATE_READERS = frozenset({"supports_vision", "image_input_from_row", "_image_input_verdict"})

# (repo-relative path, line number) -> why this site may treat unknown as no.
ALLOWED: dict[tuple[str, int], str] = {}


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _truthiness_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """The construct that consumes ``node``'s truth value, or "" when none does."""
    parent = parents.get(node)
    if isinstance(parent, ast.NamedExpr) and parent.value is node:
        # ``if (v := reader(m)):`` tests the call's value through the walrus.
        return _truthiness_context(parent, parents)
    if isinstance(parent, (ast.If, ast.While)) and parent.test is node:
        return type(parent).__name__.lower()
    if isinstance(parent, ast.IfExp) and parent.test is node:
        return "ternary"
    if isinstance(parent, ast.Assert) and parent.test is node:
        return "assert"
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        return "not"
    if isinstance(parent, ast.BoolOp):
        return "and" if isinstance(parent.op, ast.And) else "or"
    if isinstance(parent, ast.comprehension) and node in parent.ifs:
        return "comprehension-if"
    if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
            and parent.func.id == "bool" and node in parent.args):
        return "bool()"
    return ""


def _scope_of(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    scope = parents.get(node)
    while scope is not None and not isinstance(
            scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)):
        scope = parents.get(scope)
    return scope


def _assigned_names(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    parent = parents.get(node)
    if isinstance(parent, ast.Assign) and parent.value is node:
        return [target.id for target in parent.targets if isinstance(target, ast.Name)]
    if isinstance(parent, ast.AnnAssign) and parent.value is node and isinstance(parent.target, ast.Name):
        return [parent.target.id]
    if isinstance(parent, ast.NamedExpr) and parent.value is node:
        return [parent.target.id]
    return []


def truthiness_hits(source: str) -> list[tuple[int, str, str]]:
    """``(line, what, context)`` for every truthiness use of a tri-state reader."""
    tree = ast.parse(source)
    parents = _parents(tree)
    reader_names = set(TRISTATE_READERS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in TRISTATE_READERS:
                    reader_names.add(alias.asname or alias.name)

    def is_reader_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in reader_names
        return isinstance(func, ast.Attribute) and func.attr in TRISTATE_READERS

    hits: list[tuple[int, str, str]] = []
    tristate_locals: set[tuple[ast.AST, str]] = set()
    for node in ast.walk(tree):
        if not is_reader_call(node):
            continue
        context = _truthiness_context(node, parents)
        if context:
            hits.append((node.lineno, "call", context))
        scope = _scope_of(node, parents)
        for name in _assigned_names(node, parents):
            tristate_locals.add((scope, name))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        if (_scope_of(node, parents), node.id) not in tristate_locals:
            continue
        context = _truthiness_context(node, parents)
        if context:
            hits.append((node.lineno, f"local {node.id}", context))
    return sorted(hits)


def _sources():
    paths = [REPO / name for name in SCAN_FILES]
    for directory in SCAN_DIRS:
        paths.extend(sorted((REPO / directory).rglob("*.py")))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # Every reader use spells the reader's own name somewhere in the file (an
        # alias import still names it), so a file without one cannot hold a hit.
        if any(name in text for name in TRISTATE_READERS):
            yield path.relative_to(REPO).as_posix(), text


@functools.lru_cache(maxsize=1)
def _repository_hits() -> dict[tuple[str, int], str]:
    found: dict[tuple[str, int], str] = {}
    for rel, text in _sources():
        for line, what, context in truthiness_hits(text):
            found[(rel, line)] = f"{what} in {context}"
    return found


def test_no_tristate_capability_reader_is_tested_for_truth():
    hits = _repository_hits()
    unexplained = {key: value for key, value in hits.items() if key not in ALLOWED}
    assert not unexplained, (
        "A tri-state capability reader was tested for truth, so its 'unknown' (None) "
        "acts as 'no'. Compare explicitly (`is False` withholds only on evidence; "
        "`is True` requires evidence), or add the site to ALLOWED with a written reason.\n"
        + "\n".join(f"  {rel}:{line}: {why}" for (rel, line), why in sorted(unexplained.items()))
    )


def test_the_allowlist_has_no_stale_entries():
    hits = _repository_hits()
    stale = sorted(key for key in ALLOWED if key not in hits)
    assert not stale, f"ALLOWED entries no longer match a truthiness use: {stale}"


def test_the_scan_reaches_the_reader_and_its_consumers():
    scanned = {rel for rel, _text in _sources()}
    # The reader's definition and at least one consumer must be in view, or a
    # relocated module would silence the lint above.
    assert "ouroboros/provider_models.py" in scanned
    assert len(scanned) >= 2, scanned


def test_the_lint_catches_every_truthiness_form():
    caught = {
        "if": "from ouroboros.provider_models import supports_vision\nif supports_vision(m):\n    pass\n",
        "not": "from ouroboros.provider_models import supports_vision\nok = not supports_vision(m)\n",
        "or False": "from ouroboros.provider_models import supports_vision\nok = supports_vision(m) or False\n",
        "and": "from ouroboros.provider_models import supports_vision\nok = flag and supports_vision(m)\n",
        "bool": "from ouroboros.provider_models import supports_vision\nok = bool(supports_vision(m))\n",
        "ternary": "from ouroboros.provider_models import supports_vision\nx = a if supports_vision(m) else b\n",
        "assert": "from ouroboros.provider_models import supports_vision\nassert supports_vision(m)\n",
        "comprehension-if": (
            "from ouroboros.provider_models import supports_vision\n"
            "xs = [m for m in models if supports_vision(m)]\n"),
        "while": "from ouroboros.provider_models import supports_vision\nwhile supports_vision(m):\n    break\n",
        "walrus": (
            "from ouroboros.provider_models import supports_vision\n"
            "if (v := supports_vision(m)):\n    pass\n"),
        "alias import": "from ouroboros.provider_models import supports_vision as sees\nif sees(m):\n    pass\n",
        "function-local import": (
            "def build(m):\n"
            "    from ouroboros.provider_models import supports_vision\n"
            "    if not (supports_vision(m) or supports_vision(n)):\n"
            "        return 1\n"),
        "qualified call": "from ouroboros import provider_models\nif provider_models.supports_vision(m):\n    pass\n",
        "local assignment": (
            "from ouroboros.provider_models import supports_vision\n"
            "def f(m):\n"
            "    verdict = supports_vision(m)\n"
            "    if not verdict:\n"
            "        return 1\n"),
    }
    for label, source in caught.items():
        assert truthiness_hits(source), label


def test_the_lint_allows_explicit_tristate_comparisons():
    ignored = {
        "is False inside if/and": (
            "from ouroboros.provider_models import supports_vision\n"
            "if mode == 'auto' and supports_vision(m, model_role='main') is not False:\n    pass\n"),
        "is False inside ternary/or": (
            "from ouroboros.provider_models import supports_vision\n"
            "x = '' if local or supports_vision(m) is False else m\n"),
        "is True": "from ouroboros.provider_models import supports_vision\nif supports_vision(m) is True:\n    pass\n",
        "is None": "from ouroboros.provider_models import supports_vision\nunknown = supports_vision(m) is None\n",
        "value passed on": "from ouroboros.provider_models import supports_vision\nrecord(verdict=supports_vision(m))\n",
        "local compared": (
            "from ouroboros.provider_models import supports_vision\n"
            "def f(m):\n"
            "    verdict = supports_vision(m)\n"
            "    if verdict is False:\n"
            "        return 1\n"),
        "unrelated local of the same name in another scope": (
            "from ouroboros.provider_models import supports_vision\n"
            "def f(m):\n"
            "    verdict = supports_vision(m)\n"
            "    return verdict\n"
            "def g(verdict):\n"
            "    if verdict:\n"
            "        return 1\n"),
        "unrelated function": "if supports_images(m):\n    pass\n",
    }
    for label, source in ignored.items():
        assert not truthiness_hits(source), label
