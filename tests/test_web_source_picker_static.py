"""Static guard: a model source is CHOSEN, never spelled (``docs/DESIGN.md`` §7).

The owner's report was that Settings → Models lets you PICK a provider from a
grouped select while Settings → Agents makes you TYPE the ``::`` prefix, and
that the surrounding help copy teaches that prefix as if it were the interface.
It is not: ``provider::model``, ``claudexor::source=model`` and ``harness=model``
are serialization the editor authors from the chosen source
(``ouroboros/provider_models.py`` is the backend SSOT for the spellings).

Two structural facts hold that rule closed, and both are checkable without a
browser:

1. no owner-facing field placeholder or help string instructs the prefix, so
   the copy never re-teaches what the editor is supposed to author;
2. every model-assigning editor takes its source list from ONE primitive
   (``web/modules/route_editor_primitives.js``), so the groups and their order
   cannot drift apart per surface.

Pattern follows ``tests/test_web_dialogs_static.py`` and
``tests/test_web_typography_static.py``: read the sources, assert the
structural fact, no browser needed, so the quick tier enforces it.
"""

from __future__ import annotations

import ast
import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "web"
WEB_MODULES = WEB / "modules"
INDEX_HTML = WEB / "index.html"
SETUP_CONTRACT = REPO_ROOT / "ouroboros" / "settings_setup_contract.py"
DESIGN_DOC = REPO_ROOT / "docs" / "DESIGN.md"
PRIMITIVES = WEB_MODULES / "route_editor_primitives.js"

# The serialized separator the owner must never be asked to type.
PREFIX = "::"

# Copy-carrying keys. A `::` inside one of these values is owner-facing text:
# a field placeholder, a card note, a hint, a step footer. Everything else in
# JS -- `value.includes('::')`, `` `claudexor::${source}=` ``, a lastIndexOf
# separator -- is the editor doing its own serialization, which is the point of
# the rule rather than a violation of it.
COPY_KEYS = (
    "placeholder",      # also matches apiPlaceholder / modelPlaceholder
    "note",
    "hint",
    "help",
    "helpText",
    "description",
    "copy",             # also matches railCopy / providerCopy / modelCopy
    "footer",
    "emptyText",
    "label",
    "title",
)

# `key: '...'`, `key = "..."` and the HTML attribute `key="..."` in one shape.
# A trailing-suffix key (apiPlaceholder, providerCopy) matches through the
# leading `\w*`; a leading-substring key does not (`noteworthy:` needs the
# separator right after the name).
_COPY_VALUE = re.compile(
    r"\w*(?:" + "|".join(COPY_KEYS) + r")\s*[:=]\s*"
    r"(?P<quote>['\"`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)

# Module-level Python names whose `::` strings are real stored model ids --
# serialization DATA offered to the chooser, not copy telling anyone to type
# it. Keep this list short and named; a new entry is a design decision.
PY_SERIALIZATION_NAMES = frozenset({
    "_MODEL_SUGGESTIONS",   # ids the model chooser suggests, e.g. openai::gpt-5.6-terra
})


def _code_lines(source: str) -> list[tuple[int, str]]:
    """Source lines in code position: full-line ``//`` / ``*`` comment lines are
    skipped, nothing else is stripped.

    Same deliberate over-approximation as ``tests/test_web_dialogs_static.py``:
    no block-comment blanking, because a ``/*`` sweep cannot tell a comment
    opener from the same two characters inside a string. A prefix smuggled into
    a trailing comment is reported; a blind region would be a hole.
    """
    lines: list[tuple[int, str]] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        lines.append((lineno, line))
    return lines


def _text_offenses(path: pathlib.Path) -> list[str]:
    """Copy-carrying values in a JS/HTML source that contain the prefix."""
    lines = _code_lines(path.read_text(encoding="utf-8"))
    rel = path.relative_to(REPO_ROOT)
    offenses: list[str] = []
    for lineno, line in lines:
        for match in _COPY_VALUE.finditer(line):
            value = match.group("value")
            if PREFIX in value:
                offenses.append(f"{rel}:{lineno}: {value.strip()[:160]}")
    return offenses


def _owner_name(node: ast.stmt) -> str:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    return f"<line {node.lineno}>"


def _python_offenses(path: pathlib.Path) -> list[str]:
    """Prefix-bearing string constants in a Python contract module, attributed
    to their top-level owner so the allowlist can be read by name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(REPO_ROOT)
    offenses: list[str] = []
    for node in tree.body:
        owner = _owner_name(node)
        if owner in PY_SERIALIZATION_NAMES:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and PREFIX in sub.value:
                offenses.append(f"{rel}:{sub.lineno} ({owner}): {sub.value.strip()[:160]}")
    return offenses


def test_no_model_field_placeholder_or_help_instructs_a_provider_prefix() -> None:
    """No owner-facing placeholder or help string spells the stored prefix.

    docs/DESIGN.md §7: the stored spellings are "never required from the owner,
    never a field placeholder or help-text instruction".
    """
    offenses: list[str] = []
    for path in sorted(WEB_MODULES.glob("**/*.js")):
        offenses.extend(_text_offenses(path))
    offenses.extend(_text_offenses(INDEX_HTML))
    offenses.extend(_python_offenses(SETUP_CONTRACT))

    assert not offenses, (
        "owner-facing copy instructs a serialized provider prefix; the source is "
        "chosen, never spelled (docs/DESIGN.md §7):\n  " + "\n  ".join(offenses)
    )


def _import_specifiers(source: str, symbol: str) -> set[str]:
    """Module specifiers from which ``source`` imports ``symbol``.

    Handles the three real shapes in ``web/modules``: a named import, an
    aliased named import (``routeChoiceGroups as x``) and a namespace import
    whose object is then read as ``ns.routeChoiceGroups``.
    """
    specifiers: set[str] = set()
    named = re.compile(r"import\s*\{(?P<names>[^}]*)\}\s*from\s*['\"](?P<from>[^'\"]+)['\"]", re.DOTALL)
    for match in named.finditer(source):
        names = [part.split(" as ")[0].strip() for part in match.group("names").split(",")]
        if symbol in names:
            specifiers.add(match.group("from"))
    namespace = re.compile(r"import\s*\*\s*as\s+(?P<alias>\w+)\s*from\s*['\"](?P<from>[^'\"]+)['\"]")
    for match in namespace.finditer(source):
        if re.search(rf"\b{re.escape(match.group('alias'))}\s*\.\s*{re.escape(symbol)}\b", source):
            specifiers.add(match.group("from"))
    return specifiers


def _reexports(source: str, symbol: str) -> bool:
    """``export * from './route_editor_primitives.js'`` or an explicit
    ``export { symbol } from`` / re-declared wrapper of the same name."""
    if re.search(r"export\s*\*\s*from\s*['\"]\./route_editor_primitives\.js['\"]", source):
        return True
    for match in re.finditer(
        r"export\s*\{(?P<names>[^}]*)\}\s*from\s*['\"](?P<from>[^'\"]+)['\"]", source, re.DOTALL
    ):
        names = [part.split(" as ")[-1].strip() for part in match.group("names").split(",")]
        if symbol in names and "route_editor_primitives" in match.group("from"):
            return True
    # A same-named wrapper that delegates to the primitive counts: the source
    # list still has ONE owner.
    if re.search(rf"export\s+function\s+{re.escape(symbol)}\b", source):
        return bool(_import_specifiers(source, symbol) or re.search(
            rf"\w+\s*\.\s*{re.escape(symbol)}\b", source))
    return False


def test_model_assigning_editors_share_the_grouped_source_select() -> None:
    """Models roles and Available subagents (whose rows marked Reviewer are the
    reviewers) take their source groups from ONE primitive, directly or through a
    module that re-exports it.

    docs/DESIGN.md §7: every model-assigning surface offers "one grouped source
    select with the same groups in the same order". Private copies of the group
    list is exactly how that order drifts apart per surface.
    """
    symbol = "routeChoiceGroups"
    editors = ("model_roles.js", "subagents_settings.js")

    primitives_source = PRIMITIVES.read_text(encoding="utf-8")
    assert re.search(rf"export\s+(?:function|const)\s+{symbol}\b", primitives_source), (
        f"web/modules/route_editor_primitives.js must export {symbol}"
    )
    assert re.search(r"export\s+(?:function|const)\s+configuredApiProviders\b", primitives_source), (
        "web/modules/route_editor_primitives.js must export configuredApiProviders — the "
        "API keys group lists one entry per provider with a stored credential (docs/DESIGN.md §7)"
    )

    missing: list[str] = []
    for name in editors:
        path = WEB_MODULES / name
        source = path.read_text(encoding="utf-8")
        specifiers = _import_specifiers(source, symbol)
        if any("route_editor_primitives" in spec for spec in specifiers):
            continue
        # One hop: an editor may take the primitive through a sibling that
        # re-exports (or thinly wraps) it rather than importing it directly.
        if any(
            (WEB_MODULES / pathlib.Path(spec).name).is_file()
            and _reexports((WEB_MODULES / pathlib.Path(spec).name).read_text(encoding="utf-8"), symbol)
            for spec in specifiers
        ):
            continue
        missing.append(f"web/modules/{name}")

    assert not missing, (
        f"these model-assigning editors do not take {symbol} from "
        "web/modules/route_editor_primitives.js (directly or via a re-export): "
        + ", ".join(missing)
    )


def test_design_doc_states_the_source_is_chosen_rule() -> None:
    """The rule itself lives in the design document, with its three groups
    named in the order every surface renders them."""
    design = DESIGN_DOC.read_text(encoding="utf-8")
    rule_sentence = "A source is chosen, never spelled."

    # Scope the group check to the rule's own paragraph. "API keys" is ordinary
    # vocabulary elsewhere in the document, so a whole-file index would match
    # the Accounts sentence instead of the group list.
    paragraphs = [" ".join(block.split()) for block in design.split("\n\n")]
    rule = next((block for block in paragraphs if rule_sentence in block), "")
    assert rule, f"docs/DESIGN.md must state the rule: {rule_sentence!r}"

    # Reviewers are catalog rows now, so no surface offers a configured-subagent
    # reference group: the review lanes editor that did is retired.
    assert "configured subagents" not in rule
    groups = ("Subscriptions · models", "API keys", "Agents · sessions")
    positions = []
    for group in groups:
        assert group in rule, f"the rule paragraph must name the source group {group!r}"
        positions.append(rule.index(group))
    assert positions == sorted(positions), (
        "docs/DESIGN.md must name the source groups in their rendered order: "
        + ", ".join(groups)
    )
