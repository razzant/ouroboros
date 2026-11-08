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
