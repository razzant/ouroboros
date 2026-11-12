"""Static guard: the typography scale holds on the surfaces that adopted it.

The owner's report was "too much small high-contrast white text". Four
independent causes produced it (docs/DESIGN.md):

1. ``class="muted"`` was written at ~50 call sites while the ONLY rule that
   matched it was the scoped ``.marketplace-card-title .muted`` — so muted text
   everywhere else inherited near-white ``--text-primary``;
2. ``.harness-chip`` / ``.reviewer-slot-meta`` declared a size and no colour,
   inheriting the same primary ink;
3. field labels were 12px UPPERCASE at a hand-written ``rgba(255,255,255,.68)``,
   repeated dozens of times per panel;
4. there was no scale at all — every size was a literal px, across three
   mutually inconsistent grey families.

This guard keeps that class closed on every first-party stylesheet. Until the
06.11.2026 convergence it read a delimited "migrated" subset of ``style.css``
(marker pairs in the file); the sheets now carry one system, so there is no
subset to delimit and nothing for a literal to hide behind.

Pattern follows ``tests/test_web_dialogs_static.py``: read the sources, assert
the structural fact, no browser needed.
"""

from __future__ import annotations

import pathlib
from html.parser import HTMLParser
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "web"

# The four sizes, the three line heights, the two named foregrounds, and the
# four status pairs. The scale is closed: a fifth size token is a design change
# that goes through docs/DESIGN.md, not a stylesheet edit.
TYPE_TOKENS = ("--type-meta", "--type-body", "--type-section", "--type-page")
LINE_TOKENS = ("--line-meta", "--line-body", "--line-title")
FOREGROUND_TOKENS = ("--text-meta", "--text-disabled")
STATUS_TOKENS = (
    "--status-ok-fg", "--status-ok-bg",
    "--status-warn-fg", "--status-warn-bg",
    "--status-error-fg", "--status-error-bg",
    "--status-neutral-fg", "--status-neutral-bg",
)

# Any numeric font-size below the 12px meta floor, in any unit the stylesheets
# actually write: px directly; rem against the 16px root; em against the same
# 16px equivalence (an em resolves against the parent, but a sub-0.75em value
# is sub-meta against every parent size in the four-token scale). Fractions
# (10.5px, 11.5px, 0.7em) count — the old integer-only pattern waved them by.
FONT_SIZE_VALUE = re.compile(r"font-size\s*:\s*(\d+(?:\.\d+)?)(px|rem|em)\b")
TINY_FONT_FLOOR_PX = 12.0


def _is_tiny_font(line: str) -> bool:
    m = FONT_SIZE_VALUE.search(line)
    if not m:
        return False
    value, unit = float(m.group(1)), m.group(2)
    if unit == "px":
        return value < TINY_FONT_FLOOR_PX
    return value * 16.0 < TINY_FONT_FLOOR_PX  # rem/em vs the 12px equivalent


UPPERCASE = re.compile(r"text-transform\s*:\s*uppercase")
# Innermost rule blocks only: the body pattern forbids braces, so an @media
# wrapper cannot match as a selector and the rules nested inside it are matched
# individually. No CSS parser needed for a structural ban.
RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _decommented(css: str) -> str:
    """Blank out ``/* ... */`` while preserving line numbers.

    Comments must go before anything is matched: these stylesheets carry long
    rationale comments that name the very selectors, sizes and rgba() literals
    the rules below them retired, and every one of those mentions would read as
    both a bogus violation and — worse — a bogus selector attached to the next
    real rule."""
    return COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), css)


def _sheet_sources() -> dict[str, str]:
    """Every first-party stylesheet, comments blanked (vendored sheets are not ours)."""
    return {
        str(path.relative_to(REPO_ROOT)): _decommented(path.read_text(encoding="utf-8"))
        for path in sorted(WEB.glob("*.css"))
    }


# ---------------------------------------------------------------------------
# The scale itself
# ---------------------------------------------------------------------------


def test_type_scale_tokens_are_declared_once_in_the_root_block() -> None:
    css = _decommented(_read("web/ui.css"))
    root = css[: css.index("\n}")]
    assert root.lstrip().startswith(":root"), "expected :root to open web/ui.css"
    for token in TYPE_TOKENS + LINE_TOKENS + FOREGROUND_TOKENS + STATUS_TOKENS:
        assert f"{token}:" in root, f"{token} missing from web/ui.css :root"
    # Exactly four sizes: a fifth --type-* token means the scale grew without a
    # docs/DESIGN.md decision.
    declared = set(re.findall(r"(--type-[a-z]+)\s*:", root))
    assert declared == set(TYPE_TOKENS), (
        "the type scale is closed at four sizes (docs/DESIGN.md 'Type scale'); "
        f"found {sorted(declared)}"
    )


def _root_declarations(rel: str) -> dict[str, str]:
    """The ``:root`` block of a stylesheet as ``{token: value}``.

    The block is the first rule of both files; reading only it keeps a
    component-local ``--foo`` override out of the comparison."""
    css = _decommented(_read(rel))
    root = css[: css.index("\n}")]
    assert root.lstrip().startswith(":root"), f"expected :root to open {rel}"
    return {
        name: " ".join(value.split())
        for name, value in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", root)
    }


class _Stylesheets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        href = values.get("href") or ""
        # The wizard links the same builtin files under the core-owned /onboarding/static prefix.
        prefix = next((p for p in ("/static/", "/onboarding/static/") if href.startswith(p)), "")
        if tag == "link" and values.get("rel") == "stylesheet" and prefix:
            self.paths.append("web/" + href.removeprefix(prefix).split("?", 1)[0])


def _document_stylesheets(document: str) -> list[str]:
    parser = _Stylesheets()
    parser.feed(_read(document))
    assert parser.paths, f"no local stylesheets in {document}"
    return parser.paths


def test_no_tiny_raw_font_sizes_in_any_sheet() -> None:
    violations: list[str] = []
    for label, source in _sheet_sources().items():
        for lineno, line in enumerate(source.splitlines(), 1):
            if _is_tiny_font(line):
                violations.append(f"{label}:{lineno}: {line.strip()}")
    assert not violations, (
        "Raw sub-12px text is retired: below 12px this dark theme forces a "
        "choice between illegible and glaring, and glaring is what the owner "
        "reported. Use var(--type-meta) (docs/DESIGN.md 'Type scale').\n"
        + "\n".join(violations)
    )


def test_no_uppercase_label_pattern_in_any_sheet() -> None:
    violations: list[str] = []
    for label, source in _sheet_sources().items():
        for selector, body in RULE.findall(source):
            if "label" not in selector.lower():
                continue
            if UPPERCASE.search(body):
                violations.append(f"{label}: {' '.join(selector.split())}")
    assert not violations, (
        "The 12px UPPERCASE label pattern is retired "
        "(docs/DESIGN.md 'Hierarchy rule'): all-caps at a small size costs "
        "legibility, widens every label, and a panel that repeats it dozens of "
        "times makes the labels out-shout the values they describe. Author the "
        "string in sentence case instead of manufacturing caps in CSS.\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Token hygiene: declared <-> used, in both directions
# ---------------------------------------------------------------------------

# Derive consumers from the real documents; a declaration in the SPA cannot
# accidentally satisfy an unresolved wizard variable (or the reverse).
ROOT_CONSUMERS = tuple(dict.fromkeys(
    _document_stylesheets("web/index.html") + _document_stylesheets("web/onboarding_template.html")
))

VAR_REFERENCE = re.compile(r"var\(\s*(--[a-z0-9-]+)")
DECLARATION = re.compile(r"^\s*(--[a-z0-9-]+)\s*:", re.MULTILINE)


def _js_sources() -> str:
    """Every first-party web script, concatenated: the two page scripts (app.js
    and theme.js, beside the vendored *.min.js bundles) and the modules.

    JS participates in the variable contract from both ends: it writes measured
    values with ``setProperty('--chat-input-reserve', …)`` and it reads themed
    ones with ``getComputedStyle(...).getPropertyValue('--diagram-bg')``. A
    token at either end is live even though no CSS rule mentions it: app.js's
    handles write the dragged box's ``--drag-width``."""
    scripts = [WEB / "app.js", WEB / "theme.js", *sorted((WEB / "modules").rglob("*.js"))]
    return "".join(path.read_text(encoding="utf-8") for path in scripts)


def test_every_css_variable_is_declared_somewhere() -> None:
    """A `var(--typo)` is silent: the declaration simply does not apply and the
    property keeps whatever it inherited. This codebase had six of them —
    `--surface-1`, `--surface-2`, `--danger`, `--warning`, `--mono` and
    `--text-link` — each carrying a hardcoded fallback that was the value
    actually rendering, and three of those fallbacks (`#e5534b`, `#b58900`,
    `#16181d`) were colours from no palette in this product."""
    # Only a JS WRITE declares a variable: `setProperty('--x', …)` or an inline `--x: …` in a
    # style string. A read (`getPropertyValue('--x')`) consumes one, so it must not vouch for it.
    js_written = set(re.findall(r"""setProperty\(\s*['"`](--[\w-]+)|(?<![\w-])(--[a-z][\w-]*)\s*:""", _js_sources()))
    js_written = {name for pair in js_written for name in pair if name}
    dangling: list[str] = []
    for document in ("web/index.html", "web/onboarding_template.html"):
        sheets = _document_stylesheets(document)
        declared = set().union(*(set(DECLARATION.findall(_decommented(_read(rel)))) for rel in sheets))
        for rel in sheets:
            for lineno, line in enumerate(_decommented(_read(rel)).splitlines(), 1):
                for name in VAR_REFERENCE.findall(line):
                    if name not in declared and name not in js_written:
                        dangling.append(f"{document}: {rel}:{lineno}: var({name})")
    assert not dangling, (
        "these variables are never declared, in CSS or by a JS setProperty, so "
        "every rule naming one silently renders its fallback (or nothing). Name "
        "an existing token instead of declaring a new one — the point of the "
        "palette is that it is small (docs/DESIGN.md).\n" + "\n".join(dangling)
    )


def test_every_root_token_has_a_reader() -> None:
    """The other direction, and the one that actually bites. `--tone-ok`,
    `--tone-warn`, `--tone-danger`, `--accent-task/system/user/project` and
    `--ui-tone-*` were named in docs/DESIGN.md as the shared vocabulary and
    referenced by NOTHING — so seven surfaces each invented their own literal
    for the same four states while the file said they were unified. A token
    with no reader is not a reserve; it is a claim the code does not make.

    There is no allowlist. If a token is worth keeping, something uses it."""
    root = _root_declarations("web/ui.css")
    used = set()
    for rel in ROOT_CONSUMERS:
        used |= set(VAR_REFERENCE.findall(_decommented(_read(rel))))
    js = _js_sources()

    orphans = sorted(name for name in root if name not in used and name not in js)
    assert not orphans, (
        "these :root tokens in web/ui.css have no reader in the stylesheets "
        "or the web modules. Either use them or delete them: a documented token "
        "that resolves nowhere is why surfaces reach for literals "
        "(docs/DESIGN.md 'Status and chips').\n"
        + "\n".join(f"  {name}" for name in orphans)
    )


# ---------------------------------------------------------------------------
# Focus canon: one ring vocabulary across the whole app (docs/DESIGN.md "Focus")
# ---------------------------------------------------------------------------

FOCUS_FILES = ROOT_CONSUMERS
FOCUS_TOKENS = ("var(--focus-accent-border)", "var(--focus-accent-ring)")


def test_every_focus_visible_selector_gets_the_canonical_ring() -> None:
    """Keyboard focus has ONE appearance (docs/DESIGN.md 'Focus'): the
    `--focus-accent-border` outline (or the field idiom's
    `--focus-accent-ring` box-shadow). A `:focus-visible` rule painted in some
    other colour is a second focus vocabulary; a `:focus-visible` selector with
    no ring anywhere is hover paint masquerading as focus.

    The unit is the SELECTOR, not the block: the sanctioned hybrid pattern
    keeps shared hover/focus paint in one rule (no ring) and puts the ring in a
    dedicated `:focus-visible` rule beside it, so a selector passes when ANY of
    its blocks names a focus token. The exception allowlist is empty — a
    legitimate exception must be added here with its justification."""
    per_selector: dict[str, list[bool]] = {}
    for rel in FOCUS_FILES:
        source = _decommented(_read(rel))
        for selector_list, body in RULE.findall(source):
            has_token = any(token in body for token in FOCUS_TOKENS)
            for selector in selector_list.split(","):
                selector = " ".join(selector.split())
                if ":focus-visible" not in selector:
                    continue
                per_selector.setdefault(f"{rel}: {selector}", []).append(has_token)
    assert per_selector, "no :focus-visible rules found; is the parse broken?"
    unringed = sorted(
        selector for selector, hits in per_selector.items() if not any(hits)
    )
    assert not unringed, (
        "these :focus-visible selectors never name var(--focus-accent-border) "
        "or var(--focus-accent-ring) in any of their rule blocks, so keyboard "
        "focus there is either invisible or a second colour vocabulary "
        "(docs/DESIGN.md 'Focus'):\n" + "\n".join(f"  {s}" for s in unringed)
    )


# ---------------------------------------------------------------------------
# Horizontal overflow: the red scrollbar on Settings -> Advanced
# ---------------------------------------------------------------------------


def test_every_mask_keeps_its_webkit_companion() -> None:
    """A fade is a mask (`.scroll-fade-y`, the sideways table fade): each rule that
    sets `mask-image` also sets `-webkit-mask-image`, as the file's first fade does,
    so a WebKit view that reads only the prefixed property still fades and unfades."""
    css = _decommented(_read("web/style.css"))
    unpaired = [selector.strip() for selector, body in RULE.findall(css)
                if re.search(r"(?<![-\w])mask-image\s*:", body) and "-webkit-mask-image" not in body]
    assert not unpaired, f"mask-image without its -webkit- companion: {unpaired}"
