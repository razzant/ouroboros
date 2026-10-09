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

This guard keeps that class closed on the **migrated** surfaces only. It is
deliberately NOT a sweep of the historical stylesheet: ``web/style.css`` still
carries unmigrated skills/marketplace/widget/log rules whose literals are a
later pass, and a guard that fails on all of them would be turned off. The
migrated slices of ``style.css`` are delimited in the file itself by
``design-system:migrated-begin`` / ``design-system:migrated-end`` marker PAIRS
— several, because migrated surfaces (harness accounts, the chat transcript,
the chat page chrome, structured chat delivery) are not contiguous in the file
and moving hundreds of unrelated lines to join them would destroy blame. So
migrating a new surface means moving a marker or adding a pair (or a file
below) in the same commit that migrates it.

Pattern follows ``tests/test_web_dialogs_static.py``: read the sources, assert
the structural fact, no browser needed.
"""

from __future__ import annotations

import pathlib
from html.parser import HTMLParser
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "web"

BEGIN_MARKER = "design-system:migrated-begin"
END_MARKER = "design-system:migrated-end"

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


def _style_marker_spans() -> list[tuple[int, int]]:
    """Every ``(begin, end)`` marker pair of style.css, structurally validated.

    N pairs are allowed (migrated surfaces are not contiguous in the file), but
    the pairing itself must stay honest: as many ends as begins, strictly
    alternating begin/end — which rules out nested and overlapping regions and
    a stray marker mention that would silently truncate a guarded slice."""
    css = _read("web/style.css")
    events = sorted(
        [(m.start(), "begin") for m in re.finditer(re.escape(BEGIN_MARKER), css)]
        + [(m.start(), "end") for m in re.finditer(re.escape(END_MARKER), css)]
    )
    assert events, "no design-system markers in web/style.css"
    kinds = [kind for _, kind in events]
    assert kinds == ["begin", "end"] * (len(events) // 2), (
        "design-system markers must be strictly alternating begin/end pairs — "
        "nesting, overlap, or an unpaired mention silently reshapes the "
        f"guarded regions; got sequence {kinds}"
    )
    return [
        (events[i][0], events[i + 1][0]) for i in range(0, len(events), 2)
    ]


def _migrated_style_region(raw: bool = False) -> str:
    """The concatenated marked (migrated) slices of style.css."""
    css = _read("web/style.css")
    slices = [css[start:end] for start, end in _style_marker_spans()]
    if not raw:
        slices = [_decommented(s) for s in slices]
    return "\n".join(slices)


def _migrated_sources() -> dict[str, str]:
    return {
        "web/ui.css": _decommented(_read("web/ui.css")),
        "web/settings.css": _decommented(_read("web/settings.css")),
        "web/onboarding.css": _decommented(_read("web/onboarding.css")),
        "web/model_roles.css": _decommented(_read("web/model_roles.css")),
        "web/model_wait.css": _decommented(_read("web/model_wait.css")),
        "web/style.css (migrated regions)": _migrated_style_region(),
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


def test_chat_reading_ladder_is_relative_and_read_only_by_chat_headings() -> None:
    """The one exception to the closed scale (docs/DESIGN.md §1): Markdown
    headings in a full rich chat answer step up from the bubble's own reading
    text. The steps are relative, so they cannot become a fifth UI size, and
    only the rich answer's heading rules may read them — shared, in the same
    rules, with a delivered document in the reader (DESIGN "Document reading"):
    compact Markdown in a bubble (a Skill Review report) keeps its --type-body
    labels."""
    root = _root_declarations("web/ui.css")
    ladder = {"--md-heading-major": "1.25em", "--md-heading-minor": "1.125em"}
    assert {name: root.get(name) for name in ladder} == ladder
    css = _decommented(_read("web/style.css"))
    readers = [
        " ".join(selector.split())
        for selector, body in RULE.findall(css)
        if any(f"var({name})" in body for name in ladder)
    ]
    assert readers == [
        ".chat-bubble .message:where(.ui-rich-content) :is(.md-h1, .md-h2), .document-reader-markdown :is(.md-h1, .md-h2)",
        ".chat-bubble .message:where(.ui-rich-content) .md-h3, .document-reader-markdown .md-h3",
    ], readers


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
        if tag == "link" and values.get("rel") == "stylesheet" and href.startswith("/static/"):
            self.paths.append("web/" + href.removeprefix("/static/").split("?", 1)[0])


def _document_stylesheets(document: str) -> list[str]:
    parser = _Stylesheets()
    parser.feed(_read(document))
    assert parser.paths, f"no local stylesheets in {document}"
    return parser.paths


def test_both_documents_load_one_real_shared_palette_before_page_styles() -> None:
    """No copied palette: both hosts must load the same nonempty source.

    Required roles prevent an empty source from passing. Page declarations
    cannot silently shadow a shared role and make onboarding a second theme.
    Actual effective palette and controls are also exercised in the browser.
    """
    shared = _root_declarations("web/ui.css")
    required = TYPE_TOKENS + LINE_TOKENS + FOREGROUND_TOKENS + STATUS_TOKENS + ("--accent",)
    for token in required:
        assert shared.get(token), f"{token} missing from shared palette"
    assert len(shared) > 20, "shared source must contain the actual palette"
    for document in ("web/index.html", "web/onboarding_template.html"):
        sheets = _document_stylesheets(document)
        assert sheets[0] == "web/ui.css" and sheets.count("web/ui.css") == 1
        for sheet in sheets[1:]:
            declared = set(DECLARATION.findall(_decommented(_read(sheet))))
            assert not (declared & shared.keys()), f"{sheet} shadows shared roles: {declared & shared.keys()}"


def test_no_tiny_raw_font_sizes_on_migrated_surfaces() -> None:
    violations: list[str] = []
    for label, source in _migrated_sources().items():
        for lineno, line in enumerate(source.splitlines(), 1):
            if _is_tiny_font(line):
                violations.append(f"{label}:{lineno}: {line.strip()}")
    assert not violations, (
        "Raw sub-12px text is retired on migrated surfaces: below 12px this "
        "dark theme forces a choice between illegible and glaring, and glaring "
        "is what the owner reported. Use var(--type-meta) (docs/DESIGN.md "
        "'Type scale').\n" + "\n".join(violations)
    )


def test_no_uppercase_label_pattern_on_migrated_surfaces() -> None:
    violations: list[str] = []
    for label, source in _migrated_sources().items():
        for selector, body in RULE.findall(source):
            if "label" not in selector.lower():
                continue
            if UPPERCASE.search(body):
                violations.append(f"{label}: {' '.join(selector.split())}")
    assert not violations, (
        "The 12px UPPERCASE label pattern is retired on migrated surfaces "
        "(docs/DESIGN.md 'Hierarchy rule'): all-caps at a small size costs "
        "legibility, widens every label, and a panel that repeats it dozens of "
        "times makes the labels out-shout the values they describe. Author the "
        "string in sentence case instead of manufacturing caps in CSS.\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# The four root causes, pinned individually
# ---------------------------------------------------------------------------


def test_muted_is_a_global_colour_only_utility() -> None:
    """Root cause #1. `.muted` must resolve globally, and must NOT set a size:
    its call sites are sized by their contexts, so a font-size here would
    silently resize all of them."""
    css = _decommented(_read("web/ui.css"))
    bodies = [body for selector, body in RULE.findall(css) if selector.strip() == ".muted"]
    assert bodies, "no global `.muted` rule in web/ui.css"
    declarations = [
        part.split(":", 1)[0].strip()
        for body in bodies
        for part in body.split(";")
        if part.strip()
    ]
    assert declarations == ["color"], (
        "`.muted` is a colour-only utility (docs/DESIGN.md '.muted'); it "
        f"declares {declarations}"
    )
    assert any("var(--text-meta)" in body for body in bodies)


def test_chips_and_meta_lines_declare_their_own_foreground() -> None:
    """Root cause #2. A rule that declares a size and no colour inherits
    near-white --text-primary — invisible in the CSS, loudest on screen."""
    region = _migrated_style_region() + _decommented(_read("web/settings.css"))
    for selector in (".harness-chip", ".available-subagent-review", ".harness-account-main strong"):
        bodies = [
            body for sel, body in RULE.findall(region) if sel.strip() == selector
        ]
        assert bodies, f"{selector} missing from the migrated region of web/style.css or web/settings.css"
        assert any("color:" in body for body in bodies), (
            f"{selector} declares no colour, so it inherits --text-primary "
            "(docs/DESIGN.md 'Status and chips')"
        )


def test_settings_field_labels_use_the_named_meta_foreground() -> None:
    """Root cause #3. The hand-written rgba(255,255,255,0.68) is now the named
    --text-meta, in both stylesheets that carried a copy of it."""
    for rel in ("web/settings.css", "web/onboarding.css"):
        css = _read(rel)
        # Only the shared source declares the value. Page rules name its role.
        assert "rgba(255, 255, 255, 0.68)" not in css
        # The wizard's former private grey family remains retired.
        assert "rgba(237, 242, 247, 0.68)" not in css
        assert "var(--text-meta)" in css, f"{rel} never names --text-meta"


def test_migrated_region_markers_do_not_swallow_unmigrated_surfaces() -> None:
    """Root cause #4's guard rail: the scoping must stay honest in BOTH
    directions. The regions have to actually contain the migrated rules —
    including the chat surface and its `.chat-live-executor-chip`, which
    migrated with the chat typography pass — and they must not creep over
    neighbours (skills, marketplace, logs, evolution) that still carry their
    historical literals. Marker-pair structure itself (as many ends as begins,
    strictly alternating) is asserted by ``_style_marker_spans`` on every call
    that reads a region."""
    region = _migrated_style_region(raw=True)
    assert ".harness-account-row" in region
    # The Dashboard -> Updates tab migrated on 2026-08-31; its rules must stay
    # inside the guarded region so a later edit cannot drift them out of it.
    assert ".updates-status" in region
    assert ".updates-restore-row" in region
    # Chat migrated on 2026-09-01 (frontend sprint, Q1=B): page chrome,
    # transcript/bubbles/live cards/composer, and the structured-delivery +
    # quiz-card slice, executor chip now included.
    assert ".chat-page-header" in region
    assert ".chat-bubble.progress" in region
    assert ".chat-live-title" in region
    assert ".chat-live-executor-chip" in region
    assert ".chat-quiz-card" in region
    # Unmigrated neighbours stay out until their own pass. (`.log-entry` and
    # `.evo-runtime-pill` are NOT in this list: the shared status-tone rules
    # inside the chat region legitimately name them as co-selectors.)
    for selector in (".skills-card", ".marketplace-card", ".widgets-card", ".evo-runtime-card"):
        assert selector not in region, (
            f"{selector} is an unmigrated surface; a marker crept over it"
        )
    # NOTE: deliberately NOT asserting that debt still exists out there. A guard
    # that fails when someone independently improves an unmigrated surface would
    # punish exactly the work it wants. The marker pairing (asserted in
    # ``_style_marker_spans``) is what proves the regions are really scoped.


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
    """Every web module, concatenated.

    JS participates in the variable contract from both ends: it writes measured
    values with ``setProperty('--chat-input-reserve', …)`` and it reads themed
    ones with ``getComputedStyle(...).getPropertyValue('--diagram-bg')``. A
    token at either end is live even though no CSS rule mentions it."""
    return "".join(
        path.read_text(encoding="utf-8") for path in sorted((WEB / "modules").rglob("*.js"))
    )


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


def test_quiz_question_reads_as_text_with_its_own_emphasis() -> None:
    """Pins the owner-visible defect "the whole question is one bold block": a
    real question is a marked title plus several lines, so the question itself
    is regular weight and only what it marks is semibold (DESIGN "Quiz card")."""
    css = _decommented(_read("web/style.css"))

    def weights(wanted: str) -> list[str]:
        return [
            part.split(":", 1)[1].strip()
            for selector, body in RULE.findall(css)
            if selector.strip() == wanted
            for part in body.split(";")
            if part.strip() and part.split(":", 1)[0].strip() == "font-weight"
        ]

    assert weights(".chat-quiz-question") == ["400"]
    assert weights(".chat-quiz-question strong") == ["600"]


def test_select_control_clips_its_value() -> None:
    """Pins the owner-visible defect "Settings -> Advanced paints a horizontal
    scrollbar in the desktop app": WebKit computes `overflow: visible` on a
    native select (Blink's UA sheet clips it), so a long selected option leaks
    out of the control and into the page scroller."""
    css = _decommented(_read("web/ui.css"))
    bodies = [
        body for selector, body in RULE.findall(css)
        if selector.strip() == "select.ui-control"
    ]
    assert bodies, "no `select.ui-control` rule in web/ui.css"
    values = [
        part.split(":", 1)[1].strip().lower()
        for body in bodies
        for part in body.split(";")
        if part.strip() and part.split(":", 1)[0].strip() == "overflow"
    ]
    assert values, (
        "`select.ui-control` declares no `overflow`, so WebKit lets a long "
        "selected option paint past the control's own box and widen the "
        "settings scroller"
    )
    assert all(value != "visible" for value in values), (
        f"`select.ui-control` re-opens the clip: overflow {values}"
    )


def test_webkit_scrollbar_recipe_covers_both_axes() -> None:
    """Pins the owner-visible defect "the scrollbar is thick and red": the
    global `::-webkit-scrollbar` recipe sized only `width`, which is the
    VERTICAL bar, so any horizontal bar kept the 16-17px UA thickness while
    still wearing the accent thumb."""
    css = _decommented(_read("web/style.css"))
    bodies = [
        body for selector, body in RULE.findall(css)
        if selector.strip() == "::-webkit-scrollbar"
    ]
    assert bodies, "no global `::-webkit-scrollbar` rule in web/style.css"
    declarations: dict[str, str] = {}
    for body in bodies:
        for part in body.split(";"):
            if ":" not in part:
                continue
            name, _, value = part.partition(":")
            declarations[name.strip().lower()] = value.strip().lower()
    assert "width" in declarations and "height" in declarations, (
        "the global scrollbar recipe must size both axes; it declares "
        f"{sorted(declarations)}"
    )
    assert declarations["width"] == declarations["height"], (
        "the horizontal bar must be as thin as the vertical one: "
        f"width {declarations['width']} vs height {declarations['height']}"
    )


def test_every_mask_keeps_its_webkit_companion() -> None:
    """A fade is a mask (`.scroll-fade-y`, the sideways table fade): each rule that
    sets `mask-image` also sets `-webkit-mask-image`, as the file's first fade does,
    so a WebKit view that reads only the prefixed property still fades and unfades."""
    css = _decommented(_read("web/style.css"))
    unpaired = [selector.strip() for selector, body in RULE.findall(css)
                if re.search(r"(?<![-\w])mask-image\s*:", body) and "-webkit-mask-image" not in body]
    assert not unpaired, f"mask-image without its -webkit- companion: {unpaired}"

def test_chat_transcript_reserves_composer_space_as_one_flex_spacer():
    """End space is a flex item; keyboard flow removes it and its extra gap."""
    css = _decommented(_read("web/style.css"))
    rules = [(selector.strip(), body) for selector, body in RULE.findall(css)]
    assert not re.search(r"padding-bottom:\s*(?:calc\()?var\(--chat-input-reserve", css)
    spacers = [(selector, body) for selector, body in rules if "chat-messages::after" in selector]
    bases = [body for _, body in spacers if "flex:" in body]
    assert len(bases) == 1
    assert "content: '';" in bases[0]
    assert "flex: 0 0 calc(var(--chat-input-reserve, 108px) - 8px);" in bases[0]
    mobile = [body for _, body in spacers if "env(safe-area-inset-bottom" in body]
    assert len(mobile) == 1
    assert "- 8px" in mobile[0]
    panel = [body for selector, body in spacers if selector == ".chat-instance-panel .chat-messages::after"]
    assert len(panel) == 1
    assert "flex-basis: calc(var(--chat-input-reserve, 108px) - 8px);" in panel[0]
    keyboard = [body for selector, body in spacers if "body.keyboard-open" in selector]
    assert len(keyboard) == 1
    assert "content: none;" in keyboard[0]
