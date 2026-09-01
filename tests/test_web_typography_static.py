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
        "web/settings.css": _decommented(_read("web/settings.css")),
        "web/onboarding.css": _decommented(_read("web/onboarding.css")),
        "web/style.css (migrated regions)": _migrated_style_region(),
    }


# ---------------------------------------------------------------------------
# The scale itself
# ---------------------------------------------------------------------------


def test_type_scale_tokens_are_declared_once_in_the_root_block() -> None:
    css = _decommented(_read("web/style.css"))
    root = css[: css.index("\n}")]
    assert root.lstrip().startswith(":root"), "expected :root to open web/style.css"
    for token in TYPE_TOKENS + LINE_TOKENS + FOREGROUND_TOKENS + STATUS_TOKENS:
        assert f"{token}:" in root, f"{token} missing from web/style.css :root"
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


def test_onboarding_mirrors_the_scale_by_value() -> None:
    """onboarding.css is INLINED into a standalone first-run page and cannot
    import style.css, so it must carry the same tokens itself or every wizard
    rule that names one silently resolves to nothing.

    Carrying them is not enough: a mirror that drifts is worse than no mirror,
    because both sides look tokenised while the wizard quietly renders a
    different product. The wizard shipped its own brand red (``#e85d6f``
    against the app's ``#c93545``), its own green, and its own foreground greys
    — so the first screen a new owner saw was the one screen that did not match
    the app. Every name declared in BOTH files must therefore resolve to the
    SAME value; a name that exists on only one side stays free."""
    style = _root_declarations("web/style.css")
    onboarding = _root_declarations("web/onboarding.css")

    for token in TYPE_TOKENS + LINE_TOKENS + FOREGROUND_TOKENS + STATUS_TOKENS:
        assert token in onboarding, f"{token} missing from web/onboarding.css :root"

    shared = sorted(set(style) & set(onboarding))
    # Guard against a vacuous pass if either :root is reshaped: the mirror is
    # the point, so it has to actually overlap.
    assert len(shared) > 20, f"only {len(shared)} shared tokens; is the mirror still real?"
    drifted = {
        token: (style[token], onboarding[token])
        for token in shared
        if style[token] != onboarding[token]
    }
    assert not drifted, (
        "web/onboarding.css mirrors web/style.css BY VALUE (docs/DESIGN.md "
        "header): it is inlined standalone and cannot import the app "
        "stylesheet, so a shared token name that resolves differently is a "
        "second product. Fix the value, or rename the wizard-local token so it "
        "stops claiming to be the shared one.\n"
        + "\n".join(
            f"  {token}: style.css={s!r} onboarding.css={o!r}"
            for token, (s, o) in sorted(drifted.items())
        )
    )


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
    css = _decommented(_read("web/style.css"))
    bodies = [body for selector, body in RULE.findall(css) if selector.strip() == ".muted"]
    assert bodies, "no global `.muted` rule in web/style.css"
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
    region = _migrated_style_region()
    for selector in (".harness-chip", ".reviewer-slot-meta", ".harness-account-main strong"):
        bodies = [
            body for sel, body in RULE.findall(region) if sel.strip() == selector
        ]
        assert bodies, f"{selector} missing from the migrated region of web/style.css"
        assert any("color:" in body for body in bodies), (
            f"{selector} declares no colour, so it inherits --text-primary "
            "(docs/DESIGN.md 'Status and chips')"
        )


def test_settings_field_labels_use_the_named_meta_foreground() -> None:
    """Root cause #3. The hand-written rgba(255,255,255,0.68) is now the named
    --text-meta, in both stylesheets that carried a copy of it."""
    for rel in ("web/settings.css", "web/onboarding.css"):
        css = _read(rel)
        # The token's own declaration is the one place the literal may appear:
        # onboarding.css mirrors style.css by value and has no import.
        body = css.replace("--text-meta: rgba(255, 255, 255, 0.68)", "")
        assert "rgba(255, 255, 255, 0.68)" not in body
        # The wizard's former private grey family, retired when it started
        # mirroring style.css by value.
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
    assert ".reviewer-slots-heading" in region
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

# Files that resolve their variables against web/style.css `:root`. They are
# loaded together by web/index.html, so a token declared in one and named in
# the other is correct. web/onboarding.css is NOT here: it is inlined into a
# standalone page with its own `:root` and is covered by the mirror test above.
ROOT_CONSUMERS = ("web/style.css", "web/settings.css")

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
    declared = set()
    for rel in ROOT_CONSUMERS:
        declared |= set(DECLARATION.findall(_decommented(_read(rel))))
    js = _js_sources()

    dangling: list[str] = []
    for rel in ROOT_CONSUMERS:
        source = _decommented(_read(rel))
        for lineno, line in enumerate(source.splitlines(), 1):
            for name in VAR_REFERENCE.findall(line):
                if name in declared or name in js:
                    continue
                dangling.append(f"{rel}:{lineno}: var({name})")
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
    root = _root_declarations("web/style.css")
    used = set()
    for rel in ROOT_CONSUMERS:
        used |= set(VAR_REFERENCE.findall(_decommented(_read(rel))))
    js = _js_sources()

    orphans = sorted(name for name in root if name not in used and name not in js)
    assert not orphans, (
        "these :root tokens in web/style.css have no reader in the stylesheets "
        "or the web modules. Either use them or delete them: a documented token "
        "that resolves nowhere is why surfaces reach for literals "
        "(docs/DESIGN.md 'Status and chips').\n"
        + "\n".join(f"  {name}" for name in orphans)
    )


# ---------------------------------------------------------------------------
# Focus canon: one ring vocabulary across the whole app (docs/DESIGN.md "Focus")
# ---------------------------------------------------------------------------

FOCUS_FILES = ("web/style.css", "web/settings.css", "web/onboarding.css")
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
