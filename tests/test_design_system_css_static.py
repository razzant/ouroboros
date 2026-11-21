"""Design-system debt ratchets for first-party sheets (P10/P11)."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def first_party_sheets():
    return sorted(path for path in WEB.rglob("*.css") if "vendor" not in path.relative_to(WEB).parts)


def without_comments(text):
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def test_settings_minimum_spans_the_content_tracks_without_the_result():
    css = without_comments((WEB / 'settings.css').read_text(encoding='utf-8'))

    def rule(selector):
        found = re.search(re.escape(selector) + r'\s*\{([^{}]*)\}', css)
        assert found, selector
        return found.group(1)

    minimum = rule('.settings-row::before')
    assert 'content: "";' in minimum
    assert 'grid-area: 1 / 1 / 3 / -1;' in minimum
    assert 'min-height: calc(52px - 2 * (var(--space-3) + 1px));' in minimum
    assert 'min-height: 52px;' in rule('.ui-card > .settings-row:only-child::before')
    assert 'grid-row: 1 / 4;' in rule('.settings-row[data-wide]::before')
    assert 'grid-row: 1 / 4;' in rule('.settings-row[data-kind="choices"]::before')
    assert 'content: none;' in rule('.dashboard-settings[data-solo] > .settings-row::before')
    assert 'grid-column: 1 / -1;' in rule('.settings-row > .settings-row-result')


def test_header_presence_comes_from_writers_and_telegram_owns_its_back():
    css = without_comments((WEB / 'style.css').read_text(encoding='utf8'))
    settings = without_comments((WEB / 'settings.css').read_text(encoding='utf8'))
    assert 'html[data-telegram-back="back"] .app-page-return { display: none; }' in css
    assert '.app-page-header[data-return]' in css and '[data-search-in-home]' in settings
    assert ':has([data-settings-search-slot]:empty)' not in settings
    assert 'data-page-back' not in css and '.dashboard-back' not in css
    scope = 'html[data-telegram-fullscreen="active"] #content > .page:not(#page-chat) > .app-page-header'
    leading = re.search(re.escape(scope) + r' > \.app-page-leading \{([^}]+)\}', css)[1]
    assert 'display: flex;' in leading, 'sections keep a visible way to the drawer in Telegram'
    assert '--telegram-header-start: 108px;' in css and '--telegram-header-end: 100px;' in css
    assert scope + ' {' in css, 'Telegram section placement excludes room bars'
    slot = re.search(r'\.app-page-return \{([^}]+)\}', css)[1]
    assert 'flex: none;' in slot, 'the round Back never yields to the page controls'
    assert 'width' not in slot and '.shell-back-name' not in css, 'no word beside the chevron, so nothing to truncate'


# No exception: status/provenance/stat labels keep the renderer's sentence case.
UPPERCASE_ALLOWLIST = {}


def test_first_party_styles_do_not_uppercase_labels():
    violations = []
    for path in first_party_sheets():
        css = without_comments(path.read_text(encoding="utf-8"))
        if re.search(r"text-transform\s*:\s*uppercase\b", css, re.I):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, f"Sentence case is a design-system rule: {violations}"
    assert not UPPERCASE_ALLOWLIST


# Exact, shrink-only inventory at the L3 base. Counts matter: a new query at an
# existing width is new debt too. When removing a query, lower its count here in
# the same commit. L1 removes the shell entries during integration; content
# queries should become container queries, never another viewport breakpoint.
MAX_WIDTH_DEBT = {
    'web/model_roles.css': {'720px': 1},  # existing model editor stack
    'web/onboarding.css': {'900px': 1, '760px': 1, '720px': 1},  # wizard
    'web/room_work.css': {'900px': 1},  # existing Shade height recipe (L1)
    'web/settings.css': {'980px': 2, '760px': 1, '640px': 1, '600px': 1},  # existing form/MCP layout debt
    'web/style.css': {'980px': 1, '640px': 2, '620px': 2},
    'web/ui.css': {'640px': 1},  # native field font-size/zoom recipe
}


def media_widths(css):
    from collections import Counter

    return Counter(
        re.sub(r'\s+', '', value).lower()
        for query in re.findall(r'@media\b([^{}]+)\{', without_comments(css), re.I)
        for value in re.findall(r'\(\s*max-width\s*:\s*([^)]+)\)', query, re.I)
    )


def test_viewport_width_queries_match_the_shrinking_debt_inventory():
    actual = {path.relative_to(ROOT).as_posix(): dict(counts) for path in first_party_sheets()
              if (counts := media_widths(path.read_text(encoding='utf-8')))}
    assert actual == MAX_WIDTH_DEBT, (
        'New raw viewport breakpoints are forbidden; components follow their container. '
        'When paying down existing queries, shrink MAX_WIDTH_DEBT in the same commit. '
        f'\nActual: {actual}\nAllowed debt: {MAX_WIDTH_DEBT}'
    )


def test_width_inventory_sees_multiline_compound_queries_and_repeated_debt():
    assert media_widths('''
        /* @media (max-width: 999px) { ignored } */
        @media (max-height: 820px),
            (max-width: 900px) { .card { max-width: 100px; } }
        @media (max-width: 900px) and (pointer: coarse) { .card { display: grid; } }
        @container card (max-width: 22rem) { .field { display: block; } }
    ''') == {'900px': 2}


# DEVELOPMENT "Design System": `:has()` stays in its rule's subject compound, and no `:is()`/`:where()`/`:not()` in its
# argument holds a combinator. For other forms WebKit can re-check every element of the document on a DOM write the
# argument may match. Through Safari 26 (the macOS app's WKWebView, iOS and the Telegram Mini App on it) that holds for
# nearly every form (26.1 scopes three parent and ancestor shapes) and costs about 1-2 ms per write on a 20k-element
# window in 26.4, so a busy chat paints in jerks; current trunk (Playwright's WebKit) still does it for a `:has()`
# before `+`/`~` when an attribute or class in its argument changes, and for a scope-breaking argument. Browser tests
# assert geometry, not cost, so this static check is the guard. Rewrite by asking from the styled element
# (`.a > .x:has(~ .b)`) or by letting a common parent hold the `:has()` and set an inherited custom property the styled
# element reads (`--project-answer-tail` in style.css).
COMBINATORS = " \t\n>+~"


def selectors_by_line(css):
    """(line, selector list) of every style rule; at-rule preludes and declarations are not selectors."""
    css = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group()), css, flags=re.S)
    for match in re.finditer(r"([^{};]+)\{", css):
        selectors = match.group(1).strip()
        if selectors and not selectors.startswith("@"):
            yield css.count("\n", 0, match.end(1) - len(match.group(1).lstrip())) + 1, selectors


def top_level(text, separators):
    """Pieces between separators outside brackets; a run of spaces and combinators is one boundary."""
    pieces, depth, start, i = [], 0, 0, 0
    while i < len(text):
        depth += (text[i] in "([") - (text[i] in ")]")
        if depth == 0 and text[i] in separators:
            end = i + 1
            while separators != "," and end < len(text) and text[end] in COMBINATORS:
                end += 1
            pieces.append(text[start:i])
            start = i = end
            continue
        i += 1
    return [piece.strip() for piece in pieces + [text[start:]] if piece.strip()]


def arguments(text, names):
    """(name, argument) of each named pseudo-class function in text, at any depth."""
    for function in re.finditer(rf":({names})\(", text, re.I):
        depth, end = 1, function.end()
        while depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        yield function.group(1).lower(), text[function.end():end - 1]


def unscoped_has(selectors):
    """A `:has()` left of a combinator, or one whose argument hides a combinator in `:is()`/`:where()`/`:not()`."""
    selectors = re.sub(r'"[^"]*"|\'[^\']*\'', '""', selectors)  # quoted text is never a selector
    for selector in top_level(selectors, ","):
        *left, subject = top_level(selector, COMBINATORS)
        if ":has(" in " ".join(left).lower():
            return True
        for name, argument in arguments(subject, "has|is|where|not|matches|-webkit-any"):
            if name != "has" and unscoped_has(argument):
                return True
            if name == "has" and any(len(top_level(inner, COMBINATORS)) > 1
                                     for _, logical in arguments(argument, "is|where|not|matches|-webkit-any")
                                     for inner in top_level(logical, ",")):
                return True
    return False


def test_has_stays_in_the_subject_compound():
    rules = [(path.relative_to(ROOT).as_posix(), line, selectors) for path in first_party_sheets()
             for line, selectors in selectors_by_line(path.read_text(encoding="utf-8"))]
    assert not [rule for rule in rules if "&" in rule[2]], "CSS nesting: teach selectors_by_line the parent selector"
    found = [f"{path}:{line}: {' '.join(selectors.split())}" for path, line, selectors in rules
             if unscoped_has(selectors)]
    assert not found, "Keep :has() in its subject compound (comment above):\n" + "\n".join(found)


# A `:has()` on what holds a room's rows. Measured in WebKit (2026-11-05): a room's `:has()` beside the rows' own made each
# live card's log frame cost 3-4x (about 8 -> 2 ms with 20 rooms kept) and grow with every room kept away, though hidden; a
# busy burst's script doubled. No browser test sees cost: ask from a narrower element, or from the element itself with
# `:empty` (`.chat-panel-statusbar`, style.css).
ROW_HOLDERS = re.compile(r"(?:\.chat-instance-panel|\.project-panel|\.chat-messages|#chat-messages|#page-chat|^body|^html|:root)(?![\w-])")


def has_on_a_row_holder(selectors):
    """A `:has()`, at any depth of the subject compound, whose compound names an element holding a room's rows."""
    for selector in top_level(re.sub(r'"[^"]*"|\'[^\']*\'', '""', selectors), ","):
        subject = top_level(selector, COMBINATORS)[-1]
        if ":has(" in subject.lower() and ROW_HOLDERS.search(re.sub(r"\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)", "", subject)):
            return True
    return False


def test_no_has_on_an_element_holding_a_rooms_rows():
    found = [f"{path.relative_to(ROOT).as_posix()}:{line}: {' '.join(selectors.split())}" for path in first_party_sheets()
             for line, selectors in selectors_by_line(path.read_text(encoding="utf-8")) if has_on_a_row_holder(selectors)]
    assert not found, "Ask from a narrower element or with :empty (comment above):\n" + "\n".join(found)
    for selector in (".project-panel .chat-instance-panel:has(> .bar > :not([hidden]))", ".x, .chat-messages:not(:has(> *))",
                     "#page-chat:has(.y)", "body:has(.y)", ".project-panel:has(.y)", ":root:has(.y)"):
        assert has_on_a_row_holder(selector), selector
    for selector in (".chat-bubble:has(> .x)", ".project-panel-bar:has(> .x)", ".chat-messages > .row:has(.x)",
                     ".project-panel .chat-panel-statusbar:not(:empty) ~ .chat-messages", ".nobody:has(.chat-messages)"):
        assert not has_on_a_row_holder(selector), selector


# DEVELOPMENT "Design System": no `:has()` argument tests `:empty`, `:focus`, or `:focus-within`, at any depth. The system WebKit (the macOS app's
# WKWebView, Safari, iOS; measured on 26.4, 2026-11-15) does not re-match it when content comes or goes after the
# first paint, so the rule keeps its first answer until something else restyles the subtree: Back placed in a header
# painted without it stood alone at the window's top edge (owner 14.11), and a Settings row whose words came later kept
# its line hidden. Playwright's WebKit re-matches it, so no browser test sees this; this static check is the guard. Ask
# for a child (`.x:has(> .slot > *)`) where the tested element only holds elements, or let the text's one writer say it
# in an attribute the sheet keys on (`data-no-meta`, ui_primitives.js setRowMeta). A plain `:empty` on the element
# itself is re-matched.
def late_state_in_has(selectors):
    """A `:has()` whose argument tests `:empty`, `:focus`, or `:focus-within`, at any depth (inside `:not()`/`:is()`, after a combinator)."""
    selectors = re.sub(r'"[^"]*"|\'[^\']*\'', '""', selectors)  # quoted text is never a selector
    return any(re.search(r"(?<!\\):(?:empty|focus|focus-within)(?![\w-])", argument, re.I) for _, argument in arguments(selectors, "has"))


def test_has_never_tests_late_state():
    found = [f"{path.relative_to(ROOT).as_posix()}:{line}: {' '.join(selectors.split())}" for path in first_party_sheets()
             for line, selectors in selectors_by_line(path.read_text(encoding="utf-8")) if late_state_in_has(selectors)]
    assert not found, "Ask for a child or a writer's attribute, never :empty, :focus, or :focus-within inside :has() (comment above):\n" + "\n".join(found)
    for selector in (".a:has(:focus)", ".a:has(:not(:focus-within))", ".a:has(:empty)", ".a:has(> .b:empty)", ".a:not(:has(> :not(:empty)))", ".a:has(> :is(.b, .c:EMPTY))",
                     ".a:has(+ .b > .c:not(:empty))", ".a:has(~ .b .c:empty) .d", ".x, .a:is(.b:has(> .c > :empty))"):
        assert late_state_in_has(selector), selector
    for selector in (".a:has(> .b:focus-visible)", ".a:has(> [data-focused])", ".a:empty", ".a:not(:empty) ~ .b", ".a:has(> *)", ".a:not(:has(> .b > *))", ".a:empty:has(> .b)",
                     ".a:has(> .is-empty)", '.a:has([aria-label=":has(:empty)"])', ".a:has(> .b\\:empty)"):
        assert not late_state_in_has(selector), selector
    sheet = '/* .a:has(:empty) { } */ .b:empty::after { content: ":has(:empty)"; } .c:has(> .d > *) { color: red; }'
    assert [(line, late_state_in_has(selectors)) for line, selectors in selectors_by_line(sheet)] == [(1, False), (1, False)]


def test_has_guard_reads_every_selector_position():
    css = """
        /* .x:has(.y) .z { color: red; } */
        .project-panel .chat-panel-statusbar:has(> :not([hidden])) + .chat-messages { padding-top: 8px; }
        .project-panel .chat-instance-panel:has(> .chat-panel-statusbar > :not([hidden])) { --chat-feed-reserve: 8px; }
        @media (hover: hover) { .a:has(> .b:hover) > .c { color: red; } }
        @supports selector(:has(*)) { .d:has(+ .e)::after { content: ".f:has(.g) .h"; } }
        .i:not(:has(> :not([hidden]))), .j:is(.k:has(.l)) { display: none; }
        .m > :is(.n, .o:has(.p) ~ .q) { color: red; }
        .r:where(.s .t:has(.u)), .v { color: red; }
        .w:has(.x),
        .y:has(> .z) .aa { color: red; }
        .ab:has(:not(.cd .ef)), .gh:has(> :is(.ij, .kl:hover)) { color: red; }
        .mn:has(> :is(.op, [aria-label="q r :has(.s) t"])) { color: red; }
        .card > .button:not(:has(~ .foot > .meta > *)), .p:has(.btn):has(+ .t) { padding: 0; }
        #page .hdr[data-empty]:has(~ .scroll .card.dragging), .bar:has(.btn) + .time { margin: 0; }
    """
    assert [(line, unscoped_has(selectors)) for line, selectors in selectors_by_line(css)] == [
        (3, True), (4, False), (5, True), (6, False), (7, False), (8, True), (9, False), (10, True), (12, True), (13, False),
        (14, False), (15, True)]


# Principle 9's static seam: this guard catches the exact form the stylesheet can state without a DOM walk — a rule
# whose selector explicitly names a shared `.scroll-fade-y` scroller and a descendant that is sticky/fixed and painted.
# Portalled floating glass, the composer and a table head are the named allowed forms; an opaque edge in any other
# explicit descendant is a cut. Rules such as `.logs-filters` whose ancestor relationship exists only in markup are
# deliberately outside this static check and remain visual evidence.
SCROLLER_DESCENDANT = re.compile(r"(?:\.scroll-fade-y|#chat-messages|\.chat-messages|\.document-reader-body|\.rw-shade)\s+[^{}]*")
POSITIONED = re.compile(r"\bposition\s*:\s*(?:sticky|fixed)\b", re.I)
PAINT = re.compile(r"\bbackground(?:-color)?\s*:\s*([^;]+)", re.I)


def positioned_opaque_scroller_descendants(css):
    css = without_comments(css)
    violations = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = match.group(1).strip(), match.group(2)
        if not SCROLLER_DESCENDANT.search(selector) or not POSITIONED.search(body):
            continue
        paint = PAINT.search(body)
        if not paint:
            continue
        value = paint.group(1).strip().lower()
        if value in {"none", "transparent", "initial", "inherit", "unset"}:
            continue
        lower_selector = selector.lower()
        if "backdrop-filter" in body.lower() or "table" in lower_selector or "thead" in lower_selector \
                or "composer" in lower_selector or "#chat-input-area" in lower_selector:
            continue
        violations.append(selector)
    return violations


def test_scroller_edges_do_not_gain_opaque_positioned_descendants():
    actual = [str(path.relative_to(ROOT)) for path in first_party_sheets()
              if positioned_opaque_scroller_descendants(path.read_text(encoding="utf-8"))]
    assert not actual, "Opaque sticky/fixed paint must not cover a moving scroller: " + ", ".join(actual)

    broken = ".scroll-fade-y .bad-edge { position: sticky; top: 0; background: var(--bg-primary); }"
    assert positioned_opaque_scroller_descendants(broken), "the guard must fire on an opaque cut"
    allowed = """
        .scroll-fade-y .composer { position: fixed; background: var(--surface-float); backdrop-filter: var(--glass); }
        .scroll-fade-y table thead { position: sticky; top: 0; background: var(--surface-object); }
        .scroll-fade-y .quiet { position: sticky; top: 0; background: transparent; }
    """
    assert not positioned_opaque_scroller_descendants(allowed), "composer, table heads and transparent edges stay allowed"
