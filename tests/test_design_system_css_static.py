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
# element reads.
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


# Every check box is the shared .ui-checkbox and every radio the shared .ui-radio (DESIGN "Controls and editable
# choices", owner decision 27.11.2026): their size, shape, fill, mark and border live only in ui.css. A page places a
# box; it never draws one.
CHECKBOX_PAINT = re.compile(r"(?:^|[;{\s])(?:(?:min-|max-)?(?:width|height)|(?:min-|max-)?(?:inline|block)-size|border[\w-]*"
                            r"|background[\w-]*|accent-color|appearance|-webkit-appearance|outline[\w-]*|box-shadow|color"
                            r"|(?:-webkit-)?mask[\w-]*)\s*:", re.I)
CHECKBOX_SUBJECT = re.compile(r"\[type=[\"']?(?:checkbox|radio)|\.ui-checkbox|\.ui-radio|\.md-checkbox|\.message-share-check|\.message-share-task", re.I)


def checkbox_paint_outside_the_control(css):
    """Rules of a page sheet that target a check box and paint it, or style a checkbox input by its type."""
    found = []
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", without_comments(css)):
        target = re.sub(r":not\((?:[^()]|\([^()]*\))*\)", "", selector)
        if re.search(r"\[type=[\"']?(?:checkbox|radio)", target, re.I) or (CHECKBOX_SUBJECT.search(target) and CHECKBOX_PAINT.search(body)):
            found.append(" ".join(selector.split()))
    return found


def shape_problems(ui_css):
    """The check box's shape is the one token, whatever its value; a radio is a circle whatever the token says."""
    rules = re.findall(r"([^{}]+)\{([^{}]*)\}", without_comments(ui_css))
    found = []
    for selector, body in rules:
        radius = re.search(r"border-radius:\s*([^;]+);", body)
        if not radius or not re.search(r"\.ui-(?:checkbox|radio)\b", selector):
            continue
        names = [part.strip() for part in selector.split(",")]
        value = radius[1].strip()
        if any(re.fullmatch(r"\.ui-radio", name) for name in names) and len(names) == 1:
            if value != "50%":
                found.append(f"a radio is shaped {value}")
        elif value != "var(--checkbox-radius)":
            found.append(f"{' '.join(selector.split())} shapes the box {value}, not the one token")
    if not any(re.fullmatch(r"\s*\.ui-radio\s*", sel) and re.search(r"border-radius\s*:\s*50%\s*(?:;|$)", body)
               for sel, body in rules):
        found.append("no rule keeps the radio round on its own")
    return found


def test_only_the_shared_control_draws_a_check_box():
    sheets = {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8") for path in first_party_sheets()}
    found = [f"{name}: {rule}" for name, css in sheets.items() if name != "web/ui.css"
             for rule in checkbox_paint_outside_the_control(css)]
    assert not found, "A page places a check box, never sizes, recolours or reshapes it:\n" + "\n".join(found)
    ui = without_comments(sheets["web/ui.css"])
    shared = [" ".join(body.split()) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", ui)
              if {part.strip() for part in sel.split(",")} == {".ui-checkbox", ".ui-radio"}]
    assert len(shared) == 1, "one base rule draws both"
    assert "border-radius: var(--checkbox-radius);" in shared[0] and "width: var(--checkbox-size);" in shared[0]
    assert len(re.findall(r"--checkbox-radius:", ui)) == 1, "one token flips every box between round and square"
    assert shape_problems(ui) == [], shape_problems(ui)

    assert checkbox_paint_outside_the_control('.local-toggle input[type="checkbox"] { cursor: pointer; }')
    assert checkbox_paint_outside_the_control(".x .ui-checkbox { width: 18px; }")
    assert checkbox_paint_outside_the_control(".md-checkbox { border-radius: 4px; }")
    assert not checkbox_paint_outside_the_control(".x > .ui-checkbox { grid-column: 1; margin-top: 2px; }")
    assert checkbox_paint_outside_the_control('.x input[type="radio"] { margin: 0; }')
    assert not checkbox_paint_outside_the_control("input:not(.ui-control):not(.ui-checkbox) { border: 1px solid; }")
    good = ".ui-checkbox, .ui-radio { border-radius: var(--checkbox-radius); }\n.ui-radio { border-radius: 50%; }"
    assert shape_problems(good) == [] and shape_problems(good.replace("var(--checkbox-radius)", "50%"))
    assert shape_problems(".ui-checkbox, .ui-radio { border-radius: var(--checkbox-radius); }"), "radios follow the token"
    assert shape_problems(good + "\n.ui-checkbox:checked { border-radius: 4px; }")


FORCED_SYSTEM_COLOUR = re.compile(r"^(?:Canvas|CanvasText|Highlight|HighlightText|ButtonText|ButtonFace|ButtonBorder|"
                                  r"LinkText|GrayText|Field|FieldText|Mark|MarkText|AccentColor|AccentColorText)$")


def forced_mark_problems(ui_css):
    """Inside forced colours, a drawn check box that opts out of the user's palette paints only in system colours, and
    its edge, its ticked fill and its tick are all given (unchecked and checked both show)."""
    css = without_comments(ui_css)
    found, given = [], set()
    for block in re.findall(r"@media \(forced-colors: active\) \{((?:[^{}]*\{[^{}]*\})*)\s*\}", css):
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", block):
            if ".ui-checkbox" not in selector or "input" in selector.split(":not")[0].split(".")[0]:
                continue
            for prop, value in re.findall(r"(border-color|background-color|background)\s*:\s*([^;]+);", body):
                if not FORCED_SYSTEM_COLOUR.match(value.strip()):
                    found.append(f"{' '.join(selector.split())} {prop}: {value.strip()}")
                given.add(("::before" in selector, "checked" in selector, prop.split("-")[0]))
    for need in [(False, False, "border"), (False, True, "background"), (True, False, "background")]:
        if need not in given:
            found.append(f"forced colours give no {need}")
    if "forced-color-adjust: none" in css and not given:
        found.append("a drawn mark leaves the palette without system colours")
    # The edge differs from the surface under it, and the tick from the ticked fill (CanvasText on Canvas, HighlightText
    # on Highlight): a system colour on its own colour is the 1:1 mark again.
    colours = {}
    for block in re.findall(r"@media \(forced-colors: active\) \{((?:[^{}]*\{[^{}]*\})*)\s*\}", css):
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", block):
            if ".ui-checkbox:not(input)" not in selector:
                continue
            state = "tick" if "::before" in selector else "checked" if "checked" in selector else "unchecked"
            for prop, value in re.findall(r"(border-color|background-color|background)\s*:\s*([^;]+);", body):
                colours[(state, prop.split("-")[0])] = value.strip()
    if colours.get(("unchecked", "border")) and colours.get(("unchecked", "border")) == colours.get(("unchecked", "background")):
        found.append("the unchecked edge is the colour of its own fill")
    if colours.get(("tick", "background")) and colours.get(("tick", "background")) == colours.get(("checked", "background")):
        found.append("the tick is the colour of the ticked fill")
    return found


def test_forced_colours_keep_every_drawn_mark_visible():
    ui = (WEB / "ui.css").read_text(encoding="utf-8")
    assert forced_mark_problems(ui) == [], forced_mark_problems(ui)
    broken = "@media (forced-colors: active) {\n    .ui-checkbox:not(input) { forced-color-adjust: none; }\n}"
    assert forced_mark_problems(broken), "opting out of the palette with nothing in its place is the 1:1 white-on-white mark"
    themed = ("@media (forced-colors: active) {\n    .ui-checkbox:not(input) { forced-color-adjust: none; border-color: var(--x);"
              " background-color: Canvas; }\n    .ui-checkbox:not(input):is(.is-checked) { background-color: Highlight; }\n"
              "    .ui-checkbox:not(input)::before { background: HighlightText; }\n}")
    assert forced_mark_problems(themed) and not forced_mark_problems(themed.replace("var(--x)", "CanvasText"))
    assert forced_mark_problems(themed.replace("var(--x)", "Canvas")), "Canvas on Canvas is the vanished edge"
    assert forced_mark_problems(themed.replace("var(--x)", "CanvasText").replace("HighlightText", "Highlight"))


def checkbox_column_problems(css):
    """A grid that places a check box in its first column reserves that column at the box's own size."""
    rules = re.findall(r"([^{}]+)\{([^{}]*)\}", without_comments(css))
    owners = {" ".join(sel.split()).split(" > .ui-checkbox")[0] for sel, body in rules
              if " > .ui-checkbox" in sel and re.search(r"grid-column:\s*1\b", body)}
    found = []
    for selector, body in rules:
        name = " ".join(selector.split())
        track = re.search(r"grid-template-columns:\s*([^\s;]+)", body)
        if name in owners and track and track[1] != "var(--checkbox-size)":
            found.append(f"{name} reserves {track[1]}")
    return found


def test_a_check_box_column_is_the_box_s_own_size():
    found = [f"{path.name}: {item}" for path in first_party_sheets()
             for item in checkbox_column_problems(path.read_text(encoding="utf-8"))]
    assert not found, found
    rows = ".r { display: grid; grid-template-columns: 14px minmax(0, 1fr); }\n.r > .ui-checkbox { grid-column: 1; }"
    assert checkbox_column_problems(rows) and not checkbox_column_problems(rows.replace("14px", "var(--checkbox-size)"))


CHECKBOX_TYPE = re.compile(r"""type\s*=\s*\\?["'](checkbox|radio)""")
DOM_CHECKBOX_TYPE = re.compile(r"""(?:\.type\s*=\s*|setAttribute\(\s*['"]type['"]\s*,\s*)['"](checkbox|radio)""")
SHARED_CLASS = {"checkbox": "ui-checkbox", "radio": "ui-radio"}


def tags_in(source, name="input"):
    """Every `<name …>` tag of a markup string or template, `>` inside a `${…}` expression included (as tagFrom in
    web/tests/ui_primitives.test.js reads them)."""
    found, start = [], source.find("<" + name)
    while start >= 0:
        depth, index = 0, start
        while index < len(source):
            if source.startswith("${", index):
                depth += 1
                index += 1
            elif depth and source[index] == "{":
                depth += 1
            elif depth and source[index] == "}":
                depth -= 1
            elif not depth and source[index] == ">":
                found.append(source[start:index + 1])
                break
            index += 1
        start = source.find("<" + name, start + 1)
    return found


SWITCH_ROLE = re.compile(r"""role\s*=\s*\\?["']switch""")


def checkbox_inputs_outside_the_control(js):
    """Check box and radio inputs a source builds without their shared class: an `<input …>` tag, or a DOM-built input
    whose type is set within three lines of no shared class. A switch (role="switch") is its own control."""
    found = []
    for tag in tags_in(js):
        kind = CHECKBOX_TYPE.search(tag)
        if kind and not re.search(rf"\b{SHARED_CLASS[kind[1]]}\b", tag) and not SWITCH_ROLE.search(tag):
            found.append(" ".join(tag.split())[:120])
    lines = js.splitlines()
    for index, line in enumerate(lines):
        kind = DOM_CHECKBOX_TYPE.search(line)
        if kind:
            near = "\n".join(lines[max(0, index - 3):index + 4])
            if not re.search(rf"\b{SHARED_CLASS[kind[1]]}\b", near) and "switch" not in near:
                found.append(f"line {index + 1}: {line.strip()[:120]}")
    return found


def first_party_markup_sources():
    vendored = {"marked.min.js", "mermaid.min.js", "chart.umd.min.js", "highlight.min.js", "purify.min.js"}
    return sorted(path for pattern in ("*.html", "*.js", "modules/**/*.js") for path in WEB.glob(pattern)
                  if path.name not in vendored and "vendor" not in path.parts and "tests" not in path.parts)


def test_every_module_check_box_is_the_shared_control():
    found = [f"{path.relative_to(ROOT).as_posix()}: {item}" for path in first_party_markup_sources()
             for item in checkbox_inputs_outside_the_control(path.read_text(encoding="utf-8"))]
    assert not found, "Every check box is .ui-checkbox (DESIGN 'Controls and editable choices'):\n" + "\n".join(found)

    assert checkbox_inputs_outside_the_control('html = `<input type="checkbox" data-x>`;')
    assert checkbox_inputs_outside_the_control('html = `<input ${n > 0 ? "checked" : ""} type="checkbox">`;')
    assert checkbox_inputs_outside_the_control("const box = document.createElement('input');\nbox.type = 'checkbox';")
    assert not checkbox_inputs_outside_the_control('html = `<input class="ui-checkbox" type="checkbox" data-x>`;')
    assert not checkbox_inputs_outside_the_control('html = `<input ${n > 0 ? "checked" : ""} class="ui-checkbox" type="checkbox">`;')
    assert not checkbox_inputs_outside_the_control("const box = document.createElement('input');\nbox.type = 'checkbox';\nbox.className = 'ui-checkbox';")
    assert not checkbox_inputs_outside_the_control('html = `<input type="checkbox" role="switch" data-x>`;')
    assert not checkbox_inputs_outside_the_control("html = `<input type='checkbox' role='switch'>`;")
    assert checkbox_inputs_outside_the_control('html = `<input class="ui-checkbox" type="radio" name="x">`;')
    assert not checkbox_inputs_outside_the_control('html = `<input class="ui-radio" type="radio" name="x">`;')


def attach_command_tray_rules(sheets):
    """Every rule, in any sheet, whose subject is the terminal command tray itself."""
    return [(name, " ".join(selector.split()), body) for name, css in sheets.items()
            for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", without_comments(css))
            if any(re.search(r"(?:^|[\s>+~])\.harness-attach-command(?::[\w-]+)*$", part.strip())
                   for part in selector.split(","))]


PLACEMENT = re.compile(r"^(?:margin[\w-]*|align-self|justify-self|place-self|order|grid-[\w-]+)$")


def tray_drawing_rules(sheets):
    """The rules that draw the tray: any declaration beyond placing it in its parent."""
    return [(name, selector) for name, selector, body in attach_command_tray_rules(sheets)
            if any(not PLACEMENT.match(prop.strip().lower()) for prop in re.findall(r"([\w-]+)\s*:", body))]


def test_the_terminal_command_tray_is_defined_once():
    sheets = {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8") for path in first_party_sheets()}
    assert len(tray_drawing_rules(sheets)) == 1, tray_drawing_rules(sheets)
    one = ".harness-attach-command { padding: 8px; background: var(--surface-tray); }"
    placed = "\n.harness-external-terminal > .harness-attach-command { align-self: stretch; margin: 0; }"
    assert len(tray_drawing_rules({"a": one + placed})) == 1, "placing the tray is not a second recipe"
    assert len(tray_drawing_rules({"a": one + "\n.harness-login-card .harness-attach-command { padding: 0; }"})) == 2
    assert len(tray_drawing_rules({"a": one, "b": ".harness-attach-command { padding: 4px; }"})) == 2
    assert not attach_command_tray_rules({"a": ".harness-external-terminal > .harness-attach-command-note { margin: 0; }"})
