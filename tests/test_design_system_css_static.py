"""Design-system debt ratchets for first-party sheets (P10/P11)."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def first_party_sheets():
    return sorted(path for path in WEB.rglob("*.css") if "vendor" not in path.relative_to(WEB).parts)


def without_comments(text):
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


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
    'web/onboarding.css': {'1100px': 1, '900px': 1, '760px': 1, '720px': 1},  # wizard
    'web/room_work.css': {'900px': 1},  # existing Shade height recipe (L1)
    'web/settings.css': {'980px': 2, '760px': 1, '640px': 1, '600px': 1},  # existing form/MCP layout debt
    'web/style.css': {'980px': 1, '640px': 2, '620px': 2},  # legacy content only: the shell's queries are gone (L1)
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
# element reads (`--chat-feed-reserve` in style.css).
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
        .card > .button:not(:has(~ .foot > .meta:not(:empty))), .p:has(.btn):has(+ .t) { padding: 0; }
        #page .hdr[data-empty]:has(~ .scroll .card.dragging), .bar:has(.btn) + .time { margin: 0; }
    """
    assert [(line, unscoped_has(selectors)) for line, selectors in selectors_by_line(css)] == [
        (3, True), (4, False), (5, True), (6, False), (7, False), (8, True), (9, False), (10, True), (12, True), (13, False),
        (14, False), (15, True)]
