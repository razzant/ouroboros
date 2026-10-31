"""Amber is the owner's attention and nothing else (docs/DESIGN.md "Status marks").

Every stylesheet rule that paints amber — through an amber token, a custom property that aliases one, or an amber
literal — is listed in ``AMBER_REGISTRY`` under one of its four meanings: the owner's move, a warning worth reading,
the System voice, or a data category that is not a status (plus the palette that defines the tokens). A new rule that
paints amber fails until it is listed under one of them; a listed rule that no longer paints amber fails until its
line is removed, so the list only says what is true. Scripts may not paint amber beyond the counts in
``SCRIPT_AMBER``. What goes on by itself or what someone chose (queued, waiting, paused, stopped, unconfirmed, work in
flight, a mode, a level, a setting value) never belongs here.

The scan reads the first-party sheets as text: comments and formatting do not matter, nested at-rules and nested
rules do, and every selector of a selector list is its own entry.
"""

from __future__ import annotations

import colorsys
import pathlib
import re
from typing import Iterator

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "web"
SHEETS = ("web/ui.css", "web/style.css", "web/settings.css", "web/model_roles.css", "web/model_wait.css",
          "web/room_work.css", "web/onboarding.css", "web/cards.css", "web/consciousness.css")
CATEGORIES = ("owner", "warning", "system", "data", "palette")

# category | sheet | [at-rule context » ]selector | properties
AMBER_REGISTRY = """
# The owner's move: a question to answer or an action only the owner can take (sign in, a spending limit, a
# suggested trust to confirm, local changes before an update, an Evolution chip that needs the owner).
owner | web/ui.css | .status-mark[data-hue="owner"] | color
owner | web/style.css | .chat-budget-pill[data-attention="1"] .chat-budget-bar | background
owner | web/style.css | .chat-budget-pill[data-attention="1"] .chat-budget-bar-fill | background
owner | web/style.css | .chat-budget-pill[data-attention="1"] .chat-budget-text | color
owner | web/style.css | .chat-live-phase[data-chip-tone="owner"] | background, border-color, color
owner | web/style.css | .chat-question-nav | background, border, color
owner | web/style.css | .chat-question-nav:hover | border-color
owner | web/style.css | .skills-trust-value.is-suggested | color
owner | web/style.css | :root[data-theme] .update-pill:is(.has-local, [data-tone="warn"]):not(:disabled) | background, color
owner | web/style.css | @media (hover: hover) » :root[data-theme] .update-pill:is(.has-local, [data-tone="warn"]):not(:disabled):hover | background
owner | web/style.css | :root[data-theme] .chat-budget-pill.ui-chrome-control[data-attention="1"] | background
owner | web/style.css | :root[data-theme] .chat-question-nav | background, color
owner | web/style.css | :root[data-theme] .chat-question-nav:hover | background
owner | web/style.css | @media (hover: hover) » :root[data-theme] .chat-budget-pill.ui-chrome-control[data-attention="1"]:hover | background
owner | web/model_wait.css | .model-wait-row[data-owner="1"] | background, border-color
owner | web/room_work.css | .rw-row .rw-pill-question | background, border-color, color
owner | web/room_work.css | .rw-row .rw-pill-question:hover | background, border-color
owner | web/room_work.css | .rw-row[data-tone="wait"] .rw-state | color
owner | web/room_work.css | .rw-trigger[data-tone="wait"] | color
owner | web/room_work.css | :root[data-theme] .rw-row .rw-pill-question | background, color
owner | web/room_work.css | :root[data-theme] .rw-row .rw-pill-question:not(:disabled):hover | background
# A warning worth reading: warn sentences and chips, toasts, notes, warnings inside timelines, reviews, delegated
# work, Logs events, widget tones, account quota and spent windows, unsaved settings and file drafts.
warning | web/ui.css | .settings-inline-status[data-tone="warn"] | color
warning | web/ui.css | .settings-test-result[data-tone="warn"] | color
warning | web/ui.css | .ui-chip[data-tone="warn"] | background, border-color, color
warning | web/ui.css | .ui-status[data-tone="warn"] | color
warning | web/ui.css | :is(.ui-status, .settings-inline-status, .wizard-runtime-status)[data-tone="warn"]:not(:empty)::before | background
warning | web/style.css | .chat-composer-chip[data-tone="warn"] .chat-chip-dot | background
warning | web/style.css | .chat-delegated-final | color
warning | web/style.css | .chat-delegated-problem | color
warning | web/style.css | .chat-live-line.warn .chat-live-line-title | color
warning | web/style.css | .chat-quiz-storage-note | color
warning | web/style.css | .document-reader-notice | background, border-bottom, color
warning | web/style.css | @container files (max-width: 700px) » .files-entry:is(.is-draft, [data-unsaved]) .files-entry-unsaved | color
warning | web/style.css | .harness-exhausted | background
warning | web/style.css | .harness-exhausted .harness-account-meta | color
warning | web/style.css | .harness-maintenance-notice[data-tone="warn"] | color
warning | web/style.css | .log-entry .log-phase:is(.warn, .empty):not([data-chip-tone]) | background, border-color, color
warning | web/style.css | .msg-attachment-outcome | color
warning | web/style.css | .new-project-trust-note | border, color
warning | web/style.css | .settings-lan-hint[data-tone="warn"] | border-color, color
warning | web/style.css | .skills-skip-glyph | color
warning | web/style.css | .toast-warn | border-color, color
warning | web/style.css | .updates-status-dot[data-tone="warn"] | color
warning | web/style.css | .widget-callout[data-tone="warn"] | border-left-color
warning | web/style.css | .widget-metric[data-tone="warn"] | border-left-color
warning | web/style.css | .widget-table-status[data-tone="warn"] | color
warning | web/style.css | :root[data-theme] :is(.harness-account-row, .agent-family-card .harness-account-row).harness-exhausted | background
warning | web/settings.css | .available-subagent-status-reason | color
warning | web/settings.css | .available-subagents-review-copies | color
warning | web/settings.css | .available-subagents-review-empty | color
warning | web/settings.css | .local-toggle[data-warning="1"] | color
warning | web/settings.css | .mcp-server-status-warn | background, color
warning | web/settings.css | .resource-message[data-tone="warn"] | background, color
warning | web/settings.css | .resource-window[data-spent="true"] .resource-fill | background
warning | web/settings.css | .settings-inline-note[data-tone="warn"] | color
warning | web/settings.css | .settings-unsaved-indicator | color
warning | web/onboarding.css | .agent-family-status[data-tone="warn"] | color
warning | web/onboarding.css | .agent-service-note[data-tone="warn"] | color
warning | web/onboarding.css | .available-subagent-status-reason | color
warning | web/onboarding.css | .available-subagents-review-copies | color
warning | web/onboarding.css | .available-subagents-review-empty | color
warning | web/onboarding.css | .settings-inline-note[data-tone="warn"] | color
warning | web/onboarding.css | .settings-inline-status[data-tone="warn"] | color
warning | web/onboarding.css | .wizard-inline-note[data-tone="warn"] | color
warning | web/onboarding.css | .wizard-test-result[data-tone="warn"] | border-color, color
# The System voice: its bubble, sender and text (chat and the Place reader), quotes of a System message, the skill review it shows.
system | web/style.css | .chat-bubble.system | background
system | web/style.css | .chat-bubble.system .message | color
system | web/style.css | .chat-bubble.system .sender | color
system | web/style.css | .chat-field > .chat-quote--composer | background
system | web/style.css | .chat-quote | background
system | web/style.css | .chat-quote-author | color
system | web/style.css | .chat-quote::before | background
system | web/style.css | .chat-quote[data-author="system"] | --quote-ink, --quote-tint
system | web/style.css | .skill-review-full | color
system | web/style.css | .skill-review-summary-button:hover | background
system | web/style.css | .skill-review-summary-main | color
system | web/style.css | .skill-review-toggle-label | color
system | web/style.css | :root[data-theme="light"] .skill-review-summary-main | color
system | web/consciousness.css | .place-convo-row.is-system | background
system | web/consciousness.css | .place-convo-row.is-system .place-convo-text | color
system | web/consciousness.css | .place-convo-row.is-system .place-convo-who | color
system | web/consciousness.css | .place-sheet-section p.place-silent-card | background, color
# Data categories, not statuses: the Logs task category, code highlighting, the Storage temporary segment, the
# fourth default chart series.
data | web/style.css | .log-type.tasks | color
data | web/style.css | .md-code-block .hljs-class .hljs-title | color
data | web/style.css | .md-code-block .hljs-type | color
data | web/style.css | :is(.storage-bar-seg, .storage-key)[data-group="temporary"] | background
data | web/style.css | :root[data-theme="light"] .md-code-block .hljs-class .hljs-title | color
data | web/ui.css | :root | --chart-series-4
data | web/ui.css | :root[data-theme="light"] | --chart-series-4
# The palette that defines the amber tokens and their aliases.
palette | web/ui.css | :root | --amber, --mark-owner, --status-warn-fg, --status-warn-word, --system-sender
palette | web/ui.css | :root[data-theme="light"] | --amber, --amber-dim, --status-warn-bg, --status-warn-border, --status-warn-fg
palette | web/ui.css | :root[data-theme="light"] | --system-bubble-bg, --system-sender, --system-text, --tone-amber
palette | web/ui.css | :root[data-theme="light"] | --tone-amber-hover
"""

# How many amber references each first-party script may hold, all data categories: the fourth default chart series
# (theme_palette.js, which host charts draw from) and Logs' task category.
SCRIPT_AMBER = {"web/modules/log_events.js": 1, "web/modules/theme_palette.js": 1}


def _decommented(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _closing(css: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(css)


def _split_top(text: str, separator: str) -> list[str]:
    """Split on a separator outside parentheses and quotes (a data URI or a selector list may hold one)."""
    parts, depth, quote, start = [], 0, "", 0
    for index, char in enumerate(text):
        if quote:
            quote = "" if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _rules(css: str, context: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str, str]]:
    """(at-rule context, selector list, own declarations) for every rule, nested at-rules and rules included."""
    position = 0
    while (opening := css.find("{", position)) >= 0:
        header_start = max(position, css.rfind(";", position, opening) + 1, css.rfind("}", position, opening) + 1)
        header = " ".join(css[header_start:opening].split())
        closing = _closing(css, opening)
        body = css[opening + 1:closing]
        if header.startswith("@"):
            if not header.startswith("@font-face"):
                yield from _rules(body, context + (header,))
        else:
            own, nested, cursor = [], [], 0
            while (inner := body.find("{", cursor)) >= 0:
                start = max(cursor, body.rfind(";", cursor, inner) + 1, body.rfind("}", cursor, inner) + 1)
                own.append(body[cursor:start])
                end = _closing(body, inner)
                nested.append(body[start:end + 1])
                cursor = end + 1
            own.append(body[cursor:])
            yield context, header, "".join(own)
            if nested:
                yield from _rules("".join(nested), context + (header,))
        position = closing + 1


def declarations(css: str) -> Iterator[tuple[str, str, str, str]]:
    """(context » selector, property, value) per single selector; the sheet's custom properties included."""
    for context, selectors, body in _rules(_decommented(css)):
        for declaration in _split_top(body, ";"):
            prop, colon, value = declaration.partition(":")
            prop = prop.strip().lower()
            if not colon or not prop:
                continue
            value = " ".join(value.split())
            for selector in _split_top(selectors, ","):
                key = " ".join(selector.split())
                yield " » ".join(context + (key,)), prop, value


_HEX = re.compile(r"#([0-9a-fA-F]{3,8})\b")
_RGB = re.compile(r"rgba?\(\s*([0-9.]+)[\s,]+([0-9.]+)[\s,]+([0-9.]+)")
_NAMED = re.compile(r"(?<![\w-])(orange|gold|goldenrod|darkgoldenrod|darkorange)(?![\w-])", re.I)
_VAR = re.compile(r"var\(\s*(--[\w-]+)")


def _amber_rgb(red: float, green: float, blue: float) -> bool:
    hue, lightness, saturation = colorsys.rgb_to_hls(red / 255, green / 255, blue / 255)
    return 28 / 360 <= hue <= 52 / 360 and saturation >= 0.45 and 0.15 <= lightness <= 0.9


def amber_literal(value: str) -> bool:
    for match in _HEX.finditer(value):
        digits = match.group(1)
        if len(digits) in (3, 4):
            digits = "".join(char * 2 for char in digits[:3])
        elif len(digits) in (6, 8):
            digits = digits[:6]
        else:
            continue
        if _amber_rgb(int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)):
            return True
    if any(_amber_rgb(*(float(part) for part in match.groups())) for match in _RGB.finditer(value)):
        return True
    return bool(_NAMED.search(value))


def amber_declarations(sheets: dict[str, str]) -> set[tuple[str, str, str]]:
    """Every (sheet, selector, property) that paints amber, through a literal or any alias of an amber token."""
    rows = [(sheet, selector, prop, value) for sheet, css in sheets.items()
            for selector, prop, value in declarations(css)]
    tokens: set[str] = set()
    grew = True
    while grew:
        grew = False
        for _sheet, _selector, prop, value in rows:
            if prop.startswith("--") and prop not in tokens and (
                    amber_literal(value) or tokens.intersection(_VAR.findall(value))):
                tokens.add(prop)
                grew = True
    return {(sheet, selector, prop) for sheet, selector, prop, value in rows
            if amber_literal(value) or tokens.intersection(_VAR.findall(value))}


def registry(text: str = AMBER_REGISTRY) -> dict[tuple[str, str, str], str]:
    entries: dict[tuple[str, str, str], str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        category, sheet, selector, props = (part.strip() for part in line.split(" | "))
        assert category in CATEGORIES, line
        for prop in props.split(","):
            key = (sheet, " ".join(selector.split()), prop.strip())
            assert key not in entries, f"listed twice: {key}"
            entries[key] = category
    return entries


def _sheets() -> dict[str, str]:
    return {sheet: (REPO_ROOT / sheet).read_text(encoding="utf-8") for sheet in SHEETS}


def test_every_rule_that_paints_amber_is_listed_with_its_meaning() -> None:
    found = amber_declarations(_sheets())
    listed = registry()
    unlisted = sorted(found - listed.keys())
    stale = sorted(listed.keys() - found)
    assert not unlisted, (
        "these rules paint amber without a listed meaning. Amber is the owner's attention: list a rule under owner "
        "(a question or an action only the owner can take), warning (worth reading), system (the System voice) or "
        "data (a category, not a status) — or give it a neutral tone (docs/DESIGN.md \"Status marks\"):\n"
        + "\n".join(" | ".join(key) for key in unlisted))
    assert not stale, "these listed rules no longer paint amber; remove their lines:\n" + "\n".join(
        " | ".join(key) for key in stale)


def test_the_sheets_scanned_are_every_first_party_sheet() -> None:
    assert sorted(f"web/{path.name}" for path in WEB.glob("*.css")) == sorted(SHEETS)


def test_scripts_paint_no_amber_beyond_their_listed_counts() -> None:
    scripts = [WEB / "app.js", WEB / "theme.js", *sorted((WEB / "modules").rglob("*.js"))]
    counts = {}
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        tokens = re.findall(r"var\(--(?:amber|status-warn|tone-amber|mark-owner|system-)[\w-]*", text)
        literals = [match for match in re.findall(r"#[0-9a-fA-F]{6}\b|rgba?\([^)]*\)", text) if amber_literal(match)]
        if tokens or literals:
            counts[path.relative_to(REPO_ROOT).as_posix()] = len(tokens) + len(literals)
    assert counts == SCRIPT_AMBER, (
        "a script paints amber outside the stylesheet registry; paint through a listed class instead, or update "
        f"SCRIPT_AMBER with the meaning: {counts}")


def test_the_scan_sees_every_way_a_rule_can_paint_amber() -> None:
    css = """
    /* .comment { color: var(--amber) } */
    :root { --amber: #f59e0b; --quiet: #808080; }
    .alias { --lamp: var(--amber); }
    .uses-alias { border-color: var(--lamp); }
    @media (hover: hover) { @supports (display: grid) { .nested:hover { background: rgba(245, 158, 11, .2); } } }
    .a, .b { color: var(--amber) }
    .host { color: red; & .child { outline-color: gold; } }
    .fine { color: var(--quiet); background: url("data:image/svg+xml;utf8,<svg fill='%23000'/>"); }
    .crimson { color: #c93545; } .green { color: #22c55e; }
    """
    found = {(selector, prop) for _sheet, selector, prop in amber_declarations({"x.css": css})}
    assert found == {
        (":root", "--amber"), (".alias", "--lamp"), (".uses-alias", "border-color"),
        ("@media (hover: hover) » @supports (display: grid) » .nested:hover", "background"),
        (".a", "color"), (".b", "color"), (".host » & .child", "outline-color"),
    }


def test_the_registry_check_fails_on_a_new_rule_and_on_a_stale_line() -> None:
    sheet = {"web/x.css": ":root { --amber: #f59e0b }\n.owner-question { color: var(--amber); }\n"
                          ".new-lamp { color: var(--amber); }"}
    listed = registry("palette | web/x.css | :root | --amber\nowner | web/x.css | .owner-question | color\n"
                      "warning | web/x.css | .gone | color")
    found = amber_declarations(sheet)
    assert found - listed.keys() == {("web/x.css", ".new-lamp", "color")}
    assert listed.keys() - found == {("web/x.css", ".gone", "color")}
    # Formatting and comments never change what is found.
    reformatted = {"web/x.css": ":root{--amber:#F59E0B}/* lamp */ .owner-question,\n .new-lamp {\n  color :  var( --amber ) ;\n}"}
    assert amber_declarations(reformatted) == found
