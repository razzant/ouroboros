"""Regression test: settings_ui.js template literal backtick balance.

Prevents re-introduction of the v4.39.2 bug where an unescaped backtick
inside renderSettingsPage()'s outer template literal caused:
  ReferenceError: main is not defined
crashing initSettings, initCosts, and initAbout on page load.

Root cause: JS template literals use backtick delimiters. An inner
literal backtick (not escaped as \\`) terminates the outer template
string. The token following it is then evaluated as JS — e.g. `[main` —
which throws ReferenceError when `main` is not a defined variable.

This test isolates the body of renderSettingsPage() and verifies that
its outer template literal has balanced backticks.
"""
import pathlib
import re

SETTINGS_UI_JS = pathlib.Path(__file__).parent.parent / "web" / "modules" / "settings_ui.js"
def _count_unescaped_backticks(text: str) -> int:
    """Count backticks that are not preceded by a backslash."""
    count = 0
    i = 0
    while i < len(text):
        if text[i] == '\\':
            i += 2  # skip escaped character
            continue
        if text[i] == '`':
            count += 1
        i += 1
    return count


def _extract_render_settings_page_body(content: str) -> str:
    """Extract the body of renderSettingsPage() from settings_ui.js.

    Returns the text between 'export function renderSettingsPage()' and the
    matching closing brace so that backtick checks are scoped to this function
    only, not to the whole file.
    """
    # Find the function start
    start_marker = "export function renderSettingsPage()"
    start_idx = content.find(start_marker)
    if start_idx == -1:
        return ""
    # Walk forward to find the opening brace
    brace_start = content.find("{", start_idx)
    if brace_start == -1:
        return ""
    # Count braces to find the matching closing brace
    depth = 0
    i = brace_start
    while i < len(content):
        c = content[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return content[brace_start:i + 1]
        i += 1
    return content[brace_start:]  # fallback: rest of file


# ``test_settings_ui_js_exists`` and ``test_render_settings_page_function_present``
# were retired in v5.8.3-rc.5 — both are subsumed by the assertions below.
# ``_extract_render_settings_page_body`` returns empty when the file is
# missing or the function body cannot be located, and the backtick /
# bracket-backtick tests assert ``body`` is truthy with an explicit
# message that points at the same root cause. Keeping the two tiny
# existence tests added cost without adding signal.


def test_render_settings_page_backtick_balance():
    """renderSettingsPage() body has balanced (even) unescaped backticks.

    The function uses one large outer template literal. All inner backtick
    occurrences must be escaped (\\`) OR appear in matched pairs.

    An odd count inside the function body means an unmatched backtick that
    terminates the outer template and causes a ReferenceError at runtime.

    The count is performed on the isolated function body only (not the whole
    file) so a stray backtick elsewhere cannot mask a real imbalance here.
    """
    content = SETTINGS_UI_JS.read_text(encoding="utf-8")
    body = _extract_render_settings_page_body(content)
    assert body, "Could not extract renderSettingsPage() body — check _extract_render_settings_page_body"
    count = _count_unescaped_backticks(body)
    assert count % 2 == 0, (
        f"Odd number of unescaped backticks in renderSettingsPage() body ({count}). "
        "This likely means a bare backtick inside the template literal will "
        "terminate the string and produce a ReferenceError at runtime. "
        "Either escape it (\\`) or ensure it forms a matched pair."
    )


def test_render_settings_page_no_bare_bracket_backtick():
    """No unescaped `[...` pattern inside renderSettingsPage() body.

    The v4.39.2 regression: description text contained `[main, light, light]`
    with literal backtick delimiters, which JS interpreted as terminating the
    outer template and evaluating [main] as an expression.

    Checks the isolated function body so other functions cannot create
    false negatives or false positives for this specific guard.
    """
    content = SETTINGS_UI_JS.read_text(encoding="utf-8")
    body = _extract_render_settings_page_body(content)
    assert body, "Could not extract renderSettingsPage() body"
    # Backtick not preceded by backslash, followed by [
    pattern = re.compile(r'(?<!\\)`\[')
    matches = list(pattern.finditer(body))
    assert len(matches) == 0, (
        f"Found {len(matches)} unescaped backtick-bracket pattern(s) in renderSettingsPage(). "
        "This is the v4.39.2 regression pattern. Lines in body: "
        + str([body[:m.start()].count('\n') + 1 for m in matches])
    )
