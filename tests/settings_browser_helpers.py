"""Settings in a browser test (docs/DESIGN.md "Settings"): open a path, read a home row's answer, commit a row and
wait for its result line, open an entry's editor and press its Save.

Settings has no URL: a path is opened through the shell's opener (the loopback debug hook ``window.__ouroSettings``,
the same function app.js's ``openSettings`` calls), so a test reaches a page the way any caller does. The hooks are
the shell's (settings.js: ``[data-settings-home]``, ``[data-settings-page]``, ``[data-settings-back]``), the home's
(settings_home.js: ``[data-settings-open]``, ``[data-settings-answer]``) and the row helper's (settings_controls.js:
``[data-setting-row]``, ``[data-setting-control]``, ``[data-setting-result]``; an editor's ``[data-entry-id]``,
``[data-editor-save]``, ``[data-editor-result]``)."""
from __future__ import annotations

SETTINGS = "#page-settings"
HOME = f"{SETTINGS} [data-settings-home]"


def settings_page(page, path: str = ""):
    """The locator of the level showing `path` ('' is the home)."""
    return page.locator(HOME if not path else f'{SETTINGS} [data-settings-page="{path}"]')


def open_settings(page, path: str = "", row: str = "", *, timeout: float = 10_000):
    """Show Settings at `path`, revealing `row` when given, and wait until that level is visible."""
    page.wait_for_function("() => Boolean(window.__ouroSettings)", timeout=timeout)
    opened = page.evaluate("([path, row]) => window.__ouroSettings.open(path, { row })", [path, row])
    assert opened is not False, f"Settings refused to open {path!r}"
    level = settings_page(page, path)
    level.wait_for(state="visible", timeout=timeout)
    return level


def top_path(page) -> str:
    """The path Settings shows now ('' is the home): the visible level, else the home."""
    return page.evaluate("""() => {
        const levels = [...document.querySelectorAll('#page-settings [data-settings-page]')].filter((e) => !e.hidden);
        return levels.length ? levels.at(-1).dataset.settingsPage : '';
    }""")


def go_back(page):
    """Press the open level's first row, Back, and return the path shown after it."""
    before = top_path(page)
    settings_page(page, before).locator("[data-settings-back]").click()
    page.wait_for_function("before => { const levels = [...document.querySelectorAll('#page-settings "
                           "[data-settings-page]')]; return !levels.some((e) => e.dataset.settingsPage === before); }",
                           arg=before)
    return top_path(page)


def home_answer(page, target: str) -> str:
    """The answer the home shows under the row that opens `target` ('' while it shows none)."""
    row = page.locator(f'{HOME} [data-settings-open="{target}"]')
    row.wait_for(state="attached")
    return row.locator("[data-settings-answer]").inner_text().strip()


def setting_row(page, row_id: str):
    """The row `row_id` on the level Settings shows now."""
    return settings_page(page, top_path(page)).locator(f'[data-setting-row="{row_id}"]')


def commit_row(page, row_id: str, value=None, *, timeout: float = 10_000) -> str:
    """Commit `value` to a row the way the owner does and return its result line once the write settled.

    A switch is set to ``bool(value)`` (toggled when `value` is None); a select picks `value`; a strip presses its
    option; a typed field is filled and left, which is its commit (settings_controls.js commits on ``change``)."""
    row = setting_row(page, row_id)
    row.wait_for(state="visible", timeout=timeout)
    result = row.locator("[data-setting-result]")
    before = result.inner_text()
    kind = row.get_attribute("data-kind") or ""
    control = row.locator("[data-setting-control]").first
    if kind == "switch":
        if value is None or control.is_checked() != bool(value):
            control.click()
    elif row.locator(f'[data-setting-value="{value}"]').count():
        row.locator(f'[data-setting-value="{value}"]').click()
    elif kind == "select":
        control.select_option(str(value))
    else:
        control.fill(str(value))
        control.evaluate("element => element.blur()")
    page.wait_for_function("""([id, before]) => {
        const rows = [...document.querySelectorAll(`#page-settings [data-setting-row="${CSS.escape(id)}"]`)];
        const row = rows.find((node) => node.getClientRects().length);
        const line = row?.querySelector('[data-setting-result]');
        return Boolean(line && !row.querySelector('[aria-busy="true"]') && line.textContent.trim()
            && line.textContent !== before);
    }""", arg=[row_id, before], timeout=timeout)
    return result.inner_text().strip()


def open_entry(page, entry_id: str, *, timeout: float = 10_000):
    """Press an entity list's head and return the editor level it opened."""
    before = top_path(page)
    settings_page(page, before).locator(f'[data-entry-id="{entry_id}"]').first.click()
    page.wait_for_function("before => { const levels = [...document.querySelectorAll('#page-settings "
                           "[data-settings-page]')].filter((e) => !e.hidden); return levels.length && "
                           "levels.at(-1).dataset.settingsPage !== before; }", arg=before, timeout=timeout)
    return settings_page(page, top_path(page))


def press_editor_save(page, *, timeout: float = 10_000) -> str:
    """Press the open editor's Save and return its result line once the write settled."""
    editor = settings_page(page, top_path(page))
    result = editor.locator("[data-editor-result]").first
    before = result.inner_text() if result.count() else ""
    editor.locator("[data-editor-save]").first.click()
    page.wait_for_function("""before => {
        const levels = [...document.querySelectorAll('#page-settings [data-settings-page]')].filter((e) => !e.hidden);
        const editor = levels.at(-1);
        const line = editor?.querySelector('[data-editor-result]');
        const save = editor?.querySelector('[data-editor-save]');
        return Boolean(line && line.textContent.trim() && line.textContent !== before
            && save?.getAttribute('aria-busy') !== 'true');
    }""", arg=before, timeout=timeout)
    return result.inner_text().strip()
