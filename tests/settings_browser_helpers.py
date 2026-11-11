"""Settings in a browser test (docs/DESIGN.md "Settings"): open a path, read a home row's answer, commit a row and
wait for its result line, open an entry's editor and press its Save.

Settings has no URL: a path is opened through the shell's opener (the loopback debug hook ``window.__ouroSettings``,
the same function app.js's ``openSettings`` calls), so a test reaches a page the way any caller does. The hooks are
the shell's (settings.js: ``[data-settings-home]``, ``[data-settings-page]``, ``[data-settings-back]``), the home's
(settings_home.js: ``[data-settings-open]``, ``[data-settings-answer]``, the card ``[data-settings-card]`` with its
rows ``[data-needs]``, the search field ``#settings-search`` and a result ``[data-settings-result]``) and the row
helper's (settings_controls.js:
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


# Settings has no tab strip: what it shows is its home, the pages a home row opens or a module registered, and, where a
# page lists entries, the editor of one. A skill's page needs a skill: the caller names one that registered a form.
SETTINGS_PATHS = """async (skill) => {
    const { settingsPagePrefixes } = await import('/static/modules/settings_pages.js');
    const { HOME_ROWS } = await import('/static/modules/settings_rows.js');
    const home = HOME_ROWS.map((row) => row.to).filter((to) => !/^(dashboard|page|main):/.test(to));
    return [...new Set([...home, ...settingsPagePrefixes()])].filter((path) => path !== 'skill' || skill)
        .map((path) => (path === 'skill' ? `skill/${skill}` : path));
}"""
SETTLED = "() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))"


def walk_settings(page, measure, *, skill: str = "", settle=None):
    """Show every Settings view in turn and call ``measure(name)`` at each: the home, every page, and one editor of
    a page that lists entries. Ends at the home, where it started, and returns the paths it walked."""
    settle = settle or (lambda: page.evaluate(SETTLED))
    open_settings(page, "")
    page.locator(f"{HOME} [data-settings-card]").wait_for(state="visible")
    settle()
    measure("home")
    paths = page.evaluate(SETTINGS_PATHS, skill)
    for path in paths:
        level = open_settings(page, path)
        if path.startswith("skill/"):
            level.locator("[data-extension-settings-form]").wait_for(state="visible")
        settle()
        measure(path)
        head = level.locator("[data-entry-id]").first
        if head.count():
            open_entry(page, head.get_attribute("data-entry-id"))
            settle()
            measure(top_path(page))
    open_settings(page, "")
    settle()
    return paths


def top_path(page) -> str:
    """The path Settings shows now ('' is the home): the visible level, else the home."""
    return page.evaluate("""() => {
        const levels = [...document.querySelectorAll('#page-settings [data-settings-page]')].filter((e) => !e.hidden);
        return levels.length ? levels.at(-1).dataset.settingsPage : '';
    }""")


def go_back(page):
    """Press the open level's first row, Back, and return the path shown after it. In the wide layout a page opened
    from the home has no Back (the home stands beside it): there the home is opened, as its row in the rail would."""
    before = top_path(page)
    back = settings_page(page, before).locator("[data-settings-back]")
    if back.is_visible():
        back.click()
    else:
        page.evaluate("() => window.__ouroSettings.open('')")
    page.wait_for_function("before => { const levels = [...document.querySelectorAll('#page-settings "
                           "[data-settings-page]')]; return !levels.some((e) => e.dataset.settingsPage === before); }",
                           arg=before)
    return top_path(page)


def home_answer(page, target: str) -> str:
    """The answer the home shows under the row that opens `target` ('' while it shows none)."""
    row = page.locator(f'{HOME} [data-settings-open="{target}"]')
    row.wait_for(state="attached")
    return row.locator("[data-settings-answer]").inner_text().strip()


def needs(page) -> list[str]:
    """What Ouroboros's card lists now, as ``kind:id`` of each row it shows or keeps behind "Show all"."""
    return page.locator(f"{HOME} [data-settings-card] [data-needs]").evaluate_all(
        "rows => rows.map((row) => `${row.dataset.needs}:${row.dataset.needId}`)")


def search_settings(page, words: str) -> list[str]:
    """Type `words` into the home's search and return the ids of the results it shows, in order."""
    page.locator("#settings-search").fill(words)
    page.wait_for_function("words => (document.querySelector('#page-settings [data-settings-ask]') !== null) === Boolean(words.trim())",
                           arg=words)
    return page.locator(f"{HOME} [data-settings-result]").evaluate_all("rows => rows.map((row) => row.dataset.settingsResult)")


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
