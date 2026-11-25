"""Settings as a section with pages (docs/DESIGN.md "Settings", "Page header", §10 "A section with pages").

The open page's name and its one Back stand in the section's header (settings.js `paintLead`), never inside the page:
``lead_name`` is the name, ``header_back`` the Back, which says where it leads in its accessible name and shows only
where it leads to a level not on screen. Ouroboros's card stands on the list in one column; from 40rem the list stands
beside a page that is always open, and the card is the pane the list's Ouroboros row opens. ``show_card`` reaches it
the way the owner does in each layout."""
from __future__ import annotations

SETTINGS = "#page-settings"
HEADER_BACK = f"{SETTINGS} > .app-page-header [data-shell-back]"
LEAD_NAME = f"{SETTINGS} [data-settings-lead-name]"
CARD = f"{SETTINGS} [data-settings-card]"
TWO_PANES = """async () => (await import('/static/modules/layout.js'))
    .sectionHoldsTwoPanes(document.getElementById('page-settings'))"""


# Extensions (extensions.js) opens its pages with the same levels: its header holds the same name and one Back.
EXTENSIONS_BACK = "#page-skills > .app-page-header [data-shell-back]"
EXTENSIONS_NAME = "#page-skills [data-extensions-lead-name]"
# Extensions has no place in the sidebar's rail (owner 24.11, «как задумано»): Settings › Connections holds its door.
EXTENSIONS_DOOR = f'{SETTINGS} [data-settings-open="page:skills"]'


def press_nav(page, name: str) -> None:
    """Press the sidebar row of `name` in place, as a script click (a phone's closed drawer needs no opening).
    Extensions has no row (owner 24.11): Settings' row, then its door in Settings' list, pressed the same way."""
    if name == 'skills':
        press_nav(page, 'settings')
        page.wait_for_function("() => document.getElementById('page-settings')?.classList.contains('active')")
        page.locator(EXTENSIONS_DOOR).first.evaluate('row => row.click()')
        return
    page.locator(f'#primary-sidebar [data-nav-page="{name}"]').first.evaluate('row => row.click()')


def open_extensions(page, *, timeout: float = 10_000):
    """Open Extensions as the owner does: Settings from the rail (in the drawer where the shell is one column), the
    list back in view where one column shows a page, then Settings' door to Extensions. Returns its page."""
    toggle = page.locator('[data-mobile-nav-toggle]:visible').first
    if toggle.count() and toggle.get_attribute('aria-expanded') != 'true':
        toggle.click()
    page.locator('#primary-sidebar [data-nav-page="settings"]').click(timeout=timeout)
    page.locator(SETTINGS).wait_for(state='visible', timeout=timeout)
    door = page.locator(EXTENSIONS_DOOR).first
    for _ in range(4):
        if door.is_visible():
            break
        back = page.locator(f'{HEADER_BACK}:visible').first
        if not back.count():
            break
        back.click()
    door.click(timeout=timeout)
    page.locator('#page-skills').wait_for(state='visible', timeout=timeout)
    return page.locator('#page-skills')


def header_back(page):
    """The shown section's Back: Settings', or Extensions' where a page of Extensions is shown."""
    return page.locator(f"{SETTINGS}.active > .app-page-header [data-shell-back], #page-skills.active > .app-page-header [data-shell-back]")


def lead_name(page) -> str:
    """The open page's name in the shown section's header ('' over the list or Ouroboros's pane)."""
    return page.locator(f"{SETTINGS}.active [data-settings-lead-name], #page-skills.active [data-extensions-lead-name]").inner_text().strip()


def two_panes(page) -> bool:
    return bool(page.evaluate(TWO_PANES))


def show_card(page, *, timeout: float = 10_000):
    """Show Ouroboros's card and return it: in one column the list holds it; beside a page, its Ouroboros row."""
    page.wait_for_function("() => Boolean(window.__ouroSettings)", timeout=timeout)
    page.evaluate("() => window.__ouroSettings.open('')")
    if two_panes(page):
        row = page.locator(f"{SETTINGS} [data-settings-overview-row]")
        row.wait_for(state="visible", timeout=timeout)
        row.click()
    card = page.locator(CARD)
    card.wait_for(state="visible", timeout=timeout)
    return card


def show_page(page, name: str, *, timeout: float = 10_000):
    """Show the app page `name` as its sidebar row does, pressed in place (a phone's closed drawer needs no opening).
    The Settings list reads its answers when it comes on screen (settings.js look): another page, then Settings,
    brings it back on screen."""
    press_nav(page, name)
    page.wait_for_function("name => document.getElementById(`page-${name}`)?.classList.contains('active')", arg=name,
                           timeout=timeout)
