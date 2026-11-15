"""Settings as a section with pages (docs/DESIGN.md "Settings", "Page header", §10 "A section with pages").

The open page's name and its one Back stand in the section's header (settings.js `paintLead`), never inside the page:
``lead_name`` is the name, ``header_back`` the Back, which says where it leads in its accessible name and shows only
where it leads to a level not on screen. Ouroboros's card stands on the list in one column; from 40rem the list stands
beside a page that is always open, and the card is the pane the list's Ouroboros row opens. ``show_card`` reaches it
the way the owner does in each layout."""
from __future__ import annotations

SETTINGS = "#page-settings"
HEADER_BACK = f"{SETTINGS} > .app-page-header [data-settings-back]"
LEAD_NAME = f"{SETTINGS} [data-settings-lead-name]"
CARD = f"{SETTINGS} [data-settings-card]"
TWO_PANES = """async () => (await import('/static/modules/layout.js'))
    .sectionHoldsTwoPanes(document.getElementById('page-settings'))"""


def header_back(page):
    return page.locator(HEADER_BACK)


def lead_name(page) -> str:
    """The open page's name in the header ('' over the list or Ouroboros's pane)."""
    return page.locator(LEAD_NAME).inner_text().strip()


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
    page.locator(f'#primary-sidebar [data-nav-page="{name}"]').first.evaluate("row => row.click()")
    page.wait_for_function("name => document.getElementById(`page-${name}`)?.classList.contains('active')", arg=name,
                           timeout=timeout)
