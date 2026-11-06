"""A task-control menu is dismissed by a scroll that can have moved its trigger, and by no other.

The room's work list opens its row menu as a portal over a floating sheet while the chat feed behind the sheet re-anchors on
its own (a late history page, a text refit). A scroll of that feed cannot move the trigger, yet it dismissed the menu within
a millisecond of opening it (CI: test_real_row_hurry_wrap_up_stop_and_restart_with_page_open, the menu never visible). A
scroller that holds the trigger, and the page itself, still dismiss (owner decision 1D, test_s3_task_control_browser.py:
"only scroll that can stale anchor geometry invokes 1D close").

The real module in the real SPA page (the static-server/API fixture of test_subscription_setup_browser), in both engines;
no runtime. The menus open from real pointer clicks, the container scroll is a real wheel, and every state is screenshotted
(OUROBOROS_UI_EVIDENCE_DIR). The first check is the one that fails without the fix; the other two keep what must still close.
"""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

# A scroller behind a floating sheet (the chat feed), the sheet holding a trigger, and a scroller that holds its trigger.
STAGE = """async () => {
    const { openTaskControlMenu } = await import('/modules/task_control_menu.js');
    const box = (id, css, html) => {
        const node = document.createElement('div');
        node.id = id; node.style.cssText = css; node.innerHTML = html;
        document.body.append(node);
        return node;
    };
    const feed = box('ob-feed', 'position:fixed;inset:0;overflow:auto;z-index:1', '<div style="height:4000px"></div>');
    box('ob-sheet', 'position:fixed;top:60px;left:60px;width:260px;height:60px;z-index:60;background:#222',
        '<button id="ob-trigger" style="margin:12px">trigger</button>');
    const holder = box('ob-holder', 'position:fixed;top:200px;left:60px;width:260px;height:80px;overflow:auto;z-index:60;'
        + 'background:#222', '<div style="height:400px;padding:12px"><button id="ob-inner">inner</button></div>');
    for (const id of ['ob-trigger', 'ob-inner']) {
        const button = document.getElementById(id);
        button.addEventListener('click', () => openTaskControlMenu(button, { actions: ['hurry', 'stop_now'], onAction() {} }));
    }
    window.__ob = { scrolls: { feed: 0, holder: 0 }, menus: () => document.querySelectorAll('body > .task-control-menu').length };
    feed.addEventListener('scroll', () => { window.__ob.scrolls.feed += 1; });
    holder.addEventListener('scroll', () => { window.__ob.scrolls.holder += 1; });
}"""
# Two frames and a turn after a scroll: the scroll event it queued has been dispatched.
SETTLED = "() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(done, 60))))"
MENU = "body > .task-control-menu"


@pytest.fixture(params=["chromium", "webkit"])
def staged(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    page = ui["page"]
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")
    page.evaluate(STAGE)
    return page, request.param


def _shot(page, engine, name):
    setup_browser.capture(page, f"menu-scroll-{engine}-{name}")


def test_a_scroll_that_cannot_have_moved_the_trigger_leaves_its_menu_open(staged):
    page, engine = staged
    page.locator("#ob-trigger").click()
    page.locator(MENU).wait_for(state="visible")
    _shot(page, engine, "1-open")
    # The feed behind the sheet re-anchors (it holds no trigger): the menu stays, and so does its trigger's state.
    page.evaluate("() => { document.getElementById('ob-feed').scrollTop = 240; }")
    page.evaluate(SETTLED)
    _shot(page, engine, "2-after-the-feed-scrolled")
    assert page.evaluate("() => window.__ob.scrolls.feed") >= 1, "the scroll under test never fired"
    assert page.evaluate("() => window.__ob.menus()") == 1, "a scroll that cannot move the trigger dismissed its menu"
    assert page.locator("#ob-trigger").get_attribute("aria-expanded") == "true"
    assert page.locator(MENU).is_visible()


def test_a_scroll_of_the_container_holding_the_trigger_still_dismisses_the_menu(staged):
    page, engine = staged
    page.locator("#ob-inner").click()
    page.locator(MENU).wait_for(state="visible")
    _shot(page, engine, "3-open-inside-a-scroller")
    box = page.locator("#ob-holder").bounding_box()
    # A wheel over the holder's top-right corner (clear of the menu, which opens beneath its trigger at the left).
    page.mouse.move(box["x"] + box["width"] - 10, box["y"] + 10)
    page.mouse.wheel(0, 60)
    page.locator(MENU).wait_for(state="detached")
    _shot(page, engine, "4-after-the-holder-scrolled")
    assert page.evaluate("() => window.__ob.scrolls.holder") >= 1, "the scroll under test never fired"
    assert page.locator("#ob-inner").get_attribute("aria-expanded") == "false"


def test_a_page_scroll_still_dismisses_the_menu(staged):
    page, engine = staged
    page.locator("#ob-trigger").click()
    page.locator(MENU).wait_for(state="visible")
    page.evaluate("() => document.dispatchEvent(new Event('scroll'))")  # the document is the event's target
    page.locator(MENU).wait_for(state="detached")
    _shot(page, engine, "5-after-a-page-scroll")
    assert page.locator("#ob-trigger").get_attribute("aria-expanded") == "false"
