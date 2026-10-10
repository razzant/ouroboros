"""The empty-Main greeting through the real server, its hidden preference and chat history.

The greeting is host-owned empty-state copy, never a chat bubble, a history row or
a model reply. It appears only in Main, only after a successful recent history read
whose own window reports complete coverage while the feed holds no content, and its
copy comes from the install-wide ``welcome`` UI preference (default, hidden or
custom). The preference is hidden: Settings has no control for it; the owner edits
``state/ui_preferences.json`` or POSTs it to /api/ui/preferences, and Main reads it
each time it connects (docs/DESIGN.md "Chat authorship and System rows").

The late-history case holds Main's own history reads in page JavaScript and settles
each one from the REAL server response, rewritten only where the scenario says so,
so the pager, the renderer and the gate all consume production payloads. Nothing is
timed: every wait is a DOM condition or a held request.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytest_plugins = ("tests.test_ui_smoke_playwright",)

DEFAULT = "Ouroboros has awakened"
CUSTOM = "Hello <b>x</b>\nSecond line & more"
FILE_CUSTOM = "Доброе утро — the file says so"
RECONNECT_CUSTOM = "Read again on reconnect"
HYDRATED = '#chat-messages[data-history-hydrated="true"]'
WELCOME = '#chat-messages > .chat-empty-welcome[data-welcome-state="ready"]'
ANY_WELCOME = ".chat-empty-welcome"
SETTLE_FRAMES = "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
# Screenshots wait for running CSS transitions (tab pills, the narrow drawer) to end.
TRANSITIONS_DONE = """() => Promise.all(document.getAnimations()
    .filter(animation => typeof CSSTransition === 'function' && animation instanceof CSSTransition)
    .map(animation => animation.finished.catch(() => null)))
    .then(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))))"""
DESKTOP = {"width": 1280, "height": 860}
NARROW = {"width": 390, "height": 844}

# Records, for every greeting Main mounts, whether its history read was already
# stamped at that moment: the empty state never precedes a landed read.
_WATCH_WELCOME = """() => {
    window.__welcomeMounts = [];
    new MutationObserver(() => {
        const node = document.querySelector('#chat-messages > .chat-empty-welcome');
        if (node && !node.__seen) {
            node.__seen = true;
            window.__welcomeMounts.push(node.parentElement.dataset.historyHydrated || '');
        }
    }).observe(document, {subtree: true, childList: true});
}"""

# Main's recent-history reads wait here until the test settles them.
_HOLD_MAIN_HISTORY = r"""() => {
    const nativeFetch = window.fetch.bind(window);
    window.__historyHeld = [];
    window.fetch = (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input?.url || '', location.href);
        if (url.pathname !== '/api/chat/history' || url.searchParams.has('chat_id')) return nativeFetch(input, init);
        return new Promise((resolve, reject) => window.__historyHeld.push({input, init, resolve, reject}));
    };
    const json = (body, status) => new Response(JSON.stringify(body),
        {status, headers: {'Content-Type': 'application/json'}});
    window.__settleHistory = async (mode) => {
        const held = window.__historyHeld.shift();
        if (!held) throw new Error('no held history read');
        if (mode === 'error') return held.resolve(json({error: 'Synthetic history outage'}, 500));
        const response = await nativeFetch(held.input, held.init);
        const body = await response.json();
        if (mode === 'partial') body.window = {complete: false, truncated_by: ['quota']};
        if (mode === 'message') body.messages = [...body.messages,
            {role: 'assistant', text: 'Late history message.', ts: new Date().toISOString()}];
        held.resolve(json(body, response.status));
    };
}"""


def _settle(page):
    page.evaluate(SETTLE_FRAMES)


def _welcome_text(page):
    return page.locator(WELCOME).evaluate("node => node.lastElementChild.textContent")


def _stored_welcome(page, url):
    return page.request.get(url + "/api/ui/preferences").json()["welcome"]


def _post_preferences(page, url, body):
    return page.request.post(url + "/api/ui/preferences", data=json.dumps(body),
                             headers={"Content-Type": "application/json"})


def _edit_welcome(prefs_file, welcome):
    """The documented hand edit: replace the ``welcome`` key and keep every other one."""
    prefs = json.loads(prefs_file.read_text(encoding="utf-8")) if prefs_file.exists() else {}
    prefs["welcome"] = welcome
    prefs_file.parent.mkdir(parents=True, exist_ok=True)
    prefs_file.write_text(json.dumps(prefs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _open_main(page, url):
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.locator('[data-nav-page="chat"]').click()
    page.wait_for_selector(HYDRATED, state="attached", timeout=30_000)
    _settle(page)


def _shoot_both_widths(page, evidence, name, target=None):
    """Desktop then narrow, the greeting (when present) kept inside the narrow viewport."""
    page.evaluate(TRANSITIONS_DONE)
    page.screenshot(path=str(evidence / f"{name}-desktop.png"))
    page.set_viewport_size(NARROW)
    page.evaluate(TRANSITIONS_DONE)
    if target is not None:
        box = page.locator(target).bounding_box()
        assert box and box["x"] >= 0 and box["x"] + box["width"] <= NARROW["width"]
    page.screenshot(path=str(evidence / f"{name}-narrow.png"))
    page.set_viewport_size(DESKTOP)
    _settle(page)


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_empty_main_greeting_contract(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))) / f"welcome-{engine}"
    evidence.mkdir(parents=True, exist_ok=True)
    create_project(data_dir, "welcome-room", name="Welcome room")
    prefs_file = data_dir / "state" / "ui_preferences.json"

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport=DESKTOP)
            page.add_init_script(f"({_WATCH_WELCOME})()")
            settings_posts = []
            page.on("request", lambda request: settings_posts.append(request.url)
                    if request.method == "POST" and request.url.endswith("/api/settings") else None)

            # (a) A fresh install's empty Main shows the built-in sentence as host copy,
            # only once its history read has landed, and never as a bubble or a row.
            _open_main(page, url)
            page.wait_for_selector(WELCOME)
            assert _welcome_text(page) == DEFAULT
            assert page.evaluate("() => window.__welcomeMounts") == ["true"]
            assert page.locator("#chat-messages .chat-bubble:not(.typing-bubble)").count() == 0
            history = page.request.get(url + "/api/chat/history").json()
            assert history["messages"] == [] and history["window"]["complete"] is True
            _shoot_both_widths(page, evidence, "main-default", WELCOME)

            # ... and a Project room never shows it.
            page.locator('.nav-project-row[data-project-id="welcome-room"]').click()
            page.wait_for_selector("#project-panel:not([hidden])")
            page.wait_for_selector('#project-panel [data-history-hydrated="true"]', state="attached")
            _settle(page)
            assert page.locator(f"#project-panel {ANY_WELCOME}").count() == 0
            page.click("#project-panel-close")

            # (b) The preference is hidden: Settings -> Appearance has no greeting control.
            page.locator('[data-nav-page="settings"]').click()
            page.wait_for_selector("#page-settings.active")
            page.locator('[data-settings-tab="appearance"]').click()
            page.wait_for_selector('[data-settings-panel="appearance"].active')
            assert page.locator('[data-settings-tab="appearance"]').get_attribute("aria-selected") == "true"
            assert page.locator("#page-settings [data-welcome-settings], #page-settings [data-welcome-mode],"
                                " #page-settings [data-welcome-text], #page-settings [data-welcome-save]").count() == 0
            appearance = page.locator('[data-settings-panel="appearance"]').inner_text().lower()
            assert "theme" in appearance
            assert "greeting" not in appearance and "welcome" not in appearance
            _shoot_both_widths(page, evidence, "settings-appearance")
            page.locator('[data-nav-page="chat"]').click()

            # (c) The existing API validates and merges a custom sentence. Nothing pushes
            # it to an open page; Main reads it when it next connects, as text, not markup.
            response = _post_preferences(page, url, {"welcome": {"mode": "custom", "text": CUSTOM}})
            assert response.status == 200, response.text()
            assert _stored_welcome(page, url) == {"mode": "custom", "text": CUSTOM}
            _settle(page)
            assert _welcome_text(page) == DEFAULT
            _open_main(page, url)
            page.wait_for_selector(WELCOME)
            assert _welcome_text(page) == CUSTOM
            assert page.locator(f"{WELCOME} b").count() == 0
            assert page.locator(f"{WELCOME} p").evaluate("node => getComputedStyle(node).whiteSpace") == "pre-wrap"
            _shoot_both_widths(page, evidence, "main-custom", WELCOME)

            # (d) The endpoint refuses what the documented contract forbids and writes nothing.
            stored = prefs_file.read_bytes()
            for invalid in ({"mode": "custom", "text": "   "}, {"mode": "custom", "text": "x" * 501},
                            {"mode": "loud", "text": ""}, {"mode": "custom"},
                            {"mode": "hidden", "text": "", "extra": True}):
                response = _post_preferences(page, url, {"welcome": invalid})
                assert response.status == 400, response.text()
            assert prefs_file.read_bytes() == stored

            # (e) Editing the file by hand, keeping its other keys, and reloading hides it.
            assert _post_preferences(page, url, {"nested_subagents_expanded": True}).status == 200
            _edit_welcome(prefs_file, {"mode": "hidden", "text": CUSTOM})
            _open_main(page, url)
            assert page.locator(ANY_WELCOME).count() == 0
            assert page.evaluate("() => window.__welcomeMounts") == []
            prefs = page.request.get(url + "/api/ui/preferences").json()
            assert prefs["welcome"] == {"mode": "hidden", "text": CUSTOM}
            assert prefs["nested_subagents_expanded"] is True
            page.evaluate(TRANSITIONS_DONE)
            page.screenshot(path=str(evidence / "main-hidden-desktop.png"))

            # (f) A custom sentence and (g) default again, each from the file after a reload.
            _edit_welcome(prefs_file, {"mode": "custom", "text": FILE_CUSTOM})
            _open_main(page, url)
            page.wait_for_selector(WELCOME)
            assert _welcome_text(page) == FILE_CUSTOM
            _edit_welcome(prefs_file, {"mode": "default", "text": ""})
            _open_main(page, url)
            page.wait_for_selector(WELCOME)
            assert _welcome_text(page) == DEFAULT

            # (h) A hand edit the endpoint would refuse reads as the default and keeps the rest.
            _edit_welcome(prefs_file, {"mode": "Hidden", "text": ""})
            _open_main(page, url)
            page.wait_for_selector(WELCOME)
            assert _welcome_text(page) == DEFAULT
            assert page.request.get(url + "/api/ui/preferences").json()["nested_subagents_expanded"] is True
            _edit_welcome(prefs_file, {"mode": "default", "text": ""})

            # (i) Only a successful read that reports complete coverage confirms an empty
            # Main. Every read withdraws the greeting while it is in flight, so an empty
            # feed shows it loading; a failed read then shows its failure, a partial read
            # shows nothing, a reconnect reads the preference again, and a late read that
            # brings a message removes the greeting.
            late = browser.new_page(viewport=DESKTOP)
            for script in (_CAPTURE_TEST_SOCKET, _HOLD_MAIN_HISTORY, _WATCH_WELCOME):
                late.add_init_script(f"({script})()")
            held = "() => window.__historyHeld.length > 0"
            # The failed read's own chrome: Retry on the feed edge, its note in Main's header.
            failure = """() => document.querySelector('#chat-messages .chat-load-older-btn')
                ?.textContent === 'Retry loading messages'
                && document.querySelector('#page-chat .chat-page-header .chat-history-status')
                ?.textContent.includes('could not be loaded') === true"""
            # A held read over a feed of chrome only: its loading state, never an old greeting.
            loading = """() => window.__historyHeld.length > 0
                && document.querySelector('#chat-messages .chat-load-older')?.getAttribute('aria-busy') === 'true'
                && document.querySelector('#chat-messages .chat-load-older-btn')
                ?.textContent.includes('Loading saved history') === true"""
            notices = "document.querySelectorAll('.chat-bubble[data-system-type=\"reconnect\"]').length"

            def settle(mode):
                late.wait_for_function(held)
                late.evaluate(f"() => window.__settleHistory('{mode}')")

            def held_empty_read():
                late.wait_for_function(loading)
                assert late.locator(ANY_WELCOME).count() == 0

            def wait_notices(count):
                late.wait_for_function(f"() => {notices} >= {count}")
                _settle(late)

            def retry():
                # Retry reads again unless another hydration trigger already does; one
                # page task decides, and every Main read in flight is a held one.
                late.evaluate("""() => { if (!window.__historyHeld.length)
                    document.querySelector('#chat-messages .chat-load-older-btn')?.click(); }""")

            def reconnect():
                # A reconnect with no established build SHA deliberately reloads the page.
                late.wait_for_function("() => typeof window.__ouroWs?._lastSha === 'string'"
                                       " && window.__ouroWs._lastSha.length > 0")
                late.evaluate("() => window.__testSockets.at(-1).close()")

            late.goto(url, wait_until="domcontentloaded", timeout=30_000)
            settle("error")
            late.wait_for_function(f"() => window.__historyHeld.length > 0 || ({failure})()")
            assert late.locator(ANY_WELCOME).count() == 0
            assert late.locator(HYDRATED).count() == 0
            retry()
            settle("complete")
            late.wait_for_selector(WELCOME)
            assert _welcome_text(late) == DEFAULT
            reconnect()
            held_empty_read()
            settle("error")
            late.wait_for_function(failure)
            assert late.locator(ANY_WELCOME).count() == 0
            retry()
            settle("partial")
            wait_notices(1)
            assert late.locator(ANY_WELCOME).count() == 0
            _edit_welcome(prefs_file, {"mode": "custom", "text": RECONNECT_CUSTOM})
            reconnect()
            held_empty_read()
            settle("complete")
            late.wait_for_selector(WELCOME)
            assert _welcome_text(late) == RECONNECT_CUSTOM
            # The ephemeral reconnect notice is chrome, not conversation.
            wait_notices(2)
            assert late.locator(WELCOME).count() == 1
            # A later reconnect over the greeting and its notices withdraws it while
            # its read is held, whatever that read then answers.
            reconnect()
            held_empty_read()
            late.evaluate(TRANSITIONS_DONE)
            late.screenshot(path=str(evidence / "main-reconnect-held-desktop.png"))
            settle("partial")
            wait_notices(3)
            assert late.locator(ANY_WELCOME).count() == 0
            reconnect()
            held_empty_read()
            settle("complete")
            late.wait_for_selector(WELCOME)
            wait_notices(4)
            reconnect()
            held_empty_read()
            settle("error")
            late.wait_for_function(failure)
            assert late.locator(ANY_WELCOME).count() == 0
            retry()
            settle("complete")
            late.wait_for_selector(WELCOME)
            assert _welcome_text(late) == RECONNECT_CUSTOM
            wait_notices(5)
            reconnect()
            held_empty_read()
            settle("message")
            late.locator(".chat-bubble", has_text="Late history message.").wait_for()
            late.wait_for_selector(ANY_WELCOME, state="detached")
            wait_notices(6)
            # A painted transcript keeps its messages under a held read, without loading chrome.
            reconnect()
            late.wait_for_function(held)
            assert late.locator(".chat-bubble", has_text="Late history message.").count() >= 1
            assert late.locator("#chat-messages .chat-load-older[aria-busy]").count() == 0
            assert late.locator(ANY_WELCOME).count() == 0
            settle("message")
            wait_notices(7)
            assert late.evaluate("() => window.__welcomeMounts") == ["true"] * 4
            late.close()
            _edit_welcome(prefs_file, {"mode": "default", "text": ""})

            # (j) The owner's first real message replaces the empty state, live and
            # in the durable history, which never carries the greeting itself.
            _open_main(page, url)
            page.wait_for_selector(WELCOME)
            page.fill("#chat-input", "First owner message")
            page.click("#chat-send")
            page.locator(".chat-bubble.user", has_text="First owner message").wait_for()
            page.wait_for_selector(ANY_WELCOME, state="detached")
            _open_main(page, url)
            page.locator(".chat-bubble.user", has_text="First owner message").wait_for()
            assert page.locator(ANY_WELCOME).count() == 0
            body = page.request.get(url + "/api/chat/history").text()
            for copy in (DEFAULT, "Second line", FILE_CUSTOM, RECONNECT_CUSTOM):
                assert copy not in body
            assert settings_posts == []
        finally:
            browser.close()
