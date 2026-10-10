"""Recovery through real history routes, retained files and project bindings."""
from __future__ import annotations

import os

import pytest

from tests.test_chat_history_paging_browser import (
    MAIN, _human, _idle, _open, _open_project, _reads, _screenshot, _step,
    _to_beginning, _write,
)
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _assert_unique_rows(page, feed):
    ids = page.locator(f"{feed} [data-history-id]").evaluate_all(
        "nodes => nodes.map(node => node.dataset.historyId)")
    assert len(ids) == len(set(ids)), "source rows must remain unique after recovery"


@pytest.mark.skipif(os.name == "nt", reason="POSIX archive permissions trigger the real read failure")
@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_unreadable_archive_keeps_recent_messages_and_retries_real_history(
    direct_server_with_data, browser_engine, tmp_path,
):
    from playwright.sync_api import sync_playwright

    root, url = direct_server_with_data["data_dir"], direct_server_with_data["url"]
    archive = root / "archive" / "chat_20260901T000000.jsonl"
    _write(archive, [_human(0)])
    _write(root / "logs" / "chat.jsonl", [_human(index) for index in range(1, 176)])
    mode = archive.stat().st_mode
    archive.chmod(0)
    try:
        # Do not count a run under an identity that bypasses this failure.
        try:
            with archive.open("rb"):
                pytest.skip("the current identity bypasses unreadable-file permissions")
        except PermissionError:
            pass
        with sync_playwright() as pw:
            browser = getattr(pw, browser_engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 850})
                _open(page, url)
                fallback = _reads(page)[-1]
                assert fallback["status"] == 200
                payload = fallback["body"]
                assert payload["reason_code"] == "history_source_unavailable"
                assert payload["window"]["complete"] is False
                assert not payload.get("page_cursor") and not payload.get("next_cursor")
                recent = [row for row in payload["messages"] if row.get("role") == "user"]
                assert len(recent) == 150
                assert {row["text"] for row in recent} == {
                    f"history-human-{index:04d}" for index in range(26, 176)
                }
                assert page.locator(f"{MAIN} .message").filter(has_text="history-human-0175").count() == 1
                assert page.locator(f"{MAIN} .message").filter(has_text="history-human-0026").count() == 1
                assert page.locator(f"{MAIN} .chat-load-older button").inner_text() == "Retry loading messages"
                assert "Beginning of saved history" not in page.locator(MAIN).inner_text()
                _screenshot(page, tmp_path, f"archive-recent-readable-{browser_engine}")
                page.locator(MAIN).evaluate("root => { root.scrollTop = 0; }")
                _idle(page, MAIN)
                _screenshot(page, tmp_path, f"archive-unavailable-{browser_engine}")

                archive.chmod(mode)
                before_retry = len(_reads(page))
                _step(page, MAIN)
                # At the top edge a successful retry can immediately trigger
                # another real older-page fetch; assert the retry response itself.
                recovered = _reads(page)[before_retry]
                assert recovered["cursor"] is None and recovered["status"] == 200
                assert recovered["body"]["page_cursor"] and recovered["body"]["next_cursor"]
                _to_beginning(page, MAIN)
                assert page.locator(f"{MAIN} .message").filter(has_text="history-human-0000").count() == 1
                _assert_unique_rows(page, MAIN)
                _screenshot(page, tmp_path, f"archive-recovered-{browser_engine}")
            finally:
                browser.close()
    finally:
        archive.chmod(mode)


# Issue #1102 (defect 3). The shared observer only faults PAGED reads; the first
# read of a room has no cursor, so it gets its own switch. Installed before
# `_open` adds the observer, it sits beneath it: the observer still records the
# read, and sees it as unfinished while held and as an error when it fails.
#
# A failing room reads more than once on its own (the instance bootstrap asks
# again after the open transaction failed), so the fault is lifted by the Retry
# click itself, in the capture phase before the button's handler runs. Nothing
# but the reader's Retry can then be the read that succeeds.
_FAULT_RECENT_READ = """() => {
    const fetch = window.fetch.bind(window);
    window.__recentFault = null;
    window.fetch = async (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.href);
        const fault = window.__recentFault;
        if (url.pathname !== '/api/chat/history' || url.searchParams.get('cursor') || !fault
                || Number(url.searchParams.get('chat_id') || 1) !== fault.chatId) return fetch(input, init);
        if (fault.mode === 'fail') throw new TypeError('injected recent-read failure');
        await new Promise(resolve => { window.__releaseRecent = resolve; });
        return fetch(input, init);
    };
    document.addEventListener('click', event => {
        if (window.__recentFault?.mode !== 'fail' || !event.target.closest?.('.chat-load-older button')) return;
        window.__recentFault = null;
        window.__readsAtRetry = window.__historyReads.length;
    }, true);
}"""
_RECENT_STATE = """feed => {
    const root = document.querySelector(feed);
    const controls = root?.querySelector('.chat-load-older');
    const button = controls?.querySelector('button');
    const note = root.parentElement.querySelector('.chat-load-older-note');
    const shown = node => Boolean(node) && node.getClientRects().length > 0
        && getComputedStyle(node).visibility !== 'hidden';
    return {
        busy: controls?.getAttribute('aria-busy') === 'true',
        button: shown(button) ? button.textContent : '',
        disabled: Boolean(button?.disabled),
        note: shown(note) ? note.textContent : '',
        messages: root ? root.querySelectorAll('.message').length : -1,
    };
}"""


def _click_project(page, project):
    """`_open_project` without its settle: the read under test is still in flight."""
    row = page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]')
    row.wait_for(state="attached", timeout=30_000)
    mobile_toggle = page.locator('#page-chat [data-mobile-nav-toggle]')
    if mobile_toggle.is_visible() and not page.locator('#primary-sidebar').evaluate(
            "node => node.classList.contains('open')"):
        mobile_toggle.click()
    row.click()
    feed = f'#pchat-{project["id"]}-messages'
    page.locator(feed).wait_for(state="visible", timeout=30_000)
    return feed


def _wait_recent_state(page, feed, predicate):
    page.wait_for_function(f"feed => {{ const s = ({_RECENT_STATE})(feed); return {predicate}; }}",
                           arg=feed, timeout=30_000)
    return page.evaluate(_RECENT_STATE, feed)


def _is_seen_ack(project):
    def match(request):
        body = request.post_data or ""
        return (request.method == "POST" and request.url.endswith("/api/ui/preferences")
                and "project_seen_revision" in body and project["id"] in body)
    return match


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_project_panel_shows_a_slow_first_read_as_loading_and_a_failed_one_as_retry(
    direct_server_with_data, browser_engine, theme, tmp_path,
):
    from playwright.sync_api import sync_playwright
    from ouroboros.projects_registry import create_project

    root, url = direct_server_with_data["data_dir"], direct_server_with_data["url"]
    slow = create_project(root, "history-slow", name="Slow history room")
    failing = create_project(root, "history-failing", name="Failing history room")
    _write(root / "logs" / "chat.jsonl", [
        *[_human(index, slow["chat_id"]) for index in range(1, 6)],
        *[_human(index, failing["chat_id"]) for index in range(11, 16)],
    ])
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 850})
            acked = []
            page.on("request", lambda request: acked.extend(
                project["id"] for project in (slow, failing) if _is_seen_ack(project)(request)))
            page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
            page.add_init_script(f"({_FAULT_RECENT_READ})()")
            _open(page, url)

            # Slow: the response is withheld, and the panel must already say so.
            page.evaluate("id => { window.__recentFault = {chatId: id, mode: 'hold'}; }", slow["chat_id"])
            feed = _click_project(page, slow)
            loading = _wait_recent_state(page, feed, "s.busy && s.button === 'Loading saved history…'")
            assert loading["disabled"] and loading["messages"] == 0, loading
            assert loading["note"] == '', 'one read has one loading indicator'
            assert page.locator('#project-panel .chat-history-status').count() == 0
            assert [read for read in _reads(page, slow["chat_id"]) if not read["done"]], \
                "the loading state is asserted while the read is provably unanswered"
            assert slow["id"] not in acked, "an unpainted revision is never acknowledged"
            for width in (1343, 390):
                page.set_viewport_size({'width': width, 'height': 876})
                _screenshot(page, tmp_path, f'project-loading-{theme}-{width}-{browser_engine}')
            page.set_viewport_size({'width': 1280, 'height': 850})

            with page.expect_request(_is_seen_ack(slow), timeout=30_000):
                page.evaluate("() => { window.__recentFault = null; window.__releaseRecent(); }")
            _idle(page, feed)
            loaded = page.evaluate(_RECENT_STATE, feed)
            assert not loaded["busy"] and loaded["button"] != "Loading saved history…", loaded
            assert page.locator(f"{feed} .message").filter(has_text="history-human-0005").count() == 1
            _screenshot(page, tmp_path, f"project-first-read-loaded-{browser_engine}")

            # Failed: the same panel chrome carries the error and the existing Retry.
            page.evaluate("id => { window.__recentFault = {chatId: id, mode: 'fail'}; }", failing["chat_id"])
            feed = _click_project(page, failing)
            failed = _wait_recent_state(page, feed, "!s.busy && s.button === 'Retry loading messages'")
            assert not failed["disabled"] and failed["messages"] == 0, failed
            assert failed["note"] == "Some saved history could not be loaded.", failed
            note = page.locator('#project-panel .chat-history-status')
            assert note.evaluate('e => getComputedStyle(e).backgroundColor') == 'rgba(0, 0, 0, 0)'
            for width in (1343, 390):
                page.set_viewport_size({'width': width, 'height': 876})
                page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
                box = note.bounding_box()
                assert box['x'] >= 0 and box['x'] + box['width'] <= width + 1
                assert note.evaluate('e => e.scrollWidth <= e.clientWidth')
                _screenshot(page, tmp_path, f'project-error-{theme}-{width}-{browser_engine}')
            page.set_viewport_size({'width': 1280, 'height': 850})
            assert failing["id"] not in acked, "a failed read is never acknowledged"
            _screenshot(page, tmp_path, f"project-first-read-failed-{browser_engine}")

            # The reader's own click, on the visible control, is the recovery.
            with page.expect_request(_is_seen_ack(failing), timeout=30_000):
                page.locator(f"{feed} .chat-load-older button").click()
            _idle(page, feed)
            retried = page.evaluate(
                "id => window.__historyReads.slice(window.__readsAtRetry).filter(read => read.chatId === id)",
                failing["chat_id"])
            assert len(retried) == 1, "Retry is one read: the open transaction, not a second fetch beside it"
            assert retried[0]["cursor"] is None and retried[0]["status"] == 200
            recovered = page.evaluate(_RECENT_STATE, feed)
            assert recovered["button"] != "Retry loading messages", recovered
            assert "could not be loaded" not in recovered["note"], recovered
            assert page.locator(f"{feed} .message").filter(has_text="history-human-0015").count() == 1
            _assert_unique_rows(page, feed)
            _screenshot(page, tmp_path, f"project-first-read-recovered-{browser_engine}")
        finally:
            browser.close()


def _pin_reading_selection(page, feed):
    return page.locator(feed).evaluate("""root => {
        const rows = [...root.querySelectorAll('[data-history-id]')];
        const node = rows[Math.floor(rows.length / 2)];
        root.scrollTop += node.getBoundingClientRect().top - root.getBoundingClientRect().top - 180;
        const range = document.createRange(); range.selectNodeContents(node);
        const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
        window.__recoveryReading = {node, text: selection.toString(),
            top: node.getBoundingClientRect().top - root.getBoundingClientRect().top};
        return {text: selection.toString(), remaining: root.scrollHeight - root.scrollTop - root.clientHeight};
    }""")


def _assert_reading_preserved(page, feed):
    kept = page.locator(feed).evaluate("""root => {
        const old = window.__recoveryReading;
        return {sameNode: root.contains(old.node), selected: getSelection().toString() === old.text,
            drift: Math.abs(old.node.getBoundingClientRect().top - root.getBoundingClientRect().top - old.top)};
    }""")
    assert kept["sameNode"] and kept["selected"], kept
    assert kept["drift"] <= 6, kept


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_project_membership_refresh_keeps_reading_anchor_and_replaces_cursor(
    direct_server_with_data, browser_engine, tmp_path,
):
    from playwright.sync_api import sync_playwright
    from ouroboros.projects_registry import bind_task_to_project, create_project

    root, url = direct_server_with_data["data_dir"], direct_server_with_data["url"]
    project = create_project(root, "history-recovery", name="History recovery room")
    _write(root / "archive" / "chat_20260901T000000.jsonl", [
        _human(0, task_id="newly-bound-old-task", text="NEWLY_BOUND_OLD_ROW"),
        *[_human(index, project["chat_id"]) for index in range(1, 901)],
    ])
    _write(root / "logs" / "chat.jsonl", [_human(901, project["chat_id"])])
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 850})
            _open(page, url)
            feed = _open_project(page, project)
            for _ in range(3):
                _step(page, feed)
            reading = _pin_reading_selection(page, feed)
            assert reading["text"] and reading["remaining"] > 100, reading
            old_page = _reads(page, project["chat_id"])[-1]["body"]["page_cursor"]
            _screenshot(page, tmp_path, f"project-before-membership-change-{browser_engine}")

            bind_task_to_project(root, "newly-bound-old-task", project["id"], origin={"absent": "system"})
            _step(page, feed)
            changed = _reads(page, project["chat_id"])[-1]
            assert changed["status"] == 409
            assert changed["body"]["reason_code"] == "history_view_changed"
            assert page.locator(f"{feed} .chat-load-older button").inner_text() == "Refresh history"
            _assert_reading_preserved(page, feed)

            _step(page, feed)
            refreshed = _reads(page, project["chat_id"])[-1]
            assert refreshed["status"] == 200 and refreshed["cursor"] is None
            assert refreshed["body"]["page_cursor"] != old_page
            _assert_reading_preserved(page, feed)
            _screenshot(page, tmp_path, f"project-after-membership-refresh-{browser_engine}")
            # Follow the new handle through the same visible control. A renamed
            # button alone must not leave the reader retrying an invalid cursor.
            _step(page, feed)
            advanced = _reads(page, project["chat_id"])[-1]
            assert advanced["status"] == 200
            assert advanced["cursor"] == refreshed["body"]["next_cursor"]
            _assert_reading_preserved(page, feed)
            _assert_unique_rows(page, feed)
            page.evaluate("() => getSelection().removeAllRanges()")
            _to_beginning(page, feed)
            _idle(page, feed)
            assert page.locator(f"{feed} .message").filter(has_text="NEWLY_BOUND_OLD_ROW").count() == 1
        finally:
            browser.close()
