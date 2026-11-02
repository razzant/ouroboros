"""Quote navigation in the real app over real dialogue/history/project endpoints.

The fixture is also the reusable inspection stand's seed. Transport holds/failures
delay real responses; no successful lookup or history response is fabricated.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from ouroboros import dialogue_store as ds
from ouroboros.chat_chain import format_address, row_address
from ouroboros.chat_references import reference_from_row
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import bind_task_to_project, create_project
from tests._shared import producer_evidence_dir
from tests.test_chat_history_paging_browser import _FRAMES, _idle, _open, _result, _step
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _emit_ws_frame

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def seed_reply_history(root, *, outside_sidebar=False):
    """S1 quotes triage's owner row, also retained as S1's origin; long history in both rooms."""
    a = create_project(root, "quote-triage", name="Triage source", origin="owner_ui")
    b = create_project(root, "quote-s1", name="S1 quoting work", origin="owner_ui")
    x = create_project(root, "quote-converted", name="Converted source", origin="owner_ui")
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    saved = {}

    def say(chat, index, text, *, stream="chat", name="", **extra):
        row = {"chat_id": chat, "ts": (start + timedelta(seconds=index)).isoformat(),
               "direction": "in", "text": text, **extra}
        ds.append(root, stream, row, feed=stream == "chat")
        if name:
            saved[name] = {"row": row, "address": format_address(row_address(row))}
        return row

    for index in range(40):
        for chat in (1, a["chat_id"]):
            say(chat, index, f"Earlier context {index:03}")
    main = say(1, 40, "Exact old Main source", name="main", client_message_id="old-main")
    owner = say(a["chat_id"], 40, "Exact triage owner source", name="a", client_message_id="owner-a")
    say(a["chat_id"], 40, owner["text"], name="twin", client_message_id="other-same-words-time")
    say(1, 41, "A task's first row", task_id="converted-source", direction="out")
    moved = say(1, 42, "Exact later row of converted work", name="converted", task_id="converted-source", direction="out")
    bind_task_to_project(root, "s1-task", b["id"], origin={"ref": build_owner_message_ref(
        chat_id=a["chat_id"], client_message_id="owner-a", ts=owner["ts"], text=owner["text"]), "text": owner["text"]})
    # A late narration in a collapsed child's long bounded timeline: same words elsewhere cannot land for it.
    for index in range(50):
        say(a["chat_id"], 50 + index, f"Child timeline line {index:02}: " + "retained narration " * 12,
            stream="progress", name="line" if index == 30 else "", task_id="child-source", parent_task_id="parent-source",
            root_task_id="parent-source", direction="out", narration=True, delegation_role="subagent",
            subagent_task_id="child-source", subagent_role="worker")
    for task, extra in (("s1-task", {"chat_id": b["chat_id"]}), ("converted-source", {"chat_id": 1}),
                        ("parent-source", {"chat_id": a["chat_id"]}),
                        ("child-source", {"chat_id": a["chat_id"], "parent_task_id": "parent-source", "root_task_id": "parent-source"})):
        _result(root, task, **extra)
    for index in range(100, 570):
        for chat in (1, a["chat_id"]):
            say(chat, index, f"Recent context {index:03}")
    for name, source in (("a", owner), ("converted", moved), ("line", saved["line"]["row"])):
        say(1, 600 + len(saved), f"Reply quoting {name}", direction="out", task_id=f"answer-{name}",
            reply_to=reference_from_row(source))
    say(a["chat_id"], 620, "Newest Project message remains unread", direction="out")
    say(a["chat_id"], 621, "Project answer quoting Main", direction="out", reply_to=reference_from_row(main))
    if outside_sidebar:
        for index in range(205):
            create_project(root, f"newer-than-source-{index}", origin="owner_ui")
    return {"a": a, "b": b, "x": x, "sources": saved}


_OBSERVE_QUOTES = """() => {
    const fetch = window.fetch.bind(window);
    window.__quoteReads = []; window.__quoteSends = []; window.__quoteAcks = [];
    window.__quoteToasts = [];
    document.addEventListener('DOMContentLoaded', () => new MutationObserver(() => {
        for (const toast of document.querySelectorAll('#toast-stack .toast'))
            if (!window.__quoteToasts.includes(toast.textContent)) window.__quoteToasts.push(toast.textContent);
    }).observe(document.body, {childList:true, subtree:true}));
    const send = WebSocket.prototype.send;
    WebSocket.prototype.send = function(value) { window.__quoteSends.push(value); return send.call(this, value); };
    window.fetch = async (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.href);
        if (init?.body && String(init.body).includes('project_seen_revision')) window.__quoteAcks.push(JSON.parse(init.body));
        const address = url.searchParams.get('around');
        const lookup = url.pathname === '/api/projects' && url.searchParams.has('chat_id');
        if (!address && !lookup) return fetch(input, init);
        const read = {url: url.pathname + url.search, address, lookup, done: false};
        window.__quoteReads.push(read);
        const response = await fetch(input, init);  // always the real endpoint, even the held answer
        read.status = response.status; read.body = await response.clone().json();
        if (window.__quoteHold === (lookup ? 'lookup' : address)) {
            window.__quoteHold = null; window.__quoteHeld = read;
            await new Promise(resolve => { window.__quoteRelease = resolve; });
        }
        read.done = true;
        if (address && window.__quoteFail === address) { window.__quoteFail = null; throw new TypeError('injected offline read'); }
        return response;
    };
}"""


def _quote(page, address, feed="#chat-messages"):
    return page.locator(f'{feed} .chat-quote[data-address={json.dumps(address)}] .chat-quote-main')


def _landed(page, address, feed):
    page.wait_for_function("""([address, feed]) => {
        const target = document.querySelector(`${feed} .is-quote-target`);
        return target && (target.dataset.rowAddress === address || target.dataset.historyId
            === window.__quoteReads.findLast(r => r.address === address && r.body?.anchor?.state === 'found')?.body.anchor.history_id);
    }""", arg=[address, feed], timeout=15_000)
    target = page.locator(f"{feed} .is-quote-target")
    facts = target.evaluate("""(node, selector) => {
        const feed = document.querySelector(selector), page = feed.closest('.page, .chat-instance-panel'), r = node.getBoundingClientRect();
        const bounds = [feed.getBoundingClientRect()];
        for (let p = node.parentElement; p && p !== feed; p = p.parentElement)
            if (getComputedStyle(p).overflowY.match(/auto|scroll/)) bounds.push(p.getBoundingClientRect());
        const header = page?.querySelector('.chat-page-header, .chat-panel-statusbar')?.getBoundingClientRect();
        return {address:node.dataset.rowAddress, id:node.dataset.historyId, focused:document.activeElement === node,
            visible:node.getClientRects().length > 0, top:r.top, bottom:r.bottom,
            clear:Math.max(header?.bottom || 0, ...bounds.map(b => b.top)), edge:Math.min(...bounds.map(b => b.bottom))};
    }""", feed)
    assert facts["focused"] and facts["visible"] and facts["top"] >= facts["clear"] - 2, facts
    assert min(facts["bottom"], facts["edge"]) - facts["top"] >= 12, facts
    return facts


def _evidence(page, root, name, fixture, facts):
    destination = producer_evidence_dir(root, "quoted-navigation")
    page.screenshot(path=str(destination / f"{name}.png"), animations="disabled")
    (destination / f"{name}.html").write_text(page.content())
    state = page.evaluate("() => ({reads:window.__quoteReads, history:window.__historyReads, sends:window.__quoteSends, acks:window.__quoteAcks, toasts:window.__quoteToasts})")
    (destination / f"{name}.json").write_text(json.dumps({"fixture": fixture, "facts": facts, **state}, indent=2))


def _settled_row(page, target, feed):
    """Observe the actual row beyond panel/highlight motion and pending history, without moving it.

    A first-frame focus is insufficient: WebKit's later resize used to restore the
    opening anchor at scrollTop 0 and put this same focused line behind the composer.
    """
    page.wait_for_function("() => window.__historyReads.every(r => r.done) && window.__quoteReads.every(r => r.done)")
    return target.evaluate("""(node, selector) => new Promise((resolve, reject) => {
        const feed = document.querySelector(selector), panel = feed.closest('.page, .chat-instance-panel');
        const rect = n => { const r = n.getBoundingClientRect(); return {x:r.x, top:r.top, bottom:r.bottom, width:r.width}; };
        const start = performance.now(), samples = [];
        let previous = '', changed = start;
        const sample = () => {
            const now = performance.now(), box = rect(node), clips = [];
            for (let p = node.parentElement; p; p = p.parentElement) {
                if (p === feed || /auto|scroll/.test(getComputedStyle(p).overflowY))
                    clips.push({...rect(p), id:p.id, scrollTop:p.scrollTop});
                if (p === feed) break;
            }
            const blockers = [...panel.querySelectorAll('.chat-scroll-bottom-btn.visible, .chat-input-area .chat-input-wrap')]
                .filter(p => p.getClientRects().length && getComputedStyle(p).visibility !== 'hidden')
                .map(p => ({...rect(p), role:p.className}));
            const header = panel.querySelector('.chat-page-header, .chat-panel-statusbar');
            const facts = {id:node.dataset.historyId, address:node.dataset.rowAddress, focused:document.activeElement === node,
                ...box, feedTop:feed.scrollTop, clips, blockers,
                clear:Math.max(header?.getBoundingClientRect().bottom || 0, ...clips.map(p => p.top)),
                edge:Math.min(...clips.map(p => p.bottom)),
                overlaps:blockers.filter(p => p.x < box.x+box.width && p.x+p.width > box.x && p.top < box.bottom && p.bottom > box.top)};
            const key = JSON.stringify(facts);
            if (key !== previous) { previous = key; changed = now; samples.push({elapsed:now-start, ...facts}); }
            if (now-start >= 2200 && now-changed >= 600) resolve({...facts, elapsed:now-start, quiet:now-changed, samples});
            else if (now-start > 8000) reject(new Error('Quote geometry did not settle: '+JSON.stringify(samples)));
            else requestAnimationFrame(sample);
        };
        requestAnimationFrame(sample);
    })""", feed)


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_exact_cross_room_quote_and_return_preserve_drafts_unread_and_history(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright
    fixture = seed_reply_history(direct_server_with_data["data_dir"], outside_sidebar=True)
    source = fixture["sources"]["a"]["address"]
    feed = "#pchat-quote-triage-messages"
    facts = {}
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        page.add_init_script(f"({_OBSERVE_QUOTES})()")
        try:
            _open(page, direct_server_with_data["url"])
            assert page.locator('.nav-project-row[data-project-id="quote-triage"]').count() == 0
            page.locator('#chat-input').fill("Keep this unsent draft")
            page.locator('#chat-file-input').set_input_files({"name": "draft.txt", "mimeType": "text/plain", "buffer": b"unsent"})
            _quote(page, source).click()
            facts["triage"] = _landed(page, source, feed)
            _evidence(page, tmp_path, f"triage-target-{browser_engine}", fixture, facts)
            assert page.locator('#project-panel-title').inner_text() == "Triage source"
            assert not page.evaluate("() => window.__quoteAcks.some(x => x.project_seen_revision?.['quote-triage'])")
            reads = page.evaluate("() => window.__quoteReads")
            assert any(r["lookup"] and str(fixture["a"]["chat_id"]) in r["url"] for r in reads)
            assert next(r["body"]["projects"][0]["visible_revision"] for r in reads if r["lookup"]) > 0
            _step(page, feed)
            assert page.locator(f'{feed} [data-client-message-id="owner-a"]').count() == 1
            page.locator('#pchat-quote-triage-scroll-bottom').click()
            _idle(page, feed)
            main = fixture["sources"]["main"]["address"]
            _quote(page, main, feed).click()
            facts["main"] = _landed(page, main, "#chat-messages")
            _evidence(page, tmp_path, f"main-target-{browser_engine}", fixture, facts)
            assert page.locator('#chat-input').input_value() == "Keep this unsent draft"
            assert "draft.txt" in page.locator('#chat-attachment-preview').inner_text()
            # Reply in the composer survives reload, and its quote retries from an evicted window.
            # Evidence capture may outlast the landing highlight; Reply must still address that saved row.
            page.locator('#chat-messages .is-quote-target').wait_for(state='detached')
            page.locator(f'#chat-messages [data-row-address={json.dumps(main)}] .chat-message-reply').click()
            assert page.evaluate("() => JSON.parse(sessionStorage.getItem('ouro_chat_reply')).address") == main
            page.locator('#chat-scroll-bottom').click()
            assert not page.evaluate("() => window.__quoteSends.some(x => JSON.parse(x).type === 'chat')")
            page.reload(wait_until="domcontentloaded")
            _idle(page, '#chat-messages')
            page.locator('#page-chat .chat-quote--composer .chat-quote-main').click()
            facts["composer"] = _landed(page, main, "#chat-messages")
            sends = page.evaluate("() => window.__quoteSends.map(x => {try{return JSON.parse(x)}catch{return {}}})")
            assert not any(frame.get("type") == "chat" for frame in sends)
        finally:
            _evidence(page, tmp_path, f"cross-room-{browser_engine}", fixture, facts)
            browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
@pytest.mark.parametrize("viewport", [(1280, 850), (1084, 724)], ids=["wide", "sender"])
def test_exact_nested_line_and_post_quote_conversion(direct_server_with_data, browser_engine, viewport, tmp_path):
    from playwright.sync_api import sync_playwright
    root = direct_server_with_data["data_dir"]
    fixture = seed_reply_history(root)
    facts = {}
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        page.add_init_script(f"({_OBSERVE_QUOTES})()")
        try:
            _open(page, direct_server_with_data["url"])
            address = fixture["sources"]["converted"]["address"]
            # The quote is already painted with its old chat label before the binding changes.
            assert _quote(page, address).count() == 1
            bind_task_to_project(root, "converted-source", fixture["x"]["id"], origin={"absent": "system"})
            _quote(page, address).click()
            facts["converted"] = _landed(page, address, '#pchat-quote-converted-messages')
            _evidence(page, tmp_path, f"converted-target-{browser_engine}", fixture, facts)
            page.locator('#project-panel-close').click()
            _idle(page, '#chat-messages')
            address = fixture["sources"]["line"]["address"]
            _quote(page, address).click()
            feed = '#pchat-quote-triage-messages'
            facts["line_initial"] = _landed(page, address, feed)
            target = page.locator(f'{feed} .chat-live-line[data-history-id={json.dumps(facts["line_initial"]["id"])}]')
            facts["line"] = _settled_row(page, target, feed)
            _evidence(page, tmp_path, f"line-target-{browser_engine}", fixture, facts)
            assert facts["line"]["id"].startswith('progress:')
            assert facts["line"]["focused"] and "Child timeline line 30" in target.inner_text()
            assert facts["line"]["top"] >= facts["line"]["clear"] - 2, facts["line"]
            assert facts["line"]["bottom"] <= facts["line"]["edge"] + 2, facts["line"]
            assert not facts["line"]["overlaps"], facts["line"]
            timeline_top = "n => n.closest('[data-live-timeline]').scrollTop"
            before = target.evaluate(timeline_top)
            assert before > 0
            # Real input reads further inside the bounded timeline. A subsequent
            # composer resize must not reveal the old quote again or take focus.
            page.keyboard.press('PageDown')
            facts["after_key"] = _settled_row(page, target, feed)
            after = target.evaluate(timeline_top)
            assert after > before + 40, (before, after, facts["after_key"])
            page.locator('#pchat-quote-triage-input').fill('Project draft\n' * 5)
            facts["after_resize"] = _settled_row(page, target, feed)
            assert abs(target.evaluate(timeline_top) - after) <= 2, facts["after_resize"]
            assert not facts["after_resize"]["focused"], 'late layout stole focus from the composer'
            _evidence(page, tmp_path, f"line-after-gesture-{browser_engine}", fixture, facts)
            page.locator('#project-panel-close').click()
            _idle(page, '#chat-messages')
            # Ordinary reopening still reads the present; source navigation adds no global Back.
            page.locator('.nav-project-row[data-project-id="quote-triage"]').click()
            _idle(page, '#pchat-quote-triage-messages')
            assert page.locator('#pchat-quote-triage-messages').evaluate(
                'n => n.scrollHeight - n.scrollTop - n.clientHeight') < 48
        finally:
            _evidence(page, tmp_path, f"line-conversion-{browser_engine}", fixture, facts)
            browser.close()


def _arrive(page, root, index, text):
    """A reply the server stores and delivers live while the reader is in Main."""
    ts = (datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(seconds=700 + index)).isoformat()
    ds.append(root, "chat", {"chat_id": 1, "ts": ts, "direction": "out", "text": text, "format": "markdown"})
    _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1, "content": text, "ts": ts})
    page.locator("#chat-messages .chat-bubble").filter(has_text=text.split()[0]).wait_for(state="attached")


_LIVE_EDGE = "n => ({gap: n.scrollHeight - n.scrollTop - n.clientHeight, top: n.scrollTop, height: n.scrollHeight})"


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_cancelled_failed_and_overlapping_quotes_keep_newer_intent(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright
    root = direct_server_with_data["data_dir"]
    fixture = seed_reply_history(root)
    a, main = [fixture["sources"][key]["address"] for key in ("a", "main")]
    facts = {}
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        page.add_init_script(f"({_OBSERVE_QUOTES})()")
        try:
            _open(page, direct_server_with_data["url"])
            page.locator('#chat-input').fill('Draft through failed lookup')
            # The reader follows the live edge. A reply arrives while the quote's read is held; the read then
            # fails. Nothing moved, so that reply and the next one are followed with no ↓ press.
            page.evaluate('address => { window.__quoteHold = address; window.__quoteFail = address; }', a)
            _quote(page, a).click()
            page.wait_for_function('() => window.__quoteHeld')
            _arrive(page, root, 0, "LIVE_DURING_FAILED_QUOTE " + "while the quoted read is held " * 40)
            facts["held"] = page.locator('#chat-messages').evaluate(_LIVE_EDGE)
            assert facts["held"]["gap"] > 48, facts  # held below while the read was pending
            page.evaluate('() => window.__quoteRelease()')
            page.get_by_text('Could not load the quoted message. Try again.').wait_for()
            page.wait_for_function("() => { const n = document.querySelector('#chat-messages');"
                                   " return n.scrollHeight - n.scrollTop - n.clientHeight < 4; }", timeout=10_000)
            facts["after_failure"] = page.locator('#chat-messages').evaluate(_LIVE_EDGE)
            _arrive(page, root, 1, "LIVE_AFTER_FAILED_QUOTE " + "the next reply is still followed " * 40)
            page.wait_for_function("() => { const n = document.querySelector('#chat-messages');"
                                   " return n.scrollHeight - n.scrollTop - n.clientHeight < 4; }", timeout=10_000)
            facts["next_reply"] = page.locator('#chat-messages').evaluate(_LIVE_EDGE)
            _evidence(page, tmp_path, f"failed-read-follows-{browser_engine}", fixture, facts)
            assert page.locator('#project-panel').get_attribute('class').find('open') == -1
            for interrupt in ('gesture', 'leave', 'newer'):
                page.evaluate("() => { window.__quoteHold = 'lookup'; window.__quoteHeld = null; }")
                _quote(page, a).click()
                page.wait_for_function('() => window.__quoteHeld?.lookup')
                if interrupt == 'gesture':
                    page.locator('#chat-messages').dispatch_event('wheel', {'deltaY': -100})
                elif interrupt == 'leave':
                    page.locator('[data-nav-page="files"]').click()
                else:
                    _quote(page, fixture['sources']['converted']['address']).click()
                    facts['newer'] = _landed(page, fixture['sources']['converted']['address'], '#chat-messages')
                page.evaluate('() => window.__quoteRelease()')
                page.wait_for_function('() => window.__quoteReads.every(r => r.done)')
                page.evaluate(_FRAMES)
                assert page.locator('#pchat-quote-triage-messages').count() == 0, 'stale lookup created a destination'
                if interrupt == 'leave':
                    page.locator('[data-nav-page="chat"]').click()
                page.locator('#chat-scroll-bottom').evaluate('n => n.click()')
                _idle(page, '#chat-messages')
            assert page.locator('#chat-input').input_value() == 'Draft through failed lookup'
            # Same original quote, now healthy, retries without a permanently unavailable label/state.
            _quote(page, a).click()
            facts['retry'] = _landed(page, a, '#pchat-quote-triage-messages')
        finally:
            _evidence(page, tmp_path, f"races-{browser_engine}", fixture, facts)
            browser.close()
