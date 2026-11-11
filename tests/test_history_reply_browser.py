"""Quote navigation in the real app over real dialogue/history/project endpoints.

The fixture is also the reusable inspection stand's seed. Transport holds/failures
delay real responses; no successful lookup or history response is fabricated.

Search in a chat lands on its matches through the same quote landing (the scenes at the end).
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
from tests import _room_browser_shared as room_browser
from tests._shared import producer_evidence_dir
from tests.test_chat_history_paging_browser import _FRAMES, _idle, _open, _result, _step
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _SETTLE_RESTORE_FRAMES, _emit_ws_frame

pytestmark = [pytest.mark.ui_browser]


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


# ---- Search in a chat (docs/DESIGN.md "Search in a chat"): a match lands through the same quote landing ----------
# Main holds four hundred saved messages, so most matches lie far older than its newest page, and a Project holds its
# own, with a question Main shows only as its copy. Ctrl+F opens the room in front's search with the keyboard in its
# field and the owner's draft untouched; Enter and Shift+Enter land on the exact rows (reading the page around one that
# is not loaded) below the open sheet; Esc closes it and gives the field back. On a phone, "+" → Search this chat
# opens it for a finger. Screenshots of each scene, in both themes, go to the run's evidence.

SEARCH_START = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
SEARCH_LONG = ("We walked along the river for most of the afternoon and stopped by the weir where the water slows down. "
        "Somewhere past the alders a Kingfisher flashed blue and orange, gone before the camera came up, and we "
        "waited another hour at the bank hoping it would come back to the same branch above the pool.")
# Owner messages are the odd ones: the matches alternate Ouroboros and the owner, newest first.
SEARCH_MAIN_MATCHES = {31: "The kingfisher nested by the old mill.", 100: SEARCH_LONG,
                201: "Saw a KINGFISHER again near the bridge", 390: "Remember the kingfisher photo?"}
SEARCH_QUESTION = "Which kingfisher route should the survey take?"


def _seed_search_rooms(root):
    from ouroboros.owner_quiz import record_asked

    for index in range(400):
        text = SEARCH_MAIN_MATCHES.get(index, f"Routine note {index}: nothing to report about the garden today.")
        ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(minutes=index)).isoformat(),
                                 "direction": "in" if index % 2 else "out", "text": text}, feed=True)
    room = create_project(root, "search-room", name="Bird notes", origin="owner_ui")
    for index, text in enumerate(["Plan the kingfisher survey for the project.", "Weather looks fine on Saturday.",
                                  "Kingfisher survey route is ready."]):
        ds.append(root, "chat", {"chat_id": room["chat_id"], "ts": (SEARCH_START + timedelta(days=1, minutes=index)).isoformat(),
                                 "direction": "out" if index % 2 else "in", "text": text}, feed=True)
    # The Project's open question: Main draws it only as a copy of the Project's, so only the Project finds it.
    record_asked(root, "survey-asker", quiz_id="route", question=SEARCH_QUESTION, options=["River", "Weir"],
                 wait_for_answer=True)
    ds.append(root, "chat", {"chat_id": room["chat_id"], "ts": (SEARCH_START + timedelta(days=1, minutes=5)).isoformat(),
                             "direction": "out", "type": "quiz", "task_id": "survey-asker", "text": SEARCH_QUESTION,
                             "quiz": {"quiz_id": "route", "question": SEARCH_QUESTION, "options": ["River", "Weir"],
                                      "wait_for_answer": True}}, feed=True)
    return room


_SEARCH_STATE = """page => {
    const root = document.querySelector(page);
    const find = root?.querySelector('.chat-find');
    const input = find?.querySelector('.chat-find-input');
    const active = document.activeElement;
    return {
        open: Boolean(find && !find.hidden),
        focused: active === input,
        activeId: active?.id || '',
        count: find?.querySelector('.chat-find-count')?.textContent || '',
        note: find?.querySelector('.chat-find-note')?.textContent || '',
        results: [...(find?.querySelectorAll('.chat-find-result') || [])].map(node => ({
            selected: node.getAttribute('aria-selected') === 'true',
            snippet: node.querySelector('.chat-find-snippet').textContent,
            marks: [...node.querySelectorAll('mark')].map(mark => mark.textContent),
            meta: node.querySelector('.chat-find-meta').textContent})),
        authored: Boolean(find?.querySelector('.chat-find-results')?.closest('[data-i18n-authored]')),
        bottom: find && !find.hidden ? find.getBoundingClientRect().bottom : 0,
    };
}"""

# The innermost shown row holding ``words``: a message bubble, or the card it is drawn in (a question's).
_SEARCH_ROW = """([page, words]) => {
    const rows = [...document.querySelectorAll(`${page} [data-row-address], ${page} [data-history-id]`)]
        .filter(row => row.textContent.includes(words) && row.getClientRects().length);
    return rows.find(row => !rows.some(other => other !== row && row.contains(other))) || null;
}"""
_SEARCH_LANDED = """([page, words]) => {
    const node = (""" + _SEARCH_ROW + """)([page, words]);
    if (!node) return null;
    const box = node.getBoundingClientRect(), feed = node.closest('.chat-messages, #chat-messages').getBoundingClientRect();
    return {top: box.top, bottom: box.bottom, feedBottom: feed.bottom, target: node.classList.contains('is-quote-target'),
            highlighted: CSS.highlights ? CSS.highlights.has('ouro-chat-find') : null};
}"""


def _search_evidence(tmp_path):
    return producer_evidence_dir(tmp_path / "chat-search-evidence", "chat-search", env="CHAT_SEARCH_EVIDENCE_DIR")


def _open_main_for_search(page, url):
    """Main, as a client opening the app sees it: its socket open and its newest saved page drawn."""
    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
    page.locator("#chat-messages [data-history-id]").first.wait_for(state="attached", timeout=30_000)
    page.evaluate(_SETTLE_RESTORE_FRAMES)


def _search_state(page, scope):
    return page.evaluate(_SEARCH_STATE, scope)


def _wait_search_results(page, scope, count):
    page.wait_for_function("""([scope, n]) => {
        const find = document.querySelector(`${scope} .chat-find`);
        return find && !find.hasAttribute('data-searching') && find.querySelectorAll('.chat-find-result').length === n
            && (n !== 0 || find.querySelector('.chat-find-count').textContent === '0 found');
    }""", arg=[scope, count], timeout=30_000)


def _search_landed(page, scope, words):
    """The row holding ``words`` is shown whole below the open sheet (as it stands after the landing)."""
    page.wait_for_function(f"args => ({_SEARCH_ROW})(args)?.getBoundingClientRect().height > 0", arg=[scope, words],
                           timeout=30_000)
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    landed = page.evaluate(_SEARCH_LANDED, [scope, words])
    below = _search_state(page, scope)["bottom"]
    assert landed and landed["top"] >= below - 1 and landed["bottom"] <= landed["feedBottom"] + 1, (landed, below)
    return landed


def _search_both_themes(page, path):
    """The scene in the dark theme it opened in, then in the light one a new client takes from the system."""
    page.screenshot(path=str(path.with_name(f"{path.stem}-dark.png")))
    page.emulate_media(color_scheme="light")
    page.wait_for_function("() => document.documentElement.dataset.theme === 'light'")
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    page.screenshot(path=str(path.with_name(f"{path.stem}-light.png")))
    page.emulate_media(color_scheme="dark")
    page.wait_for_function("() => document.documentElement.dataset.theme === 'dark'")
    page.evaluate(_SETTLE_RESTORE_FRAMES)


@pytest.mark.parametrize("browser_engine", ["webkit", "chromium"])
def test_ctrl_f_finds_saved_messages_of_the_room_in_front_and_lands_on_them(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright

    room = _seed_search_rooms(direct_server_with_data["data_dir"])
    shots = _search_evidence(tmp_path)
    errors = []
    with sync_playwright() as pw:
        try:
            browser = getattr(pw, browser_engine).launch(headless=True)
        except Exception as error:  # an engine this host has not installed is named, never passed silently
            if "Executable doesn't exist" in str(error):
                pytest.skip(f"{browser_engine} is not installed for this Playwright")
            raise
        page = browser.new_page(viewport={"width": 1512, "height": 949}, device_scale_factor=2, color_scheme="dark")
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.fill("#chat-input", "my unsent draft")
            # A pointer-opened '+' menu leaves focus in the composer, but still owns the key.
            page.locator("#chat-plus").click()
            menu = page.locator(".chat-plus-menu:not([data-leaving])")
            menu.wait_for(state="visible")
            assert page.evaluate("document.activeElement.id") == "chat-input"
            page.keyboard.press("Control+f")
            assert not page.locator(".chat-find:visible").count(), "an open menu keeps Find even without keyboard focus"
            assert menu.is_visible()
            page.keyboard.press("Escape")
            menu.wait_for(state="hidden")
            page.keyboard.press("Control+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            state = _search_state(page, "#page-chat")
            assert state["open"] and state["focused"] and not state["results"], state
            page.keyboard.type("Kingfisher")
            _wait_search_results(page, "#page-chat", 4)
            state = _search_state(page, "#page-chat")
            assert state["count"] == "4 found" and state["authored"], state  # never the Project's question
            assert [result["marks"][0].casefold() for result in state["results"]] == ["kingfisher"] * 4, state
            assert state["results"][0]["snippet"] == "Remember the kingfisher photo?", state
            assert state["results"][2]["snippet"].startswith("…") and state["results"][2]["snippet"].endswith("…"), state
            assert "Ouroboros" in state["results"][0]["meta"] and "You" in state["results"][1]["meta"], state
            _search_both_themes(page, shots / f"{browser_engine}-main-results.png")

            page.keyboard.press("Enter")  # the newest match, loaded already
            landed = _search_landed(page, "#page-chat", "Remember the kingfisher photo?")
            assert landed["target"], landed
            state = _search_state(page, "#page-chat")
            assert state["focused"] and state["count"] == "1 of 4" and state["results"][0]["selected"], state
            page.keyboard.press("Enter")
            page.keyboard.press("Enter")  # far older than the newest page: the page around it is read
            _search_landed(page, "#page-chat", "flashed blue and orange")
            state = _search_state(page, "#page-chat")
            assert state["focused"] and state["count"] == "3 of 4" and state["results"][2]["selected"], state
            _search_both_themes(page, shots / f"{browser_engine}-main-landed-older.png")
            page.keyboard.press("Shift+Enter")
            _search_landed(page, "#page-chat", "Saw a KINGFISHER again")
            assert _search_state(page, "#page-chat")["count"] == "2 of 4"

            page.keyboard.press("Escape")
            state = _search_state(page, "#page-chat")
            assert not state["open"] and state["activeId"] == "chat-input", state
            assert page.input_value("#chat-input") == "my unsent draft"
            assert page.evaluate("() => CSS.highlights ? !CSS.highlights.has('ouro-chat-find') : true")

            # A Project in front: the key searches the Project's own messages, never Main behind it.
            room_browser._open(page, room["id"])
            panel, feed = f'[id="panel-pchat-{room["id"]}"]', f'[id="pchat-{room["id"]}-messages"]'
            page.locator(feed).click(position={"x": 40, "y": 200})
            page.keyboard.press("Meta+f" if browser_engine == "webkit" else "Control+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            assert not _search_state(page, "#page-chat")["open"]
            page.keyboard.type("kingfisher survey")
            _wait_search_results(page, panel, 2)
            state = _search_state(page, panel)
            assert [result["snippet"] for result in state["results"]] == [
                "Kingfisher survey route is ready.", "Plan the kingfisher survey for the project."], state
            page.keyboard.press("Enter")
            _search_landed(page, panel, "Kingfisher survey route is ready.")
            _search_both_themes(page, shots / f"{browser_engine}-project-landed.png")
            # Its question is the Project's: found and landed on here, where Main finds nothing of it.
            page.fill(f"{panel} .chat-find-input", "kingfisher route")
            _wait_search_results(page, panel, 1)
            assert _search_state(page, panel)["results"][0]["snippet"] == SEARCH_QUESTION
            page.keyboard.press("Enter")
            assert _search_landed(page, panel, SEARCH_QUESTION)["target"]
            page.keyboard.press("Escape")
            assert not _search_state(page, panel)["open"]
            assert page.locator(feed).is_visible(), "Esc closed the search, not the Project"
        finally:
            browser.close()
    assert not errors, errors


def test_search_lands_on_a_rendered_project_answer_and_other_pages_keep_find(direct_server_with_data, tmp_path):
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    room = _seed_search_rooms(root)
    answer = "The swallows returned to their nests."
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(days=2)).isoformat(),
                              "direction": "system", "type": "project_completion_summary", "project_id": room["id"],
                              "text": "The project has finished.", "completion_answer": answer}, feed=True)
    # Production start/handoff rows carry both IDs: they bind a work portal, not a navigable chat bubble.
    for index, kind in enumerate(("project_started", "project_handoff")):
        ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(days=2, minutes=index + 1)).isoformat(),
                                  "direction": "system", "type": kind, "project_id": room["id"],
                                  "task_id": f"portal-work-{index}", "text": "portal-only falcon",
                                  "completion_answer": "portal-only falcon answer"}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(days=2, minutes=3)).isoformat(),
                              "direction": "in", "source": "web", "text": "Here is the document.\n\n[Attached file: bird.txt]",
                              "attachments": [{"name": "bird.txt", "unavailable": "missing"}]}, feed=True)
    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 949}, device_scale_factor=2)
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.keyboard.press("Control+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            page.keyboard.type("swallows")
            _wait_search_results(page, "#page-chat", 1)
            result = _search_state(page, "#page-chat")["results"][0]
            assert result["snippet"] == answer and "Ouroboros" in result["meta"]
            page.keyboard.press("Enter")
            _search_landed(page, "#page-chat", answer)
            page.screenshot(path=str(_search_evidence(tmp_path) / "webkit-project-answer-found.png"))
            page.fill("#page-chat .chat-find-input", "portal-only falcon")
            _wait_search_results(page, "#page-chat", 0)
            assert _search_state(page, "#page-chat")["count"] == "0 found"
            page.fill("#page-chat .chat-find-input", "kingfisher route")  # the Project's question Main copies
            _wait_search_results(page, "#page-chat", 0)
            page.fill("#page-chat .chat-find-input", "Attached file")
            _wait_search_results(page, "#page-chat", 0)
            page.fill("#page-chat .chat-find-input", "Here is the document")
            _wait_search_results(page, "#page-chat", 1)
            assert _search_state(page, "#page-chat")["results"][0]["snippet"] == "Here is the document."
            page.keyboard.press("Enter")
            _search_landed(page, "#page-chat", "Here is the document.")
            page.keyboard.press("Escape")
            room_browser._open(page, room["id"])
            room_browser._nav(page, "settings")
            page.keyboard.press("Control+f")
            assert not page.locator(".chat-find:visible").count(), "a departed Project never steals Find on Settings"
        finally:
            browser.close()


@pytest.mark.parametrize("browser_engine", ["webkit"])
def test_a_finger_opens_search_from_plus_and_lands_on_a_tapped_match(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright

    _seed_search_rooms(direct_server_with_data["data_dir"])
    shots = _search_evidence(tmp_path)
    errors = []
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, has_touch=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.locator("#page-chat #chat-plus").tap()
            item = page.locator(".chat-plus-menu:not([data-leaving]) [data-plus-item='search']")
            assert page.locator(".chat-plus-menu:not([data-leaving]) [role=menuitem]").all_inner_texts()[:2] == [
                "Attach file", "Search this chat"]
            item.tap()
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            page.keyboard.type("kingfisher")
            _wait_search_results(page, "#page-chat", 4)
            sheet = page.locator("#page-chat .chat-find").bounding_box()
            assert sheet["x"] >= 0 and sheet["x"] + sheet["width"] <= 390.5, sheet
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.screenshot(path=str(shots / f"{browser_engine}-phone-results.png"))
            page.locator("#page-chat .chat-find-result").nth(3).tap()
            _search_landed(page, "#page-chat", "The kingfisher nested by the old mill.")
            assert page.locator("#page-chat .chat-find-results").is_hidden(), "the list folds over a phone's landing"
            page.screenshot(path=str(shots / f"{browser_engine}-phone-landed.png"))
            page.locator("#page-chat .chat-find-input").tap()
            assert page.locator("#page-chat .chat-find-results").is_visible(), "touching the field opens the list again"
            page.locator("#page-chat .chat-find-close").tap()
            assert not _search_state(page, "#page-chat")["open"]
        finally:
            browser.close()
    assert not errors, errors


# The sheet's frame: its corner, its padding and its field's corner as drawn (DESIGN "Search in a chat").
_SEARCH_FRAME = """page => {
    const find = document.querySelector(`${page} .chat-find`), bar = find.querySelector('.chat-find-bar');
    const sheet = find.getBoundingClientRect(), field = bar.getBoundingClientRect(), style = getComputedStyle(find);
    const note = find.querySelector('.chat-find-note');
    return {height: sheet.height, barHeight: field.height, pad: parseFloat(style.paddingTop),
            radius: parseFloat(style.borderTopLeftRadius),
            barRadius: Math.min(parseFloat(getComputedStyle(bar).borderTopLeftRadius), field.height / 2),
            note: note.getClientRects().length ? note.textContent : null, searching: find.hasAttribute('data-searching')};
}"""


def _search_frame(page, scope, *, alone):
    """The field is concentric with the sheet; the sheet is a capsule while it holds the field alone, else a card with
    the same corner (never an oval, never a box whose corner fights its field's)."""
    frame = page.evaluate(_SEARCH_FRAME, scope)
    assert abs(frame["radius"] - frame["pad"] - frame["barRadius"]) <= 0.5, frame
    if alone:
        assert frame["note"] is None, frame
        assert abs(frame["height"] - frame["barHeight"] - 2 * frame["pad"]) <= 0.5, frame
        assert abs(frame["radius"] - frame["height"] / 2) <= 0.5, frame
    else:
        assert frame["height"] > frame["barHeight"] + 2 * frame["pad"] + 1, frame
    return frame


@pytest.mark.parametrize("viewport", [(1512, 949), (1100, 750), (390, 844)], ids=["owner", "window", "phone"])
def test_the_search_sheet_is_a_capsule_until_it_has_more_to_say(direct_server_with_data, viewport, tmp_path):
    from playwright.sync_api import sync_playwright

    _seed_search_rooms(direct_server_with_data["data_dir"])
    shots, (width, height) = _search_evidence(tmp_path), viewport
    name = f"webkit-{width}x{height}"
    errors, held = [], []
    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2, color_scheme="dark")
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.keyboard.press("Control+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            empty = _search_frame(page, "#page-chat", alone=True)
            _search_both_themes(page, shots / f"{name}-empty.png")

            page.keyboard.type("Kingfisher")
            _wait_search_results(page, "#page-chat", 4)
            assert _search_frame(page, "#page-chat", alone=False)["radius"] == empty["radius"]
            _search_both_themes(page, shots / f"{name}-results.png")

            # A read held in transport: the sheet says it is searching (the real answer is released after).
            page.route("**/api/chat/search?*", lambda route: held.append(route))
            page.fill("#page-chat .chat-find-input", "routine note 3")
            page.wait_for_function("() => document.querySelector('#page-chat .chat-find')?.hasAttribute('data-searching')")
            page.wait_for_function("() => document.querySelector('#page-chat .chat-find-note')?.textContent.length > 0")
            assert _search_frame(page, "#page-chat", alone=False)["searching"]
            _search_both_themes(page, shots / f"{name}-loading.png")
            for route in held:
                route.abort()  # an offline read: the sheet says the history could not be searched, with Try again
            page.unroute("**/api/chat/search?*")
            page.wait_for_function("""() => document.querySelector('#page-chat .chat-find-more')?.textContent === 'Try again'""")
            failed = _search_frame(page, "#page-chat", alone=False)
            assert failed["note"].startswith("Saved history could not be searched."), failed
            _search_both_themes(page, shots / f"{name}-error.png")
            page.locator("#page-chat .chat-find-more").click()
            page.wait_for_function("""() => {
                const find = document.querySelector('#page-chat .chat-find');
                return !find.hasAttribute('data-searching') && find.querySelectorAll('.chat-find-result').length > 0;
            }""", timeout=30_000)
            _search_frame(page, "#page-chat", alone=False)
            page.fill("#page-chat .chat-find-input", "")
            page.wait_for_function("() => document.querySelector('#page-chat .chat-find-results')?.hidden")
            _search_frame(page, "#page-chat", alone=True)  # emptied words give the capsule back
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        finally:
            browser.close()
    assert not errors, errors
