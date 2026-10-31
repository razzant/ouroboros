"""Notes between works in real rooms, using the isolated dirty-candidate server (docs/DESIGN.md "Replies").

Authored fixtures cover the long notes, mixed sender runs, a run of eight unnamed notes and a Main stack addressing two
works, without depending on a private owner's history export. The geometry of the stack is measured as painted: nothing
above the front card, the ledges inside the card's box below its face, the fill a step back from a reply's while the
work card keeps its own, the column's motion never past the feed's end, reduced motion travelling nothing.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from ouroboros import dialogue_store as ds
from ouroboros.chat_references import reference_from_row
from ouroboros.owner_relay import publish_relay, write_relay
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import write_task_result
from tests._shared import producer_evidence_dir
from tests.test_chat_history_paging_browser import _FRAMES, _open, _open_project, MAIN
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

QUOTE = "\n".join([
    "Please keep the existing environment and check the dependency versions before changing the setup.",
    "The Project room should explain what was verified and where the remaining uncertainty comes from.",
    "Include the exact browser sizes, check the phone layout, and keep the note readable in both themes.",
    "The final handoff needs the relevant logs, screenshots and the commands used to reproduce the checks.",
    "Do not publish until I have read the result and compared it with the approved design.",
])
NOTE = "\n\n".join([
    "The receiving work should verify the candidate against the retained baseline. The dependency check completed; the browser comparison still needs both engines.",
    "Use the existing Project room and keep its saved history. The owner asked for the full quoted instructions to remain available alongside this handoff.",
    "Check the narrow layout with a long sender name and a note containing multiple paragraphs. Open the deck, expand this note, then fold it back without losing the row's address or Reply and Copy actions.",
    "Record the exact checks and preserve the evidence in the candidate's QA folder. Return a concise report after all isolated processes have stopped.",
])
UNNAMED = [
    "Found and read the primary text of the earlier failure report. Its saved repair is a retained wait from the previous pause; the restart then refuses the return and cancels.",
    "Main is taking apart the hold and the manual return. Does your nanny also see a consumed grant after the restart while the leaf is alive? Only the current fact, no new repair.",
    "Clarification: the shared research failed before the provider answered; I am now in this room myself, the read-only reviewer beside me.",
    "My recovery scope is this owner request only, not cancel or resume: the live leaf cannot be left without exact control.",
    "The header child refused a single resume with a spent grant. After the fresh terminal read the carrier was cancelled and one exact target transfer went through.",
    "The adjacent repair does not cover the stale grant. Using the sources passed on earlier for the research; nothing of yours is touched.",
    "Seven scenarios passed through the real transport and the isolated engine; one missed transition remains, the reply to a message sent before the start.",
    "Removing my temporary rule about starting sessions from the root. The first request of a new helper goes through again; new helpers can be started as usual.",
]
TARGET_A = {"task_id": "work-resume", "title": "Resume the works and watch the cache"}
TARGET_B = {"task_id": "work-statuses", "title": "Interface statuses without eternal plates"}


def seed_relay_notes(root):
    room = create_project(root, 'relay-receiver', name='Relay notes', origin='owner_ui')
    sender = create_project(root, 'relay-sender', name='Environment checks and dependency verification', origin='owner_ui')
    other = create_project(root, 'relay-other', name='Release follow-up', origin='owner_ui')
    start = datetime(2026, 10, 22, 11, tzinfo=timezone.utc)
    index = 0

    def say(text, chat_id=room['chat_id'], **extra):
        nonlocal index
        index += 1
        row = {'chat_id': chat_id, 'ts': (start + timedelta(minutes=index)).isoformat(),
               'direction': 'out', 'text': text, **extra}
        ds.append(root, 'chat', row, feed=True)
        return row

    def note(key, text, sender_facts=None, *, quotes=(), summary='', target=None, chat_id=room['chat_id']):
        relay = {'relay_id': key, 'quotes': list(quotes), 'summary': summary}
        if sender_facts:
            relay['sender'] = sender_facts
        if target:
            relay['target'] = target
        return say(text, chat_id, type='relay_entry', client_message_id=key, relay=relay)

    source = {'chat_id': 1, 'label': 'Main', 'ts': start.isoformat()}
    quote = {'text': QUOTE, 'source': source}
    note('lone', NOTE, {'name': 'Main', 'chat_id': 1}, summary='The environment check is ready for the next work.')
    say('Please keep the results together in this room.', direction='in')
    first = note('quoted', NOTE, sender, quotes=[quote, {'text': 'Keep the attached log with the handoff.',
                 'source': source, 'attachments': [{'name': 'dependency-check.txt'}]}],
                 summary='The owner asked for both engines and the full verification record.')
    note('main', 'The runtime check is complete; the candidate can proceed to visual verification.', {'name': 'Main', 'chat_id': 1})
    note('legacy', 'An older work left this note before sender rooms were recorded.', {'task_id': 'old-unbound'}, quotes=[quote])
    front = note('front', NOTE, other, summary='The next work can compare the browser results and finish the handoff.')
    # A real reply reference exercises the room's reveal path to a tucked relay row.
    say('Return to the earlier verification instructions.', direction='in', reply_to=reference_from_row(first))
    # The long series: eight notes from a work whose room was never recorded, without the teller's line, at the
    # room's newest end (where a stack fans out at the bottom of the room).
    for i, text in enumerate(UNNAMED):
        note(f'unnamed-{i}', text, {'task_id': f'old-{i}'})
    # Main: a Project's note and a Main work's note, addressed to two different works, at Main's newest end. These two
    # come through the real publisher (`tell_task` → mailbox entry → `publish_relay`), which stamps the sender's room
    # and the receiving work's display name itself and saves the row the way a live note is saved.
    say('What did the works in the other rooms find?', 1, direction='in')
    for target in (TARGET_A, TARGET_B):
        write_task_result(root, target['task_id'], 'running', chat_id=1, suggested_name=target['title'],
                          text=f"The owner asked: {target['title'].lower()}, in full, as a task description.")
    write_task_result(root, 'teller-project', 'running', chat_id=sender['chat_id'], text='A work in the sender room.')
    bind_task_to_project(root, 'teller-project', sender['id'], origin={'absent': 'system'})
    write_task_result(root, 'teller-main', 'running', chat_id=1, text='A Main work writing to another.')
    for key, target, teller, text, summary in (
            ('to-a', TARGET_A, 'teller-project', 'The dependency check is complete; the cache stays warm after the resume.',
             'The sender room reports the cache stays warm after the resume.'),
            ('to-b', TARGET_B, 'teller-main', 'Resume the statuses work and keep the plates out of the room.',
             'A new instruction after the restart: resume and keep the room quiet.')):
        entry, _ = write_relay(root, target['task_id'], relay_id=key, note=text, quotes=[], chat_id=1,
                               teller_task_id=teller, summary=summary)
        publish_relay(root, target['task_id'], entry, None)
    return room, sender, first, front


def _snapshot(page, root, name, feed, facts):
    dest = producer_evidence_dir(root, 'relay-note-deck')
    page.screenshot(path=str(dest / f'{name}.png'), animations='disabled', scale='css')
    measured = page.locator(feed).evaluate("""feed => {
        const composer = feed.closest('.chat-instance-panel, .page')?.querySelector('.chat-input-area');
        return { viewport: [innerWidth, innerHeight], pageWidth: document.documentElement.scrollWidth,
            feedWidth: feed.clientWidth, feedScrollWidth: feed.scrollWidth,
            cards: [...feed.querySelectorAll('.chat-relay-note')].filter(n => n.getClientRects().length).map(n => {
                const r = n.getBoundingClientRect();
                return {text: n.innerText, x: r.x, width: r.width, raw: n.dataset.raw, deck: n.dataset.deck || '',
                    controls: [...n.querySelectorAll('button')].filter(b => b.getClientRects().length).map(b => ({
                        text: b.textContent, label: b.getAttribute('aria-label'), width: b.getBoundingClientRect().width,
                        height: b.getBoundingClientRect().height}))};
            }), composerTop: composer?.getBoundingClientRect().top };
    }""")
    assert measured['pageWidth'] <= measured['viewport'][0], measured
    assert measured['feedScrollWidth'] <= measured['feedWidth'] + 1, measured
    facts[name] = measured
    (dest / f'{name}.json').write_text(json.dumps(measured, indent=2))


# The stack's geometry as painted: the row, its face, its ledges and everything else inside it.
_GEOMETRY = """n => {
    const box = (el) => { const r = el.getBoundingClientRect(); return {top: r.top, bottom: r.bottom, left: r.left, right: r.right, width: r.width}; };
    const prev = (() => { let p = n.previousElementSibling; while (p && !p.getClientRects().length) p = p.previousElementSibling; return p; })();
    const face = n.querySelector(':scope > .chat-relay-face');
    const ledges = [...n.querySelectorAll(':scope > .chat-relay-ledge')].filter(l => getComputedStyle(l).display !== 'none')
        .sort((a, b) => Number(a.dataset.level) - Number(b.dataset.level));
    const parts = [...n.querySelectorAll('*')].filter(el => el.getClientRects().length).map(el => el.getBoundingClientRect().top);
    return { row: box(n), prev: prev ? box(prev) : null, face: box(face), ledges: ledges.map(box),
        faceShadow: getComputedStyle(face).boxShadow, faceFill: getComputedStyle(face).backgroundColor,
        ledgeFills: ledges.map(l => getComputedStyle(l).backgroundColor),
        rowFill: getComputedStyle(n).backgroundColor, rowShadow: getComputedStyle(n).boxShadow,
        topmost: Math.min(...parts), gap: parseFloat(getComputedStyle(n.parentElement).rowGap || getComputedStyle(n.parentElement).gap) };
}"""


# Every card's starting translation as the deck asks the engine for it, read whenever the test looks (a finished
# animation drops out of getAnimations(), and a loaded machine can finish it before the read).
_RECORD_STARTS = r"""(() => {
    const animate = Element.prototype.animate;
    window.__relayStarts = [];
    Element.prototype.animate = function (frames, options) {
        const first = Array.isArray(frames) ? frames[0] : null;
        const m = this.classList?.contains('chat-relay-note') && first && /translateY\((-?[\d.]+)px\)/.exec(String(first.transform || ''));
        if (m) window.__relayStarts.push(Number(m[1]));
        return animate.call(this, frames, options);
    };
})()"""


def _pixel(page, x, y):
    """The painted colour at a viewport point (device scale 2: the point's own pixel)."""
    from PIL import Image
    shot = page.screenshot(clip={'x': x, 'y': y, 'width': 1, 'height': 1}, animations='disabled', scale='css')
    return Image.open(io.BytesIO(shot)).convert('RGB').getpixel((0, 0))


def _check_stack_geometry(geometry, edges):
    row, face, ledges = geometry['row'], geometry['face'], geometry['ledges']
    # Nothing paints above the front card: no part of the row starts above the row, and the row keeps the feed's gap.
    assert geometry['topmost'] >= row['top'] - 0.5, geometry
    if geometry['prev']:
        assert abs(row['top'] - geometry['prev']['bottom'] - geometry['gap']) < 1, geometry
    assert geometry['rowFill'] in ('rgba(0, 0, 0, 0)', 'transparent') and geometry['rowShadow'] == 'none', geometry
    assert 'px' in geometry['faceShadow'] and geometry['faceShadow'] != 'none', 'the face carries the rest shadow'
    # The face stops above the reserve; each ledge lies inside the row, below the face, 12px narrower per side.
    assert abs(face['top'] - row['top']) < 0.5 and abs(face['bottom'] - (row['bottom'] - 8 * edges)) < 0.5, geometry
    assert len(ledges) == edges, geometry
    for level, ledge in enumerate(ledges, 1):
        assert ledge['top'] >= row['top'] and ledge['bottom'] <= row['bottom'] + 0.5, geometry
        assert abs(ledge['bottom'] - (face['bottom'] + 8 * level)) < 0.5, geometry
        assert abs(ledge['left'] - (face['left'] + 12 * level)) < 0.5 and abs(face['right'] - ledge['right'] - 12 * level) < 0.5, geometry


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('viewport', [(1100, 750), (390, 844)], ids=['desktop', 'phone'])
def test_relay_notes_stack_fans_out_and_folds(direct_server_with_data, browser_engine, viewport, tmp_path):
    from playwright.sync_api import sync_playwright

    room, sender, first, front = seed_relay_notes(direct_server_with_data['data_dir'])
    feed = f'#pchat-{room["id"]}-messages'
    prefix = f'{browser_engine}-{viewport[0]}'
    phone = viewport[0] < 600
    errors, facts = [], {}
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        # The phone is a touch device: a coarse pointer, where the controls' hit areas reach the finger's minimum.
        page = browser.new_page(viewport={'width': viewport[0], 'height': viewport[1]}, device_scale_factor=2, has_touch=phone)
        page.add_init_script(_RECORD_STARTS)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda message: errors.append({'text': message.text, 'location': message.location}) if message.type == 'error' else None)
        page.on('response', lambda response: facts.setdefault('http_errors', []).append({'url': response.url, 'status': response.status}) if response.status >= 400 else None)
        try:
            response = page.request.post(direct_server_with_data['url'] + '/api/ui/preferences',
                                         data={'project_panel_width': 580, 'sidebar_width': 306})
            assert response.ok
            _open(page, direct_server_with_data['url'])
            # --- Main: a stack of two notes addressed to two works, the front one from another Main work. ---
            main_cards = page.locator(f'{MAIN} .chat-relay-note')
            assert main_cards.count() == 2
            main_front = main_cards.nth(1)
            assert main_front.locator('.chat-relay-from').inner_text() == 'Another Main work'
            assert main_front.locator('.chat-relay-to').inner_text() == '2 works'
            assert main_front.locator('.chat-relay-rooms').inner_text() == 'and 1 more room'
            assert main_front.locator('.chat-relay-act').inner_text() == '2 notes'
            assert main_cards.nth(0).get_attribute('data-deck-tucked') == '1'
            main_cards.nth(0).locator('[data-intent="open-project"]').wait_for(state='attached')
            main_front.scroll_into_view_if_needed()
            page.evaluate(_FRAMES)
            _snapshot(page, tmp_path, f'{prefix}-dark-main-closed', MAIN, facts)
            main_front.locator('.chat-relay-act').click()
            assert main_cards.nth(0).get_attribute('data-deck-first') == '1'
            assert main_cards.nth(0).locator('.chat-relay-to').inner_text() == TARGET_A['title'], 'the display name the host stamped, not the task text'
            assert main_front.locator('.chat-relay-to').inner_text() == TARGET_B['title']
            assert main_cards.nth(0).locator('[data-intent="open-project"]').is_visible(), 'the Project capsule stands in the address'
            assert all(to.evaluate('n => n.scrollWidth <= n.clientWidth + 1') for to in main_cards.locator('.chat-relay-to').all()), \
                'each receiver shows its whole name, on its own line below a long capsule if need be'
            page.evaluate(_FRAMES)
            _snapshot(page, tmp_path, f'{prefix}-dark-main-open', MAIN, facts)
            main_cards.nth(0).locator('.chat-relay-act').click()
            assert main_front.get_attribute('data-deck') == 'closed'
            # --- The Project room: a lone note, a stack of four with his words, and a stack of eight at the end. ---
            _open_project(page, room)
            cards = page.locator(f'{feed} .chat-relay-note')
            assert cards.count() == 13
            latest = cards.nth(4)
            latest.locator('[data-intent="open-project"]').wait_for()
            assert latest.locator('.chat-relay-words-inside').inner_text() == 'Your words in 2 earlier notes'
            assert latest.locator('.chat-relay-act').inner_text() == '4 notes'
            assert latest.locator('.chat-relay-act').get_attribute('aria-label') == 'Show 4 notes'
            assert latest.locator('.chat-relay-to').inner_text() == 'this work'
            assert cards.nth(1).get_attribute('data-deck-tucked') == '1'
            assert cards.nth(0).get_attribute('data-deck') is None
            assert cards.nth(0).locator('.chat-relay-act').is_hidden(), 'a card on its own has no capsule'
            assert cards.nth(0).evaluate("n => n.querySelectorAll(':scope > .chat-relay-ledge').length") == 2
            assert cards.nth(0).evaluate("n => [...n.querySelectorAll(':scope > .chat-relay-ledge')].every(l => getComputedStyle(l).display === 'none')")
            assert page.locator(f'{feed} .chat-relay-tile, {feed} .chat-clamp-toggle, {feed} .chat-relay-deck-top').count() == 0
            eight = cards.nth(12)
            assert eight.locator('.chat-relay-act').inner_text() == '8 notes'
            assert eight.locator('.chat-relay-from').inner_text() == 'Another work'
            assert eight.get_attribute('data-deck-edges') == '2'
            assert cards.nth(5).evaluate("n => n.hidden") is False, 'a tucked note is never hidden: a jump still lands on it'
            dest = producer_evidence_dir(tmp_path, 'relay-note-deck')
            latest.screenshot(path=str(dest / f'{prefix}-card@2x.png'), animations='disabled')
            # The stack as painted, in both themes: the layers' geometry and the material a step back from a reply's.
            for theme in ('dark', 'light'):
                page.evaluate('theme => document.documentElement.dataset.theme = theme', theme)
                latest.scroll_into_view_if_needed()
                page.evaluate(_FRAMES)
                geometry = latest.evaluate(_GEOMETRY)
                _check_stack_geometry(geometry, 2)
                facts[f'{prefix}-{theme}-geometry'] = geometry
                reply_fill = page.locator(f'{feed} .chat-bubble.assistant:not(.chat-relay-note)').first.evaluate('n => getComputedStyle(n).backgroundColor')
                assert geometry['faceFill'] != reply_fill, (theme, geometry['faceFill'], reply_fill)
                assert len(set(geometry['ledgeFills'] + [geometry['faceFill']])) == 3, 'each level is its own step'
                _snapshot(page, tmp_path, f'{prefix}-{theme}-collapsed', feed, facts)
            # In dark the work card keeps the voice's cherry: the note took a new token, not the card's.
            card_fill = page.locator(feed).evaluate("""feed => { const card = document.createElement('div'); card.className = 'chat-live-card';
                feed.append(card); const fill = getComputedStyle(card).backgroundColor; card.remove(); return fill; }""")
            page.evaluate("document.documentElement.dataset.theme = 'dark'")
            page.evaluate(_FRAMES)
            dark_reply = page.locator(f'{feed} .chat-bubble.assistant:not(.chat-relay-note)').first.evaluate('n => getComputedStyle(n).backgroundColor')
            assert dark_reply == 'rgb(118, 22, 43)', dark_reply
            dark_card = page.locator(feed).evaluate("""feed => { const card = document.createElement('div'); card.className = 'chat-live-card';
                feed.append(card); const fill = getComputedStyle(card).backgroundColor; card.remove(); return fill; }""")
            assert dark_card == dark_reply, (dark_card, dark_reply)
            assert card_fill != 'rgb(118, 22, 43)', 'the light card keeps its rose'
            # A frame while the width moves (lazy rows) keeps the ledges painted: they lie inside the row's own box.
            eight.scroll_into_view_if_needed()
            page.evaluate(_FRAMES)
            room_fill = page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--bg-primary').trim()")
            page.locator(feed).evaluate("feed => feed.setAttribute('data-resizing', '')")
            page.evaluate(_FRAMES)
            geometry = eight.evaluate(_GEOMETRY)
            _check_stack_geometry(geometry, 2)
            for ledge in geometry['ledges']:
                x, y = (ledge['left'] + ledge['right']) / 2, ledge['bottom'] - 3
                pixel = _pixel(page, x, y)
                assert pixel != (15, 15, 17), (ledge, pixel, room_fill)  # the room shows through nowhere a ledge lies
            page.locator(feed).evaluate("feed => feed.removeAttribute('data-resizing')")
            # A long press lifts the row and dims the feed through the face's shadow.
            eight.evaluate("n => { n.dataset.rowPress = 'lift'; }")
            page.evaluate(_FRAMES)
            page.wait_for_timeout(300)
            shadow = eight.locator(':scope > .chat-relay-face').evaluate('n => getComputedStyle(n).boxShadow')
            assert '0.45' in shadow or '0.24' in shadow, shadow  # the dim colour (--row-lift-dim) in the face's shadow
            feed_box = page.locator(feed).bounding_box()
            dimmed = _pixel(page, feed_box['x'] + 4, feed_box['y'] + 4)
            assert dimmed != (15, 15, 17), dimmed  # the room under the feed is dimmed, not bare
            eight.evaluate("n => { delete n.dataset.rowPress; }")
            page.evaluate(_FRAMES)
            # Fanning out at the feed's end: the column grows at once, nothing moves past the feed's end while the cards
            # glide, and the first card of a column that does not fit lands under the header (owner question 6, A).
            page.locator(feed).evaluate('feed => { feed.scrollTop = feed.scrollHeight; }')
            page.evaluate(_FRAMES)
            page.evaluate('window.__relayStarts.length = 0')
            eight.locator('.chat-relay-act').focus()
            page.keyboard.press('Enter')
            # The cards fan downward from where the stack lay, even though placing the column scrolled the feed: none
            # sets out from below its own place.
            starts = page.evaluate('window.__relayStarts.slice()')
            assert starts and max(starts) <= 0.5, starts
            heights = page.locator(feed).evaluate("""feed => new Promise(done => { const hs = []; const t0 = performance.now();
                (function tick() { hs.push(feed.scrollHeight); if (performance.now() - t0 < 500) requestAnimationFrame(tick); else done(hs); })(); })""")
            assert max(heights) <= heights[-1] + 1, heights
            first_of_eight = cards.nth(5)
            assert first_of_eight.get_attribute('data-deck-first') == '1'
            assert first_of_eight.locator('.chat-relay-act').inner_text() == 'Show less'
            assert first_of_eight.locator('.chat-relay-act').evaluate('n => n === document.activeElement'), 'the keyboard opener had focus: Show less takes it'
            band = page.locator(feed).evaluate("feed => { const r = feed.getBoundingClientRect(); return [r.top, r.bottom]; }")
            head = first_of_eight.locator('.chat-relay-act').bounding_box()
            assert band[0] <= head['y'] and head['y'] + head['height'] <= band[1], (band, head)
            assert all(first_of_eight.locator('.chat-relay-to').nth(0).inner_text() == 'this work' for _ in [0])
            _snapshot(page, tmp_path, f'{prefix}-dark-eight-open', feed, facts)
            # Folding from the keyboard returns the focus to the capsule (a pointer press moves no focus: WebKit does
            # not focus a pressed button, Chromium does; the rule follows the control that had it).
            first_of_eight.locator('.chat-relay-act').focus()
            page.keyboard.press('Enter')
            assert eight.get_attribute('data-deck') == 'closed'
            assert eight.locator('.chat-relay-act').evaluate('n => n === document.activeElement'), 'folded, the capsule has the focus again'
            page.evaluate(_FRAMES)
            assert page.locator(feed).evaluate('feed => feed.scrollHeight - feed.scrollTop - feed.clientHeight') <= 48, 'folding follows the bottom again'
            # Reduced motion: the column lands with nothing travelling.
            page.emulate_media(reduced_motion='reduce')
            latest.scroll_into_view_if_needed()
            latest.locator('.chat-relay-act').click()
            transforms = page.locator(feed).evaluate("""feed => [...feed.querySelectorAll('.chat-relay-note[data-deck="open"]')]
                .map(n => getComputedStyle(n).transform + '|' + n.getAnimations().filter(a => a.playState === 'running' && JSON.stringify(a.effect.getKeyframes()).includes('transform')).length)""")
            assert transforms and all(t == 'none|0' for t in transforms), transforms
            page.emulate_media(reduced_motion='no-preference')
            earlier = cards.nth(1)
            assert earlier.get_attribute('data-deck') == 'open'
            assert latest.locator('.chat-relay-words-inside').count() == 0
            assert cards.nth(3).locator('[data-intent="open-project"]').count() == 0
            assert cards.nth(3).locator('.chat-relay-from').inner_text() == 'Another work'
            assert cards.nth(3).locator('.chat-relay-to').inner_text() == 'this work'
            earlier.evaluate("n => { const f = n.closest('.chat-messages'); f.scrollTop += n.getBoundingClientRect().top - f.getBoundingClientRect().top - 110; }")
            page.evaluate(_FRAMES)
            _snapshot(page, tmp_path, f'{prefix}-dark-fanned', feed, facts)
            # `more`: the note as written, its actions and every quotation whole.
            excerpt = earlier.locator('.chat-quote-text').first
            assert len(excerpt.inner_text()) < len(QUOTE)
            more = earlier.locator('.chat-relay-more')
            more.focus()
            page.keyboard.press('Enter')
            assert more.inner_text() == 'less'
            assert more.get_attribute('aria-expanded') == 'true'
            assert more.get_attribute('aria-label').startswith('Collapse note from ')
            assert more.evaluate('n => n === document.activeElement')
            assert excerpt.inner_text() == QUOTE
            assert earlier.locator('.chat-quote-text').nth(1).inner_text() == 'Keep the attached log with the handoff.'
            assert earlier.locator('.chat-quote-files').inner_text() == 'Attached: dependency-check.txt'
            assert earlier.locator('.message').inner_text() == NOTE
            assert earlier.locator('.chat-message-reply').is_visible()
            assert earlier.locator('.chat-message-copy').is_visible()
            for theme in ('dark', 'light'):
                page.evaluate('theme => document.documentElement.dataset.theme = theme', theme)
                earlier.evaluate("n => { const f = n.closest('.chat-messages'); f.scrollTop += n.getBoundingClientRect().top - f.getBoundingClientRect().top - 110; }")
                page.evaluate(_FRAMES)
                _snapshot(page, tmp_path, f'{prefix}-{theme}-more-start', feed, facts)
                more.scroll_into_view_if_needed()
                page.evaluate(_FRAMES)
                _snapshot(page, tmp_path, f'{prefix}-{theme}-more-end', feed, facts)
            more.click()
            assert earlier.get_attribute('data-raw') == '0'
            assert more.inner_text() == 'more'
            assert len(excerpt.inner_text()) < len(QUOTE)
            earlier.locator('.chat-relay-act').click()
            assert latest.get_attribute('data-deck') == 'closed'
            # Exact row navigation unfolds the same row before focusing it; the ring is on the face, not around the box.
            page.locator(f'{feed} .chat-quote-main').click()
            page.wait_for_function("selector => { const n = document.querySelector(selector); return n?.dataset.deck === 'open' && n === document.activeElement; }", timeout=5000,
                                   arg=f'{feed} .chat-relay-note.is-quote-target')
            # Touch targets: on the phone every control of the stack carries the hit area and, where the engine emulates
            # the coarse pointer, reaches the finger's minimum through it.
            if phone:
                coarse = page.evaluate("matchMedia('(pointer: coarse)').matches")
                controls = page.locator(feed).evaluate("""feed => [...feed.querySelectorAll('.chat-relay-note:not([data-deck-tucked]) button')]
                    .filter(b => b.getClientRects().length).map(b => { const r = b.getBoundingClientRect();
                        const after = getComputedStyle(b, '::after'); const inset = parseFloat(after.top) || 0;
                        return {text: b.textContent.trim(), hit: b.classList.contains('ui-hit-area') || b.classList.contains('chat-message-action'),
                            h: r.height - 2 * Math.min(0, inset), pos: after.position}; })""")
                facts[f'{prefix}-touch'] = {'coarse': coarse, 'controls': controls}
                assert controls and all(c['hit'] for c in controls), controls
                if coarse:
                    assert all(c['h'] >= 43.5 for c in controls), controls
            # The sender reference really opens the source Project; no synthetic click handler.
            earlier.locator('[data-intent="open-project"]').click()
            page.locator(f'#pchat-{sender["id"]}-messages').wait_for(state='visible')
            # The isolated gateway has no native Claudexor maintenance bridge. Preserve its
            # expected 503 in evidence; all other console, script and HTTP errors fail this check.
            maintenance = '/api/claudexor/maintenance/harnesses'
            unexpected = [error for error in errors if not (
                isinstance(error, dict) and urlsplit(error['location'].get('url', '')).path == maintenance
                and '503' in error['text'])]
            assert not unexpected, unexpected
            assert not [error for error in facts.get('http_errors', []) if not (
                urlsplit(error['url']).path == maintenance and error['status'] == 503)], facts
        finally:
            dest = producer_evidence_dir(tmp_path, 'relay-note-deck')
            page.screenshot(path=str(dest / f'{prefix}-last.png'), animations='disabled')
            (dest / f'{prefix}-last.html').write_text(page.content())
            (dest / f'{prefix}-errors.json').write_text(json.dumps(errors))
            (dest / f'{prefix}-facts.json').write_text(json.dumps(facts, indent=2))
            browser.close()


_BAND = """feed => {
    const panel = feed.closest('.project-panel');
    const header = panel.querySelector('.project-panel-bar'), composer = panel.querySelector('.chat-input-area');
    const cards = [...feed.querySelectorAll('.chat-relay-note')].map(n => n.getBoundingClientRect());
    const act = feed.querySelector('.chat-relay-note[data-deck-first="1"] .chat-relay-act');
    return { headerBottom: header.getBoundingClientRect().bottom, composerTop: composer.getBoundingClientRect().top,
             feedTop: feed.getBoundingClientRect().top, feedBottom: feed.getBoundingClientRect().bottom,
             column: cards[cards.length - 1].bottom - cards[0].top, showLessTop: act ? act.getBoundingClientRect().top : null };
}"""


@pytest.mark.parametrize('browser_engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('viewport', [(1100, 750), (390, 844)], ids=['desktop', 'phone'])
def test_a_column_hidden_only_by_the_composer_comes_under_the_header(direct_server_with_data, browser_engine, viewport):
    """Five ordinary notes fit the feed's box but not the band above the composer: the room lends the deck that band,
    so the column's first card and its Show less come under the header (owner question 6, A)."""
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data['data_dir']
    room = create_project(root, 'relay-five', name='Five notes', origin='owner_ui')
    start = datetime(2026, 10, 22, 11, tzinfo=timezone.utc)
    ds.append(root, 'chat', {'chat_id': room['chat_id'], 'ts': start.isoformat(), 'direction': 'in',
                             'text': 'What did the other works find?'}, feed=True)
    for i, text in enumerate(UNNAMED[:5]):
        ds.append(root, 'chat', {'chat_id': room['chat_id'], 'ts': (start + timedelta(minutes=i + 1)).isoformat(),
                                 'direction': 'out', 'text': text, 'type': 'relay_entry', 'client_message_id': f'five-{i}',
                                 'relay': {'relay_id': f'five-{i}', 'quotes': [], 'sender': {'task_id': f'old-{i}'}}},
                  feed=True)
    feed = f'#pchat-{room["id"]}-messages'
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={'width': viewport[0], 'height': viewport[1]}, has_touch=viewport[0] < 600)
        try:
            assert page.request.post(direct_server_with_data['url'] + '/api/ui/preferences',
                                     data={'project_panel_width': 580, 'sidebar_width': 306}).ok
            _open(page, direct_server_with_data['url'])
            _open_project(page, room)
            cards = page.locator(f'{feed} .chat-relay-note')
            assert cards.count() == 5
            page.locator(feed).evaluate('feed => { feed.scrollTop = feed.scrollHeight; }')
            page.evaluate(_FRAMES)
            cards.nth(4).locator('.chat-relay-act').click()
            page.wait_for_timeout(900)
            page.evaluate(_FRAMES)
            band = page.locator(feed).evaluate(_BAND)
            seen, box = band['composerTop'] - band['headerBottom'], band['feedBottom'] - band['feedTop']
            assert seen < band['column'] < box, band  # taller than what a reader sees, shorter than the feed itself
            assert band['showLessTop'] is not None and band['showLessTop'] >= band['headerBottom'] - 0.5, band
        finally:
            browser.close()
