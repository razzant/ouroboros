"""A room's feed keeps the reading rules of docs/DESIGN.md through recorded sessions, in real time.

Each session (tests/fixtures/feed_sessions/*.json) is played on the production page against the real history endpoint
over the session's synthetic store (tests/_feed_sessions.py): openings, replies, older pages, photos, width, height,
the phone keyboard, wheel, keys and touch, in the order a reader meets them. After every event the rules it touches are
checked (``Reader``), and a session reports every rule it broke with the feed's facts before and after each event.

Three engines: Chromium, Playwright's WebKit, and that WebKit with ``overflow-anchor: none`` on the feeds, as the
macOS window's WKWebView (macOS 26) has no scroll anchoring: the feed holds its place itself. Real input where the
engine has it (wheel, keys, mouse, Chromium's touch); WebKit has no touch input to drive, so a swipe there is the
touch events and the scroll a finger makes, written by the page only where the engine would scroll (a move nobody
cancelled, on a feed that scrolls). Uses the static server and the real history endpoint; never starts the runtime,
never reads live data.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from tests._feed_sessions import FACTS, INIT, NO_ANCHOR, REST, Reader, Stand, load, photo_url, sessions
from tests._phone_room_shared import VIEWPORT
from tests._shared import producer_evidence_dir
from tests.ui_chat_viewport_smoke import wait_drawer_closed

pytestmark = [pytest.mark.ui_browser]
ENGINES = ['chromium', 'webkit', 'webkit-no-anchor']

# Frames from ↓ until the feed stands at its retained end: one to handle the click, one to write the place. Sampled
# from before the press (``pressed``: the samples taken before its click), so the frames right after it are seen.
LATEST = """sel => {
    const feed = document.querySelector(sel), run = window.__latest = {gaps: [], pressed: null};
    feed.parentElement.addEventListener('click', () => { run.pressed = run.gaps.length; }, {capture: true, once: true});
    const tick = () => {
        run.gaps.push(Math.round(feed.scrollHeight - feed.scrollTop - feed.clientHeight));
        if (run.pressed === null ? run.gaps.length < 600 : run.gaps.length - run.pressed < 12) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}"""
# A finger on WebKit, which Playwright cannot drive: the touch events on the feed and the scroll the finger makes, only
# where the engine would make it: a move no listener cancelled, on a feed that scrolls. Each step lands between frames,
# as an engine's touch scroll lands before a frame's callbacks; written inside a frame, a resize in the same frame (the
# phone composer folding) put the place of the frame before back, and 11 of 18 steps were lost. Returns the moves held.
TOUCH = """([sel, dy]) => new Promise(done => {
    const feed = document.querySelector(sel), rect = feed.getBoundingClientRect();
    const x = rect.left + rect.width * 0.6, y0 = rect.top + rect.height * (dy > 0 ? 0.7 : 0.3);
    const send = (type, y) => {
        const touches = type === 'touchend' ? [] : [{clientX: x, clientY: y, identifier: 1, target: feed}];
        const event = new Event(type, {bubbles: true, cancelable: true});
        Object.defineProperty(event, 'touches', {value: touches});
        Object.defineProperty(event, 'changedTouches', {value: [{clientX: x, clientY: y, identifier: 1, target: feed}]});
        return feed.dispatchEvent(event);
    };
    const steps = 18, held = {cancelled: 0, unscrollable: 0};
    let step = 0;
    send('touchstart', y0);
    const move = () => {
        step += 1;
        const go = send('touchmove', y0 - dy * step / steps);
        const scrolls = /auto|scroll|overlay/.test(getComputedStyle(feed).overflowY);
        if (!go) held.cancelled += 1;
        else if (!scrolls) held.unscrollable += 1;
        else feed.scrollTop += dy / steps;
        if (step < steps) setTimeout(move, 16);
        else { send('touchend', y0 - dy); done(held); }
    };
    setTimeout(move, 16);
})"""


def _require_browser():
    if os.environ.get('OUROBOROS_RUN_UI_SMOKE') != '1':
        pytest.skip('Set OUROBOROS_RUN_UI_SMOKE=1 to run the browser flow')
    return pytest.importorskip('playwright.sync_api')


class Player:
    """Plays one session's events on one page and hands each one's facts to the reader."""

    def __init__(self, page, stand: Stand, session: dict, engine: str):
        self.page, self.stand, self.session, self.engine = page, stand, session, engine
        self.reader = Reader(session)
        self.trace: list[dict] = []
        self.started = False
        self.phone = bool(session.get('phone'))
        self.receipted = 0  # the stand's receipts already handed to the reader

    def feed(self, room: str) -> str:
        if room == 'main':
            return '#chat-messages'
        return f'[id="panel-pchat-{self.stand.rooms[room]["project_id"]}"] .chat-messages'

    def facts(self, room: str, named: str = '', photo: str = ''):
        place = (self.reader.place.get(room) or {}).get('id')
        reply = self.stand.newest_reply(room) if room else ''
        return self.page.evaluate(FACTS, [self.feed(room), place, named, reply, photo]) if room else None

    def marked(self, room: str, unread: bool) -> bool:
        """Whether the room's row in the list comes to show (``unread``) or lose its unread mark, the row's own
        attribute (tests/_room_browser_shared.py ``_unread``), within the state poll's reach."""
        row = f'.nav-project-row[data-project-id="{self.stand.rooms[room]["project_id"]}"]'
        try:
            self.page.locator(f'{row}[data-unread]').wait_for(state='attached' if unread else 'detached', timeout=5000)
            return True
        except Exception:  # Playwright's TimeoutError: the mark did not come (or did not go)
            return False

    def read(self, room: str, after: dict | None) -> bool:
        """Whether the room shown has its read receipt at its revision: waited for when its newest reply shows (the
        client acknowledges after the paint), observed for a moment when it does not (an early receipt would show by
        then). A room not shown is held by every receipt the stand records (``Reader.receipts``), whenever it comes."""
        revision = self.stand.revision(room)
        for _ in range(30 if after and after.get('replyShown') else 5):
            if self.stand.seen.get(room, 0) >= revision:
                return True
            self.page.wait_for_timeout(100)
        return False

    def rest(self, room: str, held: bool = False):
        self.page.evaluate(REST, [self.feed(room), held])

    def play(self):
        for event in self.session['events']:
            photo = f'Photo {event["image"]}' if 'image' in event else ''
            before = self.facts(self.reader.room, photo=photo)
            seen = dict(self.stand.seen)
            problems = len(self.reader.problems)
            kind = next(iter(event))
            extra = getattr(self, f'_{kind}')(event)
            room = self.reader.room
            after = self.facts(room, (extra or {}).get('named', ''), photo)
            # A room with something unread is read once its reader is shown its newest reply, and not before: checked
            # where that can change (an opening, a reply here, ↓, a held read let go). A read room stays read, and
            # its row in the list keeps the unread mark exactly while the room is not read.
            if (kind in ('open', 'latest', 'release') or event.get('arrive') == room) \
                    and seen.get(room, 0) < self.stand.revision(room):
                extra = {**(extra or {}), 'read': self.read(room, after)}
                if room != 'main':
                    extra['marked'] = self.marked(room, not extra['read'])
            # A reply in a Project room not shown marks its row; no receipt reaches it while it is away (below).
            away = event.get('arrive')
            if away and away not in (room, 'main') and event.get('role') != 'owner':
                extra = {**(extra or {}), 'away': {'room': away, 'marked': self.marked(away, True)}}
            # Every receipt that came during the event, with the room shown as it came: a late one is still held.
            receipts = self.stand.receipts[self.receipted:]
            self.receipted += len(receipts)
            if receipts:
                extra = {**(extra or {}), 'receipts': receipts}
            self.reader.check(event, before, after, extra)
            self.trace.append({'event': event, 'extra': extra, 'before': _brief(before), 'after': _brief(after),
                               'problems': self.reader.problems[problems:]})
        late, problems = self.stand.receipts[self.receipted:], len(self.reader.problems)
        if late:  # a receipt that came after the last event's checks
            self.reader.receipts(late, None)
            self.trace.append({'receipts': late, 'problems': self.reader.problems[problems:]})

    # -- events ------------------------------------------------------------------------------------------------
    def _open(self, event):
        room = event['open']
        if not self.started:
            self.started = True
            self.page.goto(self.stand.url + '/', wait_until='domcontentloaded')
            self.page.wait_for_function('() => window.__feedOpen && window.__feedOpen()')
            self.page.locator('#chat-messages[data-history-hydrated="true"]').wait_for(state='attached')
            if 'main' in self.session['rooms']:
                self.rest('main')
                if room != 'main':  # the app's start showed Main first: it was opened at its newest message
                    self.reader.room = 'main'
                    self.reader.check({'open': 'main'}, None, self.facts('main'))
        self._show(room)
        if room == 'main':
            if self.reader.room and self.reader.room != 'main':
                self.page.locator('#project-panel-close').click()
                self.page.wait_for_function("() => !document.querySelector('#project-panel.open')")
        else:
            row = f'.nav-project-row[data-project-id="{self.stand.rooms[room]["project_id"]}"]'
            if self.phone:
                self.page.locator('[data-mobile-nav-toggle]:visible').first.click()
                self.page.wait_for_function("() => window.__ouroLayers?.top() === 'drawer'")
                self.page.locator(row).click()
                wait_drawer_closed(self.page)
            else:
                self.page.locator(row).evaluate('el => el.click()')
            self.page.locator(f'{self.feed(room)}[data-history-hydrated="true"]').wait_for(state='attached')
            self.page.wait_for_function("() => !document.querySelector('.chat-instance-panel[data-leaving]')")
        self.reader.room = room
        self.rest(room)
        # An unread room's opening lands on its newest arrival (DESIGN "Unread").
        return {'named': self.stand.newest_reply(room)} if self.session['rooms'][room].get('unread') else None

    def _show(self, room: str):
        """The page shows ``room`` from the moment the reader turns to it. The Project room left may still be read as
        far as the newest reply it showed (a receipt in flight); a passage left unread, or a reply after, may not."""
        left, keeps = self.reader.room, 0
        if left and left not in ('main', room):
            facts = self.facts(left)
            if facts and (facts['replyShown'] or self.reader.following(facts)):
                keeps = self.stand.revision(left)
        self.stand.show(room, keeps)

    def _arrive(self, event):
        room = event['arrive']
        frame = self.stand.arrive(room, count=event.get('words', 40), role=event.get('role', 'assistant'))
        self.page.evaluate('frame => window.__feedPush(frame)', frame)
        self.page.evaluate("() => window.__feedPush({type: 'projects_changed'})")
        if room == self.reader.room:
            self.page.wait_for_function('([sel, text]) => document.querySelector(sel)?.textContent.includes(text)',
                                        arg=[self.feed(room), frame['content'][:60]])
        self.rest(self.reader.room)
        return {'text': frame['content'][:40]}

    def _box(self):
        box = self.page.locator(self.feed(self.reader.room)).bounding_box()
        return box['x'] + box['width'] * 0.6, box['y'] + box['height'] * 0.5

    def _wheel(self, event):
        x, y = self._box()
        self.page.mouse.move(x, y)
        steps = max(1, event.get('steps', abs(event['wheel']) // 300))
        for _ in range(steps):
            self.page.mouse.wheel(0, event['wheel'] / steps)
            self.page.wait_for_timeout(40)
        self.rest(self.reader.room)

    def _swipe(self, event):
        dy = event['swipe']
        if self.engine == 'chromium':
            x, y = self._box()
            cdp = self.page.context.new_cdp_session(self.page)
            try:
                y0 = y + (120 if dy > 0 else -120)
                for i in range(19):
                    cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart' if i == 0 else 'touchMove',
                                                          'touchPoints': [{'x': x, 'y': y0 - dy * i / 18, 'id': 1}]})
                    self.page.wait_for_timeout(12)
                # The finger rests before it lifts, so the drag ends where it stops, as the page-event swipe does: a
                # fling would add a length set by the timing of the moves under load (144-496 px seen), not the gesture.
                self.page.wait_for_timeout(150)
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
            finally:
                cdp.detach()
        else:
            held = self.page.evaluate(TOUCH, [self.feed(self.reader.room), dy])
        self.rest(self.reader.room)
        return {'touch': 'cdp'} if self.engine == 'chromium' else {'touch': 'page events', 'held back': held}

    def _key(self, event):
        # Keys scroll the room the reader last pressed in: a press on the text being read, as the owner does.
        x, y = self._box()
        self.page.mouse.click(x, y)
        self.page.keyboard.press(event['key'])
        self.rest(self.reader.room)

    def _image(self, event):
        self.stand.release_photo(event['image'])
        self.page.wait_for_function('([sel, src]) => [...document.querySelectorAll(sel + " img.chat-photo")]'
                                    '.some(img => img.getAttribute("src") === src && img.complete && img.naturalHeight)',
                                    arg=[self.feed(self.reader.room), photo_url(event['image'])])
        self.rest(self.reader.room)

    def _viewport(self, event):
        width, height = event['viewport']
        self.page.set_viewport_size({'width': width, 'height': height})
        self.rest(self.reader.room)

    def _drag(self, event):
        """A handle dragged one step per displayed frame: the Project panel's edge, or the sidebar's beside Main."""
        handle = ('#primary-sidebar .resize-handle-right' if self.reader.room == 'main'
                  else '#project-panel.open .resize-handle-left')
        box = self.page.locator(handle).bounding_box()
        x, y = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
        self.page.mouse.move(x, y)
        self.page.mouse.down()
        steps = event.get('steps', 24)
        for step in range(1, steps + 1):
            self.page.mouse.move(x + event['drag'] * step / steps, y)
            self.page.wait_for_timeout(16)
        self.page.mouse.up()
        self.rest(self.reader.room)

    def _keyboard(self, event):
        field = self.page.locator(f'{self.feed(self.reader.room)} ~ .chat-input-area textarea').first
        if event['keyboard']:
            field.focus()
            self.page.evaluate('h => window.__viewport(h)', event['keyboard'])
            self.page.wait_for_function("() => document.body.classList.contains('keyboard-open')")
        else:
            self.page.evaluate('h => window.__viewport(h)', self.page.viewport_size['height'])
            field.blur()
            self.page.wait_for_function("() => !document.body.classList.contains('keyboard-open')")
        self.rest(self.reader.room)

    def _older(self, event):
        """One older page read at the reading edge: the passage read when it lands is held to stay."""
        hold = self.stand.hold('older')
        feed = self.page.locator(self.feed(self.reader.room))
        start = feed.evaluate('el => el.scrollTop')
        step = 800
        x, y = self._box()
        self.page.mouse.move(x, y)
        # The same saved rows can wrap taller on another engine or font. Allow
        # enough real wheel steps to reach the edge and send a gesture there.
        for _ in range(max(40, math.ceil(start / step) + 2)):
            if hold['caught']:
                break
            self.page.mouse.wheel(0, -step)
            self.page.wait_for_timeout(80)
        assert hold['caught'], f"the reading edge read no older page (scrollTop {start} to {feed.evaluate('el => el.scrollTop')})"
        self.rest(self.reader.room, held=True)
        self.reader.settle(self.reader.room, self.facts(self.reader.room))
        self.stand.release()
        self.rest(self.reader.room)
        return {'page': hold['caught']['kind']}  # which history read landed ('read' is the read receipt)

    def _quote(self, event):
        """The owner presses a reply's quote: its room lands on the row it quotes."""
        quote = self.stand.quotes[event['quote']]
        bubble = self.page.locator(f'{self.feed(self.reader.room)} .chat-bubble', has_text=quote['text'][:60])
        bubble.locator('.chat-quote-main').click()
        self.page.wait_for_function('([sel, text]) => document.querySelector(sel)?.textContent.includes(text)',
                                    arg=[self.feed(self.reader.room), quote['target'][:60]])
        self.rest(self.reader.room)
        return {'named': quote['target'][:60]}

    def _hold(self, event):
        self.stand.hold(event['hold'])

    def _release(self, event):
        self.stand.release()
        self.rest(self.reader.room)

    def _latest(self, event):
        held = any(not h['caught'] or not h['event'].is_set() for h in self.stand.holds)
        button = self.page.locator(f'{self.feed(self.reader.room)} ~ .chat-input-area .chat-scroll-bottom-btn')
        box = button.bounding_box()
        self.page.evaluate(LATEST, self.feed(self.reader.room))
        self.page.mouse.click(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
        self.page.wait_for_function('() => window.__latest.pressed !== null'
                                    ' && window.__latest.gaps.length - window.__latest.pressed >= 12')
        run = self.page.evaluate('() => window.__latest')
        gaps = run['gaps'][run['pressed']:]
        frames = next((i for i, gap in enumerate(gaps) if gap <= 2), None)
        self.rest(self.reader.room, held=held)
        return {'gaps': gaps, 'frames': frames, 'held': held}

    def _pause(self, event):
        self.page.wait_for_timeout(event['pause'])
        self.rest(self.reader.room)


def _brief(facts):
    if not facts:
        return None
    return {key: facts.get(key) for key in ('scrollTop', 'scrollHeight', 'clientHeight', 'gap', 'band', 'first',
                                            'place', 'named', 'picture', 'newest', 'rows', 'latest', 'reads')}


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('name', sessions())
def test_a_recorded_session_keeps_the_reading_rules(name, engine, tmp_path):
    playwright = _require_browser()
    session = load(name)
    data = tmp_path / 'data'
    data.mkdir()
    evidence = producer_evidence_dir(tmp_path / 'evidence', 'feed-sessions') / f'{name}-{engine}'
    stand = Stand(session, data)
    try:
        with playwright.sync_playwright() as pw:
            browser = getattr(pw, engine.split('-')[0]).launch(headless=True)
            try:
                width, height = session.get('viewport', [1280, 860])
                phone = bool(session.get('phone'))
                context = browser.new_context(viewport={'width': width, 'height': height}, has_touch=phone,
                                              is_mobile=phone and engine == 'chromium')
                page = context.new_page()
                page.set_default_timeout(15_000)
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)[:300]))
                page.add_init_script(INIT)
                if phone:
                    page.add_init_script(VIEWPORT)
                    # The stand-in viewport starts at the height of the laid-out page, not of the document's first
                    # moment (a phone page is laid out at its meta viewport's width only once that is parsed).
                    page.add_init_script("addEventListener('DOMContentLoaded', () => window.__viewport(innerHeight))")
                if engine == 'webkit-no-anchor':
                    page.add_init_script(f"document.addEventListener('DOMContentLoaded', () => {NO_ANCHOR})")
                player = Player(page, stand, session, engine)
                try:
                    player.play()
                finally:
                    # Every event's facts and the last frame, kept beside the run (a green run's too).
                    evidence.with_suffix('.json').write_text(json.dumps(player.trace, indent=1), encoding='utf-8')
                    page.screenshot(path=str(evidence.with_suffix('.png')))
            finally:
                browser.close()
    finally:
        stand.close()
    assert not errors, errors
    assert not player.reader.problems, '\n'.join([f'{name} in {engine}: {len(player.reader.problems)} broken rules',
                                                  *player.reader.problems, f'trace: {evidence.with_suffix(".json")}'])
