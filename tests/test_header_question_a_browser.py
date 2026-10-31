"""The owner's selected composition A (19.10) on the real SPA: question cards and page bars.

Real documents and modules over the existing static-server fixture and synthetic API rows; no runtime starts and no
live data is read. Chromium and WebKit: WebKit takes `chat_input_sizing.js`'s measured path, Chromium the engine's
content sizing, so the answer's growth is proven on both. The macOS band is the launcher's own two writes
(`data-window-chrome`, `--window-chrome-top`); the native drag itself is not exercised here.
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.fixture(params=['chromium', 'webkit'])
def engine_ui(request, monkeypatch):
    # The fixture launches the engine named before it starts (as the design-system gate does).
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui

PROJECT = {'id': 'proj-sound', 'name': 'Notifications and settings: a quiet sound kept after the macOS app update',
           'chat_id': 42, 'lifecycle': 'active', 'visible_revision': 0}
FACTS = 'Asked by task c7845ef5, started by your message of 2026-10-19 12:10 UTC; your last message in this chat: unknown.'
BLOCK = {'quiz_id': 'q-sound', 'state': 'open', 'asked_at': '2026-10-19T15:18:00Z', 'host_facts': FACTS,
         'question': 'Which switch do you turn off when the sound keeps playing: **Sound in Settings → Appearance**, '
                     'or **the Ouroboros sound in the macOS Notifications settings**? Your own words are welcome.',
         'options': [], 'option_details': [],
         'assumption': 'Checking both paths without test sounds and without changing your settings.'}


def _pointer(block):
    return {'role': 'system', 'system_type': 'project_question_pointer', 'task_id': 'task-sound',
            'quiz_id': block['quiz_id'], 'quiz_state': block['state'], 'project_id': PROJECT['id'],
            'project_name': PROJECT['name'], 'project_chat_id': 42, 'chat_id': 1, 'ts': block['asked_at'],
            **{key: block[key] for key in ('question', 'options', 'option_details', 'assumption', 'host_facts')}}


def _seed(page):
    page.route('**/api/projects', lambda r: r.fulfill(json={'projects': [PROJECT]}))
    page.route('**/api/state', lambda r: r.fulfill(json={
        'sha': 'browser-fixture', 'supervisor_ready': True, 'active_chat_activities': [], 'projects': [PROJECT],
        'project_chat_ids': [42], 'current_questions': [_pointer(BLOCK)], 'current_questions_complete': True,
        'main_attention': [], 'main_attention_complete': True}))

    def history(route):
        chat = parse_qs(urlparse(route.request.url).query).get('chat_id', ['1'])[0]
        own = {'role': 'assistant', 'msg_type': 'quiz', 'task_id': 'task-sound', 'chat_id': 42, 'ts': BLOCK['asked_at'],
               'text': BLOCK['question'], 'quiz': {k: v for k, v in BLOCK.items() if k != 'question'}}
        rows = [own] if chat == '42' else [_pointer(BLOCK)]
        route.fulfill(json={'messages': rows, 'progress': []})
    page.route('**/api/chat/history*', history)
    page.route('**/api/tasks/task-sound', lambda r: r.fulfill(json={
        'task_id': 'task-sound', 'project_id': PROJECT['id'], 'owner_quiz': {BLOCK['quiz_id']: BLOCK},
        'question_rooms': {BLOCK['quiz_id']: {'state': 'known', 'chat_id': 42, 'project_id': PROJECT['id']}}}))
    page.route('**/api/files/list*', lambda r: r.fulfill(json={
        'path': '.', 'parent_path': '.', 'breadcrumb': [{'name': '/Users/owner', 'path': '.'}],
        'entries': [{'name': 'notes.md', 'path': 'notes.md', 'type': 'file', 'size': 50}]}))
    page.route('**/api/extensions', lambda r: r.fulfill(json={'skills': [dict(
        name='gate_skill', description='An installed skill.', enabled=True, loaded=True, review_status='pass',
        review_gate='pass', source='self_authored', tools=['gate_tool'])], 'live': {}}))
    page.route('**/api/widgets', lambda r: r.fulfill(json={'ui_tabs': [dict(
        skill='gate_skill', tab_id='radar', title='Release radar', render={'kind': 'declarative', 'schema_version': 1,
        'components': [{'type': 'metric', 'label': 'Processed', 'value': '42'}]})]}))


def _open(ui, width, height, theme=None, seed=None):
    page = ui['new_page'](viewport={'width': width, 'height': height}, has_touch=width < 720)
    page.set_default_timeout(10_000)
    if theme:  # the System theme under this OS appearance (no stored choice: an app frame has no storage)
        page.emulate_media(color_scheme=theme)
    (seed or _seed)(page)
    origin = urlparse(ui['url']).netloc
    page.route('**/*', lambda r: r.fallback() if urlparse(r.request.url).netloc == origin else r.abort())
    page.goto(ui['url'] + '/')
    page.wait_for_selector('#chat-input', state='attached')
    if theme:
        assert page.evaluate('document.documentElement.dataset.theme') == theme
    return page


def _band(page, top=52):
    # What launcher_window_chrome.py writes into an integrated document, measured; nothing else is simulated.
    page.evaluate("""top => { const root = document.documentElement; root.dataset.windowChrome = 'integrated';
        root.style.setProperty('--window-chrome-top', `${top}px`); root.style.setProperty('--window-chrome-start', '78px'); }""", top)


QUESTION_GEOMETRY = """() => {
    const box = el => { const r = el.getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height, right: r.right, bottom: r.bottom, cy: r.y + r.height / 2}; };
    const bubble = document.querySelector('#chat-messages .chat-bubble.project-question');
    const card = bubble.querySelector('.chat-quiz-card');
    const head = card.querySelector('.chat-quiz-head');
    const frame = card.querySelector('.chat-quiz-comment-box');
    // The bubble's last line is its time alone (owner, refined A 21.10): no host sentence beside it.
    const foot = bubble.querySelector(':scope > .msg-time');
    const capsule = bubble.querySelector('.chat-bubble-voice > .chat-quiz-project');
    return {voice: [...bubble.querySelector('.chat-bubble-voice').children].map(el => el.classList[0]),
        capsule: box(capsule), capsuleName: capsule.getAttribute('aria-label'), cardRight: box(card).right,
        head: box(head), centres: [...head.querySelectorAll('.chat-quiz-chip, .chat-quiz-status, .chat-quiz-close')].map(el => box(el).cy),
        headOrder: [...head.children].map(el => el.className),
        toQuestion: box(card.querySelector('.chat-quiz-question')).y - box(bubble).y,
        frame: box(frame), send: box(frame.querySelector('.chat-quiz-send')),
        field: frame.querySelector('.chat-quiz-comment').getAttribute('aria-label'),
        parts: [...bubble.children].filter(el => !el.hidden).map(el => el.classList[0]), footText: foot.textContent.trim(),
        bubbleText: bubble.textContent.replace(/\\s+/g, ' '), factsAnywhere: !!bubble.querySelector('.chat-quiz-host-facts'),
        question: card.querySelector('.chat-quiz-question').textContent.replace(/\\s+/g, ' ').trim(),
        assumption: card.querySelector('.chat-quiz-assumption')?.textContent || '',
        clear: !!bubble.querySelector('.chat-quiz-clear'),
        bubble: box(bubble), footBox: box(foot), page: document.scrollingElement.scrollWidth - innerWidth};
}"""

# Every box in the Main question's bubble, keyed by its place in the tree.
BOXES = """() => {
    const bubble = document.querySelector('#chat-messages .chat-bubble.project-question');
    const key = (el) => { const path = []; for (let n = el; n !== bubble; n = n.parentElement) path.unshift(`${n.tagName}${[...n.parentElement.children].indexOf(n)}`); return path.join('/'); };
    return Object.fromEntries([bubble, ...bubble.querySelectorAll('*')].map((el) => { const r = el.getBoundingClientRect();
        return [el === bubble ? 'bubble' : key(el), [r.x, r.y, r.width, r.height].map((v) => Math.round(v * 100) / 100)]; }));
}"""

GROWTH = """async () => {
    const field = document.querySelector('#chat-messages .project-question-card .chat-quiz-comment');
    const frame = field.closest('.chat-quiz-comment-box'), send = frame.querySelector('.chat-quiz-send');
    const settle = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const steps = [];
    field.focus();
    for (const lines of [0, 1, 2, 4, 12, 3, 1]) {
        field.value = Array.from({length: lines}, (_, i) => `answer line ${i + 1}`).join('\\n');
        field.dispatchEvent(new Event('input', {bubbles: true}));
        await settle();
        const f = frame.getBoundingClientRect(), s = send.getBoundingClientRect();
        steps.push({lines, height: Math.round(field.getBoundingClientRect().height), mode: field.dataset.sizing,
            scrolls: field.scrollHeight > field.clientHeight + 1,
            inside: s.left >= f.left && s.right <= f.right && s.top >= f.top && s.bottom <= f.bottom});
    }
    return {steps, ring: getComputedStyle(frame).boxShadow};
}"""


# The anatomy holds in both themes: the dark voices change colours and the materials inside the answer, never a size.
@pytest.mark.parametrize('theme', ['light', 'dark'])
@pytest.mark.parametrize('width,height', [(956, 801), (390, 844)])
def test_question_card_voice_line_one_row_head_answer_frame_and_foot(engine_ui, width, height, theme):
    engine = engine_ui['engine']
    page = _open(engine_ui, width, height, theme)
    try:
        page.locator('#chat-messages .project-question-card .chat-quiz-comment').wait_for()
        facts = page.evaluate(QUESTION_GEOMETRY)
        print(json.dumps({'question_a_geometry': facts, 'engine': engine, 'viewport': [width, height], 'theme': theme}))
        # Voice line: `Ouroboros` and the Project it comes from, one 28px row; the name gives way inside the card.
        assert facts['voice'] == ['sender', 'chat-live-project-card-btn'], facts
        assert round(facts['capsule']['h']) == 28 and facts['capsule']['right'] <= facts['cardRight'] + 0.5, facts
        assert facts['capsuleName'] == f"Open this question in {PROJECT['name']}", 'the accessible name stays whole'
        # Head: `[Question] ● status … ×`, one row, every item on one centre line.
        assert facts['headOrder'] == ['chat-quiz-chip', 'chat-quiz-status', 'chat-quiz-head-end'], facts
        assert round(facts['head']['h']) == 28 and max(facts['centres']) - min(facts['centres']) <= 1, facts
        assert facts['toQuestion'] <= 92, 'two lines, not four, before the question'
        # Answer: one frame holding its words and Send at its trailing bottom edge.
        assert facts['field'] == 'Your answer' and not facts['clear'], facts
        assert facts['frame']['right'] - facts['send']['right'] <= 6 and facts['frame']['bottom'] - facts['send']['bottom'] <= 6, facts
        # Foot (owner, refined A 21.10): the time alone ends the bubble; the host's sentence is gone from the bubble,
        # and everything the card says (question, the owner's answer frame, the signature line) stays as it was.
        assert facts['parts'] == ['chat-bubble-voice', 'chat-quiz-card', 'msg-time'], facts
        assert facts['footText'] and FACTS not in facts['bubbleText'] and not facts['factsAnywhere'], facts
        assert facts['question'].startswith('Which switch do you turn off when the sound keeps playing'), facts
        assert facts['assumption'] == 'Continuing meanwhile: ' + BLOCK['assumption'], facts
        assert facts['footBox']['right'] <= facts['bubble']['right'] and facts['page'] <= 0, facts
        # The other theme repaints the same card in place and moves no box of it.
        boxes = page.evaluate(BOXES)
        for scheme in ('dark' if theme == 'light' else 'light', theme):
            page.emulate_media(color_scheme=scheme)
            page.wait_for_function('scheme => document.documentElement.dataset.theme === scheme', arg=scheme)
            moved = {key: [box, boxes.get(key)] for key, box in page.evaluate(BOXES).items() if box != boxes.get(key)}
            assert not moved and page.evaluate(QUESTION_GEOMETRY) == facts, (scheme, moved)
        growth = page.evaluate(GROWTH)
        print(json.dumps({'question_a_growth': growth, 'engine': engine}))
        heights = {step['lines']: step['height'] for step in growth['steps']}
        assert heights[0] == heights[1] == heights[2], growth  # two lines at rest
        assert heights[2] < heights[4] < heights[12], growth   # grows with its words
        assert heights[12] == 140 and growth['steps'][4]['scrolls'], growth  # to one cap, then scrolls
        assert growth['steps'][-1]['height'] == heights[1], 'and shrinks again'
        assert all(step['inside'] for step in growth['steps']), growth
        assert growth['ring'] not in ('', 'none'), 'the frame wears the field focus idiom'
        setup_browser.capture(page, f'question-a-{engine}-{width}-{theme}')
        # The Project's own card, opened by its reference as tests/test_ui_result_browser.py presses it: the capsule
        # is in the bubble's voice line, so a locator inside the card finds none. The same head, no Project to name.
        pointer = page.locator('#chat-messages .project-question-card[data-quiz-id="q-sound"]')
        main_question = page.locator('#chat-messages .chat-bubble.project-question').filter(
            has=page.locator('.project-question-card[data-quiz-id="q-sound"]'))
        assert pointer.locator('.chat-quiz-project').count() == 0
        main_question.locator('.chat-bubble-voice > .chat-quiz-project').click()
        own = page.locator('#project-panel .chat-quiz-card[data-quiz-id="q-sound"]')
        own.wait_for()
        assert own.evaluate("el => [...el.querySelector('.chat-quiz-head').children].map(c => c.className)") == facts['headOrder']
        assert own.evaluate("el => el.closest('.chat-bubble').querySelector('.chat-bubble-voice')") is None
        assert own.evaluate("el => el.closest('.chat-bubble').textContent").count(FACTS) == 0
        assert own.evaluate("el => !!el.closest('.chat-bubble').querySelector(':scope > .msg-time')")
        own.evaluate("el => el.scrollIntoView({block: 'center'})")
        setup_browser.capture(page, f'question-a-project-{engine}-{width}-{theme}')
    finally:
        page.close()


SETTLED = {**BLOCK, 'quiz_id': 'q-settled', 'state': 'answered', 'answered_index': 1, 'recommended_index': 0,
           'options': ['A. Keep the quiet treatment', 'B. Use my own direction'],
           'option_details': ['One colour family inside the answer.', 'A distinct recorded answer, with readable detail.']}


def _seed_settled(page):
    """Main holds one question its own task asked, answered by the owner (history replays the settled card)."""
    _seed(page)
    page.unroute('**/api/chat/history*')
    page.route('**/api/chat/history*', lambda r: r.fulfill(json={'progress': [], 'messages': [{
        'role': 'assistant', 'msg_type': 'quiz', 'task_id': 'task-settled', 'chat_id': 1, 'ts': SETTLED['asked_at'],
        'text': SETTLED['question'], 'quiz': {k: v for k, v in SETTLED.items() if k != 'question'}}]}))


SETTLED_FACTS = """(card) => {
    const box = (el) => el.getBoundingClientRect();
    const head = card.querySelector('.chat-quiz-head');
    const centres = [...head.querySelectorAll('.chat-quiz-chip, .chat-quiz-status')].map((el) => box(el).y + box(el).height / 2);
    const chosen = card.querySelector('.chat-quiz-option.chosen');
    const cue = chosen.querySelector('.chat-quiz-option-chosen-cue');
    const parts = [...chosen.querySelectorAll('.chat-quiz-option-chosen-name, .chat-quiz-option-chosen-cue')];
    const kept = box(chosen).height, line = cue.getClientRects().length ? box(cue).height : 0;
    const places = parts.map((part) => [part, part.parentNode, part.nextSibling]);
    parts.forEach((part) => part.remove());
    const without = box(chosen).height;
    places.reverse().forEach(([part, parent, next]) => parent.insertBefore(part, next));
    return {status: card.querySelector('.chat-quiz-status-text').textContent, head: Math.round(box(head).height),
        spread: Math.max(...centres) - Math.min(...centres), parts: parts.length, kept, without, line,
        gap: parseFloat(getComputedStyle(chosen).rowGap), shown: cue.getClientRects().length > 0};
}"""


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_a_settled_question_names_the_owners_choice_and_keeps_its_anatomy(engine_ui, theme):
    # The owner's recorded answer says `Your choice` in its accessible name in both themes; on dark the word also shows
    # inside the chosen card as exactly one more line, on white it takes no room. The head keeps its one row.
    page = _open(engine_ui, 956, 801, theme, _seed_settled)
    try:
        card = page.locator('#chat-messages .chat-quiz-card[data-quiz-id="q-settled"]')
        card.wait_for()
        facts = card.evaluate(SETTLED_FACTS)
        print(json.dumps({'question_settled': facts, 'engine': engine_ui['engine'], 'theme': theme}))
        assert facts['status'] == 'You answered' and facts['head'] == 28 and facts['spread'] <= 1, facts
        chosen, other = card.locator('.chat-quiz-option.chosen'), card.locator('.chat-quiz-option:not(.chosen)')
        assert chosen.count() == 1 and other.count() == 1 and facts['parts'] == 2, facts
        assert chosen.aria_snapshot().count('Your choice') == 1 and 'Your choice' not in other.aria_snapshot()
        if theme == 'dark':
            assert facts['shown'] and abs(facts['kept'] - facts['without'] - facts['line'] - facts['gap']) <= 0.5, facts
        else:
            assert not facts['shown'] and facts['kept'] == facts['without'], facts
        setup_browser.capture(page, f'question-settled-{engine_ui["engine"]}-{theme}')
    finally:
        page.close()


TARGETS = """() => {
    const bubble = document.querySelector('#chat-messages .chat-bubble.project-question');
    const targets = {project: bubble.querySelector('.chat-bubble-voice > .chat-quiz-project'),
        close: bubble.querySelector('.chat-quiz-head .chat-quiz-close')};
    const touch = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--touch-target-min'));
    // A target's own finger area: its box grown about its centre to the touch minimum each way. Every point of it,
    // every 2px across and 1px down, must land on that target.
    const report = {};
    for (const [name, el] of Object.entries(targets)) {
        const r = el.getBoundingClientRect(), w = Math.max(r.width, touch), h = Math.max(r.height, touch);
        const left = r.left + r.width / 2 - w / 2, top = r.top + r.height / 2 - h / 2;
        const missed = [];
        let points = 0;
        for (let y = top + 0.5; y < top + h; y += 1) for (let x = left + 0.5; x < left + w; x += 2) {
            const hit = document.elementFromPoint(x, y);
            points += 1;
            if (!hit || !el.contains(hit)) missed.push([x, y, hit ? hit.getAttribute('class') || hit.tagName : '']);
        }
        report[name] = {box: [r.left, r.top, r.right, r.bottom], points, missed: missed.length, first: missed.slice(0, 4)};
    }
    const words = targets.project.querySelector('.chat-live-project-name');
    return {coarse: matchMedia('(pointer: coarse)').matches, touch, enabled: !targets.close.disabled, report,
        cut: words.scrollWidth > words.clientWidth, gap: report.close.box[1] - report.project.box[3]};
}"""


@pytest.mark.parametrize('width,height', [(390, 844)])
def test_a_finger_keeps_the_project_reference_and_close_apart(engine_ui, width, height):
    # Under a finger the voice line's Project capsule and the head's × each keep the touch minimum, centred on
    # themselves, and neither takes a point of the other's: a low tap on the ↗ never closes the question. The long
    # name fills the capsule's row, the case where ↗ reaches furthest; the rows keep their height.
    engine = engine_ui['engine']
    page = _open(engine_ui, width, height)
    try:
        page.locator('#chat-messages .project-question-card .chat-quiz-comment').wait_for()
        facts = page.evaluate(TARGETS)
        print(json.dumps({'finger_targets': facts, 'engine': engine, 'viewport': [width, height]}))
        assert facts['coarse'] and facts['touch'] == 44 and facts['enabled'] and facts['cut'], facts
        assert facts['report']['project']['missed'] == 0 and facts['report']['close']['missed'] == 0, facts['report']
        assert facts['gap'] <= 10, 'the voice line and the head keep their spacing'
        setup_browser.capture(page, f'finger-targets-{engine}-{width}')
    finally:
        page.close()


BAR = """name => {
    const box = el => { const r = el.getBoundingClientRect(); return {x: r.x, y: r.y, h: r.height, bottom: r.bottom, cy: r.y + r.height / 2}; };
    const page = document.querySelector('#page-' + name), header = page.querySelector('.app-page-header');
    const lead = header.querySelector('.app-page-lead');
    const items = [...header.querySelectorAll('.app-page-lead [role=tab], .app-page-lead .files-crumb, .app-page-toolbar button')]
        .filter(el => el.getClientRects().length && getComputedStyle(el).visibility !== 'hidden');
    const hb = box(header);
    const hit = document.elementFromPoint(hb.x + header.clientWidth / 2, hb.bottom + 8);
    return {lead: !!lead?.children.length, attrs: [...header.attributes].map(a => a.name).filter(a => a.startsWith('data-')),
        height: hb.h, centres: items.map(el => Math.round(box(el).cy * 2) / 2), first: items[0] ? box(items[0]).x - hb.x : null,
        under: !!page.querySelector('.app-page-tabs > *'), hitsPage: !!hit && !hit.closest('.app-page-header'),
        display: getComputedStyle(header).display};
}"""


def _goto(page, name):
    page.locator(f'[data-nav-page="{name}"]').click()
    page.locator(f'#page-{name}').wait_for(state='visible')
    page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')


def test_page_bars_hold_their_navigation_on_a_desk_and_keep_it_under_the_row_on_a_phone(engine_ui):
    engine = engine_ui['engine']
    page = _open(engine_ui, 956, 801)
    try:
        bars = {}
        for name in ('dashboard', 'settings', 'skills', 'files'):
            _goto(page, name)
            bars[name] = page.evaluate(BAR, name)
            bar = bars[name]
            # Two ends on Main's line: the page's tabs or location at the start, its actions at the end.
            assert bar['lead'] and 'data-lead' in bar['attrs'] and not bar['under'], (name, bar)
            assert bar['height'] <= 56 and bar['centres'] and max(bar['centres']) - min(bar['centres']) <= 1, (name, bar)
            assert abs(bar['centres'][0] - 25) <= 1 and abs(bar['first'] - 20) <= 1, (name, bar)
            # The strip lends its shadows room below the bar but never takes a press meant for the page.
            assert bar['hitsPage'], (name, bar)
            setup_browser.capture(page, f'bar-a-{name}-{engine}-956')
        # Eight Settings tabs scroll inside the bar; End reveals About without moving the page.
        _goto(page, 'settings')
        strip = page.locator('#page-settings .app-page-lead .app-tab-strip')
        assert strip.evaluate('el => el.scrollWidth > el.clientWidth')
        page.locator('#page-settings [data-settings-tab]').first.focus()
        page.keyboard.press('End')
        about = page.locator('#page-settings [data-settings-tab="about"]')
        assert about.evaluate("""el => { const s = el.closest('.app-tab-strip').getBoundingClientRect(), r = el.getBoundingClientRect();
            return r.left >= s.left - 1 && r.right <= s.right + 1; }""")
        assert page.evaluate("document.querySelector('#page-settings').dataset.activeSettingsTab") == 'about'
        # The macOS window: the bar is the band, its controls centred on it; an empty header (Widgets) is only the
        # window's handle over the page, and the page's first card sits one inset (not two) under the band.
        _band(page)
        settings_band = page.evaluate(BAR, 'settings')
        assert settings_band['height'] == 52 and abs(settings_band['centres'][0] - 26) <= 1, settings_band
        _goto(page, 'widgets')
        page.locator('.widgets-card').first.wait_for()
        widgets = page.evaluate("""() => ({bar: document.querySelector('#page-widgets .app-page-header').getBoundingClientRect().height,
            empty: document.querySelector('#page-widgets .app-page-header').hasAttribute('data-empty'),
            card: document.querySelector('.widgets-card').getBoundingClientRect().top,
            margin: getComputedStyle(document.querySelector('#widgets-list')).marginTop})""")
        assert widgets['empty'] and widgets['bar'] == 52 and widgets['margin'] == '0px', widgets
        assert abs(widgets['card'] - (52 + 16)) <= 1, widgets
        setup_browser.capture(page, f'bar-a-widgets-band-{engine}-956')
        # A phone: the same nodes go back under the row, focus and identity kept; the desk takes them back.
        _goto(page, 'dashboard')
        page.evaluate("window.__strip = document.querySelector('#page-dashboard .app-tab-strip')")
        page.locator('#page-dashboard [data-dashboard-tab="costs"]').focus()
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_function("document.documentElement.dataset.layout === 'stack'")
        phone = page.evaluate(BAR, 'dashboard')
        assert phone['under'] and not phone['lead'] and 'data-lead' not in phone['attrs'] and phone['height'] <= 56, phone
        assert page.evaluate("""() => document.querySelector('#page-dashboard .app-page-tabs .app-tab-strip') === window.__strip
            && document.activeElement?.dataset.dashboardTab === 'costs'""")
        setup_browser.capture(page, f'bar-a-dashboard-{engine}-390')
        page.set_viewport_size({'width': 956, 'height': 801})
        page.wait_for_function("document.documentElement.dataset.layout === 'docked'")
        assert page.evaluate("document.querySelector('#page-dashboard .app-page-lead .app-tab-strip') === window.__strip")
    finally:
        page.close()


# A board taller than the window: a route iframe (the skill's own page) the owner keeps running, then declarative
# cards. One load of its page means the frame never reloaded.
FRAME = ('<!doctype html><body style="margin:0;padding:16px;font:15px system-ui;color:#fff;background:#3a6ea5">'
         '<b>Cache Efficiency Snapshot</b><p>94.12% of 3.81 B input tokens were cache reads.</p></body>')
BOARD = [dict(skill='gate_skill', tab_id='cache', title='Cache Efficiency Snapshot',
              render={'kind': 'iframe', 'route': 'cache', 'height': 360, 'start': 'retain'})] + [dict(
    skill='gate_skill', tab_id=f'quota-{index}', title=f'Weekly quota {index}', render={
        'kind': 'declarative', 'schema_version': 1, 'components': [
            {'type': 'metric', 'label': 'Of 17 accounts', 'value': f'{index}.95'},
            {'type': 'markdown', 'text': '11 restricted · next reset Wed 21 Oct 09:00 · +0.24 if unused.'}]})
    for index in range(10)]

BOARD_EDGES = """() => {
    const page = document.querySelector('#page-widgets'), band = page.querySelector('.app-page-header');
    const list = page.querySelector('.widgets-scroll'), style = getComputedStyle(list);
    const b = band.getBoundingClientRect(), l = list.getBoundingClientRect();
    // Every point of the band across the board lands on the window's handle, never on a card or its frame, even
    // while a card passes under it.
    const missed = [];
    for (let y = b.top + 1; y < b.bottom; y += 10) for (let x = l.left + 1; x < l.right; x += 20) {
        const hit = document.elementFromPoint(x, y);
        if (hit !== band) missed.push([x, y, hit ? hit.className || hit.tagName : '']);
    }
    const under = [...page.querySelectorAll('.widgets-card')].some((c) => {
        const r = c.getBoundingClientRect(); return r.top < b.bottom && r.bottom > b.top; });
    return {scrolled: list.scrollTop, room: list.scrollHeight - list.clientHeight, mask: style.maskImage,
        padTop: parseFloat(style.paddingTop), under,
        webkitMask: style.webkitMaskImage, band: b.height, empty: band.hasAttribute('data-empty'),
        cursor: getComputedStyle(band).cursor, listTop: l.top, missed: missed.length, first: missed.slice(0, 4),
        frame: page.querySelector('[data-widget-key="gate_skill:cache"] iframe')?.__kept === true,
        page: document.scrollingElement.scrollWidth - innerWidth};
}"""


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_the_widgets_board_scrolls_with_crisp_edges_under_its_plain_band(engine_ui, theme):
    """The owner's window (1100 × 750, macOS band): scrolled, the board's live cards keep crisp edges where the list
    cuts them, with no fade (DESIGN "Widgets board"), while other scroll bodies keep `.scroll-fade-y`. The band is
    the window's, not a row of the page (owner 22.10: the empty strip cut the board): the board starts at the window's
    top edge, its first row below the band, scrolled cards pass under the band, and a press anywhere in the band still
    lands on the window's handle. A frame the owner keeps running is the same frame, never reloaded, across
    scrolling, another page and a phone width."""
    engine = engine_ui['engine']
    settle = '() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))'
    loads = []

    def frame_page(route):
        loads.append(route.request.url)
        route.fulfill(content_type='text/html', body=FRAME)

    page = _open(engine_ui, 1100, 750)
    try:
        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
        page.unroute('**/api/widgets')
        page.route('**/api/widgets', lambda r: r.fulfill(json={'ui_tabs': BOARD}))
        page.route('**/api/extensions/gate_skill/cache', frame_page)
        _band(page)
        _goto(page, 'widgets')
        frame = page.locator('[data-widget-key="gate_skill:cache"] iframe')
        frame.wait_for()
        frame.evaluate("el => { el.__kept = true; }")
        child = frame.element_handle().content_frame()
        child.wait_for_load_state()
        child.evaluate("() => { window.__kept = 'same window'; }")
        page.wait_for_function("n => document.querySelectorAll('#widgets-list .widgets-card').length === n", arg=len(BOARD))
        page.locator('#page-widgets .widgets-scroll').evaluate("el => { el.scrollTop = 300; }")
        page.evaluate(settle)
        edges = page.evaluate(BOARD_EDGES)
        print(json.dumps({'widgets_board_edges': edges, 'engine': engine, 'theme': theme, 'viewport': [1100, 750]}))
        setup_browser.capture(page, f'widgets-board-scrolled-{engine}-{theme}-1100')
        assert edges['scrolled'] == 300 and edges['room'] > 300, edges
        assert edges['mask'] in (None, 'none') and edges['webkitMask'] in (None, 'none'), 'no fade over the cards'
        assert edges['empty'] and edges['band'] == 52 and edges['listTop'] == 0 and edges['cursor'] == 'auto', edges
        assert edges['padTop'] >= 52 + 16 - 0.5 and edges['under'], edges
        assert not edges['missed'] and edges['frame'] and edges['page'] <= 0, edges
        # A dragged card: the band lets the pointer through to the board's top edge, where the engine scrolls a drag
        # upward, and is the window's handle again once the card is dropped.
        through = """dragging => { const card = document.querySelector('#widgets-list .widgets-card');
            card.classList.toggle('dragging', dragging); const l = document.querySelector('#page-widgets .widgets-scroll');
            return Boolean(document.elementFromPoint(l.getBoundingClientRect().left + 40, 10)?.closest('.widgets-scroll')); }"""
        assert page.evaluate(through, True), 'a drag reaches the board under the band'
        assert not page.evaluate(through, False), 'at rest the band is the handle'
        # Another page's scroll body keeps the shared fade.
        _goto(page, 'settings')
        settings = page.locator('#page-settings .settings-scroll').element_handle()
        page.wait_for_function("el => el.scrollHeight - el.clientHeight > 200", arg=settings)
        settings.evaluate("el => { el.scrollTop = 200; }")
        page.wait_for_function("el => el.hasAttribute('data-scroll-above')", arg=settings)
        assert 'gradient' in settings.evaluate("el => getComputedStyle(el).webkitMaskImage")
        # Back on the board, then a phone width without the band: the same frame and window, one load, no fade.
        _goto(page, 'widgets')
        page.evaluate("() => { delete document.documentElement.dataset.windowChrome; }")
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_function("document.documentElement.dataset.layout === 'stack'")
        page.locator('#page-widgets .widgets-scroll').evaluate("el => { el.scrollTop = 300; }")
        page.evaluate(settle)
        phone = page.evaluate(BOARD_EDGES)
        print(json.dumps({'widgets_board_phone': phone, 'engine': engine, 'theme': theme, 'loads': len(loads)}))
        setup_browser.capture(page, f'widgets-board-scrolled-{engine}-{theme}-390')
        assert phone['scrolled'] == 300 and phone['mask'] in (None, 'none') and phone['webkitMask'] in (None, 'none'), phone
        assert phone['frame'] and phone['page'] <= 0, phone
        assert child.evaluate("() => window.__kept") == 'same window' and len(loads) == 1, loads
    finally:
        page.close()


# Locations as the gateway writes them (ouroboros/gateway/files.py): the root's absolute path, then one crumb per
# folder. Nine deep, and two deep with a name longer than a narrow bar holds.
CURRENT = 'review-ledger-primary-builder-response-change-coupling'
DEEP = ['/Users/owner', 'Ouroboros', 'data', 'task_results', 'artifacts', 'ceb8e6e2', 'source_handles',
        'context_checkpoints', CURRENT]

LOCATION = """() => {
    const box = el => { const r = el.getBoundingClientRect(); return {left: r.left, right: r.right, w: r.width, h: r.height, cx: r.left + r.width / 2, cy: r.top + r.height / 2}; };
    const header = document.querySelector('#page-files .app-page-header');
    const trail = header.querySelector('.app-page-lead > .files-breadcrumb');
    const crumbs = [...trail.querySelectorAll('.files-crumb')];
    const view = box(trail), end = box(header.querySelector('.app-page-toolbar')).left;
    // Whole within a pixel (a scroll offset is whole pixels, a strip's width need not be), and pressable there.
    const seen = el => { const r = box(el); const hit = document.elementFromPoint(r.cx, r.cy);
        return r.left >= view.left - 1 && r.right <= Math.min(view.right, end) + 1 && (hit === el || el.contains(hit)); };
    const current = crumbs[crumbs.length - 1];
    return {names: crumbs.map(el => el.textContent), seen: crumbs.map(seen), bar: box(header).h, view: view.w,
        current: box(current).w, cut: current.scrollWidth > current.clientWidth, ellipsis: getComputedStyle(current).textOverflow,
        hidden: trail.scrollWidth - trail.clientWidth, overflow: getComputedStyle(trail).overflowX,
        fades: ['data-scroll-start', 'data-scroll-end'].filter(name => trail.hasAttribute(name)),
        page: document.scrollingElement.scrollWidth - innerWidth};
}"""


def _location(page, names):
    crumbs = [{'name': name, 'path': '/'.join(names[1:index + 1]) or '.'} for index, name in enumerate(names)]
    page.unroute('**/api/files/list*')
    page.route('**/api/files/list*', lambda r: r.fulfill(json={
        'path': crumbs[-1]['path'], 'parent_path': crumbs[-2]['path'], 'breadcrumb': crumbs,
        'entries': [{'name': 'notes.md', 'path': 'notes.md', 'type': 'file', 'size': 50}]}))
    page.locator('#files-refresh').click()
    page.wait_for_function("names => [...document.querySelectorAll('#page-files .files-crumb')].map(el => el.textContent)"
                           ".join('/') === names.join('/')", arg=names)
    page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')


@pytest.mark.parametrize('width', [720, 1060])
def test_files_location_keeps_every_folder_reachable_in_the_bar(engine_ui, width):
    # The bar is one row, so a location longer than its start scrolls there, as a tab strip does: it rests at its
    # end with the folder you are in whole in view (ellipsized only past the bar's start), fades where folders hide,
    # and a sideways wheel and the keyboard reach every one of them. No folder is dropped or cut off unseen.
    engine = engine_ui['engine']
    page = _open(engine_ui, width, 801)
    try:
        _goto(page, 'files')
        for kind, names in (('deep', DEEP), ('long', DEEP[:1] + [CURRENT])):
            _location(page, names)
            rest = page.evaluate(LOCATION)
            print(json.dumps({'files_location': rest, 'kind': kind, 'engine': engine, 'width': width}))
            setup_browser.capture(page, f'files-location-{kind}-{engine}-{width}')
            assert rest['names'] == names and rest['bar'] <= 56 and rest['page'] <= 0, rest
            assert rest['seen'][-1] and rest['current'] <= rest['view'] + 0.5, 'the folder you are in, whole in view'
            assert not rest['cut'] or rest['ellipsis'] == 'ellipsis', rest
            if rest['hidden'] > 1:
                assert rest['overflow'] in ('auto', 'scroll') and rest['fades'] == ['data-scroll-start'], rest
            # A sideways wheel (a trackpad's swipe) scrolls the location to its start and back.
            trail = page.locator('#page-files .app-page-lead > .files-breadcrumb')
            trail.hover()
            page.mouse.wheel(-4000, 0)
            page.wait_for_function("el => el.scrollLeft <= 1", arg=trail.element_handle())
            page.wait_for_function(f"() => ({LOCATION})().seen[0]")
            assert page.evaluate(LOCATION)['seen'][0], 'the root is reachable by a pointer'
            page.mouse.wheel(4000, 0)
            page.wait_for_function("el => el.scrollLeft >= el.scrollWidth - el.clientWidth - 1", arg=trail.element_handle())
            # Wait for hit-tested geometry too: the wheel's visual scroll can settle after its offset.
            page.wait_for_function(f"() => ({LOCATION})().seen.at(-1)")
            assert page.evaluate(LOCATION)['seen'][-1]
            # The keyboard: each folder in turn comes into view where it can be pressed.
            crumbs = page.locator('#page-files .app-page-lead .files-crumb')
            for index in range(crumbs.count()):
                crumbs.nth(index).focus()
                page.wait_for_function(f"index => ({LOCATION})().seen[index]", arg=index)
                assert page.evaluate(LOCATION)['seen'][index], (kind, index)
    finally:
        page.close()


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_files_navigation_and_actions_survive_sidebar_resizing(engine_ui, theme):
    """A docked 720px window can leave only 160px for Files; use its real resize handle."""
    page = _open(engine_ui, 720, 801)
    engine = engine_ui['engine']
    try:
        page.evaluate("""theme => { document.documentElement.dataset.theme = theme;
            document.documentElement.style.setProperty('--sidebar-width', '180px'); }""", theme)
        _goto(page, 'files')
        _location(page, DEEP)
        page.evaluate("""() => { window.__filesNodes = ['files-breadcrumb', 'files-new-file', 'files-new-dir']
            .map(id => document.getElementById(id)); document.getElementById('files-new-file').focus(); }""")
        handle = page.locator('#primary-sidebar > .resize-handle').bounding_box()
        start_x = handle['x'] + handle['width'] / 2
        page.mouse.move(start_x, handle['y'] + handle['height'] / 2)
        page.mouse.down()
        # Check during the drag as well as after release, without a window resize event.
        page.mouse.move(start_x + 545 - 180, handle['y'] + handle['height'] / 2, steps=12)
        page.wait_for_function("Math.round(document.getElementById('content').getBoundingClientRect().width) === 175")
        page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
        setup_browser.capture(page, f'files-sidebar-drag-{engine}-{theme}')
        assert page.locator('#files-breadcrumb').evaluate("el => el.closest('.files-browser-header') !== null")
        assert page.evaluate("document.activeElement === window.__filesNodes[1]")
        page.mouse.move(start_x + 560 - 180, handle['y'] + handle['height'] / 2)
        page.mouse.up()
        page.wait_for_function("Math.round(document.getElementById('content').getBoundingClientRect().width) === 160")
        # Reveal Paste through the real file menu, making the action group wider than the available column.
        page.locator('#files-list .files-entry').filter(has_text='notes.md').click(button='right')
        page.locator('#files-context-menu [data-action="copy"]').click()
        controls = page.locator('#files-breadcrumb .files-crumb, .files-browser-actions button:visible, #files-refresh')
        measured = []
        for index in range(controls.count()):
            control = controls.nth(index)
            control.scroll_into_view_if_needed()
            assert control.evaluate("""el => { const r = el.getBoundingClientRect(), content = document.getElementById('content').getBoundingClientRect();
                const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
                return r.width > 0 && r.left >= content.left && r.right <= content.right + 1 && (hit === el || el.contains(hit)); }""")
            control.click(trial=True)
            measured.append({'name': control.inner_text(), 'box': control.bounding_box()})
        print(json.dumps({'files_narrow_controls': measured, 'engine': engine, 'theme': theme, 'content_width': 160}))
        assert page.evaluate('document.scrollingElement.scrollWidth <= innerWidth')
        assert page.locator('#page-files .app-page-header').get_by_role('heading', name='Files', exact=True).count() == 1
        page.locator('#files-new-file').scroll_into_view_if_needed()
        setup_browser.capture(page, f'files-sidebar-narrow-{engine}-{theme}')
        page.locator('#files-new-file').click()
        page.locator('.files-editor').fill('Draft kept while the sidebar and window change width.')
        # Same nodes and focus when the column widens and when the window enters the phone layout.
        page.locator('#files-new-file').focus()
        page.set_viewport_size({'width': 1060, 'height': 801})
        page.wait_for_function("document.querySelector('#page-files .app-page-header').hasAttribute('data-lead')")
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_function("document.documentElement.dataset.layout === 'stack'")
        assert page.evaluate("""window.__filesNodes.every(el => el === document.getElementById(el.id))
            && document.activeElement === window.__filesNodes[1]
            && !!window.__filesNodes[0].closest('.files-browser-header')""")
        assert page.locator('.files-editor').input_value() == 'Draft kept while the sidebar and window change width.'
    finally:
        setup_browser.capture(page, f'files-sidebar-final-{engine}-{theme}')
        page.close()


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_skills_tabs_and_existing_toolbar_survive_a_narrow_content_column(engine_ui, theme):
    """Existing toolbar actions compete with navigation just like Files' moved actions."""
    page = _open(engine_ui, 720, 801)
    try:
        page.unroute('**/api/extensions')
        page.route('**/api/extensions', lambda route: route.fulfill(json={'skills': [
            {'name': f'update-{i}', 'description': 'Test update', 'enabled': True, 'loaded': True,
             'review_status': 'pass', 'review_gate': 'pass', 'install': {'basis': 'trusted',
             'source': {'id': 'official'}, 'release': {'version': '1'},
             'update_available': {'digest': str(i) * 64, 'version': '2'}}} for i in (1, 2)], 'live': {}}))
        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
        _goto(page, 'skills')
        page.locator('#skills-update-all').wait_for()
        page.evaluate("window.__skillsTabs = document.querySelector('#page-skills .app-tab-strip')")
        page.locator('#page-skills [data-tab="installed"]').focus()
        handle = page.locator('#primary-sidebar > .resize-handle').bounding_box()
        page.mouse.move(handle['x'] + handle['width'] / 2, handle['y'] + handle['height'] / 2)
        page.mouse.down()
        page.mouse.move(558, handle['y'] + handle['height'] / 2, steps=12)
        page.mouse.up()
        page.wait_for_function("document.querySelector('#page-skills .app-page-header').hasAttribute('data-narrow')")
        assert page.locator('#page-skills .app-page-header').get_by_role('heading', name='Skills', exact=True).count() == 1
        strip = page.locator('#page-skills .app-page-tabs .app-tab-strip')
        assert strip.evaluate('el => el === window.__skillsTabs && el.clientWidth > 80')
        assert page.evaluate("document.activeElement.dataset.tab === 'installed'")
        for selector in ('#skills-update-all', '#skills-refresh'):
            button = page.locator(selector)
            assert button.evaluate("el => { const r=el.getBoundingClientRect(), c=document.getElementById('content').getBoundingClientRect(); return r.left >= c.left && r.right <= c.right; }")
            button.click(trial=True)
        setup_browser.capture(page, f'skills-sidebar-narrow-{engine_ui["engine"]}-{theme}')
        for name in ('sources', 'browse', 'installed'):
            button = strip.locator(f'[data-tab="{name}"]')
            button.click()
            assert button.get_attribute('aria-selected') == 'true'
        page.set_viewport_size({'width': 1280, 'height': 801})
        page.wait_for_function("document.querySelector('#page-skills .app-page-header').hasAttribute('data-lead')")
        assert page.locator('#page-skills .app-page-lead .app-tab-strip').evaluate('el => el === window.__skillsTabs')
        assert page.evaluate('document.scrollingElement.scrollWidth <= innerWidth')
    finally:
        page.close()


ANSWERED = {'quiz_id': 'q-chosen', 'state': 'answered', 'asked_at': '2026-10-21T19:56:00Z', 'host_facts': FACTS,
            'question': 'Who finishes the saved Codex work on SQLite?',
            'options': ['A — keep it with the same Codex', 'B — hand the saved candidate to one Ouroboros author',
                        'C — keep it saved, no continuation'],
            'option_details': ['Keeps the original author and context.', 'Reconcile the base first, then tests and review.',
                               'Code and results stay; nothing new runs.'],
            'recommended_index': 0, 'answered_index': 1,
            'assumption': 'Finishing the audit and keeping an exact handoff.'}
WORDS = {**ANSWERED, 'quiz_id': 'q-words', 'asked_at': '2026-10-21T19:57:00Z', 'answered_index': None,
         'comment': 'finish it yourself, but ask me questions after the review panel'}
CHOICE_VIEW = """() => [...document.querySelectorAll('#chat-messages .chat-bubble')].filter(b => b.querySelector('.chat-quiz-card'))
    .map(bubble => { const card = bubble.querySelector('.chat-quiz-card'); const options = [...card.querySelectorAll('.chat-quiz-option')];
        return {id: card.dataset.quizId, state: card.dataset.state,
            chosen: options.map(o => o.classList.contains('chosen')),
            names: options.map(o => o.querySelector('.chat-quiz-option-chosen-name')?.textContent || ''),
            fills: options.map(o => getComputedStyle(o).backgroundColor),
            recommended: options.map(o => !!o.querySelector('.chat-quiz-option-recommended')),
            details: options.map(o => o.querySelector('.chat-quiz-option-detail')?.textContent || ''),
            answer: card.querySelector('.chat-quiz-answer')?.textContent || '',
            assumption: card.querySelector('.chat-quiz-assumption')?.textContent || '',
            parts: [...bubble.children].filter(el => !el.hidden).map(el => el.classList[0]),
            text: bubble.textContent}; })"""


def _seed_answered(page):
    def row(block):
        return {'role': 'assistant', 'msg_type': 'quiz', 'task_id': 'task-sqlite', 'chat_id': 1, 'ts': block['asked_at'],
                'text': block['question'], 'quiz': {k: v for k, v in block.items() if k != 'question' and v is not None}}
    page.route('**/api/projects', lambda r: r.fulfill(json={'projects': []}))
    page.route('**/api/state', lambda r: r.fulfill(json={
        'sha': 'browser-fixture', 'supervisor_ready': True, 'active_chat_activities': [], 'projects': [],
        'project_chat_ids': [], 'current_questions': [], 'current_questions_complete': True,
        'main_attention': [], 'main_attention_complete': True}))
    page.route('**/api/chat/history*', lambda r: r.fulfill(json={'messages': [row(ANSWERED), row(WORDS)], 'progress': []}))
    page.route('**/api/tasks/task-sqlite', lambda r: r.fulfill(json={
        'task_id': 'task-sqlite', 'owner_quiz': {ANSWERED['quiz_id']: ANSWERED, WORDS['quiz_id']: WORDS}}))


# Refined A (owner 21.10) changed only the bubble's last line: an answered question keeps every option, its details and
# the recommendation, marks the recorded choice (`Your choice`, the chosen fill) and keeps the owner's own words and the
# signature line, live and after a reload alike. Words without a chosen option mark no option.
@pytest.mark.parametrize('theme', ['light', 'dark'])
@pytest.mark.parametrize('width,height', [(1100, 750), (400, 800)])
def test_answered_question_keeps_its_choice_and_words_after_reload_without_the_host_sentence(engine_ui, width, height, theme):
    page = _open(engine_ui, width, height, theme, seed=_seed_answered)
    try:
        page.locator('#chat-messages .chat-quiz-card[data-quiz-id="q-words"]').wait_for()
        for attempt in ('live', 'reload'):
            if attempt == 'reload':
                engine_ui['errors'].watch(page)  # the reload may cut off its own reads (tests/ui_console_errors.py)
                page.reload()
                page.locator('#chat-messages .chat-quiz-card[data-quiz-id="q-words"]').wait_for()
            views = {view['id']: view for view in page.evaluate(CHOICE_VIEW)}
            chosen, words = views['q-chosen'], views['q-words']
            print(json.dumps({'answered_question': views, 'attempt': attempt, 'engine': engine_ui['engine'],
                              'viewport': [width, height], 'theme': theme}))
            assert chosen['state'] == words['state'] == 'answered', views
            assert chosen['chosen'] == [False, True, False] and chosen['names'] == ['', 'Your choice', ''], chosen
            assert chosen['fills'][1] != chosen['fills'][0] == chosen['fills'][2], 'the recorded choice wears its own fill'
            assert chosen['recommended'] == [True, False, False] and chosen['details'] == ANSWERED['option_details'], chosen
            assert words['chosen'] == [False, False, False] and words['names'] == ['', '', ''], words
            assert words['answer'] == "Owner's answer: " + WORDS['comment'], words
            for view in (chosen, words):
                assert view['assumption'], view
                assert view['parts'][-1] == 'msg-time' and FACTS not in view['text'], view
        page.locator('#chat-messages .chat-quiz-card[data-quiz-id="q-chosen"]').evaluate(
            "el => el.scrollIntoView({block: 'center'})")
        setup_browser.capture(page, f'question-answered-{engine_ui["engine"]}-{width}-{theme}')
    finally:
        page.close()
