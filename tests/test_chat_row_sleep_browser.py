"""Rows asleep keep their own size when what they show changes without a new width (owner 20.10, 3A; review REV1).

A row far from the screen sleeps (chat_row_sleep.js, style.css `> [data-laid]`) only at the size it was laid out at.
Two things change a row's size without touching the feed's width:
- the theme: the dark theme's recorded answer has a `Your choice` line the light one has not, so a theme switch wakes
  every row and they sleep again at their new size, both ways, and the reading owner places the reader again (the
  answered question above changed size at once, which WebKit's own anchoring does not absorb): a passage read
  mid-history stays where it was, and a reader at the newest message stays there;
- a late canonical answer adding the owner's words to a question far above the passage: the reading owner restores
  the passage inside its write, and the row the write changed is laid out before it does (`flush`), so the passage
  stays where it was, at once, two frames later and once the row sleeps again.
The feed's own scroll anchoring is switched off for the second check, so the reading owner alone holds the passage.
- a row above the passage growing outside any write (a work card's buttons arriving after an older page): the browser's
  own scroll anchoring holds the passage, which WebKit does only while the rows near the view are not contained; rows
  sleep only far from the view (and a history control never), so the passage stays.
Uses the static-server fixture; never starts the runtime or reads live data."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.test_design_system_gate_browser import rest, save_evidence, seed_page
from tests.test_header_question_a_browser import SETTLED

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

FRAMES = "async n => { for (let i = 0; i < n; i += 1) await new Promise(r => requestAnimationFrame(r)); }"
# The answered question's row: the height it keeps, the height it has once woken for a moment, whether it sleeps.
QUESTION = """() => { const row = document.querySelector('#chat-messages [data-quiz-id="q-settled"]').closest('.chat-bubble');
    const laid = row.hasAttribute('data-laid'), kept = row.getBoundingClientRect().height;
    row.removeAttribute('data-laid'); const own = row.getBoundingClientRect().height;
    if (laid) row.setAttribute('data-laid', '');
    return { kept, own, laid, theme: document.documentElement.dataset.theme }; }"""
NEWEST_GAP = "() => { const f = document.querySelector('#chat-messages'); return f.scrollHeight - f.clientHeight - f.scrollTop; }"
LATE_ANSWER = """async () => {
    const frame = () => new Promise(r => requestAnimationFrame(r));
    const feed = document.querySelector('#chat-messages'); feed.style.overflowAnchor = 'none';
    feed.dispatchEvent(new WheelEvent('wheel', { deltaY: -1 }));
    feed.scrollTop = feed.scrollHeight / 2;
    for (let i = 0; i < 45; i += 1) await frame();
    const top = feed.getBoundingClientRect().top;
    const passage = [...feed.children].find(n => n.getBoundingClientRect().top >= top);
    const row = document.querySelector('#chat-messages [data-quiz-id="q-settled"]').closest('.chat-bubble');
    const take = () => ({ y: passage.getBoundingClientRect().top, h: row.getBoundingClientRect().height,
        laid: row.hasAttribute('data-laid'), above: row.getBoundingClientRect().bottom < top });
    const before = take();
    window.__ouroWs.emit('quiz_state', { quiz_id: 'q-settled', task_id: 'task-settled', chat_id: 1, state: 'answered',
        answered_index: 1, comment: 'These are my recorded words. '.repeat(100) });
    const sync = take(); await frame(); await frame(); const after = take();
    for (let i = 0; i < 45; i += 1) await frame();
    return { before, sync, after, settled: take() }; }"""
# The theme changes while the reader is mid-history, the answered question far above the passage.
THEME_AT_PASSAGE = """async () => {
    const frame = () => new Promise(r => requestAnimationFrame(r));
    const frames = async (n) => { for (let i = 0; i < n; i += 1) await frame(); };
    const feed = document.querySelector('#chat-messages'); feed.style.overflowAnchor = '';
    feed.dispatchEvent(new WheelEvent('wheel', { deltaY: -1 }));
    feed.scrollTop = feed.scrollHeight / 2;
    await frames(45);
    const top = feed.getBoundingClientRect().top;
    const passage = [...feed.children].find(n => n.getBoundingClientRect().bottom > top + 1 && n.getBoundingClientRect().height > 0);
    const at = () => passage.getBoundingClientRect().top;
    const before = at(), moves = [];
    for (const theme of ['light', 'dark']) { window.ouroTheme.set(theme); await frames(45); moves.push(at() - before); }
    return { moves }; }"""
# A row a little above the passage grows by itself (no write of the reading owner): the browser anchors the passage.
GROWS_ABOVE = """async () => {
    const frame = () => new Promise(r => requestAnimationFrame(r));
    const feed = document.querySelector('#chat-messages'); feed.style.overflowAnchor = '';
    feed.dispatchEvent(new WheelEvent('wheel', { deltaY: -1 }));
    feed.scrollTop = feed.scrollHeight / 3;
    for (let i = 0; i < 45; i += 1) await frame();
    const top = feed.getBoundingClientRect().top;
    const rows = [...feed.children];
    const passage = rows.find(n => n.getBoundingClientRect().bottom > top + 1 && n.getBoundingClientRect().height > 0);
    const above = rows[rows.indexOf(passage) - 3];
    const before = passage.getBoundingClientRect().top;
    const grown = above.getBoundingClientRect().height;
    const extra = document.createElement('p'); extra.textContent = 'Arrived later. '.repeat(60); above.append(extra);
    await frame(); await frame();
    return { moved: passage.getBoundingClientRect().top - before, grew: above.getBoundingClientRect().height - grown,
        passageLaid: passage.hasAttribute('data-laid'), control: document.querySelector('#chat-messages > .chat-load-older')?.hasAttribute('data-laid') ?? null }; }"""


@pytest.fixture(params=['webkit', 'chromium'])
def sleep_ui(request, monkeypatch):
    monkeypatch.setenv('OUROBOROS_UI_BROWSER_ENGINE', request.param)
    ui = request.getfixturevalue('subscription_ui')
    ui['engine'] = request.param
    return ui


def test_rows_asleep_wake_for_a_theme_and_for_a_late_answer_above_the_passage(sleep_ui, tmp_path):
    page = seed_page(sleep_ui, 390, 844, telegram=True)
    page.add_init_script("localStorage.setItem('ouroboros.theme', 'dark')")
    page.emulate_media(color_scheme='dark')
    start = datetime(2026, 10, 19, 15, 20, tzinfo=timezone.utc)
    rows = [{'role': 'assistant', 'msg_type': 'quiz', 'task_id': 'task-settled', 'chat_id': 1, 'ts': SETTLED['asked_at'],
             'text': SETTLED['question'], 'quiz': {k: v for k, v in SETTLED.items() if k != 'question'}}]
    rows += [{'role': 'user', 'chat_id': 1, 'text': f'Reading row {i}. ' + 'Some words to read. ' * 12,
              'ts': (start + timedelta(minutes=i)).isoformat(), 'row_address': f'main-row-{i}'} for i in range(130)]
    page.route('**/api/chat/history*', lambda route: route.fulfill(json={'messages': rows, 'progress': []}))
    facts = {'engine': sleep_ui['engine']}
    try:
        page.goto(sleep_ui['url'] + '/')
        page.wait_for_selector('#chat-input', state='attached')
        page.wait_for_function("document.querySelectorAll('#chat-messages > *').length > 120")
        rest(page)
        page.evaluate(FRAMES, 45)
        page.evaluate("document.querySelector('#chat-messages').scrollTop = document.querySelector('#chat-messages').scrollHeight")
        page.evaluate(FRAMES, 35)
        for theme in ('dark', 'light', 'dark'):
            page.evaluate("theme => window.ouroTheme.set(theme)", theme)
            page.evaluate(FRAMES, 45)
            facts.setdefault('newest', []).append(page.evaluate(NEWEST_GAP))
            facts.setdefault('themes', []).append(page.evaluate(QUESTION))
        facts['late'] = page.evaluate(LATE_ANSWER)
        facts['grows'] = page.evaluate(GROWS_ABOVE)
        facts['themeAtPassage'] = page.evaluate(THEME_AT_PASSAGE)
    finally:
        save_evidence(tmp_path, f'row-sleep-{sleep_ui["engine"]}', facts)
        print(json.dumps(facts))
        page.close()
    dark, light, again = facts['themes']
    assert dark['laid'] and light['laid'] and again['laid'], ('far from the screen, the answered question sleeps', facts)
    assert light['own'] < dark['own'], ('the light theme has no `Your choice` line: the row is shorter', facts)
    for state in facts['themes']:
        assert abs(state['kept'] - state['own']) <= 0.5, ('asleep, the row keeps its own size in either theme', facts)
    for gap in facts['newest']:
        assert abs(gap) <= 1, ('at the newest message, a theme switch keeps the reader there', facts)
    late = facts['late']
    assert late['before']['laid'] and late['before']['above'], ('the question sleeps above the passage', facts)
    assert late['settled']['h'] > late['before']['h'] + 500, ('the late answer grew the row', facts)
    for moment in ('sync', 'after', 'settled'):
        assert abs(late[moment]['y'] - late['before']['y']) <= 1, (f'the passage stays ({moment})', facts)
    assert late['settled']['laid'], ('and the row sleeps again at its new size', facts)
    grows = facts['grows']
    assert grows['grew'] > 100 and not grows['passageLaid'], ('a row above grew; the passage near the view is uncontained', facts)
    assert abs(grows['moved']) <= 1, ('the browser holds the passage when a row above it grows by itself', facts)
    assert grows['control'] in (False, None), ('a history control never sleeps', facts)
    for moved in facts['themeAtPassage']['moves']:
        assert abs(moved) <= 1, ('a theme switch keeps the passage being read where it was', facts)
