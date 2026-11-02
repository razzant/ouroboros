// The Subagents capsule (docs/DESIGN.md "Composer owner controls"): the room's own Off · Auto · Swarm, the third
// capsule of the Context/Effort family. Closed it says the value; open, the one hint above it (ui_hint.js) says what the
// value means and how a change applies; Settings-off locks it on Off and the hint says why, with a way to Settings; a
// choice saves the room's mode.
import assert from 'node:assert/strict';
import test from 'node:test';
import { SUBAGENT_LOCKS, SUBAGENT_MEANINGS, subagentsView } from '../modules/composer_subagents.js';
import { CHANGE_RULE } from '../modules/composer_owner_controls.js';
import { hoverWindow, mountPair } from './composer_controls_fixture.js';

const value = (el) => el.querySelector('.chat-capsule-value').textContent;
const said = (hint) => ['word', 'text', 'rule', 'action'].map((part) => {
    const node = hint.querySelector(`.ui-hint-${part}`);
    return node.hidden ? '' : node.textContent;
});
const flush = () => new Promise((resolve) => setImmediate(resolve));

test('the words: one meaning per value, one rule for the three capsules, and the two reasons Settings can lock it', () => {
    assert.deepEqual(subagentsView('swarm'), {
        mode: 'swarm', stored: 'swarm', locked: false, reason: '', word: 'Swarm', meaning: SUBAGENT_MEANINGS.swarm,
        said: `Subagents: Swarm. ${SUBAGENT_MEANINGS.swarm}`,
    });
    assert.equal(CHANGE_RULE, 'Applies to new steps; running work finishes.');
    assert.equal(SUBAGENT_MEANINGS.off, 'Ouroboros works alone in this chat: no subagents, no Claude Code or Codex sessions. Reviews still run.');
    // The hint says the module's words for the chosen value (below); the three say different things, and Off stays the
    // longest: tests/test_ui_hint_browser.py opens it as Subagents' longest hint.
    assert.equal(new Set(Object.values(SUBAGENT_MEANINGS)).size, 3);
    assert.ok(['auto', 'swarm'].every((mode) => SUBAGENT_MEANINGS[mode].length < SUBAGENT_MEANINGS.off.length));
    const locked = subagentsView('swarm', 'disabled');
    assert.deepEqual([locked.mode, locked.stored, locked.word, locked.reason], ['off', 'swarm', 'Off', 'Subagents are turned off in Settings › Agents']);
    assert.equal(subagentsView('auto', 'unsaved').reason, 'Subagents are not set up yet');
    assert.deepEqual(Object.keys(SUBAGENT_LOCKS), ['disabled', 'unsaved']);
    assert.equal(subagentsView('auto', 'available').locked, false);
    assert.equal(subagentsView('auto', undefined).locked, false, 'a server that does not say locks nothing');
    assert.equal(subagentsView('surprise').mode, 'auto', 'an unknown stored value reads as the default');
});

test('closed, the capsule says the room\'s value, unlit until the room answers; it stands first in the row', () => {
    const m = mountPair({ subagents: true });
    const el = m.subagentsEl;
    const order = [el, m.contextEl, m.effortEl].map((node) => m.row.children.indexOf(node));
    assert.ok(order.every((index, i) => index >= 0 && (i === 0 || index > order[i - 1])), 'Subagents · Context · Effort');
    assert.equal(m.row.children.length, 3, 'nothing else joins the row: the hint lives inside its capsule');
    assert.equal(el.children.includes(m.subHint), true);
    assert.equal(m.subHead.getAttribute('aria-describedby'), m.subHint.id, 'a reader hears the hint as the description');
    assert.deepEqual([el.dataset.known, value(el), el.dataset.mode], ['false', 'Auto', 'auto'], 'unlit until the room answers');
    m.controls.syncState({ subagents_mode: 'swarm', subagents_availability: 'available' });
    assert.deepEqual([el.dataset.known, value(el), el.dataset.mode, el.dataset.locked], ['true', 'Swarm', 'swarm', 'false']);
    assert.equal(m.subHead.getAttribute('aria-label'), `Subagents: Swarm. ${SUBAGENT_MEANINGS.swarm}`);
    assert.ok(!m.subHead.title, 'no hover-only title: the hint says it on every surface');
    m.controls.syncState({ subagents_mode: 'off' });
    assert.deepEqual([value(el), el.dataset.mode], ['Off', 'off']);
    m.controls.syncState({ context_mode: 'low' });
    assert.equal(el.dataset.mode, 'off', 'an answer without the mode keeps the last one');
    assert.equal(m.subHint.hidden, true, 'closed, the hint says nothing');
    m.controls.destroy();
    assert.equal(el.children.includes(m.subHint), false, 'the hint goes with the capsule');
});

test('open: Off · Auto · Swarm and the hint above; a choice saves the room\'s mode, reads the room back and the hint follows', async () => {
    const m = mountPair({ subagents: true, projectId: 'p7' });
    m.controls.syncState({ projects: [{ id: 'p7', cognition: { context_mode: 'max', subagents_mode: 'auto' } }],
        subagents_availability: 'available' });
    await m.press(m.subHead);
    const el = m.subagentsEl;
    assert.equal(el.dataset.open, 'true');
    assert.equal(m.subHint.hidden, false, 'the hint above the open capsule');
    assert.deepEqual(said(m.subHint), ['Auto', SUBAGENT_MEANINGS.auto, CHANGE_RULE, '']);
    assert.deepEqual(['off', 'auto', 'swarm'].map((mode) => [m.subSeg(mode).getAttribute('aria-checked'), m.subSeg(mode).tabIndex]),
        [['false', -1], ['true', 0], ['false', -1]], 'one tab stop, the chosen value');
    await m.chooseSub('swarm');
    assert.equal(m.controls.hasPendingSave(), true, 'Send waits for it');
    assert.equal(await m.controls.pendingSave(), true);
    assert.deepEqual(m.posts.at(-1), ['/api/owner/subagents-mode', { mode: 'swarm', project_id: 'p7' }], 'this room only');
    assert.deepEqual([el.dataset.mode, value(el)], ['swarm', 'Swarm']);
    assert.deepEqual(said(m.subHint), ['Swarm', SUBAGENT_MEANINGS.swarm, CHANGE_RULE, '']);
    assert.deepEqual(m.refreshes, [true], 'the room is read again after the save');
    await m.chooseSub('swarm');
    assert.equal(m.posts.length, 1, 'the chosen value again saves nothing');
    await m.press(m.subHead);
    assert.deepEqual([el.dataset.open, m.subHint.hidden], ['false', true]);
});

test('a mouse resting on a value previews what it means; leaving the strip says the chosen one again', async () => {
    const m = mountPair({ subagents: true, win: hoverWindow(true) });
    m.controls.syncState({ subagents_mode: 'auto', subagents_availability: 'available' });
    await m.press(m.subHead);
    const strip = m.subagentsEl.querySelector('.chat-subagents-strip');
    await m.preview(strip, 'swarm');
    assert.deepEqual(said(m.subHint), ['Swarm', SUBAGENT_MEANINGS.swarm, CHANGE_RULE, '']);
    assert.equal(m.posts.length, 0, 'a preview chooses nothing');
    await m.previewEnd(strip);
    assert.deepEqual(said(m.subHint), ['Auto', SUBAGENT_MEANINGS.auto, CHANGE_RULE, '']);
});

test('a refused save keeps the saved value and says why', async () => {
    const m = mountPair({ subagents: true });
    m.controls.syncState({ subagents_mode: 'auto', subagents_availability: 'available' });
    m.subagentAnswer.value = Promise.reject(new Error('Choose off, auto or swarm.'));
    await m.press(m.subHead);
    await m.chooseSub('off');
    assert.equal(await m.controls.pendingSave(), false);
    assert.deepEqual(m.toasts.at(-1), ['Choose off, auto or swarm.', 'error']);
    assert.deepEqual([m.subagentsEl.dataset.mode, value(m.subagentsEl)], ['auto', 'Auto'], 'never shown as saved');
    assert.deepEqual(said(m.subHint), ['Auto', SUBAGENT_MEANINGS.auto, CHANGE_RULE, '']);
});

test('Settings-off: Off with a lock, only Off can be chosen, the hint says why with Open Settings; the room keeps its value', async () => {
    const m = mountPair({ subagents: true });
    const el = m.subagentsEl;
    m.controls.syncState({ subagents_mode: 'swarm', subagents_availability: 'disabled' });
    assert.deepEqual([el.dataset.locked, el.dataset.mode, value(el)], ['true', 'off', 'Off']);
    assert.equal(m.subHead.getAttribute('aria-label'), 'Subagents: Off. Subagents are turned off in Settings › Agents.');
    await m.press(m.subHead);
    assert.deepEqual(['auto', 'swarm'].map((mode) => [m.subSeg(mode).dataset.disabled, m.subSeg(mode).getAttribute('aria-disabled')]),
        [['true', 'true'], ['true', 'true']]);
    assert.equal(m.subSeg('off').getAttribute('aria-checked'), 'true');
    await m.chooseSub('swarm');
    assert.equal(m.posts.length, 0, 'a value Settings rules out saves nothing');
    assert.deepEqual(said(m.subHint), ['', 'Subagents are turned off in Settings › Agents.', '', 'Open Settings'],
        'the reason in the rule\'s place, and the way to Settings');
    // A press comes before its click (a browser's order): the hint is the capsule's own, so a press on Open Settings
    // leaves the strip and its button there for the click.
    const action = m.subHint.querySelector('.ui-hint-action');
    m.docDown(action);
    assert.deepEqual([el.dataset.open, m.subHint.hidden], ['true', false], 'a press on the hint is not an outside press');
    action.click();
    assert.deepEqual(m.settingsOpened, ['agents']);
    assert.deepEqual([el.dataset.open, m.subHint.hidden], ['false', true], 'the way to Settings closes the strip');
    m.controls.syncState({ subagents_availability: 'unsaved' });
    assert.equal(said(m.subHint)[1], 'Subagents are not set up yet.');
    m.controls.syncState({ subagents_availability: 'available' });
    assert.deepEqual([el.dataset.locked, el.dataset.mode, value(el)], ['false', 'swarm', 'Swarm'], 'the room\'s own value applies again');
    assert.deepEqual(said(m.subHint), ['Swarm', SUBAGENT_MEANINGS.swarm, CHANGE_RULE, ''], 'and the hint says it again');
});

test('keyboard: arrows, Home and End choose among the values that can be chosen', async () => {
    const m = mountPair({ subagents: true });
    m.controls.syncState({ subagents_mode: 'off', subagents_availability: 'available' });
    await m.press(m.subHead, 0);
    assert.equal(m.subKey('off', 'ArrowRight').defaultPrevented, true);
    await flush();
    assert.deepEqual(m.posts.at(-1), ['/api/owner/subagents-mode', { mode: 'auto' }]);
    await m.controls.pendingSave();
    m.subKey('auto', 'End');
    await m.controls.pendingSave();
    assert.deepEqual(m.posts.at(-1), ['/api/owner/subagents-mode', { mode: 'swarm' }]);
    assert.equal(m.subKey('swarm', 'Enter').defaultPrevented, false, 'other keys are the browser\'s');
    m.controls.syncState({ subagents_availability: 'disabled' });
    const before = m.posts.length;
    assert.equal(m.subKey('off', 'ArrowRight').defaultPrevented, true);
    assert.equal(m.posts.length, before, 'locked: nothing past Off');
});

test('one capsule is open at a time across all three, each with its own hint', async () => {
    const m = mountPair({ subagents: true });
    m.controls.syncState({ subagents_mode: 'auto', subagents_availability: 'available' });
    await m.press(m.subHead);
    assert.equal(m.subagentsEl.dataset.open, 'true');
    m.docDown(m.contextHead);
    await m.press(m.contextHead);
    assert.deepEqual([m.subagentsEl.dataset.open, m.contextEl.dataset.open, m.subHint.hidden, m.contextHint.hidden],
        ['false', 'true', true, false]);
    m.docDown(m.effortHead);
    await m.press(m.effortHead);
    assert.deepEqual([m.contextEl.dataset.open, m.effortEl.dataset.open, m.contextHint.hidden, m.effortHint.hidden],
        ['false', 'true', true, false]);
    m.docDown(m.subHead);
    await m.press(m.subHead);
    assert.deepEqual([m.effortEl.dataset.open, m.subagentsEl.dataset.open, m.effortHint.hidden, m.subHint.hidden],
        ['false', 'true', true, false]);
});

test('painting the same state again writes no words: the row\'s text observer refits on every write', async () => {
    const m = mountPair({ subagents: true });
    m.controls.syncState({ subagents_mode: 'auto', subagents_availability: 'available' });
    await m.press(m.subHead);
    const writes = [];
    const spy = (node) => {
        const { get, set } = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(node), 'textContent');
        Object.defineProperty(node, 'textContent', { configurable: true, get() { return get.call(this); },
            set(text) { writes.push(text); set.call(this, text); } });
    };
    ['.chat-capsule-value', '.ui-hint-word', '.ui-hint-text', '.ui-hint-rule', '.ui-hint-action']
        .forEach((selector) => spy(m.subagentsEl.querySelector(selector)));
    m.controls.syncState({ subagents_mode: 'auto', subagents_availability: 'available' });
    m.controls.syncState({ subagents_availability: 'available' });
    assert.deepEqual(writes, [], 'an unchanged state repaints nothing a text observer sees');
    m.controls.syncState({ subagents_mode: 'swarm' });
    assert.deepEqual(writes, ['Swarm', 'Swarm', SUBAGENT_MEANINGS.swarm], 'a change writes each changed word once');
});
