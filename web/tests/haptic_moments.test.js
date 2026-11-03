// The owner's 2A haptic moments (19.10; haptics.js): Stop on a work is felt as one rigid tap and its other actions are
// silent; an error toast plays the error haptic unless it is a notice the owner did not cause; a widget picked up to be
// moved plays a medium tap. Swarm's tick rides its click in composer.js (tests/test_chat_row_gestures_browser.py).
import assert from 'node:assert/strict';
import test from 'node:test';
import { ACTION_FINALIZE, ACTION_STOP_EARLIER, ACTION_STOP_NOW, closeTaskControlMenu, openTaskControlMenu } from '../modules/task_control_menu.js';
import { showToast } from '../modules/toast.js';
import { bindWidgetCardReorder } from '../modules/widget_reorder.js';
import { installDomDouble } from './dom_double.js';

function telegram(t) {
    const env = installDomDouble();
    const played = [];
    globalThis.window.Telegram = { WebApp: { initData: 'q', isVersionAtLeast: () => true, HapticFeedback: {
        impactOccurred: (style) => played.push(style), notificationOccurred: (type) => played.push(type),
        selectionChanged: () => played.push('tick') } } };
    const prior = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    t.after(() => { delete globalThis.window.Telegram; globalThis.IntersectionObserver = prior; env.restore(); });
    return { ...env, played };
}

test('Stop now and stopping an earlier attempt are one rigid tap; the menu\'s other actions are silent', (t) => {
    const { doc, el, played } = telegram(t);
    const anchor = el('button', { text: 'Stop…' });
    doc.body.append(anchor);
    const done = [];
    for (const action of [ACTION_STOP_NOW, ACTION_STOP_EARLIER, ACTION_FINALIZE]) {
        openTaskControlMenu(anchor, { actions: [action], onAction: (a) => done.push(a) });
        const item = doc.body.querySelector(`[data-task-control="${action}"]`);
        assert.ok(item, action);
        item.click();
        closeTaskControlMenu();
    }
    assert.deepEqual(done, [ACTION_STOP_NOW, ACTION_STOP_EARLIER, ACTION_FINALIZE]);
    assert.deepEqual(played, ['rigid', 'rigid'], 'Finalize is not a stop');
});

test('an error toast plays the error haptic; a notice the owner did not cause and other tones do not', (t) => {
    const { played } = telegram(t);
    showToast('Could not change context mode.', 'error');
    assert.deepEqual(played, ['error']);
    showToast('A host incident', 'error', { haptic: false });
    showToast('Saved', 'success');
    showToast('Heads up', 'warning');
    assert.deepEqual(played, ['error']);
});

test('a widget picked up by its handle plays one medium tap', (t) => {
    const { doc, el, played } = telegram(t);
    const handle = el('button', { attrs: { 'data-widget-reorder-handle': '' } });
    const card = el('div', { className: 'widgets-card', dataset: { widgetKey: 'a' } }, handle);
    const list = el('div', {}, card, el('div', { className: 'widgets-card', dataset: { widgetKey: 'b' } }));
    doc.body.append(list);
    bindWidgetCardReorder(list, () => ['a', 'b'], () => {});
    handle.emit('dragstart', { dataTransfer: { setData() {} } });
    assert.deepEqual(played, ['medium']);
    assert.ok(card.classList.contains('dragging'));
});
