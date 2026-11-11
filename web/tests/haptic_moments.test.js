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
    globalThis.requestAnimationFrame = () => 1;
    globalThis.cancelAnimationFrame = () => {};
    t.after(() => { delete globalThis.requestAnimationFrame; delete globalThis.cancelAnimationFrame; });
    const handle = el('button', { attrs: { 'data-widget-reorder-handle': '' } });
    const card = el('div', { className: 'widgets-card', dataset: { widgetKey: 'a' } }, handle);
    const other = el('div', { className: 'widgets-card', dataset: { widgetKey: 'b' } });
    const list = el('div', {}, card, other);
    doc.body.append(list);
    const plan = { placements: [{ left: 0, top: 0, width: 100, span: 1 }, { left: 114, top: 0, width: 100, span: 1 }] };
    const board = { laid: () => ({ plan, items: [card, other], planOrder: () => plan }), carry() {}, announce() {} };
    bindWidgetCardReorder(list, () => ['a', 'b'], () => {}, board);
    // The pointer carry (no HTML5 drag): a press on the grip picks the card up once it has moved far enough.
    handle.emit('pointerdown', { button: 0, pointerId: 1, pointerType: 'touch', clientX: 10, clientY: 10 });
    handle.emit('pointermove', { pointerId: 1, clientX: 12, clientY: 11 });
    assert.deepEqual(played, [], 'a press that has not travelled yet picks nothing up');
    handle.emit('pointermove', { pointerId: 1, clientX: 20, clientY: 10 });
    assert.deepEqual(played, ['medium']);
    assert.ok(card.hasAttribute('data-widget-lifted'));
    handle.emit('pointermove', { pointerId: 1, clientX: 40, clientY: 12 });
    handle.emit('pointerup', { pointerId: 1, clientX: 40, clientY: 12 });
    assert.deepEqual(played, ['medium'], 'carrying and dropping it are silent');
});
