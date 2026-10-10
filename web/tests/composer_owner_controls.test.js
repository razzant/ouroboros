// The composer's owner controls (docs/DESIGN.md "Composer owner controls"): the effort
// range's handle math, the open/pin state machine with and without a hovering pointer,
// keyboard, serialized saves with a refused save returning to the server's value, the
// one global value from /api/state, teardown — and the context-mode toggle it took over.
import test from 'node:test';
import assert from 'node:assert/strict';

import { ElementStub } from './chat_dom_fixture.js';
import {
    applyHandle, createComposerOwnerControls, createEffortRangeControl, nearestHandle, snapRange,
} from '../modules/composer_owner_controls.js';

/* A document whose listeners can be fired, and a window that may or may not hover. */
function stubDocument() {
    const listeners = new Map();
    const doc = {
        byId: new Map(), activeElement: null,
        createElement: (tag) => new ElementStub(tag, doc),
        addEventListener(type, fn) { listeners.set(type, [...(listeners.get(type) || []), fn]); },
        removeEventListener(type, fn) { listeners.set(type, (listeners.get(type) || []).filter((f) => f !== fn)); },
        fire(type, event) { for (const fn of listeners.get(type) || []) fn(event); },
        count: (type) => (listeners.get(type) || []).length,
    };
    return doc;
}
const hoverWindow = (matches) => ({ matchMedia: (query) => ({ matches: matches && query === '(hover: hover) and (pointer: fine)' }) });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function mount({ hover = false, save, win } = {}) {
    const doc = stubDocument();
    const row = doc.createElement('div');
    row.className = 'chat-toolbar-row';
    const toasts = [];
    const refreshes = [];
    const layouts = [];
    const saves = [];
    const control = createEffortRangeControl({
        row, doc, win: win === undefined ? hoverWindow(hover) : win,
        saveEffortRange: save || (async (triple) => { saves.push(triple); return { ok: true, effort_range: triple }; }),
        showToast: (message, tone) => toasts.push([message, tone]),
        refreshState: (force) => refreshes.push(force),
        onLayout: () => layouts.push(control?.isOpen?.()),
    });
    const el = control.el;
    const segs = el.querySelectorAll('.chat-effort-seg');
    // The strip has layout: seven 40px segments, so a boundary is x / 40.
    segs.forEach((seg, i) => { seg.offsetLeft = i * 40; seg.offsetWidth = 40; });
    const head = el.querySelector('.chat-effort-head');
    const strip = el.querySelector('[data-effort-strip]');
    const handles = { min: el.querySelector('.chat-effort-min'), rec: el.querySelector('.chat-effort-rec'), max: el.querySelector('.chat-effort-max') };
    const fire = (node, type, event) => { for (const fn of node.listeners.get(type) || []) fn(event); };
    const last = (node, type) => (node.listeners.get(type) || []).at(-1);
    const pointer = (clientX, { handle = null, pointerId = 1 } = {}) => ({
        button: 0, clientX, pointerId, preventDefault() {},
        target: { closest: (selector) => (selector === '[data-handle]' ? handle : null) },
    });
    return {
        control, el, row, doc, head, strip, handles, segs, toasts, refreshes, layouts, saves,
        headClick: (detail = 1) => fire(head, 'click', { detail }),
        key: (which, key) => fire(handles[which], 'keydown', { key, currentTarget: handles[which], preventDefault() {} }),
        down: (x, options) => fire(strip, 'pointerdown', pointer(x, options)),
        move: (x, pointerId = 1) => last(strip, 'pointermove')?.(pointer(x, { pointerId })),
        up: (x, pointerId = 1) => last(strip, 'pointerup')?.(pointer(x, { pointerId })),
        cancel: (pointerId = 1) => last(strip, 'pointercancel')?.(pointer(0, { pointerId })),
        enter: (pointerType = 'mouse') => fire(el, 'pointerenter', { pointerType }),
        leave: (pointerType = 'mouse') => fire(el, 'pointerleave', { pointerType }),
        outside: () => doc.fire('pointerdown', { target: new ElementStub('div', doc) }),
        inside: () => doc.fire('pointerdown', { target: head }),
        escape: () => doc.fire('keydown', { key: 'Escape' }),
        reset: () => fire(el.querySelector('.chat-effort-reset'), 'click', {}),
        settle: () => control.pendingSave(),
    };
}

test('handle math: min and max push the recommended level, the recommended level stays between them', () => {
    const base = { min: 1, rec: 2, max: 3 };
    assert.deepEqual(applyHandle(base, 'rec', 5), { min: 1, rec: 3, max: 3 }, 'rec is clamped into the range');
    assert.deepEqual(applyHandle(base, 'rec', 0), { min: 1, rec: 1, max: 3 });
    assert.deepEqual(applyHandle(base, 'min', 3), { min: 3, rec: 3, max: 3 }, 'min pushes rec up');
    assert.deepEqual(applyHandle(base, 'min', 6), { min: 3, rec: 3, max: 3 }, 'min never passes max');
    assert.deepEqual(applyHandle(base, 'max', 1), { min: 1, rec: 1, max: 1 }, 'max pushes rec down');
    assert.deepEqual(applyHandle(base, 'max', -4), { min: 1, rec: 1, max: 1 }, 'max never passes min');
    assert.deepEqual(applyHandle(base, 'min', 0), { min: 0, rec: 2, max: 3 });
    assert.deepEqual(applyHandle(base, 'max', 6), { min: 1, rec: 2, max: 6 });
    assert.deepEqual(snapRange({ min: 0.6, rec: 0.2, max: 2.4 }), { min: 1, rec: 1, max: 2 }, 'a snap keeps the order');
    // A tap inside the range moves the recommended level, outside it the bracket on that side.
    assert.equal(nearestHandle(base, 2), 'rec');
    assert.equal(nearestHandle(base, 1), 'rec');
    assert.equal(nearestHandle(base, 0), 'min');
    assert.equal(nearestHandle(base, 6), 'max');
    // Coinciding levels: the brackets stay reachable from either side of the pill.
    assert.equal(nearestHandle({ min: 3, rec: 3, max: 3 }, 3), 'rec');
    assert.equal(nearestHandle({ min: 3, rec: 3, max: 3 }, 2), 'min');
    assert.equal(nearestHandle({ min: 3, rec: 3, max: 3 }, 4), 'max');
});

test('the control mounts after the pills, closed, with three sliders out of the tab order and the ring glyph', () => {
    const m = mount();
    assert.equal(m.el.dataset.open, 'false');
    assert.equal(m.el.getAttribute('data-effort-range'), '', 'the i18n DOM scope of the strip words');
    assert.equal(m.head.getAttribute('aria-expanded'), 'false');
    for (const which of ['min', 'rec', 'max']) {
        assert.equal(m.handles[which].getAttribute('role'), 'slider');
        assert.equal(m.handles[which].tabIndex, -1);
        assert.equal(m.handles[which].getAttribute('aria-valuemax'), '6');
    }
    assert.deepEqual(m.control.shown(), { min: 1, rec: 2, max: 3 }, 'the default before the first /api/state');
    assert.equal(m.el.dataset.known, 'false');
    assert.equal(m.handles.rec.getAttribute('aria-valuetext'), 'Medium');
    assert.match(m.head.title, /^Effort: Medium, range Low–High\. Applies to new work\.$/);
    assert.match(m.el.querySelector('[data-ring-band]').getAttribute('d'), /^M[\d.]+ [\d.]+ A70 70 0 0 1 /);
    assert.equal(m.segs.length, 7);
    assert.deepEqual(m.segs.map((seg) => seg.dataset.in), ['false', 'true', 'true', 'true', 'false', 'false', 'false']);
    assert.deepEqual(m.segs.map((seg) => seg.dataset.rec), ['false', 'false', 'true', 'false', 'false', 'false', 'false']);
    assert.ok(m.row.children.includes(m.el));
});

test('/api/state is the one global value: it paints every composer, a stored minimal shows at Low', () => {
    const m = mount();
    m.control.syncState({ effort_range: { min: 'minimal', recommended: 'high', max: 'ultra' } });
    assert.equal(m.el.dataset.known, 'true');
    assert.deepEqual(m.control.shown(), { min: 1, rec: 3, max: 6 });
    assert.deepEqual(m.control.stored(), { min: 'minimal', recommended: 'high', max: 'ultra' });
    assert.equal(m.handles.min.getAttribute('aria-valuetext'), 'Low');
    assert.equal(m.el.querySelector('.chat-effort-reset').dataset.atDefault, 'false');
    m.control.syncState({ context_mode: 'max' });
    assert.deepEqual(m.control.shown(), { min: 1, rec: 3, max: 6 }, 'a snapshot without the range changes nothing');
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    assert.equal(m.el.querySelector('.chat-effort-reset').dataset.atDefault, 'true');
});

test('without a hovering pointer the press toggles; Enter/Space open and focus the recommended handle; Esc returns focus', () => {
    const m = mount({ hover: false });
    assert.equal(m.control.hoverCapable, false);
    assert.equal(m.el.listeners.get('pointerenter'), undefined, 'no hover listeners without the media');
    m.enter();
    assert.equal(m.control.isOpen(), false, 'a pointer arriving does not open');
    m.headClick(1);
    assert.equal(m.control.isOpen(), true);
    assert.equal(m.head.getAttribute('aria-expanded'), 'true');
    assert.equal(m.handles.rec.tabIndex, 0);
    assert.deepEqual(m.layouts, [true], 'the chat recomputes its composer reserve');
    assert.notEqual(m.doc.activeElement, m.handles.rec, 'a press does not steal focus');
    m.headClick(1);
    assert.equal(m.control.isOpen(), false);
    m.headClick(0);  // Enter / Space
    assert.equal(m.control.isOpen(), true);
    assert.equal(m.doc.activeElement, m.handles.rec);
    m.escape();
    assert.equal(m.control.isOpen(), false);
    assert.equal(m.doc.activeElement, m.head);
    assert.equal(m.handles.rec.tabIndex, -1);
    m.headClick(1);
    m.inside();
    assert.equal(m.control.isOpen(), true, 'a press inside keeps it open');
    m.outside();
    assert.equal(m.control.isOpen(), false, 'an outside press closes');
});

test('with a hovering pointer: opens after ~160 ms, closes ~450 ms after leaving unless pinned by a press', async () => {
    const m = mount({ hover: true });
    assert.equal(m.control.hoverCapable, true);
    m.enter('touch');
    await sleep(200);
    assert.equal(m.control.isOpen(), false, 'a touch pointer never hover-opens');
    m.enter();
    assert.equal(m.control.isOpen(), false, 'not yet');
    await sleep(200);
    assert.equal(m.control.isOpen(), true);
    assert.equal(m.control.isPinned(), false);
    m.leave();
    await sleep(520);
    assert.equal(m.control.isOpen(), false, 'closed after the leave delay');
    m.enter();
    await sleep(200);
    m.headClick(1);
    assert.equal(m.control.isPinned(), true, 'a press pins the hover-opened strip');
    m.leave();
    await sleep(520);
    assert.equal(m.control.isOpen(), true, 'pinned: leaving does not close');
    m.headClick(1);
    assert.equal(m.control.isOpen(), false, 'a second press unpins and closes');
    m.enter();
    m.leave();
    await sleep(200);
    assert.equal(m.control.isOpen(), false, 'leaving before the delay cancels the open');
    m.headClick(1);
    assert.equal(m.control.isOpen(), true, 'a press on the closed button pins it open');
    assert.equal(m.control.isPinned(), true);
    m.escape();
    assert.equal(m.control.isOpen(), false);
});

test('keyboard: arrows, Home, End, PageUp/PageDown move a handle, save the full triple on each step, and push neighbours', async () => {
    const m = mount();
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.headClick(0);
    m.key('rec', 'ArrowRight');
    assert.deepEqual(m.control.shown(), { min: 1, rec: 3, max: 3 });
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'low', recommended: 'high', max: 'high' });
    m.key('rec', 'ArrowUp');
    await m.settle();
    assert.deepEqual(m.control.shown(), { min: 1, rec: 3, max: 3 }, 'rec never passes max');
    assert.equal(m.saves.length, 1, 'an unchanged range saves nothing');
    m.key('max', 'End');
    m.key('min', 'Home');
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'none', recommended: 'high', max: 'ultra' });
    m.key('min', 'PageUp');
    m.key('min', 'PageUp');
    m.key('min', 'PageUp');
    m.key('min', 'ArrowRight');
    await m.settle();
    assert.deepEqual(m.control.shown(), { min: 4, rec: 4, max: 6 }, 'min pushes rec along');
    assert.deepEqual(m.saves.at(-1), { min: 'xhigh', recommended: 'xhigh', max: 'ultra' });
    m.key('max', 'PageDown');
    m.key('max', 'ArrowDown');
    m.key('max', 'ArrowLeft');
    await m.settle();
    assert.deepEqual(m.control.shown(), { min: 4, rec: 4, max: 4 }, 'max comes down to min and never passes it');
    assert.equal(m.handles.max.getAttribute('aria-valuenow'), '4');
    assert.equal(m.handles.max.getAttribute('aria-valuetext'), 'X-High');
    assert.deepEqual(m.saves.at(-1), { min: 'xhigh', recommended: 'xhigh', max: 'xhigh' });
    assert.equal(m.refreshes.at(-1), true, 'every settled save re-reads /api/state for every composer');
});

test('a stored tier outside the owner tiers survives until the owner moves that handle', async () => {
    const m = mount();
    m.control.syncState({ effort_range: { min: 'minimal', recommended: 'medium', max: 'high' } });
    m.headClick(0);
    m.key('max', 'ArrowRight');
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'minimal', recommended: 'medium', max: 'xhigh' }, 'min keeps its stored minimal');
    m.key('min', 'ArrowLeft');
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'none', recommended: 'medium', max: 'xhigh' }, 'moved: rewritten to an owner tier');
    m.control.syncState({ effort_range: { min: 'minimal', recommended: 'medium', max: 'high' } });
    m.key('rec', 'ArrowLeft');
    m.key('rec', 'ArrowLeft');
    await m.settle();
    assert.deepEqual(m.control.shown(), { min: 1, rec: 1, max: 3 }, 'rec stops at the shown minimum');
    assert.deepEqual(m.saves.at(-1), { min: 'minimal', recommended: 'low', max: 'high' }, 'the untouched minimum keeps minimal');
});

test('pointer: a press inside the range moves the recommended level, outside it the nearest bracket; drag keeps the grab offset', async () => {
    const m = mount();
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.headClick(1);
    m.down(60);                       // inside: segment 1 -> rec
    assert.deepEqual(m.control.shown(), { min: 1, rec: 2, max: 3 }, 'a preview is not a commit');
    assert.equal(m.handles.rec.dataset.dragging, 'true');
    assert.equal(m.doc.activeElement, m.handles.rec);
    m.move(180);                      // segment 4
    m.up(180);
    await m.settle();
    assert.deepEqual(m.control.shown(), { min: 1, rec: 3, max: 3 }, 'rec dragged past max stops at max');
    assert.equal(m.handles.rec.dataset.dragging, 'false');
    assert.deepEqual(m.saves.at(-1), { min: 'low', recommended: 'high', max: 'high' });
    m.down(250);                      // outside above: segment 6 -> max
    m.up(250);
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'low', recommended: 'high', max: 'ultra' });
    m.down(10);                       // outside below: segment 0 -> min
    m.up(10);
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'none', recommended: 'high', max: 'ultra' });
    // Grab the max bracket (boundary x=280 is level 7-1 = 6) and drag it by two segments: the
    // offset between the grab point and the bracket is kept, so the bracket lands at 4.
    m.down(285, { handle: m.handles.max });
    assert.equal(m.handles.max.dataset.dragging, 'true');
    m.move(205);
    m.up(205);
    await m.settle();
    assert.deepEqual(m.control.shown(), { min: 0, rec: 3, max: 4 });
    assert.deepEqual(m.saves.at(-1), { min: 'none', recommended: 'high', max: 'xhigh' });
    // pointercancel cancels the preview without a save.
    const before = m.saves.length;
    m.down(60);
    m.move(0);
    assert.deepEqual(m.control.shown(), { min: 0, rec: 3, max: 4 }, 'shown is the committed value');
    m.cancel();
    assert.equal(m.handles.rec.dataset.dragging, 'false');
    await m.settle();
    assert.equal(m.saves.length, before);
    assert.deepEqual(m.control.shown(), { min: 0, rec: 3, max: 4 });
    // A second pointer's events are ignored while the first drags; a press on the recommended
    // level's own segment released there changes nothing and saves nothing.
    m.down(140);
    m.move(200, 2);
    m.up(200, 2);
    assert.equal(m.handles.rec.dataset.dragging, 'true');
    m.up(140);
    await m.settle();
    assert.equal(m.saves.length, before, 'released where it was pressed: no change, no save');
});

test('saves are serialized, last wins; Send can wait for them; a refused save returns to the server value and says why', async () => {
    const pending = [];
    const m = mount({
        save: (triple) => new Promise((resolve, reject) => pending.push({ triple, resolve, reject })),
    });
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.headClick(0);
    assert.equal(m.control.hasPendingSave(), false);
    m.key('rec', 'ArrowRight');
    assert.equal(m.control.hasPendingSave(), true);
    assert.equal(m.el.dataset.saving, 'true');
    m.key('rec', 'ArrowLeft');
    m.key('max', 'ArrowRight');
    assert.equal(pending.length, 1, 'one request in flight; later gestures wait');
    assert.deepEqual(pending[0].triple, { min: 'low', recommended: 'high', max: 'high' });
    m.control.syncState({ effort_range: { min: 'none', recommended: 'none', max: 'none' } });
    assert.deepEqual(m.control.shown(), { min: 1, rec: 2, max: 4 }, 'a stale poll never overwrites a save in flight');
    const waiter = m.control.pendingSave();
    pending[0].resolve({ ok: true, effort_range: pending[0].triple });
    await sleep(0);
    assert.equal(pending.length, 2, 'the latest queued triple follows, the superseded one never leaves');
    assert.deepEqual(pending[1].triple, { min: 'low', recommended: 'medium', max: 'xhigh' });
    pending[1].resolve({ ok: true, effort_range: pending[1].triple });
    assert.equal(await waiter, true);
    assert.equal(m.control.hasPendingSave(), false);
    assert.equal(m.el.dataset.saving, 'false');
    assert.deepEqual(m.control.stored(), { min: 'low', recommended: 'medium', max: 'xhigh' });
    assert.deepEqual(m.refreshes, [true]);
    // A refusal: the server's sentence as a toast, the control back at the server's value.
    m.key('min', 'ArrowLeft');
    assert.deepEqual(m.control.shown(), { min: 0, rec: 2, max: 4 });
    const refused = m.control.pendingSave();
    pending[2].reject(Object.assign(new Error('Effort range must be ordered: minimum at most the recommended level.'), { status: 400 }));
    assert.equal(await refused, false);
    assert.deepEqual(m.toasts, [['Effort range must be ordered: minimum at most the recommended level.', 'error']]);
    assert.deepEqual(m.control.shown(), { min: 1, rec: 2, max: 4 }, 'the unsaved preview is gone');
    assert.deepEqual(m.control.stored(), { min: 'low', recommended: 'medium', max: 'xhigh' });
    assert.equal(await m.control.pendingSave(), false, 'the last outcome stays readable');
    // The reset returns to Low · Medium · High as one full triple.
    m.reset();
    pending[3].resolve({ ok: true, effort_range: pending[3].triple });
    assert.equal(await m.control.pendingSave(), true);
    assert.deepEqual(pending[3].triple, { min: 'low', recommended: 'medium', max: 'high' });
    assert.equal(m.el.querySelector('.chat-effort-reset').dataset.atDefault, 'true');
});

test('an edit made while saves are on their way builds on the latest choice, never on an older answer', async () => {
    const pending = [];
    const m = mount({ save: (triple) => new Promise((resolve, reject) => pending.push({ triple, resolve, reject })) });
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.headClick(0);
    m.key('min', 'Home');        // in flight: none · medium · high
    m.key('rec', 'ArrowRight');  // queued:    none · high · high
    pending[0].resolve({ ok: true, effort_range: pending[0].triple });
    await sleep(0);
    assert.deepEqual(m.control.shown(), { min: 0, rec: 3, max: 3 }, 'the older answer does not pull the draft back');
    m.key('max', 'End');         // while the second save is in flight
    pending[1].resolve({ ok: true, effort_range: pending[1].triple });
    await sleep(0);
    pending[2].resolve({ ok: true, effort_range: pending[2].triple });
    assert.equal(await m.control.pendingSave(), true);
    assert.deepEqual(pending.map((p) => p.triple), [
        { min: 'none', recommended: 'medium', max: 'high' },
        { min: 'none', recommended: 'high', max: 'high' },
        { min: 'none', recommended: 'high', max: 'ultra' },
    ]);
    assert.deepEqual(m.control.stored(), { min: 'none', recommended: 'high', max: 'ultra' });
    assert.deepEqual(m.control.shown(), { min: 0, rec: 3, max: 6 });
    // A refusal drops edits queued on top of it and returns to the server's value.
    m.key('rec', 'ArrowLeft');
    m.key('rec', 'ArrowLeft');
    pending[3].reject(new Error('Could not save.'));
    assert.equal(await m.control.pendingSave(), false);
    assert.equal(pending.length, 4, 'the edit queued on the refused draft never leaves');
    assert.deepEqual(m.control.shown(), { min: 0, rec: 3, max: 6 });
});

test('destroy removes the element, its document listeners and its timers', async () => {
    const m = mount({ hover: true });
    assert.equal(m.doc.count('pointerdown'), 1);
    assert.equal(m.doc.count('keydown'), 1);
    m.enter();
    m.control.destroy();
    assert.equal(m.doc.count('pointerdown'), 0);
    assert.equal(m.doc.count('keydown'), 0);
    assert.ok(!m.row.children.includes(m.el));
    await sleep(220);
    assert.equal(m.control.isOpen(), false, 'the hover timer died with the control');
});

test('the context-mode toggle posts the owner endpoint, shows a refusal, and always re-reads /api/state', async () => {
    const doc = stubDocument();
    const row = doc.createElement('div');
    const contextMode = doc.createElement('div');
    contextMode.dataset.contextMode = 'max';
    const seg = { closest: (selector) => (selector === '.chat-seg' ? seg : null), dataset: { mode: 'low' } };
    const posts = [];
    const toasts = [];
    const refreshes = [];
    let reply = { ok: true };
    const controls = createComposerOwnerControls({
        row, doc, win: hoverWindow(false), byId: (suffix) => (suffix === 'context-mode' ? contextMode : null),
        apiFetch: async (url, init) => { posts.push([url, JSON.parse(init.body)]); return reply; },
        saveEffortRange: async (triple) => ({ effort_range: triple }),
        showToast: (message, tone) => toasts.push([message, tone]),
        refreshState: (force) => refreshes.push(force),
    });
    const click = (target) => Promise.all((contextMode.listeners.get('click') || []).map((fn) => fn({ target })));
    await click(seg);
    assert.deepEqual(posts, [['/api/owner/context-mode', { mode: 'low' }]]);
    assert.equal(contextMode.dataset.contextMode, 'low');
    assert.equal(contextMode.dataset.disabled, 'false');
    assert.deepEqual(refreshes, [true]);
    await click(seg);
    assert.equal(posts.length, 1, 'the current mode is not re-posted');
    seg.dataset.mode = 'max';
    reply = { ok: false, json: async () => ({ error: 'Context mode can only be lowered while Ouroboros is idle.' }) };
    await click(seg);
    assert.equal(contextMode.dataset.contextMode, 'low', 'a refusal leaves the shown value');
    assert.deepEqual(toasts, [['Context mode can only be lowered while Ouroboros is idle.', 'error']]);
    assert.deepEqual(refreshes, [true, true]);
    assert.equal(controls.hasPendingSave(), false);
    assert.equal(await controls.pendingSave(), true);
    controls.destroy();
    assert.equal(contextMode.listeners.get('click').length, 1, 'the stub keeps the registration; removal is a no-op there');
});
