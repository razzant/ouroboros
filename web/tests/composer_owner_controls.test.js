// The composer's owner controls (docs/DESIGN.md "Composer owner controls"): two labelled
// capsules. The effort range's handle math, the open/pin state machine with and without a
// hovering pointer, keyboard, serialized saves with a refused save returning to the server's
// value, independent room readback, teardown; the context capsule's three modes, its
// save and a scrolling strip that keeps the chosen mode in view; one capsule open at a time, and
// the names that give way on a narrow line.
import test from 'node:test';
import assert from 'node:assert/strict';

import { ElementStub } from './chat_dom_fixture.js';
import { CHANGE_RULE, applyHandle, createEffortRangeControl, snapRange } from '../modules/composer_owner_controls.js';
import { hoverWindow, mount, mountPair, sleep, stubDocument } from './composer_controls_fixture.js';

test('primary and helper bounds are independent, and helper bounds cannot cross', () => {
    const base = { min: 1, rec: 2, max: 3 };
    for (const rec of [0, 3, 6]) assert.deepEqual(applyHandle(base, 'rec', rec), { ...base, rec });
    assert.deepEqual(applyHandle(base, 'min', 6), { min: 3, rec: 2, max: 3 });
    assert.deepEqual(applyHandle(base, 'max', -4), { min: 1, rec: 2, max: 1 });
    assert.deepEqual(applyHandle(base, 'rec', 9), { min: 1, rec: 6, max: 3 });
    assert.deepEqual(snapRange({ min: 0.6, rec: 0.2, max: 2.4 }), { min: 1, rec: 0, max: 2 });
});

test('the control mounts after the context capsule, closed: ring, name and the recommended level only; three sliders out of the tab order', () => {
    const m = mount();
    assert.equal(m.el.dataset.open, 'false');
    assert.ok(m.el.classList.contains('chat-capsule'));
    assert.match(m.el.innerHTML, /class="chat-capsule-name">Effort</);
    // Closed, the capsule names only the recommended level: no range in words (the ring and the
    // button's name carry the range).
    assert.equal(m.el.querySelector('.chat-capsule-value').textContent, 'Medium');
    assert.equal(m.el.querySelector('.chat-capsule-value').dataset.i18nSkip, '', 'a catalog word the overlay leaves alone');
    assert.equal(m.el.getAttribute('data-effort-range'), '', 'the i18n DOM scope of the strip words');
    assert.equal(m.head.getAttribute('aria-expanded'), 'false');
    // The recommended level may go anywhere (it pushes the brackets); a bracket stops at the other one.
    const bounds = { min: ['0', '3'], rec: ['0', '6'], max: ['1', '6'] };
    for (const which of ['min', 'rec', 'max']) {
        assert.equal(m.handles[which].getAttribute('role'), 'slider');
        assert.equal(m.handles[which].tabIndex, -1);
        assert.deepEqual([m.handles[which].getAttribute('aria-valuemin'), m.handles[which].getAttribute('aria-valuemax')], bounds[which], which);
    }
    assert.deepEqual(m.control.shown(), { min: 1, rec: 2, max: 3 }, 'the default before the first /api/state');
    assert.equal(m.el.dataset.known, 'false');
    assert.equal(m.handles.rec.getAttribute('aria-valuetext'), 'Medium');
    assert.equal(m.head.getAttribute('aria-label'), 'Primary effort: Medium; helpers Low–High.', 'the closed button says the state, not just "Effort"');
    assert.ok(!m.head.title, 'no hover-only title: the hint says it on every surface');
    const hint = m.el.querySelector('.ui-hint');
    assert.equal(m.head.getAttribute('aria-describedby'), hint.id, 'a reader hears the hint as the description');
    assert.deepEqual([hint.hidden, ...['word', 'text', 'rule'].map((part) => hint.querySelector(`.ui-hint-${part}`).textContent)],
        [true, 'Medium', 'Primary effort; helpers between Low and High.', CHANGE_RULE], 'closed, the hint waits with its words');
    assert.match(m.el.querySelector('[data-ring-band]').getAttribute('d'), /^M[\d.]+ [\d.]+ A70 70 0 0 1 /);
    assert.equal(m.segs.length, 7);
    // The strip is the seven levels alone: no title word of its own, no separator element, no reset button.
    for (const gone of ['.chat-effort-label', '.chat-effort-sep', '.chat-effort-reset']) assert.equal(m.el.querySelector(gone), null, gone);
    assert.deepEqual(m.segs.map((seg) => seg.dataset.in), ['false', 'true', 'true', 'true', 'false', 'false', 'false']);
    assert.deepEqual(m.segs.map((seg) => seg.dataset.rec), ['false', 'false', 'true', 'false', 'false', 'false', 'false']);
    assert.ok(m.row.children.includes(m.el));
});

test('an authoritative room value paints the control; a stored minimal shows at Low', () => {
    const m = mount();
    m.control.syncState({ effort_range: { min: 'minimal', recommended: 'high', max: 'ultra' } });
    assert.equal(m.el.dataset.known, 'true');
    assert.deepEqual(m.control.shown(), { min: 1, rec: 3, max: 6 });
    assert.deepEqual(m.control.stored(), { min: 'minimal', recommended: 'high', max: 'ultra' });
    assert.equal(m.handles.min.getAttribute('aria-valuetext'), 'Low');
    assert.equal(m.el.querySelector('.chat-capsule-value').textContent, 'High', 'the closed capsule follows the saved level');
    m.control.syncState({ context_mode: 'max' });
    assert.deepEqual(m.control.shown(), { min: 1, rec: 3, max: 6 }, 'a snapshot without the range changes nothing');
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

test('with a hovering pointer: opens after ~160 ms; closes 1 s after the mouse leaves a 32 px zone, unless pinned', async () => {
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
    m.leave('mouse', { x: 120, y: 10 });   // 20 px right of the control: inside the zone
    await sleep(1150);
    assert.equal(m.control.isOpen(), true, 'a mouse that drifts into the zone keeps it open');
    m.docMove(200, 10);                    // 100 px away: the zone is left, the delay starts
    await sleep(500);
    assert.equal(m.control.isOpen(), true, 'not before the delay');
    await sleep(650);
    assert.equal(m.control.isOpen(), false, 'closed a second after leaving the zone');
    assert.equal(m.doc.count('pointermove'), 0, 'zone tracking stops once closed');
    m.enter();
    await sleep(200);
    m.leave();
    await sleep(300);
    m.docMove(110, 30);                    // back into the zone before the delay ran out
    await sleep(900);
    assert.equal(m.control.isOpen(), true, 'coming back cancels the close');
    m.enter();
    m.headClick(1);
    assert.equal(m.control.isPinned(), true, 'a press pins the hover-opened strip');
    m.leave();
    await sleep(1150);
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

test('a drag that ends inside the hover zone does not close the strip a second later', async () => {
    const m = mount({ hover: true });
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.enter();
    await sleep(200);
    m.down(100);
    m.up(110);                             // released 10 px past the control's right edge
    await m.settle();
    m.leave('mouse', { x: 110, y: 10 });
    await sleep(1150);
    assert.equal(m.control.isOpen(), true, 'the pointer is still within the zone');
    m.docMove(300, 10);
    await sleep(1150);
    assert.equal(m.control.isOpen(), false, 'leaving the zone closes it');
});

test('keyboard moves primary beyond either bound while each helper handle leaves primary alone', async () => {
    const m = mount();
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.headClick(0);
    m.key('rec', 'End');
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'low', recommended: 'ultra', max: 'high' });
    m.key('rec', 'Home');
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'low', recommended: 'none', max: 'high' });
    m.key('min', 'End');
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'high', recommended: 'none', max: 'high' });
    m.key('max', 'Home');
    await m.settle();
    assert.deepEqual(m.control.shown(), { min: 3, rec: 0, max: 3 });
    m.key('max', 'PageUp');
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'high', recommended: 'none', max: 'xhigh' });
    m.control.destroy();
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
    await m.settle();
    assert.deepEqual(m.control.shown(), { min: 1, rec: 1, max: 3 }, 'rec meets the shown minimum');
    assert.deepEqual(m.saves.at(-1), { min: 'minimal', recommended: 'low', max: 'high' }, 'the untouched minimum keeps minimal');
    m.key('rec', 'ArrowLeft');
    await m.settle();
    assert.deepEqual(m.control.shown(), { min: 1, rec: 0, max: 3 }, 'primary may pass the shown minimum');
    assert.deepEqual(m.saves.at(-1), { min: 'minimal', recommended: 'none', max: 'high' }, 'the untouched bound keeps its stored tier');
});

test('pointer taps anywhere choose primary, a drag never moves helper bounds, and cancel restores', async () => {
    const m = mount();
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.headClick(1);
    m.down(60);
    m.move(250);
    assert.equal(m.handles.rec.style.left, '240px');
    assert.equal(m.handles.max.getAttribute('aria-valuenow'), '3');
    m.up(250);
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'low', recommended: 'ultra', max: 'high' });
    m.down(10); m.up(10);
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'low', recommended: 'none', max: 'high' });
    m.down(165, { handle: m.handles.max }); m.move(285); m.up(285);
    await m.settle();
    assert.deepEqual(m.saves.at(-1), { min: 'low', recommended: 'none', max: 'ultra' });
    const before = m.saves.length;
    m.down(10); m.move(270); m.cancel();
    await m.settle();
    assert.equal(m.saves.length, before);
    assert.deepEqual(m.control.shown(), { min: 1, rec: 0, max: 6 });
    m.down(10); m.move(200, 2); m.up(200, 2);
    assert.equal(m.handles.rec.dataset.dragging, 'true');
    m.up(10); await m.settle();
    assert.equal(m.saves.length, before);
    m.control.destroy();
});

test('a strip that scrolls keeps the handle that moved in view, clear of the fade, and says where it continues', () => {
    const m = mount();                // seven 40 px segments: the strip's content is 280 px
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.headClick(0);
    m.el.dataset.fit = 'overflow';    // what fit() sets when even the tightest padding does not fit
    Object.assign(m.strip, { clientWidth: 200, scrollWidth: 280, scrollLeft: 0 });
    m.handles.min.getBoundingClientRect = () => ({ width: 8 });
    m.key('max', 'End');              // the max bracket goes to Ultra: 272..280, past the 200 px view
    assert.equal(m.strip.scrollLeft, 280 + 12 - 200, 'the whole bracket plus the fade margin is in view');
    assert.equal(m.strip.dataset.scroll, 'end', 'scrolled to the end: only the start fades');
    m.key('max', 'Home');             // max comes down to min (Low): 72..80
    assert.equal(m.strip.scrollLeft, 72 - 12);
    assert.equal(m.strip.dataset.scroll, 'middle', 'levels hide on both sides: both fade');
    m.key('min', 'Home');             // min goes to None: 0..8
    assert.equal(m.strip.scrollLeft, 0);
    assert.equal(m.strip.dataset.scroll, 'start');
});

test('hover opens only where the button stays under the mouse; alone or scrolling, the press opens it', async () => {
    const doc = stubDocument();
    const row = doc.createElement('div');
    row.className = 'chat-toolbar-row';
    const subagents = doc.createElement('div');  // the other capsules, closed
    subagents.className = 'chat-capsule chat-subagents-mode';
    subagents.getBoundingClientRect = () => ({ width: 220 });
    row.appendChild(subagents);
    const styles = new Map([[row, { paddingLeft: '8px', paddingRight: '8px', columnGap: '8px' }]]);
    const win = { ...hoverWindow(true), getComputedStyle: (node) => styles.get(node) || {} };
    const control = createEffortRangeControl({ row, doc, win, saveEffortRange: async body => ({ effort_range: body }), showToast: () => {} });
    const el = control.el;
    el.querySelector('.chat-effort-head').getBoundingClientRect = () => ({ width: 30 });
    el.querySelectorAll('.chat-effort-seg').forEach((seg) => {
        Object.defineProperty(seg, 'firstElementChild', { value: { getBoundingClientRect: () => ({ width: 32 }) } });
    });
    const enter = () => el.listeners.get('pointerenter').forEach((fn) => fn({ pointerType: 'mouse' }));
    const leave = () => el.listeners.get('pointerleave').forEach((fn) => fn({ pointerType: 'mouse', clientX: 900, clientY: 900 }));
    row.getBoundingClientRect = () => ({ width: 440 });  // a Project pane: the strip would stand alone
    enter();
    await sleep(220);
    assert.equal(control.isOpen(), false, 'no hover-open where the others would step aside under the mouse');
    leave();
    row.getBoundingClientRect = () => ({ width: 900 });  // room beside the others: the button stays put
    enter();
    await sleep(220);
    assert.equal(control.isOpen(), true, 'hover opens beside the others');
    control.destroy();
});

test('a mouse press on a level does not keep a hover-opened strip from closing; keyboard focus does', async () => {
    const m = mount({ hover: true });
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.enter();
    await sleep(200);
    m.doc.fire('pointerdown', { target: m.head });  // the document sees every press first
    m.down(100);
    m.up(100);
    await m.settle();
    assert.equal(m.doc.activeElement, m.handles.rec, 'the press focused the handle');
    m.leave('mouse', { x: 600, y: 10 });
    await sleep(1150);
    assert.equal(m.control.isOpen(), false, 'a mouse press does not hold the strip open');
    m.enter();
    await sleep(200);
    m.doc.fire('keydown', { key: 'ArrowRight' });   // the last input was the keyboard
    m.key('rec', 'ArrowRight');
    await m.settle();
    m.handles.rec.focus();
    m.leave('mouse', { x: 600, y: 10 });
    await sleep(1150);
    assert.equal(m.control.isOpen(), true, 'keyboard focus inside keeps it open');
    m.escape();
});

test('fit: beside the other controls when the strip fits there, alone when it does not, its name giving way, scrolling last', () => {
    const doc = stubDocument();
    const row = doc.createElement('div');
    row.className = 'chat-toolbar-row';
    const subagents = doc.createElement('div');
    subagents.className = 'chat-capsule chat-subagents-mode';
    const context = doc.createElement('div');
    context.className = 'chat-context-mode';
    const sized = (node, width) => { node.getBoundingClientRect = () => ({ width }); };
    sized(subagents, 120);            // the other capsules, closed
    sized(context, 100);
    row.appendChild(subagents);
    row.appendChild(context);
    const styles = new Map([
        [row, { paddingLeft: '8px', paddingRight: '8px', columnGap: '8px' }],
    ]);
    const resize = [];
    const win = {
        ...hoverWindow(false), getComputedStyle: (node) => styles.get(node) || {},
        addEventListener: (type, fn) => { if (type === 'resize') resize.push(fn); }, removeEventListener: () => {},
    };
    const control = createEffortRangeControl({
        row, doc, win, saveEffortRange: async body => ({ effort_range: body }), showToast: () => {},
    });
    const el = control.el;
    assert.equal(row.children.indexOf(el), 2, 'after the Subagents and Context capsules');
    const q = (selector) => el.querySelector(selector);
    styles.set(el, { borderLeftWidth: '1px', borderRightWidth: '1px' });
    styles.set(q('[data-effort-strip]'), { marginRight: '9.33px' });  // the end gutter
    styles.set(q('.chat-capsule-name'), { marginLeft: '-2px', marginRight: '8px' });
    styles.set(q('.chat-capsule-body'), { width: '1px', marginRight: '4px' });  // its ::before, the divider
    sized(q('.chat-capsule-glyph'), 30);
    sized(q('.chat-capsule-name'), 36);
    // Fractional words: whole-pixel offsets would lose almost a pixel here.
    [31.41, 24.19, 48.08, 28.19, 42.11, 24.94, 29.8].forEach((width, i) => {
        Object.defineProperty(el.querySelectorAll('.chat-effort-seg')[i], 'firstElementChild', { value: { getBoundingClientRect: () => ({ width }) } });
    });
    const head = q('.chat-effort-head');
    const open = (width) => { sized(row, width); head.listeners.get('click').forEach((fn) => fn({ detail: 1 })); };
    const close = () => head.listeners.get('click').forEach((fn) => fn({ detail: 1 }));
    const facts = () => [el.dataset.fit, el.style.getPropertyValue('--capsule-seg-pad'), el.dataset.tight, el.dataset.label, row.dataset.ownerSolo];
    // Words 228.72 and 14 paddings; the gutter and the borders 11.33; the head 30 bare, 77 with the
    // name and the divider; beside it Subagents 120 and Context 100, a row gap each (236).
    open(700);
    assert.deepEqual(facts(), ['inline', '8px', 'false', 'true', undefined], 'the owner row: beside the others at full padding');
    close();
    assert.deepEqual([el.dataset.fit, row.dataset.ownerSolo], [undefined, undefined], 'closing restores the others');
    assert.deepEqual([el.style.getPropertyValue('--capsule-seg-pad'), el.dataset.tight], ['8px', 'false'],
        'the fitted padding stays while the strip collapses (the next open fits again)');
    open(640);
    assert.deepEqual(facts(), ['inline', '4px', 'true', 'true', undefined], 'beside them, the padding tightens to var(--space-1)');
    close();
    open(560);
    assert.deepEqual(facts(), ['solo', '8px', 'false', 'true', 'effort'], 'no room beside them: they step aside');
    sized(subagents, 0);              // hidden controls measure 0: the width seen before hiding stays
    sized(context, 0);
    resize.forEach((fn) => fn());
    assert.deepEqual(facts(), ['solo', '8px', 'false', 'true', 'effort'], 'a resize while alone keeps them aside');
    close();
    sized(subagents, 120);
    sized(context, 100);
    open(390);
    assert.deepEqual(facts(), ['solo', '3px', 'true', 'true', 'effort'], 'alone, the name stays down to 3 px');
    close();
    open(350);
    assert.deepEqual(facts(), ['solo', '4px', 'true', 'false', 'effort'], 'a phone: the name and the divider give way first');
    // Open, the line narrows until the strip scrolls: it shows what opening shows (the recommended
    // level, Medium at 80..120 in a 100 px view); once it scrolls, a relayout keeps the owner's scroll.
    const strip = q('[data-effort-strip]');
    el.querySelectorAll('.chat-effort-seg').forEach((seg, i) => { seg.offsetLeft = i * 40; seg.offsetWidth = 40; });
    Object.assign(strip, { clientWidth: 100, scrollLeft: 0 });
    sized(row, 320);
    resize.forEach((fn) => fn());
    assert.deepEqual([el.dataset.fit, strip.scrollLeft, strip.dataset.scroll], ['overflow', 120 + 12 - 100, 'middle']);
    strip.scrollLeft = 0;             // the owner swipes back to None
    resize.forEach((fn) => fn());
    assert.deepEqual([el.dataset.fit, strip.scrollLeft, strip.dataset.scroll], ['overflow', 0, 'start'], 'the owner\'s scroll stays');
    close();
    open(350);
    control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'ultra' } });
    q('.chat-effort-max').focus();
    strip.scrollLeft = 0;
    sized(row, 320);
    resize.forEach((fn) => fn());
    assert.equal(strip.scrollLeft, 280 + 12 - 100, 'new overflow reveals the focused bracket, not the recommended pill');
    close();
    doc.activeElement = null;
    open(320);
    assert.deepEqual(facts(), ['overflow', '3px', 'true', 'false', 'effort'], 'even that does not fit: the strip scrolls');
    close();
    assert.equal(row.dataset.ownerSolo, undefined);
    control.destroy();
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

test('the context capsule: ring, Context and the mode; it opens into exactly three modes, one tab stop among them', async () => {
    const m = mountPair();
    assert.ok(m.contextEl.classList.contains('chat-capsule'));
    assert.equal(m.contextEl.getAttribute('aria-label'), 'Context size mode');
    assert.match(m.contextEl.innerHTML, /class="chat-capsule-name">Context</);
    assert.equal(m.contextEl.querySelector('.chat-capsule-value').textContent, 'Max');
    assert.ok(!m.contextHead.title, 'no hover-only title: the hint says it on every surface');
    assert.deepEqual(['word', 'text', 'rule'].map((part) => m.contextHint.querySelector(`.ui-hint-${part}`).textContent),
        ['Max', 'The full context window.', CHANGE_RULE]);
    const segs = m.contextEl.querySelectorAll('.chat-seg');
    // Three modes, a position among them, never a fill level: three arcs and three radios.
    assert.deepEqual(segs.map((node) => node.dataset.mode), ['nano', 'low', 'max']);
    assert.equal(m.contextEl.querySelectorAll('.chat-context-arc').length, 3);
    assert.ok(m.contextEl.querySelectorAll('[data-mode]').every((node) => node.classList.contains('chat-seg')),
        'inside the control [data-mode] still means a mode button: the arcs carry data-arc');
    assert.deepEqual(segs.map((node) => node.getAttribute('aria-checked')), ['false', 'false', 'true']);
    assert.deepEqual(segs.map((node) => node.tabIndex), [-1, -1, -1], 'closed: out of the tab order');
    await m.press(m.contextHead, 0);  // Enter / Space
    assert.equal(m.contextEl.dataset.open, 'true');
    assert.equal(m.contextHead.getAttribute('aria-expanded'), 'true');
    assert.deepEqual(segs.map((node) => node.tabIndex), [-1, -1, 0], 'one tab stop: the chosen mode');
    assert.equal(m.doc.activeElement, m.seg('max'), 'the keyboard lands on the chosen mode');
    const escape = { key: 'Escape', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } };
    m.doc.fire('keydown', escape);
    assert.equal(m.contextEl.dataset.open, 'false');
    assert.equal(escape.defaultPrevented, true, 'the capsule took the Escape: a Project room under it stays open');
    assert.equal(m.doc.activeElement, m.contextHead, 'Esc returns focus to the head');
    const later = { key: 'Escape', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } };
    m.doc.fire('keydown', later);
    assert.equal(later.defaultPrevented, false, 'a closed capsule leaves Escape to the room');
    m.controls.syncState({ context_mode: 'nano' });
    assert.equal(m.contextEl.dataset.contextMode, 'nano', 'the Main composer reads Main mode');
    assert.equal(m.contextEl.querySelector('.chat-capsule-value').textContent, 'Nano');
    m.controls.syncState({ context_mode: 'huge' });
    assert.equal(m.contextEl.dataset.contextMode, 'max', 'an unknown mode reads as Max');
    m.controls.destroy();
});

test('a context choice posts the owner endpoint at once, waits for the answer, shows a refusal and always re-reads /api/state', async () => {
    const m = mountPair();
    await m.press(m.contextHead);
    await m.choose('low');
    assert.deepEqual(m.posts, [['/api/owner/context-mode', { mode: 'low' }]]);
    assert.equal(m.contextEl.dataset.contextMode, 'low');
    assert.equal(m.contextEl.querySelector('.chat-capsule-value').textContent, 'Low');
    assert.equal(m.seg('low').getAttribute('aria-checked'), 'true');
    assert.equal(m.contextEl.dataset.disabled, 'false');
    assert.deepEqual(m.refreshes, [true]);
    await m.choose('low');
    assert.equal(m.posts.length, 1, 'the current mode is not re-posted');
    m.answer.value = { ok: false, json: async () => ({ error: 'Context mode can only be lowered while Ouroboros is idle.' }) };
    await m.choose('max');
    assert.equal(m.contextEl.dataset.contextMode, 'low', 'a refusal leaves the saved mode');
    assert.equal(m.contextEl.querySelector('.chat-capsule-value').textContent, 'Low');
    assert.deepEqual(m.toasts, [['Context mode can only be lowered while Ouroboros is idle.', 'error']]);
    assert.deepEqual(m.refreshes, [true, true]);
    // While a save is on its way the strip waits: a second choice and a stale poll change nothing.
    let release;
    m.answer.value = new Promise((resolve) => { release = resolve; });
    const pending = m.choose('nano');
    assert.equal(m.contextEl.dataset.disabled, 'true');
    await m.choose('max');
    m.controls.syncState({ context_mode: 'max' });
    assert.equal(m.contextEl.dataset.contextMode, 'low', 'the save answers for itself, not a stale poll');
    release({ ok: true });
    await pending;
    assert.deepEqual(m.posts.map(([, body]) => body.mode), ['low', 'max', 'nano']);
    assert.equal(m.contextEl.dataset.contextMode, 'nano');
    // Arrows move to a neighbour and choose it; Home and End go to the ends.
    m.answer.value = { ok: true };
    await m.key('nano', 'ArrowRight');
    assert.equal(m.doc.activeElement, m.seg('low'));
    assert.equal(m.contextEl.dataset.contextMode, 'low');
    await m.key('low', 'End');
    assert.equal(m.contextEl.dataset.contextMode, 'max');
    await m.key('max', 'ArrowDown');
    assert.equal(m.posts.length, 5, 'past the end nothing moves and nothing is posted');
    assert.equal(m.controls.hasPendingSave(), false, 'both saves are settled');
    assert.equal(await m.controls.pendingSave(), true);
    m.controls.destroy();
});

test('a context strip that scrolls shows the chosen mode whole on opening, and the mode an arrow, Home or End reaches', async () => {
    const m = mountPair();
    const strip = m.contextEl.querySelector('.chat-context-strip');
    // Long translated words on a phone: three 90 px modes in a 160 px view.
    m.contextEl.querySelectorAll('.chat-seg').forEach((seg, i) => { seg.offsetLeft = i * 90; seg.offsetWidth = 90; });
    Object.assign(strip, { clientWidth: 160, scrollLeft: 0 });
    m.contextEl.dataset.fit = 'overflow';  // what fit() sets when even the tightest padding does not fit
    await m.press(m.contextHead, 0);
    assert.equal(strip.scrollLeft, 270 + 12 - 160, 'opening shows Max, the last mode, whole and clear of the fade');
    assert.equal(strip.dataset.scroll, 'end');
    await m.key('max', 'Home');
    assert.deepEqual([m.contextEl.dataset.contextMode, strip.scrollLeft, strip.dataset.scroll], ['nano', 0, 'start']);
    await m.key('nano', 'ArrowRight');
    assert.deepEqual([m.contextEl.dataset.contextMode, strip.scrollLeft, strip.dataset.scroll], ['low', 180 + 12 - 160, 'middle']);
    await m.key('low', 'End');
    assert.deepEqual([m.contextEl.dataset.contextMode, strip.scrollLeft, strip.dataset.scroll], ['max', 270 + 12 - 160, 'end']);
    m.controls.destroy();
});

test('one capsule is open at a time: opening one closes the other at once, a narrow solo strip included', async () => {
    const m = mountPair();
    await m.press(m.contextHead);
    assert.equal(m.contextEl.dataset.open, 'true');
    // The press on the other capsule is its own to answer: the document handler leaves it alone,
    // and opening Effort closes Context at once.
    m.docDown(m.effortHead);
    assert.equal(m.contextEl.dataset.open, 'true');
    await m.press(m.effortHead);
    assert.deepEqual([m.contextEl.dataset.open, m.effortEl.dataset.open], ['false', 'true']);
    await m.press(m.contextHead);
    assert.deepEqual([m.contextEl.dataset.open, m.effortEl.dataset.open], ['true', 'false']);
    // A narrow line: Effort stands alone (Subagents and Context step aside); once it closes and
    // Context opens, Context is the one alone, never hidden by Effort's flag.
    m.row.dataset.ownerSolo = 'effort';
    m.effortEl.dataset.fit = 'solo';
    await m.press(m.effortHead);
    assert.equal(m.effortEl.dataset.open, 'true');
    m.row.dataset.ownerSolo = 'effort';
    m.effortEl.dataset.fit = 'solo';
    m.docDown(new ElementStub('div', m.doc));
    assert.equal(m.effortEl.dataset.open, 'false');
    assert.equal(m.row.dataset.ownerSolo, undefined, 'closing gives the line back to every control');
    await m.press(m.contextHead);
    assert.equal(m.contextEl.dataset.open, 'true');
    assert.notEqual(m.row.dataset.ownerSolo, 'effort', 'Effort\'s flag never hides the capsule that opened');
    m.controls.destroy();
});

test('hover never opens a capsule while the other one is open', async () => {
    const m = mountPair({ hover: true });
    await m.press(m.effortHead);  // pinned open
    for (const fn of m.contextEl.listeners.get('pointerenter')) fn({ pointerType: 'mouse' });
    await sleep(220);
    assert.deepEqual([m.contextEl.dataset.open, m.effortEl.dataset.open], ['false', 'true']);
    await m.press(m.effortHead);  // unpins and closes
    for (const fn of m.contextEl.listeners.get('pointerenter')) fn({ pointerType: 'mouse' });
    await sleep(220);
    assert.equal(m.contextEl.dataset.open, 'true', 'with the other closed, hover opens it');
    m.controls.destroy();
});

test('closed, the capsules keep their names while the line holds them; a narrower line keeps rings and values', () => {
    const m = mountPair();
    const line = (top) => { m.effortEl.getBoundingClientRect = () => ({ top, bottom: top + 32, width: 130 }); };
    line(40);                         // the names push Effort onto a second line
    m.controls.syncState({});
    assert.equal(m.row.dataset.ownerCompact, 'true');
    line(0);
    m.controls.syncState({});
    assert.equal(m.row.dataset.ownerCompact, undefined, 'room again: the names come back');
    line(40);
    m.contextHead.listeners.get('click').forEach((fn) => fn({ detail: 1 }));
    m.controls.syncState({});
    assert.equal(m.row.dataset.ownerCompact, undefined, 'never decided while a strip is open');
    m.controls.destroy();
});

test('a phone\'s ☰, gone down beside the field while the row folds, is not the line the capsules are read by', () => {
    const m = mountPair();
    const menu = new ElementStub('button', m.doc);
    menu.className = 'mobile-nav-toggle composer-menu';
    menu.getBoundingClientRect = () => ({ top: 21, bottom: 53, width: 32 });  // composer_fold.js: at the field's middle
    m.row.prepend(menu);
    m.effortEl.getBoundingClientRect = () => ({ top: 40, bottom: 72, width: 130 });  // the names push Effort onto a second line
    m.controls.syncState({});
    assert.equal(m.row.dataset.ownerCompact, 'true', 'the capsules\' own line still wraps: the row keeps its fit');
    m.effortEl.getBoundingClientRect = () => ({ top: 0, bottom: 32, width: 130 });
    m.controls.syncState({});
    assert.equal(m.row.dataset.ownerCompact, undefined, 'room again: the names come back');
    m.controls.destroy();
});

test('row/strip resize delivery defers feedback, coalesces work and cancels it on destroy', () => {
    const observers = [], frames = new Map();
    let nextFrame = 0;
    const win = {
        ...hoverWindow(false),
        ResizeObserver: class {
            constructor(callback) { this.callback = callback; observers.push(this); }
            observe() {}
            disconnect() { this.disconnected = true; }
        },
        requestAnimationFrame(callback) { frames.set(++nextFrame, callback); return nextFrame; },
        cancelAnimationFrame(id) { frames.delete(id); },
    };
    const deliver = () => observers.forEach((observer) => observer.callback());
    const paint = () => { const pending = [...frames.values()]; frames.clear(); pending.forEach((callback) => callback()); };
    const m = mountPair({ win });
    const thumb = m.contextEl.querySelector('.chat-context-thumb');
    const chosen = m.seg('max');
    Object.assign(chosen, { offsetLeft: 80, offsetWidth: 40 });
    m.controls.syncState({ context_mode: 'max' });
    chosen.offsetLeft = 120;
    m.effortEl.getBoundingClientRect = () => ({ top: 40, bottom: 72, width: 130 });
    deliver();
    deliver();
    assert.equal(thumb.style.left, '80px', 'resize delivery cannot write the observed layout');
    assert.equal(m.row.dataset.ownerCompact, undefined, 'the row cannot wrap and unwrap during delivery');
    assert.equal(frames.size, observers.length, 'duplicate deliveries share each pending frame');
    paint();
    assert.equal(thumb.style.left, '120px', 'the next frame aligns the mark with the resized word');
    assert.equal(m.row.dataset.ownerCompact, 'true', 'the next frame fits the closed names');
    m.effortEl.getBoundingClientRect = () => ({ top: 0, bottom: 32, width: 130 });
    deliver();
    paint();
    assert.equal(m.row.dataset.ownerCompact, undefined, 'names return when room returns');
    deliver();
    assert.ok(frames.size > 0);
    m.controls.destroy();
    assert.equal(frames.size, 0, 'destroy cancels every pending layout');
    assert.ok(observers.every((observer) => observer.disconnected));
});

test('docking-box geometry and changed text settle descendants before paint, with full disposal', () => {
    const sizes = [], texts = [], frames = new Map();
    let nextFrame = 0;
    const win = {
        ...hoverWindow(false),
        ResizeObserver: class {
            constructor(callback) { this.callback = callback; this.nodes = []; sizes.push(this); }
            observe(node) { this.nodes.push(node); }
            disconnect() { this.disconnected = true; }
        },
        MutationObserver: class {
            constructor(callback) { this.callback = callback; texts.push(this); }
            observe(node, options) { this.node = node; this.options = options; }
            disconnect() { this.disconnected = true; }
        },
        requestAnimationFrame(callback) { frames.set(++nextFrame, callback); return nextFrame; },
        cancelAnimationFrame(id) { frames.delete(id); },
    };
    const m = mountPair({ win, withBox: true });
    const boxObserver = sizes.find((observer) => observer.nodes.includes(m.box));
    assert.ok(boxObserver, 'chat supplies the independently sized docking box');
    assert.deepEqual(boxObserver.nodes, [m.box], 'not the descendant whose fit we change');
    const thumb = m.contextEl.querySelector('.chat-context-thumb');
    const chosen = m.seg('max');
    Object.assign(chosen, { offsetLeft: 120, offsetWidth: 40 });
    m.effortEl.getBoundingClientRect = () => ({ top: 40, bottom: 72, width: 130 });
    boxObserver.callback();
    assert.equal(thumb.style.left, '120px');
    assert.equal(m.row.dataset.ownerCompact, 'true', 'fits on the initial box delivery, without syncState');
    assert.equal(frames.size, 0, 'no one-frame deferral for independent geometry');
    assert.equal(texts.length, 1);
    assert.equal(texts[0].node, m.row);
    assert.deepEqual(texts[0].options, { childList: true, characterData: true, subtree: true });
    chosen.offsetLeft = 160;
    m.effortEl.getBoundingClientRect = () => ({ top: 0, bottom: 32, width: 130 });
    texts[0].callback();
    assert.equal(thumb.style.left, '160px', 'new words realign the mark in the text microtask');
    assert.equal(m.row.dataset.ownerCompact, undefined);
    // The value is what a gesture's renderer changes. Its MO must fit a closed line without
    // settling the strip and cancelling the thumb's own transition (real motion: browser test).
    chosen.offsetLeft = 180;
    const value = m.contextEl.querySelector('.chat-capsule-value');
    value.nodeType = 1;
    value.closest = (selector) => value.matches(selector) ? value : null; // The shared stub only knows .page.active.
    m.effortEl.getBoundingClientRect = () => ({ top: 40, bottom: 72, width: 130 });
    texts[0].callback([{ target: value }]);
    assert.equal(thumb.style.left, '160px');
    assert.equal(m.row.dataset.ownerCompact, 'true');
    texts[0].callback([{ target: { nodeType: 3, parentNode: value } }]);
    assert.equal(thumb.style.left, '160px', 'i18n changes the existing value text node');
    texts[0].callback([{ target: value }, { target: chosen }]);
    assert.equal(thumb.style.left, '180px', 'a mixed batch with a strip word still realigns');
    sizes.filter((observer) => observer !== boxObserver).forEach((observer) => observer.callback());
    assert.ok(frames.size > 0, 'feedback from the changed row/strips is still deferred');
    m.controls.destroy();
    assert.equal(frames.size, 0);
    assert.ok([...sizes, ...texts].every((observer) => observer.disconnected));
    chosen.offsetLeft = 200;
    boxObserver.callback();
    texts[0].callback();
    assert.equal(thumb.style.left, '180px', 'already queued deliveries after destroy cannot write');
});

/* The refit of a wrapped row shrinks the composer that chat.js observes higher in the tree. Inside ResizeObserver
   delivery that is a resize of a box the pass cannot deliver (WebKit: "loop completed with undelivered notifications"),
   so the observer only schedules and the fit runs in a frame; a window resize, which precedes layout, fits at once. */
function observedWindow() {
    const observers = [];
    const frames = [];
    const cancelled = [];
    const listeners = new Map();
    class FakeObserver {
        constructor(callback) { this.callback = callback; observers.push(this); }
        observe() {}
        unobserve() {}
        disconnect() { this.disconnected = true; }
    }
    const win = {
        ...hoverWindow(false),
        ResizeObserver: FakeObserver,
        requestAnimationFrame: (fn) => { frames.push(fn); return frames.length; },
        cancelAnimationFrame: (id) => cancelled.push(id),
        addEventListener: (type, fn) => listeners.set(type, [...(listeners.get(type) || []), fn]),
        removeEventListener: (type, fn) => listeners.set(type, (listeners.get(type) || []).filter((f) => f !== fn)),
    };
    return { win, observers, frames, cancelled, fire: (type) => (listeners.get(type) || []).forEach((fn) => fn({})), count: (type) => (listeners.get(type) || []).length };
}

test('the row refit never changes layout inside ResizeObserver delivery: the observer schedules a frame, the frame fits', () => {
    const w = observedWindow();
    const m = mountPair({ win: w.win });
    const fit = w.observers.at(-1);  // the row's own observer is the last one the controls create
    const line = (top) => { m.effortEl.getBoundingClientRect = () => ({ top, bottom: top + 32, width: 130 }); };
    line(40);  // the names push Effort onto a second line
    fit.callback([]);
    assert.equal(m.row.dataset.ownerCompact, undefined, 'nothing is written to the row while the observer delivers');
    assert.equal(w.frames.length, 1, 'one frame is asked for');
    fit.callback([]);
    fit.callback([]);
    assert.equal(w.frames.length, 1, 'a burst of deliveries asks for the one frame');
    w.frames[0]();
    assert.equal(m.row.dataset.ownerCompact, 'true', 'the frame fits the row');
    line(0);
    fit.callback([]);
    assert.equal(w.frames.length, 2, 'the next delivery can ask again once the frame ran');
    assert.equal(m.row.dataset.ownerCompact, 'true', 'room again, but not inside the delivery');
    w.frames[1]();
    assert.equal(m.row.dataset.ownerCompact, undefined, 'the names come back in the frame');
    m.controls.destroy();
});

test('a window resize fits the row immediately; destroy cancels a pending frame and the listener', () => {
    const w = observedWindow();
    const m = mountPair({ win: w.win });
    const line = (top) => { m.effortEl.getBoundingClientRect = () => ({ top, bottom: top + 32, width: 130 }); };
    line(40);
    w.fire('resize');
    assert.equal(m.row.dataset.ownerCompact, 'true', 'the resize listener fits synchronously');
    assert.equal(w.frames.length, 0, 'and no frame was needed');
    line(0);
    w.observers.at(-1).callback([]);
    assert.equal(w.frames.length, 1);
    m.controls.destroy();
    assert.deepEqual(w.cancelled, [1], 'the pending frame is cancelled');
    assert.equal(w.count('resize'), 0, 'no resize listener outlives the controls');
    w.frames[0]();
    assert.equal(m.row.dataset.ownerCompact, undefined, 'a late frame writes nothing');
});

test('Main and two projects read and save only their room, including delayed context saves', async () => {
    const main = mountPair(), alpha = mountPair({ projectId: 'alpha' }), beta = mountPair({ projectId: 'beta' });
    const data = { context_mode: 'max', effort_range: { min: 'low', recommended: 'ultra', max: 'high' }, projects: [
        { id: 'alpha', cognition: { context_mode: 'low', effort_range: { min: 'high', recommended: 'none', max: 'max' } } },
        { id: 'beta', cognition: { context_mode: 'nano', effort_range: { min: 'none', recommended: 'max', max: 'low' } } },
    ] };
    for (const m of [main, alpha, beta]) m.controls.syncState(data);
    assert.deepEqual([main, alpha, beta].map(m => m.contextEl.dataset.contextMode), ['max', 'low', 'nano']);
    assert.deepEqual([main, alpha, beta].map(m => m.controls.effort.stored().recommended), ['ultra', 'none', 'max']);
    alpha.controls.syncState({ context_mode: 'max', effort_range: data.effort_range });
    assert.equal(alpha.contextEl.dataset.contextMode, 'low', 'missing project never falls back to Main');
    let release;
    alpha.answer.value = new Promise(resolve => { release = resolve; });
    const saving = alpha.choose('nano');
    assert.equal(alpha.controls.hasPendingSave(), true);
    release({ ok: true }); await saving;
    assert.equal(await alpha.controls.pendingSave(), true);
    assert.deepEqual(alpha.posts[0], ['/api/owner/context-mode', { mode: 'nano', project_id: 'alpha' }]);
    assert.equal(main.contextEl.dataset.contextMode, 'max');
    assert.equal(beta.contextEl.dataset.contextMode, 'nano');
    alpha.answer.value = { ok: true, json: async () => ({ project_id: 'beta', context_mode: 'max' }) };
    const refused = alpha.choose('max');
    const send = alpha.controls.pendingSave();
    await refused;
    assert.equal(alpha.contextEl.dataset.contextMode, 'nano', 'foreign-scope response cannot repaint this room');
    assert.equal(await send, false, 'the save this Send awaited was refused');
    assert.equal(await alpha.controls.pendingSave(), true, 'a reported refusal does not block a later Send');
    for (const m of [main, alpha, beta]) m.controls.destroy();
});

for (const failed of ['context', 'effort']) for (const accepts of [true, false]) {
    test(`Send waits only for pending controls: old ${failed} refusal, new save ${accepts ? 'accepted' : 'refused'}`, async () => {
        const m = mountPair({ projectId: 'alpha' });
        const effort = () => {
            const handle = m.controls.effort.el.querySelector('.chat-effort-rec');
            for (const fn of handle.listeners.get('keydown') || []) fn({ key: 'ArrowRight', currentTarget: handle, preventDefault() {} });
        };
        try {
            if (failed === 'context') {
                m.answer.value = { ok: false, json: async () => ({ error: 'Earlier refusal' }) };
                await m.choose('low');
            } else {
                m.effortAnswer.value = Promise.reject(new Error('Earlier refusal'));
                effort(); await m.controls.effort.pendingSave();
            }
            assert.equal(m.toasts.length, 1);
            assert.equal(m.controls.hasPendingSave(), false);
            let resolve, reject;
            const pending = new Promise((yes, no) => { resolve = yes; reject = no; });
            if (failed === 'context') { m.effortAnswer.value = pending; effort(); }
            else { m.answer.value = pending; void m.choose('low'); }
            assert.equal(m.controls.hasPendingSave(), true);
            const send = m.controls.pendingSave();
            if (!accepts) reject(new Error('Current refusal'));
            else resolve(failed === 'context'
                ? { project_id: 'alpha', effort_range: { min: 'low', recommended: 'high', max: 'high' } }
                : { ok: true });
            assert.equal(await send, accepts, 'only this pending save decides whether Send proceeds');
            assert.equal(m.toasts.length, accepts ? 1 : 2, 'a refused current save remains visible');
            assert.equal(m.controls.hasPendingSave(), false);
        } finally { m.controls.destroy(); }
    });
}
