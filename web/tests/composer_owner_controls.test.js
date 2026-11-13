// The composer's owner controls (docs/DESIGN.md "Composer owner controls"): two labelled
// capsules. The effort range's handle math, the open/pin state machine with and without a
// hovering pointer, keyboard, serialized saves with a refused save returning to the server's
// value, independent room readback, teardown; the context capsule's three modes, its
// save and a scrolling strip that keeps the chosen mode in view; one capsule open at a time, and
// the names that give way on a narrow line.
import test from 'node:test';
import assert from 'node:assert/strict';

import { ElementStub } from './chat_dom_fixture.js';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';
import { CHANGE_RULE } from '../modules/capsule.js';
import { applyHandle, createEffortRangeControl, snapRange } from '../modules/composer_owner_controls.js';
import { hoverWindow, mount, mountPair, sleep, stubDocument } from './composer_controls_fixture.js';
import { observeContext } from '../modules/context_observation.js';

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

test('with a hovering pointer: opens after ~160 ms; closes ~300 ms after the mouse leaves its small zone, unless pinned', async () => {
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
    m.leave('mouse', { x: 104, y: 10 });   // 4 px right of the control: inside the zone
    await sleep(450);
    assert.equal(m.control.isOpen(), true, 'a mouse that drifts into the zone keeps it open');
    m.docMove(200, 10);                    // 100 px away: the zone is left, the delay starts
    await sleep(150);
    assert.equal(m.control.isOpen(), true, 'not before the delay');
    await sleep(300);
    assert.equal(m.control.isOpen(), false, 'closed about 300 ms after leaving the zone');
    assert.equal(m.doc.count('pointermove'), 0, 'zone tracking stops once closed');
    m.enter();
    await sleep(200);
    m.leave();
    await sleep(100);
    m.docMove(104, 22);                    // back into the zone before the delay ran out
    await sleep(450);
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

test('a drag that ends inside the hover zone does not close the strip after the grace', async () => {
    const m = mount({ hover: true });
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.enter();
    await sleep(200);
    m.down(100);
    m.up(104);                             // released 4 px past the control's right edge
    await m.settle();
    m.leave('mouse', { x: 104, y: 10 });
    await sleep(450);
    assert.equal(m.control.isOpen(), true, 'the pointer is still within the zone');
    m.docMove(300, 10);
    await sleep(450);
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

test('a hover-opened strip closes about 300 ms after the mouse leaves, and the field below is not its zone (#2873)', async () => {
    const m = mount({ hover: true });
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.enter();
    await sleep(200);
    assert.equal(m.control.isOpen(), true);
    m.leave('mouse', { x: 50, y: 24 });  // 4 px under the control: still its zone
    await sleep(450);
    assert.equal(m.control.isOpen(), true, 'a small drift is forgiven');
    m.docMove(50, 40);  // 20 px under it, where the message field begins
    await sleep(180);
    assert.equal(m.control.isOpen(), true, 'the grace has not run out yet');
    await sleep(250);
    assert.equal(m.control.isOpen(), false, 'closed about 300 ms after leaving the zone');
    m.control.destroy();
});

test('after Escape, a strip the mouse opens again does not inherit keyboard focus and closes on leave (#2873)', async () => {
    const m = mount({ hover: true });
    m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    m.enter();
    await sleep(200);
    m.escape();
    assert.equal(m.control.isOpen(), false);
    assert.equal(m.doc.activeElement, m.head, 'Escape returns focus to the head');
    m.leave('mouse', { x: 600, y: 10 });
    m.enter();
    await sleep(200);
    assert.equal(m.control.isOpen(), true, 'hover opens it again');
    m.leave('mouse', { x: 600, y: 10 });
    await sleep(450);
    assert.equal(m.control.isOpen(), false, 'the focus Escape left does not hold a hover-opened strip');
    m.control.destroy();
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
    // Closed, the value is what this room's latest requests were sent in: before any census, a dash, never the choice.
    assert.equal(m.contextEl.querySelector('.chat-capsule-value').textContent, '—');
    assert.equal(m.contextEl.dataset.contextActual, 'unknown');
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
    assert.equal(m.seg('nano').getAttribute('aria-checked'), 'true', 'the open strip marks the saved mode');
    assert.equal(m.contextEl.querySelector('.chat-capsule-value').textContent, '—', 'a saved mode is not an observation');
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
    assert.equal(m.contextEl.querySelector('.chat-capsule-value').textContent, '—', 'a choice never paints the observation');
    assert.equal(m.seg('low').getAttribute('aria-checked'), 'true');
    assert.equal(m.contextEl.dataset.disabled, 'false');
    assert.deepEqual(m.refreshes, [true]);
    await m.choose('low');
    assert.equal(m.posts.length, 1, 'the current mode is not re-posted');
    m.answer.value = { ok: false, json: async () => ({ error: 'Context mode can only be lowered while Ouroboros is idle.' }) };
    await m.choose('max');
    assert.equal(m.contextEl.dataset.contextMode, 'low', 'a refusal leaves the saved mode');
    assert.equal(m.seg('low').getAttribute('aria-checked'), 'true');
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

// A census row of a room's own work whose latest sent request carries its physical context (neutral fixture).
const sent = (applied, { chat = 1, id = 'work-a', preferred = 'max', cause = 'memory', at = '2026-01-05T10:04:00Z', ...rest } = {}) => ({
    activity_id: id, chat_id: chat, project_id: '', kind: 'managed_task', phase: 'working', task_attempt: 1, ...rest,
    context_attempt: { attempt_id: `${id}-1`, ledger_state: 'dispatched', round_id: 'r1', preferred_at_send: preferred, applied,
        profile: `owner_${applied}`, cause: applied === preferred ? '' : cause, observed_at: at, task_attempt: 1,
        llm_call_id: 'c1', execution_id: 'e1', call_attempt: 1 },
});
const closed = (m) => ({
    value: m.contextEl.querySelector('.chat-capsule-value').textContent, actual: m.contextEl.dataset.contextActual,
    lit: m.contextEl.querySelectorAll('.chat-context-arc').filter((arc) => arc.getAttribute('data-lit') === 'true').map((arc) => arc.dataset.arc),
    dot: m.contextEl.dataset.contextDiffers === 'true'
        ? [m.contextEl.querySelector('.chat-context-choice').getAttribute('cx'), m.contextEl.querySelector('.chat-context-choice').getAttribute('cy')] : null,
    label: m.contextHead.getAttribute('aria-label'),
    fact: m.contextHint.querySelector('.ui-hint-fact').hidden ? null
        : [m.contextHint.querySelector('.ui-hint-fact-word').textContent, m.contextHint.querySelector('.ui-hint-fact-text').textContent],
});

test('closed, Context says the mode this room\'s latest request was sent in; the saved mode is a dot on its arc only while they disagree', async () => {
    const m = mountPair();
    m.controls.syncState({ context_mode: 'max', active_chat_activities: [sent('low')] });
    const low = closed(m);
    assert.deepEqual([low.value, low.actual, low.lit, low.dot, low.label], ['Low', 'low', ['low'], ['157.78', '115.94'], 'Context: Low now, Max chosen'],
        'one word and one lit arc for the observation; Effort\'s dot on the Max arc keeps the choice');
    assert.equal(low.fact[0], 'Low now');
    assert.match(low.fact[1], /^The history does not fit the Max window\. Request at .+\.$/, 'one short cause and the request\'s time');
    assert.deepEqual(m.posts, [], 'painting the observation saves nothing');
    assert.equal(m.contextEl.dataset.contextMode, 'max', 'nor changes the saved mode');
    await m.press(m.contextHead);
    assert.equal(m.seg('max').getAttribute('aria-checked'), 'true', 'open, the strip marks the saved mode');
    assert.deepEqual(['word', 'text'].map((part) => m.contextHint.querySelector(`.ui-hint-${part}`).textContent), ['Max', 'The full context window.']);
    // Choosing the observed mode makes them agree: today's closed capsule, no dot, the head's own words; open, the hint
    // still dates the observation (no cause: nothing differs).
    await m.choose('low');
    const agreed = closed(m);
    assert.deepEqual({ ...agreed, fact: null }, { value: 'Low', actual: 'low', lit: ['low'], dot: null, label: '', fact: null });
    assert.equal(agreed.fact[0], 'Low now');
    assert.match(agreed.fact[1], /^Request at .+\.$/);
    // A choice changes the dot, never the word: the next request will try Nano, the last one was sent in Low.
    await m.choose('nano');
    assert.deepEqual([closed(m).value, closed(m).lit, closed(m).dot], ['Low', ['low'], ['26.22', '115.94']]);
    assert.equal(closed(m).fact[1].split(' Request at')[0], 'The history does not fit the Max window.',
        'the cause is about the mode its work held at send');
    m.controls.destroy();
});

test('each recorded reason of a lower request reads as its own sentence; a lower request without one states none', () => {
    const m = mountPair();
    const reason = (cause) => {
        m.controls.syncState({ context_mode: 'max', active_chat_activities: [sent('low', { cause, id: `work-${cause}` })] });
        return closed(m).fact[1].split(/ ?Request at /)[0];
    };
    const said = ['memory', 'route', 'window', 'after_start'].map(reason);
    assert.ok(said.every(Boolean) && new Set(said).size === said.length, said.join(' | '));
    assert.equal(reason(''), '', 'an unknown start (no recorded reason) is not given one');
    m.controls.destroy();
});

test('a room reads only its own root works: now, mixed, last and unknown; helpers, other rooms and older attempts never count', () => {
    const main = mountPair(), project = mountPair({ projectId: 'alpha', chatId: 42 });
    const rows = [
        sent('max', { id: 'helper', parent_task_id: 'work-a', delegation_role: 'subagent' }),
        sent('nano', { id: 'other-room', chat: 42 }),
        { ...sent('nano', { id: 'old-attempt' }), task_attempt: 2 },
        sent('low', { id: 'work-a' }),
    ];
    const data = (activities) => ({ context_mode: 'max', projects: [{ id: 'alpha', cognition: { context_mode: 'max' } }], active_chat_activities: activities });
    for (const m of [main, project]) m.controls.syncState(data(rows));
    assert.deepEqual([closed(main).value, closed(main).lit], ['Low', ['low']], 'Main: its root work alone');
    assert.deepEqual([closed(project).value, closed(project).lit], ['Nano', ['nano']], 'the Project: its own room');
    // Two of Main's own works in two modes: both arcs, one word, never one of them passed off as the room's.
    main.controls.syncState(data([...rows, sent('max', { id: 'work-b', at: '2026-01-05T10:06:00Z' })]));
    assert.deepEqual([closed(main).value, closed(main).actual, closed(main).lit, closed(main).label],
        ['Mixed', 'mixed', ['low', 'max'], 'Context: Mixed, Low and Max, Max chosen']);
    // Each work's own latest request; the time is the newest of them (10:06 here), not a moment both shared.
    assert.equal(closed(main).fact[0], 'Mixed');
    assert.match(closed(main).fact[1], /^Latest requests of works here: Low and Max\. Newest at .+\.$/);
    // A body without the census (a Project's own cognition read) keeps what was seen.
    main.controls.syncState({ context_mode: 'max' });
    assert.equal(closed(main).value, 'Mixed');
    // No work in the census now: the newest request this page saw, said as the last one; it agrees with Max here, so
    // closed it is today's capsule, and open the hint still dates it.
    main.controls.syncState(data([]));
    assert.deepEqual([closed(main).value, closed(main).lit, closed(main).dot, closed(main).label], ['Max', ['max'], null, '']);
    assert.equal(closed(main).fact[0], 'Max last');
    assert.match(closed(main).fact[1], /^Request at .+\.$/);
    project.controls.syncState(data([]));
    assert.deepEqual([closed(project).value, closed(project).label, closed(project).fact[0]], ['Nano', 'Context: Nano last, Max chosen', 'Nano last']);
    // A page that saw nothing says so: a dash in secondary ink, no lit arc, the choice's dot.
    const fresh = mountPair();
    fresh.controls.syncState(data([]));
    assert.deepEqual(closed(fresh), { value: '—', actual: 'unknown', lit: [], dot: ['157.78', '115.94'],
        label: 'Context: actual unknown, Max chosen', fact: ['Unknown', 'No request from this room seen since this page opened.'] });
    for (const m of [main, project, fresh]) m.controls.destroy();
});

test('staggered completion preserves the newest request even when an older work outlives it', () => {
    const m = mountPair();
    const read = (rows) => m.controls.syncState({ context_mode: 'max', active_chat_activities: rows });
    const older = sent('low'), newer = sent('max', { id: 'work-b', at: '2026-01-05T10:06:00Z' });
    read([older, newer]);
    assert.equal(closed(m).value, 'Mixed');
    read([older]);
    assert.deepEqual([closed(m).value, closed(m).fact[0]], ['Low', 'Low now']);
    read([]);
    const last = closed(m);
    assert.deepEqual([last.value, last.fact[0]], ['Max', 'Max last']);
    const fresh = mountPair();
    fresh.controls.syncState({ context_mode: 'max', active_chat_activities: [newer] });
    fresh.controls.syncState({ context_mode: 'max', active_chat_activities: [] });
    assert.deepEqual(last, closed(fresh), 'the date too is the newer request, not the last work to leave');
    for (const control of [m, fresh]) control.controls.destroy();
});

test('a released dispatch retracts only that attempt in current and retained facts, keeping its previous genuine send', () => {
    for (const priorSend of [false, true]) for (const partial of [false, true]) for (const ended of [false, true]) {
        const m = mountPair();
        const read = (rows, incomplete = false) => m.controls.syncState({ context_mode: 'max', active_chat_activities: rows,
            active_chat_activities_complete: !incomplete });
        const previous = sent('max', { at: '2026-01-05T10:01:00Z' }).context_attempt;
        previous.attempt_id = 'earlier-send'; previous.ledger_state = 'settled';
        const candidate = sent('low');
        if (priorSend) candidate.context_attempt.previous = previous;
        read([candidate]);
        assert.equal(closed(m).value, 'Low');
        if (ended) { read([]); assert.equal(closed(m).fact[0], 'Low last'); }
        const retracted = { ...candidate, context_attempt: { ...candidate.context_attempt, ledger_state: 'released' } };
        read([retracted], partial);
        assert.equal(closed(m).value, priorSend ? 'Max' : '—');
        if (priorSend) {
            assert.equal(closed(m).fact[0], partial ? 'Max last' : 'Max now');
            assert.doesNotMatch(closed(m).fact[1], /history/);
        }
        read([candidate]);  // a late poll cannot revive the retracted identity
        assert.equal(closed(m).value, priorSend ? 'Max' : '—');
        read([]);
        assert.equal(closed(m).value, priorSend ? 'Max' : '—');
        if (priorSend) assert.equal(closed(m).fact[0], 'Max last');
        m.controls.destroy();
    }
});

test('a later release retracts a previous send carried by a resumed released attempt', () => {
    const m = mountPair();
    const read = (rows) => m.controls.syncState({ context_mode: 'max', active_chat_activities: rows });
    const b = sent('low', { at: '2026-01-05T10:04:00Z' });
    read([b]);
    const c = sent('low', { at: '2026-01-05T10:05:00Z' });
    c.context_attempt.attempt_id = 'work-a-2';
    c.context_attempt.previous = { ...b.context_attempt, attempt_id: 'work-a-1' };
    c.context_attempt.ledger_state = 'released';
    read([c]);
    assert.equal(closed(m).value, 'Low', 'the resumed release temporarily exposes its earlier send');
    // The production event stream corrects C in place: supervisor/events.py
    // removes C.previous when B is proved never to have started.
    const cCorrected = { ...c, context_attempt: { ...c.context_attempt } };
    delete cCorrected.context_attempt.previous;
    read([cCorrected]);
    assert.deepEqual([closed(m).value, closed(m).actual], ['—', 'unknown'],
        'once the earlier attempt is disproven, the carried observation is retracted too');
    read([c]);  // a stale retry frame cannot restore the disproven earlier send
    assert.deepEqual([closed(m).value, closed(m).actual], ['—', 'unknown']);
    m.controls.destroy();
});

// The supervisor's public fact of one work: its newest attempt, and the next attempt not proven unsent behind it.
const attemptFact = (id, applied, state, minute, previous) => ({ attempt_id: id, ledger_state: state, applied, task_attempt: 1,
    preferred_at_send: 'max', cause: '', observed_at: `2026-01-05T10:0${minute}:00Z`, ...(previous ? { previous } : {}) });
const foldFacts = (facts) => facts.reduce((seen, fact) => observeContext(seen,
    [{ activity_id: 'work-a', chat_id: 1, task_attempt: 1, context_attempt: fact }], 1), null);
const shownAttempt = (seen) => [seen.state, seen.modes.join(), seen.latest?.attempt_id ?? null];

test('a released attempt takes its revised previous, older as each earlier send is disproven, and no stale read brings one back', () => {
    const p = attemptFact('p', 'max', 'settled', 1);
    const a = attemptFact('a', 'low', 'dispatched', 2);
    const aSettled = { ...a, ledger_state: 'settled' };
    const b = attemptFact('b', 'max', 'released', 3);
    const withPrevious = (previous) => (previous ? { ...b, previous } : b);
    assert.deepEqual(shownAttempt(foldFacts([withPrevious(a)])), ['now', 'low', 'a'], 'A is still not proven unsent');
    assert.deepEqual(shownAttempt(foldFacts([withPrevious(a), withPrevious(aSettled)])), ['now', 'low', 'a'],
        'a late settlement of the older attempt changes its state, not the answer');
    const aDisproven = [withPrevious(a), withPrevious(p)];
    assert.deepEqual(shownAttempt(foldFacts(aDisproven)), ['now', 'max', 'p'], 'A proven unsent: P takes its place');
    assert.deepEqual(shownAttempt(foldFacts([...aDisproven, withPrevious(a)])), ['now', 'max', 'p'],
        'a stale read cannot revive the disproven A');
    assert.deepEqual(shownAttempt(foldFacts([...aDisproven, withPrevious(null)])), ['unknown', '', null],
        'every earlier send disproven: nothing stands in for it');
    assert.deepEqual(shownAttempt(foldFacts([...aDisproven, withPrevious(null), withPrevious(p), withPrevious(a)])), ['unknown', '', null]);
});

test('a read out of order never trades an older previous for a newer one, and a late dispatch never revives a released attempt', () => {
    const p = attemptFact('p', 'max', 'settled', 1);
    const a = attemptFact('a', 'low', 'dispatched', 2);
    const released = attemptFact('b', 'max', 'released', 3, p);
    assert.deepEqual(shownAttempt(foldFacts([released, { ...released, previous: a }])), ['now', 'max', 'p'],
        'the newer previous is the older read');
    assert.deepEqual(shownAttempt(foldFacts([released, attemptFact('b', 'max', 'dispatched', 3, p)])), ['now', 'max', 'p'],
        'B stays released');
    assert.deepEqual(shownAttempt(foldFacts([attemptFact('b', 'low', 'dispatched', 3, a)])), ['now', 'low', 'b'],
        'a live dispatch is the newest candidate');
    assert.deepEqual(shownAttempt(foldFacts([attemptFact('c', 'low', 'dispatched', 4), attemptFact('b', 'max', 'released', 3, p)])),
        ['now', 'low', 'c'], 'an older attempt of the same task attempt cannot replace a newer one');
    const early = { ...p, observed_at: '2026-01-05T10:01:00.000101Z' };
    const later = { ...a, observed_at: '2026-01-05T10:01:00.000102Z' };
    assert.deepEqual(shownAttempt(foldFacts([{ ...released, previous: early }, { ...released, previous: later }])),
        ['now', 'max', 'p'], 'even the first stale predecessor read keeps microsecond dispatch ordering');
    assert.deepEqual(shownAttempt(foldFacts([{ ...released, previous: later }, { ...released, previous: early },
        { ...released, previous: later }])), ['now', 'max', 'p'], 'a narrowed predecessor never revives');
});

test('without a current read (a socket close or a failed read) now is never claimed: the dated last stands until a read returns', () => {
    const m = mountPair();
    m.controls.syncState({ context_mode: 'max', active_chat_activities: [sent('low')] });
    assert.equal(closed(m).label, 'Context: Low now, Max chosen');
    m.controls.syncState({ projects: [] });  // a Project list read carries no census: nothing changes
    assert.equal(closed(m).label, 'Context: Low now, Max chosen');
    m.controls.syncState({ accounting: { available: false } });  // chat.js's frame on a close and on a failed read
    assert.deepEqual([closed(m).value, closed(m).label, closed(m).fact[0]], ['Low', 'Context: Low last, Max chosen', 'Low last']);
    assert.match(closed(m).fact[1], /The current read is incomplete\.$/, 'no read is no departure of the work');
    assert.equal(m.contextEl.dataset.contextMode, 'max');
    m.controls.syncState({ context_mode: 'max', active_chat_activities: [sent('low')] });
    assert.equal(closed(m).label, 'Context: Low now, Max chosen', 'a current read says now again');
    m.controls.destroy();
});

test('a disconnection is no departure: a send released after it never takes the earlier ended work\'s request with it', () => {
    const m = mountPair();
    const read = (rows) => m.controls.syncState({ context_mode: 'max', active_chat_activities: rows });
    const a = sent('max', { at: '2026-01-05T10:01:00Z' });
    const b = sent('low', { id: 'work-b', at: '2026-01-05T10:06:00Z' });
    read([a]);
    read([]);  // A completed in Max
    read([b]);  // B provisionally dispatched in Low
    assert.deepEqual([closed(m).value, closed(m).fact[0]], ['Low', 'Low now']);
    m.controls.syncState({ accounting: { available: false } });  // the socket closes: no current read
    assert.deepEqual([closed(m).value, closed(m).fact[0]], ['Low', 'Low last']);
    read([{ ...b, context_attempt: { ...b.context_attempt, ledger_state: 'released' } }]);  // B never started
    assert.deepEqual([closed(m).value, closed(m).actual, closed(m).fact[0]], ['Max', 'max', 'Max last']);
    read([]);
    assert.deepEqual([closed(m).value, closed(m).fact[0]], ['Max', 'Max last']);
    m.controls.destroy();
});

test('an explicitly incomplete census never claims the room\'s current mode: its facts count as seen, the newest stands as last', () => {
    const m = mountPair();
    const read = (rows, flags = {}) => m.controls.syncState({ context_mode: 'max', active_chat_activities: rows, ...flags });
    read([sent('low')], { active_chat_activities_complete: true, supervisor_ready: true });
    assert.equal(closed(m).label, 'Context: Low now, Max chosen', 'a complete read is current');
    // An incomplete read lists a newer request of another work: seen (it becomes the newest), never "now" or Mixed.
    read([sent('max', { id: 'work-b', at: '2026-01-05T10:07:00Z' })], { active_chat_activities_complete: false, supervisor_ready: true });
    assert.deepEqual([closed(m).value, closed(m).actual, closed(m).lit, closed(m).dot, closed(m).label], ['Max', 'max', ['max'], null, '']);
    assert.equal(closed(m).fact[0], 'Max last');
    assert.match(closed(m).fact[1], /^Request at .+\. The current read is incomplete\.$/);
    // An incomplete read that lists nothing for this room (a supervisor not ready): the newest seen stays, as last.
    read([], { active_chat_activities_complete: true, supervisor_ready: false });
    assert.deepEqual([closed(m).value, closed(m).fact[0]], ['Max', 'Max last']);
    assert.match(closed(m).fact[1], /The current read is incomplete\.$/);
    // A complete read again: current, and only what it lists.
    read([sent('low')], { active_chat_activities_complete: true, supervisor_ready: true });
    assert.deepEqual([closed(m).value, closed(m).label], ['Low', 'Context: Low now, Max chosen']);
    assert.doesNotMatch(closed(m).fact[1], /incomplete/);
    m.controls.destroy();
    // A fresh page whose first read is incomplete: its listed fact is kept, dated, as last (never now).
    const fresh = mountPair();
    fresh.controls.syncState({ context_mode: 'max', active_chat_activities: [sent('low')], active_chat_activities_complete: false });
    assert.deepEqual([closed(fresh).value, closed(fresh).label, closed(fresh).fact[0]], ['Low', 'Context: Low last, Max chosen', 'Low last']);
    assert.match(closed(fresh).fact[1], /^The history does not fit the Max window\. Request at .+\. The current read is incomplete\.$/);
    // ... and one whose incomplete read lists nothing: unknown, saying why.
    const empty = mountPair();
    empty.controls.syncState({ context_mode: 'max', active_chat_activities: [], supervisor_ready: false });
    assert.deepEqual([closed(empty).value, closed(empty).actual, closed(empty).fact], ['—', 'unknown', ['Unknown', 'The current read is incomplete.']]);
    for (const x of [fresh, empty]) x.controls.destroy();
});

test('a late census read never takes a work back to an older request of the same attempt', () => {
    const m = mountPair();
    m.controls.syncState({ context_mode: 'max', active_chat_activities: [sent('max', { at: '2026-01-05T10:09:00Z' })] });
    m.controls.syncState({ context_mode: 'max', active_chat_activities: [sent('low', { at: '2026-01-05T10:04:00Z' })] });
    assert.deepEqual([closed(m).value, closed(m).dot], ['Max', null]);
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

test('closed names give way one at a time, the longest first: two equal values keep the words that tell them apart (U123)', () => {
    const m = mountPair();
    const name = (capsule) => capsule.querySelector('.chat-capsule-name');
    name(m.contextEl).getBoundingClientRect = () => ({ top: 0, bottom: 20, width: 52 });
    name(m.effortEl).getBoundingClientRect = () => ({ top: 0, bottom: 20, width: 40 });
    // The line holds the row once Context's longer name has stepped aside.
    m.effortEl.getBoundingClientRect = () => {
        const top = m.contextEl.dataset.nameFit === 'off' ? 0 : 40;
        return { top, bottom: top + 32, width: 130 };
    };
    m.controls.syncState({});
    assert.equal(m.contextEl.dataset.nameFit, 'off', 'the longest name goes first');
    assert.equal(m.effortEl.dataset.nameFit, undefined, 'the rest keep theirs');
    assert.equal(m.row.dataset.ownerCompact, undefined, 'and the row is not compact');
    // A line that still wraps without every name compacts the row as before.
    m.effortEl.getBoundingClientRect = () => ({ top: 40, bottom: 72, width: 130 });
    m.controls.syncState({});
    assert.deepEqual([m.contextEl.dataset.nameFit, m.effortEl.dataset.nameFit, m.row.dataset.ownerCompact], ['off', 'off', 'true']);
    m.effortEl.getBoundingClientRect = () => ({ top: 0, bottom: 32, width: 130 });
    m.controls.syncState({});
    assert.deepEqual([m.contextEl.dataset.nameFit, m.effortEl.dataset.nameFit], [undefined, undefined], 'room again: every name back');
    m.controls.destroy();
});

// A row laid out by numbers: each closed capsule is its glyph, its name while it has one and its value as the row's
// fit leaves it (cut to the cap, or folded away with the glyphs); the last capsule drops to a second line while the
// four are wider than the row.
function layRow(m, { width, values, names = { model: 30, subagents: 70, context: 52, effort: 40 } }) {
    const GLYPH = 34;
    const els = { model: m.modelEl, subagents: m.subagentsEl, context: m.contextEl, effort: m.effortEl };
    const kinds = Object.keys(els).filter((kind) => els[kind].parentNode === m.row && els[kind].children.length);
    const cap = () => { const said = m.row.style.getPropertyValue('--capsule-value-cap'); return said ? parseFloat(said) : Infinity; };
    const valueWidth = (kind) => (m.row.dataset.ownerGlyphs === 'true' && kind !== 'model' ? 0
        : m.row.dataset.ownerCompact === 'values' ? Math.min(values[kind], cap()) : values[kind]);
    const nameWidth = (kind) => (els[kind].dataset.nameFit === 'off' ? 0 : names[kind]);
    const capsuleWidth = (kind) => GLYPH + nameWidth(kind) + valueWidth(kind);
    const total = () => kinds.reduce((sum, kind) => sum + capsuleWidth(kind), 0);
    m.row.getBoundingClientRect = () => ({ top: 0, bottom: 32, left: 0, right: width, width, height: 32 });
    for (const kind of kinds) {
        const last = kind === kinds.at(-1);
        els[kind].getBoundingClientRect = () => { const top = last && total() > width ? 40 : 0; return { top, bottom: top + 32, width: capsuleWidth(kind) }; };
        els[kind].querySelector('.chat-capsule-name').getBoundingClientRect = () => ({ top: 0, bottom: 20, width: nameWidth(kind) });
        els[kind].querySelector('.chat-capsule-value').getBoundingClientRect = () => ({ top: 0, bottom: 20, width: valueWidth(kind) });
    }
    return { total, said: () => [m.row.dataset.ownerCompact, m.row.dataset.ownerGlyphs, m.row.style.getPropertyValue('--capsule-value-cap')] };
}

test('the model keeps its value: where the cap would cut it, Subagents, Context and Effort fold to their glyphs first (owner 10.11)', () => {
    const store = createModelCatalogStore({ readCatalog: async () => ({ items: [] }), readPreferences: async () => ({}) });
    const m = mountPair({ subagents: true, model: { store } });
    const values = { model: 90, subagents: 40, context: 30, effort: 50 };
    // Wide enough for every name and value: nothing gives way.
    let line = layRow(m, { width: 560, values });
    m.controls.syncState({});
    assert.deepEqual(line.said(), [undefined, undefined, '']);
    // The names are not enough, and the cap that would hold the line (46 px) cuts the model's 90: the neighbours fold.
    line = layRow(m, { width: 300, values });
    m.controls.syncState({});
    assert.deepEqual(line.said(), ['true', 'true', ''], 'the glyphs alone hold the line: the model\'s value is whole');
    assert.ok(line.total() <= 300);
    // A line that still wraps with the glyphs folded cuts the model's own value, last.
    line = layRow(m, { width: 200, values });
    m.controls.syncState({});
    assert.deepEqual(line.said(), ['values', 'true', '63px']);
    assert.ok(line.total() <= 200);
    // Room again: values and names come back.
    line = layRow(m, { width: 560, values });
    m.controls.syncState({});
    assert.deepEqual(line.said(), [undefined, undefined, '']);
    // A model with a short value is not what the cap cuts: the widest values are cut as before, nothing folds.
    line = layRow(m, { width: 380, values: { model: 20, subagents: 80, context: 80, effort: 80 } });
    m.controls.syncState({});
    assert.deepEqual(line.said(), ['values', undefined, '74px']);
    m.controls.destroy();
    assert.deepEqual([m.row.dataset.ownerCompact, m.row.dataset.ownerGlyphs], [undefined, undefined]);
});

test('a row without the model keeps the fit it had: names, then the widest values cut, never glyphs alone', () => {
    const m = mountPair({ subagents: true });
    const line = layRow(m, { width: 180, values: { subagents: 40, context: 30, effort: 50 } });
    m.controls.syncState({});
    assert.deepEqual(line.said(), ['values', undefined, '25px']);
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
