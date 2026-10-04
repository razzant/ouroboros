import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
    createWidgetWidths, mergeWidgetOrder, moveWidgetKey, normalizeWidgetOrder, sortTabsByWidgetOrder,
} from '../modules/widget_reorder.js';
import { requestWidgetListPayload } from '../modules/widget_list.js';

// Widgets lifecycle phase 3: a reorder is a pure move in the KEY order; the
// handles never move an <article> (a moved <iframe> reloads). The card widths
// (`createWidgetWidths`) change custom properties and `ui_preferences.widget_size` only.

test('moveWidgetKey moves a key to a clamped index and returns the same array when nothing changes', () => {
    const order = ['a', 'b', 'c', 'd'];
    assert.deepEqual(moveWidgetKey(order, 'b', 2), ['a', 'c', 'b', 'd']);
    assert.deepEqual(moveWidgetKey(order, 'b', 0), ['b', 'a', 'c', 'd']);
    assert.deepEqual(moveWidgetKey(order, 'c', 0), ['c', 'a', 'b', 'd']);
    assert.deepEqual(moveWidgetKey(order, 'a', Number.MAX_SAFE_INTEGER), ['b', 'c', 'd', 'a']);
    assert.deepEqual(moveWidgetKey(order, 'd', -5), ['d', 'a', 'b', 'c']);
    // Identity means "not moved": same slot, first card moved up, unknown key.
    assert.equal(moveWidgetKey(order, 'b', 1), order);
    assert.equal(moveWidgetKey(order, 'a', -1), order);
    assert.equal(moveWidgetKey(order, 'zzz', 0), order);
    assert.equal(moveWidgetKey([], 'a', 0).length, 0);
    // The input is never mutated.
    assert.deepEqual(order, ['a', 'b', 'c', 'd']);
});

test('a drop onto a target lands after a target the key was before, before a target it was after', () => {
    const order = ['a', 'b', 'c', 'd'];
    // Drag a onto c: a was before c → a lands after c.
    assert.deepEqual(moveWidgetKey(order, 'a', order.indexOf('c')), ['b', 'c', 'a', 'd']);
    // Drag d onto b: d was after b → d lands before b.
    assert.deepEqual(moveWidgetKey(order, 'd', order.indexOf('b')), ['a', 'd', 'b', 'c']);
});

test('normalizeWidgetOrder and sortTabsByWidgetOrder keep the phase-2 contract', () => {
    assert.deepEqual(normalizeWidgetOrder([' a ', '', 'b', 'a', null]), ['a', 'b']);
    assert.deepEqual(normalizeWidgetOrder('nope'), []);
    const tabs = [{ key: 'x' }, { key: 'y' }, { key: 'z' }];
    assert.deepEqual(sortTabsByWidgetOrder(tabs, ['z']).map((tab) => tab.key), ['z', 'x', 'y']);
});

const keysOf = (tabs) => tabs.map((tab) => tab.key);
const tabsOf = (...keys) => keys.map((key) => ({ key }));

test('a reorder of the shown cards keeps every stored key that is not on screen in its slot', () => {
    // Stored [A, H, B]; H's skill is off; the owner moves B before A.
    assert.deepEqual(mergeWidgetOrder(['a', 'h', 'b'], ['b', 'a']), ['b', 'h', 'a']);
    // Stored [A, B, C]; B is off; C moves up: B keeps the middle slot, never the end.
    assert.deepEqual(mergeWidgetOrder(['a', 'b', 'c'], ['c', 'a']), ['c', 'b', 'a']);
    // A shown card the stored order lacks takes a new slot at its end first.
    assert.deepEqual(mergeWidgetOrder(['a', 'h', 'b'], ['n', 'a', 'b']), ['n', 'h', 'a', 'b']);
    // Nothing stored yet: the shown order is the order.
    assert.deepEqual(mergeWidgetOrder([], ['c', 'a']), ['c', 'a']);
    assert.deepEqual(mergeWidgetOrder(null, ['c', 'a']), ['c', 'a']);
    // Both inputs are normalised, neither is mutated.
    const stored = [' a ', 'h', 'a', 'b'];
    assert.deepEqual(mergeWidgetOrder(stored, ['b', 'a', 'b']), ['b', 'h', 'a']);
    assert.deepEqual(stored, [' a ', 'h', 'a', 'b']);
});

test('disable a widget, reorder the others, enable it again: it is back in its old slot', () => {
    const stored = ['s:A', 's:B', 's:C', 's:D'];
    // B's skill is off: the board shows A, C, D, and the owner moves D before C.
    const shown = keysOf(sortTabsByWidgetOrder(tabsOf('s:A', 's:C', 's:D'), stored));
    const written = mergeWidgetOrder(stored, moveWidgetKey(shown, 's:D', shown.indexOf('s:C')));
    assert.deepEqual(written, ['s:A', 's:B', 's:D', 's:C']);
    // B's skill is on again: it stands between A and D, where it stood.
    assert.deepEqual(keysOf(sortTabsByWidgetOrder(tabsOf('s:A', 's:B', 's:C', 's:D'), written)), written);
});

test('a widget that appears joins the end, even when its key sorts before the cards on screen', () => {
    // The server lists by key; nothing is arranged yet and this window shows m, z.
    const listed = tabsOf('a:main', 'm:main', 'z:main');
    assert.deepEqual(keysOf(sortTabsByWidgetOrder(listed, [], ['m:main', 'z:main'])), ['m:main', 'z:main', 'a:main']);
    // The stored order still wins; the shown order places only the keys it lacks.
    assert.deepEqual(keysOf(sortTabsByWidgetOrder(listed, ['z:main'], ['m:main', 'z:main'])), ['z:main', 'm:main', 'a:main']);
    // A window's first list (nothing shown yet) keeps the listing order.
    assert.deepEqual(keysOf(sortTabsByWidgetOrder(listed, [])), ['a:main', 'm:main', 'z:main']);
});

test('the reorder module moves keys only: no node insertion or move API; the masonry places the cards', () => {
    const source = readFileSync(new URL('../modules/widget_reorder.js', import.meta.url), 'utf8');
    assert.match(source, /applyMasonry\(list, \{ order: options\.tabs\(\)\.map\(widgetKey\), spans, onLayout \}\)/);
    for (const forbidden of ['.before(', '.after(', '.prepend(', '.append(', 'insertBefore', 'appendChild', 'replaceWith']) {
        assert.equal(source.includes(forbidden), false, `widget_reorder.js must not use ${forbidden}`);
    }
    assert.match(source, /export function bindWidgetCardReorder\(list, currentOrder, onOrderChange\)/);
});

// --- card widths -------------------------------------------------------------
// The widths controller over the real masonry (web/modules/masonry.js): frames
// run at once, so every relayout is visible in the cards' `--masonry-w`.

function listener() {
    const handlers = new Map();
    return {
        handlers,
        addEventListener(type, fn) { handlers.set(type, [...(handlers.get(type) || []), fn]); },
        removeEventListener(type, fn) { handlers.set(type, (handlers.get(type) || []).filter((item) => item !== fn)); },
        fire(type, event) { (handlers.get(type) || []).forEach((fn) => fn(event)); },
    };
}

function classes(...names) {
    const set = new Set(names);
    return { set, add: (name) => set.add(name), remove: (name) => set.delete(name), contains: (name) => set.has(name) };
}

function styleOf(props) {
    return {
        getPropertyValue: (name) => props.get(name) || '',
        setProperty: (name, value) => props.set(name, value),
        removeProperty: (name) => props.delete(name),
    };
}

function board(keys = ['demo:a', 'demo:b'], { width = 1200, spans = {} } = {}) {
    globalThis.requestAnimationFrame = (callback) => { callback(); return 1; };
    globalThis.cancelAnimationFrame = () => {};
    globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
    globalThis.MutationObserver = class { observe() {} disconnect() {} };
    const doc = listener();
    const cardOf = (key) => {
        const props = new Map();
        const attrs = new Set();
        const card = {
            dataset: { widgetKey: key },
            classList: classes(...(spans[key] === 2 ? ['widgets-card-span-2'] : [])),
            isConnected: true,
            offsetHeight: 100,
            props,
            style: styleOf(props),
            hasAttribute: (name) => attrs.has(name),
            toggleAttribute: (name, on) => (on ? attrs.add(name) : attrs.delete(name), on),
        };
        card.handle = {
            ...listener(),
            captured: null,
            closest: (selector) => (selector === '[data-widget-key]' ? card : null),
            setPointerCapture(id) { this.captured = id; },
            hasPointerCapture(id) { return this.captured === id; },
            releasePointerCapture() { this.captured = null; },
        };
        return card;
    };
    const cards = keys.map(cardOf);
    const status = { textContent: '', dataset: { tone: 'neutral' } };
    const list = {
        dataset: {},
        classList: classes(),
        clientWidth: width,
        ownerDocument: doc,
        parentElement: { querySelector: (selector) => (selector === '[data-widget-arrange-status]' ? status : null) },
        querySelectorAll: (selector) => (selector === '[data-widget-resize-handle]' ? cards.map((card) => card.handle) : cards),
        contains: (item) => cards.includes(item),
        style: styleOf(new Map()),
    };
    const saves = [];
    let prefs = { widget_size: {} };
    const tabs = keys.map((key) => ({ key, span: spans[key] || 1 }));
    const widths = createWidgetWidths(list, {
        tabs: () => tabs,
        prefs: () => prefs,
        adopt(sizes) { prefs = { ...prefs, widget_size: sizes }; },
        save(payload) {
            let settle;
            const done = new Promise((resolve, reject) => { settle = { resolve, reject }; });
            saves.push({ payload, ...settle });
            return done;
        },
    });
    widths.relayout();
    return {
        list, cards, doc, saves, status, widths, prefs: () => prefs,
        adopt(sizes) { prefs = { ...prefs, widget_size: sizes }; },
        add(key) {
            cards.push(cardOf(key));
            tabs.push({ key, span: spans[key] || 1 });
            widths.relayout();
            return cards[cards.length - 1];
        },
    };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
const key = (name, extra = {}) => ({ key: name, preventDefault() { this.prevented = true; }, ...extra });
const width = (card) => card.props.get('--masonry-w');
const deferred = () => {
    let resolve;
    const promise = new Promise((done) => { resolve = done; });
    return { promise, resolve };
};

test('a width from the menu shows at once, saves one write at a time and merges what lands meanwhile', async () => {
    const { cards, saves, widths, prefs } = board();
    // Two one-column cards share the 1200px list in halves.
    assert.deepEqual(cards.map(width), ['593px', '593px'], 'the author span is the starting width');
    widths.setWidth('demo:a', 12);
    assert.deepEqual(cards.map(width), ['1200px', '593px'], 'Full width: the whole row, the other card unchanged');
    assert.deepEqual(prefs().widget_size, { 'demo:a': { w: 12, h: 0 } });
    await settle();
    assert.deepEqual(saves.map((save) => save.payload), [{ widget_size: { 'demo:a': { w: 12, h: 0 } } }]);
    // Three changes while the first write is out: one merged write follows it.
    widths.setWidth('demo:b', 2);
    widths.setWidth('demo:a', 3);
    widths.setWidth('demo:b', null);
    await settle();
    assert.equal(saves.length, 1, 'never two writes in flight');
    // A list read that began before these changes cannot undo them.
    assert.deepEqual(widths.readSizes({ 'demo:a': { w: 1 }, 'demo:b': { w: 12 }, 'other:c': { w: 2 } }), {
        'demo:a': { w: 3, h: 0 }, 'other:c': { w: 2, h: 0 },
    });
    saves[0].resolve({ ok: true });
    await settle();
    assert.deepEqual(saves[1].payload, { widget_size: { 'demo:b': null, 'demo:a': { w: 3, h: 0 } } });
    // Three of four columns, and Reset puts the other card back on its author's one column.
    assert.deepEqual(cards.map(width), ['895px', '289px']);
    saves[1].resolve({ ok: true });
    await settle();
    assert.deepEqual(widths.readSizes({ 'demo:a': { w: 3, h: 0 } }), { 'demo:a': { w: 3, h: 0 } });
});

test('a failed save stays visible, rides along with the next change and clears once saved', async () => {
    const { saves, status, widths } = board();
    widths.setWidth('demo:a', 2);
    await settle();
    saves[0].reject(new Error('HTTP 500'));
    await settle();
    assert.deepEqual([status.textContent, status.dataset.tone], ['Size not saved: HTTP 500', 'error']);
    assert.equal(saves.length, 1, 'nothing retries on its own');
    // A list read still shows the width that failed to save.
    assert.deepEqual(widths.readSizes({}), { 'demo:a': { w: 2, h: 0 } });
    widths.setWidth('demo:b', 3);
    await settle();
    assert.deepEqual(saves[1].payload, { widget_size: { 'demo:a': { w: 2, h: 0 }, 'demo:b': { w: 3, h: 0 } } });
    saves[1].resolve({ ok: true });
    await settle();
    assert.deepEqual([status.textContent, status.dataset.tone], ['', 'neutral']);
    assert.deepEqual(widths.readSizes({}), {});
});

test('an old list reply that lands after a confirmed width write keeps the new width on the card', async () => {
    // The page's composition: the reader is taken as the read begins, then the
    // preferences and the cards arrive under one deadline (requestWidgetListPayload).
    const { cards, saves, widths, adopt } = board();
    const preferences = deferred();
    const list = deferred();
    const readSizes = widths.beginRead();
    const reading = requestWidgetListPayload(
        { uiPreferences: () => preferences.promise, widgets: () => list.promise }, new AbortController(), 60_000);
    preferences.resolve({ widget_size: {} });
    widths.setWidth('demo:a', 12);
    await settle();
    saves[0].resolve({ ok: true });
    await settle();
    list.resolve({ ui_tabs: [] });
    const [, prefs] = await reading;
    adopt(readSizes(prefs.widget_size));
    widths.relayout();
    assert.equal(width(cards[0]), '1200px', 'the stored width stays on screen');
});

test('a read that began before a write completed shows it; a read begun after is the stored truth', async () => {
    const { saves, widths } = board();
    const early = widths.beginRead();
    widths.setWidth('demo:a', 12);
    await settle();
    saves[0].resolve({ ok: true });
    await settle();
    assert.deepEqual(early({ 'other:c': { w: 2 } }), { 'demo:a': { w: 12, h: 0 }, 'other:c': { w: 2, h: 0 } });
    // Begun after the write: the reply as stored, another window's later change included.
    assert.deepEqual(widths.beginRead()({ 'demo:a': { w: 2, h: 0 } }), { 'demo:a': { w: 2, h: 0 } });
    // A Reset written while a read was out is not undone by its reply either.
    const out = widths.beginRead();
    widths.setWidth('demo:a', null);
    await settle();
    saves[1].resolve({ ok: true });
    await settle();
    assert.deepEqual(out({ 'demo:a': { w: 12, h: 0 } }), {});
});

test('a failed save keeps its notice until a write succeeds: no step announced meanwhile replaces it', async () => {
    const { cards, saves, status } = board();
    const handle = cards[0].handle;
    const notice = ['Size not saved: offline', 'error'];
    handle.fire('keydown', key('End'));
    await settle();
    saves[0].reject(new Error('offline'));
    await settle();
    assert.deepEqual([status.textContent, status.dataset.tone], notice);
    // End again: the card is already full width, nothing is written, the notice stays.
    handle.fire('keydown', key('End'));
    await settle();
    assert.equal(saves.length, 1);
    assert.deepEqual([status.textContent, status.dataset.tone], notice);
    // A new step rides along with the failed width; while that write is out the notice stays.
    handle.fire('keydown', key('ArrowLeft'));
    await settle();
    assert.deepEqual(saves[1].payload, { widget_size: { 'demo:a': { w: 3, h: 0 } } });
    assert.deepEqual([status.textContent, status.dataset.tone], notice);
    saves[1].resolve({ ok: true });
    await settle();
    assert.deepEqual([status.textContent, status.dataset.tone], ['', 'neutral']);
    handle.fire('keydown', key('ArrowLeft'));
    assert.equal(status.textContent, 'Width: 2 columns', 'steps are named again once nothing failed is left');
});

test('a width picked from the card menu is named in the live region like a key or a drag', () => {
    // The menu calls setWidth (web/modules/widget_card.js); on a narrow list it is the only path.
    const { status, widths } = board();
    widths.setWidth('demo:a', 2);
    assert.deepEqual([status.textContent, status.dataset.tone], ['Width: 2 columns', 'neutral']);
    widths.setWidth('demo:a', null);
    assert.equal(status.textContent, 'Width: 1 column', 'Reset size names the author default');
});

test('the edge handle keys step through 1, 2, 3 columns and Full width and announce it; other keys pass through', async () => {
    const { cards, saves, status } = board();
    const handle = cards[0].handle;
    const right = key('ArrowRight');
    handle.fire('keydown', right);
    assert.equal(right.prevented, true);
    assert.deepEqual([width(cards[0]), status.textContent], ['794px', 'Width: 2 columns']);
    handle.fire('keydown', key('ArrowRight'));
    assert.deepEqual([width(cards[0]), status.textContent], ['895px', 'Width: 3 columns']);
    handle.fire('keydown', key('End'));
    assert.deepEqual([width(cards[0]), status.textContent], ['1200px', 'Width: full width']);
    handle.fire('keydown', key('Home'));
    assert.deepEqual([width(cards[0]), status.textContent], ['593px', 'Width: 1 column']);
    const left = key('ArrowLeft');
    handle.fire('keydown', left);
    assert.equal(left.prevented, true, 'held at the first step, still answered');
    assert.equal(status.textContent, 'Width: 1 column');
    for (const ignored of [key('ArrowDown'), key('Enter'), key('ArrowRight', { altKey: true }), key('ArrowRight', { metaKey: true })]) {
        handle.fire('keydown', ignored);
        assert.equal(ignored.prevented, undefined, ignored.key);
    }
    assert.equal(width(cards[0]), '593px');
    await settle();
    assert.equal(saves.length, 1, 'the keys share the one-at-a-time write');
});

test('Home on a card the masonry widened beyond its author span stores one column and narrows it', async () => {
    // Two wide cards and one narrow card on a four-column board: the masonry
    // shows the lone narrow card two columns wide while the owner has set no width.
    const { cards, saves, status, prefs } = board(['demo:a', 'demo:b', 'demo:c'], {
        width: 1112, spans: { 'demo:a': 2, 'demo:b': 2 },
    });
    assert.deepEqual(cards.map(width), ['548px', '548px', '548px'], 'the lone narrow card is widened');
    cards[2].handle.fire('keydown', key('Home'));
    assert.deepEqual([width(cards[2]), status.textContent], ['267px', 'Width: 1 column']);
    assert.deepEqual(prefs().widget_size, { 'demo:c': { w: 1, h: 0 } });
    await settle();
    assert.deepEqual(saves.map((save) => save.payload), [{ widget_size: { 'demo:c': { w: 1, h: 0 } } }]);
    // Once the owner's one column is stored, the same key only names it again.
    cards[2].handle.fire('keydown', key('ArrowLeft'));
    await settle();
    assert.equal(saves.length, 1);
    assert.equal(status.textContent, 'Width: 1 column');
});

test('dragging the edge previews the widths its steps can give through the masonry; drop saves, Escape cancels', async () => {
    const { list, cards, doc, saves, status } = board();
    const handle = cards[0].handle;
    // Two one-column cards on 1200px: the steps give the first 593, 794, 895 or 1200px.
    const down = { button: 0, pointerId: 7, clientX: 400, currentTarget: handle, preventDefault() {} };
    handle.fire('pointerdown', down);
    assert.equal(list.classList.contains('resizing'), true);
    assert.equal(handle.captured, 7);
    handle.fire('pointermove', { pointerId: 7, clientX: 400 + 607 });
    assert.deepEqual(cards.map(width), ['1200px', '593px'], 'the right edge at the row\'s end: Full width');
    handle.fire('pointermove', { pointerId: 7, clientX: 400 + 80 });
    assert.equal(width(cards[0]), '593px', 'back over its own width: no preview');
    handle.fire('pointermove', { pointerId: 7, clientX: 400 + 607 });
    await settle();
    assert.equal(saves.length, 0, 'a preview is not a save');
    handle.fire('pointerup', { pointerId: 7 });
    handle.fire('lostpointercapture', { pointerId: 7 });
    assert.equal(list.classList.contains('resizing'), false);
    assert.equal(status.textContent, 'Width: full width');
    await settle();
    assert.deepEqual(saves[0].payload, { widget_size: { 'demo:a': { w: 12, h: 0 } } });

    // From the whole row back toward one column, then Escape.
    handle.fire('pointerdown', { ...down, pointerId: 8 });
    handle.fire('pointermove', { pointerId: 8, clientX: 400 - 607 });
    assert.equal(width(cards[0]), '593px');
    const escape = { key: 'Escape', preventDefault() {}, stopPropagation() {} };
    doc.fire('keydown', escape);
    assert.equal(width(cards[0]), '1200px', 'Escape restores the saved width');
    assert.equal(doc.handlers.get('keydown').length, 0, 'the drag releases its document listener');
    handle.fire('pointerup', { pointerId: 8 });
    saves[0].resolve({ ok: true });
    await settle();
    assert.equal(saves.length, 1);
});

test('a list too narrow for two columns is a stack: widths do not apply and the edge does not drag', () => {
    const { list, cards, widths } = board(['demo:a', 'demo:b'], { width: 500 });
    assert.equal(list.dataset.widgetLayout, 'stack');
    widths.setWidth('demo:a', 12);
    assert.deepEqual(cards.map(width), ['500px', '500px'], 'every card is the column');
    cards[0].handle.fire('pointerdown', { button: 0, pointerId: 9, clientX: 100, currentTarget: cards[0].handle, preventDefault() {} });
    assert.equal(list.classList.contains('resizing'), false);
    // Wide again: the board and the stored width come back.
    list.clientWidth = 1200;
    widths.relayout();
    assert.equal(list.dataset.widgetLayout, 'columns');
    assert.deepEqual(cards.map(width), ['1200px', '593px']);
});

test('on a two-card board the edge drag reaches every width the menu gives, 2 and 3 columns included', async () => {
    // Two one-column cards on a 1400px list: the plan has two columns, but an owner
    // width changes the column count, so the steps give 693, 928, 1045 or 1400px.
    const { cards, saves, status } = board(['demo:a', 'demo:b'], { width: 1400 });
    const handle = cards[0].handle;
    handle.fire('pointerdown', { button: 0, pointerId: 3, clientX: 500, currentTarget: handle, preventDefault() {} });
    const at = (travel) => {
        handle.fire('pointermove', { pointerId: 3, clientX: 500 + travel });
        return width(cards[0]);
    };
    assert.deepEqual([at(235), at(352), at(707), at(0)], ['928px', '1045px', '1400px', '693px']);
    at(352);
    handle.fire('pointerup', { pointerId: 3 });
    assert.equal(status.textContent, 'Width: 3 columns');
    await settle();
    assert.deepEqual(saves[0].payload, { widget_size: { 'demo:a': { w: 3, h: 0 } } });
});

test('the only card on a wide board has no edge to drag; the menu still sizes it; a second card brings the edge back', async () => {
    for (const spans of [{}, { 'demo:solo': 2 }]) {
        const { list, cards, saves, widths, add } = board(['demo:solo'], { width: 1400, spans });
        const solo = cards[0];
        assert.equal(list.dataset.widgetLayout, 'columns', 'a wide list, not a stack: the menu note stays hidden');
        assert.equal(solo.hasAttribute('data-widget-width-fixed'), true, 'every step is the whole row');
        let prevented = false;
        solo.handle.fire('pointerdown', { button: 0, pointerId: 4, clientX: 500, currentTarget: solo.handle, preventDefault() { prevented = true; } });
        assert.deepEqual([list.classList.contains('resizing'), prevented], [false, false]);
        widths.setWidth('demo:solo', 2);
        await settle();
        assert.deepEqual(saves[0].payload, { widget_size: { 'demo:solo': { w: 2, h: 0 } } }, 'the menu keeps saving');
        // A second card: a step can now change the first card, so its edge is offered again.
        const other = add('demo:other');
        assert.equal(solo.hasAttribute('data-widget-width-fixed'), false);
        assert.equal(other.hasAttribute('data-widget-width-fixed'), false);
        solo.handle.fire('pointerdown', { button: 0, pointerId: 5, clientX: 500, currentTarget: solo.handle, preventDefault() {} });
        assert.equal(list.classList.contains('resizing'), true);
    }
});
