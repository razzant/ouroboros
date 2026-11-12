import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
    bindWidgetCardReorder, cancelWidgetCardCarry, createWidgetWidths, mergeWidgetOrder, moveWidgetKey, normalizeWidgetOrder,
    sortTabsByWidgetOrder,
} from '../modules/widget_reorder.js';
import { requestWidgetListPayload } from '../modules/widget_list.js';
import { installDomDouble } from './dom_double.js';

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
    assert.match(source, /applyMasonry\(list, \{ order: preview \|\| options\.tabs\(\)\.map\(widgetKey\), spans, held, onLayout \}\)/);
    for (const forbidden of ['.before(', '.after(', '.prepend(', '.append(', 'insertBefore', 'appendChild', 'replaceWith']) {
        assert.equal(source.includes(forbidden), false, `widget_reorder.js must not use ${forbidden}`);
    }
    assert.match(source, /export function bindWidgetCardReorder\(list, currentOrder, onOrderChange, board = null\)/);
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

function board(keys = ['demo:a', 'demo:b'], { width = 1200, spans = {}, heightInfo = null, resized = null } = {}) {
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
        const edge = (attribute) => ({
            ...listener(),
            captured: null,
            closest: (selector) => (selector === '[data-widget-key]' ? card : null),
            hasAttribute: (name) => name === attribute,
            setPointerCapture(id) { this.captured = id; },
            hasPointerCapture(id) { return this.captured === id; },
            releasePointerCapture() { this.captured = null; },
        });
        card.handle = edge('data-widget-resize-handle');
        // The bottom edge, offered when the board knows the cards' heights (`heightInfo`).
        card.heightHandle = edge('data-widget-height-handle');
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
        querySelectorAll: (selector) => {
            if (selector === '[data-widget-resize-handle]') return cards.map((card) => card.handle);
            if (selector === '[data-widget-height-handle]') return heightInfo ? cards.map((card) => card.heightHandle) : [];
            if (selector === '[data-widget-corner-handle]') return [];
            return cards;
        },
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
        ...(heightInfo ? { heightInfo } : {}),
        ...(resized ? { resized } : {}),
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
    assert.equal(status.textContent, 'Size reset: 1 column, natural height', 'Reset size names the author default and the natural height');
});

test('a height keeps the width, a width keeps the height, Reset clears both, Show all is never saved', async () => {
    const { saves, status, widths, prefs, cards } = board(['demo:a', 'demo:b']);
    widths.setHeight('demo:a', 432);
    assert.deepEqual(prefs().widget_size, { 'demo:a': { w: 0, h: 432 } }, 'a height alone pins no width');
    assert.equal(status.textContent, 'Height: 432 px');
    assert.equal(widths.widthOf('demo:a'), 1, "the author's width still decides");
    assert.equal(cards[0].props.get('--masonry-w'), cards[1].props.get('--masonry-w'), 'the board did not widen or narrow it');
    widths.setWidth('demo:a', 3);
    assert.deepEqual(prefs().widget_size['demo:a'], { w: 3, h: 432 });
    widths.setHeight('demo:a', 0);
    assert.deepEqual(prefs().widget_size['demo:a'], { w: 3, h: 0 }, 'Natural height resets h alone');
    widths.setHeight('demo:a', 560);
    widths.setExpanded('demo:a', true);
    assert.equal(widths.isExpanded('demo:a'), true);
    assert.equal(widths.heightOf('demo:a'), 560, 'Show all keeps the stored height');
    assert.equal(status.textContent, 'Showing all of it');
    widths.setExpanded('demo:a', false);
    assert.equal(status.textContent, 'Back to 560 px');
    widths.resetSize('demo:a');
    assert.equal(prefs().widget_size['demo:a'], undefined);
    // Every write carried only sizes; Show all and Back wrote nothing.
    saves.forEach((save) => save.resolve());
    await settle();
    saves.forEach((save) => save.resolve());
    await settle();
    const written = saves.map((save) => save.payload.widget_size['demo:a']);
    assert.deepEqual(written[0], { w: 0, h: 432 });
    assert.ok(written.every((size) => size === null || (Number.isInteger(size.w) && Number.isInteger(size.h))));
    assert.deepEqual(written[written.length - 1], null, 'Reset size deletes the key');
});

test('a list read that began before a height write keeps the new height on the card', async () => {
    const { saves, widths } = board(['demo:a', 'demo:b']);
    const read = widths.beginRead();
    widths.setHeight('demo:a', 480);
    await settle();
    saves[0].resolve();
    await settle();
    // The reply of the read that was out meanwhile still holds no height for the card.
    assert.deepEqual(read({ 'demo:b': { w: 2, h: 0 } }), { 'demo:a': { w: 0, h: 480 }, 'demo:b': { w: 2, h: 0 } });
    // A read begun after the write is the stored truth.
    assert.deepEqual(widths.beginRead()({ 'demo:a': { w: 0, h: 480 } }), { 'demo:a': { w: 0, h: 480 } });
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

test('a card under 360px wide is marked narrow from the board plan (its lifecycle row goes under its title), a wider one is not', async () => {
    const { cards, widths } = board(['demo:a', 'demo:b'], { width: 600 });
    const marks = () => cards.map((card) => [width(card), card.hasAttribute('data-widget-narrow')]);
    assert.deepEqual(marks(), [['293px', true], ['293px', true]]);
    widths.setWidth('demo:a', 2);
    await settle();
    assert.deepEqual(marks(), [['600px', false], ['293px', true]], 'widened, the card loses the mark; its neighbour keeps it');
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

// --- the bottom edge over a card showing all ------------------------------------
// A self-sizing card whose content is 1500 px with the owner's 400 px saved: what the card shows comes from the
// controller (Show all, the saved height, a height being dragged), and every fit is recorded as shown.
function tallCard({ natural = 1500, fixed = 0 } = {}) {
    const shown = [];
    let api = null;
    const display = (key) => {
        if (fixed) return api.widths.heightOf(key) && !api.widths.isExpanded(key) ? api.widths.heightOf(key) : fixed;
        return api.widths.isExpanded(key) ? natural : api.widths.heightOf(key) || natural;
    };
    api = board(['demo:a', 'demo:b'], {
        heightInfo: (key) => ({ content: display(key), natural: fixed ? 0 : natural }),
        resized: (key) => { if (api) shown.push(display(key)); },
    });
    const press = (pointerId, clientY) => api.cards[0].heightHandle.fire('pointerdown',
        { button: 0, pointerId, clientX: 0, clientY, currentTarget: api.cards[0].heightHandle, preventDefault() {} });
    const move = (pointerId, clientY, clientX = 0) => api.cards[0].heightHandle.fire('pointermove', { pointerId, clientX, clientY });
    const release = (pointerId) => api.cards[0].heightHandle.fire('pointerup', { pointerId });
    return { ...api, shown, display: () => display('demo:a'), press, move, release };
}

async function savedAt(card, h) {
    card.widths.setHeight('demo:a', h);
    await settle();
    card.saves.forEach((save) => save.resolve());
    await settle();
}

test('a height key under Show all steps from the height on screen and ends Show all, even at the saved height', async () => {
    const card = tallCard();
    await savedAt(card, 400);
    card.widths.setExpanded('demo:a', true);
    card.cards[0].heightHandle.fire('keydown', key('ArrowUp'));
    assert.equal(card.widths.isExpanded('demo:a'), false, 'Show all ended');
    const stepped = card.widths.heightOf('demo:a');
    assert.ok(stepped > 1400 && stepped < 1500, `from the 1500 px on screen, not the saved 400: ${stepped}`);
    await savedAt(card, 120);
    card.widths.setExpanded('demo:a', true);
    card.cards[0].heightHandle.fire('keydown', key('Home'));
    assert.equal(card.widths.isExpanded('demo:a'), false, 'the least height again is still a choice');
    assert.equal(card.widths.heightOf('demo:a'), 120);
});

test('a drag whose card leaves the board ends there, and the other edges answer again', () => {
    const { list, cards, widths } = board(['demo:a', 'demo:b'], { heightInfo: () => ({ content: 400, natural: 0 }) });
    const press = (card, pointerId) => card.heightHandle.fire('pointerdown',
        { button: 0, pointerId, clientX: 0, clientY: 500, currentTarget: card.heightHandle, preventDefault() {} });
    press(cards[0], 1);
    assert.equal(list.classList.contains('resizing'), true);
    cards[0].isConnected = false;
    widths.relayout();
    assert.equal(list.classList.contains('resizing'), false, 'the drag ended with its card');
    press(cards[1], 2);
    assert.equal(list.classList.contains('resizing'), true, 'another edge starts');
});

test('a height dragged from Show all back onto the saved height collapses there, and writes nothing', async () => {
    const card = tallCard();
    await savedAt(card, 400);
    card.widths.setExpanded('demo:a', true);
    assert.equal(card.display(), 1500);
    card.press(3, 1500);
    card.move(3, 400);
    assert.equal(card.display(), 400, 'the preview shows the saved height');
    card.release(3);
    assert.equal(card.widths.isExpanded('demo:a'), false, 'Show all ended with the drop');
    assert.equal(card.display(), 400, 'the card rests at the saved height, not back at 1500');
    assert.equal(card.status.textContent, 'Height: 400 px');
    await settle();
    assert.equal(card.saves.length, 1, 'the saved height is not written again');
    assert.deepEqual(card.prefs().widget_size['demo:a'], { w: 0, h: 400 });
});

test('a press on the bottom edge of a card showing all that hardly moves keeps Show all and the saved height', async () => {
    const card = tallCard();
    await savedAt(card, 400);
    card.widths.setExpanded('demo:a', true);
    card.shown.length = 0;
    card.press(4, 1500);
    card.move(4, 1500);
    card.move(4, 1497, 6);
    card.move(4, 1490);
    assert.equal(card.display(), 1500, 'no preview: under the travel that counts, or the height it already shows');
    card.release(4);
    assert.equal(card.widths.isExpanded('demo:a'), true);
    assert.deepEqual(card.prefs().widget_size['demo:a'], { w: 0, h: 400 }, 'the saved cap stays');
    assert.ok(card.shown.every((height) => height === 1500), `it never showed less meanwhile: ${card.shown}`);
    await settle();
    assert.equal(card.saves.length, 1, 'nothing written');
});

test('from Show all a real drag sets a new height, and Escape puts Show all and the saved height back', async () => {
    const card = tallCard();
    await savedAt(card, 400);
    card.widths.setExpanded('demo:a', true);
    card.press(5, 1500);
    card.move(5, 600);
    assert.equal(card.display(), 600);
    card.list.ownerDocument.fire('keydown', { key: 'Escape', preventDefault() {}, stopPropagation() {} });
    assert.equal(card.widths.isExpanded('demo:a'), true, 'Escape: Show all again');
    assert.equal(card.display(), 1500);
    card.release(5);
    card.press(6, 1500);
    card.move(6, 800);
    card.release(6);
    assert.equal(card.widths.isExpanded('demo:a'), false);
    assert.equal(card.display(), 800);
    await settle();
    assert.deepEqual(card.saves.at(-1).payload, { widget_size: { 'demo:a': { w: 0, h: 800 } } });
});

test('a press on a fixed frame\'s bottom edge that does not travel pins no height', async () => {
    // A fixed 320 px frame with no owner height: its content height is unknown, so the drag starts from 320.
    const card = tallCard({ fixed: 320 });
    card.press(7, 320);
    card.move(7, 320);
    card.move(7, 322);
    card.release(7);
    await settle();
    assert.equal(card.saves.length, 0);
    assert.equal(card.prefs().widget_size['demo:a'], undefined);
    // A real drag still sizes it.
    card.press(8, 320);
    card.move(8, 420);
    card.release(8);
    await settle();
    assert.deepEqual(card.saves.at(-1).payload, { widget_size: { 'demo:a': { w: 0, h: 424 } } });
});

test('under a coarse pointer the bottom edge is the finger\'s grip: up and down the height, sideways the width', async () => {
    // A mouse on the bottom edge sets the height alone, whatever its sideways travel (the corner is its two axes).
    const mouse = tallCard();
    mouse.press(1, 1500);
    mouse.move(1, 700, 607);
    mouse.release(1);
    await settle();
    assert.deepEqual(mouse.saves.at(-1).payload, { widget_size: { 'demo:a': { w: 0, h: 704 } } });
    // A finger (theme.js paints the primary input coarse: style.css puts the grip in the card's own foot) on a
    // two-column board: the same travel sets both, saved as one size, and named in one sentence.
    const finger = tallCard();
    finger.list.ownerDocument.documentElement = { dataset: { pointer: 'coarse' } };
    finger.press(2, 1500);
    finger.move(2, 700, 607);
    finger.release(2);
    await settle();
    assert.deepEqual(finger.saves.at(-1).payload, { widget_size: { 'demo:a': { w: 12, h: 704 } } });
    assert.equal(finger.status.textContent, 'Width: full width · Height: 704 px');
    // On a stack (a phone) widths do not apply: the grip sets the height alone.
    const phone = board(['demo:a', 'demo:b'], { width: 500, heightInfo: () => ({ content: 1500, natural: 1500 }) });
    phone.list.ownerDocument.documentElement = finger.list.ownerDocument.documentElement;
    const grip = phone.cards[0].heightHandle;
    grip.fire('pointerdown', { button: 0, pointerId: 3, clientX: 0, clientY: 1500, currentTarget: grip, preventDefault() {} });
    grip.fire('pointermove', { pointerId: 3, clientX: 300, clientY: 700 });
    grip.fire('pointerup', { pointerId: 3 });
    await settle();
    assert.deepEqual(phone.saves.at(-1).payload, { widget_size: { 'demo:a': { w: 0, h: 704 } } });
});

test('Natural height and Reset size tell the fit whether an owner height was taken off', () => {
    const calls = [];
    const { widths } = board(['demo:a', 'demo:b'], { resized: (key, picked) => { calls.push([key, picked]); } });
    calls.length = 0;
    widths.setHeight('demo:a', 360);
    assert.deepEqual(calls, [['demo:a', null]], 'a height is a plain fit');
    calls.length = 0;
    widths.setHeight('demo:a', 0);
    assert.deepEqual(calls, [['demo:a', { hadHeight: true }]], 'Natural height on an owner height starts the fit over');
    calls.length = 0;
    widths.setHeight('demo:a', 0);
    assert.deepEqual(calls, [['demo:a', { hadHeight: false }]], 'already natural');
    calls.length = 0;
    widths.setHeight('demo:b', 360);
    calls.length = 0;
    widths.resetSize('demo:b');
    assert.deepEqual(calls, [['demo:b', { hadHeight: true }]], 'Reset size counts as the same choice');
});

test('a card the board places at a new width is fitted again; a relayout that keeps its width is not', () => {
    const fitted = [];
    const { list, widths } = board(['demo:a', 'demo:b'], { resized: (key) => fitted.push(key) });
    assert.deepEqual(fitted.sort(), ['demo:a', 'demo:b'], 'each card is fitted at its first width');
    fitted.length = 0;
    widths.relayout();
    assert.deepEqual(fitted, [], 'the same widths: nothing to fit');
    // The window narrows: one column, both cards placed wider. Their content rewraps under any owner height.
    list.clientWidth = 500;
    widths.relayout();
    assert.deepEqual(fitted.sort(), ['demo:a', 'demo:b']);
    fitted.length = 0;
    list.clientWidth = 1200;
    widths.relayout();
    fitted.length = 0;
    // A card's own width changes the board's columns here (593 + 593 px become 895 + 289 px): both are fitted.
    widths.setWidth('demo:a', 3);
    assert.deepEqual([...new Set(fitted)].sort(), ['demo:a', 'demo:b']);
});

test('leaving the page puts a carried card back and saves nothing; so does the card leaving the board', (t) => {
    const env = installDomDouble();
    const ticks = [];
    globalThis.requestAnimationFrame = (callback) => { ticks.push(callback); return ticks.length; };
    globalThis.cancelAnimationFrame = () => {};
    t.after(() => env.restore());
    const { doc, el } = env;
    const carried = () => {
        const handle = el('button', { attrs: { 'data-widget-reorder-handle': '' } });
        const card = el('div', { className: 'widgets-card', dataset: { widgetKey: 'a' } }, handle);
        const other = el('div', { className: 'widgets-card', dataset: { widgetKey: 'b' } });
        const list = el('div', {}, card, other);
        list.clientWidth = 600;
        list.rect = { left: 0, top: 0, right: 600, bottom: 400, width: 600, height: 400 };
        card.rect = { left: 0, top: 0, right: 200, bottom: 100, width: 200, height: 100 };
        doc.body.append(list);
        const plan = { placements: [{ left: 0, top: 0, width: 200, span: 1 }, { left: 214, top: 0, width: 200, span: 1 }] };
        const changes = [];
        const shown = [];
        bindWidgetCardReorder(list, () => ['a', 'b'], (order) => changes.push(order), {
            laid: () => ({ plan, items: [card, other], planOrder: () => plan }),
            carry: (key, order) => shown.push([key, order]), announce() {},
        });
        handle.emit('pointerdown', { button: 0, pointerId: 1, pointerType: 'mouse', clientX: 100, clientY: 10 });
        handle.emit('pointermove', { pointerId: 1, clientX: 400, clientY: 30 });
        assert.equal(card.hasAttribute('data-widget-lifted'), true);
        return { handle, card, list, changes, shown };
    };
    // Another page is shown while the card is held: it goes back, the board shows the order as it was, and the
    // release on that page saves nothing; the document hears nothing more and the scroll loop stops.
    const captured = () => [...doc.captures.values()].reduce((count, set) => count + set.size, 0);
    const left = carried();
    assert.ok(captured() > 0, 'a carry listens on the document');
    cancelWidgetCardCarry(left.list);
    assert.equal(left.card.hasAttribute('data-widget-lifted'), false);
    assert.deepEqual(left.shown.at(-1), ['', null]);
    assert.equal(captured(), 0, 'its document listeners are gone');
    left.handle.emit('pointermove', { pointerId: 1, clientX: 500, clientY: 30 });
    left.handle.emit('pointerup', { pointerId: 1, clientX: 500, clientY: 30 });
    ticks.splice(0).forEach((tick) => tick());
    assert.deepEqual(left.changes, []);
    assert.equal(ticks.length, 0, 'no scroll frame asked for after the carry ended');
    // The carried card leaves the board (its skill turned off elsewhere): its next move, or frame, ends the carry.
    const gone = carried();
    gone.card.remove();
    gone.handle.emit('pointermove', { pointerId: 1, clientX: 500, clientY: 30 });
    assert.equal(gone.card.hasAttribute('data-widget-lifted'), false);
    gone.handle.emit('pointerup', { pointerId: 1, clientX: 500, clientY: 30 });
    assert.deepEqual(gone.changes, []);
});

test('a press holds its pointer from the moment it lands and lets it go however it ends; a release it missed saves nothing', (t) => {
    const env = installDomDouble();
    globalThis.requestAnimationFrame = () => 1;
    globalThis.cancelAnimationFrame = () => {};
    t.after(() => env.restore());
    const { doc, el } = env;
    const listening = () => [...doc.captures.values()].reduce((count, set) => count + set.size, 0);
    const pressed = () => {
        const grip = el('button', { attrs: { 'data-widget-reorder-handle': '' } });
        const head = el('div', { className: 'widgets-card-head' }, grip);
        const card = el('div', { className: 'widgets-card', dataset: { widgetKey: 'a' } }, head);
        const other = el('div', { className: 'widgets-card', dataset: { widgetKey: 'b' } });
        const list = el('div', {}, card, other);
        list.clientWidth = 600;
        list.rect = { left: 0, top: 0, right: 600, bottom: 400, width: 600, height: 400 };
        card.rect = { left: 0, top: 0, right: 200, bottom: 100, width: 200, height: 100 };
        doc.body.append(list);
        // The engine's pointer capture on the head and the grip.
        const holder = new Map();
        for (const node of [grip, head]) {
            Object.assign(node, {
                setPointerCapture(id) { holder.set(id, node); },
                hasPointerCapture(id) { return holder.get(id) === node; },
                releasePointerCapture(id) { if (holder.get(id) === node) holder.delete(id); },
            });
        }
        const plan = { placements: [{ left: 0, top: 0, width: 200, span: 1 }, { left: 214, top: 0, width: 200, span: 1 }] };
        const changes = [];
        bindWidgetCardReorder(list, () => ['a', 'b'], (order) => changes.push(order),
            { laid: () => ({ plan, items: [card, other], planOrder: () => plan }), carry() {}, announce() {} });
        return { grip, head, card, list, holder, changes };
    };
    // A mouse press on the head holds the pointer before any move, so its moves and its release over the card's own
    // frame reach the board.
    const head = pressed();
    head.head.emit('pointerdown', { button: 0, pointerId: 1, pointerType: 'mouse', buttons: 1, clientX: 100, clientY: 10 });
    assert.equal(head.holder.get(1), head.head);
    assert.equal(head.list.hasAttribute('data-widget-pressed'), true, 'the frames ignore the pointer from the press on');
    // Its release went where the board did not hear it: the next move reports no button. The press ends there; a
    // later move far away carries nothing, and nothing is saved.
    head.head.emit('pointermove', { pointerId: 1, pointerType: 'mouse', buttons: 0, clientX: 400, clientY: 30 });
    assert.equal(head.holder.has(1), false, 'the pointer is let go');
    assert.equal(head.list.hasAttribute('data-widget-pressed'), false, 'and the frames hear it again');
    assert.equal(listening(), 0, 'the document hears nothing more');
    head.head.emit('pointermove', { pointerId: 1, pointerType: 'mouse', buttons: 0, clientX: 420, clientY: 30 });
    assert.equal(head.card.hasAttribute('data-widget-lifted'), false);
    assert.deepEqual(head.changes, []);
    // A plain click on the head (no travel) lets its pointer go too.
    head.head.emit('pointerdown', { button: 0, pointerId: 2, pointerType: 'mouse', buttons: 1, clientX: 100, clientY: 10 });
    head.head.emit('pointerup', { pointerId: 2, pointerType: 'mouse', buttons: 0, clientX: 100, clientY: 10 });
    assert.equal(head.holder.has(2), false);
    assert.equal(head.list.hasAttribute('data-widget-pressed'), false);
    assert.equal(listening(), 0);
    // A carry from the grip whose release was missed goes back where it was, unsaved, its pointer let go.
    const grip = pressed();
    grip.grip.emit('pointerdown', { button: 0, pointerId: 3, pointerType: 'mouse', buttons: 1, clientX: 190, clientY: 10 });
    grip.grip.emit('pointermove', { pointerId: 3, pointerType: 'mouse', buttons: 1, clientX: 400, clientY: 30 });
    assert.equal(grip.card.hasAttribute('data-widget-lifted'), true);
    grip.grip.emit('pointermove', { pointerId: 3, pointerType: 'mouse', buttons: 0, clientX: 420, clientY: 30 });
    assert.equal(grip.card.hasAttribute('data-widget-lifted'), false);
    assert.equal(grip.holder.has(3), false);
    assert.equal(grip.list.hasAttribute('data-widget-pressed'), false);
    assert.equal(listening(), 0);
    assert.deepEqual(grip.changes, []);
    // Paired: a held carry released over another place saves the new order, and lets its pointer go.
    const held = pressed();
    held.grip.emit('pointerdown', { button: 0, pointerId: 4, pointerType: 'mouse', buttons: 1, clientX: 190, clientY: 10 });
    held.grip.emit('pointermove', { pointerId: 4, pointerType: 'mouse', buttons: 1, clientX: 400, clientY: 30 });
    held.grip.emit('pointerup', { pointerId: 4, pointerType: 'mouse', buttons: 0, clientX: 400, clientY: 30 });
    assert.deepEqual(held.changes, [['b', 'a']]);
    assert.equal(held.holder.has(4), false);
    assert.equal(held.list.hasAttribute('data-widget-pressed'), false);
    // A finger on the grip is held the same way and carries; a cancel puts the card back.
    const finger = pressed();
    finger.grip.emit('pointerdown', { button: 0, pointerId: 5, pointerType: 'touch', clientX: 190, clientY: 10 });
    assert.equal(finger.holder.get(5), finger.grip);
    finger.grip.emit('pointermove', { pointerId: 5, pointerType: 'touch', clientX: 400, clientY: 30 });
    assert.equal(finger.card.hasAttribute('data-widget-lifted'), true);
    finger.grip.emit('pointercancel', { pointerId: 5, pointerType: 'touch', clientX: 400, clientY: 30 });
    assert.deepEqual(finger.changes, [], 'a cancelled carry saves nothing');
    assert.equal(finger.holder.has(5), false);
    assert.equal(finger.list.hasAttribute('data-widget-pressed'), false);
    // Leaving the page while a press waits to become a carry ends it the same way.
    const left = pressed();
    left.head.emit('pointerdown', { button: 0, pointerId: 6, pointerType: 'mouse', buttons: 1, clientX: 100, clientY: 10 });
    cancelWidgetCardCarry(left.list);
    assert.deepEqual([left.holder.has(6), left.list.hasAttribute('data-widget-pressed'), listening()], [false, false, 0]);
    // So does Escape: the key is the press's while it lasts, and a later move carries nothing.
    const escaped = pressed();
    escaped.head.emit('pointerdown', { button: 0, pointerId: 7, pointerType: 'mouse', buttons: 1, clientX: 100, clientY: 10 });
    assert.equal(escaped.head.emit('keydown', { key: 'Escape' }).defaultPrevented, true);
    assert.deepEqual([escaped.holder.has(7), escaped.list.hasAttribute('data-widget-pressed'), listening()], [false, false, 0]);
    escaped.head.emit('pointermove', { pointerId: 7, pointerType: 'mouse', buttons: 1, clientX: 400, clientY: 30 });
    assert.equal(escaped.card.hasAttribute('data-widget-lifted'), false);
    // Any other key leaves the press alone.
    const typed = pressed();
    typed.head.emit('pointerdown', { button: 0, pointerId: 8, pointerType: 'mouse', buttons: 1, clientX: 100, clientY: 10 });
    assert.notEqual(typed.head.emit('keydown', { key: 'a' }).defaultPrevented, true);
    typed.head.emit('pointerup', { pointerId: 8, pointerType: 'mouse', buttons: 0, clientX: 100, clientY: 10 });
    // The card leaves the board while the press waits: its next move, however short, ends the press.
    const gone = pressed();
    gone.head.emit('pointerdown', { button: 0, pointerId: 9, pointerType: 'mouse', buttons: 1, clientX: 100, clientY: 10 });
    gone.card.remove();
    gone.head.emit('pointermove', { pointerId: 9, pointerType: 'mouse', buttons: 1, clientX: 101, clientY: 10 });
    assert.deepEqual([gone.holder.has(9), gone.list.hasAttribute('data-widget-pressed'), listening()], [false, false, 0]);
    assert.deepEqual(gone.changes, []);
});

test('a carried card stays over the board: not past its left or right edge, nor above its top', (t) => {
    const env = installDomDouble();
    globalThis.requestAnimationFrame = () => 1;
    globalThis.cancelAnimationFrame = () => {};
    t.after(() => env.restore());
    const { doc, el } = env;
    const handle = el('button', { attrs: { 'data-widget-reorder-handle': '' } });
    const card = el('div', { className: 'widgets-card', dataset: { widgetKey: 'a' } }, handle);
    const other = el('div', { className: 'widgets-card', dataset: { widgetKey: 'b' } });
    const list = el('div', {}, card, other);
    list.clientWidth = 600;
    list.rect = { left: 0, top: 0, right: 600, bottom: 400, width: 600, height: 400 };
    card.rect = { left: 0, top: 0, right: 200, bottom: 100, width: 200, height: 100 };
    doc.body.append(list);
    const plan = { placements: [{ left: 0, top: 0, width: 200, span: 1 }, { left: 214, top: 0, width: 200, span: 1 }] };
    bindWidgetCardReorder(list, () => ['a', 'b'], () => {},
        { laid: () => ({ plan, items: [card, other], planOrder: () => plan }), carry() {}, announce() {} });
    // Taken by its grip at the head's right end, then pulled far left and above the board.
    handle.emit('pointerdown', { button: 0, pointerId: 1, pointerType: 'mouse', clientX: 190, clientY: 10 });
    handle.emit('pointermove', { pointerId: 1, clientX: 20, clientY: -50 });
    assert.deepEqual([card.css.get('--masonry-x'), card.css.get('--masonry-y')], ['0px', '0px']);
    // Far right: its right edge stops at the board's.
    handle.emit('pointermove', { pointerId: 1, clientX: 900, clientY: 300 });
    assert.deepEqual([card.css.get('--masonry-x'), card.css.get('--masonry-y')], ['400px', '290px']);
    handle.emit('pointerup', { pointerId: 1, clientX: 900, clientY: 300 });
});

test('a carried card goes on over the cards shown now when another window takes one off the board', () => {
    const env = installDomDouble();
    const names = ['requestAnimationFrame', 'cancelAnimationFrame', 'ResizeObserver', 'MutationObserver'];
    const before = Object.fromEntries(names.map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
    const frames = new Map();
    let frameId = 0;
    globalThis.requestAnimationFrame = (callback) => { frames.set(++frameId, callback); return frameId; };
    globalThis.cancelAnimationFrame = (id) => frames.delete(id);
    globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
    globalThis.MutationObserver = class { observe() {} disconnect() {} };
    const frame = () => { for (const [id, callback] of [...frames.entries()]) { frames.delete(id); callback(); } };
    try {
        const { doc, el } = env;
        const handles = {};
        const cards = Object.fromEntries(['a', 'b', 'c'].map((key) => {
            handles[key] = el('button', { attrs: { 'data-widget-reorder-handle': '' } });
            const card = el('article', { className: 'widgets-card', dataset: { widgetKey: key } }, handles[key]);
            card.rect = { left: 0, top: 0, right: 500, bottom: 100, width: 500, height: 100 };
            card.toggleAttribute = (name, on) => { if (on) card.setAttribute(name, ''); else card.removeAttribute(name); };
            return [key, card];
        }));
        const list = el('div', {}, ...Object.values(cards));
        list.clientWidth = 500;
        list.rect = { left: 0, top: 0, right: 500, bottom: 328, width: 500, height: 328 };
        const scroller = el('div', { className: 'widgets-scroll' }, list);
        scroller.scrollTop = 0;
        scroller.rect = { left: 0, top: 0, right: 500, bottom: 1000, width: 500, height: 1000 };
        doc.body.append(scroller);
        let keys = ['a', 'b', 'c'];
        const prefs = { widget_size: {} };
        const saves = [];
        const widths = createWidgetWidths(list, { tabs: () => keys.map((key) => ({ key })), prefs: () => prefs,
            adopt: (sizes) => { prefs.widget_size = sizes; }, save: () => Promise.resolve() });
        widths.relayout();
        frame();
        bindWidgetCardReorder(list, () => keys, (order) => saves.push(order), widths);
        const press = (type, y) => handles.c.emit(type, { button: 0, buttons: type === 'pointerup' ? 0 : 1, pointerId: 1,
            pointerType: 'mouse', clientX: 10, clientY: y });
        press('pointerdown', 238);
        press('pointermove', 244);
        frame();
        assert.equal(cards.c.hasAttribute('data-widget-lifted'), true);
        // Another window takes A off the board while C is carried here; the page binds the board again, as after a sync.
        keys = ['b', 'c'];
        cards.a.remove();
        bindWidgetCardReorder(list, () => keys, (order) => saves.push(order), widths);
        widths.relayout();
        frame();
        // Held over the second of the two places left, C stays there: the preview and the drop agree, nothing is saved.
        const second = widths.laid().plan.placements[1].top;
        press('pointermove', second + 10);
        frame();
        assert.deepEqual(widths.laid().items.map((card) => card.dataset.widgetKey), ['b', 'c']);
        press('pointerup', second + 10);
        frame();
        assert.deepEqual(saves, [], 'no order with a card that is gone');
        // Carried again to the first place, the order saved holds only the cards still there.
        press('pointerdown', second + 10);
        press('pointermove', second + 16);
        frame();
        press('pointermove', widths.laid().plan.placements[0].top + 10);
        frame();
        press('pointerup', widths.laid().plan.placements[0].top + 10);
        frame();
        assert.deepEqual(saves, [['c', 'b']]);
        cancelWidgetCardCarry(list);
        widths.dispose();
    } finally {
        for (const name of names) {
            if (before[name]) Object.defineProperty(globalThis, name, before[name]);
            else delete globalThis[name];
        }
        env.restore();
    }
});

// --- the one card a finger adjusts --------------------------------------------
// Under a coarse pointer no card has a foot until the menu's Resize picks one (style.css keys the foot and the grip off
// `data-widget-adjusting`). The board double gains what the lifecycle reads: the cards' grip, menu button and Done,
// focus, a window, and the primary input theme.js paints.
function fingerBoard(pointer = 'coarse') {
    const api = board(['demo:a', 'demo:b'], { width: 500, heightInfo: () => ({ content: 400, natural: 400 }) });
    const doc = api.list.ownerDocument;
    doc.documentElement = { dataset: { pointer } };
    doc.defaultView = listener();
    doc.body = { tagName: 'BODY' };
    doc.activeElement = doc.body;
    const focusable = (name, card) => ({
        name, card, tagName: 'BUTTON', scrolled: [], focusOptions: null,
        focus(options) { this.focusOptions = options; doc.activeElement = this; },
        scrollIntoView(options) { this.scrolled.push(options); },
        closest(selector) { return selector === '[data-widget-adjust-done]' && name === 'done' ? this : null; },
    });
    for (const card of api.cards) {
        Object.assign(card.heightHandle, focusable('grip', card));
        card.trigger = focusable('trigger', card);
        card.done = focusable('done', card);
        card.inner = focusable('content', card);
        card.frame = { tagName: 'IFRAME', card };
        const parts = { '[data-widget-height-handle]': card.heightHandle, '[data-widget-menu-trigger]': card.trigger };
        card.querySelector = (selector) => parts[selector] || null;
        card.contains = (node) => node === card || node?.card === card;
    }
    api.list.contains = (node) => api.cards.some((card) => card.contains(node));
    const tap = (target) => doc.fire('click', { target });
    const escape = (extra = {}) => {
        const event = { key: 'Escape', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra };
        doc.fire('keydown', event);
        return event;
    };
    const adjusting = () => api.cards.filter((card) => card.hasAttribute('data-widget-adjusting')).map((card) => card.dataset.widgetKey);
    const listeners = () => [...doc.handlers.values(), ...doc.defaultView.handlers.values()].reduce((n, fns) => n + fns.length, 0);
    return { ...api, doc, tap, escape, adjusting, listeners };
}

test('a mouse never adjusts a card: Resize is not offered and asking for it changes nothing', () => {
    const mouse = fingerBoard('fine');
    assert.equal(mouse.widths.canAdjust(), false);
    mouse.widths.adjust('demo:a');
    assert.deepEqual(mouse.adjusting(), []);
    assert.equal(mouse.widths.adjusting(), '');
    assert.equal(mouse.listeners(), 0);
});

test('under a finger Resize gives one card its foot, its grip focused and in view; another Resize moves it', async () => {
    const finger = fingerBoard();
    assert.equal(finger.widths.canAdjust(), true);
    assert.deepEqual(finger.adjusting(), [], 'no card has a foot by default');
    finger.widths.adjust('demo:a');
    const [a, b] = finger.cards;
    assert.deepEqual(finger.adjusting(), ['demo:a']);
    assert.equal(finger.doc.activeElement, a.heightHandle, 'the grip takes the focus');
    assert.deepEqual(a.heightHandle.focusOptions, { preventScroll: true });
    assert.deepEqual(a.heightHandle.scrolled, [{ block: 'nearest', inline: 'nearest' }], 'and is brought into view');
    // Taps inside the card (its content, its Show all, its own menu while open) keep it; a scroll is no tap.
    finger.tap(a.inner);
    a.trigger.expanded = true;
    a.querySelector = ((query) => (selector) => (selector === '[data-widget-menu-trigger][aria-expanded="true"]' ? a.trigger : query(selector)))(a.querySelector);
    finger.tap({ closest: (selector) => (selector === '[role="menu"]' ? {} : null) });
    assert.deepEqual(finger.adjusting(), ['demo:a']);
    finger.widths.adjust('demo:b');
    assert.deepEqual(finger.adjusting(), ['demo:b'], 'one card at a time');
    assert.equal(finger.doc.activeElement, b.heightHandle);
    // Done ends it and gives the focus back to the card's menu button without scrolling the board.
    finger.tap(b.done);
    assert.deepEqual(finger.adjusting(), []);
    assert.equal(finger.doc.activeElement, b.trigger);
    assert.deepEqual(b.trigger.focusOptions, { preventScroll: true });
    assert.equal(finger.listeners(), 0, 'nothing listens once no card is adjusted');
    await settle();
    assert.equal(finger.saves.length, 0, 'adjusting writes nothing');
});

test('a tap outside, a tap in another card\'s frame, Escape, leaving and a mouse put the foot away', () => {
    const finger = fingerBoard();
    const [a, b] = finger.cards;
    finger.widths.adjust('demo:a');
    finger.tap(b.inner);
    assert.deepEqual(finger.adjusting(), [], 'a tap on another card');
    assert.notEqual(finger.doc.activeElement, a.trigger, 'a tap outside moves no focus');

    finger.widths.adjust('demo:a');
    finger.doc.activeElement = a.frame;
    finger.doc.defaultView.fire('blur', {});
    assert.deepEqual(finger.adjusting(), ['demo:a'], 'a tap in its own frame keeps it');
    finger.doc.activeElement = b.frame;
    finger.doc.defaultView.fire('blur', {});
    assert.deepEqual(finger.adjusting(), [], 'a tap in another card\'s frame');

    finger.widths.adjust('demo:a');
    finger.doc.activeElement = finger.doc.body;
    assert.equal(finger.escape().defaultPrevented, false);
    assert.deepEqual(finger.adjusting(), ['demo:a'], 'an Escape while the focus is off the board');
    a.heightHandle.focus();
    assert.equal(finger.escape({ defaultPrevented: true }).defaultPrevented, true);
    assert.deepEqual(finger.adjusting(), ['demo:a'], 'an Escape a menu or a drag took first');
    assert.equal(finger.escape().defaultPrevented, true);
    assert.deepEqual(finger.adjusting(), []);
    assert.equal(finger.doc.activeElement, a.trigger, 'Escape from the grip returns to the menu button');

    finger.widths.adjust('demo:a');
    finger.doc.defaultView.fire('ouro:layout-changed', { detail: { layout: 'stack', pointer: 'coarse' } });
    assert.deepEqual(finger.adjusting(), ['demo:a'], 'a rotation alone keeps it');
    finger.doc.documentElement.dataset.pointer = 'fine';
    finger.doc.defaultView.fire('ouro:layout-changed', { detail: { layout: 'stack', pointer: 'fine' } });
    assert.deepEqual(finger.adjusting(), [], 'a mouse has the edges again');
    finger.doc.documentElement.dataset.pointer = 'coarse';

    finger.widths.adjust('demo:a');
    finger.widths.cancel();
    assert.deepEqual(finger.adjusting(), [], 'leaving Widgets');
    finger.widths.adjust('demo:b');
    finger.widths.dispose();
    assert.deepEqual(finger.adjusting(), [], 'the page going away');
    assert.equal(finger.listeners(), 0);
});

test('Resize selects the live replacement while its previous card retires', async () => {
    const finger = fingerBoard();
    const [retiring, live] = finger.cards;
    retiring.toggleAttribute('data-widget-removed', true);
    live.dataset.widgetKey = retiring.dataset.widgetKey;
    finger.widths.relayout();
    finger.widths.adjust('demo:a');
    assert.equal(retiring.hasAttribute('data-widget-adjusting'), false);
    assert.equal(live.hasAttribute('data-widget-adjusting'), true);
    assert.equal(finger.doc.activeElement, live.heightHandle);
    live.toggleAttribute('data-widget-removed', true);
    finger.widths.relayout();
    finger.widths.adjust('demo:a');
    assert.deepEqual(finger.adjusting(), [], 'retired nodes alone cannot enter adjustment');
    await settle();
    assert.equal(finger.saves.length, 0);
    finger.widths.dispose();
});

test('a card adjusted leaves the board: its foot goes; a drag on its grip is cancelled first, saving nothing', async () => {
    const finger = fingerBoard();
    const [a] = finger.cards;
    finger.widths.adjust('demo:a');
    const grip = a.heightHandle;
    grip.fire('pointerdown', { button: 0, pointerId: 4, clientX: 0, clientY: 400, currentTarget: grip, preventDefault() {} });
    grip.fire('pointermove', { pointerId: 4, clientX: 0, clientY: 600 });
    assert.equal(a.classList.contains('resizing'), true);
    finger.tap(finger.cards[1].inner);
    assert.equal(a.classList.contains('resizing'), false, 'the drag ended with the foot');
    grip.fire('pointerup', { pointerId: 4 });
    await settle();
    assert.equal(finger.saves.length, 0);

    finger.widths.adjust('demo:a');
    a.isConnected = false;
    finger.widths.relayout();
    assert.equal(finger.widths.adjusting(), '');
    assert.equal(a.hasAttribute('data-widget-adjusting'), false);
    assert.equal(finger.listeners(), 0);
});
