// The message field's sizing contract (chat_input_sizing.js): one selected mode for CSS
// and script, chosen by what the engine's content sizing does with a placeholder; content
// sizing without height writes; and the measured fallback's cap, box conversion,
// hidden/reveal, re-measure triggers (unrounded width, fonts, placeholder), scroll rules
// and disposal. Real engines, wrapping and scrollbars: tests/test_chat_input_sizing_browser.py.
import test from 'node:test';
import assert from 'node:assert/strict';

import { attachChatInputSizing, chatInputSizingMode } from '../modules/chat_input_sizing.js';

const LINE = 23.2;
const CHAR = 8;
const SCROLLBAR = 6;

/* A textarea whose text wraps at its width, whose scrollHeight is a rounded integer and
   whose capped (overflow: auto) state takes a classic scrollbar's width. */
class Field {
    constructor({ width = 400, style = {} } = {}) {
        this.width = width;
        this.charWidth = CHAR;  // the face's advance; a wider face wraps sooner
        this.computed = { maxHeight: '140px', minHeight: '32px', boxSizing: 'border-box', paddingTop: '5px',
            paddingBottom: '5px', borderTopWidth: '0px', borderBottomWidth: '0px', ...style };
        // The used border-box width, unrounded; `auto` when the field is not rendered.
        Object.defineProperty(this.computed, 'width', { get: () => (this.width ? `${this.width}px` : 'auto') });
        this.dataset = {};
        this.value = '';
        this.selectionStart = 0;
        this.selectionEnd = 0;
        this._scrollTop = 0;
        this.props = new Map();
        this.writes = [];
        this.style = {
            setProperty: (name, value) => { this.props.set(name, value); this.writes.push([name, value]); },
            removeProperty: (name) => { this.props.delete(name); },
            getPropertyValue: (name) => this.props.get(name) || '',
        };
    }
    type(value, caret = value.length) { this.value = value; this.selectionStart = this.selectionEnd = caret; }
    get pad() { return parseFloat(this.computed.paddingTop) + parseFloat(this.computed.paddingBottom); }
    get frame() { return parseFloat(this.computed.borderTopWidth) + parseFloat(this.computed.borderBottomWidth); }
    // A whole-pixel clientWidth over a layout width that may be fractional.
    get innerWidth() { return this.width ? this.width - (this.dataset.sizing === 'capped' ? SCROLLBAR : 0) : 0; }
    get clientWidth() { return Math.round(this.innerWidth); }
    get rows() {
        const perLine = Math.max(1, Math.floor((this.innerWidth - 8) / this.charWidth));
        return this.value.split('\n').reduce((sum, line) => sum + Math.max(1, Math.ceil(line.length / perLine)), 0);
    }
    get borderBox() {
        const set = this.props.get('--chat-input-height');
        const contentBox = this.computed.boxSizing !== 'border-box';
        const auto = LINE + this.pad + this.frame;  // rows="1"
        const declared = set ? parseFloat(set) + (contentBox ? this.pad : 0) : auto;
        const max = parseFloat(this.computed.maxHeight);
        return Math.min(Number.isNaN(max) ? Infinity : max + (contentBox ? this.pad : 0), Math.max(32, declared));
    }
    get clientHeight() { return this.width ? Math.round(this.borderBox - this.frame) : 0; }
    get scrollHeight() { return this.width ? Math.round(Math.max(this.borderBox - this.frame, this.rows * LINE + this.pad)) : 0; }
    get scrollTop() { return this._scrollTop; }
    set scrollTop(value) { this._scrollTop = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)); }
}

/* A probe textarea laid out like an engine's content sizing: `a a` is two 10px lines
   with 2px padding at either edge. `placeholder` says what that engine does with a
   placeholder-only field: `text` sizes it like the same text; `webkit` (26) to its lines
   alone, without padding or max-height. */
function probeDocument(placeholder = 'text') {
    const doc = { probes: [], body: { children: new Set(), append: (el) => doc.body.children.add(el) } };
    doc.createElement = (tag) => {
        const probe = { tag, value: '', placeholder: '', style: { cssText: '', maxHeight: '' },
            remove: () => doc.body.children.delete(probe) };
        Object.defineProperty(probe, 'offsetHeight', { get: () => {
            assert.ok(doc.body.children.has(probe), 'measured only while in the document');
            const lines = 2 * 10;
            const max = parseFloat(probe.style.maxHeight) || Infinity;
            if (!probe.value && placeholder === 'webkit') return lines;
            return Math.min(max, lines + 4);
        } });
        doc.probes.push(probe);
        return probe;
    };
    return doc;
}

function installEngine({ fieldSizing = false, placeholder = 'text', fontsStatus = 'loaded' } = {}) {
    const prior = { CSS: globalThis.CSS, ResizeObserver: globalThis.ResizeObserver, getComputedStyle: globalThis.getComputedStyle,
        requestAnimationFrame: globalThis.requestAnimationFrame, cancelAnimationFrame: globalThis.cancelAnimationFrame,
        document: globalThis.document, MutationObserver: globalThis.MutationObserver };
    const fontListeners = new Set();
    globalThis.document = Object.assign(probeDocument(placeholder), { fonts: {
        status: fontsStatus,
        ready: Promise.resolve(),
        addEventListener: (type, fn) => type === 'loading' && fontListeners.add(fn),
        removeEventListener: (type, fn) => type === 'loading' && fontListeners.delete(fn),
    } });
    const mutations = [];
    globalThis.MutationObserver = class {
        constructor(callback) { this.callback = callback; this.targets = []; mutations.push(this); }
        observe(target, options) { this.targets.push([target, options]); }
        disconnect() { this.targets = []; this.disconnected = true; }
    };
    const observers = [];
    const frames = new Map();
    let nextFrame = 1;
    globalThis.CSS = { supports: (property, value) => fieldSizing && property === 'field-sizing' && value === 'content' };
    globalThis.ResizeObserver = class {
        constructor(callback) { this.callback = callback; this.targets = []; observers.push(this); }
        observe(target) { this.targets.push(target); }
        disconnect() { this.targets = []; this.disconnected = true; }
    };
    globalThis.getComputedStyle = (el) => el.computed;
    globalThis.requestAnimationFrame = (fn) => { frames.set(nextFrame, fn); return nextFrame++; };
    globalThis.cancelAnimationFrame = (id) => frames.delete(id);
    return {
        observers,
        mutations,
        // The field's placeholder attribute is rewritten (the language dictionary arriving).
        rewritePlaceholder: () => mutations.forEach((o) => o.targets.some(([, options]) =>
            options.attributeFilter?.includes('placeholder')) && o.callback([{ attributeName: 'placeholder' }])),
        report: () => observers.forEach((o) => o.targets.length && o.callback([])),
        // A loading period: `loading`, then `ready` settles.
        fontsLoaded: async () => { fontListeners.forEach((fn) => fn()); await globalThis.document.fonts.ready; await null; },
        fontListeners,
        flush: () => { const queued = [...frames.values()]; frames.clear(); queued.forEach((fn) => fn()); },
        pending: () => frames.size,
        restore: () => Object.assign(globalThis, prior),
    };
}

const lines = (count, word = 'line') => Array.from({ length: count }, (_, n) => `${word} ${n + 1}`).join('\n');

test('content sizing is chosen only where it sizes a placeholder like the same text', () => {
    const supports = { supports: () => true };
    assert.equal(chatInputSizingMode(supports, probeDocument('text')), 'content');
    // WebKit 26: a wrapped placeholder is sized without its padding and past max-height.
    assert.equal(chatInputSizingMode(supports, probeDocument('webkit')), 'measured');
    const unsupported = probeDocument();
    assert.equal(chatInputSizingMode({ supports: () => false }, unsupported), 'measured');
    assert.equal(unsupported.probes.length, 0, 'no probe without field-sizing');
    assert.equal(chatInputSizingMode(undefined, probeDocument()), 'measured');
    assert.equal(chatInputSizingMode(supports, {}), 'measured', 'no body to lay a probe out in');
    const doc = probeDocument();
    chatInputSizingMode(supports, doc);
    chatInputSizingMode(supports, doc);
    assert.equal(doc.probes.length, 1, 'one probe per document: an engine fact');
    assert.equal(doc.body.children.size, 0, 'the probe is removed');
});

test('the selected mode is what the stylesheet keys on', (t) => {
    for (const fieldSizing of [true, false]) {
        const engine = installEngine({ fieldSizing });
        t.after(engine.restore);
        const field = new Field();
        const sizing = attachChatInputSizing(field);
        assert.equal(sizing.mode, fieldSizing ? 'content' : 'measured');
        assert.equal(field.dataset.sizing, fieldSizing ? 'content' : 'fit');
        engine.restore();
    }
});

test('content sizing writes no height and observes nothing; edits keep the right line in view', (t) => {
    const engine = installEngine({ fieldSizing: true });
    t.after(engine.restore);
    const field = new Field();
    // The engine sizes the field; this model stands in for it at the cap.
    field.props.set('--chat-input-height', '140px');
    const sizing = attachChatInputSizing(field);
    assert.equal(engine.observers.length, 0);
    field.type(lines(9));
    sizing.sync();
    assert.equal(field.scrollTop, field.scrollHeight - field.clientHeight, 'a caret at the end keeps the last line in view');
    field.scrollTop = 0;
    field.type(`Z${lines(9)}`, 1);
    sizing.sync();
    assert.equal(field.scrollTop, 0, 'an edit at the start does not snap to the end');
    field.scrollTop = 20;
    field.type('');
    sizing.sync();
    assert.equal(field.scrollTop, 0, 'an empty field shows its placeholder from the top');
    assert.deepEqual(field.writes, [], 'no height or custom-property writes');
    assert.equal(field.dataset.sizing, 'content');
});

test('the measured fallback fits its text below the computed cap and scrolls at it', (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    const field = new Field();
    const sizing = attachChatInputSizing(field);
    field.type(lines(5));
    sizing.sync();
    // Five rows are 126px: under the one CSS cap of 140px (the former script cap was 120px).
    assert.equal(field.style.getPropertyValue('--chat-input-height'), '126px');
    assert.equal(field.dataset.sizing, 'fit');
    assert.equal(field.scrollHeight - field.clientHeight, 0);
    field.type(lines(9));
    sizing.sync();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), '140px');
    assert.equal(field.dataset.sizing, 'capped');
    assert.ok(field.scrollTop > 0 && field.scrollTop === field.scrollHeight - field.clientHeight);
    field.type('short');
    sizing.sync();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), '33px');
    assert.equal(field.dataset.sizing, 'fit');
    assert.equal(field.scrollTop, 0);
    // The cap is whatever the stylesheet computes, not a number in script.
    field.computed.maxHeight = '200px';
    field.type(lines(7));
    sizing.sync();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), '172px');
    assert.equal(field.dataset.sizing, 'fit');
    field.computed.maxHeight = 'none';
    field.type(lines(30));
    sizing.sync();
    assert.equal(field.dataset.sizing, 'fit', 'no max-height never caps');
});

test('the row around the measured field keeps its height while the field is measured, then gets its own back', (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    // Where the composer is in flow (the phone's keyboard is up), the field's one-row instant would lengthen the feed
    // and WebKit clamps the feed's scroll to it: the newest message stays under the field (owner 20.10).
    const props = new Map([['min-height', '20px']]);
    const row = { offsetHeight: 57, style: { setProperty: (n, v) => props.set(n, v), removeProperty: (n) => props.delete(n),
        getPropertyValue: (n) => props.get(n) || '' } };
    const field = new Field();
    field.parentElement = row;
    const seen = [];
    const measured = Object.getOwnPropertyDescriptor(Field.prototype, 'scrollHeight').get;
    Object.defineProperty(field, 'scrollHeight', { get() { seen.push(props.get('min-height')); return measured.call(this); } });
    const sizing = attachChatInputSizing(field);
    field.type(lines(3));
    sizing.sync();
    assert.equal(seen[0], '57px', 'the measuring read happens while the row is held at its height');
    assert.equal(props.get('min-height'), '20px', 'its own min-height is back');
    props.delete('min-height');
    field.type(lines(4));
    sizing.sync();
    assert.equal(props.has('min-height'), false, 'a row without one gets none');
    // A field with no row (a detached measure) still measures.
    const alone = new Field();
    attachChatInputSizing(alone).sync();
    assert.equal(alone.dataset.sizing, 'fit');
});

test('the measured height converts to the declared box', (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    const contentBox = new Field({ style: { boxSizing: 'content-box' } });
    attachChatInputSizing(contentBox).sync();
    assert.equal(contentBox.style.getPropertyValue('--chat-input-height'), '23px', 'padding is outside a content box');
    const bordered = new Field({ style: { borderTopWidth: '1px', borderBottomWidth: '1px' } });
    const sizing = attachChatInputSizing(bordered);
    bordered.type(lines(2));
    sizing.sync();
    assert.equal(bordered.style.getPropertyValue('--chat-input-height'), `${Math.round(2 * LINE + 10) + 2}px`);
});

test('a hidden field pins nothing and its reveal measures; only a new width re-measures', (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    const field = new Field({ width: 0 });
    const sizing = attachChatInputSizing(field);
    field.type(lines(3, 'a longer line of draft text'));
    sizing.sync();
    assert.deepEqual(field.writes, [], 'a hidden draft is not collapsed to a zero-height box');
    field.width = 400;  // the panel is shown
    engine.report();
    assert.deepEqual(field.writes, [], 'the write waits for the next frame');
    engine.report();
    assert.equal(engine.pending(), 1, 'reports coalesce into one frame');
    engine.flush();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), '80px');
    const written = field.writes.length;
    engine.report();  // the height just written reports back at the same width
    engine.flush();
    assert.equal(field.writes.length, written, 'a height-only report is ignored');
    field.width = 120;  // e.g. the busy Send label narrows the field
    engine.report();
    engine.flush();
    assert.ok(parseFloat(field.style.getPropertyValue('--chat-input-height')) > 80);
});

test('a font that finishes loading re-measures the same width once, in a frame', async (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    const field = new Field();
    const sizing = attachChatInputSizing(field);
    field.type('a line that fits the proportional face');
    sizing.sync();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), '33px');
    engine.flush();
    const written = field.writes.length;
    field.charWidth = 3 * CHAR;  // the loaded face is wider: the same text wraps to three rows
    await engine.fontsLoaded();
    await engine.fontsLoaded();
    assert.equal(engine.pending(), 1, 'one frame for any number of loading periods before it');
    engine.flush();
    assert.equal(field.writes.length, written + 1);
    assert.equal(field.style.getPropertyValue('--chat-input-height'), `${Math.round(3 * LINE + 10)}px`);
    sizing.dispose();
    assert.equal(engine.fontListeners.size, 0, 'dispose removes the font listener');
});

test('a loading period that settles after dispose schedules nothing', async (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    const field = new Field();
    const sizing = attachChatInputSizing(field);
    let settle;
    globalThis.document.fonts.ready = new Promise((resolve) => { settle = resolve; });
    engine.fontListeners.forEach((fn) => fn());
    sizing.dispose();
    settle();
    await null;
    await null;
    assert.equal(engine.pending(), 0);
    assert.deepEqual(field.writes, []);
});

test('content sizing adds no font listener or observer', (t) => {
    const engine = installEngine({ fieldSizing: true, fontsStatus: 'loading' });
    t.after(engine.restore);
    attachChatInputSizing(new Field());
    assert.equal(engine.fontListeners.size, 0);
    assert.equal(engine.observers.length, 0);
    assert.equal(engine.mutations.length, 0);
});

test('an engine whose content sizing cuts a placeholder measures', (t) => {
    const engine = installEngine({ fieldSizing: true, placeholder: 'webkit' });
    t.after(engine.restore);
    const field = new Field();
    const sizing = attachChatInputSizing(field);
    assert.equal(sizing.mode, 'measured');
    assert.equal(field.dataset.sizing, 'fit');
    field.type(lines(3));
    sizing.sync();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), `${Math.round(3 * LINE + 10)}px`);
});

test('a font loading period already running when the field attaches re-measures as it settles', async (t) => {
    const engine = installEngine({ fontsStatus: 'loading' });
    t.after(engine.restore);
    let settle;
    globalThis.document.fonts.ready = new Promise((resolve) => { settle = resolve; });
    const field = new Field();
    const sizing = attachChatInputSizing(field);
    field.type('a line that fits the proportional face');
    sizing.sync();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), '33px');
    field.charWidth = 3 * CHAR;  // the face that was loading is wider
    settle();
    await null;
    await null;
    assert.equal(engine.pending(), 1);
    engine.flush();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), `${Math.round(3 * LINE + 10)}px`);
});

test('a placeholder rewritten in place re-measures in a frame; dispose stops it', (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    const field = new Field();
    const sizing = attachChatInputSizing(field);
    sizing.sync();
    engine.flush();
    const written = field.writes.length;
    engine.rewritePlaceholder();
    assert.equal(engine.pending(), 1, 'the write waits for the next frame');
    engine.flush();
    assert.equal(field.writes.length, written + 1, 'the same width, measured again');
    sizing.dispose();
    assert.ok(engine.mutations[0].disconnected);
    engine.rewritePlaceholder();
    assert.equal(engine.pending(), 0);
});

test('a sub-pixel width change that keeps clientWidth still rewraps', (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    const field = new Field({ width: 400.4 });
    const sizing = attachChatInputSizing(field);
    field.type('x'.repeat(49));  // 49 advances fit 392.4px of text width, not 391.8px
    sizing.sync();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), '33px');
    field.width = 399.8;  // e.g. a window narrowed by one device pixel at 125%
    assert.equal(field.clientWidth, 400, 'the rounded width did not change');
    engine.report();
    engine.flush();
    assert.equal(field.style.getPropertyValue('--chat-input-height'), `${Math.round(2 * LINE + 10)}px`);
});

test('a width change keeps a capped selection and its scroll place', (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    const field = new Field();
    const sizing = attachChatInputSizing(field);
    field.type(lines(12));
    sizing.sync();
    field.selectionStart = 2;
    field.selectionEnd = 6;
    field.scrollTop = 0;
    field.width = 300;
    engine.report();
    engine.flush();
    assert.equal(field.dataset.sizing, 'capped');
    assert.deepEqual([field.selectionStart, field.selectionEnd, field.scrollTop], [2, 6, 0]);
    // An edit with the caret mid-text keeps the place the engine scrolled to.
    field.scrollTop = 40;
    field.type(`x${lines(12)}`, 3);
    sizing.sync();
    assert.equal(field.scrollTop, 40);
});

test('dispose disconnects, cancels a queued frame and makes later calls inert', (t) => {
    const engine = installEngine();
    t.after(engine.restore);
    const field = new Field();
    const sizing = attachChatInputSizing(field);
    field.width = 300;
    engine.report();
    assert.equal(engine.pending(), 1);
    sizing.dispose();
    assert.equal(engine.pending(), 0);
    assert.ok(engine.observers[0].disconnected);
    field.type(lines(4));
    sizing.sync();
    engine.report();
    engine.flush();
    assert.deepEqual(field.writes, []);
    sizing.dispose();
});
