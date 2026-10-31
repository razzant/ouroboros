import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

/* The macOS shell's document script (ouroboros/launcher_window_chrome.py, DOCUMENT_SCRIPT) is the
   window's, not the page's: it ships with the launcher, so the test reads it from there and fills
   its two placeholders exactly as the module does. */
const leaf = fs.readFileSync(new URL('../../ouroboros/launcher_window_chrome.py', import.meta.url), 'utf8');
const constant = (name) => leaf.match(new RegExp(`^${name} = "([^"]+)"$`, 'm'))[1];
const HANDLER = constant('HANDLER');
const INTEGRATED = constant('INTEGRATED');
const source = leaf.match(/^DOCUMENT_SCRIPT = r"""([\s\S]*?)""" % \{/m)[1]
    .replaceAll('%(handler)s', HANDLER).replaceAll('%(integrated)s', INTEGRATED);

class FakeElement {
    constructor({ matches = [], cursor = 'auto' } = {}) {
        this.matches = matches;
        this.cursor = cursor;
    }

    closest(selector) {
        return this.matches.some((tag) => selector.split(',').map((part) => part.trim()).includes(tag)) ? this : null;
    }
}

/* A document whose parser the test drives: head nodes arrive, then the body. */
function boot({ port = true, origin = 1_791_100_000_000.25, painted = { themeChoice: 'system', theme: 'dark' }, lateRoot = false } = {}) {
    const messages = [];
    const listeners = new Map();
    const observers = [];
    const properties = new Map();
    let meta = null;
    let now = 0;
    const root = {
        dataset: { ...painted },
        style: {
            setProperty: (name, value) => properties.set(name, value),
            removeProperty: (name) => properties.delete(name),
        },
    };
    const document = {
        documentElement: lateRoot ? null : root,
        body: null,
        readyState: 'loading',
        querySelector: (selector) => (selector === 'meta[name="ouroboros-window-chrome"]' ? meta : null),
        addEventListener: (type, fn) => listeners.set(`document:${type}`, fn),
    };
    class MutationObserver {
        constructor(callback) { this.callback = callback; this.live = false; }
        observe(target, options) { this.options = options; this.live = true; observers.push(this); }
        disconnect() { this.live = false; }
    }
    const window = {
        performance: { timeOrigin: origin, now: () => now },
        addEventListener: (type, fn) => listeners.set(type, fn),
        ...(port ? { webkit: { messageHandlers: { [HANDLER]: { postMessage: (message) => messages.push({ ...message }) } } } } : {}),
    };
    const context = { window, document, MutationObserver, Element: FakeElement, getComputedStyle: (el) => ({ cursor: el.cursor }) };
    vm.runInNewContext(source, context);
    const mutate = (kind) => observers.filter((o) => o.live && (kind === 'attributes' ? o.options.attributes : o.options.childList))
        .forEach((o) => o.callback([]));
    return {
        messages, root, properties, window,
        parseRoot() { document.documentElement = root; mutate('childList'); },
        parseMeta(content = INTEGRATED) { meta = { content }; mutate('childList'); },
        parseBody() { document.body = {}; mutate('childList'); },
        contentLoaded() { document.readyState = 'interactive'; listeners.get('document:DOMContentLoaded')?.(); },
        paint(dataset) { Object.assign(root.dataset, dataset); mutate('attributes'); },
        push(state) { window.__ouroWindowChrome(state); },
        press(init) {
            const event = { button: 0, detail: 1, clientY: 20, target: new FakeElement(), defaultPrevented: false, ...init };
            event.prevented = false;
            event.preventDefault = () => { event.prevented = true; };
            listeners.get('mousedown')(event);
            return event;
        },
        restoreFromCache(at) { now = at; listeners.get('pageshow')({ persisted: true }); },
    };
}

test('without the shell its script does nothing (a browser never runs it, an old page world has no port)', () => {
    const page = boot({ port: false });
    assert.equal(page.window.__ouroWindowChrome, undefined);
    assert.deepEqual(page.messages, []);
});

test('a page declares once while parsing, with the palette it painted', () => {
    const page = boot();
    assert.deepEqual(page.messages, [], 'nothing is decided before the head is read');
    page.parseMeta();
    page.parseBody();
    page.contentLoaded();
    assert.deepEqual(page.messages, [{ doc: 1_791_100_000_000.25, type: 'declare', chrome: INTEGRATED, choice: 'system', theme: 'dark' }]);
});

test('a document whose element the parser creates after the shell script still declares (engines differ)', () => {
    const page = boot({ lateRoot: true });
    page.parseRoot();
    page.parseMeta();
    page.paint({ themeChoice: 'dark', theme: 'dark' });
    page.push({ doc: 1_791_100_000_000.25, top: 52, start: 73 });
    assert.deepEqual(page.messages.map((m) => [m.type, m.choice]), [['declare', 'system'], ['appearance', 'dark']]);
    assert.equal(page.root.dataset.windowChrome, INTEGRATED);
});

test('a page without the declaration (older, custom or replaced UI) asks for the ordinary bar when its body starts', () => {
    const page = boot({ painted: {} });
    page.parseBody();
    page.contentLoaded();
    assert.deepEqual(page.messages, [{ doc: 1_791_100_000_000.25, type: 'declare', chrome: '', choice: '', theme: '' }]);
    const other = boot();
    other.parseMeta('ordinary');
    assert.equal(other.messages[0].chrome, '');
});

test('only the shell\'s measured state for this document is written, and zero removes it', () => {
    const page = boot();
    page.parseMeta();
    page.push({ doc: 1_791_100_000_000.25 - 1, top: 52, start: 73 });  // meant for the previous document
    assert.equal(page.root.dataset.windowChrome, undefined);
    assert.equal(page.properties.size, 0);
    page.push({ doc: 1_791_100_000_000.25, top: 52, start: 73 });
    assert.equal(page.root.dataset.windowChrome, INTEGRATED);
    assert.deepEqual(Object.fromEntries(page.properties), { '--window-chrome-top': '52px', '--window-chrome-start': '73px' });
    page.push({ doc: 1_791_100_000_000.25, top: 0, start: 0 });  // full screen or the ordinary bar
    assert.equal(page.root.dataset.windowChrome, undefined);
    assert.equal(page.properties.size, 0);
});

test('a plain left press in the band, off any control, drags; a second press is the title bar double-click', () => {
    const page = boot();
    page.parseMeta();
    assert.equal(page.press().prevented, false, 'no band, no drag: an ordinary bar drags itself');
    page.push({ doc: 1_791_100_000_000.25, top: 52, start: 73 });
    const drag = page.press();  // a title's text computes `auto`: it drags like a title
    assert.equal(drag.prevented, true, 'the press neither selects text nor moves focus');
    page.press({ detail: 2 });
    assert.deepEqual(page.messages.slice(1).map((m) => m.type), ['drag', 'double-click']);

    const ignored = [
        { clientY: 52 },  // below the band
        { button: 2 },
        { metaKey: true }, { ctrlKey: true }, { altKey: true }, { shiftKey: true },
        { defaultPrevented: true },  // the page took it
        { target: new FakeElement({ matches: ['button'] }) },
        { target: new FakeElement({ matches: ['[role="button"]'] }) },
        { target: new FakeElement({ matches: ['input'] }) },
        { target: new FakeElement({ cursor: 'pointer' }) },  // a clickable element without a role
        { target: new FakeElement({ cursor: 'col-resize' }) },  // the page's own drag handle
        { target: new FakeElement({ cursor: 'text' }) },  // text the page declares selectable (the reader's name)
        { target: null },
    ];
    for (const init of ignored) assert.equal(page.press(init).prevented, false, JSON.stringify(init));
    assert.equal(page.messages.length, 3);
});

test('the painted palette is reported as it changes, after the declaration', () => {
    const page = boot();
    page.paint({ theme: 'light' });
    assert.deepEqual(page.messages, [], 'before the declaration the shell has nothing to update');
    page.parseMeta();
    page.paint({ themeChoice: 'light', theme: 'light' });
    assert.deepEqual(page.messages.at(-1), { doc: 1_791_100_000_000.25, type: 'appearance', choice: 'light', theme: 'light' });
});

test('back from the page cache the document is the window\'s page again, under a newer name', () => {
    const page = boot();
    page.parseMeta();
    page.restoreFromCache(9_000.5);
    const renewed = 1_791_100_000_000.25 + 9_000.5;
    assert.deepEqual(page.messages.at(-1), { doc: renewed, type: 'declare', chrome: INTEGRATED, choice: 'system', theme: 'dark' });
    page.push({ doc: 1_791_100_000_000.25, top: 52, start: 73 });
    assert.equal(page.root.dataset.windowChrome, undefined, 'a state for its previous name is stale');
    page.push({ doc: renewed, top: 52, start: 73 });
    assert.equal(page.root.dataset.windowChrome, INTEGRATED);
});
