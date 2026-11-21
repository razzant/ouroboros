import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { openDesktopRoomSession } from '../modules/desktop_room_session.js';
import { createLayerStack } from '../modules/layers.js';
import { filesPageTemplate, initFiles } from '../modules/files.js';
import { setup } from './dashboard_dom_fixture.js';
import { Stub } from './helpers/settings_dom.js';

test('Files keeps location and actions in the band, with the filter first in its scrolling list', () => {
    const page = new Stub('div');
    page.innerHTML = filesPageTemplate();
    const band = page.querySelector('.files-band'), list = page.querySelector('.files-sidebar');
    for (const id of ['files-breadcrumb', 'files-new-file', 'files-new-dir', 'files-paste']) {
        assert.ok(band.contains(page.querySelector(`#${id}`)), id);
        assert.equal(list.contains(page.querySelector(`#${id}`)), false, id);
    }
    assert.ok(list.firstElementChild.contains(page.querySelector('#files-search')));
    assert.ok(list.contains(page.querySelector('#files-list')));
    assert.ok(list.classList.contains('scroll-fade-y'), 'the filter and entries fade together on their one scroller');
    assert.equal(page.querySelector('#files-list').classList.contains('scroll-fade-y'), false, 'no second mask on the inner list');
    assert.equal(band.contains(page.querySelector('#files-search')), false);
    assert.equal(page.querySelector('#files-search').getAttribute('type'), 'search', 'Escape can clear a filter before Back');
    assert.equal(page.querySelector('[data-files-back]'), null, 'a preview uses the shell slot too');
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.match(css, /\.files-layout \{[^}]*position: absolute;[^}]*inset: 0;/);
    const sidebar = css.match(/\.files-sidebar \{([^}]+)\}/)[1];
    assert.doesNotMatch(sidebar, /border-right:/, 'the scrolling mask does not erase the column divider');
    assert.match(css, /\.files-preview \{[^}]*border-left: 1px solid var\(--divider\)/);
    assert.match(css, /@container files \(max-width: 700px\) \{[^@]+?\.files-preview \{ border-left: 0; \}/,
        'a file shown alone has no column divider');
    assert.doesNotMatch(sidebar, /background:/, 'the column does not paint its own sheet');
    assert.match(css, /\.files-sidebar\[data-scroll-above\] \{ --scroll-fade-top-clear: var\(--files-band-h\); --scroll-fade-top: calc\(var\(--files-band-h\) \+ var\(--scroll-fade-edge\)\); \}/);
    // The band's location row takes the band's inset alone (a phone showed it 28px in, the rows 12px).
    assert.match(css, /\.files-band > \.files-browser-header \{ padding-inline: 0; \}/);
    assert.match(css, /\.app-page-tabs \{[^}]*padding: 0 var\(--page-header-padding-x\)/, 'the band itself keeps the page\'s inset');
    // With a return the rail keeps 14rem only while the band is wide: a tablet's column keeps its last action in it.
    assert.match(css, /html\[data-layout="docked"\] \.app-page-header\[data-lead\]\[data-return\] \{ grid-template-columns: minmax\(min\(14rem, 40%\), 1fr\) auto; \}/);
    assert.doesNotMatch(css, /minmax\(14rem, 1fr\)/, 'a fixed 14rem pushed Refresh past a 768px window\'s edge');
});

test('a file shown alone names its folder in the shell layer; beside the list it owes no level Back', () => {
    const source = readFileSync(new URL('../modules/files.js', import.meta.url), 'utf8');
    const layers = createLayerStack({ win: null, doc: null });
    let closed = 0;
    const state = { path: '.' };
    const context = vm.createContext({ layers, state, FILE_LAYER: 'files-file', previewEl: null, closeFile: () => { closed++; } });
    vm.runInContext(source.match(/    function syncLayer\(open\) \{[\s\S]+?\n    \}/)[0], context);
    context.syncLayer(true);
    assert.equal(layers.page().back.label, 'Files');
    assert.equal(layers.page().back.title, 'Files', 'the root is "Back to Files", never "Files › ."');
    layers.back(); assert.equal(closed, 1); assert.equal(layers.size(), 0);
    state.path = 'notes/work';
    context.syncLayer(true);
    assert.equal(layers.page().back.label, 'work');
    assert.equal(layers.page().back.title, 'Files › notes/work');
    context.syncLayer(false);
    assert.equal(layers.size(), 0); assert.equal(closed, 1, 'wide layout takes no Back step');
    state.path = '/';
    context.syncLayer(true);
    assert.equal(layers.page().back.label, 'Files', 'the filesystem root has a name too');
    layers.destroy();
});

test('a file shown alone gives focus to its Back, unless the caller names a control: a new file\'s name', () => {
    const source = readFileSync(new URL('../modules/files.js', import.meta.url), 'utf8');
    const node = (name) => ({ name, getClientRects: () => [1], focus() { context.document.activeElement = this; } });
    const back = node('back'), nameField = node('name'), previewEl = node('preview'), row = node('row');
    const context = vm.createContext({
        document: { activeElement: null, body: {} }, lastFocus: null, listScroll: 0, previewEl,
        layoutEl: { dataset: { filesView: 'list' }, contains: () => false }, sidebarEl: {},
        listEl: { querySelectorAll: () => [row] }, page: { querySelector: () => back, contains: () => false },
        placeStatus() {}, singlePane: () => true, syncLayer() {},
    });
    vm.runInContext(source.match(/    function showPane\(view, focusTarget = null\) \{[\s\S]+?\n    \}/)[0], context);
    context.showPane('file', previewEl);
    assert.equal(context.document.activeElement, back, 'an opened file: Back, where it leads');
    context.layoutEl.dataset.filesView = 'list';
    context.showPane('file', nameField);
    assert.equal(context.document.activeElement, nameField, '"+ File": its name field, where the owner types');
});

test('focus on the round Back goes to the file\'s row when both panes come back and the Back goes with its layer', () => {
    const source = readFileSync(new URL('../modules/files.js', import.meta.url), 'utf8');
    let backShown = true;
    const node = (name, rects = () => [1]) => ({ name, getClientRects: rects, focus() { context.document.activeElement = this; } });
    const back = node('back', () => (backShown ? [1] : [])), row = node('row'), editor = node('editor');
    const pageNodes = new Set([back, row, editor]);
    const context = vm.createContext({
        document: { activeElement: back, body: {} }, lastFocus: null, listScroll: 0, previewEl: node('preview'),
        // Back stands in the page's header, outside the panes.
        layoutEl: { dataset: { filesView: 'file' }, contains: (el) => el === row || el === editor }, sidebarEl: {},
        listEl: { querySelectorAll: () => [row] },
        page: { querySelector: () => (backShown ? back : null), contains: (el) => backShown ? pageNodes.has(el) : el !== back && pageNodes.has(el) },
        placeStatus() {}, singlePane: () => false,
        // The wide layout owes no Back: its layer ends and the shell takes the button away.
        syncLayer(open) { if (!open) backShown = false; },
    });
    vm.runInContext(source.match(/    function showPane\(view, focusTarget = null\) \{[\s\S]+?\n    \}/)[0], context);
    context.showPane('file');
    assert.equal(context.document.activeElement, row, 'focus lands on a shown control, never on the page\'s body');
    // A control still in sight keeps its focus.
    context.document.activeElement = editor;
    context.showPane('file');
    assert.equal(context.document.activeElement, editor);
});

// The Files page itself, on the dashboard fixture's document: the page the owner edits in, with a save the test answers
// when it chooses, and the desktop room session recording which editors hold the window (each change, `true` or `false`).
const json = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const settle = async () => { for (let round = 0; round < 12; round += 1) await new Promise((resolve) => setImmediate(resolve)); };
let shimmed = false;

async function filesPage(t) {
    if (!shimmed) {  // what this page asks of a node and the fixture's does not have
        Object.assign(Stub.prototype, {
            before(node) { this.parentElement.insertBefore(node, this); },
            prepend(...nodes) { const first = this.children[0] || null; nodes.forEach((node) => this.insertBefore(node, first)); },
            getBoundingClientRect() { return { top: 0, bottom: 20, left: 0, right: 100, width: 100, height: 20 }; },
        });
        Object.defineProperty(Stub.prototype, 'style', { get() { return this._style ??= { setProperty() {}, removeProperty() {}, getPropertyValue: () => '' }; } });
        shimmed = true;
    }
    const holds = [], writes = [], answers = [];
    const host = Object.assign(new EventTarget(), { webkit: { messageHandlers: { ouroborosRoomSession: {
        postMessage(message) { if (message.op === 'editor') holds.push(message.dirty); return {}; },
    } } } });
    host.top = host;
    await openDesktopRoomSession(host, { documentElement: { dataset: {} }, querySelector: () => ({ content: 'files-test', remove() {} }) }, assert.fail);
    const env = setup(t);
    env.document.createElement = (tag) => new Stub(tag);
    env.document.createComment = () => new Stub('#comment');
    const page = new Stub('div');
    page.id = 'content';
    env.document.body.appendChild(page);
    const entries = [{ name: 'a.txt', path: 'a.txt', type: 'file', size: 5 }, { name: 'b.txt', path: 'b.txt', type: 'file', size: 5 }];
    const listing = () => json({ path: '.', parent_path: '.', root_path: '/work', entries, breadcrumb: [] });
    env.routes.set('/api/files/list', async () => listing());
    env.routes.set('/api/files/list?path=.', async () => listing());
    const disk = Object.fromEntries(['a.txt', 'b.txt'].map((name) => [name, 'saved']));
    for (const name of ['a.txt', 'b.txt']) {
        env.routes.set(`/api/files/read?path=${name}`, async () => json({ path: name, name, display_path: name, content: disk[name], is_text: true, truncated: false, size: 5 }));
    }
    env.routes.set('/api/files/write', (url, init) => new Promise((resolve, reject) => {
        writes.push(JSON.parse(init.body));
        answers.push({ resolve: (body, status = 200) => resolve(json(body, status)), fail: () => reject(new TypeError('Failed to fetch')) });
    }));
    initFiles({ state: {}, setBeforePageLeave() {}, layers: null });
    await settle();
    const editor = () => page.querySelector('.files-editor');
    return {
        holds, writes,
        async open(name = 'a.txt') {
            await page.querySelectorAll('.files-entry').find((row) => row.dataset.path === name).fire('click', {});
            await settle();
        },
        async type(text) { editor().value = text; await editor().fire('input', {}); },
        async save() { await page.querySelector('#files-save').fire('click', {}); await settle(); },
        // Answer the save the page has out: landed, refused by the server (a 400 says nothing was written), or no answer at all.
        async land() { answers.shift().resolve({ ok: true, path: 'a.txt', name: 'a.txt', display_path: 'a.txt', size: 7 }); await settle(); },
        async refuse() { answers.shift().resolve({ error: 'Not a text file.' }, 400); await settle(); },
        async silence() { answers.shift().fail(); await settle(); },
        // The server side of a save whose reply the client never saw: it completes at the server, later or never.
        lands(name, text) { disk[name] = text; },
        text: () => editor()?.value,
        status: () => page.querySelector('#files-preview-status').textContent,
        shown: () => page.querySelector('#files-preview-path').textContent,
        canSave: () => !page.querySelector('#files-save').disabled,
        // The rows marked unsaved once the list is drawn again (the filter draws it; the fixture reads no compound selector).
        async marked() {
            await page.querySelector('#files-search').fire('input', {});
            return page.querySelectorAll('.files-entry').filter((row) => row.hasAttribute('data-unsaved')).map((row) => row.dataset.path);
        },
        // The question leaving a file with unsaved text asks, and the owner's answer to it: Discard or Stay.
        asked: () => env.document.body.querySelector('.confirm-dialog-backdrop')?.querySelector('p')?.textContent ?? null,
        async answer(discard) {
            const dialog = env.document.body.querySelector('.confirm-dialog-backdrop');
            await dialog.fire('click', { target: dialog.querySelector(discard ? '[data-confirm-ok]' : '[data-confirm-cancel]') });
            await settle();
        },
        // Whether closing the page asks the browser to keep it.
        closing() {
            let asked = false;
            env.window.dispatchEvent({ type: 'beforeunload', preventDefault() { asked = true; } });
            return asked;
        },
    };
}

test('Files: words typed back to the saved text keep holding the window while another save is out, then follow its answer', async (t) => {
    const files = await filesPage(t);
    await files.open();
    await files.type('changed');
    assert.deepEqual(files.holds, [true], 'an unsaved edit holds the window');
    await files.save();
    assert.deepEqual(files.writes, [{ path: 'a.txt', content: 'changed', create: false }]);
    await files.type('saved');  // back to what the file held: nothing is unsaved against the file as it is
    assert.deepEqual(files.holds, [true], 'the save out may still land and leave these words unsaved: the hold stays');
    await files.land();
    assert.equal(files.text(), 'saved', 'the owner\'s newest words are kept');
    assert.deepEqual(files.holds, [true], 'it landed: they are unsaved against what it wrote, and still hold the window');
    assert.match(files.status(), /Saved\. Newer edits are still unsaved\./);
    await files.type('changed');
    assert.deepEqual(files.holds, [true, false], 'typed what landed: nothing is unsaved any more');
});

test('Files: a save the server refused wrote nothing, so words equal to the saved text are saved and release the hold', async (t) => {
    const files = await filesPage(t);
    await files.open();
    await files.type('changed');
    await files.save();
    await files.type('saved');
    assert.deepEqual(files.holds, [true]);
    await files.refuse();
    assert.deepEqual(files.holds, [true, false], 'a definite no-write: the words are the saved text');
    assert.equal(files.text(), 'saved');
    assert.match(files.status(), /Not a text file/);
    await files.type('other');
    assert.deepEqual(files.holds, [true, false, true], 'any other words still hold it');
});

// A save that got no answer may have landed or may yet: the last read text typed back is unsaved against it in every place
// that asks (the window's wait, Save, the row's mark, leaving the file, closing the page), not only for the window.
async function unanswered(t) {
    const files = await filesPage(t);
    await files.open();
    await files.type('changed');
    await files.save();
    await files.type('saved');  // back to the text last read
    await files.silence();
    assert.equal(files.text(), 'saved');
    assert.deepEqual(files.holds, [true], 'no answer is no refusal: the write may have landed');
    return files;
}

test('Files: opening another file does not let go of a save that got no answer; it asks, and Stay keeps the file and the hold', async (t) => {
    const files = await unanswered(t);
    assert.deepEqual(await files.marked(), ['a.txt'], 'the row says its text is unsaved');
    assert.equal(files.closing(), true, 'closing the page asks too');
    await files.open('b.txt');
    assert.ok(files.asked(), 'leaving asks, as it does for any unsaved text');
    await files.answer(false);
    assert.equal(files.shown(), 'a.txt', 'Stay keeps the file open');
    assert.equal(files.text(), 'saved');
    assert.deepEqual(files.holds, [true], 'and the window still waits');
    assert.equal(files.writes.length, 1, 'nothing was sent again by leaving');
});

test('Files: a later save that is answered does not retire an earlier save that got no answer', async (t) => {
    const files = await unanswered(t);
    assert.equal(files.canSave(), true, 'the last read text can be saved: it is not what the unanswered save carried');
    assert.equal(files.writes.length, 1, 'the unanswered save is never sent again by itself');
    await files.save();
    assert.deepEqual(files.writes.at(-1), { path: 'a.txt', content: 'saved', create: false }, 'the press saves the text in the editor');
    await files.land();  // answered after the save that got no answer
    assert.match(files.status(), /An earlier save that got no answer may still replace this text/,
        'the server\'s answer speaks for this save alone');
    assert.deepEqual(files.holds, [true], 'the older save may still complete after the answered one');
    assert.equal(files.canSave(), true);
    assert.deepEqual(await files.marked(), ['a.txt'], 'the row keeps its unsaved mark');
    assert.equal(files.closing(), true, 'closing the page still asks');
    // It did complete, after the answered one: the disk holds the older save\'s text, seen once the draft is let go.
    files.lands('a.txt', 'changed');
    await files.open('b.txt');
    assert.ok(files.asked(), 'leaving still asks, as it does for any unsaved text');
    await files.answer(true);
    await files.open('a.txt');
    assert.equal(files.text(), 'changed', 'reopened, the file is what the older save left on the disk');
    assert.deepEqual(files.holds, [true, false], 'nothing is unsaved against what the disk holds');
    assert.equal(files.canSave(), false);
});

test('Files: a refused second save says nothing of the first that got no answer; Discard is the owner letting go', async (t) => {
    const files = await unanswered(t);
    await files.save();
    await files.refuse();  // the server wrote nothing this time: the first save may still land
    assert.deepEqual(files.holds, [true], 'a refusal of the second is no answer to the first');
    assert.equal(files.canSave(), true);
    await files.save();
    await files.silence();  // nor is a second save without an answer, of the very text shown
    assert.deepEqual(files.holds, [true]);
    await files.open('b.txt');
    assert.ok(files.asked());
    await files.answer(true);
    assert.equal(files.shown(), 'b.txt', 'Discard leaves the file');
    assert.deepEqual(files.holds, [true, false], 'and lets go of the saves that got no answer');
    await files.open('a.txt');
    assert.equal(files.canSave(), false, 'read again, the file is what the disk holds');
    await files.type('changed');
    await files.save();
    await files.land();
    assert.deepEqual(files.holds, [true, false, true, false], 'a later save that is answered settles the file');
});
