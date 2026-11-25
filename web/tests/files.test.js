import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createLayerStack } from '../modules/layers.js';
import { filesPageTemplate } from '../modules/files.js';
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

test('a Files dialog takes its tone from its caller, never from its words', async () => {
    const source = readFileSync(new URL('../modules/files.js', import.meta.url), 'utf8');
    const asked = [];
    const context = vm.createContext({
        openConfirmDialog: async (options) => { asked.push(options); return false; },
        state: { editorDirty: false, path: '.', selectedPath: 'notes.md', entries: [{ path: 'notes.md', name: 'notes.md', type: 'file' }] },
    });
    for (const name of ['showModal', 'canLeaveEditor', 'createDirectory', 'deleteSelectedEntry'])
        vm.runInContext(source.match(new RegExp(`    async function ${name}\\([\\s\\S]+?\\n    \\}`))[0], context);
    await context.showModal({ title: 'Delete the note?', message: 'This cannot be undone.', confirmLabel: 'Delete' });
    await context.showModal({ title: 'Remove the note?', message: 'This cannot be undone.', confirmLabel: 'Remove', danger: true });
    assert.deepEqual(asked.splice(0).map((options) => options.danger), [false, true], 'the word Delete is not red and Remove is when the caller says so');
    // Each of the page's three dialogs, opened by its own action, says its tone: leaving unsaved edits and deleting are
    // destructive, a new folder is not (whatever their words).
    context.state.editorDirty = true;
    await context.canLeaveEditor();
    context.state.editorDirty = false;
    await context.createDirectory();
    await context.deleteSelectedEntry();
    assert.deepEqual(asked.map((options) => options.danger), [true, false, true]);
    assert.doesNotMatch(source, /danger:\s*\/[^/]+\/i?\.test\(/, 'no tone guessed from a label');
});
