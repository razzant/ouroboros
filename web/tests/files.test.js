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
