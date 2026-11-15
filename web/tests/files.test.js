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
    assert.equal(band.contains(page.querySelector('#files-search')), false);
    assert.equal(page.querySelector('#files-search').getAttribute('type'), 'search', 'Escape can clear a filter before Back');
    assert.equal(page.querySelector('[data-files-back]'), null, 'a preview uses the shell slot too');
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
    layers.destroy();
});
