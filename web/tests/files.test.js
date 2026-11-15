import assert from 'node:assert/strict';
import test from 'node:test';
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
});
