// The main-interface choice in Settings (This computer › Advanced): one row, applied at the next restart, beside
// what the core actually serves right now.
import assert from 'node:assert/strict';
import test from 'node:test';

import { ROWS, rowById } from '../modules/settings_rows.js';
import { servedUiStatus } from '../modules/settings_ui.js';

test('the choice is one row of This computer › Advanced, a plain field that travels only when edited', () => {
    const row = rowById('OUROBOROS_UI_DIR');
    assert.deepEqual([row.page, row.fold, row.control.kind, row.writer], ['computer/advanced', false, 'text', 'generic']);
    assert.equal(ROWS.filter((each) => each.keys.includes('OUROBOROS_UI_DIR')).length, 1);
    assert.ok(row.aliases.includes('Interface directory'), 'the old label finds it');
});

test('the row is truthful about what is served, refresh ownership and the way back', () => {
    const { meta, more } = rowById('OUROBOROS_UI_DIR');
    assert.equal(meta, 'Empty serves the built-in interface.');
    assert.match(more, /^none: no main page/);
    assert.match(more, /an absolute or ~ path/);
    assert.match(more, /keeps the built-in interface and says why/);
    assert.match(more, /the built-in page reloads into the new choice; a page from a folder or another host refreshes itself and only reconnects/);
    assert.match(more, /To return to the built-in interface, clear this field .* restart, and open the Ouroboros address again/);
    assert.match(more, /a saved choice, empty included, wins/);
    // When it applies is the server's fact (`_meta.effects`), said by the row's result line: no sentence here restates it.
    assert.doesNotMatch(meta + more, /Applies at the next restart/);
});

test('the applied-state line names what is served now, and a refused choice with its reason', () => {
    assert.deepEqual(servedUiStatus({ source: 'builtin' }), { text: 'Serving now: the built-in interface.', tone: 'muted' });
    assert.deepEqual(servedUiStatus({ source: 'directory' }), { text: 'Serving now: the interface directory.', tone: 'muted' });
    assert.deepEqual(servedUiStatus({ source: 'none' }), { text: 'Serving now: no main page.', tone: 'muted' });
    const refused = "OUROBOROS_UI_DIR '/srv/ui' was not applied: /srv/ui has no index.html";
    assert.deepEqual(servedUiStatus({ source: 'builtin', error: refused }), {
        text: `Serving now: the built-in interface. ${refused}`, tone: 'warn',
    });
    // A core older than the choice, or an unreadable fact, shows nothing.
    for (const value of [undefined, null, {}, { source: 'other' }, 'builtin']) {
        assert.equal(servedUiStatus(value), null);
    }
});
