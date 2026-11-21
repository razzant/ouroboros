// The one switch of consciousness (consciousness_switch.js; owner 15.11, 5A): every Turn on / Turn off in the app asks
// first only where the install has places to name, in the host's words, and sends the same /bg every surface sends.
import assert from 'node:assert/strict';
import test from 'node:test';

import { switchActionWords, switchDialog, turnConsciousness } from '../modules/consciousness_switch.js';
import { readSwitchPreview } from '../modules/consciousness_api.js';
import { renderConfirmDialogDetails } from '../modules/confirm_dialog.js';

const PREVIEW = {
    lines: ['Ouroboros will act only when you write to it.', '3 places will be heard, not answered: Ops, Team and Home.'],
    places: [{ id: 'ops', label: 'Ops', state: 'active', people: 12 }, { id: 'home', label: 'Home', state: 'silent', people: 1 },
        { id: 'team', label: 'Team', state: 'active', people: null }],
};

test('the switch says what it does next by its buttons\' own words', () => {
    assert.equal(switchActionWords(true), 'Turn off');
    assert.equal(switchActionWords(false), 'Turn on');
});

test('the dialog names what changes, one sentence a line, every place behind its one disclosure', () => {
    const dialog = switchDialog('off', PREVIEW);
    assert.equal(dialog.title, 'Turn off consciousness?');
    assert.equal(dialog.body, PREVIEW.lines.join('\n'));
    assert.deepEqual([dialog.confirmLabel, dialog.cancelLabel], ['Turn off', 'Keep on']);
    assert.equal(dialog.details.summary, 'All 3 places');
    // A place's own name is authored: the page's translator leaves it as written (review r3).
    assert.deepEqual(dialog.details.rows, [{ label: '12 people', value: 'Ops', authored: true },
        { label: '1 person', value: 'Home · Silent', authored: true }, { label: '—', value: 'Team', authored: true }]);
    assert.equal(switchDialog('off', { ...PREVIEW, places: [{ id: 'big', label: 'Big', state: 'active', people: 2000, atLeast: true }] })
        .details.rows[0].label, 'at least 2000 people', 'a capped place says at least');
    assert.equal(switchDialog('off', { ...PREVIEW, places: [{ id: 'one', label: 'One', state: 'active', people: 1, atLeast: true }] })
        .details.rows[0].label, 'at least 1 person');
    assert.deepEqual([switchDialog('on', PREVIEW).confirmLabel, switchDialog('on', PREVIEW).cancelLabel], ['Turn on', 'Keep off']);
    // Nothing to name: no dialog. A preview that could not be read still asks, and says so.
    assert.equal(switchDialog('off', { lines: ['Ouroboros will act only when you write to it.'], places: [] }), null);
    assert.equal(switchDialog('off', null).body, 'What it would change could not be read.');
    // The host could not read the places (`places: null`): the dialog still asks, with the host's lines, no details.
    const unread = switchDialog('off', { lines: ['Ouroboros will act only when you write to it.', 'Its places will be heard, not answered; which ones could not be read.'], places: null });
    assert.equal(unread.body, 'Ouroboros will act only when you write to it.\nIts places will be heard, not answered; which ones could not be read.');
    assert.equal(unread.details, null);
});

test('a turn sends the same /bg after a yes, nothing after a no, and asks nothing where nothing is named', async () => {
    const sent = [];
    const ws = { send: (frame) => sent.push(frame) };
    let asked = 0;
    const yes = async () => { asked += 1; return true; };
    const no = async () => { asked += 1; return false; };
    assert.equal(await turnConsciousness('off', { ws, read: async () => PREVIEW, ask: yes }), true);
    assert.deepEqual(sent.pop(), { type: 'command', cmd: '/bg stop', ui_control: { name: 'consciousness_toggle' } });
    assert.equal(await turnConsciousness('on', { ws, read: async () => PREVIEW, ask: no }), false);
    assert.equal(sent.length, 0, 'a no sends nothing');
    assert.equal(await turnConsciousness('on', { ws, read: async () => ({ lines: [], places: [] }), ask: no }), true);
    assert.deepEqual(sent.pop(), { type: 'command', cmd: '/bg start', ui_control: { name: 'consciousness_toggle' } });
    assert.equal(asked, 2, 'no question where nothing is named');
    // A preview read that fails never holds the switch: the dialog asks without it.
    assert.equal(await turnConsciousness('off', { ws, read: async () => { throw new Error('down'); }, ask: yes }), true);
    assert.equal(sent.pop().cmd, '/bg stop');
    // The caller hears when the read is over, before any question (its button stops saying «Reading…»).
    const order = [];
    await turnConsciousness('off', { ws, read: async () => { order.push('read'); return PREVIEW; },
        onRead: () => order.push('onRead'), ask: async () => { order.push('ask'); return false; } });
    assert.deepEqual(order, ['read', 'onRead', 'ask']);
    // Places the host could not read are not «no places»: it asks before the turn.
    assert.equal(await turnConsciousness('on', { ws, read: async () => ({ lines: ['Where it may answer could not be read.'], places: null }), ask: no }), false);
    assert.equal(asked, 4);
    assert.equal(sent.length, 0);
});

test('the preview read keeps «could not be read» (null) apart from «no place» ([])', async () => {
    const previous = globalThis.fetch;
    try {
        for (const [places, expected] of [[null, null], [[], []]]) {
            globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({ lines: ['x'], places }) });
            assert.deepEqual((await readSwitchPreview('off')).places, expected);
        }
    } finally {
        globalThis.fetch = previous;
    }
});

test('an answer the preview cannot read, or one that never ends, is never «no place»: the switch still asks', async () => {
    // A cut 200 body or an older host's answer has no lines: unread, so the dialog asks (review r3).
    const previous = globalThis.fetch;
    try {
        for (const body of [{ to: 'off' }, { places_brief: [{ place_id: 'lab' }] }, null]) {
            globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => body });
            await assert.rejects(readSwitchPreview('off'));
        }
        globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({ lines: ['x'], places: [{ place_id: 'b', label: 'Big', state: 'active', people: 2000, people_capped: true }] }) });
        assert.equal((await readSwitchPreview('off')).places[0].atLeast, true);
        // A row with its name keeps its place in the details; a row with neither id nor name makes the list
        // unreadable, never «no place» or a silently shorter list (reviews r3, r4).
        globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({ lines: ['x'], places: [{ place_id: 'lab', label: 'Lab', state: 'active' }, { label: 'Ops', state: 'active' }] }) });
        assert.deepEqual((await readSwitchPreview('off')).places.map((row) => row.label), ['Lab', 'Ops']);
        globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({ lines: ['x'], places: [{ place_id: 'lab', label: 'Lab' }, { state: 'active' }] }) });
        await assert.rejects(readSwitchPreview('off'));
        globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({ lines: ['x'], places: [{ label: '   ', state: 'active' }] }) });
        await assert.rejects(readSwitchPreview('off'), undefined, 'a name of blanks is no name (review r5)');
        // A preview that never answers ends at its lifetime (here shortened), so the switch is never held.
        globalThis.fetch = (_url, init) => new Promise((_resolve, reject) => {
            init?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })));
        });
        await assert.rejects(readSwitchPreview('off', { lifetimeMs: 20 }));
    } finally {
        globalThis.fetch = previous;
    }
    const sent = [];
    let asked = 0;
    const turned = await turnConsciousness('off', { ws: { send: (frame) => sent.push(frame) },
        read: async () => { throw new Error('unreadable'); }, ask: async () => { asked += 1; return false; } });
    assert.deepEqual([turned, asked, sent.length], [false, 1, 0], 'an unread preview asks, and a no sends nothing');
});

test('the dialog keeps a place\'s own name out of the page\'s translator', () => {
    const html = renderConfirmDialogDetails(switchDialog('off', PREVIEW).details);
    assert.match(html, /<dd data-i18n-authored>Ops<\/dd>/);
    assert.match(renderConfirmDialogDetails({ rows: [{ label: 'Cause', value: 'host words' }] }), /<dd>host words<\/dd>/,
        'a host value stays translatable');
});

