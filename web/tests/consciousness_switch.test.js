// The one switch of consciousness (consciousness_switch.js; owner 15.11, 5A): every Turn on / Turn off in the app asks
// first only where the install has places to name, in the host's words, and sends the same /bg every surface sends.
import assert from 'node:assert/strict';
import test from 'node:test';

import { switchActionWords, switchDialog, turnConsciousness } from '../modules/consciousness_switch.js';

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
    assert.deepEqual(dialog.details.rows, [{ label: '12 people', value: 'Ops' }, { label: '1 person', value: 'Home · Silent' },
        { label: '—', value: 'Team' }]);
    assert.deepEqual([switchDialog('on', PREVIEW).confirmLabel, switchDialog('on', PREVIEW).cancelLabel], ['Turn on', 'Keep off']);
    // Nothing to name: no dialog. A preview that could not be read still asks, and says so.
    assert.equal(switchDialog('off', { lines: ['Ouroboros will act only when you write to it.'], places: [] }), null);
    assert.equal(switchDialog('off', null).body, 'What it would change could not be read.');
});

test('a turn sends the same /bg after a yes, nothing after a no, and asks nothing where nothing is named', async () => {
    const sent = [];
    const ws = { send: (frame) => sent.push(frame) };
    let asked = 0;
    const yes = async () => { asked += 1; return true; };
    const no = async () => { asked += 1; return false; };
    assert.equal(await turnConsciousness('off', { ws, read: async () => PREVIEW, confirm: yes }), true);
    assert.deepEqual(sent.pop(), { type: 'command', cmd: '/bg stop', ui_control: { name: 'consciousness_toggle' } });
    assert.equal(await turnConsciousness('on', { ws, read: async () => PREVIEW, confirm: no }), false);
    assert.equal(sent.length, 0, 'a no sends nothing');
    assert.equal(await turnConsciousness('on', { ws, read: async () => ({ lines: [], places: [] }), confirm: no }), true);
    assert.deepEqual(sent.pop(), { type: 'command', cmd: '/bg start', ui_control: { name: 'consciousness_toggle' } });
    assert.equal(asked, 2, 'no question where nothing is named');
    // A preview read that fails never holds the switch: the dialog asks without it.
    assert.equal(await turnConsciousness('off', { ws, read: async () => { throw new Error('down'); }, confirm: yes }), true);
    assert.equal(sent.pop().cmd, '/bg stop');
});
