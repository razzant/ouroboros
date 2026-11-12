// The address other devices type (This computer › Who can reach it) is drawn again by every settings read, so its
// Copy link button carries the address that read drew and one listener, bound once, reads it at click time. The
// painter and the binder run against a double of the row's end; the row's own line says the note.
import assert from 'node:assert/strict';
import test from 'node:test';

import { rowById } from '../modules/settings_rows.js';
import { bindNetworkHintCopy, lanAddress, paintNetworkHint, tuneComputerRows } from '../modules/settings_ui.js';

test('the address row draws one Copy link button per read and copies the address it was drawn with', () => {
    const copies = [];
    const hint = {
        innerHTML: '', listeners: new Map(),
        addEventListener(type, handler) { this.listeners.set(type, handler); },
        removeEventListener(type) { this.listeners.delete(type); },
    };
    const unbind = bindNetworkHintCopy(hint, {
        copy: (url, options) => { copies.push([url, { ...options }]); return Promise.resolve(true); },
    });
    assert.equal(hint.listeners.size, 1, 'one listener, whatever is drawn later');
    const buttons = () => [...hint.innerHTML.matchAll(/<button\b([^>]*)>([^<]*)<\/button>/g)];
    const click = () => {
        const [, attrs = ''] = buttons()[0] || [];
        const raw = /data-copy-lan-url="([^"]*)"/.exec(attrs)?.[1];
        const url = raw?.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
        hint.listeners.get('click')({ target: { closest: (selector) => (selector === '[data-copy-lan-url]' && raw !== undefined ? { dataset: { copyLanUrl: url } } : null) } });
    };

    const warned = { reachability: 'lan_reachable', recommended_url: 'http://192.0.2.44:8765', warning: 'Set a Network Password.' };
    paintNetworkHint(hint, warned);
    assert.equal(buttons().length, 1);
    assert.match(buttons()[0][1], /class="btn btn-default btn-sm ui-hit-area"/, 'the shared button recipe with the shared touch reach');
    assert.equal(buttons()[0][2], 'Copy link');
    assert.match(hint.innerHTML, /^<a class="settings-row-value settings-row-link" href="http:\/\/192\.0\.2\.44:8765" target="_blank" rel="noopener">http:\/\/192\.0\.2\.44:8765<\/a><button/,
        'the address is a link, and the action sits after it');
    // The row's one line says where to type the address; the server's warning about it stands beside it as a warning.
    assert.deepEqual(lanAddress(warned), { url: 'http://192.0.2.44:8765', copy: true, warning: 'Set a Network Password.',
        note: 'Type it in a browser on the same network.' });
    assert.equal(lanAddress({ ...warned, warning: '' }).warning, '');
    click();
    assert.deepEqual(copies, [['http://192.0.2.44:8765', { okMessage: 'LAN URL copied.' }]]);

    // A later read (the address or the bind changed) leaves one button, and it copies the new address.
    paintNetworkHint(hint, { reachability: 'lan_reachable', recommended_url: 'http://192.0.2.45:9000/?a=1&b="2"', warning: '' });
    assert.equal(buttons().length, 1);
    click();
    assert.equal(copies.at(-1)[0], 'http://192.0.2.45:9000/?a=1&b="2"', 'the address as it is, not as it is escaped in markup');
    assert.equal(copies.length, 2);

    // Only a real LAN address can be copied: a placeholder offers nothing to copy, and loopback or an unknown state no row.
    paintNetworkHint(hint, { reachability: 'host_ip_unknown', recommended_url: 'http://your-host-ip:8765' });
    assert.equal(buttons().length, 0);
    assert.match(hint.innerHTML, /^<span class="settings-row-value">http:\/\/your-host-ip:8765<\/span>$/);
    const unknown = lanAddress({ reachability: 'host_ip_unknown', recommended_url: 'http://your-host-ip:8765', warning: 'No password is set.' });
    assert.match(unknown.note, /could not be found\. Put its address in place of your-host-ip\.$/);
    assert.deepEqual([unknown.copy, unknown.warning], [false, 'No password is set.']);
    click();
    for (const meta of [{ reachability: 'loopback_only', recommended_url: '' }, { reachability: 'elsewhere' }, undefined]) {
        paintNetworkHint(hint, meta);
        assert.equal(hint.innerHTML, '');
        assert.equal(lanAddress(meta), null);
    }
    assert.equal(copies.length, 2, 'no click outside the button copied anything');
    unbind();
    assert.equal(hint.listeners.size, 0);
});

const COMPUTER = ['OUROBOROS_SERVER_HOST', 'OUROBOROS_NETWORK_PASSWORD', 'computer:address', 'computer:about'];
const tuned = (settings) => Object.fromEntries(tuneComputerRows(COMPUTER.map((id) => structuredClone(rowById(id))), settings).map((row) => [row.id, row]));

test('the network rows say whether a password is set and when the way it was started decides the address', () => {
    const closed = tuned({ OUROBOROS_SERVER_HOST: '127.0.0.1', OUROBOROS_NETWORK_PASSWORD: '', _meta: { bind_host: '127.0.0.1', reachability: 'loopback_only' } });
    assert.match(closed.OUROBOROS_NETWORK_PASSWORD.meta, /^Not set\. .* set it before you open to your network\.$/);
    assert.deepEqual(closed.OUROBOROS_SERVER_HOST.control.options.map((option) => option.note),
        ['Only this computer.', 'Any device on your network, with the password.']);
    const open = tuned({ OUROBOROS_SERVER_HOST: '0.0.0.0', OUROBOROS_NETWORK_PASSWORD: '***set***',
        _meta: { bind_host: '0.0.0.0', reachability: 'lan_reachable', recommended_url: 'http://192.0.2.44:8765', warning: '' } });
    assert.equal(open.OUROBOROS_NETWORK_PASSWORD.meta, rowById('OUROBOROS_NETWORK_PASSWORD').meta, 'a set one says Saved in its own place');
    assert.equal(open['computer:address'].meta, 'Type it in a browser on the same network.');
    // A saved choice that waits for a restart is the row helper's "Running … after restart" line: no note here competes with it.
    const waiting = tuned({ OUROBOROS_SERVER_HOST: '0.0.0.0', OUROBOROS_NETWORK_PASSWORD: '***set***',
        _meta: { bind_host: '127.0.0.1', restart_state: { restart_keys: ['OUROBOROS_SERVER_HOST'], running: { OUROBOROS_SERVER_HOST: '127.0.0.1' } } } });
    assert.equal(waiting.OUROBOROS_SERVER_HOST.control.options[1].note, 'Any device on your network, with the password.');
    // Started with its own address (an environment or a command line): the choice does not decide, and says so.
    const outside = tuned({ OUROBOROS_SERVER_HOST: '127.0.0.1', OUROBOROS_NETWORK_PASSWORD: 'x', _meta: { bind_host: '0.0.0.0', restart_state: { restart_keys: [] } } });
    for (const option of outside.OUROBOROS_SERVER_HOST.control.options) assert.match(option.note, /^It listens on 0\.0\.0\.0 now: the way Ouroboros was started sets that/);
    // A saved address the two choices cannot say (set by hand) is named, never shown as one of them.
    assert.equal(tuned({ OUROBOROS_SERVER_HOST: '192.0.2.7', _meta: {} }).OUROBOROS_SERVER_HOST.meta, 'Listens on 192.0.2.7: set outside Settings.');
    // Before any read the rows are as described.
    assert.equal(tuned(null).OUROBOROS_NETWORK_PASSWORD.meta, rowById('OUROBOROS_NETWORK_PASSWORD').meta);
});
