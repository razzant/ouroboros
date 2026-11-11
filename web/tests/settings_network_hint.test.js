// The LAN hint under the bind host is rebuilt by every settings read, so its Copy link button carries the address
// that read drew and one listener, bound once, reads it at click time. The painter and the binder run against a hint
// double; the section that holds the hint is unmounted in settings_ui.js until its page is rebuilt.
import assert from 'node:assert/strict';
import test from 'node:test';

import { bindNetworkHintCopy, paintNetworkHint } from '../modules/settings_ui.js';

test('the LAN hint draws one Copy link button per read and copies the address it was drawn with', () => {
    const copies = [];
    const hint = {
        hidden: true, dataset: {}, innerHTML: '', listeners: new Map(),
        addEventListener(type, handler) { this.listeners.set(type, handler); },
        removeEventListener(type) { this.listeners.delete(type); },
    };
    const unbind = bindNetworkHintCopy(hint, {
        copy: (url, options) => { copies.push([url, { ...options }]); return Promise.resolve(true); },
    });
    assert.equal(hint.listeners.size, 1, 'one listener for the hint, whatever it draws later');
    const buttons = () => [...hint.innerHTML.matchAll(/<button\b([^>]*)>([^<]*)<\/button>/g)];
    const click = () => {
        const [, attrs = ''] = buttons()[0] || [];
        const raw = /data-copy-lan-url="([^"]*)"/.exec(attrs)?.[1];
        const url = raw?.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
        hint.listeners.get('click')({ target: { closest: (selector) => (selector === '[data-copy-lan-url]' && raw !== undefined ? { dataset: { copyLanUrl: url } } : null) } });
    };

    paintNetworkHint(hint, { reachability: 'lan_reachable', recommended_url: 'http://192.0.2.44:8765', warning: 'Set a Network Password.' });
    assert.equal(buttons().length, 1);
    assert.match(buttons()[0][1], /class="btn btn-default btn-sm ui-hit-area"/, 'the shared button recipe with the shared touch reach');
    assert.equal(buttons()[0][2], 'Copy link');
    assert.match(hint.innerHTML, /^<div class="ui-action-row"><span>LAN URL: <a href="http:\/\/192\.0\.2\.44:8765" target="_blank" rel="noopener">http:\/\/192\.0\.2\.44:8765<\/a> — <strong>Set a Network Password\.<\/strong><\/span><button /,
        'the link keeps its words and the warning its place; the action sits right of the status');
    assert.deepEqual([hint.hidden, hint.dataset.tone], [false, 'warn']);
    click();
    assert.deepEqual(copies, [['http://192.0.2.44:8765', { okMessage: 'LAN URL copied.' }]]);

    // A later read (the address or the bind changed) leaves one button, and it copies the new address.
    paintNetworkHint(hint, { reachability: 'lan_reachable', recommended_url: 'http://192.0.2.45:9000/?a=1&b="2"', warning: '' });
    assert.equal(buttons().length, 1);
    assert.deepEqual([hint.dataset.tone, hint.innerHTML.includes('<strong>')], ['ok', false]);
    click();
    assert.equal(copies.at(-1)[0], 'http://192.0.2.45:9000/?a=1&b="2"', 'the address as it is, not as it is escaped in markup');
    assert.equal(copies.length, 2);

    // Only a real LAN address can be copied: a placeholder, loopback or an unknown state offers nothing to copy.
    for (const meta of [{ reachability: 'host_ip_unknown', recommended_url: 'http://your-host-ip:8765' },
        { reachability: 'loopback_only', recommended_url: '' }]) {
        paintNetworkHint(hint, meta);
        assert.equal(buttons().length, 0, meta.reachability);
        click();
    }
    paintNetworkHint(hint, { reachability: 'elsewhere' });
    assert.equal(hint.hidden, true);
    assert.equal(copies.length, 2, 'no click outside the button copied anything');
    unbind();
    assert.equal(hint.listeners.size, 0);
});
