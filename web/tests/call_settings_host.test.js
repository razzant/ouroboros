import test from 'node:test';
import assert from 'node:assert/strict';
import { renderExtensionSettingsSections } from '../modules/settings.js';

test('only a successful extension form Save announces changed settings', async () => {
    const prior = { fetch: globalThis.fetch, window: globalThis.window };
    const win = new EventTarget(), changed = [];
    win.addEventListener('ouro:settings-updated', event => changed.push(event.detail));
    globalThis.window = win;
    let submit;
    const status = { dataset: {}, textContent: '' }, button = {};
    const form = { dataset: { skill:'test-skill', route:'save', extensionSettingsKey:'test-skill:save' },
        addEventListener(name, handler) { if (name === 'submit') submit = handler; },
        querySelector: selector => selector === '[data-extension-settings-status]' ? status
            : selector === 'button[type="submit"]' ? button : null };
    const host = { innerHTML:'', querySelectorAll: () => [form] };
    const root = { querySelector: () => host };
    try {
        await renderExtensionSettingsSections(root, [{ skill:'test-skill', render:{components:[{type:'form',route:'save',fields:[]}]} }]);
        for (const [code, body, expected] of [[500, {}, 0], [200, {error:'refused'}, 0], [200, {ok:false, message:'refused'}, 0], [200, {ok:true}, 1]]) {
            globalThis.fetch = async () => ({ ok:code === 200, status:code, json:async () => body });
            await submit({ preventDefault() {} });
            assert.equal(changed.length, expected, JSON.stringify(body));
        }
        assert.deepEqual(changed[0], { reason:'extension settings saved', source:'settings', skill:'test-skill', route:'save' });
    } finally { Object.assign(globalThis, prior); }
});
