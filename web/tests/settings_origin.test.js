// A settings write says where it came from (CLIENT_CONTRACT "Declared settings origin"): the app's own pages declare
// `app` and the owner, so the server records their change without a row in Main; nothing else declares by default.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { APP_SETTINGS_ORIGIN, apiClient } from '../modules/api_client.js';
import { initCosts } from '../modules/costs.js';
import { sendModelWaitDecision } from '../modules/model_wait.js';
import { forgetSettingRows } from '../modules/settings_controls.js';
import { setup } from './dashboard_dom_fixture.js';

test('the generic save carries the declaration only when its caller passes one', async () => {
    const seen = [];
    const previous = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        seen.push(init.headers);
        return { ok: true, status: 200, json: async () => ({ status: 'saved' }) };
    };
    try {
        await apiClient.saveSettings({ TOTAL_BUDGET: 25 });
        await apiClient.saveSettings({ TOTAL_BUDGET: 25 }, { headers: APP_SETTINGS_ORIGIN });
    } finally { globalThis.fetch = previous; }
    assert.deepEqual(APP_SETTINGS_ORIGIN, { 'X-Ouroboros-Surface': 'app', 'X-Ouroboros-Actor': 'owner' });
    assert.deepEqual(seen, [{ 'Content-Type': 'application/json' }, { 'Content-Type': 'application/json', ...APP_SETTINGS_ORIGIN }]);
});

test('a Costs limit row declares the app on its settings write', async (t) => {
    forgetSettingRows();
    const env = setup(t);
    const posts = [];
    env.routes.set('/api/cost-breakdown', { ok: false, status: 503, json: async () => ({}) });
    env.routes.set('/api/settings', async (_url, init = {}) => {
        if (init.method !== 'POST') return { ok: true, status: 200, json: async () => ({ TOTAL_BUDGET: 100 }) };
        posts.push(init.headers);
        return { ok: true, status: 200, json: async () => ({ status: 'saved' }) };
    });
    initCosts({ mount: env.mount, state: {} });
    const field = env.mount.querySelector('#setting-TOTAL_BUDGET');
    field.value = '200';
    await env.mount.querySelector('.costs-budget').fire('change', { target: field });
    for (let round = 0; round < 8; round += 1) await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(posts, [{ 'Content-Type': 'application/json', ...APP_SETTINGS_ORIGIN }]);
});

test('a model-wait card declares the app on its decision, which may save the chosen model in Settings', async () => {
    const seen = [];
    const previous = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        seen.push([url, init.headers]);
        return { ok: true, status: 202, json: async () => ({ ok: true, wait: { wait_id: 'w' } }) };
    };
    try {
        await sendModelWaitDecision({ request_id: 'r', decision_id: 'model_wait:t:w', action: 'switch',
            persist_role: true });
    } finally { globalThis.fetch = previous; }
    assert.deepEqual(seen, [['/api/decisions', { 'Content-Type': 'application/json', ...APP_SETTINGS_ORIGIN }]]);
});

test('the one Settings writer passes the declaration to its generic save', () => {
    // Every Settings row writes through settings_controls.js (the Costs row above drives it by behavior); the shell
    // in settings.js keeps no Save of its own.
    const writer = readFileSync(new URL('../modules/settings_controls.js', import.meta.url), 'utf8');
    assert.match(writer, /apiClient\.saveSettings\(body, \{ headers: APP_SETTINGS_ORIGIN \}\)/);
    const shell = readFileSync(new URL('../modules/settings.js', import.meta.url), 'utf8');
    assert.doesNotMatch(shell, /saveSettings|\/api\/settings/);
});
