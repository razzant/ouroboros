// A settings write says where it came from (CLIENT_CONTRACT "Declared settings origin"): the app's own pages declare
// `app` and the owner, so the server records their change without a row in Main; nothing else declares by default.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { APP_SETTINGS_ORIGIN, apiClient } from '../modules/api_client.js';
import { initCosts } from '../modules/costs.js';
import { sendModelWaitDecision } from '../modules/model_wait.js';
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

test('the Costs budget save declares the app on its settings write', async (t) => {
    const env = setup(t);
    const posts = [];
    env.routes.set('/api/settings', async (_url, init) => {
        posts.push(init.headers);
        return { ok: true, status: 200, json: async () => ({ immediate_changed: true }) };
    });
    initCosts({ mount: env.mount, state: {} });
    env.mount.querySelector('#s-budget').value = '200';
    env.mount.querySelector('#s-per-task-cost').value = '50';
    const save = env.mount.querySelector('#btn-save-budget');
    await save.fire('click', { currentTarget: save });
    await new Promise((resolve) => setImmediate(resolve));
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

test('the Settings page passes the declaration to its generic save', () => {
    // The page's Save runs inside its whole form (browser lanes drive it); the Safety writes are checked by
    // behavior in consciousness_settings.test.js.
    const source = readFileSync(new URL('../modules/settings.js', import.meta.url), 'utf8');
    assert.match(source, /apiClient\.saveSettings\(body, \{ headers: APP_SETTINGS_ORIGIN \}\)/);
});
