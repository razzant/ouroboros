// A settings write says where it came from (CLIENT_CONTRACT "Declared settings origin"): the app's own pages declare
// `app` and the owner, so the server records their change without a row in Main; nothing else declares by default.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { APP_SETTINGS_ORIGIN, apiClient } from '../modules/api_client.js';

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

test('Settings and the Costs budget save declare the app on their settings writes', () => {
    const source = (name) => readFileSync(new URL(`../modules/${name}`, import.meta.url), 'utf8');
    assert.match(source('settings.js'), /apiClient\.saveSettings\(body, \{ headers: APP_SETTINGS_ORIGIN \}\)/);
    assert.match(source('costs.js'), /apiFetch\('\/api\/settings', \{\s*method: 'POST',\s*headers: \{ 'Content-Type': 'application\/json', \.\.\.APP_SETTINGS_ORIGIN \}/);
    // The checker transaction, both levels, the rules and "Check consciousness fully".
    assert.equal(source('settings_safety.js').match(/headers: APP_SETTINGS_ORIGIN/g).length, 4);
});
