import assert from 'node:assert/strict';
import test from 'node:test';

import {
    builtinTestRequest,
    providerTestNetworkErrorStatus,
    providerTestResultIsCurrent,
    providerTestStatusText,
    reachesAnotherPair,
} from '../modules/settings_connections.js';
import { PROVIDER_CARDS, providerSettingsCard } from '../modules/settings_provider_cards.js';

// Setup retains the shared provider cards, each with its keys and forms.
const renderSettingsPage = () => PROVIDER_CARDS.map(providerSettingsCard).join('');

test('provider test status uses only the controlled compact vocabulary', () => {
    assert.equal(providerTestStatusText({ ok: true }), 'Works');
    assert.equal(providerTestStatusText({ ok: false }), 'Not ready');
    assert.equal(
        providerTestStatusText({ ok: false, error: 'Invalid key' }),
        'Not ready — Invalid key',
    );
    assert.equal(
        providerTestNetworkErrorStatus(new Error('raw internal HTTP body')),
        'Not ready',
    );
    const refusal = (status, body) => Object.assign(new Error(body.error), { status, body });
    const moved = { error: 'Enter its key again to test the new address.', code: 'masked_key_endpoint_changed' };
    assert.equal(providerTestNetworkErrorStatus(refusal(400, moved)),
        'Not ready — Enter its key again to test the new address.');
    assert.equal(providerTestNetworkErrorStatus(refusal(400, { error: 'raw upstream text' })), 'Not ready');
    assert.equal(providerTestNetworkErrorStatus(refusal(502, moved)), 'Not ready');
});

test('provider test responses apply only to the exact draft generation', () => {
    assert.equal(providerTestResultIsCurrent({
        sentGeneration: 3,
        currentGeneration: 3,
        sentFingerprint: '{"OPENAI_API_KEY":"draft"}',
        currentFingerprint: '{"OPENAI_API_KEY":"draft"}',
    }), true);
    assert.equal(providerTestResultIsCurrent({
        sentGeneration: 3,
        currentGeneration: 4,
        sentFingerprint: '{}',
        currentFingerprint: '{}',
    }), false);
    assert.equal(providerTestResultIsCurrent({
        sentGeneration: 3,
        currentGeneration: 3,
        sentFingerprint: '{"OPENAI_API_KEY":"old"}',
        currentFingerprint: '{"OPENAI_API_KEY":"new"}',
    }), false);
});

test('the built-in compatible key names the pair its draft reaches; other keys send their own edits', () => {
    assert.deepEqual(builtinTestRequest('openai', { OPENAI_API_KEY: 'typed' }, { OPENAI_BASE_URL: 'http://l/v1' }),
        { provider_id: 'openai', overrides: { OPENAI_API_KEY: 'typed' } });
    const legacy = { OPENAI_BASE_URL: 'http://l/v1', OPENAI_API_KEY: '***set***' };
    // The legacy pair travels as the page holds it, a kept key still masked.
    assert.deepEqual(builtinTestRequest('openai-compatible', {}, legacy), { provider_id: 'openai-compatible',
        api_key_setting: 'OPENAI_API_KEY', overrides: legacy });
    // Both saved, the dedicated address cleared in the row: the legacy pair, with the key as just edited.
    assert.deepEqual(builtinTestRequest('openai-compatible', { OPENAI_COMPATIBLE_BASE_URL: '' },
        { ...legacy, OPENAI_API_KEY: 'sk-edited', OPENAI_COMPATIBLE_BASE_URL: '', OPENAI_COMPATIBLE_API_KEY: '***set***' }),
    { provider_id: 'openai-compatible', api_key_setting: 'OPENAI_API_KEY',
        overrides: { OPENAI_COMPATIBLE_BASE_URL: '', OPENAI_BASE_URL: 'http://l/v1', OPENAI_API_KEY: 'sk-edited' } });
    // A dedicated address: the row's own edits only, and the legacy pair stays home.
    assert.deepEqual(builtinTestRequest('openai-compatible', { OPENAI_COMPATIBLE_BASE_URL: 'http://d/v1' },
        { ...legacy, OPENAI_COMPATIBLE_BASE_URL: 'http://d/v1' }), { provider_id: 'openai-compatible',
        api_key_setting: 'OPENAI_COMPATIBLE_API_KEY', overrides: { OPENAI_COMPATIBLE_BASE_URL: 'http://d/v1' } });
});

test('the built-in compatible row keeps its facts only while its draft reaches their pair, as Test reads it', () => {
    const saved = { OPENAI_BASE_URL: 'http://l/v1', OPENAI_API_KEY: 'sk-legac...' };
    // Read as Test reads it: trimmed, and a kept key compared as the mask the page holds.
    assert.equal(reachesAnotherPair(saved, { ...saved, OPENAI_BASE_URL: ' http://l/v1 ' }), false);
    // Edited beside the row (the address, the key it travels with) or in it (a dedicated address): another pair.
    for (const edit of [{ OPENAI_BASE_URL: 'http://b/v1' }, { OPENAI_API_KEY: '' }, { OPENAI_API_KEY: 'sk-typed' },
        { OPENAI_COMPATIBLE_BASE_URL: 'http://d/v1' }]) {
        assert.equal(reachesAnotherPair(saved, { ...saved, ...edit }), true, JSON.stringify(edit));
    }
    // A saved dedicated pair is the endpoint: the legacy pair is beside the point until its address is cleared.
    const both = { ...saved, OPENAI_COMPATIBLE_BASE_URL: 'http://d/v1', OPENAI_COMPATIBLE_API_KEY: 'sk-dedic...' };
    assert.equal(reachesAnotherPair(both, { ...both, OPENAI_BASE_URL: 'http://b/v1', OPENAI_API_KEY: '' }), false);
    assert.equal(reachesAnotherPair(both, { ...both, OPENAI_COMPATIBLE_BASE_URL: '' }), true);
});

test('every provider test button warns that one charged request is sent', () => {
    const html = renderSettingsPage();
    const buttons = [...html.matchAll(/data-provider-test="[^"]+"[^>]*>/g)];
    assert.equal(buttons.length, 9);
    for (const [button] of buttons) {
        assert.match(
            button,
            /title="Sends one short model request\. Provider charges may apply\."/,
        );
    }
});

test('each key row docks Test beside its other actions and reports in its own live status line', () => {
    const html = renderSettingsPage();
    const rows = [...html.matchAll(/<div class="harness-account-row connection-row"[\s\S]*?data-connection-cache hidden><\/div>/g)]
        .map(([row]) => row);
    assert.equal(rows.length, 9, 'one built-in key row per provider card');
    for (const row of rows) {
        const actions = row.slice(row.indexOf('harness-account-actions'), row.indexOf('connection-fields'));
        for (const action of ['data-connection-test', 'data-connection-toggle', 'data-connection-remove']) {
            assert.match(actions, new RegExp(action), `${action} sits in the row's action slot`);
        }
        assert.match(row, /data-connection-test-status[^>]*role="status" aria-live="polite"/);
    }
    // The model catalogue action now belongs to Models, verified by settings_models.test.js.
    assert.doesNotMatch(html, /settings-action-row/);
    assert.doesNotMatch(html, /settings-claude-code/);
    assert.doesNotMatch(html, /settings-ghost-btn/);
});
