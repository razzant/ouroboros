import assert from 'node:assert/strict';
import test from 'node:test';

import {
    providerTestNetworkErrorStatus,
    providerTestResultIsCurrent,
    providerTestStatusText,
} from '../modules/settings.js';
import { renderSettingsPage } from '../modules/settings_ui.js';

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

test('every provider test button warns that one charged request is sent', () => {
    const html = renderSettingsPage();
    const buttons = [...html.matchAll(/data-provider-test="[^"]+"[^>]*>/g)];
    assert.equal(buttons.length, 8);
    for (const [button] of buttons) {
        assert.match(
            button,
            /title="Sends one short model request\. Provider charges may apply\."/,
        );
    }
});

test('provider actions use the shared status-first action row contract', () => {
    const html = renderSettingsPage();
    const rows = [...html.matchAll(/<div class="settings-action-row(?:"|\s)[\s\S]*?<\/div>/g)]
        .map(([row]) => row);
    // Eight provider probes plus the catalog action. The Claude-runtime
    // status/Repair panel is retired with its product surface (the advisory
    // pre-review runs on a configured routed model or agent session now).
    assert.equal(rows.length, 9, 'eight provider probes plus the catalog action');
    assert.doesNotMatch(html, /settings-claude-code/);
    assert.doesNotMatch(html, /settings-ghost-btn/);
    for (const row of rows) {
        const statusAt = row.indexOf('role="status"');
        const actionAt = row.indexOf('class="btn btn-default"');
        assert.ok(statusAt >= 0, 'each action row exposes a live status');
        assert.ok(actionAt > statusAt, 'the action is docked after the status in source order');
        assert.match(row, /aria-live="polite"/);
    }
});
