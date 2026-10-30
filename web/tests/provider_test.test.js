import assert from 'node:assert/strict';
import test from 'node:test';

import {
    providerTestNetworkErrorStatus,
    providerTestResultIsCurrent,
    providerTestStatusText,
} from '../modules/settings_connections.js';
import { cyberReviewNote } from '../modules/settings_policy.js';
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
    // The catalog refresh keeps the status-first single-action row. The Claude-runtime
    // status/Repair panel is retired with its product surface.
    assert.equal([...html.matchAll(/<div class="settings-action-row(?:"|\s)/g)].length, 1);
    assert.doesNotMatch(html, /settings-claude-code/);
    assert.doesNotMatch(html, /settings-ghost-btn/);
});

test('access, supervisor, and review remain independent owner controls', () => {
    const html = renderSettingsPage();
    assert.match(html, /id="s-runtime-mode"/);
    assert.match(html, /id="s-safety-mode"/);
    assert.match(html, /id="s-review-enforcement"/);
    assert.match(html, /data-policy-state="access"/);
    assert.match(html, /data-policy-state="supervisor"/);
    assert.match(html, /data-policy-state="review"/);
    assert.match(html, /data-effort-value="cyber_pro">Cyber Pro</);
    assert.match(html, /data-effort-value="blocking">Blocking</);
    // Cyber Pro keeps the owner's Blocking choice; review findings do not block actions (#1736).
    // Other controls (Panic, the owner's own, money and deadlines) still stop work, so no
    // copy may say nothing blocks.
    const access = html.split('<h3>Access</h3>')[1].split('<h3>')[0];
    const enforcement = html.split('<h3>Review Enforcement</h3>')[1].split('<h3>')[0];
    const cycles = html.split('<h3>Max Review Cycles</h3>')[1].split('<h3>')[0];
    assert.doesNotMatch(html, /Blocking<\/code> stays available in Cyber Pro|nothing blocks/);
    assert.match(access, /<code>Cyber Pro<\/code>[\s\S]*Review still runs[\s\S]*review findings do not block actions in Cyber Pro/);
    assert.match(enforcement, /<code>Blocking<\/code> stops commits[\s\S]*outside Cyber Pro/);
    assert.match(cycles, /Blocking still requires reviewer approval outside Cyber Pro\./);
});

test('the Cyber Pro note keeps current enforcement apart from the selected next one', () => {
    // The shape gateway/settings._build_policy_state returns; drafts are the controls' values.
    const policy = ({ now, saved = now, review = 'advisory', next = review }) => ({
        access: { current_process: now, effective: now, configured: saved, next_task: saved },
        review: { current_process: review, effective: review, configured: next, next_task: next } });
    const note = (state, draft) => cyberReviewNote(state, draft);
    // Enforced or not in play: no clause.
    assert.equal(note(policy({ now: 'pro', review: 'blocking' })), '');
    assert.equal(note(policy({ now: 'cyber_pro' })), '');
    // Current Blocking under the current Cyber Pro, kept for the next start.
    assert.equal(note(policy({ now: 'cyber_pro', review: 'blocking' })),
        'Cyber Pro: Blocking is not enforced; review findings do not block actions');
    // Current Pro with Advisory, unsaved Blocking and Cyber Pro: only a conditional, never "enforced".
    const drafted = note(policy({ now: 'pro' }), { enforcement: 'blocking', access: 'cyber_pro' });
    assert.equal(drafted, 'Selected Blocking would not be enforced under the selected Cyber Pro');
    assert.doesNotMatch(drafted, /is enforced|until/);
    // Current Pro with Blocking, Advisory drafted (with or without Cyber Pro): nothing to qualify.
    assert.equal(note(policy({ now: 'pro', review: 'blocking' }), { enforcement: 'advisory' }), '');
    assert.equal(note(policy({ now: 'pro', review: 'blocking' }), { enforcement: 'advisory', access: 'cyber_pro' }), '');
    // An unknown current Access states nothing about now.
    assert.equal(note(policy({ now: undefined, saved: 'cyber_pro', review: 'blocking' })),
        'Selected Blocking would not be enforced under the selected Cyber Pro');
    assert.equal(note(policy({ now: 'unknown', saved: 'pro', review: 'blocking' })), '');
    // Current Cyber Pro with Blocking, then drafts.
    const cyber = policy({ now: 'cyber_pro', review: 'blocking' });
    assert.equal(note(cyber, { access: 'pro' }),
        'Cyber Pro: Blocking is not enforced now; selected Blocking would be enforced only after a restart under the selected Pro');
    assert.equal(note(cyber, { enforcement: 'advisory' }), 'Cyber Pro: Blocking is not enforced now');
    assert.equal(note(policy({ now: 'cyber_pro' }), { enforcement: 'blocking', access: 'advanced' }),
        'Selected Blocking would be enforced only after a restart under the selected Advanced');
});

test('settings copy distinguishes Cyber configuration authority without auto-enabling evolution', () => {
    const html = renderSettingsPage();
    // Each card up to the next one, wherever the Behavior order puts it.
    const supervisor = html.split('<h3>Safety Supervisor</h3>')[1].split('<h3>')[0];
    const evolution = html.split('<h3>Post-Task Self-Evolution</h3>')[1].split('<h3>')[0];
    assert.match(supervisor, /outside Cyber Pro, Ouroboros cannot lower its own checks/);
    assert.match(supervisor, /Cyber Pro also lets it configure Supervisor coverage/);
    assert.doesNotMatch(supervisor, /Owner-only/);
    assert.match(evolution, /outside Cyber Pro, only the owner can enable this/);
    assert.match(evolution, /Cyber Pro also lets the agent configure it/);
    assert.match(evolution, /selecting Cyber Pro does not enable evolution automatically/);
    for (const section of [supervisor, evolution]) assert.match(section, /Changes apply on the next task/);
    assert.doesNotMatch(html, /available in Cyber Pro|Review scope and enforcement stay owner-controlled/);
});
