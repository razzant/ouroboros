// The Skills card and the dependency owner (sprint 2026-10-19): an ON skill whose packages are
// missing, failed or out of date offers "Install packages" (= Enable, idempotent) instead of a
// repair; a declaration mismatch keeps the repair path; a pending owner action's chip shows the
// job's own message; the menu renders exactly the removal the host names. Every guard asserts
// both directions.
import assert from 'node:assert/strict';
import test from 'node:test';

import { installedSkillModel, renderInstalledSkillCard } from '../modules/skill_card_renderer.js';
import { mergeLifecycleEvents } from '../modules/skills.js';
import { apiClient, jsonPost } from '../modules/api_client.js';

function onExtension(overrides = {}) {
    return {
        name: 'demo', type: 'extension', version: '1.0.0', description: 'Demo skill', enabled: true,
        source: 'ouroboroshub', payload_root: 'skills/ouroboroshub/demo', review_status: 'clean', review_stale: false,
        review_gate: { executable_review: true, blocking_reason: 'ready' }, executable_review: true, permissions: [],
        grants: { all_granted: true }, owner_attestable: false, desired_live: false, live_loaded: false,
        install: { source: { id: 'hub', name: 'Hub', kind: 'hub' }, release: { version: '1.0.0' } }, removal: 'uninstall',
        ...overrides,
    };
}

const traysOf = (html) => html.split('data-skill-trays>')[1].split('<div data-skill-errors>')[0];

test('only pending-aware skill consumers opt into early HTTP answers', async () => {
    const original = globalThis.fetch;
    const requests = [];
    globalThis.fetch = async (url, init) => {
        requests.push({ url, init });
        return { ok: true, json: async () => ({ ok: true }) };
    };
    try {
        await apiClient.skillGrants('demo', ['inject_chat'], 'hash', true);
        await apiClient.deleteSkill('demo', 'skills/external/demo', 'hash');
        await jsonPost('/api/skills/browser_speech/toggle', { enabled: true });
        assert.equal(requests[0].init.headers.Prefer, 'respond-async');
        assert.equal(requests[1].init.headers.Prefer, 'respond-async');
        assert.equal(requests[2].init.headers.Prefer, undefined, 'plain dictation POST stays terminal');
        assert.equal(JSON.parse(requests[0].init.body).enable_after_grant, true);
    } finally {
        globalThis.fetch = original;
    }
});

test('a saved grant and failed packages do not become Grant failed', () => {
    const rows = mergeLifecycleEvents([onExtension({ enabled: false, live_reason: 'deps_failed',
        load_error: 'Its packages did not install.' })], [{ target: 'demo', kind: 'grant', status: 'succeeded',
        message: 'Access granted; Its packages did not install.' }]);
    const html = renderInstalledSkillCard(rows[0]);
    assert.doesNotMatch(html, /Grant failed|Needs access/);
    assert.match(html, /Install packages/);
});

test('an ON skill whose packages are missing, failed or stale offers Install packages (Enable), never a repair', () => {
    for (const reason of ['deps_missing', 'deps_failed', 'deps_stale']) {
        const html = renderInstalledSkillCard(onExtension({ live_reason: reason, load_error: 'Its packages are not installed.' }));
        const trays = traysOf(html);
        assert.match(trays, /Its packages are not installed\./, reason);
        assert.match(trays, /data-skill-action="turn_on"[^>]*>Install packages</, reason);
        assert.doesNotMatch(trays, /data-skill-action="repair"/, reason);
    }
});

test('a declaration mismatch stays on the repair path and a healthy ON skill offers neither', () => {
    const desync = traysOf(renderInstalledSkillCard(onExtension({ live_reason: 'deps_declaration_desync', load_error: 'Its package list does not match its files.' })));
    assert.match(desync, /data-skill-action="repair"[^>]*>Ask Ouroboros to fix</);
    assert.doesNotMatch(desync, /Install packages/);
    const healthy = renderInstalledSkillCard(onExtension({ live_reason: 'ready', live_loaded: true, desired_live: true }));
    assert.doesNotMatch(healthy, /Install packages|data-skill-action="repair"/);
});

test('a pending owner action shows the job’s own message; other jobs keep their label', () => {
    const chip = (skill) => installedSkillModel(skill).chip;
    assert.match(chip(onExtension({ lifecycle_pending: true, lifecycle_kind: 'enable', lifecycle_message: 'Installing packages…' })), />Installing packages…</);
    assert.match(chip(onExtension({ lifecycle_pending: true, lifecycle_kind: 'enable', lifecycle_message: '' })), />Turning on…</);
    assert.match(chip(onExtension({ lifecycle_pending: true, lifecycle_kind: 'review', lifecycle_message: 'Reviewing demo' })), />Reviewing…</);
    const merged = mergeLifecycleEvents([onExtension()], [{ target: 'demo', status: 'running', kind: 'enable', message: 'Installing packages…' }]);
    assert.equal(merged[0].lifecycle_message, 'Installing packages…');
    assert.equal(merged[0].lifecycle_pending, true);
});

test('the menu renders exactly the removal the host names', () => {
    const menu = (removal) => renderInstalledSkillCard(onExtension({ removal })).split('skills-card-menu-dialog')[1];
    assert.match(menu('uninstall'), /skills-uninstall"[^>]*>Uninstall</);
    assert.doesNotMatch(menu('uninstall'), /skills-delete-local/);
    assert.match(menu('delete'), /skills-delete-local"[^>]*>Delete</);
    assert.doesNotMatch(menu('delete'), /skills-uninstall/);
    assert.doesNotMatch(menu(''), /skills-uninstall|skills-delete-local/);
    assert.doesNotMatch(menu(undefined), /skills-uninstall|skills-delete-local/, 'no field, no derivation of its own');
});
