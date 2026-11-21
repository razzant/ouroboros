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
const factsOf = (html) => html.split('data-skill-detail-facts>')[1]
    .split('<div class="skills-details-facts" data-skill-detail-publication=')[0];

test('a retained companion warns on a loaded skill alongside any independent failure', () => {
    const live = onExtension({ desired_live: true, live_loaded: true });
    const held = { ...live, retained_companions: { beta: 'stop requested' } };
    const warning = 'Waiting for beta to exit.';
    const card = renderInstalledSkillCard(held);
    assert.ok(traysOf(card).includes(warning));
    assert.doesNotMatch(card, /Couldn’t start|data-skill-action="repair"/);
    const failed = traysOf(renderInstalledSkillCard({ ...held, load_error: 'Alpha failed to import' }));
    assert.ok(failed.includes(warning) && failed.includes('Alpha failed to import'));
    assert.ok(!traysOf(renderInstalledSkillCard(live)).includes(warning));
});

test('every currently failed companion is named and escaped in Details, and recovery removes only its own name', () => {
    const failed = onExtension({ desired_live: true, live_loaded: true, dispatch_live: true, process: 'server',
        live_reason: 'companion_restart_exhausted:alpha,beta,gamma,delta,<epsilon>',
        load_error: 'Only the first diagnostic is visible before the names are needed' });
    const before = factsOf(renderInstalledSkillCard(failed));
    assert.match(before, /background process stopped: alpha, beta, gamma, delta, &lt;epsilon&gt;/);
    assert.doesNotMatch(before, /<epsilon>/);
    const recovered = factsOf(renderInstalledSkillCard({ ...failed,
        live_reason: 'companion_failed:beta,<epsilon>', load_error: 'beta; <epsilon>' }));
    assert.match(recovered, /background process stopped: beta, &lt;epsilon&gt;/);
    assert.doesNotMatch(recovered, /alpha|gamma|delta/);
    const ready = factsOf(renderInstalledSkillCard({ ...failed, live_reason: 'ready', load_error: null }));
    assert.doesNotMatch(ready, /background process stopped|alpha|beta|gamma|delta|epsilon/);
});

test('a pending companion replacement is neutral, not a repair or a Needs you fact', () => {
    const pending = onExtension({ desired_live: true, live_loaded: true, dispatch_live: true,
        live_reason: 'companion_refresh_pending', load_error: null });
    const model = installedSkillModel(pending);
    assert.match(model.chip, /Background restart pending/);
    assert.equal(model.needs, false);
    assert.match(renderInstalledSkillCard(pending), /Background restart pending/);
    assert.doesNotMatch(renderInstalledSkillCard(pending), /data-skill-action="repair"|data-needs-you/);
});

test('only a typed companion failure on an already loaded skill is a stopped background process', () => {
    const loaded = onExtension({ desired_live: true, live_loaded: true, dispatch_live: true, process: 'server', load_error: 'Daemon unavailable' });
    for (const live_reason of ['companion_failed:daemon', 'companion_restart_exhausted:daemon']) {
        assert.match(installedSkillModel({ ...loaded, live_reason }).chip, /Background process stopped/);
        assert.match(renderInstalledSkillCard({ ...loaded, live_reason }), />Loaded · server · background process stopped: daemon</);
    }
    assert.match(renderInstalledSkillCard({ ...loaded, live_reason: 'ready', load_error: null }), />Loaded · server</);
    assert.match(installedSkillModel({ ...loaded, live_reason: 'load_error' }).chip, /Couldn’t start/);
    assert.match(installedSkillModel({ ...loaded, live_loaded: false, live_reason: 'companion_failed:daemon' }).chip, /Couldn’t start/);
});

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
        await apiClient.skillMcpConnect('demo', { expected_content_hash: 'hash', settings_digest: 'digest' });
        assert.equal(requests[0].init.headers.Prefer, 'respond-async');
        assert.equal(requests[1].init.headers.Prefer, 'respond-async');
        assert.equal(requests[2].init.headers.Prefer, undefined, 'plain dictation POST stays terminal');
        assert.equal(requests[3].init.headers.Prefer, 'respond-async', 'Connect tools follows its job when queued');
        assert.equal(JSON.parse(requests[0].init.body).enable_after_grant, true);
    } finally {
        globalThis.fetch = original;
    }
});

test('a queued Connect tools job: the card follows it and says its outcome in the owner\'s words', () => {
    const server = { id: 'chrome_browser', title: 'Chrome (shared tab)', conflict: false, configured: false, tool_count: 0, listed: false, error: '' };
    const ready = onExtension({ type: 'script', mcp_servers: [server] });
    const running = renderInstalledSkillCard(mergeLifecycleEvents([ready], [{ id: 'job', target: 'demo', kind: 'connect_mcp',
        status: 'running', message: 'running' }])[0]);
    assert.match(running, /Connecting tools…/);
    assert.doesNotMatch(running, /data-skill-action="connect_tools"|configured ·/, 'no second confirm and no early success');
    const refusal = 'The MCP settings changed while connecting, so nothing was written. Open Connect tools again.';
    const failed = renderInstalledSkillCard(mergeLifecycleEvents([ready], [{ id: 'job', target: 'demo', kind: 'connect_mcp',
        status: 'failed', error: refusal }])[0]);
    assert.match(failed, /Connect tools failed/);
    assert.ok(failed.includes(refusal), 'the job\'s own refusal is shown');
    assert.doesNotMatch(failed, /Connect_mcp|connect_mcp failed/);
    const settled = renderInstalledSkillCard(mergeLifecycleEvents([onExtension({ type: 'script',
        mcp_servers: [{ ...server, configured: true }] })], [{ id: 'job', target: 'demo', kind: 'connect_mcp', status: 'succeeded' }])[0]);
    assert.match(settled, /Chrome \(shared tab\): configured · listing its tools/);
    assert.doesNotMatch(settled, /Connecting tools…|Connect tools failed/);
});

test('a failed Connect tools keeps its own retry on every reload, and a successful retry stops looking failed', () => {
    const server = { id: 'chrome_browser', title: 'Chrome (shared tab)', conflict: false, configured: false, tool_count: 0, listed: false, error: '' };
    const ready = onExtension({ type: 'script', mcp_servers: [server] });
    const refusal = 'The MCP settings changed while connecting, so nothing was written. Open Connect tools again.';
    const failedJob = { id: 'job-1', target: 'demo', kind: 'connect_mcp', status: 'failed', error: refusal };
    // The queue keeps the failed job: every listing read (a refresh, a reopened page) merges it again.
    for (const read of ['first', 'refresh', 'reopened']) {
        const row = mergeLifecycleEvents([ready], [failedJob])[0];
        const html = renderInstalledSkillCard(row);
        assert.match(html, /Connect tools failed/, read);
        assert.ok(html.includes(refusal), `${read}: the earlier failure stays visible`);
        assert.match(traysOf(html), /data-skill-action="connect_tools"[^>]*>Connect tools</, `${read}: the stated retry is reachable`);
        assert.equal(installedSkillModel(row).needs, true, `${read}: the failure still asks for the owner`);
    }
    const retrying = renderInstalledSkillCard(mergeLifecycleEvents([ready], [failedJob,
        { id: 'job-2', target: 'demo', kind: 'connect_mcp', status: 'running', message: 'running' }])[0]);
    assert.match(retrying, /Connecting tools…/);
    assert.doesNotMatch(retrying, /Connect tools failed|data-skill-action="connect_tools"/);
    const succeeded = renderInstalledSkillCard(mergeLifecycleEvents([onExtension({ type: 'script',
        mcp_servers: [{ ...server, configured: true }] })], [failedJob,
        { id: 'job-2', target: 'demo', kind: 'connect_mcp', status: 'succeeded' }])[0]);
    assert.match(succeeded, /Chrome \(shared tab\): configured · listing its tools/);
    assert.doesNotMatch(succeeded, /Connect tools failed|data-skill-action="connect_tools"|data-needs-you/);
    assert.ok(!succeeded.includes(refusal));
});

test('a failed job never opens Connect past the skill’s own blockers, a running job or a manual server conflict', () => {
    const server = { id: 'chrome_browser', title: 'Chrome (shared tab)', conflict: false, configured: false, tool_count: 0, listed: false, error: '' };
    const failed = (overrides) => mergeLifecycleEvents([onExtension({ type: 'script', mcp_servers: [server], ...overrides })],
        [{ id: 'job', target: 'demo', kind: 'connect_mcp', status: 'failed', error: 'Nothing was written.' }])[0];
    const blocked = {
        'missing grants': failed({ grants: { all_granted: false, missing_permissions: ['net'] } }),
        'missing secrets': failed({ secret_config: { fields: [{ key: 'TOKEN', state: 'missing' }] } }),
        'not reviewed': failed({ review_status: 'pending', review_stale: true, review_gate: { executable_review: false, blocking_reason: 'review_stale' } }),
        'review blockers': failed({ review_status: 'blockers', review_gate: { executable_review: false, blocking_reason: 'review_blockers' } }),
        'load error': failed({ load_error: 'It could not start.' }),
        'packages': failed({ live_reason: 'deps_failed', load_error: 'Its packages did not install.' }),
        off: failed({ enabled: false }),
        untrusted: failed({ trust: { state: 'untrusted', detail: 'Not delivered by your organization.' } }),
    };
    for (const [label, row] of Object.entries(blocked)) {
        const html = renderInstalledSkillCard(row);
        assert.ok(html.includes('Nothing was written.'), `${label}: the failure stays visible`);
        assert.doesNotMatch(html, /data-skill-action="connect_tools"/, label);
        assert.equal(installedSkillModel(row).needs, true, label);
    }
    const pending = renderInstalledSkillCard(mergeLifecycleEvents([onExtension({ type: 'script', mcp_servers: [server] })],
        [{ id: 'job', target: 'demo', kind: 'grant', status: 'running', message: 'Granting…' }])[0]);
    assert.doesNotMatch(pending, /data-skill-action="connect_tools"/, 'a running job');
    const conflict = traysOf(renderInstalledSkillCard(failed({ mcp_servers: [{ ...server, conflict: true }] })));
    assert.match(conflict, /already has an MCP server named chrome_browser that isn’t this skill’s/);
    assert.doesNotMatch(conflict, /connect_tools/);
    const unusable = traysOf(renderInstalledSkillCard(failed({ mcp_servers: [{ ...server, configured: true, error: 'x' }] })));
    assert.match(unusable, /Its tools are configured but can’t be used now\./);
    assert.doesNotMatch(unusable, /connect_tools/);
});

test('a running review or pending update never stands in for the skill’s readiness before Connect', () => {
    // The listing's `enabled` is the saved switch; the gate, grants and load error are the current bytes' facts, and a
    // running review (skill_review.current) or a pending update names the card's state ahead of them.
    const server = { id: 'chrome_browser', title: 'Chrome (shared tab)', conflict: false, configured: false, tool_count: 0, listed: false, error: '' };
    const failedJob = [{ id: 'job', target: 'demo', kind: 'connect_mcp', status: 'failed', error: 'Nothing was written.' }];
    const running = { skill_review: { current: { lifecycle_status: 'running', started_at: '' } } };
    const pendingUpdate = (extra = {}) => ({ install: { source: { id: 'hub', name: 'Hub', kind: 'hub' }, release: { version: '1.0.0' },
        pending_update: { version: '1.1.0' } }, ...extra });
    const card = (overrides, events = failedJob) => renderInstalledSkillCard(mergeLifecycleEvents(
        [onExtension({ type: 'script', mcp_servers: [server], ...overrides })], events)[0]);
    const stale = { review_status: 'pending', review_stale: true, review_gate: { executable_review: false, blocking_reason: 'review_stale' } };
    const shadowed = {
        'review running, gate refused': { ...running, ...stale },
        'review running, access missing': { ...running, grants: { all_granted: false, missing_permissions: ['net'] } },
        'review running, start failed': { ...running, load_error: 'It could not start.' },
        'update pending, access missing': pendingUpdate({ grants: { all_granted: false, missing_permissions: ['net'] } }),
        'update pending, gate refused': pendingUpdate(stale),
    };
    for (const [label, overrides] of Object.entries(shadowed)) {
        for (const events of [failedJob, []]) {
            assert.doesNotMatch(card(overrides, events), /data-skill-action="connect_tools"/, `${label} (${events.length ? 'after a failed job' : 'no job'})`);
        }
    }
    // A further review or a staged update of bytes that may run now does not take Connect away.
    for (const [label, overrides] of Object.entries({ 'review running': running, 'update pending': pendingUpdate() })) {
        const html = card(overrides);
        assert.match(html, /Reviewing…/, label);
        assert.match(traysOf(html), /data-skill-action="connect_tools"[^>]*>Connect tools</, label);
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
