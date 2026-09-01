import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import { apiClient, updateStrategyForPlan } from '../modules/api_client.js';
import { updatePillText, verifiedUpdatePlan } from '../modules/update_status.js';
import {
    bindUpdateRefreshEvents,
    restartStatusCanSettle,
} from '../modules/updates.js';


class FakeWS {
    constructor() { this.listeners = new Map(); }
    on(type, handler) {
        this.listeners.set(type, handler);
        return () => this.listeners.delete(type);
    }
    emit(type, payload) { this.listeners.get(type)?.(payload); }
}


test('ordinary update selects clean or assisted merge without recovery replacement or client-side stash', () => {
    assert.equal(updateStrategyForPlan({
        available: true,
        kind: 'clean',
        local_dirty_count: 0,
        code_conflict_paths: [],
        doc_conflict_paths: [],
    }), 'auto_merge');
    assert.equal(updateStrategyForPlan({
        available: true,
        kind: 'conflicting',
        recommended_strategy: 'assisted',
        local_dirty_count: 2,
        code_conflict_paths: ['ouroboros/config.py'],
    }), 'assisted');
    assert.equal(updateStrategyForPlan({
        available: true,
        kind: 'clean',
        recommended_strategy: 'assisted',
        local_dirty_count: 0,
        code_conflict_paths: [],
        doc_conflict_paths: [],
    }), 'auto_merge');
    assert.equal(updateStrategyForPlan({ available: true, kind: 'unknown' }), '');
    assert.equal(updateStrategyForPlan({ available: false, kind: 'clean' }), '');

    const source = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8');
    const ordinary = source.slice(
        source.indexOf('async function applyUpdate()'),
        source.indexOf('async function replaceWithOfficial()'),
    );
    assert.doesNotMatch(ordinary, /['"]replace['"]/);
    assert.doesNotMatch(ordinary, /updateApply\(['"]stash['"]/);

    // The pill is a pointer, not a second apply surface (owner decision
    // 2026-08-31): it must never call updateApply or preflight itself.
    const pill = readFileSync(new URL('../modules/update_status.js', import.meta.url), 'utf8');
    assert.doesNotMatch(pill, /updateApply\(/);
    assert.doesNotMatch(pill, /updatePreflight\(/);
    assert.match(pill, /openDashboardTab\?\.\('updates'\)/);
});


test('update apply sends exact preflight pins and recovery confirmation only when asked', async () => {
    const originalFetch = globalThis.fetch;
    const calls = [];
    globalThis.fetch = async (url, init) => {
        calls.push({ url, init, body: JSON.parse(init.body) });
        return { ok: true, status: 200, json: async () => ({ status: 'ok' }) };
    };
    try {
        const plan = { base_sha: 'base123', target_sha: 'target456' };
        await apiClient.updateApply('auto_merge', plan);
        await apiClient.updateApply('replace', plan, { confirmRecovery: true });
    } finally {
        globalThis.fetch = originalFetch;
    }

    assert.deepEqual(calls[0].body, {
        strategy: 'auto_merge',
        expected_base_sha: 'base123',
        expected_target_sha: 'target456',
    });
    assert.deepEqual(calls[1].body, {
        strategy: 'replace',
        expected_base_sha: 'base123',
        expected_target_sha: 'target456',
        confirm_recovery: true,
    });
});


test('detailed Updates UI makes destructive replacement an explicit recovery action', () => {
    const source = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8');
    assert.match(source, />Save recovery point<\/button>/);
    assert.doesNotMatch(source, />Promote to QA<\/button>/);
    assert.doesNotMatch(source, />Promote to Stable<\/button>/);
    assert.match(source, /does not change the official QA feed/);
    assert.match(source, /<summary>Recovery<\/summary>/);
    assert.match(source, /Replace with Official Version \(Recovery\)/);
    assert.match(source, /apiClient\.updateApply\('replace', plan, \{ confirmRecovery: true \}\)/);
    assert.match(source, /updatePreflight\(\)/);
    assert.match(source, /data\.check_ok === false/);
    assert.match(source, /!data\.from_cache.*official_status_requires_check/);
    assert.match(source, /restart_required/);
    // Restart now is honest about refusals: apiFetch does not reject on 4xx/5xx.
    assert.match(source, /async function restartNow\(\)/);
    assert.match(source, /if \(!resp\.ok\) throw new Error/);
    assert.match(source, /Rollback completed:.*Restart Ouroboros to finish/s);
    assert.match(source, /Rollback failed:.*Runtime shutdown was incomplete/s);

    const pillSource = readFileSync(new URL('../modules/update_status.js', import.meta.url), 'utf8');
    assert.match(pillSource, /update_status_ready/);
});


test('same-version QA updates show commit identity in the main pill', () => {
    assert.equal(updatePillText({
        current_version: '6.87.6',
        latest_version: '6.87.6',
        current_sha: 'aaaaaaaa00000000',
        latest_sha: 'bbbbbbbb11111111',
    }), 'Update aaaaaaaa → bbbbbbbb');
});


test('main update dialog never invents facts for an unverified preflight', () => {
    assert.equal(verifiedUpdatePlan(null), null);
    assert.equal(verifiedUpdatePlan({ merge_plan: {
        available: true,
        kind: 'unknown',
        base_sha: 'base123',
        target_sha: 'target456',
        local_dirty_count: 0,
    } }), null);
    assert.deepEqual(verifiedUpdatePlan({ merge_plan: {
        available: true,
        kind: 'clean',
        base_sha: 'base123',
        target_sha: 'target456',
        local_dirty_count: 0,
    } }), {
        plan: {
            available: true,
            kind: 'clean',
            base_sha: 'base123',
            target_sha: 'target456',
            local_dirty_count: 0,
        },
        strategy: 'auto_merge',
    });

    // The verification helper still guards the one apply surface: the panel
    // imports it, and an unverified plan never reaches updateApply.
    const panel = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8');
    assert.match(panel, /verifiedUpdatePlan\(preflight\)/);
    assert.match(panel, /could not be verified\. No files were changed\./);
});

test('restart continuation and Replace gating are fail-closed in source', () => {
    const src = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8');
    // The panel-lifetime flag re-applies restart_needed on every status refresh.
    assert.match(src, /restartNeeded && !data\?\.update_tx\?\.active \? 'restart_needed' : ''/);
    // A failed status read keeps the continuation too.
    assert.match(src, /setPhase\(restartNeeded \? 'restart_needed' : ''\)/);
    // render() alone owns the Replace gate, failing closed on unreadable status
    // and on every busy/blocked state including restart_needed.
    assert.match(src, /statusReadFailed\(latestStatus\) \|\| \[\s*'loading', 'checking', 'updating', 'preflighting', 'restarting',\s*'restart_required', 'restart_needed', 'resolving', 'unmanaged',\s*\]/);
    // The catches never assign replaceBtn.disabled themselves.
    const catches = src.split('catch').slice(1);
    for (const c of catches) {
        assert.doesNotMatch(c.slice(0, 400), /replaceBtn\.disabled\s*=/);
    }
});

test('Replace carries an in-flight latch that render respects across re-renders', () => {
    const src = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8');
    assert.match(src, /replaceBtn\.disabled = replaceInFlight \|\| statusReadFailed/);
    // The latch opens before the preflight request and closes in finally.
    const fn = src.slice(src.indexOf('async function replaceWithOfficial'));
    const body = fn.slice(0, fn.indexOf('async function', 10));
    assert.match(body, /replaceInFlight = true;[\s\S]*updatePreflight/);
    assert.match(body, /finally \{\s*replaceInFlight = false;\s*render\(\);/);
});

test('restart refresh waits for a real reconnect before honoring boot status', () => {
    const ws = new FakeWS();
    let phase = 'restarting';
    const restartReads = [];
    let ordinaryReads = 0;
    const binding = bindUpdateRefreshEvents({
        ws,
        getPhase: () => phase,
        reconcileRestart: (options) => { restartReads.push(options); },
        loadStatus: () => { ordinaryReads += 1; },
    });

    binding.beginRestarting();
    ws.emit('update_status_ready');
    ws.emit('open', { previouslyConnected: false });
    assert.equal(restartReads.length, 0, 'an old boot signal or first open cannot settle a restart');

    ws.emit('open', { previouslyConnected: true });
    assert.deepEqual(restartReads, [{ afterBootNotice: false }], 'same-SHA reconnect rereads durable update status');
    ws.emit('update_status_ready');
    assert.deepEqual(restartReads, [
        { afterBootNotice: false },
        { afterBootNotice: true },
    ], 'boot finalization gets one typed post-reconnect reconciliation');

    binding.beginRestarting();
    ws.emit('update_status_ready');
    assert.equal(restartReads.length, 2, 'a new restart episode resets the reconnect proof');

    phase = '';
    ws.emit('open', { previouslyConnected: true });
    phase = 'restarting';
    ws.emit('update_status_ready');
    assert.equal(restartReads.length, 2, 'an idle reconnect cannot leak proof into a later restart');

    phase = '';
    ws.emit('update_status_ready');
    assert.equal(ordinaryReads, 1, 'ordinary ready refresh remains intact');
    binding.dispose();
    ws.emit('open', { previouslyConnected: true });
    ws.emit('update_status_ready');
    assert.equal(restartReads.length, 2, 'disposed restart listeners are inert');
    assert.equal(ordinaryReads, 1, 'disposed ordinary listener is inert');
});

test('restart reconciliation waits only for boot-owned transaction phases', () => {
    assert.equal(restartStatusCanSettle(null), false);
    assert.equal(restartStatusCanSettle('invalid'), false);
    assert.equal(restartStatusCanSettle({ warnings: ['status_error:down'] }), false);
    assert.equal(restartStatusCanSettle({}), true);
    assert.equal(restartStatusCanSettle({ update_tx: { active: false } }), true);
    assert.equal(restartStatusCanSettle({
        update_tx: { active: true, phase: 'pending_boot_smoke' },
    }), false);
    assert.equal(restartStatusCanSettle({
        update_tx: { active: true, phase: 'applying_replace' },
    }), false);
    assert.equal(restartStatusCanSettle({
        update_tx: { active: true, phase: 'pending_boot_smoke' },
    }, { afterBootNotice: true }), true);
    assert.equal(restartStatusCanSettle({
        update_tx: { active: true, phase: 'gate_blocked' },
    }), true);
    assert.equal(restartStatusCanSettle({
        update_tx: { active: true, phase: 'rolling_back' },
    }), true);
});

test('all successful restart paths share one lifecycle entry', () => {
    const source = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8');
    assert.equal((source.match(/setPhase\('restarting'\)/g) || []).length, 1);
    // Definition plus rollback, ordinary apply, recovery replace, and Restart now.
    assert.equal((source.match(/enterRestarting\(\)/g) || []).length, 5);
    assert.doesNotMatch(source, /restartNeeded = false;\s*enterRestarting\(\)/);
});
