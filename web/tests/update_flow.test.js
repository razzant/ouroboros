import assert from 'node:assert/strict';
import test from 'node:test';
import { readdirSync, readFileSync } from 'node:fs';

import { apiClient, updateStrategyForPlan } from '../modules/api_client.js';
import { updatePillDetail, updatePillText, verifiedUpdatePlan } from '../modules/update_status.js';
import {
    applyFailureText,
    updateVerdict,
    bindUpdateRefreshEvents,
    initUpdates,
    restartStatusCanSettle,
} from '../modules/updates.js';


test('update failure renders the rollback stash disclosure without losing diagnostics', () => {
    const stash = 'a'.repeat(40);
    const note = `rolled back to 123456789abc; conflicting paths: "draft\\nname.txt"; `
        + `recover with \`git stash apply ${stash}\`\nGit diagnostics:\nCONFLICT from stdout`;
    const rendered = applyFailureText({ message: 'smoke failed', body: {
        rolled_back: true, rollback: note, stash_note: note,
    } });
    assert.ok(rendered.includes(note));
    // The update, not necessarily the checkout: a rollback may verify its target already present.
    assert.ok(rendered.includes('the update was rolled back') && !rendered.includes('checkout was rolled back'));
    assert.equal(rendered.split(`git stash apply ${stash}`).length - 1, 1);
    const pending = applyFailureText({ message: 'write failed', body: {
        reason: 'stash_recovery_incomplete', rolled_back: false,
        stash_note: 'Automatic recovery is unconfirmed; current files were left unchanged.',
    } });
    assert.ok(pending.includes('unconfirmed'));
    assert.ok(!pending.includes('was rolled back'));
});


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
    assert.match(pill, /openDashboardTab\('updates'\)/);
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
    assert.match(source, /Rollback \$\{data\.switch_deferred \? 'recorded' : 'completed'\}:.*Restart Ouroboros to finish/s);
    assert.match(source, /Rollback \$\{data\.switch_deferred \? 'requested' : 'successful'\}:.*Server is restarting/s);
    assert.match(source, /Rollback failed:.*Runtime shutdown was incomplete/s);

    const pillSource = readFileSync(new URL('../modules/update_status.js', import.meta.url), 'utf8');
    assert.match(pillSource, /update_status_ready/);
});


test('the main pill says "Update available"; the commit identity is its title (owner Q6=A)', () => {
    const qa = { current_version: '6.87.6', latest_version: '6.87.6', current_sha: 'aaaaaaaa00000000', latest_sha: 'bbbbbbbb11111111' };
    assert.equal(updatePillText(qa), 'Update available');
    assert.equal(updatePillDetail(qa), 'Update aaaaaaaa → bbbbbbbb');
    assert.equal(updatePillText({ current_version: '7.6.0', latest_version: '7.6.1' }), 'Update available');
    assert.equal(updatePillDetail({ current_version: '7.6.0', latest_version: '7.6.1' }), 'Update 7.6.0 → 7.6.1');
    assert.equal(updatePillDetail({}), 'Update available');
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

test('Replace outcomes say a restart was requested, never that the version already landed', () => {
    const src = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8');
    const fn = src.slice(src.indexOf('async function replaceWithOfficial'));
    const body = fn.slice(0, fn.indexOf('async function', 10));
    const toasts = [...body.matchAll(/showToast\('([^']*)'/g)].map((match) => match[1]);
    assert.deepEqual(toasts, [
        'Recovery needs a restart that could not be requested. Restart Ouroboros to finish.',
        'Restart requested to finish recovery with the official version.',
        'Recovery failed: ',
    ]);
    assert.ok(toasts.every((text) => !/restored|landed/i.test(text)));
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
    assert.equal(ordinaryReads, 2, 'idle reconnect and ordinary ready both refresh status');
    binding.dispose();
    ws.emit('open', { previouslyConnected: true });
    ws.emit('update_status_ready');
    assert.equal(restartReads.length, 2, 'disposed restart listeners are inert');
    assert.equal(ordinaryReads, 2, 'disposed ordinary listener is inert');
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
    // Definition plus rollback, the in-place apply, recovery replace, Restart now, and Restart to update.
    assert.equal((source.match(/enterRestarting\(\)/g) || []).length, 6);
    assert.doesNotMatch(source, /restartNeeded = false;\s*enterRestarting\(\)/);
});


test('pending local-work recovery never offers the refused in-app Restart', () => {
    const ordinary = { update_tx: { active: true, phase: 'pending_boot_smoke' } };
    assert.equal(updateVerdict(ordinary).action.id, 'restart');
    const pending = { update_tx: { ...ordinary.update_tx, local_work_recovery: true } };
    for (const phase of ['', 'restarting', 'restart_needed', 'restart_required']) {
        const view = updateVerdict(pending, phase);
        assert.equal(view.action.id, 'check');
        assert.match(view.hint, /Quit and reopen/);
        assert.match(view.hint, /restart the server process/);
        assert.match(view.hint, /stash is retained/);
    }
});


// --- The unified Update: prepared beside the running app (UPDATE-TECHNICAL-PLAN §5.6, §6.1, §10 item 17) -------
// Invented values: 7.8.2, the hashes and the paths. The projection's shape is supervisor/update_prepare.py's.

const SHA = { base: '88acfdbe3c1f0a2b4d5e6f708192a3b4c5d6e7f8', target: 'd2f6a91c7e5b4a3928170f6e5d4c3b2a19087f6e',
    newer: 'e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4' };
const STATUS = {
    managed: true, check_ok: true, checked_at: new Date().toISOString(),
    current_version: '7.8.1', current_short_sha: '88acfdbe', current_sha: SHA.base,
    latest_version: '7.8.2', latest_short_sha: 'd2f6a91c', latest_sha: SHA.target,
    official_repo_url: 'https://github.com/razzant/ouroboros', target_ref: 'managed/ouroboros', update_channel: 'development',
    behind: 6, ahead: 0, dirty: false, dirty_count: 0, local_changes: 'absent', available: true, safe_to_apply: true,
    warnings: [], update_tx: { active: false }, update_progress: {},
};
const LOCAL = { ahead: 2, dirty: true, dirty_count: 3, local_changes: 'present', local_changes_base: '80a38cb9aa00', safe_to_apply: false };
const PASSED = { ok: true, tests: { file_count: 14, ok: true, blocking: [], already_failing: [] } };
// The carried layers as update_prepare.carried_layers records them: C -> I' (staged) and I' -> W' (the files).
const CARRIED = { index: { count: 1, paths: [['M', 'web/modules/interface_size.js']] },
    files: { count: 2, paths: [['A', 'notes/local-ideas.md'], ['M', 'supervisor/queue.py']] } };
const candidate = (fields) => ({ candidate_id: 'update_d2f6a91c7e5b_4f9a1c', reason: '', target_sha: SHA.target,
    target_ref: 'managed/ouroboros', target_version: '7.8.2', base_version: '7.8.1', channel: 'development',
    base_sha: SHA.base, local_changes: false, without_local_changes: false, merge_kind: 'fast_forward', local_commits: 0,
    carried: {}, conflict_paths: [], conflict_count: 0, layer_conflicts: {}, environment_same: true,
    checks: {}, activation: {}, newer_available: false, updated_at: new Date().toISOString(), actions: [], ...fields });
const buttons = (verdict) => [...(verdict.secondary || []).map((item) => item.label), verdict.action ? `*${verdict.action.label}` : '']
    .filter(Boolean);
const chip = (verdict, label) => verdict.chips.find((item) => item.label === label)?.value;

test('with a candidate projection the primary update action is Prepare: the update is built beside the running app', () => {
    const plain = updateVerdict({ ...STATUS, candidate: { state: 'none', actions: ['prepare'] } });
    assert.deepEqual([plain.state, plain.tone, buttons(plain)], ['available', 'ok', ['*Prepare update']]);
    assert.equal(plain.headline, 'Update available: 7.8.1 (88acfdbe) -> 7.8.2 (d2f6a91c)');
    assert.equal(plain.hint, 'Ouroboros prepares it in a separate copy; 7.8.1 keeps running until you restart.');
    const local = updateVerdict({ ...STATUS, ...LOCAL, candidate: { state: 'none', actions: ['prepare'] } });
    assert.deepEqual([local.state, local.tone, buttons(local)], ['available', 'ok', ['*Prepare update']],
        'local work is carried, so it is no caution and no second control');
    assert.equal(local.hint, 'Your local commits and 3 uncommitted file(s) will be carried into 7.8.2. Uncommitted files stay uncommitted.');
    assert.deepEqual([chip(local, 'Commits'), chip(local, 'Local changes'), chip(local, 'Uncommitted')],
        ['6 incoming / 2 local', 'present since 80a38cb9', '3 file(s)']);
    for (const verdict of [plain, local]) assert.doesNotMatch(verdict.hint, /stash/i);
    const refused = updateVerdict({ ...STATUS, candidate: { state: 'none', actions: ['prepare'], refusal: {
        code: 'UPDATE_LOCAL_STATE_UNSUPPORTED', text: 'local work cannot be carried as it is: branch:main' } } });
    assert.match(refused.hint, /^The last preparation was refused: local work cannot be carried as it is: branch:main\./);
    // A status without the projection (an older core) keeps the in-place path and its own words.
    assert.deepEqual(buttons(updateVerdict(STATUS)), ['*Update to 7.8.2']);
});

test('every candidate state has one verdict: its words, its buttons and its facts, from the payload alone', () => {
    const cases = [
        ['preparing (no copy yet)', { candidate: { state: 'preparing', actions: [] } },
            'candidate_preparing', 'Preparing the update…', /^7\.8\.1 keeps running\. Nothing in this installation changes until you restart\.$/, []],
        ['preparing', { candidate: candidate({ state: 'preparing' }) },
            'candidate_preparing', 'Preparing 7.8.2 (d2f6a91c)…', /keeps running/, []],
        ['checking', { ...LOCAL, candidate: candidate({ state: 'checking', merge_kind: 'clean', local_changes: true }) },
            'candidate_checking', 'Checking 7.8.2 (d2f6a91c)…', /its checks run in its own copy/, []],
        ['ready', { candidate: candidate({ state: 'ready', checks: PASSED, actions: ['activate', 'discard_candidate'] }) },
            'candidate_ready', '7.8.2 (d2f6a91c) is ready. Restart to switch to it.', /^If it does not start, 7\.8\.1 comes back/,
            ['Discard', '*Restart to update']],
        ['ready with local work', { ...LOCAL, candidate: candidate({ state: 'ready', merge_kind: 'clean', local_changes: true,
            local_commits: 2, carried: CARRIED, checks: PASSED, actions: ['activate', 'discard_candidate'] }) },
            'candidate_ready', '7.8.2 (d2f6a91c) is ready with your local changes. Restart to switch to it.',
            /^Your 2 local commit\(s\) are merged into it, and your 3 uncommitted file\(s\) are carried over and stay uncommitted\./,
            ['Discard', '*Restart to update']],
        // A record without its layers (no count to read) names the work, never a count it does not have.
        ['ready with local work, no recorded layers', { ...LOCAL, candidate: candidate({ state: 'ready', merge_kind: 'clean',
            local_changes: true, local_commits: 2, checks: PASSED, actions: ['activate', 'discard_candidate'] }) },
            'candidate_ready', '7.8.2 (d2f6a91c) is ready with your local changes. Restart to switch to it.',
            /^Your 2 local commit\(s\) are merged into it, and your staged, unstaged and untracked changes are carried over/,
            ['Discard', '*Restart to update']],
        ['ready, a newer version out', { latest_version: '7.8.3', latest_short_sha: 'e1f2a3b4', latest_sha: SHA.newer,
            candidate: candidate({ state: 'ready', checks: PASSED, newer_available: true, actions: ['activate', 'discard_candidate'] }) },
            'candidate_ready', '7.8.2 (d2f6a91c) is ready. Restart to switch to it.', /A newer version is available: 7\.8\.3 \(e1f2a3b4\)\./,
            ['Restart to update', '*Prepare 7.8.3']],
        ['ready, a newer version out, a record without its pinned version', { latest_version: '7.8.3',
            latest_short_sha: 'e1f2a3b4', latest_sha: SHA.newer, candidate: candidate({ state: 'ready', checks: PASSED,
                newer_available: true, target_version: '', actions: ['activate', 'discard_candidate'] }) },
            'candidate_ready', 'Update d2f6a91c is ready. Restart to switch to it.', /A newer version is available/,
            ['Restart to update', '*Prepare 7.8.3']],
        ['needs adaptation', { ...LOCAL, candidate: candidate({ state: 'needs_adaptation', reason: 'files_conflicts', merge_kind: 'clean',
            local_changes: true, layer_conflicts: { files: ['supervisor/queue.py'] }, conflict_paths: ['supervisor/queue.py'],
            conflict_count: 1, adaptation: { step: 'files' },
            actions: ['apply_without_local_changes', 'discard_candidate', 'prepare'] }) },
            'candidate_needs_adaptation', 'Your uncommitted changes conflict with 7.8.2 (d2f6a91c). Nothing was changed.',
            /^Conflicting: supervisor\/queue\.py\. 7\.8\.1 keeps running with your local changes\./,
            ['Prepare without local changes', 'Discard', '*Prepare again']],
        ['failed checks', { ...LOCAL, candidate: candidate({ state: 'failed', reason: 'checks_failed', checks: { ok: false,
            tests: { blocking: ['tests/test_queue.py'], already_failing: ['tests/test_net.py'] } },
            actions: ['prepare', 'discard_candidate'] }) },
            'candidate_failed', '7.8.2 (d2f6a91c) did not pass its checks. Nothing was changed.',
            /^Tests that fail only with the update: tests\/test_queue\.py\.$/, ['Discard', '*Prepare again']],
        // A run that did not complete verified nothing: never "did not pass", never ready.
        ['unverified checks', { ...LOCAL, candidate: candidate({ state: 'failed', reason: 'checks_unverified', checks: { ok: false,
            tests: { unverified: 'the selected run did not complete (parallel: exit 124, 0 case(s))', blocking: [] } },
            actions: ['prepare', 'discard_candidate'] }) },
            'candidate_failed', '7.8.2 (d2f6a91c) could not be verified. Nothing was changed.',
            /^Its checks did not complete, so it was neither passed nor failed\. Prepare it again\.$/, ['Discard', '*Prepare again']],
        ['failed', { candidate: candidate({ state: 'failed', reason: 'UPDATE_SPACE: 2.1 GB free, the copy needs 3.4 GB',
            actions: ['prepare', 'discard_candidate'] }) },
            'candidate_failed', '7.8.2 (d2f6a91c) could not be prepared. Nothing was changed.',
            /^2\.1 GB free, the copy needs 3\.4 GB\.$/, ['Discard', '*Prepare again']],
        ['stale', { ...LOCAL, candidate: candidate({ state: 'stale', reason: 'local_changed', actions: ['prepare', 'discard_candidate'] }) },
            'candidate_stale', 'This checkout changed after the update was prepared.', /^Prepare it again/, ['Discard', '*Prepare again']],
        ['activating', { candidate: candidate({ state: 'activating', activation: { adoption_id: 'a1' } }) },
            'candidate_activating', 'Switching to 7.8.2 (d2f6a91c)…', /updates itself/, []],
        ['returning', { candidate: candidate({ state: 'activating', activation: { outcome: 'returning' } }) },
            'candidate_returning', '7.8.2 (d2f6a91c) did not start; Ouroboros is returning to 7.8.1 (88acfdbe)…', /proven/, []],
        // Nothing was local: the headline names both versions and claims no local changes back (F6's control).
        ['activation returned', { candidate: candidate({ state: 'failed', reason: 'activation:returned', activation: {
            outcome: 'returned', sha: SHA.base, previous: SHA.target, why: 'supervisor_not_ready',
            decision: { trigger: 'return_requested' }, note: 'n' }, actions: ['prepare', 'discard_candidate'] }) },
            'candidate_returned', 'Ouroboros is back on 7.8.1 (88acfdbe). 7.8.2 (d2f6a91c) did not start.',
            /^It switched back on its own because 7\.8\.1 was proven to read your current data; nothing was rolled back\.$/,
            ['Discard', '*Prepare again']],
        ['activation returned with local changes', { ...LOCAL, candidate: candidate({ state: 'failed', local_changes: true,
            reason: 'activation:returned', activation: { outcome: 'returned', sha: SHA.base, previous: SHA.target,
                why: 'import_probe_failed', decision: { trigger: 'import_probe_failed' } }, actions: ['prepare', 'discard_candidate'] }) },
            'candidate_returned', 'Ouroboros is back on 7.8.1 (88acfdbe). 7.8.2 (d2f6a91c) did not start.',
            /^It switched back before the new version ran, so your data is as it was\. Your local changes are back as they were\.$/,
            ['Discard', '*Prepare again']],
        ['start failed, recovery', { available: false, candidate: candidate({ state: 'failed', reason: 'activation:start_failed',
            activation: { outcome: 'start_failed', sha: SHA.target, previous: SHA.base, reasons: ['engine_pin_changed'], note: 'n' },
            actions: ['prepare', 'discard_candidate'] }) },
            'candidate_start_failed', '7.8.2 (d2f6a91c) did not start correctly. Nothing was rolled back.',
            /could not prove that 7\.8\.1 can run on what is here now/, ['*Restore previous version']],
        ['not switched to', { candidate: candidate({ state: 'failed', reason: 'activation:not_applied', activation: {
            outcome: 'not_applied', note: 'Adoption of d2f6a91c7e5b was not applied (x).' }, actions: ['prepare', 'discard_candidate'] }) },
            'candidate_failed', '7.8.2 (d2f6a91c) was not switched to. Nothing was changed.', /^The restart ended before the switch/,
            ['Discard', '*Prepare again']],
    ];
    for (const [name, fields, state, headline, hint, labels] of cases) {
        const verdict = updateVerdict({ ...STATUS, ...fields });
        assert.equal(verdict.state, state, name);
        assert.equal(verdict.headline, headline, name);
        assert.match(verdict.hint, hint, name);
        assert.deepEqual(buttons(verdict), labels, name);
        assert.equal(chip(verdict, 'Official repo'), 'razzant/ouroboros', `${name}: the status chips stay`);
        assert.doesNotMatch(`${verdict.headline} ${verdict.hint}`, /stashed and restored/, name);
        // Codes and the server's notes stay in the record (the Last attempt's raw record), never in these words.
        assert.doesNotMatch(`${verdict.headline} ${verdict.hint}`,
            /(?:^|[\s(])[a-z]+_[a-z_]+(?=[\s).,;:]|$)|\b[A-Z]+_[A-Z_]+\b|\bAdoption of\b/, `${name}: no raw code`);
        assert.ok(!verdict.facts.some((row) => ['Record', 'Not proven', 'Running again'].includes(row.label)),
            `${name}: the restart's outcome is the Last attempt's, not this update's facts`);
    }
    const recovery = updateVerdict({ ...STATUS, candidate: candidate({ state: 'failed', activation: { outcome: 'start_failed',
        previous: SHA.base, reasons: ['engine_pin_changed', 'later_edits:web/app.js'] } }) });
    assert.deepEqual(recovery.action, { id: 'restore', label: 'Restore previous version', target: SHA.base,
        name: '7.8.1 (88acfdbe)' }, 'the existing Restore, naming the version the record started from');
    const adaptation = updateVerdict({ ...STATUS, candidate: candidate({ state: 'needs_adaptation', reason: 'merge_conflicts',
        merge_kind: 'conflicting', conflict_paths: ['ouroboros/config.py', 'supervisor/queue.py'], conflict_count: 2,
        layer_conflicts: { merge: ['ouroboros/config.py', 'supervisor/queue.py'] }, adaptation: { step: 'merge' },
        actions: ['discard_candidate', 'prepare'] }) });
    assert.equal(adaptation.headline, 'Your local commits conflict with 7.8.2 (d2f6a91c). Nothing was changed.');
    assert.deepEqual(adaptation.facts.find((row) => row.label === 'Conflicts with your local commits'),
        { label: 'Conflicts with your local commits', value: 'ouroboros/config.py\nsupervisor/queue.py', code: true },
        'plain paths');
    // F3: a conflict of the uncommitted files is never called a local-commit conflict, even under a clean merge.
    const files = updateVerdict({ ...STATUS, ...LOCAL, candidate: candidate({ state: 'needs_decision', merge_kind: 'clean',
        reason: 'adaptation_question', local_changes: true, conflict_paths: ['supervisor/queue.py'], conflict_count: 1,
        layer_conflicts: { files: ['supervisor/queue.py'] }, adaptation: { step: 'files', task_id: 'update_adapt_1' },
        question: 'Which should win?', actions: ['apply_without_local_changes', 'discard_candidate', 'prepare'] }) });
    const labels = files.facts.map((row) => row.label);
    assert.ok(labels.includes('Conflicts in your uncommitted files') && !labels.some((label) => /local commits/.test(label)),
        labels.join(' | '));
    assert.equal(files.facts.find((row) => row.label === 'Combined').value, 'merged with this installation’s own commits, without conflicts');
    const many = updateVerdict({ ...STATUS, ...LOCAL, candidate: candidate({ state: 'needs_adaptation', reason: 'index_conflicts',
        local_changes: true, merge_kind: 'clean', conflict_paths: ['a.py', 'b.py'], conflict_count: 25,
        layer_conflicts: { index: ['a.py', 'b.py'] }, adaptation: { step: 'index' },
        actions: ['apply_without_local_changes', 'discard_candidate', 'prepare'] }) });
    assert.equal(many.headline, 'Your staged changes conflict with 7.8.2 (d2f6a91c). Nothing was changed.');
    assert.match(many.hint, /^Conflicting: a\.py, b\.py and 23 more\./, 'a bounded list counts what it did not carry');
    assert.equal(many.facts.find((row) => row.label === 'Conflicts in your staged changes').value, 'a.py\nb.py\nand 23 more');
    assert.deepEqual(buttons(adaptation), ['Discard', '*Prepare again'], 'an action the server does not list is never offered');
    const unknown = updateVerdict({ ...STATUS, candidate: { state: 'unknown', error: 'OSError: registry unreadable' } });
    assert.equal(unknown.state, 'available');
    assert.deepEqual(buttons(unknown), ['*Check again'], 'an unreadable record offers no Prepare');
    assert.match(unknown.warnings.join(' '), /prepared update could not be read: OSError/);
});

test('a start that failed leads to recovery: what failed, the kept paths and the Restore list', () => {
    const failure = { cause: 'import_failed', composition: 'unverified', rescue: '/data/archive/rescue/20261027-141205',
        paths: [' M supervisor/queue.py', '?? notes/local-ideas.md'], detail: 'SyntaxError: expected \':\'', at: new Date().toISOString() };
    const verdict = updateVerdict({ ...STATUS, available: false, candidate: { state: 'none', actions: ['prepare'], start_failure: failure } });
    assert.equal(verdict.state, 'start_failed');
    assert.equal(verdict.headline, 'The last start failed: this checkout’s code did not import.');
    assert.match(verdict.hint, /never reached a successful start break it \(2 path\(s\)\)\. They stay in place and were copied to \/data\/archive\/rescue\/20261027-141205/);
    assert.deepEqual(verdict.action, { id: 'recovery', label: 'Choose a version to restore' });
    assert.deepEqual(verdict.secondary, [], 'recovery is the one thing here: no Prepare beside it');
    const kept = verdict.facts.find((row) => row.label === 'Kept in place');
    assert.equal(kept.value, ' M supervisor/queue.py\n?? notes/local-ideas.md');
    assert.equal(verdict.facts.find((row) => row.label === 'Import error').value, failure.detail);
    const healthy = updateVerdict({ ...STATUS, candidate: { state: 'none', start_failure: { cause: 'import_failed', composition: 'healthy' } } });
    assert.match(healthy.hint, /started successfully before, so something outside them changed/);
    // A failed import outside the keep policy: the checkout stays, no older branch starts on its own, Restore chooses.
    const plain = updateVerdict({ ...STATUS, candidate: { state: 'none', start_failure: { cause: 'import_failed',
        detail: 'ImportError: boom' } } });
    assert.equal(plain.state, 'start_failed');
    assert.equal(plain.hint, 'Restore starts a version you choose.');
    assert.deepEqual(plain.action, { id: 'recovery', label: 'Choose a version to restore' });
    assert.equal(plain.facts.find((row) => row.label === 'Not proven'), undefined, 'no automatic fallback is described');
});

test('a request in flight holds every control of its verdict; Restart to update has its own words', () => {
    const data = { ...STATUS, candidate: candidate({ state: 'needs_adaptation', reason: 'files_conflicts', local_changes: true,
        layer_conflicts: { files: ['a.py'] }, actions: ['apply_without_local_changes', 'discard_candidate', 'prepare'] }) };
    const held = updateVerdict(data, 'requesting');
    assert.equal(held.headline, updateVerdict(data).headline);
    assert.ok([held.action, ...held.secondary].every((item) => item.disabled === true));
    const ready = { ...STATUS, candidate: candidate({ state: 'ready', actions: ['activate', 'discard_candidate'] }) };
    const activating = updateVerdict(ready, 'activating');
    assert.deepEqual([activating.state, activating.action], ['activating', null]);
    assert.equal(activating.headline, 'Preparing the restart into 7.8.2 (d2f6a91c)…');
    const switching = { ...STATUS, candidate: candidate({ state: 'activating' }) };
    assert.equal(updateVerdict(switching, 'restarting').headline, 'Restarting into 7.8.2 (d2f6a91c)…');
    assert.equal(updateVerdict(STATUS, 'restarting').headline, 'Restarting the server…');
});

test('the candidate actions post their strategy and the candidate the panel showed', async () => {
    const originalFetch = globalThis.fetch;
    const calls = [];
    globalThis.fetch = async (url, init) => {
        calls.push({ url, body: JSON.parse(init.body) });
        return { ok: true, status: 202, json: async () => ({ status: 'preparing' }) };
    };
    try {
        await apiClient.updateCandidate('prepare');
        await apiClient.updateCandidate('activate', 'update_d2f6a91c7e5b_4f9a1c');
    } finally {
        globalThis.fetch = originalFetch;
    }
    assert.deepEqual(calls.map((call) => [call.url, call.body]), [
        ['/api/update/apply', { strategy: 'prepare' }],
        ['/api/update/apply', { strategy: 'activate', candidate_id: 'update_d2f6a91c7e5b_4f9a1c' }],
    ]);
});

test('the panel reads only candidate fields the server projection emits', () => {
    const py = readFileSync(new URL('../../supervisor/update_prepare.py', import.meta.url), 'utf8');
    const fn = py.slice(py.indexOf('def projection('));
    assert.ok(fn.length > 100, 'projection() moved — update this test');
    const emitted = new Set([...fn.slice(0, fn.indexOf('\n\n\n') > 0 ? fn.indexOf('\n\n\n') : undefined)
        .matchAll(/"([a-z_]+)":/g)].map((match) => match[1]));
    for (const key of ['state', 'actions', 'start_failure', 'refusal', 'error']) assert.ok(emitted.has(key), key);
    const source = readFileSync(new URL('../modules/update_candidate.js', import.meta.url), 'utf8');
    const read = new Set([...source.matchAll(/\bcandidate\.([a-z_]+)/g)].map((match) => match[1]));
    assert.ok(read.size >= 12, 'no candidate reads found — the regex or the module moved');
    assert.deepEqual([...read].filter((field) => !emitted.has(field)), [], 'update_candidate.js reads fields never emitted');
});

test('only updates.js asks for the candidate verdict, and the pill never prepares or activates', () => {
    const modules = new URL('../modules/', import.meta.url);
    const files = readdirSync(modules).filter((name) => name.endsWith('.js'));
    const callers = files.filter((name) => /\bcandidateVerdict\b/.test(readFileSync(new URL(name, modules), 'utf8')));
    assert.deepEqual(callers.sort(), ['update_candidate.js', 'updates.js']);
    for (const name of files.filter((file) => !['updates.js', 'api_client.js'].includes(file))) {
        assert.doesNotMatch(readFileSync(new URL(name, modules), 'utf8'), /updateCandidate\(/, `${name} sends an Update action`);
    }
    for (const name of ['updates.js', 'update_candidate.js', 'update_status.js']) {
        assert.doesNotMatch(readFileSync(new URL(name, modules), 'utf8'), /stashed and restored/, name);
    }
});

// A DOM double just large enough for initUpdates: every element answers every selector with its own stub.
function stubElement(tag = 'div') {
    const nodes = new Map();
    const el = {
        tagName: tag.toUpperCase(), hidden: false, disabled: false, open: false, textContent: '', dateTime: '', value: '',
        dataset: {}, style: {}, children: [], listeners: {},
        get childElementCount() { return this.children.length; },
        get parentElement() { return this._parent || (this._parent = stubElement()); },
        get innerHTML() { return this._html || ''; },
        set innerHTML(value) { this._html = String(value); this.children = []; },
        addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
        removeEventListener() {},
        setAttribute() {}, removeAttribute() {}, getAttribute() { return null; }, hasAttribute() { return false; },
        toggleAttribute() {},
        appendChild(child) { this.children.push(child); return child; },
        insertBefore(child) { this.children.push(child); return child; },
        remove() {}, focus() {}, scrollIntoView() {}, closest() { return null; }, contains() { return false; },
        get ownerDocument() { return globalThis.document; },
        querySelector(selector) { if (!nodes.has(selector)) nodes.set(selector, stubElement()); return nodes.get(selector); },
        querySelectorAll() { return []; },
        classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    };
    return el;
}

test('navigation to Updates reads the status and applies nothing, whatever the candidate offers', async () => {
    const statuses = [
        { ...STATUS, candidate: { state: 'none', actions: ['prepare'] } },
        { ...STATUS, candidate: candidate({ state: 'ready', checks: PASSED, actions: ['activate', 'discard_candidate'] }) },
        { ...STATUS, candidate: candidate({ state: 'needs_adaptation', reason: 'files_conflicts', local_changes: true,
            layer_conflicts: { files: ['a.py'] }, actions: ['apply_without_local_changes', 'discard_candidate', 'prepare'] }) },
        { ...STATUS, candidate: { state: 'none', start_failure: { cause: 'import_failed', composition: 'healthy' } } },
    ];
    const saved = { fetch: globalThis.fetch, window: globalThis.window, document: globalThis.document };
    try {
        for (const status of statuses) {
            const calls = [];
            const shown = [];
            globalThis.fetch = async (url, init = {}) => {
                calls.push(`${init.method || 'GET'} ${url}`);
                const body = String(url).startsWith('/api/update/status') ? status : { branch: 'ouroboros', sha: '88acfdbe', commits: [], tags: [] };
                return { ok: true, status: 200, json: async () => body };
            };
            globalThis.window = { addEventListener: (type, fn) => { if (type === 'ouro:dashboard-subtab-shown') shown.push(fn); } };
            globalThis.document = { createElement: (tag) => stubElement(tag) };
            const page = { appended: [], appendChild(child) { this.appended.push(child); } };
            initUpdates({ mount: page, state: { activePage: 'dashboard' }, ws: new FakeWS(), openSettingsTab() {} });
            assert.equal(calls.length, 0, 'mounting reads nothing');
            shown.forEach((fn) => fn({ detail: { tab: 'updates' } }));
            await new Promise((resolve) => setTimeout(resolve, 0));
            await new Promise((resolve) => setTimeout(resolve, 0));
            assert.deepEqual(calls.sort(), ['GET /api/git/log', 'GET /api/update/status'], JSON.stringify(status.candidate));
        }
    } finally {
        Object.assign(globalThis, saved);
    }
});

// The real page: its click handler, the real confirm dialog, a status refresh while the dialog is open.
async function confirmActivation({ refreshTo = null, reply = { status: 'ok', restarting: true } }) {
    const ready = (id, version) => ({ ...STATUS, candidate: candidate({ candidate_id: id, state: 'ready', checks: PASSED,
        target_version: version, actions: ['activate', 'discard_candidate'] }) });
    let current = ready('update_aaaaaaaaaaaa_000001', '7.8.2');
    const posts = [];
    const saved = { fetch: globalThis.fetch, window: globalThis.window, document: globalThis.document };
    const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
    try {
        globalThis.fetch = async (url, init = {}) => {
            if ((init.method || 'GET') === 'POST') posts.push([String(url), JSON.parse(init.body || '{}')]);
            const body = String(url).startsWith('/api/update/status') ? current
                : String(url).startsWith('/api/update/apply') ? reply
                    : { branch: 'ouroboros', sha: '88acfdbe', commits: [], tags: [] };
            return { ok: true, status: 200, json: async () => body };
        };
        const shown = [];
        globalThis.window = { addEventListener: (type, fn) => { if (type === 'ouro:dashboard-subtab-shown') shown.push(fn); } };
        globalThis.document = { createElement: (tag) => stubElement(tag), body: stubElement('body'), activeElement: null,
            addEventListener() {}, removeEventListener() {}, getElementById: () => null };
        const page = { appended: [], appendChild(child) { this.appended.push(child); } };
        initUpdates({ mount: page, state: { activePage: 'dashboard' }, ws: new FakeWS(), openSettingsTab() {} });
        shown.forEach((fn) => fn({ detail: { tab: 'updates' } }));
        for (let i = 0; i < 4; i += 1) await tick();
        const actions = page.appended[0].querySelector('#updates-actions');
        actions.listeners.click[0]({ target: { closest: () => ({ disabled: false, dataset: { action: 'activate' } }) } });
        await tick();
        const dialog = globalThis.document.body.children.at(-1);
        if (refreshTo) {  // a progress refresh replaces the status while the owner reads the dialog
            current = ready(refreshTo, '7.8.3');
            shown.forEach((fn) => fn({ detail: { tab: 'updates' } }));
            for (let i = 0; i < 4; i += 1) await tick();
        }
        dialog.listeners.click[0]({ target: { closest: (selector) => (selector === '[data-confirm-ok]' ? {} : null) } });
        for (let i = 0; i < 6; i += 1) await tick();
        const sent = posts.filter(([url]) => url.startsWith('/api/update/apply'));
        sent.headline = page.appended[0].querySelector('#updates-summary').textContent;
        return sent;
    } finally {
        Object.assign(globalThis, saved);
    }
}

test('Restart to update sends the candidate the dialog described, and never one that replaced it meanwhile', async () => {
    const unchanged = await confirmActivation({});
    assert.deepEqual([...unchanged], [['/api/update/apply', { strategy: 'activate', candidate_id: 'update_aaaaaaaaaaaa_000001' }]]);
    const replaced = await confirmActivation({ refreshTo: 'update_bbbbbbbbbbbb_000002' });
    assert.deepEqual([...replaced], [], 'A was confirmed while B became current: nothing is sent, B is shown again');
});

test('the restart names the candidate the server activates, from its reply, not the status read before it', async () => {
    // The server marks the candidate activating and returns it (update_activation._activate_fenced): that reply is
    // what the restart headline names. A reply naming another candidate, or none (an older core), is not merged.
    const activating = (id) => ({ status: 'ok', restarting: true, strategy: 'activate', candidate: candidate({
        candidate_id: id, state: 'activating', target_version: '7.8.2', activation: { adoption_id: 'a1' } }) });
    const named = await confirmActivation({ reply: activating('update_aaaaaaaaaaaa_000001') });
    assert.equal(named.headline, 'Restarting into 7.8.2 (d2f6a91c)…');
    const other = await confirmActivation({ reply: activating('update_cccccccccccc_000003') });
    assert.equal(other.headline, 'Restarting the server…', 'another candidate in the reply is not taken for this one');
    const older = await confirmActivation({});
    assert.equal(older.headline, 'Restarting the server…', 'without the reply candidate nothing is named');
});

test('a target that needs another desktop app: the manual instruction when the server gives one, else the fact', () => {
    const instruction = 'Download Ouroboros 7.8.2 from https://github.com/razzant/ouroboros/releases/download/v7.8.2/'
        + 'Ouroboros-7.8.2.dmg, quit Ouroboros, then drag Ouroboros.app from that disk image over the current app at '
        + '/Users/anton/Desktop/Ouroboros.app.';
    const manual = (native, question = native.instruction || '') => updateVerdict({ ...STATUS, candidate: candidate({
        state: 'needs_decision', reason: `native_app: ${native.reason}`, question, native,
        actions: ['discard_candidate', 'prepare'] }) });
    const given = manual({ status: 'manual', reason: 'download_failed: offline', version: '7.8.2', instruction });
    assert.equal(given.state, 'candidate_needs_decision');
    assert.equal(given.tone, 'warn');
    assert.equal(given.headline, '7.8.2 (d2f6a91c) needs another desktop app. Nothing was changed.');
    assert.equal(given.hint, `${instruction} Then prepare it again.`);
    assert.deepEqual(buttons(given), ['Discard', '*Prepare again']);
    assert.equal(given.facts.find((row) => row.label === 'Desktop app').value,
        '7.8.2 is needed: the new app could not be downloaded (offline)');
    const bare = manual({ status: 'manual', reason: 'no_app_bundle' });
    assert.equal(bare.hint, 'No macOS desktop app was found for this installation. Then prepare it again.',
        'no instruction: the reason itself, as a fact');
    const staged = updateVerdict({ ...STATUS, candidate: candidate({ state: 'ready', actions: ['activate', 'discard_candidate'],
        native: { status: 'staged', reason: 'protocol_differs', version: '7.8.2' } }) });
    assert.match(staged.hint, /^Restart to update also replaces this desktop app with 7\.8\.2; the previous app stays beside it\./);
    assert.deepEqual(buttons(staged), ['Discard', '*Restart to update']);
    for (const verdict of [given, bare, staged]) {
        assert.doesNotMatch(`${verdict.headline} ${verdict.hint}`, /Interface size|older|reinstall/i, 'the app step names no feature');
    }
});

test('an adaptation under review, a budget pause and an owner question each keep the installation unchanged', () => {
    const adapting = updateVerdict({ ...STATUS, ...LOCAL, candidate: candidate({ state: 'adapting', local_changes: true,
        conflict_paths: ['supervisor/queue.py'], adaptation: { step: 'files', task_id: 'a1b2c3' }, actions: ['discard_candidate'] }) });
    assert.equal(adapting.headline, 'Adapting your local changes to 7.8.2 (d2f6a91c)…');
    assert.match(adapting.hint, /^A reviewed task resolves supervisor\/queue\.py in the separate copy \(task a1b2c3\)\. 7\.8\.1 keeps running\./);
    assert.deepEqual(buttons(adapting), ['Discard'], 'no primary while it adapts');
    const paused = updateVerdict({ ...STATUS, candidate: candidate({ state: 'paused_budget',
        actions: ['prepare', 'discard_candidate'] }) });
    assert.equal(paused.headline, 'Preparing 7.8.2 (d2f6a91c) stopped at the spending limit. Nothing was changed.');
    assert.deepEqual(buttons(paused), ['Discard', '*Prepare again']);
    const asked = updateVerdict({ ...STATUS, candidate: candidate({ state: 'needs_decision', reason: 'meaning_change',
        question: 'Keeping your edit changes how 7.8.2 schedules tasks.', actions: ['discard_candidate', 'prepare'] }) });
    assert.equal(asked.headline, 'This update needs your decision. Nothing was changed.');
    assert.equal(asked.hint, 'Keeping your edit changes how 7.8.2 schedules tasks.');
});

test('a current source under an app that cannot host it offers the app-only Update, and its candidate says only the app changes', () => {
    const CURRENT = { ...STATUS, behind: 0, available: false, latest_sha: SHA.base, latest_version: '7.8.1' };
    const needed = updateVerdict({ ...CURRENT, candidate: { state: 'none', actions: ['prepare'],
        app_needed: { status: 'needed', reason: 'signed_entry_missing', required: 1 } } });
    assert.deepEqual([needed.state, needed.tone, buttons(needed)], ['app_needed', 'warn', ['*Prepare app update']]);
    assert.equal(needed.headline, 'This desktop app needs the official app that runs the current version.');
    assert.match(needed.hint, /^This app has no signed entry for the current source, so its restarts lose the app’s profile and notification identity\. Prepare downloads and checks it beside this app; nothing changes until you restart, and your files stay exactly as they are\.$/);
    // Without the fact the panel claims the ordinary verdict; with a source update the app comes with it.
    assert.equal(updateVerdict({ ...CURRENT, candidate: { state: 'none', actions: ['prepare'] } }).state, 'current');
    assert.equal(updateVerdict({ ...STATUS, candidate: { state: 'none', actions: ['prepare'],
        app_needed: { status: 'needed', reason: 'signed_entry_missing' } } }).state, 'available');
    const ready = updateVerdict({ ...CURRENT, candidate: candidate({ state: 'ready', merge_kind: 'current', target_sha: SHA.base,
        checks: PASSED, native: { status: 'staged', version: '7.8.0' }, actions: ['activate', 'discard_candidate'] }) });
    assert.deepEqual([ready.state, ready.headline, buttons(ready)], ['candidate_ready',
        'The desktop app 7.8.0 is ready. Restart to switch to it.', ['Discard', '*Restart to update']]);
    assert.match(ready.hint, /Restart to update also replaces this desktop app with 7\.8\.0; the previous app stays beside it\./);
    const fact = (label) => ready.facts.find((row) => row.label === label)?.value;
    assert.equal(fact('Combined'), 'this checkout already has it: only the desktop app changes, the files stay exactly as they are');
    assert.equal(fact('Python packages'), 'its own copy, prepared by the new app’s Python');  // never "the lock differs"
});
