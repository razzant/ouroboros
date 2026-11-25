import assert from 'node:assert/strict';
import test from 'node:test';
import { assistedUpdatePhase, checkFailureReason, checkOutcome, dismissUpdateAttempt, initUpdateStatus, RECOVERY_RETRY, unfinishedRecovery, updateAttemptView } from '../modules/update_status.js';
import { installDomDouble } from './dom_double.js';
import { bindUpdateRefreshEvents, checkFailure, updateVerdict } from '../modules/updates.js';
import { apiClient } from '../modules/api_client.js';

const VERSION = 'v7.11.1-alpha.0';

const observed = (stage, extra = {}) => ({
    managed: true,
    update_progress: { operation_id: 'operation-a', generation: 'server-a', stage, active: true, ...extra },
});

test('only a completed update with a known base offers its local Restore target', () => {
    for (const outcome of ['applied', 'not_applied', 'rolled_back', 'failed', 'unknown']) {
        const view = updateAttemptView({ operation_id: 'a', outcome, active: false, base_sha: 'a'.repeat(40) });
        assert.equal(view.restoreTarget, outcome === 'applied' ? 'a'.repeat(40) : '');
    }
    assert.equal(updateAttemptView({ operation_id: 'a', outcome: 'applied', active: false }).restoreTarget, '');
    assert.equal(updateAttemptView({ operation_id: 'a', active: true, base_sha: 'a'.repeat(40) }).restoreTarget, undefined);
});

test('server stages refine the pending apply and survive reopening before a transaction exists', () => {
    for (const phase of ['updating', '', 'loading']) {
        const verdict = updateVerdict(observed('stopping_workers'), phase);
        assert.equal(verdict.headline, 'Stopping worker processes…');
        assert.equal(verdict.state, 'updating');
        assert.equal(verdict.action.disabled, true);
    }
    assert.equal(updateVerdict(observed('checking'), 'updating').headline, 'Checking the update and dependencies…');
});

test('durable recovery outranks an earlier running observation', () => {
    for (const recovery of ['corrupt', 'gate_blocked', 'marker_cleanup_retry']) {
        const data = { ...observed('stopping_workers'), update_tx: { active: true, phase: recovery } };
        const verdict = updateVerdict(data, 'updating');
        assert.equal(verdict.tone, 'error');
        assert.notEqual(verdict.headline, 'Stopping worker processes…');
    }
});

test('terminal failure without a transaction remains actionable after reopen', () => {
    const failure = observed('stopping_services', { active: false, result: 'failed', error: 'service remains alive', restart_required: true });
    assert.equal(updateVerdict(failure).state, 'restart_needed');
    assert.equal(updateVerdict(failure).action.id, 'restart');
    assert.match(updateVerdict(failure).hint, /service remains alive/);
    const retry = observed('preparing', { active: false, result: 'failed', error: 'target moved' });
    assert.equal(updateVerdict(retry).action.id, 'check');
});

test('current restart outcomes and surviving transactions outrank an old failed observation', () => {
    const old = observed('preparing', { active: false, result: 'failed', error: 'old refusal' });
    for (const phase of ['restart_needed', 'restart_required']) {
        assert.equal(updateVerdict(old, phase).state, phase);
    }
    const stashing = { ...old, update_tx: { active: true, phase: 'stashing_local_work' } };
    assert.equal(updateVerdict(stashing).state, 'resolving');
    assert.match(updateVerdict(stashing).hint, /stashing_local_work/);
});

test('retained failure does not replace a new request that is still pending', () => {
    const old = observed('preparing', { active: false, result: 'failed', error: 'old refusal' });
    for (const phase of ['checking', 'preflighting', 'updating']) {
        const verdict = updateVerdict(old, phase);
        assert.equal(verdict.state, phase);
        assert.equal(verdict.action.disabled, true);
    }
});

test('reconnect preserves a preflight or apply whose request is still pending', () => {
    for (const phase of ['preflighting', 'updating']) {
        const listeners = new Map(), reads = [];
        const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
        const binding = bindUpdateRefreshEvents({ ws, getPhase: () => phase,
            loadStatus: (value) => reads.push(value), reconcileRestart: () => assert.fail('not restarting') });
        listeners.get('open')({ previouslyConnected: true });
        assert.deepEqual(reads, [{ fetchRemote: false, preservePhase: true }]);
        binding.dispose();
    }
});

test('assisted handoff and fresh generation use existing durable state', () => {
    const data = { ...observed('applying', { active: false, result: 'assisted_started' }),
        update_tx: { active: true, phase: 'assisted_resolution', task_id: 'resolver-a' } };
    assert.match(updateVerdict(data).hint, /resolver-a/);
    assert.match(updateVerdict({ managed: true, update_tx: { active: true, phase: 'pending_boot_smoke' } }).headline, /waiting for a restart/);
    assert.notEqual(updateVerdict({ managed: true, update_progress: {} }).state, 'updating');
});

test('progress invalidation refreshes an apply but cannot certify boot completion', () => {
    const listeners = new Map();
    const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
    let phase = 'updating';
    const reads = [], restart = [];
    const binding = bindUpdateRefreshEvents({ ws, getPhase: () => phase,
        loadStatus: (value) => reads.push(value), reconcileRestart: (value) => restart.push(value) });
    listeners.get('update_progress_changed')();
    assert.deepEqual(reads, [{ fetchRemote: false, preservePhase: true }]);
    phase = 'restarting';
    binding.beginRestarting();
    listeners.get('update_progress_changed')();
    assert.equal(restart.length, 0);
    listeners.get('open')({ previouslyConnected: true });
    listeners.get('update_progress_changed')();
    assert.deepEqual(restart, [{ afterBootNotice: false }, { afterBootNotice: false }]);
    binding.dispose();
    assert.equal(listeners.size, 0);
});

test('the requested description task ends through its own task_done, which alone reads the status again', () => {
    const listeners = new Map();
    const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
    let phase = '';
    let requested = ['d3adb33f', 'receipt-task'];
    const reads = [];
    const binding = bindUpdateRefreshEvents({ ws, getPhase: () => phase, loadStatus: (value) => reads.push(value),
        reconcileRestart: () => {}, getRequestTaskIds: () => requested });
    const log = listeners.get('log');
    log({ data: { type: 'task_done', task_id: 'other-task' } });
    log({ data: { type: 'task_heartbeat', task_id: 'd3adb33f' } });
    log({ data: { type: 'task_done', task_id: 'd3adb33f', _historical: true } });
    assert.deepEqual(reads, [], 'another task, another event or a replayed frame is not this request ending');
    log({ data: { type: 'task_done', task_id: 'd3adb33f' } });
    assert.deepEqual(reads, [{ fetchRemote: false, preservePhase: true }]);
    // A host retry ends under its own id with the request as its root; a child of the request does not end it.
    log({ data: { type: 'task_done', task_id: 'd3adb33f-retry', root_task_id: 'd3adb33f', parent_task_id: 'd3adb33f' } });
    assert.equal(reads.length, 1);
    log({ data: { type: 'task_done', task_id: 'd3adb33f-retry', root_task_id: 'd3adb33f' } });
    assert.equal(reads.length, 2);
    reads.pop();
    log({ data: { type: 'task_done', task_id: 'receipt-task' } });
    assert.equal(reads.length, 2, 'a receipt can name another task than the last status');
    requested = [];
    log({ data: { type: 'task_done', task_id: 'd3adb33f' } });
    phase = 'restarting';
    requested = ['d3adb33f'];
    log({ data: { type: 'task_done', task_id: 'd3adb33f' } });
    assert.equal(reads.length, 2, 'no request, or a restart in progress, reads nothing');
    binding.dispose();
    assert.equal(listeners.size, 0);
});

test('Continue admitted in another client refreshes both panels and follows its terminal', () => {
    const clients = [0, 1].map(() => {
        const listeners = new Map(), reads = [];
        const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
        let current = 'original';
        const binding = bindUpdateRefreshEvents({ ws, getPhase: () => '', reconcileRestart() {},
            getRequestTaskIds: () => [current],
            loadStatus: (options) => { reads.push(options); current = 'successor'; } });
        return { listeners, reads, binding };
    });
    // The host's existing log frame reaches both sockets. No document-local event is sent.
    for (const client of clients) {
        client.listeners.get('log')({ data: { type: 'owner_continue_admitted', predecessor_task_id: 'other' } });
        assert.equal(client.reads.length, 0);
        client.listeners.get('log')({ data: { type: 'owner_continue_admitted', task_id: 'successor', predecessor_task_id: 'original' } });
        assert.equal(client.reads.length, 1);
        client.listeners.get('log')({ data: { type: 'task_done', task_id: 'successor' } });
        assert.equal(client.reads.length, 2);
        client.binding.dispose();
        assert.equal(client.listeners.size, 0);
    }
});

const refusal = {
    operation_id: 'attempt-a', generation: 'server-a', current_generation: true, stage: 'draining_writers',
    active: false, result: 'failed', outcome: 'not_applied', outcome_at: '2026-10-21T00:20:00Z',
    error: 'Could not prove every repository writer stopped.', reason: 'update_writer_fence_blocked',
    blockers: ['active:Main conversation'], work_stop: 'none', concluded_by: 'executor', restart_required: false,
    base_sha: 'f'.repeat(40), target_sha: 'b'.repeat(40),
};
const settle = async () => {
    for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

test('a refusal before any teardown is not applied, stopped nobody, and names its blocker', () => {
    const view = updateAttemptView(refusal);
    assert.deepEqual([view.terminal, view.pill, view.tone, view.headline],
        [true, 'Update not applied', 'error', 'No update was applied.']);
    assert.equal(view.text, 'Work was still finishing: Main conversation. This attempt stopped no running work. Try again when it finishes.');
    assert.deepEqual(view.details.find((row) => row.label === 'Reason'), { label: 'Reason', value: 'update_writer_fence_blocked' });
    assert.equal(view.details.some((row) => row.label === 'Saved local work'), false, 'no ref is ever manufactured');
});

test('after teardown or a restart the attempt never claims it stopped nobody', () => {
    for (const work_stop of ['invoked', '']) {
        assert.doesNotMatch(updateAttemptView({ ...refusal, blockers: ['service:mcp'], work_stop }).text, /stopped no running work/);
    }
    assert.match(updateAttemptView({ ...refusal, blockers: [], work_stop: 'invoked' }).text, /may have been interrupted/);
});

test('only the restarted server concludes success; reconnect, version and SHA are not evidence', () => {
    const awaiting = { ...refusal, outcome: '', result: 'restart_requested', stage: 'restart_requested',
        current_generation: false, work_stop: '', blockers: [] };
    const view = updateAttemptView(awaiting);
    assert.deepEqual([view.terminal, view.pill], [false, 'Confirming update…']);
    assert.doesNotMatch(view.headline, /completed|applied/i);
    const unknown = updateAttemptView({ ...awaiting, outcome: 'unknown', reason: 'no_completion_record' });
    assert.deepEqual([unknown.pill, unknown.tone, unknown.terminal], ['Update outcome unconfirmed', 'neutral', true]);
    assert.match(unknown.text, /running version alone does not establish the outcome/);
    const applied = updateAttemptView({ ...awaiting, outcome: 'applied', concluded_by: 'boot' });
    // Owner 1B: the confirmed success is a record (headline, facts, Restore target), never a capsule word.
    assert.deepEqual([applied.pill, applied.tone, applied.headline], ['', 'ok', 'Update completed.']);
    assert.equal(updateAttemptView({ ...awaiting, outcome: 'unknown', reason: 'startup_unconfirmed' }).text,
        'The restarted server did not confirm its startup checks.');
});

test('a rollback and a preserved Replace ref read as facts, not actions', () => {
    const rolled = updateAttemptView({ ...refusal, outcome: 'rolled_back', error: 'pre-restart smoke failed', blockers: [],
        work_stop: 'invoked' });
    assert.deepEqual([rolled.pill, rolled.headline], ['Previous version restored', 'The update failed; the previous version was restored.']);
    const kept = updateAttemptView({ ...refusal, outcome: 'applied', concluded_by: 'boot', blockers: [], work_stop: '',
        keep_branch: 'local-keep-20261021_032000-a1b2c3' });
    assert.match(kept.text, /kept on branch local-keep-20261021_032000-a1b2c3/);
    assert.equal(kept.details.find((row) => row.label === 'Saved local work').value, 'local-keep-20261021_032000-a1b2c3');
});

test('a failed save discloses retention uncertainty without changing the actual outcome or stage', () => {
    for (const active of [false, true]) {
        const progress = { ...refusal, active, retention_unconfirmed: true };
        const view = updateAttemptView(progress);
        const ordinary = updateAttemptView({ ...progress, retention_unconfirmed: false });
        assert.deepEqual([view.pill, view.headline, view.terminal], [ordinary.pill, ordinary.headline, ordinary.terminal]);
        assert.match(view.text, /could not be saved.*after restart/);
        assert.doesNotMatch(ordinary.text, /could not be saved/);
    }
});

test('a retained outcome never takes the primary action: Update stays an explicit retry', () => {
    const data = { managed: true, check_ok: true, available: true, safe_to_apply: true, latest_version: '7.1.0',
        update_tx: { active: false }, update_progress: refusal };
    const verdict = updateVerdict(data);
    assert.deepEqual([verdict.state, verdict.action.id, verdict.action.label], ['available', 'update', 'Update to 7.1.0']);
    assert.deepEqual([updateVerdict({ ...data, available: false }).state, updateVerdict({ ...data, available: false }).action.id],
        ['current', 'check']);
});

test('a restart need belongs to the server that had it', () => {
    const failure = { ...refusal, outcome: 'failed', restart_required: true, blockers: ['service:mcp'], work_stop: 'invoked' };
    assert.equal(updateVerdict({ managed: true, update_progress: failure }).state, 'restart_needed');
    assert.notEqual(updateVerdict({ managed: true, update_progress: { ...failure, current_generation: false } }).state,
        'restart_needed');
    assert.doesNotMatch(updateAttemptView({ ...failure, current_generation: false }).text, /needs a restart/);
});

// One indicator on the DOM double: its listeners, storage and a window whose events reach their listeners, on the
// sidebar's foot as pages.js renders it (the system rows' card with the version row), so the version's own check
// and its answer are exercised by the same world the capsule is.
async function withIndicator(run) {
    const env = installDomDouble();
    const stored = new Map();
    const priorStorage = globalThis.localStorage;
    const priorDispatch = globalThis.window.dispatchEvent;
    globalThis.localStorage = { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, String(value)) };
    globalThis.window.dispatchEvent = (event) => globalThis.window.emit(event.type);
    let controller;
    try {
        const settings = env.el('button', { className: 'nav-row', dataset: { navPage: 'settings' } });
        const row = env.el('div', { className: 'sidebar-foot-meta' }, env.el('span', { attrs: { id: 'nav-version' }, text: VERSION }));
        env.doc.body.append(env.el('nav', { attrs: { id: 'primary-sidebar' } }, settings, row));
        const listeners = new Map();
        const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
        const opened = [];
        const world = { status: null, reads: 0, checks: [] };
        controller = initUpdateStatus({ ws, showPage: (page) => opened.push(page), openDashboardTab: (tab) => opened.push(tab),
            upToDateMs: 20,
            readStatus: async () => { world.reads += 1; if (!world.status) throw new Error('offline'); return structuredClone(world.status); },
            // Each press's check, answered when the test says so; the selection the press believed is recorded.
            checkNow: (selection) => new Promise((resolve, reject) => world.checks.push({ selection, resolve, reject })) });
        const show = async (status, event = 'update_progress_changed') => {
            world.status = status;
            listeners.get(event)();
            await settle();
        };
        const pill = () => env.doc.getElementById('update-pill');
        const close = () => env.doc.body.querySelector('.update-notice-close');
        const $ = (id) => env.doc.getElementById(id);
        const parts = () => ({ check: $('nav-version-check'), version: $('nav-version'), pill: pill(),
            note: row.querySelector('.nav-check-note'), said: $('nav-check-said') });
        await run({ env, row, settings, show, pill, close, opened, world, stored, listeners, parts });
    } finally {
        controller?.dispose();
        globalThis.window.dispatchEvent = priorDispatch;
        globalThis.localStorage = priorStorage;
        env.restore();
    }
}

const applied = { ...refusal, operation_id: 'attempt-ok', outcome: 'applied', result: 'applied', concluded_by: 'boot',
    blockers: [], work_stop: '', reason: '' };

test('a confirmed success never wears the indicator, hides nothing, and never masks a newer version', async () => {
    await withIndicator(async ({ show, pill, close, stored }) => {
        await show({ managed: true, available: false, update_progress: { ...refusal, active: true, outcome: '', result: '',
            stage: 'checking', work_stop: 'invoked' } });
        assert.deepEqual([pill().hidden, pill().textContent], [false, 'Checking update…']);
        assert.equal(close().hidden, true, 'a stage has no close: it ends by itself');
        // The server concludes: nothing announces it. The record stays in Dashboard → Updates (updateAttemptView).
        await show({ managed: true, available: false, update_progress: applied });
        assert.equal(pill().hidden, true, 'no capsule, no toast, no timer: the success lives in the panel alone');
        assert.equal(pill().dataset.tone, undefined);
        assert.equal(stored.size, 0, 'nothing is stored: there is nothing to dismiss');
        assert.equal(updateAttemptView(applied).headline, 'Update completed.', 'the retained record is untouched');
        // An old success must not stand where a newer version is announced.
        await show({ managed: true, available: true, latest_version: '7.1.0', update_progress: applied });
        assert.deepEqual([pill().hidden, pill().textContent, pill().dataset.tone], [false, 'Update', undefined]);
        assert.equal(close().hidden, true, 'availability is a current fact: it ends with the fact');
    });
});

test('a failed or unconfirmed outcome stays with its own close; closing hides this notice and nothing else', async () => {
    await withIndicator(async ({ env, show, pill, close, opened, world, stored }) => {
        await show({ managed: true, available: false, update_progress: refusal });
        assert.deepEqual([pill().hidden, pill().textContent, pill().dataset.tone], [false, 'Update not applied', 'error']);
        assert.equal(close().hidden, false, 'the close stands beside the notice it ends');
        assert.equal(close().getAttribute('aria-label'), 'Hide this notice');
        assert.equal(close().tagName, 'BUTTON');
        pill().click();
        assert.deepEqual(opened, ['updates'], 'the capsule still opens the one panel');
        const before = world.reads;
        close().click();
        assert.equal(pill().hidden, true, 'hidden: no available update to point at');
        assert.equal(world.reads, before, 'hiding sends no request');
        assert.deepEqual([...stored], [['ouroboros.update-attempt-dismissed', 'attempt-a']], 'one attempt id, this browser only');
        assert.equal(updateAttemptView(refusal).headline, 'No update was applied.', 'the saved result is untouched');
        // A newer notice-worthy attempt shows again, closable, and a failure does not outlast a newer version.
        await show({ managed: true, available: true, update_progress: { ...refusal, operation_id: 'attempt-b', outcome: 'unknown',
            reason: 'no_completion_record', work_stop: '', blockers: [] } });
        assert.deepEqual([pill().hidden, pill().textContent, pill().dataset.tone], [false, 'Update outcome unconfirmed', 'neutral']);
        assert.equal(close().hidden, false, 'unconfirmed stays until closed, even beside a newer version');
        close().focus();
        assert.equal(env.doc.activeElement, close());
        close().click();
        assert.equal(env.doc.activeElement, pill(), 'a keyboard reader lands on what the close unmasked, not on the page');
        assert.deepEqual([pill().hidden, pill().textContent, pill().dataset.tone], [false, 'Update', undefined],
            'once hidden, the version it was hiding shows');
        assert.equal(close().hidden, true);
        // Another browser has no such memory: the same record shows the notice again there.
        stored.clear();
        globalThis.window.emit('ouro:update-attempt-dismissed');
        assert.deepEqual([pill().textContent, close().hidden], ['Update outcome unconfirmed', false]);
    });
});

test('restart messages are untouched: a needed restart is a stage without a close, a failed one keeps its words', async () => {
    await withIndicator(async ({ show, pill, close }) => {
        await show({ managed: true, available: false, update_progress: { ...refusal, active: false, outcome: '',
            result: 'restart_required', restart_required: true, stage: 'stopping_services', blockers: ['service:mcp'] } });
        assert.deepEqual([pill().textContent, pill().dataset.tone, close().hidden], ['Restart needed', 'neutral', true]);
        await show({ managed: true, available: false, update_progress: { ...refusal, outcome: 'failed', restart_required: true,
            blockers: ['service:mcp'], work_stop: 'invoked', error: 'A service could not be stopped.' } });
        assert.deepEqual([pill().textContent, pill().dataset.tone, close().hidden], ['Update failed', 'error', false]);
        assert.match(pill().title, /Ouroboros needs a restart before updates can continue\./);
    });
});

// #2671: the shapes /api/update/status serves for a resolver start whose boot rollback did not finish
// (tests/test_update_attempt_consumers.py drives the real boot for each).
const resolverStart = { ...refusal, operation_id: 'attempt-r', generation: 'server-a', current_generation: false,
    stage: 'preparing', result: 'assisted_started', outcome: '', concluded_by: undefined, reason: '', error: '',
    blockers: [], work_stop: undefined, tx_attempt_id: 'tx-r' };
const unfinished = { ...resolverStart, outcome: 'failed', reason: 'rollback_incomplete', concluded_by: 'boot',
    error: 'rollback could not clear update intent before restoring the repository',
    stash_note: 'the conflicting update could not resume after the restart, and its rollback did not finish.',
    blockers: ['worker:4242'] };
const boundTx = (phase, attempt = 'tx-r') => ({ active: true, phase, task_id: '', restart_required: false, attempt_id: attempt });

test('a boot rollback that did not finish reads as incomplete recovery everywhere, with no invented control', () => {
    const view = updateAttemptView(unfinished, boundTx('rolling_back'));
    assert.deepEqual([view.terminal, view.tone, view.pill, view.restoreTarget],
        [true, 'error', 'Update recovery incomplete', '']);
    assert.equal(view.headline, 'The update failed, and restoring the previous version did not finish.');
    assert.match(view.text, /^Quit and reopen the desktop app/);
    assert.equal(view.details.find((row) => row.label === 'Error').value, unfinished.error);
    assert.equal(view.details.find((row) => row.label === 'Recovery note').value, unfinished.stash_note);
    assert.doesNotMatch(view.text, /rollback could not clear|the conflicting/);
    assert.equal(view.details.find((row) => row.label === 'Blocked by').value, 'worker:4242');
    // History once its transaction is gone: the fact stays, the retry instruction does not.
    assert.doesNotMatch(updateAttemptView(unfinished, { active: false }).text, /Quit and reopen/);
    assert.doesNotMatch(updateAttemptView(unfinished, boundTx('rolling_back', 'tx-other')).text, /Quit and reopen/);
    for (const phase of ['rolling_back', 'assisted_resolution', 'pending_boot_smoke']) {
        const verdict = updateVerdict({ managed: true, update_progress: unfinished, update_tx: boundTx(phase) });
        // pending_boot_smoke keeps the in-app Restart its phase allows (server_restart._RESTARTABLE_UPDATE_PHASES).
        assert.deepEqual([verdict.state, verdict.tone, verdict.action, verdict.recoveryInstruction], ['resolving', 'error',
            phase === 'pending_boot_smoke' ? { id: 'restart', label: 'Restart now' } : { id: 'check', label: 'Check again' }, true]);
        assert.equal(verdict.headline, view.headline, `${phase}: the panel and the indicator say one thing`);
        assert.equal(verdict.hint, view.text);
        assert.doesNotMatch(verdict.headline, /under review|waiting for a restart/);
    }
    // An executor's own failure keeps its ordinary words: the reason names this boot report alone.
    const executor = updateAttemptView({ ...unfinished, reason: 'update_writer_fence_blocked', concluded_by: 'executor' },
        boundTx('rolling_back'));
    assert.deepEqual([executor.pill, executor.headline], ['Update failed', 'The update did not complete.']);
});

test('only a bound transaction still in an assisted phase is a resolution at work', () => {
    assert.equal(updateAttemptView(resolverStart, boundTx('assisted_resolution')).pill, 'Resolving update…');
    assert.equal(updateAttemptView(resolverStart, boundTx('committing_assisted')).pill, 'Resolving update…');
    for (const tx of [boundTx('marker_cleanup_retry'), boundTx('gate_blocked'), boundTx('rolling_back'),
        { ...boundTx('pending_boot_smoke'), local_work_recovery: true }, boundTx('assisted_resolution', 'tx-other'),
        { active: false }, null]) {
        const view = updateAttemptView(resolverStart, tx);
        assert.deepEqual([view.terminal, view.tone, view.pill, view.headline],
            [false, 'neutral', 'Update not concluded', 'This update has not concluded yet.'], JSON.stringify(tx));
    }
});

test('the panel can say the recovery instruction once without hiding other retained facts', () => {
    const tx = boundTx('assisted_resolution');  // a failed write can leave this old phase on disk
    for (const extra of [{ keep_branch: 'local-keep-test' }, { retention_unconfirmed: true },
        { keep_branch: 'local-keep-test', retention_unconfirmed: true }]) {
        const progress = { ...unfinished, ...extra };
        const verdict = updateVerdict({ managed: true, update_progress: progress, update_tx: tx });
        assert.equal(verdict.hint, RECOVERY_RETRY);
        const view = updateAttemptView(progress, tx, { showRecoveryInstruction: !verdict.recoveryInstruction });
        assert.doesNotMatch(view.text, /Quit and reopen/);
        if (extra.keep_branch) assert.match(view.text, /local-keep-test/);
        if (extra.retention_unconfirmed) assert.match(view.text, /could not be saved/);
        assert.match(updateAttemptView(progress, tx).text, /Quit and reopen/, 'the indicator retains its instruction');
    }
});

test('a verdict that gives the recovery instruction owns it, and a recorded completion outdates "did not finish"', () => {
    const quitReopen = /Quit and reopen/g;
    // local_work_recovery outranks the failed record: stash recovery after an unfinished rollback (still true), and a
    // finished rollback awaiting its restart (marker_cleanup_retry, rollback_restart_pending).
    for (const [phase, stale] of [['rolling_back', false], ['assisted_resolution', false], ['marker_cleanup_retry', true]]) {
        const tx = { ...boundTx(phase), local_work_recovery: true };
        const verdict = updateVerdict({ managed: true, update_progress: unfinished, update_tx: tx });
        assert.equal(verdict.headline, 'Local changes still need recovery confirmation.');
        assert.equal(verdict.recoveryInstruction, true);
        const view = updateAttemptView(unfinished, tx, { showRecoveryInstruction: !verdict.recoveryInstruction });
        const said = [verdict.hint, view.text].join(' ').match(quitReopen) || [];
        assert.equal(said.length, 1, `${phase}: one quit/reopen instruction`);
        assert.equal(view.headline, stale ? 'The update did not complete.'
            : 'The update failed, and restoring the previous version did not finish.', phase);
        assert.equal(Boolean(unfinishedRecovery(unfinished, tx)), !stale, phase);
        // The exact diagnostics remain reachable, historical wording included.
        assert.equal(view.details.find((row) => row.label === 'Recovery note').value, unfinished.stash_note);
    }
    // Cleanup retry alone (no local work): its own gate verdict gives the instruction; the record is history.
    const cleanup = updateVerdict({ managed: true, update_progress: unfinished, update_tx: boundTx('marker_cleanup_retry') });
    assert.equal(cleanup.headline, 'An update attempt stopped mid-flight and needs recovery.');
    assert.equal(cleanup.recoveryInstruction, true);
    const history = updateAttemptView(unfinished, boundTx('marker_cleanup_retry'));
    assert.deepEqual([history.pill, history.headline], ['Update failed', 'The update did not complete.']);
    assert.doesNotMatch(history.text, quitReopen);
    // A corrupt marker's own instruction is not repeated either.
    const corrupt = updateVerdict({ managed: true, update_progress: unfinished, update_tx: boundTx('corrupt') });
    assert.equal(corrupt.recoveryInstruction, true);
    // A verdict without its own instruction leaves the attempt's single one in place.
    const plain = updateVerdict({ managed: true, update_progress: resolverStart, update_tx: boundTx('assisted_resolution') });
    assert.notEqual(plain.recoveryInstruction, true);
});

test('one assisted-phase reading serves the panel verdict and the attempt record', () => {
    for (const phase of ['materializing_assisted', 'assisted_resolution', 'committing_assisted',
        'assisted_unknown_future', 'gate_blocked', 'pending_boot_smoke', '']) {
        const tx = boundTx(phase);
        const verdict = updateVerdict({ managed: true, update_progress: resolverStart, update_tx: tx });
        const view = updateAttemptView(resolverStart, tx);
        const resolving = assistedUpdatePhase(phase);
        assert.equal(verdict.headline === 'A conflicting update is being resolved under review.', resolving, phase);
        assert.equal(view.pill === 'Resolving update…', resolving, phase);
    }
});

test('a hidden recovery notice stays hidden for its attempt while Last attempt takes the later conclusion', async () => {
    await withIndicator(async ({ show, pill, close }) => {
        await show({ managed: true, available: false, update_progress: resolverStart, update_tx: boundTx('assisted_resolution') });
        assert.deepEqual([pill().textContent, close().hidden], ['Resolving update…', true]);
        await show({ managed: true, available: false, update_progress: unfinished, update_tx: boundTx('rolling_back') });
        assert.deepEqual([pill().hidden, pill().textContent, pill().dataset.tone, close().hidden],
            [false, 'Update recovery incomplete', 'error', false]);
        assert.match(pill().title, /Quit and reopen/);
        close().click();
        const restored = { ...unfinished, outcome: 'rolled_back', reason: '', error: '', stash_note: '', blockers: [] };
        await show({ managed: true, available: false, update_progress: restored, update_tx: { active: false } });
        assert.equal(pill().hidden, true, 'the same operation stays dismissed in this browser');
        assert.equal(updateAttemptView(restored, { active: false }).pill, 'Previous version restored');
    });
});

test('the indicator coalesces notices, never renders a superseded read, and disposes every subscription', async () => {
    const env = installDomDouble();
    try {
        const listeners = new Map();
        const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
        const pending = [];
        const windowListeners = globalThis.window.listenerCount();
        const controller = initUpdateStatus({ ws, readStatus: () => new Promise((resolve) => pending.push(resolve)) });
        listeners.get('update_progress_changed')();
        listeners.get('update_progress_changed')();
        assert.equal(pending.length, 1, 'one read in flight; later notices queue one more');
        pending.shift()({ managed: true, available: false, update_progress: { ...refusal, operation_id: 'stale' } });
        await settle();
        assert.equal(env.doc.getElementById('update-pill'), null, 'the superseded answer is not rendered');
        assert.equal(pending.length, 1, 'the queued read runs');
        pending.shift()({ managed: true, available: false, update_progress: refusal });
        await settle();
        assert.equal(env.doc.getElementById('update-pill').textContent, 'Update not applied');
        controller.dispose();
        assert.equal(listeners.size, 0);
        assert.equal(globalThis.window.listenerCount(), windowListeners);
    } finally {
        env.restore();
    }
});

// The sidebar's version is the check for updates (owner 21.11, docs/DESIGN.md "Navigation"): no Check button; a press
// runs the panel's one check and its answer stands where the Update capsule would. The tests below are that row's
// own contract, on the same indicator world the capsule's tests use.

test('before the first status read the version is plain text, never a control that would do nothing', async () => {
    await withIndicator(async ({ parts, row, show, world }) => {
        // The row has been mounted and asked for its status; that first read has not answered yet, so this is the
        // row's opening paint. A control here is a button whose press does nothing, and the read may take seconds.
        assert.equal(parts().check.hidden, true, 'nothing to press before a status says this install can check');
        assert.equal(parts().version.parentElement, row, 'the version stands as the plain text it is');
        parts().check.click();
        assert.equal(world.checks.length, 0, 'and that press reaches nothing');
        // The status answers: the same row is now the check, and its answer stands in place of a capsule.
        await show({ managed: true, available: false, checked_at: 't0' });
        const { check, version } = parts();
        assert.deepEqual([check.hidden, version.parentElement], [false, check]);
    });
});

test('at rest the version is a quiet button named by the action and the version, its glyph kept in place', async () => {
    await withIndicator(async ({ env, row, parts, show }) => {
        // Until a passive status actually establishes checkability the version stays plain text.
        await settle();
        assert.equal(parts().check.hidden, true);
        assert.equal(parts().version.parentElement, row);
        await show({ managed: true, available: false, checked_at: 't0' });
        const { check, version, note, said } = parts();
        assert.equal(check.tagName, 'BUTTON');
        assert.equal(version.parentElement, check, 'the version is the control\'s own text: utils.js still writes it');
        assert.equal(version.textContent, VERSION);
        assert.equal(check.getAttribute('aria-labelledby'), 'nav-version-check-word nav-version');
        assert.equal(env.doc.getElementById('nav-version-check-word').textContent, 'Check for updates');
        assert.equal(check.title, 'Check for updates');
        assert.equal(check.getAttribute('aria-describedby'), 'nav-check-said');
        assert.ok(check.classList.contains('ui-hit-area'), 'a finger reaches 44 px however small it looks');
        assert.ok(check.querySelector('.nav-version-glyph'), 'the turning glyph is always in its box');
        assert.equal(check.getAttribute('aria-busy'), null);
        assert.equal(note.hidden, true);
        assert.equal(said.getAttribute('role'), 'status');
        assert.equal(parts().pill.hidden, true);
    });
});

test('one press is one check: a press while it runs starts nothing, and the version never changes', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.click();
        assert.equal(world.checks.length, 1);
        assert.equal(parts().check.getAttribute('aria-busy'), 'true');
        parts().check.click();
        parts().check.click();
        assert.equal(world.checks.length, 1, 'coalesced: no second request');
        assert.equal(parts().version.textContent, VERSION);
        assert.equal(parts().note.hidden, true, 'nothing is claimed while it checks');
        world.checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't1' });
        await settle();
        assert.equal(parts().check.getAttribute('aria-busy'), null);
        parts().check.click();
        assert.equal(world.checks.length, 2, 'a press after the answer checks again');
    });
});

test('up to date says so in place and aloud, then the row is at rest again by itself', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.click();
        world.checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't1' });
        await settle();
        const { note, said, check, pill } = parts();
        assert.deepEqual([note.hidden, note.textContent, note.dataset.tone], [false, 'Up to date', undefined]);
        assert.equal(said.textContent, 'Up to date');
        assert.equal(check.hidden, false);
        assert.equal(pill.hidden, true);
        await new Promise((resolve) => setTimeout(resolve, 40));
        assert.equal(parts().note.hidden, true);
        assert.equal(parts().said.textContent, '');
    });
});

test('a failed check is said in the error tone with its reason, and the version stays its retry', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.click();
        world.checks[0].resolve({ managed: true, available: false, check_ok: false, checked_at: 't0', warnings: ['fetch_failed'] });
        await settle();
        let { note, said, check } = parts();
        assert.deepEqual([note.hidden, note.textContent, note.dataset.tone], [false, 'Check failed', 'error']);
        assert.equal(note.title, 'Could not check the official update channel: fetch_failed');
        assert.equal(said.textContent, note.title, 'the reason is the version\'s description and is announced');
        assert.equal(check.hidden, false, 'the version is still the check: a retry');
        // A progress notice that brings no newer good check leaves the failure standing.
        await show({ managed: true, available: false, checked_at: 't0' });
        assert.equal(parts().note.hidden, false);
        // A request that never reached the server is a failure too, never silence.
        parts().check.click();
        assert.equal(parts().note.hidden, true, 'a new check clears the old answer');
        world.checks[1].reject(new Error('Server unavailable'));
        await settle();
        ({ note } = parts());
        assert.deepEqual([note.hidden, note.textContent], [false, 'Check failed']);
        assert.match(note.title, /Server unavailable$/);
        // A later good check, here or from Dashboard → Updates, ends it: the two surfaces never disagree.
        await show({ managed: true, available: false, checked_at: 't2' });
        assert.equal(parts().note.hidden, true);
    });
});

test('an update found is the crimson Update capsule; the version is plain text beside it and focus follows', async () => {
    await withIndicator(async ({ env, row, parts, show, opened, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.focus();
        parts().check.click();
        world.checks[0].resolve({ managed: true, available: true, check_ok: true, local_changes: 'absent', current_version: '7.11.1', latest_version: '7.12.0' });
        await settle();
        const { pill, check, version, note, said } = parts();
        assert.deepEqual([pill.hidden, pill.textContent, pill.title], [false, 'Update', 'Update 7.11.1 → 7.12.0']);
        assert.ok(pill.classList.contains('btn-primary') && !pill.classList.contains('btn-default'));
        assert.equal(check.hidden, true, 'one action per row');
        assert.equal(version.parentElement, row, 'the version stands as plain text');
        assert.equal(note.hidden, true);
        assert.equal(said.textContent, 'Update 7.11.1 → 7.12.0');
        assert.equal(env.doc.activeElement, pill, 'a keyboard reader lands on the capsule, never on the page');
        pill.click();
        assert.deepEqual(opened, ['updates'], 'it opens Dashboard → Updates and starts nothing');
        assert.equal(world.checks.length, 1);
    });
});

test('local changes keep the amber caveat, an attempt its own words; the version is plain text beside either', async () => {
    await withIndicator(async ({ row, parts, show }) => {
        await show({ managed: true, available: true, local_changes: 'present', checked_at: 't0' });
        let { pill, check, version } = parts();
        assert.ok(pill.classList.contains('has-local') && !pill.classList.contains('btn-primary'));
        assert.equal(pill.textContent, 'Update');
        assert.deepEqual([check.hidden, version.parentElement], [true, row]);
        await show({ managed: true, available: false, checked_at: 't0',
            update_progress: { operation_id: 'a', generation: 'g', stage: 'applying', active: true } });
        ({ pill, check } = parts());
        assert.deepEqual([pill.hidden, pill.textContent, pill.dataset.tone], [false, 'Applying update…', 'neutral']);
        assert.ok(pill.classList.contains('btn-default'));
        assert.equal(check.hidden, true);
        // The attempt over and nothing on offer: the version is the check again.
        await show({ managed: true, available: false, checked_at: 't0' });
        ({ pill, check, version } = parts());
        assert.deepEqual([pill.hidden, check.hidden, version.parentElement], [true, false, check]);
    });
});

test('checkOutcome reads a check\'s answer: only this check\'s own positive result is Up to date', () => {
    assert.deepEqual(checkOutcome({ check_ok: true, available: false }), { kind: 'current' });
    assert.deepEqual(checkOutcome({ check_ok: true, available: true }), { kind: 'available' });
    assert.deepEqual(checkOutcome({ check_ok: false, checked_at: 'x' }),
        { kind: 'failed', reason: 'the check did not complete', lastGood: 'x' });
    assert.equal(checkOutcome(null, new Error('boom')).reason, 'boom');
    // A success needs its explicit flag AND a boolean availability: a truthy or missing flag reports nothing.
    for (const answer of [{ check_ok: true, available: 'yes' }, { check_ok: true },
        { check_ok: true, available: 1 }, { update_tx: { active: true } }, { available: false }, { managed: true, available: false, check_ok: null },
        { update_progress: { active: true, operation_id: 'a' } }, { available: true }]) {
        assert.equal(checkOutcome(answer).kind, 'failed', JSON.stringify(answer));
    }
    assert.equal(checkOutcome({ available: false, warnings: ['status_error:x'] }).reason, 'status_error:x');
    // An answer that arrived unreadable was cut in transport: said so on every surface, never as silence.
    assert.equal(checkOutcome(null, Object.assign(new Error(''), { unreadable: true })).reason,
        'the response to the update check could not be read');
    assert.equal(checkFailure(null, {}, Object.assign(new Error(''), { unreadable: true })).text,
        'the response to the update check could not be read', 'the panel\'s record and the row say one thing');
    assert.equal(checkFailureReason({ warnings: ['fetch_error:x'] }), 'fetch_error:x');
    // An install that offers no in-app update is no failure (and never a network one).
    for (const answer of [{ managed: false, check_ok: false, warnings: ['remote_config_error:x'] },
        { managed: true, check_ok: true, warnings: ['managed_updates_unavailable'] },
        { managed: true, check_ok: true, available: true, managed_by_deployment: 'Updated by your administrator.' }]) {
        assert.deepEqual(checkOutcome(answer), { kind: 'unoffered' }, JSON.stringify(answer));
    }
});

test('an answer with no check result is a failed check: never Up to date, and the version stays its retry', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        const answers = [{ managed: true, available: false }, { update_tx: { active: true } },
            { managed: true, available: false, check_ok: null, checked_at: 't0' }];
        for (const [i, answer] of answers.entries()) {
            parts().check.click();
            assert.equal(world.checks.length, i + 1, 'one request per press');
            world.checks[i].resolve(answer);
            await settle();
            const { check, note, said, version, pill } = parts();
            assert.deepEqual([note.hidden, note.textContent, note.dataset.tone], [false, 'Check failed', 'error'], JSON.stringify(answer));
            assert.equal(note.title, 'Could not check the official update channel: the check did not complete');
            assert.equal(said.textContent, note.title, 'announced as the failure it shows');
            assert.doesNotMatch(`${note.textContent} ${said.textContent}`, /Up to date/);
            assert.deepEqual([check.hidden, version.parentElement, pill.hidden], [false, check, true], 'the version is the retry');
        }
    });
});

test('a partial answer creates no check fact: an offer it never established paints no capsule', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        // A readable status whose check result is unknown: this row holds no availability at all.
        await show({ managed: true, checked_at: '' });
        assert.equal(parts().check.hidden, false, 'the plain version is the check');
        parts().check.click();
        world.checks[0].resolve({ managed: true, available: true, check_ok: false, warnings: ['fetch_error: no network'] });
        await settle();
        const { note, pill, check } = parts();
        assert.deepEqual([note.hidden, note.textContent, note.dataset.tone], [false, 'Check failed', 'error']);
        assert.equal(pill.hidden, true, 'an incomplete answer never offers an update it did not establish');
        assert.equal(check.hidden, false, 'the version stays its retry');
    });
});

test('unmanaged and organization-managed installs offer no check and no Update; an attempt or a restart still shows', async () => {
    for (const status of [{ managed: false, available: false }, { managed: true, available: false, warnings: ['managed_updates_unavailable'] },
        { managed: true, available: true, local_changes: 'absent', managed_by_deployment: 'Updated by your administrator.' }]) {
        await withIndicator(async ({ row, parts, show, world }) => {
            await show(status);
            let { check, version, pill, note } = parts();
            assert.deepEqual([check.hidden, version.parentElement, pill.hidden, note.hidden], [true, row, true, true],
                `a readable plain version: ${JSON.stringify(status)}`);
            check.click();
            assert.equal(world.checks.length, 0, 'nothing to check here');
            // A needed restart and an attempt are still said where such an install has them.
            await show({ ...status, update_progress: { operation_id: 'r', result: 'restart_required', restart_required: true,
                active: false, stage: 'stopping_services' } });
            ({ check, pill } = parts());
            assert.deepEqual([pill.hidden, pill.textContent, check.hidden], [false, 'Restart needed', true]);
            await show({ ...status, update_progress: { operation_id: 'u', active: true } });
            assert.deepEqual([parts().pill.hidden, parts().pill.textContent], [false, 'Updating…']);
        });
    }
});

test('a check that finds no in-app update here says why, never a failure, and focus lands on the last row', async () => {
    await withIndicator(async ({ env, row, settings, parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.focus();
        parts().check.click();
        world.checks[0].resolve({ managed: false, available: false, check_ok: false, warnings: ['remote_config_error:no remote'] });
        await settle();
        const { check, version, note, said, pill } = parts();
        assert.deepEqual([check.hidden, version.parentElement, note.hidden, pill.hidden], [true, row, true, true]);
        assert.equal(said.textContent, 'Managed updates are unavailable for this checkout.');
        assert.equal(env.doc.activeElement, settings, 'the control that held focus is gone: focus stays in the column');
    });
});

test('a needed restart or a running attempt takes the row from the version; a long outcome keeps its close', async () => {
    await withIndicator(async ({ row, parts, show, world }) => {
        await show({ managed: true, available: true, local_changes: 'absent', checked_at: 't0',
            update_progress: { operation_id: 'r', result: 'restart_required', restart_required: true, active: false } });
        let { pill, check, version } = parts();
        assert.deepEqual([pill.textContent, pill.dataset.tone, check.hidden, version.parentElement], ['Restart needed', 'neutral', true, row]);
        assert.ok(row.querySelector('.update-notice-close').hidden, 'a needed restart has no close');
        await show({ managed: true, available: false, checked_at: 't0', update_progress: { operation_id: 'x', outcome: 'unknown', active: false,
            reason: 'startup_unconfirmed' } });
        ({ pill, check } = parts());
        assert.deepEqual([pill.textContent, check.hidden], ['Update outcome unconfirmed', true]);
        assert.equal(row.querySelector('.update-notice-close').hidden, false, 'a concluded outcome keeps its ×');
        check.click();
        assert.equal(world.checks.length, 0, 'the hidden version starts nothing');
    });
});

test('the check is one request for every surface: a second call while one runs shares its answer', async () => {
    const prior = globalThis.fetch;
    const posts = [];
    let answer;
    globalThis.fetch = (url, init) => {
        posts.push([url, init?.method]);
        return new Promise((resolve) => { answer = resolve; });
    };
    try {
        const first = apiClient.updateCheck();
        const second = apiClient.updateCheck();
        assert.equal(posts.length, 1);
        answer({ ok: true, status: 200, json: async () => ({ managed: true, available: false, check_ok: true }) });
        assert.deepEqual(await first, await second);
        assert.deepEqual(posts, [['/api/update/check', 'POST']]);
        const third = apiClient.updateCheck();
        assert.equal(posts.length, 2, 'after the answer a call checks again');
        answer({ ok: true, status: 200, json: async () => ({ managed: true, available: false, check_ok: true }) });
        await third;
    } finally { globalThis.fetch = prior; }
});

test('checks coalesce per channel selection: another selection never shares an old one\'s answer', async () => {
    const prior = globalThis.fetch;
    const posts = [];
    const answers = [];
    globalThis.fetch = (url, init) => {
        posts.push([url, init?.method]);
        return new Promise((resolve) => { answers.push(resolve); });
    };
    try {
        const stable = apiClient.updateCheck('stable');
        const alsoStable = apiClient.updateCheck('stable');
        const dev = apiClient.updateCheck('development');
        assert.equal(posts.length, 2, 'the same selection shares one request; another fetches its own');
        answers[0]({ ok: true, status: 200, json: async () => ({ managed: true, available: false, check_ok: true, update_channel: 'stable' }) });
        answers[1]({ ok: true, status: 200, json: async () => ({ managed: true, available: false, check_ok: true, update_channel: 'development' }) });
        const [a, b, c] = await Promise.all([stable, alsoStable, dev]);
        assert.equal(a.update_channel, 'stable');
        assert.equal(b.update_channel, 'stable', 'the shared answer is the same selection\'s');
        assert.equal(c.update_channel, 'development', 'the new selection gets its own answer, never the old one\'s');
        const again = apiClient.updateCheck('stable');
        assert.equal(posts.length, 3, 'after the answer a call checks again');
        answers[2]({ ok: true, status: 200, json: async () => ({ managed: true, available: false, check_ok: true }) });
        await again;
    } finally { globalThis.fetch = prior; }
});

test('a passive read that began while the check ran never paints over the check\'s answer', async () => {
    const env = installDomDouble();
    try {
        const settings = env.el('button', { className: 'nav-row', dataset: { navPage: 'settings' } });
        const row = env.el('div', { className: 'sidebar-foot-meta' }, env.el('span', { attrs: { id: 'nav-version' }, text: VERSION }));
        env.doc.body.append(env.el('nav', { attrs: { id: 'primary-sidebar' } }, settings, row));
        const listeners = new Map();
        const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
        const reads = [];
        const checks = [];
        const controller = initUpdateStatus({ ws,
            readStatus: () => new Promise((resolve) => reads.push(resolve)),
            checkNow: (selection) => new Promise((resolve, reject) => checks.push({ selection, resolve, reject })) });
        const $ = (id) => env.doc.getElementById(id);
        const note = () => row.querySelector('.nav-check-note');
        reads.shift()({ managed: true, available: false, checked_at: 't0' });  // the initial passive read
        await settle();
        // A progress notice begins a passive read whose answer is still pending when the owner presses.
        listeners.get('update_progress_changed')();
        $('nav-version-check').click();
        assert.equal(checks.length, 1);
        // The check answers first: the row is up to date at t2.
        checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't2' });
        await settle();
        assert.deepEqual([note().hidden, note().textContent], [false, 'Up to date']);
        assert.equal($('nav-check-said').textContent, 'Up to date');
        // The older passive read lands last with the stale cache and an offer: it paints nothing — its overlap
        // with the check's answer cannot be ordered against it, so one coalesced re-read observes the settled
        // state instead of this answer being used or discarded blindly.
        reads.shift()({ managed: true, available: true, checked_at: 't0', local_changes: 'absent', latest_version: '7.2.0' });
        await settle();
        assert.equal($('update-pill').hidden, true, 'the stale offer never shows over the completed check');
        assert.deepEqual([note().hidden, note().textContent], [false, 'Up to date'], 'the check\'s own answer stands');
        assert.equal($('nav-check-said').textContent, 'Up to date', 'the live region keeps the visible state');
        // The re-read of the overlap is that fresh observation: the newer state it fetches applies.
        reads.shift()({ managed: true, available: true, checked_at: 't2', local_changes: 'absent', latest_version: '7.2.0' });
        await settle();
        assert.deepEqual([$('update-pill').hidden, $('update-pill').textContent], [false, 'Update'],
            'the state the check left behind applies through the coalesced re-read');
        // A read that begins after the answer by its own notice applies again as before.
        listeners.get('update_progress_changed')();
        await settle();
        reads.shift()({ managed: true, available: true, checked_at: 't2', local_changes: 'absent', latest_version: '7.2.0' });
        await settle();
        assert.equal($('update-pill').hidden, false, 'a later read applies once it no longer races a check');
        controller.dispose();
    } finally {
        env.restore();
    }
});

test('a check answer delayed in transit cannot erase an attempt or a needed restart a later read already read', async () => {
    // The server assembles a check's answer from the state it sampled; the reply can then be delayed while a
    // progress notice makes the row read a running attempt. The answer is still this row's own check, but the state
    // it sampled before that attempt existed never replaces what the row read later.
    const stopped = { operation_id: 'attempt-live', generation: 'server-a', stage: 'applying', active: true,
        started_at: '2026-11-21T21:00:00Z', stage_started_at: '2026-11-21T21:00:05Z' };
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.click();
        assert.equal(world.checks.length, 1);
        // The update starts while the check is in flight: the notice's read shows the stage.
        await show({ managed: true, available: false, checked_at: 't0', update_progress: stopped,
            update_tx: { active: true, phase: 'applying', attempt_id: 'tx-1' } });
        assert.deepEqual([parts().pill.hidden, parts().pill.textContent], [false, 'Applying update…']);
        world.checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't1' });
        await settle();
        assert.deepEqual([parts().pill.hidden, parts().pill.textContent], [false, 'Applying update…'],
            'the answer sampled before the attempt does not erase it');
    });
    // The same delayed answer cannot erase a restart the row has already read either.
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.click();
        await show({ managed: true, available: false, checked_at: 't0',
            update_progress: { operation_id: 'attempt-r', generation: 'server-a', result: 'restart_required',
                restart_required: true, active: false, stage: 'stopping_services', started_at: '2026-11-21T21:00:00Z' } });
        assert.equal(parts().pill.textContent, 'Restart needed');
        world.checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't1' });
        await settle();
        assert.equal(parts().pill.textContent, 'Restart needed', 'the needed restart still takes the row');
    });
    // A reviewed resolution is a transaction bound to its attempt: the delayed answer carries neither of them away.
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.click();
        await show({ managed: true, available: false, checked_at: 't0',
            update_progress: { ...stopped, active: false, result: 'assisted_started', tx_attempt_id: 'tx-9' },
            update_tx: { active: true, phase: 'assisted_resolution', task_id: 'resolver-a', attempt_id: 'tx-9' } });
        assert.equal(parts().pill.textContent, 'Resolving update…');
        world.checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't1',
            update_tx: { active: false } });
        await settle();
        assert.equal(parts().pill.textContent, 'Resolving update…',
            'the transaction the row read still belongs to its attempt');
    });
});

test('an answer for the selection the row has left is not the row\'s: neither facts nor words', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0', update_channel: 'stable' });
        parts().check.click();
        assert.equal(world.checks[0].selection, 'stable');
        // Another surface changes the channel; this row reads the new one before that answer lands.
        await show({ managed: true, available: false, update_channel: 'development', latest_version: '8.0.0' });
        assert.equal(parts().check.hidden, false, 'this channel knows no offer: the version is the check again');
        // The deferred answer speaks of the channel the press named, and offers that channel's update.
        world.checks[0].resolve({ managed: true, available: true, check_ok: true, checked_at: 't1', update_channel: 'stable',
            current_version: '7.11.1', latest_version: '7.12.0', local_changes: 'absent' });
        await settle();
        assert.equal(parts().pill.hidden, true, 'another channel\'s offer never takes this row');
        assert.equal(parts().note.hidden, true, 'and none of its words are attributed to the new channel');
        assert.equal(parts().said.textContent, '', 'the live region says nothing of it either');
        // A press now is one fresh check, naming the channel the row actually holds.
        parts().check.click();
        assert.equal(world.checks.length, 2, 'one press, one check');
        assert.equal(world.checks[1].selection, 'development');
        world.checks[1].resolve({ managed: true, available: true, check_ok: true, checked_at: 't2', update_channel: 'development',
            current_version: '7.11.1', latest_version: '8.0.0', local_changes: 'absent' });
        await settle();
        assert.deepEqual([parts().pill.hidden, parts().pill.textContent, parts().pill.title],
            [false, 'Update', 'Update 7.11.1 → 8.0.0'], 'this channel\'s own answer is the row\'s');
    });
});

test('a failed check keeps the last good check\'s facts; only a genuinely newer success ends its retry', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't1', update_channel: 'stable',
            latest_version: '7.1.0', current_version: '7.1.0' });
        parts().check.click();
        assert.equal(world.checks[0].selection, 'stable', 'the press names the selection it believes current');
        world.checks[0].resolve({ managed: true, available: false, check_ok: false, warnings: ['fetch_error: no network'], update_channel: 'stable' });
        await settle();
        let { note, said } = parts();
        assert.deepEqual([note.hidden, note.textContent, note.dataset.tone], [false, 'Check failed', 'error']);
        assert.match(note.title, /fetch_error: no network/);
        assert.equal(said.textContent, note.title);
        // A passive read of the same cached check (the failure recorded no newer one) leaves the retry standing.
        await show({ managed: true, available: false, checked_at: 't1', update_channel: 'stable' });
        assert.equal(parts().note.hidden, false, 'the same stamp is the same check: no newer success');
        // An older cached stamp is not a newer success either: it never ends the failure.
        await show({ managed: true, available: false, checked_at: 't0', update_channel: 'stable' });
        assert.equal(parts().note.hidden, false, 'an older stamp never ends the failure');
        // A genuinely newer successful check — here or on another surface — ends it.
        await show({ managed: true, available: false, checked_at: 't2', update_channel: 'stable' });
        assert.equal(parts().note.hidden, true);
    });
});

test('a failure of one channel\'s selection does not stand on another\'s row', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't1', update_channel: 'stable' });
        parts().check.click();
        world.checks[0].resolve({ managed: true, available: false, check_ok: false, warnings: ['fetch_error:x'], update_channel: 'stable' });
        await settle();
        assert.equal(parts().note.hidden, false, 'its own selection keeps its failure');
        // The channel changed elsewhere (the settings path): the old selection's failure is not the new one's.
        await show({ managed: true, available: false, checked_at: 't1', update_channel: 'development' });
        assert.equal(parts().note.hidden, true, 'another selection owns the row now');
        assert.equal(parts().said.textContent, '', 'and nothing of the old failure is announced');
    });
});

test('a rapid retry replaces the standing answer, and the first result\'s timer never clears the new one', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.click();
        world.checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't1' });
        await settle();
        assert.equal(parts().note.textContent, 'Up to date');
        parts().check.click();
        world.checks[1].resolve({ managed: true, available: false, check_ok: false, warnings: ['fetch_error:x'], checked_at: 't1' });
        await settle();
        assert.deepEqual([parts().note.textContent, parts().note.dataset.tone], ['Check failed', 'error']);
        await new Promise((resolve) => setTimeout(resolve, 40));
        assert.equal(parts().note.hidden, false, 'the first result\'s timer cleared only its own');
        assert.equal(parts().note.textContent, 'Check failed');
    });
});

test('without the foot\'s row an older page keeps the capsule beside the version', async () => {
    const env = installDomDouble();
    try {
        const foot = env.el('div', { className: 'sidebar-foot' }, env.el('span', { attrs: { id: 'nav-version' }, text: VERSION }));
        env.doc.body.append(foot);
        const listeners = new Map();
        const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
        const controller = initUpdateStatus({ ws, readStatus: async () => ({ managed: true, available: true,
            local_changes: 'absent', current_version: '7.11.1', latest_version: '7.12.0' }) });
        await settle();
        const pill = env.doc.getElementById('update-pill');
        assert.deepEqual([pill.hidden, pill.textContent], [false, 'Update']);
        assert.equal(pill.parentElement.className, 'update-notice');
        assert.equal(pill.parentElement.parentElement, foot, 'the old page keeps the capsule in the version\'s own foot');
        assert.equal(env.doc.getElementById('nav-version').nextElementSibling, pill.parentElement,
            'beside the version, where it always stood');
        controller.dispose();
    } finally {
        env.restore();
    }
});

test('disposed, the row keeps its version as plain text and no check is left to press', async () => {
    const env = installDomDouble();
    try {
        const row = env.el('div', { className: 'sidebar-foot-meta' }, env.el('span', { attrs: { id: 'nav-version' }, text: VERSION }));
        env.doc.body.append(row);
        let checks = 0;
        const controller = initUpdateStatus({ readStatus: async () => ({ managed: true, available: false }),
            checkNow: async () => { checks += 1; return {}; } });
        await settle();
        controller.dispose();
        assert.equal(env.doc.getElementById('nav-version').parentElement, row);
        assert.equal(env.doc.getElementById('nav-version-check'), null);
        assert.equal(row.querySelector('.nav-check-note'), null);
        assert.equal(checks, 0);
    } finally {
        env.restore();
    }
});

// The R2 witnesses: one test per defect, in the same row world.

test('a notification read that overlapped a check answer is re-read once, not silently discarded', async () => {
    // The GET began before the press answered and returns after, carrying the server's newer attempt state (a
    // needed restart the check's own sample never saw) beside its older tip facts. Neither painting it over the
    // settled check nor dropping it whole is right: exactly one coalesced re-read observes the settled state.
    const env = installDomDouble();
    try {
        const settings = env.el('button', { className: 'nav-row', dataset: { navPage: 'settings' } });
        const row = env.el('div', { className: 'sidebar-foot-meta' }, env.el('span', { attrs: { id: 'nav-version' }, text: VERSION }));
        env.doc.body.append(env.el('nav', { attrs: { id: 'primary-sidebar' } }, settings, row));
        const listeners = new Map();
        const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
        const reads = [];
        const checks = [];
        const controller = initUpdateStatus({ ws,
            readStatus: () => new Promise((resolve) => reads.push(resolve)),
            checkNow: (selection) => new Promise((resolve, reject) => checks.push({ selection, resolve, reject })) });
        const $ = (id) => env.doc.getElementById(id);
        reads.shift()({ managed: true, available: false, checked_at: 't0' });  // the initial passive read
        await settle();
        assert.equal($('update-pill').hidden, true);
        // A progress notice begins a read; the owner presses while it is pending.
        listeners.get('update_progress_changed')();
        $('nav-version-check').click();
        checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't1' });
        await settle();
        // The overlapped read lands with the older tip but a newer attempt than the check sampled.
        reads.shift()({ managed: true, available: false, checked_at: 't0',
            update_progress: { operation_id: 'attempt-r', generation: 'server-a', result: 'restart_required',
                restart_required: true, active: false, stage: 'stopping_services', started_at: '2026-11-21T21:00:00Z' } });
        await settle();
        assert.equal($('update-pill').hidden, true, 'the overlapped answer paints nothing over the settled check');
        assert.equal(reads.length, 1, 'exactly one coalesced re-read is pending — no polling');
        // The re-read delivers the state the check left behind: the newer attempt survives the overlap.
        reads.shift()({ managed: true, available: false, checked_at: 't1',
            update_progress: { operation_id: 'attempt-r', generation: 'server-a', result: 'restart_required',
                restart_required: true, active: false, stage: 'stopping_services', started_at: '2026-11-21T21:00:00Z' } });
        await settle();
        assert.deepEqual([$('update-pill').hidden, $('update-pill').textContent], [false, 'Restart needed'],
            'the newer attempt the GET fetched is not lost with the GET');
        controller.dispose();
    } finally {
        env.restore();
    }
});

test('a late check answer never overwrites equal-stamp checkout observations', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        // The row reads the channel at t1 with a clean checkout: the check is pressable.
        await show({ managed: true, available: false, check_ok: true, checked_at: 't1', local_changes: 'absent',
            update_channel: 'stable' });
        parts().check.click();
        // While the check runs, a notice's read observes the checkout at the same tip stamp: local work appeared.
        await show({ managed: true, available: true, check_ok: true, checked_at: 't1', local_changes: 'present',
            dirty: true, update_channel: 'stable' });
        let { pill } = parts();
        assert.ok(pill.classList.contains('has-local'), 'the row\'s later reading of the checkout stands');
        // The check's own answer for the same tip arrives late: its checkout sample predates the read above (the
        // stamp is the fetched tip, not the checkout's snapshot moment) and must not revert the later reading.
        world.checks[0].resolve({ managed: true, available: true, check_ok: true, checked_at: 't1',
            local_changes: 'absent', dirty: false, update_channel: 'stable' });
        await settle();
        ({ pill } = parts());
        assert.ok(pill.classList.contains('has-local'), 'an equal stamp does not revert the later checkout reading');
        assert.ok(!pill.classList.contains('btn-primary'), 'the older sample does not take the caveat\'s place');
        // A genuinely newer stamp is the check's own sample of the checkout and does own it.
        await show({ managed: true, available: false, check_ok: true, checked_at: 't2', local_changes: 'present',
            dirty: true, update_channel: 'stable' });
        assert.equal(parts().check.hidden, false, 'no capsule above it: the row\'s own check is back on its own');
        parts().check.click();
        world.checks[1].resolve({ managed: true, available: true, check_ok: true, checked_at: 't3',
            local_changes: 'absent', dirty: false, update_channel: 'stable' });
        await settle();
        ({ pill } = parts());
        assert.ok(!pill.classList.contains('has-local'), 'a newer stamp\'s own sample takes over');
        assert.ok(pill.classList.contains('btn-primary'));
    });
});

test('the first confirmed stamp retires a failure that had no last good check to name', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        // An install whose only check failed: no good check ever completed, so the failure carries no stamp to
        // compare against. The passive status route proves a completed check by its cache-proven `checked_at`
        // alone and always carries `check_ok: null`; the row reads that shape, never a POST's flags.
        await show({ managed: true, available: false, check_ok: null, update_channel: 'stable' });
        parts().check.click();
        world.checks[0].resolve({ managed: true, available: false, check_ok: false,
            warnings: ['fetch_error: no network'], update_channel: 'stable' });
        await settle();
        let { note, said } = parts();
        assert.deepEqual([note.hidden, note.textContent, note.dataset.tone], [false, 'Check failed', 'error']);
        // Repeats with no confirmed stamp keep the failure: an un-stamped failure never retires itself.
        await show({ managed: true, available: false, check_ok: null, update_channel: 'stable' });
        ({ note } = parts());
        assert.equal(note.hidden, false, 'a repeat with no newer good check keeps the failure');
        // The first confirmed stamp where none stood before — the passive route's own timestamp, `check_ok`
        // null — retires the failure on the same channel, visible and announced alike.
        await show({ managed: true, available: false, check_ok: null, checked_at: 't5', update_channel: 'stable' });
        ({ note, said } = parts());
        assert.equal(note.hidden, true, 'the first confirmed check ends the un-stamped failure');
        assert.equal(said.textContent, '', 'nothing of the failure is announced anymore');
        // A failure that does name a last good stamp keeps it on the same stamp: only a newer one ends it.
        parts().check.click();
        world.checks[1].resolve({ managed: true, available: false, check_ok: false,
            warnings: ['fetch_error: still no network'], update_channel: 'stable' });
        await settle();
        await show({ managed: true, available: false, check_ok: null, checked_at: 't5', update_channel: 'stable' });
        assert.equal(parts().note.hidden, false, 'the same stamp is the same check: the failure stands');
        await show({ managed: true, available: false, check_ok: null, checked_at: 't6', update_channel: 'stable' });
        assert.equal(parts().note.hidden, true, 'a genuinely newer confirmed check ends it');
    });
});

test('a late check answer never revives an offer the row already read consumed at the same stamp', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        // Nothing on offer yet, so the version is the check; the owner presses.
        await show({ managed: true, available: false, check_ok: null, update_channel: 'stable' });
        parts().check.click();
        // The check completes at t1 and finds an update; before its answer arrives the update is consumed and
        // a notice's read recomputes availability at the same disclosed stamp: no offer anymore.
        await show({ managed: true, available: false, check_ok: null, checked_at: 't1', update_channel: 'stable' });
        assert.equal(parts().pill.hidden, true);
        // The delayed answer still carries its own older sample — the tip as it was when the check ran.
        world.checks[0].resolve({ managed: true, available: true, check_ok: true, checked_at: 't1',
            local_changes: 'absent', latest_version: '7.2.0', update_channel: 'stable' });
        await settle();
        assert.equal(parts().pill.hidden, true, 'the consumed offer is not revived by the older sample');
        assert.equal(parts().said.textContent, '', 'nothing of it is announced either');
    });
});

test('a transient Up to date retires with the selection its press answered for', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, check_ok: null, checked_at: 't1', update_channel: 'stable' });
        parts().check.click();
        world.checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't2',
            update_channel: 'stable' });
        await settle();
        let { note, said } = parts();
        assert.deepEqual([note.hidden, note.textContent], [false, 'Up to date']);
        assert.equal(said.textContent, 'Up to date');
        // The channel changed elsewhere; this row's next read names the new one: the old selection's
        // transient success — visible and announced alike — is not the new channel's word.
        await show({ managed: true, available: false, check_ok: null, update_channel: 'development' });
        ({ note, said } = parts());
        assert.equal(note.hidden, true, 'the success retires with its selection');
        assert.equal(said.textContent, '');
    });
});

test('a passive read that names another channel carries nothing of the one before it', async () => {
    await withIndicator(async ({ parts, show }) => {
        // Stable was checked at t9 and offers an update; the row holds that reading.
        await show({ managed: true, available: true, check_ok: null, checked_at: 't9', local_changes: 'absent',
            update_channel: 'stable', latest_version: '7.2.0' });
        assert.equal(parts().pill.hidden, false, 'the offer stands on the row that read it');
        // The channel changed elsewhere; the first Development read carries Development's own older cache
        // (checked at t2, nothing on offer). Stable's newer stamp, offer and checkout sample are not carried
        // into it: the new selection owns nothing until its own check runs.
        await show({ managed: true, available: false, check_ok: null, checked_at: 't2', local_changes: 'present',
            dirty: true, update_channel: 'development' });
        const { pill, note, said } = parts();
        assert.equal(pill.hidden, true, 'the old channel\'s offer never stands on the new one\'s row');
        assert.ok(!pill.classList.contains('has-local'), 'nor its checkout sample');
        assert.equal(note.hidden, true);
        assert.equal(said.textContent, '');
    });
});

test('a resolved POST that names another channel is not the pressing selection\'s answer', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, check_ok: null, checked_at: 't1', update_channel: 'stable' });
        parts().check.click();
        assert.equal(world.checks[0].selection, 'stable');
        const reads = world.reads;
        // The channel changed on the server after the press was sent, and the check ran on the new one: its
        // resolved answer names development and offers that channel's update.
        world.status = { managed: true, available: false, check_ok: null, update_channel: 'development' };
        world.checks[0].resolve({ managed: true, available: true, check_ok: true, checked_at: 't9',
            latest_version: '8.0.0', local_changes: 'absent', update_channel: 'development' });
        await settle();
        let { pill, note, said } = parts();
        assert.equal(pill.hidden, true, 'another channel\'s answer paints no offer on this row');
        assert.equal(note.hidden, true, 'and claims no outcome for the selection that pressed');
        assert.equal(said.textContent, '');
        // The row learns the state the parallel change left behind through one passive re-read — never a
        // second POST — and a press after it names the channel the row now holds.
        assert.equal(world.checks.length, 1, 'no extra POST');
        assert.equal(world.reads, reads + 1, 'exactly one passive re-read');
        assert.equal(parts().check.hidden, false, 'the new channel\'s row is at rest and checkable');
        parts().check.click();
        assert.equal(world.checks[1].selection, 'development');
        world.checks[1].resolve({ managed: true, available: false, check_ok: true, checked_at: 't9',
            update_channel: 'development' });
        await settle();
        assert.deepEqual([parts().note.hidden, parts().note.textContent], [false, 'Up to date'],
            'the new channel\'s own answer is the row\'s');
    });
});

test('a saved channel selection invalidates the old selection\'s outcomes and in-flight answers', async () => {
    const channelSaved = () => globalThis.window.emit('ouro:settings-updated',
        { detail: { reason: 'setting saved', source: 'settings-row', row: 'OUROBOROS_UPDATE_CHANNEL',
            page: 'dashboard:updates', keys: ['OUROBOROS_UPDATE_CHANNEL'] } });
    // A standing offer of the old selection, then the accepted channel save: the offer, its words and every
    // outcome kind are invalidated at once, and the row reads the new selection's own state.
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: true, check_ok: true, checked_at: 't1', local_changes: 'absent',
            update_channel: 'stable', latest_version: '7.2.0' });
        let { pill, note, said } = parts();
        assert.deepEqual([pill.hidden, pill.textContent], [false, 'Update']);
        const reads = world.reads, checks = world.checks.length;
        world.status = { managed: true, available: false, check_ok: true, checked_at: '', update_channel: 'development' };
        channelSaved();
        await settle();
        ({ pill, note, said } = parts());
        assert.equal(pill.hidden, true, 'the old selection\'s offer never stands on the new one\'s row');
        assert.equal(note.hidden, true, 'no outcome of the old selection is said');
        assert.equal(said.textContent, '', 'the ARIA reading is invalidated with the visible one');
        assert.equal(world.checks.length, checks, 'the save runs no check');
        assert.equal(world.reads, reads + 1, 'one read of the new selection\'s own state');
        assert.equal(parts().check.hidden, false, 'the new selection\'s row is at rest and checkable');
        // A press for the new selection answers normally: the invalidation does not break the row.
        parts().check.click();
        world.checks[world.checks.length - 1].resolve({ managed: true, available: false, check_ok: true,
            checked_at: 't9', update_channel: 'development' });
        await settle();
        assert.equal(parts().note.hidden, false, 'the new selection\'s own answer applies');
    });
    // A press in flight when the save lands: neither its response nor its rejection is the row's anymore.
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't1', update_channel: 'stable' });
        parts().check.click();
        world.status = { managed: true, available: false, check_ok: true, checked_at: '', update_channel: 'development' };
        globalThis.window.emit('ouro:settings-updated',
            { detail: { source: 'settings-row', row: 'OUROBOROS_UPDATE_CHANNEL', page: 'dashboard:updates' } });
        world.checks[0].resolve({ managed: true, available: true, check_ok: true, checked_at: 't9',
            update_channel: 'stable', local_changes: 'absent', latest_version: '7.2.0' });
        await settle();
        assert.equal(parts().pill.hidden, true, 'a stale response on the old selection paints nothing');
        assert.equal(parts().note.hidden, true, 'and says no words');
        // The stale rejection of the old selection is not the new selection\'s failure either.
        parts().check.click();
        world.status = { managed: true, available: false, check_ok: true, checked_at: '', update_channel: 'beta' };
        globalThis.window.emit('ouro:settings-updated', { detail: { source: 'settings-row', keys: ['OUROBOROS_UPDATE_CHANNEL'] } });
        world.checks[1].reject(new Error('Server unavailable'));
        await settle();
        assert.equal(parts().note.hidden, true, 'a stale rejection is not attributed to the new selection');
        assert.equal(parts().said.textContent, '');
    });
});

test('the pill losing visibility hands focus to the visible check or the last row, only when it held it', async () => {
    await withIndicator(async ({ env, row, settings, parts, show }) => {
        await show({ managed: true, available: false,
            update_progress: { operation_id: 'a', generation: 'g', stage: 'applying', active: true } });
        const pill = parts().pill;
        pill.focus();
        assert.equal(env.doc.activeElement, pill);
        // The attempt concludes: the success has no capsule word, nothing is on offer, the pill goes away — and
        // the row\'s own visible check inherits the focus a keyboard reader held on the pill.
        await show({ managed: true, available: false,
            update_progress: { operation_id: 'a', generation: 'g', outcome: 'applied', result: 'applied',
                active: false, concluded_by: 'boot' } });
        assert.equal(parts().pill.hidden, true);
        assert.equal(env.doc.activeElement, parts().check, 'the row\'s visible check takes the focus the pill held');
        // Only the disappearing element\'s own focus moves: a pill that never held focus steals nothing.
        await show({ managed: true, available: false,
            update_progress: { operation_id: 'b', generation: 'g', stage: 'applying', active: true } });
        row.focus();
        await show({ managed: true, available: false,
            update_progress: { operation_id: 'b', generation: 'g', outcome: 'applied', result: 'applied',
                active: false, concluded_by: 'boot' } });
        assert.equal(env.doc.activeElement, row, 'focus that was elsewhere stays where it was');
    });
});

// The R3 witnesses: the two raced directions of the checkout axis, the accepted channel save, the two directions
// of a disappearing close, and the live region's authored source — one test each, in the same row world.

test('the first stamped Check supplies its own coherent checkout facts over the unstamped first read', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        // The row's first reading is the producer's unstamped passive GET (git_ops_updates.py: no check has completed,
        // so no checked_at) with its own live `dirty` sample — clean then.
        await show({ managed: true, available: false, check_ok: null, dirty: false,
            warnings: ['official_status_requires_check'], update_channel: 'stable' });
        parts().check.click();
        // The first completed Check mints the stamp and samples the checkout itself: uncommitted work appeared while
        // it ran. Its own coherent sample stands — the earlier unstamped reading cannot overwrite it.
        world.checks[0].resolve({ managed: true, available: true, check_ok: true, checked_at: 't1',
            dirty: true, local_changes: 'absent', safe_to_apply: false, latest_version: '7.2.0',
            update_channel: 'stable' });
        await settle();
        const { pill, note } = parts();
        assert.ok(pill.classList.contains('has-local'), 'the check\'s own sample keeps the amber caveat');
        assert.ok(!pill.classList.contains('btn-primary'), 'not the clean offer the first read implied');
        assert.equal(note.hidden, true);
    });
});

test('an old delayed failed POST never reverts the checkout a later passive read sampled', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0', update_channel: 'stable' });
        parts().check.click();
        // While the pressed check is in flight, a notice's read samples the checkout at t1: an update is on offer
        // with uncommitted work in it (the amber caveat).
        await show({ managed: true, available: true, check_ok: null, checked_at: 't1', dirty: true,
            local_changes: 'absent', safe_to_apply: false, latest_version: '7.2.0', update_channel: 'stable' });
        assert.ok(parts().pill.classList.contains('has-local'), 'the read\'s later sample wears the caveat');
        // The delayed answer sampled before that read and failed besides: its stale `dirty: false` and its base
        // `available`/`safe_to_apply` never replace the dirt, local changes or safety the row read later.
        world.checks[0].resolve({ managed: true, available: false, check_ok: false, dirty: false,
            safe_to_apply: false, warnings: ['fetch_error: no network'], update_channel: 'stable' });
        await settle();
        const { pill } = parts();
        assert.ok(pill.classList.contains('has-local'), 'the later reading of the checkout stands');
        assert.ok(!pill.classList.contains('btn-primary'), 'not the clean offer the stale sample implied');
        assert.equal(pill.hidden, false);
    });
});

test('after an accepted channel save the version is plain until the row\'s own read names the selection', async () => {
    const env = installDomDouble();
    try {
        const settings = env.el('button', { className: 'nav-row', dataset: { navPage: 'settings' } });
        const row = env.el('div', { className: 'sidebar-foot-meta' }, env.el('span', { attrs: { id: 'nav-version' }, text: VERSION }));
        env.doc.body.append(env.el('nav', { attrs: { id: 'primary-sidebar' } }, settings, row));
        const listeners = new Map();
        const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
        const reads = [];
        const checks = [];
        const controller = initUpdateStatus({ ws,
            readStatus: () => new Promise((resolve, reject) => reads.push({ resolve, reject })),
            checkNow: (selection) => new Promise((resolve, reject) => checks.push({ selection, resolve, reject })) });
        const $ = (id) => env.doc.getElementById(id);
        const check = () => $('nav-version-check');
        const note = () => row.querySelector('.nav-check-note');
        const saved = (shape) => globalThis.window.emit('ouro:settings-updated',
            { detail: { reason: 'setting saved', source: 'settings-row', ...shape } });
        reads.shift().resolve({ managed: true, available: false, checked_at: 't1', update_channel: 'stable' });
        await settle();
        assert.deepEqual([check().hidden, $('nav-version').parentElement], [false, check()],
            'the stable row is at rest and checkable');
        // The Dashboard card's save is one entry (row); the Settings page's is the other (keys). While the save's own
        // read is pending, no press can name a selection the row cannot vouch for: the stale channel keys no POST.
        saved({ row: 'OUROBOROS_UPDATE_CHANNEL', page: 'dashboard:updates' });
        assert.deepEqual([check().hidden, $('nav-version').parentElement], [true, row],
            'the accepted save leaves the selection unknown: the version is the plain text it was');
        check().click();
        assert.equal(checks.length, 0, 'no POST keys the stale selection');
        // The read names the channel this install holds now — the only evidence a press keys.
        reads.shift().resolve({ managed: true, available: false, check_ok: null, update_channel: 'development' });
        await settle();
        assert.equal(check().hidden, false);
        check().click();
        assert.equal(checks[0].selection, 'development', 'the press names the read\'s selection');
        checks[0].resolve({ managed: true, available: false, check_ok: true, checked_at: 't2', update_channel: 'development' });
        await settle();
        assert.equal(note().textContent, 'Up to date');
        // A save whose read fails proves nothing: the version stays plain, and a later read re-arms the row.
        saved({ keys: ['OUROBOROS_UPDATE_CHANNEL'] });
        assert.equal(check().hidden, true);
        check().click();
        assert.equal(checks.length, 1, 'still no POST without evidence');
        reads.shift().reject(new Error('offline'));
        await settle();
        assert.equal(check().hidden, true, 'a failed read leaves the channel unknown');
        listeners.get('update_status_ready')();
        await settle();
        reads.shift().resolve({ managed: true, available: false, check_ok: null, update_channel: 'beta' });
        await settle();
        assert.equal(check().hidden, false);
        check().click();
        assert.equal(checks[1].selection, 'beta');
        checks[1].resolve({ managed: true, available: false, check_ok: true, checked_at: 't3', update_channel: 'beta' });
        await settle();
        assert.equal(note().textContent, 'Up to date');
        controller.dispose();
    } finally {
        env.restore();
    }
});

test('a press the save overtakes completes with no outcome, and the next press is the new selection\'s own', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't1', update_channel: 'stable' });
        parts().check.click();
        assert.equal(parts().check.getAttribute('aria-busy'), 'true');
        // The channel is saved while the press is in flight; the row reads the new selection before the answer lands.
        world.status = { managed: true, available: false, check_ok: null, update_channel: 'development' };
        globalThis.window.emit('ouro:settings-updated',
            { detail: { source: 'settings-row', row: 'OUROBOROS_UPDATE_CHANNEL', page: 'dashboard:updates' } });
        world.checks[0].resolve({ managed: true, available: true, check_ok: true, checked_at: 't9',
            latest_version: '7.2.0', local_changes: 'absent', update_channel: 'stable' });
        await settle();
        let { check, pill, note, said } = parts();
        assert.equal(check.getAttribute('aria-busy'), null, 'the row repaints without the stale press\'s busy mark');
        assert.equal(check.hidden, false, 'the read re-armed the row');
        assert.equal(pill.hidden, true, 'the stale answer painted no offer');
        assert.equal(note.hidden, true, 'and wrote no outcome');
        assert.equal(said.textContent, '');
        // The next valid press is the new selection's own and completes normally.
        check.click();
        assert.equal(world.checks[1].selection, 'development');
        world.checks[1].resolve({ managed: true, available: false, check_ok: true, checked_at: 't2',
            update_channel: 'development' });
        await settle();
        ({ note, check } = parts());
        assert.deepEqual([note.hidden, note.textContent], [false, 'Up to date']);
        assert.equal(check.getAttribute('aria-busy'), null);
    });
});

test('the old selection\'s unanswered check neither blocks nor clears the new selection\'s press', async () => {
    // After an accepted save and the read that names the new selection, the version is the check again even though
    // the old press has not replied. That press belongs to the epoch the row left: its busy state is not this
    // selection's, and its late response or rejection writes no outcome and clears nothing of the newer press.
    for (const late of ['resolve', 'reject']) {
        await withIndicator(async ({ parts, show, world }) => {
            await show({ managed: true, available: false, checked_at: 't1', update_channel: 'stable' });
            parts().check.click();
            assert.equal(world.checks.length, 1);
            assert.equal(world.checks[0].selection, 'stable');
            assert.equal(parts().check.getAttribute('aria-busy'), 'true', 'A is in flight');
            // The channel is saved elsewhere while A is unanswered; the row stays plain until its own read names it.
            world.status = { managed: true, available: false, check_ok: null, update_channel: 'development' };
            globalThis.window.emit('ouro:settings-updated',
                { detail: { source: 'settings-row', row: 'OUROBOROS_UPDATE_CHANNEL', page: 'dashboard:updates' } });
            assert.equal(parts().check.hidden, true, 'the accepted save leaves the selection unknown');
            await settle();  // the save\'s own read names B
            assert.equal(parts().check.hidden, false, 'the read names B: the version is the check again');
            assert.equal(parts().check.getAttribute('aria-busy'), null, 'A\'s pending press is not this selection\'s busy mark');
            // B is pressable while A has not replied: exactly one POST, naming B.
            parts().check.click();
            assert.equal(world.checks.length, 2, 'one press, one check');
            assert.equal(world.checks[1].selection, 'development');
            assert.equal(parts().check.getAttribute('aria-busy'), 'true', 'B is busy');
            assert.equal(parts().note.hidden, true, 'nothing is claimed while B checks');
            // A answers late: B keeps its busy state and no word of A reaches the row\'s visible or ARIA reading.
            if (late === 'resolve') {
                world.checks[0].resolve({ managed: true, available: true, check_ok: true, checked_at: 't9',
                    update_channel: 'stable', local_changes: 'absent', latest_version: '7.2.0' });
            } else {
                world.checks[0].reject(new Error('Server unavailable'));
            }
            await settle();
            assert.equal(parts().check.getAttribute('aria-busy'), 'true', 'B is still the check in flight');
            assert.equal(parts().pill.hidden, true, 'A painted no offer');
            assert.equal(parts().note.hidden, true, 'and wrote no outcome');
            assert.equal(parts().said.textContent, '', 'nothing of A is announced');
            parts().check.click();
            assert.equal(world.checks.length, 2, 'A did not clear B\'s busy state: the press coalesced');
            // B completes normally and owns the row\'s answer.
            world.checks[1].resolve({ managed: true, available: false, check_ok: true, checked_at: 't3',
                update_channel: 'development' });
            await settle();
            assert.equal(parts().check.getAttribute('aria-busy'), null);
            assert.deepEqual([parts().note.hidden, parts().note.textContent], [false, 'Up to date']);
            assert.equal(parts().said.textContent, 'Up to date');
        });
    }
});

test('the disappearing close hands focus to the capsule, the row\'s check or the last row, only when it held it', async () => {
    await withIndicator(async ({ env, row, settings, parts, show, close }) => {
        await show({ managed: true, available: false, update_progress: { ...refusal } });
        // A new attempt starts elsewhere while the reader's focus is on the ×: the notice becomes a stage.
        close().focus();
        assert.equal(env.doc.activeElement, close());
        await show({ managed: true, available: false,
            update_progress: { operation_id: 'b', generation: 'g', stage: 'applying', active: true } });
        assert.equal(close().hidden, true);
        assert.equal(env.doc.activeElement, parts().pill, 'the capsule the × stood beside inherits the focus');
        // The attempt concludes with nothing on offer: the pill goes away, and focus held elsewhere never moves.
        row.focus();
        await show({ managed: true, available: false,
            update_progress: { operation_id: 'b', generation: 'g', outcome: 'applied', result: 'applied',
                active: false, concluded_by: 'boot' } });
        assert.equal(env.doc.activeElement, row, 'no element steals focus it did not hold');
        // Explicit dismiss with nothing beneath: the row's own visible check is the landing, not the page.
        await show({ managed: true, available: false, update_progress: { ...refusal, operation_id: 'attempt-c' } });
        close().focus();
        close().click();
        assert.equal(env.doc.activeElement, parts().check, 'the row\'s visible check, the retry the failure left');
        // An install that offers no check has only the sidebar's last row to land on.
        await show({ managed: false, available: false, update_progress: { ...refusal, operation_id: 'attempt-u' } });
        close().focus();
        close().click();
        assert.equal(env.doc.activeElement, settings, 'the last navigation row when the row offers nothing else');
    });
});

test('the live region dedupes by the authored source: a translation in place survives later visits', async () => {
    await withIndicator(async ({ parts, show, world }) => {
        await show({ managed: true, available: false, checked_at: 't0' });
        parts().check.click();
        world.checks[0].resolve({ managed: true, available: false, check_ok: false, warnings: ['fetch_error:x'] });
        await settle();
        let { said } = parts();
        assert.equal(said.textContent, 'Could not check the official update channel: fetch_error:x');
        // i18n's overlay rewrites the region's text in place (its MutationObserver watches characterData): the
        // authored source alone says whether the words changed, never the translated text.
        said.textContent = 'Не удалось проверить официальный канал обновлений: fetch_error:x';
        await show({ managed: true, available: false, checked_at: 't0' });
        assert.equal(said.textContent, 'Не удалось проверить официальный канал обновлений: fetch_error:x',
            'the same authored words write nothing over the translation');
        // A genuinely new authored reading still writes.
        parts().check.click();
        world.checks[1].resolve({ managed: true, available: false, check_ok: true, checked_at: 't1' });
        await settle();
        assert.equal(parts().said.textContent, 'Up to date');
    });
});

test('an unoffered install\'s static reason is authored once, not re-announced on every visit', async () => {
    await withIndicator(async ({ parts, show }) => {
        await show({ managed: false, available: false });
        const { said } = parts();
        assert.equal(said.textContent, 'Managed updates are unavailable for this checkout.');
        said.textContent = 'Управляемые обновления недоступны для этой рабочей копии.';  // the overlay translated it
        await show({ managed: false, available: false });
        await show({ managed: false, available: false, warnings: [] });
        assert.equal(said.textContent, 'Управляемые обновления недоступны для этой рабочей копии.',
            'repeat visits author the same reason and announce nothing');
    });
});
