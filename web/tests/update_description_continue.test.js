// Regression: executes the candidate's real event binding, pure panel projector,
// and exact (unmodified) loadStatus closure. The HTTP transport is controlled to
// deliver a running snapshot after the successor's terminal WebSocket frame, and an
// explicit check's failure after a status notice queued behind it.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { bindUpdateRefreshEvents, checkFailure, lastObservedAttempt, retainUnreadLetter, updateLetterView, updateVerdict } from '../modules/updates.js';

const source = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8');
const start = source.indexOf('    let statusRefreshPromise = null;');
const end = source.indexOf('    // The Recovery list of earlier versions', start);
assert.ok(start >= 0 && end > start, 'the production status-loader boundaries must exist');
const loader = source.slice(start, end);
const loaderSha256 = createHash('sha256').update(loader).digest('hex');
const key = { base_sha: 'a'.repeat(40), target_sha: 'b'.repeat(40) };
const status = (taskId, taskStatus) => ({
    managed: true, available: true, check_ok: true,
    running_sha: key.base_sha, checked_target_sha: key.target_sha,
    letter: { state: 'none', text: '', checked_key: key,
        request: { task_id: 'original', current_task_id: taskId, status: taskStatus,
            mode: 'write', key, saved: false } },
});
const tick = () => new Promise((resolve) => setImmediate(resolve));

// Only the surrounding UI plumbing is substituted. The entire async loader below is
// taken byte-for-byte from initUpdates (including coalescing), with the module's own
// checkFailure it calls.
function harness(apiClient, initialStatus) {
    return new Function('apiClient', 'initialStatus', 'checkFailure', 'retainUnreadLetter', 'lastObservedAttempt', `
        let latestStatus = initialStatus;
        let phase = '';
        let restartNeeded = false;
        let letterRefreshError = null;
        let unrefreshedReceipt = null;
        const officialTagsDiv = { childElementCount: 1 };
        const versions = { renderReleases: () => {} };
        const render = () => {};
        const setPhase = (next) => { phase = next; };
        ${loader}
        return { loadStatus, getStatus: () => latestStatus, getPhase: () => phase,
            getRefreshError: () => letterRefreshError };
    `)(apiClient, initialStatus, checkFailure, retainUnreadLetter, lastObservedAttempt);
}

function client(name) {
    const requests = [];
    const apiClient = {
        updateStatus() {
            return new Promise((resolve) => requests.push({ resolve }));
        },
        updateCheck() { assert.fail('a lifecycle cue must never perform a fetching Check'); },
        updateDescription() { assert.fail('a lifecycle cue must never start duplicate work'); },
        updateDescribeLatest() { assert.fail('a lifecycle cue must never start duplicate work'); },
    };
    const state = harness(apiClient, status('original', 'cancelled'));
    const listeners = new Map();
    const binding = bindUpdateRefreshEvents({
        ws: { on(event, fn) { listeners.set(event, fn); return () => listeners.delete(event); } },
        getPhase: state.getPhase,
        getRequestTaskIds: () => [updateLetterView(state.getStatus()).request?.taskId].filter(Boolean),
        reconcileRestart() { assert.fail('not a restart'); },
        loadStatus: state.loadStatus,
    });
    return { name, requests, binding, state,
        send(data) { listeners.get('log')({ type: 'log', data, chat_id: 1 }); },
        view() { return updateLetterView(state.getStatus()); },
    };
}

const admitted = { type: 'owner_continue_admitted', task_id: 'successor',
    predecessor_task_id: 'original', owner_visible: true };
const terminal = (outcome) => ({ type: 'task_done', task_id: 'successor',
    root_task_id: 'successor', status: outcome });

test('an unreadable letter on a status notice retains the text and task until a readable answer', async () => {
    const c = client('reader');
    try {
        c.send(admitted);
        const before = status('successor', 'running');
        Object.assign(before.letter, { text: 'Saved text.', key, revision: 3, written_at: '2026-11-17T09:00:00Z',
            author: { task_id: 'author' }, description_current: true });
        c.requests[0].resolve(before);
        await tick();
        c.send(terminal('completed'));
        c.requests[1].resolve({ ...before, letter: null, letter_unavailable: true, latest_version: 'new target' });
        await tick();
        assert.equal(c.state.getStatus().latest_version, 'new target');
        assert.deepEqual(c.state.getStatus().letter.author, before.letter.author);
        assert.equal(c.state.getStatus().letter.revision, 3);
        assert.equal(c.view().meta.writtenAt, before.letter.written_at);
        assert.equal(c.view().markdown, 'Saved text.');
        assert.equal(c.view().request.taskId, 'successor');
        assert.equal(c.view().request.status, 'unknown');
        assert.equal(c.view().descriptionCurrent, false);
        assert.match(c.view().note, /last readable description/);
        c.send(terminal('completed'));
        c.requests[2].resolve({ ...before, letter: null });
        await tick();
        assert.equal(c.view().markdown, '');
        assert.equal(c.view().request, null);
    } finally { c.binding.dispose(); }
});

test('ordinary two-client admission and terminal delivery uses passive reads only', async () => {
    const clients = [client('first'), client('second')];
    try {
        for (const c of clients) {
            c.send({ ...admitted, predecessor_task_id: 'other' });
            c.send({ ...admitted, _historical: true });
            assert.equal(c.requests.length, 0);
            c.send(admitted);
            assert.equal(c.requests.length, 1);
            c.requests[0].resolve(status('successor', 'running'));
        }
        await tick();
        for (const c of clients) {
            assert.equal(c.view().action.id, 'check', 'while it writes, the action is a check that writes nothing');
            assert.equal(c.view().request.taskId, 'successor');
            c.send(terminal('cancelled'));
            assert.equal(c.requests.length, 2);
            c.requests[1].resolve(status('successor', 'cancelled'));
        }
        await tick();
        for (const c of clients) {
            assert.equal(c.view().request.status, 'cancelled');
            assert.equal(c.view().action.id, 'describe');
        }
        console.log(JSON.stringify({ scenario: 'ordinary', reads: clients.map(c => c.requests.length), loaderSha256 }));
    } finally { clients.forEach(c => c.binding.dispose()); }
});

for (const outcome of ['cancelled', 'completed']) {
    test(`a ${outcome} successor is still observed when its terminal arrives before the admission status response`, async () => {
        const fast = client('fast-response-client');
        const slow = client('delayed-response-client');
        try {
            fast.send(admitted);
            slow.send(admitted);
            // Both HTTP requests have already sampled the durable successor as
            // running. Only fast's response has reached the panel so far.
            const snapshot = status('successor', 'running');
            fast.requests[0].resolve(structuredClone(snapshot));
            await tick();
            // The successor ends (or is stopped in the other client). A fresh
            // logical root has root_task_id=successor, not the predecessor.
            fast.send(terminal(outcome));
            slow.send(terminal(outcome));
            assert.equal(fast.requests.length, 2, 'the settled fast client hears the same terminal');
            fast.requests[1].resolve(status('successor', outcome));
            slow.requests[0].resolve(structuredClone(snapshot));
            await tick();
            // If a follow-up read was requested, return the terminal truth.
            for (const pending of slow.requests.slice(1)) pending.resolve(status('successor', outcome));
            await tick();
            console.log(JSON.stringify({ scenario: 'terminal-during-admission-read', outcome, loaderSha256,
                fast: { reads: fast.requests.length, note: fast.view().note, action: fast.view().action?.id },
                slow: { reads: slow.requests.length, note: slow.view().note, action: slow.view().action?.id,
                    trackedTask: slow.view().request.taskId, trackedStatus: slow.view().request.status } }));
            assert.equal(slow.view().request.status, outcome,
                'terminal delivery must not be lost while HTTP refresh still exposes the predecessor');
            assert.equal(slow.view().action.id, 'describe');
        } finally { fast.binding.dispose(); slow.binding.dispose(); }
    });
}


test('a failed queued progress read releases the loading or checking phase and restores the plain recovery check', async () => {
    const recovery = { managed: true, update_tx: { active: true, phase: 'gate_blocked' } };
    for (const fetchRemote of [false, true]) {
        const calls = [];
        const request = method => new Promise((resolve, reject) => calls.push({ method, resolve, reject }));
        const state = harness({ updateStatus: () => request('GET'), updateCheck: () => request('POST') }, recovery);
        const events = new Map();
        const binding = bindUpdateRefreshEvents({
            ws: { on(name, handler) { events.set(name, handler); } }, getPhase: state.getPhase, loadStatus: state.loadStatus,
        });
        try {
            const finished = state.loadStatus({ fetchRemote });
            assert.equal(state.getPhase(), fetchRemote ? 'checking' : 'loading');
            events.get('update_progress_changed')();
            assert.equal(calls.length, 1);
            if (fetchRemote) calls[0].reject(new Error('Check interrupted'));
            else calls[0].resolve(recovery);
            await tick();
            assert.deepEqual(calls.map(c => c.method), [fetchRemote ? 'POST' : 'GET', 'GET']);
            calls[1].reject(new Error('HTTP 503'));
            await finished;
            const data = state.getStatus();
            assert.match(data.lastObservedAttempt.guidance, /Quit and reopen/);
            assert.deepEqual(updateVerdict(data, state.getPhase()).action, { id: 'check', label: 'Check again' });
        } finally { binding.dispose(); }
    }
});

// An explicit check that fails while a status notice queued behind it: the notice's newer read is shown (its writer
// included), and the check's own failure stays in the transient slot the panel says it from, until a later check
// that completes. Both a thrown failure and an answer that says check_ok:false.
for (const failure of ['thrown', 'check_ok:false']) {
    test(`an explicit check's ${failure} outcome outlives the status read queued behind it`, async () => {
        const checks = [];
        const reads = [];
        const apiClient = {
            updateCheck() { return new Promise((resolve, reject) => checks.push({ resolve, reject })); },
            updateStatus() { return new Promise((resolve) => reads.push({ resolve })); },
        };
        const before = { ...status('original', 'cancelled'), checked_at: '2026-11-17T08:00:00Z' };
        const state = harness(apiClient, before);
        const done = state.loadStatus({ fetchRemote: true, preservePhase: true });
        assert.equal(checks.length, 1);
        state.loadStatus({ fetchRemote: false, preservePhase: true }); // the WebSocket notice, queued behind the check
        if (failure === 'thrown') checks[0].reject(new Error('Failed to fetch'));
        else checks[0].resolve({ ...before, check_ok: false, available: false, warnings: ['fetch_error:offline'] });
        await tick();
        assert.equal(reads.length, 1, 'the queued notice still reads the status');
        reads[0].resolve({ ...status('successor', 'running'), checked_at: before.checked_at, check_ok: null });
        await done;
        // The newer read and its writer are shown; the check's failure stays said, about the last good check.
        assert.equal(updateLetterView(state.getStatus()).request.taskId, 'successor');
        const kept = state.getRefreshError();
        assert.equal(kept.text, failure === 'thrown' ? 'Failed to fetch' : 'fetch_error:offline');
        assert.equal(kept.checkedAt, before.checked_at);
        // A later explicit check that completes clears it.
        const again = state.loadStatus({ fetchRemote: true, preservePhase: true });
        checks[1].resolve({ ...status('successor', 'running'), checked_at: '2026-11-17T09:00:00Z', check_ok: true });
        await again;
        assert.equal(state.getRefreshError(), null);
        console.log(JSON.stringify({ scenario: 'explicit-check-failure', failure, loaderSha256, kept }));
    });
}
