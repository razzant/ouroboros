// Regression: executes the candidate's real event binding, pure panel projector,
// and exact (unmodified) loadStatus closure. The HTTP transport is controlled to
// deliver a running snapshot after the successor's terminal WebSocket frame.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { bindUpdateRefreshEvents, updateLetterView } from '../modules/updates.js';

const source = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8');
const start = source.indexOf('    let statusRefreshPromise = null;');
const end = source.indexOf('    function renderRestoreRow(', start);
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

function client(name) {
    const requests = [];
    const apiClient = {
        updateStatus() {
            return new Promise((resolve) => requests.push({ resolve }));
        },
        updateCheck() { assert.fail('a lifecycle cue must never perform a fetching Check'); },
        updateDescription() { assert.fail('a lifecycle cue must never start duplicate work'); },
    };
    // Only the surrounding UI plumbing is substituted. The entire async loader
    // below is taken byte-for-byte from initUpdates (including coalescing).
    const makeState = new Function('apiClient', 'initialStatus', `
        let latestStatus = initialStatus;
        let phase = '';
        let restartNeeded = false;
        let letterRefreshError = null;
        const officialTagsDiv = { childElementCount: 1 };
        const renderOfficialTags = () => {};
        const render = () => {};
        const setPhase = (next) => { phase = next; };
        const bindPreflightDescription = () => { throw new Error('unexpected preflight'); };
        ${loader}
        return { loadStatus, getStatus: () => latestStatus, getPhase: () => phase };
    `);
    const state = makeState(apiClient, status('original', 'cancelled'));
    const listeners = new Map();
    const binding = bindUpdateRefreshEvents({
        ws: { on(event, fn) { listeners.set(event, fn); return () => listeners.delete(event); } },
        getPhase: state.getPhase,
        getRequestTaskId: () => updateLetterView(state.getStatus()).request?.taskId || '',
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
            assert.equal(c.view().write.disabled, true);
            assert.equal(c.view().request.taskId, 'successor');
            c.send(terminal('cancelled'));
            assert.equal(c.requests.length, 2);
            c.requests[1].resolve(status('successor', 'cancelled'));
        }
        await tick();
        for (const c of clients) {
            assert.equal(c.view().request.status, 'cancelled');
            assert.equal(c.view().write.disabled, false);
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
                fast: { reads: fast.requests.length, note: fast.view().note, writeDisabled: fast.view().write.disabled },
                slow: { reads: slow.requests.length, note: slow.view().note, writeDisabled: slow.view().write.disabled,
                    trackedTask: slow.view().request.taskId, trackedStatus: slow.view().request.status } }));
            assert.equal(slow.view().request.status, outcome,
                'terminal delivery must not be lost while HTTP refresh still exposes the predecessor');
            assert.equal(slow.view().write.disabled, false);
        } finally { fast.binding.dispose(); slow.binding.dispose(); }
    });
}
