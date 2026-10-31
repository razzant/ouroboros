// The extracted login-card CONTROLLER (phase 2 seam). The view helpers keep
// their original assertions in harness_accounts.test.js — what is new here is
// the lifecycle the Settings section used to own privately: create → poll →
// verdict, the verify-race re-check against live account status, the store
// hold that keeps the account rows moving while a login runs, and the disposer
// that must leave nothing armed.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createClaudexorStatusStore } from '../modules/claudexor_status_store.js';
import {
    normalizeProfileName,
    profileNameSubmission,
    preserveCardFocus,
    JOB_POLL_GIVE_UP_FAILURES,
    LOGIN_CUSTODY_RELEASED,
    LOGIN_CUSTODY_RETAINED,
    LOGIN_CUSTODY_UNKNOWN,
    cancelLoginJob,
    createLoginCardController,
    loginCardHtml,
    loginReleaseProven,
    reconcileLoginJob,
    resolvedJobProfileId,
} from '../modules/harness_login_cards.js';

const json = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body });

function fakeHost() {
    return {
        innerHTML: '',
        contains: () => false,
        querySelector: () => null,
        querySelectorAll: () => [],
    };
}

function interactiveHost() {
    const listeners = new Map();
    return {
        innerHTML: '',
        contains: () => false,
        querySelector(selector) {
            const marker = selector.match(/\[([^\]]+)\]/)?.[1] || '';
            if (!marker || !this.innerHTML.includes(marker)) return null;
            return {
                open: false,
                addEventListener(type, callback) { listeners.set(`${selector}:${type}`, callback); },
            };
        },
        querySelectorAll: () => [],
        click(selector) { listeners.get(`${selector}:click`)?.({ preventDefault() {} }); },
    };
}

function statusPayload(loggedIn) {
    // The producer's unconditional shape (daemon/harnesses/profiles/quota) —
    // the store refuses to derive facets from anything less.
    return {
        daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
        config_dir: '/home/agent',
        harnesses: [{ id: 'codex' }],
        // The engine's bootstrap registry row for an empty family's login.
        profiles: {
            profiles: [{ profile: { harness_id: 'codex', profile_id: 'codex-default' },
                status: { verification: loggedIn ? 'passed' : 'not_run', verification_source: 'local_store' } }],
            harnessAccounts: [], accountPools: [],
        },
        quota: [],
    };
}

const flush = async () => { for (let i = 0; i < 40; i += 1) await Promise.resolve(); };

test('the controller drives create → poll → Signed in, and holds the status poll while it runs', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let jobState = 'running';
    let statusReads = 0;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => { statusReads += 1; return json(200, statusPayload(jobState === 'succeeded')); },
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    // Off-surface subscriber: only the login hold can make this store poll.
    store.subscribe(() => {}, { visible: () => false });
    assert.equal(store.polling, false);

    const host = fakeHost();
    let settled = 0;
    const ctl = createLoginCardController({
        host,
        store,
        onSettled: () => { settled += 1; },
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-1', job: { state: 'running', phase: 'awaiting_user' }, attach_command: '' });
            }
            if (url.startsWith('/api/claudexor/login/')) return json(200, { job: { state: jobState } });
            return json(404, {});
        },
    });

    await ctl.start('codex', '');
    assert.ok(host.innerHTML.includes('data-harness-identity="codex"'), 'the card rendered');
    assert.ok(host.innerHTML.includes('>Codex</span>'), 'the readable product name rendered');
    assert.ok(host.innerHTML.includes('data-login-state'), 'a live job shows the progress line');
    assert.equal(store.polling, true, 'a live login holds the shared status poll open');

    // The 3s job poll lands a still-running snapshot, then a succeeded one.
    t.mock.timers.tick(3000);
    await flush();
    assert.ok(!host.innerHTML.includes('data-login-verdict'), 'still pending, no verdict');
    jobState = 'succeeded';
    t.mock.timers.tick(3000);
    await flush();
    assert.ok(host.innerHTML.includes('Signed in.'), `verified state reached: ${host.innerHTML}`);
    assert.equal(settled, 1, 'the host was told to re-render its rows');
    assert.equal(store.polling, false, 'the settled login released the poll hold');
    assert.ok(statusReads >= 1, 'the verdict refreshed the shared status');

    ctl.dispose();
    store.dispose();
    t.mock.timers.reset();
});

test('poll replaces the whole canonical envelope, preserving envelope-level device disclosure', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-device', job: { state: 'running' },
                    attach_command: 'claudexor setup attach job-device', attach_shell: 'powershell',
                    setup_login_source: 'per_harness', disclosure_native: false });
            }
            if (init.method === 'DELETE') return json(200, { job: { state: 'cancelled' } });
            return json(200, {
                job: { state: 'waiting_for_input', phase: 'awaiting_user' },
                cursor: 'c1', sequence: 2,
                deviceCode: { flow: 'chatgptDeviceCode',
                    verificationUrl: 'https://auth.example/device', userCode: 'ABCD-1234' },
            });
        },
    });
    await ctl.start('codex', '');
    t.mock.timers.tick(3000);
    await flush();
    assert.equal(ctl.active?.envelope?.sequence, 2);
    assert.equal(ctl.active?.attachCommand, 'claudexor setup attach job-device',
        'replaceable poll envelope must not erase create-only metadata');
    assert.equal(ctl.active?.attachShell, 'powershell');
    assert.equal(ctl.active?.setupLoginSource, 'per_harness');
    assert.match(host.innerHTML, /data-open-signin/);
    assert.match(host.innerHTML, /ABCD-1234/);
    await ctl.dispose();
    store.dispose();
    t.mock.timers.reset();
});

test('a verify-race failure is re-checked against live account status before the card says failed', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    // codex clears its auth store when a login STARTS, so the job's own
    // verification read can say "not logged in" while the vendor login is
    // succeeding. The account rows decide, not that one stale read.
    let loggedIn = false;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(loggedIn)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            // The engine resolves the empty family's login onto its row and names it.
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-2', job: { state: 'running', profileId: 'codex-default' } });
            }
            if (url.startsWith('/api/claudexor/login/')) {
                return json(200, { job: { state: 'failed', profileId: 'codex-default',
                    outcome: { reason: 'auth_not_ready' } } });
            }
            return json(404, {});
        },
    });
    await ctl.start('codex', '');
    // The account really IS logged in by the time the re-check runs.
    loggedIn = true;
    t.mock.timers.tick(3000);
    await flush();
    assert.ok(host.innerHTML.includes('Signed in.'),
        `the verify-race must resolve to success, not "Sign-in failed": ${host.innerHTML}`);
    ctl.dispose();
    store.dispose();
    t.mock.timers.reset();
});

test('an unconfirmed re-check says unknown, never a hard failure', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-3', job: { state: 'running' } });
            }
            if (url.startsWith('/api/claudexor/login/')) {
                return json(200, { job: { state: 'failed', outcome: { reason: 'auth_not_ready' },
                    message: 'native Codex session is not logged in' } });
            }
            return json(404, {});
        },
    });
    await ctl.start('codex', '');
    t.mock.timers.tick(3000);
    await flush();
    // The bounded re-check sleeps between its attempts; drive them all.
    assert.ok(host.innerHTML.includes('Confirming the sign-in…'), 'the in-between state is shown');
    for (let i = 0; i < 4; i += 1) { t.mock.timers.tick(2500); await flush(); }
    assert.ok(host.innerHTML.includes('Could not confirm the sign-in yet'), host.innerHTML);
    // The engine's own sentence rides beside the fixed verdict text.
    assert.ok(host.innerHTML.includes('native Codex session is not logged in'), host.innerHTML);
    assert.ok(!host.innerHTML.includes('Sign-in failed'), 'an unproven verdict is never a failure');
    ctl.dispose();
    store.dispose();
    t.mock.timers.reset();
});

test('dispose CANCELS the live job before releasing custody, and clears the card', async (t) => {
    // It used to clear ctl.active, release the hold and return — with no DELETE
    // for a live job and the card still on screen. The wizard mounts this
    // controller on a step the owner can cancel mid-login, so an orphaned job
    // kept a sign-in running server-side for a card that no longer existed.
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let jobPolls = 0;
    const calls = [];
    let custodyAtDelete = 'never issued';
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    store.subscribe(() => {}, { visible: () => false });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            calls.push(`${init.method || 'GET'} ${url}`);
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-4', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') {
                custodyAtDelete = ctl.active?.jobId || '';
                return json(200, { job: { state: 'cancelled' } });
            }
            jobPolls += 1;
            return json(200, { job: { state: 'running' } });
        },
    });
    await ctl.start('codex', '');
    assert.equal(store.polling, true);
    assert.ok(host.innerHTML.includes('data-harness-identity="codex"'),
        'the card is on screen before the disposer');

    const released = await ctl.dispose();
    assert.ok(calls.includes('DELETE /api/claudexor/login/job-4'),
        `the live job must be cancelled: ${calls.join(' | ')}`);
    assert.equal(custodyAtDelete, 'job-4', 'custody was still held WHEN the DELETE went out');
    assert.equal(released, LOGIN_CUSTODY_RELEASED, 'a proven cancel releases custody');
    assert.equal(ctl.active, null, '…and only then');
    assert.equal(store.polling, false, 'the login hold was released');
    assert.equal(host.innerHTML, '', 'the disposer cleared the rendered card');
    const before = jobPolls;
    t.mock.timers.tick(60000);
    await flush();
    assert.equal(jobPolls, before, 'no job-poll timer survived the disposer');
    ctl.render();
    assert.equal(host.innerHTML, '', 'a disposed controller renders no card');
    // Idempotent, and it does not re-DELETE.
    const deletes = calls.filter((c) => c.startsWith('DELETE')).length;
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RELEASED);
    assert.equal(calls.filter((c) => c.startsWith('DELETE')).length, deletes);
    store.dispose();
    t.mock.timers.reset();
});

test('a dispose whose cancel is UNPROVEN keeps the job id instead of forgetting it', async (t) => {
    // Same rule Close has always had (C7): a 5xx/network death means the daemon
    // may still be running the login, so custody is retained and the caller is
    // told so — the surface goes away, the honesty does not.
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    store.subscribe(() => {}, { visible: () => false });
    const host = fakeHost();
    let jobPolls = 0;
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-5', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') return json(503, { error: 'daemon unreachable' });
            jobPolls += 1;
            return json(200, { job: { state: 'running' } });
        },
    });
    await ctl.start('codex', '');

    const released = await ctl.dispose();
    assert.equal(released, LOGIN_CUSTODY_UNKNOWN, 'an unproven cancel is reported, not swallowed');
    assert.equal(ctl.active?.jobId, 'job-5', 'the job id is RETAINED — it may still be live');
    assert.equal(host.innerHTML, '', 'the host is cleared either way');
    assert.equal(store.polling, false, 'and nothing stays armed');
    const before = jobPolls;
    t.mock.timers.tick(60000);
    await flush();
    assert.equal(jobPolls, before);
    ctl.detach();
    store.dispose();
    t.mock.timers.reset();
});

test('a Close during the create POST cancels the job that POST installs', async (t) => {
    // The busy guard DROPPED a transition that arrived while another ran, so
    // Close answered false and vanished; the create then installed a live job
    // and no DELETE was ever issued. Transitions queue now: the close is
    // applied to the job it could not see yet.
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const calls = [];
    let custodyAtDelete = 'never issued';
    let releaseCreate = null;
    const createGate = new Promise((resolve) => { releaseCreate = resolve; });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            calls.push(`${init.method || 'GET'} ${url}`);
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                await createGate;
                return json(200, { job_id: 'job-after-close', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') {
                custodyAtDelete = ctl.active?.jobId || '';
                return json(200, { job: { state: 'cancelled' } });
            }
            return json(200, { job: { state: 'running' } });
        },
    });
    const starting = ctl.start('codex', '');
    const closing = ctl.close();          // pressed while the create is in flight
    releaseCreate();
    await starting;
    const closed = await closing;

    assert.equal(closed, LOGIN_CUSTODY_RELEASED, 'the close RAN — it is queued, never dropped');
    assert.ok(calls.includes('DELETE /api/claudexor/login/job-after-close'),
        `the job the create installed must be cancelled: ${calls.join(' | ')}`);
    assert.ok(calls.indexOf('DELETE /api/claudexor/login/job-after-close')
        > calls.indexOf('POST /api/claudexor/login'), 'the cancel follows the create');
    assert.equal(custodyAtDelete, 'job-after-close', 'custody was held WHEN the DELETE went out');
    assert.equal(ctl.active, null, 'and released only after it was proven gone');
    assert.equal(host.innerHTML, '', 'no card left behind');
    ctl.dispose();
    store.dispose();
    t.mock.timers.reset();
});

test('dispose queued during create adopts a returned fence without repeating cancel', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let releaseCreate;
    const gate = new Promise((resolve) => { releaseCreate = resolve; });
    let createStarted = false;
    let deletes = 0;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                createStarted = true;
                await gate;
                return json(200, { job_id: 'job-create-fence', job: {
                    state: 'interrupted_unknown', outcome: { reason: 'termination_unconfirmed' },
                } });
            }
            if (init.method === 'DELETE') { deletes += 1; return json(503, {}); }
            throw new Error(`unexpected ${url}`);
        },
    });
    const starting = ctl.start('codex', '');
    await flush();
    assert.equal(createStarted, true);
    const disposing = ctl.dispose();
    releaseCreate();
    await starting;
    assert.equal(await disposing, LOGIN_CUSTODY_RETAINED);
    assert.equal(deletes, 0);
    assert.equal(ctl.active?.jobId, 'job-create-fence');
    assert.equal(host.innerHTML, '');
    ctl.detach();
    store.dispose();
    t.mock.timers.reset();
});

test('a 2xx create with no job id fails loudly instead of waiting forever', async (t) => {
    // A 200 carrying non-JSON left no job id, no error and a card polling
    // nothing: "Starting the sign-in…" with no verdict and no way out. A job
    // id is the minimum a created job must carry — without it there is nothing
    // to poll and nothing to cancel.
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    store.subscribe(() => {}, { visible: () => false });
    const host = fakeHost();
    let polls = 0;
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return { ok: true, status: 200, json: async () => { throw new Error('not json'); } };
            }
            polls += 1;
            return json(200, { job: { state: 'running' } });
        },
    });
    await ctl.start('codex', '');
    assert.ok(host.innerHTML.includes('data-login-retry'), `the error face offers a retry: ${host.innerHTML}`);
    assert.ok(host.innerHTML.includes('no job id'), host.innerHTML);
    assert.equal(ctl.active.jobId, '', 'no job id was invented');
    assert.equal(store.polling, false, 'a failed create releases the status hold');
    t.mock.timers.tick(60000);
    await flush();
    assert.equal(polls, 0, 'nothing is polled for a job that was never created');
    const closing = ctl.close(ctl.active);
    assert.equal(host.innerHTML, '', 'Close remains usable without inventing a server identity');
    assert.equal(await closing, LOGIN_CUSTODY_UNKNOWN);
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_UNKNOWN);
    store.dispose();
    t.mock.timers.reset();
});

test('malformed 2xx polls count toward the SAME bounded give-up', async (t) => {
    // A 2xx carrying no `job` used to RESET the failure streak, so a stream of
    // them polled forever: twelve malformed answers still left a pending
    // verdict and another armed timer, and the documented ten-failure give-up
    // was unreachable.
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    store.subscribe(() => {}, { visible: () => false });
    const host = fakeHost();
    let polls = 0;
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-6', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') return json(200, { job: { state: 'cancelled' } });
            polls += 1;
            return json(200, { ok: true });     // 2xx, no job — meaningless
        },
    });
    await ctl.start('codex', '');
    for (let i = 0; i < 12; i += 1) { t.mock.timers.tick(30000); await flush(); }

    assert.ok(polls >= JOB_POLL_GIVE_UP_FAILURES, `the chain really polled: ${polls}`);
    assert.ok(polls <= JOB_POLL_GIVE_UP_FAILURES, `and STOPPED at the documented bound: ${polls}`);
    assert.ok(host.innerHTML.includes('Could not confirm the sign-in yet'),
        `an honest unconfirmed verdict, not a forever-pending card: ${host.innerHTML}`);
    assert.equal(store.polling, false, 'the give-up released the status hold');
    await ctl.dispose();
    store.dispose();
    t.mock.timers.reset();
});

test('a 2xx poll carrying an EMPTY job object is a failure, not a healthy pending read', async (t) => {
    // Reachable on the real wire, not hypothetical: the gateway normalizes a
    // non-object engine reply to `{}` (gateways/claudexor.py::setup_job_call),
    // so the proxy answers {job:{}} — an object, so the old guard called it a
    // success, reset the failure streak and armed another timer. Twelve of
    // them left the verdict null and the card pending for good.
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    store.subscribe(() => {}, { visible: () => false });
    const host = fakeHost();
    let polls = 0;
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-7', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') return json(200, { job: { state: 'cancelled' } });
            polls += 1;
            return json(200, { job: {} });          // present, and says nothing
        },
    });
    await ctl.start('codex', '');
    for (let i = 0; i < 12; i += 1) { t.mock.timers.tick(30000); await flush(); }

    assert.equal(polls, JOB_POLL_GIVE_UP_FAILURES,
        `an empty job must count toward the bounded give-up: ${polls} polls`);
    assert.ok(host.innerHTML.includes('Could not confirm the sign-in yet'),
        `and the card settles honestly instead of polling forever: ${host.innerHTML}`);
    await ctl.dispose();
    store.dispose();
    t.mock.timers.reset();
});

test('duplicate starts are COALESCED, so a double-click cannot create-cancel-create', async (t) => {
    // Serializing was not enough: the queue ran the second start AFTER the
    // first, whose C7 guard then cancelled the job the first had installed and
    // created another — so the device link the owner was reading was
    // invalidated as it appeared, and the daemon saw two creates.
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const calls = [];
    let releaseCreate = null;
    const gate = new Promise((resolve) => { releaseCreate = resolve; });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const host = fakeHost();
    let created = 0;
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            calls.push(`${init.method || 'GET'} ${url}`);
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                created += 1;
                await gate;
                return json(200, { job_id: `job-${created}`, job: { state: 'running' } });
            }
            if (init.method === 'DELETE') return json(200, { job: { state: 'cancelled' } });
            return json(200, { job: { state: 'running' } });
        },
    });
    const first = ctl.start('codex', '');
    const second = ctl.start('codex', '');           // the second click
    assert.equal(first, second, 'the same pending start is shared, not queued behind itself');
    releaseCreate();
    await Promise.all([first, second]);

    assert.equal(created, 1, `exactly one job was created: ${calls.join(' | ')}`);
    assert.equal(calls.filter((c) => c.startsWith('DELETE')).length, 0,
        'and nothing was cancelled to make room for a duplicate');
    assert.equal(ctl.active.jobId, 'job-1');

    // A DIFFERENT account is not the same start, and once a start has settled
    // the next one is a real (guarded) restart — coalescing is not caching.
    await ctl.start('codex', 'work');
    assert.equal(created, 2);
    assert.ok(calls.includes('DELETE /api/claudexor/login/job-1'), 'the C7 guard still runs');
    await ctl.dispose();
    store.dispose();
    t.mock.timers.reset();
});

test('the pending-start key includes transport and the external start uses the release guard', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let releaseCreate = null;
    const gate = new Promise((resolve) => { releaseCreate = resolve; });
    const bodies = [];
    let deletes = 0;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const ctl = createLoginCardController({
        host: fakeHost(), store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                bodies.push(JSON.parse(init.body));
                if (bodies.length === 1) await gate;
                return json(200, { job_id: `job-${bodies.length}`, job: { state: 'running' },
                    ...(bodies.at(-1).transport === 'client_pty'
                        ? { attach_command: 'claudexor setup attach job-2', attach_shell: 'posix' } : {}) });
            }
            if (init.method === 'DELETE') {
                deletes += 1;
                return json(200, { job: { state: 'cancelled' } });
            }
            return json(200, { job: { state: 'running' } });
        },
    });
    const ordinary = ctl.start('codex', 'work');
    const external = ctl.start('codex', 'work', 'client_pty');
    assert.notEqual(ordinary, external, 'different transports must not coalesce');
    releaseCreate();
    await Promise.all([ordinary, external]);
    assert.equal(bodies.length, 2);
    assert.equal(bodies[0].transport, undefined);
    assert.equal(bodies[1].transport, 'client_pty');
    assert.equal(deletes, 1, 'the external start passed through the existing custody release guard');
    assert.equal(ctl.active.jobId, 'job-2');
    await ctl.dispose();
    store.dispose();
    t.mock.timers.reset();
});

test('an unknown dispose can retry cancellation against the same retained job id', async (t) => {
    // The verdict is only useful if the caller can act on it. A retained job
    // must stay cancellable — otherwise a host that refuses to remount while
    // custody is held is stuck forever.
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let deleteStatus = 503;
    const deletes = [];
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host,
        store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-8', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') {
                deletes.push(url);
                return deleteStatus === 200
                    ? json(200, { job: { state: 'cancelled' } })
                    : json(deleteStatus, {});
            }
            return json(200, { job: { state: 'running' } });
        },
    });
    await ctl.start('codex', '');

    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_UNKNOWN, 'the daemon refusal leaves custody unknown');
    assert.equal(ctl.active?.jobId, 'job-8');
    assert.equal(deletes.length, 1);

    // The retry re-runs the SAME proven-cancel path — it used to answer a
    // permanent `false` off the idempotence branch and never try again.
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_UNKNOWN, 'still refused, still honest');
    assert.equal(deletes.length, 2, 'and it really re-attempted the cancel');
    deleteStatus = 200;
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RELEASED, 'the daemon answers, custody is released');
    assert.equal(ctl.active, null);
    assert.equal(deletes.length, 3);
    // …and now it is idempotent again: nothing left to cancel.
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RELEASED);
    assert.equal(deletes.length, 3);
    store.dispose();
    t.mock.timers.reset();
});

test('first active Close retains terminal-unconfirmed custody; second recovery Close detaches synchronously', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const calls = [];
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            calls.push(`${init.method || 'GET'} ${url}`);
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-recovery', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') return json(200, { job: {
                state: 'interrupted_unknown', outcome: { reason: 'termination_unconfirmed' },
            } });
            return json(200, { job: { state: 'running' } });
        },
    });
    await ctl.start('codex', '');

    assert.equal(await ctl.close(), LOGIN_CUSTODY_RETAINED);
    assert.equal(calls.filter((call) => call.startsWith('DELETE ')).length, 1);
    assert.equal(ctl.active?.jobId, 'job-recovery');
    assert.match(host.innerHTML, /data-login-reconcile/);

    const before = calls.length;
    const second = ctl.close(ctl.active);
    assert.equal(host.innerHTML, '', 'the recovery-face Close hides synchronously');
    assert.equal(ctl.active, null);
    assert.equal(ctl.disposed, true);
    assert.equal(calls.length, before, 'local detach starts no lifecycle HTTP');
    assert.equal(await second, LOGIN_CUSTODY_RETAINED);
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RETAINED,
        'idempotent cleanup must not relabel local detach as release proof');
    assert.equal(await ctl.close(), LOGIN_CUSTODY_RETAINED);
    await ctl.start('codex', '');
    assert.equal(calls.length, before);
    store.dispose();
    t.mock.timers.reset();
});

test('dispose on an already-visible recovery face returns retained without repeating cancel', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let deletes = 0;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-fence', job: {
                    state: 'interrupted_unknown', outcome: { reason: 'termination_unconfirmed' },
                } });
            }
            if (init.method === 'DELETE') { deletes += 1; return json(503, {}); }
            throw new Error(`unexpected ${url}`);
        },
    });
    await ctl.start('codex', '');
    assert.match(host.innerHTML, /data-login-reconcile/);
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RETAINED);
    assert.equal(deletes, 0, 'repeating cancel cannot reconcile terminal-unconfirmed custody');
    assert.equal(ctl.active?.jobId, 'job-fence');
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RETAINED);
    assert.equal(deletes, 0);
    assert.equal(ctl.detach(), LOGIN_CUSTODY_RETAINED);
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RETAINED);
    assert.equal(deletes, 0);
    store.dispose();
    t.mock.timers.reset();
});

test('explicit reconcile retains on 409, becomes safe on proof, and only a later retry creates', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const calls = [];
    let reconcileRound = 0;
    let creates = 0;
    const retainedJob = {
        state: 'interrupted_unknown', outcome: { reason: 'termination_unconfirmed' },
    };
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            calls.push(`${init.method || 'GET'} ${url}`);
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                creates += 1;
                return json(200, { job_id: `job-${creates}`, job: creates === 1
                    ? retainedJob : { state: 'running' } });
            }
            if (url.endsWith('/reconcile')) {
                reconcileRound += 1;
                return reconcileRound === 1
                    ? json(409, { error: 'process group is still present',
                        code: 'setup_termination_unconfirmed',
                        required_actions: ['retry_setup_reconciliation'] })
                    : json(200, { job: { ...retainedJob,
                        terminationReconciliation: { status: 'empty' } } });
            }
            if (init.method === 'DELETE') return json(200, { job: { state: 'cancelled' } });
            return json(200, { job: { state: 'running' } });
        },
    });
    await ctl.start('codex', '');
    assert.match(host.innerHTML, /data-login-reconcile/, 'a create fence lands directly in recovery');

    const first = await ctl.reconcile(ctl.active);
    assert.equal(first.status, LOGIN_CUSTODY_RETAINED);
    assert.equal(ctl.active?.jobId, 'job-1');
    assert.match(host.innerHTML, /process group is still present/);
    assert.match(host.innerHTML, /data-login-reconcile/);
    assert.equal(creates, 1, 'reconcile never creates');

    const second = await ctl.reconcile(ctl.active);
    assert.equal(second.status, LOGIN_CUSTODY_RELEASED);
    assert.match(host.innerHTML, /data-login-retry/);
    assert.match(host.innerHTML, /no longer blocking/);
    assert.equal(creates, 1, 'successful reconcile still creates nothing');

    await ctl.start('codex', '');
    assert.equal(creates, 2, 'only the later explicit retry creates exactly once');
    ctl.detach();
    store.dispose();
    t.mock.timers.reset();
});

test('an absent reconcile lands the unavailable face, whose next Close actually hides it', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const calls = [];
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            calls.push(`${init.method || 'GET'} ${url}`);
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-gone', job: {
                    state: 'interrupted_unknown', outcome: { reason: 'termination_unconfirmed' },
                } });
            }
            if (url.endsWith('/reconcile')) return json(410, {});
            throw new Error(`unexpected ${url}`);
        },
    });
    await ctl.start('codex', '');
    const result = await ctl.reconcile(ctl.active);
    assert.equal(result.status, LOGIN_CUSTODY_RELEASED);
    assert.match(host.innerHTML, /no longer available/);
    const before = calls.length;
    assert.equal(await ctl.close(ctl.active), LOGIN_CUSTODY_RELEASED);
    assert.equal(host.innerHTML, '');
    assert.equal(calls.length, before, 'closing the informational face starts no request');
    ctl.detach();
    store.dispose();
    t.mock.timers.reset();
});

test('poll 404 settles immediately to unavailable instead of entering failure backoff', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let polls = 0;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-poll-gone', job: { state: 'running' } });
            }
            polls += 1;
            return json(404, {});
        },
    });
    await ctl.start('codex', '');
    t.mock.timers.tick(3000);
    await flush();
    assert.equal(polls, 1);
    assert.match(host.innerHTML, /no longer available/);
    t.mock.timers.tick(60000);
    await flush();
    assert.equal(polls, 1, 'absent is terminal client evidence, not a retryable poll failure');
    assert.equal(await ctl.close(ctl.active), LOGIN_CUSTODY_RELEASED);
    assert.equal(host.innerHTML, '');
    ctl.detach();
    store.dispose();
    t.mock.timers.reset();
});

test('a stale in-flight GET cannot overwrite reconciled-safe state', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let releasePoll;
    const pollGate = new Promise((resolve) => { releasePoll = resolve; });
    let pollStarted = false;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const retainedJob = {
        state: 'interrupted_unknown', outcome: { reason: 'termination_unconfirmed' },
    };
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-stale', job: { state: 'running' } });
            }
            if (url.endsWith('/reconcile')) return json(200, { job: { ...retainedJob,
                terminationReconciliation: { status: 'empty' } } });
            if (init.method === 'DELETE') return json(200, { job: retainedJob });
            pollStarted = true;
            await pollGate;
            return json(200, { job: { state: 'running', phase: 'awaiting_user' } });
        },
    });
    await ctl.start('codex', '');
    t.mock.timers.tick(3000);
    await flush();
    assert.equal(pollStarted, true);
    assert.equal(await ctl.close(), LOGIN_CUSTODY_RETAINED);
    assert.equal((await ctl.reconcile(ctl.active)).status, LOGIN_CUSTODY_RELEASED);
    assert.match(host.innerHTML, /no longer blocking/);
    releasePoll();
    await flush();
    assert.match(host.innerHTML, /no longer blocking/);
    assert.ok(!host.innerHTML.includes('Waiting for the sign-in link'));
    ctl.detach();
    store.dispose();
    t.mock.timers.reset();
});
test('a terminal GET overtaking an unconfirmed DELETE keeps the settled card visible', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let releaseDelete; const deleteGate = new Promise((resolve) => { releaseDelete = resolve; });
    const calls = { create: 0, delete: 0, get: 0 }; const store = createClaudexorStatusStore({ fetchImpl: async () => json(200, statusPayload(true)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} } });
    const host = fakeHost(); const ctl = createLoginCardController({ host, store, fetchImpl: async (url, init = {}) => {
        if (url === '/api/claudexor/login' && init.method === 'POST') {
            calls.create += 1; return json(200, { job_id: 'job-delete-race', job: { state: 'running' } });
        }
            if (init.method === 'DELETE') { calls.delete += 1; await deleteGate;
                return json(503, { error: 'daemon busy' }); }
        calls.get += 1; return json(200, { job: { state: 'succeeded' } });
    } });
    await ctl.start('codex', '');
    const closing = ctl.close(ctl.active); await flush();
    assert.equal(calls.delete, 1, 'the active-card Close owns one DELETE');
    t.mock.timers.tick(3000); await flush();
    assert.match(host.innerHTML, /Signed in\./);
    releaseDelete(); assert.equal(await closing, LOGIN_CUSTODY_RELEASED);
    assert.equal(ctl.active?.verdict?.kind, 'success'); assert.match(host.innerHTML, /Signed in\./,
        'the late DELETE must not erase the settled face');
    assert.deepEqual(calls, { create: 1, delete: 1, get: 1 });
    t.mock.timers.tick(60000); await flush();
    assert.equal(calls.get, 1, 'a settled face never rearms job polling');
    ctl.detach(); store.dispose(); t.mock.timers.reset();
});

test('unknown dispose remains retryable while an already-flying poll continuation stays inert', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let releasePoll;
    const pollGate = new Promise((resolve) => { releasePoll = resolve; });
    let polls = 0;
    let deletes = 0;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'job-dispose', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') { deletes += 1; return json(503, { error: 'down' }); }
            polls += 1;
            await pollGate;
            return json(200, { job: { state: 'running' } });
        },
    });
    await ctl.start('codex', '');
    t.mock.timers.tick(3000);
    await flush();
    assert.equal(polls, 1);
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_UNKNOWN);
    assert.equal(ctl.disposed, true);
    assert.equal(ctl.active?.jobId, 'job-dispose');
    assert.equal(host.innerHTML, '');

    releasePoll();
    await flush();
    t.mock.timers.tick(60000);
    await flush();
    assert.equal(polls, 1, 'disposed state fences repaint and reschedule');
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_UNKNOWN);
    assert.equal(deletes, 2, 'disposed+active unknown cleanup retries the same DELETE');
    assert.equal(ctl.detach(), LOGIN_CUSTODY_UNKNOWN);
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_UNKNOWN,
        'after local detach, unknown must not become false release proof');
    assert.equal(deletes, 2);
    store.dispose();
    t.mock.timers.reset();
});

test('typed reconcile transport classifies proof, retryable conflict, malformed success and absence', async () => {
    const retained = await reconcileLoginJob('j1', async () => json(409, {
        error: 'still present', code: 'setup_termination_unconfirmed',
        required_actions: ['retry_setup_reconciliation'],
    }));
    assert.equal(retained.status, LOGIN_CUSTODY_RETAINED);
    assert.deepEqual(retained.requiredActions, ['retry_setup_reconciliation']);
    const safe = await reconcileLoginJob('j1', async () => json(200, { job: {
        state: 'interrupted_unknown', outcome: { reason: 'termination_unconfirmed' },
        terminationReconciliation: { status: 'empty' },
    } }));
    assert.equal(safe.status, LOGIN_CUSTODY_RELEASED);
    assert.equal((await reconcileLoginJob('j1', async () => json(200, {}))).status,
        LOGIN_CUSTODY_UNKNOWN);
    const malformed = await reconcileLoginJob('j1', async () => json(200, { job: {} }));
    assert.equal(malformed.status, LOGIN_CUSTODY_UNKNOWN);
    assert.equal(malformed.envelope, null, 'malformed success cannot erase the latest valid envelope');
    assert.equal((await reconcileLoginJob('j1', async () => json(200, {
        job: { state: 'cancelling' },
    }))).status, LOGIN_CUSTODY_RETAINED);
    for (const status of [404, 410]) {
        const gone = await reconcileLoginJob('j1', async () => json(status, {}));
        assert.equal(gone.status, LOGIN_CUSTODY_RELEASED);
        assert.equal(gone.absent, true);
    }
});

test('the pre-job external action creates client_pty and immediately exposes a labelled command', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = interactiveHost();
    const bodies = [];
    let deletes = 0;
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                const body = JSON.parse(init.body);
                bodies.push(body);
                if (!body.transport) return json(409, {
                    error: 'terminal helper unavailable',
                    code: 'terminal_transport_unavailable',
                    required_actions: ['use_external_terminal'],
                });
                return json(200, {
                    job_id: 'external-job', job: { state: 'running' },
                    attach_command: 'CLAUDEXOR_CONFIG_DIR=/owned claudexor setup attach external-job',
                    attach_shell: 'posix', disclosure_native: false,
                    setup_login_source: 'per_harness',
                });
            }
            if (init.method === 'DELETE') {
                deletes += 1;
                return json(200, { job: { state: 'cancelled' } });
            }
            return json(200, { job: { state: 'running' } });
        },
    });
    await ctl.start('one', 'work');
    assert.equal(ctl.active.absent, true);
    assert.equal(ctl.active.custodyStatus, LOGIN_CUSTODY_RELEASED);
    assert.ok(host.innerHTML.includes('data-login-external-terminal'));
    host.click('[data-login-external-terminal]');
    await flush();
    assert.equal(bodies.length, 2);
    assert.equal(bodies[1].transport, 'client_pty');
    assert.equal(deletes, 0, 'the proven pre-job absence needs no invented DELETE');
    assert.ok(host.innerHTML.includes('data-login-external-command'), host.innerHTML);
    assert.ok(host.innerHTML.includes('Command for POSIX shell'), host.innerHTML);
    assert.ok(host.innerHTML.includes('claudexor setup attach external-job'), host.innerHTML);
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RELEASED);
    assert.equal(deletes, 1, 'dispose releases the newly created external job');
    store.dispose();
});

test('Retry preserves the exact active client_pty transport in the next create body', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = interactiveHost();
    const bodies = [];
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                bodies.push(JSON.parse(init.body));
                if (bodies.length === 1) return json(400, {
                    error: 'temporary refusal', code: 'profile_temporarily_unavailable',
                });
                return json(200, { job_id: 'retry-external', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') return json(200, { job: { state: 'cancelled' } });
            return json(200, { job: { state: 'running' } });
        },
    });

    await ctl.start('one', 'work', 'client_pty');
    assert.ok(host.innerHTML.includes('data-login-retry'), host.innerHTML);
    host.click('[data-login-retry]');
    await flush();
    assert.equal(bodies.length, 2);
    assert.equal(bodies[0].transport, 'client_pty');
    assert.equal(bodies[1].transport, 'client_pty');
    assert.equal(bodies[1].profile_id, 'work');
    await ctl.dispose();
    store.dispose();
});

test('a durable native-command terminal receipt exposes the external action after polling', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, statusPayload(false)),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(200, { job_id: 'durable-job', job: { state: 'running' } });
            }
            return json(200, { job: {
                state: 'failed',
                outcome: { reason: 'command_failed' },
                nativeCommand: { errorCode: 'terminal_transport_failed' },
            } });
        },
    });
    await ctl.start('one', 'work');
    t.mock.timers.tick(3000);
    await flush();
    assert.ok(host.innerHTML.includes('data-login-external-terminal'), host.innerHTML);
    assert.ok(host.innerHTML.includes('Continue in external terminal'), host.innerHTML);
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RELEASED);
    store.dispose();
    t.mock.timers.reset();
});

// ---------------------------------------------------------------------------
// The "name the account" face: an engine that returns the typed required-
// profile action for a DEFAULT login asks for a named account instead of
// leaving a dead error. The discriminator is the engine's exact typed code +
// required action — never a harness name, family emptiness or prose.
// ---------------------------------------------------------------------------

function nameFaceStatusPayload(rows) {
    return {
        daemon: { state: 'running', engine_version: '3.5.0', runtime: {} },
        config_dir: '/home/agent',
        harnesses: [{ id: 'zephyr' }],
        profiles: { harnessAccounts: rows, profiles: [] },
        quota: [],
    };
}

const ENGINE_SAID = 'harness "zephyr" has no default credential store: '
    + 'sign in from a named account (add one first, then start the login from it)';
const REQUIRED_PROFILE = {
    error: ENGINE_SAID,
    code: 'credential_profile_required',
    required_actions: ['add_named_account'],
};

test('create-400 on an EMPTY family becomes the name-the-account face, and its submit runs the standard NAMED flow', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, nameFaceStatusPayload([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    const host = fakeHost();
    const posts = [];
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                const body = JSON.parse(init.body);
                posts.push(body);
                if (!body.profile_id) return json(400, REQUIRED_PROFILE);
                return json(200, { job_id: 'job-named', job: { state: 'running', phase: 'awaiting_user' } });
            }
            if (init.method === 'DELETE') return json(200, { job: { state: 'cancelled' } });
            return json(200, { job: { state: 'running' } });
        },
    });
    await ctl.start('zephyr', '');
    // The engine's own sentence, verbatim (quotes render HTML-escaped).
    assert.ok(host.innerHTML.includes('has no default credential store: '
        + 'sign in from a named account (add one first, then start the login from it)'),
        `the engine's own sentence is shown: ${host.innerHTML}`);
    assert.equal(ctl.active.needsProfile.message, ENGINE_SAID,
        'the card holds the engine sentence untouched');
    assert.match(host.innerHTML, /data-profile-name-input/);
    assert.match(host.innerHTML, /data-profile-name-submit/);
    assert.ok(!host.innerHTML.includes('data-login-retry'),
        'not the dead-end error face — its Try again would repeat the refused default login');
    assert.ok(!host.innerHTML.includes('data-login-state'),
        'no live progress line: nothing is running');
    assert.equal(store.polling, false, 'a refused create holds no status poll');

    // Same validation as Add account: a name normalization would rewrite is
    // shown back editable, and does NOT start a login.
    ctl.active.profileNameValue = 'Work Laptop';
    ctl.submitProfileName();
    assert.equal(posts.length, 1, 'an unstable name must not be submitted');
    assert.ok(host.innerHTML.includes('will be saved as &quot;work-laptop&quot;')
        || host.innerHTML.includes('will be saved as "work-laptop"'), host.innerHTML);
    assert.equal(ctl.active.profileNameValue, 'work-laptop');

    // The stable name starts the NAMED login through the one standard flow.
    ctl.submitProfileName();
    await flush();
    assert.equal(posts.length, 2);
    assert.equal(posts[1].harness, 'zephyr');
    assert.equal(posts[1].profile_id, 'work-laptop');
    assert.ok(host.innerHTML.includes('(work-laptop)'), `the card now runs the named login: ${host.innerHTML}`);
    assert.ok(!host.innerHTML.includes('data-profile-name-input'), 'the name face was replaced');
    await ctl.dispose();
    store.dispose();
});

test('the name-the-account continuation preserves an explicit client_pty transport', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, nameFaceStatusPayload([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    const host = fakeHost();
    const posts = [];
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                const body = JSON.parse(init.body);
                posts.push(body);
                if (!body.profile_id) return json(400, REQUIRED_PROFILE);
                return json(200, { job_id: 'named-external', job: { state: 'running' } });
            }
            if (init.method === 'DELETE') return json(200, { job: { state: 'cancelled' } });
            return json(200, { job: { state: 'running' } });
        },
    });

    await ctl.start('zephyr', '', 'client_pty');
    ctl.active.profileNameValue = 'work';
    ctl.submitProfileName();
    await flush();
    assert.equal(posts.length, 2);
    assert.equal(posts[0].transport, 'client_pty');
    assert.equal(posts[1].transport, 'client_pty');
    assert.equal(posts[1].profile_id, 'work');
    await ctl.dispose();
    store.dispose();
});

test('an unrelated create-400 stays ordinary even when the family is empty', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, nameFaceStatusPayload([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(400, { error: 'loginFlow is not accepted for this harness' });
            }
            return json(404, {});
        },
    });
    await ctl.start('zephyr', '');
    assert.ok(host.innerHTML.includes('data-login-retry'), `ordinary error face: ${host.innerHTML}`);
    assert.ok(!host.innerHTML.includes('data-profile-name-input'),
        'empty-family heuristics must not turn an unrelated 400 into a name request');
    await ctl.dispose();
    store.dispose();
});

test('the typed required-profile action selects the name face independent of family rows', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, nameFaceStatusPayload([
            { harness_id: 'zephyr', native_login_detected: true },
        ])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(400, REQUIRED_PROFILE);
            }
            return json(404, {});
        },
    });
    await ctl.start('zephyr', '');
    assert.match(host.innerHTML, /data-profile-name-input/);
    await ctl.dispose();
    store.dispose();
});

test('required-profile code without its action and unrelated 409 remain ordinary errors', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, nameFaceStatusPayload([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    const host = fakeHost();
    let response = json(400, {
        error: ENGINE_SAID, code: 'credential_profile_required', required_actions: [],
    });
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => (
            url === '/api/claudexor/login' && init.method === 'POST'
                ? response : json(404, {})),
    });
    await ctl.start('zephyr', '');
    assert.ok(!host.innerHTML.includes('data-profile-name-input'));
    response = json(409, { error: 'different conflict', code: 'credential_profile_ambiguous',
        required_actions: ['disable_extra_profiles'] });
    await ctl.start('zephyr', '');
    assert.ok(!host.innerHTML.includes('data-profile-name-input'));
    assert.equal(ctl.active.absent, true, 'a typed pre-job conflict proves no job exists');
    assert.equal(ctl.active.custodyStatus, LOGIN_CUSTODY_RELEASED);
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_RELEASED);
    store.dispose();
});

test('create 5xx / transport death stays the ordinary error face even for an empty family', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, nameFaceStatusPayload([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    const host = fakeHost();
    let mode = '503';
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                if (mode === '503') return json(503, { error: 'daemon_unreachable: connect refused' });
                throw new Error('network down');
            }
            return json(404, {});
        },
    });
    await ctl.start('zephyr', '');
    assert.ok(host.innerHTML.includes('data-login-retry'), host.innerHTML);
    assert.ok(!host.innerHTML.includes('data-profile-name-input'),
        'a 503 is no verdict about the login shape');
    // Transport death: same rule.
    mode = 'throw';
    await ctl.start('zephyr', '');
    assert.ok(host.innerHTML.includes('data-login-retry'), host.innerHTML);
    assert.ok(!host.innerHTML.includes('data-profile-name-input'));
    assert.equal(ctl.active.absent, false,
        'untyped discovery/transport failure cannot prove whether create ran');
    assert.equal(await ctl.dispose(), LOGIN_CUSTODY_UNKNOWN);
    store.dispose();
});

test('a create-400 NAMED login never asks for a name again (only the default-login shape does)', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, nameFaceStatusPayload([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    const host = fakeHost();
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(400, { error: 'profile id is not acceptable' });
            }
            return json(404, {});
        },
    });
    await ctl.start('zephyr', 'work');
    assert.ok(host.innerHTML.includes('data-login-retry'), host.innerHTML);
    assert.ok(!host.innerHTML.includes('data-profile-name-input'));
    await ctl.dispose();
    store.dispose();
});

test('Close on the name-the-account face detaches as released — the refused create provably made no job', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, nameFaceStatusPayload([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    const host = fakeHost();
    const calls = [];
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async (url, init = {}) => {
            calls.push(`${init.method || 'GET'} ${url}`);
            if (url === '/api/claudexor/login' && init.method === 'POST') {
                return json(400, REQUIRED_PROFILE);
            }
            return json(404, {});
        },
    });
    await ctl.start('zephyr', '');
    assert.match(host.innerHTML, /data-profile-name-input/);
    const closed = await ctl.close(ctl.active);
    assert.equal(closed, LOGIN_CUSTODY_RELEASED, 'no job was ever created, custody is trivially released');
    assert.equal(host.innerHTML, '', 'the card is gone');
    assert.ok(!calls.some((c) => c.startsWith('DELETE')), 'nothing to DELETE');
    store.dispose();
});

test('the verify-race adopts the job\'s OWN resolved profile id, and only a string one', () => {
    // Unified engines resolve a default login onto their bootstrap registry
    // row and the job record names it (plan §K.4) — the row the verify-race
    // must ask about, because no row with the empty id exists there. A legacy
    // engine's default job carries null, which reads as the empty-string
    // address, byte-identical to the old behavior.
    assert.equal(resolvedJobProfileId({ job: { profileId: 'codex-default' } }), 'codex-default');
    assert.equal(resolvedJobProfileId({ job: { profileId: null } }), '');
    assert.equal(resolvedJobProfileId({ job: {} }), '');
    assert.equal(resolvedJobProfileId({}), '');
    assert.equal(resolvedJobProfileId(null), '');
    // Fail-safe against a future non-string spelling: never a coerced name.
    assert.equal(resolvedJobProfileId({ job: { profileId: 42 } }), '');
});

test('preserveCardFocus keeps the caret in the name-the-account input across a re-render', () => {
    let focused = null;
    const nextInput = {
        disabled: false,
        focus() { focused = this; },
        setSelectionRange(a, b) { this.sel = [a, b]; },
    };
    const prior = {
        selectionStart: 2, selectionEnd: 4,
        hasAttribute: (attr) => attr === 'data-profile-name-input',
    };
    const host = {
        contains: (el) => el === prior,
        querySelector: (sel) => (sel === '[data-profile-name-input]' ? nextInput : null),
    };
    preserveCardFocus(host, () => {}, { activeElement: prior });
    assert.equal(focused, nextInput, 'focus lands on the replacement input');
    assert.deepEqual(nextInput.sel, [2, 4], 'the selection survives the swap');
});

test('normalizeProfileName enforces the engine slug contract ^[a-z0-9][a-z0-9_-]{0,63}$', () => {
    // The engine's profile registration refuses anything else; a name the
    // normalization cannot make legal comes back '' so the dialog asks again
    // instead of submitting a doomed create.
    assert.equal(normalizeProfileName('Work Laptop'), 'work-laptop');
    assert.equal(normalizeProfileName('-lead'), 'lead', 'a slug may not start with a separator');
    assert.equal(normalizeProfileName('__x_'), 'x_');
    assert.equal(normalizeProfileName('Работа'), '', 'no ASCII alphanumeric at all cannot become a slug');
    assert.equal(normalizeProfileName('a'.repeat(80)).length, 64, 'capped at the engine 64');
    assert.equal(normalizeProfileName('9start'), '9start', 'a digit is a legal first character');
    const submitted = profileNameSubmission('Работа');
    assert.equal(submitted.profile, '', 'an unslugifiable name never starts a login');
    assert.ok(submitted.note.includes('starts with a lowercase letter or digit'));
});
