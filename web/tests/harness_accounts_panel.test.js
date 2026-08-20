import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
    STATUS_FACETS,
    accountLoginConfirmed,
    createClaudexorStatusStore,
} from '../modules/claudexor_status_store.js';
import {
    READ_FACETS,
    daemonAnswered,
    daemonStatusLine,
    destroyHarnessAccounts,
    initHarnessAccounts,
    refreshActionKind,
    refreshActionLabel,
    serviceBannerLine,
    startLogin,
    unreadFacets,
    wakeDaemon,
} from '../modules/harness_accounts.js';
import { fakeResponse } from './harness_accounts_helpers.js';

// ---------------------------------------------------------------------------
// The custody verdict has a CONSUMER. `dispose()` answers asynchronously
// whether the daemon still runs the login, and the section used to drop that
// answer and rebuild immediately — so the advertised "one live login" held only
// inside one controller instance, and a remount created a second one.
// ---------------------------------------------------------------------------

function fakeElement(id) {
    const el = {
        id,
        innerHTML: '',
        textContent: '',
        dataset: {},
        offsetParent: {},
        listeners: [],
        addEventListener(type, fn) { el.listeners.push([type, fn]); },
        removeEventListener() {},
        querySelector: () => null,
        querySelectorAll: () => [],
        contains: () => false,
        closest: () => null,
    };
    return el;
}

function mountSection() {
    const elements = {};
    // The ids the Agents tab REALLY renders. `harness-accounts-rows` and
    // `harness-daemon-status` were the pre-tab markup; building them here let
    // the refusal below "reach the owner" through a node production no longer
    // has, so an empty, silently unmounted panel passed as a warning shown.
    for (const id of ['agents-service-banner', 'harness-login-card',
        'btn-harness-refresh']) elements[id] = fakeElement(id);
    const doc = {
        hidden: false,
        activeElement: null,
        getElementById: (id) => elements[id] || null,
        addEventListener() {}, removeEventListener() {},
    };
    const win = { addEventListener() {}, removeEventListener() {} };
    return { elements, doc, win };
}

function captureCardControls(element) {
    const controls = new Map();
    element.querySelector = (selector) => {
        const marker = selector.match(/\[([^\]]+)\]/)?.[1] || '';
        if (!marker || !element.innerHTML.includes(marker)) return null;
        const control = {
            listeners: [],
            addEventListener(type, fn) { control.listeners.push([type, fn]); },
        };
        controls.set(selector, control);
        return control;
    };
    return (selector, type = 'click') => controls.get(selector)?.listeners
        .filter(([kind]) => kind === type).map(([, fn]) => fn).at(-1);
}

test('Settings destroy detaches locally, and explicit Connect recreates the disposed controller', async () => {
    const { elements, doc, win } = mountSection();
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    const priorFetch = globalThis.fetch;
    globalThis.document = doc;
    globalThis.window = win;

    const calls = [];
    let creates = 0;
    // The login controller talks through the app's own apiFetch (global fetch);
    // the status store is injected below, so only login traffic lands here.
    globalThis.fetch = async (url, init = {}) => {
        calls.push(`${init.method || 'GET'} ${url}`);
        if (url === '/api/claudexor/login' && init.method === 'POST') {
            creates += 1;
            return creates === 1
                ? fakeResponse(200, { job_id: 'fenced-job', job: { state: 'running' } })
                : fakeResponse(200, { job_id: 'fenced-job', job: {
                    state: 'interrupted_unknown',
                    outcome: { reason: 'termination_unconfirmed' },
                } });
        }
        if (init.method === 'DELETE') return fakeResponse(200, { job: { state: 'cancelled' } });
        return fakeResponse(200, { job: { state: 'running' } });
    };
    const store = createClaudexorStatusStore({
        fetchImpl: async () => fakeResponse(200, {
            daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
            config_dir: '/home/agent',
            harnesses: [{ id: 'codex' }],
            profiles: { harnessAccounts: [{ harness_id: 'codex', native_login_detected: false }], profiles: [] },
            quota: [],
        }),
        doc,
    });

    try {
        assert.equal(await initHarnessAccounts({ store }), true, 'a clean mount succeeds');
        await startLogin('codex', '');
        assert.equal(creates, 1);

        const mountedCard = elements['harness-login-card'].innerHTML;
        const mountedLifecycle = calls.filter((c) => /\/api\/claudexor\/login/.test(c)).length;
        await store.refresh();
        assert.equal(elements['harness-login-card'].innerHTML, mountedCard,
            'ordinary mounted Settings refresh/hide-show does not tear down the card');
        assert.equal(calls.filter((c) => /\/api\/claudexor\/login/.test(c)).length,
            mountedLifecycle, 'ordinary mounted Settings activity starts no lifecycle request');

        const lifecycleBefore = calls.filter((c) => /\/api\/claudexor\/login/.test(c)).length;
        assert.equal(destroyHarnessAccounts(), true);
        assert.equal(calls.filter((c) => /\/api\/claudexor\/login/.test(c)).length, lifecycleBefore,
            'exported destroy initiates zero create/DELETE/reconcile requests');
        assert.equal(elements['harness-login-card'].innerHTML, '');
        assert.doesNotMatch(elements['agents-service-banner'].textContent, /holding off|could not be cancelled/);

        // A detached old handler/exported call cannot recreate outside a mount.
        await startLogin('codex', '');
        assert.equal(creates, 1);

        // Re-init owns no hidden lifecycle mutation. The next explicit Connect
        // creates exactly once and re-adopts the daemon's same fenced job.
        assert.equal(await initHarnessAccounts({ store }), true);
        const beforeConnect = creates;
        await startLogin('codex', '');
        assert.equal(creates, beforeConnect + 1);
        assert.match(elements['harness-login-card'].innerHTML, /data-login-reconcile/,
            'the recreated controller adopted the fence into recovery');
    } finally {
        destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
        globalThis.fetch = priorFetch;
    }
});

test('Settings explicit Connect recreates a cached controller disposed by recovery-face Close', async () => {
    const { elements, doc, win } = mountSection();
    const control = captureCardControls(elements['harness-login-card']);
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    const priorFetch = globalThis.fetch;
    globalThis.document = doc;
    globalThis.window = win;
    let creates = 0;
    let deletes = 0;
    const retainedJob = { state: 'interrupted_unknown',
        outcome: { reason: 'termination_unconfirmed' } };
    globalThis.fetch = async (url, init = {}) => {
        if (url === '/api/claudexor/login' && init.method === 'POST') {
            creates += 1;
            return fakeResponse(200, { job_id: 'same-fenced-job',
                job: creates === 1 ? { state: 'running' } : retainedJob });
        }
        if (init.method === 'DELETE') {
            deletes += 1;
            return fakeResponse(200, { job: retainedJob });
        }
        return fakeResponse(200, { job: { state: 'running' } });
    };
    const store = createClaudexorStatusStore({
        fetchImpl: async () => fakeResponse(200, {
            daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
            config_dir: '/home/agent', harnesses: [{ id: 'codex' }],
            profiles: { harnessAccounts: [], profiles: [] }, quota: [],
        }),
        doc,
    });
    try {
        await initHarnessAccounts({ store });
        await startLogin('codex', '');
        assert.equal(creates, 1);

        await control('[data-login-dismiss]')();
        assert.equal(deletes, 1);
        assert.match(elements['harness-login-card'].innerHTML, /data-login-reconcile/);

        const secondClose = control('[data-login-dismiss]')();
        assert.equal(elements['harness-login-card'].innerHTML, '', 'second Close detaches synchronously');
        await secondClose;
        assert.equal(deletes, 1, 'recovery-face Close does not repeat cancel');

        await startLogin('codex', '');
        assert.equal(creates, 2, 'explicit Connect built a fresh controller and created once');
        assert.match(elements['harness-login-card'].innerHTML, /data-login-reconcile/);
    } finally {
        destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
        globalThis.fetch = priorFetch;
    }
});

// ---------------------------------------------------------------------------
// Provenance invariants carried over from the pre-store panel (9 review
// rounds), re-bound to the store-oriented structure: daemonAnswered /
// unreadFacets are pure helpers OVER the store's one facet reader, the wake is
// a STORE method (single writer), and the wake error's lifecycle lives in the
// panel. The serialization pins got simpler on purpose — one store means the
// generation bookkeeping the old panel needed is now structural.
// ---------------------------------------------------------------------------

test('READ_FACETS mirrors the store facet list, so the parity literal cannot drift', () => {
    // The store's STATUS_FACETS is the literal the backend parity test greps
    // (tests/test_gateway_parity.py ↔ ClaudexorStatusReads); READ_FACETS is
    // this module's restatement, welded here so it inherits that transitively;
    // the store's STATUS_FACETS is the one runtime reader. This pin welds the
    // two spellings together.
    assert.deepEqual(READ_FACETS, ['catalog', 'accounts', 'quota']);
    assert.deepEqual(READ_FACETS, [...STATUS_FACETS]);
});

test('a daemon that answered some reads is not called dead by the aggregate', () => {
    // The partial refusal this whole surface exists for: quota times out while
    // the catalog and the account store both land, and the backend still
    // reports the aggregate as `unreachable`. A predicate written on that
    // aggregate kept a failed wake's error standing above accounts that were
    // genuinely read, and made Refresh promise to start a daemon already
    // answering — the coarse-signal mistake, committed inside the fix for it.
    const partial = { daemon: { state: 'unreachable', engine_version: '3.3.13', runtime: {} },
                      reads: { catalog: 'ok', accounts: 'ok', quota: 'failed' } };
    assert.equal(daemonAnswered(partial), true, 'two landed reads read as no answer at all');
    assert.deepEqual(unreadFacets(partial), ['quota']);
    // The ACCOUNTS facet is not privileged: a catalog that landed alone still
    // proves the daemon answered.
    assert.equal(daemonAnswered({ daemon: { state: 'unreachable' },
        reads: { catalog: 'ok', accounts: 'failed', quota: 'failed' } }), true);
    assert.equal(daemonAnswered({ daemon: { state: 'unreachable' },
        reads: { catalog: 'failed', accounts: 'failed', quota: 'ok' } }), true);
    // A read block that is not a facet MAP answers for nothing — an array is
    // `typeof 'object'`, and a SECOND parse of the block here once disagreed
    // with the store's reader about exactly this. The helpers go through
    // `facetReadState`, so the two can no longer split.
    assert.equal(daemonAnswered({ daemon: { state: 'unreachable' }, reads: ['ok'] }), false);
    assert.equal(refreshActionKind({ daemon: { state: 'unreachable' }, reads: ['ok'] }), 'wake');
    assert.equal(refreshActionKind(partial), 'refresh',
        'the button offered to start a daemon that is answering');

    // Nothing read at all — a pre-fan-out discovery/handshake failure — still
    // means the daemon never spoke, and there the button must start it.
    const silent = { daemon: { state: 'unreachable' },
                     reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } };
    assert.equal(daemonAnswered(silent), false);
    assert.equal(refreshActionKind(silent), 'wake');
    // The aggregate is never the NEGATIVE answer, but with no facet evidence
    // either (a legacy unreachable payload) nothing is proven answered.
    assert.equal(daemonAnswered({ daemon: { state: 'unreachable' } }), false);
    assert.equal(daemonAnswered({ daemon: { state: 'running' } }), true, 'a literal running is positive evidence');
});

test('the refresh button says exactly what pressing it does', () => {
    // Label and handler were written apart once and drifted: on an
    // `unreachable` daemon the label promised a plain re-read while the click
    // called wake. One predicate now feeds both, so the button cannot promise
    // less than it does.
    assert.equal(refreshActionKind({ daemon: { state: 'running' } }), 'refresh');
    assert.equal(refreshActionLabel({ daemon: { state: 'running' } }), 'Refresh');
    for (const state of ['unreachable', 'stale', 'not_provisioned', 'foreign_daemon', '']) {
        assert.equal(refreshActionKind({ daemon: { state } }), 'wake', `state ${state}`);
        assert.match(refreshActionLabel({ daemon: { state } }), /starts the agent daemon/i,
            `the label hides the start on state ${state}`);
    }
});

test('a staged update does not claim the engine is serving when nothing answered', () => {
    // The runtime branches used to return ABOVE the facet logic, and
    // update_staged did worse than hide the gaps: "Engine X keeps running
    // until then" is a positive claim about a daemon that, in this window,
    // answered nothing — printed over a button offering to START it.
    const silentStaged = { daemon: { state: 'unreachable', engine_version: '3.3.13',
                                     runtime: { state: 'update_staged', staged_version: '3.3.14' } },
                           reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } };
    const line = daemonStatusLine(silentStaged, {});
    assert.ok(!/keeps running/.test(line.text), 'claimed the engine is serving on a reading that saw nothing');
    assert.match(line.text, /were not read/);
    assert.match(line.text, /staged/, 'the staged update is still disclosed');

    // Everything read: the staged-update line is earned and keeps its wording.
    const servingStaged = { daemon: { state: 'running', engine_version: '3.3.13',
                                      runtime: { state: 'update_staged', staged_version: '3.3.14' } },
                            reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' } };
    assert.match(daemonStatusLine(servingStaged, {}).text, /keeps running/);

    // installing / error keep their own wording but stop hiding the facets —
    // named in the store's own subjects, never by raw facet id.
    const installing = { daemon: { state: 'unreachable', runtime: { state: 'installing', target_version: '3.3.14' } },
                         reads: { catalog: 'ok', accounts: 'failed', quota: 'failed' } };
    assert.match(daemonStatusLine(installing, {}).text,
        /[Aa]gent accounts and subscription limits were not read/);
    const broken = { daemon: { state: 'unreachable', runtime: { state: 'error', last_error: 'exit 1' } },
                     reads: { catalog: 'ok', accounts: 'failed', quota: 'ok' } };
    assert.match(daemonStatusLine(broken, {}).text, /[Aa]gent accounts were not read/);
});

test('a facet that failed without the aggregate hearing it is still reported', () => {
    // An envelope in the wrong shape is a FAILED read, not an exception, so the
    // daemon goes on reporting `running`. A status line that trusted the
    // literal printed green "Claudexor ready" directly above a row saying the
    // accounts were not checked: one screen, two contradictory claims, the
    // reassuring one on top.
    const drifted = { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
                      config_dir: '/x', reads: { catalog: 'failed', accounts: 'failed', quota: 'ok' } };
    const line = daemonStatusLine(drifted, {});
    assert.equal(line.tone, 'warn', 'a running daemon with two dead reads was called ready');
    assert.ok(!/Claudexor ready/.test(line.text));
    assert.match(line.text, /agents and agent accounts were not read/);

    // Everything read: the green claim is earned and must still be made.
    const whole = { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
                    config_dir: '/x', reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' } };
    assert.equal(daemonStatusLine(whole, {}).tone, 'ok');
    assert.match(daemonStatusLine(whole, {}).text, /Claudexor ready/);

    // A legacy payload with no read block at all keeps the old meaning.
    const legacy = { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} }, config_dir: '/x' };
    assert.equal(daemonStatusLine(legacy, {}).tone, 'ok');

    // Division of labor with the banner: a gap that is merely `not_read`
    // (nobody asked) keeps the daemon's own ready line — the daemon IS proven
    // up — and the banner's muted note explains the unasked reads (pinned in
    // the serviceBannerLine tests above). Only a REAL refusal demotes the line.
    const neverAsked = { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
                         reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } };
    assert.equal(daemonStatusLine(neverAsked, {}).tone, 'ok');
});

test('a partial refusal is not announced as a dead daemon', () => {
    // The aggregate reports `unreachable` whenever ANY read refused, so the
    // status line printed red "Daemon unreachable" directly above the account
    // rows the same poll had just delivered — the panel contradicting itself,
    // with the false half on top.
    const partial = { daemon: { state: 'unreachable', last_error: 'daemon_unreachable: quota refused', runtime: {} },
                      reads: { catalog: 'ok', accounts: 'ok', quota: 'failed' } };
    const line = daemonStatusLine(partial, {});
    assert.equal(line.tone, 'warn', 'a daemon that answered two reads was called dead');
    assert.ok(!/Daemon unreachable/.test(line.text));
    assert.match(line.text, /[Ss]ubscription limits were not read/,
        'the line must name WHICH facet is missing, not claim everything visible was read');
    assert.match(line.text, /quota refused/, 'the reason the owner needs was dropped');

    // Total silence keeps the hard verdict.
    const silent = { daemon: { state: 'unreachable', last_error: 'handshake failed', runtime: {} },
                     reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } };
    assert.equal(daemonStatusLine(silent, {}).tone, 'error');
    assert.match(daemonStatusLine(silent, {}).text, /Daemon unreachable/);
});

test('a payload with no unread facets never renders an empty list', () => {
    // The contract declares `daemon` optional, and a payload carrying only a
    // reads block used to take the partial-refusal branch with an EMPTY
    // complement — a sentence that starts with a space and names nothing.
    const noDaemon = { reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' } };
    const line = daemonStatusLine(noDaemon, {});
    assert.ok(!/^\s/.test(line.text), 'the line begins with an empty facet list');
    assert.ok(!/were not read/.test(line.text));
});

const WAKE_STILL_DOWN = {
    daemon: { state: 'stale', engine_version: '3.3.13', runtime: {} },
    config_dir: '/home/agent', harnesses: [], profiles: {}, quota: [],
    reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' },
};
const WAKE_UP = {
    daemon: { state: 'running', engine_version: '3.3.14', runtime: {} },
    config_dir: '/home/agent', harnesses: [], profiles: {}, quota: [],
    reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' },
};

test('the wake is a store method: single-flighted and serialized against the poll in both orders', async () => {
    // Two writers, two orders — now both inside the ONE store, which is what
    // makes the old generation bookkeeping unnecessary. A refresh started
    // during a wake JOINS it (no second GET runs); a wake pressed during a
    // read waits that read out before POSTing, so the wake's daemon-side read
    // causally follows the poll's commit and can never resurrect an older
    // snapshot.
    const events = [];
    let wakeGate = null;
    let readGate = null;
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => {
            if ((init.method || 'GET') === 'POST') {
                events.push('wake-posted');
                if (wakeGate) await wakeGate;
                return fakeResponse(200, WAKE_UP);
            }
            events.push('read');
            if (readGate) await readGate;
            return fakeResponse(200, WAKE_STILL_DOWN);
        },
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    try {
        // Order 1: wake first — a second wake and a refresh both JOIN it.
        let releaseWake;
        wakeGate = new Promise((resolve) => { releaseWake = resolve; });
        const first = store.wake();
        const second = store.wake();
        const joined = store.refresh();
        assert.equal(events.filter((e) => e === 'wake-posted').length, 1, 'the wake is single-flighted');
        assert.equal(events.filter((e) => e === 'read').length, 0,
            'a second reader was started while the wake was in flight');
        releaseWake();
        const [a, b] = await Promise.all([first, second]);
        await joined;
        assert.equal(a.ok, true);
        assert.equal(b.ok, true);
        assert.equal(store.snapshot.daemon.engine_version, '3.3.14', 'the wake reading was committed');
        assert.equal(store.error, '', 'a committed wake reading retires staleness like every read');

        // Order 2: poll first — the wake waits it out before POSTing, and the
        // causally-later wake reading wins.
        events.length = 0;
        wakeGate = null;
        let releaseRead;
        readGate = new Promise((resolve) => { releaseRead = resolve; });
        const slow = store.refresh();
        const late = store.wake();
        assert.deepEqual(events, ['read'], 'the wake POSTed before the in-flight poll finished');
        releaseRead();
        await Promise.all([slow, late]);
        assert.deepEqual(events, ['read', 'wake-posted']);
        assert.equal(store.snapshot.daemon.engine_version, '3.3.14',
            'the wake reading (causally later) did not win');
    } finally {
        store.dispose();
    }
});

test('a wake error reaches the banner and expires only when the daemon is proven up', async () => {
    // Both edges of the same lie. The error must not outlive the daemon coming
    // up on its own (login, delegated run) — but it must also not be wiped by a
    // 200 that REPORTS the daemon still down: the reason the owner asked for
    // then vanishes within one 5s tick, before it can be read. The POST is the
    // store's; the LIFECYCLE (shown, kept, expired on `daemonAnswered`) is the
    // panel's and is what this exercises end to end.
    const { elements, doc, win } = mountSection();
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    globalThis.document = doc;
    globalThis.window = win;
    let statusBody = WAKE_STILL_DOWN;
    let wakeStatus = 503;
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => ((init.method || 'GET') === 'POST'
            ? fakeResponse(wakeStatus, { error: 'claudexord_not_installed: no binary' })
            : fakeResponse(200, statusBody)),
        doc,
    });
    const banner = () => elements['agents-service-banner'].textContent;
    try {
        assert.equal(await initHarnessAccounts({ store }), true);
        await store.refresh();

        // The refusal reaches the owner, verbatim.
        await wakeDaemon();
        assert.match(banner(), /Could not start the agent daemon/i);
        assert.match(banner(), /claudexord_not_installed/);

        // A 200 that says the daemon is STILL down must not erase the reason.
        await store.refresh();
        assert.match(banner(), /Could not start the agent daemon/i,
            'a still-down reading erased the reason the wake failed');

        // The daemon comes up on its own: the error has stopped mattering.
        statusBody = WAKE_UP;
        await store.refresh();
        assert.ok(!/Could not start the agent daemon/i.test(banner()),
            'the stale wake error outlived the daemon that came up anyway');
        assert.match(banner(), /3\.3\.14/);

        // …and a refusal that lands while the daemon is ALREADY answering is
        // moot and never shown (the failure is real; it stopped mattering).
        await wakeDaemon();
        assert.ok(!/Could not start the agent daemon/i.test(banner()),
            'a moot refusal was printed over a daemon that is answering');
    } finally {
        await destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
    }
});

test('the refresh button routes the click through the same predicate as its label', async () => {
    // Pinning the predicate alone does not stop the handler from ignoring it —
    // replacing its body with an unconditional wake would leave every pure
    // test green. Drive the real click listener both ways.
    const { elements, doc, win } = mountSection();
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    globalThis.document = doc;
    globalThis.window = win;
    const requests = [];
    let statusBody = WAKE_STILL_DOWN;
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => {
            requests.push(`${init.method || 'GET'} ${url}`);
            if ((init.method || 'GET') === 'POST') return fakeResponse(200, WAKE_UP);
            return fakeResponse(200, statusBody);
        },
        doc,
    });
    try {
        assert.equal(await initHarnessAccounts({ store }), true);
        await store.refresh();
        const click = elements['btn-harness-refresh'].listeners
            .filter(([type]) => type === 'click').map(([, fn]) => fn).at(-1);
        assert.ok(click, 'the refresh button lost its listener');

        // Daemon asleep -> the press must START it (and the label says so).
        assert.match(elements['btn-harness-refresh'].textContent, /starts the agent daemon/i);
        requests.length = 0;
        await click();
        assert.ok(requests.some((r) => r.startsWith('POST /api/claudexor/wake')),
            'a sleeping daemon was only re-read, so the button could not help');

        // Daemon live (the wake committed an answering reading) -> the press
        // stays a plain re-read, and the label agrees.
        statusBody = WAKE_UP;
        requests.length = 0;
        await click();
        assert.ok(requests.every((r) => !r.includes('/api/claudexor/wake')),
            'a live daemon was provisioned by a button that says Refresh');
        assert.ok(requests.some((r) => r.startsWith('GET ')), 'the live press did not re-read');
        assert.equal(elements['btn-harness-refresh'].textContent, 'Refresh');
    } finally {
        await destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
    }
});

test('a login seen online is not un-seen by the re-check running out', () => {
    // Re-bound from the pre-store panel (which pinned its own settleVerdict) to
    // the shared controller the PR extracted: the bounded re-check can close a
    // tick before the account row lands, while a reading that arrived meanwhile
    // (the held status poll, an owner wake) already shows the login. Seeing the
    // login is MONOTONE evidence — the newest committed snapshot is judged with
    // the same predicate before "unconfirmed" may be said. Structural, because
    // settleVerdict is controller-private.
    const src = readFileSync(new URL('../modules/harness_login_cards.js', import.meta.url), 'utf8');
    const body = String(src.split('async function settleVerdict(')[1] || '');
    assert.match(body, /const confirmed = check\.confirmed\s*\n?\s*\|\|\s*accountLoginConfirmed\(store\?\.snapshot/,
        'a positive confirmation no longer wins on its own');
    assert.match(body, /active\.verdict = confirmed/,
        'the monotone predicate does not decide the verdict');
});
