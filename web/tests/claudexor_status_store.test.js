// The ONE shared reader over /api/claudexor/status (phase 2 seam).
//
// Three surfaces used to fetch this endpoint independently, and all three
// mapped "we could not ask" onto the same empty answer a stopped daemon gives
// — which is how a saved reviewer row came to be labeled "(not in discovery)"
// with no explanation (owner report, 2026-08-08). These tests pin the two
// properties that fix stays honest by: one request per moment in time, and
// three states nobody can collapse back together.

import assert from 'node:assert/strict';
import test from 'node:test';

import { readFileSync } from 'node:fs';

import {
    DAEMON_STATES_STOPPED,
    FACET_ACCOUNTS,
    FACET_CATALOG,
    FACET_QUOTA,
    READ_FAILED,
    READ_INDETERMINATE,
    READ_NOT_READ,
    READ_OK,
    READ_TRANSPORT,
    READ_UNREAD,
    accountLoginConfirmed,
    accountRows,
    bindStatusSurface,
    claudexorPreparationLine,
    createClaudexorStatusStore,
    facetGapClause,
    facetKnown,
    facetReadState,
    nextUpAccount,
    readsFor,
    statusPayloadValid,
    statusUnavailableNote,
} from '../modules/claudexor_status_store.js';

// A payload shaped like the producer's own: `_status_payload` sets
// daemon/harnesses/profiles/quota unconditionally, before it reaches the
// daemon at all, and the store now requires exactly those four.
const payloadOf = (daemon, extra = {}) => ({
    daemon, config_dir: '/home/agent', harnesses: [], profiles: {}, quota: [], ...extra,
});

const RUNNING = payloadOf({ state: 'running', engine_version: '3.3.13', runtime: {} },
    { harnesses: [{ id: 'codex' }] });
const STOPPED = payloadOf({ state: 'stale', runtime: { state: 'ready', version: '3.3.13' } });

function fakeDoc({ hidden = false } = {}) {
    const listeners = {};
    return {
        hidden,
        addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
        removeEventListener(type, fn) {
            listeners[type] = (listeners[type] || []).filter((f) => f !== fn);
        },
        fire(type) { for (const fn of [...(listeners[type] || [])]) fn(); },
        count(type) { return (listeners[type] || []).length; },
    };
}

const okResponse = (body) => ({ ok: true, status: 200, json: async () => body });

// Let an in-flight async read settle while the timers are mocked (the read
// chain is a handful of microtasks, and only a SETTLED read re-arms the poll).
const flush = async () => { for (let i = 0; i < 20; i += 1) await Promise.resolve(); };

// ---------------------------------------------------------------------------
// The three states — the whole point of the seam.
// ---------------------------------------------------------------------------

test('a stopped daemon, an unreachable endpoint and a real answer are three DIFFERENT states', () => {
    // The backend serves `harnesses: []` whenever the daemon is not running,
    // and a dead fetch leaves the same emptiness — so "no harnesses" alone can
    // never tell a consumer which world it is in. The derivation must.
    assert.equal(facetReadState(RUNNING, FACET_CATALOG), READ_OK);
    assert.equal(facetReadState(STOPPED, FACET_CATALOG), READ_NOT_READ);
    // Nothing at all to derive from is NOT "the daemon is stopped", and it is
    // not a verdict on this facet either: the honest answer is that the answer
    // did not complete and it does not say which read landed.
    assert.equal(facetReadState(null, FACET_CATALOG), READ_INDETERMINATE);
    assert.equal(facetReadState(RUNNING, FACET_CATALOG, { transportError: 'HTTP 503' }), READ_TRANSPORT);
    // A transport failure over a STALE snapshot is still "we could not ask":
    // the payload in hand describes a past read, never the current one.
    assert.equal(facetReadState(RUNNING, FACET_ACCOUNTS, { transportError: 'boom' }), READ_TRANSPORT);

    // Only a read that actually happened licenses a row-level
    // "(not in discovery)" claim.
    assert.equal(facetKnown(READ_OK), true);
    assert.equal(facetKnown(READ_NOT_READ), false);
    assert.equal(facetKnown(READ_FAILED), false);
    assert.equal(facetKnown(READ_TRANSPORT), false);
    assert.equal(facetKnown(READ_UNREAD), false);
    assert.equal(facetKnown(READ_INDETERMINATE), false);
});

test('facets are INDEPENDENT: one refused read never downgrades its siblings', () => {
    // TODAY'S wire: the producer stamps per-facet reads unconditionally (see
    // the golden test below). One read can fail while the others land, and a
    // consumer that collapses them mislabels every row the survivors describe.
    const partial = {
        daemon: { state: 'unreachable', last_error: 'quota read died' },
        harnesses: [{ id: 'codex' }], profiles: {}, quota: [],
        reads: { catalog: READ_OK, accounts: READ_OK, quota: READ_FAILED },
    };
    assert.deepEqual(readsFor(partial),
        { catalog: READ_OK, accounts: READ_OK, quota: READ_FAILED });
    // …even though daemon.state says unreachable, which the LEGACY derivation
    // would have read as "nothing was read".
    assert.equal(facetReadState(partial, FACET_CATALOG), READ_OK);
    assert.equal(facetReadState(partial, FACET_QUOTA), READ_FAILED);
    // A transport failure still outranks every stamp: the whole answer is old.
    assert.deepEqual(readsFor(partial, { transportError: 'net' }),
        { catalog: READ_TRANSPORT, accounts: READ_TRANSPORT, quota: READ_TRANSPORT });
});

test('a payload WITHOUT the reads block is read legacy-style, per facet', () => {
    // The backend field is landing separately; until it does, the daemon state
    // carries the same fact at coarser resolution.
    assert.deepEqual(readsFor(RUNNING), { catalog: READ_OK, accounts: READ_OK, quota: READ_OK });
    assert.deepEqual(readsFor(STOPPED),
        { catalog: READ_NOT_READ, accounts: READ_NOT_READ, quota: READ_NOT_READ });
});

test('a PRESENT but invalid reads stamp fails closed — it never falls back to the global verdict', () => {
    // (Future wire again — the stamp is not emitted yet.) A 2xx is a transport
    // fact, not a semantic one. The stamp existed and was junk, and the store
    // answered with the daemon's global "running" — so an unparseable
    // provenance block silently became three authoritative facets, the exact
    // collapse this seam exists to end, in the other direction.
    const running = payloadOf({ state: 'running' });
    assert.equal(facetReadState({ ...running, reads: { catalog: 'weird' } }, FACET_CATALOG), READ_FAILED);
    // A PARTIAL stamp answers only for what it stamps; the unstamped siblings
    // are gaps, not inherited successes.
    const partial = { ...running, reads: { catalog: READ_OK } };
    assert.deepEqual(readsFor(partial),
        { catalog: READ_OK, accounts: READ_FAILED, quota: READ_FAILED });
    // …and a `reads` that is not even a map fails closed for every facet.
    for (const bad of [null, 'ok', 42, ['ok']]) {
        assert.deepEqual(readsFor({ ...running, reads: bad }),
            { catalog: READ_FAILED, accounts: READ_FAILED, quota: READ_FAILED },
            `reads: ${JSON.stringify(bad)} must not license a discovery claim`);
    }
});

test('unreachable is NOT stopped: only a genuinely stopped daemon gets the stopped sentence', () => {
    for (const state of DAEMON_STATES_STOPPED) {
        assert.deepEqual(readsFor(payloadOf({ state })),
            { catalog: READ_NOT_READ, accounts: READ_NOT_READ, quota: READ_NOT_READ },
            `${state} genuinely means nobody was asked`);
        // The CALM sentence, and it stops at what this state establishes:
        // nobody asked. WHY nobody asked — a stopped daemon, a runtime awaiting
        // repair, a foreign daemon on the stale port, an ownership problem — is
        // the tab banner's job, because only the daemon state can tell those
        // apart and `foreign_daemon` lands here with a daemon that IS running.
        assert.match(statusUnavailableNote(READ_NOT_READ).text, /daemon was not asked/);
        assert.doesNotMatch(statusUnavailableNote(READ_NOT_READ).text, /could not be read/);
    }
    // …and the WORDING per state, which is what the owner actually reads.
    const wording = (state) => statusUnavailableNote(
        facetReadState(payloadOf({ state }), FACET_ACCOUNTS), { facet: FACET_ACCOUNTS }).text;
    // A stopped daemon is the calm branch — and it still names no diagnosis:
    // "was not asked" is all this read state establishes.
    assert.match(wording('stale'), /daemon was not asked/);
    assert.doesNotMatch(wording('stale'), /not running/);
    // A global refusal is the coarse state, not the calm one: a read that DID
    // happen and did not land must never be dressed as a read nobody
    // attempted, and it may not claim the daemon is down either.
    assert.match(wording('unreachable'), /did not finish answering/);
    assert.doesNotMatch(wording('unreachable'), /was not asked/);
    assert.doesNotMatch(wording('unreachable'), /not running/);
    assert.doesNotMatch(wording('error'), /was not asked/);
    assert.doesNotMatch(wording('error'), /not running/);
    assert.doesNotMatch(wording(''), /was not asked/);
    assert.doesNotMatch(wording(''), /not running/);
    assert.equal(statusUnavailableNote(facetReadState(payloadOf({ state: 'running' }), FACET_ACCOUNTS)), null);
});

test('a GLOBAL refusal is one coarse indeterminate — never three per-facet verdicts', () => {
    // The round-one fix traded one lie for another. It stopped calling a
    // running daemon "stopped", but it turned the endpoint's ONE global
    // `unreachable` into three `failed` facets carrying the same error — so
    // the accounts panel accused the ACCOUNT read using the QUOTA probe's
    // message, over accounts that were sitting right there in the payload
    // (reviewer probe against the live producer, reproduced in the golden
    // test below). A verdict about a facet may only come from a stamp that
    // names that facet.
    for (const state of ['unreachable', 'error', 'unknown', '']) {
        assert.deepEqual(readsFor(payloadOf({ state })),
            { catalog: READ_INDETERMINATE, accounts: READ_INDETERMINATE, quota: READ_INDETERMINATE },
            `${state || '(no state)'} says only that the answer did not complete`);
    }
    // ONE neutral global sentence: no facet subject, and the SAME text for
    // every facet a surface might ask about — three surfaces, one truth.
    const notes = [FACET_CATALOG, FACET_ACCOUNTS, FACET_QUOTA]
        .map((facet) => statusUnavailableNote(READ_INDETERMINATE, { facet, error: 'quota_probe_failed' }));
    assert.equal(new Set(notes.map((n) => n.text)).size, 1, 'the sentence is not per facet');
    for (const note of notes) {
        assert.match(note.text, /did not finish answering/);
        assert.match(note.text, /which parts is not known/);
        assert.doesNotMatch(note.text, /agent accounts|subscription limits|your agents/i,
            'no facet may be named in a verdict the payload does not carry');
        assert.doesNotMatch(note.text, /not running/);
        assert.equal(note.tone, 'warn');
    }
    // The daemon's own global error is still the only explanation there is, so
    // it rides the GLOBAL sentence.
    assert.match(notes[0].text, /quota_probe_failed/);
    // …and no facet is named as a gap either — there is nothing to attribute.
    assert.equal(facetGapClause(
        { catalog: READ_INDETERMINATE, accounts: READ_INDETERMINATE, quota: READ_INDETERMINATE },
        [FACET_CATALOG, FACET_ACCOUNTS, FACET_QUOTA]), '');
    // Nothing is claimed as discovered, either — the whole point of the state.
    assert.equal(facetKnown(READ_INDETERMINATE), false);
});

test("GOLDEN: today's producer payload, with two reads that SUCCEEDED and one that refused", () => {
    // Produced by the REAL producer (`gateway/claudexor_accounts.py::
    // _status_payload`) over faked engine HTTP: the catalog and a unified
    // roster land, and only the quota read is refused with a typed problem.
    // tests/test_claudexor_status_passive.py holds this file to that producer
    // and says how to regenerate it; the point of a golden is that this file,
    // not a hand-written double, is what the client is judged against.
    const golden = JSON.parse(readFileSync(
        new URL('./fixtures/status_quota_refused.json', import.meta.url), 'utf8'));

    // Today's wire: a full per-facet stamp beside the survivors' data. The
    // refusal is the quota facet's own typed error; the daemon that answered
    // the other two reads stays `running`.
    assert.deepEqual(golden.reads,
        { catalog: 'ok', accounts: 'ok', quota: 'failed' },
        'the producer stamps every facet');
    assert.equal(golden.daemon.state, 'running');
    assert.deepEqual([golden.facets.catalog.error, golden.facets.accounts.error, golden.facets.quota.error],
        [null, null, 'quota_probe_failed']);
    assert.equal(golden.harnesses.length, 1, 'the CATALOG read succeeded and is in the payload');
    assert.equal(golden.profiles.profiles.length, 1, 'the ACCOUNTS read succeeded too');
    assert.equal(golden.unified_accounts, true, 'as the unified account model');
    assert.deepEqual(accountRows(golden).map((row) => [row.harness, row.profile_id]), [['codex', 'koshak']]);
    assert.deepEqual(golden.quota, [], 'only the quota read refused');
    assert.equal(statusPayloadValid(golden), true, 'and it is a valid status answer');

    // The stamp — and only the stamp — answers per facet: the two reads that
    // worked stay authoritative, the one that refused is the only accusation.
    assert.deepEqual(readsFor(golden), {
        catalog: READ_OK, accounts: READ_OK, quota: READ_FAILED,
    });
    // The refusal's note carries the quota read's own error, never lands on
    // the account subject, and never claims the daemon is not running.
    const note = statusUnavailableNote(READ_FAILED,
        { facet: FACET_QUOTA, error: golden.facets.quota.error });
    assert.match(note.text, /quota_probe_failed/);
    assert.doesNotMatch(note.text, /Your agent accounts could not be read/);
    assert.doesNotMatch(note.text, /not running/);
});

test('a surface renders more than one facet, and the note names the second gap too', () => {
    // The accounts panel renders rows AND windows, the reviewer rows render
    // routes AND account pins: a banner that consults one facet leaves the
    // other's stale value on screen dressed as fresh.
    const reads = { catalog: READ_OK, accounts: READ_OK, quota: READ_FAILED };
    assert.equal(facetGapClause(reads, [FACET_CATALOG]), '');
    assert.match(facetGapClause(reads, [FACET_QUOTA]), /^Subscription limits were not read/);
    assert.match(facetGapClause(reads, [FACET_QUOTA]), /is last known\.$/);
    // Two gaps read as one sentence, and an unread-yet facet is not a gap.
    const both = facetGapClause({ catalog: READ_NOT_READ, accounts: READ_UNREAD, quota: READ_TRANSPORT },
        [FACET_CATALOG, FACET_ACCOUNTS, FACET_QUOTA]);
    assert.match(both, /Agents and subscription limits were not read/);
    assert.match(both, /shown for them is last known/);
    assert.equal(facetGapClause({ catalog: READ_OK }, [FACET_CATALOG]), '');
    assert.equal(facetGapClause({}, [FACET_CATALOG]), '');
    // Coalescing is by SUBJECT, not by enum: facets in the SAME state are
    // still each named, because one sentence about accounts says nothing about
    // agent discovery or the windows.
    const allFailed = { catalog: READ_FAILED, accounts: READ_FAILED, quota: READ_FAILED };
    const secondary = facetGapClause(allFailed, [FACET_CATALOG, FACET_QUOTA]);
    assert.match(secondary, /Agents and subscription limits were not read/);
});

test('each unavailable read state has ONE shared sentence, and an ok read has none', () => {
    const down = statusUnavailableNote(READ_NOT_READ);
    // NOT READ = never asked. It is NOT a diagnosis: a runtime that needs
    // repair, a foreign daemon and an ownership problem all land here, and
    // once the backend stamps `reads` per facet a RUNNING daemon can leave
    // one facet unasked. Naming a cause here would be a lie in all four cases.
    assert.match(down.text, /daemon was not asked/);
    assert.doesNotMatch(down.text, /is not running/);
    assert.match(down.text, /saved choices are unchanged/);
    assert.doesNotMatch(down.text, /not in discovery/);

    const dead = statusUnavailableNote(READ_TRANSPORT, { error: 'HTTP 503' });
    assert.match(dead.text, /HTTP 503/);
    assert.equal(dead.tone, 'error');

    const refused = statusUnavailableNote(READ_FAILED);
    assert.match(refused.text, /could not be read/);
    // A read that did not land NEVER claims a stopped daemon: the same state is
    // reached by a running daemon refusing one read and by an `unreachable`
    // answer, and both would be mislabeled by the stopped sentence.
    assert.doesNotMatch(refused.text, /not running/);
    const coarse = statusUnavailableNote(READ_INDETERMINATE);
    assert.match(coarse.text, /did not finish answering/);
    assert.doesNotMatch(coarse.text, /not running/);
    // All four are distinct: "nobody asked", "the request died", "this read
    // refused", "the answer did not complete and we cannot say which read".
    assert.equal(new Set([down.text, dead.text, refused.text, coarse.text]).size, 4);

    assert.match(statusUnavailableNote(READ_UNREAD).text, /Reading your agent accounts/);
    assert.equal(statusUnavailableNote(READ_OK), null);
    for (const state of [READ_NOT_READ, READ_FAILED, READ_TRANSPORT, READ_UNREAD, READ_INDETERMINATE]) {
        for (const facet of [FACET_CATALOG, FACET_ACCOUNTS, FACET_QUOTA]) {
            assert.doesNotMatch(statusUnavailableNote(state, { facet }).text, /coding[ -]agent/i);
        }
    }

    // The note names the facet it is about, so three surfaces can share it.
    // D-10: the subject is "agents", not "coding agents" — the same
    // subscriptions build presentations and run arbitrary tasks.
    assert.match(statusUnavailableNote(READ_NOT_READ, { facet: FACET_CATALOG }).text, /your agents were never checked/);
    assert.doesNotMatch(statusUnavailableNote(READ_NOT_READ, { facet: FACET_CATALOG }).text, /coding/);
    assert.match(statusUnavailableNote(READ_NOT_READ, { facet: FACET_QUOTA }).text, /subscription limits/);

    // The ACTION slot exists and is empty on this branch — an owner action that
    // could change the answer attaches here without reshaping any consumer.
    assert.equal(down.action, null);
    assert.ok('action' in down);
});

test('a store over a stopped daemon reports every facet as never-read', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => okResponse(STOPPED), doc: fakeDoc(),
    });
    await store.refresh();
    assert.deepEqual(store.reads,
        { catalog: READ_NOT_READ, accounts: READ_NOT_READ, quota: READ_NOT_READ });
    assert.equal(store.catalogKnown, false);
    assert.equal(store.accountsKnown, false);
    assert.equal(store.error, '');           // nothing failed — we DID get an answer
    assert.deepEqual(store.snapshot, STOPPED);
    store.dispose();
});

test('the store says the right sentence for EVERY daemon state it can be served', async () => {
    // One case per state, because the sentence is what the owner reads and the
    // mislabel is invisible in the data. The reviewer's live probe: catalog and
    // accounts SUCCEEDED, quota failed, the payload carried both successes —
    // and the panel announced a daemon that was not running.
    const cases = [
        ['running', READ_OK, null],
        ['stale', READ_NOT_READ, /daemon was not asked/],
        ['not_provisioned', READ_NOT_READ, /daemon was not asked/],
        ['foreign_daemon', READ_NOT_READ, /daemon was not asked/],
        ['unreachable', READ_INDETERMINATE, /did not finish answering/],
        ['error', READ_INDETERMINATE, /did not finish answering/],
        ['', READ_INDETERMINATE, /did not finish answering/],
    ];
    for (const [daemonState, expected, wording] of cases) {
        const store = createClaudexorStatusStore({
            fetchImpl: async () => okResponse(payloadOf({ state: daemonState })),
            doc: fakeDoc(),
        });
        await store.refresh();
        assert.deepEqual(store.reads, { catalog: expected, accounts: expected, quota: expected },
            `daemon.state=${daemonState || '(none)'}`);
        const note = store.unavailableNote(FACET_ACCOUNTS);
        if (wording === null) {
            assert.equal(note, null, `daemon.state=${daemonState} needs no note`);
        } else {
            assert.match(note.text, wording, `daemon.state=${daemonState}`);
            if (expected !== READ_NOT_READ) {
                assert.doesNotMatch(note.text, /was not asked/,
                    `daemon.state=${daemonState} was asked — it just did not answer`);
            }
        }
        store.dispose();
    }
});

test("a read that did not land carries the daemon's own explanation", async () => {
    // The exact payload the reviewer probed: the daemon IS running, one fanned
    // -out read refused, and the endpoint collapsed that into `unreachable`
    // plus a last_error. Routing the state to the shared sentence must not lose
    // the only detail the owner has — while still not attributing that error to
    // any one facet's subject.
    const store = createClaudexorStatusStore({
        fetchImpl: async () => okResponse(payloadOf(
            { state: 'unreachable', last_error: 'quota_read_failed: window read died' },
            { harnesses: [{ id: 'codex' }] })),
        doc: fakeDoc(),
    });
    await store.refresh();
    const note = store.unavailableNote(FACET_ACCOUNTS);
    assert.match(note.text, /did not finish answering/);
    assert.match(note.text, /window read died/, "the daemon's own reason survives");
    assert.doesNotMatch(note.text, /not running/);
    assert.doesNotMatch(note.text, /Your agent accounts could not be read/);
    // A stopped daemon keeps its calm line — a crashed daemon also lands in
    // `stale`, and that decision is deliberate.
    const stopped = createClaudexorStatusStore({
        fetchImpl: async () => okResponse(payloadOf({ state: 'stale', last_error: 'boom' })),
        doc: fakeDoc(),
    });
    await stopped.refresh();
    assert.doesNotMatch(stopped.unavailableNote(FACET_ACCOUNTS).text, /boom/);
    store.dispose();
    stopped.dispose();
});

test('a 200 that is not a status answer is a protocol failure, not an empty world', async () => {
    // A 200 carrying non-JSON parsed to {} and sailed through every facet
    // derivation, so the app confidently rendered "nothing is connected" off a
    // body it never understood. The bar is the producer's UNCONDITIONAL fields
    // with their types — checking `daemon` alone was not depth: a bare
    // {daemon:{state:'running'}} passed and yielded three authoritative facets
    // over collections the body did not contain.
    for (const body of [null, {}, [], 'ok', { daemon: 'running' },
        { daemon: { state: 'running' } },
        { daemon: { state: 'running' }, harnesses: [], profiles: {} },
        { daemon: { state: 'running' }, harnesses: {}, profiles: {}, quota: [] },
        { daemon: { state: 'running' }, harnesses: [], profiles: [], quota: [] }]) {
        const store = createClaudexorStatusStore({
            fetchImpl: async () => ({ ok: true, status: 200, json: async () => body }),
            doc: fakeDoc(),
        });
        await store.refresh();
        assert.deepEqual(store.reads,
            { catalog: READ_TRANSPORT, accounts: READ_TRANSPORT, quota: READ_TRANSPORT },
            `body ${JSON.stringify(body)} must not read as an answer`);
        assert.match(store.error, /could not be understood/);
        assert.equal(store.accountsKnown, false);
        store.dispose();
    }
    // A body that IS a status answer still lands, empty collections included.
    const good = createClaudexorStatusStore({
        fetchImpl: async () => okResponse(RUNNING), doc: fakeDoc(),
    });
    await good.refresh();
    assert.equal(good.error, '');
    assert.equal(good.accountsKnown, true);
    good.dispose();
});

test('a non-2xx answer is a transport error, not a silent stale snapshot', async () => {
    // The old private poller kept the last good payload and said NOTHING on an
    // HTTP failure, so a dead endpoint was indistinguishable from a healthy
    // idle daemon.
    let ok = true;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => (ok
            ? okResponse(RUNNING)
            : { ok: false, status: 503, json: async () => ({ error: 'daemon exploded' }) }),
        doc: fakeDoc(),
    });
    await store.refresh();
    assert.equal(store.catalogKnown, true);
    ok = false;
    await store.refresh();
    assert.equal(store.facet(FACET_CATALOG), READ_TRANSPORT);
    assert.equal(store.facet(FACET_ACCOUNTS), READ_TRANSPORT);
    assert.match(store.error, /daemon exploded/);
    // The snapshot is kept for whoever still wants to show it, but it no
    // longer counts as discovery.
    assert.equal(store.catalogKnown, false);
    assert.deepEqual(store.snapshot, RUNNING);
    store.dispose();
});

// ---------------------------------------------------------------------------
// One request per moment in time.
// ---------------------------------------------------------------------------

test('concurrent refreshes share ONE http request and ONE snapshot', async () => {
    let started = 0;
    let release = null;
    const gate = new Promise((resolve) => { release = resolve; });
    const store = createClaudexorStatusStore({
        fetchImpl: async () => { started += 1; await gate; return okResponse(RUNNING); },
        doc: fakeDoc(),
    });
    const calls = [store.refresh(), store.refresh(), store.refresh()];
    assert.equal(started, 1, 'overlapping callers did not start a second read');
    release();
    const answers = await Promise.all(calls);
    assert.equal(started, 1, 'the shared read served every caller');
    assert.equal(new Set(answers).size, 1, 'every caller got the SAME snapshot object');
    // A settled read releases the slot: the next call really reads again.
    await store.refresh();
    assert.equal(started, 2);
    store.dispose();
});

test('includeModels is sticky-upgrading: no later read downgrades the shared snapshot', async () => {
    const urls = [];
    let release = null;
    const gate = new Promise((resolve) => { release = resolve; });
    const store = createClaudexorStatusStore({
        fetchImpl: async (url) => { urls.push(url); await gate; return okResponse(RUNNING); },
        doc: fakeDoc(),
    });
    // The accounts panel's model-less poll is in flight when the review-lane
    // rows ask for discovery: the upgrade cannot be served by that request, so ONE
    // follow-up is queued — and every upgrading caller shares it.
    const plain = store.refresh();
    const upgrade = [store.refresh({ includeModels: true }), store.refresh({ includeModels: true })];
    assert.equal(urls.length, 1);
    release();
    await Promise.all([plain, ...upgrade]);
    assert.deepEqual(urls, ['/api/claudexor/status', '/api/claudexor/status?include=models']);
    assert.equal(store.includesModels, true);

    // Sticky: a later plain refresh KEEPS models rather than downgrading the
    // snapshot the review-lane and delegation selects depend on.
    await store.refresh();
    assert.equal(urls.at(-1), '/api/claudexor/status?include=models');
    assert.equal(store.includesModels, true);
    store.dispose();
});

// ---------------------------------------------------------------------------
// Polling is a held resource, and every acquisition has a disposer.
// ---------------------------------------------------------------------------

test('polling needs a subscriber AND a visible surface; unsubscribing disarms the timer', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let reads = 0;
    let visible = false;
    const doc = fakeDoc();
    const store = createClaudexorStatusStore({
        fetchImpl: async () => { reads += 1; return okResponse(RUNNING); },
        doc, pollMs: 5000,
    });
    // No subscriber: an explicit read runs, but nothing is armed afterwards.
    await store.refresh();
    assert.equal(reads, 1);
    assert.equal(store.polling, false, 'a store nobody listens to never arms a timer');

    const off = store.subscribe(() => {}, { visible: () => visible });
    assert.equal(store.polling, false, 'subscribed but off-screen: still no timer');
    visible = true;
    // A visibility event re-evaluates the gate (and catches up on the stretch
    // spent hidden).
    doc.fire('visibilitychange');
    await flush();
    assert.equal(reads, 2, 'becoming visible catches up on the hidden stretch');
    assert.equal(store.polling, true, 'visible surface + subscriber: polling');

    t.mock.timers.tick(5000);
    await flush();
    assert.equal(reads, 3, 'the armed tick performed exactly one more read');

    off();
    assert.equal(store.polling, false, 'the unsubscribe disposer disarmed the timer');
    assert.equal(store.subscriberCount, 0);
    store.dispose();
    t.mock.timers.reset();
});

test('any bound surface can keep the poll armed — not only the accounts panel', async (t) => {
    // The two sections that moved off the accounts panel's tab subscribed with
    // NO visibility predicate and no activation hook, so the store had nobody
    // who could say "I am on screen": polling stayed false and the fetch count
    // stuck at 1, while their comments promised the daemon recovering would be
    // picked up without a reload. Here the accounts panel is HIDDEN and only the
    // other surface is visible.
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let reads = 0;
    const elements = {
        'harness-accounts-rows': { offsetParent: null },        // its tab is not active
        'reviewer-triad-rows': { offsetParent: {} },            // this surface IS on screen
    };
    const doc = fakeDoc();
    doc.getElementById = (id) => elements[id] || null;
    const win = (() => {
        const listeners = {};
        return {
            addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
            removeEventListener(type, fn) {
                listeners[type] = (listeners[type] || []).filter((f) => f !== fn);
            },
            fire(type) { for (const fn of [...(listeners[type] || [])]) fn(); },
            count(type) { return (listeners[type] || []).length; },
        };
    })();
    const store = createClaudexorStatusStore({
        fetchImpl: async () => { reads += 1; return okResponse(RUNNING); }, doc, pollMs: 5000,
    });
    const releaseAccounts = bindStatusSurface(store, {
        elementId: 'harness-accounts-rows', listener: () => {}, doc, win,
    });
    const releaseReviewer = bindStatusSurface(store, {
        elementId: 'reviewer-triad-rows', listener: () => {}, includeModels: true, doc, win,
    });
    await store.refresh();
    assert.equal(reads, 1);
    assert.equal(store.polling, true, 'the visible surface arms the timer on its own');

    t.mock.timers.tick(5000);
    await flush();
    assert.equal(reads, 2, 'a SECOND tick happened for a surface that is not the accounts panel');

    // Activation is judged by the same predicate — no tab name anywhere — so the
    // section that is on screen catches up, and the hidden one costs nothing.
    win.fire('ouro:settings-subtab-shown');
    await flush();
    assert.equal(reads, 3, 'reaching the surface reads immediately, without a reload');

    // Hide it: the last visible surface leaving disarms the timer.
    elements['reviewer-triad-rows'].offsetParent = null;
    doc.fire('visibilitychange');
    assert.equal(store.polling, false);

    releaseReviewer();
    releaseAccounts();
    assert.equal(store.subscriberCount, 0, 'the binding disposer released the subscription');
    assert.equal(win.count('ouro:page-shown'), 0, 'and both window listeners');
    assert.equal(win.count('ouro:settings-subtab-shown'), 0);
    store.dispose();
    t.mock.timers.reset();
});

test('surface binding delegates a visible activation to its owner action', async () => {
    const elements = { accounts: { offsetParent: {} }, hidden: { offsetParent: null } };
    const doc = fakeDoc();
    doc.getElementById = (id) => elements[id] || null;
    const listeners = {};
    const win = {
        addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
        removeEventListener(type, fn) {
            listeners[type] = (listeners[type] || []).filter((item) => item !== fn);
        },
        fire(type) { return Promise.all((listeners[type] || []).map((fn) => fn())); },
    };
    const store = createClaudexorStatusStore({ fetchImpl: async () => okResponse(RUNNING), doc });
    let activations = 0;
    const releaseVisible = bindStatusSurface(store, {
        elementId: 'accounts', listener: () => {}, doc, win,
        onActivate: () => { activations += 1; },
    });
    const releaseHidden = bindStatusSurface(store, {
        elementId: 'hidden', listener: () => {}, doc, win,
        onActivate: () => { activations += 100; },
    });

    await win.fire('ouro:settings-subtab-shown');
    assert.equal(activations, 1, 'only the visible surface ran its owner action');

    releaseHidden();
    releaseVisible();
    store.dispose();
});

test('a hidden page pauses polling, and a login hold keeps it awake off-surface', () => {
    const doc = fakeDoc();
    const store = createClaudexorStatusStore({
        fetchImpl: async () => okResponse(RUNNING), doc, pollMs: 5000,
    });
    store.subscribe(() => {}, { visible: () => true });
    assert.equal(store.polling, true);

    doc.hidden = true;
    doc.fire('visibilitychange');
    assert.equal(store.polling, false, 'document.hidden pauses the poll');
    doc.hidden = false;
    doc.fire('visibilitychange');
    assert.equal(store.polling, true, 'and it resumes on visibility');

    // Off-surface (settings closed) the poll stops — unless a login job holds it.
    const store2 = createClaudexorStatusStore({
        fetchImpl: async () => okResponse(RUNNING), doc: fakeDoc(), pollMs: 5000,
    });
    store2.subscribe(() => {}, { visible: () => false });
    assert.equal(store2.polling, false);
    const release = store2.holdPolling('login-job');
    assert.equal(store2.polling, true, 'a live login keeps the account rows moving');
    release();
    assert.equal(store2.polling, false, 'the hold disposer releases it');
    store.dispose();
    store2.dispose();
});

test('dispose releases the timer, the listeners and the visibilitychange handler, and refuses a LATER refresh', async () => {
    const doc = fakeDoc();
    let reads = 0;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => { reads += 1; return okResponse(RUNNING); }, doc, pollMs: 5000,
    });
    const seen = [];
    store.subscribe((v) => seen.push(v.reads.accounts), { visible: () => true });
    store.holdPolling('login-job');
    await store.refresh();
    assert.ok(seen.length > 0);
    assert.equal(doc.count('visibilitychange'), 1);
    assert.equal(store.polling, true);

    store.dispose();
    assert.equal(store.polling, false, 'no timer left armed');
    assert.equal(doc.count('visibilitychange'), 0, 'the document listener is removed');
    assert.equal(store.subscriberCount, 0);
    // A disposed store neither notifies nor reads again.
    const before = reads;
    seen.length = 0;
    await store.refresh();
    assert.equal(reads, before, 'a disposed store performs no further reads');
    assert.deepEqual(seen, []);
    // …and it is idempotent (a double destroy must not throw).
    store.dispose();
});

test('an IN-FLIGHT model upgrade queued before dispose never reads afterwards', async () => {
    // The disposer test above only proves that an EXPLICIT later refresh is
    // refused — its old name overstated what it covered. The queued upgrade
    // reaches the read path directly, bypassing refresh()'s guard: a section
    // that asked for model discovery while a plain poll was in flight left a
    // continuation that fired AFTER dispose, spending four CLI-probing daemon
    // round-trips for a surface with zero subscribers and polling off.
    const urls = [];
    let release = null;
    const gate = new Promise((resolve) => { release = resolve; });
    const store = createClaudexorStatusStore({
        fetchImpl: async (url) => { urls.push(url); await gate; return okResponse(RUNNING); },
        doc: fakeDoc(),
    });
    const plain = store.refresh();
    const upgrade = store.refresh({ includeModels: true });   // queued behind it
    assert.deepEqual(urls, ['/api/claudexor/status']);

    store.dispose();
    release();
    await Promise.all([plain, upgrade]);
    await flush();
    assert.deepEqual(urls, ['/api/claudexor/status'],
        'the queued upgrade must not read for a disposed store');
});

test('subscribers are notified with the settled view, and one broken listener cannot silence the rest', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => okResponse(RUNNING), doc: fakeDoc(),
    });
    const seen = [];
    store.subscribe(() => { throw new Error('this consumer is broken'); });
    store.subscribe((v) => seen.push(v));
    await store.refresh();
    // Two paints for the FIRST read only: the announcement before it lands
    // ("Checking Claudexor…", which the panel needs because the daemon probes
    // every CLI and can take a minute) and the settled answer. Later reads
    // notify once — a repaint per poll tick is churn.
    assert.equal(seen.length, 2);
    assert.equal(seen[0].loading, true);
    // Before the first read settles EVERY facet is honestly "unread" — the
    // store's own dimension, which the wire cannot carry.
    assert.deepEqual(seen[0].reads,
        { catalog: READ_UNREAD, accounts: READ_UNREAD, quota: READ_UNREAD });
    assert.deepEqual(seen[1].reads, { catalog: READ_OK, accounts: READ_OK, quota: READ_OK });
    assert.equal(seen[1].generation, 1);
    assert.equal(seen[1].loading, false);
    seen.length = 0;
    await store.refresh();
    assert.equal(seen.length, 1, 'no pre-request repaint once anything has been said');
    assert.equal(store.everSettled, true);
    store.dispose();
});

// ---------------------------------------------------------------------------
// The payload projection lives with the payload.
// ---------------------------------------------------------------------------

test('accountRows and accountLoginConfirmed read the wire shape from ONE place', () => {
    // Every account is a named registry row (#244): an old client's
    // `harnessAccounts` carrier beside them is never projected, so no row
    // with an empty id exists to confirm a login.
    const payload = { profiles: {
        harnessAccounts: [{ harness_id: 'codex', native_login_detected: true }],
        profiles: [{ profile: { harness_id: 'claude', profile_id: 'work' }, status: { verification: 'passed', verification_source: 'vendor' } }],
    } };
    const rows = accountRows(payload);
    assert.deepEqual(rows.map((r) => [r.harness, r.profile_id, r.kind]), [['claude', 'work', 'profile']]);
    assert.equal(accountLoginConfirmed(payload, 'codex', ''), false);
    assert.equal(accountLoginConfirmed(payload, 'claude', 'work'), true);
    assert.equal(accountLoginConfirmed(payload, 'claude', 'other'), false);
    assert.equal(accountLoginConfirmed({}, 'codex', ''), false);
});

test('a unified payload gets every row from profiles — no synthetic natives', () => {
    // Migrated default logins are ordinary registry rows; a compatibility row
    // that slipped into `harnessAccounts` never double-renders the account,
    // with or without the server's `unified_accounts` stamp.
    const profiles = {
        harnessAccounts: [{ harness_id: 'codex', native_login_detected: true }],
        accountPools: [{ harness_id: 'codex', next_up: { kind: 'profile', profileId: 'codex-default' } }],
        profiles: [
            { profile: { harness_id: 'codex', profile_id: 'codex-default', enabled: true },
                status: { verification: 'passed', verification_source: 'local_store' } },
            { profile: { harness_id: 'codex', profile_id: 'work', enabled: false },
                status: { verification: 'passed', verification_source: 'vendor' } },
        ],
    };
    for (const payload of [{ unified_accounts: true, profiles }, { profiles }]) {
        assert.deepEqual(accountRows(payload).map((r) => [r.harness, r.profile_id, r.kind, r.enabled]),
            [['codex', 'codex-default', 'profile', true], ['codex', 'work', 'profile', false]]);
        // The verify-race addresses the migrated row by its REAL id; the empty
        // address resolves nothing.
        assert.equal(accountLoginConfirmed(payload, 'codex', 'codex-default'), true);
        assert.equal(accountLoginConfirmed(payload, 'codex', ''), false);
    }
});

test('the enabled projection fails open on payloads that predate the field', () => {
    const payload = { profiles: {
        profiles: [{ profile: { harness_id: 'claude', profile_id: 'work' },
            status: {} }],
    } };
    // No `enabled` anywhere: nothing may claim an exclusion from rotation.
    assert.deepEqual(accountRows(payload).map((r) => r.enabled), [true]);
});

test('nextUpAccount reads the accountPools verdict alone', () => {
    // One routing authority. An old client's `harnessAccounts[].next_up` is
    // never read, so a harness only it speaks for is an honest null.
    const both = { profiles: {
        accountPools: [{ harness_id: 'codex', next_up: { kind: 'api_key_route' } }],
        harnessAccounts: [{ harness_id: 'codex', next_up: { kind: 'native', route: 'local_session' } }],
    } };
    assert.deepEqual(nextUpAccount(both, 'codex'), { kind: 'api_key_route' });
    const legacy = { profiles: {
        harnessAccounts: [{ harness_id: 'codex', next_up: { kind: 'native', route: 'local_session' } }],
    } };
    assert.equal(nextUpAccount(legacy, 'codex'), null);
    assert.equal(nextUpAccount({}, 'codex'), null);
    assert.equal(nextUpAccount(both, ''), null);
    // An unknown future kind rides through verbatim — the consumer renders it
    // fail-safe; the reader never filters or rewrites it.
    const future = { profiles: {
        accountPools: [{ harness_id: 'codex', next_up: { kind: 'quantum_pool' } }],
    } };
    assert.deepEqual(nextUpAccount(future, 'codex'), { kind: 'quantum_pool' });
});

test('a refused wake does not stop the visible panel from polling', async (t) => {
    // The poll tick that fires during the POST disarms itself and joins the
    // wake; re-arming only on success left the panel timerless after a 503 —
    // it could never notice the daemon coming up on its own, and the owner's
    // only recovery was another click or a tab switch.
    const timers = [];
    const origSet = globalThis.setTimeout; const origClear = globalThis.clearTimeout;
    globalThis.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length; };
    globalThis.clearTimeout = () => {};
    try {
        const store = createClaudexorStatusStore({
            fetchImpl: async (url, opts) => {
                if (opts && opts.method === 'POST') {
                    return { ok: false, status: 503, json: async () => ({ error: 'claudexord_not_installed' }) };
                }
                return { ok: true, json: async () => ({ daemon: { state: 'stale', runtime: {} },
                    reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } }) };
            },
        });
        const unsub = store.subscribe(() => {}, { visible: () => true });
        const armedBefore = timers.length;
        const outcome = await store.wake();
        assert.equal(outcome.ok, false);
        assert.ok(timers.length > armedBefore,
            'no poll timer was re-armed after the refused wake');
        unsub();
    } finally {
        globalThis.setTimeout = origSet; globalThis.clearTimeout = origClear;
    }
});

test('claudexorPreparationLine phases by runtime state and daemon liveness', () => {
    assert.equal(claudexorPreparationLine({
        daemon: { state: 'unreachable', runtime: { state: 'installing', target_version: '3.3.14' } },
    }), 'Installing Claudexor 3.3.14…');
    assert.equal(claudexorPreparationLine({
        daemon: { state: 'unreachable', runtime: { state: 'installing' } },
    }), 'Installing Claudexor…', 'no version claim without a version');
    assert.equal(claudexorPreparationLine({
        daemon: { state: 'stale', runtime: { state: 'ready' } },
    }), 'Starting the Claudexor daemon…');
    assert.equal(claudexorPreparationLine({
        daemon: { state: 'running', runtime: { state: 'ready' } },
    }), 'Checking Claudexor…', 'a serving engine is being checked, not started');
    assert.equal(claudexorPreparationLine({
        daemon: { state: 'unreachable', runtime: { state: 'ready' } },
    }), 'Checking Claudexor…',
        'a failed fan-out rewrites a LIVE daemon to unreachable — never a Starting claim');
    assert.equal(claudexorPreparationLine({
        daemon: { state: 'stale', ownership_problem: 'foreign home', runtime: { state: 'ready' } },
    }), 'Checking Claudexor…',
        'ensure refuses a foreign daemon home — never promise a start that is refused');
    assert.equal(claudexorPreparationLine({
        daemon: { state: 'unreachable', ownership_problem: 'foreign home',
            runtime: { state: 'installing', target_version: '3.3.14' } },
    }), 'Checking Claudexor…',
        'an ownership refusal outranks every positive phase claim, installing included');
    assert.equal(claudexorPreparationLine(null), 'Checking Claudexor…',
        'no snapshot is no phase evidence — the honest generic');
    assert.equal(claudexorPreparationLine({
        daemon: { state: 'running', runtime: {} },
    }), 'Checking Claudexor…');
});
