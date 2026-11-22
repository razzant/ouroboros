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
import { accountGroups, accountRowFacts, bareRowStatusText, confirmRemoveAccount, refreshActionKind, serviceBannerLine } from '../modules/harness_accounts.js';

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
    accountCapabilityMarks,
    accountLoginConfirmed,
    accountRows,
    capabilityMarksReadFailedAt,
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

test('an event requests models after an older model read; ordinary callers still share it', async () => {
    let release;
    let reads = 0;
    const gate = new Promise(resolve => { release = resolve; });
    const store = createClaudexorStatusStore({doc: fakeDoc(), fetchImpl: async () => {
        const ordinal = ++reads;
        if (ordinal === 1) await gate;
        return okResponse({...RUNNING, harnesses: [{id: 'future', models: [{id: ordinal === 1 ? 'old' : 'new'}]}]});
    }});
    const old = store.refresh({includeModels: true});
    const plain = store.refresh();
    assert.equal(reads, 1);
    const after = store.refresh({includeModels: true, afterCurrent: true});
    const another = store.refresh({includeModels: true, afterCurrent: true});
    release();
    await Promise.all([old, plain, after, another]);
    assert.equal(reads, 2, 'settled-event callers share one read that starts after the older one');
    assert.equal(store.snapshot.harnesses[0].models[0].id, 'new');
    await store.refresh({includeModels: true, afterCurrent: true});
    assert.equal(reads, 3, 'no extra read when none was in flight');
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

for (const [name, requests, withModels] of [
    ['event then models', [{ afterCurrent: true }, { includeModels: true }], true],
    ['models then event', [{ includeModels: true }, { afterCurrent: true }], true],
    ['events without models', [{ afterCurrent: true }, { afterCurrent: true }], false],
]) {
    test(`queued discovery shares the requested catalog: ${name}`, async (t) => {
        const urls = [];
        let release;
        const gate = new Promise(resolve => { release = resolve; });
        const models = [{ id: 'discovered-model' }];
        const store = createClaudexorStatusStore({ doc: fakeDoc(), fetchImpl: async url => {
            urls.push(url);
            if (urls.length === 1) await gate;
            return okResponse({ ...RUNNING, harnesses: [{ id: 'codex',
                models: url.includes('?include=models') ? models : [] }] });
        } });
        t.after(() => store.dispose());
        const first = store.refresh();
        assert.equal(store.refresh(), first, 'ordinary callers share the current model-less read');
        const joined = requests.map(options => store.refresh(options));
        assert.equal(joined[0], joined[1], 'both intents share one queued read');
        assert.deepEqual(urls, ['/api/claudexor/status']);
        release();
        const [before, ...answers] = await Promise.all([first, ...joined]);
        const expectedURL = withModels ? '/api/claudexor/status?include=models' : '/api/claudexor/status';
        assert.deepEqual(urls, ['/api/claudexor/status', expectedURL]);
        assert.deepEqual(before.harnesses[0].models, []);
        assert.equal(answers[0], answers[1], 'joiners receive the same snapshot');
        assert.deepEqual(answers[0].harnesses[0].models, withModels ? models : []);
        assert.equal(store.includesModels, withModels);

        await store.refresh();
        assert.deepEqual(urls, ['/api/claudexor/status', expectedURL, expectedURL],
            'later reads keep requested models without inventing a model request');
        assert.equal(store.includesModels, withModels);
    });
}

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
    win.fire('ouro:settings-page-shown');
    await flush();
    assert.equal(reads, 4, 'activation reads the narrow observation and upgrades missing models independently');

    // Hide it: the last visible surface leaving disarms the timer.
    elements['reviewer-triad-rows'].offsetParent = null;
    doc.fire('visibilitychange');
    assert.equal(store.polling, false);

    releaseReviewer();
    releaseAccounts();
    assert.equal(store.subscriberCount, 0, 'the binding disposer released the subscription');
    assert.equal(win.count('ouro:page-shown'), 0, 'and both window listeners');
    assert.equal(win.count('ouro:settings-page-shown'), 0);
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

    await win.fire('ouro:settings-page-shown');
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

// Fast observations and controls keep working while discovery is blocked.
const accountPayload = enabled => ({ ...RUNNING,
    reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' },
    profiles: { profiles: [{ profile: { harness_id: 'codex', profile_id: 'one', enabled } }], accountPools: [] },
});
const passivePayload = enabled => ({ view: 'quota', profiles: accountPayload(enabled).profiles,
    reads: { catalog: 'not_read', accounts: 'ok', quota: 'ok' }, quota: [], quota_absences: [],
    resources: [{ target: { harness: 'codex', profile_id: 'one' }, balances: { observed_at: '2026-11-13T19:00:00Z' } }] });

const rawMixedRoster = () => ({ profiles: [
    { profile: { harness_id: 'openrouter', profile_id: 'key', kind: 'api_key', enabled: true } },
    { profile: { harness_id: 'codex', profile_id: 'one', kind: 'native_session', enabled: false } },
    { profile: { harness_id: 'new-native', profile_id: 'session', kind: 'native_session', enabled: true } },
], accountPools: [] });
const mixedPassive = () => ({ ...passivePayload(false), profiles: rawMixedRoster() });
const loginPolicyStatus = (ids = ['codex', 'new-native']) => ({ ...accountPayload(true),
    // Catalog absence is NOT policy: this native harness is intentionally missing there.
    harnesses: [{ id: 'codex' }],
    profiles: { ...accountPayload(true).profiles, login_capable_harnesses: ids },
});

test('raw passive roster keeps the full producer login policy in shared account rows and groups', async () => {
    const raw = mixedPassive();
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url =>
        okResponse(url.includes('view=quota') ? raw : loginPolicyStatus()) });
    await store.refresh(); await store.observe();
    assert.deepEqual(accountRows(store.snapshot).map(row => row.harness), ['codex', 'new-native']);
    assert.equal(accountRows(store.snapshot)[0].enabled, false, 'legitimate rows still update');
    assert.equal(accountGroups(store.snapshot).some(group => group.harness === 'openrouter'), false,
        'the raw API profile must not create a fake login family');
    assert.equal(raw.profiles.profiles.length, 3, 'the passive envelope stays unchanged');
    store.dispose();
});

test('initial passive waits for policy, then an older full read projects the newer raw roster', async () => {
    let finishFull;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url =>
        url.includes('view=quota') ? okResponse(mixedPassive())
            : new Promise(resolve => { finishFull = resolve; }) });
    const full = store.refresh();
    await store.observe();
    assert.deepEqual(accountRows(store.snapshot), [], 'no login eligibility was read yet');
    assert.equal(store.accountsKnown, false, 'raw membership alone is not Accounts presentation');
    finishFull(okResponse(loginPolicyStatus())); await full;
    assert.deepEqual(accountRows(store.snapshot).map(row => row.harness), ['codex', 'new-native']);
    assert.equal(store.accountsKnown, true, 'the completed policy admits the current raw roster');
    assert.equal(accountRows(store.snapshot)[0].enabled, false, 'older full roster cannot roll it back');
    store.dispose();
});

test('explicit unknown policy and legacy full status retain the established fail-open behavior', async () => {
    for (const full of [loginPolicyStatus(null), accountPayload(true)]) {
        const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url =>
            okResponse(url.includes('view=quota') ? mixedPassive() : full) });
        await store.refresh(); await store.observe();
        assert.deepEqual(accountRows(store.snapshot).map(row => row.harness),
            ['openrouter', 'codex', 'new-native']);
        store.dispose();
    }
});

test('new manifest policy survives a failed roster read and reprojects the next raw observation', async () => {
    let full = loginPolicyStatus(['codex', 'new-native']);
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url =>
        okResponse(url.includes('view=quota') ? mixedPassive() : full) });
    await store.refresh(); await store.observe();
    full = { ...loginPolicyStatus(['codex']), reads: { catalog: 'ok', accounts: 'failed', quota: 'ok' } };
    await store.refresh(); await store.observe();
    assert.deepEqual(accountRows(store.snapshot).map(row => row.harness), ['codex']);
    store.dispose();
});

test('explicit policy and hidden raw rows survive PATCH then a stale full completion', async () => {
    let phase = 'first', finishFull, finishPatch;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async (url, init = {}) => {
        if (init.method === 'PATCH') return new Promise(resolve => { finishPatch = resolve; });
        if (url.includes('view=quota')) return okResponse(mixedPassive());
        return phase === 'first' ? okResponse(loginPolicyStatus())
            : new Promise(resolve => { finishFull = resolve; });
    } });
    await store.refresh(); await store.observe(); phase = 'stale';
    const full = store.refresh(); const patch = store.setEnabled('codex', 'one', false);
    finishPatch(okResponse({ ok: true, harness: 'codex', profile_id: 'one', enabled: false }));
    await patch;
    finishFull(okResponse(loginPolicyStatus())); await full; await flush();
    assert.deepEqual(accountRows(store.snapshot).map(row => [row.harness, row.enabled]),
        [['codex', false], ['new-native', true]]);
    assert.equal(accountGroups(store.snapshot).some(group => group.harness === 'openrouter'), false);
    store.dispose();
});

test('confirmed toggle fences reads begun before and during PATCH without waiting for discovery', async () => {
    let finishFull, finishPatch, finishPassive;
    let mode = 'initial';
    const requests = [];
    const store = createClaudexorStatusStore({ doc: fakeDoc(), fetchImpl: async (url, init = {}) => {
        requests.push([url, init.method]);
        if (init.method === 'PATCH') return new Promise(resolve => { finishPatch = resolve; });
        if (url.includes('view=quota')) {
            if (mode === 'during') return new Promise(resolve => { finishPassive = resolve; });
            return okResponse(passivePayload(false));
        }
        if (mode === 'initial') return okResponse(accountPayload(true));
        return new Promise(resolve => { finishFull = resolve; });
    } });
    await store.refresh();
    mode = 'during';
    const full = store.refresh();
    const write = store.setEnabled('codex', 'one', false);
    await flush();
    assert.ok(finishPatch, 'PATCH dispatch did not wait for full discovery');
    const during = store.observe();
    assert.equal(store.profileAction('codex', 'one').busy, true);
    finishPatch(okResponse({ ok: true, harness: 'codex', profile_id: 'one', enabled: false }));
    await write;
    assert.equal(accountRows(store.snapshot)[0].enabled, false);
    mode = 'after';
    // The confirmation's follow-up read waits behind the held passive read, so only the
    // dispatch fence can keep each older answer from reverting the confirmed switch.
    finishFull(okResponse(accountPayload(true)));
    await full; await flush();
    assert.equal(accountRows(store.snapshot)[0].enabled, false, 'a full read begun before PATCH reverted it');
    finishPassive(okResponse(passivePayload(true)));
    await during; await flush();
    assert.equal(accountRows(store.snapshot)[0].enabled, false, 'a passive read begun during PATCH reverted it');
    await store.observe(); await flush();
    assert.equal(accountRows(store.snapshot)[0].enabled, false);
    assert.equal(store.profileAction('codex', 'one').busy, false);
    assert.equal(store.snapshot.resources[0].balances.observed_at, '2026-11-13T19:00:00Z');
    assert.ok(requests.some(([url]) => url.includes('view=quota')));
    store.dispose();
});

test('passive failures retain roster and catalog without claiming membership; successful empty clears it', async () => {
    let body = passivePayload(false);
    const store = createClaudexorStatusStore({ doc: fakeDoc(), fetchImpl: async url =>
        okResponse(url.includes('view=quota') ? body : accountPayload(true)) });
    await store.refresh({ includeModels: true });
    await store.observe();
    assert.equal(store.catalogKnown, true);
    assert.equal(store.snapshot.resources.length, 1);
    body = { ...passivePayload(false), profiles: {}, reads: { catalog: 'not_read', accounts: 'failed', quota: 'ok' } };
    await store.observe();
    assert.equal(accountRows(store.snapshot).length, 1);
    assert.equal(store.accountsKnown, false);
    assert.equal(store.catalogKnown, true);
    body = { ...passivePayload(false), profiles: { profiles: [], accountPools: [] } };
    await store.observe();
    assert.equal(accountRows(store.snapshot).length, 0);
    store.dispose();
});

test('fast facets recover beside failed discovery without clearing its catalog error', async () => {
    let fail = false;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => {
        if (url.includes('view=quota')) return okResponse(passivePayload(false));
        return fail ? { ok: false, status: 503, json: async () => ({ error: 'catalog held' }) }
            : okResponse(accountPayload(true));
    } });
    await store.refresh(); fail = true; await store.refresh();
    assert.equal(store.accountsKnown, false);
    await store.observe();
    assert.equal(store.accountsKnown, true);
    assert.equal(store.quotaKnown, true);
    assert.equal(store.catalogKnown, false);
    assert.match(store.unavailableNote('catalog').text, /catalog held/);
    assert.equal(store.unavailableNote('accounts'), null);
    store.dispose();
});

test('resource-only completion leaves an overlapping roster eligible; omitted resources keep their own date', async () => {
    let finishProvider, finishRoster;
    const resource = { target: { harness: 'codex', profile_id: 'one' }, balances: { observed_at: '2026-11-13T19:00:00Z' } };
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => {
        if (url.includes('/quota/refresh')) return new Promise(resolve => { finishProvider = resolve; });
        if (url.includes('view=quota')) return new Promise(resolve => { finishRoster = resolve; });
        return okResponse(accountPayload(true));
    } });
    await store.refresh();
    const provider = store.refreshResources();
    const roster = store.observe();
    finishProvider(okResponse({ snapshots: [], absences: [], resources: [resource] }));
    await provider;
    const body = passivePayload(false); delete body.resources;
    finishRoster(okResponse(body)); await roster;
    assert.equal(accountRows(store.snapshot)[0].enabled, false);
    assert.deepEqual(store.snapshot.resources, [resource]);
    assert.equal(store.catalogKnown, true);
    store.dispose();
});

test('visible cadence reads snapshots frequently, never asks providers by itself and leaves discovery lazy', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let clock = 1000;
    t.mock.method(Date, 'now', () => clock);
    const urls = [];
    const store = createClaudexorStatusStore({ doc: fakeDoc(), fetchImpl: async url => {
        urls.push(url);
        return okResponse(url.includes('/quota/refresh') ? { snapshots: [], absences: [] }
            : url.includes('view=quota') ? passivePayload(true) : accountPayload(true));
    } });
    await store.refresh();
    const off = store.subscribe(() => {}, { visible: () => true });
    for (let tick = 1; tick <= 24; tick++) { clock += 5000; t.mock.timers.tick(5000); await flush(); }
    assert.equal(urls.filter(url => url === '/api/claudexor/status').length, 2);
    assert.equal(urls.filter(url => url.includes('view=quota')).length, 24);
    assert.equal(urls.filter(url => url.includes('/quota/refresh')).length, 0,
        'the engine refreshes ordinary limits itself; providers are asked only by Refresh');
    assert.equal(store.resourceAction(null).busy, undefined, 'nothing in the background occupies Refresh');
    off(); const before = urls.length;
    clock += 120000; t.mock.timers.tick(120000); await flush();
    assert.equal(urls.length, before);
    store.dispose();
});

// Review of #3378: full-only facts, failed passive ordering and recovery of the new cadence.
const markedStatus = ({ enabled = true, failedMarks = false } = {}) => ({ ...loginPolicyStatus(['codex']),
    profiles: { ...accountPayload(enabled).profiles, login_capable_harnesses: ['codex'],
        ...(failedMarks ? { routing_evidence_read: { state: 'failed', observed_at: '2026-11-14T08:00:00Z' } } : {
            routing_evidence: { capabilities: [{ harness_id: 'codex', profile_id: 'one', requested_model: 'gpt-x' }] },
            routing_evidence_read: { state: 'ok', observed_at: '2026-11-14T08:00:00Z' } }) } });
const marksOn = store => accountCapabilityMarks(store.snapshot, accountRows(store.snapshot)[0]).length;

test('model marks of a full read survive a newer passive roster that answered first', async () => {
    let finishFull;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => url.includes('view=quota')
        ? okResponse(passivePayload(true)) : new Promise(resolve => { finishFull = resolve; }) });
    const full = store.refresh();
    await store.observe();
    finishFull(okResponse(markedStatus())); await full; await flush();
    assert.equal(marksOn(store), 1, 'the marks were dropped with the superseded roster');
    await store.observe();
    assert.equal(marksOn(store), 1, 'a passive roster keeps full-only marks');
    store.dispose();
});

test('a failed marks read stays one dated note across passive reads; a confirmed toggle clears marks', async () => {
    let failedMarks = true, finishPatch, written = false;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async (url, init = {}) => {
        if (init.method === 'PATCH') return new Promise(resolve => { finishPatch = resolve; });
        // The confirmation's follow-up discovery is held: only marks from before the write exist.
        if (written && !url.includes('view=quota')) return new Promise(() => {});
        return okResponse(url.includes('view=quota') ? passivePayload(true) : markedStatus({ failedMarks }));
    } });
    await store.refresh(); await store.observe();
    assert.equal(capabilityMarksReadFailedAt(store.snapshot), '2026-11-14T08:00:00Z');
    failedMarks = false;
    await store.refresh(); await store.observe();
    assert.equal(marksOn(store), 1);
    const write = store.setEnabled('codex', 'one', false);
    await flush();
    written = true;
    finishPatch(okResponse({ ok: true, harness: 'codex', profile_id: 'one', enabled: false }));
    await write; await store.observe(); await flush();
    assert.equal(marksOn(store), 0, 'marks observed before the mutation remain');
    assert.equal(store.snapshot.profiles.routing_evidence_read, undefined, 'the host drops both mark fields');
    store.dispose();
});

test('a failed passive read does not discard a later-completing successful wake', async () => {
    let finishWake, passiveFails = false;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async (url, init = {}) => {
        if (init.method === 'POST') return new Promise(resolve => { finishWake = resolve; });
        if (url.includes('view=quota')) return passiveFails
            ? { ok: false, status: 503, json: async () => ({ error: 'daemon starting' }) } : okResponse(passivePayload(true));
        return okResponse(loginPolicyStatus(['codex']));
    } });
    await store.refresh(); await store.observe();
    passiveFails = true;
    const waking = store.wake();
    await flush();
    await store.observe();
    assert.equal(store.snapshot.reads.accounts, 'failed');
    finishWake(okResponse({ ...loginPolicyStatus(['codex']), profiles: { ...accountPayload(false).profiles, login_capable_harnesses: ['codex'] } }));
    await waking; await flush();
    assert.equal(store.snapshot.reads.accounts, 'ok', 'the successful wake answer was discarded');
    assert.equal(accountRows(store.snapshot)[0].enabled, false);
    store.dispose();
});

test('a failed full discovery is retried on the next visible tick, not after the discovery interval', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let clock = 1000, fullCalls = 0, failFull = true;
    t.mock.method(Date, 'now', () => clock);
    const store = createClaudexorStatusStore({ doc: fakeDoc(), fetchImpl: async url => {
        if (url.includes('view=quota')) return okResponse(passivePayload(true));
        if (url.includes('/quota/refresh')) return okResponse({ snapshots: [], absences: [] });
        fullCalls += 1;
        return failFull ? { ok: false, status: 503, json: async () => ({ error: 'discovery timed out' }) }
            : okResponse(loginPolicyStatus(['codex']));
    } });
    const off = store.subscribe(() => {}, { visible: () => true });
    await store.refresh(); await store.observe();
    assert.equal(fullCalls, 1);
    assert.deepEqual(accountRows(store.snapshot), [], 'raw rows wait for the first login policy');
    failFull = false;
    for (let tick = 0; tick < 2; tick++) { clock += 5000; t.mock.timers.tick(5000); await flush(); }
    assert.equal(fullCalls, 2, 'a failed discovery waited for the 120 s interval');
    assert.equal(accountRows(store.snapshot).length, 1);
    off(); store.dispose();
});

test('a failed full roster keeps newer passive membership and its date; a successful empty one replaces it', async () => {
    const one = { profile: { harness_id: 'codex', profile_id: 'one', enabled: true } };
    const two = { profile: { harness_id: 'codex', profile_id: 'two', enabled: true } };
    const fullAt = '2026-11-14T10:00:00.000Z', urls = [];
    let full = { ...loginPolicyStatus(['codex']), facets: { accounts: { observed_at: fullAt, stale: false, error: null } } };
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => {
        urls.push(url);
        return okResponse(url.includes('view=quota')
            ? { ...passivePayload(true), profiles: { profiles: [one, two], accountPools: [] } } : full);
    } });
    await store.refresh(); await store.observe();
    const seen = store.snapshot.facets.accounts.observed_at;
    assert.deepEqual(accountRows(store.snapshot).map(row => row.profile_id), ['one', 'two'], 'a login added "two"');
    // The server serves its stale memory of the last full roster: it never saw the passive one.
    full = { ...full, reads: { catalog: 'ok', accounts: 'failed', quota: 'ok' },
        facets: { accounts: { observed_at: fullAt, stale: true, error: 'discovery_timeout' } } };
    await store.refresh();
    assert.deepEqual(accountRows(store.snapshot).map(row => row.profile_id), ['one', 'two'],
        'the failed full read restored the older roster');
    assert.equal(store.facet(FACET_ACCOUNTS), READ_FAILED);
    assert.deepEqual(store.snapshot.facets.accounts, { observed_at: seen, stale: true, error: 'discovery_timeout' });
    assert.match(store.unavailableNote(FACET_ACCOUNTS).text, /discovery_timeout/);
    full = { ...loginPolicyStatus(['codex']), profiles: { profiles: [], accountPools: [], login_capable_harnesses: ['codex'] } };
    await store.refresh();
    assert.deepEqual(accountRows(store.snapshot), [], 'a successful empty roster is authoritative');
    assert.equal(store.facet(FACET_ACCOUNTS), READ_OK);
    assert.equal(urls.length, 4, 'no additional request');
    store.dispose();
});

test('a switch answer that did not arrive, or the host\'s 503, is unconfirmed; only a 4xx refusal says so', async () => {
    // The host answers an engine refusal and a transport loss after sending alike with 503 (CLIENT_CONTRACT).
    const answers = [() => { throw new TypeError('Failed to fetch'); },
        () => okResponse({ ok: true }),
        () => ({ ok: false, status: 504, json: async () => { throw new SyntaxError('a proxy page'); } }),
        () => ({ ok: false, status: 503, json: async () => ({ error: 'claudexor_unavailable: engine refused' }) }),
        () => ({ ok: false, status: 401, json: async () => ({ error: 'Authentication required.' }) })];
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async (url, init = {}) =>
        init.method === 'PATCH' ? answers.shift()() : okResponse(url.includes('view=quota') ? passivePayload(false)
            : accountPayload(false)) });
    await store.refresh();
    for (const [error, refused] of [['Failed to fetch', false], ['The answer did not match the request', false],
        ['HTTP 504', false], ['claudexor_unavailable: engine refused', false], ['Authentication required.', true]]) {
        assert.equal(await store.setEnabled('codex', 'one', true), null);
        const action = store.profileAction('codex', 'one');
        assert.deepEqual([action.error, Boolean(action.refused), action.busy], [error, refused, false]);
    }
    store.dispose();
});

// Review of #3378, first screen: a passive answer reads neither the catalog nor, before the
// first login policy, the account presentation, so only the first full read may end "checking".
const CHECKING = 'Checking Claudexor… the first read probes each agent CLI and can take a minute or more.';

test('a passive answer before the first full read keeps the checking state without delaying quota', async () => {
    let finishFull;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => url.includes('view=quota')
        ? okResponse(passivePayload(true)) : new Promise(resolve => { finishFull = resolve; }) });
    const full = store.refresh();
    await store.observe();
    assert.equal(store.loading, true);
    assert.equal(serviceBannerLine(store).text, CHECKING);
    assert.equal(bareRowStatusText(store.facet(FACET_ACCOUNTS)), 'Checking…');
    assert.equal(store.facet(FACET_CATALOG), READ_UNREAD);
    assert.equal(store.unavailableNote(FACET_ACCOUNTS).text, 'Reading your agent accounts…');
    assert.equal(store.quotaKnown, true, 'the passive quota waited for discovery');
    finishFull(okResponse(loginPolicyStatus(['codex']))); await full;
    assert.equal(serviceBannerLine(store).text, 'Claudexor ready');
    assert.deepEqual(accountRows(store.snapshot).map(row => row.profile_id), ['one']);
    store.dispose();
});

test('a failed first full read ends the checking state with its own failure', async () => {
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => url.includes('view=quota')
        ? okResponse(passivePayload(true)) : { ok: false, status: 503, json: async () => ({ error: 'discovery timed out' }) } });
    await store.observe();
    await store.refresh();
    assert.equal(store.loading, false);
    assert.equal(bareRowStatusText(store.facet(FACET_ACCOUNTS)), 'Not checked: the status request did not complete');
    assert.notEqual(serviceBannerLine(store).text, CHECKING);
    assert.match(serviceBannerLine(store).text, /discovery timed out/);
    store.dispose();
});

test('a failed full read never outranks a concurrent successful passive read', async () => {
    let finishPassive;
    const quotaRow = { subject: { harness: 'codex', subject_id: 'one' }, freshness: 'fresh', constraints: [] };
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => url.includes('view=quota')
        ? new Promise(resolve => { finishPassive = resolve; })
        : okResponse({ ...loginPolicyStatus(['codex']), profiles: { login_capable_harnesses: ['codex'] },
            reads: { catalog: 'ok', accounts: 'failed', quota: 'failed' } }) });
    const passive = store.observe(); // dispatched first: an older ticket
    await store.refresh(); // a cold host serves no roster memory; accounts and quota failed
    assert.deepEqual(accountRows(store.snapshot), []);
    finishPassive(okResponse({ ...passivePayload(true), quota: [quotaRow] }));
    await passive;
    assert.deepEqual(accountRows(store.snapshot).map(row => row.profile_id), ['one'],
        'the failed full roster discarded the answered passive one');
    assert.equal(store.facet(FACET_QUOTA), READ_OK, 'the failed full quota discarded the answered passive one');
    assert.deepEqual(store.snapshot.quota, [quotaRow]);
    store.dispose();
});

test('a confirmed removal is ordered like a toggle: no older read or lost follow-up restores the row', async t => {
    const row = id => ({ profile: { harness_id: 'codex', profile_id: id, enabled: true } });
    const full = rows => ({ ...loginPolicyStatus(['codex']), profiles: { profiles: rows, accountPools: [],
        login_capable_harnesses: ['codex'], routing_evidence_read: { state: 'ok', observed_at: '2026-11-14T08:00:00Z' },
        routing_evidence: { capabilities: [{ harness_id: 'codex', profile_id: 'two', requested_model: 'gpt-x' }] } } });
    const deletes = [];
    const answerDelete = async (url, init) => {
        deletes.push([url, init.method]);
        return okResponse({ removed: true, profile: { harness_id: 'codex', profile_id: 'two' } });
    };
    t.mock.method(globalThis, 'fetch', answerDelete); // a removal sent outside the store's fetchImpl
    let phase = 'first', finishOld;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async (url, init = {}) => {
        if (init.method === 'DELETE') return answerDelete(url, init);
        if (url.includes('view=quota')) return new Promise(() => {}); // isolate the full reads
        if (phase === 'first') return okResponse(full([row('one'), row('two')]));
        if (phase === 'old') return new Promise(resolve => { finishOld = resolve; });
        throw new TypeError('Failed to fetch'); // the follow-up read is lost
    } });
    await store.refresh();
    phase = 'old';
    const old = store.refresh(); // its roster is read before the DELETE
    await flush();
    phase = 'lost';
    const removal = confirmRemoveAccount('codex', 'two', { store, dialogImpl: async () => true, renderImpl: () => {} });
    await flush();
    assert.deepEqual(deletes, [['/api/claudexor/credential-profiles/codex/two', 'DELETE']]);
    finishOld(okResponse(full([row('one'), row('two')])));
    await removal; await old;
    assert.deepEqual(accountRows(store.snapshot).map(item => item.profile_id), ['one'],
        'an older read restored the removed account');
    assert.equal(accountCapabilityMarks(store.snapshot, { harness: 'codex', profile_id: 'two' }).length, 0,
        'an older read restored the removed account marks');
    store.dispose();
});

// Review of #3378, lifecycle: only full status carries daemon/runtime and model marks.
const STOPPED_STATUS = { ...accountPayload(true), daemon: { state: 'stale', runtime: { state: 'ready' } },
    reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' }, profiles: {} };

test('a held login job reads the lifecycle every tick; without a hold discovery keeps its interval', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let clock = 1000, fullReads = 0;
    t.mock.method(Date, 'now', () => clock);
    const store = createClaudexorStatusStore({ doc: fakeDoc(), fetchImpl: async url => {
        if (url.includes('view=quota')) return okResponse(passivePayload(true));
        fullReads += 1;
        return okResponse(loginPolicyStatus(['codex']));
    } });
    const follow = store.subscribe(() => {}); // the login card's phase follow: never a visible surface
    await store.refresh();
    const release = store.holdPolling('login-job');
    for (let tick = 0; tick < 3; tick++) { clock += 5000; t.mock.timers.tick(5000); await flush(); }
    assert.equal(fullReads, 4, 'a login hold left the lifecycle to the 120 s discovery');
    release();
    const visible = store.subscribe(() => {}, { visible: () => true });
    for (let tick = 0; tick < 3; tick++) { clock += 5000; t.mock.timers.tick(5000); await flush(); }
    assert.equal(fullReads, 4, 'a visible surface without a hold read discovery on every tick');
    visible(); follow(); store.dispose();
});

test('a passive answer from a daemon held as stopped reads the lifecycle at once; a quota gap does not', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let clock = 1000, fullReads = 0, daemon = 'stale', passiveQuota = 'ok';
    t.mock.method(Date, 'now', () => clock);
    const store = createClaudexorStatusStore({ doc: fakeDoc(), fetchImpl: async url => {
        if (url.includes('view=quota')) {
            return okResponse({ ...passivePayload(true), reads: { accounts: 'ok', quota: passiveQuota } });
        }
        fullReads += 1;
        return okResponse(daemon === 'stale' ? STOPPED_STATUS : loginPolicyStatus(['codex']));
    } });
    const off = store.subscribe(() => {}, { visible: () => true });
    await store.refresh();
    assert.match(serviceBannerLine(store).text, /agent daemon is not running/);
    daemon = 'running'; // a delegated run started it
    for (let tick = 0; tick < 2; tick++) { clock += 5000; t.mock.timers.tick(5000); await flush(); }
    assert.equal(fullReads, 2, 'a daemon the passive read answered stayed "not running" until discovery');
    assert.equal(serviceBannerLine(store).text, 'Claudexor ready');
    passiveQuota = 'failed'; // a quota gap of a running daemon is no lifecycle change
    for (let tick = 0; tick < 3; tick++) { clock += 5000; t.mock.timers.tick(5000); await flush(); }
    assert.equal(fullReads, 2);
    off(); store.dispose();
});

test('a confirmed toggle reads its marks again without holding the switch', async () => {
    let fullReads = 0, finishFollow;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async (url, init = {}) => {
        if (init.method === 'PATCH') return okResponse({ ok: true, harness: 'codex', profile_id: 'one', enabled: false });
        if (url.includes('view=quota')) return okResponse(passivePayload(false));
        fullReads += 1;
        return fullReads === 1 ? okResponse(markedStatus()) : new Promise(resolve => { finishFollow = resolve; });
    } });
    await store.refresh();
    assert.equal(marksOn(store), 1);
    assert.ok(await store.setEnabled('codex', 'one', false));
    await flush();
    assert.equal(store.profileAction('codex', 'one').busy, false, 'the switch waited for the follow-up read');
    assert.equal(marksOn(store), 0, 'marks observed before the mutation remain');
    assert.equal(fullReads, 2, 'no full read followed the confirmed write');
    finishFollow(okResponse(markedStatus({ enabled: false }))); await flush();
    assert.equal(marksOn(store), 1, 'marks observed after the write did not return');
    assert.equal(accountRows(store.snapshot)[0].enabled, false);
    store.dispose();
});

// Review of #3378, second panel: lifecycle transitions both ways, discarded answers, the login-policy wait,
// optional resources, the DELETE receipt and unknown switch outcomes.
const passiveOf = ({ accounts = 'ok', quota = 'ok', discovery = '' } = {}) => ({ ...passivePayload(true),
    profiles: accounts === 'ok' ? passivePayload(true).profiles : {}, reads: { catalog: 'not_read', accounts, quota },
    read_errors: discovery ? { discovery: { code: discovery } } : {} });
const UNREACHABLE_STATUS = { ...STOPPED_STATUS, daemon: { state: 'unreachable', last_error: 'daemon_not_answering: handshake timed out' } };
const NOT_RUNNING = { discovery: 'daemon_not_discovered', accounts: 'not_read', quota: 'not_read' };

function lifecycleRig(t, full, passive) {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const rig = { clock: 1000, full, passive, fullReads: 0 };
    t.mock.method(Date, 'now', () => rig.clock);
    rig.store = createClaudexorStatusStore({ doc: fakeDoc(), fetchImpl: async url => {
        if (url.includes('view=quota')) return okResponse(rig.passive);
        rig.fullReads += 1;
        return okResponse(rig.full);
    } });
    rig.off = rig.store.subscribe(() => {}, { visible: () => true });
    rig.tick = async (n = 1) => { for (let i = 0; i < n; i++) { rig.clock += 5000; t.mock.timers.tick(5000); await flush(); } };
    return rig;
}

for (const [name, before, after, passive, words, action] of [
    ['a start with a quota gap (raw accounts answered before the first policy)', STOPPED_STATUS, loginPolicyStatus(['codex']),
        passiveOf({ quota: 'failed' }), /^(?![\s\S]*(not running|was not asked))/, 'refresh'],
    ['a clean stop (passive discovery fails)', loginPolicyStatus(['codex']), STOPPED_STATUS, passiveOf(NOT_RUNNING),
        /agent daemon is not running/, 'wake'],
    ['a crash that leaves its descriptor (passive reads fail)', loginPolicyStatus(['codex']), STOPPED_STATUS,
        passiveOf({ accounts: 'failed', quota: 'failed' }), /agent daemon is not running/, 'wake'],
]) {
    test(`${name} reads the lifecycle on the passive answer, once; Refresh and Wake follow it`, async t => {
        const rig = lifecycleRig(t, before, before === STOPPED_STATUS ? passiveOf(NOT_RUNNING) : passiveOf());
        await rig.store.refresh(); await rig.tick();
        assert.equal(rig.fullReads, 1);
        rig.full = after; rig.passive = passive;
        await rig.tick();
        assert.equal(rig.fullReads, 2, 'the held lifecycle stood until the 120 s discovery');
        assert.match(serviceBannerLine(rig.store).text, words);
        assert.equal(refreshActionKind(rig.store.snapshot), action);
        await rig.tick(10);
        assert.equal(rig.fullReads, 2, 'a steady state read the lifecycle again');
        assert.match(serviceBannerLine(rig.store).text, words, 'later passive answers outranked the lifecycle line');
        assert.equal(refreshActionKind(rig.store.snapshot), action);
        rig.off(); rig.store.dispose();
    });
}

test('a lifecycle read that turns unreachable beside answering passive reads is checked once, a tick later', async t => {
    const rig = lifecycleRig(t, loginPolicyStatus(['codex']), passiveOf());
    await rig.store.refresh();
    rig.full = UNREACHABLE_STATUS;
    await rig.tick(24); // the scheduled discovery meets a transient handshake failure
    assert.equal(rig.fullReads, 2);
    await rig.tick(10);
    assert.equal(rig.fullReads, 3, 'an unreachable answer was not checked once, or was re-read repeatedly');
    rig.full = loginPolicyStatus(['codex']);
    await rig.tick(24);
    assert.equal(serviceBannerLine(rig.store).text, 'Claudexor ready');
    rig.off(); rig.store.dispose();
});

test('degraded running (passive reads failing beside a running lifecycle) reads full status once, not per tick', async t => {
    const rig = lifecycleRig(t, loginPolicyStatus(['codex']), passiveOf());
    await rig.store.refresh(); await rig.tick();
    rig.passive = passiveOf({ accounts: 'failed', quota: 'failed' });
    await rig.tick(12);
    assert.equal(rig.fullReads, 2);
    rig.off(); rig.store.dispose();
});

// Review of #3378, third panel: a held "not serving" verdict keeps quiet only what it left unread.
test('a passive answer that served nothing retires a Refresh reading even beside a lifecycle held as not serving', async () => {
    const target = { harness: 'codex', profile_id: 'one' };
    const reading = { snapshots: [{ subject: { harness: 'codex', subject_id: 'one' }, freshness: 'fresh',
        observed_at: '2026-11-14T10:00:00Z', constraints: [] }], absences: [] };
    let full = loginPolicyStatus(['codex']), passive = passiveOf();
    const store = createClaudexorStatusStore({ doc: null, storage: () => null, fetchImpl: async url =>
        okResponse(url.includes('/quota/refresh') ? reading : url.includes('view=quota') ? passive : full) });
    await store.refresh(); await store.observe();
    full = UNREACHABLE_STATUS; // a transient handshake failure: the engine still serves
    await store.refresh();
    await store.refreshResources(target);
    assert.equal(store.resourceAction(target).resourceRead, true);
    passive = passiveOf(NOT_RUNNING); // then it stops before the next lifecycle read
    await store.observe();
    assert.equal(store.resourceAction(target).resourceRead, false, 'the quiet answer hid a later failure from the reading');
    assert.match(store.unavailableNote(FACET_ACCOUNTS).text, /daemon was not asked/, 'the lifecycle read\'s unread facet still speaks');
    store.dispose();
});

test('a facet held as answered is not kept quiet: a later passive failure beside a non-serving lifecycle dates it', async () => {
    let finishFull, passive = passiveOf(), slow = false;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => {
        if (url.includes('view=quota')) return okResponse({ ...passive,
            profiles: passive.reads.accounts === 'ok' ? verifiedStatus().profiles : {} });
        return slow ? new Promise(resolve => { finishFull = resolve; }) : okResponse(verifiedStatus());
    } });
    await store.refresh();
    slow = true;
    const lifecycle = store.refresh(); // a slow lifecycle read ...
    await store.observe(); // ... and a newer passive answer lands first
    finishFull(okResponse(UNREACHABLE_STATUS)); await lifecycle; // its facets are older, its verdict is held
    assert.equal(badgeOf(store), 'Login verified');
    passive = passiveOf(NOT_RUNNING); // the engine stops: this answer agrees with the held verdict
    await store.observe();
    assert.equal(store.facet(FACET_ACCOUNTS), READ_NOT_READ, 'a later failure was kept quiet over an answered facet');
    assert.equal(store.facet(FACET_QUOTA), READ_NOT_READ);
    assert.equal(badgeOf(store), 'Login verified — last known');
    store.dispose();
});

test('an answer the host discarded keeps the held facet; a daemon nobody asked still says so', async () => {
    let passive = passiveOf(), full = loginPolicyStatus(['codex']);
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url =>
        okResponse(url.includes('view=quota') ? passive : full) });
    await store.refresh(); await store.observe();
    // Another window confirmed a PATCH/DELETE while this read crossed it (claudexor_accounts.coherent_accounts).
    passive = { ...passiveOf(), profiles: {}, unified_accounts: false, reads: { catalog: 'not_read', accounts: 'not_read', quota: 'ok' } };
    await store.observe();
    assert.equal(store.facet(FACET_ACCOUNTS), READ_OK);
    assert.equal(accountRows(store.snapshot).length, 1);
    assert.equal(store.unavailableNote(FACET_ACCOUNTS), null);
    // A full read discarded as older than a newer held read serves that memory under `not_read`.
    full = { ...full, reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } };
    await store.refresh();
    for (const name of [FACET_CATALOG, FACET_ACCOUNTS, FACET_QUOTA]) assert.equal(store.facet(name), READ_OK, name);
    full = STOPPED_STATUS; passive = passiveOf(NOT_RUNNING);
    await store.refresh(); await store.observe();
    assert.match(store.unavailableNote(FACET_ACCOUNTS).text, /daemon was not asked/);
    store.dispose();
});

test('raw rows waiting for the login policy speak for the full read that supplies it, never "not asked"', async () => {
    let finishFull, fail = true;
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => {
        if (url.includes('view=quota')) return okResponse(passiveOf());
        if (fail) return { ok: false, status: 503, json: async () => ({ error: 'discovery timed out' }) };
        return new Promise(resolve => { finishFull = resolve; });
    } });
    await store.refresh(); await store.observe(); await store.observe(); // the passive answers are newer
    assert.equal(store.facet(FACET_ACCOUNTS), READ_TRANSPORT);
    assert.match(store.unavailableNote(FACET_ACCOUNTS).text, /^Could not read your agent accounts \(discovery timed out\)/);
    fail = false;
    const read = store.refresh();
    assert.equal(store.facet(FACET_ACCOUNTS), READ_TRANSPORT, 'a retry after a failure is no endless "Reading"');
    finishFull(okResponse(STOPPED_STATUS)); await read; await store.observe();
    assert.equal(store.facet(FACET_ACCOUNTS), READ_FAILED, 'an answer without the policy did not read the rows');
    assert.doesNotMatch(store.unavailableNote(FACET_ACCOUNTS).text, /was not asked/);
    const again = store.refresh();
    assert.equal(store.unavailableNote(FACET_ACCOUNTS).text, 'Reading your agent accounts…');
    finishFull(okResponse(loginPolicyStatus(['codex']))); await again;
    assert.equal(store.facet(FACET_ACCOUNTS), READ_OK);
    store.dispose();
});

test('a slower full read keeps resources a newer passive answer did not observe; an explicit empty one wins', async () => {
    const resource = [{ target: { harness: 'codex', profile_id: 'one' }, balances: { observed_at: '2026-11-14T10:00:00Z' } }];
    for (const [passive, expected] of [[{}, resource], [{ resources: [] }, []]]) {
        let finishFull;
        const body = { ...passiveOf(), ...passive }; delete body.resources;
        if ('resources' in passive) body.resources = passive.resources;
        const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => url.includes('view=quota')
            ? okResponse(body) : new Promise(resolve => { finishFull = resolve; }) });
        const full = store.refresh(); // older ticket: the legacy fallback reads resources here only
        await store.observe();
        finishFull(okResponse({ ...loginPolicyStatus(['codex']), resources: resource })); await full;
        assert.deepEqual(store.snapshot.resources, expected);
        store.dispose();
    }
});

test('a full answer that could not negotiate resources neither erases nor fences them', async () => {
    // Accepted synthetic order: it pins that a read which observed no resources dates none (one rule, host and browser).
    const list = at => [{ target: { harness: 'codex', profile_id: 'one' }, balances: { observed_at: at } }];
    let finishPassive, full = { ...loginPolicyStatus(['codex']), resources: list('2026-11-14T09:00:00Z'),
        resource_capabilities: { read: true }, resource_capabilities_read: 'ok' };
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => url.includes('view=quota')
        ? new Promise(resolve => { finishPassive = resolve; }) : okResponse(full) });
    await store.refresh();
    const passive = store.observe(); // dispatched before the blind read, answered after it
    full = { ...loginPolicyStatus(['codex']), resource_capabilities: { read: false }, resource_capabilities_read: 'failed' };
    await store.refresh(); // quota answered, operations catalog unread: no list could be negotiated
    assert.deepEqual(store.snapshot.resources, list('2026-11-14T09:00:00Z'), 'the unread catalog erased the held list');
    assert.deepEqual(store.snapshot.resource_capabilities, { read: true }, 'the held capabilities stand in');
    finishPassive(okResponse({ ...passiveOf(), resources: list('2026-11-14T10:00:00Z') })); await passive;
    assert.deepEqual(store.snapshot.resources, list('2026-11-14T10:00:00Z'), 'a read that observed no resources fenced an older observation of them');
    // A negotiated absence is an observation: it removes the list and outranks older answers.
    full = { ...loginPolicyStatus(['codex']), resource_capabilities: { read: false }, resource_capabilities_read: 'ok' };
    await store.refresh();
    assert.equal(store.snapshot.resources, undefined);
    store.dispose();
});

test('resources the merge accepted stand when the held quota facet had failed: one owner decides them', async () => {
    const resource = [{ target: { harness: 'codex', profile_id: 'one' }, balances: { observed_at: '2026-11-14T10:00:00Z' } }];
    let finishFull, quota = 'ok';
    const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url => {
        if (!url.includes('view=quota')) return new Promise(resolve => { finishFull = resolve; });
        const body = passiveOf({ quota }); delete body.resources;
        return okResponse(body);
    } });
    const full = store.refresh(); // older ticket: the legacy fallback reads resources here only
    await store.observe();
    quota = 'failed';
    await store.observe(); // the newer quota failure is what the facet says
    finishFull(okResponse({ ...loginPolicyStatus(['codex']), resources: resource })); await full;
    assert.equal(store.facet(FACET_QUOTA), READ_FAILED, 'the older full answer outranked the newer failure');
    assert.deepEqual(store.snapshot.resources, resource, 'the quota stand-in overrode the resources the merge had accepted');
    store.dispose();
});

test('the list the host leaves beside a quota answer it discarded is owned in ticket order; a failed read\'s is not', async () => {
    const own = [{ target: { harness: 'codex', profile_id: 'one' }, balances: { observed_at: '2026-11-14T10:00:00Z' } }];
    const rows = [{ subject: { harness: 'codex', subject_id: 'one' }, freshness: 'fresh', constraints: [] }];
    const passive = { ...passiveOf(), quota: rows }; delete passive.resources; // the legacy passive view carries none
    for (const first of [false, true]) {
        let full = { ...loginPolicyStatus(['codex']), quota: rows };
        const store = createClaudexorStatusStore({ doc: null, fetchImpl: async url =>
            okResponse(url.includes('view=quota') ? passive : full) });
        if (!first) { await store.refresh(); await store.observe(); }
        // Its rows are older than a newer held read that could not negotiate resources (claudexor_accounts._with_facet_memory).
        full = { ...full, quota: [{ ...rows[0], freshness: 'stale' }], resources: own,
            reads: { catalog: 'ok', accounts: 'ok', quota: 'not_read' } };
        await store.refresh();
        assert.deepEqual(store.snapshot.resources, own, 'the read\'s own resources were discarded with its rows');
        if (!first) assert.deepEqual(store.snapshot.quota, rows, 'the discarded rows replaced the held ones');
        assert.equal(store.facet(FACET_QUOTA), first ? READ_NOT_READ : READ_OK);
        await store.observe();
        assert.deepEqual(store.snapshot.resources, own, 'a passive answer without resources observed none');
        assert.equal(store.facet(FACET_QUOTA), READ_OK);
        // A read that failed serves the host's memory of any age: never an observation, an empty list included.
        full = { ...full, resources: [], reads: { catalog: 'ok', accounts: 'ok', quota: 'failed' } };
        await store.refresh();
        assert.deepEqual(store.snapshot.resources, own);
        // A discarded answer without the list keeps what is held; an explicit empty one beside it is owned.
        full = { ...full, reads: { catalog: 'ok', accounts: 'ok', quota: 'not_read' } }; delete full.resources;
        await store.refresh();
        assert.deepEqual(store.snapshot.resources, own);
        full = { ...full, resources: [] };
        await store.refresh();
        assert.deepEqual(store.snapshot.resources, []);
        store.dispose();
    }
});

// Evidence order (review of #3378, third panel): a fence orders merges and observes nothing.
const lostStatus = () => ({ ok: false, status: 503, json: async () => ({ error: 'discovery timed out' }) });
const verifiedStatus = () => { const status = loginPolicyStatus(['codex']);
    status.profiles.profiles[0].status = { verification: 'passed', verification_source: 'vendor' }; return status; };
const badgeOf = store => accountRowFacts(accountRows(store.snapshot)[0], store.snapshot,
    { accountsRead: store.facet(FACET_ACCOUNTS) }).badge.label;

for (const [name, answer] of [
    ['refused', () => ({ ok: false, status: 401, json: async () => ({ error: 'Authentication required.' }) })],
    ['unanswered', () => { throw new TypeError('Failed to fetch'); }],
    ['confirmed', () => okResponse({ ok: true, harness: 'codex', profile_id: 'one', enabled: false })]]) {
    test(`a ${name} row write is no roster read: after a lost status request the rows stay last known until a read answers`, async () => {
        let phase = 'answer', finishPassive;
        const store = createClaudexorStatusStore({ doc: null, fetchImpl: async (url, init = {}) => {
            if (init.method === 'PATCH') return answer();
            if (url.includes('view=quota')) return new Promise(resolve => { finishPassive = resolve; });
            return phase === 'answer' ? okResponse(verifiedStatus()) : phase === 'lost' ? lostStatus() : new Promise(() => {});
        } });
        await store.refresh();
        assert.equal(badgeOf(store), 'Login verified');
        phase = 'lost';
        await store.refresh();
        assert.equal(badgeOf(store), 'Login verified — last known');
        phase = 'held'; // the write's own follow-up discovery never answers here
        await store.setEnabled('codex', 'one', false);
        await flush();
        assert.equal(store.profileAction('codex', 'one').busy, false);
        assert.equal(store.facet(FACET_ACCOUNTS), READ_TRANSPORT, 'the write was taken for a roster read');
        assert.equal(badgeOf(store), 'Login verified — last known');
        assert.equal(accountRows(store.snapshot)[0].enabled, name !== 'confirmed', 'the confirmation still edits its row');
        // The read begun after the write is the observation that ends it.
        const roster = verifiedStatus().profiles;
        roster.profiles[0].profile.enabled = name !== 'confirmed';
        finishPassive(okResponse({ ...passivePayload(true), profiles: roster })); await flush();
        assert.equal(store.facet(FACET_ACCOUNTS), READ_OK);
        assert.equal(badgeOf(store), 'Login verified');
        store.dispose();
    });
}

test('a 200 DELETE that is not the published receipt is unconfirmed: the row stays, nothing is resent', async () => {
    const removed = { harness_id: 'codex', profile_id: 'one', display_name: 'One' };
    for (const [answer, confirmed] of [
        [{ ok: true, status: 200, json: async () => { throw new SyntaxError('proxy page'); } }, false],
        [okResponse({}), false],
        [okResponse({ removed: true, profile: { ...removed, profile_id: 'two' }, credentialCleanup: 'none' }), false],
        [okResponse({ removed: true, profile: removed, credentialCleanup: 'none' }), true],
    ]) {
        const deletes = [];
        const store = createClaudexorStatusStore({ doc: null, fetchImpl: async (url, init = {}) => {
            if (init.method === 'DELETE') { deletes.push(url); return answer; }
            return url.includes('view=quota') ? new Promise(() => {}) : okResponse(loginPolicyStatus(['codex']));
        } });
        await store.refresh();
        const outcome = await store.removeAccount('codex', 'one').then(() => 'confirmed', error => error);
        assert.equal(outcome === 'confirmed', confirmed);
        if (!confirmed) assert.deepEqual([outcome.message, Boolean(outcome.refused)], ['The answer did not match the request', false]);
        assert.equal(accountRows(store.snapshot).length, confirmed ? 0 : 1);
        assert.equal(deletes.length, 1);
        store.dispose();
    }
});

test('an unknown switch outcome reads its marks again without holding the switch; a refusal does not', async () => {
    for (const [answer, reads] of [
        [() => ({ ok: false, status: 503, json: async () => ({ error: 'daemon_unreachable: kept' }) }), 2],
        [() => { throw new TypeError('Failed to fetch'); }, 2],
        [() => ({ ok: false, status: 401, json: async () => ({ error: 'Authentication required.' }) }), 1],
    ]) {
        let fullReads = 0, held = false, patches = 0;
        const store = createClaudexorStatusStore({ doc: null, fetchImpl: async (url, init = {}) => {
            if (init.method === 'PATCH') { patches += 1; return answer(); }
            if (url.includes('view=quota')) return new Promise(() => {});
            fullReads += 1;
            return held ? new Promise(() => {}) : okResponse(markedStatus());
        } });
        await store.refresh();
        held = true;
        assert.equal(await store.setEnabled('codex', 'one', false), null);
        await flush();
        assert.equal(store.profileAction('codex', 'one').busy, false, 'the switch waited for the read');
        assert.deepEqual([fullReads, patches], [reads, 1]);
        store.dispose();
    }
});

test('a successful wake is the lifecycle read: the next ticks add no confirming discovery', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let clock = 1000, discoveries = 0, passive = passiveOf(NOT_RUNNING);
    t.mock.method(Date, 'now', () => clock);
    const store = createClaudexorStatusStore({ doc: fakeDoc(), fetchImpl: async (url, init = {}) => {
        if (init.method === 'POST') return okResponse(loginPolicyStatus(['codex'])); // the owner's Check accounts
        if (url.includes('view=quota')) return okResponse(passive);
        discoveries += 1;
        return okResponse(STOPPED_STATUS);
    } });
    const off = store.subscribe(() => {}, { visible: () => true });
    await store.refresh();
    for (let tick = 0; tick < 2; tick++) { clock += 5000; t.mock.timers.tick(5000); await flush(); }
    assert.equal((await store.wake()).ok, true);
    passive = passiveOf();
    for (let tick = 0; tick < 6; tick++) { clock += 5000; t.mock.timers.tick(5000); await flush(); }
    assert.equal(discoveries, 1, 'the passive answers from before the wake asked for another discovery');
    assert.equal(serviceBannerLine(store).text, 'Claudexor ready');
    off(); store.dispose();
});

// A stalled discovery and fast observations share a virtual clock; either transport may ignore abort.
function boundedRig(t, answer) {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const rig = { clock: 1000, fullReads: 0, passiveReads: 0 };
    t.mock.method(Date, 'now', () => rig.clock);
    rig.store = createClaudexorStatusStore({ doc: fakeDoc(), readLifetimeMs: 60000,
        fetchImpl: async (url) => {
            const passive = url.includes('view=quota');
            if (passive) rig.passiveReads += 1;
            else rig.fullReads += 1;
            return answer({ passive, full: rig.fullReads, passiveN: rig.passiveReads });
        } });
    rig.off = rig.store.subscribe(() => {}, { visible: () => true });
    rig.tick = async (n = 1) => { for (let i = 0; i < n; i += 1) { rig.clock += 5000; t.mock.timers.tick(5000); await flush(); } };
    return rig;
}

for (const hang of ['fetch', 'body']) {
    test(`a status ${hang} that never answers settles at its lifetime: the held snapshot stands and the next refresh goes through`, async t => {
        let phase = 'answer';
        const held = loginPolicyStatus(['codex']);
        const rig = boundedRig(t, ({ passive }) => passive ? okResponse(passivePayload(false))
            : phase === 'answer' ? okResponse(held)
                : hang === 'fetch' ? new Promise(() => {}) : { ok: true, status: 200, json: () => new Promise(() => {}) });
        await rig.store.refresh();
        const kept = rig.store.snapshot;
        phase = 'hang';
        const stalling = rig.store.refresh();
        await rig.tick(12);                       // the 60s lifetime runs out
        await stalling;
        assert.deepEqual([rig.store.loading, rig.store.error], [false, 'timed out']);
        assert.deepEqual(rig.store.snapshot.daemon, kept.daemon, 'the failed full read leaves the last lifecycle intact');
        assert.deepEqual(rig.store.snapshot.harnesses, kept.harnesses);
        assert.equal(accountRows(rig.store.snapshot)[0].enabled, false, 'newer passive account data survives');
        phase = 'answer';
        await rig.tick();
        assert.deepEqual([rig.fullReads, rig.store.error], [3, ''], 'the next visible tick retries a failed read');
        rig.off(); rig.store.dispose();
    });
}

test('a queued follow-up demand survives the timed-out read and lands once', async t => {
    const rig = boundedRig(t, ({ passive, full }) => passive ? okResponse(passivePayload(true))
        : full === 1 ? new Promise(() => {}) : okResponse(loginPolicyStatus(['codex'])));
    const stalling = rig.store.refresh();
    const demand = rig.store.refresh({ afterCurrent: true });      // one queued follow-up read
    await rig.tick(12);
    await stalling;
    assert.equal(rig.fullReads, 2, 'the queued follow-up read lands once the stalled read settles');
    await demand;
    assert.equal(rig.fullReads, 2, 'and exactly once');
    rig.off(); rig.store.dispose();
});

test('an unresolved disagreement is not an answered observation when its full read times out', async t => {
    let passive = passiveOf(), phase = 'answer';
    const rig = boundedRig(t, ({ passive: isPassive }) => isPassive ? okResponse(passive)
        : phase === 'answer' ? okResponse(loginPolicyStatus(['codex'])) : new Promise(() => {}));
    await rig.store.refresh();                    // the lifecycle read: running and serving
    await rig.store.observe();                    // the passive reads agree
    phase = 'hang';
    passive = passiveOf(NOT_RUNNING);             // the passive verdict flips: it serves no more
    await rig.store.observe();                    // the flip asks for its one lifecycle check
    assert.equal(rig.fullReads, 2);
    await rig.tick(12);                           // the check runs out of its lifetime
    assert.equal(rig.store.error, 'timed out');
    phase = 'answer';
    await rig.tick();
    assert.equal(rig.fullReads, 3, 'the pending disagreement can be checked again');
    rig.off(); rig.store.dispose();
});

test('an answer that keeps coming after its lifetime, or after disposal, is never committed', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    t.mock.method(Date, 'now', () => 1000);
    const held = loginPolicyStatus(['codex']);
    const late = [];
    let n = 0;
    const store = createClaudexorStatusStore({ doc: fakeDoc(), readLifetimeMs: 60000,
        fetchImpl: async () => { n += 1; return n === 1 ? okResponse(held) : new Promise((resolve) => { late.push(resolve); }); } });
    // Explicit reads only: every item in `late` belongs to the full read being tested.
    await store.refresh();
    const kept = store.snapshot;
    const stalled = store.refresh();
    t.mock.timers.tick(60000); await flush();
    await stalled;
    late[0](okResponse({ ...held, daemon: { state: 'stale' } }));
    await flush();
    assert.equal(store.snapshot, kept, 'the answer that outran its lifetime is rejected');
    const stalledAgain = store.refresh();         // a second stall, disposed this time
    store.dispose();
    await stalledAgain;
    late[1](okResponse({ ...held, daemon: { state: 'stale' } }));
    await flush();
    assert.equal(store.snapshot, kept, 'and a disposed store commits nothing');
});

test('a passive GET that never answers settles at its lifetime and the next observation reads anew', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    t.mock.method(Date, 'now', () => 1000);
    let passiveN = 0;
    const store = createClaudexorStatusStore({ doc: fakeDoc(), readLifetimeMs: 60000,
        fetchImpl: async (url) => {
            if (!url.includes('view=quota')) return okResponse(loginPolicyStatus(['codex']));
            passiveN += 1;
            return passiveN === 1 ? new Promise(() => {}) : okResponse(passiveOf());
        } });
    const off = store.subscribe(() => {}, { visible: () => true });
    await store.refresh();
    const stalling = store.observe();
    t.mock.timers.tick(60000); await flush();
    await stalling;
    assert.equal(passiveN, 1);
    await store.observe();
    assert.equal(passiveN, 2, 'the single-flight slot freed: the next observation reads anew');
    off(); store.dispose();
});
