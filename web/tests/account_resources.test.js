import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { accountResourcesMarkup, confirmAccountReset, resourceAmount, resourceFacetNote,
    resourceReadGap, resetOutcome, resourceSummary } from '../modules/account_resources.js';
import { createClaudexorStatusStore, mergeAccountResources } from '../modules/claudexor_status_store.js';
import { quotaSummary } from '../modules/harness_accounts.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/account_resources.json', import.meta.url)));
const target = fixture.receipt.request.target;
const row = { ...target, kind: 'profile', enabled: false, identity: { display_name: 'Personal' } };
const snapshot = () => ({ daemon: { state: 'running' }, harnesses: [], profiles: {},
    reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' },
    quota: structuredClone(fixture.quota.snapshots), quota_absences: [],
    resources: structuredClone(fixture.quota.resources),
    resource_capabilities: { read: true, refresh: true, reset: true, inspect_reset: true } });
const response = data => ({ ok: true, json: async () => structuredClone(data) });
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { resolve, promise }; };

test('unknown quota is not zero, including a missing field; explicit zero remains visible', () => {
    for (const used of [null, undefined, '', false]) {
        const windows = structuredClone(fixture.quota.snapshots);
        windows[0].constraints = [{ used_ratio: used }];
        assert.equal(quotaSummary(windows, target.harness, target.profile_id).label, 'Usage unavailable');
    }
    const windows = structuredClone(fixture.quota.snapshots);
    windows[0].constraints = [{ used_ratio: 0 }];
    assert.match(quotaSummary(windows, target.harness, target.profile_id).label, /^0% used/);
});

test('amount strings, unknown limits and independently stale balance survive the rendered disclosure', () => {
    assert.equal(resourceAmount('123456789012345678.0010', { unit: 'credits' }), '123456789012345678.0010 credits');
    assert.equal(resourceAmount('2460', { unit: 'cents', currency: 'USD' }), '2460 cents (USD)');
    assert.equal(resourceAmount('1240', { unit: 'minor', currency: 'USD', decimal_places: 2 }), '12.40 USD');
    assert.equal(resourceAmount('1', { unit: 'minor', currency: 'KWD', decimal_places: 3 }), '0.001 KWD');
    assert.equal(resourceAmount('1234567890123456780010', { unit: 'minor', currency: 'USD', decimal_places: 2 }), '12345678901234567800.10 USD');
    assert.equal(resourceAmount('1240', { unit: 'minor', currency: 'USD' }), '1240 minor (USD)');
    assert.equal(resourceAmount(null, { unit: 'USD' }), 'Not reported');
    const html = accountResourcesMarkup(row, snapshot());
    assert.match(html, /24.60 USD/);
    assert.match(html, /0.00 USD/);
    assert.match(html, /Last known.*Last check failed/);
    assert.match(html, /<details[^>]*>.*Source details[\s\S]*balance_read_failed/);
    assert.match(html, /Weekly limits still apply/);
    assert.match(html, /data-resource-grant="grant-one"/);
    assert.match(html, /Disabled for automatic routing; account management remains available/);
    assert.match(html, /data-resource-reset="opaque-refill"/);
    assert.match(html, /Available now · count not reported/);
    assert.match(resourceFacetNote(null), /Not reported/);
});

test('known outcomes without effect preserve prior usage and availability', () => {
    for (const outcome of ['nothing_to_reset', 'no_credit', 'not_eligible', 'cooldown', 'unavailable']) {
        const action = { request: fixture.receipt.request, receipt: { ...fixture.receipt, outcome } };
        assert.equal(resourceReadGap(action), false, outcome);
        const html = accountResourcesMarkup(row, snapshot(), { action });
        assert.doesNotMatch(html, /Last known · 100% used/);
        assert.match(resourceSummary(row, snapshot(), { action }), /1 available/);
    }
});

test('normal copy uses engine descriptions and keeps native provenance in optional details', () => {
    const payload = snapshot();
    const offer = payload.resources[0].resets.value[0];
    offer.grants[0].description = 'Restores the five-hour and weekly included limits.';
    offer.grants[0].clears = ['five_hour', 'seven_day'];
    payload.resources[0].diagnostics.value = [{ code: 'can_refill', detail: 'true' },
        { code: 'account_state', detail: 'Included usage is exhausted.' }];
    const [normal, details] = accountResourcesMarkup(row, payload).split('<details');
    assert.match(normal, /Restores the five-hour and weekly included limits/);
    assert.match(normal, /Included usage is exhausted/);
    assert.doesNotMatch(normal, /provider_fixture|five_hour|seven_day|>true<|2026-10-09T/);
    assert.match(details, /can_refill/);
    assert.match(details, /five_hour, seven_day/);
});

test('offer count is authoritative even when grant details are capped or absent', () => {
    const payload = snapshot();
    const offer = payload.resources[0].resets.value[0];
    offer.available_count = 8;
    offer.grants = null;
    assert.match(accountResourcesMarkup(row, payload), /8 available/);
    assert.match(resourceSummary(row, payload), /8 available/);
    payload.resources[0].resets.freshness = 'stale';
    assert.equal(resourceSummary(row, payload), '');
    // Stale facts do not prohibit an explicit known operation.
    assert.match(accountResourcesMarkup(row, payload), /data-resource-reset="opaque-grant"/);
});

test('confirmed unavailable grants are disabled independently of refill; stale or missing counts permit explicit action', () => {
    const payload = snapshot();
    const offer = payload.resources[0].resets.value[0];
    for (const unavailable of [{ available_count: 0 }, { usable_now: false }, { eligible: false }]) {
        Object.assign(offer, { available_count: 1, usable_now: true, eligible: true }, unavailable);
        const html = accountResourcesMarkup(row, payload);
        assert.match(html, /data-resource-reset="opaque-grant"[^>]* disabled/);
        assert.match(html, /data-resource-reset="opaque-refill" aria-disabled="false"/);
        assert.match(html, /Refresh to check again/);
    }
    payload.resources[0].resets.freshness = 'stale';
    assert.doesNotMatch(accountResourcesMarkup(row, payload), / disabled/);
    Object.assign(offer, { available_count: null, usable_now: null, eligible: null, grants: null });
    payload.resources[0].resets.freshness = 'fresh';
    assert.match(accountResourcesMarkup(row, payload), /data-resource-reset="opaque-grant" aria-disabled="false"/);
});

test('known usability does not become unknown when the count is missing', () => {
    const payload = snapshot();
    const offer = payload.resources[0].resets.value[0];
    offer.available_count = null;
    offer.grants[0].available_count = null;
    const html = accountResourcesMarkup(row, payload);
    assert.match(html, /Included usage reset · Available now · count not reported/);
    assert.doesNotMatch(html, /Availability not reported/);
    offer.usable_now = null;
    assert.match(accountResourcesMarkup(row, payload), /Availability not reported/);
});

test('reset applied and failed readback remain separate; already_used is not own success', () => {
    for (const outcome of ['reset', 'already_redeemed']) {
        const action = { request: fixture.receipt.request, receipt: { ...fixture.receipt, outcome } };
        assert.equal(resetOutcome(action).title, 'Reset applied');
        assert.equal(resourceReadGap(action), true);
        const html = accountResourcesMarkup(row, snapshot(), { action });
        assert.match(html, /Last known · 100% used/);
        assert.match(html, /Last reported: 1 available · current availability unknown/);
        assert.match(html, /Use another reset/);
        assert.doesNotMatch(html, /Last reported: 0 available/);
    }
    const unconfirmed = { request: fixture.receipt.request, receipt: { ...fixture.receipt, outcome: 'already_used' } };
    assert.equal(resetOutcome(unconfirmed).tone, 'warn');
    assert.match(resetOutcome(unconfirmed).body, /does not confirm/);
});

test('a refresh or reset envelope replaces only the addressed subject, including named default', () => {
    const payload = snapshot();
    const other = { ...payload.quota[0], subject: { ...payload.quota[0].subject, subject_id: 'other' } };
    const resources = { ...payload.resources[0], target: { ...target, profile_id: 'other' } };
    payload.quota.push(other);
    payload.resources.push(resources);
    const incoming = structuredClone(fixture.quota);
    incoming.snapshots[0].constraints[0].used_ratio = 0;
    const merged = mergeAccountResources(payload, incoming, target);
    assert.equal(merged.quota[0], other);
    assert.equal(merged.resources[0], resources);
    assert.equal(merged.quota[1].constraints[0].used_ratio, 0);
});

test('lost POST response retains exact key/body through remount; recovery spends no new logical operation', async () => {
    const values = new Map();
    const storage = () => ({ getItem: key => values.get(key), setItem: (key, value) => values.set(key, value) });
    const calls = [];
    let lose = true;
    const fetchImpl = async (url, init) => {
        if (url.includes('/status')) return response(snapshot());
        calls.push({ url, ...init });
        if (lose) { lose = false; throw new Error('HTTP timeout'); }
        return response(fixture.receipt);
    };
    let store = createClaudexorStatusStore({ fetchImpl, storage, doc: null });
    await store.refresh();
    await store.resetAccount(fixture.receipt.request, { key: 'original-logical-key' });
    assert.equal(store.resourceAction(target).receipt, undefined);
    assert.equal(store.resourceAction(target).key, 'original-logical-key');
    store.dispose();
    store = createClaudexorStatusStore({ fetchImpl, storage, doc: null });
    await store.refresh();
    await store.resetAccount(fixture.receipt.request, { recover: true });
    assert.deepEqual(calls[0], calls[1]);
    assert.equal(calls[1].headers['Idempotency-Key'], 'original-logical-key');
    assert.deepEqual(JSON.parse(calls[1].body), fixture.receipt.request);
    assert.equal(JSON.parse(values.get('ouroboros.account-reset-requests'))[0].receipt.resources, null);
    const refreshes = [];
    store.dispose();
    store = createClaudexorStatusStore({ storage, doc: null, fetchImpl: async (url, init) => {
        if (url.includes('/status')) return response(snapshot());
        refreshes.push([url, JSON.parse(init.body)]);
        return response(fixture.quota);
    } });
    await store.refresh();
    await store.refreshResources(target);
    assert.deepEqual(refreshes, [['/api/claudexor/quota/refresh', { target }]]);
    assert.equal(store.resourceAction(target).receipt.outcome, 'reset');
    assert.deepEqual(JSON.parse(values.get('ouroboros.account-reset-requests')), []);
    store.dispose();
});

test('running receipts use their read URL; completed unknown recovers with original key', async () => {
    const calls = [];
    const receipt = { ...fixture.receipt, state: 'running', outcome: 'pending' };
    const store = createClaudexorStatusStore({ storage: () => null, doc: null, fetchImpl: async (url, init) => {
        if (url.includes('/status')) return response(snapshot());
        calls.push([url, init]);
        return response(receipt);
    } });
    await store.refresh();
    await store.resetAccount(receipt.request, { key: 'once' });
    receipt.state = 'completed'; receipt.outcome = 'unknown';
    await store.resetAccount(receipt.request, { recover: true });
    assert.equal(calls[1][0], '/api/claudexor/account-resets/reset-fixture');
    await store.resetAccount(receipt.request, { recover: true });
    assert.equal(calls[2][1].headers['Idempotency-Key'], 'once');
    store.dispose();
});

test('one shared confirmation authorizes new reset, cancel creates no key or operation', async () => {
    const calls = [];
    const store = { resourceAction: () => ({ request: fixture.receipt.request, receipt: fixture.receipt }),
        resetAccount: async request => calls.push(request) };
    const offer = fixture.quota.resources[0].resets.value[1];
    for (const answer of [false, null, { confirmed: true }]) {
        await confirmAccountReset(row, offer, null, { store, dialogImpl: async () => answer });
    }
    assert.deepEqual(calls, []);
    let count = 0;
    await confirmAccountReset(row, offer, null, { store, dialogImpl: async options => {
        count += 1;
        assert.match(options.body, /separate request.*additional reset/);
        assert.match(options.body, /Weekly limits still apply/);
        assert.equal(options.confirmLabel, 'Refill session again');
        assert.equal(options.title, 'Refill session again for claude-default?');
        return true;
    } });
    assert.equal(count, 1);
    assert.deepEqual(calls, [{ target, offer_id: 'opaque-refill' }]);
});

for (const nativeUuid of [true, false]) {
    for (const offer of fixture.quota.resources[0].resets.value) {
        test(`confirmed ${offer.kind} retains its generated key through lost-reply recovery (native UUID: ${nativeUuid})`, async () => {
            const priorCrypto = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
            let generated = 0;
            Object.defineProperty(globalThis, 'crypto', { configurable: true,
                value: nativeUuid ? { randomUUID: () => `native-request-${++generated}` } : {} });
            const values = new Map(), calls = [];
            const storage = () => ({ getItem: key => values.get(key), setItem: (key, value) => values.set(key, value) });
            const grant = offer.grants?.[0] || null;
            const request = { target, offer_id: offer.id, ...(grant ? { grant_id: grant.id } : {}) };
            const fetchImpl = async (url, init) => {
                const key = init.headers['Idempotency-Key'];
                assert.ok(key);
                assert.ok(JSON.parse(values.get('ouroboros.account-reset-requests'))
                    .some(entry => entry.key === key && JSON.stringify(entry.request) === init.body),
                'the original key and body must be saved before transport');
                calls.push({ url, key, body: JSON.parse(init.body) });
                if (calls.length === 1) throw new Error('lost response');
                return response({ ...fixture.receipt, request, outcome: 'no_credit', resources: null });
            };
            let store = createClaudexorStatusStore({ doc: null, storage, fetchImpl });
            try {
                await confirmAccountReset(row, offer, grant, { store, dialogImpl: async () => false });
                assert.equal(calls.length, 0);
                assert.equal(values.size, 0);
                assert.equal(generated, 0);
                let confirmations = 0;
                const confirm = () => { confirmations += 1; return true; };
                await confirmAccountReset(row, offer, grant, { store, dialogImpl: confirm });
                assert.equal(confirmations, 1);
                assert.deepEqual(calls[0].body, request);
                const original = calls[0];
                assert.equal(store.resourceAction(target).key, original.key);
                store.dispose();
                store = createClaudexorStatusStore({ doc: null, storage, fetchImpl });
                assert.equal(calls.length, 1, 'reopening must not resend a reset');
                await store.resetAccount(request, { recover: true });
                assert.deepEqual(calls[1], original);
                assert.equal(generated, nativeUuid ? 1 : 0, 'recovery never mints a new key');
                await confirmAccountReset(row, offer, grant, { store, dialogImpl: confirm });
                assert.equal(confirmations, 2);
                assert.notEqual(calls[2].key, original.key, 'a deliberate new intent has its own key');
                assert.deepEqual(calls[2].body, request);
            } finally {
                store.dispose();
                if (priorCrypto) Object.defineProperty(globalThis, 'crypto', priorCrypto);
                else delete globalThis.crypto;
            }
        });
    }
}

const atTime = (at, used, count) => {
    const envelope = structuredClone(fixture.quota);
    for (const quota of envelope.snapshots) {
        quota.observed_at = at;
        quota.constraints[0].used_ratio = used;
    }
    for (const resource of envelope.resources) {
        for (const name of ['balances', 'spending', 'resets', 'diagnostics']) {
            resource[name].observed_at = resource[name].last_attempt_at = at;
        }
        resource.resets.value[0].available_count = count;
    }
    return envelope;
};

for (const heldAbsence of [false, true]) {
    for (const receiptAbsence of [false, true]) {
        test(`receipt replay retains newer quota/absence and refresh acknowledgement (${heldAbsence}/${receiptAbsence})`, async () => {
            const older = atTime('2026-10-09T10:00:00Z', 0.9, 1);
            const newer = atTime('2026-10-09T12:00:00Z', 0.4, 0);
            const absent = envelope => {
                envelope.absences = [{ subject: envelope.snapshots[0].subject,
                    observed_at: envelope.snapshots[0].observed_at, reason: 'source_unavailable' }];
                envelope.snapshots = [];
            };
            if (heldAbsence) absent(newer);
            if (receiptAbsence) absent(older);
            const payload = snapshot();
            const sibling = { ...payload.quota[0], subject: { ...payload.quota[0].subject, subject_id: 'sibling' } };
            const siblingResource = { ...payload.resources[0], target: { ...target, profile_id: 'sibling' } };
            payload.quota.push(sibling);
            payload.resources.push(siblingResource);
            const receipt = { ...fixture.receipt, outcome: 'already_used', resources: older };
            const store = createClaudexorStatusStore({ doc: null, storage: () => null,
                fetchImpl: async url => response(url.includes('/status') ? payload
                    : url.includes('/quota/refresh') ? newer : receipt) });
            await store.refresh();
            await store.resetAccount(receipt.request, { key: 'old-operation' });
            await store.refreshResources(target);
            const held = structuredClone(store.snapshot);
            await store.resetAccount(receipt.request, { recover: true });
            assert.deepEqual(store.snapshot, held);
            assert.equal(store.resourceAction(target).refreshed, true);
            assert.equal(store.resourceAction(target).refreshDone, true);
            assert.equal(resourceReadGap(store.resourceAction(target)), false);
            // Equivalent instants with different offsets are ties, not new reads.
            receipt.resources = atTime('2026-10-09T15:00:00+03:00', 0.8, 7);
            await store.resetAccount(receipt.request, { recover: true });
            assert.deepEqual(store.snapshot, held);
            // Receiving that old receipt after a failed status read is not a new read.
            payload.reads.quota = 'failed';
            await store.refresh();
            assert.equal(store.resourceAction(target).resourceRead, false);
            const generation = store.generation;
            await store.resetAccount(receipt.request, { recover: true });
            assert.equal(store.resourceAction(target).resourceRead, false);
            assert.equal(store.generation, generation);
            // Recovery can still deliver genuinely newer observations.
            receipt.resources = atTime('2026-10-09T09:30:00-03:00', 0.2, 2);
            if (receiptAbsence) absent(receipt.resources);
            await store.resetAccount(receipt.request, { recover: true });
            if (receiptAbsence) assert.deepEqual(store.snapshot.quota, [sibling]);
            else assert.equal(store.snapshot.quota.at(-1).constraints[0].used_ratio, 0.2);
            assert.deepEqual(store.snapshot.quota_absences, receipt.resources.absences);
            assert.equal(store.snapshot.resources.at(-1).resets.value[0].available_count, 2);
            assert.equal(store.resourceAction(target).resourceRead, true);
            assert.deepEqual(store.snapshot.quota[0], sibling);
            assert.deepEqual(store.snapshot.resources[0], siblingResource);
            store.dispose();
        });
    }
}

test('receipt observations merge each facet independently using observation and attempt clocks', async () => {
    const newer = atTime('2026-10-09T12:00:00Z', 0.4, 0);
    const mixed = atTime('2026-10-09T10:00:00Z', 0.9, 1);
    mixed.resources[0].balances.observed_at = '2026-10-09T09:30:00-03:00';
    mixed.resources[0].balances.value[0].amount = '100';
    mixed.resources[0].spending.last_attempt_at = '2026-10-09T12:30:00Z';
    mixed.resources[0].spending.last_error = 'spending_read_failed';
    mixed.resources[0].spending.freshness = 'stale';
    mixed.resources[0].diagnostics.observed_at = null;
    mixed.resources[0].diagnostics.last_attempt_at = null;
    const store = createClaudexorStatusStore({ doc: null, storage: () => null,
        fetchImpl: async url => response(url.includes('/status') ? snapshot()
            : url.includes('/quota/refresh') ? newer : { ...fixture.receipt, resources: mixed }) });
    await store.refresh();
    await store.refreshResources(target);
    await store.resetAccount(fixture.receipt.request, { key: 'mixed-observations' });
    const resource = store.snapshot.resources[0];
    assert.deepEqual(resource.balances, mixed.resources[0].balances);
    assert.deepEqual(resource.spending, mixed.resources[0].spending);
    assert.deepEqual(resource.resets, newer.resources[0].resets);
    assert.deepEqual(resource.diagnostics, newer.resources[0].diagnostics);
    store.dispose();
});

test('new no-effect intent keeps every earlier unresolved key through close and exact recovery', async () => {
    let saved = '[]';
    const storage = () => ({ getItem: () => saved, setItem: (_, value) => { saved = value; } });
    const calls = [];
    const fetchImpl = async (url, init) => {
        if (url.includes('/status')) return response(snapshot());
        calls.push({ url, ...init });
        if (calls.length <= 2) throw new Error('lost reply');
        return response({ ...fixture.receipt, request: JSON.parse(init.body), outcome: 'no_credit', resources: null });
    };
    let store = createClaudexorStatusStore({ doc: null, storage, fetchImpl });
    await store.refresh();
    const refill = { target, offer_id: 'opaque-refill' };
    await store.resetAccount(fixture.receipt.request, { key: 'first-lost' });
    await store.resetAccount(refill, { key: 'second-lost' });
    await store.resetAccount(refill, { key: 'deliberate-no-effect' });
    assert.deepEqual(JSON.parse(saved).map(entry => entry.key), ['first-lost', 'second-lost']);
    assert.equal(store.resourceAction(target).receipt.outcome, 'no_credit');
    assert.equal(store.resourceAction(target).earlierRequests.length, 2);
    const html = accountResourcesMarkup(row, store.snapshot, { action: store.resourceAction(target) });
    assert.match(html, /Earlier unresolved requests/);
    assert.match(html, /data-recover-reset="first-lost"/);
    assert.match(html, /data-recover-reset="second-lost"/);
    assert.equal(resourceReadGap(store.resourceAction(target)), true);
    store.dispose();
    store = createClaudexorStatusStore({ doc: null, storage, fetchImpl });
    assert.equal(calls.length, 3, 'restoration sends nothing');
    await store.resetAccount(fixture.receipt.request, { recover: true, key: 'first-lost' });
    assert.deepEqual(calls[3], calls[0]);
    assert.deepEqual(JSON.parse(saved).map(entry => entry.key), ['second-lost']);
    assert.equal(store.resourceAction(target).key, 'second-lost');
    await store.resetAccount(refill, { recover: true, key: 'second-lost' });
    assert.deepEqual(calls[4], calls[1]);
    assert.deepEqual(JSON.parse(saved), []);
    store.dispose();
});

for (const read of ['refresh', 'wake']) {
    test(`unread operations catalog preserves resource facts, independently of successful agent catalog (${read})`, async () => {
        let payload = snapshot();
        const store = createClaudexorStatusStore({ doc: null, storage: () => null, fetchImpl: async () => response(payload) });
        await store.refresh();
        payload = { ...snapshot(), resources: undefined, resource_capabilities: { read: false, reset: false },
            resource_capabilities_read: 'failed' };
        payload.quota[0].constraints[0].used_ratio = 0.3;
        payload.profiles = { profiles: [{ profile: { harness_id: 'claude', profile_id: 'new-account' } }] };
        await store[read]();
        assert.equal(store.catalogKnown, true);
        assert.equal(store.quotaKnown, true);
        assert.equal(store.snapshot.quota[0].constraints[0].used_ratio, 0.3);
        assert.deepEqual(store.snapshot.profiles, payload.profiles);
        assert.deepEqual(store.snapshot.resources, fixture.quota.resources);
        const html = accountResourcesMarkup(row, store.snapshot);
        assert.match(html, /capabilities could not be checked/i);
        assert.doesNotMatch(html, /does not expose account resources/);
        assert.match(html, /Last reported: 1 available/);
        assert.match(html, /data-resource-reset="opaque-refill"/);
        assert.equal(resourceSummary(row, store.snapshot), '');
        // An actual legacy catalog is authoritative absence, and a later rich read recovers.
        payload.resource_capabilities_read = 'ok';
        await store[read]();
        assert.equal(store.snapshot.resources, undefined);
        assert.match(accountResourcesMarkup(row, store.snapshot), /does not expose account resources/);
        payload = snapshot();
        payload.resource_capabilities_read = 'ok';
        await store[read]();
        assert.match(resourceSummary(row, store.snapshot), /1 available/);
        store.dispose();
    });
}

test('refill keeps its verb with and without previous operations', async () => {
    const offer = fixture.quota.resources[0].resets.value[1];
    for (const prior of [{}, { request: fixture.receipt.request, receipt: fixture.receipt }]) {
        const label = prior.request ? 'Refill session again' : 'Refill session';
        assert.match(accountResourcesMarkup(row, snapshot(), { action: prior }),
            new RegExp(`data-resource-reset="opaque-refill"[^>]*>${label}</button>`));
        await confirmAccountReset(row, offer, null, { store: { resourceAction: () => prior },
            dialogImpl: async options => { assert.equal(options.confirmLabel, label); return false; } });
    }
});

test('foreground refresh follows pre-existing status read; later poll cannot overwrite its result', async () => {
    const old = deferred(), post = deferred();
    const calls = [];
    const store = createClaudexorStatusStore({ storage: () => null, doc: null, fetchImpl: async (url, init) => {
        calls.push([url, init]);
        return calls.length === 1 ? old.promise : url.includes('/refresh') ? post.promise : response(snapshot());
    } });
    const read = store.refresh();
    const refresh = store.refreshResources(target);
    assert.equal(calls.length, 1);
    old.resolve(response(snapshot()));
    await read;
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(calls[1][0], '/api/claudexor/quota/refresh');
    const poll = store.refresh();
    assert.equal(calls.length, 2);
    post.resolve(response(fixture.quota));
    await refresh;
    await poll;
    assert.equal(calls.length, 3);
    store.dispose();
});

test('failed status quota read retains resource facts with failed provenance', async () => {
    let payload = snapshot();
    const store = createClaudexorStatusStore({ storage: () => null, doc: null,
        fetchImpl: async url => response(url.includes('/quota/refresh') ? fixture.quota : payload) });
    await store.refresh();
    await store.refreshResources(target);
    assert.equal(store.resourceAction(target).resourceRead, true);
    payload = { ...snapshot(), quota: [], resources: [], reads: { catalog: 'ok', accounts: 'ok', quota: 'failed' } };
    await store.refresh();
    assert.equal(store.resourceAction(target).resourceRead, false);
    assert.deepEqual(store.snapshot.resources, fixture.quota.resources);
    assert.equal(store.facet('quota'), 'failed');
    assert.match(accountResourcesMarkup(row, store.snapshot, {
        quotaRead: store.facet('quota'), action: store.resourceAction(target) }), /Last known · 100%/);
    store.dispose();
});

test('a full refresh preceding a queued reset cannot freshen its failed readback', async () => {
    const foreground = deferred();
    const calls = [];
    const store = createClaudexorStatusStore({ storage: () => null, doc: null, fetchImpl: async url => {
        calls.push(url);
        if (url.includes('/status')) return response(snapshot());
        return url.includes('/quota/refresh') ? foreground.promise : response(fixture.receipt);
    } });
    await store.refresh();
    const refresh = store.refreshResources();
    const reset = store.resetAccount(fixture.receipt.request, { key: 'queued-reset' });
    assert.equal(calls.length, 2);
    foreground.resolve(response(fixture.quota));
    await refresh;
    await reset;
    assert.equal(calls.at(-1), '/api/claudexor/account-resets');
    assert.equal(resourceReadGap(store.resourceAction(target)), true);
    assert.match(accountResourcesMarkup(row, store.snapshot, { action: store.resourceAction(target) }), /Last known · 100% used/);
    store.dispose();
});
