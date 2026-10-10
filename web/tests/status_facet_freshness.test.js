import assert from 'node:assert/strict';
import test from 'node:test';

import {
    FACET_ACCOUNTS,
    FACET_QUOTA,
    createClaudexorStatusStore,
} from '../modules/claudexor_status_store.js';
import { daemonStatusLine } from '../modules/harness_accounts.js';

// The status answer keeps a running daemon `running` when one fanned-out read refuses;
// the refused facet carries its own typed error in `facets`, and a stale last read of it
// may stand in its collection. The sentences name that facet's own error.
const runningWithRefusedQuota = (facets) => ({
    daemon: { state: 'running', engine_version: '3.25.1', runtime: {} },
    config_dir: '/home/agent',
    harnesses: [{ id: 'codex' }],
    profiles: { profiles: [], harnessAccounts: [] },
    quota: [{ subject: { harness: 'codex' }, freshness: 'fresh', constraints: [] }],
    reads: { catalog: 'ok', accounts: 'ok', quota: 'failed' },
    ...(facets ? { facets } : {}),
});

const fakeDoc = () => ({ hidden: false, addEventListener() {}, removeEventListener() {} });
const okResponse = (body) => ({ ok: true, status: 200, json: async () => body });

test("a refused facet's note carries that facet's own error, not another read's", async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => okResponse(runningWithRefusedQuota({
            catalog: { observed_at: '2026-10-10T10:05:00Z', stale: false, error: null },
            accounts: { observed_at: '2026-10-10T10:05:00Z', stale: false, error: null },
            quota: { observed_at: '2026-10-10T10:00:00Z', stale: true, error: 'daemon_busy' },
        })),
        doc: fakeDoc(),
    });
    await store.refresh();
    assert.match(store.unavailableNote(FACET_QUOTA).text, /could not be read \(daemon_busy\)/);
    assert.equal(store.unavailableNote(FACET_ACCOUNTS), null, 'the accounts read landed: nothing to explain');
    store.dispose();
});

test('the Agents status line names the refused facet and its error under a running daemon', () => {
    const line = daemonStatusLine(runningWithRefusedQuota({
        quota: { observed_at: null, stale: false, error: 'daemon_busy' },
    }));
    assert.equal(line.tone, 'warn');
    assert.match(line.text, /^Claudexor is running, but subscription limits were not read: daemon_busy\./);
    // A backend without `facets` keeps the daemon's own last_error as the detail.
    const legacy = runningWithRefusedQuota();
    legacy.daemon.last_error = 'quota_probe_failed: read died';
    assert.match(daemonStatusLine(legacy).text, /were not read: quota_probe_failed: read died\./);
});

test("a refused quota read keeps this client's last value with this client's own time", async () => {
    // This client read q1 at 10:00; a foreground refresh then showed it q3 (the server's memory never
    // saw it); another client refreshed the server memory to q2 at 10:05; now the quota read fails.
    const q = (id) => [{ subject: { harness: 'codex' }, freshness: 'fresh', constraints: [{ id }] }];
    const ok = { ...runningWithRefusedQuota({ quota: { observed_at: '2026-10-10T10:00:00Z', stale: false, error: null } }),
        quota: q('q1'), reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' } };
    const refused = { ...runningWithRefusedQuota({
        quota: { observed_at: '2026-10-10T10:05:00Z', stale: true, error: 'daemon_busy' },
    }), quota: q('q2') };
    const legacy = { ...runningWithRefusedQuota(), quota: [] };   // no `facets`: an older backend
    const serve = [ok, refused, ok, legacy];
    const store = createClaudexorStatusStore({ fetchImpl: async () => okResponse(structuredClone(serve.shift())), doc: fakeDoc() });
    await store.refresh();
    store.snapshot.quota = q('q3');   // what a successful foreground refresh merges into the snapshot
    await store.refresh();
    assert.deepEqual(store.snapshot.quota, q('q3'), 'no rollback of what this client already showed');
    assert.equal(store.snapshot.facets.quota.observed_at, '2026-10-10T10:00:00Z', "this client's own stamp, not the server's");
    assert.equal(store.snapshot.facets.quota.stale, true);
    assert.equal(store.snapshot.facets.quota.error, 'daemon_busy', "this read's own error");
    await store.refresh();
    await store.refresh();
    assert.deepEqual(store.snapshot.quota, q('q1'), 'an older backend keeps the client value as before');
    store.dispose();
});
