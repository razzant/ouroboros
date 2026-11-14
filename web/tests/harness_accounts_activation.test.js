import assert from 'node:assert/strict';
import test from 'node:test';

import { apiClient } from '../modules/api_client.js';
import { createClaudexorStatusStore } from '../modules/claudexor_status_store.js';
import {
    destroyHarnessAccounts,
    initHarnessAccounts,
    serviceBannerLine,
} from '../modules/harness_accounts.js';

const payload = (state, reads = 'not_read') => ({
    daemon: { state, engine_version: '3.8.0', runtime: {} },
    config_dir: '/tmp/visual-test', harnesses: [], profiles: {}, quota: [],
    reads: { catalog: reads, accounts: reads, quota: reads },
});
const STALE = {
    ...payload('stale'),
    daemon: { ...payload('stale').daemon, runtime: { state: 'ready' } },
};
const RUNNING = payload('running', 'ok');
const passive = (body) => ({
    view: 'quota', profiles: body.profiles, quota: body.quota, quota_absences: [],
    reads: { accounts: body.reads.accounts, quota: body.reads.quota },
});

function response(body) {
    return { ok: true, status: 200, json: async () => body };
}

function element(id) {
    return {
        id, offsetParent: {}, dataset: {}, textContent: '', innerHTML: '', disabled: false,
        addEventListener() {}, querySelector: () => null, querySelectorAll: () => [],
    };
}

function surface() {
    const elements = Object.fromEntries([
        'agents-service-banner', 'harness-login-card', 'harness-accounts-groups',
        'btn-harness-refresh',
    ].map((id) => [id, element(id)]));
    // Settings mounts before the provider's page is shown. A visible fake here
    // starts program inspection from the subscription, not page activation.
    elements['harness-accounts-groups'].offsetParent = null;
    const listeners = {};
    const doc = {
        hidden: false,
        getElementById: (id) => elements[id] || null,
        addEventListener() {}, removeEventListener() {},
    };
    const win = {
        addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
        removeEventListener(type, fn) {
            listeners[type] = (listeners[type] || []).filter((item) => item !== fn);
        },
        // The Settings shell's signal: which Settings page is shown now (`detail.path`; '' is the home).
        fire(type, path = PAGE) {
            if (type === 'ouro:settings-page-shown') {
                elements['harness-accounts-groups'].offsetParent = path === PAGE ? {} : null;
            }
            return Promise.all((listeners[type] || []).map((fn) => fn({ detail: { path } })));
        },
    };
    return { elements, doc, win };
}
const PAGE = 'accounts/codex';

test('showing the provider\'s page wakes only an already-provisioned idle daemon; no other page of Settings reads or wakes', async (t) => {
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    const { elements, doc, win } = surface();
    globalThis.document = doc;
    globalThis.window = win;
    const requests = [];
    t.mock.method(apiClient, 'harnessMaintenanceInventory', async () => ({ harnesses: [] }));
    let body = STALE;
    let failNextRead = false;
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => {
            requests.push(`${init.method || 'GET'} ${url}`);
            // Fail lifecycle evidence, not the independent passive read that
            // now precedes it. A successful full read may authorize a wake.
            if (url === '/api/claudexor/status' && failNextRead) {
                failNextRead = false;
                return { ok: false, status: 503, json: async () => ({ error: 'temporary outage' }) };
            }
            return response(url.includes('?view=quota') ? passive(body)
                : (init.method || 'GET') === 'POST' ? RUNNING : body);
        },
        doc,
    });
    try {
        await initHarnessAccounts({ store, family: 'codex', path: PAGE });
        await store.refresh();
        await store.observe();
        assert.ok(!requests.some(request => request.startsWith('POST ')), 'mount must not wake');
        // The home, the list of providers, another provider's page and another app page are not this page's showing:
        // none of them reads the status or starts the daemon, though the idle daemon is one press from a wake.
        requests.length = 0;
        for (const other of ['', 'accounts', 'accounts/claude', 'accounts/key/openai', 'models']) await win.fire('ouro:settings-page-shown', other);
        await win.fire('ouro:page-shown', PAGE);
        assert.deepEqual(requests, []);
        await win.fire('ouro:settings-page-shown');
        assert.ok(requests.includes('POST /api/claudexor/wake'));
        assert.equal(requests.filter((request) => request === 'POST /api/claudexor/wake').length, 1);

        // A failed refresh keeps the last snapshot for display. It must not
        // make that retained `stale + ready` value look like fresh wake proof.
        body = STALE;
        await store.refresh();
        requests.length = 0;
        failNextRead = true;
        await win.fire('ouro:settings-page-shown');
        assert.deepEqual(requests, ['GET /api/claudexor/status?view=quota', 'GET /api/claudexor/status']);
        assert.equal(store.error, 'temporary outage');

        body = payload('not_provisioned');
        await store.refresh();
        requests.length = 0;
        await win.fire('ouro:settings-page-shown');
        assert.ok(!requests.includes('POST /api/claudexor/wake'));
        assert.match(elements['agents-service-banner'].textContent, /No accounts connected yet/i);

        for (const daemon of [
            { state: 'stale', runtime: { state: 'error' } },
            { state: 'stale', runtime: { state: 'update_available' } },
            { state: 'stale', runtime: { state: 'ready' }, ownership_problem: 'foreign home' },
        ]) {
            body = { ...payload('stale'), daemon };
            await store.refresh();
            requests.length = 0;
            await win.fire('ouro:settings-page-shown');
            assert.ok(!requests.includes('POST /api/claudexor/wake'));
        }
    } finally {
        destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
    }
});

test('the service banner reports an automatic wake in progress', () => {
    assert.deepEqual(serviceBannerLine({}, { wakeBusy: true }), {
        tone: 'muted', text: 'Starting the agent daemon…',
    });
});

test('mount reads status only; showing the provider\'s page reads the program inventory once', async (t) => {
    const before = {document: globalThis.document, window: globalThis.window};
    const {elements, doc, win} = surface();
    globalThis.document = doc; globalThis.window = win;
    const inventories = [];
    // The program controller intentionally retains its session observation.
    // Advance its documented freshness/latest clocks, rather than depending on
    // whether another test happened to inspect the same whole-inventory slot.
    const now = Date.now() + 1800001;
    t.mock.method(Date, 'now', () => now);
    t.mock.method(globalThis, 'fetch', async url => { inventories.push(String(url)); return response({harnesses: []}); });
    const inventoryRead = apiClient.harnessMaintenanceInventory.bind(apiClient);
    const inventoryReads = [];
    t.mock.method(apiClient, 'harnessMaintenanceInventory', (...args) => {
        const read = inventoryRead(...args);
        inventoryReads.push(read);
        return read;
    });
    const store = createClaudexorStatusStore({fetchImpl: async url => response(
        url.includes('?view=quota') ? passive(RUNNING) : RUNNING), doc});
    try {
        await initHarnessAccounts({store, family: 'codex', path: PAGE});
        await store.refresh();
        await store.observe();
        assert.deepEqual(inventories, []);
        await win.fire('ouro:settings-page-shown', 'accounts');
        await win.fire('ouro:settings-page-shown', '');
        assert.deepEqual(inventories, [], 'the list of providers and the home read no program');
        assert.ok(elements['harness-accounts-groups']);
        await win.fire('ouro:settings-page-shown');
        await Promise.all(inventoryReads);
        assert.equal(inventories.length, 1);
        assert.match(inventories[0], /fresh=true/);
        assert.match(inventories[0], /checkLatest=true/);
        await win.fire('ouro:settings-page-shown');
        assert.equal(inventories.length, 1, 'a current inspection is reused on reopening');
    } finally {
        destroyHarnessAccounts(); store.dispose();
        globalThis.document = before.document; globalThis.window = before.window;
    }
});

for (const [daemon, shouldWake] of [
    [STALE.daemon, true],
    [{ ...STALE.daemon, ownership_problem: 'foreign home' }, false],
    [{ state: 'not_provisioned', runtime: {} }, false],
]) test(`first activation joins held lifecycle discovery after passive paint (${daemon.state}/${shouldWake})`, async (t) => {
    const before = { document: globalThis.document, window: globalThis.window };
    const { doc, win } = surface();
    globalThis.document = doc; globalThis.window = win;
    t.mock.method(apiClient, 'harnessMaintenanceInventory', async () => ({ harnesses: [] }));
    const calls = [];
    let finishFull, fullReads = 0;
    const full = new Promise(resolve => { finishFull = resolve; });
    const store = createClaudexorStatusStore({ doc, fetchImpl: async (url, init = {}) => {
        calls.push(`${init.method || 'GET'} ${url}`);
        if (url.includes('?view=quota')) return response(passive(RUNNING));
        if ((init.method || 'GET') === 'POST') return response(RUNNING);
        fullReads++;
        return full;
    } });
    try {
        await initHarnessAccounts({ store, family: 'codex', path: PAGE });
        const initialRead = store.refresh();
        const activated = win.fire('ouro:settings-page-shown');
        await store.observe();
        assert.equal(store.snapshot?.daemon?.state, undefined, 'passive paint has no lifecycle evidence');
        assert.ok(!calls.some(call => call.startsWith('POST ')), 'unknown lifecycle cannot authorize a wake');
        finishFull(response({ ...STALE, daemon }));
        await activated;
        await initialRead;
        assert.equal(fullReads, 1, 'activation joins the existing full read, without another discovery RPC');
        assert.equal(calls.filter(call => call === 'POST /api/claudexor/wake').length, shouldWake ? 1 : 0);
    } finally {
        finishFull(response(RUNNING));
        destroyHarnessAccounts(); store.dispose();
        globalThis.document = before.document; globalThis.window = before.window;
    }
});
