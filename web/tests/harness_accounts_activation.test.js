import assert from 'node:assert/strict';
import test from 'node:test';

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
        fire(type, path = PAGE) { return Promise.all((listeners[type] || []).map((fn) => fn({ detail: { path } }))); },
    };
    return { elements, doc, win };
}
const PAGE = 'accounts/codex';

test('showing the provider\'s page wakes only an already-provisioned idle daemon; no other page of Settings reads or wakes', async () => {
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    const { elements, doc, win } = surface();
    globalThis.document = doc;
    globalThis.window = win;
    const requests = [];
    let body = STALE;
    let failNextRead = false;
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => {
            requests.push(`${init.method || 'GET'} ${url}`);
            if ((init.method || 'GET') === 'GET' && failNextRead) {
                failNextRead = false;
                return { ok: false, status: 503, json: async () => ({ error: 'temporary outage' }) };
            }
            return response((init.method || 'GET') === 'POST' ? RUNNING : body);
        },
        doc,
    });
    try {
        await initHarnessAccounts({ store, family: 'codex', path: PAGE });
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
        assert.deepEqual(requests, ['GET /api/claudexor/status']);

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
    t.mock.method(globalThis, 'fetch', async url => { inventories.push(String(url)); return response({harnesses: []}); });
    const store = createClaudexorStatusStore({fetchImpl: async () => response(RUNNING), doc});
    try {
        await initHarnessAccounts({store, family: 'codex', path: PAGE});
        assert.deepEqual(inventories, []);
        await win.fire('ouro:settings-page-shown', 'accounts');
        await win.fire('ouro:settings-page-shown', '');
        assert.deepEqual(inventories, [], 'the list of providers and the home read no program');
        assert.ok(elements['harness-accounts-groups']);
        await win.fire('ouro:settings-page-shown');
        assert.equal(inventories.length, 1);
        assert.match(inventories[0], /fresh=true/);
        assert.match(inventories[0], /checkLatest=true/);
    } finally {
        destroyHarnessAccounts(); store.dispose();
        globalThis.document = before.document; globalThis.window = before.window;
    }
});
