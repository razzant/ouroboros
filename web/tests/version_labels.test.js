import assert from 'node:assert/strict';
import test from 'node:test';

import { lastAppVersion, loadVersion } from '../modules/utils.js';
import { bindAppVersionRender, updateVerdict } from '../modules/updates.js';
import { WS } from '../modules/ws.js';

// The two version labels' one reader (utils.loadVersion, #2249) under a scripted /api/health:
// what each answer writes, which answers write nothing, the order reads land in, and the
// socket-open binding app.js makes. The real SPA over a reconnect: tests/test_version_labels_browser.py.

function labels({ nav = '', about = '', present = ['nav-version', 'about-version'] } = {}) {
    const nodes = { 'nav-version': { textContent: nav }, 'about-version': { textContent: about } };
    globalThis.document = { getElementById: (id) => (present.includes(id) ? nodes[id] : null) };
    return () => [nodes['nav-version'].textContent, nodes['about-version'].textContent];
}

// Every request waits until the test answers it, so reads can land in any order.
function scriptFetch() {
    const reads = [];
    globalThis.fetch = (url, init) => new Promise((resolve, reject) => reads.push({ url, init, resolve, reject }));
    const respond = (index, body, { status = 200, malformed = false } = {}) => reads[index].resolve({
        ok: status >= 200 && status < 300,
        status,
        json: async () => {
            if (malformed) throw new SyntaxError('Unexpected token < in JSON');
            return body;
        },
    });
    return { reads, respond, fail: (index) => reads[index].reject(new TypeError('Failed to fetch')) };
}

async function readOnce(body, options) {
    const health = scriptFetch();
    const read = loadVersion();
    health.respond(0, body, options);
    await read;
    return health.reads;
}

test('one uncached health read writes the sidebar runtime version and the About app/runtime pair', async () => {
    const shown = labels();
    const reads = await readOnce({ status: 'ok', version: '7.8.1', runtime_version: '7.8.1', app_version: '7.7.0' });
    assert.deepEqual(reads.map((read) => [read.url, read.init.cache]), [['/api/health', 'no-store']]);
    assert.deepEqual(shown(), ['v7.8.1', 'app 7.7.0 | rt 7.8.1']);
});

test('an app version that is absent, empty or not a string shows the runtime version in its place', async () => {
    for (const app_version of [undefined, '', '   ', 42, null, { v: '1' }]) {
        const shown = labels({ nav: 'v7.0.0', about: 'app 7.0.0 | rt 7.0.0' });
        await readOnce({ runtime_version: '7.8.1', version: '7.8.1', app_version });
        assert.deepEqual(shown(), ['v7.8.1', 'app 7.8.1 | rt 7.8.1'], String(app_version));
    }
});

test('the legacy version field stands in for an absent or invalid runtime version', async () => {
    for (const runtime_version of [undefined, '', '  ', 7.8, null, ['7.9.0']]) {
        const shown = labels();
        await readOnce({ runtime_version, version: '7.6.0', app_version: '7.6.2' });
        assert.deepEqual(shown(), ['v7.6.0', 'app 7.6.2 | rt 7.6.0'], String(runtime_version));
    }
    const shown = labels();
    await readOnce({ version: ' 7.6.0 ' });
    assert.deepEqual(shown(), ['v7.6.0', 'app 7.6.0 | rt 7.6.0'], 'an old core reports only the legacy field');
});

test('an answer without a usable version leaves the labels as they are and never says "?"', async () => {
    const answers = [
        [{}, {}], [{ runtime_version: 42, version: null, app_version: '7.8.1' }, {}], [{ version: '   ' }, {}],
        [null, {}], ['7.8.1', {}], [['7.8.1'], {}],
        [{ error: 'boot failed', version: '9.9.9' }, { status: 500 }],  // refused: fetchJson rejects
        [{ error: 'unauthorized', runtime_version: '9.9.9' }, { status: 401 }],
        [undefined, { malformed: true }],  // a 200 page that is not JSON reads as an error object
    ];
    for (const [known, before] of [[true, ['v7.8.0', 'app 7.8.0 | rt 7.8.0']], [false, ['', '']]]) {
        for (const [body, options] of answers) {
            const shown = labels(known ? { nav: before[0], about: before[1] } : {});
            await readOnce(body, options);
            assert.deepEqual(shown(), before, JSON.stringify([body, options]));
        }
        const shown = labels(known ? { nav: before[0], about: before[1] } : {});
        const health = scriptFetch();
        const read = loadVersion();
        health.fail(0);
        await read;  // resolves: a socket listener calls it without a catch
        assert.deepEqual(shown(), before, 'network failure');
    }
});

test('an older answer that lands after a newer one changes nothing', async () => {
    const shown = labels({ nav: 'v7.8.0', about: 'app 7.8.0 | rt 7.8.0' });
    const health = scriptFetch();
    const older = loadVersion();
    const newer = loadVersion();
    health.respond(1, { runtime_version: '7.8.1', app_version: '7.8.1' });
    await newer;
    assert.deepEqual(shown(), ['v7.8.1', 'app 7.8.1 | rt 7.8.1']);
    health.respond(0, { runtime_version: '7.8.0', app_version: '7.8.0' });
    await older;
    assert.deepEqual(shown(), ['v7.8.1', 'app 7.8.1 | rt 7.8.1'], 'the late old answer is dropped');
});

test('a newer read that fails drops the older pending one too, so the labels keep their last text', async () => {
    for (const before of [['v7.8.0', 'app 7.8.0 | rt 7.8.0'], ['', '']]) {
        for (const failNewer of [(health) => health.fail(1), (health) => health.respond(1, null, { malformed: true }),
            (health) => health.respond(1, { error: 'starting' }, { status: 503 })]) {
            const shown = labels({ nav: before[0], about: before[1] });
            const health = scriptFetch();
            const older = loadVersion();
            const newer = loadVersion();
            failNewer(health);
            await newer;
            // The exact overlap: the older read is still pending when the newer one fails.
            health.respond(0, { runtime_version: '7.8.1', app_version: '7.8.1' });
            await older;
            assert.deepEqual(shown(), before);
            assert.equal(health.reads.length, 2, 'no retry');
        }
    }
});

test('a label that is not in the page is skipped and the other is still written', async () => {
    let shown = labels({ present: [] });
    await readOnce({ runtime_version: '7.8.1' });
    assert.deepEqual(shown(), ['', '']);
    shown = labels({ present: ['nav-version'] });  // Settings not mounted
    await readOnce({ runtime_version: '7.8.1', app_version: '7.7.0' });
    assert.deepEqual(shown(), ['v7.8.1', '']);
    shown = labels({ present: ['about-version'] });
    await readOnce({ runtime_version: '7.8.1', app_version: '7.7.0' });
    assert.deepEqual(shown(), ['', 'app 7.7.0 | rt 7.8.1']);
});

class FakeSocket {
    static OPEN = 1;
    static CONNECTING = 0;
    static instances = [];

    constructor(url) {
        this.url = url;
        this.readyState = FakeSocket.CONNECTING;
        this.sent = [];
        FakeSocket.instances.push(this);
    }

    send(data) { this.sent.push(data); }

    close() {}
}

test('every socket open re-reads the labels without holding up its state read, sends or reload decision', async () => {
    const shown = labels();
    const replaced = [];
    globalThis.WebSocket = FakeSocket;
    globalThis.window = { location: { href: 'http://127.0.0.1:8000/', replace: (url) => replaced.push(String(url)) } };
    const fetches = scriptFetch();
    const health = (index) => fetches.reads.filter((read) => read.url === '/api/health')[index];
    const reads = (url) => fetches.reads.filter((read) => read.url === url).length;
    const answer = (read, body) => read.resolve({ ok: true, status: 200, json: async () => body });
    const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
    const ws = new WS('ws://unused');
    ws.on('open', () => { loadVersion(); });  // app.js's binding
    try {
        ws.connect();
        const first = FakeSocket.instances.at(-1);
        first.readyState = FakeSocket.OPEN;
        first.onopen();
        assert.deepEqual(fetches.reads.map((read) => read.url), ['/api/health', '/api/state'],
            'the open listener returned before the post-open state read');
        answer(fetches.reads[1], { sha: 'abc1234', served_ui: { source: 'builtin' } });
        answer(health(0), { runtime_version: '7.8.0' });
        await settle();
        assert.deepEqual(shown(), ['v7.8.0', 'app 7.8.0 | rt 7.8.0']);

        first.readyState = 3;
        first.onclose();
        const queued = ws.send({ type: 'chat', text: 'Draft sent while the core restarts' });
        assert.equal(queued.status, 'queued');
        ws._clearReconnectTimer();
        const second = FakeSocket.instances.at(-1);
        assert.notEqual(second, first);
        second.readyState = FakeSocket.OPEN;
        second.onopen();
        // The health read is still unanswered: the open already read the state and sent the queue.
        assert.equal(reads('/api/state'), 2);
        assert.equal(second.sent.length, 1, 'the queued send flushed on the open');
        answer(fetches.reads.filter((read) => read.url === '/api/state')[1],
            { sha: 'def5678', served_ui: { source: 'builtin' } });
        await settle();
        assert.deepEqual(replaced, [], 'an unmarked page keeps itself under a core that reports served_ui');
        answer(health(1), { runtime_version: '7.8.1', app_version: '7.8.1' });
        await settle();
        assert.deepEqual(shown(), ['v7.8.1', 'app 7.8.1 | rt 7.8.1']);
        assert.equal(reads('/api/health'), 2, 'one read per open, no polling');
    } finally {
        ws._clearUiRecoveryTimer();
        ws._clearReconnectTimer();
        ws._clearWatchdogTimer();
        ws._clearPostOpenRetryTimer();
        ws.ws = null;
    }
});

test('the Updates page shows the app version About reads as one App chip, and none without it', async () => {
    labels();
    await readOnce({ runtime_version: '7.9.0', app_version: '7.10.0' });
    assert.equal(lastAppVersion(), '7.10.0');
    assert.deepEqual(updateVerdict({}, '', lastAppVersion()).chips, [{ label: 'App', value: '7.10.0' }]);
    await readOnce({ runtime_version: '7.9.0' });  // no app version reported: no chip
    assert.deepEqual(updateVerdict({}, '', lastAppVersion()).chips, []);
});

test('a health read landing after the update status renders the App chip; one landing first waits for the status', async () => {
    labels();
    await readOnce({ runtime_version: '7.9.0' });  // no app version yet
    let status = null;
    const shown = [];
    const render = () => shown.push(updateVerdict(status || {}, '', lastAppVersion()).chips);
    const dispose = bindAppVersionRender({ getStatus: () => status, render });
    try {
        status = { managed: true };  // the status lands first and renders without a chip
        render();
        await readOnce({ runtime_version: '7.9.0', app_version: '7.10.0' });
        assert.deepEqual(shown.at(-1), [{ label: 'App', value: '7.10.0' }], 'the chip waited for another status refresh');
        status = null;
        shown.length = 0;
        await readOnce({ runtime_version: '7.9.0', app_version: '7.11.0' });  // health first: the loading line stays
        assert.deepEqual(shown, []);
        status = { managed: true };
        render();  // the status's own render shows it
        assert.deepEqual(shown, [[{ label: 'App', value: '7.11.0' }]]);
    } finally {
        dispose();
    }
});
