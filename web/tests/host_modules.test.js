// Skills' page modules (host_modules.js) over the real contributions store and a fake page: the versioned context
// and nothing else, every failure (import, throw, reject, no start in time, a provider it does not declare) turning
// that one contribution off with one notice while the rest run, disposal on leave, revision change, Panic and unload,
// and recovery through the start ledger: an entry before every import, a module held after a start the page did not
// outlive (named once, Start it mounts it, a new revision mounts as usual), and `?hold=all`.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createContributionsStore } from '../modules/contributions.js';
import { createHolds, HOLDS_KEY } from '../modules/holds.js';
import { createHostModules, originFacts } from '../modules/host_modules.js';

const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

function events() {
    const listeners = new Map();
    return {
        addEventListener(type, fn) {
            if (!listeners.has(type)) listeners.set(type, new Set());
            listeners.get(type).add(fn);
        },
        removeEventListener(type, fn) { listeners.get(type)?.delete(fn); },
        emit(type, event = {}) { for (const fn of [...(listeners.get(type) || [])]) fn({ type, ...event }); },
        listenerCount: () => [...listeners.values()].reduce((count, set) => count + set.size, 0),
    };
}

function storage({ refuse = false } = {}) {
    const map = new Map();
    return { map, getItem: (key) => (map.has(key) ? map.get(key) : null),
        setItem(key, value) { if (refuse) throw new Error('QuotaExceededError'); map.set(key, String(value)); },
        removeItem: (key) => map.delete(key) };
}

const hostItem = (key, extra = {}) => {
    const [skill, id] = key.split(':');
    return { key, skill, skill_label: skill, id, label: skill, title: '', icon: '', order: 0, revision: 'r1',
        render: { kind: 'host', url: `/api/extensions/${skill}/ui/r1/ui/${id}.js` }, ...extra };
};

const ledger = (store) => JSON.parse(store.map.get(HOLDS_KEY) || 'null')?.entries || {};
const seeded = (entries) => {
    const local = storage();
    local.setItem(HOLDS_KEY, JSON.stringify({ v: 1, entries }));
    return local;
};

async function setup({ items = [], modules = {}, href = 'http://127.0.0.1:8765/#chat', local = storage(),
    session = storage(), now = 1_000_000, start = true } = {}) {
    let payload = { contributions: items };
    const reads = [];
    const store = createContributionsStore({ client: { uiContributions: async () => { reads.push(1); return payload; } } });
    const clock = { now };
    const win = Object.assign(events(), {
        location: { href, hostname: new URL(href).hostname, replace: (url) => win.navigations.push(url) },
        navigations: [], localStorage: local, sessionStorage: session, isSecureContext: true,
        matchMedia: (query) => ({ matches: query.includes('reduce') }),
    });
    const doc = Object.assign(events(), { documentElement: { dataset: { theme: 'dark' } } });
    const imported = [];
    const ledgerAtImport = [];
    const notices = [];
    const drafts = [];
    // The ledger's own page-lifetime listener sits on a window of its own: the runtime's listeners are counted below.
    const holds = createHolds({ win: { location: win.location, localStorage: local, sessionStorage: session }, now: () => clock.now });
    const runtime = createHostModules({ store, win, doc, holds, mountTimeoutMs: 40, settleMs: 10,
        draft: { insert: (text) => { drafts.push(text); return true; } },
        notice: (text, tone, options = {}) => notices.push({ text, tone, ...options }),
        importModule: async (url) => {
            imported.push(url);
            ledgerAtImport.push({ ...ledger(local), ...ledger(session) });
            const found = modules[url];
            if (found instanceof Error) throw found;
            return typeof found === 'function' ? found() : found;
        } });
    await store.refresh();
    if (start) runtime.start();
    await tick();
    return { store, runtime, holds, win, doc, local, session, clock, imported, ledgerAtImport, notices, drafts, reads,
        async publish(next) { payload = { contributions: next }; await store.refresh(); await tick(); } };
}

test('a live page module mounts with exactly the versioned context and is let go when its skill leaves', async () => {
    const seen = [];
    let disposed = 0;
    const impl = { label: 'Device voice', voices: () => [], speak: async () => {}, stop() {} };
    const voice = hostItem('device_voice:tts', { provides: 'tts', runtime: 'client', destination: 'device',
        label: 'Device voice', skill_label: 'Device voice' });
    const plain = { ...hostItem('templates:simpler'), placement: 'composer.plus', render: { kind: 'declarative', url: '' } };
    const h = await setup({ items: [voice, plain], modules: { [voice.render.url]: { mount(ctx) {
        seen.push(ctx);
        ctx.providers.register('tts', impl);
        return { dispose() { disposed += 1; } };
    } } } });
    assert.deepEqual(h.imported, [voice.render.url], 'only render.kind host runs page code');
    const [ctx] = seen;
    assert.deepEqual(Object.keys(ctx).sort(), ['api', 'draft', 'key', 'notice', 'origin', 'providers', 'signal', 'skill', 'theme'],
        'nothing else: no composer, no feed, no stubs');
    assert.equal(Object.isFrozen(ctx), true);
    assert.deepEqual([ctx.api, ctx.skill, ctx.key], [1, 'device_voice', 'device_voice:tts']);
    assert.deepEqual(Object.keys(ctx.providers), ['register']);
    assert.deepEqual(Object.keys(ctx.draft), ['insert']);
    assert.deepEqual({ ...ctx.origin }, { secure: true, loopback: true });
    assert.equal(ctx.theme.dark, true);
    h.doc.documentElement.dataset.theme = 'light';
    assert.equal(ctx.theme.dark, false, 'theme facts are read when asked');
    assert.equal(ctx.theme.reducedMotion, true);
    assert.equal(ctx.draft.insert('Read this to me'), true);
    assert.deepEqual(h.drafts, ['Read this to me']);
    ctx.notice('No voices on this device');
    assert.deepEqual(h.notices.map((notice) => notice.text), ['Device voice: No voices on this device'], 'a notice names its skill');
    assert.deepEqual(h.runtime.mounted(), ['device_voice:tts']);
    assert.deepEqual(h.runtime.clientProviders('tts').map((entry) => [entry.key, entry.kind, entry.destination, entry.impl]),
        [['device_voice:tts', 'tts', 'device', impl]]);
    assert.deepEqual(h.runtime.clientProviders('stt'), []);
    let heard = 0;
    h.runtime.onProvidersChange(() => { heard += 1; });
    await h.publish([plain]);
    assert.equal(disposed, 1, 'a skill that left is disposed');
    assert.equal(ctx.signal.aborted, true);
    assert.deepEqual(h.runtime.clientProviders('tts'), [], 'and its provider is gone');
    assert.equal(heard, 1);
    assert.equal(ctx.draft.insert('late'), false, 'a disposed module can no longer write');
    assert.deepEqual(h.drafts, ['Read this to me']);
});

test('a module that fails in any way is off for this page with one notice naming it; the others run', async () => {
    const good = hostItem('good:ok');
    const failing = {
        missing: [hostItem('missing:x'), new Error('HTTP 404')],
        thrower: [hostItem('thrower:x'), { mount() { throw new Error('boom'); } }],
        rejecter: [hostItem('rejecter:x'), { mount: async () => { throw new Error('nope'); } }],
        silent: [hostItem('silent:x', { label: 'Silent clock', skill_label: 'Clock' }), { mount: () => new Promise(() => {}) }],
        empty: [hostItem('empty:x'), {}],
    };
    const modules = { [good.render.url]: { mount: () => ({ dispose() {} }) } };
    for (const [item, module] of Object.values(failing)) modules[item.render.url] = module;
    const h = await setup({ items: [good, ...Object.values(failing).map(([item]) => item)], modules });
    await tick(80);
    assert.deepEqual(h.runtime.mounted(), ['good:ok']);
    assert.deepEqual(Object.fromEntries(Object.values(failing).map(([item]) => [item.key, h.store.fault(item.key)])), {
        'missing:x': 'HTTP 404', 'thrower:x': 'boom', 'rejecter:x': 'nope', 'silent:x': 'it did not start in time',
        'empty:x': 'the module exports no mount(ctx)' });
    assert.equal(h.notices.length, 5);
    assert.ok(h.notices.every((notice) => notice.tone === 'warn' && notice.ttl === 0), 'each one stays until read');
    assert.ok(h.notices.some((notice) => notice.text === 'Silent clock (Clock) stopped working: it did not start in time. '
        + 'It stays off until you reload or update the skill.'));
    assert.equal(h.reads.length, 2, 'a failed import reads the projection again before it calls the module broken');
    const other = { ...hostItem('other:y'), placement: 'composer.plus', render: { kind: 'declarative', url: '' } };
    await h.publish([good, other, ...Object.values(failing).map(([item]) => item)]);
    assert.equal(h.imported.length, 6, 'a changed projection with the same revisions mounts none of them again');
    const fixed = hostItem('thrower:x', { revision: 'r2', render: { kind: 'host', url: '/api/extensions/thrower/ui/r2/ui/x.js' } });
    modules[fixed.render.url] = { mount: () => ({ dispose() {} }) };
    await h.publish([good, fixed]);
    assert.deepEqual(h.runtime.mounted().sort(), ['good:ok', 'thrower:x'], 'the skill\'s next revision is back on');
});

test('a revision change disposes the old module before the new revision mounts; a late start is let go', async () => {
    const order = [];
    let release = null;
    const first = hostItem('clock:x');
    const second = hostItem('clock:x', { revision: 'r2', render: { kind: 'host', url: '/api/extensions/clock/ui/r2/ui/x.js' } });
    const slow = hostItem('slow:x');
    const h = await setup({ items: [first, slow], modules: {
        [first.render.url]: { mount: () => { order.push('mount r1'); return { dispose: () => order.push('dispose r1') }; } },
        [second.render.url]: { mount: () => { order.push('mount r2'); return { dispose: () => order.push('dispose r2') }; } },
        [slow.render.url]: { mount: () => new Promise((resolve) => { release = resolve; }) },
    } });
    await h.publish([second, slow]);
    assert.deepEqual(order, ['mount r1', 'dispose r1', 'mount r2']);
    await tick(80);
    assert.equal(h.store.fault('slow:x'), 'it did not start in time');
    let lateDisposed = 0;
    release({ dispose: () => { lateDisposed += 1; } });
    await tick();
    assert.equal(lateDisposed, 1, 'a module that starts after its bound is disposed when it lands');
});

test('a module may register only the client provider its contribution declares', async () => {
    const placed = hostItem('notes:panel', { placement: 'composer.plus' });
    const voice = hostItem('speech:stt', { provides: 'stt', runtime: 'client', destination: 'vendor:Google' });
    const h = await setup({ items: [placed, voice], modules: {
        [placed.render.url]: { mount(ctx) { ctx.providers.register('stt', { start() {} }); } },
        [voice.render.url]: { mount(ctx) { ctx.providers.register('stt', null); } },
    } });
    assert.equal(h.store.fault('notes:panel'), 'notes:panel declares no client stt provider to register');
    assert.equal(h.store.fault('speech:stt'), 'providers.register(kind, impl) needs the provider object');
    assert.deepEqual(h.runtime.clientProviders('stt'), []);
    const fine = hostItem('speech:stt', { provides: 'stt', runtime: 'client', destination: 'vendor:Google', revision: 'r2',
        render: { kind: 'host', url: '/api/extensions/speech/ui/r2/ui/stt.js' } });
    const impl = { label: 'Browser speech', available: () => ({ ok: true }), start() {} };
    let unregister = null;
    const h2 = await setup({ items: [fine], modules: { [fine.render.url]: { mount(ctx) {
        unregister = ctx.providers.register('stt', impl);
        return { dispose() {} };
    } } } });
    assert.deepEqual(h2.runtime.clientProviders('stt').map((entry) => [entry.key, entry.destination, entry.impl]),
        [['speech:stt', 'vendor:Google', impl]]);
    unregister();
    assert.deepEqual(h2.runtime.clientProviders('stt'), [], 'a module may withdraw its provider itself');
});

test('each start stands in the ledger before its import and goes a second after it settled', async () => {
    const one = hostItem('one:x');
    const two = hostItem('two:x');
    const h = await setup({ items: [one, two], modules: {
        [one.render.url]: { mount: () => ({ dispose() {} }) }, [two.render.url]: new Error('HTTP 404') } });
    assert.equal(h.ledgerAtImport.length, 2);
    assert.deepEqual(h.ledgerAtImport[0]['module:one:x'], { reason: 'start', rev: 'r1', at: 1_000_000, n: 1 },
        'written before the module\'s first byte runs');
    await tick(30);
    assert.deepEqual(ledger(h.local), {}, 'a start that mounted or failed with the page alive leaves nothing');
});

test('a module whose last start the page did not outlive is held, named once, and Start it mounts it', async () => {
    const one = hostItem('one:x', { skill_label: 'Device voice' });
    const two = hostItem('two:x');
    let mounted = 0;
    const modules = { [one.render.url]: { mount: () => { mounted += 1; return { dispose() {} }; } },
        [two.render.url]: { mount: () => ({ dispose() {} }) } };
    const local = seeded({ 'module:one:x': { reason: 'start', rev: 'r1', at: 1 }, 'module:two:x': { reason: 'start', rev: 'r0', at: 1 } });
    const h = await setup({ items: [one, two], modules, local });
    assert.deepEqual(h.imported, [two.render.url], 'a hold of an older revision is not this revision\'s');
    const [held] = h.notices;
    assert.equal(held.text, 'Device voice did not start: the page stopped while it was starting last time.');
    assert.deepEqual([held.tone, held.ttl, held.action.label], ['warn', 0, 'Start it']);
    await h.publish([one, two, { ...hostItem('plain:z'), placement: 'composer.plus', render: { kind: 'declarative', url: '' } }]);
    assert.equal(h.notices.length, 1, 'a lifecycle change neither mounts it nor names it again');
    held.action.run();
    await tick();
    assert.equal(mounted, 1);
    assert.deepEqual(h.runtime.mounted().sort(), ['one:x', 'two:x']);
    const fixed = hostItem('one:x', { revision: 'r2', render: { kind: 'host', url: '/api/extensions/one/ui/r2/ui/x.js' } });
    modules[fixed.render.url] = { mount: () => ({ dispose() {} }) };
    const updated = await setup({ items: [fixed], modules, local: seeded({ 'module:one:x': { reason: 'start', rev: 'r1', at: 1 } }) });
    assert.deepEqual(updated.runtime.mounted(), ['one:x'], 'the skill\'s next revision starts as usual');
    assert.deepEqual(updated.notices, []);
});

test('?hold=all holds every page module for that load, each released by Start it', async () => {
    const one = hostItem('one:x');
    const modules = { [one.render.url]: { mount: () => ({ dispose() {} }) } };
    const h = await setup({ items: [one], modules, href: 'http://127.0.0.1:8765/?hold=all#chat' });
    assert.deepEqual(h.imported, []);
    const [held] = h.notices;
    assert.equal(held.text, 'one did not start: this page was opened with everything held.');
    held.action.run();
    await tick();
    assert.deepEqual(h.runtime.mounted(), ['one:x']);
    const plain = await setup({ items: [one], modules });
    assert.deepEqual(plain.runtime.mounted(), ['one:x'], 'without the flag and with no hold, page code runs');
    assert.deepEqual(plain.notices, []);
});

test('Panic and an unload dispose every module and none mounts again; a page kept for back-forward keeps them', async () => {
    let disposed = 0;
    const one = hostItem('one:x');
    const modules = { [one.render.url]: { mount: () => ({ dispose: () => { disposed += 1; } }) } };
    const h = await setup({ items: [one], modules });
    h.win.emit('pagehide', { persisted: true });
    assert.deepEqual([disposed, h.runtime.mode()], [0, 'running']);
    h.win.emit('ouro:panic');
    assert.deepEqual([disposed, h.runtime.mode()], [1, 'stopped']);
    const next = hostItem('one:x', { revision: 'r2', render: { kind: 'host', url: '/api/extensions/one/ui/r2/ui/x.js' } });
    await h.publish([next]);
    assert.equal(h.imported.length, 1, 'after Panic nothing mounts again in this page');
    const u = await setup({ items: [one], modules });
    u.win.emit('pagehide', { persisted: false });
    assert.deepEqual([disposed, u.runtime.mode()], [2, 'stopped']);
    assert.deepEqual(ledger(u.local), {}, 'an orderly end leaves no hold');
    u.runtime.destroy();
    assert.equal(u.win.listenerCount() + u.doc.listenerCount(), 0, 'destroy releases the page listeners');
});

test('where local storage refuses a write the ledger lives in session storage', async () => {
    const one = hostItem('one:x');
    const h = await setup({ items: [one], local: storage({ refuse: true }),
        modules: { [one.render.url]: { mount: () => ({ dispose() {} }) } } });
    assert.ok(h.ledgerAtImport[0]['module:one:x'], 'the entry stood during the import');
    await tick(30);
    assert.deepEqual(ledger(h.session), {}, 'and went after it');
});

test('origin facts', () => {
    assert.deepEqual(originFacts({ location: { hostname: 'ouro.localhost' }, isSecureContext: true }), { secure: true, loopback: true });
    assert.deepEqual(originFacts({ location: { hostname: '192.168.1.20' }, isSecureContext: false }), { secure: false, loopback: false });
    assert.deepEqual(originFacts({ location: { hostname: '[::1]' } }), { secure: false, loopback: true });
});
