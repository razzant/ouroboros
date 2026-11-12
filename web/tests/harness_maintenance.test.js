import assert from 'node:assert/strict';
import test from 'node:test';
import {
    createHarnessMaintenanceController, harnessMaintenanceMarkup,
    maintenanceOperationLine, maintenanceOperationPending, maintenanceVersionLine,
} from '../modules/harness_maintenance.js';
import { harnessFamilyMarkup, serviceBannerLine } from '../modules/harness_accounts.js';
import {
    harnessMaintenanceInventory, startHarnessMaintenance,
    harnessMaintenanceOperation, cancelHarnessMaintenance,
} from '../modules/api_client.js';

const allowed = { available: true, reason: null, commands: [{kind: 'exec', command: '/vendor/program', args: ['update']}] };
const entry = (extra = {}) => ({
    harness: 'fixture', actions: { install: allowed, update: allowed, exactVersion: allowed },
    selection: { path: '/vendor/program', realPath: '/vendor/program', version: '2.0.0', versionOutput: '2.0.0', overrideEnv: null, overrideValue: null },
    newest: {status: 'not_read', source: null}, updateAvailable: 'cannot_tell', releaseNotesUrl: null,
    observedAt: '2026-10-01T12:00:00Z', operation: null, ...extra,
});
const operation = (extra = {}) => ({
    id: 'operation-1', harness: 'fixture', state: 'running', phase: 'installing',
    target: { kind: 'latest', version: null }, mutation: 'unknown', outcome: 'unknown',
    termination: 'not_applicable', ...extra,
});
const deferred = () => {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
};
const flush = async () => { for (let n = 0; n < 15; n++) await Promise.resolve(); };
function setup(overrides = {}) {
    const calls = [], intents = new Map();
    const api = {
        harnessMaintenanceInventory: async (query) => { calls.push(['inventory', query]); return { harnesses: [entry()] }; },
        startHarnessMaintenance: async (body, key) => { calls.push(['start', body, key]); return operation(); },
        harnessMaintenanceOperation: async (id) => { calls.push(['read', id]); return operation(); },
        cancelHarnessMaintenance: async (id) => { calls.push(['cancel', id]); return operation(); },
        ...overrides,
    };
    const controller = createHarnessMaintenanceController({ api, intents, newKey: () => 'same-key' });
    return { api, calls, intents, controller };
}

test('selected version is the answering program, including overrides and absence', () => {
    assert.equal(maintenanceVersionLine(entry()), 'Program 2.0.0');
    const unknown = entry({selection: {path: '/custom/bin', version: null}});
    assert.equal(maintenanceVersionLine(unknown), 'Program version unknown');
    assert.match(maintenanceVersionLine(entry(), {unknownCurrent: true}), /version unknown/);
    assert.match(maintenanceVersionLine(entry(), {inspectionError: 'offline'}), /Last known/);
    const missing = entry({selection: {path: null, version: null}});
    assert.equal(maintenanceVersionLine(missing), 'Not installed');
    assert.match(harnessMaintenanceMarkup(missing), /action="latest"[^>]*>Install</);
    assert.equal(maintenanceOperationLine({state: 'failed'}, {missing: true}), 'Install failed');
});

test('engine verdict and actions determine buttons for every program', () => {
    for (const harness of ['codex', 'claude', 'cursor', 'agy', 'future-harness']) {
        for (const updateAvailable of ['yes', 'no', 'cannot_tell']) {
            const html = harnessMaintenanceMarkup(entry({harness, updateAvailable,
                newest: {status: 'available', version: '1.0.0'}}));
            assert.equal(/action="latest"/.test(html), updateAvailable !== 'no', 'no host version ordering');
            assert.match(html, />Check latest</);
            assert.match(html, updateAvailable === 'no' ? /Up to date/ : updateAvailable === 'yes' ? /1.0.0 available/ : /Latest unknown/);
            assert.doesNotMatch(html, /Managed|External install|Bundled baseline|Release verification|Install exact version|Install release version/);
        }
        const html = harnessMaintenanceMarkup(entry({harness, actions: {update: {available: false, reason: 'Clear the override first.', commands: []}}}));
        assert.match(html, /Clear the override first/);
        assert.doesNotMatch(html, /action="latest"/);
    }
});

test('one maintenance host per family, independent of the account count', () => {
    for (const count of [1, 18]) {
        const group = {
            harness: 'fixture', label: 'Fixture', status: { tone: 'muted', label: 'Ready' },
            rows: Array.from({ length: count }, (_, id) => ({ harness: 'fixture', profile_id: `p${id}`, kind: 'profile' })),
        };
        const html = harnessFamilyMarkup(group, {});
        assert.equal((html.match(/data-family-maintenance=/g) || []).length, 1);
        assert.equal((html.match(/class="ui-row harness-account-row/g) || []).length, count);
        // The program's row stands above the accounts, in a card of its own.
        assert.ok(html.indexOf('data-family-maintenance=') < html.indexOf('agent-family-rows'));
    }
    assert.doesNotMatch(harnessFamilyMarkup({ harness: 'api-agent', label: 'API agent', rows: [], maintenanceOnly: true }, {}), /data-family-add/);
});

test('version details and failure/progress text are escaped, successful no-change is not called updated', () => {
    const html = harnessMaintenanceMarkup(entry({ selection: { path: '<img onerror=bad>' } }), {
        operation: operation({ problem: { message: '<script>bad()</script>' }, progress: ['<b>verifying</b>'] }),
    });
    assert.doesNotMatch(html, /<script>|<img |<b>/);
    assert.match(html, /&lt;script&gt;bad/);
    assert.match(html, /&lt;b&gt;verifying/);
    assert.equal(maintenanceOperationLine(operation({ state: 'succeeded', target: { kind: 'latest', version: null },
        mutation: 'applied', outcome: 'not_changed', after: { version: '2.0.0' } })),
    'Nothing changed');
});

test('named API helpers preserve keys, exact target and typed error receipt', async () => {
    const prior = globalThis.fetch;
    const calls = [];
    globalThis.fetch = async (url, init = {}) => {
        calls.push([url, init]);
        return { ok: true, json: async () => ({ id: 'same' }) };
    };
    try {
        await harnessMaintenanceInventory({ harness: 'a b', fresh: true, checkLatest: true });
        const body = { harness: 'fixture', target: { kind: 'version', version: '2.9.0' } };
        await startHarnessMaintenance(body, 'intent-1');
        await harnessMaintenanceOperation('operation/1');
        await cancelHarnessMaintenance('operation/1');
        assert.equal(calls[0][0], '/api/claudexor/maintenance/harnesses?harness=a+b&fresh=true&checkLatest=true');
        assert.equal(calls[1][1].headers['Idempotency-Key'], 'intent-1');
        assert.deepEqual(JSON.parse(calls[1][1].body), body);
        assert.equal(calls[2][0], '/api/claudexor/maintenance/operations/operation%2F1');
        assert.equal(calls[3][1].method, 'POST');
        globalThis.fetch = async () => ({ ok: false, status: 503,
            json: async () => ({ error: { code: 'capability_unavailable', message: 'Upgrade the engine.' } }) });
        await assert.rejects(harnessMaintenanceInventory(), (error) =>
            error.message === 'Upgrade the engine.' && error.body.error.code === 'capability_unavailable');
    } finally { globalThis.fetch = prior; }
});

test('old/unreachable engine has only inspection retry and never a legacy install', async () => {
    const { controller, calls } = setup({ harnessMaintenanceInventory: async () => { throw new Error('Maintenance not supported'); } });
    await controller.refresh();
    assert.match(controller.view('fixture').inventoryError, /Maintenance not supported/);
    await controller.start('fixture', { kind: 'latest' });
    assert.equal(calls.length, 0);
    assert.doesNotMatch(harnessMaintenanceMarkup(null, { inspectionError: controller.view('fixture').inventoryError }), /action="latest"/);
});

test('explicit inspection asks newest independently of update capability', async () => {
    const queries = [];
    const { controller } = setup({ harnessMaintenanceInventory: async (query) => {
        queries.push(query);
        return { harnesses: [entry()] };
    } });
    await controller.refresh(); await controller.act('fixture', 'inspect');
    assert.deepEqual(queries[1], { harness: 'fixture', fresh: true, checkLatest: true });
});

test('HTTP clients without secure-context randomUUID can still create a retained request key', async () => {
    const prior = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
    Object.defineProperty(globalThis, 'crypto', { value: {}, configurable: true });
    try {
        const { api, calls } = setup();
        const controller = createHarnessMaintenanceController({ api, intents: new Map() });
        await controller.refresh(); await controller.start('fixture', { kind: 'latest' });
        assert.match(calls.find(([kind]) => kind === 'start')[2], /^maintenance-[a-z0-9]+-[a-z0-9]+$/);
    } finally {
        if (prior) Object.defineProperty(globalThis, 'crypto', prior);
        else delete globalThis.crypto;
    }
});

test('double-click shares one promise and one accepted operation', async () => {
    const reply = deferred();
    let creates = 0;
    const { controller } = setup({ startHarnessMaintenance: async () => { creates++; return reply.promise; } });
    await controller.refresh();
    const a = controller.start('fixture', { kind: 'latest' });
    const b = controller.start('fixture', { kind: 'latest' });
    assert.equal(a, b);
    await flush();
    assert.equal(creates, 1);
    reply.resolve(operation());
    await a;
    assert.equal(controller.view('fixture').operation.id, 'operation-1');
});

test('lost POST reply retains exact key/body through remount and ignores unrelated old inventory operation', async () => {
    const requests = [];
    const { controller, intents, api } = setup({
        harnessMaintenanceInventory: async () => ({ harnesses: [entry({ operation: { id: 'old-op', state: 'succeeded' } })] }),
        harnessMaintenanceOperation: async () => operation({ id: 'old-op', state: 'succeeded', mutation: 'none' }),
        startHarnessMaintenance: async (body, key) => {
            requests.push({ body, key });
            if (requests.length === 1) throw new Error('Connection lost');
            return operation();
        },
    });
    await controller.refresh(); await flush();
    await controller.start('fixture', { kind: 'version', version: '3.1.0' });
    assert.equal(controller.view('fixture').unknownCurrent, true);
    assert.match(harnessMaintenanceMarkup(entry(), controller.view('fixture')), /Retry same request/);
    controller.dispose();
    const remount = createHarnessMaintenanceController({ api, intents, newKey: () => { throw new Error('must reuse'); } });
    await remount.refresh();
    assert.ok(remount.view('fixture').request, 'old retained result does not confirm this press');
    await remount.act('fixture', 'rejoin');
    assert.deepEqual(requests[0], requests[1]);
    assert.equal(remount.view('fixture').operation.id, 'operation-1');
});

test('a fresh view rejoins inventory operation and does not submit work', async () => {
    const { controller, calls } = setup({ harnessMaintenanceInventory: async () => ({ harnesses: [entry({ operation: operation() })] }) });
    await controller.refresh(); await flush();
    assert.equal(calls.filter(([kind]) => kind === 'read').length, 0, 'inventory already has the full receipt');
    assert.equal(calls.filter(([kind]) => kind === 'start').length, 0);
    assert.equal(controller.view('fixture').operation.phase, 'installing');
});

test('historical unknown effect keeps a newer inspected version and its warning', async () => {
    const terminal = operation({ state: 'failed', phase: 'settled', mutation: 'unknown',
        termination: 'confirmed', finishedAt: '2026-10-09T12:00:00Z' });
    const row = entry({ selection: { path: '/vendor/program', version: '3.1.0' }, observedAt: '2026-10-09T12:01:00Z',
        operation: terminal });
    let inspections = 0;
    const { controller, calls } = setup({
        harnessMaintenanceInventory: async () => { inspections++; return { harnesses: [row] }; },
        harnessMaintenanceOperation: async () => terminal,
    });
    await controller.refresh(); await flush();
    for (let read = 0; read < 2; read++) {
        const view = controller.view('fixture');
        assert.match(maintenanceVersionLine(view.entry, view), /Program 3\.1\.0/);
        assert.match(harnessMaintenanceMarkup(view.entry, view), /Update failed · Installed files may have changed/);
        assert.equal(view.operation.mutation, 'unknown');
        assert.equal(view.operation.termination, 'confirmed');
        assert.equal(inspections, 1, 'historical detail needs no second inspection to preserve the newer observation');
        await controller.readOperation('fixture');
    }
    assert.deepEqual(calls, [], 'no start, cancel or task action follows inspection');
});

test('historical unknown effect cannot borrow stale or absent version evidence', async () => {
    for (const [observedAt, version, finishedAt] of [
        ['2026-10-09T11:59:00Z', '2.0.0', '2026-10-09T12:00:00Z'],
        ['2026-10-09T12:00:00Z', '2.0.0', '2026-10-09T12:00:00Z'],
        [null, '2.0.0', '2026-10-09T12:00:00Z'],
        ['unreadable', '2.0.0', '2026-10-09T12:00:00Z'],
        ['2026-10-09T12:01:00Z', '2.0.0', null],
        ['2026-10-09T12:01:00Z', null, '2026-10-09T12:00:00Z'],
        ['2026-10-09T12:01:00Z', undefined, '2026-10-09T12:00:00Z'],
    ]) {
        const terminal = operation({ state: 'interrupted', phase: 'settled', mutation: 'unknown',
            termination: 'confirmed', finishedAt });
        const row = entry({ observedAt, selection: { path: '/vendor/program', version },
            operation: terminal });
        const { controller } = setup({ harnessMaintenanceInventory: async () => ({ harnesses: [row] }),
            harnessMaintenanceOperation: async () => terminal });
        await controller.refresh(); await flush();
        assert.match(maintenanceVersionLine(row, controller.view('fixture')), /Program version unknown/);
        await controller.refresh();
        assert.match(maintenanceVersionLine(row, controller.view('fixture')), /Program version unknown/,
            `re-reading the same physical evidence cannot prove a current version: ${observedAt} / ${finishedAt}`);
    }
});

test('fresh terminal version observation does not release uncertain operation custody', async () => {
    const terminal = operation({ state: 'cancelled', phase: 'settled', mutation: 'unknown',
        termination: 'unconfirmed', finishedAt: '2026-10-09T12:00:00Z' });
    const row = entry({ observedAt: '2026-10-09T12:01:00Z', selection: { path: '/vendor/program', version: '3.1.0' },
        operation: terminal });
    const { controller } = setup({ harnessMaintenanceInventory: async () => ({ harnesses: [row] }),
        harnessMaintenanceOperation: async () => terminal });
    await controller.refresh(); await flush();
    const view = controller.view('fixture');
    assert.match(maintenanceVersionLine(row, view), /Program 3\.1\.0/);
    assert.equal(maintenanceOperationPending(view.operation), true);
    const html = harnessMaintenanceMarkup(row, view);
    assert.match(html, /Installer may still be running/);
    assert.match(html, /data-maintenance-action="cancel"/);
    assert.match(html, /data-maintenance-action="latest" disabled/);
});

test('newer timestamps cannot unmask an in-flight install', async () => {
    const active = operation();
    const row = entry({ observedAt: '2026-10-09T12:01:00Z', operation: active });
    const { controller } = setup({ harnessMaintenanceInventory: async () => ({ harnesses: [row] }),
        harnessMaintenanceOperation: async () => active });
    await controller.refresh(); await flush();
    assert.equal(maintenanceOperationPending(controller.view('fixture').operation), true);
    assert.match(maintenanceVersionLine(row, controller.view('fixture')), /Program version unknown/);
});

test('active-operation conflict rejoins its typed handle', async () => {
    const { controller, calls } = setup({ startHarnessMaintenance: async () => {
        throw Object.assign(new Error('Already active'), { status: 409,
            body: { error: { code: 'maintenance_already_active', context: { operationId: 'operation-1' } } } });
    } });
    await controller.refresh();
    await controller.start('fixture', { kind: 'latest' });
    assert.equal(controller.view('fixture').operation.id, 'operation-1');
    assert.ok(calls.some(([kind]) => kind === 'read'));
    assert.equal(controller.view('fixture').request, null);
});

test('visible existing ticks poll active work once, stop on lost contact and allow explicit check', async () => {
    const intents = new Map();
    let visible = true, reads = 0, fail = true;
    const { api } = setup({ harnessMaintenanceOperation: async () => { reads++; if (fail) throw new Error('offline'); return operation(); } });
    const controller = createHarnessMaintenanceController({ api, intents, visible: () => visible, newKey: () => 'key' });
    await controller.refresh();
    controller.poll(); await flush(); assert.equal(reads, 0);
    await controller.start('fixture', { kind: 'latest' });
    visible = false; controller.poll(); await flush(); assert.equal(reads, 0);
    visible = true; controller.poll(); controller.poll(); await flush(); assert.equal(reads, 1);
    controller.poll(); await flush(); assert.equal(reads, 1);
    assert.match(controller.view('fixture').operationError, /unconfirmed/);
    fail = false; await controller.act('fixture', 'rejoin'); assert.equal(reads, 2);
    controller.poll(); await flush(); assert.equal(reads, 3);
    controller.dispose(); controller.poll(); await flush(); assert.equal(reads, 3);
});

test('cancel acknowledgement is not settlement or termination proof', async () => {
    const { controller } = setup();
    await controller.refresh(); await controller.start('fixture', { kind: 'latest' });
    await controller.cancel('fixture');
    assert.equal(controller.view('fixture').operation.state, 'running');
    assert.ok(controller.view('fixture').cancelling);
    const unconfirmed = operation({ state: 'cancelled', termination: 'unconfirmed' });
    assert.equal(maintenanceOperationPending(unconfirmed), true);
    assert.match(harnessMaintenanceMarkup(entry(), { operation: unconfirmed }), /Installer may still be running/);
    assert.match(harnessMaintenanceMarkup(entry(), { operation: unconfirmed }), /action="latest" disabled/);
    assert.equal(maintenanceOperationPending({ ...unconfirmed, termination: 'confirmed' }), false);
});

test('partial failure hides old version until fresh inventory, retains result and never retries task', async () => {
    let reads = 0, settles = 0;
    const inspect = deferred();
    const { api, intents } = setup({
        harnessMaintenanceInventory: async () => ++reads === 1 ? { harnesses: [entry()] } : inspect.promise,
        harnessMaintenanceOperation: async () => operation({ state: 'failed', phase: 'settled', mutation: 'unknown',
            termination: 'confirmed', finishedAt: '2026-10-09T12:00:00Z' }),
    });
    const controller = createHarnessMaintenanceController({ api, intents, newKey: () => 'key', onSettled: () => { settles++; } });
    await controller.refresh(); await controller.start('fixture', { kind: 'latest' });
    await controller.readOperation('fixture');
    assert.equal(controller.view('fixture').unknownCurrent, true);
    assert.match(maintenanceVersionLine(controller.view('fixture').entry, controller.view('fixture')), /version unknown/);
    assert.equal(settles, 1);
    inspect.resolve({ harnesses: [entry({ observedAt: '2026-10-09T12:01:00Z',
        selection: { path: '/vendor/program', version: '2.5.0' } })] });
    await flush();
    assert.equal(controller.view('fixture').unknownCurrent, false);
    assert.equal(controller.view('fixture').entry.selection.version, '2.5.0');
    assert.equal(controller.view('fixture').operation.state, 'failed');
});

test('an old running poll cannot undo a confirmed cancellation', async () => {
    const stale = deferred();
    const { controller } = setup({ harnessMaintenanceOperation: async () => stale.promise,
        cancelHarnessMaintenance: async () => operation({ state: 'cancelled', phase: 'settled', termination: 'confirmed' }) });
    await controller.refresh(); await controller.start('fixture', { kind: 'latest' });
    const read = controller.readOperation('fixture');
    await controller.cancel('fixture');
    stale.resolve(operation()); await read;
    assert.equal(controller.view('fixture').operation.state, 'cancelled');
    assert.equal(controller.view('fixture').cancelling, false);
});

test('an old inventory cannot erase a terminal receipt whose termination is still unconfirmed', async () => {
    const stale = deferred();
    let reads = 0, settled = 0;
    const failed = operation({state: 'failed', phase: 'settled', termination: 'unconfirmed', mutation: 'unknown'});
    const {api} = setup({harnessMaintenanceInventory: async () => ++reads === 2 ? stale.promise
        : {harnesses: [entry({operation: reads > 2 ? failed : null})]}, harnessMaintenanceOperation: async () => failed});
    const controller = createHarnessMaintenanceController({api, intents: new Map(), onSettled: () => settled++});
    await controller.refresh(); await controller.start('fixture', {kind: 'latest'});
    const read = controller.refresh(); await flush();
    await controller.readOperation('fixture');
    stale.resolve({harnesses: [entry({operation: operation()})]}); await read; await flush();
    assert.equal(controller.view('fixture').operation.state, 'failed');
    assert.equal(controller.view('fixture').operation.termination, 'unconfirmed');
    assert.equal(settled, 0);
});

test('an inspection begun before update cannot restore the old current version', async () => {
    let reads = 0;
    const stale = deferred();
    const { controller } = setup({ harnessMaintenanceInventory: async () => {
        reads++;
        if (reads === 2) return stale.promise;
        return { harnesses: [entry({ selection: { path: '/vendor/program', version: reads > 2 ? '3.0.0' : '2.0.0' } })] };
    } });
    await controller.refresh();
    const read = controller.refresh(); await flush();
    await controller.start('fixture', { kind: 'latest' });
    stale.resolve({ harnesses: [entry()] }); await read; await flush();
    assert.equal(controller.view('fixture').entry.selection.version, '3.0.0');
    assert.equal(controller.view('fixture').unknownCurrent, true, 'active replacement still stays unknown');
});

test('Return uses only the changed receipt before.version and the declared exact action', async () => {
    const receipt = operation({state: 'succeeded', outcome: 'changed', mutation: 'applied', before: {version: '1.7.0'}});
    const {api, calls} = setup({harnessMaintenanceInventory: async () => ({harnesses: [entry({operation: receipt})]}),
        harnessMaintenanceOperation: async () => receipt});
    const controller = createHarnessMaintenanceController({api, intents: new Map()});
    await controller.refresh(); await flush(); await controller.act('fixture', 'return');
    assert.deepEqual(calls.find(([kind]) => kind === 'start')[1], {harness: 'fixture', target: {kind: 'version', version: '1.7.0'}});
    assert.match(harnessMaintenanceMarkup(entry({operation: receipt})), /Return to 1.7.0/);
    for (const outcome of ['not_changed', 'changed_elsewhere', 'unknown']) {
        assert.doesNotMatch(harnessMaintenanceMarkup(entry({operation: {...receipt, outcome}})), /Return to/);
    }
    assert.doesNotMatch(harnessMaintenanceMarkup(entry({operation: receipt, actions: {exactVersion: {available: false}}})), /Return to/);
});

test('disposed view retains late acceptance without rendering or refreshing unrelated state', async () => {
    const pending = deferred(); let settles = 0;
    const { api, intents, calls } = setup({ startHarnessMaintenance: async () => pending.promise });
    const controller = createHarnessMaintenanceController({ api, intents, newKey: () => 'key', onSettled: () => { settles++; } });
    await controller.refresh(); const request = controller.start('fixture', { kind: 'latest' });
    controller.dispose(); pending.resolve(operation({ state: 'succeeded', mutation: 'applied' })); await request;
    assert.equal(settles, 0);
    assert.equal(intents.get('fixture').operation.state, 'succeeded');
    assert.equal(calls.filter(([kind]) => kind === 'inventory').length, 1);
});


function acquisitionError(status = 409) {
    return Object.assign(new Error('Could not acquire the engine'), {
        status, body: { error: { code: 'daemon_spawn_failed', context: {
            acceptance: 'not_submitted', scope: 'this_attempt',
        } } },
    });
}

test('a local pre-submit response cannot clear an earlier lost POST or change its key/body', async () => {
    const sent = [];
    const { controller, intents, api } = setup({ startHarnessMaintenance: async (body, key) => {
        sent.push({ body, key });
        if (sent.length === 1) throw new Error('Lost POST');
        if (sent.length === 2) throw acquisitionError();
        return operation();
    } });
    await controller.refresh();
    await controller.start('fixture', { kind: 'latest' });
    await controller.act('fixture', 'rejoin');
    assert.ok(controller.view('fixture').request);
    assert.match(controller.view('fixture').operationError, /acceptance unconfirmed/);
    controller.dispose();
    const resumed = createHarnessMaintenanceController({ api, intents, newKey: () => { throw new Error('must retain'); } });
    await resumed.refresh();
    await resumed.act('fixture', 'rejoin');
    assert.deepEqual(sent, [sent[0], sent[0], sent[0]]);
    assert.equal(resumed.view('fixture').operation.id, 'operation-1');
});

test('a host-observed pre-submit transport failure is not unknown acceptance on a first press', async () => {
    const { controller } = setup({ startHarnessMaintenance: async () => { throw acquisitionError(503); } });
    await controller.refresh();
    await controller.start('fixture', { kind: 'latest' });
    assert.equal(controller.view('fixture').request, null);
    assert.notEqual(controller.view('fixture').unknownCurrent, true);
});

for (const code of ['daemon_spawn_failed', 'daemon_starting', 'foreign_daemon_home', 'attach_only_engine']) {
    for (const lostPost of [false, true]) {
        test(`${code}: acquisition refusal names no update, but preserves an earlier unknown POST`, async () => {
            const sent = [];
            const { controller } = setup({ startHarnessMaintenance: async (body, key) => {
                sent.push({ body, key });
                if (lostPost && sent.length === 1) throw new Error('Lost POST');
                throw Object.assign(new Error('cannot acquire gateway'), { status: 503, body: { error: {
                    code, context: { acceptance: 'not_submitted', scope: 'this_attempt' },
                } } });
            } });
            await controller.refresh();
            await controller.start('fixture', { kind: 'latest' });
            if (lostPost) await controller.act('fixture', 'rejoin');
            const view = controller.view('fixture');
            if (lostPost) {
                assert.ok(view.request);
                assert.match(view.operationError, /^Update acceptance unconfirmed:/);
                assert.deepEqual(sent[1], sent[0], 'rejoin keeps the unknown POST key and body');
            } else {
                assert.equal(view.request, null);
                assert.notEqual(view.unknownCurrent, true);
                assert.match(view.operationError, /^Update not started: cannot acquire gateway$/);
                assert.match(harnessMaintenanceMarkup(entry(), view), /Update not started: cannot acquire gateway/);
            }
        });
    }
}
// Refined A (owner 21.10): the program row says its state concisely — the operation status, a short cause only where
// a typed source names it, and this view's own actionable sentence — while the engine's exact text, what an update
// does and the version facts sit under one Details toggle whose open state the controller keeps across rebuilds.
test('details carry actual commands once, script evidence, limitations and no internal IDs', () => {
    const receipt = operation({state: 'succeeded', outcome: 'changed', mutation: 'applied', exitCode: 7,
        commands: [{binary: '/vendor/program', args: ['update'], env: {}, script: {url: 'https://vendor.test/install', size: 123, sha256: 'abcdef'}}],
        before: {version: '1.0.0'}, after: {version: '2.0.0'},
        limitations: ['Installer exited 7; the replacement answers its version.'], progress: ['Vendor finished.']});
    const html = harnessMaintenanceMarkup(entry({operation: receipt, releaseNotesUrl: 'https://vendor.test/notes'}));
    assert.doesNotMatch(html, /<dt>Ran|operation-1/);
    for (const text of ['123 bytes', 'abcdef', 'Installer exited 7', 'Vendor finished.', 'Exit 7', 'https://vendor.test/notes']) assert.ok(html.includes(text));
    assert.match(html, /Updated from 1.0.0/);
    assert.match(html, /target="_blank" rel="noopener noreferrer" data-skill-external/);
    assert.doesNotMatch(harnessMaintenanceMarkup(entry()), /Release notes|data-skill-external/);
    const different = {...receipt, commands: [{binary: '/other/program', args: ['update'], env: {}}]};
    assert.match(harnessMaintenanceMarkup(entry({operation: different})), /<dt>Ran/);
    const latest = harnessMaintenanceMarkup(entry({newest: {status: 'unavailable', problem: {message: 'Pointer unreadable'}}}));
    assert.match(latest, /Latest unknown/);
    assert.match(latest, /Pointer unreadable/);
    assert.match(latest, /action="latest"/);
});

// ---- A current inspection whose reported executable is gone (#2461) ----
const MISSING = '/Applications/Moved.app/Contents/Frameworks/node-standalone/bin/node';
const spawnError = (extra = {}) => Object.assign(new Error(`spawn ${MISSING} ENOENT`), { status: 500, body: { error: {
    code: 'ENOENT', message: `spawn ${MISSING} ENOENT`, retryable: false, requiredActions: [],
    context: { observation: 'reported_executable_missing', observedAt: '2026-11-04T01:30:00Z', engineVersion: '5.0.1',
        engineBuildSha: 'f10efc2f', remedy: 'host words for tools' }, ...extra,
} } });
const OBSERVED = '2026-10-09T12:00:00Z';
const readyStore = () => ({
    snapshot: { daemon: { state: 'running', engine_version: '5.0.1', runtime: { state: 'ready' } } },
    reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' }, everSettled: true,
});
// Answers `failing` to a current read, and otherwise the daemon's cache, like the captured pinned engine.
function engine() {
    const world = { failing: true, reads: [] };
    world.api = {
        harnessMaintenanceInventory: async (query) => {
            world.reads.push(query);
            if ((query.fresh || query.checkLatest) && world.failing) throw spawnError();
            return { observedAt: '2026-11-04T01:31:00Z', harnesses: [entry({ observedAt: OBSERVED })] };
        },
    };
    return world;
}

test('a failed fresh inspection stays one dated section fact, through a cached read and a remount, until a real fresh success', async () => {
    const world = engine(), intents = new Map();
    let bannerPaints = 0;
    const controller = createHarnessMaintenanceController({ api: world.api, intents, onInspectionProblem: () => { bannerPaints++; } });
    await controller.refresh();
    assert.equal(controller.inspectionProblem, null);
    await controller.refresh({ fresh: true });
    assert.equal(bannerPaints, 1);
    assert.equal(controller.inspectionProblem.message, `spawn ${MISSING} ENOENT`);
    await controller.refresh();
    assert.equal(controller.inspectionProblem?.code, 'ENOENT', 'a cached success proves nothing current');
    assert.ok(controller.view('fixture').inventoryError);
    controller.dispose();
    const before = world.reads.length;
    const remount = createHarnessMaintenanceController({ api: world.api, intents });
    assert.equal(remount.view('fixture').entry.observedAt, OBSERVED, 'remount keeps the dated last observation without a read');
    assert.equal(remount.inspectionProblem?.code, 'ENOENT');
    assert.equal(world.reads.length, before);
    await remount.refresh();
    assert.equal(remount.inspectionProblem?.code, 'ENOENT', 'survives the remount and its first cached read');
    world.failing = false;
    await remount.refresh({ fresh: true });
    assert.equal(remount.inspectionProblem, null);
    assert.equal(remount.view('fixture').inventoryError, '');
    assert.equal(remount.view('fixture').entry.observedAt, OBSERVED);
});

test('any failed fresh inspection outlives a cached read; a failed cached read never outlives one', async () => {
    let answer = () => { throw new Error('child timed out'); };
    const { controller } = setup({ harnessMaintenanceInventory: async () => answer() });
    await controller.refresh({ harness: 'fixture', fresh: true });
    assert.match(controller.view('fixture').inspectionError, /child timed out/);
    assert.equal(controller.inspectionProblem, null, 'an unclassified failure earns no recovery advice');
    answer = () => { throw new Error('cached read lost'); };
    await controller.refresh({ harness: 'fixture' });
    assert.match(controller.view('fixture').inspectionError, /child timed out/);
    answer = () => ({ harnesses: [entry()] });
    await controller.refresh({ harness: 'fixture' });
    assert.match(controller.view('fixture').inspectionError, /child timed out/);
    await controller.refresh({ harness: 'fixture', fresh: true });
    assert.equal(controller.view('fixture').inspectionError, '');
    answer = () => { throw new Error('cached read lost'); };
    await controller.refresh({ harness: 'fixture' });
    answer = () => ({ harnesses: [entry()] });
    await controller.refresh({ harness: 'fixture' });
    assert.equal(controller.view('fixture').inspectionError, '', 'a cached failure is still healed by a cached answer');
});

test('an empty fresh answer spawned nothing and clears nothing; a later generic fresh failure replaces the advice', async () => {
    const world = engine();
    const controller = createHarnessMaintenanceController({ api: world.api, intents: new Map() });
    await controller.refresh({ fresh: true });
    world.api.harnessMaintenanceInventory = async () => ({ harnesses: [] });
    await controller.refresh({ fresh: true });
    assert.equal(controller.inspectionProblem?.code, 'ENOENT');
    world.api.harnessMaintenanceInventory = async () => { throw new Error('engine stopped'); };
    await controller.refresh({ fresh: true });
    assert.equal(controller.inspectionProblem, null, 'the newest current read did not observe a missing file');
    assert.match(controller.view('fixture').inventoryError, /engine stopped/);
});

test('one family press that proves a child can spawn clears the engine-wide fact, not an unrelated failure', async () => {
    const world = engine();
    const controller = createHarnessMaintenanceController({ api: world.api, intents: new Map() });
    await controller.refresh({ harness: 'other', fresh: true });
    assert.equal(controller.inspectionProblem?.code, 'ENOENT');
    world.failing = false;
    await controller.refresh({ harness: 'fixture', fresh: true });
    assert.equal(controller.inspectionProblem, null);
    assert.equal(controller.view('other').inspectionError, '');
});

test('a fresh request cannot become success by joining a cached read in flight', async () => {
    const cached = deferred(), asked = [];
    const { controller } = setup({ harnessMaintenanceInventory: async (query) => {
        asked.push(query);
        if (!query.fresh) return cached.promise;
        throw spawnError();
    } });
    const first = controller.refresh();
    const stronger = controller.refresh({ fresh: true });
    assert.notEqual(first, stronger);
    assert.equal(controller.refresh(), first, 'an equal or weaker read still joins');
    await flush();
    assert.equal(asked.length, 1);
    cached.resolve({ harnesses: [entry()] });
    await stronger;
    assert.deepEqual(asked.map((query) => query.fresh), [false, true]);
    assert.equal(controller.inspectionProblem?.code, 'ENOENT');
});

test('a latest check does not join a fresh read that will not check latest; a fresh one joins it', async () => {
    const running = deferred(), asked = [];
    const { controller } = setup({ harnessMaintenanceInventory: async (query) => {
        asked.push(query);
        return asked.length === 1 ? running.promise : { harnesses: [entry()] };
    } });
    const fresh = controller.refresh({ harness: 'fixture', fresh: true });
    assert.equal(controller.refresh({ harness: 'fixture', fresh: true }), fresh);
    const latest = controller.refresh({ harness: 'fixture', fresh: true, checkLatest: true });
    running.resolve({ harnesses: [entry()] });
    await latest;
    assert.deepEqual(asked.map((query) => query.checkLatest), [false, true]);
});

test('a released view leaves a late failed answer to its successor', async () => {
    const late = deferred(), intents = new Map();
    const api = { harnessMaintenanceInventory: async () => late.promise };
    const controller = createHarnessMaintenanceController({ api, intents });
    const read = controller.refresh({ fresh: true });
    controller.dispose();
    late.reject(spawnError());
    await read;
    assert.equal(createHarnessMaintenanceController({ api, intents }).inspectionProblem, null);
});

test('the row says Last known with the row\'s own date, and no paragraph or diagnostic repeats per family', async () => {
    const world = engine();
    const controller = createHarnessMaintenanceController({ api: world.api, intents: new Map() });
    await controller.refresh();
    await controller.refresh({ fresh: true });
    const view = controller.view('fixture');
    const row = entry({ observedAt: OBSERVED });
    const line = maintenanceVersionLine(row, { inspectionError: view.inventoryError });
    assert.match(line, /Program 2\.0\.0 · Last known /);
    assert.match(line, /2026/);
    assert.doesNotMatch(line, /T12:00:00Z|2026-11/, 'human-readable, and never the top inventory time');
    assert.equal(maintenanceVersionLine(entry({observedAt: null}), { inspectionError: 'offline' }).endsWith('· Last known'), true);
    const html = harnessMaintenanceMarkup(row, { inspectionProblem: controller.inspectionProblem, inspectionError: view.inventoryError });
    assert.doesNotMatch(html, /ENOENT|no longer present|Restart|Program inspection unavailable/);
    assert.match(html, /data-maintenance-action="inspect"/, 'the current check stays available');
    assert.doesNotMatch(html, /data-maintenance-action="latest"/, 'a failed inspection offers no update, as before');
});

test('an unclassified failure keeps its exact sentence on its own row', () => {
    const html = harnessMaintenanceMarkup(entry(), { inspectionError: 'Program inspection unavailable: spawn node ENOENT' });
    assert.match(html, /Program inspection unavailable: spawn node ENOENT/);
});

test('the banner explains once, conditionally, with the exact text under Diagnostics', () => {
    const problem = spawnError().body.error;
    const line = serviceBannerLine(readyStore(), { programProblem: problem });
    assert.equal(line.tone, 'warn');
    assert.match(line.text, /^Claudexor running · The executable reported by the last program inspection is no longer present\./);
    assert.match(line.text, /Versions below are last known\./);
    assert.match(line.text, /If the app was moved or replaced, Restart Claudexor in Accounts may help; it ends every run on the engine, and success is not guaranteed\./);
    assert.doesNotMatch(line.text, /chat header/, 'the header Restart keeps the engine');
    assert.doesNotMatch(line.text, /ENOENT|Moved\.app/);
    const facts = Object.fromEntries(line.facts);
    assert.match(facts['Last program inspection'], new RegExp(`ENOENT: spawn ${MISSING} ENOENT$`));
    assert.match(facts['Last program inspection'], /2026/);
    assert.equal(facts['Inspected by engine'], '5.0.1 · f10efc2f');
    assert.equal(serviceBannerLine(readyStore()).text, 'Claudexor ready');
    assert.deepEqual(serviceBannerLine({}, { wakeBusy: true, programProblem: problem }), { tone: 'muted', text: 'Starting the agent daemon…' });
    assert.match(serviceBannerLine(readyStore(), { wakeError: 'refused', programProblem: problem }).text, /^Could not start/);
});

test('a retained failed operation after a healthy replacement advises nothing and keeps its custody facts', async () => {
    const failed = operation({ state: 'failed', phase: 'settled', mutation: 'unknown', termination: 'unconfirmed',
        finishedAt: '2026-10-09T12:00:00Z', problem: { code: 'ENOENT', message: `spawn ${MISSING} ENOENT`, retryable: false } });
    // The replacement fixture engine answers healthy, its row still pointing at the retained failure.
    const row = entry({ observedAt: '2026-10-09T12:05:00Z', selection: { path: '/vendor/program', version: '3.1.0' },
        operation: failed });
    const asked = [], keys = [];
    let settles = 0;
    const api = {
        harnessMaintenanceInventory: async (query) => {
            asked.push(query);
            return { harnesses: [asked.length === 1 ? entry() : row] };
        },
        startHarnessMaintenance: async (_body, key) => { keys.push(key); return operation(); },
        harnessMaintenanceOperation: async () => failed,
    };
    const controller = createHarnessMaintenanceController({ api, intents: new Map(), newKey: () => 'same-key', onSettled: () => { settles++; } });
    await controller.refresh();
    await controller.start('fixture', { kind: 'latest' });
    await controller.readOperation('fixture');
    await controller.refresh({harness: 'fixture', fresh: true}); await flush();
    assert.deepEqual(keys, ['same-key']);
    const view = controller.view('fixture');
    assert.equal(controller.inspectionProblem, null);
    assert.equal(settles, 0, 'unconfirmed termination retains custody');
    assert.deepEqual(asked.at(-1), { harness: 'fixture', fresh: true, checkLatest: false }, 'the existing settled read supplies current evidence');
    assert.equal(view.operation.mutation, 'unknown');
    assert.equal(view.operation.termination, 'unconfirmed');
    const html = harnessMaintenanceMarkup(row, view);
    assert.match(html, /Update failed · Installer may still be running/);
    assert.match(html, /Exact message: <code>spawn .* ENOENT<\/code>/);
    assert.doesNotMatch(html, /Restart|no longer present/);
    assert.doesNotMatch(serviceBannerLine(readyStore(), { programProblem: controller.inspectionProblem }).text, /Restart/);
});

test('a retained failed operation with the engine still broken keeps its history beside the current fact', async () => {
    const failed = operation({ state: 'failed', phase: 'settled', mutation: 'none', termination: 'confirmed',
        finishedAt: '2026-10-09T12:00:00Z', problem: { code: 'ENOENT', message: `spawn ${MISSING} ENOENT` } });
    let reads = 0;
    const api = {
        harnessMaintenanceInventory: async () => {
            if (++reads > 1) throw spawnError();
            return { harnesses: [entry({ observedAt: OBSERVED })] };
        },
        startHarnessMaintenance: async () => operation(),
        harnessMaintenanceOperation: async () => failed,
    };
    const controller = createHarnessMaintenanceController({ api, intents: new Map(), newKey: () => 'key' });
    await controller.refresh();
    await controller.start('fixture', { kind: 'latest' });
    assert.equal(controller.inspectionProblem, null);
    await controller.readOperation('fixture'); await flush();
    assert.equal(controller.inspectionProblem?.code, 'ENOENT', 'the current read, not the old problem, is the source');
    assert.equal(controller.view('fixture').operation.problem.message, `spawn ${MISSING} ENOENT`);
});

test('a lost POST keeps its key and its retry through a failed fresh inspection', async () => {
    const sent = [];
    const world = engine();
    world.failing = false;
    const api = { ...world.api, startHarnessMaintenance: async (body, key) => {
        sent.push({ body, key });
        if (sent.length === 1) throw new Error('Lost POST');
        return operation();
    } };
    const controller = createHarnessMaintenanceController({ api, intents: new Map(), newKey: () => 'same-key' });
    await controller.refresh();
    await controller.start('fixture', { kind: 'latest' });
    world.failing = true;
    await controller.refresh({ harness: 'fixture', fresh: true });
    const view = controller.view('fixture');
    assert.match(harnessMaintenanceMarkup(view.entry, { ...view, inspectionProblem: controller.inspectionProblem }), /Retry same request/);
    await controller.act('fixture', 'rejoin');
    assert.deepEqual(sent, [sent[0], sent[0]]);
    assert.equal(controller.view('fixture').operation.id, 'operation-1');
});

// ---- Answers that cross (#2461, round 2): the later-asked read stands, whichever answers last ----
test('a released view\'s late answer, success or failure, never changes what its successor read', async () => {
    for (const successorFails of [true, false]) {
        const late = deferred(), intents = new Map();
        let answer = () => late.promise;
        const api = { harnessMaintenanceInventory: async () => answer() };
        const released = createHarnessMaintenanceController({ api, intents });
        const read = released.refresh({ fresh: true });
        await flush();
        released.dispose();
        answer = async () => {
            if (successorFails) throw spawnError();
            return { harnesses: [entry({ observedAt: OBSERVED })] };
        };
        const successor = createHarnessMaintenanceController({ api, intents });
        await successor.refresh({ fresh: true });
        const reading = () => ({ problem: successor.inspectionProblem?.code || null,
            error: successor.view('fixture').inventoryError });
        const seen = reading();
        assert.equal(seen.problem, successorFails ? 'ENOENT' : null);
        if (successorFails) late.resolve({ harnesses: [entry({ observedAt: OBSERVED })] });
        else late.reject(spawnError());
        await read; await flush();
        assert.deepEqual(reading(), seen, `the released view's late ${successorFails ? 'success' : 'failure'} is older`);
    }
});

const answerOf = {
    ok: () => ({ harnesses: [entry({ observedAt: OBSERVED })] }),
    missing: () => spawnError(),
    timeout: () => Object.assign(new Error('inspection child timed out'), { status: 502 }),
    offline: () => new Error('offline'),
};
// Asks a whole and a family read in the given order, answers them in order or reversed, and returns what
// the section and the fixture row then say.
async function crossing({ first, whole, family, cachedWhole = false, reversed }) {
    const pending = [deferred(), deferred()];
    let asked = 0;
    const controller = createHarnessMaintenanceController({ intents: new Map(),
        api: { harnessMaintenanceInventory: async () => pending[asked++].promise } });
    const scopes = first === 'whole' ? ['whole', 'family'] : ['family', 'whole'];
    const reads = scopes.map((scope) => controller.refresh(scope === 'family'
        ? { harness: 'fixture', fresh: true } : { fresh: !cachedWhole }));
    await flush();
    assert.equal(asked, 2, 'both reads are asked before either answers');
    for (const index of reversed ? [1, 0] : [0, 1]) {
        const answer = answerOf[scopes[index] === 'whole' ? whole : family]();
        if (answer instanceof Error) pending[index].reject(answer);
        else pending[index].resolve(answer);
        await flush();
    }
    await Promise.all(reads);
    const view = controller.view('fixture');
    return { problem: controller.inspectionProblem?.code || null,
        standing: view.inspectionError || view.inventoryError || controller.inspectionProblem?.message || '' };
}

for (const first of ['whole', 'family']) {
    for (const failure of ['missing', 'timeout']) {
        test(`crossing current reads, ${first} asked first, ${failure}: failure→success and success→failure end as asked`, async () => {
            const later = first === 'whole' ? 'family' : 'whole';
            for (const [older, newer] of [[failure, 'ok'], ['ok', failure]]) {
                const label = `${first} ${older}, then ${later} ${newer}`;
                const answers = { [first]: older, [later]: newer };
                const inOrder = await crossing({ first, ...answers, reversed: false });
                const crossed = await crossing({ first, ...answers, reversed: true });
                assert.deepEqual(crossed, inOrder, `${label}: arrival order decides nothing`);
                if (newer === 'ok') assert.deepEqual(crossed, { problem: null, standing: '' }, label);
                else {
                    assert.equal(crossed.problem, failure === 'missing' ? 'ENOENT' : null, label);
                    assert.match(crossed.standing, failure === 'missing' ? /ENOENT/ : /timed out/, label);
                }
            }
        });
    }
}

test('the cached whole read asked on mount crosses a family check without arrival deciding', async () => {
    const offline = 'Program inspection unavailable: offline';
    const missingLine = `Program inspection unavailable: spawn ${MISSING} ENOENT`;
    for (const [first, whole, family, expected] of [
        ['whole', 'offline', 'ok', { problem: null, standing: '' }],
        ['family', 'offline', 'ok', { problem: null, standing: offline }],
        ['whole', 'missing', 'ok', { problem: null, standing: '' }],
        ['family', 'missing', 'ok', { problem: 'ENOENT', standing: missingLine }],
        // A cached success proves nothing current: a fresh failure stands whichever was asked later.
        ['whole', 'ok', 'missing', { problem: 'ENOENT', standing: missingLine }],
        ['family', 'ok', 'missing', { problem: 'ENOENT', standing: missingLine }],
    ]) {
        for (const reversed of [false, true]) {
            assert.deepEqual(await crossing({ first, whole, family, cachedWhole: true, reversed }), expected,
                `${first} asked first, whole ${whole}, family ${family}, ${reversed ? 'reversed' : 'in order'}`);
        }
    }
});

test('Checking… answers the check the owner asked for, never the cached read a mount starts', async () => {
    const reads = [];
    const { controller } = setup({ harnessMaintenanceInventory: (query) => {
        const pending = deferred();
        reads.push({ query, pending });
        return pending.promise;
    } });
    const mount = controller.refresh();
    assert.equal(controller.view('fixture').inspectionPending, false, 'a cached read of every program relabels no card');
    await flush();
    reads[0].pending.resolve({ harnesses: [entry()] });
    await mount;
    const own = controller.act('fixture', 'inspect');
    assert.equal(controller.view('fixture').inspectionPending, true, "the pressed card's own check");
    assert.equal(controller.view('other').inspectionPending, false, 'a neighbour keeps its words');
    await flush();
    reads[1].pending.resolve({ harnesses: [entry()] });
    await own;
    const all = controller.refresh({ fresh: true });
    assert.equal(controller.view('other').inspectionPending, true, 'Refresh checks every program');
    await flush();
    reads[2].pending.resolve({ harnesses: [entry()] });
    await all;
    assert.equal(controller.view('fixture').inspectionPending, false);
    controller.dispose();
});

test('a program read the transport lost is said in owner words; an engine sentence stays as it is', async () => {
    const { controller } = setup({ harnessMaintenanceInventory: async () => { throw new TypeError('Load failed'); } });
    await controller.refresh({ fresh: true });
    assert.equal(controller.view('fixture').inventoryError, "Program inspection unavailable: Couldn't reach Ouroboros.");
    assert.doesNotMatch(controller.view('fixture').inventoryError, /Load failed/);
    const engine = setup({ harnessMaintenanceInventory: async () => { throw Object.assign(new Error('child timed out'), { status: 502 }); } });
    await engine.controller.refresh({ fresh: true });
    assert.equal(engine.controller.view('fixture').inventoryError, 'Program inspection unavailable: child timed out');
    controller.dispose(); engine.controller.dispose();
});

test('an external settled inventory receipt refreshes catalogs once; unconfirmed custody waits', async () => {
    const settled = operation({state: 'succeeded', outcome: 'not_changed', phase: 'settled', mutation: 'none', termination: 'confirmed'});
    let receipt = {...settled, termination: 'unconfirmed'};
    let settles = 0;
    const controller = createHarnessMaintenanceController({intents: new Map(), onSettled: () => settles++, api: {
        harnessMaintenanceInventory: async () => ({harnesses: [entry({operation: receipt})]}),
    }});
    await controller.refresh(); assert.equal(settles, 0);
    receipt = settled;
    await controller.refresh(); assert.equal(settles, 1);
    await controller.refresh(); assert.equal(settles, 1);
});

for (const prior of ['history', 'pending', 'empty', 'different']) {
    test(`seven programs: ${prior} receipts cause at most one catalog refresh per inventory`, async () => {
        let stage = 0;
        const settled = [];
        const receipt = (harness, n) => operation({harness, id: `retained-${n}`,
            state: 'succeeded', termination: 'confirmed', mutation: 'applied'});
        const rows = () => Array.from({length: 7}, (_, n) => {
            const harness = `program-${n}`;
            let op = receipt(harness, n);
            if (!stage && prior === 'pending') op = {...op, state: 'running'};
            if (!stage && prior === 'empty') op = null;
            if (!stage && prior === 'different') op = {...op, id: `older-${n}`};
            return entry({harness, operation: op});
        });
        const controller = createHarnessMaintenanceController({intents: new Map(),
            api: {harnessMaintenanceInventory: async () => ({harnesses: rows()})},
            onSettled: (...args) => settled.push(args)});
        await controller.refresh();
        assert.equal(settled.length, 0, 'first settled receipts are history');
        stage++;
        await controller.refresh();
        assert.equal(settled.length, prior === 'history' ? 0 : 1);
        await controller.refresh();
        assert.equal(settled.length, prior === 'history' ? 0 : 1, 'unchanged receipts stay quiet');
        controller.dispose();
    });
}

for (const reason of ['Update the app or connect an engine at version 6.0.0 or newer to install or update harness programs.', 'Engine offline']) test(`inspection refusal is introduced once: ${reason}`, async () => {
    const {controller} = setup({harnessMaintenanceInventory: async () => { throw new Error(reason); }});
    await controller.refresh();
    const message = controller.view('fixture').inventoryError;
    assert.equal((message.match(/Program inspection unavailable/g) || []).length, 1);
    assert.equal(message, `Program inspection unavailable: ${reason}`);
    controller.dispose();
});
