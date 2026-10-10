import assert from 'node:assert/strict';
import test from 'node:test';
import {
    createHarnessMaintenanceController, harnessMaintenanceMarkup,
    maintenanceOperationLine, maintenanceOperationPending, maintenanceVersionLine,
} from '../modules/harness_maintenance.js';
import { harnessFamilyMarkup } from '../modules/harness_accounts.js';
import {
    harnessMaintenanceInventory, startHarnessMaintenance,
    harnessMaintenanceOperation, cancelHarnessMaintenance,
} from '../modules/api_client.js';

const entry = (extra = {}) => ({
    harness: 'fixture', maintainable: true, canCheckLatest: true, targets: ['latest', 'version', 'previous', 'baseline'],
    selection: { kind: 'managed', binary: '/managed/program', version: '2.0.0' },
    installed: { version: '2.0.0', binary: '/managed/program', proved: true },
    releaseTested: { version: '1.0.0', verification: 'deterministic_only' },
    available: null, previous: { version: '1.8.0', operationId: 'previous-op' },
    operation: null, ...extra,
});
const operation = (extra = {}) => ({
    id: 'operation-1', harness: 'fixture', state: 'running', phase: 'installing',
    target: { kind: 'latest', version: '3.0.0' }, mutation: 'unknown',
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

test('selected version and release baseline stay independent, including external selection', () => {
    assert.equal(maintenanceVersionLine(entry()), 'Program 2.0.0 · Managed · Latest not checked');
    const external = entry({ selection: { kind: 'override', binary: '/custom/bin', overrideEnv: 'CUSTOM_BIN' } });
    assert.match(maintenanceVersionLine(external), /version unknown · External install/);
    assert.doesNotMatch(maintenanceVersionLine(external), /2\.0\.0|1\.0\.0/);
    assert.match(maintenanceVersionLine({ ...external, selection: { ...external.selection, version: '4.0.0' } }), /Program 4\.0\.0/);
    assert.match(maintenanceVersionLine(entry(), { unknownCurrent: true }), /version unknown/);
    assert.match(maintenanceVersionLine(entry({ selection: { kind: 'managed', version: null } })), /version unknown/,
        'a failed selected-program probe cannot borrow the managed package version');
    assert.match(maintenanceVersionLine(entry(), { inspectionError: 'offline' }), /Last known/);
    assert.match(harnessMaintenanceMarkup(entry()), /Bundled baseline<\/dt><dd>1\.0\.0/);
});

test('engine facts alone determine buttons for all harnesses and unknown future ones', () => {
    for (const harness of ['codex', 'claude', 'cursor', 'agy', 'future-harness']) {
        const html = harnessMaintenanceMarkup(entry({ harness, canCheckLatest: false, targets: ['latest'] }));
        assert.match(html, /data-maintenance-action="latest"/);
        assert.match(html, />Check version</);
        assert.doesNotMatch(html, />Check latest</);
        assert.doesNotMatch(html, /data-maintenance-action="(previous|version|baseline)"/);
    }
    const unavailable = harnessMaintenanceMarkup(entry({ maintainable: false, remedy: 'Use the selected external installer.' }));
    assert.match(unavailable, /Use the selected external installer/);
    assert.doesNotMatch(unavailable, /data-maintenance-action="(latest|previous|version|baseline)"/);
    const noHistory = harnessMaintenanceMarkup(entry({ previous: null, releaseTested: { version: null } }));
    assert.doesNotMatch(noHistory, /data-maintenance-action="(previous|baseline)"/);
});

test('one maintenance host per family, independent of the account count', () => {
    for (const count of [1, 18]) {
        const group = {
            harness: 'fixture', label: 'Fixture', status: { tone: 'muted', label: 'Ready' },
            rows: Array.from({ length: count }, (_, id) => ({ harness: 'fixture', profile_id: `p${id}`, kind: 'profile' })),
        };
        const html = harnessFamilyMarkup(group, {});
        assert.equal((html.match(/data-family-maintenance=/g) || []).length, 1);
        assert.equal((html.match(/class="harness-account-row/g) || []).length, count);
    }
    assert.doesNotMatch(harnessFamilyMarkup({ harness: 'api-agent', label: 'API agent', rows: [], maintenanceOnly: true }, {}), /data-family-add/);
});

test('version details and failure/progress text are escaped, successful no-change is not called updated', () => {
    const html = harnessMaintenanceMarkup(entry({ selection: { kind: 'path', binary: '<img onerror=bad>' } }), {
        operation: operation({ problem: { message: '<script>bad()</script>' }, progress: ['<b>verifying</b>'] }),
    });
    assert.doesNotMatch(html, /<script>|<img |<b>/);
    assert.match(html, /&lt;script&gt;bad/);
    assert.match(html, /&lt;b&gt;verifying/);
    assert.equal(maintenanceOperationLine(operation({ state: 'succeeded', target: { kind: 'latest', version: null },
        mutation: 'applied', after: { version: '2.0.0', selected: true, proved: true } })),
    'Operation completed · Version 2.0.0');
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

test('native updater inspection never pretends it can check latest', async () => {
    const queries = [];
    const { controller } = setup({ harnessMaintenanceInventory: async (query) => {
        queries.push(query);
        return { harnesses: [entry({ canCheckLatest: false, targets: ['latest'] })] };
    } });
    await controller.refresh(); await controller.act('fixture', 'inspect');
    assert.deepEqual(queries[1], { harness: 'fixture', fresh: true, checkLatest: false });
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
    assert.equal(calls.filter(([kind]) => kind === 'read').length, 1);
    assert.equal(calls.filter(([kind]) => kind === 'start').length, 0);
    assert.equal(controller.view('fixture').operation.phase, 'installing');
});

test('historical unknown effect keeps a newer inspected version and its warning', async () => {
    const terminal = operation({ state: 'failed', phase: 'settled', mutation: 'unknown',
        termination: 'confirmed', finishedAt: '2026-10-09T12:00:00Z' });
    const row = entry({ selection: { kind: 'managed', version: '3.1.0' }, observedAt: '2026-10-09T12:01:00Z',
        operation: { id: terminal.id, state: terminal.state, phase: terminal.phase, finishedAt: terminal.finishedAt } });
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
        const row = entry({ observedAt, selection: { kind: 'managed', version },
            operation: { id: terminal.id, state: terminal.state, phase: terminal.phase, finishedAt } });
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
    const row = entry({ observedAt: '2026-10-09T12:01:00Z', selection: { kind: 'managed', version: '3.1.0' },
        operation: { id: terminal.id, state: terminal.state, phase: terminal.phase, finishedAt: terminal.finishedAt } });
    const { controller } = setup({ harnessMaintenanceInventory: async () => ({ harnesses: [row] }),
        harnessMaintenanceOperation: async () => terminal });
    await controller.refresh(); await flush();
    const view = controller.view('fixture');
    assert.match(maintenanceVersionLine(row, view), /Program 3\.1\.0/);
    assert.equal(maintenanceOperationPending(view.operation), true);
    const html = harnessMaintenanceMarkup(row, view);
    assert.match(html, /Installer may still be running/);
    assert.match(html, /data-maintenance-action="cancel"/);
    assert.doesNotMatch(html, /data-maintenance-action="latest"/);
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
    assert.doesNotMatch(harnessMaintenanceMarkup(entry(), { operation: unconfirmed }), /action="latest"/);
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
        selection: { kind: 'managed', version: '2.5.0' }, installed: { version: '2.5.0', proved: true } })] });
    await flush();
    assert.equal(controller.view('fixture').unknownCurrent, false);
    assert.equal(controller.view('fixture').entry.installed.version, '2.5.0');
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

test('an inspection begun before update cannot restore the old current version', async () => {
    let reads = 0;
    const stale = deferred();
    const { controller } = setup({ harnessMaintenanceInventory: async () => {
        reads++;
        if (reads === 2) return stale.promise;
        return { harnesses: [entry({ installed: { version: reads > 2 ? '3.0.0' : '2.0.0', proved: true } })] };
    } });
    await controller.refresh();
    const read = controller.refresh(); await flush();
    await controller.start('fixture', { kind: 'latest' });
    stale.resolve({ harnesses: [entry()] }); await read; await flush();
    assert.equal(controller.view('fixture').entry.installed.version, '3.0.0');
    assert.equal(controller.view('fixture').unknownCurrent, true, 'active replacement still stays unknown');
});

test('exact-version input uses shared dialog and typed request', async () => {
    const { api, calls } = setup();
    const controller = createHarnessMaintenanceController({ api, intents: new Map(), newKey: () => 'exact-key',
        dialog: async (options) => { assert.equal(options.input, true); return { confirmed: true, value: ' 1.7.0 ' }; } });
    await controller.refresh(); await controller.act('fixture', 'version');
    assert.deepEqual(calls.find(([kind]) => kind === 'start')[1], { harness: 'fixture', target: { kind: 'version', version: '1.7.0' } });
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
