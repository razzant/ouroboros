import assert from 'node:assert/strict';
import test from 'node:test';
import {
    byHandHtml, createHarnessMaintenanceController, currentMissing, harnessMaintenanceMarkup, maintenanceLatestLine,
    maintenanceOperationLine, maintenanceOperationPending, maintenanceVersionLine, missingProgram,
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
    assert.match(harnessMaintenanceMarkup(missing), /action="install"[^>]*>Install</);
    assert.equal(maintenanceOperationLine({state: 'failed'}, {missing: true}), 'Install failed');
    // Beside "Not installed" only a newest release that was read is said; "Latest unknown" says nothing there.
    assert.equal(maintenanceLatestLine(missing), '');
    assert.equal(maintenanceLatestLine({...missing, newest: {status: 'available', version: '1.11.3'}}), '1.11.3 available');
});

test('an absent program is installed where the engine can, and read again after an install by hand', () => {
    const absent = (install = allowed, operation = null) => entry({selection: {path: null, version: null}, operation,
        actions: {install, update: {available: false, reason: 'No program is selected; use Install.', commands: []}, exactVersion: allowed}});
    const refused = {available: false, reason: 'npm is not available on this machine; install it before running this vendor command.', commands: []};
    assert.equal(missingProgram(entry()), null, 'an installed program is not missing');
    assert.equal(missingProgram(null), null, 'an unread one is not either');
    assert.deepEqual(missingProgram(absent()), {installable: true, failed: false, byHand: false});
    assert.deepEqual(missingProgram(absent(refused)), {installable: false, failed: false, byHand: true});
    assert.deepEqual(missingProgram(absent(allowed, operation({state: 'failed', termination: 'confirmed'}))), {installable: true, failed: true, byHand: true});
    assert.equal(missingProgram(absent(allowed, operation({state: 'cancelled', termination: 'confirmed'}))).byHand, false, 'a cancel is the owner\'s own choice');
    assert.equal(missingProgram(absent(allowed, operation({state: 'failed', termination: 'unconfirmed'}))).failed, false, 'an installer that may still run has not ended');

    const plain = harnessMaintenanceMarkup(absent());
    assert.match(plain, /action="install"[^>]*>Install</);
    assert.doesNotMatch(plain, /action="latest"|>Check again<|Install it yourself/, 'an Install is never the Update\'s press');
    assert.doesNotMatch(harnessMaintenanceMarkup(entry()), /action="install"/, 'an installed program offers no Install');
    assert.match(harnessMaintenanceMarkup(absent(), {operation: operation({state: 'running'})}), /action="install" disabled>Install</, 'under way: the same press, disabled');
    assert.match(plain, /action="inspect"[^>]*>Check latest</, 'an ordinary absent program keeps Check latest in Details');
    // The engine cannot install it here: its own reason, no Install, and the read again beside it instead of in Details.
    const html = harnessMaintenanceMarkup(absent(refused));
    assert.match(html, /npm is not available on this machine/);
    assert.doesNotMatch(html, /action="(latest|install)"|Install it yourself|Check latest/);
    assert.match(html, /<div class="harness-maintenance-actions"><button[^>]*action="inspect"[^>]*>Check again</);
    // An install that ended without the program: Install again and Check again. A release-notes page is no install guide
    // (OpenCode's leads to an incompatible line) and what the engine ran, its environment included, never runs again.
    const lost = operation({state: 'failed', termination: 'confirmed', commands: [{binary: 'bash', args: [], env: {HOME: '/tmp/x'},
        script: {url: 'https://vendor.invalid/install.sh', size: 9, sha256: 'x'}}]});
    const failed = harnessMaintenanceMarkup({...absent(allowed, lost), releaseNotesUrl: 'https://vendor.invalid/releases'});
    assert.match(failed, /Install failed/);
    assert.match(failed, /action="install"[^>]*>Install<\/button><button[^>]*action="inspect"[^>]*>Check again</);
    assert.equal(byHandHtml(absent(allowed, lost)), '', 'no verified step: nothing composed, Install and Check again stand');
    assert.doesNotMatch(failed.split('harness-maintenance-details')[0], /install\.sh|HOME=|vendor\.invalid\/releases/);
    // The one verified step, for the line the engine runs: OpenCode 1.x from npm, on any system (claudexor #668).
    const opencode = (install, op) => ({...absent(install, op), harness: 'opencode'});
    const step = 'To install it by hand where Claudexor runs, run <code>npm install -g opencode-ai@1</code> (the 1.x line Claudexor runs), then press Check again.';
    assert.equal(byHandHtml(opencode(allowed, lost)), step);
    const windows = {available: false, reason: 'OpenCode 1.x has no supported unattended Windows installer (#668).', commands: []};
    assert.equal(byHandHtml(opencode(windows)), step, 'refused here for want of an installer, it still runs here when installed by hand');
    assert.match(harnessMaintenanceMarkup(opencode(windows)), /no supported unattended Windows installer[\s\S]*npm install -g opencode-ai@1/);
    // Any other refusal (Cursor on Windows: this engine cannot run it, #667): its reason and the way out, never a call to
    // install what may not run here.
    assert.equal(byHandHtml(absent(refused)), 'If this system cannot run it, choose another program or run Ouroboros on a system that supports it.');
    assert.match(harnessMaintenanceMarkup(absent(refused)), /npm is not available[\s\S]*If this system cannot run it/);
    // An Update that failed before the program went is no failed install: nothing by hand is said of it.
    const update = operation({state: 'failed', termination: 'confirmed', before: {path: '/vendor/program', version: '1.0.0'}});
    assert.deepEqual(missingProgram(absent(allowed, update)), {installable: true, failed: false, byHand: false});
    assert.equal(byHandHtml({...absent(allowed, update), harness: 'opencode'}), '');
});

// A read that failed leaves the row without a current fact: the manual way out is advice that fact can no
// longer support (the rule model_program keeps for the same program), while the way back stays.
test('a failed read withholds the manual way out but keeps the way back', () => {
    const refused = {available: false, reason: 'OpenCode 1.x has no supported unattended Windows installer (#668).', commands: []};
    const opencode = entry({harness: 'opencode', selection: {path: null, version: null},
        actions: {install: refused, update: refused, exactVersion: refused}});
    const step = /npm install -g opencode-ai@1/;
    assert.match(harnessMaintenanceMarkup(opencode), step, 'without a failed read the step stands');
    const failed = harnessMaintenanceMarkup(opencode,
        {inspectionError: 'Program inspection unavailable: synthetic inspection refused'});
    assert.doesNotMatch(failed, step, 'a failed read withholds the advice');
    assert.match(failed, /Program inspection unavailable/, 'the reason stays');
    assert.match(failed, /action="inspect"[^>]*>Check again</, 'the way back stays');
});

// An absent program as the engine reports it (no installed path, `install` offered), and the shape of an Install's calls.
const absentEntry = (extra = {}) => entry({selection: {path: null, version: null},
    actions: {install: allowed, update: {available: false, reason: 'No program is selected; use Install.', commands: []}, exactVersion: allowed}, ...extra});
const shape = (calls) => calls.map(([kind, body]) => (kind === 'start' ? [kind, body] : [kind, body?.fresh]));
const START = ['start', {harness: 'fixture', target: {kind: 'latest'}}];

test('an Install reads the program once more and is sent once, only while that fresh read still finds it absent', async () => {
    const {calls, controller, intents} = setup({ harnessMaintenanceInventory: async (query) => { calls.push(['inventory', query]); return { harnesses: [absentEntry()] }; } });
    await controller.refresh();  // another surface's read: a cached absence proves nothing at send time
    calls.length = 0;
    const first = controller.act('fixture', 'install');
    const second = controller.act('fixture', 'install');  // a double press shares the one read and the one request
    await Promise.all([first, second]);
    await flush();
    assert.deepEqual(shape(calls), [['inventory', true], START], 'one fresh read, one POST');
    assert.equal(calls.find(([kind]) => kind === 'start')[2], 'same-key');
    assert.equal(controller.view('fixture').operation.id, 'operation-1');
    assert.equal(intents.get('fixture').busy, null);
});

test('an Install press never becomes an Update: found installed, by the dispatcher or by the read, it sends nothing', async () => {
    let now = entry();  // installed, with an update offered: `latest` would update it
    const renders = [];
    const {api, calls, intents} = setup({ harnessMaintenanceInventory: async (query) => { calls.push(['inventory', query]); return { harnesses: [now] }; } });
    const controller = createHarnessMaintenanceController({ api, intents, newKey: () => 'same-key', onChange: () => renders.push(1) });
    await controller.refresh();
    calls.length = 0;
    // The press was drawn from an earlier absent read; the dispatcher already holds the program selected.
    await controller.act('fixture', 'install');
    await flush();
    assert.deepEqual(calls, [], 'no read and no request: there is nothing to install');
    assert.equal(intents.get('fixture').request, null);
    // A press the dispatcher still holds absent reads it again, which finds it installed since: nothing is sent either,
    // and the row is left released for the next press.
    intents.get('fixture').entry = absentEntry();
    renders.length = 0;
    await controller.act('fixture', 'install');
    await flush();
    assert.deepEqual(shape(calls), [['inventory', true]]);
    assert.equal(intents.get('fixture').request, null);
    assert.equal(intents.get('fixture').busy, null, 'the press is released');
    assert.ok(renders.length >= 2, 'and the row is drawn again, no longer busy');
    // The explicit Update is unchanged: an installed program is sent `latest` at once, with no extra read.
    calls.length = 0;
    await controller.act('fixture', 'latest');
    assert.deepEqual(shape(calls), [START]);
});

test('a failed, empty or foreign fresh answer authorizes no Install, whatever the cache says', async () => {
    const answers = {
        'a refused read': () => { throw new Error('The engine is not running.'); },
        'an empty answer': () => ({ harnesses: [] }),
        'an answer that leaves the program out': () => ({ harnesses: [absentEntry({harness: 'other'})] }),
        'an answer that finds it installed': () => ({ harnesses: [entry()] }),
        'an answer that finds it absent where the engine cannot install it': () => ({ harnesses: [absentEntry({
            actions: {install: {available: false, reason: 'No installer here.', commands: []}, update: allowed, exactVersion: allowed}})] }),
    };
    for (const [why, answer] of Object.entries(answers)) {
        const {calls, controller, intents} = setup({ harnessMaintenanceInventory: async (query) => {
            calls.push(['inventory', query]);
            return query.fresh ? answer() : { harnesses: [absentEntry()] };  // the cache: absent and installable
        } });
        await controller.refresh();
        assert.ok(currentMissing(controller.view('fixture'))?.installable, `${why}: the cached read offered it`);
        calls.length = 0;
        await controller.act('fixture', 'install');
        await flush();
        assert.deepEqual(calls.map(([kind]) => kind), ['inventory'], `${why}: read once, started nothing`);
        assert.equal(intents.get('fixture').request, null, why);
        assert.equal(intents.get('fixture').busy, null, `${why}: the press is released`);
        assert.ok(!currentMissing(controller.view('fixture'))?.installable, `${why}: the cached absence no longer stands as a fact to install by`);
    }
});

test('an Install rejoins an unresolved press under its own key and an accepted operation; it never starts a second', async () => {
    const calls = [], intents = new Map();
    let keys = 0, lose = true;
    const api = {
        harnessMaintenanceInventory: async (query) => { calls.push(['inventory', query]); return { harnesses: [absentEntry()] }; },
        startHarnessMaintenance: async (body, key) => { calls.push(['start', body, key]); if (lose) throw new TypeError('Failed to fetch'); return operation(); },
        harnessMaintenanceOperation: async (id) => { calls.push(['read', id]); return operation(); },
    };
    const controller = createHarnessMaintenanceController({ api, intents, newKey: () => `key-${++keys}` });
    await controller.refresh();
    calls.length = 0;
    await controller.act('fixture', 'install');
    assert.deepEqual(calls.map(([kind, , key]) => [kind, key]), [['inventory', undefined], ['start', 'key-1']]);
    assert.deepEqual(intents.get('fixture').request.body, {harness: 'fixture', target: {kind: 'latest'}}, 'the lost reply keeps its request');
    calls.length = 0;
    lose = false;
    await controller.act('fixture', 'install');  // pressed again, with the program still read absent
    assert.deepEqual(calls.map(([kind, , key]) => [kind, key]), [['inventory', undefined], ['start', 'key-1']],
        'its own fresh read first, then the same key and body');
    assert.equal(keys, 1, 'no second key was made');
    calls.length = 0;
    await controller.act('fixture', 'install');  // the engine accepted it: its operation is read, not started again
    assert.deepEqual(calls.map(([kind]) => kind), ['read']);
});

test('a fresh answer this view did not apply authorizes no Install: the cached absence beside it is not a fact', async () => {
    const gate = deferred();
    let fresh = 0;
    const {calls, controller, intents} = setup({ harnessMaintenanceInventory: async (query) => {
        calls.push(['inventory', query]);
        if (query.fresh && ++fresh === 1) await gate.promise;  // the press's own read stays out until released
        return { harnesses: [absentEntry()] };
    } });
    await controller.refresh();  // cached: absent and installable
    calls.length = 0;
    const press = controller.act('fixture', 'install');
    await flush();
    intents.get('fixture').revision += 1;  // another surface's operation was accepted while that read was out: its answer is superseded
    gate.resolve();
    await press;
    await flush();
    assert.deepEqual(calls.map(([kind]) => kind), ['inventory', 'inventory'], 'the superseded answer is read again by the view itself');
    assert.equal(intents.get('fixture').request, null, 'and no Install was sent on it');
});

test('two views of one program share one Install press', async () => {
    const intents = new Map(), calls = [];
    const api = {
        harnessMaintenanceInventory: async (query) => { calls.push(['inventory', query]); return { harnesses: [absentEntry()] }; },
        startHarnessMaintenance: async (body, key) => { calls.push(['start', body, key]); return operation(); },
        harnessMaintenanceOperation: async (id) => operation(),
    };
    const views = [1, 2].map(() => createHarnessMaintenanceController({ api, intents, newKey: () => 'same-key' }));
    await views[0].refresh();
    calls.length = 0;
    await Promise.all(views.map((view) => view.act('fixture', 'install')));
    await flush();
    assert.deepEqual(shape(calls), [['inventory', true], START]);
});

test('the Install a row shows carries its intent to the dispatcher, whatever the dispatcher reads by then', async () => {
    const calls = [], intents = new Map();
    const api = {
        harnessMaintenanceInventory: async (query) => { calls.push(['inventory', query]); return { harnesses: [absentEntry()] }; },
        startHarnessMaintenance: async (body, key) => { calls.push(['start', body, key]); return operation(); },
        harnessMaintenanceOperation: async () => operation(),
    };
    // The row's host as the markup is drawn into it: each control it lists is pressed by its own name.
    const element = { dataset: {familyMaintenance: 'fixture'}, ownerDocument: {}, contains: () => false, querySelector: () => null, controls: [],
        set innerHTML(html) {
            this.html = html;
            this.controls = [...html.matchAll(/<button[^>]*data-maintenance-action="([^"]+)"[^>]*>([^<]*)</g)].map(([, action, label]) => ({
                dataset: {maintenanceAction: action}, label, listeners: [], addEventListener(_type, listener) { this.listeners.push(listener); } }));
        },
        querySelectorAll(selector) { return selector === '[data-maintenance-action]' ? this.controls : []; } };
    const controller = createHarnessMaintenanceController({ api, intents, newKey: () => 'same-key', host: () => ({ querySelectorAll: () => [element] }) });
    await controller.refresh();
    const press = (label) => element.controls.find((control) => control.label === label).listeners.forEach((listener) => listener());
    assert.deepEqual(element.controls.map((control) => [control.dataset.maintenanceAction, control.label]).filter(([, label]) => label === 'Install'),
        [['install', 'Install']], 'the visible Install says install, not the Update\'s name');
    // The dispatcher now holds the program installed with an update offered: the same Install press sends nothing.
    intents.get('fixture').entry = entry();
    calls.length = 0;
    press('Install');
    await flush();
    assert.deepEqual(calls, []);
    assert.equal(intents.get('fixture').request, null);
    // Absent as drawn: that press reads once and sends one Install.
    intents.get('fixture').entry = absentEntry();
    press('Install');
    await flush();
    assert.deepEqual(shape(calls), [['inventory', true], START]);
});

test('a newer answered list that leaves a program out takes its absence away; its request and operation stay', async () => {
    const absent = entry({harness: 'gone', selection: {path: null, version: null}});
    let rows = [entry(), absent];
    const {controller, intents} = setup({ harnessMaintenanceInventory: async () => ({ harnesses: rows }) });
    await controller.refresh();
    assert.deepEqual(currentMissing(controller.view('gone')), {installable: true, failed: false, byHand: false});
    intents.get('gone').operation = operation({harness: 'gone', state: 'failed', termination: 'unconfirmed'});
    rows = [entry()];
    await controller.refresh({fresh: true});
    assert.equal(currentMissing(controller.view('gone')), null, 'a row the newest list left out is no fact to install by');
    assert.equal(controller.view('gone').operation.id, 'operation-1', 'the owner\'s operation is kept to check');
    rows = [entry(), absent];
    await controller.refresh({fresh: true});
    assert.ok(currentMissing(controller.view('gone')), 'listed again, it is current again');
});

test('a row the newest list left out offers no stale Install, Update or by-hand advice; receipt and Check again stay', () => {
    const refused = {available: false, reason: 'No installer here.', commands: []};
    const stale = (view) => harnessMaintenanceMarkup(absentEntry(), view);
    // Installable absence: the fresh Check is the row's own press; no stale Install stands on the retained row.
    assert.doesNotMatch(stale({listed: false}), /action="(install|latest)"/);
    assert.match(stale({listed: false}), /<div class="harness-maintenance-actions"><button[^>]*action="inspect"[^>]*>Check again</);
    // Un-installable absence: the reason that was read stays; no stale advice to install by hand, no Check latest.
    const unable = absentEntry({ actions: { install: refused, update: allowed, exactVersion: allowed } });
    const byHand = harnessMaintenanceMarkup(unable, { listed: false });
    assert.match(byHand, /No installer here\./);
    assert.doesNotMatch(byHand, /If this system cannot run it|npm install -g|>Check latest</);
    assert.match(byHand, /<div class="harness-maintenance-actions"><button[^>]*action="inspect"[^>]*>Check again</);
    // The owner's unresolved press keeps its receipt: rejoin stands in the row, Check keeps its Details place.
    const pending = harnessMaintenanceMarkup(absentEntry(), { listed: false,
        request: { key: 'k', body: { harness: 'fixture', target: { kind: 'latest' } } } });
    assert.match(pending, /action="rejoin"[^>]*>Retry same request/);
    assert.doesNotMatch(pending.split('harness-maintenance-details')[0], /action="inspect"/);
    // A listed row keeps its Install: the withhold is the staleness, not the absence.
    assert.match(harnessMaintenanceMarkup(absentEntry(), { listed: true }), /action="install"[^>]*>Install</);
});

test('a family with a known-absent program the engine cannot install offers no header button, whatever the roster read', () => {
    const group = (rows = []) => ({ harness: 'fixture', label: 'Fixture', rows,
        status: { tone: 'muted', label: 'Ready' } });
    const ready = { daemon: { runtime: { state: 'ready' } } };
    const unInstallable = { installable: false, failed: false, byHand: true };
    for (const accountsRead of ['ok', 'failed']) {
        // An empty family gets no Connect; a family with accounts keeps its rows and loses only the header press.
        assert.doesNotMatch(harnessFamilyMarkup(group([]), ready, { accountsRead, missing: unInstallable }), /data-family-add/);
        const withRows = harnessFamilyMarkup(group([{ harness: 'fixture', profile_id: 'work', kind: 'profile' }]),
            ready, { accountsRead, missing: unInstallable });
        assert.match(withRows, /class="ui-row harness-account-row/);
        assert.doesNotMatch(withRows, /data-family-add/, 'the program row says why and offers Check again');
    }
    // The Install & connect naming still claims a roster read empty; unread keeps the runtime's own words.
    const installable = { installable: true, failed: false, byHand: false };
    assert.match(harnessFamilyMarkup(group([]), ready, { accountsRead: 'ok', missing: installable }),
        /data-family-add>[^<]*Install &amp; connect</);
    assert.match(harnessFamilyMarkup(group([]), ready, { accountsRead: 'failed', missing: installable }),
        /data-family-add>[^<]*Connect</);
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

test('a status tick polls pending operations only; inspecting programs is an owner act', async t => {
    // Owner choice 2026-11-14 (B) and the base contract: a provider's page, an opened chooser,
    // Check latest, Refresh and an operation's settlement inspect; ticks never do.
    let clock = 1000000;
    t.mock.method(Date, 'now', () => clock);
    const calls = [], answers = [], intents = new Map();
    const api = {
        harnessMaintenanceInventory: query => {
            calls.push(['inventory', query]); answers.push(deferred()); return answers.at(-1).promise;
        },
        harnessMaintenanceOperation: async id => { calls.push(['read', id]); return operation({ state: 'running' }); },
    };
    const first = createHarnessMaintenanceController({ intents, api });
    for (let n = 0; n < 30; n++) { clock += 5000; first.poll(); await flush(); }
    assert.deepEqual(calls, [], 'an idle tick inspected the programs');
    first.refresh({ fresh: true, checkLatest: true }); await flush(); // the owner showed the page
    assert.deepEqual(calls, [['inventory', { harness: '', fresh: true, checkLatest: true }]]);
    first.dispose(); // Settings navigation leaves while that inspection runs
    const second = createHarnessMaintenanceController({ intents, api });
    second.refresh({ fresh: true, checkLatest: true }); await flush(); // the owner shows it again
    assert.equal(calls.length, 2, 'the reopened view relied on an answer its predecessor discards');
    answers[0].resolve({ harnesses: [entry({ selection: { ...entry().selection, version: '1.0.0' } })] }); await flush();
    assert.equal(second.view('fixture').entry, undefined, 'a released view applied its late answer');
    answers[1].resolve({ harnesses: [entry({ operation: operation() })] }); await flush();
    assert.equal(second.view('fixture').entry.selection.version, '2.0.0');
    calls.length = 0;
    clock += 1800000; second.poll(); await flush();
    assert.deepEqual(calls, [['read', 'operation-1']], 'a tick reads a pending operation and nothing else');
    second.dispose();
});

// ---- W2 (rl-20261126T182402): the read clock is shared through `intents`, so a later-asked read stands
// across two controllers and in the same view; the install gate no longer keys on object identity; and a
// pending request wins over a historical receipt in the row's actions. ----
const probeApi = (pending, sent) => ({
    harnessMaintenanceInventory: () => new Promise((resolve) => pending.push({ resolve })),
    startHarnessMaintenance: async (body, key) => { sent.push({ body, key }); return operation({ harness: 'opencode' }); },
    harnessMaintenanceOperation: async () => operation({ harness: 'opencode' }),
});

test('an older install-check read never overwrites a newer installed reading, across two controllers', async () => {
    const pending = [], sent = [], intents = new Map();
    const a = createHarnessMaintenanceController({ api: probeApi(pending, sent), intents, newKey: () => 'press-key' });
    const b = createHarnessMaintenanceController({ api: probeApi(pending, sent), intents });
    const absentRow = absentEntry({ harness: 'opencode' });
    const installedRow = entry({ harness: 'opencode', selection: { path: '/x/opencode', version: '1.4.0' } });
    const seed = a.refresh({ fresh: true });
    await Promise.resolve();
    pending.shift().resolve({ harnesses: [absentRow] });
    await seed;
    const press = a.act('opencode', 'install');
    await Promise.resolve();
    const latest = b.refresh({ checkLatest: true });
    await Promise.resolve();
    assert.equal(pending.length, 2);
    pending[1].resolve({ harnesses: [installedRow] });  // the newer read answers first
    await latest;
    assert.equal(a.view('opencode').entry.selection.version, '1.4.0');
    pending[0].resolve({ harnesses: [absentRow] });     // the older read answers last
    await press; await flush();
    assert.equal(sent.length, 0, 'Install never turns into Update');
    assert.equal(a.view('opencode').entry.selection.version, '1.4.0');
    a.dispose(); b.dispose();
});

test('two different absent answers still let one Install go out under the press key', async () => {
    const pending = [], sent = [], intents = new Map();
    const a = createHarnessMaintenanceController({ api: probeApi(pending, sent), intents, newKey: () => 'press-key' });
    const b = createHarnessMaintenanceController({ api: probeApi(pending, sent), intents });
    const seed = a.refresh({ fresh: true });
    await Promise.resolve();
    pending.shift().resolve({ harnesses: [absentEntry({ harness: 'opencode' })] });
    await seed;
    const press = a.act('opencode', 'install');
    await Promise.resolve();
    const latest = b.refresh({ checkLatest: true });
    await Promise.resolve();
    assert.equal(pending.length, 2);
    pending[0].resolve({ harnesses: [absentEntry({ harness: 'opencode' })] });  // same evidence, new object
    pending[1].resolve({ harnesses: [absentEntry({ harness: 'opencode' })] });
    await Promise.all([press, latest]); await flush();
    assert.equal(sent.length, 1);
    assert.equal(sent[0].key, 'press-key');
    assert.equal(sent[0].body.target.kind, 'latest');
    a.dispose(); b.dispose();
});

test('a pending request wins over a historical receipt: Retry same request, never Check operation', () => {
    const receipt = operation({ id: 'old-receipt', state: 'failed', mutation: 'none', termination: 'confirmed', before: {} });
    const html = harnessMaintenanceMarkup(absentEntry({ operation: receipt }), {
        operation: receipt, request: { key: 'new-press', body: { harness: 'fixture', target: { kind: 'latest' } } },
        operationError: 'Update acceptance unconfirmed: lost',
    });
    assert.match(html, /action="rejoin"[^>]*>Retry same request</);
    assert.doesNotMatch(html, /Check operation/);
});

// W2 residual: the maintenance controller's completion boundary when its view is released mid-read.
test('a view released while its Install read is in flight leaves the shared request alone', async () => {
    const {createHarnessMaintenanceController} = await import('../modules/harness_maintenance.js');
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const harness = 'released-install';
    const intents = new Map();
    const absent = {harness, selection: {path: null, version: null}, updateAvailable: 'cannot_tell',
        actions: {install: {available: true}, update: {available: false, reason: 'no update'}}};
    const posts = [];
    let call = 0, release;
    const api = {
        async harnessMaintenanceInventory() {
            call += 1;
            if (call === 1) return {harnesses: [absent]};
            return new Promise((resolve) => { release = () => resolve({harnesses: [absent]}); });
        },
        async startHarnessMaintenance(body, key) {
            posts.push([body, key]);
            return {id: 'op', harness: body.harness, state: 'running'};
        },
    };
    let controller;
    let released = false;
    controller = createHarnessMaintenanceController({api, intents, onChange: () => {
        if (release && !released) { released = true; controller.dispose(); }
    }});
    await controller.refresh({harness, fresh: true});
    const press = controller.act(harness, 'install');
    await flush();
    assert.ok(release, 'the Install reads the program once more before it sends');
    release();
    await press; await flush();
    assert.deepEqual(posts, [], 'a released view sends nothing');
    assert.equal(intents.get(harness).request, null, 'and it leaves the shared request as it was');
    assert.equal(intents.get(harness).operation, null, 'no receipt of its own either');
});

// ---- W3 (rl-20261126T203549-c634366eecb0): one order for applied facts, a separate freshness clock for
// failures, and the whole-inventory refusal scoped per program. The first two reproduce the four negative
// cases of qa-install/w4/probe-program-order.mjs (confirmed on candidate 063e2c12): they fail on the
// pre-W3 code. ----

test('a newer inventory (passive installed, empty current, omitting current) stands over a late older install check', async () => {
    for (const [fresh, rows, verify] of [
        [false, [entry({ harness: 'opencode', selection: { path: '/x/opencode', version: '1.4.0' } })],
            (view) => assert.equal(view.entry?.selection?.version, '1.4.0', 'a passive confirmation of installed')],
        [true, [], (view) => assert.equal(view.listed, false, 'an empty current inventory omits it')],
        [true, [entry({ harness: 'codex' })], (view) => assert.equal(view.listed, false, 'a current inventory that omits it')],
    ]) {
        const reads = [], sent = [], intents = new Map();
        const api = {
            harnessMaintenanceInventory: (opts) => new Promise((resolve) => reads.push({ opts, resolve })),
            startHarnessMaintenance: async (body, key) => { sent.push({ body, key }); return operation({ harness: body.harness }); },
        };
        const a = createHarnessMaintenanceController({ api, intents, newKey: () => 'press-key' });
        const b = createHarnessMaintenanceController({ api, intents });
        const seed = a.refresh({ fresh: true });
        await Promise.resolve();
        reads.shift().resolve({ harnesses: [absentEntry({ harness: 'opencode' }), absentEntry({ harness: 'agy' })] });
        await seed;
        assert.ok(currentMissing(a.view('opencode'))?.installable, 'seed: absent and installable');
        const press = a.act('opencode', 'install');
        await Promise.resolve();
        const targeted = reads.shift();
        assert.equal(targeted.opts.harness, 'opencode');
        assert.equal(targeted.opts.fresh, true, 'the Install reads its program fresh, once');
        const later = b.refresh({ fresh });
        await Promise.resolve();
        const whole = reads.shift();
        assert.equal(whole.opts.harness, '');
        whole.resolve({ harnesses: rows });                        // the newer inventory answers first
        await later;
        verify(a.view('opencode'));
        targeted.resolve({ harnesses: [absentEntry({ harness: 'opencode' })] });  // the older check answers last
        await press; await flush();
        assert.equal(sent.length, 0, 'no operation is made from the superseded absence');
        verify(a.view('opencode'));
        a.dispose(); b.dispose();
    }
});

test('a fresh targeted answer clears the whole-inventory refusal for its own program only', async () => {
    const reads = [], intents = new Map();
    const api = { harnessMaintenanceInventory: (opts) => new Promise((resolve, reject) => reads.push({ opts, resolve, reject })) };
    const a = createHarnessMaintenanceController({ api, intents });
    const b = createHarnessMaintenanceController({ api, intents });
    const seed = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [absentEntry({ harness: 'opencode' }), absentEntry({ harness: 'agy' }), entry({ harness: 'codex' })] });
    await seed;
    const refused = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().reject(new Error('inventory service refused'));
    await refused;
    assert.match(a.view('agy').inventoryError, /inventory service refused/);
    assert.equal(currentMissing(a.view('agy')), null);
    const codex = b.refresh({ harness: 'codex', fresh: true });
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [entry({ harness: 'codex' })] });
    await codex;
    assert.equal(a.view('codex').inventoryError, '', 'its own fresh answer clears it for itself');
    assert.match(a.view('agy').inventoryError, /inventory service refused/, 'the sibling keeps the refusal');
    assert.equal(currentMissing(a.view('agy')), null);
    assert.equal(currentMissing(a.view('opencode')), null);
    a.dispose(); b.dispose();
});

test('an unresolved Install press and its receipt survive a remount on the shared intents', async () => {
    const sent = [], intents = new Map();
    const api = {
        harnessMaintenanceInventory: async () => ({ harnesses: [absentEntry({ harness: 'remount' })] }),
        startHarnessMaintenance: async (body, key) => {
            sent.push({ body, key });
            if (sent.length === 1) throw new Error('Lost POST');
            return operation({ id: 'op-1', harness: 'remount', state: 'running' });
        },
        harnessMaintenanceOperation: async (id) => operation({ id, harness: 'remount', state: 'running' }),
    };
    const a = createHarnessMaintenanceController({ api, intents, newKey: () => 'press-key' });
    await a.refresh({ harness: 'remount', fresh: true }); await flush();
    await a.act('remount', 'install'); await flush();
    assert.ok(a.view('remount').request, 'the lost POST is retained');
    assert.match(a.view('remount').operationError, /Install acceptance unconfirmed: Lost POST/, 'the press names its own gesture');
    assert.doesNotMatch(a.view('remount').operationError, /Retry the same request/,
        'the saved reason names the gesture, not a second instruction the row\'s own control gives');
    assert.match(harnessMaintenanceMarkup(absentEntry({ harness: 'remount' }), a.view('remount')), /Retry same request/);
    a.dispose();
    const b = createHarnessMaintenanceController({ api, intents, newKey: () => { throw new Error('must reuse the retained key'); } });
    assert.ok(b.view('remount').request, 'the unresolved press survives the remount');
    assert.match(harnessMaintenanceMarkup(absentEntry({ harness: 'remount' }), b.view('remount')), /Retry same request/);
    await b.act('remount', 'rejoin'); await flush();
    assert.deepEqual(sent[1], sent[0], 'rejoin sends the same key and body');
    assert.equal(b.view('remount').operation.id, 'op-1', 'the accepted receipt is there');
    await b.readOperation('remount'); await flush();
    assert.equal(b.view('remount').operation.state, 'running', 'the pending receipt is read again after the remount');
    a.dispose(); b.dispose();
});

test('an Install the engine never received says Install, not Update', async () => {
    const { controller } = setup({
        harnessMaintenanceInventory: async () => ({ harnesses: [absentEntry()] }),
        startHarnessMaintenance: async () => { throw acquisitionError(503); },
    });
    await controller.refresh();
    await controller.act('fixture', 'install'); await flush();
    assert.equal(controller.view('fixture').request, null, 'a typed not-submitted refusal has no installation effect');
    assert.match(controller.view('fixture').operationError, /^Install not started: Could not acquire the engine$/);
    controller.dispose();
});

// ---- W5: one evidence order for the Install gate (a newer omission, refusal or installed reading
// contradicts the press; a newer agreeing absence does not lose it), an empty answer never moves the
// failure clock, an explicit checkLatest survives a later passive not_read, and a released issuer's
// terminal transition still reaches the mounted views. ----

const installedEntry = (harness) => entry({ harness, selection: { path: `/x/${harness}`, version: '1.4.0' } });

test('the Install gate reads the press answer and the standing view together', async () => {
    for (const [later, count] of [
        [[entry({ harness: 'codex' })], 0],                                        // the newer whole read omits the pressed program
        [[entry({ harness: 'opencode', selection: { path: '/x/opencode', version: '1.4.0' } })], 0],  // ... or finds it installed
        [[absentEntry({ harness: 'opencode' })], 1],                                // ... or agrees on the absence
    ]) {
        const reads = [], sent = [], intents = new Map();
        const api = {
            harnessMaintenanceInventory: (opts) => new Promise((resolve) => reads.push({ opts, resolve })),
            startHarnessMaintenance: async (body, key) => { sent.push({ body, key }); return operation({ harness: body.harness }); },
        };
        const a = createHarnessMaintenanceController({ api, intents, newKey: () => 'press-key' });
        const b = createHarnessMaintenanceController({ api, intents });
        const seed = a.refresh({ fresh: true });
        await Promise.resolve();
        reads.shift().resolve({ harnesses: [absentEntry({ harness: 'opencode' }), absentEntry({ harness: 'agy' })] });
        await seed;
        assert.ok(currentMissing(a.view('opencode'))?.installable, 'seed: absent and installable');
        const press = a.act('opencode', 'install');
        await Promise.resolve();
        const targeted = reads.shift();
        assert.equal(targeted.opts.harness, 'opencode');
        assert.equal(targeted.opts.fresh, true, 'the Install reads its program fresh, once');
        const passive = b.refresh();
        await Promise.resolve();
        const whole = reads.shift();
        assert.equal(whole.opts.harness, '');
        whole.resolve({ harnesses: later });                          // the newer whole read answers first
        await passive;
        targeted.resolve({ harnesses: [absentEntry({ harness: 'opencode' })] });  // the press's own read answers last
        await press; await flush();
        assert.equal(sent.length, count, JSON.stringify(later));
        if (count) assert.deepEqual(sent[0], { key: 'press-key', body: { harness: 'opencode', target: { kind: 'latest' } } });
        a.dispose(); b.dispose();
    }
});

test('an empty targeted answer neither clears nor re-dates a standing whole refusal', async () => {
    const reads = [], intents = new Map();
    const api = { harnessMaintenanceInventory: (opts) => new Promise((resolve, reject) => reads.push({ opts, resolve, reject })) };
    const a = createHarnessMaintenanceController({ api, intents });
    const b = createHarnessMaintenanceController({ api, intents });
    const seed = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [absentEntry({ harness: 'opencode' }), absentEntry({ harness: 'agy' })] });
    await seed;
    const refused = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().reject(new Error('synthetic inspection refused'));
    await refused;
    const empty = b.refresh({ harness: 'opencode', fresh: true });
    await Promise.resolve();
    reads.shift().resolve({});  // no harness list at all: empty evidence
    await empty; await flush();
    assert.match(b.view('opencode').inventoryError, /synthetic inspection refused/, 'the whole refusal still stands');
    assert.equal(currentMissing(b.view('opencode')), null, 'an empty answer proves no absence either');
    a.dispose(); b.dispose();
});

test('a later empty whole answer never re-dates an older refusal; a program\'s own fresh answer clears only itself', async () => {
    const reads = [], intents = new Map();
    const api = { harnessMaintenanceInventory: (opts) => new Promise((resolve, reject) => reads.push({ opts, resolve, reject })) };
    const a = createHarnessMaintenanceController({ api, intents });
    const b = createHarnessMaintenanceController({ api, intents });
    const seed = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [absentEntry({ harness: 'opencode' }), absentEntry({ harness: 'agy' })] });
    await seed;
    const refused = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().reject(new Error('synthetic inspection refused'));
    await refused;
    const proof = b.refresh({ harness: 'opencode', fresh: true });
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [installedEntry('opencode')] });
    await proof;
    const empty = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().resolve({});  // a later empty whole answer
    await empty; await flush();
    assert.equal(b.view('opencode').inventoryError, '', 'its own fresh success cleared the refusal for itself');
    assert.match(b.view('agy').inventoryError, /synthetic inspection refused/, 'the sibling keeps the original, not re-dated refusal');
    a.dispose(); b.dispose();
});

test('an explicit checkLatest is retained for the same selected program when a later passive read did not check it', async () => {
    const reads = [], intents = new Map();
    const api = { harnessMaintenanceInventory: (opts) => new Promise((resolve) => reads.push({ opts, resolve })) };
    const a = createHarnessMaintenanceController({ api, intents });
    const b = createHarnessMaintenanceController({ api, intents });
    const seed = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [installedEntry('opencode')] });
    await seed;
    const checked = a.refresh({ harness: 'opencode', fresh: true, checkLatest: true });
    await Promise.resolve();
    const targeted = reads.shift();
    assert.equal(targeted.opts.checkLatest, true);
    const passive = b.refresh();
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [installedEntry('opencode')] });  // passive: newest not_read
    await passive;
    targeted.resolve({ harnesses: [{ ...installedEntry('opencode'), newest: { status: 'available', version: '1.5.0' }, updateAvailable: 'yes' }] });
    await checked; await flush();
    assert.equal(b.view('opencode').entry.newest.version, '1.5.0', 'the explicit release check holds');
    assert.equal(b.view('opencode').entry.updateAvailable, 'yes');
    a.dispose(); b.dispose();
});

test('a released issuer\'s lost POST still notifies the mounted views under the same key', async () => {
    let rejectPost, beginPost;
    const began = new Promise((resolve) => { beginPost = resolve; });
    const answer = new Promise((resolve, reject) => { rejectPost = reject; });
    const sent = [], observations = [], intents = new Map();
    const api = {
        harnessMaintenanceInventory: async () => ({ harnesses: [absentEntry({ harness: 'opencode' })] }),
        startHarnessMaintenance: async (body, key) => { sent.push({ body, key }); beginPost(); return answer; },
    };
    const a = createHarnessMaintenanceController({ api, intents, newKey: () => 'press-key' });
    const b = createHarnessMaintenanceController({ api, intents, onChange: () => observations.push(b.view('opencode')) });
    await a.refresh({ fresh: true }); await flush();
    const press = a.act('opencode', 'install'); await flush();
    assert.equal(sent.length, 1);
    assert.ok(b.view('opencode').busy, 'the mounted view sees the shared press under way');
    a.dispose();
    const before = observations.length;
    rejectPost(new TypeError('synthetic network gap'));
    await press; await flush();
    assert.ok(observations.length > before, 'the terminal shared transition reached the mounted view');
    assert.ok(!observations.at(-1).busy, 'the press is released');
    assert.equal(observations.at(-1).request?.key, 'press-key', 'the retained request keeps its key');
    assert.equal(sent.length, 1, 'no second request was made');
    b.dispose();
});

// ---- W5 (final): a whole read's own refusal must not outlive a newer whole answer, and a released
// issuer's terminal transition must reach the mounted views exactly once. ----
test('two whole reads in different views: the newer answer stands and the older refusal is not shown', async () => {
    const intents = new Map(), reads = [];
    const api = { harnessMaintenanceInventory: (query) => new Promise((resolve, reject) => reads.push({ query, resolve, reject })) };
    const a = createHarnessMaintenanceController({ api, intents });
    const b = createHarnessMaintenanceController({ api, intents });
    const older = a.refresh();               // a whole, cached read
    await flush();
    const newer = b.refresh();               // another whole, cached read: the later-asked one
    await flush();
    assert.equal(reads.length, 2, 'both views asked their own read');
    reads[1].resolve({ harnesses: [entry()] });                 // the newer read answers first
    await newer; await flush();
    reads[0].reject(new Error('synthetic inspection refused')); // the older read refuses last
    await older; await flush();
    assert.equal(b.inventoryError(), '', 'the newer whole answer stands: no refusal is left to show');
    assert.equal(b.view('fixture').inventoryError, '', 'and the program row shows none either');
    assert.equal(b.view('fixture').inspectionError, '', 'nor does the row carry the refusal of its own');
    a.dispose(); b.dispose();
});

test('a released issuer\'s terminal Install is settled once by the next live read', async () => {
    const intents = new Map(), reads = [], sent = [];
    let settles = 0, release;
    const post = new Promise((resolve) => { release = resolve; });
    const settledOp = operation({ harness: 'opencode', state: 'succeeded', mutation: 'applied' });
    let accepted = null;
    const api = {
        harnessMaintenanceInventory: async (query) => { reads.push(query); return { harnesses: [absentEntry({ harness: 'opencode', operation: accepted })] }; },
        startHarnessMaintenance: async (body, key) => { sent.push({ body, key }); return post; },
    };
    const issuer = createHarnessMaintenanceController({ api, intents, newKey: () => 'press-key', onSettled: () => { settles++; } });
    await issuer.refresh();                  // the cached absence the press needs
    const press = issuer.act('opencode', 'install');
    await flush();                           // the press's own fresh read answered; the POST is out
    assert.equal(sent.length, 1, 'the Install is sent once');
    assert.equal(sent[0].key, 'press-key', 'under the press\'s own key');
    const readsBefore = reads.length;
    issuer.dispose();                        // the issuer is released while the POST is in flight
    accepted = settledOp;
    release(settledOp);
    await press; await flush();
    assert.equal(settles, 0, 'a released view delivers no settlement of its own');
    assert.equal(intents.get('opencode').operation.state, 'succeeded', 'the receipt is kept');
    assert.equal(reads.length, readsBefore, 'and the released view read nothing');
    const live = createHarnessMaintenanceController({ api, intents, onSettled: () => { settles++; } });
    await live.refresh(); await flush();
    assert.equal(settles, 1, 'the live view that reads the operation delivers the settlement once');
    await live.refresh(); await flush();
    assert.equal(settles, 1, 'and not again');
    assert.equal(sent.length, 1, 'no second request was made');
    issuer.dispose(); live.dispose();
});

// ---- W5 (rl-20261127T050408-87b92a606451), area 1: the newest observation of a program orders
// its facts, a lost Install is not sent again once the program has moved on, and the engine's
// receipt under this very key is the answer to that press.

test('a newer installed observation stands over a later-asked Install read of an older absence', async () => {
    const reads = [], sent = [], intents = new Map();
    const api = {
        harnessMaintenanceInventory: (opts) => new Promise((resolve) => reads.push({ opts, resolve })),
        startHarnessMaintenance: async (body, key) => { sent.push({ body, key }); return operation({ harness: body.harness }); },
    };
    const a = createHarnessMaintenanceController({ api, intents, newKey: () => 'press-key' });
    const b = createHarnessMaintenanceController({ api, intents });
    const seed = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [absentEntry({ harness: 'opencode', observedAt: '2026-11-27T12:00:01Z' })] });
    await seed;
    a.refresh({ fresh: true });                                    // asked before the press's own read
    await Promise.resolve();
    const newerRead = reads.shift();
    const press = b.act('opencode', 'install');
    await Promise.resolve();
    newerRead.resolve({ harnesses: [entry({ harness: 'opencode', selection: { path: '/x/opencode', version: '1.4.0' },
        observedAt: '2026-11-27T12:00:03Z' })] });                 // the newer observation lands first
    reads.shift().resolve({ harnesses: [absentEntry({ harness: 'opencode', observedAt: '2026-11-27T12:00:02Z' })] });
    await press; await flush();
    assert.equal(sent.length, 0, 'the newest observation says the program is installed');
    assert.equal(b.view('opencode').entry.selection.version, '1.4.0', 'the older absence does not replace it');
    a.dispose(); b.dispose();
});

test('a lost Install is not sent again once the program reads installed, and its key is kept', async () => {
    const sent = [];
    let installed = false;
    const { controller } = setup({
        harnessMaintenanceInventory: async () => ({ harnesses: [installed
            ? entry({ harness: 'fixture', selection: { path: '/vendor/program', version: '2.0.0' } })
            : absentEntry({ harness: 'fixture' })] }),
        startHarnessMaintenance: async (body, key) => { sent.push({ body, key }); throw new Error('Connection lost'); },
    });
    await controller.refresh(); await flush();
    await controller.act('fixture', 'install'); await flush();
    assert.equal(sent.length, 1, 'the press was sent once');
    assert.ok(controller.view('fixture').request, 'its unknown acceptance keeps the request');
    installed = true;
    await controller.refresh(); await flush();
    const view = controller.view('fixture');
    assert.match(view.retryBlocked, /installed now/, 'the row says why the press is not sent again');
    await controller.act('fixture', 'rejoin'); await flush();
    assert.equal(sent.length, 1, 'the same request is not sent again as an Update');
    assert.ok(controller.view('fixture').request, 'the original body and key stay');
    assert.deepEqual(controller.view('fixture').request.key, 'same-key');
    const html = harnessMaintenanceMarkup(controller.view('fixture').entry, controller.view('fixture'));
    assert.match(html, /installed now/, 'the row says it');
    assert.doesNotMatch(html, /action="rejoin"/, 'no enabled Retry stands in its place');
    assert.match(html, /action="inspect"[^>]*>Check again</, 'the read-only check stands instead');
});

test('a lost Install still retries under its own key while the program reads absent', async () => {
    const sent = [];
    const { controller } = setup({
        harnessMaintenanceInventory: async () => ({ harnesses: [absentEntry({ harness: 'fixture' })] }),
        startHarnessMaintenance: async (body, key) => { sent.push({ body, key }); throw new Error('Connection lost'); },
    });
    await controller.refresh(); await flush();
    await controller.act('fixture', 'install'); await flush();
    assert.equal(controller.view('fixture').retryBlocked, '', 'an absent program keeps the retry');
    await controller.act('fixture', 'rejoin'); await flush();
    assert.equal(sent.length, 2);
    assert.deepEqual(sent[1], sent[0], 'the same key and body');
});

test('the engine\'s receipt under this very key recovers a lost Install; another press\'s receipt does not', async () => {
    const sent = [];
    let receipt = null;
    const { controller } = setup({
        harnessMaintenanceInventory: async () => ({ harnesses: [absentEntry({ harness: 'fixture', operation: receipt })] }),
        startHarnessMaintenance: async (body, key) => { sent.push({ body, key }); throw new Error('Connection lost'); },
    });
    await controller.refresh(); await flush();
    await controller.act('fixture', 'install'); await flush();
    assert.ok(controller.view('fixture').request, 'the press is unresolved');
    const base = { id: 'op-9', harness: 'fixture', state: 'succeeded', phase: 'settled',
        target: { kind: 'latest', version: null }, mutation: 'applied', outcome: 'changed', termination: 'confirmed',
        commands: [], limitations: [], progress: [] };
    receipt = { ...base, requester: { source: 'owner', requestId: 'another-key' } };
    await controller.refresh(); await flush();
    assert.ok(controller.view('fixture').request, 'another press\'s receipt does not clear ours');
    receipt = { ...base, requester: { source: 'owner', requestId: 'same-key' } };
    await controller.refresh(); await flush();
    const view = controller.view('fixture');
    assert.equal(view.operation?.id, 'op-9', 'the engine\'s own receipt is adopted');
    assert.equal(view.request, null, 'and the press is resolved by it');
});

test('a newer-asked observation of an older date never replaces the known selection', async () => {
    const reads = [], intents = new Map();
    const api = { harnessMaintenanceInventory: (opts) => new Promise((resolve) => reads.push({ opts, resolve })) };
    const controller = createHarnessMaintenanceController({ api, intents });
    const seed = controller.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [entry({ harness: 'opencode', selection: { path: '/x/opencode', version: '1.4.0' },
        observedAt: '2026-11-27T12:00:03Z' })] });
    await seed;
    const later = controller.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift().resolve({ harnesses: [entry({ harness: 'opencode', selection: { path: '/x/opencode', version: '1.3.0' },
        observedAt: '2026-11-27T12:00:02Z' })] });
    await later; await flush();
    assert.equal(controller.view('opencode').entry.selection.version, '1.4.0',
        'the older-dated selection never lands under the known one');
    controller.dispose();
});

test('a released Install preflight still tells the mounted sibling that the row is free', async () => {
    const reads = [], intents = new Map(), seen = [];
    const api = {
        harnessMaintenanceInventory: () => new Promise((resolve) => reads.push(resolve)),
        startHarnessMaintenance: async () => operation({ harness: 'opencode' }),
    };
    const a = createHarnessMaintenanceController({ api, intents, newKey: () => 'press-key' });
    const b = createHarnessMaintenanceController({ api, intents, onChange: () => seen.push(intents.get('opencode')?.busy ?? null) });
    const seed = a.refresh({ fresh: true });
    await Promise.resolve();
    reads.shift()({ harnesses: [absentEntry({ harness: 'opencode' })] });
    await seed;
    const press = a.act('opencode', 'install');
    await Promise.resolve();
    a.dispose();                                                       // the pressing view is released mid-read
    reads.shift()({ harnesses: [absentEntry({ harness: 'opencode' })] });
    await press; await flush();
    assert.ok(seen.length, 'the mounted sibling was told');
    assert.equal(seen[seen.length - 1], null, 'and that the row is free again');
    b.dispose();
});

// ---- W5 area 1, the unknown-Install retry's own guard: the press reads its program once before any
// second POST, and a row already read as no longer absent sends nothing at all.

test('a lost Install reads its own program once before Retry and sends nothing when it is installed by then', async () => {
    const sent = [];
    let installed = false;
    const { controller } = setup({
        harnessMaintenanceInventory: async () => ({ harnesses: [installed
            ? entry({ harness: 'fixture', selection: { path: '/vendor/program', version: '2.0.0' } })
            : absentEntry({ harness: 'fixture' })] }),
        startHarnessMaintenance: async (body, key) => { sent.push([body, key]); throw new Error('Connection lost'); },
    });
    await controller.refresh(); await flush();
    await controller.act('fixture', 'install'); await flush();
    assert.equal(sent.length, 1, 'the press was sent once');
    installed = true;                                            // installed elsewhere since the lost reply
    assert.equal(controller.view('fixture').retryBlocked, '', 'the last read still says absent: the row offers Retry');
    await controller.act('fixture', 'rejoin'); await flush();
    assert.equal(sent.length, 1, 'the retry read its program first and sent nothing');
    const view = controller.view('fixture');
    assert.ok(view.request, 'the original key and body stay');
    assert.match(view.retryBlocked, /installed now/);
    const html = harnessMaintenanceMarkup(view.entry, view);
    assert.match(html, /installed now/);
    assert.doesNotMatch(html, /Retry same request/, 'no enabled Retry stands in its place');
});

test('a lost Install whose row no longer reads absent sends nothing and does not even re-read', async () => {
    let listed = true, reads = 0;
    const sent = [];
    const { controller } = setup({
        harnessMaintenanceInventory: async () => { reads += 1; return { harnesses: listed ? [absentEntry({ harness: 'fixture' })] : [] }; },
        startHarnessMaintenance: async (body, key) => { sent.push([body, key]); throw new Error('Connection lost'); },
    });
    await controller.refresh(); await flush();
    await controller.act('fixture', 'install'); await flush();
    assert.equal(sent.length, 1, 'the press was sent once');
    listed = false;                                              // the next read answers without the program
    await controller.refresh(); await flush();
    const before = reads;
    await controller.act('fixture', 'rejoin'); await flush();
    assert.equal(sent.length, 1, 'no second POST without a current absence');
    assert.equal(reads, before, 'the blocked press sends nothing, not even a read');
    const view = controller.view('fixture');
    assert.ok(view.request, 'the original key and body stay');
    assert.match(view.retryBlocked, /no longer finds the program absent/);
    const html = harnessMaintenanceMarkup(view.entry, view);
    assert.doesNotMatch(html, /Retry same request/, 'no enabled Retry stands in its place');
    assert.match(html, /action="inspect"[^>]*>Check again</, 'the read-only Check again stands instead');
});
