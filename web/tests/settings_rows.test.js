// The row that saves itself (settings_controls.js) and its data (settings_rows.js), against a fake settings server
// that answers like the server with its receipt (`changes`: old, new and the installed value's token).
import assert from 'node:assert/strict';
import test from 'node:test';

import { bindSettingRows, commitSetting, forgetSettingRows, readSettings, settingsRow, writeEntry } from '../modules/settings_controls.js';
import { HAND_WRITTEN, ROWS, rowById, rowsOf } from '../modules/settings_rows.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 8) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };

const option = (value, label) => ({ value, label });
const testRow = (id, control, extra = {}) => ({ id, keys: [id], page: 'test', fold: false, name: id, meta: `${id} meta`, more: '',
    control, aliases: [], writer: 'generic', codec: '', confirm: '', hiddenWhen: '', ...extra });
const SWITCH = testRow('T_SWITCH', { kind: 'switch' }, { codec: 'bool' });
const STRIP = testRow('T_REVIEW', { kind: 'strip', options: [option('advisory', 'Advisory'), option('blocking', 'Blocking')] });
const SECRET = testRow('T_SECRET', { kind: 'secret' });
const ACCESS = testRow('T_ACCESS', { kind: 'strip', options: [option('light', 'Light'), option('pro', 'Pro'), option('cyber_pro', 'Cyber Pro')] },
    { writer: 'runtimeMode', confirm: 'raiseAccess' });
const CHECKER = testRow('OUROBOROS_MODEL_SAFETY', { kind: 'text' }, { writer: 'checker' });
const TEST_ROWS = [SWITCH, STRIP, SECRET, ACCESS, CHECKER];
const [TOTAL, PER_TASK, ALLOWANCE, ROUNDS] = ['TOTAL_BUDGET', 'OUROBOROS_PER_TASK_COST_USD', 'OUROBOROS_CONSCIOUSNESS_DAILY_USD',
    'OUROBOROS_MAX_ROUNDS'].map(rowById);

const EFFECTS = { TOTAL_BUDGET: 'now', OUROBOROS_PER_TASK_COST_USD: 'next_task', OUROBOROS_CONSCIOUSNESS_DAILY_USD: 'next_wake',
    OUROBOROS_MAX_ROUNDS: 'next_task', OUROBOROS_TASK_ABS_CEILING_SEC: 'next_task', T_SWITCH: 'now', T_REVIEW: 'next_task',
    T_SECRET: 'next_task', T_ACCESS: 'restart', OUROBOROS_MODEL_SAFETY: 'next_call' };

function fakeServer(routes, values = {}) {
    const server = { log: [], posts: [], answers: [], version: 0, values: { TOTAL_BUDGET: 200, OUROBOROS_PER_TASK_COST_USD: 50,
        OUROBOROS_CONSCIOUSNESS_DAILY_USD: 10, OUROBOROS_MAX_ROUNDS: 'unlimited', OUROBOROS_TASK_ABS_CEILING_SEC: 600, T_SWITCH: false,
        T_REVIEW: 'advisory', T_SECRET: '', T_ACCESS: 'light', OUROBOROS_MODEL_SAFETY: 'model-a', ...values } };
    server.tokens = Object.fromEntries(Object.keys(server.values).map((key) => [key, `${key}@0`]));
    server.document = () => ({ ...structuredClone(server.values), _meta: { effects: EFFECTS, tokens: { ...server.tokens },
        setup_contract: { budgetFields: [{ stateKey: 'totalBudget', settingKey: 'TOTAL_BUDGET', min: '0.01' },
            { stateKey: 'perTaskCostUsd', settingKey: 'OUROBOROS_PER_TASK_COST_USD', min: '0.01' }] } } });
    server.save = (body) => {
        const changes = {};
        for (const [key, value] of Object.entries(body)) {
            if (key === 'if_unchanged' || key === 'allow_empty_review_pool') continue;
            server.version += 1;
            changes[key] = { old: server.values[key], new: value, token: `${key}@${server.version}` };
            server.values[key] = value;
            server.tokens[key] = changes[key].token;
        }
        return response({ status: 'saved', changes });
    };
    routes.set('/api/settings', async (_url, init = {}) => {
        if (init.method !== 'POST') { server.log.push('GET'); return response(server.document()); }
        const body = JSON.parse(init.body);
        server.log.push('POST');
        server.posts.push(body);
        const answer = server.answers.shift();
        return answer ? answer(body) : server.save(body);
    });
    return server;
}

async function start(t, rows, { values = {}, ctx = {} } = {}) {
    forgetSettingRows();
    const env = setup(t);
    const server = fakeServer(env.routes, values);
    await readSettings();
    return { ...env, server, view: draw(rows, ctx) };
}

function draw(rows, ctx = {}) {
    const container = new NodeStub();
    container.innerHTML = rows.map((row) => settingsRow(row, ctx)).join('');
    const bound = bindSettingRows(container, { rows: TEST_ROWS, ...ctx });
    return {
        container, bound,
        field: (row) => container.querySelector(`#setting-${row.id}`),
        line: (row) => container.querySelector(`#setting-${row.id}-result`).textContent,
        meta: (row) => container.querySelector(`#setting-${row.id}-meta`).textContent,
        undo: () => container.querySelector('[data-setting-undo]'),
        async type(row, text) {
            const input = container.querySelector(`#setting-${row.id}`);
            input.value = text;
            await container.fire('input', { target: input });
            await container.fire('change', { target: input });
            await settle();
        },
        async press(target) { await container.fire('click', { target }); await settle(); },
    };
}

test('a typed value commits on change, never on input, once; it says what it was and Undo re-posts it with the token', async (t) => {
    const { server, view } = await start(t, [TOTAL]);
    const input = view.field(TOTAL);
    assert.equal(input.value, '200');
    for (const text of ['2', '25', '250']) {
        input.value = text;
        await view.container.fire('input', { target: input });
    }
    await settle();
    assert.deepEqual(server.posts, [], 'typing sends nothing');
    await view.container.fire('change', { target: input });
    await settle();
    assert.deepEqual(server.posts, [{ TOTAL_BUDGET: 250 }], 'one body: the row\'s key, no token for an ordinary write');
    assert.equal(view.line(TOTAL), 'Saved · was 200 · Undo');
    await view.press(view.undo());
    assert.deepEqual(server.posts[1], { TOTAL_BUDGET: 200, if_unchanged: { TOTAL_BUDGET: 'TOTAL_BUDGET@1' } });
    assert.equal(input.value, '200');
    assert.equal(view.line(TOTAL), 'Saved · was 250 · Undo');
});

test('Undo restores absence with unset, and restores a previously stored value with an assignment', async (t) => {
    for (const oldAbsent of [true, false]) await t.test(oldAbsent ? 'absent' : 'stored', async (t) => {
        const { server, view } = await start(t, [TOTAL]);
        server.answers.push(async (body) => {
            const receipt = await server.save(body).json();
            if (oldAbsent) receipt.changes.TOTAL_BUDGET.old_absent = true;
            return response(receipt);
        });
        const outcome = await commitSetting(TOTAL, 250);
        assert.equal(outcome.oldAbsent, oldAbsent);
        assert.equal(view.line(TOTAL), 'Saved · was 200 · Undo');
        // The real server resolves the omitted key to its effective default in the receipt.
        if (oldAbsent) server.answers.push(() => server.save({ TOTAL_BUDGET: 200 }));
        await view.press(view.undo());
        assert.deepEqual(server.posts[1], {
            ...(oldAbsent ? { unset: ['TOTAL_BUDGET'] } : { TOTAL_BUDGET: 200 }),
            if_unchanged: { TOTAL_BUDGET: 'TOTAL_BUDGET@1' },
        });
        assert.equal(server.posts.length, 2);
        assert.equal(view.field(TOTAL).value, '200');
        assert.equal(view.line(TOTAL), 'Saved · was 250 · Undo');
    });
});

test('Undo of a value changed on another device says so, shows the current value and writes nothing more', async (t) => {
    const { server, view } = await start(t, [TOTAL]);
    await view.type(TOTAL, '250');
    Object.assign(server.values, { TOTAL_BUDGET: 300 });
    server.tokens.TOTAL_BUDGET = 'elsewhere';
    server.answers.push(() => response({ error: 'TOTAL_BUDGET changed since it was read.', saved: false, code: 'settings_changed_elsewhere' }, 409));
    await view.press(view.undo());
    assert.deepEqual(server.posts.at(-1), { TOTAL_BUDGET: 200, if_unchanged: { TOTAL_BUDGET: 'TOTAL_BUDGET@1' } });
    assert.equal(view.line(TOTAL), 'Changed on another device. This is the current value.');
    assert.equal(view.field(TOTAL).value, '300');
    assert.equal(server.posts.length, 2);
    assert.equal(server.log.at(-1), 'GET', 'the row read the current value after the refusal');
});

test('one row is one queue: a value given during a write waits, the newest replaces it, and it goes after alone', async (t) => {
    const { server, view, document } = await start(t, [PER_TASK]);
    const gate = deferred();
    server.answers.push((body) => gate.promise.then(() => server.save(body)));
    const input = view.field(PER_TASK);
    input.focus();
    const first = commitSetting(PER_TASK, 60);
    await settle();
    assert.equal(input.disabled, true, 'the control in flight is disabled');
    assert.equal(input.getAttribute('aria-busy'), 'true');
    assert.equal(view.line(PER_TASK), '', 'nothing else is drawn for saving');
    document.activeElement = document.body;  // disabling took the focus away
    const second = commitSetting(PER_TASK, 70);
    const third = commitSetting(PER_TASK, 80);
    await settle();
    assert.deepEqual(server.posts, [{ OUROBOROS_PER_TASK_COST_USD: 60 }], 'never a second write beside the one in flight');
    gate.resolve();
    const outcomes = await Promise.all([first, second, third]);
    await settle();
    assert.deepEqual(server.posts.map((body) => body.OUROBOROS_PER_TASK_COST_USD), [60, 80], 'the newest waiting value, alone');
    assert.equal(outcomes[1], outcomes[2], 'the replaced value answers what replaced it');
    assert.equal(input.disabled, false);
    assert.equal(input.getAttribute('aria-busy'), null);
    assert.equal(document.activeElement, input, 'the focus comes back to the control when it is free');
    assert.equal(view.line(PER_TASK), 'Saved · applies to new tasks · was 60 · Undo');
});

test('a refusal keeps the typed text and says the server\'s sentence; a value refused here sends nothing', async (t) => {
    const { server, view } = await start(t, [TOTAL, ALLOWANCE, ROUNDS]);
    server.answers.push(() => response({ error: "OUROBOROS_MAX_ROUNDS must be a positive integer or 'unlimited'.", saved: false }, 400));
    await view.type(ROUNDS, '0');
    assert.deepEqual(server.posts, [{ OUROBOROS_MAX_ROUNDS: 0 }], 'a bound that may be unlimited goes to the server as typed');
    assert.equal(view.line(ROUNDS), "Not saved. OUROBOROS_MAX_ROUNDS must be a positive integer or 'unlimited'.");
    assert.equal(view.field(ROUNDS).value, '0');
    await readSettings();
    view.bound.paint();
    assert.equal(view.field(ROUNDS).value, '0', 'a refresh does not put the stored value over the refused text');
    await view.type(TOTAL, 'abc');
    await view.type(ALLOWANCE, '-1');
    assert.equal(server.posts.length, 1);
    assert.equal(view.line(TOTAL), 'Not saved. Enter a number, 0.01 or more.', 'the setup contract\'s bound');
    assert.equal(view.line(ALLOWANCE), 'Not saved. Enter a number, 0 or more.', 'the server would clamp it, so the row refuses it');
    assert.equal(view.field(ALLOWANCE).value, '-1');
    await view.container.fire('input', { target: view.field(ALLOWANCE) });
    assert.equal(view.line(ALLOWANCE), '', 'an edit of the row ends its result');
});

test('"No limit" is its own choice beside the number', async (t) => {
    const { server, view } = await start(t, [ROUNDS], { values: { OUROBOROS_MAX_ROUNDS: 40 } });
    const none = view.container.querySelector('[data-setting-no-limit]');
    assert.equal(none.getAttribute('aria-pressed'), 'false');
    await view.press(none);
    assert.deepEqual(server.posts, [{ OUROBOROS_MAX_ROUNDS: 'unlimited' }]);
    assert.equal(none.getAttribute('aria-pressed'), 'true');
    assert.equal(view.field(ROUNDS).value, '', 'the field is empty under its "No limit" placeholder');
    assert.equal(view.line(ROUNDS), 'Saved · applies to new tasks · was 40 · Undo');
});

test('an unknown outcome says "Not confirmed yet", reads again before anything else is sent, then says what happened', async (t) => {
    const { server, view } = await start(t, [ALLOWANCE]);
    server.answers.push(() => response({ error: 'The save did not answer in time.', saved: null }, 503));
    await view.type(ALLOWANCE, '12');
    assert.equal(view.line(ALLOWANCE), 'Not confirmed yet');
    assert.deepEqual(server.log, ['GET', 'POST', 'GET'], 'the row read again at once');
    await view.type(ALLOWANCE, '13');
    assert.deepEqual(server.log, ['GET', 'POST', 'GET', 'GET', 'POST'], 'a read before the next write');
    assert.equal(view.line(ALLOWANCE), 'Saved · applies from the next wake · was 10 · Undo');

    server.answers.push(() => response({ error: 'The save did not answer in time.', saved: null }, 503));
    await view.type(ALLOWANCE, '14');
    assert.equal(view.line(ALLOWANCE), 'Not confirmed yet');
    server.values.OUROBOROS_CONSCIOUSNESS_DAILY_USD = 14;  // the timed-out save did land
    await readSettings();
    assert.equal(view.line(ALLOWANCE), 'Saved', 'a later read that shows the value says it saved, without a receipt');
});

test('saved, then a later step failed: the line says both', async (t) => {
    const { server, view } = await start(t, [TOTAL]);
    server.answers.push((body) => {
        server.save(body);
        return response({ error: 'Settings were saved to disk, but the restart_state step failed afterwards: OSError: disk',
            status: 'saved_with_post_commit_error', saved: true, post_commit_failed: 'restart_state' }, 500);
    });
    await view.type(TOTAL, '210');
    const result = view.container.querySelector('#setting-TOTAL_BUDGET-result');
    assert.match(view.line(TOTAL), /^Saved\. The restart_state step failed afterwards\./);
    assert.match(result.querySelector('.ui-result-details').textContent, /OSError: disk/, 'the raw text, one disclosure away');
    assert.equal(view.undo(), null);
    assert.equal(server.posts.length, 1, 'never re-saved');
});

test('no timer ends a result: it stays until the next result, an edit, or the page being left', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
    const { view } = await start(t, [TOTAL]);
    await view.type(TOTAL, '220');
    t.mock.timers.tick(60 * 60 * 1000);
    await settle();
    assert.equal(view.line(TOTAL), 'Saved · was 200 · Undo');
    view.bound.leave();
    assert.equal(view.line(TOTAL), '', 'leaving the page ends a result the owner saw');
});

test('without a receipt a save says only "Saved"; a withheld value has no was and no Undo', async (t) => {
    const { server, view } = await start(t, [TOTAL, SECRET]);
    server.answers.push((body) => { server.save(body); return response({ status: 'saved', receipt: 'unavailable' }); });
    await view.type(TOTAL, '230');
    assert.equal(view.line(TOTAL), 'Saved');
    server.answers.push(() => response({ status: 'saved', changes: { T_SECRET: { withheld: 'secret' } } }));
    await view.type(SECRET, 'sk-new');
    assert.deepEqual(server.posts.at(-1), { T_SECRET: 'sk-new' });
    assert.equal(view.line(SECRET), 'Saved · applies to new tasks');
    assert.equal(view.field(SECRET).value, '', 'a secret field never shows the stored secret');
    assert.equal(view.undo(), null);
    // An older server (no `changes` at all) is the same honest "Saved".
    server.answers.push((body) => { server.save(body); return response({ status: 'saved', next_task_changed: true }); });
    await view.type(TOTAL, '240');
    assert.equal(view.line(TOTAL), 'Saved');
});

test('a switch or strip that acts now is quiet on success and speaks on refusal; a delayed one says when', async (t) => {
    const { server, view } = await start(t, [SWITCH, STRIP]);
    const box = view.field(SWITCH);
    assert.equal(box.checked, false);
    assert.equal(box.getAttribute('role'), 'switch');
    box.checked = true;
    await view.container.fire('click', { target: box });
    assert.deepEqual(server.posts, [], 'the press is not the commit; its change is');
    await view.container.fire('change', { target: box });
    await settle();
    assert.deepEqual(server.posts, [{ T_SWITCH: true }]);
    assert.equal(view.line(SWITCH), '', 'the control is its own confirmation');
    assert.equal(box.getAttribute('aria-checked'), 'true');
    server.answers.push(() => response({ error: 'Turning this off needs Cyber Pro.', saved: false }, 403));
    box.checked = false;
    await view.container.fire('change', { target: box });
    await settle();
    assert.equal(view.line(SWITCH), 'Not saved. Turning this off needs Cyber Pro.');
    assert.equal(box.checked, true, 'the switch shows what is stored');

    const blocking = view.container.querySelector('[data-setting-value="blocking"]');
    await view.press(blocking);
    assert.deepEqual(server.posts.at(-1), { T_REVIEW: 'blocking' });
    assert.equal(view.line(STRIP), 'Saved · applies to new tasks · was Advisory · Undo');
    assert.equal(blocking.getAttribute('aria-pressed'), 'true');
    assert.equal(blocking.classList.contains('active'), true);
});

test('a write waiting for a restart says so and shows both values; its owner endpoint takes no token, so no Undo', async (t) => {
    const { routes, server, view } = await start(t, [ACCESS], { ctx: { ask: async () => true } });
    const sent = [];
    routes.set('/api/owner/runtime-mode', async (_url, init) => {
        sent.push(JSON.parse(init.body));
        server.values.T_ACCESS = 'pro';
        return response({ ok: true, changes: { T_ACCESS: { old: 'light', new: 'pro', token: 'T_ACCESS@1' } },
            restart_state: { running: { T_ACCESS: 'light' } } });
    });
    const asked = [];
    view.bound.dispose();
    const again = draw([ACCESS], { ask: async (dialog) => { asked.push(dialog); return true; } });
    await again.press(again.container.querySelector('[data-setting-value="pro"]'));
    assert.deepEqual(sent, [{ mode: 'pro' }]);
    assert.deepEqual(asked.map((dialog) => dialog.title), ['Raise Access'], 'raising Access asks first');
    assert.equal(again.line(ACCESS), 'Saved · applies after restart');
    assert.equal(again.meta(ACCESS), 'Running: Light · after restart: Pro');
    assert.equal(again.undo(), null);
});

test('a page torn down before the answer shows the refusal and the typed text when drawn again, after one request', async (t) => {
    const { server, view } = await start(t, [PER_TASK]);
    const gate = deferred();
    server.answers.push(() => gate.promise.then(() => response({ error: 'The per-task cap must be at least $0.01.', saved: false }, 400)));
    view.field(PER_TASK).value = '0.5';
    await view.container.fire('change', { target: view.field(PER_TASK) });
    await settle();
    view.bound.dispose();  // the page is gone while the write is in flight
    gate.resolve();
    await settle();
    const again = draw([PER_TASK]);
    assert.equal(again.line(PER_TASK), 'Not saved. The per-task cap must be at least $0.01.');
    assert.equal(again.field(PER_TASK).value, '0.5');
    assert.equal(server.posts.length, 1, 'nothing submitted twice');
    again.bound.leave();
    assert.equal(again.line(PER_TASK), '');
});

test('a checker key goes through the owner\'s checker transaction; a roster with no reviewer asks, then sends the same', async (t) => {
    const { routes, server, view } = await start(t, [CHECKER]);
    const sent = [];
    routes.set('/api/owner/safety-model', async (_url, init) => {
        const body = JSON.parse(init.body);
        sent.push(body);
        const next = body.settings.OUROBOROS_MODEL_SAFETY;
        if (next === 'model-b' && !body.allow_empty_review_pool) {
            return response({ error: 'This leaves no agent that reviews.', saved: false, code: 'empty_review_pool' }, 400);
        }
        return response({ ok: true, changes: { OUROBOROS_MODEL_SAFETY: { old: next === 'model-b' ? 'model-a' : 'model-b', new: next, token: `S@${sent.length}` } } });
    });
    view.bound.dispose();
    const asked = [];
    const again = draw([CHECKER], { ask: async (dialog) => { asked.push(dialog); return true; } });
    await again.type(CHECKER, 'model-b');
    assert.deepEqual(server.posts, [], 'never the generic save');
    assert.deepEqual(sent, [{ settings: { OUROBOROS_MODEL_SAFETY: 'model-b' } },
        { settings: { OUROBOROS_MODEL_SAFETY: 'model-b' }, allow_empty_review_pool: true }]);
    assert.equal(asked[0].body, 'This leaves no agent that reviews.');
    assert.equal(again.line(CHECKER), 'Saved · applies from the next model call · was model-a · Undo');
    await again.press(again.undo());
    assert.deepEqual(sent.at(-1), { settings: { OUROBOROS_MODEL_SAFETY: 'model-a' }, if_unchanged: { OUROBOROS_MODEL_SAFETY: 'S@2' } });
});

test('writeEntry changes one entry of a fresh read with its token, and a stale list is refused, never retried', async (t) => {
    const servers = [{ name: 'a', url: 'http://a' }, { name: 'b', url: 'http://b' }];
    const { server } = await start(t, [], { values: { MCP_SERVERS: JSON.stringify(servers) } });
    const outcome = await writeEntry('MCP_SERVERS', 'b', { url: 'http://b2' });
    assert.equal(outcome.state, 'saved');
    assert.deepEqual(server.posts, [{ MCP_SERVERS: [servers[0], { name: 'b', url: 'http://b2' }], if_unchanged: { MCP_SERVERS: 'MCP_SERVERS@0' } }]);
    server.answers.push(() => response({ error: 'MCP_SERVERS changed since it was read.', saved: false, code: 'settings_changed_elsewhere' }, 409));
    const stale = await writeEntry('MCP_SERVERS', 'a', { url: 'http://a2' });
    assert.equal(stale.state, 'changed_elsewhere');
    assert.equal(server.posts.length, 2, 'the refused list is not sent again');
    const asked = [];
    const removed = await writeEntry('MCP_SERVERS', 'a', null, { name: 'Tool server a', ask: async (dialog) => { asked.push(dialog.title); return false; } });
    assert.equal(removed.state, 'unchanged');
    assert.deepEqual(asked, ['Remove Tool server a?'], 'removing an entry asks first');
    assert.equal(server.posts.length, 2);
});

test('writeEntry: the empty-pool refusal asks and repeats the same body with the flag; an editor\'s adapter carries the write', async (t) => {
    const roster = { items: [{ subagent_id: 'helper', enabled: true, review: true }] };
    const { server } = await start(t, [], { values: { OUROBOROS_SUBAGENTS: JSON.stringify(roster) } });
    server.answers.push(() => response({ error: 'No row is marked Reviewer.', saved: false, code: 'empty_review_pool' }, 400));
    const asked = [];
    const outcome = await writeEntry('OUROBOROS_SUBAGENTS', 'helper', { review: false }, { ask: async (dialog) => { asked.push(dialog); return true; } });
    assert.equal(outcome.state, 'saved');
    assert.equal(asked[0].body, 'No row is marked Reviewer.');
    assert.deepEqual(server.posts[1], { ...server.posts[0], allow_empty_review_pool: true }, 'the same roster, never a newer one');
    assert.equal(server.log.filter((step) => step === 'GET').length, 2, 'one read for the write, none between the two posts');

    const carried = [];
    const viaChecker = await writeEntry('OUROBOROS_SUBAGENTS', 'helper', { enabled: false },
        { checker: async ({ body }) => { carried.push(body); return { ok: true }; } });
    assert.equal(viaChecker.state, 'saved');
    assert.equal(server.posts.length, 2, 'the adapter wrote it, not the generic save');
    assert.equal(carried[0].OUROBOROS_SUBAGENTS.items[0].enabled, false);
    assert.ok(carried[0].if_unchanged.OUROBOROS_SUBAGENTS);
});

test('the rows are data: they survive JSON, ids are unique, and every key has exactly one row', () => {
    assert.deepEqual(JSON.parse(JSON.stringify(ROWS)), ROWS);
    assert.deepEqual(JSON.parse(JSON.stringify(HAND_WRITTEN)), HAND_WRITTEN);
    const ids = [...ROWS, ...HAND_WRITTEN].map((row) => row.id);
    assert.equal(new Set(ids).size, ids.length);
    const keys = [...ROWS, ...HAND_WRITTEN].flatMap((row) => row.keys);
    assert.equal(new Set(keys).size, keys.length, 'one key, one control');
    for (const row of ROWS) {
        assert.equal(typeof row.name, 'string');
        assert.ok(row.name && row.page && row.control?.kind, row.id);
        assert.ok(Array.isArray(row.aliases), row.id);
    }
    assert.deepEqual(rowsOf('dashboard:costs', { fold: false }).map((row) => row.id),
        ['TOTAL_BUDGET', 'OUROBOROS_PER_TASK_COST_USD', 'OUROBOROS_CONSCIOUSNESS_DAILY_USD']);
    assert.deepEqual(rowsOf('dashboard:costs', { fold: true }).map((row) => row.id), ['OUROBOROS_MAX_ROUNDS', 'OUROBOROS_TASK_ABS_CEILING_SEC']);
    assert.equal(rowById('TOTAL_BUDGET').control.from, 'setup:totalBudget', 'the setup contract describes the budget fields');
    assert.equal(rowById('nope'), null);
});
