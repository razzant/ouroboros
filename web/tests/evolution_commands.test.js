import assert from 'node:assert/strict';
import test from 'node:test';
import { installDomDouble } from './dom_double.js';
import { createStateSnapshotSequencer } from '../modules/chat_activity.js';
import { initEvolution } from '../modules/evolution.js';
import { WS } from '../modules/ws.js';

const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(setImmediate); };
const NO_HISTORY = { versions: [], saved_at: '', refresh: { state: 'idle' }, install: {} };

function mountEvolution(env, state = { activePage: 'dashboard', dashboardActiveSubtab: 'evolution' }, dialog = undefined) {
    const mount = env.doc.createElement('div');
    env.doc.body.append(mount);
    const ws = new WS('ws://unused');
    return { ws, dispose: initEvolution({ ws, mount, state, dialog }).dispose };
}

test('Pause stays disabled while its command is in flight, a refused send is said, and light mode keeps Start off', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });  // the toast's own dismissal timer
    const env = installDomDouble();
    const savedFetch = globalThis.fetch;
    const commands = [];
    let answer;
    globalThis.fetch = async (url, init) => {
        if (String(url) === '/api/command') {
            commands.push(JSON.parse(init.body).cmd);
            return new Promise((resolve) => { answer = resolve; });
        }
        return Response.json(String(url).includes('evolution-data') ? NO_HISTORY
            : { runtime_mode: 'light', evolution_state: { status: 'state_unknown', enabled: null } });
    };
    let dispose;
    try {
        const mounted = mountEvolution(env);
        dispose = mounted.dispose;
        mounted.ws.emit('open');
        await flush();
        const start = env.doc.getElementById('evo-start');
        const pause = env.doc.getElementById('evo-stop');
        assert.equal(start.disabled, true, 'light mode blocks a campaign start');
        assert.equal(pause.disabled, false);
        assert.deepEqual([start.hidden, pause.hidden], [false, false], 'an unknown control offers both commands');
        assert.match(env.doc.getElementById('evo-campaign-chip').textContent, /^Unknown$/);

        pause.focus();
        pause.click();
        await flush();
        assert.deepEqual([pause.disabled, start.disabled], [true, true], 'both controls wait for the gateway');
        pause.dispatchEvent(new Event('click'));
        await flush();
        assert.deepEqual(commands, ['/evolve off'], 'a second press while in flight sends nothing');

        env.doc.activeElement = env.doc.body;  // what a browser does when the pressed button is disabled
        answer(Response.json({ error: 'Complete provider setup before sending this command.' }, { status: 409 }));
        await flush();
        assert.equal(pause.disabled, false);
        assert.equal(start.disabled, true, 'light mode still blocks Start after the command settles');
        assert.equal(env.doc.activeElement, pause, 'the pressed control gets keyboard focus back');
        const [toast] = env.doc.getElementById('toast-stack').children;
        assert.equal(toast.textContent, 'Could not send /evolve off: Complete provider setup before sending this command.');
        assert.equal(toast.getAttribute('role'), 'alert');
    } finally {
        dispose?.();
        globalThis.fetch = savedFetch;
        env.restore();
    }
});

test('the campaign offers the commands its state allows: Start when off, Pause and a new objective while one is on', async () => {
    const env = installDomDouble();
    const savedFetch = globalThis.fetch;
    let runtime = { runtime_mode: 'advanced', evolution_state: { status: 'disabled', enabled: false, detail: 'Evolution mode is off.' } };
    globalThis.fetch = async (url) => Response.json(String(url).includes('evolution-data') ? NO_HISTORY : runtime);
    let dispose;
    try {
        const mounted = mountEvolution(env);
        dispose = mounted.dispose;
        mounted.ws.emit('open');
        await flush();
        const start = env.doc.getElementById('evo-start');
        const pause = env.doc.getElementById('evo-stop');
        assert.deepEqual([start.hidden, start.disabled, pause.hidden], [false, false, true]);
        assert.equal(env.doc.getElementById('evo-campaign-note').textContent, 'Improvement cycles run only during a campaign.');
        assert.equal(env.doc.getElementById('evo-campaign-chip').textContent, 'Off');

        runtime = { runtime_mode: 'advanced', evolution_state: { status: 'running', enabled: true, detail: 'Evolution task is running now.',
            cycle: 3, budget_remaining_usd: 12.5, campaign: { objective: 'Make the history tab fast', cycles_done: 2 } } };
        mounted.ws.emit('open');
        await flush();
        assert.deepEqual([start.hidden, start.textContent, start.disabled, pause.hidden, pause.disabled], [false, 'New objective…', false, false, false],
            'the same command gives a running campaign a new objective in place');
        const chip = env.doc.getElementById('evo-campaign-chip');
        assert.deepEqual([chip.textContent, chip.getAttribute('data-tone')], ['Running', 'ok']);
        assert.equal(env.doc.getElementById('evo-campaign-note').textContent, 'Evolution task is running now.');
        const lines = env.doc.body.querySelectorAll('.evo-campaign-lines p').map((node) => node.textContent);
        assert.deepEqual(lines, ['Objective: Make the history tab fast', 'Cycle 3 · 2 attempts · Budget left $12.50']);
    } finally {
        dispose?.();
        globalThis.fetch = savedFetch;
        env.restore();
    }
});

test('Start waits for a current state: off while loading or after a failed read, and a stale answer says its age', async () => {
    const env = installDomDouble();
    const savedFetch = globalThis.fetch;
    let answerState;
    let mode = 'pending';
    globalThis.fetch = (url) => {
        if (String(url).includes('evolution-data')) return Promise.resolve(Response.json(NO_HISTORY));
        if (mode === 'pending') return new Promise((resolve) => { answerState = resolve; });
        if (mode === 'fail') return Promise.reject(new TypeError('Failed to fetch'));
        return Promise.resolve(Response.json({ runtime_mode: 'advanced', evolution_enabled: true, bg_consciousness_enabled: true,
            evolution_state: { status: 'running', enabled: true, detail: 'Evolution task is running now.' },
            bg_consciousness_state: { status: 'sleeping', spent_24h_usd: 1.25, daily_usd: 10, unknown_unmetered: 1, integrity_degraded: true } }));
    };
    let dispose;
    try {
        const mounted = mountEvolution(env);
        dispose = mounted.dispose;
        const start = () => env.doc.getElementById('evo-start');
        const pause = () => env.doc.getElementById('evo-stop');
        const freshness = () => env.doc.getElementById('evo-campaign-freshness');
        mounted.ws.emit('open');
        await flush();
        assert.equal(env.doc.getElementById('evo-campaign-note').textContent, 'Reading the runtime state…');
        assert.equal(start().disabled, true, 'no paid start before the state is known');

        mode = 'fail';
        answerState(Response.json({}, { status: 503 }));
        await flush();
        assert.match(env.doc.getElementById('evo-campaign-note').textContent, /could not be read/);
        assert.equal(start().disabled, true, 'an unread state keeps Start off');
        assert.equal(freshness().hidden, true, 'nothing older to date');

        mode = 'ok';
        mounted.ws.emit('open');
        await flush();
        assert.deepEqual([start().textContent, pause().hidden, pause().disabled], ['New objective…', false, false]);
        assert.equal(env.doc.getElementById('evo-consciousness-allowance').textContent,
            'Allowance (24 h): ≥ $1.25 / $10.00 (ledger integrity degraded)');
        assert.equal(freshness().hidden, true);

        mode = 'fail';
        mounted.ws.emit('open');
        await flush();
        assert.equal(env.doc.getElementById('evo-campaign-chip').textContent, 'Running', 'the last answer stays readable');
        assert.equal(freshness().hidden, false, 'and is marked as not current');
        assert.match(freshness().textContent, /^Couldn’t read the runtime state again; this is the answer from today at \d\d:\d\d\.$/);
        assert.equal(freshness().getAttribute('data-tone'), 'warn');
        assert.equal(start().disabled, true, 'Start stays off until a current answer');
        assert.equal(pause().disabled, false, 'stopping stays possible');

        mode = 'ok';
        mounted.ws.emit('open');
        await flush();
        assert.equal(freshness().hidden, true, 'a new answer clears the mark');
    } finally {
        dispose?.();
        globalThis.fetch = savedFetch;
        env.restore();
    }
});

test('the campaign follows every state answer of the page: a Pause the runtime applies late arrives with a later page read', async () => {
    const env = installDomDouble();
    const savedFetch = globalThis.fetch;
    const running = { runtime_mode: 'advanced', evolution_state: { status: 'running', enabled: true, detail: 'Evolution task is running now.' } };
    let runtime = running;
    const commands = [];
    const fetchState = async (url, init) => {
        if (String(url).includes('evolution-data')) return Response.json(NO_HISTORY);
        if (String(url) === '/api/command') {
            commands.push(JSON.parse(init.body).cmd);
            return Response.json({ status: 'ok' });  // queued: the runtime applies it later
        }
        return Response.json(runtime);
    };
    globalThis.fetch = fetchState;
    // app.js: one sequencer for every /api/state read on the page, its answers handed to the open rooms and to Evolution.
    let evolution = null;
    const applied = [];
    const snapshots = createStateSnapshotSequencer((data) => { applied.push(data); evolution?.hydrateStateSnapshot(data); },
        undefined, () => evolution?.hydrateStateSnapshot(null));
    const pageRead = async () => {  // app.js's own periodic read (refreshProjectsNav(false))
        const request = await snapshots.gate(false);
        if (!request) return;
        try {
            const response = await fetch('/api/state');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            snapshots.apply(request, await response.json());
        } catch {
            snapshots.fail(request);
        }
    };
    try {
        const mount = env.doc.createElement('div');
        env.doc.body.append(mount);
        const ws = new WS('ws://unused');
        evolution = initEvolution({ ws, mount, state: { activePage: 'dashboard', dashboardActiveSubtab: 'evolution' }, stateSnapshots: snapshots });
        ws.emit('open');
        await flush();
        const start = env.doc.getElementById('evo-start');
        const pause = env.doc.getElementById('evo-stop');
        const chip = env.doc.getElementById('evo-campaign-chip');
        assert.deepEqual([chip.textContent, start.textContent, pause.hidden], ['Running', 'New objective…', false]);
        assert.equal(applied.length, 1, 'its own read runs on the page’s clock, so every consumer of the page has it');

        pause.click();
        await flush();
        assert.deepEqual(commands, ['/evolve off']);
        assert.equal(applied.length, 2, 'the read after the command');
        assert.deepEqual([chip.textContent, start.textContent, pause.hidden, pause.disabled], ['Running', 'New objective…', false, false],
            'the runtime has only queued the Pause');

        runtime = { runtime_mode: 'advanced', evolution_state: { status: 'disabled', enabled: false } };
        await pageRead();
        await flush();
        assert.deepEqual([chip.textContent, start.textContent, start.disabled, pause.hidden], ['Off', 'Start campaign…', false, true],
            'the page’s next read brings what the runtime applied, without a poll of Evolution’s own');

        globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };
        await pageRead();
        await flush();
        const freshness = env.doc.getElementById('evo-campaign-freshness');
        assert.equal(freshness.hidden, false, 'a failed page read marks the answer on screen as not current');
        assert.equal(start.disabled, true, 'and a paid Start waits for a current one');
        globalThis.fetch = fetchState;
        await pageRead();
        await flush();
        assert.deepEqual([freshness.hidden, start.disabled], [true, false]);
        evolution.dispose();
        runtime = running;
        await pageRead();
        await flush();
        assert.equal(chip.textContent, 'Off', 'a disposed page paints nothing');
    } finally {
        evolution?.dispose();
        globalThis.fetch = savedFetch;
        env.restore();
    }
});

test('Run review… stands beside the campaign and opens the one reviewer picker; only a confirmed choice sends /review', async () => {
    const env = installDomDouble();
    const savedFetch = globalThis.fetch;
    globalThis.fetch = async (url) => Response.json(String(url).startsWith('/api/settings')
        ? { OUROBOROS_SUBAGENTS: { items: [{ subagent_id: 'row-a', enabled: true }] } }
        : String(url).includes('evolution-data') ? NO_HISTORY
            : { runtime_mode: 'advanced', evolution_state: { status: 'disabled', enabled: false } });
    const asked = [];
    let answer = { confirmed: false };
    let dispose;
    try {
        const mounted = mountEvolution(env, undefined, async (options) => { asked.push(options); return answer; });
        dispose = mounted.dispose;
        const sent = [];
        mounted.ws.send = (frame) => { sent.push(frame); return { status: 'sent' }; };
        mounted.ws.emit('open');
        await flush();
        const review = env.doc.getElementById('evo-review');
        assert.deepEqual(review.parent.children.map((node) => node.id), ['evo-start', 'evo-stop', 'evo-review']);
        assert.equal(review.textContent, 'Run review…');
        review.click();
        await flush();
        assert.equal(asked[0].title, 'Deep self-review', 'the picker /review uses');
        assert.deepEqual(asked[0].choices.map((choice) => choice.value), ['', 'row-a'], 'the Main model, then the enabled rows');
        assert.deepEqual(sent, [], 'a cancelled picker sends nothing');
        answer = { confirmed: true, value: 'row-a' };
        review.click();
        await flush();
        assert.deepEqual(sent, [{ type: 'command', cmd: '/review row-a' }]);
    } finally {
        dispose?.();
        globalThis.fetch = savedFetch;
        env.restore();
    }
});
