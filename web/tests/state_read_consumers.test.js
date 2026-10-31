// The page's other /api/state readers under a request or a body that never settles (#2343): the one Restart
// confirmation (chat header and Settings), Activity, the Dashboard's consciousness fallback and apiClient.state
// (Settings' served interface, Skills). Each keeps its own answer to a refusal, an unparsable body and a
// cancellation; a read that never answers ends at the state lifetime instead of holding its reader forever.
// The transport ignores cancellation, the worst case: the read must settle by itself.
import assert from 'node:assert/strict';
import test from 'node:test';

import * as api from '../modules/api_client.js';
import { confirmAndSendRestart } from '../modules/chat_activity.js';
import { initActivity } from '../modules/activity.js';
import { readConsciousnessFromState } from '../modules/consciousness_api.js';
import { setup } from './dashboard_dom_fixture.js';
import { clock, flush, observe } from './state_read_fixture.js';

// A namespace read keeps this file loadable on a tree without the bounded read: its negative control.
const LIFETIME = api.STATE_READ_LIFETIME_MS ?? 60_000;
const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
const unparsable = () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError('Unexpected token <'); } });
const CENSUS = { active_chat_activities: [{ activity_id: 'a1', phase: 'pausing' }], active_chat_activities_complete: true,
    works: [], bg_consciousness_enabled: true, bg_consciousness_state: { running: true }, served_ui: { source: 'builtin' } };

/** /api/state that never answers: the request itself, or the body after its headers. */
function hanging(kind, signals) {
    return (_url, init = {}) => {
        signals.push(init.signal);
        if (kind === 'fetch') return new Promise(() => {});
        return { ok: true, status: 200, json: () => new Promise(() => {}) };
    };
}

function withFetch(t, route) {
    const prior = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => route(String(url), init);
    t.after(() => { globalThis.fetch = prior; });
}

/** The shared confirmation's page: its dialog answers `confirm`, its socket is connected or not; nothing is sent anywhere. */
function restartPage({ confirm = true, connected = true } = {}) {
    const dialogs = [];
    const sends = [];
    return {
        dialogs, sends,
        openConfirmDialog: async (options) => { dialogs.push(options); return confirm; },
        ws: { send: (frame, options) => { sends.push({ frame, options }); return { status: connected ? 'sent' : 'failed' }; } },
    };
}

// ------------------------------------------------ the Restart confirmation (header and Settings share it)

for (const hang of ['fetch', 'body']) {
    test(`Restart: a census read whose ${hang} never answers still opens the confirmation, as unknown, at its lifetime`, async (t) => {
        const advance = clock(t);
        const signals = [];
        withFetch(t, hanging(hang, signals));
        const page = restartPage({ confirm: false });
        const outcome = observe(confirmAndSendRestart({ openConfirmDialog: page.openConfirmDialog, ws: page.ws }));
        await advance(LIFETIME - 1000);
        assert.deepEqual([page.dialogs.length, outcome.settled], [0, false], 'the confirmation waits on its census read');
        await advance(1000);
        assert.equal(page.dialogs.length, 1, 'the expired read opens the confirmation');
        assert.match(page.dialogs[0].body, /Pause status could not be read/);
        assert.equal(signals[0].aborted, true, 'the expired read cancels its transport');
        await flush();
        assert.deepEqual([outcome.value, page.sends], ['cancelled', []]);
    });
}

for (const [name, route] of [['a refusal', () => reply({ error: 'busy' }, 503)], ['an unparsable body', unparsable],
    ['a failed request', () => { throw new TypeError('Failed to fetch'); }]]) {
    test(`Restart: ${name} opens the confirmation at once, saying the pause status is unknown`, async (t) => {
        withFetch(t, route);
        const page = restartPage({ confirm: false });
        assert.equal(await confirmAndSendRestart({ openConfirmDialog: page.openConfirmDialog, ws: page.ws }), 'cancelled');
        assert.match(page.dialogs[0].body, /Pause status could not be read/);
        assert.deepEqual(page.sends, []);
    });
}

test('Restart: a complete census names the pausing task; confirming sends once without queueing, and offline says so', async (t) => {
    withFetch(t, () => reply(CENSUS));
    const page = restartPage();
    assert.equal(await confirmAndSendRestart({ openConfirmDialog: page.openConfirmDialog, ws: page.ws }), 'sent');
    assert.match(page.dialogs[0].body, /1 task is still pausing/);
    assert.doesNotMatch(page.dialogs[0].body, /could not be read/);
    assert.deepEqual(page.sends, [{ frame: { type: 'command', cmd: '/restart' }, options: { queue: false } }]);
    const offline = restartPage({ connected: false });
    assert.equal(await confirmAndSendRestart({ openConfirmDialog: offline.openConfirmDialog, ws: offline.ws }), 'not_connected');
    assert.deepEqual(offline.sends.map((send) => send.options), [{ queue: false }]);
});

// ------------------------------------------------ Activity

const QUEUE = '/api/tasks?queue_only=1';
const SCHEDULES = '/api/schedules';
const section = (mount, name) => mount.querySelector(`[data-activity-section="${name}"]`);
const message = (node) => node.querySelector('.ui-status').textContent;

function activityPage(t, stateRoute) {
    const { mount, routes, ws } = setup(t);
    routes.set(QUEUE, reply({ queue: { running: [], pending: [] } }));
    routes.set(SCHEDULES, reply({ tasks: [] }));
    routes.set('/api/state', stateRoute);
    return { mount, activity: initActivity({ mount, ws }) };
}

for (const hang of ['fetch', 'body']) {
    test(`Activity: a census read whose ${hang} never answers no longer holds the queue and schedules`, async (t) => {
        const advance = clock(t);
        const signals = [];
        const { mount, activity } = activityPage(t, hanging(hang, signals));
        const refreshed = observe(activity.refresh());
        await advance(LIFETIME - 1000);
        assert.equal(refreshed.settled, false, 'the refresh waits on all three reads');
        assert.equal(section(mount, 'queue').getAttribute('aria-busy'), 'true');
        await advance(1000);
        assert.equal(refreshed.settled, true, 'the expired census read ends the refresh');
        assert.match(section(mount, 'queue').textContent, /Queue empty; live turns unknown/);
        assert.equal(section(mount, 'queue').getAttribute('aria-busy'), null);
        assert.match(section(mount, 'schedules').textContent, /No scheduled tasks/);
        assert.match(message(section(mount, 'background')), /Could not load/);
        assert.equal(section(mount, 'background').querySelectorAll('button').length, 0, 'unknown invents no Start');
        assert.equal(signals[0].aborted, true);
        activity.destroy();
    });
}

test('Activity: a migrating store keeps fetchJson\'s typed answer for the census', async (t) => {
    const { mount, activity } = activityPage(t, reply({ reason_code: 'migrating' }, 503));
    await activity.refresh();
    assert.equal(message(section(mount, 'background')), 'Saved tasks are being upgraded…');
    activity.destroy();
});

test('Activity: a refused census stays unknown beside the queue it does not hold', async (t) => {
    const { mount, activity } = activityPage(t, reply({ error: 'offline' }, 500));
    await activity.refresh();
    assert.match(message(section(mount, 'background')), /Could not load/);
    assert.match(section(mount, 'queue').textContent, /Queue empty; live turns unknown/);
    activity.destroy();
});

// ------------------------------------------------ the Dashboard's consciousness fallback

for (const hang of ['fetch', 'body']) {
    test(`Dashboard fallback: a state read whose ${hang} never answers ends as expired at its lifetime`, async (t) => {
        const advance = clock(t);
        const signals = [];
        withFetch(t, hanging(hang, signals));
        const read = observe(readConsciousnessFromState());
        await advance(LIFETIME - 1000);
        assert.equal(read.settled, false);
        await advance(1000);
        assert.deepEqual([read.error?.name, read.error?.reason], ['StateReadEnded', 'expired']);
        assert.equal(signals[0].aborted, true);
    });
}

test('Dashboard fallback keeps fetchJson\'s answers: typed refusal, unparsable body unavailable, success normalized', async (t) => {
    let route = () => reply({ reason_code: 'migrating' }, 503);
    withFetch(t, (url, init) => route(url, init));
    const refusal = await readConsciousnessFromState().catch((error) => error);
    assert.deepEqual([refusal.status, refusal.body?.reason_code, api.isStoreMigrating(refusal)], [503, 'migrating', true]);
    route = unparsable;
    await assert.rejects(readConsciousnessFromState(), /Consciousness state unavailable/);
    route = () => reply(CENSUS);
    assert.equal((await readConsciousnessFromState()).enabled, true);
});

test('Dashboard fallback: the caller\'s cancellation ends it at once as an AbortError, though the transport ignores it', async (t) => {
    const signals = [];
    withFetch(t, hanging('fetch', signals));
    const controller = new AbortController();
    const read = observe(readConsciousnessFromState({ signal: controller.signal }));
    await flush();
    assert.equal(read.settled, false);
    controller.abort();
    await flush();
    assert.equal(read.error?.name, 'AbortError');
    assert.equal(signals[0].aborted, true, 'the cancellation reaches the transport');
});

// ------------------------------------------------ apiClient.state (Settings' served interface, Skills)

for (const hang of ['fetch', 'body']) {
    test(`apiClient.state: a read whose ${hang} never answers rejects at its lifetime, so Settings and Skills keep their fallbacks`, async (t) => {
        const advance = clock(t);
        const signals = [];
        withFetch(t, hanging(hang, signals));
        const read = observe(api.apiClient.state());
        await advance(LIFETIME - 1000);
        assert.equal(read.settled, false);
        await advance(1000);
        assert.deepEqual([read.error?.name, read.error?.reason], ['StateReadEnded', 'expired']);
        assert.equal(signals[0].aborted, true);
    });
}

test('apiClient.state keeps fetchJson\'s contract for a refusal, an unparsable body and an answer', async (t) => {
    let route = () => reply({ error: 'store busy', reason_code: 'migrating' }, 503);
    withFetch(t, (url, init) => route(url, init));
    const refusal = await api.apiClient.state().catch((error) => error);
    assert.deepEqual([refusal.message, refusal.status, refusal.payload?.reason_code], ['store busy', 503, 'migrating']);
    route = unparsable;
    assert.deepEqual(await api.apiClient.state(), { error: 'non-json response (HTTP 200)' });
    route = () => reply(CENSUS);
    assert.deepEqual((await api.apiClient.state()).served_ui, { source: 'builtin' });
});
