// The Summary's optional Consciousness row and its one ordinary Start (#1603): offered only by the server's
// completion record, unticked, never part of the save; after a confirmed save a ticked row waits for THIS
// server's chat queue, sends `/bg start` at most once and calls it on only when the state shows it.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    createConsciousnessOffer, limitPhrases, normalizeOffer, offerRowHtml, runScreenHtml, startConsciousnessAfterSave as startFlow,
} from '../modules/onboarding_consciousness.js';

const text = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const DEFAULTS = { offered: true, enabled: null, autonomy: 'act', dailyUsd: 20, maxTasks: 2 };
// These flow cases begin with Finish having observed S1; unknown and replaced identities are explicit cases.
const startConsciousnessAfterSave = (options) => startFlow({ expectedSince: 'S1', ...options });

/** A scripted gateway: `health` and `state` answers in order (the last repeats), `command` one answer; every call recorded. */
function gateway({ health = [], state = [], command = { status: 200, body: { status: 'ok' } } } = {}) {
    const calls = [];
    const queue = { health: [...health], state: [...state] };
    const next = (name) => (queue[name].length > 1 ? queue[name].shift() : queue[name][0]);
    const reply = (answer) => {
        if (answer instanceof Error) throw answer;
        return { ok: answer.status >= 200 && answer.status < 300, status: answer.status, json: async () => answer.body };
    };
    const fetchImpl = async (url, init = {}) => {
        calls.push({ url, method: init.method || 'GET', body: init.body || '' });
        if (url === '/api/health') return reply(next('health'));
        if (url === '/api/state') return reply(next('state'));
        if (url === '/api/command') return reply(command);
        throw new Error(`unexpected ${url}`);
    };
    return { fetchImpl, calls, posts: () => calls.filter((call) => call.url === '/api/command') };
}
const boot = (fields) => ({ status: 200, body: { status: 'ok', boot: { phase: 'ready', since: 'S1', dialogue_ready: false, ...fields } } });
const bg = (value) => ({ status: 200, body: { bg_consciousness_enabled: value } });
const clock = () => { let t = 0; return { now: () => t, wait: async (ms) => { t += ms; } }; };

test('the offer: an older bootstrap offers nothing; values keep unknown as unknown', () => {
    assert.equal(normalizeOffer(undefined).offered, false);
    assert.equal(offerRowHtml(normalizeOffer({})), '');
    assert.deepEqual(normalizeOffer({ offered: true, enabled: 'yes', autonomy: 'act', dailyUsd: '', maxTasks: 'x' }),
        { offered: true, enabled: null, autonomy: 'act', dailyUsd: null, maxTasks: null });
});

test('the row names the effective limits in Settings and Costs words, unticked, and where each is changed', () => {
    const html = offerRowHtml(normalizeOffer(DEFAULTS));
    assert.match(html, /<input type="checkbox" class="ui-checkbox" data-consciousness-start>/);
    assert.doesNotMatch(html, /checked/);
    assert.equal(text(html), 'Consciousness Start consciousness with Ouroboros '
        + 'Between your requests it continues topics that interest it and you, notices opportunities and brings you suggestions. '
        + 'Autonomy: Acts. Also acts on its own: starts work, runs commands, speaks and acts in places. It does not change itself. '
        + 'Up to $20.00 a day in recorded spending. Subscription quota is separate. At most 2 of its own tasks at once. '
        + 'Change the allowance in Dashboard › Costs. Its other settings, and Stop and Start, are in Dashboard › Consciousness.');
    assert.deepEqual(limitPhrases(normalizeOffer({ ...DEFAULTS, autonomy: 'observe', dailyUsd: 5, maxTasks: 0 })), [
        'Autonomy: Advises. On its own it studies, keeps its memory and tells you; it changes nothing outside.',
        'Up to $5.00 a day in recorded spending. Subscription quota is separate.', 'It starts no tasks of its own.']);
    assert.deepEqual(limitPhrases(normalizeOffer({ ...DEFAULTS, autonomy: 'full', dailyUsd: 0, maxTasks: 1 })).slice(1), [
        'Daily allowance $0.00: it may not spend.', 'At most 1 of its own tasks at once.']);
    assert.deepEqual(limitPhrases(normalizeOffer({ offered: true })), [
        'Autonomy unknown.', 'Daily allowance unknown.', 'Its own-task limit is unknown.']);
});

test('a consciousness already on is not offered a Start', () => {
    const offer = createConsciousnessOffer({ ...DEFAULTS, enabled: true });
    const host = { innerHTML: '', addEventListener() {}, removeEventListener() {} };
    offer.mount(host);
    assert.match(text(host.innerHTML), /^Consciousness On already Autonomy: Acts\./);
    assert.doesNotMatch(host.innerHTML, /checkbox/);
    assert.equal(offer.wanted(), false);
});

test('the tick lives in this page across renders and is the only thing that wants a Start', () => {
    const offer = createConsciousnessOffer(DEFAULTS);
    const handlers = {};
    const host = () => ({ innerHTML: '', addEventListener: (type, fn) => { handlers[type] = fn; }, removeEventListener() {} });
    const first = host();
    offer.mount(first);
    assert.equal(offer.wanted(), false);
    const box = { checked: true, closest: (selector) => (selector === '[data-consciousness-start]' ? box : null) };
    handlers.change({ target: box });
    assert.equal(offer.wanted(), true);
    const second = host();
    offer.mount(second);  // the wizard re-renders the Summary: the tick is painted back
    assert.match(second.innerHTML, /data-consciousness-start checked/);
    assert.equal(createConsciousnessOffer({ ...DEFAULTS, offered: false }).wanted(), false);
});

test('the Start is sent once, only after this server opened its chat queue, and "on" waits for the state', async () => {
    const api = gateway({ health: [boot({}), boot({}), boot({ dialogue_ready: true })], state: [bg(false), bg(null), bg(true)] });
    const phases = [];
    const outcome = await startConsciousnessAfterSave({ fetchImpl: api.fetchImpl, ...clock(), onPhase: (phase) => phases.push(phase) });
    assert.deepEqual(outcome, { outcome: 'started', reason: 'observed_on', sent: 'accepted', detail: '' });
    assert.deepEqual(phases, ['waiting', 'sending', 'confirming']);
    assert.deepEqual(api.calls.map((call) => call.url), ['/api/health', '/api/health', '/api/health', '/api/command',
        '/api/state', '/api/state', '/api/state']);
    assert.deepEqual(api.posts().map((call) => [call.method, JSON.parse(call.body)]),
        [['POST', { cmd: '/bg start', if_boot: 'S1', ui_control: { name: 'setup_consciousness' } }]]);
});

test('HTTP 200 is only queue acceptance: a Start the state never shows stays unconfirmed', async () => {
    const api = gateway({ health: [boot({ dialogue_ready: true })], state: [bg(false)] });
    const outcome = await startConsciousnessAfterSave({ fetchImpl: api.fetchImpl, ...clock(), confirmWaitMs: 5000 });
    assert.deepEqual(outcome, { outcome: 'unknown', reason: 'unconfirmed', sent: 'accepted', detail: '' });
    assert.equal(api.posts().length, 1);
});

test('a refused or unanswered Start is never retried', async () => {
    const refused = gateway({ health: [boot({ dialogue_ready: true })],
        command: { status: 409, body: { error: 'Complete provider setup before sending this command.', accepted: false } } });
    assert.deepEqual(await startConsciousnessAfterSave({ fetchImpl: refused.fetchImpl, ...clock() }), {
        outcome: 'not_started', reason: 'refused', sent: 'refused', detail: 'Complete provider setup before sending this command.' });
    assert.deepEqual(refused.calls.map((call) => call.url), ['/api/health', '/api/command']);
    const lost = gateway({ health: [boot({ dialogue_ready: true })], command: new Error('connection reset') });
    assert.deepEqual(await startConsciousnessAfterSave({ fetchImpl: lost.fetchImpl, ...clock() }), {
        outcome: 'unknown', reason: 'unanswered', sent: 'unanswered', detail: '' });
    assert.deepEqual(lost.calls.map((call) => call.url), ['/api/health', '/api/command']);
});

test('Panic, a restart, a failed boot or a slow one before the Start: nothing is sent', async () => {
    const cases = [
        [[boot({}), new Error('connection refused')], 'unreachable'],          // Panic exits the server
        [[boot({}), boot({ since: 'S2', dialogue_ready: true })], 'restarted'],   // another server answers
        [[boot({ phase: 'failed', error: 'Supervisor init failed: boom' })], 'startup_failed'],
        [[boot({})], 'not_ready'],
    ];
    for (const [health, reason] of cases) {
        const api = gateway({ health });
        const outcome = await startConsciousnessAfterSave({ fetchImpl: api.fetchImpl, ...clock(), readyWaitMs: 3000 });
        assert.equal(outcome.outcome, 'not_started', reason);
        assert.equal(outcome.reason, reason);
        assert.equal(outcome.sent, 'none');
        assert.equal(api.posts().length, 0, reason);
    }
});

test('Open Ouroboros now during the wait cancels what was not sent; after acceptance it stops only the reading', async () => {
    let stop = false;
    const waiting = gateway({ health: [boot({})] });
    const pending = startConsciousnessAfterSave({ fetchImpl: waiting.fetchImpl, cancelled: () => stop,
        wait: async () => { stop = true; } });
    assert.deepEqual(await pending, { outcome: 'cancelled', reason: 'cancelled', sent: 'none', detail: '' });
    assert.equal(waiting.posts().length, 0);
    let reads = 0;
    const confirming = gateway({ health: [boot({ dialogue_ready: true })], state: [bg(false)] });
    const outcome = await startConsciousnessAfterSave({ fetchImpl: async (url, init) => {
        if (url === '/api/state') reads += 1;
        return confirming.fetchImpl(url, init);
    }, cancelled: () => reads > 0, wait: async () => {} });
    assert.deepEqual(outcome, { outcome: 'cancelled', reason: 'cancelled', sent: 'accepted', detail: '' });
    assert.equal(confirming.posts().length, 1);
});

test('afterSave: no tick hands off at once and sends nothing', async () => {
    let started = 0;
    let announced = 0;
    const offer = createConsciousnessOffer(DEFAULTS, { start: async () => { started += 1; return {}; } });
    await offer.afterSave({ announce: () => { announced += 1; } });
    assert.deepEqual([started, announced, offer.running()], [0, 1, false]);
});

function tickedOffer(options) {
    const handlers = {};
    const offer = createConsciousnessOffer(DEFAULTS, { fetchImpl: gateway({ health: [boot({})] }).fetchImpl, ...options });
    offer.mount({ innerHTML: '', addEventListener: (type, fn) => { handlers[type] = fn; }, removeEventListener() {} });
    const box = { checked: true, closest: () => box };
    handlers.change({ target: box });
    return offer;
}
const clickOpen = (offer) => {
    let click = null;
    offer.bindScreen({ querySelector: () => ({ addEventListener: (_type, fn) => { click = fn; } }) });
    click();
};

test('afterSave: an observed "on" hands off; a Start that did not happen waits for the owner to continue', async () => {
    let announced = 0;
    const on = tickedOffer({ hold: 0, start: async ({ onPhase }) => { onPhase('confirming'); return { outcome: 'started', reason: 'observed_on', sent: 'accepted', detail: '' }; } });
    await on.afterSave({ announce: () => { announced += 1; } });
    assert.equal(announced, 1);

    const screens = [];
    const off = tickedOffer({ onChange: () => screens.push(off.screenHtml()),
        start: async () => ({ outcome: 'not_started', reason: 'unreachable', sent: 'none', detail: '' }) });
    const settled = off.afterSave({ announce: () => { announced += 1; } });
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(announced, 1, 'the result is shown before the hand-off');
    assert.equal(text(screens.at(-1)), 'Setup saved Your settings are on disk. Consciousness was not started '
        + 'Ouroboros stopped answering before the Start could be sent, so nothing was sent. '
        + 'Dashboard › Consciousness shows whether it is on; its Start is there. Open Ouroboros');
    clickOpen(off);
    await settled;
    assert.equal(announced, 2);
    await off.afterSave({ announce: () => { announced += 1; } });  // the same page never sends a second Start
    assert.equal(announced, 2, 'the same completion does not repeat the hand-off');
});

test('afterSave: a page left during the wait never hands off and never resumes', async () => {
    let disposed = false;
    let announced = 0;
    const offer = tickedOffer({ start: async ({ cancelled }) => { disposed = true; return cancelled() ? { outcome: 'cancelled', reason: 'cancelled', sent: 'none' } : {}; } });
    await offer.afterSave({ isDisposed: () => disposed, announce: () => { announced += 1; } });
    assert.equal(announced, 0);
});

test('Open Ouroboros now during the real wait: cancelled before sending, then the hand-off', async () => {
    const api = gateway({ health: [boot({})] });
    let announced = 0;
    const offer = tickedOffer({ start: (options) => startConsciousnessAfterSave({ ...options, fetchImpl: api.fetchImpl, pollMs: 50 }) });
    await offer.beforeSave();
    const settled = offer.afterSave({ announce: () => { announced += 1; } });
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.match(text(offer.screenHtml()), /Starting consciousness… Waiting for Ouroboros to finish starting\. Nothing is sent before it is ready\. Open Ouroboros now$/);
    clickOpen(offer);
    assert.match(offer.screenHtml(), /disabled>Opening…</);
    await settled;
    assert.equal(announced, 1);
    assert.equal(api.posts().length, 0);
});

test('the run screen keeps saved, accepted and on apart', () => {
    assert.match(text(runScreenHtml({ phase: 'confirming' })), /^Setup saved .* Starting consciousness… The Start was accepted\. Waiting for consciousness to show as on\./);
    assert.match(text(runScreenHtml({ phase: 'started' })), /Consciousness is on\. Open Ouroboros$/);
    assert.match(text(runScreenHtml({ phase: 'done', outcome: 'unknown', reason: 'unconfirmed' })),
        /Consciousness is not confirmed The Start was accepted, but consciousness has not shown as on yet\./);
    assert.match(runScreenHtml({ phase: 'done', outcome: 'not_started', reason: 'refused', detail: '<b>x</b>' }), /\(&lt;b&gt;x&lt;\/b&gt;\)/);
});

// Regression consumers found in the first candidate: no real transport or server is used here.
const nextTurn = () => new Promise((resolve) => setImmediate(resolve));

test('a restart between Finish and the first post-save health read never receives the old Start', async () => {
    const api = gateway({ health: [boot({ since: 'S2', dialogue_ready: true })], state: [bg(true)] });
    const outcome = await startConsciousnessAfterSave({ fetchImpl: api.fetchImpl, expectedSince: 'S1', ...clock() });
    assert.equal(api.posts().length, 0);
    assert.equal(outcome.reason, 'restarted');
});

for (const status of [400, 502]) {
    test(`HTTP ${status} without an explicit pre-enqueue refusal leaves the Start unknown`, async () => {
        const api = gateway({ health: [boot({ dialogue_ready: true })], state: [bg(true)],
            command: { status, body: { error: 'The reply failed; enqueue may have happened.' } } });
        const outcome = await startConsciousnessAfterSave({ fetchImpl: api.fetchImpl, expectedSince: 'S1', ...clock() });
        assert.equal(outcome.outcome, 'unknown');
        assert.equal(outcome.sent, 'unanswered');
        assert.equal(api.posts().length, 1);
    });
}

for (const stalled of ['health request', 'command body']) {
    test(`cancellation settles a stalled ${stalled}, even when transport ignores abort`, async () => {
        const controller = new AbortController();
        const signals = [];
        let commands = 0, settled = false;
        const outcome = startConsciousnessAfterSave({ expectedSince: 'S1', signal: controller.signal,
            fetchImpl: async (url, init = {}) => {
                signals.push(init.signal);
                if (url === '/api/health') {
                    if (stalled === 'health request') return new Promise(() => {});
                    return { ok: true, status: 200, json: async () => boot({ dialogue_ready: true }).body };
                }
                assert.equal(url, '/api/command'); commands += 1;
                return { ok: true, status: 200, json: () => new Promise(() => {}) };
            },
        }).then((value) => { settled = true; return value; });
        await nextTurn();
        controller.abort();
        await nextTurn();
        assert.equal(settled, true, 'aborting must end the caller, not just the next poll delay');
        assert.ok(signals.at(-1)?.aborted, 'the active transport received cancellation; earlier completed reads are retired');
        assert.equal(commands, stalled === 'command body' ? 1 : 0);
        const result = await outcome;
        assert.equal(result.outcome, 'cancelled');
        if (commands) assert.equal(result.sent, 'unanswered', 'an aborted POST is not rolled back');
    });
}

test('disposing the page releases the displayed unconfirmed result without a hand-off', async () => {
    let announced = 0, settled = false;
    const offer = tickedOffer({ start: async () => ({ outcome: 'unknown', reason: 'unconfirmed', sent: 'accepted' }) });
    const outcome = offer.afterSave({ announce: () => { announced += 1; } }).then(() => { settled = true; });
    await nextTurn();
    assert.equal(offer.snapshot().run.phase, 'done');
    assert.equal(typeof offer.dispose, 'function');
    offer.dispose();
    await nextTurn();
    assert.equal(settled, true);
    await outcome;
    assert.equal(announced, 0);
});

test('Finish captures its boot before save; a later confirmed receipt never adopts the new boot', async () => {
    const api = gateway({ health: [boot({}), boot({ since: 'S2', dialogue_ready: true })], state: [bg(true)] });
    const offer = tickedOffer({ fetchImpl: api.fetchImpl });
    await offer.beforeSave();
    let announced = 0;
    const result = offer.afterSave({ announce: () => { announced += 1; } });
    await nextTurn();
    assert.equal(offer.snapshot().run.reason, 'restarted');
    assert.equal(api.posts().length, 0);
    clickOpen(offer);
    await result;
    assert.equal(announced, 1);
});

test('without a readable Finish boot the save may complete but no Start is sent', async () => {
    const api = gateway({ health: [new Error('no response'), boot({ dialogue_ready: true })] });
    const offer = tickedOffer({ fetchImpl: api.fetchImpl });
    await offer.beforeSave();
    const result = offer.afterSave({ announce: () => {} });
    await nextTurn();
    assert.equal(offer.snapshot().run.reason, 'boot_unknown');
    assert.equal(api.posts().length, 0);
    clickOpen(offer);
    await result;
});

test('disposing during Finish cancels its identity read and makes no later hand-off', async () => {
    let transport;
    const offer = tickedOffer({ fetchImpl: async (_url, init) => {
        transport = init.signal; return new Promise(() => {});
    } });
    let prepared = false, announced = false;
    const preparing = offer.beforeSave().then(() => { prepared = true; });
    await nextTurn();
    offer.dispose();
    await nextTurn();
    assert.equal(prepared, true);
    assert.equal(transport.aborted, true);
    await preparing;
    await offer.afterSave({ announce: () => { announced = true; } });
    assert.equal(announced, false);
});

// Review repairs: the readiness window, and a page left after Finish (#1603).
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test('the readiness window is checked before another read: past it the Start is not ready, never unreachable', async () => {
    const t = clock();
    const late = [];
    const fetchImpl = async () => {
        if (t.now() >= 3000) { late.push(t.now()); return new Promise(() => {}); }  // slower than any lifetime left
        return { ok: true, status: 200, json: async () => boot({}).body };
    };
    assert.deepEqual(await startConsciousnessAfterSave({ fetchImpl, ...t, readyWaitMs: 3000 }),
        { outcome: 'not_started', reason: 'not_ready', sent: 'none', detail: '' });
    assert.deepEqual(late, [], 'no read starts once the window has ended');
});

test('a read the window cuts short ends the wait as not ready; a read that fails by itself stays unreachable', async () => {
    let reads = 0;
    const slow = async () => {
        reads += 1;
        return reads > 1 ? new Promise(() => {}) : { ok: true, status: 200, json: async () => boot({}).body };
    };
    const cut = await startConsciousnessAfterSave({ fetchImpl: slow, ...clock(), readyWaitMs: 30, pollMs: 20 });
    assert.deepEqual([cut.outcome, cut.reason, cut.sent, reads], ['not_started', 'not_ready', 'none', 2]);
    const refused = gateway({ health: [boot({}), new Error('connection refused')] });
    const failed = await startConsciousnessAfterSave({ fetchImpl: refused.fetchImpl, ...clock(), readyWaitMs: 30, pollMs: 20 });
    assert.deepEqual([failed.outcome, failed.reason, refused.posts().length], ['not_started', 'unreachable', 0]);
});

test('a page left during the wait never sends its Start, even shown again from the back/forward cache', async () => {
    let away = false;
    const calls = [];
    const fetchImpl = async (url) => {
        calls.push(url);
        if (url === '/api/health') return { ok: true, status: 200, json: async () => boot({ dialogue_ready: away }).body };
        return { ok: true, status: 200, json: async () => (url === '/api/state' ? bg(true).body : { status: 'ok' }) };
    };
    let announced = 0;
    const offer = tickedOffer({ fetchImpl, start: (options) => startFlow({ ...options, pollMs: 5 }) });
    await offer.beforeSave();
    const settled = offer.afterSave({ announce: () => { announced += 1; } });
    await pause(20);
    assert.equal(offer.snapshot().run.phase, 'waiting');
    offer.leave();  // pagehide with persisted: the page and its timers stay alive in the cache
    away = true;    // meanwhile the server becomes ready: a stale loop would send now
    await pause(30);
    assert.equal(calls.filter((url) => url === '/api/command').length, 0, 'no Start after leaving');
    assert.equal(text(offer.screenHtml()), 'Setup saved Your settings are on disk. Consciousness was not started '
        + 'You left this page before the Start could be sent, so nothing was sent. '
        + 'Dashboard › Consciousness shows whether it is on; its Start is there. Open Ouroboros');
    assert.equal(announced, 0, 'shown again, it waits for the owner instead of handing off by itself');
    clickOpen(offer);
    await settled;
    assert.equal(announced, 1);
    await offer.afterSave({ announce: () => { announced += 1; } });
    assert.equal(announced, 1, 'the same page never sends a second Start');
});

test('a page left while its Finish was saving sends no Start once that save is confirmed', async () => {
    const api = gateway({ health: [boot({ dialogue_ready: true })], state: [bg(true)] });
    let announced = 0;
    const offer = tickedOffer({ fetchImpl: api.fetchImpl });
    await offer.beforeSave();
    offer.leave();
    const settled = offer.afterSave({ announce: () => { announced += 1; } });
    await nextTurn();
    assert.equal(api.posts().length, 0);
    assert.deepEqual(offer.snapshot().run, { outcome: 'not_started', reason: 'left', sent: 'none', detail: '', phase: 'done' });
    clickOpen(offer);
    await settled;
    assert.equal(announced, 1);
});

test('leaving before Finish retires nothing: a Finish chosen after coming back is a new choice', async () => {
    const api = gateway({ health: [boot({ dialogue_ready: true })], state: [bg(true)] });
    let announced = 0;
    const offer = tickedOffer({ fetchImpl: api.fetchImpl, hold: 0 });
    offer.leave();
    await offer.beforeSave();
    await offer.afterSave({ announce: () => { announced += 1; } });
    assert.equal(api.posts().length, 1);
    assert.equal(announced, 1);
});

test('a page left after acceptance keeps what reached the server: accepted, unconfirmed, nothing resent', async () => {
    const api = gateway({ health: [boot({ dialogue_ready: true })], state: [bg(false)] });
    const offer = tickedOffer({ fetchImpl: api.fetchImpl, start: (options) => startFlow({ ...options, pollMs: 5 }) });
    await offer.beforeSave();
    const settled = offer.afterSave({ announce: () => {} });
    await pause(20);
    assert.equal(offer.snapshot().run.phase, 'confirming');
    offer.leave();
    await pause(10);
    const { outcome, reason, sent } = offer.snapshot().run;
    assert.deepEqual([outcome, reason, sent, api.posts().length], ['unknown', 'unconfirmed', 'accepted', 1]);
    clickOpen(offer);
    await settled;
});

test('the wizard retires the Start on every pagehide, before a back/forward-cached page is kept alive', async () => {
    const { readFile } = await import('node:fs/promises');
    const source = await readFile(new URL('../modules/onboarding_wizard.js', import.meta.url), 'utf8');
    const handler = source.slice(source.indexOf("window.addEventListener('pagehide'"));
    const leave = handler.indexOf('consciousness.leave();');
    assert.ok(leave > 0 && leave < handler.indexOf('if (event.persisted) return;'), 'leave() runs before the persisted return');
});
