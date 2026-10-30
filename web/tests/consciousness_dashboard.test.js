import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { setup, NodeStub } from './dashboard_dom_fixture.js';
import { clock, observe } from './state_read_fixture.js';
import * as api from '../modules/api_client.js';

import { normalizeConsciousness, normalizePlacesResponse, normalizeWake } from '../modules/consciousness_api.js';
import { createContributionsStore } from '../modules/contributions.js';
import {
    countedRootsText, dashboardPlaceRowHtml, factsLineParts, headlineView, initConsciousnessDashboard, liveWake,
    paintWakeRows, placesCardHtml, summaryHtml, wakeRowHtml, wakesCardHtml,
} from '../modules/consciousness_dashboard.js';
import { noteLeadText, wakeCheckText, wakeCostText } from '../modules/consciousness_words.js';
import { applyPayload, setMissTransport } from '../modules/i18n.js';

const FIXTURE = JSON.parse(fs.readFileSync(new URL('./fixtures/consciousness_contract.json', import.meta.url), 'utf8'));
const STATE_LIFETIME = api.STATE_READ_LIFETIME_MS ?? 60_000;  // a namespace read keeps base trees loadable
const full = () => normalizeConsciousness(FIXTURE.consciousness);
const fromState = () => normalizeConsciousness(FIXTURE.state.bg_consciousness_state, { enabled: true });
const wake = (id) => full().recentWakes.find((row) => row.taskId === id);
const strip = (html) => html.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();

test('Dashboard keeps two titled wake lists and their separate empty sentences', () => {
    for (const rows of [full().recentWakes, [], [wake('214a31b6')], [wake('w-focus-user')]]) {
        const html = wakesCardHtml(rows, full().coverage);
        assert.equal(html.split(full().coverage).length - 1, 1);
        assert.ok(html.indexOf(full().coverage) < html.indexOf('<section '));
        const groups = html.split('<section ').slice(1);
        assert.deepEqual(groups.map((group) => group.match(/<h4[^>]*>([^<]+)</)[1]), ['Its own wakes', 'Wakes in places']);
        for (const [n, cause, empty] of [[0, 'own', 'No own wakes with a recorded model call in the last 48 h.'],
            [1, 'focus', 'No wakes in places in the last 48 h.']]) {
            const expected = rows.filter((row) => row.cause === cause).map((row) => row.taskId);
            assert.deepEqual([...groups[n].matchAll(/data-wake="([^"]+)"/g)].map((match) => match[1]), expected);
            assert.equal(groups[n].includes(empty), expected.length === 0);
        }
    }
});

test('Dashboard keeps the host headline as its only clock', (t) => {
    t.mock.method(Date, 'now', () => Date.parse('2027-01-15T12:00:00Z'));
    for (const [status, detail, next_wake_at] of [
        ['due', 'Due since 11:00; the wake-up has not started yet.', '2027-01-15T11:30:00Z'],
        ['sleeping', 'Sleeping until 13:00.', '2027-01-15T13:00:00Z'],
        ['wake_paused', 'Wake-up own is paused or waiting in the queue; its card shows why. The next wake-up follows it.', '2027-01-15T11:30:00Z'],
    ]) {
        const snapshot = normalizeConsciousness({ enabled: true, status, detail, next_wake_at,
            appointment_at: '2027-01-15T11:00:00Z' });
        const text = strip(summaryHtml(snapshot));
        assert.equal(text.split(detail).length - 1, 1);
        assert.doesNotMatch(text.replace(detail, ''), /\d\d:\d\d|[Dd]ue|since|[Nn]ext wake/);
        assert.deepEqual(headlineView(snapshot), { text: detail, moving: false });
    }
});

test('focus account stamps cross normalization and preserve the wake anatomy and authored note lead', () => {
    const raw = { ...FIXTURE.consciousness.recent_wakes.find((row) => row.task_id === '214a31b6'),
        cause: 'focus', place_label: 'Mailbox', chat_id: 1099511627777 };
    const before = wakeRowHtml(normalizeWake(raw));
    for (const owner of [false, true]) {
        const html = wakeRowHtml(normalizeWake({ ...raw, as: { label: '<Mail> & account', owner } }));
        const stamp = owner ? 'as you' : 'as <span data-i18n-authored>&lt;Mail&gt; &amp; account</span>';
        assert.ok(html.includes(`Mailbox</span> · ${stamp} · Done · no message`));
        assert.equal(html.replace(` · ${stamp}`, ''), before, 'time, phase, duration, cost, note lead and disclosure are unchanged');
        if (owner) assert.ok(!html.includes('&lt;Mail&gt;'));
        else assert.ok(html.includes('data-i18n-authored>&lt;Mail&gt; &amp; account'));
        assert.ok(html.includes('class="consciousness-wake-lead" data-i18n-authored'));
        assert.ok(html.includes(raw.note));
    }
    assert.equal(wakeRowHtml(normalizeWake({ ...raw, as: null })), before);
    assert.doesNotMatch(before, /as you|as <span/);
});

test('wake account words use translations without translating the authored label', (t) => {
    setMissTransport(() => Promise.resolve(null));
    applyPayload({ language: 'ru', english: false, entries: {
        'as {label}': { text: 'как {label}' }, 'code:places.identity.as_you': { text: 'как вы' },
    } });
    t.after(() => applyPayload({ language: '', english: true, entries: {} }));
    const row = wake('214a31b6');
    assert.match(wakeRowHtml({ ...row, as: { label: 'Settings', owner: false } }), /как <span data-i18n-authored>Settings<\/span>/);
    assert.match(wakeRowHtml({ ...row, as: { label: 'Settings', owner: true } }), /как вы/);
    assert.doesNotMatch(wakeRowHtml(row), /как вы|как <span/);
});

test('server status alone owns the headline even with finished or active rows in the list', () => {
    assert.equal(headlineView(full()).text, FIXTURE.consciousness.detail);
    assert.match(headlineView(full()).text, /^Sleeping until/);
    assert.equal(headlineView(full()).moving, false);
    assert.ok(liveWake(full()).live);
    assert.equal(liveWake({ recentWakes: [wake('214a31b6')] }), null);
    for (const [status, detail] of [['due', 'Due; the wake-up has not started yet.'],
        ['wake_failed', 'The last wake-up failed.'], ['wake_rejected', 'The last wake-up was refused.'],
        ['wake_paused', 'The last wake-up returned while paused.'], ['disabled', 'Background consciousness is off.'],
        ['waiting_for_first_conversation', 'No owner chat is bound yet; the first conversation binds it.']]) {
        assert.deepEqual(headlineView({ ...full(), status, detail }), { text: detail, moving: false });
    }
    assert.equal(headlineView({ ...full(), status: 'thinking' }).moving, true);
});

test('facts keep money with its cause and say when some calls have no price', () => {
    assert.deepEqual(factsLineParts(fromState(), 3).map(strip),
        ['on its own clock', '3 places', 'own $0.62', 'Slack · Acme $0.88', 'started work $0.48', 'Some calls have no price']);
    assert.deepEqual(factsLineParts({ ...fromState(), enabled: false, spendByCause: [] }, 1).map(strip), ['off', '1 place']);
    assert.equal(countedRootsText(fromState()), 'Its own tasks: 1 of 2 · Release notes draft (Slack · Acme)');
    assert.match(summaryHtml(fromState()), /data-cx-act="toggle" data-enabled="1">Stop</);
    assert.doesNotMatch(summaryHtml(normalizeConsciousness({})), /data-cx-act="toggle"/);
});

test('producer rows retain terminal truth and all five active phases', () => {
    assert.match(strip(wakeRowHtml(wake('214a31b6'))), /Done · no message/);
    assert.doesNotMatch(wakeRowHtml(wake('214a31b6')), /Working|is-live|data-shape="moving"/);
    assert.doesNotMatch(wakeRowHtml(wake('w-spoke')), /no message/);
    for (const [id, word] of [['running', 'Working'], ['waiting', 'Waiting'], ['queued', 'Queued'], ['paused', 'Paused'], ['unknown', 'State unknown']]) {
        const html = wakeRowHtml(wake('w-' + id));
        assert.ok(strip(html).includes(word));
        assert.equal(html.includes('data-shape="moving"'), id === 'running');
        if (id !== 'running') assert.doesNotMatch(html, /Working/);
    }
    assert.doesNotMatch(wakeRowHtml({ ...wake('214a31b6'), endedAt: '' }), /Working/);
});

test('summary is authored, note is whole and disclosed, only flagged checks get chips', () => {
    const old = wake('214a31b6'), named = wake('w-summary');
    const oldHtml = wakeRowHtml(old), namedHtml = wakeRowHtml(named);
    assert.doesNotMatch(oldHtml, /class="consciousness-wake-summary"/);
    const collapsed = oldHtml.split('<details')[0];
    assert.match(collapsed, /class="consciousness-wake-lead" data-i18n-authored>В 07:46 UTC сделал короткую сверку/);
    assert.doesNotMatch(collapsed, /9 rounds/);
    assert.ok(oldHtml.slice(oldHtml.indexOf('<details')).includes(old.note));
    assert.match(wakeRowHtml({ ...old, note: '' }).split('<details')[0], /9 rounds/);
    assert.ok(namedHtml.includes(`class="consciousness-wake-summary" data-i18n-authored>${named.summary}</p>`));
    assert.ok(namedHtml.indexOf(named.summary) < namedHtml.indexOf('<details'));
    assert.ok(namedHtml.includes(named.note));
    assert.match(namedHtml, /Its own note/);
    assert.doesNotMatch(oldHtml, /ui-chip/);
    assert.equal((namedHtml.match(/ui-chip/g) || []).length, 3);
    assert.match(oldHtml, /Safety Supervisor checked run_command/);
    assert.match(namedHtml, /Not checked \(update_memory\)/);
    assert.doesNotMatch(namedHtml, /Safety Supervisor checked update_memory|unmetered|omitted/);
    assert.match(namedHtml, /data-open-task="w-summary">Open in Main ↗/);
    assert.doesNotMatch(wakeRowHtml({ ...named, cause: 'focus', chatId: 1099511627777, placeLabel: 'Slack' }), /data-open-task/);
    const focusRow = wakeRowHtml({ ...named, cause: 'focus', chatId: 1099511627777, placeLabel: 'Slack' });
    assert.match(focusRow, /data-cx-open-room="1099511627777"/, 'a focus row opens its captured conversation');
    assert.match(focusRow, /Open the conversation/);
    assert.doesNotMatch(wakeRowHtml({ ...named, cause: 'focus', chatId: 0, placeLabel: 'Slack' }), /data-cx-open-room/, 'no room, no door');
    assert.match(wakesCardHtml(full().recentWakes, full().coverage), /wakes that reached a model call in the last 48 h/);
});

test('unknown, zero, known, open and mixed costs stay distinct in visible and accessible words', () => {
    assert.equal(wakeCostText(wake('214a31b6')), 'spend unknown');
    assert.equal(wakeCostText(wake('w-running')), 'up to $0.02 so far');
    const priced = (amount, extra = {}) => normalizeWake({ cost_presentation: {
        has_rows: true, tracked_amount: amount, tracked_final: true, has_unpriced: false, accounting_open: false, ...extra } });
    assert.equal(wakeCostText(priced(0)), '$0.00');
    assert.equal(wakeCostText(priced(1.23)), '$1.23');
    assert.equal(wakeCostText(priced(null)), 'spend unknown');
    assert.equal(wakeCostText(priced(1.23, { tracked_final: false, accounting_open: true })), 'up to $1.23 so far');
    assert.equal(wakeCostText(priced(1.23, { has_unpriced: true })), 'Tracked: $1.23');
    const { places } = normalizePlacesResponse(FIXTURE.places);
    const slack = dashboardPlaceRowHtml(places[0], { awake: true });
    assert.match(strip(slack), /Slack · Acme active · wakes for direct messages, mentions and replies · 2 open turns · Connector on \$0\.88/);
    assert.match(slack, /class="status-mark consciousness-mark"/);
    assert.match(slack, /aria-label="Open Slack · Acme, awake here now: /);
    assert.match(slack, /aria-label="[^"]*Connector on/);
    const off = dashboardPlaceRowHtml({ ...places[0], readiness: { text: 'Off in Skills', tone: 'warn' } });
    assert.match(off, /aria-label="[^"]*Off in Skills/);
    assert.doesNotMatch(off, /Connector on/);
    assert.match(dashboardPlaceRowHtml(places[1]), /class="consciousness-spend is-none">unknown</);
    assert.match(dashboardPlaceRowHtml(places[1]), /spend unknown/);
    assert.doesNotMatch(dashboardPlaceRowHtml(places[1]), /nothing spent|\$0\.00 spent/);
    assert.match(strip(dashboardPlaceRowHtml(places[1])), /^Telegram · Ouroboros community silent · wakes for mentions unknown$/, 'Silent still wakes for its list');
    assert.match(strip(dashboardPlaceRowHtml(places[2])), /^Mail · inbox paused \$0\.00$/);
    assert.match(dashboardPlaceRowHtml(places[2]), />\$0\.00</);
    assert.match(dashboardPlaceRowHtml(places[2]), /\$0\.00 spent/);
    assert.doesNotMatch(dashboardPlaceRowHtml(places[2]), /spend unknown/);
    assert.match(placesCardHtml(places), /data-place-connect>Connect a place</, "the one Connect control");
    assert.match(placesCardHtml([]), /No place yet\. Connect one from an installed connector\./);
    assert.match(placesCardHtml([]), /A connector you install can connect a place here: Connect a place\./);
    assert.doesNotMatch(placesCardHtml(places), /Each connector you install adds/);
    assert.match(placesCardHtml(null, { error: 'Places could not be read: HTTP 404' }), /data-tone="warn"/);
});

test('the Dashboard always reads the Places projection, even with an empty snapshot list or a failed snapshot', async (t) => {
    const { window } = setup(t);
    const mount = fakeMount();
    let reads = 0, offline = false, placesOffline = false;
    const controller = initConsciousnessDashboard({ mount, state: { activePage: 'chat' },
        read: async () => { if (offline) throw new Error('snapshot offline'); return { ...full(), places: [] }; },
        readFallback: async () => { throw new Error('state offline'); },
        readPlacesFn: async () => {
            ++reads;
            if (placesOffline) throw new Error('places offline');
            return normalizePlacesResponse(FIXTURE.places);
        },
    });
    const settled = () => new Promise((resolve) => setImmediate(resolve));
    const placesHtml = () => mount.regions.get('[data-cx-places]').innerHTML;
    await settled();
    assert.equal(reads, 1, 'mount reads Places even when the Dashboard is hidden');
    await controller.refresh();
    assert.match(placesHtml(), /Slack · Acme/);
    assert.match(placesHtml(), /Connector on/);
    offline = true;
    await controller.refresh();
    assert.match(placesHtml(), /Slack · Acme/);
    placesOffline = true;
    window.dispatchEvent(new CustomEvent('ouro:places-changed')); await settled();
    assert.match(placesHtml(), /Places could not be read: places offline/);
    assert.doesNotMatch(placesHtml(), /No place yet/);
    placesOffline = false;
    window.dispatchEvent(new CustomEvent('ouro:places-changed')); await settled();
    assert.match(placesHtml(), /Slack · Acme/);
    assert.doesNotMatch(placesHtml(), /places offline/);
    const before = reads;
    controller.destroy();
    window.dispatchEvent(new CustomEvent('ouro:places-changed')); await settled();
    assert.equal(reads, before);
});

test('showing the tab reads Places again, so an updated skill\'s place icon is the one Settings shows', async (t) => {
    const { window } = setup(t);
    const mount = fakeMount();
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'consciousness' };
    const mark = (path) => ({ kind: 'svg', svg: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="${path}" /></svg>` });
    const response = normalizePlacesResponse(FIXTURE.places);
    // The installed connector's mark before and after a skill update on the server; no places-changed event.
    let icon = mark('M9 3v12M15 9v12M3 9h12M9 15h12');
    let reads = 0;
    const controller = initConsciousnessDashboard({ mount, state, read: async () => full(),
        readPlacesFn: async () => { ++reads; return { ...response, places: response.places.map((place, index) => (index ? place : { ...place, icon })) }; } });
    const settled = () => new Promise((resolve) => setImmediate(resolve));
    const placesHtml = () => mount.regions.get('[data-cx-places]').innerHTML;
    await settled();
    assert.match(placesHtml(), /M9 3v12/);
    icon = mark('M19 8.5V10h1.5A1.5 1.5 0 1 0 19 8.5');
    await controller.refresh();
    assert.equal(reads, 1, 'the poll leaves Places to their own reads');
    window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'consciousness' } })); await settled();
    assert.equal(reads, 2);
    assert.match(placesHtml(), /M19 8\.5V10h1\.5/);
    assert.doesNotMatch(placesHtml(), /M9 3v12/);
    icon = mark('M4 4h16');
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'dashboard' } })); await settled();
    assert.equal(reads, 3, 'returning to the Dashboard on this tab reads them too');
    assert.match(placesHtml(), /M4 4h16/);
    state.dashboardActiveSubtab = 'logs';
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'dashboard' } })); await settled();
    assert.equal(reads, 3, 'another tab of the Dashboard reads nothing here');
    controller.destroy();
});

test('a skill lifecycle end reads Places at once while shown and at the next show while hidden; the newest read paints', async (t) => {
    const { window, document } = setup(t);
    const mount = fakeMount();
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'consciousness' };
    // The page's own contributions store on a stub socket: its lifecycle triggers are the ones the panel hears.
    const socket = new Map();
    const ws = { on(type, fn) { socket.set(type, fn); return () => socket.delete(type); }, send() {} };
    const store = createContributionsStore({ client: { uiContributions: async () => ({}) } });
    const detach = store.attach({ ws, target: window });
    const mark = (path) => ({ kind: 'svg', svg: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="${path}" /></svg>` });
    const response = normalizePlacesResponse(FIXTURE.places);
    // The first place's mark on the server now: an off connector, which the contributions projection leaves out.
    let served = mark('M9 3v12M15 9v12M3 9h12M9 15h12');
    const held = [];  // each Places read answers what the server held when it started, when the test says
    const controller = initConsciousnessDashboard({ mount, state, ws, contributions: store, read: async () => full(),
        readPlacesFn: () => {
            const answer = { ...response, places: response.places.map((place, index) => (index ? place : { ...place, icon: served })) };
            return new Promise((resolve) => held.push(() => resolve(answer)));
        } });
    const settled = () => new Promise((resolve) => setImmediate(resolve));
    const placesHtml = () => mount.regions.get('[data-cx-places]').innerHTML;
    const answerAll = async () => { held.splice(0).forEach((answer) => answer()); await settled(); };
    const ended = (phase = 'completed') => socket.get('chat')({ type: 'chat', role: 'system', is_progress: true,
        system_type: 'skill_lifecycle', lifecycle: { kind: 'update', target: 'slack-bridge', phase } });
    await settled();
    await answerAll();
    assert.match(placesHtml(), /M9 3v12/);

    await controller.refresh();
    socket.get('open')();
    ended('running');
    await settled();
    assert.equal(held.length, 0, 'the poll, a socket open and a running job read no Places');

    served = mark('M19 8.5V10h1.5A1.5 1.5 0 1 0 19 8.5');
    ended();
    assert.equal(held.length, 1, 'shown: the end of the update reads Places at once');
    await answerAll();
    assert.match(placesHtml(), /M19 8\.5V10h1\.5/);
    assert.doesNotMatch(placesHtml(), /M9 3v12/);

    // A show's read was already in flight when the next update ended: its older answer arrives last and loses.
    window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'consciousness' } }));
    served = mark('M2 2h20');
    ended();
    assert.equal(held.length, 2);
    const [older, newer] = held.splice(0);
    newer(); await settled();
    older(); await settled();
    assert.match(placesHtml(), /M2 2h20/);
    assert.doesNotMatch(placesHtml(), /M19 8\.5/);

    state.activePage = 'chat';
    served = mark('M3 3h18');
    ended();
    socket.get('extension_lifecycle')({ type: 'extension_lifecycle', skill: 'slack-bridge', action: 'reload' });
    document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(held.length, 0, 'hidden on another page: nothing is read yet');
    state.activePage = 'dashboard';
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'dashboard' } }));
    assert.equal(held.length, 1, 'its next show reads once');
    await answerAll();
    assert.match(placesHtml(), /M3 3h18/);
    document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(held.length, 0, 'that show already caught up');

    document.hidden = true;
    served = mark('M4 4h16');
    window.dispatchEvent(new Event('ouro:skill-lifecycle'));
    document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(held.length, 0, 'a document out of view waits too');
    document.hidden = false;
    document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(held.length, 1, 'and reads once when it comes back');
    await answerAll();
    assert.match(placesHtml(), /M4 4h16/);
    document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(held.length, 0);

    controller.destroy();
    ended();
    window.dispatchEvent(new Event('ouro:skill-lifecycle'));
    document.dispatchEvent(new Event('visibilitychange'));
    assert.equal(held.length, 0, 'destroy unsubscribes');
    detach();
});

function fakeMount() {
    const regions = new Map();
    return {
        innerHTML: '',
        regions,
        querySelector(selector) {
            if (!regions.has(selector)) regions.set(selector, { innerHTML: '', textContent: '', dataset: {}, querySelector: () => null, querySelectorAll: () => [], closest: () => null });
            return regions.get(selector);
        },
        addEventListener() {}, removeEventListener() {},
    };
}

test('while the full snapshot is not served, /api/state and /api/places still give status and places', async () => {
    const priorWindow = globalThis.window;
    const priorDocument = globalThis.document;
    globalThis.window = { addEventListener() {}, removeEventListener() {}, dispatchEvent() {} };
    globalThis.document = { hidden: false, addEventListener() {}, removeEventListener() {} };
    const mount = fakeMount();
    const notServed = Object.assign(new Error('HTTP 404'), { status: 404 });
    const controller = initConsciousnessDashboard({
        mount, ws: { send() {} }, state: { activePage: 'dashboard', dashboardActiveSubtab: 'other' },
        read: async () => { throw notServed; },
        readFallback: async () => fromState(),
        readPlacesFn: async () => normalizePlacesResponse(FIXTURE.places),
    });
    try {
        await controller.refresh();
        assert.match(mount.regions.get('[data-cx-summary]').innerHTML, /Consciousness · on/);
        assert.equal((mount.regions.get('[data-cx-places]').innerHTML.match(/data-place-open=/g) || []).length, 3);
        assert.match(mount.regions.get('[data-cx-wakes]').innerHTML, /does not report its recent wakes yet/);
        assert.equal(mount.regions.get('[data-cx-status]').textContent, '');
    } finally {
        controller.destroy();
        globalThis.window = priorWindow;
        globalThis.document = priorDocument;
    }
});

for (const hang of ['fetch', 'body']) {
    test(`a /api/state fallback whose ${hang} never answers ends the Dashboard's read at its lifetime and says so`, async (t) => {
        const advance = clock(t);
        const priorWindow = globalThis.window;
        const priorDocument = globalThis.document;
        const priorFetch = globalThis.fetch;
        const signals = [];
        globalThis.window = { addEventListener() {}, removeEventListener() {}, dispatchEvent() {} };
        globalThis.document = { hidden: false, addEventListener() {}, removeEventListener() {} };
        globalThis.fetch = async (_url, init = {}) => {  // the real fallback reads /api/state; it never answers here
            signals.push(init.signal);
            return hang === 'fetch' ? new Promise(() => {}) : { ok: true, status: 200, json: () => new Promise(() => {}) };
        };
        const mount = fakeMount();
        const controller = initConsciousnessDashboard({
            mount, ws: { send() {} }, state: { activePage: 'dashboard', dashboardActiveSubtab: 'other' },
            read: async () => { throw Object.assign(new Error('HTTP 404'), { status: 404 }); },
            readPlacesFn: async () => normalizePlacesResponse(FIXTURE.places),
        });
        try {
            const refreshed = observe(controller.refresh());
            await advance(STATE_LIFETIME - 1000);
            assert.equal(refreshed.settled, false);
            assert.equal(mount.regions.get('[data-cx-status]').textContent, 'Reading…');
            await advance(1000);
            assert.equal(refreshed.settled, true, 'the expired fallback ends the read');
            assert.match(mount.regions.get('[data-cx-status]').textContent, /^Could not read consciousness: state read expired\./);
            assert.equal(signals[0].aborted, true);
        } finally {
            controller.destroy();
            Object.assign(globalThis, { window: priorWindow, document: priorDocument, fetch: priorFetch });
        }
    });
}

test('a failed read keeps the earlier values and says so', async () => {
    const priorWindow = globalThis.window;
    const priorDocument = globalThis.document;
    globalThis.window = { addEventListener() {}, removeEventListener() {}, dispatchEvent() {} };
    globalThis.document = { hidden: false, addEventListener() {}, removeEventListener() {} };
    const mount = fakeMount();
    let fail = false;
    const controller = initConsciousnessDashboard({
        mount, state: { activePage: 'chat' },
        read: async () => { if (fail) throw new Error('offline'); return full(); },
        readFallback: async () => { throw new Error('offline'); },
        readPlacesFn: async () => normalizePlacesResponse(FIXTURE.places),
    });
    try {
        await controller.refresh();
        assert.match(mount.regions.get('[data-cx-summary]').innerHTML, /Sleeping until/);
        fail = true;
        await controller.refresh();
        assert.match(mount.regions.get('[data-cx-status]').textContent, /Could not read consciousness: offline\. Earlier values are shown\./);
        assert.match(mount.regions.get('[data-cx-summary]').innerHTML, /Sleeping until/);
        assert.match(mount.regions.get('[data-cx-summary]').innerHTML, /data-enabled="1" disabled>Stop</, 'no socket, no toggle press');
    } finally {
        controller.destroy();
        globalThis.window = priorWindow;
        globalThis.document = priorDocument;
    }
});

test('the relay note lead keeps every word and only trims its edges', () => {
    assert.equal(noteLeadText('  First words\n\nThen the whole note.  '), 'First words\n\nThen the whole note.');
    assert.equal(noteLeadText(null), '');
    assert.equal(noteLeadText(wake('214a31b6').note), wake('214a31b6').note);
    const relay = fs.readFileSync(new URL('../modules/chat_relay_deck.js', import.meta.url), 'utf8');
    assert.match(relay, /words: noteLeadText\(/, 'the relay and Dashboard share the lead implementation');
});

test('unavailable wakes keep the authored reason and painted rows, unlike an absent key or empty window', (t) => {
    const { document } = setup(t);
    document.documentElement = new NodeStub('html');
    document.createTreeWalker = () => ({ nextNode: () => null });
    setMissTransport(() => Promise.resolve(null));
    const reason = 'Recent wakes could not be read: storage is busy.';
    const snapshot = normalizeConsciousness({ recent_wakes: null, recent_wakes_coverage: reason });
    assert.equal(snapshot.recentWakes, null);
    assert.equal(snapshot.coverage, reason);
    const legacy = normalizeConsciousness({ recent_wakes_coverage: reason });
    assert.equal(legacy.recentWakes, null);
    assert.equal(legacy.coverage, '');
    applyPayload({ language: 'ru', english: false, revision: 1, entries: {
        ['code:' + reason]: { text: 'Must not translate authored text' },
        'code:consciousness.no_own_wakes': { text: 'За последние 48 ч пробуждений нет.' },
    } });
    t.after(() => applyPayload({ language: '', english: true, revision: 0, entries: {} }));
    const unavailable = wakesCardHtml(snapshot.recentWakes, snapshot.coverage);
    assert.match(unavailable, /role="status" data-tone="warn" data-i18n-authored/);
    assert.equal(unavailable.split(reason).length - 1, 1);
    assert.equal((unavailable.match(/role="status"/g) || []).length, 1);
    assert.ok(unavailable.indexOf(reason) < unavailable.indexOf('<section '));
    assert.doesNotMatch(unavailable, /Must not translate|does not report|No wakes/);
    assert.match(wakesCardHtml(legacy.recentWakes, legacy.coverage), /does not report its recent wakes yet/);
    assert.match(wakesCardHtml([]), /За последние 48 ч пробуждений нет/);
    const region = new NodeStub();
    region.innerHTML = wakesCardHtml([wake('214a31b6')], full().coverage, 'own');
    const row = region.querySelector('[data-wake]');
    const put = (node, html) => { node.innerHTML = html; };
    paintWakeRows(region, null, reason, new Set(), put);
    assert.equal(region.querySelector('[data-wake]'), row, 'closed rows survive a failed read too');
    assert.equal(region.querySelector('[role="status"]'), null, 'the card owns the warning, not either list');
    paintWakeRows(region, [], full().coverage, new Set(), put);
    assert.equal(region.querySelector('[data-wake]'), null);
    assert.equal(region.querySelector('[role="status"]'), null, 'success clears the warning');
    applyPayload({ language: '', english: true, revision: 0, entries: {} });
    assert.match(wakesCardHtml([]), /No own wakes with a recorded model call in the last 48 h\./);
    assert.doesNotMatch(wakesCardHtml([]), /No wakes yet/);
});

for (const rich of [true, false]) test(`opened ${rich ? 'fenced code' : 'plain'} notes mount once and release on close and removal`, async (t) => {
    const cleanups = [];
    const { mount } = setup({ after: cleanup => cleanups.push(cleanup) });
    let controller;
    t.after(() => { controller?.destroy(); cleanups.forEach(cleanup => cleanup()); });
    // Only the HTML template API missing from this bookkeeping fixture; no layout is simulated.
    Object.defineProperty(NodeStub.prototype, 'content', { configurable: true, get() { return this; } });
    t.after(() => { delete NodeStub.prototype.content; });
    NodeStub.prototype.getBoundingClientRect = () => ({ top: 0 });
    t.after(() => { delete NodeStub.prototype.getBoundingClientRect; });
    let next = { ...full(), recentWakes: [{ ...wake('214a31b6'), note: rich ? '~~~js\nanswer();\n~~~' : 'The whole plain note.' }] };
    const mounted = [], enhanced = [], destroyed = [];
    const markdown = {
        mount: (node, text) => { mounted.push([node, text]); node.innerHTML = rich ? '<div class="md-code-block"><pre>answer();</pre></div>' : `<p>${text}</p>`; },
        enhance: node => enhanced.push(node), destroy: node => destroyed.push(node),
    };
    controller = initConsciousnessDashboard({ mount, state: { activePage: 'chat' }, markdown,
        read: async () => next, readPlacesFn: async () => normalizePlacesResponse(FIXTURE.places) });
    await controller.refresh();
    assert.equal(mounted.length, 0, 'closed notes do not acquire Markdown resources');
    const row = mount.querySelector('[data-wake]'), details = row.querySelector('details'), note = row.querySelector('.consciousness-wake-note');
    details.open = true;
    await mount.fire('toggle', { target: details });
    assert.equal(mounted[0][0], note, 'mount into the actual note element');
    assert.equal(mounted[0][1], next.recentWakes[0].note);
    assert.equal(enhanced.length, rich ? 1 : 0);
    await controller.refresh();
    assert.equal(mount.querySelector('[data-wake]'), row);
    assert.equal(row.querySelector('details').open, true);
    assert.equal(row.querySelector('.consciousness-wake-note'), note);
    assert.equal(mounted.length, 1);
    details.open = false;
    await mount.fire('toggle', { target: details });
    assert.equal(destroyed[0], note);
    const reopened = row.querySelector('details');
    reopened.open = true;
    await mount.fire('toggle', { target: reopened });
    assert.equal(mounted.length, 2);
    // A queued close can race a refresh removing its row; cleanup belongs to removal too.
    reopened.open = false;
    next = { ...next, recentWakes: [] };
    await controller.refresh();
    await mount.fire('toggle', { target: reopened });
    assert.equal(destroyed.length, 2);
    assert.equal(mount.querySelector('[data-wake]'), null);
    next = { ...next, recentWakes: [wake('214a31b6')] };
    await controller.refresh();
    const last = mount.querySelector('details');
    last.open = true;
    await mount.fire('toggle', { target: last });
    controller.destroy(); controller = null;
    assert.equal(destroyed.length, 3, 'teardown releases an open note too');
});


test('the committed wake document uses only the neutral home path', () => {
    const document = fs.readFileSync(new URL('../../tests/fixtures/wake_document_214a31b6.json', import.meta.url), 'utf8');
    assert.match(document, /\/Users\/owner\//);
    assert.deepEqual([...new Set([...document.matchAll(/\/Users\/([A-Za-z0-9_-]+)/g)].map(match => match[1]))], ['owner']);
});

test('a focus row door opens the place sheet on that conversation; an unknown room says so', async (t) => {
    const cleanups = [];
    const { mount } = setup({ after: cleanup => cleanups.push(cleanup) });
    let controller;
    t.after(() => { controller?.destroy(); cleanups.forEach(cleanup => cleanup()); });
    Object.defineProperty(NodeStub.prototype, 'content', { configurable: true, get() { return this; } });
    t.after(() => { delete NodeStub.prototype.content; });
    NodeStub.prototype.getBoundingClientRect = () => ({ top: 0 });
    t.after(() => { delete NodeStub.prototype.getBoundingClientRect; });
    const slack = { ...normalizePlacesResponse(FIXTURE.places).places[0], rooms: [{ chatId: 1099511627777 }] };
    const snapshot = { ...full(), recentWakes: [
        { ...wake('214a31b6'), taskId: 'focus-known', cause: 'focus', chatId: 1099511627777, placeLabel: 'Slack' },
        { ...wake('214a31b6'), taskId: 'focus-lost', cause: 'focus', chatId: 1099511627778, placeLabel: 'Gone' }] };
    const opened = [];
    let reads = 0;
    controller = initConsciousnessDashboard({ mount, state: { activePage: 'chat' },
        read: async () => snapshot,
        // The first read predates the second room's capture; the door's own re-read finds it.
        readPlacesFn: async () => ({ places: [++reads === 1 ? slack : { ...slack, rooms: [...slack.rooms, { chatId: 1099511627778 }] }], connectors: [], migrationProblems: [] }),
        openSheet: (place, options) => opened.push([place.id, options.openRoomChatId]) });
    await controller.refresh();
    const [known, later] = mount.querySelectorAll('[data-cx-open-room]');
    await mount.fire('click', { target: known });
    assert.deepEqual(opened, [[slack.id, 1099511627777]], 'the sheet opens on the row\'s conversation');
    assert.equal(reads, 1, 'an indexed room needs no second read');
    await mount.fire('click', { target: later });
    assert.equal(reads, 2, 'an unknown room reads Places once more before saying anything');
    assert.deepEqual(opened[1], [slack.id, 1099511627778], 'a room captured after the mount-time read opens');
    assert.equal(later.disabled, false);
    const orphan = { ...wake('214a31b6'), taskId: 'focus-orphan', cause: 'focus', chatId: 1099511627779, placeLabel: 'Gone' };
    snapshot.recentWakes = [...snapshot.recentWakes, orphan];
    await controller.refresh();
    const lost = mount.querySelectorAll('[data-cx-open-room]')[2];
    await mount.fire('click', { target: lost });
    assert.equal(reads, 3);
    assert.equal(opened.length, 2, 'a room no place owns even after a fresh read opens nothing');
    assert.equal(lost.disabled, true);
    assert.match(lost.textContent, /Conversation unavailable/);
});

test('a served panel failure keeps the painted rows and says so; only a 404 reads the state as the panel', async (t) => {
    const cleanups = [];
    const { mount } = setup({ after: cleanup => cleanups.push(cleanup) });
    let controller;
    t.after(() => { controller?.destroy(); cleanups.forEach(cleanup => cleanup()); });
    Object.defineProperty(NodeStub.prototype, 'content', { configurable: true, get() { return this; } });
    t.after(() => { delete NodeStub.prototype.content; });
    NodeStub.prototype.getBoundingClientRect = () => ({ top: 0 });
    t.after(() => { delete NodeStub.prototype.getBoundingClientRect; });
    let failure = null;
    controller = initConsciousnessDashboard({ mount, state: { activePage: 'chat' },
        read: async () => { if (failure) throw failure; return full(); },
        readFallback: async () => fromState(),
        readPlacesFn: async () => normalizePlacesResponse(FIXTURE.places) });
    await controller.refresh();
    const rows = mount.querySelectorAll('[data-wake]').length;
    assert.ok(rows > 0, 'the fixture paints wake rows');
    failure = Object.assign(new Error('HTTP 500'), { status: 500 });
    await controller.refresh();
    assert.equal(mount.querySelectorAll('[data-wake]').length, rows, 'a served failure keeps the rows');
    assert.match(mount.querySelector('[data-cx-status]').textContent, /Could not read consciousness: HTTP 500/);
    assert.doesNotMatch(mount.querySelector('[data-cx-wakes]').innerHTML, /does not report its recent wakes yet/);
    failure = Object.assign(new Error('HTTP 404'), { status: 404 });
    await controller.refresh();
    assert.match(mount.querySelector('[data-cx-wakes]').innerHTML, /does not report its recent wakes yet/, 'a server without the panel');
    assert.equal(mount.querySelector('[data-cx-status]').textContent, '');
});

test('a second press during the door\'s Places re-read is ignored; a failed re-read leaves the door alone', async (t) => {
    const cleanups = [];
    const { mount } = setup({ after: cleanup => cleanups.push(cleanup) });
    let controller;
    t.after(() => { controller?.destroy(); cleanups.forEach(cleanup => cleanup()); });
    Object.defineProperty(NodeStub.prototype, 'content', { configurable: true, get() { return this; } });
    t.after(() => { delete NodeStub.prototype.content; });
    NodeStub.prototype.getBoundingClientRect = () => ({ top: 0 });
    t.after(() => { delete NodeStub.prototype.getBoundingClientRect; });
    const slack = { ...normalizePlacesResponse(FIXTURE.places).places[0], rooms: [{ chatId: 1099511627777 }] };
    const snapshot = { ...full(), recentWakes: [
        { ...wake('214a31b6'), taskId: 'focus-late', cause: 'focus', chatId: 1099511627778, placeLabel: 'Slack' },
        { ...wake('214a31b6'), taskId: 'focus-lost', cause: 'focus', chatId: 1099511627779, placeLabel: 'Slack' }] };
    const opened = [];
    let reads = 0, settle = null, fail = false;
    controller = initConsciousnessDashboard({ mount, state: { activePage: 'chat' },
        read: async () => snapshot,
        readPlacesFn: () => {
            reads += 1;
            if (reads === 1) return Promise.resolve({ places: [slack], connectors: [], migrationProblems: [] });
            if (fail) return Promise.reject(new Error('HTTP 503'));
            return new Promise((resolve) => { settle = resolve; });
        },
        openSheet: (place, options) => opened.push([place.id, options.openRoomChatId]) });
    await controller.refresh();
    const [late, lost] = mount.querySelectorAll('[data-cx-open-room]');
    const first = mount.fire('click', { target: late });
    await new Promise((resolve) => setImmediate(resolve));
    await mount.fire('click', { target: late });
    assert.equal(reads, 2, 'the second press starts no third read');
    settle({ places: [{ ...slack, rooms: [...slack.rooms, { chatId: 1099511627778 }] }], connectors: [], migrationProblems: [] });
    await first;
    assert.deepEqual(opened, [[slack.id, 1099511627778]], 'one sheet for two presses');
    assert.equal(late.disabled, false);
    fail = true;
    await mount.fire('click', { target: lost });
    assert.equal(opened.length, 1);
    assert.equal(lost.disabled, false, 'a failed re-read is not absence: the door stays');
    assert.match(mount.querySelector('[data-cx-places]').innerHTML, /HTTP 503/, 'the Places card names the failed read');
});

test('a first read that fails names the error without claiming earlier values', async (t) => {
    const cleanups = [];
    const { mount } = setup({ after: cleanup => cleanups.push(cleanup) });
    let controller;
    t.after(() => { controller?.destroy(); cleanups.forEach(cleanup => cleanup()); });
    controller = initConsciousnessDashboard({ mount, state: { activePage: 'chat' },
        read: async () => { throw Object.assign(new Error('HTTP 500'), { status: 500 }); },
        readFallback: async () => fromState(),
        readPlacesFn: async () => normalizePlacesResponse(FIXTURE.places) });
    await controller.refresh();
    const status = mount.querySelector('[data-cx-status]').textContent;
    assert.match(status, /Could not read consciousness: HTTP 500/);
    assert.doesNotMatch(status, /Earlier values are shown/);
    assert.match(mount.querySelector('[data-cx-wakes]').innerHTML, /Recent wakes could not be read: HTTP 500/);
});

test('Safety chips name what they checked in sentence case, keeping raw tool ids as they are', () => {
    assert.equal(wakeCheckText({ chip: 'warned', tool: 'place_reply' }), 'Safety Supervisor warned about the reply');
    assert.equal(wakeCheckText({ chip: 'sent_despite', tool: 'place_reply' }), 'The reply sent despite the check');
    assert.equal(wakeCheckText({ chip: 'held', tool: 'send_user_message' }), 'Safety Supervisor held the message');
    assert.equal(wakeCheckText({ chip: 'sent_despite', tool: 'slack_send' }), 'slack_send sent despite the check');
    assert.equal(wakeCheckText({ chip: 'not_checked', tool: 'update_memory' }), 'Not checked (update_memory)');
    assert.equal(wakeCheckText({ chip: 'not_checked', tool: '' }), 'Not checked');
    assert.equal(wakeCheckText({ chip: 'unknown', tool: 'x' }), '');
});
