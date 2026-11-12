import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { setup, NodeStub } from './dashboard_dom_fixture.js';
import { clock, observe } from './state_read_fixture.js';
import * as api from '../modules/api_client.js';

import { normalizeConsciousness, normalizePlacesResponse, normalizeWake } from '../modules/consciousness_api.js';
import { createContributionsStore } from '../modules/contributions.js';
import {
    countedRootsText, headlineView, initConsciousnessDashboard, liveWake,
    paintWakeRows, spendDetails, summaryHtml, wakeRowHtml, wakesCardHtml,
} from '../modules/consciousness_dashboard.js';
import { consciousnessMark, wakeCheckChip, wakeCheckText, wakeCostText } from '../modules/consciousness_words.js';
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
        assert.ok(html.indexOf(full().coverage) > html.lastIndexOf('</section>'), 'the lists come first, their scope is their caption');
        const groups = html.split('<section ').slice(1);
        assert.deepEqual(groups.map((group) => group.match(/<h3[^>]*>([^<]+)</)[1]), ['Its own wakes', 'Wakes in places']);
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
        ['wake_paused', 'The last wake-up returned while paused.'], ['disabled', 'Off: no own wakes; 2 places only hear.'],
        ['waiting_for_first_conversation', 'No owner chat is bound yet; the first conversation binds it.']]) {
        assert.deepEqual(headlineView({ ...full(), status, detail }), { text: detail, moving: false });
    }
    assert.equal(headlineView({ ...full(), status: 'thinking' }).moving, true);
});

test('the headline card says the money of the last 24 hours alone; where it went is the card\'s Details', () => {
    const snapshot = fromState();
    const html = summaryHtml(snapshot);
    const [main, details] = [html.split('<details')[0], html.slice(html.indexOf('<details'))];
    assert.match(strip(main), /\$\d[\d,]*\.\d\d of \$\d[\d,]*\.\d\d in the last 24 hours/, 'one money line, grouped like Costs');
    assert.doesNotMatch(main, /Slack · Acme \$|Started work|Some calls have no price|on its own clock/, 'no run-on of causes on the card');
    assert.match(main, /class="dashboard-hero"/);
    // Each cause that spent something, every place's included (no place row shows its own spend any more); the
    // caveat once, at the end.
    assert.deepEqual(spendDetails(snapshot).rows.map((row) => `${row.words} ${row.amount}`),
        ['Own time $0.62', 'Slack · Acme $0.88', 'Started work $0.48']);
    assert.equal(spendDetails(snapshot).unpriced, true);
    assert.deepEqual(spendDetails({ ...snapshot, spendByCause: [{ cause: 'own', label: '', usd: 0, unmeteredCalls: 0 }] }),
        { rows: [], unpriced: false }, 'a zero is not listed');
    // The allowance, its own tasks, its conversations, its model and its places are rows of the settings cards
    // (dashboard_settings.test.js), so no line here leads to Costs, Settings or Circles.
    assert.doesNotMatch(html, /data-cx-act="(costs|settings|circles)"|Costs ›|Settings ›|Circles ›/);
    assert.match(strip(main), /Stop also ends a wake that is running now\./, 'stopping is said to end a running wake');
    assert.doesNotMatch(summaryHtml({ ...snapshot, enabled: false }), /Stop also ends/, 'said only while there is a Stop');
    assert.match(details, /^<details class="ui-disclosure consciousness-details"><summary>Details<\/summary>/);
    assert.match(strip(details), /Own time\s*\$0\.62.*Some calls have no price\..*Release notes draft \(Slack · Acme\)/);
    assert.equal(countedRootsText(snapshot), 'Its own tasks: 1 of 2 · Release notes draft (Slack · Acme)');
    assert.match(strip(main), /Its own tasks: 1 of 2/);
    assert.match(summaryHtml(snapshot), /data-cx-act="toggle" data-enabled="1">Stop</);
    assert.doesNotMatch(summaryHtml(normalizeConsciousness({})), /data-cx-act="toggle"/);
    assert.match(summaryHtml(snapshot, { detailsOpen: true }), /consciousness-details" open>/, 'an opened Details stays open');
});

test('a state that cannot wake wears its failure mark and offers Restart; starting is said without alarm', () => {
    const stopped = normalizeConsciousness({ enabled: true, status: 'stopped',
        detail: 'Cannot wake: Ouroboros did not finish starting. Restart Ouroboros.' });
    const html = summaryHtml(stopped);
    assert.match(html, /data-shape="dot" data-hue="failed"/);
    assert.match(html, /data-cx-act="restart">Restart</);
    assert.ok(html.indexOf('data-cx-act="restart"') < html.indexOf('data-cx-act="toggle"'), 'the next step leads the actions');
    const starting = summaryHtml(normalizeConsciousness({ enabled: true, status: 'starting',
        detail: 'Waits for Ouroboros to finish starting.' }));
    assert.doesNotMatch(starting, /data-hue="failed"|data-cx-act="restart"/);
    assert.doesNotMatch(summaryHtml(normalizeConsciousness({ enabled: false, status: 'stopped', detail: 'Off.' })), /data-cx-act="restart"/);
    // One mark per state for the bar's chip and this headline (consciousness_words.consciousnessMark).
    assert.deepEqual(consciousnessMark(stopped), { shape: 'dot', hue: 'failed' });
    assert.deepEqual(consciousnessMark({ ...stopped, status: 'allowance_exhausted' }), { shape: 'dot', hue: 'owner' });
    assert.deepEqual(consciousnessMark({ ...stopped, status: 'thinking' }), { shape: 'moving', hue: 'ink' });
    assert.deepEqual(consciousnessMark({ ...stopped, enabled: false }), { shape: 'none', hue: 'ink' });
    // A held wake keeps the word `wake_paused`; only a proven allowance hold is the owner's move (amber), every other
    // or unproven cause keeps the pause bars.
    const held = normalizeConsciousness({ enabled: true, status: 'wake_paused', live_wake_task_id: 'w1', live_wake_held: true,
        detail: 'Wake-up w1 stopped at the allowance it had left when it started ($2.99).', live_wake_hold: { cause: 'allowance', ceiling_usd: 2.99 } });
    for (const cause of ['allowance', 'allowance_and_cap']) {
        assert.deepEqual(consciousnessMark({ ...held, holdCause: cause }), { shape: 'dot', hue: 'owner' }, cause);
    }
    for (const cause of ['per_task_cap', 'launch_ceiling', 'per_task_fence', 'total_budget', 'owner', 'sleep', 'technical', '']) {
        assert.deepEqual(consciousnessMark({ ...held, holdCause: cause }), { shape: 'pause', hue: 'ink' }, cause);
    }
    assert.deepEqual(consciousnessMark({ ...held, enabled: false }), { shape: 'none', hue: 'ink' }, 'Off wins');
    assert.deepEqual(consciousnessMark({ ...held, liveWakeHeld: false }), { shape: 'pause', hue: 'ink' }, 'a hold needs the held wake');
    assert.match(summaryHtml(held), /consciousness-headline">Wake-up w1 stopped at the allowance.*data-hue="owner"/);
    // What follows from the hold is said beneath the headline, quieter; no other state has a note.
    const noted = normalizeConsciousness({ ...held, detail_note: 'Resume keeps that limit.' });
    assert.match(summaryHtml(noted), /<p class="dashboard-hero-meta consciousness-headline-note">Resume keeps that limit\.<\/p>/);
    assert.doesNotMatch(summaryHtml(held), /consciousness-headline-note/);
    assert.equal(headlineView(noted).note, 'Resume keeps that limit.');
    assert.ok(!('note' in headlineView(held)), 'a state without a note keeps the headline shape it had');
    // A Start/Stop on its way says so and waits; nothing else on the card changes.
    assert.match(summaryHtml(fromState(), { toggling: 'stop' }), /data-cx-act="toggle" data-enabled="1" disabled>Stopping…</);
    assert.match(summaryHtml({ ...fromState(), enabled: false }, { toggling: 'start' }), /disabled>Starting…</);
    assert.match(summaryHtml(fromState()), /class="btn btn-default btn-sm" data-cx-act="toggle"/, 'the neutral small capsule');
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
    assert.match(wakesCardHtml(full().recentWakes, full().coverage), /Wakes that reached a model call in the last 48 h/);
    // The details of a row are the one shared disclosure, its words "Details", no own marker and no indent.
    assert.match(oldHtml, /<details class="ui-disclosure consciousness-wake-more"><summary>Details<\/summary>/);
});

test('a warning chip says what the Safety Supervisor did; a raw tool id waits in the row\'s Details', () => {
    assert.equal(wakeCheckChip({ chip: 'warned', tool: 'ext_8_r_github_github_place_message' }), 'Safety Supervisor warned about a step');
    assert.equal(wakeCheckChip({ chip: 'held', tool: 'run_command' }), 'Safety Supervisor held a step');
    assert.equal(wakeCheckChip({ chip: 'sent_despite', tool: 'slack_send' }), 'A step was let through despite the check');
    assert.equal(wakeCheckChip({ chip: 'sent_despite', tool: 'run_command' }), 'A step was let through despite the check');
    assert.equal(wakeCheckChip({ chip: 'sent_despite', tool: 'place_reply' }), 'The reply sent despite the check');
    assert.equal(wakeCheckChip({ chip: 'warned', tool: 'place_reply' }), 'Safety Supervisor warned about the reply', 'a tool it names in words stays');
    const row = wakeRowHtml({ ...wake('w-summary'), checks: [{ chip: 'warned', tool: 'ext_8_r_github_github_place_message' }] });
    const [main, details] = [row.split('<details')[0], row.slice(row.indexOf('<details'))];
    assert.doesNotMatch(main, /ext_8_r_github/, 'no raw id on the row');
    assert.match(main, /Safety Supervisor warned about a step/);
    assert.match(details, /Safety Supervisor warned about ext_8_r_github_github_place_message/, 'the exact tool is one click away');
});

test('unknown, zero, known, open and mixed costs stay distinct in visible and accessible words', () => {
    // An unknown cost leaves the row's money column empty and is said in the row's Details, never as a zero.
    assert.equal(wakeCostText(wake('214a31b6')), '');
    assert.match(wakeRowHtml(wake('214a31b6')), /<span class="consciousness-wake-cost"><\/span>/);
    assert.match(wakeRowHtml(wake('214a31b6')).slice(wakeRowHtml(wake('214a31b6')).indexOf('<details')), /Cost unknown/);
    assert.doesNotMatch(wakeRowHtml(wake('w-running')), /Cost unknown/);
    assert.equal(wakeCostText(wake('w-running')), 'up to $0.02 so far');
    const priced = (amount, extra = {}) => normalizeWake({ cost_presentation: {
        has_rows: true, tracked_amount: amount, tracked_final: true, has_unpriced: false, accounting_open: false, ...extra } });
    assert.equal(wakeCostText(priced(0)), '$0.00');
    assert.equal(wakeCostText(priced(1.23)), '$1.23');
    // No amount — unknown, still pending, or settled without a price — leaves the column empty for the row's Details.
    assert.equal(wakeCostText(priced(null)), '');
    assert.equal(wakeCostText(priced(null, { has_unpriced: true })), '');
    assert.equal(wakeCostText(priced(null, { has_unpriced: true, accounting_open: true, tracked_final: false })), '');
    assert.equal(wakeCostText(priced(1.23, { tracked_final: false, accounting_open: true })), 'up to $1.23 so far');
    assert.equal(wakeCostText(priced(1.23, { has_unpriced: true })), 'Tracked: $1.23');
    // Custody facts ride with the wake: an unreadable view without a known amount says
    // unknown and gives no pending explanation; a known subtotal keeps its qualified money.
    const custody = (amount) => normalizeWake({ task_id: `w-custody-${amount}`, cost_custody: { status: 'unavailable' },
        cost_presentation: { has_rows: true, tracked_amount: amount, tracked_final: false, has_unpriced: true, accounting_open: true } });
    assert.equal(custody(null).cost.cost_custody.status, 'unavailable');
    assert.equal(wakeCostText(custody(null)), '');
    assert.doesNotMatch(wakeRowHtml(custody(null), { open: true }), /accounting is still open/);
    assert.equal(wakeCostText(custody(2)), 'Tracked: up to $2.00 so far');
    assert.match(wakeRowHtml(custody(2), { open: true }), /Tracked counts only the priced steps/);
});

test('the Dashboard lists no places: no card and no place read at mount, show, poll, change or lifecycle', async (t) => {
    const { window } = setup(t);
    const mount = fakeMount();
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'consciousness' };
    let reads = 0;
    const controller = initConsciousnessDashboard({ mount, state, read: async () => full(),
        readPlacesFn: async () => { ++reads; return normalizePlacesResponse(FIXTURE.places); } });
    const settled = () => new Promise((resolve) => setImmediate(resolve));
    await settled();
    await controller.refresh();
    window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'consciousness' } })); await settled();
    window.dispatchEvent(new CustomEvent('ouro:places-changed')); await settled();
    window.dispatchEvent(new Event('ouro:skill-lifecycle')); await settled();
    assert.equal(reads, 0, 'the places are listed on Circles; a wake\'s door reads them when pressed');
    assert.equal(mount.regions.has('[data-cx-places]'), false, 'no Places card region');
    assert.match(mount.regions.get('[data-cx-summary]').innerHTML, /class="dashboard-hero"/);
    controller.destroy();
});

function fakeMount() {
    const regions = new Map();
    return {
        innerHTML: '',
        regions,
        querySelector(selector) {
            if (!regions.has(selector)) regions.set(selector, { innerHTML: '', textContent: '', dataset: {}, querySelector: () => null, querySelectorAll: () => [], closest: () => null,
                children: [], addEventListener() {}, removeEventListener() {} });
            return regions.get(selector);
        },
        addEventListener() {}, removeEventListener() {},
    };
}

test('while the full snapshot is not served, /api/state still gives the status; no place list is painted', async () => {
    const priorWindow = globalThis.window;
    const priorDocument = globalThis.document;
    globalThis.window = { addEventListener() {}, removeEventListener() {}, dispatchEvent() {} };
    globalThis.document = { hidden: false, addEventListener() {}, removeEventListener() {} };
    const mount = fakeMount();
    const notServed = Object.assign(new Error('HTTP 404'), { status: 404 });
    let reads = 0;
    const controller = initConsciousnessDashboard({
        mount, ws: { send() {} }, state: { activePage: 'dashboard', dashboardActiveSubtab: 'other' },
        read: async () => { throw notServed; },
        readFallback: async () => fromState(),
        readPlacesFn: async () => { ++reads; return normalizePlacesResponse(FIXTURE.places); },
    });
    try {
        await controller.refresh();
        assert.match(mount.regions.get('[data-cx-summary]').innerHTML, /Consciousness · on/);
        assert.equal(reads, 0);
        assert.equal(mount.regions.has('[data-cx-places]'), false);
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

test('a wake without a summary leads with its whole note, edges trimmed; a summary leads instead; more holds the note', () => {
    // A short first paragraph, then the part that matters: the lead holds every word; CSS alone cuts it to one line.
    const note = '\n  Swept the stale ports.\n\nThe tail that matters: the Slack token expires on Friday.  \n';
    const lead = wakeRowHtml({ ...wake('214a31b6'), note });
    const [collapsed, disclosed] = [lead.split('<details')[0], lead.slice(lead.indexOf('<details'))];
    assert.ok(collapsed.includes(`<p class="consciousness-wake-lead" data-i18n-authored>${note.trim()}</p>`), collapsed);
    assert.ok(disclosed.includes(`data-i18n-authored>${note}</div>`), 'more holds the note as stored');
    const named = wakeRowHtml({ ...wake('214a31b6'), note, summary: 'Ports swept; one token to renew.' });
    const summary = '<p class="consciousness-wake-summary" data-i18n-authored>Ports swept; one token to renew.</p>';
    assert.ok(named.split('<details')[0].includes(summary), 'the summary leads, whole');
    assert.doesNotMatch(named, /consciousness-wake-lead/, 'a summary leads instead of the note');
    assert.ok(named.includes(`data-i18n-authored>${note}</div>`), 'and more still holds the whole note');
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
    const last = mount.querySelector('[data-wake]').querySelector('details');
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
        // Each press reads the places as they stand: the second press's read already holds the later room.
        readPlacesFn: async () => ({ places: [++reads === 1 ? slack : { ...slack, rooms: [...slack.rooms, { chatId: 1099511627778 }] }], connectors: [], migrationProblems: [] }),
        openSheet: (place, options) => opened.push([place.id, options.openRoomChatId]) });
    await controller.refresh();
    const [known, later] = mount.querySelectorAll('[data-cx-open-room]');
    await mount.fire('click', { target: known });
    assert.deepEqual(opened, [[slack.id, 1099511627777]], 'the sheet opens on the row\'s conversation');
    assert.equal(reads, 1, 'nothing is read before a press; one read for it');
    await mount.fire('click', { target: later });
    assert.equal(reads, 2, 'each press reads the places as they stand');
    assert.deepEqual(opened[1], [slack.id, 1099511627778], 'a room captured after an earlier press opens');
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

test('a second press during the door\'s read is ignored; a failed read leaves the door and says so in the tab', async (t) => {
    const cleanups = [];
    const { mount } = setup({ after: cleanup => cleanups.push(cleanup) });
    let controller;
    t.after(() => { controller?.destroy(); cleanups.forEach(cleanup => cleanup()); });
    Object.defineProperty(NodeStub.prototype, 'content', { configurable: true, get() { return this; } });
    t.after(() => { delete NodeStub.prototype.content; });
    NodeStub.prototype.getBoundingClientRect = () => ({ top: 0 });
    t.after(() => { delete NodeStub.prototype.getBoundingClientRect; });
    const slack = { ...normalizePlacesResponse(FIXTURE.places).places[0], rooms: [{ chatId: 1099511627778 }] };
    const snapshot = { ...full(), recentWakes: [
        { ...wake('214a31b6'), taskId: 'focus-late', cause: 'focus', chatId: 1099511627778, placeLabel: 'Slack' },
        { ...wake('214a31b6'), taskId: 'focus-lost', cause: 'focus', chatId: 1099511627779, placeLabel: 'Slack' }] };
    const opened = [];
    let reads = 0, settle = null, fail = false;
    controller = initConsciousnessDashboard({ mount, state: { activePage: 'chat' },
        read: async () => snapshot,
        readPlacesFn: () => {
            reads += 1;
            if (fail) return Promise.reject(new Error('HTTP 503'));
            return new Promise((resolve) => { settle = resolve; });
        },
        openSheet: (place, options) => opened.push([place.id, options.openRoomChatId]) });
    await controller.refresh();
    const [late, lost] = mount.querySelectorAll('[data-cx-open-room]');
    const first = mount.fire('click', { target: late });
    await new Promise((resolve) => setImmediate(resolve));
    await mount.fire('click', { target: late });
    assert.equal(reads, 1, 'the second press starts no second read');
    settle({ places: [slack], connectors: [], migrationProblems: [] });
    await first;
    assert.deepEqual(opened, [[slack.id, 1099511627778]], 'one sheet for two presses');
    assert.equal(late.disabled, false);
    fail = true;
    await mount.fire('click', { target: lost });
    assert.equal(opened.length, 1);
    assert.equal(lost.disabled, false, 'a failed read is not absence: the door stays');
    assert.match(mount.querySelector('[data-cx-status]').textContent, /Places could not be read: HTTP 503/, 'the tab names the failed read');
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

test('a read that never reached Ouroboros says so in owner words; a server answer keeps its own', async (t) => {
    const cleanups = [];
    const { mount } = setup({ after: cleanup => cleanups.push(cleanup) });
    let controller;
    t.after(() => { controller?.destroy(); cleanups.forEach(cleanup => cleanup()); });
    controller = initConsciousnessDashboard({ mount, state: { activePage: 'chat' },
        read: async () => { throw new TypeError('Load failed'); },
        readFallback: async () => { throw new TypeError('Load failed'); },
        readPlacesFn: async () => { throw new TypeError('Load failed'); } });
    await controller.refresh();
    const said = mount.querySelector('[data-cx-status]').textContent;
    assert.match(said, /Could not read consciousness: Couldn't reach Ouroboros\.(?!\.)/, 'one full stop, never two');
    assert.doesNotMatch(said, /Load failed/, "the browser's transport text is never the sentence");
    controller.destroy();
    // The places are read at a wake's door: a read that never reached Ouroboros is said in owner words there too.
    Object.defineProperty(NodeStub.prototype, 'content', { configurable: true, get() { return this; } });
    t.after(() => { delete NodeStub.prototype.content; });
    NodeStub.prototype.getBoundingClientRect = () => ({ top: 0 });
    t.after(() => { delete NodeStub.prototype.getBoundingClientRect; });
    const snapshot = { ...full(), recentWakes: [{ ...wake('214a31b6'), taskId: 'focus-x', cause: 'focus', chatId: 1099511627777, placeLabel: 'Slack' }] };
    controller = initConsciousnessDashboard({ mount, state: { activePage: 'chat' }, read: async () => snapshot,
        readPlacesFn: async () => { throw new TypeError('Load failed'); } });
    await controller.refresh();
    await mount.fire('click', { target: mount.querySelector('[data-cx-open-room]') });
    const door = mount.querySelector('[data-cx-status]').textContent;
    assert.match(door, /Places could not be read: Couldn(?:'|&#39;)t reach Ouroboros\./);
    assert.doesNotMatch(door, /Load failed/);
});

test('Safety chips name what they checked in sentence case, keeping raw tool ids as they are', () => {
    assert.equal(wakeCheckText({ chip: 'warned', tool: 'place_reply' }), 'Safety Supervisor warned about the reply');
    assert.equal(wakeCheckText({ chip: 'sent_despite', tool: 'place_reply' }), 'The reply sent despite the check');
    assert.equal(wakeCheckText({ chip: 'held', tool: 'send_user_message' }), 'Safety Supervisor held the message');
    assert.equal(wakeCheckText({ chip: 'sent_despite', tool: 'slack_send' }), 'slack_send let through despite the check');
    assert.equal(wakeCheckText({ chip: 'sent_despite', tool: 'run_command' }), 'run_command let through despite the check');
    assert.equal(wakeCheckText({ chip: 'not_checked', tool: 'update_memory' }), 'Not checked (update_memory)');
    assert.equal(wakeCheckText({ chip: 'not_checked', tool: '' }), 'Not checked');
    assert.equal(wakeCheckText({ chip: 'unknown', tool: 'x' }), '');
});
