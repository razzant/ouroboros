// The consciousness chip in a room's bar and its Shade (docs/DESIGN.md "Page header", "Status marks"): the chip's
// word and one mark for every state the host tells apart, Evolving even with consciousness off, the host's own state
// sentence, the Shade's money line and three wakes in the Dashboard's phase words, and a chip that opens its Shade
// without switching anything.
import assert from 'node:assert/strict';
import test from 'node:test';
import { allowanceText, chipView, createConsciousnessStatus, wakeRows } from '../modules/consciousness_status.js';
import { normalizeConsciousness } from '../modules/consciousness_api.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

test.beforeEach((t) => {
    const started = performance.now();
    t.mock.method(Date, 'now', () => new Date(2026, 9, 25, 14, 0).getTime() + performance.now() - started);
});

const at = (hours, minutes) => new Date(2026, 9, 25, hours, minutes).toISOString();
const answer = (bg = {}, extra = {}) => ({
    bg_consciousness_enabled: true,
    bg_consciousness_state: { status: 'sleeping', detail: 'Sleeping until 14:20.', next_wake_at: at(14, 20), spent_24h_usd: 1.5,
        daily_usd: 20, ...bg },
    evolution_enabled: false, evolution_state: { status: 'idle' },
    ...extra,
});
const shapeOf = (view) => `${view.mark.shape}/${view.mark.hue}`;

test('a due chip names its appointment with a still mark; a held wake has a pause mark and no time', () => {
    const due = chipView(answer({ status: 'due', detail: 'Due since 13:10; the wake-up has not started yet.',
        next_wake_at: at(13, 50), appointment_at: at(13, 10) }));
    assert.deepEqual([due.word, shapeOf(due)], ['since 13:10', 'still/ink']);
    assert.equal(due.said, 'Consciousness: Due since 13:10; the wake-up has not started yet.');
    const legacy = chipView(answer({ status: 'due', next_wake_at: at(13, 50) }));
    assert.deepEqual([legacy.word, shapeOf(legacy)], ['due', 'still/ink']);
    const held = chipView(answer({ status: 'wake_paused', live_wake_task_id: 'own', live_wake_held: true,
        detail: 'Wake-up own is paused or waiting in the queue.', next_wake_at: at(13, 50), appointment_at: at(13, 10) }));
    assert.deepEqual([held.word, shapeOf(held)], ['', 'pause/ink']);
    assert.doesNotMatch(held.said, /13:|14:|since/);
});

test('the chip says the next wake, Off or Evolving, beside one mark of the status-mark family', () => {
    const sleeping = chipView(answer());
    assert.deepEqual([sleeping.word, shapeOf(sleeping), sleeping.off], ['14:20', 'none/ink', false]);
    assert.equal(sleeping.said, 'Consciousness: Sleeping until 14:20.', 'the host\'s own sentence (consciousness_view.py)');
    const thinking = chipView(answer({ status: 'thinking', detail: 'Wake-up wake-1 is running.', live_wake_task_id: 'wake-1' }));
    assert.deepEqual([thinking.word, shapeOf(thinking)], ['', 'moving/ink'], 'awake: the moving dots, never a breathing glyph');
    const off = chipView(answer({ status: 'disabled', detail: 'Off: no own wakes; 2 places only hear.' }, { bg_consciousness_enabled: false }));
    assert.deepEqual([off.word, shapeOf(off), off.off], ['Off', 'none/ink', true]);
    // Evolving replaces the time while a campaign runs, even with consciousness off.
    const evolvingOff = chipView(answer({ status: 'disabled', detail: 'Off: no own wakes.' },
        { bg_consciousness_enabled: false, evolution_enabled: true }));
    assert.deepEqual([evolvingOff.word, evolvingOff.off, evolvingOff.evolving], ['Evolving', true, true]);
    assert.equal(evolvingOff.said, 'Consciousness: Off: no own wakes. · evolution campaign running');
    assert.equal(chipView(answer({}, { evolution_enabled: true })).word, 'Evolving');
    assert.equal(shapeOf(chipView(answer({ status: 'thinking', live_wake_task_id: 'w' }, { evolution_enabled: true }))), 'moving/ink');
});

test('amber only for the owner\'s move; a failure is the outcome\'s red dot, an unconfirmed state a ring', () => {
    const spent = chipView(answer({ status: 'allowance_exhausted', allowance_resets_at: at(15, 5), next_wake_at: at(15, 5),
        detail: 'Daily allowance spent ($20.00 of $20.00 in the last 24 h); next check at 15:05.' }));
    assert.deepEqual([spent.word, shapeOf(spent)], ['15:05', 'dot/owner'], 'stopped by its spending limit: the time it frees');
    assert.match(spent.said, /next check at 15:05\. · waits for you$/);
    const expired = chipView(answer({ status: 'allowance_exhausted', allowance_resets_at: at(13, 5),
        detail: 'Daily allowance spent ($20.00 of $20.00 in the last 24 h); next check due since 13:05.' }));
    assert.deepEqual([expired.word, shapeOf(expired)], ['', 'dot/owner']);
    assert.equal(expired.said, 'Consciousness: Daily allowance spent ($20.00 of $20.00 in the last 24 h); next check due since 13:05. · waits for you');
    assert.equal(shapeOf(chipView(answer({ status: 'waiting_for_first_conversation' }))), 'dot/owner');
    // A campaign that waits for the owner is the owner's move too, whatever consciousness does.
    const campaign = answer({ status: 'disabled' }, { bg_consciousness_enabled: false, evolution_enabled: true,
        evolution_state: { status: 'budget_stopped' } });
    assert.equal(shapeOf(chipView(campaign)), 'dot/owner');
    assert.equal(shapeOf(chipView({ ...campaign, evolution_enabled: false })), 'none/ink', 'no campaign, no campaign amber');
    for (const status of ['wake_failed', 'wake_rejected', 'stopped']) {
        const view = chipView(answer({ status }));
        assert.deepEqual([shapeOf(view), view.word], ['dot/failed', '14:20'], status);
        assert.equal(chipView(answer({ status, next_wake_at: at(13, 20) })).word, '');
    }
    for (const status of ['wake_outcome_unknown', 'allowance_unknown']) assert.equal(shapeOf(chipView(answer({ status }))), 'ring/ink', status);
    assert.equal(shapeOf(chipView(answer({ status: 'wake_paused' }))), 'pause/ink');
    // A wake held at the allowance it started with, proven by its records, is the owner's move; the chip keeps no time.
    const held = { status: 'wake_paused', live_wake_task_id: 'w1', live_wake_held: true,
        detail: 'Wake-up w1 stopped at the allowance it had left when it started ($2.99).' };
    const allowance = chipView(answer({ ...held, live_wake_hold: { cause: 'allowance', ceiling_usd: 2.99 } }));
    assert.deepEqual([allowance.word, shapeOf(allowance)], ['', 'dot/owner']);
    assert.equal(allowance.said, `Consciousness: ${held.detail} · waits for you`);
    for (const cause of ['per_task_cap', 'launch_ceiling', 'owner', '']) {
        assert.equal(shapeOf(chipView(answer({ ...held, live_wake_hold: { cause, ceiling_usd: null } }))), 'pause/ink', cause);
    }
    // An unknown control is never Off: a ring and no word.
    const unknown = chipView({ bg_consciousness_state: {}, bg_consciousness_enabled: null });
    assert.deepEqual([unknown.word, shapeOf(unknown), unknown.off], ['', 'ring/ink', false]);
    // What goes on by itself is never amber.
    for (const status of ['sleeping', 'thinking', 'wake_paused', 'wake_failed']) {
        assert.notEqual(chipView(answer({ status })).mark.hue, 'owner', status);
    }
});

test('the money of the last 24 hours, and the three last wakes in the Dashboard\'s phase words', () => {
    const snapshot = (bg) => normalizeConsciousness(bg, { enabled: true });
    assert.equal(allowanceText(snapshot({ spent_24h_usd: 1.5, daily_usd: 20 })), '$1.50 of $20.00 in the last 24 hours');
    assert.equal(allowanceText(snapshot({ spent_24h_usd: 1.5, daily_usd: 20, unknown_unmetered: 2 })), '≥ $1.50 of $20.00 in the last 24 hours');
    assert.equal(allowanceText(snapshot({ daily_usd: 20 })), '', 'an unknown spend is unknown, never $0');
    const wakes = normalizeConsciousness({ recent_wakes: [
        { started_at: at(13, 52), summary: 'Reminded about the Swarm question', phase: 'done', ended_at: at(13, 53) },
        { started_at: at(12, 10), cause: 'focus', place_label: 'Slack · Acme', phase: 'done', quiet: true, ended_at: at(12, 11) },
        { started_at: at(11, 0), summary: 'Reading notes', phase: 'running', live: true },
        { started_at: at(10, 31), summary: 'Too old for the Shade', phase: 'done', ended_at: at(10, 32) },
    ] }).recentWakes;
    assert.deepEqual(wakeRows(wakes), [
        { time: '13:52', text: 'Reminded about the Swarm question', outcome: 'Done' },
        { time: '12:10', text: 'Slack · Acme', outcome: 'Done · no message' },
        { time: '11:00', text: 'Reading notes', outcome: 'Working' },
    ]);
    assert.deepEqual(wakeRows(null), []);
});

test('the chip opens its Shade and switches nothing; the Shade reads the wakes once and holds Start/Stop and its doors', async () => {
    const env = installDom();
    try {
        const bar = document.createElement('div');
        const title = document.createElement('div');
        title.className = 'app-page-title-block';
        const actions = document.createElement('div');
        bar.append(title, actions);
        const sent = [];
        const tabs = [];
        let reads = 0;
        let refreshed = 0;
        const status = createConsciousnessStatus({
            bar, ws: { send: (frame) => { sent.push(frame); return { status: 'sent' }; }, isConnected: () => true },
            openDashboardTab: (tab) => tabs.push(tab), refreshState: () => { refreshed += 1; },
            read: async () => {
                reads += 1;
                return normalizeConsciousness({ recent_wakes: [
                    { started_at: at(13, 52), summary: 'Reminded about the Swarm question', phase: 'done', ended_at: at(13, 53) },
                ] });
            },
        });
        assert.equal(status.chip.hidden, true, 'nothing is said before the first answer');
        status.update({ accounting: { available: false } });
        assert.equal(status.chip.hidden, true, 'a failed read says nothing new');
        status.update(answer());
        assert.equal(bar.children.indexOf(status.chip), 1, 'right after the room\'s title and status');
        assert.equal(status.chip.querySelector('.cx-chip-word').textContent, '14:20');
        assert.equal(status.chip.querySelector('.cx-chip-word').hidden, false);
        assert.equal(status.chip.getAttribute('aria-label'), 'Consciousness: Sleeping until 14:20.');

        status.chip.click();
        assert.equal(sent.length, 0, 'a tap on the chip never toggles anything');
        assert.equal(status.isOpen(), true);
        assert.equal(status.chip.getAttribute('aria-expanded'), 'true');
        assert.equal(bar.hasAttribute('data-cx-shade'), true, 'Main\'s history note waits behind the open Shade');
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(reads, 1, 'one read of the wakes per opening, no poller');
        const shade = status.shade;
        assert.equal(shade.querySelector('.cx-state').textContent, 'Sleeping until 14:20.');
        assert.equal(shade.querySelector('.cx-money').textContent, '$1.50 of $20.00 in the last 24 hours');
        assert.deepEqual(shade.querySelectorAll('.cx-wake').map((row) => row.children.map((cell) => cell.textContent)),
            [['13:52', 'Reminded about the Swarm question', 'Done']]);
        const toggle = shade.querySelector('.cx-toggle');
        assert.equal(toggle.textContent, 'Stop');
        const doors = shade.querySelectorAll('.cx-door');
        assert.deepEqual(doors.map((door) => [door.textContent, door.hidden]),
            [['Dashboard ›', false], ['Costs ›', true], ['Evolution ›', true]], 'Costs › only while the allowance is the next step');

        toggle.click();
        assert.deepEqual(sent, [{ type: 'command', cmd: '/bg stop', ui_control: { name: 'consciousness_toggle' } }],
            'Stop is the Dashboard\'s own command, not typed owner text');
        assert.equal(toggle.disabled, true, 'it waits for the readback');
        await new Promise((resolve) => setTimeout(resolve, 450));
        assert.equal(refreshed, 1, 'the room\'s state clock is asked again');
        status.update(answer({ status: 'disabled' }, { bg_consciousness_enabled: false, evolution_enabled: true }));
        assert.deepEqual([toggle.textContent, toggle.disabled], ['Start', false]);
        assert.equal(status.chip.querySelector('.cx-chip-word').textContent, 'Evolving');
        assert.equal(doors[2].hidden, false, 'Evolution › while a campaign runs');

        // A door closes the Shade and opens its Dashboard tab.
        doors[2].parentElement.listeners.get('click')[0]({ target: { closest: () => doors[2] } });
        assert.deepEqual(tabs, ['evolution']);
        assert.equal(status.isOpen(), false);
        assert.equal(bar.hasAttribute('data-cx-shade'), false);

        status.destroy();
        assert.equal(bar.children.includes(status.chip), false, 'its nodes go with it');
    } finally {
        restoreDom(env.prior);
    }
});

test('while the allowance is the next step the Shade offers Costs, and its door opens Costs', async () => {
    const env = installDom();
    try {
        const bar = document.createElement('div');
        const tabs = [];
        const status = createConsciousnessStatus({ bar, ws: { send() {}, isConnected: () => true },
            openDashboardTab: (tab) => tabs.push(tab), read: async () => normalizeConsciousness({ recent_wakes: [] }) });
        const held = { status: 'wake_paused', live_wake_task_id: 'w1', live_wake_held: true,
            detail: 'Wake-up w1 stopped at the allowance it had left when it started ($2.99).',
            live_wake_hold: { cause: 'allowance', ceiling_usd: 2.99 } };
        status.update(answer(held));
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const costs = status.shade.querySelectorAll('.cx-door').find((door) => door.textContent === 'Costs ›');
        assert.equal(costs.hidden, false, 'a wake held at its allowance');
        assert.equal(status.shade.querySelector('.cx-state').textContent, held.detail, 'the host\'s reason, no clock added');
        const note = status.shade.querySelector('.cx-state-note');
        assert.equal(note.hidden, true, 'no note without one');
        status.update(answer({ ...held, detail_note: 'Resume keeps that limit.' }));
        assert.deepEqual([note.textContent, note.hidden], ['Resume keeps that limit.', false]);
        for (const [bg, shown] of [[{ status: 'allowance_exhausted', detail: 'Daily allowance spent.' }, true],
            [{ ...held, live_wake_hold: { cause: 'per_task_cap', ceiling_usd: 10 } }, false],
            [{ ...held, live_wake_hold: { cause: 'owner', ceiling_usd: null } }, false],
            [{ status: 'sleeping' }, false]]) {
            status.update(answer(bg));
            assert.equal(costs.hidden, !shown, JSON.stringify(bg.live_wake_hold || bg.status));
        }
        status.update(answer({ status: 'allowance_exhausted', detail: 'Daily allowance spent.' }, { bg_consciousness_enabled: false }));
        assert.equal(costs.hidden, true, 'Off is no allowance step');
        status.update(answer(held));
        costs.parentElement.listeners.get('click')[0]({ target: { closest: () => costs } });
        assert.deepEqual(tabs, ['costs']);
        assert.equal(status.isOpen(), false, 'the door closes the Shade');
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

test('a wake in a place is a door to that place\'s sheet, read at the press; a failed read keeps it, a place gone disables it', async () => {
    // Owner 06.11: pausing one Slack took four presses in another section; the Shade's row is one press from its sheet.
    const env = installDom();
    let status;
    try {
        const bar = document.createElement('div');
        const opened = [];
        let places = [{ id: 'slack-1', label: 'Slack · Acme' }];
        let fail = false;
        status = createConsciousnessStatus({ bar,
            read: async () => normalizeConsciousness({ recent_wakes: [
                { cause: 'focus', place_id: 'slack-1', place_label: 'Slack · Acme', summary: 'Answered Ann',
                    started_at: at(13, 5), phase: 'done', ended_at: at(13, 6) },
                { cause: 'focus', place_label: 'Mail', summary: 'Read the inbox', started_at: at(12, 0), phase: 'done', ended_at: at(12, 1) },
                { cause: 'own', place_id: 'slack-1', summary: 'Own time', started_at: at(11, 0), phase: 'done', ended_at: at(11, 1) },
            ] }),
            readPlacesFn: async () => { if (fail) throw new Error('offline'); return { places }; },
            openSheet: (place, { returnFocus }) => opened.push([place.id, returnFocus === status.chip]),
        });
        status.update(answer());
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        let doors = status.shade.querySelectorAll('.cx-wake-door');
        assert.equal(doors.length, 1, 'only a place wake that names its place is a door; an own wake never is');
        assert.equal(doors[0].getAttribute('aria-label'), 'Open Slack · Acme: 13:05 · Answered Ann · Done');
        assert.deepEqual(doors[0].children.map((cell) => cell.textContent), ['13:05', 'Answered Ann', 'Done'], 'it keeps its words');

        fail = true;
        doors[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(opened, []);
        assert.equal(status.isOpen(), true);
        assert.equal(status.shade.querySelector('.cx-wakes-coverage').textContent, 'Places could not be read: offline');
        assert.equal(doors[0].disabled, false, 'a failed read keeps the door for another try');

        fail = false;
        doors[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(opened, [['slack-1', true]], 'the place as it stands now, focus returning to the chip');
        assert.equal(status.isOpen(), false, 'the Shade gives way to the sheet');

        places = [];
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        doors = status.shade.querySelectorAll('.cx-wake-door');
        doors[0].focus();
        doors[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(opened, [['slack-1', true]]);
        assert.equal(doors[0].disabled, true, 'a place gone since is no door');
        assert.equal(document.activeElement, status.shade.querySelector('.cx-wakes-coverage'), 'its focus moves to the line saying why');
        assert.equal(status.shade.querySelector('.cx-wakes-coverage').textContent, 'This place is disconnected.');
    } finally {
        status?.destroy();
        restoreDom(env.prior);
    }
});

test('Shade keeps three own wakes and three place wakes, with independent empty sentences on the next opening', async () => {
    const env = installDom();
    let status;
    try {
        const bar = document.createElement('div');
        let wakes = ['focus', 'own'].flatMap((cause) => Array.from({ length: 4 }, (_, n) => ({
            cause, summary: `${cause}-${n}`, started_at: at(13, n), phase: 'done',
        })));
        let coverage = 'wakes that reached a model call in the last 48 h';
        status = createConsciousnessStatus({ bar, read: async () => normalizeConsciousness({
            recent_wakes: wakes, recent_wakes_coverage: coverage }) });
        status.update(answer());
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(status.shade.querySelectorAll('.cx-wakes-title').map((node) => node.textContent),
            ['Its own wakes', 'Wakes in places']);
        const lists = status.shade.querySelectorAll('.cx-wakes');
        assert.deepEqual(lists.map((list) => list.getAttribute('aria-label')), ['Its own wakes', 'Wakes in places']);
        assert.deepEqual(lists.map((list) => list.querySelectorAll('.cx-wake-text').map((node) => node.textContent)),
            [['own-0', 'own-1', 'own-2'], ['focus-0', 'focus-1', 'focus-2']]);
        const note = status.shade.querySelector('.cx-wakes-coverage');
        assert.equal(note.textContent, coverage);
        assert.deepEqual(status.shade.querySelectorAll('.cx-wakes-coverage').map((node) => node.textContent), [coverage]);
        assert.ok(note.parentElement.children.indexOf(note) < note.parentElement.children.indexOf(lists[0]));
        status.chip.click();
        wakes = null;
        coverage = 'Recent wakes could not be read: dialogue store busy.';
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(status.shade.querySelectorAll('.cx-wakes-coverage').map((node) => node.textContent), [coverage]);
        assert.ok(status.shade.querySelectorAll('.cx-wakes-note').every((node) => node.hidden));
        assert.ok(status.shade.querySelector('.cx-wakes-coverage').hasAttribute('data-i18n-authored'));  // the host's words
        status.chip.click();
        coverage = '';
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const fallback = status.shade.querySelector('.cx-wakes-coverage');
        assert.equal(fallback.textContent, 'Recent wakes could not be read.');
        assert.equal(fallback.hasAttribute('data-i18n-authored'), false);  // the client's own sentence is translated
        coverage = 'wakes that reached a model call in the last 48 h';
        for (const cause of ['focus', 'own', 'none']) {
            status.chip.click();
            wakes = cause === 'none' ? [] : [{ cause, summary: cause, phase: 'running' }];
            status.chip.click();
            await new Promise((resolve) => setTimeout(resolve, 0));
            assert.equal(note.textContent, coverage, 'a readable response clears the failure');
            const notes = status.shade.querySelectorAll('.cx-wakes-note');
            assert.equal(notes[0].hidden, cause === 'own');
            assert.equal(notes[1].hidden, cause === 'focus');
            if (cause !== 'own') assert.equal(notes[0].textContent, 'No own wakes with a recorded model call in the last 48 h.');
            if (cause !== 'focus') assert.equal(notes[1].textContent, 'No wakes in places in the last 48 h.');
        }
    } finally {
        status?.destroy();
        restoreDom(env.prior);
    }
});
