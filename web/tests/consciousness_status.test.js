// The consciousness chip in a room's bar and its Shade (docs/DESIGN.md "Page header", "Status marks"): the chip's
// word and one mark for every state the host tells apart, Evolving even with consciousness off, the host's own state
// sentence, the Shade's money line and three wakes in the Dashboard's phase words, and a chip that opens its Shade
// without switching anything.
import assert from 'node:assert/strict';
import test from 'node:test';
import { allowanceText, chipView, createConsciousnessStatus, wakeRows } from '../modules/consciousness_status.js';
import { normalizeConsciousness } from '../modules/consciousness_api.js';
import { turnConsciousness } from '../modules/consciousness_switch.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { createRoomHeader } from '../modules/chat_header.js';
import { createStateSnapshotSequencer } from '../modules/state_snapshots.js';

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

test('a due chip names its appointment with a still mark; a held wake has no time, its mark its cause\'s', () => {
    const due = chipView(answer({ status: 'due', detail: 'Due since 13:10; the wake-up has not started yet.',
        next_wake_at: at(13, 50), appointment_at: at(13, 10) }));
    assert.deepEqual([due.word, shapeOf(due)], ['since 13:10', 'still/ink']);
    assert.equal(due.said, 'Consciousness: Due since 13:10; the wake-up has not started yet.');
    const legacy = chipView(answer({ status: 'due', next_wake_at: at(13, 50) }));
    assert.deepEqual([legacy.word, shapeOf(legacy)], ['due', 'still/ink']);
    const held = chipView(answer({ status: 'wake_paused', live_wake_task_id: 'own', live_wake_held: true,
        detail: 'Its own wake-up is paused or waiting in the queue; its card shows why.', next_wake_at: at(13, 50),
        appointment_at: at(13, 10), live_wake_hold: { cause: 'owner', ceiling_usd: null } }));
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
    // An owner chat that could not be read is unconfirmed, never the owner's move (a known empty one wakes in Main).
    for (const status of ['wake_outcome_unknown', 'allowance_unknown', 'owner_chat_unknown']) {
        assert.equal(shapeOf(chipView(answer({ status }))), 'ring/ink', status);
    }
    assert.equal(shapeOf(chipView(answer({ status: 'wake_paused' }))), 'pause/ink', 'a wake that returned while paused');
    // A wake held by any spending limit its records prove is the owner's move; the chip keeps no time.
    const held = { status: 'wake_paused', live_wake_task_id: 'w1', live_wake_held: true,
        detail: 'Its own wake-up stopped at its allowance ($2.99).' };
    const allowance = chipView(answer({ ...held, live_wake_hold: { cause: 'allowance', ceiling_usd: 2.99 } }));
    assert.deepEqual([allowance.word, shapeOf(allowance)], ['', 'dot/owner']);
    assert.equal(allowance.said, `Consciousness: ${held.detail} · waits for you`);
    // The total budget that stopped it is amber too, never the two bars of a Pause someone chose (owner 27.11).
    for (const cause of ['allowance_and_cap', 'per_task_cap', 'launch_ceiling', 'total_budget', 'per_task_fence']) {
        const view = chipView(answer({ ...held, live_wake_hold: { cause, ceiling_usd: null } }));
        assert.deepEqual([shapeOf(view), view.said.endsWith(' · waits for you')], ['dot/owner', true], cause);
    }
    // A Pause is two bars; a sleep or a wake queued for a worker goes on by itself (still); an unproven hold a ring.
    for (const [cause, shape] of [['owner', 'pause/ink'], ['sleep', 'still/ink'], ['queued', 'still/ink'],
        ['technical', 'ring/ink'], ['', 'ring/ink']]) {
        const view = chipView(answer({ ...held, live_wake_hold: { cause, ceiling_usd: null } }));
        assert.deepEqual([shapeOf(view), view.said.includes('waits for you')], [shape, false], cause);
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
    // A whole-dollar limit reads in whole dollars, the spend rounded up (storage_format.js `formatLimitUsd`).
    assert.equal(allowanceText(snapshot({ spent_24h_usd: 1.5, daily_usd: 20 })), '$2 of $20 in the last 24 hours');
    // What the number leaves out is said after it in words, where a floor sign stood before it.
    assert.equal(allowanceText(snapshot({ spent_24h_usd: 1.5, daily_usd: 20, unknown_unmetered: 2 })),
        '$2 of $20 in the last 24 hours; 2 calls without a known price are not counted');
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
        let releaseRead = () => {};
        const status = createConsciousnessStatus({
            bar, ws: { send: (frame) => { sent.push(frame); return { status: 'sent' }; }, isConnected: () => true },
            openDashboardTab: (tab) => tabs.push(tab), refreshState: () => { refreshed += 1; },
            read: async () => {
                reads += 1;
                return normalizeConsciousness({ recent_wakes: [
                    { started_at: at(13, 52), summary: 'Reminded about the Swarm question', phase: 'done', ended_at: at(13, 53) },
                ] });
            },
            // The one switch with an install that has no place to name: no question, the same /bg (owner 15.11, 5A).
            // Its read is held open once, to see the button say so while it reads.
            turn: (to, options) => turnConsciousness(to, { ...options, read: () => new Promise((resolve) => {
                releaseRead = () => resolve({ lines: [], places: [] });
            }) }),
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
        assert.equal(bar.hasAttribute('data-cx-shade'), true, 'the open Shade covers the room: its notices leave (chat.js)');
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(reads, 1, 'one read of the wakes per opening, no poller');
        const shade = status.shade;
        assert.equal(shade.querySelector('.cx-state').textContent, 'Sleeping until 14:20.');
        assert.equal(shade.querySelector('.cx-money').textContent, '$2 of $20 in the last 24 hours');
        assert.deepEqual(shade.querySelectorAll('.cx-wake').map((row) => row.children.map((cell) => cell.textContent)),
            [['13:52', 'Reminded about the Swarm question', 'Done']]);
        const toggle = shade.querySelector('.cx-toggle');
        assert.equal(toggle.textContent, 'Turn off');
        const doors = shade.querySelectorAll('.cx-door');
        assert.deepEqual(doors.map((door) => [door.textContent, door.hidden]),
            [['Dashboard ›', false], ['Costs ›', true], ['Evolution ›', true]], 'Costs › only while the allowance is the next step');

        toggle.click();
        await new Promise((resolve) => setTimeout(resolve, 0));  // the switch reads what it would change first
        assert.deepEqual([toggle.textContent, toggle.disabled, sent.length], ['Reading…', true, 0],
            'a read that takes seconds is never a silent pause');
        releaseRead();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(sent, [{ type: 'command', cmd: '/bg stop', ui_control: { name: 'consciousness_toggle' } }],
            'Turn off is the Dashboard\'s own command, not typed owner text');
        assert.equal(toggle.disabled, true, 'it waits for the readback');
        await new Promise((resolve) => setTimeout(resolve, 450));
        assert.equal(refreshed, 1, 'the room\'s state clock is asked again');
        status.update(answer({ status: 'disabled' }, { bg_consciousness_enabled: false, evolution_enabled: true }));
        assert.deepEqual([toggle.textContent, toggle.disabled], ['Turn on', false]);
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

test('the Costs door stands where its limit is the next step: under the Shade while new starts wait, in a held wake\'s block', async () => {
    const env = installDom();
    try {
        const bar = document.createElement('div');
        const tabs = [];
        const status = createConsciousnessStatus({ bar, ws: { send() {}, isConnected: () => true },
            openDashboardTab: (tab, options) => tabs.push([tab, options?.row || '']),
            read: async () => normalizeConsciousness({ recent_wakes: [] }) });
        status.update(answer({ status: 'allowance_exhausted', detail: 'Daily allowance spent.' }));
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const costs = status.shade.querySelectorAll('.cx-door').find((door) => door.textContent === 'Costs ›');
        const block = status.shade.querySelector('.cx-held');
        assert.deepEqual([costs.hidden, block.hidden], [false, true], 'new starts wait for the allowance: the Shade\'s own door');
        const held = { status: 'wake_paused', live_wake_task_id: 'w1', live_wake_held: true, position: 'Also on its own and in places',
            detail: 'Its own wake-up stopped at its allowance ($2.99).', live_wake_hold: { cause: 'allowance', ceiling_usd: 2.99 } };
        const blockCosts = () => status.shade.querySelector('.cx-held-actions').children
            .find((node) => node.classList.contains('cx-held-costs')) || null;
        // A wake held by a limit Costs edits: the door is its block's, the Shade's own gives way; a limit raised elsewhere
        // (the work's own cap, a fence of its own work) or a Pause has none.
        for (const [cause, shown] of [['allowance', true], ['allowance_and_cap', true], ['launch_ceiling', true],
            ['total_budget', true], ['per_task_cap', false], ['per_task_fence', false], ['owner', false], ['sleep', false]]) {
            status.update(answer({ ...held, live_wake_hold: { cause, ceiling_usd: null } }));
            assert.deepEqual([costs.hidden, block.hidden, Boolean(blockCosts())], [true, false, shown], cause);
        }
        status.update(answer({ status: 'allowance_exhausted', detail: 'Daily allowance spent.' }, { bg_consciousness_enabled: false }));
        assert.deepEqual([costs.hidden, block.hidden], [true, true], 'Off is no allowance step and holds no block');
        status.update(answer(held));
        blockCosts().click();
        assert.deepEqual(tabs, [['costs', 'OUROBOROS_CONSCIOUSNESS_DAILY_USD']], 'the allowance\'s own row');
        assert.equal(status.isOpen(), false, 'the door closes the Shade');
        status.update(answer({ status: 'allowance_exhausted', detail: 'Daily allowance spent.' }));
        status.chip.click();
        costs.parentElement.listeners.get('click')[0]({ target: { closest: () => costs } });
        assert.deepEqual(tabs.at(-1), ['costs', '']);
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

// The held own wake as its own block (owner 27.11, «Блок с кнопками»): what the switch's header says, what holds the wake
// and what the owner can do about it are two objects.
const WAKE = 'e629aa8e';
const wakeWork = (controls) => ({ work_id: WAKE, room: 1, episode: { task_id: WAKE, live: true }, activity: 'paused',
    confirmed: true, mark: 'pause', controls });
const OFFERED = [{ action: 'resume', target: WAKE }, { action: 'stop', target: WAKE, scope: 'cascade' }];
const heldBy = (cause, controls = OFFERED, bg = {}) => answer({ status: 'wake_paused', live_wake_task_id: WAKE,
    live_wake_held: true, position: 'Also on its own and in places', detail: `Its own wake-up is held (${cause}).`,
    detail_note: 'What follows from it.', live_wake_hold: { cause, ceiling_usd: null, started_at: at(13, 20) }, ...bg },
{ works: controls === null ? [] : [wakeWork(controls)] });

test('a held own wake is its own block under the switch: its start, mark and word, the host\'s cause and next step', async () => {
    const env = installDom();
    try {
        const bar = document.createElement('div');
        const status = createConsciousnessStatus({ bar, ws: { send() {}, isConnected: () => true },
            read: async () => normalizeConsciousness({ recent_wakes: [] }) });
        status.update(heldBy('total_budget'));
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const shade = status.shade;
        const body = shade.querySelector('.cx-body');
        const block = shade.querySelector('.cx-held');
        assert.equal(body.children.indexOf(block), body.children.indexOf(shade.querySelector('.cx-head')) + 1,
            'directly under the switch\'s header');
        assert.equal(block.getAttribute('aria-labelledby'), block.querySelector('.cx-held-name').id);
        assert.equal(block.querySelector('.cx-held-name').textContent, 'Own time · 13:20', 'its start in the viewer\'s clock');
        const mark = block.querySelector('.cx-held-mark');
        assert.deepEqual([mark.dataset.shape, mark.dataset.hue, block.querySelector('.cx-held-word').textContent],
            ['dot', 'owner', 'Paused'], 'a spending limit is the owner\'s move');
        assert.deepEqual([block.querySelector('.cx-held-text').textContent, block.querySelector('.cx-held-note').textContent],
            ['Its own wake-up is held (total_budget).', 'What follows from it.']);
        // The header keeps the switch: its position, money and Turn off; the hold is said once, in the block.
        assert.equal(shade.querySelector('.cx-state').textContent, 'Also on its own and in places');
        assert.equal(shade.querySelector('.cx-state-note').hidden, true);
        assert.equal(shade.querySelector('.cx-toggle').textContent, 'Turn off');
        for (const [cause, shape, word] of [['owner', 'pause', 'Paused'], ['sleep', 'still', 'Waiting'],
            ['queued', 'still', 'Queued'], ['technical', 'ring', 'Paused'], ['', 'ring', 'State unknown']]) {
            status.update(heldBy(cause));
            assert.deepEqual([mark.dataset.shape, mark.dataset.hue, block.querySelector('.cx-held-word').textContent],
                [shape, 'ink', word], cause || 'unproven');
        }
        // A wake with no start recorded (only queued) is named alone; no held wake, no block, and the state is the
        // header's again.
        status.update(heldBy('queued', OFFERED, { live_wake_hold: { cause: 'queued', ceiling_usd: null } }));
        assert.equal(block.querySelector('.cx-held-name').textContent, 'Own time');
        status.update(answer({ position: 'Also on its own and in places' }));
        assert.deepEqual([block.hidden, shade.querySelector('.cx-state-note').textContent], [true, 'Sleeping until 14:20.']);
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

test('the held wake\'s block offers exactly its work\'s controls on the shared task controls, Costs leading for the total budget', async () => {
    const posts = [];
    const env = installDom(async (url, init = {}) => {
        if (init.method === 'POST') posts.push(String(url));
        return { ok: true, status: 200, json: async () => ({ ok: true }) };
    });
    try {
        const bar = document.createElement('div');
        let online = true;
        let refreshed = 0;
        const status = createConsciousnessStatus({ bar, ws: { send() {}, isConnected: () => online },
            refreshState: () => { refreshed += 1; }, read: async () => normalizeConsciousness({ recent_wakes: [] }) });
        const actions = () => status.shade.querySelector('.cx-held-actions').children.map((node) => node.classList.contains('cx-held-door')
            ? `door:${node.querySelector('[data-open-task]')?.dataset.openTask}` : node.textContent);
        const button = (label) => status.shade.querySelector('.cx-held-actions').children.find((node) => node.textContent === label);
        status.update(heldBy('total_budget'));
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        // The total budget holds it: raising that budget leads, and Resume stays pressable, neutral (the server refuses
        // honestly while nothing is left).
        assert.deepEqual(actions(), ['Costs ›', 'Resume', 'Stop now', `door:${WAKE}`]);
        assert.deepEqual([button('Costs ›').classList.contains('btn-primary'), button('Resume').classList.contains('btn-primary'),
            button('Resume').classList.contains('btn-default'), button('Resume').disabled], [true, false, true, false]);
        assert.equal(button('Stop now').dataset.taskControlDirect, '1', 'its one action after the shared confirmation');
        // Held at its allowance: Resume is the one primary, Costs follows the controls.
        status.update(heldBy('allowance'));
        assert.deepEqual(actions(), ['Resume', 'Stop now', 'Costs ›', `door:${WAKE}`]);
        assert.deepEqual([button('Resume').classList.contains('btn-primary'), button('Costs ›').classList.contains('btn-primary')],
            [true, false]);
        // Only what the server offers: a Pause someone chose with a Stop alone, then nothing but the door, then no work at all.
        status.update(heldBy('owner', [OFFERED[1]]));
        assert.deepEqual(actions(), ['Stop now', `door:${WAKE}`]);
        status.update(heldBy('owner', []));
        assert.deepEqual(actions(), [`door:${WAKE}`]);
        status.update(heldBy('owner', null));
        assert.deepEqual(actions(), [`door:${WAKE}`], 'a work the answer does not project offers no control');
        // A passed deadline offers its exact Close, never Resume.
        const token = 'a'.repeat(64);
        status.update(heldBy('owner', [{ action: 'close', target: WAKE, close_expired: token }, OFFERED[1]]));
        assert.deepEqual(actions(), ['Close', 'Stop now', `door:${WAKE}`]);
        // A closed socket leaves every control in place, disabled; the door still leads to the card.
        online = false;
        status.update(heldBy('allowance', [...OFFERED, { action: 'close', target: WAKE, close_expired: token }]));
        assert.deepEqual(['Resume', 'Close', 'Stop now'].map((label) => button(label).disabled), [true, true, true]);
        online = true;
        status.update(heldBy('allowance'));
        assert.equal(button('Resume').disabled, false);
        // The press is the shared Resume: the work's own task, once, then the room's state is read again.
        await button('Resume').listeners.get('click')[0]({ target: button('Resume'), stopPropagation() {} });
        assert.deepEqual(posts, [`/api/tasks/${WAKE}/resume`]);
        assert.ok(refreshed >= 1, 'the settling read');
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

test('own wakes are doors to their cards in Main, like the wakes in places to their sheets; the block\'s door too', async () => {
    const env = installDom();
    const events = [];
    globalThis.window.dispatchEvent = (event) => events.push([event.type, event.detail]);
    try {
        const bar = document.createElement('div');
        const status = createConsciousnessStatus({ bar, ws: { send() {}, isConnected: () => true },
            read: async () => normalizeConsciousness({ recent_wakes: [
                { task_id: 'own-1', chat_id: 1, summary: 'Read the notes', started_at: at(13, 5), phase: 'done', ended_at: at(13, 6) },
                { task_id: 'own-2', summary: 'An older row without its room', started_at: at(12, 5), phase: 'done', ended_at: at(12, 6) },
            ] }) });
        status.update(heldBy('owner'));
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const own = status.shade.querySelector('[data-wakes="own"]');
        const doors = own.querySelectorAll('.cx-wake-door');
        assert.equal(doors.length, 1, 'a row whose room is unknown stays words');
        assert.equal(doors[0].getAttribute('aria-label'), 'Open in Main: 13:05 · Read the notes · Done');
        doors[0].click();
        assert.deepEqual(events, [['ouro:open-task', { task_id: 'own-1', chat_id: 1 }]]);
        assert.equal(status.isOpen(), false, 'the Shade gives way to Main');
        status.chip.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const door = status.shade.querySelector('.cx-held-door');
        const link = door.querySelector('[data-open-task]');
        door.listeners.get('click')[0]({ target: { closest: () => link }, stopPropagation() {} });
        assert.deepEqual(events.at(-1), ['ouro:open-task', { task_id: WAKE, chat_id: 1 }]);
        assert.equal(status.isOpen(), false);
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

test('keyboard opening skips a disabled or hidden toggle and lands on Dashboard', () => {
    const env = installDom();
    let status = null;
    try {
        const doc = globalThis.document;
        const bar = doc.createElement('div');
        env.mount.appendChild(bar);
        let online = true;
        status = createConsciousnessStatus({
            bar,
            ws: { send() {}, isConnected: () => online },
            read: async () => normalizeConsciousness({ recent_wakes: [] }),
        });
        status.update(answer());
        assert.equal(status.chip.isConnected, true);

        // Offline: Turn off is disabled on open via paintShade(), so ArrowDown lands on Dashboard.
        online = false;
        status.update(answer());
        for (const fn of status.chip.listeners.get('keydown') || []) fn({ target: status.chip, key: 'ArrowDown', preventDefault() {} });
        assert.equal(status.isOpen(), true);
        assert.equal(status.shade.querySelector('.cx-toggle').disabled, true);
        assert.equal(doc.activeElement, status.shade.querySelector('.cx-door'), 'offline keyboard open skips disabled toggle');
        status.close();

        // Unknown enabled state: toggle is hidden, so Enter/Space (detail === 0) lands on Dashboard.
        online = true;
        status.update({ bg_consciousness_enabled: null, bg_consciousness_state: {} });
        for (const fn of status.chip.listeners.get('click') || []) fn({ target: status.chip, detail: 0 });
        assert.equal(status.isOpen(), true);
        assert.equal(status.shade.querySelector('.cx-toggle').hidden, true);
        assert.equal(doc.activeElement, status.shade.querySelector('.cx-door'), 'unknown-state keyboard open skips hidden toggle');

        // Already-on consciousness with remaining Panic-stopped tasks exposes Allow Panic resume beside Turn off.
        status.close();
        status.update(answer({ panic_choice_available: true }));
        status.chip.click();
        const panicBtn = status.shade.querySelector('.cx-panic-choice');
        assert.equal(panicBtn.hidden, false);
        assert.equal(panicBtn.textContent, 'Allow Panic resume');
        status.update(answer());
        assert.equal(panicBtn.hidden, true);
    } finally {
        status?.destroy();
        restoreDom(env.prior);
    }
});

// ---- fix round 1: authority, expiry, the client's lines, presses and focus

const openHeld = async (bg, { ws, fetchImpl, refreshState, read, confirmStop } = {}) => {
    const env = installDom(fetchImpl);
    const bar = document.createElement('div');
    env.mount.appendChild(bar);
    const status = createConsciousnessStatus({ bar, ws: ws || { send() {}, isConnected: () => true }, confirmStop,
        refreshState: refreshState || (() => {}), read: read || (async () => normalizeConsciousness({ recent_wakes: [] })) });
    status.update(bg);
    status.chip.click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    const block = status.shade.querySelector('.cx-held');
    const button = (label) => status.shade.querySelector('.cx-held-actions').children.find((node) => node.textContent === label);
    return { env, status, block, button };
};
// A press as a browser makes it: a disabled button dispatches no click; every listener of an enabled one runs in order.
const press = async (node) => {
    if (node.disabled) return;
    const evt = { target: node, stopPropagation() {}, preventDefault() {} };
    await Promise.all((node.listeners.get('click') || []).map((fn) => fn(evt)));
};

test('a failed or works-less state read leaves the last controls in place, disabled, and a press sends nothing', async () => {
    const posts = [];
    const { env, status, button } = await openHeld(heldBy('allowance'), { fetchImpl: async (url, init = {}) => {
        if (init.method === 'POST') posts.push(String(url));
        return { ok: true, status: 200, json: async () => ({ ok: true }) };
    } });
    try {
        assert.deepEqual([button('Resume').disabled, button('Stop now').disabled], [false, false]);
        // The room's failed read reaches the chip as a body without facts or works (chat.js): authority is withdrawn.
        status.update({ accounting: { available: false } });
        assert.deepEqual([Boolean(button('Resume')), button('Resume').disabled, button('Stop now').disabled], [true, true, true]);
        // Even a press the paint did not reach re-checks: the shared door asks the block's authority at the press.
        await button('Resume').listeners.get('click')[0]({ target: button('Resume'), stopPropagation() {} });
        assert.deepEqual(posts, [], 'no Resume on facts the newest read did not confirm');
        // A body with works alone renews them; one that no longer offers Resume removes it.
        status.update({ works: [wakeWork([OFFERED[1]])] });
        assert.deepEqual([Boolean(button('Resume')), button('Stop now').disabled], [false, false]);
        status.update(heldBy('allowance'));
        assert.equal(button('Resume').disabled, false);
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

test('a Stop now confirmed after the state read failed sends no stop; the confirmation keeps the Shade open', async () => {
    const posts = [];
    let answer = () => {};
    const asked = [];
    const { env, status, button } = await openHeld(heldBy('owner'), {
        confirmStop: (dialog) => { asked.push(dialog.title); return new Promise((resolve) => { answer = resolve; }); },
        fetchImpl: async (url, init = {}) => {
            if (init.method === 'POST') posts.push(String(url));
            return { ok: true, status: 200, json: async () => ({ ok: true }) };
        } });
    try {
        await press(button('Stop now'));
        assert.deepEqual(asked, ['Stop this work now?'], 'the shared confirmation asks first');
        // Focus moving into the modal and a press inside it are part of the Shade meanwhile (Cancel returns to Stop now).
        const inModal = { closest: (selector) => (selector.includes('aria-modal') ? {} : null) };
        for (const fn of status.shade.listeners.get('focusout') || []) fn({ relatedTarget: inModal });
        assert.equal(status.isOpen(), true, 'focus in its own confirmation does not close the Shade');
        status.update({ accounting: { available: false } });
        answer(true);
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(posts, [], 'the press re-checks authority after the confirmation');
        // With the read standing again, a confirmed Stop now reaches the work's own stop.
        status.update(heldBy('owner'));
        await press(button('Stop now'));
        answer(true);
        for (let turn = 0; turn < 4; turn += 1) await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(posts, [`/api/tasks/${WAKE}/cancel`]);
        // Elsewhere, focus leaving still closes it.
        for (const fn of status.shade.listeners.get('focusout') || []) fn({ relatedTarget: { closest: () => null } });
        assert.equal(status.isOpen(), false);
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

test('a passed deadline says the shared expired word and offers Close, without the continuation advice or Costs', async () => {
    const token = 'b'.repeat(64);
    const budget = { accounting: { available: true, limit_usd: 62700, remaining_known_usd: 0 } };
    const current = { ...heldBy('total_budget'), ...budget };
    const { env, status, block, button } = await openHeld(current);
    try {
        const said = () => [block.querySelector('.cx-held-word').textContent, block.querySelector('.cx-held-note').hidden,
            block.querySelector('.cx-held-advice').hidden, Boolean(button('Costs ›'))];
        assert.deepEqual(said(), ['Paused', false, false, true], 'a current pause keeps its advice and the Costs lead');
        status.update({ ...heldBy('total_budget', [{ action: 'close', target: WAKE, close_expired: token }, OFFERED[1]]), ...budget });
        assert.deepEqual(said(), ['Deadline expired', true, true, false]);
        assert.equal(block.querySelector('.cx-held-mark').dataset.hue, 'ink', 'nothing waits on the owner');
        assert.ok(button('Close') && !button('Resume'));
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

test('the block says Resume\'s estimated room at an allowance and whether the total budget has room now', async () => {
    const { env, status, block, button } = await openHeld(heldBy('allowance'));
    try {
        const advice = () => (block.querySelector('.cx-held-advice').hidden ? '' : block.querySelector('.cx-held-advice').textContent);
        for (const [hold, words] of [[{ resume_additional_usd: 7.6 }, 'Estimated room on Resume: $7.60'],
            [{ resume_additional_usd: 0 }, 'Estimated room on Resume: none'], [{}, '']]) {
            status.update(heldBy('allowance', OFFERED, { live_wake_hold: { cause: 'allowance', ceiling_usd: 2.99, ...hold } }));
            assert.equal(advice(), words, JSON.stringify(hold));
        }
        // The total budget keeps its pause after a raise: the same body's budget decides the step and the lead.
        const byBudget = (accounting) => ({ ...heldBy('total_budget'), ...(accounting ? { accounting } : {}) });
        status.update(byBudget({ available: true, limit_usd: 62700, remaining_known_usd: 0 }));
        assert.equal(advice(), 'The total budget has nothing left now: raise it in Costs, then Resume.');
        assert.deepEqual([button('Costs ›').classList.contains('btn-primary'), button('Resume').classList.contains('btn-primary')], [true, false]);
        status.update(byBudget({ available: true, limit_usd: 62700, remaining_known_usd: 2379.5 }));
        assert.equal(advice(), 'The total budget has $2,379 left now: Resume continues it.');
        assert.deepEqual([button('Resume').classList.contains('btn-primary'), button('Costs ›').classList.contains('btn-primary')], [true, false]);
        assert.equal(status.shade.querySelector('.cx-held-actions').children[0], button('Resume'), 'Resume leads');
        status.update(byBudget(null));
        assert.equal(advice(), '', 'a body that does not say the budget claims neither');
        assert.equal(button('Costs ›').classList.contains('btn-primary'), true, 'the decided default: Costs leads');
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

test('a press of one control disables its siblings until it settles; a press elsewhere is re-checked, not seen', async () => {
    let release = () => {};
    const { env, status, button } = await openHeld(heldBy('allowance'), { fetchImpl: async (url, init = {}) => {
        if (init.method === 'POST') await new Promise((resolve) => { release = resolve; });
        return { ok: true, status: 200, json: async () => ({ ok: true }) };
    } });
    try {
        const pressed = press(button('Resume'));
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual([button('Resume').disabled, button('Stop now').disabled], [true, true], 'while the Resume is out');
        release();
        await pressed;
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(button('Stop now').disabled, false, 'its settling read let the siblings go');
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

test('a keyboard open of the Shade lands on the held wake\'s first enabled control, before Turn off', async () => {
    const env = installDom();
    try {
        const bar = document.createElement('div');
        env.mount.appendChild(bar);
        const status = createConsciousnessStatus({ bar, ws: { send() {}, isConnected: () => true },
            read: async () => normalizeConsciousness({ recent_wakes: [] }) });
        status.update(heldBy('allowance'));
        for (const fn of status.chip.listeners.get('keydown') || []) fn({ target: status.chip, key: 'ArrowDown', preventDefault() {} });
        assert.equal(document.activeElement?.textContent, 'Resume');
        status.close();
        status.update(answer());
        for (const fn of status.chip.listeners.get('keydown') || []) fn({ target: status.chip, key: 'ArrowDown', preventDefault() {} });
        assert.equal(document.activeElement, status.shade.querySelector('.cx-toggle'), 'no block: the switch, as before');
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});

test('an own wake written in an owner chat no Project holds is a door too: Main shows its card', () => {
    const rows = (chat, projects) => wakeRows(normalizeConsciousness({ recent_wakes: [
        { task_id: 'own-7', chat_id: chat, summary: 'Read notes', started_at: at(13, 5), phase: 'done', ended_at: at(13, 6) },
    ] }).recentWakes, projects)[0];
    assert.equal(rows(1, null).taskId, 'own-7');
    assert.equal(rows(7, [3, 4]).taskId, 'own-7', 'a bound owner chat, not a Project room');
    assert.equal(rows(7, [7]).taskId, undefined, 'a Project room keeps its own card');
    assert.equal(rows(7, null).taskId, undefined, 'unknown Projects: no door claimed');
});

// ---- fix round 2

test('a room header\'s Shade follows the page\'s state clock: a failed read or a retire withdraws its controls', async () => {
    const posts = [];
    const env = installDom(async (url, init = {}) => {
        if (init.method === 'POST') posts.push(String(url));
        return { ok: true, status: 200, json: async () => ({ ok: true, recent_wakes: [] }) };
    });
    try {
        const bar = document.createElement('div');
        env.mount.appendChild(bar);
        // A Project panel's bar as app.js wires it: applied bodies through the sequencer's callback, the rest by following.
        let header = null;
        const snapshots = createStateSnapshotSequencer((data) => header?.update(data));
        header = createRoomHeader({ bar, ws: { send() {}, isConnected: () => true } });
        header.followState(snapshots);
        const apply = async (body) => snapshots.apply(await snapshots.gate(), body);
        const button = (label) => bar.querySelector('.cx-held-actions')?.children.find((node) => node.textContent === label);
        await apply(heldBy('allowance'));
        bar.querySelector('.cx-chip').click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual([button('Resume').disabled, button('Stop now').disabled], [false, false]);
        // A failed read: the last controls stay, disabled, and a press sends nothing.
        snapshots.fail(await snapshots.gate());
        assert.deepEqual([Boolean(button('Resume')), button('Resume').disabled, button('Stop now').disabled], [true, true, true]);
        await button('Resume').listeners.get('click')[0]({ target: button('Resume'), stopPropagation() {} });
        assert.deepEqual(posts, [], 'no Resume on a read that failed');
        await apply(heldBy('allowance'));
        assert.equal(button('Resume').disabled, false, 'the next applied body stands again');
        // A retired body (a socket change, a confirmed change elsewhere) withdraws them the same way.
        snapshots.retire();
        assert.deepEqual([button('Resume').disabled, button('Stop now').disabled], [true, true]);
        // A replacement that no longer offers Resume removes it.
        await apply(heldBy('allowance', [OFFERED[1]]));
        assert.deepEqual([Boolean(button('Resume')), button('Stop now').disabled], [false, false]);
        header.destroy();
        snapshots.fail(await snapshots.gate());  // a destroyed header follows nothing
    } finally {
        restoreDom(env.prior);
    }
});

test('a passed deadline keeps no detail that promises the wake goes on; a live one keeps the host\'s words', async () => {
    const token = 'c'.repeat(64);
    const SLEEPS = 'Its own wake-up sleeps until what it waits for happens, then goes on by itself; its card shows what.';
    const sleeping = (controls) => heldBy('sleep', controls, { detail: SLEEPS });
    const { env, status, block, button } = await openHeld(sleeping(OFFERED));
    try {
        const text = block.querySelector('.cx-held-text');
        assert.deepEqual([text.hidden, text.textContent, block.querySelector('.cx-held-word').textContent], [false, SLEEPS, 'Waiting']);
        status.update(sleeping([{ action: 'close', target: WAKE, close_expired: token }, OFFERED[1]]));
        assert.deepEqual([text.hidden, block.querySelector('.cx-held-word').textContent, Boolean(button('Close'))],
            [true, 'Deadline expired', true], 'beside Close, no "goes on by itself"');
        status.update(sleeping(OFFERED));
        assert.deepEqual([text.hidden, text.textContent], [false, SLEEPS]);
        status.destroy();
    } finally {
        restoreDom(env.prior);
    }
});
